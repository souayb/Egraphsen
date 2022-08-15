from ast import Try
from distutils.command.config import config
from tkinter import W
from turtle import done, shape
from itertools import count
from PyQt5.QtWidgets import QMessageBox
from .grab_pop import  LabelGrab
import black
from matplotlib.pyplot import box ##################### 
# from detection.detect import detect_bbox #################################################
from PIL  import Image, ImageStat
import cv2
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
import numpy as np 
#from labelme import QT5
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from shape import Shape
import utils
from logger import SysVars
PY2 = SysVars.PY2
QT5 = SysVars.QT5
__appname__ = SysVars.APPNAME
__version__ = SysVars.VERSION

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor


class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    edgeSelected = QtCore.Signal(bool, object)
    vertexSelected = QtCore.Signal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line,  point or grab
    _createMode = 'polygon'

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop('epsilon', 10.0)
        self.double_click = kwargs.pop('double_click', 'close')
        self.full_args = kwargs.pop('full_args', 10.0)

        print("WHAT ARE THESE")
        print(self.full_args)
        if self.double_click not in [None, 'close']:
            raise ValueError(
                'Unexpected value for double_click event: {}'
                .format(self.double_click)
            )
        self._config = kwargs.pop('config_data')
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        #   - createMode == 'grab': foue points
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.yolo = None
        self.current_gid = None

        # Update
        self.MIN_POINT_DISTANCE = 0.01 ### bei dist<(max_dist*self.MIN_POINT_DISTANCE)**1.5)
        self.MAX_POINT_DISTANCE = 0.03#80. # Euklidische distanz in pixeln

        ###################### adding the 
          # Main widgets and related state.
        self.labepop = LabelGrab(
            parent=self,
            labels=self._config['labels'],
            context=self._config['context'],
            state=self._config['state'],
            person=self._config['person'],
            orient=self._config['orient'],
            phrase=self._config['phrase'],
            sort_labels=self._config['sort_labels'],
            show_text_field= self._config['show_label_text_field'],
            completion=self._config['label_completion'],
            fit_to_content=self._config['fit_to_content'],
            flags=self._config['label_flags']
        )

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in ['polygon', 'rectangle', 'circle',
           'line', 'point', 'linestrip', 'grab']:
            raise ValueError('Unsupported createMode: %s' % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) >= 10:
            self.shapesBackups = self.shapesBackups[-9:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.repaint()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None
    def QPixmapToArray(self, pixmap):
        ## Get the size of the current pixmap
        size = pixmap.size()
        h = size.width()
        w = size.height()
        channels_count = 4
        qimg = pixmap.toImage()
        byte_str = qimg.bits().asstring(w*h*channels_count)

        ## Using the np.frombuffer function to convert the byte string into an np array
        img = np.fromstring(byte_str, dtype=np.uint8).reshape((w,h,channels_count))

        return img[:,:,:3]
    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.prevMovePoint = pos
        self.restoreCursor()

        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif len(self.current) > 1 and self.createMode == 'polygon' and\
                    self.closeEnough(pos, self.current[0]):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)

            elif len(self.current) > 1 and self.createMode == 'grab' and\
                    self.closeEnough(pos, self.current[0]):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)

            if self.createMode in ['polygon', 'linestrip', 'grab']: ### added grab here 
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode == 'rectangle':
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == 'circle':
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == 'line':
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == 'point':
                self.line.points = [self.current[0]]
                self.line.close()

            # elif self.createMode == 'grab':
            #     self.line.points = [self.current[0], pos]
            #     self.line.close()

            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = \
                    [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = index
                self.hShape = shape
                self.hEdge = index_edge
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = None
                self.hShape = shape
                self.hEdge = index_edge
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label)
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.edgeSelected.emit(self.hEdge is not None, self.hShape)
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        point = self.prevMovePoint
        if shape is None or point is None:
            return
        index = shape.nearestVertex(point, self.epsilon)
        shape.removePoint(index)
        # shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = None
        self.hEdge = None
        self.movingShape = True  # Save changes

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    
    def is_grayscale(self, arr):
        im = Image.fromarray(arr)
        stat = ImageStat.Stat(im)

        if sum(stat.sum)/3 == stat.sum[0]:
            return True
        else:
            return False
    def get_wbg_image(self, img:np.ndarray, coord, rect:bool=True):
        """""
        https://www.life2coding.com/cropping-polygon-or-non-rectangular-region-from-image-using-opencv-python/
        """
        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        if rect:
            cv2.rectangle(mask, (coord[2],coord[3]), (coord[0], coord[1]), thickness= -1, color= (255, 255, 255))
        else:
            cv2.drawContours(mask, [coord], -1, (255, 255, 255), -1, cv2.LINE_AA)
        res = cv2.bitwise_and(img,img,mask = mask)
        wbg = np.ones_like(img, np.uint8)*255
        cv2.bitwise_not(wbg,wbg, mask=mask) 
        dst = wbg+res
        return dst, res
    def grab_cat_mask(self, image,
                    rect:list=None,
                    use_sobel:bool=True,
                    apply_blur:bool=True,
                    grab_iteration:int=10,
                    kernel_size:int=5):
        """
        Input:
            image: type= numpy array
            rect:(default=None) the focus box for grabcut if None the entire image is used
            plot_contour: This allow to plot the contours on the input image
            apply_blur:(default=True) blur the image to remove small rectangle
        Ouptut:
            contour:(type:list) of all the contours
            painted:(type:image in nd.array) results with the countour plot on the image
        """
        assert isinstance(image, np.ndarray), F"image must be a numpy array, but received: {type(image)}"

        image_copy = image.copy()
        if self.is_grayscale(image_copy):

            if rect:
                thresh = np.ones(image_copy.shape[:2],np.uint8)
                x, y, w, h = rect

                crop = image_copy[y:y+h, x:x+w]
                # cv2.imshow('in the grabbbb', crop)

                try:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                except Exception as e:
                    raise ValueError("PLease very that the rectangle coordinate is correct")

                if apply_blur:
                        # crop =  cv2.GaussianBlur(crop, (3, 3), 0)
                        crop = cv2.blur(crop, ksize = (10, 10))
                # ret, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV)
                threshold = cv2.bilateralFilter(crop, 11, 17, 17) 
                # ret, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

                thresh[y:y+h, x:x+w] = threshold
                if apply_blur:
                    img_gray =  cv2.GaussianBlur(thresh, (3, 3), 0)
            else:
                img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
                if apply_blur:
                        img_gray =  cv2.GaussianBlur(img_gray, (3, 3), 0)
                ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
        
        else:
            height, width  = image_copy.shape[:2]
            mask = np.zeros((height , width),np.uint8)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            left_margin_proportion = 0.01
            right_margin_proportion = 0.01
            up_margin_proportion = 0.01
            down_margin_proportion = 0.01

            if not rect:
                rect = (
                    int(width * left_margin_proportion),
                    int(height * up_margin_proportion),
                    int(width * (1 - right_margin_proportion)),
                    int(height * (1 - down_margin_proportion)),
                )
            print("rerer", rect)
            x, y, w, h = rect
            
            crop = image_copy[y:y+h, x:w+x]
            # cv2.imshow('in the grabbbb', crop)
            try:
                cv2.grabCut(image_copy,mask,rect,bgdModel,fgdModel,grab_iteration,cv2.GC_INIT_WITH_RECT)
            except Exception as e:
                print("Oops!", e.__class__, "Please verify the coordinate of the rectangle")


            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            image_copy = image_copy*mask2[:,:,np.newaxis]

            # cv2.imshow('old grab', image_copy)
            # cv2.waitKey(0)
            img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
            # img_gray = cv2.blur(img_gray, ksize = (10, 10))
            
            ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_TRIANGLE)
            
            kernel = np.ones((5,5),np.uint8)
            # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(kernel_size,kernel_size))
            morphed_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel=kernel, iterations=3) # remove external noises 
            morphed_open_closed  = cv2.morphologyEx(morphed_open, cv2.MORPH_CLOSE, kernel=kernel, iterations=3) # remove interior noises

        # binary = cv2.bitwise_not(morphed_open_closed)
        if use_sobel:
            H = cv2.Sobel(morphed_open_closed, cv2.CV_8U, 0, 2)
            V = cv2.Sobel(morphed_open_closed, cv2.CV_8U, 2, 0)
            edged = H+V
        else : 
            edged = cv2.Canny(morphed_open_closed, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours, hierarchy
    def _show_dialog(self):
        text, flags, group_id, context, state, person, orient, phrase, levels = self.labepop.popUp()
        if text != None:
            return text, flags, group_id, context, state, person, orient, phrase, levels

        else: 
            self._show_dialog()

    """"Return a list of nb_points equally spaced points
        between p1 and p2"""
    def intermediates(self,p1, p2, nb_points=8):     
        # If we have 8 intermediate points, we have 8+1=9 spaces
        # between p1 and p2
        x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
        y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

        return [[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
                for i in range(1, nb_points+1)]


    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == 'polygon':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ['rectangle', 'circle', 'line']:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        print("result rextangle ", self.current.points)
                        self.finalise()

                    elif self.createMode == 'grab':
                        stop_polygon = False
                        print("Current poinnt 0",self.line[0])
                        print("Current poinnt 1",self.line[1])

                        if len(self.current.points)>=2 and self.line[1] == self.current.points[0]:
                            stop_polygon = True
                        else:
                            self.current.addPoint(self.line[1])
                            self.line[0] = self.current[-1]

                        if stop_polygon:#len(self.current.points) == 5:
                            #print("Done:")
                            #for pp in self.current.points:
                            #    print(pp)
                            self.current.isClosed()
                            pts = [[int(sp.x()), int(sp.y())] for sp in self.current.points]
                            # text, flags, group_id, context, state, person, orient, phrase, levels = self.labepop.popUp()
                            text, flags, group_id, context, state, person, orient, phrase, levels = self._show_dialog()


                            rect = cv2.boundingRect(np.array(pts))

                            result = self.QPixmapToArray(self.pixmap)
                            x,y,w,h= rect

                            if self.labepop.ch_grab.isChecked(): # Apply grabe cut 
                                
                                contour, hier= self.grab_cat_mask(result, rect=rect, apply_blur=True, use_sobel=False, kernel_size=1)
                                area_thresh = 0
                                second_thres = 0 
                                big_contour = contour
                                print("number of contour", len(contour))
                                self.current = Shape(shape_type='grab')
                                if contour:
                                    areas = [cv2.contourArea(c) for c in contour ]
                                    big_contour_num = np.argmax(np.array(areas))
                                    big_contour = contour[big_contour_num]
                                    areas[big_contour_num] = 0 

                                    found_points = (big_contour.reshape(-1,2))

                                    max_dist = np.sqrt(np.sum((np.max(found_points,0)-np.min(found_points,0))**2))
                                    print("This is the max distance:",max_dist)

                                    # Reduce number of points in polygon
                                    new_points = [] 
                                    cur_p = 0
                                    cur_points = [found_points[0]]

                                    for pidx in range(1,len(found_points)):
                                        if np.sqrt(np.sum((np.mean(cur_points,0).astype(int)-found_points[pidx])**2,-1))>(max_dist*self.MIN_POINT_DISTANCE)**1.5:#:0.01
                                            new_points.append(np.mean(cur_points,0))
                                            cur_points = [found_points[pidx]]
                                            cur_p = pidx
                                            
                                        else:
                                            cur_points.append(found_points[pidx])
                                    new_points = np.array(new_points).astype(int)

                                    qpoly =  [QtCore.QPointF(p[0], p[1]) for p in new_points]
                                    if qpoly[0] != qpoly[-1]: #flags, group_id, context, state, person, orient, phrase, levels
                                        qpoly.append(qpoly[0])

                                    
                                    self.current.points = qpoly
                                    self.current.label = text
                                    self.current.group_id = group_id
                                    self.current.context =   context 
                                    self.current.state   = state
                                    self.current.person  = person
                                    self.current.orient  = orient
                                    self.current.phrase  = phrase
                                    self.current.parent  = levels
                                    assert self.current
                                    self.current.close()
                                    self.shapes.append(self.current)
                                    self.storeShapes()
                                    # self.current = None
                                    self.setHiding(False)
                                    self.newShape.emit()
                                    self.update()
                                else: 
                                    print("contour not found")
                                    self.current.close()
                                    self.current = None
                                    # self.setHiding(False)
                                    # self.newShape.emit()
                                    self.update()
                                self.current = None # removig the rectangle after dawing the contour plot 

                            else :  # USE detection models 
                                crope = result[y:y+h, x:x+w]
                                print("Original Boxes")
                                print(x,y,x+w,y+h)
                                print("Original points")
                                print(np.array(pts))
                                det_poly,yolo,group_ids = detect_bbox(crope,
                                                        pol_cut = np.array(pts),
                                                        label = text,
                                                        context = context, 
                                                        state = state,
                                                        group_id=group_id,
                                                        group_ids = self.current_gid,
                                                        model_path=self.full_args.model_path,
                                                        anchors_path = self.full_args.anchors_path,
                                                        saved_yolo = self.yolo,
                                                        )
                                self.current_gid = group_ids

                                print("hiuiuiijijj")
                                for peepo in det_poly:
                                    print(peepo)
                                    print()
                                
                                #start_pol = [text, np.array(pts)]
                                #det_poly.append(start_pol)


                                if self.yolo is None:
                                    self.yolo = yolo
                                contour = det_poly
                            
                                area_thresh = 0
                                big_contour = contour
                                print("number of contour", len(contour))
                                print(det_poly)
                                if contour:
                                    poly_countert = count(0)
                                    for c in contour:

                                        # Add intermediate points if distance between 2 points is too large
                                        new_points = []
                                        for pidx in range(1,len(c[1])):

                                            cur_dist = np.sqrt(np.sum((c[1][pidx-1]-c[1][pidx])**2))

                                            new_points.append(c[1][pidx-1])

                                            if cur_dist > np.max(self.imageSize)*self.MAX_POINT_DISTANCE:
                                                new_p = int(cur_dist/(np.max(self.imageSize)*self.MAX_POINT_DISTANCE))

                                                inter_points = self.intermediates(list(c[1][pidx-1]), list(c[1][pidx]), nb_points=new_p)

                                                for in_p in inter_points:
                                                    new_points.append(in_p)

                                        new_points.append(c[1][-1])

                                        ct = next(poly_countert)
                                        self.current = Shape(shape_type='polygon')
                                        #area = cv2.contourArea(c)
                                        #if area > area_thresh:
                                        #    area_thresh = area
                                        #    big_contour = c
                                        if c[0] == text:
                                            qpoly =  [QtCore.QPointF(p[0], p[1]) for p in new_points]
                                        else:
                                            qpoly =  [QtCore.QPointF(p[0]+x, p[1]+y) for p in new_points]
                                        self.current.points = qpoly
                                        self.current.label  = c[0]
                                        self.current.group_id = c[-1]
                                        self.current.parent = c[-2]

                                        self.current.context = None
                                        self.current.state   = None
                                        self.current.person  = None
                                        self.current.orient  = None
                                        self.current.phrase  = None
                                        assert self.current
                                        self.current.close()
                                        self.shapes.append(self.current)
                                        self.storeShapes()
                                        # self.current = None
                                        self.setHiding(False)
                                        self.newShape.emit()
                                        self.update()

                                self.current = None
                                
                    elif self.createMode == 'linestrip':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == 'point':
                        self.finalise()
                    else:
                        if self.createMode == 'circle':
                            self.current.shape_type = 'circle'
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            else:
                group_mode = (int(ev.modifiers()) == QtCore.Qt.ControlModifier)
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = (int(ev.modifiers()) == QtCore.Qt.ControlModifier)
            self.selectShapePoint(pos, multiple_selection_mode=group_mode)
            self.prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) \
                    and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton and self.selectedShapes:
            self.overrideCursor(CURSOR_GRAB)

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (self.shapesBackups[-1][index].points !=
                    self.shapes[index].points):
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.repaint()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (self.double_click == 'close' and self.canCloseShape() and
                len(self.current) > 3):
            self.current.popPoint()
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.calculateOffsets(shape, point)
                    self.setHiding()
                    if multiple_selection_mode:
                        if shape not in self.selectedShapes:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape])
                    else:
                        self.selectionChanged.emit([shape])
                    return
        self.deSelectShape()

    def calculateOffsets(self, shape, point):
        rect = shape.boundingRect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width() - 1) - point.x()
        y2 = (rect.y() + rect.height() - 1) - point.y()
        self.offsets = QtCore.QPoint(x1, y1), QtCore.QPoint(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(min(0, self.pixmap.width() - o2.x()),
                                 min(0, self.pixmap.height() - o2.y()))
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def copySelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPoint(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and \
                    self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (self.fillDrawing() and self.createMode == 'polygon' and
                self.current is not None and len(self.current.points) >= 2):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.fill_color.setAlpha(64)
            drawing_shape.paint(p)
        
        elif (self.fillDrawing() and self.createMode == 'grab' and
               self.current is not None and len(self.current.points) >= 2):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.fill_color.setAlpha(64)
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPoint(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def finalise_grab(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width() - 1, 0),
                  (size.width() - 1, size.height() - 1),
                  (0, size.height() - 1)]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPoint(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPoint(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPoint((x3 + x4) / 2, (y3 + y4) / 2)
                d = utils.distance(m - QtCore.QPoint(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical)
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == QtCore.Qt.Key_Escape and self.current:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == QtCore.Qt.Key_Return and self.canCloseShape():
            self.finalise()

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def getLastLabel(self):
        return self.shapes[-1].points

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ['polygon', 'linestrip','grab']:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ['rectangle', 'line', 'circle']:
            print("undolast")
            self.current.points = self.current.points[0:1]
        elif self.createMode == 'point':
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.repaint()

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        self.repaint()

    def loadShapes(self, shapes, replace=True):
        for testest in shapes[0]:
            print("loadShape Conva 906", testest)
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.repaint()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()