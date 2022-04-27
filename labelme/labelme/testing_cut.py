from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import traceback
from PIL  import Image, ImageStat


def cut_grey(image,
                  rect:list=None,
                  plot_contour:bool=True,
                  apply_blur:bool=True):
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


    if rect:
        thresh = np.zeros(image.shape[:2],np.uint8)
        x,y, w, h = rect
        crop = image[y:y+h, x:x+w]

        try:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ValueError("PLease very that the rectangle coordinate is correct")

        if apply_blur:
                crop =  cv2.GaussianBlur(crop, (3, 3), 0)
        ret, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

        thresh[y:y+h, x:x+w] = threshold
    else:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if apply_blur:
                img_gray =  cv2.GaussianBlur(img_gray, (3, 3), 0)
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)


    edged = cv2.Canny(thresh, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if plot_contour:
        cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)

        return contours, image_copy
    return contours


def is_grayscale(arr):
    im = Image.fromarray(arr)
    stat = ImageStat.Stat(im)

    if sum(stat.sum)/3 == stat.sum[0]:
        return True
    else:
        return False

def grab_cat_mask(image,
                  rect:list=None,
                  plot_contour:bool=True,
                  apply_blur:bool=True):
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
    if is_grayscale(image):

        if rect:
            print(rect)
            thresh = np.zeros(image.shape[:2],np.uint8)
            x, y, w, h = rect

            crop = image[y:y+h, x:x+w]

            try:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                raise ValueError("PLease very that the rectangle coordinate is correct")

            if apply_blur:
                    crop =  cv2.GaussianBlur(crop, (3, 3), 0)
            ret, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

            thresh[y:y+h, x:x+w] = threshold
        else:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if apply_blur:
                    img_gray =  cv2.GaussianBlur(img_gray, (3, 3), 0)
            ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

    else:
        height, width  = image.shape[:2]
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




        try:
            cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            print("Oops!", e.__class__, "occurred.: PLease verify the rectangle coordinate")

        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        image = image*mask2[:,:,np.newaxis]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_TRIANGLE)


    edged = cv2.Canny(thresh, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if plot_contour:
        cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)

        return contours, image_copy
    return contours


class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self.pixmap_item)

        self.points = []

        self._polygon_item = QtWidgets.QGraphicsPolygonItem(self.pixmap_item)
        # self.polygon_item.setPen(QtGui.QPen(QtCore.Qt.black, 5, QtCore.Qt.SolidLine))
        # self.polygon_item.setBrush(QtGui.QBrush(QtCore.Qt.red, QtCore.Qt.VerPattern))

    @property
    def pixmap_item(self):
        return self._pixmap_item

    @property
    def polygon_item(self):
        return self._polygon_item

    def setPixmap(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)


    def resizeEvent(self, event):
        self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.pos())
        lp = self.pixmap_item.mapFromScene(sp)
        self.points.append([int(sp.x()), int(sp.y())])
        # print("SP", sp.x())
        # print("LP", lp.toPoint())
        poly = self.polygon_item.polygon()
        poly.append(lp)
        if poly.count() == 4:
            imge = cv2.imread("/Users/souaybGA_1/Downloads/Desktop/egraphsen-tool-local/Images/test_image.jpg")
            # imge = cv2.imread("/Users/souayboubagayoko/Desktop/Musium/reast.png")
            pts = np.array(self.points)
            # print(pts)

            ## (1) Crop the bounding rect
            rect = cv2.boundingRect(pts)
            print('rect', rect)
            x,y,w,h = rect
            croped = imge[y:y+h, x:x+w].copy()
            contour, painted = grab_cat_mask(imge, rect=rect, apply_blur=True)#grab_cat_mask(imge, rect=rect, apply_blur=True)
            # print(contour, type(contour), len(contour))
            # if len(contour) > 1:
            # #     return
            # resh = np.array(contour)

            # resh = resh.reshape(-1,2)
            # print(resh.shape)
            cv2.imwrite("croped.png", croped)
            cv2.imwrite("painted.png", painted)
            poly.clear()
            self.points.clear()

        dir(poly)
        self.polygon_item.setPolygon(poly)




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        view = GraphicsView()
        self.setCentralWidget(view)
        view.setPixmap(QtGui.QPixmap("/Users/souaybGA_1/Downloads/Desktop/egraphsen-tool-local/Images/test_image.jpg"))
        # view.setPixmap(QtGui.QPixmap("/Users/souayboubagayoko/Desktop/Musium/reast.png"))
        self.resize(640, 480)


if __name__ == "__main__":

    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
