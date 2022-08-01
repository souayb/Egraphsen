# from PyQt5 import QtCore, QtGui, QtWidgets
# import cv2
# import numpy as np
# import traceback
# from PIL  import Image, ImageStat


# def cut_grey(image,
#                   rect:list=None,
#                   plot_contour:bool=True,
#                   apply_blur:bool=True):
#     """
#     Input:
#         image: type= numpy array
#         rect:(default=None) the focus box for grabcut if None the entire image is used
#         plot_contour: This allow to plot the contours on the input image
#         apply_blur:(default=True) blur the image to remove small rectangle
#     Ouptut:
#         contour:(type:list) of all the contours
#         painted:(type:image in nd.array) results with the countour plot on the image
#     """
#     assert isinstance(image, np.ndarray), F"image must be a numpy array, but received: {type(image)}"

#     image_copy = image.copy()


#     if rect:
#         thresh = np.zeros(image.shape[:2],np.uint8)
#         x,y, w, h = rect
#         crop = image[y:y+h, x:x+w]

#         try:
#             crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#         except Exception as e:
#             raise ValueError("PLease very that the rectangle coordinate is correct")

#         if apply_blur:
#                 crop =  cv2.GaussianBlur(crop, (3, 3), 0)
#         ret, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

#         thresh[y:y+h, x:x+w] = threshold
#     else:
#         img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         if apply_blur:
#                 img_gray =  cv2.GaussianBlur(img_gray, (3, 3), 0)
#         ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)


#     edged = cv2.Canny(thresh, 30, 200)
#     contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if plot_contour:
#         cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)

#         return contours, image_copy
#     return contours


# def is_grayscale(arr):
#     im = Image.fromarray(arr)
#     stat = ImageStat.Stat(im)

#     if sum(stat.sum)/3 == stat.sum[0]:
#         return True
#     else:
#         return False

# def grab_cat_mask(image,
#                   rect:list=None,
#                   plot_contour:bool=True,
#                   apply_blur:bool=True):
#     """
#     Input:
#         image: type= numpy array
#         rect:(default=None) the focus box for grabcut if None the entire image is used
#         plot_contour: This allow to plot the contours on the input image
#         apply_blur:(default=True) blur the image to remove small rectangle
#     Ouptut:
#         contour:(type:list) of all the contours
#         painted:(type:image in nd.array) results with the countour plot on the image
#     """
#     assert isinstance(image, np.ndarray), F"image must be a numpy array, but received: {type(image)}"

#     image_copy = image.copy()
#     if is_grayscale(image):

#         if rect:
#             print(rect)
#             thresh = np.zeros(image.shape[:2],np.uint8)
#             x, y, w, h = rect

#             crop = image[y:y+h, x:x+w]

#             try:
#                 crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#             except Exception as e:
#                 raise ValueError("PLease very that the rectangle coordinate is correct")

#             if apply_blur:
#                     crop =  cv2.GaussianBlur(crop, (3, 3), 0)
#             ret, threshold = cv2.threshold(crop, 255, 255, cv2.THRESH_BINARY_INV)
#             #ret, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
            
#             thresh[y:y+h, x:x+w] = threshold
#         else:
#             img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             if apply_blur:
#                     img_gray =  cv2.GaussianBlur(img_gray, (3, 3), 0)
#             ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

#     else:
#         height, width  = image.shape[:2]
#         mask = np.zeros((height , width),np.uint8)
#         bgdModel = np.zeros((1,65),np.float64)
#         fgdModel = np.zeros((1,65),np.float64)

#         left_margin_proportion = 0.01
#         right_margin_proportion = 0.01
#         up_margin_proportion = 0.01
#         down_margin_proportion = 0.01

#         if not rect:
#             rect = (
#                 int(width * left_margin_proportion),
#                 int(height * up_margin_proportion),
#                 int(width * (1 - right_margin_proportion)),
#                 int(height * (1 - down_margin_proportion)),
#             )




#         try:
#             cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#         except Exception as e:
#             print("Oops!", e.__class__, "occurred.: PLease verify the rectangle coordinate")

#         mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#         image = image*mask2[:,:,np.newaxis]
#         img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
#         ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_TRIANGLE)


#     edged = cv2.Canny(thresh, 30, 200)
#     contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     print('nombre de coontours', len(contours), 'type:', type(contours))
#     for it, cont in  enumerate(contours):
#         print(f"contour nubmer {it}\n coontour element {cont}")
#         print('---------------------------------------')
#         print('\n')

#     # contours = np.c_(contours)


#     if plot_contour:
#         cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)

#         return contours, image_copy
#     return contours


# class GraphicsView(QtWidgets.QGraphicsView):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         scene = QtWidgets.QGraphicsScene(self)
#         self.setScene(scene)

#         self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
#         scene.addItem(self.pixmap_item)

#         self.points = []

#         self._polygon_item = QtWidgets.QGraphicsPolygonItem(self.pixmap_item)
#         # self.polygon_item.setPen(QtGui.QPen(QtCore.Qt.black, 5, QtCore.Qt.SolidLine))
#         # self.polygon_item.setBrush(QtGui.QBrush(QtCore.Qt.red, QtCore.Qt.VerPattern))

#     @property
#     def pixmap_item(self):
#         return self._pixmap_item

#     @property
#     def polygon_item(self):
#         return self._polygon_item

#     def setPixmap(self, pixmap):
#         self.pixmap_item.setPixmap(pixmap)


#     def resizeEvent(self, event):
#         self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
#         super().resizeEvent(event)

#     def mousePressEvent(self, event):
#         sp = self.mapToScene(event.pos())
#         lp = self.pixmap_item.mapFromScene(sp)
#         self.points.append([int(sp.x()), int(sp.y())])
#         # print("SP", sp.x())
#         # print("LP", lp.toPoint())
#         poly = self.polygon_item.polygon()
#         poly.append(lp)
#         if poly.count() == 4:
#             imge = cv2.imread("/Users/souaybGA_1/Downloads/Desktop/egraphsen-tool-local/for_detect.png")
#             # imge = cv2.imread("/Users/souayboubagayoko/Desktop/Musium/reast.png")
#             pts = np.array(self.points)
#             print(pts,'type', type(pts))

#             ## (1) Crop the bounding rect
#             rect = cv2.boundingRect(pts)
#             print('rect', rect)
#             x,y,w,h = rect
#             croped = imge[y:y+h, x:x+w].copy()
#             contour, painted = grab_cat_mask(imge, rect=rect, apply_blur=True)#grab_cat_mask(imge, rect=rect, apply_blur=True)
#             # print(contour, type(contour), len(contour))
#             # if len(contour) > 1:
#             # #     return
#             # resh = np.array(contour)

#             # resh = resh.reshape(-1,2)
#             # print(resh.shape)
#             cv2.imwrite("croped.png", croped)
#             cv2.imwrite("painted.png", painted)
#             poly.clear()
#             self.points.clear()

#         dir(poly)
#         self.polygon_item.setPolygon(poly)




# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, parent=None):
#         super().__init__(parent)

#         view = GraphicsView()
#         self.setCentralWidget(view)
#         view.setPixmap(QtGui.QPixmap("/Users/souaybGA_1/Downloads/Desktop/egraphsen-tool-local/for_detect.png"))
#         # view.setPixmap(QtGui.QPixmap("/Users/souayboubagayoko/Desktop/Musium/reast.png"))
#         self.resize(640, 480)


# if __name__ == "__main__":

#     import sys

#     app = QtWidgets.QApplication(sys.argv)
#     w = MainWindow()
#     w.show()
#     sys.exit(app.exec_())

#-------------------------------------------
# SEGMENT HAND REGION FROM A VIDEO SEQUENCE
#-------------------------------------------

# # organize imports
# import cv2
# import imutils
# import numpy as np

# # global variables
# bg = None

# #--------------------------------------------------
# # To find the running average over the background
# #--------------------------------------------------
# def run_avg(image, aWeight):
#     global bg
#     # initialize the background
#     if bg is None:
#         bg = image.copy().astype("float")
#         return

#     # compute weighted average, accumulate it and update the background
#     cv2.accumulateWeighted(image, bg, aWeight)

# #---------------------------------------------
# # To segment the region of hand in the image
# #---------------------------------------------
# def segment(image, threshold=25):
#     global bg
#     # find the absolute difference between background and current frame
#     diff = cv2.absdiff(bg.astype("uint8"), image)

#     # threshold the diff image so that we get the foreground
#     thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

#     # get the contours in the thresholded image
#     (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # return None, if no contours detected
#     if len(cnts) == 0:
#         return
#     else:
#         # based on contour area, get the maximum contour which is the hand
#         segmented = max(cnts, key=cv2.contourArea)
#         return (thresholded, segmented)

# #-----------------
# # MAIN FUNCTION
# #-----------------
# if __name__ == "__main__":
#     # initialize weight for running average
#     aWeight = 0.5

#     # get the reference to the webcam
#     camera = cv2.VideoCapture(0)

#     # region of interest (ROI) coordinates
#     top, right, bottom, left = 10, 350, 225, 590

#     # initialize num of frames
#     num_frames = 0

#     # keep looping, until interrupted
#     while(True):
#         # get the current frame
#         (grabbed, frame) = camera.read()

#         # resize the frame
#         frame = imutils.resize(frame, width=700)

#         # flip the frame so that it is not the mirror view
#         frame = cv2.flip(frame, 1)

#         # clone the frame
#         clone = frame.copy()

#         # get the height and width of the frame
#         (height, width) = frame.shape[:2]

#         # get the ROI
#         roi = frame[top:bottom, right:left]

#         # convert the roi to grayscale and blur it
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (7, 7), 0)

#         # to get the background, keep looking till a threshold is reached
#         # so that our running average model gets calibrated
#         if num_frames < 30:
#             run_avg(gray, aWeight)
#         else:
#             # segment the hand region
#             hand = segment(gray)

#             # check whether hand region is segmented
#             if hand is not None:
#                 # if yes, unpack the thresholded image and
#                 # segmented region
#                 (thresholded, segmented) = hand

#                 # draw the segmented region and display the frame
#                 cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
#                 cv2.imshow("Thesholded", thresholded)

#         # draw the segmented hand
#         cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

#         # increment the number of frames
#         num_frames += 1

#         # display the frame with segmented hand
#         cv2.imshow("Video Feed", clone)

#         # observe the keypress by the user
#         keypress = cv2.waitKey(1) & 0xFF

#         # if the user pressed "q", then stop looping
#         if keypress == ord("q"):
#             break

# # free up memory
# camera.release()
# cv2.destroyAllWindows()

# import sys

# from PyQt5.QtCore import pyqtSlot
# from PyQt5.QtWidgets import QWidget, QListWidget, QListWidgetItem, QLabel, QApplication, QDialog


# class ExampleWidget(QWidget):

#     def __init__(self):
#         super().__init__()
#         listWidget = QListWidget(self)
#         listWidget.itemDoubleClicked.connect(self.buildExamplePopup)
#         for n in ["Jack", "Chris", "Joey", "Kim", "Duncan"]:
#             QListWidgetItem(n, listWidget)
#         self.setGeometry(100, 100, 100, 100)
#         self.show()

#     @pyqtSlot(QListWidgetItem)
#     def buildExamplePopup(self, item):
#         exPopup = ExamplePopup(item.text(), self)
#         exPopup.setGeometry(100, 200, 100, 100)
#         exPopup.show()


# class ExamplePopup(QDialog):

#     def __init__(self, name, parent=None):
#         super().__init__(parent)
#         self.name = name
#         self.label = QLabel(self.name, self)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     ex = ExampleWidget()
#     sys.exit(app.exec_())

# from PyQt5 import QtCore, QtGui, QtWidgets
# import sys
 
# class Ui_MainWindow(QtWidgets.QWidget):
#     def setupUi(self, MainWindow):
#         MainWindow.resize(422, 255)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
 
#         self.pushButton = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton.setGeometry(QtCore.QRect(160, 130, 93, 28))
 
#         # For displaying confirmation message along with user's info.
#         self.label = QtWidgets.QLabel(self.centralwidget)   
#         self.label.setGeometry(QtCore.QRect(170, 40, 201, 111))
 
#         # Keeping the text of label empty initially.      
#         self.label.setText("")    
 
#         MainWindow.setCentralWidget(self.centralwidget)
#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)
 
#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
#         self.pushButton.setText(_translate("MainWindow", "Proceed"))
#         self.pushButton.clicked.connect(self.takeinputs)
         
#     def takeinputs(self):
#         name, done1 = QtWidgets.QInputDialog.getText(
#              self, 'Input Dialog', 'Enter your name:')
 
#         roll, done2 = QtWidgets.QInputDialog.getInt(
#            self, 'Input Dialog', 'Enter your roll:') 
 
#         cgpa, done3 = QtWidgets.QInputDialog.getDouble(
#               self, 'Input Dialog', 'Enter your CGPA:')
 
#         langs =['C', 'c++', 'Java', 'Python', 'Javascript']
#         lang, done4 = QtWidgets.QInputDialog.getItem(
#           self, 'Input Dialog', 'Language you know:', langs)
 
#         if done1 and done2 and done3 and done4 :
#              # Showing confirmation message along
#              # with information provided by user.
#              self.label.setText('Information stored Successfully\nName: '
#                                  +str(name)+'('+str(roll)+')'+'\n'+'CGPA: '
#                                  +str(cgpa)+'\nSelected Language: '+str(lang))  
  
#              # Hide the pushbutton after inputs provided by the use.
#              self.pushButton.hide()     
               
              
              
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())


import re
from PyQt5.QtWidgets import QComboBox

from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

QT5 = QT_VERSION[0] == '5'  # NOQA

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from logger import logger
import utils
from logger import SysVars
PY2 = SysVars.PY2
QT5 = SysVars.QT5
__appname__ = SysVars.APPNAME
__version__ = SysVars.VERSION
# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


class LabelQLineEdit(QtWidgets.QLineEdit):

    def __init__(self,autocomplete=False):
        super(LabelQLineEdit, self).__init__()
        self.autocomplete = autocomplete

    def setListWidget(self, list_widget):
        self.list_widget = list_widget

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)

            # Read the current groupid for the class
            if self.autocomplete:

                if self.text().split("_")[0] in self.group_ids.keys():
                    print(self.group_ids[self.text().split("_")[0]])

                    completer = QtWidgets.QCompleter([self.text().split("_")[0]+"_"+str(self.group_ids[self.text().split("_")[0]])])
                    self.setCompleter(completer)

                    #self.setText()
                else:
                    completer = QtWidgets.QCompleter([key for key in self.group_ids.keys() if self.text() in key])
                    self.setCompleter(completer)
                    print("Nope")


    def setGroupID(self,id_dic):
        self.group_ids = id_dic


class LabelGrab(QtWidgets.QDialog):

    def __init__(self, text="Enter object label", parent=None, labels=None,
                 context=None, sort_labels=True, show_text_field=True,

                 state=None, person=None, orient=None, phrase=None,

                 completion='startswith', fit_to_content=None, flags=None,
                 levels=["Head","Shoulders"]):
        if fit_to_content is None:
            fit_to_content = {'row': False, 'column': True}
        self._fit_to_content = fit_to_content

        # Custom size
        self.height = 500
        self.width  = 10
        self.completion = "startswith" #completion  # added (souayb)
        print("self.completion", self.completion)
        self.resized_triggered = False
        self.autocomplet_trigger = True
        # Automatic group id
        self.group_ids = {}

        super(LabelGrab, self).__init__(parent)
        self.edit = LabelQLineEdit()
        self.edit.setPlaceholderText(text)
        self.edit.setValidator(utils.labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        if flags:
            self.edit.textChanged.connect(self.updateFlags)
        self.edit_group_id = QtWidgets.QLineEdit()
        self.edit_group_id.setPlaceholderText('Group ID')
        self.edit_group_id.setValidator(
            QtGui.QRegExpValidator(QtCore.QRegExp(r'\d*'), None)
        )

        ## New Label Options Edit
        self.edit_context = LabelQLineEdit()
        self.edit_context.setPlaceholderText('Context')

        self.edit_state = LabelQLineEdit()
        self.edit_state.setPlaceholderText('State')

        self.edit_person = LabelQLineEdit()
        self.edit_person.setPlaceholderText('Person')

        self.edit_orient = LabelQLineEdit()
        self.edit_orient.setPlaceholderText('Orientation')

        self.edit_phrase = LabelQLineEdit()
        self.edit_phrase.setPlaceholderText('Phrase')

        self.edit_level = LabelQLineEdit()
        self.edit_level.setPlaceholderText('Parent')
        

        layout = QtWidgets.QVBoxLayout()
        if show_text_field:
            layout_edit = QtWidgets.QHBoxLayout()
            layout_edit.addWidget(self.edit, 7)
            layout_edit.addWidget(self.edit_group_id, 3)

            # Add new options to widget
            layout_edit.addWidget(self.edit_context,10)
            layout_edit.addWidget(self.edit_state,10)
            layout_edit.addWidget(self.edit_person,10)
            layout_edit.addWidget(self.edit_orient,10)
            layout_edit.addWidget(self.edit_phrase,10)
            layout.addLayout(layout_edit)

        # Add context to Widget
        #layout_context = QtWidgets.QHBoxLayout()
        #layout_context.addWidget(self.edit_context, 10)
        #layout.addLayout(layout_context)
        # buttons
        self.buttonBox = bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(utils.newIcon('done'))
        bb.button(bb.Cancel).setIcon(utils.newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        

        # label_list
        self.labelList = QtWidgets.QListWidget()
        if self._fit_to_content['row']:
            self.labelList.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        if self._fit_to_content['column']:
            self.labelList.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        self._sort_labels = sort_labels
        if labels:
            self.labelList.addItems(labels)
        if self._sort_labels:
            self.labelList.sortItems()
        else:
            self.labelList.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.labelList.currentItemChanged.connect(self.labelSelected)
        self.labelList.itemDoubleClicked.connect(self.labelDoubleClicked)
        self.edit.setListWidget(self.labelList)
        #layout.addWidget(self.labelList)


        # Add lists for new options 
        def createList(oList, option):
            if self._fit_to_content['row']:
                oList.setHorizontalScrollBarPolicy(
                    QtCore.Qt.ScrollBarAlwaysOff
                )
            if self._fit_to_content['column']:
                oList.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarAlwaysOff
                )

            if option:
                oList.addItems(option)
            if self._sort_labels:
                oList.sortItems()
            else:
                oList.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.contextList = QtWidgets.QListWidget()
        createList(self.contextList,context)
        self.contextList.currentItemChanged.connect(self.contextSelected)
        self.contextList.itemDoubleClicked.connect(self.contextDoubleClicked)
        self.edit_context.setListWidget(self.contextList)

        self.stateList = QtWidgets.QListWidget()
        createList(self.stateList,state)
        self.stateList.currentItemChanged.connect(self.stateSelected)
        self.stateList.itemDoubleClicked.connect(self.stateDoubleClicked)
        self.edit_state.setListWidget(self.stateList)

        self.personList = QtWidgets.QListWidget()
        createList(self.personList,person)
        self.personList.currentItemChanged.connect(self.personSelected)
        self.personList.itemDoubleClicked.connect(self.personDoubleClicked)
        self.edit_person.setListWidget(self.personList)

        self.orientList = QtWidgets.QListWidget()

        createList(self.orientList,orient)
        self.orientList.currentItemChanged.connect(self.orientSelected)
        self.orientList.itemDoubleClicked.connect(self.orientDoubleClicked)
        self.edit_orient.setListWidget(self.orientList)

        self.phraseList = QtWidgets.QListWidget()
        createList(self.phraseList,phrase)
        self.phraseList.currentItemChanged.connect(self.phraseSelected)
        self.phraseList.itemDoubleClicked.connect(self.phraseDoubleClicked)
        self.edit_phrase.setListWidget(self.phraseList)
        

        layout_lists = QtWidgets.QHBoxLayout()
        layout_lists.addWidget(self.labelList, 10)
        layout_lists.addWidget(self.contextList, 10)
        layout_lists.addWidget(self.stateList, 10)
        layout_lists.addWidget(self.personList, 10)
        layout_lists.addWidget(self.orientList, 10)
        layout_lists.addWidget(self.phraseList, 10)
        layout.addLayout(layout_lists)




        # Hierarchy selection
        self.levelList = QtWidgets.QListWidget()
        self.levelList.currentItemChanged.connect(self.levelSelected)
        self.levelList.itemDoubleClicked.connect(self.levelDoubleClicked)
        self.edit_level.setListWidget(self.levelList)

        if self._fit_to_content['row']:
            self.levelList.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        if self._fit_to_content['column']:
            self.levelList.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )

        #if levels:
        #    self.levelList.addItems(levels)
        if self._sort_labels:
            self.levelList.sortItems()
        else:
            self.levelList.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)


        ######## SETTING UP THE BUTTON TO CHOOSE AUTOCOMPLETION OR DROPDOWN ####3
        # added by souayb
        self.cb = QComboBox()
        if self.autocomplet_trigger:
            self.cb.addItem("Dropdown")
            self.cb.addItem("Autocomp")
        else :
            self.cb.addItem("Autocomp")
            self.cb.addItem("Dropdown")
        self.cb.activated[str].connect(self.autocomplet)
        # layout.addWidget(self.edit_level) ## removed (souayb)
        layout_l = QtWidgets.QHBoxLayout()
        layout_l.addWidget(self.edit_level, 4)
        layout_l.addStretch()
        layout_l.addWidget(self.cb, 1)
        layout.addLayout(layout_l)
        #### end ######

        layout.addWidget(self.levelList)

        # Moved Buttons
        layout.addWidget(bb)

        # label_flags
        if flags is None:
            flags = {}
        self._flags = flags
        self.flagsLayout = QtWidgets.QVBoxLayout()
        self.resetFlags()
        layout.addItem(self.flagsLayout)
        self.edit.textChanged.connect(self.updateFlags)
        self.setLayout(layout)
        # completion Label
        

        # completion Context
        ######################### CHANGED BY SOUAYBOU #################
        # def setCompleter(newCompleter, completion):
        #     if not QT5 and completion != 'startswith':
        #         logger.warn(
        #             "completion other than 'startswith' is only "
        #             "supported with Qt5. Using 'startswith'"
        #         )
        #         completion = 'startswith'
        #     if False and completion == 'startswith':
        #         newCompleter.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
        #         # Default settings.
        #         # completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        #     elif True or completion == 'contains':
        #         newCompleter.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        #         #newCompleter.setFilterMode(QtCore.Qt.MatchContains)
        #     else:
        #         raise ValueError('Unsupported completion: {}'.format(completion))
        ###############################################################
        
        completer = QtWidgets.QCompleter()
        #completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.setCompleter(completer, self.completion)
        completer.setModel(self.labelList.model())
        self.edit.setCompleter(completer)   

        completer_context = QtWidgets.QCompleter()
        self.setCompleter(completer_context, self.completion)
        completer_context.setModel(self.contextList.model())
        self.edit_context.setCompleter(completer_context)

        completer_state = QtWidgets.QCompleter()
        self.setCompleter(completer_state, self.completion)
        completer_state.setModel(self.stateList.model())
        self.edit_state.setCompleter(completer_state)

        completer_person = QtWidgets.QCompleter()
        self.setCompleter(completer_person, self.completion)
        completer_person.setModel(self.personList.model())
        self.edit_person.setCompleter(completer_person)

        completer_orient = QtWidgets.QCompleter()
        self.setCompleter(completer_orient, self.completion)
        completer_orient.setModel(self.orientList.model())
        self.edit_orient.setCompleter(completer_orient)

        completer_phrase = QtWidgets.QCompleter()
        self.setCompleter(completer_phrase, self.completion)
        completer_phrase.setModel(self.phraseList.model())
        self.edit_phrase.setCompleter(completer_phrase)

        completer_level = QtWidgets.QCompleter()
        self.setCompleter(completer_level, self.completion)
        completer_level.setModel(self.levelList.model())
        self.edit_level.setCompleter(completer_level)


    def setCompleter(self, newCompleter, completion):

        if not QT5 and completion != 'startswith':
            logger.warn(
                "completion other than 'startswith' is only "
                "supported with Qt5. Using 'startswith'"
            )
            completion = 'startswith'

        if  self.autocomplet_trigger :
            newCompleter.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            
            # Default settings.
            # completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        else :
            newCompleter.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            #newCompleter.setFilterMode(QtCore.Qt.MatchContains)
        
    def autocomplet(self, text):
        if text == "Dropdown":
            self.autocomplet_trigger = True
            completer = QtWidgets.QCompleter()
            #completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            self.setCompleter(completer, self.completion)
            completer.setModel(self.labelList.model())
            self.edit.setCompleter(completer)   

            completer_context = QtWidgets.QCompleter()
            self.setCompleter(completer_context, self.completion)
            completer_context.setModel(self.contextList.model())
            self.edit_context.setCompleter(completer_context)

            completer_state = QtWidgets.QCompleter()
            self.setCompleter(completer_state, self.completion)
            completer_state.setModel(self.stateList.model())
            self.edit_state.setCompleter(completer_state)

            completer_person = QtWidgets.QCompleter()
            self.setCompleter(completer_person, self.completion)
            completer_person.setModel(self.personList.model())
            self.edit_person.setCompleter(completer_person)

            completer_orient = QtWidgets.QCompleter()
            self.setCompleter(completer_orient, self.completion)
            completer_orient.setModel(self.orientList.model())
            self.edit_orient.setCompleter(completer_orient)

            completer_phrase = QtWidgets.QCompleter()
            self.setCompleter(completer_phrase, self.completion)
            completer_phrase.setModel(self.phraseList.model())
            self.edit_phrase.setCompleter(completer_phrase)

            completer_level = QtWidgets.QCompleter()
            self.setCompleter(completer_level, self.completion)
            completer_level.setModel(self.levelList.model())
            self.edit_level.setCompleter(completer_level)
        else:
            self.autocomplet_trigger = False
            completer = QtWidgets.QCompleter()
            #completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            self.setCompleter(completer, self.completion)
            completer.setModel(self.labelList.model())
            self.edit.setCompleter(completer)   

            completer_context = QtWidgets.QCompleter()
            self.setCompleter(completer_context, self.completion)
            completer_context.setModel(self.contextList.model())
            self.edit_context.setCompleter(completer_context)

            completer_state = QtWidgets.QCompleter()
            self.setCompleter(completer_state, self.completion)
            completer_state.setModel(self.stateList.model())
            self.edit_state.setCompleter(completer_state)

            completer_person = QtWidgets.QCompleter()
            self.setCompleter(completer_person, self.completion)
            completer_person.setModel(self.personList.model())
            self.edit_person.setCompleter(completer_person)

            completer_orient = QtWidgets.QCompleter()
            self.setCompleter(completer_orient, self.completion)
            completer_orient.setModel(self.orientList.model())
            self.edit_orient.setCompleter(completer_orient)

            completer_phrase = QtWidgets.QCompleter()
            self.setCompleter(completer_phrase, self.completion)
            completer_phrase.setModel(self.phraseList.model())
            self.edit_phrase.setCompleter(completer_phrase)

            completer_level = QtWidgets.QCompleter()
            self.setCompleter(completer_level, self.completion)
            completer_level.setModel(self.levelList.model())
            self.edit_level.setCompleter(completer_level)

    def setLevels(self,levels):
        self.levels = levels

    def addLabelHistory(self, label):
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.labelList.addItem(label)
        if self._sort_labels:
            self.labelList.sortItems()

    def addHistory(self, option, oList):
        if oList.findItems(option, QtCore.Qt.MatchExactly):
            return
        oList.addItem(option)
        if self._sort_labels:
            oList.sortItems()

    def addContextHistory(self, context):
        if self.contextList.findItems(context, QtCore.Qt.MatchExactly):
            return
        self.contextList.addItem(context)
        if self._sort_labels:
            self.contextList.sortItems()

    def updateLevels(self, level):
        self.levelList.clear()
        self.levelList.addItems(level)
        if self._sort_labels:
            self.levelList.sortItems()

    def labelSelected(self, item):
        self.edit.setText(item.text())

    def levelSelected(self,item):
        print("Level Selected",item)
        try:
            self.edit_level.setText(item.text())
        except AttributeError:
            print("Loading registry...")

    def contextSelected(self, item):
        self.edit_context.setText(item.text())

    def stateSelected(self, item):
        self.edit_state.setText(item.text())

    def personSelected(self, item):
        self.edit_person.setText(item.text())

    def orientSelected(self, item):
        self.edit_orient.setText(item.text())

    def phraseSelected(self, item):
        self.edit_phrase.setText(item.text())    


    def optionSelected(self, item, edit_option):
        edit_option.setText(item.text())

    def validate(self):
        text = self.edit.text()
        if hasattr(text, 'strip'):
            text = text.strip()
        else:
            text = text.trimmed()
        if text:
            self.accept()

    def validate_context(self):
        text = self.edit_context.text()
        if hasattr(text, 'strip'):
            text = text.strip()
        else:
            text = text.trimmed()
        if text:
            self.accept()

    def validate_option(self,edit_option):
        text = edit_option.text()
        if hasattr(text, 'strip'):
            text = text.strip()
        else:
            text = text.trimmed()
        if text:
            self.accept()

    def labelDoubleClicked(self, item):
        self.validate()

    def contextDoubleClicked(self, item):
        self.validate_context()

    def stateDoubleClicked(self, item):
        self.validate_option(self.edit_state)

    def personDoubleClicked(self, item):
        self.validate_option(self.edit_person)

    def orientDoubleClicked(self, item):
        self.validate_option(self.edit_orient)

    def phraseDoubleClicked(self, item):
        self.validate_option(self.edit_phrase)

    def levelDoubleClicked(self, item):
        self.validate_option(self.edit_level)

    def optionDoubleClicked(self, item, edit_option):
        self.validate_option(edit_option)

    def postProcess(self):
        text = self.edit.text()
        if hasattr(text, 'strip'):
            text = text.strip()
        else:
            text = text.trimmed()
        self.edit.setText(text)

    def updateFlags(self, label_new):
        # keep state of shared flags
        flags_old = self.getFlags()

        flags_new = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label_new):
                for key in keys:
                    flags_new[key] = flags_old.get(key, False)
        self.setFlags(flags_new)

    def deleteFlags(self):
        for i in reversed(range(self.flagsLayout.count())):
            item = self.flagsLayout.itemAt(i).widget()
            self.flagsLayout.removeWidget(item)
            item.setParent(None)

    def resetFlags(self, label=''):
        flags = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label):
                for key in keys:
                    flags[key] = False
        self.setFlags(flags)

    def setFlags(self, flags):
        self.deleteFlags()
        for key in flags:
            item = QtWidgets.QCheckBox(key, self)
            item.setChecked(flags[key])
            self.flagsLayout.addWidget(item)
            item.show()

    def getFlags(self):
        flags = {}
        for i in range(self.flagsLayout.count()):
            item = self.flagsLayout.itemAt(i).widget()
            flags[item.text()] = item.isChecked()
        return flags

    def getGroupId(self):
        group_id = self.edit_group_id.text()
        if group_id:
            return int(group_id)
        return None

    def getContext(self):
        context = self.edit_context.text()
        if context:
            return str(context)
        return None

    def getOption(self,edit_option):
        option = edit_option.text()
        if option:
            return str(option)
        return None

    def getLevel(self):

        level = [item.text() for item in self.levelList.selectedItems()]
        if level:
            return level
        return None

    # Sets the size of the label dialog widget
    def setWinSize(self,size):
        print("Setting size to",size)
        height,width = size
        self.height = height-30
        self.width = width

    def moveEvent(self, a0: QtGui.QMoveEvent) -> None:
        # return super().moveEvent(a0)
        self.pos_x, self.pos_y = self.frameGeometry().left(), self.frameGeometry().top()
        self.resized_triggered = True

    def resizeEvent(self,event):
        self.width = self.frameGeometry().width()
        self.height = self.frameGeometry().height()
        self.resized_triggered = True
        #settings_obj = QtCore.QSettings(self.settings_path, QtCore.QSettings.IniFormat)
        #settings_obj.setValue("windowGeometry", self.saveGeometry())

    def popUp(self, text=None, move=False, flags=None, group_id=None,context=None, state=None, person=None, orient=None, phrase=None,level=None):
        self.resize(self.width,self.height)

        if self._fit_to_content['row']:
            self.labelList.setMinimumHeight(
                self.labelList.sizeHintForRow(0) * self.labelList.count() + 2
            )
        if self._fit_to_content['column']:
            self.labelList.setMinimumWidth(
                self.labelList.sizeHintForColumn(0) + 2
            )
        # if text is None, the previous label in self.edit is kept
        if text is None:
            text = self.edit.text()
        if flags:
            self.setFlags(flags)
        else:
            self.resetFlags(text)

        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        if group_id is None:
            self.edit_group_id.clear()
        else:
            self.edit_group_id.setText(str(group_id))

        if context is None:
            self.edit_context.clear()
        else:
            self.edit_context.setText(str(context))

        if state is None:
            self.edit_state.clear()
        else:
            self.edit_state.setText(str(state))

        if person is None:
            self.edit_person.clear()
        else:
            self.edit_person.setText(str(person))

        if orient is None:
            self.edit_orient.clear()
        else:
            self.edit_orient.setText(str(orient))

        if phrase is None:
            self.edit_phrase.clear()
        else:
            self.edit_phrase.setText(str(phrase))

        if level is None:
            self.edit_level.clear()
        else:
            self.edit_level.setText(level[0])

        # TEST
        self.edit_level.setGroupID(self.group_ids)

        #if level is not None:
        #    try:
        #        item = self.levelList.findItems(level[0], QtCore.Qt.MatchFixedString)
        #        self.levelList.setCurrentItem(item[0])
        #    except IndexError:
        #        print("Copied")


        items = self.contextList.findItems(str(context), QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.contextList.setCurrentItem(items[0])

        items = self.stateList.findItems(str(state), QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.stateList.setCurrentItem(items[0])

        items = self.personList.findItems(str(person), QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.personList.setCurrentItem(items[0])

        #print("Oriental",orient,type(orient))
        #print("List",[str(self.orientList.item(i).text()) for i in range(self.orientList.count())])
        items = self.orientList.findItems(str(orient), QtCore.Qt.MatchFixedString)
        #print("Found items",items)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            #print("Setting current item for orient:",items[0].text())
            self.orientList.setCurrentItem(items[0])

        items = self.phraseList.findItems(str(phrase), QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.phraseList.setCurrentItem(items[0])

        if level is not None:
            #print("Level",level,type(level))

            items = []
            for i in range(self.levelList.count()):
                if str(self.levelList.item(i).text()) == level:
                    items.append(self.levelList.item(i))

            #print("List",[str(self.levelList.item(i).text()) for i in range(self.levelList.count())])
            #items = self.levelList.findItems(level[0], QtCore.Qt.MatchFixedString)
            #print("Found items",items)
        else:
            items = self.levelList.findItems(None, QtCore.Qt.MatchFixedString)
        
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            print("Setting current item for level",items)
            self.levelList.setCurrentItem(items[0])

        items = self.labelList.findItems(text, QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.labelList.setCurrentItem(items[0])
            row = self.labelList.row(items[0])
            self.edit.completer().setCurrentRow(row)
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)
        if move:
            self.move(QtGui.QCursor.pos())
        ##### function added that postioned the popup widget according to the given postion 
        if self.resized_triggered:
            self.move( self.pos_x, self.pos_y)

        # Experimental id
        

        if self.exec_():

            if self.getGroupId() is None:
                if self.edit.text() not in self.group_ids.keys():
                    self.group_ids[self.edit.text()] = 0
                else:
                    self.group_ids[self.edit.text()] += 1

                new_groupid = self.group_ids[self.edit.text()]

            else:
                print(self.getGroupId())
                new_groupid = self.getGroupId()


            #print(self.getLevel())
            return self.edit.text(), self.getFlags(), new_groupid, self.getOption(self.edit_context), self.getOption(self.edit_state), self.getOption(self.edit_person), self.getOption(self.edit_orient), self.getOption(self.edit_phrase), self.getOption(self.edit_level)
        else:
            return None, None, None, None, None, None, None, None, None,

