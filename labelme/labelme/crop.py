# import numpy as np 
# import cv2 
# # import the necessary packages
# import argparse

# # initialize the list of reference points and boolean indicating
# # whether cropping is being performed or not
# ref_point = []
# cropping = True

# def shape_selection(event, x, y, flags, param):
#   # grab references to the global variables
#   global ref_point, cropping

#   # if the left mouse button was clicked, record the starting
#   # (x, y) coordinates and indicate that cropping is being
#   # performed
#   if event == cv2.EVENT_LBUTTONDOWN:
#     ref_point = [(x, y)]
#     cropping = True

#   # check to see if the left mouse button was released
#   elif event == cv2.EVENT_LBUTTONUP:
#     # record the ending (x, y) coordinates and indicate that
#     # the cropping operation is finished
#     ref_point.append((x, y))
#     cropping = False

#     # draw a rectangle around the region of interest
#     cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
#     cv2.imshow("image", image)

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# # load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"])
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", shape_selection)

# # keep looping until the 'q' key is pressed
# while True:
#   # display the image and wait for a keypress
#   cv2.imshow("image", image)
#   key = cv2.waitKey(1) & 0xFF

#   # if the 'r' key is pressed, reset the cropping region
#   if key == ord("r"):
#     image = clone.copy()

#   # if the 'c' key is pressed, break from the loop
#   elif key == ord("c"):
#     break

# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# if len(ref_point) == 2:
#   crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
#   cv2.imshow("crop_img", crop_img)
#   cv2.waitKey(0)

# # close all open windows
# cv2.destroyAllWindows()


# import numpy as np
# import cv2
# path = "/Users/souaybGA_1/Desktop/egraphsen-tool-local/Fazzu_test.png"
# img = cv2.imread(path)
# pts = np.array([[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]])

# ## (1) Crop the bounding rect
# rect = cv2.boundingRect(pts)
# x,y,w,h = rect
# croped = img[y:y+h, x:x+w].copy()

# ## (2) make mask
# pts = pts - pts.min(axis=0)

# mask = np.zeros(croped.shape[:2], np.uint8)
# cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

# ## (3) do bit-op
# dst = cv2.bitwise_and(croped, croped, mask=mask)

# ## (4) add the white background
# bg = np.ones_like(croped, np.uint8)*255
# cv2.bitwise_not(bg,bg, mask=mask)
# dst2 = bg+ dst


# cv2.imwrite("croped.png", croped)
# cv2.imwrite("mask.png", mask)
# cv2.imwrite("dst.png", dst)
# cv2.imwrite("dst2.png", dst2)

# cv2.waitKey(0)



# import numpy as np
# import cv2
# import os 
# directory = "/Users/souaybGA/Desktop/egraphsen-tool-local/painter/Makron/[Makron] Vermutend zugeordnet"
# print(os.listdir(directory))
# from pathlib import Path
# filelist = os.listdir(directory)

# for file in filelist:
#     filename = Path(file).stem
#     print(filename)



# if __name__ == '__main__':
#     path = "/Users/souaybGA/Desktop/egraphsen-tool-local/Images/test_image.jpg"
#     img = cv2.imread(path)
#     mask = np.zeros(img.shape[0:2], dtype=np.uint8)
#     points = np.array([[[100,350],[120,400],[310,350],[360,200],[350,20],[25,120]]])
#     #method 1 smooth region
#     cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
#     #method 2 not so smooth region
#     # cv2.fillPoly(mask, points, (255))
#     res = cv2.bitwise_and(img,img,mask = mask)
#     rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
#     cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
#     ## crate the white background of the same size of original image
#     wbg = np.ones_like(img, np.uint8)*255
#     cv2.bitwise_not(wbg,wbg, mask=mask)
#     # overlap the resulted cropped image on the white background
#     dst = wbg+res
#     # cv2.imshow('Original',img)
#     # cv2.imshow("Mask",mask)
#     cv2.imshow("Cropped", cropped )
#     cv2.imshow("Samed Size Black Image", res)
#     cv2.imshow("Samed Size White Image", dst)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()





from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)

def thresh_callback(val):
    threshold = val
    
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    
    
    _, contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    
    
    cv.imshow('Contours', drawing)
    
parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()
