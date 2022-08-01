# import cv2
# import numpy as np 
# from PIL  import Image, ImageStat

# import traceback

 

# def is_grayscale(arr):
#     im = Image.fromarray(arr)
#     stat = ImageStat.Stat(im)

#     if sum(stat.sum)/3 == stat.sum[0]:
#         return True
#     else:
#         return False

# def grab_cat_mask(image, rect:list=None, 
#                   plot_contour:bool=True, 
#                   finegrain_contour:bool=False, 
#                   apply_blur:bool=True):
#     """
#     Input:
#         image: type= numpy array
#         rect:(default=None) the focus box for grabcut if None the entire image is used
#         plot_contour
#     Ouptut:
#         contour:list of all the contours 
#         mask 
#     """
#     assert isinstance(image, np.ndarray), 'image must be a numpy array'
    
#     image_copy = image.copy()
#     if is_grayscale(image):
        
#         if rect:
#             thresh = np.zeros(img.shape[:2],np.uint8)
#             x,y, w, h = rect
#             crop = image[y:y+h, x:x+w]
            
#             try:
#                 crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#             except Exception as e:
#                 raise ValueError("PLease very that the rectangle coordinate is correct")

#             if apply_blur:
#                  crop =  cv2.GaussianBlur(crop, (3, 3), 0)
#             ret, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
            
#             thresh[y:y+h, x:x+w] = threshold
#         else:
#             img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             if apply_blur:
#                  img_gray =  cv2.GaussianBlur(img_gray, (3, 3), 0)
#             ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
        
#     else:
#         height, width  = image.shape[:2]
#         mask = np.zeros((height, width),np.uint8)
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
#             print("Oops!", e.__class__, "occurred.: PLease very that the rectangle coordinate")
#             print("erroorrr", traceback.print_exc())    
#         mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#         image = image*mask2[:,:,np.newaxis]
#         img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
#         ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_TRIANGLE)

    
#     edged = cv2.Canny(thresh, 30, 200)
#     contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if plot_contour:
#         cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 3, cv2.LINE_AA)
    
#         return contours, image_copy
#     return contours 
# if __name__ == '__main__':
#     # img = cv2.imread('/Users/souaybGA_1/Desktop/tt.png')
#     img = cv2.imread("/Users/souaybGA_1/Downloads/Desktop/1007_hand.png")
 
#     rect = None        #(700,400,850,500)
#     contour, painted = grab_cat_mask(img, rect=rect, apply_blur=True) 
    
#     cv2.imshow("ouput", painted)
#     cv2.imshow("oupututut", img)
#     cv2.waitKey(0)



# import cv2
# import numpy as np

# # read input
# img = cv2.imread("/Users/souaybGA_1/Downloads/Desktop/egraphsen-tool-local/painted.png")#"/Users/souaybGA_1/Downloads/Desktop/1007_hand.png")

# # convert to hsv and get saturation channel
# sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]

# # do a little Gaussian filtering
# blur = cv2.GaussianBlur(sat, (3,3), 0)


# # threshold and invert to create initial mask
# mask = 255 - cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]

# # apply morphology close to fill interior regions in mask
# kernel = np.ones((15,15), np.uint8)
# mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# # get outer contours from inverted mask and get the largest (presumably only one due to morphology filtering)
# cntrs = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
# result = img.copy()
# area_thresh = 0
# for c in cntrs:
#     area = cv2.contourArea(c)
#     if area > area_thresh:
#         area = area_thresh
#         big_contour = c

# # draw largest contour
# cv2.drawContours(result, [big_contour], -1, (0,0,255), 2)


# # display it
# cv2.imshow("Blur", blur)
# cv2.imshow("Sat", sat)
# cv2.imshow("IMAGE", img)
# cv2.imshow("MASK", mask)
# cv2.imshow("MASK1", mask1)
# cv2.imshow("RESULT", result)
# cv2.waitKey(0)




# #-------------------------------------------
# # SEGMENT HAND REGION FROM A VIDEO SEQUENCE
# #-------------------------------------------

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


import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                fingers = count(thresholded, segmented)

                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()