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



import cv2
import numpy as np

# read input
img = cv2.imread("/Users/souaybGA_1/Downloads/Desktop/egraphsen-tool-local/painted.png")#"/Users/souaybGA_1/Downloads/Desktop/1007_hand.png")

# convert to hsv and get saturation channel
sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]

# do a little Gaussian filtering
blur = cv2.GaussianBlur(sat, (3,3), 0)


# threshold and invert to create initial mask
mask = 255 - cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]

# apply morphology close to fill interior regions in mask
kernel = np.ones((15,15), np.uint8)
mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# get outer contours from inverted mask and get the largest (presumably only one due to morphology filtering)
cntrs = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
result = img.copy()
area_thresh = 0
for c in cntrs:
    area = cv2.contourArea(c)
    if area > area_thresh:
        area = area_thresh
        big_contour = c

# draw largest contour
cv2.drawContours(result, [big_contour], -1, (0,0,255), 2)


# display it
cv2.imshow("Blur", blur)
cv2.imshow("Sat", sat)
cv2.imshow("IMAGE", img)
cv2.imshow("MASK", mask)
cv2.imshow("MASK1", mask1)
cv2.imshow("RESULT", result)
cv2.waitKey(0)