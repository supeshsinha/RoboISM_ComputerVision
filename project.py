
import cv2
import numpy as np
import imutils
import math
from stacksgo import *
 
 
def processImg(img):    #Main Function
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)           #Drawing Contours
            peri = cv2.arcLength(cnt,True)

            approx = cv2.approxPolyDP(cnt,0.02*peri,True)           #approx is coordinates of corners
            print(len(approx))


            objCor = len(approx)                # objCor is Number of Corners
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 4:         # Difference of lenghts of two adjecent sides using distance formula
                lendiff= (approx[0][0][0] - approx[1][0][0])**2 + (approx[0][0][1] - approx[1][0][1])**2 - (approx[1][0][0] - approx[2][0][0])**2 - (approx[1][0][1] - approx[2][0][1])**2

                if lendiff > -2000 and lendiff < 2000:      # Difference should be approximately less than 150px
                    objectType= "Square"

                    id=1
                    roi = imgContour[y:y+h, x:x+w]          # Identifying Colors
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    lower_val = np.array([37,42,0])         #Identifying Green
                    upper_val = np.array([84,255,255])
                    mask = cv2.inRange(hsv, lower_val, upper_val)
                    hasGreen = np.sum(mask)
                    if hasGreen > 0:
                        id=1

                    lower_val = np.array([32.8,100,100])    # Identifying Orange
                    upper_val = np.array([48.8,255,255])
                    mask = cv2.inRange(hsv, lower_val, upper_val)
                    hasOrange = np.sum(mask)
                    if hasOrange > 0:
                        id=2
                    
                    lower_val = np.array([-10,100,100])     # Identifying Black
                    upper_val = np.array([10,255,255])
                    mask = cv2.inRange(hsv, lower_val, upper_val)
                    hasBlack = np.sum(mask)
                    if hasBlack > 0:
                        id=3

                    lower_val = np.array([18.3,100,100])    # Identifying Peach Pink
                    upper_val = np.array([32.3,255,255])
                    mask = cv2.inRange(hsv, lower_val, upper_val)
                    hasPeach = np.sum(mask)
                    if hasPeach > 0:
                        id=4

                                    # w=h= edge of square using distance formula
                    width=height = int(((approx[0][0][0] - approx[1][0][0])**2 + (approx[0][0][1] - approx[1][0][1])**2)**0.5)

                    pts1 = np.float32([corner[id][0][0][0],corner[id][0][0][1],corner[id][0][0][2],corner[id][0][0][3]])
                    pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])      #Coordinates of aruco Corners
                    matrix = cv2.getPerspectiveTransform(pts1,pts2)
                    ar1 = cv2.warpPerspective(aruco[id],matrix,(width,height))          # Straightning, Cropping and Resizing Aruco
                    
                    
                    slope = (approx[1][0][1]-approx[0][0][1])/(approx[1][0][0]-approx[0][0][0])   # Slope of Upper Edge of Square
                    angle= math.degrees(math.atan(slope))           #angle is tan inverse of slope
                    
                    rotated = imutils.rotate_bound(ar1, angle)      #Aruco Rotated by angle of slope to match square orientation
                    small_img = cv2.resize(rotated,(w,h))
                    rows,columns,chanels = small_img.shape
                    small_img_gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
                    ret, mask = cv2.threshold(small_img_gray, 120, 255, cv2.THRESH_BINARY)
                    bg = cv2.bitwise_not(roi,roi,mask = mask)
                    fg = cv2.bitwise_and(small_img,small_img, mask=mask)
                    final_roi = cv2.add(bg,fg)              # Pasting Aruco over the squares
                    small_img = final_roi
                    imgContour[y : y + small_img.shape[0], x : x + small_img.shape[1]]= small_img

            else:objectType="Not square"


            cv2.putText(imgContour,objectType,                  # Putting text on squares for identification
                         (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                         (255,0,0),2)

 


imgContour = cv2.imread("assets/cvtask.jpg")  # Importing Image

aruco=[0,0,0,0,0]
corner=[0,0,0,0,0]

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
arucoParams = cv2.aruco.DetectorParameters_create()
i=1
while i<5:
    aruco[i] = cv2.imread("assets/image"+str(i)+".jpg")   #Identifying aruco and putting the image and corners in diff arrays
    (corner[i], ids, rejected) = cv2.aruco.detectMarkers(aruco[i], arucoDict, parameters=arucoParams)
    i=i+1


img = imgContour.copy()
 
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray,50,50)             # Canny Image
processImg(imgCanny)
imgStack = stackImages(0.4,([img,imgContour]))      # Stancking Original and Processed Images Side by Side
cv2.imshow("Stack", imgStack)
 
 
cv2.waitKey(0)