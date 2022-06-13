
import cv2
import numpy as np
import imutils
import math
from stacksgo import *
 
 
def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))


            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 4:
                lendiff= (approx[0][0][0] - approx[1][0][0])**2 + (approx[0][0][1] - approx[1][0][1])**2 - (approx[1][0][0] - approx[2][0][0])**2 - (approx[1][0][1] - approx[2][0][1])**2

                if lendiff > -2000 and lendiff < 2000:
                    objectType= "Square"


                    roi = imgContour[y:y+h, x:x+w]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    lower_val = np.array([37,42,0]) 
                    upper_val = np.array([84,255,255])
                    mask = cv2.inRange(hsv, lower_val, upper_val)
                    hasGreen = np.sum(mask)
                    if hasGreen > 0:
                        print('Green detected!')


                    width=height = int(((approx[0][0][0] - approx[1][0][0])**2 + (approx[0][0][1] - approx[1][0][1])**2)**0.5)
                    pts1 = np.float32([corner1[0][0][0],corner1[0][0][1],corner1[0][0][2],corner1[0][0][3]])
                    pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])
                    matrix = cv2.getPerspectiveTransform(pts1,pts2)
                    global ar1
                    ar1 = cv2.warpPerspective(aruco1,matrix,(width,height))
                    slope = (approx[1][0][1]-approx[0][0][1])/(approx[1][0][0]-approx[0][0][0])
                    angle= math.degrees(math.atan(slope))
                    rotated = imutils.rotate_bound(ar1, angle)
                    small_img = cv2.resize(rotated,(w,h))
                    rows,columns,chanels = small_img.shape
                    small_img_gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
                    ret, mask = cv2.threshold(small_img_gray, 120, 255, cv2.THRESH_BINARY)
                    bg = cv2.bitwise_not(roi,roi,mask = mask)
                    fg = cv2.bitwise_and(small_img,small_img, mask=mask)
                    final_roi = cv2.add(bg,fg)
                    small_img = final_roi
                    imgContour[y : y + small_img.shape[0], x : x + small_img.shape[1]]= small_img

            else:objectType="Not square"


            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,
                         (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                         (255,0,0),2)

 
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
arucoParams = cv2.aruco.DetectorParameters_create()



imgContour = cv2.imread("assets/cvtask.jpg")
aruco1 = cv2.imread("assets/1.jpg")
(corner1, ids, rejected) = cv2.aruco.detectMarkers(aruco1, arucoDict, parameters=arucoParams)
aruco2 = cv2.imread("assets/2.jpg")
aruco3 = cv2.imread("assets/3.jpg")
aruco4 = cv2.imread("assets/4.jpg")

img = imgContour.copy()
 
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgGray,50,50)
getContours(imgCanny)
 
imgBlank = np.zeros_like(img)
# imgStack = stackImages(0.3,([img,imgGray],
#                             [imgCanny,imgContour]))

imgStack = stackImages(0.3,([imgBlank,imgBlank],
                            [imgCanny,imgContour]))
 
cv2.imshow("Stack", imgStack)
cv2.imshow("Stackj", ar1)
 
cv2.waitKey(0)