

import numpy as np
import cv2

im = cv2.imread('D:/practice/contour/sample_images/result/11.jpg')
cv2.imshow('original image',im)
cv2.waitKey()

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image',imgray)
cv2.waitKey()

ret,thresh = cv2.threshold(imgray,80,255,1) ### !! inverse binary !! ###
cv2.imshow('thresh image',thresh)
cv2.waitKey()

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

ctt=0
for cnt in contours:
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    print(len(approx)) ### !! fix indentation !! ###
    if len(approx)==5:
        print("pentagon")
        cv2.drawContours(im,[cnt],0,255,-1)
        ctt+=1
        break
    elif len(approx)==3:
        print("triangle")
        cv2.drawContours(im,[cnt],0,(0,255,0),-1)
        ctt+=1
        break
    elif len(approx)==4:
        print("square")
        cv2.drawContours(im,[cnt],0,(0,0,255),-1)
        ctt+=1
        break
    elif len(approx) >= 6:
        print("circle")
        cv2.drawContours(im,[cnt],0,(0,255,255),-1)
        ctt+=1
        break
    else:
        print("unknown")
        break

cv2.imshow('img',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Total no.= ",ctt)