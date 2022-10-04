import cv2
import numpy as np
# load image
img = cv2.imread('D:/practice/contour/sample_images/sample_ppt/7.jpg')

blur = cv2.GaussianBlur(img, (5, 5), 0)
imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
th1 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)  # gaussian
kernel = np.ones((7, 3), np.uint8)

# threshold
thresh  = cv2.morphologyEx(th1, cv2.MORPH_ERODE, kernel, iterations=3)

# apply close morphology
#kernel = np.ones((5,5), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

imFlood = thresh.copy()
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(imFlood, mask, (0,0), 0)
cv2.imwrite('D:/practice/contour/im_1_floodfill.png',imFlood)

# Combine flood filled image with original objects
imFlood[np.where(thresh==0)]=255
cv2.imwrite('D:/practice/contour/im_2_floodfill.png',imFlood)

# Invert output colors
imFlood=~imFlood
cv2.imwrite('D:/practice/contour/im_3_floodfill.png',imFlood)

# Find objects and draw bounding box
contours, high = cv2.findContours(imFlood, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:

    [x, y, w, h] = cv2.boundingRect(contour)

    if w * h < 900:
        continue




    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

# Save final image
cv2.imwrite('D:/practice/contour/im_4_floodfill.png',img)