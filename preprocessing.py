import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('D:/practice/contour/sample_images/sample_ppt/4.jpg')
img_temp = cv2.imread('D:/practice/contour/sample_images/sample_ppt/4.jpg')


def preprocessing(image):
    k = 0
    bboxes = {}

    blur = cv2.GaussianBlur(image, (3, 3), 0)
    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    th1 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 10, 8)  # gaussian
    kernel = np.ones((4,4) ,np.uint8)
    erode = cv2.erode(th1, kernel, iterations=8)
    contours, high =cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        [x, y, w, h] = cv2.boundingRect(contour)
        if w < 30 and h < 40:
            continue

        k = k + 1
        bboxes[k]={x,y,w,h}
        print(bboxes)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cropped_image = img[y:y + h, x:x + w]
        resize = cv2.resize(cropped_image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'D:/practice/contour/sample_images/result/{k}.jpg', resize)


    cv2.imshow('captcha_result', img)
    cv2.waitKey(0)
    cv2.destroyALLWindows()



preprocessing(img)



