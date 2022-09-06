import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('D:/practice/contour/sample_images/sample_ppt/4.jpg')
img_temp = cv2.imread('D:/practice/contour/sample_images/sample_ppt/4.jpg')
def morphology(image):
    k = 0

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    th1 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)  # gaussian
    kernel = np.ones((3,3) ,np.uint8)
    erode = cv2.erode(th1, kernel, iterations=8)
    diliation = cv2.dilate(th1, kernel, iterations=2)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    close = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)

    titles = ['Original', 'Erode','Diliation', 'Opening', 'Close']
    images = [image, erode,diliation,opening,close]

    for i in range(5):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    contours, high =cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        [x, y, w, h] = cv2.boundingRect(contour)



        if w < 30 and h < 30:
            print([x, y, w, h])
            continue
        k = k + 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cropped_image = img[y:y + h, x:x + w]
        resize = cv2.resize(cropped_image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'D:/practice/contour/sample_images/result/{k}.jpg', resize)

    cv2.imshow('captcha_result', img)
    cv2.waitKey(0)
    cv2.destroyALLWindows()

def size_down(image):
    global img
    down_width = 600
    down_height = 800
    down_points = (down_width, down_height)
    img = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)

def contour():
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(imgray, 127, 255, 0) #global
    th2 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10) #mean
    th3 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5) #gaussian

    contours, high =cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    titles = ['Original', 'Grayscale', 'Global', 'Mean', 'Gaussian']
    images = [img,imgray,th1, th2, th3, ]
    for i in range(5):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()




    #cv2.imshow('orginal', img)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    #cv2.imshow('grayscale', imgray)
    #cv2.imshow('global', th1)
    #cv2.imshow('mean', th2)
    cv2.imshow('contours', img)
    #cv2.imshow('gaussian', th3)

    #cv2.imshow('result: Global', img)

    cv2.waitKey(0)
    cv2.destroyALLWindows()

def gaussianfilter():
    sigma=1
    dst = cv2.GaussianBlur(img, (0, 0), sigma)

    desc = 'sigma = {}'.format(sigma)
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 255, 1, cv2.LINE_AA)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyALLWindows()




#contour()
morphology(img)



