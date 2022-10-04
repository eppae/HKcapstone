import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('D:/practice/contour/sample_images/sample_ppt/7.jpg')
from PIL import Image
coordinate=[]
import os
import scipy as sp
import matplotlib.pylab as plt
import seaborn as sns

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

   # titles = ['Original', 'Erode','Diliation', 'Opening', 'Close']
    #images = [image, erode,diliation,opening,close]

    #for i in range(5):
      #  plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
       # plt.title(titles[i])
       # plt.xticks([]), plt.yticks([])
    #plt.show()

    contours, high =cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        [x, y, w, h] = cv2.boundingRect(contour)



        if w < 30 and h < 30:
            continue
        k = k + 1


        coordinate.append([f'{k}.jpg', x, y, w, h])


        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cropped_image = erode[y:y + h, x:x + w]
        original_image = imgray[y: y + h, x: x + w]
        resize = cv2.resize(cropped_image, dsize=(28,28), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'D:/practice/contour/sample_images/original_result/{k}.jpg',original_image)
        cv2.imwrite(f'D:/practice/contour/sample_images/result/{k}.jpg', resize)

    print(coordinate)
   # cv2.imshow('captcha_result', img)
   # cv2.waitKey(0)
   # cv2.destroyALLWindows()

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



def image_crop(infilename, save_path):
        img = Image.open(infilename)
        (img_h, img_w) = img.size


        # crop 할 사이즈 : grid_w, grid_h
        grid_w = 9.3  # crop width
        grid_h = 9.3  # crop height
        range_w = (int)(img_w / grid_w)
        range_h = (int)(img_h / grid_h)


        i = 0

        for w in range(range_w):
            for h in range(range_h):
                bbox = (h * grid_h, w * grid_w, (h + 1) * (grid_h), (w + 1) * (grid_w))

                # 가로 세로 시작, 가로 세로 끝
                crop_img = img.crop(bbox)

                fname = "{}.jpg".format("{0:05d}".format(i))
                savename = save_path + fname
                crop_img.save(savename)

                i += 1

def findpixel():
    n = 0
    density_list = []
    high_density = 0
    medium_density = 0
    low_density = 0
    avg_density = 0
    relative_density = 0
    for n in range(9):
        img = cv2.imread(f'D:/practice/contour/sample_images/crop/0000{n}.jpg')
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hitrate = 0
        density = 0
        plt.show()
        height, width = img_gray.shape
        for j in range(height):
            for i in range(width):
                if (img_gray[j, i] < 128):
                    img_gray[j, i] = 0
                    hitrate = hitrate + 1
                else:
                    img_gray[j, i] = 255


        density = hitrate / (height * width)
        avg_density = avg_density + density
       # print('밀도: ', density)
        density_list.append(density)
    avg_density = avg_density / 9
    #print('평균밀도', avg_density)
    m = 0
    for m in range(9):
        relative_density = density_list[m] - avg_density

        print(relative_density)
        if relative_density > 0.12:
            high_density = high_density + 1
        elif relative_density < -0.08:
            low_density = low_density + 1
        else:
            medium_density = medium_density + 1
    print('high: ', high_density, 'medium: ', medium_density, 'low: ', low_density)
    if high_density == 4: #and medium_density== 4 and low_density == 1:
        coordinate[f].append("사각형")
        print(coordinate[f])
    elif high_density == 3: #and medium_density== 3 and low_density == 3:
        coordinate[f].append("삼각형")
        print(coordinate[f])
    elif high_density == 2: #and medium_density ==6 and low_density == 1:
        coordinate[f].append("원")
        print(coordinate[f])

#contour()
morphology(img)
#image_crop(f'D:/practice/contour/sample_images/result/7.jpg', 'D:/practice/contour/sample_images/crop/')
#findpixel_test()

f=0
for f in range(len(os.listdir('D:/practice/contour/sample_images/result/'))):
    image_crop(f'D:/practice/contour/sample_images/result/{f+1}.jpg', 'D:/practice/contour/sample_images/crop/')
    print(f'D:/practice/contour/sample_images/result/{f+1}.jpg')
    findpixel()


