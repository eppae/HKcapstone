import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('D:/practice/contour/sample_images/sample_ppt/7.jpg')
from PIL import Image
coordinate=[]
import os
import matplotlib.pylab as plt
import json

def morphology(image):
    k = 0
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    th1 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)  # gaussian
    kernel = np.ones((7, 3), np.uint8)
    kernel2= np.ones((5, 5), np.uint8)
    erode = cv2.morphologyEx(th1, cv2.MORPH_ERODE, kernel, iterations=3)
    erode2 = cv2.morphologyEx(th1, cv2.MORPH_ERODE, kernel2, iterations=3)


    height, width = erode.shape

    for j in range(height):
        for i in range(width):
            if (erode[j, i] < 128):
                erode[j, i] = 0

            else:
                erode[j, i] = 255



    #titles = ['Original', 'Erode','Diliation', 'Opening', 'Close']
    #images = [image, erode,diliation,opening,close]

    #for i in range(5):
    #    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    #    plt.title(titles[i])
    #    plt.xticks([]), plt.yticks([])
    #plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)

    imFlood = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(imFlood, mask, (0, 0), 0)


    # Combine flood filled image with original objects
    imFlood[np.where(thresh == 0)] = 255


    # Invert output colors
    imFlood = ~imFlood

    contours, high =cv2.findContours(imFlood, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    for contour in contours:

        [x, y, w, h] = cv2.boundingRect(contour)

        if w * h < 900:
            continue
        k = k + 1


        coordinate.append([f'{k}.jpg', [x, y, w, h]])
        print(coordinate)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)



        cropped_image = erode2[y:y + h, x:x + w]
        original_image = imgray[y: y + h, x: x + w]
        resize = cv2.resize(cropped_image, dsize=(28,28), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'D:/practice/contour/sample_images/original_result/{k}.jpg',original_image)
        cv2.imwrite(f'D:/practice/contour/sample_images/result/{k}.jpg', resize)



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

                fname = f"{i}.jpg"
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

        img = cv2.imread(f'D:/practice/contour/sample_images/crop/{n}.jpg')
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
        if relative_density > 0.152:
            high_density = high_density + 1
        elif relative_density < -0.12:
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
    elif low_density == 1 and medium_density>=4: #and medium_density ==6 and low_density == 1:
        coordinate[f].append("원")
        print(coordinate[f])

def detect_figure():
    for f in range(len(os.listdir('D:/practice/contour/sample_images/result/'))):
        im = cv2.imread(f'D:/practice/contour/sample_images/result/{f + 1}.jpg')

        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(imgray, 80, 255, 1)  ### !! inverse binary !! ###

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ctt = 0
        for cnt in contours:
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            print(len(approx))  ### !! fix indentation !! ###
            if len(approx) == 5:
                print("pentagon")
                cv2.drawContours(im, [cnt], 0, 255, -1)
                coordinate[f].append("")
                coordinate[f].append("pentagon")
                break
            elif len(approx) == 3:
                print("triangle")
                cv2.drawContours(im, [cnt], 0, (0, 255, 0), -1)
                coordinate[f].append("")
                coordinate[f].append("triangle")
                break
            elif len(approx) == 4:
                print("square")
                cv2.drawContours(im, [cnt], 0, (0, 0, 255), -1)
                coordinate[f].append("")
                coordinate[f].append("square")
                break
            elif len(approx) >= 6:
                print("circle")
                cv2.drawContours(im, [cnt], 0, (0, 255, 255), -1)
                coordinate[f].append("")
                coordinate[f].append("circle")
                break
            else:
                print("unknown")
                break
        cv2.imwrite(f'D:/practice/contour/sample_images/approx/{f + 1}.jpg', im)


#전체 로직
morphology(img)
f=0
for f in range(len(os.listdir('D:/practice/contour/sample_images/result/'))):
    image_crop(f'D:/practice/contour/sample_images/result/{f+1}.jpg', 'D:/practice/contour/sample_images/crop/')
    print(f'D:/practice/contour/sample_images/result/{f+1}.jpg')
    #findpixel()
detect_figure()
dict_list = ['image_number','location','text','figure']
for k in range(len(os.listdir('D:/practice/contour/sample_images/result/'))):
            dictionary = dict(zip(dict_list, coordinate[k]))
            print(json.dump(dictionary, open('D:/practice/contour/json/'+ f'Gvidata{k+1}.json', 'w')))





