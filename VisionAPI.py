# -*- coding: utf-8 -*-
import glob
import cv2
import numpy as np
import glob
import os, io
import json
import pickle

from google.cloud import vision

client = vision.ImageAnnotatorClient()

original_number = 0
data_root_path = 'C:/Users/minje/PycharmProjects/pytorchproject/VisionAPI/'
save_root_path = 'C:/Users/minje/PycharmProjects/pytorchproject/VisionAPI/images/'
json_root_path = 'C:/Users/minje/PycharmProjects/pytorchproject/VisionAPI/Gvidatas/'
temp = []
temp = os.listdir(json_root_path)
temp_len = len(os.listdir('C:/Users/minje/PycharmProjects/pytorchproject/VisionAPI/images/original_result/')) - 1
textlist = []

a = []


def detect_text(path):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        return ('\n"{}"'.format(text.description))
        # print(textbox)
        
        # print(save_textlist)
        # with open('save_textlist.pkl','wb')as f:
        #    pickle.dump(save_textlist,f)

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #           for vertex in text.bounding_poly.vertices])

        # print('bounds: {}'.format(','.join(vertices)))

        # with open("textdata.txt", 'a') as f:
        #    f.write(textbox)


images = glob.glob(save_root_path + '/all_images/*.jpg')

for j in images:
    file_name = os.path.join('C:/Users/minje/PycharmProjects/pytorchproject/VisionAPI/images/original_result', f'{j}')
    # print(file_name)
    detect_text(file_name)

    if detect_text(file_name) == None:
        a.append('')
        original_number += 1
    else:
        a.append(detect_text(file_name))



for k in range(2, len(temp)):  # 1번 이미지는 추출할 이미지 전체 파일을 잡아서 2번부터, 시작 이미지는 1부터 시작해서 temp -1
    # print(len(temp))
    # print(type(a))

    with open(json_root_path + f'Gvidata{k}.json', 'r', encoding='UTF8') as file:  # k번째 Gvidata.json을 json_data로읽어들인다
        json_data = json.load(file)
        json_data['text'] = a[k - 1]  # 해당 텍스트 값을 집어넣기

        print(json.dump(json_data, open(json_root_path + f'Gvidata{k}.json', 'w')))  # 수정한 json데이터를 Gvidatak.json파일로 저장
