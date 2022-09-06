import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
tf.random.set_seed(seed=777)

##########데이터 로드

def load_data_with_encoded_y(data_dir, width, height):
    x_data = []
    y_data = []
    folder_names = os.listdir(data_dir)
    folder_names = [folder_name for folder_name in folder_names if not folder_name.startswith ('.')] #.DS_Store 제외
    folder_names = folder_names[:10]
    for folder_name in folder_names:
        file_names = os.listdir('{}/{}'.format(data_dir, folder_name))
        file_names = [file_name for file_name in file_names if not file_name.startswith ('.')] #.DS_Store 제외
        file_names = file_names[:10]
        for file_name in file_names:
            #print(file_name) #r_236_100.jpg
            file_path = '{}/{}/{}'.format(data_dir, folder_name, file_name)
            image = Image.open(file_path).convert('L')
            image = image.resize((width, height))
            image = np.array(image)
            #print(image.shape) #(150, 150)
            image = image.reshape((width, height, 1))
            #print(image.shape) #(150, 150, 1)
            x_data.append(image)
            if 'circles' in folder_name:
                y_data.append([1, 0, 0])
            elif 'squares' in folder_name:
                y_data.append([0, 1, 0])
            elif 'triangles' in folder_name:
                y_data.append([0, 0, 1])
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

width = 28
height = 28
color = 1 #색수
x_data, y_data = load_data_with_encoded_y('C:\pythonProject\shapes\shapes', width, height)

labels = ['circles', 'squares', 'triangles']

##########데이터 분석

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

x_train = x_train / 255
x_test = x_test / 255

##########모델 학습
##########모델 검증

input = Input(shape=(width, height, color))

filters = 3
kernel_size = 2
strides = 1
net = Conv2D(input_shape=(width, height, color), filters=filters, kernel_size=(kernel_size, kernel_size), strides=strides)(input)
net = Activation('relu')(net)
pool_size = 2
strides = 2
net = MaxPooling2D(pool_size=(pool_size, pool_size), strides=strides)(net)
net = Flatten()(net)
net = Dense(units=len(labels))(net)
net = Activation('softmax')(net)
model = Model(inputs=input, outputs=net)

model.summary()
'''
'''

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), callbacks=[ModelCheckpoint(filepath='model/figure_shape_classification_model.h5', save_best_only=True, verbose=1)])

##########모델 예측

file_path = tf.keras.utils.get_file('800px-Circle_-_black_simple.svg.png', 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Circle_-_black_simple.svg/800px-Circle_-_black_simple.svg.png')
image = Image.open(file_path).convert('L')
image = image.resize((width,height))
image = np.array(image)
print(image.shape) #(150, 150, 3)
image = image.reshape((width, height, color))
print(image.shape) #(28, 28, 1)
x_test = [image]
x_test = np.array(x_test)
x_test = x_test / 255

y_predict = model.predict(x_test)
print(y_predict) #[[0.20842288 0.41051054 0.38106653]]
print(y_predict.argmax(axis=1)) #[1]
print(y_predict.argmax(axis=1)[0]) #1
print(labels[y_predict.argmax(axis=1)[0]]) #B