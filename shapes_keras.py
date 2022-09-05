from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory('C:/pythonProject/shape/circle', target_size=(256, 256), color_mode='grayscale', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset='training', interpolation='nearest')

test_generator = [0]

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32,
        kernel_size=(3,3),
        activation='relu',
        input_shape= (256, 256, 1)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categprical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=150
)

score=model.evaluate_generator(test_generator, steps=3)