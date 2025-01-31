
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
print(train_generator)
print(validation_generator)

# Model Architecture
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

nb_epoch = 30
nb_train_samples = 2002
nb_validation_samples = 832

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

model.save_weights('basic_cnn_20_epochs.h5')

print("Compute Loss and accuracy: ")
print(model.evaluate_generator(validation_generator, nb_validation_samples))
print()

# VGG16 Model Architecture
# model_vgg = Sequential()
# model_vgg.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height,3)))
# model_vgg.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
# model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
# model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
# model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
# model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
# model_vgg.add(ZeroPadding2D((1, 1)))
# model_vgg.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
# model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))
