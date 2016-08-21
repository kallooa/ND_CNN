from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pandas as pd
import random
import math
import cv2

batch_size = 48 #originally 32
nb_classes = 2
nb_epoch = 200
data_augmentation = True
test_proportion = 0.2
train_proportion = 1 - test_proportion
img_rows, img_cols = 400, 400 #32, 32
img_channels = 1 #3
augment_data = False
learning_rate = 0.001

allimages = np.load('C:\\ML\\Project 1 - Dermoscopy\\Data\\Extracted 1\\imgarray.npy')
file_df = pd.read_csv('C:\\ML\\Project 1 - Dermoscopy\\Data\\Extracted 1\\Extracted1_AllImages_Status.csv')
if augment_data:
    first_img_array_len = allimages.shape[0]
    h_flipped = cv2.flip(allimages, 0)
    allimages = np.concatenate((allimages, h_flipped), axis=0)
    v_flipped = cv2.flip(allimages, 1)
    allimages = np.concatenate((allimages, v_flipped), axis=0)
    final_img_array_len = allimages.shape[0]
    augmentation_ratio = round(final_img_array_len/first_img_array_len)
    tmp_list = []
    for i in range(0, augmentation_ratio):
        tmp_list.append(file_df)
    file_df = pd.concat(tmp_list)
    file_df = file_df.reset_index(drop=True)
    del h_flipped, v_flipped, tmp_list, first_img_array_len, final_img_array_len

random.seed(1149)
#allimages_bk = allimages
shuffled_indices = np.random.permutation(np.arange(len(file_df)))
shuffled_images = allimages[shuffled_indices]
shuffled_df = file_df.iloc[shuffled_indices]

X_train = shuffled_images[0:math.floor(train_proportion*len(shuffled_df))]
X_test = shuffled_images[(math.floor(train_proportion*len(shuffled_df))+1):len(shuffled_df)]
df_train = shuffled_df[0:math.floor(train_proportion*len(shuffled_df))]
df_test = shuffled_df[(math.floor(train_proportion*len(shuffled_df))+1):len(shuffled_df)]
y_train = df_train['Status'].values
y_test = df_test['Status'].values
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
del shuffled_images, allimages
#Y_train = Y_train[:, np.newaxis, :, :]
#Y_test = Y_test[:, np.newaxis, :, :]
images_per_epoch = X_train.shape[0]*2
# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

num_conv_filters_layer1 = 32
num_conv_kernel_rows = 3
num_conv_kernel_cols = 3
num_conv_filters_layer2 = 32

model = Sequential()

model.add(Convolution2D(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=False)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=images_per_epoch,
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))
model.save('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn3.h5')
