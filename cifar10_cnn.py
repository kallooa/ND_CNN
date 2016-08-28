from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import pandas as pd
import random
import math
import cv2

def reshape_image_array(image_array):
    image_array = image_array[..., 0]
    image_array = np.dstack(image_array)
    image_array = np.rollaxis(image_array, -1)
    image_array = image_array[:, np.newaxis, :, :]
    return image_array

def augment_data(allimages, file_df):
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
    allimages = allimages[:, np.newaxis, :, :]
    del h_flipped, v_flipped, tmp_list, first_img_array_len, final_img_array_len
    return allimages, file_df

def load_data(csv_path, image_path):
     #location of csv with filename and dermoscopy status; columns: Filename, Status
    image_array_with_filenames = np.load(image_path)
    filename_array = image_array_with_filenames[..., 1]
    status_array = image_array_with_filenames[...,2]
    allimages = reshape_image_array(image_array_with_filenames)
    del image_array_with_filenames
    #list_of_images = get_file_list(csv_
    #file_df = pd.read_csv(csv_path)
    random.seed(1149)
    shuffled_indices = np.random.permutation(np.arange(len(filename_array)))
    shuffled_images = allimages[shuffled_indices]
    shuffled_df = filename_array[shuffled_indices]
    shuffled_status = status_array[shuffled_indices]
    shuffled_filedf = pd.DataFrame(shuffled_df, columns = ['Filename'])
    shuffled_filedf['Status'] = shuffled_status
    return shuffled_images, shuffled_filedf

def split_datasets(shuffled_images, shuffled_df, train_proportion):
    X_train = shuffled_images[0:math.floor(train_proportion*len(shuffled_df))]
    X_test = shuffled_images[(math.floor(train_proportion*len(shuffled_df))+1):len(shuffled_df)]
    df_train = shuffled_df[0:math.floor(train_proportion*len(shuffled_df))]
    df_test = shuffled_df[(math.floor(train_proportion*len(shuffled_df))+1):len(shuffled_df)]
    y_train = df_train['Status'].values
    y_test = df_test['Status'].values
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    #X_train = X_train[:, np.newaxis, :, :]
    #X_test = X_test[:, np.newaxis, :, :]
    del shuffled_images, shuffled_df

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, X_test, Y_train, Y_test, df_train, df_test    

def create_model(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols, num_conv_filters_layer2):
    model = Sequential()
    act = 'relu' #relu
    model.add(Convolution2D(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation(act))
    model.add(Convolution2D(num_conv_filters_layer1, num_conv_kernel_rows, num_conv_kernel_cols))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.7)) #0.25

    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols, border_mode='same'))
    model.add(Activation(act))
    model.add(Convolution2D(num_conv_filters_layer2, num_conv_kernel_rows, num_conv_kernel_cols))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.7)) #0.25

    model.add(Flatten())
    model.add(Dense(512)) #model.add(Dense(512))
    model.add(Dropout(0.8)) #0.5
    model.add(Activation(act))
    model.add(Dense(128)) #model.add(Dense(512)) #added
    model.add(Dropout(0.5)) #0.5 #added
    model.add(Activation(act)) #added
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model



# let's train the model using SGD + momentum (how original).
#sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
#  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

if __name__ == '__main__':
    #images_per_epoch = X_train.shape[0] #37*759 arbitrary, wanted ~12000 #X_train.shape[0]*2
    num_conv_filters_layer1 = 128
    num_conv_filters_layer2 = 32
    num_conv_kernel_rows, num_conv_kernel_cols = 3, 3
    batch_size = 32#48 #originally 32
    nb_classes = 2
    nb_epoch = 110
    data_augmentation = True
    test_proportion = 0.25
    train_proportion = 1 - test_proportion
    img_rows, img_cols = 400, 400 #32, 32
    img_channels = 1 #3
    learning_rate = 0.001 #0.001
    augmentdata1 = False
    n_folds = 10
    image_path = 'C:\\ML\\Project 1 - Dermoscopy\\Data\\results_allwithMEU.npy'
    csv_path = 'C:\\ML\\Project 1 - Dermoscopy\\Data\\AllwithMEU\\Dermoscopic_Status_withMEU.csv'
    print("Loading data...")
    Shuffled_images, Shuffled_filedf = load_data(csv_path, image_path)
    print("Data loaded.")
    #if augmentdata1:
        #Shuffled_images, Shuffled_filedf = augment_data(Shuffled_images[:,0,:,:], Shuffled_filedf)
        #print("Data augmented.")
    train_data, test_data, train_labels, test_labels, train_filenames, test_filenames = split_datasets(Shuffled_images, Shuffled_filedf, train_proportion)
    print("Datasets split.")
    model = create_model(32, 3, 3, 32)
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print("Model created.")
    #skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(train_data, train_labels,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(test_data, test_labels),
                  shuffle=True)
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
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(train_data)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(train_data, train_labels,
                            batch_size=batch_size),
                            samples_per_epoch=5024,
                            nb_epoch=nb_epoch,
                            validation_data=(test_data, test_labels))
    model.save('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn5.h5')
