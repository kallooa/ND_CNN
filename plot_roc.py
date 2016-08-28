from matplotlib import pyplot as plt
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import pandas as pd
import random
import math
import cv2
import sklearn
from sklearn.metrics import auc, roc_curve

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

csv_path = 'D:\\Research\\DermData\\Dermoscopic_Status.csv' #location of csv with filename and dermoscopy status; columns: Filename, Status

allimages = np.load('C:\\ML\\Project 1 - Dermoscopy\\Data\\derm15000.npy')
file_df = pd.read_csv(csv_path)
model = load_model('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn4.h5')

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

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_score = model.predict_proba(X_test)
a = Y_test[:, 0]
b = y_score[:, 0]
false_positive_rate, true_positive_rate, threshold = sklearn.metrics.roc_curve(a, b)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()