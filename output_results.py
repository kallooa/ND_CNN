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

def get_file_list(csvpath):
	#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #read all files in mypath; uses os module
	files_df = pd.read_csv(csvpath) #read csv with pandas function
	onlyfiles = files_df['Filename'] #select only filename column
	return onlyfiles #return list of image files

def extract_scalebar(file):
	img0 = cv2.imread(file) #read image
	img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #convert image to grayscale
	kernel = np.ones((2,2), np.float32)/4 #create 2x2 kernel for smoothing
	img = cv2.filter2D(img, -1, kernel) #smooth image
	laplacian = cv2.Laplacian(img, cv2.CV_8U) #use laplacian filter ("edge detector") to find edges and lines
	laplacian = cv2.filter2D(laplacian, -1, kernel) #smooth image
	laplacian = cv2.GaussianBlur(laplacian, (3, 3), 0) #blur image
	laplacian = (255 - laplacian) #invert color/gray values
	ret, laplacian = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #use Otsu adaptive thresholding to extract scale bar
	laplacian = cv2.resize(laplacian, (400, 400)) #convert to 500x500 and return
	return laplacian
	
def make_decision(laplac_image):
  y_score = model.predict_proba(laplac_image)
  return y_score

def create_result_array(filelist, images_path): #only needs to be done once for an image set and then ndarray can be saved
	for im in range(0, len(filelist)):
		fname = filelist[im] #filename
		file = images_path + fname #full file path
		image = extract_scalebar(file) #get image with scalebar extracted
		result = make_decision(laplac_image)
		result_array = result
		if im > 1: #im will be >1 most of the time and next elifs can be skipped
			result_array = np.concatenate((result_array, result), axis = 0) #concatenate arrays on axis 0 to allow for easy indexing
		if (im % 10) == 0:
			print('Read %f of %f', (im, len(filelist)))
	return result_array

image_path = 'C:\\ML\\Data\\Dermoscopy\\Extracted 1\\AllImages\\' #location of images
csv_path = 'C:\\ML\\Data\\Dermoscopy\\Extracted 1\\Extracted1_AllImages_Status.csv' #location of csv with filename and dermoscopy status; columns: Filename, Status
model = load_model('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn4.h5')

list_of_images = get_file_list(csv_path)
result_array = create_result_array(list_of_images, image_path)
