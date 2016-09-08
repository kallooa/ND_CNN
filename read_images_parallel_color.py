from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import cv2
import sklearn
from sklearn.metrics import auc, roc_curve
import joblib
import multiprocessing
import time
from imutils import paths
import imutils
import os
from os import listdir
from os.path import isfile

def get_file_list(mypath):
	inc_extensions = ['jpg', 'png']
	onlyfiles = [f for f in listdir(mypath) if any(f.endswith(ext) for ext in inc_extensions)]#isfile((mypath + f))] #read all files in mypath; uses os module
	#files_df = pd.read_csv(csv_path) #read csv with pandas function
	#onlyfiles = files_df['Filename'] #select only filename column
	#status_df = files_df['Status']
	return onlyfiles#return list of image files

def extract_color_histogram(image, bins=(8*3, 8*3, 8*3)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()

def readImage(num, file, image_path, csv_path):
	file = image_path + file
	img0 = cv2.imread(file) #read image
	print("Read ", file)
	color_hist = extract_color_histogram(img0)
	img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB) #convert image to grayscale
	#kernel = np.ones((2,2), np.float32)/4 #create 2x2 kernel for smoothing
	#img = cv2.filter2D(img, -1, kernel) #smooth image
	#laplacian = cv2.Laplacian(img, cv2.CV_8U) #use laplacian filter ("edge detector") to find edges and lines
	#laplacian = cv2.filter2D(laplacian, -1, kernel) #smooth image
	#laplacian = cv2.GaussianBlur(laplacian, (3, 3), 0) #blur image
	#laplacian = (255 - laplacian) #invert color/gray values
	#laplacian = cv2.GaussianBlur(laplacian, (7, 7), 0) #blur image
	#ret, laplacian = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #use Otsu adaptive thresholding to extract scale bar
	img = cv2.resize(img, (200, 200)) #convert to 500x500 and return
	#status = status_df[num]
	return (img, file, color_hist, status)
	
if __name__ == '__main__':
	image_path = 'G:\\AllImages\\' #location of images
	csv_path = 'C:\\Dermoscopy77000.csv' #location of csv with filename and dermoscopy status; columns: Filename, Status
	#model = load_model('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn4.h5')
	list_of_images = get_file_list(image_path, csv_path)
	#list_of_images = list_of_images[0:1000]
	time1 = time.time()
	num_cores = multiprocessing.cpu_count()
	results_array = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(readImage)(i, image, image_path) for i, image in enumerate(list_of_images))
	time2 = time.time()
	print('read function took %0.3f s' % ((time2-time1)*1))
	print("Saving...")
	np.save('C:\\ML\\Project 1 - Dermoscopy\\dermoscopy_77000', results_array)
	print("Model saved.")
