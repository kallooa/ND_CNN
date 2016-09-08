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

def get_file_list(csv_path):
	#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #read all files in mypath; uses os module
	files_df = pd.read_csv(csv_path) #read csv with pandas function
	onlyfiles = files_df['Filename'] #select only filename column
	status_df = files_df['Status']
	return onlyfiles, status_df #return list of image files

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

def readImage(num, file, image_path, wantStatus, status_df=[]):
	file = image_path + file
	img0 = cv2.imread(file) #read image
	if num % 100 == 0:
		print(num)
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
	img = cv2.resize(img, (400, 400)) #convert to 500x500 and return
	img = img.astype('float16')
	status = status_df[num]
	return img, file, color_hist, status
	
if __name__ == '__main__':
	time1 = time.time()
	wantStatus = True
	image_path = 'G:\\AllImages\\' #location of images
	csv_path = 'G:\\Dermoscopy77000.csv' #location of csv with filename and dermoscopy status; columns: Filename, Status
	#model = load_model('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn4.h5')
	list_of_images_master, status_df_master = get_file_list(csv_path)
	#list_of_images = list_of_images[0:1000]
	random.seed(1149)
	shuffled_indices = np.random.permutation(np.arange(len(list_of_images_master)))
	list_of_images_master = list_of_images_master[shuffled_indices]
	status_df_master = status_df_master[shuffled_indices]
	list_of_images_master = ((pd.DataFrame(list_of_images_master)['Filename']).reset_index())['Filename']
	status_df_master = ((pd.DataFrame(status_df_master)['Status']).reset_index())['Status']

	images_per_pickle = 5000
	num_loops = math.ceil(len(list_of_images_master)/images_per_pickle)
	num_cores = multiprocessing.cpu_count()
	for counter in range(0, num_loops):
		start_range = counter*images_per_pickle
		if counter != num_loops:
			end_range = (counter + 1)*images_per_pickle - 1
		else:
			end_range = len(list_of_images_master)
		list_of_images = list_of_images_master[start_range:end_range]
		print(len(list_of_images))
		status_df = status_df_master[start_range:end_range]
		list_of_images = ((pd.DataFrame(list_of_images)['Filename']).reset_index())['Filename']
		status_df = ((pd.DataFrame(status_df)['Status']).reset_index())['Status']
		temp_array = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(readImage)(i, image, image_path, wantStatus, status_df) for i, image in enumerate(list_of_images))
		
		img_array, file_array, hist_array, status_array = zip(*temp_array)
		img_array = np.array(img_array)
		img_array = np.rollaxis(img_array, 3, 1)
		#print(img_array.shape)
		np.save('C:\\ML\\dermoscopy77000images-' + str(counter), img_array)
		np.save('C:\\ML\\dermoscopy77000filenames-' + str(counter), file_array)
		np.save('C:\\ML\\dermoscopy77000status-' + str(counter), status_array)
		np.save('C:\\ML\\dermoscopy77000hist-' + str(counter), hist_array)

	time2 = time.time()
	print('read function took %0.3f s' % ((time2-time1)))
	print("Saving...")
	
	print("Model saved.")
