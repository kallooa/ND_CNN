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

def get_file_list(csv_path):
	#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] #read all files in mypath; uses os module
	files_df = pd.read_csv(csv_path) #read csv with pandas function
	onlyfiles = files_df['Filename'] #select only filename column
	return onlyfiles #return list of image files

def readImage(file, image_path):
	file = image_path + file
	img0 = cv2.imread(file) #read image
	print("Read ", file)
	#	return (img0, file)
	#def extract_scalebar(img0):
	#img0 = cv2.imread(file) #read image
	img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #convert image to grayscale
	kernel = np.ones((2,2), np.float32)/4 #create 2x2 kernel for smoothing
	img = cv2.filter2D(img, -1, kernel) #smooth image
	laplacian = cv2.Laplacian(img, cv2.CV_8U) #use laplacian filter ("edge detector") to find edges and lines
	laplacian = cv2.filter2D(laplacian, -1, kernel) #smooth image
	laplacian = cv2.GaussianBlur(laplacian, (3, 3), 0) #blur image
	laplacian = (255 - laplacian) #invert color/gray values
	ret, laplacian = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #use Otsu adaptive thresholding to extract scale bar
	laplacian = cv2.resize(laplacian, (400, 400)) #convert to 500x500 and return
	return (laplacian, file)
	
if __name__ == '__main__':
	image_path = 'D:\\ExportedImagesFromVctra1\\MRNS01to50\\' #location of images
	csv_path = 'D:\\ExportedImagesFromVctra1\\a0_mrns150.csv' #location of csv with filename and dermoscopy status; columns: Filename, Status
	#model = load_model('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn4.h5')
	list_of_images = get_file_list(csv_path)
	#list_of_images = list_of_images[0:1000]
	#result_array = np.array([99, 99])
	time1 = time.time()
	num_cores = multiprocessing.cpu_count()
	result_array = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(readImage)(i, image_path) for i in list_of_images)
	time2 = time.time()
	print('read function took %0.3f ms' % ((time2-time1)*1000))
	#result = create_result_array(list_of_images, image_path)
	#result_array = np.concatenate((result_array, result), axis = 0)
	#print(type(filearray))
	np.save('D:\\ExportedImagesFromVctra1\\results150', result_array)
