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
#import cv2
import sklearn
from sklearn.metrics import auc, roc_curve
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

def form_hist_2D_array(image_array_with_filenames, result_array, wantStatus):
	hist_data = image_array_with_filenames[..., 3] #...,3]
	hist_data = np.dstack(hist_data)
	hist_data = hist_data.reshape(hist_data.shape[1:])
	hist_data = np.rollaxis(hist_data, -1)
	status_array = []
	if wantStatus:
		status_array = image_array_with_filenames[...,2]
	#status_array = status_array.tolist()
	#status_array = np.asarray(status_array)
	#status_array.shape
	#np.concatenate((hist_data, result_array), axis =1)
	hist_data = np.insert(hist_data, 0, values=result_array, axis=1)
	#hist_data.shape
	#result_array.shape
	return hist_data, status_array

def reshape_image_array(image_array):
	image_array = image_array_with_filenames[..., 0]
	image_array = np.dstack(image_array)
	image_array = np.rollaxis(image_array, -1)
	image_array = image_array[:, np.newaxis, :, :]
	return image_array

def make_decision(processed_image_array, model):
	y_score = model.predict_proba(processed_image_array)
	#print(filename, ': ', y_score)
	return y_score


if __name__ == '__main__':
	wantStatus = False
	image_array_with_filenames = np.load('D:\\Research\\DermData\\Extracted Backup\\Extracted 9\\results_extracted9.npy') #location of images
	pickle_loc = 'C:\\ML\\Project 1 - Dermoscopy\\Results\\RandomForestModel\\RF_model2.pkl'
	model = load_model('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn5.h5')
	dermfoler = 'dermoscopic\\'
	filename_array = image_array_with_filenames[..., 1]
	image_array = reshape_image_array(image_array_with_filenames)
	#status_array = image_array_with_filenames[...,2]
	#del image_array_with_filenames
	#list_of_images = get_file_list(csv_path)
	nnresults = make_decision(image_array, model)
	nnresults= nnresults[...,1] #probability of being 1 (dermoscopic)
	rfmodel = joblib.load(pickle_loc)

	hist_data, status_array = form_hist_2D_array(image_array_with_filenames, nnresults, wantStatus)
	preds = model.predict(hist_data)
	#results = np.concatenate((filename_array, nnresults), axis=1)
	#np.save('C:\\ML\\Project 1 - Dermoscopy\\Data\\results1', nnresults)
	results = pd.DataFrame(filename_array, columns = ['filename'])
	results['prediction'] = preds
	#results['color_hist'] = image_array_with_filenames[..., 2] #...,2]
	
	results.to_csv('C:\\ML\\Project 1 - Dermoscopy\\Results\\results_extracted9.csv')
	print("Results written!")
