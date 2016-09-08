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
import os
import sklearn
from sklearn.metrics import auc, roc_curve
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

def form_hist_2D_array(image_array_with_filenames, result_array, wantStatus):
	hist_data = image_array_with_filenames[..., 2] 
	hist_data = np.dstack(hist_data)
	hist_data = hist_data.reshape(hist_data.shape[1:])
	hist_data = np.rollaxis(hist_data, -1)
	status_array = []
	if wantStatus:
		status_array = image_array_with_filenames[...,3]
	hist_data = np.insert(hist_data, 0, values=result_array, axis=1)
	return hist_data, status_array

def reshape_image_array(image_array):
    tup1 = image_array[:, 0]
    newarray = tup1[0]
    newarray = newarray[:, :, :, np.newaxis]
    for i in range(0, len(tup1)):
        newarray = np.concatenate((newarray, tup1[i][..., np.newaxis]), axis=3)
    newarray = np.rollaxis(newarray, -1)
    newarray = np.rollaxis(newarray, 3, 1)
    newarray = np.delete(newarray, [0], axis=0)
    newarray = newarray.astype('float16')
    print(newarray.shape)
    return newarray

def make_decision(processed_image_array, model):
	y_score = model.predict_proba(processed_image_array)
	return y_score

def output_metric_results(crosstabresults, score):
	TN = crosstabresults[0][0]
	FN = crosstabresults[0][1]
	FP = crosstabresults[1][0]
	TP = crosstabresults[1][1]
	print(crosstabresults)
	Sensitivity = TP/(TP+FN)
	Specificity = TN/(TN+FP)
	PPV = TP/(TP+FP)
	NPV = TN/(TN+FN)
	print("Accuracy: ", score)
	print("Sensitivity: ", Sensitivity)
	print("Specificity: ", Specificity)
	print("PPV: ", PPV)
	print("NPV: ", NPV)


if __name__ == '__main__':
	wantStatus = True
	image_array_with_filenames = np.load('D:\\Research\\color_extracted9.npy') #location of images
	pickle_loc = 'C:\\ML\\Project 1 - Dermoscopy\\Results\\RandomForestModel\\RF_model3.pkl'
	model = load_model('C:\\ML\\Project 1 - Dermoscopy\\Results\\nn6_color.h5')
	dermfolder = 'G:\\dermoscopic\\'

	filename_array = image_array_with_filenames[..., 1]
	image_array = reshape_image_array(image_array_with_filenames)
	print(image_array.shape)
	nnresults = make_decision(image_array, model)
	nnresults= nnresults[...,1] #probability of being 1 (dermoscopic)
	rfmodel = joblib.load(pickle_loc)
	hist_data, status_array = form_hist_2D_array(image_array_with_filenames, nnresults, wantStatus)
	preds = rfmodel.predict(hist_data)
	results = pd.DataFrame(filename_array, columns = ['filename'])
	results['prediction'] = preds
	score = accuracy_score(status_array, preds)
	crosstabresults = pd.crosstab(status_array, preds, rownames=['actual'], colnames=['preds:'])
	output_metric_results(crosstabresults, score)
	'''
	for index, file in enumerate(results['filename']):
	    if results.iloc[index, 1] == 1:
	        os.rename(file, dermfolder+os.path.split(file)[1])'''
	results.to_csv('C:\\ML\\Project 1 - Dermoscopy\\Results\\results_extracted9_2.csv')
	print("Results written!")
