from matplotlib import pyplot as plt
import keras
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import pandas as pd
import random
import math
#import cv2
import sklearn
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


def load_data(nnmodel_path, results, image_array_path):
	nnmodel = []#load_model(model_path)
	result_array = np.load(results)
	image_array_with_filenames = np.load(image_array_path) #location of images
	return nnmodel, result_array, image_array_with_filenames

def form_hist_2D_array(image_array_with_filenames, result_array):
	hist_data = image_array_with_filenames[..., 3] #...,3]
	hist_data = np.dstack(hist_data)
	hist_data = hist_data.reshape(hist_data.shape[1:])
	hist_data = np.rollaxis(hist_data, -1)
	status_array = image_array_with_filenames[...,2]
	#status_array = status_array.tolist()
	#status_array = np.asarray(status_array)
	#status_array.shape
	#np.concatenate((hist_data, result_array), axis =1)
	hist_data = np.insert(hist_data, 0, values=result_array, axis=1)
	#hist_data.shape
	#result_array.shape
	return hist_data, status_array

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

def train_RF(hist_data, status_array, train_prop):
	arr_len = len(hist_data)
	train_len = math.floor(arr_len*train_prop)
	model = RandomForestClassifier(n_estimators=25,max_features=None, n_jobs=-1, verbose=True)
	model.fit(hist_data[0:train_len], status_array[0:train_len])
	return model, arr_len, train_len

if __name__ == '__main__':
	train_prop = 0.75
	nnmodel_path = 'C:\\ML\\Project 1 - Dermoscopy\\Results\\nn5.h5'
	results = 'C:\\ML\\Project 1 - Dermoscopy\\Results\\nn5_27000_predictions.npy'
	image_array_path = 'C:\\ML\\Project 1 - Dermoscopy\\Data\\results_allwithMEU_chist.npy'
	pickle_loc = 'C:\\ML\\Project 1 - Dermoscopy\\Results\\RandomForestModel\\RF_model.pkl'

	nn, result_array, image_array_with_filenames = load_data(nnmodel_path, results, image_array_path)
	hist_data, status_array = form_hist_2D_array(image_array_with_filenames, result_array)
	model, arr_len, train_len = train_RF(hist_data, status_array, train_prop)
	preds = model.predict(hist_data[(train_len+1):arr_len])
	score = accuracy_score(status_array[(train_len+1):arr_len], preds)
	joblib.dump(model, pickle_loc) 
	crosstabresults = pd.crosstab(status_array[(train_len+1):arr_len], preds, rownames=['actual'], colnames=['preds:'])
	output_metric_results(crosstabresults, score)
