import cv2
import numpy as np
from matplotlib import pyplot as plt
import timeit
import time
from os import listdir
from os.path import isfile, join
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
import pandas as pd
import theano
time1 = time.time()

image_path = 'C:\\ML\\Data\\Dermoscopy\\Extracted 1\\AllImages\\' #location of images
csv_path = 'C:\\ML\\Data\\Dermoscopy\\Extracted 1\\Extracted1_AllImages_Status.csv' #location of csv with filename and dermoscopy status; columns: Filename, Status

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
	laplacian = cv2.resize(laplacian, (500, 500)) #convert to 500x500 and return
	return laplacian

def create_image_array(filelist, images_path): #only needs to be done once for an image set and then ndarray can be saved
	for im in range(0, len(filelist)):
		fname = filelist[im] #filename
		file = images_path + fname #full file path
		image = extract_scalebar(file) #get image with scalebar extracted
		if im > 1: #im will be >1 most of the time and next elifs can be skipped
			image_array = np.concatenate((image_array, image[np.newaxis,...]), axis = 0) #concatenate arrays on axis 0 to allow for easy indexing
		elif im == 0:
			img0 = image #temporarily store first image; will then be added to 3d array
		elif im == 1:
			image_array = np.concatenate((img0[np.newaxis,...], image[np.newaxis,...]), axis = 0) #initialize 3d array
			del img0 #remove previous image variable to free memory
		if (im % 10) == 0:
			print('Read %f of %f', (im, len(filelist)))
	return image_array


list_of_images = get_file_list(csv_path)
im3darray = create_image_array(list_of_images, image_path)
np.save('C:\\ML\\Data\\Dermoscopy\\Extracted 1\\imgarray', im3darray)


#plt.subplot(1,2,1),plt.imshow(gray,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(1,2,2),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,3),plt.imshow(img9,cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
time2 = time.time()
print('function took %0.3f s' % ((time2-time1)))
#cv2.imwrite('C:\\ML\\laplacian.jpg', laplacian)
#plt.show()

