import cv2
import numpy as np
from matplotlib import pyplot as plt
import timeit
import time
from os import listdir
from os.path import isfile, join
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
#import theano
time1 = time.time()

mypath='C:\\ML\\Data\\Dermoscopy\\Extracted 1\\Dermoscopic\\'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
fname = onlyfiles[204]
print(fname)

file = mypath + fname
img0 = cv2.imread(file)

# converting to gray scale
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
kernel = np.ones((2,2),np.float32)/4
img = cv2.filter2D(img,-1,kernel)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_8U)
laplacian = cv2.filter2D(laplacian,-1,kernel)
laplacian = cv2.GaussianBlur(laplacian,(3,3),0)
laplacian = (255-laplacian)
ret, laplacian = cv2.threshold(laplacian,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

laplacian = cv2.resize(laplacian, (500,500))


plt.subplot(1,2,1),plt.imshow(gray,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,3),plt.imshow(img9,cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
time2 = time.time()
print('function took %0.3f ms' % ((time2-time1)*1000.0))
cv2.imwrite('C:\\ML\\laplacian.jpg', laplacian)
plt.show()

