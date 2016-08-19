import cv2
import numpy as np
from matplotlib import pyplot as plt
import timeit

time1 = time.time()

file = 'D:\\test\\y.jpg'
# loading image
#img0 = cv2.imread('SanFrancisco.jpg',)
img0 = cv2.imread(file)

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)
img = cv2.resize(img, (500,500))
kernel = np.ones((2,2),np.float32)/4
img = cv2.filter2D(img,-1,kernel)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_8U)
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

laplacian = cv2.filter2D(laplacian,-1,kernel)
#sobelx = cv2.filter2D(sobelx,-1,kernel)
#sobely = cv2.filter2D(sobely,-1,kernel)

laplacian = (255-laplacian)
print(laplacian)
#th, laplacian = cv2.threshold(laplacian, 198, 255, cv2.THRESH_BINARY)

#laplacian = cv2.adaptiveThreshold(laplacian,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#               cv2.THRESH_BINARY,7,3)
ret, laplacian = cv2.threshold(laplacian,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
time2 = time.time()
print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)

plt.show()

