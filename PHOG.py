from commonfunctions import *
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage import filters
from skimage.feature import hog
import numpy as np
import cv2
import sys

def HOGDescriptor(img, numberOfBins = 8):
    Hist = np.zeros(numberOfBins)
    stepAngle = int(360 / numberOfBins)
    Gy = sobel_v(img)
    Gx = sobel_h(img)
    gradient_values = np.sqrt(np.square(Gy)+np.square(Gx))
    Gx[Gx == 0] = 0.000001
    gradient_angel = np.divide((np.arctan2(Gy, Gx) + np.pi) * 180., np.pi)
    for i in range(gradient_angel.shape[0]):
        for j in range(gradient_angel.shape[1]):  
            index = int((gradient_angel[i][j] / stepAngle))
            ratio = 1 - ((gradient_angel[i][j] - (index * stepAngle)) / stepAngle)
            if index == numberOfBins or index == 0:
                if Gx[i][j] >= 0:
                    index = int(180 / stepAngle)
                else:
                    index = 0
            Hist[index] += ratio * gradient_values[i][j]
            if index + 1 == numberOfBins:
                index = -1
            Hist[index + 1] += (1-ratio) * gradient_values[i][j]
    if np.sum(Hist) != 0:
        Hist = Hist / np.sum(Hist)
    return Hist

# Phog recieves a gray image
def PHOG(img,numberOfBins = 8,numberOfLevels = 3):
    x,y = img.shape
    IsXEven = (x %2 == 0)
    IsYEven = (y %2 == 0)
    median = np.median(img)
    sigma = 1/3
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    canny_img = canny(img , sigma = 1.7 , high_threshold = upper , low_threshold = lower)
    Hist = np.zeros((0))
    for i in range(numberOfLevels):
        width = int((x) / (2**i))
        if IsXEven == 0:
            width = width + 1
        length = int((y) / (2**i))
        if IsYEven == 0:
            length = length + 1
        # print(width,length,x,y)
        for w in range(0,x,width):
            for l in range (0,y,length):
                # print('w = ', w, 'l = ', l, 'width = ', width, 'length =', length, 'x = ', x, 'y= ',y)
                copy = np.copy(canny_img[w:w+width,l:l+length])
                Hist = np.append(Hist,HOGDescriptor(copy,numberOfBins))
    return Hist 