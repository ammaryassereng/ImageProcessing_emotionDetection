from commonfunctions import *
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage import filters
from skimage.feature import hog
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
# from google.cloud import vision
import numpy as np
import cv2

def getCroppedFaces(data):
    imgListCropped = []
    for faceimg in data:
        colorB = np.copy(faceimg[:,:,0])
        colorR = np.copy(faceimg[:,:,2])
        faceimg[:,:,0] = colorR
        faceimg[:,:,2] = colorB
        # show_images([faceimg])

        faces = faceCascade.detectMultiScale(faceimg,minNeighbors=5)

        Cropped_faces = []
        for face in faces:
            Cropped_faces.append(np.copy(faceimg[face[1]:face[1]+face[3],face[0]:face[0]+face[2],:]))

        # show_images(Cropped_faces)
        imgListCropped.append(Cropped_faces)
    return imgListCropped

def detectMouth(mouth_cascade, img=np.zeros((1,1))):
    copyImg = np.copy(img)
    flag = 1
    start = 0
#     end = 1
    step = 0.1
    while flag:
        mouth = mouth_cascade.detectMultiScale(copyImg[copyImg.shape[0]//2:copyImg.shape[0],:,:], minSize=(25, 15),minNeighbors=int(start*copyImg.shape[0]))
        # print(len(mouth), start, step)
#         end += step
        if len(mouth) == 1:
            # print(start,mouth)
            # show_images([copyImg,np.copy(copyImg[copyImg.shape[0]//2 + mouth[0][1]:copyImg.shape[0]//2 + mouth[0][1] + mouth[0][3],mouth[0][0]:mouth[0][0] + mouth[0][2],:])])
            return np.copy(copyImg[copyImg.shape[0]//2 + mouth[0][1]:copyImg.shape[0]//2 + mouth[0][1] + mouth[0][3],mouth[0][0]:mouth[0][0] + mouth[0][2],:])
        elif len(mouth) == 0:
            if(start == 0):
                # print("I am here")
                return np.zeros((1,1))
            start -= step
            step = step /10
            flag = 1
            # print(start)
        else:
            flag = 1
        start += step