import cv2 as cv
import numpy as np
import os
import msvcrt as m
import time

path = 'faces'
#faceDetector = cv.CascadeClassifier('myfacedetector.xml')
faceDetector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

files = os.listdir(path)

#img = cv.imread('IMG_20170709_195800959.jpg', cv.IMREAD_COLOR) #cam.read()
img = cv.imread('a1.bmp', cv.IMREAD_COLOR) #cam.read()
img = cv.bilateralFilter(img, 9,75,75)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = faceDetector.detectMultiScale(gray, 1.3,5)
for(x,y,w,h) in faces:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
'''
while True:
    cv.imshow("face", img)
    if(cv.waitKey(1)==ord('q')):
        cv.destroyAllWindows()
        continue

'''
i = 0
for file in files:
    #img = cv.imread('IMG_20170709_195800959.jpg', cv.IMREAD_COLOR) #cam.read()
    img = cv.imread(path+'/'+file, cv.IMREAD_COLOR) #cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.bilateralFilter(img, 9,75,75)
    faces = faceDetector.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
		i=i+1
		cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cv.imshow("face", img)
    if(cv.waitKey(1)==ord('q')):
        cv.destroyAllWindows()
        continue
print i