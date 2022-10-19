import numpy as np
import cv2
import matplotlib.pyplot as plt



imageUrl = '/Users/yh/Downloads/7.jpeg'#Desktop/img/test2.jpg
img = cv2.imread(imageUrl)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img1 = cv2.GaussianBlur(img1,ksize=(5,5),sigmaX=0)
ret, sep_thresh = cv2.threshold(img1, 80,255,cv2.THRESH_BINARY)


edged = cv2.Canny(img1,10,250)
cv2.imshow('',edged)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
closed = cv2.morphologyEx(edged,cv2.MORPH_CLOSE,kernel)
cv2.imshow('',closed)
cv2.waitKey(0)