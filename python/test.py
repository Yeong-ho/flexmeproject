import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import PIL.Image as pilimag
import matplotlib.pyplot as plt
import sys
import json
import os
import requests
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

path = '/Users/yh/Desktop/src/python/search_img/croll/'
file_list = os.listdir(path)

def im_rotate(img, degree):
    h, w = img.shape[:-1]

    crossLine = int(((w * h + h * w) ** 0.5))
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w

    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
    return result


for img in file_list:
        if img =='.DS_Store':
            continue

        image = cv2.imread(path+img)
        
        for i in range(3):
            
            image = im_rotate(image,i*90+90)
            
            save_img = img.split('.')[0]+'_'+str(i+1)+'.jpg'
            print(save_img)
            cv2.imwrite(path+save_img,image)