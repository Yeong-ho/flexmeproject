import tensorflow as tf
import os
import matplotlib.pyplot as plt
import requests
import json
import numpy as np
import PIL.Image as pilimg
import cv2
from tensorflow.keras.preprocessing.image import img_to_array


#x_train, x_test, y_train, y_test = np.load('./../archive/images/2246.jpg')

from keras.models import load_model
from keras.preprocessing import image

image = []

img = cv2.imread('./../archive/images/1526.jpg')
img = cv2.resize(img,(96,96))


test_img = img_to_array(img)
image.append(test_img)
image = np.array(image,dtype="float")/255.0





#print(test_img)


print(image.shape)


'''
#print(test_img)
plt.imshow(test_imglist[0])
plt.show()


#print(test_img)

'''

data = json.dumps({"signature_name": "predict", "instances": image[0:1].tolist()})
#print('Data : {}...{}'.format(data[:50],data[len(data)-52:]))
headers = {"content-type": "application/json"}
json_response = requests.post('http://54.180.99.107:8502/v1/models/saved_model:predict',data=data, headers=headers)



print(json_response)

