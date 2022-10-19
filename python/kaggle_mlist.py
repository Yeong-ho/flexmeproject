import tensorflow as tf
import os
import matplotlib.pyplot as plt
import requests
import json
import numpy as np
import PIL.Image as pilimg
import pandas as pd
#df
df = pd.read_csv("/Users/yh/Desktop/src/archive/styles.csv", nrows=5000, error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.sample(frac=1).reset_index(drop=True)
df.head(10)

batch_size = 32


#x_train, x_test, y_train, y_test = np.load('./../archive/images/2246.jpg')

from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image	import ImageDataGenerator
image_generator = ImageDataGenerator(validation_split=0.2)

test_images= image_generator.flow_from_dataframe(
    directory="/Users/yh/Desktop/src/archive/images",
    target_size=(96,96),
    batch_size=batch_size
)


#print(test_images.shape)




#plt.imshow(test_img)
#plt.show()

'''
test_imglist = np.zeros([2,96,96,3])
test_imglist = np.expand_dims(test_img,axis=0)


#print(test_img)
plt.imshow(test_imglist[0])
plt.show()


#print(test_img)


data = json.dumps({"signature_name": "predict","instances": img})
#print('Data : {}...{}'.format(data[:50],data[len(data)-52:]))

headers = {"content-type": "application/json"}
json_response = requests.post('http://54.180.99.107:8502/v1/models/saved_model:predict',data=data, headers=headers)



print(json_response)


'''
