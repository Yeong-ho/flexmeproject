import tensorflow as tf
import numpy as np # linear algebr
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_iris
from sklearn.preprocessing import MultiLabelBinarizer


























from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from tensorflow.keras.models import Sequential


data = []

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)



image = cv2.imread('/Users/yh/Desktop/src/archive/images/3165.jpg')
image = cv2.resize(image,(80,60))
image = img_to_array(image)
data.append(image)

data = np.array(data,dtype="float")/255.0


new_model = tf.keras.models.load_model('../multi.h5')

preds=new_model.predict(data)
#print(preds)


pred_binarized = []
pred_test_labels = mlb.inverse_transform(pred_binarized)
print(list(pred_test_labels[0]))
