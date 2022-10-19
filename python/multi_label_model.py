import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import PIL.Image as pilimag
import matplotlib.pyplot as plt
import sys
import json
import requests
from PIL import Image

def app():
 df = pd.read_csv('/Users/yh/Desktop/src/archive/styles.csv',on_bad_lines='skip')

 

 df = df.dropna()
 df.nunique()
 df.columns


 article_label_use = ['Tshirts', 'Shirts', 'Casual Shoes', 'Watches', 'Sports Shoes','Tops', 'Handbags', 'Heels','Flip Flops','Backpacks','Caps','Track Pants','Shorts','Sweatshirts','Dress']
 color_label_use = ['Navy Blue', 'Blue', 'Silver', 'Black', 'Grey', 'Green', 'Purple', 'White','Brown','Red', 'Khaki', 'Orange', 'Yellow']




 df = df[df['articleType'].isin(article_label_use)]
 df = df[df['baseColour'].isin(color_label_use)]
 #number of examples we are left with

 data = []

 # Reading all the images and processing the data in them 

 from tensorflow.keras.preprocessing.image import img_to_array
 import cv2

 IX = 80
 IY = 60
 
 invalid_ids = []
 
 for name in df.id:
 
    try:
        image = cv2.imread('/Users/yh/Desktop/src/archive/images/'+str(name)+'.jpg')
        image = cv2.resize(image, (IX,IY) )
        image = img_to_array(image)
        data.append(image)
    except: 
        # Images for certain ids are missing, so they are not added to the dataset  
        invalid_ids.append(name)



 labels = []
 
 used_columns = ['articleType','baseColour']
 
 
 # getting labels for the columns used 
 
 for index, row in df.iterrows():
 
    if row['id'] in invalid_ids:
        continue

    tags = []

    for col in used_columns:
        tags.append(row[col])

    labels.append(tags)



 import numpy as np

 # converting data into numpy arrays

 data = np.array(data, dtype="float") / 255.0
 labels = np.array(labels)

 from sklearn.preprocessing import MultiLabelBinarizer
 mlb = MultiLabelBinarizer()
 labels = mlb.fit_transform(labels)

 import cv2
 from tensorflow.keras.preprocessing.image import img_to_array
 from tensorflow.keras.models import Sequential

 target =[]


 image = cv2.imread(sys.argv[1])
 image = cv2.resize(image,(80,60))
 image = img_to_array(image)
 target.append(image)


 target = np.array(target,dtype="float")/255.0


 new_model = tf.keras.models.load_model('/Users/yh/Desktop/src/t_multi1.h5')

 preds = new_model.predict(target)


 pred_binarized = []


 for pred in preds:
	 vals = []
	 for val in pred:
		 if val>0.5:
			 vals.append(1)
		 else:
			 vals.append(0)
	 pred_binarized.append(vals)

 pred_binarized = np.array(pred_binarized)
 pred_test_labels = mlb.inverse_transform(pred_binarized)

 print(pred_test_labels[0])




app()