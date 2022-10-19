import tensorflow as tf
import numpy as np # linear algebr
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_iris
from sklearn.preprocessing import MultiLabelBinarizer

import cv2

df = pd.read_csv('/Users/yh/Desktop/src/archive/styles.csv',on_bad_lines='skip')

#print(df.head())

df = df.dropna()
df.nunique()
df.columns

cat_columns = ['gender', 'masterCategory', 'subCategory', 'articleType','baseColour', 'season', '    year', 'usage']

value_counts = df['articleType'].value_counts()

indexes = value_counts.index

values = value_counts.values

for i in range(len(value_counts)):

    if values[i] <1000:
        break

types_used = indexes[:i]
#print('Article types used: ',types_used)

value_counts = df['baseColour'].value_counts()

indexes = value_counts.index

values = value_counts.values

for i in range(len(value_counts)):

    if values[i] <1000:
        break

colours_used = indexes[:i]
#print('Base Colours used: ',colours_used)

df = df[df['articleType'].isin(types_used)]
df = df[df['baseColour'].isin(colours_used)]

#print(len(df))
tmp = []
invalid_ids = []

for name in df.id:

    try:
        image = cv2.imread('/Users/yh/Desktop/src/archive/images/'+str(name)+'.jpg')
        image = cv2.resize(image, (IX,IY) )
        image = img_to_array(image)
        tmp.append(image)        
    except: 
        # Images for certain ids are missing, so they are not added to the dataset  
        invalid_ids.append(name)
#print(invalid_ids)

labels = []

print(invalid_ids)

used_columns = ['subCategory','baseColour']

# getting labels for the columns used

for index, row in df.iterrows():

    if row['id'] in invalid_ids:
        continue

    tags = []

    for col in used_columns:
        tags.append(row[col])

    labels.append(tags)

labels = np.array(labels)

print(labels)






from tensorflow.keras.preprocessing.image import img_to_array
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

for pred in preds:
    vals = []
    for val in pred:
        if val > 0.5:
            vals.append(1)
        else:
            vals.append(0)
    pred_binarized.append(vals) 

#pred_binarized = np.array(pred_binarized)  
#print(pred_binarized.shape)




#pred_test_labels = mlb.inverse_transform(pred_binarized)





