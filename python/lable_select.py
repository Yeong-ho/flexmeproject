import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import PIL.Image as pilimag
import matplotlib.pyplot as plt
import sys
import json
import requests
from PIL import Image

df = pd.read_csv('/Users/yh/Desktop/src/archive/styles.csv',on_bad_lines='skip')

#print(df.head())

df = df.dropna()
df.nunique()
df.columns

cat_columns = ['gender', 'masterCategory', 'subCategory', 'articleType','baseColour', 'season', 'year', 'usage']


    
article_label_use = ['Tshirts', 'Shirts', 'Casual Shoes', 'Watches', 'Sports Shoes','Tops', 'Handbags', 'Heels','Flip Flops','Backpacks','Caps','Track Pants','Shorts']
print('Article types used: ',article_label_use)
color_label_use = ['Navy Blue', 'Blue', 'Silver', 'Black', 'Grey', 'Green', 'Purple', 'White','Brown','Red', 'Khaki', 'Orange', 'Yellow']
print('Base Colours used: ',color_label_use)


#print('Base Colours used: ',colours_used)

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

#print('invalid ids:')
#print(invalid_ids)

labels = []

used_columns = ['articleType','baseColour']


# getting labels for the columns used

for index, row in df.iterrows():

    if row['id'] in invalid_ids:
        continue

    tags = []

    for col in used_columns:
        tags.append(row[col])
#    print(tags)
    labels.append(tags)



#print(labels)
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

#image = cv2.imread('/Users/yh/Desktop/src/archive/images/2741.jpg')
image = cv2.imread(sys.argv[1])
image = cv2.resize(image,(80,60))
image = img_to_array(image)
target.append(image)


target = np.array(target,dtype="float")/255.0


new_model = tf.keras.models.load_model('../multi3.h5')

preds = new_model.predict(target)


#-------------------------------------------------------------------
'''


headers = {"content-type": "application/json"}
json_data = json.dumps({"signature_name": "serving_default", "instances": target[0:1].tolist()})

json_response = requests.post('http://54.180.99.107:8503/v1/models/multi_lable:predict',data=json_data, headers=headers)

print(json_response)

predictions=json.loads(json_response.text)#['predictions']
print(predictions)#np.argmax(predictions[0:]))


t_pred_binarized = []

for pred in predictions:
	vals = []
	for val in pred:
		if val>0.5:
			vals.append(1)
		else:
			vals.append(0)
	t_pred_binarized.append(vals)

t_pred_binarized = np.array(t_pred_binarized)
t_pred_test_labels = mlb.inverse_transform(t_pred_binarized)
print(t_pred_test_labels[0])


'''




#--------------------------------------------------------------------



pred_binarized = []




#print(preds)

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

plt.imshow(target[0])
plt.title('\n\n{}'.format(pred_test_labels[0]),fontdict={'size':18})
plt.show()




'''

# since the predictions of the model are sigmoid, we will first binarize them to 0 or 1
pred_binarized = []

for pred in preds:
    vals = []
    for val in pred:
        if val > 0.5:
            vals.append(1)
        else:
            vals.append(0)
    pred_binarized.append(vals) 

pred_binarized = np.array(pred_binarized)   


# we convert the output vectors to the predicted labels
true_test_labels = mlb.inverse_transform(testY)
pred_test_labels = mlb.inverse_transform(pred_binarized)

correct = 0
wrong = 0
print(list(pred_test_labels[0]))

# Evaluating the predictions of the model

for i in range(len(testY)):

    true_labels = list(true_test_labels[i])

    pred_labels = list(pred_test_labels[i])

    label1 = true_labels[0]
    label2 = true_labels[1]

    if label1 in pred_labels:
        correct+=1
    else:
        wrong+=1

    if label2 in pred_labels:
        correct+=1
    else:
        wrong+=1    



print('correct: ', correct)
print('missing/wrong: ', wrong)
print('Accuracy: ',correct/(correct+wrong))




for i in range(20):
    print('True labels: ',true_test_labels[i],' Predicted labels: ',pred_test_labels[i])

'''
#model.save("/Users/yh/Desktop/src/multi.h5")
