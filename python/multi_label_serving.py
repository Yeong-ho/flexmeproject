import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from tensorflow import keras
df = pd.read_csv('/Users/yh/Desktop/src/archive/styles.csv',on_bad_lines='skip')

#print(df.head())

df = df.dropna()
df.nunique()
df.columns

cat_columns = ['gender', 'masterCategory', 'subCategory', 'articleType','baseColour', 'season', 'year', 'usage']



value_counts = df['articleType'].value_counts()

indexes = value_counts.index

values = value_counts.values

for i in range(len(value_counts)):

    if values[i] <50:
        break

    


types_used = indexes[:i]
#print('Article types used: ',types_used)
article_label_use = ['Tshirts', 'Shirts', 'Casual Shoes', 'Watches', 'Sports Shoes','Tops', 'Handbags', 'Heels','Flip Flops','Backpacks','Caps','Track Pants','Shorts','Sweatshirts','Dress']
print('Article types used: ',article_label_use)
color_label_use = ['Navy Blue', 'Blue', 'Silver', 'Black', 'Grey', 'Green', 'Purple', 'White','Brown','Red', 'Khaki', 'Orange', 'Yellow']
print('Base Colours used: ',color_label_use)

value_counts = df['baseColour'].value_counts()

indexes = value_counts.index

values = value_counts.values

for i in range(len(value_counts)):

    if values[i] <1000:
        break

colours_used = indexes[:i]


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




#print(labels)

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

inputShape = (IY, IX, 3)

# A very simple sequential model is used since the images are very low resolution and the categories are fiarly distinct
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten()) 

model.add(Dense(128))
model.add(Activation('sigmoid'))


out = len(mlb.classes_)

model.add(Dense(out))
model.add(Activation('sigmoid')) # activation function for the final layer has to be sigmoid, since mutiple output labels can have value 1
                    
model.compile(loss='binary_crossentropy', # loss function has to be binary_crossentropy, it is calculated seperately for each of the outputs
              optimizer='adam',
              metrics=['mse'])
model.summary()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def custom_loss1(input):
    def loss1(y_true, y_pred):
        return tf.norm(input - y_pred) # use your custom loss 1
    return loss1

def custom_loss2(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) # use your custom loss 2

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
out = len(mlb.classes_)
model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet',input_tensor=None,input_shape=inputShape,
    pooling='same',classes=out
)

model.summary()
input_layer=Input(shape=inputShape)
model.compile(optimizer='adadelta', loss='binary_crossentropy') 
'''
#_____________________________________________________________________________________



from sklearn.model_selection import train_test_split

# splitting data into testing and training set 

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.1, random_state=42)
batch = 32
E = 50

#training the model 
model.fit(x=trainX,y=trainY,epochs=E, verbose=1)
from sklearn.metrics import classification_report
print(testX.shape)
preds = model.predict(testX[3:4])
print(preds,'\n',testX[3:4].shape)

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

#학습됬는지 확인을 위한 테스트 
print(list(pred_test_labels[0]))


model.save("/Users/yh/Desktop/src/t_multi1.h5")

# Evaluating the predictions of the model
'''
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
'''
def make_directory(target_path):
  if not os.path.exists(target_path):
    os.mkdir(target_path)
    print('Directory ', target_path, ' Created ')
  else:
    print('Directory ', target_path, ' already exists')


SAVED_MODEL_PATH = '/Users/yh/multi_lable_model'
make_directory(SAVED_MODEL_PATH)
MODEL_DIR = SAVED_MODEL_PATH

version = 1
export_path = os.path.join(MODEL_DIR, str(version))

tf.keras.models.save_model(
  model,
  export_path,
  overwrite=True,
  include_optimizer=True,
  save_format=None,
  signatures=None,
  options=None
)
'''



