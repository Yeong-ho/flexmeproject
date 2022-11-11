import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

print('x_train.shape = ', x_train.shape, 'x_test.shape =' ,x_test.shape)
print('t_train.shape = ', t_train.shape, ', t_test.shape = ',t_test.shape)

model = Sequential()
model.add(Conv2D(input_shape=(28,28,1),kernel_size=3,filters=32,strides=(1,1),activation='relu',use_bias=True, padding='SAME'))
model.add(MaxPool2D(pool_size=(2,2),padding='SAME'))
model.add(Conv2D(input_shape=(28,28,1),kernel_size=4,filters=64,strides=(1,1),activation='relu',use_bias=True, padding='SAME'))
model.add(MaxPool2D(pool_size=(2,2),padding='SAME'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))


model.compile(optimizer=Adam(lr=0.001), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
hist = model.fit(x_train.reshape(-1,28,28,1),t_train,batch_size=50,epochs=50,validation_split=0.2)
model.evaluate(x_test.reshape(-1,28,28,1),t_test)

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'],label='training loss')
plt.plot(hist.history['val_loss'],label='validation loss')
plt.legend(loc='best')
plt.show()

# index_label_prediction 코드 구현

ret_val = model.predict(x_test.reshape(-1,28,28,1))

print('type(ret_val) = ', type(ret_val), ', ret_val.shape = ', ret_val.shape)

# predict 실행 결과는 softmax 에 의한 확률 결과이므로, argmax 이용해서 10진수로 바꾸어 주어야함

predicted_val = np.argmax(ret_val, axis=1)  # 행 단위로 argmax 실행

prediction_label_comp_val = np.equal(predicted_val, t_test)

# list comprehension 이용하여 index_label_prediction 구현

index_label_prediction_list = [ [index, t_test[index], predicted_val[index] ]  for index, result in enumerate(prediction_label_comp_val)  if result == False ]

print(len(index_label_prediction_list))

print('Accuracy = ', 1 - ( len(index_label_prediction_list) / len(t_test) ))

# 임의의 false prediction 이미지 출력

false_data_index = np.random.randint(len(index_label_prediction_list))

#print('len of index_label_prediction_list => ', len(index_label_prediction_list), ', false_data_index => ', false_data_index)

mnist_index = index_label_prediction_list[false_data_index][0]
label = index_label_prediction_list[false_data_index][1]
prediction = index_label_prediction_list[false_data_index][2]

title_str = 'index = ' + str(mnist_index) + ' , label = ' + str(label) + ' , prediction = ' + str(prediction)

img = x_test[mnist_index].reshape(28,28)

plt.title(title_str)
plt.imshow(img, cmap='gray')
plt.show()