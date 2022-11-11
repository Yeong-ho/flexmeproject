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
model.add(Flatten())
model.add(Dense(10,activation='softmax'))


model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
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