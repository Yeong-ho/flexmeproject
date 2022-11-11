from contextlib import AsyncExitStack
from pkg_resources import add_activation_listener
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, t_train),(x_test,t_test) = mnist.load_data()


print('x_train.shape = ',x_train.shape,' t_train.shape = ',t_train.shape)
print('x_test.shape = ',x_test.shape,' t_test.shape',t_test.shape)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))

x_train = x_train /255.0
x_test = x_test / 255.0

#t_train = to_categorical(t_train,10)
#t_test = to_categorical(t_test,10)


model = Sequential()
model.add(Flatten(input_shape=(28, 28,1)))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))

from tensorflow.keras.optimizers import SGD
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#model.compile(optimizer=SGD(),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

hist = model.fit(x_train,t_train,epochs=30,validation_split=0.2)
model.evaluate(x_test,t_test)

import matplotlib.pyplot as plt
plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'],label='training loss')
plt.plot(hist.history['val_loss'],label='validation loss')
plt.legend(loc='best')
plt.show()


plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'],label='training loss')
plt.plot(hist.history['val_accuracy'],label='validation loss')
plt.legend(loc='best')
plt.show()

pred = model.predict(x_test)
#print(np.argmax(pred[:5], axis=1))
#print(t_test[:5],np.argmax(t_test[:5],axis=1))
np.random.choice()
