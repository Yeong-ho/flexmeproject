from tkinter.tix import Meter
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np

try:

    loaded_data = np.loadtxt('./diabetes.csv',delimiter= ',')

    x_data = loaded_data[ : , 0:-1]
    t_data =  loaded_data[:,[-1]]












    print("x_data.shape = ", x_data.shape)
    print("t_data.shape = ", t_data.shape)

except Exception as err:
    print(str(err))


#VAL TEST(test 
TEST_SPLIT_RATIO = 0.1

test_nums = int(TEST_SPLIT_RATIO*len(x_data))

print('test_nums = ', test_nums)

x_test = x_data[:test_nums]
t_test = t_data[:test_nums]

x_data = x_data[test_nums:]
t_data = t_data[test_nums:]

print(x_data.shape, t_data.shape)
print(x_test.shape, t_test.shape)




#VAL_NUM (validation)
SPLIT_RATIO = 0.1
val_num = int(SPLIT_RATIO*len(x_data))

x_val = x_data[:val_num]
t_val = t_data[:val_num]

x_data = x_data[val_num:]
t_data = t_data[val_num:]








model = Sequential()
model.add(Dense(t_data.shape[1],input_shape=(x_data.shape[1],),activation= 'sigmoid'))


model.compile(optimizer = SGD(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

hist = model.fit(x_data,t_data,epochs=500, validation_data=(x_val, t_val),verbose=2)

model.evaluate(x_data,t_data)

import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')
plt.legend(loc='best')

plt.show()

