import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD


import numpy as np

x_data = np.array([1,2,3,4,5,6])
t_data = np.array([3,4,5,6,7,8])
'''
model = Sequential()
model.add(Flatten(input_shape=(1,) ))
model.add(Dense(1,activation = 'linear'))

model.compile(optimizer=SGD(),loss='mse')
model.summary()

print(model.weights)

hist = model.fit(x_data, t_data, epochs = 1000, batch_size = 2)

test_input_data = np.array([-3.1,3.0,3.5,15.0,20.1])
label_data = test_input_data + 2.0

result = model.predict(test_input_data)

print(result)
print(label_data.reshape(5,1))
print(model.weights)


_____________________________________________________
'''


input_ = Input(shape=(1,))
output_= Dense(1,activation = 'linear')(input_)
model = Model(inputs=input_,outputs=output_)

model.compile(optimizer=SGD(),loss='mse')
model.summary()


hist = model.fit(x_data,t_data,epochs =1000)

test_input_data = np.array([-3.1,3.0,3.5,15.0,20.1])
label_data = test_input_data+2.0

result = model.predict(test_input_data)

print(result)
print(label_data.reshape(5,1))


