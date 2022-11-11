import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD


import numpy as np


model = Sequential()

model.add(Dense(8,activation = 'relu',input_shape=(4,)))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))


model.summary()