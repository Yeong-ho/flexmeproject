import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD


import numpy as np


loaded_data = np.loadtxt('./TF2_Example_1.csv', delimiter=',', dtype=np.float32)
x_data = loaded_data[ : , :-1]
t_data = loaded_data[ : , [-1]]

model = Sequential()
model.add(Dense(1, input_shape=(3, ),activation='linear')) 
model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')
model.summary()



#model 학습
from datetime import datetime

start_time = datetime.now()

hist = model.fit(x_data, t_data, epochs=100)

end_time = datetime.now()




#그래프 라이브러리
import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.legend(loc='best')

plt.show()

model.evaluate(x_test, t_test)