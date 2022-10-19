from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import requests
from PIL import Image

def show(idx, title):
  plt.figure(figsize=(12, 3))
  plt.imshow(test_images[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})
  plt.show()

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
test_images = test_images / 255.0
#print(test_images[0].shape)
# reshape for feeding into the model
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#print(test_images.shape)



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))
rando = random.randint(0,len(test_images)-1)
#show(rando, 'An Example Image: {}'.format(class_names[test_labels[rando]]))

#test_image=np.zeros([2,28,28,1])
#test_image[0]=test_images[rando]
plt.imshow(test_images[rando])
plt.show()



data = json.dumps({"signature_name": "serving_default", "instances": test_images[rando:rando+1].tolist()})
#print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))
print(data)

# send data using POST request and receive prediction result
headers = {"content-type": "application/json"}
json_response = requests.post('http://54.180.99.107:8501/v1/models/saved_model:predict', data=data, headers=headers)
print(json_response)
predictions = json.loads(json_response.text)['predictions']
# show first prediction result
if(np.argmax(predictions[0])!=test_labels[rando]):
	print('false : {}, {}'.format(rando,class_names[test_labels[rando]]))


show(rando, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[rando]], test_labels[rando]))



'''


# set model version and send data using POST request and receive prediction result
json_response = requests.post('http://54.180.99.107:8501/v1/models/saved_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
# show all prediction result
for i in range(0,3):
  show(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[test_labels[i]], test_labels[i]))

'''
