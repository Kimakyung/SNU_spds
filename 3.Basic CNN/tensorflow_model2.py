from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import time
start=time.time()
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['accuracy'])

def to_one_hot(labels,dimension=10):
    results=np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i,label]=1.
    return results
trainingstart=time.time()
model.fit(train_images, to_one_hot(train_labels), epochs=5, batch_size=32)
trainingend=time.time()
test_loss, test_acc = model.evaluate(test_images, to_one_hot(test_labels), verbose=2)
print(test_acc)
print("training_time:",trainingend-trainingstart)
print("Entire code execute time:",time.time()-start)