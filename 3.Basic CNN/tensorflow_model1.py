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

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

trainingstart=time.time()
model.fit(train_images, train_labels, epochs=5, batch_size=32)
trainingend=time.time()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
print("training_time:",trainingend-trainingstart)
print("Entire code execute time:",time.time()-start)