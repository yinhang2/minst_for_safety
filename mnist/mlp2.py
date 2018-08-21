from __future__ import print_function

import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Dropout
from keras.optimizers import RMSprop
from utils import loadData
from PIL import Image

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 2
path='./mnist.npz'
(x_train, y_train), (x_val, y_val), (x_test, y_test) = loadData(path)

itrain = (y_train==0) + (y_train==1)
ival = (y_val==0) + (y_val==1)
itest = (y_test==0) + (y_test==1)

x_train = x_train[itrain]
y_train = y_train[itrain]
x_val = x_val[ival]
y_val = y_val[ival]
x_test = x_test[itest]
y_test = y_test[itest]
y_test2 = y_test

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val))

model.save('model5.json')


score = model.evaluate(x_test, y_test, verbose=0)
print(y_test2)
#yp = model.predict(x_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

new_im = x_test[0]*255
new_im = Image.fromarray(new_im.reshape(28,28).astype(np.uint8))
new_im.show()
