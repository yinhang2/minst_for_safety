from __future__ import print_function

import keras
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
import random
random.seed(100)

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 2
path = './mnist.npz'
(x_train, y_train), (x_val, y_val), (x_test, y_test) = loadData(path)

itrainA = random.sample(range(len(x_train)), len(x_train) // 2)
ivalA = random.sample(range(len(x_val)), len(x_val) // 2)
itestA = random.sample(range(len(x_test)), len(x_test) // 2)

itrainB = list(set(range(len(x_train))) - set(itrainA))
ivalB = list(set(range(len(x_val))) - set(ivalA))
itestB = list(set(range(len(x_test))) - set(itestA))

# x_train = x_train[itrain]

y_train[itrainA] = 0
y_train[itrainB] = 1
# x_val = x_val[ival]
y_val[ivalA] = 0
y_val[ivalB] = 1
# x_test = x_test[itest]
y_test[itestA] = 0
y_test[itestB] = 1

for i in range(len(itrainA)):
    x_train[i][392:] = x_train[i][:392][::-1]
for i in range(len(ivalA)):
    x_train[i][392:] = x_train[i][:392][::-1]
for i in range(len(itestA)):
    x_train[i][392:] = x_train[i][:392][::-1]

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
# model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_val, y_val))

model.save('modelb10.json')


score = model.evaluate(x_test, y_test, verbose=0)
print(y_test2)
# yp = model.predict(x_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#
# new_im = x_test[0] * 255
# new_im = Image.fromarray(new_im.reshape(28, 28).astype(np.uint8))
# new_im.show()
