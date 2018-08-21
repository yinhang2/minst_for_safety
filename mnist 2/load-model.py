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
import copy

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

model1 = load_model("modelb1.json")
model2 = load_model("modelb2.json")
model3 = load_model("modelb3.json")
model4 = load_model("modelb4.json")
model5 = load_model("modelb5.json")
model6 = load_model("modelb6.json")
model7 = load_model("modelb7.json")
model8 = load_model("modelb8.json")
model9 = load_model("modelb9.json")
model10 = load_model("modelb10.json")
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
#models = [model1, model2, model3, model4, model5]

yps = []
num1 = 20
num2 = 40
scase = 1
for scase in [1, 2, 3]:
    yps = []
    for model in models:
        if scase == 1:
            yp = model.predict(x_train)
        else:
            yp = model.predict(x_test)

        if scase == 3:
            yps.append(yp[itestB][num1:num2, 0])
        elif scase == 2:
            yps.append(yp[itestA][num1:num2, 0])
        else:
            yps.append(yp[itrainA][num1:num2, 0])
    # print(yps)
    yps = np.array(yps)
    #errors = np.zeros(num2 - num1)
   # for i in range(num2 - num1):
    #    tmp = yps#yps[:, i]
        #print(tmp.max(),tmp.min())
      #  errors[i] = tmp.max() - tmp.min()

    # print(errors)
    print(yps.max(), yps.min(), yps.max() - yps.min())
# yp = model1.predict(x_train)



# outs= yp[itrainA]
# print(outs[0:20, 0])
# print(y_train[itrainA][0:20])



# worse = 0.999261#0.999802#0.999731#0.999841#0.999731
#
# for i in range(28*28):
#     xtest = copy.deepcopy(x_test)
#     xtest[0][410] = 1.0
#     xtest[0][437] = 1.0
#     xtest[0][434] = 0.00392157
#     if i==410 or i==437 or i==434:
#        continue
#     xtest[0][i] = 1 - xtest[0][i]
#     yp = model.predict(xtest)
#     # print(yp)
#     yp = yp[0][1]
#     #print(yp)
#     if yp < worse:
#         worse = yp
#         print(i)
#         print(xtest[0][i])
#
# print('------worse-----')
# print(worse)
#  #   print(yp[0][1])
# print(y_test[0])
