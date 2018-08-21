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

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 2
path = './mnist.npz'
(x_train, y_train), (x_val, y_val), (x_test, y_test) = loadData(path)

itrain = (y_train == 0) + (y_train == 1)
ival = (y_val == 0) + (y_val == 1)
itest = (y_test == 0) + (y_test == 1)
#itest3 = (y_test == 3)
#test3 = x_test[itest3]
#ytest3=y_test[itest3]

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

model = load_model("model5.json")



worse = 0.999261#0.999802#0.999731#0.999841#0.999731

for i in range(28*28):
    xtest = copy.deepcopy(x_test)
    xtest[0][410] = 1.0
    xtest[0][437] = 1.0
    xtest[0][434] = 0.00392157
    if i==410 or i==437 or i==434:
       continue
    xtest[0][i] = 1 - xtest[0][i]
    yp = model.predict(xtest)
    # print(yp)
    yp = yp[0][1]
    #print(yp)
    if yp < worse:
        worse = yp
        print(i)
        print(xtest[0][i])

print('------worse-----')
print(worse)
 #   print(yp[0][1])
print(y_test[0])
