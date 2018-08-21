#!/usr/bin/python
import numpy as np
def loadData(path):
    f = np.load(path)
    X_train, Y_train = f['x_train'], f['y_train']
    x_train = X_train[0:55000]
    y_train = Y_train[0:55000]
    x_val = X_train[55000:]
    y_val = Y_train[55000:]
    
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    
    x_train = x_train.reshape(55000, 784).astype('float32')/255.0
    x_val = x_val.reshape(5000, 784).astype('float32')/255.0
    x_test = x_test.reshape(10000, 784).astype('float32')/255.0
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
