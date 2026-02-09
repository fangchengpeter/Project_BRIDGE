# read and preprocess data
import random
import numpy as np
import pickle
import os
import pandas as pd
import math
import timeit
import matplotlib.pyplot as plt
import platform
from subprocess import check_output
from MNIST_read import mnist_read

class dis_data:
    def __init__(self, data, label, nodes, shuffle=False, index=None, one_hot=False):
        self.size = len(data)
        self.nodes = nodes
        self.all_data = data
        self.all_label = label
        
        if index:
            self.index = index
        else:
            self.index = list(range(self.size))
        if shuffle:
            self.shuffle()
        self.dist_data, self.dist_label = self.distribute(nodes)
        if one_hot:
            new_label = []
            for node in self.dist_label:
                new_label.append(_one_hot(node))
            self.dist_label = new_label
        
    def shuffle(self):
        random.shuffle(self.index)
        new_data = []
        new_label = []
        for ind in self.index:
            new_data.append(self.all_data[ind])
            new_label.append(self.all_label[ind])
        self.all_data = new_data
        self.all_label = new_label
        return new_data, new_label
    
    def distribute(self, nodes):  #Evenly distributed the data into nodes
        remainder = self.size % nodes
        frac = int(self.size/nodes)
        dist_data = []
        dist_label = []
        for n in range(nodes):
            if n == 0:
                dist_data.append(self.all_data[n * frac : (n + 1) * frac + remainder])
                dist_label.append(self.all_label[n * frac : (n + 1) * frac + remainder])
            else:                
                dist_data.append(self.all_data[n * frac : (n + 1) * frac])
                dist_label.append(self.all_label[n * frac : (n + 1) * frac])
        return dist_data, dist_label
    def next_batch(self, node, size):
        l = len(self.dist_data[node])
        sample = []
        label = []
        for _ in range(size):
            index = random.randint(0, l-1)
            sample.append(self.dist_data[node][index])
            label.append(self.dist_label[node][index])
        return sample, label

def data_prep(dataset, nodes, size=0, one_hot=True):
    if dataset == 'MNIST':
        train_data, train_label, test_data, test_label = mnist_read()
        if one_hot:
            test_label = _one_hot(test_label)
    elif dataset == 'CIFAR':
        #with open('cifar_dataset.pickle', 'rb') as handle:
        (train_data, train_label, test_data, test_label) = get_CIFAR10_data()
        train_label = _one_hot(train_label)
        test_label = _one_hot(test_label)
        
    else:
        raise NameError("Cannot find %s dataset") % (dataset)
    
    if size:
        train_data = train_data[:size]
        train_label = train_label[:size]
        
    full_data = dis_data(train_data, train_label, nodes, shuffle = False, one_hot=one_hot)
    return full_data, test_data, test_label
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = './data/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    #mask = range(num_training, num_training + num_validation)
    #X_val = X_train[mask]
    #y_val = y_train[mask]
    #mask = range(num_training)
    #X_train = X_train[mask]
    #y_train = y_train[mask]
    #mask = range(num_test)
    #X_test = X_test[mask]
    #y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test
def _one_hot(label):
    l_oh = []
    for i in label:
        new_l = [0] * 10
        #print(new_l[i])
        new_l[i] = 1
        l_oh.append(new_l)
    return l_oh
#def _one_hot(vec, vals=10):
    #n = len(vec)
    #out = np.zeros((n, vals))
    #out[range(n), vec] = 1
    #return out
