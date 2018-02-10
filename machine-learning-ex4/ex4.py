# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 07:09:41 2018

@author: Batman
"""
import sys
sys.path.insert(0, '../shared')

import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os

import neuralNetworks as nn

from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

path = os.getcwd() + '\ex4'

data = loadmat(path + '\ex4data1.mat')
weights = loadmat(path + '\ex4weights.mat')

numInputs = 400
numOutputs = 10
nnArch = [numInputs, 25, numOutputs] #first element = # of inputs; last element = # of outputs, between are arbitrary hidden layers
y = np.array(data['y'])
x = np.array(data['X'])
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=42)

encoder = OneHotEncoder(sparse=False)
oneHotYTrain = encoder.fit_transform(yTrain)

thetaFlat = nn.randInitWeightFlat(nnArch)

learningRate = 1.0

thetaArr = nn.rollupArrayList(thetaFlat, nnArch)

#a, z = nn.forwardPropogate(thetaArr, X)
#J, grad = nn.nnTop(thetaFlat, X, oneHotY, learningRate, nnArch)
#
fmin = minimize(fun=nn.nnTop,
                x0=thetaFlat, 
                args=(xTrain, oneHotYTrain, learningRate, nnArch), 
                method='TNC', 
                jac=True, 
                options={'maxiter': 1000, 'disp': True}
               )

a, z = nn.forwardPropogate(nn.rollupArrayList(fmin['x'], nnArch), xTrain)
pred = np.argmax(a[len(a)-1], axis=1)+1
print('Test set accuracy: {} %'.format(np.mean(pred == yTrain.ravel())*100))

a, z = nn.forwardPropogate(nn.rollupArrayList(fmin['x'], nnArch), xTest)
pred = np.argmax(a[len(a)-1], axis=1)+1
print('Test set accuracy: {} %'.format(np.mean(pred == yTest.ravel())*100))

