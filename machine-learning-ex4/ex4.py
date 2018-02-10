# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 07:09:41 2018

@author: Batman
"""
import sys
sys.path.insert(0, 'D:/machine_learning/ng_coursera')

import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os

from neuralNetworks import forwardPropogate, cost, costReg, sigmoidGradient, randInitWeightFlat, backPropogate, nnTop, flattenArrayList, rollupArrayList

from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

path = os.getcwd() + '\machine-learning-ex4\ex4'

data = loadmat(path + '\ex4data1.mat')
weights = loadmat(path + '\ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

numInputs = 400
numOutputs = 10
nnArch = [numInputs, 25, numOutputs] #first element = # of inputs; last element = # of outputs, between are arbitrary hidden layers
y = np.array(data['y']).flatten()
X = np.array(data['X'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

encoder = OneHotEncoder(sparse=False)
oneHotY_train = encoder.fit_transform(y)

thetaFlat = randInitWeightFlat(nnArch)

learningRate = 0.1

thetaArr = rollupArrayList(thetaFlat, nnArch)

#a, z = forwardPropogate(thetaArr, X)
#J, grad = nnTop(thetaFlat, X, oneHotY, learningRate, nnArch)

fmin = minimize(fun=nnTop,
                x0=thetaFlat, 
                args=(X, oneHotY_train, learningRate, nnArch), 
                method='TNC', 
                jac=True, 
                options={'maxiter': 1000, 'disp': True}
               )

a, z = forwardPropogate(rollupArrayList(fmin['x'], nnArch), X)
pred = np.argmax(a[len(a)-1], axis=1)+1
print('Test set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))

#a, z = forwardPropogate(rollupArrayList(fmin['x'], nnArch), X_train)
#pred = np.argmax(a[len(a)-1], axis=1)+1
#print('Train set accuracy: {} %'.format(np.mean(pred == y_train.ravel())*100))

