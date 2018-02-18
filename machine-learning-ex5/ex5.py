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

from logisticRegression import sigmoid, costFunctionReg, gradientReg
from neuralNetworks import forwardPropogate

from scipy.io import loadmat
from scipy.optimize import minimize

path = os.getcwd() + '\machine-learning-ex5\ex5'

data = loadmat(path + '\ex5data1.mat')

yTest = np.array(data['ytest']).flatten()
yCv = np.array(data['yval']).flatten()
y = np.array(data['y']).flatten()
x = np.array(data['X']).flatten()
xTest = np.array(data['Xtest']).flatten()
xCv = np.array(data['Xval']).flatten()
xTrain = np.c_[np.ones(y.shape[0]), x]

initialTheta = np.array([1.0, 1.0])
learningRate = 1.0

plt.scatter(x, y)
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of dam")

J2, h = costFunctionReg(initialTheta, xTrain, y, learningRate)
#
#initialTheta = np.zeros(X.shape[1])
#finalTheta = np.zeros((10, X.shape[1]))
#
#learningRate=0.1
#
#for K in range(1,11):
#    result = opt.fmin_tnc(func=costFunctionReg, x0=initialTheta, fprime=gradientReg, args=(X,(y == K)*1,learningRate))
#    finalTheta[K-1] = result[0]
#
#predictions = sigmoid(X.dot(finalTheta.T))
#predictions = np.argmax(predictions, axis=1)+1
#print('Training set accuracy: {} %'.format(np.mean(predictions == y.ravel())*100))
#
#thetaList = [theta1, theta2]

#pred = predict3Layer(theta1, theta2, X)
#[h,a,z] = forwardPropogate(thetaList, X)
#pred = np.argmax(h, axis=1)+1
#print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))