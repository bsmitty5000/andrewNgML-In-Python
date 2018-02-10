# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 07:22:01 2018

@author: Batman
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidGradient(z):
    g = 1 / (1 + np.exp(-z))
    return g * (1 - g)

def costFunction(theta, X, y):
    m = y.size
    if m == 0:
        return np.inf
    
    h = sigmoid(X.dot(theta))
    hMinusOne = h-1
    hMinusOne[hMinusOne < 1e-315] = 1e-315
    J = (1/m)*(np.log(h).dot(-y)-np.log(hMinusOne).dot(1-y))
    
    return float(J)

def costFunctionReg(theta, X, y, learningRate):
    m = y.size
    if m == 0:
        return np.inf
    
    h = sigmoid(X.dot(theta))
    hMinusOne = h-1
    hMinusOne[hMinusOne < 1e-315] = 1e-315
    J = (1/m)*(np.log(h).dot(-y)-np.log(hMinusOne).dot(1-y)) + (learningRate / (2 * m)) * np.sum(theta**2)
    
    return float(J), h

def gradient(theta, X, y):
    m = y.size
    if m == 0:
        return np.inf
    
    error = sigmoid(X.dot(theta)) - y
    
    return X.T.dot(error) / m

def gradientReg(theta, X, y, learningRate):
    m = y.size
    if m == 0:
        return np.inf
    
    error = sigmoid(X.dot(theta)) - y
    
    return (X.T.dot(error) + learningRate*np.r_[np.array([0]), theta[1:]]) / m

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))