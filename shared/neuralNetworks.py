# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:55:32 2018

@author: Batman
"""
import numpy as np
import math

def sigmoid(z):
    """ Returns sigmoid of z
    """
    return 1 / (1 + np.exp(-z))

def sigmoidGradient(z):
    """ Returns gradient of sigmoid evaluated at z
    """
    g = 1 / (1 + np.exp(-z))
    return g * (1 - g)

def randInitWeightFlat(nnArch):
    """ Randomly initializes input weights and returns a flat array
    
        The random value for each weight is calculated as (rand * 2 * epsilon - epsilon)
        where epsilon is calculated at each layer as sqrt(6) / sqrt(nnArch[l+1] + nnArch[l])
        
        Args:
            param1 (List): A List of integers giving the number of units for each layer,
                            including input & output layers with any number of hidden
        
        Returns:
            array: A vector of all weights in the entire architecture
    """
    layers = len(nnArch)
    
    for l in range(layers-1):
        epsilon = math.sqrt(6) / math.sqrt(nnArch[l+1] + nnArch[l])
        if l==0:
            thetaArr = np.random.rand((nnArch[l+1] * (1+nnArch[l]))) * 2 * epsilon - epsilon
        else:
            thetaArr = np.concatenate((thetaArr, np.random.rand((nnArch[l+1] * (1+nnArch[l]))) * 2 * epsilon - epsilon))
        
    return thetaArr

def rollupArrayList(flatArray, arch):
    """ Pass in a flat input weight arrays and get a list of their matrices
    
        Args:
            param1 (array): A flat, concatenated array that contains all weights
            param2 (List):  A List of integers giving the number of units for each layer,
                            including input & output layers with any number of hidden
                            
        Returns:
            List: A list of matrices for each layer. Length will be len(param2) - 1
    """
    thetaList = []
    startIndex = 0
    for i in range(len(arch)-1):
        endIndex = startIndex + (arch[i] + 1) * arch[i+1]
        thetaList.append(np.reshape(flatArray[startIndex:endIndex], ((arch[i+1], arch[i] + 1))))
        startIndex = endIndex
        
    return thetaList

def flattenArrayList(arrayList):
    """ Pass in a list of matrices and get a single flat array
    
        Args:
            param1 (List):  A List of matrices for each layer
                            
        Returns:
            array: Flattened, concatenated array of all weights
    """
    flattenedArray = arrayList[0].flatten()
    for i in range(1, len(arrayList), 1):
        flattenedArray = np.concatenate((flattenedArray, arrayList[i].flatten()))
        
    return flattenedArray

def forwardPropogate(thetaArr, featureIn):
    """ Go through a single forward pass
    
        Args:
            param1 (List):  A List of weight matrices for each layer
            param2 (array):  The input vector
                            
        Returns:
            List: List of matrices containing activation values for each node in each layer
            List: List of matrices containing the input values for each node in each layer
    """
    layers = len(thetaArr)
    
    a = []
    z = []
    
    a.append(np.c_[np.ones((featureIn.shape[0],1)), featureIn])
    
    for l in range(layers-1):
        z.append(a[l].dot(thetaArr[l].T))
        a.append(np.c_[np.ones((a[l].shape[0],1)), sigmoid(z[l])])
    
    z.append(a[layers-1].dot(thetaArr[layers-1].T))
    a.append(sigmoid(z[layers-1]))
    
    return(a, z)

def cost(thetaArr, X, y):
    m = y.shape[0]
    if m == 0:
        return np.inf
    
    [a, z] = forwardPropogate(thetaArr, X)
    
    J = 0.0
    h = a[len(a)-1]
    for i in range(m):
        J = J + (np.log(h[i,:]).dot(-y[i,:])-np.log(1-h[i,:]).dot(1-y[i,:]))
    
    return float((1/m)*J)

def costReg(thetaArr, X, y, learningRate):
    """ Cost function with regularization
    
        Args:
            param1 (List):  A List of weight matrices for each layer
            param2 (array):  The input vector
            param3 (array): The expected output values
            param4 (float): Learning rate
                            
        Returns:
            float: Cost using squared error
    """
    m = y.shape[0]
    if m == 0:
        return np.inf
    
    [a, z] = forwardPropogate(thetaArr, X)
    
    J = 0.0
    h = a[len(a)-1]
    
    #add this to prevent log(0) in the case where h[] = 1
    hMinusOne = 1-h
    hMinusOne[hMinusOne < 1e-315] = 1e-315
    J = (np.log(h)*(-y)-np.log(hMinusOne)*(1-y))
        
    regFactor = 0.0
    for l in range(len(thetaArr)):
        regFactor += (learningRate / (2 * m)) * np.sum(thetaArr[l]**2)
    
    return float(np.sum(J)/m + regFactor)

def backPropogate(theta, X, oneHotY, learningRate):
    """ Go through a single backward pass using regularized cost function
    
        Args:
            param1 (List):  A List of weight matrices for each layer
            param2 (array):  The input vector
            param3 (array): The expected output values (one hot encoded if necessary)
            param4 (float): Learning rate
                            
        Returns:
            List: List of matrices containing cost at each node in each layer
    """
    m = oneHotY.shape[0]
    if m == 0:
        return np.inf
    
    smallDelta = []
    bigDelta = []
    
    [a, z] = forwardPropogate(theta, X)
    
    tempDelta = a[len(a)-1] - oneHotY
    smallDelta.insert(0, tempDelta) #smallDelta indexed in reverse, ie 0 = last layer
    
    for l in range(len(a)-2,0,-1): # count backwards from next-to-last layer until layer just after input layer
        curTheta = theta[l][:,1:]
        tempDelta = smallDelta[0].dot(curTheta)*(sigmoidGradient(z[l-1]))
        smallDelta.insert(0, tempDelta)
        
    for i in range(len(smallDelta)):
        reg = np.c_[np.zeros((theta[i].shape[0],1)), theta[i][:,1:]]
        bigDelta.append((smallDelta[i].T.dot(a[i]) + learningRate * reg) / m)
    
    return bigDelta
        
def nnTop(thetaFlat, X, y, learningRate, nnArch):
    """ Top level function to be passed into minimizing functions
    
        Args:
            param1 (array):  A flattened array of each weight for each node excluding bias nodes
            param2 (array):  The input vector
            param3 (array): The expected output values (one hot encoded if necessary)
            param4 (float): Learning rate
            param5 (List):  A List of integers giving the number of units for each layer,
                            including input & output layers with any number of hidden
                            
        Returns:
            float: Cost using squared error
            array: A flattened array of the gradient for each node in each layer
    """
    
    thetaList = rollupArrayList(thetaFlat, nnArch)
    
    bigDelta = backPropogate(thetaList, X, y, learningRate)
    J = costReg(thetaList, X, y, learningRate)
    #print(J)
    unrolledGrad = flattenArrayList(bigDelta)
        
    return J, unrolledGrad
