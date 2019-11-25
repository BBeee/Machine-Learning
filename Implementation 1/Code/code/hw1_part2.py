#!/usr/bin/env python3
from hw1_part0 import *
from math import sqrt,log
from matplotlib import pyplot as plt
import numpy as np
from numpy import dot

def learning_a(): # implementation of part 1.b with making graph
    data, y, featureIndex = readData()
    data_dev, y_dev, featureIndex_dev = readDevData()
    epsilon = 0.5
    featureLength = len(data[0])
    w = [0 for i in range(featureLength)] # the weight
    deltaW = [0 for i in range(featureLength)]  
    
    w_np = np.array(w)
    deltaW_np = np.array(deltaW)
    data_np = np.array(data)
    y_np = np.array(y)

    #data_dev_np = np.array(data_dev)
    #y_dev_np = np.array(y_dev)

    #learningRate = [-0.000057,-0.00001,-0.000001,-0.0000001] 
    learningRate = -0.000057
    #penalties = [0, 0.001, 0.01, 0.1, 1, 10, 100] # 0-6
    penalties = [0, 0.01, 10] # 0-6
    
    # ---
    # learning
    #while iterations <= lll*gap:
    for penalty in penalties[::-1]:
        iterations = 0
        w_np = np.array(w)
        l = float('inf')
        while l > epsilon:
            deltaW_np = np.array([0.0 for i in range(featureLength)])
            for i,_ in enumerate(data_np):
                deltaW_np += (dot(w_np,data_np[i]) - y_np[i]) * data_np[i]

            deltaW_np[1:] += w_np[1:] * penalty # except dummy 
            #deltaW_np += w_np * penalty # except dummy 
            
            l = getLength(deltaW_np)
            # update w
            w_np = w_np + learningRate * deltaW_np
            iterations += 1
        
        print("---")
        printSmall(w_np)
        print("iterations: ", iterations)
        print(w_np)
        print(getSSE(w_np,data,y))
        print(getSSE(w_np,data_dev,y_dev))
        print("LR: ", learningRate," lambda: ", penalty)
        print("---")
    
    
def dotProduct (w,x):
    s = 0
    for i,_ in enumerate(w):
        s += w[i] * x[i]
    return s

def mulVector(a,x): # a coefficient x a vector -> return a vector
    tx = x[:]
    for i,_ in enumerate(x):
        tx[i] *= a
    return tx

def addVector(x,y): # x + y return x
    for i,_ in enumerate(x):
        x[i] += y[i]
    return x

def getLength(v):
    s = 0
    for elem in v:
        s += elem*elem
    return sqrt(s)

def getSSE(w,data,y):
    s = 0
    for i,_ in enumerate(data):
        wx = dotProduct(w,data[i])
        wx = wx - y[i]
        s += wx * wx
    return s
'''
def getRegularizedSSE(w,data,y,l):
    s = 0
    for i,_ in enumerate(data):
        wx = dotProduct(w,data[i])
        wx = wx - y[i]
        s += wx * wx
    #print("s : ",s)
    s += l*getLength(w)
    #print("s + L2 : ", s)
    return s
'''
def printSmall(w):

    w_name = ['dummy','bedrooms','bathrooms', 'sqft_living', 'sqft_lot', 
              'floors', 'waterfront', 'view', 'condition', 'grade', 
              'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
              'zipcode zip', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'month', 'day', 'year']
    www = w[:]
    pw = [0 for i in range(22)]
    for i,_ in enumerate(www):
        pw[i] = (abs(www[i]),w_name[i])
    pw.sort()
    print(pw)
    

def printW(w):
    s = '['
    for elem in w:
        s = s + str(elem) + ','
    s = s[:-1]
    s = s + ']'
    print(s)

'''
def L2Regularization(r, w):
    x = w[:]
    for i,_ in enumerate(w):
        if i != 0:
            x[i] = x[i]*x[i]*r

    return x
'''
if __name__ == '__main__':
    learning_a()
    