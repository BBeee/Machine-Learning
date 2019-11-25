#!/usr/bin/env python3
from hw1_part3_readData import *
from math import sqrt,log
from matplotlib import pyplot as plt
import numpy as np

# lambda = 0

def learning_b(): # implementation of part 1.b with making graph
    data, y, featureIndex = readDataWithoutNormalization()
    data_dev, y_dev, featureIndex_dev = readDevDataWithoutNormalization()
    epsilon = 0.5
    featureLength = len(data[0])
    w = [0 for i in range(featureLength)] # the weight
    deltaW = [0 for i in range(featureLength)]  
    


    #data_dev_np = np.array(data_dev)
    #y_dev_np = np.array(y_dev)

    #learningRate = [-0.000057,-0.00001,-0.000001,-0.0000001] 
    learningRate =  [-1, 0, -0.0001, -0.0000001, -0.0000000001, -0.0000000000000001] #1, 0, 10^−3, 10^−6, 10^−9, 10^−15

    lll = 1000
    gap = 10
    x_G = [gap*i for i in range(lll)]
    y_G = [0 for i in range(lll)]
    y_G_dev = [0*i for i in range(lll)]
    
    
    # calculate deltaW  
    for i,_ in enumerate(data):
        addVector(deltaW, mulVector((dotProduct(w, data[i])-y[i]), data[i]))
    l = getLength(deltaW)
    
    # ---
    # learning
    
    for elem in learningRate:
        print("LR: ", elem)
        iterations = 0
        w_np = np.array(w)
        deltaW_np = np.array(deltaW)
        data_np = np.array(data)
        y_np = np.array(y)
        while iterations <= lll*gap:  #Fix the number of iterations; 10000
            print(iterations,"th iteration : |W| =  ",end="")
            deltaW_np = np.array([0.0 for i in range(featureLength)])
            for i,_ in enumerate(data_np):
                deltaW_np += (w_np.dot(data_np[i]) - y_np[i]) * data_np[i]
            
            l = getLength(deltaW_np)
            print(l)
            if iterations in x_G:
                pos = x_G.index(iterations)
                y_G[pos] = log(getSSE(w_np,data,y)/len(data))
                y_G_dev[pos] = log(getSSE(w_np,data_dev,y_dev)/len(data_dev))
        
            # update w
            w_np = w_np + elem * deltaW_np
            iterations += 1
        
    
        #print("w.length", getLength(w_np))
        plt.plot(x_G,y_G)
        plt.plot(x_G,y_G_dev)
        print(y_G)
        print(y_G_dev)
        plt.show()
        file = open('output'+str(learningRate.index(elem))+'.txt','w')
        file.writelines(str(y_G))
        file.writelines(str(y_G_dev))
        file.close()

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
def test(w):
    '''
    w = [-2.02634712, -9.18111809,  3.33191882,  7.26277662,  2.31369184,  0.1177015,
  4.7110764,   2.32284374,  1.25636615,  8.87445446,  7.7251708,   1.30696076,
 -2.93634346,  0.28851064, -1.02065174,  3.76772538, -2.6743741,   1.33430038,
 -2.91740009,  0.18228678, -0.17278706,  0.38104872]
'''
    #w = [-2.343950260391391,-0.6781706894213994,3.2214360675143943,5.642484030403182,0.21831479210441554,0.4721100741571651,3.9408116180380133,2.561791349698079,1.155025558462022,8.372155792130272,5.965764233294304,1.1276138341064343,-2.9381281547221523,0.3158372728247155,-0.9331436311507566,3.752749935851464,-2.6190142610798204,3.0881699949969432,-0.0681654980041252,0.14738839190814138,-0.18604683667127941,0.35602477192445964]

    #w = [0 for i in range(22)]
    data_dev, y_dev, featureIndex = readDevData()
    
    print(" SSE ",getSSE(w,data_dev,y_dev))

def printW(w):
    s = '['
    for elem in w:
        s = s + str(elem) + ','
    s = s[:-1]
    s = s + ']'
    print(s)

if __name__ == '__main__':
    learning_b()
    #test()
