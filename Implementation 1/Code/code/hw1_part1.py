#!/usr/bin/env python3
from hw1_part0 import *
from math import sqrt,log
from matplotlib import pyplot as plt
import numpy as np

# lambda = 0


def learning(): # naive implementation
    data, y, featureIndex = readData()
    print(featureIndex)
    epsilon = 0.5
    learningRate = -0.00001#0.0000575
    # feature length
    featureLength = len(data[0])
    w = [0 for i in range(featureLength)] # the weight
    #w = [-2.0269718949575051, -9.1503107278509077, 3.3298283669595592, 7.260567251767001, 2.0973287898614794, 0.1182811160529204, 4.7112495122210598, 2.3240526325279953, 1.2557178422406909, 8.8768799535700076, 7.7230543273503844, 1.3058341903008572, -2.93679841017557, 0.28833488189522843, -1.0207455304257818, 3.767340212313858, -2.6749311570565437, 1.3318061874551816, -2.7590331167617421, 0.18221104910508959, -0.17272965980641097, 0.38103305509199797]
    deltaW = [0 for i in range(featureLength)]  
    # calculate deltaW  
    for i,_ in enumerate(data):
        addVector(deltaW, mulVector((dotProduct(w, data[i])-y[i]), data[i]))

    # judge with epsilon
    l = getLength(deltaW)
    
    while l > epsilon:
        deltaW = [0 for i in range(featureLength)]
        for i,_ in enumerate(data):
            deltaW = addVector(deltaW, mulVector((dotProduct(w, data[i])-y[i]), data[i]))
        print(l)
        l = getLength(deltaW)
        print(w)
        # update w
        w = addVector(w, mulVector(learningRate, deltaW))
    #deltaW
    print(w)


def learning_a(): # implementation with making graph
    data, y, featureIndex = readData()
    epsilon = 0.5
    #learningRate = [-1,-0.1,-0.01,-0.001,-0.0001] #,-0.00001,-0.000001,-0.0000001] #0.0000575
    learningRate = [-0.00001,-0.000001,-0.0000001] #,-0.00001,-0.000001,-0.0000001] #0.0000575

    #x_G = [1, 5, 10, 15, 20]
    x_G = [3*i+1 for i in range(20)]
    y_G = [0 for i in range(len(x_G))]
    # feature length
    featureLength = len(data[0])
    w = [0 for i in range(featureLength)] # the weight
    deltaW = [0 for i in range(featureLength)]  
    
    # calculate deltaW  
    for i,_ in enumerate(data):
        addVector(deltaW, mulVector((dotProduct(w, data[i])-y[i]), data[i]))

    #iterations = [1000,000,]
    # judge with fixed # of iterations
    l = getLength(deltaW)
    
    start = 0
    end = x_G[-1]

    # current y_G is depending on l, change it to SSE
    for elem in learningRate:
        print("LR: ", elem)
        while  start <= end:
            deltaW = [0 for i in range(featureLength)]
            for i,_ in enumerate(data):
                deltaW = addVector(deltaW, mulVector((dotProduct(w, data[i])-y[i]), data[i]))
            
            #print(getSSE(w,data,y))
            l = getLength(deltaW)
            if start in x_G:
                pos = x_G.index(start)
                y_G[pos] = (getSSE(w,data,y))

            # update w
            w = addVector(w, mulVector(elem, deltaW)) # elem in learningrate
            start += 1
        print(x_G,y_G)
        plt.plot(x_G,y_G)
        # reset the parameter
        start = 0
        w = [0 for i in range(featureLength)] # the weight
    plt.show()



def learning_b(): # implementation of part 1.b with making graph
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

    lll = 20
    gap = 25
    x_G = [gap*i for i in range(lll)]
    y_G = [0 for i in range(lll)]
    y_G_dev = [0*i for i in range(lll)]
    
    
    # calculate deltaW  
    for i,_ in enumerate(data):
        addVector(deltaW, mulVector((dotProduct(w, data[i])-y[i]), data[i]))
    l = getLength(deltaW)
    
    # ---
    # learning
    print("LR: ", learningRate)
    temp_SSE = float('inf')    
    min_SSE = float('inf')
    train_iter = 0
    iterations = 0

    while iterations <= lll*gap:
        print(iterations)
        deltaW_np = np.array([0.0 for i in range(featureLength)])
        for i,_ in enumerate(data_np):
            deltaW_np += (w_np.dot(data_np[i]) - y_np[i]) * data_np[i]
        
        '''
        if iterations in x_G:
            pos = x_G.index(iterations)
            y_G[pos] = (getSSE(w_np,data,y)/len(data))
            y_G_dev[pos] = (getSSE(w_np,data_dev,y_dev)/len(data_dev))
        '''

        temp_SSE = getSSE(w_np,data_dev,y_dev)    
        if min_SSE > temp_SSE:
            min_SSE = temp_SSE
            w = w_np
            train_iter = iterations

        # update w
        w_np = w_np + learningRate * deltaW_np
        iterations += 1
    printW(w)
    print("---")
    print(train_iter,min_SSE)
    
    #plt.plot(x_G,y_G)
    #plt.plot(x_G,y_G_dev)
    #plt.show()

def learning_addtion(): # implementation of part 1.b with making graph
    data, y, featureIndex = readData()
    data_dev, y_dev, featureIndex_dev = readDevData()
    print(len(data_dev))
    epsilon = 0.5
    featureLength = len(data[0])
    w = [0 for i in range(featureLength)]
    deltaW = [0 for i in range(featureLength)]  
    
    w_np = np.array(w)
    deltaW_np = np.array(deltaW)
    data_np = np.array(data)
    y_np = np.array(y)

    #data_dev_np = np.array(data_dev)
    #y_dev_np = np.array(y_dev)

    #learningRate = [-0.000057,-0.00001,-0.000001,-0.0000001] 
    learningRate = -0.000057

    lll = 100
    gap = 25
    
    # calculate deltaW  
    for i,_ in enumerate(data):
        addVector(deltaW, mulVector((dotProduct(w, data[i])-y[i]), data[i]))
    l = getLength(deltaW)
    
    # ---
    # learning
    print("LR: ", learningRate)

    temp_SSE = float('inf')    
    min_SSE = float('inf')

    iterations = 0
    #while l > epsilon:
    while iterations <2:
        print(l)
        print("---")
        #print(np.dot(data_np.transpose(),data_np))
        print("---")
        
        deltaW_np = np.dot(np.linalg.inv(np.dot(data_np.transpose(),data_np)), np.dot(data_np.transpose(), y_np))
        #print(deltaW)
        #l = getLength(deltaW_np)
        
        # update w
        #w_np = w_np + learningRate * deltaW_np
        #printW(w_np)
        printW(deltaW_np)
        iterations += 1

    
    
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
def test():
    print("===---===")
    data_dev, y_dev, featureIndex = readDevData()
    lll = 50
    x_G = [i for i in range(lll)]
    y_G_real = y_dev[:lll]
    y_G_pred = [0 for i in range(lll)]
    y_G_over = [0 for i in range(lll)]
    
    w1 = [-2.33305302607,-0.63864621805,3.22057616517,5.61848200793,0.215808758955,0.491356675519,3.89496616043,2.57352968474,1.14897892201,8.32943754185,5.934882291,1.13999928855,-2.94186492965,0.315285474835,-0.931449463156,3.75095749501,-2.60095170975,3.13733375559,-0.0605626739493,0.142941633338,-0.187155084499,0.353414185215]
    w2 = [-2.02634712, -9.18111809,  3.33191882,  7.26277662,  2.31369184,  0.1177015,
  4.7110764,   2.32284374,  1.25636615,  8.87445446,  7.7251708,   1.30696076,
 -2.93634346,  0.28851064, -1.02065174,  3.76772538, -2.6743741,   1.33430038,
 -2.91740009,  0.18228678, -0.17278706,  0.38104872]
    for i in range(lll):
        y_G_pred[i] = np.dot(w1,data_dev[i])
        y_G_over[i] = np.dot(w2,data_dev[i])
    
    print(getSSE(w1,data_dev,y_dev))
    print(getSSE(w2,data_dev,y_dev))
    
    plt.plot(x_G,y_G_real,c='r')
    plt.plot(x_G,y_G_pred,c='g')
    plt.plot(x_G,y_G_over,c='b')
    plt.show()
def printW(w):
    s = '['
    for elem in w:
        s = s + str(elem) + ','
    s = s[:-1]
    s = s + ']'
    print(s)

if __name__ == '__main__':
    #learning_b()
    test()
