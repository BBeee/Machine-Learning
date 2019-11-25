#!/usr/bin/env python3
from readFile import *
from numpy import dot,array,sign
#from matplotlib import pyplot as plt

def k(x1,x2,p):
    s = 1 + dot(x1,x2)
    return pow(s,p)

def predictTest(alpha,p):
    data,y = readTrain()
    data_test = readTest()

    l_data = len(data)
    l_test = len(data_test)
    
    data = array(data)
    y = array(y)
    alpha = array(alpha)
    data_test = array(data_test)

    # need double check: dev x train or train x dev
    Kdata = [[0 for i in range(l_test)] for j in range(l_data)]
    for i,_ in enumerate(Kdata):
        for j,_ in enumerate(Kdata[i]):
            Kdata[i][j] = k(data[i],data_test[j],p)

    rs = []
    for i,_ in enumerate(data_test):
        u = 0
        for j in range(l_data):
            u += Kdata[j][i] * alpha[j] * y[j]
        if sign(u) >= 0:
            rs.append('+1\n')
        else:
            rs.append('-1\n')

    f = open('kplabel.csv','w')
    for elem in rs:
        f.write(elem)
    f.close()


def kernel(p):
    data,y = readTrain()
    data_dev,y_dev = readValid()

    l_data = len(data)
    l_dev = len(data_dev)
    alpha = [0 for i in range(l_data)]
    
    alpha = array(alpha)
    data = array(data)
    y = array(y)
    data_dev = array(data_dev)
    y_dev = array(y_dev)
    
    # get K
    Kdata = [[0 for i in range(l_data)] for j in range(l_data)]
    for i,_ in enumerate(Kdata):
        for j,_ in enumerate(Kdata[i]):
            Kdata[i][j] = k(data[i],data[j],p)

    # get K_dev
    Kdata_dev = [[0 for i in range(l_dev)] for j in range(l_data)]
    for i,_ in enumerate(Kdata_dev): # i = 5k
        for j,_ in enumerate(Kdata_dev[i]): # j = 1k6
            Kdata_dev[i][j] = k(data[i],data_dev[j],p)
    rs1,rs2 = [],[]
    epochs = 15
    # train
    bestError = 100000
    bestAlpha = []
    bestEpoch = 0
    bestP = 0
    for epoch in range(epochs):
        print("Epoch: ", epoch+1)
        for i,_ in enumerate(data):
            u = 0
            for j in range(l_data):
                u += Kdata[j][i] * alpha[j] * y[j] # notation: K(i,j) = K(j,i) ???
            if y[i]*sign(u) <= 0:
                alpha[i] += 1
        
        # training set
        error = 0
        for i,_ in enumerate(data):
            u = 0
            for j in range(l_data):
                u += Kdata[j][i] * alpha[j] * y[j]
            if y[i]*sign(u) <= 0:
                error += 1
        rs1.append(round(error*100/l_data,2))
        print("p: ",p, " Training error rate: ",round(error*100/l_data,2)," %")


        # validation set
        error = 0
        for i,_ in enumerate(data_dev):
            u = 0
            for j in range(l_data):
                u += Kdata_dev[j][i] * alpha[j] * y[j]
            if y_dev[i]*sign(u) <= 0:
                error += 1
        if bestError > error:
            bestError = error
            bestAlpha = alpha[:]
            bestEpoch = epoch + 1
            bestP = p
        rs2.append(round(error*100/l_dev,2))
        print("p: ",p, " Validation error rate: ",round(error*100/l_dev,2)," %")
    
    print("best error rate: ",bestError, " best epoch: ", bestEpoch," bestP: ",bestP)
    #predictTest(bestAlpha,bestP)
    return rs1,rs2,bestError,bestAlpha,bestP,bestEpoch

if __name__ =='__main__':
    rs1,rs2 = [],[]
    p = [1,2,3,7,15]
    #p = [3]
    bestError = 100000
    bestAlpha = []
    bestP = 0
    bestEpoch = 0 
 
    for elem in p:
        rs = kernel(elem)
        
        rs1.append(rs[0])
        rs2.append(rs[1])
        if bestError > rs[2]: # best error at each epoch
            bestError = rs[2]
            bestAlpha = rs[3][:]
            bestP = rs[4]
            bestEpoch = rs[5] + 1


    # predict with best model
    predictTest(bestAlpha,bestP)
    
    #x = [i+1 for i in range(15)]
   
    print('---')
    print(rs1)
    print('---')
    print(rs2)
    print('---')
    print(bestError,bestP,bestEpoch)



