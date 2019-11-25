#!/usr/bin/env python3
from readFile import *
from numpy import dot,array,sign
#from matplotlib import pyplot as plt

def predictTest(w):
    data = readTest()
    y = []
    for i,_ in enumerate(data):
        s = dot(w,data[i])
        if s >= 0:
            y.append('+1\n')
        else:
            y.append('-1\n')
    f = open('oplabel.csv','w')
    for elem in y:
        f.write(elem)
    f.close()


def perceptron():
    data,y = readTrain()
    data_dev,y_dev = readValid()

    l_feature = len(data[0])
    l_data = len(data)
    l_dev = len(data_dev)
    w = [0 for i in range(l_feature)]
    
    w = array(w)
    data = array(data)
    y = array(y)
    data_dev = array(data_dev)
    y_dev = array(y_dev)
    
    rs1 = []
    rs2 = []
    epochs = 30
    bestEpoch = 0
    bestW = 0
    bestError = 10000
    # train
    for epoch in range(epochs):
        for i,_ in enumerate(data):
            # when predict wrong -> update w
            if (y[i]*sign(w.dot(data[i]))) <= 0:
                w = w + y[i]*data[i]

        # predict on train
        error = 0
        for i,_ in enumerate(data):
            if (y[i]*sign(w.dot(data[i]))) <= 0:
                error += 1
        print("epoch: ", epoch+1)
        print("Training error rate: ",round(error*100/l_data,2)," %")
        rs1.append(round(error*100/l_data,2))
        
        # predict on dev
        error = 0
        for i,_ in enumerate(data_dev):
            if (y_dev[i]*sign(w.dot(data_dev[i]))) <= 0:
                error += 1

        # getting the best w
        if error < bestError:
            bestError = error
            bestW = w[:]
            bestEpoch = epoch+1

        print("Validation error rate: ",round(error*100/l_dev,2)," %")
        rs2.append(round(error*100/l_dev,2))

    print(bestEpoch,round(bestError*100/l_dev,2))
    x = [i+1 for i in range(epochs)]
    #plt.plot(x,rs1,'r')
    #plt.plot(x,rs2,'g')
    #plt.show()
    #print(rs1,rs2)
    # predict best model W in Test data set, write in file
    predictTest(bestW)

if __name__ =='__main__':
    perceptron()




