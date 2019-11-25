#!/usr/bin/env python3
from readFile import *
from numpy import dot,array
#from matplotlib import pyplot as plt

def avg_perceptron():
    data,y = readTrain()
    data_dev,y_dev = readValid()

    l_feature = len(data[0])
    l_data = len(data)
    l_dev = len(data_dev)
    w = [0 for i in range(l_feature)]
    wa = [0 for i in range(l_feature)]
    s = 0
    c = 0

    rs1 = []
    rs2 = []
    
    w = array(w)
    wa = array(wa)
    data = array(data)
    y = array(y)
    data_dev = array(data_dev)
    y_dev = array(y_dev)
    
    epochs = 30
    bestError = 10000
    bestModel = w[:]
    bestEpoch = 0
    for epoch in range(epochs):
        for i,elem in enumerate(data):
            # when predict wrong -> update w
            if (y[i]*w.dot(data[i])) <= 0:
                if s+c > 0:
                    wa = (s*wa + c*w)/(s+c)
                s = s+c
                w = w + y[i]*data[i]
                c = 0
            else:
                c += 1
        if c > 0:
            wa = (s*wa + c*w)/(s+c)

        # predict on train
        error = 0
        for i,_ in enumerate(data):
            if (y[i]*wa.dot(data[i])) <= 0:
                error += 1
        print("epoch: ", epoch+1)
        print("Training error rate: ",round(error*100/l_data,2)," %")
        rs1.append(round(error*100/l_data,2))
        
        # predict
        error = 0
        for i,elem in enumerate(data_dev):
            if (y_dev[i]*wa.dot(data_dev[i])) <= 0:
                error += 1
        if bestError > error:
            bestError = error
            bestEpoch = epoch+1
            bestModel = w[:]


        print("Validation error rate: ",round(error*100/l_dev,2)," %")
        rs2.append(round(error*100/l_dev,2))

    print(bestEpoch,round(bestError*100/l_dev,2)," %")

    x = [i+1 for i in range(epochs)]
    rs3 = [5.26, 4.23, 5.01, 3.52, 4.32, 4.23, 4.11, 4.34, 4.09, 3.78, 3.78, 4.09, 3.54, 3.38, 4.36, 3.23, 3.31, 3.85, 3.62, 3.15, 3.5, 3.81, 3.54, 3.62, 3.03, 3.97, 3.25, 4.09, 3.29, 3.5, 3.68, 2.64, 2.93, 3.42, 2.93, 3.38, 2.99, 3.25, 4.17, 2.56, 3.62, 2.95, 2.76, 2.99, 3.31, 2.72, 3.31, 3.33, 2.82, 2.45, 3.36, 2.7, 3.03, 2.68, 3.6, 3.19, 3.03, 2.54, 2.62, 2.5, 2.27, 2.88, 3.09, 2.93, 2.86, 2.82, 2.56, 3.15, 3.15, 3.42, 2.5, 2.74, 2.35, 3.05, 2.95, 2.93, 2.37, 2.66, 2.68, 2.52, 2.37, 2.11, 2.17, 2.72, 2.21, 2.19, 2.91, 2.97, 2.56, 2.11, 2.76, 2.74, 2.56, 2.23, 2.54, 2.7, 2.95, 2.23, 2.56, 2.78] 
    rs4 = [6.63, 5.46, 6.51, 5.16, 6.32, 5.65, 6.2, 6.14, 5.22, 6.08, 5.95, 5.71, 5.77, 4.73, 5.4, 5.59, 5.89, 5.95, 6.02, 5.22, 5.4, 5.95, 4.91, 5.28, 5.34, 5.95, 5.59, 5.83, 5.16, 5.83, 5.22, 4.91, 4.97, 5.59, 5.34, 5.65, 5.1, 5.03, 5.89, 5.34, 5.4, 5.52, 5.22, 5.65, 5.59, 5.28, 5.28, 5.71, 5.59, 5.16, 5.89, 5.22, 5.28, 5.28, 6.51, 5.83, 5.46, 5.65, 5.22, 5.52, 5.03, 5.16, 5.71, 5.65, 5.22, 5.59, 5.59, 5.83, 5.77, 5.52, 5.46, 5.4, 5.16, 5.28, 5.71, 5.65, 5.46, 5.46, 5.71, 4.97, 5.59, 5.1, 5.22, 5.77, 4.97, 5.95, 5.83, 5.16, 5.22, 5.46, 5.46, 5.71, 5.71, 5.34, 5.83, 5.4, 5.77, 5.34, 5.71, 6.14]
    #plt.plot(x,rs1,'r')
    #plt.plot(x,rs2,'g')
    #plt.plot(x,rs3,'b')
    #plt.plot(x,rs4,'black')
    
    #plt.show()

if __name__ =='__main__':
    avg_perceptron()