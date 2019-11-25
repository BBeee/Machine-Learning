#!/usr/bin/env python
from numpy import array

def readTrain():
    data,y = [],[]
    for line in open('pa2_train.csv'):
        line = line.strip()
        line = line.split(',')
        if line[0] == '3':
            y.append(1)
        else:
            y.append(-1)
        for i,elem in enumerate(line):
            line[i] = float(elem)
        # add bias
        line.append(1)
        data.append(line[1:])
        
    return data,y

def readValid():
    data,y = [],[]
    for line in open('pa2_valid.csv'):
        line = line.strip()
        line = line.split(',')
        if line[0] == '3':
            y.append(1)
        else:
            y.append(-1)
        for i,elem in enumerate(line):
            line[i] = float(elem)
        # add bias
        line.append(1)
        data.append(line[1:])
        
    return data,y

def readTest():
    data = []
    for line in open('pa2_test_no_label.csv'):
        line = line.strip()
        line = line.split(',')
        for i,elem in enumerate(line):
            line[i] = float(elem)
        # add bias
        line.append(1)
        data.append(line)
        
    return data

if __name__ == '__main__':

    data,y = readValid()
    data = array(data)
    y = array(y)
    print(data)
    print(y)