#!/usr/bin/env python3
import numpy as np
from collections import defaultdict

# standard for normalization: v = sqrt(a1^2+...+an^2)

def readDevDataWithoutNormalization():
    featureIndex = {}
    data = []
    y = []

    for i, line in enumerate(open('PA1_dev.csv')):
        if i == 0:
            line = line.strip()
            line = line.split(',')
            line.remove('id')
            line.remove('date')
            line.remove('price')
            
            line.append('month')
            line.append('day')
            line.append('year')
            
            for j, elem in enumerate(line):
                featureIndex[elem] = j

        if i != 0:
            line = line.strip()
            line = line.split(',')
            #line = line[0].append(line[2:]) 
            line.remove(line[1])
            date = line[1]
            m,d,year = date.split('/')
            line.remove(date)
            line, tempy = line[:-1], line[-1]
            
            line.append(m)
            line.append(d)
            line.append(year)
            data.append([])
            y.append(float(tempy))
            for elem in line:
                try:
                    data[i-1].append(int(elem))
                except:
                    data[i-1].append(float(elem))
            #print(data,len(data))
            #break
    

    #print(data)
    return data,y,featureIndex


def readDataWithoutNormalization():
    featureIndex = {}
    data = []
    y = []

    for i, line in enumerate(open('PA1_train.csv')):
        if i == 0:
            line = line.strip()
            line = line.split(',')
            line.remove('id')
            line.remove('date')
            line.remove('price')
            
            line.append('month')
            line.append('day')
            line.append('year')
            
            for j, elem in enumerate(line):
                featureIndex[elem] = j

        if i != 0:
            line = line.strip()
            line = line.split(',')
            #line = line[0].append(line[2:]) 
            line.remove(line[1])
            date = line[1]
            m,d,year = date.split('/')
            line.remove(date)
            line, tempy = line[:-1], line[-1]
            
            line.append(m)
            line.append(d)
            line.append(year)
            data.append([])
            y.append(float(tempy))
            for elem in line:
                try:
                    data[i-1].append(int(elem))
                except:
                    data[i-1].append(float(elem))
            #print(data,len(data))
            #break
    

    return data,y,featureIndex



if __name__ == '__main__':
    readDataWithoutNormalization()
    readDevDataWithoutNormalization()
    pass
