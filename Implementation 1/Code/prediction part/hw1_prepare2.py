#!/usr/bin/env python3
import numpy as np
from collections import defaultdict

# standard for normalization: v = sqrt(a1^2+...+an^2)
binarized = [1,2,21]

def readDevData(fI):
    data = []
    y = []
    fI = {}
    start = 63
    #binarized = [1,2,21]

    for i, line in enumerate(open('PA1_dev.csv')):
        if i != 0: # rest data
            line = line.strip()
            line = line.split(',')
            newLine = []
            # new data, y
            for i,elem in enumerate(line):
                if i not in binarized:
                    try:
                        newLine.append(int(elem))
                    except:
                        newLine.append(float(elem))
            for i in range(45): # 12+31+2+70
                newLine.append(0)
            # 1 remove
            # 2 date
            month,day,year = line[2].split('/')
            month = int(month) + 17
            day = int(day) + 29
            year = int(year) - 2013 + 60
            newLine[month] = 1
            newLine[day] = 1
            newLine[year] = 1
            '''
            # 16 zipcode
            zipcode = line[16]
            zipcode = int(zipcode)
            if zipcode in fI:
                newLine[fI[zipcode]] = 1
            
            if zipcode in fI:
                newLine[fI[zipcode]] = 1
            else:
                fI[zipcode] = start
                newLine[fI[zipcode]] = 1
                start += 1
            '''
            # 21 y
            y.append(float(line[-1]))
            data.append(newLine)
    # normalization
    npData = np.array(data)
    #print(npData)
    rMin = np.min(npData,axis=0)
    rMax = np.max(npData,axis=0)
    mean = np.mean(npData,axis=0)
    std = np.std(npData,axis=0)
    for i,_ in enumerate(data):
        for j,_ in enumerate(data[i]):
            if j != 0:
                if rMax[j] != rMin[j]:
                    data[i][j] = (data[i][j] - rMin[j]) / (rMax[j] - rMin[j]) 
                # check
                if data[i][j] < 0 or data[i][j] > 1:
                    print(data[i][j])
    #print(data)   
            
    return data,y

def readData():
    data = []
    y = []
    fI = {}
    start = 63
    
    for i, line in enumerate(open('PA1_train.csv')):
        if i != 0: # rest data
            line = line.strip()
            line = line.split(',')
            newLine = []
            # new data, y
            for i,elem in enumerate(line):
                if i not in binarized:
                    try:
                        newLine.append(int(elem))
                    except:
                        newLine.append(float(elem))
            for i in range(45): # 12+31+2+70 = 115
                newLine.append(0)
            # 1 remove
            # 2 date
            month,day,year = line[2].split('/')
            month = int(month) + 17
            day = int(day) + 29
            year = int(year) - 2013 + 60
            newLine[month] = 1
            newLine[day] = 1
            newLine[year] = 1
            '''
            # 16 zipcode
            zipcode = line[16]
            zipcode = int(zipcode)
            if zipcode in fI:
                newLine[fI[zipcode]] = 1
            else:
                fI[zipcode] = start
                newLine[fI[zipcode]] = 1
                start += 1
                '''
            # 21 y
            y.append(float(line[-1]))
            data.append(newLine)
    # normalization
    npData = np.array(data)
    #print(npData)
    rMin = np.min(npData,axis=0)
    rMax = np.max(npData,axis=0)
    mean = np.mean(npData,axis=0)
    std = np.std(npData,axis=0)
    for i,_ in enumerate(data):
        for j,_ in enumerate(data[i]):
            if j != 0:
                if rMax[j] != rMin[j]:
                    data[i][j] = (data[i][j] - rMin[j]) / (rMax[j] - rMin[j]) 
                # check
                if data[i][j] < 0 or data[i][j] > 1:
                    print(data[i][j])# check
                #if data[i][j] < 0 or data[i][j] > 1:
                #    print(data[i][j])
    print(len(data[0]))
            
    return data,y,fI


if __name__ == '__main__':
    data, y, fI = readData()
    #data_dev, y_dev = readDevData(fI)
    
    