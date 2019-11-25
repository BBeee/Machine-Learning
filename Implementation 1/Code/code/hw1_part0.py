#!/usr/bin/env python3
import numpy as np
from collections import defaultdict

# standard for normalization: v = sqrt(a1^2+...+an^2)

def readDevData():
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
                data[i][j] = (data[i][j] - rMin[j]) / (rMax[j] - rMin[j]) 
                # check
                #if data[i][j] < 0 or data[i][j] > 1:
                #    print(data[i][j])
    #print(data)
    return data,y,featureIndex

    '''
    # part 0 (c)
    npData = np.array(data)
    #print(npData)
    mean = np.mean(npData,axis=0)
    rMin = np.min(npData,axis=0)
    rMax = np.max(npData,axis=0)
    std = np.std(npData,axis=0)

    name = []
    for elem in featureIndex:
        name.append(elem)

    for i,_ in enumerate(mean):
        print(name[i], ": mean: ", round(mean[i],2), "deviation: ", round(std[i],2), "range: ", rMin[i], "-", rMax[i])

    #print(featureIndex)
    print("\n","--- ---","\n")

    # waterfront: 6, condition: 8, grade: 9
    target = [6,8,9]
    for num in target:
        temp = defaultdict(int)
        for elem in npData[:,num]:
            temp[elem] += 1
        s = 0
        for elem in temp:
            s += temp[elem]
        for elem in temp:
            temp[elem] = round(temp[elem]/s,4)
        print(name[num], ": ", temp)
    '''

def readData():
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
                data[i][j] = (data[i][j] - rMin[j]) / (rMax[j] - rMin[j]) 
                # check
                #if data[i][j] < 0 or data[i][j] > 1:
                #    print(data[i][j])
    #print(data)
    

    '''
    # part 0 (c)
    npData = np.array(data)
    #print(npData)
    mean = np.mean(npData,axis=0)
    rMin = np.min(npData,axis=0)
    rMax = np.max(npData,axis=0)
    std = np.std(npData,axis=0)

    name = []
    for elem in featureIndex:
        name.append(elem)

    for i,_ in enumerate(mean):
        print(name[i], ": mean: ", round(mean[i],2), "deviation: ", round(std[i],2), "range: ", rMin[i], "-", rMax[i])

    #print(featureIndex)
    print("\n","--- ---","\n")

    # waterfront: 6, condition: 8, grade: 9
    target = [6,8,9]
    for num in target:
        temp = defaultdict(int)
        for elem in npData[:,num]:
            temp[elem] += 1
        s = 0
        for elem in temp:
            s += temp[elem]
        for elem in temp:
            temp[elem] = round(temp[elem]/s,4)
        print(name[num], ": ", temp)
    '''

if __name__ == '__main__':
    readData()
    pass
