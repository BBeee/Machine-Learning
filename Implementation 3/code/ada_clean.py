#!/usr/bin/env python3
from readfile import *
#from matplotlib import pyplot as plt
from random import choice
from copy import deepcopy
import numpy as np
from math import e,log

def getFeatureThreshold(data,y):
    D = len(data)
    F = len(data[0])
    rs = 0,0,0,0
    feature,threshold = 0,0
    # find the first
    pos,neg = 0,0
    for elem in y:
        if elem == 1:
            pos += 1
        elif elem == -1:
            neg += 1
    U = getEntrophy(pos,neg)
        
    biggestGain = 0 # calculated Uncertainty
    for i in range(F): # # of features
        
        ithSortedData = []
        for j in range(D): # length of data
            ithSortedData.append((data[j][i],j)) # sorted data = value, real index
        ithSortedData.sort()
        last = ithSortedData[0][1]        # first sorted element: label, real index
        for elem in ithSortedData[1:]: # only find middle
            if y[last] != y[elem[1]]: # last label != current label
                t = (data[elem[1]][i] + data[last][i])/2 # temp threshold
                posL,negL,posR,negR = count(data, y, i, t) 
                if posL+negL != 0 and posR+negR != 0:
                    b = getBenefit(posL,negL,posR,negR,pos+neg,U)
                    if biggestGain < b:
                        biggestGain = b
                        feature = i
                        threshold = t
                        rs = posL,negL,posR,negR
                #if biggestGain < t2:
                #    biggestGain = t2
            last = elem[1]
        # calculate each Entrophy
    #print(rs)
    #print(biggestGain,i,feature,threshold)
    return feature,threshold # ,rs     
def getFirstFeatureThreshold(data,y):
    D = len(data)
    F = len(data[0])
    
    feature,threshold = 0,0
    # find the first
    pos,neg = 0,0
    for elem in y:
        if elem == 1:
            pos += 1
        elif elem == -1:
            neg += 1
    U = getEntrophy(pos,neg)
    #print(U,pos,neg)
    
    biggestGain = 0 # calculated Uncertainty
    for i in range(F): # # of features
        
        ithSortedData = []
        for j in range(D): # length of data
            ithSortedData.append((data[j][i],j)) # sorted data = value, real index
        ithSortedData.sort()
        
        last = ithSortedData[0][1]        # first sorted element: label, real index
        
        for elem in ithSortedData[1:]: # only find middle
            if y[last] != y[elem[1]]: # last label != current label
                t = (data[elem[1]][i] + data[last][i])/2 # temp threshold
                posL,negL,posR,negR = count(data, y, i, t) 
                if posL+negL != 0 and posR+negR != 0:
                    b = getBenefit(posL,negL,posR,negR,pos+neg,U)
                    if biggestGain < b:
                        biggestGain = b
                        feature = i
                        threshold = t
                #if biggestGain < t2:
                #    biggestGain = t2
            last = elem[1]
        # calculate each Entrophy
        #print(ithSortedData)
        
    #print(biggestGain,feature,threshold)
    return feature,threshold
# tree data structure found on Website
def binary_tree(r):
    return [r, [], []]

def getEntrophy(pos,neg):
    # pos + neg cannot equal 0
    return 1 - pow(pos/(pos+neg),2) - pow(neg/(pos+neg),2)

def getBenefit(posL,negL,posR,negR,total,U):
    # total = pos + neg
    UL = getEntrophy(posL, negL)
    UR = getEntrophy(posR, negR)
    pl = (posL+negL)/total
    pr = (posR+negR)/total
    return U - pl * UL - pr * UR

def count(data,y,ithFeature,threshold):
    posL,negL = 0,0
    posR,negR = 0,0
    for i,_ in enumerate(data):
        if data[i][ithFeature] <= threshold:
            if y[i] == 1:
                posL += 1
            else:
                negL += 1
        else:
            if y[i] == 1:
                posR += 1
            else:
                negR += 1
    return posL,negL,posR,negR


def getPath(rowData,tree,depth):
    if tree == []:
        return ''
    if tree[0][0] == 1:
        return '+'
    if tree[0][0] == -1:
        return '-'
    if depth > 1:
        label, feature, threshold = tree[0]
        if rowData[feature] <= threshold:
            return 'l' + getPath(rowData,tree[1],depth-1) 
        else:
            return 'r' + getPath(rowData,tree[2],depth-1) 
    else:
        label, feature, threshold = tree[0]
        if rowData[feature] <= threshold:
            return 'l'
        else:
            return 'r'

def getError(D,tree):
    data, y = readTrain()
    for i,_ in enumerate(data):
        for j,_ in enumerate(data[i]):
            data[i][j] *= D[i]
    
    d = {}
    rs = []
    epsilon = 0
    depth = 10
    # split data
    for i,line in enumerate(data):
        p = getPath(line,tree,depth)
        if p[-1] == '+' or p[-1] == '-':
            # met class label & do nothing
            pass
        else:
            # add this data to: d[path] = [[all data],[all y]]
            if p not in d:
                d[p] = [[],[]]
            d[p][0].append(line)
            d[p][1].append(y[i])
    
    # testing on training
    error = 0
    route = ''
    # data_dev, y_dev
    for i,line in enumerate(data):
        route = getPath(line,tree,depth)
        u = 0
        # predict
        if route[-1] == '+':
            u = 1
        elif route[-1] == '-':
            u = -1 
        else:
            # majority
            t = getNode(tree,route)
            node = t[0]
            posL,negL,posR,negR = count(d[route][0],d[route][1],node[1],node[2])
            if line[node[1]] <= node[2]:
                if posL >= negL:
                    u = 1
                else: 
                    u = -1
            else:
                if posR >= negR:
                    u = 1
                else:
                    u = -1
        if u*y[i] <= 0:
            error += 1
        if u == 1:
            rs.append(1)
        else:
            rs.append(-1)
    epsilon = error/len(data)
    #print(epsilon)
    return epsilon,rs

def getValError(D,tree):
    data, y = readDev()
    for i,_ in enumerate(data):
        for j,_ in enumerate(data[i]):
            data[i][j] *= D[i]
    
    d = {}
    rs = []
    epsilon = 0
    depth = 10
    # split data
    for i,line in enumerate(data):
        p = getPath(line,tree,depth)
        if p[-1] == '+' or p[-1] == '-':
            # met class label & do nothing
            pass
        else:
            # add this data to: d[path] = [[all data],[all y]]
            if p not in d:
                d[p] = [[],[]]
            d[p][0].append(line)
            d[p][1].append(y[i])
    
    # testing on training
    error = 0
    route = ''
    # data_dev, y_dev
    for i,line in enumerate(data):
        route = getPath(line,tree,depth)
        u = 0
        if route[-1] == '+':
            u = 1
        elif route[-1] == '-':
            u = -1 
        else:
            # majority
            t = getNode(tree,route)
            node = t[0]
            posL,negL,posR,negR = count(d[route][0],d[route][1],node[1],node[2])
            if line[node[1]] <= node[2]:
                if posL >= negL:
                    u = 1
                else: 
                    u = -1
            else:
                if posR >= negR:
                    u = 1
                else:
                    u = -1
        if u*y[i] <= 0:
            error += 1
        if u == 1:
            rs.append(1)
        else:
            rs.append(-1)
    epsilon = error/len(data)
    #print(epsilon,"validation")
    return epsilon,rs
def normalize(D):
    norm = np.linalg.norm(D)
    if norm == 0:
        return D
    for i,_ in enumerate(D):
        D[i] = D[i]/norm
    return D

def ada(L):
    data, y = readTrain()
    # initial
    D = [[1/len(data) for i in range(len(data))] for j in range(L+1)]
    alpha = [0 for i in range(L)]
    forest = []
    for i in range(L):
        tree = dl(D[i])
        forest.append(tree)
        epsilon,rs = getError(D[i],tree)
        # update 
        alpha[i] = 0.5*log((1-epsilon)/epsilon)
        for j,elem in enumerate(D[i]):
            if y[j] == rs[j]:
                D[i+1][j] *= pow(e,-alpha[i])
            elif y[j] != rs[j]:
                D[i+1][j] *= pow(e,alpha[i])

        D[i+1] = normalize(D[i+1])
    
    # predicting on Test
    # using forest & alpha to predict
    predictSet = [0 for i in range(len(data))]
    for i in range(L):
        _,result = getError(D[i],forest[i])
        for j,_ in enumerate(predictSet):
            predictSet[j] += result[j] * alpha[i]
    
    correct = 0

    for i,_ in enumerate(predictSet):
        if predictSet[i] * y[i] >= 0:
            correct += 1
    rs1 = correct/len(data)

    # predict validation
    data_dev, y_dev = readDev()
    predictSet = [0 for i in range(len(data_dev))]
    for i in range(L):
        _,result = getValError(D[i],forest[i])
        for j,_ in enumerate(predictSet):
            predictSet[j] += result[j] * alpha[i]
    
    correct = 0

    for i,_ in enumerate(predictSet):
        if predictSet[i] * y_dev[i] >= 0:
            correct += 1
    rs2 = correct/len(data_dev)
    return rs1,rs2

    
def dl(D):
    # get data 
    data, y = readTrain()
    for i,_ in enumerate(data):
        for j,_ in enumerate(data[i]):
            data[i][j] *= D[i]
    
    #data=data[:2000] #for faster
    #y = y[:2000]
    # each node: (label: +1,0,-1, classified feature: 0-99, threshold)
    
    feature,threshold = getFirstFeatureThreshold(data,y) #for faster
    tree = binary_tree((0,feature,threshold))
    #tree = binary_tree((0,3,0.040029725859247134))

    p = '' # for memorizing path
    
    # training: 
    for i in range(1,11): # i is the maximum depth
        d = {}
        #print("this is ", i, " iterations")
        for j,line in enumerate(data):
            p = getPath(line,tree,i)
            if p[-1] == '+' or p[-1] == '-':
                # met class label & do nothing
                data.remove(line)
                #pass
            else:
                # add this data to: d[path] = [[all data],[all y]]
                if p not in d:
                    d[p] = [[],[]]
                d[p][0].append(line)
                d[p][1].append(y[j])
        
        # check data at each path
        for path in d:
            # get best feature & threshold at this path 
            feature,threshold = getFeatureThreshold(d[path][0],d[path][1])
            
            # expend node:
            t = getNode(tree,path)
            node = t[0]
            posL,negL,posR,negR = count(d[path][0],d[path][1],node[1],node[2])
            
            # expend tree with [(label,0,0),[],[]]
            label = 0
            if posL == 0 and negL != 0:
                label = -1
                expandLabel(tree,path,label) # l -1
            elif posL != 0 and negL == 0:
                label = 1
                expandLabel(tree,path,label)
            elif posR == 0 and negR != 0:
                label = -1
                expandLabel(tree,path,label)
            elif posR != 0 and negR == 0:
                label = 1
                expandLabel(tree,path,label)
            # expend tree with [(0,feature,threshold),[],[]]
            else: 
                expandNode(tree,path,feature,threshold)
        
    return tree

def expandLabel(tree,path,label):
    if path[0] == 'l' and len(path) > 1:
        return expandLabel(tree[1],path[1:],label)
    elif path[0] == 'r' and len(path) > 1: 
        return expandLabel(tree[2],path[1:],label)
    elif path[0] == 'l' and len(path) == 1:
        tree[1] = [(label,0,0),[],[]]
        return tree
    else:
        tree[2] = [(label,0,0),[],[]]
        return tree

def getNode(tree,path):
    if path[0] == 'l' and len(path) > 1:
        return getNode(tree[1],path[1:])
    elif path[0] == 'r' and len(path) > 1: 
        return getNode(tree[2],path[1:])
    elif len(path) == 1:
        return tree
    

def expandNode(tree,path,feature,threshold):
    if path[0] == 'l' and len(path) > 1:
        return expandNode(tree[1],path[1:],feature,threshold)
    elif path[0] == 'r' and len(path) > 1: 
        return expandNode(tree[2],path[1:],feature,threshold)
    elif path[0] == 'l' and len(path) == 1:
        tree[1] = [(0,feature,threshold),[],[]]
        return tree
    else:
        tree[2] = [(0,feature,threshold),[],[]]
        return tree



if __name__ == '__main__':
    y1 = []
    y2 = []
    L = [1,5,10,20]
    for i in L:
        rs1,rs2 = ada(i)
        y1.append(rs1)
        y2.append(rs2)
        print(i,rs1,rs2)
    print(y1,y2)
    '''
    plt.plot(L,y1,label='training')
    plt.plot(L,y2,label='validation')
    plt.grid(True)
    plt.legend(shadow=False, fontsize = 'small')
    plt.title('figure 3')
    plt.show()
    '''

