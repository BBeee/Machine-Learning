#!/usr/bin/env python3
from math import sqrt,log
from matplotlib import pyplot as plt
import numpy as np
from numpy import dot


def readData3():
    data = []
    
    for i, line in enumerate(open('PA1_test.csv')):
        if i != 0:
            line = line.strip()
            line = line.split(',')
            #line = line[0].append(line[2:]) 
            line.remove(line[1])
            date = line[1]
            m,d,year = date.split('/')
            line.remove(line[1])
            
            line.append(m)
            line.append(d)
            line.append(year)
            data.append([])
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
    return data


def readData2(fI):
    data = []
    start = 63
    binarized = [1,2,16] # no y

    for i, line in enumerate(open('PA1_test.csv')):
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
            for i in range(115): # 12+31+2+70
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
            
            # 16 zipcode
            zipcode = line[16]
            zipcode = int(zipcode)
            if zipcode in fI:
                newLine[fI[zipcode]] = 1
            
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
        
    return data

def readData():
    fI = {}
    start = 63
    
    for i, line in enumerate(open('PA1_train.csv')):
        if i != 0: # rest data
            line = line.strip()
            line = line.split(',')
            
            # 16 zipcode
            zipcode = line[16]
            zipcode = int(zipcode)
            if zipcode not in fI:
                fI[zipcode] = start
                start += 1
            
    return fI

def writeData(y):
    data = []
    for i,line in enumerate(open('PA1_test.csv')):
        if i == 0:
            line = line[:-1] # no\n
            line = line + ',price\n'
        else:
            line = line[:-1] # no\n
            line = line + ',' + str(y[i-1]) + '\n'
        data.append(line)
    
    f = open('test_result.csv','w')
    for elem in data:
        f.write(elem)
    f.close()
    
    f = open('y_result.csv','w')
    for elem in y:
        s = '\n'
        s = str(elem) + s
        f.write(s)
    f.close()


if __name__ == '__main__':
    #learning()
    fi = readData()
    data = readData2(fi)
    data2 = readData3()

    wtest = [-2.33305303, -0.63864622,  3.22057617,  5.61848201,  0.21580876,  0.49135668,
  3.89496616,  2.57352968,  1.14897892,  8.32943754,  5.93488229,  1.13999929,
 -2.94186493,  0.31528547, -0.93144946,  3.7509575,  -2.60095171,  3.13733376,
 -0.06056267,  0.14294163, -0.18715508,  0.35341419]
    ytest = []
    for line in data2:
        ytest.append(np.dot(wtest,line))


    w = [-1.25031898395,-0.545534160098,2.83625248069,6.12617787723,0.660307244726,-0.771024552983,4.54231879197,2.47896702915,1.10026899968,6.53879968982,6.74110345564,0.400457740038,-1.21903657579,0.247187167953,3.00165096259,-1.61145283217,2.23286648331,0.367662178446,-0.617653527994,-0.596081186751,-0.241542054924,-0.235649046015,0.0452273141427,0.0211717183772,0.0833793533191,0.108129385295,0.0871720529843,0.0158460829439,0.0764175546294,0.00326337004282,0.0157802449054,-0.0211864474218,0.06573557534,-0.0258721987851,-0.1684053527,-0.0367708182176,0.0226739248374,0.0909130492062,-0.0472411572148,-0.109154442225,0.0624639742174,-0.15695833297,0.183588329395,0.0487715161388,-0.0496610561406,-0.0387349338517,-0.193831273703,-0.0623015184526,0.217187200367,-0.123725969142,-0.0878524132158,-0.121582975103,-0.0719854851647,-0.03444108345,0.0513824276594,-0.180530863526,-0.148981738546,-0.158816380701,-0.0594120124204,0.0858812973866,-0.197250070453,-0.960856222204,-0.289462761746,-0.31424342879,-0.0649257721382,0.550495447494,-0.76744092489,0.114209259164,-0.49505898373,-0.715516218409,-0.181505942793,-0.818969368578,-0.888723357308,-0.906749089512,-0.717179079858,-0.245288423097,0.934185027411,-0.486485444258,0.362637778827,-1.68631676121,0.523238535587,0.208192028603,1.81379551757,-1.35514622372,0.539354012514,-0.999356539685,-0.766323413723,0.492894960727,5.03524043478,2.9231467049,-0.80877113337,-1.30389749303,-1.42289597577,-0.178242169219,-0.999713991437,0.738271415136,0.145130214311,0.17508944405,-1.13327182478,-0.206386307113,-1.24381708261,-1.41340332617,-0.88724465618,-1.07361721817,0.140791075172,0.300233796765,-0.786268347043,-0.801682560141,-1.10931186195,-0.181758574334,-1.00585654811,0.316491090762,-1.25502415183,0.80559543194,0.736119799258,-1.0209910545,-0.393739995838,1.67231817399,3.37118328049,0.612539279269,0.168737773341,-0.94656285351,1.94786894071,-0.454420103583,1.11603704114,-1.53760378516,-0.567645402976,-0.736069853061,-1.13474741978,-1.02639860031,7.34221313587,1.56947464732,-0.867231969364]
    y = []
    for line in data:
        y.append(np.dot(w,line))
    
    s = 0
    for i,_ in enumerate(y):
        s += (y[i]-ytest[i])*(y[i]-ytest[i])

    print(s,len(y))

    writeData(y)
