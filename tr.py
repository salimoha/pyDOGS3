import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as pl
import matplotlib.pyplot as plt
import os
import math


def readInputFile(filePath):
    #    retVal = []
    #    with open(filePath, 'rb') as csvfile:
    #        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #        for row in filereader:
    #            retVal.append([int(row[0]), int(row[1]), int(row[2])])

    retVal = []
    with open(filePath) as file:
        line = file.readline()
        arr = [float(a) for a in line.split(',')]
        #        retVal.append(file.readline())
        retVal.append(arr)
    return retVal[0]


def transient_removal(x):
    N = len(x)
    k = int(N / 2)
    y = np.zeros((k, 1))
    for kk in np.arange(k):
        y[kk] = np.var(x[kk + 1:]) * 1.0 / (N - kk - 1.0)
    y = np.array(-y)
    ind = y.argmax(0)
#     print('index of transient point in the signal:')
#     print(ind)
    return ind

def transient_drag(x,Safety=13):
    
    index = 0; IND=0;  INDEX=0
    viol_safe =0;
    xc=np.copy(x)
    for idx in range(100,len(x),2000):
        index = transient_removal( x[IND:idx+IND])
        index0 = np.copy(index)
        IND = IND + int(index)

        if index <5:
            viol_safe = viol_safe + 1
#             continue
            if viol_safe >=Safety:
                xc=x[int(INDEX):] 
                print('index of transient start',INDEX)
                return INDEX,xc
            else:
                if index0!=0:
                    INDEX = IND + index
                    
                else:
                    index = 100;
                    IND = IND + index
                    INDEX = np.copy(IND)
# x = readInputFile(data1FilePath)

def relax_data(xc,numSTD=6):
#     x = pd.Series(xc,index='Drag')
    thershold = np.median(xc)+np.std(xc)*numSTD
    idt = xc>thershold 
    xc[idt]=np.nan
    idt2 = xc< -thershold
    xc[idt2]=np.nan
    return xc

def data_moving_average(ym,mm=40):
    #reducing the size of data. Since it is stationary its std and mean do not change
    NN=len(ym);
    ym2 = ym[:int(math.floor(NN/mm)*mm)]
    #     print(ym2)
    x = pd.DataFrame(np.reshape(ym2,(math.floor(NN/mm),mm)))
    y = np.mean(x, axis=1)
    #     print(y)
    return y

