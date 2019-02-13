
# -*- coding: utf-8 -*-

"""

Created on Tue Sep  5 21:21:58 2017

@author: wjw

模拟产生数据集，然后再进行拟合

"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cross_validation import train_test_split

def nomalization(X):#不归一化时梯度下降时数值太大，报错
    maxX = max(X)
    minX = min(X)
    normalized_X = []
    for x in X:
        normalized_X.append((x-minX)/(maxX-minX))
    return normalized_X 

def bgd(x,init_w,y,iter_size,lr):
    w = init_w
    x=np.array(x)
    m = x.shape[0]
    for i in range(iter_size):
        predict = x.dot(w)
        grad = x.T.dot((predict - y)) / m * lr
        w -= grad 
    print (w)

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split(','))
    print(numFeat)
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split(',')   
        yArr.append(float(curLine[1]))
        for i in range(2,numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
    return xArr, yArr 

if __name__ == "__main__":
    
    X, Y = loadDataSet('waterdata1.csv')
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=1)

    #X_train = nomalization(X_train)
    init_w=[0,0,0,0]
    iter_size=1000
    lr=0.0001
    for i in range(len(X_train)):
        bgd(X_train[i],init_w,Y_train[i],iter_size,lr)

