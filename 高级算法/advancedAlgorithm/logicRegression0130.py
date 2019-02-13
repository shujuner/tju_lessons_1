# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:08:27 2019

@author: jiaojie
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import re
import random

# 传入数据，返回b0，b1的估计值
def rmse(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
def mae(y_test,y_pred):
    return np.mean((y_test-y_pred)**2)
def predict(x, b0, b1):
    return b0 + x*b1
def filterDataSet(x,time,s,k):
    num = len(x)
    
    xArr = []; yArr = []
    for line in x_train:
        lineArr =[]
        curLine = line.strip().split(',')   
        yArr.append(float(curLine[1]))
        for i in range(2,numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
    return xArr, yArr
if __name__ == "__main__":
    #for i in range(1,14):
        var1  = 'waterdata3.csv'
        
        df=pd.read_csv(var1)
        trainCol=['time','xiaoxi_out','lengshuijiang_add','xinhua_add','zhexi_add','zhexi_in']
        X=df[trainCol]
        X=X.values
        split=(int)(len(X)*0)
       
        rightTest=(int)(len(X)*0.9)
        train=X[split:rightTest]
        test=X[rightTest:len(X)]
        print(len(train))
        print(len(test))
        print(re.split('\/| |:',train[0][0]))
        #
       # print(x_train['xiaoxi_out'])
        #regr = linear_model.LinearRegression()
        #regr.fit(x_train, y_train)
        #yPred = regr.predict(x_test)
        #print(rmse(y_test,yPred))
 