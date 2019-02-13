# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 21:25:46 2019

@author: kg
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
 
# 传入数据，返回b0，b1的估计值
def rmse(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
def mae(y_test,y_pred):
    return np.mean((y_test-y_pred)**2)

def trainTestSplit(X,test_size=0.1):
    X_num=X.shape[0]
    train_index=range(X_num)
    test_index=[]
    test_num=int(X_num*test_size)
    for i in range(test_num):
        randomIndex=int(np.random.uniform(0,len(train_index)))
        test_index.append(train_index[randomIndex])
        del train_index[randomIndex]
    #train,test的index是抽取的数据集X的序号
    train=X.ix[train_index] 
    test=X.ix[test_index]
    return train,test
def fitSLR(x, y):
    n = len(x)
    dinominator = 0 #分母
    numerator = 0   # 分子
    for i in range(0, n):
        numerator += (x[i] - np.mean(x))*(y[i] - np.mean(y))
        dinominator += (x[i] - np.mean(x))**2
    
    print("numerator:"+str(numerator))
    print("dinominator:"+str(dinominator))
    
    b1 = numerator/float(dinominator)
    b0 = np.mean(y)/float(np.mean(x))
    
    return b0, b1
 
 
def predict(x, b0, b1):
    return b0 + x*b1
 
if __name__ == "__main__":
    df=pd.read_csv('waterdata.csv')
    trainCol=['xiaoxi_out','lengshuijiang_add','xinhua_add','zhexi_add']
    target=['zhexi_in']
    X,Y=df[trainCol],df[target]
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=0)
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    yPred = regr.predict(x_test)
    print(rmse(y_test,yPred))
 