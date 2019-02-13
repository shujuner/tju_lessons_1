# -*- coding:utf-8 -*-

from matplotlib.font_manager import FontProperties

import matplotlib.pyplot as plt

import numpy as np
import time
import sys

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

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))										#创建权重对角矩阵
    for j in range(m):                      							#遍历数据集计算每个样本的权重
        diffMat = testPoint - xMat[j, :]     							
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)										
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))							#计算回归系数
    return testPoint * ws



def lwlrTest(testArr, xArr, yArr, k=1.0):

	m = np.shape(testArr)[0]											#计算测试数据集大小

	yHat = np.zeros(m)	

	for i in range(m):													#对每个样本点进行预测

		yHat[i] = lwlr(testArr[i],xArr,yArr,k)

	return yHat

def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat							#根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def rssError(yArr, yHatArr):
	return ((yArr - yHatArr) **2).sum()



if __name__ == '__main__':
    abX, abY = loadDataSet('waterdata1.csv')
    yHat01 = lwlrTest(abX, abX, abY, 0.1)
    yHat1 = lwlrTest(abX, abX, abY, 1)
    yHat10 = lwlrTest(abX, abX, abY, 10)
    print('k=0.1时,误差大小为:',rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:',rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:',rssError(abY[0:99], yHat10.T))