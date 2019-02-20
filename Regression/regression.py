# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:20:12 2019

@author: bioknow
"""

import numpy as np

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')-1)
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(curLine[i])
        dataMat.append(lineArr)
        labelMat.append(curLine[-1])
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
       print('This matrix is singular, cannot do inverse')
       return 
   # 最小二乘(OLS)
   ws = xTx.I * (xMat.T * yMat)
   return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    """局部加权线性回归：给待预测的每个点赋予一定的权重，
        然后在这个子集上基于最小均方差来进行普通的回归，这种算法每次预测均需要事先选取出对应的数据子集"""
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws 


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat