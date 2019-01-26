#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:40:40 2019

@author: bo
"""

from math import log
import operator

# 计算香农熵
def calcShannoEnt(dataSet):
    numEntries = len(dataSet)  # 信息量
    labelCounts = {} # 分类字典
    for featVec in dataSet:
        currentLabel = featVec[-1] # 提取标签
        if currentLabel not in labelCounts.keys(): # 这个地方如果用in dict 而不是in dict.keys()会更快一点
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0 # 初始化 H
    for key in labelCounts:
        prob = float(labelCounts[key] / numEntries)
        shannonEnt -= prob * log(prob, 2) # 期望

    return shannonEnt


# 创造数据集
def creatDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 划分数据集
def splitDataSet(dataSet, axis, value):
    """paras:
    dataSet: 准备划分得数据集
    axis: 划分数据集得特征所在维度
    value: 特征的返回值
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVac = featVec[:axis]
            reducedFeatVac.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVac)
    return retDataSet

# 选择最好的划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannoEnt(dataSet) # 基线信息熵
    bestInforGain = 0.0  # 初始化信息增益
    bestFeature = -1 # 初始化最好得划分特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet)) # 子数据集包含信息的百分比
            newEntropy += prob * calcShannoEnt(subDataSet) # 子数据集所包含得信息熵
        infoGain = baseEntropy - newEntropy # 计算信息增益
        if (infoGain > bestInforGain):
            bestInforGain = infoGain
            bestFeature = i # bestFeature index
    return bestFeature



def majorityCnt(classList):
    classCount = {} # initial class count
    for vote in classList:
        if vote not in classCount.keys():   # count every vote
            classCount[vote] = 0
        classCount[vote] += 1

        sortedClassCount = sorted(classCount.iteritems(),\
        key=operator.itemgetter(1), reverse=True)   # group by count
        return sortedClassCount[0][0]  # return zhe biggest Class

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels.pop(bestFeat)
    myTree = {bestFeatLabel:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,\
                                                  bestFeat, value), subLabels)
    return myTree
