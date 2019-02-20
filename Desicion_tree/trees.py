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
        shannonEnt -= prob * log(prob, 2)
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
# 划分数据集得意义：增强数据得一致性。对分类有好处
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


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannoEnt(dataSet)  # 基线信息熵
    bestInforGain = 0.0  # 初始化信息增益
    bestFeature = -1  # 初始化最好得划分特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))  # 子数据集包含信息的百分比
            newEntropy += prob * calcShannoEnt(subDataSet)  # 信息熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if (infoGain > bestInforGain):
            bestInforGain = infoGain
            bestFeature = i  # bestFeature index
    return bestFeature


def majorityCnt(classList):
    classCount = {} # initial class count
    for vote in classList:
        if vote not in classCount.keys():   # count every vote
            classCount[vote] = 0
        classCount[vote] += 1

        sortedClassCount = sorted(classCount.items(),
        key=operator.itemgetter(1), reverse=True)   # group by count
        return sortedClassCount[0][0]  # return zhe biggest Class


def createTree(dataSet, labels):
    # 递归得思想
    classList = [example[-1] for example in dataSet]    # 提取数据集最后一column作为类别清单
    if classList.count(classList[0]) == len(classList): # 基线条件，清单内剩余如果是同一类，返回这类
        return classList[0]
    if len(dataSet[0]) == 1:    # 另一个基线条件，到最后也没统一，则返回计数最多得一类
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)   # 获得分割数据集最好得特征维度(index)
    bestFeatLabel = labels.pop(bestFeat)    # 将最佳特征分离出来作为label
    myTree = {bestFeatLabel:{}}     # 构建树
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                  bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)