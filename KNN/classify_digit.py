import numpy as np
from matplotlib import pyplot as plt
import operator
from os import listdir

"""
手写识别系统
1.收集数据，提供文本文件
2.准本数据： 编写函数classify0()，将图片格式转化为分类器使用得list 格式
3.分析数据：Python命令提示符中检查数据，确保它符合要求
4.训练算法：K-NN算法无需训练
5.测试算法：编写函数用部分数据集作为测试样本测试分类器
6.使用算法：构建完整得应用程序，从图像中提取数字，并完成识别数字
"""

# load img, img to array
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def classify0(inX, dataSet, labels, k):
    # 求欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDiffMat = sqDiffMat.sum(axis=1)
    distances = sqDiffMat ** 0.5
    # 排序
    sortedDistancies = distances.argsort()
    # 选与inX距离最及得K个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistancies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' %(classifierResult, classNumStr))
        if (classifierResult != classNumStr) : errorCount += 1.0
    print('\nthe total number of errors is: %d' % errorCount)
    print('\nthe total error rate is: %2f' % (errorCount/float(mTest)))