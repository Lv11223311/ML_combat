import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))    # 初始化预测向量
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 设置flag的思想
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray  # 得到预测向量


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 0)))
    minError = np.inf
    for i in range(n):  # 第一层循环：对每个特征维度操作
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
        stepSzie = (rangeMax - rangeMin) / numSteps     # 设置步长
        for j in range(-1, int(numSteps)+1):    # 第二层循环：对每个步长做操作
            for inequal in ["lt", "gt"]:    # 这个地方不明白为什么是lt, gt
                threshVal = (rangeMin + float(j) * stepSzie)    # 阈值递增函数
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)    # 简易决策树分类，得到根据第i维的分类
                errArr = np.mat(np.ones((m, 1)))    # 初始化错误矩阵为1
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr    # 加权，这个是boost的评价指标
                # print('split: dim %d, thresh %.2f, thresh inequal:%s, the weighted error is %.3f'\
                #       % (i, threshVal, inequal, weightedError))     # [debug]
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化权重向量D
    aggClassEst = np.mat(np.zeros((m, 1)))  # 初始化误差矩阵
    for i in range(numIt):      # 循环几次就是几个弱分类器
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print('D:{}'.format(D.T))  # [debug]
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # alpha = 0.5 * log((1-e)/e)
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
       # print('ClassEst:{}'.format(classEst.T))
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
       # print('aggClassEst:', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error:', errorRate, '\n')
        if errorRate == 0.0:
            break
    return weakClassArr

# 集成弱分类器，
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = 1.0
        else:
            delX = xStep; delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    plt.title('ROC curve for adaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is:', ySum, xStep)

