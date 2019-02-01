"""
Created on Sat Jan 31 11:40:40 2019

@author: bo
"""

from numpy import *

# 1.从文本构建词向量
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop','posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   # 1代表侮辱性词汇， 0代表正常词汇
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # 创造一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 求并集，换句话说就是将文本去重
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """词集模型
    input：vocabList 词汇表
           inputSet 输入文档
    output： 与词汇表对应的词汇向量，一个0-1矩阵"""
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('this word: %s is not in my Vocabulary!' % word)
    return returnVec


def bagOfwords2VecMN(vocabList, inputSet):
    """词袋模型
    input：vocabList 词汇表
           inputSet 输入文档
    output： 与词汇表对应的词汇向量，一个单词频数矩阵"""
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 2.训练算法：从词向量计算概率
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)  # 初始化p(0)和p(1)的词频数
    p0Denom = 2.; p1Denom = 2.    # 初始化文档的总词数
    for i in range(numTrainDocs):   # 遍历所有文档，记录累加随机变量x=1,x=0的词频数和文档总词数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify * p1Vect) + log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + log(1. - pClass1)
    return 1 if p1 > p0 else 0


def testingNB():
    listOposts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pAb))


# 文件解析以及完整的垃圾邮件测试函数
def textParse(bigSting):    # 去掉标点符号，将所有单词转化为小写
    import re
    listOfTokens = re.split(r'\w*', bigSting)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []; classList = []; fullText = []  # 初始化文档列表，分类列表，正文单词列表
    for i in range(1, 26):  # 读取50封邮件，分类
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet = []
    for i in range(10):  # 选10个样本作为测试样本
        randIndex = int(random.uniform(0, trainingSet))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:    # 用训练集训得到先验概率
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:    # 计算后验概率，计算错误率
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount)/len(testSet))


# 挑出30个高频词
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:    # 除去高频词,正确率增长20%
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfwords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNB0(array(trainMat, docList[docIndex]))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfwords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount)/len(testSet))
    return vocabList, p0v, p1v


def getTopWords(ny, sf):
    import operator
    vocabList, p0v, p1v = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0v)):
        if p0v[i] > -6.0:
            topSF.append((vocabList[i], p0v[i]))
        if p1v[i] > -6.0:
            topNY.append((vocabList[i], p1v[i]))
    sortedSF = sorted(topSF, key = lambda pair: pair[1], reverse=True)
    print('**SF**' * 10)
    for item in sortedSF:
        print(item[0])
    print('**NY**'*10)
    sortedNY = sorted(topNY, key = lambda pair: pair[1], reverse=True)
    for item in sortedNY:
        print(item[0])
