# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt


# 设置全局变量，决策单元，分支单元，和连接的样式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round', fc='0.8')
arrow_args = dict(arrowstyle='<-')

# 注解单元，利用上面的全局变量设置整个图得样式
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)
    
# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=False)
#     plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()



def getNumLeafs(myTree):
    # 用递归来字节点（结果）的个数
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    # 递归求深度，二叉树的深度
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    # 创建两个树的实例数据
    listOfTrees = [{'no surfacing': {0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}},
                   {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}}
                   ]
    return listOfTrees[i]

# 填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


# 填充注解树
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  # 树的叶子
    depth = getTreeDepth(myTree)    # 树得深度
    firstStr = next(iter(myTree))   # decision node
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /2.0/plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 利用plotNode画出决策单元
    secondDict = myTree[firstStr]   # 进入下一个节点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  # y偏移
    for key in secondDict.keys():   # 来个循环递归对所有节点绘制，这里的思路和上面求深度和数量得函数是一样得
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


# 做图
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建画板
    fig.clf()    # 清空画板
    axprops = dict(xticks=[], yticks=[])    # 参数字典
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) # 去掉X ，Y轴
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW    # X偏移
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5, 1.0), '')
    plt.show()
