# -*- coding: utf-8 -*-
# @Time  : 2020/6/30 8:15
# @Author : VMice
# ==========================

from math import log
import operator

def calcShannonEnt(dataSet):
    '''
    :param dataSet: data set to calculate information entropy
    :return: the value of information entropy
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    ''' types of statistical samples '''
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    ''' calculate the the information entropy '''
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key] / numEntries)
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    '''
    :return: create the result of the dataset
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    featName = ['no surfacing', 'flippers']
    return dataSet, featName

def splitDataSet(dataSet, axis, value):
    '''
    :param dataSet: dataset to be split
    :param axis: what are the characteristics
    :param value: the value of the feature
    :return: in the dataset, the eigenvalue is a set of 'value'
    '''
    retDataSet = [] # return list
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    :param dataSet: the data set to calculate the information gain
    :return: Optimal information gain feature
    '''
    numFeatures = len(dataSet[0]) - 1 # number of features
    baseEntropy = calcShannonEnt(dataSet) # global information entropy
    bestInfoGain, bestFeatrue = 0.0, -1
    for i in range(numFeatures): # loop each feature
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals: # Loop the value of each feature
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain): # the greater the information gain, the higher the purity
            bestInfoGain = infoGain
            bestFeatrue = i
    return bestFeatrue

def majorityCnt(classList):
    '''
    :param classList: labels list
    :return: most frequent label
    '''
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # most frequent label name

def createTree(dataSet, featName):
    '''
    :param dataSet: the dataset to create the decision tree
    :param featName: there is name of feature for the no node
    :return: the decision tree created under this dataset
    '''
    classList = [example[-1] for example in dataSet] # labels in the dataset
    if classList.count(classList[0]) == len(classList): # there is only one label
        return classList[0] # returns the label as a leaf node
    if len(dataSet[0]) == 1: # only the label is left
        return majorityCnt(classList) # most frequent label as a leaf node
    bestFeat = chooseBestFeatureToSplit(dataSet) # feature of maximum information gain
    bestFeatName = featName[bestFeat] # the name of this feature
    myTree = {bestFeatName: {}} # tree structure
    del(featName[bestFeat]) # delete the feature
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) # get all values of this feature
    for value in uniqueVals: # loop all values of this feature
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        subFeatName = featName[:]
        myTree[bestFeatName][value] = createTree(subDataSet, subFeatName)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''
    :param inputTree: decision tree for training completion
    :param featLabels: existence label of decision tree
    :param testVec: data to be predicted
    :return: predicted value
    '''
    global classLabel
    firstStr = list(inputTree.keys())[0] # the name of the root of the tree
    secondDict = inputTree[firstStr] # a layer of leaves beneath the tree
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else : # this subtree is a leaf
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename): # save decision tree
    import pickle
    file = open(filename, 'wb')
    pickle.dump(inputTree, file)
    file.close()

def grabTree(filename): # get decision tree
    import pickle
    file = open(filename, 'rb')
    content = pickle.load(file)
    file.close()
    return content

