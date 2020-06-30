# -*- coding: utf-8 -*-
# @Time  : 2020/6/29 20:44
# @Author : VMice
# ==========================

from numpy import *
from os import listdir
import operator


def createDataSet(): # Creating a test dataset
    '''
    :return: creating a test data sets and return.
    '''
    ''' 'group' is a test data '''
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    ''' 'labels' is the label of 'group' result '''
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(Xo, dataSet, labels, k):
    '''
    :param Xo: the data to be classified on the data sets.
    :param dataSet: data sets used in kNN algorithm.
    :param labels: labels is label of result of data sets.
    :param k: parameter K in kNN algorithm.
    :return: Among the K label closest to the dataset,
            the label that appears the most frequently in the label classified.
    '''
    dataSetSize = dataSet.shape[0] # the dataset size

    ''' The Euclidean distance from the source point to each data set is calculated '''
    diffMat = tile(Xo, (dataSetSize, 1)) - dataSet
    powerDiffMat = diffMat ** 2
    tmpDistances = powerDiffMat.sum(axis=1)
    distances = tmpDistances ** 0.5
    sortedDistIndicies = distances.argsort() # argsort: returns the original position after sorting

    ''' The K labels with the smallest distance are counted '''
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    ''' In the statistical results, the number of labels is sorted from large to small '''
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # return the largest number of labels as the answer


def file2matrix(filename):
    '''
    :param filename: file path in case
    :return: data sets and labels in files
    '''
    file = open(filename)

    arrayOflines = file.readlines()
    numberOfLines = len(arrayOflines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOflines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''
    :param dataSet: data set to be normalized
    :return: results of normalization
    '''

    ''' formula: (x - min) / (max - min) '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m , 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    '''
        appointment matching cases
    :return: none
    '''
    testRate = 0.1 # test set scale
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * testRate) # total number of test sets
    errorCount = 0

    ''' prediction results for each data '''
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m],
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real a %d."
              % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1
    print("the error rate is: %f %%" % (errorCount / float(m) * 100)) # prediction error rate

def img2vector(filename):
    '''
    :param filename: file path in case
    :return: the result of changing the matrix shape to (1, 1024)
    '''
    returnVect = zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        lineStr = file.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    '''
        Handwriting recognition case
    :return: none
    '''
    filepath = 'E:/Code/Python/machinelearninginaction/Ch02/digits/'

    ''' getting data and labels '''
    hwLabels = [] # label list corresponding to data
    trainingFileList = listdir(filepath + 'trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0]) # label corresponding to data
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(filepath + 'trainingDigits/' + fileNameStr)

    ''' empirical error of algorithm learners '''
    testFileList = listdir(filepath + 'testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(filepath + 'testDigits/' + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 5)
        print('the classifier came back with: %d, the real is %d.' % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1

    print('\nthe total number of error is: %d.' % errorCount)
    print('\nthe error rate is: %f.' % (errorCount / float(mTest)))

