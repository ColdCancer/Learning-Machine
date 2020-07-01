# -*- coding: utf-8 -*-
# @Time  : 2020/7/1 10:49
# @Author : VMice
# ==========================

from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWordVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in inputSet:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!')
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # p0Denom = p1Denom = 0.0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
    # p0Vect = p0Num / p0Denom
    # p1Vect = p1Num / p1Denom
    p0Vect = log(p0Num / p0Denom)
    p1Vect = log(p1Num / p1Denom)
    return p0Vect, p1Vect, pAbusive
