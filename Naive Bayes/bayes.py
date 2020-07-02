# -*- coding: utf-8 -*-
# @Time  : 2020/7/1 10:49
# @Author : VMice
# ==========================

from numpy import *
import random


def loadDataSet():
    '''
    :return: returns the generated dataset
    '''
    # 'postingList' is a division of a sentence
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # classVec: 0 is ham, 1 is spam
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    '''
    :param dataSet: dataset to be processed
    :return: a set in a dataset
    '''
    vocabSet = set([]) # set data
    for document in dataSet:
        vocabSet = vocabSet | set(document) # or operation
    return list(vocabSet) # set data becomes list data


def setOfWordVec(vocabList, inputSet):
    '''
    :param vocabList: existing data feature list
    :param inputSet: data to process
    :return: count of occurrence of data feature, it is list
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # returnVec[vocabList.index(word)] = 1
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word: %s is not in my vocabulary!')
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    :param trainMatrix: feature vector dataset
    :param trainCategory: category label
    :return: conditional expectation under various labels and probability of 'spam'
    '''
    numTrainDocs = len(trainMatrix) # number of dataset
    numWords = len(trainMatrix[0]) # number of feature
    pAbusive = sum(trainCategory) / float(numTrainDocs) # probability of 'spam'
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # p0Denom = p1Denom = 0.0
    # count number of feature
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = p1Denom = 2.0 # count total of feature
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p0Vect = p0Num / p0Denom
    # p1Vect = p1Num / p1Denom
    # calculate conditional probability
    p0Vect = log(p0Num / p0Denom)
    p1Vect = log(p1Num / p1Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vecClassify, p0Vec, p1Vec, pClass):
    '''
    :param vecClassify: feature vector
    :param p0Vec: probability feature of '0' kind
    :param p1Vec: probability feature of '1' kind
    :param pClass: probability of '1' kind
    :return:
    '''
    p1 = sum(vecClassify * p1Vec) + log(pClass)
    p0 = sum(vecClassify * p0Vec) + log(1.0 - pClass)
    return 1 if p1 > p0 else 0


def testingNB():
    '''
        text classification cases
    :return: none
    '''
    listPosts, listClass = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for positnDoc in listPosts:
        trainMat.append(setOfWordVec(myVocabList, positnDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClass))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWordVec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWordVec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):
    '''
    :param bigString: text to parse
    :return: the result of parsing is divided into each word
    '''
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    '''
        filtering spam cases
    :return:
    '''
    filepath = 'E:\\Code\\Python\\machinelearninginaction\\Ch04\\email\\%s\\%d.txt'
    docList = [] # text feature data set
    classList = [] # text category label
    for i in range(1, 26): # 25 examples
        wordList = textParse(open(filepath % ('spam', i)).read()) # read and parse
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open(filepath % ('ham', i)).read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) # create feature set
    ''' split training set and test set '''
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    ''' using training set to train learners '''
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWordVec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    ''' using test set to calculate the empirical error '''
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWordVec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: %.2f %%' % (float(errorCount) / len(testSet) * 100))
