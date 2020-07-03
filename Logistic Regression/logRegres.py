# -*- coding: utf-8 -*-
# @Time  : 2020/7/3 10:28
# @Author : VMice
# ==========================

from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    file = open('testSet.txt')
    for line in file.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatIn)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        delta = (labelMat - h)
        weights += alpha * dataMatrix.transpose() * delta
    return weights

