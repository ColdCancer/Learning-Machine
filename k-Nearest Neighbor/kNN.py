from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(Xo, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    diffMat = tile(Xo, (dataSetSize, 1)) - dataSet
    powerDiffMat = diffMat ** 2
    tmpDistances = powerDiffMat.sum(axis=1)
    distances = tmpDistances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def file2matrix(filename):
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
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m , 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    testRate = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * testRate)
    errorCount = 0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m],
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real a %d."
              % (classifierResult, datingLabels[i]))

        if(classifierResult != datingLabels[i]):
            errorCount += 1

    print("the error rate is: %f %%" % (errorCount / float(m) * 100))