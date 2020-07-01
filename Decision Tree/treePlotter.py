# -*- coding: utf-8 -*-
# @Time  : 2020/6/30 16:25
# @Author : VMice
# ==========================

from graphviz import Digraph

nodeCount = {}

def searchTree(tree, dot):
    '''
    :param tree: operate on this tree
    :param dot: dot object, used to draw a directed graph
    :return: The root name of the tree to be operated on
    '''
    rootName = list(tree.keys())[0]
    nodeCount[rootName] = nodeCount.get(rootName, 0) + 1
    rootTag = rootName + str(nodeCount[rootName])
    dot.node(rootTag, rootName, color='blue')
    childen = tree[rootName]
    for index in childen.keys():
        if type(childen[index]).__name__ != 'dict': # this is the leaf node
            nodeCount[childen[index]] = nodeCount.get(childen[index], 0) + 1
            childTag = childen[index] + str(nodeCount[childen[index]])
            dot.node(childTag, childen[index], color='red')
            dot.edge(rootTag, childTag)
        else:
            childTag = searchTree(childen[index], dot)
            dot.edge(rootTag, childTag)
    return rootTag

def plotTree(Mytree):
    dot = Digraph('MyTree', 'Decision Tree')
    searchTree(Mytree, dot)
    dot.view()
