# -*- coding: utf-8 -*-
# @Time  : 2020/6/30 16:25
# @Author : VMice
# ==========================

from graphviz import Digraph

def searchTree(tree, dot, leafCount):
    '''
    :param tree: operate on this tree
    :param dot: dot object, used to draw a directed graph
    :param leafCount: how many leaf nodes have been visited
    :return: The root name of the tree to be operated on
    '''
    rootName = list(tree.keys())[0]
    root = tree[rootName]
    dot.node(rootName, color='blue')
    for index in root.keys():
        if type(root[index]).__name__ != 'dict': # this is the leaf node
            leafCount += 1
            childName = root[index] + str(leafCount)
            dot.node(childName, root[index], color='red')
            dot.edge(rootName, childName)
        else:
            childName = searchTree(root[index], dot, leafCount)
            dot.edge(rootName, childName)
    return rootName

def plotTree(Mytree):
    dot = Digraph('MyTree', 'Decision Tree')
    searchTree(Mytree, dot, 0)
    dot.view()
