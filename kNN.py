# -*- coding: utf8 -*-
import numpy as np


# create a dataset 
def createDataset():
    group = np.array([[1.0,0.9],[1.0,1.0],[0.1,0.2],[0.0,0.1]])
    lable = ['A','A','B','B']
    return group , lable

def KNNClassify(newinput , dataset , lable , k):
    numsamples = dataset.shape[0]
    
    #step 1 , calculate Euclidean Distance
    diff = np.tile(newinput, (numsamples,1)) - dataset
    squredDiff = diff ** 2
    squredDist = np.sum(squredDiff , axis = 1)
    distance = squredDist ** 0.5
    
    #step 2 , sort the distance 
    SortedDistance = np.argsort(distance)
    
    #step 3 choose the min k distance and count the times lable occur
    ClassCount = {}
    for i in xrange(k):
        votelable = lable[SortedDistance[i]]
        ClassCount[votelable] = ClassCount.get(votelable , 0) + 1
        
    #step 4 reuturn the max vote class
    maxCount = 0
    for key,value in ClassCount.items():
        if value > maxCount:
            maxCount = value
            maxindex = key
    
    return maxindex