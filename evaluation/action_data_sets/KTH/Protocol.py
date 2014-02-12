'''
Created on Mar 31, 2011
@author: Stephen O'Hara
Colorado State University
All Rights Reserved

High level functions for implementing KTH data partitioning
'''

def partitionKTHProtocol(vidData, bundleValidation=True):
    '''partitions kth data into test and training sets
    By the KTH classification protocol from Schuldt, Laptev, et al 2004
    Training:   person11, 12, 13, 14, 15, 16, 17, 18
    Validation: person19, 20, 21, 23, 24, 25, 01, 04
    Test:       person22, 02, 03, 05, 06, 07, 08, 09, 10.
    @param bundleValidation: Set to True if you want to return
    only test and training data sets, with the validation set
    considered to be part of the training set.
    @return: A tuple of tuples. Either ( trainT, testT, validT) or
    (trainT, testT) if bundleValidation = True. Each entry, trainT e.g.,
    is a tuple of (videolist, labels).
    '''
    testSet_idxs = [22, 2, 3, 5, 6, 7, 8, 9, 10]  #1 based
    trainSet_idxs= [11, 12, 13, 14, 15, 16, 17, 18] #1 based
    validSet_idxs= [19, 20, 21, 23, 24, 25, 01, 04] #1 based
    
    trainvids = []
    trainlabels = []
    testvids = []
    testlabels = []
    validvids = []
    validlabels = []
    
    for i in range(25): #25 subjects, i = 0..24
        vs = vidData[i].getData()
        ls = vidData[i].getLabels()
        
        idx = i + 1 #convert to 1-based set index
        if idx in testSet_idxs:
            testvids += vs
            testlabels += ls
        elif idx in trainSet_idxs:
            trainvids += vs
            trainlabels += ls
        elif idx in validSet_idxs:
            validvids += vs
            validlabels += ls
        else:
            raise ValueError("Index %d is not in any of the test,train,or validation lists."%idx)
    
    if bundleValidation:
        trainvids += validvids
        trainlabels += validlabels
        return ( (trainvids,trainlabels),(testvids,testlabels)) 
    else:
        return ((trainvids,trainlabels),(testvids,testlabels),(validvids,validlabels)) 