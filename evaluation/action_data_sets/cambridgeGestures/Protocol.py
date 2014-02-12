'''
Created on Mar 31, 2011
@author: Stephen O'Hara
Colorado State University
All Rights Reserved

High level functions for implementing KTH data partitioning.

Standard partition is set5 = training, sets 1-4 for testing.
It is also instructive to try leave-one-set-out protocols,
so a partitioning for that strategy is provided as well.
'''
import proximityforest as pf
import scipy

def partitionGestures(vidData, leaveout_set=0):
    '''partitions gesture data into a leave-one-set-out test and training set'''
    testvids = []
    testlabels = []
    trainvids = []
    trainlabels = []
    for i in range(5):
        if i==leaveout_set:
            testvids = vidData[i].getData()
            testlabels = vidData[i].getLabels()
        else:
            tv = vidData[i].getData()
            tl = vidData[i].getLabels()
            trainvids += tv
            trainlabels += tl
            
    return ( (trainvids,trainlabels),(testvids,testlabels))

def partitionCambridgeProtocol(vidData, test_set=0):
    '''
    training set is set 5 (4, using zero-based index),
    test set 0-3 specified by parameter.
    '''
    set5 = vidData[4]
    trainvids = set5.getData()
    trainlabels = set5.getLabels()
    
    setX = vidData[test_set]
    testvids = setX.getData()
    testlabels = setX.getLabels()
    
    return ( (trainvids,trainlabels),(testvids,testlabels))
    
def cambridgeStandardClassification(GDat, treeClass=pf.TensorSubspaceTree, N=9, Tau=21, method="knn", K=1, **kwargs):
    '''
    Performs the standard cambridge classification protocol, using
    Set 5 for training, and testing on Sets 1-4.
    @param GDat: The cambridge gestures tracklet data structure generated
    via loadGestureData() function in the DataSource module, or via unPickling
    a previously stored object.
    '''        
    (train, _) = partitionCambridgeProtocol(GDat)
    tracklet_data = pf.build_simple_TD_Structure(train[0], train[1])
    #step 1: train a forest on set 5
    forest = pf.TensorSubspaceForest(N, tracklet_data, treeClass=treeClass, Tau=Tau, **kwargs)
    forest.addList(range(len(train[0])), train[1])
    
    #step 2: test against each set 1-4 independently
    errList = []
    confusionList = []
    for i in range(4):
        print "Classifying Set%d vs. Set5..."%(i+1)
        testvids = GDat[i].getData()
        testlabels = GDat[i].getLabels()
        pfa = pf.ProximityForestAnalysis(forest)
        (err, cfList) = pfa.classifyAll(testvids, testlabels, 9, method, K, verbose=False)
        #cfm = pf.confusionMatrix(cfList, [20]*9)  #a single set has 20 repetitions of each of 9 gestures
        errList.append(err)
        confusionList.append(cfList)
        print
        print "Set%d error rate: %f"%(i+1,err)
    print        
    avg_err = scipy.mean(errList)
    print "Average Error Rate = %f"%avg_err
    print "Error by Set = %s"%str(errList)
    return (avg_err, errList, confusionList)
    
if __name__ == '__main__':
    pass
