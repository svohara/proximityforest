'''
Created on Mar 31, 2011
@author: Stephen O'Hara
Colorado State University
All Rights Reserved

Protocols for partitioning UT data sets (Tower, Interactions)
'''

def partitionLOSO(vidData, leaveout_set=0):
    '''
    partitions data into a leave-one-set-out test and training set
    @param vidData: A list of data source objects (like UTTowerData), one per set/sequence
    @param leaveout_set: The index of which set/sequence in the vidData list to use for testing, the
    rest will be for training'
    @return: Tuple of tuples, ( (trainvids,trainlabels),(testvids,testlabels))
    '''
    testvids = []
    testlabels = []
    trainvids = []
    trainlabels = []
    for i in range(len(vidData)):
        if i==leaveout_set:
            testvids = vidData[i].getData()
            testlabels = vidData[i].getLabels()
        else:
            tv = vidData[i].getData()
            tl = vidData[i].getLabels()
            trainvids += tv
            trainlabels += tl
            
    return ( (trainvids,trainlabels),(testvids,testlabels))
    

if __name__ == '__main__':
    pass
