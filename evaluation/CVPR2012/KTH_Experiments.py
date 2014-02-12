'''
Created on Apr 19, 2012
@author: Stephen O'Hara
A consolidation and refactoring of the code required to produce
the experimental results on KTH Data reported in my CVPR2012 paper.

Note, since there is a stochastic nature in the subspace forest
construction, numerical results may differ slightly from those
published.

Code is copyright by the authors, all rights reserved. You may use
the code for non-commercial purposes if you cite my paper. Please
contact the authors if you wish to use this code for commercial
purposes or if you would like to include this code (or derivative)
in another work, such as a computer vision toolkit, etc.

Please cite:
Stephen O'Hara and Bruce A. Draper, "Scalable Action Recognition with a Subspace Forest",
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
'''

import proximityforest as pf
from evaluation.action_data_sets.KTH.DataSource import *
from evaluation.action_data_sets.KTH.Protocol import *

import os
import sys
import scipy
#==========================================================================
# START OF SCRIPTS TO AUTOMATE VARIOUS EXPERIMENTS/EVALUATIONS
# The following top-level functions are what you will want to
# run to repeat the reported results.

# The basic pattern of usage is as follows:
# KDat = unPickleData()  #load the pre-computed tracklet data into memory
# res = <top_level_function>(KDat, ...other params as necessary...)
#==========================================================================
def computeAvgConfusion(KDat, treeClass=pf.TensorSubspaceTree, N=9, NumTrials=10, method="knn", K=1, **kwargs):
    '''
    Computes the average confusion matrix for a set of classification trials.
    @param KDat: The tracklets extracted from the KTH source, computed via calling loadKTHData(), or by
    unPickling a previously saved KDat file, via unPickleData().
    @param treeClass: the tensor subspace tree class to use in the forest
    @param N: The number of trees to use in the forest
    @param NumTrials: The number of repetitions over which to average results.    
    @param method: Specify "quick", "popular", or "knn". Quick voting means that we use the weighted class distribution
    over all neighbors in a leaf node. KNN Voting means we sort the leaf neighbors and use the top-K equally weighted
    to vote. Popular is like KNN, but instead of K votes, there are N*K votes (all KNN from all trees vote, as opposed to
    using them to sub-select the top K from the forest to vote).
    @param K: When classifying, use the K-nearest-neighbors from each tree to vote in the forest. Not used if method="quick"
    @param kwargs: Additional keyword parameters required by the treeClass used in the forest, like Ht, Tau, etc.
    '''
    errs = []
    cfmAccum = sp.zeros((6,6))  #accumulation of 6x6 class confusion matrix
    
    (train_dat, test_dat) = partitionKTHProtocol(KDat)
    train_tracklets = train_dat[0]
    train_labels = train_dat[1]
    tracklet_data = pf.build_simple_TD_Structure(train_tracklets, train_labels)
    
    forest = pf.TensorSubspaceForest(N, tracklet_data, treeClass=treeClass, **kwargs)
    
    for trial in range(NumTrials):
        (err, _, cfm) = processOneTrial(forest, trial, train_dat, test_dat, method, K)
        errs.append(err)
        cfmAccum = cfmAccum + cfm
    
    avgCFM = cfmAccum / NumTrials
    pct = (1-sp.mean(errs))*100.0
    print "Average confusion matrix, %d trials"%NumTrials
    print avgCFM
    print "Average accuracy: %f"%pct
    
    return (avgCFM, errs, pct)

def testTauValues(KDat, TauVals = (10,15,20,25), treeClass=pf.TensorSubspaceTree, N=9,
                  NumTrials=10, method="knn", K=1, **kwargs):
    '''
    Tests how the value for Tau, the node splitting threshold, affects performance.
    @param KDat: The tracklets extracted from the KTH source, computed via calling loadKTHData(), or by
    unPickling a previously saved KDat file, via unPickleData().
    @param TauVals: A list of Tau values to try.
    @param treeClass: the tensor subspace tree class to use in the forest
    @param N: The number of trees to use in the forest
    @param NumTrials: The number of repetitions over which to average results.    
    @param method: Specify "quick", "popular", or "knn". Quick voting means that we use the weighted class distribution
    over all neighbors in a leaf node. KNN Voting means we sort the leaf neighbors and use the top-K equally weighted
    to vote. Popular is like KNN, but instead of K votes, there are N*K votes (all KNN from all trees vote, as opposed to
    using them to sub-select the top K from the forest to vote).
    @param K: When classifying, use the K-nearest-neighbors from each tree to vote in the forest. Not used if method="quick"
    @param kwargs: Additional keyword parameters required by the treeClass used in the forest, like Ht, etc.
    @return: An instance of pf.ExperimentalErrorResults object. This can be used in conjunction
    with pf.plotRes to generate a corresponding plot using matplotlib.
    '''
    (train_dat, test_dat) = partitionKTHProtocol(KDat)  #split sets into training and test partitions
    train_tracklets = train_dat[0]
    train_labels = train_dat[1]
    tracklet_data = pf.build_simple_TD_Structure(train_tracklets, train_labels)        
    
    ResMatrix = sp.ones( (len(TauVals),NumTrials))  #rows=parameter setting (N), cols=trial result
    for idx in range( len(TauVals) ):
        Tau = TauVals[idx]
        print "=================================="
        print "Starting %d Trials with Tau=%d"%(NumTrials, Tau)
        print "=================================="
        forest = pf.TensorSubspaceForest(N, tracklet_data, treeClass=treeClass, Tau=Tau, **kwargs)
        for trial in range(NumTrials):
            (err, _, _) = processOneTrial(forest, trial, train_dat, test_dat, method, K)
            ResMatrix[idx, trial] = err
            
        print "==============================="
        print "With Tau=%d, Avg. Err = %f"%(Tau, sp.mean( ResMatrix[idx,:])) 
        print "==============================="
                
    props = {'N':N, 'Voting':method, 'K':K, 'treeClass':treeClass.__name__}
    props.update(kwargs)
    res = pf.ExperimentalErrorResults(ResMatrix, TauVals, "Tau", desc="KTH Classification varying Tau", props=props)
    return res

def testForestSizes(KDat, treeClass=pf.TensorSubspaceTree, forestSizes=[3, 9, 15, 21, 27, 33], 
                    NumTrials=10, method="knn", K=1, **kwargs):
    '''
    tests how the size of the ssforest affects performance.
    @param KDat: The tracklets extracted from the KTH source, computed via calling loadKTHData(), or by
    unPickling a previously saved KDat file, via unPickleData().
    @param treeClass: the tensor subspace tree class to use in the forest
    @param forestSizes: A list of forest sizes (num trees) to evaluate.
    @param NumTrials: The number of repetitions over which to average results.    
    @param method: Specify "quick", "popular", or "knn". Quick voting means that we use the weighted class distribution
    over all neighbors in a leaf node. KNN Voting means we sort the leaf neighbors and use the top-K equally weighted
    to vote. Popular is like KNN, but instead of K votes, there are N*K votes (all KNN from all trees vote, as opposed to
    using them to sub-select the top K from the forest to vote).
    @param K: When classifying, use the K-nearest-neighbors from each tree to vote in the forest. Not used if method="quick"
    @param kwargs: Additional keyword parameters required by the treeClass used in the forest, like Ht, Tau, etc.
    @return: An instance of pf.ExperimentalErrorResults object. This can be used in conjunction
    with pf.plotRes to generate a corresponding plot using matplotlib.
    '''
    (train_dat, test_dat) = partitionKTHProtocol(KDat)  #split sets into training and test partitions
    train_tracklets = train_dat[0]
    train_labels = train_dat[1]
    tracklet_data = pf.build_simple_TD_Structure(train_tracklets, train_labels)
    
    ResMatrix = sp.ones( (len(forestSizes),NumTrials))  #rows=parameter setting (N), cols=trial result
    for idx in range( len(forestSizes) ):
        N = forestSizes[idx]
        print "======================================="
        print "Starting %d Trials with forest size=%d"%(NumTrials, N)
        print "======================================="
        forest = pf.TensorSubspaceForest(N, tracklet_data, treeClass=treeClass, **kwargs)
        #classifier = SSForestClassifier(6, N, treeClass, verbose=False, K=K, **kwargs)
        for trial in range(NumTrials):
            (err, _, _) = processOneTrial(forest, trial, train_dat, test_dat, method, K)
            ResMatrix[idx, trial] = err
            
        print "==============================="
        print "With N=%d, Avg. Err = %f"%(N, sp.mean( ResMatrix[idx,:])) 
        print "==============================="
    
    props = {'Voting':method, 'K':K, 'treeClass':treeClass.__name__}
    props.update(kwargs)
    res = pf.ExperimentalErrorResults(ResMatrix, forestSizes, "Forest Size", desc="KTH Classification varying Forest Size", props=props)
    return res

def testEntropyThreshold(KDat, minHt=2.1, maxHt=2.3, divisions=10, N=27, NumTrials=10, method="knn", K=1,  **kwargs):
    '''
    Tests a range of Ht values for an Entropy Splitting subspace forest.
    Entropy is in nats (base e). With a uniform distribution
    over 10 bins, max Ht possible is log_e(10) = 2.3025. With a distribution
    concentrated into a single bin, H=0, but this is uninformative. The
    more useful lower bound is when something looks like a single gaussian
    peak, which will have Ht around 1.8 nats (more or less, depending on
    how sharp the peak.)
    @return: An instance of pf.ExperimentalErrorResults object. This can be used in conjunction
    with pf.plotRes to generate a corresponding plot using matplotlib.
    '''
    HtList = sp.linspace(minHt, maxHt, divisions)
    
    (train_dat, test_dat) = partitionKTHProtocol(KDat)  #split sets into training and test partitions
    train_tracklets = train_dat[0]
    train_labels = train_dat[1]
    tracklet_data = pf.build_simple_TD_Structure(train_tracklets, train_labels)
    
    ResMatrix = sp.ones( (len(HtList),NumTrials))  #rows=parameter setting (N), cols=trial result
    for idx in range( len(HtList) ):
        Ht = HtList[idx]
        forest = pf.TensorSubspaceForest(N, tracklet_data, treeClass=pf.TensorEntropySplitSubspaceTree, Ht=Ht, **kwargs)
        print "======================================="
        print "Starting %d Trials with Ht=%3.4f"%(NumTrials, Ht)
        print "======================================="
        for trial in range(NumTrials):
            (err, _, _) = processOneTrial(forest, trial, train_dat, test_dat, method, K)
            ResMatrix[idx, trial] = err
            
        print "==============================="
        print "With Ht=%3.4f, Avg. Err = %f"%(Ht, sp.mean( ResMatrix[idx,:])) 
        print "==============================="
                
    props = {'N':N, 'Voting':method, 'K':K, 'treeClass':type(pf.TensorEntropySplitSubspaceTree).__name__}
    props.update(kwargs)
    res = pf.ExperimentalErrorResults(ResMatrix, HtList, "Entropy Splitting Threshold", desc="KTH Classification varying Ht", props=props)
    return res

#==========================================================================
# SUPPORTING CODE: DATA HANDLING
#==========================================================================
def loadTracklets(KDat=None, tfile=os.path.join(KTH_DATA_DIR,"KDat_32x20x20.p")):
    '''
    For convenience, tracklets have already been extracted from the source KTH videos
    and saved to a python pickle file. This function loads these pre-extracted tracklets
    and returns data objects we find convenient to use in the experiments.
    @param KDat: If you already have the KDat object in memory, you can provide it
    to this function, which will then create the derivative data structures. Otherwise,
    specify None and KDat will be loaded.
    @param tfile: If KDat is None, this is the on-disk version to be loaded.
    @return: A tuple (tracklet_data, (train_dat,test_dat), KDat), where tracklet_data
    is of type TrackletData, and is the structure required to build a tensor subspace forest.
    (train_dat, test_dat) is a partitioning of KDat based on Schuldt's partitioning.
    train_dat and test_dat are tuples of the form (tracklets,labels).
    '''
    if KDat is None: KDat = unPickleData(tfile)
    (train_dat, test_dat) = partitionKTHProtocol(KDat)  #split sets into training and test partitions
    tracklet_data = pf.build_simple_TD_Structure(train_dat[0], train_dat[1])
    return(tracklet_data, (train_dat,test_dat), KDat)

def loadTrackletData(td_filename=os.path.join(KTH_DATA_DIR,"kth_tracklet_data.p")):
    '''
    Wrapper function for loading a saved tracklet_data object from a file. Note: if you want
    to use the parallel python implementation, then filenames should be absolute paths to
    a disc location that all computing nodes in the cluster can access.
    '''
    td = pf.TrackletData()
    td.load(td_filename)
    return td
    
#==========================================================================
# SUPPORTING CODE: SINGLE-TRIAL CLASSIFICATION ENGINE
#==========================================================================
def processOneTrial(forest, trial, train_dat, test_dat, method, K):
    '''
    The common 'engine' used to classify the KTH videos using the
    test tracklets and a particular forest structure.
    '''
    train_tracklets = train_dat[0]
    train_labels = train_dat[1]
    
    forest.clear()

    print "------------Starting Trial %s------------"%str(trial+1).zfill(2)
    print "Building Forest from %d trees of type: %s"%(len(forest),str(type(forest[0]).__name__))
    print "From %d training video clips"%len(train_tracklets)

    forest.addList( range(len(train_tracklets)), train_labels)
    print "Classifying Test Samples"
    pfa = pf.ProximityForestAnalysis(forest)
    (err, cfList) = pfa.classifyAll(test_dat[0], test_dat[1], 6, method, K, verbose=False)
    cfm = pf.confusionMatrix(cfList, [36]*6)
    print "-----------------------------------------"
    print "Trial accuracy: %3.3f"%((1.0-err)*100)
    print cfm
    print "-----------------------------------------"
    return (err, cfList, cfm)
    
