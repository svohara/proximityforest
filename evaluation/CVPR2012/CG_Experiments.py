'''
Created on Apr 19, 2012
@author: Stephen O'Hara
A consolidation and refactoring of the code required to produce
the experimental results on Cambridge Gesture Data reported in my CVPR2012 paper.

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
from evaluation.action_data_sets.cambridgeGestures.DataSource import *
from evaluation.action_data_sets.cambridgeGestures.Protocol import *

import os
import sys
import scipy

#==========================================================================
# SUPPORTING CODE: SINGLE-TRIAL CLASSIFICATION ENGINE
#==========================================================================
def processOneTrial(GDat, trial, treeClass=pf.TensorSubspaceTree, N=9, method="knn", K=1, **kwargs):
    '''
    The common 'engine' used to classify the Cambridge Gesture videos
    '''    
    cfmAccum = scipy.zeros((9,9))  #accumulation of 9x9 class confusion matrix
    
    print "------------Starting Trial %s------------"%str(trial+1).zfill(2)
    print "Building Forest from %d trees of type: %s"%(N,treeClass.__name__)
    (avg_err, err_by_set, confusionList)=cambridgeStandardClassification(GDat, treeClass=treeClass, N=N, method=method, K=K, **kwargs)
    
    #Generate a consolidated confusion matrix over all test sets 1-4    
    for cfList in confusionList:
        cfm = pf.confusionMatrix(cfList, [20]*9)
        cfmAccum = cfmAccum + cfm
    cfm = cfmAccum / 4.0

    print "-----------------------------------------"
    print "Trial accuracy: %3.3f"%((1.0-avg_err)*100)
    print cfm
    print "-----------------------------------------"
    return (avg_err, cfm, err_by_set)

#==============================================
# Code to evaluate the algorithms over
# several trials and varying parameters
#==============================================


def computeAvgConfusion(GDat, treeClass=pf.TensorSubspaceTree, N=9, NumTrials=10, method="knn", K=1, **kwargs):
    '''
    Computes the average confusion over a set of trials using the standard partitioning of the data set.
    @param GDat: The tracklets extracted from the cambridge gestures source, computed via calling loadGestureData(), or by
    unPickling a previously saved GDat file, via unPickleData().
    @param treeClass: the tensor subspace tree class to use in the forest
    @param N: The number of trees to use in the forest
    @param NumTrials: The number of repetitions over which to average results.    
    @param method: Specify "quick", "popular", or "knn". Quick voting means that we use the weighted class distribution
    over all neighbors in a leaf node. KNN Voting means we sort the leaf neighbors and use the top-K equally weighted
    to vote. Popular is like KNN, but instead of K votes, there are N*K votes (all KNN from all trees vote, as opposed to
    using them to sub-select the top K from the forest to vote).
    @param K: When classifying, use the K-nearest-neighbors from each tree to vote in the forest. Not used if method="quick"
    @param kwargs: Additional keyword parameters required by the treeClass used in the forest, like Ht, etc.
    '''
    errs = []
    cfmAccum = scipy.zeros((9,9))  #accumulation of 9x9 class confusion matrix
    
    for trial in range(NumTrials):
        print "=================="
        print "Starting Trial%d"%(trial+1)
        print "=================="
        (err, cfm, _) = processOneTrial(GDat, trial, treeClass=treeClass, N=N, method=method, K=K, **kwargs)
        errs.append(err)
        cfmAccum = cfmAccum + cfm
    
    avgCFM = cfmAccum / NumTrials
    pct = (1-scipy.mean(errs))*100.0
    print "Average confusion matrix, %d trials"%NumTrials
    print avgCFM
    print "Average accuracy: %f"%pct
    
    return (avgCFM, errs, pct)

def computeAvgErrBySet(GDat, treeClass=pf.TensorSubspaceTree, N=9, NumTrials=10, method="knn", K=1, **kwargs):
    '''
    Process the standard cambridge classification protocol a number of times and report average results per set.
    '''
    ResMatrix = scipy.ones( (NumTrials,4))  #rows=trials, columns=test set 1-4
    
    for trial in range(NumTrials):
        (_, _, err_by_set) = processOneTrial(GDat, trial, treeClass=treeClass, N=N, method=method, K=K, **kwargs)
        ResMatrix[trial,:] = err_by_set
        print "========================================="
        print "Error by Set for Trial %d: %s."%(trial+1,str(err_by_set))
        print "========================================="
    
    print "==========================================="
    print "All Trials Completed. Mean accuracy by set:"
    print [ float("%3.1f"%(100*(1-v))) for v in scipy.mean(ResMatrix,0)]
    print "==========================================="
    return ResMatrix

def testTauValues(GDat, TauVals = (10,15,20,25), treeClass=pf.TensorSubspaceTree, N=9, NumTrials=10, method="knn", K=1, **kwargs):
    '''
    Tests how the value for Tau, the node splitting threshold, affects performance.
    @param GDat: The tracklets extracted from the cambridge gestures source, computed via calling loadGestureData(), or by
    unPickling a previously saved GDat file, via unPickleData().
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
    @return: Tuple (ErrMatrix, TauVals) where ErrMatrix has |TauVvals| rows, each with the classification results for NumTrials.
    TauVals is echoed on the output as well.
    '''
    print "================================================================="
    print "Beginning Experiment 1: Varying Tau."
    print "================================================================="
        
    ResMatrix = scipy.ones( (len(TauVals),4))  #rows=parameter setting, cols=err vals for test sets1..4, avg'd over numtrials
    for idx in range( len(TauVals) ):
        Tau = TauVals[idx]
        print "=================================="
        print "Starting %d Trials with Tau=%d, K=%d"%(NumTrials, Tau, K)
        print "=================================="
        errs = scipy.ones( (NumTrials,4))        
        for trial in range(NumTrials):
            print "Starting Trial %d"%(trial+1)
            (_, errList, _) = cambridgeStandardClassification(GDat, treeClass=treeClass, N=N, Tau=Tau,
                                                                                method=method, K=K, **kwargs)
            errs[trial,:] = errList
        print "==============================="
        print "With Tau=%d, Avg. Err by Set = %s"%(Tau, str([ int(1000*v)/10.0 for v in scipy.mean(errs,0)]) )
        print "==============================="
        ResMatrix[idx, :] = scipy.mean(errs,0)
        
    return (ResMatrix, TauVals)
    
