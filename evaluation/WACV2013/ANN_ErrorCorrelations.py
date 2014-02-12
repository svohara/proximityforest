'''
Created on Apr 17, 2013
@author: Stephen O'Hara

The scripts in the module are used to measure how
much the trees of a Proximity Forest / KD-Tree Forest
make the same NN indexing errors. One hypothesis for
why PF outperforms KDT with 5 or more trees is that the
trees in a PF make less correlated errors, and thus
benefit more from combining the results.
'''

import proximityforest as pf
from ANN_Comparisons import dist_euclidean_squared,DIST_EUCLSQR, exact_knn
from data_preparation import loadMSER_npy, loadSift10K, loadSift10K_asMatrix, loadScissors, partitionDataM
from scipy.stats.distributions import entropy

import scipy as sp
import sys
import pyflann as pyf

def err_entropy(ErrM):
    X = sp.sum(ErrM, axis=1)
    return entropy(X)

def correlate_errs(X,Y):
    X2 = (X - sp.mean(X))/sp.std(X)
    Y2 = (Y - sp.mean(Y))/sp.std(Y)
    c = sp.correlate(X2, Y2)/len(X)
    return abs( c[0] )

def avg_corr(ErrM):
    (_N,p) = ErrM.shape #N=number of query points, p=#trees
    corr_list = []
    for i in range(p):
        for j in range(i+1,p):
            corr_list.append( correlate_errs(ErrM[:,i],ErrM[:,j]) )
    print "There are %d pairs of correlated observations"%len(corr_list)
    mean = sp.mean(corr_list)
    std = sp.std(corr_list)
    print "Average pair-wise correlation is: %3.3f"%mean
    print " with standard deviation: %3.3f"%std
    
    return corr_list, mean, std

def compute_errors( resX, resT, binary_score=True ):
    '''
    Computes the errors made for each probe point.
    @note: resX and resT are numpy arrays, N rows by K columns, and represent the indexes of
    the K nearest neighbors of N samples found by an algorithm.
    @param resX: The KNN from a given ANN algorithm
    @param resT: The ground truth (exact) KNN for each corresponding points
    @param binary_score: If true, score is 0/1, indicating presence (1) or absence (0)
    of any errors in the KNN results for each probe. Otherwise the score is an integer
    in the range [0,K] indicating the number of incorrect KNN for each point.
    @return: A vector of scores corresponding to each of the N probes.
    '''   
    assert( resX.shape == resT.shape)
    try:
        (N,K) = resX.shape
    except:
        print "Error: input data must be a scipy matrix, with dimensions (N,K)"
        print "You probably passed in data for K=1, and it is a vector...shape is (N,0)"
        print "Try doing resX.reshape(N,1) before passing into this function."
        return -1
    
    #fun with nested list comprehensions!
    hits = [ [ (e in list(resT)[i]) for e in list(resX)[i]] for i in range(N)]
    X = sp.sum( sp.array(hits), axis=1 ) #will be NxK
    if binary_score:
        err_vector = (X != K).astype(int)  #no errors if number of hits equals the number of neighbors returned   
    else:
        err_vector = K - X
        
    return err_vector

def KDTree_ErrorCor(galleryL, probesL, ground_truth, K=3, numTrees=10, binary_score=True):
    '''
    Compute the correlation in errors between different KDTrees indexed
    on the same input data.
    @param galleryL: The list of gallery (training) points to index
    @param probesL: The list of probe (test) points
    @param ground_truth: A sp.array of size (nxk) where n is the number
    of probe points and k is the number of neighbors being retrieved.
    Each entry is the exact kth-nearest-neighbor from the gallery to the probe.
    @param K: Number of neighbors to retrieve per probe.
    @param numTrees: The number of trees over which to gather statistics
    @return: sp.array of size (nxp) where n is the number of query points
    and p is the number of trees being tested. The entries are boolean,
    1 = a knn error occurred for this query point in this tree, 0 = no
    error. Columns are the "error signal" which can be correlated.
    '''
    Np = len(probesL)
    gx = sp.array(galleryL)
    px = sp.array(probesL)
    
    ground_truth = sp.reshape(ground_truth,(Np,K)) if K==1 else ground_truth
    
    trees = []
    print "Building %d separate KDTrees from input data..."%numTrees
    for _idx in range(numTrees):
        f = pyf.FLANN()
        _params = f.build_index(gx,algorithm='kdtree', trees=1)
        trees.append(f)
    
    errs = []
    
    print "Testing %d Probe Points across %d KDTrees"%(Np, numTrees)
    for f in trees:
        print ".",
        sys.stdout.flush()
        [res, _ds] = f.nn_index(px, K)
        if K==1:
            res = sp.reshape(res,(Np,1))
        err_vec = compute_errors(res, ground_truth, binary_score=binary_score)
        errs.append( sp.reshape(err_vec, (Np,1) ) )
    print ""
      
    ErrM = sp.hstack(errs)
    return ErrM

def KDTForest_ErrorCor(galleryL, probesL, ground_truth, K=3, forest_size=6,
                       numForests=5, binary_score=True):
    Np = len(probesL)
    gx = sp.array(galleryL)
    px = sp.array(probesL)
    
    ground_truth = sp.reshape(ground_truth,(Np,K)) if K==1 else ground_truth
    
    forests = []
    print "Building %d separate KDT Forests from input data..."%numForests
    for _idx in range(numForests):
        f = pyf.FLANN()
        _params = f.build_index(gx,algorithm='kdtree', trees=forest_size)
        forests.append(f)
    
    errs = []
    
    print "Testing %d Probe Points across %d KDT Forestsa"%(Np, numForests)
    for f in forests:
        print ".",
        sys.stdout.flush()
        [res, _ds] = f.nn_index(px, K)
        if K==1:
            res = sp.reshape(res,(Np,1))
        err_vec = compute_errors(res, ground_truth, binary_score=binary_score)
        errs.append( sp.reshape(err_vec, (Np,1) ) )
    print ""
      
    ErrM = sp.hstack(errs)
    return ErrM


def ProxTree_ErrorCor(galleryL, probesL, ground_truth, K=3, numTrees=10, binary_score=True):
    '''
    Compute the correlation in errors between different Proximity Trees indexed
    on the same input data.
    @param galleryL: The list of gallery (training) points to index
    @param probesL: The list of probe (test) points
    @param ground_truth: A sp.array of size (nxk) where n is the number
    of probe points and k is the number of neighbors being retrieved.
    Each entry is the exact kth-nearest-neighbor from the gallery to the probe.
    @param K: Number of neighbors to retrieve per probe.
    @param numTrees: The number of trees over which to gather statistics
    @return: sp.array of size (nxp) where n is the number of query points
    and p is the number of trees being tested. The entries are boolean,
    1 = a knn error occurred for this query point in this tree, 0 = no
    error. Columns are the "error signal" which can be correlated.
    '''
    Np = len(probesL)
    #gx = sp.array(galleryL)
    #px = sp.array(probesL)
    ground_truth = sp.reshape(ground_truth,(Np,K)) if K==1 else ground_truth
    
    trees = []
    print "Building %d separate Proximity Trees from input data..."%numTrees
    for _idx in range(numTrees):
        t = pf.ProximityTree(dist_func=dist_euclidean_squared, Tau=15)
        trees.append(t)
        t.addList(galleryL, labels=range(len(galleryL)))
    
    errs = []
    
    print "Testing %d Probe Points across %d Proximity Trees"%(Np, numTrees)  
    Res = sp.zeros((Np,K))  
    for idx,ptree in enumerate(trees):
        print "\nTree %d..."%(idx+1)
        
        for i,q in enumerate(probesL):
            if i%100 == 0:
                print ".",
                sys.stdout.flush()
                
            res = ptree.getKNearest( q, K=K)
            idxs = [idx for (_,_,idx) in res]
            Res[i,:] = idxs
      
        err_vec = compute_errors(Res, ground_truth, binary_score=binary_score)
        errs.append( sp.reshape(err_vec, (Np,1) ) )

    print ""
      
    ErrM = sp.hstack(errs)
    return ErrM

def ProxForest_ErrorCor(galleryL, probesL, ground_truth, K=3, 
                        forest_size=6, numForests=5, binary_score=True):
    Np = len(probesL)
    #gx = sp.array(galleryL)
    #px = sp.array(probesL)
    ground_truth = sp.reshape(ground_truth,(Np,K)) if K==1 else ground_truth
    
    forests = []
    print "Building %d separate Proximity Forests from input data..."%numForests
    for _idx in range(numForests):
        f = pf.ProximityForest(N=forest_size, dist_func=dist_euclidean_squared, Tau=15)
        forests.append(f)
        f.addList(galleryL, labels=range(len(galleryL)))
    
    errs = []
    
    print "Testing %d Probe Points across %d Proximity Forests"%(Np, numForests)  
    Res = sp.zeros((Np,K))  
    for idx,pforest in enumerate(forests):
        print "\nForest %d..."%(idx+1)
        
        for i,q in enumerate(probesL):
            if i%100 == 0:
                print ".",
                sys.stdout.flush()
                
            res = pforest.getKNearest(q, K=K)
            idxs = [idx for (_,_,idx) in res]
            Res[i,:] = idxs
      
        err_vec = compute_errors(Res, ground_truth, binary_score=binary_score)
        errs.append( sp.reshape(err_vec, (Np,1) ) )

    print ""
      
    ErrM = sp.hstack(errs)
    return ErrM

def script1(data=None, K=1, numTrees=20):
    '''
    Script to compute how correlated errors are from
    different trees of (KD-Tree/Proximity) forests.
    @param data: A tuple (g,p) where g is a list of gallery
    ponts and p is a list of probe points. If set to None,
    then the Sift10K data will be used.
    @param K: How many neighbors to return for each probe?
    @param numTrees: How many trees to test?
    @return: Tuple (Err_KD, Err_PF) where Err_KD/PF are
    the matrices of errors, nxp, where n is the number of
    query points and p is the number of trees.
    '''
    if data is None:
        (g,p) = loadSift10K()
    else:
        (g,p) = data
        
    resT, _ = exact_knn(g, p, K=K)
    
    print "===================================="
    print "Computing errors using KD-Trees"
    print "===================================="
    errM_KD = KDTree_ErrorCor(g, p, resT, K=K, numTrees=numTrees, binary_score=True)
    print ""
    print "===================================="
    print "Computing errors using Proximity Trees"
    print "===================================="
    errM_PF = ProxTree_ErrorCor(g, p, resT, K=K, numTrees=numTrees, binary_score=True)
    print ""
    print "===================================="
    print "Error correlations: KD-Trees"
    print "===================================="
    rc1 = avg_corr(errM_KD)
    print "===================================="
    print "Error correlations: Proximity Trees"
    print "===================================="
    rc2 = avg_corr(errM_PF)
    print ""
    print "===================================="
    print "Computing error entropy..."
    print "===================================="
    X_KD = sp.sum(errM_KD, axis=1)
    X_PF = sp.sum(errM_PF, axis=1)    
    H_KD = entropy(X_KD[X_KD>0])
    H_PF = entropy(X_PF[X_PF>0])
    print "Entropy of KD_Tree Errors: %3.3f"%H_KD
    print "Entropy of Prox Tree Errors: %3.3f"%H_PF
    
    return (errM_KD, errM_PF)
    
def script2(data=None, K=3, forest_size=6, numForests=5):
    '''
    Script to compute how correlated errors are between
    different forests (KDT,PF). A forest of KD-Trees and a
    Proximity Forest are constructed that have approximately
    equal predictive power (this happens on SIFT data with 3NN
    queries at about 6 trees). Errors are correlated between
    successive forests (of same size), to determine if
    proximity forests have more diversity than KDT Forests.
    @param data: A tuple (g,p) where g is a list of gallery
    points and p is a list of probe points. If set to None,
    then the Sift10K data will be used.
    @param K: How many neighbors to return for each probe?
    @param forest_size: The forest size to use when testing
    error correlation between forests.
    @return: Tuple (Err_KD, Err_PF) where Err_KD/PF are
    the matrices of errors, nxp, where n is the number of
    query points and p is the number of trees.
    '''
    if data is None:
        (g,p) = loadSift10K()
    else:
        (g,p) = data
        
    resT, _ = exact_knn(g, p, K=K)
    
    print "===================================="
    print "Computing errors using KD-Trees"
    print "===================================="
    errM_KD = KDTForest_ErrorCor(g, p, resT, K=K, forest_size=forest_size, numForests=numForests, binary_score=True)
    print ""
    print "===================================="
    print "Computing errors using Proximity Trees"
    print "===================================="
    errM_PF = ProxForest_ErrorCor(g, p, resT, K=K, forest_size=forest_size, numForests=numForests, binary_score=True)
    print ""
    print "===================================="
    print "Error correlations: KD-Trees"
    print "===================================="
    rc1 = avg_corr(errM_KD)
    print "===================================="
    print "Error correlations: Proximity Trees"
    print "===================================="
    rc2 = avg_corr(errM_PF)
    print ""
    print "===================================="
    print "Computing error entropy..."
    print "===================================="
    X_KD = sp.sum(errM_KD, axis=1)
    X_PF = sp.sum(errM_PF, axis=1)    
    H_KD = entropy(X_KD[X_KD>0])
    H_PF = entropy(X_PF[X_PF>0])
    print "Entropy of KD_Tree Errors: %3.3f"%H_KD
    print "Entropy of Prox Tree Errors: %3.3f"%H_PF
    
    return (errM_KD, errM_PF)

if __name__ == '__main__':
    pass