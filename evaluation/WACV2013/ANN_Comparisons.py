'''
Created on Jul 13, 2012
@author: Stephen O'Hara
Colorado State University

Code to support the comparative evaluation of the Proximity Forest to
other ANN methods available in Muja and Lowe's FLANN library.

NOTE: The Proximity Forest indexing structure is shown to be
highly accurate (see O'Hara and Draper, Are You Using the Right Nearest Neighbors
Algorithm?, Workshop on Applications of Computer Vision, 2013), but
the simple python implementation used here is nowhere near as fast as
the well-optimized flann library.

Please DO NOT make timing comparisons between the two implementations
 as it is unfair. More fair would be a C++ implementation of Proximity Forest,
 with a threaded implementation to take advantage of the inherent parallelism
 on multi-core machines, and vector/Matrix distance computations (instead of iterating through lists)
 when you have vector space data, and so on. The latest version of FLANN even
 has CUDA-based GPU acceleration.

Again, these are ACCURACY comparisons, not speed. If the accuracy improvement
is verified by independent researchers, then perhaps that would encourage software engineers to make
an optimized implementation.
'''
import proximityforest as pf
from data_preparation import loadMSER_npy, loadSift10K, loadSift10K_asMatrix, loadScissors, partitionDataM

import scipy
import sys
import pyflann as pyf

#DISTANCE FUNCTIONS FOR USE IN EVALUATIONS
DIST_EUCLIDEAN = 1  #euclidean distance
DIST_EUCLSQR = 2    #square of euclidean distance, for efficiency
DIST_CHISQR = 3
DIST_MANHATTAN = 4  #L1 or manhattan distance

def dist_manhattan(pt1, pt2):
    u = scipy.array(pt1)
    v = scipy.array(pt2)
    return sum( scipy.absolute(u-v) )
def dist_euclidean(pt1,pt2):
    u = scipy.array(pt1)
    v = scipy.array(pt2)
    return scipy.sqrt(scipy.sum((u-v)**2))
def dist_euclidean_squared(pt1,pt2):
    u = scipy.array(pt1)
    v = scipy.array(pt2)
    return scipy.sum((u-v)**2)
def dist_chi_squared(pt1,pt2):
    '''
    A symmetric histogram chi-square distance function.
    @param pt1: The measured point, which should be a 1D histogram
    @param pt2: The expected point, which should be a 1D histogram, same size as pt1
    @return: The chi-squared distance as \sum ( (pt2-pt1)^2 / (pt1+pt2) ) for those elements
    where pt1(i) + pt2(i) > 0, else 0 for that component.
    @note: We do not normalize the input histograms, to maintain consistency with FLANN's implementation.
    ''' 
    u = scipy.array(pt1) #/sum(pt1)  #to be compatible with FLANN's implementation, don't normalize histograms
    v = scipy.array(pt2) #/sum(pt2)
    s = u+v
    d = (u-v)**2
    return scipy.sum( d[s>0] / s[s>0] )

    
def calc_num_correct( resX, resT ):
    '''
    @note: resX and resT are numpy arrays, N rows by K columns, and represent the indexes of
    the K nearest neighbors of N samples found by an algorithm.
    @param resX: The KNN from a given ANN algorithm
    @param resT: The ground truth (exact) KNN for each corresponding points
    @return: The total number of correct KNN. This is computed by determining how many of
    the approximate KNN are in the ground truth. We don't want to worry too much about ordering
    because equidistant neighbors could occur and may not be in the same order from different algorithms.
    '''
    
    assert( resX.shape == resT.shape)
    try:
        (N,_) = resX.shape
    except:
        print "Error: input data must be a scipy matrix, with dimensions (N,K)"
        print "You probably passed in data for K=1, and it is a vector...shape is (N,0)"
        print "Try doing resX.reshape(N,1) before passing into this function."
        return -1
    
    #fun with nested list comprehensions!
    hits = [ [ (e in list(resT)[i]) for e in list(resX)[i]] for i in range(N)]
    X = scipy.array(hits)  #will be NxK of True/False
    return sum(sum(X)) #total number of 3NN found over all N points
    

def pf_test(galleryL, probesL, K=3, dist_func=dist_euclidean_squared, numTrees=10, **kwargs):
    '''
    @return: Tuple (result_indexes, distances) as numpy matrices.
    '''
    Np = len(probesL)
    Res = scipy.zeros((Np,K))
    Dists = scipy.zeros((Np,K))
    
    #Native python code, no parallelism implemented, not-optimized, so slower
    # Also, we make no assumptions about the type of objects being sorted, so we
    # can't necessarily use matrix representation for efficiency (use lists instead)
    pForest = pf.ProximityForest(N=numTrees, Tau=20, dist_func=dist_func, **kwargs)
    pForest.addList( galleryL, range(len(galleryL)))
    
    print "Testing %d Probe Points"%Np
    for i,q in enumerate(probesL):
        if i%100 == 0:
            print ".",
            sys.stdout.flush()
            
        res = pForest.getKNearest( q, K=K)
        idxs = [idx for (_,_,idx) in res]
        ds = [d for (d,_,_) in res]
        Res[i,:] = idxs
        Dists[i,:] = ds
        
    print "\nCompleted."
    return (Res, Dists)

def kdtree_test(galleryL, probesL, K=3, numTrees=10):
    '''
    Forest of randomized kdtrees, as implemented by FLANN library.
    '''
    Np = len(probesL)
    gx = scipy.array(galleryL)
    px = scipy.array(probesL)
    
    f = pyf.FLANN()
    params = f.build_index(gx,algorithm='kdtree', trees=numTrees)
    print params
    print "Testing %d Probe Points"%Np
    [res, ds] = f.nn_index(px, K)
    print "Completed"
    return (res, ds)
    
def hkm_test(galleryL, probesL, K=3):
    '''
    Hierarchical KMeans for ANN search, as implemented by FLANN library.
    '''
    Np = len(probesL)
    gx = scipy.array(galleryL)
    px = scipy.array(probesL)
    
    f = pyf.FLANN()
    #params = f.build_index(gx,algorithm='autotuned', target_precision=0.9, build_weight=0)
    params = f.build_index(gx,algorithm='kmeans', iterations=15, branching=128)
    #iterations=15, branching=128 referenced in FLANN paper as being good settings for 100K SIFT data
    # in my tests, it didn't matter much whether these are used, or the default of branching=32, iterations=5
    
    print params
    print "Testing %d Probe Points"%Np
    [res, ds] = f.nn_index(px, K, checks=params["checks"])
    print "Completed"
    return (res, ds)

def exact_knn(galleryL, probesL, K=3):
    '''
    Linear, exact NN search, as implemented by FLANN library.
    Note how fast even this is on the 10K sift features, due to efficient implementation
    of FLANN library.
    '''
    Np = len(probesL)
    gx = scipy.array(galleryL)
    px = scipy.array(probesL)
    
    f = pyf.FLANN()
    params = f.build_index(gx,algorithm='linear')
    print params
    print "Testing %d Probe Points"%Np
    [res, ds] = f.nn_index(px, K)
    print "Completed"
    return (res, ds)



def repeated_trials(N, DataM, Ng, Np, **kwargs):
    '''
    Repeatedly calls experiment_engine() function over
    number N of trials. Each trial creates a different
    partitioning of the data into Probe/Gallery sets.
    Results are aggregated over the trials.
    @param N: Number of trials
    @param DataM: Data matrix, use one of the load_x()
    functions from the data_preparation.py module
    @param Ng: Number of samples from DataM to use as
    the gallery (database of known points)
    @param Np: Number of samples from DataP to use as
    the probes (query points not in the gallery)
    @param kwargs: Keyword arguments to be passed on to
    experiment_engine(), such as K, numTrees, dist.
    @return: Tuple of results list and average results. In each list,
    first entry is PF, then KDT, HKM, and Exact NN (max count). Accuracy
    is ratio of count to max count, i.e. PF/ExactNN.
    '''
    print "Starting %d Trials with parameters as follows:"%N 
    print kwargs
    resList = []
    for i in range(N):
        print "========================"
        print "Beginning Trial %d"%(i+1)
        print "========================"
        (G,P) = partitionDataM(DataM, Ng, Np)  #new random partition for each trial
        r = experiment_engine(Data=(G,P), **kwargs)
        print "Trial %d Results:"%(i+1)
        print r
        resList.append(r)
                
    avgRes = (sum(scipy.array(resList)) *1.0) / N
    print "Average results over %d Trials"%N
    print "Using settings: ", str(kwargs)
    print avgRes
    
    return (resList, avgRes)

def experiment_engine(K=3, numTrees=10, Data=None, dist=DIST_EUCLSQR, skip_hkm=False):
    '''
    Computes, for a given set of parameters, the index accuracy of the ANN algorithms.
    @param K: The number of nearest neighbors to find per query point
    @param numTrees: The number of trees to use in Proximity Forest and KD Trees algorithms
    @param Data: A tuple of (G,P) where G is a list of gallery samples, and P is a list
    of probe samples. See the partitionDataM() function for generating this input from
    the data matrix.
    @param dist: The distance function to use. Choices are DIST_EUCLIDEAN, DIST_EUCLSQR,
    DIST_CHISQR, DIST_MANHATTAN
    @param skip_hkm: If the intent of your test is to compare how forest sizes affect
    performance, then you may wish to skip HKM algorithm, as it is not implemented as
    a forest. Thus this parameter is available.
    @return: The counts of exact nearest neighbors returned by each method. See the function
    calc_num_correct() for how this is computed. Format of returned data is a list
    of [c1,c2,c3,c4] where c1 is proximity forest count, c2 is KDT, c3 is HKM, c4 is 
    exact NN, and thus will always be the max count (K*num_queries). Accuracy of a given
    method can be computed as c1/c4, or the fraction of exact neighbors found over all queries.
    '''
    if Data is None:
        (G,P) = loadSift10K()
    else:
        (G,P) = Data
        
    if dist==DIST_EUCLSQR:
        df = dist_euclidean_squared
        pyf.set_distance_type('euclidean')  #pyflann's euclidean distance is actually the squared dist
    elif dist==DIST_EUCLIDEAN:
        df = dist_euclidean
        pyf.set_distance_type('euclidean')
        print "Warning: FLANN will report the square of the euclidean distance,"
        print "so a conversion may be required before comparing distances to other methods."
    elif dist==DIST_CHISQR:
        df = dist_chi_squared
        pyf.set_distance_type('cs')
    elif dist==DIST_MANHATTAN:
        df = dist_manhattan
        pyf.set_distance_type('manhattan')
    else:
        print "Error: Unknown distance function type specified."
        return
    
    print "==========================="
    print "Exact Nearest Neighbors"
    print "==========================="
    (res, _) = exact_knn(G, P, K)

    print "==========================="
    print "Proximity Forest Nearest Neighbors"
    print "==========================="    
    (res1, _) = pf_test(G, P, K, dist_func=df, numTrees=numTrees)
    
    
    print "==========================="
    print "kd-trees Nearest Neighbors"
    print "==========================="
    (res2, _) = kdtree_test(G, P, K, numTrees)
    
    if not skip_hkm:
        print "==========================="
        print "HKM Nearest Neighbors"
        print "==========================="
        (res3, _) = hkm_test(G, P, K)
    else:
        res3 = None
    
    if K == 1:
        N = len(P)
        res = res.reshape(N,1)
        res1 = res1.reshape(N,1)
        res2 = res2.reshape(N,1)
        if not res3 is None: res3 = res3.reshape(N,1)
        
    c1 = calc_num_correct(res1, res)
    c2 = calc_num_correct(res2, res)
    c3 = calc_num_correct(res3, res) if not res3 is None else 0
    c4 = len(P)*K  #max score: is P samples time K neighbors correct
        
    return [c1,c2,c3,c4]
    
    #compare each ANN result with the exact KNN, showing results for the first, second,...Kth NN
    #res_table =scipy.zeros((K,4))
    #for k in range(K):
    #    res_table[k,0] = sum(res[:,k] == res1[:,k])
    #    res_table[k,1] = sum(res[:,k] == res2[:,k])
    #    res_table[k,2] = sum(res[:,k] == res3[:,k])
    #    res_table[k,3] = len(P)
        
    #return res_table
    
'''----------------------------------------------
 The following are the top-level functions
 used to generate the results presented
 in the WACV paper. 
 
 For results with different distance functions,
 just pass the parameter: dist=DIST_EUCLSQR
 as part of the keyword arguments. The available
 choices are defined as constants at the top
 of this module: DIST_XYZ 
 
 You can either use the top-level functions
 below, or directly call the experiment_engine()
 or repeated_trials() functions, to specify
 exactly what set of configurations you want to
 evaluate.
 ----------------------------------------------'''

def varyingNumTrees(DataM, Ng=9000, Np=1000, tvals=[1,3,5,10,15,20,25], K=3, numTrials=5, **kwargs):
    '''
    Experiment to measure how the size of the forest affects
    performance. This parameter only affects ProximityForest
    and KD-Trees as the FLANN implementation of HKM doesn't
    use multiple trees, but rather varies the branching factor
    @param DataM: The data matrix, rows are samples, columns are dimensions
    @param Ng: The number of gallery samples to randomly select to build the index each trial
    @param Np: The number of probe samples to query the index
    '''
    T = numTrials #number of trials
    all_average_results = []
    trial_details = []
    for numTrees in tvals:
        print "==============================="
        print "Starting comparison when numTrees=%d"%numTrees
        print "==============================="
        (resList,avgRes) = repeated_trials(T, DataM, Ng, Np, K=K, numTrees=numTrees, skip_hkm=True, **kwargs)
        all_average_results.append(avgRes)
        trial_details.append(resList)
        print all_average_results
        
    print "====================="
    print "Final Results"
    print "====================="
    for i,nt in enumerate(tvals): print "numTrees=%d: %s"%(nt, str(all_average_results[i]) )
    return (all_average_results, trial_details)    
    

def varyingK(DataM, Ng=9000, Np=1000, kvals=[1,3,5,10], numTrees=15, numTrials=5, **kwargs):
    '''
    Experiment to determine how the methods compare for various
    k-values, the number of nearest neighbors to return.
    '''
    T = numTrials #number of trials
    all_average_results = []
    trial_details = []
    for K in kvals:
        print "==============================="
        print "Starting comparison when K=%d"%K
        print "==============================="
        (resList,avgRes) = repeated_trials(T, DataM, Ng, Np, K=K, numTrees=numTrees, **kwargs)
        all_average_results.append(avgRes)
        trial_details.append(resList)
        print all_average_results
        
    print "====================="
    print "Final Results"
    print "====================="
    for i,K in enumerate(kvals): print "K=%d: %s"%(K, str(all_average_results[i]) )
    return (all_average_results, trial_details)

def varyingDataDim(numTrees=15, numTrials=5, K=3, **kwargs):
    '''
    Using the same number of gallery (9K) and probes (1K), test performance on data having different
    dimensions. SIFT=128, MSER=12, Scissors=3.
    '''
    
    print "Loading required data files for SIFT, MSER, and Point Cloud data."
    SIFT = loadSift10K_asMatrix()
    data_sift = ("SIFT", SIFT)
    MSER = loadMSER_npy()
    data_mser = ("MSER", MSER)
    SCIS = loadScissors()
    data_3d = ("Scissors", SCIS)
    data_sets = ( data_sift, data_mser, data_3d)
    
    T = numTrials
    all_average_results = []
    trial_details = []
    for D in data_sets:
        (Dstr, M) = D
        print "==============================="
        print "Starting comparison of %s"%Dstr
        print "==============================="
        (resList,avgRes) = repeated_trials(T, M, 9000, 1000, K=K, numTrees=numTrees, **kwargs)
        all_average_results.append(avgRes)
        trial_details.append(resList)
        print all_average_results
        
    print "====================="
    print "Final Results"
    print "====================="
    for i,(Dstr,_,_) in enumerate(data_sets): print "%s: %s"%(Dstr, str(all_average_results[i]) )
    return (all_average_results,trial_details)


if __name__ == '__main__':
    pass