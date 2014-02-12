'''
Created on Jan 29, 2013
@author: Stephen O'Hara

This module contains a set of functions used to import
data for use in comparative evaluation.

For the very large MSER data set, the source data should
be downloaded from the ukbench (Univ. of Kentucky) site,
and then extracted into a convenient python numpy file
using loadMSER() function. With the pre-compiled numpy
file, future loads use loadMSER_npy and function much
faster. NOTE: The precomputed numpy data file should
already be available in the nn_data_sets package directory,
so no need to re-download the ukbench.zip file and
deal with the extraction pain unless you really want to.

The SIFT data was downloaded from Muja and Lowe's FLANN
package in hd5 format. I have converted into a convenient
python pickle data file and made it available in the
nn_data_sets package directory.

The 3D Point cloud data sampled from the scissors model
is small enough to be kept in its native text file format
and loaded as required. The scissors/3D data is also
available in the nn_data_sets package directory.
'''
import cPickle
import scipy
import os
import sys
import glob
import random

import evaluation.nn_data_sets as nn_data_sets
NN_DATA_DIR = nn_data_sets.__path__[0]

def loadSift10K(datadir=NN_DATA_DIR):
    '''
    Loads the 10K SIFT feature data set created by Muja as part of the FLANN package, converted
    from the original hd5f format to a tuple of numpy ndarrays saved in a single pickle file.
    @return: Tuple (gallery_list, probles_list). gallery_list and probes_list represent the data,
    but formated as lists of tuples, gallery_list has 9000 128-element tuples,
    probes_list has 1000 128-element tuples. To convert into matrixes, just call scipy.array(gallery_list),
    fore example.
    '''
    (galleryM, probesM) = cPickle.load( open(os.path.join(datadir,nn_data_sets.NN_DATA_SIFT),"rb"))
    galleryL = [ tuple(g) for g in galleryM]
    probesL = [ tuple(p) for p in probesM]
    return (galleryL, probesL)

def loadSift10K_asMatrix(datadir=NN_DATA_DIR):
    (G,P) = loadSift10K(datadir=datadir)
    X = G + P #combine lists
    M = scipy.array(X)
    return M

def loadScissors(datadir=NN_DATA_DIR, fn=nn_data_sets.NN_DATA_3D):
    '''
    Loads the 250K 3D points from the scissors point cloud data. These are simply 3D coordinates
    representing points scanned from the handle of a pair of scissors.
    '''
    with open( os.path.join(datadir,fn), "r") as f:
        tmp = f.readlines()
        
    #determine where the comments/header info in the file ends
    cur = 0
    while cur < len(tmp):
        linetxt = tmp[cur]
        if not (linetxt[0] == '#'): break
        cur += 1
    
    assert (cur < len(tmp)) #if this fails, then the whole file is header/comments
    
    dat = [ s.split('\n')[0].split() for s in tmp[cur:] ]
    M = scipy.float32(dat)
    return M
    

def loadMSER(datadir=os.path.join(NN_DATA_DIR,"ukbench_extract") ):
    '''
    From the Nister and Stewenius university of kentucky benchmark data set, you can download
    ukbench.zip, which includes 10,200 MSER files, each containing about 1000 MSER interest points
    extracted from images. Total data set size is 7,034,780 MSER points!
    @Note: Once the big matrix has been constructed by loading data from the 1000's of
    files, it's much more efficient for future work to save the big matrix to a single
    numpy file and reload in the future. See loadMSER_npy() function that will load
    the data from a numpy file named "MSER_7M.npy"
    '''
    mfiles = glob.glob( os.path.join( datadir, "mser*"))
    print "There are %d MSER files in the directory: %s."%(len(mfiles), datadir)
    if len(mfiles) < 1: return None
    
    dataMs = []
    for i,mserFile in enumerate(mfiles):
        with open(mserFile,"r") as f:
            lines = f.readlines()
        #first two lines are non-MSER fields
        lines = lines[2:]
        
        if i % 1000 == 0:
            print ""   #new line every 1000 MSER files processed
        if i % 100 == 0:
            print ". ", #show a dot every 100 MSER files processed
            sys.stdout.flush()
        
        #convert into a scipy 2D array N lines (points) by D dimensions
        X = scipy.array( [ scipy.float32(lx.split()) for lx in lines] )
        dataMs.append(X)
    
    M = scipy.vstack( dataMs ) #one giant matrix is the stack of the 10K matrices
    (rs,cs) = M.shape
    
    print "Data loaded. There are %d samples of %d-dimensional MSER points."%(rs, cs)
    return M
    
def loadMSER_npy(fn=nn_data_sets.NN_DATA_MSER,datadir=NN_DATA_DIR):
    '''
    As a shortcut to loading the MSER data set from the 1000's of files found in the
    ukbench_extract folder, one should call loadMSER() once, and save the resulting
    numpy array to a single file. This function assumes you have done so, and will
    load the MSER data from the specified numpy file.
    @Note: This function really doesn't do anything but put comments around the
    use of numpy.load(...). Use numpy.save( filename, M) to create the saved file
    in the first place.
    '''
    return scipy.load( os.path.join(datadir, fn) )

def partitionDataM(X, Ng, Np):
    '''
    Partitions the data matrix X into two LISTS, one is the gallery and one is the probe set.
    @param X: the big matrix of data points, N samples by Q dimensions
    @param Ng: the number of gallery samples, randomly selected from X
    @param Np: the number of probe samples, randomly selected from X
    @return: A tuple (G,P) where G is a list of scipy vectors, one per sample representing
    the gallery. P is the list representing the probe samples.
    '''
    N = X.shape[0]
    D = Ng + Np
    assert D <= N 
    
    Y = random.sample( xrange(N), D)  #xrange is faster range for large numbers
    
    G = [ tuple(X[i,:]) for i in Y[0:Ng]]
    P = [ tuple(X[i,:]) for i in Y[Ng:D]]
    return (G,P)

if __name__ == '__main__':
    pass