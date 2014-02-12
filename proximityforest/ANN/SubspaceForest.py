'''
Created on Apr 16, 2012
Refactored from code originally written in 2011
@author: Stephen O'Hara

This module implements SubspaceTrees and SubspaceForests, refactored to
extend from ProximityTrees/Forests, and to have a more memory-efficient
implementation and convenient load/save of forests.

Copyright (C) 2012 Stephen O'Hara

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import proximityforest as pf

import scipy
from scipy.stats.distributions import entropy
from scipy.cluster.vq import kmeans

class SubspaceTree(pf.ProximityTree):
    '''
    The base SubspaceTree class is also known as a Median Splitting Subspace Tree (OHara, CVPR 2012).
    Other variants of SubspaceTrees can inherit from this class and override methods as required.
    The main difference from ProximityTree is that the distance function is the chordal distance
    between subspaces.
    
    NOTE: For my application to tracklet data, I do not directly use this class. Instead, look
    at TensorSubspaceTree in the tracklets module.
    '''
    def __init__(self, **kwargs):
        '''
        Constructor is just an alias to the super class, ProximityTree constructor. All arguments
        and options are the same.
        '''
        pf.ProximityTree.__init__(self, **kwargs)
              
    def _dist(self, A, B):
        '''
        Given A and B, full rank matrices of the same dimension.
        Compute Q1 and Q2, the orthogonal bases for A and B respectively.
        This could be from a QR decomposition, for example, but we use thin svd for speed.
        A = Q1*R1, B = Q2*R2
        
        dist(Q1,Q2) is based on the chordal distance of the canonical angles
        between Q1 and Q2.
        '''
        Theta = pf.canonicalAngles(A.T,B.T)
        return pf.chordalDistance(Theta)
    
    def _typeCheck(self, M):
        '''
        Samples added to a SubspaceTree must be ndarrays
        '''
        assert type(M) == scipy.ndarray, "Invalid input type for SubspaceTrees. Input must be numpy/scipy ndarrays."

class EntropySplitSubspaceTree(SubspaceTree):
    '''
    Entropy Splitting variant of a Subspace Tree, as described by (OHara, CVPR 2012)
    '''
    def __init__(self, Ht=2.19, **kwargs):
        '''
        Constructor
        All params same as for parent class except Ht.
        @param Ht: The entropy threshold that determines when to split a node
        '''
        self.Ht = Ht
        kwargs['Ht'] = Ht
        SubspaceTree.__init__(self, **kwargs)
                
    def _splittingCriteria(self, T, L):
        '''
        Split a node when size >= Tau and Entropy is < Ht, indicating some clusters have formed
        in the distribution. (I.e., the distribution of distances to the pivot has become structured enough...
        entropy is maximized under a uniform/random distribution.)
        @param T: The new data element being added to the node, which has cause the evaluation of the split decision
        @param L: The label or id of the sample being added. Not used in this version.
        '''
        doSplit = False
        if (len(self.items) >= self.Tau):  #condition 1, we have at least Tau items in node
            if self.Pivot is None:
                self._genPivot()
                #since we just picked a pivot, we need to compute the list of distances from
                # all the current samples in this node to the pivot
                self._computeDs()
            else:
                #we've already selected a pivot, so we need to compute the distance from
                # this sample to the pivot and add to the self.Ds list.
                D = self._dist(self.Pivot, T)
                self.Ds.append(D)
                        
            H = self._computeEntropy()
            if H < self.Ht:                 #condition 2, we have met the Entropy Splitting threshold
                #print "\nSplitting Node: %s, Entropy: %f"%(self.ID, H)
                doSplit=True
        
        return doSplit
    
    def _genSplittingThreshold(self, Ds):
        ''' Compute the entropy over a fixed-bin histogram of the distances
        between the samples and the pivot. Find two cluster centers in the
        distribution and choose a midway point as the splitting threshold.
        '''
        #create a list of distances not including the sample which was selected as the pivot
        #...which will have a distance of zero, within numerical errors.
        Dx = [D for D in Ds if D>0.01]
        Dx = scipy.array(Dx).reshape( (len(Dx),1))  #turn Dx list into column vector
        (centroids, _) = kmeans(Dx, 2)  #find two best centroids of distribution
        self.SplitD = scipy.mean(centroids)
    
    def _computeEntropy(self):
        ''' Compute the entropy of the histogram of distances from the pivot element. Low entropy
        scores means the distribution is concentrated, and thus may be a good candidate for splitting.
        High entropy (at limit, a uniform distribution), may indicate that there is no good separation
        of the elements in this node.
        '''
        assert(self.Pivot != None)
        assert(self.Ds != None)
        assert(len(self.Ds) >= self.Tau)
        assert(len(self.Ds) == len(self.items))
            
        #create a list of distances not including the sample which was selected as the pivot
        #...which will have a distance of zero, within numerical errors.
        Dx = [D for D in self.Ds if D>0.01]
        
        #compute histogram using 10 bins of the Dx list
        HistInfo = scipy.histogram(Dx, bins=10)
        pk = scipy.array( HistInfo[0] )
        epsilon = 0.000001
        H = entropy(pk+epsilon)  #avoids log0 warnings
        #print "Histogram: ", HistInfo[0]
        #print "Entropy: %f"%H
        #print "Range: Min(>0)=%f, Max=%f, Mean=%f, Median=%f"%(min(Dx),max(self.Ds),scipy.mean(self.Ds),scipy.median(self.Ds))
        
        return H
        