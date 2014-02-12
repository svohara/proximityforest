'''
Created on Apr 16, 2012
@author: Stephen O'Hara

To make working with video clips (tracklets) easier, this module
implements variants of the Subspace Trees that work with 3-mode
tensors, unfolding each input sample along a specified axis to
generate the associated matrix.

A forest should contain an equal number of trees that
unfold along each of the three axes.

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
import os
import cPickle
import math
import numpy
import sys
import glob 

class TensorSubspaceTree(pf.SubspaceTree):
    '''
    Convenient version of SubspaceTree that takes a 3-mode tensor
    input, but will unfold the samples along a fixed axis before
    adding to the tree.
    '''
    def __init__(self, tracklet_data=None, axis=1, **kwargs):
        '''
        Constructor
        Parameters are same as for SubspaceTree, but with two exceptions.
        @param tracklet_data: A data structure of type TrackletData
        that has the tracklet tensors and related information.
        @param axis: The unfolding axis 1,2 or 3. (1-based indexing to be compatible with utility function)
        '''
        self.tracklet_data = tracklet_data
        if not tracklet_data is None:
            self.td_filename = tracklet_data.filename
        else:
            self.td_filename = None
        
        self.axis=axis
        kwargs['tracklet_data'] = tracklet_data
        kwargs['axis'] = axis
        pf.SubspaceTree.__init__(self, **kwargs)
        
    def _typeCheck(self, X):
        '''
        X must be a valid index into the tracklet_data structure
        '''
        return self.tracklet_data.tracklet_info.has_key(X)
        
    def _dist(self, Idx1, Idx2):
        '''
        Distance between the axis unfoldings of the tracklets at idx1 and idx2.
        '''
        try:
            A = self.tracklet_data.getUnfolding(Idx1, self.axis)  #note: getUnfolding caches the unfolding for performance
        except:
            print "Unexpected error with tracklet_data.getUnfolding on item index %d, axis %d!"%(Idx1,self.axis)
            sys.exc_info()[0]
            raise
        try:
            B = self.tracklet_data.getUnfolding(Idx2, self.axis)
        except:
            print "Unexpected error with tracklet_data.getUnfolding on item index %d, axis %d!"%(Idx2,self.axis)
            sys.exc_info()[0]
            raise           
                        
        Theta = pf.canonicalAngles(A,B)
        return pf.chordalDistance(Theta)
    
    def _persistent_id(self, obj):
        '''
        This method is used to hook into a capability of the pickle module to treat some
        objects as 'external', and thus not include them in the pickling process. Instead,
        a 'persistent id' is pickled in place of the externalized object. Specifically,
        this code is used in this module to prevent tensor subspace trees from including
        the external tracklet_data in their pickle files.
        '''
        if isinstance(obj, pf.TrackletData):
            #print "DEBUG: Found a tracklet data object."
            #print "DEBUG: filename is %s"%obj.filename            
            return obj.filename
        else:
            return None
        
    def save(self, d, fn):
        '''
        save this tree to the file fn in directory d. We must override the base class's version,
        because we don't want to save the tracklet_data to the pickle file. Instead, the tracklet_data
        is saved/loaded via the appropriate methods of the TrackletData class.
        @param d: The directory to write the pickle file
        @param fn: The filename of the pickle file that will store this tree
        '''
        f = os.path.join(d,fn)
        
        #if tracklet data hasn't been saved yet, do so.
        if not self.tracklet_data is None:
            if self.tracklet_data.filename is None:
                print "Saving tracklet data file as 'tracklet_data.p' in directory %s"%d 
                self.td_filename = os.path.join(d,'tracklet_data.p')
                self.tracklet_data.save( os.path.join(d,'tracklet_data.p'))   
            
        p = cPickle.Pickler(open(f,"wb"), protocol=-1)
        p.persistent_id = self._persistent_id
            
        print "Saving Tensor Subspace Tree to file %s"%f
        p.dump(self)
          
        
    def _connectData(self, tracklet_data):
        '''
        Used when loading a tree from disk. The tracklet data will be stored separately, so
        a process must restore the tree structure, separately load the tracklet data, and then
        call this method to connect the tracklet data object to the tree.
        '''
        if tracklet_data is None:
            return
        
        #connect this sub-tree to the tracklet_data structure
        self.tracklet_data = tracklet_data
        self.tree_kwargs['tracklet_data'] = tracklet_data
        self.td_filename = tracklet_data.filename
            
        #and process the children
        if not self.Children[0] is None:
            for childNode in self.Children: childNode._connectData(tracklet_data)  
        
        return

        
    def getLeafNode(self, T):
        ''' 
        Find the leaf node that represents the neighborhood of a probe sample tracklet.        
        @param T: The input Tracklet to be sorted by the tree, or an index to a tracklet already stored in
        the tracklet data structure.
        @return: The node tree object where T would be sorted. This is equivalent to the node where the approximate
         nearest neighbors of T should be found.
        @note: If T is a Tracklet (not an index), then we assume it is a novel tracklet not already in tracklet data,
        and so this method will temporarily add it, get its new index, call getLeafNode using the index, and after
        getting the results, T will be deleted from tracklet data. Thus repeated calls to this method with a set of
        testing tracklets will not change the size of tracklet data.
        '''
        #is T a tracklet or an index into the tracklet data structure?
        if type(T) == int:
            return pf.ProximityTree.getLeafNode(self,T)
        else:
            #temporarily add probe to tracklet data structure
            idx = self.tracklet_data.add(T, (-1, -1,'TEST DATA'))
            #call superclass now that we've transformed input tracklet to an indexed value
            rc=pf.ProximityTree.getLeafNode(self, idx)        
            #now remove probe from tracklet data once we're done 
            self.tracklet_data.remove(idx)
            #return answer
            return rc
        
    def getSortedNeighbors(self, T):
        ''' Return the node neighbors of T, sorted according to increasing distance, as
        a list of tuples (D,X,L) where X is the sample, L is the label, and D is the distance from X to T.
        @param T: The input sample tracklet (if novel) or integer index of the tracklet in the tracklet data structure.
        @return: The neighbors of T, sorted according to distance, as a list
        of tuples (D,X,L), where X is the neighbor (index), L is its label, D is the dist(X,T)
        '''
        if type(T) == int:
            return pf.ProximityTree.getSortedNeighbors(self, T)
        else:
            #temporarily add probe to tracklet data structure
            idx = self.tracklet_data.add(T, (-1, -1,'TEST DATA'))
            #call superclass method     
            rc = pf.ProximityTree.getSortedNeighbors(self, idx)
            #now remove probe from tracklet data once we're done 
            self.tracklet_data.remove(idx)              
            return rc

class TensorEntropySplitSubspaceTree(TensorSubspaceTree, pf.EntropySplitSubspaceTree):
    '''
    Convenient version of EntropySplitSubspaceTree that takes a 3-mode tensor
    input, but will unfold the samples along a fixed axis before
    adding to the tree.
    '''
    def __init__(self, tracklet_data=None, axis=1, Ht=2.19, **kwargs):
        '''
        Constructor
        Accepts the same basic parameters of a ProximityTree, plus the following additions.
        @param tracklet_data: A data structure of type TrackletData
        that has the tracklet tensors and related information.
        @param axis: The unfolding axis 1,2 or 3. (1-based indexing to be compatible with utility function)
        @param Ht: The entropy splitting threshold. (See EntropySplitSubpaceTree comments)
        '''
        self.Ht = Ht
        kwargs['Ht'] = Ht       
        TensorSubspaceTree.__init__(self, tracklet_data=tracklet_data, axis=axis, **kwargs)
        
    def _splittingCriteria(self, T, L):
        return pf.EntropySplitSubspaceTree._splittingCriteria(self, T, L)
    
    def _genSplittingThreshold(self, Ds):
        return pf.EntropySplitSubspaceTree._genSplittingThreshold(self, Ds)

class TensorSubspaceForest(pf.ProximityForest):
    '''
    Subclass of Proximity forest for use with tracklet/3-mode tensor data,
    a collection of TensorSubspaceTrees (or descendants). The main difference is that because
    we unfold tracklets along 3 different axes, and we construct on subspace tree
    per unfolding, a TensorSubspaceForest must contain subspace trees for each unfolding.
    '''
    def __init__(self, N, tracklet_data=None, trees=None, treeClass=TensorSubspaceTree, **kwargs):
        '''
        Constructor
        @param N: The number of trees in the forest.
        @param trees: If not None, then this forest will be constructed around the provided list of
        existing trees. In this case, N will be overwritten to be len(trees), treeClass will be overwritten
        to be the class of the first tree in the list.
        @param treeClass: The class used for the component trees in the forest.
        @param kwargs: The keyword arguments required by treeClass to create the component trees.
        '''
        assert N%3==0, "ERROR: The size of a Tensor Subspace Forest must be a multiple of 3."
        
        self.tree_kwargs = kwargs
        #self.tracklet_data = tracklet_data
        
        if trees is None:
            self.treeClass = treeClass
            self.N = N
            self.trees = []
            pad = int(round(math.log10(N)))+1
            for i in range(N):
                axis = (i % 3) + 1
                tmp = self.treeClass(tree_id="t%s.root"%str(i).zfill(pad), tracklet_data=tracklet_data, axis=axis, **self.tree_kwargs)
                self.trees.append(tmp)
        else:
            self.N = len(trees)
            self.treeClass = type(trees[0])
            self.trees = trees
        
    def save(self, base_dir, forest_name, skip_tracklet_data=False, forest_idx=None):
        '''
        Saves the forest as a set of files in the forest_name
        subdirectory of base_dir.
        The following files will be created
        1) tree_<num>.p, one pickle file for each tree in the forest,
        and <num> will be a zero-padded number, like 001, 002, ..., 999.
        2) forest_info.p, a single file with information about the forest     
        3) tracklet_data.p, a single file containing the video tracklet data and aux info, unless
        skip_tracklet_data is True, in which case we assume the tracklet_data has been stored
        elsewhere, and it is the responsibility of the user on loading the saved forest to
        reconnect it to the appropriate tracklet data. 
        @note: skip_tracklet_data will only work if the tracklet_data object associated
        with the forest/trees has a non-None filename parameter. Otherwise, the first tree
        that is saved in the forest will save a copy of the tracklet data file.
        @param forest_idx: Specify an offset index to be considered when
        saving the trees in the forest. Instead of tree_000.py, maybe
        you want to start at tree_273.py. This is required to support
        saving a parallel forest. Normally, use None for serial forest
        implementations.  
        '''
        d = os.path.join(base_dir,forest_name)
        if not os.path.exists(d): os.makedirs(d, 0777)
        
        #save tracklet data
        if not skip_tracklet_data:
            print "Saving tracklet data"
            td = self[0].tracklet_data  #we don't store tracklet data at forest, so get it from first tree
            td.save(os.path.join(d,"tracklet_data.p"))
        
        print "Saving constituent trees"
        pf.ProximityForest.save(self, base_dir, forest_name, forest_idx=forest_idx)

        
    def _getTrackletData(self):
        '''
        Method to return the tracklet_data structure used in the forest. Note that
        only the trees have a handle to tracklet_data, but since they are all the
        same, we can just use the tracklet_data from the first tree.
        '''
        return self.trees[0].tracklet_data
    
    def _setTrackletData(self, tracklet_data):
        '''
        Set the tracklet_data object to be used by all trees in the forest.
        '''
        for tree in self.trees:
            tree._connectData(tracklet_data)
    
    def _pmd(self, ID1, ID2):
        '''
        Computes the product manifold distance between two tracklet samples.
        The product manifold distance (Lui et al, CVPR 2010) concatenates the
        canonical angles from all three unfoldings and then uses the chordal
        distance on the combined angles.
        '''
        theta = self._canonicalAngles_Tensor(ID1, ID2)
        theta = numpy.hstack(theta)  #single vector
        return pf.chordalDistance(theta)
        
    def _canonicalAngles_Tensor(self, ID1, ID2):
        ''' Computes the set of canonical angle vectors between two
        3-mode tensors (data cubes). Each cube is unfolded in 3 ways,
        and the canonical angles from each unfolding are computed.
        @param ID1: The ID of the tracklet in the tracklet_data structure
        @param ID2: The ID of the other tracklet
        @return: ThetaList is a list of the three Theta vectors along the three unfoldings.
        '''
        td = self._getTrackletData()

        ThetaList = []  
        for axis in [1,2,3]:
            #unfold on each axis 
            A = td.getUnfolding(ID1, axis)  #note: getUnfolding caches the unfolding for performance
            B = td.getUnfolding(ID2, axis)
            Theta = pf.canonicalAngles(A,B)
            ThetaList.append(Theta)
        return ThetaList
            
    def _addTestSampleToTD(self, T):
        '''
        Often the tracklet data associated with a tensor subspace forest only
        contains the training data. So when processing a test sample for querying
        nearest-neighbors, it can be convenient to temporarily add the sample tracklet
        to the tracklet_data structure, do some work, and then remove it from the
        structure when done. The advantage is the cacheing of the tensor unfoldings
        that is performed by the tracklet_data structure, and shared by all trees.
        '''
        if type(T) != int:  #we have a tracklet as input, thus it is not already in the tracklet-data structure
            #temporarily add probe to tracklet data structure
            td = self._getTrackletData()
            TI = td.add(T, (-1, -1,'TEST DATA'))
        else:
            TI = T  #if provided an integer index, we do nothing and assume it's okay
            
        return TI  #return the index
            
    def getKNearestFromEachTree(self, T, K):
        '''
        @return: A list of lists, representing the k-nearest-neighbors found
        in each tree of this forest. The length of the outer list is the size
        of the forest (# trees), and the length of the inner lists is K.
        @note: Note that there is likely to be duplicate entries, as many trees will agree
        that a give sample is a k-neighbor of the probe.
        '''
        if type(T) != int:
            TI = self._addTestSampleToTD(T)
        else:
            TI = T 
            
        KNN_List = [] #there will be a KNN list for each tree in forest...
        for i in range(self.N):
            KNN = self.trees[i].getKNearest(TI,K)
            for item in KNN: KNN_List.append(item)
            
        #now remove probe from tracklet data once we're done 
        if type(T) != int: self._getTrackletData().remove(TI)
           
        return KNN_List
            
    def getKNearest(self, T, K):
        '''
        Returns the K-nearest-neighbors in the forest.
        We must override the base class version of this method because we have trees of different types (axis unfoldings),
        and so the distances aren't directly comparable.
        '''
        if type(T) != int:
            TI = self._addTestSampleToTD(T)
        else:
            TI = T
        
        #get KNN from each tree
        KNN_List = self.getKNearestFromEachTree(TI, K)

        #just the sample id's...the distances are meaningless
        KNN_List = [ (idx,L) for (_,idx,L) in KNN_List]
        
        #remove duplicates b/c many trees will return the same answer as closest, etc.
        KNNs = list(set(KNN_List))  
        
        #now compute the product manifold distance, which contains all the axes, for each candidate, and find top K        
        pmdList = sorted( [ (self._pmd(TI,X),X,L) for (X,L) in KNNs] )
           
        #now remove probe from tracklet data once we're done 
        if type(T) != int: self._getTrackletData().remove(TI)   
             
        return pmdList[0:K] #like this, if K=3: [ (d1,T1,L1), (d2,T2,L2), (d3,T3,L3)]  
    

def loadTensorSubspaceTree(filename, tracklet_data=None, skip_tracklets=False):
    '''
    Loads a pickled tree and connects it to the given tracklet data structure.
    @param filename: The pickle file for the tree to load
    @param tracklet_data: The tracklet_data object to be referenced by the indexes
    in the tree. If None, then this method will attempt to load the tracklet data
    by inspecting the persistent id in the pickled tree file.
    @return: A tensor subspace tree.
    '''
    p = cPickle.Unpickler( open(filename,"rb"))
    L = []
    p.persistent_load = L
    tree = p.load()
    
    if not skip_tracklets:
        if tracklet_data is None:
            td_filename = L[0]
            print "Loading tracklet_data from file: %s"%td_filename
            (ts,t_info) = cPickle.load( open(td_filename,"rb"))
            tracklet_data = pf.TrackletData(ts, t_info)
            tracklet_data.filename = td_filename
            
        print "Connecting tracklet data to tree..."
        tree._connectData(tracklet_data)
        return (tree, tracklet_data)
    else:
        return (tree, None)
        
def loadTensorSubspaceForest(base_dir, forest_name, tracklet_data=None, tree_idxs=None):
    '''
    Loads an entire forest from the files located in base_dir/forest_name/....
    @param base_dir: The directory containing the forest sub-directory
    @param forest_name: Will be the name of a subdirectory off base_dir where the
    forest files are found
    @param tracklet_data: Specify an already-loaded tracklet data object that corresponds
    with the indexes used in the forest. If None, then the tracklet data object stored in
    the forest directory, if any, will be loaded.
    @param tree_idxs: Specify None to load all trees into the forest. Otherwise,
    specify the tree indexes to load. When implementing a parallel version of a forest,
    we may wish to load a subset of the trees in a given saved forest. This allows us
    to have a different distribution of trees <--> computing nodes from one run to
    the next, as resources and circumstances dictate.
    '''
    fdir = os.path.join(base_dir, forest_name)
    print "Loading forest information"
    
    #there may be either a single forest_info.p file, if the forest was
    # built using the serial implementation, or there may be multiple
    # forest_info_x.p if parallel implementation.
    fi_list = glob.glob(os.path.join(fdir,"forest_info*.p"))
    if len(fi_list) < 1:
        print "Error: Unable to find forest_info file in forest directory."
        return
    elif len(fi_list) > 1:
        N=0
        #multiple forests were used in parallel
        for fi in fi_list:
            forest_info = cPickle.load(open(fi, "rb"))
            N += forest_info['N']
    else:
        #single forest/serial implementation    
        forest_info = cPickle.load(open(fi_list[0],"rb"))
        N = forest_info['N']
        
    del(forest_info['N'])  #remove this from the keywords because we need to specify the number of trees
                                # directly in the forest constructor, and we might not be using all N
                            
    treeClass = forest_info['treeClass']
    assert treeClass in [TensorSubspaceTree, TensorEntropySplitSubspaceTree]  #ensure that we're loading the right kind of trees
    
    trees = []
    if tree_idxs is None:
        tree_idxs = range(N)
    
    if tracklet_data is None:
        td_file = os.path.join(fdir,"tracklet_data.p")
        print "Loading tracklet data from %s"%td_file
        (ts, t_info) = cPickle.load( open(td_file,"rb"))
        tracklet_data = pf.TrackletData(ts, t_info)
    
    for idx in tree_idxs:
        assert idx <= N-1
        tree_name = "tree_%s.p"%str(idx).zfill(3)
        print "Loading tree: %s..."%tree_name
        (t, tracklet_data) = loadTensorSubspaceTree( os.path.join(fdir,tree_name),tracklet_data)
        trees.append(t)
        
    forest = TensorSubspaceForest(len(tree_idxs), trees=trees, **forest_info)
    return forest    

