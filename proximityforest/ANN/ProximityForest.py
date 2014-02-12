'''
Created on Mar 1, 2012
@author: Stephen O'Hara
A tree that sorts elements based on the median
distance to a randomly selected pivot element.

"ProximityTree" because we only consider the scalar
proximity between pairs of points, and not any spatial
configuration. Points in a proximity tree can live in
non-euclidean space.

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
import scipy
import sys
import math
import os
import cPickle
import types


    

class ProximityTree(object):
    '''
    Base class for a tree which sorts elements based on a pair-wise distance function.
    Used in Latent Configuration Clustering (O'Hara et al.)
    Subclasses of this base class should override self._dist(A,B) to provide the
    correct distance function for the structure to use in proximity computations.
    '''
    zero_median_warning_given = False
    
    def __init__(self, tree_id="root", dist_func=None, depth=0, parent=None, Tau=15, maxdepth=50, **kwargs):
        '''
        Constructor
        @param tree_id: A string to identify this tree / subtree
        @param dist_func: The distance function to use in proximity computations between samples. If None,
        then the built-in _dist() method of the class is used.
        @param depth: The depth of this tree, the root node should be zero.
        @param parent: If a subtree (non-root node), then the parent should point back to the
        containing node.
        @param Tau: The splitting number. This node will collect samples in an item list until
        this number is reached, at which point it splits into two subtrees and sends half of
        the samples down each path.
        @param maxdepth: The deepest allowable tree.
        @note: Some subclasses of the Proximity Tree, such as an EntropySplitSubspaceTree, require
        additional keyword arguments which must be passed onto the constructor of all subtrees when
        splitting nodes. To allow the subclasses to avoid having to override the _split() method,
        all kwargs will be passed onto all child subtrees.
        '''
        self.ID = tree_id
        self.dist_func = dist_func
        self.depth = depth
        self.Tau = Tau
        self.parent = parent
        self.Pivot = None 
        self.SplitD = None
        self.items = [] #samples we 'collect' until reaching a splitting size
        self.Children = [None, None] #references to children of this node
        self.State = "Collecting"
        self.maxdepth = maxdepth #to prevent pathological imbalances from tripping a recursion exception and crashing
        self.tree_kwargs = kwargs
        self.Ds = None
          
    def __str__(self):
        if self.State == "Collecting":
            s = "%s. State=Collecting. NumItems=%d."%(self.ID, len(self.items))
        else:
            s = "%s. State=Splitting.\n{%s}\n{%s}"%(self.ID, self.Children[0], self.Children[1])            
        return s
    
    def _dist(self, A, B):
        '''
        This simple distance function will work if the minus "-" operator has
        been overloaded correctly for input object A and B.
        @param a: The first element, of any data type that implements the - operator
        as a dissimilarity function.
        @param b: The second element of the same data type as a.
        @return: A scalar number representing the dissimilarity or distance between a and b.
        '''
        if self.dist_func is None:
            return A-B
        else:
            return self.dist_func(A,B)
    
    def clear(self):
        '''clears out all data in tree'''
        self.items = []
        self.State = "Collecting"
        self.Children = [None,None]
        self.SplitD = None
        self.Pivot = None
        self.Ds = None
            
    def add(self, T, label=None):
        '''add sample T to tree
        @param T: A sample representing the data item being added
        @param label: A label (integer) that is used to tag the sample T being added. This can be
        useful, for example, when looking at a node in the tree and wanting to know the majority
        label of the items (if you have a label), or for knowing the index number of each sample
        in the node.
        @note: T must be hashable.
        '''
        assert self._typeCheck(T)
        #Check if samples are hashable by testing the first
        try:
            hash( T )
        except TypeError:
            print "Error: The input data sample is not hashable."
            print "If the samples are arrays or lists, try converting to tuples first."
            return
        
        if self.depth == self.maxdepth:
            raise ValueError("Bad Tree %s, Max Depth (%d) Reached"%(self.ID, self.maxdepth))
        
        if self.State == "Collecting":
            self.items.append( (T,label) )
            if self._splittingCriteria(T,label):
                #print "Splitting Node: %s"%self.ID
                self.State = "Splitting"
                try:
                    self._split()
                except ZeroMedianDistanceError:
                    #this happens when the leaf node has duplicate points and
                    # the median distance is zero. We must not split in this case.
                    self._issueZeroMedianWarning()
                    self.State = "Collecting"
                    self.Pivot = None 
                    self.SplitD = None
        else:            
            self._sendToChild(T, label)
        
    def addList(self, samples, labels, report_frac=0.01):
        '''
    	Adds a list of samples to the proximity tree.
    	@param samples: a list of samples
    	@param labels: a list of integer labels, this could be either a class indicator
    	or the id of the sample, depending on how the resulting tree will be used
    	@param report_frac: How often do you want feedback during the tree construction
    	process, in terms of progress. The default 0.01 means that there will be
    	progress information printed after every 1% of the data has been added.
    	@note: The samples added to the forest should be hashable. This is important when
    	performing KNN queries across the forest, where a set(...) operation is performed
    	to remove duplicate answers. If you can't hash the sample, you can't do this.
    	'''
        
        #Check if samples are hashable by testing the first
        try:
            hash( samples[0] )
        except TypeError:
            print "Error: The input data samples are not hashable."
            print "If the samples are arrays or lists, try converting to tuples first."
            return
        
        print "Randomizing order of input data..."
        idxs = scipy.random.permutation( len(samples) ) #shuffle order that we add samples to tree
        
        rpt_pct = int( len(samples) * report_frac ) #how many samples is the fraction of the data set
        if rpt_pct < 1: rpt_pct = 1
        
        print "Adding input data to tree..."
        counter = 1
        for i in idxs:
            if counter%rpt_pct==0: print counter,
            if counter%(rpt_pct*10)==0: print
            sys.stdout.flush()
            counter += 1    
            self.add(samples[i], labels[i])
        print "\n%d samples added to tree."%len(samples)
    
        
    def _selectBranch(self, T):
        ''' Helper function to determine which of the branches this sample should
        be assigned to, based on median of distance to pivot. '''
        D = self._dist(self.Pivot, T)
        if D <= self.SplitD:
            return 0
        else:
            return 1
    
    def _genSplittingThreshold(self, Ds):
        '''
        For the basic Proximity tree, the splitting threshold will be
        the median distance between the samples in a node and the pivot sample.
        Subclasses may overwrite this method for other strategies.
        '''
        median = scipy.median(Ds)
        if median == 0.0:
            raise ZeroMedianDistanceError("Warning: median distance to pivot node is zero.")
        self.SplitD = median
    
    def _splittingCriteria(self, T, L):
        '''
        The criteria for when a node should be split. In a basic proximity tree,
        this is simply whenever the number of samples in a node meets or exceeds the threshold Tau.
        @param T: The new data element being added to the node, which has cause the evaluation of the split decision
        @param L: The label or id of the sample being added. Not used in this version.
        @Note: For a basic proximity tree, the splitting criteria doesn't care
        about the latest sample, so T and L parameters are ignored. However, some subclasses might base
        a splitting criteria not only on the current state of the node, but may also need information
        from the current sample which is in the process of being added to the node.
        '''
        return ( len(self.items) >= self.Tau )
            
    def _split(self):
        ''' We keep collecting samples (items) until we reach Tau, at which point we
        split this node into a left tree and right tree using a median point in the
        samples, as compared to a distance from a pivot matrix.
        '''
        self._genPivot()        
        if self.Ds is None: self._computeDs()  #some subclasses incrementally update self.Ds for performance reasons...
        self._genSplittingThreshold(self.Ds)
            
        self.Children[0] = self.__class__(tree_id="%s0"%self.ID, dist_func=self.dist_func, depth=self.depth+1,
                                       parent=self, Tau=self.Tau, maxdepth=self.maxdepth, **self.tree_kwargs)
        self.Children[1] = self.__class__(tree_id="%s1"%self.ID, dist_func=self.dist_func, depth=self.depth+1,
                                       parent=self, Tau=self.Tau, maxdepth=self.maxdepth, **self.tree_kwargs)
        for i,D in enumerate(self.Ds):
            if D <= self.SplitD:
                self.Children[0].add(*self.items[i])
            else:
                self.Children[1].add(*self.items[i])
                
        self.items = []  #all items have been sent down to children, so now should be empty
            
    def _sendToChild(self, T, label):
        ''' When this is a splitting node, we need to send
        the sample T down to the left or right child sub-tree'''
        n = self._selectBranch(T)
        self.Children[n].add(T,label)

    def _genPivot(self):
        ''' Internal method used to select the pivot element required
        by the tree for sorting the items in a node.'''
        x = len(self.items)
        idx = scipy.random.permutation(range(x))[0]
        (T,_) = self.items[idx]     
        self.Pivot = T
    
    def _typeCheck(self, X):
        '''
        Checks that sample X is of the correct data type for this tree,
        to be overridden by subclasses that assume a certain kind of data
        for use in the distance function.
        '''
        return True
    
    def _computeDs(self):
        ''' Computes the list of distances to the pivot from all samples in the node.
        Stores the result in self.Ds. Order of this list is same order as self.items,
        so that the pre-computed distance between an item in self.items is given by
        the corresponding D in self.Ds.
        '''
        assert(self.Pivot !=None)
        self.Ds = [ self._dist(self.Pivot,T) for (T,_) in self.items]

    def _set_dist_func(self, dist_func):
        '''
        This will set the distance function of the tree and all children to the
        specified function. This is mainly used to restore a tree when loaded from
        a pickled file.
        '''
        self.dist_func = dist_func
        #and process the children
        if not self.Children[0] is None:
            for childNode in self.Children: childNode._set_dist_func(dist_func)

    def _persistent_id(self, obj):
        '''
        This method is used to hook into a capability of the pickle module to treat some
        objects as 'external', and thus not include them in the pickling process. Instead,
        a 'persistent id' is pickled in place of the externalized object. Specifically,
        we are not allowed to pickle function references, so we need to deal with user-specified
        dist_funct values as externalized references.
        '''
        if isinstance(obj, types.FunctionType):           
            return obj.__name__
        else:
            return None
        
    def _issueZeroMedianWarning(self):
        if not ProximityTree.zero_median_warning_given:                                 
            print "WARNING: Median distance to the pivot in a splitting node is zero."
            print "You have multiple objects added to the forest that have zero distance to each other."
            print "Leaf node splitting will be deferred until a non-zero median value occurs."
            ProximityTree.zero_median_warning_given = True
                
    def save(self, d, fn):
        '''
        save this tree to the file fn in directory d.
        @param d: The directory to write the pickle file
        @param fn: The filename of the pickle file that will store this tree
        '''
        f = os.path.join(d,fn)
        p = cPickle.Pickler(open(f,"wb"), protocol=-1)
        p.persistent_id = self._persistent_id
        
        print "Saving Proximity Tree to file %s"%f
        p.dump(self)
    
    def visit(self, func):
        '''
        Visits every node in the tree and calls func(node), appending
        the result to a list as tuple (node_id, res) where res is
        whatever is returned by func.
        @param func: A function that takes a node as input
        @return: A list of results of applying function to every node
        in the tree
        '''
        reslist = []
        reslist.append( (self.ID, func(self)))
        for childNode in self.Children:
            if childNode is None: continue
            rc = childNode.visit(func)
            reslist += rc
            
        return reslist
        
    def getTreeDepth(self):
        '''
        Recursively compute the depth of the tree
        @return: The depth of the tree. D=0 indicates only the root node
        exists. 
        '''
        if self.Children[0] == None:
            return self.depth
        else:
            Ds = []
            for childNode in self.Children:
                D = childNode.getTreeDepth()
                Ds.append(D)            
            return max(Ds)
                
    def getLeafNodes(self):
        '''
        @return: A list of the leaf-nodes, ordered left-to-right, as generated by
        a depth-first tree traversal.
        '''        
        if self.State == "Collecting":
            #this is a leaf node, so just return self as a 1-element list
            return [self]
        else:
            #we have children, so recursively traverse depth-first L-to-R
            leaves = []
            N = len(self.Children)
            idxs = range(N)
            for i in idxs:
                tmp = self.Children[i].getLeafNodes()
                leaves += tmp                
            return leaves
        
    def getLeafSizes(self):
        '''
        @return: A list of tuples (ID, size) indicating the ID of the leaf node and its
        corresponding size, in terms of the number of items stored in the node.
        '''
        nodes = self.getLeafNodes()
        sizes = []
        for node in nodes:
            sizes.append( (node.ID, len(node.items)))
            
        return sizes
     
    def getNumLeaves(self):
        '''
        @return: The number of leaves in the tree
        '''
        x = self.getLeafNodes()
        return len(x)
     
    def getLeafNode(self, T):
        ''' 
        @param T: The input sample to be sorted by the tree
        @return: The node tree object where T would be sorted. This is equivalent
        to the node where the approximate nearest neighbors of T should be found.
        '''
        ptr = self
        while ptr != None:
            if ptr.State == "Collecting":
                #we reached a leaf node
                break 
            else:
                #travel down either the left or right branch
                n = ptr._selectBranch(T)
                ptr = ptr.Children[n]
                
        #at this point, we are at a leaf node
        return ptr
            
    def getLeafNodeID(self, T):
        ''' given an existing tree and a new sample, T, this method
        returns the leaf node ID where T would sort if it were to be added to
        the tree.
        '''
        node = self.getLeafNode(T)
        return node.ID
        
    def getNeighbors(self, T):
        ''' Use this method when you don't want to add T to the tree, but rather, you want
        to return the items in the leaf node where T would sort...i.e. the neighborhood
        based on the tree structure.
        @return: ( NeighborList, NodeID ) where NeighborList is a list of tuples (X,L) where
        X is the sample, L is the sample label or id. NodeID is the identifier for which leaf node forms
        the neighborhood.
        '''
        node = self.getLeafNode(T)
        return (node.items, node.ID)
    
    def getSortedNeighbors(self, T):
        ''' Return the node neighbors of T, sorted according to increasing distance, as
        a list of tuples (X,L) where X is the sample, L is the label.
        @param T: The input sample
        @return: The neighbors of T, sorted according to distance, as a list
        of tuples (D,X,L), where X is the neighbor sample, L is its label, and D is its distance to T
        '''
        (Neighbors,_) = self.getNeighbors(T)        
        Ds = [ (self._dist(X,T),tiebreaker,X,L) for (tiebreaker,(X,L)) in enumerate(Neighbors)]         
        sorted_neighbors = [ (D,X,L) for (D,_,X,L) in sorted(Ds)]        
        return sorted_neighbors
        
    def getKNearest(self, T, K=3):
        '''
        Get the K nearest neighbors to T from within the leaf node where
        T is sorted. This function, which is O(N) in terms of the size of
        the leaf node (between self.Tau/2 and self.Tau) is intended to be used
        to refine the results of an approximate nearest neighbors search
        using a subspace tree.
        The "neighborhood" of samples that fall in the same tree node
        will typically be a small number, and so we add only a constant time
        additional computation to the retrieval time of the tree structure.
        @param T: The input sample
        @param K: The number of nearest neighbors to return
        @return: A list of tuples (d,X,L) of the nearest neighbors, where d is distance, X is
        the neighbor, and L is the label of the neighbor.
        '''
        sorted_neighbors = self.getSortedNeighbors(T)
        if len(sorted_neighbors) < K:
            #print "getKNearest WARNING: neighborhood size is < K."
            return sorted_neighbors
        else:
            return sorted_neighbors[0:K]

class ProximityTree_Matrix(ProximityTree):
    """
    This is a subclass of ProximityTree that is designed
    to work with Matrix data (2D numpy arrays) as input.
    This is the suggested tree type for use with typical
    feature-vector input.
    
    This tree is compatible with the distance functions defined
    in scipy.spatial.distances.
    """
       
    #TODO: This is not an efficient implementation. Instead, the structure needs to be reworked
    # to allow for faster distance computations between the pivot and a set of other samples...
    def addData(self, D, L=None):
        """
        Adds the samples from a data matrix to the index
        @param D: The data matrix, samples are in rows, features are in columns
        @type D: A 2D numpy ndarray.
        @param L: A label vector, if None, the row index will be used instead.
        @type L: A 1D numpy ndarray (vector), or a list.
        """
        if L is None:
            L = range(D.shape[0])
            
        datalist = [ tuple(v) for v in D ]
        self.addList(datalist, L)

        
class ProximityForest(object):
    '''
    A collection of proximity trees used for ANN queries. A forest can have any
    reasonable number of trees, but all trees must be of the same type, and
    descend from ProximityTree or AbstractSubspaceTree.
    '''
    def __init__(self, N, trees=None, treeClass=ProximityTree, dist_func=None, **kwargs):
        '''
        Constructor
        @param N: The number of trees in the forest.
        @param trees: If not None, then this forest will be constructed around the provided list of
        existing trees. In this case, N will be overwritten to be len(trees), treeClass will be overwritten
        to be the class of the first tree in the list.
        @param treeClass: The class used for the component trees in the forest.
        @param dist_func: Specify the distance function to be used by the consituent proximity trees.
        None means that the default distance function of the treeClass will be used.
        @param kwargs: The keyword arguments required by treeClass to create the component trees.
        '''
        self.tree_kwargs = kwargs
        
        if trees is None:
            self.treeClass = treeClass
            self.N = N
            self.trees = []
            pad = int(round(math.log10(N)))+1
            for i in range(N):
                tmp = self.treeClass(tree_id="t%s.root"%str(i).zfill(pad), dist_func=dist_func, **self.tree_kwargs)
                self.trees.append(tmp)
        else:
            self.N = len(trees)
            self.treeClass = type(trees[0])
            self.trees = trees
        
    def __str__(self):
        return "Proximity Forest of %d Trees"%self.N
    
    def __len__(self):
        '''
        Return the number of trees in the forest.
        '''
        return self.N
    
    def __getitem__(self, i):
        '''
        Support indexing the forest to get the nth tree. forest[n] is nth tree.
        '''
        #assert i in range(self.N)
        return self.trees[i] 
    
    def save(self, base_dir, forest_name, forest_idx=None):
        '''
        Saves the forest as a set of files in the forest_name
        subdirectory of base_dir.
        The following files will be created
        1) tree_<num>.p, one pickle file for each tree in the forest,
        and <num> will be a zero-padded number, like 001, 002, ..., 999.
        2) forest_info.p, a single file with information about the forest     
        @param base_dir: The base directory which holds saved forests
        @param forest_name: The files for the forest will be stored in this
        subdirectory of the base_dir
        @param forest_idx: Specify an offset index to be considered when
        saving the trees in the forest. Instead of tree_000.py, maybe
        you want to start at tree_273.py. This is required to support
        saving a parallel forest. Normally, use None for serial forest
        implementations.   
        '''        
        d = os.path.join(base_dir,forest_name)
        if not os.path.exists(d): os.makedirs(d, 0777)
        
        for i,tree in enumerate(self.trees):
            if not forest_idx is None:
                tidx = i + (forest_idx*len(self))
            else:
                tidx = i
            tn = "tree_%s.p"%str(tidx).zfill(3)
            tree.save(d, tn)
            
        finfo = {}
        finfo['treeClass']=self.treeClass
        finfo['N']=self.N
        finfo['tree_kwargs']=self.tree_kwargs
        
        if not forest_idx is None:
            finfo_filename = "forest_info_%d.p"%forest_idx
        else:
            finfo_filename = "forest_info.p"
            
        cPickle.dump( finfo, open(os.path.join(d,finfo_filename), "wb"), protocol=-1)
        
        if not forest_idx is None:
            print "Forest %d data saved to directory: %s"%(forest_idx,str(d))
        else:
            print "Forest data saved to directory: %s"%str(d)
    
    def clear(self):
        """
        Clears the forest so that it is empty, consisting only of a set of root nodes.
        """
        for tree in self.trees: tree.clear()
        
    def add(self, T, Label):
        """
        Adds sample to the forest.
        @param T: A single sample to add to the forest.
        @type T: Any hashable object appropriate for use with the forests' distance function
        @param Label: An integer label for the sample, this is often either 1) an index that uniquely
        identifies the sample, or 2) a classification label
        @Note: For adding multiple samples to the forest, use the addList() method. Repeatedly adding
        samples that are highly self-similar may yield unbalanced trees. addList() will randomize the
        order that samples are added to the forest, which tends to improve quality.
        """
        #Check if samples are hashable by testing the first
        try:
            hash( T )
        except TypeError:
            print "Error: The input data sample is not hashable."
            print "If the samples are arrays or lists, try converting to tuples first."
            return
        
        for tree in self.trees: tree.add(T, Label)
        
    def addList(self, samples, labels, report_frac = 0.01, build_log = None):
        '''
    	Adds a list of samples to the proximity tree
    	@param samples: a list of samples
    	@param labels: a list of integer labels, this could be either a class indicator
    	or the id of the sample, depending on how the resulting tree will be used
    	@param report_frac: How often do you want feedback during the tree construction
    	process, in terms of progress. The default 0.01 means that there will be
    	progress information printed after every 1% of the data has been added.
    	@param build_log: Specify a full path+file and the build progress will be logged
    	to this file. This is helpful when using the iPython parallel version of
    	a forest, where the progress information from the remote nodes can't be displayed.
    	Specify None (default) to not log the build progress.
    	@note: elements of samples list should be hashable
    	'''
        #Check if samples are hashable by testing the first
        try:
            hash( samples[0] )
        except TypeError:
            print "Error: The input data samples are not hashable."
            print "If the samples are arrays or lists, try converting to tuples first."
            return
        
        print "Randomizing order of input data..."
        idxs = scipy.random.permutation( len(samples) ) #shuffle order that we add samples to tree
        
        rpt_pct = int( len(samples) * report_frac ) #how many samples is the fraction of the data set
        if rpt_pct < 1: rpt_pct = 1
        
        if not build_log is None:
            print "A log file of the build process will be written to: %s."%build_log
            log = open(build_log,"w")
            log.write("============================\n")
            log.write("Proximity Forest Build Log\n")
            log.write("Adding %d samples to forest.\n"%len(samples))
            log.write("============================\n")
        
        print "Adding input data to forest..."
        counter = 1
        for i in idxs:
            if counter%rpt_pct==0:
                print counter,
                sys.stdout.flush()
                if not build_log is None:
                    log.write( str(counter)+"\t")
                    log.flush()
            if counter%(rpt_pct*10)==0:
                print
                if not build_log is None: log.write( "\n" )            
            counter += 1    
            for tree in self.trees: tree.add(samples[i], labels[i])
        print "\n%d samples added to forest."%len(samples)
        if not build_log is None: log.close()
     
    def getKNearestFromEachTree(self, T, K):
        '''
        @return: A list of lists, representing the k-nearest-neighbors found
        in each tree of this forest. The length of the outer list is the size
        of the forest (# trees), and the length of the inner lists is K.
        @note: Note that there is likely to be duplicate entries, as many trees will agree
        that a give sample is a k-neighbor of the probe.
        '''
        KNN_List = [] #there will be a KNN list for each tree in forest...
        for i in range(self.N):
            KNN = self.trees[i].getKNearest(T,K)
            for item in KNN: KNN_List.append(item)
        return KNN_List
    
    def getKNearest(self, T, K):
        '''
        Returns the K-nearest-neighbors in the forest.
        '''
        KNN_List = self.getKNearestFromEachTree(T, K)
        KNNs = list(set(KNN_List))  #remove duplicates b/c many trees will return the same answer as closest, etc.
                
        return sorted(KNNs)[0:K] #like this, if K=3: [ (d1,T1,L1), (d2,T2,L2), (d3,T3,L3)]  

class ProximityForest_Matrix(ProximityForest):
    """
    Subclass of Proximity Forest designed for feature-matrix style input.
    """
    def __init__(self, N, trees=None, dist_func=None, **kwargs):
        """
        Constructor is essentially the same as the parent class (please reference
        for additional documentation), but the treeClass is fixed as
        ProximityTree_Matrix.
        """
        tc = ProximityTree_Matrix
        ProximityForest.__init__(self, N, trees=trees, treeClass=tc, dist_func=dist_func, **kwargs)
        
    def addData(self, D, L=None):
        """
        Adds the samples from a data matrix to the index
        @param D: The data matrix, samples are in rows, features are in columns
        @type D: A 2D numpy ndarray.
        @param L: A label vector, if None, the row index will be used instead.
        @type L: A 1D numpy ndarray (vector), or a list.
        """
        if L is None:
            L = range(D.shape[0])
            
        datalist = [ tuple(v) for v in D ]
        self.addList(datalist, L)

class ZeroMedianDistanceError(ValueError):
    pass
    
def loadProximityTree(filename, dist_func):
    '''
    Loads a pickled tree and connects it to the given distance function, which
    can not be saved to the pickle file.
    @param filename: The pickle file for the tree to load
    @param dist_func: If your tree used a custom distance function, you must
    specify that same function here to properly unpickle the structure. Specify
    None, if you are using the tree class's built-in _dist() function.
    @return: A proximity tree or descendant object.
    '''
    p = cPickle.Unpickler( open(filename,"rb"))
    print "Loading tree from file %s"%filename
    p.persistent_load = []
    tree = p.load()
        
    print "Setting distance function on all subtrees"
    tree._set_dist_func(dist_func)
    return tree 
       
def loadProximityForest(base_dir, forest_name, dist_func=None, tree_idxs=None):
    '''
    Loads an entire forest from the files located in base_dir/forest_name/....
    @param base_dir: The directory containing the forest sub-directory
    @param forest_name: Will be the name of a subdirectory off base_dir where the
    forest files are found
    @param dist_func: The distance function can not be saved to the pickle file
    of the ProximityTree. Thus, if you used a custom dist_func when you originally
    created the tree, you must also specify it here when reloading from file.
    @param tree_idxs: Specify None to load all trees into the forest. Otherwise,
    specify the tree indexes to load. When implementing a parallel version of a forest,
    we may wish to load a subset of the trees in a given saved forest. This allows us
    to have a different distribution of trees <--> computing nodes from one run to
    the next, as resources and circumstances dictate.
    '''
    if dist_func is None:
        print "WARNING: No distance function provided."
        print "If a custom distance function was used to build the forest,"
        print "it is not saved with the forest. You should specify the"
        print "distance function to be re-attached to the forest."
    
    fdir = os.path.join(base_dir, forest_name)
    print "Loading forest information"
    forest_info = cPickle.load(open(os.path.join(fdir,"forest_info.p"),"rb"))
    N = forest_info['N']
    del(forest_info['N'])  #remove this from the keywords because we need to specify the number of trees
                            # directly in the forest constructor, and we might not be using all N
       
    trees = []
    if tree_idxs is None:
        tree_idxs = range(N)
        
    for idx in tree_idxs:
        assert idx <= N-1
        tree_name = "tree_%s.p"%str(idx).zfill(3)
        t=loadProximityTree( os.path.join(fdir,tree_name), dist_func)
        trees.append(t)
        
    forest = ProximityForest(len(tree_idxs), trees=trees, **forest_info)
    return forest

      
