"""
Created on Mar 3, 2014
@author: Stephen O'Hara

Proximity Forest Indexing
"""
import scipy
import sys
import os
import cPickle
#import types
import random
import math
from collections import Counter

from multiprocessing import Process, Manager, Queue, Pool
from multiprocessing.managers import BaseManager
from functools import partial

    
class ZeroMedianDistanceError(ValueError):
    pass

class DuplicateKey(ValueError):
    pass

class PF_Item(object):
    """
    This class is used to wrap input items, encapsulating
    an index/id for the object and the distance function.
    Subclass this and override the dist(...) method to
    provide for different distance functions. This base
    version uses the '-' operator between the encapsulated
    objects, so if your value object overrides '-' to
    provide a correct distance computation, that's another
    way to allow for customization.
    @note: The original Proximity Forest implementation simply
    had the user pass in a distance function reference, but
    this has issues with the non-pickle-ability of functions.
    @note: Also note that an item added to the proximity forest
    should be hashable. This wrapper class ensures that by
    using an assignable unique id (could just be an int), which
    should be hashable.
    """
    def __init__(self, unique_id, val, label=None):
        """
        Constructor
        @param id: A hashable id that uniquely identifies this object
        @param val: The object being wrapped.
        @param label: set if an auxiliary label should be stored with the item,
        such as a class membership label, otherwise defaults to None
        """
        self.id = unique_id
        self.val = val
        self.label = label
        
    def dist(self, other):
        """
        You can subclass this object and override this method
        to provide for different distance functions.
        """
        return abs( other.val - self.val )
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        """
        If the unique ids are the same, then this must be the same object,
        else equality is false.
        """
        return self.id == other.id
    
    def __repr__(self):
        if not (self.label is None):
            return "%s (%s): %s"%(str(self.id), str(self.label), repr(self.val))
        else:
            return "%s: %s"%(str(self.id), repr(self.val))
    
    def __cmp__(self, other):
        """
        Comparison via unique id, helps with things like sorting
        """
        return cmp(self.id, other.id)

class ProximityTree(object):
    '''
    Base class for a tree which sorts elements based on a pair-wise distance function.
    '''
    zero_median_warning_given = False
    
    def __init__(self, tree_id="root", item_wrapper=PF_Item, depth=0, parent=None, root=None,
                 items_dict=None, Tau=15, maxdepth=50, **kwargs):
        '''
        Constructor
        @param tree_id: A string to identify this tree / subtree
        @param item_wrapper: The wrapper class, like PF_Item, that will wrap input samples
        and defines the distance function between samples.
        @param depth: The depth of this tree, the root node should be zero.
        @param parent: If a subtree (non-root node), then the parent should point back to the
        containing node.
        @param root: If a subtree (non-root node), then the root argument should be a reference
        back to the root node of the tree.
        @param items_dict: If this tree will index items already collected into an item_dict,
        this can be specified here. If None, the root node of the tree will create the item_dict
        used internally. Generally a ProximityForest will hold the items_dict, and the
        individual trees will all reference the same structure.
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
        
        self.depth = depth
        self.Tau = Tau
        self.parent = parent
        self.root = root
        
        self.item_wrapper = item_wrapper
        if not items_dict is None:
            self.items_dict = items_dict
        else:
            if self.root is None:
                self.items_dict = {}
            else:
                self.items_dict = root.items_dict  #all subtrees share the same items_dict
            
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
    
    def _wrap_item(self, x, key=None, label=None):
        """
        Wraps the value x and adds the resulting item to the items dictionary.
        @param x: The input object being wrapped / added to the proximity tree
        @param key: Optional. If None, then a new key is generated to uniquely
        identify the object and act as the key into the items dictionary. Else,
        the user-supplied key is used, and a check is made to ensure that this
        key has not already been used in the dict.
        @return: The key to the wrapped item, which can be retrieved using
        the self.items_dict[key]. Key is hashable and unique.
        """
        if not key:
            key = len(self.items_dict) #TODO replace with safer mechanism
        
        if key in self.items_dict:
            raise DuplicateKey
        
        item = self.item_wrapper(key,x,label=label)
        self.items_dict[key] = item
            
        return key
    
    def _dist1(self, key1, key2):
        """
        Computes the distance between two wrapped items,
        both of which are referenced by their keys in 
        the item dict
        """
        T1 = self.items_dict[key1]
        T2 = self.items_dict[key2]
        return T1.dist(T2)
        
    def _dist2(self, item, key):
        """
        Computes the distance between an input wrapped
        item, not in the item_dict, and another which
        is in the dict and referenced by key.
        """
        T = self.items_dict[key]
        return item.dist(T)
        
    def clear(self):
        """
        clears out all data in tree
        """
        #TODO: do we want to clear the items_dict too? at the root node?
        self.items = []
        self.State = "Collecting"
        self.Children = [None,None]
        self.SplitD = None
        self.Pivot = None
        self.Ds = None
            
    def _add(self, item_key):
        """
        Internal method for adding the sample to the tree, knowing that
        the sample has already been "item_wrapped" and exists in the
        item_dict with the item_key.
        """
        if self.depth == self.maxdepth:
            raise ValueError("Bad Tree %s, Max Depth (%d) Reached"%(self.ID, self.maxdepth))
        
        if self.State == "Collecting":
            self.items.append( item_key )
            if self._splittingCriteria(item_key):
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
            self._sendToChild(item_key)
        
    def add(self, T, label=None):
        '''add sample T to tree
        @param T: A sample representing the data item being added
        @param label: A label (integer) that is used to tag the sample T being added. This can be
        useful, for example, when looking at a node in the tree and wanting to know the majority
        label of the items (if you have a label), or for knowing the index number of each sample
        in the node.
        '''
        item_key = self._wrap_item(T, label=label)
        self._add(item_key)
        
    def _addList(self, item_keys, report_frac=0.01):
        """
        Internal method for adding a batch of samples to the
        tree, where the samples are already in the items_dict,
        and the input order has already been shuffled.
        """
        rpt_pct = int( len(item_keys) * report_frac ) #how many samples is the fraction of the data set
        if rpt_pct < 1: rpt_pct = 1
        
        #print "Adding input data to tree: %s"%self.ID
        #print "DEBUG: Len(item_keys) = %d"%len(item_keys)
        #print "DEBUG: Len(self.items_dict) = %d"%len(self.items_dict)
        #counter = 1
        for key in item_keys:
            #if counter%rpt_pct==0: print counter,
            #if counter%(rpt_pct*10)==0: print
            #sys.stdout.flush()
            #counter += 1    
            self._add(key)
        print "\n%d samples added to tree %s."%(len(item_keys), self.ID)
        
    def addList(self, samples, labels, report_frac=0.01):
        '''
        Adds a list of samples to the proximity tree.
        @param samples: a list of samples
        @param labels: a list of integer labels, this could be either a class indicator
        or the id of the sample, depending on how the resulting tree will be used
        @param report_frac: How often do you want feedback during the tree construction
        process, in terms of progress. The default 0.01 means that there will be
        progress information printed after every 1% of the data has been added.
        @note: The samples added to the forest will be wrapped into "items" using
        the item_wrapper class identified in the constructor. The item_wrapper class
        defines the distance function used between samples as well as provides a
        unique key used internally in the forest.
        '''
        
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
    
        
    def _selectBranch(self, item):
        """
        Helper function to determine which of the branches this sample should
        be assigned to, based on median of distance to pivot.
        """
        if isinstance(item, self.item_wrapper):
            #this is an item not in the items_dict, which is used when
            # we are sorting a sample that is not part of the index to
            # find its nearest neighbors
            D = self._dist2(item, self.Pivot)
        else:
            #item is a key into the items_dict, so we can use the convenience function
            D = self._dist1(item, self.Pivot)
        
        if D <= self.SplitD:
            return 0
        else:
            return 1
    
    def _genSplittingThreshold(self, Ds):
        """
        For the basic Proximity tree, the splitting threshold will be
        the median distance between the samples in a node and the pivot sample.
        Subclasses may overwrite this method for other strategies.
        """
        median = scipy.median(Ds)
        if median == 0.0:
            raise ZeroMedianDistanceError("Warning: median distance to pivot node is zero.")
        self.SplitD = median
    
    def _splittingCriteria(self, item_key):
        """
        The criteria for when a node should be split. In a basic proximity tree,
        this is simply whenever the number of samples in a node meets or exceeds the threshold Tau.
        @param item_key: The key for the new object being added to the tree,
         which has caused the evaluation of the split decision.
        @Note: For a basic proximity tree, the splitting criteria doesn't care
        about the latest sample, so item_key is ignored. However, some subclasses might base
        a splitting criteria not only on the current state of the node, but may also need information
        from the current sample which is in the process of being added to the node.
        """
        return ( len(self.items) >= self.Tau )
            
    def _getRoot(self):
        if (self.root is None):
            return self
        else:
            return self.root
        
    def _split(self):
        """
        We keep collecting samples (items) until we reach Tau, at which point we
        split this node into a left tree and right tree using a median point in the
        samples, as compared to a distance from a pivot matrix.
        """
        self._genPivot()        
        if self.Ds is None: self._computeDs()  #some subclasses incrementally update self.Ds for performance reasons...
        self._genSplittingThreshold(self.Ds)
        
        r = self._getRoot()
        self.Children[0] = self.__class__(tree_id="%s0"%self.ID, item_wrapper=self.item_wrapper, depth=self.depth+1,
                                       parent=self, root=r, Tau=self.Tau, maxdepth=self.maxdepth, **self.tree_kwargs)
        self.Children[1] = self.__class__(tree_id="%s1"%self.ID, item_wrapper=self.item_wrapper, depth=self.depth+1,
                                       parent=self, root=r, Tau=self.Tau, maxdepth=self.maxdepth, **self.tree_kwargs)
        for i,D in enumerate(self.Ds):
            if D <= self.SplitD:
                self.Children[0]._add(self.items[i])
            else:
                self.Children[1]._add(self.items[i])
                
        self.items = []  #all items have been sent down to children, so now should be empty
            
    def _sendToChild(self, item_key):
        """
        When this is a splitting node, we need to send
        the sample T down to the left or right child sub-tree
        """
        n = self._selectBranch(item_key)
        self.Children[n]._add(item_key)

    def _genPivot(self):
        """
        Internal method used to select the pivot element required
        by the tree for sorting the items in a node. In this case,
        we just randomly pick one of the items.
        """
        x = len(self.items)
        idx = random.choice(range(x))
        item_key = self.items[idx]     
        self.Pivot = item_key
    
    def _computeDs(self):
        """
        Computes the list of distances to the pivot from all samples in the node.
        Stores the result in self.Ds. Order of this list is same order as self.items,
        so that the pre-computed distance between an item in self.items is given by
        the corresponding D in self.Ds.
        """
        assert(self.Pivot !=None)
        self.Ds = [ self._dist1(item_key, self.Pivot) for item_key in self.items]
        
    def _issueZeroMedianWarning(self):
        if not ProximityTree.zero_median_warning_given:                                 
            print "WARNING: Median distance to the pivot in a splitting node is zero."
            print "You have multiple objects added to the forest that have zero distance to each other."
            print "Leaf node splitting will be deferred until a non-zero median value occurs."
            ProximityTree.zero_median_warning_given = True
                
    def save(self, d, fn):
        """
        save this tree to the file fn in directory d.
        @param d: The directory to write the pickle file
        @param fn: The filename of the pickle file that will store this tree
        """
        f = os.path.join(d,fn)
        print "Saving Proximity Tree to file %s"%f
        cPickle.dump(self, open(f,"wb"), protocol=-1)

    def visit(self, func):
        """
        Visits every node in the tree and calls func(node), appending
        the result to a list as tuple (node_id, res) where res is
        whatever is returned by func.
        @param func: A function that takes a node as input
        @return: A list of results of applying function to every node
        in the tree
        """
        reslist = []
        reslist.append( (self.ID, func(self)))
        for childNode in self.Children:
            if childNode is None: continue
            rc = childNode.visit(func)
            reslist += rc
        return reslist
        
    def getTreeDepth(self):
        """
        Recursively compute the depth of the tree
        @return: The depth of the tree. D=0 indicates only the root node
        exists. 
        """
        if self.Children[0] == None:
            return self.depth
        else:
            Ds = []
            for childNode in self.Children:
                D = childNode.getTreeDepth()
                Ds.append(D)            
            return max(Ds)
                
    def getLeafNodes(self):
        """
        @return: A list of the leaf-nodes, ordered left-to-right, as generated by
        a depth-first tree traversal.
        """       
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
        """
        @return: A list of tuples (ID, size) indicating the ID of the leaf node and its
        corresponding size, in terms of the number of items stored in the node.
        """
        nodes = self.getLeafNodes()
        sizes = []
        for node in nodes:
            sizes.append( (node.ID, len(node.items)))
        return sizes
     
    def getNumLeaves(self):
        """
        @return: The number of leaves in the tree
        """
        x = self.getLeafNodes()
        return len(x)
     
    def getNeighborhood(self, T):
        """
        @param T: The input sample to be sorted by the tree, but NOT added to it.
        @return: The node tree object where T would be sorted. This is equivalent
        to the node where the approximate nearest neighbors of T should be found.
        """
        #this item isn't being added to tree, so by assigning None as the key,
        # we assure it won't ever have the same id as samples in the items_dict
        input_item = self.item_wrapper(None,T)  

        ptr = self
        while ptr != None:
            if ptr.State == "Collecting":
                #we reached a leaf node
                break 
            else:
                #travel down either the left or right branch
                n = ptr._selectBranch(input_item)
                ptr = ptr.Children[n]
                
        #at this point, we are at a leaf node
        return ptr
            
    def getNeighborhoodID(self, T):
        """
        given an existing tree and a new sample, T, this method
        returns the leaf node ID where T would sort if it were to be added to
        the tree.
        """
        node = self.getNeighborhood(T)
        return node.ID
        
    def getNeighbors(self, T):
        """
        Use this method when you don't want to add T to the tree, but rather, you want
        to return the items in the leaf node where T would sort...i.e. the neighborhood
        based on the tree structure.
        @return: ( NeighborList, NodeID ) where NeighborList is a list of tuples (key,L) where
        key is the key of the sample in the items_dict, L is the sample label.
        NodeID is the identifier for which leaf node forms the neighborhood.
        """
        node = self.getNeighborhood(T)
        return (node.items, node.ID)
    
    def getSortedNeighbors(self, T):
        """ 
        Return the node neighbors of T, sorted according to increasing distance, as
        a list of tuples (key,L) where key is the key of the sample in items_dict, L is the label.
        @param T: The input sample
        @return: The neighbors of T, sorted according to distance, as a list
        of tuples (D,key,L), where key is the sample key in items_dict, L is its label, and D is its distance to T
        """
        #self._dist2(item, key)
        
        item = self.item_wrapper(None,T)
        (Neighbors,_) = self.getNeighbors(T)        
        Ds = [ (self._dist2(item,k),tiebreaker,k) for (tiebreaker,k) in enumerate(Neighbors)]         
        sorted_neighbors = [ (D,k) for (D,_,k) in sorted(Ds)]        
        return sorted_neighbors
        
    def getKNearest(self, T, K=3):
        """
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
        @return: A list of tuples (d,k,L) of the nearest neighbors, where d is distance, k is
        the neighbor key in items_dict, and L is the label of the neighbor.
        """
        sorted_neighbors = self.getSortedNeighbors(T)
        if len(sorted_neighbors) < K:
            #print "getKNearest WARNING: neighborhood size is < K."
            return sorted_neighbors
        else:
            return sorted_neighbors[0:K]


class PT_Manager(BaseManager):
    pass
PT_Manager.register('ProxTree', ProximityTree, exposed=['_add','_addList','getLeafSizes','getKNearest','save','visit'])
    
class ProximityForest(object):
    '''
    A collection of proximity trees used for ANN queries. A forest can have any
    reasonable number of trees, but all trees must be of the same type, and
    descend from ProximityTree.
    '''
    def __init__(self, N, trees=None, item_wrapper=PF_Item, **kwargs):
        '''
        Constructor
        @param N: The number of trees in the forest.
        @param trees: If not None, then this forest will be constructed around the provided list of
        existing trees. In this case, N will be overwritten to be len(trees), treeClass will be overwritten
        to be the class of the first tree in the list.
        @param item_wrapper: The item wrapper class, like PF_Item, that implements a hashable
        unique key and the distance function used to compare samples
        @param kwargs: The keyword arguments required by treeClass to create the component trees.
        '''
        self.tree_kwargs = kwargs
        self.item_wrapper = item_wrapper
        
        self.item_manager = Manager() #shared state manager for the items dict
        self.items_dict = self.item_manager.dict()
        
        self.tree_manager = PT_Manager()  #shared state manager for the individual trees
        self.tree_manager.start()
        
        if trees is None:
            self.N = N
            self.trees = []
            pad = int(round(math.log10(N)))+1
            for i in range(N):
                tmp = self.tree_manager.ProxTree(tree_id="t%s.root"%str(i).zfill(pad), item_wrapper=item_wrapper,
                                     items_dict=self.items_dict, **self.tree_kwargs)
                #tmp = ProximityTree(tree_id="t%s.root"%str(i).zfill(pad), item_wrapper=item_wrapper,
                #                     items_dict=self.items_dict, **self.tree_kwargs)
                self.trees.append(tmp)
        else:
            #construct forest from a set of existing proximity trees
            self.N = len(trees)
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
        #finfo['treeClass']=self.treeClass
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
        This will also clear out the items_dict.
        """
        self.items_dict.clear()
        for tree in self.trees: tree.clear()
        
    def add(self, T, label, key=None):
        """
        Adds sample to the forest.
        @param T: A single sample to add to the forest.
        @param label: An integer label for the sample, like a class label or other auxiliary information.
        @param key: A unique id that will be used to identify the sample in the local items dictionary.
        None means that an internal id will be used.
        @Note: For adding multiple samples to the forest, use the addList() method. Repeatedly adding
        samples that are highly self-similar may yield unbalanced trees. addList() will randomize the
        order that samples are added to the forest, which tends to improve quality.
        """        
        item_key = len(self.items_dict) if key is None else key
        self.items_dict[item_key] = self.item_wrapper(item_key, T, label=label)
        
        procList = [ Process(target=t._add, args=(item_key,)) for t in self.trees]
        for proc in procList: proc.start()
        for proc in procList: proc.join()
        
        
    def addList(self, samples, labels, keys=None, report_frac = 0.01): #, build_log = None):
        '''
        Adds a list of samples to the proximity tree
        @param samples: a list of samples
        @param labels: a list of integer labels, this could be either a class indicator
        or the id of the sample, depending on how the resulting tree will be used
        @param keys: a list of unique keys, of same length and order as the samples list,
        to be used to key the item dictionary in the forest. None means that internal
        keys will be used.
        @param report_frac: How often do you want feedback during the tree construction
        process, in terms of progress. The default 0.01 means that there will be
        progress information printed after every 1% of the data has been added.
        @param build_log: Specify a full path+file and the build progress will be logged
        to this file. Specify None (default) to not log the build progress.
        '''
        
        print "Adding samples to the forest's items dictionary..."
        item_keys = keys if keys else range(len(samples))
        for (k,s,lb) in zip(item_keys, samples, labels):
            self.items_dict[k] = self.item_wrapper(k,s,label=lb)
        
        print "Randomizing order of input data..."
        key_list = scipy.random.permutation( self.items_dict.keys() ) #shuffle order of keys before adding
        
        #if not build_log is None:
        #    print "A log file of the build process will be written to: %s."%build_log
        #    log = open(build_log,"w")
        #    log.write("============================\n")
        #    log.write("Proximity Forest Build Log\n")
        #    log.write("Adding %d samples to forest.\n"%len(samples))
        #    log.write("============================\n")
        
        print "Adding input data to forest..."
        procList = [ Process(target=t._addList, args=(key_list,report_frac)) for t in self.trees]
        for proc in procList: proc.start()  #start them all
        for proc in procList: proc.join()   #wait for them all
        
        print "\n%d samples added to forest."%len(samples)
        #if not build_log is None: log.close()
     
    def _getKNN(self, tree, T, K, Q):
        """
        Internal method to allow for multiprocessing of queries to the forest
        @param tree: The tree to evaluate
        @param T: The input sample
        @param K: The number of neighbors
        """
        res = tree.getKNearest(T,K)
        Q.put(res)
        return
        
    def getKNearestFromEachTree_serial(self, T, K):
        '''
        @return: A list of lists, representing the k-nearest-neighbors found
        in each tree of this forest. The length of the outer list is the size
        of the forest (# trees), and the length of the inner lists is K.
        @note: Note that there is likely to be duplicate entries, as many trees will agree
        that a give sample is a k-neighbor of the probe.
        '''
        res = [ t.getKNearest(T,K) for t in self.trees]
        KNN_List = []
        for knn in res:
            for item in knn: KNN_List.append(item)
        return KNN_List
        
    def getKNearestFromEachTree_parallel(self, T, K):
        '''
        @return: A list of lists, representing the k-nearest-neighbors found
        in each tree of this forest. The length of the outer list is the size
        of the forest (# trees), and the length of the inner lists is K.
        @note: Note that there is likely to be duplicate entries, as many trees will agree
        that a give sample is a k-neighbor of the probe.
        '''
        KNN_List = [] #there will be a KNN list for each tree in forest...
        qList = [ Queue() for _ in self.trees ]
        procList = [Process(target=self._getKNN, args=(self[idx],T,K,qList[idx])) for idx in range(self.N)]
        
        for proc in procList: proc.start()
        for proc in procList: proc.join()
        
        res = [ Q.get() for Q in qList]  #the KNN for each tree in forest
        
        for KNN in res:
            for item in KNN: KNN_List.append(item)
        return KNN_List
    
    def getKNearest(self, T, K, parallel=False):
        '''
        Returns the K-nearest-neighbors in the forest.
        '''
        if parallel:
            #sometimes this is bad if the multiprocess start/join overhead
            # outweights the speedup.
            KNN_List = self.getKNearestFromEachTree_parallel(T, K)
        else:
            KNN_List = self.getKNearestFromEachTree_serial(T, K)
        KNNs = list(set(KNN_List))  #remove duplicates b/c many trees will return the same answer as closest, etc.
                
        return sorted(KNNs)[0:K] #like this, if K=3: [ (d1,k1), (d2,k2), (d3,k3)]  

class ProximityForestClassifier():
    """
    KNN classification using a ProximityForest index
    """
    def __init__(self, N=15, K=3, Tau=15, item_wrapper=PF_Item):
        """
        Constructor
        @param N: The number of trees in the forest
        @param K: The number of nearest neighbors to use in classification
        @param Tau: The splitting threshold of the forest. Leaves in the forest will
        have no more than (Tau-1) items.
        @param item_wrapper: The class used to wrap the samples, as per
        the default PF_Item class. The class must be hashable and implement
        a dist method that returns a scalar distance between two samples.
        """
        self.N = N 
        self.Tau = Tau
        self.K = K
        self.item_wrapper = item_wrapper
        self.forest = ProximityForest(N, item_wrapper=item_wrapper, Tau=Tau)
        
    def fit(self, X, L):
        """
        Indexes the set of samples in X with associated labels in L, thus
        "fitting" an Approx. Nearest Neighbor model to the data.
        @param X: A list of input samples, which could be most anything
        @param L: A corresponding list of integer labels representing class membership
        """
        self.forest.addList(X, L)
        
    def _predictOne(self, T):
        """
        Internal method that generates the KNN prediction for a single test sample T
        """
        KNNs = self.forest.getKNearest(T, self.K)
        labels = [ self.forest.items_dict[k].label for (_,k) in KNNs]
        C = Counter(labels)
        return C.most_common(1)[0][0] #the class of the first most common label in the KNN
    
    def predict(self, Y):
        """
        Predict the label for all samples in list Y
        """
        return [ self._predictOne(y) for y in Y ]
    
    def _scoreOne(self, T, L):
        """
        Internal method that returns 1 if predict(T) == L else 0.
        In other words, the score is the number of correct predictions.
        @param T: A single sample to test
        @param L: The true class label of T
        @return: 1 if the prediction is correct, 0 if incorrect
        """
        p = self._predictOne(T)
        return 1 if (p==L) else 0

    def score(self, Y, L):
        """
        Returns the accuracy for the KNN prediction
        of the samples in Y given ground truth labels in L.
        @param Y: The input samples list to test
        @param L: The ground truth class labels
        @return: The mean accuracy, computed as the number of correct
        predictions divided by the total number of predictions.
        """
        M = float( len(L) )
        hits = sum( [ self._scoreOne(y,lbl) for (y,lbl) in zip(Y,L)] )
        return (hits/M)
        
        
if __name__ == '__main__':
    pass