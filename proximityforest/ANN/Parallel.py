'''
Created on Apr 23, 2012
@author: Stephen O'Hara

Uses iPython parallel library to implement a parallel version
of a proximity forest. The parallelized version will have nearly
the same interface as the serial implementation.

See the iPython website for details on setting up a cluster of
computational nodes using iPython parallel.
http://ipython.org/ipython-doc/dev/parallel/index.html

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

class ParallelProximityForest(pf.ProximityForest):
    '''
    A parallelized version of a proximity forest.
    Public interface is same as ProximityForest, but
    requires a handle to an iPython parallel client
    object for distributing the workload. Also note
    that the design assumes there is a shared disk
    area common to all computing nodes.
    '''
    def __init__(self, ipp_client, N, treeClass=pf.ProximityTree, dist_func=None, **kwargs):
        '''
        Constructor
        All parameters except one are the same as for the base class ProximityForest. Also,
        there is no trees kwarg because we dont want to instantiate large trees on one node
        and then push them to other computing nodes. Instead, if you want to instantiate a
        parallel forest from saved files, use the "loadParallelProximityForest" function instead.
        @param ipp_client: The ipython parallel client object, which will have information about
        the available computing engines in the environment.
        @type ipp_client: IPython.parallel.client
        '''
        self.client = ipp_client
        self.tree_kwargs = kwargs        
        self.treeClass = treeClass
        self.N = N
        self.treeDistrib = self._computeTreeDistribution()
        cidxs = range(len(self.client))
        
        print "Clearing old data from remote engines..."
        dview = self.client[:]
        dview.execute("import proximityforest as pf", block=True)
        try:
            dview.execute("del(forest); del(samples); del(labels)")
            dview.purge_results('all')
        except:
            print "...deletion operation failed."
            pass #likely the variables don't exist yet on remote nodes
        
        funcName = dist_func.func_name
        
        for ci in cidxs:
            print "Creating new forest on node %d..."%ci
            forest = pf.ProximityForest(self.treeDistrib[ci], treeClass=treeClass, **kwargs)
            self.client[ci].push({funcName:dist_func},block=True)
            self.client[ci].push({'forest':forest},block=True)
            
        #Hook up the custom distance function. I am unable to figure out how to do that
        # in one shot. It may be because when we push an object (like a forest) to another
        # computing node, it is pickled during transfer. Pickling doesn't like functions...
        #So we pass over a forest with no custom dist_func, and then manually hook it up.
        dview.execute('for tree in forest.trees: tree.dist_func=%s'%funcName, block=True)
            
    def __str__(self):
        return "%s of %d Trees divided over %d Nodes"%(type(self).__name__, self.N, len(self.client))
    
    def __len__(self):
        '''
        Return the number of trees in the forest.
        '''
        return self.N
    
    def __getitem__(self, i):
        '''
        Support indexing the forest to get the nth tree. forest[n] is nth tree.
        NOTE: Not currently implemented for parallel forests because we don't want
        to schlep around large trees between computing nodes.
        
        Descendant classes that create trees based only on indexes into a shared
        memory structure may be okay to override this method and return a copy of
        a tree built on a remote node...
        '''
        raise NotImplementedError
      
    def _computeTreeDistribution(self):
        '''
        Given the desired number of trees and the number of computing nodes available
        in the ipp_client, this computes a list that indicates how many trees we need
        to distribute to each client.
        '''
        treeDist = [ self.N / len(self.client) ] * len(self.client)
        for i in range( self.N % len(self.client)):
            treeDist[i] += 1
        return treeDist               
        
    def clear(self):
        dview = self.client[:]
        dview.execute('forest.clear()', block=True) #wait until all remote forests are done
        
    def add(self, T, Label):
        for cx in range(len(self.client)):
            self.client[cx].push({'T':T, 'Label':Label}, block=True)
            
        dview = self.client[:]
        dview.execute('forest.add(T,Label)', block=True)  #wait until all remote forests are done
        
    def addList(self, samples, labels):
        print "Pushing samples and labels to remote computing nodes..."
        for cx in range(len(self.client)):
            self.client[cx].push({'logfile':'proximity_forest_%d_build.log'%cx})
            self.client[cx].push({'samples':samples, 'labels':labels}, block=True)
            
        print "Adding samples to remote forests..."
        dview = self.client[:]
        dview.execute('forest.addList(samples,labels, build_log=logfile)', block=True) #wait until all remote forests are done
        
    def save(self, base_dir, forest_name):
        '''
        Saves the forest as a set of files in the forest_name
        subdirectory of base_dir. NOTE: It is assumed that the computing nodes share
        a common storage system so that base_dir will resolve to the same directory
        for all nodes.
        The following files will be created
        1) tree_<num>.p, one pickle file for each tree in the forest,
        and <num> will be a zero-padded number, like 001, 002, ..., 999.
        2) forest_info.p, a single file with information about the forest        
        '''
        d = os.path.join(base_dir,forest_name)
        if not os.path.exists(d): os.makedirs(d, 0777)
        
        for cx in range(len(self.client)):
            #each node will have a different forest_idx value, so that their trees will not overwrite each other
            self.client[cx].push({'base_dir':base_dir, 'forest_name':forest_name, 'forest_idx':cx}, block=True)
        
        dview = self.client[:]    
        dview.execute('forest.save(base_dir,forest_name,forest_idx)', block=True) #wait until all nodes finished saving

        print "Forest saved to directory %s"%d 
        
    def getKNearestFromEachTree(self, T, K):
        '''
        @return: A list of lists representing the k-nearest samples to T that are
        found in each tree in the forest.
        '''
        dview = self.client[:]
        dview.block = True
        dview['T'] = T
        dview['K'] = K
        dview.execute('knnList=forest.getKNearestFromEachTree(T,K)', block=True)
        
        KNN_List = []
        for ci in range(len(self.client)):
            tmp = self.client[ci]['knnList']  #get results from sub-forest on computing node ci
            KNN_List += tmp
            
        return KNN_List
        
    def getKNearest(self, T, K):
        '''
        Returns the K-nearest-neighbors in the forest.
        '''
        dview = self.client[:]
        dview.block = True
        dview['T'] = T
        dview['K'] = K
        dview.execute('knn=forest.getKNearest(T,K)', block=True)
        
        KNN_List = []
        for ci in range(len(self.client)):
            KNN = self.client[ci]['knn']  #get results from computing node ci
            for item in KNN: KNN_List.append(item)
        
        KNNs = list(set(KNN_List))  #remove duplicates b/c many trees will return the same answer as closest, etc.                
        return sorted(KNNs)[0:K] #like this, if K=3: [ (d1,T1,L1), (d2,T2,L2), (d3,T3,L3)] 
       

def loadParallelProximityForest(base_dir, forest_name, dist_func=None, tree_idxs=None):
    '''
    Loads a saved proximity forest into a ParallelProximityForest structure distributed over
    a set of computing nodes.
    '''
    raise NotImplemented      


    
    