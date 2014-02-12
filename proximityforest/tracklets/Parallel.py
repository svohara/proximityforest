'''
Created on Apr 25, 2012
@author: Stephen O'Hara

This module provides the parallel implementation of TensorSubspaceForest

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
import glob
import cPickle


class ParallelTensorSubspaceForest(pf.ParallelProximityForest, pf.TensorSubspaceForest):
    '''
    Parallel implementation of TensorSubspaceForest
    '''
    def __init__(self, ipp_client, N, td_filename=None, treeClass=pf.TensorSubspaceTree, 
                 trees=None, no_tracklet_cache=True, **kwargs):
        '''
        Constructor
        Parameters are same as for parent class, but with two exceptions.
        @param ipp_client: The IPython parallel client object.
        @param td_filename: A filename of a stored TrackletData object
        that has the tracklet tensors and related information. Using a filename
        allows each computing node to separately load the tracklet data, assuming
        the filename is valid for all cluster nodes. Also, if there exists
        a tracklet_data object in global memory of a computing node, called
        'tracklet_data', and the tracklet_data.filename property is the
        same as this parameter, then the existing object will be re-used.
        @param no_tracklet_cache: If True (default) then the tracklet data object
        loaded by the remote nodes will not cache tensor unfoldings. For large
        tracklet data objects, this may be advisable to save memory.
        '''        
        self.client = ipp_client
        self.tree_kwargs = kwargs        
        self.treeClass = treeClass
        self.N = N
        self.td_filename = td_filename
        self.no_cache = no_tracklet_cache
        self.treeDistrib = self._computeTreeDistribution()
        for tN in self.treeDistrib:
            assert tN%3==0, "ERROR: You must have a multiple of 3 trees for each computing node."
                     
        dview = self.client[:]
        #load required modules in remote computing nodes
        dview.block = True
        dview.execute("import proximityforest as pf")
        
        end_idx = 0
        for ci in range(len(self.client)):
            print "Creating new forest on node %d..."%ci
            start_idx = end_idx
            end_idx = start_idx + self.treeDistrib[ci]
            dummy_td = pf.TrackletData()
            dummy_td.filename = self.td_filename
            if trees is None:
                forest = pf.TensorSubspaceForest(self.treeDistrib[ci], tracklet_data=dummy_td, treeClass=treeClass, **kwargs)
            else:
                forest = pf.TensorSubspaceForest(self.treeDistrib[ci], tracklet_data=dummy_td, 
                                                 trees=trees[start_idx:end_idx], treeClass=treeClass, **kwargs)
                
            self.client[ci].push({'forest':forest},block=True)
            
        self._setTrackletData(td_filename)
            
    def _getTrackletData(self):
        if self.td_filename is None:
            print "No tracklet data is currently associated with this forest."
            return None
        else:
            td = pf.TrackletData()
            td.load(self.td_filename)
            return td
        
    def _setTrackletData(self, td_filename):
        if td_filename is None:
            print "Can not set tracklet data for forest because None was provided."
            print "You must manually call _setTrackletData(td_filename) later."
            return
        else:
            self.td_filename = td_filename
            
        self._loadRemoteTD()
        self.client[:].execute('forest._setTrackletData(tracklet_data)', block=True)    
            
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
        dview.execute('forest.save(base_dir,forest_name,skip_tracklet_data=True,forest_idx=forest_idx)', block=True) #wait until all nodes finished saving

        print "Forest saved to directory %s"%d 
          
            
    def _loadRemoteTD(self):
        '''
        Internal method to encapsulate code for remotely loading tracklet_data where required.
        '''
        loadIdxs = []  #loadIdxs is the list of client nodes that need to load tracklet data
        for ci in range(len(self.client)):
            #test if remote computing node already has tracklet_data loaded
            self.client[ci].execute("tmp='tracklet_data' in dir()",block=True)
            if self.client[ci]['tmp']:
                self.client[ci].execute("tmp=tracklet_data.filename",block=True)
                if self.td_filename == self.client[ci]['tmp']:
                    print "Computing Node %d has tracklet data already in memory"%ci
                    continue
            loadIdxs.append(ci)
        
        if len(loadIdxs) < 1: return
        
        print "The following remote nodes are loading tracklet data file: %s"%str(loadIdxs)
        dview2 = self.client[loadIdxs]
        dview2.block = True
        dview2['tmp']=self.td_filename
        if self.no_cache:
            dview2.execute("tracklet_data=pf.TrackletData(nocache=True); tracklet_data.load(tmp)")
        else:
            dview2.execute("tracklet_data=pf.TrackletData(); tracklet_data.load(tmp)")
        

def loadParallelTensorSubspaceForest(ipar, base_dir, forest_name, tree_idxs=None):
    '''
    Initializes a parallel tensor subspace forest from a forest saved on disk
    The forest_name directory off base_dir holds the forest data and must also
    have the tracklet_data.p file.
    @param ipar: The ipython parallel client object
    @param base_dir: where the forest subdirectory lives
    @param forest_name: the name of the subdirectory in base_dir for the forest
    @param tree_idxs: A list of tree indexes to load into the forest, None means all.
    @return: A parallel tensor subspace forest, with trees evenly distributed
    over the number of nodes connected to by ipar. Forest parameters are the
    same as those loaded from the forest_info.p file.
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
    assert treeClass in [pf.TensorSubspaceTree, pf.TensorEntropySplitSubspaceTree]  #ensure that we're loading the right kind of trees
    td_file = os.path.join(fdir,"tracklet_data.p")
        
    #print "Loading tracklet data from %s"%td_file
    #(ts, t_info) = cPickle.load( open(td_file,"rb"))
    #tracklet_data = pf.TrackletData(ts, t_info)
    
    trees = []
    if tree_idxs is None:
        tree_idxs = range(N)
    
    for idx in tree_idxs:
        assert idx <= N-1
        tree_name = "tree_%s.p"%str(idx).zfill(3)
        print "Loading tree: %s..."%tree_name
        (t, _) = pf.loadTensorSubspaceTree( os.path.join(fdir,tree_name),None, skip_tracklets=True)
        trees.append(t)
    
    forest = pf.ParallelTensorSubspaceForest(ipar, N, td_file, no_tracklet_cache=True, trees=trees, **forest_info)
    
    return forest
    
    
