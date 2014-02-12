'''
Created on Apr 25, 2012
@author: Stephen O'Hara

This module contains functions for working with tracklet data objects.
Tracklets are thumbnail-sized short duration video segments which are
used to demonstrate action recognition and clustering using proximity
forest structures.

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
import cPickle

class TrackletData:
    '''
    A convenience class to deal with 'tracklets', which are short thumbnail-sized video clips,
    often something like 32x32 pixels and 48 frames long. Tracklets can be sorted using TensorSubspaceForests
    and used in clustering, etc., with the help of this class.
    '''
    def __init__(self, tracklets=[], tracklet_info={}, nocache=False):
        '''
        Constructor
        @param tracklets: A list of tracklet tensors
        @param tracklet_info: A dictionary {idx : info} which stores additional info on each tracklet
        @param nocache: If True, then tensor unfoldings will not be cached. This will decrease performance,
        but it will also use less memory.
        '''
        self.tracklets=tracklets
        self.tracklet_info=tracklet_info
        self.unfoldings = {}
        self.filename = None #set when data is loaded from a file
        self.nocache = nocache
        
    def getTracklet(self, idx):
        return self.tracklets[idx]
    
    def getInfo(self, idx):
        if idx in self.tracklet_info:
            return self.tracklet_info[idx]
        else:
            print "There is no tracklet information for index %d"%idx
            return None
        
    def add(self, tracklet, tracklet_info):
        '''
        Add a new tracklet to this collection and return the index
        '''
        idx = len(self)
        self.tracklets.append(tracklet)
        self.tracklet_info[idx] = tracklet_info
        return idx
        
    def remove(self, idx):
        '''
        Remove the tracklet, info, and unfoldings associated with the index
        '''
        del(self.tracklet_info[idx])
        del(self.tracklets[idx])
        for axis in [1,2,3]:
            if (idx, axis) in self.unfoldings:
                del( self.unfoldings[(idx, axis)])
    
    def clearCache(self):
        '''
        Removes all the stored unfoldings data from this object.
        '''
        del(self.unfoldings)
        self.unfoldings = {}
    
    def __len__(self):
        '''
        return the number of tracklets in the data structure
        '''
        return len(self.tracklets)
    
    def __getitem__(self, i):
        '''
        return the information about tracklet i
        '''
        return self.tracklet_info.__getitem__(i)
        
    def keys(self):
        '''
        return the list of indices in the td structure
        '''
        return range(len(self))
        
    def printInfo(self, idx):
        print "Track %d, Tracklet %d, Video %s"%self[idx]
    
    def getUnfolding(self, idx, axis):
        '''
        Retrieve or compute the unfolding of tracklet[idx] along axis 1, 2 or 3.
        If we've already done this unfolding since the tracklet data was loaded, it
        will exist in the self.unfoldings cache. Otherwise it will be computed and
        then added to the unfoldings cache.
        '''
        if (idx, axis) in self.unfoldings:
            M = self.unfoldings[(idx, axis)]
        else:
            M = pf.UnfoldCube(self.tracklets[idx], axis)
            M = M.T #we want rows>cols, which is transpose of what UnfoldCube provides
            if not self.nocache : self.unfoldings[(idx, axis)] = M            
        return M
    
    def play(self, idxList, layout=(5,8), size=(80,80)):
        ''' Show the tracklets in idxList in a video montage '''
        Ts = [ self.tracklets[idx] for idx in idxList ]
        pf.play_tracklets(Ts, layout=layout, size=size, labels=None, window="Tracklet Playback")
        
    def save(self, filename, clearCache=True):
        ''' Save the tracklets and tracklet_info data to a single pickle file
        @param clearCache: If True, the cache of tracklet unfoldings will be deleted
        prior to save. Future uses of this tracklet data object will cause unfoldings
        to be recomputed as required. False will save the unfoldings data as part of the object
        '''
        if clearCache:
            self.clearCache()
        cPickle.dump((self.tracklets, self.tracklet_info), open(filename,"wb"), protocol=-1)
        self.filename = filename
        
    def load(self, filename):
        ''' Load the tracklets and tracklet_info from the specified file.
        @note: This will OVERWRITE the current data in this object.
        '''
        print "Loading tracklet data and info from file %s."%filename
        (ts,t_info) = cPickle.load(open(filename,"rb"))
        self.tracklets = ts
        self.tracklet_info = t_info
        self.unfoldings = {}  #clear out old data
        self.filename = filename #remember the file we loaded the tracklet data from
        print "Load completed. There are now %d tracklets."%len(ts)
        
def build_simple_TD_Structure(tracklets, labels):
    '''
    Construct tracklet_data object using a list of tracklets and associated labels.
    This is used when the tracklet_info is essentially not needed, and can be replaced by the class label.
    In particular, I use this helper function when testing subspace forest classification on several
    benchmark action recognition data sets.
    '''
    idxs = range(len(tracklets))
    info = dict(zip(idxs,labels))
    return TrackletData(tracklets, info)   