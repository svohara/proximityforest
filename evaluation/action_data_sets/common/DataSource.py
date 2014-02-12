'''
Created on Apr 26, 2012
@author: Stephen O'Hara

Base class for interfacing with action data sets, which can be
extended with subclasses for specific loading mechanisms.
'''
import numpy
import pyvision as pv

class AbstractActionData(object):
    '''
    A base class with common method definitions to be extended by the different
    specific data sources.
    '''
    def __init__(self, source_dir, nSets, nClasses, nVids, cube=(32,20,20) ):
        '''
        Constructor
        @param soure_dir: the root directory of the data set
        @param cube: the size of the tensor (frames,x,y)
        @param nSets: The number of sets in the data. Often data is partitioned into
        sets for training/validation/testing purposes.
        @param nClasses: The number of classes in the data
        @param nVids: The number of videos per class per set. In other words, the
        number of samples of each class of action in a single set of the data.
        If required, subclasses of this class may alter the definition of this
        and other parameters as required based on the data structure.
        '''
        self.dir = source_dir
        self.cube = cube
        self.nSets = nSets
        self.nClasses = nClasses
        self.nVids = nVids
        self.data = []
        self.labels = []
        self.tracklet_info = {}
            
    def getVideo(self, s=0, c=0, v=0):
        '''
        Returns a pyvision video object for the specified sample, usually so 
        that the user can easy view a single data sample. Calling this method,
        unlike loadvid(...), will NOT add a trackle to the self.data/self.labels
        variables.
        @param s: which set to use 0..nSets-1
        @param c: which class index 0..nClasses-1
        @param v: which video within this set/class to load 0..nVids-1
        @return: An object which implements the pyvision video interface, allowing 
         for easy playback or other processing.
        '''
        raise NotImplementedError
    
    def playVid(self, s=0, c=0, v=0):
        '''
        Plays the specified video sample (as opposed to a tracklet extracted
        from the sample). This method does not affect the internal state of the object
        (i.e., adds no data to the self.data/self.labels variables)
        '''
        v = self.getVideo(s, c, v)
        v.play(window="Source Video Playback", delay=30)
        
            
    def extractTracklet(self, s=0, c=0, v=0 ):
        '''
        Extract the tracklet(s) from a specified sample.
        @param s: which set to use 0..nSets-1
        @param c: which class index 0..nClasses-1
        @param v: which video within this set/class to load 0..nVids-1
        @return: A tracklet of dimension self.cube
        '''
        raise NotImplementedError     
    
    def _makeTracklet(self, imageStack):
        '''
        we might have too few images (or too many) in the stack for the desired cube size.
        If too few, we need to pad the tracklet by repeating frames from the beginning...
        so we use the index modulo len(imageStack). If too many frames, then we simply
        take the middle set of frames for the tracklet.
        @param imageStack: An ordered list of thumbnail-sized images to be used in the
        tracklet.
        @return: A Tracklet (numpy 3D ndarray) of fixed number of frames.
        '''
        tracklet = numpy.zeros( self.cube )
        if isinstance(imageStack, pv.ImageBuffer):
            stackLen = imageStack.getCount()
        else:
            stackLen = len(imageStack)
        
        if stackLen <= self.cube[0]:
            #repeat frames as necessary
            for idx in range(self.cube[0]):
                idx_mod = idx%stackLen
                #print idx_mod, stackLen, self.cube[0]
                try:
                    tracklet[idx,:,:] = imageStack[idx_mod].asMatrix2D()  
                except:
                    return None
        else:
            #take middle frames
            offset = int(numpy.floor( (stackLen-self.cube[0])/2))
            for idx in range(self.cube[0]):
                #print offset, idx, idx+offset, stackLen, self.cube[0]
                try:
                    tracklet[idx,:,:] = imageStack[idx+offset].asMatrix2D()
                except:
                    for i in range(stackLen): print i, type(imageStack[i])
                    imageStack.show(N=12, delay=0)
                    return None
                
        
        return tracklet        
    
    def loadDataSet(self, sidx=-1, preview=False):
        '''
        Loads the tracklets for all videos for the specified set. 
        @param sidx: which set to load, zero-based indexing. Specify -1 to load all sets.
        @param preview: if True, then the tracklet being loaded will be displayed in a preview playback window
        @note: self.data will contain the tracklets and self.labels the corresponding labels.
        '''
        
        #NF, Xs, Ys = self.cube
        #vidsPerSet = 180  #20 videos in each of 9 classes = 180 per set
        self.data = []        
        self.labels = []
        self.tracklet_info = {}
        counter=0
        if sidx == -1:
            print 'Loading tracklets from all sets'
            for s in range(self.nSets):
                print s, self.sets[s]
                #print "Processing %s"%self.sets[s]
                for c in range(self.nClasses):  #classes
                    for v in range(self.nVids): #vids
                        tnsr = self.extractTracklet(s, c, v)
                        if tnsr is None:
                            print "Warning: Sample %s produced no tracklets."%str((s,c,v))
                            continue  #sometimes there are missing videos in data sets (like KTH)
                        
                        if type(tnsr) == list:
                            #this data source produces multiple tracklets per video sample
                            self.labels += [c]*len(tnsr)
                            self.data += tnsr
                            if preview: self._play_tracklet_montage(tnsr)  
                            #buddies = [ i+counter for (i,_) in enumerate(tnsr) ] #all tracklets from this video
                            for _ in range(len(tnsr)):
                                self.tracklet_info[counter] = {'source':(s,c,v) } 
                                counter += 1                           
                        else:                            
                            #this data source produces one tracklet per video sample
                            self.data.append(tnsr)
                            self.labels.append(c)
                            if preview: self._play_tracklet(tnsr)
                            self.tracklet_info[counter] = {'source':(s,c,v)} 
                            counter += 1
        else:
            print 'Loading tracklets from set %d'%sidx
            for c in range(self.nClasses):  # classes
                for v in range(self.nVids): # vids
                    tnsr = self.extractTracklet(sidx, c, v)
                    if tnsr is None:
                            print "Warning: Sample %s produced no tracklets."%str((sidx,c,v))
                            continue  #sometimes there are missing videos in data sets (like KTH)                    
                    
                    if type(tnsr) == list:
                            #this data source produces multiple tracklets per video sample
                            self.labels += [c]*len(tnsr)
                            self.data += tnsr
                            if preview: self._play_tracklet_montage(tnsr)  
                            #buddies = [ i+counter for (i,_) in enumerate(tnsr) ] #all tracklets from this video
                            for _ in range(len(tnsr)):
                                self.tracklet_info[counter] = {'source':(sidx,c,v) } 
                                counter += 1                                
                    else:                            
                        #this data source produces one tracklet per video sample
                        self.data.append(tnsr)
                        self.labels.append(c)
                        if preview: self._play_tracklet(tnsr)
                        self.tracklet_info[counter] = {'source':(sidx,c,v) }
                        counter += 1
    
    def getLabels(self):
        return self.labels
    
    def getData(self):
        return self.data
    
    def getTrackletInfo(self, tracklet_idx):
        return self.tracklet_info[tracklet_idx]
    
    def _play_tracklet(self, tracklet, window="Action Tracklet", size=(100,100) ):
        '''
        Internal method for playing a tracklet video clip in window.
        '''
        vid = pv.VideoFromImageStack(tracklet, size=size)
        vid.play(window=window, delay=34)
        
    def _play_tracklet_montage(self, tracklets, window="Action Tracklets", layout=None, size=(100,100)):
        if layout is None:
            layout = (1, len(tracklets))
        
        #print "Number of tracklets to play %d"%len(tracklets) 
        #print type(tracklets[0])
        #videoDict = {}
        #for i,t in enumerate(tracklets):
        #    videoDict[i]= pv.VideoFromImageStackt
           
        videoDict = dict([ (i,pv.VideoFromImageStack(t)) for (i,t) in enumerate(tracklets)])
        vm = pv.VideoMontage(videoDict, layout=layout, tileSize=size)
        vm.play(window=window, delay=34)
        
    def test_extract(self, s, c, v):
        '''
        Convenience method for testing tracklet extraction on a single source video.
        This will not add the tracklet to the self.data/self.labels structures.
        '''
        tnsr = self.loadVid(s,c,v)
        self.test_play_vid(tnsr)
            
    def test_load_one_set(self, sidx=0, preview=True):
        '''
        Convenience method for testing loading of a single set from the data,
        and previewing the tracklets as they are loaded. This WILL reset the
        self.data/self.labels variables. Previously loaded tracklet data will be lost.
        '''
        self.loadDataSet(sidx, preview=preview)
        print "Loaded all tracklets from set %d"%sidx
        print "Number of tracklets: %d, number of labels: %d"%( len(self.data), len(self.labels))
        
        
        