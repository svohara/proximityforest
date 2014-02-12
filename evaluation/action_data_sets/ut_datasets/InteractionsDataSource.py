'''
Interface object for working with the UT Interactions Data Set
Created on Sep 19, 2011
@author: Stephen O'Hara
'''
import os
import pyvision as pv
import numpy as np
import cPickle
import glob
import proximityforest as pf

UTI_DATA_DIR = os.path.join(os.path.dirname(__file__),'data/interactions')

class UTInteractData:
    def __init__(self, source_dir, cube=(48,32,32)):
        '''
        @param dir: the root directory of the data set
        @param cube: the size of the tensor (frames,x,y)
        '''
        self.dir = source_dir
        self.cube = cube     
        self.actionTxt = ['hand-shake', 'hugging', 'kicking', 'pointing', 'punching', 'pushing']
            
    def loadVid(self, action=0, seq=1):
        '''
        @param action: which class index 0-5
        @param seq: which sequence to load 1-20. Note: 1-10 are from 'set 1' and 11-20
        are from 'set 2'.
        '''
        assert action in range(6), "action parameter must be from 0-5"
        assert seq-1 in range(20), "seq parameter must be from 1-20"
        
        print "Loading Video for Sequence %d, Action %d (%s)."%(seq,action,self.actionTxt[action])
        #action_str = str(action).zfill(2)
        #seq_str = str(seq).zfill(2)
        if seq>10:
            set_dir = "segmented_set2"
        else:
            set_dir = "segmented_set1"
            
        source_dir = os.path.join(self.dir,set_dir)    
        
        #source files are formatted as index_seq_action.avi, where index is a fairly
        # useless number based on the order the action was segmented from the long video.
        # The problem is that we can't predict the index because the actions come in
        # different orders. But there should only be one file per seq/action, so we
        # can find it with a wildcard search *_seq_action.avi
        filelist = glob.glob("%s/*_%d_%d.avi"%(source_dir,seq,action))        
        if len(filelist) > 1:
            print "Error: We found more than one source video for the given sequence and action."
            return None        
        elif len(filelist) < 1:
            print "Error: Specified sequence/action pair not found in source files."
            return None
        
        #now we have the source avi, we need to load each frame    
        print "Loading from avi file: %s"%filelist[0]
        frameList = []
        vid = pv.Video(filelist[0])
        for f in vid:
            frameList.append(f)
            
        numFrames = len(frameList)
        F, Xs, Ys = self.cube
        
        if numFrames > F:
            startIdx = int(np.ceil( (numFrames-F)/2))
            f_range = range( startIdx, startIdx+F)
        else:
            startIdx = 0
            f_range = range( startIdx, startIdx+numFrames)
        
        #print "DEBUG:"
        #print "Number of Frames: %d"%numFrames
        #print "Frame Range is: %d - %d"%(f_range[0], f_range[-1])
        imageStack = []        
        for idx,fn in enumerate(f_range):
            #print idx, fn
            img = frameList[fn]
            img = pf.crop_black_borders(img)
            #img.annotateLabel(pv.Point(10,10), "%d"%fn, color='white', background='black')
            #img.show(window="Input", delay=1)
            tile = img.resize((Xs,Ys))
            #tile.show(window="Tile", delay=30)
            imageStack.append(tile)

        #we might have too few images in the stack for the desired cube size,
        # in which case we need to pad the cube of what we have by repeating
        # frames from the beginning...so we use the index modulo len(imageStack)
        viddat = np.zeros( self.cube )
        maxx = len(imageStack)
        for idx in range(self.cube[0]):
            idx_mod = idx%maxx
            viddat[idx,:,:] = imageStack[idx_mod].asMatrix2D()
            
        return viddat     

    
    def LoadDataSet(self, seq=-1, nclass=6):
        '''
        @param seq: Loads all videos for the specified sequence. Use -1 for all sequences,
        otherwise choose from 1-20. Note: sequences 1-10 are set1, sequences 11-20 are set2.
        @param nclass: if you wish to limit the number of classes, default to all 6
        '''
        self.data = []        
        self.labels = []
        self.loadErrors = []
        if seq == -1:
            for s in range(20):
                print "Sequence %d"%(s+1)
                for c in range(nclass):
                    tnsr = self.loadVid(action=c, seq=s+1)
                    if tnsr != None:
                        self.data.append(tnsr)
                        self.labels.append(c)
                    else:
                        self.loadErrors.append( (s+1, c))
        else:
            print 'Loading UT-Tower Actions from sequence %d'%seq
            for c in range(nclass):
                tnsr = self.loadVid(action=c, seq=seq)
                if tnsr != None:
                    self.data.append(tnsr)
                    self.labels.append(c)
                else:
                    self.loadErrors.append( (seq,c) )
     
    def getLabels(self):
        return self.labels
    
    def getData(self):
        return self.data
    
    def play_vid(self, idx=0, size=None):
        if self.data == None:
            print "Error: No data loaded in object. Call LoadDataSet() first."
            return
        tnsr = self.data[idx]
        dims = tnsr.shape
        for v in range(dims[0]):
            fMat = tnsr[np.ix_([v],range(dims[1]), range(dims[2]))].copy()
            img = pv.Image(np.mat(fMat))
            img.show(window="UT-Tower Action", size=size, delay=34)
    
    def test_play_vid(self, tnsr, window="Action", size=None):
        dims = tnsr.shape
        print dims
        for v in range(dims[0]):
            fMat = tnsr[np.ix_([v],range(dims[1]), range(dims[2]))].copy()
            img = pv.Image(np.mat(fMat))
            img.show(window=window, size=size, delay=34)
        
    def test_load_one_vid(self):
        tnsr = self.loadVid(1,1)
        self.test_play_vid(tnsr)
            
    def test_load_one_set(self, s=5):
        self.LoadDataSet(seq=s)
        print "Number of labels %d"%len(self.labels)
        count = 1
        for vid in self.data:
            print "Playing video %d"%count
            print "Label: " + self.actionTxt[ self.labels[count-1] ]
            self.test_play_vid(vid)
            count += 1
    
 
def loadUTInteractData(source_dir, cube = (48,32,32)):
    print "Loading UT Interaction Videos..."
    tData = [] #list of UT-Interactions objects 1 per sequence (20 total sequences)
    for i in range(20):
        tmp = UTInteractData(cube=cube, dir=source_dir)
        tmp.LoadDataSet(seq=i+1)  #all 6 classes, 1 vid per class per seq
        tData.append(tmp)        
    return tData
        
def pickleData(vData, filen=os.path.join(UTI_DATA_DIR,"UTInteractDat_48x32x32.p")):
    '''Save the UT Interaction data object, such as is produced with
    the loadUTInteractData() function to a file.'''
    return cPickle.dump(vData, open(filen,'wb'), protocol=-1)
        
def unPickleData(filen=os.path.join(UTI_DATA_DIR,"UTInteractDat_48x32x32.p")):
    ''' Load the pre-computed UT Interaction Data video tensors from a pickled file '''
    return cPickle.load(open(filen,'rb'))
            
if __name__ == '__main__':
    pass




