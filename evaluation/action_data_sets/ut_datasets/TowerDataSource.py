'''
Interface object for working with the UT Tower Data Set
Created on Sep 14, 2011
@author: Stephen O'Hara
'''
import os
import pyvision as pv
import numpy as np
import cPickle
import scipy.io as spio
import glob
import proximityforest as pf


UTT_DATA_DIR = os.path.join(os.path.dirname(__file__),'data','tower')


class UTTowerData:
    def __init__(self, source_dir, cube=(32,48,48), padding=12, preview=False):
        '''
        @param dir: the root directory of the data set
        @param cube: the size of the tensor (frames,x,y)
        @param padding: how much additional pixels should be padded to the bounding boxes
        for clipping the tracks? Set to 0 for no additional padding.
        @param preview: True if you want to preview each tracklet as it is loaded into the data set.
        '''
        self.dir = source_dir
        self.cube = cube     
        self.bboxes = self.loadBoundingBoxes()
        self.padding = padding
        self.preview_on_load = preview
        self.actionTxt = ['pointing', 'standing', 'digging', 'walking', 'carrying',
                           'running', 'wave1', 'wave2', 'jumping']
        
    def loadBoundingBoxes(self):
        filename = self.dir+"/BBoxes.mat"
        x = spio.loadmat(filename, struct_as_record=True)
        bboxes = x['BBoxes']
        print "Loaded bounding box data, size: %s"%str(bboxes.shape)
        
        bbox_lookup = {}
        for row in range(bboxes.shape[0]):
            action = bboxes[row,0]
            seq = bboxes[row,1]
            frame = bboxes[row,2]
            key = "%d.%d.%d"%(action,seq,frame)
            rect = pv.Rect(bboxes[row,3],bboxes[row,4], bboxes[row,5], bboxes[row,6])
            bbox_lookup[key]=rect
        
        return bbox_lookup
    
    def clipFrame(self, image, frame, action, seq):
        ''' Crop the UT-Tower video frame to its corresponding bounding box.
        @param image: The input image to crop
        @param frame: The frame number, 1-based indexing.
        @param action: The action identifier {1-9}
        @param seq: The sequence identifier {1-12}
        @return: A pv.Image representing only the tile within the bounding box area.
        '''
        bkey = "%d.%d.%d"%(action,seq,frame)
        rect = self.bboxes[bkey]
        tile = image.crop(rect)
        #tile = imutil.crop_black_borders(tile)
        return tile
    
    def clipNarrowTracklet(self, filelist, action, seq, f_range):
        '''
        A narrow tracklet is the "basic" method for clipping image tiles from the
        source video by simply using the bounding boxes as specified per frame. Because
        we do so, translational motion will be eliminated (target is always centered in box),
        so differences in running vs. walking must be determined by gait motion only. The 
        problem with this method is that track jitter causes "bouncing" effects in many cases.
        '''        
        (_, Xs, Ys) = self.cube
        imageStack = []        
        for idx,fn in enumerate(f_range):
            img = pv.Image( filelist[fn] )
            rect = self.bboxes[self._makeBBKey(action, seq, fn+1)] #frames are 1-based in bbox dict
            tile = img.crop(rect)
            tile2 = tile.resize((Xs,Ys))
            imageStack.append(tile2)

        #we might have too few images in the stack for the desired cube size,
        # in which case we need to pad the cube of what we have by repeating
        # frames from the beginning...so we use the index modulo len(imageStack)
        tracklet = np.zeros( self.cube )
        MAX = len(imageStack)
        for idx in range(self.cube[0]):
            idx_mod = idx%MAX
            tracklet[idx,:,:] = imageStack[idx_mod].asMatrix2D()
        
        return tracklet       
        
    def clipWideTracklet(self, filelist, action, seq, f_range):
        ''' 
        A wide tracklet is where a single bounding box is used to clip the image tiles
        from the source video. The single bounding box is large enough to span the range of
        all the individual bounding boxes within the tracklet duration. This can greatly
        reduce "track bouncing/jitter" effects, and also allows for translational motion
        to be observed.
        @return: A wide tracklet
        '''        
        wide_rect = self.getWideRect(action,seq,f_range)
        
        (_, Xs, Ys) = self.cube
        imageStack = []        
        for idx,fn in enumerate(f_range):
            img = pv.Image( filelist[fn] )
            tile = img.crop(wide_rect)
            tile2 = tile.resize((Xs,Ys))
            imageStack.append(tile2)

        #we might have too few images in the stack for the desired cube size,
        # in which case we need to pad the cube of what we have by repeating
        # frames from the beginning...so we use the index modulo len(imageStack)
        tracklet = np.zeros( self.cube )
        MAX = len(imageStack)
        for idx in range(self.cube[0]):
            idx_mod = idx%MAX
            tracklet[idx,:,:] = imageStack[idx_mod].asMatrix2D()
        
        return tracklet
        
    def getWideRect(self, action, seq, f_range):
        '''
        Returns a single rectangle that spans the bounds of all rects
        for this action/seq in f_range
        '''
        minX=10000
        minY=10000
        maxX=0
        maxY=0
        for f in f_range:
            r = self.bboxes[ self._makeBBKey(action, seq, f+1)]  #bboxes wants 1-based indexing on frames
            if r.x<minX: minX = r.x
            if r.y<minY: minY = r.y 
            if r.x+r.w > maxX: maxX = r.x+r.w
            if r.y+r.h > maxY: maxY = r.y+r.h
            
        return pv.Rect(minX,minY,maxX-minX, maxY-minY)
        
    def loadVid(self, action=1, seq=1, wideTracklets=False):
        '''
        @param action: which class index 1-9
        @param seq: which sequence to load 1-12
        @return: A numpy 3D array (frames,x,y) representing the data cube tensor for the requested video.
        '''
        pad = self.padding
        print "Loading Video for Sequence %d, Action %d (%s)."%(seq,action,self.actionTxt[action-1])
        action_str = str(action).zfill(2)
        seq_str = str(seq).zfill(2)
        file_prefix = "ut-tower/Video_Frames/im%s_%s_"%(action_str,seq_str)
        filelist = sorted( glob.glob("%s/%s*.bmp"%(self.dir,file_prefix)) )
        
        numFrames = len(filelist)
        (F,_,_) = self.cube
        if numFrames > F:
            startIdx = int(np.ceil( (numFrames-F)/2))
            f_range = range( startIdx, startIdx+F)
        else:
            startIdx = 0
            f_range = range( startIdx, startIdx+numFrames)
            
        if wideTracklets:
            wide_rect = self.getWideRect(action,seq,f_range)
        
        #load list of source images and corresponding clipping rectangles
        # from the filelist
        imageList = []
        rectsList = []
        for fn in f_range:
            imageList.append( pv.Image( filelist[fn] ) )
            if not wideTracklets:
                tmp = self.bboxes[self._makeBBKey(action, seq, fn+1)] #frames are 1-based in bbox dict
                rect = pv.Rect(tmp.x-pad,tmp.y-pad, tmp.w+2*pad, tmp.h+2*pad)
            else:
                rect = wide_rect
            rectsList.append(rect)
        
        T = pf.makeTensorFromImages(imageList, rectsList, self.cube)
        
        if self.preview_on_load:            
            preview=pv.VideoFromImageStack(T, size=(100,100))
            preview.play(window="Tracklet", delay=35)
        
        return T   
      
    def _makeBBKey(self, action, seq, frame):
        return '%d.%d.%d'%(action,seq,frame)
    
    def LoadDataSet(self, seq=-1, nclass=9):
        '''
        @param seq: Loads all videos for the specified sequence. Use -1 for all sequences,
        otherwise choose from 1-12
        @param nclass: if you wish to limit the number of classes, default to all 9
        '''
        self.data = []        
        self.labels = []

        if seq == -1:
            for s in range(12):
                print "Sequence %d"%(s+1)
                for c in range(nclass):
                    tnsr = self.loadVid(action=c+1, seq=s+1)
                    self.data.append(tnsr)
                    self.labels.append(c)
        else:
            print 'Loading UT-Tower Actions from sequence %d'%seq
            for c in range(nclass):
                tnsr = self.loadVid(action=c+1, seq=seq)
                self.data.append(tnsr)
                self.labels.append(c)
     
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
    
#============================
# Stand-alone functions to
# interact with the data set.
#============================    
def loadUTTowerData(source_dir, cube = (32,48,48), **kwargs):
    print "Loading UT Tower Videos..."
    tData = [] #list of UT-Tower objects 1 per set (12 total sets/sequences)
    for i in range(12):
        tmp = UTTowerData(dir=dir, cube=cube, **kwargs)
        tmp.LoadDataSet(seq=i+1)  #all 9 classes, 1 vid per class per seq
        tData.append(tmp)        
    return tData
  
def pickleData(vData, filen=os.path.join(UTT_DATA_DIR,"TowerDat_32x48x48.p")):
    '''Save the UT Interaction data object, such as is produced with
    the loadUTInteractData() function to a file.'''
    return cPickle.dump(vData, open(filen,'wb'), protocol=-1)
        
def unPickleData(filen=os.path.join(UTT_DATA_DIR,"TowerDat_32x48x48.p")):
    ''' Load the pre-computed UT Interaction Data video tensors from a pickled file '''
    return cPickle.load(open(filen,'rb'))

            
if __name__ == '__main__':
    pass




