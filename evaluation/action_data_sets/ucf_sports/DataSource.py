'''
Created on Nov 3, 2011
@author: Stephen O'Hara
Interface code for working with the UCF Sports data set
'''
import os
import pyvision as pv
import scipy as sp

import sys
import glob
import csv
import cPickle

UCF_DATA_DIR = os.path.join(os.path.dirname(__file__),'data')

class UCFData:

    def __init__(self, source_dir, cube=(64,32,32)):
        '''
        @param sdir: the root directory of the data set
        @param cube: the size of the tensor (frames,x,y)
        '''
        self.dir = source_dir
        self.cube = cube
        
        self.subdirs = ("Diving-Side","Golf-Swing-Back","Golf-Swing-Front",
                        "Golf-Swing-Side", "Kicking-Front", "Kicking-Side",
                        "Lifting", "Riding-Horse", "Run-Side", "SkateBoarding-Front",
                        "Swing-Bench", "Swing-SideAngle", "Walk-Front")
        self.num_samples = (14, 5, 8, 5, 10, 10, 6, 12, 13, 12, 20, 13, 22)
        self.classmap = {0:0, 1:1, 2:1, 3:1, 4:2, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11:8, 12:9}
 
        #Note: we combine all the golf directories into a single class,
        # both kicking into a single class.
        # Sing-Bench is pommel horse, Swing-SideAngle is swinging on high bars
        self.classes = ("dive", "golf", "kick", "lift", "ride", "run", "skateboard", "pommel", "highbars", "walk")
        self.class_sizes = (14, 18, 20, 6, 12, 13, 12, 20, 13, 22)

    def _getClassIdx(self, subdir_idx):
        '''
        Given the subdirectory idx (into self.subdirs), returns
        the corresponding class index
        '''
        return self.classmap[subdir_idx]        

    def _getRect(self, vidpath, imgname):
        '''
        most of the videos in the UCF data set have a "gt" (ground truth)
        subdirectory off the video directory. In the gt directory, there is
        a text file corresponding to each image file that has the action
        bounding box and label
        '''
        gt_dir = os.path.join(vidpath,"gt")        
        if os.path.isdir(gt_dir):
            tmp = os.path.basename(os.path.splitext(imgname)[0])
            gt_file_name = gt_dir+"/%s.tif.txt"%tmp
            gtReader = csv.reader( open(gt_file_name,'rb'), delimiter='\t')
            row = gtReader.next()
            x = int(row[0])
            y = int(row[1])
            w = int(row[2])
            h = int(row[3])
            return pv.Rect(x,y,w,h)
        else:
            print("Error: No gt subdirectory for %s"%vidpath)
            return None
        
    def _getRectIdx(self, vidpath, idx):
        '''
        as per _getRect, but used when we don't know the name, but rather
        we have the index of which frame in the sorted list we need
        annotations for.
        '''
        gt_dir = os.path.join(vidpath,"gt")        
        if os.path.isdir(gt_dir):
            gtList = sorted( glob.glob(gt_dir+"/*.txt"))
            gt_file_name = os.path.join(gt_dir,gtList[idx])
            gtReader = csv.reader( open(gt_file_name,'rb'), delimiter='\t')
            row = gtReader.next()
            x = int(row[0])
            y = int(row[1])
            w = int(row[2])
            h = int(row[3])
            return pv.Rect(x,y,w,h)
        else:
            print("Error: No gt subdirectory for %s"%vidpath)
            return None        
                
    def loadVid(self, subdir="Diving-Side", vid="001" ):
        '''
        @param subdir: a member of self.subdirs
        @param vid: the string representation of which video to load within the given subdir
        '''

        print 'Loading video: %s/%s'%(subdir, vid)
        vidpath = os.path.join(self.dir, subdir, vid)
        if not os.path.isdir(vidpath):
            print "Error: Specified subdir/vid doesn't exist."
            return None
        imageStack = []
        
        '''        
        tmp = glob.glob(vidpath+"/*.avi")  #there should be exactly one
        if len(tmp) < 1:
            print "Error: Can't find an avi in the specified directory"
            return None
        
        
        video = pv.Video(os.path.join(vidpath,tmp[0]))
        for i,img in enumerate(video):
            rect = self._getRectIdx(vidpath, i)
            if rect==None:
                return None
            tile = img.crop(rect, (self.cube[1],self.cube[2]))
            tile.show(window="debug", delay=0)
            print "Frame %d"%i
            imageStack.append(tile)        
        '''
        
        imageList = sorted(glob.glob(vidpath+"/*.jpg"))
        if len(imageList) < 1:
            #then try the jpeg subdirectory, which some videos use, others do not...
            imageList = sorted(glob.glob(vidpath+"/jpeg/*.jpg"))
            
        for fn in imageList:
            #print "DEBUG: Loading image: %s"%fn
            #sys.stdout.flush()
            img = pv.Image( os.path.join(vidpath, fn))
            rect = self._getRect(vidpath, fn)
            if rect==None:
                return None
            tile = img.crop(rect, (self.cube[1],self.cube[2]))
            imageStack.append(tile)
  
        print "Image Stack size: %d"%len(imageStack)
        viddat = sp.zeros( self.cube )
        
        #we might have too few images in the stack for the desired cube size,
        # in which case we need to build a cube of what we have and then repeat
        # frames from the beginning...so we use the index modulo len(imageStack)
        maxx = len(imageStack)
        for idx in range(self.cube[0]):
            idx_mod = idx%maxx
            viddat[idx,:,:] = imageStack[idx_mod].asMatrix2D()
            
        return viddat
          
    def LoadDataSet(self):
        '''
        Loads all videos for the data set.
        '''
        
        #NF, Xs, Ys = self.cube
        #vidsPerSet = 180  #20 videos in each of 9 classes = 180 per set
        self.data = []        
        self.labels = []

        print 'Loading UCF Actions videos...'
        for si,subdir in enumerate(self.subdirs):
            print 'Processing %s'%subdir
            sys.stdout.flush()
            c = self._getClassIdx(si)
            for v in range(self.num_samples[si]):
                vid = str(v+1).zfill(3)  #video ids are 1-based
                tnsr = self.loadVid(subdir=subdir, vid=vid)
                if tnsr == None:
                    print "Skipping %s/%s because it lacks gt information."%(subdir,vid)
                    continue
                self.data.append(tnsr)
                self.labels.append(c)
    
    def getLabels(self):
        return self.labels
    
    def getData(self):
        return self.data
    
    def play_tensor(self, tnsr, window="UCF Action", size=(96,96), delay=35):
        dims = tnsr.shape
        print dims
        for v in range(dims[0]):
            fMat = tnsr[sp.ix_([v],range(dims[1]), range(dims[2]))].copy()
            img = pv.Image(sp.mat(fMat))
            img.show(window=window, size=size, delay=delay)

        
     
def loadUCFData(source_dir, cube = (64,32,32)):
    print "Loading UCF Videos..."
    UDat = UCFData(source_dir=source_dir, cube=cube)  #unlike others, only 1 set
    UDat.LoadDataSet()
    return UDat

def pickleData(UDat,filen=os.path.join(UCF_DATA_DIR,"UDat_64x32x32.p")):
    cPickle.dump(UDat, open(filen,'wb'), protocol=-1)

def unPickleData(filen=os.path.join(UCF_DATA_DIR,"UDat_64x32x32.p")):
    return cPickle.load(open(filen,'rb'))
      
    
if __name__ == '__main__':
    pass


