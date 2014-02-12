'''
Interface object for working with the Cambridge Gestures Data Set
Created on Oct 14, 2010
Updated April 2011
@author: Stephen O'Hara
'''
from evaluation.action_data_sets.common.DataSource import AbstractActionData
import os
import pyvision as pv
import numpy
import cPickle
import glob
import proximityforest as pf

GESTURE_DATA_DIR = os.path.join(os.path.dirname(__file__),'data')

class GestureData(AbstractActionData):
    def __init__(self, source_dir, cube=(32,20,20)):
        '''
        Constructor
        @param source_dir: the root directory of the data set
        @param cube: the size of the tensor (frames,x,y)
        '''
        AbstractActionData.__init__(self, source_dir, 5, 9, 20, cube=cube)        
        self.sets = ('Set1','Set2','Set3','Set4','Set5')
        self.classes = [ str(i).zfill(4) for i in range(9)] # '0000',...'0008'
        self.vids = [ str(i).zfill(4) for i in range(20)]    # '0000'...'0019'
            
    def getVideo(self, s=0, c=0, v=0):
        '''
        See description from base class AbstractActionData
        '''
        cstr = self.classes[c]
        vstr = self.vids[v]
        setstr = self.sets[s]
        print 'Loading video: '+setstr+'/'+cstr+'/'+vstr
        
        vidpath = os.path.join(self.dir, setstr, cstr, vstr)
        imfiles = glob.glob(vidpath+'/*.jpg')
        return pv.VideoFromFileList( sorted(imfiles))
            
    def extractTracklet(self, s=0, c=0, v=0 ):
        '''
        Extract the tracklet(s) from a specified sample.
        @param s: which set to use 0..nSets-1
        @param c: which class index 0..nClasses-1
        @param v: which video within this set/class to load 0..nVids-1
        @return: A tracklet of dimension self.cube
        '''
        cstr = self.classes[c]
        vstr = self.vids[v]
        setstr = self.sets[s]
        print 'Loading video: '+setstr+'/'+cstr+'/'+vstr
        vidpath = os.path.join(self.dir, setstr, cstr, vstr)
        lst = os.listdir(vidpath)
        lst = sorted(lst)
        
        #filter list for only the jpg images in directory, sorted
        jpgs = lambda x: x.split('.')[1]=='jpg'
        lst = filter(jpgs, lst)
                
        numFrames = len(lst)
        F, _, _ = self.cube
        
        if numFrames > F:
            startFrame = int(numpy.ceil( (numFrames-F)/2))
        elif numFrames < F:
            print "Error: Video has too few frames!"
            raise ValueError("Video has only %d frames, you requested %d."%(numFrames,F))
        else:
            startFrame = 0
         
        f_range = range( startFrame, startFrame+F)       
        imageStack = []
        for idx in f_range:
            fname = 'frame-'+str(idx).zfill(4)+'.jpg'
            imageStack.append( pv.Image( os.path.join(vidpath,fname)) )
            
        T = pf.makeTensorFromImages(imageStack, rects=None, cube=self.cube)       
            
        return T      
     
def loadGestureData(source_dir, cube = (32,20,20)):
    print "Loading Gesture Videos..."
    gData = [] #list of GestureData objects 1 per set (5 total sets)
    for i in range(5):
        tmp = GestureData(source_dir=source_dir, cube=cube)
        tmp.loadDataSet(i)  #all 9 classes, 20 vids/class
        gData.append(tmp)        
    return gData
        
def pickleData(GDat, filen=os.path.join(GESTURE_DATA_DIR,"GDat_32x20x20.p")):
    '''Save the gesture data object, such as is produced with
    the loadGestureData() function to a file.'''
    return cPickle.dump(GDat, open(filen,'wb'), protocol=-1)
        
def unPickleData(filen=os.path.join(GESTURE_DATA_DIR,"GDat_32x20x20.p")):
    ''' Load the pre-computed Gesture Data video tensors from a pickled file '''
    return cPickle.load(open(filen,'rb'))

            
if __name__ == '__main__':
    pass

#gd = GestureData()
#gd.test_load_one_set(5)



