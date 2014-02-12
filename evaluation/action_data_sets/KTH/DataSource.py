'''
Interface object for working with KTH Actions Data Set
Using Yui Man Lui's annotations for defining the tracklet
selection from the source video clips.

Created on April 11, 2011
@author: Stephen O'Hara
'''
import os
import pyvision as pv
import scipy as sp
import cPickle

KTH_DATA_DIR = os.path.join(os.path.dirname(__file__),'data')


class KTHData:

    def __init__(self, source_dir, cube=(32,20,20)):
        '''
        @param source_dir: the root directory of the data set
        @param cube: the size of the tensor (frames,x,y)
        '''
        self.dir = source_dir
        self.cube = cube
        self.classes = ("walking", "jogging", "running", "boxing", "handclapping", "handwaving")
        self.sets = [] #each set is one of 25 subjects that demonstrates the action
        self.vids = ('Scene1','Scene2','Scene3','Scene4')  #4 'scenes' or repetitions per class/subject
        
        for i in range(25):
            self.sets.append( 'Person%d'%(i+1))
            
        self._initAnnotations( os.path.join(KTH_DATA_DIR,"lui_annotations"))
        
    def _initAnnotations(self, d):
        walking_annot = {}
        jogging_annot = {}
        running_annot = {}
        boxing_annot = {}
        handwaving_annot = {}
        handclapping_annot = {}
        self.annotations = (walking_annot, jogging_annot, running_annot,
                            boxing_annot, handwaving_annot, handclapping_annot)
        
        for cidx in range(6):
            cstr = self.classes[cidx]
            annot = self.annotations[cidx]
            #print "DEBUG: cidx=%d cstr=%s"%(cidx,cstr)
            self._buildAnnot(annot, os.path.join(d,"%s.txt"%cstr))
        
    def _buildAnnot(self, annot, filename):
        with open( filename, 'r') as f:
            print "Loading annotations from: %s."%filename
            while True:
                x = f.readline()
                if len(x) == 0: break #EOF
                x = x.split()
                if x == []: continue #blank line...
                
                subjIdx = int(x[0])-1  #annotation indexes are 1-based, but python is 0-based
                sceneIdx = int(x[1])-1
                start = int(x[2])
                dur = int(x[3])
                xpos = int(x[4])
                ypos = int(x[5])
                width = int(x[6])
                height = int(x[7])
                rect = pv.Rect(xpos,ypos,width,height)
                
                subjStr = self.sets[subjIdx]
                if not annot.has_key(subjStr):
                    annot[subjStr] = {}
                
                tmp_dict = annot[subjStr]                
                sceneStr = self.vids[sceneIdx]
                tmp_dict[sceneStr] = (start,dur,rect)
                #print x
            f.closed    
        
            
    def loadVid(self, s=0, c=0, v=0 ):
        '''
        @param s: which set to load 0..24 == Person1...Person25
        @param c: which class index 0..5 == "walking"..."handclapping"
        @param v: which scene video within the class/subject to load 0..3 == Scene1..4
        '''
        cstr = self.classes[c]
        vstr = self.vids[v]
        setstr = self.sets[s]
                
        if s==12 and c==4 and v==2: #Person13, Handclapping, Scene3
            #this is the missing KTH video...so we'll load a different section
            # of scene2
            print "NOTE: handclapping/Person13/Scene3 is the missing video."
            print "Loading a different section from Scene2 instead."
            print 'Loading video: '+cstr+'/'+setstr+'/Scene2'
            vidpath = os.path.join(self.dir, cstr, setstr, 'Scene2')
            v_alt = True
        else:
            print 'Loading video: '+cstr+'/'+setstr+'/'+vstr
            vidpath = os.path.join(self.dir, cstr, setstr, vstr)
            v_alt = False     
        
        class_annot = self.annotations[c]
        subj_annot = class_annot[setstr] #the subject info for this class
        (startF, dur, rect) = subj_annot[vstr]  #the scene info for this subject
        
        imageStack = []

        #load images in desired range from directory, resizing to desired thumbnail size
        for frameNum in [ i + startF for i in range(dur)]:
            if v_alt:
                fname = "person_2_%d.png"%(frameNum )
            else:
                fname = "person_%d_%d.png"%( (v+1),frameNum )
            img = pv.Image( os.path.join(vidpath,fname))
            tile = img.crop(rect, (self.cube[1],self.cube[2]))
            imageStack.append(tile)
        
        
        viddat = sp.zeros( self.cube )
        
        #we might have too few images in the stack for the desired cube size,
        # in which case we need to build a cube of what we have and then repeat
        # frames from the beginning...so we use the index modulo len(imageStack)
        maxx = len(imageStack)
        for idx in range(self.cube[0]):
            idx_mod = idx%maxx
            viddat[idx,:,:] = imageStack[idx_mod].asMatrix2D()
            
        return viddat      
    
    def LoadDataSet(self, sidx=-1, nclass=6, nvids=4):
        '''
        Loads all videos for the specified sidx. Use -1 for all sets,
        otherwise choose a sidx from 0 to 24
        @param sidx: which set index (subject) to load
        @param nclass: if you wish to limit the number of classes
        @param nvids: if you wish to limit the number of scenes / class
        '''
        
        #NF, Xs, Ys = self.cube
        #vidsPerSet = 180  #20 videos in each of 9 classes = 180 per sidx
        self.data = []        
        self.labels = []

        if sidx == -1:
            for s in range(25):
                print s, self.sets[s]
                #print "Processing %s"%self.sets[s]
                for c in range(nclass):  #6 classes
                    for v in range(nvids): #4 scenes
                        tnsr = self.loadVid(s, c, v)
                        self.data.append(tnsr)
                        self.labels.append(c)
        else:
            print 'Loading KTH Actions from %s'%self.sets[sidx]
            for c in range(nclass):  #6 classes
                for v in range(nvids): #4 scenes
                    tnsr = self.loadVid(sidx, c, v)
                    self.data.append(tnsr)
                    self.labels.append(c)
    
    def getLabels(self):
        return self.labels
    
    def getData(self):
        return self.data
    
    def test_play_vid(self, tnsr, window="KTH Action", size=None, delay=35):
        dims = tnsr.shape
        print dims
        for v in range(dims[0]):
            fMat = tnsr[sp.ix_([v],range(dims[1]), range(dims[2]))].copy()
            img = pv.Image(sp.mat(fMat))
            img.show(window=window, size=size, delay=delay)
        
    def test_load_one_vid(self):
        tnsr = self.loadVid(3,5,0)
        self.test_play_vid(tnsr)
            
    def test_load_one_set(self, sidx=5):
        self.LoadDataSet(sidx, nclass=2, nvids=2)
        print "Number of labels %d"%len(self.labels)
        count = 1
        for vid in self.data:
            print "Playing video %d"%count
            print "Label: " + self.classes[ self.labels[count-1] ]
            self.test_play_vid(vid)
            count += 1
        
#============================
# Stand-alone functions to
# interact with the KTH
# data set.
#============================        
def loadKTHData(source_dir, cube = (32,20,20)):
    print "Loading KTH Videos..."
    kData = [] #list of KTHData objects 1 per set (25 total sets)
    for i in range(25):
        tmp = KTHData(source_dir=source_dir, cube=cube)
        tmp.LoadDataSet(i, 6, 4)  #all 6 classes, 4 scenes/class/person
        kData.append(tmp)
        
    return kData

def pickleData(KDat,filen=os.path.join(KTH_DATA_DIR,"KDat_32x20x20.p")):
    cPickle.dump(KDat, open(filen,'wb'), protocol=-1)

def unPickleData(filen=os.path.join(KTH_DATA_DIR,"KDat_32x20x20.p")):
    return cPickle.load(open(filen,'rb'))

        
    
if __name__ == '__main__':
    pass


