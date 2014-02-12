'''
Created on Nov 4, 2011
@author: Steve O'Hara
Module to generate tracks on certain UCF sports
videos that don't have annotations.
'''
import pyvision as pv
import neofilter as nf
import glob 
import os

base_dir = "/Users/Steve/Data/ucf_sports_actions/ucf_action"

vidlist = [ "Walk-Front/005", "Walk-Front/019", "Walk-Front/020",
           "Walk-Front/021", "Lifting/001", "Lifting/002", "Lifting/003",
           "Lifting/004", "Lifting/005" "Lifting/006"]

rectlist = [ pv.Rect(390,0,240,330), pv.Rect(300,90,245,315), 
            pv.Rect(190,50,190,385), pv.Rect(204,172,120,228),
            pv.Rect(190,25,410,370), pv.Rect(190,25,410,370),
            pv.Rect(190,25,410,370), pv.Rect(190,25,410,370),
            pv.Rect(190,25,410,370), pv.Rect(190,25,410,370) ]

#Some rectangles should be tracked, some should be constant for all frames in video
# I guess we track Walking but use constant rects for Lifting
trackit = [True, True, True, True, False, False, False, False, False, False]

def saveBounds(rect, filename):   
    rt = list( rect.asTuple() )
    r = tuple( map(int, rt) )    
    with open(filename,'w') as f:
        f.write("%d\t%d\t%d\t%d"%r)
    f.closed

if __name__ == '__main__':
    pass

IDX=4

viddir = os.path.join(base_dir, vidlist[IDX])
rect = rectlist[IDX]
outdir = os.path.join(viddir,"gt")

jpgList = sorted(glob.glob( viddir+"/*.jpg"))
vid = pv.VideoFromFileList(jpgList)

tracker = None
for i,frame in enumerate(vid):
    print "Processing Frame %d..."%i
    
    if trackit[IDX]:        
        if tracker == None:
            tracker = nf.MOSSETrack(frame,rect)
        else:
            tracker.update(frame)
        r = tracker.asRect()
        bounds = pv.Rect( **r.asInt() )
        tracker.annotateFrame(frame)
    else:
        bounds = rect
        frame.annotateRect(bounds)
        
    print bounds
    fn = os.path.splitext( os.path.basename( jpgList[i] ) )[0] + ".tif.txt"
    fn = os.path.join(outdir,fn)
    print fn
    saveBounds(bounds,fn)
    frame.show(window="UCF Sports", delay=10)

    
