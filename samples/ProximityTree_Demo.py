'''
Created on Mar 1, 2012
@author: Stephen O'Hara

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
import shapely.geometry as geom
import math

try:
    from IPython.parallel import require
except:
    print "Unable to import IPython parallel library. Parallel implementations will be unavailable."
    
import pyvision as pv
#import cv
import scipy
rint=scipy.random.randint

def genRandom2DPoints(N=3000, size=(640,480)):
    xs = rint(low=0, high=size[0]+1, size=N)
    ys = rint(low=0, high=size[1]+1, size=N)
    return zip(xs,ys)


def demoAllInOne(N=100, Tau=15):
    pts = genRandom2DPoints(N=N)
    #imgPts = plotPoints(pts)
    #res = drawProximityTreeLeafContours(N=500, presetPoints=pts, curImg=imgPts, color="white")
    (_,img,_) = drawProximityTreeRegions(N=N, presetPoints=pts, showImg=False, Tau=Tau)
    (_,img2,_) = drawProximityForest(Tau=Tau, presetGallery=pts, curImg=img, showPts=False, showImg=False)
    img2.show(window="Nearest Neighbors", delay=25)
    return img2
    
def plotPoints(pts, size=(640,480), colors=None, curImg=None):
    if curImg is None:
        X = scipy.zeros(size)
        img = pv.Image(X)  #convert zeros matrix to an all-black image
    else:
        img = curImg
        
    points = [ pv.Point(x, y) for (x,y) in pts]
    
    if colors is None:
        colors = ['red' for p in pts]
    
    for (p,c) in zip(points,colors):
        img.annotatePoint(p, color=c)
    
    img.show(window="Points", delay=25)
    return img

@require('scipy')
def pt_dist(pt1,pt2):
    u = scipy.array(pt1)
    v = scipy.array(pt2)
    return scipy.sqrt(scipy.sum((u-v)**2))

def RGBToHTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple
    return hexcolor

def drawProximityForest(parallel_client=None, NumPts=(500,20), K=3, FSize=27, Tau=20, size=(640,480),
                        presetGallery=None, curImg=None, showPts=False, showImg=True):
    '''
    @param parallel_client: Provide the ipython parallel client object if you want to use a parallel proximity forest
    implementation. If you don't have ipython parallel, then specify None, and a serial implementation will be used.
    @param NumPts: A tuple (number of gallery points, number of test points)
    @param K: How many k-nearest-neighbors?
    @param FSize: The proximity forest size (number of trees in forest)
    @param Tau: The node splitting threshold used in Prox. Trees
    @param size: The size of the area from which to sample (and display) points. In pixels, as per an image resolution.
    @param presetGallery: If not None, then this indicates the gallery points to use instead of random generation.
    @param curImg: If specified, this will be the background image over which neighbor connections are drawn
    @param showPts: If true, then the output image will have red circles for each of the gallery points.
    '''
    if presetGallery is None:
        gallery = list(set(genRandom2DPoints(N=NumPts[0], size=size))) #no duplicates allowed
    else:
        gallery = presetGallery
        
    ids = range(len(gallery))
    probes = list(set(genRandom2DPoints(N=NumPts[1], size=size)))
            
    
    if not parallel_client is None:
        print "Using a parallel proximity forest"
        forest = pf.ParallelProximityForest(parallel_client, FSize, Tau=Tau, dist_func=pt_dist)
    else:
        print "Using a serial proximity forest"
        forest = pf.ProximityForest(FSize, Tau=Tau, dist_func=pt_dist)
        
    print "Building forest from gallery points..."
    forest.addList(gallery, ids)
    
    if curImg is None:
        X = scipy.zeros(size) #clear image data each time, because points can change leaf node membership as tree grows
        img = pv.Image(X)  #convert zeros matrix to an all-black image
    else:
        img = curImg
    
    print "Testing KNN of probes..."
    if showPts:
        for p in gallery: img.annotatePoint(pv.Point(*p), color="red")  #background image is just the point cloud
    
    for i,p in enumerate(probes):
        #c = RGBToHTMLColor((rint(80,256), rint(80,256), rint(80,256)))
        c = "yellow"
                
        #draw a line between this point and its KNN
        knn = forest.getKNearest(p, K) #list like [(d1,pt1,id1), (d2,pt2,id2)...k]
        if len(knn) < 1:
            print "ERROR! Point %s has no nearest neighbors!"%str(p)
            continue
        for (_,p2,_) in knn:
            img.annotateLine(pv.Point(*p), pv.Point(*p2), color=c, width=1)
            img.annotateCircle(pv.Point(*p2), color=c, fill=c)
          
        #draw label last so it isn't overwritten  
        img.annotateLabel(pv.Point(*p), "%d"%(i+1), mark="right", color=c, background="blue")
        #img.annotateCircle(pv.Point(*p), color=c, fill=c)
                              
        if showImg: img.show(window="Proximity Forest", delay=25)
    
    return forest, img, gallery+probes
    

def computePartitions(tree, parent_region=None, restricted=None, universe=(640,480)):
    '''
    construct the polygonal containers at each level in the tree
    using the shapely geometry package. This is a recursive function
    and should be called at the top-level root node in the tree
    '''
    if parent_region is None:
        ctr = (universe[0]/2.0, universe[1]/2.0)
        r = math.sqrt( ctr[0]**2 + ctr[1]**2)
        parent_region = geom.Point(*ctr).buffer(r)
        
    containers = {}
    
    if not tree.Pivot is None:
        region = geom.Point(*tree.Pivot).buffer(tree.SplitD)
        region = region.intersection(parent_region)
        if not restricted is None:
            try:
                region=region.difference(restricted)
            except geom.TopologicalError:
                #sometimes produces a null region
                region = geom.Point(*tree.Pivot).buffer(0.0)
            
        containers[tree.ID] = (region, pv.Point(*tree.Pivot))
        
        left = computePartitions(tree.Children[0], parent_region=region,
                          restricted=restricted, universe=universe)
        
        if restricted is None:
            restricted = region
        else:
            restricted = restricted.union(region)
        right= computePartitions(tree.Children[1], parent_region=parent_region,
                          restricted=restricted, universe=universe)
        
        containers.update(left)
        containers.update(right)
    
    return containers
  
def getCircle(node):
    '''
    Returns the center point (pivot) and radius (splitD)
    of the circle that partitions a node in a proximity tree,
    as well as the parent's pivot and radius.
    '''
    if node.Pivot is None: return (None, None, None, None)
    pt = pv.Point(*node.Pivot)
    radius = node.SplitD
    
    if not (node.parent is None):
        ppt = pv.Point(*node.parent.Pivot)
        pr = node.parent.SplitD
    else:
        ppt = None
        pr = None
        
    return (pt,radius, ppt, pr)
    
def drawRegions(tree, curImg=None, color=None, size=(640,480), colordict = {}):   
    '''
    Draw circles and arcs on the curImg or a new image representing the
    partitioning of the proximity tree
    ''' 
    if curImg is None:
        X = scipy.zeros(size) #clear image data each time, because points can change leaf node membership as tree grows
        img = pv.Image(X)  #convert zeros matrix to an all-black image
    else:
        img = curImg
        
    containers = computePartitions(tree, universe=size)   
    
    #draw the circles for each splitting node    
    for node_id in containers:
        (region, _) = containers[node_id] 
        
        if type(region) == geom.Polygon:
            region = [region]
        elif type(region) == geom.MultiPolygon:
            pass
        else:
            raise ValueError("Don't know what to do with region of type: %s")%str(type(region))
        
        pv_polys = []
        for poly in region:
            pv_polys.append( [ pv.Point(x,y) for (x,y) in poly.exterior.coords] )
        
        if color is None:
            if node_id in colordict:
                hc = colordict[node_id]
            else:
                colr = (rint(80,256), rint(80,256), rint(80,256))
                hc = RGBToHTMLColor(colr)
                colordict[node_id] = hc
        else:
            hc=color
          
        print "Node: %s"%node_id    
        if node_id == "root":
            w = 6
        else:
            s = node_id[4:]
            w = 6-len(s) if len(s) <= 5 else 1 
            
        for p in pv_polys: img.annotatePolygon(p, color=hc, width=w)    
        
    #draw the points at the leaf nodes in the same color as 
    # their containing splitting node 
    lnodes = tree.getLeafNodes()
    if len(lnodes) > 1:
        for n in lnodes:
            parent = n.parent
            c = colordict[parent.ID] 
            for pt in n.items:
                    img.annotatePoint(pv.Point(*pt), color=c) 
    else:
        for pt in lnodes[0].items:
            img.annotatePoint(pv.Point(*pt), color='red')    
        
    #draw the text labels on top of everything else
    for node_id in containers:
        (_,pvt) = containers[node_id]
        hc = colordict[node_id]
        img.annotateCircle(pvt, 3, color=hc, fill=hc)  #draw the pivot
        img.annotateLabel(pvt, node_id, color="black", background=hc)
        
    return img, colordict
    
def drawProximityTreeRegions(N=500, Tau=20, size=(640,480), presetPoints=None, curImg=None,
                              color=None, showImg=True):
    tree = pf.ProximityTree(dist_func=pt_dist, Tau=Tau)
    
    if presetPoints is None:
        pts = list(set(genRandom2DPoints(N=N, size=size))) #no duplicates allowed
        ids = range(N)
    else:
        pts = presetPoints
        ids = range(len(presetPoints))
    
    tree.addList(pts, ids)
    (img, _) = drawRegions(tree, curImg, color, size)
    
    if showImg: img.show("Proximity Tree Leaf Nodes", delay=25)
    return tree, img, pts
    
def animateProximityTree(N=50, Tau=10, size=(640,480), presetPoints=None,
                          delay=1000, window="Proximity Tree Construction",
                          video_out = None):
    tree = pf.ProximityTree(dist_func=pt_dist, Tau=Tau)
    
    if presetPoints is None:
        pts = list(set(genRandom2DPoints(N=N, size=size))) #no duplicates allowed
    else:
        pts = presetPoints
    ids = range(N)
    colordict = {}  #color assigned per node
    #pt_list = []  #list of points that have been added so far
    if not video_out is None:
        vw = pv.VideoWriterVSP(video_out, window=None, size=(640,480) )
    else:
        vw = None
    
    for (pt,idx) in zip(pts,ids):
        #img = pv.Image(scipy.zeros(size)) #clear background each time
        
        print "Adding point %d at position %s."%(idx, str(pt))
        #pt_list.append(pv.Point(*pt))
        #img.annotatePoints(pt_list, color='red')
        tree.add(pt,idx)
        
        #(img, colordict) = drawRegions(tree, curImg=img, colordict=colordict)
        (img, colordict) = drawRegions(tree, curImg=None, colordict=colordict)
        img.show(window=window, delay=delay)
        if not vw is None:
            vw.addFrame(img)
    
    return tree, img, pts
   
   
def pForest_vs_flann_20Trials(numTrees=10):
    print "Comparing FLANN to Proximity Forest on 500 Random 2D Points"
    flann_scores=[]
    pf_scores=[]
    discrepancies=[]
    for i in range(20):
        print "=============================================="
        print "TRIAL: %d"%(i+1)
        print "=============================================="
        (nd, sum_flann, sum_pf) = pForest_vs_flann(numTrees=numTrees, verbose=False)
        flann_scores.append(sum_flann)
        pf_scores.append(sum_pf)
        discrepancies.append(nd)
        print "=============================================="
        print "Discrepancies: %d, Cost per Discrepancy: %3.2f"%(nd,(sum_flann - sum_pf)*1.0/nd)
        print "=============================================="
        
    print "=============================================="
    print "20 TRIAL SUMMARY"
    print "Average Discrepancies: %3.2f"%( 1.0*sum(discrepancies)/len(discrepancies))
    flann_scores = scipy.array(flann_scores)
    pf_scores = scipy.array(pf_scores)
    avg_delta_score = (sum(flann_scores) - sum(pf_scores))*1.0/len(discrepancies)
    print "Average Cost Per Discrepancy: %3.2f"%avg_delta_score
    print "Average FLANN Distance: %3.2f, StdDev: %3.2f"%(scipy.mean(flann_scores),scipy.std(flann_scores))
    print "Average Proximity Forest Distance: %3.2f, StdDev: %3.2f"%(scipy.mean(pf_scores),scipy.std(pf_scores))
    print "=============================================="
    return (discrepancies, flann_scores, pf_scores)
        
def pForest_vs_flann(numTrees=10, parallel_client=None, verbose=True):
    pts = genRandom2DPoints(N=500)
    
    #forest of 10 randomized kdtrees
    #Note that this is calling optimized C-code, so it's very fast.
    flann = pv.FLANNTree(scipy.array(pts)*1.0, trees=numTrees)
    
    #proximity forest of 10 trees
    #Native python code, not-optimized, so slower
    if not parallel_client is None:
        pForest = pf.ParallelProximityForest(parallel_client,numTrees, Tau=20, dist_func=pt_dist)
    else:
        pForest = pf.ProximityForest(N=numTrees, Tau=20, dist_func=pt_dist)
    pForest.addList(pts, range(len(pts)))

    discrepancies = 0
    sum_flann_ds = 0
    sum_pf_ds = 0
    #delta_score=0
    queryPts = genRandom2DPoints(N=500)
    for q in queryPts:
        qf = scipy.array(q)*1.0
        (ds,idxs) = flann.query(qf, k=3)
        sum_flann_ds += sum(ds)
        
        res = pForest.getKNearest(q, K=3)
        idxs2 = [idx for (_,_,idx) in res]
        ds2 = [d for (d,_,_) in res]
        sum_pf_ds += sum(ds2)
        
        if not scipy.all(idxs==idxs2):
            discrepancies += 1
            #delta_score += sum(ds-ds2)
            if verbose:
                print "KNN Disagreement at query point %s."%str(q)
                print "Flann: %s"%str(ds)
                print "Proximity Forest: %s"%str(ds2)            
                print "Discrepancy for point = %3.2f. Total score=%3.2f"%(sum(ds-ds2), (sum_flann_ds-sum_pf_ds))
            
    print "Total number of discrepancies: %d"%discrepancies
    delta_score = (sum_flann_ds - sum_pf_ds)*1.0
    print "Average difference per discrepancy: %3.2f"%(delta_score/discrepancies)
    print "(Positive Number means pForest is finding closer neighbors)"
    return (discrepancies, sum_flann_ds, sum_pf_ds)

        
        
    
if __name__ == '__main__':
    pass



