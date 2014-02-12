'''
Created on Mar 2, 2012

@author: Stephen O'Hara

Latent Configuration Clustering
Algorithm:
Compute Proximity Forest over a data set that has an unknown configuration
Compute graph from connectivity of leaf nodes in forest
Cluster using the weighted graph, hierarchical or spectral clustering

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
import pylab as pl
import networkx as nx
try:
    from IPython.parallel import require
except:
    print "Unable to import IPython parallel library. Parallel implementations will be unavailable."
    
import pyvision as pv
import cv
import scipy
import scipy.cluster.hierarchy as spc
try:
    import sklearn
except:
    print "Warning: Could not import scikits.learn (sklearn) package. Swiss Roll data set and related samples won't work."

rint=scipy.random.randint

def genRandom2DPoints(N=3000, size=(640,480)):
    xs = rint(low=0, high=size[0]+1, size=N)
    ys = rint(low=0, high=size[1]+1, size=N)
    return zip(xs,ys)
    
def genNormalClusters(N=100, size=(640,480)):
    group1 = zip( 96*scipy.randn(N)+size[0]/4, 96*scipy.randn(N)+size[1]/4)
    group2 = zip( 96*scipy.randn(N)+3*size[0]/4, 96*scipy.randn(N)+size[1]/4)
    group3 = zip( 96*scipy.randn(N)+3*size[0]/4, 96*scipy.randn(N)+3*size[1]/4)
    group4 = zip( 96*scipy.randn(N)+size[0]/4, 96*scipy.randn(N)+3*size[1]/4)
    samples = group1+group2+group3+group4
    labels = [0]*N + [1]*N + [2]*N + [3]*N 
    return (samples, labels)
    
def plotPoints(pts, size=(640,480), colors=None, window="Points"):
    X = scipy.zeros(size)
    img = pv.Image(X)  #convert zeros matrix to an all-black image
    points = [ pv.Point(x, y) for (x,y) in pts]
    
    if colors is None:
        colors = ['red' for p in pts]
    
    for (p,c) in zip(points,colors):
        img.annotatePoint(p, color=c)
    
    if not window is None:
        img.show(window=window, delay=0)
    
    return img

def plotClusts(pts, labels, size=(640,480), colors=None, window="Points"):
    cv_img = cv.CreateImage(size, 8, 1)
    cv.Set(cv_img, 225.0) #light gray background
    img = pv.Image(cv_img)  #convert to a pyvision image
    points = [ pv.Point(x, y) for (x,y) in pts]
    
    if colors is None:
        colors = ['red' for p in pts]
    
    for (p,l,c) in zip(points,labels,colors):
        #img.annotateLabel(p, str(l), color=c)
        #img.annotatePoint(p, color=c)
        img.annotateCircle(p,radius=5,color=c, fill=c)
    
    if not window is None: img.show(window=window, delay=0)
    return img

def plotProximityTreeLeaves(ptree, size=(640,480)):
    X = scipy.zeros(size) #clear image data each time, because points can change leaf node membership as tree grows
    img = pv.Image(X)  #convert zeros matrix to an all-black image
    nodes = ptree.getLeafNodes()
    colordict={}
    for node_id,n in enumerate(nodes):
        if node_id in colordict:
            hc = colordict[node_id]
        else:
            colr = (rint(80,256), rint(80,256), rint(80,256))
            hc = RGBToHTMLColor(colr)
            colordict[node_id] = hc
        
        pts = [pt for (pt,_) in n.items]
        poly = [pv.Point(*p) for p in convexHull(pts)]
        img.annotatePolygon(poly, color=hc, width=1)
    img.show(window="Proximity Tree Construction", delay=0)
    
#@require(scipy)  #NOTE: If using an IPython parallel proximity forest, you may need to uncomment this @requires statement.
def pt_dist(pt1,pt2):
    u = scipy.array(pt1)
    v = scipy.array(pt2)
    return scipy.sqrt(scipy.sum((u-v)**2))

def RGBToHTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple
    return hexcolor

def convexHull(pts):
    storage = cv.CreateMemStorage(0)
    hull = cv.ConvexHull2(pts, storage, cv.CV_CLOCKWISE, 1)
    polygon = []
    for i in hull:
        polygon.append(i)
    return polygon

def demoSwiss(k=6, parallel_client=None):
    '''
    Demonstrate the performance of LCC
    on the swiss roll data set.
    Some of the code is from the scikits.learn example for applying
    ward's clustering to the swiss roll data, but appropriately modified
    to use LCC instead.
    
    Original authors of the non-LCC version:
    # Authors : Vincent Michel, 2010
    #           Alexandre Gramfort, 2010
    #           Gael Varoquaux, 2010
    # License: BSD
    '''
    import numpy as np
    import pylab as pl
    import mpl_toolkits.mplot3d.axes3d as p3
    from sklearn.datasets.samples_generator import make_swiss_roll
    
    # Generate data (swiss roll dataset)
    n_samples = 1000
    noise = 0.05
    X, _ = make_swiss_roll(n_samples, noise)
    # Make it thinner
    X[:, 1] *= .5

    #Convert data matrix X to a list of samples
    N = X.shape[0]
    dat = [X[i,:] for i in range(N)]
    
    #generate LCC clustering
    print "Generating LCC Clustering"
    (label, _, _, _) = pf.LatentConfigurationClustering(dat, pt_dist, k, numtrees=27, parallel_client=parallel_client)
    
    # Plot result
    fig = pl.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(label):
        ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                  'o', color=pl.cm.jet(np.float(l) / np.max(label + 1)))
    pl.title('Latent Configuration Clustering')
    
    pl.show()

def demoFourGs():
    '''
    Demonstrate the performance of LCC
    on points drawn from a four gaussians
    '''           
    s=(640,480)
    dat = genNormalClusters(N=100, size=s)
    cList = ['red', 'blue','green','yellow']
    img_truth = plotClusts(dat[0], dat[1], size=s, 
                           colors=[cList[i] for i in dat[1]], window=None)
    
    #generate normal hierarchical clustering off euclidean data points
    print "Generating Hierarchical Clustering on Raw Data"
    Z2 = spc.ward(scipy.array(dat[0]))
    clusts2 = spc.fcluster(Z2, 4, criterion="maxclust")
    img_HC = plotClusts(dat[0], clusts2, size=s, 
                           colors=[cList[i-1] for i in clusts2], window=None)
    
    #generate LCC clustering
    print "Generating LCC Clustering"
    (clusts, _,_,_) = pf.LatentConfigurationClustering(dat[0], pt_dist, 4, numtrees=27)
    img_LCC = plotClusts(dat[0], clusts, size=s, 
                           colors=[cList[i-1] for i in clusts], window=None)
    
    im = pv.ImageMontage([img_truth, img_LCC, img_HC], layout=(1,3), gutter=3,
                          tileSize=(320,240), labels=None )
    im.show(window="Truth vs. LCC vs. HC")
    
def position_dictionary(pts):
    return {i:pts[i] for i in range(len(pts))}

def demoCluster500(K=10, edgeThresh=9):
    '''
    Clusters 500 2d points into K clusters. Shows the clusters
    using colors and overlays the connectivity graph for all edges
    with weights >= edgeThresh
    '''
    pts = genRandom2DPoints(500)
    pos_dict = position_dictionary(pts)
    (clusts, _, pforest, g) = pf.LatentConfigurationClustering(pts, pt_dist, K)
    g2 = pf.filter_edges(g, thresh=edgeThresh)
    colors = [ float(c+1) for c in clusts]
    nx.draw(g2, pos=pos_dict, with_labels=False,node_size=35, node_color=colors)
    pl.show()
    return (clusts, pforest, g)
    
    
def demoLCC_stages():
    '''
    Plots three diagrams showing input points, connectivity graph, and output labels
    '''    
    pts, labels = genNormalClusters(N=250, size=(1200,800))
    pos_dict = position_dictionary(pts)
    (clusts, _, pforest, g) = pf.LatentConfigurationClustering(pts, pt_dist, 4)
    g2 = pf.filter_edges(g, thresh=9)
    colors = [ float(c+1) for c in clusts]
    
    pl.figure(1, (18,6))
    
    #figure 1, input points
    yl = (-200,1000)
    pl.subplot(1,3,1)
    nx.draw_networkx_nodes(g2, pos=pos_dict, with_labels=False, node_size=35)
    pl.ylim(yl)
    
    #figure 2, connections above a threshold
    pl.subplot(1,3,2)
    nx.draw_networkx_edges(g2, pos=pos_dict, with_labels=False)
    pl.ylim(yl)
    
    #figure 3, coloring of nodes based on cluster labels
    pl.subplot(1,3,3)
    nx.draw_networkx(g2, pos=pos_dict, with_labels=False, node_size=35, node_color=colors, cmap="jet")
    pl.ylim(yl)
    
    pl.subplots_adjust(left=0.05, right=0.95)
    pl.show()
    
    return (pts, labels, clusts, pforest, g)

if __name__ == '__main__':
    pass