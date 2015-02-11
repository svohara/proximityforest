'''
Created on Oct 14, 2010
@author: Stephen O'Hara

This code module provides a set of handy functions
used throughout the library.

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

import scipy as sp
import pyvision as pv
import PIL.Image as PIL
import math
from scipy.stats.distributions import entropy
#TODO: Check col/row mapping to x/y when using pv.Image().asMatrix2D()...pyvision
# may do a transpose to make indexing more 'convenient.'
def crop_black_borders(img, tolerance=20):
    '''
    Sometimes images have black borders around them, either due to letterboxing a 
    video or for other reasons. This function will look along all four sides of
    the image to remove rows/cols that are black.
    @param img: A pyvision image
    @param tolerance: How black do we have to be? Between the intensities of 0=black, 255=white,
    what is the threshold for considering a pixel to be black...default=10.
    @return: A pyvision image that is a cropped copy of the input image.
    '''
    #convert img to gray scale matrix, for border detection
    img_mat = img.asMatrix2D() #numpy 2D array
    
    #look along top, bottom, left, and right for rows/cols that are essentially all black
    newX = 0
    while True:
        if sp.all( img_mat[newX,:] < tolerance):
            newX += 1
        else:
            break

    newY = 0
    while True:
        if sp.all( img_mat[:,newY] < tolerance):
            newY += 1
        else:
            break
        
    newX2 = img_mat.shape[0]-1
    while True:
        if sp.all( img_mat[newX2,:] < tolerance):
            newX2 -= 1
        else:
            break
    
    newY2 = img_mat.shape[1]-1
    while True:
        if sp.all( img_mat[:,newY2] < tolerance):
            newY2 -= 1
        else:
            break    
        
    newW = newX2 - newX
    newH = newY2 - newY
    #print "DEBUG: newX, newY, newW, newH = ",(newX,newY,newW,newH)    
    #once we know the new starting row,col and ending row,col, we
    # can crop the original, full-color image.
    roi = pv.Rect( newX+1, newY+1, newW-1, newH-1)
    cropped = img.crop(roi)
    return cropped

def makeTensorFromImages(imgStack, rects=None, cube=(48,32,32)):
    '''
    Given a list of 'stacked' images, most likely representing a tracked
    image region over time, this function will create a 3D array (aka "Tensor" or "Data Cube")
    with dimensions (numframes, thumbnail width, thumbnail height). Each image in
    imgStack will first be cropped to the corresponding rectangle provided in rects (if any),
    and then be resized using antialiasing and bilinear interpolation
    to the desired thumbnail-size. If there are more images in the imgStack than
    number of frames in cube size, then the middle will be chosen. If there are fewer images,
    then images will be repeated as necessary to pad the 3D array.
    @param imgStack: A list of pvImage objects that should represent a tracked image region
    in consecutive frames (pre-cropped tiles), OR the source images from which tiles will
    be cropped using corresponding rectangles in the rects parameter.
    @param rects: If None, then imgStack is considered pre-cropped to the region of interest.
    Else, this is a list of pv.Rect objects, same length as imgStack, where the rectangle
    will be used to crop the specified region from the source image (corresponding index).
    @param cube: The dimensions of the output data cube. A tuple of (numframes, width, height)
    @return: A numpy 3D array.
    '''
    tiles = [] #will hold list of cropped (if necessary) and resized tiles
    if not rects is None:
        assert(len(imgStack)==len(rects))
        
    numFrames = len(imgStack)  #number of frames of input data provided
    #print "DEBUG: Number of frames in imgStack: %d"%numFrames
    (F,Xs,Ys) = cube
    if F < numFrames:
        startIdx = int(sp.ceil( (numFrames-F)/2))
        f_range = range( startIdx, startIdx+F)
    else:
        f_range = range(len(imgStack))
                 
    #for the selected range of images we use from imgStack,
    # (optionally crop) and resize each to desired thumbnail width and height               
    for fn in f_range:
        #print fn
        #sys.stdout.flush()
        img = imgStack[fn]
        if not rects is None:
            tmp = img.crop( rects[fn] )
            tile = crop_black_borders(tmp)
        else:
            tile = img
        #tile2 = tile.resize((Xs,Ys))
        tmp = tile.asPIL()
        pilImage = tmp.resize((Xs,Ys), PIL.ANTIALIAS)
        tiles.append(pv.Image(pilImage))
        

    #we might have too few images in the tiles list for the desired cube size,
    # in which case we need to pad the cube of what we have by repeating
    # frames from the beginning...so we use the index modulo len(tiles)
    tensor = sp.zeros( cube )
    for idx in range(F):
        idx_mod = idx%len(tiles)
        tensor[idx,:,:] = tiles[idx_mod].asMatrix2D()
    
    return tensor  


def vote(inputList, numClasses):
    '''
    Returns the element of the inputList that occurs most often...simple majority voting.
    In case of a tie, lowest index wins?
    '''
    h,_ = sp.histogram(inputList, range=[0,numClasses-1], bins=numClasses)
    return sp.argmax(h)


def trackletEntropy(tracklet, hist_bins=16):
    '''
    This function determines the average shannon entropy of pixel jets in
    a tracklet. A pixel jet is the value of a particular pixel location (x,y)
    through all frames of the tracklet. There are (width x height) pixel
    jets, each of length = #frames. A low entropy means that there is little
    pixel intensity variation over time at a given location. If all locations
    have little variation, then the tracklet may be "bad" -- such as when
    a tracked region is mostly still for the tracklet duration.
    @param tracklet: The input tracklet as an ndarray
    @param hist_bins: The number of bins over which to compute the entropy. Pixel
    values will vary from 0-255, so bins <= 256. Some quantization is probably
    good, so a default of 16 is used.
    @return: (H_mu, H_sigma), the mean and standard deviation of the
    pixel jet entropy.
    '''
    (frames, width, height) = tracklet.shape
    
    Ttmp = tracklet.reshape(frames, width*height)
    L = list( Ttmp.T )  #L is a list of jets
    epsilon = 0.000001 #to avoid warnings in entropy calc
    Hs = [ entropy( sp.histogram(jet, bins=hist_bins, range=[0,255])[0] + epsilon) for jet in L ] #one H per jet
    
    #for i in range(Ttmp.shape[1]):
    #    hist,_ = scipy.histogram(Ttmp[:,i], bins=hist_bins, range=[0,255]) #256 pixel values
    #    hist = hist + 0.000001 #an epsilon value to avoid warnings
    #    Hs.append( entropy(hist) )
    
    H_mu = sp.mean(Hs)
    H_sigma = sp.std(Hs)
    return (H_mu, H_sigma)  #return mean and std dev of pixel-jet entropy values

def play_tracklets(tracklets, layout=None, size=None, labels=None, window="Video Set", delay=34, save_as=None):
    '''
    play a montage video of the list of video tensors
    @param tracklets: a list of tracklets representing thumbnail videos. All tracklets must
    be of same size/shape. Dimensions should be ordered as (frame,x,y) for each tensor.
    @param layout: A tuple (rows,cols) of the montage layout
    @param size: A tuple (width,height) that specifies the size that each frame will be displayed
    in the montage
    @param labels: A list of labels in the same order as the list of tracklets that will be
    shown in the video montage. None (default) has no labels
    @param window: A string representing the window title of the montage
    @param delay: The number of milliseconds to pause between frames in the montage. 34 is default.
    @param save_as: If not None, then this is a full path and filename of an output video avi
    that the video montage will be recorded as.
    '''
    dims = tracklets[0].shape
    
    if size==None: size = (dims[1],dims[2])
    if layout==None: layout = (1,len(tracklets))
    if labels==None:
        labels = range(len(tracklets))
    else:
        labels = [ "%s %s"%(str(i).zfill(2),l) for (i,l) in enumerate(labels)]
    
    videoDict = {}
    for (t,l) in zip(tracklets,labels):
        videoDict[l]=pv.VideoFromImageStack(t)
        
    vm = pv.VideoMontage(videoDict, layout=layout, tileSize=size)
    if save_as is None:
        vm.play(window=window, delay=delay, onNewFrame=None)
    else:
        vidsize = (layout[0]*(size[0]+2), layout[1]*(size[1]+2))
        vsp = pv.VideoWriterVSP(save_as, window=None, size=vidsize)
        vm.play(window=window, delay=delay, onNewFrame=vsp)
        
def play_tracklets_set(tracklets_set, size=None, nolabels=True, window_sfx = "Video Set"):
    '''
    play a montage video of the set of video tracklet lists, for example from
    the set of neighbors returned from a forest of 3 trees.
    @param tracklets_set: a list of tracklet lists. All tracklets must
    be of same size/shape. Dimensions should be ordered as (frame,x,y) for each tensor.
    '''
    dims = tracklets_set[0][0].shape  #shape of one video clip tensor
    if size==None: size = (dims[1],dims[2])
    for v in range(dims[0]):        
        for i,tensors in enumerate(tracklets_set):
            layout =  (1,len(tensors))
            imageList = []
            for t in tensors:
                fmat = t[v,:,:].copy()
                imageList.append( pv.Image(fmat))    
            im = pv.ImageMontage(imageList, layout, size, gutter=2, byrow=True, nolabels=nolabels)
            im.show(window="Tree %d %s"%((i+1),window_sfx), delay=34)

def circle_intersection_pts(circle1, circle2):
    '''
    Returns a list of points where two circles intersect. The list
    could be empty (no intersection), one point, two points. If the
    circles are co-incident (the same), there is an infinite number
    of intersections, in which case an error is raised.
    @param circle1: A tuple (x,y,r) defining the center point (x,y)
    and radius, r, of the first circle.
    @param circle2: A tuple as per circle1 for the second circle.
    @return: A list containing 0, 1, or 2 points of intersection,
    as tuples [(x,y)...]. If the circles are coincidental, and
    thus intersect at infinite points, the return is None.
    @note: This implementation follows the notes provided at:
    paulbourke.net/geometry/2circle/
    '''
    (x1,y1,r1) = circle1
    (x2,y2,r2) = circle2
    
    d = math.sqrt( (x2-x1)**2 + (y2-y1)**2 )
    
    if d > r1 + r2:
        #circles do not intersect, too far apart
        return []
    
    if d < abs(r2-r1):
        #one circle is contained in the other, no intersection pts
        return []
    
    if (d==0) and (r1==r2):
        return None
    
    a = (r1**2 - r2**2 + d**2)/(2.0*d)
    h = math.sqrt( r1**2 - a**2)
    
    #coordinates of midpoint between the two circles, "P2" in paul bourke's notes
    x3 = x1 + (a/d)*(x2-x1)
    y3 = y1 + (a/d)*(y2-y1)

    if d == r1 + r2:
        #single point of intersection at (x3,y3)
        return [(x3,y3)]
    
    #else, we have 2 points of intersection
    
    #first point of the pair
    x4a = x3 + (h/d)*(y2-y1)
    y4a = y3 - (h/d)*(x2-x1)
    #second point of the pair
    x4b = x3 - (h/d)*(y2-y1)
    y4b = y3 + (h/d)*(x2-x1)
    
    return [(x4a,y4a),(x4b,y4b)]

def getAngleBetweenPoints( (x1,y1), (x2,y2) ):
    '''
    Computes the angle, in degrees, between two distinct
    points. Assumes typical graphic coordinates with +y pointing down.
    '''
    deltaX = x2 - x1
    deltaY = y1 - y2
    a = math.atan2(deltaX,deltaY) * (180.0/math.pi)
    return a - 90 #rotate for where PIL expects zero degree to be

if __name__ == '__main__':
    pass

