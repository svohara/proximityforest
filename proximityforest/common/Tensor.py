'''
Created on Oct 14, 2010
@author: Stephen O'Hara

This code module provides a set of functions to be
applied to 3-mode tensors, such as unfolding along
an axis into a matrix, computing canonical angles,
chordal distance, and so on.

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
import scipy.linalg as LA

def UnfoldCube(X, k=1):
    ''' Unfold a 3rd order tensor along mode 1, 2 or 3.
    @param X: data cube to be unfolded, 3rd order tensor, (depth, row, col) in python 
    @param k: which order to unfold on. For compatibility with MATLAB 1=row, 2=col, 3=depth
    @return: 2D array of appropriate dimension
        - k=1 will be of size row x (col*depth),
        - k=2 will be of size col x(row*depth)
        - k=3 will be of size depth x (row*col)
    @raise IndexError: if k is not 1,2,or 3 
    '''
    #depth, row, col = X.shape
    if k==1:
        perm = (0,2,1)
    elif k==2:
        perm = (0,1,2)
    elif k==3:
        perm = (2,0,1)
    else:
        raise IndexError
        print "Error: Unfold cube k must be 1,2, or 3."
        
    Y = X.transpose(perm)
    Y = sp.hstack(Y[:,])
    return(Y)
    
def test_UnfoldCube():
    #generating test tensor
    A = sp.ones( (2,4,3)) #create a depth=2, row=4, col=3 tensor
    count = 1
    for k in range(2):
        for i in range(4):
            for j in range(3):
                A[k,i,j] = count
                count += 1
    
    print 'Testing Tensor has 2 slices of a 4x3 matrix'
    print A
    print 'Unfold mode 1 = 4x(3x2) = 4x6 matrix'
    print UnfoldCube(A,1)
    print 'Unfold mode 2 = 3x(4x2) = 3x8 matrix'
    print UnfoldCube(A,2)
    print 'Unfold mode 3 = 2x(4x3) = 2x12 matrix'
    print UnfoldCube(A,3)
    
def canonicalAngles(A, B):
    ''' Computes the canonical angles between the subspaces defined by
    the column spaces of matrix A and B.
    @param A: A 2D array (matrix) with rows > cols.
    @param B: A 2D array (matrix) with rows > cols.
    @return: The 1D array of canonical angles (Theta) between the subspaces defined by A and B.
    '''
    (r,c) = A.shape
    assert( r > c)
    
    (r,c) = B.shape
    assert( r > c)
    
    #get orthonormal bases
    #NOTE: in scipy.linalg, using the thin svd to get the orthonormal bases is MUCH FASTER
    # than using either the LA.orth(A) function or "economy" mode of QR decomposition!
    (Qa,_,_) = LA.svd(A, full_matrices=False)
    (Qb,_,_) = LA.svd(B, full_matrices=False)
    X = sp.dot(Qa.T,Qb)
    S = LA.svdvals( X )  #singular vals of Qa'*Qb
    #S = cos(Theta)
    Theta = sp.arccos(S)
    return Theta

def canonicalAngles_Tensor(TA, TB):
    ''' Computes the set of canonical angle vectors between two
    3-mode tensors (data cubes). Each cube is unfolded in 3 ways,
    and the canonical angles from each unfolding are computed.
    @param TA: A 3D data cube. (like x-y-time video clip)
    @param TB: Another 3D data cube.
    @return: ThetaList is a list of the three Theta vectors along the three unfoldings.
    '''
    ThetaList = []  
    for axis in [1,2,3]:
        #unfold on each axis        
        A = UnfoldCube(TA, k=axis)
        B = UnfoldCube(TB, k=axis)
        Theta = canonicalAngles(A.T,B.T)
        ThetaList.append(Theta)
    return ThetaList
    
def chordalDistance( Theta ):
    ''' Computes the chordal distance of a canonical angle vector.
    The L2 norm of the sine of the angles.
    D = ||Sin(Theta)||_2
    '''
    return LA.norm( sp.sin(Theta)  )

def productManifoldDistance(TA, TB):
    '''
    Computes the product manifold distance between two data cubes.
    Implemented from the description from the following paper:
    Y.M. Lui, et al., "Action Classification on Product Manifolds", CVPR 2010.
    If you use this function, please cite Yui Man Lui's paper!
    @param TA: The first tracklet (3D Tensor)
    @param TB: The second tracklet (3D Tensor)
    @return: a real number indicating the distance
    '''
    theta = canonicalAngles_Tensor(TA, TB) #list of 3 vectors
    theta = sp.hstack(theta)  #single vector
    dist = chordalDistance(theta)
    return dist

def flipTensor(T):
    ''' It may be advantageous to flip the horizontal aspect of a video cube so that one can add "both directions"
    of an action to a subspace tree, so if theres a video of a man punching leftwards, when you flip it, you also
    get a pseudo-new sample of the same man punching rightwards...'''
    T2 = T[:,::-1,:] #numpy indexing magic...flips the 2nd array dimension LR, which is what we want.
    return T2

     

