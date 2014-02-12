'''
Created on Apr 20, 2012
@author: Stephen O'Hara

Common classes functions relating to the analysis of experimental results,
which I use across multiple data sets. Plotting code requires matplotlib.

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

import numpy as np
import math
import cPickle
import sys

def confusionMatrix(cnfList, class_sizes, one_based=False):
    ''' Computes the confusion matrix given a list of confusers
    from an experimental result. Takes into account potential uneven
    class distribution.
    @param cnfList: A list of tuples indicating ONLY the
    errors, or confusions, present in a classification result.
    The tuples are ordered (predicted, actual).
    @param class_sizes: A list of the size of each class. Length
    of list indicates total number of classes.
    @param one_based: Class labels are assumed to be contiguous zero-based
    integers. However, if your labels are instead 1-based, then set
    this parameter to True. In which case, labels are assumed to be
    contiguous starting from 1..N, instead of 0..N-1, where N is the
    number of classes.
    '''
    numClasses = len(class_sizes)
    classes = range(1,numClasses+1) if one_based else range(numClasses)
    
    cfMatrix = np.zeros( (numClasses,numClasses) )
    for classIdx in classes:
        bads = [p for (p,t) in cnfList if t == classIdx]
        #record the diag entry for this row
        cf_idx = (classIdx-1) if one_based else classIdx
        
        cfMatrix[cf_idx,cf_idx] = class_sizes[cf_idx] - len(bads)
        #record the off-diagonal misses
        for b in bads:
            cfMatrix[cf_idx, b] += 1
        #normalize row to a percentage
        for col in range(numClasses):
            cfMatrix[cf_idx,col] = cfMatrix[cf_idx,col] / class_sizes[cf_idx]
     
    tmp = (cfMatrix * 10000).astype(int)
    cfMatrix = (tmp*1.0/100)
    
    print np.round_(cfMatrix, decimals=1)
    
    return cfMatrix

def latexMatrix(cfMx, rowlabels=None):
    '''
    if you have the asciitable package installed, avail through mac ports
    and other distributions, then we can output latex format tables for
    our data. This is handy for generating latex output of a confusion matrix.
    '''
    try:
        import asciitable
        import asciitable.latex
    except:
        print "Error importing asciitable...you may not have this package installed."
        return None
    
    #build data dictionary, per-column structure
    if rowlabels != None:
        data = {'col000':rowlabels}
        col_align = ['l']
    else:
        data = {}
        col_align = []
    numCols = cfMx.shape[1]
    for i in range(numCols):
        data['col%s'%(str(i+1).zfill(3))] = cfMx[:,i]
        col_align.append('c')
        
    col_align_str = "|%s|"%("|".join(col_align))
    asciitable.write(data, sys.stdout, Writer=asciitable.Latex, col_align=col_align_str)
    
class ExperimentalErrorResults:
    '''
    This class is designed to store results of running repeated trials over a set of parameter values,
    where the trials measure the error in some task, typically a classification task.
    
    The results are stored as a numpy 2D array, where rows represent the results for a single parameter
    setting and columns within a row are the results of repeated trials for the same setting.
    
    This class provides convenience methods for analyzing the results and can be used with some
    high-level matplotlib-based plotting code for visualization. Also provides convenience methods
    for saving/loading results to disk.
    '''
    def __init__(self, ResMatrix, paramList, paramName="Parameter", desc=None, props={}):
        '''
        ResMatrix has rows corresponding to different forest sizes and columns corresponding
        to repeated trials with the same size tree.
        @param ResMatrix: The results matrix. One row per entry in paramList, one column for each
        trial for each param setting. The entries in the matrix represent error rates, as a float. So 0.025 indicates
        that a given trial of a given parameter setting reported a 2.5% error rate (97.5% correct).
        @param paramList: A list of numerical parameter settings. Order in list corresponds to row-order of results matrix.
        @param paramName: A string representation of the parameter name, used in data output/charts
        @param desc: A string describing other salient information about this result object.
        @param props: A dictionary with any additional key/value information you wish to store with this object
        '''
        self.R = ResMatrix
        self.params = paramList
        self.paramName = paramName
        self.desc = desc
        self.props = props        
    
    def __str__(self):
        return "Experimental Results: %s"%self.desc
    
    def save(self, filename):
        cPickle.dump(self, open(filename,"wb"), protocol=-1)
    
    @staticmethod
    def load(filename):
        '''
        Static method that creates a new instance of this class from a saved pickle file
        '''
        return cPickle.load(open(filename,"rb"))
    
    def setDescription(self, desc):
        self.desc = desc
    
    def compute95ci(self):
        '''
        Computes the 95% confidence interval around the mean values for each row
        '''
        numtrials = len(self.R[0,:])
        x = self.getStdvs() * 1.96 / math.sqrt(numtrials)
        meanErrs = np.mean(self.R,1)
        low = meanErrs - x
        high = meanErrs + x
        return(low,high)
    
    def getStdvs(self):
        stdvs = np.std(self.R,1) #st.devs. for each row
        return stdvs
        
    def getMeanAccuracy(self):
        means = np.mean(self.R,1) #mean value for each row (trials of a single forest size)
        return 1 - means
    
    def getMeanAccuracy_ci(self):
        '''@return the 95% confidence interval about each mean accuracy score'''
        (low,high) = self.compute95ci()
        return (1-high, 1-low)
