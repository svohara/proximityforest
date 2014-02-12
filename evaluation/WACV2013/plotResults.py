'''
Created on Nov 29, 2012
@author: Stephen O'Hara

Module with plotting code
'''

import pylab as pl
import cPickle
import os
import math

def loadRes_varyingTrees(datadir=".", fn="SIFT_10K_5trials_VaryingTrees.p"):
    '''
    Loads pickle files which store the trial details for the varyingNumTrees() experiment.
    Pickle file stores a single dictionary object with layout:
    {"NumTrees":list_of_forest_sizes, "PF":pf_trials_matrix, "KDT":kdt_trials_matrix}
    The data stored in the pf/kdt_trials_matrix is layed out as follows:
        rows: each row is a trial
        cols: each column is a parameter setting, in this case num trees in forest
    '''
    infile = os.path.join(datadir, fn)
    data_dict = cPickle.load(open(infile,'rb'))
    return data_dict
    
def plotRes_varyingTrees( data_dict, dataset_name, max_correct=3000 , show=True):
    '''
    Plots the results of a varyingNumTrees() experiment, using a dictionary
    structure to hold the data. See the loadRes_varyingTrees() comments on the
    dictionary layout.
    '''
    xvals = data_dict['NumTrees']
    
    #prox forest trials
    pf_avg = data_dict['PF'].mean(axis=0)
    pf_std = data_dict['PF'].std(axis=0)
    pf_95_conf = 1.96 * pf_std / math.sqrt(data_dict['PF'].shape[0])

    #kdt forest trials
    kdt_avg = data_dict['KDT'].mean(axis=0)
    kdt_std = data_dict['KDT'].std(axis=0)
    kdt_95_conf = 1.96 * kdt_std / math.sqrt(data_dict['KDT'].shape[0])
    
    #plot average results of each, bounded by lower and upper bounds of 95% conf intv
    pl.hold(True)
    pl.errorbar(xvals, pf_avg/max_correct, yerr=pf_95_conf/max_correct, fmt='-r', 
                label="PF")
    pl.errorbar(xvals, kdt_avg/max_correct, yerr=kdt_95_conf/max_correct, fmt='-.b',
                label="KDT")
    pl.ylim([0,1.05])
    pl.title(dataset_name)
    pl.xlabel("Number of Trees in Forest")
    pl.ylabel("Percent Correct")
    pl.legend(loc='lower right')
    if show: pl.show()
    
def plotRes_all_varyingTrees(datadir=".", 
                             fn_list=["SIFT_10K_5trials_VaryingTrees.p",
                                      "MSER_10K_5trials_VaryingTrees.p",
                                      "SCIS_10K_5trials_VaryingTrees.p"]):
    titles=['SIFT 10K','MSER 10K', 'Scissors 10K']
    pl.hold(True)
    for i,fn in enumerate(fn_list):
        pl.subplot(1,3,i+1)
        d = loadRes_varyingTrees(datadir, fn)
        plotRes_varyingTrees(d, titles[i], max_correct=3000, show=False)
    pl.subplots_adjust(bottom=.15)
    pl.show()
    
if __name__ == '__main__':
    pass