'''
Created on Mar 28, 2012
@author: Stephen O'Hara

This module demonstrates the Latent Configuration Clustering algorithm,
which uses proximity forests and graph analysis as key components.

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

def LatentConfigurationClustering( dataList, dist_func, K, parallel_client=None, numtrees=27, true_labels=None ):
    '''
    High level function for computing latent configuration clustering. If you want
    more control, look at the functions that support the individual steps instead...
    @param dataList: A python list of data elements
    @param dist_func: A function which takes a pair of data elements and returns a
    scalar dissimilarity value
    @param K: The number of clusters to use
    @param parallel_client: If None, then the serial implementation of a proximity forest will be used.
    Otherwise, provide an ipython parallel client object.
    @param numtrees: The number of trees to use in the approx nearest neighbor
    proximity forest
    @param true_labels: If provided, the true labels will be used to score the v_measure
    score between the predicted labels (clusters) and the true labels.
    @return: A tuple of form (clusts, score, pforest, g), where clusts is the cluster
    membership of each sample, score is a number between 0 and 1 if true_labels are provided (-1 else),
    pforest is the proximity forest that was created, g is the connectivity graph that was created.
    '''
    #step 1, construct a proximity forest
    if not parallel_client is None:
        pforest = pf.ParallelProximityForest(parallel_client, numtrees, dist_func=dist_func)
    else:
        pforest = pf.ProximityForest(N=numtrees, treeClass=pf.ProximityTree, Tau=15, dist_func=dist_func)
    
    pforest.addList(dataList, range(len(dataList)))
    
    #step 2, generate connectivity graph
    g = pf.generateConnectivityGraph(pforest)
    
    #step 3, perform ward's linkage clustering on distance matrix generated from g
    if not true_labels is None:
        (clusts, score) = pf.hierarchicalClustering(g, K, true_labels, max_affinity=numtrees)
    else:
        (clusts, _) = pf.hierarchicalClustering(g, K, range(len(dataList)), max_affinity=numtrees)
        score = -1
        
    return (clusts, score, pforest, g)

def LCC_Exemplars( dataList, dist_func, K, numtrees=27 ):
    '''
    Unsupervised exemplar selection using Latent Configuration Clustering
    Requires networkx graph library to use.
    @return: A list of exemplar indexes. There will be K exemplars, and the index
    will indicate which of the elements of dataList is the exemplar.
    '''
    (clusts, _, _, g) = LatentConfigurationClustering(dataList, dist_func, K, numtrees)
    return LCC_ExemplarIdxs(clusts,g,numtrees)
      
def LCC_ExemplarIdxs(clusts, g, max_weight):
    '''
    Finds the exemplars from the clusts using connectivity graph g based on closeness centrality.
    @return: A list of exemplar indexes
    '''
    clust_ids = set(clusts)
    exemplar_idxs = []
    for c in clust_ids:
        e = pf.getExemplarNodeForCluster(g, clusts, c, max_weight=max_weight)
        exemplar_idxs.append(e)
        
    return exemplar_idxs    
        