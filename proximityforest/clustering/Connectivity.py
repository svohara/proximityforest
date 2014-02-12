'''
Created on Feb 13, 2012
@author: Stephen O'Hara

This module supports the generation of connectivity graphs from
a proximity forest structure, and methods for using the graph to
identify clusters and exemplars.

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

def _is_required(library):
    print "="*40
    print "Error loading proximityforest.clustering.Connectivity.py module."
    print "The %s library is required for Proximity Forest clustering."%library
    print "If you are not doing clustering, you can ignore this import warning."
    print "="*40
    return

try:
    import networkx as nx
except:
    _is_required('networkx')

try:
    import fastcluster as fc
except:
    _is_required('fastcluster')
    
try:
    import sklearn.cluster
    import sklearn.metrics
except:
    _is_required('scikits.learn (sklearn)')
    
import proximityforest as pf
import itertools as it
import scipy.cluster.hierarchy as spc
import numpy as np  
    
def generateConnectivityGraph(forest, uid_field=1):
    '''
    Given a proximity forest, a connectivity graph is produced where the
    weight between data elements is based on how often those two elements fall
    within the same leaf node of trees in the forest. (i.e., how often they
    are in the nearest neighbors of a tree in the forest.)
    @param uid_field: Defaults with the assumption that items in forest are
    tuples (sample_data, uid) where uid is the unique id of the sample. Other times,
    the items are organized as (uid, class/label), in which case set uid_field = 0.
    @param forest: A subspace forest or proximity forest object, where the data elements have unique
    identifiers as labels.
    @return: A weighted undirected graph containing the connectivity weights 
    '''
    g = nx.Graph()
    if isinstance(forest, pf.ParallelProximityForest):
        pfa = pf.ParallelProximityForestAnalysis(forest)
    else:
        pfa = pf.ProximityForestAnalysis(forest)
        
    allNB = pfa.getForestNeighborhoods(uid_field=uid_field)
    for ti, treeNB in enumerate(allNB):
        print "Processing connections from tree %d"%(ti+1)
        for id_list in treeNB:
            cxns = it.combinations(id_list,2)  #iterator for all-pairs w/i leaf node
            for (p,q) in cxns:
                try:
                    g[p][q]['weight'] += 1
                except KeyError:
                    g.add_weighted_edges_from([(p,q,1)])
    
    print "Connectivity graph construction completed."
    print "Graph has %d nodes and %e edges."%(g.order(), g.size())
    return g

def generateAffinityMatrix(g, max_affinity=None):
    '''
    Given a connectivity graph, g, created by the generateConnectivityGraph(ssforest) method,
    this function returns the data as a matrix.
    '''
    X = nx.adj_matrix(g) #will have zeros on diag, but diags should be max similarity...
    
    if max_affinity == None:
        max_affinity = X.max()  #max element in array
    
    di = np.diag_indices_from(X)
    X[di] = max_affinity  #set the diagonals to have maximum similarity/affinity
    
    return np.array(X)

def generateDistanceGraph(g, max_weight=27):
    '''
    Assuming the edge weights on g represent affinity (connectivity), we
    generate a version of the same graph where the edge weights have been
    converted to distances by simply subtracting from the max affinity.
    
    Since the edge weights of g represent
    affinities, a copy will be created that changes edges to be distances
    by subtracting the edge weight from the max_weight.
    g' has edge weights: W'(i,j) = max_weight - W(i,j) if i<>j, else 0.
    '''
    g2 = g.copy()
    #print "Creating distance graph from connectivity/affinity graph..."
    for i in range(g2.number_of_nodes() ):
        for j in range(i, g2.number_of_nodes() ):
            try:
                w = g2[i][j]['weight']
                g2[i][j]['weight'] = max_weight - w
            except KeyError:
                pass #not a valid edge...that's okay
    return g2
    
def generateAllPairsDistance(g, max_weight=27):
    '''
    Given connectivity graph g, this will return
    a numpy matrix that represents the all-pairs shortest path lengths
    using floyd-warshall algorithm. Since the edge weights of g represent
    affinities, a copy will be created that changes edges to be distances
    by subtracting the edge weight from the max_weight.
    g' has edge weights: W'(i,j) = max_weight - W(i,j) if i<>j, else 0.
    D = floyd_warshall_all_pairs(g')
    '''
    g2 = generateDistanceGraph(g, max_weight)
            
    print "Computing all-pairs shortest path distances..."
    D = nx.floyd_warshall_numpy(g2)
    return D

def getClusterIdxs(clusters, label):
    '''
    Return a list of indexes in the clusters list where the value == label.
    '''
    return sorted([i for i in range(len(clusters)) if clusters[i]==label])

def getNodesForCluster(g, clusters, label):
    '''
    Returns the subgraph of g that contains only the nodes with a given cluster assignment.
    '''
    node_ids = getClusterIdxs(clusters, label)
    return nx.subgraph(g, node_ids)
    
def getExemplarNodeForCluster(g, clusters, label, max_weight):
    '''
    @param g: The ssforest connectivity graph
    @param clusters: The output of spectral clustering (or other method) on the nodes of g. Let N be the
    number of nodes in g. Clusters is a list of length N, sorted in the same order as the sort order of
    the nodes of g, where each node is assigned an integer cluster label from the set {0...N-1}
    @param label: Which cluster to use for exemplar selection, an integer from set {0..N-1}
    @param max_weight: Specify the max affinity possible in the graph, typically this will be the size of the ssforest.
    @return: The index of the exemplar for that cluster.
    '''
    subG = getNodesForCluster(g, clusters, label) #edge weights are similarities
    if len(subG) < 1:
        print "The subgraph for label %d is empty."%label
        return None
    
    subGD = generateDistanceGraph(subG, max_weight) #edge weights are distances
    centrality_scores = nx.closeness_centrality(subGD)
    tmp = sorted([(score,idx) for (idx,score) in centrality_scores.viewitems()])
    (_, idx) = tmp[-1]  #sorted by score, so return last in list
    return idx
    
def hierarchicalClustering(g,k, labels, max_affinity=None):
    '''
    Performs hierarchical clustering using the connections in graph g. Edge weights
    are assumed to be affinity, thus higher weights means the nodes are more similar.
    Computes a distance matrix from the graph affinities, and clusters using
    the 'fastcluster' library implementation of ward's linkage.
    @param g: The graph as from generateConnectivityGraph
    @param k: Number of clusters
    @param labels: The ground truth labels used for measuring cluster accuracy
    @param max_affinity: The maximum similarity score that is possible on the graph.
    If None, then the max edge weight of the graph is used.
    @return: A tuple (clusts, score) where clusts is the ordered list of cluster
    indexes and score is the v-measure between clusts and labels.
    '''
    M = generateAffinityMatrix(g, max_affinity=max_affinity)
    if max_affinity is None:
        max_affinity = M.max()
        
    D = max_affinity - M
    
    Z = fc.ward(D) #linkage structure Z
    clusts = spc.fcluster(Z, k, criterion="maxclust")    
    try:
        score = sklearn.metrics.v_measure_score(labels, clusts)
    except:
        print "Warning: sklearn module not loaded. V_measure_score not computed."
        score = -1
    
    clusts = clusts - 1 #convert from 1-based to 0-based indexes
    return (clusts,score) 
    
def hierarchicalClusteringDendrogram(g, max_affinity=None,show_dendrogram=False):
    '''
    Generates the Ward's linkage structure on the connections in graph g. This
    function works the same as hierarchicalClustering(), but instead of returning
    the cluster membership for a given K, it returns the linkage structure and
    optionally shows the dendrogram.
    @param g: The graph as from generateConnectivityGraph
    @param max_affinity: The maximum similarity score that is possible on the graph.
    If None, then the max edge weight of the graph is used.
    @return: Z, the linkage structure
    '''
    M = generateAffinityMatrix(g, max_affinity=max_affinity)
    if max_affinity is None:
        max_affinity = M.max()
        
    D = max_affinity - M
    Z = fc.ward(D) #linkage structure Z
    if show_dendrogram:
        import pylab
        fig = pylab.figure()
        spc.dendrogram(Z)
        fig.show()

    return Z
    
def filter_edges(g, thresh):
    '''
    Given a weighted graph, g, from networkx library,
    this returns a new graph g2 containing only those
    edges with weights >= threshold value.
    '''
    edges_to_remove = []
    for (u,v) in g.edges_iter():
        if g[u][v]['weight'] < thresh:
            edges_to_remove.append( (u,v))
    g2 = g.copy()
    for (u,v) in edges_to_remove:
        g2.remove_edge(u,v)
    return g2
                   
                
if __name__ == '__main__':
    pass