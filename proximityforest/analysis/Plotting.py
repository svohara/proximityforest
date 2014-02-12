'''
Created on Apr 25, 2012
@author: Stephen O'Hara

This module contains code dependant on matplotlib for plotting
and visualizing results, and networkx for visualizing graph
and tree structures. Pygraphviz should also be installed for
visualizing the trees and forests.

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
import copy

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
except:
    print "Warning, unable to import required modules from matplotlib."
    print "Plotting features from the analysis module will not work."
    
try:
    import networkx as nx
except:
    print "Warning, unable to import networkx package."
    print "Plotting of trees, forests, and connectivity graphs will not work."
    
#----------------------------------------
# VISUALIZING PROXIMITY TREE AND FORESTS
#----------------------------------------
    
def drawForestGraph(forest, tree_idxs = None, colors=None):
    '''
    Generates a plot for each tree in the forest.
    @param forest: An object instance that is a ProximityForest or descendant
    @param tree_idxs: If None, a plot for each tree in the forest will be generated.
    Else, specify a list of tree indexes to draw, from 0 to len(forest)-1. For example,
    tree_idxs = [0,3,10] will draw the graphs of forest[0], forest[3] and forest[10].
    @param colors: A list of colors, same length as tree_idxs, to specify the node colors
    for the corresponding tree.
    '''
    forestGraph = nx.Graph()
    
    if tree_idxs is None:
        tree_idxs = range( len(forest))    
    
    if colors is None:
        colors = ['blue']*len(tree_idxs)
    
    for i in tree_idxs:
        print "Generating graph for tree %d"%i 
        labelDict = {}
        tg = generateTreeGraph( forest[i], labelDict=labelDict )
        print tg.name
        forestGraph = nx.disjoint_union(forestGraph, tg)
        #treeGraphs.append(tg)
        #drawTreeGraph(tg, node_size=40, show=False, color=colors[i])
    
    forestGraph.name = "Forest of %d Trees"%len(tree_idxs)
    plt.figure(1, figsize=(20,8))
    pos = nx.graphviz_layout(forestGraph, prog="dot")
    
    C = nx.connected_component_subgraphs(forestGraph)
    #C2 = sorted( [ (g.nodes()[0].split('.')[0], g) for g in C] ) #sort by tree id
    #for i,(treeid,g) in enumerate(C2):
        #print "Processing tree: %s"%treeid
    for i,g in enumerate(C):
        c = colors[i]
        nx.draw(g,
             pos,
             node_size=40,
             node_color=c,
             vmin=0.0,
             vmax=1.0,
             with_labels=False
             )
    
    plt.show()
    return forestGraph
        
def generateTreeGraph( cur_node, tg=None, labelDict=None ):
    '''
    Produces a networkx graph of the structure of a proximity tree
    @note: requires the networkx package is installed
    '''
    if tg is None:
        tg = nx.Graph()
        tg.name = cur_node.ID.split('.')[0]  #like t00 for tree 0
        
    if labelDict is None:
        labelDict = {}
        
    if cur_node.ID in labelDict:
        nid = labelDict[cur_node.ID]
    else:
        if len(labelDict) < 1:
            nid=0
        else:
            nid = max( labelDict.values() ) + 1
        labelDict[cur_node.ID] = nid
    
    #depth first search
    tg.add_node(nid)
    for n in cur_node.Children:
        if not n is None:
            tg = generateTreeGraph( n, tg=tg, labelDict=labelDict)
            tg.add_edge(nid, labelDict[n.ID] )  #connect parent to child
        
    return tg

def drawTreeGraph( G, color='blue' , node_size=40, save_as=None, show=False):
    '''
    @param G: The tree graph for a single tree of a proximity forest, generated
    by a call to generateTreeGraph
    @param color: The color used for the nodes in the output graph
    @param node_size: Specifies how big a circle to draw for the nodes
    @param save_as: A full path and file name, ending in .png, where an image
    of the resulting graph should be saved. Specify None to not save the image automatically.
    '''
    pos=nx.graphviz_layout(G,prog='dot',args='')
    plt.figure(figsize=(8,8))
    nx.draw(G,pos,node_size=node_size,alpha=1.0,node_color=color, with_labels=False)
    if not save_as is None:
        plt.savefig(save_as)
    if show: plt.show() 

#----------------------------------------
# VISUALIZING CLUSTERING / CONNECTIVITY GRAPHS
#----------------------------------------
def plotNodes(g, labels, labelstrs=None, colormap_str='gist_rainbow'): 
    '''
    Using matplotlib, which must be installed, this function
    will plot the nodes of a connectivity graph, grouped spatially based
    on edge weights, and colored according to the class labels.
    @param g: The graph structure from generateConnectivityGraph
    @param labels: The true class labels of each node, in the same order
    as the sort order of the nodes.
    @param labelstrs: Optional label names to use in legend in place of the
    actual label, which is typically an integer. If None, then the actual
    label values will be used.
    @param colormap_str: A string that indicates one of the matplotlib built-in colormaps to use.
    '''
    plt.clf()
    if labelstrs == None:
        labelstrs = sorted([str(i) for i in list(set(labels))]) #the set of unique labels
    NUM_COLORS = len(labelstrs)
    #generate the list of color assignments to the nodes based on class label
    cm = plt.get_cmap(colormap_str)
    nodeColors = [ cm(1.0*i/NUM_COLORS) for i in labels]
    nx.draw_networkx_nodes(g, nx.spring_layout(g),node_color=nodeColors, cmap=cm)
    
    #plot legend
    rs = []
    for i in range( len(labelstrs) ):
        colr = cm(1.0*i/NUM_COLORS) #colormap index between 0.0 and 1.0, transformed to rgba tuple
        #print colr
        rs.append( plt.Circle((0,0),1,fc=colr ))
    plt.legend(rs, labelstrs, loc="lower right")
    plt.show()
    
def plotCxns(g, min_weight=1):
    '''
    Using matplotlib, which must be installed, this function
    will visualize the connections in the graph g, where the
    edge weights meet or exceed the min_weight threshold.
    '''
    #create a subgraph containing only desired edges
    sg=nx.Graph( [ (u,v,d) for u,v,d in g.edges(data=True) if d['weight']>=min_weight] ) 
    nx.draw(sg)
    plt.show()
    return sg

#----------------------------------------
# PLOTTING RESULTS OF EVALUATIONS
# WHERE RESULTS ARE STORED IN AN 
# EXPERIMENTALERRORRESULTS OBJECT
#----------------------------------------      
    
def plotRes(res, title="Classification Accuracy", subtitle=None, highlight_idx=None, axis=None, 
            xlabel=None, legendLoc='center right'):
    '''
    Generates a simple plot for a single results object of type 'ExperimentalErrorResults', from
    the analysis.Evaluation package.
    @param axis: A tuple (x1,x2,y1,y2) that specifies the extents of the x and y-axis respectively
    in terms of plot-coordinates.
    @param highlight_idx: To highlight one of the results in the plot, specify its index. A horizontal
    and vertical dashed line will be drawn and its numerical score will be printed on the lines.
    @param xlabel: The label for the x-axis. If None, then res.paramName will be used.
    @param legendLoc: The location for the legend, using the terms provided by matplotlib. Set to None
    to suppress the legend. 'center right' is default. The contents of the legend is the res.desc field
    of the results object.
    '''
    if axis == None:
        plt.axis([3,50,90,100])
    else:
        plt.axis(axis)
    
    ax = plt.axes()        
    plt.plot(res.params, 100*res.getMeanAccuracy(), "b", label=res.desc) 
    
    #plot a shaded region around res1 line
    # that indicates the confidence interval.
    (low,high) = res.getMeanAccuracy_ci()
    low = 100*low
    high = 100*high
    tmp = copy.copy(list(res.params))
    tmp2 = copy.copy(list(high))
    tmp.reverse()
    tmp2.reverse()
    z1 = zip(res.params,low)
    z2 = zip(tmp,tmp2)
    z3 = z1+z2
    codes = [1 for _ in z3] #lineTo
    codes[0] = 0 #moveto
    codes[-1] = 79 #close polygon
    path = mpath.Path(z3) #, codes)
    patch = mpatches.PathPatch(path, alpha=0.15)
    ax.add_patch(patch)
    
    if not legendLoc is None: plt.legend(loc=legendLoc)   
    if subtitle != None:
        titlestr = "%s\n%s"%(title,subtitle)
        plt.title(titlestr)
    else:
        plt.title(title)    
    
    plt.ylabel("Accuracy")
    if xlabel==None:
        xlabel = res.paramName
    plt.xlabel(xlabel)
    
    if highlight_idx != None:
        idx = highlight_idx
        score = res.getMeanAccuracy()[idx]*100
        sz = res.params[idx]
        #print plt.axis()
        plt.hlines(score, plt.axis()[0], plt.axis()[1], linestyles='dotted', colors=[(0,0,0,1)])
        plt.vlines(sz, plt.axis()[2], plt.axis()[3], linestyles='dotted', colors=[(0,0,0,1)])
        plt.text(1.001*plt.axis()[0],1.001*score,"%2.2f"%score)
        plt.text(sz, 1.001*plt.axis()[2], "%2.2f"%sz)
        
    plt.show()
        
def plotComparison(res1, res2, res3, key_result=1, highlight_idx=3, 
                   axis=None, title=None, xlabel=None, comparison=None):
    '''
    Generates the comparison plot between three 'ExperimentalErrorResults' result objects.
    @param res1: A result object for the first data set in comparison
    @param res2: A result object
    @param res3: A result object
    @param key_result: Which of the three result objects is the one to be highlighted? 1,2,or3?
    @param highlight_idx: Of the highlighted result set, which index in the series should be
    highlighted?
    @param comparison: Specify a tuple (text,score) to plot an additional horizontal line
    on the chart that represents another score to compare to, such as a published best
    result from another method. Specify None (default) for no such comparison. Score should
    be a percentage [0,100].
    '''
    if axis == None:
        plt.axis([0,50,92,100])
    else:
        plt.axis(axis)

    if key_result == 2:
        keyres = res2
    elif key_result == 3:
        keyres = res3
    else:
        keyres = res1

    plt.plot(res1.params, 100*res1.getMeanAccuracy(), "bd-", label=res1.desc) 
    #
    #plt.plot(res1.params, low, "b--")
    #plt.plot(res1.params, high, "b--")
    
    plt.plot(res2.params, 100*res2.getMeanAccuracy(), "ro-", label=res2.desc) 
    plt.plot(res3.params, 100*res3.getMeanAccuracy(), "g*-", label=res3.desc) 
    
    plt.legend(loc='lower right')    
    
    if title != None:
        plt.title(title, fontsize=24)
        
    plt.ylabel("Percent Correct", fontsize=18)
    
    if xlabel is None:
        xlabel = res1.paramName
    plt.xlabel(xlabel, fontsize=18)
        
    if highlight_idx != None:
        idx = highlight_idx
        score = 100*keyres.getMeanAccuracy()[idx]
        sz = keyres.params[idx]
        plt.hlines(score, plt.axis()[0], plt.axis()[1], linestyles='dotted', colors=[(0,0,0,1)])
        plt.vlines(sz, plt.axis()[2], plt.axis()[3], linestyles='dotted', colors=[(0,0,0,1)])
        plt.text(plt.axis()[0]+0.5,score+.2,"Ours: %3.2f%% "%score)
        #plt.text(sz, 92.1, "%d"%sz)    
        
        #plot comparison score
        if not comparison is None:
            (ctxt,cscore) = comparison
            plt.hlines(cscore, plt.axis()[0], plt.axis()[1], linestyles='dashed', colors="red")
            plt.text(plt.axis()[0]+0.5,cscore+.3,"%s: %3.2f%%"%(ctxt,cscore) )
        
    ax = plt.axes()
    ax.set_xticks(keyres.params)
    
    #plot a shaded region around res1 line
    # that indicates the confidence interval.
    (low,high) = keyres.getMeanAccuracy_ci()
    low = 100*low
    high = 100*high
    tmp = copy.copy(list(keyres.params))
    tmp2 = copy.copy(list(high))
    tmp.reverse()
    tmp2.reverse()
    z1 = zip(keyres.params,low)
    z2 = zip(tmp,tmp2)
    z3 = z1+z2
    codes = [1 for _ in z3] #lineTo
    codes[0] = 0 #moveto
    codes[-1] = 79 #close polygon
    path = mpath.Path(z3) #, codes)
    patch = mpatches.PathPatch(path, alpha=0.15)
    ax.add_patch(patch)
    
    plt.show()
    