'''
Created on Apr 20, 2012
@author: Stephen O'Hara

Classes used to perform various analysis operations
on proximity tree and forest structures, such as using
a forest for NN-classification.

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
#import proximityforest as pf
import scipy
import sys

class ProximityTreeAnalysis:
    '''
    This class implements useful functions for analyzing proximity trees, but where the functions
    may not be applicable to all trees, and thus separated from the base class implementation. For
    example, many of the functions assume the data items in the trees are (DataElement,Label) where
    Label is an integer class or cluster label. But in a base ProximityTree, the "Label" field may
    be a unique identifier to the sample, and thus not directly useful for cluster/class analysis.
    '''
    def __init__(self, proximityTree):
        self.tree = proximityTree
        
    def getLabelDistribution(self, node, numClasses):
        '''
        For use in trees that store an integer class label with the node items, this method
        returns the distribution of labels in a leaf node.
        @param node: A leaf node in the proximity tree to analyze
        @param numClasses: Since a node is likely to have only a small set of class labels represented,
        we need to be told how many classes there could be in order to return the distribution.
        @return: A histogram of the (integer) labels present in the current node. Histogram has a single
        bin per integer from 0 to numClasses - 1.
        @note: This function assumes that the items in the node are (DataElement,Label),
        where label is the class label, specified by an integer from 0 to numClasses-1.
        '''    
        labels = [ L for (_,L) in node.items]        
        h,_ = scipy.histogram(labels, bins=numClasses, range=[0,numClasses-1])
        return h*1.0/sum(h)
    
    def getNeighborLabelDistribution(self, T, numClasses):
        '''
        Get the distribution of class labels for the set of items (Data,Label) that are
        considered neighbors to the input sample T.
        '''
        node = self.tree.getLeafNode(T)
        return self.getLabelDistribution(node, numClasses)
    
    def computePurity(self):
        '''
        Computes the purity of all leaf nodes in tree
        @return: Tuple (nodeList, purityScores)
        @note: This function assumes that the items in the node are (DataElement,Label),
        where label is a cluster or class label, specified by an integer.
        '''
        nodeList = self.tree.getLeafNodes()
        purityScores = []
        for node in nodeList:
            purityScores.append( self.leafNodePurity(node))
        return (nodeList, purityScores)        
        
    def leafNodePurity(self, node):
        '''
        Compute the label purity of a single leaf node.
        @param node: A leaf node in the proximity tree to analyze
        @return: The label purity of a node in the proximity tree as the ratio of the class
        with the most samples to the total number of samples.
        @note: This function assumes that the items in the node are (DataElement,Label),
        where label is a cluster or class label, specified by an integer.
        '''        
        N = len(node.items)
        if N == 0: return 1.0
        
        labels_count = {}
        for (_,L) in node.items:
            if L in labels_count:
                labels_count[L] +=1
            else:
                labels_count[L] = 1
                
        maxcount = 0
        for lbl in labels_count:
            cnt = labels_count[lbl]
            if cnt > maxcount: maxcount = cnt
        return ( maxcount*1.0 )/N
    
    def getNeighborhoods(self, uid_field=1):
        '''
        Assumes the items are (sample_data, unique_id). Returns a list for each leaf node
        of just the unique_ids in that node.
        @param uid_field: By default, we assume the items in a leaf are organized
        as (sample,uid) where uid is the unique id of the sample. However, with some
        trees, the format is (uid, class). In which case, set uid_field=0.
        @return: List of lists, indicating which unique_ids are grouped together
        into the 'neighborhoods' (leaf nodes) of the tree.
        '''
        neighborhoods = []
        leaves = self.tree.getLeafNodes()
        for leaf in leaves:
            neighbors = [itm[uid_field] for itm in leaf.items]
            neighborhoods.append(neighbors)
        return neighborhoods
    
class ProximityForestAnalysis:
    '''
    A helper class for performing certain operations on forests that may not
    be universally applicable. KNN-voting, for example, assumes that the data
    elements have been added to the trees with class labels.
    '''
    def __init__(self, proximityForest):
        self.forest = proximityForest
    
    def _normalizeVotes(self, votes, numClasses):
        ''' Given a list of votes, which is just a list of integer labels,
        this computes the normalized histogram over the possible classes.
        '''
        h,_ = scipy.histogram(votes, range=[0,numClasses-1], bins=numClasses)
        return h*1.0/sum(h)
       
    def popular_vote(self, T, K, numClasses):
        '''
        Like KNN voting, but we allow duplicate neighbors...meaning if sample A is a KNN
        to query T in 3 different trees, it gets 3 votes. In pure KNN voting, we take
        the KNN from each tree, combine into a set (removes duplicates), and then rank
        order the combined set to yield the KNN of the forest. Another way of looking at
        it is whether we vote using all KNN from all trees, or if we use all KNN to select
        a single KNN list of the forest to vote.
        '''
        KNN_List = self.forest.getKNearestFromEachTree(T,K)         
        votes = [L for (_,_,L) in KNN_List]  #if there are N trees and K neighbors in each, then there are N*K votes
        nvotes = self._normalizeVotes(votes, numClasses)        
        return (scipy.argmax(nvotes), max(nvotes))
         
    def knn_vote(self, T, K, numClasses):
        '''Find the K-Nearest Neighbors in the leaf node. Use
        these to vote on the label. Do for each tree, and combine results.
        @param T: the input sample to be labeled
        @param K: The number of neighbors from each tree used in the voting.
        @param numClasses: How many classes in the data? We assume the labels
        in the forest go from 0 to numClasses-1.
        @return: A tuple (label,score) where label is the predicted label
        that received the most votes, and score is the normalized fraction
        of the votes it received (1.0=all votes).'''
        
        KNNs = self.forest.getKNearest(T, K)
        #KNNs like [(D1,N1,L1),(D2,N2,L2),(D3,N3,L3)], D is distance, N is neighbor, L is label
        
        votes = [ L for (_,_,L) in KNNs] #note, if we wanted to do weighted voting, we have the distances in KNN_List   
        nvotes = self._normalizeVotes(votes, numClasses)      
        
        return (scipy.argmax(nvotes), max(nvotes))
    
    def quick_vote(self, T, numClasses):
        '''
        Like knn_vote, but instead of inspecting each sample of a matching leaf node and returning
        the k-nearest to vote, we vote based on the label distribution of that node. In this way, we
        avoid the numerous distance comparisons that can occur for leaf nodes with many samples.
        @return: Tuple (label, score) for the label with max cumulative score across the tree.
        '''
        dist = self.getLabelDistribution(T, numClasses)
        c = scipy.argmax(dist)
        score = dist[0,c]
        return(c,score)
    
    def getLabelDistribution(self, T, numClasses, normalized=True):
        '''
        @param T: Input sample
        @param numClasses: How many integer-label classes are in the data set? [0..numClasses-1]
        @param normalized: If True (default), then the return distribution is normalized. Otherwise,
        it returns the sum of the distributions from each tree in the forest. The non-normalized version
        can be useful when combining distributions from two forests that have unequal number of trees.
        @return: The label distribution over the entire forest based on the combined distributions
        of the leaf nodes at each tree where T is sorted.
        '''
        cumulative_dist = scipy.zeros((1,numClasses))
        for i in range(self.forest.N):
            pta = ProximityTreeAnalysis( self.forest[i] )
            tmp_dist = pta.getNeighborLabelDistribution(T, numClasses)
            cumulative_dist += tmp_dist
            
        if normalized:
            return cumulative_dist / scipy.sum(cumulative_dist) 
        else:
            return cumulative_dist
    
    def getForestNeighborhoods(self, uid_field=1):
        '''
        @param uid_field: default assumes items in trees are (sample_data, uid) where uid is
        the unique id of the sample. For classification problems, often the items will
        be (uid, class). In which case, set uid_field=0.
        @return: a list of neighborhoods over the entire forest. Neighborhoods are list-of-lists structure returned
        by a call to ProximityTreeAnalysis.getNeighborhoods().
        '''
        allNeighborHoods = []
        for i in range(self.forest.N):
            pta = ProximityTreeAnalysis( self.forest[i] )
            nbrhd = pta.getNeighborhoods(uid_field=uid_field)
            allNeighborHoods.append(nbrhd)      
        return allNeighborHoods
    
    def classifySample(self, sample, numClasses, method="quick", K=1):
        '''
        Use nearest neighbors on the forest to classify a single input sample.
        For this to work, the items list in the leaf nodes of the trees must
        be a tuple (sample, label).
        @param sample: The new sample to classify
        @param numClasses: The number of class labels possible
        @param method: Specify either "quick" for quick-voting based on the leaf distributions,
        or "knn" for using the k-nearest of the forest to vote, or "popular" to use the k-nearest
        from all trees to vote (N*K votes).
        @param K: Used if method is "knn" or "popular". It is the number of neighbors to consider from
        each leaf node in the forest where the new sample is sorted.
        @return: A tuple (class,score) indicating the predicted class label (0-based index),
        and the confidence score.
        '''
        if method.lower() == "quick":
            res = self.quick_vote(sample, numClasses)
        elif method.lower() == "knn":
            res = self.knn_vote(sample, K, numClasses) 
        elif method.lower() == "popular":
            res = self.popular_vote(sample, K, numClasses)
        else:
            raise ValueError("Invalid classification method. Specify 'quick', 'knn', or 'popular'.")   
                           
        return res
        
    def classifyAll(self, samples, labels, numClasses, method="quick", K=1, verbose=False):    
        '''
        Use the proximity forest to perform nearest-neighbors classification on
        the test samples.
        @param samples: A list of test samples to classify
        @param labels: The target (true) labels of the samples
        @param numClasses: The number of class labels possible
        @param method: Specify one of "quick", "knn", or "popular". See comments for classifySample method.
        @param K: Only used if method is "knn" or "popular". It is the number of neighbors from
        each leaf node in the forest where the new sample is sorted.
        @return: Tuple (error_rate, confusions). Confusions is a list of pairs where a label
        mistake occurred [(pred,targ),....]. Confusion list can be used as input to showing
        a confusion matrix, such as with ConfusionMx() method of the Utility module.
        @note: labels are assumed to be 0-based contiguous integers. I.e. The set of unique
        labels is 0..(C-1) where C is the number of classes.
        '''
        confusions = []  #will be a list of pairs where a label mistake occurred [(pred,targ),(pred,targ),...]
        errors = 0
        
        print "Computing %d Predictions..."%len(labels)
        total = len(labels)
        for idx,T in enumerate(samples):
            (predict, conf) = self.classifySample(T, numClasses, method=method, K=K)
            if predict != labels[idx]:
                confusions.append( (predict, labels[idx]) )
                errors += 1
                if verbose:
                    print "%d:\tMiss!\tPred=%d\tTarget=%d\tConfidence=%f"%(idx,predict,labels[idx],conf)
            else:
                if verbose:
                    print "%d:\tHit! \tPred=%d\tTarget=%d\tConfidence=%f"%(idx,predict,labels[idx],conf)
            
            if not verbose:
                print_progress(idx, total)
            sys.stdout.flush()
            
        if not verbose: print ""
        error_rate = errors*1.0 / len(labels)
        
        print "Error rate is %3.3f"%error_rate
        
        return (error_rate, confusions)
        
class ParallelProximityForestAnalysis(ProximityForestAnalysis): 
    ''' A version of ProximityForestAnalysis that works correctly when applied
    to a ParallelProximityForest'''
    def __init__(self, forest):
        try:
            self.client = forest.client
        except:
            print "ERROR: Input forest to ParallelProximityForestAnalysis doesn't seem to be parallel."
            raise ValueError("Missing IPython Parallel client object in input forest.")
        self.forest = forest
        dview = self.client[:]
        with dview.sync_imports():
            import scipy
            from proximityforest import ProximityForestAnalysis, ProximityTreeAnalysis
        
    def getLabelDistribution(self, T, numClasses):
        '''
        @param T: Input sample
        @param numClasses: How many integer-label classes are in the data set? [0..numClasses-1]
        @return: The label distribution over the entire forest based on the combined distributions
        of the leaf nodes at each tree where T is sorted.
        '''
        dview = self.client[:]
        dview.block=True
        dview.execute('pfa = ProximityForestAnalysis(forest)', block=True)  #now each sub-forest in parallel structure has own pfa
        dview.push({'T':T, 'numClasses':numClasses}, block=True)
        dview.execute('tmp=pfa.getLabelDistribution(T,numClasses,normalize=False)', block=True)
        tmpList = dview['tmp']        
        X = scipy.vstack(tmpList)
        cumulative_dist = scipy.sum(X,0)            
        return cumulative_dist / scipy.sum(cumulative_dist) 
        
    def getForestNeighborhoods(self):
        '''
        @return: a dictionary where key=tree.ID and value is the neighborhoods list-of-lists structure returned
        by a call to ProximityTreeAnalysis.getNeighborhoods()
        '''
        dview = self.client[:]
        dview.block=True
        dview.execute('pfa = ProximityForestAnalysis(forest)', block=True)  #now each sub-forest in parallel structure has own pfa
        dview.execute('tmp=pfa.getForestNeighborhoods()', block=True)
        tmpList = dview['tmp']
        allNeighborhoods = [ nbrhd for sublist in tmpList for nbrhd in sublist ]
        return allNeighborhoods

def print_progress(cur, total):
    '''
    This function can be called in a processing loop
    to print out a progress indicator represented
    by up to 10 lines of dots, where each line 
    represents completion of 10% of the total iterations.
    @param cur: The current value (integer) of the iteration/count.
    @param total: The total number of iterations that must occur.
    '''
    one_line = 40 if total < 400 else round( total / 10.0 )
    one_dot = 1 if one_line / 40.0 < 1 else round( one_line / 40.0)    
    if (cur+1)%one_line == 0:
        print ' [%d]'%(cur+1)
    elif (cur+1)%one_dot == 0:
        print '.',
        sys.stdout.flush()    
    if cur+1 == total: print ""
