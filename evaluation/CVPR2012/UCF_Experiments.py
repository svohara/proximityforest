'''
Created on Nov 4, 2011
@author: Steve O'Hara
Classification experiments on UCF sports using tensor subspace forests
'''
import proximityforest as pf
from evaluation.action_data_sets.ucf_sports.DataSource import *
#import scipy as sp


def LeaveOneOut(UCFDat, flippage=True, treeClass=pf.TensorSubspaceTree, N=9, method="knn", K=1, **kwargs):
    '''Leave one out classification protocol
    NOTE: This protocol is not well-optimized for randomized forests, because we have to rebuild the
    forest at each iteration. So if we have 150 videos to test, we'll have to build the forest 150
    times...so this protocol will take a while to process!
    '''
    labels = UCFDat.getLabels()
    vidclips = UCFDat.getData()
    errs = 0
    cfs = []
    for i in range(len(labels)):
        test = (vidclips[i],labels[i])
        ll = labels[0:i] + labels[i+1:]
        vv = vidclips[0:i] + vidclips[i+1:]
        
        if flippage:
            #Add mirror-images of all training as well
            print "Adding mirror image video clips to training set..."
            vv2 = []
            for clip in vv: vv2.append( pf.flipTensor(clip) )
            vv += vv2
            ll += ll
            
        train = (vv,ll)  
        print "\n----------------------------------"  
        print "LOO Classification of Video %d"%i 
        print "----------------------------------"
        #build forest from all the other samples
        tracklet_data = pf.build_simple_TD_Structure(train[0], train[1])    
        forest = pf.TensorSubspaceForest(N, tracklet_data, treeClass=treeClass, **kwargs)
        forest.addList(range(len(vv)), ll)
        #then classify this sample
        pfa = pf.ProximityForestAnalysis(forest) 
        pred, _ = pfa.classifySample(test[0], 10, method=method, K=K)
        targ = test[1]
        if pred != targ:
            cfs.append((pred,targ))
            errs += 1

        print "Cumulative mean error rate: %3.3f"%( (errs*1.0)/(i+1) )
        print "----------------------------------"

    cfMx = pf.confusionMatrix(cfs, UCFDat.class_sizes)
    print cfMx
    errRate = (errs*1.0)/len(labels)
    pct = 100*(1-errRate)
    return (cfMx, pct, cfs)
    
if __name__ == '__main__':
    pass