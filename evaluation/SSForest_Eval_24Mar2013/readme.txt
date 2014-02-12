After having published Classification results at CVPR, and in my dissertation,
I found that rewrites to the code base not only improved efficiency and
modularity, but that a tweak to how the forest is used for NN classification
resulted in improved performance.

The data files in this directory were created in March 2013. The high-level
result is that median splitting is as good or better than entropy splitting,
with the additional benefit of simplicity.

Classification results on KTH using schuldt's partitions are improved to
about 98.3% averaged over 10 trials. Classification on Cambridge Gestures
was improved to 90.9% average over 10 trials.

The data files are python pickle files. Use cPickle to restore them. It would
probably work best of you were in the .../evaluations directory, ran IPython,
and then loaded/ran the script for KTH or CG, so that all the appropriate objects
are defined, etc. Then you can unpickle the result files. Some of the KTH
results are in a format that can be used directly with pf.plotRes(...) function.


--Stephen O'Hara
Colorado State University
svohara@cs.colostate.edu


