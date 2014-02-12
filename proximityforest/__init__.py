'''
Proximity Forest
Top-level Package and Namespace

Stephen O'Hara
April 25, 2012

The top level of this package contains commonly used classes
and functions from the Proximity Forest library. Some less
commonly used components must be directly imported by the
user from the appropriate sub-package.

A good convention is to import the top-level namespace
using something like:

    import proximityforest as pf
    
And then reference the desired components like:
    forest = pf.ParallelProximityForest(...)
  
===============================================================  
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

__all__ = ['ANN','clustering','common','tracklets']


#Approximate Nearest Neighbor (ANN) using ProximityForests
from ANN.ProximityForest import *
from ANN.Parallel import *
from ANN.SubspaceForest import *

#Application of ProximityForests to tracklet data
from tracklets.Tracklets import *
from tracklets.TensorSubspaceForest import *
from tracklets.Parallel import *

#Application of ProximityForests for clustering
from clustering.Connectivity import *
from clustering.LCC import *

#Misc useful functions
from common.Utility import *
from common.Tensor import *

#Functions for analyzing a ProximityForest and
# evaluating and visualizing performance
from analysis.Analysis import *
from analysis.Evaluation import *
from analysis.Plotting import *
