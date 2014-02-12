'''
This package is basically just a folder to store
pre-packaged data sets for use in approximate
nearest neighbor evaluation of Proximity Forest.

The three data sets are 10,000 SIFT features,
1 Million MSER features, and 250,000 3D point
cloud data.

See paper:
S. O'Hara, B.A. Draper, Are you using the right approximate nearest neighbor algorithm,
Proc. IEEE Workshop on Applications of Computer Vision (WACV), 2013

'''

NN_DATA_SIFT = "sift_10K.p"  #pre-extracted SIFT features in pickle file
NN_DATA_MSER = "MSER_7M.npy" #pre-extracted MSER features in a numpy array
NN_DATA_3D = "Scissors_handle_points.txt" #text file with 3D point cloud data
