#Copyright 2008-2010  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
#Copyright 2008-2010  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
#
#THE BSD LICENSE
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions
#are met:
#
#1. Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
#IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
#NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
#THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from bindings.flann_ctypes import *
from io.dataset import *
import numpy.random as _rn


index_type = int32

def set_distance_type(distance_type, order = 0):
    """
    Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, 
    hik, hellinger, cs, kl.
    """
    
    distance_translation = { "euclidean" : 1, 
                            "manhattan" : 2, 
                            "minkowski" : 3,
                            "max_dist" : 4,
                            "hik" : 5,
                            "hellinger" : 6,
                            "chi_square" : 7,
                            "cs" : 7,
                            "kullback_leibler" : 8,
                            "kl" : 8,
                            }
    if type(distance_type)==str:
        distance_type = distance_translation[distance_type]

    flannlib.flann_set_distance_type(distance_type,order)

# This class is derived from an initial implementation by Hoyt Koepke (hoytak@cs.ubc.ca)
class FLANN:
    """
    This class defines a python interface to the FLANN lirary.
    """
    __rn_gen = _rn.RandomState()
    
    _as_parameter_ = property( lambda self: self.__curindex )

    def __init__(self, **kwargs):
        """
        Constructor for the class and returns a class that can bind to
        the flann libraries.  Any keyword arguments passed to __init__
        override the global defaults given.
        """
        
        self.__rn_gen.seed()

        self.__curindex = None
        self.__curindex_data = None
        self.__curindex_type = None
        
        self.__flann_parameters = FLANNParameters()        
        self.__flann_parameters.update(kwargs)

    def __del__(self):
        self.delete_index()

        
    ################################################################################
    # actual workhorse functions

    def nn(self, pts, qpts, num_neighbors = 1, **kwargs):
        """
        Returns the num_neighbors nearest points in dataset for each point
        in testset.
        """
        
        if not pts.dtype.type in allowed_types:
            raise FLANNException("Cannot handle type: %s"%pts.dtype)

        if not qpts.dtype.type in allowed_types:
            raise FLANNException("Cannot handle type: %s"%pts.dtype)

        if pts.dtype != qpts.dtype:
            raise FLANNException("Data and query must have the same type")
        
        pts = ensure_2d_array(pts,default_flags) 
        qpts = ensure_2d_array(qpts,default_flags) 
        
        npts, dim = pts.shape
        nqpts = qpts.shape[0]

        assert(qpts.shape[1] == dim)
        assert(npts >= num_neighbors)

        result = empty( (nqpts, num_neighbors), dtype=index_type)
        if pts.dtype==float64:
            dists = empty( (nqpts, num_neighbors), dtype=float64)
        else:
            dists = empty( (nqpts, num_neighbors), dtype=float32)
                
        self.__flann_parameters.update(kwargs)

        flann.find_nearest_neighbors[pts.dtype.type](pts, npts, dim, 
                                                     qpts, nqpts, result, dists, num_neighbors, 
                                                     pointer(self.__flann_parameters))

        if num_neighbors == 1:
            return (result.reshape( nqpts ), dists.reshape(nqpts))
        else:
            return (result,dists)


    def build_index(self, pts, **kwargs):
        """
        This builds and internally stores an index to be used for
        future nearest neighbor matchings.  It erases any previously
        stored indexes, so use multiple instances of this class to
        work with multiple stored indices.  Use nn_index(...) to find
        the nearest neighbors in this index.

        pts is a 2d numpy array or matrix. All the computation is done
        in float32 type, but pts may be any type that is convertable
        to float32. 
        """
        
        if not pts.dtype.type in allowed_types:
            raise FLANNException("Cannot handle type: %s"%pts.dtype)

        pts = ensure_2d_array(pts,default_flags) 
        npts, dim = pts.shape
        
        self.__ensureRandomSeed(kwargs)
        
        self.__flann_parameters.update(kwargs)

        if self.__curindex != None:
            flann.free_index[self.__curindex_type](self.__curindex, pointer(self.__flann_parameters))
            self.__curindex = None
                
        speedup = c_float(0)
        self.__curindex = flann.build_index[pts.dtype.type](pts, npts, dim, byref(speedup), pointer(self.__flann_parameters))
        self.__curindex_data = pts
        self.__curindex_type = pts.dtype.type
        
        params = dict(self.__flann_parameters)
        params["speedup"] = speedup.value
        
        return params


    def save_index(self, filename):
        """
        This saves the index to a disk file.
        """
        
        if self.__curindex != None:
            flann.save_index[self.__curindex_type](self.__curindex, c_char_p(filename))

    def load_index(self, filename, pts):
        """
        Loads an index previously saved to disk.
        """
                
        if not pts.dtype.type in allowed_types:
            raise FLANNException("Cannot handle type: %s"%pts.dtype)

        pts = ensure_2d_array(pts,default_flags) 
        npts, dim = pts.shape

        if self.__curindex != None:
            flann.free_index[self.__curindex_type](self.__curindex, pointer(self.__flann_parameters))
            self.__curindex = None
            self.__curindex_data = None
            self.__curindex_type = None
        
        self.__curindex = flann.load_index[pts.dtype.type](c_char_p(filename), pts, npts, dim)
        self.__curindex_data = pts
        self.__curindex_type = pts.dtype.type

    def nn_index(self, qpts, num_neighbors = 1, **kwargs):
        """
        For each point in querypts, (which may be a single point), it
        returns the num_neighbors nearest points in the index built by
        calling build_index.
        """

        if self.__curindex == None:
            raise FLANNException("build_index(...) method not called first or current index deleted.")

        if not qpts.dtype.type in allowed_types:
            raise FLANNException("Cannot handle type: %s"%qpts.dtype)

        if self.__curindex_type != qpts.dtype.type:
            raise FLANNException("Index and query must have the same type")

        qpts = ensure_2d_array(qpts,default_flags) 

        npts, dim = self.__curindex_data.shape

        if qpts.size == dim:
            qpts.reshape(1, dim)

        nqpts = qpts.shape[0]

        assert(qpts.shape[1] == dim)
        assert(npts >= num_neighbors)
        
        result = empty( (nqpts, num_neighbors), dtype=index_type)
        if self.__curindex_type==float64:
            dists = empty( (nqpts, num_neighbors), dtype=float64)
        else:
            dists = empty( (nqpts, num_neighbors), dtype=float32)

        self.__flann_parameters.update(kwargs)

        flann.find_nearest_neighbors_index[self.__curindex_type](self.__curindex, 
                    qpts, nqpts,
                    result, dists, num_neighbors,
                    pointer(self.__flann_parameters))

        if num_neighbors == 1:
            return (result.reshape( nqpts ), dists.reshape( nqpts ))
        else:
            return (result,dists)
        
        
    def nn_radius(self, query, radius, **kwargs):
        
        if self.__curindex == None:
            raise FLANNException("build_index(...) method not called first or current index deleted.")

        if not query.dtype.type in allowed_types:
            raise FLANNException("Cannot handle type: %s"%query.dtype)

        if self.__curindex_type != query.dtype.type:
            raise FLANNException("Index and query must have the same type")

        npts, dim = self.__curindex_data.shape        
        assert(query.shape[0]==dim)
        
        result = empty( npts, dtype=index_type)
        if self.__curindex_type==float64:
            dists = empty( npts, dtype=float64)
        else:
            dists = empty( npts, dtype=float32)
        
        self.__flann_parameters.update(kwargs)

        nn = flann.radius_search[self.__curindex_type](self.__curindex, query, 
                                         result, dists, npts,
                                         radius, pointer(self.__flann_parameters))
        
        
        return (result[0:nn],dists[0:nn])

    def delete_index(self, **kwargs):
        """
        Deletes the current index freeing all the momory it uses. 
        The memory used by the dataset that was indexed is not freed.
        """

        self.__flann_parameters.update(kwargs)
        
        if self.__curindex != None:
            flann.free_index[self.__curindex_type](self.__curindex, pointer(self.__flann_parameters))
            self.__curindex = None
            self.__curindex_data = None

    ##########################################################################################
    # Clustering functions

    def kmeans(self, pts, num_clusters, max_iterations = None,
               dtype = None, **kwargs):
        """
        Runs kmeans on pts with num_clusters centroids.  Returns a
        numpy array of size num_clusters x dim.  

        If max_iterations is not None, the algorithm terminates after
        the given number of iterations regardless of convergence.  The
        default is to run until convergence.

        If dtype is None (the default), the array returned is the same
        type as pts.  Otherwise, the returned array is of type dtype.  

        """
        
        if int(num_clusters) != num_clusters or num_clusters < 1:
            raise FLANNException('num_clusters must be an integer >= 1')
        
        if num_clusters == 1:
            if dtype == None or dtype == pts.dtype:
                return mean(pts, 0).reshape(1, pts.shape[1])
            else:
                return dtype(mean(pts, 0).reshape(1, pts.shape[1]))

        return self.hierarchical_kmeans(pts, int(num_clusters), 1, 
                                        max_iterations, 
                                        dtype, **kwargs)
        
    def hierarchical_kmeans(self, pts, branch_size, num_branches,
                            max_iterations = None, 
                            dtype = None, **kwargs):
        """
        Clusters the data by using multiple runs of kmeans to
        recursively partition the dataset.  The number of resulting
        clusters is given by (branch_size-1)*num_branches+1.
        
        This method can be significantly faster when the number of
        desired clusters is quite large (e.g. a hundred or more).
        Higher branch sizes are slower but may give better results.

        If dtype is None (the default), the array returned is the same
        type as pts.  Otherwise, the returned array is of type dtype.  
        
        """
        
        # First verify the paremeters are sensible.

        if not pts.dtype.type in allowed_types:
            raise FLANNException("Cannot handle type: %s"%pts.dtype)

        if int(branch_size) != branch_size or branch_size < 2:
            raise FLANNException('branch_size must be an integer >= 2.')

        branch_size = int(branch_size)

        if int(num_branches) != num_branches or num_branches < 1:
            raise FLANNException('num_branches must be an integer >= 1.')

        num_branches = int(num_branches)

        if max_iterations == None: 
            max_iterations = -1
        else:
            max_iterations = int(max_iterations)


        # init the arrays and starting values
        pts = ensure_2d_array(pts,default_flags) 
        npts, dim = pts.shape
        num_clusters = (branch_size-1)*num_branches+1;
        
        if pts.dtype.type == float64:
            result = empty( (num_clusters, dim), dtype=float64)
        else:
            result = empty( (num_clusters, dim), dtype=float32)

        # set all the parameters appropriately
        
        self.__ensureRandomSeed(kwargs)
        
        params = {"iterations"       : max_iterations,
                    "algorithm"        : 'kmeans',
                    "branching"        : branch_size,
                    "random_seed"      : kwargs['random_seed']}
        
        self.__flann_parameters.update(params)
        
        numclusters = flann.compute_cluster_centers[pts.dtype.type](pts, npts, dim,
                                        num_clusters, result, 
                                        pointer(self.__flann_parameters))
        if numclusters <= 0:
            raise FLANNException('Error occured during clustering procedure.')

        if dtype == None:
            return result
        else:
            return dtype(result)
        
    ##########################################################################################
    # internal bookkeeping functions

        
    def __ensureRandomSeed(self, kwargs):
        if not 'random_seed' in kwargs:
            kwargs['random_seed'] = self.__rn_gen.randint(2**30)
        


