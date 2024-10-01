# Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
# Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
#
# THE BSD LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from ctypes import *
#from ctypes.util import find_library
from numpy import float32, float64, uint8, int32, matrix, array, empty, reshape, require
from numpy.ctypeslib import load_library, ndpointer
import os
from pyflann.exceptions import FLANNException
import sys

STRING = c_char_p


class CustomStructure(Structure):
    """
        This class extends the functionality of the ctype's structure
        class by adding custom default values to the fields and a way of translating
        field types.
    """
    _defaults_ = {}
    _translation_ = {}

    def __init__(self):
        Structure.__init__(self)
        self.__field_names = [f for (f, t) in self._fields_]
        self.update(self._defaults_)

    def update(self, dict):
        for k, v in dict.iteritems():
            if k in self.__field_names:
                setattr(self, k, self.__translate(k, v))

    def __getitem__(self, k):
        if k in self.__field_names:
            return self.__translate_back(k, getattr(self, k))

    def __setitem__(self, k, v):
        if k in self.__field_names:
            setattr(self, k, self.__translate(k, v))
        else:
            raise KeyError("No such member: " + k)

    def keys(self):
        return self.__field_names

    def __translate(self, k, v):
        if k in self._translation_:
            if v in self._translation_[k]:
                return self._translation_[k][v]
        return v

    def __translate_back(self, k, v):
        if k in self._translation_:
            for tk, tv in self._translation_[k].iteritems():
                if tv == v:
                    return tk
        return v


class FLANNParameters(CustomStructure):
    _fields_ = [
        ('algorithm', c_int),
        ('checks', c_int),
        ('cb_index', c_float),
        ('eps', c_float),
        ('trees', c_int),
        ('leaf_max_size', c_int),
        ('branching', c_int),
        ('iterations', c_int),
        ('centers_init', c_int),
        ('target_precision', c_float),
        ('build_weight', c_float),
        ('memory_weight', c_float),
        ('sample_fraction', c_float),
        ('log_level', c_int),
        ('random_seed', c_long),
    ]
    _defaults_ = {
        'algorithm': 'kdtree',
        'checks': 32,
        'eps': 0.0,
        'cb_index': 0.5,
        'trees': 1,
        'leaf_max_size': 4,
        'branching': 32,
        'iterations': 5,
        'centers_init': 'random',
        'target_precision': 0.9,
        'build_weight': 0.01,
        'memory_weight': 0.0,
        'sample_fraction': 0.1,
        'log_level': "warning",
        'random_seed': -1
    }
    _translation_ = {
        "algorithm": {"linear": 0, "kdtree": 1, "kmeans": 2, "composite": 3, "kdtree_simple": 4, "saved": 254, "autotuned": 255, "default": 1},
        "centers_init": {"random": 0, "gonzales": 1, "kmeanspp": 2, "default": 0},
        "log_level": {"none": 0, "fatal": 1, "error": 2, "warning": 3, "info": 4, "default": 2}
    }


default_flags = ['C_CONTIGUOUS', 'ALIGNED']
allowed_types = [float32, float64, uint8, int32]

FLANN_INDEX = c_void_p


def load_flann_library():

    root_dir = os.path.abspath(os.path.dirname(__file__))

    libnames = ['linux/libflann.so']
    libdir = 'lib'
    if sys.platform == 'win32':
        if sys.maxsize > 2 ** 32:
            libnames = ['win32/x64/flann.dll', 'win32/x64/libflann.dll']
        else:
            libnames = ['win32/x86/flann.dll', 'win32/x86/libflann.dll']
    elif sys.platform == 'darwin':
        libnames = ['darwin/libflann.dylib']

    while root_dir != None:
        for libname in libnames:
            try:
                flannlib = cdll[os.path.join(root_dir, libdir, libname)]
                return flannlib
            except Exception, e:
                pass
        tmp = os.path.dirname(root_dir)
        if tmp == root_dir:
            root_dir = None
        else:
            root_dir = tmp

    # if we didn't find the library so far, try loading without
    # a full path as a last resort
    for libname in libnames:
        try:
            # print "Trying",libname
            flannlib = cdll[libname]
            return flannlib
        except:
            pass

    return None

flannlib = load_flann_library()
if flannlib == None:
    raise ImportError('Cannot load dynamic library. Did you compile FLANN?')


class FlannLib:
    pass
flann = FlannLib()


flannlib.flann_log_verbosity.restype = None
flannlib.flann_log_verbosity.argtypes = [
    c_int  # level
]


flannlib.flann_set_distance_type.restype = None
flannlib.flann_set_distance_type.argtypes = [
    c_int,
    c_int,
]

type_mappings = (('float', 'float32'),
                 ('double', 'float64'),
                 ('byte', 'uint8'),
                 ('int', 'int32'))


def define_functions(str):
    for type in type_mappings:
        exec str % {'C': type[0], 'numpy': type[1]}

flann.build_index = {}
define_functions(r"""
flannlib.flann_build_index_%(C)s.restype = FLANN_INDEX
flannlib.flann_build_index_%(C)s.argtypes = [ 
        ndpointer(%(numpy)s, ndim = 2, flags='aligned, c_contiguous'), # dataset
        c_int, # rows
        c_int, # cols
        POINTER(c_float), # speedup 
        POINTER(FLANNParameters)  # flann_params
]
flann.build_index[%(numpy)s] = flannlib.flann_build_index_%(C)s
""")

flann.save_index = {}
define_functions(r"""
flannlib.flann_save_index_%(C)s.restype = None
flannlib.flann_save_index_%(C)s.argtypes = [
        FLANN_INDEX, # index_id
        c_char_p #filename                                   
] 
flann.save_index[%(numpy)s] = flannlib.flann_save_index_%(C)s
""")

flann.load_index = {}
define_functions(r"""
flannlib.flann_load_index_%(C)s.restype = FLANN_INDEX
flannlib.flann_load_index_%(C)s.argtypes = [
        c_char_p, #filename                                   
        ndpointer(%(numpy)s, ndim = 2, flags='aligned, c_contiguous'), # dataset
        c_int, # rows
        c_int, # cols
]
flann.load_index[%(numpy)s] = flannlib.flann_load_index_%(C)s
""")

flann.find_nearest_neighbors = {}
define_functions(r"""                          
flannlib.flann_find_nearest_neighbors_%(C)s.restype = c_int
flannlib.flann_find_nearest_neighbors_%(C)s.argtypes = [ 
        ndpointer(%(numpy)s, ndim = 2, flags='aligned, c_contiguous'), # dataset
        c_int, # rows
        c_int, # cols
        ndpointer(%(numpy)s, ndim = 2, flags='aligned, c_contiguous'), # testset
        c_int,  # tcount
        ndpointer(int32, ndim = 2, flags='aligned, c_contiguous, writeable'), # result
        ndpointer(float32, ndim = 2, flags='aligned, c_contiguous, writeable'), # dists
        c_int, # nn
        POINTER(FLANNParameters)  # flann_params
]
flann.find_nearest_neighbors[%(numpy)s] = flannlib.flann_find_nearest_neighbors_%(C)s
""")

# fix definition for the 'double' case

flannlib.flann_find_nearest_neighbors_double.restype = c_int
flannlib.flann_find_nearest_neighbors_double.argtypes = [
    ndpointer(float64, ndim=2, flags='aligned, c_contiguous'),  # dataset
    c_int,  # rows
    c_int,  # cols
    ndpointer(float64, ndim=2, flags='aligned, c_contiguous'),  # testset
    c_int,  # tcount
    ndpointer(int32, ndim=2, flags='aligned, c_contiguous, writeable'),  # result
    ndpointer(float64, ndim=2, flags='aligned, c_contiguous, writeable'),  # dists
    c_int,  # nn
    POINTER(FLANNParameters)  # flann_params
]
flann.find_nearest_neighbors[
    float64] = flannlib.flann_find_nearest_neighbors_double


flann.find_nearest_neighbors_index = {}
define_functions(r"""
flannlib.flann_find_nearest_neighbors_index_%(C)s.restype = c_int
flannlib.flann_find_nearest_neighbors_index_%(C)s.argtypes = [ 
        FLANN_INDEX, # index_id
        ndpointer(%(numpy)s, ndim = 2, flags='aligned, c_contiguous'), # testset
        c_int,  # tcount
        ndpointer(int32, ndim = 2, flags='aligned, c_contiguous, writeable'), # result
        ndpointer(float32, ndim = 2, flags='aligned, c_contiguous, writeable'), # dists
        c_int, # nn
        POINTER(FLANNParameters) # flann_params
]
flann.find_nearest_neighbors_index[%(numpy)s] = flannlib.flann_find_nearest_neighbors_index_%(C)s
""")

flannlib.flann_find_nearest_neighbors_index_double.restype = c_int
flannlib.flann_find_nearest_neighbors_index_double.argtypes = [
    FLANN_INDEX,  # index_id
    ndpointer(float64, ndim=2, flags='aligned, c_contiguous'),  # testset
    c_int,  # tcount
    ndpointer(int32, ndim=2, flags='aligned, c_contiguous, writeable'),  # result
    ndpointer(float64, ndim=2, flags='aligned, c_contiguous, writeable'),  # dists
    c_int,  # nn
    POINTER(FLANNParameters)  # flann_params
]
flann.find_nearest_neighbors_index[
    float64] = flannlib.flann_find_nearest_neighbors_index_double

flann.radius_search = {}
define_functions(r"""
flannlib.flann_radius_search_%(C)s.restype = c_int
flannlib.flann_radius_search_%(C)s.argtypes = [ 
        FLANN_INDEX, # index_id
        ndpointer(%(numpy)s, ndim = 1, flags='aligned, c_contiguous'), # query
        ndpointer(int32, ndim = 1, flags='aligned, c_contiguous, writeable'), # indices
        ndpointer(float32, ndim = 1, flags='aligned, c_contiguous, writeable'), # dists
        c_int, # max_nn
        c_float, # radius
        POINTER(FLANNParameters) # flann_params
]
flann.radius_search[%(numpy)s] = flannlib.flann_radius_search_%(C)s
""")

flannlib.flann_radius_search_double.restype = c_int
flannlib.flann_radius_search_double.argtypes = [
    FLANN_INDEX,  # index_id
    ndpointer(float64, ndim=1, flags='aligned, c_contiguous'),  # query
    ndpointer(int32, ndim=1, flags='aligned, c_contiguous, writeable'),  # indices
    ndpointer(float64, ndim=1, flags='aligned, c_contiguous, writeable'),  # dists
    c_int,  # max_nn
    c_float,  # radius
    POINTER(FLANNParameters)  # flann_params
]
flann.radius_search[float64] = flannlib.flann_radius_search_double


flann.compute_cluster_centers = {}
define_functions(r"""
flannlib.flann_compute_cluster_centers_%(C)s.restype = c_int
flannlib.flann_compute_cluster_centers_%(C)s.argtypes = [ 
        ndpointer(%(numpy)s, ndim = 2, flags='aligned, c_contiguous'), # dataset
        c_int,  # rows
        c_int,  # cols
        c_int,  # clusters 
        ndpointer(float32, flags='aligned, c_contiguous, writeable'), # result
        POINTER(FLANNParameters)  # flann_params
]
flann.compute_cluster_centers[%(numpy)s] = flannlib.flann_compute_cluster_centers_%(C)s
""")
# double is an exception
flannlib.flann_compute_cluster_centers_double.restype = c_int
flannlib.flann_compute_cluster_centers_double.argtypes = [
    ndpointer(float64, ndim=2, flags='aligned, c_contiguous'),  # dataset
    c_int,  # rows
    c_int,  # cols
    c_int,  # clusters
    ndpointer(float64, flags='aligned, c_contiguous, writeable'),  # result
    POINTER(FLANNParameters)  # flann_params
]
flann.compute_cluster_centers[
    float64] = flannlib.flann_compute_cluster_centers_double


flann.free_index = {}
define_functions(r"""
flannlib.flann_free_index_%(C)s.restype = None
flannlib.flann_free_index_%(C)s.argtypes = [ 
        FLANN_INDEX,  # index_id
        POINTER(FLANNParameters) # flann_params
]
flann.free_index[%(numpy)s] = flannlib.flann_free_index_%(C)s
""")


def ensure_2d_array(array, flags, **kwargs):
    array = require(array, requirements=flags, **kwargs)
    if len(array.shape) == 1:
        array = array.reshape(-1, array.size)
    return array
