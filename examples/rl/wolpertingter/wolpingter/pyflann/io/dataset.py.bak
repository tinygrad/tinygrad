#Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
#Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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

from __future__ import with_statement

from pyflann.exceptions import FLANNException
import binary_dataset
import dat_dataset
import npy_dataset
import hdf5_dataset

import os.path
from numpy import float32

dataset_formats = { 
    'bin' : binary_dataset, 
    'dat' : dat_dataset, 
    'npy' : npy_dataset,
    'hdf5' : hdf5_dataset 
}


def load(filename, rows = -1, cols = -1, dtype = float32, **kwargs):
    
    for format in dataset_formats.values():
        if format.check(filename):
            return format.load(filename, rows, cols, dtype, **kwargs)
    raise FLANNException("Error: Unknown dataset format")
    
    
def save(dataset, filename, format = None, **kwargs):    
    try:
        if format is None:
            basename,extension = os.path.splitext(filename)
            format = extension[1:]
        handler = dataset_formats[format]
        handler.save(dataset, filename, **kwargs)
    except Exception,e:
        raise FLANNException(e)
