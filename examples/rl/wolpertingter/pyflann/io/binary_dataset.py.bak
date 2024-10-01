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
import numpy
import os.path

def check(filename):
    f = open(filename,"r")
    header = f.read(6)
    if header[0:6]=="BINARY":
        return True
    return False

def save(dataset, filename):
    if not isinstance(dataset,numpy.ndarray):
        raise FLANNException("Dataset must be in numpy format")
    
    with open(filename+".meta", 'w') as fd_meta:
        fd_meta.write(\
"""BINARY
%d
%d
%s"""%(dataset.shape[0],dataset.shape[1],dataset.dtype.name))

    dataset.tofile(filename)


def load(filename, rows = -1, cols = -1, dtype = numpy.float32):
    
    if os.path.isfile(filename+".meta"):        
        with open(filename+".meta","r") as fd:
            header = fd.readline()
            assert( header[0:6] == "BINARY")
            rows = int(fd.readline())
            cols = int(fd.readline())
            dtype = numpy.dtype(fd.readline().strip())
    else:
        if rows==-1 or cols==-1:
            raise "No .meta file present, you must specify dataset rows, cols asd dtype"
    data = numpy.fromfile(file=filename, dtype=dtype, count=rows*cols)
    data.shape = (rows,cols)
    return data
    