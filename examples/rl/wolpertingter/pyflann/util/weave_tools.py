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

import sys

try:
    from scipy.weave import ext_tools
except ImportError:
    sys.stderr.write('Weave is required by the %s module.\n'%__name__)
    sys.stderr.write('It is included as a standalone package or as part of scipy.\n')
    sys.stderr.write('See www.scipy.org/Weave for more information.')
    sys.exit(1)

from string import split, join
import os
import numpy
import struct


# type values
float64_1d = numpy.empty((1),dtype=numpy.float64)
float64_2d = numpy.empty((1,1),dtype=numpy.float64)
float64_3d = numpy.empty((1,1,1),dtype=numpy.float64)

float32_1d = numpy.empty((1),dtype=numpy.float32)
float32_2d = numpy.empty((1,1),dtype=numpy.float32)
float32_3d = numpy.empty((1,1),dtype=numpy.float32)

int32_1d = numpy.empty((1),dtype=numpy.int32)
int32_2d = numpy.empty((1,1),dtype=numpy.int32)
int32_3d = numpy.empty((1,1,1),dtype=numpy.int32)


struct_support_code = r"""
template <typename S>
S* py_to_struct(PyObject* obj)
{
    S* ptr;
    int length;
    PyString_AsStringAndSize(obj,(char**)&ptr,&length);
    return ptr;
}
"""


class CModule:

    def __init__(self, suppress_warnings = True, force_name = None):
        
        if type(force_name) != type(""):
            call_frame = sys._getframe().f_back
            name = call_frame.f_globals['__name__']
        else:
            name = force_name
        self.module = sys.modules[name]
        self.dest_dir = os.path.dirname(self.module.__file__)
        
        self._module_name = split(name,".")[-1]+"_c"

        # check to see if rebuild needed
        self.extension = ext_tools.ext_module(self._module_name)
        self.customize = self.extension.customize
        
        self.customize.add_include_dir(self.dest_dir)
        self.customize.add_support_code(struct_support_code)
        
        if suppress_warnings:
            self.customize.add_extra_compile_arg('-Wno-unused-variable')
            self.customize.add_extra_compile_arg('-Wno-write-strings')
            #self.customize.add_extra_compile_arg('-Wno-deprecated')
            #self.customize.add_extra_compile_arg('-Wno-unused')
        

    def get_name():
        return self._module_name

    def include(self,header):
        self.customize.add_header(header)
        
    def add_support_code(self,code):
        self.customize.add_support_code(code)

    def extra_args(self,*varargs):
        for t in varargs:
            assert(type(t) == tuple)
            assert(len(t) == 2)
            assert(type(t[0]) == str)
        
        def decorate(func):
            name = func.__name__                
            code = func()
            if type(code) != type(""):
                code = func.__doc__
            import inspect                
            (args,_,_,defaults) = inspect.getargspec(func)
            (file,line) = inspect.getframeinfo(inspect.currentframe().f_back)[0:2]
            code = ('#line %d "%s"\n'%(line,file))+code
            defaults = [] if defaults==None else defaults
            if len(args) != len(defaults):
                raise Exception("The %s function must have default values for all arguments"%name)
            arg_tuples = list(zip(args,defaults)) + list(varargs)
            self.add_function(name,code,*arg_tuples)
            return func
        return decorate        

    def __call__(self,func):        
        name = func.__name__
        code = func.__doc__
        if code == None:
            code = func()
        import inspect                
        (args,_,_,defaults) = inspect.getargspec(func)
        (file,line) = inspect.getframeinfo(inspect.currentframe().f_back)[0:2]
        code = ('#line %d "%s"\n'%(line,file))+code
        defaults = [] if defaults==None else defaults
        if len(args) != len(defaults):
            raise Exception("The %s function must have default values for all arguments"%name)
        vardict = dict(list(zip(args,defaults)))
        self.extension.add_function(ext_tools.ext_function(name, code, args, local_dict = vardict))
        return func
        
    def add_function(self,name, code, *varlist):
        for t in varlist:
            assert(type(t) == tuple)
            assert(len(t) == 2)
            assert(type(t[0]) == str)

        args = [n for n, v in varlist]
        vardict = dict(varlist)
        self.extension.add_function(ext_tools.ext_function(name, code, args, local_dict = vardict))
        
    def _import(self,**kw):
        self.extension.compile(location=self.dest_dir,**kw)
        return "from %s import *"%self._module_name
    


class CStruct:
    
    def __init__(self, members):
        self.__members = members
        format = join([ s for (s,_,_) in members],'')
        self.__struct_dict = dict( (v for (_,v,_) in members) )
        self.__translation_dict = dict( ( (k[0],v) for (_,k,v) in members if v != None))
        print(self.__translation_dict)
        self.__struct = struct.Struct(format)
        
        
    def pack(self, **kwargs):
        pass