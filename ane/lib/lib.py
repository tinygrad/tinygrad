#!/usr/bin/env python3
from ctypes import *
libane = cdll.LoadLibrary("libane.dylib")

libane.ANE_Compile.argtypes = [c_char_p, c_int]
libane.ANE_Compile.restype = c_void_p

libane.ANE_TensorCreate.restype = c_void_p

libane.ANE_TensorData.argtypes = [c_void_p]
libane.ANE_TensorData.restype = c_void_p

#libane.ANE_TensorRun.argtypes = [c_void_p]*3

libane.ANE_Open()

dat = open("../2_compile/model.hwx", "rb").read()
comp = libane.ANE_Compile(create_string_buffer(dat), len(dat))
print("compile", comp)

tin = libane.ANE_TensorCreate(16,16)
tout = libane.ANE_TensorCreate(16,16)

#addr = libane.ANE_TensorData(dat)
#print(dat, addr)




