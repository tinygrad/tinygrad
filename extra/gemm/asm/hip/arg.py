# https://github.com/ROCm/aiter/blob/d0c313d78eb04b495f6d126a281fe9e29a8d2d89/csrc/py_itfs_cu/asm_gemm_a16w16.cu#L13
import ctypes

class p2(ctypes.Structure):
  _pack_ = 1
  _fields_ = [("pad", ctypes.c_ubyte * 8)]

class p3(ctypes.Structure):
  _pack_ = 1
  _fields_ = [("pad", ctypes.c_ubyte * 12)]

class KernelArgs(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
      ("ptr_D", ctypes.c_void_p),
      ("_p0", p2),
      ("ptr_C", ctypes.c_void_p),
      ("_p1", p2),
      ("ptr_A", ctypes.c_void_p),
      ("_p2", p2),
      ("ptr_B", ctypes.c_void_p),
      ("_p3", p2),
      ("alpha", ctypes.c_float),
      ("_p4", p3),
      ("beta", ctypes.c_float),
      ("_p5", p3),
      ("stride_D0", ctypes.c_uint32),
      ("_p6", p3),
      ("stride_D1", ctypes.c_uint32),
      ("_p7", p3),
      ("stride_C0", ctypes.c_uint32),
      ("_p8", p3),
      ("stride_C1", ctypes.c_uint32),
      ("_p9", p3),
      ("stride_A0", ctypes.c_uint32),
      ("_p10", p3),
      ("stride_A1", ctypes.c_uint32),
      ("_p11", p3),
      ("stride_B0", ctypes.c_uint32),
      ("_p12", p3),
      ("stride_B1", ctypes.c_uint32),
      ("_p13", p3),
      ("M", ctypes.c_uint32),
      ("_p14", p3),
      ("N", ctypes.c_uint32),
      ("_p15", p3),
      ("K", ctypes.c_uint32),
      ("_p16", p3),
      ("splitk", ctypes.c_uint32),
      ("_p17", p3),
      ("is_out_b16", ctypes.c_uint32),
      ("_p18", p3),
      ("ptr_Bias", ctypes.c_void_p),
      ("_p19", p2),
      ("add_bias", ctypes.c_uint32),
      ("_p20", p3),
      ]
