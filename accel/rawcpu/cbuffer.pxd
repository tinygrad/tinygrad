# distutils: language = c++
# distutils: sources = cbuffer.h

cdef extern from "cbuffer.h":
  cdef cppclass CBuffer:
    CBuffer(int size)
    CBuffer(int size, void *dat)
    void add(CBuffer *a, CBuffer *b)
    float *buf

