# distutils: language = c++
# distutils: sources = cbuffer.h

cdef extern from "cbuffer.h":
  cdef cppclass CBuffer:
    CBuffer(int size)
    void copyin(void *dat)
    void add(CBuffer *a, CBuffer *b)
    void mul(CBuffer *a, CBuffer *b)
    float *buf
    int size

