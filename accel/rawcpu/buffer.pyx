from cbuffer cimport CBuffer
cimport numpy as np

cdef class RawCPUBuffer:
  cdef CBuffer *buf
  shape: list[int]

  def __init__(self, shape): self.shape = shape

  @property
  def shape(self): return tuple(self.shape)

  @staticmethod
  def fromCPU(np.ndarray x):
    ret = RawCPUBuffer([x.shape[i] for i in range(x.ndim)])
    ret.buf = new CBuffer(x.size, x.data)
    return ret

  def toCPU(self):

  def binary_op(x, op, y):
    ret = RawCPUBuffer(x.shape)
    # TODO: write binary op in c++
    return ret


