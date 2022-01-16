# pip3 install pycuda
import pycuda.driver as cuda
import numpy as np

class CudaBuffer:
  def __init__(self, shape, hostbuf=None):
    import pycuda.autoinit

    # TODO: these are generic
    self.shape = shape
    self.sz = int(np.prod(shape)*4)

    self.buf = cuda.mem_alloc(self.sz)
    if hostbuf is not None:
      cuda.memcpy_htod(self.buf, hostbuf)

  # TODO: this is generic
  @staticmethod
  def fromCPU(data):
    return CudaBuffer(data.shape, data)

  def toCPU(self):
    ret = numpy.empty(self.shape)
    cuda.memcpy_dtoh(ret, self.buf)
    return ret


