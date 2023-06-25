# used for compilation only speed tests
import numpy as np
from tinygrad.helpers import dtypes, prod
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBuffer

class RawFakeBuffer(RawBuffer):
  @classmethod
  def fromCPU(cls, x:np.ndarray, **kwargs): return cls(prod(x.shape), dtypes.from_np(x.dtype), **kwargs)
  def toCPU(self): return np.empty(self.size, dtype=self.dtype.np)

class FakeProgram:
  def __init__(self, name:str, prg:str): pass
  def __call__(self, global_size, local_size, *args, wait=False): pass

# NOTE: you have to set a codegen to use this
FakeBuffer = Compiled(RawFakeBuffer, None, FakeProgram)
