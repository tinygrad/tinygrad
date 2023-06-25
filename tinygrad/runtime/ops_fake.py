# used for compilation speed tests of CLCodegen
import numpy as np
from tinygrad.helpers import dtypes, prod, DEBUG
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBuffer
from tinygrad.runtime.ops_gpu import CLCodegen

class RawFakeBuffer(RawBuffer):
  @classmethod
  def fromCPU(cls, x:np.ndarray, **kwargs): return cls(prod(x.shape), dtypes.from_np(x.dtype), **kwargs)
  def toCPU(self): return np.empty(self.size, dtype=self.dtype.np)

class FakeProgram:
  def __init__(self, name:str, prg:str):
    if DEBUG >= 3: print(prg)
  def __call__(self, global_size, local_size, *args, wait=False): pass

FakeBuffer = Compiled(RawFakeBuffer, CLCodegen, FakeProgram)
