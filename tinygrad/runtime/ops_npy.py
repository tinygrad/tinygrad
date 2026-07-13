import numpy as np
from tinygrad.helpers import flat_mv, mv_address
from tinygrad.device import Compiled, Allocator, MMIOInterface

class NpyAllocator(Allocator['NpyDevice']):
  def _alloc(self, size:int, options=None) -> np.ndarray: return np.empty(size, dtype=np.uint8)
  def _mmio(self, src:np.ndarray) -> MMIOInterface:
    mv = flat_mv(np.require(src, requirements='C').data)
    return MMIOInterface(mv_address(mv), mv.nbytes, owner=src)
  def _as_mmio(self, src) -> MMIOInterface: return self._mmio(src._buf)
  def _copyout(self, dest:memoryview, src:np.ndarray): dest[:] = self._mmio(src)[:]

class NpyDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, NpyAllocator(self), [], None)
