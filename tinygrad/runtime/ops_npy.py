import numpy as np
from tinygrad.device import Compiled, Allocator, MMIOInterface

class NpyAllocator(Allocator['NpyDevice']):
  def _alloc(self, size:int, options=None) -> np.ndarray: return np.empty(size, dtype=np.uint8)
  def _as_mmio(self, src:np.ndarray) -> MMIOInterface: return MMIOInterface(src.ctypes.data, src.nbytes, owner=src)
  def _copyout(self, dest:memoryview, src:np.ndarray): dest[:] = self._as_mmio(src).mv

class NpyDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, NpyAllocator(self), [], None)
