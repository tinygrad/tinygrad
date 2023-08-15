from __future__ import annotations
import hashlib
from weakref import WeakValueDictionary
from torch import float32
import numpy as np
import pycuda.autoprimaryctx # type: ignore # noqa: F401
import pycuda.driver as cuda # type: ignore

import triton # type: ignore # noqa: F401
import triton.language as tl  # type: ignore # noqa: F401

from typing import Union, Tuple, Optional, Dict
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, LazyOp, Op, GlobalCounters
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.helpers import prod, DEBUG
from tinygrad.runtime.ops_gpu import CLBuffer


    fn = f"/tmp/{hash}.py"
    exec(codeObject, globals())

class TritonBuffer():
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[TritonBuffer]=None, backing:Optional[np.ndarray]=None, force_create=False):
    super().__init__(shape, hostbuf)
    self._buf : Optional[TritonDeviceAllocation] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    if force_create: self.cuda

  @property
  def cuda(self):
    if self._buf is None:
      self._buf = TritonDeviceAllocation(4*prod(self._base_shape))
      if self._backing is not None: self._buf.copyin(self._backing, stream)
    return self._buf

  @staticmethod
  def fromCPU(x): return TritonBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    buf = self.contiguous()
    buf.cuda
    buf._buf.copyout(data)
    return data

  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer:Optional[TritonBuffer]=None):
    k = TritonASTKernel(ast, output_buffer)
    k.codegen()(*k.bufs)
    return k.ret

class TritonDeviceAllocation(CLBuffer):
  def __init__(self, size):
    super().__init__(size)
    self.dtype = float32

  def data_ptr(self): return int(self._cl)
