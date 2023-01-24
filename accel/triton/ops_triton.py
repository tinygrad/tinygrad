from __future__ import annotations
import torch
import numpy as np
import triton
import triton.language as tl

from typing import Union, Tuple, Optional
from tinygrad.ops import ExplicitExecAST
from tinygrad.shape import ShapeTracker
from tinygrad.helpers import prod
from tinygrad.ast import ASTKernel

class TritonASTKernel(ASTKernel):
  def codegen(self):
    self.process()

    @triton.jit
    def program(b0, b1, b2):
      idx = tl.program_id(0)
      x = tl.load(b1 + idx)
      y = tl.load(b2 + idx)
      tl.store(b0 + idx, x+y)

    return lambda *bufs: program[(prod(self.bufs[0].shape),)](*[x.torch for x in bufs])


class TritonBuffer(ExplicitExecAST):
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[TritonBuffer]=None, backing:Optional[np.ndarray]=None, force_create=False):
    super().__init__(shape, hostbuf)
    self._buf : Optional[TritonBuffer] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing

  @property
  def torch(self):
    if self._buf is None:
      if self._backing is not None: self._buf = torch.from_numpy(self._backing).cuda()
      else: self._buf = torch.empty(prod(self._base_shape), dtype=torch.float32, device='cuda')
    return self._buf

  @staticmethod
  def fromCPU(x): return TritonBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())
  def toCPU(x): return x.torch.cpu().numpy().reshape(x.shape)

  @classmethod
  def exec_ast(cls, ast:LazyOp):
    k = TritonASTKernel(ast)
    k.codegen()(*k.bufs)
    return k.ret
