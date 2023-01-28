from __future__ import annotations
import torch
import hashlib
import numpy as np

from typing import Union, Tuple, Optional
from tinygrad.ops import ExplicitExecAST, LazyOp
from tinygrad.shape import ShapeTracker
from tinygrad.helpers import prod
from tinygrad.ast import ASTKernel

class TritonASTKernel(ASTKernel):
  def codegen(self):
    self.process()
    self.kernel = ["import triton", "import triton.language as tl", "@triton.jit"]
    self.kernel.append("def fxn("+','.join(f"data{i}" for i in range(len(self.bufs)))+"):")

    self.kernel.append("  idx = tl.program_id(0)")
    self.kernel.append("  x = tl.load(data1 + idx)")
    self.kernel.append("  y = tl.load(data2 + idx)")
    self.kernel.append("  tl.store(data0 + idx, x+y)")

    # Torch inductor seems to write out files too!
    hash = hashlib.md5(self.key.encode('utf-8')).hexdigest()
    fn = f"/tmp/{hash}.py"
    kernel = '\n'.join(self.kernel)
    print(kernel)
    with open(fn, "w") as f: f.write(kernel)
    codeObject = compile(kernel, fn, "exec")
    exec(codeObject, globals())
    program = globals()['fxn']

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
