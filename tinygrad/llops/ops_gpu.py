from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Union, ClassVar
from tinygrad.helpers import prod, IMAGE, getenv
from tinygrad.ops import UnaryOps, MovementOps, LazyOp, CompiledAST, GlobalCounters
from tinygrad.shape import ShapeTracker
from tinygrad.compiler.cl import CLASTKernel

# TODO: select runtimes in a smarter way
CUDA,METAL,CLANG = getenv("CUDA", 0), getenv("METAL", 0), getenv("CLANG", 0)
if not CUDA and not METAL and not CLANG:
  from tinygrad.runtime.opencl import CLBuffer, CLImage, CLProgram # NOTE: using CL will not work for the CUDA runtime # noqa: F401
else:
  class CLImage:  # type: ignore
    def __init__(self, shape): raise NotImplementedError("current runtime doesn't support images")
  if CUDA: from tinygrad.runtime.cuda import CLBuffer, CLProgram  # type: ignore
  elif METAL: from tinygrad.runtime.metal import CLBuffer, CLProgram  # type: ignore
  elif CLANG: from tinygrad.runtime.clang import CLBuffer, CLProgram  # type: ignore

KOPT = getenv("KOPT", -1)
PRINT_AST = getenv("PRINT_AST", "0")
TEST_AST = getenv("TEST_AST", 0)

class GPUProgram(CLASTKernel):
  program : ClassVar = staticmethod(CLProgram)

class GPUBuffer(CompiledAST):
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[GPUBuffer]=None, backing:Optional[np.ndarray]=None, force_create=False):
    super().__init__(shape, hostbuf)
    self._buf : Optional[Union[CLImage, CLBuffer]] = hostbuf._buf if hostbuf is not None else None
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    # early copy in for large buffers
    if (self._backing is not None and self._backing.shape != (1,)) or force_create:
      self.cl
  
  # TODO: refactor this to return self._buf and not import pyopencl
  @property
  def cl(self) -> Union[CLBuffer, CLImage]:
    if self._buf is None:
      self._buf = CLImage(self._base_shape) if (len(self._base_shape) == 3 and self._base_shape[2] == 4 and IMAGE >= 2) else CLBuffer(4*prod(self._base_shape))
    assert self._buf is not None
    if self._backing is not None:
      assert GlobalCounters.cache is None, f"can't copy in {self._backing.shape} while caching"
      self._buf.copyin(self._backing)
      self._backing = None
    return self._buf._cl

  # TODO: we don't always need a hostbuf
  def __repr__(self): return f"GPUBuffer(shape={self.st}, hostbuf=GPUBuffer(shape={self._base_shape}" + (f", backing=np.array({self._backing}, dtype=np.float32)))" if self._backing else ", force_create=True))")

  @staticmethod
  def fromCPU(x): return GPUBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self) -> np.ndarray:
    cl_buf = self.contiguous()
    cl_buf.cl   # force buffer creation, happens if it's a backed buffer that hasn't been created yet
    cl_buf = cl_buf if isinstance(cl_buf._buf, CLBuffer) else type(self).exec_ast(LazyOp(op=UnaryOps.NOOP, src=(self.movement_op(MovementOps.RESHAPE, tuple(list(self.shape)+[1])), )))
    assert prod(cl_buf._base_shape) == prod(self.shape), f"shape product mismatch {cl_buf._base_shape} vs {self.shape}"
    data = np.empty(self.shape, dtype=np.float32)
    assert GlobalCounters.cache is None, f"can't copy out {self} while caching"
    cl_buf._buf.copyout(data)
    return data

  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer:Optional[GPUBuffer]=None):
    k = GPUProgram(ast, output_buffer)
    if KOPT > 0:
      from extra.kernel_search import apply_optimization
      apply_optimization(k, ast, max_interventions=KOPT)
    prg = k.codegen(KOPT == -1 or IMAGE == 2)
    if GlobalCounters.cache is not None: GlobalCounters.cache.append((prg, k.bufs))
    prg(*k.bufs)
    if PRINT_AST == "1" or (hasattr(k, "fxn") and PRINT_AST == k.fxn.name):
      print(k.fxn.name)
      k.print()
    if TEST_AST:
      from extra.lib_test_ast import test_ast  # type: ignore
      test_ast(k)
    return k.ret
