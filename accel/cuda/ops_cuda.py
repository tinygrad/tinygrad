from typing import Dict

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from tinygrad.helpers import dedup, prod
from tinygrad.ops import (DEBUG, BinaryOps, ExplicitExecAST, LazyOp, Op, ProcessingOps, ReduceOps, UnaryOps, get_buffers, get_lazyop_info, get_lazyops)

class CUDABuffer(ExplicitExecAST):
  code_for_op: Dict[Op, str] = {
      BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
      BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
  }
  start_for_op = {}

  def __init__(self, shape, hostbuf=None):
    super().__init__(shape, hostbuf)
    self._buf = cuda.mem_alloc(4*prod(self.shape))

    if hostbuf is not None: 
      cuda.memcpy_htod(self._buf, hostbuf)

  def __repr__(self): return f"<CUDABuffer with shape {self.shape!r}>"

  @staticmethod
  def fromCPU(x):
    return CUDABuffer(x.shape, hostbuf=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self):
    np_arr = np.empty(self.shape, dtype=np.float32)
    cuda.memcpy_dtoh(np_arr, self._buf)
    return np_arr

  @classmethod
  def exec_ast(cls, ast: LazyOp):
    # copied from llvm
    bufs = dedup(get_buffers(ast))
    reduceops = dedup([x for x in get_lazyops(ast) if isinstance(x.op, ReduceOps) or isinstance(x.op, ProcessingOps)])
    assert len(reduceops) <= 1, f"max one reduce op in an ast, {reduceops}"
    earlybufs = dedup(get_buffers(reduceops[0])) if len(reduceops) > 0 else []
    reduce_shape = (earlybufs[0].shape, reduceops[0].arg) if len(reduceops) > 0 and isinstance(reduceops[0].op, ReduceOps) else None
    info = get_lazyop_info(ast)
    ret = cls(info.shape)

    mod = SourceModule(f"""
    __global__ void binary_kernel(float *dest, float *a, float *b)
    {{
      const int i = threadIdx.x;
      dest[i] = {cls.code_for_op[ast.op].replace('A', 'a[i]').replace('B', 'b[i]')};
    }}
    """)

    binary_kernel = mod.get_function("binary_kernel")

    binary_kernel(ret._buf, bufs[0]._buf, bufs[1]._buf, block=(400, 1, 1), grid=(1, 1))

    return ret
