from __future__ import annotations
from typing import List, NamedTuple, Optional, Dict, Union

import numpy as np
import pycuda.autoinit # pylint: disable=unused-import
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from tinygrad.helpers import dedup, prod
from tinygrad.ops import (DEBUG, BinaryOps, ExplicitExecAST, LazyOp, Op, ReduceOps, UnaryOps, get_buffers, get_lazyop_info, get_lazyops)

class ReduceView(NamedTuple):
  name: str
  buf: CUDABuffer
  early: bool
  late: bool


class ExplicitReduceExecAST(ExplicitExecAST):
  @classmethod
  def exec_ast(cls, ast: LazyOp):
    bufs = dedup(get_buffers(ast))
    reduceops = dedup([x for x in get_lazyops(ast) if isinstance(x.op, ReduceOps)])
    assert len(reduceops) <= 1, f"max one reduce op in an ast, {reduceops}"
    reduce_ast = reduceops[0] if reduceops else None
    earlybufs = dedup(get_buffers(reduce_ast)) if reduce_ast is not None else []
    reduce_shape = (earlybufs[0].shape, reduce_ast.arg) if reduce_ast is not None else None
    
    info = get_lazyop_info(ast)
    ret = CUDABuffer(info.shape)

    views = [ReduceView(f"arg_{i}", buf, buf in earlybufs, buf not in earlybufs) for i, buf in enumerate(bufs)]

    # get the input/output shape and the reduce amount
    reduce_shape = (views[0].buf.shape, ret.shape) if reduce_shape is None else reduce_shape
    red = prod([s for s,n in zip(*reduce_shape) if n == 1])

    # if it's a partial reduce, assert last non reduced axis is before the first reduced axis
    if red > 1 and prod(ret.shape) != 1:
      assert max([i for i,(s,n) in enumerate(zip(*reduce_shape)) if s == n and n != 1]) < min([i for i,(s,n) in enumerate(zip(*reduce_shape)) if s != 1 and n == 1])

    return ret.exec_reduce_ast(views, ast, reduce_ast, red)

  def exec_reduce_ast(ret, views: List[ReduceView], ast: LazyOp, reduce_ast: Optional[LazyOp]=None, red: float=0): 
    raise NotImplementedError("must be implemented")


class CUDABuffer(ExplicitReduceExecAST):
  threads_per_block = 32 # TODO why are there illegal memory accesses when the thread size is over 32 
  code_for_op: Dict[Op, str] = {
      UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)",
      UnaryOps.EXP: "exp(A)", UnaryOps.LOG: "log(A)", UnaryOps.SIGN: "sign(A)", UnaryOps.RECIPROCAL: "((float)1.0/A)",
      BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
      BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
      ReduceOps.SUM: "(acc + A)", ReduceOps.MAX: "max(A, acc)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def __init__(self, shape, hostbuf=None):
    super().__init__(shape, hostbuf)
    self._buf = cuda.mem_alloc(4*prod(self.shape)) if hostbuf is None else hostbuf._buf  

  def __repr__(self): return f"<CUDABuffer with shape {self.shape!r}>"

  @staticmethod
  def fromCPU(x):
    buf = CUDABuffer(x.shape)
    cuda.memcpy_htod(buf._buf, x.view(np.ndarray).astype(np.float32).ravel())
    return buf

  def toCPU(self):
    np_arr = np.empty(self.shape, dtype=np.float32)
    cuda.memcpy_dtoh(np_arr, self.contiguous()._buf)
    return np_arr

  def exec_reduce_ast(ret:CUDABuffer, views: List[ReduceView], ast: LazyOp, reduce_ast: Optional[LazyOp]=None, red: float=0) -> CUDABuffer:
    assert red < 2**31, f"reduce must be under 2**31, {red} isn't"

    def view_decleration_code(x, name:str, reduce:Optional[int]=None) -> str:
      expr = x.st.expr().replace('//', '/')
      return f"""__device__ float get_{name}(float *x, int gid{", int subidx" if reduce is not None else "" }) {{ 
        {"int valid = 1;" if "valid" in expr else ""} {'long' if prod(x.shape) >= 2**31 else 'int'} idx = gid; 
        {'idx *= '+str(reduce)+'; idx += subidx;' if reduce is not None else ''} {expr};
        {"return valid ? x[idx] : 0.0;" if "valid" in expr else "return x[idx];"}
      }}"""

    def view_call_code(name:str, is_reduce: bool=False):
      return f"""float {name} = get_{name}({name}_g, gid{", subidx" if is_reduce else ""});"""

    def ast_code(x: Union[CUDABuffer, LazyOp], views: List[ReduceView], code_for_op: Dict[Op, str], allow_reduce=False) -> str:
      if isinstance(x, CUDABuffer):
        return next(view.name for view in views if view.buf is x)
      if not allow_reduce and type(x.op) in [ReduceOps]:
        return "acc"
      srcs_code = [ast_code(src, views, code_for_op) for src in x.src]
      code = code_for_op[x.op]
      if len(srcs_code) >= 1:
        code = code.replace("A", srcs_code[0])
      if len(srcs_code) >= 2:
        code = code.replace("B", srcs_code[1])
      return code

    earlycode = ast_code(reduce_ast, views, CUDABuffer.code_for_op, allow_reduce=True) if reduce_ast is not None else "acc"
    code = ast_code(ast, views, CUDABuffer.code_for_op)

    source = f"""
      {chr(10).join([view_decleration_code(view.buf, view.name, red) for view in views if view.early])}
      {chr(10).join([view_decleration_code(view.buf, view.name) for view in views if view.late])}

      {"__device__ float sign(float x) { int t = x < 0 ? -1 : 0; return x > 0 ? 1 : t;}" if "sign" in code or "sign" in earlycode else ""}

      __global__ void reduce_kernel(float *dest, {", ".join([f"float *{view.name}_g" for view in views])}) {{
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;

        float acc = {CUDABuffer.start_for_op[reduce_ast.op if reduce_ast is not None else ReduceOps.SUM]};
        for (int subidx = 0; subidx < {red}; subidx++) {{
          {chr(10).join([view_call_code(view.name, is_reduce=True) for view in views if view.early])}
          acc = {earlycode};
        }}
        
        {chr(10).join([view_call_code(view.name) for view in views if view.late])}
        dest[gid] = {code};
      }}"""

    num_kernel_count = prod(ret.shape)
    num_blocks = (num_kernel_count + (CUDABuffer.threads_per_block - 1)) // CUDABuffer.threads_per_block

    if DEBUG >= 2:
      print("\n", "\n".join([f"{view.name}: {view.buf.st.expr()}" for view in views]))
      if earlycode != "acc": print(earlycode)
      print(code)
      print("num_kernel_count", num_kernel_count, "num_blocks", num_blocks, "threads_per_block", CUDABuffer.threads_per_block)

    if DEBUG >= 3:
      print(source)

    reduce_kernel = SourceModule(source).get_function("reduce_kernel")
    reduce_kernel(ret._buf, *[view.buf._buf for view in views], block=(CUDABuffer.threads_per_block, 1, 1), grid=(num_blocks, 1))
    
    return ret
