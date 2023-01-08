from __future__ import annotations
import functools
from math import ceil, log, log2
from typing import List, NamedTuple, Optional, Dict, Union

import numpy as np
import pycuda.autoinit # type: ignore # pylint: disable=unused-import # noqa: F401
import pycuda.driver as cuda # type: ignore
from pycuda.compiler import SourceModule # type: ignore

from tinygrad.helpers import dedup, prod
from tinygrad.ops import (DEBUG, BinaryOps, ExplicitExecAST, LazyOp, Op, ReduceOps, UnaryOps, get_buffers, get_lazyop_info, get_lazyops)

dev = cuda.Context.get_device()
MAX_THREADS_BLOCK = dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
cuda.Context.set_cache_config(cuda.func_cache.PREFER_L1) # more L1 cache should be better(caches e.g. weights), but I don't really see any speedups
# TODO analyze with profiler https://documen.tician.de/pycuda/driver.html#profiler-control

class ReduceView(NamedTuple):
  name: str
  buf: CUDABuffer
  early: bool
  late: bool

@functools.lru_cache(maxsize=None)
class CUDAProgram:
  def __init__(self, source:str):
    self.kernel = SourceModule(source)
    if DEBUG >= 3: 
      print(source)
      
  def __call__(self, *args, **kwargs):
    # TODO make kernel call async
    return self.kernel.get_function("reduce_kernel")(*args, **kwargs)

class CUDACodeBuilder:
  code_for_op: Dict[Op, str] = {
      UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)",
      UnaryOps.EXP: "exp(A)", UnaryOps.LOG: "log(A)", UnaryOps.SIGN: "sign(A)", UnaryOps.RECIPROCAL: "((float)1.0/A)",
      BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
      BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
      ReduceOps.SUM: "(acc + A)", ReduceOps.MAX: "max(A, acc)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def elementwise_kernel_code(self, ret:CUDABuffer, views: List[ReduceView], ast: LazyOp) -> str:
    kernel = f"""
      __global__ void reduce_kernel(float *dest, {", ".join([f"float *{view.name}_g" for view in views])}) {{
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= {prod(ret.shape)}) return;
        {chr(10).join([self.view_call_code(view.name) for view in views if view.late])}
        dest[gid] = {self.ast_code(ast, views, CUDACodeBuilder.code_for_op)};
      }}"""
    return self.program_code(kernel, views)

  def reduce_kernel_code(self, views: List[ReduceView], reduce_ast: LazyOp, ast: LazyOp, start:bool, end:bool, elements_per_reduce:int, reduce:int, threads_per_block:int) -> str:
    acc_start = CUDACodeBuilder.start_for_op[reduce_ast.op if isinstance(reduce_ast.op, ReduceOps) else ReduceOps.SUM]
    start_elementwise_code = f"""
      {chr(10).join([self.view_call_code(view.name, is_reduce=True) for view in views if view.early])}
      float tmp = {self.ast_code(reduce_ast, views, CUDACodeBuilder.code_for_op, allow_reduce=True).replace("acc", acc_start)};"""
    end_elementwise_code = f"""
      {chr(10).join([self.view_call_code(view.name, gid="blockIdx.y") for view in views if view.late])}
      acc = {self.ast_code(ast, views, CUDACodeBuilder.code_for_op)};"""

    # Is a Parallel Prefix Sum kernel: https://www.youtube.com/watch?v=bpbit8SPMxU https://github.com/CoffeeBeforeArch/cuda_programming/tree/master/03_sum_reduction
    # TODO make this kernel faster
    kernel = f"""
      __global__ void reduce_kernel(float *dest, float *block_reduce, {", ".join([f"float *{view.name}_g" for view in views])}) {{
        __shared__ float sm[{threads_per_block}];

        const int gid = blockIdx.x;
        const int subidx = blockIdx.y * blockDim.x + threadIdx.x;
        if(subidx < {elements_per_reduce}) {{
          {start_elementwise_code if start else f"float tmp = block_reduce[gid * {elements_per_reduce} + subidx];"}
          sm[threadIdx.x] = tmp;
        }}
        else {{
          sm[threadIdx.x] = {acc_start};
        }}
        __syncthreads();

        #pragma unroll
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
          if (threadIdx.x < s) {{
            sm[threadIdx.x] = {CUDACodeBuilder.code_for_op[reduce_ast.op].replace("A", "sm[threadIdx.x]").replace("acc", "sm[threadIdx.x + s]")};
          }}
          __syncthreads();
        }}
        
        if (threadIdx.x == 0) {{
          int destidx = blockIdx.x * gridDim.y + blockIdx.y;
          float acc = sm[0];
          {end_elementwise_code if end else ""}
          dest[destidx] = acc;
        }}
      }}
    """
    return self.program_code(kernel, views, reduce)

  def program_code(self, kernel:str, views: List[ReduceView], reduce:Optional[int]=None) -> str:
    return f"""
      {chr(10).join([self.view_decleration_code(view.buf, view.name, reduce) for view in views if view.early])}
      {chr(10).join([self.view_decleration_code(view.buf, view.name) for view in views if view.late])}
      {"__device__ float sign(float x) { int t = x < 0 ? -1 : 0; return x > 0 ? 1 : t;}" if "sign" in kernel else ""}
      {kernel}
      """

  def view_decleration_code(self, buf: CUDABuffer, name:str, reduce:Optional[int]=None) -> str:
    # TODO inline constants
    expr = buf.st.expr().replace('//', '/')
    return f"""__device__ float get_{name}(float *x, int gid{", int subidx" if reduce is not None else "" }) {{ 
      {"int valid = 1;" if "valid" in expr else ""} {'long' if prod(buf.shape) >= 2**31 else 'int'} idx = gid; 
      {'idx *= '+str(reduce)+'; idx += subidx;' if reduce is not None else ''} {expr};
      {"return valid ? x[idx] : 0.0;" if "valid" in expr else "return x[idx];"}
    }}"""

  def view_call_code(self, name:str, is_reduce: bool=False, gid="gid", subidx="subidx") -> str:
    return f"""float {name} = get_{name}({name}_g, {gid}{f", {subidx}" if is_reduce else ""});"""

  def ast_code(self, x: Union[CUDABuffer, LazyOp], views: List[ReduceView], code_for_op: Dict[Op, str], allow_reduce=False) -> str:
    if isinstance(x, CUDABuffer):
      return next(view.name for view in views if view.buf is x)
    if not allow_reduce and type(x.op) in [ReduceOps]:
      return "acc"
    srcs_code = [self.ast_code(src, views, code_for_op) for src in x.src]
    code = code_for_op[x.op]
    if len(srcs_code) >= 1:
      code = code.replace("A", srcs_code[0])
    if len(srcs_code) >= 2:
      code = code.replace("B", srcs_code[1])
    return code

class CUDABuffer(ExplicitExecAST):
  def __init__(self, shape, hostbuf=None):
    super().__init__(shape, hostbuf)
    self._buf = cuda.mem_alloc(4*prod(self.shape)) if hostbuf is None else hostbuf._buf  

  def __repr__(self): return f"<CUDABuffer with shape {self.shape!r}>"

  @staticmethod
  def fromCPU(x):
    buf = CUDABuffer(x.shape)
    # TODO make this async, use memcpy_htod_async
    cuda.memcpy_htod(buf._buf, x.view(np.ndarray).astype(np.float32).ravel())
    return buf

  def toCPU(self):
    np_arr = np.empty(self.shape, dtype=np.float32)
    cuda.memcpy_dtoh(np_arr, self.contiguous()._buf)
    return np_arr

  @classmethod
  def exec_ast(cls, ast: LazyOp):
    # TODO code with new GPU backend
    bufs = dedup(get_buffers(ast))
    reduceops = dedup([x for x in get_lazyops(ast) if isinstance(x.op, ReduceOps)])
    assert len(reduceops) <= 1, f"max one reduce op in an ast, {reduceops}"
    reduce_ast = reduceops[0] if len(reduceops) == 1 else None
    earlybufs = dedup(get_buffers(reduce_ast)) if reduce_ast is not None else []
    reduce_shape = (earlybufs[0].shape, reduce_ast.arg) if reduce_ast is not None else None
    
    info = get_lazyop_info(ast)
    ret = CUDABuffer(info.shape)

    views = [ReduceView(f"arg_{i}", buf, buf in earlybufs, buf not in earlybufs) for i, buf in enumerate(bufs)]

    # get the input/output shape and the reduce amount
    reduce_shape = (views[0].buf.shape, ret.shape) if reduce_shape is None else reduce_shape
    reduce = prod([s for s,n in zip(*reduce_shape) if n == 1])

    # if it's a partial reduce, assert last non reduced axis is before the first reduced axis
    if reduce > 1 and prod(ret.shape) != 1:
      assert max([i for i,(s,n) in enumerate(zip(*reduce_shape)) if s == n and n != 1]) < min([i for i,(s,n) in enumerate(zip(*reduce_shape)) if s != 1 and n == 1])

    return ret.exec_reduce_ast(views, ast, reduce_ast, reduce)

  def exec_reduce_ast(ret:CUDABuffer, views: List[ReduceView], ast: LazyOp, reduce_ast: Optional[LazyOp]=None, reduce: int=0) -> CUDABuffer:
    cb = CUDACodeBuilder()

    def optimal_threads_per_block(num_kernels:int, only_powers_of_two=False, max_threads_per_block=512):
      return min(min(max(32, (2 ** ceil(log2(num_kernels))) if only_powers_of_two else 32 * ceil(num_kernels / 32)), max_threads_per_block), MAX_THREADS_BLOCK)

    if reduce_ast is None:
      threads_per_block = optimal_threads_per_block(prod(ret.shape), max_threads_per_block=256)
      grid_x = ceil(prod(ret.shape) / threads_per_block)
      elementwise_kernel = CUDAProgram(cb.elementwise_kernel_code(ret, views, ast))
      elementwise_kernel(ret._buf, *[view.buf._buf for view in views], block=(threads_per_block, 1, 1), grid=(grid_x, 1))
    else:
      num_kernel_count = reduce
      # only 2**n threads_per_block are supported when using this type of sum reduction
      threads_per_block = optimal_threads_per_block(num_kernel_count, only_powers_of_two=True,  max_threads_per_block=512)

      reduce_iterations = ceil(log(num_kernel_count, threads_per_block))
      last_buf = cuda.mem_alloc(4) # Will never be used, only a placeholder for the first call
      for i in range(reduce_iterations):
        first, last = i == 0, i == reduce_iterations - 1
        x_block_count = prod(ret.shape)
        y_block_count = ceil(num_kernel_count / threads_per_block)
        # TODO remove grid y dimension, max size of grid y is 2^16, max size of grid x is 2^32
        assert x_block_count < 2**31, f"elements in reduce result must be under 2**32, {x_block_count} isn't"
        assert y_block_count < 2**16, f"reduce must be under 2**16, {reduce} isn't"
        
        ret_buf = ret._buf if last else cuda.mem_alloc(4 * x_block_count * y_block_count)

        reduce_kernel = CUDAProgram(cb.reduce_kernel_code(views, reduce_ast, ast, first, last, num_kernel_count, reduce, threads_per_block))
        reduce_kernel(ret_buf, last_buf, *[view.buf._buf for view in views], block=(threads_per_block, 1, 1), grid=(x_block_count, y_block_count))
        last_buf = ret_buf
        num_kernel_count = ceil(num_kernel_count / threads_per_block)

    return ret
