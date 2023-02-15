from __future__ import annotations
import hashlib
from weakref import WeakValueDictionary
from torch import float32
import numpy as np
import pycuda.autoprimaryctx # type: ignore # noqa: F401
import pycuda.driver as cuda # type: ignore # noqa: F401

import triton # type: ignore # noqa: F401
import triton.language as tl  # type: ignore # noqa: F401
from triton.compiler import compile as triton_compile

from typing import Union, Tuple, Optional, Dict
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, LazyOp, Op, ExplicitExecAST, DEBUG, GlobalCounters
from tinygrad.shape import ShapeTracker
from tinygrad.helpers import prod
from tinygrad.runtime.cuda import CLBuffer
from tinygrad.ast import ASTKernel

class TritonASTKernel(ASTKernel):
  code_for_op : Dict[Op, str] = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "tl.maximum(A, 0.0)", UnaryOps.GT0: "tl.where(A>0,1,0)",
    UnaryOps.EXP: "tl.exp(A)", UnaryOps.LOG: "tl.log(A)", UnaryOps.RECIPROCAL: "(1.0/A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)",
    #BinaryOps.MUL: "A*B",
    BinaryOps.MUL: "tl.sum(A*B, axis=2)",
    BinaryOps.DIV: "(A/B)", BinaryOps.POW: "tl.exp(tl.log(A)*B)", BinaryOps.CMPEQ: "(A==B)",
    ReduceOps.SUM: "A += B", ReduceOps.MAX: "A = tl.maximum(A,B)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "float('-inf')"}

  def ast_parse(self, x:Union[TritonBuffer, LazyOp], acc:str, do_reduce=False) -> str:
    if not isinstance(x, LazyOp):
      # this is a load
      buf_index = self.bufs.index(x)
      if buf_index not in self.loaded:
        idx, valid = self.sts[buf_index].expr_idxs()
        if valid.min == 1:
          self.kernel.append(self.kernel_prefix + f"  val{buf_index} = tl.load(data{buf_index} + {idx.render()}{''.join(self.ranges(buf_index))})")
        else:
          valid_expr = valid.render().replace("&&", "*1*")
          self.kernel.append(self.kernel_prefix + f"  val{buf_index} = tl.where({valid_expr}, tl.load(data{buf_index} + {idx.render()}, mask={valid_expr}), 0.0)")
        self.loaded.add(buf_index)
      return f"val{buf_index}"
    if isinstance(x.op, ReduceOps) and not do_reduce: return acc

    values = ([acc] if isinstance(x.op, ReduceOps) else []) + [self.ast_parse(v, acc, do_reduce) for v in x.src]

    code = TritonASTKernel.code_for_op[x.op]  # TODO: replace this with a function
    code = code.replace("A", values[0])
    if len(values) == 2: code = code.replace("B", values[1])
    return code

  def ranges(self, buf_index):
    assert len(self.bufs[buf_index].st.views) == 1
    ranges = []
    al = len(self.buftokens[buf_index].axis)
    for i, (s,a,r) in enumerate(self.buftokens[buf_index].axis):
      idxxx = ["None"] * al
      idxxx[i] = ":"
      if a != 0: # or True:
        ranges.append(f" + (tl.arange(0, {s})[{','.join(idxxx)}] * {a})")
    return ranges

  func_cache: WeakValueDictionary = WeakValueDictionary()
  def codegen(self):
    if self.key in self.func_cache: return self.func_cache[self.key]

    self.process()
    self.kernel_prefix = ""
    self.loaded = set()
    self.kernel = ["@triton.jit"]
    self.kernel.append("def fxn("+','.join(f"data{i}" for i in range(len(self.bufs)))+"):")

    self.output_shape = list(self.sts[0].shape[:self.first_reduce])

    # copied from ops_gpu
    # TODO CUDA only supports a grid of (2^31-1, 65535, 65535), that results in invalid kernel launches for some shapes, so flattern the grid for now.
    MAX_OUTPUT_SHAPE = 3
    self.kernel += [f"  idx{len(self.output_shape)-1-i} = tl.program_id({i})" for i in range(min(MAX_OUTPUT_SHAPE, len(self.output_shape)))]
    if len(self.output_shape) > MAX_OUTPUT_SHAPE:
      final_dimension = len(self.output_shape)-MAX_OUTPUT_SHAPE
      for i in range(final_dimension-1, -1, -1):
        self.kernel += [f"  idx{i} = idx{final_dimension} % {self.output_shape[i]}", f"  idx{final_dimension} = idx{final_dimension} // {self.output_shape[i]}"]
      self.output_shape = [prod(self.output_shape[0:final_dimension+1])] + list(self.output_shape[final_dimension+1:])
      if DEBUG >= 3: print(f"replaced output shape with {self.output_shape}")
    elif len(self.output_shape) == 0: self.output_shape = [1]

    if self.reduceop:
      full_shape = [st.shape for st in self.sts if st.shape != self.sts[0].shape]
      full_shape = self.sts[0].shape if len(full_shape) == 0 else full_shape[0]
      if len(self.buftokens[0].axis):
        self.kernel += ["  acc = tl.zeros(("+','.join([str(x[0]) for x in self.buftokens[0].axis])+",), dtype=tl.float32)"]
      else:
        self.kernel += [f"  acc = {TritonASTKernel.start_for_op[self.reduceop.op]}"]
      self.kernel += [("  "*(i-self.first_reduce)+f"  for idx{i} in range(0, {full_shape[i]}):") for i in range(self.first_reduce, self.shape_len)]
      self.kernel_prefix =  "  "*(self.shape_len - self.first_reduce)
      self.kernel.append("  "+self.kernel_prefix+self.ast_parse(self.reduceop, "acc", True))
      self.kernel_prefix =  ""

    if DEBUG >= 3:
      print("output shape", self.output_shape)
      self.printbufs("new:")

    code = self.ast_parse(self.ast, "acc")

    # store
    idx, valid = self.sts[0].expr_idxs()
    self.kernel.append(f"  tl.store(data0 + {idx.render()}{''.join(self.ranges(0))}, {code})")

    # Torch inductor seems to write out files too!
    hash = hashlib.md5(self.key.encode('utf-8')).hexdigest()
    fn = f"/tmp/{hash}.py"
    kernel = '\n'.join(self.kernel)

    replace_kernel = """
@triton.jit
def fxn(data0,data1,data2):
  idx1 = tl.program_id(0)
  idx0 = tl.program_id(1)
  acc = tl.zeros((64,64,), dtype=tl.float32)
  for idx2 in range(0, 48):
    #val1 = tl.load(data1 + ((idx0*49152)+(idx2*32)) + (tl.arange(0, 64)[:,None] * 768) + (tl.arange(0, 32)[None,:] * 1))
    #val2 = tl.load(data2 + ((idx1*64)+(idx2*24576)) + (tl.arange(0, 64)[None,:] * 1) + (tl.arange(0, 32)[:,None] * 768))
    val1 = tl.load(data1 + ((idx0*49152)+(idx2*16)) + (tl.arange(0, 64)[:,None] * 768) + (tl.arange(0, 16)[None,:] * 1))                                             
    val2 = tl.load(data2 + ((idx1*64)+(idx2*12288)) + (tl.arange(0, 64)[None,:] * 1) + (tl.arange(0, 16)[:, None] * 768)) 
    acc += tl.dot(val1, val2, allow_tf32=False)
  tl.store(data0 + ((idx0*49152)+(idx1*64)) + (tl.arange(0, 64)[:,None] * 768) + (tl.arange(0, 64)[None,:] * 1), acc)
"""
    if 'tl.zeros((64,64,)' in kernel: kernel = replace_kernel

    replace_kernel = """
@triton.jit
def fxn(data0,data1,data2):
  idx1 = tl.program_id(0)
  idx0 = tl.program_id(1)
  acc = tl.zeros((128,64,), dtype=tl.float32)
  for idx2 in range(0, 24):
    val1 = tl.load(data1 + ((idx0*98304)+(idx2*32)) + (tl.arange(0, 128)[:,None] * 768) + (tl.arange(0, 32)[None,:] * 1))
    val2 = tl.load(data2 + ((idx1*64)+(idx2*24576)) + (tl.arange(0, 64)[None,:] * 1) + (tl.arange(0, 32)[:,None] * 768))
    acc += tl.dot(val1, val2, allow_tf32=False)
  tl.store(data0 + ((idx0*98304)+(idx1*64)) + (tl.arange(0, 128)[:,None] * 768) + (tl.arange(0, 64)[None,:] * 1), acc)
"""
    if 'tl.zeros((128,64,)' in kernel: kernel = replace_kernel
    if DEBUG >= 4: print(kernel)
    with open(fn, "w") as f: f.write(kernel)
    codeObject = compile(kernel, fn, "exec")
    exec(codeObject, globals())
    program_jit = globals()['fxn']
    config = program_jit._get_config(*[x.cl for x in self.bufs])
    # num_warps=8 is faster
    compiled = triton_compile(program_jit, configs=(config,), signature={i:'*fp32' for i in range(len(self.bufs))}, device=0, num_stages=1, num_warps=8)
    from tinygrad.runtime.cuda import CLProgram
    real_name = compiled.asm['ptx'].split(".visible .entry ")[1].split("(")[0]
    self.fxn = CLProgram(real_name, compiled.asm['ptx'], binary=True, shared=compiled.shared)

    if DEBUG >= 4:
      print(f"num_warps: {compiled.num_warps} shared: 0x{compiled.shared:x}")

    if DEBUG >= 5:
      #print(compiled.asm['ttir'])
      #print(compiled.asm['llir'])
      print(compiled.asm['ptx'])

    mem_estimate = sum(prod(x._base_shape) for x in self.bufs)
    def runner(*bufs):
      #compiled[tuple(self.output_shape[::-1] + [1]*(3-len(self.output_shape)))](*[x.cl for x in bufs], stream=0)
      self.fxn.prg(*[x.cl._cl for x in bufs], grid=tuple(self.output_shape[::-1] + [1]*(3-len(self.output_shape))), block=(32*compiled.num_warps, 1, 1), shared=compiled.shared)
      GlobalCounters.log_kernel(self.info.flops, mem_estimate)
    self.func_cache[self.key] = runner
    return runner

class TritonBuffer(ExplicitExecAST):
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[TritonBuffer]=None, backing:Optional[np.ndarray]=None, force_create=False):
    super().__init__(shape, hostbuf)
    self._buf : Optional[TritonDeviceAllocation] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    if force_create: self.cl

  @property
  def cl(self):
    if self._buf is None:
      self._buf = TritonDeviceAllocation(4*prod(self._base_shape))
      if self._backing is not None: self._buf.copyin(self._backing)
    return self._buf

  @staticmethod
  def fromCPU(x): return TritonBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    buf = self.contiguous()
    buf.cl
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
