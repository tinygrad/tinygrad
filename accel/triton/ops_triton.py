from __future__ import annotations
import hashlib
from weakref import WeakValueDictionary
from torch import float32
import numpy as np
# import pycuda.autoinit # type: ignore # pylint: disable=unused-import # noqa: F401
import pycuda.autoprimaryctx # type: ignore # noqa: F401
import pycuda.driver as cuda # type: ignore

import triton # type: ignore # noqa: F401
import triton.language as tl  # type: ignore # noqa: F401

from typing import Union, Tuple, Optional, Dict
from tinygrad.ops import MovementOps, UnaryOps, BinaryOps, ReduceOps, LazyOp, Op, ExplicitExecAST, DEBUG, GlobalCounters
from tinygrad.shape import ShapeTracker
from tinygrad.helpers import prod
from tinygrad.ast import ASTKernel

from tinygrad.shape import View, ZeroView
from tinygrad.shape.symbolic import Variable

stream = cuda.Stream()

class TritonASTKernel(ASTKernel):
  code_for_op : Dict[Op, str] = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "tl.maximum(A, 0.0)", UnaryOps.SIGN: "tl.where(A>0,1,0)",
    UnaryOps.EXP: "tl.exp(A)", UnaryOps.LOG: "tl.log(A)", UnaryOps.RECIPROCAL: "(1.0/A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
    BinaryOps.DIV: "(A/B)", BinaryOps.POW: "tl.exp(tl.log(A)*B)", BinaryOps.CMPEQ: "(A==B)",
    ReduceOps.SUM: "A += B", ReduceOps.MAX: "A = tl.maximum(A,B)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "float('-inf')"}

  # TODO: move to shapetracker
  def compute_buf_index_symbolic(self, st, buf_index, offset=0):
    view = View(self.shapes[buf_index], self.strides[buf_index], self.offsets[buf_index] + offset)
    idx = view.expr_idxs([f"idx{i}" for i in range(self.shape_len)])
    valid = Variable.num(1)
    for v in st.views[0:-1][::-1]:
      if isinstance(v, ZeroView): valid = v.expr_node(valid, idx)
      else: idx = v.expr_node(idx)
    return idx, valid

  def ast_parse(self, x:Union[TritonBuffer, LazyOp], acc:str, do_reduce=False) -> str:
    if not isinstance(x, LazyOp):
      # this is a load
      buf_index = self.bufs.index(x)
      if buf_index not in self.loaded:
        idx, valid = self.compute_buf_index_symbolic(self.bufs[buf_index].st, buf_index)
        valid_expr = str(valid).replace("&&", "*1*")
        self.kernel.append(self.kernel_prefix + f"  val{buf_index} = tl.where({valid_expr}, tl.load(data{buf_index} + {idx}, mask={valid_expr}), 0.0)")
        self.loaded.add(buf_index)
      return f"val{buf_index}"
    if isinstance(x.op, ReduceOps) and not do_reduce: return acc

    values = ([acc] if isinstance(x.op, ReduceOps) else []) + [self.ast_parse(v, acc, do_reduce) for v in x.src]

    code = TritonASTKernel.code_for_op[x.op]  # TODO: replace this with a function
    code = code.replace("A", values[0])
    if len(values) == 2: code = code.replace("B", values[1])
    return code

  func_cache: WeakValueDictionary = WeakValueDictionary()
  def codegen(self):
    if self.key in self.func_cache: return self.func_cache[self.key]

    self.process()
    self.kernel_prefix = ""
    self.loaded = set()
    self.kernel = ["@triton.jit"]
    self.kernel.append("def fxn("+','.join(f"data{i}" for i in range(len(self.bufs)))+"):")

    self.output_shape = list(self.shapes[0][:self.first_reduce])

    # copied from ops_gpu
    # TODO CUDA only supports a grid of (2^31-1, 65535, 65535), that results in invalid kernel launches for some shapes, so flattern the grid for now.
    MAX_OUTPUT_SHAPE = 1
    self.kernel += [f"  idx{len(self.output_shape)-1-i} = tl.program_id({i})" for i in range(min(MAX_OUTPUT_SHAPE, len(self.output_shape)))]
    if len(self.output_shape) > MAX_OUTPUT_SHAPE:
      final_dimension = len(self.output_shape)-MAX_OUTPUT_SHAPE
      for i in range(final_dimension-1, -1, -1):
        self.kernel += [f"  idx{i} = idx{final_dimension} % {self.output_shape[i]}", f"  idx{final_dimension} = idx{final_dimension} // {self.output_shape[i]}"]
      self.output_shape = [prod(self.output_shape[0:final_dimension+1])] + list(self.output_shape[final_dimension+1:])
      if DEBUG >= 3: print(f"replaced output shape with {self.output_shape}")
    elif len(self.output_shape) == 0: self.output_shape = [1]
    
    if self.reduceop:
      full_shape = [x for x in self.shapes if x != self.shapes[0]]
      full_shape = self.shapes[0] if len(full_shape) == 0 else full_shape[0]
      self.kernel += [f"  acc = {TritonASTKernel.start_for_op[self.reduceop.op]}"]
      self.kernel += [("  "*(i-self.first_reduce)+f"  for idx{i} in range(0, {full_shape[i]}):") for i in range(self.first_reduce, self.shape_len)]
      self.kernel_prefix =  "  "*(self.shape_len - self.first_reduce)
      self.kernel.append("  "+self.kernel_prefix+self.ast_parse(self.reduceop, "acc", True))
      self.kernel_prefix =  ""

    code = self.ast_parse(self.ast, "acc")

    # store
    idx, valid = self.compute_buf_index_symbolic(self.bufs[0].st, 0)
    self.kernel.append(f"  tl.store(data0 + {idx}, {code})")

    # Torch inductor seems to write out files too!
    hash = hashlib.md5(self.key.encode('utf-8')).hexdigest()
    fn = f"/tmp/{hash}.py"
    kernel = '\n'.join(self.kernel)
    if DEBUG >= 4: print(kernel)
    with open(fn, "w") as f: f.write(kernel)
    codeObject = compile(kernel, fn, "exec")
    exec(codeObject, globals())
    program = globals()['fxn']

    mem_estimate = sum(prod(x) for x in self.shapes)
    def runner(*bufs):
      GlobalCounters.global_ops += self.info.flops
      GlobalCounters.global_mem += mem_estimate
      return program[tuple(self.output_shape[::-1])](*[TritonWrapper(x.torch) for x in bufs], stream=stream.handle)
    self.func_cache[self.key] = runner
    return runner

class TritonBuffer(ExplicitExecAST):
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[TritonBuffer]=None, backing:Optional[np.ndarray]=None, force_create=False):
    super().__init__(shape, hostbuf)
    if hostbuf is not None and hostbuf._buf is None: hostbuf.torch
    self._buf : Optional[TritonBuffer] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing

  @property
  def torch(self):
    if self._buf is None:
      self._buf = cuda.mem_alloc(4*prod(self._base_shape))
      if self._backing is not None: cuda.memcpy_htod_async(self._buf, self._backing, stream=stream)
    return self._buf

  @staticmethod
  def fromCPU(x): return TritonBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    buf = self.contiguous() if self._buf is not None else self.movement_op(MovementOps.RESHAPE, list(self.shape)+[1]).unary_op(UnaryOps.NOOP)
    # TODO should this be sync?
    cuda.memcpy_dtoh_async(data, buf._buf, stream=stream)
    return data

  @classmethod
  def exec_ast(cls, ast:LazyOp):
    k = TritonASTKernel(ast)
    k.codegen()(*k.bufs)
    return k.ret

class TritonWrapper:
  def __init__(self, ptr):
    self.ptr = ptr
    self.dtype = float32

  def data_ptr(self):
    return int(self.ptr)
