from __future__ import annotations
import os, functools
from enum import Enum
import numpy as np
import pyopencl as cl  # type: ignore
from collections import defaultdict
from functools import partial
from typing import List, Tuple, Optional, Dict, Union, Set, Any
from tinygrad.helpers import prod, ConvArgs, dedup
from tinygrad.ops import ASTKernel
from tinygrad.ops import DEBUG, ProcessingOps, UnaryOps, BinaryOps, ReduceOps, LazyOp, get_buffers, get_lazyops, Op, get_lazyop_info, ExplicitExecAST, GlobalCounters
from tinygrad.shapetracker import ShapeTracker

CLCACHE = int(os.getenv("CLCACHE", "1"))
class CLBuffer:
  def __init__(self, size):
    if len(CL.BUFFER_CACHE[size]) > 0:
      self.cl = CL.BUFFER_CACHE[size].pop()
    else:
      # TODO: on GPU OOM, clear the cache
      self.cl = cl.Buffer(CL().cl_ctx, cl.mem_flags.READ_WRITE, size)
      CL.mem_used += self.cl.size

  def __del__(self):
    if CLCACHE:
      CL.BUFFER_CACHE[self.cl.size].append(self.cl)
    else:
      CL.mem_used -= self.cl.size

class CL:
  CACHE, kernel_count, mem_used, time_sum, ops_sum = None, -1, 0, 0.0, 0.0
  BUFFER_CACHE : Dict[int, List[cl.Buffer]] = defaultdict(list)
  cl_ctx : Optional[cl.Context] = None
  cl_queue : Optional[cl.CommandQueue] = None
  def __init__(self):
    if CL.cl_queue is not None:  # already initted
      return
    devices = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0:  # settle for CPU
      devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], [])
    CL.cl_ctx = cl.Context(devices=[devices[int(os.getenv("CL_DEVICE", "0"))]])
    if len(devices) > 1 or DEBUG >= 1:
      print(f"using {CL.cl_ctx.devices}")
    CL.cl_queue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # this is an in-order command queue

  @staticmethod
  def enqueue_copy(a, b, is_blocking=False):
    if CL.CACHE is not None:
      assert False, "can't copy while caching"
    if DEBUG >= 1:
      print(f"**CL**        copy in {b.shape}" if isinstance(b, np.ndarray) else f"**CL**        copy OUT {a.shape}")
    cl.enqueue_copy(CL().cl_queue, a, b, is_blocking=is_blocking)

@functools.lru_cache(maxsize=None)
class CLProgram:
  kernel_cnt = 0
  def __init__(self, name:str, prg:str, options:Tuple[str, ...]=tuple(), argdtypes=None, rename=True, binary=False):
    self.name, self.prg, self.options, self.argdtypes = f"{name}_{CLProgram.kernel_cnt}" if rename else name, prg.replace(f"{name}(", f"{name}_{CLProgram.kernel_cnt}(") if rename else prg, options, argdtypes
    self.clprogram = cl.Program(CL().cl_ctx, CL().cl_ctx.devices, [self.prg]) if binary else cl.Program(CL().cl_ctx, self.prg)  # type: ignore
    self.clprg = self.clprogram.build(options=list(self.options)).__getattr__(self.name)
    if self.argdtypes is not None:
      self.clprg.set_scalar_arg_dtypes(self.argdtypes)
    CLProgram.kernel_cnt += 1
  def __call__(self, *args, op_estimate=0):
    CL.kernel_count += 1
    if CL.CACHE is not None:
      CL.CACHE.append((self, args))
    else:
      e = self.clprg(CL().cl_queue, *args)
    if DEBUG >= 4:
      print(self.prg)
    if DEBUG >= 2:
      CL.cl_queue.finish()
    if DEBUG >= 1:
      CL.time_sum += 0 if DEBUG <= 1 or CL.CACHE is not None else (e.profile.end - e.profile.start)
      CL.ops_sum += op_estimate
      print(f"**CL** {CL.kernel_count:6d} {self.name:20s} args {len(args[2:]):5d}  kernels {str(args[0]):18s} {str(args[1]):12s} OPs {op_estimate/1e6:7.1f}M/{CL.ops_sum/1e9:7.2f}G  mem {CL.mem_used/1e9:5.2f} GB " +
            ("" if DEBUG <= 1 or CL.CACHE is not None else f"tm {(e.profile.end - e.profile.start)/1e3:9.2f}us/{CL.time_sum/1e6:9.2f}ms ({op_estimate/(e.profile.end - e.profile.start):8.2f} GFLOPS)"))
    GlobalCounters.global_ops += op_estimate
    GlobalCounters.global_mem += sum([x.size//4 for x in args[2:] if isinstance(x, cl.Buffer)])

# **** end CL wrappers ****

def ast_kernel_codegen(cls, ast:LazyOp, k:ASTKernel):
  k.process()
  if DEBUG >= 2:
    print("old:", k.shapes)
    print("old:", k.strides)
  
  first_reduce = k.first_reduce
  last_reduce = len(k.shapes[0])
  reduce_dim = 1

  if len(k.shapes[0]) == 8:
    # channels
    """
    k.reshape_and_permute(
      lambda s: (s[0], s[1], s[2], 4, s[4], s[5], s[6], s[7]),
      (0,1,2,4,5,6,7,3))
    first_reduce -= 1
    last_reduce -= 1
    """

    reduce_dim = 4
    last_reduce -= 1

    """
    # pad out ox. this needs to be handled on the store
    for s in k.shapes:
      s[1] = (s[1]+3)//4 * 4
    # s[4:8] is reduce axis, 
    k.reshape_and_permute(
      lambda s: (s[0], s[1]//4, 4, s[2], 4, s[4], s[5], s[6], s[7]),
      (0,1,2,3,4,5,6,7,8))
      #(0,1,3,5,6,7, 8, 2,4))
    """


  if DEBUG >= 2:
    print("new:", k.shapes)
    print("new:", k.strides)

  """
  CACHE_DIM = 32
  if len(k.shapes[0]) == 2:
    # cache tiling, makes permute fast
    k.reshape_and_permute(
      lambda shape: (shape[0]//CACHE_DIM, CACHE_DIM, shape[1]//CACHE_DIM, CACHE_DIM),
      (0,2,1,3))
  """

  # split for reduce
  """
  if len(k.shapes[0]) == 1 and k.reduceop:
    DIM = 2048
    k.reshape_and_permute(
      lambda shape: (DIM, shape[0]//DIM) if shape != [1] else [1,1],
      (0,1)
    )
    k.shapes[0] = (DIM,1)
    k.strides[0] = (1,1)
    k.first_reduce = 1
    print(k.shapes, k.strides)
  """

  output_shape = k.shapes[0] if not k.reduceop else k.shapes[0][:k.first_reduce]

  kernel = ["__kernel void exec(",] + [', '.join(f'__global float* data{i}' for i in range(len(k.bufs)))] + [") {\n"]
  kernel += [f"int idx{i} = get_global_id({min(3, len(output_shape))-1-i});\n" for i in range(min(3, len(output_shape)))]
  if len(output_shape) > 3:
    # compact all the dimensions into the final one
    for i in range(len(output_shape)-1, 2, -1):
      kernel += [f"int idx{i} = idx2 % {output_shape[i]};", f"idx2 = idx2 / {output_shape[i]};\n"]
    output_shape = list(output_shape[0:2]) + [prod(output_shape[2:])]

  Types = Enum("Types", ["FLOAT", "FLOAT4"])

  @functools.lru_cache(None)   # without this cache it'll generate the index twice
  def idx_deref(buf_index) -> Tuple[str, Types]:
    div = 1
    if reduce_dim == 4 and k.bufs[buf_index] in k.earlybufs:
      div = 4
    st = k.bufs[buf_index].st
    idx_pieces = [str(st.offset)] + [(f"idx{i}*{st//div}" if st//div != 1 else f"idx{i}") for i,(sh,st) in enumerate(zip(k.shapes[buf_index][0:last_reduce], k.strides[buf_index][0:last_reduce])) if sh != 1 and st != 0]
    if st.needs_valid(): kernel.append(f"bool bufvalid{buf_index} = true;")
    kernel.append(f"int bufi{buf_index} = " + '('+' + '.join(idx_pieces)+');\n')
    if len(st.views) > 1:
      extra_idx = ';'.join([v.expr for v in st.views[0:-1][::-1] if v.expr not in ['', 'idx=idx', 'valid=valid']])
      kernel.append(extra_idx.replace("//", "/").replace("idx", f"bufi{buf_index}").replace("valid", f"bufvalid{buf_index}") + ";\n")
    if reduce_dim == 4 and k.bufs[buf_index] in k.earlybufs:
      assert not st.needs_valid()
      return f"vload4(bufi{buf_index}, data{buf_index})", Types.FLOAT4
    else:
      return f"(bufvalid{buf_index} ? data{buf_index}[bufi{buf_index}] : 0.0)" if st.needs_valid() else f"data{buf_index}[bufi{buf_index}]", Types.FLOAT

  def ast_parse(x, reduce=False) -> Tuple[str, Types]:
    if not isinstance(x, LazyOp):
      buf_index = k.bufs.index(x)
      return idx_deref(buf_index)
    if isinstance(x.op, ReduceOps) and not reduce:
      return "acc", Types.FLOAT
    values = [ast_parse(v) for v in x.src]
    code = GPUBuffer.code_for_op[x.op]
    if isinstance(x.op, ReduceOps) and values[0][1] != Types.FLOAT:
      # reduce produce float
      kernel.insert(0, "float clsum(float4 x) { return x.x + x.y + x.z + x.w; }\n")
      return code.replace("A", f"clsum({values[0][0]})"), Types.FLOAT

    if len(values) >= 1: code = code.replace("A", values[0][0])
    if len(values) >= 2: code = code.replace("B", values[1][0])
    return code, values[0][1]  # pass back type of first value

  # early ast
  if k.reduceop:
    full_shape = [x for x in k.shapes if x != k.shapes[0]]
    full_shape = k.shapes[0] if len(full_shape) == 0 else full_shape[0]

    kernel.append(f"float acc = {cls.start_for_op[k.reduceop.op]};\n")
    for i in range(first_reduce, last_reduce):
      kernel.append(f"for (int idx{i} = 0; idx{i} < {full_shape[i]}; idx{i}++) {{\n")
    kernel.append("  acc = " + ast_parse(k.reduceop, reduce=True)[0] + ";\n")
    kernel += ["}\n"] * (last_reduce - first_reduce)
  
  # late ast
  kernel.append(f"{idx_deref(0)[0]} = {ast_parse(ast)[0]};\n}}")
  if DEBUG >= 2:
    print(first_reduce, last_reduce, ast)
    print(' '.join(kernel))
  return partial(CLProgram("exec", ' '.join(kernel)), output_shape[::-1] if len(output_shape) > 0 else [1], None, op_estimate=k.info.flops)

class GPUBuffer(ExplicitExecAST):
  code_for_op : Dict[Op, str] = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)",
    UnaryOps.EXP: "exp(A)", UnaryOps.LOG: "log(A)", UnaryOps.SIGN: "sign(A)", UnaryOps.RECIPROCAL: "((float)1.0/A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
    BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
    ReduceOps.SUM: "(acc + A)", ReduceOps.MAX: "max(A, acc)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[GPUBuffer]=None, backing:Optional[np.ndarray]=None):
    super().__init__(shape, hostbuf)
    self._buf : Optional[CLBuffer] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    # early copy in for large buffers
    if self._backing is not None and self._backing.shape != (1,):
      self.cl
  
  @property
  def cl(self):
    if self._buf is None:
      self._buf = CLBuffer(4*prod(self._base_shape))
    if self._backing is not None:
      CL.enqueue_copy(self._buf.cl, self._backing, is_blocking=False)
      self._backing = None
    return self._buf.cl

  def __repr__(self): return f"<GPUBuffer {str(self.st)}>"

  @staticmethod
  def fromCPU(x): return GPUBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    CL.enqueue_copy(data, self.contiguous().cl, is_blocking=True)
    return data

  func_cache : Dict[str, Any] = {}
  @classmethod
  def exec_ast(cls, ast:LazyOp) -> GPUBuffer:
    k = ASTKernel(ast)
    if k.key not in GPUBuffer.func_cache: GPUBuffer.func_cache[k.key] = ast_kernel_codegen(cls, ast, k)
    GPUBuffer.func_cache[k.key](*[x.cl for x in k.bufs])
    return k.ret
