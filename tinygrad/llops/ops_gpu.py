from __future__ import annotations
import os, functools
from enum import Enum
import numpy as np
import pyopencl as cl  # type: ignore
from collections import defaultdict
from functools import partial
from typing import List, Tuple, Optional, Dict, Union, Set, Any
from tinygrad.helpers import prod, ConvArgs, dedup, all_same
from tinygrad.ops import ASTKernel
from tinygrad.ops import DEBUG, ProcessingOps, UnaryOps, BinaryOps, ReduceOps, MovementOps, LazyOp, get_buffers, get_lazyops, Op, get_lazyop_info, ExplicitExecAST, GlobalCounters
from tinygrad.shapetracker import ShapeTracker

FLOAT16 = int(os.getenv("FLOAT16", 0))
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

class CLImage:
  fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.HALF_FLOAT if FLOAT16 else cl.channel_type.FLOAT)

  def __init__(self, shape):
    self.cl = cl.Image(CL.cl_ctx, cl.mem_flags.READ_WRITE, CLImage.fmt, shape=(shape[0], shape[1]))
    CL.mem_used += self.cl.row_pitch * self.cl.height

  def __del__(self):
    if hasattr(self, "cl"):
      CL.mem_used -= self.cl.row_pitch * self.cl.height

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
      assert False, f"can't copy {a} -> {b} while caching"
    if DEBUG >= 1:
      print(f"**CL**        copy in {b.shape}" if isinstance(b, np.ndarray) else f"**CL**        copy OUT {a.shape}")
    cl.enqueue_copy(CL().cl_queue, a, b, is_blocking=is_blocking)

@functools.lru_cache(maxsize=None)
class CLProgram:
  kernel_cnt = defaultdict(int)
  def __init__(self, name:str, prg:str, options:Tuple[str, ...]=tuple(), argdtypes=None, rename=True, binary=False):
    self.name = f"{name}{('_N'+str(CLProgram.kernel_cnt[name])) if CLProgram.kernel_cnt[name] else ''}" if rename else name
    self.prg, self.options, self.argdtypes = prg.replace(f"{name}(", f"{self.name}(") if rename else prg, options, argdtypes
    self.clprogram = cl.Program(CL().cl_ctx, CL().cl_ctx.devices, [self.prg]) if binary else cl.Program(CL().cl_ctx, self.prg)  # type: ignore
    self.clprg = self.clprogram.build(options=list(self.options)).__getattr__(self.name)
    if self.argdtypes is not None:
      self.clprg.set_scalar_arg_dtypes(self.argdtypes)
    CLProgram.kernel_cnt[name] += 1
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
      print(f"**CL** {CL.kernel_count:6d} {self.name:28s} args {len(args[2:]):5d}  kernels {str(args[0]):18s} {str(args[1]):12s} OPs {op_estimate/1e6:7.1f}M/{CL.ops_sum/1e9:7.2f}G  mem {CL.mem_used/1e9:5.2f} GB " +
            ("" if DEBUG <= 1 or CL.CACHE is not None else f"tm {(e.profile.end - e.profile.start)/1e3:9.2f}us/{CL.time_sum/1e6:9.2f}ms ({op_estimate/(e.profile.end - e.profile.start):8.2f} GFLOPS)"))
    GlobalCounters.global_ops += op_estimate
    GlobalCounters.global_mem += sum([x.size//4 for x in args[2:] if isinstance(x, cl.Buffer)])

# **** end CL wrappers ****

def ast_kernel_codegen(cls, ast:LazyOp, k:ASTKernel):
  k.process()
  buftypes = [f"{'read_only' if i > 0 else 'write_only'} image2d_t" if isinstance(x._buf, CLImage) else "__global float *" for i,x in enumerate(k.bufs)]
  #print(buftypes)

  if DEBUG >= 2:
    print("old:", k.shapes)
    print("old:", k.strides)
  if DEBUG >= 3:
    for x in k.bufs:
      print(x.st)
  
  first_reduce = k.first_reduce
  last_reduce = len(k.shapes[0])

  early_loads_are_float4 = False
  late_are_float4 = False

  early_loads_are_non_reduce_float4 = False

  # if there's images in the earlybufs, we have to make an axis the 4 loading one
  # shove the axis to the end and remove it
  any_early_images = any(isinstance(buf._buf, CLImage) for buf in k.earlybufs)
  if any_early_images:
    eb_valids = [True] * len(k.shapes[0])
    for i in range(len(k.bufs)):
      if isinstance(k.bufs[i]._buf, CLImage) and k.bufs[i] in k.earlybufs:
        assert len(k.bufs[i].st.views) == 1, f"images can't have views {k.bufs[i].st}"
        valids = [k.shapes[i][j]%4 == 0 and k.strides[i][j] == 1 for j in range(len(k.shapes[i]))]
        eb_valids = [x and y for x,y in zip(eb_valids, valids)]
    assert any(eb_valids), f"invalid op with images {buftypes} {eb_valids}"
    eb_valid = eb_valids.index(True)

    # no change, we added a dimension
    k.reshape_and_permute(lambda x: list(x[0:eb_valid]) + ([x[eb_valid]//4, 4] if x[eb_valid] > 1 else [1,1]) + list(x[eb_valid+1:]), [i for i in range(k.shape_len+1) if i != eb_valid+1] + [eb_valid+1])
    #last_reduce -= 1

    if eb_valid < first_reduce:
      #assert eb_valid >= first_reduce, f"only support in the reduce for now {eb_valids} and first reduce is {first_reduce}"
      early_loads_are_non_reduce_float4 = True
      late_are_float4 = True
    else:
      early_loads_are_float4 = True
  
  # if there's images in the latebufs, we have to make an axis the 4 storing one. this affects the kernel shape
  any_late_images = any(isinstance(buf._buf, CLImage) for buf in k.bufs if buf not in k.earlybufs)
  if any_late_images and not early_loads_are_non_reduce_float4:
    lb_valids = [True] * len(k.shapes[0])
    for i in range(len(k.bufs)):
      #assert len(k.bufs[i].st.views) == 1 or not isinstance(k.bufs[i]._buf, CLImage)  # images can't have views
      valids = [k.shapes[i][j]%4 == 0 and (k.strides[i][j] == 1 or not isinstance(k.bufs[i]._buf, CLImage) or k.bufs[i] in k.earlybufs) for j in range(len(k.shapes[i]))]
      lb_valids = [x and y for x,y in zip(lb_valids, valids)]
    assert any(lb_valids), f"invalid op with images {buftypes}"
    lb_valid = lb_valids.index(True)
    assert lb_valid < first_reduce, f"can't be in the reduce {lb_valid}"
    k.reshape_and_permute(lambda x: list(x[0:lb_valid]) + [x[lb_valid]//4, 4] + list(x[lb_valid+1:]), [i for i in range(k.shape_len+1) if i != lb_valid+1] + [lb_valid+1])
    # no change, we added a dimension
    #last_reduce -= 1
    #first_reduce -= 1
    late_are_float4 = True
  #print(f"early_loads_are_float4: {early_loads_are_float4} late_are_float4: {late_are_float4} first_reduce: {first_reduce} last_reduce: {last_reduce}")

  if DEBUG >= 2:
    print("new:", k.shapes)
    print("new:", k.strides)

  output_shape = k.shapes[0][:k.first_reduce]

  prekernel = set()
  kernel = ["const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"]
  kernel += [f"int idx{i} = get_global_id({min(3, len(output_shape))-1-i});\n" for i in range(min(3, len(output_shape)))]
  if len(output_shape) > 3:
    # compact all the dimensions into the final one
    for i in range(len(output_shape)-1, 2, -1):
      kernel += [f"int idx{i} = idx2 % {output_shape[i]};", f"idx2 = idx2 / {output_shape[i]};\n"]
    #print(output_shape)
    output_shape = list(output_shape[0:2]) + [prod(output_shape[2:])]

  bufs_to_delete = set()

  Types = Enum("Types", ["FLOAT", "FLOAT4"])
  class Token:
    def __init__(self, tok:str, typ:Types):
      assert isinstance(tok, str)
      self.tok = tok
      self.typ = typ
    def __str__(self): return self.tok
    def __repr__(self): return f"<{self.typ} {self.tok}>"
    

  seen_idx = set()
  def compute_buf_index(st, buf_index, offset):
    key = f"{buf_index}_{offset}"
    # add the index if we don't have it
    if key not in seen_idx:
      idx_pieces = [str(st.offset + offset)] + [(f"idx{i}*{st}" if st != 1 else f"idx{i}") for i,(sh,st) in enumerate(zip(k.shapes[buf_index][0:last_reduce], k.strides[buf_index][0:last_reduce])) if sh != 1 and st != 0]
      if st.needs_valid(): kernel.append(f"bool bufvalid{key} = true;")
      kernel.append(f"int bufi{key} = " + '('+' + '.join(idx_pieces)+');\n')
      if len(st.views) > 1:
        extra_idx = ';'.join([v.expr for v in st.views[0:-1][::-1] if v.expr not in ['', 'idx=idx', 'valid=valid']])
        kernel.append(extra_idx.replace("//", "/").replace("idx", f"bufi{key}").replace("valid", f"bufvalid{key}") + ";\n")
      seen_idx.add(key)
    return key

  def store(buf_index, value, offset=0):
    st = k.bufs[buf_index].st
    if offset > 0: assert len(st.views) == 1
    key = compute_buf_index(st, buf_index, offset)

    if isinstance(k.bufs[buf_index]._buf, CLImage):
      W = k.bufs[buf_index]._base_shape[1]
      assert value.typ == Types.FLOAT4, f"image can only store float4: {value} isn't"
      kernel.append(f"write_imagef(data{buf_index}, (int2)((bufi{key})/{W*4}, ((bufi{key})/4)%{W}), {value.tok});\n")
    else:
      if value.typ == Types.FLOAT4:
        #assert len(st.views) == 1
        kernel.append(f"float4 to_store = {value.tok};\n")
        for i in range(4):
          lkey = compute_buf_index(st, buf_index, offset+i*k.strides[buf_index][-1])
          kernel.append(f"data{buf_index}[bufi{lkey}] = to_store.s{i};\n")
      else:
        kernel.append(f"data{buf_index}[bufi{key}] = {value.tok};\n")

  @functools.lru_cache(None)
  def load(buf_index, offset=0):
    st = k.bufs[buf_index].st
    if offset > 0: assert len(st.views) == 1
    key = compute_buf_index(st, buf_index, offset)

    # constant folding
    constant_fold = None
    if k.bufs[buf_index]._base_shape == (1,) and k.bufs[buf_index]._backing:
      bufs_to_delete.add(buf_index)
      constant_fold = f"({k.bufs[buf_index]._backing[0]})"

    if isinstance(k.bufs[buf_index]._buf, CLImage):
      W = k.bufs[buf_index]._base_shape[1]
      #assert not st.needs_valid()
      ldr = Token(f"read_imagef(data{buf_index}, smp, (int2)((bufi{key})/{W*4}, ((bufi{key})/4)%{W}))", Types.FLOAT4)
    else:
      if late_are_float4 or (early_loads_are_float4 and k.bufs[buf_index] in k.earlybufs):
        #assert len(st.views) == 1, st.views
        if k.strides[buf_index][-1] == 1 and len(st.views) == 1 and not st.needs_valid():
          ldr = Token(f"((__global float4*)data{buf_index})[bufi{key}/4]", Types.FLOAT4)
        else:
          mst = []
          for i in range(4):
            lkey = compute_buf_index(st, buf_index, offset+i*k.strides[buf_index][-1])
            mst.append(f"data{buf_index}[bufi{lkey}]" if not constant_fold else constant_fold)
            if st.needs_valid(): mst[-1] = f"(bufvalid{key} ? {mst[-1]} : 0.0)"
          ldr = Token(f"(float4)({','.join(mst)})", Types.FLOAT4)
      else:
        ldr = f"data{buf_index}[bufi{key}]" if not constant_fold else constant_fold
        ldr = Token(f"(bufvalid{key} ? {ldr} : 0.0)" if st.needs_valid() else ldr, Types.FLOAT)
    kernel.append(f"{'float' if ldr.typ == Types.FLOAT else 'float4'} val{key} = {ldr.tok};\n")
    return Token(f"val{key}", ldr.typ)

  def ast_parse(x, offset=0, reduce=False) -> Token:
    if not isinstance(x, LazyOp):
      buf_index = k.bufs.index(x)
      return load(buf_index, offset=offset*k.strides[buf_index][-1])
    if isinstance(x.op, ReduceOps) and not reduce:
      return Token(f"acc", Types.FLOAT4 if late_are_float4 or early_loads_are_non_reduce_float4 else Types.FLOAT)
    values = [ast_parse(v, offset, reduce) for v in x.src]
    code = GPUBuffer.code_for_op[x.op]  # TODO: replace this with a function
    if isinstance(x.op, ReduceOps) and values[0].typ != Types.FLOAT:
      if early_loads_are_non_reduce_float4:
        return Token(code.replace("A", values[0].tok), Types.FLOAT4)
      else:
        prekernel.add("float clsum(float4 x) { return x.x + x.y + x.z + x.w; }\n")
        return Token(code.replace("A", f"clsum({values[0].tok})").replace("acc", f"acc.s{offset}" if late_are_float4 else "acc"), Types.FLOAT)
    assert all_same([x.typ for x in values]), f"type mismatch in {values}"
    if len(values) >= 1: code = code.replace("A", values[0].tok)
    if len(values) >= 2: code = code.replace("B", values[1].tok)
    return Token(code, values[0].typ)  # pass back type of first value

  # early ast
  if k.reduceop:
    full_shape = [x for x in k.shapes if x != k.shapes[0]]
    full_shape = k.shapes[0] if len(full_shape) == 0 else full_shape[0]

    if late_are_float4 or early_loads_are_non_reduce_float4:
      kernel.append(f"float4 acc = (float4)({cls.start_for_op[k.reduceop.op]}, {cls.start_for_op[k.reduceop.op]}, {cls.start_for_op[k.reduceop.op]}, {cls.start_for_op[k.reduceop.op]});\n")
    else:
      kernel.append(f"float acc = {cls.start_for_op[k.reduceop.op]};\n")
    for i in range(first_reduce, last_reduce):
      kernel.append(f"for (int idx{i} = 0; idx{i} < {full_shape[i]}; idx{i}++) {{\n")
    if late_are_float4 and not early_loads_are_non_reduce_float4:
      future_kernel = []
      for j in range(4):
        future_kernel.append(f"  acc.s{j} = " + ast_parse(k.reduceop, offset=j, reduce=True).tok + ";\n")
      kernel += future_kernel
    else:
      kernel.append("  acc = " + ast_parse(k.reduceop, reduce=True).tok + ";\n")
    kernel += ["}\n"] * (last_reduce - first_reduce)
  
  # late ast
  process_ast = ast_parse(ast)
  store(0, process_ast)
  kernel.append("}")

  # kernel function definition
  function_name = ("re_S" if k.reduceop else "ew_S") + '_'.join([str(x) for x in k.bufs[0].shape if x != 1])
  kernel = list(prekernel) + [f"__kernel void {function_name}(",] + [', '.join(f'{t} data{i}' for i,t in enumerate(buftypes) if i not in bufs_to_delete)] + [") {\n"] + kernel
  if DEBUG >= 2:
    print(first_reduce, last_reduce, ast)
    print(' '.join(kernel))

  # compile kernel
  fxn = CLProgram(function_name, ' '.join(kernel))

  def runner(*bufs):
    clbufs = [x.cl for i,x in enumerate(bufs) if i not in bufs_to_delete]
    return fxn(output_shape[::-1] if len(output_shape) > 0 else [1], None, *clbufs, op_estimate=k.info.flops)
  return runner

class GPUBuffer(ExplicitExecAST):
  code_for_op : Dict[Op, str] = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)",
    UnaryOps.EXP: "exp(A)", UnaryOps.LOG: "log(A)", UnaryOps.SIGN: "sign(A)", UnaryOps.RECIPROCAL: "((float)1.0/A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
    BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
    ReduceOps.SUM: "(acc + A)", ReduceOps.MAX: "max(A, acc)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[GPUBuffer]=None, backing:Optional[np.ndarray]=None, image=False):
    super().__init__(shape, hostbuf)
    self._buf : Optional[CLBuffer] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing

    # image
    if image and hostbuf is None:
      assert self._backing is None
      self._buf = CLImage(self._base_shape)

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
    CL.enqueue_copy(data, self.movement_op(MovementOps.RESHAPE, list(self.shape)+[1]).unary_op(UnaryOps.NOOP).cl if isinstance(self._buf, CLImage) else self.contiguous().cl, is_blocking=True)
    return data

  func_cache : Dict[str, Any] = {}
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_shape:Optional[Tuple[int, ...]]=None) -> GPUBuffer:
    k = ASTKernel(ast, output_shape)
    # can't cache with constant folding
    #if k.key not in GPUBuffer.func_cache:
    GPUBuffer.func_cache[k.key] = ast_kernel_codegen(cls, ast, k)
    GPUBuffer.func_cache[k.key](*k.bufs)
    return k.ret
