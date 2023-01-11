from __future__ import annotations
import os, functools
from enum import Enum
import numpy as np
import pyopencl as cl  # type: ignore
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union, Set
from tinygrad.helpers import prod, all_same
from tinygrad.ops import DEBUG, ASTKernel, UnaryOps, BinaryOps, ReduceOps, MovementOps, LazyOp, Op, ExplicitExecAST, GlobalCounters
from tinygrad.lazy import IMAGE
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

FLOAT16 = int(os.getenv("FLOAT16", "0"))
class CLImage:
  fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.HALF_FLOAT if FLOAT16 else cl.channel_type.FLOAT)

  def __init__(self, shape):
    self.cl = cl.Image(CL.cl_ctx, cl.mem_flags.READ_WRITE, CLImage.fmt, shape=(shape[0], shape[1]))
    CL.mem_used += self.cl.row_pitch * self.cl.height

  def __del__(self):
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
  kernel_cnt : Dict[str, int] = defaultdict(int)
  def __init__(self, name:str, prg:str, options:Tuple[str, ...]=tuple(), argdtypes=None, rename=True, binary=False):
    self.name = f"{name}{('_N'+str(CLProgram.kernel_cnt[name])) if CLProgram.kernel_cnt[name] else ''}" if rename else name
    self.prg, self.options, self.argdtypes = prg.replace(f"{name}(", f"{self.name}(") if rename else prg, options, argdtypes
    self.clprogram = cl.Program(CL().cl_ctx, CL().cl_ctx.devices, [self.prg]) if binary else cl.Program(CL().cl_ctx, self.prg)  # type: ignore
    try:
      self.clprg = self.clprogram.build(options=list(self.options)).__getattr__(self.name)
    except cl.RuntimeError as e:
      print("FAILED TO BUILD", self.prg)
      raise e
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

Types = Enum("Types", ["FLOAT", "FLOAT4"])
class Token:
  def __init__(self, tok:str, typ:Types):
    assert isinstance(tok, str)
    self.tok = tok
    self.typ = typ
  def decltype(self): return 'float' if self.typ == Types.FLOAT else 'float4'
  def __repr__(self): return f"<{self.typ} {self.tok}>"

class CLASTKernel(ASTKernel):
  def __init__(self, ast:LazyOp):
    super().__init__(ast)

  def compute_buf_index(self, st, buf_index, offset=0):
    key = f"{buf_index}_{offset}"
    # add the index if we don't have it
    if key not in self.seen_idx:
      idx_pieces = [str(st.offset + offset)] + [(f"idx{i}*{st}" if st != 1 else f"idx{i}") for i,(sh,st) in enumerate(zip(self.shapes[buf_index][0:self.last_reduce], self.strides[buf_index][0:self.last_reduce])) if sh != 1 and st != 0]
      if st.needs_valid(): self.kernel.append(f"bool bufvalid{key} = true;")
      self.kernel.append(f"int bufi{key} = " + '('+' + '.join(idx_pieces)+');\n')
      if len(st.views) > 1:
        extra_idx = ';'.join([v.expr for v in st.views[0:-1][::-1] if v.expr not in ['', 'idx=idx', 'valid=valid']])
        self.kernel.append(extra_idx.replace("//", "/").replace("idx", f"bufi{key}").replace("valid", f"bufvalid{key}") + ";\n")
      self.seen_idx.add(key)
    return key

  def store(self, buf_index, value:Token, offset=0):
    st = self.bufs[buf_index].st
    if offset > 0: assert len(st.views) == 1
    key = self.compute_buf_index(st, buf_index, offset)
    if isinstance(self.bufs[buf_index]._buf, CLImage):
      W = self.bufs[buf_index]._base_shape[1]
      assert value.typ == Types.FLOAT4, f"image can only store float4: {value} isn't"
      self.kernel.append(f"write_imagef(data{buf_index}, (int2)((bufi{key})/{W*4}, ((bufi{key})/4)%{W}), {value.tok});  /* {self.bufs[buf_index]._base_shape} */\n")
    else:
      if value.typ == Types.FLOAT4:
        #assert len(st.views) == 1
        self.kernel.append(f"float4 to_store = {value.tok};\n")
        for i in range(4):
          lkey = self.compute_buf_index(st, buf_index, offset+i*self.strides[buf_index][-1])
          self.kernel.append(f"data{buf_index}[bufi{lkey}] = to_store.s{i};\n")
      else:
        self.kernel.append(f"data{buf_index}[bufi{key}] = {value.tok};\n")

  def load(self, buf_index, offset=0) -> Token:
    if buf_index not in self.loaded_keys:
      st = self.bufs[buf_index].st
      if offset > 0: assert len(st.views) == 1
      key = f"{buf_index}_{offset}"

      # constant folding
      constant_fold = None
      if self.bufs[buf_index]._base_shape == (1,) and self.bufs[buf_index]._backing:
        self.bufs_to_delete.add(buf_index)
        constant_fold = f"({self.bufs[buf_index]._backing[0]})"
      
      if isinstance(self.bufs[buf_index]._buf, CLImage):
        W = self.bufs[buf_index]._base_shape[1]
        assert not st.needs_valid()
        assert len(st.views) == 1

        c0, c1 = [str(offset//(W*4))], [str(offset//4)]
        for i, (shape, stride) in enumerate(zip(self.shapes[buf_index][0:self.last_reduce], self.strides[buf_index][0:self.last_reduce])):
          if shape == 1 or stride == 0: continue

          if stride%(W*4) == 0:
            if stride//(W*4) != 0:
              c0.append(f"(idx{i} * {stride//(W*4)})")
          else:
            if shape*stride > W*4:
              c0.append(f"((idx{i} * {stride})/({W*4}))")

          if stride%4 == 0:
            if stride//4 < W:
              if (shape * stride//4) <= W: # no mod required
                c1.append(f"(idx{i} * {stride//4})")
              else:
                c1.append(f"(idx{i} * {stride//4})%{W}")
          else:
            c1.append(f"((idx{i} * {stride})/4)%{W}")
        ldr = Token(f"read_imagef(data{buf_index}, smp, (int2)({'+'.join(c0)}, {'+'.join(c1)}))  /* {self.bufs[buf_index]._base_shape} */", Types.FLOAT4)
      else:
        self.compute_buf_index(st, buf_index, offset)
        if (self.bufs[buf_index] in self.earlybufs and self.early_is_float4) or (self.bufs[buf_index] not in self.earlybufs and self.late_is_float4):
          if self.strides[buf_index][-1] == 1 and len(st.views) == 1 and not st.needs_valid():
            ldr = Token(f"((__global float4*)data{buf_index})[bufi{key}/4]", Types.FLOAT4)
          else:
            mst = []
            for i in range(4):
              lkey = self.compute_buf_index(st, buf_index, offset+i*self.strides[buf_index][-1])
              mst.append(f"data{buf_index}[bufi{lkey}]" if not constant_fold else constant_fold)
              if st.needs_valid(): mst[-1] = f"(bufvalid{key} ? {mst[-1]} : 0.0)"
            ldr = Token(f"(float4)({','.join(mst)})", Types.FLOAT4)
        else:
          ldr = f"data{buf_index}[bufi{key}]" if not constant_fold else constant_fold
          ldr = Token(f"(bufvalid{key} ? {ldr} : 0.0)" if st.needs_valid() else ldr, Types.FLOAT)
      self.kernel.append(f"{ldr.decltype()} val{key} = {ldr.tok};\n")
      self.loaded_keys[buf_index] = Token(f"val{key}", ldr.typ)
    return self.loaded_keys[buf_index]

  def ast_parse(self, x, reduce=False) -> Token:
    if not isinstance(x, LazyOp): return self.load(self.bufs.index(x))
    if isinstance(x.op, ReduceOps) and not reduce: return self.reduce_token
    values = [self.ast_parse(v, reduce) for v in x.src]
    code = GPUBuffer.code_for_op[x.op]  # TODO: replace this with a function
    if isinstance(x.op, ReduceOps):
      if self.reduce_token.typ == Types.FLOAT and values[0].typ == Types.FLOAT4:
        self.prekernel.add("float clsum(float4 x) { return x.x + x.y + x.z + x.w; }\n")
        reduce = Token(f"clsum({values[0].tok})", Types.FLOAT)
      else:
        reduce = values[0]
      return Token(code.replace("A", reduce.tok), self.reduce_token.typ)
    assert all_same([x.typ for x in values]), f"type mismatch in {values}"
    if len(values) >= 1: code = code.replace("A", values[0].tok)
    if len(values) >= 2: code = code.replace("B", values[1].tok)
    return Token(code, values[0].typ)

  def codegen(self):
    # TODO: fetch from quick cache before processing
    self.process()
    buftypes = [f"{'read_only' if i > 0 else 'write_only'} image2d_t" if isinstance(x._buf, CLImage) else "__global float *" for i,x in enumerate(self.bufs)]
    self.prekernel = set()

    # three things can be float4
    self.early_is_float4 = any(isinstance(x._buf, CLImage) for x in self.bufs[1:])
    self.late_is_float4 = isinstance(self.bufs[0]._buf, CLImage)

    self.bufs_to_delete : Set[int] = set()
    self.seen_idx : Set[str] = set()
    self.loaded_keys : Dict[int, Token] = {}

    self.reduce_token = Token("acc", Types.FLOAT4 if self.late_is_float4 else Types.FLOAT)

    self.output_shape = self.shapes[0][:self.first_reduce]
    self.kernel : List[str] = ["const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"]
    self.kernel += [f"int idx{i} = get_global_id({min(3, len(self.output_shape))-1-i});\n" for i in range(min(3, len(self.output_shape)))]
    if len(self.output_shape) > 3:
      # compact all the dimensions into the final one
      for i in range(len(self.output_shape)-1, 2, -1):
        self.kernel += [f"int idx{i} = idx2 % {self.output_shape[i]};", f"idx2 = idx2 / {self.output_shape[i]};\n"]
      self.output_shape = list(self.output_shape[0:2]) + [prod(self.output_shape[2:])]

    # early ast
    if self.reduceop:
      full_shape = [x for x in self.shapes if x != self.shapes[0]]
      full_shape = self.shapes[0] if len(full_shape) == 0 else full_shape[0]

      self.kernel.append(f"{self.reduce_token.decltype()} {self.reduce_token.tok} = {GPUBuffer.start_for_op[self.reduceop.op]};\n")
      for i in range(self.first_reduce, self.last_reduce):
        self.kernel.append(f"for (int idx{i} = 0; idx{i} < {full_shape[i]}; idx{i}++) {{\n")
      ast_reduced = self.ast_parse(self.reduceop, reduce=True)
      self.kernel.append(f"  {self.reduce_token.tok} = {ast_reduced.tok};\n")
      self.kernel += ["}\n"] * (self.last_reduce - self.first_reduce)

    # late ast
    self.store(0, self.ast_parse(self.ast))
    self.kernel.append("}")

    # kernel function definition
    function_name = ("re_S" if self.reduceop else "ew_S") + '_'.join([str(x) for x in self.bufs[0].shape if x != 1])
    self.kernel = list(self.prekernel) + [f"__kernel void {function_name}(",] + [', '.join(f'{t} data{i}' for i,t in enumerate(buftypes) if i not in self.bufs_to_delete)] + [") {\n"] + self.kernel

    # compile kernel
    fxn = CLProgram(function_name, ' '.join(self.kernel))

    def runner(*bufs):
      clbufs = [x.cl for i,x in enumerate(bufs) if i not in self.bufs_to_delete]
      return fxn(self.output_shape[::-1] if len(self.output_shape) > 0 else [1], None, *clbufs, op_estimate=self.info.flops)
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

  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[GPUBuffer]=None, backing:Optional[np.ndarray]=None):
    super().__init__(shape, hostbuf)
    self._buf : Optional[Union[CLBuffer, CLImage]] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    # early copy in for large buffers
    if self._backing is not None and self._backing.shape != (1,):
      self.cl
  
  @property
  def cl(self):
    if self._buf is None:
      if len(self._base_shape) == 3 and self._base_shape[2] == 4:
        self._buf = CLImage(self._base_shape)
      else:
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

  @classmethod
  def exec_ast(cls, ast:LazyOp):
    k = CLASTKernel(ast)
    k.codegen()(*k.bufs)
    return k.ret
