from __future__ import annotations
import os, functools
import numpy as np
import pyopencl as cl  # type: ignore
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union, Set
from tinygrad.helpers import prod, all_same
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, MovementOps, LazyOp, Op, ExplicitExecAST, GlobalCounters
from tinygrad.ast import ASTKernel, Token, Types
from tinygrad.lazy import IMAGE
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.shape.symbolic import Variable, ModNode

VALIDHACKS = int(os.getenv("VALIDHACKS", "0"))    # TODO: remove the need for this
NATIVE_EXPLOG = int(os.getenv("NATIVE_EXPLOG", 0))  # this is needed as a switch for the tests to pass

CLCACHE = int(os.getenv("CLCACHE", "1"))
FLOAT16 = int(os.getenv("FLOAT16", "0"))
PRINT_AST = int(os.getenv("PRINT_AST", "0"))
TEST_AST = int(os.getenv("TEST_AST", "0"))

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
    self.cl = cl.Image(CL().cl_ctx, cl.mem_flags.READ_WRITE, CLImage.fmt, shape=(shape[1], shape[0]))
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
  def __init__(self, name:str, prg:str, options:Tuple[str, ...]=tuple(), argdtypes=None, rename=True, binary=False, op_estimate=0):
    self.name = f"{name}{('_N'+str(CLProgram.kernel_cnt[name])) if CLProgram.kernel_cnt[name] else ''}" if rename else name
    self.prg, self.options, self.argdtypes, self.op_estimate = prg.replace(f"{name}(", f"{self.name}(") if rename else prg, options, argdtypes, op_estimate
    self.clprogram = cl.Program(CL().cl_ctx, CL().cl_ctx.devices, [self.prg]) if binary else cl.Program(CL().cl_ctx, self.prg)  # type: ignore
    try:
      self.clprg = self.clprogram.build(options=list(self.options)).__getattr__(self.name)
    except cl.RuntimeError as e:
      print("FAILED TO BUILD", self.prg)
      raise e
    if self.argdtypes is not None:
      self.clprg.set_scalar_arg_dtypes(self.argdtypes)
    CLProgram.kernel_cnt[name] += 1
  def __call__(self, *args):
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
      CL.ops_sum += self.op_estimate
      print(f"**CL** {CL.kernel_count:6d} {self.name:28s} args {len(args[2:]):5d}  kernels {str(args[0]):18s} {str(args[1]):12s} OPs {self.op_estimate/1e6:7.1f}M/{CL.ops_sum/1e9:7.2f}G  mem {CL.mem_used/1e9:5.2f} GB " +
            ("" if DEBUG <= 1 or CL.CACHE is not None else f"tm {(e.profile.end - e.profile.start)/1e3:9.2f}us/{CL.time_sum/1e6:9.2f}ms ({self.op_estimate/(e.profile.end - e.profile.start):8.2f} GFLOPS)"))
    GlobalCounters.global_ops += self.op_estimate
    GlobalCounters.global_mem += sum([x.size//4 for x in args[2:] if isinstance(x, cl.Buffer)])

# **** end CL wrappers ****

def group_float4(x): return [Token(f"(float4)({','.join([x[i+j].tok for j in range(4)])})", Types.FLOAT4) for i in range(0, len(x), 4)]
def split_float4(x): return sum([[Token(acc.tok+f".s{s}", Types.FLOAT) for s in range(4)] for acc in x], [])

class CLASTKernel(ASTKernel):
  code_for_op : Dict[Op, str] = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)", UnaryOps.SIGN: "sign(A)",
    UnaryOps.EXP: "native_exp(A)" if NATIVE_EXPLOG else "exp(A)",
    UnaryOps.LOG: "native_log(A)" if NATIVE_EXPLOG else "log(A)",
    UnaryOps.RECIPROCAL: "native_recip(A)" if NATIVE_EXPLOG else "((float)1.0/A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
    BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
    ReduceOps.SUM: "(acc + A)", ReduceOps.MAX: "max(A, acc)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def __init__(self, ast:LazyOp):
    super().__init__(ast)

  # TODO: move to shapetracker
  def compute_buf_index_symbolic(self, st, buf_index, offset=0):
    view = View(self.shapes[buf_index][0:self.last_reduce], self.strides[buf_index][0:self.last_reduce], self.offsets[buf_index] + offset)
    idx = view.expr_idxs([f"idx{i}" for i in range(self.last_reduce)])
    valid = Variable.num(1)
    for v in st.views[0:-1][::-1]:
      if isinstance(v, ZeroView): valid = v.expr_node(valid, idx)
      else: idx = v.expr_node(idx)
    return idx, valid

  def store(self, buf_index, value:List[Token]):
    if len(value) == self.buftokens[buf_index].size()*4: value = group_float4(value)
    if len(value)*4 == self.buftokens[buf_index].size(): value = split_float4(value)
    assert len(value) == self.buftokens[buf_index].size(), f"size mismatch {len(value)} != {self.buftokens[buf_index].size()}"
    for v, o in zip(value, self.buftokens[buf_index].offsets()):
      idxy, valid = self.compute_buf_index_symbolic(self.bufs[buf_index].st, buf_index, o)
      assert str(valid) == "1"
      if isinstance(self.bufs[buf_index]._buf, CLImage):
        assert self.buftokens[buf_index].typ == Types.FLOAT4, "image must be FLOAT4"
        idx = (idxy//4)%self.bufs[buf_index]._base_shape[1]
        idy = (idxy//(4*self.bufs[buf_index]._base_shape[1]))%self.bufs[buf_index]._base_shape[0]
        self.kernel.append(f"write_imagef(data{buf_index}, (int2)({idx.cl}, {idy.cl}), {v.tok});  /* {self.bufs[buf_index]._base_shape} */\n")
      else:
        assert self.buftokens[buf_index].typ == v.typ, f"buf must be {v.typ}"
        self.kernel.append(f"data{buf_index}[{(idxy//(4 if v.typ == Types.FLOAT4 else 1)).cl}] = {v.tok};\n")

    """
    st = self.bufs[buf_index].st
    idxy, valid = self.compute_buf_index_symbolic(st, buf_index, offset)
    assert str(valid) == "1"
    if isinstance(self.bufs[buf_index]._buf, CLImage):
      W = self.bufs[buf_index]._base_shape[1]
      assert value.typ == Types.FLOAT4, f"image can only store float4: {value} isn't"

      idx = (idxy//4)%W
      idy = (idxy//(W*4))%self.bufs[buf_index]._base_shape[0]
      self.kernel.append(f"write_imagef(data{buf_index}, (int2)({idx.cl}, {idy.cl}), {value.tok});  /* {self.bufs[buf_index]._base_shape} */\n")
    else:
      if value.typ == Types.FLOAT4:
        for i in range(4):
          # TODO: this isn't tested
          lidxy, lvalid = self.compute_buf_index_symbolic(st, buf_index, offset+i*self.strides[buf_index][-1])
          assert str(lvalid) == "1"
          self.kernel.append(f"data{buf_index}[{lidxy.cl}] = {value.tok}.s{i};\n")
      else:
        self.kernel.append(f"data{buf_index}[{idxy.cl}] = {value.tok};\n")
    """

  def load(self, buf_index:int) -> List[Token]:
    # constant folding
    constant_fold = None
    if self.bufs[buf_index]._base_shape == (1,) and self.bufs[buf_index]._backing:
      assert self.buftokens[buf_index].typ == Types.FLOAT
      self.bufs_to_delete.add(buf_index)
      return [Token(f"({self.bufs[buf_index]._backing[0]})", self.buftokens[buf_index].typ)] * self.buftokens[buf_index].size()

    # not constant folded
    tokens = []
    for o in self.buftokens[buf_index].offsets():
      if (buf_index, o) not in self.loaded_keys:
        idxy, valid = self.compute_buf_index_symbolic(self.bufs[buf_index].st, buf_index, o)
        if isinstance(self.bufs[buf_index]._buf, CLImage):
          assert self.buftokens[buf_index].typ == Types.FLOAT4, f"image must be FLOAT4 {self.buftokens[buf_index]} {self.bufs[buf_index].st}"
          idx = (idxy//4)%self.bufs[buf_index]._base_shape[1]
          idy = (idxy//(4*self.bufs[buf_index]._base_shape[1]))%self.bufs[buf_index]._base_shape[0]

          if VALIDHACKS:
            if isinstance(idx, ModNode) and idx.max < idx.b*2: idx = idx.a
            if isinstance(idy, ModNode) and idy.max < idy.b*2: idy = idy.a
            valid = None

          ldrt = f"read_imagef({self.buftokens[buf_index].tok}, smp, (int2)({idx.cl}, {idy.cl})) /* {self.bufs[buf_index]._base_shape} */"
          ldr = Token(f"({valid.cl} ? \\ \n   {ldrt} : (float4)(0.0, 0.0, 0.0, 0.0))" if str(valid) != "1" and valid is not None else ldrt, Types.FLOAT4)
        else:
          ldr = Token(f"{self.buftokens[buf_index].tok}[{(idxy//(4 if self.buftokens[buf_index].typ == Types.FLOAT4 else 1)).cl}]", self.buftokens[buf_index].typ)
          ldr = Token(f"({valid.cl} ? {ldr.tok} : 0.0)", ldr.typ) if str(valid) != "1" else ldr
        self.kernel.append(f"{ldr.decltype()} val{buf_index}_{o} = {ldr.tok};\n")
        self.loaded_keys[(buf_index,o)] = Token(f"val{buf_index}_{o}", ldr.typ)
      tokens.append(self.loaded_keys[(buf_index,o)])
    return tokens



    key = f"{buf_index}_{offset}"
    if key not in self.loaded_keys:
      st = self.bufs[buf_index].st

      # constant folding
      constant_fold = None
      if self.bufs[buf_index]._base_shape == (1,) and self.bufs[buf_index]._backing:
        self.bufs_to_delete.add(buf_index)
        constant_fold = f"({self.bufs[buf_index]._backing[0]})"

      offset_index = -2 if self.late_are_float4 and self.bufs[buf_index] in self.earlybufs else -1

      if isinstance(self.bufs[buf_index]._buf, CLImage):
        # TODO: why isn't this always right? it should be
        #assert self.strides[buf_index][offset_index] == 1
        W = self.bufs[buf_index]._base_shape[1]
        idxy, valid = self.compute_buf_index_symbolic(st, buf_index, offset)
        idx = (idxy//4)%W
        idy = (idxy//(W*4))%self.bufs[buf_index]._base_shape[0]
        # TODO: apply the validity assumptions to the indexes

        if VALIDHACKS:
          if isinstance(idx, ModNode) and idx.max < idx.b*2: idx = idx.a
          if isinstance(idy, ModNode) and idy.max < idy.b*2: idy = idy.a
          valid = None

        ldrt = f"read_imagef(data{buf_index}, smp, (int2)({idx.cl}, {idy.cl})) /* {self.bufs[buf_index]._base_shape} */"
        ldr = Token(f"({valid.cl} ? \\ \n   {ldrt} : (float4)(0.0, 0.0, 0.0, 0.0))" if st.needs_valid() and valid is not None else ldrt, Types.FLOAT4)
      else:
        idxy, valid = self.compute_buf_index_symbolic(st, buf_index, offset)
        if self.late_are_float4 or (self.early_loads_are_float4 and self.bufs[buf_index] in self.earlybufs):
          if self.strides[buf_index][offset_index] == 1 and len(st.views) == 1 and not st.needs_valid():
            ldr = Token(f"((__global float4*)data{buf_index})[{(idxy//4).cl}]", Types.FLOAT4)
          else:
            mst = []
            for i in range(4):
              lidxy,lvalid = self.compute_buf_index_symbolic(st, buf_index, offset+i*self.strides[buf_index][offset_index])
              mst.append(f"data{buf_index}[{lidxy.cl}]" if not constant_fold else constant_fold)
              if st.needs_valid(): mst[-1] = f"({lvalid.cl} ? {mst[-1]} : 0.0)"
            ldr = Token(f"(float4)({','.join(mst)})", Types.FLOAT4)
        else:
          ldrt = f"data{buf_index}[{idxy.cl}]" if not constant_fold else constant_fold
          ldr = Token(f"({valid.cl} ? {ldrt} : 0.0)" if st.needs_valid() else ldrt, Types.FLOAT)
      self.kernel.append(f"{ldr.decltype()} val{key} = {ldr.tok};\n")
      self.loaded_keys[key] = Token(f"val{key}", ldr.typ)
    return self.loaded_keys[key]

  def ast_parse(self, x:Union[GPUBuffer, LazyOp], reduce:Optional[List[Token]]=None) -> List[Token]:
    if not isinstance(x, LazyOp): return self.load(self.bufs.index(x))
    if isinstance(x.op, ReduceOps) and reduce is not None: return reduce
    values = [self.ast_parse(v, reduce) for v in x.src]
    code = CLASTKernel.code_for_op[x.op]  # TODO: replace this with a function
    if len(values) == 2:
      if values[0][0].typ != values[1][0].typ:
        if values[0][0].typ == Types.FLOAT: values[0] = group_float4(values[0])
        if values[1][0].typ == Types.FLOAT: values[1] = group_float4(values[1])
      assert len(values[0]) == len(values[1])
      return [Token(code.replace("A", a.tok).replace("B", b.tok), a.typ) for a,b in zip(values[0], values[1])]
    else:
      return [Token(code.replace("A", a.tok), a.typ) for a in values[0]]

    #if isinstance(x.op, ReduceOps) and values[0].typ != Types.FLOAT and not self.early_loads_are_non_reduce_float4:
    #  self.prekernel.add("float clsum(float4 x) { return x.x + x.y + x.z + x.w; }\n")
    #  return Token(code.replace("A", f"clsum({values[0].tok})").replace("acc", f"acc.s{offset}" if self.late_are_float4 else "acc"), Types.FLOAT)
    #assert all_same([x.typ for x in values]), f"type mismatch in {values}"
    print(code, values)
    if len(values) >= 1: code = code.replace("A", values[0].tok)
    if len(values) >= 2: code = code.replace("B", values[1].tok)
    return Token(code, values[0].typ)

  def codegen(self):
    # TODO: fetch from quick cache before processing
    self.process()
    if DEBUG >= 2:
      print("old:", self.shapes)
      print("old:", self.strides)

    self.prekernel = set()

    # four toggles determine the kernel
    self.early_loads_are_non_reduce_float4 = False
    self.early_loads_are_float4 = False
    self.late_are_float4 = False   # store float4
    self.four_float4 = False

    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    # shove the axis to the end and remove 
    if any(isinstance(buf._buf, CLImage) for buf in self.earlybufs):
      eb_valids = [True] * len(self.shapes[0])
      for i in range(len(self.bufs)):
        if isinstance(self.bufs[i]._buf, CLImage) and self.bufs[i] in self.earlybufs:
          #assert len(self.bufs[i].st.views) == 1, f"images can't have views {self.bufs[i].st}"
          valids = [self.shapes[i][j]%4 == 0 and self.strides[i][j] == 1 for j in range(len(self.shapes[i]))]
          eb_valids = [x and y for x,y in zip(eb_valids, valids)]
      assert any(eb_valids), f"invalid op with images {eb_valids}"
      eb_valid = eb_valids.index(True)
      if DEBUG >= 2: print(f"early merging axis {eb_valid}")

      # no change, we added a dimension
      self.reshape_and_permute(
        lambda x: list(x[0:eb_valid]) + ([x[eb_valid]//4, 4] if x[eb_valid] > 1 else [1,1]) + list(x[eb_valid+1:]),
        [i for i in range(self.shape_len+1) if i != eb_valid+1] + [eb_valid+1])

      # drop the last dimension
      self.upcast()
    
    # if there's images in the latebufs, we have to make an axis the 4 storing one. this affects the kernel shape
    if any(isinstance(buf._buf, CLImage) for buf in self.bufs if buf not in self.earlybufs) and self.buftokens[0].typ != Types.FLOAT4:
      lb_valids = [True] * len(self.shapes[0])
      for i in range(len(self.bufs)):
        #assert len(self.bufs[i].st.views) == 1 or not isinstance(self.bufs[i]._buf, CLImage)  # images can't have views
        valids = [self.shapes[i][j]%4 == 0 and (self.strides[i][j] == 1 or not isinstance(self.bufs[i]._buf, CLImage) or self.bufs[i] in self.earlybufs) for j in range(len(self.shapes[i]))]
        lb_valids = [x and y for x,y in zip(lb_valids, valids)]
      assert any(lb_valids), f"invalid op with images {lb_valids}"
      lb_valid = lb_valids.index(True)
      assert lb_valid < self.first_reduce, f"can't be in the reduce {lb_valid}"
      if DEBUG >= 2: print(f"late merging axis {lb_valid}")

      # no change, we added a dimension
      self.reshape_and_permute(
        lambda x: list(x[0:lb_valid]) + [x[lb_valid]//4, 4] + list(x[lb_valid+1:]),
        [i for i in range(self.shape_len+1) if i != lb_valid+1] + [lb_valid+1])

      # drop the last dimension
      self.upcast()

    # split to 4 float4s
    """
    if self.buftokens[0].typ == Types.FLOAT4:
      xb_choices = []
      for i in range(self.first_reduce):
        if all(x[i]%4 == 0 for x in self.shapes) and any([(x[i] != 0 and x[-1] == 0) or (x[i] == 0 and x[-1] != 0) for x in self.strides]):
          xb_choices.append((sum(x[i] for x in self.strides), i))

      if len(xb_choices):
        xb_choice = sorted(xb_choices)[0][1]
        if DEBUG >= 2: print(f"float4 merging axis {xb_choice}")

        # this leaves the last axis in place
        self.reshape_and_permute(
          lambda x: list(x[0:xb_choice]) + [x[xb_choice]//4, 4] + list(x[xb_choice+1:]),
          [i for i in range(self.shape_len+1) if i != xb_choice+1] + [xb_choice+1])

        # drop the last dimension
        self.upcast()
    """

    # first simplify
    self.simplify_ones()

    # use more opencl indexing
    if self.first_reduce == 2 and isinstance(self.bufs[0]._buf, CLImage):
      base_shape = self.bufs[0]._base_shape
      if all([(base_shape[0]*base_shape[1])%x[0] == 0 for x in self.shapes]):
        #print("split here", base_shape, self.shapes[0])
        self.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
        self.last_reduce += 1
        self.simplify_ones()

    self.output_shape = self.shapes[0][:min(self.first_reduce, self.last_reduce)]

    if DEBUG >= 2:
      print(f"early_loads_are_non_reduce_float4: {self.early_loads_are_non_reduce_float4} early_loads_are_float4: {self.early_loads_are_float4} late_are_float4: {self.late_are_float4} four_float4: {self.four_float4}")
      print(f"first_reduce: {self.first_reduce} last_reduce: {self.last_reduce} shape_len: {len(self.bufs[0].shape)}")
      print("output shape", self.output_shape)
      for i in range(len(self.bufs)):
        print(self.buftokens[i], self.bufs[i] in self.earlybufs, self.shapes[i], self.strides[i])

    self.bufs_to_delete : Set[int] = set()
    self.loaded_keys : Dict[Tuple[int,int], Token] = {}

    self.kernel : List[str] = ["const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"]
    self.kernel += [f"int idx{i} = get_global_id({min(3, len(self.output_shape))-1-i}); /* {self.output_shape[i]} */\n" for i in range(min(3, len(self.output_shape)))]
    if len(self.output_shape) > 3:
      # compact all the dimensions into the final one
      for i in range(len(self.output_shape)-1, 2, -1):
        self.kernel += [f"int idx{i} = idx2 % {self.output_shape[i]};", f"idx2 = idx2 / {self.output_shape[i]};\n"]
      self.output_shape = list(self.output_shape[0:2]) + [prod(self.output_shape[2:])]

    # early ast
    accumulators : List[Token] = [Token("acc%d" % i, self.buftokens[0].typ) for i in range(self.buftokens[0].size())]
    if self.reduceop:
      full_shape = [x for x in self.shapes if x != self.shapes[0]]
      full_shape = self.shapes[0] if len(full_shape) == 0 else full_shape[0]

      self.kernel += [f"{accumulator.decltype()} {accumulator.tok} = {CLASTKernel.start_for_op[self.reduceop.op]};\n" for accumulator in accumulators]
      self.kernel += [f"for (int idx{i} = 0; idx{i} < {full_shape[i]}; idx{i}++) {{\n" for i in range(self.first_reduce, self.last_reduce)]

      asts = self.ast_parse(self.reduceop.src[0])

      new_accumulators = split_float4(accumulators) if len(accumulators) != len(asts) else accumulators
      for ast, accumulator in zip(asts, new_accumulators):
        if ast.typ != accumulator.typ:
          assert self.reduceop.op == ReduceOps.SUM
          assert ast.typ == Types.FLOAT4 and accumulator.typ == Types.FLOAT
          self.prekernel.add("float clsum(float4 x) { return x.x + x.y + x.z + x.w; }\n")
          self.kernel.append(f"  {accumulator.tok} = {accumulator.tok} + clsum(" + ast.tok.replace("acc", accumulator.tok) + ");\n")
        else:
          if self.reduceop.op == ReduceOps.MAX:
            self.kernel.append(f"  {accumulator.tok} = max({accumulator.tok}, " + ast.tok.replace("acc", accumulator.tok) + ");\n")
          else:
            self.kernel.append(f"  {accumulator.tok} = {accumulator.tok} + " + ast.tok.replace("acc", accumulator.tok) + ";\n")

      #for accnum, accumulator in enumerate(accumulators):
      """
      print(ast)
      tmp_kernel = []
      for accnum, accumulator in enumerate(accumulators):
        if self.late_are_float4 and not self.early_loads_are_non_reduce_float4:
          tmp_kernel += [f"  {accumulator.tok}.s{j} = " + self.ast_parse(self.reduceop, offset=j, alt_offset=accnum).tok.replace("acc", f"acc{accnum}") + ";\n" for j in range(4)]
        else:
          tmp_kernel.append(f"  {accumulator.tok} = " + self.ast_parse(self.reduceop, alt_offset=accnum).tok.replace("acc", f"acc{accnum}") + ";\n")
      """

      self.kernel += ["}\n"] * (self.last_reduce - self.first_reduce)
    
    # late ast
    outs : List[Token] = []
    out : List[Token] = self.ast_parse(self.ast, reduce=accumulators)
    for i, out in enumerate(out):
      self.kernel.append(f"{out.decltype()} outs{i} = {out.tok};\n")
      outs.append(Token(f"outs{i}", out.typ))

    self.store(0, outs)
    self.kernel.append("}")

    # kernel function definition
    function_name = ("re_S" if self.reduceop else "ew_S") + '_'.join([str(x) for x in self.bufs[0].shape if x != 1])
    buftypes = [f"{'read_only' if i > 0 else 'write_only'} image2d_t" if isinstance(x._buf, CLImage) else ("__global "+self.buftokens[i].decltype()) for i,x in enumerate(self.bufs)]
    self.kernel = list(self.prekernel) + [f"__kernel void {function_name}(",] + [', '.join(f'{t} data{i}' for i,t in enumerate(buftypes) if i not in self.bufs_to_delete)] + [") {\n"] + self.kernel

    # compile kernel
    self.fxn = CLProgram(function_name, ' '.join(self.kernel), op_estimate=self.info.flops)

    def runner(*bufs):
      clbufs = [x.cl for i,x in enumerate(bufs) if i not in self.bufs_to_delete]
      return self.fxn(self.output_shape[::-1] if len(self.output_shape) > 0 else [1], None, *clbufs)
    return runner
  def print(self):
    super().print()
    print(self.fxn.prg)

class GPUBuffer(ExplicitExecAST):
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf:Optional[GPUBuffer]=None, backing:Optional[np.ndarray]=None, force_create=False):
    super().__init__(shape, hostbuf)
    self._buf : Optional[CLBuffer] = hostbuf._buf if hostbuf is not None else None
    self._base_shape : Tuple[int, ...] = hostbuf._base_shape if hostbuf is not None else self.shape
    self._backing : Optional[np.ndarray] = hostbuf._backing if hostbuf is not None else backing
    # early copy in for large buffers
    if (self._backing is not None and self._backing.shape != (1,)) or force_create:
      self.cl
  
  @property
  def cl(self):
    if self._buf is None:
      self._buf = CLImage(self._base_shape) if (len(self._base_shape) == 3 and self._base_shape[2] == 4 and IMAGE >= 2) else CLBuffer(4*prod(self._base_shape))
    if self._backing is not None:
      CL().enqueue_copy(self._buf.cl, self._backing, is_blocking=False)
      self._backing = None
    return self._buf.cl

  def __repr__(self): return f"GPUBuffer(shape={self.st}, hostbuf=GPUBuffer(shape={self._base_shape}" + (f", backing=np.array({self._backing}, dtype=np.float32)))" if self._backing else ", force_create=True))")

  @staticmethod
  def fromCPU(x): return GPUBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    cl_buf = self.contiguous()
    cl_buf = cl_buf if isinstance(cl_buf._buf, CLBuffer) else self.movement_op(MovementOps.RESHAPE, list(self.shape)+[1]).unary_op(UnaryOps.NOOP)
    CL().enqueue_copy(data, cl_buf.cl, is_blocking=True)
    return data

  @classmethod
  def exec_ast(cls, ast:LazyOp):
    k = CLASTKernel(ast)
    k.codegen()(*k.bufs)
    if PRINT_AST:
      print(k.fxn.name)
      k.print()
    if TEST_AST:
      k.test()
    return k.ret
