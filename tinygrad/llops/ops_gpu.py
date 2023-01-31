from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Set
from tinygrad.helpers import prod
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, MovementOps, LazyOp, Op, ExplicitExecAST, GlobalCounters
from tinygrad.ast import ASTKernel, Token, Types
from tinygrad.lazy import IMAGE
from tinygrad.shape import ShapeTracker
from tinygrad.shape.symbolic import ModNode   # this will go away when VALIDHACKS does
from tinygrad.helpers import getenv

CUDA = getenv("CUDA", 0)
if not CUDA: from tinygrad.runtime.opencl import CLBuffer, CLImage, CLProgram, CL   # NOTE: using CL will not work for the CUDA runtime # noqa: F401
else: from tinygrad.runtime.cuda import CLBuffer, CLImage, CLProgram  # type: ignore

VALIDHACKS = getenv("VALIDHACKS", 0)    # TODO: remove the need for this
NATIVE_EXPLOG = getenv("NATIVE_EXPLOG", 0)  # this is needed as a switch for the tests to pass

KOPT = getenv("KOPT", 0)
PRINT_AST = getenv("PRINT_AST", "0")
TEST_AST = getenv("TEST_AST", 0)

def group_float4(x):
  assert all(y.typ == Types.FLOAT for y in x) and len(x)%4 == 0
  return [Token(f"(float4)({','.join([x[i+j].tok for j in range(4)])})", Types.FLOAT4) for i in range(0, len(x), 4)]
def split_float4(x):
  assert all(y.typ == Types.FLOAT4 for y in x)
  return sum([[Token(acc.tok+f".s{s}", Types.FLOAT) for s in range(4)] for acc in x], [])

class CLASTKernel(ASTKernel):
  code_for_op : Dict[Op, str] = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)",
    UnaryOps.GT0:  "(A > 0.)" if CUDA else "((float)1.-step((float)0.,(-A)))",
    UnaryOps.EXP: "native_exp(A)" if NATIVE_EXPLOG else "exp(A)",
    UnaryOps.LOG: "native_log(A)" if NATIVE_EXPLOG else "log(A)",
    UnaryOps.RECIPROCAL: "native_recip(A)" if NATIVE_EXPLOG else "((float)1.0/A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
    BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
    ReduceOps.SUM: "A+=B", ReduceOps.MAX: "A=max(A,B)"
  }
  start_for_op = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def image_idx(self, buf_index, idxy, validhacks=False):
    assert self.buftokens[buf_index].typ == Types.FLOAT4, f"image must be FLOAT4 {self.buftokens[buf_index]} {self.bufs[buf_index].st}"
    idx = (idxy//4)%self.bufs[buf_index]._base_shape[1]
    idy = (idxy//(4*self.bufs[buf_index]._base_shape[1]))%self.bufs[buf_index]._base_shape[0]
    if validhacks: idx, idy = [x.a if isinstance(x, ModNode) and x.max < x.b*2 else x for x in (idx, idy)]
    return f"(int2)({idx.cl}, {idy.cl})"

  def store(self, buf_index, value:List[Token]):
    if len(value) == self.buftokens[buf_index].size()*4: value = group_float4(value)
    if len(value)*4 == self.buftokens[buf_index].size(): value = split_float4(value)
    assert len(value) == self.buftokens[buf_index].size(), f"size mismatch {len(value)} != {self.buftokens[buf_index].size()}"
    for v, o in zip(value, self.buftokens[buf_index].offsets()):
      idxy, valid = self.sts[buf_index].expr_idxs(o)
      assert str(valid) == "1", "store must always be valid"
      assert self.buftokens[buf_index].typ == v.typ, f"buf must be {v.typ}"
      if isinstance(self.bufs[buf_index]._buf, CLImage):
        self.kernel.append(f"write_imagef(data{buf_index}, {self.image_idx(buf_index, idxy)}, {v.tok});  /* {self.bufs[buf_index]._base_shape} */\n")
      else:
        self.kernel.append(f"data{buf_index}[{(idxy//(4 if v.typ == Types.FLOAT4 else 1)).cl}] = {v.tok};\n")

  def load(self, buf_index:int) -> List[Token]:
    # constant folding
    const = None
    if self.bufs[buf_index]._base_shape == (1,) and self.bufs[buf_index]._backing is not None:
      assert self.buftokens[buf_index].typ == Types.FLOAT
      self.bufs_to_delete.add(buf_index)
      const = Token(f"({self.bufs[buf_index]._backing[0]}f)", self.buftokens[buf_index].typ)

    tokens = []
    for o in self.buftokens[buf_index].offsets():
      key = f"val{buf_index}_{o}" if o >= 0 else f"val{buf_index}_m{-o}"
      if (buf_index, o) not in self.loaded_keys:
        idxy, valid = self.sts[buf_index].expr_idxs(o)
        if const is not None:
          ldr = const
        elif isinstance(self.bufs[buf_index]._buf, CLImage):
          ldr = Token(f"read_imagef({self.buftokens[buf_index].tok}, smp, {self.image_idx(buf_index, idxy, VALIDHACKS)}) /* {self.bufs[buf_index]._base_shape} */", Types.FLOAT4)
        else:
          ldr = Token(f"{self.buftokens[buf_index].tok}[{(idxy//(4 if self.buftokens[buf_index].typ == Types.FLOAT4 else 1)).cl}]", self.buftokens[buf_index].typ)
        ldr = ldr if str(valid) == "1" or (VALIDHACKS and isinstance(self.bufs[buf_index]._buf, CLImage)) else Token(f"({valid.cl} ? {ldr.tok} : 0.0f)", ldr.typ)
        if const is not None:
          self.loaded_keys[(buf_index,o)] = ldr
        else:
          self.kernel.append(f"{ldr.decltype()} {key} = {ldr.tok};\n")
          self.loaded_keys[(buf_index,o)] = Token(key, ldr.typ)
      tokens.append(self.loaded_keys[(buf_index,o)])
    return tokens

  def ast_parse(self, x:Union[GPUBuffer, LazyOp], acc:List[Token], do_reduce=False) -> List[Token]:
    if not isinstance(x, LazyOp): return self.load(self.bufs.index(x))
    if isinstance(x.op, ReduceOps) and not do_reduce: return acc
    values = ([acc] if isinstance(x.op, ReduceOps) else []) + [self.ast_parse(v, acc, do_reduce) for v in x.src]
    code = CLASTKernel.code_for_op[x.op]  # TODO: replace this with a function
    if len(values) == 2:
      # TODO: sometimes this is split, sometimes it's multiply
      if isinstance(x.op, ReduceOps) and values[0][0].typ == Types.FLOAT4 and len(values[0])*4 == len(values[1]): values[0] = split_float4(values[0])
      if values[0][0].typ != values[1][0].typ:
        if isinstance(x.op, ReduceOps):
          if x.op == ReduceOps.SUM: self.prekernel.add("float clreduce(float4 x) { return x.x + x.y + x.z + x.w; }\n")
          elif x.op == ReduceOps.MAX: self.prekernel.add("float clreduce(float4 x) { return max(max(x.x, x.y), max(x.z, x.w)); }\n")
          values[1] = [Token(f"clreduce({x.tok})", Types.FLOAT) for x in values[1]]
        elif values[0][0].typ == Types.FLOAT: values[0] = group_float4(values[0])
        elif values[1][0].typ == Types.FLOAT: values[1] = group_float4(values[1])
      assert len(values[0]) == len(values[1]), f"values mismatch {values}"
      return [Token(code.replace("A", a.tok).replace("B", b.tok), a.typ) for a,b in zip(values[0], values[1])]
    else:
      return [Token(code.replace("A", a.tok), a.typ) for a in values[0]]

  def hand_coded_optimizations(self):
    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    # shove the axis to the end and remove 
    if any(isinstance(buf._buf, CLImage) for buf in self.earlybufs):
      eb_valids = [True] * self.shape_len
      for i in range(len(self.bufs)):
        if isinstance(self.bufs[i]._buf, CLImage) and self.bufs[i] in self.earlybufs:
          valids = [self.sts[i].shape[j]%4 == 0 and self.sts[i].views[-1].strides[j] == 1 for j in range(self.shape_len)]
          eb_valids = [x and y for x,y in zip(eb_valids, valids)]
      assert any(eb_valids), f"invalid op with images {eb_valids}"
      eb_valid = eb_valids.index(True)
      if DEBUG >= 3: print(f"early merging axis {eb_valid} from {eb_valids}")

      # no change, we added a dimension
      self.reshape_and_permute(
        lambda x: list(x[0:eb_valid]) + ([x[eb_valid]//4, 4] if x[eb_valid] > 1 else [1,1]) + list(x[eb_valid+1:]),
        [i for i in range(self.shape_len+1) if i != eb_valid+1] + [eb_valid+1])

      # drop the last dimension
      self.upcast()

    # simplify (sets first_reduce)
    self.simplify_ones()

    # are we grouping?
    if self.buftokens[0].typ != Types.FLOAT4 and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
      for sz in ([256, 16] if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
        if all([st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts]):
          self.group_for_reduce.append(sz)
          break

    # TODO: this makes re_S3_32_3_3 at least 10x faster
    #if self.first_reduce == 4 and self.shape_len == 7:
      #self.group_for_reduce.append(112//2)
      # TODO: this shouldn't have to be permuted
      #self.reshape_and_permute(None, [0,1,2,3,6,4,5])

    # if there's images in the latebufs, we have to make an axis the 4 storing one. this affects the kernel shape
    self.upcast_in_mid_reduce = False
    if any(isinstance(buf._buf, CLImage) for buf in self.bufs if buf not in self.earlybufs) and self.buftokens[0].typ != Types.FLOAT4:
      lb_valids = [True] * self.shape_len
      for i in range(len(self.bufs)):
        valids = [self.sts[i].shape[j]%4 == 0 and (self.sts[i].views[-1].strides[j] == 1 or not isinstance(self.bufs[i]._buf, CLImage) or self.bufs[i] in self.earlybufs) for j in range(self.shape_len)]
        lb_valids = [x and y for x,y in zip(lb_valids, valids)]
      assert any(lb_valids), f"invalid op with images {lb_valids}"
      lb_valid = lb_valids.index(True)
      assert lb_valid < self.first_reduce, f"can't be in the reduce {lb_valid}"
      if DEBUG >= 3: print(f"late merging axis {lb_valid} from {lb_valids}")

      # no change, we added a dimension
      self.reshape_and_permute(
        lambda x: list(x[0:lb_valid]) + [x[lb_valid]//4, 4] + list(x[lb_valid+1:]),
        [i for i in range(self.shape_len+1) if i != lb_valid+1] + [lb_valid+1])

      if self.group_for_reduce and self.first_reduce <= 2:
        self.upcast_in_mid_reduce = True
        self.group_for_reduce.append(4)
      else:
        # drop the last dimension
        self.upcast()

    # simplify (sets first_reduce)
    self.simplify_ones()

    # split to 4 float4s
    if self.buftokens[0].typ == Types.FLOAT4 and any(isinstance(buf._buf, CLImage) for buf in self.earlybufs) and prod(self.sts[0].shape[:self.first_reduce]) >= 2048 and not self.group_for_reduce:
      xb_choices = []
      for i in range(self.first_reduce):
        if all(st.shape[i]%4 == 0 for st in self.sts):
          xb_choices.append((sum(st.views[-1].strides[i]>0 for st in self.sts), sum(st.views[-1].strides[i] for st in self.sts), i))

      if len(xb_choices):
        xb_choice = sorted(xb_choices)[0][2]
        if DEBUG >= 3: print(f"float4 merging axis {xb_choice} : {xb_choices}")

        # this leaves the last axis in place
        self.reshape_and_permute(
          lambda x: list(x[0:xb_choice]) + [x[xb_choice]//4, 4] + list(x[xb_choice+1:]),
          [i for i in range(self.shape_len+1) if i != xb_choice+1] + [xb_choice+1])

        # drop the last dimension
        self.upcast()

        # re-simplify
        self.simplify_ones()

    # use more opencl indexing
    if self.first_reduce == 2 and isinstance(self.bufs[0]._buf, CLImage):
      base_shape = self.bufs[0]._base_shape
      if all([(base_shape[0]*base_shape[1])%st.shape[0] == 0 and st.shape[0]//base_shape[0] != 0 for st in self.sts]):
        if DEBUG >= 3: print("split opencl", base_shape, self.sts[0].shape)
        self.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
        self.simplify_ones()

    # group for reduce
    if len(self.group_for_reduce):
      # with permute for memory coalesing
      if len(self.group_for_reduce) == 2:
        permute_axis = list(range(0, self.first_reduce)) + [self.first_reduce+1, self.shape_len, self.first_reduce] + list(range(self.first_reduce+2, self.shape_len))
      else:
        permute_axis = list(range(0, self.first_reduce)) + [self.first_reduce+1, self.first_reduce] + list(range(self.first_reduce+2, self.shape_len+1))
      self.reshape_and_permute(lambda x: list(x[0:self.first_reduce]) + [max(1, x[self.first_reduce]//self.group_for_reduce[0]), min(x[self.first_reduce], self.group_for_reduce[0])] + list(x[self.first_reduce+1:]), permute_axis)

    # if last dim <= 3 and it's a reduce dim, upcast (loop unrolling)
    end_dimension = max([st.shape[-1] for st in self.sts])
    if self.first_reduce < self.shape_len and end_dimension > 1 and end_dimension <= 3 and max([x.size() for i,x in enumerate(self.buftokens) if self.bufs[i] in self.earlybufs]) <= 4:
      self.upcast()

  def printbufs(self, prefix=""):
    print(f"first_reduce: {self.first_reduce} shape_len: {self.shape_len} group_for_reduce: {self.group_for_reduce}")
    for i in range(len(self.sts)):
      print(prefix, self.buftokens[i], f"early:{'T' if i < len(self.bufs) and self.bufs[i] in self.earlybufs else 'F'} image:{'T' if i < len(self.bufs) and isinstance(self.bufs[i]._buf, CLImage) else 'F'}", self.sts[i].shape, self.sts[i].views[-1].strides)

  # STOP WASTING TIME WITH DOING THE RESHAPES AND PERMUTES BY HAND. KERNEL SEARCH IS THE ONLY WAY IT WILL EVER BE GOOD
  # group_for_reduce will have to be better first
  def codegen(self):
    self.hand_coded_optimizations()

    # add a local buffer for multistage reduce
    if len(self.group_for_reduce):
      self.sts.append(ShapeTracker(tuple([1] * self.first_reduce + self.group_for_reduce + [1] * (self.shape_len - len(self.group_for_reduce) - self.first_reduce))))
      self.buftokens.append(Token("temp", Types.FLOAT, ptr=True))

    self.output_shape = list(self.sts[0].shape[:self.first_reduce]) + self.group_for_reduce
    if DEBUG >= 3:
      print("output shape", self.output_shape)
      self.printbufs("new:")

    self.bufs_to_delete : Set[int] = set()
    self.loaded_keys : Dict[Tuple[int,int], Token] = {}

    self.prekernel : Set[str] = set()
    self.kernel : List[str] = ["const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"] if any(isinstance(buf._buf, CLImage) for buf in self.bufs) else []

    # output_shape[-1] is get_global_id(0)
    MAX_OUTPUT_SHAPE = 3
    self.kernel += [f"int idx{len(self.output_shape)-1-i} = {f'blockDim.{chr(120+i)}*blockIdx.{chr(120+i)}+threadIdx.{chr(120+i)}' if CUDA else f'get_global_id({i})'}; /* {self.output_shape[-1-i]} */\n" for i in range(min(MAX_OUTPUT_SHAPE, len(self.output_shape))) if self.output_shape[-1-i] != 1]
    if len(self.output_shape) > MAX_OUTPUT_SHAPE:
      # sometimes, there's more dimensions. compact all the dimensions into the first one
      # TODO: these compactions should be searchable
      final_dimension = len(self.output_shape)-MAX_OUTPUT_SHAPE
      for i in range(final_dimension-1, -1, -1):
        self.kernel += [f"int idx{i} = idx{final_dimension} % {self.output_shape[i]};", f"idx{final_dimension} = idx{final_dimension} / {self.output_shape[i]};\n"]
      self.output_shape = [prod(self.output_shape[0:final_dimension+1])] + list(self.output_shape[final_dimension+1:])
      if DEBUG >= 3: print(f"replaced output shape with {self.output_shape}")

    # early ast
    accumulators : List[Token] = [Token("acc%d" % i, self.buftokens[0].typ) for i in range(self.buftokens[0].size())]
    if self.reduceop:
      full_shape = [x.shape for x in self.sts if x.shape != self.sts[0].shape]
      full_shape = self.sts[0].shape if len(full_shape) == 0 else full_shape[0]

      acc_offsets = self.buftokens[self.bufs.index(self.earlybufs[0])].acc_offsets()
      self.kernel += [f"{accumulator.decltype()} {accumulator.tok} = {CLASTKernel.start_for_op[self.reduceop.op]};\n" for accumulator in accumulators]
      self.kernel += [f"for (int idx{i} = 0; idx{i} < {full_shape[i]}; idx{i}++) {{\n" for i in range(self.first_reduce+len(self.group_for_reduce), self.shape_len)]
      self.kernel += [f"{x.tok};\n" for x in self.ast_parse(self.reduceop, [accumulators[off] for off in acc_offsets], do_reduce=True)] + ["}\n"] * (self.shape_len - (self.first_reduce + len(self.group_for_reduce)))
    
    # middle
    if self.group_for_reduce:
      lidx, lvalid = self.sts[-1].expr_idxs()
      assert str(lvalid) == "1", "local buffer must be valid"
      self.kernel.append(f"int mid_idx = {lidx.cl};")
      for i,acc in enumerate(accumulators):
        self.kernel.append(("__shared__ " if CUDA else "__local ") + f"{acc.decltype()} {self.buftokens[-1].tok}{i}[{prod(self.group_for_reduce)}];  // second stage\n")
        self.kernel.append(f"{self.buftokens[-1].tok}{i}[mid_idx] = {acc.tok};\n")
      self.kernel.append("barrier(CLK_LOCAL_MEM_FENCE);\n" if not CUDA else "__syncthreads();\n")

      if self.upcast_in_mid_reduce:
        assert len(self.group_for_reduce) == 2
        # it should be the last dimension
        self.reshape_and_permute(None, [i for i in range(self.shape_len) if i != self.first_reduce+1] + [self.first_reduce+1])
        self.upcast()

      self.kernel.append("if (mid_idx == 0) {\n")
      new_accumulators = [Token(f"output{i}", self.buftokens[0].typ) for i in range(len(accumulators))]
      for i,acc in enumerate(new_accumulators):
        self.kernel.append(f"{acc.decltype()} {acc.tok} = 0.0;\n")
        if self.upcast_in_mid_reduce:
          self.kernel.append(f"for (int mid = 0; mid < {prod(self.group_for_reduce)//4}; mid++) {{ {CLASTKernel.code_for_op[self.reduceop.op].replace('A', acc.tok).replace('B', f'vload4(0, &temp{i}[mid*4])')}; }}\n")
        else:
          self.kernel.append(f"for (int mid = 0; mid < {prod(self.group_for_reduce)}; mid++) {{ {CLASTKernel.code_for_op[self.reduceop.op].replace('A', acc.tok).replace('B', f'temp{i}[mid]')}; }}\n")
      accumulators = new_accumulators
    
    # late ast
    self.store(0, self.ast_parse(self.ast, accumulators))
    if self.group_for_reduce: self.kernel.append("}")
    self.kernel.append("}")

    # kernel function definition
    function_name = ("re_S" if self.reduceop else "ew_S") + '_'.join([str(x) for x in self.bufs[0].shape if x != 1])
    buftypes = [f"{'read_only' if i > 0 else 'write_only'} image2d_t" if isinstance(x._buf, CLImage) else ("__global "+self.buftokens[i].decltype()) for i,x in enumerate(self.bufs)] if not CUDA else [self.buftokens[i].decltype() for i,x in enumerate(self.bufs)]
    self.kernel = list(self.prekernel) + [f"{'__global__' if CUDA else '__kernel'} void {function_name}(",] + \
      [', '.join([f'{t} data{i}' for i,t in enumerate(buftypes) if i not in self.bufs_to_delete])] + \
      [") {\n"] + self.kernel

    # compile kernel
    self.fxn = CLProgram(function_name, ' '.join(self.kernel), op_estimate=self.info.flops)
    mem_estimate = sum(prod(x._base_shape) for x in self.bufs)

    if DEBUG >= 3 and len(self.bufs_to_delete): print(f"deleting buffers {self.bufs_to_delete}")
    def runner(*bufs):
      GlobalCounters.global_ops += self.info.flops
      GlobalCounters.global_mem += mem_estimate
      clbufs = [x.cl for i,x in enumerate(bufs) if i not in self.bufs_to_delete]
      return self.fxn(self.output_shape[::-1] if len(self.output_shape) > 0 else [1], (self.group_for_reduce[::-1] + [1]*(len(self.output_shape)-len(self.group_for_reduce))) if self.group_for_reduce else None, *clbufs)
    return runner

  def print(self):
    super().print()
    for i in range(len(self.bufs)):
      print(self.buftokens[i], self.bufs[i] in self.earlybufs, self.sts[i])
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
      self._buf.copyin(self._backing)
      self._backing = None
    return self._buf.cl

  def __repr__(self): return f"GPUBuffer(shape={self.st}, hostbuf=GPUBuffer(shape={self._base_shape}" + (f", backing=np.array({self._backing}, dtype=np.float32)))" if self._backing else ", force_create=True))")

  @staticmethod
  def fromCPU(x): return GPUBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())

  def toCPU(self) -> np.ndarray:
    cl_buf = self.unary_op(UnaryOps.NOOP) if not self.st.contiguous or prod(self._base_shape) != prod(self.shape) else self
    cl_buf = cl_buf if isinstance(cl_buf._buf, CLBuffer) else self.movement_op(MovementOps.RESHAPE, list(self.shape)+[1]).unary_op(UnaryOps.NOOP)
    assert prod(cl_buf._base_shape) == prod(self.shape), f"shape product mismatch {cl_buf._base_shape} vs {self.shape}"
    data = np.empty(self.shape, dtype=np.float32)
    cl_buf._buf.copyout(data)
    return data

  @classmethod
  def exec_ast(cls, ast:LazyOp):
    k = CLASTKernel(ast)
    if KOPT:
      from extra.kernel_search import apply_optimization
      apply_optimization(k, ast, max_interventions=KOPT)
    k.codegen()(*k.bufs)
    if PRINT_AST == "1" or (hasattr(k, "fxn") and PRINT_AST == k.fxn.name):
      print(k.fxn.name)
      k.print()
    if TEST_AST:
      from extra.lib_test_ast import test_ast  # type: ignore
      test_ast(k)
    return k.ret
