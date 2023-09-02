from __future__ import annotations
from typing import List, Tuple, Any, Optional, cast, DefaultDict, NamedTuple, Dict, Iterator, Union, Sequence, Final
import itertools, math, functools
from collections import defaultdict
from enum import Enum, auto

from tinygrad.helpers import colored, ImageDType, DEBUG, dtypes, DType, partition, prod, PtrDType
from tinygrad.ops import LazyOp, UnaryOps
from tinygrad.ops import ReduceOps, BinaryOps, TernaryOps
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, NumNode, Node, SumNode, MulNode, DivNode, ModNode, LtNode, AndNode, sym_rename
from tinygrad.codegen.optimizer import OptimizedKernel
from tinygrad.codegen.kernel import LocalBuffer
VariableOrNum = Union[Variable, NumNode, Node]

# bottom ones are asm only
class UOps(Enum):
  LOOP = auto(); ENDLOOP = auto() # loops can be global, local, or other # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # this defines buffers # noqa: E702
  LOAD = auto(); STORE = auto(); CONST = auto(); BARRIER = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto(); GEP = auto() # noqa: E702
  # TODO: add CONST. use ALU WHERE for gated load
  # *** assembly only UOps ***
  SPECIAL = auto(); LABEL = auto(); COND_BRANCH = auto() # TODO: replace these with LOOP and ENDLOOP # noqa: E702

def to_image_idx(base_shape:Tuple[int, ...], idxy:Node, valid:Node, validhacks=False) -> Tuple[Node, Node]:
  idy = (idxy//(4*base_shape[1]))
  if validhacks and valid.min == 0:
    idx = (idxy//4) + (idy*-base_shape[1])
    # find the ones in idx that didn't factorize and remove them (TODO: this is not universal)
    if isinstance(idx, SumNode):
      unfactored, idx_nodes = partition(idx.nodes, lambda x: isinstance(x, MulNode) and x.b == -base_shape[1])
      assert len(unfactored) <= 1
      idx = Variable.sum(idx_nodes)
      unfactored = (Variable.sum(unfactored) // base_shape[1])
      idy += unfactored
      # ugh really...handtuned garbage
      if idx.min >= (base_shape[1]*3)//4:
        idx -= base_shape[1]
        idy += 1
  else:
    idx = (idxy//4)%base_shape[1]
  if DEBUG >= 5: print("to_image_idx", base_shape, idx.min, idx.max, idy.min, idy.max, idx, idy)
  return idx, idy

def expand_idxs(idxs:Sequence[Node]) -> Iterator[Tuple[Node, ...]]:
  for x in itertools.product(*[idx.expand() for idx in idxs[::-1]]):
    yield x[::-1]

class MemOp(NamedTuple):
  name: str
  idx: Node
  local: bool
  memory_dtype: DType

  # shared
  valid: Node
  invalid_value: Union[float, int] = 0.0

class UOp(NamedTuple):
  uop: UOps
  dtype: Optional[DType]
  vin: Tuple[UOp, ...]
  arg: Any
  def __repr__(self): return f"{self.num:4d} {str(self.uop):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.num for x in self.vin]):32s} {self.arg}"
  #def __repr__(self): return f"{str(self.uop):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str(self.vin):32s} {self.arg}"

  # UOps are unique
  num: int
  def __hash__(self): return self.num
  def __eq__(self, x): return self.num == x.num


def get_grouped_dims(prefix, start_dim, local_dims, maxdim:int=0):
  local_idxs = loop_local_idxs = [Variable(f"{prefix}{start_dim+i}", 0, s-1) for i,s in enumerate(local_dims[0:maxdim-1] + (prod(local_dims[maxdim-1:]),) if len(local_dims) > maxdim else local_dims)]
  if maxdim != 0 and len(local_dims) > maxdim:
    dd = local_idxs[maxdim-1]
    nli = []
    for s in local_dims[maxdim-1:][::-1]:
      nli.append(dd % s)
      dd //= s
    local_idxs = local_idxs[0:maxdim-1] + nli[::-1]
  return local_idxs, loop_local_idxs

class Linearizer(OptimizedKernel):
  def get_buffer_name(self, i):
    if self.bufs[i].__class__ == LocalBuffer: return self.bufs[i].name
    assert self.bufs[i].realized.__class__ is not RawConst  # constants shouldn't be loaded with memops
    return self.arg_bufs[self.bufs[i].realized]

  def uop_alu_idx(self, a:UOp, b, ops, ctx:Linearizer, op, dtype=dtypes.int32):
    render_b:UOp = cast(UOp, (NumNode(b) if not isinstance(b, Node) else b).render(ops, ctx))
    return self.uop(UOps.ALU, dtype, (a, render_b), op, cachable=True)

  render_ops: Any = { Variable: lambda self, ops, ctx: ctx.uop(UOps.SPECIAL, dtypes.int32, tuple(), self),
                NumNode: lambda self, ops, ctx: ctx.uop(UOps.CONST, dtypes.int32, tuple(), self.b),
                MulNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MUL),
                DivNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.DIV),
                ModNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MOD),
                LtNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.CMPLT, dtype=dtypes.bool),
    SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.ADD), self.nodes[1:], self.nodes[0].render(ops,ctx)),
    AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.MUL, dtype=dtypes.bool), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

  def global_load(self, i:int, idxs:Sequence[VariableOrNum], acc=None) -> List[UOp]:
    const = self.bufs[i].realized._buf if isinstance(self.bufs[i].realized, RawConst) else acc

    expanded_nodes = [idx.expand() for idx in idxs]
    _idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    upcast_dim = self.get_upcast_dim(i)

    amt = 1
    if len(upcast_dim) == 1 and len(expanded_nodes[upcast_dim[0]]) in [4,2]:
      dim, amt = upcast_dim[0], len(expanded_nodes[upcast_dim[0]])

    ret = []
    invalid_value = 0 if dtypes.is_int(self.bufs[i].dtype) else 0.0
    for _idx in _idxs:
      if amt > 1:
        idx, valid = self.sts[i].expr_idxs((_idx[:dim] + (expanded_nodes[dim][0],) + _idx[dim+1:]))
        localtype = dtypes._float4 if amt == 4 else dtypes._float2
        if idx.render() != ((idx//amt)*amt).render():
          idx, valid = self.sts[i].expr_idxs(_idx)
          localtype = dtypes.float32
      else:
        idx, valid = self.sts[i].expr_idxs(_idx)
        localtype = dtypes.float32
      this_const, idx, valid = (invalid_value, Variable.num(0), Variable.num(1)) if valid.max == 0 else (const, idx, valid)
      key = f"{acc}{localtype}{this_const if this_const is not None and acc is None else self.get_buffer_name(i)}{idx.render()}{valid.render()}"
      if key not in self.load_cache:
        if isinstance(self.bufs[i].dtype, ImageDType): idx = to_image_idx(self.bufs[i].dtype.shape, idx, valid)
        if acc is not None:
          assert valid.min == 1
          self.load_cache[key] = self.uop(UOps.DEFINE_ACC, localtype, [], this_const)
        elif this_const is not None:
          self.load_cache[key] = self.uop(UOps.CONST, localtype, [], this_const)
          if valid.min == 0 and valid.max == 1:
            valid_rendered = valid.render(self.render_ops, self)
            alt = self.uop(UOps.CONST, localtype, [], invalid_value)
            self.load_cache[key] = self.uop(UOps.ALU, localtype, [valid_rendered, self.load_cache[key], alt], TernaryOps.WHERE)
        else:
          self.load_cache[key] = self.uop(UOps.LOAD, localtype, [], MemOp(self.get_buffer_name(i), idx, self.bufs[i].__class__ is LocalBuffer, self.bufs[i].dtype, valid, invalid_value))
      ret.append(self.uop(UOps.GEP, dtypes.float32, [self.load_cache[key]], expanded_nodes[dim].index(_idx[dim])) if localtype != dtypes.float else self.load_cache[key])
    return ret

  def global_store(self, i:int, idxs:List[VariableOrNum], store:List[UOp]) -> None:
    expanded_nodes = [idx.expand() for idx in idxs]
    _idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    store_offset = dict(zip(_idxs, store))

    # float4 grouping
    upcast_dim = self.get_upcast_dim(i)
    if len(upcast_dim) == 1 and len(expanded_nodes[upcast_dim[0]]) in [2,4]:
      grouped_store_offset = defaultdict(list)
      for k in store_offset:
        _idx = k[:upcast_dim[0]] + (expanded_nodes[upcast_dim[0]][0],) + k[upcast_dim[0]+1:]
        grouped_store_offset[_idx].append(store_offset[k])
      store_offset_new = {}
      for k,out_tokens in grouped_store_offset.items():
        amt = len(out_tokens)
        idx, valid = self.sts[i].expr_idxs(k)
        assert idx.render() == ((idx//amt)*amt).render(), "float4 stores are always aligned"
        assert valid.min == 1, "stores are always valid"
        store_offset_new[k] = self.uop(UOps.CAST, dtypes._float4 if amt == 4 else dtypes._float2, out_tokens)
      store_offset = store_offset_new

    for idx, var in store_offset.items():
      idx, valid = self.sts[i].expr_idxs(idx)
      if isinstance(self.bufs[i].dtype, ImageDType): idx = to_image_idx(self.bufs[i].dtype.shape, idx, valid)
      self.uop(UOps.STORE, None, [var], MemOp(self.get_buffer_name(i), idx, self.bufs[i].__class__ is LocalBuffer, self.bufs[i].dtype, valid))

  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)
  def linearize(self):
    self.process()

    # global uop cache
    self.saved_exprs: Dict[Tuple, UOp] = dict()

    # limit dims if we need to
    # TODO: broken, and doesn't really belong here
    #if self.opts.global_max and self.opts.local_max: self.limit_dims_to_max(self.opts.global_max, self.opts.local_max)

    # uops
    self.uops: List[UOp] = []

    # add global buffers
    for buf,name in self.arg_bufs.items():
      self.uop(UOps.DEFINE_GLOBAL, PtrDType(buf.dtype), [], (name, buf.dtype))
    # add variables from symbolic shapes
    for var in sorted(set(v for buf in self.ast.buffers for v in buf.st.var_vals), key=lambda k: k.key):
      self.uop(UOps.DEFINE_GLOBAL, dtypes.int32, [], (var.expr, dtypes._arg_int32))
    # define local buffers
    for lb in self.local_alias.values():
      self.uop(UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), [], (lb.name, self.sts[self.bufs.index(lb)].size()))
    # add a local buffer for multistage reduce. # TODO: use local alias
    if self.group_for_reduce:
      # TODO: the strides of this can be controlled
      self.sts.append(ShapeTracker(tuple([1] * self.first_reduce + self.group_for_reduce + [1] * (self.shape_len - self.upcasted - len(self.group_for_reduce) - self.first_reduce) + [x[0] for x in self.upcasted_axis(0)])))
      self.bufs.append(LocalBuffer("temp", self.sts[-1].size()))
      self.uop(UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), [], ("temp", self.sts[-1].size()))

    # print
    if DEBUG >= 3: self.printbufs()

    # kernel name (before late upcast)
    self.function_name = ("r_" if self.reduceop else "E_") + '_'.join([str(x) if isinstance(x, int) else sym_rename(x) for x in self.full_shape])
    self.display_name = ("r_" if self.reduceop else "E_") + colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

    # name the function something unique
    Linearizer.kernel_cnt[self.function_name] += 1
    suffix = f"{'n'+str(Linearizer.kernel_cnt[self.function_name]-1)}" if Linearizer.kernel_cnt[self.function_name] > 1 else ""
    self.function_name, self.display_name = self.function_name+suffix, self.display_name+colored(suffix, 'BLACK')

    # define indexes
    global_idxs, loop_global_idxs = get_grouped_dims("gidx", 0, self.full_shape[:self.global_dims], 3 if self.opts.has_local else 0)
    local_idxs, loop_local_idxs = get_grouped_dims("lidx", self.global_dims, self.full_shape[self.global_dims:self.first_reduce+len(self.group_for_reduce)], 3 if self.opts.has_local else 0)
    full_upcast_idxs = [Variable(None, 0, s-1) for s in self.full_shape[self.shape_len-self.upcasted:]]
    upcast_idxs = [Variable(None, 0, s-1) for s in self.output_shape[self.shape_len-self.upcasted:]]

    # global and local loops
    self.uop(UOps.LOOP, None, [], (loop_global_idxs, "global"))
    self.uop(UOps.LOOP, None, [], (loop_local_idxs, "local"))

    # parse AST
    loaded_buffers = {}
    acc = []
    self.load_cache: Dict[str, UOp] = {}

    # reduce op
    fake_reduce_idxs = []
    if self.reduceop is not None:
      # define indexes
      reduce_idxs = [Variable(f"ridx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce+len(self.group_for_reduce), self.shape_len-self.upcasted)]
      fake_reduce_idxs = [x*0 for x in reduce_idxs]

      # define accumulator
      acc = self.global_load(0, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, {ReduceOps.SUM: 0.0, ReduceOps.MAX: -math.inf}[cast(ReduceOps, self.reduceop.op)])

      # reduce loop
      self.uop(UOps.LOOP, None, [], (reduce_idxs, "reduce"))

      # barrier for fast GEMM
      if self.use_tensor_cores: self.uop(UOps.BARRIER, None, [], ())

      # compute local aliases
      # TODO: this is garbage code and should be at least moved elsewhere
      locals_to_store = []
      for i in self.local_alias:
        localbuf_idx = self.bufs.index(self.local_alias[i])
        strides = self.sts[i].real_strides()
        extra_locals = [lidx for lidx,st in zip(local_idxs[self.exclude_local_upcast:], strides[len(global_idxs)+self.exclude_local_upcast:self.first_reduce]) if st == 0]
        this_upcast_idxs: List[Node] = []
        # TODO: just flipping the order here is likely not generic at all
        for j,v in list(enumerate(full_upcast_idxs))[::-1] if self.reverse_upcast_dir else list(enumerate(full_upcast_idxs)):
          if strides[len(global_idxs)+len(local_idxs)+len(reduce_idxs)+j] == 0:
            if DEBUG >= 4: print(f"upcasting@{j} stride 0")
            this_upcast_idxs.append(Variable.num(0))
          elif (elc:=[el for el in extra_locals if v.min == el.min and v.max == el.max]):
            if DEBUG >= 4: print(f"upcasting@{j} matched stride {elc[0]}")
            this_upcast_idxs.append(elc[0])
            extra_locals.remove(elc[0])
          elif (elc:=[el for el in extra_locals if v.min == el.min and (v.max+1)%(el.max+1) == 0]):
            tacc = Variable.num(0)
            rem = v.max+1
            while len(elc) and rem%(elc[0].max+1) == 0:
              if DEBUG >= 4: print(f"upcasting@{j} partial stride {rem} {elc[0]} left: {elc[1:]}")
              rem = rem//(elc[0].max+1)
              tacc += (elc[0] * rem)
              extra_locals.remove(elc[0])
              elc = [el for el in extra_locals if v.min == el.min and rem%(el.max+1) == 0]
            if DEBUG >= 4 and rem > 1: print(f"failed upcasting@{j} partial stride {rem} extra locals {extra_locals}")
            this_upcast_idxs.append(tacc + Variable(None, 0, rem-1))
          else:
            if DEBUG >= 4: print(f"failed upcasting@{j} stride {v} extra locals {extra_locals}")
            this_upcast_idxs.append(v)
        idxs = global_idxs+local_idxs+reduce_idxs+(this_upcast_idxs[::-1] if self.reverse_upcast_dir else this_upcast_idxs)
        idxs = [idx*0 if s == 0 else idx for idx,s in zip(idxs,strides)]
        if DEBUG >= 3: print(f"{localbuf_idx} alias {i}:", idxs)
        ll = self.global_load(i, idxs)
        locals_to_store.append((localbuf_idx, idxs, ll))

      # copy in any global buffers
      if self.use_tensor_cores:
        if self.bufs[0].device == "METAL":
          i = 0
          for y0,y1 in zip(locals_to_store[1][2][::2], locals_to_store[1][2][1::2]):
            for x0,x1 in zip(locals_to_store[0][2][::2], locals_to_store[0][2][1::2]):
              self.uop(UOps.WMMA, None, [x0, x1, y0, y1, acc[i], acc[i+1]], "METAL")
              i += 2
        elif self.bufs[0].device == "HIP":
          i = 0
          for y in range(0, len(locals_to_store[1][2]), 0x10):
            for x in range(0, len(locals_to_store[0][2]), 0x10):
              self.uop(UOps.WMMA, None, acc[i:i+8]+locals_to_store[0][2][x:x+0x10]+locals_to_store[1][2][y:y+0x10], "HIP")
              i += 8
      else:
        if locals_to_store:
          self.uop(UOps.BARRIER, None, [], ())
          for i, idxs, ll in locals_to_store: self.global_store(i, idxs, ll)
          self.uop(UOps.BARRIER, None, [], ())

        # load earlybufs
        loaded_buffers.update({b:self.global_load(self.bufs.index(self.local_alias[i]) if i in self.local_alias else i, global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs[1:], start=1) if b in self.earlybufs})

        # run early AST (with reduce)
        self.ast_parse(self.reduceop, [acc[off] for off in self.acc_offsets(self.full_buf_index)], loaded_buffers, do_reduce=True)

      # end the reduce loop
      self.uop(UOps.ENDLOOP, None, [], (reduce_idxs, "reduce"))
      self.load_cache.clear()

      # end the local loop, do the local reduce
      if self.group_for_reduce:
        fake_global_idxs = [x*0 for x in global_idxs]
        self.global_store(-1, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, acc)  # store accumulators
        self.uop(UOps.BARRIER, None, [], ())
        self.uop(UOps.ENDLOOP, None, [], (loop_local_idxs, "local"))

        # local indexs are over, 0 them out
        local_idxs = [x*0 for x in local_idxs]

        # if any group_for_reduce items aren't reduces, upcast them here
        for j in self.upcast_in_mid_reduce_axes:
          self.reshape_and_permute(None, [i for i in range(self.shape_len) if i != j] + [j])
          self.upcast()
          self.group_for_reduce.pop()
          local_idxs = local_idxs[:-1]
          # regenerate upcast_idxs
          upcast_idxs = [Variable(None, 0, s-1) for s in self.output_shape[self.shape_len-self.upcasted:]]

        # NOTE: this structure is the same as the reduce op above

        # define late accumulator
        acc = self.global_load(-1, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, {ReduceOps.SUM: 0.0, ReduceOps.MAX: -math.inf}[cast(ReduceOps, self.reduceop.op)])

        # late reduce loop
        end_local_idxs = [Variable(f"tidx{i}", 0, self.full_shape[i]-1 if i >= self.first_reduce else 0) for i in range(0, self.first_reduce+len(self.group_for_reduce))]
        self.uop(UOps.LOOP, None, [], (end_local_idxs, "late_reduce"))

        # load localbufs
        loaded_buffers["LOCAL_BUFFER"] = self.global_load(-1, end_local_idxs+fake_reduce_idxs+upcast_idxs)

        # there's no AST here (and there's no shape for the reduce LazyOp)
        self.ast_parse(LazyOp(self.reduceop.op, ("LOCAL_BUFFER",)), [acc[off] for off in self.acc_offsets(-1)], loaded_buffers, do_reduce=True) # type: ignore

        # end the late reduce loop
        self.uop(UOps.ENDLOOP, None, [], (end_local_idxs, "late_reduce"))
        self.load_cache.clear()

    # load latebufs
    loaded_buffers.update({b:self.global_load(i, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs) for i,b in enumerate(self.bufs) if b not in self.earlybufs and i != 0 and b.__class__ is not LocalBuffer})

    # run late AST
    val = self.ast_parse(self.ast, acc, loaded_buffers)

    # store
    self.global_store(0, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, val)

    # end the global (and maybe local) loop
    self.uop(UOps.ENDLOOP, None, [], (loop_global_idxs+loop_local_idxs, "global+local") if not self.group_for_reduce else (loop_global_idxs, "global"))

    return self

  def uop(self, uop:UOps, dtype:Optional[DType], vin:Union[Tuple[UOp, ...], List[UOp]], arg:Any=None, cachable=False) -> UOp:
    key = (uop, dtype, tuple(vin), arg)
    if cachable and key in self.saved_exprs: return self.saved_exprs[key]
    self.uops.append(UOp(uop, dtype, tuple(vin), arg, len(self.uops)))
    if DEBUG >= 4: print(self.uops[-1])
    if cachable: self.saved_exprs[key] = self.uops[-1]
    return self.uops[-1]

  def ast_parse(self, x, acc, loaded_buffers, do_reduce=False) -> List[UOp]:
    if x.__class__ is not LazyOp: return loaded_buffers[x]
    if x.op in [UnaryOps.NOOP, UnaryOps.CAST]: return self.ast_parse(x.src[0], acc, loaded_buffers)  # cast isn't an ALU op
    if x.op in ReduceOps and not do_reduce: return acc
    # MULACC fusion. TODO: this is copied from Interpreted
    if x.op == ReduceOps.SUM and x.src[0].__class__ is LazyOp and x.src[0].op == BinaryOps.MUL:
      x = LazyOp(TernaryOps.MULACC, x.src[0].src, x.arg)
    if x.op == ReduceOps.SUM and x.src[0].__class__ is LazyOp and x.src[0].op == UnaryOps.CAST and x.src[0].src[0].__class__ is LazyOp and x.src[0].src[0].op == BinaryOps.MUL:
      x = LazyOp(TernaryOps.MULACC, x.src[0].src[0].src, x.arg)
    values = [self.ast_parse(v, acc, loaded_buffers) for v in x.src]
    ops = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX, TernaryOps.MULACC:TernaryOps.MULACC}
    if x.op in ops:
      ret = [(idx, self.uop(UOps.STORE, dtypes.float32, [val[-1], self.uop(UOps.ALU, dtypes.float32, list(val), ops[x.op])])) for idx, val in zip([[i] for i in range(len(values[0]))], zip(*values, acc))]
    else:
      ret = [(idx, self.uop(UOps.ALU, dtypes.float32, list(val), x.op, cachable=True)) for idx, val in zip([[i] for i in range(len(values[0]))], zip(*values))]
    ordered_ret: List[Optional[UOp]] = [None]*len(values[0])
    # scatter
    for i,j in ret:
      for k in i:
        ordered_ret[k] = j
    assert all(isinstance(x, UOp) for x in ordered_ret), "some tokens didn't get scattered?"
    return cast(List[UOp], ordered_ret)
