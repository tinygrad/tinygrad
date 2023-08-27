from typing import List, Tuple, Any, Optional, cast, DefaultDict, NamedTuple, TypeVar, Dict, Iterator, Union, Sequence, Final
import itertools, math
from collections import defaultdict
from enum import Enum, auto

from tinygrad.helpers import colored, ImageDType, DEBUG, dtypes, mnum, DType, all_same, partition
from tinygrad.ops import LazyOp, UnaryOps, Op
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import ReduceOps, BinaryOps, TernaryOps
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, NumNode, Node, SumNode, MulNode, sym_rename
from tinygrad.codegen.optimizer import OptimizedKernel
from tinygrad.codegen.kernel import LocalBuffer, LinearizerOptions # noqa: F401 # pylint:disable=unused-import
VariableOrNum = Union[Variable, NumNode, Node]

# bottom ones are asm only
class UOps(Enum):
  LOOP = auto(); ENDLOOP = auto() # loops can be global, local, or other # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto() # this defines buffers # noqa: E702
  LOAD = auto(); STORE = auto(); BARRIER = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto() # noqa: E702
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

class Token(NamedTuple):
  name: str
  dtype: DType
  offset: Optional[int] = None
  def render(self, with_type=False):
    if with_type:
      assert self.offset is None
      return f"{self.dtype.name} {self.name}"
    if self.offset is None: return self.name
    assert self.dtype in [dtypes._float4, dtypes._float2], f"{self.dtype} isn't okay with offset {self.offset}"
    return self.name+"."+"xyzw"[int(self.offset)]
  def __repr__(self): return f"<{self.name}>" if self.offset is None and self.dtype == dtypes.float32 else f"<{self.name}:{self.dtype.name}:{self.offset}>"

# TODO: the next three functions are poorly written
def get_grouped_float4_idxs(acc:List[Token]) -> Optional[List[int]]:
  idxs: Optional[List[int]] = []
  for i,a in enumerate(acc):
    if idxs is None: break
    if i in idxs: continue
    if a.dtype.sz > 1 and a.offset == 0:
      idxs.append(i)
      friends: List[int] = []
      for j,b in enumerate(acc):
        if len(friends) == 3: break
        if j in idxs: continue
        if a.name == b.name and b.dtype.sz > 1 and b.offset == len(friends)+1:
          friends.append(j)
      if len(friends) == 3: idxs += friends
      else: idxs = None
    else:
      idxs = None
  return idxs

def to_float4(x:List[Token]) -> Optional[Token]:
  if all_same(x): return x[0]
  if all_same([y.name for y in x]) and all(y.dtype == dtypes._float4 and y.offset == i for i,y in enumerate(x)):
    return Token(x[0].name, dtypes._float4)
  return None

def get_grouped_maybe_float4(*values:List[Token], grouping_allowed=True):
  assert all_same([len(x) for x in values]), f"all values are not the same length {values}"
  # these use accumulators, we can only fold if the acc is a float4
  idxs = get_grouped_float4_idxs(values[-1]) if grouping_allowed else None
  if idxs is not None:
    new_idxs = []
    new_values = []
    for i in range(0, len(idxs), 4):
      nv = [to_float4([v[j] for j in idxs[i:i+4]]) for v in values]
      if any(x is None for x in nv): break
      new_idxs.append(idxs[i:i+4])
      new_values.append(nv)
    if len(new_values) == len(idxs)//4:
      return zip(new_idxs, new_values)
  return zip([[i] for i in range(len(values[0]))], zip(*values))

# TODO: generic visitor pattern?
def expand_node(idx:Node) -> List[Node]:
  if isinstance(idx, Variable): return [idx] if idx.expr is not None else [Variable.num(j) for j in range(idx.min, idx.max+1)]
  if isinstance(idx, NumNode): return [idx]
  if isinstance(idx, MulNode): return [x*idx.b for x in expand_node(idx.a)]
  if isinstance(idx, SumNode): return [Variable.sum(list(it)) for it in itertools.product(*[expand_node(x) for x in idx.nodes])]
  raise NotImplementedError(idx)

def expand_idxs(idxs:Sequence[Node]) -> Iterator[Tuple[Node, ...]]:
  for x in itertools.product(*[expand_node(idx) for idx in idxs[::-1]]):
    yield x[::-1]

class MemOp(NamedTuple):
  name: str
  idx: Node
  local: bool
  memory_dtype: DType

  # shared
  valid: Node
  invalid_value: Union[float, int] = 0.0

class ConstOp(NamedTuple):
  value: Union[float, int]

  # shared
  valid: Node
  invalid_value: Union[float, int] = 0.0

class UOp(NamedTuple):
  uop: UOps
  out: Optional[Token]
  vin: List[Token]
  arg: Any
  def __repr__(self): return f"{str(self.uop):20s}: {str(self.out) if self.out is not None else '':25s} {str(self.vin):32s} {self.arg}"

class Linearizer(OptimizedKernel):
  def get_buffer_name(self, i):
    if self.bufs[i].__class__ == LocalBuffer: return self.bufs[i].name
    assert self.bufs[i].realized.__class__ is not RawConst  # constants shouldn't be loaded with memops
    return self.arg_bufs[self.bufs[i].realized]

  def global_load(self, i:int, idxs:Sequence[VariableOrNum], acc=None) -> List[Token]:
    const = self.bufs[i].realized._buf if isinstance(self.bufs[i].realized, RawConst) else acc

    expanded_nodes = [expand_node(idx) for idx in idxs]
    _idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    upcast_dim = self.get_upcast_dim(i)

    amt = 1
    if len(upcast_dim) == 1 and len(expanded_nodes[upcast_dim[0]]) in [4,2]:
      dim, amt = upcast_dim[0], len(expanded_nodes[upcast_dim[0]])

    ret = []
    invalid_value = 0 if dtypes.is_int(self.bufs[i].dtype) else 0.0
    for load_i, _idx in enumerate(_idxs):
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
        self.load_cache[key] = self.uop(UOps.LOAD, Token(f"val{mnum(i)}_{load_i}", localtype), [], MemOp(self.get_buffer_name(i), idx, self.bufs[i].__class__ is LocalBuffer, self.bufs[i].dtype, valid, invalid_value)) if this_const is None else \
                               self.uop(UOps.LOAD, Token(f"{'const' if acc is None else 'acc'}{mnum(i)}_{load_i}", localtype), [], ConstOp(this_const, valid))
      ret.append(Token(self.load_cache[key].name, self.load_cache[key].dtype, expanded_nodes[dim].index(_idx[dim])) if localtype != dtypes.float else self.load_cache[key])
    return ret

  def global_store(self, i, idxs:List[VariableOrNum], store:List[Token], ssa) -> None:
    expanded_nodes = [expand_node(idx) for idx in idxs]
    _idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    upcast_dim = self.get_upcast_dim(i)

    store_offset = dict(zip(_idxs, store))

    # float4 grouping
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
        if all_same([x.name for x in out_tokens]) and tuple(range(amt)) == tuple(x.offset for x in out_tokens):
          store_offset_new[k] = Token(out_tokens[0].name, dtypes._float4 if amt == 4 else dtypes._float2)
        else:
          store_offset_new[k] = self.uop(UOps.CAST, ssa("alu", dtypes._float4 if amt == 4 else dtypes._float2), out_tokens)
      store_offset = store_offset_new

    for idx, var in store_offset.items():
      idx, valid = self.sts[i].expr_idxs(idx)
      if isinstance(self.bufs[i].dtype, ImageDType): idx = to_image_idx(self.bufs[i].dtype.shape, idx, valid)
      self.uop(UOps.STORE, None, [var], MemOp(self.get_buffer_name(i), idx, self.bufs[i].__class__ is LocalBuffer, self.bufs[i].dtype, valid))

  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)
  def linearize(self):
    self.process()

    # limit dims if we need to
    if self.opts.global_max and self.opts.local_max: self.limit_global_dims(3, self.opts.global_max, self.opts.local_max)

    # uops
    self.uops: List[UOp] = []
    self.load_cache: Dict[str, Token] = {}
    self.saved_exprs: Dict[Tuple[Op, Tuple[Token, ...]], Token] = dict()

    # add global buffers
    for buf,name in self.arg_bufs.items():
      self.uop(UOps.DEFINE_GLOBAL, None, [], (name, buf.dtype))
    # add variables from symbolic shapes
    for var in sorted(set(v for buf in self.ast.buffers for v in buf.st.var_vals), key=lambda k: k.key):
      self.uop(UOps.DEFINE_GLOBAL, None, [], (var.expr, dtypes._arg_int32))

    # add a local buffer for multistage reduce
    if self.group_for_reduce:
      # TODO: the strides of this can be controlled
      self.sts.append(ShapeTracker(tuple([1] * self.first_reduce + self.group_for_reduce + [1] * (self.shape_len - self.upcasted - len(self.group_for_reduce) - self.first_reduce) + [x[0] for x in self.upcasted_axis(0)])))
      self.bufs.append(LocalBuffer("temp", self.sts[-1].size()))
      self.uop(UOps.DEFINE_LOCAL, None, [], ("temp", self.sts[-1].size()))

    # define local buffers
    for lb in self.local_alias.values():
      self.uop(UOps.DEFINE_LOCAL, None, [], (lb.name, self.sts[self.bufs.index(lb)].size()))

    # print
    if DEBUG >= 3: self.printbufs()

    # kernel name (before late upcast)
    self.function_name = ("r_" if self.reduceop else "E_") + '_'.join([str(x) if isinstance(x, int) else sym_rename(x) for x in self.full_shape])
    self.display_name = ("r_" if self.reduceop else "E_") + colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

    # parse AST
    loaded_buffers = {}
    acc = []

    # ssa
    _ssa:DefaultDict[str,int] = defaultdict(int)
    def ssa(name, ltype=dtypes.float) -> Token:
      _ssa[name] += 1
      return Token(f"{name}{_ssa[name]-1}", ltype)

    # global loop
    global_idxs = [Variable(f"gidx{i}", 0, self.full_shape[i]-1) for i in range(0, self.first_reduce-self.local_dims)]
    self.uop(UOps.LOOP, None, [], (global_idxs, "global"))

    # local loop
    local_idxs = [Variable(f"lidx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce-self.local_dims, self.first_reduce+len(self.group_for_reduce))]
    self.uop(UOps.LOOP, None, [], (local_idxs, "local"))

    # upcast indexes
    full_upcast_idxs = [Variable(None, 0, s-1) for s in self.full_shape[self.shape_len-self.upcasted:]]
    upcast_idxs = [Variable(None, 0, s-1) for s in self.output_shape[self.shape_len-self.upcasted:]]

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
      locals_to_store = []
      for i in self.local_alias:
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
        ll = self.global_load(i, idxs)
        locals_to_store.append((self.bufs.index(self.local_alias[i]), idxs, ll))

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
          for i, idxs, ll in locals_to_store: self.global_store(i, idxs, ll, ssa)
          self.uop(UOps.BARRIER, None, [], ())

        # load earlybufs
        loaded_buffers.update({b:self.global_load(self.bufs.index(self.local_alias[i]) if i in self.local_alias else i, global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs[1:], start=1) if b in self.earlybufs})

        # run early AST (with reduce)
        self.ast_parse(self.reduceop, [acc[off] for off in self.acc_offsets(self.full_buf_index)], loaded_buffers, ssa, do_reduce=True)

      # end the reduce loop
      self.uop(UOps.ENDLOOP, None, [], (reduce_idxs, "reduce"))
      self.load_cache.clear()

      # end the local loop, do the local reduce
      if self.group_for_reduce:
        fake_global_idxs = [x*0 for x in global_idxs]
        self.global_store(-1, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, acc, ssa)  # store accumulators
        self.uop(UOps.BARRIER, None, [], ())
        self.uop(UOps.ENDLOOP, None, [], (local_idxs, "local"))

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
        self.ast_parse(LazyOp(self.reduceop.op, ("LOCAL_BUFFER",)), [acc[off] for off in self.acc_offsets(-1)], loaded_buffers, ssa, do_reduce=True) # type: ignore

        # end the late reduce loop
        self.uop(UOps.ENDLOOP, None, [], (end_local_idxs, "late_reduce"))
        self.load_cache.clear()

    # load latebufs
    loaded_buffers.update({b:self.global_load(i, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs) for i,b in enumerate(self.bufs) if b not in self.earlybufs and i != 0 and b.__class__ is not LocalBuffer})

    # run late AST
    val = self.ast_parse(self.ast, acc, loaded_buffers, ssa)

    # store
    self.global_store(0, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, val, ssa)

    if not self.group_for_reduce:
      # end the global+local loop
      self.uop(UOps.ENDLOOP, None, [], (global_idxs+local_idxs, "global+local"))
    else:
      # end the global loop
      self.uop(UOps.ENDLOOP, None, [], (global_idxs, "global"))

    # name the function something unique
    Linearizer.kernel_cnt[self.function_name] += 1
    suffix = f"{'n'+str(Linearizer.kernel_cnt[self.function_name]-1)}" if Linearizer.kernel_cnt[self.function_name] > 1 else ""
    self.function_name, self.display_name = self.function_name+suffix, self.display_name+colored(suffix, 'BLACK')
    return self

  _OT = TypeVar("_OT")
  def uop(self, uop:UOps, out:_OT, vin:List[Token], arg:Any=None) -> _OT:
    self.uops.append(UOp(uop, cast(Optional[Token], out), vin, arg))
    if DEBUG >= 4: print(self.uops[-1])
    return out

  def uop_alu(self, out: Token, vin: List[Token], op: Op) -> Token:
    key = (op, tuple(vin))
    if key not in self.saved_exprs: self.saved_exprs[key] = self.uop(UOps.ALU, out, vin, op)
    return self.saved_exprs[key]

  def ast_parse(self, x, acc, loaded_buffers, ssa, do_reduce=False) -> List[Token]:
    if x.__class__ is not LazyOp: return loaded_buffers[x]
    if x.op in [UnaryOps.NOOP, UnaryOps.CAST]: return self.ast_parse(x.src[0], acc, loaded_buffers, ssa)  # cast isn't an ALU op
    if x.op in ReduceOps and not do_reduce: return acc
    # MULACC fusion. TODO: this is copied from Interpreted
    if x.op == ReduceOps.SUM and x.src[0].__class__ is LazyOp and x.src[0].op == BinaryOps.MUL:
      x = LazyOp(TernaryOps.MULACC, x.src[0].src, x.arg)
    if x.op == ReduceOps.SUM and x.src[0].__class__ is LazyOp and x.src[0].op == UnaryOps.CAST and x.src[0].src[0].__class__ is LazyOp and x.src[0].src[0].op == BinaryOps.MUL:
      x = LazyOp(TernaryOps.MULACC, x.src[0].src[0].src, x.arg)
    if x.op in {BinaryOps.ADD, BinaryOps.MUL}:
      # Reorder sources to put constants first so get_grouped_maybe_float4 can fold the op
      srcs = sorted(x.src, key=lambda x: (x.realized.__class__ != RawConst) if x.__class__ == LazyBuffer else 0)
      x.src = tuple(srcs)
    values = [self.ast_parse(v, acc, loaded_buffers, ssa) for v in x.src]
    ops = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX, TernaryOps.MULACC:TernaryOps.MULACC}
    if x.op in ops:
      ret = [(idx, self.uop(UOps.ALU, val[-1], list(val), ops[x.op])) for idx, val in get_grouped_maybe_float4(*values, acc, grouping_allowed=self.opts.supports_float4_alu)]
    else:
      ret = [(idx, self.uop_alu(ssa('alu', dtypes._float4) if any(x.dtype == dtypes._float4 and x.offset is None for x in val) else ssa('alu'), list(val), x.op)) for idx, val in get_grouped_maybe_float4(*values, grouping_allowed=self.opts.supports_float4_alu and x.op not in {BinaryOps.CMPLT, TernaryOps.WHERE})]
    ordered_ret: List[Optional[Token]] = [None]*len(values[0])
    # scatter
    for i,j in ret:
      for o,k in enumerate(i):
        ordered_ret[k] = Token(j.name, j.dtype, o) if j.dtype == dtypes._float4 else j
    assert all(isinstance(x, Token) for x in ordered_ret), "some tokens didn't get scattered?"
    return cast(List[Token], ordered_ret)
