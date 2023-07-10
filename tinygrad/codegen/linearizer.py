from typing import List, Tuple, Any, Optional, cast, DefaultDict, NamedTuple, TypeVar, Dict, Iterator, Union, Sequence
import itertools, math
from collections import defaultdict
from enum import Enum, auto

from tinygrad.helpers import dedup, colored, ImageDType, DEBUG, prod, dtypes, mnum, DType, all_same, partition, getenv
from tinygrad.ops import LazyOp, FlopCounter, get_lazyop_info, UnaryOps
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import MovementOps, ReduceOps, BinaryOps, TernaryOps
from tinygrad.runtime.lib import RawConst
from tinygrad.shape.shapetracker import ShapeTracker, strides_for_shape, View
from tinygrad.shape.symbolic import Variable, NumNode, Node, SumNode, MulNode
VariableOrNum = Union[Variable, NumNode, Node]

# bottom ones are asm only
class UOps(Enum): LOOP = auto(); DEFINE_LOCAL = auto(); LOAD = auto(); ALU = auto(); CONST = auto(); ENDLOOP = auto(); STORE = auto(); CAST = auto(); BARRIER = auto(); WMMA = auto(); \
                  SPECIAL = auto(); DEFINE_REGISTER = auto(); LABEL = auto(); COND_BRANCH = auto() # noqa: E702

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

class LocalBuffer(NamedTuple):
  name: str
  size: int
  dtype: DType = dtypes.float32
  realized: None = None
  def __str__(self): return f"localbuffer<{self.name}[{self.size}]>"

class Token(NamedTuple):
  name: str
  dtype: DType
  offset: Optional[int] = None
  def render(self, with_type=False):
    if with_type:
      assert self.offset is None
      return f"{self.dtype.name} {self.name}"
    if self.offset is None: return self.name
    assert self.dtype in [dtypes._float4, dtypes._float2]
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
  if isinstance(idx, Variable):  return [idx] if idx.expr is not None else [Variable.num(j) for j in range(idx.min, idx.max+1)]
  if isinstance(idx, NumNode): return [idx]
  if isinstance(idx, MulNode): return [x*idx.b for x in expand_node(idx.a)]
  if isinstance(idx, SumNode): return [Variable.sum(list(it)) for it in itertools.product(*[expand_node(x) for x in idx.nodes])]
  raise NotImplementedError(idx)

def expand_idxs(idxs:Sequence[Node]) -> Iterator[Tuple[Node, ...]]:
  for x in itertools.product(*[expand_node(idx) for idx in idxs[::-1]]):
    yield x[::-1]

class MemOp(NamedTuple):
  i: int
  idx: Variable
  valid: Variable

class UOp(NamedTuple):
  uop: UOps
  out: Optional[Token]
  vin: List[Token]
  arg: Any
  def __repr__(self): return f"{str(self.uop):20s}: {str(self.out) if self.out is not None else '':25s} {str(self.vin):32s} {self.arg}"

class Linearizer:
  supports_float4: bool = False
  supports_float4_alu: bool = False

  def __init__(self, ast:LazyOp, output_buffer:LazyBuffer):
    # NOTE: if there's a RESHAPE, we skip it. the output shape is set from the reduce op or a latebuf
    self.ast = ast.src[0] if ast.op == MovementOps.RESHAPE else ast

    # get the output buffers
    self.bufs = [output_buffer] + dedup(ast.buffers)

    # key for lookup in cache (can change, str might not be right)
    # bufs are needed because kernels like f(x) = x + x and f(x, y) = x + y have the same str(ast), but are different kernels.
    # mapping the buffers to integers is required because a-b != b-a (and how would you tell a and b apart?)
    self.key = (ast.map_buffers({x:i for i,x in enumerate(self.bufs)}).key, tuple([x.key for x in self.bufs]))

  def process(self) -> None:
    if hasattr(self, "sts"): return   # already processed

    # fetch lazyop info
    self.info: FlopCounter = get_lazyop_info(cast(LazyOp, self.ast))
    self.mem_estimate: int = sum(x.dtype.itemsize*(x.realized.size if x.realized is not None else prod(x.shape)) for x in self.bufs if x is not None)

    # there's only allowed to be one reduceop
    reduceops = [x for x in self.ast.get_lazyops() if x.op in ReduceOps]
    assert len(dedup(reduceops)) <= 1, "max one reduce op in an ast"
    self.reduceop = reduceops[0] if reduceops else None

    # get earlybufs, before the one reduce op
    self.earlybufs = dedup(self.reduceop.buffers) if self.reduceop else []

    # create new shapetrackers inside this kernel, we will permute them
    self.sts: List[ShapeTracker] = [x.st.copy() for x in self.bufs]
    for st in self.sts: st.simplify()

    # make the output buffer shape correct in here
    self.sts[0].reshape(self.info.shape)
    self.full_buf_index: int = self.bufs.index(self.earlybufs[0]) if len(self.earlybufs) > 0 else 0

    # move all reduce axes to the end
    reduce = list(enumerate(zip(self.full_shape, self.sts[0].shape)))
    permute = tuple([i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])
    self.reshape_and_permute(None, permute)

    # parameters
    self.group_for_reduce: List[int] = []
    self.upcasted: int = 0
    self.local_dims: int = 0
    self.local_alias: Dict[int, LocalBuffer] = {}
    self.use_tensor_cores: bool = False
    self.exclude_local_upcast: int = 0

    # group simplifies
    self.simplify_ones()
    self.simplify_merge_adjacent()

    # print early
    if DEBUG >= 5: self.printbufs("early")

  def shape_offsets(self, i): return itertools.product(*[list(range(s)) for s in self.sts[i].shape[self.shape_len-self.upcasted:][::-1]]) if self.upcasted > 0 else [tuple()]
  def float4_axis(self, i): return [x-(self.shape_len-self.upcasted) for x in self.sts[i].unit_stride_axes() if x >= self.shape_len-self.upcasted and self.sts[i].shape[x]%4 == 0]

  def upcasted_axis(self, i):
    return list(zip(self.sts[i].shape[self.shape_len-self.upcasted:],
                    self.sts[i].real_strides()[self.shape_len-self.upcasted:],
                    [x!=y for x,y in zip(self.sts[0].shape[self.shape_len-self.upcasted:], self.full_shape[self.shape_len-self.upcasted:])]))

  # TODO: is there a better way to write this?
  def acc_offsets(self, i):
    if self.upcasted == 0: return [0]
    upcasted_i = self.upcasted_axis(i)
    acc_strides = [x*(1-upcasted_i[::-1][i][2]) for i,x in enumerate(strides_for_shape(tuple(1 if r else s for s,_,r in upcasted_i[::-1])))]
    return [sum(t) for t in itertools.product(*[[y*acc_strides[i] for y in range(x[0])] for i,x in enumerate(upcasted_i[::-1])])]

  def get_upcast_dim(self, i) -> List[int]:
    should_upcast = self.supports_float4 and (self.bufs[i].dtype in [dtypes.float32, dtypes.float16] or isinstance(self.bufs[i].dtype, ImageDType))
    return [x for x in self.sts[i].unit_stride_axes() if should_upcast and x >= self.shape_len-self.upcasted and self.sts[i].shape[x] > 1]

  def global_load(self, i:int, idxs:Sequence[VariableOrNum], const=None) -> List[Token]:
    expanded_nodes = [expand_node(idx) for idx in idxs]
    _idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    upcast_dim = self.get_upcast_dim(i)

    amt = 1
    if len(upcast_dim) == 1 and len(expanded_nodes[upcast_dim[0]]) in [4,2]:
      dim, amt = upcast_dim[0], len(expanded_nodes[upcast_dim[0]])

    cache: Dict[str, Token] = {}
    ret = []
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
      key = f"{localtype}{idx.render()}{valid.render()}"
      if key not in cache:
        if isinstance(self.bufs[i].dtype, ImageDType): idx = to_image_idx(self.bufs[i].dtype.shape, idx, valid)
        cache[key] = self.uop(UOps.LOAD, Token(f"val{mnum(i)}_{len(cache)}", localtype), [], MemOp(i, idx, valid)) if const is None else \
                     self.uop(UOps.CONST, Token(f"acc{mnum(i)}_{len(cache)}", localtype), [], const)
      ret.append(Token(cache[key].name, cache[key].dtype, expanded_nodes[dim].index(_idx[dim])) if localtype != dtypes.float else cache[key])
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
      self.uop(UOps.STORE, None, [var], MemOp(i, idx, valid))

  def linearize(self):
    # uops
    self.uops: List[UOp] = []
    self.saved_exprs: Dict[LazyOp, List[Token]] = dict()

    # add a local buffer for multistage reduce
    if len(self.group_for_reduce):
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
    self.function_name = ("r_" if self.reduceop else "E_") + '_'.join([str(x) for x in self.full_shape])
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
        for j,v in enumerate(full_upcast_idxs):
          if strides[len(global_idxs)+len(local_idxs)+len(reduce_idxs)+j] == 0:
            if DEBUG >= 4: print("upcasting stride 0")
            this_upcast_idxs.append(Variable.num(0))
          elif (elc:=[el for el in extra_locals if v.min == el.min and v.max == el.max]):
            if DEBUG >= 4: print(f"upcasting matched stride {elc[0]}")
            this_upcast_idxs.append(elc[0])
            extra_locals.remove(elc[0])
          elif (elc:=[el for el in extra_locals if v.min == el.min and (v.max+1)%(el.max+1) == 0]):
            tacc = Variable.num(0)
            rem = v.max+1
            while len(elc) and rem%(elc[0].max+1) == 0:
              if DEBUG >= 4: print(f"upcasting partial stride {rem} {elc[0]} left: {elc[1:]}")
              rem = rem//(elc[0].max+1)
              tacc += (elc[0] * rem)
              extra_locals.remove(elc[0])
              elc = [el for el in extra_locals if v.min == el.min and rem%(el.max+1) == 0]
            if DEBUG >= 4 and rem > 1: print(f"failed upcasting partial stride {rem} extra locals {extra_locals}")
            this_upcast_idxs.append(tacc + Variable(None, 0, rem-1))
          else:
            if DEBUG >= 4: print(f"failed upcasting stride {v} extra locals {extra_locals}")
            this_upcast_idxs.append(v)
        idxs = global_idxs+local_idxs+reduce_idxs+this_upcast_idxs
        ll = self.global_load(i, idxs)
        locals_to_store.append((self.bufs.index(self.local_alias[i]), idxs, ll))

      # copy in any global buffers
      if self.use_tensor_cores:
        i = 0
        for y0,y1 in zip(locals_to_store[1][2][::2], locals_to_store[1][2][1::2]):
          for x0,x1 in zip(locals_to_store[0][2][::2], locals_to_store[0][2][1::2]):
            self.uop(UOps.WMMA, None, [x0, x1, y0, y1, acc[i], acc[i+1]], ())
            i += 2
      else:
        if locals_to_store:
          self.uop(UOps.BARRIER, None, [], ())
          for i, idxs, ll in locals_to_store: self.global_store(i, idxs, ll, ssa)
          self.uop(UOps.BARRIER, None, [], ())

        # load earlybufs
        loaded_buffers.update({b:self.global_load(self.bufs.index(self.local_alias[i]) if i in self.local_alias else i, global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs) if b in self.earlybufs and i != 0})

        # run early AST (with reduce)
        self.ast_parse(self.reduceop, [acc[off] for off in self.acc_offsets(self.full_buf_index)], loaded_buffers, ssa, do_reduce=True)

      # end the reduce loop
      self.uop(UOps.ENDLOOP, None, [], (reduce_idxs, "reduce"))

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

  _OT = TypeVar("_OT")
  def uop(self, uop:UOps, out:_OT, vin:List[Token], arg:Any=None) -> _OT:
    self.uops.append(UOp(uop, cast(Optional[Token], out), vin, arg))
    if DEBUG >= 4: print(self.uops[-1])
    return out

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
    if x not in self.saved_exprs:
      values = [self.ast_parse(v, acc, loaded_buffers, ssa) for v in x.src]
      ops = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX, TernaryOps.MULACC:TernaryOps.MULACC}
      if x.op in ops:
        ret = [(idx, self.uop(UOps.ALU, val[-1], list(val), ops[x.op])) for idx, val in get_grouped_maybe_float4(*values, acc, grouping_allowed=self.supports_float4_alu)]
      else:
        ret = [(idx, self.uop(UOps.ALU, ssa('alu', dtypes._float4) if any(x.dtype == dtypes._float4 and x.offset is None for x in val) else ssa('alu'), list(val), x.op)) for idx, val in get_grouped_maybe_float4(*values, grouping_allowed=self.supports_float4_alu and x.op not in {BinaryOps.CMPEQ, TernaryOps.WHERE})]
      ordered_ret: List[Optional[Token]] = [None]*len(values[0])
      # scatter
      for i,j in ret:
        for o,k in enumerate(i):
          ordered_ret[k] = Token(j.name, j.dtype, o) if j.dtype == dtypes._float4 else j
      assert all(isinstance(x, Token) for x in ordered_ret), "some tokens didn't get scattered?"
      self.saved_exprs[x] = cast(List[Token], ordered_ret)
    return self.saved_exprs[x]

  @property
  def first_reduce(self) -> int: return [x!=y for x,y in zip(self.sts[0].shape[:self.shape_len-self.upcasted]+(0,), self.full_shape[:self.shape_len-self.upcasted]+(1,))].index(True)

  @property
  def output_shape(self) -> Tuple[int, ...]: return self.sts[0].shape

  @property
  def full_shape(self) -> Tuple[int, ...]: return self.sts[self.full_buf_index].shape

  @property
  def full_unupcasted_shape(self) -> Tuple[int, ...]: return self.full_shape[:self.shape_len-self.upcasted]

  @property
  def shape_len(self) -> int: return len(self.sts[0].shape)

  @property
  def upcast_in_mid_reduce_axes(self) -> List[int]: return [j for j in range(self.first_reduce, self.first_reduce+len(self.group_for_reduce)) if self.full_shape[j] == self.sts[0].shape[j]]

  # there's seven chunks of the shape
  # blue   -- global dims
  # cyan   -- local dims
  #  *** self.first_reduce
  # green  -- reduce-local dims
  # white  -- reduce-late upcasted dim (self.upcast_in_mid_reduce_axes)
  # red    -- reduce loops
  #  *** self.upcasted
  # purple -- reduce upcasted
  # yellow -- normal upcasted dimensions
  def colors(self) -> List[str]:
    # up to first_reduce, they are all global (blue)
    colors = ["blue"] * (self.first_reduce-self.local_dims)
    # except the local_dims, these are non-reduce locals (cyan)
    colors += ["cyan"] * (self.local_dims)
    # between first_reduce and first_reduce + group_for_reduce, they are either local (cyan), or late upcasted (green)
    colors += ["white" if i in self.upcast_in_mid_reduce_axes else "green" for i in range(self.first_reduce, self.first_reduce + len(self.group_for_reduce))]
    # between first_reduce + group_for_reduce and upcasted, they are reduce (red)
    colors += ["red"] * ((self.shape_len-self.upcasted) - (self.first_reduce + len(self.group_for_reduce)))
    # upcasted dimensions are reduce (magenta) or normal (yellow)
    colors += ["magenta" if self.full_shape[i] != self.sts[0].shape[i] else "yellow" for i in range(self.shape_len-self.upcasted, self.shape_len)]
    assert len(colors) == self.shape_len, "colors size mismatch"
    return colors

  def colored_shape(self) -> str: return ' '.join(colored(f"{s:4d}", color) for s,color in zip(self.full_shape, self.colors()))
  def printbufs(self, prefix=""):
    for i in range(len(self.sts)):
      print(prefix, f"{i:3d} {str(self.bufs[i].realized) if self.bufs[i].realized is not None else str(self.bufs[i]):47s}", self.sts[i].views)
    print(self.colored_shape())

  # ******************** base simplifiers ********************

  # apply reshape and permute to all shapetrackers
  def reshape_and_permute(self, new_shape_fxn, axis):
    for st in self.sts:
      if new_shape_fxn is not None: st.reshape(tuple(new_shape_fxn(st.shape)))
      if axis is not None: st.permute(tuple(axis))

  # drops the final dimension
  def upcast(self):
    assert self.full_shape[-1] != 1, "can't upcast a dimension with size 1"
    self.upcasted += 1

  # axis : the axis to pull from
  # amount : the amount to take
  # top : if you want to pull that amount from the top
  # insert_before : place to insert the new stuff
  def shift_to(self, axis, amount, top=False, insert_before=None):
    if insert_before is None: insert_before = self.shape_len
    move_axis = axis if top else axis+1
    if move_axis < insert_before: insert_before += 1
    self.reshape_and_permute(
      lambda x: list(x[0:axis]) + (([amount, x[axis]//amount] if top else [x[axis]//amount, amount]) if x[axis] > 1 else [1,1]) + list(x[axis+1:]),
      [i for i in range(insert_before) if i != move_axis] + [move_axis] + [i for i in range(insert_before, self.shape_len+1) if i != move_axis])

  # ******************** complex simplifiers ********************

  def simplify_ones(self):
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    if self.shape_len == 0: return
    all_ones = [all(st.shape[i]==1 for st in self.sts) for i in range(self.shape_len)]
    # keep at least 1 one
    if all(all_ones): all_ones[-1] = False
    self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)

  def simplify_merge_adjacent(self):
    if self.shape_len == 0: return
    shapes, strides = [x.shape for x in self.sts], [x.real_strides() for x in self.sts]

    # merge dimensions if we can, multi get_shape_strides
    # TODO: does this always preserve the reduce dimension, NO
    # TODO: move this into shapetracker, with tests!
    rets = [[(shapes[j][0], strides[j][0])] for j in range(len(shapes))]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for j in range(len(shapes)):
        # TODO: added the always mergeability of 1s, is this right? if so, add to shapetracker in the 1 case
        can_merge.append(strides[j][i] is not None and ((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*cast(int, strides[j][i])) or (strides[j][i] == 0 and rets[j][-1][1] == 0)))
      # more can merge than this
      mergeable = all(can_merge) and i != self.first_reduce
      for j in range(len(shapes)):
        if mergeable: rets[j][-1] = (rets[j][-1][0] * shapes[j][i], strides[j][i])
        else: rets[j].append((shapes[j][i], strides[j][i]))

    # do the reshapes
    for i,x in enumerate(rets): self.sts[i].reshape(tuple([y[0] for y in x]))

  # ******************** GPU simplifiers ********************

  def required_optimizations(self, early_only=False):
    for buf_index,buf in enumerate(self.bufs):
      unit_stride_axes_mul_4 = [i for i in self.sts[buf_index].unit_stride_axes(ignore_valid=True) if self.sts[buf_index].shape[i]%4 == 0]
      if (not early_only or buf in self.earlybufs) and self.bufs[buf_index].dtype.__class__ is ImageDType:
        assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {self.bufs[buf_index]}"
        if all(x < (self.shape_len-self.upcasted) for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in self.upcast_in_mid_reduce_axes:
          self.shift_to(unit_stride_axes_mul_4[0], 4)
          self.upcast()

  def limit_global_dims(self, limit):
    # sometimes, there's more dimensions than len(self.lang.gid).
    # compact all the dimensions into the first
    # NOTE: this might make multiview shapetrackers
    if limit and (self.first_reduce-self.local_dims) > limit:
      num_to_merge = ((self.first_reduce-self.local_dims) - limit)+1
      self.reshape_and_permute(lambda x: (prod(x[0:num_to_merge]),)+x[num_to_merge:], None)
      if DEBUG >= 3: print("reshaped to", self.full_shape, "due to too many global dimensions")

  def alias_buffer(self, i, pattern):
    assert len(pattern) == len(self.sts[i].shape), f"must include a pattern for each shape {pattern} {self.sts[i].shape}"

    bst = 1
    real_strides = self.sts[i].real_strides()
    shp, stride = [(s if p != 0 else 1) for s,p in zip(self.sts[i].shape, pattern)], [0]*len(pattern)
    for priority in range(1, max(pattern)+1):  # priority. 0 is non local and ignored
      for j,p in enumerate(pattern):
        if priority == p and real_strides[j] != 0:
          stride[j] = bst
          bst *= shp[j]

    self.sts.append(ShapeTracker(tuple(shp), [View(tuple(shp), tuple(stride))]))
    self.bufs.append(LocalBuffer(name=f"ldata{i}", size=self.sts[-1].size()))
    if DEBUG >= 4: print("aliasing buffer", self.sts[i])
    self.local_alias[i] = self.bufs[-1]

  def hand_coded_optimizations(self):
    if getenv("NOOPT"): return

    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    self.required_optimizations(early_only=True)

    # simplify
    self.simplify_ones()

    # should use tensor cores?
    # first, confirm it's a straightforward mulacc on a device with real locals
    tensor_cores_allowed = getenv("TC", 1) != 0 and (getenv("TC", 1) == 2 or (self.bufs[0].device == "METAL" and getenv("CI", "") != "true"))
    if tensor_cores_allowed and self.reduceop and self.reduceop.op == ReduceOps.SUM and \
       isinstance(self.reduceop.src[0], LazyOp) and self.reduceop.src[0].op == BinaryOps.MUL and \
       isinstance(self.reduceop.src[0].src[0], LazyBuffer) and isinstance(self.reduceop.src[0].src[1], LazyBuffer) and hasattr(self, 'lang') and len(self.lang.lid):
      buf0 = self.bufs.index(self.reduceop.src[0].src[0])
      buf1 = self.bufs.index(self.reduceop.src[0].src[1])
      buf0_strides = self.sts[buf0].real_strides()
      buf1_strides = self.sts[buf1].real_strides()
      axis_buf0 = [(i,self.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides) if s == 0 and self.full_shape[i]%8 == 0]
      axis_buf1 = [(i,self.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides) if s == 0 and self.full_shape[i]%8 == 0]
      if len(axis_buf0) and len(axis_buf1) and self.full_shape[self.first_reduce]%8 == 0 and (self.shape_len-self.first_reduce) == 1:
        if DEBUG >= 3: print("TENSOR CORES", axis_buf0, axis_buf1)
        self.use_tensor_cores = getenv("TC", 1) == 1  # TC=2 will do the shape ops without the WMMA

        # TODO: select axis in smart way
        s0, s1 = axis_buf0[-1][0], axis_buf1[-1][0]
        global_count = self.first_reduce

        # upcast first
        if self.full_shape[self.first_reduce] > 8: self.shift_to(self.first_reduce, 8)
        self.upcast()

        # 2 locals
        self.shift_to(s1, 8, insert_before=self.first_reduce)  # axis 2
        self.shift_to(s0, 8, insert_before=self.first_reduce)  # axis 3

        # permuted+upcast for tensor cores
        self.shift_to(global_count, 4, insert_before=self.first_reduce)
        self.shift_to(global_count+1, 4, insert_before=self.first_reduce)
        self.shift_to(self.first_reduce-1, 2)
        self.upcast()

        # final global upcast
        for ax in [s1, s0]:
          for upc in [4,3,2]:
            if self.full_shape[ax]%upc == 0:
              self.shift_to(ax, upc)
              self.upcast()
              break

        # alias buffer
        self.local_dims = self.first_reduce - global_count
        alias_pattern = [0]*global_count + [2] * self.local_dims + [0] * (self.shape_len-self.upcasted-self.first_reduce) + [1,1] + [3] * (self.upcasted-2)
        self.alias_buffer(buf0, alias_pattern)
        self.alias_buffer(buf1, alias_pattern)

        # very late upcast to run group at the same time. only if actually using real tensor cores, otherwise local isn't a simdgroup
        if self.use_tensor_cores and self.full_shape[s0] % 2 == 0:
          self.shift_to(s0, 2, insert_before=self.first_reduce-self.local_dims)
          self.local_dims += 1
          self.exclude_local_upcast += 1

        # early exit
        return

    # are we grouping? (requires local shape support)
    if not self.float4_axis(0) and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
      # TODO: use 1024 if it's allowed in a smarter way
      for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
        if all(st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts):
          self.shift_to(self.first_reduce, sz, top=True, insert_before=self.first_reduce + len(self.group_for_reduce))
          self.group_for_reduce.append(sz)
          break

    # are we upcasting in mid reduce? (only for images)
    if self.bufs[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduce and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1:
      axes = self.sts[0].unit_stride_axes()
      assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
      if self.sts[0].shape[axes[0]]%4 == 0:
        self.shift_to(axes[0], 4, insert_before=self.first_reduce + len(self.group_for_reduce))   # insert at the end of the grouped axis
        self.group_for_reduce.append(4)

    # now do everything required
    self.required_optimizations()

    # simplify (sets first_reduce)
    self.simplify_ones()

    # use more opencl indexing if the output buffer is an image and we have room
    if self.bufs[0].dtype.name.startswith('image') and self.first_reduce+len(self.group_for_reduce) < 3:
      base_shape = self.bufs[0].dtype.shape
      if (base_shape[0]*base_shape[1]) % self.sts[0].shape[0] == 0 and self.sts[0].shape[0]//base_shape[0] != 0:
        if DEBUG >= 4: print("split opencl", base_shape, self.sts[0].shape)
        self.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
        self.simplify_ones()

    # no more opt if we are grouping
    if self.group_for_reduce: return

    # **** below this line need to be optional and benchmarked ****

    # potentially do more upcasts of non reduce axes based on a heuristic
    upcasted_axis = set()
    while prod(self.sts[0].shape[:self.first_reduce]) >= 1024:
      xb_choices = []
      for axis, upcast_amount in itertools.product(range(self.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
        # if we haven't upcasted it, it mods, and some buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
        if axis not in upcasted_axis and self.full_shape[axis]%upcast_amount == 0 and any(self.sts[buf_index].views[-1].strides[axis] == 0 and not any(x[1] == 0 for x in self.upcasted_axis(buf_index)) for buf_index in range(len(self.sts))):
          xb_choices.append((sum(st.views[-1].strides[axis]>0 for st in self.sts), sum(st.views[-1].strides[axis] for st in self.sts), axis, upcast_amount))
      if len(xb_choices):
        xb_choices = sorted(xb_choices)
        if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
        self.shift_to(xb_choices[0][2], amount=xb_choices[0][3])
        self.upcast()
        self.simplify_ones()
        upcasted_axis.add(xb_choices[0][2])
      else:
        break

    # if last dim is small(ish) and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast. NOTE: careful, this has broken VALIDHACKS
    if self.first_reduce < (self.shape_len-self.upcasted) and (len(list(self.shape_offsets(self.full_buf_index))) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))):
      if (s:=self.full_unupcasted_shape[-1]) <= 32:
        self.upcast()
        # if it's small, upcast a second reduce dimension too
        if self.first_reduce < (self.shape_len-self.upcasted) and s <= 3 and self.full_unupcasted_shape[-1] <= 3: self.upcast()
      else:
        for splits in [4]:
          if self.full_unupcasted_shape[-1]%splits == 0:
            self.shift_to(len(self.full_unupcasted_shape)-1, splits, insert_before=len(self.full_unupcasted_shape))
            self.upcast()
            break

    # if nothing at all is upcasted and it's easy to, do an upcast
    # TODO: this is breaking the tests
    for splits in [4]:
      if self.upcasted == 0 and len(self.full_unupcasted_shape) > 0 and self.full_unupcasted_shape[-1] % splits == 0:
        self.shift_to(len(self.full_unupcasted_shape)-1, splits, insert_before=len(self.full_unupcasted_shape))
        self.upcast()

    # **** local groups ****

    for axis in range(self.first_reduce - self.local_dims - 1, -1, -1):
      local_size = prod(self.full_shape[self.first_reduce-self.local_dims:self.first_reduce])
      if self.full_shape[axis] == 1: continue
      last_try = self.local_dims == 0 and axis == 0
      if any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))) or last_try:
        for sz in [x for x in (([32] if last_try else []) + [16,8,4,3]) if self.full_shape[axis] % x == 0 and local_size*x <= 128]:
          self.shift_to(axis, sz, insert_before=self.first_reduce-self.local_dims)
          self.local_dims += 1
          break
      if self.local_dims >= 3: break
    self.simplify_ones()

