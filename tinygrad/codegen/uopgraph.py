from __future__ import annotations
from typing import Iterator, Optional, Tuple, Any, Dict, List, DefaultDict, Set, Callable, Union, cast, TypeVar, TYPE_CHECKING
import functools, itertools, heapq, math
from collections import defaultdict
from dataclasses import dataclass, field
from tinygrad.dtype import dtypes, DType, PtrDType, ImageDType
from tinygrad.shape.symbolic import Variable
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, exec_alu
from tinygrad.helpers import DEBUG, getenv, flatten, all_same, dedup, TRANSCENDENTAL
from tinygrad.codegen.uops import UOp, UOps, END_FOR_UOP, type_verify
from tinygrad.codegen.transcendental import xexp2, xlog2, xsin, TRANSCENDENTAL_SUPPORTED_DTYPES

if TYPE_CHECKING:
  from tinygrad.renderer import Renderer

# *** simplification logic ***

@dataclass(frozen=True)
class UPat:
  op: Optional[Union[UOps, Set[UOps]]] = None
  arg: Any = None
  src: Optional[Union[Tuple[UPat, ...], List[UPat], UPat]] = None
  name: Optional[str] = None
  dtype: Optional[Union[DType, Set[DType]]] = None
  allow_len: Set[int] = field(default_factory=set)
  allow_any_len: bool = False

  @staticmethod
  def compile(u: UOp, name:Optional[str]=None) -> UPat:
    if u.op is UOps.VAR: return UPat(name=name or u.arg, dtype=u.dtype) if len(u.src) == 0 else UPat.compile(u.src[0], name or u.arg)
    return UPat(u.op, u.arg, (list if u.commutative() else tuple)([UPat.compile(src) for src in u.src]) if u.src != () else None,
                name, u.dtype, allow_any_len=(isinstance(name, str) and 'allow_any_len' in name))

T = TypeVar("T")
def __unmatch(m1:Union[T, Set[T]], m2:T) -> bool: return m2 not in m1 if isinstance(m1, set) else m2 != m1

def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:
  if pat.name is not None and store.setdefault(pat.name, uop) is not uop: return []
  if pat.arg is not None and __unmatch(pat.arg, uop.arg): return []
  if pat.dtype is not None and uop.dtype is not None and __unmatch(pat.dtype, uop.dtype): return []
  if pat.op is not None and __unmatch(pat.op, uop.op): return []
  if pat.src is None: return [store]
  # only one if it's a tuple
  # try all permutations if it's a list
  # repeat if it's a UPat
  res: List[Dict[str, UOp]] = []
  for vp in itertools.permutations(pat.src) if isinstance(pat.src,list) else ([pat.src] if isinstance(pat.src,tuple) else [(pat.src,)*len(uop.src)]):
    if len(uop.src) != len(vp) and (len(uop.src) not in pat.allow_len) and not pat.allow_any_len: return []
    new_stores = [store.copy()]
    for uu, vv in zip(uop.src, vp): new_stores = [rstore for nstore in new_stores for rstore in _match(uu, vv, nstore)]
    res.extend(new_stores)
  return res

class PatternMatcher:
  def __init__(self, patterns:List[Tuple[Union[UPat, UOp], Callable]]):
    self.patterns = patterns
    self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable]]] = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      if isinstance(p, UOp): p = UPat.compile(p)
      assert p.op is not None
      if isinstance(p.op, set):
        for uop in p.op: self.pdict[(uop, p.arg)].append((p, fxn))
      else:
        self.pdict[(p.op, p.arg)].append((p, fxn))

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    for p,fxn in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None: return ret # NOTE: if it returns None, we keep trying to match
    return None

def expand_nodes(parents:Set[UOp], expands:List[UOp], base:UOp) -> List[UOp]:
  # just in case, dedup expands
  expands = dedup(expands)

  # get children and define_accs
  children = defaultdict(list)
  define_accs = []
  for p in parents:
    if p.op is UOps.PHI:
      wmma_reduce_axes = flatten([x.arg[7] for x in p.parents if x.op is UOps.WMMA])
      parent_expands_for_acc = [x.arg for x in p.parents if x in expands and x.arg not in wmma_reduce_axes]
      define_accs.append((p.src[0], parent_expands_for_acc))
    for x in p.src:
      children[x].append(p)

  # get nodes on the path from root to the expand node
  on_path: Dict[UOp, None] = {}
  search = expands[:]
  while len(search):
    t = search.pop(0)
    for cc in children[t]:
      if cc in on_path: continue
      on_path[cc] = None
      search.append(cc)

  # toposort the nodes on the path
  # TODO: library!
  in_degree: DefaultDict[UOp, int] = defaultdict(int)
  for n in on_path:
    for x in children[n]:
      in_degree[x] += 1
  toposort: List[UOp] = []
  search2 = [p for p in on_path if in_degree[p] == 0]
  seen: Set[UOp] = set()
  while len(search2):
    n = search2.pop(0)
    if n in seen: continue
    toposort.append(n)
    for x in children[n]:
      in_degree[x] -= 1
      if in_degree[x] == 0:
        search2.append(x)

  # get replacements by index
  replacements: Dict[int, List[int]] = {}
  for r in expands:
    if r.arg in replacements: assert len(replacements[r.arg]) == len(r.src)
    else: replacements[r.arg] = list(range(0, len(r.src)))

  # get nodes on the path from root to the expand node
  rps = list(itertools.product(*replacements.values()))

  acc_number = 0
  replaces: List[Dict[UOp, UOp]] = []
  acc_cache: Dict[Tuple[Tuple[UOp, int, int], ...], UOp] = {}
  for rp in rps:
    rpk = dict(zip(replacements.keys(), rp))
    replace = {r:r.src[rpk[r.arg]] for r in expands}
    for d, acc_parents in define_accs:
      acc_index = tuple((d,x,rpk[x]) for x in acc_parents)
      if acc_index in acc_cache:
        replace[d] = acc_cache[acc_index]
      else:
        replace[d] = acc_cache[acc_index] = UOp(d.op, d.dtype, d.src, d.arg + (acc_number,))
        acc_number += 1
    replaces.append(replace)

  for cc in toposort:
    if cc.op is UOps.BARRIER:
      super_replace = UOp(cc.op, cc.dtype, sum([tuple(replace.get(x, x) for x in cc.src) for replace in replaces], ()), cc.arg)
      for replace in replaces:
        replace[cc] = super_replace
    else:
      for replace in replaces:
        tcc = replace.get(cc, cc)  # NOTE: handle expands that are already replaced
        replace[cc] = UOp(tcc.op, tcc.dtype, tuple(replace.get(x, x) for x in tcc.src), tcc.arg)
  return [x.get(base, base) for x in replaces]

# ***** reduce+image+contract handling *****

def expand_wmma(wmma):
  expands = [x for x in wmma.parents if x.op is UOps.EXPAND and (x.arg in wmma.arg[-1] or x.arg in wmma.arg[-2])]
  if len(expands) == 0: return None
  new_uops = expand_nodes(wmma.sparents, expands, wmma)
  # TODO: assert that these are all the same. they have to be
  return new_uops[0]

acc_number = 0
def replace_reduce(root):
  global acc_number
  expands = [x for x in root.src[1:] if x.op is UOps.EXPAND]

  # add other expands for float4. TODO: should be a faster way
  expand_args = [x.arg for x in expands]
  new_expands = [x for x in root.parents if x.op is UOps.EXPAND and x.arg in expand_args]
  expands = dedup(expands + new_expands)

  if len(expands):
    new_uops = expand_nodes(root.parents, expands, root.src[0])
  else:
    new_uops = [root.src[0]]

  const = UOp.const(root.dtype.scalar(), dtypes.as_const(0, root.dtype.scalar()) if root.arg is ReduceOps.SUM else dtypes.min(root.dtype.scalar()))
  acc = UOp(UOps.DEFINE_ACC, root.dtype, (const,) + tuple(x for x in root.src[1:] if x not in expands), (acc_number,))
  acc_number += 1
  ret = acc
  for xx in new_uops: ret = UOp.alu({ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[cast(ReduceOps, root.arg)], ret, xx)
  return UOp(UOps.PHI, ret.dtype, (acc, ret))

def replace_contract(root:UOp):
  parents, dtype = root.parents, cast(DType, root.dtype)
  expands: List[UOp] = [x for x in parents if x.op is UOps.EXPAND and x.arg in root.arg]
  assert all_same(expand_lens := [dtype.count] + [len(x.src) for x in expands]), expand_lens
  ret = expand_nodes(parents, expands, root.src[0])
  if len(ret) == 1: ret = ret*dtype.count   # TODO: why is this needed?
  return UOp(UOps.VECTORIZE, dtype, tuple(ret))

def fix_image_idx(ls:UOp):
  if ls.src[1].dtype is None or ls.src[1].dtype.count != 1: return None
  if not isinstance(ls.src[0].dtype, ImageDType): return None
  assert ls.op is not UOps.STORE or cast(DType, ls.src[2].dtype).count == 4, "image store must be float4"
  idxy = ls.src[1]
  #if not idxy.divides(4): raise RuntimeError("image index must divide 4")
  base_shape = ls.src[0].dtype.shape
  idx, idy = (idxy // 4) % base_shape[1], (idxy // (4 * base_shape[1]))
  image_idx = UOp(UOps.VECTORIZE, cast(DType, idxy.dtype).vec(2), (idx, idy))
  if ls.op is UOps.LOAD and cast(DType, ls.dtype).count == 1:
    cconst = (UOp(UOps.VECTORIZE, cast(DType, ls.dtype).vec(4), src=(ls.src[3], ls.src[3], ls.src[3], ls.src[3])),) if len(ls.src) >= 3 else ()
    loaded = UOp(ls.op, cast(DType, ls.dtype).vec(4), (ls.src[0], image_idx) + ls.src[2:3] + cconst, ls.arg)
    subidx = idxy%4
    ret = UOp.const(ls.dtype, 0)
    for i in range(4): ret = UOp.alu(TernaryOps.WHERE, subidx.ne(i), ret, UOp(UOps.GEP, ls.dtype, (loaded,), i))
    return ret
  return UOp(ls.op, ls.dtype, (ls.src[0], image_idx) + ls.src[2:], ls.arg)

def cast_reduce(cst):
  if cst.dtype.scalar() == cst.dtype: return None  # not for normal CAST. TODO: the merging one shouldn't be CAST
  if not all_same([(x.arg, x.src[1:]) for x in cst.src]): return None
  fst_red = cst.src[0]
  red = UOp(UOps.VECTORIZE, cst.dtype, tuple(x.src[0] for x in cst.src))
  return UOp(UOps.REDUCE, red.dtype, (red,) + fst_red.src[1:], fst_red.arg)

contractor = PatternMatcher([
  # contracts
  (UPat(UOps.CONTRACT, name="root"), replace_contract),
  # VECTORIZE after REDUCEs -> one REDUCE (breaks TestConv.test_two_binops_no_rerun)
  (UPat(UOps.VECTORIZE, name="cst", src=UPat(UOps.REDUCE)), cast_reduce),
])

reducer = PatternMatcher([
  (UPat(UOps.REDUCE, name="root"), replace_reduce),
  (UPat(UOps.WMMA, name="wmma"), expand_wmma),
  # image indexing. TODO: why can't this just go after the float stuff?
  (UPat({UOps.LOAD, UOps.STORE}, name="ls"), fix_image_idx),
])

# ***** float4 handling *****

def float4_expand_load(load, buf, ex, idx=UOp.const(dtypes.int, 0), idx2=None):
  if len(ex.src) != 4: return None
  if tuple(x.arg for x in ex.src if x.op is UOps.CONST) != tuple(range(len(ex.src))): return None
  if buf.dtype != PtrDType(dtypes.float) and not isinstance(buf.dtype, ImageDType): return None
  if idx2 is not None: idx = idx + idx2
  if not idx.divides(len(ex.src)): return None

  if load.dtype.scalar() != load.dtype: return None  # how does this happen?
  vec_load = UOp(UOps.LOAD, load.dtype.vec(len(ex.src)), (buf, idx))
  return UOp(UOps.EXPAND, load.dtype, tuple(UOp(UOps.GEP, load.dtype, (vec_load,), i) for i in range(len(ex.src))), ex.arg)

def float4_contract_store(buf, ex, var, store_allow_any_len, idx=UOp.const(dtypes.int, 0), idx2=None, idx3=None):
  if len(ex.src) not in [2, 4]: return None
  if tuple(x.arg for x in ex.src if x.op is UOps.CONST) != tuple(range(len(ex.src))): return None
  if buf.dtype != PtrDType(dtypes.float) and not isinstance(buf.dtype, ImageDType): return None
  if idx2 is not None: idx = idx + idx2
  if idx3 is not None: idx = idx + idx3
  if not idx.divides(len(ex.src)): return None

  new_var = UOp(UOps.CONTRACT, var.dtype.vec(len(ex.src)), (var,), (ex.arg,))
  return UOp(UOps.STORE, None, (buf, idx, new_var) + store_allow_any_len.src[3:])

def no_float4_alu(alu):
  if alu.dtype.count == 1: return None
  alus = tuple(UOp(UOps.ALU, alu.dtype.scalar(),
                   tuple(UOp(UOps.GEP, s.dtype.scalar(), (s,), i) for s in alu.src), alu.arg) for i in range(alu.dtype.count))
  return UOp(UOps.VECTORIZE, alu.dtype, alus)

float4_folding = PatternMatcher([
  (UOp(UOps.STORE, dtype=dtypes.float, src=(UOp.var("buf"), UOp.var("idx")+
    (UOp(UOps.EXPAND, src=tuple(UOp.const(dtypes.int, i) for i in range(4))).name("ex")+UOp.var("idx2")), UOp.var("var"))).name("store"),
    lambda buf, store, idx, idx2, ex, var: UOp(UOps.STORE, store.dtype, (buf, idx+idx2+ex, var), store.arg)),
  # float(2,4) load
  (UOp(UOps.LOAD, dtype=dtypes.float, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx")+UOp.var("idx2"))).name("load"),
    float4_expand_load),
  (UOp(UOps.LOAD, dtype=dtypes.float, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx"))).name("load"), float4_expand_load),
  (UOp(UOps.LOAD, dtype=dtypes.float, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex"))).name("load"), float4_expand_load),
  # float(2,4) store
  # TODO: fold ADDs into one UOp and remove add chains
  (UOp(UOps.STORE, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx")+UOp.var("idx2")+UOp.var("idx3"), UOp.var("var"))).name("store_allow_any_len"),
    float4_contract_store),
  (UOp(UOps.STORE, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx")+UOp.var("idx2"), UOp.var("var"))).name("store_allow_any_len"),
    float4_contract_store),
  (UOp(UOps.STORE, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex")+UOp.var("idx"), UOp.var("var"))).name("store_allow_any_len"), float4_contract_store),
  (UOp(UOps.STORE, src=(UOp.var("buf"),
    UOp(UOps.EXPAND).name("ex"), UOp.var("var"))).name("store_allow_any_len"), float4_contract_store),
  # no ALU on float4 (float4 constructor doesn't work in METAL/GPU)
  (UPat(UOps.ALU, name="alu"), no_float4_alu),
])

# ***** transcendental *****

transcendental_folding = PatternMatcher([
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat(name="x"),), arg=UnaryOps.EXP2), xexp2),
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat(name="d"),), arg=UnaryOps.LOG2), xlog2),
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat(name="d"),), arg=UnaryOps.SIN), xsin),
])

# ***** main rewriter *****

def reduce_before_expand(reduce_allow_any_len, expand, x):
  red = UOp(UOps.REDUCE, x.dtype, (x,)+reduce_allow_any_len.src[1:], reduce_allow_any_len.arg)
  gep = tuple(UOp(UOps.GEP, reduce_allow_any_len.dtype, (red,), i) for i in range(x.dtype.count))
  return UOp(expand.op, expand.dtype, gep, expand.arg)

def sum_collapse(phi_input, loop, val1, val2):
  for v1,v2 in [(val1, val2), (val2, val1)]:
    if loop not in v1.parents:
      loop_range = loop.src[1]-loop.src[0]
      ret = v1*loop_range.cast(v1.dtype)
      return UOp(UOps.PHI, phi_input.dtype, (phi_input, v2))+ret
  return None

def loop_collapse(loop_start, loop_end, compval, idx, mval, multconst, rng):
  if getenv("DISABLE_LOOP_COLLAPSE") or not rng.arg[1]: return None  # must be a REDUCE
  if mval.arg >= 0 or loop_start.arg != 0:
    # TODO: support and test this with other mvals and loop_starts
    if DEBUG >= 1: print(f"WARNING, NOT FOLDING: mval:{mval.arg} loop_start:{loop_start.arg}")
    return None
  comprange = UOp.min(loop_end, UOp.max(UOp.alu(BinaryOps.IDIV, idx-compval-mval, mval) + (loop_end-loop_start), loop_start))
  return UOp(UOps.UNMUL, multconst.dtype, (comprange.cast(multconst.dtype) * multconst, loop_end-loop_start))

# this is symbolic 2.0
constant_folder = PatternMatcher([
  # bigint is rewritten to int32
  (UPat({UOps.CONST, UOps.ALU, UOps.SPECIAL, UOps.RANGE, UOps.EXPAND}, dtype=dtypes.bigint, name="x"),
   lambda x: UOp(x.op, dtypes.int32, x.src, x.arg)),
  # VECTORIZE/GEP
  (UOp(UOps.GEP, src=(UOp(UOps.VECTORIZE).name("cast"),)).name("gep"), lambda gep, cast: cast.src[gep.arg]),
  (UOp(UOps.VECTORIZE, dtypes.float.vec(2), tuple(UOp(UOps.GEP, dtypes.float, src=(UOp.var('x'),), arg=i) for i in range(2))), lambda x: x),
  (UOp(UOps.VECTORIZE, dtypes.float.vec(4), tuple(UOp(UOps.GEP, dtypes.float, src=(UOp.var('x'),), arg=i) for i in range(4))), lambda x: x),
  (UOp(UOps.VECTORIZE, dtypes.float.vec(8), tuple(UOp(UOps.GEP, dtypes.float, src=(UOp.var('x'),), arg=i) for i in range(8))), lambda x: x),
  # tensor core with a 0 input is acc
  (UOp(UOps.WMMA, src=(UOp.const(None, 0.0), UOp.var(), UOp.var('acc'))), lambda acc: acc),
  (UOp(UOps.WMMA, src=(UOp.var(), UOp.const(None, 0.0), UOp.var('acc'))), lambda acc: acc),
  # tensor core cleanups
  (UOp(UOps.REDUCE, src=(UOp(UOps.EXPAND, src=tuple(UOp(UOps.GEP, dtypes.float, src=(UOp.var('x'),), arg=i) for i in range(2))).name("expand"),))
   .name("reduce_allow_any_len"), reduce_before_expand),
  (UOp(UOps.REDUCE, src=(UOp(UOps.EXPAND, src=tuple(UOp(UOps.GEP, dtypes.float, src=(UOp.var('x'),), arg=i) for i in range(8))).name("expand"),))
   .name("reduce_allow_any_len"), reduce_before_expand),
  (UOp.var("add") + UOp(UOps.WMMA).name("wmma"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
  # arange loop folding (early)
  (UPat(UOps.ALU, TernaryOps.WHERE, src=(UPat(UOps.ALU, BinaryOps.CMPLT, src=(
    UPat(UOps.ALU, BinaryOps.ADD, src=[UPat(name="idx"), UPat(UOps.ALU, BinaryOps.MUL, src=[UPat(UOps.CONST, name="mval"),
      UPat(UOps.RANGE, name="rng", src=(UPat(name="loop_start"), UPat(name="loop_end")))])]),
      UPat(UOps.CONST, name="compval"))), UPat(UOps.CONST, name="multconst"), UPat(UOps.CONST, 0))), loop_collapse),
  (UPat(UOps.ALU, TernaryOps.WHERE, src=(UPat(UOps.ALU, BinaryOps.CMPLT, src=(
    UPat(UOps.ALU, BinaryOps.ADD, src=[UPat(name="idx"), UPat(UOps.ALU, UnaryOps.NEG, src=[
      UPat(UOps.RANGE, name="rng", src=(UPat(name="loop_start"), UPat(name="loop_end")))])]),
      UPat(UOps.CONST, name="compval"))), UPat(UOps.CONST, name="multconst"), UPat(UOps.CONST, 0))),
      lambda **kwargs: loop_collapse(mval=UOp.const(dtypes.int, -1), **kwargs)),
  # sum collapse to mul (with possible GEP)
  (UPat(UOps.PHI, src=(UPat(UOps.DEFINE_ACC, name="phi_input", src=[UPat(UOps.CONST), UPat(UOps.RANGE, name="loop")]),
                       UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(name="val1"), UPat(name="val2"))))), sum_collapse),
  (UPat(UOps.PHI, src=(UPat(UOps.GEP, name="phi_input", src=(UPat(UOps.DEFINE_ACC, src=[UPat(UOps.CONST), UPat(UOps.RANGE, name="loop")]),)),
                       UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(name="val1"), UPat(name="val2"))))), sum_collapse),
  # deal with UNMUL
  (UPat(UOps.ALU, BinaryOps.MUL, [UPat(UOps.CONST, name="c1"), UPat(UOps.UNMUL, src=[UPat(UOps.CONST, name="c2"), UPat(name="v")])]),
   lambda c1,c2,v: v if c1.arg == c2.arg else None),
  (UOp(UOps.UNMUL, src=(UOp.const(None, 0).name('zero'), UOp.var())), lambda zero: zero),
  (UOp(UOps.UNMUL).name('unmul').cast().name('root'), lambda root,unmul: UOp(UOps.UNMUL, root.dtype, (unmul.src[0].cast(root.dtype), unmul.src[1]))),
  # indexing (with a multiply offset)!
  (UOp.var('idx').eq(UOp(UOps.RANGE).name("rng")).where(
    UOp(UOps.LOAD, src=(UOp.var("buf"), UOp.var('add')+UOp.var('mul')*UOp(UOps.RANGE).name("rng"))).name("ld"), UOp.const(None, 0.0)),
    lambda idx,rng,buf,add,mul,ld: UOp(UOps.UNMUL, ld.dtype, (UOp(ld.op, ld.dtype, (buf, add+mul*idx)), rng.src[1]-rng.src[0]))),
  # other arange folders
  (UOp.cvar("c1") - (UOp.var("x") + UOp.cvar("c2")), lambda c1, c2, x: (c1-c2)-x),  # c1 - (x + c2) -> (c1-c2) - x
  # max on special can go away (TODO: special should be variable, same thing applies)
  (UOp.max(UOp.cvar('c'), UOp(UOps.SPECIAL).name('s')), lambda c,s: c if (s.arg[2]-1) <= c.arg else None),
  (UOp.max(UOp.cvar('c'), UOp(UOps.SPECIAL).name('s')+UOp.cvar('c2')), lambda c,s,c2: (s+c2) if 0 >= c.arg else None),  # TODO: generic
  (UOp.max(UOp.cvar('c'), -(UOp(UOps.SPECIAL).name('s')+UOp.cvar('c2'))), lambda c,s,c2: -(s+c2) if -(s.arg[2]-1+c2.arg) >= c.arg else None),
  # max on range can go away (ugh: copy of SPECIAL, and with/without const)
  (UOp.max(UOp.cvar('c'), UOp(UOps.RANGE).name('s')), lambda c,s: s if s.src[0].arg >= c.arg else None),  # TODO: generic
  (UOp.max(UOp.cvar('c'), UOp(UOps.RANGE).name('s')+UOp.cvar('c2')), lambda c,s,c2: (s+c2) if s.src[0].arg >= c.arg else None),  # TODO: generic
  (UOp.max(UOp.cvar('c'), -(UOp(UOps.RANGE).name('s'))), lambda c,s: -s if -(s.src[1].arg-1) >= c.arg else None),
  (UOp.max(UOp.cvar('c'), -(UOp(UOps.RANGE).name('s')+UOp.cvar('c2'))), lambda c,s,c2: -(s+c2) if -(s.src[1].arg-1+c2.arg) >= c.arg else None),
  # const rules
  (UPat(UOps.GEP, name="root", src=(UPat(UOps.CONST, name="c"),)), lambda root, c: UOp.const(root.dtype, c.arg)),
  (UPat(UOps.CAST, name="root", src=UPat(UOps.CONST, name="c")), lambda root, c: UOp.const(root.dtype, c.arg)),
  (UPat(UOps.VECTORIZE, name="root", src=UPat(UOps.CONST, name="c")), lambda root, c: UOp.const(root.dtype, c.arg)),
  # a phi on a DEFINE_ACC without loops or a CONST is a noop. this is for correctness, not just speed
  (UPat(UOps.PHI, src=(UPat(UOps.DEFINE_ACC, name="acc"), UPat(name="acc"))), lambda acc: UOp.cast(acc.src[0], acc.dtype)),
  (UPat(UOps.PHI, src=(UPat(UOps.DEFINE_ACC, src=(UPat(UOps.CONST),)), UPat(name="x"))), lambda x: x),
  (UPat(UOps.PHI, src=(UPat(UOps.CONST), UPat(name="x"))), lambda x: x),
  # a DEFINE_ACC without inputs is a const + GEP on a const is the const
  (UPat(UOps.DEFINE_ACC, name="root", src=(UPat(UOps.CONST),)), lambda root: UOp.cast(root.src[0], root.dtype)),
  (UPat(UOps.GEP, name="root", src=(UPat(UOps.CONST, name="x"),)), lambda root,x: UOp.const(root.dtype, x.arg)),
  # max -2147483648
  (UOp.max(UOp.var('x'), UOp.const(dtypes.int, -2147483648)), lambda x: x),
  # bool < False is always false, True < bool is always false
  (UOp.var().lt(UOp.const(dtypes.bool, False)), lambda: UOp.const(dtypes.bool, False)),
  (UOp.const(dtypes.bool, True).lt(UOp.var()), lambda: UOp.const(dtypes.bool, False)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UOp.alu(TernaryOps.WHERE, UOp.var(), UOp.var("val"), UOp.var("val")), lambda val: val),
  (UOp.alu(TernaryOps.WHERE, UOp.cvar('gate'), UOp.var('c0'), UOp.var('c1')), lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  (UPat(UOps.ALU, name="root", src=UPat(UOps.CONST)), lambda root: UOp.const(root.dtype, exec_alu(root.arg, root.dtype, [x.arg for x in root.src]))),
  # ** self folding **
  (-(-UOp.var('x')), lambda x: x),    # -(-x) -> x
  (UOp.var('x') + 0, lambda x: x),    # x+0 -> x
  (UOp.var('x') * 1, lambda x: x),    # x*1 -> x
  (UOp.var('x') * -1, lambda x: -x),  # x*-1 -> -x
  (UOp.var('x') // UOp.var('x'), lambda x: UOp.const(x.dtype, 1)), # x//x -> 1
  (UOp.var('x') // 1, lambda x: x),   # x//1 -> x
  (UOp.var('x') // -1, lambda x: -x), # x//-1 -> -x
  (UOp.var('x') / UOp.var('x'), lambda x: UOp.const(x.dtype, 1)), # x/x -> 1
  (UOp.var('x') / UOp.cvar('c'), lambda x,c: x*exec_alu(UnaryOps.RECIP, c.dtype, [c.arg])),    # x/c -> x*(1/c)
  (UOp.var('x', dtype=dtypes.bool).max(UOp.const(dtypes.bool, False)), lambda x: x),  # max(x, False) -> x
  # ** zero folding **
  #x*0 -> 0 or 0*x -> 0
  #if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UOp.var('x') * 0, lambda x: UOp.const(x.dtype, float('nan') if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  (UOp.var('x') - UOp.var('x'), lambda x: UOp.const(x.dtype, 0)),   # x-x -> 0
  # ** load/store folding **
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.load(UOp.var("buf"), UOp.var("idx"))), lambda buf,idx:UOp(UOps.NOOP)),
  # ** two stage add/sub folding **
  ((UOp.var('x') + UOp.cvar('c1')) + UOp.cvar('c2'), lambda x,c1,c2: x+UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, c2.arg]))),
  ((UOp.var('x') - UOp.cvar('c1')) + UOp.cvar('c2'), lambda x,c1,c2: x+UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c2.arg, -c1.arg]))),
  # *** rules from symbolic ***
  # two stage mul, (x*c1)*c2 = x*(c1*c2)
  ((UOp.var("x") * UOp.cvar("c1")) * UOp.cvar("c2"), lambda x,c1,c2: x*UOp.const(x.dtype, exec_alu(BinaryOps.MUL, x.dtype, [c1.arg, c2.arg]))),
  # -(x+y) -> -x + -y
  #(-(UOp.var("x") + UOp.var("y")), lambda x,y: (-x)+(-y)),
  # x%1 -> 0
  (UOp.var("x") % UOp.const(None, 1), lambda x: UOp.const(x.dtype, 0)),
  # (x*c0)+(x*c1) -> x*(c0+c1)
  (UOp.var("x") * UOp.cvar("c0") + UOp.var("x") * UOp.cvar("c1"), lambda x,c0,c1: x*exec_alu(BinaryOps.ADD, x.dtype, [c0.arg, c1.arg])),
  # (x*c0)+(y*c0) -> (x+y)*c0
  #((UOp.var("x") * UOp.cvar("c0")) + (UOp.var("y") * UOp.cvar("c0")), lambda x,y,c0: c0*(x+y)),
  # (x*c0)//c0 -> x
  ((UOp.var("x") * UOp.cvar("c0")) // UOp.cvar("c0"), lambda x,c0: x if c0.arg != 0 else None),
  # (x*x2)/x2 -> x
  ((UOp.var("x") * UOp.var("x2")) / UOp.var("x2"), lambda x,x2: x),
  # (x//c0)//c1 -> x//(c0*c1)
  ((UOp.var("x") // UOp.cvar("c0")) // UOp.cvar("c1"), lambda x,c0,c1: x//UOp.const(x.dtype, exec_alu(BinaryOps.MUL, x.dtype, [c0.arg, c1.arg]))),
  # (x/x1)/x2 -> x/(x1*x2)
  ((UOp.var("x") / UOp.var("x2")) / UOp.var("x3"), lambda x,x2,x3: x/(x2*x3)),
  # c0 + x < c1 -> x < c1 - c0
  ((UOp.cvar("c0") + UOp.var("x")).lt(UOp.cvar("c1")),
    lambda x,c0,c1: UOp.lt(x, UOp.const(x.dtype, exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, -c0.arg])))),
  # (x+x*c0)-> x*(c0+1)
  (UOp.var("x") + UOp.var("x") * UOp.cvar("c0"), lambda x,c0: x*UOp.const(x.dtype, c0.arg+1)),
  # x!=0 -> (bool)x
  (UOp.var("x").ne(0), lambda x: x.cast(dtypes.bool)),
  # bool != 1 -> not bool
  (UOp.var("x", dtype=dtypes.bool).ne(1), lambda x: -x),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.alu(TernaryOps.WHERE, UOp.var("gate"), UOp.var("alt"), UOp.load(UOp.var("buf"), UOp.var("idx")))),
   lambda buf, idx, gate, alt: UOp.store(buf, idx, alt, gate)),
  # store float4/float2 directly (remove VECTORIZE/GEP)
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp(UOps.VECTORIZE, src=tuple(
    UOp(UOps.GEP, arg=i, src=(UOp.var("val"),)) for i in range(4)))), UOp.store),
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp(UOps.VECTORIZE, src=tuple(
    UOp(UOps.GEP, arg=i, src=(UOp.var("val"),)) for i in range(2)))), UOp.store),
  # VECTORIZE-PHI-GEP -> PHI-VECTORIZE
  (UPat(UOps.VECTORIZE, name="root", src=tuple(
    UPat(UOps.PHI, src=(UPat(UOps.GEP, i, src=(UPat(name="val"),)), UPat(name=f"v{i}"))) for i in range(4))),
    lambda root, val, v0, v1, v2, v3: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1, v2, v3))))),
  (UPat(UOps.VECTORIZE, name="root", src=tuple(
    UPat(UOps.PHI, src=(UPat(UOps.GEP, i, src=(UPat(name="val"),)), UPat(name=f"v{i}"))) for i in range(2))),
    lambda root, val, v0, v1: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1))))),
  # NEG/CMPLT -> CMPLT
  (UOp.lt(-UOp.var('x'), UOp.cvar('c', dtypes.int)), lambda c,x: UOp.lt(UOp.const(c.dtype, -c.arg), x)),
  # cast NOOP (NOTE: it's str to deal with PtrDType)
  (UPat(UOps.CAST, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  (UPat(UOps.VECTORIZE, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  # fold gated LOAD/STORE
  (UOp.load(UOp.var("buf"), UOp.var("idx"), UOp.const(dtypes.bool, True), UOp.cvar("var")), lambda buf,idx,var: UOp.load(buf, idx, dtype=var.dtype)),
  (UOp.load(UOp.var("buf"), UOp.var("idx"), UOp.const(dtypes.bool, True), UOp.cvar("var"), UOp.var("barrier")),
   lambda buf,idx,var,barrier: UOp.load(buf, idx, barrier, dtype=var.dtype)),
  (UOp.load(UOp.var(), UOp.var(), UOp.const(dtypes.bool, False), UOp.cvar("var")), lambda var: var),
  (UOp.load(UOp.var(), UOp.var(), UOp.const(dtypes.bool, False), UOp.cvar("var"), UOp.var()), lambda var: var),
  (UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.var("val"), UOp.const(dtypes.bool, True)), UOp.store),
  (UOp.store(UOp.var(), UOp.var(), UOp.var(), UOp.const(dtypes.bool, False)), lambda: UOp(UOps.NOOP)),
  # remove NOOPs from SINK
  (UPat(UOps.SINK, name="root"),
    lambda root: UOp(UOps.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not UOps.NOOP)) != len(root.src) else None)
])

constant_folder_w_f4 = PatternMatcher(constant_folder.patterns + float4_folding.patterns)

# *** uop graph ***

def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], in_degree:Dict[UOp, int]):
  if u in children: return
  children[u] = []
  for x in u.src:
    get_children_dfs(x, children, in_degree)
    children[x].append(u)
  in_degree[u] = len(u.src)

def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:
  nodes: Dict[Tuple, UOp] = {}
  replace: Dict[UOp, UOp] = {}
  def __inner_rewrite(n:UOp) -> UOp:
    if n in replace: return replace[n]
    replace_source = (n.op, n.dtype, tuple(__inner_rewrite(y) for y in n.src), n.arg)
    if found := nodes.get(replace_source): replace[n] = found
    else: nodes[replace_source] = replace[n] = __inner_rewrite(new_x) if (new_x := pm.rewrite(x:=UOp(*replace_source))) else x
    return replace[n]
  return __inner_rewrite(sink)

class UOpGraph:
  def __init__(self, sink:Union[UOp, List[UOp]], opts:Optional[Renderer]=None):
    self.sink: UOp = sink if isinstance(sink, UOp) else UOp(UOps.SINK, None, tuple(sink))
    # used by linearizer
    self._uops: Optional[List[UOp]] = None
    self.opts = opts
    self.folder = constant_folder if opts is None or not opts.supports_float4 else constant_folder_w_f4
    if TRANSCENDENTAL >= 2 or (opts is not None and TRANSCENDENTAL >= 1 and opts.device in {"CLANG", "LLVM"}):
      # TODO: slow to rebuild this...
      self.folder = PatternMatcher(self.folder.patterns + transcendental_folding.patterns)

  def __reduce__(self): return self.__class__, (self.sink, self.opts)
  def __iter__(self) -> Iterator[UOp]: return iter(self.uops)
  def __getitem__(self, index) -> UOp: return self.uops[index]

  def vars(self) -> List[Variable]: return sorted([x.arg for x in self.uops if x.op is UOps.DEFINE_VAR], key=lambda v: v.expr)
  def globals(self) -> List[Tuple[int, bool]]: return [x.arg for x in self.uops if x.op is UOps.DEFINE_GLOBAL]

  @property
  def uops(self) -> List[UOp]:
    if self._uops is None: self.linearize()
    return cast(List[UOp], self._uops)

  def graph(self):
    from tinygrad.engine.graph import graph_uops
    graph_uops(self.uops)

  def print(self):
    for i,u in enumerate(self):
      formatted_parents = [self.uops.index(x) if x.op is not UOps.CONST else f"{x.arg}" for x in u.src]
      print(f"{i:4d} {str(u.op):20s}: {str(u.dtype) if u.dtype is not None else '':25s} " f"{str(formatted_parents):32s} {u.arg}")

  cnt = 0
  def linearize(self, extra_pm:Optional[PatternMatcher]=None):
    global acc_number
    acc_number = 0

    # NOTE: relinearizering should be okay
    #assert self._uops is None, "already linearized"

    # fixup gated stores with an IF block to save extra local loads
    @functools.lru_cache(None)
    def _dfs(u:UOp, gate:UOp) -> UOp:
      if u.op is UOps.LOAD and u.src[-1].op is UOps.BARRIER:
        if_uop = UOp(UOps.IF, None, (gate, u.src[-1]))
        return UOp(u.op, u.dtype, u.src[:-1]+(if_uop,), u.arg)
      if (replace_source:=tuple(_dfs(x, gate) for x in u.src)) != u.src: return UOp(u.op, u.dtype, replace_source, u.arg)
      return u
    sink_srcs = list(self.sink.src)
    for i, s in enumerate(sink_srcs[:]):
      # breaks for WMMA
      if all(x.op is not UOps.WMMA for x in s.parents):
        if s.op is UOps.STORE and len(s.src) == 4 and (rw:=_dfs(s, s.src[3])) != s:
          sink_srcs[i] = UOp(rw.op, rw.dtype, rw.src[:3], rw.arg)
    sink = UOp(UOps.SINK, None, tuple(sink_srcs))

    # do graph rewrite
    sink = graph_rewrite(sink, self.folder)
    if extra_pm: sink = graph_rewrite(sink, PatternMatcher(self.folder.patterns+extra_pm.patterns))

    UOpGraph.cnt += 1
    if UOpGraph.cnt != getenv("DEBUG_EXPAND", 0):
      # do contracts/reduces
      sink = graph_rewrite(sink, contractor)
      sink = graph_rewrite(sink, reducer)

      # do upcasts (after reduce unrolls and rewrites)
      expands = list(sorted(x for x in sink.sparents if x.op is UOps.EXPAND))
      new_nodes = expand_nodes(sink.sparents, expands, sink)
      sink = UOp(UOps.SINK, None, tuple(flatten([x.src for x in new_nodes])))  # merge the sinks

    # do graph rewrite (2)
    sink = graph_rewrite(sink, self.folder)
    if extra_pm: sink = graph_rewrite(sink, PatternMatcher(self.folder.patterns+extra_pm.patterns))

    # filter nodes that don't link to a sink
    # BFS toposort
    children: Dict[UOp, List[UOp]] = {}
    in_degree: Dict[UOp, int] = {}
    get_children_dfs(sink, children, in_degree)

    @functools.lru_cache(None)
    def get_recursive_children(x:UOp, end:UOps, include_self=False) -> Set[UOp]:
      if x.op is UOps.SINK: return set()
      return set.union(set((x,)) if include_self else set(), *([get_recursive_children(u, end, True) for u in children[x] if x.op is not end]))

    # scope children impact the toposort and END* insertion
    scope_children = {p:get_recursive_children(p, END_FOR_UOP[p.op][0]) for p in reversed(in_degree) if p.op in END_FOR_UOP}

    queue:List[Tuple[int, UOp]] = []
    def push(u:UOp):
      priority = 0
      # prefer uops that are loop children
      for l, ss in scope_children.items():
        if l.op is UOps.RANGE and u in ss: priority -= l.arg[0]*1000 + l.arg[1]
      heapq.heappush(queue, (priority, u))

    for u in children:
      if in_degree[u] == 0: push(u)

    scope_end: Dict[UOp, UOp] = {}
    self._uops = []
    while queue:
      p,x = heapq.heappop(queue)
      if DEBUG >= 7: print(p,x)
      if x in scope_children: scope_end[x] = x
      if x.op is UOps.DEFINE_ACC:
        idx = min([self._uops.index(l) for l in x.src if l.op is UOps.RANGE])
        self._uops.insert(idx, x)
      else: self._uops.append(x)
      for u, ss in scope_children.items():
        if x in ss:
          ss.remove(x)
          if len(ss) == 0: scope_end[u] = x
      for u in children[x]:
        in_degree[u] -= 1
        if in_degree[u] == 0: push(u)

    # end scopes in toposort order
    for u, x in scope_end.items(): self._uops.insert(self._uops.index(x)+1, UOp(END_FOR_UOP[u.op][1], None, (u,)))

    # sanity checks (NOTE: these can cause things to be skipped in BEAM)
    try:
      type_verify(self.uops)
      assert self._uops[-1].op is UOps.SINK, f"didn't end with SINK, ended with {self._uops[-1]}"
      assert all(x.op not in {UOps.EXPAND, UOps.CONTRACT, UOps.REDUCE, UOps.UNMUL} for x in self._uops), "fake UOp left in list"
      # TODO: this should be enabled, and the valid clause should be removed
      # NOTE: multiple identical stores to DEFINE_LOCAL is okay
      assert len(all_stores := [x.src[0:2]+x.src[3:] for x in self._uops if x.op is UOps.STORE and x.src[0].op is not UOps.DEFINE_LOCAL]) \
        == len(dedup(all_stores)), "repeated stores in uops"
    except AssertionError as e:
      self.print()
      if getenv("GRAPHUOPS"): self.graph()
      raise e

    # strip the SINK
    self._uops = self._uops[:-1]

    if getenv("FUZZ_UOPS"):
      from test.external.fuzz_uops import fuzz_uops
      self._fuzz_paths = fuzz_uops(self)
