from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Set, cast, TYPE_CHECKING, Any, DefaultDict, Callable
import functools, itertools, heapq, math, operator
from collections import defaultdict
from tinygrad.dtype import dtypes, PtrDType, ImageDType, ConstType
from tinygrad.ops import UnaryOps, BinaryOps, exec_alu, UOp, UOps, END_FOR_UOP, type_verify, print_uops, identity_element
from tinygrad.ops import UPat, PatternMatcher, graph_rewrite
from tinygrad.helpers import DEBUG, getenv, flatten, dedup, TRANSCENDENTAL, AMX, prod, CI, partition, all_same
from tinygrad.codegen.transcendental import xexp2, xlog2, xsin, TRANSCENDENTAL_SUPPORTED_DTYPES
if TYPE_CHECKING: from tinygrad.renderer import Renderer

# ***** float4/image store handling *****

def fold_expanded(ex, buf):
  if buf.dtype != PtrDType(dtypes.float) and buf.dtype != PtrDType(dtypes.half) and not isinstance(buf.dtype, ImageDType): return None
  new_srcs = dedup(list(ex.src))
  old_new_srcs = new_srcs[:]
  is_load, is_image = new_srcs[0].op is UOps.LOAD, isinstance(buf.dtype, ImageDType)

  # first, extract all the relevant offsets
  offsets_rootsrc: DefaultDict[Any, dict] = defaultdict(dict)
  for i,s in enumerate(new_srcs):
    if s.dtype.count != 1 or (is_image and s.src[1].dtype.count == 2): continue
    idx = s.src[1]
    if idx.arg is BinaryOps.ADD and idx.src[1].op is UOps.CONST: root_src, arg = idx.src[0], idx.src[1].arg
    elif idx.op is UOps.CONST: root_src, arg = "CONST", idx.arg
    else: root_src, arg = idx, 0
    # add gates for gated
    if len(s.src) >= 4: root_src = (s.src[3], root_src)
    assert arg not in offsets_rootsrc[root_src]
    offsets_rootsrc[root_src][arg] = i

  # then rewrite everything we can
  lengths = [4] if is_image else ([8,4,2] if buf.dtype == PtrDType(dtypes.half) and getenv("ALLOW_HALF8") else ([16,8,4,2] if AMX else [4,2]))
  used = set()
  for rootsrc, offsets in offsets_rootsrc.items():
    for o in offsets:
      for fold_length in lengths:
        if all((rootsrc,o+i) not in used and o+i in offsets for i in range(fold_length)):
          load_1 = new_srcs[offsets[o]]
          new_src = list(load_1.src)
          if not new_src[1].divides(fold_length): continue
          # for images, we rewrite the index. it must evenly divide 4 from the above check
          if is_image:
            new_src[1] = UOp(UOps.VECTORIZE, dtypes.int.vec(2), ((new_src[1] // 4) % buf.dtype.shape[1], (new_src[1] // (4 * buf.dtype.shape[1]))))
          # vectorize the store/loadconst
          if not is_load or len(new_src) >= 4:
            new_src[2] = UOp(UOps.VECTORIZE, new_src[2].dtype.vec(fold_length), tuple(new_srcs[offsets[o+i]].src[2] for i in range(fold_length)))
          # generate the folded new_srcs
          if is_load:
            new_load = UOp(UOps.LOAD, load_1.dtype.vec(fold_length), tuple(new_src))
            for i in range(fold_length): new_srcs[offsets[o+i]] = new_load.gep(i)
          else:
            for i in range(fold_length): new_srcs[offsets[o+i]] = UOp(UOps.STORE, dtypes.void, tuple(new_src)) if i == 0 else None
          for i in range(fold_length): used.add((rootsrc,o+i))

  # dedup expand for LOAD
  if is_load and len(old_new_srcs) != len(ex.src): new_srcs = [new_srcs[old_new_srcs.index(s)] for s in ex.src]
  # remove Nones for STORE
  return UOp(ex.op, ex.dtype, tuple(x for x in new_srcs if x is not None), ex.arg) if len(used) else None

def fix_unfoldable_image_load(load:UOp, buf:UOp):
  if not isinstance(buf.dtype, ImageDType) or load.src[1].dtype.count == 2: return None
  id4 = load.src[1] % 4
  new_src = list(load.src)
  # TODO: copied logic from above
  new_src[1] = UOp(UOps.VECTORIZE, dtypes.int.vec(2), ((load.src[1] // 4) % buf.dtype.shape[1], (load.src[1] // (4 * buf.dtype.shape[1]))))
  if len(new_src) >= 4:
    new_src[2] = UOp(UOps.VECTORIZE, new_src[2].dtype.vec(4), tuple(new_src[2] for _ in range(4)))
  vec_load = UOp(UOps.LOAD, load.dtype.vec(4), tuple(new_src))
  return functools.reduce(lambda ret, i: id4.ne(i).where(ret, vec_load.gep(i)), range(4), load.const_like(float('nan')))

float4_folding = PatternMatcher([
  (UPat(UOps.VECTORIZE, src=UPat(UOps.LOAD, src=(UPat.var("buf"), UPat()), allow_any_len=True), name="ex"), fold_expanded),
  (UPat((UOps.BARRIER, UOps.SINK), src=UPat(UOps.STORE, src=(UPat.var("buf"), UPat(), UPat()), allow_any_len=True), name="ex"), fold_expanded),
])

# ***** mod *****

def _get_chain(x:UOp, sep:BinaryOps):
  if x.op is UOps.ALU and x.arg is sep:
    for s in x.src: yield from _get_chain(s, sep)
  else: yield x

def mod_folding(x:UOp, c:int) -> Optional[UOp]:
  # simplify x % c, None means no change

  # simple cancel mod case
  if 0 < c and 0 <= x.vmin and (quotient:=x.vmin//c) == x.vmax//c: return x-quotient*c

  remainder, something_changed = [], False
  for u in _get_chain(x, BinaryOps.ADD):
    if (factor:=u.const_factor())%c != factor:
      divides = u.divides(factor)*(factor%c)
      assert divides is not None
      remainder.append(divides)
      something_changed = True
    elif u.op is UOps.ALU and u.arg is BinaryOps.MOD and (s1:=u.src[1]).op is UOps.CONST and s1.arg%c == 0:
      remainder.append(u.src[0])
      something_changed = True
    else: remainder.append(u)
  if not something_changed: return None
  return functools.reduce(operator.add, remainder)%c if remainder else x.const_like(0)

def div_folding(x:UOp, c:int) -> Optional[UOp]:
  # simplify x // c, None means no change

  # simple cancel div case
  if 0 <= x.vmin and x.vmax < c: return x.const_like(0)

  quotient, remainder, rem_const, something_changed, gcd, divisor = [], [], 0, False, c, 1
  for u in _get_chain(x, BinaryOps.ADD):
    if u.op is UOps.CONST:
      # add all const together first
      if rem_const != 0: something_changed = True
      rem_const += u.arg
    elif (factor:=u.const_factor())%c == 0:
      if factor:
        divides = u.divides(c)
        assert divides is not None
        quotient.append(divides)
      something_changed = True
    else:
      # divisor is the smallest common divisor of all MULs
      if u.op is UOps.ALU and u.arg is BinaryOps.MUL and factor > 1 and c % factor == 0 and (divisor == 1 or divisor > factor): divisor = factor
      remainder.append(u)
      gcd = math.gcd(gcd, factor)

  # handle the const
  if rem_const%c != rem_const:
    something_changed = True
    quotient.append(x.const_like(rem_const//c))
    rem_const = rem_const%c
  if rem_const != 0: remainder.append(x.const_like(rem_const))

  # x // c -> quotient + (remainder // div) // (c // div)
  div = gcd if gcd > 1 else divisor

  if not something_changed: return newx//(c//div) if 1 < div < c and (newx:=div_folding(x, div)) is not None else None
  rem:Optional[UOp] = functools.reduce(operator.add, remainder) if remainder else None
  quo:Optional[UOp] = functools.reduce(operator.add, quotient) if quotient else None
  if quo is None: return x.const_like(0) if rem is None else cast(UOp, div_folding(rem, div))//(c//div)
  return quo if rem is None else cast(UOp, div_folding(rem, div))//(c//div)+quo

def lt_folding(x:UOp, c:int) -> Optional[UOp]:
  if (newx:=div_folding(x,c)) is not None and newx.op is UOps.ALU and newx.arg is BinaryOps.IDIV: return newx.src[0].lt(newx.src[1])
  return cast(UOp, x.divides(g)).lt(c//g) if ((g:=math.gcd(x.const_factor(), c)) > 1) else None

def fold_unrolled_divs(divs:UOp):
  # div pattern in unrolled arange
  # example: (x//4+(x+1)//4+(x+2)//4+(x+3)//4 -> x
  add_chain, seen_const, ans = list(_get_chain(divs, BinaryOps.ADD)), [], None
  for u in add_chain:
    if not (u.op is UOps.ALU and u.arg is BinaryOps.IDIV and u.src[1].op is UOps.CONST and u.src[1].arg==len(add_chain)): return None
    # assumed CONST is the last of an ADD
    if (s0:=u.src[0]).op is UOps.ALU and s0.arg is BinaryOps.ADD and s0.src[1].op is UOps.CONST and s0.src[1].op is UOps.CONST:
      seen_const.append(s0.src[1].arg)
      s0 = s0.src[0]
    else: seen_const.append(0)
    if ans is None: ans = s0
    if ans != s0: return None
  return ans if ans is not None and sorted(seen_const)==list(range(len(add_chain))) else None

# ***** image load valid simplification *****

def is_irreducible(u:UOp): return u.op in (UOps.DEFINE_VAR, UOps.SPECIAL, UOps.RANGE)

def canonicalize_simplex(X:UOp) -> Optional[UOp]:
  # (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
  # returns x0 + x1 + ... in such case, or None if not
  changed, ret = False, []
  for u in _get_chain(X, BinaryOps.ADD):
    # assumed the const is the last src of MUL
    if u.op is UOps.ALU and u.arg is BinaryOps.MUL and u.src[1].op is UOps.CONST and u.src[1].arg > 0:
      changed = True
      u = u.src[0]
    if not (is_irreducible(u) and u.vmin >= 0): return None
    ret.append(u)
  return functools.reduce(operator.add, ret) if changed else None

def is_increasing(f:UOp):
  # is f a monotonically increasing function regards its input
  if f.op is UOps.CONST or is_irreducible(f): return True
  if f.op is UOps.ALU and f.arg is BinaryOps.ADD: return is_increasing(f.src[0]) and is_increasing(f.src[1])
  if f.op is UOps.ALU and f.arg in (BinaryOps.MUL, BinaryOps.IDIV) and f.src[1].op is UOps.CONST and f.src[1].arg >= 0: return is_increasing(f.src[0])
  return False  # False if not sure

def replace_uop(uop:UOp, old:UOp, new:UOp):
  # replace all `old` in `uop` to `new`
  return new if uop is old else UOp(uop.op, uop.dtype, tuple(replace_uop(s, old, new) for s in uop.src), uop.arg)

def parse_valid(valid:UOp) -> Tuple[UOp, bool, int]:
  # if it's X <= c, returns X, True, c
  # if it's X >= c, returns X, False, c

  # (X < c).ne(True) -> X >= c
  if valid.op is UOps.ALU and valid.arg is BinaryOps.CMPNE and valid.src[1].op is UOps.CONST and valid.src[1].arg == 1 and \
    (s0:=valid.src[0]).op is UOps.ALU and s0.arg is BinaryOps.CMPLT and s0.src[1].op is UOps.CONST: return s0.src[0], False, s0.src[1].arg
  # X < c -> X <= c-1
  if valid.op is UOps.ALU and valid.arg is BinaryOps.CMPLT and valid.src[1].op is UOps.CONST: return valid.src[0], True, valid.src[1].arg-1
  raise ValueError(f"not able to parse {valid=}")

def simplify_valid_image_load(load:UOp, buf:UOp):
  if not isinstance(buf_dtype:=buf.dtype, ImageDType) or len(load.src) < 4: return None
  buf, idx, invalid_val, valid = load.src
  start_idx = idx

  # first, parse valid into {expr: (lower_bound, upper_bound)}
  bounds:DefaultDict[UOp, List[Optional[ConstType]]] = defaultdict(lambda: [None, None])
  for stmt in _get_chain(valid, BinaryOps.AND):
    expr, is_upper, c = parse_valid(stmt)
    bounds[expr][int(is_upper)] = c

  # simplify idx given that valid is True
  for uop,v in bounds.items():
    # some expr has lower bound > upper bound -> valid is an empty set
    if v[0] is not None and v[1] is not None and v[0] > v[1]:
      return UOp(UOps.LOAD, load.dtype, (buf, idx, invalid_val, valid.const_like(False)))
    new = UOp.define_var("fake", uop.dtype, uop.vmin if v[0] is None else v[0], uop.vmax if v[1] is None else v[1])
    newidx = replace_uop(graph_rewrite(replace_uop(idx, uop, new), constant_folder), new, uop)
    if newidx.key != idx.key: idx = newidx

  # can drop valid if idx is out of bound when valid is False
  drop_stmt = []
  for stmt in _get_chain(valid, BinaryOps.AND):
    X, is_upper_bound, c = parse_valid(stmt)
    # if X <= c, check if it's out of bound when X = c+1
    # if X >= c, check if it's out of bound when X = c-1
    test_value = c + 1 if is_upper_bound else c - 1
    for i,b in zip(idx.src, (buf_dtype.shape[1], buf_dtype.shape[0])):
      if is_increasing(i):
        rw = graph_rewrite(replace_uop(i, X, X.const_like(test_value)), constant_folder)
        if rw.vmin >= b or rw.vmax < 0: drop_stmt.append(stmt)

  if drop_stmt or idx.key != start_idx.key:
    new_valid = functools.reduce(operator.and_, ss) if (ss:=[s for s in _get_chain(valid, BinaryOps.AND) if s not in drop_stmt]) else None
    return UOp(UOps.LOAD, load.dtype, (buf, idx, invalid_val, new_valid)) if new_valid else UOp(UOps.LOAD, load.dtype, (buf, idx))
  return None

# ***** transcendental *****

@functools.lru_cache(None)
def transcendental_folding(ops):
  return PatternMatcher([(UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat.var("d"),), arg=k), cast(Callable, v))
                         for k,v in ((UnaryOps.EXP2, xexp2), (UnaryOps.LOG2, xlog2), (UnaryOps.SIN, xsin)) if k not in ops])

# ***** threefry *****

def threefry2x32(x: UOp, seed: UOp):
  # split x into two uint32, since x in a uint64
  x0, x1 = (x & 0xffffffff).cast(dtypes.uint32), ((x // 2**32) & 0xffffffff).cast(dtypes.uint32)

  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  ks = [0x0, (seed := seed.cast(dtypes.uint32)) ^ 0x1BD11BDA, seed]
  xr = [x0 + ks[-1], x1 + ks[0]]
  for i in range(5):
    for r in rotations[i % 2]: xr[0], xr[1] = (x0 := xr[0] + xr[1]), x0 ^ ((xr[1] * 2**r) + (xr[1] // 2**(32 - r)))
    xr = [(xr[0] + ks[i % 3]), (xr[1] + ks[(i + 1) % 3] + i + 1)]

  return xr[1].cast(dtypes.uint64) * 2**32 | xr[0].cast(dtypes.uint64)

# ***** main rewriter *****

def loop_collapse(compval, idx, multconst, rng:UOp, reduce, idx2=None, idx3=None, extra=None, vec=None, ne=None, mval:UOp=UOp.const(dtypes.int32, 1)):
  if getenv("DISABLE_LOOP_COLLAPSE") or rng not in reduce.src: return None  # must be the right REDUCE
  loop_start, loop_end = rng.src
  mval_arg = mval.arg
  if loop_start.arg != 0:
    # TODO: support and test this with other mvals and loop_starts
    if DEBUG >= 1: print(f"WARNING, NOT FOLDING: mval:{mval.arg} loop_start:{loop_start.arg}")
    return None
  if idx2 is not None: idx = idx + idx2
  if idx3 is not None: idx = idx + idx3
  if vec is not None:
    # idx, mval, loop_start, loop_end
    def dvec(x): return UOp(UOps.VECTORIZE, x.dtype.vec(vec.dtype.count), src=(x,)*vec.dtype.count)
    idx, mval, loop_start, loop_end = dvec(idx), dvec(mval), dvec(loop_start), dvec(loop_end)
  if mval_arg > 0 and ne is not None:
    comprange = UOp.min(loop_end, UOp.max((idx-compval)//mval + (loop_end-loop_start), loop_start))
  elif mval_arg < 0 and ne is None:
    comprange = UOp.min(loop_end, UOp.max((idx-compval-mval)//mval + (loop_end-loop_start), loop_start))
  else:
    return None
  new_reduce_op = comprange.cast(multconst.dtype) * multconst
  ret = UOp(UOps.REDUCE, reduce.dtype, (new_reduce_op,) + tuple(x for x in reduce.src[1:] if x is not rng), reduce.arg)
  if extra is not None: ret = ret + UOp(UOps.REDUCE, reduce.dtype, (extra,) + reduce.src[1:], reduce.arg)
  return ret

def index_collapse(idx,rng,buf,ld,reduce,add=UOp.const(dtypes.int, 0),mul=UOp.const(dtypes.int, 1)):
  if rng not in reduce.src: return None
  return UOp(reduce.op, reduce.dtype, (UOp(ld.op, ld.dtype, (buf, add+mul*idx, ld.const_like(0), idx.ge(rng.src[0]) & idx.lt(rng.src[1]))),)+
             tuple(x for x in reduce.src[1:] if x is not rng), reduce.arg)

# TODO: there's a lot shared with no_vectorized_wmma here
def gep_through_wmma(gep:UOp, wmma:UOp):
  out_sz = prod(x[1] for x in wmma.arg[6][-1])
  wmma_idxs = gep.arg[::out_sz]
  for i in range(out_sz):
    if tuple(x-i for x in gep.arg[i::out_sz]) != wmma_idxs: return None
  tsrcs = []
  for s,sz in zip(wmma.src, wmma.arg[6]):
    src_args = []
    ssz = prod(x[1] for x in sz)
    for w in wmma_idxs: src_args += list(range((w//out_sz)*ssz, (w//out_sz)*ssz + ssz))
    tsrcs.append(s.gep(tuple(src_args)))
  return UOp(UOps.WMMA, gep.dtype, tuple(tsrcs), wmma.arg)

def no_vectorized_wmma(wmma:UOp):
  out_sz = prod(x[1] for x in wmma.arg[6][-1])
  if wmma.dtype.count == out_sz: return None
  tsrcs = []
  for s,sz in zip(wmma.src, wmma.arg[6]):
    ssz = prod(x[1] for x in sz)
    tsrcs.append([s.gep(tuple(range(grp, grp+ssz))) for grp in range(0, s.dtype.count, ssz)])
  wmmas = [UOp(UOps.WMMA, wmma.dtype.scalar().vec(out_sz), tsrc, wmma.arg) for tsrc in zip(*tsrcs)]
  wmma_ex = flatten([[e.gep(i) for i in range(out_sz)] for e in wmmas])
  return UOp(UOps.VECTORIZE, wmma.dtype, tuple(wmma_ex))

# this is symbolic 2.0
constant_folder = PatternMatcher([
  # bool ADD is OR, MUL is AND. prevents other rules to rewrite bool ADD/MUL incorrectly
  (UPat(UOps.ALU, dtypes.bool, arg=BinaryOps.ADD, name="x"), lambda x: UOp(x.op, x.dtype, x.src, BinaryOps.OR)),
  (UPat(UOps.ALU, dtypes.bool, arg=BinaryOps.MUL, name="x"), lambda x: UOp(x.op, x.dtype, x.src, BinaryOps.AND)),
  # self ASSIGN is just self
  (UPat(UOps.ASSIGN, src=(UPat.var('x'), UPat.var('x'))), lambda x: x),
  # VECTORIZE/GEP: the expander rule allows tuple GEP creation, this is just for removal
  (UPat(UOps.VECTORIZE, src=UPat(UOps.GEP, src=(UPat(name="x"),)), name="vec"),
   lambda vec,x: x if x.dtype == vec.dtype and tuple(y.arg[0] for y in vec.src) == tuple(range(len(vec.src))) else None),
  # reorder ALU/VECTORIZE
  (UPat(UOps.ALU, src=(UPat(UOps.VECTORIZE, src=UPat(name='x')), UPat(UOps.VECTORIZE, src=UPat(name='y'))), name='alu'),
   lambda x,y,alu: UOp(UOps.VECTORIZE, alu.dtype, (UOp(UOps.ALU, alu.dtype.scalar(), (x,y), alu.arg),)*alu.dtype.count)),
  # VECTORIZE of a single element is just that element
  (UPat(UOps.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
  # VECTORIZE void is SINK
  (UPat(UOps.VECTORIZE, dtype=dtypes.void, src=UPat(UOps.BARRIER, name='b')), lambda b: b),
  (UPat(UOps.VECTORIZE, dtype=dtypes.void, name='x'), lambda x: UOp(UOps.SINK, dtypes.void, x.src)),
  # GEP/VECTORIZE, GEP/GEP, GEP/CONST, GEP/VCONST
  (UPat(UOps.GEP, src=(UPat(UOps.GEP, name='g2'),), name='g1'),
   lambda g1, g2: g2.src[0].gep(tuple(g2.arg[g1.arg[i]] for i in range(g1.dtype.count)))),
  (UPat(UOps.GEP, src=(UPat(UOps.VECTORIZE, name="vec"),), name="gep"),
   lambda gep, vec: UOp(UOps.VECTORIZE, gep.dtype, tuple(vec.src[i] for i in gep.arg)) if len(gep.arg) > 1 else vec.src[gep.arg[0]]),
  (UPat(UOps.GEP, src=(UPat.cvar("c", vec=False),), name="gep"), lambda gep, c: gep.const_like(c.arg)),
  (UPat(UOps.GEP, src=(UPat(UOps.VCONST, name="c"),), name="gep"), lambda gep, c: gep.const_like(tuple(c.arg[x] for x in gep.arg))),
  # push all GEPs through ALUs (fix arange stuff)
  (UPat(UOps.GEP, src=(UPat((UOps.ALU, UOps.CAST, UOps.BITCAST), name='alu'),), name='gep'),
   lambda gep,alu: UOp(alu.op, alu.dtype.scalar().vec(gep.dtype.count), tuple(x.gep(gep.arg) for x in alu.src), alu.arg)),
  # push some GEPs through WMMAs
  (UPat(UOps.GEP, src=(UPat(UOps.WMMA, name="wmma"),), name="gep"), gep_through_wmma),
  # tensor core with a 0 input is acc
  (UPat(UOps.WMMA, src=(UPat.const(None, 0.0), UPat.var(), UPat.var("acc"))), lambda acc: acc),
  (UPat(UOps.WMMA, src=(UPat.var(), UPat.const(None, 0.0), UPat.var("acc"))), lambda acc: acc),
  # tensor core cleanups
  (UPat.var("add") + UPat(UOps.WMMA, name="wmma"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
  # threefry
  (UPat(UOps.ALU, dtype=dtypes.uint64, src=(UPat.var("x"), UPat.var("seed")), arg=BinaryOps.THREEFRY), threefry2x32),
  # arange loop folding
  (UPat(UOps.REDUCE, src=(UPat.any(m2:=UPat.any(
    m1:=(UPat.var("idx") + UPat.cvar("mval") * UPat(UOps.RANGE, name="rng")),
    m1 + UPat.var("idx2"), m1 + UPat.var("idx2") + UPat.var("idx3"), UPat(UOps.VECTORIZE, name="vec", src=m1))
    .lt(UPat.cvar("compval")).where(UPat.cvar("multconst"), UPat.const(None, 0)), m2 + UPat.var("extra")),),
    arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  # arange loop folding (new ge)
  (UPat(UOps.REDUCE, src=(UPat.any(m2:=UPat.any(
    m1:=(UPat.var("idx") + UPat.any(UPat.cvar("mval") * UPat(UOps.RANGE, name="rng"), UPat(UOps.RANGE, name="rng"))),
    m1 + UPat.var("idx2"), m1 + UPat.var("idx2") + UPat.var("idx3"), UPat(UOps.VECTORIZE, name="vec", src=m1))
    .lt(UPat.cvar("compval")).ne(UPat(UOps.CONST, name="ne", arg=True))
    .where(UPat.cvar("multconst"), UPat.const(None, 0)), m2 + UPat.var("extra")),),
    arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  # unrolled arange div folding
  (UPat(UOps.ALU, name="divs", src=[UPat(), UPat(UOps.ALU, arg=BinaryOps.IDIV)], arg=BinaryOps.ADD), fold_unrolled_divs),
  # indexing, with cast or where
  (UPat(UOps.REDUCE, src=(UPat.var("idx").eq(UPat(UOps.RANGE, name="rng")).cast()*
    UPat(UOps.LOAD, src=(UPat.var("buf"), UPat.any(UPat.var("add")+UPat.var("mul")*UPat(UOps.RANGE, name="rng"), UPat(UOps.RANGE, name="rng"))),
         name="ld"),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True), index_collapse),
  (UPat(UOps.REDUCE, src=(UPat.var("idx").eq(UPat(UOps.RANGE, name="rng")).where(
    UPat(UOps.LOAD, src=(UPat.var("buf"), UPat.any(UPat.var("add")+UPat.var("mul")*UPat(UOps.RANGE, name="rng"), UPat(UOps.RANGE, name="rng"))),
         name="ld"), UPat.const(None, 0.0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True), index_collapse),
  # max folding
  (UPat.max(UPat.var("x"), UPat.var("y")), lambda x,y: x if x.vmin >= y.vmax else y if x.vmax <= y.vmin else None),
  # GEP/CAST const rules
  (UPat(UOps.CAST, name="root", src=UPat.cvar("c")), lambda root, c: root.const_like(c.arg)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  (UPat.var().where(UPat.var("val"), UPat.var("val")), lambda val: val),
  (UPat.cvar("gate", vec=False).where(UPat.var("c0"), UPat.var("c1")), lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  (UPat(UOps.ALU, name="root", src=UPat((UOps.VCONST, UOps.CONST))),
   lambda root: root.const_like(exec_alu(root.arg, root.dtype, [x.arg for x in root.src]))),
  # ** self folding **
  # cast NOOP (NOTE: it's str to deal with PtrDType)
  (UPat(UOps.CAST, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  (UPat(UOps.REDUCE, src=(UPat.var("x"),)), lambda x: x),  # a REDUCE without ranges is a NOOP
  (UPat.var("x") + 0, lambda x: x),    # x+0 -> x
  (UPat.var("x") * 1, lambda x: x),    # x*1 -> x
  (UPat.var("x") // UPat.var("x"), lambda x: x.const_like(1)), # x//x -> 1
  (UPat.var("x") // 1, lambda x: x),   # x//1 -> x
  (UPat.var("x") // -1, lambda x: -x), # x//-1 -> -x
  (UPat.var("x") / UPat.var("x"), lambda x: x.const_like(1)), # x/x -> 1
  ((UPat.var("x") * UPat.var("x2")) / UPat.var("x2"), lambda x,x2: x), # (x*x2)/x2 -> x
  (UPat.var("x", dtype=dtypes.bool) & UPat.cvar("c", vec=False), lambda x,c: x if c.arg else c),
  (UPat.var("x", dtype=dtypes.bool) | UPat.cvar("c", vec=False), lambda x,c: c if c.arg else x),
  # ** zero folding **
  # x*0 -> 0 or 0*x -> 0
  # if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (UPat.var("x") * 0, lambda x: x.const_like(float("nan") if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  # min==max -> CONST (slow!)
  (UPat((UOps.ALU, UOps.DEFINE_VAR), name="x"), lambda x: x.const_like(x.vmin) if x.vmin == x.vmax else None),
  # ** load/store folding **
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat.load(UPat.var("buf"), UPat.var("idx"))), lambda buf,idx:UOp(UOps.NOOP)),
  # ** two stage add/mul folding **
  ((UPat.var("x") + UPat.cvar("c1")) + UPat.cvar("c2"), lambda x,c1,c2: x+(c1+c2)),
  ((UPat.var("x") * UPat.cvar("c1")) * UPat.cvar("c2"), lambda x,c1,c2: x*(c1*c2)),
  ((UPat.var("x") & UPat.cvar("c1")) & UPat.cvar("c2"), lambda x,c1,c2: x&(c1&c2)),
  ((UPat.var("x") | UPat.cvar("c1")) | UPat.cvar("c2"), lambda x,c1,c2: x|(c1|c2)),
  # *** rules from symbolic ***
  # ** lt **
  # c0*x<c1 for positive int c0,c1
  ((UPat.cvar("c0", vec=False)*UPat.var("x")).lt(UPat.cvar("c1", vec=False)),
   lambda x,c0,c1: x.lt(math.ceil(c1.arg/c0.arg)) if dtypes.is_int(x.dtype) and c0.arg > 0 and c1.arg > 0 else None),
  # c0*x<c1 for negative int c0 and non-positive c1
  ((UPat.cvar("c0", vec=False)*UPat.var("x")).lt(UPat.cvar("c1", vec=False)),
   lambda x,c0,c1: (-x).lt(-(math.floor(-c1.arg/-c0.arg))) if dtypes.is_int(x.dtype) and c0.arg < 0 and c0.arg != -1 and c1.arg <= 0 else None),
  # x//c0<c1 for positive int c0
  ((UPat.var("x")//UPat.cvar("c0", vec=False)).lt(UPat.cvar("c1", vec=False)),
   lambda x,c0,c1: x.lt(c1.arg*c0.arg) if dtypes.is_int(x.dtype) and c0.arg > 0 else None),
  # mul add lt
  (((UPat.cvar("c0", vec=False)*UPat.var("x"))+UPat.var("x2")).lt(UPat.cvar("c1", vec=False)),
   lambda x,x2,c0,c1: x.lt(c1//c0) if c1.arg % c0.arg == 0 and c0.arg > x2.vmax and x2.vmin >= 0 else None),
  # generic lt folding
  (UPat.var("x").lt(UPat.cvar("c", vec=False)),
    lambda x,c: lt_folding(x, c.arg) if 0 < c.arg and dtypes.is_int(x.dtype) and not dtypes.is_unsigned(x.dtype) else None),
  # canonicalize a simplex with positive coefficients > 0
  # not x < 1 -> X > 0
  (UPat.var("x").lt(1).ne(True), lambda x: newx.lt(1).ne(True) if dtypes.is_int(x.dtype) and (newx:=canonicalize_simplex(x)) is not None else None),
  # ** div **
  # # div folding
  (UPat.var("x") // UPat.cvar("c", vec=False), lambda x,c:
   newx if 0 < c.arg and not dtypes.is_unsigned(x.dtype) and (newx:=div_folding(x,c.arg)) is not None else None),
  # ** mod **
  # mod folding
  (UPat.var("x") % UPat.cvar("c", vec=False), lambda x,c: newx if 0 < c.arg and (newx:=mod_folding(x,c.arg)) is not None else None),
  # ** combine terms **
  (UPat.var("x")%UPat.cvar("c")+(UPat.var("x")//UPat.cvar("c"))*UPat.cvar("c"), lambda x,c: x), # (x%c)+(x//c)*c = x
  (UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1"), lambda x,c0,c1: x*(c0+c1)), # (x*c0)+(x*c1) -> x*(c0+c1)
  (UPat.var("x") + UPat.var("x") * UPat.cvar("c"), lambda x,c: x*(c+1)), # (x+x*c)-> x*(c+1)
  (UPat.var("x") + UPat.var("x"), lambda x: x*2), # (x+x)-> x*2
  ((UPat.var("x") // UPat.cvar("c0")) // UPat.cvar("c1"), lambda x,c0,c1: x//(c0*c1)), # (x//c0)//c1 -> x//(c0*c1)
  ((UPat.var("x") / UPat.var("x2")) / UPat.var("x3"), lambda x,x2,x3: x/(x2*x3)), # (x/x2)/x3 -> x/(x2*x3)
  (-1 * (UPat.var("x") + UPat.var("y")), lambda x,y: (-x)+(-y)),  # -(x+y) -> -x + -y
  ((UPat.cvar("c0") + UPat.var("x")).lt(UPat.cvar("c1")), lambda x,c0,c1: UOp.lt(x, c1-c0)),  # c0 + x < c1 -> x < c1 - c0
  # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
  ((UPat.var("x") + UPat.var("y")) * UPat.cvar("c"), lambda x,y,c: x*c+y*c if dtypes.is_int(x.dtype) else None),
  # x!=0 -> (bool)x
  (UPat.var("x").ne(0), lambda x: x.cast(dtypes.bool.vec(x.dtype.count))),
  # bitwise noops
  ((UPat.var("x") & UPat.var("x")), lambda x: x),
  ((UPat.var("x") | UPat.var("x")), lambda x: x),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat.var("gate").where(UPat.var("alt"), UPat.load(UPat.var("buf"), UPat.var("idx")))),
   lambda buf, idx, gate, alt: UOp.store(buf, idx, alt, gate)),
  # fold gated LOAD/STORE
  (UPat.load(UPat.var("buf"), UPat.var("idx"), UPat.var("var"), UPat.const(None, True)),
   lambda buf,idx,var: UOp.load(buf, idx, dtype=var.dtype)),
  (UPat.load(UPat.var("buf"), UPat.var("idx"), UPat.var("var"), UPat.const(None, True), UPat.var("barrier")),
   lambda buf,idx,var,barrier: UOp.load(buf, idx, barrier, dtype=var.dtype)),
  (UPat.load(UPat.var(), UPat.var(), UPat.var("var"), UPat.const(None, False)), lambda var: var),
  (UPat.load(UPat.var(), UPat.var(), UPat.var("var"), UPat.const(None, False), UPat.var()), lambda var: var),
  (UPat.store(UPat.var("buf"), UPat.var("idx"), UPat.var("val"), UPat.const(None, True)),
   lambda buf,idx,val: UOp.store(buf, idx, val)), # pylint: disable=unnecessary-lambda
  (UPat.store(UPat.var(), UPat.var(), UPat.var(), UPat.const(None, False)), lambda: UOp(UOps.NOOP)),
  # remove NOOPs from SINK
  (UPat(UOps.SINK, name="root"),
    lambda root: UOp(UOps.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not UOps.NOOP)) != len(root.src) else None),
  # remove EXPANDs from SINK/BARRIER
  (UPat(UOps.BARRIER, src=(UPat((UOps.VECTORIZE, UOps.SINK), name='sink'),)), lambda sink: UOp(UOps.BARRIER, dtypes.void, sink.src)),
  (UPat(UOps.SINK, name="root"),
    lambda root: UOp(UOps.SINK, root.dtype, tuple(flatten(x.src if x.op in {UOps.SINK, UOps.EXPAND} else (x,) for x in root.src)), root.arg)
      if any(x.op in {UOps.SINK, UOps.EXPAND} for x in root.src) else None),
  # ** move add consts to end (NOTE: this is still happening before constant folding) **
  (UPat(UOps.ALU, arg=BinaryOps.ADD, src=(UPat.cvar("c1"), UPat.var("x"))), lambda c1,x: x+c1 if x.op not in (UOps.CONST, UOps.VCONST) else None),
  (UPat(UOps.ALU, arg=BinaryOps.ADD, src=(UPat.var("x"), UPat.cvar("c1"))) + UPat.var("y"), lambda x,c1,y: (x+y)+c1),
])

# *** uop expander ***

def _expand_arg_to_idx(args:Tuple[Tuple[int, int], ...], rpk:Dict[int, int]) -> int:
  idx, mul = 0, 1
  for axis,m in args[::-1]:
    idx += rpk[axis] * mul
    mul *= m
  return idx

def _choices_from_args(args:Tuple[Tuple[int, int], ...]) -> List[Dict[int, int]]:
  return [dict(x) for x in itertools.product(*[zip(itertools.repeat(axis), range(m)) for axis,m in args])]

@functools.lru_cache(None)
def _swizzle_args(cargs:Tuple[Tuple[int, int], ...], eargs:Tuple[Tuple[int, int], ...], exclude_args:Tuple[int, ...]) -> List[int]:
  return [_expand_arg_to_idx(eargs, {**rpk, **{x:0 for x in exclude_args}} if exclude_args else rpk) for rpk in _choices_from_args(cargs)]

def do_expand(root:UOp):
  expands = [x for x in root.src if x.op is UOps.EXPAND]
  if len(expands) == 0: return None
  # NOTE: we 0 out the reduce axis for WMMA. in theory they should all be the same, but is this always correct?
  exclude_args = tuple(dedup(root.arg[-1] + tuple(y[0] for y in flatten(root.arg[-2])))) if root.op is UOps.WMMA else ()
  if all_same(expands_args:=[x.arg for x in expands]) and len(exclude_args) == 0:
    # if there's only one expand arg, it's okay to use it (optimization)
    expand_args = expands[0].arg
  else:
    # otherwise, we sort them and GEP
    expand_args = tuple(x for x in sorted(dedup(flatten(expands_args))) if x[0] not in exclude_args)
  expand_sz = prod([x[1] for x in expand_args])
  new_srcs = []
  for i,src in enumerate(root.src):
    if src.op is UOps.EXPAND:
      if root.op is UOps.IF and i == 0:
        # IF means OR on first arg to IF
        new_srcs.append(functools.reduce(operator.__or__, [src.src[0].gep(i) for i in range(expand_sz)]))
      elif expand_args == src.arg:
        # just remove the expand
        new_srcs.append(src.src[0])
      else:
        lst = _swizzle_args(expand_args, src.arg, exclude_args)
        # if the base dtype is > 1, put those at the end
        if src.dtype.count > 1: lst = flatten([[i*src.dtype.count+j for j in range(src.dtype.count)] for i in lst])
        new_srcs.append(src.src[0].gep(tuple(lst)))
    else:
      # non-EXPAND input
      if (root.op in {UOps.LOAD, UOps.STORE} and i == 0) or (root.op is UOps.REDUCE and i != 0):
        # for the first arg of LOAD/STORE and the RANGE args of REDUCE, just pass them through ignoring EXPANDS
        new_srcs.append(src)
      elif src.dtype.count > 1:
        # put any input dtype > 1 grouped together
        new_srcs.append(UOp(UOps.VECTORIZE,
                            src.dtype.scalar().vec(expand_sz*src.dtype.count), tuple(src.gep(i) for i in range(src.dtype.count))*expand_sz))
      else:
        # repeat the arg
        new_srcs.append(src.broadcast(expand_sz))

  new_arg = root.arg
  if root.op is UOps.GEP:
    assert root.dtype.count == 1
    # is this right?
    new_arg = tuple(range(root.arg[0], new_srcs[0].dtype.count, new_srcs[0].dtype.count // expand_sz))
  nsrc = UOp(root.op, root.dtype.scalar().vec(root.dtype.count*expand_sz), tuple(new_srcs), new_arg)
  return UOp(UOps.EXPAND, root.dtype, (nsrc,), expand_args)

acc_number = 0
def do_reduce(root:UOp):
  global acc_number
  reduce_parented, reduce_unparented = partition(root.src[1:], lambda x: x in root.src[0].sparents)
  ret = root.src[0]
  if len(reduce_parented):
    acc = UOp(UOps.DEFINE_ACC, root.dtype,
              (root.const_like(identity_element(root.arg, root.dtype.scalar())),) + tuple(reduce_parented), (acc_number,))
    acc_number += 1
    ret = UOp(UOps.ASSIGN, root.dtype, (acc, acc.alu(root.arg, ret)))
  # for MAX, we can just ignore the unparented
  if root.arg is BinaryOps.ADD:
    for r in reduce_unparented:ret = ret * (r.src[1]-r.src[0]).cast(ret.dtype.scalar()).broadcast(ret.dtype.count)
  return ret

def do_contract(con:UOp):
  ex = con.src[0]
  # CONTRACT without EXPAND repeats the element VECTORIZED
  if ex.op is not UOps.EXPAND: return UOp(UOps.VECTORIZE, con.dtype, con.src*con.dtype.count)
  # CONTRACT may remove several axes from EXPAND
  assert con.dtype.count == prod([x[1] for x in con.arg]), "dtype is wrong"
  idxs = []
  for rpk in _choices_from_args(new_ex_args:=tuple(x for x in ex.arg if x not in con.arg)):
    idxs += [_expand_arg_to_idx(ex.arg, {**rpk, **lrpk}) for lrpk in _choices_from_args(con.arg)]
  return UOp(UOps.EXPAND, con.dtype, (ex.src[0].gep(tuple(idxs)),), new_ex_args)

def no_vectorized_alu(alu):
  if alu.dtype.count == 1: return None
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.count))
  return UOp(UOps.VECTORIZE, alu.dtype, alus)

def create_gate(root:UOp) -> Optional[UOp]:
  @functools.lru_cache(None)
  def _gate_srcs(u:UOp, gate:UOp) -> UOp:
    if u.op is UOps.BARRIER: return u
    if u.op is UOps.LOAD and u.src[-1].op is UOps.BARRIER:
      return UOp(u.op, u.dtype, u.src[:-1]+(UOp(UOps.IF, dtypes.void, (gate, u.src[-1])),), u.arg)
    return u if (replace_source:=tuple(_gate_srcs(x, gate) for x in u.src)) == u.src else UOp(u.op, u.dtype, replace_source, u.arg)
  return None if len(root.src) == 3 or (ret:=_gate_srcs(root, root.src[3])) is root else ret

expander = PatternMatcher([
  (UPat(UOps.VECTORIZE, src=UPat(UOps.CONST), name="vec"), lambda vec: UOp.const(vec.dtype, tuple(x.arg for x in vec.src))),
  (UPat(UOps.VECTORIZE, src=UPat(UOps.GEP, src=(UPat(name="x"),)), name="vec"), lambda vec,x: x.gep(tuple(y.arg[0] for y in vec.src))),
  # create gate MUST BE BEFORE expander
  (UPat(UOps.STORE, name="root"), create_gate),
  # double expand
  (UPat(UOps.EXPAND, name="outer", src=(UPat(UOps.EXPAND, name="inner"),)),
   lambda outer, inner: UOp(UOps.EXPAND, outer.dtype, (inner.src[0],), inner.arg+outer.arg)),
  # do expansion
  (UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.GEP, UOps.WMMA, UOps.LOAD, UOps.STORE,
         UOps.VECTORIZE, UOps.REDUCE, UOps.IF), name="root", custom_early_reject=set([(UOps.EXPAND, None)])), do_expand),
  (UPat(UOps.CONTRACT, name="con"), do_contract),
  # remove EXPANDs from SINK
  (UPat(UOps.SINK, name="root"),
   lambda root: UOp(UOps.SINK, root.dtype, a, root.arg)
    if len(a:=tuple(flatten(x.src if x.op is UOps.EXPAND else (x,) for x in root.src))) != len(root.src) else None),
  # BARRIERs aren't actually expanded
  (UPat(UOps.BARRIER, src=(UPat(UOps.EXPAND, name="ex"),)),
   lambda ex: UOp(UOps.EXPAND, dtypes.void, (UOp(UOps.BARRIER, dtypes.void, ex.src),)*len(ex.src), ex.arg)),
  # empty EXPAND is NOOP
  (UPat(UOps.EXPAND, src=(UPat.var('x'),), arg=()), lambda x: x),
  # EXPAND GEP (needed for WMMA, generalize this) -> vectorized ALU
  (UPat(UOps.EXPAND, name="ex", src=tuple(UPat.var('x').gep(i)+UPat.var('y').gep(i) for i in range(256 if AMX else 8))),
    lambda ex,x,y: UOp(UOps.EXPAND, ex.dtype, tuple((x+y).gep(i) for i in range(256 if AMX else 8)), ex.arg)),
])

def no_vectorized_load_store(ls:UOp):
  idx = ls.src[1]
  if idx.dtype.count == 1: return None
  # ugh, the meaning of a dtype.count idx is overloaded
  if ls.op is UOps.LOAD and idx.dtype.count != ls.dtype.count: return None
  if ls.op is UOps.STORE and idx.dtype.count != ls.src[2].dtype.count: return None
  tv = [UOp(ls.op, ls.dtype.scalar(), (ls.src[0],) + tuple(j.gep(i) for j in ls.src[1:])) for i in range(idx.dtype.count)]
  return UOp(UOps.VECTORIZE, ls.dtype, tuple(tv))

def no_vectorized_acc(acc:UOp):
  if acc.dtype.count == 1: return None
  alus = tuple(UOp(acc.op, acc.dtype.scalar(),
    tuple(s.gep(i) if j == 0 else s for j,s in enumerate(acc.src)), acc.arg+(i,)) for i in range(acc.dtype.count))
  return UOp(UOps.VECTORIZE, acc.dtype, alus)

def delete_redundant_gates(root:UOp) -> Optional[UOp]:
  @functools.lru_cache(None)
  def find_gate(x:UOp) -> Optional[UOp]:
    if x.op is UOps.IF: return x
    return next((ret for s in x.src if (ret:=find_gate(s)) is not None), None)
  if len(root.src) == 3 or (gate:=find_gate(root)) is None or gate.src[0] is not root.src[3]: return None
  return UOp(UOps.STORE, root.dtype, root.src[:3], root.arg)

just_reduce = PatternMatcher([
  # do reduce
  (UPat(UOps.REDUCE, name="root"), do_reduce),
])

devectorize = PatternMatcher([
  # no ALU on vectorized dtypes
  (UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.ASSIGN), name="alu"), no_vectorized_alu),
  (UPat(UOps.WMMA, name="wmma"), no_vectorized_wmma),
  (UPat(UOps.DEFINE_ACC, name="acc"), no_vectorized_acc),
  (UPat((UOps.LOAD, UOps.STORE), name="ls"), no_vectorized_load_store),
])

reducer = PatternMatcher([
  (UPat(UOps.CONST, name='c'),
   lambda c: UOp(UOps.VECTORIZE, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.count) if c.dtype.count > 1 else None),
  (UPat(UOps.VCONST, name='c'), lambda c: UOp(UOps.VECTORIZE, c.dtype, tuple(UOp.const(c.dtype.scalar(), x) for x in c.arg))),
  (UPat(UOps.GEP, name='gep'), lambda gep: UOp(UOps.VECTORIZE, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  # delete_redundant_gates (after expand, is this still needed?)
  (UPat(UOps.STORE, name="root"), delete_redundant_gates),
  # late fixup of unfoldable image loads
  (UPat(UOps.LOAD, src=(UPat.var("buf"), UPat()), allow_any_len=True, name="load"), fix_unfoldable_image_load),
  # image load valid simplification
  (UPat(UOps.LOAD, src=(UPat.var("buf"), UPat()), allow_any_len=True, name="load"), simplify_valid_image_load),
])

no_pyint = PatternMatcher([(UPat((UOps.CONST, UOps.VCONST, UOps.ALU, UOps.SPECIAL, UOps.RANGE, UOps.EXPAND, UOps.VECTORIZE), name="x"),
  lambda x: UOp(x.op, dtypes.int32.vec(x.dtype.count), x.src, x.arg) if x.dtype.scalar() == dtypes.pyint else None)])

# *** uop graph ***

def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], srcs:Dict[UOp, Dict[UOp, None]], in_degree:Dict[UOp, int]):
  if u in children: return srcs[u]
  srcs[u] = {}
  children[u] = []
  for x in u.src:
    srcs[u].update(get_children_dfs(x, children, srcs, in_degree))
    if x.op is UOps.RANGE and x.arg[1]: srcs[u][x] = None
    children[x].append(u)
  in_degree[u] = len(u.src)
  return srcs[u]

linearize_cnt = 0
def full_graph_rewrite(sink:UOp, opts:Optional[Renderer]=None) -> UOp:
  global linearize_cnt, acc_number
  assert sink.op is UOps.SINK, f"sink isn't sink, it's {sink.op}"
  folder = constant_folder + transcendental_folding(tuple() if TRANSCENDENTAL >= 2 or opts is None else tuple(opts.code_for_op.keys()))

  # do graph rewrite
  acc_number = 0
  sink = graph_rewrite(sink, folder)

  # rewrite pyint to int32
  sink = graph_rewrite(sink, no_pyint)

  # expand
  linearize_cnt += 1
  if linearize_cnt != (de:=getenv("DEBUG_EXPAND", 0)) and de != -1:
    sink = graph_rewrite(sink, folder+expander)
    if getenv("DO_REDUCE", 1):
      sink = graph_rewrite(sink, folder+just_reduce)
      sink = graph_rewrite(sink, folder+(devectorize+float4_folding if opts is not None and opts.supports_float4 else devectorize))
      sink = graph_rewrite(sink, folder+reducer)

  # for PTX only
  if opts is not None and opts.extra_matcher is not None: sink = graph_rewrite(sink, folder+opts.extra_matcher)
  return sink

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is UOps.SINK, f"sink isn't sink, it's {sink.op}"
  # filter nodes that don't link to a sink
  # BFS toposort
  children: Dict[UOp, List[UOp]] = {}
  range_srcs: Dict[UOp, Dict[UOp, None]] = {}
  in_degree: Dict[UOp, int] = {}
  get_children_dfs(sink, children, range_srcs, in_degree)

  @functools.lru_cache(None)
  def get_recursive_children(x:UOp, end:UOps, include_self=False) -> Set[UOp]:
    if x.op is UOps.SINK: return set()
    return set.union({x} if include_self else set(), *([get_recursive_children(u, end, True) for u in children[x] if x.op is not end]))

  # scope children impact the toposort and END* insertion
  scope_children = {p:get_recursive_children(p, END_FOR_UOP[p.op][0]) for p in reversed(in_degree) if p.op in END_FOR_UOP}
  range_phi = {r:[p for p in scope_children[r] if p.op is UOps.ASSIGN] for r in scope_children if r.op is UOps.RANGE}

  queue:List[Tuple[int, UOp]] = []
  def push(u:UOp):
    priority = 0
    # prefer ranges that depend on the least number of independent ranges
    if u.op is UOps.RANGE and u.arg[1]:
      priority += u.arg[0]
      for p in range_phi[u]:
        priority += 10000*len([r for r in range_srcs[p] if not any(i in range_phi[u] for i in range_phi[r])])
    # prefer uops that are loop children
    else:
      priority -= sum([(l.arg[0]+1) + 1000*l.arg[1] for l,ss in scope_children.items() if l.op is UOps.RANGE and u in ss])
    heapq.heappush(queue, (priority, u))

  for u in children:
    if in_degree[u] == 0: push(u)

  scope_end: Dict[UOp, UOp] = {}
  _uops: List[UOp] = []
  while queue:
    p,x = heapq.heappop(queue)
    if DEBUG >= 7: print(f"{p:5d}",x)
    if x in scope_children: scope_end[x] = x
    if x.op is UOps.DEFINE_ACC:
      idx = min([_uops.index(l) for l in x.src if l.op is UOps.RANGE])
      _uops.insert(idx, x)
    else: _uops.append(x)
    for u, ss in scope_children.items():
      if x in ss:
        ss.remove(x)
        if len(ss) == 0: scope_end[u] = x
    for u in children[x]:
      in_degree[u] -= 1
      if in_degree[u] == 0: push(u)

  # end scopes in toposort order
  for u, x in scope_end.items(): _uops.insert(_uops.index(x)+1, UOp(END_FOR_UOP[u.op][1], dtypes.void, (u,)))

  # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  if not skip_check:
    bad_ops = dedup([x.op for x in _uops if x.op in {UOps.EXPAND, UOps.CONTRACT, UOps.REDUCE, UOps.REDUCE_AXIS, UOps.SHAPETRACKER}])
    try:
      type_verify(_uops)
      assert _uops[-1].op is UOps.SINK, f"didn't end with SINK, ended with {_uops[-1]}"
      assert len(bad_ops) == 0, f"bad UOps left in list: {bad_ops}"
      assert not any(x.dtype == dtypes.pyint for x in _uops), "can't return UOp with pyint"
      # TODO: this should be enabled, and the valid clause should be removed
      # NOTE: multiple identical stores to DEFINE_LOCAL is okay
      # NOTE: for PTX you have to propogate through some the calculations to determine if it is a store to DEFINE_LOCAL
      def _islocalbuf(u: UOp): return u.op is UOps.DEFINE_LOCAL or any(_islocalbuf(x) for x in u.src if u.op in [UOps.ALU, UOps.CAST])
      all_stores = [x.src[0:2]+x.src[3:] for x in _uops if x.op is UOps.STORE and not _islocalbuf(x.src[0])]
      assert len(all_stores) == len(dedup(all_stores)), "repeated stores in uops"
    except AssertionError as e:
      print_uops(_uops)
      if not CI and not getenv("VIZ"):
        from tinygrad.engine.graph import graph_uops
        graph_uops(_uops)
      raise e

  # strip the SINK
  return _uops[:-1]
