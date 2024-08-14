from __future__ import annotations
from typing import Iterator, Optional, Tuple, Dict, List, Set, Union, cast, TYPE_CHECKING, Any, DefaultDict, Callable
import functools, itertools, heapq, math, operator
from collections import defaultdict
from tinygrad.dtype import dtypes, PtrDType, ImageDType, DType
from tinygrad.ops import UnaryOps, BinaryOps, exec_alu
from tinygrad.helpers import DEBUG, getenv, flatten, dedup, TRANSCENDENTAL, prod, CI, all_same, partition
from tinygrad.codegen.uops import UOp, NOp, UOps, UPat, PatternMatcher, END_FOR_UOP, type_verify, print_uops
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
    if (s.dtype is not None and s.dtype.count != 1) or (is_image and s.src[1].dtype != dtypes.int.vec(3)): continue
    idx = s.src[1] if not is_image else s.src[1].src[2]  # only id4 for image
    if idx.arg is BinaryOps.ADD and idx.src[1].op is UOps.CONST: root_src, arg = idx.src[0], idx.src[1].arg
    elif idx.op is UOps.CONST: root_src, arg = "CONST", idx.arg
    else: root_src, arg = idx, 0
    # add idx and idy for image
    if is_image: root_src = (s.src[1].src[0:2], root_src)
    # add gates for gated
    if len(s.src) >= 4: root_src = (s.src[3], root_src)
    assert arg not in offsets_rootsrc[root_src]
    offsets_rootsrc[root_src][arg] = i

  # then rewrite everything we can
  used = set()
  for rootsrc, offsets in offsets_rootsrc.items():
    for o in offsets:
      for fold_length in [4] if is_image else ([8,4,2] if buf.dtype == PtrDType(dtypes.half) and getenv("ALLOW_HALF8") else [4,2]):
        if all((rootsrc,o+i) not in used and o+i in offsets for i in range(fold_length)):
          load_1 = new_srcs[offsets[o]]
          new_src = list(load_1.src)
          if not is_image and not new_src[1].divides(fold_length): continue
          # for images, we rewrite the index
          if is_image: new_src[1] = UOp(UOps.VECTORIZE, dtypes.int.vec(2), (new_src[1].src[0], new_src[1].src[1]))
          # vectorize the store/loadconst
          if not is_load or len(new_src) >= 4:
            new_src[2] = UOp(UOps.VECTORIZE, new_src[2].dtype.vec(fold_length), tuple(new_srcs[offsets[o+i]].src[2] for i in range(fold_length)))
          # generate the folded new_srcs
          if is_load:
            new_load = UOp(UOps.LOAD, load_1.dtype.vec(fold_length), tuple(new_src))
            for i in range(fold_length): new_srcs[offsets[o+i]] = UOp(UOps.GEP, load_1.dtype, (new_load,), i)
          else:
            for i in range(fold_length): new_srcs[offsets[o+i]] = UOp(UOps.STORE, None, tuple(new_src)) if i == 0 else None
          for i in range(fold_length): used.add((rootsrc,o+i))

  # dedup expand for LOAD
  if is_load and len(old_new_srcs) != len(ex.src): new_srcs = [new_srcs[old_new_srcs.index(s)] for s in ex.src]
  # remove Nones for STORE
  return UOp(ex.op, ex.dtype, tuple(x for x in new_srcs if x is not None), ex.arg) if len(used) else None

def vectorize_reduce(vec:UOp):
  if all_same(vec.src): return None  # don't REDUCE the same thing multiple times
  if not all_same([(x.src[1:], x.arg) for x in vec.src]): return None
  return UOp(UOps.REDUCE, vec.dtype, (UOp(UOps.VECTORIZE, vec.dtype, tuple(x.src[0] for x in vec.src)),) + vec.src[0].src[1:], vec.src[0].arg)

def vectorize_alu(vec:UOp):
  if not all_same([x.arg for x in vec.src]): return None
  return UOp(vec.src[0].op, vec.dtype, tuple(UOp(UOps.VECTORIZE, cast(DType, vec.src[0].src[i].dtype).vec(cast(DType, vec.dtype).count),
                                             tuple(x.src[i] for x in vec.src)) for i in range(len(vec.src[0].src))), vec.src[0].arg)

float4_folding = PatternMatcher([
  (UPat(UOps.EXPAND, src=UPat(UOps.LOAD, src=(UPat(name="buf"), UPat()), allow_any_len=True), name="ex"), fold_expanded),
  (UPat({UOps.BARRIER, UOps.SINK}, src=UPat(UOps.STORE, src=(UPat(name="buf"), UPat(), UPat()), allow_any_len=True), name="ex"), fold_expanded),
  (UPat(UOps.VECTORIZE, src=UPat(UOps.REDUCE), name="vec"), vectorize_reduce),
  (UPat(UOps.VECTORIZE, src=UPat({UOps.ALU, UOps.CAST, UOps.BITCAST}), name="vec"), vectorize_alu),
])

# ***** mod *****

def _get_add_chain(x:UOp):
  if x.op is UOps.ALU and x.arg is BinaryOps.ADD:
    for s in x.src: yield from _get_add_chain(s)
  else: yield x

def mod_folding(x:UOp, c:int) -> Optional[UOp]:
  # simplify x in x % c
  # None means no change
  remainder, something_changed = [], False
  for u in _get_add_chain(x):
    if (factor:=u.const_factor())%c != factor:
      remainder.append(u.divides(factor)*(factor%c))
      something_changed = True
    else: remainder.append(u)
  if not something_changed: return None
  return functools.reduce(operator.add, remainder) if remainder else x.const(0)

def div_folding(x:UOp, c:int) -> Optional[UOp]:
  # simplify x // c, None means no change
  # simple cancel div case
  if 0 <= x.vmin.arg and x.vmax.arg < c: return x.const(0)

  quotient, remainder, rem_const, something_changed, gcd, divisor = [], [], 0, False, c, 1
  for u in _get_add_chain(x):
    if u.op is UOps.CONST:
      # add all const together first
      if rem_const != 0: something_changed = True
      rem_const += u.arg
    elif (factor:=u.const_factor())%c == 0:
      if factor: quotient.append(u.divides(c))
      something_changed = True
    else:
      # divisor is the smallest common divisor of all MULs
      if u.op is UOps.ALU and u.arg is BinaryOps.MUL and factor > 1 and c % factor == 0 and (divisor == 1 or divisor > factor): divisor = factor
      remainder.append(u)
      gcd = math.gcd(gcd, factor)

  # handle the const
  if rem_const%c != rem_const:
    something_changed = True
    quotient.append(x.const(rem_const//c))
    rem_const = rem_const%c
  if rem_const != 0: remainder.append(x.const(rem_const))

  # x // c -> quotient + (remainder // div) // (c // div)
  div = gcd if gcd > 1 else divisor

  if not something_changed: return newx//(c//div) if 1 < div < c and (newx:=div_folding(x, div)) is not None else None
  rem:Optional[UOp] = functools.reduce(operator.add, remainder) if remainder else None
  quo:Optional[UOp] = functools.reduce(operator.add, quotient) if quotient else None
  if quo is None: return x.const(0) if rem is None else cast(UOp, div_folding(rem, div))//(c//div)
  return quo if rem is None else cast(UOp, div_folding(rem, div))//(c//div)+quo

# ***** transcendental *****

def transcendental_folding(ops):
  return PatternMatcher([(UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat(name="d"),), arg=k), cast(Callable, v))
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

def reduce_before_expand(reduce, expand, x):
  # if the expand is being reduced, you can't push it through
  # NOTE: could do a partial push here in some cases
  expands = flatten([x.arg for x in reduce.src[1:] if x.op is UOps.EXPAND])
  if any(x in expands for x in expand.arg): return None
  red = UOp(UOps.REDUCE, x.dtype, (x,)+reduce.src[1:], reduce.arg)
  return UOp(expand.op, expand.dtype, tuple(UOp(UOps.GEP, reduce.dtype, (red,), i) for i in range(x.dtype.count)), expand.arg)

def loop_collapse(loop_start, loop_end, compval, idx, mval, multconst, rng, reduce, idx2=None, idx3=None, extra=None):
  if getenv("DISABLE_LOOP_COLLAPSE") or rng not in reduce.src: return None  # must be the right REDUCE
  if mval.arg >= 0 or loop_start.arg != 0:
    # TODO: support and test this with other mvals and loop_starts
    if DEBUG >= 1: print(f"WARNING, NOT FOLDING: mval:{mval.arg} loop_start:{loop_start.arg}")
    return None
  if idx2 is not None: idx = idx + idx2
  if idx3 is not None: idx = idx + idx3
  comprange = UOp.min(loop_end, UOp.max((idx-compval-mval)//mval + (loop_end-loop_start), loop_start))
  new_reduce_op = comprange.cast(multconst.dtype) * multconst
  ret = UOp(UOps.REDUCE, reduce.dtype, (new_reduce_op,) + tuple(x for x in reduce.src[1:] if x is not rng), reduce.arg)
  if extra is not None: ret = ret + UOp(UOps.REDUCE, reduce.dtype, (extra,) + reduce.src[1:], reduce.arg)
  return ret

def index_collapse(idx,rng,buf,add,mul,ld,reduce):
  if rng not in reduce.src: return None
  return UOp(reduce.op, reduce.dtype, (UOp(ld.op, ld.dtype, (buf, add+mul*idx)),)+
             tuple(x for x in reduce.src[1:] if x is not rng), reduce.arg)

# this is symbolic 2.0
constant_folder = PatternMatcher([
  # VECTORIZE/GEP
  (NOp(UOps.GEP, src=(NOp(UOps.VECTORIZE, name="cast"),), name="gep"), lambda gep, cast: cast.src[gep.arg]),
  *[(NOp(UOps.VECTORIZE, dtypes.float.vec(i), tuple(NOp(UOps.GEP, dtypes.float,
                         src=(NOp.var('x', dtype=dtypes.float.vec(i)),), arg=j) for j in range(i))), lambda x: x) for i in [2, 4, 8, 16]],
  *[(NOp(UOps.VECTORIZE, dtypes.half.vec(i), tuple(NOp(UOps.GEP, dtypes.half,
                         src=(NOp.var('x', dtype=dtypes.half.vec(i)),), arg=j) for j in range(i))), lambda x: x) for i in [2, 4, 8, 16]],
  # tensor core with a 0 input is acc
  *[(NOp(UOps.WMMA, src=(NOp(UOps.VECTORIZE, src=tuple(NOp.const(None, 0.0) for _ in range(i))), NOp.var(), NOp.var('acc'))),
     lambda acc: acc) for i in [2, 4, 8]],
  *[(NOp(UOps.WMMA, src=(NOp.var(), NOp(UOps.VECTORIZE, src=tuple(NOp.const(None, 0.0) for _ in range(i))), NOp.var('acc'))),
     lambda acc: acc) for i in [2, 4, 8]],
  # tensor core cleanups
  *[(NOp(UOps.REDUCE, src=(NOp(UOps.EXPAND, src=tuple(NOp(UOps.GEP, dtypes.float, src=(NOp.var('x'),), arg=i) for i in range(j)), name="expand"),)
    ,name="reduce", allow_any_len=True), reduce_before_expand) for j in [2,4,8]],
  (NOp.var("add") + NOp(UOps.WMMA, name="wmma"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
  # threefry
  (NOp(UOps.ALU, dtype=dtypes.uint64, src=(NOp.var("x"), NOp.var("seed")), arg=BinaryOps.THREEFRY), threefry2x32),
  # extra arange loop folding because we don't fold adds. TODO: fold adds
  (NOp(UOps.REDUCE, src=((NOp.var("idx") + NOp.cvar("mval") * NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng") +
                          NOp.var("idx2") + NOp.var("idx3"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  (NOp(UOps.REDUCE, src=((NOp.var("idx") + NOp.cvar("mval") * NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng") +
                          NOp.var("idx2"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  # arange loop folding (reduce)
  (NOp(UOps.REDUCE, src=((NOp.var("idx") + NOp.cvar("mval") * NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  (NOp(UOps.REDUCE, src=((NOp.var("idx") - NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)),), arg=BinaryOps.ADD, name="reduce", allow_any_len=True),
   lambda **kwargs: loop_collapse(mval=UOp.const(dtypes.int, -1), **kwargs)),
  # arange loop folding (unrolled)
  (NOp(UOps.REDUCE, src=((NOp.var("idx") + NOp.cvar("mval") * NOp(UOps.RANGE, src=(NOp.var("loop_start"), NOp.var("loop_end")), name="rng"))
   .lt(NOp.cvar("compval")).where(NOp.cvar("multconst"), NOp.const(None, 0)) + NOp.var("extra"),),
   arg=BinaryOps.ADD, name="reduce", allow_any_len=True), loop_collapse),
  # indexing (with a multiply offset)!
  (NOp(UOps.REDUCE, src=(NOp.var('idx').eq(NOp(UOps.RANGE, name="rng")).cast()*
    NOp(UOps.LOAD, src=(NOp.var("buf"), NOp.var('add')+NOp.var('mul')*NOp(UOps.RANGE, name="rng")), name="ld"),),
    arg=BinaryOps.ADD, name="reduce", allow_any_len=True), index_collapse),
  (NOp(UOps.REDUCE, src=(NOp.var('idx').ne(NOp(UOps.RANGE, name="rng")).__neg__().cast()*
    NOp(UOps.LOAD, src=(NOp.var("buf"), NOp(UOps.RANGE, name="rng")), name="ld"),),
    arg=BinaryOps.ADD, name="reduce", allow_any_len=True),
    lambda **kwargs: index_collapse(add=UOp.const(dtypes.int, 0), mul=UOp.const(dtypes.int, 1), **kwargs)),
  (NOp(UOps.REDUCE, src=(NOp.var('idx').eq(NOp(UOps.RANGE, name="rng")).where(
    NOp(UOps.LOAD, src=(NOp.var("buf"), NOp.var('add')+NOp.var('mul')*NOp(UOps.RANGE, name="rng")), name="ld"), NOp.const(None, 0.0)),),
    arg=BinaryOps.ADD, name="reduce", allow_any_len=True), index_collapse),
  # other arange folders
  (NOp.cvar("c1") - (NOp.var("x") + NOp.cvar("c2")), lambda c1, c2, x: (c1-c2)-x),  # c1 - (x + c2) -> (c1-c2) - x
  (-(NOp.var("x") * NOp.cvar("c1")), lambda x, c1: x*-c1),
  # max folding
  (NOp.max(NOp.var('x'), NOp.var('y')), lambda x,y: x if x.vmin.arg >= y.vmax.arg else y if x.vmax.arg <= y.vmin.arg else None),
  # const rules
  (NOp(UOps.GEP, src=(NOp.cvar("c"),), name="root"), lambda root, c: root.const(c.arg)),
  (UPat(UOps.CAST, name="root", src=UPat(UOps.CONST, name="c")), lambda root, c: root.const(c.arg)),
  # a REDUCE without ranges is a NOOP
  (NOp(UOps.REDUCE, src=(NOp.var('x'),)), lambda x: x),
  # GEP on a const is the const
  (NOp(UOps.GEP, src=(NOp.cvar("x"),), name="root"), lambda root,x: root.const(x.arg)),
  # a conditional with the same results either way is a noop, also fold const conditionals
  (NOp.var().where(NOp.var("val"), NOp.var("val")), lambda val: val),
  (NOp.cvar('gate').where(NOp.var('c0'), NOp.var('c1')), lambda gate, c0, c1: c0 if gate.arg else c1),
  # ** constant folding **
  (UPat(UOps.ALU, name="root", src=UPat(UOps.CONST)), lambda root: root.const(exec_alu(root.arg, root.dtype, [x.arg for x in root.src]))),
  # ** self folding **
  (-(-NOp.var('x')), lambda x: x),    # -(-x) -> x
  (NOp.var('x') + 0, lambda x: x),    # x+0 -> x
  (NOp.var('x') * 1, lambda x: x),    # x*1 -> x
  (NOp.var('x') * -1, lambda x: -x),  # x*-1 -> -x
  (NOp.var('x') // NOp.var('x'), lambda x: x.const(1)), # x//x -> 1
  (NOp.var('x') // 1, lambda x: x),   # x//1 -> x
  (NOp.var('x') // -1, lambda x: -x), # x//-1 -> -x
  (NOp.var('x') / NOp.var('x'), lambda x: x.const(1)), # x/x -> 1
  (NOp.var('x') / NOp.cvar('c'), lambda x,c: x*exec_alu(UnaryOps.RECIP, c.dtype, [c.arg])),    # x/c -> x*(1/c)
  # ** zero folding **
  # x*0 -> 0 or 0*x -> 0
  # if x is nan or inf it should render the nan value.
  # NOTE: this can be wrong for loaded NaN
  (NOp.var('x') * 0, lambda x: x.const(float('nan') if isinstance(x.arg, float) and (math.isnan(x.arg) or math.isinf(x.arg)) else 0)),
  # x-x -> 0
  (NOp.var('x') - NOp.var('x'), lambda x: x.const(0)),
  (UPat(UOps.ALU, name='x'), lambda x: x.const(x.vmin.arg) if x.vmin.arg == x.vmax.arg else None),
  # ** load/store folding **
  (NOp.store(NOp.var("buf"), NOp.var("idx"), NOp.load(NOp.var("buf"), NOp.var("idx"))), lambda buf,idx:UOp(UOps.NOOP)),
  # ** two stage add/mul folding **
  ((NOp.var('x') + NOp.cvar('c1')) + NOp.cvar('c2'), lambda x,c1,c2: x+x.const(exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, c2.arg]))),
  ((NOp.var("x") * NOp.cvar("c1")) * NOp.cvar("c2"), lambda x,c1,c2: x*x.const(exec_alu(BinaryOps.MUL, x.dtype, [c1.arg, c2.arg]))),
  # *** rules from symbolic ***
  # ** lt **
  # c0*x<c1 for positive int c0,c1
  ((NOp.cvar('c0')*NOp.var('x')).lt(NOp.cvar('c1')),
   lambda x,c0,c1: x.lt(math.ceil(c1.arg/c0.arg)) if dtypes.is_int(x.dtype) and c0.arg > 0 and c1.arg > 0 else None),
  # mul add lt
  (((NOp.cvar('c0')*NOp.var('x'))+NOp.var('x2')).lt(NOp.cvar('c1')),
   lambda x,x2,c0,c1: x.lt(c1.arg//c0.arg) if c1.arg % c0.arg == 0 and c0.arg > x2.vmax.arg and x2.vmin.arg >= 0 else None),
  # generic lt folding (use div)
  (NOp.var('x').lt(NOp.cvar('c')), lambda x,c: newx.src[0].lt(newx.src[1]) if 0 < c.arg and dtypes.is_int(x.dtype) and \
   not dtypes.is_unsigned(x.dtype) and (newx:=div_folding(x,c.arg)) is not None and newx.op is UOps.ALU and newx.arg is BinaryOps.IDIV else None),
  # ** div **
  # # div folding
  (NOp.var('x') // NOp.cvar('c'), lambda x,c:
   newx if 0 < c.arg and not dtypes.is_unsigned(x.dtype) and (newx:=div_folding(x,c.arg)) is not None else None),
  # mul add div
  (((NOp.cvar('c0')*NOp.var('x'))+NOp.var('x2')) // NOp.cvar('c1'), lambda x,x2,c0,c1:\
   x*(c0.arg//g)//(c1.arg//g) if c0.arg > 0 and c1.arg > 0 and (g:=math.gcd(c0.arg,c1.arg)) > 1 and g > x2.vmax.arg and x2.vmin.arg >= 0 else None),
  # ** mod **
  # apply mod to mod input
  (NOp.var('x') % NOp.cvar('c'), lambda x,c: newx%c if 0 < c.arg and (newx:=mod_folding(x,c.arg)) is not None else None),
  # remove mod
  (NOp.var('x') % NOp.cvar('c'), lambda x,c:\
   x-(x.vmin.arg//c.arg)*c.arg if 0 < c.arg and 0 <= x.vmin.arg and x.vmin.arg//c.arg == x.vmax.arg//c.arg else None),
  # mul mod
  ((NOp.cvar('c0')*NOp.var('x')) % NOp.cvar('c1'), lambda x,c0,c1: (x%(c1.arg//c0.arg))*c0 if c1.arg%c0.arg == 0 else None),
  # mod mod
  ((NOp.var('x') % NOp.cvar('c0')) % NOp.cvar('c1'), lambda x,c0,c1: x % c1 if c0.arg % c1.arg == 0 else None),
  # (x%c)+(x//c)*c = x
  (NOp.var('x')%NOp.cvar('c')+(NOp.var('x')//NOp.cvar('c'))*NOp.cvar('c'), lambda x,c: x),
  # ** combine terms **
  # -(x+y) -> -x + -y
  (-(NOp.var("x") + NOp.var("y")), lambda x,y: (-x)+(-y)),
  # (x+c0)*c1 -> x*c1+c0*c1. only for signed int, float have inf*0=nan issue
  ((NOp.var("x") + NOp.cvar("c0")) * NOp.cvar("c1"), lambda x,c0,c1:
   x*c1+c0.arg*c1.arg if dtypes.is_int(x.dtype) and not dtypes.is_unsigned(x.dtype) else None),
  # (x*c0)+(x*c1) -> x*(c0+c1)
  (NOp.var("x") * NOp.cvar("c0") + NOp.var("x") * NOp.cvar("c1"), lambda x,c0,c1: x*exec_alu(BinaryOps.ADD, x.dtype, [c0.arg, c1.arg])),
  # (x*c0)+(y*c0) -> (x+y)*c0
  #((NOp.var("x") * NOp.cvar("c0")) + (NOp.var("y") * NOp.cvar("c0")), lambda x,y,c0: c0*(x+y)),
  # (x*x2)/x2 -> x
  ((NOp.var("x") * NOp.var("x2")) / NOp.var("x2"), lambda x,x2: x),
  # (x//c0)//c1 -> x//(c0*c1)
  ((NOp.var("x") // NOp.cvar("c0")) // NOp.cvar("c1"), lambda x,c0,c1: x//x.const(exec_alu(BinaryOps.MUL, x.dtype, [c0.arg, c1.arg]))),
  # (x/x1)/x2 -> x/(x1*x2)
  ((NOp.var("x") / NOp.var("x2")) / NOp.var("x3"), lambda x,x2,x3: x/(x2*x3)),
  # c0 + x < c1 -> x < c1 - c0
  ((NOp.cvar("c0") + NOp.var("x")).lt(NOp.cvar("c1")), lambda x,c0,c1: UOp.lt(x, x.const(exec_alu(BinaryOps.ADD, x.dtype, [c1.arg, -c0.arg])))),
  # (x+x*c0)-> x*(c0+1)
  (NOp.var("x") + NOp.var("x") * NOp.cvar("c0"), lambda x,c0: x*(c0.arg+1)),
  # x!=0 -> (bool)x
  (NOp.var("x").ne(0), lambda x: x.cast(dtypes.bool)),
  # bool != 1 -> not bool
  (NOp.var("x", dtype=dtypes.bool).ne(1), lambda x: -x),
  # TODO: can do the invert of this (flip alt/load) when we fix double ops
  (NOp.store(NOp.var("buf"), NOp.var("idx"), NOp.var("gate").where(NOp.var("alt"), NOp.load(NOp.var("buf"), NOp.var("idx")))),
   lambda buf, idx, gate, alt: UOp.store(buf, idx, alt, gate)),
  # VECTORIZE-PHI-GEP -> PHI-VECTORIZE
  (NOp(UOps.VECTORIZE, src=tuple(NOp(UOps.PHI, src=(NOp(UOps.GEP, src=(NOp.var("val"),), arg=i), NOp.var(f"v{i}"))) for i in range(4)), name="root"),
   lambda root, val, v0, v1, v2, v3: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1, v2, v3))))),
  (NOp(UOps.VECTORIZE, src=tuple(NOp(UOps.PHI, src=(NOp(UOps.GEP, src=(NOp.var("val"),), arg=i), NOp.var(f"v{i}"))) for i in range(2)), name="root"),
   lambda root, val, v0, v1: UOp(UOps.PHI, root.dtype, (val, UOp(UOps.VECTORIZE, val.dtype, (v0, v1))))),
  # cast NOOP (NOTE: it's str to deal with PtrDType)
  (NOp(UOps.CAST, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  (NOp(UOps.VECTORIZE, name="root"), lambda root: root.src[0] if str(root.dtype) == str(root.src[0].dtype) else None),
  # fold gated LOAD/STORE
  (NOp.load(NOp.var("buf"), NOp.var("idx"), NOp.var("var"), NOp.const(dtypes.bool, True)), lambda buf,idx,var: UOp.load(buf, idx, dtype=var.dtype)),
  (NOp.load(NOp.var("buf"), NOp.var("idx"), NOp.var("var"), NOp.const(dtypes.bool, True), NOp.var("barrier")),
   lambda buf,idx,var,barrier: UOp.load(buf, idx, barrier, dtype=var.dtype)),
  (NOp.load(NOp.var(), NOp.var(), NOp.var("var"), NOp.const(dtypes.bool, False)), lambda var: var),
  (NOp.load(NOp.var(), NOp.var(), NOp.var("var"), NOp.const(dtypes.bool, False), NOp.var()), lambda var: var),
  (NOp.store(NOp.var("buf"), NOp.var("idx"), NOp.var("val"), NOp.const(dtypes.bool, True)), UOp.store),
  (NOp.store(NOp.var(), NOp.var(), NOp.var(), NOp.const(dtypes.bool, False)), lambda: UOp(UOps.NOOP)),
  # remove NOOPs from SINK
  (NOp(UOps.SINK, name="root"),
    lambda root: UOp(UOps.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not UOps.NOOP)) != len(root.src) else None),
  # ** move add consts to end (NOTE: this is still happening before constant folding) **
  (UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(UOps.CONST, name='c1'), UPat(name='x'))), lambda c1,x: x+c1 if x.op is not UOps.CONST else None),
  (UPat(UOps.ALU, BinaryOps.ADD, src=[UPat(UOps.ALU, BinaryOps.ADD, src=(UPat(name='x'), UPat(UOps.CONST, name='c1'))), UPat(name='y')]),
    lambda x,c1,y: (x+y)+c1),
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

def do_expand(root:UOp):
  expands = [x for x in root.src if x.op is UOps.EXPAND]
  if len(expands) == 0: return None
  expand_args = tuple(sorted(dedup(flatten([x.arg for x in expands]))))
  if root.op is UOps.WMMA:
    # both the reduce and upcast args are not expanded here
    dont_expand_args = tuple(x for x in expand_args if x[0] in root.arg[-1] or x[0] in [y[0] for y in flatten(root.arg[-2])])
    expand_args = tuple(x for x in expand_args if x not in dont_expand_args)
  else:
    dont_expand_args = ()
  new_srcs: List[UOp] = []
  lrpks = _choices_from_args(dont_expand_args)
  for rpk in _choices_from_args(expand_args):
    new_src: List[UOp] = []
    for src in root.src:
      if src.op is UOps.EXPAND:
        lnew_src = tuple(src.src[_expand_arg_to_idx(src.arg, {**rpk, **lrpk})] for lrpk in lrpks)
        # TODO: is this right for UOps.WMMA? when there's more than one, all lnew_src should be the same
        new_src.append(lnew_src[0] if len(lnew_src) == 1 or root.op is UOps.WMMA else UOp(UOps.EXPAND, root.dtype, lnew_src, dont_expand_args))
      else:
        new_src.append(src)
    new_srcs.append(UOp(root.op, root.dtype, tuple(new_src), root.arg))
  if root.op is UOps.EXPAND:
    # merge two expands
    expand_args, old_args = tuple(sorted(root.arg+expand_args)), expand_args
    assert len(expand_args) == (len(old_args) + len(root.arg))
    new_srcs = [new_srcs[_expand_arg_to_idx(old_args, rpk)].src[_expand_arg_to_idx(root.arg, rpk)] for rpk in _choices_from_args(expand_args)]
  if root.op is UOps.IF:
    # merge ifs into an or
    conditions = functools.reduce(lambda x,y: x|y, dedup(x.src[0] for x in new_srcs if x.src[0].op is not UOps.CONST))
    barriers = tuple(set(x.src[1] for x in new_srcs))
    new_srcs = [UOp(UOps.IF, src=(conditions,)+barriers) for _ in new_srcs]
  assert prod([x[1] for x in expand_args]) == len(new_srcs)
  return UOp(UOps.EXPAND, root.dtype, tuple(new_srcs), expand_args)

acc_number = 0
def do_reduce(root:UOp):
  global acc_number
  reduce_parented, reduce_unparented = partition(root.src[1:], lambda x: x in root.src[0].parents)
  ret = root.src[0]
  if len(reduce_parented):
    assert root.dtype is not None
    const = UOp.const(root.dtype, 0 if root.arg is BinaryOps.ADD else dtypes.min(root.dtype.scalar()))
    acc = UOp(UOps.DEFINE_ACC, root.dtype, (const,) + tuple(reduce_parented), (acc_number,))
    acc_number += 1
    ret = UOp(UOps.PHI, root.dtype, (acc, acc.alu(root.arg, ret)))
  # for MAX, we can just ignore the unparented
  if root.arg is BinaryOps.ADD:
    for r in reduce_unparented: ret = ret * (r.src[1]-r.src[0]).cast(ret.dtype)
  return ret

def do_contract(con:UOp):
  ex = con.src[0]
  assert con.dtype is not None
  # CONTRACT without EXPAND repeats the element VECTORIZED
  if ex.op is not UOps.EXPAND: return UOp(UOps.VECTORIZE, con.dtype, con.src*con.dtype.count)
  # CONTRACT may remove several axes from EXPAND
  assert con.dtype.count == prod([x[1] for x in con.arg]), "dtype is wrong"
  srcs = []
  for rpk in _choices_from_args(new_ex_args:=tuple(x for x in ex.arg if x not in con.arg)):
    lsrcs = [ex.src[_expand_arg_to_idx(ex.arg, {**rpk, **lrpk})] for lrpk in _choices_from_args(con.arg)]
    srcs.append(UOp(UOps.VECTORIZE, con.dtype, tuple(lsrcs)))
  return srcs[0] if len(srcs) == 1 else UOp(UOps.EXPAND, con.dtype, tuple(srcs), new_ex_args)

def no_vectorized_alu(alu):
  if alu.dtype.count == 1: return None
  alus = tuple(UOp(alu.op, alu.dtype.scalar(),
                   tuple(UOp(UOps.GEP, s.dtype.scalar(), (s,), i) for s in alu.src), alu.arg) for i in range(alu.dtype.count))
  return UOp(UOps.VECTORIZE, alu.dtype, alus)

def create_gate(root:UOp) -> Optional[UOp]:
  @functools.lru_cache(None)
  def _gate_srcs(u:UOp, gate:UOp) -> UOp:
    if u.op is UOps.LOAD and u.src[-1].op is UOps.BARRIER: return UOp(u.op, u.dtype, u.src[:-1]+(UOp(UOps.IF, None, (gate, u.src[-1])),), u.arg)
    return u if (replace_source:=tuple(_gate_srcs(x, gate) for x in u.src)) == u.src else UOp(u.op, u.dtype, replace_source, u.arg)
  return None if len(root.src) == 3 or (ret:=_gate_srcs(root, root.src[3])) is root else ret

expander = PatternMatcher([
  # create gate MUST BE BEFORE expander
  (NOp(UOps.STORE, name="root"), create_gate),
  # do expansion
  (UPat({UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.GEP, UOps.WMMA, UOps.LOAD, UOps.STORE,
         UOps.VECTORIZE, UOps.REDUCE, UOps.EXPAND, UOps.IF}, name="root"), do_expand),
  (NOp(UOps.CONTRACT, name="con"), do_contract),
  # remove EXPANDs from SINK
  (NOp(UOps.SINK, name="root"),
   lambda root: UOp(UOps.SINK, root.dtype, a, root.arg)
    if len(a:=tuple(flatten(x.src if x.op is UOps.EXPAND else (x,) for x in root.src))) != len(root.src) else None),
  # BARRIERs aren't actually expanded
  (NOp(UOps.BARRIER, src=(NOp(UOps.EXPAND, name="ex"),)), lambda ex: UOp(UOps.EXPAND, None, (UOp(UOps.BARRIER, None, ex.src),)*len(ex.src), ex.arg)),
  # empty EXPAND is NOOP
  (NOp(UOps.EXPAND, src=(NOp.var('x'),), arg=()), lambda x: x),
  # EXPAND GEP (needed for WMMA, generalize this) -> vectorized ALU
  (NOp(UOps.EXPAND, name="ex", src=tuple(NOp.var('x').gep(i)+NOp.var('y').gep(i) for i in range(8))),
    lambda ex,x,y: UOp(UOps.EXPAND, ex.dtype, tuple((x+y).gep(i) for i in range(8)), ex.arg)),
])

def delete_redundant_gates(root:UOp) -> Optional[UOp]:
  @functools.lru_cache(None)
  def find_gate(x:UOp) -> Optional[UOp]:
    if x.op is UOps.IF: return x
    return next((ret for s in x.src if (ret:=find_gate(s)) is not None), None)
  if len(root.src) == 3 or (gate:=find_gate(root)) is None or gate.src[0] is not root.src[3]: return None
  return UOp(UOps.STORE, root.dtype, root.src[:3], root.arg)

reducer = PatternMatcher([
  (NOp(UOps.REDUCE, name="root"), do_reduce),
  # no ALU on vectorized dtypes
  (UPat({UOps.ALU, UOps.CAST, UOps.BITCAST}, name="alu"), no_vectorized_alu),
  # delete_redundant_gates (after expand, is this still needed?)
  (NOp(UOps.STORE, name="root"), delete_redundant_gates),
])

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

def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:
  nodes: Dict[Tuple, UOp] = {}
  replace: Dict[UOp, UOp] = {}
  def __inner_rewrite(n:UOp) -> UOp:
    if n in replace: return replace[n]
    replace_source = (n.op, n.dtype, tuple(__inner_rewrite(y) for y in n.src), n.arg)
    if found := nodes.get(replace_source): replace[n] = found
    else: nodes[replace_source] = replace[n] = found = __inner_rewrite(new_x) if (new_x := pm.rewrite(x:=UOp(*replace_source))) else x
    return found
  return __inner_rewrite(sink)

class UOpGraph:
  def __init__(self, sink:Union[UOp, List[UOp]], opts:Optional[Renderer]=None):
    self.sink: UOp = sink if isinstance(sink, UOp) else UOp(UOps.SINK, None, tuple(sink))
    assert self.sink.op is UOps.SINK, f"sink isn't sink, it's {self.sink.op}"
    # used by linearizer
    self._uops: Optional[List[UOp]] = None
    self.opts = opts
    self.folder = constant_folder + transcendental_folding({} if TRANSCENDENTAL >= 2 or opts is None else opts.code_for_op.keys())

  def __reduce__(self): return self.__class__, (self.sink, self.opts)
  def __iter__(self) -> Iterator[UOp]: return iter(self.uops)
  def __getitem__(self, index) -> UOp: return self.uops[index]

  @property
  def uops(self) -> List[UOp]:
    if self._uops is None: self.linearize()
    return cast(List[UOp], self._uops)

  def graph(self):
    from tinygrad.engine.graph import graph_uops
    graph_uops(self.uops)

  def print(self): print_uops(self.uops)

  cnt = 0
  def linearize(self, extra_pm:Optional[PatternMatcher]=None, skip_check=False) -> UOpGraph:
    global acc_number
    acc_number = 0

    # NOTE: relinearizering should be okay
    #assert self._uops is None, "already linearized"

    # do graph rewrite
    sink = graph_rewrite(self.sink, self.folder)

    # rewrite pyint to int32
    sink = graph_rewrite(sink, PatternMatcher([(UPat({UOps.CONST, UOps.ALU, UOps.SPECIAL, UOps.RANGE}, dtype=dtypes.pyint, name="x"),
      lambda x: UOp(x.op, dtypes.int32, x.src, x.arg))]))

    # expand
    UOpGraph.cnt += 1
    if UOpGraph.cnt != getenv("DEBUG_EXPAND", 0):
      sink = graph_rewrite(sink, self.folder+expander+float4_folding if self.opts is not None and self.opts.supports_float4 else self.folder+expander)
      sink = graph_rewrite(sink, self.folder+expander+reducer)

    # for PTX only
    if extra_pm: sink = graph_rewrite(sink, self.folder+extra_pm)

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
    range_phi = {r:[p for p in scope_children[r] if p.op is UOps.PHI] for r in scope_children if r.op is UOps.RANGE}

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
    self._uops = []
    while queue:
      p,x = heapq.heappop(queue)
      if DEBUG >= 7: print(f"{p:5d}",x)
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
    if not skip_check:
      bad_ops = dedup([x.op for x in self._uops if x.op in {UOps.EXPAND, UOps.CONTRACT, UOps.REDUCE}])
      try:
        type_verify(self.uops)
        assert self._uops[-1].op is UOps.SINK, f"didn't end with SINK, ended with {self._uops[-1]}"
        assert len(bad_ops) == 0, f"bad UOps left in list: {bad_ops}"
        # TODO: this should be enabled, and the valid clause should be removed
        # NOTE: multiple identical stores to DEFINE_LOCAL is okay
        assert len(all_stores := [x.src[0:2]+x.src[3:] for x in self._uops if x.op is UOps.STORE and x.src[0].op is not UOps.DEFINE_LOCAL]) \
          == len(dedup(all_stores)), "repeated stores in uops"
      except AssertionError as e:
        self.print()
        if not CI: self.graph()
        raise e

    # strip the SINK
    self._uops = self._uops[:-1]
    return self
