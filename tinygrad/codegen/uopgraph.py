from __future__ import annotations
from typing import Optional, Tuple, Dict, List, cast, TYPE_CHECKING, Any, DefaultDict, Callable
import functools, itertools, operator
from collections import defaultdict
from tinygrad.dtype import dtypes, ImageDType, PtrDType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, UOp, UOps, UPat, PatternMatcher, symbolic_flat, symbolic_simple
from tinygrad.ops import graph_rewrite, is_irreducible, split_uop, identity_element, uop_given_valid, parse_valid, is_increasing
from tinygrad.helpers import DEBUG, getenv, flatten, dedup, TRANSCENDENTAL, AMX, prod, partition, all_same
from tinygrad.codegen.transcendental import xexp2, xlog2, xsin, TRANSCENDENTAL_SUPPORTED_DTYPES

if TYPE_CHECKING: from tinygrad.renderer import Renderer

# ***** float4/image store handling *****

def fold_expanded(ex, buf):
  if buf.dtype.base != dtypes.float and buf.dtype.base != dtypes.half and not isinstance(buf.dtype, ImageDType): return None
  new_srcs = dedup(list(ex.src))
  old_new_srcs = new_srcs[:]
  is_load, is_image = new_srcs[0].op is UOps.LOAD, isinstance(buf.dtype, ImageDType)

  # first, extract all the relevant offsets
  offsets_rootsrc: DefaultDict[Any, dict] = defaultdict(dict)
  for i,s in enumerate(new_srcs):
    idx = s.src[0].src[1]
    if s.dtype.count != 1 or (is_image and idx.dtype.count == 2): continue
    if idx.arg is BinaryOps.ADD and idx.src[1].op is UOps.CONST: root_src, arg = idx.src[0], idx.src[1].arg
    elif idx.op is UOps.CONST: root_src, arg = "CONST", idx.arg
    else: root_src, arg = idx, 0
    # add gates for gated
    if len(s.src[0].src) == 3: root_src = (s.src[0].src[2], root_src)
    assert arg not in offsets_rootsrc[root_src], f"{offsets_rootsrc[root_src][arg]} != {i} with {len(s.src)} sources"
    offsets_rootsrc[root_src][arg] = i

  # then rewrite everything we can
  lengths = [4] if is_image else ([8,4,2] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else ([16,8,4,2] if AMX else [4,2]))
  used = set()
  for rootsrc, offsets in offsets_rootsrc.items():
    for o in offsets:
      for fold_length in lengths:
        if all((rootsrc,o+i) not in used and o+i in offsets for i in range(fold_length)):
          load_1 = new_srcs[offsets[o]]
          new_src = list(load_1.src)
          oidx = new_src[0].src[1]
          if oidx.divides(fold_length) is None: continue
          if is_image:
            # for images, we rewrite the index. it must evenly divide 4 from the above check
            new_src[0] = buf.index(
              UOp(UOps.VECTORIZE, dtypes.int.vec(2), ((oidx // 4) % buf.dtype.shape[1], (oidx // (4*buf.dtype.shape[1])))),
              rootsrc[0] if isinstance(rootsrc, tuple) else None)
          else:
            # for non image, we upcast the index pointer
            new_src[0] = new_src[0].cast(new_src[0].dtype.base.vec(fold_length).ptr(new_src[0].dtype.local))
          # vectorize the store
          if not is_load:
            new_src[1] = UOp(UOps.VECTORIZE, new_src[1].dtype.vec(fold_length), tuple(new_srcs[offsets[o+i]].src[1] for i in range(fold_length)))
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
  if not isinstance(buf.dtype, ImageDType) or (oidx:=load.src[0].src[1]).dtype.count == 2: return None
  id4 = oidx % 4
  new_src = list(load.src)
  # TODO: copied logic from above
  new_src[0] = load.src[0].src[0].index(
    UOp(UOps.VECTORIZE, dtypes.int.vec(2), ((oidx // 4) % buf.dtype.shape[1], (oidx // (4*buf.dtype.shape[1])))),
    load.src[0].src[2] if len(load.src[0].src) == 3 else None)
  vec_load = UOp(UOps.LOAD, load.dtype.vec(4), tuple(new_src))
  return functools.reduce(lambda ret, i: id4.ne(i).where(ret, vec_load.gep(i)), range(4), load.const_like(float('nan')))

buf_idx_pat = UPat(UOps.INDEX, src=(UPat.var("buf"),), allow_any_len=True)
float4_folding = PatternMatcher([
  (UPat(UOps.VECTORIZE, src=UPat(UOps.LOAD, src=(buf_idx_pat,), allow_any_len=True), name="ex"), fold_expanded),
  (UPat((UOps.BARRIER, UOps.SINK), src=UPat(UOps.STORE, src=(buf_idx_pat,), allow_any_len=True), name="ex"), fold_expanded),
])

# ***** image load valid simplification *****

def simplify_valid_load(buf:UOp, start_idx:UOp, valid:UOp) -> Optional[UOp]:
  if (idx:=uop_given_valid(valid, start_idx)) is None: return buf.const_like(0)
  if not isinstance(buf.dtype, ImageDType): return None if idx is start_idx else buf.index(idx, valid)

  # wait for it to be image indexed before running simplification
  if start_idx.dtype.count != 2: return None

  # can drop valid if idx is out of bound when valid is False
  drop_stmt = []
  for stmt in split_uop(valid, BinaryOps.AND):
    X, is_upper_bound, c = parse_valid(stmt)

    # for X0 + X1 + ... >= 1, check if it's out of bound when Xi = 0 for all i
    if not is_upper_bound and c == 1 and all(is_irreducible(u) and u.vmin == 0 for u in split_uop(X, BinaryOps.ADD)):
      testidx = functools.reduce(lambda nowidx,u: nowidx.substitute({u:u.const_like(0)}), split_uop(X, BinaryOps.ADD), idx)
      testidx = testidx.simplify()
      if testidx.gep(0).vmax < 0 or testidx.gep(1).vmax < 0:
        drop_stmt.append(stmt)
        continue

    # if X <= c, check if it's out of bound when X = c+1
    # if X >= c, check if it's out of bound when X = c-1
    test_value = c + 1 if is_upper_bound else c - 1
    for i,b in zip(idx.src, (buf.dtype.shape[1], buf.dtype.shape[0])):
      if is_increasing(i):
        rw = i.substitute({X:X.const_like(test_value)}).simplify()
        if rw.vmin >= b or rw.vmax < 0:
          drop_stmt.append(stmt)
          break

  if not drop_stmt and idx is start_idx: return None
  new_valid = functools.reduce(operator.and_, ss) if (ss:=[s for s in split_uop(valid, BinaryOps.AND) if s not in drop_stmt]) else None
  return buf.index(idx, new_valid)

# ***** optional patterns *****

transcendental_patterns = [
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat.var("d"),), arg=UnaryOps.EXP2), xexp2),
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat.var("d"),), arg=UnaryOps.LOG2), xlog2),
  (UPat(UOps.ALU, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat.var("d"),), arg=UnaryOps.SIN), xsin),
]

@functools.lru_cache(None)
def get_transcendental_patterns(ops, force_transcendental=False):
  pat = [(p[0], cast(Callable, p[1])) for p in transcendental_patterns if p[0].arg not in ops or force_transcendental]
  return PatternMatcher(pat)

powers_of_two = {2**i:i for i in range(64)}
@functools.lru_cache(None)
def get_late_rewrite_patterns(ops):
  pat: List[Tuple[UPat, Callable]] = []
  # rewrite MOD to AND (which should always be supported, but not for generic in tests)
  if BinaryOps.AND in ops:
    pat += [(UPat(UOps.ALU, arg=BinaryOps.MOD, src=(UPat.var('base'), UPat.cvar("const"))),
            lambda base,const: base & (const.arg-1) if const.arg in powers_of_two else None)]
  # rewrite MUL/IDIV to SHL+SHR
  if BinaryOps.SHL in ops and BinaryOps.SHR in ops:
    pat += [
    (UPat(UOps.ALU, arg=BinaryOps.MUL, dtype=dtypes.ints, src=[UPat.cvar("const"), UPat.var("mul")]), lambda mul, const:
      UOp(UOps.ALU, mul.dtype, (mul, UOp.const(dtypes.int, powers_of_two[const.arg])), BinaryOps.SHL) if const.arg in powers_of_two else None),
    (UPat(UOps.ALU, arg=BinaryOps.IDIV, src=(UPat.var("div"), UPat.cvar("const"))), lambda div, const:
      UOp(UOps.ALU, div.dtype, (div, UOp.const(dtypes.int, powers_of_two[const.arg])), BinaryOps.SHR) if const.arg in powers_of_two else None)]
  if UnaryOps.NEG in ops:
    pat += [(UPat.var('x')*-1, lambda x: x.alu(UnaryOps.NEG))]
    if BinaryOps.SUB in ops: pat += [(UPat.var('x')+UPat.var('y').alu(UnaryOps.NEG), lambda x,y: x.alu(BinaryOps.SUB, y))]
  if TernaryOps.MULACC in ops:
    pat += [(UPat.var('a')*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(TernaryOps.MULACC, b, c))]
  return PatternMatcher(pat)

# ***** threefry *****

def threefry2x32(x: UOp, key: UOp):
  # split x into two uint32, since x in a uint64
  x0, x1 = (x & 0xffffffff).cast(dtypes.uint32), ((x // 2**32) & 0xffffffff).cast(dtypes.uint32)

  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  key0, key1 = (key & 0xffffffff).cast(dtypes.uint32), ((key // 2**32) & 0xffffffff).cast(dtypes.uint32)
  ks = [key1, key0 ^ key1 ^ 0x1BD11BDA, key0]
  xr = [x0 + ks[-1], x1 + ks[0]]
  for i in range(5):
    for r in rotations[i % 2]: xr[0], xr[1] = (x0 := xr[0] + xr[1]), x0 ^ ((xr[1] * 2**r) + (xr[1] // 2**(32 - r)))
    xr = [(xr[0] + ks[i % 3]), (xr[1] + ks[(i + 1) % 3] + i + 1)]

  return xr[1].cast(dtypes.uint64) * 2**32 | xr[0].cast(dtypes.uint64)

# ***** main rewriter *****

def loop_collapse(compval, multconst, rng:UOp, acc:UOp, idx2=None,idx3=None,extra=None,vec=None,ne=None,
                  add=UOp.const(dtypes.int, 0), mul:UOp=UOp.const(dtypes.int, 1)):
  if getenv("DISABLE_LOOP_COLLAPSE") or rng not in acc.src: return None  # must be the right REDUCE
  loop_start, loop_end = rng.src
  if loop_start.arg != 0:
    # TODO: support and test this with other mul and loop_starts
    if DEBUG >= 1: print(f"WARNING, NOT FOLDING: mul:{mul.arg} loop_start:{loop_start.arg}")
    return None
  if idx2 is not None: add = add + idx2
  if idx3 is not None: add = add + idx3
  if vec is not None:
    # add, mul, loop_start, loop_end
    def dvec(x:UOp):
      if x.op is UOps.CONST: return UOp.const(x.dtype.vec(vec.dtype.count), x.arg)
      return UOp(UOps.VECTORIZE, x.dtype.vec(vec.dtype.count), src=(x,)*vec.dtype.count)
    add, mul, loop_start, loop_end = dvec(add), dvec(mul), dvec(loop_start), dvec(loop_end)
  if mul.vmin > 0 and ne is not None:
    comprange = UOp.minimum(loop_end, UOp.maximum((add-compval)//mul + (loop_end-loop_start), loop_start))
  elif mul.vmax < 0 and ne is None:
    comprange = UOp.minimum(loop_end, UOp.maximum((add-compval-mul)//mul + (loop_end-loop_start), loop_start))
  else:
    return None
  new_reduce_op = comprange.cast(multconst.dtype) * multconst
  # TODO: what does it mean to have the same numbered DEFINE_ACC with different ranges?
  new_acc = acc.replace(src=acc.src[0:1]+tuple(x for x in acc.src[1:] if x is not rng))
  ret = new_acc.assign(new_acc+new_reduce_op)
  if extra is not None: ret = ret + acc.assign(acc+extra)
  return ret

def index_collapse(idx:UOp,rng:UOp,buf:UOp,ld:UOp,acc:UOp,add=UOp.const(dtypes.int, 0),mul=UOp.const(dtypes.int, 1)):
  if rng not in acc.src: return None
  new_load = UOp.load(buf.index(add+mul*idx, idx.ge(rng.src[0]) & idx.lt(rng.src[1])), dtype=ld.dtype)
  new_acc = acc.replace(src=acc.src[0:1]+tuple(x for x in acc.src[1:] if x is not rng))
  return new_acc.assign(new_acc+new_load)

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

acc_pat, rng_pat = UPat(UOps.DEFINE_ACC, name="acc"), UPat(UOps.RANGE, name="rng")
rng_aug = UPat.any(rng_pat, UPat.var("add")+rng_pat, UPat.var("mul")*rng_pat, UPat.var("add")+UPat.var("mul")*rng_pat)

index_load = UPat.var("buf").index(rng_aug).load(name="ld")

arange_augrng = UPat.any(rng_aug, rng_aug+UPat.var("idx2"), rng_aug+UPat.var("idx2")+UPat.var("idx3"), UPat(UOps.VECTORIZE, name="vec", src=rng_aug))
arange_m = arange_augrng.lt(UPat.cvar("compval")).ne(UPat(UOps.CONST, name="ne", arg=True)).where(UPat.cvar("multconst"), UPat.const(None, 0))

# this is symbolic 2.0
sym = symbolic_flat+PatternMatcher([
  # self ASSIGN is just self
  (UPat(UOps.ASSIGN, src=(UPat.var('x'), UPat.var('x'))), lambda x: x),
  # ASSIGN to global is just self
  (UPat(UOps.ASSIGN, src=(UPat(UOps.DEFINE_GLOBAL), UPat.var("x"))), lambda x: x),
  # VECTORIZE/CONST, VECTORIZE/GEP
  (UPat(UOps.VECTORIZE, src=UPat(UOps.CONST), name="vec"), lambda vec: UOp.const(vec.dtype, tuple(x.arg for x in vec.src))),
  (UPat(UOps.VECTORIZE, src=UPat(UOps.GEP, src=(UPat(name="x"),)), name="vec"), lambda vec,x: x.gep(tuple(y.arg[0] for y in vec.src))),
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
  (UPat(UOps.ALU, dtype=dtypes.uint64, src=(UPat.var("x"), UPat.var("key")), arg=BinaryOps.THREEFRY), threefry2x32),
  # arange loop folding
  (acc_pat.assign(UPat.any(arange_m, arange_m+UPat.var("extra"))+acc_pat), loop_collapse),
  # indexing, with cast or where
  (acc_pat.assign(UPat.var("idx").eq(UPat(UOps.RANGE, name="rng")).cast()*index_load+acc_pat), index_collapse),
  (acc_pat.assign(UPat.var("idx").eq(UPat(UOps.RANGE, name="rng")).where(index_load, UPat.const(None, 0.0))+acc_pat), index_collapse),
  # ** self folding **
  (UPat(UOps.DEFINE_ACC, src=(UPat.var("x"),)), lambda x: x),            # a DEFINE_ACC without ranges is a CONST
  (UPat(UOps.ASSIGN, src=(UPat.cvar(),UPat.var("x"))), lambda x: x),     # an ASSIGN to a const is a NOOP
  # x!=0 -> (bool)x
  (UPat.var("x").ne(0), lambda x: x.cast(dtypes.bool.vec(x.dtype.count))),
  # ** load/store folding **
  (UPat.store(UPat(UOps.INDEX, name="index"), UPat.load(UPat(UOps.INDEX, name="index"))), lambda index: UOp(UOps.NOOP)),
  (UPat.store(UPat(UOps.INDEX, name="index"), UPat.var("gate").where(UPat.var("alt"), UPat.load(UPat(UOps.INDEX, name="index")))),
   lambda index, gate, alt: UOp.store(index.src[0].index(index.src[1], gate), alt)),
  # fold gated LOAD/STORE
  (UPat().index(UPat(), UPat.const(dtypes.bool, True)).named("idx"), lambda idx: idx.replace(src=idx.src[0:2])), # remove True
  (UPat().index(UPat(), UPat.const(dtypes.bool, False)).named("idx"), lambda idx: idx.const_like(0)),      # False -> NULL pointer
  (UPat(UOps.LOAD, src=(UPat.const(None, 0),), allow_any_len=True, name="x"), lambda x: x.const_like(0)),  # NULL pointer load loads 0
  (UPat(UOps.STORE, src=(UPat.const(None, 0),), allow_any_len=True), lambda: UOp(UOps.NOOP)),  # NULL pointer store does nothing
  # remove NOOPs from SINK
  (UPat(UOps.SINK, name="root"),
    lambda root: UOp(UOps.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not UOps.NOOP)) != len(root.src) else None),
  # remove EXPANDs from SINK/BARRIER
  (UPat(UOps.BARRIER, src=(UPat((UOps.VECTORIZE, UOps.SINK), name='sink'),)), lambda sink: UOp(UOps.BARRIER, dtypes.void, sink.src)),
  (UPat(UOps.SINK, name="root"),
    lambda root: UOp(UOps.SINK, root.dtype, tuple(flatten(x.src if x.op in {UOps.SINK, UOps.EXPAND} else (x,) for x in root.src)), root.arg)
      if any(x.op in {UOps.SINK, UOps.EXPAND} for x in root.src) else None),
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
      if (root.op is UOps.IF) or (root.op is UOps.REDUCE and i != 0):
        # for the first arg of IF and the RANGE args of REDUCE, just pass them through ignoring EXPANDS
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

def do_reduce(ctx:List[int], root:UOp):
  reduce_parented, reduce_unparented = partition(root.src[1:], lambda x: x in root.src[0].sparents)
  ret = root.src[0]
  if len(reduce_parented):
    acc = UOp(UOps.DEFINE_ACC, root.dtype,
              (root.const_like(identity_element(root.arg, root.dtype.scalar())),) + tuple(reduce_parented), (ctx[0],))
    ctx[0] += 1
    ret = acc.assign(acc.alu(root.arg, ret))
  # for MAX, we can just ignore the unparented
  if root.arg is BinaryOps.ADD:
    for r in reduce_unparented: ret = ret * (r.src[1]-r.src[0]).cast(ret.dtype.scalar()).broadcast(ret.dtype.count)
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
  if alu.dtype.vcount == 1: return None
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(UOps.VECTORIZE, alu.dtype, alus)

def create_gate(root:UOp) -> Optional[UOp]:
  @functools.lru_cache(None)
  def _gate_srcs(u:UOp, gate:UOp) -> UOp:
    if u.op is UOps.BARRIER: return u
    if u.op is UOps.LOAD and u.src[-1].op is UOps.BARRIER:
      return UOp(u.op, u.dtype, u.src[:-1]+(UOp(UOps.IF, dtypes.void, (gate, u.src[-1])),), u.arg)
    return u if (replace_source:=tuple(_gate_srcs(x, gate) for x in u.src)) == u.src else UOp(u.op, u.dtype, replace_source, u.arg)
  idx = root.src[0]
  if idx.op is UOps.CAST: idx = idx.src[0]
  return None if idx.op is not UOps.INDEX or len(idx.src) == 2 or (ret:=_gate_srcs(root, idx.src[2])) is root else ret

expander = PatternMatcher([
  # double expand
  (UPat(UOps.EXPAND, name="outer", src=(UPat(UOps.EXPAND, name="inner"),)),
   lambda outer, inner: UOp(UOps.EXPAND, outer.dtype, (inner.src[0],), inner.arg+outer.arg)),
  # do expansion
  (UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.GEP, UOps.WMMA, UOps.LOAD, UOps.STORE, UOps.INDEX,
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
  idx = ls.src[0]
  assert isinstance(idx.dtype, PtrDType)
  if idx.dtype.v == 1: return None
  tv = [UOp(ls.op, ls.dtype.scalar(), tuple(j.gep(i) for j in ls.src)) for i in range(idx.dtype.v)]
  return UOp(UOps.VECTORIZE, ls.dtype, tuple(tv))

def no_vectorized_acc(acc:UOp):
  if acc.dtype.count == 1: return None
  alus = tuple(UOp(acc.op, acc.dtype.scalar(),
    tuple(s.gep(i) if j == 0 else s for j,s in enumerate(acc.src)), acc.arg+(i,)) for i in range(acc.dtype.count))
  return UOp(UOps.VECTORIZE, acc.dtype, alus)

just_reduce = PatternMatcher([
  # do reduce
  (UPat(UOps.REDUCE, name="root"), do_reduce),
])

devectorize = PatternMatcher([
  # no ALU on vectorized dtypes
  (UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.ASSIGN, UOps.INDEX), name="alu"), no_vectorized_alu),
  (UPat(UOps.WMMA, name="wmma"), no_vectorized_wmma),
  (UPat(UOps.DEFINE_ACC, name="acc"), no_vectorized_acc),
  (UPat((UOps.LOAD, UOps.STORE), name="ls"), no_vectorized_load_store),
])

def delete_redundant_gates(buf:UOp, idx:UOp, val:UOp, store_gate:UOp, cast:Optional[UOp]=None) -> Optional[UOp]:
  if store_gate not in [gate.src[0] for gate in val.sparents if gate.op is UOps.IF]: return None
  # remove the gate from the index
  return UOp.store(buf.index(idx).cast(cast.dtype) if cast is not None else buf.index(idx), val)

load_store_indexing = PatternMatcher([
  # late fixup of unfoldable image loads
  (UPat(UOps.LOAD, src=(UPat.var("buf"), UPat()), allow_any_len=True, name="load"), fix_unfoldable_image_load),
  # image load valid idx simplification
  (UPat(UOps.INDEX, src=(UPat.var("buf"), UPat.var("start_idx"), UPat.var("valid"))), simplify_valid_load),
  # delete_redundant_gates (after expand)
  (UPat(UOps.STORE, src=(UPat.any(stidx:=UPat.var("buf").index(UPat.var("idx"), UPat.var("store_gate")), stidx.cast().named("cast")),
                                  UPat.var("val"))), delete_redundant_gates),
])

def idx_load_store(x:UOp):
  idx = x.src[0].index(x.src[1], x.src[3] if len(x.src) > 3 else None)
  v = x.dtype.count if x.op is UOps.LOAD else x.src[2].dtype.count
  if v > 1 and not isinstance(x.src[0].dtype, ImageDType): idx = idx.cast(idx.dtype.base.vec(v).ptr(idx.dtype.local))
  post_mask = x.src[4:] if len(x.src) > 3 else (x.src[2:] if x.op is UOps.LOAD else x.src[3:])
  if x.op is UOps.LOAD: return UOp(x.op, x.dtype, (idx,)+post_mask, x.arg)
  return UOp(x.op, x.dtype, (idx,x.src[2])+post_mask, x.arg)

migrate_indexing = PatternMatcher([
  # use indexing for LOAD/STORE
  (UPat((UOps.LOAD, UOps.STORE), src=(UPat((UOps.DEFINE_GLOBAL, UOps.DEFINE_LOCAL)),), allow_any_len=True, name="x"), idx_load_store),
  # create gate MUST BE BEFORE expander
  (UPat(UOps.STORE, name="root"), create_gate),
])

def move_mask(x:UOp, buf:UOp, idx:UOp, mask:UOp, cast:Optional[UOp]=None) -> UOp:
  # this moves the mask from the indexing to the load/store op for rendering
  nidx = buf.index(idx).cast(cast.dtype) if cast is not None else buf.index(idx)
  return UOp.load(nidx, x.const_like(0), mask, *x.src[1:], dtype=x.dtype) if x.op is UOps.LOAD else UOp.store(nidx, x.src[1], mask, *x.src[2:])

pm_render = PatternMatcher([
  # for rendering, we use explicit VECTORIZE
  (UPat(UOps.CONST, name='c'),
   lambda c: UOp(UOps.VECTORIZE, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  (UPat(UOps.VCONST, name='c'), lambda c: UOp(UOps.VECTORIZE, c.dtype, tuple(UOp.const(c.dtype.scalar(), x) for x in c.arg))),
  (UPat(UOps.GEP, name='gep'), lambda gep: UOp(UOps.VECTORIZE, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  (UPat(UOps.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
  # move masks of loads/stores
  (UPat((UOps.LOAD, UOps.STORE), src=(UPat.any(masked_index:=UPat(UOps.INDEX, src=(UPat(name="buf"), UPat(name="idx"), UPat(name="mask"))),
                                               masked_index.cast(None).named("cast")),), allow_any_len=True, name="x"), move_mask),
])

# *** uop graph ***

def full_graph_rewrite(sink:UOp, opts:Optional[Renderer]=None) -> UOp:
  assert sink.op is UOps.SINK, f"sink isn't sink, it's {sink.op}"
  supported_ops = tuple(opts.code_for_op.keys()) if opts is not None else ()
  extra_matcher = opts.extra_matcher if opts is not None and opts.extra_matcher is not None else PatternMatcher([])

  # initial symbolic + migrate indexing (remove this) + transcendental
  sink = graph_rewrite(sink, sym+migrate_indexing+get_transcendental_patterns(supported_ops, TRANSCENDENTAL>=2))

  # expand
  sink = graph_rewrite(sink, sym+expander)

  # convert REDUCE to DEFINE_ACC + ASSIGN (contextual)
  sink = graph_rewrite(sink, sym+just_reduce, ctx=[0])

  # devectorize + load_store_indexing
  sink = graph_rewrite(sink, sym+(devectorize+float4_folding if opts is not None and opts.supports_float4 else devectorize)+load_store_indexing)

  # final rules for the renderer (without sym)
  sink = graph_rewrite(sink, symbolic_simple+get_late_rewrite_patterns(supported_ops)+pm_render+extra_matcher)
  return sink
