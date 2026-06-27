import functools, itertools
from dataclasses import dataclass
from tinygrad.dtype import dtypes, ImageDType, DType, AddrSpace, Invalid, PtrDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp, identity_element
from tinygrad.uop.symbolic import uop_given_valid, parse_valid, invalid_gate
from tinygrad.helpers import getenv, flatten, prod, OSX, ceildiv

# ***** image load valid simplification *****

@functools.cache
def _drop_valid_stmts(valid:UOp, idx:UOp, height:int, width:int) -> list[UOp]:
  # can drop valid if idx is out of bound when valid is False
  drop_stmt = []
  for i,stmt in enumerate(valid.split_uop(Ops.AND)):
    if (res:=parse_valid(stmt)) is None: continue
    X, is_upper_bound, c = res

    # for X0 + X1 + ... >= 1, check if it's out of bound when Xi = 0 for all i
    if not is_upper_bound and c == 1 and all(u.op in GroupOp.Irreducible and u.vmin == 0 for u in X.split_uop(Ops.ADD)):
      testidx = functools.reduce(lambda nowidx,u: nowidx.substitute({u:u.const_like(0)}), X.split_uop(Ops.ADD), idx)
      if testidx.gep(0).vmax < 0 or testidx.gep(1).vmax < 0:
        drop_stmt.append(stmt)
        continue

    # check if idx is out of bound when X is on the wrong side of the bound: X in [c+1, vmax] or [vmin, c-1]
    lo, hi = (c + 1, X.vmax) if is_upper_bound else (X.vmin, c - 1)
    if lo <= hi:
      fake = UOp.variable(f"fake{i}", lo, hi, X.dtype)
      for coord,b in zip(idx.src, (width, height)):
        rw = coord.substitute({X:fake}).simplify()
        if rw.vmin >= b or rw.vmax < 0:
          drop_stmt.append(stmt)
          break
  return drop_stmt

def simplify_valid_load(buf:UOp, start_idx:UOp, valid:UOp) -> UOp|None:
  idx = uop_given_valid(valid, start_idx)
  return None if idx is start_idx else buf.index(idx.valid(valid), ptr=True)

def simplify_valid_image_load(buf:UOp, idx_y:UOp, idx_x:UOp, valid:UOp) -> UOp|None:
  if not isinstance(buf.dtype, ImageDType): return None
  start_idx = UOp.vectorize(idx_x, idx_y)
  idx = uop_given_valid(valid, start_idx)
  drop_stmt = _drop_valid_stmts(valid, idx, buf.dtype.shape[0], buf.dtype.shape[1])

  if not drop_stmt and idx is start_idx: return None
  new_valid = UOp.uprod(*ss) if (ss:=[s for s in valid.split_uop(Ops.AND) if s not in drop_stmt]) else None
  idx_y, idx_x = idx.gep(1), idx.gep(0)
  return buf.index(idx_y.valid(new_valid), idx_x.valid(new_valid), ptr=True) if new_valid is not None else buf.index(idx_y, idx_x, ptr=True)

indexing_simplify = PatternMatcher([
  # image load valid idx simplification
  (UPat(Ops.INDEX, src=(UPat.var("buf"), invalid_gate)), lambda buf,x,i,cond: simplify_valid_load(buf, x, cond)),
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("valid").where(UPat.var("idx_y"), UPat(arg=Invalid)),
                                         UPat.var("valid").where(UPat.var("idx_x"), UPat(arg=Invalid)))), simplify_valid_image_load),
])

# ***** load/store grouping *****

# get list of (height, width) that do not require pitch padding
def image_valid_dims(base:DType, size:int, arch:str) -> list[tuple[int,int]]:
  if (ALIGN:=next((int(p.split('=')[1]) for p in arch.split(',') if p.startswith("IMAGE_PITCH_ALIGNMENT=")), 0)) == 0: return []
  MAXW, pxls = 16384, size // 4
  if base not in (dtypes.half, dtypes.float) or size > 4*MAXW*MAXW: return []
  # height=1 images just need to abide by alignment requirements in bytes, not pixels!
  if size % (ALIGN * 4) != 0: return [] if (base.itemsize * size) % (64 if OSX else ALIGN) != 0 or pxls > MAXW else [(1, pxls)]
  return [(pxls//ALIGN//k, ALIGN*k) for k in range(ceildiv(pxls//ALIGN, MAXW), min(pxls//ALIGN, MAXW//ALIGN)+1) if (pxls//ALIGN)%k == 0]

def expand_index(ctx, buf:UOp, vec:UOp):
  if getenv("UNSAFE_DISABLE_MASK", 0): vec = vec.get_idx()
  # generate the individual indexes
  return UOp(Ops.STACK, buf.dtype, tuple(buf.index(vec.gep(i), ptr=True) for i in range(vec.dtype.count)))

def load_stack(stack:UOp, ld:UOp):
  offset, ret = 0, []
  for x in stack.src:
    src = [x]
    for s in ld.src[1:]:
      src.append(s.gep(tuple(range(offset, offset+x.dtype.count))) if s.dtype.vcount > 1 else s)
    ret.append(ld.replace(dtype=x.dtype.base, src=tuple(src)))
    offset += x.dtype.count
  return UOp(Ops.STACK, stack.dtype.base.vec(len(stack.src)), tuple(ret))

def store_stack(stack:UOp, data:UOp):
  offset, ret = 0, []
  for x in stack.src:
    ret.append(x.store(data.gep(tuple(range(offset, offset+x.dtype.count)))))
    offset += x.dtype.count
  return UOp.group(*ret)

load_store_folding = PatternMatcher([
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, src=UPat(name="buf")), UPat.var("vec"))), expand_index),
  # put STACK of indexes after LOAD/STORE
  (UPat(Ops.LOAD, src=(UPat(Ops.STACK, src=UPat(Ops.INDEX), name="stack"),), name="ld", allow_any_len=True), load_stack),
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, src=UPat(Ops.INDEX), name="stack"), UPat(name="data"))), store_stack),
])

# *** uop expander ***

# TODO: there's a lot shared with gep_through_wmma here
def no_vectorized_wmma(wmma:UOp):
  out_sz = prod(x[1] for x in wmma.arg[6][-1])
  if wmma.dtype.count == out_sz: return None
  tsrcs = []
  for s,sz in zip(wmma.src, wmma.arg[6]):
    ssz = prod(x[1] for x in sz)
    tsrcs.append([s.gep(tuple(range(grp, grp+ssz))) for grp in range(0, s.dtype.count, ssz)])
  wmmas = [UOp(Ops.WMMA, wmma.dtype.scalar().vec(out_sz), tsrc, wmma.arg) for tsrc in zip(*tsrcs)]
  wmma_ex = flatten([[e.gep(i) for i in range(out_sz)] for e in wmmas])
  return UOp(Ops.STACK, wmma.dtype, tuple(wmma_ex))

def no_vectorized_alu(alu:UOp):
  if alu.dtype.vcount == 1: return None
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(Ops.STACK, alu.dtype, alus)

def no_vectorized_buf(buf:UOp):
  if not isinstance(buf.dtype, PtrDType): return None
  if buf.addrspace not in (AddrSpace.LOCAL, AddrSpace.REG): return None
  # TODO: this fails on regs
  #assert buf.max_numel() == buf.ptrdtype.size
  sz = buf.ptrdtype.size*buf.ptrdtype.count
  return buf.replace(dtype=buf.ptrdtype.base.scalar().ptr(sz, buf.addrspace), src=(UOp.const(dtypes.int, sz),)).cast(buf.dtype)

def no_vectorized_index(buf:UOp, cast:UOp, idx:UOp, bcast:UOp|None=None):
  if buf.addrspace not in (AddrSpace.LOCAL, AddrSpace.REG): return None
  cnt = cast.dtype.count
  if bcast is not None and bcast.op is Ops.GEP:
    # GEP selects specific lanes; bcast.arg[k] is the offset for lane k, iterate groups × selected lanes
    pairs = [(k, g + bcast.arg[k]) for g, k in itertools.product(range(cast.dtype.vcount), range(len(bcast.arg)))]
  elif bcast is not None:
    # BROADCAST: cross product of components × lanes
    pairs = [(j, c) for c, j in itertools.product(range(cnt), range(bcast.dtype.vcount))]
  else:
    # simple scalar index: one lane, all components
    pairs = [(0, c) for c in range(cnt)]
  idx_lanes, offsets = (tuple(x) for x in zip(*pairs))
  return buf.broadcast(len(pairs)).index(idx.gep(idx_lanes)*cnt + UOp.const(dtypes.weakint.vec(len(pairs)), offsets), ptr=True)

devectorize_buf_and_index = PatternMatcher([
  (UPat(Ops.BUFFER, name="buf"), no_vectorized_buf),
  (UPat(Ops.BUFFER).or_after(name="buf").cast(name="cast").index(UPat.var("idx")), no_vectorized_index),
  (UPat(Ops.BUFFER).or_after(name="buf").cast(name="cast").broadcast(name="bcast").index(UPat.var("idx")), no_vectorized_index),
  (UPat(Ops.BUFFER).or_after(name="buf").cast(name="cast").gep(name="bcast").index(UPat.var("idx")), no_vectorized_index),
])

devectorize_alu = PatternMatcher([
  # CAST after AFTER
  (UPat(Ops.CAST, name="c").f(Ops.AFTER, allow_any_len=True, name="a"), lambda c,a: c.src[0].after(*a.src[1:]).cast(c.dtype)),
  # no ALU on vectorized dtypes
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
  (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
])

pm_render = PatternMatcher([
  # for rendering, we use explicit VECTORIZE
  (UPat(Ops.CONST, name='c'),
   lambda c: UOp(Ops.STACK, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: UOp(Ops.STACK, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: gep.src[0] if gep.src[0].dtype.vcount == 1 and gep.arg == (0,) else None),
  (UPat(Ops.STACK, src=(UPat(name='x'),)), lambda x: x),
])

# *** Ops.REDUCE -> Ops.DEFINE_ACC ***

@dataclass
class ReduceContext:
  acc_num: int = 0

def horizontal_reduce(inp:UOp, out_dtype:DType) -> list[UOp]:
  # if this has a horizontal reduction component, do that first
  if inp.dtype != out_dtype:
    # NOTE: [0 1 2 3 4 5 6 7] -> [0+4, 1+5, 2+6, 3+7]
    horizontal_amount = inp.dtype.count//out_dtype.count
    return [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount))) for i in range(0, horizontal_amount)]
  return [inp]

def reduce_to_acc(ctx:ReduceContext, red:UOp):
  inp, reduce_range = red.src[0], red.src[1:]
  lst = horizontal_reduce(inp, red.dtype)
  assert all(x.dtype == red.dtype for x in lst), f"horizontal reduction mismatch {lst[0].dtype} != {red.dtype}"
  # if we have a range
  if len(reduce_range) != 0:
    topo = inp.toposort()
    ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
    input_ranges = tuple([x for x in topo if x.op is Ops.RANGE and x not in reduce_range and x not in ended_ranges])
    identity = red.const(red.dtype, identity_element(red.arg[0], red.dtype.scalar()))
    acc = UOp.placeholder((1,), red.dtype, ctx.acc_num, AddrSpace.REG)
    acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(identity)
    lst = [acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))] + lst  # put acc as the first element
    ctx.acc_num += 1
  ret = functools.reduce(lambda x,y: x.alu(red.arg[0], y), lst)
  if len(reduce_range) == 0: return ret
  end = acc.index(UOp.const(dtypes.weakint, 0)).store(ret).end(*reduce_range).rtag("mergeable")
  return acc.after(end).index(UOp.const(dtypes.weakint, 0))

def merge_reduce_ends(ctx:ReduceContext, sink:UOp):
  # merge ENDs that share the same range and nesting context (only those created by reduce_to_acc)
  # ENDs at different nesting depths get cloned RANGEs so each RANGE maps to one END
  range_to_ends: dict[tuple[UOp, ...], list[UOp]] = {}
  for u in sink.backward_slice:
    if u.op is Ops.END and u.tag == "mergeable": range_to_ends.setdefault(u.src[1:], []).append(u)
  subs: dict[UOp, UOp] = {}
  next_axis = max((u.arg[0] for u in sink.backward_slice if u.op is Ops.RANGE), default=-1) + 1
  for r, ends in range_to_ends.items():
    if len(ends) <= 1: continue
    by_ctx: dict[frozenset[UOp], list[UOp]] = {}
    for e in ends: by_ctx.setdefault(frozenset(e.ranges), []).append(e)
    for i, group in enumerate(by_ctx.values()):
      tr = r if i == 0 else tuple(rr.replace(arg=(next_axis + j, *rr.arg[1:])) for j, rr in enumerate(r))
      if i > 0: next_axis += len(r)
      mapped = [e.substitute(dict(zip(r, tr))) if i > 0 else e for e in group]
      merged = mapped[0] if len(mapped) == 1 else UOp.group(*(e.src[0] for e in mapped)).end(*tr)
      for e in group: subs[e] = merged
  return sink.substitute(subs) if subs else None

pm_reduce = PatternMatcher([
  # invalid -> identity element
  (UPat(Ops.REDUCE, src=(invalid_gate,), allow_any_len=True, name="red"), lambda red,cond,x,i:
    red.replace(src=(cond.where(x, identity_element(red.arg[0], x.dtype.scalar())),)+red.src[1:])),
  # REDUCE -> DEFINE_ACC+ASSIGN, then merge ENDs with same range
  (UPat(Ops.REDUCE, name="red"), reduce_to_acc),
  (UPat(Ops.SINK, name="sink"), merge_reduce_ends),
  # tensor core built in accumulate
  (UPat(Ops.WMMA, name="wmma") + UPat.var("add"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
])

# add loads

def add_load(idx:UOp):
  if isinstance(idx.dtype, PtrDType): return None
  assert isinstance(idx.src[0].dtype, PtrDType), f"param is not PtrDType {idx.src[0].dtype}"
  return idx.replace(dtype=idx.src[0].dtype).load(dtype=idx.dtype.base)

pm_add_loads = PatternMatcher([
  # add loads to non ptr index
  (UPat(Ops.INDEX, name="idx"), add_load),
  # remove loads from stores
  (UPat(Ops.STORE, src=(UPat(Ops.LOAD),), allow_any_len=True, name="s"), lambda s: s.replace(src=(s.src[0].src[0],)+s.src[1:])),
  (UPat(Ops.LOAD, src=(UPat(Ops.LOAD),), allow_any_len=True, name="l"), lambda l: l.replace(src=(l.src[0].src[0],)+l.src[1:])),
])
