import functools
from dataclasses import dataclass
from tinygrad.dtype import dtypes, ImageDType, DType, Invalid
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp
from tinygrad.uop.symbolic import uop_given_valid, parse_valid, invalid_gate
from tinygrad.helpers import OSX, ceildiv

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
      if testidx.index(0).vmax < 0 or testidx.index(1).vmax < 0:
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
  start_idx = idx_x._stack(idx_y)
  idx = uop_given_valid(valid, start_idx)
  drop_stmt = _drop_valid_stmts(valid, idx, buf.dtype.shape[0], buf.dtype.shape[1])

  if not drop_stmt and idx is start_idx: return None
  new_valid = UOp.uprod(*ss) if (ss:=[s for s in valid.split_uop(Ops.AND) if s not in drop_stmt]) else None
  idx_y, idx_x = idx.index(1), idx.index(0)
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

pm_render = PatternMatcher([
  # for rendering, we use explicit VECTORIZE
  (UPat(Ops.CONST, name='c'),
   lambda c: UOp(Ops.STACK, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  (UPat(Ops.STACK, src=(UPat(name='x'),)), lambda x: x),
])

# *** Ops.REDUCE -> Ops.DEFINE_ACC ***

@dataclass
class ReduceContext:
  acc_num: int = 0

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
