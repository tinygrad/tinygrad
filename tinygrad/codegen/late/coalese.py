from typing import Any
import itertools, functools
from collections import defaultdict
from tinygrad.dtype import dtypes, AddrSpace, Invalid, ImageDType, DType
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, GroupOp
from tinygrad.uop.symbolic import uop_given_valid, parse_valid, invalid_gate
from tinygrad.helpers import getenv, IMAGE, OSX, ceildiv
from tinygrad.renderer import Renderer

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

# get list of (height, width) that do not require pitch padding
def image_valid_dims(base:DType, size:int, arch:str) -> list[tuple[int,int]]:
  if (ALIGN:=next((int(p.split('=')[1]) for p in arch.split(',') if p.startswith("IMAGE_PITCH_ALIGNMENT=")), 0)) == 0: return []
  MAXW, pxls = 16384, size // 4
  if base not in (dtypes.half, dtypes.float) or size > 4*MAXW*MAXW: return []
  # height=1 images just need to abide by alignment requirements in bytes, not pixels!
  if size % (ALIGN * 4) != 0: return [] if (base.itemsize * size) % (64 if OSX else ALIGN) != 0 or pxls > MAXW else [(1, pxls)]
  return [(pxls//ALIGN//k, ALIGN*k) for k in range(ceildiv(pxls//ALIGN, MAXW), min(pxls//ALIGN, MAXW//ALIGN)+1) if (pxls//ALIGN)%k == 0]

def transform_to_image(ctx, buf:UOp, x:UOp) -> UOp|None:
  shapes, ren = ctx
  if not IMAGE or ren.target.device not in {"QCOM", "CL", "PYTHON", "NULL"}: return None
  valid = UOp.const(dtypes.bool, True)
  if x.op == Ops.WHERE and x.src[2].op == Ops.CONST and x.src[2].arg == Invalid: valid,x,_= x.src
  # search for dims that drop the most valid statements
  best_drop, cands = -1, []
  for ch, cw in [shapes[buf.arg.slot]] if buf.arg.slot in shapes else image_valid_dims(buf.dtype, buf.max_numel(), ren.target.arch):
    cidx = uop_given_valid(valid, ((x//4)%cw)._stack(x//(4*cw)))
    dropped = len(_drop_valid_stmts(valid, cidx, ch, cw))
    if dropped > best_drop: best_drop, cands = dropped, [(ch, cw, cidx)]
    elif dropped == best_drop: cands.append((ch, cw, cidx))
  # if no candidates, we don't rewrite
  if len(cands) == 0: return None
  # and tiebreak with indexing complexity (ie. number of nodes)
  h, w, cidx = cands[0] if len(cands) == 1 else min(cands, key=lambda cand: len(cand[2].index(1).simplify().backward_slice))
  buf = buf.replace(dtype=(dtypes.imageh if buf.dtype.itemsize == 2 else dtypes.imagef)((h, w, 4)))
  shapes[buf.arg.slot] = (h, w)
  if valid.op is not Ops.CONST or valid.arg is not True:
    return buf.index(valid.where(cidx.src[1], cidx.src[1].const_like(Invalid)),
                     valid.where(cidx.src[0], cidx.src[0].const_like(Invalid)))
  else:
    return buf.index(cidx.src[1], cidx.src[0])

pm_simplify_add_image = PatternMatcher([
  (UPat(Ops.SHRINK, src=(UPat(Ops.PARAM, name="buf"), UPat(name="x"), UPat(arg=4))), transform_to_image),
  # image load/store is always float
  (UPat(Ops.INDEX, dtype=dtypes.float, name="x").load(dtype=dtypes.half), lambda x: x.load().cast(dtypes.half)),
  (UPat(Ops.INDEX, dtype=dtypes.float, name="x").store(UPat(name="d", dtype=dtypes.half)), lambda x,d: x.store(d.cast(dtypes.float))),
  (UPat.var("x", dtype=dtypes.float).cast(dtypes.half).cast(dtypes.float), lambda x: x),
])

def memory_coalesing(sink:UOp, ctx:Renderer) -> UOp:
  if getenv("DMC"): return sink

  # collect
  memory: defaultdict[tuple[Ops, UOp, Any, Any], dict[int, list[UOp]]]  = defaultdict(dict)
  for u in sink.toposort():
    # TODO: this should handle images too, it's just memory coalesing
    if u.op in {Ops.LOAD, Ops.STORE}:
      assert len(u.src) == (2 if u.op is Ops.STORE else 1), "memory coalesing does not support gated loads/stores"
      assert u.src[0].op is Ops.INDEX, f"memory coalesing should be on INDEX, not {u.src[0].op}"
      buf, idx_u = u.src[0].src
      if buf.addrspace == AddrSpace.REG: continue
      idx: Any = idx_u.src[1] if idx_u.op is Ops.WHERE and idx_u.src[2].arg is Invalid else idx_u
      valid: Any = idx_u.src[0] if idx_u.op is Ops.WHERE and idx_u.src[2].arg is Invalid else None
      if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: root_src, arg = idx.src[0], idx.src[1].arg
      elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST: root_src, arg = idx.src[1], idx.src[0].arg
      elif idx.op is Ops.CONST and idx.arg is Invalid: root_src, arg = "INVALID", 0
      elif idx.op is Ops.CONST: root_src, arg = "CONST", idx.arg
      else: root_src, arg = idx, 0
      memory[(u.op, buf, root_src, valid)].setdefault(arg, []).append(u)

  # build replacements
  replacements = {}
  for (op,buf,base,valid),offsets in memory.items():
    # allowed lengths (copied in)
    lengths = []
    must_divide = True
    if ctx is not None and ctx.target.device == "DSP":
      lengths = [128,64,32,16,8,4]
      must_divide = False
    elif buf.dtype not in (dtypes.float, dtypes.half, *dtypes.fp8s) and not isinstance(buf.dtype, ImageDType):
      pass
    elif buf.addrspace == AddrSpace.REG:
      pass
    elif isinstance(buf.dtype, ImageDType):
      lengths = [4]
    elif ctx is not None and ctx.supports_float4:
      # TODO: a better way to get this than ctx
      lengths = [8,4,2] if buf.dtype == dtypes.half and getenv("ALLOW_HALF8") else [4,2]
    lengths.append(1)  # worst case, it's not folded
    # do the grouping
    grouped_offsets = [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for full_grp in grouped_offsets:
      while len(full_grp):
        offset = (base+full_grp[0]) if isinstance(base, UOp) else UOp.const(dtypes.weakint, full_grp[0])
        length = [l for l in lengths if l <= len(full_grp) and (not must_divide or offset.divides(l) is not None)][0]
        grp = full_grp[:length]
        # NOTE: we apply the valid again after we determine the length
        offset = valid.where(offset, UOp(Ops.CONST, offset.dtype, arg=Invalid)) if valid is not None else offset
        idx = UOp(Ops.SHRINK, dtype=buf.dtype, src=(buf, offset, UOp.const(dtypes.weakint, len(grp)))) if len(grp) > 1 else buf.index(offset)
        if op == Ops.STORE:
          datas = []
          for i,g in enumerate(grp):
            assert len(offsets[g]) == 1, f"attempting multiple stores: {len(offsets[g])}"
            datas.append(offsets[g][0].src[1])
          store = idx.store(UOp._stack(*datas) if len(datas) > 1 else datas[0])
          for i,g in enumerate(grp): replacements[offsets[g][0]] = store
        else:
          ld = idx.load()
          for i,g in enumerate(grp):
            for oo in offsets[g]:
              replacements[oo] = ld.index(UOp.const(dtypes.weakint, i)) if len(grp) > 1 else ld
        full_grp = full_grp[length:]

  # apply
  return sink.substitute(replacements, name="memory coalesing")
