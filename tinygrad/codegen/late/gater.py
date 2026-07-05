# this is a temporary intermediate step while we remove this index style
from tinygrad.uop.ops import PatternMatcher, UPat, Ops
from tinygrad.dtype import Invalid, dtypes

def _load_idx(buf, gate, idx):
  # a gated load's address must stay in-bounds even when the gate is false
  if 0 <= idx.vmin and idx.vmax < buf.max_numel(): return idx
  # TODO: this is complicated
  base = idx.src[0] if idx.op is Ops.CAST and idx.dtype is dtypes.weakint and dtypes.is_int(idx.src[0].dtype) else idx
  return gate.where(base, base.const_like(0)).cast(idx.dtype)

pm_move_gates_from_index = PatternMatcher([
  # for image idx (must be first)
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx_y"), UPat(arg=Invalid)),
                         UPat.var("gate").where(UPat.var("idx_x"), UPat(arg=Invalid))).load(name="l"),
   lambda buf,gate,idx_y,idx_x,l: buf.index(idx_y, idx_x).load(l.vconst_like(0), gate)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx_y"), UPat(arg=Invalid)),
                         UPat.var("gate").where(UPat.var("idx_x"), UPat(arg=Invalid))).store(UPat.var("data")),
   lambda buf,gate,idx_y,idx_x,data: buf.index(idx_y, idx_x).store(data, gate)),

  # here we create the alt value for load to be 0s and remove the where Invalid
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(), UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid)),), name="mop", allow_any_len=True) \
   .load(name="l"), lambda mop,gate,idx,l: mop.replace(src=(mop.src[0],_load_idx(mop.src[0],gate,idx))+mop.src[2:]).load(l.vconst_like(0), gate)),
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(), UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid)),), name="mop", allow_any_len=True) \
   .store(UPat.var("data")), lambda mop,gate,idx,data: mop.replace(src=(mop.src[0],idx)+mop.src[2:]).store(data, gate)),

  # Where after gated load becomes alt value
  (UPat.var("gate").where(UPat().load(UPat(), UPat.var("gate", dtype=dtypes.bool), name="l").or_casted(), UPat.var("a")), lambda gate,l,a:
   l.replace(src=(l.src[0], a.src[0] if a.op is Ops.CAST and a.src[0].dtype == l.dtype else a.cast(l.dtype), l.src[2])).cast(a.dtype)),
  (UPat.var("gate").where(UPat.var("a"), UPat().load(UPat(), ~UPat.var("gate", dtype=dtypes.bool), name="l").or_casted()), lambda gate,l,a:
   l.replace(src=(l.src[0], a.src[0] if a.op is Ops.CAST and a.src[0].dtype == l.dtype else a.cast(l.dtype), l.src[2])).cast(a.dtype)),
])
