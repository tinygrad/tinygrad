# this is a temporary intermediate step while we remove this index style
from tinygrad.uop.ops import PatternMatcher, UPat, Ops
from tinygrad.uop.symbolic import invalid_pat
from tinygrad.dtype import dtypes

pm_move_gates_from_index = PatternMatcher([
  # for image idx (must be first)
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx_y"), invalid_pat),
                         UPat.var("gate").where(UPat.var("idx_x"), invalid_pat)).load(name="l"),
   lambda buf,gate,idx_y,idx_x,l: buf.index(idx_y, idx_x, dtype=dtypes.float).load(l.vconst_like(0), gate)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx_y"), invalid_pat),
                         UPat.var("gate").where(UPat.var("idx_x"), invalid_pat)).store(UPat.var("data")),
   lambda buf,gate,idx_y,idx_x,data: buf.index(idx_y, idx_x, dtype=dtypes.float).store(data, gate)),

  # here we create the alt value for load to be 0s and remove the where Invalid
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(), UPat.var("gate").where(UPat.var("idx"), invalid_pat),), name="mop", allow_any_len=True) \
   .load(name="l"), lambda mop,gate,idx,l: mop.replace(src=(mop.src[0],idx)+mop.src[2:]).load(l.vconst_like(0), gate)),
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(), UPat.var("gate").where(UPat.var("idx"), invalid_pat),), name="mop", allow_any_len=True) \
   .store(UPat.var("data")), lambda mop,gate,idx,data: mop.replace(src=(mop.src[0],idx)+mop.src[2:]).store(data, gate)),

  # Where after gated load becomes alt value
  (UPat.var("gate").where(UPat().load(UPat(), UPat.var("gate", dtype=dtypes.bool), name="l").or_casted(), UPat.var("a")).named("w"),
   lambda gate,l,a,w: l.replace(src=(l.src[0],
     a.src[0] if a.op is Ops.CAST and a.src[0].dtype == l.dtype else a.cast(l.dtype), l.src[2])).cast(w.dtype)),
  (UPat.var("gate").where(UPat.var("a"), UPat().load(UPat(), ~UPat.var("gate", dtype=dtypes.bool), name="l").or_casted()).named("w"),
   lambda gate,l,a,w: l.replace(src=(l.src[0],
     a.src[0] if a.op is Ops.CAST and a.src[0].dtype == l.dtype else a.cast(l.dtype), l.src[2])).cast(w.dtype)),
])
