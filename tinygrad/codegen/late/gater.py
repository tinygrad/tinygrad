# this is a temporary intermediate step while we remove this index style
from tinygrad.uop.ops import PatternMatcher, UPat, Ops
from tinygrad.dtype import Invalid, dtypes

def move_where_load(gate, l, a): return l.replace(src=(l.src[0], l.vconst_like(0) if a.is_invalid else a.cast(l.dtype), l.src[2]))

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
   .load(name="l"), lambda mop,gate,idx,l: mop.replace(src=(mop.src[0],idx)+mop.src[2:]).load(l.vconst_like(0), gate)),
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(), UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid)),), name="mop", allow_any_len=True) \
   .store(UPat.var("data")), lambda mop,gate,idx,data: mop.replace(src=(mop.src[0],idx)+mop.src[2:]).store(data, gate)),

  # Where after gated load becomes alt value
  (UPat.var("gate").where(UPat().load(UPat(), UPat.var("gate", dtype=dtypes.bool), name="l"), UPat.var("a")), move_where_load),
  (UPat.var("gate").where(UPat.var("a"), UPat().load(UPat(), ~UPat.var("gate", dtype=dtypes.bool), name="l")), move_where_load),
])
