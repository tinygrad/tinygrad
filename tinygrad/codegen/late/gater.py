# this is a temporary intermediate step while we remove this index style
from tinygrad.uop.ops import PatternMatcher, UPat, Ops
from tinygrad.dtype import Invalid, dtypes

_invalid = UPat(arg=Invalid)
pm_move_gates_from_index = PatternMatcher([
  # here we create the alt value for load to be 0s and remove the where Invalid (non-image: 2 srcs)
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), _invalid)).or_casted(name="cast").load(name="l"),
   lambda buf,gate,idx,cast,l: buf.index(idx, ptr=True).cast(cast.dtype).load(l.const_like(0), gate, dtype=l.dtype)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), _invalid)).or_casted(name="cast").store(UPat.var("data")),
   lambda buf,gate,idx,cast,data: buf.index(idx, ptr=True).cast(cast.dtype).store(data, gate)),

  # image: 3 srcs with matching gates on y and x
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("y"), _invalid),
                         UPat.var("gate").where(UPat.var("x"), _invalid)).or_casted(name="cast").load(name="l"),
   lambda buf,gate,y,x,cast,l: buf.index(y, x, ptr=True).cast(cast.dtype).load(l.const_like(0), gate, dtype=l.dtype)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("y"), _invalid),
                         UPat.var("gate").where(UPat.var("x"), _invalid)).or_casted(name="cast").store(UPat.var("data")),
   lambda buf,gate,y,x,cast,data: buf.index(y, x, ptr=True).cast(cast.dtype).store(data, gate)),
  # image: 3 srcs with mismatched gates on y and x — combine with AND
  (UPat.var("buf").index(UPat.var("gate_y").where(UPat.var("y"), _invalid),
                         UPat.var("gate_x").where(UPat.var("x"), _invalid)).or_casted(name="cast").load(name="l"),
   lambda buf,gate_y,y,gate_x,x,cast,l: buf.index(y, x, ptr=True).cast(cast.dtype).load(l.const_like(0), gate_y&gate_x, dtype=l.dtype)),
  (UPat.var("buf").index(UPat.var("gate_y").where(UPat.var("y"), _invalid),
                         UPat.var("gate_x").where(UPat.var("x"), _invalid)).or_casted(name="cast").store(UPat.var("data")),
   lambda buf,gate_y,y,gate_x,x,cast,data: buf.index(y, x, ptr=True).cast(cast.dtype).store(data, gate_y&gate_x)),

  # Where after gated load becomes alt value
  (UPat.var("gate").where(UPat().load(UPat(), UPat.var("gate", dtype=dtypes.bool), name="l").or_casted(), UPat.var("a")), lambda gate,l,a:
   l.replace(src=(l.src[0], a.src[0] if a.op is Ops.CAST and a.src[0].dtype == l.dtype else a.cast(l.dtype), l.src[2])).cast(a.dtype)),
  (UPat.var("gate").where(UPat.var("a"), UPat().load(UPat(), ~UPat.var("gate", dtype=dtypes.bool), name="l").or_casted()), lambda gate,l,a:
   l.replace(src=(l.src[0], a.src[0] if a.op is Ops.CAST and a.src[0].dtype == l.dtype else a.cast(l.dtype), l.src[2])).cast(a.dtype)),
])
