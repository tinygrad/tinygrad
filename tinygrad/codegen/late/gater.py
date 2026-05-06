# this transforms Invalid into gated load/stores

from tinygrad.uop.ops import PatternMatcher, UPat
from tinygrad.dtype import Invalid, dtypes

pm_move_gates_from_index = PatternMatcher([
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid))).or_casted(name="cast").load(name="l"),
   lambda buf,gate,idx,cast,l: buf.index(idx, ptr=True).cast(cast.dtype).load(l.const_like(0), gate, dtype=l.dtype)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid))).or_casted(name="cast").store(UPat.var("data")),
   lambda buf,gate,idx,cast,data: buf.index(idx, ptr=True).cast(cast.dtype).store(data, gate)),
  # remove hanging weakint casts
  (UPat.var("buf").index(UPat.var("idx", dtypes.ints).cast()), lambda buf,idx: buf.index(idx, ptr=True)),
])
