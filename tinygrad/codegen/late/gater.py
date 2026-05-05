from tinygrad.uop.ops import PatternMatcher, UPat
from tinygrad.dtype import Invalid

pm_move_gates_from_index = PatternMatcher([
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid))).or_casted(name="cast").load(name="l"),
   lambda buf,gate,idx,cast,l: buf.index(idx).cast(cast.dtype).load(gate, l.const_like(0), dtype=l.dtype)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid))).or_casted(name="cast").store(UPat.var("data")),
   lambda buf,gate,idx,cast,data: buf.index(idx).cast(cast.dtype).store(data, gate)),
])
