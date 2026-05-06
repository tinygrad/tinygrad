# this is a temporary intermediate step while we remove this index style
from tinygrad.uop.ops import PatternMatcher, UPat

pm_move_gates_from_index = PatternMatcher([
  (UPat.var("buf").index(UPat.var("idx"), UPat.var("gate")).or_casted(name="cast").load(name="l"),
    lambda buf,gate,idx,cast,l: buf.index(idx, ptr=True).cast(cast.dtype).load(l.const_like(0), gate, dtype=l.dtype)),
  (UPat.var("buf").index(UPat.var("idx"), UPat.var("gate")).or_casted(name="cast").store(UPat.var("data")),
    lambda buf,gate,idx,cast,data: buf.index(idx, ptr=True).cast(cast.dtype).store(data, gate)),
])
