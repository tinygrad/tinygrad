# this is a temporary intermediate step while we remove this index style
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.dtype import Invalid, dtypes, ImageDType

def move_image_load_gate(buf:UOp, gate:UOp, x:UOp, y:UOp, cast:UOp, l:UOp):
  if not isinstance(buf.dtype, ImageDType): return None
  return buf.index(x, y, ptr=True).cast(cast.dtype).load(l.const_like(0), gate, dtype=l.dtype)

def move_image_store_gate(buf:UOp, gate:UOp, x:UOp, y:UOp, cast:UOp, data:UOp):
  if not isinstance(buf.dtype, ImageDType): return None
  return buf.index(x, y, ptr=True).cast(cast.dtype).store(data, gate)

def image_coords_to_int(idx:UOp, buf:UOp, x:UOp, y:UOp):
  if not isinstance(buf.dtype, ImageDType) or (x.dtype != dtypes.long and y.dtype != dtypes.long): return None
  return idx.replace(src=(buf, x.cast(dtypes.int) if x.dtype == dtypes.long else x, y.cast(dtypes.int) if y.dtype == dtypes.long else y))

def index_and_valid(idx:UOp) -> tuple[UOp, UOp]:
  if idx.dtype.scalar() is dtypes.weakint: return idx.get_idx(), idx.get_valid()
  if idx.op is Ops.WHERE and idx.src[2].arg is Invalid: return idx.src[1], idx.src[0]
  return idx, UOp.const(dtypes.bool, idx.arg is not Invalid)

def valid_idx(idx:UOp, valid:UOp) -> UOp:
  return idx if valid.op is Ops.CONST and valid.arg is True else valid.where(idx, idx.const_like(Invalid))

def get_image_idx(idx:UOp, height:int, width:int) -> UOp:
  x, valid = index_and_valid(idx.src[1])
  px = x // 4
  idx_x, idx_y = (px, px.const_like(0)) if height == 1 else (px % width, px // width)
  return idx.replace(src=(idx.src[0], valid_idx(idx_x, valid), valid_idx(idx_y, valid)))

def image_fixup(ls:UOp):
  # normal image load/store from split_load_store: casted linear offset -> image x/y coordinates
  if ls.src[0].op is Ops.CAST and (cast_idx:=ls.src[0].src[0]).op is Ops.INDEX and isinstance(dt:=cast_idx.src[0].dtype, ImageDType):
    assert ls.src[0].dtype.count == 4, "image must be casted to 4"
    return ls.replace(src=(cast_idx if len(cast_idx.src) == 3 else get_image_idx(cast_idx, dt.shape[0], dt.shape[1]),)+ls.src[1:])

  if ls.src[0].op is not Ops.INDEX or not isinstance(dt:=ls.src[0].src[0].dtype, ImageDType) or len(ls.src[0].src) == 3: return None

  # this is an unprocessed image without a cast, we should just make it a buffer
  idx = ls.src[0].src[0].replace(dtype=(new_dt:=dtypes.half if dt.itemsize == 2 else dtypes.float).ptr(dt.size)).index(ls.src[0].src[1])
  return ls.replace(src=(idx,), dtype=new_dt).cast(dtypes.float) if ls.op is Ops.LOAD else ls.replace(src=(idx, ls.src[1].cast(new_dt)))

pm_image_index = PatternMatcher([
  (UPat((Ops.LOAD, Ops.STORE), name="ls"), image_fixup),
])

pm_move_gates_from_index = PatternMatcher([
  # here we create the alt value for load to be 0s and remove the where Invalid
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid))).or_casted(name="cast").load(name="l"),
   lambda buf,gate,idx,cast,l: buf.index(idx, ptr=True).cast(cast.dtype).load(l.const_like(0), gate, dtype=l.dtype)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("idx"), UPat(arg=Invalid))).or_casted(name="cast").store(UPat.var("data")),
   lambda buf,gate,idx,cast,data: buf.index(idx, ptr=True).cast(cast.dtype).store(data, gate)),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("x"), UPat(arg=Invalid)),
                           UPat.var("gate").where(UPat.var("y"), UPat(arg=Invalid))).or_casted(name="cast").load(name="l"),
   move_image_load_gate),
  (UPat.var("buf").index(UPat.var("gate").where(UPat.var("x"), UPat(arg=Invalid)),
                           UPat.var("gate").where(UPat.var("y"), UPat(arg=Invalid))).or_casted(name="cast").store(UPat.var("data")),
   move_image_store_gate),

  # Where after gated load becomes alt value
  (UPat.var("gate").where(UPat().load(UPat(), UPat.var("gate"), name="l").or_casted(), UPat.var("a")), lambda gate,l,a:
   l.replace(src=(l.src[0], a.src[0] if a.op is Ops.CAST and a.src[0].dtype == l.dtype else a.cast(l.dtype), l.src[2])).cast(a.dtype)),
  (UPat.var("gate").where(UPat.var("a"), UPat().load(UPat(), ~UPat.var("gate", dtype=dtypes.bool), name="l").or_casted()), lambda gate,l,a:
   l.replace(src=(l.src[0], a.src[0] if a.op is Ops.CAST and a.src[0].dtype == l.dtype else a.cast(l.dtype), l.src[2])).cast(a.dtype)),

  # vectorized indexes must be int
  (UPat(Ops.INDEX, src=(UPat(), UPat(Ops.STACK, dtypes.long, name="vec")), allow_any_len=True, name="idx"),
   lambda idx,vec: idx.replace(src=(idx.src[0], UOp.vectorize(*(u.cast(dtypes.int) for u in vec.src)), *idx.src[2:]))),
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("x"), UPat.var("y")), name="idx"), image_coords_to_int),
])
