import itertools
from tinygrad.dtype import Invalid, dtypes
from tinygrad.uop.ops import UOp, Ops, GroupOp, shape_to_shape_arg

def _index_scalar(x:UOp, idx_c:tuple[UOp, ...]) -> UOp:
  from tinygrad.schedule.indexing import apply_movement_op
  ret = x.index(*idx_c)
  while ret.op is Ops.INDEX:
    if ret.src[0].op is Ops.INDEX and all(x._shape == () for x in ret.src[0].src[1:]+ret.src[1:]):
      ret = ret.src[0].src[0].index(*ret.src[0].src[1:], *ret.src[1:])
      continue
    if ret.src[0].op not in GroupOp.Movement: break
    r, idxs = ret.src[0], ret.src[1:]
    if len(idxs) == len(r.shape):
      ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idxs), dtype=ret.dtype, arg=ret.arg)
      continue
    if r.op is Ops.RESHAPE:
      src_prefix = len(r.src[0].shape) - len(r.shape[len(idxs):])
      if src_prefix >= 0 and r.src[0].shape[src_prefix:] == r.shape[len(idxs):]:
        if src_prefix == 0:
          if r.src[0].dtype == ret.dtype: ret = r.src[0]
          break
        moved = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape[:src_prefix], r.shape[:len(idxs)], idxs), dtype=ret.dtype, arg=ret.arg)
        if moved.shape == ret.shape:
          ret = moved
          continue
    break
  return ret

def do_devectorize_fast(b:UOp) -> UOp|None:
  if (shape:=b._shape) is None: raise RuntimeError(f"shape requested, but {b.op} doesn't have a shape")
  if shape == (): return None
  if not all(x._shape == shape or x.base.arg is Invalid for x in b.src): return None
  src_info:list[tuple[UOp, UOp]] = []
  for x in b.src: src_info.append((x, x.base))
  src:list[UOp] = []
  if len(shape) == 1:
    for i in range(int(shape[0])):
      idx_c_1:tuple[UOp, ...] = tuple([UOp.const(i, dtypes.int)])
      new_src:list[UOp] = []
      for x,base in src_info: new_src.append(base if base.arg is Invalid else _index_scalar(x, idx_c_1))
      src.append(UOp(b.op, None, tuple(new_src), b.arg, b.tag))
  else:
    for idx in itertools.product(*map(range, shape)):
      idx_c_n:tuple[UOp, ...] = tuple(UOp.const(i, dtypes.int) for i in idx)
      new_src = []
      for x,base in src_info: new_src.append(base if base.arg is Invalid else _index_scalar(x, idx_c_n))
      src.append(UOp(b.op, None, tuple(new_src), b.arg, b.tag))
  if b.op is Ops.STORE: return UOp.group(*src)
  return UOp(Ops.RESHAPE, b.dtype, (UOp(Ops.STACK, b.dtype, tuple(src)), shape_to_shape_arg(shape)))
