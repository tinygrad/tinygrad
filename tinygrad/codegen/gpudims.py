from tinygrad.uop.ops import UOp, Ops, sint, PatternMatcher, UPat, KernelInfo, ssimplify, AxisType
from tinygrad.helpers import partition, flatten, prod, dedup
from tinygrad.dtype import dtypes
from tinygrad.renderer import Renderer

def get_grouped_dims(prefix, dims:tuple[sint, ...], reverse=False) -> list[UOp]:
  if reverse: dims = dims[::-1]
  spec = UOp(Ops.SPECIAL, dtypes.int, (), (f"{prefix}0", ssimplify(prod(dims))))
  ret = []
  for d in dims:
    ret.append(spec % d)
    spec //= d
  return ret[::-1] if reverse else ret

def add_gpudims(ctx:Renderer, s:UOp):
  if s.arg is None: return None
  s_topo = list(s.toposort())
  if any(x.op is Ops.SPECIAL for x in s_topo): return None

  # get ranges
  all_ranges = {x.arg[0]%1000:x for x in s_topo if x.op is Ops.RANGE}

  # extract global/local dims
  global_dims = sorted(dedup([x.arg[0]%1000 for x in all_ranges.values() if x.arg[1] is AxisType.GLOBAL]))
  local_dims = sorted(dedup([x.arg[0]%1000 for x in all_ranges.values() if x.arg[1] in (AxisType.LOCAL, AxisType.GROUP_REDUCE)]))
  if not global_dims and not local_dims: return None

  # get global and local shape
  ranges = [all_ranges[r] for r in global_dims+local_dims if r in all_ranges]
  global_shape = tuple([ssimplify(r.src[0]) for r in ranges if r.arg[0]%1000 in global_dims])
  local_shape = tuple([ssimplify(r.src[0]) for r in ranges if r.arg[0]%1000 in local_dims])

  # get the idxs
  ki: KernelInfo = s.arg
  if ki.dont_use_locals:
    assert not local_dims, "can't use locals if there's no local dims"
    idxs = get_grouped_dims("idx", global_shape, reverse=True)
  else:
    # define indexes for GPU-like execution
    idxs = get_grouped_dims("gidx", global_shape, reverse=True) + get_grouped_dims("lidx", local_shape)

  # apply to multiple ranges
  subs = {}
  for r in s_topo:
    if r.op is not Ops.RANGE: continue
    try:
      ii = (global_dims+local_dims).index(r.arg[0]%1000)
      if r.arg[0] < 2000 and r.arg[1] == AxisType.GROUP_REDUCE: continue
      subs[r] = idxs[ii]
    except ValueError: continue
  return s.substitute(subs)

def fix_reduce_unroll(x:UOp):
  reduce_range, reduce_expand = partition(x.src[1:], lambda y: y.op is Ops.RANGE)
  if len(reduce_expand) == 0: return None
  assert all(x.op is Ops.UNROLL for x in reduce_expand), f"not all UNROLLS in {reduce_expand} for {x.axis_arg}"
  ret = x.src[0]
  if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
    ret = UOp(Ops.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis), tag=1)
  # REDUCE supports both "horizontal" reduction and range reduction. the horizontal elements are taken in the nearest group
  return x.replace(src=(ret,)+tuple(reduce_range))

def fix_store_unroll(x:UOp):
  store_expand, store_range = partition(x.src[2:], lambda y: y.op is Ops.UNROLL)
  if len(store_expand) == 0: return None
  return UOp(Ops.CONTRACT, dtypes.void, (x.replace(src=x.src[:2]+tuple(store_range)),), tuple(flatten(x.arg for x in store_expand)), tag=1)

pm_add_gpudims = PatternMatcher([
  (UPat(Ops.SINK, name="s"), add_gpudims),
  # rewrite UPCAST/UNROLL range to something to be expanded
  (UPat(Ops.RANGE, name="r"),
   lambda r: UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(s:=r.vmax+1), tuple(range(s))),), ((r.arg[0],s),)) \
    if r.arg[1] in {AxisType.UNROLL, AxisType.UPCAST} else None),
  # fix REDUCEs with UNROLLs
  (UPat(Ops.REDUCE, name="x"), fix_reduce_unroll),
  (UPat(Ops.STORE, name="x"), fix_store_unroll),
])
