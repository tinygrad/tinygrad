import math
from tinygrad.uop.ops import UOp, Ops, sint, PatternMatcher, UPat, KernelInfo, ssimplify, AxisType, sint_to_uop
from tinygrad.helpers import all_int, dedup
from tinygrad.dtype import dtypes, AddrSpace, Invalid
from tinygrad.renderer import Renderer

def _group_dims(dims:tuple[sint, ...], max_sizes:tuple[int, ...]):
  # TODO: symbolic shape
  if not all_int(dims): return dims
  while len(dims) > len(max_sizes) or any(d > m for d,m in zip(dims, max_sizes)):
    for i,m in enumerate(max_sizes):
      if i < (len(dims)-1) and dims[i] * dims[i+1] <= m:
        dims = dims[:i] + (dims[i]*dims[i+1],) + dims[i+2:]
        break
    else: return None
  return dims

def _split_dims(dims, max_sizes):
  if all(d <= m for d,m in zip(dims, max_sizes)): return dims
  _dims = list(dims) + [1]*(3-len(dims))
  for i in range(len(_dims)):
    while _dims[i] > max_sizes[i]:
      div = next((d for d in range(2, math.ceil(math.sqrt(_dims[i])) + 1) if (_dims[i] % d) == 0), 1)
      if div == 1: raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
      _dims[i], _dims[(i+1)%len(_dims)] = _dims[i]//div, _dims[(i+1)%len(_dims)]*div
  while len(_dims) > 1 and _dims[-1] == 1: _dims.pop()
  return tuple(_dims)

def get_grouped_dims(prefix, dims:tuple[sint, ...], max_sizes:tuple[int, ...]|None, reverse=False) -> list[UOp]:
  if reverse: return get_grouped_dims(prefix, dims[::-1], max_sizes)[::-1]
  if max_sizes is None: limited = dims
  else:
    # try to group first: (a, b, c, d) -> (ab, c, d)
    limited = grouped if (grouped := _group_dims(dims, max_sizes)) else dims
    # check if grouping failed
    if len(limited) > len(max_sizes): raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
    # try to split up dims: (a,) -> (b, c)
    if limited == dims: limited = _split_dims(dims, max_sizes)
  raw_idxs = [UOp(Ops.SPECIAL, dtypes.index, (sint_to_uop(s),), (f"{prefix}{i}")) for i,s in enumerate(limited)]
  if limited == dims: return raw_idxs
  if len(limited) < len(dims):
    # per-group decomposition: each limited dim is a product of consecutive original dims
    j, ret = 0, []
    for raw_idx, l in zip(raw_idxs, limited):
      group_start: int = j
      prod: sint = 1
      while prod < l:
        prod *= dims[j]
        j += 1
      for k in range(group_start, j-1):
        ret.append(raw_idx % dims[k])
        raw_idx = raw_idx // dims[k]
      ret.append(raw_idx)
    return ret
  # flatten limited indices to 1D, then extract original dims
  flat = raw_idxs[0]
  for i in range(1, len(limited)): flat = flat * limited[i] + raw_idxs[i]
  ret = []
  for i in range(len(dims)-1, 0, -1):
    ret.append(flat % dims[i])
    flat = flat // dims[i]
  return [flat] + ret[::-1]

def add_gpudims(ctx:Renderer, s:UOp):
  if s.arg is None: return None
  s_topo = list(s.toposort())
  if any(x.op is Ops.SPECIAL for x in s_topo): return None

  # get ranges
  all_ranges = {x.arg[0:-1]:x for x in s_topo if x.op is Ops.RANGE}

  # extract global/local dims
  global_dims = sorted(dedup([x.arg[0:-1] for x in all_ranges.values() if x.arg[-1] in (AxisType.GLOBAL, AxisType.THREAD)]))
  local_dims = sorted(dedup([x.arg[0:-1] for x in all_ranges.values() if x.arg[-1] in (AxisType.WARP, AxisType.LOCAL, AxisType.GROUP_REDUCE)]))
  if not global_dims and not local_dims: return None

  # get global and local shape
  ranges = [all_ranges[r] for r in global_dims+local_dims if r in all_ranges]
  global_shape = tuple([ssimplify(r.src[0]) for r in ranges if r.arg[0:-1] in global_dims])
  local_shape = tuple([ssimplify(r.src[0]) for r in ranges if r.arg[0:-1] in local_dims])

  # get the idxs
  ki: KernelInfo = s.arg
  if ctx.has_threads: idxs = [UOp.variable("core_id", 0, int(global_shape[0])-1, dtypes.int).cast(dtypes.index)]
  elif ki.dont_use_locals:
    assert not local_dims, "can't use locals if there's no local dims"
    idxs = get_grouped_dims("idx", global_shape, ctx.global_max, reverse=True)
  else:
    # define indexes for GPU-like execution
    idxs = get_grouped_dims("gidx", global_shape, ctx.global_max, reverse=True) + get_grouped_dims("lidx", local_shape, ctx.local_max)

  # apply to multiple ranges
  subs = {}
  for r in s_topo:
    # look for local INDEXes that are not used in the GLOBAL store, then add them as an INVALID
    if r.op is Ops.STORE and (idx := r.src[0]).src[0].ptrdtype.addrspace == AddrSpace.GLOBAL:
      missing_locals = [all_ranges[rng] for rng in local_dims if all_ranges[rng] not in idx.ranges]
      if len(missing_locals):
        assert len(idx.src) == 2, "index has 2 sources"
        mask: UOp = UOp.prod(*[x.eq(0) for x in missing_locals])
        subs[idx] = idx.replace(src=(idx.src[0], mask.broadcast(idx.src[1].dtype.count).where(idx.src[1], Invalid)))
    if r.op is not Ops.RANGE: continue
    try:
      ii = (global_dims+local_dims).index(r.arg[0:-1])
      if r.arg[1] == AxisType.REDUCE: continue
      subs[r] = idxs[ii]
    except ValueError: continue
  return s.substitute(subs)

pm_add_gpudims = PatternMatcher([
  # add gpudims must be last
  (UPat(Ops.SINK, name="s"), add_gpudims),
])
