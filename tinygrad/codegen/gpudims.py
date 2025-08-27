import math, functools, operator
from tinygrad.uop.ops import UOp, Ops, sint, PatternMatcher, UPat, KernelInfo, ssimplify, AxisType, graph_rewrite
from tinygrad.helpers import all_int, partition, flatten, prod, dedup, USE_TC, DEBUG
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.shape.view import get_contraction
from tinygrad.renderer import Renderer
from tinygrad.codegen.opt.tc import TensorCore

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
  return tuple(_dims[:2] if _dims[2] == 1 else _dims[0] if _dims[1:3] == [1,1] else _dims)

def get_grouped_dims(prefix, dims:tuple[sint, ...], max_sizes:tuple[int, ...]|None, reverse=False) -> list[UOp]:
  if reverse: dims = dims[::-1]
  # try to group first: (a, b, c, d) -> (ab, c, d)
  limited = (grouped if (grouped := _group_dims(dims, max_sizes)) else dims) if max_sizes is not None else dims
  # check if grouping failed
  if max_sizes is not None and len(limited) > len(max_sizes): raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
  # try to split up dims: (a,) -> (b, c)
  if limited == dims: limited = _split_dims(dims, max_sizes) if max_sizes is not None else dims
  ret = raw_idxs = [UOp(Ops.SPECIAL, dtypes.int, (), (f"{prefix}{i}", s)) for i,s in enumerate(limited)]
  if len(limited) < len(dims):
    ret = []
    if (contraction:=get_contraction(dims, limited)) is None: raise AssertionError(f"get_contraction should not be None {dims=} {limited=}")
    for idx, contraction_group in zip(raw_idxs, contraction):
      for c in contraction_group[:-1]:
        ret.append(idx % dims[c])
        idx //= dims[c]
      ret.append(idx)
  elif len(limited) > len(dims):
    a, b = len(limited), len(dims)
    if a == 2 and b == 1: ret = [raw_idxs[0] * limited[1] + raw_idxs[1]]
    if a == 3 and b == 1: ret = [raw_idxs[0] * (limited[1] * limited[2]) + raw_idxs[1] * limited[2] + raw_idxs[2]]
    if a == 3 and b == 2: ret = [raw_idxs[0] * limited[1] + raw_idxs[1], raw_idxs[2]]
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
    idxs = get_grouped_dims("idx", global_shape, ctx.global_max, reverse=True)
  else:
    # define indexes for GPU-like execution
    idxs = get_grouped_dims("gidx", global_shape, ctx.global_max, reverse=True) + get_grouped_dims("lidx", local_shape, ctx.local_max)

  # apply to multiple ranges
  subs = {}
  for r in s_topo:
    if r.op is not Ops.RANGE: continue
    try:
      ii = (global_dims+local_dims).index(r.arg[0]%1000)
      if r.arg[1] == AxisType.REDUCE: continue
      subs[r] = idxs[ii]
    except ValueError: continue
  return s.substitute(subs)

def fix_reduce_unroll(x:UOp):
  reduce_range, reduce_expand = partition(x.src[1:], lambda y: y.op is Ops.RANGE)
  if len(reduce_expand) == 0: return None
  reduce_expand = [x for x in reduce_expand if x.op is not Ops.CONST]
  assert all(x.op is Ops.UNROLL for x in reduce_expand), f"not all UNROLLS in {reduce_expand}"
  ret = x.src[0]
  if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
    ret = UOp(Ops.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis), tag=1)
  # REDUCE supports both "horizontal" reduction and range reduction. the horizontal elements are taken in the nearest group
  return x.replace(src=(ret,)+tuple(reduce_range))

def fix_store_unroll(x:UOp):
  store_expand, store_range = partition(x.src[2:], lambda y: y.op is Ops.UNROLL)
  if len(store_expand) == 0: return None
  return UOp(Ops.CONTRACT, dtypes.void, (x.replace(src=x.src[:2]+tuple(store_range)),), tuple(flatten(x.arg for x in store_expand)), tag=1)

def fix_group_for_reduce(x:UOp):
  reduce_gfr, reduce_r = partition(x.src[1:], lambda u: u.op is Ops.RANGE and u.arg[1] == AxisType.GROUP_REDUCE)
  if len(reduce_gfr) == 0: return None

  # NOTE: if there's other locals here, we need them in the buffer too
  upstream_locals = [u for u in x.toposort() if u.op is Ops.RANGE and u.arg[1] == AxisType.LOCAL]

  # do only the non grouped reduces early
  ret = x.replace(src=(x.src[0],)+tuple(reduce_r))
  reduce_loop = [x.replace(arg=(x.arg[0]+100, AxisType.REDUCE)) for x in reduce_gfr]
  buf = ret.bufferize(*upstream_locals, *reduce_gfr, arg=AddrSpace.LOCAL).index(*upstream_locals, *reduce_loop)

  # gate with an if on the store + do the final reduce
  buf = UOp(Ops.IF, dtype=buf.dtype, src=(functools.reduce(operator.and_, [x.eq(0) for x in reduce_gfr]), buf))
  return buf.reduce(*reduce_loop, arg=x.arg)

pm_group_for_reduce = PatternMatcher([
  # fix group for reduce
  (UPat(Ops.REDUCE, name="x"), fix_group_for_reduce),
])

pm_add_gpudims = PatternMatcher([
  # add gpudims must be last
  (UPat(Ops.SINK, name="s"), add_gpudims),
  # rewrite UPCAST/UNROLL range to something to be expanded
  (UPat(Ops.RANGE, name="r"),
   lambda r: UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(s:=r.vmax+1), tuple(range(s))),), ((r.arg[0],s),)) \
    if r.arg[1] in {AxisType.UNROLL, AxisType.UPCAST} else None),
  # fix REDUCEs with UNROLLs
  (UPat(Ops.REDUCE, name="x"), fix_reduce_unroll),
  (UPat(Ops.STORE, name="x"), fix_store_unroll),
])

def apply_tensor_cores(ctx:tuple[dict, Renderer], in0:UOp, in1:UOp, r_range:UOp, reduceop:UOp):
  if not USE_TC: return None
  # tensor cores have three ranges. X, Y, and REDUCE
  in0_ranges = [u for u in in0.ranges if u not in in1.ranges]
  in1_ranges = [u for u in in1.ranges if u not in in0.ranges]
  if len(in0_ranges) != 1 or len(in1_ranges) != 1: return None
  in0_range, in1_range = in0_ranges[0], in1_ranges[0]
  if DEBUG >= 2: print('TC', in0_range.arg, in1_range.arg, r_range.arg)

  # confirm the dtype and size is good
  tc_opts: list[TensorCore] = []
  for tc in ctx[1].tensor_cores:
    if reduceop.dtype == tc.dtype_out and in0.dtype == tc.dtype_in and in1.dtype == tc.dtype_in:
      if all(i <= j for i,j in zip(tc.dims, [in0_range.vmax+1, in1_range.vmax+1, r_range.vmax+1])):
        tc_opts.append(tc)
  if len(tc_opts) == 0: return None
  tc = tc_opts[0]

  # create the new ranges as speced by the tensor core
  old_range = [in0_range, in1_range, r_range]
  new_range = [r.replace(src=(r.src[0]//tc.dims[i],)) for i,r in enumerate(old_range)]
  new_reduce_range = new_range[2]
  tc_range = 9050 + r_range.arg[0]*100
  red_ranges = []

  ne: list[UOp] = []
  for o in tc.opts:
    lrange = UOp.range(dtypes.int, 2, tc_range, AxisType.UPCAST if o[0] == "u" else AxisType.LOCAL)
    ne.append(lrange)
    tc_range += 1
    new_range[1-int(o[1])] = (2 * new_range[1-int(o[1])]) + lrange
  for _, amt in tc.get_reduce_axes():
    lrange = UOp.range(dtypes.int, amt, tc_range, AxisType.UNROLL)
    ne.append(lrange)
    red_ranges.append(lrange)
    tc_range += 1
    new_range[2] = (amt * new_range[2]) + lrange
  tne = [x.replace(tag=1) for x in ne]

  # replace ranges in other parts of the graph
  for x,y in zip(old_range, new_range): ctx[0][x] = y

  # apply the swizzled ranges to the srcs
  srcs = [s.substitute(dict(zip(old_range, new_range))).substitute(dict(zip(ne, tne))) for s in (in0, in1)]
  srcs = [x.substitute(dict(zip(tne, [ne[i] for i in p]))) for x,p in zip(srcs, tc.permutes_for_shape_str(tc.base_shape_str()))]

  ned = dict(zip(tc.base_shape_str(), ne))
  tc_reduce_axes = tuple([ned[f"r{i}"].arg[0] for i in range(len(tc.get_reduce_axes()))])
  base_upcast_axes = tuple([(ned[s].arg[0], 2) for s in tc.base_upcast_axes()])
  tc_upcast_axes = tuple([base_upcast_axes[:int(math.log2(tc.elements_per_thread[i]))] for i in range(3)])

  # construct the op
  # TODO: remove tc_upcast_axes from the arg
  wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, ctx[1].device, tc.threads, tc_upcast_axes, tc_reduce_axes)
  wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(tc.elements_per_thread[2]), src=(
    UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(tc.elements_per_thread[0]), src=(srcs[0],), arg=tc_upcast_axes[0]),
    UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(tc.elements_per_thread[1]), src=(srcs[1],), arg=tc_upcast_axes[1]),
    UOp.const(tc.dtype_out.vec(tc.elements_per_thread[2]), 0.0)), arg=wmma_arg)
  tc_uop = UOp(Ops.UNROLL, tc.dtype_out, (wmma,), arg=tc_upcast_axes[2])
  ret = tc_uop.reduce(new_reduce_range, arg=Ops.ADD)
  # confirm the UNROLLs aren't actually used, these need to be broadcast MUL
  assert all(u not in red_ranges for u in ret.toposort()), "UNROLLs in TC"
  return ret

from tinygrad.codegen.opt.postrange import pm_flatten_range

pm_tensor_cores = PatternMatcher([
  ((UPat.var("in0")*UPat.var("in1")).reduce(UPat(Ops.RANGE, name="r_range"), name="reduceop", arg=Ops.ADD), apply_tensor_cores),

  # replace range
  #(UPat(Ops.RANGE, name="r"), lambda ctx,r: ctx[0].get(r, None)),
  (UPat(Ops.SINK, name="s"), lambda ctx,s: graph_rewrite(s.substitute(ctx[0]), pm_flatten_range, name="flatten")),
])