from tinygrad.helpers import all_same, prod, getenv, ALLREDUCE_CAST
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp, AxisType, graph_rewrite, broadcast_axes, _broadcast_shape
from tinygrad.uop.ops import sint, ssimplify, sint_to_uop

def factor_span(f:UOp) -> sint: return UOp.factor_span(f)
from tinygrad.dtype import dtypes
from tinygrad.schedule.allreduce import handle_allreduce

# ***** multi rewrite MSELECT/MSTACK *****

def _apply_shrink(marg, s:UOp, i:int) -> UOp:
  new_arg = [tuple([x.substitute({drng[0]:drng[0].const_like(i)}) if isinstance(x, UOp) and
                    (drng:=[r for r in x.ranges if r.arg[-1] is AxisType.DEVICE]) else x for x in ss]) for ss in marg]
  return s._mop(Ops.SHRINK, tuple(new_arg))

def mstack_early_shrink(ms:UOp, shrink:UOp):
  ret:list[UOp] = []
  for i, x in enumerate(ms.src):
    if x.op is Ops.COPY:
      ret.append(_apply_shrink(shrink.marg, x.src[0], i).copy_to_device(x.device))
    else:
      ret.append(_apply_shrink(shrink.marg, x, i).contiguous())
  return ms.replace(src=tuple(ret))

def lower_broadcast_copy(c:UOp, x:UOp):
  if not (isinstance(c.device, tuple) and isinstance(x.device, str)): return None
  if (sx:=x.simplify()).device is None and sx.base.op is Ops.CONST: return UOp(Ops.MSTACK, src=(sx,)*len(c.device))
  return UOp(Ops.MSTACK, src=tuple(x.copy_to_device(d) for d in c.device))

replace_allreduce = PatternMatcher([
  # BROADCAST: explicitly expand broadcast copies and combine with MSTACK
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"),)), lower_broadcast_copy),
  # COPY_TO_ONE: if copying from multidevice to one, MSELECT the first (TODO: a little from each?)
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"),)), lambda c,x:
    x.mselect(0).copy_to_device(c.device) if isinstance(c.device, str) and isinstance(x.device, tuple) else None),
  # MSELECT on MSTACK is replaced with nothing
  (UPat(Ops.MSELECT, src=(UPat(Ops.MSTACK, name="mstack"),), name="ms"), lambda mstack, ms: mstack.src[ms.arg]),
  # move shrink before MSTACK
  (UPat(Ops.SHRINK, src=(UPat(Ops.MSTACK, name="ms"),), allow_any_len=True, name="shrink"), mstack_early_shrink),
  # move MSELECT before movement ops
  (UPat(Ops.MSELECT, src=(UPat(GroupOp.Movement, src=(UPat.var("s"),), allow_any_len=True, name="v"),), name="ms"),
   lambda s,v,ms: v.replace(src=(s.mselect(ms.arg),)+v.src[1:])),
])

_early_allreduce = PatternMatcher([
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"),), name="red"), handle_allreduce),
])
if not getenv("LATE_ALLREDUCE", 1): replace_allreduce = _early_allreduce + replace_allreduce

# ***** factor algebra for UNSHARD layouts *****
# an UNSHARD's src = (value, one factor arg per axis). a factor arg is a UOp or a STACK of UOps (major -> minor),
# and every axis decomposes as an ordered list of factors f with span(f) = vmax+1. a factor that contains RANGEs is
# an OWNER factor (its value is the owning shard's coordinate on that factor); a plain const is a LOCAL factor.
# pos(idx coords) = sum over factors of coord_f * multiplier_f, multiplier_f = prod of the spans of all later factors.
# layouts: [rng, L] contiguous, [L, rng] strided, [L, rng, L] middle insert, [rng_i, rng_j, L] stacked owners.
# NOTE: layouts are pure mixed-radix interleavings: ownership of a position must be an exact digit decomposition
# (idx // w) % span. swizzled ownership (e.g. XOR-shuffled layouts) cannot be expressed and index resolution
# requires the idx's digits to symbolically cancel exactly, no vmin/vmax range analysis.

def is_owner(f:UOp) -> bool: return len(f.ranges) > 0
def owners_of(fs:tuple[UOp, ...]) -> list[UOp]: return [f for f in fs if is_owner(f)]
def local_span(fs:tuple[UOp, ...]) -> sint: return prod([factor_span(f) for f in fs if not is_owner(f)])
def full_span(fs:tuple[UOp, ...]) -> sint: return prod([factor_span(f) for f in fs])

def layout_weights(fs:tuple[UOp, ...]) -> tuple[sint, ...]:
  ws = [1]
  for f in reversed(fs[1:]): ws.append(ssimplify(ws[-1]*factor_span(f)))
  return tuple(reversed(ws))

def _is_zero(x:UOp) -> bool: return (sx := sint_to_uop(x).ssimplify()) == 0 or (isinstance(sx, UOp) and sx.op is Ops.CONST and int(sx.arg) == 0)

def _check_unshard(val:UOp, args:tuple[UOp, ...]) -> None:
  # invariant: one factor arg per axis, and the LOCAL factor spans of every axis multiply to the shard's dims
  assert len(args) == len(val.shape) and all(prod([factor_span(f) for f in (a.src if a.op is Ops.STACK else (a,)) if not is_owner(f)]) == s
                                             for a, s in zip(args, val.shape)), f"UNSHARD layout {args} does not match shard shape {val.shape}"

def _unshard_with(val:UOp, multi:UOp, axes:tuple[int, ...]|None=None) -> UOp:
  # carry an UNSHARD's layout onto a (reshaped) shard value, optionally keeping only the given axes' factor args
  args = multi.src[1:] if axes is None else tuple(multi.src[1+ax] for ax in axes)
  _check_unshard(val, args)
  return UOp(Ops.UNSHARD, src=(val, *args))

def is_contig(multi:UOp) -> bool:
  # the device-multi canonical form: every sharded axis is [owner, optional local] (contiguous blocks)
  return all(all(not is_owner(f) for f in fs[1:]) for fs in multi.factors)

def normalize_factors(fs:tuple[UOp, ...]|list[UOp]) -> list[UOp]:
  out:list[UOp] = []
  for f in fs:
    if not is_owner(f) and int(factor_span(f)) == 1: continue  # span-1 locals carry no information
    if out and not is_owner(out[-1]) and not is_owner(f):
      out[-1] = UOp.local_factor(int(factor_span(out[-1])) * int(factor_span(f)))  # merge adjacent locals
    else: out.append(f)
  return out or [UOp.local_factor(1)]

def resolve_axis_index(idx:UOp, fs:tuple[UOp, ...]) -> UOp|None:
  """resolve the logical index idx on an axis with factor list fs into this shard's local index expression,
  or None if the index isn't owned by this shard. pos = sum(coord_f * w_f) with w_f = prod(spans after f), so the
  idx's digit (idx // w_f) % span_f must be exactly the owner coordinate at every owner factor, and the digits at
  the local factors assemble the local index row-major.

  NOTE: digits are computed from the original idx — no progressive subtraction, which would require the symbolic
  to cancel like-terms for arbitrary interleavings."""
  digits:list[tuple[UOp, sint]] = []
  for f, w in zip(fs, layout_weights(fs)):
    digit = (idx // w) % factor_span(f)
    if is_owner(f):
      if not _is_zero(digit - f): return None
    else: digits.append((digit, factor_span(f)))
  local = UOp.const(dtypes.weakint, 0)
  for d, k in digits: local = (local * k + d).ssimplify()
  return local

def index_multi(root:UOp, multi:UOp):
  idxs:list[UOp] = []
  for ax, idx in enumerate(root.src[1:]):
    fs = multi.factors[ax]
    if not owners_of(fs): idxs.append(idx)  # no owners on this axis: the shard index is the full index
    elif (local := resolve_axis_index(idx, fs)) is None:
      raise RuntimeError(f"index_multi: cannot shard index {idx} for UNSHARD axis {ax} with factors {fs}")
    else: idxs.append(local)
  return multi.src[0].index(*idxs)

def factor_subview(full:UOp, multi:UOp, reshape_to:tuple[sint, ...]|None=None) -> UOp:
  """the sub-view of an unsharded full-shape value `full` (shape == multi.shape) that belongs to this shard's
  owner: factor every axis by multi's factor lists, fix every owner factor at its coordinate."""
  new_shape, marg = [], []
  for fs in multi.factors:
    for f in fs:
      k = factor_span(f)
      new_shape.append(k)
      marg.append((f, f+1) if is_owner(f) else (0, k))
  view = full.reshape(tuple(new_shape)).shrink(tuple(marg))
  return view if reshape_to is None else view.reshape(reshape_to)

def store_value_multi(dest:UOp, multi:UOp):
  # storing a sharded value into an unsharded dest: every shard stores into its own sub-view of the dest
  val_shape = tuple(t for fs in multi.factors for f in fs for t in ([1] if is_owner(f) else [factor_span(f)]))
  dest, val = factor_subview(dest, multi), multi.src[0].reshape(val_shape)
  assert tuple(dest.shape) == tuple(val.shape), f"store sub-view shape mismatch {dest.shape} != {val.shape}"
  return dest.store(val)

# ***** multi functions *****

def shard_srcs(msrcs:tuple[UOp, ...], axis:int) -> list[UOp]:
  # normalize srcs to local shards on axis (single-axis contiguous resharding: the device-multi reshard)
  devices = [x.device for x in msrcs if x.device is not None]
  assert all_same(devices), f"all buffers must have the same device {devices}"
  if not len(devices):
    rng = next((r for m in msrcs if m.op is Ops.UNSHARD for fs in m.factors for e in fs for r in e.ranges), None)
    assert rng is not None, "shard_srcs requires a device or a sharding range"
  else: rng = UOp.range(len(devices[0]), -1, AxisType.DEVICE)

  out_shape = _broadcast_shape(*[x.shape for x in msrcs])
  srcs:list[UOp] = []
  for mlb in msrcs:
    src_axis = axis - (len(out_shape)-len(mlb.shape))
    if mlb.op is Ops.UNSHARD and mlb.axis == src_axis:
      # same axis, just copy through
      srcs.append(mlb.src[0])
    else:
      # otherwise every shard gets the full copy, sharded iff this src has the axis (broadcast srcs stay whole)
      full = mlb if not mlb.sharding else copy_multi(mlb, mlb.device)
      srcs.append(full if axis in broadcast_axes(mlb.shape, out_shape) else full._shard(src_axis, rng))
  return srcs

def alu_multi(root:UOp):
  multis = [m for m in root.src if m.op is Ops.UNSHARD]
  if not multis: return None
  if not root.sharding: return None
  target = multis[0]
  key = (tuple(target.src[0].shape), tuple(target.src[1:]))
  def same_layout(m:UOp) -> bool:
    return m.op is Ops.UNSHARD and m.shape == root.shape and (tuple(m.src[0].shape), tuple(m.src[1:])) == key
  def can_handle(m:UOp) -> bool:
    # target layout, a whole (unsharded) same-shape value (takes a per-shard sub-view), or a broadcast scalar
    if same_layout(m): return True
    if m.sharding: return False
    return len(m.shape) == 0 or tuple(m.shape) == tuple(root.shape)
  if all(can_handle(m) for m in root.src):
    # every src either has the target layout (peel the UNSHARD) or is whole on every shard: run the alu per-shard
    srcs:list[UOp] = []
    for m in root.src:
      if same_layout(m): srcs.append(m.src[0])
      else: srcs.append(m if len(m.shape) == 0 else factor_subview(m, target, reshape_to=target.src[0].shape))
    return _unshard_with(srcs[0].alu(root.op, *srcs[1:]), target)
  # mismatched layouts: reshard everything to the last sharded axis (device-multi only, layouts must be canonical)
  if not all(is_contig(m) for m in multis): raise RuntimeError(f"cannot reshard layouts in ALU: {multis}")
  axis = root.sharding[-1][0]
  srcs = shard_srcs(root.src, axis)
  return srcs[0].alu(root.op, *srcs[1:]).unshard(axis, next(m.sharding[0][1] for m in multis))

def reduce_multi(root:UOp, multi:UOp):
  op, num_axes = root.arg
  reduced = [f for ax in range(num_axes) for f in owners_of(multi.factors[ax])]
  remaining = tuple(ax for ax in range(num_axes, len(multi.factors)) if owners_of(multi.factors[ax]))
  local = multi.src[0]._rop(op, tuple(range(num_axes)))
  if reduced:
    assert not remaining, f"partial allreduce not supported for multi-axis sharding {multi.sharding}"
    # all sharded axes are reduced: full allreduce
    if ALLREDUCE_CAST and multi.src[0].op is Ops.CAST and multi.src[0].src[0].dtype in (dtypes.bfloat16, dtypes.half):
      orig_dtype = multi.src[0].src[0].dtype
      return local.cast(orig_dtype).allreduce(op, multi.device).cast(local.dtype)
    return local.allreduce(op, multi.device)
  # no sharded axes reduced: piecewise, keep all remaining layouts
  return _unshard_with(local, multi, tuple(range(num_axes, len(multi.factors))))

def reshape_layout(multi:UOp, new_shape:tuple[sint, ...]) -> list[list[UOp]]:
  """the factor algebra of a reshape: reshape never reorders, it splits and merges the ordered factor lists of
  multi onto the new axes (owner factors split as e//div, e%div, local factors split by size)."""
  # span-1 local factors carry no items and no information, they evaporate on reshape
  factors = [f for fs in multi.factors for f in fs if is_owner(f) or int(factor_span(f)) != 1]
  new_factors:list[list[UOp]] = []
  fi = 0
  for n in new_shape:
    need, axf = int(n), []
    while need != 1:
      if fi >= len(factors): raise RuntimeError(f"reshape {multi.shape} -> {new_shape} moved items between shards")
      f, k = factors[fi], int(factor_span(factors[fi]))
      if need % k == 0:
        axf.append(f)
        fi += 1
        need //= k
      # owner factors split as e//div (span need) for this axis and e%div (span div) for the next
      elif k % need == 0:
        div = k // need
        axf.append(UOp.local_factor(need) if not is_owner(f) else f // div)
        factors[fi] = UOp.local_factor(div) if not is_owner(f) else f % div
        need = 1
      else: raise RuntimeError(f"reshape {multi.shape} -> {new_shape} moved items between shards")
    new_factors.append(normalize_factors(axf))
  if fi < len(factors): raise RuntimeError(f"reshape {multi.shape} -> {new_shape} moved items between shards")
  return new_factors

def reshaped_shard_shape(multi:UOp, new_shape:tuple[sint, ...]) -> tuple[sint, ...]:
  """the shard shape of RESHAPE(multi, new_shape): the per-axis products of the new local spans."""
  return tuple(prod([factor_span(f) for f in fs if not is_owner(f)]) for fs in reshape_layout(multi, new_shape))

def reshape_multi(root:UOp, multi:UOp):
  if prod(multi.shape) != prod(new_shape:=root.marg):
    raise RuntimeError("reshape must maintain prod(shape)")
  shard = multi.src[0].reshape(reshaped_shard_shape(multi, new_shape))
  new_args = [UOp.factor_arg(fs) for fs in reshape_layout(multi, new_shape)]
  if tuple(new_args) == tuple(multi.src[1:]) and shard is multi.src[0]: return multi
  return UOp(Ops.UNSHARD, src=(shard, *new_args))

def expand_multi(root:UOp, multi:UOp):
  new_args = [UOp.factor_arg((UOp.local_factor(int(s)),)) for s in root.marg] + list(multi.src[1:])
  return UOp(Ops.UNSHARD, src=(multi.src[0]._mop(Ops.EXPAND, arg=root.marg), *new_args))

def pad_multi(root:UOp, multi:UOp):
  local_pad, new_args = [], []
  for ax, fs in enumerate(multi.factors):
    if owners_of(fs):
      assert root.marg[ax] == (0, multi.shape[ax]), f"padding not supported for {root.marg=}"
      local_pad.append((0, multi.src[0].shape[ax]))
      new_args.append(multi.src[1+ax])
    else:
      # PAD marg is (offset, padded size): the local factor's span tracks the padded shard dim
      assert len(fs) == 1, f"cannot pad a multi-local axis {fs=}"
      local_pad.append(root.marg[ax])
      new_args.append(UOp.factor_arg((UOp.local_factor(root.marg[ax][1]),)))
  return UOp(Ops.UNSHARD, src=(multi.src[0]._mop(Ops.PAD, tuple(local_pad)), *new_args))

def permute_multi(root:UOp, multi:UOp):
  # all permutes supported: the factor args just follow their axis
  return UOp(Ops.UNSHARD, src=(multi.src[0].permute(root.marg), *[multi.src[1+root.marg[ax]] for ax in range(len(root.marg))]))

def shrink_multi(root:UOp, multi:UOp):
  # a shrink resolves an owner factor when it selects exactly this owner's block (which is contiguous iff every
  # owner factor of the axis is more major than every local factor, i.e. the JAX-style [owners..., locals...] form)
  local_marg, new_args = [], []
  for ax, (fs, (s, l)) in enumerate(zip(multi.factors, root.marg)):
    span, lspan = full_span(fs), local_span(fs)
    if _is_zero(sint_to_uop(s)) and sint_to_uop(l).ssimplify() == span:
      local_marg.append((0, lspan))
      new_args.append(fs)  # full axis: keep the layout
      continue
    own = owners_of(fs)
    if not own:
      # unsharded axis: the interval maps straight onto the shard's local span, and the local factor shrinks with it
      assert len(fs) == 1, f"cannot partially shrink a multi-local axis {fs=}"
      local_marg.append((s, l))
      new_args.append((UOp.local_factor(l),))
      continue
    if own and all(not is_owner(f) for f in fs[len(own):]) and _is_zero(sint_to_uop(s) - sum(f*w for f, w in zip(own, layout_weights(fs)))) \
       and sint_to_uop(l).ssimplify() == lspan:
      # own-block resolve: this axis's sharding disappears, keep only the locals
      local_marg.append((0, lspan))
      new_args.append(tuple(f for f in fs if not is_owner(f)))
      continue
    # device-multi path: selecting a single contiguous partition (copied to all devices and optimized out later)
    if len(own) != 1 or len(fs) > 2 or not isinstance(multi.device, tuple) or len(multi.sharding) != 1:
      raise RuntimeError(f"shrinking not supported for {fs=} with {s=} {l=}")
    count = int(own[0].vmax)+1
    part_bounds = tuple((i*lspan, lspan) for i in range(count))
    if (s, l) not in part_bounds: raise RuntimeError(f"shrinking not supported for {fs=} with {s=} {l=}")
    non_shard_shrink = tuple((0, local_span(f2)) if i == ax else t for i, (f2, t) in enumerate(zip(multi.factors, root.marg)))
    return multi.src[0].copy_to_device(multi.device, arg=part_bounds.index((s, l)))._mop(Ops.SHRINK, non_shard_shrink)
  val = multi.src[0]._mop(Ops.SHRINK, tuple(local_marg))
  if all(tuple(f) == tuple(arg) for f, arg in zip(new_args, multi.factors)): return val
  return UOp(Ops.UNSHARD, src=(val, *[UOp.factor_arg(normalize_factors(fs)) for fs in new_args]))

def flip_multi(root:UOp, multi:UOp):
  for ax, fs in enumerate(multi.factors):
    if owners_of(fs) and root.marg[ax]: raise RuntimeError(f"flipping not supported on sharded axis {ax}")
  return _unshard_with(multi.src[0].flip([i for i,x in enumerate(root.marg) if x]), multi)

def stack_multi(root:UOp):
  # STACK adds a leading axis: srcs are sharded one axis below the output
  multis = [m for m in root.src if m.op is Ops.UNSHARD]
  if not multis: return None
  target = multis[0]
  if all(m.op is Ops.UNSHARD and m.shape == target.shape and tuple(m.src[1:]) == tuple(target.src[1:]) for m in multis):
    srcs = [m.src[0] if m.op is Ops.UNSHARD else m for m in root.src]
    return UOp(Ops.UNSHARD, src=(UOp(Ops.STACK, src=tuple(srcs)), UOp.local_factor(len(srcs)), *target.src[1:]))
  # mismatched layouts: reshard everything to the target axis (device-multi only)
  axis = root.axis
  assert axis is not None
  return UOp(Ops.STACK, src=tuple(shard_srcs(root.src, axis-1))).unshard(axis, next(m.sharding[0][1] for m in root.src if m.op is Ops.UNSHARD))

def _shard_idx(rng:UOp, dev_idx:int) -> int:
  drngs = [r for r in rng.ranges if r.arg[-1] is AxisType.DEVICE]
  return 0 if not drngs else int(rng.substitute({drngs[0]: drngs[0].const_like(dev_idx)}).ssimplify())

def copy_multi(multi:UOp, device:str | tuple[str, ...]):
  assert is_contig(multi), f"copy_multi only supports contiguous layouts, got {multi.factors}"
  sharding = multi.sharding
  if isinstance(device, str):
    # reconstruct by concatenating along each axis from last to first
    piece_info: list[tuple[tuple, UOp]] = []
    for i in range(len(multi.device)):
      idxs = tuple(_shard_idx(r, i) for _, r in sharding)
      piece_info.append((idxs, multi.src[0].mselect(i).copy_to_device(device)))
    for j in range(len(sharding) - 1, -1, -1):
      ax, rng = sharding[j]
      groups: dict[tuple, list[tuple[int, UOp]]] = {}
      for idxs, p in piece_info:
        key = idxs[:j] + idxs[j+1:]
        groups.setdefault(key, []).append((idxs[j], p))
      piece_info = []
      for key in sorted(groups):
        grp = sorted(groups[key], key=lambda x: x[0])
        piece_info.append((key, grp[0][1].cat(*[x[1] for x in grp[1:]], dim=ax)))
    return piece_info[0][1]
  # multi-device target: unshard all axes and allreduce
  val = multi.src[0]
  for ax, rng in sharding:
    bsz = val.shape[ax]
    val = val.pad(tuple((0,0) if a != ax else (bsz*rng, bsz*int(rng.vmax) - bsz*rng) for a in range(len(val.shape))))
  return val.allreduce(Ops.ADD, device)

def store_after_multi(dest:UOp, src:UOp): return _unshard_with(dest.after(dest.store(src.src[0])), src)

def passthrough_multi(root:UOp, multi:UOp):
  new_src = (multi.src[0],)+tuple(x.src[0] if x.op is Ops.UNSHARD else x for x in root.src[1:])
  val = UOp(root.op, root.dtype, src=new_src, arg=root.arg)
  return _unshard_with(val, multi)

def rewrite_into_function(call:UOp):
  if call.arg.precompile: return None
  new_body = graph_rewrite(call.src[0], multi_pm, name="subcall")
  new_args = tuple(a.src[0] if a.op is Ops.UNSHARD else a for a in call.src[1:])
  # after multi resolution, TUPLE elements may be UNSHARD — strip UNSHARD from body, create per-shard FUNCTION, wrap each GETTUPLE in its own UNSHARD
  assert new_body.op is Ops.TUPLE
  if any(s.op is Ops.UNSHARD for s in new_body.src):
    shard_call = call.replace(src=(UOp.maketuple(*[s.src[0] if s.op is Ops.UNSHARD else s for s in new_body.src]),)+new_args)
    return UOp.maketuple(*[_unshard_with(shard_call.gettuple(i), s) if s.op is Ops.UNSHARD else shard_call.gettuple(i)
                           for i, s in enumerate(new_body.src)])
  return call.replace(src=(new_body,)+new_args)

def param_to_multi(p:UOp):
  if p.axis is None: return None
  return UOp.param(p.arg.slot, p.dtype, p.shard_shape, p.device, p.arg.vmin_vmax, p.arg.multiple_of, p.arg.name, p.arg.addrspace).unshard(p.axis)

# NOTE: this is the same pattern as unrolled ranges
multi_pm = PatternMatcher([
  (UPat(Ops.PARAM, name="p"), param_to_multi),
  (UPat(GroupOp.ALU, name="root", custom_early_reject=set([Ops.UNSHARD])), alu_multi),
  (UPat(Ops.REDUCE, src=(UPat(Ops.UNSHARD, name="multi"), ), name="root"), reduce_multi),
  (UPat(Ops.RESHAPE, src=(UPat(Ops.UNSHARD, name="multi"), UPat()), name="root"), reshape_multi),
  (UPat(Ops.EXPAND, src=(UPat(Ops.UNSHARD, name="multi"), UPat()), name="root"), expand_multi),
  (UPat(Ops.PAD, src=(UPat(Ops.UNSHARD, name="multi"), UPat(), UPat()), name="root"), pad_multi),
  (UPat(Ops.SHRINK, src=(UPat(Ops.UNSHARD, name="multi"), UPat(), UPat()), name="root"), shrink_multi),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.UNSHARD, name="multi"), ), name="root"), permute_multi),
  (UPat(Ops.FLIP, src=(UPat(Ops.UNSHARD, name="multi"), ), name="root"), flip_multi),
  (UPat(Ops.STACK, name="root", custom_early_reject=set([Ops.UNSHARD])), stack_multi),
  (UPat(Ops.INDEX, src=(UPat(Ops.UNSHARD, name="multi"),), name="root", allow_any_len=True), index_multi),
  (UPat(Ops.AFTER, src=(UPat(Ops.UNSHARD), UPat(Ops.STORE, src=(UPat(Ops.UNSHARD, name="dest"), UPat(Ops.UNSHARD, name="src"))))), store_after_multi),
  (UPat(Ops.COPY, src=(UPat(Ops.UNSHARD, name="multi"),), name="copy"), lambda multi,copy: copy_multi(multi, copy.arg)),
  (UPat(Ops.ALLREDUCE, src=(UPat(Ops.UNSHARD, name="multi"),), name="red"),
    lambda multi,red: _unshard_with(multi.src[0].allreduce(*red.arg), multi)),

  # resolve TUPLE+GETTUPLE (needed in multi)
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),
  # GETTUPLE on UNSHARD: passthrough UNSHARD (e.g. when FUNCTION was replaced by UNSHARD(GETTUPLE(...)))
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.UNSHARD, name="multi"),), name="g"),
    lambda g, multi: _unshard_with(multi.src[0].gettuple(g.arg), multi) if multi.src[0].op in {Ops.FUNCTION, Ops.TUPLE} else multi),
  # rewrite into FUNCTION calls explicitly for UNSHARD (value-producing)
  (UPat(Ops.FUNCTION, name="call"), rewrite_into_function),
  (UPat((Ops.CALL, Ops.FUNCTION, Ops.AFTER), src=(UPat(Ops.UNSHARD, name="multi"), ), name="root", allow_any_len=True), passthrough_multi),
  # just strip the UNSHARD from non-value-producing CALLs (custom kernels, etc.) — FUNCTION is handled by rewrite_into_function
  (UPat(Ops.CALL, dtype=dtypes.void, name="root", custom_early_reject=set([Ops.UNSHARD])), lambda root:
    UOp(root.op, root.dtype, tuple(x.src[0] if x.op is Ops.UNSHARD else x for x in root.src), root.arg)),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD),
        src=(UPat(Ops.UNSHARD, name="multi"), ), name="root"), passthrough_multi),
  # STORE of a sharded value into an unsharded dest
  (UPat(Ops.STORE, src=(UPat.var("dest"), UPat(Ops.UNSHARD, name="multi"))), store_value_multi),
  # remove UNSHARD from STORE
  (UPat(Ops.STORE, src=(UPat(Ops.UNSHARD, name="multi"), ), name="root", allow_any_len=True),
    lambda root,multi: UOp(root.op, root.dtype, (multi.src[0],)+tuple(x.src[0] if x.op is Ops.UNSHARD else x for x in root.src[1:]), root.arg)),
  # every UNSHARD that survives multi_pm must satisfy the layout invariant (rangeify-mangled ones are resolved above)
  (UPat(Ops.UNSHARD, name="multi"), lambda multi: _check_unshard(multi.src[0], multi.src[1:])),
])+replace_allreduce
