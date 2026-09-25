from tinygrad.helpers import all_same, getenv, ALLREDUCE_CAST
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp, AxisType, graph_rewrite, broadcast_axes, _broadcast_shape, sint_to_uop
from tinygrad.dtype import dtypes
from tinygrad.schedule.allreduce import handle_allreduce

# ***** multi rewrite MSELECT/MSTACK *****

def _apply_shrink(marg, s:UOp, i:int) -> UOp:
  new_arg = [tuple([x.substitute({drng[0]:drng[0].const_like(i)}) if isinstance(x, UOp) and
                    (drng:=[r for r in x.ranges if r.axis_type is AxisType.DEVICE]) else x for x in ss]) for ss in marg]
  return s._mop(Ops.SHRINK, tuple(new_arg))

def mstack_early_shrink(ms:UOp, shrink:UOp):
  ret:list[UOp] = []
  for i, x in enumerate(ms.src):
    if x.op is Ops.COPY:
      src = _apply_shrink(shrink.marg, x.src[0], i)
      ret.append(src.contiguous() if src.device == x.device else src.copy_to_device(x.device))
    else:
      ret.append(_apply_shrink(shrink.marg, x, i).contiguous())
  return ms.replace(src=tuple(ret))

def lower_broadcast_copy(c:UOp, x:UOp):
  if not (isinstance(c.device, tuple) and isinstance(x.device, str)): return None
  if (sx:=x.simplify()).device is None: return UOp(Ops.MSTACK, src=(sx,)*len(c.device))
  return UOp(Ops.MSTACK, src=tuple(x.copy_to_device(d) for d in c.device))

replace_allreduce = PatternMatcher([
  # BROADCAST: explicitly expand broadcast copies and combine with MSTACK
  (UPat(Ops.COPY, name="c", src=(UPat(name="x"),)), lower_broadcast_copy),
  # COPY_TO_ONE: if copying from multidevice to one, MSELECT the first (TODO: a little from each?)
  (UPat(Ops.COPY, name="c", src=(UPat(name="x"),)), lambda c,x:
    (m if (m:=x.mselect(0)).device == c.device else m.copy_to_device(c.device))
    if isinstance(c.device, str) and isinstance(x.device, tuple) else None),
  # MSELECT on MSTACK is replaced with nothing
  (UPat(Ops.MSELECT, src=(UPat(Ops.MSTACK, name="mstack"),), name="ms"), lambda mstack, ms: mstack.src[ms.arg]),
  # move shrink before MSTACK
  (UPat(Ops.SHRINK, src=(UPat(Ops.MSTACK, name="ms"),), allow_any_len=True, name="shrink"), mstack_early_shrink),
  # move MSELECT before movement/ALU ops
  (UPat(Ops.MSELECT, src=(UPat(GroupOp.Movement, src=(UPat.var("s"),), allow_any_len=True, name="v"),), name="ms"),
   lambda s,v,ms: v.replace(src=(s.mselect(ms.arg),)+v.src[1:])),
  (UPat(Ops.MSELECT, src=(UPat(GroupOp.ALU, name="a"),), name="ms"), lambda a,ms:
   a.replace(src=tuple(s.mselect(ms.arg) if isinstance(s.device, tuple) else s for s in a.src))),
])

_early_allreduce = PatternMatcher([
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"),), name="red"), handle_allreduce),
])
if not getenv("LATE_ALLREDUCE", 1): replace_allreduce = _early_allreduce + replace_allreduce

# ***** multi functions *****

def unshard_like(local:UOp, multi:UOp) -> UOp:
  return local.unshard(tuple(a for a,_ in multi.sharding), tuple(r for _,r in multi.sharding))

def shard_srcs(msrcs:tuple[UOp, ...], axis:int) -> list[UOp]:
  # normalize srcs to local shards on axis
  devices = [x.device for x in msrcs if x.device is not None]
  assert all_same(devices), f"all buffers must have the same device {devices}"
  # without devices the sharding range comes from the UNSHARD itself (e.g. a LOCAL thread range);
  # device shards range over the devices instead
  if len(devices): sharding_rng = UOp.range(len(devices[0]), -1, AxisType.DEVICE)
  else:
    sharding_rng = next((m.sharding[0][1] for m in msrcs if m.sharding), None)
    assert sharding_rng is not None, "shard_srcs requires a device or a sharding range"

  out_shape = _broadcast_shape(*[x.shape for x in msrcs])
  srcs:list[UOp] = []
  for mlb in msrcs:
    src_axis = axis - (len(out_shape)-len(mlb.shape))
    if mlb.axis == src_axis:
      # same axis, just copy through
      srcs.append(mlb.shard_view)
    else:
      # otherwise every shard gets the full copy, sharded iff this src has the axis (broadcast srcs stay whole)
      full = mlb if mlb.axis is None else copy_multi(mlb, mlb.device)
      srcs.append(full if axis in broadcast_axes(mlb.shape, out_shape) else full._shard(src_axis, sharding_rng))
  return srcs

def shard_subview(full:UOp, multi:UOp) -> UOp:
  """the sub-view of an unsharded full-shape value (shape == multi.shape) that belongs to this shard:
  _shard along every sharded axis (contiguous blocks, like the device path)."""
  assert tuple(full.shape) == tuple(multi.shape), f"shard sub-view shape mismatch {full.shape} != {multi.shape}"
  # an EXPAND of a scalar over the full shape is the same broadcast on every shard: re-expand over the shard shape
  if full.op is Ops.EXPAND and full.src[0].shape == (): return full.src[0].expand(multi.shard_view.shape)
  for ax, rng in multi.sharding: full = full._shard(ax, rng)
  return full

def alu_multi(root:UOp):
  multis = [m for m in root.src if m.sharding]
  if not multis: return None
  sharding = multis[0].sharding
  target = multis[0]
  def can_handle(m:UOp) -> bool:
    # same sharding (peel the UNSHARD), or a whole unsharded value of the full tile shape (takes its per-shard
    # sub-view), or a broadcast scalar
    if m.sharding: return m.sharding == sharding
    return m.shape == () or tuple(m.shape) == tuple(target.shape)
  if all(can_handle(m) for m in root.src):
    # every src either has the target sharding or is whole on every shard: run the alu per-shard
    srcs = [m.shard_view if m.sharding else m if m.shape == () else shard_subview(m, target) for m in root.src]
    return unshard_like(srcs[0].alu(root.op, *srcs[1:]), target)
  # resharding: single-axis fallback via shard_srcs
  axis = root.axis
  assert axis is not None
  srcs = shard_srcs(root.src, axis)
  return srcs[0].alu(root.op, *srcs[1:]).unshard(axis, target.sharding[0][1])

def reduce_multi(root:UOp, multi:UOp):
  if not multi.sharding: return None
  op, num_axes = root.arg
  sharding = multi.sharding
  reduced = [(ax, rng) for ax, rng in sharding if ax < num_axes]
  remaining = [(ax, rng) for ax, rng in sharding if ax >= num_axes]
  local = multi.shard_view._rop(op, tuple(range(num_axes)))
  if reduced:
    assert not remaining, f"partial allreduce not supported for multi-axis sharding {sharding}"
    # all sharded axes are reduced: full allreduce
    if ALLREDUCE_CAST and multi.shard_view.op is Ops.CAST and multi.shard_view.src[0].dtype in (dtypes.bfloat16, dtypes.half):
      orig_dtype = multi.shard_view.src[0].dtype
      return local.cast(orig_dtype).allreduce(op, multi.device).cast(local.dtype)
    return local.allreduce(op, multi.device)
  # no sharded axes reduced: piecewise, keep all remaining sharding
  new_axes = tuple(ax - num_axes for ax, _ in remaining)
  new_rngs = tuple(rng for _, rng in remaining)
  return local.unshard(new_axes, new_rngs)

def expand_multi(root:UOp, multi:UOp):
  if not multi.sharding: return None
  shift = len(root.marg)
  return multi.shard_view._mop(Ops.EXPAND, arg=root.marg) \
    .unshard(tuple(ax+shift for ax,_ in multi.sharding), tuple(r for _,r in multi.sharding))

def pad_multi(root:UOp, multi:UOp):
  if not multi.sharding: return None
  for ax, _ in multi.sharding:
    assert root.marg[ax] == (0, multi.shape[ax]), f"padding not supported for {root.marg=}"
  counts = {a for a,_ in multi.sharding}
  local_pad = tuple((0, multi.shard_view.shape[a]) if a in counts else s for a,s in enumerate(root.marg))
  return unshard_like(multi.shard_view._mop(Ops.PAD, local_pad), multi)

def shrink_multi(root:UOp, multi:UOp):
  if not multi.sharding: return None
  # resolve each sharded axis independently: a shrink to exactly this range's own shard resolves the UNSHARD along
  # that axis (e.g. a fragment indexed by its LOCAL thread range becomes that thread's REG shard, no copy needed)
  local_marg = list(root.marg)
  remaining = list(multi.sharding)
  for ax, rng in multi.sharding:
    shard_sz = multi.shard_view.shape[ax]
    s, l = root.marg[ax]  # SHRINK marg is (start, length)
    if sint_to_uop(l).ssimplify() == shard_sz and (sint_to_uop(s)-rng*shard_sz).ssimplify() == 0:
      local_marg[ax] = (0, shard_sz)
      remaining.remove((ax, rng))
      continue
    part_bounds = tuple((i*shard_sz, shard_sz) for i in range(int(rng.vmax)+1))
    if (s, l) == (0, multi.shape[ax]): local_marg[ax] = (0, shard_sz)  # full axis stays sharded, shrink the other axes locally
    else:
      # NOTE: otherwise a shrink on the shard axis is only allowed on the legacy device path, selecting a single
      # partition (which is copied to all the devices and optimized out later)
      if len(multi.sharding) != 1 or not isinstance(multi.device, tuple) or (s, l) not in part_bounds:
        raise RuntimeError(f"shrinking not supported for {root.marg=}")
      non_shard_shrink = tuple((0, shard_sz) if i == ax else t for i, t in enumerate(root.marg))
      return multi.shard_view.copy_to_device(multi.device, arg=part_bounds.index((s, l)))._mop(Ops.SHRINK, non_shard_shrink)
  val = multi.shard_view._mop(Ops.SHRINK, tuple(local_marg))
  return val if not remaining else val.unshard(tuple(a for a,_ in remaining), tuple(r for _,r in remaining))

def flip_multi(root:UOp, multi:UOp):
  if not multi.sharding: return None
  for ax, _ in multi.sharding:
    if root.marg[ax]: raise RuntimeError(f"flipping not supported on sharded axis {ax}")
  return unshard_like(multi.shard_view.flip([i for i,x in enumerate(root.marg) if x]), multi)

def stack_multi(root:UOp):
  # STACK adds a leading axis: srcs are sharded one axis below the output
  multis = [m for m in root.src if m.sharding]
  if not multis: return None
  sharding = multis[0].sharding
  if all(m.sharding == sharding for m in multis):
    srcs = [m.shard_view if m.sharding else m for m in root.src]
    new_sharding = tuple((ax+1, rng) for ax, rng in sharding)
    return UOp(Ops.STACK, src=tuple(srcs)).unshard(tuple(a for a,_ in new_sharding), tuple(r for _,r in new_sharding))
  # resharding: single-axis fallback
  axis = root.axis
  assert axis is not None
  return UOp(Ops.STACK, src=tuple(shard_srcs(root.src, axis-1))).unshard(axis, multis[0].sharding[0][1])

def index_multi(root:UOp, multi:UOp):
  # Resolve explicit layouts before matching the range indices, including strided fragment layouts.
  if multi.op in GroupOp.Movement and multi.base.op is Ops.UNSHARD:
    from tinygrad.schedule.prepare import _mop_index
    if (ret := _mop_index(multi, root)) is not None: return ret
  if not multi.sharding: return None
  # INDEX on UNSHARD: resolve each sharded axis into this range's own shard (contiguous ownership:
  # idx = rng*shard_sz + local, thread rng owns [rng*shard_sz, ...)). Strided ownership is an explicit
  # PERMUTE/RESHAPE layout and is resolved by _mop_index above.
  idxs = list(root.src[1:])
  remaining = []
  for ax, rng in multi.sharding:
    if ax >= len(idxs):
      remaining.append((ax-len(idxs), rng))
      continue
    shard_sz = multi.shard_view.shape[ax]
    local = (idxs[ax] - rng*shard_sz).simplify()
    if local.vmin >= 0 and local.vmax < shard_sz:
      idxs[ax] = local
      continue
    raise RuntimeError(f"index_multi: cannot shard index {idxs[ax]} for UNSHARD axis {ax} with shard size {shard_sz}")
  ret = multi.shard_view.index(*idxs)
  return ret if not remaining else ret.unshard(tuple(a for a,_ in remaining), tuple(r for _,r in remaining))

def _shard_idx(rng:UOp, dev_idx:int) -> int:
  drngs = [r for r in rng.ranges if r.axis_type is AxisType.DEVICE]
  return 0 if not drngs else int(rng.substitute({drngs[0]: drngs[0].const_like(dev_idx)}).ssimplify())

def copy_multi(multi:UOp, device:str | tuple[str, ...]):
  sharding = multi.sharding
  if isinstance(device, str):
    # reconstruct by concatenating along each axis from last to first
    piece_info: list[tuple[tuple, UOp]] = []
    for i in range(len(multi.device)):
      idxs = tuple(_shard_idx(r, i) for _, r in sharding)
      piece_info.append((idxs, multi.shard_view.mselect(i).copy_to_device(device)))
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
  val = multi.shard_view
  for ax, rng in sharding:
    bsz = val.shape[ax]
    val = val.pad(tuple((0,0) if a != ax else (bsz*rng, bsz*int(rng.vmax) - bsz*rng) for a in range(len(val.shape))))
  return val.allreduce(Ops.ADD, device)

def store_value_multi(dest:UOp, multi:UOp):
  if not multi.sharding or dest.sharding: return None
  # storing a sharded value into an unsharded dest: every shard stores into its own sub-view of the dest
  return shard_subview(dest, multi).store(multi.shard_view)

def store_dest_multi(root:UOp, multi:UOp):
  if not multi.sharding: return None
  # STORE with a sharded dest: every shard stores into its own shard of the dest.
  # the value is handled like in alu_multi: UNSHARD srcs peel, full-shape values take their per-shard sub-view
  # (scalars arrive EXPANDed to the full shape by UOp.store's const_like, so they sub-view like everything else)
  srcs = [multi.shard_view] + [x.shard_view if x.sharding else shard_subview(x, multi) if tuple(x.shape) == tuple(multi.shape) else x
                              for x in root.src[1:]]
  return UOp(root.op, src=tuple(srcs), arg=root.arg)

def passthrough_multi(root:UOp, multi:UOp):
  if not multi.sharding: return None
  new_src = (multi.shard_view,)+tuple(x.shard_view if x.sharding else x for x in root.src[1:])
  return unshard_like(UOp(root.op, src=new_src, arg=root.arg), multi)

def rewrite_into_function(call:UOp):
  if not call.is_inline_call: return None
  # the call body is a plain parametric program: multi rewrites it like anything else (the output PARAM dests sub-view per
  # shard through the normal store rules), and all srcs (args and RETURNEDs) become their per-shard views
  new_body = graph_rewrite(call.body, multi_pm, name="subcall")
  assert new_body.op is Ops.SINK
  return call.replace(src=(new_body,) + tuple(a.shard_view if a.sharding else a for a in call.src[1:]))

# PERMUTE and RESHAPE carry the layout of an UNSHARD view.
multi_pat = UPat((Ops.UNSHARD, Ops.PERMUTE, Ops.RESHAPE), name="multi")

# NOTE: this is the same pattern as unrolled ranges
multi_pm = PatternMatcher([
  (UPat(GroupOp.ALU, name="root"), alu_multi),
  (UPat(Ops.REDUCE, src=(multi_pat, ), name="root"), reduce_multi),
  (UPat(Ops.EXPAND, src=(multi_pat, UPat()), name="root"), expand_multi),
  (UPat(Ops.PAD, src=(multi_pat, UPat(), UPat()), name="root"), pad_multi),
  (UPat(Ops.SHRINK, src=(multi_pat, UPat(), UPat()), name="root"), shrink_multi),
  (UPat(Ops.FLIP, src=(multi_pat, ), name="root"), flip_multi),
  (UPat(Ops.STACK, name="root"), stack_multi),
  (UPat(Ops.INDEX, src=(multi_pat,), name="root", allow_any_len=True), index_multi),
  # a COPY of a sharded value copies every shard to the target device
  (UPat(Ops.COPY, src=(multi_pat,), name="copy"), lambda multi,copy: copy_multi(multi, copy.arg) if multi.sharding else None),
  (UPat(Ops.ALLREDUCE, src=(multi_pat,), name="red"),
    lambda multi,red: unshard_like(multi.shard_view.allreduce(*red.arg), multi) if multi.sharding else None),

  # rewrite value-producing calls explicitly for UNSHARD
  (UPat(Ops.CALL, name="call"), rewrite_into_function),
  (UPat((Ops.CALL, Ops.AFTER), src=(multi_pat, ), name="root", allow_any_len=True), passthrough_multi),
  # just strip the UNSHARD from non-value-producing CALLs (custom kernels, etc.) — value-producing CALLs are handled by rewrite_into_function
  (UPat(Ops.CALL, dtype=dtypes.void, name="root"), lambda root:
    UOp(root.op, src=tuple(x.shard_view if x.sharding else x for x in root.src), arg=root.arg)),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.STAGE, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD),
        src=(multi_pat, ), name="root"), passthrough_multi),
  # STORE of a sharded value into an unsharded dest (e.g. a fragment into a full output tile)
  (UPat(Ops.STORE, src=(UPat.var("dest"), multi_pat)), store_value_multi),
  # STORE into a sharded dest (e.g. the fragment init): every shard stores into its own shard
  (UPat(Ops.STORE, src=(multi_pat, ), name="root", allow_any_len=True), store_dest_multi),
])+replace_allreduce
