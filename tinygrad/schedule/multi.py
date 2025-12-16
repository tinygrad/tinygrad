from typing import cast
import functools, itertools, operator
from tinygrad.helpers import all_same, all_int, prod, DEBUG, RING, getenv
from tinygrad.uop.ops import Ops, UOp, sint, PatternMatcher, UPat, GroupOp, graph_rewrite_map, graph_rewrite
from tinygrad.device import Device, Sharding

# *** allreduce implementation ***
def handle_allreduce_multirank(buf:UOp, red:UOp) -> UOp|None:
  if not isinstance(buf.device, (tuple, Sharding)): return None

  # Group buffers
  groups: dict[int|None, list[UOp]] = {}
  for i,dev in enumerate(buf.device):
    groups.setdefault(Device[dev].group_id, []).append(buf.mselect(i))

  # Put reduce leader of each group first
  reduce_leaders = set(getenv("REDUCE_LEADERS", "").split(","))
  groups = {gid: sorted(bufs, key=lambda x: (x.device not in reduce_leaders, x.device)) for gid,bufs in groups.items()}

  # Skip if only one group or if every group has only one buffer
  if len(groups) <= 1 or not any(len(g) > 1 for g in groups.values()): return None

  # Reduce inside each group
  inner = [UOp(Ops.MSTACK, buf.dtype, tuple(bufs)).allreduce(red.arg, (cast(str, bufs[0].device),)).mselect(0) for bufs in groups.values()]

  # Allreduce across groups
  outer = UOp(Ops.MSTACK, buf.dtype, tuple(inner)).allreduce(red.arg, tuple(buf.device for buf in inner))

  # Broadcast back to all devices in the group
  gid2bid = {Device[device].group_id: i for i,device in enumerate(outer.device)}
  return outer.mselect(gid2bid[Device[red.device].group_id]).copy_to_device(red.device) if not isinstance(red.device, (tuple, Sharding)) else \
         UOp(Ops.MSTACK, buf.dtype, tuple(outer.mselect(gid2bid[Device[device].group_id]).copy_to_device(device) for device in red.device))

def handle_allreduce(buf:UOp, red:UOp) -> UOp|None:
  if not isinstance(buf.device, (tuple, Sharding)): return None
  assert all_int(buf.shape), f"does not support symbolic shape {buf.shape}"
  n_lbs, shape, numel = len(buf.device), buf.shape, prod(buf.shape)
  # ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  # fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  use_ring = (RING >= 2 or (n_lbs > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  if DEBUG >= 2: print(f"{'RING ALLREDUCE' if use_ring else 'NAIVE ALLREDUCE'} {n_lbs}x{numel} | {buf.dtype}")

  # contiguous before we copy it
  buf = buf.contiguous()

  # copy to all devices. if you shrink later, that'll be handled
  if not use_ring: return functools.reduce(lambda x,y: x.alu(red.arg, y),
                                           [UOp(Ops.COPY, buf.dtype, (buf.mselect(i), red.src[1])) for i in range(len(buf.device))])

  # new ring reduce
  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = (numel // factor) // n_lbs, (numel // factor) % n_lbs
  chunk_sizes = [(base + 1) * factor] * left + [base * factor] * (n_lbs - left)
  chunks = list(itertools.pairwise(itertools.accumulate(chunk_sizes, initial=0)))

  # extract chunks and scatter-reduce
  reduced_chunks = []
  for i,(s,e) in enumerate(chunks):
    chunk = buf.reshape((numel,)).shrink(((s,e),))
    reduced_chunk = chunk
    for step in range(n_lbs-1):
      src, dest = (i+step)%n_lbs, (i+step+1)%n_lbs
      # copy the chunk from the src device to the dest (operating device), and select the chunk on the dest device
      reduced_chunk = reduced_chunk.copy_to_device(buf.device[dest], src if isinstance(reduced_chunk.device, (tuple, Sharding)) else None) \
        .alu(red.arg, chunk.copy_to_device(buf.device[dest], dest))
    reduced_chunks.append(reduced_chunk)

  # allgather
  copied_chunks = []
  for i,c in enumerate(reduced_chunks):
    this_chunk: list[UOp|None] = [None] * len(buf.device)
    this_chunk[(i+len(buf.device)-1)%n_lbs] = c
    for step in range(n_lbs-1):
      dest = (i+step)%n_lbs
      this_chunk[dest] = c = c.copy_to_device(buf.device[dest])
    copied_chunks.append(UOp(Ops.MSTACK, buf.dtype, tuple(cast(list[UOp], this_chunk))))

  # reassemble
  pads = [((s,numel-e),) for s,e in chunks]
  return functools.reduce(operator.add, [c.pad(pad) for pad,c in zip(pads, copied_chunks)]).reshape(shape)

# ***** multi rewrite MSELECT/MSTACK *****

def mstack_early_shrink(ms:UOp, shrink:UOp):
  ret:list[UOp] = []
  def apply_shrink(s:UOp, i:int) -> UOp:
    new_arg = [tuple([x.substitute({dvar[0]:dvar[0].const_like(i)}) if isinstance(x, UOp) and
                      (dvar:=[v for v in x.vars() if v.op is Ops.DEFINE_VAR and v.arg[0]=='_device_num']) else x for x in ss]) for ss in shrink.marg]
    return s.shrink(tuple(new_arg))
  for i, x in enumerate(ms.src):
    if x.op is Ops.COPY:
      # if src device doesn't have a renderer, we have to view after the copy
      # TODO: a way to understand this
      if x.src[0].device in {"DISK", "NPY"}:
        ret.append(apply_shrink(x, i))
      else:
        ret.append(apply_shrink(x.src[0], i).copy_to_device(x.device))
    else:
      ret.append(apply_shrink(x, i).contiguous())
  return ms.replace(src=tuple(ret))

replace_allreduce = PatternMatcher([
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"), UPat()), name="red"), handle_allreduce_multirank),
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"), UPat()), name="red"), handle_allreduce),
  (UPat(Ops.COPY, src=(UPat(Ops.BUFFER, name="buf"), UPat(Ops.DEVICE, name="dev"))),lambda buf,dev: UOp.new_buffer(dev.arg, buf.arg, buf.dtype)
   if buf.device not in {"DISK", "NPY"} and isinstance(dev.arg, (tuple, Sharding)) and isinstance(buf.device, str) else None),
  # BROADCAST: explicitly expand broadcast copies and combine with MSTACK
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"), UPat(Ops.DEVICE))), lambda c,x:
    UOp(Ops.MSTACK, c.dtype, tuple(x.copy_to_device(d) for d in c.device))
    if isinstance(c.device, (tuple, Sharding)) and isinstance(x.device, str) else None),
  # COPY_TO_ONE: if copying from multidevice to one, MSELECT the first (TODO: a little from each?)
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"), UPat(Ops.DEVICE))), lambda c,x:
    x.mselect(0).copy_to_device(c.device) if isinstance(c.device, str) and isinstance(x.device, (tuple, Sharding)) else None),
  # MSELECT on MSTACK is replaced with nothing
  (UPat(Ops.MSELECT, src=(UPat(Ops.MSTACK, name="mstack"),), name="ms"), lambda mstack, ms: mstack.src[ms.arg]),
  # move shrink before MSTACK
  (UPat(Ops.SHRINK, src=(UPat(Ops.MSTACK, name="ms"),), allow_any_len=True, name="shrink"), mstack_early_shrink),
  # move MSELECT before movement ops
  (UPat(Ops.MSELECT, src=(UPat(GroupOp.Movement, src=(UPat.var("s"),), allow_any_len=True, name="v"),), name="ms"),
   lambda s,v,ms: v.replace(src=(s.mselect(ms.arg),)+v.src[1:])),
])

# ***** multi functions *****

def _underlying_devices(device) -> tuple[str, ...]:
  """Extract underlying device tuple from device (tuple or Sharding)."""
  return device.devices if isinstance(device, Sharding) else device

def _get_multi_axis(mlb:UOp) -> int|None:
  """Get axis only for MULTI/MSTACK ops, ignore Sharding axis on regular ops."""
  return mlb.arg if mlb.op in (Ops.MULTI, Ops.MSTACK) else None

def _is_sharded(uop:UOp) -> bool:
  """Check if UOp has sharded device (Sharding or MULTI). MSTACK is handled separately by replace_allreduce."""
  if uop.op is Ops.MULTI: return True
  # NOTE: MSTACK is NOT included here - it has multiple single-device sources, handled by mstack_early_shrink
  dev = uop._device  # Use _device to avoid unwrap assertion
  return isinstance(dev, Sharding)

def _get_axis(uop:UOp) -> int|None:
  """Get sharding axis from either Sharding device or MULTI/MSTACK op."""
  dev = uop._device  # Use _device to avoid unwrap assertion
  if isinstance(dev, Sharding): return dev.axis
  if uop.op in (Ops.MULTI, Ops.MSTACK): return uop.arg
  return None

def _get_inner(uop:UOp) -> UOp:
  """Get inner op - for MULTI it's src[0], otherwise it's the op itself. MSTACK is not unwrapped here."""
  return uop.src[0] if uop.op is Ops.MULTI else uop

def _get_device(uop:UOp) -> tuple[str, ...]:
  """Get devices tuple from sharded UOp."""
  dev = uop._device  # Use _device to avoid unwrap assertion
  if isinstance(dev, Sharding): return dev.devices
  return cast(tuple, dev)

def alu_multi(root:UOp):
  msrcs = root.src
  # Check if any source is sharded
  sharded_srcs = [x for x in msrcs if _is_sharded(x)]
  if not sharded_srcs: return None
  # Only check that sharded sources have the same device - non-sharded sources will be broadcast
  sharded_devs = [_underlying_devices(x.device) for x in sharded_srcs]
  assert all_same(sharded_devs), f"all sharded buffers must have the same device {[x.device for x in msrcs]}"
  # Get the target devices from a sharded source
  target_devices = sharded_devs[0]
  axis = root.axis
  assert axis is not None

  srcs = []
  for mlb in msrcs:
    mlb_axis = _get_axis(mlb)
    if mlb_axis == axis:
      # same axis, extract inner op
      srcs.append(_get_inner(mlb))
    elif mlb_axis is None:
      # not sharded, copy to target devices first, then shard
      if not isinstance(mlb.device, (tuple, Sharding)):
        mlb = mlb.copy_to_device(target_devices)
      srcs.append(mlb._shard(axis))
    else:
      # axis mismatch, unshard and reshard
      srcs.append(_get_inner(mlb)._unshard(mlb_axis).allreduce(Ops.ADD, mlb.device)._shard(axis))
  return srcs[0].alu(root.op, *srcs[1:]).multi(axis)

def reduce_multi(root:UOp, src:UOp):
  # Check if source is sharded
  if not _is_sharded(src): return None
  op, axis = root.arg
  multi_axis = _get_axis(src)
  inner = _get_inner(src)
  device = _get_device(src)
  if multi_axis is not None and multi_axis in axis:
    # all-reduce on sharded axes
    return inner.r(op, axis).allreduce(op, device)
  # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
  return inner.r(op, axis).multi(axis=multi_axis)

def _shape_to_single_shard(axis, shape:tuple[sint, ...], lb:UOp) -> tuple[sint, ...]:
  return tuple(lb.shape[axis] if a == axis else s for a,s in enumerate(shape))

def reshape_multi(root:UOp, src:UOp):
  if not _is_sharded(src): return None
  arg = root.marg
  multi_axis = _get_axis(src)
  inner = _get_inner(src)
  if (new_axis:=root.axis) is None: return inner.reshape(arg).multi(new_axis)
  assert prod(src.shape) == prod(arg), "reshape must maintain prod(shape)"
  assert prod(inner.shape[multi_axis:])%prod(arg[new_axis+1:]) == 0, f"reshape cannot move items between shards {src.shape} -> {arg=}"
  new_shape_axis = prod(inner.shape[multi_axis:]) // prod(arg[new_axis+1:])
  return inner.reshape(tuple(s if a!=new_axis else new_shape_axis for a,s in enumerate(arg))).multi(new_axis)

def expand_multi(root:UOp, src:UOp):
  if not _is_sharded(src): return None
  multi_axis = _get_axis(src)
  inner = _get_inner(src)
  # NOTE: this assert isn't needed, sharded axis can have dim 1
  assert multi_axis is None or root.marg[multi_axis] == src.shape[multi_axis], f"expand not supported on sharded axis {root.marg=}"
  return inner.expand(_shape_to_single_shard(multi_axis, root.marg, inner)).multi(multi_axis)

def pad_multi(root:UOp, src:UOp):
  if not _is_sharded(src): return None
  multi_axis = _get_axis(src)
  inner = _get_inner(src)
  assert multi_axis is None or root.marg[multi_axis] == (0,0), f"padding not supported for {root.marg=}"
  return inner.pad(root.marg).multi(multi_axis)

def permute_multi(root:UOp, src:UOp):
  if not _is_sharded(src): return None
  inner = _get_inner(src)
  return inner.permute(root.marg).multi(root.axis)

def shrink_multi(root:UOp, src:UOp):
  if not _is_sharded(src): return None
  multi_axis = _get_axis(src)
  inner = _get_inner(src)
  assert multi_axis is None or root.marg[multi_axis] == (0, src.shape[multi_axis]) or root.marg[multi_axis] in src.bounds, \
    f"shrinking not supported for {root.marg=}"
  if multi_axis is not None and root.marg[multi_axis] in src.bounds and root.marg[multi_axis] != (0, src.shape[multi_axis]):
    assert all(root.marg[i] == (0, s) or i == multi_axis for i,s in enumerate(src.shape)), \
      "cannot shrink sharded and non-sharded axis at the same time"
    # NOTE: shrink on the shard axis is only allowed when result is a single partition, denoted by the new real
    # we just copy it to all the devices, no real. this will be optimized out later
    return inner.copy_to_device(src.device, arg=src.bounds.index(root.marg[multi_axis]))
  return inner.shrink(tuple((0, inner.shape[multi_axis]) if a == multi_axis else s for a,s in enumerate(root.marg))).multi(multi_axis)

def flip_multi(root:UOp, src:UOp):
  if not _is_sharded(src): return None
  multi_axis = _get_axis(src)
  inner = _get_inner(src)
  assert multi_axis is None or not root.marg[multi_axis], "flipping not supported on sharded axis"
  return inner.flip([i for i,x in enumerate(root.marg) if x]).multi(multi_axis)

# from multiple devices -> one
def copy_multi(src:UOp, device:UOp):
  if not _is_sharded(src): return None
  multi_axis = _get_axis(src)
  inner = _get_inner(src)
  assert multi_axis is not None, "all multi ops have axis"
  return inner._unshard(multi_axis).allreduce(Ops.ADD, device)

def assign_multi(dest:UOp, src:UOp):
  if not _is_sharded(dest) or not _is_sharded(src): return None
  dest_axis = _get_axis(dest)
  src_axis = _get_axis(src)
  if dest_axis != src_axis: raise RuntimeError(f"axis must match in assign {dest_axis} != {src_axis}")
  return _get_inner(dest).assign(_get_inner(src)).multi(src_axis)

def passthrough_multi(root:UOp, src:UOp):
  if not _is_sharded(src): return None
  multi_axis = _get_axis(src)
  inner = _get_inner(src)
  return UOp(root.op, root.dtype, (inner,), root.arg).multi(multi_axis)

# NOTE: this is the same pattern as Ops.UNROLL
# Patterns match broadly, callbacks check _is_sharded() and return None if not applicable
multi_pm = PatternMatcher([
  (UPat(GroupOp.ALU, name="root"), alu_multi),
  (UPat(Ops.REDUCE_AXIS, src=(UPat(name="src"), ), name="root"), reduce_multi),
  (UPat(Ops.RESHAPE, src=(UPat(name="src"), UPat()), name="root"), reshape_multi),
  (UPat(Ops.EXPAND, src=(UPat(name="src"), UPat()), name="root"), expand_multi),
  (UPat(Ops.PAD, src=(UPat(name="src"), UPat(), UPat()), name="root"), pad_multi),
  (UPat(Ops.SHRINK, src=(UPat(name="src"), UPat(), UPat()), name="root"), shrink_multi),
  (UPat(Ops.PERMUTE, src=(UPat(name="src"), ), name="root"), permute_multi),
  (UPat(Ops.FLIP, src=(UPat(name="src"), ), name="root"), flip_multi),
  (UPat(Ops.ASSIGN, src=(UPat(name="dest"), UPat(name="src"))), assign_multi),
  (UPat(Ops.COPY, src=(UPat(name="src"), UPat(Ops.DEVICE, name="device"))), copy_multi),
  (UPat(Ops.ALLREDUCE, src=(UPat(name="src"), UPat(Ops.DEVICE, name="device")), name="red"),
    lambda src,device,red: _get_inner(src).allreduce(red.arg, device).multi(axis=_get_axis(src)) if _is_sharded(src) else None),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD),
        src=(UPat(name="src"), ), name="root"), passthrough_multi),
  # multi supports custom kernels with CUSTOM_KERNEL + AFTER
  (UPat(Ops.CUSTOM_KERNEL, src=UPat(name="ck_srcs"), allow_any_len=True, name="ck"),
    lambda ck,ck_srcs: ck.replace(src=tuple(_get_inner(m) for m in ck.src)) if any(_is_sharded(m) for m in ck.src) else None),
  (UPat(Ops.AFTER, src=(UPat(name="src"), UPat(Ops.CUSTOM_KERNEL)), name="a"),
    lambda src,a: a.replace(src=(_get_inner(src),)+a.src[1:]).multi(_get_axis(src)) if _is_sharded(src) else None)
])+replace_allreduce

def get_multi_map(big_sink:UOp) -> dict[UOp, UOp]:
  if getenv("VIZ"): graph_rewrite(big_sink, PatternMatcher([]), name="View Multi AST")
  ret = graph_rewrite_map(big_sink, multi_pm, name="multi_pm")
  if getenv("VIZ"): graph_rewrite(ret[big_sink], PatternMatcher([]), name="View Post Multi AST")
  return ret
