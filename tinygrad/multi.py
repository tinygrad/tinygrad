from __future__ import annotations
import functools, itertools, operator
from tinygrad.helpers import all_same, all_int, dedup, prod, DEBUG, RING, getenv
from tinygrad.ops import Ops, UOp, sint

def all_reduce(bop: Ops, lbs: list[UOp]) -> list[UOp]:
  assert all_int(lbs[0].shape), f"does not support symbolic shape {lbs[0].shape}"
  assert all_same([lb.shape[0] for lb in lbs]), "allreduce with uneven shards is undefined"
  n_lbs, shape, numel = len(lbs), lbs[0].shape, prod(lbs[0].shape)
  # ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  # fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  use_ring = (RING >= 2 or (n_lbs > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  if DEBUG >= 2: print(f"{'RING ALLREDUCE' if use_ring else 'NAIVE ALLREDUCE'} {n_lbs}x{numel} | {lbs[0].dtype}")
  if not use_ring: return [functools.reduce(lambda x,y: x.alu(bop, y), [x.copy_to_device(lb.device) for x in lbs]) for lb in lbs]

  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = (numel // factor) // n_lbs, (numel // factor) % n_lbs
  chunk_sizes = [(base + 1) * factor] * left + [base * factor] * (n_lbs - left)
  chunks = list(itertools.pairwise(itertools.accumulate(chunk_sizes, initial=0)))
  chunked = [[lb.reshape((numel,)).shrink(((s,e),)) for s,e in chunks] for lb in lbs]

  # scatter-reduce
  for step in range(n_lbs-1):
    for i in range(len(chunks)):
      src, dest = (i+step)%n_lbs, (i+step+1)%n_lbs
      chunked[dest][i] = chunked[dest][i].alu(bop, chunked[src][i].copy_to_device(chunked[dest][i].device))

  # allgather
  for step in range(n_lbs-1):
    for i in range(len(chunks)):
      src, dest = (i+step-1)%n_lbs, (i+step)%n_lbs
      chunked[dest][i] = chunked[src][i].copy_to_device(chunked[dest][i].device)

  # assemble chunks back
  pads = [((s,numel-e),) for s,e in chunks]
  return [functools.reduce(operator.add, [c.pad(pad) for pad,c in zip(pads,lb_c)]).reshape(shape) for lb_c in chunked]

def to_sharded(lbs:list[UOp], axis:int, bounds: tuple[tuple[int, int], ...]) -> list[UOp]:
  if lbs[0].shape[axis] % len(lbs) != 0: raise RuntimeError(f"multi axis uneven: {lbs[0].shape=} {axis=} {len(lbs)=}, bounds={bounds}")
  return [lb.shrink(tuple((0,s) if a != axis else bound for a,s in enumerate(lb.shape))) for i, (bound, lb) in enumerate(zip(bounds, lbs))]

# ***** multi functions *****

from tinygrad.ops import PatternMatcher, UPat, GroupOp, graph_rewrite_map, track_rewrites

def alu_multi(root:UOp):
  msrcs = root.src
  assert all(x.op is Ops.MULTI for x in msrcs), f"all buffers must be MultiLazyBuffer {[x.op for x in msrcs]}"
  assert all_same([x.device for x in msrcs]), f"all buffers must have the same device {[x.device for x in msrcs]}"

  # NOTE: they all have to share an axis, we always choose [-1]
  axis, bounds = axes[-1] if len(axes := dedup([(x.axis, x.bounds) for x in msrcs if x.axis is not None])) else (None, None)
  srcs:list[list[UOp]] = []
  not_all_real = not all(all(mlb.real) for mlb in msrcs)
  new_real = [all(transposed) for transposed in zip(*[mlb.real for mlb in msrcs])] if not_all_real else msrcs[0].real
  assert any(new_real), "output contains no real lb"
  for mlb in msrcs:
    if (mlb.axis == axis and (mlb.axis is None or mlb.bounds == bounds)) or not_all_real: srcs.append(mlb.src)
    else:
      assert axis is not None and bounds is not None
      if mlb.axis is None: srcs.append(to_sharded(mlb.src, axis, bounds))
      else: srcs.append(to_sharded([mlb.copy_to_device(lb.device) for lb in mlb.src], axis, bounds))
  new_real_lbs:dict[int,UOp] = {i:lsrcs[0].alu(root.op, *lsrcs[1:]) for i,(lsrcs,r) in enumerate(zip(zip(*srcs), new_real)) if r}
  # NOTE: const dtype should match real
  new_dtype = next(iter(new_real_lbs.values())).dtype
  new_lbs = [new_real_lbs.get(i, lsrcs[0].const_like(0).cast(new_dtype)) for i,lsrcs in enumerate(zip(*srcs))]
  return UOp.multi(*new_lbs, axis=axis, real=new_real)

def reduce_multi(root:UOp, multi:UOp):
  op, axis = root.arg
  if multi.axis is not None and multi.axis in axis:
    # all-reduce on sharded axes
    reduced_parts = [(x if r else x.const_like(0)).r(op, axis) for x,r in zip(multi.src, multi.real)]
    # if all partitions are real, do all_reduce
    if all(multi.real): return UOp.multi(*all_reduce(op, reduced_parts), axis=None)
    # only one partition is real, keep it
    return UOp.multi(*reduced_parts, axis=None, real=multi.real)
  # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
  return UOp.multi(*[x.r(op, axis) for x in multi.src], axis=multi.axis, real=multi.real)

def _shape_to_single_shard(axis, shape:tuple[sint, ...], lb:UOp) -> tuple[sint, ...]:
  return tuple(lb.shape[axis] if a == axis else s for a,s in enumerate(shape))

def reshape_multi(root:UOp, multi:UOp):
  arg = root.arg
  if multi.axis is None: return UOp.multi(*[x.reshape(arg) for x in multi.src], axis=None, real=multi.real)
  arg_acc:list[sint] = list(itertools.accumulate(arg, operator.mul, initial=1))
  # new_axis is the last one that preserves prod(prior to new_axis) and must not move items between shards
  # todo: what to do about shrinking to self.shape[self.axis]==1 len(self.real_lbs)==1?
  new_axis = len(arg_acc) - arg_acc[::-1].index(prod(multi.shape[:multi.axis])) - 1
  assert all(prod(lb.shape[multi.axis:])%prod(arg[new_axis+1:])==0 for lb in multi.src), f"reshape cannot move items between shards {root.arg=}"
  lbs = [x.reshape(tuple(s if a!=new_axis else prod(x.shape[multi.axis:])//prod(arg[new_axis+1:]) for a,s in enumerate(arg))) for x in multi.src]
  return UOp.multi(*lbs, axis=new_axis, real=multi.real)

def expand_multi(root:UOp, multi:UOp):
  # NOTE: this assert isn't needed, sharded axis can have dim 1
  assert multi.axis is None or root.arg[multi.axis] == multi.shape[multi.axis], f"expand not supported on sharded axis {root.arg=}"
  return UOp.multi(*[x.expand(_shape_to_single_shard(multi.axis, root.arg, x)) for x in multi.src], axis=multi.axis, real=multi.real)

def pad_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.arg[multi.axis] == (0,0) or not all(multi.real), f"padding not supported for {root.arg=}"
  # pad on shard axis -> fill others with zeros and set real to all True
  if multi.axis is not None and root.arg[multi.axis] != (0,0):
    # pad back to whole axis, remove real mask
    assert all(root.arg[i] == (0, 0) for i in range(len(multi.shape)) if i != multi.axis), "cannot pad sharded and non-sharded axis at the same time"
    dim, bound = sum(lb.shape[multi.axis] for lb in multi.src), multi.bounds[multi.real.index(True)]
    assert root.arg[multi.axis] == (bound[0], dim-bound[1]), "can only pad to whole axis"
    return UOp.multi(*[x if r else x.const_like(0) for x,r in zip(multi.src, multi.real)], axis=multi.axis)
  return UOp.multi(*[x.pad(root.arg) for x in multi.src], axis=multi.axis, real=multi.real)

def permute_multi(root:UOp, multi:UOp):
  # all permutes supported!
  return UOp.multi(*[x.permute(root.arg) for x in multi.src], axis=root.arg.index(multi.axis) if multi.axis is not None else None, real=multi.real)

def shrink_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.arg[multi.axis] == (0, multi.shape[multi.axis]) or root.arg[multi.axis] in multi.bounds, \
    f"shrinking not supported for {root.arg=}"
  if multi.axis is not None and root.arg[multi.axis] in multi.bounds and root.arg[multi.axis] != (0, multi.shape[multi.axis]):
    assert all(root.arg[i] == (0, s) or i == multi.axis for i,s in enumerate(multi.shape)), \
      "cannot shrink sharded and non-sharded axis at the same time"
    # NOTE: shrink on the shard axis is only allowed when result is a single partition, denoted by the new real
    idx = multi.bounds.index(root.arg[multi.axis])
    # zero out other lbs to not create lb reference
    return UOp.multi(*[lb if i==idx else lb.const_like(0) for i,lb in enumerate(multi.src)],
                      axis=multi.axis, real=[i==idx for i in range(len(multi.src))])
  return UOp.multi(*[x.shrink(tuple((0, x.shape[multi.axis]) if a == multi.axis else s for a,s in enumerate(root.arg))) for x in multi.src],
                   axis=multi.axis, real=multi.real)

def stride_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.arg[multi.axis] == 1, "flipping not supported on sharded axis"
  return UOp.multi(*[x.stride(root.arg) for x in multi.src], axis=multi.axis, real=multi.real)

def copy_multi(multi:UOp, device:UOp):
  # if we already have a copy on the device, return that
  if multi.axis is None: return next((lb for lb in multi.real_lbs if lb.device == device.arg), multi.real_lbs[0].copy_to_device(device.arg))
  # copy lbs to device, pad to final shape, and sum
  llbs:list[UOp] = []
  for lb,real,(start,end) in zip(multi.src, multi.real, multi.bounds):
    if not real: continue
    pad_arg = tuple((0,0) if a != multi.axis else (start, multi.bounds[-1][1]-end) for a in range(len(lb.shape)))
    llbs.append(lb.copy_to_device(device.arg).pad(pad_arg))
  return functools.reduce(operator.add, llbs)

def passthrough_multi(root:UOp, multi:UOp): return UOp.multi(*[root.replace(src=(m,)) for m in multi.src], axis=multi.axis, real=multi.real)

# NOTE: this is the same pattern as Ops.UNROLL
multi_pm = PatternMatcher([
  (UPat(GroupOp.ALU, name="root", custom_early_reject=set([Ops.MULTI])), alu_multi),
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), reduce_multi),
  (UPat(Ops.RESHAPE, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), reshape_multi),
  (UPat(Ops.EXPAND, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), expand_multi),
  (UPat(Ops.PAD, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), pad_multi),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), permute_multi),
  (UPat(Ops.SHRINK, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), shrink_multi),
  (UPat(Ops.STRIDE, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), stride_multi),
  (UPat(Ops.COPY, src=(UPat(Ops.DEVICE, name="device"), UPat(Ops.MULTI, name="multi"), )), copy_multi),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.DETACH), src=(UPat(Ops.MULTI, name="multi"), ), name="root"), passthrough_multi),
])

@track_rewrites(named=True)
def get_multi_map(big_sink:UOp) -> dict[UOp, UOp]: return {k:v for k,v in graph_rewrite_map(big_sink, multi_pm).items() if k is not v}

"""
class MultiLazyBuffer(MathTrait):
  def __init__(self, lbs:list[UOp], axis:int|None, real:list[bool]|None=None):
    assert all(isinstance(x, UOp) for x in lbs) and len(lbs), "all lbs must be LazyBuffers, and we need at least one of them"
    assert all_same([x.dtype for x in lbs]), f"all multilazybuffer needs same dtype, getting {[x.dtype for x in lbs]}"
    self.lbs, self.axis, self.dtype, self.device, self.real = lbs, axis, lbs[0].dtype, tuple(x.device for x in lbs), real or [True]*len(lbs)

  @property
  def shape(self): return tuple(sum(y.shape[a] for y in self.real_lbs) if a == self.axis else s for a,s in enumerate(self.real_lbs[0].shape))

  @property
  def size(self): return sum(x.size for x in self.real_lbs)

  @property
  def real_lbs(self): return [lb for lb,r in zip(self.lbs, self.real) if r]

  @property
  def bounds(self):
    if self.axis is None: raise RuntimeError("bounds is not defined when axis is None")
    return tuple(itertools.pairwise(itertools.accumulate([lb.shape[self.axis] for lb in self.lbs], initial=0)))

  def __repr__(self): return f"<MLB {self.axis=} {self.real=} {chr(10)}{chr(10).join([f'{x.device} {x.st}' for x in self.lbs])}>"

  def copy_to_device(self, device:str) -> UOp:
    # if we already have a copy on the device, return that
    if self.axis is None: return next((lb for lb in self.real_lbs if lb.device == device), self.real_lbs[0].copy_to_device(device))
    # copy lbs to device, pad to final shape, and sum
    llbs:list[UOp] = []
    for lb,real,(start,end) in zip(self.lbs, self.real, self.bounds):
      if not real: continue
      pad_arg = tuple((0,0) if a != self.axis else (start, self.bounds[-1][1]-end) for a in range(len(lb.shape)))
      llbs.append(lb.copy_to_device(device).pad(pad_arg))
    return functools.reduce(operator.add, llbs)

  # passthroughs
  @property
  def is_realized(self) -> bool: return all(lb.base.realized is not None for lb in self.real_lbs)
  def cast(self, dtype:DType): return MultiLazyBuffer([x.cast(dtype) for x in self.lbs], self.axis, self.real)
  def bitcast(self, dtype:DType): return MultiLazyBuffer([x.bitcast(dtype) for x in self.lbs], self.axis, self.real)
  def const_like(self, b) -> MultiLazyBuffer: return MultiLazyBuffer([x.const_like(b) for x in self.lbs], self.axis, self.real)
  def assign(self, x:MultiLazyBuffer): return MultiLazyBuffer([s.assign(d) for s,d in zip(self.lbs, x.lbs)], self.axis, self.real)
  def contiguous(self): return MultiLazyBuffer([x.contiguous() for x in self.lbs], self.axis, self.real)
  def clone(self) -> MultiLazyBuffer: return MultiLazyBuffer([lb.clone() for lb in self.lbs], self.axis, self.real)
  def detach(self) -> MultiLazyBuffer: return MultiLazyBuffer([lb.detach() for lb in self.lbs], self.axis, self.real)
  @property
  def toposort(self) -> dict[UOp, None]: return {l:None for x in self.lbs for l in x.toposort}

  # elementwise is simple
  def alu(self, op:Ops, *in_srcs:MultiLazyBuffer) -> MultiLazyBuffer:
    msrcs = (self,)+in_srcs
    assert all(isinstance(x, MultiLazyBuffer) for x in msrcs), f"all buffers must be MultiLazyBuffer {msrcs}"
    assert all_same([x.device for x in msrcs]), f"all buffers must have the same device {[x.device for x in msrcs]}"

    # NOTE: they all have to share an axis, we always choose [-1]
    axis, bounds = axes[-1] if len(axes := dedup([(x.axis, x.bounds) for x in msrcs if x.axis is not None])) else (None, None)
    srcs:list[list[UOp]] = []
    not_all_real = not all(all(mlb.real) for mlb in msrcs)
    new_real = [all(transposed) for transposed in zip(*[mlb.real for mlb in msrcs])] if not_all_real else self.real
    assert any(new_real), "output contains no real lb"
    for mlb in msrcs:
      if (mlb.axis == axis and (mlb.axis is None or mlb.bounds == bounds)) or not_all_real: srcs.append(mlb.lbs)
      else:
        assert axis is not None and bounds is not None
        if mlb.axis is None: srcs.append(to_sharded(mlb.lbs, axis, bounds))
        else: srcs.append(to_sharded([mlb.copy_to_device(lb.device) for lb in mlb.lbs], axis, bounds))
    new_real_lbs:dict[int,UOp] = {i:lsrcs[0].alu(op, *lsrcs[1:]) for i,(lsrcs,r) in enumerate(zip(zip(*srcs), new_real)) if r}
    # NOTE: const dtype should match real
    new_dtype = next(iter(new_real_lbs.values())).dtype
    return MultiLazyBuffer([new_real_lbs.get(i, lsrcs[0].const_like(0).cast(new_dtype)) for i,lsrcs in enumerate(zip(*srcs))], axis, new_real)

  def r(self, op:Ops, axis:tuple[int, ...]) -> MultiLazyBuffer:
    if self.axis is not None and self.axis in axis:
      # all-reduce on sharded axes
      reduced_parts = [(x if r else x.const_like(0)).r(op, axis) for x,r in zip(self.lbs, self.real)]
      # if all partitions are real, do all_reduce
      if all(self.real): return MultiLazyBuffer(all_reduce(op, reduced_parts), None)
      # only one partition is real, keep it
      return MultiLazyBuffer(reduced_parts, None, self.real)
    # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
    return MultiLazyBuffer([x.r(op, axis) for x in self.lbs], self.axis, self.real)

  # *** movement ops ***

  def _shape_to_single_shard(self, shape:tuple[sint, ...], lb:UOp) -> tuple[sint, ...]:
    return tuple(lb.shape[self.axis] if a == self.axis else s for a,s in enumerate(shape))

  def reshape(self, arg:tuple[sint, ...]):
    if self.axis is None: return MultiLazyBuffer([x.reshape(arg) for x in self.lbs], None, self.real)
    assert prod(self.shape) == prod(arg), "reshape must maintain prod(shape)"
    arg_acc:list[sint] = list(itertools.accumulate(arg, operator.mul, initial=1))
    # new_axis is the last one that preserves prod(prior to new_axis) and must not move items between shards
    # todo: what to do about shrinking to self.shape[self.axis]==1 len(self.real_lbs)==1?
    new_axis = len(arg_acc) - arg_acc[::-1].index(prod(self.shape[:self.axis])) - 1
    assert all(prod(lb.shape[self.axis:])%prod(arg[new_axis+1:])==0 for lb in self.lbs), f"reshape cannot move items between shards {self=} {arg=}"
    lbs = [x.reshape(tuple(s if a!=new_axis else prod(x.shape[self.axis:])//prod(arg[new_axis+1:]) for a,s in enumerate(arg))) for x in self.lbs]
    return MultiLazyBuffer(lbs, new_axis, self.real)

  def pad(self, arg:tuple[tuple[sint, sint], ...]):
    assert self.axis is None or arg[self.axis] == (0,0) or not all(self.real), f"padding not supported for {arg=}"
    # pad on shard axis -> fill others with zeros and set real to all True
    if self.axis is not None and arg[self.axis] != (0,0):
      # pad back to whole axis, remove real mask
      assert all(arg[i] == (0, 0) for i in range(len(self.shape)) if i != self.axis), "cannot pad sharded and non-sharded axis at the same time"
      dim, bound = sum(lb.shape[self.axis] for lb in self.lbs), self.bounds[self.real.index(True)]
      assert arg[self.axis] == (bound[0], dim-bound[1]), "can only pad to whole axis"
      return MultiLazyBuffer([x if r else x.const_like(0) for x,r in zip(self.lbs, self.real)], self.axis)
    return MultiLazyBuffer([x.pad(arg) for x in self.lbs], self.axis, self.real)

  def expand(self, arg:tuple[sint, ...]):
    # NOTE: this assert isn't needed, sharded axis can have dim 1
    assert self.axis is None or arg[self.axis] == self.shape[self.axis], f"expand not supported on sharded axis {arg=}"
    return MultiLazyBuffer([x.expand(self._shape_to_single_shard(arg, x)) for x in self.lbs], self.axis, self.real)

  def permute(self, arg:tuple[int, ...]):
    # all permutes supported!
    return MultiLazyBuffer([x.permute(arg) for x in self.lbs], arg.index(self.axis) if self.axis is not None else None, self.real)

  def shrink(self, arg:tuple[tuple[sint, sint], ...]):
    assert self.axis is None or arg[self.axis] == (0, self.shape[self.axis]) or arg[self.axis] in self.bounds, f"shrinking not supported for {arg=}"
    if self.axis is not None and arg[self.axis] in self.bounds and arg[self.axis] != (0, self.shape[self.axis]):
      assert all(arg[i] == (0, s) or i == self.axis for i,s in enumerate(self.shape)), "cannot shrink sharded and non-sharded axis at the same time"
      # NOTE: shrink on the shard axis is only allowed when result is a single partition, denoted by the new real
      idx = self.bounds.index(arg[self.axis])
      # zero out other lbs to not create lb reference
      return MultiLazyBuffer([lb if i==idx else lb.const_like(0) for i,lb in enumerate(self.lbs)], self.axis, [i==idx for i in range(len(self.lbs))])
    return MultiLazyBuffer([x.shrink(tuple((0, x.shape[self.axis]) if a == self.axis else s for a,s in enumerate(arg))) for x in self.lbs],
                           self.axis, self.real)

  def stride(self, arg:tuple[int, ...]):
    assert self.axis is None or arg[self.axis] == 1, "flipping not supported on sharded axis"
    return MultiLazyBuffer([x.stride(arg) for x in self.lbs], self.axis, self.real)
"""
