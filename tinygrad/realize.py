from typing import List, Tuple, cast, Dict, Callable
import numpy as np
from tinygrad.ops import LazyOp, LoadOps, Device, UnaryOps, BufferOps, MemBuffer, get_lazyop_info
from tinygrad.graph import log_schedule_item
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DEBUG, prod, all_int, getenv, IMAGE, ImageDType, dtypes

from tinygrad.runtime.lib import RawBufferMapped, RawBufferTransfer
from tinygrad.runtime.ops_disk import RawDiskBuffer

P2P = getenv("P2P", 0)

def fix_schedule_for_images(schedule:List[Tuple[LazyOp, LazyBuffer, Tuple[LazyBuffer, ...]]]):
  # this is the fundamental fix, find unwritable or unreadable images and convert them to normal float32 (TODO: should it be float16?)
  for op,out,buffers in schedule:
    if isinstance(out.dtype, ImageDType) and (prod(out.shape) != prod(out.dtype.shape) or not any(out.shape[x]%4 == 0 for x in out.st.unit_stride_axes())):
      out.dtype = dtypes.float32
    bops = [x for x in op.get_lazyops() if x.op == BufferOps.MEM]
    for b in bops:
      if isinstance(buffers[b.arg.idx-1].dtype, ImageDType) and (b.arg.st.real_offset() % 4 != 0 or not any(b.arg.st.shape[x]%4 == 0 for x in b.arg.st.unit_stride_axes())):
        buffers[b.arg.idx-1].dtype = dtypes.float32

  # now fix up the schedule to reflect the new dtypes
  fixed_schedule = []
  for op,out,buffers in schedule:
    # fix input dtypes to match what they actually are
    bops = [x for x in op.get_lazyops() if x.op == BufferOps.MEM]
    replacements = {}
    for x in bops:
      if x.arg.dtype != buffers[x.arg.idx-1].dtype:
        replacements[x] = LazyOp(BufferOps.MEM, (), MemBuffer(x.arg.idx, buffers[x.arg.idx-1].dtype, x.arg.st))
    if replacements: op = op.map_buffers(replacements)

    # fix the ops to create the output dtype
    if op.op not in LoadOps:
      info = get_lazyop_info(op)
      if info.dtype != out.dtype:
        op = LazyOp(UnaryOps.CAST, (op,), (out.dtype, False))

    # put this in the fixed schedule
    fixed_schedule.append((op, out, buffers))
  return fixed_schedule


def run_schedule(schedule:List[Tuple[LazyOp, LazyBuffer, Tuple[LazyBuffer, ...]]]):
  # HACK: images can be not usable due to shape
  if IMAGE >= 2: schedule = fix_schedule_for_images(schedule)

  # NOTE: if you for loop the schedule it's slow because nothing frees
  while len(schedule):
    op,out,buffers = schedule.pop(0)
    log_schedule_item(op, out, buffers)
    assert all(x.realized for x in buffers), "can't run schedule, some buffers aren't realized"
    if DEBUG >= 3:
      from extra.utils import print_tree   # type: ignore
      print_tree(op)
    if op.op in LoadOps:
      # confirm the LoadOps are contiguous and in order
      for i,s in enumerate(op.src): assert isinstance(s, LazyOp) and s.op == BufferOps.MEM and s.arg.idx == i+1 and s.arg.st.contiguous, f"bad LoadOps src {i}: {s}"
      LOAD_OPS_DISPATCHER[cast(LoadOps, op.op)](out, *buffers)
    else:
      out.realized = Device[out.device].exec_ast(op, output=out, inputs=buffers, var_vals=out.var_vals, **out._device_extra_args())
    del out.op
    for v in out.views: del v.op
    assert out.realized and isinstance(out.realized, Device[out.device].buffer), f"device mismatch on realized got {type(out.realized)} expected {out.device}"
    assert out.realized.dtype == out.dtype, "realized dtype is incorrect"

# *** zero op LoadOps ***

def _realize_empty(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***     empty {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

def _realize_rand(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***      rand {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=prod(buffer.shape), dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False), **buffer._device_extra_args())

# *** one op LoadOps ***

def _realize_from(buffer: LazyBuffer, src: LazyBuffer) -> None:
  assert src.realized.size == buffer.st.size(), f"size mismatch on FROM {src.realized.size} != {buffer.st.size()}"
  assert src.st.contiguous and buffer.st.contiguous, "all must be contiguous for from"
  if DEBUG >= 2: print(f"***      copy {buffer.device} <- {src.device} size {src.realized.size:16d} shape {str(buffer.shape):23s} dtype {src.realized.dtype}")
  # TODO: make this generic
  if isinstance(src.realized, RawDiskBuffer) and issubclass(Device[buffer.device].buffer, RawBufferMapped):
    assert all_int(buffer.shape), "does not support symbolic shape"
    buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())
    src.realized.readinto(cast(RawBufferMapped, buffer.realized)._buffer())
  elif isinstance(src.realized, RawBufferTransfer) and issubclass(Device[buffer.device].buffer, RawBufferTransfer) and P2P >= 1:
    buffer.realized = cast(RawBufferTransfer, Device[buffer.device].buffer).transfer(src.realized, buffer.shape, buffer.dtype, **buffer._device_extra_args())
  else:
    # TODO: schedule this as FROM to go to CPU, and a FROM to go to device
    buffer.realized = Device[buffer.device].buffer.fromCPU(src.realized.toCPU(), **buffer._device_extra_args())

# *** n op LoadOps ***

def _realize_custom(buffer: LazyBuffer, *inputs: LazyBuffer) -> None:
  if DEBUG >= 2: print(f"***    custom {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  buffer.realized = buffer.op.arg(buffer, *inputs)

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.EMPTY: _realize_empty,
  LoadOps.RAND: _realize_rand,
  LoadOps.FROM: _realize_from,
  LoadOps.CUSTOM: _realize_custom,
}
