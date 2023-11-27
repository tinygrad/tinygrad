from typing import List, cast, Dict, Callable
import numpy as np
from tinygrad.ops import ScheduleItem, LazyOp, LoadOps, Device, BufferOps
from tinygrad.graph import log_schedule_item, print_tree
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DEBUG, prod, all_int, IMAGE
from tinygrad.features.image import fix_schedule_for_images

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  # HACK: images can be not usable due to shape
  if IMAGE >= 2: schedule = fix_schedule_for_images(schedule)

  # NOTE: if you for loop the schedule it's slow because nothing frees
  while len(schedule):
    si = schedule.pop(0)
    if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), "can't run schedule, some inputs aren't realized"
    if si.ast.op in LoadOps:
      # confirm the LoadOps are contiguous and in order
      for i,s in enumerate(si.ast.src): assert isinstance(s, LazyOp) and s.op == BufferOps.MEM and s.arg.idx == i+1 and s.arg.st.contiguous, f"bad LoadOps src {i}: {s}"
      LOAD_OPS_DISPATCHER[cast(LoadOps, si.ast.op)](si.out, *si.inputs)
    else:
      assert all(si.out.device == x.device for x in si.inputs), f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
      # check if we can reuse the output buffer
      # if it's aliased, don't use it
      # TODO: this is pretty wrong actually, who knows where else this buffer is used?
      # TODO: what if an assign is required? this silently is wrong
      # TODO: this logic doesn't belong here, it should be checked in assign or at least schedule
      if si.out.output_buffer is not None:
        for i,a in enumerate(si.inputs):
          # TODO: if this is contiguous it's fine
          if a.realized == si.out.output_buffer:
            if any(not x.arg.st.contiguous for x in si.ast.get_lazyops() if x.op == BufferOps.MEM and x.arg.idx == i+1):
              si.out.output_buffer = None
              break
      # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
      si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
        Device[si.out.device].buffer(prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype, **si.out._device_extra_args())
      # TODO: should this be handled here? it probably just shouldn't be in the schedule
      if not hasattr(si.out.realized, 'size') or si.out.realized.size != 0:
        Device[si.out.device].get_runner(si.ast).exec([si.out.realized] + [x.realized for x in si.inputs], si.var_vals)
    del si.out.op
    for v in si.out.views: del v.op
    assert si.out.realized and isinstance(si.out.realized, Device[si.out.device].buffer), f"device mismatch on realized got {type(si.out.realized)} expected {si.out.device}"
    assert si.out.realized.dtype == si.out.dtype, f"realized dtype is incorrect, {si.out.realized.dtype} != {si.out.dtype}"

# *** zero op LoadOps ***

def _realize_empty(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***     empty {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

# TODO: remove this and write the RNG in tinygrad
def _realize_rand(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***      rand {buffer.device}    seed {buffer.op.arg:<10d}  shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=prod(buffer.shape), dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False), **buffer._device_extra_args())

# *** one op LoadOps ***

def _realize_from(buffer: LazyBuffer, src: LazyBuffer) -> None:
  assert src.realized.size == buffer.st.size(), f"size mismatch on FROM {src.realized.size} != {buffer.st.size()}"
  assert src.st.contiguous and buffer.st.contiguous, "all must be contiguous for from"
  if DEBUG >= 2: print(f"***      copy {buffer.device} <- {src.device} size {src.realized.size:<16d} shape {str(buffer.shape):23s} dtype {src.realized.dtype}")
  buffer.realized = Device[buffer.device].buffer.fromBuffer(src.realized, buffer.shape, buffer.dtype, **buffer._device_extra_args())

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
