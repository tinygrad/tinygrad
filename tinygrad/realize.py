from typing import List, cast, Dict, Callable
import time
import numpy as np
from dataclasses import dataclass
from tinygrad.ops import ScheduleItem, LazyOp, LoadOps, Device, BufferOps, Interpreted, UnaryOps, print_info
from tinygrad.graph import log_schedule_item, print_tree
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DEBUG, prod, all_int, getenv, IMAGE

from tinygrad.runtime.lib import RawBuffer
from tinygrad.features.image import fix_schedule_for_images

# *** main schedule runner ***

method_cache = {}
def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  # HACK: images can be not usable due to shape
  if IMAGE >= 2: schedule = fix_schedule_for_images(schedule)

  # NOTE: if you for loop the schedule it's slow because nothing frees
  while len(schedule):
    si = schedule.pop(0)
    if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), "can't run schedule, some inputs aren't realized"
    if DEBUG >= 3: print_tree(si.ast)
    device = Device[si.out.device]
    if si.ast.op in LoadOps:
      # confirm the LoadOps are contiguous and in order
      for i,s in enumerate(si.ast.src): assert isinstance(s, LazyOp) and s.op == BufferOps.MEM and s.arg.idx == i+1 and s.arg.st.contiguous, f"bad LoadOps src {i}: {s}"
      LOAD_OPS_DISPATCHER[cast(LoadOps, si.ast.op)](si.out, *si.inputs)
    else:
      rawbufs = [x.realized for x in si.inputs]
      if isinstance(device, Interpreted):
        fxn = device.interpret_ast(si.ast)

        # run function
        st = time.perf_counter()
        ret = fxn(*rawbufs)
        et = time.perf_counter() - st

        # TODO: this shouldn't be needed
        if ret.dtype != si.out.dtype:
          ret = device.from_underlying(device.fxn_for_op[UnaryOps.CAST](device.to_underlying(ret), (si.out.dtype, False)))

        # handle assignment
        if si.out.output_buffer is not None:
          assert si.out.output_buffer.dtype == ret.dtype
          si.out.output_buffer._buf = ret._buf
          ret = si.out.output_buffer

        # final
        si.out.realized = ret
        print_info("<interpreted>", si.ast, si.var_vals, {}, et)
      else:
        # check if we can reuse the output buffer
        # if it's aliased, don't use it
        # NOTE: this is pretty wrong actually, who knows where else this buffer is used?
        si.out.realized = si.out.output_buffer
        if si.out.realized:
          for i,a in enumerate(si.inputs):
            # TODO: if this is contiguous it's fine
            if a.realized == si.out.realized:
              if any(not x.arg.st.contiguous for x in si.ast.get_lazyops() if x.op == BufferOps.MEM and x.arg.idx == i+1):
                si.out.realized = None
                break

        # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
        if not si.out.realized:
          si.out.realized = device.buffer(prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype)

        # all the rawbufs
        rawbufs = [si.out.realized] + rawbufs

        # compile the program
        runner = device.compile_ast(si.ast, rawbufs)

        # add this function to JIT
        from tinygrad.jit import CacheCollector
        CacheCollector.add(runner, rawbufs, si.var_vals)

        # run the function
        runner(rawbufs, si.var_vals)

    del si.out.op
    for v in si.out.views: del v.op
    assert si.out.realized and isinstance(si.out.realized, Device[si.out.device].buffer), f"device mismatch on realized got {type(si.out.realized)} expected {si.out.device}"
    assert si.out.realized.dtype == si.out.dtype, f"realized dtype is incorrect, {si.out.realized.dtype} != {si.out.dtype}"

# *** zero op LoadOps ***

def _realize_empty(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***     empty {buffer.device}                              shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

def _realize_rand(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  if DEBUG >= 2: print(f"***      rand {buffer.device}    seed {buffer.op.arg:<10d}  shape {str(buffer.shape):23s} dtype {buffer.dtype}")
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=prod(buffer.shape), dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False), **buffer._device_extra_args())

# *** one op LoadOps ***

from tinygrad.runtime.lib import RawBufferMapped, RawBufferTransfer
from tinygrad.runtime.ops_disk import RawDiskBuffer
def _realize_from(buffer: LazyBuffer, src: LazyBuffer) -> None:
  assert src.realized.size == buffer.st.size(), f"size mismatch on FROM {src.realized.size} != {buffer.st.size()}"
  assert src.st.contiguous and buffer.st.contiguous, "all must be contiguous for from"
  if DEBUG >= 2: print(f"***      copy {buffer.device} <- {src.device} size {src.realized.size:<16d} shape {str(buffer.shape):23s} dtype {src.realized.dtype}")
  # TODO: make this generic
  if isinstance(src.realized, RawDiskBuffer) and issubclass(Device[buffer.device].buffer, RawBufferMapped):
    assert all_int(buffer.shape), "does not support symbolic shape"
    buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())
    src.realized.readinto(cast(RawBufferMapped, buffer.realized)._buffer())
  elif isinstance(src.realized, RawBufferTransfer) and issubclass(Device[buffer.device].buffer, RawBufferTransfer) and getenv("P2P", 0) >= 1:
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
