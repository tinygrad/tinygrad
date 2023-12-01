from typing import List, cast, Dict, Callable
import numpy as np
from tinygrad.ops import ScheduleItem, LazyOp, LoadOps, BufferOps
from tinygrad.device import Device, Buffer
from tinygrad.graph import log_schedule_item, print_tree
from tinygrad.helpers import DEBUG, prod

def run_schedule(schedule:List[ScheduleItem], disable_logging=False):
  # NOTE: if you for loop the schedule it's slow because nothing frees
  while len(schedule):
    si = schedule.pop(0)
    if not disable_logging: log_schedule_item(si)
    assert all(x.realized for x in si.inputs), "can't run schedule, some inputs aren't realized"
    assert all(si.out.device == x.device for x in si.inputs) or si.ast.op is LoadOps.FROM, f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
    # check if we can reuse the output buffer
    # if it's aliased, don't use it
    # TODO: this is pretty wrong actually, who knows where else this buffer is used?
    # TODO: what if an assign is required? this silently is wrong
    # TODO: this logic doesn't belong here, it should be checked in assign or at least schedule
    if si.out.output_buffer is not None:
      for i,a in enumerate(si.inputs):
        # TODO: if this is contiguous it's fine
        if a.realized == si.out.output_buffer:
          if any(not x.arg.st.contiguous for x in si.ast.get_lazyops() if x.op == BufferOps.LOAD and x.arg.idx == i+1):
            si.out.output_buffer = None
            break
    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
      Buffer(si.out.device, prod((s if isinstance(s, int) else s.max for s in si.out.shape)), si.out.dtype)
    # TODO: size 0 should be removed from the schedule
    if si.out.realized.size != 0:
      if si.ast.op in LoadOps:
        if DEBUG >= 2: print(f"***   {si.ast.op:>15s}    {f'{si.out.device} <- {si.inputs[0].device}' if si.ast.op is LoadOps.FROM else si.out.device:25s}     sz {si.out.realized.size:5d}    shape {si.out.shape}    dtype {si.out.dtype}    arg {si.ast.arg}")
        # confirm the LoadOps are contiguous and in order
        for i,s in enumerate(si.ast.src): assert isinstance(s, LazyOp) and s.op == BufferOps.LOAD and s.arg.idx == i+1 and s.arg.st.contiguous, f"bad LoadOps src {i}: {s}"
        kwargs = {"arg": si.ast.arg} if si.ast.arg is not None else {}
        LOAD_OPS_DISPATCHER[cast(LoadOps, si.ast.op)](si.out.realized, *[x.realized for x in si.inputs], **kwargs)
      else:
        Device[si.out.device].get_runner(si.ast).exec([si.out.realized] + [x.realized for x in si.inputs], si.var_vals)
    del si.out.op
    for v in si.out.views: del v.op
    #assert si.out.realized and isinstance(si.out.realized, Device[si.out.device].buffer), f"device mismatch on realized got {type(si.out.realized)} expected {si.out.device}"
    assert si.out.realized.dtype == si.out.dtype, f"realized dtype is incorrect, {si.out.realized.dtype} != {si.out.dtype}"

# *** LoadOps implementation ***

# TODO: remove this and write the RNG in tinygrad
def _realize_rand(buffer: Buffer, arg) -> None:
  rng = np.random.default_rng(arg)
  rng_np_buffer = rng.random(size=buffer.size, dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False)
  buffer.copyin(rng_np_buffer.data)

def _realize_custom(*buffers: Buffer, arg) -> None: arg(*buffers)

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.EMPTY: lambda x: None,
  LoadOps.RAND: _realize_rand,
  LoadOps.FROM: Buffer.copy_,
  LoadOps.CUSTOM: _realize_custom
}
