from typing import List, Dict, Optional, cast
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, GlobalCounters
from tinygrad.device import Device, Buffer, BufferCopy, JITRunner, update_stats, InterpretedASTRunner, SyncEvent, BlockEvent
from tinygrad.graph import print_tree, realized_lazybuffer
from tinygrad.helpers import colored, getenv, GRAPH
from tinygrad.shape.symbolic import Variable

# *** schedule running ***

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert all(si.out.device == x.device for x in si.inputs) or si.ast.op is LoadOps.COPY, \
    f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
  if si.ast.op is LoadOps.EMPTY: return None
  if si.ast.op is LoadOps.COPY:
    # TODO: determine the copy type here
    return BufferCopy(si.out.device)
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  return Device[si.out.device].get_runner(si.ast)

logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
def run_schedule(schedule:List[ScheduleItem]):
  synced_buffers = {}
  while len(schedule):
    si = schedule.pop(0)
    if logops and si.ast.op not in LoadOps: logops.write(str(si.ast)+"\n")

    # get the program
    prg = lower_schedule_item(si)

    # invalidate the output buffer if there's a non contig usage of it in inputs
    if si.out.output_buffer is not None:
      for i,a in enumerate(si.inputs):
        if a.realized == si.out.output_buffer:
          if any(not x.arg.st.contiguous for x in si.ast.lazyops if x.op is BufferOps.LOAD and x.arg.idx == i+1):
            si.out.output_buffer = None
            break

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
      Buffer(si.out.device, si.out.size, si.out.dtype, "PLACEHOLDER" if isinstance(prg, InterpretedASTRunner) else None)
    del si.out.srcs

    # should we wait?
    if isinstance(prg, BufferCopy):
      if si.inputs[0] in synced_buffers:
        BlockEvent(si.out.device, synced_buffers[si.inputs[0]]).exec([])
        #sync_num, evt = synced_buffers[si.inputs[0]]
        #update_stats(colored(f"block {sync_num}", "RED"), 0, 0, {}, None, 1, device=si.out.device)
        #Device[si.out.device].block(evt)
      else:
        # if we don't have a sync point, we have to sync the whole device
        Device[si.inputs[0].device].synchronize()

    # run the function (put it in JIT)
    assert all(x.realized is not None for x in si.inputs), f"can't run, some inputs aren't realized {[x for x in si.inputs if x.realized is None]}"
    if prg: prg.exec([si.out.realized] + [cast(Buffer, x.realized) for x in si.inputs], si.var_vals)
    else: update_stats(colored(f"empty {si.out.st.size:10d} {si.out.dtype}", "yellow"), 0, 0, {}, None, 1, device=si.out.device)
    if GRAPH: realized_lazybuffer(si.out, GlobalCounters.kernel_count)

    # should we sync?
    if si.out.does_synchronize and hasattr(Device[si.out.device], 'event_create'):
      synced_buffers[si.out] = SyncEvent(si.out.device)
      synced_buffers[si.out].exec([])

      #sync_num = len(synced_buffers)
      #synced_buffers[si.out] = (sync_num, Device[si.out.device].event())
      #update_stats(colored(f"event {sync_num}", "red"), 0, 0, {}, None, 1, device=si.out.device)

