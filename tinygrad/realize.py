from typing import List, Dict, Optional, cast
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, GlobalCounters
from tinygrad.device import Device, Buffer, BufferCopy, BufferXfer, BufferRead, JITRunner, update_stats, InterpretedASTRunner, BufferOptions
from tinygrad.graph import print_tree, realized_lazybuffer
from tinygrad.helpers import colored, getenv, GRAPH, diskcache, to_mv
from tinygrad.dtype import dtypes
from tinygrad.shape.symbolic import Variable

# *** schedule running ***

from tinygrad.runtime.ops_hip import compile_hip
from tinygrad.device import CompiledASTRunner

@diskcache
def compile_hip_cached(x): return compile_hip(x)

class BufferXferHip(BufferCopy):
  def __init__(self, dest, src):
    self.dest, self.src = dest, src

    prg = 'extern "C" __global__ void sync(int *sem) { atomicExch_system(sem, 1); }'
    self.sync = CompiledASTRunner(None, "sync", prg, compile_hip_cached(prg), Device[self.src], [1,1,1], [1,1,1]).build(Device[self.src].runtime)

    prg = 'extern "C" __global__ void block(int *sem) { while (!atomicCAS_system(sem, 1, 1)); }'
    self.block = CompiledASTRunner(None, "block", prg, compile_hip_cached(prg), Device[self.dest], [1,1,1], [1,1,1]).build(Device[self.dest].runtime)

    self.sem = Buffer(self.dest, 4, dtypes.uint32, options=BufferOptions(host=True))

    self.mv = to_mv(self.sem._buf, 4).cast("I")
    self.mv[0] = 0
    super().__init__()

  def copy(self, dest:Buffer, src:Buffer):
    assert self.dest == dest.device
    assert self.src == src.device
    self.mv[0] = 0
    src.allocator.transfer(dest._buf, src._buf, dest.nbytes)
    #self.sync([self.sem], {})
    #self.block([self.sem], {})
    self.sync.clprg(self.sem._buf, global_size=[1,1,1], local_size=[1,1,1])
    self.block.clprg(self.sem._buf, global_size=[1,1,1], local_size=[1,1,1])

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
    if hasattr(Device[si.out.device].allocator, 'transfer') and type(Device[si.out.device]) is type(Device[si.inputs[0].device]):
      return BufferXferHip(si.out.device, si.inputs[0].device)
      #return BufferXfer()
    if si.inputs[0].device.startswith("DISK"): return BufferRead()
    return BufferCopy()
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  return Device[si.out.device].get_runner(si.ast)

logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
def run_schedule(schedule:List[ScheduleItem]):
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

    # run the function (put it in JIT)
    assert all(x.realized is not None for x in si.inputs), f"can't run, some inputs aren't realized {[x for x in si.inputs if x.realized is None]}"
    if prg: prg.exec([si.out.realized] + [cast(Buffer, x.realized) for x in si.inputs], si.var_vals)
    else: update_stats(colored(f"empty {si.out.st.size:10d} {si.out.dtype}", "yellow"), 0, 0, {}, None, 1, device=si.out.device)
    if GRAPH: realized_lazybuffer(si.out, GlobalCounters.kernel_count)
