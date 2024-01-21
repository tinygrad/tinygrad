from typing import List, Dict, Optional, cast
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, GlobalCounters
from tinygrad.device import Device, Buffer, BufferCopy, BufferXfer, BufferRead, JITRunner, update_stats, InterpretedASTRunner, CompiledASTRunner
from tinygrad.graph import print_tree, realized_lazybuffer
from tinygrad.helpers import colored, getenv, GRAPH, diskcache, to_mv
from tinygrad.shape.symbolic import Variable

# *** schedule running ***

from tinygrad.runtime.ops_hip import compile_hip

@diskcache
def compile_hip_cached(x): return compile_hip(x)

class BlockEvent(CompiledASTRunner):
  def __init__(self, device, evt):
    #prg = 'extern "C" __global__ void block(int *sem) { while (!atomicCAS_system(sem, 1, 1)) { __threadfence_system(); __builtin_amdgcn_s_sleep(100); } }'
    prg = 'extern "C" __global__ void block(int *sem) { while (!atomicCAS_system(sem, 1, 1)); }'
    super().__init__(None, "block", prg, compile_hip_cached(prg), Device[device], [1,1,1], [1,1,1])
    self.se = evt
    self.dname = device
    self.build(Device[device].runtime)
    self.sync = Device[self.dname].event_create()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    #print(self.se.mv[0])
    #Device[self.dname].block(self.se.sync)
    #Device[self.dname].event_record(self.sync)
    self.clprg(self.se.sem, global_size=[1,1,1], local_size=[1,1,1])
    update_stats(colored(f"block {self.se.event_num} @ {self.se.dname}", "RED"), 0, 0, {}, None, 0, jit=jit, device=self.dname)

class SyncEvent(CompiledASTRunner):
  def __init__(self, device, num):
    #prg = 'extern "C" __global__ void sync(int *sem, int value) { __threadfence_system(); __builtin_amdgcn_s_sleep(100); atomicExch_system(sem, value); }'
    prg = 'extern "C" __global__ void sync(int *sem, int value) { atomicExch_system(sem, value); __threadfence_system(); }'
    self.sem = Device[device].allocator._hostalloc(4)
    #enable_peer(0,1)
    #enable_peer(1,0)
    #self.sem = Device[device].allocator.alloc(4)
    self.mv = to_mv(self.sem, 4)
    self.clear()
    self.dname = device
    self.event_num = num
    super().__init__(None, "sync", prg, compile_hip_cached(prg), Device[device], [1,1,1], [1,1,1])
    self.build(Device[device].runtime)
    self.sync = Device[self.dname].event_create()
  def clear(self):
    self.mv[0] = 0
    self.mv[1] = 0
    self.mv[2] = 0
    self.mv[3] = 0
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    self.clear()
    # HSA_DISABLE_CACHE=1
    #Device[self.dname].event_record(self.sync)
    self.clprg(self.sem, 1, global_size=[1,1,1], local_size=[1,1,1])
    update_stats(colored(f"event {self.event_num}", "red"), 0, 0, {}, None, 0, jit=jit, device=self.dname)

import functools
@functools.lru_cache(None)
def enable_peer(d0, d1):
  check(hip.hipSetDevice(d0))
  check(hip.hipDeviceEnablePeerAccess(d1, 0))

from gpuctypes import hip
from tinygrad.runtime.ops_hip import check
class CopyKernel(CompiledASTRunner):
  def __init__(self, dest_device, src_device, sz):
    prg = f"""
    extern "C" __global__ void copy(char* a, volatile char* b) {{
      const int gx = blockIdx.x*blockDim.x + threadIdx.x;
      a[gx] = b[gx];
    }}"""
    super().__init__(None, "copy", prg, compile_hip_cached(prg), Device[dest_device], [sz,1,1], [1,1,1])
    self.build(Device[dest_device].runtime)
    enable_peer(Device[dest_device].device, Device[src_device].device)

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
    #if hasattr(Device[si.out.device].allocator, 'transfer') and type(Device[si.out.device]) is type(Device[si.inputs[0].device]): return CopyKernel(si.out.device, si.inputs[0].device, si.out.size*si.out.dtype.itemsize)
    if hasattr(Device[si.out.device].allocator, 'transfer') and type(Device[si.out.device]) is type(Device[si.inputs[0].device]): return BufferXfer()
    if si.inputs[0].device.startswith("DISK"): return BufferRead()
    return BufferCopy()
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  return Device[si.out.device].get_runner(si.ast)

logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
def run_schedule(schedule:List[ScheduleItem]):
  #synced_buffers = {}
  #for i,si in enumerate(schedule):
    #if si.out.does_synchronize and si.out.device.startswith("HIP"):
      #synced_buffers[si.out] = SyncEvent(si.out.device, i)

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

    #if isinstance(prg, (BufferXfer, CopyKernel)) and si.out.device.startswith("HIP"):
    #  if si.inputs[0] in synced_buffers and (evt:=synced_buffers[si.inputs[0]]) is not None:
    #    BlockEvent(si.out.device, evt).exec([]) #evt.sem])
    #  else:
    #    update_stats(colored("synchronize", "RED"), 0, 0, {}, None, 0, device=si.inputs[0].device)
    #    Device[si.inputs[0].device].synchronize()

    # run the function (put it in JIT)
    assert all(x.realized is not None for x in si.inputs), f"can't run, some inputs aren't realized {[x for x in si.inputs if x.realized is None]}"
    if prg: prg.exec([si.out.realized] + [cast(Buffer, x.realized) for x in si.inputs], si.var_vals)
    else: update_stats(colored(f"empty {si.out.st.size:10d} {si.out.dtype}", "yellow"), 0, 0, {}, None, 1, device=si.out.device)
    if GRAPH: realized_lazybuffer(si.out, GlobalCounters.kernel_count)

    if isinstance(prg, (BufferXfer, CopyKernel)) and si.out.device.startswith("HIP"):
      evt = SyncEvent(si.inputs[0].device, GlobalCounters.kernel_count)
      evt.exec([])
      BlockEvent(si.out.device, evt).exec([])

    # should we sync?
    #if si.out.does_synchronize and si.out.device.startswith("HIP"):
    #  synced_buffers[si.out].exec([]) #synced_buffers[si.out].sem])
