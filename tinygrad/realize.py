from typing import List, Dict, Optional, cast
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, GlobalCounters
from tinygrad.device import Device, Buffer, BufferCopy, BufferXfer, BufferRead, JITRunner, update_stats, InterpretedASTRunner
from tinygrad.device import CompiledASTRunner, BufferOptions, Compiled
from tinygrad.graph import print_tree, realized_lazybuffer
from tinygrad.helpers import colored, getenv, GRAPH, diskcache, to_mv
from tinygrad.shape.symbolic import Variable

# *** schedule running ***

from gpuctypes import hip
from tinygrad.runtime.ops_hip import check
import functools
@functools.lru_cache(None)
def enable_peer(d0, d1):
  check(hip.hipSetDevice(d0))
  check(hip.hipDeviceEnablePeerAccess(d1, 0))

from tinygrad.runtime.ops_hip import compile_hip

@diskcache
def compile_hip_cached(x): return compile_hip(x)

class HIPWaitOp(CompiledASTRunner):
  def __init__(self, device):
    prg = 'extern "C" __global__ void hip_wait(int *sem) { while (!atomicCAS_system(sem, 1, 0)); }'
    super().__init__(None, colored("hip_wait", "RED"), prg, compile_hip_cached(prg), Device[device], [1,1,1], [1,1,1])
    self.build(Device[device].runtime)

# TODO: these can go in the Linearizer
class HIPSyncOp(CompiledASTRunner):
  def __init__(self, device):
    prg = 'extern "C" __global__ void hip_sync(int *sem, void *data) { sem[0] = 1; }'
    super().__init__(None, colored("hip_sync", "red"), prg, compile_hip_cached(prg), Device[device], [1,1,1], [1,1,1])
    self.build(Device[device].runtime)

class HIPCopyOp(CompiledASTRunner):
  def __init__(self, dest_device, src_device, dtype, sz):
    enable_peer(Device[dest_device].device, Device[src_device].device)
    prg = f'extern "C" __global__ void hip_copy({dtype.name}* a, {dtype.name}* b)' + \
      '{ const int gx = blockIdx.x*blockDim.x + threadIdx.x; a[gx] = b[gx]; }'
    gsz, lsz = sz, 1
    while lsz < 128 and gsz%2 == 0: gsz, lsz = gsz//2, lsz*2
    super().__init__(None, colored("hip_copy", "yellow"), prg, compile_hip_cached(prg), Device[dest_device], [gsz,1,1], [lsz,1,1])
    self.build(Device[dest_device].runtime)
    self.mem_estimate = dtype.itemsize*sz

class SyncOp(JITRunner):
  def __init__(self, device):
    self.device = device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    Device[self.device].synchronize()
    update_stats(colored("synchronize", "RED"), 0, 0, {}, None, 1, device=self.device)

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert all(si.out.device == x.device for x in si.inputs) or si.ast.op in {LoadOps.COPY, LoadOps.WAIT}, \
    f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
  if si.ast.op is LoadOps.EMPTY: return None
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  if si.out.device.startswith("HIP") and all(x.device.startswith("HIP") for x in si.inputs):
    if si.ast.op is LoadOps.COPY: return HIPCopyOp(si.out.device, si.inputs[0].device, si.out.dtype, si.out.size)
    if si.ast.op is LoadOps.SYNC: return HIPSyncOp(si.out.device)
    if si.ast.op is LoadOps.WAIT: return HIPWaitOp(si.out.device)
  else:
    if si.ast.op is LoadOps.COPY:
      if hasattr(Device[si.out.device].allocator, 'transfer') and type(Device[si.out.device]) is type(Device[si.inputs[0].device]):
        return BufferXfer()
      if si.inputs[0].device.startswith("DISK"): return BufferRead()
      return BufferCopy()
    if si.ast.op is LoadOps.SYNC: return SyncOp(si.out.device) if isinstance(Device[si.out.device], Compiled) else None
    if si.ast.op is LoadOps.WAIT: return None
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
    if si.out.size > 0:
      if si.ast.op is LoadOps.SYNC: options = BufferOptions(host=True)
      elif si.out.uncached: options = BufferOptions(uncached=True)
      else: options = None

      si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
        Buffer(si.out.device, si.out.size, si.out.dtype, "PLACEHOLDER" if isinstance(prg, InterpretedASTRunner) else None, options=options)

      if si.ast.op is LoadOps.SYNC and isinstance(prg, HIPSyncOp):
        to_mv(si.out.realized._buf, 4).cast("I")[0] = 0
        prg._lru_defeat = si.out.realized
    del si.out.srcs

    # run the function (put it in JIT)
    real_buffers = [x.realized for x in (si.out,)+si.inputs if x.size != 0]
    assert all(x is not None for x in real_buffers), f"can't run, some inputs aren't realized {real_buffers}"
    if prg: prg.exec(cast(List[Buffer], real_buffers), si.var_vals)
    elif si.out.size > 0: update_stats(colored(f"empty {si.out.st.size:10d} {si.out.dtype}", "yellow"), 0, 0, {}, None, 1, device=si.out.device)
    if GRAPH: realized_lazybuffer(si.out, GlobalCounters.kernel_count)
