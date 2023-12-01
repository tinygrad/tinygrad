from __future__ import annotations
import os
os.environ['PYOPENCL_NO_CACHE'] = '1'
import pathlib, functools
import numpy as np
import pyopencl as cl
from typing import Optional, List, Tuple
from tinygrad.helpers import DEBUG, getenv, prod, ImageDType, OSX, fromimport, diskcache, DType
from tinygrad.device import Compiled, LRUAllocator
from tinygrad.renderer.opencl import OpenCLRenderer
from tinygrad.codegen.kernel import LinearizerOptions

OSX_TIMING_RATIO = (125/3) if OSX else 1.0   # see test/external/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
ROCM_LLVM_PATH = pathlib.Path("/opt/rocm/llvm/bin")
if DEBUG >= 6:
  early_exec = fromimport("extra.helpers", "enable_early_exec")()

@diskcache
def compile_gpu(prg:str) -> bytes:
  clprg = cl.Program(GPUDevice.compile_context, prg)
  clprg.build()
  return clprg.get_info(cl.program_info.BINARIES)[0]

class CLProgram:
  def __init__(self, device:GPUDevice, name:str, prg:bytes, bufs:int=0, vars:int=0):
    self.device, self.name, self.clprogram = device, name, cl.Program(device.ctx, [device.ctx.devices[0]], [prg])
    self.clprogram.build()
    self.clprg = self.clprogram.__getattr__(name)
    if DEBUG >= 5 and not OSX:
      device_name = self.device.ctx.devices[0].name
      if 'Adreno' in device_name:
        fromimport('disassemblers.adreno', 'disasm')(prg)
      elif device_name.startswith('gfx'):
        asm = early_exec(([ROCM_LLVM_PATH / "llvm-objdump", '-d', '-'], prg))
        print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))
      elif "NVIDIA" in device_name:
        # print the PTX for NVIDIA.
        print(prg.decode('utf-8'))
    if vars > 0: self.clprg.set_scalar_arg_dtypes([None]*bufs + [np.int32]*vars)

  @staticmethod
  def max_work_group_size(): return GPUDevice.compile_context.devices[0].max_work_group_size if GPUDevice.compile_context is not None else 1024

  def __call__(self, *bufs, global_size:Tuple[int,int,int], local_size:Optional[Tuple[int,int,int]]=None, wait=False) -> Optional[float]:
    e = self.clprg(self.device.queue, [int(g*l) for g,l in zip(global_size, local_size)] if local_size is not None else global_size, local_size, *bufs)
    if wait:
      e.wait()
      try:
        return ((e.profile.end - e.profile.start) * OSX_TIMING_RATIO) * 1e-9
      except cl.RuntimeError:   # no profiling info available
        return None
    return None

class CLAllocator(LRUAllocator):
  def __init__(self, device:GPUDevice):
    self.events: List[cl.Event] = []
    self.device = device
    super().__init__()
  def _alloc(self, size:int, dtype:DType):
    if size == 0: return None
    if isinstance(dtype, ImageDType):
      # NOTE: the memory is a bit off here due to padding, it's buf.row_pitch * buf.height * 4 * dtype.itemsize
      assert size == prod(dtype.shape), f"image size mismatch {size} != {dtype.shape}"
      fmt = cl.ImageFormat(cl.channel_order.RGBA, {2: cl.channel_type.HALF_FLOAT, 4: cl.channel_type.FLOAT}[dtype.itemsize])
      buf = cl.Image(self.device.ctx, cl.mem_flags.READ_WRITE, fmt, shape=(dtype.shape[1], dtype.shape[0]))
    else:
      buf = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size * dtype.itemsize)
    return buf
  def copyin(self, dest:cl.Buffer, src:memoryview): self.events.append(cl.enqueue_copy(self.device.queue, dest, src, is_blocking=False))
  def copyout(self, dest:memoryview, src:cl.Buffer):
    self.events.clear()
    cl.enqueue_copy(self.device.queue, dest, src, is_blocking=True)

class GPUDevice(Compiled):
  devices = None
  compile_context = None
  def __init__(self, device:str):
    if GPUDevice.devices is None:
      cl_platforms = cl.get_platforms()
      platform_devices: List[List[cl.Device]] = [y for y in ([x.get_devices(device_type=cl.device_type.GPU) for x in cl_platforms] + [x.get_devices(device_type=cl.device_type.CPU) for x in cl_platforms]) if y]
      GPUDevice.devices = [device for device in platform_devices[getenv('CL_PLATFORM', 0)] if device.name not in getenv('CL_EXCLUDE', "").split(",")]
      if DEBUG >= 1: print(f"using devices: {[device.hashable_model_and_version_identifier for device in GPUDevice.devices]}")
    self.device = int(device.split(":")[1]) if ":" in device else 0
    self.ctx = cl.Context(devices=[GPUDevice.devices[self.device]])
    if GPUDevice.compile_context is None: GPUDevice.compile_context = self.ctx
    self.queue = cl.CommandQueue(self.ctx, device=self.ctx.devices[0], properties=cl.command_queue_properties.PROFILING_ENABLE)
    super().__init__(CLAllocator(self), LinearizerOptions(), OpenCLRenderer, compile_gpu, functools.partial(CLProgram, self))
  def synchronize(self): self.queue.finish()