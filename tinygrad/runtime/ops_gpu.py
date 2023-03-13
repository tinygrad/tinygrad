from __future__ import annotations
import platform
import numpy as np
import pyopencl as cl  # type: ignore
from typing import Optional, List, Final
from tinygrad.helpers import IMAGE, DEBUG, getenv, dtypes
from tinygrad.ops import CompiledBuffer, GlobalCounters, Specialized
from tinygrad.runtime.lib import RawBufferCopyInOut, RawBuffer
from tinygrad.codegen.gpu import GPUCodegen, GPULanguage

OSX = platform.system() == "Darwin"
OSX_TIMING_RATIO = (125/3) if OSX else 1.0   # see test/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something
FLOAT16 = getenv("FLOAT16", 0)

class _CL:
  def __init__(self):
    devices: List[cl.Device] = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0: devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], []) # settle for CPU
    if len(devices) > 1 or DEBUG >= 1: print(f"using {devices[getenv('CL_DEVICE', 0)]}")
    self.cl_ctx: cl.Context = cl.Context(devices=[devices[getenv("CL_DEVICE", 0)]])
    self.cl_queue: cl.CommandQueue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # this is an in-order command queue
CL = _CL()

class CLBuffer(RawBufferCopyInOut):
  def __init__(self, size, dtype):
    super().__init__(size, dtype)
    self._cl = cl.Buffer(CL.cl_ctx, cl.mem_flags.READ_WRITE, self._memsz)
  def copyin(self, x:np.ndarray): cl.enqueue_copy(CL.cl_queue, self._cl, x, is_blocking=False)
  def copyout(self, x:np.ndarray): cl.enqueue_copy(CL.cl_queue, x, self._cl, is_blocking=True)

class CLImage(RawBuffer):  # pylint: disable=abstract-method
  IMAGE: Final = True
  def __init__(self, shape, dtype=dtypes.float16 if getenv("FLOAT16") else dtypes.float32):  # pylint: disable=super-init-not-called
    fmt = cl.ImageFormat(cl.channel_order.RGBA, {dtypes.float16: cl.channel_type.HALF_FLOAT, dtypes.float32: cl.channel_type.FLOAT}[dtype])
    self.size, self.dtype, self._cl = shape, dtype, cl.Image(CL.cl_ctx, cl.mem_flags.READ_WRITE, fmt, shape=(shape[1], shape[0]))
    GlobalCounters.mem_used += self._cl.row_pitch * self._cl.height
  def __del__(self): GlobalCounters.mem_used -= self._cl.row_pitch * self._cl.height

#@functools.lru_cache(maxsize=None)
class CLProgram:
  def __init__(self, name:str, prg:str, binary=False, argdtypes=None):
    self.name, self.argdtypes, self.clprogram = name, argdtypes, cl.Program(CL.cl_ctx, CL.cl_ctx.devices, [prg]) if binary else cl.Program(CL.cl_ctx, prg)  # type: ignore
    try:
      self._clprg = self.clprogram.build()
    except cl.RuntimeError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    self.clprg = self._clprg.__getattr__(name)
    if DEBUG >= 5 and not OSX:
      binary = self.clprogram.get_info(cl.program_info.BINARIES)[0]
      if 'Adreno' in CL.cl_ctx.devices[0].name:
        from disassemblers.adreno import disasm
        disasm(binary)
      else:
        # print the PTX for NVIDIA. TODO: probably broken for everything else
        print(binary.decode('utf-8'))
    if self.argdtypes is not None: self.clprg.set_scalar_arg_dtypes(self.argdtypes)

  @staticmethod
  def max_work_group_size(): return CL.cl_ctx.devices[0].max_work_group_size

  def __call__(self, global_size, local_size, *bufs, wait=False) -> Optional[float]:
    e = self.clprg(CL.cl_queue, global_size, local_size, *[x._cl if isinstance(x, (CLBuffer, CLImage)) else x for x in bufs])
    if wait:
      CL.cl_queue.finish()
      return ((e.profile.end - e.profile.start) * OSX_TIMING_RATIO) * 1e-9
    return None

class CLCodegen(GPUCodegen):
  lang = GPULanguage(
    kernel_prefix = "__kernel", buffer_prefix = "__global ", smem_prefix = "__local ",
    half_prekernel = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable",
    barrier = "barrier(CLK_LOCAL_MEM_FENCE);", float4 = "(float4)",
    gid = [f'get_global_id({i})' for i in range(3)], lid = [f'get_local_id({i})' for i in range(3)])

class GPUBuffer(CompiledBuffer):
  spec = Specialized(CLBuffer, CLCodegen, CLProgram)
  # override this method for image
  def create_raw_buffer(self, shape, backing, dtype) -> RawBuffer:
    if len(shape) == 3 and shape[2] == 4 and IMAGE >= 2 and backing is None: return CLImage(shape)   # NOTE: this is a hack. we don't pass in the dtype here, it's controlled by the FLOAT16 env var
    else: return super().create_raw_buffer(shape, backing, dtype)
