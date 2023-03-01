from __future__ import annotations
import platform
import numpy as np
import pyopencl as cl  # type: ignore
from typing import Dict, Optional, Tuple, List, ClassVar, Final
from collections import defaultdict
from typing import Tuple, Optional, Union
from tinygrad.helpers import prod, IMAGE, DEBUG, getenv
from tinygrad.ops import UnaryOps, MovementOps, LazyOp, CompiledBuffer, GlobalCounters, RawBuffer

OSX = platform.system() == "Darwin"
OSX_TIMING_RATIO = (125/3) if OSX else 1.0   # see test/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something
CLCACHE = getenv("CLCACHE", 1)
FLOAT16 = getenv("FLOAT16", 0)

class CL:
  BUFFER_CACHE : ClassVar[Dict[int, List[cl.Buffer]]] = defaultdict(list)
  cl_ctx : ClassVar[Optional[cl.Context]] = None
  cl_queue : ClassVar[Optional[cl.CommandQueue]] = None
  def __init__(self) -> None:
    if CL.cl_queue is not None: return   # already initted
    devices : List[cl.Device] = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0: devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], []) # settle for CPU
    CL.cl_ctx = cl.Context(devices=[devices[getenv("CL_DEVICE", 0)]])
    if len(devices) > 1 or DEBUG >= 1: print(f"using {CL.cl_ctx.devices}")
    CL.cl_queue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # this is an in-order command queue

class CLBuffer(RawBuffer):
  def __init__(self, size):
    if len(CL.BUFFER_CACHE[size]) > 0:
      self._cl = CL.BUFFER_CACHE[size].pop()
    else:
      # TODO: on GPU OOM, clear the cache
      self._cl = cl.Buffer(CL().cl_ctx, cl.mem_flags.READ_WRITE, size)
      GlobalCounters.mem_used += self._cl.size

  def __del__(self):
    if CLCACHE: CL.BUFFER_CACHE[self._cl.size].append(self._cl)
    else: GlobalCounters.mem_used -= self._cl.size

  def copyin(self, b:np.ndarray): cl.enqueue_copy(CL().cl_queue, self._cl, b, is_blocking=False)
  def copyout(self, a:np.ndarray): cl.enqueue_copy(CL().cl_queue, a, self._cl, is_blocking=True)

class CLImage(RawBuffer):
  fmt : Final = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.HALF_FLOAT if FLOAT16 else cl.channel_type.FLOAT)
  IMAGE : Final = True

  def __init__(self, shape):
    self._cl = cl.Image(CL().cl_ctx, cl.mem_flags.READ_WRITE, CLImage.fmt, shape=(shape[1], shape[0]))
    GlobalCounters.mem_used += self._cl.row_pitch * self._cl.height

  def __del__(self): GlobalCounters.mem_used -= self._cl.row_pitch * self._cl.height

class CLProgram:
  def __init__(self, prg:str, binary=False):
    self.prg = prg
    self.clprogram = cl.Program(CL().cl_ctx, CL().cl_ctx.devices, [self.prg]) if binary else cl.Program(CL().cl_ctx, self.prg)  # type: ignore
    try:
      self._clprg = self.clprogram.build()
    except cl.RuntimeError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", self.prg)
      raise e
    self.clprg = self._clprg.__getattr__(self._clprg.kernel_names)
    if DEBUG >= 5 and not OSX: print(self.clprogram.get_info(cl.program_info.BINARIES)[0].decode('utf-8'))  # print the PTX for NVIDIA. TODO: probably broken for everything else
    if DEBUG >= 4: print(self.prg)
  def __call__(self, global_size, local_size, *bufs, wait=False) -> Optional[float]:
    e = self.clprg(CL().cl_queue, global_size, local_size, *[x._cl for x in bufs])
    if wait:
      CL().cl_queue.finish()
      return ((e.profile.end - e.profile.start) * OSX_TIMING_RATIO) * 1e-9

from tinygrad.compiler.cl import CLASTKernel
class OpenCLProgram(CLASTKernel):
  kernel_prefix = "__kernel"
  buffer_prefix = "__global "
  smem_prefix = "__local "
  barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  float4 = "(float4)"
  gid = [f'get_global_id({i})' for i in range(3)]
  lid = [f'get_local_id({i})' for i in range(3)]
  runtime = staticmethod(CLProgram)

class GPUBuffer(CompiledBuffer):
  @staticmethod
  def create_raw_buffer(shape): return CLImage(shape) if (len(shape) == 3 and shape[2] == 4 and IMAGE >= 2) else CLBuffer(4*prod(shape))
  compiler = staticmethod(OpenCLProgram)
