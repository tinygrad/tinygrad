import functools, platform
import numpy as np
import pyopencl as cl  # type: ignore
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from tinygrad.ops import DEBUG, GlobalCounters
from tinygrad.helpers import getenv

OSX = platform.system() == "Darwin"
OSX_TIMING_RATIO = (125/3) if OSX else 1.0   # see test/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something

CLCACHE = getenv("CLCACHE", 1)
FLOAT16 = getenv("FLOAT16", 0)

class CL:
  CACHE, mem_used = None, 0
  BUFFER_CACHE : Dict[int, List[cl.Buffer]] = defaultdict(list)
  cl_ctx : Optional[cl.Context] = None
  cl_queue : Optional[cl.CommandQueue] = None
  def __init__(self):
    if CL.cl_queue is not None: return   # already initted
    devices : List[cl.Device] = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0:  # settle for CPU
      devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], [])
    CL.cl_ctx = cl.Context(devices=[devices[getenv("CL_DEVICE", 0)]])
    if len(devices) > 1 or DEBUG >= 1: print(f"using {CL.cl_ctx.devices}")
    CL.cl_queue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # this is an in-order command queue

  @staticmethod
  def enqueue_copy(a, b, is_blocking=False):
    if CL.CACHE is not None: assert False, f"can't copy {a} -> {b} while caching"
    if DEBUG >= 1: print(f"**CL**        copy in {b.shape}" if isinstance(b, np.ndarray) else f"**CL**        copy OUT {a.shape}")
    cl.enqueue_copy(CL().cl_queue, a, b, is_blocking=is_blocking)

class CLBuffer:
  def __init__(self, size):
    if DEBUG >= 4: print(f"allocate GPU Buffer {size}")
    if len(CL.BUFFER_CACHE[size]) > 0:
      self._cl = CL.BUFFER_CACHE[size].pop()
    else:
      # TODO: on GPU OOM, clear the cache
      self._cl = cl.Buffer(CL().cl_ctx, cl.mem_flags.READ_WRITE, size)
      CL.mem_used += self._cl.size

  def __del__(self):
    if CLCACHE: CL.BUFFER_CACHE[self._cl.size].append(self._cl)
    else: CL.mem_used -= self._cl.size

  def copyin(self, b:np.ndarray): CL.enqueue_copy(self._cl, b, False)
  def copyout(self, a:np.ndarray): CL.enqueue_copy(a, self._cl, True)

class CLImage:
  fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.HALF_FLOAT if FLOAT16 else cl.channel_type.FLOAT)

  def __init__(self, shape):
    self._cl = cl.Image(CL().cl_ctx, cl.mem_flags.READ_WRITE, CLImage.fmt, shape=(shape[1], shape[0]))
    CL.mem_used += self._cl.row_pitch * self._cl.height

  def __del__(self):
    CL.mem_used -= self._cl.row_pitch * self._cl.height

  def copyin(self, b:np.ndarray): raise NotImplementedError("no copyin for CLImage")
  def copyout(self, a:np.ndarray): raise NotImplementedError("no copyout for CLImage")

@functools.lru_cache(maxsize=None)
class CLProgram:
  kernel_cnt : Dict[str, int] = defaultdict(int)
  def __init__(self, name:str, prg:str, options:Tuple[str, ...]=tuple(), argdtypes=None, rename=True, binary=False, op_estimate=0, mem_estimate=0):
    self.name = f"{name}{('_N'+str(CLProgram.kernel_cnt[name])) if CLProgram.kernel_cnt[name] else str()}" if rename else name
    self.prg, self.options, self.argdtypes, self.op_estimate, self.mem_estimate = prg.replace(f"{name}(", f"{self.name}(") if rename else prg, options, argdtypes, op_estimate, mem_estimate
    self.clprogram = cl.Program(CL().cl_ctx, CL().cl_ctx.devices, [self.prg]) if binary else cl.Program(CL().cl_ctx, self.prg)  # type: ignore
    try:
      self.clprg = self.clprogram.build(options=list(self.options)).__getattr__(self.name)
    except cl.RuntimeError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", self.prg)
      raise e
    if self.argdtypes is not None:
      self.clprg.set_scalar_arg_dtypes(self.argdtypes)
    CLProgram.kernel_cnt[name] += 1
  def __call__(self, *args):
    if DEBUG >= 4: print(args[0], args[1], self.prg)
    # print the PTX for NVIDIA. TODO: probably broken for everything else
    if DEBUG >= 5: print(self.clprogram.get_info(cl.program_info.BINARIES)[0].decode('utf-8'))
    if CL.CACHE is not None: CL.CACHE.append((self, args))
    else: e = self.clprg(CL().cl_queue, *args)
    if DEBUG >= 2 and CL.CACHE is None:
      CL.cl_queue.finish()
      # NOTE: Profiling is not in ns in OS X, we multiply by a computed ratio
      et = (e.profile.end - e.profile.start) * OSX_TIMING_RATIO
      GlobalCounters.time_sum += et
    if DEBUG >= 1:
      print(f"**CL** {GlobalCounters.kernel_count:6d} {self.name:28s} args {len(args[2:]):5d}  kernels {str(args[0]):18s} {str(args[1]):12s} OPs {self.op_estimate/1e6:7.1f}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {CL.mem_used/1e9:5.2f} GB " +
            (str() if DEBUG <= 1 or CL.CACHE is not None else f"tm {et/1e3:9.2f}us/{GlobalCounters.time_sum/1e6:9.2f}ms ({self.op_estimate/et:8.2f} GFLOPS)"))
    GlobalCounters.log_kernel(self.op_estimate, self.mem_estimate)
    return e if CL.CACHE is None else None
