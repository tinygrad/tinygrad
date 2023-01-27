import os, functools, time, platform
import numpy as np
import pyopencl as cl  # type: ignore
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from tinygrad.ops import DEBUG, GlobalCounters

OSX = platform.system() == "Darwin"
CLCACHE = int(os.getenv("CLCACHE", "1"))
FLOAT16 = int(os.getenv("FLOAT16", "0"))

class CL:
  CACHE, kernel_count, mem_used, time_sum, ops_sum = None, -1, 0, 0.0, 0.0
  BUFFER_CACHE : Dict[int, List[cl.Buffer]] = defaultdict(list)
  cl_ctx : Optional[cl.Context] = None
  cl_queue : Optional[cl.CommandQueue] = None
  def __init__(self):
    if CL.cl_queue is not None: return   # already initted
    devices = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0:  # settle for CPU
      devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], [])
    CL.cl_ctx = cl.Context(devices=[devices[int(os.getenv("CL_DEVICE", "0"))]])
    if len(devices) > 1 or DEBUG >= 1: print(f"using {CL.cl_ctx.devices}")
    CL.cl_queue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # this is an in-order command queue

class CLBuffer:
  def __init__(self, size):
    if len(CL.BUFFER_CACHE[size]) > 0:
      self.cl = CL.BUFFER_CACHE[size].pop()
    else:
      # TODO: on GPU OOM, clear the cache
      self.cl = cl.Buffer(CL().cl_ctx, cl.mem_flags.READ_WRITE, size)
      CL.mem_used += self.cl.size

  def __del__(self):
    if CLCACHE: CL.BUFFER_CACHE[self.cl.size].append(self.cl)
    else: CL.mem_used -= self.cl.size

  def copyin(self, b:np.ndarray):
    assert CL.CACHE is None, f"can't copy in {b} -> {self.cl} while caching"
    if DEBUG >= 1: print(f"**CL**        copy in {b.shape}")
    cl.enqueue_copy(CL().cl_queue, self.cl, b, is_blocking=False)

  def copyout(self, a:np.ndarray):
    assert CL.CACHE is None, f"can't copy out {self.cl} -> {a} while caching"
    if DEBUG >= 1: print(f"**CL**        copy out {a.shape}")
    cl.enqueue_copy(CL().cl_queue, a, self.cl, is_blocking=True)

class CLImage:
  fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.HALF_FLOAT if FLOAT16 else cl.channel_type.FLOAT)

  def __init__(self, shape):
    self.cl = cl.Image(CL().cl_ctx, cl.mem_flags.READ_WRITE, CLImage.fmt, shape=(shape[1], shape[0]))
    CL.mem_used += self.cl.row_pitch * self.cl.height

  def __del__(self):
    CL.mem_used -= self.cl.row_pitch * self.cl.height

@functools.lru_cache(maxsize=None)
class CLProgram:
  kernel_cnt : Dict[str, int] = defaultdict(int)
  def __init__(self, name:str, prg:str, options:Tuple[str, ...]=tuple(), argdtypes=None, rename=True, binary=False, op_estimate=0):
    self.name = f"{name}{('_N'+str(CLProgram.kernel_cnt[name])) if CLProgram.kernel_cnt[name] else ''}" if rename else name
    self.prg, self.options, self.argdtypes, self.op_estimate = prg.replace(f"{name}(", f"{self.name}(") if rename else prg, options, argdtypes, op_estimate
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
    CL.kernel_count += 1
    if DEBUG >= 4: print(args[0], args[1], self.prg)
    if OSX and DEBUG >= 2: st = time.monotonic_ns()
    if CL.CACHE is not None: CL.CACHE.append((self, args))
    else: e = self.clprg(CL().cl_queue, *args)
    if DEBUG >= 2:
      CL.cl_queue.finish()
      # NOTE: Profiling is (sadly) broken in OS X, so we take the real kernel time
      # BOUNTY: will paypal $50 to anyone who fixes this
      et = (time.monotonic_ns() - st) if OSX else (e.profile.end - e.profile.start)
    if DEBUG >= 1:
      CL.time_sum += 0 if DEBUG <= 1 or CL.CACHE is not None else et
      CL.ops_sum += self.op_estimate
      print(f"**CL** {CL.kernel_count:6d} {self.name:28s} args {len(args[2:]):5d}  kernels {str(args[0]):18s} {str(args[1]):12s} OPs {self.op_estimate/1e6:7.1f}M/{CL.ops_sum/1e9:7.2f}G  mem {CL.mem_used/1e9:5.2f} GB " +
            (str() if DEBUG <= 1 or CL.CACHE is not None else f"tm {et/1e3:9.2f}us/{CL.time_sum/1e6:9.2f}ms ({self.op_estimate/et:8.2f} GFLOPS)"))
    GlobalCounters.global_ops += self.op_estimate
    GlobalCounters.global_mem += sum([x.size//4 for x in args[2:] if isinstance(x, cl.Buffer)])
    return e if CL.CACHE is None else None
