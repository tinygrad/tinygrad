from __future__ import annotations
import platform, pathlib
import numpy as np
import pyopencl as cl  # type: ignore
from typing import Optional, List
from tinygrad.helpers import DEBUG, getenv, prod, ImageDType
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

OSX = platform.system() == "Darwin"
OSX_TIMING_RATIO = (125/3) if OSX else 1.0   # see test/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something
FLOAT16 = getenv("FLOAT16", 0)

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
ROCM_LLVM_PATH = pathlib.Path("/opt/rocm/llvm/bin")
#ROCM_LLVM_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "extra/rocm/build/llvm-project/bin"
if DEBUG >= 5:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

class _CL:
  def __init__(self):
    platforms: List[List[cl.Device]] = [y for y in ([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()] + [x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()]) if len(y)]
    if DEBUG >= 1: print(f"using {platforms[getenv('CL_PLATFORM', 0)]}")
    self.cl_ctx: cl.Context = cl.Context(devices=[x for x in platforms[getenv('CL_PLATFORM', 0)] if x.name not in getenv('CL_EXCLUDE', "").split(",")])
    self.cl_queue: List[cl.CommandQueue] = [cl.CommandQueue(self.cl_ctx, device=device, properties=cl.command_queue_properties.PROFILING_ENABLE) for device in self.cl_ctx.devices]
  def synchronize(self):
    for q in self.cl_queue: q.finish()
CL = _CL()

# TODO: merge CLImage in here
class CLBuffer(RawBufferCopyInOut):
  def __init__(self, size, dtype, device='0'):
    if isinstance(dtype, ImageDType):
      fmt = cl.ImageFormat(cl.channel_order.RGBA, {2: cl.channel_type.HALF_FLOAT, 4: cl.channel_type.FLOAT}[dtype.itemsize])
      buf = cl.Image(CL.cl_ctx, cl.mem_flags.READ_WRITE, fmt, shape=(dtype.shape[1], dtype.shape[0]))
      assert size == prod(dtype.shape), f"image size mismatch {size} != {dtype.shape}"
      # NOTE: the memory is a bit off here due to padding, it's buf.row_pitch * buf.height * 4 * dtype.itemsize
    else:
      buf = cl.Buffer(CL.cl_ctx, cl.mem_flags.READ_WRITE, size * dtype.itemsize)
    setattr(buf, 'device', int(device))  # device is tracked on the underlying buffer
    super().__init__(size, dtype, buf)
  def _copyin(self, x:np.ndarray):
    assert not self.dtype.name.startswith("image"), f"can't copyin images {self.dtype}"
    cl.enqueue_copy(CL.cl_queue[self._buf.device], self._buf, x, is_blocking=False)
  def _copyout(self, x:np.ndarray):
    assert not self.dtype.name.startswith("image"), f"can't copyout images {self.dtype}"
    cl.enqueue_copy(CL.cl_queue[self._buf.device], x, self._buf, is_blocking=True)

class CLProgram:
  def __init__(self, name:str, prg:str, binary=False, argdtypes=None, options=None):
    self.name, self.argdtypes, self.clprogram = name, argdtypes, cl.Program(CL.cl_ctx, CL.cl_ctx.devices, [prg]*len(CL.cl_ctx.devices)) if binary else cl.Program(CL.cl_ctx, prg)  # type: ignore
    try:
      self._clprg = self.clprogram.build(options=options)
    except cl.RuntimeError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    self.clprg = self._clprg.__getattr__(name)
    if DEBUG >= 5 and not OSX:
      if 'Adreno' in CL.cl_ctx.devices[0].name:
        from disassemblers.adreno import disasm
        disasm(self.binary())
      elif 'gfx1100' in CL.cl_ctx.devices[0].name:
        asm = early_exec(([ROCM_LLVM_PATH / "llvm-objdump", '-d', '-'], self.binary()))
        print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))
      else:
        # print the PTX for NVIDIA. TODO: probably broken for everything else
        print(self.binary().decode('utf-8'))
    if self.argdtypes is not None: self.clprg.set_scalar_arg_dtypes(self.argdtypes)

  def binary(self): return self.clprogram.get_info(cl.program_info.BINARIES)[0]

  @staticmethod
  def max_work_group_size(): return CL.cl_ctx.devices[0].max_work_group_size

  def __call__(self, global_size, local_size, *bufs, wait=False) -> Optional[float]:
    cl_bufs = [x._buf if isinstance(x, CLBuffer) else x for x in bufs]
    e = self.clprg(CL.cl_queue[cl_bufs[0].device], global_size, local_size, *cl_bufs)
    if wait:
      e.wait()
      try:
        return ((e.profile.end - e.profile.start) * OSX_TIMING_RATIO) * 1e-9
      except cl.RuntimeError:   # no profiling info available
        return None
    return None

class CLCodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "#define int64 long\n__kernel", buffer_prefix = "__global ", smem_prefix = "__local ",
    half_prekernel = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable",
    barrier = "barrier(CLK_LOCAL_MEM_FENCE);", float4 = "(float4)",
    gid = [f'get_global_id({i})' for i in range(3)], lid = [f'get_local_id({i})' for i in range(3)], uses_vload=True)
  supports_float4_alu = True
  supports_float4 = True
GPUBuffer = Compiled(CLBuffer, CLCodegen, CLProgram, CL.synchronize)
