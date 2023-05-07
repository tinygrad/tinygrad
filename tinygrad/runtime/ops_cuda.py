import subprocess
from typing import Optional
import numpy as np
import pycuda.autoprimaryctx # type: ignore # pylint: disable=unused-import # noqa: F401
import pycuda.driver as cuda # type: ignore
from pycuda.compiler import compile as cuda_compile # type: ignore
from tinygrad.helpers import DEBUG
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

class RawCUDABuffer(RawBufferCopyInOut):
  def __init__(self, size, dtype): super().__init__(size, dtype, cuda.mem_alloc(size * dtype.itemsize))
  def _copyin(self, x:np.ndarray, stream:Optional[cuda.Stream]=None): cuda.memcpy_htod_async(self._buf, x, stream)
  def _copyout(self, x:np.ndarray): cuda.memcpy_dtoh(x, self._buf)

class CUDAProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if DEBUG >= 6:
        with open("/tmp/cubin", "wb") as f:
          f.write(cuda_compile(prg, target="cubin", no_extern_c=True))
        sass = subprocess.check_output(['nvdisasm', '/tmp/cubin']).decode('utf-8')
        print(sass)
      if not binary: prg = cuda_compile(prg, target="ptx", no_extern_c=True).decode('utf-8')
    except cuda.CompileError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 5: print(prg)
    # TODO: name is wrong, so we get it from the ptx using hacks
    self.prg = cuda.module_from_buffer(prg.encode('utf-8')).get_function(prg.split(".visible .entry ")[1].split("(")[0])

  def __call__(self, global_size, local_size, *args, wait=False):
    local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
    global_size = global_size + [1] * (3 - len(global_size))
    assert all(x%y == 0 for x,y in zip(global_size, local_size)), f"local:{local_size} must divide global:{global_size}"
    global_size = [x//y for x,y in zip(global_size, local_size)]
    if wait:
      start, end = cuda.Event(), cuda.Event()
      start.record()
    self.prg(*[x._buf for x in args], block=tuple(local_size), grid=tuple(global_size))
    if wait:
      end.record()
      end.synchronize()
      return start.time_till(end)*1e-3

class CUDACodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "__global__", smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4",
    half_prekernel = "#include <cuda_fp16.h>",
    gid = [f'blockDim.{chr(120+i)}*blockIdx.{chr(120+i)}+threadIdx.{chr(120+i)}' for i in range(3)],
    lid = [f'threadIdx.{chr(120+i)}' for i in range(3)])

CUDABuffer = Compiled(RawCUDABuffer, CUDACodegen, CUDAProgram, cuda.Context.synchronize)
