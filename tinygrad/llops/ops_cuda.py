from typing import Optional, List
import pycuda.autoprimaryctx # type: ignore # pylint: disable=unused-import # noqa: F401
import pycuda.driver as cuda # type: ignore
from pycuda.compiler import compile # type: ignore
import numpy as np
from tinygrad.helpers import DEBUG, prod
from tinygrad.ops import CompiledBuffer, RawBuffer

class RawCUDABuffer(RawBuffer):
  def __init__(self, size): self._cl = cuda.mem_alloc(size)
  def copyin(self, b:np.ndarray, stream:Optional[cuda.Stream]=None): cuda.memcpy_htod_async(self._cl, b, stream)
  def copyout(self, a:np.ndarray): cuda.memcpy_dtoh(a, self._cl)

class CUDAProgram:
  def __init__(self, name:str, prg:str, binary=False):
    if not binary: prg = compile(prg, target="ptx", no_extern_c=True).decode('utf-8')
    if DEBUG >= 5: print(prg)
    # TODO: name is wrong
    self.prg = cuda.module_from_buffer(prg.encode('utf-8')).get_function(prg.split(".visible .entry ")[1].split("(")[0])

  def __call__(self, global_size, local_size, *args, wait=False):
    local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
    global_size = global_size + [1] * (3 - len(global_size))
    assert all(x%y == 0 for x,y in zip(global_size, local_size)), f"local:{local_size} must divide global:{global_size}"
    global_size = [x//y for x,y in zip(global_size, local_size)]
    self.prg(*args, block=tuple(local_size), grid=tuple(global_size))

from tinygrad.compiler.gpu import GPUCodegen, GPULanguage

cuda_lang = GPULanguage(
  kernel_prefix = "__global__", smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4",
  gid = [f'blockDim.{chr(120+i)}*blockIdx.{chr(120+i)}+threadIdx.{chr(120+i)}' for i in range(3)],
  lid = [f'threadIdx.{chr(120+i)}' for i in range(3)])

class CUDABuffer(CompiledBuffer):
  @staticmethod
  def create_raw_buffer(shape) -> RawBuffer: return RawCUDABuffer(4*prod(shape))
  @staticmethod
  def compile(ast, output_buffer):
    k = GPUCodegen(ast, output_buffer, cuda_lang)
    return (k.codegen().build(CUDAProgram), k.bufs, k.ret)
