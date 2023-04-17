from typing import Optional
import numpy as np
import ctypes
from pyhip import hip, hiprtc # type: ignore
from tinygrad.helpers import DEBUG
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

class RawHIPBuffer(RawBufferCopyInOut):
  def __init__(self, size, dtype):
    super().__init__(size, dtype, hip.hipMalloc(size * dtype.itemsize))
    self.buf_sz = size * dtype.itemsize
  def _copyin(self, x:np.ndarray): hip.hipMemcpy_htod(self._buf, x.ctypes.data, self.buf_sz)
  def _copyout(self, x:np.ndarray): hip.hipMemcpy_dtoh(x.ctypes.data, self._buf, self.buf_sz)

class HIPProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if not binary:
        prog = hiprtc.hiprtcCreateProgram(prg, name, [], [])
        device_properties = hip.hipGetDeviceProperties(0)
        hiprtc.hiprtcCompileProgram(prog, [f'--offload-arch={device_properties.gcnArchName}'])
        prg = hiprtc.hiprtcGetCode(prog)
    except hip.hipError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 5: print(prg)
    module = hip.hipModuleLoadData(prg)

    self.prg = hip.hipModuleGetFunction(module, name)

  @staticmethod
  def numpy_to_ctypes_struct_field(idx, data):
    assert type(data) == RawHIPBuffer, "Data type error " + str(type(data))
    return (f'field{idx}', ctypes.c_void_p)

  def __call__(self, global_size, local_size, *args, wait=False):
    local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
    global_size = global_size + [1] * (3 - len(global_size))
    assert all(x%y == 0 for x,y in zip(global_size, local_size)), f"local:{local_size} must divide global:{global_size}"
    global_size = [x//y for x,y in zip(global_size, local_size)]
    if wait:
      start, end = hip.hipEventCreate(), hip.hipEventCreate()
      hip.hipEventRecord(start)
    class PackageStruct(ctypes.Structure):
      _fields_ = [HIPProgram.numpy_to_ctypes_struct_field(idx, data) for idx, data in enumerate(args)]
    struct = PackageStruct(*[data._buf for data in args])
    hip.hipModuleLaunchKernel(self.prg, global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, 0, struct)
    if wait:
      hip.hipEventRecord(end)
      hip.hipEventSynchronize(end)
      return hip.hipEventElapsedTime(start, end)*1e-3

class HIPCodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "#define INFINITY (__builtin_inff())\nextern \"C\" __global__", smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4",
    half_prekernel = "#include <cuda_fp16.h>",
    gid = [f'blockDim.{chr(120+i)}*blockIdx.{chr(120+i)}+threadIdx.{chr(120+i)}' for i in range(3)],
    lid = [f'threadIdx.{chr(120+i)}' for i in range(3)])

HIPBuffer = Compiled(RawHIPBuffer, HIPCodegen, HIPProgram)
