import numpy as np
import ctypes
import extra.hip_wrapper as hip
from tinygrad.helpers import DEBUG
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 5:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

# The default HIP stream is used for everything.

class RawHIPBuffer(RawBufferCopyInOut):
  def __init__(self, size, dtype):
    self.buf_sz = size * dtype.itemsize
    super().__init__(size, dtype, hip.hipMalloc(self.buf_sz))
  def _copyin(self, x:np.ndarray): hip.hipMemcpyAsync_htod(self._buf, x.ctypes.data, self.buf_sz, 0)
  def _copyout(self, x:np.ndarray): hip.hipMemcpyAsync_dtoh(x.ctypes.data, self._buf, self.buf_sz, 0)

class HIPProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if not binary:
        prog = hip.hiprtcCreateProgram(prg, name, [], [])
        device_properties = hip.hipGetDeviceProperties(0)
        hip.hiprtcCompileProgram(prog, [f'--offload-arch={device_properties.gcnArchName}'])
        prg = hip.hiprtcGetCode(prog)
    except Exception as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 5:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    module = hip.hipModuleLoadData(prg)
    self.prg = hip.hipModuleGetFunction(module, name)

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait:
      start, end = hip.hipEventCreate(), hip.hipEventCreate()
      hip.hipEventRecord(start)
    class PackageStruct(ctypes.Structure):
      _fields_ = [(f'field{idx}', ctypes.c_void_p) for idx in range(len(args))]
    struct = PackageStruct(*[data._buf for data in args])
    hip.hipModuleLaunchKernel(self.prg, global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, 0, struct)
    if wait:
      hip.hipEventRecord(end)
      hip.hipEventSynchronize(end)
      return hip.hipEventElapsedTime(start, end)*1e-3

class HIPCodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "#define INFINITY (__builtin_inff())\nextern \"C\" __global__", smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4",
    half_prekernel = "",
    gid = [f'blockIdx.{chr(120+i)}' for i in range(3)],
    lid = [f'threadIdx.{chr(120+i)}' for i in range(3)])

HIPBuffer = Compiled(RawHIPBuffer, HIPCodegen, HIPProgram, hip.hipDeviceSynchronize)
