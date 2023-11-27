from typing import Tuple, Optional, Union
import ctypes
import gpuctypes.opencl as cl

def check(status):
  if status != 0: raise RuntimeError(f"OpenCL Error {status}")

class CLDevice:
  device_array = None   # this is global
  compilation_context = None   # this is the first created device. we make an assumption they are all the same
  def __init__(self, device:str):
    if CLDevice.device_array is None:
      check(cl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms := ctypes.c_uint32())))
      check(cl.clGetPlatformIDs(num_platforms.value, platform_array := (cl.cl_platform_id * num_platforms.value)(), None))
      check(cl.clGetDeviceIDs(platform_array[0], cl.CL_DEVICE_TYPE_DEFAULT, 0, None, ctypes.byref(num_devices := ctypes.c_uint32())))
      CLDevice.device_array = (cl.cl_device_id * num_devices.value)()
      check(cl.clGetDeviceIDs(platform_array[0], cl.CL_DEVICE_TYPE_DEFAULT, num_devices, CLDevice.device_array, None))
    self.num = 0 if ":" not in device else int(device.split(":")[1])
    self.context = cl.clCreateContext(None, 1, CLDevice.device_array[self.num:self.num+1], ctypes.cast(None, cl.clCreateContext.argtypes[3]), None, ctypes.byref(status := ctypes.c_int32()))
    if CLDevice.compilation_context is None: CLDevice.compilation_context = self.context
    check(status.value)
    self.q = cl.clCreateCommandQueue(self.context, CLDevice.device_array[self.num], 0, ctypes.byref(status))
    check(status.value)

def compile_cl(prg:str) -> bytes:
  assert CLDevice.compilation_context is not None, 'OpenCL requires a "compilation_context", init a device before you call this'
  prg = prg.encode()
  program = cl.clCreateProgramWithSource(CLDevice.compilation_context, 1, (ctypes.c_size_t * 1)(len(prg)), ctypes.byref(status := ctypes.c_int32()))
  check(status.value)
  binary_sizes = (ctypes.c_size_t * 1)()
  cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARY_SIZES, ctypes.sizeof(binary_sizes), ctypes.byref(binary_sizes), None)
  cl.clReleaseProgram(program)

class CLBuffer:
  def __init__(self, device:CLDevice, size:int):
    self.device, self._buf = device, cl.clCreateBuffer(device.context, cl.CL_MEM_READ_WRITE, size, None, ctypes.byref(status := ctypes.c_int32()))
    check(status.value)
  def __del__(self): check(cl.clReleaseMemObject(self._buf))

class CLProgram:
  def __init__(self, name:str, prg:bytes):
    self.program = cl.clCreateProgramWithBinary
    self.kernel = cl.clCreateKernel(self.program, name.encode(), ctypes.byref(status := ctypes.c_int32()))
    check(status.value)

  def __del__(self):
    check(cl.clReleaseKernel(self.kernel))
    check(cl.clReleaseProgram(self.program))

  def __call__(self, *bufs:Union[CLBuffer, int], global_size:Tuple[int,...], local_size:Optional[Tuple[int,...]]=None, wait=False) -> Optional[float]:
    pass
