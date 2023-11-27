from __future__ import annotations
from typing import Tuple, Optional, Union
import ctypes, functools
import gpuctypes.opencl as cl
from tinygrad.helpers import to_char_p_p
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.opencl import OpenCLRenderer

def check(status, info:Optional[str]=None):
  if status != 0: raise RuntimeError(f"OpenCL Error {status}" + (("\n\n"+info) if info else ""))

def compile_cl(prg:str) -> bytes:
  assert CLDevice.compiler_context is not None, 'OpenCL requires a "compiler_context" to compile, init a device before you call this'
  prg = prg.encode()
  program = cl.clCreateProgramWithSource(CLDevice.compiler_context.context, 1, to_char_p_p([prg]), (ctypes.c_size_t * 1)(len(prg)), ctypes.byref(status := ctypes.c_int32()))
  check(status.value)
  status = cl.clBuildProgram(program, 1, ctypes.byref(CLDevice.compiler_context.device), None, ctypes.cast(None, cl.clBuildProgram.argtypes[4]), None)
  if status != 0:
    cl.clGetProgramBuildInfo(program, CLDevice.compiler_context.device, cl.CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(log_size := ctypes.c_size_t()))
    cl.clGetProgramBuildInfo(program, CLDevice.compiler_context.device, cl.CL_PROGRAM_BUILD_LOG, log_size.value, mstr := ctypes.create_string_buffer(log_size.value), None)
    check(status, ctypes.string_at(mstr, size=log_size.value).decode())
  binary_sizes = (ctypes.c_size_t * 1)()
  check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARY_SIZES, ctypes.sizeof(binary_sizes), ctypes.byref(binary_sizes), None))
  binary = ctypes.create_string_buffer(binary_sizes[0])
  binary_pointers = (ctypes.c_char_p * 1)(ctypes.addressof(binary))
  check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARIES, ctypes.sizeof(binary_pointers), ctypes.byref(binary_pointers), None))
  check(cl.clReleaseProgram(program))
  return bytes(binary)

class CLBuffer:
  def __init__(self, device:CLDevice, size:int):
    self.device, self._buf = device, cl.clCreateBuffer(device.context, cl.CL_MEM_READ_WRITE, size, None, ctypes.byref(status := ctypes.c_int32()))
    check(status.value)
  def __del__(self): check(cl.clReleaseMemObject(self._buf))

class CLProgram:
  def __init__(self, device:CLDevice, name:str, prg:bytes):
    self.program = cl.clCreateProgramWithBinary(device.context, 1, ctypes.byref(device.device), (ctypes.c_size_t * 1)(len(prg)),
                                                to_char_p_p([prg], ctypes.c_ubyte),
                                                ctypes.byref(binary_status := ctypes.c_int32()), ctypes.byref(errcode_ret := ctypes.c_int32()))
    check(binary_status.value)
    check(errcode_ret.value)
    self.kernel = cl.clCreateKernel(self.program, name.encode(), ctypes.byref(status := ctypes.c_int32()))
    check(status.value)

  def __del__(self):
    check(cl.clReleaseKernel(self.kernel))
    check(cl.clReleaseProgram(self.program))

  def __call__(self, *bufs:Union[CLBuffer, int], global_size:Tuple[int,...], local_size:Optional[Tuple[int,...]]=None, wait=False) -> Optional[float]:
    pass

class CLDevice:
  linearizer_opts, renderer, compiler = LinearizerOptions(), staticmethod(OpenCLRenderer), staticmethod(compile_cl)    # these are the same for all instantiations of the device

  devices = None                 # this is global and only initted once
  compiler_context = None        # this is the first created context. we make an assumption they are all the same for the compiler
  def __init__(self, device:str=""):
    if CLDevice.devices is None:
      check(cl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms := ctypes.c_uint32())))
      check(cl.clGetPlatformIDs(num_platforms.value, platform_array := (cl.cl_platform_id * num_platforms.value)(), None))
      check(cl.clGetDeviceIDs(platform_array[0], cl.CL_DEVICE_TYPE_DEFAULT, 0, None, ctypes.byref(num_devices := ctypes.c_uint32())))
      CLDevice.devices = (cl.cl_device_id * num_devices.value)()
      check(cl.clGetDeviceIDs(platform_array[0], cl.CL_DEVICE_TYPE_DEFAULT, num_devices, CLDevice.devices, None))
    self.device = CLDevice.devices[0 if ":" not in device else int(device.split(":")[1])]
    self.context = cl.clCreateContext(None, 1, ctypes.byref(self.device), ctypes.cast(None, cl.clCreateContext.argtypes[3]), None, ctypes.byref(status := ctypes.c_int32()))
    if CLDevice.compiler_context is None: CLDevice.compiler_context = self
    check(status.value)
    self.q = cl.clCreateCommandQueue(self.context, self.device, 0, ctypes.byref(status))
    check(status.value)

    # these both require a specific device
    self.buffer = functools.partial(CLBuffer, self)
    self.runtime = functools.partial(CLProgram, self)