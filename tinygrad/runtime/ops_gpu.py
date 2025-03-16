from __future__ import annotations
from typing import Optional, cast
import ctypes, functools, hashlib, contextlib
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.runtime.autogen import intel_ocloc as ocloc
from tinygrad.helpers import init_c_var, to_char_p_p, from_mv, OSX, DEBUG, getenv, mv_address
from tinygrad.renderer.cstyle import OpenCLRenderer, IntelRenderer
from tinygrad.device import BufferSpec, LRUAllocator, Compiled, Compiler, CompileError

# see test/external/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something
OSX_TIMING_RATIO = (125/3) if OSX else 1.0

cl_errors = {attr: k for k in dir(cl) if k.startswith("CL_") and isinstance(attr:=getattr(cl, k), int) and attr <= 0}
def check(status):
  if status != 0: raise RuntimeError(f"OpenCL Error {status}: {cl_errors.get(status, 'Unknown error')}")
def checked(ret, status): return (check(status.value), ret)[1]


"""
gpu_arch: xe-hpc(Max 1100), acm-g10(A770) (for more information see ocloc compile --help)

@brief: ocloc is a tool for managing Intel Compute GPU device binary format.
        It can be used for generation (as part of 'compile' command) as well as
        manipulation (decoding/modifying - as part of 'disasm'/'asm' commands) of such
        binary files.Intel Compute GPU device binary is a format used by Intel Compute GPU runtime
        (aka NEO). Intel Compute GPU runtime will return this binary format when queried
        using clGetProgramInfo(..., CL_PROGRAM_BINARIES, ...). It will also honor
        this format as input to clCreateProgramWithBinary function call.
        ocloc does not require Intel GPU device to be present in the system nor does it
        depend on Intel Compute GPU runtime driver to be installed. It does however rely
        on the same set of compilers (IGC, common_clang) as the runtime driver. 
"""
class IntelOfflineCompiler: 
  def __init__(self):
   pass
  def compile(self, cl_kernel:str, gpu_arch:str) -> bytes:
    # prepare ocoloc paramter and cast to proper ctype object for function call
    ocloc_arguments = [b"ocloc", b"compile", b"-file", b"kernel.cl", b"-device", gpu_arch.encode('utf-8'), b"-o", b"kernel.bin"]
    ocloc_arguments_carray = (ctypes.c_char_p * (len(ocloc_arguments)))(*ocloc_arguments)
    ocloc_arguments_len_param = ctypes.c_uint32(len(ocloc_arguments))
    ocloc_arguments_param = ctypes.cast(ocloc_arguments_carray, (ctypes.POINTER(ctypes.POINTER(ctypes.c_char))))
    # create byte array for (in-memory) kernel source code which is directly provided to ocloc lib 
    cl_kernel_bytes = (cl_kernel + "\0").encode('utf-8')
    cl_kernel_carray = (ctypes.c_ubyte * (len(cl_kernel_bytes)))(*cl_kernel_bytes)
    cl_kernel_param = (ctypes.cast(cl_kernel_carray, (ctypes.POINTER((ctypes.c_ubyte)))))
    cl_kernel_len = ctypes.c_uint64(len(cl_kernel_bytes))
    cl_kernel_len_param = ctypes.byref(cl_kernel_len)
    # create array with all provided (in-memory) kernel files which are matched with in args provided name (see -file kernel.cl)
    # provided_kernels_param shall match kernel.cl which is provided in ocloc_arguments (it's because libocloc api)
    provided_kernels_carray = (ctypes.c_char_p * 1)(b"kernel.cl") 
    provided_kernels_param = ctypes.cast(provided_kernels_carray, (ctypes.POINTER(ctypes.POINTER(ctypes.c_char))))
    # create output paramters 
    num_outputs_param = ctypes.c_uint32(0)
    data_outputs_param = ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))()
    len_outputs_param = ctypes.POINTER(ctypes.c_uint64)()
    name_outputs_param = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))()
    # compile and check result/output 
    ocloc_retcode = ocloc.oclocInvoke(ocloc_arguments_len_param, ocloc_arguments_param, ctypes.c_uint32(1), 
                            cl_kernel_param, cl_kernel_len_param, provided_kernels_param, 
                            0, None, None, None, ctypes.byref(num_outputs_param), ctypes.byref(data_outputs_param),
                            ctypes.byref(len_outputs_param), ctypes.byref(name_outputs_param))
    if ocloc_retcode != ocloc.OCLOC_SUCCESS:
      raise CompileError(f"Intel OpenCL Offline Compiler (ocloc) Error\n\n{ocloc._ocloc_error_t__enumvalues[ocloc_retcode]}")
    binary = bytes(ctypes.string_at(data_outputs_param[0], len_outputs_param[0]))
    # free memory which was internally allocated for output buffers
    ocloc_retcode = ocloc.oclocFreeOutput(ctypes.byref(num_outputs_param), ctypes.byref(data_outputs_param), ctypes.byref(len_outputs_param), ctypes.byref(name_outputs_param))
    if ocloc_retcode != ocloc.OCLOC_SUCCESS:
      print("Error: ocloc freeing memory failed!")
    return binary

class CLCompiler(Compiler):
  def __init__(self, dev:CLDevice, compile_key:str):
    self.dev = dev
    super().__init__(f"compile_cl_{compile_key}")
  def compile(self, src:str) -> bytes:
    program = checked(cl.clCreateProgramWithSource(self.dev.context, 1, to_char_p_p([src.encode()]), None, status := ctypes.c_int32()), status)
    build_status: int = cl.clBuildProgram(program, 1, self.dev.device_id, None, cl.clBuildProgram.argtypes[4](), None)
    if build_status != 0:
      cl.clGetProgramBuildInfo(program, self.dev.device_id, cl.CL_PROGRAM_BUILD_LOG, 0, None, log_size := ctypes.c_size_t())
      cl.clGetProgramBuildInfo(program, self.dev.device_id, cl.CL_PROGRAM_BUILD_LOG, log_size.value, mstr := ctypes.create_string_buffer(log_size.value), None)  # noqa: E501
      raise CompileError(f"OpenCL Compile Error\n\n{mstr.value.decode()}")
    check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARY_SIZES, ctypes.sizeof(ctypes.c_size_t), binary_sizes := (ctypes.c_size_t * 1)(), None))
    check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARIES, ctypes.sizeof(ctypes.c_void_p), (ctypes.c_void_p * 1)(ctypes.addressof(binary := ctypes.create_string_buffer(binary_sizes[0]))), None))  # noqa: E501
    check(cl.clReleaseProgram(program))
    return bytes(binary)

class CLProgram:
  def __init__(self, device:CLDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = device, name, lib
    self.program = checked(cl.clCreateProgramWithBinary(device.context, 1, device.device_id, (ctypes.c_size_t * 1)(len(lib)),
                                                        to_char_p_p([lib], ctypes.c_ubyte), binary_status := ctypes.c_int32(),
                                                        errcode_ret := ctypes.c_int32()), errcode_ret)
    check(binary_status.value)
    check(cl.clBuildProgram(self.program, 1, device.device_id, None, cl.clBuildProgram.argtypes[4](), None)) # NOTE: OSX requires this
    self.kernel = checked(cl.clCreateKernel(self.program, name.encode(), status := ctypes.c_int32()), status)

  def __del__(self):
    with contextlib.suppress(TypeError, AttributeError): check(cl.clReleaseKernel(self.kernel))
    with contextlib.suppress(TypeError, AttributeError): check(cl.clReleaseProgram(self.program))

  def __call__(self, *bufs:tuple[ctypes._CData, BufferSpec], global_size:tuple[int,int,int]=(1,1,1), local_size:Optional[tuple[int,int,int]]=None, vals:tuple[int, ...]=(), wait=False) -> Optional[float]:  # noqa: E501
    for i,(b,_) in enumerate(bufs): cl.clSetKernelArg(self.kernel, i, ctypes.sizeof(b), ctypes.byref(b))
    for i,v in enumerate(vals,start=len(bufs)): cl.clSetKernelArg(self.kernel, i, 4, ctypes.byref(ctypes.c_int32(v)))
    if local_size is not None: global_size = cast(tuple[int,int,int], tuple(int(g*l) for g,l in zip(global_size, local_size)))
    event = cl.cl_event() if wait else None
    check(cl.clEnqueueNDRangeKernel(self.dev.queue, self.kernel, len(global_size), None, (ctypes.c_size_t * len(global_size))(*global_size), (ctypes.c_size_t * len(local_size))(*local_size) if local_size else None, 0, None, event))  # noqa: E501
    if wait:
      assert event is not None
      check(cl.clWaitForEvents(1, event))
      check(cl.clGetEventProfilingInfo(event, cl.CL_PROFILING_COMMAND_START, 8, ctypes.byref(start := ctypes.c_uint64()), None))
      check(cl.clGetEventProfilingInfo(event, cl.CL_PROFILING_COMMAND_END, 8, ctypes.byref(end := ctypes.c_uint64()), None))
      return float(end.value-start.value) * OSX_TIMING_RATIO * 1e-9
    return None

class CLAllocator(LRUAllocator):
  def __init__(self, dev:CLDevice):
    self.dev = dev
    super().__init__()
  def _alloc(self, size:int, options:BufferSpec) -> tuple[ctypes._CData, BufferSpec]:
    if options.image is not None:
      return (checked(cl.clCreateImage2D(self.dev.context, cl.CL_MEM_READ_WRITE,
                                        cl.cl_image_format(cl.CL_RGBA, {2: cl.CL_HALF_FLOAT, 4: cl.CL_FLOAT}[options.image.itemsize]),
                                        options.image.shape[1], options.image.shape[0], 0, None, status := ctypes.c_int32()), status), options)
    return (checked(cl.clCreateBuffer(self.dev.context, cl.CL_MEM_READ_WRITE, size, None, status := ctypes.c_int32()), status), options)
  def _free(self, opaque:tuple[ctypes._CData, BufferSpec], options:BufferSpec): check(cl.clReleaseMemObject(opaque[0]))
  def _copyin(self, dest:tuple[ctypes._CData, BufferSpec], src:memoryview):
    if dest[1].image is not None:
      check(cl.clEnqueueWriteImage(self.dev.queue, dest[0], False, (ctypes.c_size_t * 3)(0,0,0),
                                   (ctypes.c_size_t * 3)(dest[1].image.shape[1],dest[1].image.shape[0],1), 0, 0, from_mv(src), 0, None, None))
    else:
      if mv_address(src) % 16: src = memoryview(bytearray(src))
      check(cl.clEnqueueWriteBuffer(self.dev.queue, dest[0], False, 0, len(src)*src.itemsize, from_mv(src), 0, None, None))
    self.dev.pending_copyin.append(src)    # NOTE: these can't be freed until the GPU actually executes this command
  def _copyout(self, dest:memoryview, src:tuple[ctypes._CData, BufferSpec]):
    if src[1].image is not None:
      check(cl.clEnqueueReadImage(self.dev.queue, src[0], False, (ctypes.c_size_t * 3)(0,0,0),
                                  (ctypes.c_size_t * 3)(src[1].image.shape[1],src[1].image.shape[0],1), 0, 0, from_mv(dest), 0, None, None))
    else:
      check(cl.clEnqueueReadBuffer(self.dev.queue, src[0], False, 0, len(dest)*dest.itemsize, from_mv(dest), 0, None, None))
    self.dev.synchronize()

class CLDevice(Compiled):
  device_ids = None                 # this is global and only initted once
  def __init__(self, device:str=""):
    if CLDevice.device_ids is None:
      check(cl.clGetPlatformIDs(0, None, num_platforms := ctypes.c_uint32()))
      check(cl.clGetPlatformIDs(num_platforms.value, platform_ids := (cl.cl_platform_id * num_platforms.value)(), None))
      for device_type in [cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_DEFAULT]:
        err = cl.clGetDeviceIDs(platform_ids[0], device_type, 0, None, num_devices := ctypes.c_uint32())
        if err == 0 and num_devices.value != 0: break
      if DEBUG >= 1: print(f"CLDevice: got {num_platforms.value} platforms and {num_devices.value} devices")
      CLDevice.device_ids = init_c_var((cl.cl_device_id * num_devices.value)(), lambda x: check(cl.clGetDeviceIDs(platform_ids[0], device_type, num_devices, x, None)))  # noqa: E501

    self.device_id = CLDevice.device_ids[0 if ":" not in device else int(device.split(":")[1])]
    self.device_name = (cl.clGetDeviceInfo(self.device_id, cl.CL_DEVICE_NAME, 256, buf := ctypes.create_string_buffer(256), None), buf.value.decode())[1]  # noqa: E501
    self.driver_version = (cl.clGetDeviceInfo(self.device_id, cl.CL_DRIVER_VERSION, 256, buf := ctypes.create_string_buffer(256), None), buf.value.decode())[1]  # noqa: E501
    if DEBUG >= 1: print(f"CLDevice: opening {self.device_name} with version {self.driver_version}")
    self.context = checked(cl.clCreateContext(None, 1, self.device_id, cl.clCreateContext.argtypes[3](), None, status := ctypes.c_int32()), status)
    self.queue = checked(cl.clCreateCommandQueue(self.context, self.device_id, cl.CL_QUEUE_PROFILING_ENABLE, status), status)
    self.pending_copyin: list[memoryview] = []
    self.device_exts = (cl.clGetDeviceInfo(self.device_id, cl.CL_DEVICE_EXTENSIONS, 4096, ctypes.byref(buf := ctypes.create_string_buffer(4096)), ctypes.byref(total := ctypes.c_size_t())), ctypes.string_at(buf, size=total.value).decode())[1]  # noqa: E501

    compile_key = hashlib.md5(self.device_name.encode() + self.driver_version.encode()).hexdigest()
    renderer = IntelRenderer() if "cl_intel_subgroup_matrix_multiply_accumulate" in self.device_exts and getenv("INTEL") else OpenCLRenderer()
    super().__init__(device, CLAllocator(self), renderer, CLCompiler(self, f"compile_cl_{compile_key}"), functools.partial(CLProgram, self))
  def synchronize(self):
    check(cl.clFinish(self.queue))
    self.pending_copyin.clear()

GPUDevice = CLDevice # for legacy reasons
