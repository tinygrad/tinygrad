from __future__ import annotations
import ctypes, hashlib, functools
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.runtime.support import c
from tinygrad.helpers import to_char_p_p, from_mv, cpu_profile, OSX, DEBUG, suppress_finalizing, unwrap, round_up, is_image_shape
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.device import Device, Buffer, BufferStorage, BufferSpec, HostAllocator, Compiled, Compiler, CompileError, MMIOInterface, HCQ_RUNTIME_DEV
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo
from tinygrad.engine.realize import get_call_kernel_args
from tinygrad.runtime.support.hcq2 import ccall, cstruct, layout_args, patch, make_program

CC_CB = c.CFUNCTYPE[None, [c.POINTER[ctypes.c_char], c.POINTER[None], cl.size_t, c.POINTER[None]]]
BP_CB = c.CFUNCTYPE[None, [cl.cl_program, c.POINTER[None]]]

cl_errors = {attr: k for k in dir(cl) if k.startswith("CL_") and isinstance(attr:=getattr(cl, k), int) and attr <= 0}
def check(status):
  if status != 0: raise RuntimeError(f"OpenCL Error {status}: {cl_errors.get(status, 'Unknown error')}")
def checked(ret, status): return (check(status.value), ret)[1]

class CLCompiler(Compiler):
  def __init__(self, dev:CLDevice, compile_key:str):
    self.dev = dev
    super().__init__(f"compile_cl_{compile_key}")
  def compile(self, src:str) -> bytes:
    program = checked(cl.clCreateProgramWithSource(self.dev.context, 1, to_char_p_p([src.encode()]), None, status := ctypes.c_int32()), status)
    build_status: int = cl.clBuildProgram(program, 1, self.dev.cl_dev, None, BP_CB(), None)
    if build_status != 0:
      cl.clGetProgramBuildInfo(program, self.dev.cl_dev, cl.CL_PROGRAM_BUILD_LOG, 0, None, log_size := ctypes.c_size_t())
      cl.clGetProgramBuildInfo(program, self.dev.cl_dev, cl.CL_PROGRAM_BUILD_LOG,
                               log_size.value, mstr := ctypes.create_string_buffer(log_size.value), None)
      raise CompileError(f"OpenCL Compile Error\n\n{mstr.value.decode()}")
    check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARY_SIZES, ctypes.sizeof(ctypes.c_size_t), binary_sizes := (ctypes.c_size_t * 1)(), None))
    check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARIES, ctypes.sizeof(ctypes.c_void_p),
                              (ctypes.c_void_p * 1)(ctypes.addressof(binary := ctypes.create_string_buffer(binary_sizes[0]))), None))
    check(cl.clReleaseProgram(program))
    return bytes(binary)

# *****************
# queue

def launch(ctx:tuple[CLDevice, UOp, UOp], call:UOp, prg:UOp) -> UOp:
  (dev, rt, h), args, z = ctx, get_call_kernel_args(call, prg, (ctx[0].device,)), UOp.const(0, dtypes.uint64)
  kernel = (p:=make_program(prg, 8, dev.device)).after(p.store(prg)).bitcast(dtypes.u64).index(0).load()
  imgs = {i: ccall(cl.clCreateImage, rt.index(0).load(), UOp.const(cl.CL_MEM_READ_WRITE, dtypes.uint64),
                   cstruct(cl.cl_image_format, image_channel_order=cl.CL_RGBA,
                           image_channel_data_type={2:cl.CL_HALF_FLOAT, 4:cl.CL_FLOAT}[dt.itemsize]).index(0),
                   cstruct(cl.cl_image_desc, image_type=cl.CL_MEM_OBJECT_IMAGE2D, image_width=shape[1], image_height=shape[0], buffer=args[i],
                           image_row_pitch=round_up(shape[1], 256 if OSX else 1) * 4 * dt.itemsize).index(0), z, z)
          for i, (_, _, dt, shape) in enumerate(prg.to_elf().signature) if is_image_shape(shape)}
  dims = [UOp.const(d, dtypes.uint64) for d in (*[g * l for g, l in zip(prg.arg.global_size, prg.arg.local_size)], *prg.arg.local_size)]
  rows = layout_args([imgs.get(i, a) for i, a in enumerate(args)] + dims) # [args][global][local]
  buf = patch(UOp.alloc((rows[-1][0] + 8,), dtypes.uint8, device=HCQ_RUNTIME_DEV.device).rtag("kernargs"), rows).after(h)
  sets = [ccall(cl.clSetKernelArg, kernel, i, UOp.const(w.dtype.itemsize, dtypes.uint64), buf.index(o)) for i, (o, w) in enumerate(rows[:len(args)])]
  h = ccall(cl.clEnqueueNDRangeKernel, rt.after(*sets).index(1).load(), kernel, 3, z, buf.index(rows[-6][0]), buf.index(rows[-3][0]), 0, z, z)
  for i in imgs: h = ccall(cl.clReleaseMemObject, buf.after(h).bitcast(dtypes.uint64).index(rows[i][0] // 8).load())
  return h

pm_cl = PatternMatcher([ # ctx: (device, rt_vars, the last command), the queue is in order so a wait is nothing
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), launch),
  (UPat(Ops.CALL, src=(UPat.custom_function("store"), UPat(name="sig"), UPat(name="val"))),
   lambda ctx, sig, val: sig.after(ctx[2]).index(0).store(val)),
  (UPat(Ops.CALL, src=(UPat.custom_function("timestamp"), UPat(name="sig"))), # [sig][event]
   lambda ctx, sig: ccall(cl.clEnqueueMarkerWithWaitList, ctx[1].after(ctx[2]).index(1).load(), 0, UOp.const(0, dtypes.uint64),
                          sig[1:2].getaddr(ctx[0].device))),
])

def cl_submit(s:UOp) -> UOp:
  (call, *deps), q = s.src, s.src[0].src[1]
  dev, h = Device[q.device], (rt:=UOp.param(0, dtypes.uint64, 2, q.device, name="rt_vars")) # [context, queue]
  for u in call.body.src: h = h if (x:=pm_cl.rewrite(u, ctx=(dev, rt, h))) is None else x
  return h.sink().call(UOp.alloc((2,), dtypes.uint64, 0, device=q.device).rtag(dev.tag("cl")).after(*deps), name=call.body.arg.name)

# *****************
# device

class CLAllocator(HostAllocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.host or options.cpu_access: return super()._alloc(size, options)
    mem = checked(cl.clCreateBuffer(self.dev.context, cl.CL_MEM_READ_WRITE, size, None, status := ctypes.c_int32()), status)
    return BufferStorage(unwrap(ctypes.cast(mem, ctypes.c_void_p).value), mem)

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.host is None: check(cl.clReleaseMemObject(storage.meta))
  def _copyin(self, dest:int, src:memoryview):
    self.dev.pending_copyin.append(src:=memoryview(bytearray(src))) # NOTE: these can't be freed until the GPU actually executes this command
    with cpu_profile(f"TINY -> {self.dev.device}", f"{self.dev.device}:COPY"):
      check(cl.clEnqueueWriteBuffer(self.dev.queue, ctypes.cast(dest, cl.cl_mem), False, 0, len(src)*src.itemsize, from_mv(src), 0, None, None))
  def _copyout(self, dest:memoryview, src:int):
    with cpu_profile(f"{self.dev.device} -> TINY", f"{self.dev.device}:COPY"):
      check(cl.clEnqueueReadBuffer(self.dev.queue, ctypes.cast(src, cl.cl_mem), False, 0, len(dest)*dest.itemsize, from_mv(dest), 0, None, None))
      self.dev.synchronize()

Compiled.pm_encode += PatternMatcher([(UPat.sink(arg=KernelInfo("cl_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"), cl_submit)])

class CLDevice(Compiled):
  device_ids = None                 # this is global and only initted once
  timestamp_divider = 3000 / 125 if OSX else 1000.0

  def __init__(self, device:str=""):
    if CLDevice.device_ids is None:
      check(cl.clGetPlatformIDs(0, None, num_platforms := ctypes.c_uint32()))
      check(cl.clGetPlatformIDs(num_platforms.value, platform_ids := (cl.cl_platform_id * num_platforms.value)(), None))
      for device_type in [cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_DEFAULT]:
        err = cl.clGetDeviceIDs(platform_ids[0], device_type, 0, None, num_devices := ctypes.c_uint32())
        if err == 0 and num_devices.value != 0: break
      if DEBUG >= 1: print(f"CLDevice: got {num_platforms.value} platforms and {num_devices.value} devices")
      CLDevice.device_ids = c.init_c_var((cl.cl_device_id * num_devices.value),
                                         lambda x: check(cl.clGetDeviceIDs(platform_ids[0], device_type, num_devices, x, None)))

    self.cl_dev = CLDevice.device_ids[0 if ":" not in device else int(device.split(":")[1])]
    self.device_name = (cl.clGetDeviceInfo(self.cl_dev, cl.CL_DEVICE_NAME, 256,
                                           buf:=ctypes.create_string_buffer(256), None), buf.value.decode())[1]
    self.driver_version = (cl.clGetDeviceInfo(self.cl_dev, cl.CL_DRIVER_VERSION, 256,
                                              buf:=ctypes.create_string_buffer(256), None), buf.value.decode())[1]
    if DEBUG >= 1: print(f"CLDevice: opening {self.device_name} with version {self.driver_version}")
    self.context = checked(cl.clCreateContext(None, 1, self.cl_dev, CC_CB(), None, status := ctypes.c_int32()), status)
    self.queue = checked(cl.clCreateCommandQueue(self.context, self.cl_dev, cl.CL_QUEUE_PROFILING_ENABLE, status), status)
    self.pending_copyin: list[memoryview] = []
    check(cl.clGetDeviceInfo(self.cl_dev, cl.CL_DEVICE_EXTENSIONS, 0, None, ctypes.byref(exts_len:=ctypes.c_size_t())))
    self.device_exts = (cl.clGetDeviceInfo(self.cl_dev, cl.CL_DEVICE_EXTENSIONS, exts_len.value,
                                           ctypes.byref(buf := ctypes.create_string_buffer(exts_len.value)), None),
                                           ctypes.string_at(buf).decode().split())[1]

    self.cl_compiler = CLCompiler(self, f"{hashlib.md5(self.device_name.encode() + self.driver_version.encode()).hexdigest()}")

    arch = ",".join(self.device_exts)
    if "cl_khr_image2d_from_buffer" in self.device_exts:
      check(cl.clGetDeviceInfo(self.cl_dev, cl.CL_DEVICE_IMAGE_PITCH_ALIGNMENT, 4, ctypes.byref(ipa := ctypes.c_uint32()), None))
      arch += f",IMAGE_PITCH_ALIGNMENT={ipa.value}"
    super().__init__(device, CLAllocator(self), [OpenCLRenderer], None, arch=arch)
    Compiled.pm_bufferize += PatternMatcher([(UPat(Ops.ALLOC, tag=self.tag("cl")), lambda d=self: d.handles)])
    Compiled.pm_link += PatternMatcher([(UPat(Ops.ALLOC, name="p").store(UPat(Ops.PROGRAM, name="prg")), lambda p, prg, d=self:
      p.store(UOp(Ops.BINARY, arg=d.function(prg.src[3].arg, prg.src[0].arg.function_name))) if p.device == d.device else None)])

  @functools.cached_property
  def handles(self) -> Buffer: return Buffer(HCQ_RUNTIME_DEV.device, 16, initial_value=bytes(self.context) + bytes(self.queue))

  @functools.cache
  def function(self, src:bytes, name:str) -> bytes:
    lib = self.cl_compiler.compile_cached(src.decode())
    program = checked(cl.clCreateProgramWithBinary(self.context, 1, self.cl_dev, (ctypes.c_size_t * 1)(len(lib)), to_char_p_p([lib], ctypes.c_ubyte),
                                                   None, errcode_ret := ctypes.c_int32()), errcode_ret)
    check(cl.clBuildProgram(program, 1, self.cl_dev, None, BP_CB(), None)) # NOTE: OSX requires this
    kernel = checked(cl.clCreateKernel(program, name.encode(), status := ctypes.c_int32()), status)
    return bytes(kernel)

  def count(self) -> int: return len(unwrap(self.device_ids))

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    check(cl.clFinish(self.queue))
    self.pending_copyin.clear()

  def collect_prof(self):
    for addr in [b.host.addr + i * 8 for (b, _), e in self.prof_ents.items() for i in (e.st_id, e.en_id)]:
      check(cl.clGetEventProfilingInfo(ev:=ctypes.cast(cl.cl_event.from_address(addr), cl.cl_event), cl.CL_PROFILING_COMMAND_END, 8, addr, None))
      check(cl.clReleaseEvent(ev))
    super().collect_prof()
