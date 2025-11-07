import subprocess, pathlib, struct, ctypes, tempfile, functools, contextlib, decimal, platform, sys
from typing import Any, cast
from tinygrad.helpers import prod, to_mv, getenv, round_up, cache_dir, init_c_struct_t, PROFILE, ProfileRangeEvent, cpu_profile, unwrap
import tinygrad.runtime.support.objc as objc
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator, ProfileDeviceEvent
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.autogen import metal, libsystem

# 13 is requestType that metal uses to compile source code into MTLB, there aren't any docs or symbols.
REQUEST_TYPE_COMPILE = 13

# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")

@functools.cache
def to_ns_str(s: str): return objc.msg("stringWithUTF8String:")(objc.lib.objc_getClass(b"NSString"), s.encode())
def from_ns_str(s): return bytes(objc.msg("UTF8String", ctypes.c_char_p)(s)).decode()

def to_struct(*t: int, _type: type[ctypes._SimpleCData] = ctypes.c_ulong):
  return init_c_struct_t(tuple([(f"field{i}", _type) for i in range(len(t))]))(*t)

def wait_check(cbuf: Any):
  objc.msg("waitUntilCompleted")(cbuf)
  error_check(objc.msg("error")(cbuf).retained())

def cmdbuf_label(cbuf: objc.id_) -> str|None: return from_ns_str(label) if (label:=objc.msg("label")(cbuf)).value is not None else None
def cmdbuf_st_time(cbuf: objc.id_) -> float: return cast(float, objc.msg("GPUStartTime", ctypes.c_double)(cbuf))
def cmdbuf_en_time(cbuf: objc.id_) -> float: return cast(float, objc.msg("GPUEndTime", ctypes.c_double)(cbuf))

def error_check(error: objc.id_, error_constructor: type[Exception] = RuntimeError):
  if error.value is None: return None
  raise error_constructor(from_ns_str(objc.msg("localizedDescription")(error)).retained())

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.sysdevice = metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = objc.msg("newCommandQueueWithMaxCommandBufferCount:")(self.sysdevice, 1024).retained()
    if self.mtl_queue is None: raise RuntimeError("Cannot allocate a new command queue")
    self.mtl_buffers_in_flight: list[Any] = []
    self.timeline_signal = objc.msg("newSharedEvent")(self.sysdevice).retained()
    self.timeline_value = 0

    Compiled.profile_events += [ProfileDeviceEvent(device)]

    from tinygrad.runtime.graph.metal import MetalGraph
    # NOTE: GitHub CI macOS runners use paravirtualized metal which is broken with graph.
    # This can be reproduced locally with any virtualization software (like utm) that can create macOS VMs with apple's own virtualization framework.
    super().__init__(device, MetalAllocator(self), [(MetalRenderer, MetalCompiler), (MetalRenderer, Compiler)], functools.partial(MetalProgram, self),
                     MetalGraph if 'virtual' not in from_ns_str(objc.msg('name')(self.sysdevice)).lower() else None)

  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight:
      wait_check(cbuf)
      st, en = decimal.Decimal(cmdbuf_st_time(cbuf)) * 1000000, decimal.Decimal(cmdbuf_en_time(cbuf)) * 1000000
      # NOTE: command buffers from MetalGraph are not profiled here
      if PROFILE and (lb:=cmdbuf_label(cbuf)) is not None and not lb.startswith("batched"):
        Compiled.profile_events += [ProfileRangeEvent(self.device, lb, st, en, is_copy=lb.startswith("COPY"))]
    self.mtl_buffers_in_flight.clear()

def metal_src_to_library(device:MetalDevice, src:str) -> objc.id_:
  options = objc.msg("new")(objc.lib.objc_getClass(b"MTLCompileOptions")).retained()
  objc.msg("setFastMathEnabled:")(options, getenv("METAL_FAST_MATH"))
  library = objc.msg("newLibraryWithSource:options:error:")(device.sysdevice, to_ns_str(src), options,
                                                            ctypes.byref(compileError:=objc.id_().retained())).retained()
  error_check(compileError, CompileError)
  return library

class MetalCompiler(Compiler):
  # Opening METAL after LLVM doesn't fail because ctypes.CDLL opens with RTLD_LOCAL but MTLCompiler opens it's own llvm with RTLD_GLOBAL
  # This means that MTLCompiler's llvm will create it's own instances of global state because RTLD_LOCAL doesn't export symbols, but if RTLD_GLOBAL
  # library is loaded first then RTLD_LOCAL library will just use it's symbols. On linux there is RTLD_DEEPBIND to prevent that, but on macos there
  # doesn't seem to be anything we can do.
  with contextlib.suppress(FileNotFoundError, ModuleNotFoundError):
    import tinygrad.runtime.autogen.llvm # noqa: F401
  support = ctypes.CDLL("/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler")
  support.MTLCodeGenServiceCreate.restype = ctypes.c_void_p

  def __init__(self):
    self.cgs = ctypes.c_void_p(MetalCompiler.support.MTLCodeGenServiceCreate(b"tinygrad"))
    super().__init__("compile_metal_direct")
  def __reduce__(self): return (MetalCompiler,()) # force pickle to create new instance for each multiprocessing fork
  def compile(self, src:str) -> bytes:
    ret: Exception|bytes = CompileError("MTLCodeGenServiceBuildRequest returned without calling the callback")
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
    def callback(blockptr, error, dataPtr, dataLen, errorMessage):
      nonlocal ret
      if error == 0:
        reply = bytes(to_mv(dataPtr, dataLen))
        # offset from beginning to data = header size + warning size
        ret = reply[sum(struct.unpack('<LL', reply[8:16])):]
      else:
        ret = CompileError(errorMessage.decode())

    # no changes for compute in 2.0 - 2.4 specs, use 2.0 as default for old versions.
    macos_major = int(platform.mac_ver()[0].split('.')[0])
    metal_version = "metal3.1" if macos_major >= 14 else "metal3.0" if macos_major >= 13 else "macos-metal2.0"

    # llvm will create modules.timestamp in cache path and cache compilation of metal stdlib (250ms => 8ms compilation time)
    # note that llvm won't necessarily create anything else here as apple has prebuilt versions of many standard libraries
    params = f'-fno-fast-math -std={metal_version} --driver-mode=metal -x metal -fmodules-cache-path="{cache_dir}" -fno-caret-diagnostics'
    # source blob has to be padded to multiple of 4 but at least one 'b\x00' should be added, params blob just has to be null terminated
    src_padded, params_padded = src.encode() + b'\x00'*(round_up(len(src) + 1, 4) - len(src)), params.encode() + b'\x00'
    request = struct.pack('<QQ', len(src_padded), len(params_padded)) + src_padded + params_padded
    # The callback is actually not a callback but a block which is apple's non-standard extension to add closures to C.
    # See https://clang.llvm.org/docs/Block-ABI-Apple.html#high-level for struct layout.
    # Fields other than invoke are unused in this case so we can just use ctypes.byref with negative offset to invoke field, add blockptr as a first
    # argument and pretend it's a normal callback
    MetalCompiler.support.MTLCodeGenServiceBuildRequest(self.cgs, None, REQUEST_TYPE_COMPILE, request, len(request), ctypes.byref(callback, -0x10))
    if isinstance(ret, Exception): raise ret
    assert ret[:4] == b"MTLB" and ret[-4:] == b"ENDT", f"Invalid Metal library. {ret!r}"
    return ret
  def disassemble(self, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=True) as shader:
      shader.write(lib)
      shader.flush()
      proc = subprocess.Popen(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}",
                              stdout=subprocess.PIPE, shell=True, text=True, bufsize=1)
      for line in unwrap(proc.stdout): print(line, end="")
      ret = proc.wait()
      if ret: print("Disassembler Error: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")

class MetalProgram:
  def __init__(self, dev:MetalDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    if lib[:4] == b"MTLB":
      # binary metal library
      data = libsystem.dispatch_data_create(lib, len(lib), None, None)
      self.library = objc.msg("newLibraryWithData:error:")(self.dev.sysdevice, data, ctypes.byref(error_lib:=objc.id_().retained())).retained()
      error_check(error_lib)
    else:
      # metal source. rely on OS caching
      try: self.library = metal_src_to_library(self.dev, lib.decode())
      except CompileError as e: raise RuntimeError from e
    self.fxn = objc.msg("newFunctionWithName:")(self.library, to_ns_str(name)).retained()
    descriptor = objc.msg("new")(objc.lib.objc_getClass(b"MTLComputePipelineDescriptor")).retained()
    objc.msg("setComputeFunction:")(descriptor, self.fxn)
    objc.msg("setSupportIndirectCommandBuffers:")(descriptor, True)
    self.pipeline_state = objc.msg("newComputePipelineStateWithDescriptor:options:reflection:error:")(self.dev.sysdevice, descriptor,
      metal.MTLPipelineOptionNone, None, ctypes.byref(error_pipeline_creation:=objc.id_().retained())).retained()
    error_check(error_pipeline_creation)
    # cache these msg calls
    self.max_total_threads: int = cast(int, objc.msg("maxTotalThreadsPerThreadgroup", ctypes.c_ulong)(self.pipeline_state))

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    if prod(local_size) > self.max_total_threads:
      exec_width = objc.msg("threadExecutionWidth", ctypes.c_ulong)(self.pipeline_state)
      memory_length = objc.msg("staticThreadgroupMemoryLength", ctypes.c_ulong)(self.pipeline_state)
      raise RuntimeError(f"local size {local_size} bigger than {self.max_total_threads} with exec width {exec_width} memory length {memory_length}")
    command_buffer = objc.msg("commandBuffer")(self.dev.mtl_queue).retained()
    encoder = objc.msg("computeCommandEncoder")(command_buffer).retained()
    objc.msg("setComputePipelineState:")(encoder, self.pipeline_state)
    for i,a in enumerate(bufs): objc.msg("setBuffer:offset:atIndex:")(encoder, a.buf, a.offset, i)
    for i,a in enumerate(vals, start=len(bufs)): objc.msg("setBytes:length:atIndex:")(encoder, bytes(ctypes.c_int(a)), 4, i)
    objc.msg("dispatchThreadgroups:threadsPerThreadgroup:")(encoder, to_struct(*global_size), to_struct(*local_size))
    objc.msg("endEncoding")(encoder)
    objc.msg("setLabel:")(command_buffer, to_ns_str(self.name)) # TODO: is this always needed?
    objc.msg("commit")(command_buffer)
    self.dev.mtl_buffers_in_flight.append(command_buffer)
    if wait:
      wait_check(command_buffer)
      return cmdbuf_en_time(command_buffer) - cmdbuf_st_time(command_buffer)

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class MetalAllocator(LRUAllocator[MetalDevice]):
  def _alloc(self, size:int, options) -> MetalBuffer:
    if options.external_ptr: return MetalBuffer(objc.id_(options.external_ptr), size)

    # Buffer is explicitly released in _free() rather than garbage collected via reference count
    ret = objc.msg("newBufferWithLength:options:")(self.dev.sysdevice, ctypes.c_ulong(size), metal.MTLResourceStorageModeShared)
    if ret.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return MetalBuffer(ret, size)
  def _free(self, opaque:MetalBuffer, options):
    if not sys.is_finalizing(): objc.msg("release")(opaque.buf)
  def _transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice):
    dest_dev.synchronize()
    src_command_buffer = objc.msg("commandBuffer")(src_dev.mtl_queue).retained()
    encoder = objc.msg("blitCommandEncoder")(src_command_buffer).retained()
    objc.msg("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:")(encoder, src.buf, ctypes.c_ulong(src.offset),
        dest.buf, ctypes.c_ulong(dest.offset), ctypes.c_ulong(sz))
    objc.msg("endEncoding")(encoder)
    if src_dev != dest_dev:
      objc.msg("encodeSignalEvent:value:")(src_command_buffer, src_dev.timeline_signal, src_dev.timeline_value)
      dest_command_buffer = objc.msg("commandBuffer")(dest_dev.mtl_queue).retained()
      objc.msg("encodeWaitForEvent:value:")(dest_command_buffer, src_dev.timeline_signal, src_dev.timeline_value)
      objc.msg("commit")(dest_command_buffer)
      dest_dev.mtl_buffers_in_flight.append(dest_command_buffer)
      src_dev.timeline_value += 1
    objc.msg("setLabel:")(src_command_buffer, to_ns_str(f"COPY {src_dev.device} -> {dest_dev.device}"))
    objc.msg("commit")(src_command_buffer)
    src_dev.mtl_buffers_in_flight.append(src_command_buffer)
    # Transfers currently synchronize the completion. Otherwise, copies can sometimes lead to incorrect values.
    # There is no real metal multidevice support for now, so transfer is used only for tests.
    src_dev.synchronize()
  def _cp_mv(self, dst, src, prof_desc):
    with cpu_profile(prof_desc, self.dev.device, is_copy=True): dst[:] = src
  def _as_buffer(self, src:MetalBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(cast(int, objc.msg("contents")(src.buf).value), src.size + src.offset)[src.offset:]
  def _copyin(self, dest:MetalBuffer, src:memoryview): self._cp_mv(self._as_buffer(dest), src, "TINY -> METAL")
  def _copyout(self, dest:memoryview, src:MetalBuffer): self._cp_mv(dest, self._as_buffer(src), "METAL -> TINY")
  def _offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)
