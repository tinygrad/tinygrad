from __future__ import annotations
import os, subprocess, pathlib, tempfile, functools
from tinygrad.runtime.support.metal import msg, libobjc, to_ns_str, libdispatch, int_tuple_to_struct, libmetal
from typing import List, Any, Tuple, Optional, cast
from tinygrad.helpers import prod, getenv, DEBUG
from tinygrad.device import Compiled, Compiler, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer
from ctypes import c_ulong, c_double, string_at, c_int, c_char

def wait_check(cbuf: Any):
  msg(cbuf, "waitUntilCompleted")
  if (error := msg(cbuf, "error", restype=c_ulong)) != 0: raise RuntimeError(error)

class MetalCompiler(Compiler):
  def __init__(self, device:Optional[MetalDevice]):
    self.device = device
    super().__init__("compile_metal")
  def compile(self, src:str) -> bytes:
    if self.device is None:
      # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=src.encode('utf-8'))
      return subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
    options = msg(libobjc.objc_getClass(b"MTLCompileOptions"), "new")
    msg(options, "setFastMathEnabled:", getenv("METAL_FAST_MATH"))
    library = msg(self.device.device, "newLibraryWithSource:options:error:", to_ns_str(src), options, None)
    library_contents_ptr = msg(library, "libraryDataContents")
    library_contents_bytes_ptr = msg(library_contents_ptr, "bytes")
    library_length = cast(int, msg(library_contents_ptr, "length", restype=c_ulong))
    return string_at(library_contents_bytes_ptr, library_length)

class MetalProgram:
  def __init__(self, device:MetalDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 6:
      with tempfile.NamedTemporaryFile(delete=True) as shader:
        shader.write(lib)
        shader.flush()
        ret = os.system(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}")
        if ret:
          print("Error running disassembler: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")
    assert lib[:4] == b"MTLB", "Invalid Metal library. Could be due to using conda. Try system python or METAL_XCODE=1 DISABLE_COMPILER_CACHE=1."
    data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
    self.library = msg(self.device.device, "newLibraryWithData:error:", data, None)
    self.fxn = msg(self.library, "newFunctionWithName:", to_ns_str(name))
    descriptor = msg(libobjc.objc_getClass(b"MTLComputePipelineDescriptor"), "new")
    msg(descriptor, "setComputeFunction:", self.fxn)
    msg(descriptor, "setSupportIndirectCommandBuffers:", True)
    self.pipeline_state = msg(self.device.device, "newComputePipelineStateWithDescriptor:options:reflection:error:",
                                       descriptor, 0, None, None)

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if prod(local_size) > msg(self.pipeline_state, "maxTotalThreadsPerThreadgroup", restype=c_ulong):
      raise RuntimeError("local size too big")
    command_buffer = msg(self.device.mtl_queue, "commandBuffer")
    encoder = msg(command_buffer, "computeCommandEncoder")
    msg(encoder, "setComputePipelineState:", self.pipeline_state)
    for i,a in enumerate(bufs): msg(encoder, "setBuffer:offset:atIndex:", a.buf, a.offset, i)
    for i,a in enumerate(vals,start=len(bufs)): msg(encoder, "setBytes:length:atIndex:", bytes(c_int(a)), 4, i)
    msg(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", int_tuple_to_struct(global_size), int_tuple_to_struct(local_size))
    msg(encoder, "endEncoding")
    msg(command_buffer, "commit")
    if wait:
      wait_check(command_buffer)
      return msg(command_buffer, "GPUEndTime", restype=c_double) - msg(command_buffer, "GPUStartTime", restype=c_double)
    self.device.mtl_buffers_in_flight.append(command_buffer)

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class MetalAllocator(LRUAllocator):
  def __init__(self, device:MetalDevice):
    self.device:MetalDevice = device
    super().__init__()
  def _alloc(self, size:int, options) -> MetalBuffer:
    ret = msg(self.device.device, "newBufferWithLength:options:", size, 0)
    if ret.value is None: raise RuntimeError("Metal failed to allocate buffer")
    return MetalBuffer(ret, size)
  def _free(self, opaque:MetalBuffer, options): msg(opaque.buf, "release")
  def transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice):
    dest_dev.synchronize()
    src_command_buffer = msg(src_dev.mtl_queue, "commandBuffer")
    encoder = msg(src_command_buffer, "blitCommandEncoder")
    msg(encoder, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", src.buf, src.offset, dest.buf, dest.offset, sz)
    msg(encoder, "endEncoding")
    if src_dev != dest_dev:
      msg(src_command_buffer, "encodeSignalEvent:value:", src_dev.timeline_signal, src_dev.timeline_value)
      dest_command_buffer = msg(dest_dev.mtl_queue, "commandBuffer")
      msg(dest_command_buffer, "encodeWaitForEvent:value:", src_dev.timeline_signal, src_dev.timeline_value)
      msg(dest_command_buffer, "commit")
      dest_dev.mtl_buffers_in_flight.append(dest_command_buffer)
      src_dev.timeline_value += 1
    msg(src_command_buffer, "commit")
    src_dev.mtl_buffers_in_flight.append(src_command_buffer)
  def from_buffer(self, src:memoryview) -> Optional[Any]:
    ptr = (c_char * src.nbytes).from_buffer(src)
    ret = msg(self.device.device, "newBufferWithBytesNoCopy:length:options:deallocator:", ptr, src.nbytes, 0, None)
    if ret: self.device.mv_in_metal.append(src)
    return MetalBuffer(ret, src.nbytes)
  def as_buffer(self, src:MetalBuffer) -> memoryview:
    self.device.synchronize()
    ptr = msg(src.buf, "contents")
    array = (c_char * (src.offset + src.size)).from_address(ptr.value)
    return memoryview(array).cast("B")[src.offset:]
  def copyin(self, dest:MetalBuffer, src:memoryview): self.as_buffer(dest)[:] = src
  def copyout(self, dest:memoryview, src:MetalBuffer): dest[:] = self.as_buffer(src)
  def offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.device = libmetal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = msg(self.device, "newCommandQueueWithMaxCommandBufferCount:", 1024)
    if self.mtl_queue is None: raise RuntimeError("Cannot allocate a new command queue")
    self.mtl_buffers_in_flight: List[Any] = []
    self.mv_in_metal: List[memoryview] = []
    self.timeline_signal = msg(self.device, "newSharedEvent")
    self.timeline_value = 0

    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalRenderer(), MetalCompiler(None if getenv("METAL_XCODE") else self),
                     functools.partial(MetalProgram, self), MetalGraph)
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)
    self.mv_in_metal.clear()
    self.mtl_buffers_in_flight.clear()
