from __future__ import annotations
import os, subprocess, pathlib, ctypes, tempfile, functools
import tinygrad.runtime.support.metal as cdll
from typing import List, Any, Tuple, Optional, cast
from tinygrad.helpers import prod, getenv, DEBUG, unwrap2
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer

def wait_check(cbuf: Any):
  cdll.send_message(cbuf, "waitUntilCompleted")
  if (error := cdll.send_message(cbuf, "error", restype=ctypes.c_ulong)) != 0:
    raise RuntimeError(error)

class MetalCompiler(Compiler):
  def __init__(self, device:Optional[MetalDevice]):
    self.device = device
    super().__init__("compile_metal")
  def compile(self, src:str) -> bytes:
    if self.device is None:
      # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=src.encode('utf-8'))
      return subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
    options = cdll.send_message(
                cdll.libobjc.objc_getClass(b"MTLCompileOptions"),
                "new",
            )
    cdll.send_message(options, "setFastMathEnabled:", getenv("METAL_FAST_MATH"))
    library = cdll.send_message(self.device.device, "newLibraryWithSource:options:error:", cdll.to_ns_str(src), options, None)
    library_contents_ptr = cdll.send_message(library, "libraryDataContents")
    library_contents_bytes_ptr = cdll.send_message(library_contents_ptr, "bytes")
    library_length = cast(int, cdll.send_message(library_contents_ptr, "length", restype=ctypes.c_ulong))
    library_bytes = ctypes.string_at(library_contents_bytes_ptr, library_length)
    return library_bytes

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
    data = cdll.libdispatch.dispatch_data_create(lib, len(lib), None, None)
    self.library = cdll.send_message(self.device.device, "newLibraryWithData:error:", data, None)
    self.fxn = cdll.send_message(self.library, "newFunctionWithName:", cdll.to_ns_str(name))
    descriptor = cdll.send_message(cdll.libobjc.objc_getClass(b"MTLComputePipelineDescriptor"), "new")
    cdll.send_message(descriptor, "setComputeFunction:", self.fxn)
    cdll.send_message(descriptor, "setSupportIndirectCommandBuffers:", True)
    self.pipeline_state = cdll.send_message(self.device.device, "newComputePipelineStateWithDescriptor:options:reflection:error:", descriptor, 0, None, None)

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if prod(local_size) > cdll.send_message(self.pipeline_state, "maxTotalThreadsPerThreadgroup", restype=ctypes.c_ulong): raise RuntimeError("local size too big")
    command_buffer = cdll.send_message(self.device.mtl_queue, "commandBuffer")
    encoder = cdll.send_message(command_buffer, "computeCommandEncoder")
    cdll.send_message(encoder, "setComputePipelineState:", self.pipeline_state)
    for i,a in enumerate(bufs): cdll.send_message(encoder, "setBuffer:offset:atIndex:", a.device_buf, a.offset, i)
    for i,a in enumerate(vals,start=len(bufs)): cdll.send_message(encoder, "setBytes:length:atIndex:", (ctypes.c_char * 4)(a), 4, i)
    cdll.send_message(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", cdll.int_tuple_to_struct(global_size), cdll.int_tuple_to_struct(local_size))

    cdll.send_message(encoder, "endEncoding")
    cdll.send_message(command_buffer, "commit")
    if wait:
      wait_check(command_buffer)
      return cdll.send_message(command_buffer, "GPUEndTime", restype=ctypes.c_ulong) - cdll.send_message(command_buffer, "GPUStartTime", restype=ctypes.c_ulong)
    self.device.mtl_buffers_in_flight.append(command_buffer)
class MetalBuffer:
  def __init__(self, buf:Any, device_buf: cdll.objc_id, size:int, offset=0): self.buf, self.device_buf, self.size, self.offset = buf, device_buf, size, offset

class MetalAllocator(LRUAllocator):
  def __init__(self, device:MetalDevice):
    self.device:MetalDevice = device
    super().__init__()
  def _alloc(self, size:int, options) -> MetalBuffer:
    buf_ptr = (ctypes.c_char * size)()
    buf_memoryview = memoryview(buf_ptr).cast("B")
    device_buf = cdll.send_message(self.device.device, "newBufferWithBytesNoCopy:length:options:deallocator:", buf_ptr, size, 0, None)
    if device_buf.value is None: raise RuntimeError("Metal failed to allocate buffer")
    return MetalBuffer(buf_memoryview, device_buf, size)
  def _free(self, opaque:MetalBuffer, options): cdll.send_message(opaque.device_buf, "release")
  def transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev:MetalDevice, dest_dev:MetalDevice):
    dest_dev.synchronize()
    src_command_buffer = cdll.send_message(src_dev.mtl_queue, "commandBuffer")
    encoder = cdll.send_message(src_command_buffer, "blitCommandEncoder")
    cdll.send_message(encoder, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", src.device_buf, src.offset, dest.device_buf, dest.offset, sz)
    cdll.send_message(encoder, "endEncoding")
    if src_dev != dest_dev:
      cdll.send_message(src_command_buffer, "encodeSignalEvent:value:", src_dev.timeline_signal, src_dev.timeline_value)
      dest_command_buffer = cdll.send_message(dest_dev.mtl_queue, "commandBuffer")
      cdll.send_message(dest_command_buffer, "encodeWaitForEvent:value:", src_dev.timeline_signal, src_dev.timeline_value)
      cdll.send_message(dest_command_buffer, "commit")
      dest_dev.mtl_buffers_in_flight.append(dest_command_buffer)
      src_dev.timeline_value += 1
    cdll.send_message(src_command_buffer, "commit")
    src_dev.mtl_buffers_in_flight.append(src_command_buffer)
  def from_buffer(self, src:memoryview) -> Optional[Any]:
    ptr = (ctypes.c_char * src.nbytes).from_buffer(src)
    ret = cdll.send_message(self.device.device, "newBufferWithBytesNoCopy:length:options:deallocator:", ptr, src.nbytes, 0, None)
    if ret: self.device.mv_in_metal.append(src)
    return MetalBuffer(src, ret, src.nbytes)
  def as_buffer(self, src:MetalBuffer) -> memoryview:
    self.device.synchronize()
    ret = src.buf[src.offset:src.offset+src.size]
    return ret
  def copyin(self, dest:MetalBuffer, src:memoryview): self.as_buffer(dest)[:] = src
  def copyout(self, dest:memoryview, src:MetalBuffer): dest[:] = self.as_buffer(src)
  def offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, buf.device_buf, size, offset)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.device = cdll.metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = cdll.send_message(self.device, "newCommandQueueWithMaxCommandBufferCount:", 1024)
    if self.mtl_queue is None: raise RuntimeError("Cannot allocate a new command queue")

    self.mtl_buffers_in_flight: List[Any] = []
    self.mv_in_metal: List[memoryview] = []

    self.timeline_signal = cdll.send_message(self.device, "newSharedEvent")
    self.timeline_value = 0

    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalRenderer(), MetalCompiler(None if getenv("METAL_XCODE") else self),
                     functools.partial(MetalProgram, self), MetalGraph)
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)
    self.mv_in_metal.clear()
    self.mtl_buffers_in_flight.clear()
