from __future__ import annotations
import os, subprocess, pathlib, ctypes, tempfile, functools
import Metal, libdispatch
from typing import List, Any, Tuple, Optional
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import prod, getenv, DEBUG, diskcache, unwrap2
from tinygrad.device import Compiled, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer

@diskcache
def compile_metal(prg, use_xcode=bool(getenv("METAL_XCODE"))) -> bytes:
  assert MetalDevice.compiler_device, "metal device creation is required for metal compile"
  if use_xcode:
    # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
    air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=prg.encode('utf-8'))
    return subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
  options = Metal.MTLCompileOptions.new()
  library = unwrap2(MetalDevice.compiler_device.newLibraryWithSource_options_error_(prg, options, None))
  return library.libraryDataContents().bytes().tobytes()

class MetalDevice(Compiled):
  compiler_device = None
  def __init__(self, device:str):
    self.device = Metal.MTLCreateSystemDefaultDevice()
    if MetalDevice.compiler_device is None: MetalDevice.compiler_device = self.device
    self.mtl_queue = self.device.newCommandQueueWithMaxCommandBufferCount_(1024)
    self.mtl_buffers_in_flight: List[Any] = []
    self.mv_in_metal: List[memoryview] = []
    from tinygrad.features.graph.metal import MetalGraph
    super().__init__(LinearizerOptions(device="METAL"), MetalRenderer, compile_metal, functools.partial(MetalGraph, self), lru=getenv("LRU", 1))

  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: cbuf.waitUntilCompleted()
    self.mv_in_metal.clear()
    self.mtl_buffers_in_flight.clear()

  # methods for programs

  def create_program(self, name:str, lib:bytes):
    if DEBUG >= 6:
      with tempfile.NamedTemporaryFile(delete=True) as shader:
        shader.write(lib)
        shader.flush()
        os.system(f"cd {pathlib.Path(__file__).parents[2]}/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}")
    data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
    library = unwrap2(self.device.newLibraryWithData_error_(data, None))
    fxn = library.newFunctionWithName_(name)
    return unwrap2(self.device.newComputePipelineStateWithFunction_error_(fxn, None))

  def run_program(self, pipeline_state, bufs, vars:List[int], global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    assert prod(local_size) <= pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    command_buffer = self.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline_state)
    for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex_(a, 0, i)
    for i,a in enumerate(vars): encoder.setBuffer_offset_atIndex_(ctypes.c_int32(a), 4, len(bufs)+i)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    self.device.mtl_buffers_in_flight.append(command_buffer)

  # methods for buffers

  def _alloc(self, size:int) -> Any:
    ret = self.device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
    if ret is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return ret
  def _free(self, opaque:Any): opaque.release()

  def transfer(self, dest:Any, src:Any, sz:int):
    command_buffer = self.mtl_queue.commandBuffer()
    encoder = command_buffer.blitCommandEncoder()
    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(src, 0, dest, 0, sz)
    encoder.endEncoding()
    command_buffer.commit()
    self.mtl_buffers_in_flight.append(command_buffer)
  def from_buffer(self, src:memoryview) -> Optional[Any]:
    ret = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(src, len(src), Metal.MTLResourceStorageModeShared, None)
    if ret: self.mv_in_metal.append(src)
    return ret
  def as_buffer(self, src:Any) -> memoryview:
    self.synchronize()
    return src.contents().as_buffer(src.length())
  def copyin(self, dest:Any, src:memoryview): self.as_buffer(dest)[:] = src
  def copyout(self, dest:memoryview, src:Any): dest[:] = self.as_buffer(src)
