# pip3 install pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
import os, subprocess, pathlib, ctypes, tempfile
import Metal, Cocoa, libdispatch
from typing import List, Any, Tuple
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import prod, getenv, DEBUG, DType, dtypes, diskcache
from tinygrad.ops import Compiled
from tinygrad.renderer.metal import MetalRenderer
from tinygrad.runtime.lib import RawBufferMapped, LRUAllocator

class MetalAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs): return METAL.device.newBufferWithLength_options_(size*dtype.itemsize, Metal.MTLResourceStorageModeShared)
  def _do_free(self, buf): buf.release()
  def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize) # Buffers of the same length could be reused, no matter what dtype.

class _METAL:
  def __init__(self):
    self.mtl_buffers_in_flight: List[Any] = []
    self.device = Metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = self.device.newCommandQueueWithMaxCommandBufferCount_(1024)
    self.allocator = MetalAllocator(self.device.dedicatedMemorySize() or self.device.sharedMemorySize())
  # TODO: is there a better way to do this?
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: cbuf.waitUntilCompleted()
    self.mtl_buffers_in_flight.clear()
METAL = _METAL()

class RawMetalBuffer(RawBufferMapped):
  def __init__(self, size:int, dtype:DType):
    assert dtype != dtypes.double, f"METAL does not support {dtype.name}"
    super().__init__(size, dtype, allocator=METAL.allocator)
  def _buffer(self):
    METAL.synchronize()
    return self._buf.contents().as_buffer(self._buf.length())

def unwrap(x):
  ret, err = x
  assert err is None, str(err)
  return ret

@diskcache
def compile_metal(prg, use_xcode=bool(getenv("METAL_XCODE"))) -> bytes:
  if use_xcode:
    # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
    air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=prg.encode('utf-8'))
    return subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
  options = Metal.MTLCompileOptions.new()
  library = unwrap(METAL.device.newLibraryWithSource_options_error_(prg, options, None))
  # TODO: avoid file write here?
  with tempfile.NamedTemporaryFile(delete=True) as output_file:
    unwrap(library.serializeToURL_error_(Cocoa.NSURL.URLWithString_(f"file://{output_file.name}"), None))
    return pathlib.Path(output_file.name).read_bytes()

class MetalProgram:
  def __init__(self, name:str, lib:bytes):
    data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
    self.library = unwrap(METAL.device.newLibraryWithData_error_(data, None))
    self.fxn = self.library.newFunctionWithName_(name)
    if DEBUG >= 6:
      with tempfile.NamedTemporaryFile(delete=True) as shader:
        shader.write(lib)
        shader.flush()
        os.system(f"cd {pathlib.Path(__file__).parents[2]}/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}")
    #self.pipeline_state = unwrap(METAL.device.newComputePipelineStateWithFunction_error_(self.fxn, None))
    descriptor = Metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction_(self.fxn)
    descriptor.setSupportIndirectCommandBuffers_(True)
    self.pipeline_state = unwrap(METAL.device.newComputePipelineStateWithDescriptor_options_reflection_error_(descriptor, Metal.MTLPipelineOption(0), None, None))

    icb_descriptor = Metal.MTLIndirectCommandBufferDescriptor.new()
    #print(dir(icb_descriptor))
    icb_descriptor.setCommandTypes_(Metal.MTLIndirectCommandType(Metal.MTLIndirectCommandTypeConcurrentDispatch))
    icb_descriptor.setInheritBuffers_(False)
    icb_descriptor.setInheritPipelineState_(False)
    icb_descriptor.setMaxKernelBufferBindCount_(3)
    self.icb = METAL.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(icb_descriptor, 1, Metal.MTLResourceOptions(0))
    self.icb_command = self.icb.indirectComputeCommandAtIndex_(0)
    self.icb_command.setComputePipelineState_(self.pipeline_state)

  def __call__(self, *bufs, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    for i,a in enumerate(bufs):
      self.icb_command.setKernelBuffer_offset_atIndex_(a._buf, 0, i)
    self.icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    self.icb_command.setBarrier()
    encoder.executeCommandsInBuffer_withRange_(self.icb, Cocoa.NSRange(0,1))
    """
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs):
      if isinstance(a, RawMetalBuffer): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
      elif isinstance(a, int): encoder.setBytes_length_atIndex_((arg:=ctypes.c_int32(a)), ctypes.sizeof(arg), i)
      else: raise RuntimeError(f"arg at index {i} has unsupported type {type(a)}")
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    """
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    METAL.mtl_buffers_in_flight.append(command_buffer)

MetalBuffer = Compiled(RawMetalBuffer, LinearizerOptions(device="METAL"), MetalRenderer, compile_metal, MetalProgram, METAL.synchronize)
