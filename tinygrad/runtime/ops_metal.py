# pip3 install pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
import os, subprocess, pathlib, ctypes
import Metal, Cocoa, libdispatch # type: ignore
from typing import List, Any, Tuple
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import prod, getenv, DEBUG, DType, dtypes
from tinygrad.ops import Compiled, ASTRunner, BasicBatchExecutor
from tinygrad.renderer.metal import MetalRenderer
from tinygrad.runtime.lib import RawBufferMapped, LRUAllocator

METAL_XCODE = getenv("METAL_XCODE")

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

class MetalBatchExecutor(BasicBatchExecutor):
  def __init__(self, jit_cache: List[Tuple[Any, Any, Any]]): self.use_basic_executor = (DEBUG>0 or not all(isinstance(prg, ASTRunner) and isinstance(prg.clprg, MetalProgram) for prg,_,_ in jit_cache))
  def __do_exec(self, jit_cache: List[Tuple[Any, Any, Any]]):
    if len(jit_cache) == 0: return
    command_buffer = METAL.mtl_queue.commandBufferWithUnretainedReferences()
    encoder = command_buffer.computeCommandEncoder()
    for prg, pargs, variables in jit_cache:
      global_size, local_size = prg.launch_dims(variables)
      encoder.setComputePipelineState_(prg.clprg.pipeline_state)
      for i,a in enumerate(pargs): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
      for i,a in enumerate(variables.values()): encoder.setBytes_length_atIndex_((arg:=ctypes.c_int32(a)), ctypes.sizeof(arg), len(pargs)+i)
      encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    METAL.mtl_buffers_in_flight.append(command_buffer)
  def exec(self, jit_cache: List[Tuple[Any, Any, Any]], updatable_entries):
    if self.use_basic_executor: return super().exec(jit_cache, updatable_entries) # No graph is created switch to basic executor.
    for i in range((len(jit_cache)+127)//128): self.__do_exec(jit_cache[128*i:128*(i+1)]) # Run in batches with size 128.
    super().recalc_stat(jit_cache)

def unwrap(x):
  ret, err = x
  assert err is None, str(err)
  return ret

class MetalProgram:
  def __init__(self, name:str, prg:str, binary:bool=False):
    if METAL_XCODE:
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=prg.encode('utf-8'))
      # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
      lib = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
      data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
      self.library = unwrap(METAL.device.newLibraryWithData_error_(data, None))
    else:
      options = Metal.MTLCompileOptions.alloc().init()
      self.library = unwrap(METAL.device.newLibraryWithSource_options_error_(prg, options, None))
    self.fxn = self.library.newFunctionWithName_(name)
    # hacks to disassemble shader
    if DEBUG >= 5:
      arc = unwrap(METAL.device.newBinaryArchiveWithDescriptor_error_(Metal.MTLBinaryArchiveDescriptor.alloc().init(), None))
      desc = Metal.MTLComputePipelineDescriptor.alloc().init()
      desc.setComputeFunction_(self.fxn)
      unwrap(arc.addComputePipelineFunctionsWithDescriptor_error_(desc, None))
      unwrap(arc.serializeToURL_error_(Cocoa.NSURL.URLWithString_("file:///tmp/shader.bin"), None))
      # clone https://github.com/dougallj/applegpu.git in tinygrad/disassemblers
      os.system(f"cd {pathlib.Path(__file__).parents[2]}/disassemblers/applegpu && python3 compiler_explorer.py /tmp/shader.bin")
    self.pipeline_state = unwrap(METAL.device.newComputePipelineStateWithFunction_error_(self.fxn, None))

  def __call__(self, global_size, local_size, *bufs, wait=False):
    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs):
      if isinstance(a, RawMetalBuffer): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
      elif isinstance(a, int): encoder.setBytes_length_atIndex_((arg:=ctypes.c_int32(a)), ctypes.sizeof(arg), i)
      else: raise RuntimeError(f"arg at index {i} has unsupported type {type(a)}")
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    METAL.mtl_buffers_in_flight.append(command_buffer)

MetalBuffer = Compiled(RawMetalBuffer, LinearizerOptions(device="METAL"), MetalRenderer, MetalProgram, METAL.synchronize, MetalBatchExecutor)
