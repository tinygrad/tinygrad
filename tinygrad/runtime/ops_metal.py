# pip3 install pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
import os, subprocess, pathlib, ctypes, tempfile
import Metal, Cocoa, libdispatch
from typing import List, Any, Tuple, Dict, Union, Set
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import prod, getenv, DEBUG, DType, dtypes, diskcache
from tinygrad.ops import Compiled
from tinygrad.renderer.metal import MetalRenderer
from tinygrad.runtime.lib import RawBufferMapped, RawBuffer, LRUAllocator

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
    self.last_fence = None
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
    self.pipeline_state = unwrap(METAL.device.newComputePipelineStateWithFunction_error_(self.fxn, None))

  def __call__(self, *bufs, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    if METAL.last_fence:
      encoder.waitForFence_(METAL.last_fence)
      METAL.last_fence = None
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs):
      if isinstance(a, RawMetalBuffer): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
      elif isinstance(a, int): encoder.setBytes_length_atIndex_((arg:=ctypes.c_int32(a)), ctypes.sizeof(arg), i)
      else: raise RuntimeError(f"arg at index {i} has unsupported type {type(a)}")
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    # TODO: do we need to create fences here, or are these buffers "managed"?
    encoder.endEncoding()
    command_buffer.commit()
    
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    METAL.mtl_buffers_in_flight.append(command_buffer)

from tinygrad.jit import BatchExecutor, JitItem
from tinygrad.shape.symbolic import Variable, Node
class MetalBatchExecutor(BatchExecutor):
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: Dict[Union[int, str], RawBuffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    self.input_has_variable_dims: Set[int] = set()
    self.command_buffer: Any = None

    # create metal batch exec
    self.int_buf = RawMetalBuffer(len(var_vals), dtypes.int32)
    self.int_buf_view = self.int_buf.buffer_view()    # TODO: this is metal syncing when it doesn't need to
    icb_descriptor = Metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes_(Metal.MTLIndirectCommandType(Metal.MTLIndirectCommandTypeConcurrentDispatch))
    icb_descriptor.setInheritBuffers_(False)
    icb_descriptor.setInheritPipelineState_(False)
    icb_descriptor.setMaxKernelBufferBindCount_(31)
    self.icb = METAL.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(icb_descriptor, len(self.jit_cache), Metal.MTLResourceOptions(0))
    self.icb_commands = []
    for j,ji in enumerate(self.jit_cache):
      descriptor = Metal.MTLComputePipelineDescriptor.new()
      descriptor.setComputeFunction_(ji.prg.clprg.fxn)
      descriptor.setSupportIndirectCommandBuffers_(True)
      pipeline_state = unwrap(METAL.device.newComputePipelineStateWithDescriptor_options_reflection_error_(descriptor, Metal.MTLPipelineOption(0), None, None))
      icb_command = self.icb.indirectComputeCommandAtIndex_(j)
      icb_command.setComputePipelineState_(pipeline_state)
      for i,b in enumerate(ji.rawbufs):
        if b is not None:
          icb_command.setKernelBuffer_offset_atIndex_(b._buf, 0, i)
      var_vals_keys = list(var_vals.keys())
      for i,v in enumerate(getattr(ji.prg,"vars",[])):
        icb_command.setKernelBuffer_offset_atIndex_(self.int_buf._buf, var_vals_keys.index(v)*4, len(ji.rawbufs)+i)
      global_size, local_size = ji.prg.launch_dims(var_vals)
      assert ji.prg.global_size and ji.prg.local_size, "need global and local size to JIT"
      if any(isinstance(x, Node) for x in ji.prg.global_size) or any(isinstance(x, Node) for x in ji.prg.local_size):
        self.input_has_variable_dims.add(j)
      else:
        icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
      icb_command.setBarrier()
      self.icb_commands.append(icb_command)
    self.range = Metal.MTLIndirectCommandBufferExecutionRangeMake(0,len(self.jit_cache))

  def __call__(self, input_rawbuffers: Dict[Union[int, str], RawBuffer], var_vals: Dict[Variable, int]):
    if self.command_buffer is not None and self.command_buffer in METAL.mtl_buffers_in_flight: self.command_buffer.waitUntilCompleted()    # NOTE: you at least can't update the ints if this is running
    for (j,i),input_name in self.input_replace.items():
      self.icb_commands[j].setKernelBuffer_offset_atIndex_(input_rawbuffers[input_name]._buf, 0, i)
    for j in self.input_has_variable_dims:
      global_size, local_size = self.jit_cache[j].prg.launch_dims(var_vals)
      self.icb_commands[j].concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    self.int_buf_view[:] = list(var_vals.values())
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.executeCommandsInBuffer_withRange_(self.icb, self.range)
    # NOTE: command buffers aren't synced between each other, tests will fail without this fence
    METAL.last_fence = METAL.device.newFence()
    encoder.updateFence_(METAL.last_fence)
    encoder.endEncoding()
    command_buffer.commit()
    METAL.mtl_buffers_in_flight.append(command_buffer)
    self.command_buffer = command_buffer

MetalBuffer = Compiled(RawMetalBuffer, LinearizerOptions(device="METAL"), MetalRenderer, compile_metal, MetalProgram, METAL.synchronize, batch_executor=MetalBatchExecutor)
