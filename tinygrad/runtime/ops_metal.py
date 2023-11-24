import os, subprocess, pathlib, ctypes, tempfile
import Metal, libdispatch
from typing import List, Any, Tuple, Dict, cast, Optional
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import prod, getenv, DEBUG, DType, dtypes, diskcache, dedup
from tinygrad.ops import Compiled, CompiledASTRunner, update_stats
from tinygrad.renderer.metal import MetalRenderer
from tinygrad.runtime.lib import RawBufferMapped, RawBuffer, LRUAllocator
from tinygrad.shape.symbolic import Variable
from tinygrad.jit import JitItem, get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims, GraphException

class MetalAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs):
    buf_len, max_buf_len = size*dtype.itemsize, METAL.device.maxBufferLength()
    assert buf_len < max_buf_len, f"Buffer length of {buf_len/1e9:5.2f} GB exceeds Metal's max buffer length of {max_buf_len/1e9:5.2f} GB."
    buf = METAL.device.newBufferWithLength_options_(buf_len, Metal.MTLResourceStorageModeShared)
    assert buf, f"Metal buffer allocation failed with {buf}."
    return buf
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
  return library.libraryDataContents().bytes().tobytes()

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

class MetalGraph:
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[RawBuffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache)
    self.jc_idx_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)

    # create metal batch exec
    icb_descriptor = Metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes_(Metal.MTLIndirectCommandType(Metal.MTLIndirectCommandTypeConcurrentDispatch))
    icb_descriptor.setInheritBuffers_(False)
    icb_descriptor.setInheritPipelineState_(False)
    icb_descriptor.setMaxKernelBufferBindCount_(31)
    self.icb = METAL.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(icb_descriptor, len(self.jit_cache), Metal.MTLResourceOptions(0))
    if self.icb is None: raise GraphException("create indirect command buffer failed, does your system support this?")

    self.int_buf = RawMetalBuffer(len(var_vals), dtypes.int32)
    read_resources, write_resources = [], []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledASTRunner = cast(CompiledASTRunner, ji.prg)
      descriptor = Metal.MTLComputePipelineDescriptor.new()
      descriptor.setComputeFunction_(prg.clprg.fxn)
      descriptor.setSupportIndirectCommandBuffers_(True)
      pipeline_state = unwrap(METAL.device.newComputePipelineStateWithDescriptor_options_reflection_error_(descriptor, Metal.MTLPipelineOption(0), None, None))
      icb_command = self.icb.indirectComputeCommandAtIndex_(j)
      icb_command.setComputePipelineState_(pipeline_state)
      for i,b in enumerate(ji.rawbufs):
        if b is not None:
          icb_command.setKernelBuffer_offset_atIndex_(b._buf, 0, i)
          if i == 0: write_resources.append(b._buf)
          else: read_resources.append(b._buf)
      var_vals_keys = list(var_vals.keys())
      for i,v in enumerate(prg.vars):
        icb_command.setKernelBuffer_offset_atIndex_(self.int_buf._buf, var_vals_keys.index(v)*4, len(ji.rawbufs)+i)
      if j not in self.jc_idx_with_updatable_launch_dims:
        global_size, local_size = prg.launch_dims(var_vals)
        icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
      icb_command.setBarrier()
    self.read_resources, self.write_resources = dedup(read_resources), dedup(write_resources)
    self.command_buffer: Any = None
    self.int_buf_view = self.int_buf.buffer_view()    # TODO: this is metal syncing when it doesn't need to

  def __call__(self, input_rawbuffers: List[RawBuffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # NOTE: you at least can't update the ints if this is running
    if self.command_buffer is not None and self.command_buffer in METAL.mtl_buffers_in_flight: self.command_buffer.waitUntilCompleted()
    all_read_resources = self.read_resources + [x._buf for x in input_rawbuffers]
    for (j,i),input_idx in self.input_replace.items():
      self.icb.indirectComputeCommandAtIndex_(j).setKernelBuffer_offset_atIndex_(input_rawbuffers[input_idx]._buf, 0, i)
    for j in self.jc_idx_with_updatable_launch_dims:
      global_size, local_size = cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.icb.indirectComputeCommandAtIndex_(j).concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    self.int_buf_view[:] = list(var_vals.values())
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.executeCommandsInBuffer_withRange_(self.icb, Metal.MTLIndirectCommandBufferExecutionRangeMake(0,len(self.jit_cache)))
    encoder.useResources_count_usage_(all_read_resources, len(all_read_resources), Metal.MTLResourceUsageRead)
    encoder.useResources_count_usage_(self.write_resources, len(self.write_resources), Metal.MTLResourceUsageWrite)
    encoder.endEncoding()
    command_buffer.commit()
    self.command_buffer = command_buffer
    if wait:
      command_buffer.waitUntilCompleted()
      et = command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    else:
      METAL.mtl_buffers_in_flight.append(command_buffer)
      et = None
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers), jit=jit, num_kernels=len(self.jit_cache))
    return et

MetalBuffer = Compiled(RawMetalBuffer, LinearizerOptions(device="METAL"), MetalRenderer, compile_metal, MetalProgram, METAL.synchronize, graph=MetalGraph)
