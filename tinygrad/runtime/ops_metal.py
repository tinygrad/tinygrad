import os, subprocess, pathlib, ctypes, tempfile
import numpy as np
import Metal, libdispatch
from typing import List, Any, Tuple, Dict, cast, Optional
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import prod, getenv, DEBUG, DType, dtypes, diskcache, dedup
from tinygrad.device import Compiled, CompiledASTRunner, update_stats, Buffer, Device, Allocator
from tinygrad.renderer.metal import MetalRenderer
from tinygrad.shape.symbolic import Variable
from tinygrad.jit import JitItem, get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims, GraphException

class _METAL:
  def __init__(self):
    self.mtl_buffers_in_flight: List[Any] = []
    self.device = Metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = self.device.newCommandQueueWithMaxCommandBufferCount_(1024)
METAL = _METAL()

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
      if isinstance(a, Metal.AGXG15XFamilyBuffer): encoder.setBuffer_offset_atIndex_(a, 0, i)
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
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
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

    if len(var_vals): self.int_buf = Device[input_rawbuffers[0].device].alloc(len(var_vals), dtypes.int32)
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
        icb_command.setKernelBuffer_offset_atIndex_(self.int_buf, var_vals_keys.index(v)*4, len(ji.rawbufs)+i)
      if j not in self.jc_idx_with_updatable_launch_dims:
        global_size, local_size = prg.launch_dims(var_vals)
        icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
      icb_command.setBarrier()
    self.read_resources, self.write_resources = dedup(read_resources), dedup(write_resources)
    self.command_buffer: Any = None
    if len(var_vals): self.int_buf_view = np.frombuffer(self.int_buf.contents().as_buffer(self.int_buf.length()), np.int32)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # NOTE: you at least can't update the ints if this is running
    if self.command_buffer is not None and self.command_buffer in METAL.mtl_buffers_in_flight: self.command_buffer.waitUntilCompleted()
    all_read_resources = self.read_resources + [x._buf for x in input_rawbuffers]
    for (j,i),input_idx in self.input_replace.items():
      self.icb.indirectComputeCommandAtIndex_(j).setKernelBuffer_offset_atIndex_(input_rawbuffers[input_idx]._buf, 0, i)
    for j in self.jc_idx_with_updatable_launch_dims:
      global_size, local_size = cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.icb.indirectComputeCommandAtIndex_(j).concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    if len(var_vals): self.int_buf_view[:] = list(var_vals.values())
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

class MetalAllocator(Allocator):
  def __init__(self, device): self.device = device
  def alloc(self, size:int, dtype:DType):
    ret = METAL.device.newBufferWithLength_options_(size*dtype.itemsize, Metal.MTLResourceStorageModeShared)
    if ret is None: raise MemoryError(f"Metal OOM while allocating {size=} {dtype=}")
    return ret
  def free(self, opaque): opaque.release()
  def _copy(self, dest, src):
    assert src.length() == dest.length(), f"length mismatch {src.length()=} {dest.length()=}"
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.blitCommandEncoder()
    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(src, 0, dest, 0, src.length())
    encoder.endEncoding()
    command_buffer.commit()
    METAL.mtl_buffers_in_flight.append(command_buffer)
  def _buffer(self, src): return METAL.device.newBufferWithBytesNoCopy_length_options_deallocator_(src, len(src), Metal.MTLResourceStorageModeShared, None)
  def copyin(self, dest, src:memoryview):
    self.device.copies_in_flight.append(src)
    self._copy(dest, self._buffer(src))
  def copyout(self, dest:memoryview, src):
    self._copy(self._buffer(dest), src)
    self.device.synchronize()

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.copies_in_flight: List[memoryview] = []
    super().__init__(MetalAllocator(self), LinearizerOptions(device="METAL"), MetalRenderer, compile_metal, MetalProgram, graph=MetalGraph)
  def synchronize(self):
    for cbuf in METAL.mtl_buffers_in_flight: cbuf.waitUntilCompleted()
    self.copies_in_flight.clear()
    METAL.mtl_buffers_in_flight.clear()
