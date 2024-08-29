from typing import List, Any, Dict, cast, Optional
import Metal
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, unwrap2
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_metal import wait_check

class MetalGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    # create metal batch exec
    icb_descriptor = Metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes_(Metal.MTLIndirectCommandType(Metal.MTLIndirectCommandTypeConcurrentDispatch))
    icb_descriptor.setInheritBuffers_(False)
    icb_descriptor.setInheritPipelineState_(False)
    #icb_descriptor.setInheritPipelineState_(True)
    icb_descriptor.setMaxKernelBufferBindCount_(31)
    self.icb = self.device.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(icb_descriptor, len(self.jit_cache),
                                                                                                  Metal.MTLResourceOptions(Metal.MTLResourceStorageModePrivate | Metal.MTLResourceHazardTrackingModeTracked))
    if self.icb is None: raise GraphException("create indirect command buffer failed, does your system support this?")
    #print(self.icb.gpuResourceID())
    #print(self.icb.getAllocatedSize())

    if len(self.vars): self.int_buf = self.device.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    #all_resources = [self.int_buf.buf, self.icb] if len(self.vars) else [self.icb]
    all_resources = [self.int_buf.buf] if len(self.vars) else []
    self.prg_resources = []

    assert len(self.jit_cache) == 1
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = self.icb.indirectComputeCommandAtIndex_(j)
      #icb_command.reset()
      icb_command.setComputePipelineState_(prg.clprg.pipeline_state)
      self.pipeline_state = prg.clprg.pipeline_state
      self.fxn = prg.clprg.fxn
      self.library = prg.clprg.library
      #self.ba = prg.clprg.ba
      #rsrc = self.pipeline_state.pipelineBinaries()['compute'][0]
      #print(dir(rsrc))
      #print(rsrc)
      #all_resources.append(rsrc)
      #all_resources.append(prg.clprg.fxn)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          icb_command.setKernelBuffer_offset_atIndex_(b._buf.buf, b._buf.offset, i)
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars): icb_command.setKernelBuffer_offset_atIndex_(self.int_buf.buf, self.vars.index(v)*4, len(ji.bufs)+i)
      global_size, local_size = prg.p.launch_dims(var_vals)
      icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
      icb_command.setBarrier()
      self.prg_resources.append(prg.clprg.pipeline_state) #.gpuResourceID())

    self.range = Metal.MTLIndirectCommandBufferExecutionRangeMake(0, 1)
    self.all_resources = dedup(all_resources)
    self.command_buffer: Any = None
    if len(self.vars): self.int_buf_view = self.int_buf.buf.contents().as_buffer(self.int_buf.buf.length()).cast('i')

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    if self.command_buffer is not None and self.command_buffer in self.device.mtl_buffers_in_flight: wait_check(self.command_buffer)
    all_resources = dedup(self.all_resources + [x._buf.buf for x in input_rawbuffers])

    for (j,i),input_idx in self.input_replace.items():
      assert False
      self.icb.indirectComputeCommandAtIndex_(j).setKernelBuffer_offset_atIndex_(input_rawbuffers[input_idx]._buf.buf,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      assert False
      prg = cast(CompiledRunner, self.jit_cache[j].prg)
      global_size, local_size = global_dims or prg.p.global_size, local_dims or prg.p.local_size
      self.icb.indirectComputeCommandAtIndex_(j).concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size),
                                                                                                       Metal.MTLSize(*local_size))
    for j, var in enumerate(self.vars):
      assert False
      self.int_buf_view[j] = var_vals[var]

    command_buffer = self.device.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    #print(self.pipeline_state)
    #encoder.useResource_usage_(self.pipeline_state, Metal.MTLResourceUsageRead)

    #print(all_resources)
    #for r in all_resources: encoder.useResource_usage_(r, Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)
    #print("here")
    #print(self.pipeline_state.gpuAddress())
    #encoder.useResource_usage_(self.ba, Metal.MTLResourceUsageRead)
    #print(all_resources)

    # Invalid Resource 
    #encoder.useResource_usage_(self.library.libraryDataContents(), Metal.MTLResourceUsageRead)
    #encoder.useResource_usage_(self.pipeline_state, Metal.MTLResourceUsageRead)

    #print("here2")
    #print(self.icb.gpuResourceID())
    #print(self.pipeline_state.gpuResourceID())
    #encoder.useResource_usage_(self.icb, Metal.MTLResourceUsageRead)
    #encoder.useResource_usage_(self.pipeline_state, Metal.MTLResourceUsageRead)

    #print(dir(self.icb))
    #print(self.icb.resourceRef())
    #print("here2")
    #encoder.useResources_count_usage_(self.prg_resources, len(self.prg_resources), Metal.MTLResourceUsageRead)
    #all_resources.append(self.pipeline_state)
    encoder.useResources_count_usage_(all_resources, len(all_resources), Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)
    #encoder.useResource_usage_(self.icb, Metal.MTLResourceUsageRead)
    #print("here3")
    encoder.setComputePipelineState_(self.pipeline_state)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(0,0,0), Metal.MTLSize(0,0,0))

    encoder.executeCommandsInBuffer_withRange_(self.icb, self.range)
    #encoder.executeCommandsInBuffer_withRange_(self.icb, Metal.MTLIndirectCommandBufferExecutionRangeMake(0, len(self.jit_cache)))
    encoder.endEncoding()
    self.encoder = encoder
    #print(command_buffer)
    #print(dir(command_buffer))
    command_buffer.commit()
    self.command_buffer = command_buffer

    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    self.device.mtl_buffers_in_flight.append(command_buffer)
    return None
