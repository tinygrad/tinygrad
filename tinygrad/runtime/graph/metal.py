from typing import List, Any, Dict, cast, Optional
import Metal
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, unwrap2, GraphException, colored
from tinygrad.device import Buffer, CompiledRunner, Runner
from tinygrad.engine.realize import ExecItem
from tinygrad.engine.jit import get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_metal import MetalDevice, wait_check

class MetalGraph(Runner):
  def __init__(self, device:MetalDevice, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.jc_idx_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.device: MetalDevice = device

    # create metal batch exec
    icb_descriptor = Metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes_(Metal.MTLIndirectCommandType(Metal.MTLIndirectCommandTypeConcurrentDispatch))
    icb_descriptor.setInheritBuffers_(False)
    icb_descriptor.setInheritPipelineState_(False)
    icb_descriptor.setMaxKernelBufferBindCount_(31)
    self.icb = self.device.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(icb_descriptor, len(self.jit_cache), Metal.MTLResourceOptions(0))  # noqa: E501
    if self.icb is None: raise GraphException("create indirect command buffer failed, does your system support this?")

    if len(var_vals): self.int_buf = self.device.allocator.alloc(len(var_vals)*dtypes.int32.itemsize)
    all_resources = [self.int_buf] if len(var_vals) else []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      descriptor = Metal.MTLComputePipelineDescriptor.new()
      descriptor.setComputeFunction_(prg.clprg.fxn)
      descriptor.setSupportIndirectCommandBuffers_(True)
      pipeline_state = unwrap2(self.device.device.newComputePipelineStateWithDescriptor_options_reflection_error_(descriptor, Metal.MTLPipelineOption(0), None, None))  # noqa: E501
      icb_command = self.icb.indirectComputeCommandAtIndex_(j)
      icb_command.setComputePipelineState_(pipeline_state)
      for i,b in enumerate(ji.rawbufs):
        if b is not None:
          icb_command.setKernelBuffer_offset_atIndex_(b._buf, 0, i)
          all_resources.append(b._buf)
      var_vals_keys = list(var_vals.keys())
      for i,v in enumerate(prg.vars):
        icb_command.setKernelBuffer_offset_atIndex_(self.int_buf, var_vals_keys.index(v)*4, len(ji.rawbufs)+i)
      if j not in self.jc_idx_with_updatable_launch_dims:
        global_size, local_size = prg.launch_dims(var_vals)
        icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
      icb_command.setBarrier()
    self.all_resources = dedup(all_resources)
    self.command_buffer: Any = None
    if len(var_vals): self.int_buf_view = self.int_buf.contents().as_buffer(self.int_buf.length()).cast('i')

    # clear jit inputs to allow their memory to be freed/reused
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None
    super().__init__(colored(f"<batched {len(self.jit_cache)}>", "cyan"), device.dname, *get_jit_stats(jit_cache))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # NOTE: you at least can't update the ints if this is running
    if self.command_buffer is not None and self.command_buffer in self.device.mtl_buffers_in_flight: wait_check(self.command_buffer)
    all_resources = self.all_resources + [x._buf for x in input_rawbuffers]
    for (j,i),input_idx in self.input_replace.items():
      self.icb.indirectComputeCommandAtIndex_(j).setKernelBuffer_offset_atIndex_(input_rawbuffers[input_idx]._buf, 0, i)
    for j in self.jc_idx_with_updatable_launch_dims:
      global_size, local_size = cast(CompiledRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.icb.indirectComputeCommandAtIndex_(j).concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))  # noqa: E501
    for j, value in enumerate(var_vals.values()): self.int_buf_view[j] = value
    command_buffer = self.device.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.useResources_count_usage_(all_resources, len(all_resources), Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)
    encoder.executeCommandsInBuffer_withRange_(self.icb, Metal.MTLIndirectCommandBufferExecutionRangeMake(0,len(self.jit_cache)))
    encoder.endEncoding()
    command_buffer.commit()
    self.command_buffer = command_buffer
    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    self.device.mtl_buffers_in_flight.append(command_buffer)
    return None