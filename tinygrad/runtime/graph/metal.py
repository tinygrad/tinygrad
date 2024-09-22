from typing import List, Any, Dict, cast, Optional
import Metal
import tinygrad.runtime.support.metal as cdll
import ctypes
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, getenv
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
    icb_descriptor = cdll.send_message(
      cdll.libobjc.objc_getClass(b"MTLIndirectCommandBufferDescriptor"),
      "new",
    )
    cdll.send_message(icb_descriptor, "setCommandTypes:", 32)
    cdll.send_message(icb_descriptor, "setInheritBuffers:", False)
    cdll.send_message(icb_descriptor, "setInheritPipelineState:", False)
    cdll.send_message(icb_descriptor, "setMaxKernelBufferBindCount:", 31)

    self.icb = cdll.send_message(self.device.device,
      "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
      icb_descriptor, len(self.jit_cache),0)
    if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?")
    self.needs_icb_fix = int(type(self.icb).__name__ != "AGXG15XFamilyIndirectCommandBuffer")    # not required on M3

    if len(self.vars): self.int_buf = self.device.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf.buf] if len(self.vars) else []
    all_pipelines = []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      icb_command = cdll.send_message(self.icb, "indirectComputeCommandAtIndex:", j)
      all_pipelines.append(prg.clprg.pipeline_state)
      cdll.send_message(icb_command, "setComputePipelineState:", prg.clprg.pipeline_state)
      for i,b in enumerate(ji.bufs):
        if b is not None and b not in input_rawbuffers:
          cdll.send_message(icb_command, "setKernelBuffer:offset:atIndex:", b._buf.buf, b._buf.offset, i)
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars):
        cdll.send_message(icb_command, "setKernelBuffer:offset:atIndex:", self.int_buf.buf, self.vars.index(v)*4, len(ji.bufs)+i)

      global_size, local_size = prg.p.launch_dims(var_vals)
      cdll.send_message(icb_command, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", cdll.int_tuple_to_struct(global_size), cdll.int_tuple_to_struct(local_size))
      cdll.send_message(icb_command, "setBarrier")

    self.all_resources = all_resources
    self.all_pipelines = all_pipelines
    self.command_buffer: Any = None
    if len(self.vars): self.int_buf_view = self.int_buf.buf.cast('i')
    self.range = cdll.int_tuple_to_struct((0, len(self.jit_cache)), ctypes.c_uint)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:

    if self.command_buffer is not None and self.command_buffer in self.device.mtl_buffers_in_flight: wait_check(self.command_buffer)
    all_resources = self.all_resources + [x._buf.buf for x in input_rawbuffers]

    for (j,i),input_idx in self.input_replace.items():
      computeCommand = cdll.send_message(self.icb, "indirectComputeCommandAtIndex:", j)
      cdll.send_message(computeCommand, "setKernelBuffer:offset:atIndex:", input_rawbuffers[input_idx]._buf.buf,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      prg = cast(CompiledRunner, self.jit_cache[j].prg)
      global_size, local_size = global_dims or prg.p.global_size, local_dims or prg.p.local_size
      computeCommand = cdll.send_message(self.icb, "indirectComputeCommandAtIndex:", j)
      cdll.send_message(computeCommand, "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
                        cdll.int_tuple_to_struct(global_size), cdll.int_tuple_to_struct(local_size))
    for j, var in enumerate(self.vars): self.int_buf_view[j] = var_vals[var]

    command_buffer = cdll.send_message(self.device.mtl_queue, "commandBuffer")
    encoder = cdll.send_message(command_buffer, "computeCommandEncoder")
    cdll.send_message(encoder, "useResources:count:usage:", cdll.to_ns_array(all_resources), len(all_resources), 3)

    # NOTE: the pipelines likely need to be added to the used resources to fix the crash on M1/M2, but I haven't figured out how
    # this is a O(n) hack to get them used. what should work is:
    # encoder.useResources_count_usage_(self.all_pipelines, len(self.all_pipelines), Metal.MTLResourceUsageRead)
    # but it fails with "Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)"
    # to repro the crash (which can also crash other running GPU apps), run with FIX_METAL_ICB=0
    if getenv("FIX_METAL_ICB", self.needs_icb_fix):
      for ps in self.all_pipelines:
        cdll.send_message(encoder, "setComputePipelineState:", ps)
        cdll.send_message(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", cdll.int_tuple_to_struct((0,0,0)), cdll.int_tuple_to_struct((0,0,0)))

    cdll.send_message(encoder, "executeCommandsInBuffer:withRange:", self.icb, self.range)
    cdll.send_message(encoder, "endEncoding")
    cdll.send_message(command_buffer, "commit")
    self.command_buffer = command_buffer

    if wait:
      wait_check(command_buffer)
      return cdll.send_message(command_buffer, "GPUEndTime") - cdll.send_message(command_buffer, "GPUStartTime")
    self.device.mtl_buffers_in_flight.append(command_buffer)
    return None
