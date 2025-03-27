from typing import cast
import ctypes
from tinygrad.helpers import dedup
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.runtime.autogen import webgpu
from tinygrad.runtime.ops_webgpu import read_buffer, WebGPUProgram
from tinygrad.device import Buffer, Device
from tinygrad.ops import Variable

class WebGPUGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[Variable, int]):
    # needed for exporting; TODO: extract only metadata about dtype, size
    self.input_rawbuffers, self.var_vals = input_rawbuffers, var_vals
    
    # TODO: capture this more cleanly?
    self._dev, self.timestamp_supported = (_prg:=jit_cache[0].prg._prg).dev, _prg.timestamp_supported

    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg._prg, WebGPUProgram) for ji in jit_cache): raise GraphException
    self.jc_idx_with_updatable_rawbufs = dedup([x[0] for x in self.input_replace.keys()])

    #prgs = '\n'.join(dedup([cast(CompiledRunner, ji.prg).p.src for ji in jit_cache]))

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False) -> float|None:
    for (j,i),idx in self.input_replace.items(): self.jit_cache[j].bufs[i] = rawbufs[idx]

    wait = wait and self.timestamp_supported

    # TODO: refactor to deduplicate encoder/wait with ops_webgpu.WebGPUProgram.__call__
    command_encoder = webgpu.wgpuDeviceCreateCommandEncoder(self._dev, webgpu.WGPUCommandEncoderDescriptor())
    comp_pass_desc = webgpu.WGPUComputePassDescriptor(nextInChain=None)

    if wait:
      query_set = webgpu.wgpuDeviceCreateQuerySet(self._dev, webgpu.WGPUQuerySetDescriptor(type=webgpu.WGPUQueryType_Timestamp, count=2))
      query_buf = webgpu.wgpuDeviceCreateBuffer(self._dev,
        webgpu.WGPUBufferDescriptor(size=16, usage=webgpu.WGPUBufferUsage_QueryResolve | webgpu.WGPUBufferUsage_CopySrc))
      comp_pass_desc.timestampWrites = ctypes.pointer(webgpu.WGPUComputePassTimestampWrites(
        querySet=query_set, beginningOfPassWriteIndex=0, endOfPassWriteIndex=1))

    for ji in self.jit_cache:
      _prg = cast(WebGPUProgram, (prg:=cast(CompiledRunner, ji.prg))._prg)
      vals = tuple(var_vals[k] for k in prg.p.vars)
      _prg.add_compute_pass(command_encoder, comp_pass_desc, *[b._buf for b in ji.bufs], global_size=prg.p.launch_dims(var_vals)[0], vals=vals)

    if wait: webgpu.wgpuCommandEncoderResolveQuerySet(command_encoder, query_set, 0, 2, query_buf, 0)

    cmd_buf = webgpu.wgpuCommandEncoderFinish(command_encoder, webgpu.WGPUCommandBufferDescriptor())
    webgpu.wgpuQueueSubmit(webgpu.wgpuDeviceGetQueue(self._dev), 1, (webgpu.WGPUCommandBuffer*1)(cmd_buf))

    if wait:
      time = ((timestamps:=read_buffer(self._dev, query_buf).cast("Q").tolist())[1] - timestamps[0]) / 1e9
      webgpu.wgpuBufferDestroy(query_buf)
      webgpu.wgpuQuerySetDestroy(query_set)
      return time