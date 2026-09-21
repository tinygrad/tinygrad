from typing import Any, cast
import ctypes, decimal, struct, itertools
from tinygrad.helpers import dedup, getenv, unwrap, PROFILE, round_up
from tinygrad.device import Buffer, Device, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import UOp
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.runtime.ops_metal import MetalDevice, wait_check, to_ns_str
from tinygrad.runtime.autogen import metal

class MetalGraph(GraphRunner):
  def __init__(self, linear, input_uops=()):
    super().__init__(linear, input_uops)
    self.dev = cast(MetalDevice, Device[self.device])

    # create metal batch exec
    icb_descriptor = metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    icb_descriptor.setInheritBuffers(False)
    icb_descriptor.setInheritPipelineState(False)
    icb_descriptor.setMaxKernelBufferBindCount(1)

    self.icb = self.dev.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(icb_descriptor, len(self.calls),
                                                                                                 metal.MTLResourceCPUCacheModeDefaultCache)
    if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?")
    self.needs_icb_fix = int(not self.dev.arch.startswith("Apple") or int(self.dev.arch[5:]) < 9)  # ICB fix not required on M3+ (Apple9+)

    # one args buffer: a 256-aligned struct region per kernel holding its buffer addresses and scalars
    self.layouts = [unwrap(r).layout for r in self.runtimes]
    self.offs = list(itertools.accumulate([round_up(unwrap(r).args_size, 256) for r in self.runtimes], initial=0))
    storage = self.dev.allocator.alloc(self.offs[-1])
    self.var_buf, self.var_buf_view, self.var_bind_data = storage.buf, unwrap(storage.host).mv, []

    all_pipelines, all_resources = [], [self.var_buf.buf]
    for j, ((_, ast, bufs, _), runtime, replace) in enumerate(zip(self.calls, self.runtimes, self.uop_replace)):
      assert runtime is not None
      icb_command = self.icb.indirectComputeCommandAtIndex(j).retained()
      icb_command.setComputePipelineState(runtime.pipeline_state)
      all_pipelines.append(runtime.pipeline_state)
      icb_command.setKernelBuffer_offset_atIndex(self.var_buf.buf, self.offs[j], 0)
      for i, b in enumerate(bufs):
        if not any(pos == i for pos, _ in replace):
          struct.pack_into("Q", self.var_buf_view, self.offs[j] + self.layouts[j][i][0], b._buf.addr)
          all_resources.append(b._buf.buf)
      for (nm,_,dt,_), (off, _) in zip(runtime.signature[len(bufs):], self.layouts[j][len(bufs):]):
        self.var_bind_data.append((nm, self.offs[j] + off, dt.fmt))
      global_size, local_size = ast.arg.launch_dims({v: 0 for v in self.vars})
      icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*global_size), metal.MTLSize(*local_size))
      icb_command.setBarrier()

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.command_buffer: Any = None
    self.range = metal.NSRange(0, len(self.calls))
    self.updatable = sorted({j for j,r in enumerate(self.uop_replace) if r} | self.var_vals_replace.keys() | self.launch_dims_replace.keys())

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False):
    if self.command_buffer is not None and self.command_buffer in self.dev.mtl_buffers_in_flight: wait_check(self.command_buffer)
    # NOTE: old command buffer may not be inflight anymore
    if self.command_buffer is not None and PROFILE: self.collect_timestamps()

    updated_bufs = []
    for j in self.updatable:
      for pos, iidx in self.uop_replace[j]:
        buf = cast(Buffer, input_uops[iidx].buffer)
        struct.pack_into("Q", self.var_buf_view, self.offs[j] + self.layouts[j][pos][0], buf._buf.addr)
        updated_bufs.append(buf._buf.buf)

    all_resources = dedup(self.all_resources + updated_bufs)
    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      self.icb.indirectComputeCommandAtIndex(j).concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*global_dims),
                                                                                                     metal.MTLSize(*local_dims))
    for nm,ofs,fmt in self.var_bind_data: struct.pack_into(fmt, self.var_buf_view, ofs, var_vals[nm])

    command_buffer = self.dev.mtl_queue.commandBuffer().retained()
    encoder = command_buffer.computeCommandEncoder().retained()
    encoder.useResources_count_usage(ctypes.cast((metal.MTLBuffer * len(all_resources))(*all_resources), ctypes.POINTER(metal.MTLResource)),
                                     len(all_resources), metal.MTLResourceUsageRead | metal.MTLResourceUsageWrite)

    # NOTE: the pipelines likely need to be added to the used resources to fix the crash on M1/M2, but I haven't figured out how
    # this is a O(n) hack to get them used. what should work is:
    #encoder.useResources_count_usage_(self.all_pipelines, len(self.all_pipelines), Metal.MTLResourceUsageRead)
    # but it fails with "Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)"
    # to repro the crash (which can also crash other running GPU apps), run with FIX_METAL_ICB=0
    if getenv("FIX_METAL_ICB", self.needs_icb_fix):
      for ps in self.all_pipelines:
        encoder.setComputePipelineState(ps)
        encoder.dispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(0,0,0), metal.MTLSize(0,0,0))

    encoder.executeCommandsInBuffer_withRange(self.icb, self.range)
    encoder.endEncoding()
    command_buffer.setLabel(to_ns_str(f"batched {len(self.calls)}"))
    command_buffer.commit()
    self.command_buffer = command_buffer

    self.dev.mtl_buffers_in_flight.append(command_buffer)
    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    return None

  def collect_timestamps(self):
    # create a graph event and evenly space each program
    st, en = decimal.Decimal(self.command_buffer.GPUStartTime()) * 1000000, decimal.Decimal(self.command_buffer.GPUEndTime()) * 1000000
    ents = [ProfileGraphEntry(self.device, rt.name, i, i+1, rt.profile_key) for i, rt in enumerate(self.runtimes) if rt is not None]
    self.dev.profile_events += [ProfileGraphEvent(ents, [], [st + (en-st)/len(ents)*i for i in range(len(ents)+1)])]

  def __del__(self):
    if PROFILE and self.command_buffer is not None:
      wait_check(self.command_buffer)
      self.collect_timestamps()
