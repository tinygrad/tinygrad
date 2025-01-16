import functools, struct
from tinygrad.device import  Compiled, Allocator, Compiler
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import round_up
from pydawn import utils, webgpu

def create_uniform(wgpu_device, val):
  buf = utils.create_buffer(wgpu_device, 4, webgpu.WGPUBufferUsage_Uniform | webgpu.WGPUBufferUsage_CopyDst)
  utils.write_buffer(wgpu_device, buf, 0, val.to_bytes(4, "little") if isinstance(val, int) else struct.pack('<f', val))
  return buf

class WebGPUProgram:
  def __init__(self, dev, name:str, lib:bytes):
    self.dev = dev
    self.name, self.lib, self.prg = name, lib, utils.create_shader_module(self.dev, lib.decode())   # NOTE: this is the compiler
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    tmp_bufs = [*bufs]
    buf_patch = False

    # WebGPU does not allow using the same buffer for input and output
    for i in range(1, len(bufs)):
      if bufs[i] == bufs[0]:
        tmp_bufs[0] = utils.create_buffer(self.dev, webgpu.wgpuBufferGetSize(bufs[0]), webgpu.wgpuBufferGetUsage(bufs[0]))
        buf_patch = True

    binding_layouts = [{"binding": 0, "visibility": webgpu.WGPUShaderStage_Compute, "buffer": {"type": webgpu.WGPUBufferBindingType_Uniform }}]
    binding_layouts += [{"binding": i+1, "visibility": webgpu.WGPUShaderStage_Compute,
                        "buffer": {"type": webgpu.WGPUBufferBindingType_Uniform if i >= len(tmp_bufs) else webgpu.WGPUBufferBindingType_Storage }} for i in range(len(tmp_bufs)+len(vals))]  # noqa: E501
    bindings = [{"binding": 0, "resource": {"buffer": create_uniform(self.dev, float('inf')), "offset": 0, "size": 4}}]
    bindings += [{"binding": i+1, "resource": {"buffer": create_uniform(self.dev, x) if i >= len(tmp_bufs) else x, "offset": 0,
                                            "size": 4 if i >= len(tmp_bufs) else webgpu.wgpuBufferGetSize(x)}} for i,x in enumerate(tuple(tmp_bufs)+vals)]  # noqa: E501
    bind_group_layout = utils.create_bind_group_layout(self.dev, entries=binding_layouts)
    pipeline_layout = utils.create_pipeline_layout(self.dev, bind_group_layouts=[bind_group_layout])
    bind_group = utils.create_bind_group(self.dev, layout=bind_group_layout, entries=bindings)
    compute_pipeline = utils.create_compute_pipeline(self.dev, layout=pipeline_layout,compute={"module": self.prg, "entry_point": self.name})
    command_encoder = utils.create_command_encoder(self.dev)

    if wait:
      query_set = utils.create_query_set(self.dev, type=webgpu.WGPUQueryType_Timestamp, count=2)
      query_buf = utils.create_buffer(self.dev, size=16, usage=webgpu.WGPUBufferUsage_QueryResolve | webgpu.WGPUBufferUsage_CopySrc)
      timestamp_writes = {"query_set": query_set, "beginning_of_pass_write_index": 0, "end_of_pass_write_index": 1}

    compute_pass = utils.begin_compute_pass(command_encoder, timestamp_writes if wait else None) # pylint: disable=E0606
    utils.set_pipeline(compute_pass, compute_pipeline)
    utils.set_bind_group(compute_pass, bind_group)
    utils.dispatch_workgroups(compute_pass, *global_size)
    utils.end_compute_pass(compute_pass)

    if wait: utils.resolve_query_set(command_encoder, query_set, 0, 2, query_buf, 0)

    cmd_buf = utils.command_encoder_finish(command_encoder)
    utils.submit(self.dev, [cmd_buf])
    utils.sync(self.dev)

    if buf_patch:
      utils.copy_buffer_to_buffer(self.dev, tmp_bufs[0], 0, bufs[0], 0, webgpu.wgpuBufferGetSize(bufs[0]))
      webgpu.wgpuBufferDestroy(tmp_bufs[0])

    if wait:
      time = ((timestamps:=utils.read_buffer(self.dev, query_buf).cast("Q").tolist())[1] - timestamps[0]) / 1e9
      webgpu.wgpuBufferDestroy(query_buf)
      webgpu.wgpuQuerySetDestroy(query_set)
      return time

class WebGpuAllocator(Allocator):
  def __init__(self, dev): self.dev = dev
  def _alloc(self, size: int, options):
    # WebGPU buffers have to be 4-byte aligned
    return utils.create_buffer(
      self.dev, round_up(size, 4), webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_CopySrc)
  def _copyin(self, dest, src: memoryview):
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    utils.write_buffer(self.dev, dest, 0, padded_src if src.nbytes % 4 else src)
  def _copyout(self, dest: memoryview, src):
    buffer_data = utils.read_buffer(self.dev, src)
    src_len = webgpu.wgpuBufferGetSize(src)
    dest[:] = buffer_data[:dest.nbytes] if src_len  > dest.nbytes else buffer_data
  def _free(self, opaque, options): webgpu.wgpuBufferDestroy(opaque)

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    adapter = utils.request_adapter_sync(power_preference=webgpu.WGPUPowerPreference_HighPerformance)
    wgpu_device = utils.request_device_sync(adapter, required_features=[webgpu.WGPUFeatureName_TimestampQuery, webgpu.WGPUFeatureName_ShaderF16])
    super().__init__(device, WebGpuAllocator(wgpu_device), WGSLRenderer(), Compiler(),
                     functools.partial(WebGPUProgram, (wgpu_device)))
