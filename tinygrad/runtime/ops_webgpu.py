import functools
from tinygrad.device import  Compiled, Allocator, Compiler
from tinygrad.renderer.wgsl import WGSLRenderer
import wgpu

class WGSLCompiler(Compiler):
  def compile(self, src):
    return src.encode()

def create_uniform(wgpu_device, val: int) -> wgpu.GPUBuffer:
  buf = wgpu_device.create_buffer(size=4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
  wgpu_device.queue.write_buffer(buf, 0, val.to_bytes(4, "little"))
  return buf

class WebGPUProgram:
  def __init__(self, device, name:str, lib:bytes):
    (self.device, self.timestamp_supported) = device
    self.name, self.lib, self.prg = name, lib, self.device.create_shader_module(code=lib.decode())   # NOTE: this is the compiler
    # self.max_buffers = self.device.limits["max_storage_buffers_per_shader_stage"]
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    # assert len(bufs) <= self.max_buffers, f"WEBGPU only supports {self.max_buffers} buffers"
    wait = wait and self.timestamp_supported
    binding_layouts = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.uniform if i >= len(bufs) else wgpu.BufferBindingType.storage }} for i in range(len(bufs)+len(vals))]  # noqa: E501
    bindings = [{"binding": i, "resource": {"buffer": create_uniform(self.device, x) if i >= len(bufs) else x, "offset": 0,
                                            "size": 4 if i >= len(bufs) else x.size}} for i,x in enumerate(bufs+vals)]  # noqa: E501
    bind_group_layout = self.device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = self.device.create_bind_group(layout=bind_group_layout, entries=bindings)
    compute_pipeline = self.device.create_compute_pipeline(layout=pipeline_layout,compute={"module": self.prg, "entry_point": self.name},)
    command_encoder = self.device.create_command_encoder()
    if wait:
      query_set = self.device.create_query_set(type=wgpu.QueryType.timestamp, count=2)
      query_buf = self.device.create_buffer(size=16, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
      timestamp_writes = {"query_set": query_set, "beginning_of_pass_write_index": 0, "end_of_pass_write_index": 1}
    compute_pass = command_encoder.begin_compute_pass(timestamp_writes=timestamp_writes if wait else None) # pylint: disable=E0606
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999) # last 2 not used
    compute_pass.dispatch_workgroups(*global_size)  # x y z
    compute_pass.end()
    if wait:
      command_encoder.resolve_query_set(query_set=query_set, first_query=0, query_count=2, destination=query_buf, destination_offset=0)
    self.device.queue.submit([command_encoder.finish()])
    return ((timestamps:=self.device.queue.read_buffer(query_buf).cast("Q").tolist())[1] - timestamps[0]) / 1e9 if wait else None

class WebGpuAllocator(Allocator):
  def __init__(self, device): self.device = device
  def _alloc(self, size: int, options):
    if options.wgpu_bool: size = 4 * size # storing bools as i32
    return self.device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
  # copies are hacky for booleans.
  def copyin(self, dest, src: memoryview):
    if dest.size == len(src) * 4: self.device.queue.write_buffer(dest, 0, bytearray([byte for b in src for byte in [b, 0, 0, 0]]))
    else: self.device.queue.write_buffer(dest, 0, src)
  def copyout(self, dest: memoryview, src):
    dest[:] = self.device.queue.read_buffer(src, 0)[::4] if src.size == 4 * len(dest) else self.device.queue.read_buffer(src, 0)

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    timestamp_supported = wgpu.FeatureName.timestamp_query in adapter.features
    wgpu_device = adapter.request_device(required_features=[wgpu.FeatureName.timestamp_query] if timestamp_supported else [])
    super().__init__(device, WebGpuAllocator(wgpu_device), WGSLRenderer(), WGSLCompiler(),
                     functools.partial(WebGPUProgram, (wgpu_device, timestamp_supported)))
