import functools
from wgpu.utils.device import get_default_device
from tinygrad.helpers import dtypes, DType
from tinygrad.device import Compiled, Allocator
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle
from tinygrad.renderer.wgsl import WGSLLanguage
import wgpu

wgpu_device = get_default_device()

class WebGPUProgram:
  def __init__(self, name:str, lib:bytes): self.name, self.lib, self.prg = name, lib, wgpu_device.create_shader_module(code=lib)   # NOTE: this is the compiler
  def __call__(self, *bufs, global_size, local_size, wait=False):
    assert len(bufs) <= 8, "WEBGPU only supports 8 buffers"
    binding_layouts = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}} for i in range(len(bufs))]
    bindings = [{"binding": i, "resource": {"buffer": x, "offset": 0, "size": x.size}} for i, x in enumerate(bufs)]
    bind_group_layout = wgpu_device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = wgpu_device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = wgpu_device.create_bind_group(layout=bind_group_layout, entries=bindings)
    compute_pipeline = wgpu_device.create_compute_pipeline(layout=pipeline_layout,compute={"module": self.prg, "entry_point": self.name},)
    command_encoder = wgpu_device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999) # last 2 not used
    compute_pass.dispatch_workgroups(*global_size)  # x y z
    compute_pass.end()
    wgpu_device.queue.submit([command_encoder.finish()])

class WebGpuAllocator(Allocator):
  def _alloc(self, size: int, dtype: DType):
    assert dtype not in [dtypes.int8,dtypes.uint8,dtypes.int64,dtypes.uint64,dtypes.double], f"dtype {dtype} not supported on WEBGPU"
    return wgpu_device.create_buffer(size=size*dtype.itemsize, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
  def copyin(self, dest, src: memoryview): wgpu_device.queue.write_buffer(dest, 0, src)
  def copyout(self, dest, src: memoryview): dest[:] = wgpu_device.queue.read_buffer(src, 0)    # TODO: remove this copy

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(WebGpuAllocator(), LinearizerOptions(device="WEBGPU", supports_float4=False, local_max=[256, 256, 64], global_max=[65535, 65535, 65535]),
                     functools.partial(uops_to_cstyle, WGSLLanguage()), lambda x: x, WebGPUProgram)
