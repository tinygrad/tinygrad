from pyodide.ffi import run_sync, to_js
import functools, struct, js
from tinygrad.device import  Compiled, Allocator, Compiler
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import round_up

def to_js_obj(a: dict): return to_js(a, dict_converter=js.Object.fromEntries)

def create_uniform(wgpu_device, val):
  buf = wgpu_device.createBuffer(size=4, usage=js.GPUBufferUsage.UNIFORM | js.GPUBufferUsage.COPY_DST)
  val = val.to_bytes(4, "little") if isinstance(val, int) else struct.pack('<f', val)
  array = to_js(val)
  wgpu_device.queue.writeBuffer(buf, 0, array)
  return buf

class WebGPUProgram:
  def __init__(self, dev, name:str, lib:bytes):
    (self.dev, self.timestamp_supported) = dev
    self.name, self.lib, self.prg = name, lib, self.dev.createShaderModule(code=lib.decode())   # NOTE: this is the compiler
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    wait = wait and self.timestamp_supported
    binding_layouts = [{"binding": 0, "visibility": js.GPUShaderStage.COMPUTE, "buffer": {"type": "uniform" }}]
    binding_layouts += [{"binding": i+1, "visibility": js.GPUShaderStage.COMPUTE,
                        "buffer": {"type": "uniform" if i >= len(bufs) else "storage" }} for i in range(len(bufs)+len(vals))]  # noqa: E501
    binding_layouts = [to_js(binding, dict_converter=js.Object.fromEntries) for binding in binding_layouts]
    bindings = [{"binding": 0, "resource": {"buffer": create_uniform(self.dev, float('inf')), "offset": 0, "size": 4}}]
    bindings += [{"binding": i+1, "resource": {"buffer": create_uniform(self.dev, x) if i >= len(bufs) else x, "offset": 0,
                                            "size": 4 if i >= len(bufs) else x.size}} for i,x in enumerate(bufs+vals)]  # noqa: E501
    bindings = [to_js(binding, dict_converter=js.Object.fromEntries) for binding in bindings]
    bind_group_layout = self.dev.createBindGroupLayout(entries=binding_layouts)
    pipeline_layout = self.dev.createPipelineLayout(bindGroupLayouts=[bind_group_layout])
    bind_group = self.dev.createBindGroup(layout=bind_group_layout, entries=bindings)
    compute_pipeline = self.dev.createComputePipeline(layout=pipeline_layout,compute=to_js_obj({"module": self.prg, "entry_point": self.name}),)
    command_encoder = self.dev.createCommandEncoder()
    if wait:
      query_set = self.dev.createQuerySet(type="timestamp", count=2)
      query_buf = self.dev.createBuffer(size=16, usage=js.GPUBufferUsage.QUERY_RESOLVE | js.GPUBufferUsage.COPY_SRC)
      timestamp_writes = {"query_set": query_set, "beginning_of_pass_write_index": 0, "end_of_pass_write_index": 1}
    compute_pass = command_encoder.beginComputePass(timestampWrites=timestamp_writes if wait else None) # pylint: disable=E0606
    compute_pass.setPipeline(compute_pipeline)
    compute_pass.setBindGroup(0, bind_group) # last 2 not used
    compute_pass.dispatchWorkgroups(*global_size)  # x y z
    compute_pass.end()
    if wait:
      command_encoder.resolve_query_set(query_set=query_set, first_query=0, query_count=2, destination=query_buf, destination_offset=0)
    self.dev.queue.submit([command_encoder.finish()])
    return ((timestamps:=self.dev.queue.read_buffer(query_buf).cast("Q").tolist())[1] - timestamps[0]) / 1e9 if wait else None

# WebGPU buffers have to be 4-byte aligned
class WebGpuAllocator(Allocator):
  def __init__(self, dev): self.dev = dev
  def _alloc(self, size: int, options):
    return self.dev.createBuffer(size=round_up(size, 4), usage=js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST | js.GPUBufferUsage.COPY_SRC)
  def _copyin(self, dest, src: memoryview):
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    self.dev.queue.writeBuffer(dest, 0, to_js(padded_src if src.nbytes % 4 else src))
  def _copyout(self, dest: memoryview, src):
    output_buf = self.dev.createBuffer(label="outputbuf", size=src.size, usage=js.GPUBufferUsage.COPY_DST | js.GPUBufferUsage.MAP_READ)
    commandEncoder = self.dev.createCommandEncoder(label="copyout")
    commandEncoder.copyBufferToBuffer(
      src,
      0,
      output_buf,
      0,
      src.size,
    )
    self.dev.queue.submit([commandEncoder.finish()])
    run_sync(output_buf.mapAsync(
      js.GPUMapMode.READ,
      0,
      src.size,
    ))
    buffer_data = output_buf.getMappedRange(0, src.size).to_py()
    dest[:] = buffer_data[:dest.nbytes] if buffer_data.nbytes > dest.nbytes else buffer_data

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    navigator = js.navigator
    adapter = run_sync(navigator.gpu.requestAdapter())
    required_features = []
    timestamp_supported = False # adapter.features.has("timestamp-query")
    if timestamp_supported: required_features.append("timestamp-query")
    wgpu_device = run_sync(adapter.requestDevice(to_js(required_features)))
    super().__init__(device, WebGpuAllocator(wgpu_device), WGSLRenderer(), Compiler(),
                     functools.partial(WebGPUProgram, (wgpu_device, timestamp_supported)))