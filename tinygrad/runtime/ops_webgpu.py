import functools, struct
from tinygrad.device import  Compiled, Allocator, Compiler
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import round_up
from tinygrad.runtime.autogen import webgpu
from typing import List, Any
import ctypes

instDesc = webgpu.WGPUInstanceDescriptor()
instDesc.features.timedWaitAnyEnable = True
instance = webgpu.wgpuCreateInstance(instDesc)

def to_c_string(_str):
  return ctypes.create_string_buffer(_str.encode('utf-8'))

def from_wgpu_str(string_view):
  return ctypes.string_at(string_view.data, string_view.length).decode("utf-8")

def to_wgpu_str(_str):
  return webgpu.WGPUStringView(data=ctypes.cast(ctypes.pointer(to_c_string(_str)), ctypes.POINTER(ctypes.c_char)), length=len(_str))

def wgpu_wait(future):
  assert webgpu.wgpuInstanceWaitAny(instance, 1, webgpu.WGPUFutureWaitInfo(future=future), 2**64-1) == webgpu.WGPUWaitStatus_Success, "Future failed"

def create_cb_info(cb_info_type, cb_type, cb):
  return cb_info_type(nextInChain=None, mode=webgpu.WGPUCallbackMode_WaitAnyOnly, callback=cb_type(cb))

def write_buffer(device, buf, offset, src):
    src = bytearray(src)
    src_pointer = (ctypes.c_uint8 * len(src)).from_buffer(src)
    webgpu.wgpuQueueWriteBuffer(webgpu.wgpuDeviceGetQueue(device), buf, offset, src_pointer, len(src))

def map_buffer(buf, size):
  result: List[Any] = []

  def cb(status, msg, u1, u2):
    result[:] = status, from_wgpu_str(msg)

  cb_info = create_cb_info(webgpu.WGPUBufferMapCallbackInfo2, webgpu.WGPUBufferMapCallback2, cb)
  wgpu_wait(webgpu.wgpuBufferMapAsync2(buf, webgpu.WGPUMapMode_Read, 0, size, cb_info))

  if result[0] != webgpu.WGPUBufferMapAsyncStatus_Success:
    raise RuntimeError(f"Failed to map buffer: [{webgpu.WGPUBufferMapAsyncStatus__enumvalues[result[0]]}] {result[1]}")

def copy_buffer_to_buffer(device, src, src_offset, dst, dst_offset, size):
  encoder = webgpu.wgpuDeviceCreateCommandEncoder(device, webgpu.WGPUCommandEncoderDescriptor())
  webgpu.wgpuCommandEncoderCopyBufferToBuffer(encoder, src, src_offset, dst, dst_offset, size)
  cb = webgpu.wgpuCommandEncoderFinish(encoder, webgpu.WGPUCommandBufferDescriptor())
  submit(device, [cb])
  webgpu.wgpuCommandBufferRelease(cb)
  webgpu.wgpuCommandEncoderRelease(encoder)

def read_buffer(dev, buf):
  size = webgpu.wgpuBufferGetSize(buf)
  tmp_usage = webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_MapRead
  tmp_buffer = webgpu.wgpuDeviceCreateBuffer(dev, webgpu.WGPUBufferDescriptor(size=size, usage=tmp_usage, mappedAtCreation=False))
  copy_buffer_to_buffer(dev, buf, 0, tmp_buffer, 0, size)
  map_buffer(tmp_buffer, size)
  ptr = webgpu.wgpuBufferGetConstMappedRange(tmp_buffer, 0, size)
  void_ptr = ctypes.cast(ptr, ctypes.c_void_p)
  byte_array = (ctypes.c_uint8 * size).from_address(void_ptr.value)
  result = bytearray(byte_array)
  webgpu.wgpuBufferUnmap(tmp_buffer)
  webgpu.wgpuBufferDestroy(tmp_buffer)
  return memoryview(result).cast("B")

def create_shader_module(device, source):
  shader = webgpu.WGPUShaderModuleWGSLDescriptor()
  shader.code = to_wgpu_str(source)
  shader.chain.next = None
  shader.chain.sType = webgpu.WGPUSType_ShaderSourceWGSL
  module = webgpu.WGPUShaderModuleDescriptor()
  module.nextInChain = ctypes.cast(ctypes.pointer(shader), ctypes.POINTER(webgpu.struct_WGPUChainedStruct))

  # Check compiler error
  webgpu.wgpuDevicePushErrorScope(device, webgpu.WGPUErrorFilter_Validation)
  shader_module = webgpu.wgpuDeviceCreateShaderModule(device, module)
  compiler_error = pop_error(device)

  if compiler_error: raise RuntimeError(f"Shader compilation failed: {compiler_error}")

  return shader_module

def submit(device, command_buffers):
  cb_buffers_array_type = webgpu.WGPUCommandBuffer * len(command_buffers)
  cb_buffers_array = cb_buffers_array_type(*command_buffers)
  webgpu.wgpuQueueSubmit(webgpu.wgpuDeviceGetQueue(device), len(command_buffers), cb_buffers_array)

def pop_error(device):
  result: List[Any] = []

  def cb(status, err_type, msg, i2):
    result[:] = [from_wgpu_str(msg)]

  cb_info = create_cb_info(webgpu.WGPUPopErrorScopeCallbackInfo, webgpu.WGPUPopErrorScopeCallback, cb)
  wgpu_wait(webgpu.wgpuDevicePopErrorScopeF(device, cb_info))
  return result[0] if len(result) > 0 else ""

def create_uniform(wgpu_device, val):
  buf = webgpu.wgpuDeviceCreateBuffer(wgpu_device,
    webgpu.WGPUBufferDescriptor(size=4, usage=webgpu.WGPUBufferUsage_Uniform | webgpu.WGPUBufferUsage_CopyDst))
  write_buffer(wgpu_device, buf, 0, val.to_bytes(4, "little") if isinstance(val, int) else struct.pack('<f', val))
  return buf

class WebGPUProgram:
  def __init__(self, dev, name:str, lib:bytes):
    (self.dev, self.timestamp_supported) = dev
    self.name, self.lib, self.prg = name, lib, create_shader_module(self.dev, lib.decode())   # NOTE: this is the compiler
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    wait = wait and self.timestamp_supported
    tmp_bufs = [*bufs]
    buf_patch = False

    # WebGPU does not allow using the same buffer for input and output
    for i in range(1, len(bufs)):
      if bufs[i] == bufs[0]:
        tmp_bufs[0] = webgpu.wgpuDeviceCreateBuffer(self.dev,
          webgpu.WGPUBufferDescriptor(size=webgpu.wgpuBufferGetSize(bufs[0]), usage=webgpu.wgpuBufferGetUsage(bufs[0])))
        buf_patch = True

    binding_layouts = [webgpu.WGPUBindGroupLayoutEntry(binding=0, visibility= webgpu.WGPUShaderStage_Compute,
      buffer=webgpu.WGPUBufferBindingLayout(type=webgpu.WGPUBufferBindingType_Uniform))]
    binding_layouts += [webgpu.WGPUBindGroupLayoutEntry(binding=i+1, visibility=webgpu.WGPUShaderStage_Compute,
      buffer=webgpu.WGPUBufferBindingLayout(type=webgpu.WGPUBufferBindingType_Uniform if i >= len(tmp_bufs)
      else webgpu.WGPUBufferBindingType_Storage)) for i in range(len(tmp_bufs)+len(vals))]
    bindings = [webgpu.WGPUBindGroupEntry(binding=0, buffer=create_uniform(self.dev, float('inf')), offset=0, size=4)]
    bindings += [webgpu.WGPUBindGroupEntry(binding=i+1, buffer=create_uniform(self.dev, x) if i >= len(tmp_bufs) else x, offset=0,
      size=4 if i >= len(tmp_bufs) else webgpu.wgpuBufferGetSize(x)) for i,x in enumerate(tuple(tmp_bufs)+vals)]

    # Creating bind group layout
    bl_arr_type = webgpu.WGPUBindGroupLayoutEntry * len(binding_layouts)
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    bind_group_layouts = [webgpu.wgpuDeviceCreateBindGroupLayout(self.dev, webgpu.WGPUBindGroupLayoutDescriptor(
      entryCount=len(binding_layouts), entries=ctypes.cast(bl_arr_type(*binding_layouts), ctypes.POINTER(webgpu.WGPUBindGroupLayoutEntry))))]
    bind_group_layout_error = pop_error(self.dev)

    if bind_group_layout_error: raise RuntimeError(f"Error creating bind group layout: {bind_group_layout_error}")

    # Creating pipeline layout
    pipeline_layout_desc = webgpu.WGPUPipelineLayoutDescriptor()
    pipeline_layout_desc.bindGroupLayoutCount = len(bind_group_layouts)
    bind_group_array_type = webgpu.WGPUBindGroupLayout * len(bind_group_layouts)
    pipeline_layout_desc.bindGroupLayouts = bind_group_array_type(*bind_group_layouts)

    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    pipeline_layout = webgpu.wgpuDeviceCreatePipelineLayout(self.dev, pipeline_layout_desc)
    pipeline_layout_error = pop_error(self.dev)

    if pipeline_layout_error: raise RuntimeError(f"Error creating pipeline layout: {pipeline_layout_error}")

    # Creating bind group
    bg_arr_type = webgpu.WGPUBindGroupEntry * len(bindings)
    bind_group_desc = webgpu.WGPUBindGroupDescriptor(layout=bind_group_layouts[0], entryCount=len(bindings), entries=bg_arr_type(*bindings))
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    bind_group = webgpu.wgpuDeviceCreateBindGroup(self.dev, bind_group_desc)
    bind_group_error = pop_error(self.dev)

    if bind_group_error: raise RuntimeError(f"Error creating bind group: {bind_group_error}")

    # Creating compute pipeline
    compute_desc = webgpu.WGPUComputePipelineDescriptor(layout=pipeline_layout,
      compute=webgpu.WGPUComputeState(module=self.prg, entryPoint=to_wgpu_str(self.name)))
    pipeline_result: List[Any] = []

    def cb(status, compute_pipeline_impl, msg, u1, u2):
      pipeline_result[:] = status, compute_pipeline_impl, from_wgpu_str(msg)

    cb_info = create_cb_info(webgpu.WGPUCreateComputePipelineAsyncCallbackInfo2,  webgpu.WGPUCreateComputePipelineAsyncCallback2, cb)
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    wgpu_wait(webgpu.wgpuDeviceCreateComputePipelineAsync2(self.dev, compute_desc, cb_info))
    pipeline_error = pop_error(self.dev)

    if pipeline_result[0] != webgpu.WGPUCreatePipelineAsyncStatus_Success:
      raise RuntimeError(f"{webgpu.WGPUCreatePipelineAsyncStatus__enumvalues[pipeline_result[0]]}: {pipeline_result[2]}, {pipeline_error}")

    command_encoder = webgpu.wgpuDeviceCreateCommandEncoder(self.dev, webgpu.WGPUCommandEncoderDescriptor())
    comp_pass_desc = webgpu.WGPUComputePassDescriptor(nextInChain=None)

    if wait:
      query_set = webgpu.wgpuDeviceCreateQuerySet(self.dev, webgpu.WGPUQuerySetDescriptor(type=webgpu.WGPUQueryType_Timestamp, count=2))
      query_buf = webgpu.wgpuDeviceCreateBuffer(self.dev,
        webgpu.WGPUBufferDescriptor(size=16, usage=webgpu.WGPUBufferUsage_QueryResolve | webgpu.WGPUBufferUsage_CopySrc))
      comp_pass_desc.timestampWrites = ctypes.pointer(webgpu.WGPUComputePassTimestampWrites(
        querySet=query_set, beginningOfPassWriteIndex=0, endOfPassWriteIndex=1))

    # Begin compute pass
    compute_pass = webgpu.wgpuCommandEncoderBeginComputePass(command_encoder, comp_pass_desc)
    webgpu.wgpuComputePassEncoderSetPipeline(compute_pass, pipeline_result[1])
    webgpu.wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, None)
    webgpu.wgpuComputePassEncoderDispatchWorkgroups(compute_pass, *global_size)
    webgpu.wgpuComputePassEncoderEnd(compute_pass)

    if wait: webgpu.wgpuCommandEncoderResolveQuerySet(command_encoder, query_set, 0, 2, query_buf, 0)

    cmd_buf = webgpu.wgpuCommandEncoderFinish(command_encoder, webgpu.WGPUCommandBufferDescriptor())
    submit(self.dev, [cmd_buf])

    if buf_patch:
      copy_buffer_to_buffer(self.dev, tmp_bufs[0], 0, bufs[0], 0, webgpu.wgpuBufferGetSize(bufs[0]))
      webgpu.wgpuBufferDestroy(tmp_bufs[0])

    if wait:
      time = ((timestamps:=read_buffer(self.dev, query_buf).cast("Q").tolist())[1] - timestamps[0]) / 1e9
      webgpu.wgpuBufferDestroy(query_buf)
      webgpu.wgpuQuerySetDestroy(query_set)
      return time

class WebGpuAllocator(Allocator):
  def __init__(self, dev): self.dev = dev
  def _alloc(self, size: int, options):
    # WebGPU buffers have to be 4-byte aligned
    return webgpu.wgpuDeviceCreateBuffer(
      self.dev, webgpu.WGPUBufferDescriptor(size=round_up(size, 4),
      usage=webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_CopySrc))
  def _copyin(self, dest, src: memoryview):
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    write_buffer(self.dev, dest, 0, padded_src if src.nbytes % 4 else src)
  def _copyout(self, dest: memoryview, src):
    buffer_data = read_buffer(self.dev, src)
    src_len = webgpu.wgpuBufferGetSize(src)
    dest[:] = buffer_data[:dest.nbytes] if src_len  > dest.nbytes else buffer_data
  def _free(self, opaque, options):
    webgpu.wgpuBufferDestroy(opaque)

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    # Requesting an adapter
    adapter_result: List[Any] = []

    def adapter_cb(status, adapter, msg, _): adapter_result[:] = status, adapter, from_wgpu_str(msg)

    cb_info = create_cb_info(webgpu.WGPURequestAdapterCallbackInfo, webgpu.WGPURequestAdapterCallback, adapter_cb)
    wgpu_wait(webgpu.wgpuInstanceRequestAdapterF(instance,
      webgpu.WGPURequestAdapterOptions(powerPreference=webgpu.WGPUPowerPreference_HighPerformance), cb_info))

    if adapter_result[0] != webgpu.WGPURequestAdapterStatus_Success:
      raise RuntimeError(f"Error requesting adapter: [{webgpu.WGPURequestAdapterStatus__enumvalues[adapter_result[0]]}] {adapter_result[2]}")

    # Get supported features
    supported_features = webgpu.WGPUSupportedFeatures()
    webgpu.wgpuAdapterGetFeatures(adapter_result[1], supported_features)
    timestamp_supported = "WGPUFeatureName_TimestampQuery" in [webgpu.WGPUFeatureName__enumvalues[supported_features.features[i]]
      for i in range(supported_features.featureCount)]

    required_features = [webgpu.WGPUFeatureName_ShaderF16]
    if timestamp_supported: required_features.append(webgpu.WGPUFeatureName_TimestampQuery)

    assert adapter_result[1] is not None, "adapter should not be none"
    device_desc = webgpu.WGPUDeviceDescriptor()

    # Disable "timestamp_quantization" for nanosecond precision: https://developer.chrome.com/blog/new-in-webgpu-120
    toggle_desc = webgpu.WGPUDawnTogglesDescriptor()
    toggle_desc.chain.sType = webgpu.WGPUSType_DawnTogglesDescriptor
    toggle_desc.enabledToggleCount = 1
    unsafe_apis = ctypes.cast(ctypes.pointer(to_c_string("allow_unsafe_apis")), ctypes.POINTER(ctypes.c_char))
    toggle_desc.enabledToggles =  ctypes.pointer(unsafe_apis)
    toggle_desc.disabledToggleCount = 1
    ts_quant = ctypes.cast(ctypes.pointer(to_c_string("timestamp_quantization")), ctypes.POINTER(ctypes.c_char))
    toggle_desc.disabledToggles = ctypes.pointer(ts_quant)
    device_desc.nextInChain = ctypes.cast(ctypes.pointer(toggle_desc), ctypes.POINTER(webgpu.struct_WGPUChainedStruct))

    # Populate required features
    feature_array_type = webgpu.WGPUFeatureName * len(required_features)
    feature_array = feature_array_type(*required_features)
    device_desc.requiredFeatureCount = len(required_features)
    device_desc.requiredFeatures = ctypes.cast(feature_array, ctypes.POINTER(webgpu.WGPUFeatureName))

    # Limits
    supported_limits = webgpu.WGPUSupportedLimits()
    webgpu.wgpuAdapterGetLimits(adapter_result[1], ctypes.cast(ctypes.pointer(supported_limits),ctypes.POINTER(webgpu.struct_WGPUSupportedLimits)))
    limits = webgpu.WGPURequiredLimits(limits=supported_limits.limits)
    device_desc.requiredLimits = ctypes.cast(ctypes.pointer(limits),ctypes.POINTER(webgpu.struct_WGPURequiredLimits))

    # Requesting a device
    device_result: List[Any] = []

    def dev_cb(status, device_impl, msg, _): device_result[:] = status, device_impl, from_wgpu_str(msg)

    cb_info = create_cb_info(webgpu.WGPURequestDeviceCallbackInfo, webgpu.WGPURequestDeviceCallback, dev_cb)
    wgpu_wait(webgpu.wgpuAdapterRequestDeviceF(adapter_result[1], device_desc, cb_info))

    if device_result[0] != webgpu.WGPURequestDeviceStatus_Success:
      raise RuntimeError(f"Failed to request device: [{webgpu.WGPURequestDeviceStatus__enumvalues[device_result[0]]}] {device_result[2]}")

    super().__init__(device, WebGpuAllocator(device_result[1]), WGSLRenderer(), Compiler(),
                     functools.partial(WebGPUProgram, (device_result[1], timestamp_supported)))
