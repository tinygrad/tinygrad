import functools, struct
from tinygrad.device import Compiled, Allocator, BufferSpec
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.helpers import round_up, suppress_finalizing, getenv
from tinygrad.runtime.autogen import webgpu
from tinygrad.runtime.support import c
from typing import cast, TypeAlias, Callable
import ctypes

WGPUDevPtr: TypeAlias = webgpu.WGPUDevice
WGPUBufPtr: TypeAlias = webgpu.WGPUBuffer

backend_types = {v: k for k, v in webgpu.enum_WGPUBackendType.items() }

instance = webgpu.wgpuCreateInstance(webgpu.WGPUInstanceDescriptor(features = webgpu.WGPUInstanceFeatures(timedWaitAnyEnable = True)))

def to_c_string(_str:str) -> ctypes.Array: return ctypes.create_string_buffer(_str.encode('utf-8'))

def from_wgpu_str(string_view:webgpu.struct_WGPUStringView) -> str: return ctypes.string_at(string_view.data, string_view.length).decode("utf-8")

def to_wgpu_str(_str:str) -> webgpu.struct_WGPUStringView:
  return webgpu.WGPUStringView(data=ctypes.cast(ctypes.pointer(to_c_string(_str)), ctypes.POINTER(ctypes.c_char)), length=len(_str))

def write_buffer(device:WGPUDevPtr, buf:WGPUBufPtr, offset:int, src:memoryview|bytearray|bytes):
  src = bytearray(src)
  webgpu.wgpuQueueWriteBuffer(webgpu.wgpuDeviceGetQueue(device), buf, offset, (ctypes.c_uint8 * len(src)).from_buffer(src), len(src))

# turns a webgpu function returning a future into python-synchronous function
# the new function handles the status code and optional error message, returning the other callback arguments
def synchronous(status_enum:dict[int, str], has_emsg:bool=False):
  def wrap(fn:Callable[..., webgpu.WGPUFuture]) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args):
      status, payload, emsg = 0, [], None

      @fn.argtypes[-1].callback.type # type: ignore
      def cb(*args):
        nonlocal status, payload, emsg
        status, *payload, emsg = args[:-2] if has_emsg else (*args[:-2], None) # the last two arguments are "userdata1" and "userdata2", which we drop

      future = fn(*args, fn.argtypes[-1](mode=webgpu.WGPUCallbackMode_WaitAnyOnly, callback=cb)) # type: ignore
      if (future_status:=webgpu.wgpuInstanceWaitAny(instance, 1, webgpu.WGPUFutureWaitInfo(future), 2**64-1)) != webgpu.WGPUWaitStatus_Success:
        raise RuntimeError(f"error while waiting for future ({fn.__name__}): {webgpu.enum_WGPUWaitStatus.get(future_status)}")

      if status != 1: raise RuntimeError(f"[{status_enum.get(status)}]{from_wgpu_str(emsg) if emsg else ''}")
      return payload if len(payload) > 1 else payload[0] if len(payload) == 1 else None
    return wrapper
  return wrap

BufferMapAsync = synchronous(webgpu.enum_WGPUBufferMapAsyncStatus, True)(webgpu.wgpuBufferMapAsync2)
DevicePopErrorScope = synchronous(webgpu.enum_WGPUPopErrorScopeStatus)(webgpu.wgpuDevicePopErrorScope2)
DeviceCreateComputePipeline = synchronous(webgpu.enum_WGPUCreatePipelineAsyncStatus, True)(webgpu.wgpuDeviceCreateComputePipelineAsync2)
InstanceRequestAdapter = synchronous(webgpu.enum_WGPURequestAdapterStatus, True)(webgpu.wgpuInstanceRequestAdapter2)
AdapterRequestDevice = synchronous(webgpu.enum_WGPURequestDeviceStatus, True)(webgpu.wgpuAdapterRequestDevice2)
QueueOnSubmittedWorkDone = synchronous(webgpu.enum_WGPUQueueWorkDoneStatus)(webgpu.wgpuQueueOnSubmittedWorkDone2)

def copy_buffer_to_buffer(dev:WGPUDevPtr, src:WGPUBufPtr, src_offset:int, dst:WGPUBufPtr, dst_offset:int, size:int):
  encoder = webgpu.wgpuDeviceCreateCommandEncoder(dev, webgpu.WGPUCommandEncoderDescriptor())
  webgpu.wgpuCommandEncoderCopyBufferToBuffer(encoder, src, src_offset, dst, dst_offset, size)
  cb = webgpu.wgpuCommandEncoderFinish(encoder, webgpu.WGPUCommandBufferDescriptor())
  webgpu.wgpuQueueSubmit(webgpu.wgpuDeviceGetQueue(dev), 1, (webgpu.WGPUCommandBuffer*1)(cb))
  webgpu.wgpuCommandBufferRelease(cb)
  webgpu.wgpuCommandEncoderRelease(encoder)

def read_buffer(dev:WGPUDevPtr, buf:WGPUBufPtr) -> memoryview:
  size = webgpu.wgpuBufferGetSize(buf)
  tmp_buffer = webgpu.wgpuDeviceCreateBuffer(dev, webgpu.WGPUBufferDescriptor(size=size,
    usage=webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_MapRead, mappedAtCreation=False))
  copy_buffer_to_buffer(dev, buf, 0, tmp_buffer, 0, size)
  BufferMapAsync(tmp_buffer, webgpu.WGPUMapMode_Read, 0, size)
  void_ptr = ctypes.cast(webgpu.wgpuBufferGetConstMappedRange(tmp_buffer, 0, size), ctypes.c_void_p)
  buf_copy = bytearray((ctypes.c_uint8 * size).from_address(void_ptr.value))
  webgpu.wgpuBufferUnmap(tmp_buffer)
  webgpu.wgpuBufferDestroy(tmp_buffer)
  return memoryview(buf_copy).cast("B")

def pop_error(device:WGPUDevPtr) -> str: return from_wgpu_str(DevicePopErrorScope(device)[1])

def create_uniform(wgpu_device:WGPUDevPtr, val:int|float) -> WGPUBufPtr:
  buf = webgpu.wgpuDeviceCreateBuffer(wgpu_device,
    webgpu.WGPUBufferDescriptor(size=4, usage=webgpu.WGPUBufferUsage_Uniform | webgpu.WGPUBufferUsage_CopyDst))
  write_buffer(wgpu_device, buf, 0, val.to_bytes(4, "little") if isinstance(val, int) else struct.pack('<f', val))
  return buf

class WebGPUProgram:
  def __init__(self, dev:tuple[WGPUDevPtr, bool], name:str, lib:bytes, **kwargs):
    (self.dev, self.timestamp_supported) = dev

    # Creating shader module
    shader = webgpu.WGPUShaderModuleWGSLDescriptor(code=to_wgpu_str(lib.decode()),
      chain=webgpu.WGPUChainedStruct(sType=webgpu.WGPUSType_ShaderSourceWGSL))
    module = webgpu.WGPUShaderModuleDescriptor()
    module.nextInChain = ctypes.cast(ctypes.pointer(shader), c.POINTER[webgpu.struct_WGPUChainedStruct])

    # Check compiler error
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    shader_module = webgpu.wgpuDeviceCreateShaderModule(self.dev, module)

    if err := pop_error(self.dev): raise RuntimeError(f"Shader compilation failed: {err}")

    self.name, self.lib, self.prg = name, lib, shader_module
  def __call__(self, *bufs:WGPUBufPtr, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int, ...]=(), wait=False, **kw) -> float|None:
    wait = wait and self.timestamp_supported
    tmp_bufs = [*bufs]
    buf_patch = False

    # WebGPU does not allow using the same buffer for input and output
    for i in range(1, len(bufs)):
      if ctypes.addressof(bufs[i]) == ctypes.addressof(bufs[0]):
        tmp_bufs[0] = webgpu.wgpuDeviceCreateBuffer(self.dev,
          webgpu.WGPUBufferDescriptor(size=webgpu.wgpuBufferGetSize(bufs[0]), usage=webgpu.wgpuBufferGetUsage(bufs[0])))
        buf_patch = True

    # Creating bind group layout
    binding_layouts = [webgpu.WGPUBindGroupLayoutEntry(binding=0, visibility= webgpu.WGPUShaderStage_Compute,
      buffer=webgpu.WGPUBufferBindingLayout(type=webgpu.WGPUBufferBindingType_Uniform))]
    binding_layouts += [webgpu.WGPUBindGroupLayoutEntry(binding=i+1, visibility=webgpu.WGPUShaderStage_Compute,
      buffer=webgpu.WGPUBufferBindingLayout(type=webgpu.WGPUBufferBindingType_Uniform if i >= len(tmp_bufs)
      else webgpu.WGPUBufferBindingType_Storage)) for i in range(len(tmp_bufs)+len(vals))]

    bl_arr_type = webgpu.WGPUBindGroupLayoutEntry * len(binding_layouts)
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    bind_group_layouts = [webgpu.wgpuDeviceCreateBindGroupLayout(self.dev, webgpu.WGPUBindGroupLayoutDescriptor(
      entryCount=len(binding_layouts), entries=ctypes.cast(bl_arr_type(*binding_layouts), ctypes.POINTER(webgpu.WGPUBindGroupLayoutEntry))))]

    if bg_layout_err := pop_error(self.dev): raise RuntimeError(f"Error creating bind group layout: {bg_layout_err}")

    # Creating pipeline layout
    pipeline_layout_desc = webgpu.WGPUPipelineLayoutDescriptor(bindGroupLayoutCount=len(bind_group_layouts),
      bindGroupLayouts = (webgpu.WGPUBindGroupLayout * len(bind_group_layouts))(*bind_group_layouts))

    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    pipeline_layout = webgpu.wgpuDeviceCreatePipelineLayout(self.dev, pipeline_layout_desc)

    if pipe_err := pop_error(self.dev): raise RuntimeError(f"Error creating pipeline layout: {pipe_err}")

    # Creating bind group
    bindings = [webgpu.WGPUBindGroupEntry(binding=0, buffer=create_uniform(self.dev, float('inf')), offset=0, size=4)]
    bindings += [webgpu.WGPUBindGroupEntry(binding=i+1, buffer=create_uniform(self.dev, cast(int, x)) if i >= len(tmp_bufs) else x, offset=0,
      size=4 if i >= len(tmp_bufs) else webgpu.wgpuBufferGetSize(x)) for i,x in enumerate(tuple(tmp_bufs)+vals)]

    bg_arr_type = webgpu.WGPUBindGroupEntry * len(bindings)
    bind_group_desc = webgpu.WGPUBindGroupDescriptor(layout=bind_group_layouts[0], entryCount=len(bindings), entries=bg_arr_type(*bindings))
    webgpu.wgpuDevicePushErrorScope(self.dev, webgpu.WGPUErrorFilter_Validation)
    bind_group = webgpu.wgpuDeviceCreateBindGroup(self.dev, bind_group_desc)

    if bind_err := pop_error(self.dev): raise RuntimeError(f"Error creating bind group: {bind_err}")

    # Creating compute pipeline
    compute_desc = webgpu.WGPUComputePipelineDescriptor(layout=pipeline_layout,
      compute=webgpu.WGPUComputeState(module=self.prg, entryPoint=to_wgpu_str(self.name)))
    pipeline_result = DeviceCreateComputePipeline(self.dev, compute_desc)

    command_encoder = webgpu.wgpuDeviceCreateCommandEncoder(self.dev, webgpu.WGPUCommandEncoderDescriptor())
    comp_pass_desc = webgpu.WGPUComputePassDescriptor()

    if wait:
      query_set = webgpu.wgpuDeviceCreateQuerySet(self.dev, webgpu.WGPUQuerySetDescriptor(type=webgpu.WGPUQueryType_Timestamp, count=2))
      query_buf = webgpu.wgpuDeviceCreateBuffer(self.dev,
        webgpu.WGPUBufferDescriptor(size=16, usage=webgpu.WGPUBufferUsage_QueryResolve | webgpu.WGPUBufferUsage_CopySrc))
      comp_pass_desc.timestampWrites = c.pointer(webgpu.WGPUComputePassTimestampWrites(
        querySet=query_set, beginningOfPassWriteIndex=0, endOfPassWriteIndex=1))

    # Begin compute pass
    compute_pass = webgpu.wgpuCommandEncoderBeginComputePass(command_encoder, comp_pass_desc)
    webgpu.wgpuComputePassEncoderSetPipeline(compute_pass, pipeline_result)
    webgpu.wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, None)
    webgpu.wgpuComputePassEncoderDispatchWorkgroups(compute_pass, *global_size)
    webgpu.wgpuComputePassEncoderEnd(compute_pass)

    if wait: webgpu.wgpuCommandEncoderResolveQuerySet(command_encoder, query_set, 0, 2, query_buf, 0)

    cmd_buf = webgpu.wgpuCommandEncoderFinish(command_encoder, webgpu.WGPUCommandBufferDescriptor())
    webgpu.wgpuQueueSubmit(webgpu.wgpuDeviceGetQueue(self.dev), 1, (webgpu.WGPUCommandBuffer*1)(cmd_buf))

    if buf_patch:
      copy_buffer_to_buffer(self.dev, tmp_bufs[0], 0, bufs[0], 0, webgpu.wgpuBufferGetSize(bufs[0]))
      webgpu.wgpuBufferDestroy(tmp_bufs[0])

    if wait:
      time = ((timestamps:=read_buffer(self.dev, query_buf).cast("Q").tolist())[1] - timestamps[0]) / 1e9
      webgpu.wgpuBufferDestroy(query_buf)
      webgpu.wgpuQuerySetDestroy(query_set)
      return time
    return None

class WebGpuAllocator(Allocator['WebGpuDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> WGPUBufPtr:
    # WebGPU buffers have to be 4-byte aligned
    return webgpu.wgpuDeviceCreateBuffer(self.dev.device_res, webgpu.WGPUBufferDescriptor(size=round_up(size, 4),
      usage=webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst | webgpu.WGPUBufferUsage_CopySrc))
  def _copyin(self, dest:WGPUBufPtr, src:memoryview):
    if src.nbytes % 4:
      padded_src = bytearray(round_up(src.nbytes, 4))
      padded_src[:src.nbytes] = src
    write_buffer(self.dev.device_res, dest, 0, padded_src if src.nbytes % 4 else src)
  def _copyout(self, dest:memoryview, src:WGPUBufPtr):
    buffer_data = read_buffer(self.dev.device_res, src)
    dest[:] = buffer_data[:dest.nbytes] if webgpu.wgpuBufferGetSize(src)  > dest.nbytes else buffer_data
  @suppress_finalizing
  def _free(self, opaque:WGPUBufPtr, options:BufferSpec): webgpu.wgpuBufferDestroy(opaque)

class WebGpuDevice(Compiled):
  def __init__(self, device:str):
    # Requesting an adapter
    adapter_res = InstanceRequestAdapter(instance, webgpu.WGPURequestAdapterOptions(
      powerPreference=webgpu.WGPUPowerPreference_HighPerformance, backendType=backend_types.get(getenv("WEBGPU_BACKEND", ""), 0)))

    # Get supported features
    supported_features = webgpu.WGPUSupportedFeatures()
    webgpu.wgpuAdapterGetFeatures(adapter_res, supported_features)
    supported = [supported_features.features[i] for i in range(supported_features.featureCount)]
    features = [feat for feat in [webgpu.WGPUFeatureName_TimestampQuery, webgpu.WGPUFeatureName_ShaderF16] if feat in supported]
    dev_desc = webgpu.WGPUDeviceDescriptor(requiredFeatureCount=len(features),
                                           requiredFeatures=c.Array(webgpu.WGPUFeatureName, len(features))(*features)) # type: ignore

    # Limits
    supported_limits = webgpu.WGPUSupportedLimits()
    webgpu.wgpuAdapterGetLimits(adapter_res, ctypes.cast(ctypes.pointer(supported_limits),ctypes.POINTER(webgpu.struct_WGPUSupportedLimits)))
    limits = webgpu.WGPURequiredLimits(limits=supported_limits.limits)
    dev_desc.requiredLimits = c.pointer(limits)

    # Requesting a device
    self.device_res = AdapterRequestDevice(adapter_res, dev_desc)

    program = functools.partial(WebGPUProgram, (self.device_res, webgpu.WGPUFeatureName_TimestampQuery in supported))
    super().__init__(device, WebGpuAllocator(self), [WGSLRenderer], program, arch="shader-f16" * (webgpu.WGPUFeatureName_ShaderF16 in supported))

  def synchronize(self): QueueOnSubmittedWorkDone(webgpu.wgpuDeviceGetQueue(self.device_res))
