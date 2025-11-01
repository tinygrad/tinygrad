# mypy: ignore-errors
import ctypes
from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support.webgpu import WEBGPU_PATH
def dll():
  try: return ctypes.CDLL(WEBGPU_PATH)
  except: pass
  return None
dll = dll()
WGPUFlags = ctypes.c_ulong
WGPUBool = ctypes.c_uint
class struct_WGPUAdapterImpl(ctypes.Structure): pass
struct_WGPUAdapterImpl._fields_ = []
WGPUAdapter = ctypes.POINTER(struct_WGPUAdapterImpl)
class struct_WGPUBindGroupImpl(ctypes.Structure): pass
struct_WGPUBindGroupImpl._fields_ = []
WGPUBindGroup = ctypes.POINTER(struct_WGPUBindGroupImpl)
class struct_WGPUBindGroupLayoutImpl(ctypes.Structure): pass
struct_WGPUBindGroupLayoutImpl._fields_ = []
WGPUBindGroupLayout = ctypes.POINTER(struct_WGPUBindGroupLayoutImpl)
class struct_WGPUBufferImpl(ctypes.Structure): pass
struct_WGPUBufferImpl._fields_ = []
WGPUBuffer = ctypes.POINTER(struct_WGPUBufferImpl)
class struct_WGPUCommandBufferImpl(ctypes.Structure): pass
struct_WGPUCommandBufferImpl._fields_ = []
WGPUCommandBuffer = ctypes.POINTER(struct_WGPUCommandBufferImpl)
class struct_WGPUCommandEncoderImpl(ctypes.Structure): pass
struct_WGPUCommandEncoderImpl._fields_ = []
WGPUCommandEncoder = ctypes.POINTER(struct_WGPUCommandEncoderImpl)
class struct_WGPUComputePassEncoderImpl(ctypes.Structure): pass
struct_WGPUComputePassEncoderImpl._fields_ = []
WGPUComputePassEncoder = ctypes.POINTER(struct_WGPUComputePassEncoderImpl)
class struct_WGPUComputePipelineImpl(ctypes.Structure): pass
struct_WGPUComputePipelineImpl._fields_ = []
WGPUComputePipeline = ctypes.POINTER(struct_WGPUComputePipelineImpl)
class struct_WGPUDeviceImpl(ctypes.Structure): pass
struct_WGPUDeviceImpl._fields_ = []
WGPUDevice = ctypes.POINTER(struct_WGPUDeviceImpl)
class struct_WGPUExternalTextureImpl(ctypes.Structure): pass
struct_WGPUExternalTextureImpl._fields_ = []
WGPUExternalTexture = ctypes.POINTER(struct_WGPUExternalTextureImpl)
class struct_WGPUInstanceImpl(ctypes.Structure): pass
struct_WGPUInstanceImpl._fields_ = []
WGPUInstance = ctypes.POINTER(struct_WGPUInstanceImpl)
class struct_WGPUPipelineLayoutImpl(ctypes.Structure): pass
struct_WGPUPipelineLayoutImpl._fields_ = []
WGPUPipelineLayout = ctypes.POINTER(struct_WGPUPipelineLayoutImpl)
class struct_WGPUQuerySetImpl(ctypes.Structure): pass
struct_WGPUQuerySetImpl._fields_ = []
WGPUQuerySet = ctypes.POINTER(struct_WGPUQuerySetImpl)
class struct_WGPUQueueImpl(ctypes.Structure): pass
struct_WGPUQueueImpl._fields_ = []
WGPUQueue = ctypes.POINTER(struct_WGPUQueueImpl)
class struct_WGPURenderBundleImpl(ctypes.Structure): pass
struct_WGPURenderBundleImpl._fields_ = []
WGPURenderBundle = ctypes.POINTER(struct_WGPURenderBundleImpl)
class struct_WGPURenderBundleEncoderImpl(ctypes.Structure): pass
struct_WGPURenderBundleEncoderImpl._fields_ = []
WGPURenderBundleEncoder = ctypes.POINTER(struct_WGPURenderBundleEncoderImpl)
class struct_WGPURenderPassEncoderImpl(ctypes.Structure): pass
struct_WGPURenderPassEncoderImpl._fields_ = []
WGPURenderPassEncoder = ctypes.POINTER(struct_WGPURenderPassEncoderImpl)
class struct_WGPURenderPipelineImpl(ctypes.Structure): pass
struct_WGPURenderPipelineImpl._fields_ = []
WGPURenderPipeline = ctypes.POINTER(struct_WGPURenderPipelineImpl)
class struct_WGPUSamplerImpl(ctypes.Structure): pass
struct_WGPUSamplerImpl._fields_ = []
WGPUSampler = ctypes.POINTER(struct_WGPUSamplerImpl)
class struct_WGPUShaderModuleImpl(ctypes.Structure): pass
struct_WGPUShaderModuleImpl._fields_ = []
WGPUShaderModule = ctypes.POINTER(struct_WGPUShaderModuleImpl)
class struct_WGPUSharedBufferMemoryImpl(ctypes.Structure): pass
struct_WGPUSharedBufferMemoryImpl._fields_ = []
WGPUSharedBufferMemory = ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl)
class struct_WGPUSharedFenceImpl(ctypes.Structure): pass
struct_WGPUSharedFenceImpl._fields_ = []
WGPUSharedFence = ctypes.POINTER(struct_WGPUSharedFenceImpl)
class struct_WGPUSharedTextureMemoryImpl(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryImpl._fields_ = []
WGPUSharedTextureMemory = ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl)
class struct_WGPUSurfaceImpl(ctypes.Structure): pass
struct_WGPUSurfaceImpl._fields_ = []
WGPUSurface = ctypes.POINTER(struct_WGPUSurfaceImpl)
class struct_WGPUTextureImpl(ctypes.Structure): pass
struct_WGPUTextureImpl._fields_ = []
WGPUTexture = ctypes.POINTER(struct_WGPUTextureImpl)
class struct_WGPUTextureViewImpl(ctypes.Structure): pass
struct_WGPUTextureViewImpl._fields_ = []
WGPUTextureView = ctypes.POINTER(struct_WGPUTextureViewImpl)
class struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER(ctypes.Structure): pass
struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER._fields_ = [
  ('unused', WGPUBool),
]
class struct_WGPUAdapterPropertiesD3D(ctypes.Structure): pass
class struct_WGPUChainedStructOut(ctypes.Structure): pass
enum_WGPUSType = CEnum(ctypes.c_uint)
WGPUSType_ShaderSourceSPIRV = enum_WGPUSType.define('WGPUSType_ShaderSourceSPIRV', 1)
WGPUSType_ShaderSourceWGSL = enum_WGPUSType.define('WGPUSType_ShaderSourceWGSL', 2)
WGPUSType_RenderPassMaxDrawCount = enum_WGPUSType.define('WGPUSType_RenderPassMaxDrawCount', 3)
WGPUSType_SurfaceSourceMetalLayer = enum_WGPUSType.define('WGPUSType_SurfaceSourceMetalLayer', 4)
WGPUSType_SurfaceSourceWindowsHWND = enum_WGPUSType.define('WGPUSType_SurfaceSourceWindowsHWND', 5)
WGPUSType_SurfaceSourceXlibWindow = enum_WGPUSType.define('WGPUSType_SurfaceSourceXlibWindow', 6)
WGPUSType_SurfaceSourceWaylandSurface = enum_WGPUSType.define('WGPUSType_SurfaceSourceWaylandSurface', 7)
WGPUSType_SurfaceSourceAndroidNativeWindow = enum_WGPUSType.define('WGPUSType_SurfaceSourceAndroidNativeWindow', 8)
WGPUSType_SurfaceSourceXCBWindow = enum_WGPUSType.define('WGPUSType_SurfaceSourceXCBWindow', 9)
WGPUSType_AdapterPropertiesSubgroups = enum_WGPUSType.define('WGPUSType_AdapterPropertiesSubgroups', 10)
WGPUSType_TextureBindingViewDimensionDescriptor = enum_WGPUSType.define('WGPUSType_TextureBindingViewDimensionDescriptor', 131072)
WGPUSType_SurfaceSourceCanvasHTMLSelector_Emscripten = enum_WGPUSType.define('WGPUSType_SurfaceSourceCanvasHTMLSelector_Emscripten', 262144)
WGPUSType_SurfaceDescriptorFromWindowsCoreWindow = enum_WGPUSType.define('WGPUSType_SurfaceDescriptorFromWindowsCoreWindow', 327680)
WGPUSType_ExternalTextureBindingEntry = enum_WGPUSType.define('WGPUSType_ExternalTextureBindingEntry', 327681)
WGPUSType_ExternalTextureBindingLayout = enum_WGPUSType.define('WGPUSType_ExternalTextureBindingLayout', 327682)
WGPUSType_SurfaceDescriptorFromWindowsSwapChainPanel = enum_WGPUSType.define('WGPUSType_SurfaceDescriptorFromWindowsSwapChainPanel', 327683)
WGPUSType_DawnTextureInternalUsageDescriptor = enum_WGPUSType.define('WGPUSType_DawnTextureInternalUsageDescriptor', 327684)
WGPUSType_DawnEncoderInternalUsageDescriptor = enum_WGPUSType.define('WGPUSType_DawnEncoderInternalUsageDescriptor', 327685)
WGPUSType_DawnInstanceDescriptor = enum_WGPUSType.define('WGPUSType_DawnInstanceDescriptor', 327686)
WGPUSType_DawnCacheDeviceDescriptor = enum_WGPUSType.define('WGPUSType_DawnCacheDeviceDescriptor', 327687)
WGPUSType_DawnAdapterPropertiesPowerPreference = enum_WGPUSType.define('WGPUSType_DawnAdapterPropertiesPowerPreference', 327688)
WGPUSType_DawnBufferDescriptorErrorInfoFromWireClient = enum_WGPUSType.define('WGPUSType_DawnBufferDescriptorErrorInfoFromWireClient', 327689)
WGPUSType_DawnTogglesDescriptor = enum_WGPUSType.define('WGPUSType_DawnTogglesDescriptor', 327690)
WGPUSType_DawnShaderModuleSPIRVOptionsDescriptor = enum_WGPUSType.define('WGPUSType_DawnShaderModuleSPIRVOptionsDescriptor', 327691)
WGPUSType_RequestAdapterOptionsLUID = enum_WGPUSType.define('WGPUSType_RequestAdapterOptionsLUID', 327692)
WGPUSType_RequestAdapterOptionsGetGLProc = enum_WGPUSType.define('WGPUSType_RequestAdapterOptionsGetGLProc', 327693)
WGPUSType_RequestAdapterOptionsD3D11Device = enum_WGPUSType.define('WGPUSType_RequestAdapterOptionsD3D11Device', 327694)
WGPUSType_DawnRenderPassColorAttachmentRenderToSingleSampled = enum_WGPUSType.define('WGPUSType_DawnRenderPassColorAttachmentRenderToSingleSampled', 327695)
WGPUSType_RenderPassPixelLocalStorage = enum_WGPUSType.define('WGPUSType_RenderPassPixelLocalStorage', 327696)
WGPUSType_PipelineLayoutPixelLocalStorage = enum_WGPUSType.define('WGPUSType_PipelineLayoutPixelLocalStorage', 327697)
WGPUSType_BufferHostMappedPointer = enum_WGPUSType.define('WGPUSType_BufferHostMappedPointer', 327698)
WGPUSType_DawnExperimentalSubgroupLimits = enum_WGPUSType.define('WGPUSType_DawnExperimentalSubgroupLimits', 327699)
WGPUSType_AdapterPropertiesMemoryHeaps = enum_WGPUSType.define('WGPUSType_AdapterPropertiesMemoryHeaps', 327700)
WGPUSType_AdapterPropertiesD3D = enum_WGPUSType.define('WGPUSType_AdapterPropertiesD3D', 327701)
WGPUSType_AdapterPropertiesVk = enum_WGPUSType.define('WGPUSType_AdapterPropertiesVk', 327702)
WGPUSType_DawnWireWGSLControl = enum_WGPUSType.define('WGPUSType_DawnWireWGSLControl', 327703)
WGPUSType_DawnWGSLBlocklist = enum_WGPUSType.define('WGPUSType_DawnWGSLBlocklist', 327704)
WGPUSType_DrmFormatCapabilities = enum_WGPUSType.define('WGPUSType_DrmFormatCapabilities', 327705)
WGPUSType_ShaderModuleCompilationOptions = enum_WGPUSType.define('WGPUSType_ShaderModuleCompilationOptions', 327706)
WGPUSType_ColorTargetStateExpandResolveTextureDawn = enum_WGPUSType.define('WGPUSType_ColorTargetStateExpandResolveTextureDawn', 327707)
WGPUSType_RenderPassDescriptorExpandResolveRect = enum_WGPUSType.define('WGPUSType_RenderPassDescriptorExpandResolveRect', 327708)
WGPUSType_SharedTextureMemoryVkDedicatedAllocationDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryVkDedicatedAllocationDescriptor', 327709)
WGPUSType_SharedTextureMemoryAHardwareBufferDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryAHardwareBufferDescriptor', 327710)
WGPUSType_SharedTextureMemoryDmaBufDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryDmaBufDescriptor', 327711)
WGPUSType_SharedTextureMemoryOpaqueFDDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryOpaqueFDDescriptor', 327712)
WGPUSType_SharedTextureMemoryZirconHandleDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryZirconHandleDescriptor', 327713)
WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor', 327714)
WGPUSType_SharedTextureMemoryD3D11Texture2DDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryD3D11Texture2DDescriptor', 327715)
WGPUSType_SharedTextureMemoryIOSurfaceDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryIOSurfaceDescriptor', 327716)
WGPUSType_SharedTextureMemoryEGLImageDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryEGLImageDescriptor', 327717)
WGPUSType_SharedTextureMemoryInitializedBeginState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryInitializedBeginState', 327718)
WGPUSType_SharedTextureMemoryInitializedEndState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryInitializedEndState', 327719)
WGPUSType_SharedTextureMemoryVkImageLayoutBeginState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryVkImageLayoutBeginState', 327720)
WGPUSType_SharedTextureMemoryVkImageLayoutEndState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryVkImageLayoutEndState', 327721)
WGPUSType_SharedTextureMemoryD3DSwapchainBeginState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryD3DSwapchainBeginState', 327722)
WGPUSType_SharedFenceVkSemaphoreOpaqueFDDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceVkSemaphoreOpaqueFDDescriptor', 327723)
WGPUSType_SharedFenceVkSemaphoreOpaqueFDExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceVkSemaphoreOpaqueFDExportInfo', 327724)
WGPUSType_SharedFenceSyncFDDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceSyncFDDescriptor', 327725)
WGPUSType_SharedFenceSyncFDExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceSyncFDExportInfo', 327726)
WGPUSType_SharedFenceVkSemaphoreZirconHandleDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceVkSemaphoreZirconHandleDescriptor', 327727)
WGPUSType_SharedFenceVkSemaphoreZirconHandleExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceVkSemaphoreZirconHandleExportInfo', 327728)
WGPUSType_SharedFenceDXGISharedHandleDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceDXGISharedHandleDescriptor', 327729)
WGPUSType_SharedFenceDXGISharedHandleExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceDXGISharedHandleExportInfo', 327730)
WGPUSType_SharedFenceMTLSharedEventDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceMTLSharedEventDescriptor', 327731)
WGPUSType_SharedFenceMTLSharedEventExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceMTLSharedEventExportInfo', 327732)
WGPUSType_SharedBufferMemoryD3D12ResourceDescriptor = enum_WGPUSType.define('WGPUSType_SharedBufferMemoryD3D12ResourceDescriptor', 327733)
WGPUSType_StaticSamplerBindingLayout = enum_WGPUSType.define('WGPUSType_StaticSamplerBindingLayout', 327734)
WGPUSType_YCbCrVkDescriptor = enum_WGPUSType.define('WGPUSType_YCbCrVkDescriptor', 327735)
WGPUSType_SharedTextureMemoryAHardwareBufferProperties = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryAHardwareBufferProperties', 327736)
WGPUSType_AHardwareBufferProperties = enum_WGPUSType.define('WGPUSType_AHardwareBufferProperties', 327737)
WGPUSType_DawnExperimentalImmediateDataLimits = enum_WGPUSType.define('WGPUSType_DawnExperimentalImmediateDataLimits', 327738)
WGPUSType_DawnTexelCopyBufferRowAlignmentLimits = enum_WGPUSType.define('WGPUSType_DawnTexelCopyBufferRowAlignmentLimits', 327739)
WGPUSType_Force32 = enum_WGPUSType.define('WGPUSType_Force32', 2147483647)

WGPUSType = enum_WGPUSType
struct_WGPUChainedStructOut._fields_ = [
  ('next', ctypes.POINTER(struct_WGPUChainedStructOut)),
  ('sType', WGPUSType),
]
WGPUChainedStructOut = struct_WGPUChainedStructOut
uint32_t = ctypes.c_uint
struct_WGPUAdapterPropertiesD3D._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('shaderModel', uint32_t),
]
class struct_WGPUAdapterPropertiesSubgroups(ctypes.Structure): pass
struct_WGPUAdapterPropertiesSubgroups._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('subgroupMinSize', uint32_t),
  ('subgroupMaxSize', uint32_t),
]
class struct_WGPUAdapterPropertiesVk(ctypes.Structure): pass
struct_WGPUAdapterPropertiesVk._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('driverVersion', uint32_t),
]
class struct_WGPUBindGroupEntry(ctypes.Structure): pass
class struct_WGPUChainedStruct(ctypes.Structure): pass
struct_WGPUChainedStruct._fields_ = [
  ('next', ctypes.POINTER(struct_WGPUChainedStruct)),
  ('sType', WGPUSType),
]
WGPUChainedStruct = struct_WGPUChainedStruct
uint64_t = ctypes.c_ulong
struct_WGPUBindGroupEntry._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('binding', uint32_t),
  ('buffer', WGPUBuffer),
  ('offset', uint64_t),
  ('size', uint64_t),
  ('sampler', WGPUSampler),
  ('textureView', WGPUTextureView),
]
class struct_WGPUBlendComponent(ctypes.Structure): pass
enum_WGPUBlendOperation = CEnum(ctypes.c_uint)
WGPUBlendOperation_Undefined = enum_WGPUBlendOperation.define('WGPUBlendOperation_Undefined', 0)
WGPUBlendOperation_Add = enum_WGPUBlendOperation.define('WGPUBlendOperation_Add', 1)
WGPUBlendOperation_Subtract = enum_WGPUBlendOperation.define('WGPUBlendOperation_Subtract', 2)
WGPUBlendOperation_ReverseSubtract = enum_WGPUBlendOperation.define('WGPUBlendOperation_ReverseSubtract', 3)
WGPUBlendOperation_Min = enum_WGPUBlendOperation.define('WGPUBlendOperation_Min', 4)
WGPUBlendOperation_Max = enum_WGPUBlendOperation.define('WGPUBlendOperation_Max', 5)
WGPUBlendOperation_Force32 = enum_WGPUBlendOperation.define('WGPUBlendOperation_Force32', 2147483647)

WGPUBlendOperation = enum_WGPUBlendOperation
enum_WGPUBlendFactor = CEnum(ctypes.c_uint)
WGPUBlendFactor_Undefined = enum_WGPUBlendFactor.define('WGPUBlendFactor_Undefined', 0)
WGPUBlendFactor_Zero = enum_WGPUBlendFactor.define('WGPUBlendFactor_Zero', 1)
WGPUBlendFactor_One = enum_WGPUBlendFactor.define('WGPUBlendFactor_One', 2)
WGPUBlendFactor_Src = enum_WGPUBlendFactor.define('WGPUBlendFactor_Src', 3)
WGPUBlendFactor_OneMinusSrc = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusSrc', 4)
WGPUBlendFactor_SrcAlpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_SrcAlpha', 5)
WGPUBlendFactor_OneMinusSrcAlpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusSrcAlpha', 6)
WGPUBlendFactor_Dst = enum_WGPUBlendFactor.define('WGPUBlendFactor_Dst', 7)
WGPUBlendFactor_OneMinusDst = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusDst', 8)
WGPUBlendFactor_DstAlpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_DstAlpha', 9)
WGPUBlendFactor_OneMinusDstAlpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusDstAlpha', 10)
WGPUBlendFactor_SrcAlphaSaturated = enum_WGPUBlendFactor.define('WGPUBlendFactor_SrcAlphaSaturated', 11)
WGPUBlendFactor_Constant = enum_WGPUBlendFactor.define('WGPUBlendFactor_Constant', 12)
WGPUBlendFactor_OneMinusConstant = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusConstant', 13)
WGPUBlendFactor_Src1 = enum_WGPUBlendFactor.define('WGPUBlendFactor_Src1', 14)
WGPUBlendFactor_OneMinusSrc1 = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusSrc1', 15)
WGPUBlendFactor_Src1Alpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_Src1Alpha', 16)
WGPUBlendFactor_OneMinusSrc1Alpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusSrc1Alpha', 17)
WGPUBlendFactor_Force32 = enum_WGPUBlendFactor.define('WGPUBlendFactor_Force32', 2147483647)

WGPUBlendFactor = enum_WGPUBlendFactor
struct_WGPUBlendComponent._fields_ = [
  ('operation', WGPUBlendOperation),
  ('srcFactor', WGPUBlendFactor),
  ('dstFactor', WGPUBlendFactor),
]
class struct_WGPUBufferBindingLayout(ctypes.Structure): pass
enum_WGPUBufferBindingType = CEnum(ctypes.c_uint)
WGPUBufferBindingType_BindingNotUsed = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_BindingNotUsed', 0)
WGPUBufferBindingType_Uniform = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Uniform', 1)
WGPUBufferBindingType_Storage = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Storage', 2)
WGPUBufferBindingType_ReadOnlyStorage = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_ReadOnlyStorage', 3)
WGPUBufferBindingType_Force32 = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Force32', 2147483647)

WGPUBufferBindingType = enum_WGPUBufferBindingType
struct_WGPUBufferBindingLayout._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('type', WGPUBufferBindingType),
  ('hasDynamicOffset', WGPUBool),
  ('minBindingSize', uint64_t),
]
class struct_WGPUBufferHostMappedPointer(ctypes.Structure): pass
WGPUCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
struct_WGPUBufferHostMappedPointer._fields_ = [
  ('chain', WGPUChainedStruct),
  ('pointer', ctypes.c_void_p),
  ('disposeCallback', WGPUCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUBufferMapCallbackInfo(ctypes.Structure): pass
enum_WGPUCallbackMode = CEnum(ctypes.c_uint)
WGPUCallbackMode_WaitAnyOnly = enum_WGPUCallbackMode.define('WGPUCallbackMode_WaitAnyOnly', 1)
WGPUCallbackMode_AllowProcessEvents = enum_WGPUCallbackMode.define('WGPUCallbackMode_AllowProcessEvents', 2)
WGPUCallbackMode_AllowSpontaneous = enum_WGPUCallbackMode.define('WGPUCallbackMode_AllowSpontaneous', 3)
WGPUCallbackMode_Force32 = enum_WGPUCallbackMode.define('WGPUCallbackMode_Force32', 2147483647)

WGPUCallbackMode = enum_WGPUCallbackMode
enum_WGPUBufferMapAsyncStatus = CEnum(ctypes.c_uint)
WGPUBufferMapAsyncStatus_Success = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_Success', 1)
WGPUBufferMapAsyncStatus_InstanceDropped = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_InstanceDropped', 2)
WGPUBufferMapAsyncStatus_ValidationError = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_ValidationError', 3)
WGPUBufferMapAsyncStatus_Unknown = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_Unknown', 4)
WGPUBufferMapAsyncStatus_DeviceLost = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_DeviceLost', 5)
WGPUBufferMapAsyncStatus_DestroyedBeforeCallback = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_DestroyedBeforeCallback', 6)
WGPUBufferMapAsyncStatus_UnmappedBeforeCallback = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_UnmappedBeforeCallback', 7)
WGPUBufferMapAsyncStatus_MappingAlreadyPending = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_MappingAlreadyPending', 8)
WGPUBufferMapAsyncStatus_OffsetOutOfRange = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_OffsetOutOfRange', 9)
WGPUBufferMapAsyncStatus_SizeOutOfRange = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_SizeOutOfRange', 10)
WGPUBufferMapAsyncStatus_Force32 = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_Force32', 2147483647)

WGPUBufferMapCallback = ctypes.CFUNCTYPE(None, enum_WGPUBufferMapAsyncStatus, ctypes.c_void_p)
struct_WGPUBufferMapCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUBufferMapCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUColor(ctypes.Structure): pass
struct_WGPUColor._fields_ = [
  ('r', ctypes.c_double),
  ('g', ctypes.c_double),
  ('b', ctypes.c_double),
  ('a', ctypes.c_double),
]
class struct_WGPUColorTargetStateExpandResolveTextureDawn(ctypes.Structure): pass
struct_WGPUColorTargetStateExpandResolveTextureDawn._fields_ = [
  ('chain', WGPUChainedStruct),
  ('enabled', WGPUBool),
]
class struct_WGPUCompilationInfoCallbackInfo(ctypes.Structure): pass
enum_WGPUCompilationInfoRequestStatus = CEnum(ctypes.c_uint)
WGPUCompilationInfoRequestStatus_Success = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Success', 1)
WGPUCompilationInfoRequestStatus_InstanceDropped = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_InstanceDropped', 2)
WGPUCompilationInfoRequestStatus_Error = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Error', 3)
WGPUCompilationInfoRequestStatus_DeviceLost = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_DeviceLost', 4)
WGPUCompilationInfoRequestStatus_Unknown = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Unknown', 5)
WGPUCompilationInfoRequestStatus_Force32 = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Force32', 2147483647)

class const_struct_WGPUCompilationInfo(ctypes.Structure): pass
size_t = ctypes.c_ulong
class struct_WGPUCompilationMessage(ctypes.Structure): pass
class struct_WGPUStringView(ctypes.Structure): pass
struct_WGPUStringView._fields_ = [
  ('data', ctypes.POINTER(ctypes.c_char)),
  ('length', size_t),
]
WGPUStringView = struct_WGPUStringView
enum_WGPUCompilationMessageType = CEnum(ctypes.c_uint)
WGPUCompilationMessageType_Error = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Error', 1)
WGPUCompilationMessageType_Warning = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Warning', 2)
WGPUCompilationMessageType_Info = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Info', 3)
WGPUCompilationMessageType_Force32 = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Force32', 2147483647)

WGPUCompilationMessageType = enum_WGPUCompilationMessageType
struct_WGPUCompilationMessage._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('message', WGPUStringView),
  ('type', WGPUCompilationMessageType),
  ('lineNum', uint64_t),
  ('linePos', uint64_t),
  ('offset', uint64_t),
  ('length', uint64_t),
  ('utf16LinePos', uint64_t),
  ('utf16Offset', uint64_t),
  ('utf16Length', uint64_t),
]
WGPUCompilationMessage = struct_WGPUCompilationMessage
const_struct_WGPUCompilationInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('messageCount', size_t),
  ('messages', ctypes.POINTER(WGPUCompilationMessage)),
]
WGPUCompilationInfoCallback = ctypes.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, ctypes.POINTER(const_struct_WGPUCompilationInfo), ctypes.c_void_p)
struct_WGPUCompilationInfoCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUCompilationInfoCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUComputePassTimestampWrites(ctypes.Structure): pass
struct_WGPUComputePassTimestampWrites._fields_ = [
  ('querySet', WGPUQuerySet),
  ('beginningOfPassWriteIndex', uint32_t),
  ('endOfPassWriteIndex', uint32_t),
]
class struct_WGPUCopyTextureForBrowserOptions(ctypes.Structure): pass
enum_WGPUAlphaMode = CEnum(ctypes.c_uint)
WGPUAlphaMode_Opaque = enum_WGPUAlphaMode.define('WGPUAlphaMode_Opaque', 1)
WGPUAlphaMode_Premultiplied = enum_WGPUAlphaMode.define('WGPUAlphaMode_Premultiplied', 2)
WGPUAlphaMode_Unpremultiplied = enum_WGPUAlphaMode.define('WGPUAlphaMode_Unpremultiplied', 3)
WGPUAlphaMode_Force32 = enum_WGPUAlphaMode.define('WGPUAlphaMode_Force32', 2147483647)

WGPUAlphaMode = enum_WGPUAlphaMode
struct_WGPUCopyTextureForBrowserOptions._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('flipY', WGPUBool),
  ('needsColorSpaceConversion', WGPUBool),
  ('srcAlphaMode', WGPUAlphaMode),
  ('srcTransferFunctionParameters', ctypes.POINTER(ctypes.c_float)),
  ('conversionMatrix', ctypes.POINTER(ctypes.c_float)),
  ('dstTransferFunctionParameters', ctypes.POINTER(ctypes.c_float)),
  ('dstAlphaMode', WGPUAlphaMode),
  ('internalUsage', WGPUBool),
]
class struct_WGPUCreateComputePipelineAsyncCallbackInfo(ctypes.Structure): pass
enum_WGPUCreatePipelineAsyncStatus = CEnum(ctypes.c_uint)
WGPUCreatePipelineAsyncStatus_Success = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Success', 1)
WGPUCreatePipelineAsyncStatus_InstanceDropped = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_InstanceDropped', 2)
WGPUCreatePipelineAsyncStatus_ValidationError = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_ValidationError', 3)
WGPUCreatePipelineAsyncStatus_InternalError = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_InternalError', 4)
WGPUCreatePipelineAsyncStatus_DeviceLost = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_DeviceLost', 5)
WGPUCreatePipelineAsyncStatus_DeviceDestroyed = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_DeviceDestroyed', 6)
WGPUCreatePipelineAsyncStatus_Unknown = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Unknown', 7)
WGPUCreatePipelineAsyncStatus_Force32 = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Force32', 2147483647)

WGPUCreateComputePipelineAsyncCallback = ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, ctypes.POINTER(struct_WGPUComputePipelineImpl), struct_WGPUStringView, ctypes.c_void_p)
struct_WGPUCreateComputePipelineAsyncCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUCreateComputePipelineAsyncCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo(ctypes.Structure): pass
WGPUCreateRenderPipelineAsyncCallback = ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, ctypes.POINTER(struct_WGPURenderPipelineImpl), struct_WGPUStringView, ctypes.c_void_p)
struct_WGPUCreateRenderPipelineAsyncCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUCreateRenderPipelineAsyncCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUDawnWGSLBlocklist(ctypes.Structure): pass
struct_WGPUDawnWGSLBlocklist._fields_ = [
  ('chain', WGPUChainedStruct),
  ('blocklistedFeatureCount', size_t),
  ('blocklistedFeatures', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
]
class struct_WGPUDawnAdapterPropertiesPowerPreference(ctypes.Structure): pass
enum_WGPUPowerPreference = CEnum(ctypes.c_uint)
WGPUPowerPreference_Undefined = enum_WGPUPowerPreference.define('WGPUPowerPreference_Undefined', 0)
WGPUPowerPreference_LowPower = enum_WGPUPowerPreference.define('WGPUPowerPreference_LowPower', 1)
WGPUPowerPreference_HighPerformance = enum_WGPUPowerPreference.define('WGPUPowerPreference_HighPerformance', 2)
WGPUPowerPreference_Force32 = enum_WGPUPowerPreference.define('WGPUPowerPreference_Force32', 2147483647)

WGPUPowerPreference = enum_WGPUPowerPreference
struct_WGPUDawnAdapterPropertiesPowerPreference._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('powerPreference', WGPUPowerPreference),
]
class struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient(ctypes.Structure): pass
struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient._fields_ = [
  ('chain', WGPUChainedStruct),
  ('outOfMemory', WGPUBool),
]
class struct_WGPUDawnEncoderInternalUsageDescriptor(ctypes.Structure): pass
struct_WGPUDawnEncoderInternalUsageDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('useInternalUsages', WGPUBool),
]
class struct_WGPUDawnExperimentalImmediateDataLimits(ctypes.Structure): pass
struct_WGPUDawnExperimentalImmediateDataLimits._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('maxImmediateDataRangeByteSize', uint32_t),
]
class struct_WGPUDawnExperimentalSubgroupLimits(ctypes.Structure): pass
struct_WGPUDawnExperimentalSubgroupLimits._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('minSubgroupSize', uint32_t),
  ('maxSubgroupSize', uint32_t),
]
class struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled(ctypes.Structure): pass
struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled._fields_ = [
  ('chain', WGPUChainedStruct),
  ('implicitSampleCount', uint32_t),
]
class struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor(ctypes.Structure): pass
struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('allowNonUniformDerivatives', WGPUBool),
]
class struct_WGPUDawnTexelCopyBufferRowAlignmentLimits(ctypes.Structure): pass
struct_WGPUDawnTexelCopyBufferRowAlignmentLimits._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('minTexelCopyBufferRowAlignment', uint32_t),
]
class struct_WGPUDawnTextureInternalUsageDescriptor(ctypes.Structure): pass
WGPUTextureUsage = ctypes.c_ulong
struct_WGPUDawnTextureInternalUsageDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('internalUsage', WGPUTextureUsage),
]
class struct_WGPUDawnTogglesDescriptor(ctypes.Structure): pass
struct_WGPUDawnTogglesDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('enabledToggleCount', size_t),
  ('enabledToggles', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
  ('disabledToggleCount', size_t),
  ('disabledToggles', ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
]
class struct_WGPUDawnWireWGSLControl(ctypes.Structure): pass
struct_WGPUDawnWireWGSLControl._fields_ = [
  ('chain', WGPUChainedStruct),
  ('enableExperimental', WGPUBool),
  ('enableUnsafe', WGPUBool),
  ('enableTesting', WGPUBool),
]
class struct_WGPUDeviceLostCallbackInfo(ctypes.Structure): pass
enum_WGPUDeviceLostReason = CEnum(ctypes.c_uint)
WGPUDeviceLostReason_Unknown = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Unknown', 1)
WGPUDeviceLostReason_Destroyed = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Destroyed', 2)
WGPUDeviceLostReason_InstanceDropped = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_InstanceDropped', 3)
WGPUDeviceLostReason_FailedCreation = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_FailedCreation', 4)
WGPUDeviceLostReason_Force32 = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Force32', 2147483647)

WGPUDeviceLostCallbackNew = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.POINTER(struct_WGPUDeviceImpl)), enum_WGPUDeviceLostReason, struct_WGPUStringView, ctypes.c_void_p)
struct_WGPUDeviceLostCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUDeviceLostCallbackNew),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUDrmFormatProperties(ctypes.Structure): pass
struct_WGPUDrmFormatProperties._fields_ = [
  ('modifier', uint64_t),
  ('modifierPlaneCount', uint32_t),
]
class struct_WGPUExtent2D(ctypes.Structure): pass
struct_WGPUExtent2D._fields_ = [
  ('width', uint32_t),
  ('height', uint32_t),
]
class struct_WGPUExtent3D(ctypes.Structure): pass
struct_WGPUExtent3D._fields_ = [
  ('width', uint32_t),
  ('height', uint32_t),
  ('depthOrArrayLayers', uint32_t),
]
class struct_WGPUExternalTextureBindingEntry(ctypes.Structure): pass
struct_WGPUExternalTextureBindingEntry._fields_ = [
  ('chain', WGPUChainedStruct),
  ('externalTexture', WGPUExternalTexture),
]
class struct_WGPUExternalTextureBindingLayout(ctypes.Structure): pass
struct_WGPUExternalTextureBindingLayout._fields_ = [
  ('chain', WGPUChainedStruct),
]
class struct_WGPUFormatCapabilities(ctypes.Structure): pass
struct_WGPUFormatCapabilities._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
]
class struct_WGPUFuture(ctypes.Structure): pass
struct_WGPUFuture._fields_ = [
  ('id', uint64_t),
]
class struct_WGPUInstanceFeatures(ctypes.Structure): pass
struct_WGPUInstanceFeatures._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('timedWaitAnyEnable', WGPUBool),
  ('timedWaitAnyMaxCount', size_t),
]
class struct_WGPULimits(ctypes.Structure): pass
struct_WGPULimits._fields_ = [
  ('maxTextureDimension1D', uint32_t),
  ('maxTextureDimension2D', uint32_t),
  ('maxTextureDimension3D', uint32_t),
  ('maxTextureArrayLayers', uint32_t),
  ('maxBindGroups', uint32_t),
  ('maxBindGroupsPlusVertexBuffers', uint32_t),
  ('maxBindingsPerBindGroup', uint32_t),
  ('maxDynamicUniformBuffersPerPipelineLayout', uint32_t),
  ('maxDynamicStorageBuffersPerPipelineLayout', uint32_t),
  ('maxSampledTexturesPerShaderStage', uint32_t),
  ('maxSamplersPerShaderStage', uint32_t),
  ('maxStorageBuffersPerShaderStage', uint32_t),
  ('maxStorageTexturesPerShaderStage', uint32_t),
  ('maxUniformBuffersPerShaderStage', uint32_t),
  ('maxUniformBufferBindingSize', uint64_t),
  ('maxStorageBufferBindingSize', uint64_t),
  ('minUniformBufferOffsetAlignment', uint32_t),
  ('minStorageBufferOffsetAlignment', uint32_t),
  ('maxVertexBuffers', uint32_t),
  ('maxBufferSize', uint64_t),
  ('maxVertexAttributes', uint32_t),
  ('maxVertexBufferArrayStride', uint32_t),
  ('maxInterStageShaderComponents', uint32_t),
  ('maxInterStageShaderVariables', uint32_t),
  ('maxColorAttachments', uint32_t),
  ('maxColorAttachmentBytesPerSample', uint32_t),
  ('maxComputeWorkgroupStorageSize', uint32_t),
  ('maxComputeInvocationsPerWorkgroup', uint32_t),
  ('maxComputeWorkgroupSizeX', uint32_t),
  ('maxComputeWorkgroupSizeY', uint32_t),
  ('maxComputeWorkgroupSizeZ', uint32_t),
  ('maxComputeWorkgroupsPerDimension', uint32_t),
  ('maxStorageBuffersInVertexStage', uint32_t),
  ('maxStorageTexturesInVertexStage', uint32_t),
  ('maxStorageBuffersInFragmentStage', uint32_t),
  ('maxStorageTexturesInFragmentStage', uint32_t),
]
class struct_WGPUMemoryHeapInfo(ctypes.Structure): pass
WGPUHeapProperty = ctypes.c_ulong
struct_WGPUMemoryHeapInfo._fields_ = [
  ('properties', WGPUHeapProperty),
  ('size', uint64_t),
]
class struct_WGPUMultisampleState(ctypes.Structure): pass
struct_WGPUMultisampleState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('count', uint32_t),
  ('mask', uint32_t),
  ('alphaToCoverageEnabled', WGPUBool),
]
class struct_WGPUOrigin2D(ctypes.Structure): pass
struct_WGPUOrigin2D._fields_ = [
  ('x', uint32_t),
  ('y', uint32_t),
]
class struct_WGPUOrigin3D(ctypes.Structure): pass
struct_WGPUOrigin3D._fields_ = [
  ('x', uint32_t),
  ('y', uint32_t),
  ('z', uint32_t),
]
class struct_WGPUPipelineLayoutStorageAttachment(ctypes.Structure): pass
enum_WGPUTextureFormat = CEnum(ctypes.c_uint)
WGPUTextureFormat_Undefined = enum_WGPUTextureFormat.define('WGPUTextureFormat_Undefined', 0)
WGPUTextureFormat_R8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8Unorm', 1)
WGPUTextureFormat_R8Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8Snorm', 2)
WGPUTextureFormat_R8Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8Uint', 3)
WGPUTextureFormat_R8Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8Sint', 4)
WGPUTextureFormat_R16Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Uint', 5)
WGPUTextureFormat_R16Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Sint', 6)
WGPUTextureFormat_R16Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Float', 7)
WGPUTextureFormat_RG8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG8Unorm', 8)
WGPUTextureFormat_RG8Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG8Snorm', 9)
WGPUTextureFormat_RG8Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG8Uint', 10)
WGPUTextureFormat_RG8Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG8Sint', 11)
WGPUTextureFormat_R32Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_R32Float', 12)
WGPUTextureFormat_R32Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R32Uint', 13)
WGPUTextureFormat_R32Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R32Sint', 14)
WGPUTextureFormat_RG16Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Uint', 15)
WGPUTextureFormat_RG16Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Sint', 16)
WGPUTextureFormat_RG16Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Float', 17)
WGPUTextureFormat_RGBA8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8Unorm', 18)
WGPUTextureFormat_RGBA8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8UnormSrgb', 19)
WGPUTextureFormat_RGBA8Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8Snorm', 20)
WGPUTextureFormat_RGBA8Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8Uint', 21)
WGPUTextureFormat_RGBA8Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8Sint', 22)
WGPUTextureFormat_BGRA8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BGRA8Unorm', 23)
WGPUTextureFormat_BGRA8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BGRA8UnormSrgb', 24)
WGPUTextureFormat_RGB10A2Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGB10A2Uint', 25)
WGPUTextureFormat_RGB10A2Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGB10A2Unorm', 26)
WGPUTextureFormat_RG11B10Ufloat = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG11B10Ufloat', 27)
WGPUTextureFormat_RGB9E5Ufloat = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGB9E5Ufloat', 28)
WGPUTextureFormat_RG32Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG32Float', 29)
WGPUTextureFormat_RG32Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG32Uint', 30)
WGPUTextureFormat_RG32Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG32Sint', 31)
WGPUTextureFormat_RGBA16Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Uint', 32)
WGPUTextureFormat_RGBA16Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Sint', 33)
WGPUTextureFormat_RGBA16Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Float', 34)
WGPUTextureFormat_RGBA32Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA32Float', 35)
WGPUTextureFormat_RGBA32Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA32Uint', 36)
WGPUTextureFormat_RGBA32Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA32Sint', 37)
WGPUTextureFormat_Stencil8 = enum_WGPUTextureFormat.define('WGPUTextureFormat_Stencil8', 38)
WGPUTextureFormat_Depth16Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth16Unorm', 39)
WGPUTextureFormat_Depth24Plus = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth24Plus', 40)
WGPUTextureFormat_Depth24PlusStencil8 = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth24PlusStencil8', 41)
WGPUTextureFormat_Depth32Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth32Float', 42)
WGPUTextureFormat_Depth32FloatStencil8 = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth32FloatStencil8', 43)
WGPUTextureFormat_BC1RGBAUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC1RGBAUnorm', 44)
WGPUTextureFormat_BC1RGBAUnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC1RGBAUnormSrgb', 45)
WGPUTextureFormat_BC2RGBAUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC2RGBAUnorm', 46)
WGPUTextureFormat_BC2RGBAUnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC2RGBAUnormSrgb', 47)
WGPUTextureFormat_BC3RGBAUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC3RGBAUnorm', 48)
WGPUTextureFormat_BC3RGBAUnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC3RGBAUnormSrgb', 49)
WGPUTextureFormat_BC4RUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC4RUnorm', 50)
WGPUTextureFormat_BC4RSnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC4RSnorm', 51)
WGPUTextureFormat_BC5RGUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC5RGUnorm', 52)
WGPUTextureFormat_BC5RGSnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC5RGSnorm', 53)
WGPUTextureFormat_BC6HRGBUfloat = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC6HRGBUfloat', 54)
WGPUTextureFormat_BC6HRGBFloat = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC6HRGBFloat', 55)
WGPUTextureFormat_BC7RGBAUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC7RGBAUnorm', 56)
WGPUTextureFormat_BC7RGBAUnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC7RGBAUnormSrgb', 57)
WGPUTextureFormat_ETC2RGB8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGB8Unorm', 58)
WGPUTextureFormat_ETC2RGB8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGB8UnormSrgb', 59)
WGPUTextureFormat_ETC2RGB8A1Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGB8A1Unorm', 60)
WGPUTextureFormat_ETC2RGB8A1UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGB8A1UnormSrgb', 61)
WGPUTextureFormat_ETC2RGBA8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGBA8Unorm', 62)
WGPUTextureFormat_ETC2RGBA8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGBA8UnormSrgb', 63)
WGPUTextureFormat_EACR11Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_EACR11Unorm', 64)
WGPUTextureFormat_EACR11Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_EACR11Snorm', 65)
WGPUTextureFormat_EACRG11Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_EACRG11Unorm', 66)
WGPUTextureFormat_EACRG11Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_EACRG11Snorm', 67)
WGPUTextureFormat_ASTC4x4Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC4x4Unorm', 68)
WGPUTextureFormat_ASTC4x4UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC4x4UnormSrgb', 69)
WGPUTextureFormat_ASTC5x4Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC5x4Unorm', 70)
WGPUTextureFormat_ASTC5x4UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC5x4UnormSrgb', 71)
WGPUTextureFormat_ASTC5x5Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC5x5Unorm', 72)
WGPUTextureFormat_ASTC5x5UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC5x5UnormSrgb', 73)
WGPUTextureFormat_ASTC6x5Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC6x5Unorm', 74)
WGPUTextureFormat_ASTC6x5UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC6x5UnormSrgb', 75)
WGPUTextureFormat_ASTC6x6Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC6x6Unorm', 76)
WGPUTextureFormat_ASTC6x6UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC6x6UnormSrgb', 77)
WGPUTextureFormat_ASTC8x5Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x5Unorm', 78)
WGPUTextureFormat_ASTC8x5UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x5UnormSrgb', 79)
WGPUTextureFormat_ASTC8x6Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x6Unorm', 80)
WGPUTextureFormat_ASTC8x6UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x6UnormSrgb', 81)
WGPUTextureFormat_ASTC8x8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x8Unorm', 82)
WGPUTextureFormat_ASTC8x8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x8UnormSrgb', 83)
WGPUTextureFormat_ASTC10x5Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x5Unorm', 84)
WGPUTextureFormat_ASTC10x5UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x5UnormSrgb', 85)
WGPUTextureFormat_ASTC10x6Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x6Unorm', 86)
WGPUTextureFormat_ASTC10x6UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x6UnormSrgb', 87)
WGPUTextureFormat_ASTC10x8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x8Unorm', 88)
WGPUTextureFormat_ASTC10x8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x8UnormSrgb', 89)
WGPUTextureFormat_ASTC10x10Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x10Unorm', 90)
WGPUTextureFormat_ASTC10x10UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x10UnormSrgb', 91)
WGPUTextureFormat_ASTC12x10Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC12x10Unorm', 92)
WGPUTextureFormat_ASTC12x10UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC12x10UnormSrgb', 93)
WGPUTextureFormat_ASTC12x12Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC12x12Unorm', 94)
WGPUTextureFormat_ASTC12x12UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC12x12UnormSrgb', 95)
WGPUTextureFormat_R16Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Unorm', 327680)
WGPUTextureFormat_RG16Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Unorm', 327681)
WGPUTextureFormat_RGBA16Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Unorm', 327682)
WGPUTextureFormat_R16Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Snorm', 327683)
WGPUTextureFormat_RG16Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Snorm', 327684)
WGPUTextureFormat_RGBA16Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Snorm', 327685)
WGPUTextureFormat_R8BG8Biplanar420Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8BG8Biplanar420Unorm', 327686)
WGPUTextureFormat_R10X6BG10X6Biplanar420Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R10X6BG10X6Biplanar420Unorm', 327687)
WGPUTextureFormat_R8BG8A8Triplanar420Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8BG8A8Triplanar420Unorm', 327688)
WGPUTextureFormat_R8BG8Biplanar422Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8BG8Biplanar422Unorm', 327689)
WGPUTextureFormat_R8BG8Biplanar444Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8BG8Biplanar444Unorm', 327690)
WGPUTextureFormat_R10X6BG10X6Biplanar422Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R10X6BG10X6Biplanar422Unorm', 327691)
WGPUTextureFormat_R10X6BG10X6Biplanar444Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R10X6BG10X6Biplanar444Unorm', 327692)
WGPUTextureFormat_External = enum_WGPUTextureFormat.define('WGPUTextureFormat_External', 327693)
WGPUTextureFormat_Force32 = enum_WGPUTextureFormat.define('WGPUTextureFormat_Force32', 2147483647)

WGPUTextureFormat = enum_WGPUTextureFormat
struct_WGPUPipelineLayoutStorageAttachment._fields_ = [
  ('offset', uint64_t),
  ('format', WGPUTextureFormat),
]
class struct_WGPUPopErrorScopeCallbackInfo(ctypes.Structure): pass
enum_WGPUPopErrorScopeStatus = CEnum(ctypes.c_uint)
WGPUPopErrorScopeStatus_Success = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_Success', 1)
WGPUPopErrorScopeStatus_InstanceDropped = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_InstanceDropped', 2)
WGPUPopErrorScopeStatus_Force32 = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_Force32', 2147483647)

enum_WGPUErrorType = CEnum(ctypes.c_uint)
WGPUErrorType_NoError = enum_WGPUErrorType.define('WGPUErrorType_NoError', 1)
WGPUErrorType_Validation = enum_WGPUErrorType.define('WGPUErrorType_Validation', 2)
WGPUErrorType_OutOfMemory = enum_WGPUErrorType.define('WGPUErrorType_OutOfMemory', 3)
WGPUErrorType_Internal = enum_WGPUErrorType.define('WGPUErrorType_Internal', 4)
WGPUErrorType_Unknown = enum_WGPUErrorType.define('WGPUErrorType_Unknown', 5)
WGPUErrorType_DeviceLost = enum_WGPUErrorType.define('WGPUErrorType_DeviceLost', 6)
WGPUErrorType_Force32 = enum_WGPUErrorType.define('WGPUErrorType_Force32', 2147483647)

WGPUPopErrorScopeCallback = ctypes.CFUNCTYPE(None, enum_WGPUPopErrorScopeStatus, enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p)
WGPUErrorCallback = ctypes.CFUNCTYPE(None, enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p)
struct_WGPUPopErrorScopeCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUPopErrorScopeCallback),
  ('oldCallback', WGPUErrorCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUPrimitiveState(ctypes.Structure): pass
enum_WGPUPrimitiveTopology = CEnum(ctypes.c_uint)
WGPUPrimitiveTopology_Undefined = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_Undefined', 0)
WGPUPrimitiveTopology_PointList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_PointList', 1)
WGPUPrimitiveTopology_LineList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_LineList', 2)
WGPUPrimitiveTopology_LineStrip = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_LineStrip', 3)
WGPUPrimitiveTopology_TriangleList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_TriangleList', 4)
WGPUPrimitiveTopology_TriangleStrip = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_TriangleStrip', 5)
WGPUPrimitiveTopology_Force32 = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_Force32', 2147483647)

WGPUPrimitiveTopology = enum_WGPUPrimitiveTopology
enum_WGPUIndexFormat = CEnum(ctypes.c_uint)
WGPUIndexFormat_Undefined = enum_WGPUIndexFormat.define('WGPUIndexFormat_Undefined', 0)
WGPUIndexFormat_Uint16 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Uint16', 1)
WGPUIndexFormat_Uint32 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Uint32', 2)
WGPUIndexFormat_Force32 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Force32', 2147483647)

WGPUIndexFormat = enum_WGPUIndexFormat
enum_WGPUFrontFace = CEnum(ctypes.c_uint)
WGPUFrontFace_Undefined = enum_WGPUFrontFace.define('WGPUFrontFace_Undefined', 0)
WGPUFrontFace_CCW = enum_WGPUFrontFace.define('WGPUFrontFace_CCW', 1)
WGPUFrontFace_CW = enum_WGPUFrontFace.define('WGPUFrontFace_CW', 2)
WGPUFrontFace_Force32 = enum_WGPUFrontFace.define('WGPUFrontFace_Force32', 2147483647)

WGPUFrontFace = enum_WGPUFrontFace
enum_WGPUCullMode = CEnum(ctypes.c_uint)
WGPUCullMode_Undefined = enum_WGPUCullMode.define('WGPUCullMode_Undefined', 0)
WGPUCullMode_None = enum_WGPUCullMode.define('WGPUCullMode_None', 1)
WGPUCullMode_Front = enum_WGPUCullMode.define('WGPUCullMode_Front', 2)
WGPUCullMode_Back = enum_WGPUCullMode.define('WGPUCullMode_Back', 3)
WGPUCullMode_Force32 = enum_WGPUCullMode.define('WGPUCullMode_Force32', 2147483647)

WGPUCullMode = enum_WGPUCullMode
struct_WGPUPrimitiveState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('topology', WGPUPrimitiveTopology),
  ('stripIndexFormat', WGPUIndexFormat),
  ('frontFace', WGPUFrontFace),
  ('cullMode', WGPUCullMode),
  ('unclippedDepth', WGPUBool),
]
class struct_WGPUQueueWorkDoneCallbackInfo(ctypes.Structure): pass
enum_WGPUQueueWorkDoneStatus = CEnum(ctypes.c_uint)
WGPUQueueWorkDoneStatus_Success = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Success', 1)
WGPUQueueWorkDoneStatus_InstanceDropped = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_InstanceDropped', 2)
WGPUQueueWorkDoneStatus_Error = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Error', 3)
WGPUQueueWorkDoneStatus_Unknown = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Unknown', 4)
WGPUQueueWorkDoneStatus_DeviceLost = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_DeviceLost', 5)
WGPUQueueWorkDoneStatus_Force32 = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Force32', 2147483647)

WGPUQueueWorkDoneCallback = ctypes.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, ctypes.c_void_p)
struct_WGPUQueueWorkDoneCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUQueueWorkDoneCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPURenderPassDepthStencilAttachment(ctypes.Structure): pass
enum_WGPULoadOp = CEnum(ctypes.c_uint)
WGPULoadOp_Undefined = enum_WGPULoadOp.define('WGPULoadOp_Undefined', 0)
WGPULoadOp_Load = enum_WGPULoadOp.define('WGPULoadOp_Load', 1)
WGPULoadOp_Clear = enum_WGPULoadOp.define('WGPULoadOp_Clear', 2)
WGPULoadOp_ExpandResolveTexture = enum_WGPULoadOp.define('WGPULoadOp_ExpandResolveTexture', 327683)
WGPULoadOp_Force32 = enum_WGPULoadOp.define('WGPULoadOp_Force32', 2147483647)

WGPULoadOp = enum_WGPULoadOp
enum_WGPUStoreOp = CEnum(ctypes.c_uint)
WGPUStoreOp_Undefined = enum_WGPUStoreOp.define('WGPUStoreOp_Undefined', 0)
WGPUStoreOp_Store = enum_WGPUStoreOp.define('WGPUStoreOp_Store', 1)
WGPUStoreOp_Discard = enum_WGPUStoreOp.define('WGPUStoreOp_Discard', 2)
WGPUStoreOp_Force32 = enum_WGPUStoreOp.define('WGPUStoreOp_Force32', 2147483647)

WGPUStoreOp = enum_WGPUStoreOp
struct_WGPURenderPassDepthStencilAttachment._fields_ = [
  ('view', WGPUTextureView),
  ('depthLoadOp', WGPULoadOp),
  ('depthStoreOp', WGPUStoreOp),
  ('depthClearValue', ctypes.c_float),
  ('depthReadOnly', WGPUBool),
  ('stencilLoadOp', WGPULoadOp),
  ('stencilStoreOp', WGPUStoreOp),
  ('stencilClearValue', uint32_t),
  ('stencilReadOnly', WGPUBool),
]
class struct_WGPURenderPassDescriptorExpandResolveRect(ctypes.Structure): pass
struct_WGPURenderPassDescriptorExpandResolveRect._fields_ = [
  ('chain', WGPUChainedStruct),
  ('x', uint32_t),
  ('y', uint32_t),
  ('width', uint32_t),
  ('height', uint32_t),
]
class struct_WGPURenderPassMaxDrawCount(ctypes.Structure): pass
struct_WGPURenderPassMaxDrawCount._fields_ = [
  ('chain', WGPUChainedStruct),
  ('maxDrawCount', uint64_t),
]
class struct_WGPURenderPassTimestampWrites(ctypes.Structure): pass
struct_WGPURenderPassTimestampWrites._fields_ = [
  ('querySet', WGPUQuerySet),
  ('beginningOfPassWriteIndex', uint32_t),
  ('endOfPassWriteIndex', uint32_t),
]
class struct_WGPURequestAdapterCallbackInfo(ctypes.Structure): pass
enum_WGPURequestAdapterStatus = CEnum(ctypes.c_uint)
WGPURequestAdapterStatus_Success = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Success', 1)
WGPURequestAdapterStatus_InstanceDropped = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_InstanceDropped', 2)
WGPURequestAdapterStatus_Unavailable = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Unavailable', 3)
WGPURequestAdapterStatus_Error = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Error', 4)
WGPURequestAdapterStatus_Unknown = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Unknown', 5)
WGPURequestAdapterStatus_Force32 = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Force32', 2147483647)

WGPURequestAdapterCallback = ctypes.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, ctypes.POINTER(struct_WGPUAdapterImpl), struct_WGPUStringView, ctypes.c_void_p)
struct_WGPURequestAdapterCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPURequestAdapterCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPURequestAdapterOptions(ctypes.Structure): pass
enum_WGPUFeatureLevel = CEnum(ctypes.c_uint)
WGPUFeatureLevel_Undefined = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Undefined', 0)
WGPUFeatureLevel_Compatibility = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Compatibility', 1)
WGPUFeatureLevel_Core = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Core', 2)
WGPUFeatureLevel_Force32 = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Force32', 2147483647)

WGPUFeatureLevel = enum_WGPUFeatureLevel
enum_WGPUBackendType = CEnum(ctypes.c_uint)
WGPUBackendType_Undefined = enum_WGPUBackendType.define('WGPUBackendType_Undefined', 0)
WGPUBackendType_Null = enum_WGPUBackendType.define('WGPUBackendType_Null', 1)
WGPUBackendType_WebGPU = enum_WGPUBackendType.define('WGPUBackendType_WebGPU', 2)
WGPUBackendType_D3D11 = enum_WGPUBackendType.define('WGPUBackendType_D3D11', 3)
WGPUBackendType_D3D12 = enum_WGPUBackendType.define('WGPUBackendType_D3D12', 4)
WGPUBackendType_Metal = enum_WGPUBackendType.define('WGPUBackendType_Metal', 5)
WGPUBackendType_Vulkan = enum_WGPUBackendType.define('WGPUBackendType_Vulkan', 6)
WGPUBackendType_OpenGL = enum_WGPUBackendType.define('WGPUBackendType_OpenGL', 7)
WGPUBackendType_OpenGLES = enum_WGPUBackendType.define('WGPUBackendType_OpenGLES', 8)
WGPUBackendType_Force32 = enum_WGPUBackendType.define('WGPUBackendType_Force32', 2147483647)

WGPUBackendType = enum_WGPUBackendType
struct_WGPURequestAdapterOptions._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('compatibleSurface', WGPUSurface),
  ('featureLevel', WGPUFeatureLevel),
  ('powerPreference', WGPUPowerPreference),
  ('backendType', WGPUBackendType),
  ('forceFallbackAdapter', WGPUBool),
  ('compatibilityMode', WGPUBool),
]
class struct_WGPURequestDeviceCallbackInfo(ctypes.Structure): pass
enum_WGPURequestDeviceStatus = CEnum(ctypes.c_uint)
WGPURequestDeviceStatus_Success = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Success', 1)
WGPURequestDeviceStatus_InstanceDropped = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_InstanceDropped', 2)
WGPURequestDeviceStatus_Error = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Error', 3)
WGPURequestDeviceStatus_Unknown = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Unknown', 4)
WGPURequestDeviceStatus_Force32 = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Force32', 2147483647)

WGPURequestDeviceCallback = ctypes.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, ctypes.POINTER(struct_WGPUDeviceImpl), struct_WGPUStringView, ctypes.c_void_p)
struct_WGPURequestDeviceCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPURequestDeviceCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUSamplerBindingLayout(ctypes.Structure): pass
enum_WGPUSamplerBindingType = CEnum(ctypes.c_uint)
WGPUSamplerBindingType_BindingNotUsed = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_BindingNotUsed', 0)
WGPUSamplerBindingType_Filtering = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Filtering', 1)
WGPUSamplerBindingType_NonFiltering = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_NonFiltering', 2)
WGPUSamplerBindingType_Comparison = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Comparison', 3)
WGPUSamplerBindingType_Force32 = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Force32', 2147483647)

WGPUSamplerBindingType = enum_WGPUSamplerBindingType
struct_WGPUSamplerBindingLayout._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('type', WGPUSamplerBindingType),
]
class struct_WGPUShaderModuleCompilationOptions(ctypes.Structure): pass
struct_WGPUShaderModuleCompilationOptions._fields_ = [
  ('chain', WGPUChainedStruct),
  ('strictMath', WGPUBool),
]
class struct_WGPUShaderSourceSPIRV(ctypes.Structure): pass
struct_WGPUShaderSourceSPIRV._fields_ = [
  ('chain', WGPUChainedStruct),
  ('codeSize', uint32_t),
  ('code', ctypes.POINTER(uint32_t)),
]
class struct_WGPUSharedBufferMemoryBeginAccessDescriptor(ctypes.Structure): pass
struct_WGPUSharedBufferMemoryBeginAccessDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('initialized', WGPUBool),
  ('fenceCount', size_t),
  ('fences', ctypes.POINTER(WGPUSharedFence)),
  ('signaledValues', ctypes.POINTER(uint64_t)),
]
class struct_WGPUSharedBufferMemoryEndAccessState(ctypes.Structure): pass
struct_WGPUSharedBufferMemoryEndAccessState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
  ('initialized', WGPUBool),
  ('fenceCount', size_t),
  ('fences', ctypes.POINTER(WGPUSharedFence)),
  ('signaledValues', ctypes.POINTER(uint64_t)),
]
class struct_WGPUSharedBufferMemoryProperties(ctypes.Structure): pass
WGPUBufferUsage = ctypes.c_ulong
struct_WGPUSharedBufferMemoryProperties._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
  ('usage', WGPUBufferUsage),
  ('size', uint64_t),
]
class struct_WGPUSharedFenceDXGISharedHandleDescriptor(ctypes.Structure): pass
struct_WGPUSharedFenceDXGISharedHandleDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('handle', ctypes.c_void_p),
]
class struct_WGPUSharedFenceDXGISharedHandleExportInfo(ctypes.Structure): pass
struct_WGPUSharedFenceDXGISharedHandleExportInfo._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('handle', ctypes.c_void_p),
]
class struct_WGPUSharedFenceMTLSharedEventDescriptor(ctypes.Structure): pass
struct_WGPUSharedFenceMTLSharedEventDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('sharedEvent', ctypes.c_void_p),
]
class struct_WGPUSharedFenceMTLSharedEventExportInfo(ctypes.Structure): pass
struct_WGPUSharedFenceMTLSharedEventExportInfo._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('sharedEvent', ctypes.c_void_p),
]
class struct_WGPUSharedFenceExportInfo(ctypes.Structure): pass
enum_WGPUSharedFenceType = CEnum(ctypes.c_uint)
WGPUSharedFenceType_VkSemaphoreOpaqueFD = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_VkSemaphoreOpaqueFD', 1)
WGPUSharedFenceType_SyncFD = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_SyncFD', 2)
WGPUSharedFenceType_VkSemaphoreZirconHandle = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_VkSemaphoreZirconHandle', 3)
WGPUSharedFenceType_DXGISharedHandle = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_DXGISharedHandle', 4)
WGPUSharedFenceType_MTLSharedEvent = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_MTLSharedEvent', 5)
WGPUSharedFenceType_Force32 = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_Force32', 2147483647)

WGPUSharedFenceType = enum_WGPUSharedFenceType
struct_WGPUSharedFenceExportInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
  ('type', WGPUSharedFenceType),
]
class struct_WGPUSharedFenceSyncFDDescriptor(ctypes.Structure): pass
struct_WGPUSharedFenceSyncFDDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('handle', ctypes.c_int),
]
class struct_WGPUSharedFenceSyncFDExportInfo(ctypes.Structure): pass
struct_WGPUSharedFenceSyncFDExportInfo._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('handle', ctypes.c_int),
]
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor(ctypes.Structure): pass
struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('handle', ctypes.c_int),
]
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo(ctypes.Structure): pass
struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('handle', ctypes.c_int),
]
class struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor(ctypes.Structure): pass
struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('handle', uint32_t),
]
class struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo(ctypes.Structure): pass
struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('handle', uint32_t),
]
class struct_WGPUSharedTextureMemoryD3DSwapchainBeginState(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryD3DSwapchainBeginState._fields_ = [
  ('chain', WGPUChainedStruct),
  ('isSwapchain', WGPUBool),
]
class struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('handle', ctypes.c_void_p),
  ('useKeyedMutex', WGPUBool),
]
class struct_WGPUSharedTextureMemoryEGLImageDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryEGLImageDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('image', ctypes.c_void_p),
]
class struct_WGPUSharedTextureMemoryIOSurfaceDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryIOSurfaceDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('ioSurface', ctypes.c_void_p),
]
class struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('handle', ctypes.c_void_p),
  ('useExternalFormat', WGPUBool),
]
class struct_WGPUSharedTextureMemoryBeginAccessDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryBeginAccessDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('concurrentRead', WGPUBool),
  ('initialized', WGPUBool),
  ('fenceCount', size_t),
  ('fences', ctypes.POINTER(WGPUSharedFence)),
  ('signaledValues', ctypes.POINTER(uint64_t)),
]
class struct_WGPUSharedTextureMemoryDmaBufPlane(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryDmaBufPlane._fields_ = [
  ('fd', ctypes.c_int),
  ('offset', uint64_t),
  ('stride', uint32_t),
]
class struct_WGPUSharedTextureMemoryEndAccessState(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryEndAccessState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
  ('initialized', WGPUBool),
  ('fenceCount', size_t),
  ('fences', ctypes.POINTER(WGPUSharedFence)),
  ('signaledValues', ctypes.POINTER(uint64_t)),
]
class struct_WGPUSharedTextureMemoryOpaqueFDDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryOpaqueFDDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('vkImageCreateInfo', ctypes.c_void_p),
  ('memoryFD', ctypes.c_int),
  ('memoryTypeIndex', uint32_t),
  ('allocationSize', uint64_t),
  ('dedicatedAllocation', WGPUBool),
]
class struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('dedicatedAllocation', WGPUBool),
]
class struct_WGPUSharedTextureMemoryVkImageLayoutBeginState(ctypes.Structure): pass
int32_t = ctypes.c_int
struct_WGPUSharedTextureMemoryVkImageLayoutBeginState._fields_ = [
  ('chain', WGPUChainedStruct),
  ('oldLayout', int32_t),
  ('newLayout', int32_t),
]
class struct_WGPUSharedTextureMemoryVkImageLayoutEndState(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryVkImageLayoutEndState._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('oldLayout', int32_t),
  ('newLayout', int32_t),
]
class struct_WGPUSharedTextureMemoryZirconHandleDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryZirconHandleDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('memoryFD', uint32_t),
  ('allocationSize', uint64_t),
]
class struct_WGPUStaticSamplerBindingLayout(ctypes.Structure): pass
struct_WGPUStaticSamplerBindingLayout._fields_ = [
  ('chain', WGPUChainedStruct),
  ('sampler', WGPUSampler),
  ('sampledTextureBinding', uint32_t),
]
class struct_WGPUStencilFaceState(ctypes.Structure): pass
enum_WGPUCompareFunction = CEnum(ctypes.c_uint)
WGPUCompareFunction_Undefined = enum_WGPUCompareFunction.define('WGPUCompareFunction_Undefined', 0)
WGPUCompareFunction_Never = enum_WGPUCompareFunction.define('WGPUCompareFunction_Never', 1)
WGPUCompareFunction_Less = enum_WGPUCompareFunction.define('WGPUCompareFunction_Less', 2)
WGPUCompareFunction_Equal = enum_WGPUCompareFunction.define('WGPUCompareFunction_Equal', 3)
WGPUCompareFunction_LessEqual = enum_WGPUCompareFunction.define('WGPUCompareFunction_LessEqual', 4)
WGPUCompareFunction_Greater = enum_WGPUCompareFunction.define('WGPUCompareFunction_Greater', 5)
WGPUCompareFunction_NotEqual = enum_WGPUCompareFunction.define('WGPUCompareFunction_NotEqual', 6)
WGPUCompareFunction_GreaterEqual = enum_WGPUCompareFunction.define('WGPUCompareFunction_GreaterEqual', 7)
WGPUCompareFunction_Always = enum_WGPUCompareFunction.define('WGPUCompareFunction_Always', 8)
WGPUCompareFunction_Force32 = enum_WGPUCompareFunction.define('WGPUCompareFunction_Force32', 2147483647)

WGPUCompareFunction = enum_WGPUCompareFunction
enum_WGPUStencilOperation = CEnum(ctypes.c_uint)
WGPUStencilOperation_Undefined = enum_WGPUStencilOperation.define('WGPUStencilOperation_Undefined', 0)
WGPUStencilOperation_Keep = enum_WGPUStencilOperation.define('WGPUStencilOperation_Keep', 1)
WGPUStencilOperation_Zero = enum_WGPUStencilOperation.define('WGPUStencilOperation_Zero', 2)
WGPUStencilOperation_Replace = enum_WGPUStencilOperation.define('WGPUStencilOperation_Replace', 3)
WGPUStencilOperation_Invert = enum_WGPUStencilOperation.define('WGPUStencilOperation_Invert', 4)
WGPUStencilOperation_IncrementClamp = enum_WGPUStencilOperation.define('WGPUStencilOperation_IncrementClamp', 5)
WGPUStencilOperation_DecrementClamp = enum_WGPUStencilOperation.define('WGPUStencilOperation_DecrementClamp', 6)
WGPUStencilOperation_IncrementWrap = enum_WGPUStencilOperation.define('WGPUStencilOperation_IncrementWrap', 7)
WGPUStencilOperation_DecrementWrap = enum_WGPUStencilOperation.define('WGPUStencilOperation_DecrementWrap', 8)
WGPUStencilOperation_Force32 = enum_WGPUStencilOperation.define('WGPUStencilOperation_Force32', 2147483647)

WGPUStencilOperation = enum_WGPUStencilOperation
struct_WGPUStencilFaceState._fields_ = [
  ('compare', WGPUCompareFunction),
  ('failOp', WGPUStencilOperation),
  ('depthFailOp', WGPUStencilOperation),
  ('passOp', WGPUStencilOperation),
]
class struct_WGPUStorageTextureBindingLayout(ctypes.Structure): pass
enum_WGPUStorageTextureAccess = CEnum(ctypes.c_uint)
WGPUStorageTextureAccess_BindingNotUsed = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_BindingNotUsed', 0)
WGPUStorageTextureAccess_WriteOnly = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_WriteOnly', 1)
WGPUStorageTextureAccess_ReadOnly = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_ReadOnly', 2)
WGPUStorageTextureAccess_ReadWrite = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_ReadWrite', 3)
WGPUStorageTextureAccess_Force32 = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_Force32', 2147483647)

WGPUStorageTextureAccess = enum_WGPUStorageTextureAccess
enum_WGPUTextureViewDimension = CEnum(ctypes.c_uint)
WGPUTextureViewDimension_Undefined = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Undefined', 0)
WGPUTextureViewDimension_1D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_1D', 1)
WGPUTextureViewDimension_2D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_2D', 2)
WGPUTextureViewDimension_2DArray = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_2DArray', 3)
WGPUTextureViewDimension_Cube = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Cube', 4)
WGPUTextureViewDimension_CubeArray = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_CubeArray', 5)
WGPUTextureViewDimension_3D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_3D', 6)
WGPUTextureViewDimension_Force32 = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Force32', 2147483647)

WGPUTextureViewDimension = enum_WGPUTextureViewDimension
struct_WGPUStorageTextureBindingLayout._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('access', WGPUStorageTextureAccess),
  ('format', WGPUTextureFormat),
  ('viewDimension', WGPUTextureViewDimension),
]
class struct_WGPUSupportedFeatures(ctypes.Structure): pass
enum_WGPUFeatureName = CEnum(ctypes.c_uint)
WGPUFeatureName_DepthClipControl = enum_WGPUFeatureName.define('WGPUFeatureName_DepthClipControl', 1)
WGPUFeatureName_Depth32FloatStencil8 = enum_WGPUFeatureName.define('WGPUFeatureName_Depth32FloatStencil8', 2)
WGPUFeatureName_TimestampQuery = enum_WGPUFeatureName.define('WGPUFeatureName_TimestampQuery', 3)
WGPUFeatureName_TextureCompressionBC = enum_WGPUFeatureName.define('WGPUFeatureName_TextureCompressionBC', 4)
WGPUFeatureName_TextureCompressionETC2 = enum_WGPUFeatureName.define('WGPUFeatureName_TextureCompressionETC2', 5)
WGPUFeatureName_TextureCompressionASTC = enum_WGPUFeatureName.define('WGPUFeatureName_TextureCompressionASTC', 6)
WGPUFeatureName_IndirectFirstInstance = enum_WGPUFeatureName.define('WGPUFeatureName_IndirectFirstInstance', 7)
WGPUFeatureName_ShaderF16 = enum_WGPUFeatureName.define('WGPUFeatureName_ShaderF16', 8)
WGPUFeatureName_RG11B10UfloatRenderable = enum_WGPUFeatureName.define('WGPUFeatureName_RG11B10UfloatRenderable', 9)
WGPUFeatureName_BGRA8UnormStorage = enum_WGPUFeatureName.define('WGPUFeatureName_BGRA8UnormStorage', 10)
WGPUFeatureName_Float32Filterable = enum_WGPUFeatureName.define('WGPUFeatureName_Float32Filterable', 11)
WGPUFeatureName_Float32Blendable = enum_WGPUFeatureName.define('WGPUFeatureName_Float32Blendable', 12)
WGPUFeatureName_Subgroups = enum_WGPUFeatureName.define('WGPUFeatureName_Subgroups', 13)
WGPUFeatureName_SubgroupsF16 = enum_WGPUFeatureName.define('WGPUFeatureName_SubgroupsF16', 14)
WGPUFeatureName_DawnInternalUsages = enum_WGPUFeatureName.define('WGPUFeatureName_DawnInternalUsages', 327680)
WGPUFeatureName_DawnMultiPlanarFormats = enum_WGPUFeatureName.define('WGPUFeatureName_DawnMultiPlanarFormats', 327681)
WGPUFeatureName_DawnNative = enum_WGPUFeatureName.define('WGPUFeatureName_DawnNative', 327682)
WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses = enum_WGPUFeatureName.define('WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses', 327683)
WGPUFeatureName_ImplicitDeviceSynchronization = enum_WGPUFeatureName.define('WGPUFeatureName_ImplicitDeviceSynchronization', 327684)
WGPUFeatureName_ChromiumExperimentalImmediateData = enum_WGPUFeatureName.define('WGPUFeatureName_ChromiumExperimentalImmediateData', 327685)
WGPUFeatureName_TransientAttachments = enum_WGPUFeatureName.define('WGPUFeatureName_TransientAttachments', 327686)
WGPUFeatureName_MSAARenderToSingleSampled = enum_WGPUFeatureName.define('WGPUFeatureName_MSAARenderToSingleSampled', 327687)
WGPUFeatureName_DualSourceBlending = enum_WGPUFeatureName.define('WGPUFeatureName_DualSourceBlending', 327688)
WGPUFeatureName_D3D11MultithreadProtected = enum_WGPUFeatureName.define('WGPUFeatureName_D3D11MultithreadProtected', 327689)
WGPUFeatureName_ANGLETextureSharing = enum_WGPUFeatureName.define('WGPUFeatureName_ANGLETextureSharing', 327690)
WGPUFeatureName_PixelLocalStorageCoherent = enum_WGPUFeatureName.define('WGPUFeatureName_PixelLocalStorageCoherent', 327691)
WGPUFeatureName_PixelLocalStorageNonCoherent = enum_WGPUFeatureName.define('WGPUFeatureName_PixelLocalStorageNonCoherent', 327692)
WGPUFeatureName_Unorm16TextureFormats = enum_WGPUFeatureName.define('WGPUFeatureName_Unorm16TextureFormats', 327693)
WGPUFeatureName_Snorm16TextureFormats = enum_WGPUFeatureName.define('WGPUFeatureName_Snorm16TextureFormats', 327694)
WGPUFeatureName_MultiPlanarFormatExtendedUsages = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatExtendedUsages', 327695)
WGPUFeatureName_MultiPlanarFormatP010 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatP010', 327696)
WGPUFeatureName_HostMappedPointer = enum_WGPUFeatureName.define('WGPUFeatureName_HostMappedPointer', 327697)
WGPUFeatureName_MultiPlanarRenderTargets = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarRenderTargets', 327698)
WGPUFeatureName_MultiPlanarFormatNv12a = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatNv12a', 327699)
WGPUFeatureName_FramebufferFetch = enum_WGPUFeatureName.define('WGPUFeatureName_FramebufferFetch', 327700)
WGPUFeatureName_BufferMapExtendedUsages = enum_WGPUFeatureName.define('WGPUFeatureName_BufferMapExtendedUsages', 327701)
WGPUFeatureName_AdapterPropertiesMemoryHeaps = enum_WGPUFeatureName.define('WGPUFeatureName_AdapterPropertiesMemoryHeaps', 327702)
WGPUFeatureName_AdapterPropertiesD3D = enum_WGPUFeatureName.define('WGPUFeatureName_AdapterPropertiesD3D', 327703)
WGPUFeatureName_AdapterPropertiesVk = enum_WGPUFeatureName.define('WGPUFeatureName_AdapterPropertiesVk', 327704)
WGPUFeatureName_R8UnormStorage = enum_WGPUFeatureName.define('WGPUFeatureName_R8UnormStorage', 327705)
WGPUFeatureName_FormatCapabilities = enum_WGPUFeatureName.define('WGPUFeatureName_FormatCapabilities', 327706)
WGPUFeatureName_DrmFormatCapabilities = enum_WGPUFeatureName.define('WGPUFeatureName_DrmFormatCapabilities', 327707)
WGPUFeatureName_Norm16TextureFormats = enum_WGPUFeatureName.define('WGPUFeatureName_Norm16TextureFormats', 327708)
WGPUFeatureName_MultiPlanarFormatNv16 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatNv16', 327709)
WGPUFeatureName_MultiPlanarFormatNv24 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatNv24', 327710)
WGPUFeatureName_MultiPlanarFormatP210 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatP210', 327711)
WGPUFeatureName_MultiPlanarFormatP410 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatP410', 327712)
WGPUFeatureName_SharedTextureMemoryVkDedicatedAllocation = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryVkDedicatedAllocation', 327713)
WGPUFeatureName_SharedTextureMemoryAHardwareBuffer = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryAHardwareBuffer', 327714)
WGPUFeatureName_SharedTextureMemoryDmaBuf = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryDmaBuf', 327715)
WGPUFeatureName_SharedTextureMemoryOpaqueFD = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryOpaqueFD', 327716)
WGPUFeatureName_SharedTextureMemoryZirconHandle = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryZirconHandle', 327717)
WGPUFeatureName_SharedTextureMemoryDXGISharedHandle = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryDXGISharedHandle', 327718)
WGPUFeatureName_SharedTextureMemoryD3D11Texture2D = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryD3D11Texture2D', 327719)
WGPUFeatureName_SharedTextureMemoryIOSurface = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryIOSurface', 327720)
WGPUFeatureName_SharedTextureMemoryEGLImage = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryEGLImage', 327721)
WGPUFeatureName_SharedFenceVkSemaphoreOpaqueFD = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceVkSemaphoreOpaqueFD', 327722)
WGPUFeatureName_SharedFenceSyncFD = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceSyncFD', 327723)
WGPUFeatureName_SharedFenceVkSemaphoreZirconHandle = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceVkSemaphoreZirconHandle', 327724)
WGPUFeatureName_SharedFenceDXGISharedHandle = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceDXGISharedHandle', 327725)
WGPUFeatureName_SharedFenceMTLSharedEvent = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceMTLSharedEvent', 327726)
WGPUFeatureName_SharedBufferMemoryD3D12Resource = enum_WGPUFeatureName.define('WGPUFeatureName_SharedBufferMemoryD3D12Resource', 327727)
WGPUFeatureName_StaticSamplers = enum_WGPUFeatureName.define('WGPUFeatureName_StaticSamplers', 327728)
WGPUFeatureName_YCbCrVulkanSamplers = enum_WGPUFeatureName.define('WGPUFeatureName_YCbCrVulkanSamplers', 327729)
WGPUFeatureName_ShaderModuleCompilationOptions = enum_WGPUFeatureName.define('WGPUFeatureName_ShaderModuleCompilationOptions', 327730)
WGPUFeatureName_DawnLoadResolveTexture = enum_WGPUFeatureName.define('WGPUFeatureName_DawnLoadResolveTexture', 327731)
WGPUFeatureName_DawnPartialLoadResolveTexture = enum_WGPUFeatureName.define('WGPUFeatureName_DawnPartialLoadResolveTexture', 327732)
WGPUFeatureName_MultiDrawIndirect = enum_WGPUFeatureName.define('WGPUFeatureName_MultiDrawIndirect', 327733)
WGPUFeatureName_ClipDistances = enum_WGPUFeatureName.define('WGPUFeatureName_ClipDistances', 327734)
WGPUFeatureName_DawnTexelCopyBufferRowAlignment = enum_WGPUFeatureName.define('WGPUFeatureName_DawnTexelCopyBufferRowAlignment', 327735)
WGPUFeatureName_FlexibleTextureViews = enum_WGPUFeatureName.define('WGPUFeatureName_FlexibleTextureViews', 327736)
WGPUFeatureName_Force32 = enum_WGPUFeatureName.define('WGPUFeatureName_Force32', 2147483647)

WGPUFeatureName = enum_WGPUFeatureName
struct_WGPUSupportedFeatures._fields_ = [
  ('featureCount', size_t),
  ('features', ctypes.POINTER(WGPUFeatureName)),
]
class struct_WGPUSurfaceCapabilities(ctypes.Structure): pass
enum_WGPUPresentMode = CEnum(ctypes.c_uint)
WGPUPresentMode_Fifo = enum_WGPUPresentMode.define('WGPUPresentMode_Fifo', 1)
WGPUPresentMode_FifoRelaxed = enum_WGPUPresentMode.define('WGPUPresentMode_FifoRelaxed', 2)
WGPUPresentMode_Immediate = enum_WGPUPresentMode.define('WGPUPresentMode_Immediate', 3)
WGPUPresentMode_Mailbox = enum_WGPUPresentMode.define('WGPUPresentMode_Mailbox', 4)
WGPUPresentMode_Force32 = enum_WGPUPresentMode.define('WGPUPresentMode_Force32', 2147483647)

WGPUPresentMode = enum_WGPUPresentMode
enum_WGPUCompositeAlphaMode = CEnum(ctypes.c_uint)
WGPUCompositeAlphaMode_Auto = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Auto', 0)
WGPUCompositeAlphaMode_Opaque = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Opaque', 1)
WGPUCompositeAlphaMode_Premultiplied = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Premultiplied', 2)
WGPUCompositeAlphaMode_Unpremultiplied = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Unpremultiplied', 3)
WGPUCompositeAlphaMode_Inherit = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Inherit', 4)
WGPUCompositeAlphaMode_Force32 = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Force32', 2147483647)

WGPUCompositeAlphaMode = enum_WGPUCompositeAlphaMode
struct_WGPUSurfaceCapabilities._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
  ('usages', WGPUTextureUsage),
  ('formatCount', size_t),
  ('formats', ctypes.POINTER(WGPUTextureFormat)),
  ('presentModeCount', size_t),
  ('presentModes', ctypes.POINTER(WGPUPresentMode)),
  ('alphaModeCount', size_t),
  ('alphaModes', ctypes.POINTER(WGPUCompositeAlphaMode)),
]
class struct_WGPUSurfaceConfiguration(ctypes.Structure): pass
struct_WGPUSurfaceConfiguration._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('device', WGPUDevice),
  ('format', WGPUTextureFormat),
  ('usage', WGPUTextureUsage),
  ('viewFormatCount', size_t),
  ('viewFormats', ctypes.POINTER(WGPUTextureFormat)),
  ('alphaMode', WGPUCompositeAlphaMode),
  ('width', uint32_t),
  ('height', uint32_t),
  ('presentMode', WGPUPresentMode),
]
class struct_WGPUSurfaceDescriptorFromWindowsCoreWindow(ctypes.Structure): pass
struct_WGPUSurfaceDescriptorFromWindowsCoreWindow._fields_ = [
  ('chain', WGPUChainedStruct),
  ('coreWindow', ctypes.c_void_p),
]
class struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel(ctypes.Structure): pass
struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel._fields_ = [
  ('chain', WGPUChainedStruct),
  ('swapChainPanel', ctypes.c_void_p),
]
class struct_WGPUSurfaceSourceXCBWindow(ctypes.Structure): pass
struct_WGPUSurfaceSourceXCBWindow._fields_ = [
  ('chain', WGPUChainedStruct),
  ('connection', ctypes.c_void_p),
  ('window', uint32_t),
]
class struct_WGPUSurfaceSourceAndroidNativeWindow(ctypes.Structure): pass
struct_WGPUSurfaceSourceAndroidNativeWindow._fields_ = [
  ('chain', WGPUChainedStruct),
  ('window', ctypes.c_void_p),
]
class struct_WGPUSurfaceSourceMetalLayer(ctypes.Structure): pass
struct_WGPUSurfaceSourceMetalLayer._fields_ = [
  ('chain', WGPUChainedStruct),
  ('layer', ctypes.c_void_p),
]
class struct_WGPUSurfaceSourceWaylandSurface(ctypes.Structure): pass
struct_WGPUSurfaceSourceWaylandSurface._fields_ = [
  ('chain', WGPUChainedStruct),
  ('display', ctypes.c_void_p),
  ('surface', ctypes.c_void_p),
]
class struct_WGPUSurfaceSourceWindowsHWND(ctypes.Structure): pass
struct_WGPUSurfaceSourceWindowsHWND._fields_ = [
  ('chain', WGPUChainedStruct),
  ('hinstance', ctypes.c_void_p),
  ('hwnd', ctypes.c_void_p),
]
class struct_WGPUSurfaceSourceXlibWindow(ctypes.Structure): pass
struct_WGPUSurfaceSourceXlibWindow._fields_ = [
  ('chain', WGPUChainedStruct),
  ('display', ctypes.c_void_p),
  ('window', uint64_t),
]
class struct_WGPUSurfaceTexture(ctypes.Structure): pass
enum_WGPUSurfaceGetCurrentTextureStatus = CEnum(ctypes.c_uint)
WGPUSurfaceGetCurrentTextureStatus_Success = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Success', 1)
WGPUSurfaceGetCurrentTextureStatus_Timeout = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Timeout', 2)
WGPUSurfaceGetCurrentTextureStatus_Outdated = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Outdated', 3)
WGPUSurfaceGetCurrentTextureStatus_Lost = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Lost', 4)
WGPUSurfaceGetCurrentTextureStatus_OutOfMemory = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_OutOfMemory', 5)
WGPUSurfaceGetCurrentTextureStatus_DeviceLost = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_DeviceLost', 6)
WGPUSurfaceGetCurrentTextureStatus_Error = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Error', 7)
WGPUSurfaceGetCurrentTextureStatus_Force32 = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Force32', 2147483647)

WGPUSurfaceGetCurrentTextureStatus = enum_WGPUSurfaceGetCurrentTextureStatus
struct_WGPUSurfaceTexture._fields_ = [
  ('texture', WGPUTexture),
  ('suboptimal', WGPUBool),
  ('status', WGPUSurfaceGetCurrentTextureStatus),
]
class struct_WGPUTextureBindingLayout(ctypes.Structure): pass
enum_WGPUTextureSampleType = CEnum(ctypes.c_uint)
WGPUTextureSampleType_BindingNotUsed = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_BindingNotUsed', 0)
WGPUTextureSampleType_Float = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Float', 1)
WGPUTextureSampleType_UnfilterableFloat = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_UnfilterableFloat', 2)
WGPUTextureSampleType_Depth = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Depth', 3)
WGPUTextureSampleType_Sint = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Sint', 4)
WGPUTextureSampleType_Uint = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Uint', 5)
WGPUTextureSampleType_Force32 = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Force32', 2147483647)

WGPUTextureSampleType = enum_WGPUTextureSampleType
struct_WGPUTextureBindingLayout._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('sampleType', WGPUTextureSampleType),
  ('viewDimension', WGPUTextureViewDimension),
  ('multisampled', WGPUBool),
]
class struct_WGPUTextureBindingViewDimensionDescriptor(ctypes.Structure): pass
struct_WGPUTextureBindingViewDimensionDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('textureBindingViewDimension', WGPUTextureViewDimension),
]
class struct_WGPUTextureDataLayout(ctypes.Structure): pass
struct_WGPUTextureDataLayout._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('offset', uint64_t),
  ('bytesPerRow', uint32_t),
  ('rowsPerImage', uint32_t),
]
class struct_WGPUUncapturedErrorCallbackInfo(ctypes.Structure): pass
struct_WGPUUncapturedErrorCallbackInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('callback', WGPUErrorCallback),
  ('userdata', ctypes.c_void_p),
]
class struct_WGPUVertexAttribute(ctypes.Structure): pass
enum_WGPUVertexFormat = CEnum(ctypes.c_uint)
WGPUVertexFormat_Uint8 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint8', 1)
WGPUVertexFormat_Uint8x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint8x2', 2)
WGPUVertexFormat_Uint8x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint8x4', 3)
WGPUVertexFormat_Sint8 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint8', 4)
WGPUVertexFormat_Sint8x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint8x2', 5)
WGPUVertexFormat_Sint8x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint8x4', 6)
WGPUVertexFormat_Unorm8 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm8', 7)
WGPUVertexFormat_Unorm8x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm8x2', 8)
WGPUVertexFormat_Unorm8x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm8x4', 9)
WGPUVertexFormat_Snorm8 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm8', 10)
WGPUVertexFormat_Snorm8x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm8x2', 11)
WGPUVertexFormat_Snorm8x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm8x4', 12)
WGPUVertexFormat_Uint16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint16', 13)
WGPUVertexFormat_Uint16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint16x2', 14)
WGPUVertexFormat_Uint16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint16x4', 15)
WGPUVertexFormat_Sint16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint16', 16)
WGPUVertexFormat_Sint16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint16x2', 17)
WGPUVertexFormat_Sint16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint16x4', 18)
WGPUVertexFormat_Unorm16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm16', 19)
WGPUVertexFormat_Unorm16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm16x2', 20)
WGPUVertexFormat_Unorm16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm16x4', 21)
WGPUVertexFormat_Snorm16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm16', 22)
WGPUVertexFormat_Snorm16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm16x2', 23)
WGPUVertexFormat_Snorm16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm16x4', 24)
WGPUVertexFormat_Float16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float16', 25)
WGPUVertexFormat_Float16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float16x2', 26)
WGPUVertexFormat_Float16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float16x4', 27)
WGPUVertexFormat_Float32 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float32', 28)
WGPUVertexFormat_Float32x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float32x2', 29)
WGPUVertexFormat_Float32x3 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float32x3', 30)
WGPUVertexFormat_Float32x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float32x4', 31)
WGPUVertexFormat_Uint32 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint32', 32)
WGPUVertexFormat_Uint32x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint32x2', 33)
WGPUVertexFormat_Uint32x3 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint32x3', 34)
WGPUVertexFormat_Uint32x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint32x4', 35)
WGPUVertexFormat_Sint32 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint32', 36)
WGPUVertexFormat_Sint32x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint32x2', 37)
WGPUVertexFormat_Sint32x3 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint32x3', 38)
WGPUVertexFormat_Sint32x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint32x4', 39)
WGPUVertexFormat_Unorm10_10_10_2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm10_10_10_2', 40)
WGPUVertexFormat_Unorm8x4BGRA = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm8x4BGRA', 41)
WGPUVertexFormat_Force32 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Force32', 2147483647)

WGPUVertexFormat = enum_WGPUVertexFormat
struct_WGPUVertexAttribute._fields_ = [
  ('format', WGPUVertexFormat),
  ('offset', uint64_t),
  ('shaderLocation', uint32_t),
]
class struct_WGPUYCbCrVkDescriptor(ctypes.Structure): pass
enum_WGPUFilterMode = CEnum(ctypes.c_uint)
WGPUFilterMode_Undefined = enum_WGPUFilterMode.define('WGPUFilterMode_Undefined', 0)
WGPUFilterMode_Nearest = enum_WGPUFilterMode.define('WGPUFilterMode_Nearest', 1)
WGPUFilterMode_Linear = enum_WGPUFilterMode.define('WGPUFilterMode_Linear', 2)
WGPUFilterMode_Force32 = enum_WGPUFilterMode.define('WGPUFilterMode_Force32', 2147483647)

WGPUFilterMode = enum_WGPUFilterMode
struct_WGPUYCbCrVkDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('vkFormat', uint32_t),
  ('vkYCbCrModel', uint32_t),
  ('vkYCbCrRange', uint32_t),
  ('vkComponentSwizzleRed', uint32_t),
  ('vkComponentSwizzleGreen', uint32_t),
  ('vkComponentSwizzleBlue', uint32_t),
  ('vkComponentSwizzleAlpha', uint32_t),
  ('vkXChromaOffset', uint32_t),
  ('vkYChromaOffset', uint32_t),
  ('vkChromaFilter', WGPUFilterMode),
  ('forceExplicitReconstruction', WGPUBool),
  ('externalFormat', uint64_t),
]
class struct_WGPUAHardwareBufferProperties(ctypes.Structure): pass
WGPUYCbCrVkDescriptor = struct_WGPUYCbCrVkDescriptor
struct_WGPUAHardwareBufferProperties._fields_ = [
  ('yCbCrInfo', WGPUYCbCrVkDescriptor),
]
class struct_WGPUAdapterInfo(ctypes.Structure): pass
enum_WGPUAdapterType = CEnum(ctypes.c_uint)
WGPUAdapterType_DiscreteGPU = enum_WGPUAdapterType.define('WGPUAdapterType_DiscreteGPU', 1)
WGPUAdapterType_IntegratedGPU = enum_WGPUAdapterType.define('WGPUAdapterType_IntegratedGPU', 2)
WGPUAdapterType_CPU = enum_WGPUAdapterType.define('WGPUAdapterType_CPU', 3)
WGPUAdapterType_Unknown = enum_WGPUAdapterType.define('WGPUAdapterType_Unknown', 4)
WGPUAdapterType_Force32 = enum_WGPUAdapterType.define('WGPUAdapterType_Force32', 2147483647)

WGPUAdapterType = enum_WGPUAdapterType
struct_WGPUAdapterInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
  ('vendor', WGPUStringView),
  ('architecture', WGPUStringView),
  ('device', WGPUStringView),
  ('description', WGPUStringView),
  ('backendType', WGPUBackendType),
  ('adapterType', WGPUAdapterType),
  ('vendorID', uint32_t),
  ('deviceID', uint32_t),
  ('compatibilityMode', WGPUBool),
]
class struct_WGPUAdapterPropertiesMemoryHeaps(ctypes.Structure): pass
WGPUMemoryHeapInfo = struct_WGPUMemoryHeapInfo
struct_WGPUAdapterPropertiesMemoryHeaps._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('heapCount', size_t),
  ('heapInfo', ctypes.POINTER(WGPUMemoryHeapInfo)),
]
class struct_WGPUBindGroupDescriptor(ctypes.Structure): pass
WGPUBindGroupEntry = struct_WGPUBindGroupEntry
struct_WGPUBindGroupDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('layout', WGPUBindGroupLayout),
  ('entryCount', size_t),
  ('entries', ctypes.POINTER(WGPUBindGroupEntry)),
]
class struct_WGPUBindGroupLayoutEntry(ctypes.Structure): pass
WGPUShaderStage = ctypes.c_ulong
WGPUBufferBindingLayout = struct_WGPUBufferBindingLayout
WGPUSamplerBindingLayout = struct_WGPUSamplerBindingLayout
WGPUTextureBindingLayout = struct_WGPUTextureBindingLayout
WGPUStorageTextureBindingLayout = struct_WGPUStorageTextureBindingLayout
struct_WGPUBindGroupLayoutEntry._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('binding', uint32_t),
  ('visibility', WGPUShaderStage),
  ('buffer', WGPUBufferBindingLayout),
  ('sampler', WGPUSamplerBindingLayout),
  ('texture', WGPUTextureBindingLayout),
  ('storageTexture', WGPUStorageTextureBindingLayout),
]
class struct_WGPUBlendState(ctypes.Structure): pass
WGPUBlendComponent = struct_WGPUBlendComponent
struct_WGPUBlendState._fields_ = [
  ('color', WGPUBlendComponent),
  ('alpha', WGPUBlendComponent),
]
class struct_WGPUBufferDescriptor(ctypes.Structure): pass
struct_WGPUBufferDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('usage', WGPUBufferUsage),
  ('size', uint64_t),
  ('mappedAtCreation', WGPUBool),
]
class struct_WGPUCommandBufferDescriptor(ctypes.Structure): pass
struct_WGPUCommandBufferDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPUCommandEncoderDescriptor(ctypes.Structure): pass
struct_WGPUCommandEncoderDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPUComputePassDescriptor(ctypes.Structure): pass
WGPUComputePassTimestampWrites = struct_WGPUComputePassTimestampWrites
struct_WGPUComputePassDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('timestampWrites', ctypes.POINTER(WGPUComputePassTimestampWrites)),
]
class struct_WGPUConstantEntry(ctypes.Structure): pass
struct_WGPUConstantEntry._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('key', WGPUStringView),
  ('value', ctypes.c_double),
]
class struct_WGPUDawnCacheDeviceDescriptor(ctypes.Structure): pass
WGPUDawnLoadCacheDataFunction = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p)
WGPUDawnStoreCacheDataFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p)
struct_WGPUDawnCacheDeviceDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('isolationKey', WGPUStringView),
  ('loadDataFunction', WGPUDawnLoadCacheDataFunction),
  ('storeDataFunction', WGPUDawnStoreCacheDataFunction),
  ('functionUserdata', ctypes.c_void_p),
]
class struct_WGPUDepthStencilState(ctypes.Structure): pass
enum_WGPUOptionalBool = CEnum(ctypes.c_uint)
WGPUOptionalBool_False = enum_WGPUOptionalBool.define('WGPUOptionalBool_False', 0)
WGPUOptionalBool_True = enum_WGPUOptionalBool.define('WGPUOptionalBool_True', 1)
WGPUOptionalBool_Undefined = enum_WGPUOptionalBool.define('WGPUOptionalBool_Undefined', 2)
WGPUOptionalBool_Force32 = enum_WGPUOptionalBool.define('WGPUOptionalBool_Force32', 2147483647)

WGPUOptionalBool = enum_WGPUOptionalBool
WGPUStencilFaceState = struct_WGPUStencilFaceState
struct_WGPUDepthStencilState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('format', WGPUTextureFormat),
  ('depthWriteEnabled', WGPUOptionalBool),
  ('depthCompare', WGPUCompareFunction),
  ('stencilFront', WGPUStencilFaceState),
  ('stencilBack', WGPUStencilFaceState),
  ('stencilReadMask', uint32_t),
  ('stencilWriteMask', uint32_t),
  ('depthBias', int32_t),
  ('depthBiasSlopeScale', ctypes.c_float),
  ('depthBiasClamp', ctypes.c_float),
]
class struct_WGPUDrmFormatCapabilities(ctypes.Structure): pass
WGPUDrmFormatProperties = struct_WGPUDrmFormatProperties
struct_WGPUDrmFormatCapabilities._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('propertiesCount', size_t),
  ('properties', ctypes.POINTER(WGPUDrmFormatProperties)),
]
class struct_WGPUExternalTextureDescriptor(ctypes.Structure): pass
WGPUOrigin2D = struct_WGPUOrigin2D
WGPUExtent2D = struct_WGPUExtent2D
enum_WGPUExternalTextureRotation = CEnum(ctypes.c_uint)
WGPUExternalTextureRotation_Rotate0Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate0Degrees', 1)
WGPUExternalTextureRotation_Rotate90Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate90Degrees', 2)
WGPUExternalTextureRotation_Rotate180Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate180Degrees', 3)
WGPUExternalTextureRotation_Rotate270Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate270Degrees', 4)
WGPUExternalTextureRotation_Force32 = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Force32', 2147483647)

WGPUExternalTextureRotation = enum_WGPUExternalTextureRotation
struct_WGPUExternalTextureDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('plane0', WGPUTextureView),
  ('plane1', WGPUTextureView),
  ('cropOrigin', WGPUOrigin2D),
  ('cropSize', WGPUExtent2D),
  ('apparentSize', WGPUExtent2D),
  ('doYuvToRgbConversionOnly', WGPUBool),
  ('yuvToRgbConversionMatrix', ctypes.POINTER(ctypes.c_float)),
  ('srcTransferFunctionParameters', ctypes.POINTER(ctypes.c_float)),
  ('dstTransferFunctionParameters', ctypes.POINTER(ctypes.c_float)),
  ('gamutConversionMatrix', ctypes.POINTER(ctypes.c_float)),
  ('mirrored', WGPUBool),
  ('rotation', WGPUExternalTextureRotation),
]
class struct_WGPUFutureWaitInfo(ctypes.Structure): pass
WGPUFuture = struct_WGPUFuture
struct_WGPUFutureWaitInfo._fields_ = [
  ('future', WGPUFuture),
  ('completed', WGPUBool),
]
class struct_WGPUImageCopyBuffer(ctypes.Structure): pass
WGPUTextureDataLayout = struct_WGPUTextureDataLayout
struct_WGPUImageCopyBuffer._fields_ = [
  ('layout', WGPUTextureDataLayout),
  ('buffer', WGPUBuffer),
]
class struct_WGPUImageCopyExternalTexture(ctypes.Structure): pass
WGPUOrigin3D = struct_WGPUOrigin3D
struct_WGPUImageCopyExternalTexture._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('externalTexture', WGPUExternalTexture),
  ('origin', WGPUOrigin3D),
  ('naturalSize', WGPUExtent2D),
]
class struct_WGPUImageCopyTexture(ctypes.Structure): pass
enum_WGPUTextureAspect = CEnum(ctypes.c_uint)
WGPUTextureAspect_Undefined = enum_WGPUTextureAspect.define('WGPUTextureAspect_Undefined', 0)
WGPUTextureAspect_All = enum_WGPUTextureAspect.define('WGPUTextureAspect_All', 1)
WGPUTextureAspect_StencilOnly = enum_WGPUTextureAspect.define('WGPUTextureAspect_StencilOnly', 2)
WGPUTextureAspect_DepthOnly = enum_WGPUTextureAspect.define('WGPUTextureAspect_DepthOnly', 3)
WGPUTextureAspect_Plane0Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane0Only', 327680)
WGPUTextureAspect_Plane1Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane1Only', 327681)
WGPUTextureAspect_Plane2Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane2Only', 327682)
WGPUTextureAspect_Force32 = enum_WGPUTextureAspect.define('WGPUTextureAspect_Force32', 2147483647)

WGPUTextureAspect = enum_WGPUTextureAspect
struct_WGPUImageCopyTexture._fields_ = [
  ('texture', WGPUTexture),
  ('mipLevel', uint32_t),
  ('origin', WGPUOrigin3D),
  ('aspect', WGPUTextureAspect),
]
class struct_WGPUInstanceDescriptor(ctypes.Structure): pass
WGPUInstanceFeatures = struct_WGPUInstanceFeatures
struct_WGPUInstanceDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('features', WGPUInstanceFeatures),
]
class struct_WGPUPipelineLayoutDescriptor(ctypes.Structure): pass
struct_WGPUPipelineLayoutDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('bindGroupLayoutCount', size_t),
  ('bindGroupLayouts', ctypes.POINTER(WGPUBindGroupLayout)),
  ('immediateDataRangeByteSize', uint32_t),
]
class struct_WGPUPipelineLayoutPixelLocalStorage(ctypes.Structure): pass
WGPUPipelineLayoutStorageAttachment = struct_WGPUPipelineLayoutStorageAttachment
struct_WGPUPipelineLayoutPixelLocalStorage._fields_ = [
  ('chain', WGPUChainedStruct),
  ('totalPixelLocalStorageSize', uint64_t),
  ('storageAttachmentCount', size_t),
  ('storageAttachments', ctypes.POINTER(WGPUPipelineLayoutStorageAttachment)),
]
class struct_WGPUQuerySetDescriptor(ctypes.Structure): pass
enum_WGPUQueryType = CEnum(ctypes.c_uint)
WGPUQueryType_Occlusion = enum_WGPUQueryType.define('WGPUQueryType_Occlusion', 1)
WGPUQueryType_Timestamp = enum_WGPUQueryType.define('WGPUQueryType_Timestamp', 2)
WGPUQueryType_Force32 = enum_WGPUQueryType.define('WGPUQueryType_Force32', 2147483647)

WGPUQueryType = enum_WGPUQueryType
struct_WGPUQuerySetDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('type', WGPUQueryType),
  ('count', uint32_t),
]
class struct_WGPUQueueDescriptor(ctypes.Structure): pass
struct_WGPUQueueDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPURenderBundleDescriptor(ctypes.Structure): pass
struct_WGPURenderBundleDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPURenderBundleEncoderDescriptor(ctypes.Structure): pass
struct_WGPURenderBundleEncoderDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('colorFormatCount', size_t),
  ('colorFormats', ctypes.POINTER(WGPUTextureFormat)),
  ('depthStencilFormat', WGPUTextureFormat),
  ('sampleCount', uint32_t),
  ('depthReadOnly', WGPUBool),
  ('stencilReadOnly', WGPUBool),
]
class struct_WGPURenderPassColorAttachment(ctypes.Structure): pass
WGPUColor = struct_WGPUColor
struct_WGPURenderPassColorAttachment._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('view', WGPUTextureView),
  ('depthSlice', uint32_t),
  ('resolveTarget', WGPUTextureView),
  ('loadOp', WGPULoadOp),
  ('storeOp', WGPUStoreOp),
  ('clearValue', WGPUColor),
]
class struct_WGPURenderPassStorageAttachment(ctypes.Structure): pass
struct_WGPURenderPassStorageAttachment._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('offset', uint64_t),
  ('storage', WGPUTextureView),
  ('loadOp', WGPULoadOp),
  ('storeOp', WGPUStoreOp),
  ('clearValue', WGPUColor),
]
class struct_WGPURequiredLimits(ctypes.Structure): pass
WGPULimits = struct_WGPULimits
struct_WGPURequiredLimits._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('limits', WGPULimits),
]
class struct_WGPUSamplerDescriptor(ctypes.Structure): pass
enum_WGPUAddressMode = CEnum(ctypes.c_uint)
WGPUAddressMode_Undefined = enum_WGPUAddressMode.define('WGPUAddressMode_Undefined', 0)
WGPUAddressMode_ClampToEdge = enum_WGPUAddressMode.define('WGPUAddressMode_ClampToEdge', 1)
WGPUAddressMode_Repeat = enum_WGPUAddressMode.define('WGPUAddressMode_Repeat', 2)
WGPUAddressMode_MirrorRepeat = enum_WGPUAddressMode.define('WGPUAddressMode_MirrorRepeat', 3)
WGPUAddressMode_Force32 = enum_WGPUAddressMode.define('WGPUAddressMode_Force32', 2147483647)

WGPUAddressMode = enum_WGPUAddressMode
enum_WGPUMipmapFilterMode = CEnum(ctypes.c_uint)
WGPUMipmapFilterMode_Undefined = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Undefined', 0)
WGPUMipmapFilterMode_Nearest = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Nearest', 1)
WGPUMipmapFilterMode_Linear = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Linear', 2)
WGPUMipmapFilterMode_Force32 = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Force32', 2147483647)

WGPUMipmapFilterMode = enum_WGPUMipmapFilterMode
uint16_t = ctypes.c_ushort
struct_WGPUSamplerDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('addressModeU', WGPUAddressMode),
  ('addressModeV', WGPUAddressMode),
  ('addressModeW', WGPUAddressMode),
  ('magFilter', WGPUFilterMode),
  ('minFilter', WGPUFilterMode),
  ('mipmapFilter', WGPUMipmapFilterMode),
  ('lodMinClamp', ctypes.c_float),
  ('lodMaxClamp', ctypes.c_float),
  ('compare', WGPUCompareFunction),
  ('maxAnisotropy', uint16_t),
]
class struct_WGPUShaderModuleDescriptor(ctypes.Structure): pass
struct_WGPUShaderModuleDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPUShaderSourceWGSL(ctypes.Structure): pass
struct_WGPUShaderSourceWGSL._fields_ = [
  ('chain', WGPUChainedStruct),
  ('code', WGPUStringView),
]
class struct_WGPUSharedBufferMemoryDescriptor(ctypes.Structure): pass
struct_WGPUSharedBufferMemoryDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPUSharedFenceDescriptor(ctypes.Structure): pass
struct_WGPUSharedFenceDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPUSharedTextureMemoryAHardwareBufferProperties(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryAHardwareBufferProperties._fields_ = [
  ('chain', WGPUChainedStructOut),
  ('yCbCrInfo', WGPUYCbCrVkDescriptor),
]
class struct_WGPUSharedTextureMemoryDescriptor(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPUSharedTextureMemoryDmaBufDescriptor(ctypes.Structure): pass
WGPUExtent3D = struct_WGPUExtent3D
WGPUSharedTextureMemoryDmaBufPlane = struct_WGPUSharedTextureMemoryDmaBufPlane
struct_WGPUSharedTextureMemoryDmaBufDescriptor._fields_ = [
  ('chain', WGPUChainedStruct),
  ('size', WGPUExtent3D),
  ('drmFormat', uint32_t),
  ('drmModifier', uint64_t),
  ('planeCount', size_t),
  ('planes', ctypes.POINTER(WGPUSharedTextureMemoryDmaBufPlane)),
]
class struct_WGPUSharedTextureMemoryProperties(ctypes.Structure): pass
struct_WGPUSharedTextureMemoryProperties._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
  ('usage', WGPUTextureUsage),
  ('size', WGPUExtent3D),
  ('format', WGPUTextureFormat),
]
class struct_WGPUSupportedLimits(ctypes.Structure): pass
struct_WGPUSupportedLimits._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStructOut)),
  ('limits', WGPULimits),
]
class struct_WGPUSurfaceDescriptor(ctypes.Structure): pass
struct_WGPUSurfaceDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
class struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten(ctypes.Structure): pass
struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten._fields_ = [
  ('chain', WGPUChainedStruct),
  ('selector', WGPUStringView),
]
class struct_WGPUTextureDescriptor(ctypes.Structure): pass
enum_WGPUTextureDimension = CEnum(ctypes.c_uint)
WGPUTextureDimension_Undefined = enum_WGPUTextureDimension.define('WGPUTextureDimension_Undefined', 0)
WGPUTextureDimension_1D = enum_WGPUTextureDimension.define('WGPUTextureDimension_1D', 1)
WGPUTextureDimension_2D = enum_WGPUTextureDimension.define('WGPUTextureDimension_2D', 2)
WGPUTextureDimension_3D = enum_WGPUTextureDimension.define('WGPUTextureDimension_3D', 3)
WGPUTextureDimension_Force32 = enum_WGPUTextureDimension.define('WGPUTextureDimension_Force32', 2147483647)

WGPUTextureDimension = enum_WGPUTextureDimension
struct_WGPUTextureDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('usage', WGPUTextureUsage),
  ('dimension', WGPUTextureDimension),
  ('size', WGPUExtent3D),
  ('format', WGPUTextureFormat),
  ('mipLevelCount', uint32_t),
  ('sampleCount', uint32_t),
  ('viewFormatCount', size_t),
  ('viewFormats', ctypes.POINTER(WGPUTextureFormat)),
]
class struct_WGPUTextureViewDescriptor(ctypes.Structure): pass
struct_WGPUTextureViewDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('format', WGPUTextureFormat),
  ('dimension', WGPUTextureViewDimension),
  ('baseMipLevel', uint32_t),
  ('mipLevelCount', uint32_t),
  ('baseArrayLayer', uint32_t),
  ('arrayLayerCount', uint32_t),
  ('aspect', WGPUTextureAspect),
  ('usage', WGPUTextureUsage),
]
class struct_WGPUVertexBufferLayout(ctypes.Structure): pass
enum_WGPUVertexStepMode = CEnum(ctypes.c_uint)
WGPUVertexStepMode_Undefined = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Undefined', 0)
WGPUVertexStepMode_Vertex = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Vertex', 1)
WGPUVertexStepMode_Instance = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Instance', 2)
WGPUVertexStepMode_Force32 = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Force32', 2147483647)

WGPUVertexStepMode = enum_WGPUVertexStepMode
WGPUVertexAttribute = struct_WGPUVertexAttribute
struct_WGPUVertexBufferLayout._fields_ = [
  ('arrayStride', uint64_t),
  ('stepMode', WGPUVertexStepMode),
  ('attributeCount', size_t),
  ('attributes', ctypes.POINTER(WGPUVertexAttribute)),
]
class struct_WGPUBindGroupLayoutDescriptor(ctypes.Structure): pass
WGPUBindGroupLayoutEntry = struct_WGPUBindGroupLayoutEntry
struct_WGPUBindGroupLayoutDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('entryCount', size_t),
  ('entries', ctypes.POINTER(WGPUBindGroupLayoutEntry)),
]
class struct_WGPUColorTargetState(ctypes.Structure): pass
WGPUBlendState = struct_WGPUBlendState
WGPUColorWriteMask = ctypes.c_ulong
struct_WGPUColorTargetState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('format', WGPUTextureFormat),
  ('blend', ctypes.POINTER(WGPUBlendState)),
  ('writeMask', WGPUColorWriteMask),
]
class struct_WGPUCompilationInfo(ctypes.Structure): pass
struct_WGPUCompilationInfo._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('messageCount', size_t),
  ('messages', ctypes.POINTER(WGPUCompilationMessage)),
]
class struct_WGPUComputeState(ctypes.Structure): pass
WGPUConstantEntry = struct_WGPUConstantEntry
struct_WGPUComputeState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('module', WGPUShaderModule),
  ('entryPoint', WGPUStringView),
  ('constantCount', size_t),
  ('constants', ctypes.POINTER(WGPUConstantEntry)),
]
class struct_WGPUDeviceDescriptor(ctypes.Structure): pass
WGPURequiredLimits = struct_WGPURequiredLimits
WGPUQueueDescriptor = struct_WGPUQueueDescriptor
class struct_WGPUDeviceLostCallbackInfo2(ctypes.Structure): pass
WGPUDeviceLostCallback2 = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.POINTER(struct_WGPUDeviceImpl)), enum_WGPUDeviceLostReason, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
struct_WGPUDeviceLostCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUDeviceLostCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPUDeviceLostCallbackInfo2 = struct_WGPUDeviceLostCallbackInfo2
class struct_WGPUUncapturedErrorCallbackInfo2(ctypes.Structure): pass
WGPUUncapturedErrorCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.POINTER(struct_WGPUDeviceImpl)), enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
struct_WGPUUncapturedErrorCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('callback', WGPUUncapturedErrorCallback),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPUUncapturedErrorCallbackInfo2 = struct_WGPUUncapturedErrorCallbackInfo2
struct_WGPUDeviceDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('requiredFeatureCount', size_t),
  ('requiredFeatures', ctypes.POINTER(WGPUFeatureName)),
  ('requiredLimits', ctypes.POINTER(WGPURequiredLimits)),
  ('defaultQueue', WGPUQueueDescriptor),
  ('deviceLostCallbackInfo2', WGPUDeviceLostCallbackInfo2),
  ('uncapturedErrorCallbackInfo2', WGPUUncapturedErrorCallbackInfo2),
]
class struct_WGPURenderPassDescriptor(ctypes.Structure): pass
WGPURenderPassColorAttachment = struct_WGPURenderPassColorAttachment
WGPURenderPassDepthStencilAttachment = struct_WGPURenderPassDepthStencilAttachment
WGPURenderPassTimestampWrites = struct_WGPURenderPassTimestampWrites
struct_WGPURenderPassDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('colorAttachmentCount', size_t),
  ('colorAttachments', ctypes.POINTER(WGPURenderPassColorAttachment)),
  ('depthStencilAttachment', ctypes.POINTER(WGPURenderPassDepthStencilAttachment)),
  ('occlusionQuerySet', WGPUQuerySet),
  ('timestampWrites', ctypes.POINTER(WGPURenderPassTimestampWrites)),
]
class struct_WGPURenderPassPixelLocalStorage(ctypes.Structure): pass
WGPURenderPassStorageAttachment = struct_WGPURenderPassStorageAttachment
struct_WGPURenderPassPixelLocalStorage._fields_ = [
  ('chain', WGPUChainedStruct),
  ('totalPixelLocalStorageSize', uint64_t),
  ('storageAttachmentCount', size_t),
  ('storageAttachments', ctypes.POINTER(WGPURenderPassStorageAttachment)),
]
class struct_WGPUVertexState(ctypes.Structure): pass
WGPUVertexBufferLayout = struct_WGPUVertexBufferLayout
struct_WGPUVertexState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('module', WGPUShaderModule),
  ('entryPoint', WGPUStringView),
  ('constantCount', size_t),
  ('constants', ctypes.POINTER(WGPUConstantEntry)),
  ('bufferCount', size_t),
  ('buffers', ctypes.POINTER(WGPUVertexBufferLayout)),
]
class struct_WGPUComputePipelineDescriptor(ctypes.Structure): pass
WGPUComputeState = struct_WGPUComputeState
struct_WGPUComputePipelineDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('layout', WGPUPipelineLayout),
  ('compute', WGPUComputeState),
]
class struct_WGPUFragmentState(ctypes.Structure): pass
WGPUColorTargetState = struct_WGPUColorTargetState
struct_WGPUFragmentState._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('module', WGPUShaderModule),
  ('entryPoint', WGPUStringView),
  ('constantCount', size_t),
  ('constants', ctypes.POINTER(WGPUConstantEntry)),
  ('targetCount', size_t),
  ('targets', ctypes.POINTER(WGPUColorTargetState)),
]
class struct_WGPURenderPipelineDescriptor(ctypes.Structure): pass
WGPUVertexState = struct_WGPUVertexState
WGPUPrimitiveState = struct_WGPUPrimitiveState
WGPUDepthStencilState = struct_WGPUDepthStencilState
WGPUMultisampleState = struct_WGPUMultisampleState
WGPUFragmentState = struct_WGPUFragmentState
struct_WGPURenderPipelineDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('layout', WGPUPipelineLayout),
  ('vertex', WGPUVertexState),
  ('primitive', WGPUPrimitiveState),
  ('depthStencil', ctypes.POINTER(WGPUDepthStencilState)),
  ('multisample', WGPUMultisampleState),
  ('fragment', ctypes.POINTER(WGPUFragmentState)),
]
enum_WGPUWGSLFeatureName = CEnum(ctypes.c_uint)
WGPUWGSLFeatureName_ReadonlyAndReadwriteStorageTextures = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ReadonlyAndReadwriteStorageTextures', 1)
WGPUWGSLFeatureName_Packed4x8IntegerDotProduct = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_Packed4x8IntegerDotProduct', 2)
WGPUWGSLFeatureName_UnrestrictedPointerParameters = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_UnrestrictedPointerParameters', 3)
WGPUWGSLFeatureName_PointerCompositeAccess = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_PointerCompositeAccess', 4)
WGPUWGSLFeatureName_ChromiumTestingUnimplemented = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingUnimplemented', 327680)
WGPUWGSLFeatureName_ChromiumTestingUnsafeExperimental = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingUnsafeExperimental', 327681)
WGPUWGSLFeatureName_ChromiumTestingExperimental = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingExperimental', 327682)
WGPUWGSLFeatureName_ChromiumTestingShippedWithKillswitch = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingShippedWithKillswitch', 327683)
WGPUWGSLFeatureName_ChromiumTestingShipped = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingShipped', 327684)
WGPUWGSLFeatureName_Force32 = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_Force32', 2147483647)

WGPUWGSLFeatureName = enum_WGPUWGSLFeatureName
WGPUBufferMapAsyncStatus = enum_WGPUBufferMapAsyncStatus
enum_WGPUBufferMapState = CEnum(ctypes.c_uint)
WGPUBufferMapState_Unmapped = enum_WGPUBufferMapState.define('WGPUBufferMapState_Unmapped', 1)
WGPUBufferMapState_Pending = enum_WGPUBufferMapState.define('WGPUBufferMapState_Pending', 2)
WGPUBufferMapState_Mapped = enum_WGPUBufferMapState.define('WGPUBufferMapState_Mapped', 3)
WGPUBufferMapState_Force32 = enum_WGPUBufferMapState.define('WGPUBufferMapState_Force32', 2147483647)

WGPUBufferMapState = enum_WGPUBufferMapState
WGPUCompilationInfoRequestStatus = enum_WGPUCompilationInfoRequestStatus
WGPUCreatePipelineAsyncStatus = enum_WGPUCreatePipelineAsyncStatus
WGPUDeviceLostReason = enum_WGPUDeviceLostReason
enum_WGPUErrorFilter = CEnum(ctypes.c_uint)
WGPUErrorFilter_Validation = enum_WGPUErrorFilter.define('WGPUErrorFilter_Validation', 1)
WGPUErrorFilter_OutOfMemory = enum_WGPUErrorFilter.define('WGPUErrorFilter_OutOfMemory', 2)
WGPUErrorFilter_Internal = enum_WGPUErrorFilter.define('WGPUErrorFilter_Internal', 3)
WGPUErrorFilter_Force32 = enum_WGPUErrorFilter.define('WGPUErrorFilter_Force32', 2147483647)

WGPUErrorFilter = enum_WGPUErrorFilter
WGPUErrorType = enum_WGPUErrorType
enum_WGPULoggingType = CEnum(ctypes.c_uint)
WGPULoggingType_Verbose = enum_WGPULoggingType.define('WGPULoggingType_Verbose', 1)
WGPULoggingType_Info = enum_WGPULoggingType.define('WGPULoggingType_Info', 2)
WGPULoggingType_Warning = enum_WGPULoggingType.define('WGPULoggingType_Warning', 3)
WGPULoggingType_Error = enum_WGPULoggingType.define('WGPULoggingType_Error', 4)
WGPULoggingType_Force32 = enum_WGPULoggingType.define('WGPULoggingType_Force32', 2147483647)

WGPULoggingType = enum_WGPULoggingType
enum_WGPUMapAsyncStatus = CEnum(ctypes.c_uint)
WGPUMapAsyncStatus_Success = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Success', 1)
WGPUMapAsyncStatus_InstanceDropped = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_InstanceDropped', 2)
WGPUMapAsyncStatus_Error = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Error', 3)
WGPUMapAsyncStatus_Aborted = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Aborted', 4)
WGPUMapAsyncStatus_Unknown = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Unknown', 5)
WGPUMapAsyncStatus_Force32 = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Force32', 2147483647)

WGPUMapAsyncStatus = enum_WGPUMapAsyncStatus
WGPUPopErrorScopeStatus = enum_WGPUPopErrorScopeStatus
WGPUQueueWorkDoneStatus = enum_WGPUQueueWorkDoneStatus
WGPURequestAdapterStatus = enum_WGPURequestAdapterStatus
WGPURequestDeviceStatus = enum_WGPURequestDeviceStatus
enum_WGPUStatus = CEnum(ctypes.c_uint)
WGPUStatus_Success = enum_WGPUStatus.define('WGPUStatus_Success', 1)
WGPUStatus_Error = enum_WGPUStatus.define('WGPUStatus_Error', 2)
WGPUStatus_Force32 = enum_WGPUStatus.define('WGPUStatus_Force32', 2147483647)

WGPUStatus = enum_WGPUStatus
enum_WGPUWaitStatus = CEnum(ctypes.c_uint)
WGPUWaitStatus_Success = enum_WGPUWaitStatus.define('WGPUWaitStatus_Success', 1)
WGPUWaitStatus_TimedOut = enum_WGPUWaitStatus.define('WGPUWaitStatus_TimedOut', 2)
WGPUWaitStatus_UnsupportedTimeout = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedTimeout', 3)
WGPUWaitStatus_UnsupportedCount = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedCount', 4)
WGPUWaitStatus_UnsupportedMixedSources = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedMixedSources', 5)
WGPUWaitStatus_Unknown = enum_WGPUWaitStatus.define('WGPUWaitStatus_Unknown', 6)
WGPUWaitStatus_Force32 = enum_WGPUWaitStatus.define('WGPUWaitStatus_Force32', 2147483647)

WGPUWaitStatus = enum_WGPUWaitStatus
WGPUMapMode = ctypes.c_ulong
WGPUDeviceLostCallback = ctypes.CFUNCTYPE(None, enum_WGPUDeviceLostReason, struct_WGPUStringView, ctypes.c_void_p)
WGPULoggingCallback = ctypes.CFUNCTYPE(None, enum_WGPULoggingType, struct_WGPUStringView, ctypes.c_void_p)
WGPUProc = ctypes.CFUNCTYPE(None, )
WGPUBufferMapCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUMapAsyncStatus, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPUCompilationInfoCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, ctypes.POINTER(const_struct_WGPUCompilationInfo), ctypes.c_void_p, ctypes.c_void_p)
WGPUCreateComputePipelineAsyncCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, ctypes.POINTER(struct_WGPUComputePipelineImpl), struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPUCreateRenderPipelineAsyncCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, ctypes.POINTER(struct_WGPURenderPipelineImpl), struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPUPopErrorScopeCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUPopErrorScopeStatus, enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPUQueueWorkDoneCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, ctypes.c_void_p, ctypes.c_void_p)
WGPURequestAdapterCallback2 = ctypes.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, ctypes.POINTER(struct_WGPUAdapterImpl), struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPURequestDeviceCallback2 = ctypes.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, ctypes.POINTER(struct_WGPUDeviceImpl), struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
class struct_WGPUBufferMapCallbackInfo2(ctypes.Structure): pass
struct_WGPUBufferMapCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUBufferMapCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPUBufferMapCallbackInfo2 = struct_WGPUBufferMapCallbackInfo2
class struct_WGPUCompilationInfoCallbackInfo2(ctypes.Structure): pass
struct_WGPUCompilationInfoCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUCompilationInfoCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPUCompilationInfoCallbackInfo2 = struct_WGPUCompilationInfoCallbackInfo2
class struct_WGPUCreateComputePipelineAsyncCallbackInfo2(ctypes.Structure): pass
struct_WGPUCreateComputePipelineAsyncCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUCreateComputePipelineAsyncCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPUCreateComputePipelineAsyncCallbackInfo2 = struct_WGPUCreateComputePipelineAsyncCallbackInfo2
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo2(ctypes.Structure): pass
struct_WGPUCreateRenderPipelineAsyncCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUCreateRenderPipelineAsyncCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPUCreateRenderPipelineAsyncCallbackInfo2 = struct_WGPUCreateRenderPipelineAsyncCallbackInfo2
class struct_WGPUPopErrorScopeCallbackInfo2(ctypes.Structure): pass
struct_WGPUPopErrorScopeCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUPopErrorScopeCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPUPopErrorScopeCallbackInfo2 = struct_WGPUPopErrorScopeCallbackInfo2
class struct_WGPUQueueWorkDoneCallbackInfo2(ctypes.Structure): pass
struct_WGPUQueueWorkDoneCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPUQueueWorkDoneCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPUQueueWorkDoneCallbackInfo2 = struct_WGPUQueueWorkDoneCallbackInfo2
class struct_WGPURequestAdapterCallbackInfo2(ctypes.Structure): pass
struct_WGPURequestAdapterCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPURequestAdapterCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPURequestAdapterCallbackInfo2 = struct_WGPURequestAdapterCallbackInfo2
class struct_WGPURequestDeviceCallbackInfo2(ctypes.Structure): pass
struct_WGPURequestDeviceCallbackInfo2._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('mode', WGPUCallbackMode),
  ('callback', WGPURequestDeviceCallback2),
  ('userdata1', ctypes.c_void_p),
  ('userdata2', ctypes.c_void_p),
]
WGPURequestDeviceCallbackInfo2 = struct_WGPURequestDeviceCallbackInfo2
WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER = struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER
WGPUAdapterPropertiesD3D = struct_WGPUAdapterPropertiesD3D
WGPUAdapterPropertiesSubgroups = struct_WGPUAdapterPropertiesSubgroups
WGPUAdapterPropertiesVk = struct_WGPUAdapterPropertiesVk
WGPUBufferHostMappedPointer = struct_WGPUBufferHostMappedPointer
WGPUBufferMapCallbackInfo = struct_WGPUBufferMapCallbackInfo
WGPUColorTargetStateExpandResolveTextureDawn = struct_WGPUColorTargetStateExpandResolveTextureDawn
WGPUCompilationInfoCallbackInfo = struct_WGPUCompilationInfoCallbackInfo
WGPUCopyTextureForBrowserOptions = struct_WGPUCopyTextureForBrowserOptions
WGPUCreateComputePipelineAsyncCallbackInfo = struct_WGPUCreateComputePipelineAsyncCallbackInfo
WGPUCreateRenderPipelineAsyncCallbackInfo = struct_WGPUCreateRenderPipelineAsyncCallbackInfo
WGPUDawnWGSLBlocklist = struct_WGPUDawnWGSLBlocklist
WGPUDawnAdapterPropertiesPowerPreference = struct_WGPUDawnAdapterPropertiesPowerPreference
WGPUDawnBufferDescriptorErrorInfoFromWireClient = struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient
WGPUDawnEncoderInternalUsageDescriptor = struct_WGPUDawnEncoderInternalUsageDescriptor
WGPUDawnExperimentalImmediateDataLimits = struct_WGPUDawnExperimentalImmediateDataLimits
WGPUDawnExperimentalSubgroupLimits = struct_WGPUDawnExperimentalSubgroupLimits
WGPUDawnRenderPassColorAttachmentRenderToSingleSampled = struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled
WGPUDawnShaderModuleSPIRVOptionsDescriptor = struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor
WGPUDawnTexelCopyBufferRowAlignmentLimits = struct_WGPUDawnTexelCopyBufferRowAlignmentLimits
WGPUDawnTextureInternalUsageDescriptor = struct_WGPUDawnTextureInternalUsageDescriptor
WGPUDawnTogglesDescriptor = struct_WGPUDawnTogglesDescriptor
WGPUDawnWireWGSLControl = struct_WGPUDawnWireWGSLControl
WGPUDeviceLostCallbackInfo = struct_WGPUDeviceLostCallbackInfo
WGPUExternalTextureBindingEntry = struct_WGPUExternalTextureBindingEntry
WGPUExternalTextureBindingLayout = struct_WGPUExternalTextureBindingLayout
WGPUFormatCapabilities = struct_WGPUFormatCapabilities
WGPUPopErrorScopeCallbackInfo = struct_WGPUPopErrorScopeCallbackInfo
WGPUQueueWorkDoneCallbackInfo = struct_WGPUQueueWorkDoneCallbackInfo
WGPURenderPassDescriptorExpandResolveRect = struct_WGPURenderPassDescriptorExpandResolveRect
WGPURenderPassMaxDrawCount = struct_WGPURenderPassMaxDrawCount
WGPURequestAdapterCallbackInfo = struct_WGPURequestAdapterCallbackInfo
WGPURequestAdapterOptions = struct_WGPURequestAdapterOptions
WGPURequestDeviceCallbackInfo = struct_WGPURequestDeviceCallbackInfo
WGPUShaderModuleCompilationOptions = struct_WGPUShaderModuleCompilationOptions
WGPUShaderSourceSPIRV = struct_WGPUShaderSourceSPIRV
WGPUSharedBufferMemoryBeginAccessDescriptor = struct_WGPUSharedBufferMemoryBeginAccessDescriptor
WGPUSharedBufferMemoryEndAccessState = struct_WGPUSharedBufferMemoryEndAccessState
WGPUSharedBufferMemoryProperties = struct_WGPUSharedBufferMemoryProperties
WGPUSharedFenceDXGISharedHandleDescriptor = struct_WGPUSharedFenceDXGISharedHandleDescriptor
WGPUSharedFenceDXGISharedHandleExportInfo = struct_WGPUSharedFenceDXGISharedHandleExportInfo
WGPUSharedFenceMTLSharedEventDescriptor = struct_WGPUSharedFenceMTLSharedEventDescriptor
WGPUSharedFenceMTLSharedEventExportInfo = struct_WGPUSharedFenceMTLSharedEventExportInfo
WGPUSharedFenceExportInfo = struct_WGPUSharedFenceExportInfo
WGPUSharedFenceSyncFDDescriptor = struct_WGPUSharedFenceSyncFDDescriptor
WGPUSharedFenceSyncFDExportInfo = struct_WGPUSharedFenceSyncFDExportInfo
WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor = struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor
WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo = struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo
WGPUSharedFenceVkSemaphoreZirconHandleDescriptor = struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor
WGPUSharedFenceVkSemaphoreZirconHandleExportInfo = struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo
WGPUSharedTextureMemoryD3DSwapchainBeginState = struct_WGPUSharedTextureMemoryD3DSwapchainBeginState
WGPUSharedTextureMemoryDXGISharedHandleDescriptor = struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor
WGPUSharedTextureMemoryEGLImageDescriptor = struct_WGPUSharedTextureMemoryEGLImageDescriptor
WGPUSharedTextureMemoryIOSurfaceDescriptor = struct_WGPUSharedTextureMemoryIOSurfaceDescriptor
WGPUSharedTextureMemoryAHardwareBufferDescriptor = struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor
WGPUSharedTextureMemoryBeginAccessDescriptor = struct_WGPUSharedTextureMemoryBeginAccessDescriptor
WGPUSharedTextureMemoryEndAccessState = struct_WGPUSharedTextureMemoryEndAccessState
WGPUSharedTextureMemoryOpaqueFDDescriptor = struct_WGPUSharedTextureMemoryOpaqueFDDescriptor
WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor = struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor
WGPUSharedTextureMemoryVkImageLayoutBeginState = struct_WGPUSharedTextureMemoryVkImageLayoutBeginState
WGPUSharedTextureMemoryVkImageLayoutEndState = struct_WGPUSharedTextureMemoryVkImageLayoutEndState
WGPUSharedTextureMemoryZirconHandleDescriptor = struct_WGPUSharedTextureMemoryZirconHandleDescriptor
WGPUStaticSamplerBindingLayout = struct_WGPUStaticSamplerBindingLayout
WGPUSupportedFeatures = struct_WGPUSupportedFeatures
WGPUSurfaceCapabilities = struct_WGPUSurfaceCapabilities
WGPUSurfaceConfiguration = struct_WGPUSurfaceConfiguration
WGPUSurfaceDescriptorFromWindowsCoreWindow = struct_WGPUSurfaceDescriptorFromWindowsCoreWindow
WGPUSurfaceDescriptorFromWindowsSwapChainPanel = struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel
WGPUSurfaceSourceXCBWindow = struct_WGPUSurfaceSourceXCBWindow
WGPUSurfaceSourceAndroidNativeWindow = struct_WGPUSurfaceSourceAndroidNativeWindow
WGPUSurfaceSourceMetalLayer = struct_WGPUSurfaceSourceMetalLayer
WGPUSurfaceSourceWaylandSurface = struct_WGPUSurfaceSourceWaylandSurface
WGPUSurfaceSourceWindowsHWND = struct_WGPUSurfaceSourceWindowsHWND
WGPUSurfaceSourceXlibWindow = struct_WGPUSurfaceSourceXlibWindow
WGPUSurfaceTexture = struct_WGPUSurfaceTexture
WGPUTextureBindingViewDimensionDescriptor = struct_WGPUTextureBindingViewDimensionDescriptor
WGPUUncapturedErrorCallbackInfo = struct_WGPUUncapturedErrorCallbackInfo
WGPUAHardwareBufferProperties = struct_WGPUAHardwareBufferProperties
WGPUAdapterInfo = struct_WGPUAdapterInfo
WGPUAdapterPropertiesMemoryHeaps = struct_WGPUAdapterPropertiesMemoryHeaps
WGPUBindGroupDescriptor = struct_WGPUBindGroupDescriptor
WGPUBufferDescriptor = struct_WGPUBufferDescriptor
WGPUCommandBufferDescriptor = struct_WGPUCommandBufferDescriptor
WGPUCommandEncoderDescriptor = struct_WGPUCommandEncoderDescriptor
WGPUComputePassDescriptor = struct_WGPUComputePassDescriptor
WGPUDawnCacheDeviceDescriptor = struct_WGPUDawnCacheDeviceDescriptor
WGPUDrmFormatCapabilities = struct_WGPUDrmFormatCapabilities
WGPUExternalTextureDescriptor = struct_WGPUExternalTextureDescriptor
WGPUFutureWaitInfo = struct_WGPUFutureWaitInfo
WGPUImageCopyBuffer = struct_WGPUImageCopyBuffer
WGPUImageCopyExternalTexture = struct_WGPUImageCopyExternalTexture
WGPUImageCopyTexture = struct_WGPUImageCopyTexture
WGPUInstanceDescriptor = struct_WGPUInstanceDescriptor
WGPUPipelineLayoutDescriptor = struct_WGPUPipelineLayoutDescriptor
WGPUPipelineLayoutPixelLocalStorage = struct_WGPUPipelineLayoutPixelLocalStorage
WGPUQuerySetDescriptor = struct_WGPUQuerySetDescriptor
WGPURenderBundleDescriptor = struct_WGPURenderBundleDescriptor
WGPURenderBundleEncoderDescriptor = struct_WGPURenderBundleEncoderDescriptor
WGPUSamplerDescriptor = struct_WGPUSamplerDescriptor
WGPUShaderModuleDescriptor = struct_WGPUShaderModuleDescriptor
WGPUShaderSourceWGSL = struct_WGPUShaderSourceWGSL
WGPUSharedBufferMemoryDescriptor = struct_WGPUSharedBufferMemoryDescriptor
WGPUSharedFenceDescriptor = struct_WGPUSharedFenceDescriptor
WGPUSharedTextureMemoryAHardwareBufferProperties = struct_WGPUSharedTextureMemoryAHardwareBufferProperties
WGPUSharedTextureMemoryDescriptor = struct_WGPUSharedTextureMemoryDescriptor
WGPUSharedTextureMemoryDmaBufDescriptor = struct_WGPUSharedTextureMemoryDmaBufDescriptor
WGPUSharedTextureMemoryProperties = struct_WGPUSharedTextureMemoryProperties
WGPUSupportedLimits = struct_WGPUSupportedLimits
WGPUSurfaceDescriptor = struct_WGPUSurfaceDescriptor
WGPUSurfaceSourceCanvasHTMLSelector_Emscripten = struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten
WGPUTextureDescriptor = struct_WGPUTextureDescriptor
WGPUTextureViewDescriptor = struct_WGPUTextureViewDescriptor
WGPUBindGroupLayoutDescriptor = struct_WGPUBindGroupLayoutDescriptor
WGPUCompilationInfo = struct_WGPUCompilationInfo
WGPUDeviceDescriptor = struct_WGPUDeviceDescriptor
WGPURenderPassDescriptor = struct_WGPURenderPassDescriptor
WGPURenderPassPixelLocalStorage = struct_WGPURenderPassPixelLocalStorage
WGPUComputePipelineDescriptor = struct_WGPUComputePipelineDescriptor
WGPURenderPipelineDescriptor = struct_WGPURenderPipelineDescriptor
WGPURenderPassDescriptorMaxDrawCount = struct_WGPURenderPassMaxDrawCount
WGPUShaderModuleSPIRVDescriptor = struct_WGPUShaderSourceSPIRV
WGPUShaderModuleWGSLDescriptor = struct_WGPUShaderSourceWGSL
WGPUSurfaceDescriptorFromAndroidNativeWindow = struct_WGPUSurfaceSourceAndroidNativeWindow
WGPUSurfaceDescriptorFromCanvasHTMLSelector = struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten
WGPUSurfaceDescriptorFromMetalLayer = struct_WGPUSurfaceSourceMetalLayer
WGPUSurfaceDescriptorFromWaylandSurface = struct_WGPUSurfaceSourceWaylandSurface
WGPUSurfaceDescriptorFromWindowsHWND = struct_WGPUSurfaceSourceWindowsHWND
WGPUSurfaceDescriptorFromXcbWindow = struct_WGPUSurfaceSourceXCBWindow
WGPUSurfaceDescriptorFromXlibWindow = struct_WGPUSurfaceSourceXlibWindow
WGPUProcAdapterInfoFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUAdapterInfo)
WGPUProcAdapterPropertiesMemoryHeapsFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUAdapterPropertiesMemoryHeaps)
class const_struct_WGPUInstanceDescriptor(ctypes.Structure): pass
const_struct_WGPUInstanceDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('features', WGPUInstanceFeatures),
]
WGPUProcCreateInstance = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUInstanceImpl), ctypes.POINTER(const_struct_WGPUInstanceDescriptor))
WGPUProcDrmFormatCapabilitiesFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUDrmFormatCapabilities)
WGPUProcGetInstanceFeatures = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUInstanceFeatures))
WGPUProcGetProcAddress = ctypes.CFUNCTYPE(ctypes.CFUNCTYPE(None, ), struct_WGPUStringView)
WGPUProcSharedBufferMemoryEndAccessStateFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUSharedBufferMemoryEndAccessState)
WGPUProcSharedTextureMemoryEndAccessStateFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUSharedTextureMemoryEndAccessState)
WGPUProcSupportedFeaturesFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUSupportedFeatures)
WGPUProcSurfaceCapabilitiesFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUSurfaceCapabilities)
class const_struct_WGPUDeviceDescriptor(ctypes.Structure): pass
const_struct_WGPUDeviceDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('requiredFeatureCount', size_t),
  ('requiredFeatures', ctypes.POINTER(WGPUFeatureName)),
  ('requiredLimits', ctypes.POINTER(WGPURequiredLimits)),
  ('defaultQueue', WGPUQueueDescriptor),
  ('deviceLostCallbackInfo2', WGPUDeviceLostCallbackInfo2),
  ('uncapturedErrorCallbackInfo2', WGPUUncapturedErrorCallbackInfo2),
]
WGPUProcAdapterCreateDevice = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(const_struct_WGPUDeviceDescriptor))
WGPUProcAdapterGetFeatures = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(struct_WGPUSupportedFeatures))
WGPUProcAdapterGetFormatCapabilities = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUAdapterImpl), enum_WGPUTextureFormat, ctypes.POINTER(struct_WGPUFormatCapabilities))
WGPUProcAdapterGetInfo = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(struct_WGPUAdapterInfo))
WGPUProcAdapterGetInstance = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUInstanceImpl), ctypes.POINTER(struct_WGPUAdapterImpl))
WGPUProcAdapterGetLimits = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(struct_WGPUSupportedLimits))
WGPUProcAdapterHasFeature = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUAdapterImpl), enum_WGPUFeatureName)
WGPUProcAdapterRequestDevice = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(const_struct_WGPUDeviceDescriptor), ctypes.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, ctypes.POINTER(struct_WGPUDeviceImpl), struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcAdapterRequestDevice2 = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(const_struct_WGPUDeviceDescriptor), struct_WGPURequestDeviceCallbackInfo2)
WGPUProcAdapterRequestDeviceF = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(const_struct_WGPUDeviceDescriptor), struct_WGPURequestDeviceCallbackInfo)
WGPUProcAdapterAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUAdapterImpl))
WGPUProcAdapterRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUAdapterImpl))
WGPUProcBindGroupSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBindGroupImpl), struct_WGPUStringView)
WGPUProcBindGroupAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBindGroupImpl))
WGPUProcBindGroupRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBindGroupImpl))
WGPUProcBindGroupLayoutSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBindGroupLayoutImpl), struct_WGPUStringView)
WGPUProcBindGroupLayoutAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBindGroupLayoutImpl))
WGPUProcBindGroupLayoutRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBindGroupLayoutImpl))
WGPUProcBufferDestroy = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBufferImpl))
WGPUProcBufferGetConstMappedRange = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong)
WGPUProcBufferGetMapState = ctypes.CFUNCTYPE(enum_WGPUBufferMapState, ctypes.POINTER(struct_WGPUBufferImpl))
WGPUProcBufferGetMappedRange = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong)
WGPUProcBufferGetSize = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.POINTER(struct_WGPUBufferImpl))
WGPUProcBufferGetUsage = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.POINTER(struct_WGPUBufferImpl))
WGPUProcBufferMapAsync = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.CFUNCTYPE(None, enum_WGPUBufferMapAsyncStatus, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcBufferMapAsync2 = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, struct_WGPUBufferMapCallbackInfo2)
WGPUProcBufferMapAsyncF = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, struct_WGPUBufferMapCallbackInfo)
WGPUProcBufferSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBufferImpl), struct_WGPUStringView)
WGPUProcBufferUnmap = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBufferImpl))
WGPUProcBufferAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBufferImpl))
WGPUProcBufferRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUBufferImpl))
WGPUProcCommandBufferSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandBufferImpl), struct_WGPUStringView)
WGPUProcCommandBufferAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandBufferImpl))
WGPUProcCommandBufferRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandBufferImpl))
class const_struct_WGPUComputePassDescriptor(ctypes.Structure): pass
const_struct_WGPUComputePassDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('timestampWrites', ctypes.POINTER(WGPUComputePassTimestampWrites)),
]
WGPUProcCommandEncoderBeginComputePass = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUComputePassEncoderImpl), ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(const_struct_WGPUComputePassDescriptor))
class const_struct_WGPURenderPassDescriptor(ctypes.Structure): pass
const_struct_WGPURenderPassDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('colorAttachmentCount', size_t),
  ('colorAttachments', ctypes.POINTER(WGPURenderPassColorAttachment)),
  ('depthStencilAttachment', ctypes.POINTER(WGPURenderPassDepthStencilAttachment)),
  ('occlusionQuerySet', WGPUQuerySet),
  ('timestampWrites', ctypes.POINTER(WGPURenderPassTimestampWrites)),
]
WGPUProcCommandEncoderBeginRenderPass = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(const_struct_WGPURenderPassDescriptor))
WGPUProcCommandEncoderClearBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong)
WGPUProcCommandEncoderCopyBufferToBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong)
class const_struct_WGPUImageCopyBuffer(ctypes.Structure): pass
const_struct_WGPUImageCopyBuffer._fields_ = [
  ('layout', WGPUTextureDataLayout),
  ('buffer', WGPUBuffer),
]
class const_struct_WGPUImageCopyTexture(ctypes.Structure): pass
const_struct_WGPUImageCopyTexture._fields_ = [
  ('texture', WGPUTexture),
  ('mipLevel', uint32_t),
  ('origin', WGPUOrigin3D),
  ('aspect', WGPUTextureAspect),
]
class const_struct_WGPUExtent3D(ctypes.Structure): pass
const_struct_WGPUExtent3D._fields_ = [
  ('width', uint32_t),
  ('height', uint32_t),
  ('depthOrArrayLayers', uint32_t),
]
WGPUProcCommandEncoderCopyBufferToTexture = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(const_struct_WGPUImageCopyBuffer), ctypes.POINTER(const_struct_WGPUImageCopyTexture), ctypes.POINTER(const_struct_WGPUExtent3D))
WGPUProcCommandEncoderCopyTextureToBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(const_struct_WGPUImageCopyTexture), ctypes.POINTER(const_struct_WGPUImageCopyBuffer), ctypes.POINTER(const_struct_WGPUExtent3D))
WGPUProcCommandEncoderCopyTextureToTexture = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(const_struct_WGPUImageCopyTexture), ctypes.POINTER(const_struct_WGPUImageCopyTexture), ctypes.POINTER(const_struct_WGPUExtent3D))
class const_struct_WGPUCommandBufferDescriptor(ctypes.Structure): pass
const_struct_WGPUCommandBufferDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
WGPUProcCommandEncoderFinish = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUCommandBufferImpl), ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(const_struct_WGPUCommandBufferDescriptor))
WGPUProcCommandEncoderInjectValidationError = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderInsertDebugMarker = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderPopDebugGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl))
WGPUProcCommandEncoderPushDebugGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderResolveQuerySet = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(struct_WGPUQuerySetImpl), ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong)
WGPUProcCommandEncoderSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderWriteBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong)
WGPUProcCommandEncoderWriteTimestamp = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(struct_WGPUQuerySetImpl), ctypes.c_uint)
WGPUProcCommandEncoderAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl))
WGPUProcCommandEncoderRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUCommandEncoderImpl))
WGPUProcComputePassEncoderDispatchWorkgroups = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
WGPUProcComputePassEncoderDispatchWorkgroupsIndirect = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong)
WGPUProcComputePassEncoderEnd = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderInsertDebugMarker = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderPopDebugGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderPushDebugGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderSetBindGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl), ctypes.c_uint, ctypes.POINTER(struct_WGPUBindGroupImpl), ctypes.c_ulong, ctypes.POINTER(ctypes.c_uint))
WGPUProcComputePassEncoderSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderSetPipeline = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl), ctypes.POINTER(struct_WGPUComputePipelineImpl))
WGPUProcComputePassEncoderWriteTimestamp = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl), ctypes.POINTER(struct_WGPUQuerySetImpl), ctypes.c_uint)
WGPUProcComputePassEncoderAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePipelineGetBindGroupLayout = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUBindGroupLayoutImpl), ctypes.POINTER(struct_WGPUComputePipelineImpl), ctypes.c_uint)
WGPUProcComputePipelineSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePipelineImpl), struct_WGPUStringView)
WGPUProcComputePipelineAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePipelineImpl))
WGPUProcComputePipelineRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUComputePipelineImpl))
class const_struct_WGPUBindGroupDescriptor(ctypes.Structure): pass
const_struct_WGPUBindGroupDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('layout', WGPUBindGroupLayout),
  ('entryCount', size_t),
  ('entries', ctypes.POINTER(WGPUBindGroupEntry)),
]
WGPUProcDeviceCreateBindGroup = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUBindGroupImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUBindGroupDescriptor))
class const_struct_WGPUBindGroupLayoutDescriptor(ctypes.Structure): pass
const_struct_WGPUBindGroupLayoutDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('entryCount', size_t),
  ('entries', ctypes.POINTER(WGPUBindGroupLayoutEntry)),
]
WGPUProcDeviceCreateBindGroupLayout = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUBindGroupLayoutImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUBindGroupLayoutDescriptor))
class const_struct_WGPUBufferDescriptor(ctypes.Structure): pass
const_struct_WGPUBufferDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('usage', WGPUBufferUsage),
  ('size', uint64_t),
  ('mappedAtCreation', WGPUBool),
]
WGPUProcDeviceCreateBuffer = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUBufferImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUBufferDescriptor))
class const_struct_WGPUCommandEncoderDescriptor(ctypes.Structure): pass
const_struct_WGPUCommandEncoderDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
WGPUProcDeviceCreateCommandEncoder = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUCommandEncoderImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUCommandEncoderDescriptor))
class const_struct_WGPUComputePipelineDescriptor(ctypes.Structure): pass
const_struct_WGPUComputePipelineDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('layout', WGPUPipelineLayout),
  ('compute', WGPUComputeState),
]
WGPUProcDeviceCreateComputePipeline = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUComputePipelineImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUComputePipelineDescriptor))
WGPUProcDeviceCreateComputePipelineAsync = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUComputePipelineDescriptor), ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, ctypes.POINTER(struct_WGPUComputePipelineImpl), struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcDeviceCreateComputePipelineAsync2 = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUComputePipelineDescriptor), struct_WGPUCreateComputePipelineAsyncCallbackInfo2)
WGPUProcDeviceCreateComputePipelineAsyncF = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUComputePipelineDescriptor), struct_WGPUCreateComputePipelineAsyncCallbackInfo)
WGPUProcDeviceCreateErrorBuffer = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUBufferImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUBufferDescriptor))
WGPUProcDeviceCreateErrorExternalTexture = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUExternalTextureImpl), ctypes.POINTER(struct_WGPUDeviceImpl))
class const_struct_WGPUShaderModuleDescriptor(ctypes.Structure): pass
const_struct_WGPUShaderModuleDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
WGPUProcDeviceCreateErrorShaderModule = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUShaderModuleImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUShaderModuleDescriptor), struct_WGPUStringView)
class const_struct_WGPUTextureDescriptor(ctypes.Structure): pass
const_struct_WGPUTextureDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('usage', WGPUTextureUsage),
  ('dimension', WGPUTextureDimension),
  ('size', WGPUExtent3D),
  ('format', WGPUTextureFormat),
  ('mipLevelCount', uint32_t),
  ('sampleCount', uint32_t),
  ('viewFormatCount', size_t),
  ('viewFormats', ctypes.POINTER(WGPUTextureFormat)),
]
WGPUProcDeviceCreateErrorTexture = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUTextureImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUTextureDescriptor))
class const_struct_WGPUExternalTextureDescriptor(ctypes.Structure): pass
const_struct_WGPUExternalTextureDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('plane0', WGPUTextureView),
  ('plane1', WGPUTextureView),
  ('cropOrigin', WGPUOrigin2D),
  ('cropSize', WGPUExtent2D),
  ('apparentSize', WGPUExtent2D),
  ('doYuvToRgbConversionOnly', WGPUBool),
  ('yuvToRgbConversionMatrix', ctypes.POINTER(ctypes.c_float)),
  ('srcTransferFunctionParameters', ctypes.POINTER(ctypes.c_float)),
  ('dstTransferFunctionParameters', ctypes.POINTER(ctypes.c_float)),
  ('gamutConversionMatrix', ctypes.POINTER(ctypes.c_float)),
  ('mirrored', WGPUBool),
  ('rotation', WGPUExternalTextureRotation),
]
WGPUProcDeviceCreateExternalTexture = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUExternalTextureImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUExternalTextureDescriptor))
class const_struct_WGPUPipelineLayoutDescriptor(ctypes.Structure): pass
const_struct_WGPUPipelineLayoutDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('bindGroupLayoutCount', size_t),
  ('bindGroupLayouts', ctypes.POINTER(WGPUBindGroupLayout)),
  ('immediateDataRangeByteSize', uint32_t),
]
WGPUProcDeviceCreatePipelineLayout = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUPipelineLayoutImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUPipelineLayoutDescriptor))
class const_struct_WGPUQuerySetDescriptor(ctypes.Structure): pass
const_struct_WGPUQuerySetDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('type', WGPUQueryType),
  ('count', uint32_t),
]
WGPUProcDeviceCreateQuerySet = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUQuerySetImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUQuerySetDescriptor))
class const_struct_WGPURenderBundleEncoderDescriptor(ctypes.Structure): pass
const_struct_WGPURenderBundleEncoderDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('colorFormatCount', size_t),
  ('colorFormats', ctypes.POINTER(WGPUTextureFormat)),
  ('depthStencilFormat', WGPUTextureFormat),
  ('sampleCount', uint32_t),
  ('depthReadOnly', WGPUBool),
  ('stencilReadOnly', WGPUBool),
]
WGPUProcDeviceCreateRenderBundleEncoder = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPURenderBundleEncoderDescriptor))
class const_struct_WGPURenderPipelineDescriptor(ctypes.Structure): pass
const_struct_WGPURenderPipelineDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('layout', WGPUPipelineLayout),
  ('vertex', WGPUVertexState),
  ('primitive', WGPUPrimitiveState),
  ('depthStencil', ctypes.POINTER(WGPUDepthStencilState)),
  ('multisample', WGPUMultisampleState),
  ('fragment', ctypes.POINTER(WGPUFragmentState)),
]
WGPUProcDeviceCreateRenderPipeline = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPURenderPipelineImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPURenderPipelineDescriptor))
WGPUProcDeviceCreateRenderPipelineAsync = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPURenderPipelineDescriptor), ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, ctypes.POINTER(struct_WGPURenderPipelineImpl), struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcDeviceCreateRenderPipelineAsync2 = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPURenderPipelineDescriptor), struct_WGPUCreateRenderPipelineAsyncCallbackInfo2)
WGPUProcDeviceCreateRenderPipelineAsyncF = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPURenderPipelineDescriptor), struct_WGPUCreateRenderPipelineAsyncCallbackInfo)
class const_struct_WGPUSamplerDescriptor(ctypes.Structure): pass
const_struct_WGPUSamplerDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('addressModeU', WGPUAddressMode),
  ('addressModeV', WGPUAddressMode),
  ('addressModeW', WGPUAddressMode),
  ('magFilter', WGPUFilterMode),
  ('minFilter', WGPUFilterMode),
  ('mipmapFilter', WGPUMipmapFilterMode),
  ('lodMinClamp', ctypes.c_float),
  ('lodMaxClamp', ctypes.c_float),
  ('compare', WGPUCompareFunction),
  ('maxAnisotropy', uint16_t),
]
WGPUProcDeviceCreateSampler = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUSamplerImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUSamplerDescriptor))
WGPUProcDeviceCreateShaderModule = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUShaderModuleImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUShaderModuleDescriptor))
WGPUProcDeviceCreateTexture = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUTextureImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUTextureDescriptor))
WGPUProcDeviceDestroy = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceForceLoss = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), enum_WGPUDeviceLostReason, struct_WGPUStringView)
WGPUProcDeviceGetAHardwareBufferProperties = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.c_void_p, ctypes.POINTER(struct_WGPUAHardwareBufferProperties))
WGPUProcDeviceGetAdapter = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceGetAdapterInfo = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(struct_WGPUAdapterInfo))
WGPUProcDeviceGetFeatures = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(struct_WGPUSupportedFeatures))
WGPUProcDeviceGetLimits = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(struct_WGPUSupportedLimits))
WGPUProcDeviceGetLostFuture = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceGetQueue = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUQueueImpl), ctypes.POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceHasFeature = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUDeviceImpl), enum_WGPUFeatureName)
class const_struct_WGPUSharedBufferMemoryDescriptor(ctypes.Structure): pass
const_struct_WGPUSharedBufferMemoryDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
WGPUProcDeviceImportSharedBufferMemory = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUSharedBufferMemoryDescriptor))
class const_struct_WGPUSharedFenceDescriptor(ctypes.Structure): pass
const_struct_WGPUSharedFenceDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
WGPUProcDeviceImportSharedFence = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUSharedFenceImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUSharedFenceDescriptor))
class const_struct_WGPUSharedTextureMemoryDescriptor(ctypes.Structure): pass
const_struct_WGPUSharedTextureMemoryDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
WGPUProcDeviceImportSharedTextureMemory = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl), ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUSharedTextureMemoryDescriptor))
WGPUProcDeviceInjectError = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), enum_WGPUErrorType, struct_WGPUStringView)
WGPUProcDevicePopErrorScope = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.CFUNCTYPE(None, enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcDevicePopErrorScope2 = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUDeviceImpl), struct_WGPUPopErrorScopeCallbackInfo2)
WGPUProcDevicePopErrorScopeF = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUDeviceImpl), struct_WGPUPopErrorScopeCallbackInfo)
WGPUProcDevicePushErrorScope = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), enum_WGPUErrorFilter)
WGPUProcDeviceSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), struct_WGPUStringView)
WGPUProcDeviceSetLoggingCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.CFUNCTYPE(None, enum_WGPULoggingType, struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcDeviceTick = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceValidateTextureDescriptor = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl), ctypes.POINTER(const_struct_WGPUTextureDescriptor))
WGPUProcDeviceAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUDeviceImpl))
WGPUProcExternalTextureDestroy = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureExpire = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureRefresh = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUExternalTextureImpl), struct_WGPUStringView)
WGPUProcExternalTextureAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUExternalTextureImpl))
class const_struct_WGPUSurfaceDescriptor(ctypes.Structure): pass
const_struct_WGPUSurfaceDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
WGPUProcInstanceCreateSurface = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUSurfaceImpl), ctypes.POINTER(struct_WGPUInstanceImpl), ctypes.POINTER(const_struct_WGPUSurfaceDescriptor))
WGPUProcInstanceEnumerateWGSLLanguageFeatures = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.POINTER(struct_WGPUInstanceImpl), ctypes.POINTER(enum_WGPUWGSLFeatureName))
WGPUProcInstanceHasWGSLLanguageFeature = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUInstanceImpl), enum_WGPUWGSLFeatureName)
WGPUProcInstanceProcessEvents = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUInstanceImpl))
class const_struct_WGPURequestAdapterOptions(ctypes.Structure): pass
const_struct_WGPURequestAdapterOptions._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('compatibleSurface', WGPUSurface),
  ('featureLevel', WGPUFeatureLevel),
  ('powerPreference', WGPUPowerPreference),
  ('backendType', WGPUBackendType),
  ('forceFallbackAdapter', WGPUBool),
  ('compatibilityMode', WGPUBool),
]
WGPUProcInstanceRequestAdapter = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUInstanceImpl), ctypes.POINTER(const_struct_WGPURequestAdapterOptions), ctypes.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, ctypes.POINTER(struct_WGPUAdapterImpl), struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcInstanceRequestAdapter2 = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUInstanceImpl), ctypes.POINTER(const_struct_WGPURequestAdapterOptions), struct_WGPURequestAdapterCallbackInfo2)
WGPUProcInstanceRequestAdapterF = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUInstanceImpl), ctypes.POINTER(const_struct_WGPURequestAdapterOptions), struct_WGPURequestAdapterCallbackInfo)
WGPUProcInstanceWaitAny = ctypes.CFUNCTYPE(enum_WGPUWaitStatus, ctypes.POINTER(struct_WGPUInstanceImpl), ctypes.c_ulong, ctypes.POINTER(struct_WGPUFutureWaitInfo), ctypes.c_ulong)
WGPUProcInstanceAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUInstanceImpl))
WGPUProcInstanceRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUInstanceImpl))
WGPUProcPipelineLayoutSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUPipelineLayoutImpl), struct_WGPUStringView)
WGPUProcPipelineLayoutAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUPipelineLayoutImpl))
WGPUProcPipelineLayoutRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUPipelineLayoutImpl))
WGPUProcQuerySetDestroy = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQuerySetImpl))
WGPUProcQuerySetGetCount = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUQuerySetImpl))
WGPUProcQuerySetGetType = ctypes.CFUNCTYPE(enum_WGPUQueryType, ctypes.POINTER(struct_WGPUQuerySetImpl))
WGPUProcQuerySetSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQuerySetImpl), struct_WGPUStringView)
WGPUProcQuerySetAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQuerySetImpl))
WGPUProcQuerySetRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQuerySetImpl))
class const_struct_WGPUImageCopyExternalTexture(ctypes.Structure): pass
const_struct_WGPUImageCopyExternalTexture._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('externalTexture', WGPUExternalTexture),
  ('origin', WGPUOrigin3D),
  ('naturalSize', WGPUExtent2D),
]
class const_struct_WGPUCopyTextureForBrowserOptions(ctypes.Structure): pass
const_struct_WGPUCopyTextureForBrowserOptions._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('flipY', WGPUBool),
  ('needsColorSpaceConversion', WGPUBool),
  ('srcAlphaMode', WGPUAlphaMode),
  ('srcTransferFunctionParameters', ctypes.POINTER(ctypes.c_float)),
  ('conversionMatrix', ctypes.POINTER(ctypes.c_float)),
  ('dstTransferFunctionParameters', ctypes.POINTER(ctypes.c_float)),
  ('dstAlphaMode', WGPUAlphaMode),
  ('internalUsage', WGPUBool),
]
WGPUProcQueueCopyExternalTextureForBrowser = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl), ctypes.POINTER(const_struct_WGPUImageCopyExternalTexture), ctypes.POINTER(const_struct_WGPUImageCopyTexture), ctypes.POINTER(const_struct_WGPUExtent3D), ctypes.POINTER(const_struct_WGPUCopyTextureForBrowserOptions))
WGPUProcQueueCopyTextureForBrowser = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl), ctypes.POINTER(const_struct_WGPUImageCopyTexture), ctypes.POINTER(const_struct_WGPUImageCopyTexture), ctypes.POINTER(const_struct_WGPUExtent3D), ctypes.POINTER(const_struct_WGPUCopyTextureForBrowserOptions))
WGPUProcQueueOnSubmittedWorkDone = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl), ctypes.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcQueueOnSubmittedWorkDone2 = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUQueueImpl), struct_WGPUQueueWorkDoneCallbackInfo2)
WGPUProcQueueOnSubmittedWorkDoneF = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUQueueImpl), struct_WGPUQueueWorkDoneCallbackInfo)
WGPUProcQueueSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl), struct_WGPUStringView)
WGPUProcQueueSubmit = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl), ctypes.c_ulong, ctypes.POINTER(ctypes.POINTER(struct_WGPUCommandBufferImpl)))
WGPUProcQueueWriteBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong)
class const_struct_WGPUTextureDataLayout(ctypes.Structure): pass
const_struct_WGPUTextureDataLayout._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('offset', uint64_t),
  ('bytesPerRow', uint32_t),
  ('rowsPerImage', uint32_t),
]
WGPUProcQueueWriteTexture = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl), ctypes.POINTER(const_struct_WGPUImageCopyTexture), ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(const_struct_WGPUTextureDataLayout), ctypes.POINTER(const_struct_WGPUExtent3D))
WGPUProcQueueAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl))
WGPUProcQueueRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUQueueImpl))
WGPUProcRenderBundleSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleImpl), struct_WGPUStringView)
WGPUProcRenderBundleAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleImpl))
WGPUProcRenderBundleRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleImpl))
WGPUProcRenderBundleEncoderDraw = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
WGPUProcRenderBundleEncoderDrawIndexed = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_uint)
WGPUProcRenderBundleEncoderDrawIndexedIndirect = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong)
WGPUProcRenderBundleEncoderDrawIndirect = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong)
class const_struct_WGPURenderBundleDescriptor(ctypes.Structure): pass
const_struct_WGPURenderBundleDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
]
WGPUProcRenderBundleEncoderFinish = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPURenderBundleImpl), ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.POINTER(const_struct_WGPURenderBundleDescriptor))
WGPUProcRenderBundleEncoderInsertDebugMarker = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderPopDebugGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderBundleEncoderPushDebugGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderSetBindGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.c_uint, ctypes.POINTER(struct_WGPUBindGroupImpl), ctypes.c_ulong, ctypes.POINTER(ctypes.c_uint))
WGPUProcRenderBundleEncoderSetIndexBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), enum_WGPUIndexFormat, ctypes.c_ulong, ctypes.c_ulong)
WGPUProcRenderBundleEncoderSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderSetPipeline = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.POINTER(struct_WGPURenderPipelineImpl))
WGPUProcRenderBundleEncoderSetVertexBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl), ctypes.c_uint, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong)
WGPUProcRenderBundleEncoderAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderBundleEncoderRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderPassEncoderBeginOcclusionQuery = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_uint)
WGPUProcRenderPassEncoderDraw = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
WGPUProcRenderPassEncoderDrawIndexed = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_uint)
WGPUProcRenderPassEncoderDrawIndexedIndirect = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong)
WGPUProcRenderPassEncoderDrawIndirect = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong)
WGPUProcRenderPassEncoderEnd = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderEndOcclusionQuery = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderExecuteBundles = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_ulong, ctypes.POINTER(ctypes.POINTER(struct_WGPURenderBundleImpl)))
WGPUProcRenderPassEncoderInsertDebugMarker = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderMultiDrawIndexedIndirect = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_uint, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong)
WGPUProcRenderPassEncoderMultiDrawIndirect = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_uint, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong)
WGPUProcRenderPassEncoderPixelLocalStorageBarrier = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderPopDebugGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderPushDebugGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderSetBindGroup = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_uint, ctypes.POINTER(struct_WGPUBindGroupImpl), ctypes.c_ulong, ctypes.POINTER(ctypes.c_uint))
class const_struct_WGPUColor(ctypes.Structure): pass
const_struct_WGPUColor._fields_ = [
  ('r', ctypes.c_double),
  ('g', ctypes.c_double),
  ('b', ctypes.c_double),
  ('a', ctypes.c_double),
]
WGPUProcRenderPassEncoderSetBlendConstant = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(const_struct_WGPUColor))
WGPUProcRenderPassEncoderSetIndexBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(struct_WGPUBufferImpl), enum_WGPUIndexFormat, ctypes.c_ulong, ctypes.c_ulong)
WGPUProcRenderPassEncoderSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderSetPipeline = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(struct_WGPURenderPipelineImpl))
WGPUProcRenderPassEncoderSetScissorRect = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
WGPUProcRenderPassEncoderSetStencilReference = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_uint)
WGPUProcRenderPassEncoderSetVertexBuffer = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_uint, ctypes.POINTER(struct_WGPUBufferImpl), ctypes.c_ulong, ctypes.c_ulong)
WGPUProcRenderPassEncoderSetViewport = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
WGPUProcRenderPassEncoderWriteTimestamp = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl), ctypes.POINTER(struct_WGPUQuerySetImpl), ctypes.c_uint)
WGPUProcRenderPassEncoderAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPipelineGetBindGroupLayout = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUBindGroupLayoutImpl), ctypes.POINTER(struct_WGPURenderPipelineImpl), ctypes.c_uint)
WGPUProcRenderPipelineSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPipelineImpl), struct_WGPUStringView)
WGPUProcRenderPipelineAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPipelineImpl))
WGPUProcRenderPipelineRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPURenderPipelineImpl))
WGPUProcSamplerSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSamplerImpl), struct_WGPUStringView)
WGPUProcSamplerAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSamplerImpl))
WGPUProcSamplerRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSamplerImpl))
WGPUProcShaderModuleGetCompilationInfo = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUShaderModuleImpl), ctypes.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, ctypes.POINTER(const_struct_WGPUCompilationInfo), ctypes.c_void_p), ctypes.c_void_p)
WGPUProcShaderModuleGetCompilationInfo2 = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUShaderModuleImpl), struct_WGPUCompilationInfoCallbackInfo2)
WGPUProcShaderModuleGetCompilationInfoF = ctypes.CFUNCTYPE(struct_WGPUFuture, ctypes.POINTER(struct_WGPUShaderModuleImpl), struct_WGPUCompilationInfoCallbackInfo)
WGPUProcShaderModuleSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUShaderModuleImpl), struct_WGPUStringView)
WGPUProcShaderModuleAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUShaderModuleImpl))
WGPUProcShaderModuleRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUShaderModuleImpl))
class const_struct_WGPUSharedBufferMemoryBeginAccessDescriptor(ctypes.Structure): pass
const_struct_WGPUSharedBufferMemoryBeginAccessDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('initialized', WGPUBool),
  ('fenceCount', size_t),
  ('fences', ctypes.POINTER(WGPUSharedFence)),
  ('signaledValues', ctypes.POINTER(uint64_t)),
]
WGPUProcSharedBufferMemoryBeginAccess = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.POINTER(const_struct_WGPUSharedBufferMemoryBeginAccessDescriptor))
WGPUProcSharedBufferMemoryCreateBuffer = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUBufferImpl), ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl), ctypes.POINTER(const_struct_WGPUBufferDescriptor))
WGPUProcSharedBufferMemoryEndAccess = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl), ctypes.POINTER(struct_WGPUBufferImpl), ctypes.POINTER(struct_WGPUSharedBufferMemoryEndAccessState))
WGPUProcSharedBufferMemoryGetProperties = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl), ctypes.POINTER(struct_WGPUSharedBufferMemoryProperties))
WGPUProcSharedBufferMemoryIsDeviceLost = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedBufferMemorySetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl), struct_WGPUStringView)
WGPUProcSharedBufferMemoryAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedBufferMemoryRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedFenceExportInfo = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedFenceImpl), ctypes.POINTER(struct_WGPUSharedFenceExportInfo))
WGPUProcSharedFenceAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedFenceImpl))
WGPUProcSharedFenceRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedFenceImpl))
class const_struct_WGPUSharedTextureMemoryBeginAccessDescriptor(ctypes.Structure): pass
const_struct_WGPUSharedTextureMemoryBeginAccessDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('concurrentRead', WGPUBool),
  ('initialized', WGPUBool),
  ('fenceCount', size_t),
  ('fences', ctypes.POINTER(WGPUSharedFence)),
  ('signaledValues', ctypes.POINTER(uint64_t)),
]
WGPUProcSharedTextureMemoryBeginAccess = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl), ctypes.POINTER(struct_WGPUTextureImpl), ctypes.POINTER(const_struct_WGPUSharedTextureMemoryBeginAccessDescriptor))
WGPUProcSharedTextureMemoryCreateTexture = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUTextureImpl), ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl), ctypes.POINTER(const_struct_WGPUTextureDescriptor))
WGPUProcSharedTextureMemoryEndAccess = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl), ctypes.POINTER(struct_WGPUTextureImpl), ctypes.POINTER(struct_WGPUSharedTextureMemoryEndAccessState))
WGPUProcSharedTextureMemoryGetProperties = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl), ctypes.POINTER(struct_WGPUSharedTextureMemoryProperties))
WGPUProcSharedTextureMemoryIsDeviceLost = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl))
WGPUProcSharedTextureMemorySetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl), struct_WGPUStringView)
WGPUProcSharedTextureMemoryAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl))
WGPUProcSharedTextureMemoryRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSharedTextureMemoryImpl))
class const_struct_WGPUSurfaceConfiguration(ctypes.Structure): pass
const_struct_WGPUSurfaceConfiguration._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('device', WGPUDevice),
  ('format', WGPUTextureFormat),
  ('usage', WGPUTextureUsage),
  ('viewFormatCount', size_t),
  ('viewFormats', ctypes.POINTER(WGPUTextureFormat)),
  ('alphaMode', WGPUCompositeAlphaMode),
  ('width', uint32_t),
  ('height', uint32_t),
  ('presentMode', WGPUPresentMode),
]
WGPUProcSurfaceConfigure = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSurfaceImpl), ctypes.POINTER(const_struct_WGPUSurfaceConfiguration))
WGPUProcSurfaceGetCapabilities = ctypes.CFUNCTYPE(enum_WGPUStatus, ctypes.POINTER(struct_WGPUSurfaceImpl), ctypes.POINTER(struct_WGPUAdapterImpl), ctypes.POINTER(struct_WGPUSurfaceCapabilities))
WGPUProcSurfaceGetCurrentTexture = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSurfaceImpl), ctypes.POINTER(struct_WGPUSurfaceTexture))
WGPUProcSurfacePresent = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSurfaceImpl))
WGPUProcSurfaceSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSurfaceImpl), struct_WGPUStringView)
WGPUProcSurfaceUnconfigure = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSurfaceImpl))
WGPUProcSurfaceAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSurfaceImpl))
WGPUProcSurfaceRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUSurfaceImpl))
class const_struct_WGPUTextureViewDescriptor(ctypes.Structure): pass
const_struct_WGPUTextureViewDescriptor._fields_ = [
  ('nextInChain', ctypes.POINTER(WGPUChainedStruct)),
  ('label', WGPUStringView),
  ('format', WGPUTextureFormat),
  ('dimension', WGPUTextureViewDimension),
  ('baseMipLevel', uint32_t),
  ('mipLevelCount', uint32_t),
  ('baseArrayLayer', uint32_t),
  ('arrayLayerCount', uint32_t),
  ('aspect', WGPUTextureAspect),
  ('usage', WGPUTextureUsage),
]
WGPUProcTextureCreateErrorView = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUTextureViewImpl), ctypes.POINTER(struct_WGPUTextureImpl), ctypes.POINTER(const_struct_WGPUTextureViewDescriptor))
WGPUProcTextureCreateView = ctypes.CFUNCTYPE(ctypes.POINTER(struct_WGPUTextureViewImpl), ctypes.POINTER(struct_WGPUTextureImpl), ctypes.POINTER(const_struct_WGPUTextureViewDescriptor))
WGPUProcTextureDestroy = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetDepthOrArrayLayers = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetDimension = ctypes.CFUNCTYPE(enum_WGPUTextureDimension, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetFormat = ctypes.CFUNCTYPE(enum_WGPUTextureFormat, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetHeight = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetMipLevelCount = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetSampleCount = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetUsage = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetWidth = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUTextureImpl), struct_WGPUStringView)
WGPUProcTextureAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUTextureImpl))
WGPUProcTextureViewSetLabel = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUTextureViewImpl), struct_WGPUStringView)
WGPUProcTextureViewAddRef = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUTextureViewImpl))
WGPUProcTextureViewRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_WGPUTextureViewImpl))
# void wgpuAdapterInfoFreeMembers(WGPUAdapterInfo value)
try: (wgpuAdapterInfoFreeMembers:=dll.wgpuAdapterInfoFreeMembers).restype, wgpuAdapterInfoFreeMembers.argtypes = None, [WGPUAdapterInfo]
except AttributeError: pass

# void wgpuAdapterPropertiesMemoryHeapsFreeMembers(WGPUAdapterPropertiesMemoryHeaps value)
try: (wgpuAdapterPropertiesMemoryHeapsFreeMembers:=dll.wgpuAdapterPropertiesMemoryHeapsFreeMembers).restype, wgpuAdapterPropertiesMemoryHeapsFreeMembers.argtypes = None, [WGPUAdapterPropertiesMemoryHeaps]
except AttributeError: pass

# WGPUInstance wgpuCreateInstance(const WGPUInstanceDescriptor *descriptor)
try: (wgpuCreateInstance:=dll.wgpuCreateInstance).restype, wgpuCreateInstance.argtypes = WGPUInstance, [ctypes.POINTER(WGPUInstanceDescriptor)]
except AttributeError: pass

# void wgpuDrmFormatCapabilitiesFreeMembers(WGPUDrmFormatCapabilities value)
try: (wgpuDrmFormatCapabilitiesFreeMembers:=dll.wgpuDrmFormatCapabilitiesFreeMembers).restype, wgpuDrmFormatCapabilitiesFreeMembers.argtypes = None, [WGPUDrmFormatCapabilities]
except AttributeError: pass

# WGPUStatus wgpuGetInstanceFeatures(WGPUInstanceFeatures *features)
try: (wgpuGetInstanceFeatures:=dll.wgpuGetInstanceFeatures).restype, wgpuGetInstanceFeatures.argtypes = WGPUStatus, [ctypes.POINTER(WGPUInstanceFeatures)]
except AttributeError: pass

# WGPUProc wgpuGetProcAddress(WGPUStringView procName)
try: (wgpuGetProcAddress:=dll.wgpuGetProcAddress).restype, wgpuGetProcAddress.argtypes = WGPUProc, [WGPUStringView]
except AttributeError: pass

# void wgpuSharedBufferMemoryEndAccessStateFreeMembers(WGPUSharedBufferMemoryEndAccessState value)
try: (wgpuSharedBufferMemoryEndAccessStateFreeMembers:=dll.wgpuSharedBufferMemoryEndAccessStateFreeMembers).restype, wgpuSharedBufferMemoryEndAccessStateFreeMembers.argtypes = None, [WGPUSharedBufferMemoryEndAccessState]
except AttributeError: pass

# void wgpuSharedTextureMemoryEndAccessStateFreeMembers(WGPUSharedTextureMemoryEndAccessState value)
try: (wgpuSharedTextureMemoryEndAccessStateFreeMembers:=dll.wgpuSharedTextureMemoryEndAccessStateFreeMembers).restype, wgpuSharedTextureMemoryEndAccessStateFreeMembers.argtypes = None, [WGPUSharedTextureMemoryEndAccessState]
except AttributeError: pass

# void wgpuSupportedFeaturesFreeMembers(WGPUSupportedFeatures value)
try: (wgpuSupportedFeaturesFreeMembers:=dll.wgpuSupportedFeaturesFreeMembers).restype, wgpuSupportedFeaturesFreeMembers.argtypes = None, [WGPUSupportedFeatures]
except AttributeError: pass

# void wgpuSurfaceCapabilitiesFreeMembers(WGPUSurfaceCapabilities value)
try: (wgpuSurfaceCapabilitiesFreeMembers:=dll.wgpuSurfaceCapabilitiesFreeMembers).restype, wgpuSurfaceCapabilitiesFreeMembers.argtypes = None, [WGPUSurfaceCapabilities]
except AttributeError: pass

# WGPUDevice wgpuAdapterCreateDevice(WGPUAdapter adapter, const WGPUDeviceDescriptor *descriptor)
try: (wgpuAdapterCreateDevice:=dll.wgpuAdapterCreateDevice).restype, wgpuAdapterCreateDevice.argtypes = WGPUDevice, [WGPUAdapter, ctypes.POINTER(WGPUDeviceDescriptor)]
except AttributeError: pass

# void wgpuAdapterGetFeatures(WGPUAdapter adapter, WGPUSupportedFeatures *features)
try: (wgpuAdapterGetFeatures:=dll.wgpuAdapterGetFeatures).restype, wgpuAdapterGetFeatures.argtypes = None, [WGPUAdapter, ctypes.POINTER(WGPUSupportedFeatures)]
except AttributeError: pass

# WGPUStatus wgpuAdapterGetFormatCapabilities(WGPUAdapter adapter, WGPUTextureFormat format, WGPUFormatCapabilities *capabilities)
try: (wgpuAdapterGetFormatCapabilities:=dll.wgpuAdapterGetFormatCapabilities).restype, wgpuAdapterGetFormatCapabilities.argtypes = WGPUStatus, [WGPUAdapter, WGPUTextureFormat, ctypes.POINTER(WGPUFormatCapabilities)]
except AttributeError: pass

# WGPUStatus wgpuAdapterGetInfo(WGPUAdapter adapter, WGPUAdapterInfo *info)
try: (wgpuAdapterGetInfo:=dll.wgpuAdapterGetInfo).restype, wgpuAdapterGetInfo.argtypes = WGPUStatus, [WGPUAdapter, ctypes.POINTER(WGPUAdapterInfo)]
except AttributeError: pass

# WGPUInstance wgpuAdapterGetInstance(WGPUAdapter adapter)
try: (wgpuAdapterGetInstance:=dll.wgpuAdapterGetInstance).restype, wgpuAdapterGetInstance.argtypes = WGPUInstance, [WGPUAdapter]
except AttributeError: pass

# WGPUStatus wgpuAdapterGetLimits(WGPUAdapter adapter, WGPUSupportedLimits *limits)
try: (wgpuAdapterGetLimits:=dll.wgpuAdapterGetLimits).restype, wgpuAdapterGetLimits.argtypes = WGPUStatus, [WGPUAdapter, ctypes.POINTER(WGPUSupportedLimits)]
except AttributeError: pass

# WGPUBool wgpuAdapterHasFeature(WGPUAdapter adapter, WGPUFeatureName feature)
try: (wgpuAdapterHasFeature:=dll.wgpuAdapterHasFeature).restype, wgpuAdapterHasFeature.argtypes = WGPUBool, [WGPUAdapter, WGPUFeatureName]
except AttributeError: pass

# void wgpuAdapterRequestDevice(WGPUAdapter adapter, const WGPUDeviceDescriptor *descriptor, WGPURequestDeviceCallback callback, void *userdata)
try: (wgpuAdapterRequestDevice:=dll.wgpuAdapterRequestDevice).restype, wgpuAdapterRequestDevice.argtypes = None, [WGPUAdapter, ctypes.POINTER(WGPUDeviceDescriptor), WGPURequestDeviceCallback, ctypes.c_void_p]
except AttributeError: pass

# WGPUFuture wgpuAdapterRequestDevice2(WGPUAdapter adapter, const WGPUDeviceDescriptor *options, WGPURequestDeviceCallbackInfo2 callbackInfo)
try: (wgpuAdapterRequestDevice2:=dll.wgpuAdapterRequestDevice2).restype, wgpuAdapterRequestDevice2.argtypes = WGPUFuture, [WGPUAdapter, ctypes.POINTER(WGPUDeviceDescriptor), WGPURequestDeviceCallbackInfo2]
except AttributeError: pass

# WGPUFuture wgpuAdapterRequestDeviceF(WGPUAdapter adapter, const WGPUDeviceDescriptor *options, WGPURequestDeviceCallbackInfo callbackInfo)
try: (wgpuAdapterRequestDeviceF:=dll.wgpuAdapterRequestDeviceF).restype, wgpuAdapterRequestDeviceF.argtypes = WGPUFuture, [WGPUAdapter, ctypes.POINTER(WGPUDeviceDescriptor), WGPURequestDeviceCallbackInfo]
except AttributeError: pass

# void wgpuAdapterAddRef(WGPUAdapter adapter)
try: (wgpuAdapterAddRef:=dll.wgpuAdapterAddRef).restype, wgpuAdapterAddRef.argtypes = None, [WGPUAdapter]
except AttributeError: pass

# void wgpuAdapterRelease(WGPUAdapter adapter)
try: (wgpuAdapterRelease:=dll.wgpuAdapterRelease).restype, wgpuAdapterRelease.argtypes = None, [WGPUAdapter]
except AttributeError: pass

# void wgpuBindGroupSetLabel(WGPUBindGroup bindGroup, WGPUStringView label)
try: (wgpuBindGroupSetLabel:=dll.wgpuBindGroupSetLabel).restype, wgpuBindGroupSetLabel.argtypes = None, [WGPUBindGroup, WGPUStringView]
except AttributeError: pass

# void wgpuBindGroupAddRef(WGPUBindGroup bindGroup)
try: (wgpuBindGroupAddRef:=dll.wgpuBindGroupAddRef).restype, wgpuBindGroupAddRef.argtypes = None, [WGPUBindGroup]
except AttributeError: pass

# void wgpuBindGroupRelease(WGPUBindGroup bindGroup)
try: (wgpuBindGroupRelease:=dll.wgpuBindGroupRelease).restype, wgpuBindGroupRelease.argtypes = None, [WGPUBindGroup]
except AttributeError: pass

# void wgpuBindGroupLayoutSetLabel(WGPUBindGroupLayout bindGroupLayout, WGPUStringView label)
try: (wgpuBindGroupLayoutSetLabel:=dll.wgpuBindGroupLayoutSetLabel).restype, wgpuBindGroupLayoutSetLabel.argtypes = None, [WGPUBindGroupLayout, WGPUStringView]
except AttributeError: pass

# void wgpuBindGroupLayoutAddRef(WGPUBindGroupLayout bindGroupLayout)
try: (wgpuBindGroupLayoutAddRef:=dll.wgpuBindGroupLayoutAddRef).restype, wgpuBindGroupLayoutAddRef.argtypes = None, [WGPUBindGroupLayout]
except AttributeError: pass

# void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout bindGroupLayout)
try: (wgpuBindGroupLayoutRelease:=dll.wgpuBindGroupLayoutRelease).restype, wgpuBindGroupLayoutRelease.argtypes = None, [WGPUBindGroupLayout]
except AttributeError: pass

# void wgpuBufferDestroy(WGPUBuffer buffer)
try: (wgpuBufferDestroy:=dll.wgpuBufferDestroy).restype, wgpuBufferDestroy.argtypes = None, [WGPUBuffer]
except AttributeError: pass

# const void *wgpuBufferGetConstMappedRange(WGPUBuffer buffer, size_t offset, size_t size)
try: (wgpuBufferGetConstMappedRange:=dll.wgpuBufferGetConstMappedRange).restype, wgpuBufferGetConstMappedRange.argtypes = ctypes.c_void_p, [WGPUBuffer, size_t, size_t]
except AttributeError: pass

# WGPUBufferMapState wgpuBufferGetMapState(WGPUBuffer buffer)
try: (wgpuBufferGetMapState:=dll.wgpuBufferGetMapState).restype, wgpuBufferGetMapState.argtypes = WGPUBufferMapState, [WGPUBuffer]
except AttributeError: pass

# void *wgpuBufferGetMappedRange(WGPUBuffer buffer, size_t offset, size_t size)
try: (wgpuBufferGetMappedRange:=dll.wgpuBufferGetMappedRange).restype, wgpuBufferGetMappedRange.argtypes = ctypes.c_void_p, [WGPUBuffer, size_t, size_t]
except AttributeError: pass

# uint64_t wgpuBufferGetSize(WGPUBuffer buffer)
try: (wgpuBufferGetSize:=dll.wgpuBufferGetSize).restype, wgpuBufferGetSize.argtypes = uint64_t, [WGPUBuffer]
except AttributeError: pass

# WGPUBufferUsage wgpuBufferGetUsage(WGPUBuffer buffer)
try: (wgpuBufferGetUsage:=dll.wgpuBufferGetUsage).restype, wgpuBufferGetUsage.argtypes = WGPUBufferUsage, [WGPUBuffer]
except AttributeError: pass

# void wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallback callback, void *userdata)
try: (wgpuBufferMapAsync:=dll.wgpuBufferMapAsync).restype, wgpuBufferMapAsync.argtypes = None, [WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallback, ctypes.c_void_p]
except AttributeError: pass

# WGPUFuture wgpuBufferMapAsync2(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo2 callbackInfo)
try: (wgpuBufferMapAsync2:=dll.wgpuBufferMapAsync2).restype, wgpuBufferMapAsync2.argtypes = WGPUFuture, [WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallbackInfo2]
except AttributeError: pass

# WGPUFuture wgpuBufferMapAsyncF(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo callbackInfo)
try: (wgpuBufferMapAsyncF:=dll.wgpuBufferMapAsyncF).restype, wgpuBufferMapAsyncF.argtypes = WGPUFuture, [WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallbackInfo]
except AttributeError: pass

# void wgpuBufferSetLabel(WGPUBuffer buffer, WGPUStringView label)
try: (wgpuBufferSetLabel:=dll.wgpuBufferSetLabel).restype, wgpuBufferSetLabel.argtypes = None, [WGPUBuffer, WGPUStringView]
except AttributeError: pass

# void wgpuBufferUnmap(WGPUBuffer buffer)
try: (wgpuBufferUnmap:=dll.wgpuBufferUnmap).restype, wgpuBufferUnmap.argtypes = None, [WGPUBuffer]
except AttributeError: pass

# void wgpuBufferAddRef(WGPUBuffer buffer)
try: (wgpuBufferAddRef:=dll.wgpuBufferAddRef).restype, wgpuBufferAddRef.argtypes = None, [WGPUBuffer]
except AttributeError: pass

# void wgpuBufferRelease(WGPUBuffer buffer)
try: (wgpuBufferRelease:=dll.wgpuBufferRelease).restype, wgpuBufferRelease.argtypes = None, [WGPUBuffer]
except AttributeError: pass

# void wgpuCommandBufferSetLabel(WGPUCommandBuffer commandBuffer, WGPUStringView label)
try: (wgpuCommandBufferSetLabel:=dll.wgpuCommandBufferSetLabel).restype, wgpuCommandBufferSetLabel.argtypes = None, [WGPUCommandBuffer, WGPUStringView]
except AttributeError: pass

# void wgpuCommandBufferAddRef(WGPUCommandBuffer commandBuffer)
try: (wgpuCommandBufferAddRef:=dll.wgpuCommandBufferAddRef).restype, wgpuCommandBufferAddRef.argtypes = None, [WGPUCommandBuffer]
except AttributeError: pass

# void wgpuCommandBufferRelease(WGPUCommandBuffer commandBuffer)
try: (wgpuCommandBufferRelease:=dll.wgpuCommandBufferRelease).restype, wgpuCommandBufferRelease.argtypes = None, [WGPUCommandBuffer]
except AttributeError: pass

# WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder commandEncoder, const WGPUComputePassDescriptor *descriptor)
try: (wgpuCommandEncoderBeginComputePass:=dll.wgpuCommandEncoderBeginComputePass).restype, wgpuCommandEncoderBeginComputePass.argtypes = WGPUComputePassEncoder, [WGPUCommandEncoder, ctypes.POINTER(WGPUComputePassDescriptor)]
except AttributeError: pass

# WGPURenderPassEncoder wgpuCommandEncoderBeginRenderPass(WGPUCommandEncoder commandEncoder, const WGPURenderPassDescriptor *descriptor)
try: (wgpuCommandEncoderBeginRenderPass:=dll.wgpuCommandEncoderBeginRenderPass).restype, wgpuCommandEncoderBeginRenderPass.argtypes = WGPURenderPassEncoder, [WGPUCommandEncoder, ctypes.POINTER(WGPURenderPassDescriptor)]
except AttributeError: pass

# void wgpuCommandEncoderClearBuffer(WGPUCommandEncoder commandEncoder, WGPUBuffer buffer, uint64_t offset, uint64_t size)
try: (wgpuCommandEncoderClearBuffer:=dll.wgpuCommandEncoderClearBuffer).restype, wgpuCommandEncoderClearBuffer.argtypes = None, [WGPUCommandEncoder, WGPUBuffer, uint64_t, uint64_t]
except AttributeError: pass

# void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder commandEncoder, WGPUBuffer source, uint64_t sourceOffset, WGPUBuffer destination, uint64_t destinationOffset, uint64_t size)
try: (wgpuCommandEncoderCopyBufferToBuffer:=dll.wgpuCommandEncoderCopyBufferToBuffer).restype, wgpuCommandEncoderCopyBufferToBuffer.argtypes = None, [WGPUCommandEncoder, WGPUBuffer, uint64_t, WGPUBuffer, uint64_t, uint64_t]
except AttributeError: pass

# void wgpuCommandEncoderCopyBufferToTexture(WGPUCommandEncoder commandEncoder, const WGPUImageCopyBuffer *source, const WGPUImageCopyTexture *destination, const WGPUExtent3D *copySize)
try: (wgpuCommandEncoderCopyBufferToTexture:=dll.wgpuCommandEncoderCopyBufferToTexture).restype, wgpuCommandEncoderCopyBufferToTexture.argtypes = None, [WGPUCommandEncoder, ctypes.POINTER(WGPUImageCopyBuffer), ctypes.POINTER(WGPUImageCopyTexture), ctypes.POINTER(WGPUExtent3D)]
except AttributeError: pass

# void wgpuCommandEncoderCopyTextureToBuffer(WGPUCommandEncoder commandEncoder, const WGPUImageCopyTexture *source, const WGPUImageCopyBuffer *destination, const WGPUExtent3D *copySize)
try: (wgpuCommandEncoderCopyTextureToBuffer:=dll.wgpuCommandEncoderCopyTextureToBuffer).restype, wgpuCommandEncoderCopyTextureToBuffer.argtypes = None, [WGPUCommandEncoder, ctypes.POINTER(WGPUImageCopyTexture), ctypes.POINTER(WGPUImageCopyBuffer), ctypes.POINTER(WGPUExtent3D)]
except AttributeError: pass

# void wgpuCommandEncoderCopyTextureToTexture(WGPUCommandEncoder commandEncoder, const WGPUImageCopyTexture *source, const WGPUImageCopyTexture *destination, const WGPUExtent3D *copySize)
try: (wgpuCommandEncoderCopyTextureToTexture:=dll.wgpuCommandEncoderCopyTextureToTexture).restype, wgpuCommandEncoderCopyTextureToTexture.argtypes = None, [WGPUCommandEncoder, ctypes.POINTER(WGPUImageCopyTexture), ctypes.POINTER(WGPUImageCopyTexture), ctypes.POINTER(WGPUExtent3D)]
except AttributeError: pass

# WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder commandEncoder, const WGPUCommandBufferDescriptor *descriptor)
try: (wgpuCommandEncoderFinish:=dll.wgpuCommandEncoderFinish).restype, wgpuCommandEncoderFinish.argtypes = WGPUCommandBuffer, [WGPUCommandEncoder, ctypes.POINTER(WGPUCommandBufferDescriptor)]
except AttributeError: pass

# void wgpuCommandEncoderInjectValidationError(WGPUCommandEncoder commandEncoder, WGPUStringView message)
try: (wgpuCommandEncoderInjectValidationError:=dll.wgpuCommandEncoderInjectValidationError).restype, wgpuCommandEncoderInjectValidationError.argtypes = None, [WGPUCommandEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuCommandEncoderInsertDebugMarker(WGPUCommandEncoder commandEncoder, WGPUStringView markerLabel)
try: (wgpuCommandEncoderInsertDebugMarker:=dll.wgpuCommandEncoderInsertDebugMarker).restype, wgpuCommandEncoderInsertDebugMarker.argtypes = None, [WGPUCommandEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuCommandEncoderPopDebugGroup(WGPUCommandEncoder commandEncoder)
try: (wgpuCommandEncoderPopDebugGroup:=dll.wgpuCommandEncoderPopDebugGroup).restype, wgpuCommandEncoderPopDebugGroup.argtypes = None, [WGPUCommandEncoder]
except AttributeError: pass

# void wgpuCommandEncoderPushDebugGroup(WGPUCommandEncoder commandEncoder, WGPUStringView groupLabel)
try: (wgpuCommandEncoderPushDebugGroup:=dll.wgpuCommandEncoderPushDebugGroup).restype, wgpuCommandEncoderPushDebugGroup.argtypes = None, [WGPUCommandEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuCommandEncoderResolveQuerySet(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount, WGPUBuffer destination, uint64_t destinationOffset)
try: (wgpuCommandEncoderResolveQuerySet:=dll.wgpuCommandEncoderResolveQuerySet).restype, wgpuCommandEncoderResolveQuerySet.argtypes = None, [WGPUCommandEncoder, WGPUQuerySet, uint32_t, uint32_t, WGPUBuffer, uint64_t]
except AttributeError: pass

# void wgpuCommandEncoderSetLabel(WGPUCommandEncoder commandEncoder, WGPUStringView label)
try: (wgpuCommandEncoderSetLabel:=dll.wgpuCommandEncoderSetLabel).restype, wgpuCommandEncoderSetLabel.argtypes = None, [WGPUCommandEncoder, WGPUStringView]
except AttributeError: pass

uint8_t = ctypes.c_ubyte
# void wgpuCommandEncoderWriteBuffer(WGPUCommandEncoder commandEncoder, WGPUBuffer buffer, uint64_t bufferOffset, const uint8_t *data, uint64_t size)
try: (wgpuCommandEncoderWriteBuffer:=dll.wgpuCommandEncoderWriteBuffer).restype, wgpuCommandEncoderWriteBuffer.argtypes = None, [WGPUCommandEncoder, WGPUBuffer, uint64_t, ctypes.POINTER(uint8_t), uint64_t]
except AttributeError: pass

# void wgpuCommandEncoderWriteTimestamp(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t queryIndex)
try: (wgpuCommandEncoderWriteTimestamp:=dll.wgpuCommandEncoderWriteTimestamp).restype, wgpuCommandEncoderWriteTimestamp.argtypes = None, [WGPUCommandEncoder, WGPUQuerySet, uint32_t]
except AttributeError: pass

# void wgpuCommandEncoderAddRef(WGPUCommandEncoder commandEncoder)
try: (wgpuCommandEncoderAddRef:=dll.wgpuCommandEncoderAddRef).restype, wgpuCommandEncoderAddRef.argtypes = None, [WGPUCommandEncoder]
except AttributeError: pass

# void wgpuCommandEncoderRelease(WGPUCommandEncoder commandEncoder)
try: (wgpuCommandEncoderRelease:=dll.wgpuCommandEncoderRelease).restype, wgpuCommandEncoderRelease.argtypes = None, [WGPUCommandEncoder]
except AttributeError: pass

# void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder computePassEncoder, uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ)
try: (wgpuComputePassEncoderDispatchWorkgroups:=dll.wgpuComputePassEncoderDispatchWorkgroups).restype, wgpuComputePassEncoderDispatchWorkgroups.argtypes = None, [WGPUComputePassEncoder, uint32_t, uint32_t, uint32_t]
except AttributeError: pass

# void wgpuComputePassEncoderDispatchWorkgroupsIndirect(WGPUComputePassEncoder computePassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
try: (wgpuComputePassEncoderDispatchWorkgroupsIndirect:=dll.wgpuComputePassEncoderDispatchWorkgroupsIndirect).restype, wgpuComputePassEncoderDispatchWorkgroupsIndirect.argtypes = None, [WGPUComputePassEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

# void wgpuComputePassEncoderEnd(WGPUComputePassEncoder computePassEncoder)
try: (wgpuComputePassEncoderEnd:=dll.wgpuComputePassEncoderEnd).restype, wgpuComputePassEncoderEnd.argtypes = None, [WGPUComputePassEncoder]
except AttributeError: pass

# void wgpuComputePassEncoderInsertDebugMarker(WGPUComputePassEncoder computePassEncoder, WGPUStringView markerLabel)
try: (wgpuComputePassEncoderInsertDebugMarker:=dll.wgpuComputePassEncoderInsertDebugMarker).restype, wgpuComputePassEncoderInsertDebugMarker.argtypes = None, [WGPUComputePassEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuComputePassEncoderPopDebugGroup(WGPUComputePassEncoder computePassEncoder)
try: (wgpuComputePassEncoderPopDebugGroup:=dll.wgpuComputePassEncoderPopDebugGroup).restype, wgpuComputePassEncoderPopDebugGroup.argtypes = None, [WGPUComputePassEncoder]
except AttributeError: pass

# void wgpuComputePassEncoderPushDebugGroup(WGPUComputePassEncoder computePassEncoder, WGPUStringView groupLabel)
try: (wgpuComputePassEncoderPushDebugGroup:=dll.wgpuComputePassEncoderPushDebugGroup).restype, wgpuComputePassEncoderPushDebugGroup.argtypes = None, [WGPUComputePassEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder computePassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, const uint32_t *dynamicOffsets)
try: (wgpuComputePassEncoderSetBindGroup:=dll.wgpuComputePassEncoderSetBindGroup).restype, wgpuComputePassEncoderSetBindGroup.argtypes = None, [WGPUComputePassEncoder, uint32_t, WGPUBindGroup, size_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

# void wgpuComputePassEncoderSetLabel(WGPUComputePassEncoder computePassEncoder, WGPUStringView label)
try: (wgpuComputePassEncoderSetLabel:=dll.wgpuComputePassEncoderSetLabel).restype, wgpuComputePassEncoderSetLabel.argtypes = None, [WGPUComputePassEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder computePassEncoder, WGPUComputePipeline pipeline)
try: (wgpuComputePassEncoderSetPipeline:=dll.wgpuComputePassEncoderSetPipeline).restype, wgpuComputePassEncoderSetPipeline.argtypes = None, [WGPUComputePassEncoder, WGPUComputePipeline]
except AttributeError: pass

# void wgpuComputePassEncoderWriteTimestamp(WGPUComputePassEncoder computePassEncoder, WGPUQuerySet querySet, uint32_t queryIndex)
try: (wgpuComputePassEncoderWriteTimestamp:=dll.wgpuComputePassEncoderWriteTimestamp).restype, wgpuComputePassEncoderWriteTimestamp.argtypes = None, [WGPUComputePassEncoder, WGPUQuerySet, uint32_t]
except AttributeError: pass

# void wgpuComputePassEncoderAddRef(WGPUComputePassEncoder computePassEncoder)
try: (wgpuComputePassEncoderAddRef:=dll.wgpuComputePassEncoderAddRef).restype, wgpuComputePassEncoderAddRef.argtypes = None, [WGPUComputePassEncoder]
except AttributeError: pass

# void wgpuComputePassEncoderRelease(WGPUComputePassEncoder computePassEncoder)
try: (wgpuComputePassEncoderRelease:=dll.wgpuComputePassEncoderRelease).restype, wgpuComputePassEncoderRelease.argtypes = None, [WGPUComputePassEncoder]
except AttributeError: pass

# WGPUBindGroupLayout wgpuComputePipelineGetBindGroupLayout(WGPUComputePipeline computePipeline, uint32_t groupIndex)
try: (wgpuComputePipelineGetBindGroupLayout:=dll.wgpuComputePipelineGetBindGroupLayout).restype, wgpuComputePipelineGetBindGroupLayout.argtypes = WGPUBindGroupLayout, [WGPUComputePipeline, uint32_t]
except AttributeError: pass

# void wgpuComputePipelineSetLabel(WGPUComputePipeline computePipeline, WGPUStringView label)
try: (wgpuComputePipelineSetLabel:=dll.wgpuComputePipelineSetLabel).restype, wgpuComputePipelineSetLabel.argtypes = None, [WGPUComputePipeline, WGPUStringView]
except AttributeError: pass

# void wgpuComputePipelineAddRef(WGPUComputePipeline computePipeline)
try: (wgpuComputePipelineAddRef:=dll.wgpuComputePipelineAddRef).restype, wgpuComputePipelineAddRef.argtypes = None, [WGPUComputePipeline]
except AttributeError: pass

# void wgpuComputePipelineRelease(WGPUComputePipeline computePipeline)
try: (wgpuComputePipelineRelease:=dll.wgpuComputePipelineRelease).restype, wgpuComputePipelineRelease.argtypes = None, [WGPUComputePipeline]
except AttributeError: pass

# WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice device, const WGPUBindGroupDescriptor *descriptor)
try: (wgpuDeviceCreateBindGroup:=dll.wgpuDeviceCreateBindGroup).restype, wgpuDeviceCreateBindGroup.argtypes = WGPUBindGroup, [WGPUDevice, ctypes.POINTER(WGPUBindGroupDescriptor)]
except AttributeError: pass

# WGPUBindGroupLayout wgpuDeviceCreateBindGroupLayout(WGPUDevice device, const WGPUBindGroupLayoutDescriptor *descriptor)
try: (wgpuDeviceCreateBindGroupLayout:=dll.wgpuDeviceCreateBindGroupLayout).restype, wgpuDeviceCreateBindGroupLayout.argtypes = WGPUBindGroupLayout, [WGPUDevice, ctypes.POINTER(WGPUBindGroupLayoutDescriptor)]
except AttributeError: pass

# WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, const WGPUBufferDescriptor *descriptor)
try: (wgpuDeviceCreateBuffer:=dll.wgpuDeviceCreateBuffer).restype, wgpuDeviceCreateBuffer.argtypes = WGPUBuffer, [WGPUDevice, ctypes.POINTER(WGPUBufferDescriptor)]
except AttributeError: pass

# WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice device, const WGPUCommandEncoderDescriptor *descriptor)
try: (wgpuDeviceCreateCommandEncoder:=dll.wgpuDeviceCreateCommandEncoder).restype, wgpuDeviceCreateCommandEncoder.argtypes = WGPUCommandEncoder, [WGPUDevice, ctypes.POINTER(WGPUCommandEncoderDescriptor)]
except AttributeError: pass

# WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice device, const WGPUComputePipelineDescriptor *descriptor)
try: (wgpuDeviceCreateComputePipeline:=dll.wgpuDeviceCreateComputePipeline).restype, wgpuDeviceCreateComputePipeline.argtypes = WGPUComputePipeline, [WGPUDevice, ctypes.POINTER(WGPUComputePipelineDescriptor)]
except AttributeError: pass

# void wgpuDeviceCreateComputePipelineAsync(WGPUDevice device, const WGPUComputePipelineDescriptor *descriptor, WGPUCreateComputePipelineAsyncCallback callback, void *userdata)
try: (wgpuDeviceCreateComputePipelineAsync:=dll.wgpuDeviceCreateComputePipelineAsync).restype, wgpuDeviceCreateComputePipelineAsync.argtypes = None, [WGPUDevice, ctypes.POINTER(WGPUComputePipelineDescriptor), WGPUCreateComputePipelineAsyncCallback, ctypes.c_void_p]
except AttributeError: pass

# WGPUFuture wgpuDeviceCreateComputePipelineAsync2(WGPUDevice device, const WGPUComputePipelineDescriptor *descriptor, WGPUCreateComputePipelineAsyncCallbackInfo2 callbackInfo)
try: (wgpuDeviceCreateComputePipelineAsync2:=dll.wgpuDeviceCreateComputePipelineAsync2).restype, wgpuDeviceCreateComputePipelineAsync2.argtypes = WGPUFuture, [WGPUDevice, ctypes.POINTER(WGPUComputePipelineDescriptor), WGPUCreateComputePipelineAsyncCallbackInfo2]
except AttributeError: pass

# WGPUFuture wgpuDeviceCreateComputePipelineAsyncF(WGPUDevice device, const WGPUComputePipelineDescriptor *descriptor, WGPUCreateComputePipelineAsyncCallbackInfo callbackInfo)
try: (wgpuDeviceCreateComputePipelineAsyncF:=dll.wgpuDeviceCreateComputePipelineAsyncF).restype, wgpuDeviceCreateComputePipelineAsyncF.argtypes = WGPUFuture, [WGPUDevice, ctypes.POINTER(WGPUComputePipelineDescriptor), WGPUCreateComputePipelineAsyncCallbackInfo]
except AttributeError: pass

# WGPUBuffer wgpuDeviceCreateErrorBuffer(WGPUDevice device, const WGPUBufferDescriptor *descriptor)
try: (wgpuDeviceCreateErrorBuffer:=dll.wgpuDeviceCreateErrorBuffer).restype, wgpuDeviceCreateErrorBuffer.argtypes = WGPUBuffer, [WGPUDevice, ctypes.POINTER(WGPUBufferDescriptor)]
except AttributeError: pass

# WGPUExternalTexture wgpuDeviceCreateErrorExternalTexture(WGPUDevice device)
try: (wgpuDeviceCreateErrorExternalTexture:=dll.wgpuDeviceCreateErrorExternalTexture).restype, wgpuDeviceCreateErrorExternalTexture.argtypes = WGPUExternalTexture, [WGPUDevice]
except AttributeError: pass

# WGPUShaderModule wgpuDeviceCreateErrorShaderModule(WGPUDevice device, const WGPUShaderModuleDescriptor *descriptor, WGPUStringView errorMessage)
try: (wgpuDeviceCreateErrorShaderModule:=dll.wgpuDeviceCreateErrorShaderModule).restype, wgpuDeviceCreateErrorShaderModule.argtypes = WGPUShaderModule, [WGPUDevice, ctypes.POINTER(WGPUShaderModuleDescriptor), WGPUStringView]
except AttributeError: pass

# WGPUTexture wgpuDeviceCreateErrorTexture(WGPUDevice device, const WGPUTextureDescriptor *descriptor)
try: (wgpuDeviceCreateErrorTexture:=dll.wgpuDeviceCreateErrorTexture).restype, wgpuDeviceCreateErrorTexture.argtypes = WGPUTexture, [WGPUDevice, ctypes.POINTER(WGPUTextureDescriptor)]
except AttributeError: pass

# WGPUExternalTexture wgpuDeviceCreateExternalTexture(WGPUDevice device, const WGPUExternalTextureDescriptor *externalTextureDescriptor)
try: (wgpuDeviceCreateExternalTexture:=dll.wgpuDeviceCreateExternalTexture).restype, wgpuDeviceCreateExternalTexture.argtypes = WGPUExternalTexture, [WGPUDevice, ctypes.POINTER(WGPUExternalTextureDescriptor)]
except AttributeError: pass

# WGPUPipelineLayout wgpuDeviceCreatePipelineLayout(WGPUDevice device, const WGPUPipelineLayoutDescriptor *descriptor)
try: (wgpuDeviceCreatePipelineLayout:=dll.wgpuDeviceCreatePipelineLayout).restype, wgpuDeviceCreatePipelineLayout.argtypes = WGPUPipelineLayout, [WGPUDevice, ctypes.POINTER(WGPUPipelineLayoutDescriptor)]
except AttributeError: pass

# WGPUQuerySet wgpuDeviceCreateQuerySet(WGPUDevice device, const WGPUQuerySetDescriptor *descriptor)
try: (wgpuDeviceCreateQuerySet:=dll.wgpuDeviceCreateQuerySet).restype, wgpuDeviceCreateQuerySet.argtypes = WGPUQuerySet, [WGPUDevice, ctypes.POINTER(WGPUQuerySetDescriptor)]
except AttributeError: pass

# WGPURenderBundleEncoder wgpuDeviceCreateRenderBundleEncoder(WGPUDevice device, const WGPURenderBundleEncoderDescriptor *descriptor)
try: (wgpuDeviceCreateRenderBundleEncoder:=dll.wgpuDeviceCreateRenderBundleEncoder).restype, wgpuDeviceCreateRenderBundleEncoder.argtypes = WGPURenderBundleEncoder, [WGPUDevice, ctypes.POINTER(WGPURenderBundleEncoderDescriptor)]
except AttributeError: pass

# WGPURenderPipeline wgpuDeviceCreateRenderPipeline(WGPUDevice device, const WGPURenderPipelineDescriptor *descriptor)
try: (wgpuDeviceCreateRenderPipeline:=dll.wgpuDeviceCreateRenderPipeline).restype, wgpuDeviceCreateRenderPipeline.argtypes = WGPURenderPipeline, [WGPUDevice, ctypes.POINTER(WGPURenderPipelineDescriptor)]
except AttributeError: pass

# void wgpuDeviceCreateRenderPipelineAsync(WGPUDevice device, const WGPURenderPipelineDescriptor *descriptor, WGPUCreateRenderPipelineAsyncCallback callback, void *userdata)
try: (wgpuDeviceCreateRenderPipelineAsync:=dll.wgpuDeviceCreateRenderPipelineAsync).restype, wgpuDeviceCreateRenderPipelineAsync.argtypes = None, [WGPUDevice, ctypes.POINTER(WGPURenderPipelineDescriptor), WGPUCreateRenderPipelineAsyncCallback, ctypes.c_void_p]
except AttributeError: pass

# WGPUFuture wgpuDeviceCreateRenderPipelineAsync2(WGPUDevice device, const WGPURenderPipelineDescriptor *descriptor, WGPUCreateRenderPipelineAsyncCallbackInfo2 callbackInfo)
try: (wgpuDeviceCreateRenderPipelineAsync2:=dll.wgpuDeviceCreateRenderPipelineAsync2).restype, wgpuDeviceCreateRenderPipelineAsync2.argtypes = WGPUFuture, [WGPUDevice, ctypes.POINTER(WGPURenderPipelineDescriptor), WGPUCreateRenderPipelineAsyncCallbackInfo2]
except AttributeError: pass

# WGPUFuture wgpuDeviceCreateRenderPipelineAsyncF(WGPUDevice device, const WGPURenderPipelineDescriptor *descriptor, WGPUCreateRenderPipelineAsyncCallbackInfo callbackInfo)
try: (wgpuDeviceCreateRenderPipelineAsyncF:=dll.wgpuDeviceCreateRenderPipelineAsyncF).restype, wgpuDeviceCreateRenderPipelineAsyncF.argtypes = WGPUFuture, [WGPUDevice, ctypes.POINTER(WGPURenderPipelineDescriptor), WGPUCreateRenderPipelineAsyncCallbackInfo]
except AttributeError: pass

# WGPUSampler wgpuDeviceCreateSampler(WGPUDevice device, const WGPUSamplerDescriptor *descriptor)
try: (wgpuDeviceCreateSampler:=dll.wgpuDeviceCreateSampler).restype, wgpuDeviceCreateSampler.argtypes = WGPUSampler, [WGPUDevice, ctypes.POINTER(WGPUSamplerDescriptor)]
except AttributeError: pass

# WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice device, const WGPUShaderModuleDescriptor *descriptor)
try: (wgpuDeviceCreateShaderModule:=dll.wgpuDeviceCreateShaderModule).restype, wgpuDeviceCreateShaderModule.argtypes = WGPUShaderModule, [WGPUDevice, ctypes.POINTER(WGPUShaderModuleDescriptor)]
except AttributeError: pass

# WGPUTexture wgpuDeviceCreateTexture(WGPUDevice device, const WGPUTextureDescriptor *descriptor)
try: (wgpuDeviceCreateTexture:=dll.wgpuDeviceCreateTexture).restype, wgpuDeviceCreateTexture.argtypes = WGPUTexture, [WGPUDevice, ctypes.POINTER(WGPUTextureDescriptor)]
except AttributeError: pass

# void wgpuDeviceDestroy(WGPUDevice device)
try: (wgpuDeviceDestroy:=dll.wgpuDeviceDestroy).restype, wgpuDeviceDestroy.argtypes = None, [WGPUDevice]
except AttributeError: pass

# void wgpuDeviceForceLoss(WGPUDevice device, WGPUDeviceLostReason type, WGPUStringView message)
try: (wgpuDeviceForceLoss:=dll.wgpuDeviceForceLoss).restype, wgpuDeviceForceLoss.argtypes = None, [WGPUDevice, WGPUDeviceLostReason, WGPUStringView]
except AttributeError: pass

# WGPUStatus wgpuDeviceGetAHardwareBufferProperties(WGPUDevice device, void *handle, WGPUAHardwareBufferProperties *properties)
try: (wgpuDeviceGetAHardwareBufferProperties:=dll.wgpuDeviceGetAHardwareBufferProperties).restype, wgpuDeviceGetAHardwareBufferProperties.argtypes = WGPUStatus, [WGPUDevice, ctypes.c_void_p, ctypes.POINTER(WGPUAHardwareBufferProperties)]
except AttributeError: pass

# WGPUAdapter wgpuDeviceGetAdapter(WGPUDevice device)
try: (wgpuDeviceGetAdapter:=dll.wgpuDeviceGetAdapter).restype, wgpuDeviceGetAdapter.argtypes = WGPUAdapter, [WGPUDevice]
except AttributeError: pass

# WGPUStatus wgpuDeviceGetAdapterInfo(WGPUDevice device, WGPUAdapterInfo *adapterInfo)
try: (wgpuDeviceGetAdapterInfo:=dll.wgpuDeviceGetAdapterInfo).restype, wgpuDeviceGetAdapterInfo.argtypes = WGPUStatus, [WGPUDevice, ctypes.POINTER(WGPUAdapterInfo)]
except AttributeError: pass

# void wgpuDeviceGetFeatures(WGPUDevice device, WGPUSupportedFeatures *features)
try: (wgpuDeviceGetFeatures:=dll.wgpuDeviceGetFeatures).restype, wgpuDeviceGetFeatures.argtypes = None, [WGPUDevice, ctypes.POINTER(WGPUSupportedFeatures)]
except AttributeError: pass

# WGPUStatus wgpuDeviceGetLimits(WGPUDevice device, WGPUSupportedLimits *limits)
try: (wgpuDeviceGetLimits:=dll.wgpuDeviceGetLimits).restype, wgpuDeviceGetLimits.argtypes = WGPUStatus, [WGPUDevice, ctypes.POINTER(WGPUSupportedLimits)]
except AttributeError: pass

# WGPUFuture wgpuDeviceGetLostFuture(WGPUDevice device)
try: (wgpuDeviceGetLostFuture:=dll.wgpuDeviceGetLostFuture).restype, wgpuDeviceGetLostFuture.argtypes = WGPUFuture, [WGPUDevice]
except AttributeError: pass

# WGPUQueue wgpuDeviceGetQueue(WGPUDevice device)
try: (wgpuDeviceGetQueue:=dll.wgpuDeviceGetQueue).restype, wgpuDeviceGetQueue.argtypes = WGPUQueue, [WGPUDevice]
except AttributeError: pass

# WGPUBool wgpuDeviceHasFeature(WGPUDevice device, WGPUFeatureName feature)
try: (wgpuDeviceHasFeature:=dll.wgpuDeviceHasFeature).restype, wgpuDeviceHasFeature.argtypes = WGPUBool, [WGPUDevice, WGPUFeatureName]
except AttributeError: pass

# WGPUSharedBufferMemory wgpuDeviceImportSharedBufferMemory(WGPUDevice device, const WGPUSharedBufferMemoryDescriptor *descriptor)
try: (wgpuDeviceImportSharedBufferMemory:=dll.wgpuDeviceImportSharedBufferMemory).restype, wgpuDeviceImportSharedBufferMemory.argtypes = WGPUSharedBufferMemory, [WGPUDevice, ctypes.POINTER(WGPUSharedBufferMemoryDescriptor)]
except AttributeError: pass

# WGPUSharedFence wgpuDeviceImportSharedFence(WGPUDevice device, const WGPUSharedFenceDescriptor *descriptor)
try: (wgpuDeviceImportSharedFence:=dll.wgpuDeviceImportSharedFence).restype, wgpuDeviceImportSharedFence.argtypes = WGPUSharedFence, [WGPUDevice, ctypes.POINTER(WGPUSharedFenceDescriptor)]
except AttributeError: pass

# WGPUSharedTextureMemory wgpuDeviceImportSharedTextureMemory(WGPUDevice device, const WGPUSharedTextureMemoryDescriptor *descriptor)
try: (wgpuDeviceImportSharedTextureMemory:=dll.wgpuDeviceImportSharedTextureMemory).restype, wgpuDeviceImportSharedTextureMemory.argtypes = WGPUSharedTextureMemory, [WGPUDevice, ctypes.POINTER(WGPUSharedTextureMemoryDescriptor)]
except AttributeError: pass

# void wgpuDeviceInjectError(WGPUDevice device, WGPUErrorType type, WGPUStringView message)
try: (wgpuDeviceInjectError:=dll.wgpuDeviceInjectError).restype, wgpuDeviceInjectError.argtypes = None, [WGPUDevice, WGPUErrorType, WGPUStringView]
except AttributeError: pass

# void wgpuDevicePopErrorScope(WGPUDevice device, WGPUErrorCallback oldCallback, void *userdata)
try: (wgpuDevicePopErrorScope:=dll.wgpuDevicePopErrorScope).restype, wgpuDevicePopErrorScope.argtypes = None, [WGPUDevice, WGPUErrorCallback, ctypes.c_void_p]
except AttributeError: pass

# WGPUFuture wgpuDevicePopErrorScope2(WGPUDevice device, WGPUPopErrorScopeCallbackInfo2 callbackInfo)
try: (wgpuDevicePopErrorScope2:=dll.wgpuDevicePopErrorScope2).restype, wgpuDevicePopErrorScope2.argtypes = WGPUFuture, [WGPUDevice, WGPUPopErrorScopeCallbackInfo2]
except AttributeError: pass

# WGPUFuture wgpuDevicePopErrorScopeF(WGPUDevice device, WGPUPopErrorScopeCallbackInfo callbackInfo)
try: (wgpuDevicePopErrorScopeF:=dll.wgpuDevicePopErrorScopeF).restype, wgpuDevicePopErrorScopeF.argtypes = WGPUFuture, [WGPUDevice, WGPUPopErrorScopeCallbackInfo]
except AttributeError: pass

# void wgpuDevicePushErrorScope(WGPUDevice device, WGPUErrorFilter filter)
try: (wgpuDevicePushErrorScope:=dll.wgpuDevicePushErrorScope).restype, wgpuDevicePushErrorScope.argtypes = None, [WGPUDevice, WGPUErrorFilter]
except AttributeError: pass

# void wgpuDeviceSetLabel(WGPUDevice device, WGPUStringView label)
try: (wgpuDeviceSetLabel:=dll.wgpuDeviceSetLabel).restype, wgpuDeviceSetLabel.argtypes = None, [WGPUDevice, WGPUStringView]
except AttributeError: pass

# void wgpuDeviceSetLoggingCallback(WGPUDevice device, WGPULoggingCallback callback, void *userdata)
try: (wgpuDeviceSetLoggingCallback:=dll.wgpuDeviceSetLoggingCallback).restype, wgpuDeviceSetLoggingCallback.argtypes = None, [WGPUDevice, WGPULoggingCallback, ctypes.c_void_p]
except AttributeError: pass

# void wgpuDeviceTick(WGPUDevice device)
try: (wgpuDeviceTick:=dll.wgpuDeviceTick).restype, wgpuDeviceTick.argtypes = None, [WGPUDevice]
except AttributeError: pass

# void wgpuDeviceValidateTextureDescriptor(WGPUDevice device, const WGPUTextureDescriptor *descriptor)
try: (wgpuDeviceValidateTextureDescriptor:=dll.wgpuDeviceValidateTextureDescriptor).restype, wgpuDeviceValidateTextureDescriptor.argtypes = None, [WGPUDevice, ctypes.POINTER(WGPUTextureDescriptor)]
except AttributeError: pass

# void wgpuDeviceAddRef(WGPUDevice device)
try: (wgpuDeviceAddRef:=dll.wgpuDeviceAddRef).restype, wgpuDeviceAddRef.argtypes = None, [WGPUDevice]
except AttributeError: pass

# void wgpuDeviceRelease(WGPUDevice device)
try: (wgpuDeviceRelease:=dll.wgpuDeviceRelease).restype, wgpuDeviceRelease.argtypes = None, [WGPUDevice]
except AttributeError: pass

# void wgpuExternalTextureDestroy(WGPUExternalTexture externalTexture)
try: (wgpuExternalTextureDestroy:=dll.wgpuExternalTextureDestroy).restype, wgpuExternalTextureDestroy.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

# void wgpuExternalTextureExpire(WGPUExternalTexture externalTexture)
try: (wgpuExternalTextureExpire:=dll.wgpuExternalTextureExpire).restype, wgpuExternalTextureExpire.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

# void wgpuExternalTextureRefresh(WGPUExternalTexture externalTexture)
try: (wgpuExternalTextureRefresh:=dll.wgpuExternalTextureRefresh).restype, wgpuExternalTextureRefresh.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

# void wgpuExternalTextureSetLabel(WGPUExternalTexture externalTexture, WGPUStringView label)
try: (wgpuExternalTextureSetLabel:=dll.wgpuExternalTextureSetLabel).restype, wgpuExternalTextureSetLabel.argtypes = None, [WGPUExternalTexture, WGPUStringView]
except AttributeError: pass

# void wgpuExternalTextureAddRef(WGPUExternalTexture externalTexture)
try: (wgpuExternalTextureAddRef:=dll.wgpuExternalTextureAddRef).restype, wgpuExternalTextureAddRef.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

# void wgpuExternalTextureRelease(WGPUExternalTexture externalTexture)
try: (wgpuExternalTextureRelease:=dll.wgpuExternalTextureRelease).restype, wgpuExternalTextureRelease.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

# WGPUSurface wgpuInstanceCreateSurface(WGPUInstance instance, const WGPUSurfaceDescriptor *descriptor)
try: (wgpuInstanceCreateSurface:=dll.wgpuInstanceCreateSurface).restype, wgpuInstanceCreateSurface.argtypes = WGPUSurface, [WGPUInstance, ctypes.POINTER(WGPUSurfaceDescriptor)]
except AttributeError: pass

# size_t wgpuInstanceEnumerateWGSLLanguageFeatures(WGPUInstance instance, WGPUWGSLFeatureName *features)
try: (wgpuInstanceEnumerateWGSLLanguageFeatures:=dll.wgpuInstanceEnumerateWGSLLanguageFeatures).restype, wgpuInstanceEnumerateWGSLLanguageFeatures.argtypes = size_t, [WGPUInstance, ctypes.POINTER(WGPUWGSLFeatureName)]
except AttributeError: pass

# WGPUBool wgpuInstanceHasWGSLLanguageFeature(WGPUInstance instance, WGPUWGSLFeatureName feature)
try: (wgpuInstanceHasWGSLLanguageFeature:=dll.wgpuInstanceHasWGSLLanguageFeature).restype, wgpuInstanceHasWGSLLanguageFeature.argtypes = WGPUBool, [WGPUInstance, WGPUWGSLFeatureName]
except AttributeError: pass

# void wgpuInstanceProcessEvents(WGPUInstance instance)
try: (wgpuInstanceProcessEvents:=dll.wgpuInstanceProcessEvents).restype, wgpuInstanceProcessEvents.argtypes = None, [WGPUInstance]
except AttributeError: pass

# void wgpuInstanceRequestAdapter(WGPUInstance instance, const WGPURequestAdapterOptions *options, WGPURequestAdapterCallback callback, void *userdata)
try: (wgpuInstanceRequestAdapter:=dll.wgpuInstanceRequestAdapter).restype, wgpuInstanceRequestAdapter.argtypes = None, [WGPUInstance, ctypes.POINTER(WGPURequestAdapterOptions), WGPURequestAdapterCallback, ctypes.c_void_p]
except AttributeError: pass

# WGPUFuture wgpuInstanceRequestAdapter2(WGPUInstance instance, const WGPURequestAdapterOptions *options, WGPURequestAdapterCallbackInfo2 callbackInfo)
try: (wgpuInstanceRequestAdapter2:=dll.wgpuInstanceRequestAdapter2).restype, wgpuInstanceRequestAdapter2.argtypes = WGPUFuture, [WGPUInstance, ctypes.POINTER(WGPURequestAdapterOptions), WGPURequestAdapterCallbackInfo2]
except AttributeError: pass

# WGPUFuture wgpuInstanceRequestAdapterF(WGPUInstance instance, const WGPURequestAdapterOptions *options, WGPURequestAdapterCallbackInfo callbackInfo)
try: (wgpuInstanceRequestAdapterF:=dll.wgpuInstanceRequestAdapterF).restype, wgpuInstanceRequestAdapterF.argtypes = WGPUFuture, [WGPUInstance, ctypes.POINTER(WGPURequestAdapterOptions), WGPURequestAdapterCallbackInfo]
except AttributeError: pass

# WGPUWaitStatus wgpuInstanceWaitAny(WGPUInstance instance, size_t futureCount, WGPUFutureWaitInfo *futures, uint64_t timeoutNS)
try: (wgpuInstanceWaitAny:=dll.wgpuInstanceWaitAny).restype, wgpuInstanceWaitAny.argtypes = WGPUWaitStatus, [WGPUInstance, size_t, ctypes.POINTER(WGPUFutureWaitInfo), uint64_t]
except AttributeError: pass

# void wgpuInstanceAddRef(WGPUInstance instance)
try: (wgpuInstanceAddRef:=dll.wgpuInstanceAddRef).restype, wgpuInstanceAddRef.argtypes = None, [WGPUInstance]
except AttributeError: pass

# void wgpuInstanceRelease(WGPUInstance instance)
try: (wgpuInstanceRelease:=dll.wgpuInstanceRelease).restype, wgpuInstanceRelease.argtypes = None, [WGPUInstance]
except AttributeError: pass

# void wgpuPipelineLayoutSetLabel(WGPUPipelineLayout pipelineLayout, WGPUStringView label)
try: (wgpuPipelineLayoutSetLabel:=dll.wgpuPipelineLayoutSetLabel).restype, wgpuPipelineLayoutSetLabel.argtypes = None, [WGPUPipelineLayout, WGPUStringView]
except AttributeError: pass

# void wgpuPipelineLayoutAddRef(WGPUPipelineLayout pipelineLayout)
try: (wgpuPipelineLayoutAddRef:=dll.wgpuPipelineLayoutAddRef).restype, wgpuPipelineLayoutAddRef.argtypes = None, [WGPUPipelineLayout]
except AttributeError: pass

# void wgpuPipelineLayoutRelease(WGPUPipelineLayout pipelineLayout)
try: (wgpuPipelineLayoutRelease:=dll.wgpuPipelineLayoutRelease).restype, wgpuPipelineLayoutRelease.argtypes = None, [WGPUPipelineLayout]
except AttributeError: pass

# void wgpuQuerySetDestroy(WGPUQuerySet querySet)
try: (wgpuQuerySetDestroy:=dll.wgpuQuerySetDestroy).restype, wgpuQuerySetDestroy.argtypes = None, [WGPUQuerySet]
except AttributeError: pass

# uint32_t wgpuQuerySetGetCount(WGPUQuerySet querySet)
try: (wgpuQuerySetGetCount:=dll.wgpuQuerySetGetCount).restype, wgpuQuerySetGetCount.argtypes = uint32_t, [WGPUQuerySet]
except AttributeError: pass

# WGPUQueryType wgpuQuerySetGetType(WGPUQuerySet querySet)
try: (wgpuQuerySetGetType:=dll.wgpuQuerySetGetType).restype, wgpuQuerySetGetType.argtypes = WGPUQueryType, [WGPUQuerySet]
except AttributeError: pass

# void wgpuQuerySetSetLabel(WGPUQuerySet querySet, WGPUStringView label)
try: (wgpuQuerySetSetLabel:=dll.wgpuQuerySetSetLabel).restype, wgpuQuerySetSetLabel.argtypes = None, [WGPUQuerySet, WGPUStringView]
except AttributeError: pass

# void wgpuQuerySetAddRef(WGPUQuerySet querySet)
try: (wgpuQuerySetAddRef:=dll.wgpuQuerySetAddRef).restype, wgpuQuerySetAddRef.argtypes = None, [WGPUQuerySet]
except AttributeError: pass

# void wgpuQuerySetRelease(WGPUQuerySet querySet)
try: (wgpuQuerySetRelease:=dll.wgpuQuerySetRelease).restype, wgpuQuerySetRelease.argtypes = None, [WGPUQuerySet]
except AttributeError: pass

# void wgpuQueueCopyExternalTextureForBrowser(WGPUQueue queue, const WGPUImageCopyExternalTexture *source, const WGPUImageCopyTexture *destination, const WGPUExtent3D *copySize, const WGPUCopyTextureForBrowserOptions *options)
try: (wgpuQueueCopyExternalTextureForBrowser:=dll.wgpuQueueCopyExternalTextureForBrowser).restype, wgpuQueueCopyExternalTextureForBrowser.argtypes = None, [WGPUQueue, ctypes.POINTER(WGPUImageCopyExternalTexture), ctypes.POINTER(WGPUImageCopyTexture), ctypes.POINTER(WGPUExtent3D), ctypes.POINTER(WGPUCopyTextureForBrowserOptions)]
except AttributeError: pass

# void wgpuQueueCopyTextureForBrowser(WGPUQueue queue, const WGPUImageCopyTexture *source, const WGPUImageCopyTexture *destination, const WGPUExtent3D *copySize, const WGPUCopyTextureForBrowserOptions *options)
try: (wgpuQueueCopyTextureForBrowser:=dll.wgpuQueueCopyTextureForBrowser).restype, wgpuQueueCopyTextureForBrowser.argtypes = None, [WGPUQueue, ctypes.POINTER(WGPUImageCopyTexture), ctypes.POINTER(WGPUImageCopyTexture), ctypes.POINTER(WGPUExtent3D), ctypes.POINTER(WGPUCopyTextureForBrowserOptions)]
except AttributeError: pass

# void wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallback callback, void *userdata)
try: (wgpuQueueOnSubmittedWorkDone:=dll.wgpuQueueOnSubmittedWorkDone).restype, wgpuQueueOnSubmittedWorkDone.argtypes = None, [WGPUQueue, WGPUQueueWorkDoneCallback, ctypes.c_void_p]
except AttributeError: pass

# WGPUFuture wgpuQueueOnSubmittedWorkDone2(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo2 callbackInfo)
try: (wgpuQueueOnSubmittedWorkDone2:=dll.wgpuQueueOnSubmittedWorkDone2).restype, wgpuQueueOnSubmittedWorkDone2.argtypes = WGPUFuture, [WGPUQueue, WGPUQueueWorkDoneCallbackInfo2]
except AttributeError: pass

# WGPUFuture wgpuQueueOnSubmittedWorkDoneF(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callbackInfo)
try: (wgpuQueueOnSubmittedWorkDoneF:=dll.wgpuQueueOnSubmittedWorkDoneF).restype, wgpuQueueOnSubmittedWorkDoneF.argtypes = WGPUFuture, [WGPUQueue, WGPUQueueWorkDoneCallbackInfo]
except AttributeError: pass

# void wgpuQueueSetLabel(WGPUQueue queue, WGPUStringView label)
try: (wgpuQueueSetLabel:=dll.wgpuQueueSetLabel).restype, wgpuQueueSetLabel.argtypes = None, [WGPUQueue, WGPUStringView]
except AttributeError: pass

# void wgpuQueueSubmit(WGPUQueue queue, size_t commandCount, const WGPUCommandBuffer *commands)
try: (wgpuQueueSubmit:=dll.wgpuQueueSubmit).restype, wgpuQueueSubmit.argtypes = None, [WGPUQueue, size_t, ctypes.POINTER(WGPUCommandBuffer)]
except AttributeError: pass

# void wgpuQueueWriteBuffer(WGPUQueue queue, WGPUBuffer buffer, uint64_t bufferOffset, const void *data, size_t size)
try: (wgpuQueueWriteBuffer:=dll.wgpuQueueWriteBuffer).restype, wgpuQueueWriteBuffer.argtypes = None, [WGPUQueue, WGPUBuffer, uint64_t, ctypes.c_void_p, size_t]
except AttributeError: pass

# void wgpuQueueWriteTexture(WGPUQueue queue, const WGPUImageCopyTexture *destination, const void *data, size_t dataSize, const WGPUTextureDataLayout *dataLayout, const WGPUExtent3D *writeSize)
try: (wgpuQueueWriteTexture:=dll.wgpuQueueWriteTexture).restype, wgpuQueueWriteTexture.argtypes = None, [WGPUQueue, ctypes.POINTER(WGPUImageCopyTexture), ctypes.c_void_p, size_t, ctypes.POINTER(WGPUTextureDataLayout), ctypes.POINTER(WGPUExtent3D)]
except AttributeError: pass

# void wgpuQueueAddRef(WGPUQueue queue)
try: (wgpuQueueAddRef:=dll.wgpuQueueAddRef).restype, wgpuQueueAddRef.argtypes = None, [WGPUQueue]
except AttributeError: pass

# void wgpuQueueRelease(WGPUQueue queue)
try: (wgpuQueueRelease:=dll.wgpuQueueRelease).restype, wgpuQueueRelease.argtypes = None, [WGPUQueue]
except AttributeError: pass

# void wgpuRenderBundleSetLabel(WGPURenderBundle renderBundle, WGPUStringView label)
try: (wgpuRenderBundleSetLabel:=dll.wgpuRenderBundleSetLabel).restype, wgpuRenderBundleSetLabel.argtypes = None, [WGPURenderBundle, WGPUStringView]
except AttributeError: pass

# void wgpuRenderBundleAddRef(WGPURenderBundle renderBundle)
try: (wgpuRenderBundleAddRef:=dll.wgpuRenderBundleAddRef).restype, wgpuRenderBundleAddRef.argtypes = None, [WGPURenderBundle]
except AttributeError: pass

# void wgpuRenderBundleRelease(WGPURenderBundle renderBundle)
try: (wgpuRenderBundleRelease:=dll.wgpuRenderBundleRelease).restype, wgpuRenderBundleRelease.argtypes = None, [WGPURenderBundle]
except AttributeError: pass

# void wgpuRenderBundleEncoderDraw(WGPURenderBundleEncoder renderBundleEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
try: (wgpuRenderBundleEncoderDraw:=dll.wgpuRenderBundleEncoderDraw).restype, wgpuRenderBundleEncoderDraw.argtypes = None, [WGPURenderBundleEncoder, uint32_t, uint32_t, uint32_t, uint32_t]
except AttributeError: pass

# void wgpuRenderBundleEncoderDrawIndexed(WGPURenderBundleEncoder renderBundleEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance)
try: (wgpuRenderBundleEncoderDrawIndexed:=dll.wgpuRenderBundleEncoderDrawIndexed).restype, wgpuRenderBundleEncoderDrawIndexed.argtypes = None, [WGPURenderBundleEncoder, uint32_t, uint32_t, uint32_t, int32_t, uint32_t]
except AttributeError: pass

# void wgpuRenderBundleEncoderDrawIndexedIndirect(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
try: (wgpuRenderBundleEncoderDrawIndexedIndirect:=dll.wgpuRenderBundleEncoderDrawIndexedIndirect).restype, wgpuRenderBundleEncoderDrawIndexedIndirect.argtypes = None, [WGPURenderBundleEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

# void wgpuRenderBundleEncoderDrawIndirect(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
try: (wgpuRenderBundleEncoderDrawIndirect:=dll.wgpuRenderBundleEncoderDrawIndirect).restype, wgpuRenderBundleEncoderDrawIndirect.argtypes = None, [WGPURenderBundleEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

# WGPURenderBundle wgpuRenderBundleEncoderFinish(WGPURenderBundleEncoder renderBundleEncoder, const WGPURenderBundleDescriptor *descriptor)
try: (wgpuRenderBundleEncoderFinish:=dll.wgpuRenderBundleEncoderFinish).restype, wgpuRenderBundleEncoderFinish.argtypes = WGPURenderBundle, [WGPURenderBundleEncoder, ctypes.POINTER(WGPURenderBundleDescriptor)]
except AttributeError: pass

# void wgpuRenderBundleEncoderInsertDebugMarker(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView markerLabel)
try: (wgpuRenderBundleEncoderInsertDebugMarker:=dll.wgpuRenderBundleEncoderInsertDebugMarker).restype, wgpuRenderBundleEncoderInsertDebugMarker.argtypes = None, [WGPURenderBundleEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuRenderBundleEncoderPopDebugGroup(WGPURenderBundleEncoder renderBundleEncoder)
try: (wgpuRenderBundleEncoderPopDebugGroup:=dll.wgpuRenderBundleEncoderPopDebugGroup).restype, wgpuRenderBundleEncoderPopDebugGroup.argtypes = None, [WGPURenderBundleEncoder]
except AttributeError: pass

# void wgpuRenderBundleEncoderPushDebugGroup(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView groupLabel)
try: (wgpuRenderBundleEncoderPushDebugGroup:=dll.wgpuRenderBundleEncoderPushDebugGroup).restype, wgpuRenderBundleEncoderPushDebugGroup.argtypes = None, [WGPURenderBundleEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuRenderBundleEncoderSetBindGroup(WGPURenderBundleEncoder renderBundleEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, const uint32_t *dynamicOffsets)
try: (wgpuRenderBundleEncoderSetBindGroup:=dll.wgpuRenderBundleEncoderSetBindGroup).restype, wgpuRenderBundleEncoderSetBindGroup.argtypes = None, [WGPURenderBundleEncoder, uint32_t, WGPUBindGroup, size_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

# void wgpuRenderBundleEncoderSetIndexBuffer(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size)
try: (wgpuRenderBundleEncoderSetIndexBuffer:=dll.wgpuRenderBundleEncoderSetIndexBuffer).restype, wgpuRenderBundleEncoderSetIndexBuffer.argtypes = None, [WGPURenderBundleEncoder, WGPUBuffer, WGPUIndexFormat, uint64_t, uint64_t]
except AttributeError: pass

# void wgpuRenderBundleEncoderSetLabel(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView label)
try: (wgpuRenderBundleEncoderSetLabel:=dll.wgpuRenderBundleEncoderSetLabel).restype, wgpuRenderBundleEncoderSetLabel.argtypes = None, [WGPURenderBundleEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuRenderBundleEncoderSetPipeline(WGPURenderBundleEncoder renderBundleEncoder, WGPURenderPipeline pipeline)
try: (wgpuRenderBundleEncoderSetPipeline:=dll.wgpuRenderBundleEncoderSetPipeline).restype, wgpuRenderBundleEncoderSetPipeline.argtypes = None, [WGPURenderBundleEncoder, WGPURenderPipeline]
except AttributeError: pass

# void wgpuRenderBundleEncoderSetVertexBuffer(WGPURenderBundleEncoder renderBundleEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size)
try: (wgpuRenderBundleEncoderSetVertexBuffer:=dll.wgpuRenderBundleEncoderSetVertexBuffer).restype, wgpuRenderBundleEncoderSetVertexBuffer.argtypes = None, [WGPURenderBundleEncoder, uint32_t, WGPUBuffer, uint64_t, uint64_t]
except AttributeError: pass

# void wgpuRenderBundleEncoderAddRef(WGPURenderBundleEncoder renderBundleEncoder)
try: (wgpuRenderBundleEncoderAddRef:=dll.wgpuRenderBundleEncoderAddRef).restype, wgpuRenderBundleEncoderAddRef.argtypes = None, [WGPURenderBundleEncoder]
except AttributeError: pass

# void wgpuRenderBundleEncoderRelease(WGPURenderBundleEncoder renderBundleEncoder)
try: (wgpuRenderBundleEncoderRelease:=dll.wgpuRenderBundleEncoderRelease).restype, wgpuRenderBundleEncoderRelease.argtypes = None, [WGPURenderBundleEncoder]
except AttributeError: pass

# void wgpuRenderPassEncoderBeginOcclusionQuery(WGPURenderPassEncoder renderPassEncoder, uint32_t queryIndex)
try: (wgpuRenderPassEncoderBeginOcclusionQuery:=dll.wgpuRenderPassEncoderBeginOcclusionQuery).restype, wgpuRenderPassEncoderBeginOcclusionQuery.argtypes = None, [WGPURenderPassEncoder, uint32_t]
except AttributeError: pass

# void wgpuRenderPassEncoderDraw(WGPURenderPassEncoder renderPassEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
try: (wgpuRenderPassEncoderDraw:=dll.wgpuRenderPassEncoderDraw).restype, wgpuRenderPassEncoderDraw.argtypes = None, [WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, uint32_t]
except AttributeError: pass

# void wgpuRenderPassEncoderDrawIndexed(WGPURenderPassEncoder renderPassEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance)
try: (wgpuRenderPassEncoderDrawIndexed:=dll.wgpuRenderPassEncoderDrawIndexed).restype, wgpuRenderPassEncoderDrawIndexed.argtypes = None, [WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, int32_t, uint32_t]
except AttributeError: pass

# void wgpuRenderPassEncoderDrawIndexedIndirect(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
try: (wgpuRenderPassEncoderDrawIndexedIndirect:=dll.wgpuRenderPassEncoderDrawIndexedIndirect).restype, wgpuRenderPassEncoderDrawIndexedIndirect.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

# void wgpuRenderPassEncoderDrawIndirect(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
try: (wgpuRenderPassEncoderDrawIndirect:=dll.wgpuRenderPassEncoderDrawIndirect).restype, wgpuRenderPassEncoderDrawIndirect.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

# void wgpuRenderPassEncoderEnd(WGPURenderPassEncoder renderPassEncoder)
try: (wgpuRenderPassEncoderEnd:=dll.wgpuRenderPassEncoderEnd).restype, wgpuRenderPassEncoderEnd.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

# void wgpuRenderPassEncoderEndOcclusionQuery(WGPURenderPassEncoder renderPassEncoder)
try: (wgpuRenderPassEncoderEndOcclusionQuery:=dll.wgpuRenderPassEncoderEndOcclusionQuery).restype, wgpuRenderPassEncoderEndOcclusionQuery.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

# void wgpuRenderPassEncoderExecuteBundles(WGPURenderPassEncoder renderPassEncoder, size_t bundleCount, const WGPURenderBundle *bundles)
try: (wgpuRenderPassEncoderExecuteBundles:=dll.wgpuRenderPassEncoderExecuteBundles).restype, wgpuRenderPassEncoderExecuteBundles.argtypes = None, [WGPURenderPassEncoder, size_t, ctypes.POINTER(WGPURenderBundle)]
except AttributeError: pass

# void wgpuRenderPassEncoderInsertDebugMarker(WGPURenderPassEncoder renderPassEncoder, WGPUStringView markerLabel)
try: (wgpuRenderPassEncoderInsertDebugMarker:=dll.wgpuRenderPassEncoderInsertDebugMarker).restype, wgpuRenderPassEncoderInsertDebugMarker.argtypes = None, [WGPURenderPassEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuRenderPassEncoderMultiDrawIndexedIndirect(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset, uint32_t maxDrawCount, WGPUBuffer drawCountBuffer, uint64_t drawCountBufferOffset)
try: (wgpuRenderPassEncoderMultiDrawIndexedIndirect:=dll.wgpuRenderPassEncoderMultiDrawIndexedIndirect).restype, wgpuRenderPassEncoderMultiDrawIndexedIndirect.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, uint64_t, uint32_t, WGPUBuffer, uint64_t]
except AttributeError: pass

# void wgpuRenderPassEncoderMultiDrawIndirect(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset, uint32_t maxDrawCount, WGPUBuffer drawCountBuffer, uint64_t drawCountBufferOffset)
try: (wgpuRenderPassEncoderMultiDrawIndirect:=dll.wgpuRenderPassEncoderMultiDrawIndirect).restype, wgpuRenderPassEncoderMultiDrawIndirect.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, uint64_t, uint32_t, WGPUBuffer, uint64_t]
except AttributeError: pass

# void wgpuRenderPassEncoderPixelLocalStorageBarrier(WGPURenderPassEncoder renderPassEncoder)
try: (wgpuRenderPassEncoderPixelLocalStorageBarrier:=dll.wgpuRenderPassEncoderPixelLocalStorageBarrier).restype, wgpuRenderPassEncoderPixelLocalStorageBarrier.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

# void wgpuRenderPassEncoderPopDebugGroup(WGPURenderPassEncoder renderPassEncoder)
try: (wgpuRenderPassEncoderPopDebugGroup:=dll.wgpuRenderPassEncoderPopDebugGroup).restype, wgpuRenderPassEncoderPopDebugGroup.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

# void wgpuRenderPassEncoderPushDebugGroup(WGPURenderPassEncoder renderPassEncoder, WGPUStringView groupLabel)
try: (wgpuRenderPassEncoderPushDebugGroup:=dll.wgpuRenderPassEncoderPushDebugGroup).restype, wgpuRenderPassEncoderPushDebugGroup.argtypes = None, [WGPURenderPassEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuRenderPassEncoderSetBindGroup(WGPURenderPassEncoder renderPassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, const uint32_t *dynamicOffsets)
try: (wgpuRenderPassEncoderSetBindGroup:=dll.wgpuRenderPassEncoderSetBindGroup).restype, wgpuRenderPassEncoderSetBindGroup.argtypes = None, [WGPURenderPassEncoder, uint32_t, WGPUBindGroup, size_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

# void wgpuRenderPassEncoderSetBlendConstant(WGPURenderPassEncoder renderPassEncoder, const WGPUColor *color)
try: (wgpuRenderPassEncoderSetBlendConstant:=dll.wgpuRenderPassEncoderSetBlendConstant).restype, wgpuRenderPassEncoderSetBlendConstant.argtypes = None, [WGPURenderPassEncoder, ctypes.POINTER(WGPUColor)]
except AttributeError: pass

# void wgpuRenderPassEncoderSetIndexBuffer(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size)
try: (wgpuRenderPassEncoderSetIndexBuffer:=dll.wgpuRenderPassEncoderSetIndexBuffer).restype, wgpuRenderPassEncoderSetIndexBuffer.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, WGPUIndexFormat, uint64_t, uint64_t]
except AttributeError: pass

# void wgpuRenderPassEncoderSetLabel(WGPURenderPassEncoder renderPassEncoder, WGPUStringView label)
try: (wgpuRenderPassEncoderSetLabel:=dll.wgpuRenderPassEncoderSetLabel).restype, wgpuRenderPassEncoderSetLabel.argtypes = None, [WGPURenderPassEncoder, WGPUStringView]
except AttributeError: pass

# void wgpuRenderPassEncoderSetPipeline(WGPURenderPassEncoder renderPassEncoder, WGPURenderPipeline pipeline)
try: (wgpuRenderPassEncoderSetPipeline:=dll.wgpuRenderPassEncoderSetPipeline).restype, wgpuRenderPassEncoderSetPipeline.argtypes = None, [WGPURenderPassEncoder, WGPURenderPipeline]
except AttributeError: pass

# void wgpuRenderPassEncoderSetScissorRect(WGPURenderPassEncoder renderPassEncoder, uint32_t x, uint32_t y, uint32_t width, uint32_t height)
try: (wgpuRenderPassEncoderSetScissorRect:=dll.wgpuRenderPassEncoderSetScissorRect).restype, wgpuRenderPassEncoderSetScissorRect.argtypes = None, [WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, uint32_t]
except AttributeError: pass

# void wgpuRenderPassEncoderSetStencilReference(WGPURenderPassEncoder renderPassEncoder, uint32_t reference)
try: (wgpuRenderPassEncoderSetStencilReference:=dll.wgpuRenderPassEncoderSetStencilReference).restype, wgpuRenderPassEncoderSetStencilReference.argtypes = None, [WGPURenderPassEncoder, uint32_t]
except AttributeError: pass

# void wgpuRenderPassEncoderSetVertexBuffer(WGPURenderPassEncoder renderPassEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size)
try: (wgpuRenderPassEncoderSetVertexBuffer:=dll.wgpuRenderPassEncoderSetVertexBuffer).restype, wgpuRenderPassEncoderSetVertexBuffer.argtypes = None, [WGPURenderPassEncoder, uint32_t, WGPUBuffer, uint64_t, uint64_t]
except AttributeError: pass

# void wgpuRenderPassEncoderSetViewport(WGPURenderPassEncoder renderPassEncoder, float x, float y, float width, float height, float minDepth, float maxDepth)
try: (wgpuRenderPassEncoderSetViewport:=dll.wgpuRenderPassEncoderSetViewport).restype, wgpuRenderPassEncoderSetViewport.argtypes = None, [WGPURenderPassEncoder, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
except AttributeError: pass

# void wgpuRenderPassEncoderWriteTimestamp(WGPURenderPassEncoder renderPassEncoder, WGPUQuerySet querySet, uint32_t queryIndex)
try: (wgpuRenderPassEncoderWriteTimestamp:=dll.wgpuRenderPassEncoderWriteTimestamp).restype, wgpuRenderPassEncoderWriteTimestamp.argtypes = None, [WGPURenderPassEncoder, WGPUQuerySet, uint32_t]
except AttributeError: pass

# void wgpuRenderPassEncoderAddRef(WGPURenderPassEncoder renderPassEncoder)
try: (wgpuRenderPassEncoderAddRef:=dll.wgpuRenderPassEncoderAddRef).restype, wgpuRenderPassEncoderAddRef.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

# void wgpuRenderPassEncoderRelease(WGPURenderPassEncoder renderPassEncoder)
try: (wgpuRenderPassEncoderRelease:=dll.wgpuRenderPassEncoderRelease).restype, wgpuRenderPassEncoderRelease.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

# WGPUBindGroupLayout wgpuRenderPipelineGetBindGroupLayout(WGPURenderPipeline renderPipeline, uint32_t groupIndex)
try: (wgpuRenderPipelineGetBindGroupLayout:=dll.wgpuRenderPipelineGetBindGroupLayout).restype, wgpuRenderPipelineGetBindGroupLayout.argtypes = WGPUBindGroupLayout, [WGPURenderPipeline, uint32_t]
except AttributeError: pass

# void wgpuRenderPipelineSetLabel(WGPURenderPipeline renderPipeline, WGPUStringView label)
try: (wgpuRenderPipelineSetLabel:=dll.wgpuRenderPipelineSetLabel).restype, wgpuRenderPipelineSetLabel.argtypes = None, [WGPURenderPipeline, WGPUStringView]
except AttributeError: pass

# void wgpuRenderPipelineAddRef(WGPURenderPipeline renderPipeline)
try: (wgpuRenderPipelineAddRef:=dll.wgpuRenderPipelineAddRef).restype, wgpuRenderPipelineAddRef.argtypes = None, [WGPURenderPipeline]
except AttributeError: pass

# void wgpuRenderPipelineRelease(WGPURenderPipeline renderPipeline)
try: (wgpuRenderPipelineRelease:=dll.wgpuRenderPipelineRelease).restype, wgpuRenderPipelineRelease.argtypes = None, [WGPURenderPipeline]
except AttributeError: pass

# void wgpuSamplerSetLabel(WGPUSampler sampler, WGPUStringView label)
try: (wgpuSamplerSetLabel:=dll.wgpuSamplerSetLabel).restype, wgpuSamplerSetLabel.argtypes = None, [WGPUSampler, WGPUStringView]
except AttributeError: pass

# void wgpuSamplerAddRef(WGPUSampler sampler)
try: (wgpuSamplerAddRef:=dll.wgpuSamplerAddRef).restype, wgpuSamplerAddRef.argtypes = None, [WGPUSampler]
except AttributeError: pass

# void wgpuSamplerRelease(WGPUSampler sampler)
try: (wgpuSamplerRelease:=dll.wgpuSamplerRelease).restype, wgpuSamplerRelease.argtypes = None, [WGPUSampler]
except AttributeError: pass

# void wgpuShaderModuleGetCompilationInfo(WGPUShaderModule shaderModule, WGPUCompilationInfoCallback callback, void *userdata)
try: (wgpuShaderModuleGetCompilationInfo:=dll.wgpuShaderModuleGetCompilationInfo).restype, wgpuShaderModuleGetCompilationInfo.argtypes = None, [WGPUShaderModule, WGPUCompilationInfoCallback, ctypes.c_void_p]
except AttributeError: pass

# WGPUFuture wgpuShaderModuleGetCompilationInfo2(WGPUShaderModule shaderModule, WGPUCompilationInfoCallbackInfo2 callbackInfo)
try: (wgpuShaderModuleGetCompilationInfo2:=dll.wgpuShaderModuleGetCompilationInfo2).restype, wgpuShaderModuleGetCompilationInfo2.argtypes = WGPUFuture, [WGPUShaderModule, WGPUCompilationInfoCallbackInfo2]
except AttributeError: pass

# WGPUFuture wgpuShaderModuleGetCompilationInfoF(WGPUShaderModule shaderModule, WGPUCompilationInfoCallbackInfo callbackInfo)
try: (wgpuShaderModuleGetCompilationInfoF:=dll.wgpuShaderModuleGetCompilationInfoF).restype, wgpuShaderModuleGetCompilationInfoF.argtypes = WGPUFuture, [WGPUShaderModule, WGPUCompilationInfoCallbackInfo]
except AttributeError: pass

# void wgpuShaderModuleSetLabel(WGPUShaderModule shaderModule, WGPUStringView label)
try: (wgpuShaderModuleSetLabel:=dll.wgpuShaderModuleSetLabel).restype, wgpuShaderModuleSetLabel.argtypes = None, [WGPUShaderModule, WGPUStringView]
except AttributeError: pass

# void wgpuShaderModuleAddRef(WGPUShaderModule shaderModule)
try: (wgpuShaderModuleAddRef:=dll.wgpuShaderModuleAddRef).restype, wgpuShaderModuleAddRef.argtypes = None, [WGPUShaderModule]
except AttributeError: pass

# void wgpuShaderModuleRelease(WGPUShaderModule shaderModule)
try: (wgpuShaderModuleRelease:=dll.wgpuShaderModuleRelease).restype, wgpuShaderModuleRelease.argtypes = None, [WGPUShaderModule]
except AttributeError: pass

# WGPUStatus wgpuSharedBufferMemoryBeginAccess(WGPUSharedBufferMemory sharedBufferMemory, WGPUBuffer buffer, const WGPUSharedBufferMemoryBeginAccessDescriptor *descriptor)
try: (wgpuSharedBufferMemoryBeginAccess:=dll.wgpuSharedBufferMemoryBeginAccess).restype, wgpuSharedBufferMemoryBeginAccess.argtypes = WGPUStatus, [WGPUSharedBufferMemory, WGPUBuffer, ctypes.POINTER(WGPUSharedBufferMemoryBeginAccessDescriptor)]
except AttributeError: pass

# WGPUBuffer wgpuSharedBufferMemoryCreateBuffer(WGPUSharedBufferMemory sharedBufferMemory, const WGPUBufferDescriptor *descriptor)
try: (wgpuSharedBufferMemoryCreateBuffer:=dll.wgpuSharedBufferMemoryCreateBuffer).restype, wgpuSharedBufferMemoryCreateBuffer.argtypes = WGPUBuffer, [WGPUSharedBufferMemory, ctypes.POINTER(WGPUBufferDescriptor)]
except AttributeError: pass

# WGPUStatus wgpuSharedBufferMemoryEndAccess(WGPUSharedBufferMemory sharedBufferMemory, WGPUBuffer buffer, WGPUSharedBufferMemoryEndAccessState *descriptor)
try: (wgpuSharedBufferMemoryEndAccess:=dll.wgpuSharedBufferMemoryEndAccess).restype, wgpuSharedBufferMemoryEndAccess.argtypes = WGPUStatus, [WGPUSharedBufferMemory, WGPUBuffer, ctypes.POINTER(WGPUSharedBufferMemoryEndAccessState)]
except AttributeError: pass

# WGPUStatus wgpuSharedBufferMemoryGetProperties(WGPUSharedBufferMemory sharedBufferMemory, WGPUSharedBufferMemoryProperties *properties)
try: (wgpuSharedBufferMemoryGetProperties:=dll.wgpuSharedBufferMemoryGetProperties).restype, wgpuSharedBufferMemoryGetProperties.argtypes = WGPUStatus, [WGPUSharedBufferMemory, ctypes.POINTER(WGPUSharedBufferMemoryProperties)]
except AttributeError: pass

# WGPUBool wgpuSharedBufferMemoryIsDeviceLost(WGPUSharedBufferMemory sharedBufferMemory)
try: (wgpuSharedBufferMemoryIsDeviceLost:=dll.wgpuSharedBufferMemoryIsDeviceLost).restype, wgpuSharedBufferMemoryIsDeviceLost.argtypes = WGPUBool, [WGPUSharedBufferMemory]
except AttributeError: pass

# void wgpuSharedBufferMemorySetLabel(WGPUSharedBufferMemory sharedBufferMemory, WGPUStringView label)
try: (wgpuSharedBufferMemorySetLabel:=dll.wgpuSharedBufferMemorySetLabel).restype, wgpuSharedBufferMemorySetLabel.argtypes = None, [WGPUSharedBufferMemory, WGPUStringView]
except AttributeError: pass

# void wgpuSharedBufferMemoryAddRef(WGPUSharedBufferMemory sharedBufferMemory)
try: (wgpuSharedBufferMemoryAddRef:=dll.wgpuSharedBufferMemoryAddRef).restype, wgpuSharedBufferMemoryAddRef.argtypes = None, [WGPUSharedBufferMemory]
except AttributeError: pass

# void wgpuSharedBufferMemoryRelease(WGPUSharedBufferMemory sharedBufferMemory)
try: (wgpuSharedBufferMemoryRelease:=dll.wgpuSharedBufferMemoryRelease).restype, wgpuSharedBufferMemoryRelease.argtypes = None, [WGPUSharedBufferMemory]
except AttributeError: pass

# void wgpuSharedFenceExportInfo(WGPUSharedFence sharedFence, WGPUSharedFenceExportInfo *info)
try: (wgpuSharedFenceExportInfo:=dll.wgpuSharedFenceExportInfo).restype, wgpuSharedFenceExportInfo.argtypes = None, [WGPUSharedFence, ctypes.POINTER(WGPUSharedFenceExportInfo)]
except AttributeError: pass

# void wgpuSharedFenceAddRef(WGPUSharedFence sharedFence)
try: (wgpuSharedFenceAddRef:=dll.wgpuSharedFenceAddRef).restype, wgpuSharedFenceAddRef.argtypes = None, [WGPUSharedFence]
except AttributeError: pass

# void wgpuSharedFenceRelease(WGPUSharedFence sharedFence)
try: (wgpuSharedFenceRelease:=dll.wgpuSharedFenceRelease).restype, wgpuSharedFenceRelease.argtypes = None, [WGPUSharedFence]
except AttributeError: pass

# WGPUStatus wgpuSharedTextureMemoryBeginAccess(WGPUSharedTextureMemory sharedTextureMemory, WGPUTexture texture, const WGPUSharedTextureMemoryBeginAccessDescriptor *descriptor)
try: (wgpuSharedTextureMemoryBeginAccess:=dll.wgpuSharedTextureMemoryBeginAccess).restype, wgpuSharedTextureMemoryBeginAccess.argtypes = WGPUStatus, [WGPUSharedTextureMemory, WGPUTexture, ctypes.POINTER(WGPUSharedTextureMemoryBeginAccessDescriptor)]
except AttributeError: pass

# WGPUTexture wgpuSharedTextureMemoryCreateTexture(WGPUSharedTextureMemory sharedTextureMemory, const WGPUTextureDescriptor *descriptor)
try: (wgpuSharedTextureMemoryCreateTexture:=dll.wgpuSharedTextureMemoryCreateTexture).restype, wgpuSharedTextureMemoryCreateTexture.argtypes = WGPUTexture, [WGPUSharedTextureMemory, ctypes.POINTER(WGPUTextureDescriptor)]
except AttributeError: pass

# WGPUStatus wgpuSharedTextureMemoryEndAccess(WGPUSharedTextureMemory sharedTextureMemory, WGPUTexture texture, WGPUSharedTextureMemoryEndAccessState *descriptor)
try: (wgpuSharedTextureMemoryEndAccess:=dll.wgpuSharedTextureMemoryEndAccess).restype, wgpuSharedTextureMemoryEndAccess.argtypes = WGPUStatus, [WGPUSharedTextureMemory, WGPUTexture, ctypes.POINTER(WGPUSharedTextureMemoryEndAccessState)]
except AttributeError: pass

# WGPUStatus wgpuSharedTextureMemoryGetProperties(WGPUSharedTextureMemory sharedTextureMemory, WGPUSharedTextureMemoryProperties *properties)
try: (wgpuSharedTextureMemoryGetProperties:=dll.wgpuSharedTextureMemoryGetProperties).restype, wgpuSharedTextureMemoryGetProperties.argtypes = WGPUStatus, [WGPUSharedTextureMemory, ctypes.POINTER(WGPUSharedTextureMemoryProperties)]
except AttributeError: pass

# WGPUBool wgpuSharedTextureMemoryIsDeviceLost(WGPUSharedTextureMemory sharedTextureMemory)
try: (wgpuSharedTextureMemoryIsDeviceLost:=dll.wgpuSharedTextureMemoryIsDeviceLost).restype, wgpuSharedTextureMemoryIsDeviceLost.argtypes = WGPUBool, [WGPUSharedTextureMemory]
except AttributeError: pass

# void wgpuSharedTextureMemorySetLabel(WGPUSharedTextureMemory sharedTextureMemory, WGPUStringView label)
try: (wgpuSharedTextureMemorySetLabel:=dll.wgpuSharedTextureMemorySetLabel).restype, wgpuSharedTextureMemorySetLabel.argtypes = None, [WGPUSharedTextureMemory, WGPUStringView]
except AttributeError: pass

# void wgpuSharedTextureMemoryAddRef(WGPUSharedTextureMemory sharedTextureMemory)
try: (wgpuSharedTextureMemoryAddRef:=dll.wgpuSharedTextureMemoryAddRef).restype, wgpuSharedTextureMemoryAddRef.argtypes = None, [WGPUSharedTextureMemory]
except AttributeError: pass

# void wgpuSharedTextureMemoryRelease(WGPUSharedTextureMemory sharedTextureMemory)
try: (wgpuSharedTextureMemoryRelease:=dll.wgpuSharedTextureMemoryRelease).restype, wgpuSharedTextureMemoryRelease.argtypes = None, [WGPUSharedTextureMemory]
except AttributeError: pass

# void wgpuSurfaceConfigure(WGPUSurface surface, const WGPUSurfaceConfiguration *config)
try: (wgpuSurfaceConfigure:=dll.wgpuSurfaceConfigure).restype, wgpuSurfaceConfigure.argtypes = None, [WGPUSurface, ctypes.POINTER(WGPUSurfaceConfiguration)]
except AttributeError: pass

# WGPUStatus wgpuSurfaceGetCapabilities(WGPUSurface surface, WGPUAdapter adapter, WGPUSurfaceCapabilities *capabilities)
try: (wgpuSurfaceGetCapabilities:=dll.wgpuSurfaceGetCapabilities).restype, wgpuSurfaceGetCapabilities.argtypes = WGPUStatus, [WGPUSurface, WGPUAdapter, ctypes.POINTER(WGPUSurfaceCapabilities)]
except AttributeError: pass

# void wgpuSurfaceGetCurrentTexture(WGPUSurface surface, WGPUSurfaceTexture *surfaceTexture)
try: (wgpuSurfaceGetCurrentTexture:=dll.wgpuSurfaceGetCurrentTexture).restype, wgpuSurfaceGetCurrentTexture.argtypes = None, [WGPUSurface, ctypes.POINTER(WGPUSurfaceTexture)]
except AttributeError: pass

# void wgpuSurfacePresent(WGPUSurface surface)
try: (wgpuSurfacePresent:=dll.wgpuSurfacePresent).restype, wgpuSurfacePresent.argtypes = None, [WGPUSurface]
except AttributeError: pass

# void wgpuSurfaceSetLabel(WGPUSurface surface, WGPUStringView label)
try: (wgpuSurfaceSetLabel:=dll.wgpuSurfaceSetLabel).restype, wgpuSurfaceSetLabel.argtypes = None, [WGPUSurface, WGPUStringView]
except AttributeError: pass

# void wgpuSurfaceUnconfigure(WGPUSurface surface)
try: (wgpuSurfaceUnconfigure:=dll.wgpuSurfaceUnconfigure).restype, wgpuSurfaceUnconfigure.argtypes = None, [WGPUSurface]
except AttributeError: pass

# void wgpuSurfaceAddRef(WGPUSurface surface)
try: (wgpuSurfaceAddRef:=dll.wgpuSurfaceAddRef).restype, wgpuSurfaceAddRef.argtypes = None, [WGPUSurface]
except AttributeError: pass

# void wgpuSurfaceRelease(WGPUSurface surface)
try: (wgpuSurfaceRelease:=dll.wgpuSurfaceRelease).restype, wgpuSurfaceRelease.argtypes = None, [WGPUSurface]
except AttributeError: pass

# WGPUTextureView wgpuTextureCreateErrorView(WGPUTexture texture, const WGPUTextureViewDescriptor *descriptor)
try: (wgpuTextureCreateErrorView:=dll.wgpuTextureCreateErrorView).restype, wgpuTextureCreateErrorView.argtypes = WGPUTextureView, [WGPUTexture, ctypes.POINTER(WGPUTextureViewDescriptor)]
except AttributeError: pass

# WGPUTextureView wgpuTextureCreateView(WGPUTexture texture, const WGPUTextureViewDescriptor *descriptor)
try: (wgpuTextureCreateView:=dll.wgpuTextureCreateView).restype, wgpuTextureCreateView.argtypes = WGPUTextureView, [WGPUTexture, ctypes.POINTER(WGPUTextureViewDescriptor)]
except AttributeError: pass

# void wgpuTextureDestroy(WGPUTexture texture)
try: (wgpuTextureDestroy:=dll.wgpuTextureDestroy).restype, wgpuTextureDestroy.argtypes = None, [WGPUTexture]
except AttributeError: pass

# uint32_t wgpuTextureGetDepthOrArrayLayers(WGPUTexture texture)
try: (wgpuTextureGetDepthOrArrayLayers:=dll.wgpuTextureGetDepthOrArrayLayers).restype, wgpuTextureGetDepthOrArrayLayers.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

# WGPUTextureDimension wgpuTextureGetDimension(WGPUTexture texture)
try: (wgpuTextureGetDimension:=dll.wgpuTextureGetDimension).restype, wgpuTextureGetDimension.argtypes = WGPUTextureDimension, [WGPUTexture]
except AttributeError: pass

# WGPUTextureFormat wgpuTextureGetFormat(WGPUTexture texture)
try: (wgpuTextureGetFormat:=dll.wgpuTextureGetFormat).restype, wgpuTextureGetFormat.argtypes = WGPUTextureFormat, [WGPUTexture]
except AttributeError: pass

# uint32_t wgpuTextureGetHeight(WGPUTexture texture)
try: (wgpuTextureGetHeight:=dll.wgpuTextureGetHeight).restype, wgpuTextureGetHeight.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

# uint32_t wgpuTextureGetMipLevelCount(WGPUTexture texture)
try: (wgpuTextureGetMipLevelCount:=dll.wgpuTextureGetMipLevelCount).restype, wgpuTextureGetMipLevelCount.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

# uint32_t wgpuTextureGetSampleCount(WGPUTexture texture)
try: (wgpuTextureGetSampleCount:=dll.wgpuTextureGetSampleCount).restype, wgpuTextureGetSampleCount.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

# WGPUTextureUsage wgpuTextureGetUsage(WGPUTexture texture)
try: (wgpuTextureGetUsage:=dll.wgpuTextureGetUsage).restype, wgpuTextureGetUsage.argtypes = WGPUTextureUsage, [WGPUTexture]
except AttributeError: pass

# uint32_t wgpuTextureGetWidth(WGPUTexture texture)
try: (wgpuTextureGetWidth:=dll.wgpuTextureGetWidth).restype, wgpuTextureGetWidth.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

# void wgpuTextureSetLabel(WGPUTexture texture, WGPUStringView label)
try: (wgpuTextureSetLabel:=dll.wgpuTextureSetLabel).restype, wgpuTextureSetLabel.argtypes = None, [WGPUTexture, WGPUStringView]
except AttributeError: pass

# void wgpuTextureAddRef(WGPUTexture texture)
try: (wgpuTextureAddRef:=dll.wgpuTextureAddRef).restype, wgpuTextureAddRef.argtypes = None, [WGPUTexture]
except AttributeError: pass

# void wgpuTextureRelease(WGPUTexture texture)
try: (wgpuTextureRelease:=dll.wgpuTextureRelease).restype, wgpuTextureRelease.argtypes = None, [WGPUTexture]
except AttributeError: pass

# void wgpuTextureViewSetLabel(WGPUTextureView textureView, WGPUStringView label)
try: (wgpuTextureViewSetLabel:=dll.wgpuTextureViewSetLabel).restype, wgpuTextureViewSetLabel.argtypes = None, [WGPUTextureView, WGPUStringView]
except AttributeError: pass

# void wgpuTextureViewAddRef(WGPUTextureView textureView)
try: (wgpuTextureViewAddRef:=dll.wgpuTextureViewAddRef).restype, wgpuTextureViewAddRef.argtypes = None, [WGPUTextureView]
except AttributeError: pass

# void wgpuTextureViewRelease(WGPUTextureView textureView)
try: (wgpuTextureViewRelease:=dll.wgpuTextureViewRelease).restype, wgpuTextureViewRelease.argtypes = None, [WGPUTextureView]
except AttributeError: pass

WGPUBufferUsage_None = 0x0000000000000000
WGPUBufferUsage_MapRead = 0x0000000000000001
WGPUBufferUsage_MapWrite = 0x0000000000000002
WGPUBufferUsage_CopySrc = 0x0000000000000004
WGPUBufferUsage_CopyDst = 0x0000000000000008
WGPUBufferUsage_Index = 0x0000000000000010
WGPUBufferUsage_Vertex = 0x0000000000000020
WGPUBufferUsage_Uniform = 0x0000000000000040
WGPUBufferUsage_Storage = 0x0000000000000080
WGPUBufferUsage_Indirect = 0x0000000000000100
WGPUBufferUsage_QueryResolve = 0x0000000000000200
WGPUColorWriteMask_None = 0x0000000000000000
WGPUColorWriteMask_Red = 0x0000000000000001
WGPUColorWriteMask_Green = 0x0000000000000002
WGPUColorWriteMask_Blue = 0x0000000000000004
WGPUColorWriteMask_Alpha = 0x0000000000000008
WGPUColorWriteMask_All = 0x000000000000000F
WGPUHeapProperty_DeviceLocal = 0x0000000000000001
WGPUHeapProperty_HostVisible = 0x0000000000000002
WGPUHeapProperty_HostCoherent = 0x0000000000000004
WGPUHeapProperty_HostUncached = 0x0000000000000008
WGPUHeapProperty_HostCached = 0x0000000000000010
WGPUMapMode_None = 0x0000000000000000
WGPUMapMode_Read = 0x0000000000000001
WGPUMapMode_Write = 0x0000000000000002
WGPUShaderStage_None = 0x0000000000000000
WGPUShaderStage_Vertex = 0x0000000000000001
WGPUShaderStage_Fragment = 0x0000000000000002
WGPUShaderStage_Compute = 0x0000000000000004
WGPUTextureUsage_None = 0x0000000000000000
WGPUTextureUsage_CopySrc = 0x0000000000000001
WGPUTextureUsage_CopyDst = 0x0000000000000002
WGPUTextureUsage_TextureBinding = 0x0000000000000004
WGPUTextureUsage_StorageBinding = 0x0000000000000008
WGPUTextureUsage_RenderAttachment = 0x0000000000000010
WGPUTextureUsage_TransientAttachment = 0x0000000000000020
WGPUTextureUsage_StorageAttachment = 0x0000000000000040