# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.helpers import WIN, OSX
import sysconfig, os
dll = DLL('webgpu', os.path.join(sysconfig.get_paths()['purelib'], 'pydawn', 'lib', 'libwebgpu_dawn.dll') if WIN else 'webgpu_dawn')
WGPUFlags = ctypes.c_uint64
WGPUBool = ctypes.c_uint32
class struct_WGPUAdapterImpl(Struct): pass
WGPUAdapter = Pointer(struct_WGPUAdapterImpl)
class struct_WGPUBindGroupImpl(Struct): pass
WGPUBindGroup = Pointer(struct_WGPUBindGroupImpl)
class struct_WGPUBindGroupLayoutImpl(Struct): pass
WGPUBindGroupLayout = Pointer(struct_WGPUBindGroupLayoutImpl)
class struct_WGPUBufferImpl(Struct): pass
WGPUBuffer = Pointer(struct_WGPUBufferImpl)
class struct_WGPUCommandBufferImpl(Struct): pass
WGPUCommandBuffer = Pointer(struct_WGPUCommandBufferImpl)
class struct_WGPUCommandEncoderImpl(Struct): pass
WGPUCommandEncoder = Pointer(struct_WGPUCommandEncoderImpl)
class struct_WGPUComputePassEncoderImpl(Struct): pass
WGPUComputePassEncoder = Pointer(struct_WGPUComputePassEncoderImpl)
class struct_WGPUComputePipelineImpl(Struct): pass
WGPUComputePipeline = Pointer(struct_WGPUComputePipelineImpl)
class struct_WGPUDeviceImpl(Struct): pass
WGPUDevice = Pointer(struct_WGPUDeviceImpl)
class struct_WGPUExternalTextureImpl(Struct): pass
WGPUExternalTexture = Pointer(struct_WGPUExternalTextureImpl)
class struct_WGPUInstanceImpl(Struct): pass
WGPUInstance = Pointer(struct_WGPUInstanceImpl)
class struct_WGPUPipelineLayoutImpl(Struct): pass
WGPUPipelineLayout = Pointer(struct_WGPUPipelineLayoutImpl)
class struct_WGPUQuerySetImpl(Struct): pass
WGPUQuerySet = Pointer(struct_WGPUQuerySetImpl)
class struct_WGPUQueueImpl(Struct): pass
WGPUQueue = Pointer(struct_WGPUQueueImpl)
class struct_WGPURenderBundleImpl(Struct): pass
WGPURenderBundle = Pointer(struct_WGPURenderBundleImpl)
class struct_WGPURenderBundleEncoderImpl(Struct): pass
WGPURenderBundleEncoder = Pointer(struct_WGPURenderBundleEncoderImpl)
class struct_WGPURenderPassEncoderImpl(Struct): pass
WGPURenderPassEncoder = Pointer(struct_WGPURenderPassEncoderImpl)
class struct_WGPURenderPipelineImpl(Struct): pass
WGPURenderPipeline = Pointer(struct_WGPURenderPipelineImpl)
class struct_WGPUSamplerImpl(Struct): pass
WGPUSampler = Pointer(struct_WGPUSamplerImpl)
class struct_WGPUShaderModuleImpl(Struct): pass
WGPUShaderModule = Pointer(struct_WGPUShaderModuleImpl)
class struct_WGPUSharedBufferMemoryImpl(Struct): pass
WGPUSharedBufferMemory = Pointer(struct_WGPUSharedBufferMemoryImpl)
class struct_WGPUSharedFenceImpl(Struct): pass
WGPUSharedFence = Pointer(struct_WGPUSharedFenceImpl)
class struct_WGPUSharedTextureMemoryImpl(Struct): pass
WGPUSharedTextureMemory = Pointer(struct_WGPUSharedTextureMemoryImpl)
class struct_WGPUSurfaceImpl(Struct): pass
WGPUSurface = Pointer(struct_WGPUSurfaceImpl)
class struct_WGPUTextureImpl(Struct): pass
WGPUTexture = Pointer(struct_WGPUTextureImpl)
class struct_WGPUTextureViewImpl(Struct): pass
WGPUTextureView = Pointer(struct_WGPUTextureViewImpl)
class struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER(Struct): pass
struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER.SIZE = 4
struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER._fields_ = ['unused']
setattr(struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER, 'unused', field(0, WGPUBool))
class struct_WGPUAdapterPropertiesD3D(Struct): pass
class struct_WGPUChainedStructOut(Struct): pass
WGPUChainedStructOut = struct_WGPUChainedStructOut
enum_WGPUSType = CEnum(ctypes.c_uint32)
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
struct_WGPUChainedStructOut.SIZE = 16
struct_WGPUChainedStructOut._fields_ = ['next', 'sType']
setattr(struct_WGPUChainedStructOut, 'next', field(0, Pointer(struct_WGPUChainedStructOut)))
setattr(struct_WGPUChainedStructOut, 'sType', field(8, WGPUSType))
uint32_t = ctypes.c_uint32
struct_WGPUAdapterPropertiesD3D.SIZE = 24
struct_WGPUAdapterPropertiesD3D._fields_ = ['chain', 'shaderModel']
setattr(struct_WGPUAdapterPropertiesD3D, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUAdapterPropertiesD3D, 'shaderModel', field(16, uint32_t))
class struct_WGPUAdapterPropertiesSubgroups(Struct): pass
struct_WGPUAdapterPropertiesSubgroups.SIZE = 24
struct_WGPUAdapterPropertiesSubgroups._fields_ = ['chain', 'subgroupMinSize', 'subgroupMaxSize']
setattr(struct_WGPUAdapterPropertiesSubgroups, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUAdapterPropertiesSubgroups, 'subgroupMinSize', field(16, uint32_t))
setattr(struct_WGPUAdapterPropertiesSubgroups, 'subgroupMaxSize', field(20, uint32_t))
class struct_WGPUAdapterPropertiesVk(Struct): pass
struct_WGPUAdapterPropertiesVk.SIZE = 24
struct_WGPUAdapterPropertiesVk._fields_ = ['chain', 'driverVersion']
setattr(struct_WGPUAdapterPropertiesVk, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUAdapterPropertiesVk, 'driverVersion', field(16, uint32_t))
class struct_WGPUBindGroupEntry(Struct): pass
class struct_WGPUChainedStruct(Struct): pass
WGPUChainedStruct = struct_WGPUChainedStruct
struct_WGPUChainedStruct.SIZE = 16
struct_WGPUChainedStruct._fields_ = ['next', 'sType']
setattr(struct_WGPUChainedStruct, 'next', field(0, Pointer(struct_WGPUChainedStruct)))
setattr(struct_WGPUChainedStruct, 'sType', field(8, WGPUSType))
uint64_t = ctypes.c_uint64
struct_WGPUBindGroupEntry.SIZE = 56
struct_WGPUBindGroupEntry._fields_ = ['nextInChain', 'binding', 'buffer', 'offset', 'size', 'sampler', 'textureView']
setattr(struct_WGPUBindGroupEntry, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUBindGroupEntry, 'binding', field(8, uint32_t))
setattr(struct_WGPUBindGroupEntry, 'buffer', field(16, WGPUBuffer))
setattr(struct_WGPUBindGroupEntry, 'offset', field(24, uint64_t))
setattr(struct_WGPUBindGroupEntry, 'size', field(32, uint64_t))
setattr(struct_WGPUBindGroupEntry, 'sampler', field(40, WGPUSampler))
setattr(struct_WGPUBindGroupEntry, 'textureView', field(48, WGPUTextureView))
class struct_WGPUBlendComponent(Struct): pass
enum_WGPUBlendOperation = CEnum(ctypes.c_uint32)
WGPUBlendOperation_Undefined = enum_WGPUBlendOperation.define('WGPUBlendOperation_Undefined', 0)
WGPUBlendOperation_Add = enum_WGPUBlendOperation.define('WGPUBlendOperation_Add', 1)
WGPUBlendOperation_Subtract = enum_WGPUBlendOperation.define('WGPUBlendOperation_Subtract', 2)
WGPUBlendOperation_ReverseSubtract = enum_WGPUBlendOperation.define('WGPUBlendOperation_ReverseSubtract', 3)
WGPUBlendOperation_Min = enum_WGPUBlendOperation.define('WGPUBlendOperation_Min', 4)
WGPUBlendOperation_Max = enum_WGPUBlendOperation.define('WGPUBlendOperation_Max', 5)
WGPUBlendOperation_Force32 = enum_WGPUBlendOperation.define('WGPUBlendOperation_Force32', 2147483647)

WGPUBlendOperation = enum_WGPUBlendOperation
enum_WGPUBlendFactor = CEnum(ctypes.c_uint32)
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
struct_WGPUBlendComponent.SIZE = 12
struct_WGPUBlendComponent._fields_ = ['operation', 'srcFactor', 'dstFactor']
setattr(struct_WGPUBlendComponent, 'operation', field(0, WGPUBlendOperation))
setattr(struct_WGPUBlendComponent, 'srcFactor', field(4, WGPUBlendFactor))
setattr(struct_WGPUBlendComponent, 'dstFactor', field(8, WGPUBlendFactor))
class struct_WGPUBufferBindingLayout(Struct): pass
enum_WGPUBufferBindingType = CEnum(ctypes.c_uint32)
WGPUBufferBindingType_BindingNotUsed = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_BindingNotUsed', 0)
WGPUBufferBindingType_Uniform = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Uniform', 1)
WGPUBufferBindingType_Storage = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Storage', 2)
WGPUBufferBindingType_ReadOnlyStorage = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_ReadOnlyStorage', 3)
WGPUBufferBindingType_Force32 = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Force32', 2147483647)

WGPUBufferBindingType = enum_WGPUBufferBindingType
struct_WGPUBufferBindingLayout.SIZE = 24
struct_WGPUBufferBindingLayout._fields_ = ['nextInChain', 'type', 'hasDynamicOffset', 'minBindingSize']
setattr(struct_WGPUBufferBindingLayout, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUBufferBindingLayout, 'type', field(8, WGPUBufferBindingType))
setattr(struct_WGPUBufferBindingLayout, 'hasDynamicOffset', field(12, WGPUBool))
setattr(struct_WGPUBufferBindingLayout, 'minBindingSize', field(16, uint64_t))
class struct_WGPUBufferHostMappedPointer(Struct): pass
WGPUCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
struct_WGPUBufferHostMappedPointer.SIZE = 40
struct_WGPUBufferHostMappedPointer._fields_ = ['chain', 'pointer', 'disposeCallback', 'userdata']
setattr(struct_WGPUBufferHostMappedPointer, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUBufferHostMappedPointer, 'pointer', field(16, ctypes.c_void_p))
setattr(struct_WGPUBufferHostMappedPointer, 'disposeCallback', field(24, WGPUCallback))
setattr(struct_WGPUBufferHostMappedPointer, 'userdata', field(32, ctypes.c_void_p))
class struct_WGPUBufferMapCallbackInfo(Struct): pass
enum_WGPUCallbackMode = CEnum(ctypes.c_uint32)
WGPUCallbackMode_WaitAnyOnly = enum_WGPUCallbackMode.define('WGPUCallbackMode_WaitAnyOnly', 1)
WGPUCallbackMode_AllowProcessEvents = enum_WGPUCallbackMode.define('WGPUCallbackMode_AllowProcessEvents', 2)
WGPUCallbackMode_AllowSpontaneous = enum_WGPUCallbackMode.define('WGPUCallbackMode_AllowSpontaneous', 3)
WGPUCallbackMode_Force32 = enum_WGPUCallbackMode.define('WGPUCallbackMode_Force32', 2147483647)

WGPUCallbackMode = enum_WGPUCallbackMode
enum_WGPUBufferMapAsyncStatus = CEnum(ctypes.c_uint32)
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
struct_WGPUBufferMapCallbackInfo.SIZE = 32
struct_WGPUBufferMapCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'userdata']
setattr(struct_WGPUBufferMapCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUBufferMapCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUBufferMapCallbackInfo, 'callback', field(16, WGPUBufferMapCallback))
setattr(struct_WGPUBufferMapCallbackInfo, 'userdata', field(24, ctypes.c_void_p))
class struct_WGPUColor(Struct): pass
struct_WGPUColor.SIZE = 32
struct_WGPUColor._fields_ = ['r', 'g', 'b', 'a']
setattr(struct_WGPUColor, 'r', field(0, ctypes.c_double))
setattr(struct_WGPUColor, 'g', field(8, ctypes.c_double))
setattr(struct_WGPUColor, 'b', field(16, ctypes.c_double))
setattr(struct_WGPUColor, 'a', field(24, ctypes.c_double))
class struct_WGPUColorTargetStateExpandResolveTextureDawn(Struct): pass
struct_WGPUColorTargetStateExpandResolveTextureDawn.SIZE = 24
struct_WGPUColorTargetStateExpandResolveTextureDawn._fields_ = ['chain', 'enabled']
setattr(struct_WGPUColorTargetStateExpandResolveTextureDawn, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUColorTargetStateExpandResolveTextureDawn, 'enabled', field(16, WGPUBool))
class struct_WGPUCompilationInfoCallbackInfo(Struct): pass
enum_WGPUCompilationInfoRequestStatus = CEnum(ctypes.c_uint32)
WGPUCompilationInfoRequestStatus_Success = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Success', 1)
WGPUCompilationInfoRequestStatus_InstanceDropped = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_InstanceDropped', 2)
WGPUCompilationInfoRequestStatus_Error = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Error', 3)
WGPUCompilationInfoRequestStatus_DeviceLost = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_DeviceLost', 4)
WGPUCompilationInfoRequestStatus_Unknown = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Unknown', 5)
WGPUCompilationInfoRequestStatus_Force32 = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Force32', 2147483647)

class struct_WGPUCompilationInfo(Struct): pass
size_t = ctypes.c_uint64
class struct_WGPUCompilationMessage(Struct): pass
WGPUCompilationMessage = struct_WGPUCompilationMessage
class struct_WGPUStringView(Struct): pass
WGPUStringView = struct_WGPUStringView
struct_WGPUStringView.SIZE = 16
struct_WGPUStringView._fields_ = ['data', 'length']
setattr(struct_WGPUStringView, 'data', field(0, Pointer(ctypes.c_char)))
setattr(struct_WGPUStringView, 'length', field(8, size_t))
enum_WGPUCompilationMessageType = CEnum(ctypes.c_uint32)
WGPUCompilationMessageType_Error = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Error', 1)
WGPUCompilationMessageType_Warning = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Warning', 2)
WGPUCompilationMessageType_Info = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Info', 3)
WGPUCompilationMessageType_Force32 = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Force32', 2147483647)

WGPUCompilationMessageType = enum_WGPUCompilationMessageType
struct_WGPUCompilationMessage.SIZE = 88
struct_WGPUCompilationMessage._fields_ = ['nextInChain', 'message', 'type', 'lineNum', 'linePos', 'offset', 'length', 'utf16LinePos', 'utf16Offset', 'utf16Length']
setattr(struct_WGPUCompilationMessage, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCompilationMessage, 'message', field(8, WGPUStringView))
setattr(struct_WGPUCompilationMessage, 'type', field(24, WGPUCompilationMessageType))
setattr(struct_WGPUCompilationMessage, 'lineNum', field(32, uint64_t))
setattr(struct_WGPUCompilationMessage, 'linePos', field(40, uint64_t))
setattr(struct_WGPUCompilationMessage, 'offset', field(48, uint64_t))
setattr(struct_WGPUCompilationMessage, 'length', field(56, uint64_t))
setattr(struct_WGPUCompilationMessage, 'utf16LinePos', field(64, uint64_t))
setattr(struct_WGPUCompilationMessage, 'utf16Offset', field(72, uint64_t))
setattr(struct_WGPUCompilationMessage, 'utf16Length', field(80, uint64_t))
struct_WGPUCompilationInfo.SIZE = 24
struct_WGPUCompilationInfo._fields_ = ['nextInChain', 'messageCount', 'messages']
setattr(struct_WGPUCompilationInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCompilationInfo, 'messageCount', field(8, size_t))
setattr(struct_WGPUCompilationInfo, 'messages', field(16, Pointer(WGPUCompilationMessage)))
WGPUCompilationInfoCallback = ctypes.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, Pointer(struct_WGPUCompilationInfo), ctypes.c_void_p)
struct_WGPUCompilationInfoCallbackInfo.SIZE = 32
struct_WGPUCompilationInfoCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'userdata']
setattr(struct_WGPUCompilationInfoCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCompilationInfoCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUCompilationInfoCallbackInfo, 'callback', field(16, WGPUCompilationInfoCallback))
setattr(struct_WGPUCompilationInfoCallbackInfo, 'userdata', field(24, ctypes.c_void_p))
class struct_WGPUComputePassTimestampWrites(Struct): pass
struct_WGPUComputePassTimestampWrites.SIZE = 16
struct_WGPUComputePassTimestampWrites._fields_ = ['querySet', 'beginningOfPassWriteIndex', 'endOfPassWriteIndex']
setattr(struct_WGPUComputePassTimestampWrites, 'querySet', field(0, WGPUQuerySet))
setattr(struct_WGPUComputePassTimestampWrites, 'beginningOfPassWriteIndex', field(8, uint32_t))
setattr(struct_WGPUComputePassTimestampWrites, 'endOfPassWriteIndex', field(12, uint32_t))
class struct_WGPUCopyTextureForBrowserOptions(Struct): pass
enum_WGPUAlphaMode = CEnum(ctypes.c_uint32)
WGPUAlphaMode_Opaque = enum_WGPUAlphaMode.define('WGPUAlphaMode_Opaque', 1)
WGPUAlphaMode_Premultiplied = enum_WGPUAlphaMode.define('WGPUAlphaMode_Premultiplied', 2)
WGPUAlphaMode_Unpremultiplied = enum_WGPUAlphaMode.define('WGPUAlphaMode_Unpremultiplied', 3)
WGPUAlphaMode_Force32 = enum_WGPUAlphaMode.define('WGPUAlphaMode_Force32', 2147483647)

WGPUAlphaMode = enum_WGPUAlphaMode
struct_WGPUCopyTextureForBrowserOptions.SIZE = 56
struct_WGPUCopyTextureForBrowserOptions._fields_ = ['nextInChain', 'flipY', 'needsColorSpaceConversion', 'srcAlphaMode', 'srcTransferFunctionParameters', 'conversionMatrix', 'dstTransferFunctionParameters', 'dstAlphaMode', 'internalUsage']
setattr(struct_WGPUCopyTextureForBrowserOptions, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCopyTextureForBrowserOptions, 'flipY', field(8, WGPUBool))
setattr(struct_WGPUCopyTextureForBrowserOptions, 'needsColorSpaceConversion', field(12, WGPUBool))
setattr(struct_WGPUCopyTextureForBrowserOptions, 'srcAlphaMode', field(16, WGPUAlphaMode))
setattr(struct_WGPUCopyTextureForBrowserOptions, 'srcTransferFunctionParameters', field(24, Pointer(ctypes.c_float)))
setattr(struct_WGPUCopyTextureForBrowserOptions, 'conversionMatrix', field(32, Pointer(ctypes.c_float)))
setattr(struct_WGPUCopyTextureForBrowserOptions, 'dstTransferFunctionParameters', field(40, Pointer(ctypes.c_float)))
setattr(struct_WGPUCopyTextureForBrowserOptions, 'dstAlphaMode', field(48, WGPUAlphaMode))
setattr(struct_WGPUCopyTextureForBrowserOptions, 'internalUsage', field(52, WGPUBool))
class struct_WGPUCreateComputePipelineAsyncCallbackInfo(Struct): pass
enum_WGPUCreatePipelineAsyncStatus = CEnum(ctypes.c_uint32)
WGPUCreatePipelineAsyncStatus_Success = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Success', 1)
WGPUCreatePipelineAsyncStatus_InstanceDropped = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_InstanceDropped', 2)
WGPUCreatePipelineAsyncStatus_ValidationError = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_ValidationError', 3)
WGPUCreatePipelineAsyncStatus_InternalError = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_InternalError', 4)
WGPUCreatePipelineAsyncStatus_DeviceLost = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_DeviceLost', 5)
WGPUCreatePipelineAsyncStatus_DeviceDestroyed = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_DeviceDestroyed', 6)
WGPUCreatePipelineAsyncStatus_Unknown = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Unknown', 7)
WGPUCreatePipelineAsyncStatus_Force32 = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Force32', 2147483647)

WGPUCreateComputePipelineAsyncCallback = ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, Pointer(struct_WGPUComputePipelineImpl), struct_WGPUStringView, ctypes.c_void_p)
struct_WGPUCreateComputePipelineAsyncCallbackInfo.SIZE = 32
struct_WGPUCreateComputePipelineAsyncCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'userdata']
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo, 'callback', field(16, WGPUCreateComputePipelineAsyncCallback))
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo, 'userdata', field(24, ctypes.c_void_p))
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo(Struct): pass
WGPUCreateRenderPipelineAsyncCallback = ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, Pointer(struct_WGPURenderPipelineImpl), struct_WGPUStringView, ctypes.c_void_p)
struct_WGPUCreateRenderPipelineAsyncCallbackInfo.SIZE = 32
struct_WGPUCreateRenderPipelineAsyncCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'userdata']
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo, 'callback', field(16, WGPUCreateRenderPipelineAsyncCallback))
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo, 'userdata', field(24, ctypes.c_void_p))
class struct_WGPUDawnWGSLBlocklist(Struct): pass
struct_WGPUDawnWGSLBlocklist.SIZE = 32
struct_WGPUDawnWGSLBlocklist._fields_ = ['chain', 'blocklistedFeatureCount', 'blocklistedFeatures']
setattr(struct_WGPUDawnWGSLBlocklist, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnWGSLBlocklist, 'blocklistedFeatureCount', field(16, size_t))
setattr(struct_WGPUDawnWGSLBlocklist, 'blocklistedFeatures', field(24, Pointer(Pointer(ctypes.c_char))))
class struct_WGPUDawnAdapterPropertiesPowerPreference(Struct): pass
enum_WGPUPowerPreference = CEnum(ctypes.c_uint32)
WGPUPowerPreference_Undefined = enum_WGPUPowerPreference.define('WGPUPowerPreference_Undefined', 0)
WGPUPowerPreference_LowPower = enum_WGPUPowerPreference.define('WGPUPowerPreference_LowPower', 1)
WGPUPowerPreference_HighPerformance = enum_WGPUPowerPreference.define('WGPUPowerPreference_HighPerformance', 2)
WGPUPowerPreference_Force32 = enum_WGPUPowerPreference.define('WGPUPowerPreference_Force32', 2147483647)

WGPUPowerPreference = enum_WGPUPowerPreference
struct_WGPUDawnAdapterPropertiesPowerPreference.SIZE = 24
struct_WGPUDawnAdapterPropertiesPowerPreference._fields_ = ['chain', 'powerPreference']
setattr(struct_WGPUDawnAdapterPropertiesPowerPreference, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUDawnAdapterPropertiesPowerPreference, 'powerPreference', field(16, WGPUPowerPreference))
class struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient(Struct): pass
struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient.SIZE = 24
struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient._fields_ = ['chain', 'outOfMemory']
setattr(struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient, 'outOfMemory', field(16, WGPUBool))
class struct_WGPUDawnEncoderInternalUsageDescriptor(Struct): pass
struct_WGPUDawnEncoderInternalUsageDescriptor.SIZE = 24
struct_WGPUDawnEncoderInternalUsageDescriptor._fields_ = ['chain', 'useInternalUsages']
setattr(struct_WGPUDawnEncoderInternalUsageDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnEncoderInternalUsageDescriptor, 'useInternalUsages', field(16, WGPUBool))
class struct_WGPUDawnExperimentalImmediateDataLimits(Struct): pass
struct_WGPUDawnExperimentalImmediateDataLimits.SIZE = 24
struct_WGPUDawnExperimentalImmediateDataLimits._fields_ = ['chain', 'maxImmediateDataRangeByteSize']
setattr(struct_WGPUDawnExperimentalImmediateDataLimits, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUDawnExperimentalImmediateDataLimits, 'maxImmediateDataRangeByteSize', field(16, uint32_t))
class struct_WGPUDawnExperimentalSubgroupLimits(Struct): pass
struct_WGPUDawnExperimentalSubgroupLimits.SIZE = 24
struct_WGPUDawnExperimentalSubgroupLimits._fields_ = ['chain', 'minSubgroupSize', 'maxSubgroupSize']
setattr(struct_WGPUDawnExperimentalSubgroupLimits, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUDawnExperimentalSubgroupLimits, 'minSubgroupSize', field(16, uint32_t))
setattr(struct_WGPUDawnExperimentalSubgroupLimits, 'maxSubgroupSize', field(20, uint32_t))
class struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled(Struct): pass
struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled.SIZE = 24
struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled._fields_ = ['chain', 'implicitSampleCount']
setattr(struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled, 'implicitSampleCount', field(16, uint32_t))
class struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor(Struct): pass
struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor.SIZE = 24
struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor._fields_ = ['chain', 'allowNonUniformDerivatives']
setattr(struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor, 'allowNonUniformDerivatives', field(16, WGPUBool))
class struct_WGPUDawnTexelCopyBufferRowAlignmentLimits(Struct): pass
struct_WGPUDawnTexelCopyBufferRowAlignmentLimits.SIZE = 24
struct_WGPUDawnTexelCopyBufferRowAlignmentLimits._fields_ = ['chain', 'minTexelCopyBufferRowAlignment']
setattr(struct_WGPUDawnTexelCopyBufferRowAlignmentLimits, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUDawnTexelCopyBufferRowAlignmentLimits, 'minTexelCopyBufferRowAlignment', field(16, uint32_t))
class struct_WGPUDawnTextureInternalUsageDescriptor(Struct): pass
WGPUTextureUsage = ctypes.c_uint64
struct_WGPUDawnTextureInternalUsageDescriptor.SIZE = 24
struct_WGPUDawnTextureInternalUsageDescriptor._fields_ = ['chain', 'internalUsage']
setattr(struct_WGPUDawnTextureInternalUsageDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnTextureInternalUsageDescriptor, 'internalUsage', field(16, WGPUTextureUsage))
class struct_WGPUDawnTogglesDescriptor(Struct): pass
struct_WGPUDawnTogglesDescriptor.SIZE = 48
struct_WGPUDawnTogglesDescriptor._fields_ = ['chain', 'enabledToggleCount', 'enabledToggles', 'disabledToggleCount', 'disabledToggles']
setattr(struct_WGPUDawnTogglesDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnTogglesDescriptor, 'enabledToggleCount', field(16, size_t))
setattr(struct_WGPUDawnTogglesDescriptor, 'enabledToggles', field(24, Pointer(Pointer(ctypes.c_char))))
setattr(struct_WGPUDawnTogglesDescriptor, 'disabledToggleCount', field(32, size_t))
setattr(struct_WGPUDawnTogglesDescriptor, 'disabledToggles', field(40, Pointer(Pointer(ctypes.c_char))))
class struct_WGPUDawnWireWGSLControl(Struct): pass
struct_WGPUDawnWireWGSLControl.SIZE = 32
struct_WGPUDawnWireWGSLControl._fields_ = ['chain', 'enableExperimental', 'enableUnsafe', 'enableTesting']
setattr(struct_WGPUDawnWireWGSLControl, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnWireWGSLControl, 'enableExperimental', field(16, WGPUBool))
setattr(struct_WGPUDawnWireWGSLControl, 'enableUnsafe', field(20, WGPUBool))
setattr(struct_WGPUDawnWireWGSLControl, 'enableTesting', field(24, WGPUBool))
class struct_WGPUDeviceLostCallbackInfo(Struct): pass
enum_WGPUDeviceLostReason = CEnum(ctypes.c_uint32)
WGPUDeviceLostReason_Unknown = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Unknown', 1)
WGPUDeviceLostReason_Destroyed = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Destroyed', 2)
WGPUDeviceLostReason_InstanceDropped = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_InstanceDropped', 3)
WGPUDeviceLostReason_FailedCreation = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_FailedCreation', 4)
WGPUDeviceLostReason_Force32 = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Force32', 2147483647)

WGPUDeviceLostCallbackNew = ctypes.CFUNCTYPE(None, Pointer(Pointer(struct_WGPUDeviceImpl)), enum_WGPUDeviceLostReason, struct_WGPUStringView, ctypes.c_void_p)
struct_WGPUDeviceLostCallbackInfo.SIZE = 32
struct_WGPUDeviceLostCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'userdata']
setattr(struct_WGPUDeviceLostCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUDeviceLostCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUDeviceLostCallbackInfo, 'callback', field(16, WGPUDeviceLostCallbackNew))
setattr(struct_WGPUDeviceLostCallbackInfo, 'userdata', field(24, ctypes.c_void_p))
class struct_WGPUDrmFormatProperties(Struct): pass
struct_WGPUDrmFormatProperties.SIZE = 16
struct_WGPUDrmFormatProperties._fields_ = ['modifier', 'modifierPlaneCount']
setattr(struct_WGPUDrmFormatProperties, 'modifier', field(0, uint64_t))
setattr(struct_WGPUDrmFormatProperties, 'modifierPlaneCount', field(8, uint32_t))
class struct_WGPUExtent2D(Struct): pass
struct_WGPUExtent2D.SIZE = 8
struct_WGPUExtent2D._fields_ = ['width', 'height']
setattr(struct_WGPUExtent2D, 'width', field(0, uint32_t))
setattr(struct_WGPUExtent2D, 'height', field(4, uint32_t))
class struct_WGPUExtent3D(Struct): pass
struct_WGPUExtent3D.SIZE = 12
struct_WGPUExtent3D._fields_ = ['width', 'height', 'depthOrArrayLayers']
setattr(struct_WGPUExtent3D, 'width', field(0, uint32_t))
setattr(struct_WGPUExtent3D, 'height', field(4, uint32_t))
setattr(struct_WGPUExtent3D, 'depthOrArrayLayers', field(8, uint32_t))
class struct_WGPUExternalTextureBindingEntry(Struct): pass
struct_WGPUExternalTextureBindingEntry.SIZE = 24
struct_WGPUExternalTextureBindingEntry._fields_ = ['chain', 'externalTexture']
setattr(struct_WGPUExternalTextureBindingEntry, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUExternalTextureBindingEntry, 'externalTexture', field(16, WGPUExternalTexture))
class struct_WGPUExternalTextureBindingLayout(Struct): pass
struct_WGPUExternalTextureBindingLayout.SIZE = 16
struct_WGPUExternalTextureBindingLayout._fields_ = ['chain']
setattr(struct_WGPUExternalTextureBindingLayout, 'chain', field(0, WGPUChainedStruct))
class struct_WGPUFormatCapabilities(Struct): pass
struct_WGPUFormatCapabilities.SIZE = 8
struct_WGPUFormatCapabilities._fields_ = ['nextInChain']
setattr(struct_WGPUFormatCapabilities, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
class struct_WGPUFuture(Struct): pass
struct_WGPUFuture.SIZE = 8
struct_WGPUFuture._fields_ = ['id']
setattr(struct_WGPUFuture, 'id', field(0, uint64_t))
class struct_WGPUInstanceFeatures(Struct): pass
struct_WGPUInstanceFeatures.SIZE = 24
struct_WGPUInstanceFeatures._fields_ = ['nextInChain', 'timedWaitAnyEnable', 'timedWaitAnyMaxCount']
setattr(struct_WGPUInstanceFeatures, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUInstanceFeatures, 'timedWaitAnyEnable', field(8, WGPUBool))
setattr(struct_WGPUInstanceFeatures, 'timedWaitAnyMaxCount', field(16, size_t))
class struct_WGPULimits(Struct): pass
struct_WGPULimits.SIZE = 160
struct_WGPULimits._fields_ = ['maxTextureDimension1D', 'maxTextureDimension2D', 'maxTextureDimension3D', 'maxTextureArrayLayers', 'maxBindGroups', 'maxBindGroupsPlusVertexBuffers', 'maxBindingsPerBindGroup', 'maxDynamicUniformBuffersPerPipelineLayout', 'maxDynamicStorageBuffersPerPipelineLayout', 'maxSampledTexturesPerShaderStage', 'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage', 'maxStorageTexturesPerShaderStage', 'maxUniformBuffersPerShaderStage', 'maxUniformBufferBindingSize', 'maxStorageBufferBindingSize', 'minUniformBufferOffsetAlignment', 'minStorageBufferOffsetAlignment', 'maxVertexBuffers', 'maxBufferSize', 'maxVertexAttributes', 'maxVertexBufferArrayStride', 'maxInterStageShaderComponents', 'maxInterStageShaderVariables', 'maxColorAttachments', 'maxColorAttachmentBytesPerSample', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupsPerDimension', 'maxStorageBuffersInVertexStage', 'maxStorageTexturesInVertexStage', 'maxStorageBuffersInFragmentStage', 'maxStorageTexturesInFragmentStage']
setattr(struct_WGPULimits, 'maxTextureDimension1D', field(0, uint32_t))
setattr(struct_WGPULimits, 'maxTextureDimension2D', field(4, uint32_t))
setattr(struct_WGPULimits, 'maxTextureDimension3D', field(8, uint32_t))
setattr(struct_WGPULimits, 'maxTextureArrayLayers', field(12, uint32_t))
setattr(struct_WGPULimits, 'maxBindGroups', field(16, uint32_t))
setattr(struct_WGPULimits, 'maxBindGroupsPlusVertexBuffers', field(20, uint32_t))
setattr(struct_WGPULimits, 'maxBindingsPerBindGroup', field(24, uint32_t))
setattr(struct_WGPULimits, 'maxDynamicUniformBuffersPerPipelineLayout', field(28, uint32_t))
setattr(struct_WGPULimits, 'maxDynamicStorageBuffersPerPipelineLayout', field(32, uint32_t))
setattr(struct_WGPULimits, 'maxSampledTexturesPerShaderStage', field(36, uint32_t))
setattr(struct_WGPULimits, 'maxSamplersPerShaderStage', field(40, uint32_t))
setattr(struct_WGPULimits, 'maxStorageBuffersPerShaderStage', field(44, uint32_t))
setattr(struct_WGPULimits, 'maxStorageTexturesPerShaderStage', field(48, uint32_t))
setattr(struct_WGPULimits, 'maxUniformBuffersPerShaderStage', field(52, uint32_t))
setattr(struct_WGPULimits, 'maxUniformBufferBindingSize', field(56, uint64_t))
setattr(struct_WGPULimits, 'maxStorageBufferBindingSize', field(64, uint64_t))
setattr(struct_WGPULimits, 'minUniformBufferOffsetAlignment', field(72, uint32_t))
setattr(struct_WGPULimits, 'minStorageBufferOffsetAlignment', field(76, uint32_t))
setattr(struct_WGPULimits, 'maxVertexBuffers', field(80, uint32_t))
setattr(struct_WGPULimits, 'maxBufferSize', field(88, uint64_t))
setattr(struct_WGPULimits, 'maxVertexAttributes', field(96, uint32_t))
setattr(struct_WGPULimits, 'maxVertexBufferArrayStride', field(100, uint32_t))
setattr(struct_WGPULimits, 'maxInterStageShaderComponents', field(104, uint32_t))
setattr(struct_WGPULimits, 'maxInterStageShaderVariables', field(108, uint32_t))
setattr(struct_WGPULimits, 'maxColorAttachments', field(112, uint32_t))
setattr(struct_WGPULimits, 'maxColorAttachmentBytesPerSample', field(116, uint32_t))
setattr(struct_WGPULimits, 'maxComputeWorkgroupStorageSize', field(120, uint32_t))
setattr(struct_WGPULimits, 'maxComputeInvocationsPerWorkgroup', field(124, uint32_t))
setattr(struct_WGPULimits, 'maxComputeWorkgroupSizeX', field(128, uint32_t))
setattr(struct_WGPULimits, 'maxComputeWorkgroupSizeY', field(132, uint32_t))
setattr(struct_WGPULimits, 'maxComputeWorkgroupSizeZ', field(136, uint32_t))
setattr(struct_WGPULimits, 'maxComputeWorkgroupsPerDimension', field(140, uint32_t))
setattr(struct_WGPULimits, 'maxStorageBuffersInVertexStage', field(144, uint32_t))
setattr(struct_WGPULimits, 'maxStorageTexturesInVertexStage', field(148, uint32_t))
setattr(struct_WGPULimits, 'maxStorageBuffersInFragmentStage', field(152, uint32_t))
setattr(struct_WGPULimits, 'maxStorageTexturesInFragmentStage', field(156, uint32_t))
class struct_WGPUMemoryHeapInfo(Struct): pass
WGPUHeapProperty = ctypes.c_uint64
struct_WGPUMemoryHeapInfo.SIZE = 16
struct_WGPUMemoryHeapInfo._fields_ = ['properties', 'size']
setattr(struct_WGPUMemoryHeapInfo, 'properties', field(0, WGPUHeapProperty))
setattr(struct_WGPUMemoryHeapInfo, 'size', field(8, uint64_t))
class struct_WGPUMultisampleState(Struct): pass
struct_WGPUMultisampleState.SIZE = 24
struct_WGPUMultisampleState._fields_ = ['nextInChain', 'count', 'mask', 'alphaToCoverageEnabled']
setattr(struct_WGPUMultisampleState, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUMultisampleState, 'count', field(8, uint32_t))
setattr(struct_WGPUMultisampleState, 'mask', field(12, uint32_t))
setattr(struct_WGPUMultisampleState, 'alphaToCoverageEnabled', field(16, WGPUBool))
class struct_WGPUOrigin2D(Struct): pass
struct_WGPUOrigin2D.SIZE = 8
struct_WGPUOrigin2D._fields_ = ['x', 'y']
setattr(struct_WGPUOrigin2D, 'x', field(0, uint32_t))
setattr(struct_WGPUOrigin2D, 'y', field(4, uint32_t))
class struct_WGPUOrigin3D(Struct): pass
struct_WGPUOrigin3D.SIZE = 12
struct_WGPUOrigin3D._fields_ = ['x', 'y', 'z']
setattr(struct_WGPUOrigin3D, 'x', field(0, uint32_t))
setattr(struct_WGPUOrigin3D, 'y', field(4, uint32_t))
setattr(struct_WGPUOrigin3D, 'z', field(8, uint32_t))
class struct_WGPUPipelineLayoutStorageAttachment(Struct): pass
enum_WGPUTextureFormat = CEnum(ctypes.c_uint32)
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
struct_WGPUPipelineLayoutStorageAttachment.SIZE = 16
struct_WGPUPipelineLayoutStorageAttachment._fields_ = ['offset', 'format']
setattr(struct_WGPUPipelineLayoutStorageAttachment, 'offset', field(0, uint64_t))
setattr(struct_WGPUPipelineLayoutStorageAttachment, 'format', field(8, WGPUTextureFormat))
class struct_WGPUPopErrorScopeCallbackInfo(Struct): pass
enum_WGPUPopErrorScopeStatus = CEnum(ctypes.c_uint32)
WGPUPopErrorScopeStatus_Success = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_Success', 1)
WGPUPopErrorScopeStatus_InstanceDropped = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_InstanceDropped', 2)
WGPUPopErrorScopeStatus_Force32 = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_Force32', 2147483647)

enum_WGPUErrorType = CEnum(ctypes.c_uint32)
WGPUErrorType_NoError = enum_WGPUErrorType.define('WGPUErrorType_NoError', 1)
WGPUErrorType_Validation = enum_WGPUErrorType.define('WGPUErrorType_Validation', 2)
WGPUErrorType_OutOfMemory = enum_WGPUErrorType.define('WGPUErrorType_OutOfMemory', 3)
WGPUErrorType_Internal = enum_WGPUErrorType.define('WGPUErrorType_Internal', 4)
WGPUErrorType_Unknown = enum_WGPUErrorType.define('WGPUErrorType_Unknown', 5)
WGPUErrorType_DeviceLost = enum_WGPUErrorType.define('WGPUErrorType_DeviceLost', 6)
WGPUErrorType_Force32 = enum_WGPUErrorType.define('WGPUErrorType_Force32', 2147483647)

WGPUPopErrorScopeCallback = ctypes.CFUNCTYPE(None, enum_WGPUPopErrorScopeStatus, enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p)
WGPUErrorCallback = ctypes.CFUNCTYPE(None, enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p)
struct_WGPUPopErrorScopeCallbackInfo.SIZE = 40
struct_WGPUPopErrorScopeCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'oldCallback', 'userdata']
setattr(struct_WGPUPopErrorScopeCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUPopErrorScopeCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUPopErrorScopeCallbackInfo, 'callback', field(16, WGPUPopErrorScopeCallback))
setattr(struct_WGPUPopErrorScopeCallbackInfo, 'oldCallback', field(24, WGPUErrorCallback))
setattr(struct_WGPUPopErrorScopeCallbackInfo, 'userdata', field(32, ctypes.c_void_p))
class struct_WGPUPrimitiveState(Struct): pass
enum_WGPUPrimitiveTopology = CEnum(ctypes.c_uint32)
WGPUPrimitiveTopology_Undefined = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_Undefined', 0)
WGPUPrimitiveTopology_PointList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_PointList', 1)
WGPUPrimitiveTopology_LineList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_LineList', 2)
WGPUPrimitiveTopology_LineStrip = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_LineStrip', 3)
WGPUPrimitiveTopology_TriangleList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_TriangleList', 4)
WGPUPrimitiveTopology_TriangleStrip = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_TriangleStrip', 5)
WGPUPrimitiveTopology_Force32 = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_Force32', 2147483647)

WGPUPrimitiveTopology = enum_WGPUPrimitiveTopology
enum_WGPUIndexFormat = CEnum(ctypes.c_uint32)
WGPUIndexFormat_Undefined = enum_WGPUIndexFormat.define('WGPUIndexFormat_Undefined', 0)
WGPUIndexFormat_Uint16 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Uint16', 1)
WGPUIndexFormat_Uint32 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Uint32', 2)
WGPUIndexFormat_Force32 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Force32', 2147483647)

WGPUIndexFormat = enum_WGPUIndexFormat
enum_WGPUFrontFace = CEnum(ctypes.c_uint32)
WGPUFrontFace_Undefined = enum_WGPUFrontFace.define('WGPUFrontFace_Undefined', 0)
WGPUFrontFace_CCW = enum_WGPUFrontFace.define('WGPUFrontFace_CCW', 1)
WGPUFrontFace_CW = enum_WGPUFrontFace.define('WGPUFrontFace_CW', 2)
WGPUFrontFace_Force32 = enum_WGPUFrontFace.define('WGPUFrontFace_Force32', 2147483647)

WGPUFrontFace = enum_WGPUFrontFace
enum_WGPUCullMode = CEnum(ctypes.c_uint32)
WGPUCullMode_Undefined = enum_WGPUCullMode.define('WGPUCullMode_Undefined', 0)
WGPUCullMode_None = enum_WGPUCullMode.define('WGPUCullMode_None', 1)
WGPUCullMode_Front = enum_WGPUCullMode.define('WGPUCullMode_Front', 2)
WGPUCullMode_Back = enum_WGPUCullMode.define('WGPUCullMode_Back', 3)
WGPUCullMode_Force32 = enum_WGPUCullMode.define('WGPUCullMode_Force32', 2147483647)

WGPUCullMode = enum_WGPUCullMode
struct_WGPUPrimitiveState.SIZE = 32
struct_WGPUPrimitiveState._fields_ = ['nextInChain', 'topology', 'stripIndexFormat', 'frontFace', 'cullMode', 'unclippedDepth']
setattr(struct_WGPUPrimitiveState, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUPrimitiveState, 'topology', field(8, WGPUPrimitiveTopology))
setattr(struct_WGPUPrimitiveState, 'stripIndexFormat', field(12, WGPUIndexFormat))
setattr(struct_WGPUPrimitiveState, 'frontFace', field(16, WGPUFrontFace))
setattr(struct_WGPUPrimitiveState, 'cullMode', field(20, WGPUCullMode))
setattr(struct_WGPUPrimitiveState, 'unclippedDepth', field(24, WGPUBool))
class struct_WGPUQueueWorkDoneCallbackInfo(Struct): pass
enum_WGPUQueueWorkDoneStatus = CEnum(ctypes.c_uint32)
WGPUQueueWorkDoneStatus_Success = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Success', 1)
WGPUQueueWorkDoneStatus_InstanceDropped = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_InstanceDropped', 2)
WGPUQueueWorkDoneStatus_Error = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Error', 3)
WGPUQueueWorkDoneStatus_Unknown = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Unknown', 4)
WGPUQueueWorkDoneStatus_DeviceLost = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_DeviceLost', 5)
WGPUQueueWorkDoneStatus_Force32 = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Force32', 2147483647)

WGPUQueueWorkDoneCallback = ctypes.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, ctypes.c_void_p)
struct_WGPUQueueWorkDoneCallbackInfo.SIZE = 32
struct_WGPUQueueWorkDoneCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'userdata']
setattr(struct_WGPUQueueWorkDoneCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUQueueWorkDoneCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUQueueWorkDoneCallbackInfo, 'callback', field(16, WGPUQueueWorkDoneCallback))
setattr(struct_WGPUQueueWorkDoneCallbackInfo, 'userdata', field(24, ctypes.c_void_p))
class struct_WGPURenderPassDepthStencilAttachment(Struct): pass
enum_WGPULoadOp = CEnum(ctypes.c_uint32)
WGPULoadOp_Undefined = enum_WGPULoadOp.define('WGPULoadOp_Undefined', 0)
WGPULoadOp_Load = enum_WGPULoadOp.define('WGPULoadOp_Load', 1)
WGPULoadOp_Clear = enum_WGPULoadOp.define('WGPULoadOp_Clear', 2)
WGPULoadOp_ExpandResolveTexture = enum_WGPULoadOp.define('WGPULoadOp_ExpandResolveTexture', 327683)
WGPULoadOp_Force32 = enum_WGPULoadOp.define('WGPULoadOp_Force32', 2147483647)

WGPULoadOp = enum_WGPULoadOp
enum_WGPUStoreOp = CEnum(ctypes.c_uint32)
WGPUStoreOp_Undefined = enum_WGPUStoreOp.define('WGPUStoreOp_Undefined', 0)
WGPUStoreOp_Store = enum_WGPUStoreOp.define('WGPUStoreOp_Store', 1)
WGPUStoreOp_Discard = enum_WGPUStoreOp.define('WGPUStoreOp_Discard', 2)
WGPUStoreOp_Force32 = enum_WGPUStoreOp.define('WGPUStoreOp_Force32', 2147483647)

WGPUStoreOp = enum_WGPUStoreOp
struct_WGPURenderPassDepthStencilAttachment.SIZE = 40
struct_WGPURenderPassDepthStencilAttachment._fields_ = ['view', 'depthLoadOp', 'depthStoreOp', 'depthClearValue', 'depthReadOnly', 'stencilLoadOp', 'stencilStoreOp', 'stencilClearValue', 'stencilReadOnly']
setattr(struct_WGPURenderPassDepthStencilAttachment, 'view', field(0, WGPUTextureView))
setattr(struct_WGPURenderPassDepthStencilAttachment, 'depthLoadOp', field(8, WGPULoadOp))
setattr(struct_WGPURenderPassDepthStencilAttachment, 'depthStoreOp', field(12, WGPUStoreOp))
setattr(struct_WGPURenderPassDepthStencilAttachment, 'depthClearValue', field(16, ctypes.c_float))
setattr(struct_WGPURenderPassDepthStencilAttachment, 'depthReadOnly', field(20, WGPUBool))
setattr(struct_WGPURenderPassDepthStencilAttachment, 'stencilLoadOp', field(24, WGPULoadOp))
setattr(struct_WGPURenderPassDepthStencilAttachment, 'stencilStoreOp', field(28, WGPUStoreOp))
setattr(struct_WGPURenderPassDepthStencilAttachment, 'stencilClearValue', field(32, uint32_t))
setattr(struct_WGPURenderPassDepthStencilAttachment, 'stencilReadOnly', field(36, WGPUBool))
class struct_WGPURenderPassDescriptorExpandResolveRect(Struct): pass
struct_WGPURenderPassDescriptorExpandResolveRect.SIZE = 32
struct_WGPURenderPassDescriptorExpandResolveRect._fields_ = ['chain', 'x', 'y', 'width', 'height']
setattr(struct_WGPURenderPassDescriptorExpandResolveRect, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPURenderPassDescriptorExpandResolveRect, 'x', field(16, uint32_t))
setattr(struct_WGPURenderPassDescriptorExpandResolveRect, 'y', field(20, uint32_t))
setattr(struct_WGPURenderPassDescriptorExpandResolveRect, 'width', field(24, uint32_t))
setattr(struct_WGPURenderPassDescriptorExpandResolveRect, 'height', field(28, uint32_t))
class struct_WGPURenderPassMaxDrawCount(Struct): pass
struct_WGPURenderPassMaxDrawCount.SIZE = 24
struct_WGPURenderPassMaxDrawCount._fields_ = ['chain', 'maxDrawCount']
setattr(struct_WGPURenderPassMaxDrawCount, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPURenderPassMaxDrawCount, 'maxDrawCount', field(16, uint64_t))
class struct_WGPURenderPassTimestampWrites(Struct): pass
struct_WGPURenderPassTimestampWrites.SIZE = 16
struct_WGPURenderPassTimestampWrites._fields_ = ['querySet', 'beginningOfPassWriteIndex', 'endOfPassWriteIndex']
setattr(struct_WGPURenderPassTimestampWrites, 'querySet', field(0, WGPUQuerySet))
setattr(struct_WGPURenderPassTimestampWrites, 'beginningOfPassWriteIndex', field(8, uint32_t))
setattr(struct_WGPURenderPassTimestampWrites, 'endOfPassWriteIndex', field(12, uint32_t))
class struct_WGPURequestAdapterCallbackInfo(Struct): pass
enum_WGPURequestAdapterStatus = CEnum(ctypes.c_uint32)
WGPURequestAdapterStatus_Success = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Success', 1)
WGPURequestAdapterStatus_InstanceDropped = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_InstanceDropped', 2)
WGPURequestAdapterStatus_Unavailable = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Unavailable', 3)
WGPURequestAdapterStatus_Error = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Error', 4)
WGPURequestAdapterStatus_Unknown = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Unknown', 5)
WGPURequestAdapterStatus_Force32 = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Force32', 2147483647)

WGPURequestAdapterCallback = ctypes.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, Pointer(struct_WGPUAdapterImpl), struct_WGPUStringView, ctypes.c_void_p)
struct_WGPURequestAdapterCallbackInfo.SIZE = 32
struct_WGPURequestAdapterCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'userdata']
setattr(struct_WGPURequestAdapterCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURequestAdapterCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPURequestAdapterCallbackInfo, 'callback', field(16, WGPURequestAdapterCallback))
setattr(struct_WGPURequestAdapterCallbackInfo, 'userdata', field(24, ctypes.c_void_p))
class struct_WGPURequestAdapterOptions(Struct): pass
enum_WGPUFeatureLevel = CEnum(ctypes.c_uint32)
WGPUFeatureLevel_Undefined = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Undefined', 0)
WGPUFeatureLevel_Compatibility = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Compatibility', 1)
WGPUFeatureLevel_Core = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Core', 2)
WGPUFeatureLevel_Force32 = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Force32', 2147483647)

WGPUFeatureLevel = enum_WGPUFeatureLevel
enum_WGPUBackendType = CEnum(ctypes.c_uint32)
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
struct_WGPURequestAdapterOptions.SIZE = 40
struct_WGPURequestAdapterOptions._fields_ = ['nextInChain', 'compatibleSurface', 'featureLevel', 'powerPreference', 'backendType', 'forceFallbackAdapter', 'compatibilityMode']
setattr(struct_WGPURequestAdapterOptions, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURequestAdapterOptions, 'compatibleSurface', field(8, WGPUSurface))
setattr(struct_WGPURequestAdapterOptions, 'featureLevel', field(16, WGPUFeatureLevel))
setattr(struct_WGPURequestAdapterOptions, 'powerPreference', field(20, WGPUPowerPreference))
setattr(struct_WGPURequestAdapterOptions, 'backendType', field(24, WGPUBackendType))
setattr(struct_WGPURequestAdapterOptions, 'forceFallbackAdapter', field(28, WGPUBool))
setattr(struct_WGPURequestAdapterOptions, 'compatibilityMode', field(32, WGPUBool))
class struct_WGPURequestDeviceCallbackInfo(Struct): pass
enum_WGPURequestDeviceStatus = CEnum(ctypes.c_uint32)
WGPURequestDeviceStatus_Success = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Success', 1)
WGPURequestDeviceStatus_InstanceDropped = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_InstanceDropped', 2)
WGPURequestDeviceStatus_Error = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Error', 3)
WGPURequestDeviceStatus_Unknown = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Unknown', 4)
WGPURequestDeviceStatus_Force32 = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Force32', 2147483647)

WGPURequestDeviceCallback = ctypes.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, Pointer(struct_WGPUDeviceImpl), struct_WGPUStringView, ctypes.c_void_p)
struct_WGPURequestDeviceCallbackInfo.SIZE = 32
struct_WGPURequestDeviceCallbackInfo._fields_ = ['nextInChain', 'mode', 'callback', 'userdata']
setattr(struct_WGPURequestDeviceCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURequestDeviceCallbackInfo, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPURequestDeviceCallbackInfo, 'callback', field(16, WGPURequestDeviceCallback))
setattr(struct_WGPURequestDeviceCallbackInfo, 'userdata', field(24, ctypes.c_void_p))
class struct_WGPUSamplerBindingLayout(Struct): pass
enum_WGPUSamplerBindingType = CEnum(ctypes.c_uint32)
WGPUSamplerBindingType_BindingNotUsed = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_BindingNotUsed', 0)
WGPUSamplerBindingType_Filtering = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Filtering', 1)
WGPUSamplerBindingType_NonFiltering = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_NonFiltering', 2)
WGPUSamplerBindingType_Comparison = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Comparison', 3)
WGPUSamplerBindingType_Force32 = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Force32', 2147483647)

WGPUSamplerBindingType = enum_WGPUSamplerBindingType
struct_WGPUSamplerBindingLayout.SIZE = 16
struct_WGPUSamplerBindingLayout._fields_ = ['nextInChain', 'type']
setattr(struct_WGPUSamplerBindingLayout, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSamplerBindingLayout, 'type', field(8, WGPUSamplerBindingType))
class struct_WGPUShaderModuleCompilationOptions(Struct): pass
struct_WGPUShaderModuleCompilationOptions.SIZE = 24
struct_WGPUShaderModuleCompilationOptions._fields_ = ['chain', 'strictMath']
setattr(struct_WGPUShaderModuleCompilationOptions, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUShaderModuleCompilationOptions, 'strictMath', field(16, WGPUBool))
class struct_WGPUShaderSourceSPIRV(Struct): pass
struct_WGPUShaderSourceSPIRV.SIZE = 32
struct_WGPUShaderSourceSPIRV._fields_ = ['chain', 'codeSize', 'code']
setattr(struct_WGPUShaderSourceSPIRV, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUShaderSourceSPIRV, 'codeSize', field(16, uint32_t))
setattr(struct_WGPUShaderSourceSPIRV, 'code', field(24, Pointer(uint32_t)))
class struct_WGPUSharedBufferMemoryBeginAccessDescriptor(Struct): pass
struct_WGPUSharedBufferMemoryBeginAccessDescriptor.SIZE = 40
struct_WGPUSharedBufferMemoryBeginAccessDescriptor._fields_ = ['nextInChain', 'initialized', 'fenceCount', 'fences', 'signaledValues']
setattr(struct_WGPUSharedBufferMemoryBeginAccessDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSharedBufferMemoryBeginAccessDescriptor, 'initialized', field(8, WGPUBool))
setattr(struct_WGPUSharedBufferMemoryBeginAccessDescriptor, 'fenceCount', field(16, size_t))
setattr(struct_WGPUSharedBufferMemoryBeginAccessDescriptor, 'fences', field(24, Pointer(WGPUSharedFence)))
setattr(struct_WGPUSharedBufferMemoryBeginAccessDescriptor, 'signaledValues', field(32, Pointer(uint64_t)))
class struct_WGPUSharedBufferMemoryEndAccessState(Struct): pass
struct_WGPUSharedBufferMemoryEndAccessState.SIZE = 40
struct_WGPUSharedBufferMemoryEndAccessState._fields_ = ['nextInChain', 'initialized', 'fenceCount', 'fences', 'signaledValues']
setattr(struct_WGPUSharedBufferMemoryEndAccessState, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
setattr(struct_WGPUSharedBufferMemoryEndAccessState, 'initialized', field(8, WGPUBool))
setattr(struct_WGPUSharedBufferMemoryEndAccessState, 'fenceCount', field(16, size_t))
setattr(struct_WGPUSharedBufferMemoryEndAccessState, 'fences', field(24, Pointer(WGPUSharedFence)))
setattr(struct_WGPUSharedBufferMemoryEndAccessState, 'signaledValues', field(32, Pointer(uint64_t)))
class struct_WGPUSharedBufferMemoryProperties(Struct): pass
WGPUBufferUsage = ctypes.c_uint64
struct_WGPUSharedBufferMemoryProperties.SIZE = 24
struct_WGPUSharedBufferMemoryProperties._fields_ = ['nextInChain', 'usage', 'size']
setattr(struct_WGPUSharedBufferMemoryProperties, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
setattr(struct_WGPUSharedBufferMemoryProperties, 'usage', field(8, WGPUBufferUsage))
setattr(struct_WGPUSharedBufferMemoryProperties, 'size', field(16, uint64_t))
class struct_WGPUSharedFenceDXGISharedHandleDescriptor(Struct): pass
struct_WGPUSharedFenceDXGISharedHandleDescriptor.SIZE = 24
struct_WGPUSharedFenceDXGISharedHandleDescriptor._fields_ = ['chain', 'handle']
setattr(struct_WGPUSharedFenceDXGISharedHandleDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedFenceDXGISharedHandleDescriptor, 'handle', field(16, ctypes.c_void_p))
class struct_WGPUSharedFenceDXGISharedHandleExportInfo(Struct): pass
struct_WGPUSharedFenceDXGISharedHandleExportInfo.SIZE = 24
struct_WGPUSharedFenceDXGISharedHandleExportInfo._fields_ = ['chain', 'handle']
setattr(struct_WGPUSharedFenceDXGISharedHandleExportInfo, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUSharedFenceDXGISharedHandleExportInfo, 'handle', field(16, ctypes.c_void_p))
class struct_WGPUSharedFenceMTLSharedEventDescriptor(Struct): pass
struct_WGPUSharedFenceMTLSharedEventDescriptor.SIZE = 24
struct_WGPUSharedFenceMTLSharedEventDescriptor._fields_ = ['chain', 'sharedEvent']
setattr(struct_WGPUSharedFenceMTLSharedEventDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedFenceMTLSharedEventDescriptor, 'sharedEvent', field(16, ctypes.c_void_p))
class struct_WGPUSharedFenceMTLSharedEventExportInfo(Struct): pass
struct_WGPUSharedFenceMTLSharedEventExportInfo.SIZE = 24
struct_WGPUSharedFenceMTLSharedEventExportInfo._fields_ = ['chain', 'sharedEvent']
setattr(struct_WGPUSharedFenceMTLSharedEventExportInfo, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUSharedFenceMTLSharedEventExportInfo, 'sharedEvent', field(16, ctypes.c_void_p))
class struct_WGPUSharedFenceExportInfo(Struct): pass
enum_WGPUSharedFenceType = CEnum(ctypes.c_uint32)
WGPUSharedFenceType_VkSemaphoreOpaqueFD = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_VkSemaphoreOpaqueFD', 1)
WGPUSharedFenceType_SyncFD = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_SyncFD', 2)
WGPUSharedFenceType_VkSemaphoreZirconHandle = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_VkSemaphoreZirconHandle', 3)
WGPUSharedFenceType_DXGISharedHandle = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_DXGISharedHandle', 4)
WGPUSharedFenceType_MTLSharedEvent = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_MTLSharedEvent', 5)
WGPUSharedFenceType_Force32 = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_Force32', 2147483647)

WGPUSharedFenceType = enum_WGPUSharedFenceType
struct_WGPUSharedFenceExportInfo.SIZE = 16
struct_WGPUSharedFenceExportInfo._fields_ = ['nextInChain', 'type']
setattr(struct_WGPUSharedFenceExportInfo, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
setattr(struct_WGPUSharedFenceExportInfo, 'type', field(8, WGPUSharedFenceType))
class struct_WGPUSharedFenceSyncFDDescriptor(Struct): pass
struct_WGPUSharedFenceSyncFDDescriptor.SIZE = 24
struct_WGPUSharedFenceSyncFDDescriptor._fields_ = ['chain', 'handle']
setattr(struct_WGPUSharedFenceSyncFDDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedFenceSyncFDDescriptor, 'handle', field(16, ctypes.c_int32))
class struct_WGPUSharedFenceSyncFDExportInfo(Struct): pass
struct_WGPUSharedFenceSyncFDExportInfo.SIZE = 24
struct_WGPUSharedFenceSyncFDExportInfo._fields_ = ['chain', 'handle']
setattr(struct_WGPUSharedFenceSyncFDExportInfo, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUSharedFenceSyncFDExportInfo, 'handle', field(16, ctypes.c_int32))
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor(Struct): pass
struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor.SIZE = 24
struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor._fields_ = ['chain', 'handle']
setattr(struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor, 'handle', field(16, ctypes.c_int32))
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo(Struct): pass
struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo.SIZE = 24
struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo._fields_ = ['chain', 'handle']
setattr(struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo, 'handle', field(16, ctypes.c_int32))
class struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor(Struct): pass
struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor.SIZE = 24
struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor._fields_ = ['chain', 'handle']
setattr(struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor, 'handle', field(16, uint32_t))
class struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo(Struct): pass
struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo.SIZE = 24
struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo._fields_ = ['chain', 'handle']
setattr(struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo, 'handle', field(16, uint32_t))
class struct_WGPUSharedTextureMemoryD3DSwapchainBeginState(Struct): pass
struct_WGPUSharedTextureMemoryD3DSwapchainBeginState.SIZE = 24
struct_WGPUSharedTextureMemoryD3DSwapchainBeginState._fields_ = ['chain', 'isSwapchain']
setattr(struct_WGPUSharedTextureMemoryD3DSwapchainBeginState, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryD3DSwapchainBeginState, 'isSwapchain', field(16, WGPUBool))
class struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor.SIZE = 32
struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor._fields_ = ['chain', 'handle', 'useKeyedMutex']
setattr(struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor, 'handle', field(16, ctypes.c_void_p))
setattr(struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor, 'useKeyedMutex', field(24, WGPUBool))
class struct_WGPUSharedTextureMemoryEGLImageDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryEGLImageDescriptor.SIZE = 24
struct_WGPUSharedTextureMemoryEGLImageDescriptor._fields_ = ['chain', 'image']
setattr(struct_WGPUSharedTextureMemoryEGLImageDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryEGLImageDescriptor, 'image', field(16, ctypes.c_void_p))
class struct_WGPUSharedTextureMemoryIOSurfaceDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryIOSurfaceDescriptor.SIZE = 24
struct_WGPUSharedTextureMemoryIOSurfaceDescriptor._fields_ = ['chain', 'ioSurface']
setattr(struct_WGPUSharedTextureMemoryIOSurfaceDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryIOSurfaceDescriptor, 'ioSurface', field(16, ctypes.c_void_p))
class struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor.SIZE = 32
struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor._fields_ = ['chain', 'handle', 'useExternalFormat']
setattr(struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor, 'handle', field(16, ctypes.c_void_p))
setattr(struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor, 'useExternalFormat', field(24, WGPUBool))
class struct_WGPUSharedTextureMemoryBeginAccessDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryBeginAccessDescriptor.SIZE = 40
struct_WGPUSharedTextureMemoryBeginAccessDescriptor._fields_ = ['nextInChain', 'concurrentRead', 'initialized', 'fenceCount', 'fences', 'signaledValues']
setattr(struct_WGPUSharedTextureMemoryBeginAccessDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSharedTextureMemoryBeginAccessDescriptor, 'concurrentRead', field(8, WGPUBool))
setattr(struct_WGPUSharedTextureMemoryBeginAccessDescriptor, 'initialized', field(12, WGPUBool))
setattr(struct_WGPUSharedTextureMemoryBeginAccessDescriptor, 'fenceCount', field(16, size_t))
setattr(struct_WGPUSharedTextureMemoryBeginAccessDescriptor, 'fences', field(24, Pointer(WGPUSharedFence)))
setattr(struct_WGPUSharedTextureMemoryBeginAccessDescriptor, 'signaledValues', field(32, Pointer(uint64_t)))
class struct_WGPUSharedTextureMemoryDmaBufPlane(Struct): pass
struct_WGPUSharedTextureMemoryDmaBufPlane.SIZE = 24
struct_WGPUSharedTextureMemoryDmaBufPlane._fields_ = ['fd', 'offset', 'stride']
setattr(struct_WGPUSharedTextureMemoryDmaBufPlane, 'fd', field(0, ctypes.c_int32))
setattr(struct_WGPUSharedTextureMemoryDmaBufPlane, 'offset', field(8, uint64_t))
setattr(struct_WGPUSharedTextureMemoryDmaBufPlane, 'stride', field(16, uint32_t))
class struct_WGPUSharedTextureMemoryEndAccessState(Struct): pass
struct_WGPUSharedTextureMemoryEndAccessState.SIZE = 40
struct_WGPUSharedTextureMemoryEndAccessState._fields_ = ['nextInChain', 'initialized', 'fenceCount', 'fences', 'signaledValues']
setattr(struct_WGPUSharedTextureMemoryEndAccessState, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
setattr(struct_WGPUSharedTextureMemoryEndAccessState, 'initialized', field(8, WGPUBool))
setattr(struct_WGPUSharedTextureMemoryEndAccessState, 'fenceCount', field(16, size_t))
setattr(struct_WGPUSharedTextureMemoryEndAccessState, 'fences', field(24, Pointer(WGPUSharedFence)))
setattr(struct_WGPUSharedTextureMemoryEndAccessState, 'signaledValues', field(32, Pointer(uint64_t)))
class struct_WGPUSharedTextureMemoryOpaqueFDDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryOpaqueFDDescriptor.SIZE = 48
struct_WGPUSharedTextureMemoryOpaqueFDDescriptor._fields_ = ['chain', 'vkImageCreateInfo', 'memoryFD', 'memoryTypeIndex', 'allocationSize', 'dedicatedAllocation']
setattr(struct_WGPUSharedTextureMemoryOpaqueFDDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryOpaqueFDDescriptor, 'vkImageCreateInfo', field(16, ctypes.c_void_p))
setattr(struct_WGPUSharedTextureMemoryOpaqueFDDescriptor, 'memoryFD', field(24, ctypes.c_int32))
setattr(struct_WGPUSharedTextureMemoryOpaqueFDDescriptor, 'memoryTypeIndex', field(28, uint32_t))
setattr(struct_WGPUSharedTextureMemoryOpaqueFDDescriptor, 'allocationSize', field(32, uint64_t))
setattr(struct_WGPUSharedTextureMemoryOpaqueFDDescriptor, 'dedicatedAllocation', field(40, WGPUBool))
class struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor.SIZE = 24
struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor._fields_ = ['chain', 'dedicatedAllocation']
setattr(struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor, 'dedicatedAllocation', field(16, WGPUBool))
class struct_WGPUSharedTextureMemoryVkImageLayoutBeginState(Struct): pass
int32_t = ctypes.c_int32
struct_WGPUSharedTextureMemoryVkImageLayoutBeginState.SIZE = 24
struct_WGPUSharedTextureMemoryVkImageLayoutBeginState._fields_ = ['chain', 'oldLayout', 'newLayout']
setattr(struct_WGPUSharedTextureMemoryVkImageLayoutBeginState, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryVkImageLayoutBeginState, 'oldLayout', field(16, int32_t))
setattr(struct_WGPUSharedTextureMemoryVkImageLayoutBeginState, 'newLayout', field(20, int32_t))
class struct_WGPUSharedTextureMemoryVkImageLayoutEndState(Struct): pass
struct_WGPUSharedTextureMemoryVkImageLayoutEndState.SIZE = 24
struct_WGPUSharedTextureMemoryVkImageLayoutEndState._fields_ = ['chain', 'oldLayout', 'newLayout']
setattr(struct_WGPUSharedTextureMemoryVkImageLayoutEndState, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUSharedTextureMemoryVkImageLayoutEndState, 'oldLayout', field(16, int32_t))
setattr(struct_WGPUSharedTextureMemoryVkImageLayoutEndState, 'newLayout', field(20, int32_t))
class struct_WGPUSharedTextureMemoryZirconHandleDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryZirconHandleDescriptor.SIZE = 32
struct_WGPUSharedTextureMemoryZirconHandleDescriptor._fields_ = ['chain', 'memoryFD', 'allocationSize']
setattr(struct_WGPUSharedTextureMemoryZirconHandleDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryZirconHandleDescriptor, 'memoryFD', field(16, uint32_t))
setattr(struct_WGPUSharedTextureMemoryZirconHandleDescriptor, 'allocationSize', field(24, uint64_t))
class struct_WGPUStaticSamplerBindingLayout(Struct): pass
struct_WGPUStaticSamplerBindingLayout.SIZE = 32
struct_WGPUStaticSamplerBindingLayout._fields_ = ['chain', 'sampler', 'sampledTextureBinding']
setattr(struct_WGPUStaticSamplerBindingLayout, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUStaticSamplerBindingLayout, 'sampler', field(16, WGPUSampler))
setattr(struct_WGPUStaticSamplerBindingLayout, 'sampledTextureBinding', field(24, uint32_t))
class struct_WGPUStencilFaceState(Struct): pass
enum_WGPUCompareFunction = CEnum(ctypes.c_uint32)
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
enum_WGPUStencilOperation = CEnum(ctypes.c_uint32)
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
struct_WGPUStencilFaceState.SIZE = 16
struct_WGPUStencilFaceState._fields_ = ['compare', 'failOp', 'depthFailOp', 'passOp']
setattr(struct_WGPUStencilFaceState, 'compare', field(0, WGPUCompareFunction))
setattr(struct_WGPUStencilFaceState, 'failOp', field(4, WGPUStencilOperation))
setattr(struct_WGPUStencilFaceState, 'depthFailOp', field(8, WGPUStencilOperation))
setattr(struct_WGPUStencilFaceState, 'passOp', field(12, WGPUStencilOperation))
class struct_WGPUStorageTextureBindingLayout(Struct): pass
enum_WGPUStorageTextureAccess = CEnum(ctypes.c_uint32)
WGPUStorageTextureAccess_BindingNotUsed = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_BindingNotUsed', 0)
WGPUStorageTextureAccess_WriteOnly = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_WriteOnly', 1)
WGPUStorageTextureAccess_ReadOnly = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_ReadOnly', 2)
WGPUStorageTextureAccess_ReadWrite = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_ReadWrite', 3)
WGPUStorageTextureAccess_Force32 = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_Force32', 2147483647)

WGPUStorageTextureAccess = enum_WGPUStorageTextureAccess
enum_WGPUTextureViewDimension = CEnum(ctypes.c_uint32)
WGPUTextureViewDimension_Undefined = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Undefined', 0)
WGPUTextureViewDimension_1D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_1D', 1)
WGPUTextureViewDimension_2D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_2D', 2)
WGPUTextureViewDimension_2DArray = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_2DArray', 3)
WGPUTextureViewDimension_Cube = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Cube', 4)
WGPUTextureViewDimension_CubeArray = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_CubeArray', 5)
WGPUTextureViewDimension_3D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_3D', 6)
WGPUTextureViewDimension_Force32 = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Force32', 2147483647)

WGPUTextureViewDimension = enum_WGPUTextureViewDimension
struct_WGPUStorageTextureBindingLayout.SIZE = 24
struct_WGPUStorageTextureBindingLayout._fields_ = ['nextInChain', 'access', 'format', 'viewDimension']
setattr(struct_WGPUStorageTextureBindingLayout, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUStorageTextureBindingLayout, 'access', field(8, WGPUStorageTextureAccess))
setattr(struct_WGPUStorageTextureBindingLayout, 'format', field(12, WGPUTextureFormat))
setattr(struct_WGPUStorageTextureBindingLayout, 'viewDimension', field(16, WGPUTextureViewDimension))
class struct_WGPUSupportedFeatures(Struct): pass
enum_WGPUFeatureName = CEnum(ctypes.c_uint32)
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
struct_WGPUSupportedFeatures.SIZE = 16
struct_WGPUSupportedFeatures._fields_ = ['featureCount', 'features']
setattr(struct_WGPUSupportedFeatures, 'featureCount', field(0, size_t))
setattr(struct_WGPUSupportedFeatures, 'features', field(8, Pointer(WGPUFeatureName)))
class struct_WGPUSurfaceCapabilities(Struct): pass
enum_WGPUPresentMode = CEnum(ctypes.c_uint32)
WGPUPresentMode_Fifo = enum_WGPUPresentMode.define('WGPUPresentMode_Fifo', 1)
WGPUPresentMode_FifoRelaxed = enum_WGPUPresentMode.define('WGPUPresentMode_FifoRelaxed', 2)
WGPUPresentMode_Immediate = enum_WGPUPresentMode.define('WGPUPresentMode_Immediate', 3)
WGPUPresentMode_Mailbox = enum_WGPUPresentMode.define('WGPUPresentMode_Mailbox', 4)
WGPUPresentMode_Force32 = enum_WGPUPresentMode.define('WGPUPresentMode_Force32', 2147483647)

WGPUPresentMode = enum_WGPUPresentMode
enum_WGPUCompositeAlphaMode = CEnum(ctypes.c_uint32)
WGPUCompositeAlphaMode_Auto = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Auto', 0)
WGPUCompositeAlphaMode_Opaque = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Opaque', 1)
WGPUCompositeAlphaMode_Premultiplied = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Premultiplied', 2)
WGPUCompositeAlphaMode_Unpremultiplied = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Unpremultiplied', 3)
WGPUCompositeAlphaMode_Inherit = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Inherit', 4)
WGPUCompositeAlphaMode_Force32 = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Force32', 2147483647)

WGPUCompositeAlphaMode = enum_WGPUCompositeAlphaMode
struct_WGPUSurfaceCapabilities.SIZE = 64
struct_WGPUSurfaceCapabilities._fields_ = ['nextInChain', 'usages', 'formatCount', 'formats', 'presentModeCount', 'presentModes', 'alphaModeCount', 'alphaModes']
setattr(struct_WGPUSurfaceCapabilities, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
setattr(struct_WGPUSurfaceCapabilities, 'usages', field(8, WGPUTextureUsage))
setattr(struct_WGPUSurfaceCapabilities, 'formatCount', field(16, size_t))
setattr(struct_WGPUSurfaceCapabilities, 'formats', field(24, Pointer(WGPUTextureFormat)))
setattr(struct_WGPUSurfaceCapabilities, 'presentModeCount', field(32, size_t))
setattr(struct_WGPUSurfaceCapabilities, 'presentModes', field(40, Pointer(WGPUPresentMode)))
setattr(struct_WGPUSurfaceCapabilities, 'alphaModeCount', field(48, size_t))
setattr(struct_WGPUSurfaceCapabilities, 'alphaModes', field(56, Pointer(WGPUCompositeAlphaMode)))
class struct_WGPUSurfaceConfiguration(Struct): pass
struct_WGPUSurfaceConfiguration.SIZE = 64
struct_WGPUSurfaceConfiguration._fields_ = ['nextInChain', 'device', 'format', 'usage', 'viewFormatCount', 'viewFormats', 'alphaMode', 'width', 'height', 'presentMode']
setattr(struct_WGPUSurfaceConfiguration, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSurfaceConfiguration, 'device', field(8, WGPUDevice))
setattr(struct_WGPUSurfaceConfiguration, 'format', field(16, WGPUTextureFormat))
setattr(struct_WGPUSurfaceConfiguration, 'usage', field(24, WGPUTextureUsage))
setattr(struct_WGPUSurfaceConfiguration, 'viewFormatCount', field(32, size_t))
setattr(struct_WGPUSurfaceConfiguration, 'viewFormats', field(40, Pointer(WGPUTextureFormat)))
setattr(struct_WGPUSurfaceConfiguration, 'alphaMode', field(48, WGPUCompositeAlphaMode))
setattr(struct_WGPUSurfaceConfiguration, 'width', field(52, uint32_t))
setattr(struct_WGPUSurfaceConfiguration, 'height', field(56, uint32_t))
setattr(struct_WGPUSurfaceConfiguration, 'presentMode', field(60, WGPUPresentMode))
class struct_WGPUSurfaceDescriptorFromWindowsCoreWindow(Struct): pass
struct_WGPUSurfaceDescriptorFromWindowsCoreWindow.SIZE = 24
struct_WGPUSurfaceDescriptorFromWindowsCoreWindow._fields_ = ['chain', 'coreWindow']
setattr(struct_WGPUSurfaceDescriptorFromWindowsCoreWindow, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceDescriptorFromWindowsCoreWindow, 'coreWindow', field(16, ctypes.c_void_p))
class struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel(Struct): pass
struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel.SIZE = 24
struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel._fields_ = ['chain', 'swapChainPanel']
setattr(struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel, 'swapChainPanel', field(16, ctypes.c_void_p))
class struct_WGPUSurfaceSourceXCBWindow(Struct): pass
struct_WGPUSurfaceSourceXCBWindow.SIZE = 32
struct_WGPUSurfaceSourceXCBWindow._fields_ = ['chain', 'connection', 'window']
setattr(struct_WGPUSurfaceSourceXCBWindow, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceSourceXCBWindow, 'connection', field(16, ctypes.c_void_p))
setattr(struct_WGPUSurfaceSourceXCBWindow, 'window', field(24, uint32_t))
class struct_WGPUSurfaceSourceAndroidNativeWindow(Struct): pass
struct_WGPUSurfaceSourceAndroidNativeWindow.SIZE = 24
struct_WGPUSurfaceSourceAndroidNativeWindow._fields_ = ['chain', 'window']
setattr(struct_WGPUSurfaceSourceAndroidNativeWindow, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceSourceAndroidNativeWindow, 'window', field(16, ctypes.c_void_p))
class struct_WGPUSurfaceSourceMetalLayer(Struct): pass
struct_WGPUSurfaceSourceMetalLayer.SIZE = 24
struct_WGPUSurfaceSourceMetalLayer._fields_ = ['chain', 'layer']
setattr(struct_WGPUSurfaceSourceMetalLayer, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceSourceMetalLayer, 'layer', field(16, ctypes.c_void_p))
class struct_WGPUSurfaceSourceWaylandSurface(Struct): pass
struct_WGPUSurfaceSourceWaylandSurface.SIZE = 32
struct_WGPUSurfaceSourceWaylandSurface._fields_ = ['chain', 'display', 'surface']
setattr(struct_WGPUSurfaceSourceWaylandSurface, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceSourceWaylandSurface, 'display', field(16, ctypes.c_void_p))
setattr(struct_WGPUSurfaceSourceWaylandSurface, 'surface', field(24, ctypes.c_void_p))
class struct_WGPUSurfaceSourceWindowsHWND(Struct): pass
struct_WGPUSurfaceSourceWindowsHWND.SIZE = 32
struct_WGPUSurfaceSourceWindowsHWND._fields_ = ['chain', 'hinstance', 'hwnd']
setattr(struct_WGPUSurfaceSourceWindowsHWND, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceSourceWindowsHWND, 'hinstance', field(16, ctypes.c_void_p))
setattr(struct_WGPUSurfaceSourceWindowsHWND, 'hwnd', field(24, ctypes.c_void_p))
class struct_WGPUSurfaceSourceXlibWindow(Struct): pass
struct_WGPUSurfaceSourceXlibWindow.SIZE = 32
struct_WGPUSurfaceSourceXlibWindow._fields_ = ['chain', 'display', 'window']
setattr(struct_WGPUSurfaceSourceXlibWindow, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceSourceXlibWindow, 'display', field(16, ctypes.c_void_p))
setattr(struct_WGPUSurfaceSourceXlibWindow, 'window', field(24, uint64_t))
class struct_WGPUSurfaceTexture(Struct): pass
enum_WGPUSurfaceGetCurrentTextureStatus = CEnum(ctypes.c_uint32)
WGPUSurfaceGetCurrentTextureStatus_Success = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Success', 1)
WGPUSurfaceGetCurrentTextureStatus_Timeout = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Timeout', 2)
WGPUSurfaceGetCurrentTextureStatus_Outdated = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Outdated', 3)
WGPUSurfaceGetCurrentTextureStatus_Lost = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Lost', 4)
WGPUSurfaceGetCurrentTextureStatus_OutOfMemory = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_OutOfMemory', 5)
WGPUSurfaceGetCurrentTextureStatus_DeviceLost = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_DeviceLost', 6)
WGPUSurfaceGetCurrentTextureStatus_Error = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Error', 7)
WGPUSurfaceGetCurrentTextureStatus_Force32 = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Force32', 2147483647)

WGPUSurfaceGetCurrentTextureStatus = enum_WGPUSurfaceGetCurrentTextureStatus
struct_WGPUSurfaceTexture.SIZE = 16
struct_WGPUSurfaceTexture._fields_ = ['texture', 'suboptimal', 'status']
setattr(struct_WGPUSurfaceTexture, 'texture', field(0, WGPUTexture))
setattr(struct_WGPUSurfaceTexture, 'suboptimal', field(8, WGPUBool))
setattr(struct_WGPUSurfaceTexture, 'status', field(12, WGPUSurfaceGetCurrentTextureStatus))
class struct_WGPUTextureBindingLayout(Struct): pass
enum_WGPUTextureSampleType = CEnum(ctypes.c_uint32)
WGPUTextureSampleType_BindingNotUsed = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_BindingNotUsed', 0)
WGPUTextureSampleType_Float = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Float', 1)
WGPUTextureSampleType_UnfilterableFloat = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_UnfilterableFloat', 2)
WGPUTextureSampleType_Depth = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Depth', 3)
WGPUTextureSampleType_Sint = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Sint', 4)
WGPUTextureSampleType_Uint = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Uint', 5)
WGPUTextureSampleType_Force32 = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Force32', 2147483647)

WGPUTextureSampleType = enum_WGPUTextureSampleType
struct_WGPUTextureBindingLayout.SIZE = 24
struct_WGPUTextureBindingLayout._fields_ = ['nextInChain', 'sampleType', 'viewDimension', 'multisampled']
setattr(struct_WGPUTextureBindingLayout, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUTextureBindingLayout, 'sampleType', field(8, WGPUTextureSampleType))
setattr(struct_WGPUTextureBindingLayout, 'viewDimension', field(12, WGPUTextureViewDimension))
setattr(struct_WGPUTextureBindingLayout, 'multisampled', field(16, WGPUBool))
class struct_WGPUTextureBindingViewDimensionDescriptor(Struct): pass
struct_WGPUTextureBindingViewDimensionDescriptor.SIZE = 24
struct_WGPUTextureBindingViewDimensionDescriptor._fields_ = ['chain', 'textureBindingViewDimension']
setattr(struct_WGPUTextureBindingViewDimensionDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUTextureBindingViewDimensionDescriptor, 'textureBindingViewDimension', field(16, WGPUTextureViewDimension))
class struct_WGPUTextureDataLayout(Struct): pass
struct_WGPUTextureDataLayout.SIZE = 24
struct_WGPUTextureDataLayout._fields_ = ['nextInChain', 'offset', 'bytesPerRow', 'rowsPerImage']
setattr(struct_WGPUTextureDataLayout, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUTextureDataLayout, 'offset', field(8, uint64_t))
setattr(struct_WGPUTextureDataLayout, 'bytesPerRow', field(16, uint32_t))
setattr(struct_WGPUTextureDataLayout, 'rowsPerImage', field(20, uint32_t))
class struct_WGPUUncapturedErrorCallbackInfo(Struct): pass
struct_WGPUUncapturedErrorCallbackInfo.SIZE = 24
struct_WGPUUncapturedErrorCallbackInfo._fields_ = ['nextInChain', 'callback', 'userdata']
setattr(struct_WGPUUncapturedErrorCallbackInfo, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUUncapturedErrorCallbackInfo, 'callback', field(8, WGPUErrorCallback))
setattr(struct_WGPUUncapturedErrorCallbackInfo, 'userdata', field(16, ctypes.c_void_p))
class struct_WGPUVertexAttribute(Struct): pass
enum_WGPUVertexFormat = CEnum(ctypes.c_uint32)
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
struct_WGPUVertexAttribute.SIZE = 24
struct_WGPUVertexAttribute._fields_ = ['format', 'offset', 'shaderLocation']
setattr(struct_WGPUVertexAttribute, 'format', field(0, WGPUVertexFormat))
setattr(struct_WGPUVertexAttribute, 'offset', field(8, uint64_t))
setattr(struct_WGPUVertexAttribute, 'shaderLocation', field(16, uint32_t))
class struct_WGPUYCbCrVkDescriptor(Struct): pass
enum_WGPUFilterMode = CEnum(ctypes.c_uint32)
WGPUFilterMode_Undefined = enum_WGPUFilterMode.define('WGPUFilterMode_Undefined', 0)
WGPUFilterMode_Nearest = enum_WGPUFilterMode.define('WGPUFilterMode_Nearest', 1)
WGPUFilterMode_Linear = enum_WGPUFilterMode.define('WGPUFilterMode_Linear', 2)
WGPUFilterMode_Force32 = enum_WGPUFilterMode.define('WGPUFilterMode_Force32', 2147483647)

WGPUFilterMode = enum_WGPUFilterMode
struct_WGPUYCbCrVkDescriptor.SIZE = 72
struct_WGPUYCbCrVkDescriptor._fields_ = ['chain', 'vkFormat', 'vkYCbCrModel', 'vkYCbCrRange', 'vkComponentSwizzleRed', 'vkComponentSwizzleGreen', 'vkComponentSwizzleBlue', 'vkComponentSwizzleAlpha', 'vkXChromaOffset', 'vkYChromaOffset', 'vkChromaFilter', 'forceExplicitReconstruction', 'externalFormat']
setattr(struct_WGPUYCbCrVkDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkFormat', field(16, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkYCbCrModel', field(20, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkYCbCrRange', field(24, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkComponentSwizzleRed', field(28, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkComponentSwizzleGreen', field(32, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkComponentSwizzleBlue', field(36, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkComponentSwizzleAlpha', field(40, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkXChromaOffset', field(44, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkYChromaOffset', field(48, uint32_t))
setattr(struct_WGPUYCbCrVkDescriptor, 'vkChromaFilter', field(52, WGPUFilterMode))
setattr(struct_WGPUYCbCrVkDescriptor, 'forceExplicitReconstruction', field(56, WGPUBool))
setattr(struct_WGPUYCbCrVkDescriptor, 'externalFormat', field(64, uint64_t))
class struct_WGPUAHardwareBufferProperties(Struct): pass
WGPUYCbCrVkDescriptor = struct_WGPUYCbCrVkDescriptor
struct_WGPUAHardwareBufferProperties.SIZE = 72
struct_WGPUAHardwareBufferProperties._fields_ = ['yCbCrInfo']
setattr(struct_WGPUAHardwareBufferProperties, 'yCbCrInfo', field(0, WGPUYCbCrVkDescriptor))
class struct_WGPUAdapterInfo(Struct): pass
enum_WGPUAdapterType = CEnum(ctypes.c_uint32)
WGPUAdapterType_DiscreteGPU = enum_WGPUAdapterType.define('WGPUAdapterType_DiscreteGPU', 1)
WGPUAdapterType_IntegratedGPU = enum_WGPUAdapterType.define('WGPUAdapterType_IntegratedGPU', 2)
WGPUAdapterType_CPU = enum_WGPUAdapterType.define('WGPUAdapterType_CPU', 3)
WGPUAdapterType_Unknown = enum_WGPUAdapterType.define('WGPUAdapterType_Unknown', 4)
WGPUAdapterType_Force32 = enum_WGPUAdapterType.define('WGPUAdapterType_Force32', 2147483647)

WGPUAdapterType = enum_WGPUAdapterType
struct_WGPUAdapterInfo.SIZE = 96
struct_WGPUAdapterInfo._fields_ = ['nextInChain', 'vendor', 'architecture', 'device', 'description', 'backendType', 'adapterType', 'vendorID', 'deviceID', 'compatibilityMode']
setattr(struct_WGPUAdapterInfo, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
setattr(struct_WGPUAdapterInfo, 'vendor', field(8, WGPUStringView))
setattr(struct_WGPUAdapterInfo, 'architecture', field(24, WGPUStringView))
setattr(struct_WGPUAdapterInfo, 'device', field(40, WGPUStringView))
setattr(struct_WGPUAdapterInfo, 'description', field(56, WGPUStringView))
setattr(struct_WGPUAdapterInfo, 'backendType', field(72, WGPUBackendType))
setattr(struct_WGPUAdapterInfo, 'adapterType', field(76, WGPUAdapterType))
setattr(struct_WGPUAdapterInfo, 'vendorID', field(80, uint32_t))
setattr(struct_WGPUAdapterInfo, 'deviceID', field(84, uint32_t))
setattr(struct_WGPUAdapterInfo, 'compatibilityMode', field(88, WGPUBool))
class struct_WGPUAdapterPropertiesMemoryHeaps(Struct): pass
WGPUMemoryHeapInfo = struct_WGPUMemoryHeapInfo
struct_WGPUAdapterPropertiesMemoryHeaps.SIZE = 32
struct_WGPUAdapterPropertiesMemoryHeaps._fields_ = ['chain', 'heapCount', 'heapInfo']
setattr(struct_WGPUAdapterPropertiesMemoryHeaps, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUAdapterPropertiesMemoryHeaps, 'heapCount', field(16, size_t))
setattr(struct_WGPUAdapterPropertiesMemoryHeaps, 'heapInfo', field(24, Pointer(WGPUMemoryHeapInfo)))
class struct_WGPUBindGroupDescriptor(Struct): pass
WGPUBindGroupEntry = struct_WGPUBindGroupEntry
struct_WGPUBindGroupDescriptor.SIZE = 48
struct_WGPUBindGroupDescriptor._fields_ = ['nextInChain', 'label', 'layout', 'entryCount', 'entries']
setattr(struct_WGPUBindGroupDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUBindGroupDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUBindGroupDescriptor, 'layout', field(24, WGPUBindGroupLayout))
setattr(struct_WGPUBindGroupDescriptor, 'entryCount', field(32, size_t))
setattr(struct_WGPUBindGroupDescriptor, 'entries', field(40, Pointer(WGPUBindGroupEntry)))
class struct_WGPUBindGroupLayoutEntry(Struct): pass
WGPUShaderStage = ctypes.c_uint64
WGPUBufferBindingLayout = struct_WGPUBufferBindingLayout
WGPUSamplerBindingLayout = struct_WGPUSamplerBindingLayout
WGPUTextureBindingLayout = struct_WGPUTextureBindingLayout
WGPUStorageTextureBindingLayout = struct_WGPUStorageTextureBindingLayout
struct_WGPUBindGroupLayoutEntry.SIZE = 112
struct_WGPUBindGroupLayoutEntry._fields_ = ['nextInChain', 'binding', 'visibility', 'buffer', 'sampler', 'texture', 'storageTexture']
setattr(struct_WGPUBindGroupLayoutEntry, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUBindGroupLayoutEntry, 'binding', field(8, uint32_t))
setattr(struct_WGPUBindGroupLayoutEntry, 'visibility', field(16, WGPUShaderStage))
setattr(struct_WGPUBindGroupLayoutEntry, 'buffer', field(24, WGPUBufferBindingLayout))
setattr(struct_WGPUBindGroupLayoutEntry, 'sampler', field(48, WGPUSamplerBindingLayout))
setattr(struct_WGPUBindGroupLayoutEntry, 'texture', field(64, WGPUTextureBindingLayout))
setattr(struct_WGPUBindGroupLayoutEntry, 'storageTexture', field(88, WGPUStorageTextureBindingLayout))
class struct_WGPUBlendState(Struct): pass
WGPUBlendComponent = struct_WGPUBlendComponent
struct_WGPUBlendState.SIZE = 24
struct_WGPUBlendState._fields_ = ['color', 'alpha']
setattr(struct_WGPUBlendState, 'color', field(0, WGPUBlendComponent))
setattr(struct_WGPUBlendState, 'alpha', field(12, WGPUBlendComponent))
class struct_WGPUBufferDescriptor(Struct): pass
struct_WGPUBufferDescriptor.SIZE = 48
struct_WGPUBufferDescriptor._fields_ = ['nextInChain', 'label', 'usage', 'size', 'mappedAtCreation']
setattr(struct_WGPUBufferDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUBufferDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUBufferDescriptor, 'usage', field(24, WGPUBufferUsage))
setattr(struct_WGPUBufferDescriptor, 'size', field(32, uint64_t))
setattr(struct_WGPUBufferDescriptor, 'mappedAtCreation', field(40, WGPUBool))
class struct_WGPUCommandBufferDescriptor(Struct): pass
struct_WGPUCommandBufferDescriptor.SIZE = 24
struct_WGPUCommandBufferDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPUCommandBufferDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCommandBufferDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPUCommandEncoderDescriptor(Struct): pass
struct_WGPUCommandEncoderDescriptor.SIZE = 24
struct_WGPUCommandEncoderDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPUCommandEncoderDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCommandEncoderDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPUComputePassDescriptor(Struct): pass
WGPUComputePassTimestampWrites = struct_WGPUComputePassTimestampWrites
struct_WGPUComputePassDescriptor.SIZE = 32
struct_WGPUComputePassDescriptor._fields_ = ['nextInChain', 'label', 'timestampWrites']
setattr(struct_WGPUComputePassDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUComputePassDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUComputePassDescriptor, 'timestampWrites', field(24, Pointer(WGPUComputePassTimestampWrites)))
class struct_WGPUConstantEntry(Struct): pass
struct_WGPUConstantEntry.SIZE = 32
struct_WGPUConstantEntry._fields_ = ['nextInChain', 'key', 'value']
setattr(struct_WGPUConstantEntry, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUConstantEntry, 'key', field(8, WGPUStringView))
setattr(struct_WGPUConstantEntry, 'value', field(24, ctypes.c_double))
class struct_WGPUDawnCacheDeviceDescriptor(Struct): pass
WGPUDawnLoadCacheDataFunction = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)
WGPUDawnStoreCacheDataFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)
struct_WGPUDawnCacheDeviceDescriptor.SIZE = 56
struct_WGPUDawnCacheDeviceDescriptor._fields_ = ['chain', 'isolationKey', 'loadDataFunction', 'storeDataFunction', 'functionUserdata']
setattr(struct_WGPUDawnCacheDeviceDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUDawnCacheDeviceDescriptor, 'isolationKey', field(16, WGPUStringView))
setattr(struct_WGPUDawnCacheDeviceDescriptor, 'loadDataFunction', field(32, WGPUDawnLoadCacheDataFunction))
setattr(struct_WGPUDawnCacheDeviceDescriptor, 'storeDataFunction', field(40, WGPUDawnStoreCacheDataFunction))
setattr(struct_WGPUDawnCacheDeviceDescriptor, 'functionUserdata', field(48, ctypes.c_void_p))
class struct_WGPUDepthStencilState(Struct): pass
enum_WGPUOptionalBool = CEnum(ctypes.c_uint32)
WGPUOptionalBool_False = enum_WGPUOptionalBool.define('WGPUOptionalBool_False', 0)
WGPUOptionalBool_True = enum_WGPUOptionalBool.define('WGPUOptionalBool_True', 1)
WGPUOptionalBool_Undefined = enum_WGPUOptionalBool.define('WGPUOptionalBool_Undefined', 2)
WGPUOptionalBool_Force32 = enum_WGPUOptionalBool.define('WGPUOptionalBool_Force32', 2147483647)

WGPUOptionalBool = enum_WGPUOptionalBool
WGPUStencilFaceState = struct_WGPUStencilFaceState
struct_WGPUDepthStencilState.SIZE = 72
struct_WGPUDepthStencilState._fields_ = ['nextInChain', 'format', 'depthWriteEnabled', 'depthCompare', 'stencilFront', 'stencilBack', 'stencilReadMask', 'stencilWriteMask', 'depthBias', 'depthBiasSlopeScale', 'depthBiasClamp']
setattr(struct_WGPUDepthStencilState, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUDepthStencilState, 'format', field(8, WGPUTextureFormat))
setattr(struct_WGPUDepthStencilState, 'depthWriteEnabled', field(12, WGPUOptionalBool))
setattr(struct_WGPUDepthStencilState, 'depthCompare', field(16, WGPUCompareFunction))
setattr(struct_WGPUDepthStencilState, 'stencilFront', field(20, WGPUStencilFaceState))
setattr(struct_WGPUDepthStencilState, 'stencilBack', field(36, WGPUStencilFaceState))
setattr(struct_WGPUDepthStencilState, 'stencilReadMask', field(52, uint32_t))
setattr(struct_WGPUDepthStencilState, 'stencilWriteMask', field(56, uint32_t))
setattr(struct_WGPUDepthStencilState, 'depthBias', field(60, int32_t))
setattr(struct_WGPUDepthStencilState, 'depthBiasSlopeScale', field(64, ctypes.c_float))
setattr(struct_WGPUDepthStencilState, 'depthBiasClamp', field(68, ctypes.c_float))
class struct_WGPUDrmFormatCapabilities(Struct): pass
WGPUDrmFormatProperties = struct_WGPUDrmFormatProperties
struct_WGPUDrmFormatCapabilities.SIZE = 32
struct_WGPUDrmFormatCapabilities._fields_ = ['chain', 'propertiesCount', 'properties']
setattr(struct_WGPUDrmFormatCapabilities, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUDrmFormatCapabilities, 'propertiesCount', field(16, size_t))
setattr(struct_WGPUDrmFormatCapabilities, 'properties', field(24, Pointer(WGPUDrmFormatProperties)))
class struct_WGPUExternalTextureDescriptor(Struct): pass
WGPUOrigin2D = struct_WGPUOrigin2D
WGPUExtent2D = struct_WGPUExtent2D
enum_WGPUExternalTextureRotation = CEnum(ctypes.c_uint32)
WGPUExternalTextureRotation_Rotate0Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate0Degrees', 1)
WGPUExternalTextureRotation_Rotate90Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate90Degrees', 2)
WGPUExternalTextureRotation_Rotate180Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate180Degrees', 3)
WGPUExternalTextureRotation_Rotate270Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate270Degrees', 4)
WGPUExternalTextureRotation_Force32 = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Force32', 2147483647)

WGPUExternalTextureRotation = enum_WGPUExternalTextureRotation
struct_WGPUExternalTextureDescriptor.SIZE = 112
struct_WGPUExternalTextureDescriptor._fields_ = ['nextInChain', 'label', 'plane0', 'plane1', 'cropOrigin', 'cropSize', 'apparentSize', 'doYuvToRgbConversionOnly', 'yuvToRgbConversionMatrix', 'srcTransferFunctionParameters', 'dstTransferFunctionParameters', 'gamutConversionMatrix', 'mirrored', 'rotation']
setattr(struct_WGPUExternalTextureDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUExternalTextureDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUExternalTextureDescriptor, 'plane0', field(24, WGPUTextureView))
setattr(struct_WGPUExternalTextureDescriptor, 'plane1', field(32, WGPUTextureView))
setattr(struct_WGPUExternalTextureDescriptor, 'cropOrigin', field(40, WGPUOrigin2D))
setattr(struct_WGPUExternalTextureDescriptor, 'cropSize', field(48, WGPUExtent2D))
setattr(struct_WGPUExternalTextureDescriptor, 'apparentSize', field(56, WGPUExtent2D))
setattr(struct_WGPUExternalTextureDescriptor, 'doYuvToRgbConversionOnly', field(64, WGPUBool))
setattr(struct_WGPUExternalTextureDescriptor, 'yuvToRgbConversionMatrix', field(72, Pointer(ctypes.c_float)))
setattr(struct_WGPUExternalTextureDescriptor, 'srcTransferFunctionParameters', field(80, Pointer(ctypes.c_float)))
setattr(struct_WGPUExternalTextureDescriptor, 'dstTransferFunctionParameters', field(88, Pointer(ctypes.c_float)))
setattr(struct_WGPUExternalTextureDescriptor, 'gamutConversionMatrix', field(96, Pointer(ctypes.c_float)))
setattr(struct_WGPUExternalTextureDescriptor, 'mirrored', field(104, WGPUBool))
setattr(struct_WGPUExternalTextureDescriptor, 'rotation', field(108, WGPUExternalTextureRotation))
class struct_WGPUFutureWaitInfo(Struct): pass
WGPUFuture = struct_WGPUFuture
struct_WGPUFutureWaitInfo.SIZE = 16
struct_WGPUFutureWaitInfo._fields_ = ['future', 'completed']
setattr(struct_WGPUFutureWaitInfo, 'future', field(0, WGPUFuture))
setattr(struct_WGPUFutureWaitInfo, 'completed', field(8, WGPUBool))
class struct_WGPUImageCopyBuffer(Struct): pass
WGPUTextureDataLayout = struct_WGPUTextureDataLayout
struct_WGPUImageCopyBuffer.SIZE = 32
struct_WGPUImageCopyBuffer._fields_ = ['layout', 'buffer']
setattr(struct_WGPUImageCopyBuffer, 'layout', field(0, WGPUTextureDataLayout))
setattr(struct_WGPUImageCopyBuffer, 'buffer', field(24, WGPUBuffer))
class struct_WGPUImageCopyExternalTexture(Struct): pass
WGPUOrigin3D = struct_WGPUOrigin3D
struct_WGPUImageCopyExternalTexture.SIZE = 40
struct_WGPUImageCopyExternalTexture._fields_ = ['nextInChain', 'externalTexture', 'origin', 'naturalSize']
setattr(struct_WGPUImageCopyExternalTexture, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUImageCopyExternalTexture, 'externalTexture', field(8, WGPUExternalTexture))
setattr(struct_WGPUImageCopyExternalTexture, 'origin', field(16, WGPUOrigin3D))
setattr(struct_WGPUImageCopyExternalTexture, 'naturalSize', field(28, WGPUExtent2D))
class struct_WGPUImageCopyTexture(Struct): pass
enum_WGPUTextureAspect = CEnum(ctypes.c_uint32)
WGPUTextureAspect_Undefined = enum_WGPUTextureAspect.define('WGPUTextureAspect_Undefined', 0)
WGPUTextureAspect_All = enum_WGPUTextureAspect.define('WGPUTextureAspect_All', 1)
WGPUTextureAspect_StencilOnly = enum_WGPUTextureAspect.define('WGPUTextureAspect_StencilOnly', 2)
WGPUTextureAspect_DepthOnly = enum_WGPUTextureAspect.define('WGPUTextureAspect_DepthOnly', 3)
WGPUTextureAspect_Plane0Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane0Only', 327680)
WGPUTextureAspect_Plane1Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane1Only', 327681)
WGPUTextureAspect_Plane2Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane2Only', 327682)
WGPUTextureAspect_Force32 = enum_WGPUTextureAspect.define('WGPUTextureAspect_Force32', 2147483647)

WGPUTextureAspect = enum_WGPUTextureAspect
struct_WGPUImageCopyTexture.SIZE = 32
struct_WGPUImageCopyTexture._fields_ = ['texture', 'mipLevel', 'origin', 'aspect']
setattr(struct_WGPUImageCopyTexture, 'texture', field(0, WGPUTexture))
setattr(struct_WGPUImageCopyTexture, 'mipLevel', field(8, uint32_t))
setattr(struct_WGPUImageCopyTexture, 'origin', field(12, WGPUOrigin3D))
setattr(struct_WGPUImageCopyTexture, 'aspect', field(24, WGPUTextureAspect))
class struct_WGPUInstanceDescriptor(Struct): pass
WGPUInstanceFeatures = struct_WGPUInstanceFeatures
struct_WGPUInstanceDescriptor.SIZE = 32
struct_WGPUInstanceDescriptor._fields_ = ['nextInChain', 'features']
setattr(struct_WGPUInstanceDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUInstanceDescriptor, 'features', field(8, WGPUInstanceFeatures))
class struct_WGPUPipelineLayoutDescriptor(Struct): pass
struct_WGPUPipelineLayoutDescriptor.SIZE = 48
struct_WGPUPipelineLayoutDescriptor._fields_ = ['nextInChain', 'label', 'bindGroupLayoutCount', 'bindGroupLayouts', 'immediateDataRangeByteSize']
setattr(struct_WGPUPipelineLayoutDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUPipelineLayoutDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUPipelineLayoutDescriptor, 'bindGroupLayoutCount', field(24, size_t))
setattr(struct_WGPUPipelineLayoutDescriptor, 'bindGroupLayouts', field(32, Pointer(WGPUBindGroupLayout)))
setattr(struct_WGPUPipelineLayoutDescriptor, 'immediateDataRangeByteSize', field(40, uint32_t))
class struct_WGPUPipelineLayoutPixelLocalStorage(Struct): pass
WGPUPipelineLayoutStorageAttachment = struct_WGPUPipelineLayoutStorageAttachment
struct_WGPUPipelineLayoutPixelLocalStorage.SIZE = 40
struct_WGPUPipelineLayoutPixelLocalStorage._fields_ = ['chain', 'totalPixelLocalStorageSize', 'storageAttachmentCount', 'storageAttachments']
setattr(struct_WGPUPipelineLayoutPixelLocalStorage, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUPipelineLayoutPixelLocalStorage, 'totalPixelLocalStorageSize', field(16, uint64_t))
setattr(struct_WGPUPipelineLayoutPixelLocalStorage, 'storageAttachmentCount', field(24, size_t))
setattr(struct_WGPUPipelineLayoutPixelLocalStorage, 'storageAttachments', field(32, Pointer(WGPUPipelineLayoutStorageAttachment)))
class struct_WGPUQuerySetDescriptor(Struct): pass
enum_WGPUQueryType = CEnum(ctypes.c_uint32)
WGPUQueryType_Occlusion = enum_WGPUQueryType.define('WGPUQueryType_Occlusion', 1)
WGPUQueryType_Timestamp = enum_WGPUQueryType.define('WGPUQueryType_Timestamp', 2)
WGPUQueryType_Force32 = enum_WGPUQueryType.define('WGPUQueryType_Force32', 2147483647)

WGPUQueryType = enum_WGPUQueryType
struct_WGPUQuerySetDescriptor.SIZE = 32
struct_WGPUQuerySetDescriptor._fields_ = ['nextInChain', 'label', 'type', 'count']
setattr(struct_WGPUQuerySetDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUQuerySetDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUQuerySetDescriptor, 'type', field(24, WGPUQueryType))
setattr(struct_WGPUQuerySetDescriptor, 'count', field(28, uint32_t))
class struct_WGPUQueueDescriptor(Struct): pass
struct_WGPUQueueDescriptor.SIZE = 24
struct_WGPUQueueDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPUQueueDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUQueueDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPURenderBundleDescriptor(Struct): pass
struct_WGPURenderBundleDescriptor.SIZE = 24
struct_WGPURenderBundleDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPURenderBundleDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURenderBundleDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPURenderBundleEncoderDescriptor(Struct): pass
struct_WGPURenderBundleEncoderDescriptor.SIZE = 56
struct_WGPURenderBundleEncoderDescriptor._fields_ = ['nextInChain', 'label', 'colorFormatCount', 'colorFormats', 'depthStencilFormat', 'sampleCount', 'depthReadOnly', 'stencilReadOnly']
setattr(struct_WGPURenderBundleEncoderDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURenderBundleEncoderDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPURenderBundleEncoderDescriptor, 'colorFormatCount', field(24, size_t))
setattr(struct_WGPURenderBundleEncoderDescriptor, 'colorFormats', field(32, Pointer(WGPUTextureFormat)))
setattr(struct_WGPURenderBundleEncoderDescriptor, 'depthStencilFormat', field(40, WGPUTextureFormat))
setattr(struct_WGPURenderBundleEncoderDescriptor, 'sampleCount', field(44, uint32_t))
setattr(struct_WGPURenderBundleEncoderDescriptor, 'depthReadOnly', field(48, WGPUBool))
setattr(struct_WGPURenderBundleEncoderDescriptor, 'stencilReadOnly', field(52, WGPUBool))
class struct_WGPURenderPassColorAttachment(Struct): pass
WGPUColor = struct_WGPUColor
struct_WGPURenderPassColorAttachment.SIZE = 72
struct_WGPURenderPassColorAttachment._fields_ = ['nextInChain', 'view', 'depthSlice', 'resolveTarget', 'loadOp', 'storeOp', 'clearValue']
setattr(struct_WGPURenderPassColorAttachment, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURenderPassColorAttachment, 'view', field(8, WGPUTextureView))
setattr(struct_WGPURenderPassColorAttachment, 'depthSlice', field(16, uint32_t))
setattr(struct_WGPURenderPassColorAttachment, 'resolveTarget', field(24, WGPUTextureView))
setattr(struct_WGPURenderPassColorAttachment, 'loadOp', field(32, WGPULoadOp))
setattr(struct_WGPURenderPassColorAttachment, 'storeOp', field(36, WGPUStoreOp))
setattr(struct_WGPURenderPassColorAttachment, 'clearValue', field(40, WGPUColor))
class struct_WGPURenderPassStorageAttachment(Struct): pass
struct_WGPURenderPassStorageAttachment.SIZE = 64
struct_WGPURenderPassStorageAttachment._fields_ = ['nextInChain', 'offset', 'storage', 'loadOp', 'storeOp', 'clearValue']
setattr(struct_WGPURenderPassStorageAttachment, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURenderPassStorageAttachment, 'offset', field(8, uint64_t))
setattr(struct_WGPURenderPassStorageAttachment, 'storage', field(16, WGPUTextureView))
setattr(struct_WGPURenderPassStorageAttachment, 'loadOp', field(24, WGPULoadOp))
setattr(struct_WGPURenderPassStorageAttachment, 'storeOp', field(28, WGPUStoreOp))
setattr(struct_WGPURenderPassStorageAttachment, 'clearValue', field(32, WGPUColor))
class struct_WGPURequiredLimits(Struct): pass
WGPULimits = struct_WGPULimits
struct_WGPURequiredLimits.SIZE = 168
struct_WGPURequiredLimits._fields_ = ['nextInChain', 'limits']
setattr(struct_WGPURequiredLimits, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURequiredLimits, 'limits', field(8, WGPULimits))
class struct_WGPUSamplerDescriptor(Struct): pass
enum_WGPUAddressMode = CEnum(ctypes.c_uint32)
WGPUAddressMode_Undefined = enum_WGPUAddressMode.define('WGPUAddressMode_Undefined', 0)
WGPUAddressMode_ClampToEdge = enum_WGPUAddressMode.define('WGPUAddressMode_ClampToEdge', 1)
WGPUAddressMode_Repeat = enum_WGPUAddressMode.define('WGPUAddressMode_Repeat', 2)
WGPUAddressMode_MirrorRepeat = enum_WGPUAddressMode.define('WGPUAddressMode_MirrorRepeat', 3)
WGPUAddressMode_Force32 = enum_WGPUAddressMode.define('WGPUAddressMode_Force32', 2147483647)

WGPUAddressMode = enum_WGPUAddressMode
enum_WGPUMipmapFilterMode = CEnum(ctypes.c_uint32)
WGPUMipmapFilterMode_Undefined = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Undefined', 0)
WGPUMipmapFilterMode_Nearest = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Nearest', 1)
WGPUMipmapFilterMode_Linear = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Linear', 2)
WGPUMipmapFilterMode_Force32 = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Force32', 2147483647)

WGPUMipmapFilterMode = enum_WGPUMipmapFilterMode
uint16_t = ctypes.c_uint16
struct_WGPUSamplerDescriptor.SIZE = 64
struct_WGPUSamplerDescriptor._fields_ = ['nextInChain', 'label', 'addressModeU', 'addressModeV', 'addressModeW', 'magFilter', 'minFilter', 'mipmapFilter', 'lodMinClamp', 'lodMaxClamp', 'compare', 'maxAnisotropy']
setattr(struct_WGPUSamplerDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSamplerDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUSamplerDescriptor, 'addressModeU', field(24, WGPUAddressMode))
setattr(struct_WGPUSamplerDescriptor, 'addressModeV', field(28, WGPUAddressMode))
setattr(struct_WGPUSamplerDescriptor, 'addressModeW', field(32, WGPUAddressMode))
setattr(struct_WGPUSamplerDescriptor, 'magFilter', field(36, WGPUFilterMode))
setattr(struct_WGPUSamplerDescriptor, 'minFilter', field(40, WGPUFilterMode))
setattr(struct_WGPUSamplerDescriptor, 'mipmapFilter', field(44, WGPUMipmapFilterMode))
setattr(struct_WGPUSamplerDescriptor, 'lodMinClamp', field(48, ctypes.c_float))
setattr(struct_WGPUSamplerDescriptor, 'lodMaxClamp', field(52, ctypes.c_float))
setattr(struct_WGPUSamplerDescriptor, 'compare', field(56, WGPUCompareFunction))
setattr(struct_WGPUSamplerDescriptor, 'maxAnisotropy', field(60, uint16_t))
class struct_WGPUShaderModuleDescriptor(Struct): pass
struct_WGPUShaderModuleDescriptor.SIZE = 24
struct_WGPUShaderModuleDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPUShaderModuleDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUShaderModuleDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPUShaderSourceWGSL(Struct): pass
struct_WGPUShaderSourceWGSL.SIZE = 32
struct_WGPUShaderSourceWGSL._fields_ = ['chain', 'code']
setattr(struct_WGPUShaderSourceWGSL, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUShaderSourceWGSL, 'code', field(16, WGPUStringView))
class struct_WGPUSharedBufferMemoryDescriptor(Struct): pass
struct_WGPUSharedBufferMemoryDescriptor.SIZE = 24
struct_WGPUSharedBufferMemoryDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPUSharedBufferMemoryDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSharedBufferMemoryDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPUSharedFenceDescriptor(Struct): pass
struct_WGPUSharedFenceDescriptor.SIZE = 24
struct_WGPUSharedFenceDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPUSharedFenceDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSharedFenceDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPUSharedTextureMemoryAHardwareBufferProperties(Struct): pass
struct_WGPUSharedTextureMemoryAHardwareBufferProperties.SIZE = 88
struct_WGPUSharedTextureMemoryAHardwareBufferProperties._fields_ = ['chain', 'yCbCrInfo']
setattr(struct_WGPUSharedTextureMemoryAHardwareBufferProperties, 'chain', field(0, WGPUChainedStructOut))
setattr(struct_WGPUSharedTextureMemoryAHardwareBufferProperties, 'yCbCrInfo', field(16, WGPUYCbCrVkDescriptor))
class struct_WGPUSharedTextureMemoryDescriptor(Struct): pass
struct_WGPUSharedTextureMemoryDescriptor.SIZE = 24
struct_WGPUSharedTextureMemoryDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPUSharedTextureMemoryDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSharedTextureMemoryDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPUSharedTextureMemoryDmaBufDescriptor(Struct): pass
WGPUExtent3D = struct_WGPUExtent3D
WGPUSharedTextureMemoryDmaBufPlane = struct_WGPUSharedTextureMemoryDmaBufPlane
struct_WGPUSharedTextureMemoryDmaBufDescriptor.SIZE = 56
struct_WGPUSharedTextureMemoryDmaBufDescriptor._fields_ = ['chain', 'size', 'drmFormat', 'drmModifier', 'planeCount', 'planes']
setattr(struct_WGPUSharedTextureMemoryDmaBufDescriptor, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSharedTextureMemoryDmaBufDescriptor, 'size', field(16, WGPUExtent3D))
setattr(struct_WGPUSharedTextureMemoryDmaBufDescriptor, 'drmFormat', field(28, uint32_t))
setattr(struct_WGPUSharedTextureMemoryDmaBufDescriptor, 'drmModifier', field(32, uint64_t))
setattr(struct_WGPUSharedTextureMemoryDmaBufDescriptor, 'planeCount', field(40, size_t))
setattr(struct_WGPUSharedTextureMemoryDmaBufDescriptor, 'planes', field(48, Pointer(WGPUSharedTextureMemoryDmaBufPlane)))
class struct_WGPUSharedTextureMemoryProperties(Struct): pass
struct_WGPUSharedTextureMemoryProperties.SIZE = 32
struct_WGPUSharedTextureMemoryProperties._fields_ = ['nextInChain', 'usage', 'size', 'format']
setattr(struct_WGPUSharedTextureMemoryProperties, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
setattr(struct_WGPUSharedTextureMemoryProperties, 'usage', field(8, WGPUTextureUsage))
setattr(struct_WGPUSharedTextureMemoryProperties, 'size', field(16, WGPUExtent3D))
setattr(struct_WGPUSharedTextureMemoryProperties, 'format', field(28, WGPUTextureFormat))
class struct_WGPUSupportedLimits(Struct): pass
struct_WGPUSupportedLimits.SIZE = 168
struct_WGPUSupportedLimits._fields_ = ['nextInChain', 'limits']
setattr(struct_WGPUSupportedLimits, 'nextInChain', field(0, Pointer(WGPUChainedStructOut)))
setattr(struct_WGPUSupportedLimits, 'limits', field(8, WGPULimits))
class struct_WGPUSurfaceDescriptor(Struct): pass
struct_WGPUSurfaceDescriptor.SIZE = 24
struct_WGPUSurfaceDescriptor._fields_ = ['nextInChain', 'label']
setattr(struct_WGPUSurfaceDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUSurfaceDescriptor, 'label', field(8, WGPUStringView))
class struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten(Struct): pass
struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten.SIZE = 32
struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten._fields_ = ['chain', 'selector']
setattr(struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten, 'selector', field(16, WGPUStringView))
class struct_WGPUTextureDescriptor(Struct): pass
enum_WGPUTextureDimension = CEnum(ctypes.c_uint32)
WGPUTextureDimension_Undefined = enum_WGPUTextureDimension.define('WGPUTextureDimension_Undefined', 0)
WGPUTextureDimension_1D = enum_WGPUTextureDimension.define('WGPUTextureDimension_1D', 1)
WGPUTextureDimension_2D = enum_WGPUTextureDimension.define('WGPUTextureDimension_2D', 2)
WGPUTextureDimension_3D = enum_WGPUTextureDimension.define('WGPUTextureDimension_3D', 3)
WGPUTextureDimension_Force32 = enum_WGPUTextureDimension.define('WGPUTextureDimension_Force32', 2147483647)

WGPUTextureDimension = enum_WGPUTextureDimension
struct_WGPUTextureDescriptor.SIZE = 80
struct_WGPUTextureDescriptor._fields_ = ['nextInChain', 'label', 'usage', 'dimension', 'size', 'format', 'mipLevelCount', 'sampleCount', 'viewFormatCount', 'viewFormats']
setattr(struct_WGPUTextureDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUTextureDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUTextureDescriptor, 'usage', field(24, WGPUTextureUsage))
setattr(struct_WGPUTextureDescriptor, 'dimension', field(32, WGPUTextureDimension))
setattr(struct_WGPUTextureDescriptor, 'size', field(36, WGPUExtent3D))
setattr(struct_WGPUTextureDescriptor, 'format', field(48, WGPUTextureFormat))
setattr(struct_WGPUTextureDescriptor, 'mipLevelCount', field(52, uint32_t))
setattr(struct_WGPUTextureDescriptor, 'sampleCount', field(56, uint32_t))
setattr(struct_WGPUTextureDescriptor, 'viewFormatCount', field(64, size_t))
setattr(struct_WGPUTextureDescriptor, 'viewFormats', field(72, Pointer(WGPUTextureFormat)))
class struct_WGPUTextureViewDescriptor(Struct): pass
struct_WGPUTextureViewDescriptor.SIZE = 64
struct_WGPUTextureViewDescriptor._fields_ = ['nextInChain', 'label', 'format', 'dimension', 'baseMipLevel', 'mipLevelCount', 'baseArrayLayer', 'arrayLayerCount', 'aspect', 'usage']
setattr(struct_WGPUTextureViewDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUTextureViewDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUTextureViewDescriptor, 'format', field(24, WGPUTextureFormat))
setattr(struct_WGPUTextureViewDescriptor, 'dimension', field(28, WGPUTextureViewDimension))
setattr(struct_WGPUTextureViewDescriptor, 'baseMipLevel', field(32, uint32_t))
setattr(struct_WGPUTextureViewDescriptor, 'mipLevelCount', field(36, uint32_t))
setattr(struct_WGPUTextureViewDescriptor, 'baseArrayLayer', field(40, uint32_t))
setattr(struct_WGPUTextureViewDescriptor, 'arrayLayerCount', field(44, uint32_t))
setattr(struct_WGPUTextureViewDescriptor, 'aspect', field(48, WGPUTextureAspect))
setattr(struct_WGPUTextureViewDescriptor, 'usage', field(56, WGPUTextureUsage))
class struct_WGPUVertexBufferLayout(Struct): pass
enum_WGPUVertexStepMode = CEnum(ctypes.c_uint32)
WGPUVertexStepMode_Undefined = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Undefined', 0)
WGPUVertexStepMode_Vertex = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Vertex', 1)
WGPUVertexStepMode_Instance = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Instance', 2)
WGPUVertexStepMode_Force32 = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Force32', 2147483647)

WGPUVertexStepMode = enum_WGPUVertexStepMode
WGPUVertexAttribute = struct_WGPUVertexAttribute
struct_WGPUVertexBufferLayout.SIZE = 32
struct_WGPUVertexBufferLayout._fields_ = ['arrayStride', 'stepMode', 'attributeCount', 'attributes']
setattr(struct_WGPUVertexBufferLayout, 'arrayStride', field(0, uint64_t))
setattr(struct_WGPUVertexBufferLayout, 'stepMode', field(8, WGPUVertexStepMode))
setattr(struct_WGPUVertexBufferLayout, 'attributeCount', field(16, size_t))
setattr(struct_WGPUVertexBufferLayout, 'attributes', field(24, Pointer(WGPUVertexAttribute)))
class struct_WGPUBindGroupLayoutDescriptor(Struct): pass
WGPUBindGroupLayoutEntry = struct_WGPUBindGroupLayoutEntry
struct_WGPUBindGroupLayoutDescriptor.SIZE = 40
struct_WGPUBindGroupLayoutDescriptor._fields_ = ['nextInChain', 'label', 'entryCount', 'entries']
setattr(struct_WGPUBindGroupLayoutDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUBindGroupLayoutDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUBindGroupLayoutDescriptor, 'entryCount', field(24, size_t))
setattr(struct_WGPUBindGroupLayoutDescriptor, 'entries', field(32, Pointer(WGPUBindGroupLayoutEntry)))
class struct_WGPUColorTargetState(Struct): pass
WGPUBlendState = struct_WGPUBlendState
WGPUColorWriteMask = ctypes.c_uint64
struct_WGPUColorTargetState.SIZE = 32
struct_WGPUColorTargetState._fields_ = ['nextInChain', 'format', 'blend', 'writeMask']
setattr(struct_WGPUColorTargetState, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUColorTargetState, 'format', field(8, WGPUTextureFormat))
setattr(struct_WGPUColorTargetState, 'blend', field(16, Pointer(WGPUBlendState)))
setattr(struct_WGPUColorTargetState, 'writeMask', field(24, WGPUColorWriteMask))
class struct_WGPUComputeState(Struct): pass
WGPUConstantEntry = struct_WGPUConstantEntry
struct_WGPUComputeState.SIZE = 48
struct_WGPUComputeState._fields_ = ['nextInChain', 'module', 'entryPoint', 'constantCount', 'constants']
setattr(struct_WGPUComputeState, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUComputeState, 'module', field(8, WGPUShaderModule))
setattr(struct_WGPUComputeState, 'entryPoint', field(16, WGPUStringView))
setattr(struct_WGPUComputeState, 'constantCount', field(32, size_t))
setattr(struct_WGPUComputeState, 'constants', field(40, Pointer(WGPUConstantEntry)))
class struct_WGPUDeviceDescriptor(Struct): pass
WGPURequiredLimits = struct_WGPURequiredLimits
WGPUQueueDescriptor = struct_WGPUQueueDescriptor
class struct_WGPUDeviceLostCallbackInfo2(Struct): pass
WGPUDeviceLostCallbackInfo2 = struct_WGPUDeviceLostCallbackInfo2
WGPUDeviceLostCallback2 = ctypes.CFUNCTYPE(None, Pointer(Pointer(struct_WGPUDeviceImpl)), enum_WGPUDeviceLostReason, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
struct_WGPUDeviceLostCallbackInfo2.SIZE = 40
struct_WGPUDeviceLostCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPUDeviceLostCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUDeviceLostCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUDeviceLostCallbackInfo2, 'callback', field(16, WGPUDeviceLostCallback2))
setattr(struct_WGPUDeviceLostCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPUDeviceLostCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
class struct_WGPUUncapturedErrorCallbackInfo2(Struct): pass
WGPUUncapturedErrorCallbackInfo2 = struct_WGPUUncapturedErrorCallbackInfo2
WGPUUncapturedErrorCallback = ctypes.CFUNCTYPE(None, Pointer(Pointer(struct_WGPUDeviceImpl)), enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
struct_WGPUUncapturedErrorCallbackInfo2.SIZE = 32
struct_WGPUUncapturedErrorCallbackInfo2._fields_ = ['nextInChain', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPUUncapturedErrorCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUUncapturedErrorCallbackInfo2, 'callback', field(8, WGPUUncapturedErrorCallback))
setattr(struct_WGPUUncapturedErrorCallbackInfo2, 'userdata1', field(16, ctypes.c_void_p))
setattr(struct_WGPUUncapturedErrorCallbackInfo2, 'userdata2', field(24, ctypes.c_void_p))
struct_WGPUDeviceDescriptor.SIZE = 144
struct_WGPUDeviceDescriptor._fields_ = ['nextInChain', 'label', 'requiredFeatureCount', 'requiredFeatures', 'requiredLimits', 'defaultQueue', 'deviceLostCallbackInfo2', 'uncapturedErrorCallbackInfo2']
setattr(struct_WGPUDeviceDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUDeviceDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUDeviceDescriptor, 'requiredFeatureCount', field(24, size_t))
setattr(struct_WGPUDeviceDescriptor, 'requiredFeatures', field(32, Pointer(WGPUFeatureName)))
setattr(struct_WGPUDeviceDescriptor, 'requiredLimits', field(40, Pointer(WGPURequiredLimits)))
setattr(struct_WGPUDeviceDescriptor, 'defaultQueue', field(48, WGPUQueueDescriptor))
setattr(struct_WGPUDeviceDescriptor, 'deviceLostCallbackInfo2', field(72, WGPUDeviceLostCallbackInfo2))
setattr(struct_WGPUDeviceDescriptor, 'uncapturedErrorCallbackInfo2', field(112, WGPUUncapturedErrorCallbackInfo2))
class struct_WGPURenderPassDescriptor(Struct): pass
WGPURenderPassColorAttachment = struct_WGPURenderPassColorAttachment
WGPURenderPassDepthStencilAttachment = struct_WGPURenderPassDepthStencilAttachment
WGPURenderPassTimestampWrites = struct_WGPURenderPassTimestampWrites
struct_WGPURenderPassDescriptor.SIZE = 64
struct_WGPURenderPassDescriptor._fields_ = ['nextInChain', 'label', 'colorAttachmentCount', 'colorAttachments', 'depthStencilAttachment', 'occlusionQuerySet', 'timestampWrites']
setattr(struct_WGPURenderPassDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURenderPassDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPURenderPassDescriptor, 'colorAttachmentCount', field(24, size_t))
setattr(struct_WGPURenderPassDescriptor, 'colorAttachments', field(32, Pointer(WGPURenderPassColorAttachment)))
setattr(struct_WGPURenderPassDescriptor, 'depthStencilAttachment', field(40, Pointer(WGPURenderPassDepthStencilAttachment)))
setattr(struct_WGPURenderPassDescriptor, 'occlusionQuerySet', field(48, WGPUQuerySet))
setattr(struct_WGPURenderPassDescriptor, 'timestampWrites', field(56, Pointer(WGPURenderPassTimestampWrites)))
class struct_WGPURenderPassPixelLocalStorage(Struct): pass
WGPURenderPassStorageAttachment = struct_WGPURenderPassStorageAttachment
struct_WGPURenderPassPixelLocalStorage.SIZE = 40
struct_WGPURenderPassPixelLocalStorage._fields_ = ['chain', 'totalPixelLocalStorageSize', 'storageAttachmentCount', 'storageAttachments']
setattr(struct_WGPURenderPassPixelLocalStorage, 'chain', field(0, WGPUChainedStruct))
setattr(struct_WGPURenderPassPixelLocalStorage, 'totalPixelLocalStorageSize', field(16, uint64_t))
setattr(struct_WGPURenderPassPixelLocalStorage, 'storageAttachmentCount', field(24, size_t))
setattr(struct_WGPURenderPassPixelLocalStorage, 'storageAttachments', field(32, Pointer(WGPURenderPassStorageAttachment)))
class struct_WGPUVertexState(Struct): pass
WGPUVertexBufferLayout = struct_WGPUVertexBufferLayout
struct_WGPUVertexState.SIZE = 64
struct_WGPUVertexState._fields_ = ['nextInChain', 'module', 'entryPoint', 'constantCount', 'constants', 'bufferCount', 'buffers']
setattr(struct_WGPUVertexState, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUVertexState, 'module', field(8, WGPUShaderModule))
setattr(struct_WGPUVertexState, 'entryPoint', field(16, WGPUStringView))
setattr(struct_WGPUVertexState, 'constantCount', field(32, size_t))
setattr(struct_WGPUVertexState, 'constants', field(40, Pointer(WGPUConstantEntry)))
setattr(struct_WGPUVertexState, 'bufferCount', field(48, size_t))
setattr(struct_WGPUVertexState, 'buffers', field(56, Pointer(WGPUVertexBufferLayout)))
class struct_WGPUComputePipelineDescriptor(Struct): pass
WGPUComputeState = struct_WGPUComputeState
struct_WGPUComputePipelineDescriptor.SIZE = 80
struct_WGPUComputePipelineDescriptor._fields_ = ['nextInChain', 'label', 'layout', 'compute']
setattr(struct_WGPUComputePipelineDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUComputePipelineDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPUComputePipelineDescriptor, 'layout', field(24, WGPUPipelineLayout))
setattr(struct_WGPUComputePipelineDescriptor, 'compute', field(32, WGPUComputeState))
class struct_WGPUFragmentState(Struct): pass
WGPUColorTargetState = struct_WGPUColorTargetState
struct_WGPUFragmentState.SIZE = 64
struct_WGPUFragmentState._fields_ = ['nextInChain', 'module', 'entryPoint', 'constantCount', 'constants', 'targetCount', 'targets']
setattr(struct_WGPUFragmentState, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUFragmentState, 'module', field(8, WGPUShaderModule))
setattr(struct_WGPUFragmentState, 'entryPoint', field(16, WGPUStringView))
setattr(struct_WGPUFragmentState, 'constantCount', field(32, size_t))
setattr(struct_WGPUFragmentState, 'constants', field(40, Pointer(WGPUConstantEntry)))
setattr(struct_WGPUFragmentState, 'targetCount', field(48, size_t))
setattr(struct_WGPUFragmentState, 'targets', field(56, Pointer(WGPUColorTargetState)))
class struct_WGPURenderPipelineDescriptor(Struct): pass
WGPUVertexState = struct_WGPUVertexState
WGPUPrimitiveState = struct_WGPUPrimitiveState
WGPUDepthStencilState = struct_WGPUDepthStencilState
WGPUMultisampleState = struct_WGPUMultisampleState
WGPUFragmentState = struct_WGPUFragmentState
struct_WGPURenderPipelineDescriptor.SIZE = 168
struct_WGPURenderPipelineDescriptor._fields_ = ['nextInChain', 'label', 'layout', 'vertex', 'primitive', 'depthStencil', 'multisample', 'fragment']
setattr(struct_WGPURenderPipelineDescriptor, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURenderPipelineDescriptor, 'label', field(8, WGPUStringView))
setattr(struct_WGPURenderPipelineDescriptor, 'layout', field(24, WGPUPipelineLayout))
setattr(struct_WGPURenderPipelineDescriptor, 'vertex', field(32, WGPUVertexState))
setattr(struct_WGPURenderPipelineDescriptor, 'primitive', field(96, WGPUPrimitiveState))
setattr(struct_WGPURenderPipelineDescriptor, 'depthStencil', field(128, Pointer(WGPUDepthStencilState)))
setattr(struct_WGPURenderPipelineDescriptor, 'multisample', field(136, WGPUMultisampleState))
setattr(struct_WGPURenderPipelineDescriptor, 'fragment', field(160, Pointer(WGPUFragmentState)))
enum_WGPUWGSLFeatureName = CEnum(ctypes.c_uint32)
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
enum_WGPUBufferMapState = CEnum(ctypes.c_uint32)
WGPUBufferMapState_Unmapped = enum_WGPUBufferMapState.define('WGPUBufferMapState_Unmapped', 1)
WGPUBufferMapState_Pending = enum_WGPUBufferMapState.define('WGPUBufferMapState_Pending', 2)
WGPUBufferMapState_Mapped = enum_WGPUBufferMapState.define('WGPUBufferMapState_Mapped', 3)
WGPUBufferMapState_Force32 = enum_WGPUBufferMapState.define('WGPUBufferMapState_Force32', 2147483647)

WGPUBufferMapState = enum_WGPUBufferMapState
WGPUCompilationInfoRequestStatus = enum_WGPUCompilationInfoRequestStatus
WGPUCreatePipelineAsyncStatus = enum_WGPUCreatePipelineAsyncStatus
WGPUDeviceLostReason = enum_WGPUDeviceLostReason
enum_WGPUErrorFilter = CEnum(ctypes.c_uint32)
WGPUErrorFilter_Validation = enum_WGPUErrorFilter.define('WGPUErrorFilter_Validation', 1)
WGPUErrorFilter_OutOfMemory = enum_WGPUErrorFilter.define('WGPUErrorFilter_OutOfMemory', 2)
WGPUErrorFilter_Internal = enum_WGPUErrorFilter.define('WGPUErrorFilter_Internal', 3)
WGPUErrorFilter_Force32 = enum_WGPUErrorFilter.define('WGPUErrorFilter_Force32', 2147483647)

WGPUErrorFilter = enum_WGPUErrorFilter
WGPUErrorType = enum_WGPUErrorType
enum_WGPULoggingType = CEnum(ctypes.c_uint32)
WGPULoggingType_Verbose = enum_WGPULoggingType.define('WGPULoggingType_Verbose', 1)
WGPULoggingType_Info = enum_WGPULoggingType.define('WGPULoggingType_Info', 2)
WGPULoggingType_Warning = enum_WGPULoggingType.define('WGPULoggingType_Warning', 3)
WGPULoggingType_Error = enum_WGPULoggingType.define('WGPULoggingType_Error', 4)
WGPULoggingType_Force32 = enum_WGPULoggingType.define('WGPULoggingType_Force32', 2147483647)

WGPULoggingType = enum_WGPULoggingType
enum_WGPUMapAsyncStatus = CEnum(ctypes.c_uint32)
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
enum_WGPUStatus = CEnum(ctypes.c_uint32)
WGPUStatus_Success = enum_WGPUStatus.define('WGPUStatus_Success', 1)
WGPUStatus_Error = enum_WGPUStatus.define('WGPUStatus_Error', 2)
WGPUStatus_Force32 = enum_WGPUStatus.define('WGPUStatus_Force32', 2147483647)

WGPUStatus = enum_WGPUStatus
enum_WGPUWaitStatus = CEnum(ctypes.c_uint32)
WGPUWaitStatus_Success = enum_WGPUWaitStatus.define('WGPUWaitStatus_Success', 1)
WGPUWaitStatus_TimedOut = enum_WGPUWaitStatus.define('WGPUWaitStatus_TimedOut', 2)
WGPUWaitStatus_UnsupportedTimeout = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedTimeout', 3)
WGPUWaitStatus_UnsupportedCount = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedCount', 4)
WGPUWaitStatus_UnsupportedMixedSources = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedMixedSources', 5)
WGPUWaitStatus_Unknown = enum_WGPUWaitStatus.define('WGPUWaitStatus_Unknown', 6)
WGPUWaitStatus_Force32 = enum_WGPUWaitStatus.define('WGPUWaitStatus_Force32', 2147483647)

WGPUWaitStatus = enum_WGPUWaitStatus
WGPUMapMode = ctypes.c_uint64
WGPUDeviceLostCallback = ctypes.CFUNCTYPE(None, enum_WGPUDeviceLostReason, struct_WGPUStringView, ctypes.c_void_p)
WGPULoggingCallback = ctypes.CFUNCTYPE(None, enum_WGPULoggingType, struct_WGPUStringView, ctypes.c_void_p)
WGPUProc = ctypes.CFUNCTYPE(None, )
WGPUBufferMapCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUMapAsyncStatus, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPUCompilationInfoCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, Pointer(struct_WGPUCompilationInfo), ctypes.c_void_p, ctypes.c_void_p)
WGPUCreateComputePipelineAsyncCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, Pointer(struct_WGPUComputePipelineImpl), struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPUCreateRenderPipelineAsyncCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, Pointer(struct_WGPURenderPipelineImpl), struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPUPopErrorScopeCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUPopErrorScopeStatus, enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPUQueueWorkDoneCallback2 = ctypes.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, ctypes.c_void_p, ctypes.c_void_p)
WGPURequestAdapterCallback2 = ctypes.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, Pointer(struct_WGPUAdapterImpl), struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
WGPURequestDeviceCallback2 = ctypes.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, Pointer(struct_WGPUDeviceImpl), struct_WGPUStringView, ctypes.c_void_p, ctypes.c_void_p)
class struct_WGPUBufferMapCallbackInfo2(Struct): pass
struct_WGPUBufferMapCallbackInfo2.SIZE = 40
struct_WGPUBufferMapCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPUBufferMapCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUBufferMapCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUBufferMapCallbackInfo2, 'callback', field(16, WGPUBufferMapCallback2))
setattr(struct_WGPUBufferMapCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPUBufferMapCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
WGPUBufferMapCallbackInfo2 = struct_WGPUBufferMapCallbackInfo2
class struct_WGPUCompilationInfoCallbackInfo2(Struct): pass
struct_WGPUCompilationInfoCallbackInfo2.SIZE = 40
struct_WGPUCompilationInfoCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPUCompilationInfoCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCompilationInfoCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUCompilationInfoCallbackInfo2, 'callback', field(16, WGPUCompilationInfoCallback2))
setattr(struct_WGPUCompilationInfoCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPUCompilationInfoCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
WGPUCompilationInfoCallbackInfo2 = struct_WGPUCompilationInfoCallbackInfo2
class struct_WGPUCreateComputePipelineAsyncCallbackInfo2(Struct): pass
struct_WGPUCreateComputePipelineAsyncCallbackInfo2.SIZE = 40
struct_WGPUCreateComputePipelineAsyncCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo2, 'callback', field(16, WGPUCreateComputePipelineAsyncCallback2))
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPUCreateComputePipelineAsyncCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
WGPUCreateComputePipelineAsyncCallbackInfo2 = struct_WGPUCreateComputePipelineAsyncCallbackInfo2
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo2(Struct): pass
struct_WGPUCreateRenderPipelineAsyncCallbackInfo2.SIZE = 40
struct_WGPUCreateRenderPipelineAsyncCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo2, 'callback', field(16, WGPUCreateRenderPipelineAsyncCallback2))
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPUCreateRenderPipelineAsyncCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
WGPUCreateRenderPipelineAsyncCallbackInfo2 = struct_WGPUCreateRenderPipelineAsyncCallbackInfo2
class struct_WGPUPopErrorScopeCallbackInfo2(Struct): pass
struct_WGPUPopErrorScopeCallbackInfo2.SIZE = 40
struct_WGPUPopErrorScopeCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPUPopErrorScopeCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUPopErrorScopeCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUPopErrorScopeCallbackInfo2, 'callback', field(16, WGPUPopErrorScopeCallback2))
setattr(struct_WGPUPopErrorScopeCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPUPopErrorScopeCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
WGPUPopErrorScopeCallbackInfo2 = struct_WGPUPopErrorScopeCallbackInfo2
class struct_WGPUQueueWorkDoneCallbackInfo2(Struct): pass
struct_WGPUQueueWorkDoneCallbackInfo2.SIZE = 40
struct_WGPUQueueWorkDoneCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPUQueueWorkDoneCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPUQueueWorkDoneCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPUQueueWorkDoneCallbackInfo2, 'callback', field(16, WGPUQueueWorkDoneCallback2))
setattr(struct_WGPUQueueWorkDoneCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPUQueueWorkDoneCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
WGPUQueueWorkDoneCallbackInfo2 = struct_WGPUQueueWorkDoneCallbackInfo2
class struct_WGPURequestAdapterCallbackInfo2(Struct): pass
struct_WGPURequestAdapterCallbackInfo2.SIZE = 40
struct_WGPURequestAdapterCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPURequestAdapterCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURequestAdapterCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPURequestAdapterCallbackInfo2, 'callback', field(16, WGPURequestAdapterCallback2))
setattr(struct_WGPURequestAdapterCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPURequestAdapterCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
WGPURequestAdapterCallbackInfo2 = struct_WGPURequestAdapterCallbackInfo2
class struct_WGPURequestDeviceCallbackInfo2(Struct): pass
struct_WGPURequestDeviceCallbackInfo2.SIZE = 40
struct_WGPURequestDeviceCallbackInfo2._fields_ = ['nextInChain', 'mode', 'callback', 'userdata1', 'userdata2']
setattr(struct_WGPURequestDeviceCallbackInfo2, 'nextInChain', field(0, Pointer(WGPUChainedStruct)))
setattr(struct_WGPURequestDeviceCallbackInfo2, 'mode', field(8, WGPUCallbackMode))
setattr(struct_WGPURequestDeviceCallbackInfo2, 'callback', field(16, WGPURequestDeviceCallback2))
setattr(struct_WGPURequestDeviceCallbackInfo2, 'userdata1', field(24, ctypes.c_void_p))
setattr(struct_WGPURequestDeviceCallbackInfo2, 'userdata2', field(32, ctypes.c_void_p))
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
WGPUProcCreateInstance = ctypes.CFUNCTYPE(Pointer(struct_WGPUInstanceImpl), Pointer(struct_WGPUInstanceDescriptor))
WGPUProcDrmFormatCapabilitiesFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUDrmFormatCapabilities)
WGPUProcGetInstanceFeatures = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUInstanceFeatures))
WGPUProcGetProcAddress = ctypes.CFUNCTYPE(ctypes.CFUNCTYPE(None, ), struct_WGPUStringView)
WGPUProcSharedBufferMemoryEndAccessStateFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUSharedBufferMemoryEndAccessState)
WGPUProcSharedTextureMemoryEndAccessStateFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUSharedTextureMemoryEndAccessState)
WGPUProcSupportedFeaturesFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUSupportedFeatures)
WGPUProcSurfaceCapabilitiesFreeMembers = ctypes.CFUNCTYPE(None, struct_WGPUSurfaceCapabilities)
WGPUProcAdapterCreateDevice = ctypes.CFUNCTYPE(Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUDeviceDescriptor))
WGPUProcAdapterGetFeatures = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUSupportedFeatures))
WGPUProcAdapterGetFormatCapabilities = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUAdapterImpl), enum_WGPUTextureFormat, Pointer(struct_WGPUFormatCapabilities))
WGPUProcAdapterGetInfo = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUAdapterInfo))
WGPUProcAdapterGetInstance = ctypes.CFUNCTYPE(Pointer(struct_WGPUInstanceImpl), Pointer(struct_WGPUAdapterImpl))
WGPUProcAdapterGetLimits = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUSupportedLimits))
WGPUProcAdapterHasFeature = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUAdapterImpl), enum_WGPUFeatureName)
WGPUProcAdapterRequestDevice = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUDeviceDescriptor), ctypes.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, Pointer(struct_WGPUDeviceImpl), struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcAdapterRequestDevice2 = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUDeviceDescriptor), struct_WGPURequestDeviceCallbackInfo2)
WGPUProcAdapterRequestDeviceF = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUDeviceDescriptor), struct_WGPURequestDeviceCallbackInfo)
WGPUProcAdapterAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUAdapterImpl))
WGPUProcAdapterRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUAdapterImpl))
WGPUProcBindGroupSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBindGroupImpl), struct_WGPUStringView)
WGPUProcBindGroupAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBindGroupImpl))
WGPUProcBindGroupRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBindGroupImpl))
WGPUProcBindGroupLayoutSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBindGroupLayoutImpl), struct_WGPUStringView)
WGPUProcBindGroupLayoutAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBindGroupLayoutImpl))
WGPUProcBindGroupLayoutRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBindGroupLayoutImpl))
WGPUProcBufferDestroy = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBufferImpl))
WGPUProcBufferGetConstMappedRange = ctypes.CFUNCTYPE(ctypes.c_void_p, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64)
WGPUProcBufferGetMapState = ctypes.CFUNCTYPE(enum_WGPUBufferMapState, Pointer(struct_WGPUBufferImpl))
WGPUProcBufferGetMappedRange = ctypes.CFUNCTYPE(ctypes.c_void_p, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64)
WGPUProcBufferGetSize = ctypes.CFUNCTYPE(ctypes.c_uint64, Pointer(struct_WGPUBufferImpl))
WGPUProcBufferGetUsage = ctypes.CFUNCTYPE(ctypes.c_uint64, Pointer(struct_WGPUBufferImpl))
WGPUProcBufferMapAsync = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.CFUNCTYPE(None, enum_WGPUBufferMapAsyncStatus, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcBufferMapAsync2 = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, struct_WGPUBufferMapCallbackInfo2)
WGPUProcBufferMapAsyncF = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, struct_WGPUBufferMapCallbackInfo)
WGPUProcBufferSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBufferImpl), struct_WGPUStringView)
WGPUProcBufferUnmap = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBufferImpl))
WGPUProcBufferAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBufferImpl))
WGPUProcBufferRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUBufferImpl))
WGPUProcCommandBufferSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandBufferImpl), struct_WGPUStringView)
WGPUProcCommandBufferAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandBufferImpl))
WGPUProcCommandBufferRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandBufferImpl))
WGPUProcCommandEncoderBeginComputePass = ctypes.CFUNCTYPE(Pointer(struct_WGPUComputePassEncoderImpl), Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUComputePassDescriptor))
WGPUProcCommandEncoderBeginRenderPass = ctypes.CFUNCTYPE(Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPURenderPassDescriptor))
WGPUProcCommandEncoderClearBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64)
WGPUProcCommandEncoderCopyBufferToBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64)
WGPUProcCommandEncoderCopyBufferToTexture = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUImageCopyBuffer), Pointer(struct_WGPUImageCopyTexture), Pointer(struct_WGPUExtent3D))
WGPUProcCommandEncoderCopyTextureToBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUImageCopyTexture), Pointer(struct_WGPUImageCopyBuffer), Pointer(struct_WGPUExtent3D))
WGPUProcCommandEncoderCopyTextureToTexture = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUImageCopyTexture), Pointer(struct_WGPUImageCopyTexture), Pointer(struct_WGPUExtent3D))
WGPUProcCommandEncoderFinish = ctypes.CFUNCTYPE(Pointer(struct_WGPUCommandBufferImpl), Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUCommandBufferDescriptor))
WGPUProcCommandEncoderInjectValidationError = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderInsertDebugMarker = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderPopDebugGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl))
WGPUProcCommandEncoderPushDebugGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderResolveQuerySet = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUQuerySetImpl), ctypes.c_uint32, ctypes.c_uint32, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64)
WGPUProcCommandEncoderSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderWriteBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, Pointer(ctypes.c_ubyte), ctypes.c_uint64)
WGPUProcCommandEncoderWriteTimestamp = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUQuerySetImpl), ctypes.c_uint32)
WGPUProcCommandEncoderAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl))
WGPUProcCommandEncoderRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUCommandEncoderImpl))
WGPUProcComputePassEncoderDispatchWorkgroups = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
WGPUProcComputePassEncoderDispatchWorkgroupsIndirect = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64)
WGPUProcComputePassEncoderEnd = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderInsertDebugMarker = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderPopDebugGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderPushDebugGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderSetBindGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl), ctypes.c_uint32, Pointer(struct_WGPUBindGroupImpl), ctypes.c_uint64, Pointer(ctypes.c_uint32))
WGPUProcComputePassEncoderSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderSetPipeline = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl), Pointer(struct_WGPUComputePipelineImpl))
WGPUProcComputePassEncoderWriteTimestamp = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl), Pointer(struct_WGPUQuerySetImpl), ctypes.c_uint32)
WGPUProcComputePassEncoderAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePipelineGetBindGroupLayout = ctypes.CFUNCTYPE(Pointer(struct_WGPUBindGroupLayoutImpl), Pointer(struct_WGPUComputePipelineImpl), ctypes.c_uint32)
WGPUProcComputePipelineSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePipelineImpl), struct_WGPUStringView)
WGPUProcComputePipelineAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePipelineImpl))
WGPUProcComputePipelineRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUComputePipelineImpl))
WGPUProcDeviceCreateBindGroup = ctypes.CFUNCTYPE(Pointer(struct_WGPUBindGroupImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUBindGroupDescriptor))
WGPUProcDeviceCreateBindGroupLayout = ctypes.CFUNCTYPE(Pointer(struct_WGPUBindGroupLayoutImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUBindGroupLayoutDescriptor))
WGPUProcDeviceCreateBuffer = ctypes.CFUNCTYPE(Pointer(struct_WGPUBufferImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUBufferDescriptor))
WGPUProcDeviceCreateCommandEncoder = ctypes.CFUNCTYPE(Pointer(struct_WGPUCommandEncoderImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUCommandEncoderDescriptor))
WGPUProcDeviceCreateComputePipeline = ctypes.CFUNCTYPE(Pointer(struct_WGPUComputePipelineImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUComputePipelineDescriptor))
WGPUProcDeviceCreateComputePipelineAsync = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUComputePipelineDescriptor), ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, Pointer(struct_WGPUComputePipelineImpl), struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcDeviceCreateComputePipelineAsync2 = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUComputePipelineDescriptor), struct_WGPUCreateComputePipelineAsyncCallbackInfo2)
WGPUProcDeviceCreateComputePipelineAsyncF = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUComputePipelineDescriptor), struct_WGPUCreateComputePipelineAsyncCallbackInfo)
WGPUProcDeviceCreateErrorBuffer = ctypes.CFUNCTYPE(Pointer(struct_WGPUBufferImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUBufferDescriptor))
WGPUProcDeviceCreateErrorExternalTexture = ctypes.CFUNCTYPE(Pointer(struct_WGPUExternalTextureImpl), Pointer(struct_WGPUDeviceImpl))
WGPUProcDeviceCreateErrorShaderModule = ctypes.CFUNCTYPE(Pointer(struct_WGPUShaderModuleImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUShaderModuleDescriptor), struct_WGPUStringView)
WGPUProcDeviceCreateErrorTexture = ctypes.CFUNCTYPE(Pointer(struct_WGPUTextureImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUTextureDescriptor))
WGPUProcDeviceCreateExternalTexture = ctypes.CFUNCTYPE(Pointer(struct_WGPUExternalTextureImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUExternalTextureDescriptor))
WGPUProcDeviceCreatePipelineLayout = ctypes.CFUNCTYPE(Pointer(struct_WGPUPipelineLayoutImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUPipelineLayoutDescriptor))
WGPUProcDeviceCreateQuerySet = ctypes.CFUNCTYPE(Pointer(struct_WGPUQuerySetImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUQuerySetDescriptor))
WGPUProcDeviceCreateRenderBundleEncoder = ctypes.CFUNCTYPE(Pointer(struct_WGPURenderBundleEncoderImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPURenderBundleEncoderDescriptor))
WGPUProcDeviceCreateRenderPipeline = ctypes.CFUNCTYPE(Pointer(struct_WGPURenderPipelineImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPURenderPipelineDescriptor))
WGPUProcDeviceCreateRenderPipelineAsync = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPURenderPipelineDescriptor), ctypes.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, Pointer(struct_WGPURenderPipelineImpl), struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcDeviceCreateRenderPipelineAsync2 = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPURenderPipelineDescriptor), struct_WGPUCreateRenderPipelineAsyncCallbackInfo2)
WGPUProcDeviceCreateRenderPipelineAsyncF = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPURenderPipelineDescriptor), struct_WGPUCreateRenderPipelineAsyncCallbackInfo)
WGPUProcDeviceCreateSampler = ctypes.CFUNCTYPE(Pointer(struct_WGPUSamplerImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUSamplerDescriptor))
WGPUProcDeviceCreateShaderModule = ctypes.CFUNCTYPE(Pointer(struct_WGPUShaderModuleImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUShaderModuleDescriptor))
WGPUProcDeviceCreateTexture = ctypes.CFUNCTYPE(Pointer(struct_WGPUTextureImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUTextureDescriptor))
WGPUProcDeviceDestroy = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl))
WGPUProcDeviceForceLoss = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), enum_WGPUDeviceLostReason, struct_WGPUStringView)
WGPUProcDeviceGetAHardwareBufferProperties = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUDeviceImpl), ctypes.c_void_p, Pointer(struct_WGPUAHardwareBufferProperties))
WGPUProcDeviceGetAdapter = ctypes.CFUNCTYPE(Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUDeviceImpl))
WGPUProcDeviceGetAdapterInfo = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUAdapterInfo))
WGPUProcDeviceGetFeatures = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUSupportedFeatures))
WGPUProcDeviceGetLimits = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUSupportedLimits))
WGPUProcDeviceGetLostFuture = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUDeviceImpl))
WGPUProcDeviceGetQueue = ctypes.CFUNCTYPE(Pointer(struct_WGPUQueueImpl), Pointer(struct_WGPUDeviceImpl))
WGPUProcDeviceHasFeature = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUDeviceImpl), enum_WGPUFeatureName)
WGPUProcDeviceImportSharedBufferMemory = ctypes.CFUNCTYPE(Pointer(struct_WGPUSharedBufferMemoryImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUSharedBufferMemoryDescriptor))
WGPUProcDeviceImportSharedFence = ctypes.CFUNCTYPE(Pointer(struct_WGPUSharedFenceImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUSharedFenceDescriptor))
WGPUProcDeviceImportSharedTextureMemory = ctypes.CFUNCTYPE(Pointer(struct_WGPUSharedTextureMemoryImpl), Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUSharedTextureMemoryDescriptor))
WGPUProcDeviceInjectError = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), enum_WGPUErrorType, struct_WGPUStringView)
WGPUProcDevicePopErrorScope = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), ctypes.CFUNCTYPE(None, enum_WGPUErrorType, struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcDevicePopErrorScope2 = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUDeviceImpl), struct_WGPUPopErrorScopeCallbackInfo2)
WGPUProcDevicePopErrorScopeF = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUDeviceImpl), struct_WGPUPopErrorScopeCallbackInfo)
WGPUProcDevicePushErrorScope = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), enum_WGPUErrorFilter)
WGPUProcDeviceSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), struct_WGPUStringView)
WGPUProcDeviceSetLoggingCallback = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), ctypes.CFUNCTYPE(None, enum_WGPULoggingType, struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcDeviceTick = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl))
WGPUProcDeviceValidateTextureDescriptor = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl), Pointer(struct_WGPUTextureDescriptor))
WGPUProcDeviceAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl))
WGPUProcDeviceRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUDeviceImpl))
WGPUProcExternalTextureDestroy = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureExpire = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureRefresh = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUExternalTextureImpl), struct_WGPUStringView)
WGPUProcExternalTextureAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUExternalTextureImpl))
WGPUProcInstanceCreateSurface = ctypes.CFUNCTYPE(Pointer(struct_WGPUSurfaceImpl), Pointer(struct_WGPUInstanceImpl), Pointer(struct_WGPUSurfaceDescriptor))
WGPUProcInstanceEnumerateWGSLLanguageFeatures = ctypes.CFUNCTYPE(ctypes.c_uint64, Pointer(struct_WGPUInstanceImpl), Pointer(enum_WGPUWGSLFeatureName))
WGPUProcInstanceHasWGSLLanguageFeature = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUInstanceImpl), enum_WGPUWGSLFeatureName)
WGPUProcInstanceProcessEvents = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUInstanceImpl))
WGPUProcInstanceRequestAdapter = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUInstanceImpl), Pointer(struct_WGPURequestAdapterOptions), ctypes.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, Pointer(struct_WGPUAdapterImpl), struct_WGPUStringView, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcInstanceRequestAdapter2 = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUInstanceImpl), Pointer(struct_WGPURequestAdapterOptions), struct_WGPURequestAdapterCallbackInfo2)
WGPUProcInstanceRequestAdapterF = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUInstanceImpl), Pointer(struct_WGPURequestAdapterOptions), struct_WGPURequestAdapterCallbackInfo)
WGPUProcInstanceWaitAny = ctypes.CFUNCTYPE(enum_WGPUWaitStatus, Pointer(struct_WGPUInstanceImpl), ctypes.c_uint64, Pointer(struct_WGPUFutureWaitInfo), ctypes.c_uint64)
WGPUProcInstanceAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUInstanceImpl))
WGPUProcInstanceRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUInstanceImpl))
WGPUProcPipelineLayoutSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUPipelineLayoutImpl), struct_WGPUStringView)
WGPUProcPipelineLayoutAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUPipelineLayoutImpl))
WGPUProcPipelineLayoutRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUPipelineLayoutImpl))
WGPUProcQuerySetDestroy = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQuerySetImpl))
WGPUProcQuerySetGetCount = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUQuerySetImpl))
WGPUProcQuerySetGetType = ctypes.CFUNCTYPE(enum_WGPUQueryType, Pointer(struct_WGPUQuerySetImpl))
WGPUProcQuerySetSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQuerySetImpl), struct_WGPUStringView)
WGPUProcQuerySetAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQuerySetImpl))
WGPUProcQuerySetRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQuerySetImpl))
WGPUProcQueueCopyExternalTextureForBrowser = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl), Pointer(struct_WGPUImageCopyExternalTexture), Pointer(struct_WGPUImageCopyTexture), Pointer(struct_WGPUExtent3D), Pointer(struct_WGPUCopyTextureForBrowserOptions))
WGPUProcQueueCopyTextureForBrowser = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl), Pointer(struct_WGPUImageCopyTexture), Pointer(struct_WGPUImageCopyTexture), Pointer(struct_WGPUExtent3D), Pointer(struct_WGPUCopyTextureForBrowserOptions))
WGPUProcQueueOnSubmittedWorkDone = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl), ctypes.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, ctypes.c_void_p), ctypes.c_void_p)
WGPUProcQueueOnSubmittedWorkDone2 = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUQueueImpl), struct_WGPUQueueWorkDoneCallbackInfo2)
WGPUProcQueueOnSubmittedWorkDoneF = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUQueueImpl), struct_WGPUQueueWorkDoneCallbackInfo)
WGPUProcQueueSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl), struct_WGPUStringView)
WGPUProcQueueSubmit = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl), ctypes.c_uint64, Pointer(Pointer(struct_WGPUCommandBufferImpl)))
WGPUProcQueueWriteBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)
WGPUProcQueueWriteTexture = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl), Pointer(struct_WGPUImageCopyTexture), ctypes.c_void_p, ctypes.c_uint64, Pointer(struct_WGPUTextureDataLayout), Pointer(struct_WGPUExtent3D))
WGPUProcQueueAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl))
WGPUProcQueueRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUQueueImpl))
WGPUProcRenderBundleSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleImpl), struct_WGPUStringView)
WGPUProcRenderBundleAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleImpl))
WGPUProcRenderBundleRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleImpl))
WGPUProcRenderBundleEncoderDraw = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
WGPUProcRenderBundleEncoderDrawIndexed = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32)
WGPUProcRenderBundleEncoderDrawIndexedIndirect = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64)
WGPUProcRenderBundleEncoderDrawIndirect = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64)
WGPUProcRenderBundleEncoderFinish = ctypes.CFUNCTYPE(Pointer(struct_WGPURenderBundleImpl), Pointer(struct_WGPURenderBundleEncoderImpl), Pointer(struct_WGPURenderBundleDescriptor))
WGPUProcRenderBundleEncoderInsertDebugMarker = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderPopDebugGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderBundleEncoderPushDebugGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderSetBindGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), ctypes.c_uint32, Pointer(struct_WGPUBindGroupImpl), ctypes.c_uint64, Pointer(ctypes.c_uint32))
WGPUProcRenderBundleEncoderSetIndexBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), Pointer(struct_WGPUBufferImpl), enum_WGPUIndexFormat, ctypes.c_uint64, ctypes.c_uint64)
WGPUProcRenderBundleEncoderSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderSetPipeline = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), Pointer(struct_WGPURenderPipelineImpl))
WGPUProcRenderBundleEncoderSetVertexBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl), ctypes.c_uint32, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64)
WGPUProcRenderBundleEncoderAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderBundleEncoderRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderPassEncoderBeginOcclusionQuery = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_uint32)
WGPUProcRenderPassEncoderDraw = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
WGPUProcRenderPassEncoderDrawIndexed = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32)
WGPUProcRenderPassEncoderDrawIndexedIndirect = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64)
WGPUProcRenderPassEncoderDrawIndirect = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64)
WGPUProcRenderPassEncoderEnd = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderEndOcclusionQuery = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderExecuteBundles = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_uint64, Pointer(Pointer(struct_WGPURenderBundleImpl)))
WGPUProcRenderPassEncoderInsertDebugMarker = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderMultiDrawIndexedIndirect = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint32, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64)
WGPUProcRenderPassEncoderMultiDrawIndirect = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint32, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64)
WGPUProcRenderPassEncoderPixelLocalStorageBarrier = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderPopDebugGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderPushDebugGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderSetBindGroup = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_uint32, Pointer(struct_WGPUBindGroupImpl), ctypes.c_uint64, Pointer(ctypes.c_uint32))
WGPUProcRenderPassEncoderSetBlendConstant = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPUColor))
WGPUProcRenderPassEncoderSetIndexBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPUBufferImpl), enum_WGPUIndexFormat, ctypes.c_uint64, ctypes.c_uint64)
WGPUProcRenderPassEncoderSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderSetPipeline = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPURenderPipelineImpl))
WGPUProcRenderPassEncoderSetScissorRect = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
WGPUProcRenderPassEncoderSetStencilReference = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_uint32)
WGPUProcRenderPassEncoderSetVertexBuffer = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_uint32, Pointer(struct_WGPUBufferImpl), ctypes.c_uint64, ctypes.c_uint64)
WGPUProcRenderPassEncoderSetViewport = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
WGPUProcRenderPassEncoderWriteTimestamp = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl), Pointer(struct_WGPUQuerySetImpl), ctypes.c_uint32)
WGPUProcRenderPassEncoderAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPipelineGetBindGroupLayout = ctypes.CFUNCTYPE(Pointer(struct_WGPUBindGroupLayoutImpl), Pointer(struct_WGPURenderPipelineImpl), ctypes.c_uint32)
WGPUProcRenderPipelineSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPipelineImpl), struct_WGPUStringView)
WGPUProcRenderPipelineAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPipelineImpl))
WGPUProcRenderPipelineRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPURenderPipelineImpl))
WGPUProcSamplerSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSamplerImpl), struct_WGPUStringView)
WGPUProcSamplerAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSamplerImpl))
WGPUProcSamplerRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSamplerImpl))
WGPUProcShaderModuleGetCompilationInfo = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUShaderModuleImpl), ctypes.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, Pointer(struct_WGPUCompilationInfo), ctypes.c_void_p), ctypes.c_void_p)
WGPUProcShaderModuleGetCompilationInfo2 = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUShaderModuleImpl), struct_WGPUCompilationInfoCallbackInfo2)
WGPUProcShaderModuleGetCompilationInfoF = ctypes.CFUNCTYPE(struct_WGPUFuture, Pointer(struct_WGPUShaderModuleImpl), struct_WGPUCompilationInfoCallbackInfo)
WGPUProcShaderModuleSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUShaderModuleImpl), struct_WGPUStringView)
WGPUProcShaderModuleAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUShaderModuleImpl))
WGPUProcShaderModuleRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUShaderModuleImpl))
WGPUProcSharedBufferMemoryBeginAccess = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUSharedBufferMemoryImpl), Pointer(struct_WGPUBufferImpl), Pointer(struct_WGPUSharedBufferMemoryBeginAccessDescriptor))
WGPUProcSharedBufferMemoryCreateBuffer = ctypes.CFUNCTYPE(Pointer(struct_WGPUBufferImpl), Pointer(struct_WGPUSharedBufferMemoryImpl), Pointer(struct_WGPUBufferDescriptor))
WGPUProcSharedBufferMemoryEndAccess = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUSharedBufferMemoryImpl), Pointer(struct_WGPUBufferImpl), Pointer(struct_WGPUSharedBufferMemoryEndAccessState))
WGPUProcSharedBufferMemoryGetProperties = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUSharedBufferMemoryImpl), Pointer(struct_WGPUSharedBufferMemoryProperties))
WGPUProcSharedBufferMemoryIsDeviceLost = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedBufferMemorySetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedBufferMemoryImpl), struct_WGPUStringView)
WGPUProcSharedBufferMemoryAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedBufferMemoryRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedFenceExportInfo = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedFenceImpl), Pointer(struct_WGPUSharedFenceExportInfo))
WGPUProcSharedFenceAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedFenceImpl))
WGPUProcSharedFenceRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedFenceImpl))
WGPUProcSharedTextureMemoryBeginAccess = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUSharedTextureMemoryImpl), Pointer(struct_WGPUTextureImpl), Pointer(struct_WGPUSharedTextureMemoryBeginAccessDescriptor))
WGPUProcSharedTextureMemoryCreateTexture = ctypes.CFUNCTYPE(Pointer(struct_WGPUTextureImpl), Pointer(struct_WGPUSharedTextureMemoryImpl), Pointer(struct_WGPUTextureDescriptor))
WGPUProcSharedTextureMemoryEndAccess = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUSharedTextureMemoryImpl), Pointer(struct_WGPUTextureImpl), Pointer(struct_WGPUSharedTextureMemoryEndAccessState))
WGPUProcSharedTextureMemoryGetProperties = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUSharedTextureMemoryImpl), Pointer(struct_WGPUSharedTextureMemoryProperties))
WGPUProcSharedTextureMemoryIsDeviceLost = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUSharedTextureMemoryImpl))
WGPUProcSharedTextureMemorySetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedTextureMemoryImpl), struct_WGPUStringView)
WGPUProcSharedTextureMemoryAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedTextureMemoryImpl))
WGPUProcSharedTextureMemoryRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSharedTextureMemoryImpl))
WGPUProcSurfaceConfigure = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSurfaceImpl), Pointer(struct_WGPUSurfaceConfiguration))
WGPUProcSurfaceGetCapabilities = ctypes.CFUNCTYPE(enum_WGPUStatus, Pointer(struct_WGPUSurfaceImpl), Pointer(struct_WGPUAdapterImpl), Pointer(struct_WGPUSurfaceCapabilities))
WGPUProcSurfaceGetCurrentTexture = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSurfaceImpl), Pointer(struct_WGPUSurfaceTexture))
WGPUProcSurfacePresent = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSurfaceImpl))
WGPUProcSurfaceSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSurfaceImpl), struct_WGPUStringView)
WGPUProcSurfaceUnconfigure = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSurfaceImpl))
WGPUProcSurfaceAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSurfaceImpl))
WGPUProcSurfaceRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUSurfaceImpl))
WGPUProcTextureCreateErrorView = ctypes.CFUNCTYPE(Pointer(struct_WGPUTextureViewImpl), Pointer(struct_WGPUTextureImpl), Pointer(struct_WGPUTextureViewDescriptor))
WGPUProcTextureCreateView = ctypes.CFUNCTYPE(Pointer(struct_WGPUTextureViewImpl), Pointer(struct_WGPUTextureImpl), Pointer(struct_WGPUTextureViewDescriptor))
WGPUProcTextureDestroy = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureGetDepthOrArrayLayers = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureGetDimension = ctypes.CFUNCTYPE(enum_WGPUTextureDimension, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureGetFormat = ctypes.CFUNCTYPE(enum_WGPUTextureFormat, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureGetHeight = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureGetMipLevelCount = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureGetSampleCount = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureGetUsage = ctypes.CFUNCTYPE(ctypes.c_uint64, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureGetWidth = ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUTextureImpl), struct_WGPUStringView)
WGPUProcTextureAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUTextureImpl))
WGPUProcTextureViewSetLabel = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUTextureViewImpl), struct_WGPUStringView)
WGPUProcTextureViewAddRef = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUTextureViewImpl))
WGPUProcTextureViewRelease = ctypes.CFUNCTYPE(None, Pointer(struct_WGPUTextureViewImpl))
try: (wgpuAdapterInfoFreeMembers:=dll.wgpuAdapterInfoFreeMembers).restype, wgpuAdapterInfoFreeMembers.argtypes = None, [WGPUAdapterInfo]
except AttributeError: pass

try: (wgpuAdapterPropertiesMemoryHeapsFreeMembers:=dll.wgpuAdapterPropertiesMemoryHeapsFreeMembers).restype, wgpuAdapterPropertiesMemoryHeapsFreeMembers.argtypes = None, [WGPUAdapterPropertiesMemoryHeaps]
except AttributeError: pass

try: (wgpuCreateInstance:=dll.wgpuCreateInstance).restype, wgpuCreateInstance.argtypes = WGPUInstance, [Pointer(WGPUInstanceDescriptor)]
except AttributeError: pass

try: (wgpuDrmFormatCapabilitiesFreeMembers:=dll.wgpuDrmFormatCapabilitiesFreeMembers).restype, wgpuDrmFormatCapabilitiesFreeMembers.argtypes = None, [WGPUDrmFormatCapabilities]
except AttributeError: pass

try: (wgpuGetInstanceFeatures:=dll.wgpuGetInstanceFeatures).restype, wgpuGetInstanceFeatures.argtypes = WGPUStatus, [Pointer(WGPUInstanceFeatures)]
except AttributeError: pass

try: (wgpuGetProcAddress:=dll.wgpuGetProcAddress).restype, wgpuGetProcAddress.argtypes = WGPUProc, [WGPUStringView]
except AttributeError: pass

try: (wgpuSharedBufferMemoryEndAccessStateFreeMembers:=dll.wgpuSharedBufferMemoryEndAccessStateFreeMembers).restype, wgpuSharedBufferMemoryEndAccessStateFreeMembers.argtypes = None, [WGPUSharedBufferMemoryEndAccessState]
except AttributeError: pass

try: (wgpuSharedTextureMemoryEndAccessStateFreeMembers:=dll.wgpuSharedTextureMemoryEndAccessStateFreeMembers).restype, wgpuSharedTextureMemoryEndAccessStateFreeMembers.argtypes = None, [WGPUSharedTextureMemoryEndAccessState]
except AttributeError: pass

try: (wgpuSupportedFeaturesFreeMembers:=dll.wgpuSupportedFeaturesFreeMembers).restype, wgpuSupportedFeaturesFreeMembers.argtypes = None, [WGPUSupportedFeatures]
except AttributeError: pass

try: (wgpuSurfaceCapabilitiesFreeMembers:=dll.wgpuSurfaceCapabilitiesFreeMembers).restype, wgpuSurfaceCapabilitiesFreeMembers.argtypes = None, [WGPUSurfaceCapabilities]
except AttributeError: pass

try: (wgpuAdapterCreateDevice:=dll.wgpuAdapterCreateDevice).restype, wgpuAdapterCreateDevice.argtypes = WGPUDevice, [WGPUAdapter, Pointer(WGPUDeviceDescriptor)]
except AttributeError: pass

try: (wgpuAdapterGetFeatures:=dll.wgpuAdapterGetFeatures).restype, wgpuAdapterGetFeatures.argtypes = None, [WGPUAdapter, Pointer(WGPUSupportedFeatures)]
except AttributeError: pass

try: (wgpuAdapterGetFormatCapabilities:=dll.wgpuAdapterGetFormatCapabilities).restype, wgpuAdapterGetFormatCapabilities.argtypes = WGPUStatus, [WGPUAdapter, WGPUTextureFormat, Pointer(WGPUFormatCapabilities)]
except AttributeError: pass

try: (wgpuAdapterGetInfo:=dll.wgpuAdapterGetInfo).restype, wgpuAdapterGetInfo.argtypes = WGPUStatus, [WGPUAdapter, Pointer(WGPUAdapterInfo)]
except AttributeError: pass

try: (wgpuAdapterGetInstance:=dll.wgpuAdapterGetInstance).restype, wgpuAdapterGetInstance.argtypes = WGPUInstance, [WGPUAdapter]
except AttributeError: pass

try: (wgpuAdapterGetLimits:=dll.wgpuAdapterGetLimits).restype, wgpuAdapterGetLimits.argtypes = WGPUStatus, [WGPUAdapter, Pointer(WGPUSupportedLimits)]
except AttributeError: pass

try: (wgpuAdapterHasFeature:=dll.wgpuAdapterHasFeature).restype, wgpuAdapterHasFeature.argtypes = WGPUBool, [WGPUAdapter, WGPUFeatureName]
except AttributeError: pass

try: (wgpuAdapterRequestDevice:=dll.wgpuAdapterRequestDevice).restype, wgpuAdapterRequestDevice.argtypes = None, [WGPUAdapter, Pointer(WGPUDeviceDescriptor), WGPURequestDeviceCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuAdapterRequestDevice2:=dll.wgpuAdapterRequestDevice2).restype, wgpuAdapterRequestDevice2.argtypes = WGPUFuture, [WGPUAdapter, Pointer(WGPUDeviceDescriptor), WGPURequestDeviceCallbackInfo2]
except AttributeError: pass

try: (wgpuAdapterRequestDeviceF:=dll.wgpuAdapterRequestDeviceF).restype, wgpuAdapterRequestDeviceF.argtypes = WGPUFuture, [WGPUAdapter, Pointer(WGPUDeviceDescriptor), WGPURequestDeviceCallbackInfo]
except AttributeError: pass

try: (wgpuAdapterAddRef:=dll.wgpuAdapterAddRef).restype, wgpuAdapterAddRef.argtypes = None, [WGPUAdapter]
except AttributeError: pass

try: (wgpuAdapterRelease:=dll.wgpuAdapterRelease).restype, wgpuAdapterRelease.argtypes = None, [WGPUAdapter]
except AttributeError: pass

try: (wgpuBindGroupSetLabel:=dll.wgpuBindGroupSetLabel).restype, wgpuBindGroupSetLabel.argtypes = None, [WGPUBindGroup, WGPUStringView]
except AttributeError: pass

try: (wgpuBindGroupAddRef:=dll.wgpuBindGroupAddRef).restype, wgpuBindGroupAddRef.argtypes = None, [WGPUBindGroup]
except AttributeError: pass

try: (wgpuBindGroupRelease:=dll.wgpuBindGroupRelease).restype, wgpuBindGroupRelease.argtypes = None, [WGPUBindGroup]
except AttributeError: pass

try: (wgpuBindGroupLayoutSetLabel:=dll.wgpuBindGroupLayoutSetLabel).restype, wgpuBindGroupLayoutSetLabel.argtypes = None, [WGPUBindGroupLayout, WGPUStringView]
except AttributeError: pass

try: (wgpuBindGroupLayoutAddRef:=dll.wgpuBindGroupLayoutAddRef).restype, wgpuBindGroupLayoutAddRef.argtypes = None, [WGPUBindGroupLayout]
except AttributeError: pass

try: (wgpuBindGroupLayoutRelease:=dll.wgpuBindGroupLayoutRelease).restype, wgpuBindGroupLayoutRelease.argtypes = None, [WGPUBindGroupLayout]
except AttributeError: pass

try: (wgpuBufferDestroy:=dll.wgpuBufferDestroy).restype, wgpuBufferDestroy.argtypes = None, [WGPUBuffer]
except AttributeError: pass

try: (wgpuBufferGetConstMappedRange:=dll.wgpuBufferGetConstMappedRange).restype, wgpuBufferGetConstMappedRange.argtypes = ctypes.c_void_p, [WGPUBuffer, size_t, size_t]
except AttributeError: pass

try: (wgpuBufferGetMapState:=dll.wgpuBufferGetMapState).restype, wgpuBufferGetMapState.argtypes = WGPUBufferMapState, [WGPUBuffer]
except AttributeError: pass

try: (wgpuBufferGetMappedRange:=dll.wgpuBufferGetMappedRange).restype, wgpuBufferGetMappedRange.argtypes = ctypes.c_void_p, [WGPUBuffer, size_t, size_t]
except AttributeError: pass

try: (wgpuBufferGetSize:=dll.wgpuBufferGetSize).restype, wgpuBufferGetSize.argtypes = uint64_t, [WGPUBuffer]
except AttributeError: pass

try: (wgpuBufferGetUsage:=dll.wgpuBufferGetUsage).restype, wgpuBufferGetUsage.argtypes = WGPUBufferUsage, [WGPUBuffer]
except AttributeError: pass

try: (wgpuBufferMapAsync:=dll.wgpuBufferMapAsync).restype, wgpuBufferMapAsync.argtypes = None, [WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuBufferMapAsync2:=dll.wgpuBufferMapAsync2).restype, wgpuBufferMapAsync2.argtypes = WGPUFuture, [WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallbackInfo2]
except AttributeError: pass

try: (wgpuBufferMapAsyncF:=dll.wgpuBufferMapAsyncF).restype, wgpuBufferMapAsyncF.argtypes = WGPUFuture, [WGPUBuffer, WGPUMapMode, size_t, size_t, WGPUBufferMapCallbackInfo]
except AttributeError: pass

try: (wgpuBufferSetLabel:=dll.wgpuBufferSetLabel).restype, wgpuBufferSetLabel.argtypes = None, [WGPUBuffer, WGPUStringView]
except AttributeError: pass

try: (wgpuBufferUnmap:=dll.wgpuBufferUnmap).restype, wgpuBufferUnmap.argtypes = None, [WGPUBuffer]
except AttributeError: pass

try: (wgpuBufferAddRef:=dll.wgpuBufferAddRef).restype, wgpuBufferAddRef.argtypes = None, [WGPUBuffer]
except AttributeError: pass

try: (wgpuBufferRelease:=dll.wgpuBufferRelease).restype, wgpuBufferRelease.argtypes = None, [WGPUBuffer]
except AttributeError: pass

try: (wgpuCommandBufferSetLabel:=dll.wgpuCommandBufferSetLabel).restype, wgpuCommandBufferSetLabel.argtypes = None, [WGPUCommandBuffer, WGPUStringView]
except AttributeError: pass

try: (wgpuCommandBufferAddRef:=dll.wgpuCommandBufferAddRef).restype, wgpuCommandBufferAddRef.argtypes = None, [WGPUCommandBuffer]
except AttributeError: pass

try: (wgpuCommandBufferRelease:=dll.wgpuCommandBufferRelease).restype, wgpuCommandBufferRelease.argtypes = None, [WGPUCommandBuffer]
except AttributeError: pass

try: (wgpuCommandEncoderBeginComputePass:=dll.wgpuCommandEncoderBeginComputePass).restype, wgpuCommandEncoderBeginComputePass.argtypes = WGPUComputePassEncoder, [WGPUCommandEncoder, Pointer(WGPUComputePassDescriptor)]
except AttributeError: pass

try: (wgpuCommandEncoderBeginRenderPass:=dll.wgpuCommandEncoderBeginRenderPass).restype, wgpuCommandEncoderBeginRenderPass.argtypes = WGPURenderPassEncoder, [WGPUCommandEncoder, Pointer(WGPURenderPassDescriptor)]
except AttributeError: pass

try: (wgpuCommandEncoderClearBuffer:=dll.wgpuCommandEncoderClearBuffer).restype, wgpuCommandEncoderClearBuffer.argtypes = None, [WGPUCommandEncoder, WGPUBuffer, uint64_t, uint64_t]
except AttributeError: pass

try: (wgpuCommandEncoderCopyBufferToBuffer:=dll.wgpuCommandEncoderCopyBufferToBuffer).restype, wgpuCommandEncoderCopyBufferToBuffer.argtypes = None, [WGPUCommandEncoder, WGPUBuffer, uint64_t, WGPUBuffer, uint64_t, uint64_t]
except AttributeError: pass

try: (wgpuCommandEncoderCopyBufferToTexture:=dll.wgpuCommandEncoderCopyBufferToTexture).restype, wgpuCommandEncoderCopyBufferToTexture.argtypes = None, [WGPUCommandEncoder, Pointer(WGPUImageCopyBuffer), Pointer(WGPUImageCopyTexture), Pointer(WGPUExtent3D)]
except AttributeError: pass

try: (wgpuCommandEncoderCopyTextureToBuffer:=dll.wgpuCommandEncoderCopyTextureToBuffer).restype, wgpuCommandEncoderCopyTextureToBuffer.argtypes = None, [WGPUCommandEncoder, Pointer(WGPUImageCopyTexture), Pointer(WGPUImageCopyBuffer), Pointer(WGPUExtent3D)]
except AttributeError: pass

try: (wgpuCommandEncoderCopyTextureToTexture:=dll.wgpuCommandEncoderCopyTextureToTexture).restype, wgpuCommandEncoderCopyTextureToTexture.argtypes = None, [WGPUCommandEncoder, Pointer(WGPUImageCopyTexture), Pointer(WGPUImageCopyTexture), Pointer(WGPUExtent3D)]
except AttributeError: pass

try: (wgpuCommandEncoderFinish:=dll.wgpuCommandEncoderFinish).restype, wgpuCommandEncoderFinish.argtypes = WGPUCommandBuffer, [WGPUCommandEncoder, Pointer(WGPUCommandBufferDescriptor)]
except AttributeError: pass

try: (wgpuCommandEncoderInjectValidationError:=dll.wgpuCommandEncoderInjectValidationError).restype, wgpuCommandEncoderInjectValidationError.argtypes = None, [WGPUCommandEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuCommandEncoderInsertDebugMarker:=dll.wgpuCommandEncoderInsertDebugMarker).restype, wgpuCommandEncoderInsertDebugMarker.argtypes = None, [WGPUCommandEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuCommandEncoderPopDebugGroup:=dll.wgpuCommandEncoderPopDebugGroup).restype, wgpuCommandEncoderPopDebugGroup.argtypes = None, [WGPUCommandEncoder]
except AttributeError: pass

try: (wgpuCommandEncoderPushDebugGroup:=dll.wgpuCommandEncoderPushDebugGroup).restype, wgpuCommandEncoderPushDebugGroup.argtypes = None, [WGPUCommandEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuCommandEncoderResolveQuerySet:=dll.wgpuCommandEncoderResolveQuerySet).restype, wgpuCommandEncoderResolveQuerySet.argtypes = None, [WGPUCommandEncoder, WGPUQuerySet, uint32_t, uint32_t, WGPUBuffer, uint64_t]
except AttributeError: pass

try: (wgpuCommandEncoderSetLabel:=dll.wgpuCommandEncoderSetLabel).restype, wgpuCommandEncoderSetLabel.argtypes = None, [WGPUCommandEncoder, WGPUStringView]
except AttributeError: pass

uint8_t = ctypes.c_ubyte
try: (wgpuCommandEncoderWriteBuffer:=dll.wgpuCommandEncoderWriteBuffer).restype, wgpuCommandEncoderWriteBuffer.argtypes = None, [WGPUCommandEncoder, WGPUBuffer, uint64_t, Pointer(uint8_t), uint64_t]
except AttributeError: pass

try: (wgpuCommandEncoderWriteTimestamp:=dll.wgpuCommandEncoderWriteTimestamp).restype, wgpuCommandEncoderWriteTimestamp.argtypes = None, [WGPUCommandEncoder, WGPUQuerySet, uint32_t]
except AttributeError: pass

try: (wgpuCommandEncoderAddRef:=dll.wgpuCommandEncoderAddRef).restype, wgpuCommandEncoderAddRef.argtypes = None, [WGPUCommandEncoder]
except AttributeError: pass

try: (wgpuCommandEncoderRelease:=dll.wgpuCommandEncoderRelease).restype, wgpuCommandEncoderRelease.argtypes = None, [WGPUCommandEncoder]
except AttributeError: pass

try: (wgpuComputePassEncoderDispatchWorkgroups:=dll.wgpuComputePassEncoderDispatchWorkgroups).restype, wgpuComputePassEncoderDispatchWorkgroups.argtypes = None, [WGPUComputePassEncoder, uint32_t, uint32_t, uint32_t]
except AttributeError: pass

try: (wgpuComputePassEncoderDispatchWorkgroupsIndirect:=dll.wgpuComputePassEncoderDispatchWorkgroupsIndirect).restype, wgpuComputePassEncoderDispatchWorkgroupsIndirect.argtypes = None, [WGPUComputePassEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

try: (wgpuComputePassEncoderEnd:=dll.wgpuComputePassEncoderEnd).restype, wgpuComputePassEncoderEnd.argtypes = None, [WGPUComputePassEncoder]
except AttributeError: pass

try: (wgpuComputePassEncoderInsertDebugMarker:=dll.wgpuComputePassEncoderInsertDebugMarker).restype, wgpuComputePassEncoderInsertDebugMarker.argtypes = None, [WGPUComputePassEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuComputePassEncoderPopDebugGroup:=dll.wgpuComputePassEncoderPopDebugGroup).restype, wgpuComputePassEncoderPopDebugGroup.argtypes = None, [WGPUComputePassEncoder]
except AttributeError: pass

try: (wgpuComputePassEncoderPushDebugGroup:=dll.wgpuComputePassEncoderPushDebugGroup).restype, wgpuComputePassEncoderPushDebugGroup.argtypes = None, [WGPUComputePassEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuComputePassEncoderSetBindGroup:=dll.wgpuComputePassEncoderSetBindGroup).restype, wgpuComputePassEncoderSetBindGroup.argtypes = None, [WGPUComputePassEncoder, uint32_t, WGPUBindGroup, size_t, Pointer(uint32_t)]
except AttributeError: pass

try: (wgpuComputePassEncoderSetLabel:=dll.wgpuComputePassEncoderSetLabel).restype, wgpuComputePassEncoderSetLabel.argtypes = None, [WGPUComputePassEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuComputePassEncoderSetPipeline:=dll.wgpuComputePassEncoderSetPipeline).restype, wgpuComputePassEncoderSetPipeline.argtypes = None, [WGPUComputePassEncoder, WGPUComputePipeline]
except AttributeError: pass

try: (wgpuComputePassEncoderWriteTimestamp:=dll.wgpuComputePassEncoderWriteTimestamp).restype, wgpuComputePassEncoderWriteTimestamp.argtypes = None, [WGPUComputePassEncoder, WGPUQuerySet, uint32_t]
except AttributeError: pass

try: (wgpuComputePassEncoderAddRef:=dll.wgpuComputePassEncoderAddRef).restype, wgpuComputePassEncoderAddRef.argtypes = None, [WGPUComputePassEncoder]
except AttributeError: pass

try: (wgpuComputePassEncoderRelease:=dll.wgpuComputePassEncoderRelease).restype, wgpuComputePassEncoderRelease.argtypes = None, [WGPUComputePassEncoder]
except AttributeError: pass

try: (wgpuComputePipelineGetBindGroupLayout:=dll.wgpuComputePipelineGetBindGroupLayout).restype, wgpuComputePipelineGetBindGroupLayout.argtypes = WGPUBindGroupLayout, [WGPUComputePipeline, uint32_t]
except AttributeError: pass

try: (wgpuComputePipelineSetLabel:=dll.wgpuComputePipelineSetLabel).restype, wgpuComputePipelineSetLabel.argtypes = None, [WGPUComputePipeline, WGPUStringView]
except AttributeError: pass

try: (wgpuComputePipelineAddRef:=dll.wgpuComputePipelineAddRef).restype, wgpuComputePipelineAddRef.argtypes = None, [WGPUComputePipeline]
except AttributeError: pass

try: (wgpuComputePipelineRelease:=dll.wgpuComputePipelineRelease).restype, wgpuComputePipelineRelease.argtypes = None, [WGPUComputePipeline]
except AttributeError: pass

try: (wgpuDeviceCreateBindGroup:=dll.wgpuDeviceCreateBindGroup).restype, wgpuDeviceCreateBindGroup.argtypes = WGPUBindGroup, [WGPUDevice, Pointer(WGPUBindGroupDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateBindGroupLayout:=dll.wgpuDeviceCreateBindGroupLayout).restype, wgpuDeviceCreateBindGroupLayout.argtypes = WGPUBindGroupLayout, [WGPUDevice, Pointer(WGPUBindGroupLayoutDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateBuffer:=dll.wgpuDeviceCreateBuffer).restype, wgpuDeviceCreateBuffer.argtypes = WGPUBuffer, [WGPUDevice, Pointer(WGPUBufferDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateCommandEncoder:=dll.wgpuDeviceCreateCommandEncoder).restype, wgpuDeviceCreateCommandEncoder.argtypes = WGPUCommandEncoder, [WGPUDevice, Pointer(WGPUCommandEncoderDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateComputePipeline:=dll.wgpuDeviceCreateComputePipeline).restype, wgpuDeviceCreateComputePipeline.argtypes = WGPUComputePipeline, [WGPUDevice, Pointer(WGPUComputePipelineDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateComputePipelineAsync:=dll.wgpuDeviceCreateComputePipelineAsync).restype, wgpuDeviceCreateComputePipelineAsync.argtypes = None, [WGPUDevice, Pointer(WGPUComputePipelineDescriptor), WGPUCreateComputePipelineAsyncCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuDeviceCreateComputePipelineAsync2:=dll.wgpuDeviceCreateComputePipelineAsync2).restype, wgpuDeviceCreateComputePipelineAsync2.argtypes = WGPUFuture, [WGPUDevice, Pointer(WGPUComputePipelineDescriptor), WGPUCreateComputePipelineAsyncCallbackInfo2]
except AttributeError: pass

try: (wgpuDeviceCreateComputePipelineAsyncF:=dll.wgpuDeviceCreateComputePipelineAsyncF).restype, wgpuDeviceCreateComputePipelineAsyncF.argtypes = WGPUFuture, [WGPUDevice, Pointer(WGPUComputePipelineDescriptor), WGPUCreateComputePipelineAsyncCallbackInfo]
except AttributeError: pass

try: (wgpuDeviceCreateErrorBuffer:=dll.wgpuDeviceCreateErrorBuffer).restype, wgpuDeviceCreateErrorBuffer.argtypes = WGPUBuffer, [WGPUDevice, Pointer(WGPUBufferDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateErrorExternalTexture:=dll.wgpuDeviceCreateErrorExternalTexture).restype, wgpuDeviceCreateErrorExternalTexture.argtypes = WGPUExternalTexture, [WGPUDevice]
except AttributeError: pass

try: (wgpuDeviceCreateErrorShaderModule:=dll.wgpuDeviceCreateErrorShaderModule).restype, wgpuDeviceCreateErrorShaderModule.argtypes = WGPUShaderModule, [WGPUDevice, Pointer(WGPUShaderModuleDescriptor), WGPUStringView]
except AttributeError: pass

try: (wgpuDeviceCreateErrorTexture:=dll.wgpuDeviceCreateErrorTexture).restype, wgpuDeviceCreateErrorTexture.argtypes = WGPUTexture, [WGPUDevice, Pointer(WGPUTextureDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateExternalTexture:=dll.wgpuDeviceCreateExternalTexture).restype, wgpuDeviceCreateExternalTexture.argtypes = WGPUExternalTexture, [WGPUDevice, Pointer(WGPUExternalTextureDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreatePipelineLayout:=dll.wgpuDeviceCreatePipelineLayout).restype, wgpuDeviceCreatePipelineLayout.argtypes = WGPUPipelineLayout, [WGPUDevice, Pointer(WGPUPipelineLayoutDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateQuerySet:=dll.wgpuDeviceCreateQuerySet).restype, wgpuDeviceCreateQuerySet.argtypes = WGPUQuerySet, [WGPUDevice, Pointer(WGPUQuerySetDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateRenderBundleEncoder:=dll.wgpuDeviceCreateRenderBundleEncoder).restype, wgpuDeviceCreateRenderBundleEncoder.argtypes = WGPURenderBundleEncoder, [WGPUDevice, Pointer(WGPURenderBundleEncoderDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateRenderPipeline:=dll.wgpuDeviceCreateRenderPipeline).restype, wgpuDeviceCreateRenderPipeline.argtypes = WGPURenderPipeline, [WGPUDevice, Pointer(WGPURenderPipelineDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateRenderPipelineAsync:=dll.wgpuDeviceCreateRenderPipelineAsync).restype, wgpuDeviceCreateRenderPipelineAsync.argtypes = None, [WGPUDevice, Pointer(WGPURenderPipelineDescriptor), WGPUCreateRenderPipelineAsyncCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuDeviceCreateRenderPipelineAsync2:=dll.wgpuDeviceCreateRenderPipelineAsync2).restype, wgpuDeviceCreateRenderPipelineAsync2.argtypes = WGPUFuture, [WGPUDevice, Pointer(WGPURenderPipelineDescriptor), WGPUCreateRenderPipelineAsyncCallbackInfo2]
except AttributeError: pass

try: (wgpuDeviceCreateRenderPipelineAsyncF:=dll.wgpuDeviceCreateRenderPipelineAsyncF).restype, wgpuDeviceCreateRenderPipelineAsyncF.argtypes = WGPUFuture, [WGPUDevice, Pointer(WGPURenderPipelineDescriptor), WGPUCreateRenderPipelineAsyncCallbackInfo]
except AttributeError: pass

try: (wgpuDeviceCreateSampler:=dll.wgpuDeviceCreateSampler).restype, wgpuDeviceCreateSampler.argtypes = WGPUSampler, [WGPUDevice, Pointer(WGPUSamplerDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateShaderModule:=dll.wgpuDeviceCreateShaderModule).restype, wgpuDeviceCreateShaderModule.argtypes = WGPUShaderModule, [WGPUDevice, Pointer(WGPUShaderModuleDescriptor)]
except AttributeError: pass

try: (wgpuDeviceCreateTexture:=dll.wgpuDeviceCreateTexture).restype, wgpuDeviceCreateTexture.argtypes = WGPUTexture, [WGPUDevice, Pointer(WGPUTextureDescriptor)]
except AttributeError: pass

try: (wgpuDeviceDestroy:=dll.wgpuDeviceDestroy).restype, wgpuDeviceDestroy.argtypes = None, [WGPUDevice]
except AttributeError: pass

try: (wgpuDeviceForceLoss:=dll.wgpuDeviceForceLoss).restype, wgpuDeviceForceLoss.argtypes = None, [WGPUDevice, WGPUDeviceLostReason, WGPUStringView]
except AttributeError: pass

try: (wgpuDeviceGetAHardwareBufferProperties:=dll.wgpuDeviceGetAHardwareBufferProperties).restype, wgpuDeviceGetAHardwareBufferProperties.argtypes = WGPUStatus, [WGPUDevice, ctypes.c_void_p, Pointer(WGPUAHardwareBufferProperties)]
except AttributeError: pass

try: (wgpuDeviceGetAdapter:=dll.wgpuDeviceGetAdapter).restype, wgpuDeviceGetAdapter.argtypes = WGPUAdapter, [WGPUDevice]
except AttributeError: pass

try: (wgpuDeviceGetAdapterInfo:=dll.wgpuDeviceGetAdapterInfo).restype, wgpuDeviceGetAdapterInfo.argtypes = WGPUStatus, [WGPUDevice, Pointer(WGPUAdapterInfo)]
except AttributeError: pass

try: (wgpuDeviceGetFeatures:=dll.wgpuDeviceGetFeatures).restype, wgpuDeviceGetFeatures.argtypes = None, [WGPUDevice, Pointer(WGPUSupportedFeatures)]
except AttributeError: pass

try: (wgpuDeviceGetLimits:=dll.wgpuDeviceGetLimits).restype, wgpuDeviceGetLimits.argtypes = WGPUStatus, [WGPUDevice, Pointer(WGPUSupportedLimits)]
except AttributeError: pass

try: (wgpuDeviceGetLostFuture:=dll.wgpuDeviceGetLostFuture).restype, wgpuDeviceGetLostFuture.argtypes = WGPUFuture, [WGPUDevice]
except AttributeError: pass

try: (wgpuDeviceGetQueue:=dll.wgpuDeviceGetQueue).restype, wgpuDeviceGetQueue.argtypes = WGPUQueue, [WGPUDevice]
except AttributeError: pass

try: (wgpuDeviceHasFeature:=dll.wgpuDeviceHasFeature).restype, wgpuDeviceHasFeature.argtypes = WGPUBool, [WGPUDevice, WGPUFeatureName]
except AttributeError: pass

try: (wgpuDeviceImportSharedBufferMemory:=dll.wgpuDeviceImportSharedBufferMemory).restype, wgpuDeviceImportSharedBufferMemory.argtypes = WGPUSharedBufferMemory, [WGPUDevice, Pointer(WGPUSharedBufferMemoryDescriptor)]
except AttributeError: pass

try: (wgpuDeviceImportSharedFence:=dll.wgpuDeviceImportSharedFence).restype, wgpuDeviceImportSharedFence.argtypes = WGPUSharedFence, [WGPUDevice, Pointer(WGPUSharedFenceDescriptor)]
except AttributeError: pass

try: (wgpuDeviceImportSharedTextureMemory:=dll.wgpuDeviceImportSharedTextureMemory).restype, wgpuDeviceImportSharedTextureMemory.argtypes = WGPUSharedTextureMemory, [WGPUDevice, Pointer(WGPUSharedTextureMemoryDescriptor)]
except AttributeError: pass

try: (wgpuDeviceInjectError:=dll.wgpuDeviceInjectError).restype, wgpuDeviceInjectError.argtypes = None, [WGPUDevice, WGPUErrorType, WGPUStringView]
except AttributeError: pass

try: (wgpuDevicePopErrorScope:=dll.wgpuDevicePopErrorScope).restype, wgpuDevicePopErrorScope.argtypes = None, [WGPUDevice, WGPUErrorCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuDevicePopErrorScope2:=dll.wgpuDevicePopErrorScope2).restype, wgpuDevicePopErrorScope2.argtypes = WGPUFuture, [WGPUDevice, WGPUPopErrorScopeCallbackInfo2]
except AttributeError: pass

try: (wgpuDevicePopErrorScopeF:=dll.wgpuDevicePopErrorScopeF).restype, wgpuDevicePopErrorScopeF.argtypes = WGPUFuture, [WGPUDevice, WGPUPopErrorScopeCallbackInfo]
except AttributeError: pass

try: (wgpuDevicePushErrorScope:=dll.wgpuDevicePushErrorScope).restype, wgpuDevicePushErrorScope.argtypes = None, [WGPUDevice, WGPUErrorFilter]
except AttributeError: pass

try: (wgpuDeviceSetLabel:=dll.wgpuDeviceSetLabel).restype, wgpuDeviceSetLabel.argtypes = None, [WGPUDevice, WGPUStringView]
except AttributeError: pass

try: (wgpuDeviceSetLoggingCallback:=dll.wgpuDeviceSetLoggingCallback).restype, wgpuDeviceSetLoggingCallback.argtypes = None, [WGPUDevice, WGPULoggingCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuDeviceTick:=dll.wgpuDeviceTick).restype, wgpuDeviceTick.argtypes = None, [WGPUDevice]
except AttributeError: pass

try: (wgpuDeviceValidateTextureDescriptor:=dll.wgpuDeviceValidateTextureDescriptor).restype, wgpuDeviceValidateTextureDescriptor.argtypes = None, [WGPUDevice, Pointer(WGPUTextureDescriptor)]
except AttributeError: pass

try: (wgpuDeviceAddRef:=dll.wgpuDeviceAddRef).restype, wgpuDeviceAddRef.argtypes = None, [WGPUDevice]
except AttributeError: pass

try: (wgpuDeviceRelease:=dll.wgpuDeviceRelease).restype, wgpuDeviceRelease.argtypes = None, [WGPUDevice]
except AttributeError: pass

try: (wgpuExternalTextureDestroy:=dll.wgpuExternalTextureDestroy).restype, wgpuExternalTextureDestroy.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

try: (wgpuExternalTextureExpire:=dll.wgpuExternalTextureExpire).restype, wgpuExternalTextureExpire.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

try: (wgpuExternalTextureRefresh:=dll.wgpuExternalTextureRefresh).restype, wgpuExternalTextureRefresh.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

try: (wgpuExternalTextureSetLabel:=dll.wgpuExternalTextureSetLabel).restype, wgpuExternalTextureSetLabel.argtypes = None, [WGPUExternalTexture, WGPUStringView]
except AttributeError: pass

try: (wgpuExternalTextureAddRef:=dll.wgpuExternalTextureAddRef).restype, wgpuExternalTextureAddRef.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

try: (wgpuExternalTextureRelease:=dll.wgpuExternalTextureRelease).restype, wgpuExternalTextureRelease.argtypes = None, [WGPUExternalTexture]
except AttributeError: pass

try: (wgpuInstanceCreateSurface:=dll.wgpuInstanceCreateSurface).restype, wgpuInstanceCreateSurface.argtypes = WGPUSurface, [WGPUInstance, Pointer(WGPUSurfaceDescriptor)]
except AttributeError: pass

try: (wgpuInstanceEnumerateWGSLLanguageFeatures:=dll.wgpuInstanceEnumerateWGSLLanguageFeatures).restype, wgpuInstanceEnumerateWGSLLanguageFeatures.argtypes = size_t, [WGPUInstance, Pointer(WGPUWGSLFeatureName)]
except AttributeError: pass

try: (wgpuInstanceHasWGSLLanguageFeature:=dll.wgpuInstanceHasWGSLLanguageFeature).restype, wgpuInstanceHasWGSLLanguageFeature.argtypes = WGPUBool, [WGPUInstance, WGPUWGSLFeatureName]
except AttributeError: pass

try: (wgpuInstanceProcessEvents:=dll.wgpuInstanceProcessEvents).restype, wgpuInstanceProcessEvents.argtypes = None, [WGPUInstance]
except AttributeError: pass

try: (wgpuInstanceRequestAdapter:=dll.wgpuInstanceRequestAdapter).restype, wgpuInstanceRequestAdapter.argtypes = None, [WGPUInstance, Pointer(WGPURequestAdapterOptions), WGPURequestAdapterCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuInstanceRequestAdapter2:=dll.wgpuInstanceRequestAdapter2).restype, wgpuInstanceRequestAdapter2.argtypes = WGPUFuture, [WGPUInstance, Pointer(WGPURequestAdapterOptions), WGPURequestAdapterCallbackInfo2]
except AttributeError: pass

try: (wgpuInstanceRequestAdapterF:=dll.wgpuInstanceRequestAdapterF).restype, wgpuInstanceRequestAdapterF.argtypes = WGPUFuture, [WGPUInstance, Pointer(WGPURequestAdapterOptions), WGPURequestAdapterCallbackInfo]
except AttributeError: pass

try: (wgpuInstanceWaitAny:=dll.wgpuInstanceWaitAny).restype, wgpuInstanceWaitAny.argtypes = WGPUWaitStatus, [WGPUInstance, size_t, Pointer(WGPUFutureWaitInfo), uint64_t]
except AttributeError: pass

try: (wgpuInstanceAddRef:=dll.wgpuInstanceAddRef).restype, wgpuInstanceAddRef.argtypes = None, [WGPUInstance]
except AttributeError: pass

try: (wgpuInstanceRelease:=dll.wgpuInstanceRelease).restype, wgpuInstanceRelease.argtypes = None, [WGPUInstance]
except AttributeError: pass

try: (wgpuPipelineLayoutSetLabel:=dll.wgpuPipelineLayoutSetLabel).restype, wgpuPipelineLayoutSetLabel.argtypes = None, [WGPUPipelineLayout, WGPUStringView]
except AttributeError: pass

try: (wgpuPipelineLayoutAddRef:=dll.wgpuPipelineLayoutAddRef).restype, wgpuPipelineLayoutAddRef.argtypes = None, [WGPUPipelineLayout]
except AttributeError: pass

try: (wgpuPipelineLayoutRelease:=dll.wgpuPipelineLayoutRelease).restype, wgpuPipelineLayoutRelease.argtypes = None, [WGPUPipelineLayout]
except AttributeError: pass

try: (wgpuQuerySetDestroy:=dll.wgpuQuerySetDestroy).restype, wgpuQuerySetDestroy.argtypes = None, [WGPUQuerySet]
except AttributeError: pass

try: (wgpuQuerySetGetCount:=dll.wgpuQuerySetGetCount).restype, wgpuQuerySetGetCount.argtypes = uint32_t, [WGPUQuerySet]
except AttributeError: pass

try: (wgpuQuerySetGetType:=dll.wgpuQuerySetGetType).restype, wgpuQuerySetGetType.argtypes = WGPUQueryType, [WGPUQuerySet]
except AttributeError: pass

try: (wgpuQuerySetSetLabel:=dll.wgpuQuerySetSetLabel).restype, wgpuQuerySetSetLabel.argtypes = None, [WGPUQuerySet, WGPUStringView]
except AttributeError: pass

try: (wgpuQuerySetAddRef:=dll.wgpuQuerySetAddRef).restype, wgpuQuerySetAddRef.argtypes = None, [WGPUQuerySet]
except AttributeError: pass

try: (wgpuQuerySetRelease:=dll.wgpuQuerySetRelease).restype, wgpuQuerySetRelease.argtypes = None, [WGPUQuerySet]
except AttributeError: pass

try: (wgpuQueueCopyExternalTextureForBrowser:=dll.wgpuQueueCopyExternalTextureForBrowser).restype, wgpuQueueCopyExternalTextureForBrowser.argtypes = None, [WGPUQueue, Pointer(WGPUImageCopyExternalTexture), Pointer(WGPUImageCopyTexture), Pointer(WGPUExtent3D), Pointer(WGPUCopyTextureForBrowserOptions)]
except AttributeError: pass

try: (wgpuQueueCopyTextureForBrowser:=dll.wgpuQueueCopyTextureForBrowser).restype, wgpuQueueCopyTextureForBrowser.argtypes = None, [WGPUQueue, Pointer(WGPUImageCopyTexture), Pointer(WGPUImageCopyTexture), Pointer(WGPUExtent3D), Pointer(WGPUCopyTextureForBrowserOptions)]
except AttributeError: pass

try: (wgpuQueueOnSubmittedWorkDone:=dll.wgpuQueueOnSubmittedWorkDone).restype, wgpuQueueOnSubmittedWorkDone.argtypes = None, [WGPUQueue, WGPUQueueWorkDoneCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuQueueOnSubmittedWorkDone2:=dll.wgpuQueueOnSubmittedWorkDone2).restype, wgpuQueueOnSubmittedWorkDone2.argtypes = WGPUFuture, [WGPUQueue, WGPUQueueWorkDoneCallbackInfo2]
except AttributeError: pass

try: (wgpuQueueOnSubmittedWorkDoneF:=dll.wgpuQueueOnSubmittedWorkDoneF).restype, wgpuQueueOnSubmittedWorkDoneF.argtypes = WGPUFuture, [WGPUQueue, WGPUQueueWorkDoneCallbackInfo]
except AttributeError: pass

try: (wgpuQueueSetLabel:=dll.wgpuQueueSetLabel).restype, wgpuQueueSetLabel.argtypes = None, [WGPUQueue, WGPUStringView]
except AttributeError: pass

try: (wgpuQueueSubmit:=dll.wgpuQueueSubmit).restype, wgpuQueueSubmit.argtypes = None, [WGPUQueue, size_t, Pointer(WGPUCommandBuffer)]
except AttributeError: pass

try: (wgpuQueueWriteBuffer:=dll.wgpuQueueWriteBuffer).restype, wgpuQueueWriteBuffer.argtypes = None, [WGPUQueue, WGPUBuffer, uint64_t, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (wgpuQueueWriteTexture:=dll.wgpuQueueWriteTexture).restype, wgpuQueueWriteTexture.argtypes = None, [WGPUQueue, Pointer(WGPUImageCopyTexture), ctypes.c_void_p, size_t, Pointer(WGPUTextureDataLayout), Pointer(WGPUExtent3D)]
except AttributeError: pass

try: (wgpuQueueAddRef:=dll.wgpuQueueAddRef).restype, wgpuQueueAddRef.argtypes = None, [WGPUQueue]
except AttributeError: pass

try: (wgpuQueueRelease:=dll.wgpuQueueRelease).restype, wgpuQueueRelease.argtypes = None, [WGPUQueue]
except AttributeError: pass

try: (wgpuRenderBundleSetLabel:=dll.wgpuRenderBundleSetLabel).restype, wgpuRenderBundleSetLabel.argtypes = None, [WGPURenderBundle, WGPUStringView]
except AttributeError: pass

try: (wgpuRenderBundleAddRef:=dll.wgpuRenderBundleAddRef).restype, wgpuRenderBundleAddRef.argtypes = None, [WGPURenderBundle]
except AttributeError: pass

try: (wgpuRenderBundleRelease:=dll.wgpuRenderBundleRelease).restype, wgpuRenderBundleRelease.argtypes = None, [WGPURenderBundle]
except AttributeError: pass

try: (wgpuRenderBundleEncoderDraw:=dll.wgpuRenderBundleEncoderDraw).restype, wgpuRenderBundleEncoderDraw.argtypes = None, [WGPURenderBundleEncoder, uint32_t, uint32_t, uint32_t, uint32_t]
except AttributeError: pass

try: (wgpuRenderBundleEncoderDrawIndexed:=dll.wgpuRenderBundleEncoderDrawIndexed).restype, wgpuRenderBundleEncoderDrawIndexed.argtypes = None, [WGPURenderBundleEncoder, uint32_t, uint32_t, uint32_t, int32_t, uint32_t]
except AttributeError: pass

try: (wgpuRenderBundleEncoderDrawIndexedIndirect:=dll.wgpuRenderBundleEncoderDrawIndexedIndirect).restype, wgpuRenderBundleEncoderDrawIndexedIndirect.argtypes = None, [WGPURenderBundleEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

try: (wgpuRenderBundleEncoderDrawIndirect:=dll.wgpuRenderBundleEncoderDrawIndirect).restype, wgpuRenderBundleEncoderDrawIndirect.argtypes = None, [WGPURenderBundleEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

try: (wgpuRenderBundleEncoderFinish:=dll.wgpuRenderBundleEncoderFinish).restype, wgpuRenderBundleEncoderFinish.argtypes = WGPURenderBundle, [WGPURenderBundleEncoder, Pointer(WGPURenderBundleDescriptor)]
except AttributeError: pass

try: (wgpuRenderBundleEncoderInsertDebugMarker:=dll.wgpuRenderBundleEncoderInsertDebugMarker).restype, wgpuRenderBundleEncoderInsertDebugMarker.argtypes = None, [WGPURenderBundleEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuRenderBundleEncoderPopDebugGroup:=dll.wgpuRenderBundleEncoderPopDebugGroup).restype, wgpuRenderBundleEncoderPopDebugGroup.argtypes = None, [WGPURenderBundleEncoder]
except AttributeError: pass

try: (wgpuRenderBundleEncoderPushDebugGroup:=dll.wgpuRenderBundleEncoderPushDebugGroup).restype, wgpuRenderBundleEncoderPushDebugGroup.argtypes = None, [WGPURenderBundleEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuRenderBundleEncoderSetBindGroup:=dll.wgpuRenderBundleEncoderSetBindGroup).restype, wgpuRenderBundleEncoderSetBindGroup.argtypes = None, [WGPURenderBundleEncoder, uint32_t, WGPUBindGroup, size_t, Pointer(uint32_t)]
except AttributeError: pass

try: (wgpuRenderBundleEncoderSetIndexBuffer:=dll.wgpuRenderBundleEncoderSetIndexBuffer).restype, wgpuRenderBundleEncoderSetIndexBuffer.argtypes = None, [WGPURenderBundleEncoder, WGPUBuffer, WGPUIndexFormat, uint64_t, uint64_t]
except AttributeError: pass

try: (wgpuRenderBundleEncoderSetLabel:=dll.wgpuRenderBundleEncoderSetLabel).restype, wgpuRenderBundleEncoderSetLabel.argtypes = None, [WGPURenderBundleEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuRenderBundleEncoderSetPipeline:=dll.wgpuRenderBundleEncoderSetPipeline).restype, wgpuRenderBundleEncoderSetPipeline.argtypes = None, [WGPURenderBundleEncoder, WGPURenderPipeline]
except AttributeError: pass

try: (wgpuRenderBundleEncoderSetVertexBuffer:=dll.wgpuRenderBundleEncoderSetVertexBuffer).restype, wgpuRenderBundleEncoderSetVertexBuffer.argtypes = None, [WGPURenderBundleEncoder, uint32_t, WGPUBuffer, uint64_t, uint64_t]
except AttributeError: pass

try: (wgpuRenderBundleEncoderAddRef:=dll.wgpuRenderBundleEncoderAddRef).restype, wgpuRenderBundleEncoderAddRef.argtypes = None, [WGPURenderBundleEncoder]
except AttributeError: pass

try: (wgpuRenderBundleEncoderRelease:=dll.wgpuRenderBundleEncoderRelease).restype, wgpuRenderBundleEncoderRelease.argtypes = None, [WGPURenderBundleEncoder]
except AttributeError: pass

try: (wgpuRenderPassEncoderBeginOcclusionQuery:=dll.wgpuRenderPassEncoderBeginOcclusionQuery).restype, wgpuRenderPassEncoderBeginOcclusionQuery.argtypes = None, [WGPURenderPassEncoder, uint32_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderDraw:=dll.wgpuRenderPassEncoderDraw).restype, wgpuRenderPassEncoderDraw.argtypes = None, [WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, uint32_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderDrawIndexed:=dll.wgpuRenderPassEncoderDrawIndexed).restype, wgpuRenderPassEncoderDrawIndexed.argtypes = None, [WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, int32_t, uint32_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderDrawIndexedIndirect:=dll.wgpuRenderPassEncoderDrawIndexedIndirect).restype, wgpuRenderPassEncoderDrawIndexedIndirect.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderDrawIndirect:=dll.wgpuRenderPassEncoderDrawIndirect).restype, wgpuRenderPassEncoderDrawIndirect.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, uint64_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderEnd:=dll.wgpuRenderPassEncoderEnd).restype, wgpuRenderPassEncoderEnd.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

try: (wgpuRenderPassEncoderEndOcclusionQuery:=dll.wgpuRenderPassEncoderEndOcclusionQuery).restype, wgpuRenderPassEncoderEndOcclusionQuery.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

try: (wgpuRenderPassEncoderExecuteBundles:=dll.wgpuRenderPassEncoderExecuteBundles).restype, wgpuRenderPassEncoderExecuteBundles.argtypes = None, [WGPURenderPassEncoder, size_t, Pointer(WGPURenderBundle)]
except AttributeError: pass

try: (wgpuRenderPassEncoderInsertDebugMarker:=dll.wgpuRenderPassEncoderInsertDebugMarker).restype, wgpuRenderPassEncoderInsertDebugMarker.argtypes = None, [WGPURenderPassEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuRenderPassEncoderMultiDrawIndexedIndirect:=dll.wgpuRenderPassEncoderMultiDrawIndexedIndirect).restype, wgpuRenderPassEncoderMultiDrawIndexedIndirect.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, uint64_t, uint32_t, WGPUBuffer, uint64_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderMultiDrawIndirect:=dll.wgpuRenderPassEncoderMultiDrawIndirect).restype, wgpuRenderPassEncoderMultiDrawIndirect.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, uint64_t, uint32_t, WGPUBuffer, uint64_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderPixelLocalStorageBarrier:=dll.wgpuRenderPassEncoderPixelLocalStorageBarrier).restype, wgpuRenderPassEncoderPixelLocalStorageBarrier.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

try: (wgpuRenderPassEncoderPopDebugGroup:=dll.wgpuRenderPassEncoderPopDebugGroup).restype, wgpuRenderPassEncoderPopDebugGroup.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

try: (wgpuRenderPassEncoderPushDebugGroup:=dll.wgpuRenderPassEncoderPushDebugGroup).restype, wgpuRenderPassEncoderPushDebugGroup.argtypes = None, [WGPURenderPassEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetBindGroup:=dll.wgpuRenderPassEncoderSetBindGroup).restype, wgpuRenderPassEncoderSetBindGroup.argtypes = None, [WGPURenderPassEncoder, uint32_t, WGPUBindGroup, size_t, Pointer(uint32_t)]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetBlendConstant:=dll.wgpuRenderPassEncoderSetBlendConstant).restype, wgpuRenderPassEncoderSetBlendConstant.argtypes = None, [WGPURenderPassEncoder, Pointer(WGPUColor)]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetIndexBuffer:=dll.wgpuRenderPassEncoderSetIndexBuffer).restype, wgpuRenderPassEncoderSetIndexBuffer.argtypes = None, [WGPURenderPassEncoder, WGPUBuffer, WGPUIndexFormat, uint64_t, uint64_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetLabel:=dll.wgpuRenderPassEncoderSetLabel).restype, wgpuRenderPassEncoderSetLabel.argtypes = None, [WGPURenderPassEncoder, WGPUStringView]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetPipeline:=dll.wgpuRenderPassEncoderSetPipeline).restype, wgpuRenderPassEncoderSetPipeline.argtypes = None, [WGPURenderPassEncoder, WGPURenderPipeline]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetScissorRect:=dll.wgpuRenderPassEncoderSetScissorRect).restype, wgpuRenderPassEncoderSetScissorRect.argtypes = None, [WGPURenderPassEncoder, uint32_t, uint32_t, uint32_t, uint32_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetStencilReference:=dll.wgpuRenderPassEncoderSetStencilReference).restype, wgpuRenderPassEncoderSetStencilReference.argtypes = None, [WGPURenderPassEncoder, uint32_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetVertexBuffer:=dll.wgpuRenderPassEncoderSetVertexBuffer).restype, wgpuRenderPassEncoderSetVertexBuffer.argtypes = None, [WGPURenderPassEncoder, uint32_t, WGPUBuffer, uint64_t, uint64_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderSetViewport:=dll.wgpuRenderPassEncoderSetViewport).restype, wgpuRenderPassEncoderSetViewport.argtypes = None, [WGPURenderPassEncoder, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
except AttributeError: pass

try: (wgpuRenderPassEncoderWriteTimestamp:=dll.wgpuRenderPassEncoderWriteTimestamp).restype, wgpuRenderPassEncoderWriteTimestamp.argtypes = None, [WGPURenderPassEncoder, WGPUQuerySet, uint32_t]
except AttributeError: pass

try: (wgpuRenderPassEncoderAddRef:=dll.wgpuRenderPassEncoderAddRef).restype, wgpuRenderPassEncoderAddRef.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

try: (wgpuRenderPassEncoderRelease:=dll.wgpuRenderPassEncoderRelease).restype, wgpuRenderPassEncoderRelease.argtypes = None, [WGPURenderPassEncoder]
except AttributeError: pass

try: (wgpuRenderPipelineGetBindGroupLayout:=dll.wgpuRenderPipelineGetBindGroupLayout).restype, wgpuRenderPipelineGetBindGroupLayout.argtypes = WGPUBindGroupLayout, [WGPURenderPipeline, uint32_t]
except AttributeError: pass

try: (wgpuRenderPipelineSetLabel:=dll.wgpuRenderPipelineSetLabel).restype, wgpuRenderPipelineSetLabel.argtypes = None, [WGPURenderPipeline, WGPUStringView]
except AttributeError: pass

try: (wgpuRenderPipelineAddRef:=dll.wgpuRenderPipelineAddRef).restype, wgpuRenderPipelineAddRef.argtypes = None, [WGPURenderPipeline]
except AttributeError: pass

try: (wgpuRenderPipelineRelease:=dll.wgpuRenderPipelineRelease).restype, wgpuRenderPipelineRelease.argtypes = None, [WGPURenderPipeline]
except AttributeError: pass

try: (wgpuSamplerSetLabel:=dll.wgpuSamplerSetLabel).restype, wgpuSamplerSetLabel.argtypes = None, [WGPUSampler, WGPUStringView]
except AttributeError: pass

try: (wgpuSamplerAddRef:=dll.wgpuSamplerAddRef).restype, wgpuSamplerAddRef.argtypes = None, [WGPUSampler]
except AttributeError: pass

try: (wgpuSamplerRelease:=dll.wgpuSamplerRelease).restype, wgpuSamplerRelease.argtypes = None, [WGPUSampler]
except AttributeError: pass

try: (wgpuShaderModuleGetCompilationInfo:=dll.wgpuShaderModuleGetCompilationInfo).restype, wgpuShaderModuleGetCompilationInfo.argtypes = None, [WGPUShaderModule, WGPUCompilationInfoCallback, ctypes.c_void_p]
except AttributeError: pass

try: (wgpuShaderModuleGetCompilationInfo2:=dll.wgpuShaderModuleGetCompilationInfo2).restype, wgpuShaderModuleGetCompilationInfo2.argtypes = WGPUFuture, [WGPUShaderModule, WGPUCompilationInfoCallbackInfo2]
except AttributeError: pass

try: (wgpuShaderModuleGetCompilationInfoF:=dll.wgpuShaderModuleGetCompilationInfoF).restype, wgpuShaderModuleGetCompilationInfoF.argtypes = WGPUFuture, [WGPUShaderModule, WGPUCompilationInfoCallbackInfo]
except AttributeError: pass

try: (wgpuShaderModuleSetLabel:=dll.wgpuShaderModuleSetLabel).restype, wgpuShaderModuleSetLabel.argtypes = None, [WGPUShaderModule, WGPUStringView]
except AttributeError: pass

try: (wgpuShaderModuleAddRef:=dll.wgpuShaderModuleAddRef).restype, wgpuShaderModuleAddRef.argtypes = None, [WGPUShaderModule]
except AttributeError: pass

try: (wgpuShaderModuleRelease:=dll.wgpuShaderModuleRelease).restype, wgpuShaderModuleRelease.argtypes = None, [WGPUShaderModule]
except AttributeError: pass

try: (wgpuSharedBufferMemoryBeginAccess:=dll.wgpuSharedBufferMemoryBeginAccess).restype, wgpuSharedBufferMemoryBeginAccess.argtypes = WGPUStatus, [WGPUSharedBufferMemory, WGPUBuffer, Pointer(WGPUSharedBufferMemoryBeginAccessDescriptor)]
except AttributeError: pass

try: (wgpuSharedBufferMemoryCreateBuffer:=dll.wgpuSharedBufferMemoryCreateBuffer).restype, wgpuSharedBufferMemoryCreateBuffer.argtypes = WGPUBuffer, [WGPUSharedBufferMemory, Pointer(WGPUBufferDescriptor)]
except AttributeError: pass

try: (wgpuSharedBufferMemoryEndAccess:=dll.wgpuSharedBufferMemoryEndAccess).restype, wgpuSharedBufferMemoryEndAccess.argtypes = WGPUStatus, [WGPUSharedBufferMemory, WGPUBuffer, Pointer(WGPUSharedBufferMemoryEndAccessState)]
except AttributeError: pass

try: (wgpuSharedBufferMemoryGetProperties:=dll.wgpuSharedBufferMemoryGetProperties).restype, wgpuSharedBufferMemoryGetProperties.argtypes = WGPUStatus, [WGPUSharedBufferMemory, Pointer(WGPUSharedBufferMemoryProperties)]
except AttributeError: pass

try: (wgpuSharedBufferMemoryIsDeviceLost:=dll.wgpuSharedBufferMemoryIsDeviceLost).restype, wgpuSharedBufferMemoryIsDeviceLost.argtypes = WGPUBool, [WGPUSharedBufferMemory]
except AttributeError: pass

try: (wgpuSharedBufferMemorySetLabel:=dll.wgpuSharedBufferMemorySetLabel).restype, wgpuSharedBufferMemorySetLabel.argtypes = None, [WGPUSharedBufferMemory, WGPUStringView]
except AttributeError: pass

try: (wgpuSharedBufferMemoryAddRef:=dll.wgpuSharedBufferMemoryAddRef).restype, wgpuSharedBufferMemoryAddRef.argtypes = None, [WGPUSharedBufferMemory]
except AttributeError: pass

try: (wgpuSharedBufferMemoryRelease:=dll.wgpuSharedBufferMemoryRelease).restype, wgpuSharedBufferMemoryRelease.argtypes = None, [WGPUSharedBufferMemory]
except AttributeError: pass

try: (wgpuSharedFenceExportInfo:=dll.wgpuSharedFenceExportInfo).restype, wgpuSharedFenceExportInfo.argtypes = None, [WGPUSharedFence, Pointer(WGPUSharedFenceExportInfo)]
except AttributeError: pass

try: (wgpuSharedFenceAddRef:=dll.wgpuSharedFenceAddRef).restype, wgpuSharedFenceAddRef.argtypes = None, [WGPUSharedFence]
except AttributeError: pass

try: (wgpuSharedFenceRelease:=dll.wgpuSharedFenceRelease).restype, wgpuSharedFenceRelease.argtypes = None, [WGPUSharedFence]
except AttributeError: pass

try: (wgpuSharedTextureMemoryBeginAccess:=dll.wgpuSharedTextureMemoryBeginAccess).restype, wgpuSharedTextureMemoryBeginAccess.argtypes = WGPUStatus, [WGPUSharedTextureMemory, WGPUTexture, Pointer(WGPUSharedTextureMemoryBeginAccessDescriptor)]
except AttributeError: pass

try: (wgpuSharedTextureMemoryCreateTexture:=dll.wgpuSharedTextureMemoryCreateTexture).restype, wgpuSharedTextureMemoryCreateTexture.argtypes = WGPUTexture, [WGPUSharedTextureMemory, Pointer(WGPUTextureDescriptor)]
except AttributeError: pass

try: (wgpuSharedTextureMemoryEndAccess:=dll.wgpuSharedTextureMemoryEndAccess).restype, wgpuSharedTextureMemoryEndAccess.argtypes = WGPUStatus, [WGPUSharedTextureMemory, WGPUTexture, Pointer(WGPUSharedTextureMemoryEndAccessState)]
except AttributeError: pass

try: (wgpuSharedTextureMemoryGetProperties:=dll.wgpuSharedTextureMemoryGetProperties).restype, wgpuSharedTextureMemoryGetProperties.argtypes = WGPUStatus, [WGPUSharedTextureMemory, Pointer(WGPUSharedTextureMemoryProperties)]
except AttributeError: pass

try: (wgpuSharedTextureMemoryIsDeviceLost:=dll.wgpuSharedTextureMemoryIsDeviceLost).restype, wgpuSharedTextureMemoryIsDeviceLost.argtypes = WGPUBool, [WGPUSharedTextureMemory]
except AttributeError: pass

try: (wgpuSharedTextureMemorySetLabel:=dll.wgpuSharedTextureMemorySetLabel).restype, wgpuSharedTextureMemorySetLabel.argtypes = None, [WGPUSharedTextureMemory, WGPUStringView]
except AttributeError: pass

try: (wgpuSharedTextureMemoryAddRef:=dll.wgpuSharedTextureMemoryAddRef).restype, wgpuSharedTextureMemoryAddRef.argtypes = None, [WGPUSharedTextureMemory]
except AttributeError: pass

try: (wgpuSharedTextureMemoryRelease:=dll.wgpuSharedTextureMemoryRelease).restype, wgpuSharedTextureMemoryRelease.argtypes = None, [WGPUSharedTextureMemory]
except AttributeError: pass

try: (wgpuSurfaceConfigure:=dll.wgpuSurfaceConfigure).restype, wgpuSurfaceConfigure.argtypes = None, [WGPUSurface, Pointer(WGPUSurfaceConfiguration)]
except AttributeError: pass

try: (wgpuSurfaceGetCapabilities:=dll.wgpuSurfaceGetCapabilities).restype, wgpuSurfaceGetCapabilities.argtypes = WGPUStatus, [WGPUSurface, WGPUAdapter, Pointer(WGPUSurfaceCapabilities)]
except AttributeError: pass

try: (wgpuSurfaceGetCurrentTexture:=dll.wgpuSurfaceGetCurrentTexture).restype, wgpuSurfaceGetCurrentTexture.argtypes = None, [WGPUSurface, Pointer(WGPUSurfaceTexture)]
except AttributeError: pass

try: (wgpuSurfacePresent:=dll.wgpuSurfacePresent).restype, wgpuSurfacePresent.argtypes = None, [WGPUSurface]
except AttributeError: pass

try: (wgpuSurfaceSetLabel:=dll.wgpuSurfaceSetLabel).restype, wgpuSurfaceSetLabel.argtypes = None, [WGPUSurface, WGPUStringView]
except AttributeError: pass

try: (wgpuSurfaceUnconfigure:=dll.wgpuSurfaceUnconfigure).restype, wgpuSurfaceUnconfigure.argtypes = None, [WGPUSurface]
except AttributeError: pass

try: (wgpuSurfaceAddRef:=dll.wgpuSurfaceAddRef).restype, wgpuSurfaceAddRef.argtypes = None, [WGPUSurface]
except AttributeError: pass

try: (wgpuSurfaceRelease:=dll.wgpuSurfaceRelease).restype, wgpuSurfaceRelease.argtypes = None, [WGPUSurface]
except AttributeError: pass

try: (wgpuTextureCreateErrorView:=dll.wgpuTextureCreateErrorView).restype, wgpuTextureCreateErrorView.argtypes = WGPUTextureView, [WGPUTexture, Pointer(WGPUTextureViewDescriptor)]
except AttributeError: pass

try: (wgpuTextureCreateView:=dll.wgpuTextureCreateView).restype, wgpuTextureCreateView.argtypes = WGPUTextureView, [WGPUTexture, Pointer(WGPUTextureViewDescriptor)]
except AttributeError: pass

try: (wgpuTextureDestroy:=dll.wgpuTextureDestroy).restype, wgpuTextureDestroy.argtypes = None, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureGetDepthOrArrayLayers:=dll.wgpuTextureGetDepthOrArrayLayers).restype, wgpuTextureGetDepthOrArrayLayers.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureGetDimension:=dll.wgpuTextureGetDimension).restype, wgpuTextureGetDimension.argtypes = WGPUTextureDimension, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureGetFormat:=dll.wgpuTextureGetFormat).restype, wgpuTextureGetFormat.argtypes = WGPUTextureFormat, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureGetHeight:=dll.wgpuTextureGetHeight).restype, wgpuTextureGetHeight.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureGetMipLevelCount:=dll.wgpuTextureGetMipLevelCount).restype, wgpuTextureGetMipLevelCount.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureGetSampleCount:=dll.wgpuTextureGetSampleCount).restype, wgpuTextureGetSampleCount.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureGetUsage:=dll.wgpuTextureGetUsage).restype, wgpuTextureGetUsage.argtypes = WGPUTextureUsage, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureGetWidth:=dll.wgpuTextureGetWidth).restype, wgpuTextureGetWidth.argtypes = uint32_t, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureSetLabel:=dll.wgpuTextureSetLabel).restype, wgpuTextureSetLabel.argtypes = None, [WGPUTexture, WGPUStringView]
except AttributeError: pass

try: (wgpuTextureAddRef:=dll.wgpuTextureAddRef).restype, wgpuTextureAddRef.argtypes = None, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureRelease:=dll.wgpuTextureRelease).restype, wgpuTextureRelease.argtypes = None, [WGPUTexture]
except AttributeError: pass

try: (wgpuTextureViewSetLabel:=dll.wgpuTextureViewSetLabel).restype, wgpuTextureViewSetLabel.argtypes = None, [WGPUTextureView, WGPUStringView]
except AttributeError: pass

try: (wgpuTextureViewAddRef:=dll.wgpuTextureViewAddRef).restype, wgpuTextureViewAddRef.argtypes = None, [WGPUTextureView]
except AttributeError: pass

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