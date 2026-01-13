from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import DLL, record, Array, POINTER, CFUNCTYPE, CEnum, _IO, _IOW, _IOR, _IOWR, init_records
from tinygrad.helpers import WIN, OSX
import sysconfig, os
dll = DLL('webgpu', os.path.join(sysconfig.get_paths()['purelib'], 'pydawn', 'lib', 'libwebgpu_dawn.dll') if WIN else 'webgpu_dawn')
WGPUFlags: TypeAlias = Annotated[int, ctypes.c_uint64]
WGPUBool: TypeAlias = Annotated[int, ctypes.c_uint32]
class struct_WGPUAdapterImpl(ctypes.Structure): pass
WGPUAdapter: TypeAlias = POINTER(struct_WGPUAdapterImpl)
class struct_WGPUBindGroupImpl(ctypes.Structure): pass
WGPUBindGroup: TypeAlias = POINTER(struct_WGPUBindGroupImpl)
class struct_WGPUBindGroupLayoutImpl(ctypes.Structure): pass
WGPUBindGroupLayout: TypeAlias = POINTER(struct_WGPUBindGroupLayoutImpl)
class struct_WGPUBufferImpl(ctypes.Structure): pass
WGPUBuffer: TypeAlias = POINTER(struct_WGPUBufferImpl)
class struct_WGPUCommandBufferImpl(ctypes.Structure): pass
WGPUCommandBuffer: TypeAlias = POINTER(struct_WGPUCommandBufferImpl)
class struct_WGPUCommandEncoderImpl(ctypes.Structure): pass
WGPUCommandEncoder: TypeAlias = POINTER(struct_WGPUCommandEncoderImpl)
class struct_WGPUComputePassEncoderImpl(ctypes.Structure): pass
WGPUComputePassEncoder: TypeAlias = POINTER(struct_WGPUComputePassEncoderImpl)
class struct_WGPUComputePipelineImpl(ctypes.Structure): pass
WGPUComputePipeline: TypeAlias = POINTER(struct_WGPUComputePipelineImpl)
class struct_WGPUDeviceImpl(ctypes.Structure): pass
WGPUDevice: TypeAlias = POINTER(struct_WGPUDeviceImpl)
class struct_WGPUExternalTextureImpl(ctypes.Structure): pass
WGPUExternalTexture: TypeAlias = POINTER(struct_WGPUExternalTextureImpl)
class struct_WGPUInstanceImpl(ctypes.Structure): pass
WGPUInstance: TypeAlias = POINTER(struct_WGPUInstanceImpl)
class struct_WGPUPipelineLayoutImpl(ctypes.Structure): pass
WGPUPipelineLayout: TypeAlias = POINTER(struct_WGPUPipelineLayoutImpl)
class struct_WGPUQuerySetImpl(ctypes.Structure): pass
WGPUQuerySet: TypeAlias = POINTER(struct_WGPUQuerySetImpl)
class struct_WGPUQueueImpl(ctypes.Structure): pass
WGPUQueue: TypeAlias = POINTER(struct_WGPUQueueImpl)
class struct_WGPURenderBundleImpl(ctypes.Structure): pass
WGPURenderBundle: TypeAlias = POINTER(struct_WGPURenderBundleImpl)
class struct_WGPURenderBundleEncoderImpl(ctypes.Structure): pass
WGPURenderBundleEncoder: TypeAlias = POINTER(struct_WGPURenderBundleEncoderImpl)
class struct_WGPURenderPassEncoderImpl(ctypes.Structure): pass
WGPURenderPassEncoder: TypeAlias = POINTER(struct_WGPURenderPassEncoderImpl)
class struct_WGPURenderPipelineImpl(ctypes.Structure): pass
WGPURenderPipeline: TypeAlias = POINTER(struct_WGPURenderPipelineImpl)
class struct_WGPUSamplerImpl(ctypes.Structure): pass
WGPUSampler: TypeAlias = POINTER(struct_WGPUSamplerImpl)
class struct_WGPUShaderModuleImpl(ctypes.Structure): pass
WGPUShaderModule: TypeAlias = POINTER(struct_WGPUShaderModuleImpl)
class struct_WGPUSharedBufferMemoryImpl(ctypes.Structure): pass
WGPUSharedBufferMemory: TypeAlias = POINTER(struct_WGPUSharedBufferMemoryImpl)
class struct_WGPUSharedFenceImpl(ctypes.Structure): pass
WGPUSharedFence: TypeAlias = POINTER(struct_WGPUSharedFenceImpl)
class struct_WGPUSharedTextureMemoryImpl(ctypes.Structure): pass
WGPUSharedTextureMemory: TypeAlias = POINTER(struct_WGPUSharedTextureMemoryImpl)
class struct_WGPUSurfaceImpl(ctypes.Structure): pass
WGPUSurface: TypeAlias = POINTER(struct_WGPUSurfaceImpl)
class struct_WGPUTextureImpl(ctypes.Structure): pass
WGPUTexture: TypeAlias = POINTER(struct_WGPUTextureImpl)
class struct_WGPUTextureViewImpl(ctypes.Structure): pass
WGPUTextureView: TypeAlias = POINTER(struct_WGPUTextureViewImpl)
@record
class struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER:
  SIZE = 4
  unused: Annotated[WGPUBool, 0]
@record
class struct_WGPUAdapterPropertiesD3D:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  shaderModel: Annotated[uint32_t, 16]
@record
class struct_WGPUChainedStructOut:
  SIZE = 16
  next: Annotated[POINTER(struct_WGPUChainedStructOut), 0]
  sType: Annotated[WGPUSType, 8]
WGPUChainedStructOut: TypeAlias = struct_WGPUChainedStructOut
enum_WGPUSType = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUSType: TypeAlias = enum_WGPUSType
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@record
class struct_WGPUAdapterPropertiesSubgroups:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  subgroupMinSize: Annotated[uint32_t, 16]
  subgroupMaxSize: Annotated[uint32_t, 20]
@record
class struct_WGPUAdapterPropertiesVk:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  driverVersion: Annotated[uint32_t, 16]
@record
class struct_WGPUBindGroupEntry:
  SIZE = 56
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  binding: Annotated[uint32_t, 8]
  buffer: Annotated[WGPUBuffer, 16]
  offset: Annotated[uint64_t, 24]
  size: Annotated[uint64_t, 32]
  sampler: Annotated[WGPUSampler, 40]
  textureView: Annotated[WGPUTextureView, 48]
@record
class struct_WGPUChainedStruct:
  SIZE = 16
  next: Annotated[POINTER(struct_WGPUChainedStruct), 0]
  sType: Annotated[WGPUSType, 8]
WGPUChainedStruct: TypeAlias = struct_WGPUChainedStruct
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@record
class struct_WGPUBlendComponent:
  SIZE = 12
  operation: Annotated[WGPUBlendOperation, 0]
  srcFactor: Annotated[WGPUBlendFactor, 4]
  dstFactor: Annotated[WGPUBlendFactor, 8]
enum_WGPUBlendOperation = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBlendOperation_Undefined = enum_WGPUBlendOperation.define('WGPUBlendOperation_Undefined', 0)
WGPUBlendOperation_Add = enum_WGPUBlendOperation.define('WGPUBlendOperation_Add', 1)
WGPUBlendOperation_Subtract = enum_WGPUBlendOperation.define('WGPUBlendOperation_Subtract', 2)
WGPUBlendOperation_ReverseSubtract = enum_WGPUBlendOperation.define('WGPUBlendOperation_ReverseSubtract', 3)
WGPUBlendOperation_Min = enum_WGPUBlendOperation.define('WGPUBlendOperation_Min', 4)
WGPUBlendOperation_Max = enum_WGPUBlendOperation.define('WGPUBlendOperation_Max', 5)
WGPUBlendOperation_Force32 = enum_WGPUBlendOperation.define('WGPUBlendOperation_Force32', 2147483647)

WGPUBlendOperation: TypeAlias = enum_WGPUBlendOperation
enum_WGPUBlendFactor = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUBlendFactor: TypeAlias = enum_WGPUBlendFactor
@record
class struct_WGPUBufferBindingLayout:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  type: Annotated[WGPUBufferBindingType, 8]
  hasDynamicOffset: Annotated[WGPUBool, 12]
  minBindingSize: Annotated[uint64_t, 16]
enum_WGPUBufferBindingType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBufferBindingType_BindingNotUsed = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_BindingNotUsed', 0)
WGPUBufferBindingType_Uniform = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Uniform', 1)
WGPUBufferBindingType_Storage = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Storage', 2)
WGPUBufferBindingType_ReadOnlyStorage = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_ReadOnlyStorage', 3)
WGPUBufferBindingType_Force32 = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Force32', 2147483647)

WGPUBufferBindingType: TypeAlias = enum_WGPUBufferBindingType
@record
class struct_WGPUBufferHostMappedPointer:
  SIZE = 40
  chain: Annotated[WGPUChainedStruct, 0]
  pointer: Annotated[POINTER(None), 16]
  disposeCallback: Annotated[WGPUCallback, 24]
  userdata: Annotated[POINTER(None), 32]
WGPUCallback: TypeAlias = CFUNCTYPE(None, POINTER(None))
@record
class struct_WGPUBufferMapCallbackInfo:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUBufferMapCallback, 16]
  userdata: Annotated[POINTER(None), 24]
enum_WGPUCallbackMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCallbackMode_WaitAnyOnly = enum_WGPUCallbackMode.define('WGPUCallbackMode_WaitAnyOnly', 1)
WGPUCallbackMode_AllowProcessEvents = enum_WGPUCallbackMode.define('WGPUCallbackMode_AllowProcessEvents', 2)
WGPUCallbackMode_AllowSpontaneous = enum_WGPUCallbackMode.define('WGPUCallbackMode_AllowSpontaneous', 3)
WGPUCallbackMode_Force32 = enum_WGPUCallbackMode.define('WGPUCallbackMode_Force32', 2147483647)

WGPUCallbackMode: TypeAlias = enum_WGPUCallbackMode
enum_WGPUBufferMapAsyncStatus = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUBufferMapCallback: TypeAlias = CFUNCTYPE(None, enum_WGPUBufferMapAsyncStatus, POINTER(None))
@record
class struct_WGPUColor:
  SIZE = 32
  r: Annotated[Annotated[float, ctypes.c_double], 0]
  g: Annotated[Annotated[float, ctypes.c_double], 8]
  b: Annotated[Annotated[float, ctypes.c_double], 16]
  a: Annotated[Annotated[float, ctypes.c_double], 24]
@record
class struct_WGPUColorTargetStateExpandResolveTextureDawn:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  enabled: Annotated[WGPUBool, 16]
@record
class struct_WGPUCompilationInfoCallbackInfo:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCompilationInfoCallback, 16]
  userdata: Annotated[POINTER(None), 24]
enum_WGPUCompilationInfoRequestStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCompilationInfoRequestStatus_Success = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Success', 1)
WGPUCompilationInfoRequestStatus_InstanceDropped = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_InstanceDropped', 2)
WGPUCompilationInfoRequestStatus_Error = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Error', 3)
WGPUCompilationInfoRequestStatus_DeviceLost = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_DeviceLost', 4)
WGPUCompilationInfoRequestStatus_Unknown = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Unknown', 5)
WGPUCompilationInfoRequestStatus_Force32 = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Force32', 2147483647)

@record
class struct_WGPUCompilationInfo:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  messageCount: Annotated[size_t, 8]
  messages: Annotated[POINTER(WGPUCompilationMessage), 16]
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@record
class struct_WGPUCompilationMessage:
  SIZE = 88
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  message: Annotated[WGPUStringView, 8]
  type: Annotated[WGPUCompilationMessageType, 24]
  lineNum: Annotated[uint64_t, 32]
  linePos: Annotated[uint64_t, 40]
  offset: Annotated[uint64_t, 48]
  length: Annotated[uint64_t, 56]
  utf16LinePos: Annotated[uint64_t, 64]
  utf16Offset: Annotated[uint64_t, 72]
  utf16Length: Annotated[uint64_t, 80]
WGPUCompilationMessage: TypeAlias = struct_WGPUCompilationMessage
@record
class struct_WGPUStringView:
  SIZE = 16
  data: Annotated[POINTER(Annotated[bytes, ctypes.c_char]), 0]
  length: Annotated[size_t, 8]
WGPUStringView: TypeAlias = struct_WGPUStringView
enum_WGPUCompilationMessageType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCompilationMessageType_Error = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Error', 1)
WGPUCompilationMessageType_Warning = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Warning', 2)
WGPUCompilationMessageType_Info = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Info', 3)
WGPUCompilationMessageType_Force32 = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Force32', 2147483647)

WGPUCompilationMessageType: TypeAlias = enum_WGPUCompilationMessageType
WGPUCompilationInfoCallback: TypeAlias = CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, POINTER(struct_WGPUCompilationInfo), POINTER(None))
@record
class struct_WGPUComputePassTimestampWrites:
  SIZE = 16
  querySet: Annotated[WGPUQuerySet, 0]
  beginningOfPassWriteIndex: Annotated[uint32_t, 8]
  endOfPassWriteIndex: Annotated[uint32_t, 12]
@record
class struct_WGPUCopyTextureForBrowserOptions:
  SIZE = 56
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  flipY: Annotated[WGPUBool, 8]
  needsColorSpaceConversion: Annotated[WGPUBool, 12]
  srcAlphaMode: Annotated[WGPUAlphaMode, 16]
  srcTransferFunctionParameters: Annotated[POINTER(Annotated[float, ctypes.c_float]), 24]
  conversionMatrix: Annotated[POINTER(Annotated[float, ctypes.c_float]), 32]
  dstTransferFunctionParameters: Annotated[POINTER(Annotated[float, ctypes.c_float]), 40]
  dstAlphaMode: Annotated[WGPUAlphaMode, 48]
  internalUsage: Annotated[WGPUBool, 52]
enum_WGPUAlphaMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUAlphaMode_Opaque = enum_WGPUAlphaMode.define('WGPUAlphaMode_Opaque', 1)
WGPUAlphaMode_Premultiplied = enum_WGPUAlphaMode.define('WGPUAlphaMode_Premultiplied', 2)
WGPUAlphaMode_Unpremultiplied = enum_WGPUAlphaMode.define('WGPUAlphaMode_Unpremultiplied', 3)
WGPUAlphaMode_Force32 = enum_WGPUAlphaMode.define('WGPUAlphaMode_Force32', 2147483647)

WGPUAlphaMode: TypeAlias = enum_WGPUAlphaMode
@record
class struct_WGPUCreateComputePipelineAsyncCallbackInfo:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCreateComputePipelineAsyncCallback, 16]
  userdata: Annotated[POINTER(None), 24]
enum_WGPUCreatePipelineAsyncStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCreatePipelineAsyncStatus_Success = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Success', 1)
WGPUCreatePipelineAsyncStatus_InstanceDropped = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_InstanceDropped', 2)
WGPUCreatePipelineAsyncStatus_ValidationError = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_ValidationError', 3)
WGPUCreatePipelineAsyncStatus_InternalError = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_InternalError', 4)
WGPUCreatePipelineAsyncStatus_DeviceLost = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_DeviceLost', 5)
WGPUCreatePipelineAsyncStatus_DeviceDestroyed = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_DeviceDestroyed', 6)
WGPUCreatePipelineAsyncStatus_Unknown = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Unknown', 7)
WGPUCreatePipelineAsyncStatus_Force32 = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Force32', 2147483647)

WGPUCreateComputePipelineAsyncCallback: TypeAlias = CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, POINTER(struct_WGPUComputePipelineImpl), struct_WGPUStringView, POINTER(None))
@record
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCreateRenderPipelineAsyncCallback, 16]
  userdata: Annotated[POINTER(None), 24]
WGPUCreateRenderPipelineAsyncCallback: TypeAlias = CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, POINTER(struct_WGPURenderPipelineImpl), struct_WGPUStringView, POINTER(None))
@record
class struct_WGPUDawnWGSLBlocklist:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  blocklistedFeatureCount: Annotated[size_t, 16]
  blocklistedFeatures: Annotated[POINTER(POINTER(Annotated[bytes, ctypes.c_char])), 24]
@record
class struct_WGPUDawnAdapterPropertiesPowerPreference:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  powerPreference: Annotated[WGPUPowerPreference, 16]
enum_WGPUPowerPreference = CEnum(Annotated[int, ctypes.c_uint32])
WGPUPowerPreference_Undefined = enum_WGPUPowerPreference.define('WGPUPowerPreference_Undefined', 0)
WGPUPowerPreference_LowPower = enum_WGPUPowerPreference.define('WGPUPowerPreference_LowPower', 1)
WGPUPowerPreference_HighPerformance = enum_WGPUPowerPreference.define('WGPUPowerPreference_HighPerformance', 2)
WGPUPowerPreference_Force32 = enum_WGPUPowerPreference.define('WGPUPowerPreference_Force32', 2147483647)

WGPUPowerPreference: TypeAlias = enum_WGPUPowerPreference
@record
class struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  outOfMemory: Annotated[WGPUBool, 16]
@record
class struct_WGPUDawnEncoderInternalUsageDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  useInternalUsages: Annotated[WGPUBool, 16]
@record
class struct_WGPUDawnExperimentalImmediateDataLimits:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  maxImmediateDataRangeByteSize: Annotated[uint32_t, 16]
@record
class struct_WGPUDawnExperimentalSubgroupLimits:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  minSubgroupSize: Annotated[uint32_t, 16]
  maxSubgroupSize: Annotated[uint32_t, 20]
@record
class struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  implicitSampleCount: Annotated[uint32_t, 16]
@record
class struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  allowNonUniformDerivatives: Annotated[WGPUBool, 16]
@record
class struct_WGPUDawnTexelCopyBufferRowAlignmentLimits:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  minTexelCopyBufferRowAlignment: Annotated[uint32_t, 16]
@record
class struct_WGPUDawnTextureInternalUsageDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  internalUsage: Annotated[WGPUTextureUsage, 16]
WGPUTextureUsage: TypeAlias = Annotated[int, ctypes.c_uint64]
@record
class struct_WGPUDawnTogglesDescriptor:
  SIZE = 48
  chain: Annotated[WGPUChainedStruct, 0]
  enabledToggleCount: Annotated[size_t, 16]
  enabledToggles: Annotated[POINTER(POINTER(Annotated[bytes, ctypes.c_char])), 24]
  disabledToggleCount: Annotated[size_t, 32]
  disabledToggles: Annotated[POINTER(POINTER(Annotated[bytes, ctypes.c_char])), 40]
@record
class struct_WGPUDawnWireWGSLControl:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  enableExperimental: Annotated[WGPUBool, 16]
  enableUnsafe: Annotated[WGPUBool, 20]
  enableTesting: Annotated[WGPUBool, 24]
@record
class struct_WGPUDeviceLostCallbackInfo:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUDeviceLostCallbackNew, 16]
  userdata: Annotated[POINTER(None), 24]
enum_WGPUDeviceLostReason = CEnum(Annotated[int, ctypes.c_uint32])
WGPUDeviceLostReason_Unknown = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Unknown', 1)
WGPUDeviceLostReason_Destroyed = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Destroyed', 2)
WGPUDeviceLostReason_InstanceDropped = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_InstanceDropped', 3)
WGPUDeviceLostReason_FailedCreation = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_FailedCreation', 4)
WGPUDeviceLostReason_Force32 = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Force32', 2147483647)

WGPUDeviceLostCallbackNew: TypeAlias = CFUNCTYPE(None, POINTER(POINTER(struct_WGPUDeviceImpl)), enum_WGPUDeviceLostReason, struct_WGPUStringView, POINTER(None))
@record
class struct_WGPUDrmFormatProperties:
  SIZE = 16
  modifier: Annotated[uint64_t, 0]
  modifierPlaneCount: Annotated[uint32_t, 8]
@record
class struct_WGPUExtent2D:
  SIZE = 8
  width: Annotated[uint32_t, 0]
  height: Annotated[uint32_t, 4]
@record
class struct_WGPUExtent3D:
  SIZE = 12
  width: Annotated[uint32_t, 0]
  height: Annotated[uint32_t, 4]
  depthOrArrayLayers: Annotated[uint32_t, 8]
@record
class struct_WGPUExternalTextureBindingEntry:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  externalTexture: Annotated[WGPUExternalTexture, 16]
@record
class struct_WGPUExternalTextureBindingLayout:
  SIZE = 16
  chain: Annotated[WGPUChainedStruct, 0]
@record
class struct_WGPUFormatCapabilities:
  SIZE = 8
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
@record
class struct_WGPUFuture:
  SIZE = 8
  id: Annotated[uint64_t, 0]
@record
class struct_WGPUInstanceFeatures:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  timedWaitAnyEnable: Annotated[WGPUBool, 8]
  timedWaitAnyMaxCount: Annotated[size_t, 16]
@record
class struct_WGPULimits:
  SIZE = 160
  maxTextureDimension1D: Annotated[uint32_t, 0]
  maxTextureDimension2D: Annotated[uint32_t, 4]
  maxTextureDimension3D: Annotated[uint32_t, 8]
  maxTextureArrayLayers: Annotated[uint32_t, 12]
  maxBindGroups: Annotated[uint32_t, 16]
  maxBindGroupsPlusVertexBuffers: Annotated[uint32_t, 20]
  maxBindingsPerBindGroup: Annotated[uint32_t, 24]
  maxDynamicUniformBuffersPerPipelineLayout: Annotated[uint32_t, 28]
  maxDynamicStorageBuffersPerPipelineLayout: Annotated[uint32_t, 32]
  maxSampledTexturesPerShaderStage: Annotated[uint32_t, 36]
  maxSamplersPerShaderStage: Annotated[uint32_t, 40]
  maxStorageBuffersPerShaderStage: Annotated[uint32_t, 44]
  maxStorageTexturesPerShaderStage: Annotated[uint32_t, 48]
  maxUniformBuffersPerShaderStage: Annotated[uint32_t, 52]
  maxUniformBufferBindingSize: Annotated[uint64_t, 56]
  maxStorageBufferBindingSize: Annotated[uint64_t, 64]
  minUniformBufferOffsetAlignment: Annotated[uint32_t, 72]
  minStorageBufferOffsetAlignment: Annotated[uint32_t, 76]
  maxVertexBuffers: Annotated[uint32_t, 80]
  maxBufferSize: Annotated[uint64_t, 88]
  maxVertexAttributes: Annotated[uint32_t, 96]
  maxVertexBufferArrayStride: Annotated[uint32_t, 100]
  maxInterStageShaderComponents: Annotated[uint32_t, 104]
  maxInterStageShaderVariables: Annotated[uint32_t, 108]
  maxColorAttachments: Annotated[uint32_t, 112]
  maxColorAttachmentBytesPerSample: Annotated[uint32_t, 116]
  maxComputeWorkgroupStorageSize: Annotated[uint32_t, 120]
  maxComputeInvocationsPerWorkgroup: Annotated[uint32_t, 124]
  maxComputeWorkgroupSizeX: Annotated[uint32_t, 128]
  maxComputeWorkgroupSizeY: Annotated[uint32_t, 132]
  maxComputeWorkgroupSizeZ: Annotated[uint32_t, 136]
  maxComputeWorkgroupsPerDimension: Annotated[uint32_t, 140]
  maxStorageBuffersInVertexStage: Annotated[uint32_t, 144]
  maxStorageTexturesInVertexStage: Annotated[uint32_t, 148]
  maxStorageBuffersInFragmentStage: Annotated[uint32_t, 152]
  maxStorageTexturesInFragmentStage: Annotated[uint32_t, 156]
@record
class struct_WGPUMemoryHeapInfo:
  SIZE = 16
  properties: Annotated[WGPUHeapProperty, 0]
  size: Annotated[uint64_t, 8]
WGPUHeapProperty: TypeAlias = Annotated[int, ctypes.c_uint64]
@record
class struct_WGPUMultisampleState:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  count: Annotated[uint32_t, 8]
  mask: Annotated[uint32_t, 12]
  alphaToCoverageEnabled: Annotated[WGPUBool, 16]
@record
class struct_WGPUOrigin2D:
  SIZE = 8
  x: Annotated[uint32_t, 0]
  y: Annotated[uint32_t, 4]
@record
class struct_WGPUOrigin3D:
  SIZE = 12
  x: Annotated[uint32_t, 0]
  y: Annotated[uint32_t, 4]
  z: Annotated[uint32_t, 8]
@record
class struct_WGPUPipelineLayoutStorageAttachment:
  SIZE = 16
  offset: Annotated[uint64_t, 0]
  format: Annotated[WGPUTextureFormat, 8]
enum_WGPUTextureFormat = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUTextureFormat: TypeAlias = enum_WGPUTextureFormat
@record
class struct_WGPUPopErrorScopeCallbackInfo:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUPopErrorScopeCallback, 16]
  oldCallback: Annotated[WGPUErrorCallback, 24]
  userdata: Annotated[POINTER(None), 32]
enum_WGPUPopErrorScopeStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUPopErrorScopeStatus_Success = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_Success', 1)
WGPUPopErrorScopeStatus_InstanceDropped = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_InstanceDropped', 2)
WGPUPopErrorScopeStatus_Force32 = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_Force32', 2147483647)

enum_WGPUErrorType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUErrorType_NoError = enum_WGPUErrorType.define('WGPUErrorType_NoError', 1)
WGPUErrorType_Validation = enum_WGPUErrorType.define('WGPUErrorType_Validation', 2)
WGPUErrorType_OutOfMemory = enum_WGPUErrorType.define('WGPUErrorType_OutOfMemory', 3)
WGPUErrorType_Internal = enum_WGPUErrorType.define('WGPUErrorType_Internal', 4)
WGPUErrorType_Unknown = enum_WGPUErrorType.define('WGPUErrorType_Unknown', 5)
WGPUErrorType_DeviceLost = enum_WGPUErrorType.define('WGPUErrorType_DeviceLost', 6)
WGPUErrorType_Force32 = enum_WGPUErrorType.define('WGPUErrorType_Force32', 2147483647)

WGPUPopErrorScopeCallback: TypeAlias = CFUNCTYPE(None, enum_WGPUPopErrorScopeStatus, enum_WGPUErrorType, struct_WGPUStringView, POINTER(None))
WGPUErrorCallback: TypeAlias = CFUNCTYPE(None, enum_WGPUErrorType, struct_WGPUStringView, POINTER(None))
@record
class struct_WGPUPrimitiveState:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  topology: Annotated[WGPUPrimitiveTopology, 8]
  stripIndexFormat: Annotated[WGPUIndexFormat, 12]
  frontFace: Annotated[WGPUFrontFace, 16]
  cullMode: Annotated[WGPUCullMode, 20]
  unclippedDepth: Annotated[WGPUBool, 24]
enum_WGPUPrimitiveTopology = CEnum(Annotated[int, ctypes.c_uint32])
WGPUPrimitiveTopology_Undefined = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_Undefined', 0)
WGPUPrimitiveTopology_PointList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_PointList', 1)
WGPUPrimitiveTopology_LineList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_LineList', 2)
WGPUPrimitiveTopology_LineStrip = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_LineStrip', 3)
WGPUPrimitiveTopology_TriangleList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_TriangleList', 4)
WGPUPrimitiveTopology_TriangleStrip = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_TriangleStrip', 5)
WGPUPrimitiveTopology_Force32 = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_Force32', 2147483647)

WGPUPrimitiveTopology: TypeAlias = enum_WGPUPrimitiveTopology
enum_WGPUIndexFormat = CEnum(Annotated[int, ctypes.c_uint32])
WGPUIndexFormat_Undefined = enum_WGPUIndexFormat.define('WGPUIndexFormat_Undefined', 0)
WGPUIndexFormat_Uint16 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Uint16', 1)
WGPUIndexFormat_Uint32 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Uint32', 2)
WGPUIndexFormat_Force32 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Force32', 2147483647)

WGPUIndexFormat: TypeAlias = enum_WGPUIndexFormat
enum_WGPUFrontFace = CEnum(Annotated[int, ctypes.c_uint32])
WGPUFrontFace_Undefined = enum_WGPUFrontFace.define('WGPUFrontFace_Undefined', 0)
WGPUFrontFace_CCW = enum_WGPUFrontFace.define('WGPUFrontFace_CCW', 1)
WGPUFrontFace_CW = enum_WGPUFrontFace.define('WGPUFrontFace_CW', 2)
WGPUFrontFace_Force32 = enum_WGPUFrontFace.define('WGPUFrontFace_Force32', 2147483647)

WGPUFrontFace: TypeAlias = enum_WGPUFrontFace
enum_WGPUCullMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCullMode_Undefined = enum_WGPUCullMode.define('WGPUCullMode_Undefined', 0)
WGPUCullMode_None = enum_WGPUCullMode.define('WGPUCullMode_None', 1)
WGPUCullMode_Front = enum_WGPUCullMode.define('WGPUCullMode_Front', 2)
WGPUCullMode_Back = enum_WGPUCullMode.define('WGPUCullMode_Back', 3)
WGPUCullMode_Force32 = enum_WGPUCullMode.define('WGPUCullMode_Force32', 2147483647)

WGPUCullMode: TypeAlias = enum_WGPUCullMode
@record
class struct_WGPUQueueWorkDoneCallbackInfo:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUQueueWorkDoneCallback, 16]
  userdata: Annotated[POINTER(None), 24]
enum_WGPUQueueWorkDoneStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUQueueWorkDoneStatus_Success = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Success', 1)
WGPUQueueWorkDoneStatus_InstanceDropped = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_InstanceDropped', 2)
WGPUQueueWorkDoneStatus_Error = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Error', 3)
WGPUQueueWorkDoneStatus_Unknown = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Unknown', 4)
WGPUQueueWorkDoneStatus_DeviceLost = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_DeviceLost', 5)
WGPUQueueWorkDoneStatus_Force32 = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Force32', 2147483647)

WGPUQueueWorkDoneCallback: TypeAlias = CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, POINTER(None))
@record
class struct_WGPURenderPassDepthStencilAttachment:
  SIZE = 40
  view: Annotated[WGPUTextureView, 0]
  depthLoadOp: Annotated[WGPULoadOp, 8]
  depthStoreOp: Annotated[WGPUStoreOp, 12]
  depthClearValue: Annotated[Annotated[float, ctypes.c_float], 16]
  depthReadOnly: Annotated[WGPUBool, 20]
  stencilLoadOp: Annotated[WGPULoadOp, 24]
  stencilStoreOp: Annotated[WGPUStoreOp, 28]
  stencilClearValue: Annotated[uint32_t, 32]
  stencilReadOnly: Annotated[WGPUBool, 36]
enum_WGPULoadOp = CEnum(Annotated[int, ctypes.c_uint32])
WGPULoadOp_Undefined = enum_WGPULoadOp.define('WGPULoadOp_Undefined', 0)
WGPULoadOp_Load = enum_WGPULoadOp.define('WGPULoadOp_Load', 1)
WGPULoadOp_Clear = enum_WGPULoadOp.define('WGPULoadOp_Clear', 2)
WGPULoadOp_ExpandResolveTexture = enum_WGPULoadOp.define('WGPULoadOp_ExpandResolveTexture', 327683)
WGPULoadOp_Force32 = enum_WGPULoadOp.define('WGPULoadOp_Force32', 2147483647)

WGPULoadOp: TypeAlias = enum_WGPULoadOp
enum_WGPUStoreOp = CEnum(Annotated[int, ctypes.c_uint32])
WGPUStoreOp_Undefined = enum_WGPUStoreOp.define('WGPUStoreOp_Undefined', 0)
WGPUStoreOp_Store = enum_WGPUStoreOp.define('WGPUStoreOp_Store', 1)
WGPUStoreOp_Discard = enum_WGPUStoreOp.define('WGPUStoreOp_Discard', 2)
WGPUStoreOp_Force32 = enum_WGPUStoreOp.define('WGPUStoreOp_Force32', 2147483647)

WGPUStoreOp: TypeAlias = enum_WGPUStoreOp
@record
class struct_WGPURenderPassDescriptorExpandResolveRect:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  x: Annotated[uint32_t, 16]
  y: Annotated[uint32_t, 20]
  width: Annotated[uint32_t, 24]
  height: Annotated[uint32_t, 28]
@record
class struct_WGPURenderPassMaxDrawCount:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  maxDrawCount: Annotated[uint64_t, 16]
@record
class struct_WGPURenderPassTimestampWrites:
  SIZE = 16
  querySet: Annotated[WGPUQuerySet, 0]
  beginningOfPassWriteIndex: Annotated[uint32_t, 8]
  endOfPassWriteIndex: Annotated[uint32_t, 12]
@record
class struct_WGPURequestAdapterCallbackInfo:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPURequestAdapterCallback, 16]
  userdata: Annotated[POINTER(None), 24]
enum_WGPURequestAdapterStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPURequestAdapterStatus_Success = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Success', 1)
WGPURequestAdapterStatus_InstanceDropped = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_InstanceDropped', 2)
WGPURequestAdapterStatus_Unavailable = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Unavailable', 3)
WGPURequestAdapterStatus_Error = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Error', 4)
WGPURequestAdapterStatus_Unknown = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Unknown', 5)
WGPURequestAdapterStatus_Force32 = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Force32', 2147483647)

WGPURequestAdapterCallback: TypeAlias = CFUNCTYPE(None, enum_WGPURequestAdapterStatus, POINTER(struct_WGPUAdapterImpl), struct_WGPUStringView, POINTER(None))
@record
class struct_WGPURequestAdapterOptions:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  compatibleSurface: Annotated[WGPUSurface, 8]
  featureLevel: Annotated[WGPUFeatureLevel, 16]
  powerPreference: Annotated[WGPUPowerPreference, 20]
  backendType: Annotated[WGPUBackendType, 24]
  forceFallbackAdapter: Annotated[WGPUBool, 28]
  compatibilityMode: Annotated[WGPUBool, 32]
enum_WGPUFeatureLevel = CEnum(Annotated[int, ctypes.c_uint32])
WGPUFeatureLevel_Undefined = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Undefined', 0)
WGPUFeatureLevel_Compatibility = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Compatibility', 1)
WGPUFeatureLevel_Core = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Core', 2)
WGPUFeatureLevel_Force32 = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Force32', 2147483647)

WGPUFeatureLevel: TypeAlias = enum_WGPUFeatureLevel
enum_WGPUBackendType = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUBackendType: TypeAlias = enum_WGPUBackendType
@record
class struct_WGPURequestDeviceCallbackInfo:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPURequestDeviceCallback, 16]
  userdata: Annotated[POINTER(None), 24]
enum_WGPURequestDeviceStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPURequestDeviceStatus_Success = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Success', 1)
WGPURequestDeviceStatus_InstanceDropped = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_InstanceDropped', 2)
WGPURequestDeviceStatus_Error = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Error', 3)
WGPURequestDeviceStatus_Unknown = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Unknown', 4)
WGPURequestDeviceStatus_Force32 = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Force32', 2147483647)

WGPURequestDeviceCallback: TypeAlias = CFUNCTYPE(None, enum_WGPURequestDeviceStatus, POINTER(struct_WGPUDeviceImpl), struct_WGPUStringView, POINTER(None))
@record
class struct_WGPUSamplerBindingLayout:
  SIZE = 16
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  type: Annotated[WGPUSamplerBindingType, 8]
enum_WGPUSamplerBindingType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUSamplerBindingType_BindingNotUsed = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_BindingNotUsed', 0)
WGPUSamplerBindingType_Filtering = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Filtering', 1)
WGPUSamplerBindingType_NonFiltering = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_NonFiltering', 2)
WGPUSamplerBindingType_Comparison = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Comparison', 3)
WGPUSamplerBindingType_Force32 = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Force32', 2147483647)

WGPUSamplerBindingType: TypeAlias = enum_WGPUSamplerBindingType
@record
class struct_WGPUShaderModuleCompilationOptions:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  strictMath: Annotated[WGPUBool, 16]
@record
class struct_WGPUShaderSourceSPIRV:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  codeSize: Annotated[uint32_t, 16]
  code: Annotated[POINTER(uint32_t), 24]
@record
class struct_WGPUSharedBufferMemoryBeginAccessDescriptor:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  initialized: Annotated[WGPUBool, 8]
  fenceCount: Annotated[size_t, 16]
  fences: Annotated[POINTER(WGPUSharedFence), 24]
  signaledValues: Annotated[POINTER(uint64_t), 32]
@record
class struct_WGPUSharedBufferMemoryEndAccessState:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
  initialized: Annotated[WGPUBool, 8]
  fenceCount: Annotated[size_t, 16]
  fences: Annotated[POINTER(WGPUSharedFence), 24]
  signaledValues: Annotated[POINTER(uint64_t), 32]
@record
class struct_WGPUSharedBufferMemoryProperties:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
  usage: Annotated[WGPUBufferUsage, 8]
  size: Annotated[uint64_t, 16]
WGPUBufferUsage: TypeAlias = Annotated[int, ctypes.c_uint64]
@record
class struct_WGPUSharedFenceDXGISharedHandleDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[POINTER(None), 16]
@record
class struct_WGPUSharedFenceDXGISharedHandleExportInfo:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  handle: Annotated[POINTER(None), 16]
@record
class struct_WGPUSharedFenceMTLSharedEventDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  sharedEvent: Annotated[POINTER(None), 16]
@record
class struct_WGPUSharedFenceMTLSharedEventExportInfo:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  sharedEvent: Annotated[POINTER(None), 16]
@record
class struct_WGPUSharedFenceExportInfo:
  SIZE = 16
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
  type: Annotated[WGPUSharedFenceType, 8]
enum_WGPUSharedFenceType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUSharedFenceType_VkSemaphoreOpaqueFD = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_VkSemaphoreOpaqueFD', 1)
WGPUSharedFenceType_SyncFD = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_SyncFD', 2)
WGPUSharedFenceType_VkSemaphoreZirconHandle = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_VkSemaphoreZirconHandle', 3)
WGPUSharedFenceType_DXGISharedHandle = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_DXGISharedHandle', 4)
WGPUSharedFenceType_MTLSharedEvent = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_MTLSharedEvent', 5)
WGPUSharedFenceType_Force32 = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_Force32', 2147483647)

WGPUSharedFenceType: TypeAlias = enum_WGPUSharedFenceType
@record
class struct_WGPUSharedFenceSyncFDDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[Annotated[int, ctypes.c_int32], 16]
@record
class struct_WGPUSharedFenceSyncFDExportInfo:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  handle: Annotated[Annotated[int, ctypes.c_int32], 16]
@record
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[Annotated[int, ctypes.c_int32], 16]
@record
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  handle: Annotated[Annotated[int, ctypes.c_int32], 16]
@record
class struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[uint32_t, 16]
@record
class struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  handle: Annotated[uint32_t, 16]
@record
class struct_WGPUSharedTextureMemoryD3DSwapchainBeginState:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  isSwapchain: Annotated[WGPUBool, 16]
@record
class struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[POINTER(None), 16]
  useKeyedMutex: Annotated[WGPUBool, 24]
@record
class struct_WGPUSharedTextureMemoryEGLImageDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  image: Annotated[POINTER(None), 16]
@record
class struct_WGPUSharedTextureMemoryIOSurfaceDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  ioSurface: Annotated[POINTER(None), 16]
@record
class struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[POINTER(None), 16]
  useExternalFormat: Annotated[WGPUBool, 24]
@record
class struct_WGPUSharedTextureMemoryBeginAccessDescriptor:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  concurrentRead: Annotated[WGPUBool, 8]
  initialized: Annotated[WGPUBool, 12]
  fenceCount: Annotated[size_t, 16]
  fences: Annotated[POINTER(WGPUSharedFence), 24]
  signaledValues: Annotated[POINTER(uint64_t), 32]
@record
class struct_WGPUSharedTextureMemoryDmaBufPlane:
  SIZE = 24
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  offset: Annotated[uint64_t, 8]
  stride: Annotated[uint32_t, 16]
@record
class struct_WGPUSharedTextureMemoryEndAccessState:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
  initialized: Annotated[WGPUBool, 8]
  fenceCount: Annotated[size_t, 16]
  fences: Annotated[POINTER(WGPUSharedFence), 24]
  signaledValues: Annotated[POINTER(uint64_t), 32]
@record
class struct_WGPUSharedTextureMemoryOpaqueFDDescriptor:
  SIZE = 48
  chain: Annotated[WGPUChainedStruct, 0]
  vkImageCreateInfo: Annotated[POINTER(None), 16]
  memoryFD: Annotated[Annotated[int, ctypes.c_int32], 24]
  memoryTypeIndex: Annotated[uint32_t, 28]
  allocationSize: Annotated[uint64_t, 32]
  dedicatedAllocation: Annotated[WGPUBool, 40]
@record
class struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  dedicatedAllocation: Annotated[WGPUBool, 16]
@record
class struct_WGPUSharedTextureMemoryVkImageLayoutBeginState:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  oldLayout: Annotated[int32_t, 16]
  newLayout: Annotated[int32_t, 20]
int32_t: TypeAlias = Annotated[int, ctypes.c_int32]
@record
class struct_WGPUSharedTextureMemoryVkImageLayoutEndState:
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  oldLayout: Annotated[int32_t, 16]
  newLayout: Annotated[int32_t, 20]
@record
class struct_WGPUSharedTextureMemoryZirconHandleDescriptor:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  memoryFD: Annotated[uint32_t, 16]
  allocationSize: Annotated[uint64_t, 24]
@record
class struct_WGPUStaticSamplerBindingLayout:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  sampler: Annotated[WGPUSampler, 16]
  sampledTextureBinding: Annotated[uint32_t, 24]
@record
class struct_WGPUStencilFaceState:
  SIZE = 16
  compare: Annotated[WGPUCompareFunction, 0]
  failOp: Annotated[WGPUStencilOperation, 4]
  depthFailOp: Annotated[WGPUStencilOperation, 8]
  passOp: Annotated[WGPUStencilOperation, 12]
enum_WGPUCompareFunction = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUCompareFunction: TypeAlias = enum_WGPUCompareFunction
enum_WGPUStencilOperation = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUStencilOperation: TypeAlias = enum_WGPUStencilOperation
@record
class struct_WGPUStorageTextureBindingLayout:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  access: Annotated[WGPUStorageTextureAccess, 8]
  format: Annotated[WGPUTextureFormat, 12]
  viewDimension: Annotated[WGPUTextureViewDimension, 16]
enum_WGPUStorageTextureAccess = CEnum(Annotated[int, ctypes.c_uint32])
WGPUStorageTextureAccess_BindingNotUsed = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_BindingNotUsed', 0)
WGPUStorageTextureAccess_WriteOnly = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_WriteOnly', 1)
WGPUStorageTextureAccess_ReadOnly = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_ReadOnly', 2)
WGPUStorageTextureAccess_ReadWrite = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_ReadWrite', 3)
WGPUStorageTextureAccess_Force32 = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_Force32', 2147483647)

WGPUStorageTextureAccess: TypeAlias = enum_WGPUStorageTextureAccess
enum_WGPUTextureViewDimension = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureViewDimension_Undefined = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Undefined', 0)
WGPUTextureViewDimension_1D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_1D', 1)
WGPUTextureViewDimension_2D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_2D', 2)
WGPUTextureViewDimension_2DArray = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_2DArray', 3)
WGPUTextureViewDimension_Cube = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Cube', 4)
WGPUTextureViewDimension_CubeArray = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_CubeArray', 5)
WGPUTextureViewDimension_3D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_3D', 6)
WGPUTextureViewDimension_Force32 = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Force32', 2147483647)

WGPUTextureViewDimension: TypeAlias = enum_WGPUTextureViewDimension
@record
class struct_WGPUSupportedFeatures:
  SIZE = 16
  featureCount: Annotated[size_t, 0]
  features: Annotated[POINTER(WGPUFeatureName), 8]
enum_WGPUFeatureName = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUFeatureName: TypeAlias = enum_WGPUFeatureName
@record
class struct_WGPUSurfaceCapabilities:
  SIZE = 64
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
  usages: Annotated[WGPUTextureUsage, 8]
  formatCount: Annotated[size_t, 16]
  formats: Annotated[POINTER(WGPUTextureFormat), 24]
  presentModeCount: Annotated[size_t, 32]
  presentModes: Annotated[POINTER(WGPUPresentMode), 40]
  alphaModeCount: Annotated[size_t, 48]
  alphaModes: Annotated[POINTER(WGPUCompositeAlphaMode), 56]
enum_WGPUPresentMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUPresentMode_Fifo = enum_WGPUPresentMode.define('WGPUPresentMode_Fifo', 1)
WGPUPresentMode_FifoRelaxed = enum_WGPUPresentMode.define('WGPUPresentMode_FifoRelaxed', 2)
WGPUPresentMode_Immediate = enum_WGPUPresentMode.define('WGPUPresentMode_Immediate', 3)
WGPUPresentMode_Mailbox = enum_WGPUPresentMode.define('WGPUPresentMode_Mailbox', 4)
WGPUPresentMode_Force32 = enum_WGPUPresentMode.define('WGPUPresentMode_Force32', 2147483647)

WGPUPresentMode: TypeAlias = enum_WGPUPresentMode
enum_WGPUCompositeAlphaMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCompositeAlphaMode_Auto = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Auto', 0)
WGPUCompositeAlphaMode_Opaque = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Opaque', 1)
WGPUCompositeAlphaMode_Premultiplied = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Premultiplied', 2)
WGPUCompositeAlphaMode_Unpremultiplied = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Unpremultiplied', 3)
WGPUCompositeAlphaMode_Inherit = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Inherit', 4)
WGPUCompositeAlphaMode_Force32 = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Force32', 2147483647)

WGPUCompositeAlphaMode: TypeAlias = enum_WGPUCompositeAlphaMode
@record
class struct_WGPUSurfaceConfiguration:
  SIZE = 64
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  device: Annotated[WGPUDevice, 8]
  format: Annotated[WGPUTextureFormat, 16]
  usage: Annotated[WGPUTextureUsage, 24]
  viewFormatCount: Annotated[size_t, 32]
  viewFormats: Annotated[POINTER(WGPUTextureFormat), 40]
  alphaMode: Annotated[WGPUCompositeAlphaMode, 48]
  width: Annotated[uint32_t, 52]
  height: Annotated[uint32_t, 56]
  presentMode: Annotated[WGPUPresentMode, 60]
@record
class struct_WGPUSurfaceDescriptorFromWindowsCoreWindow:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  coreWindow: Annotated[POINTER(None), 16]
@record
class struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  swapChainPanel: Annotated[POINTER(None), 16]
@record
class struct_WGPUSurfaceSourceXCBWindow:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  connection: Annotated[POINTER(None), 16]
  window: Annotated[uint32_t, 24]
@record
class struct_WGPUSurfaceSourceAndroidNativeWindow:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  window: Annotated[POINTER(None), 16]
@record
class struct_WGPUSurfaceSourceMetalLayer:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  layer: Annotated[POINTER(None), 16]
@record
class struct_WGPUSurfaceSourceWaylandSurface:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  display: Annotated[POINTER(None), 16]
  surface: Annotated[POINTER(None), 24]
@record
class struct_WGPUSurfaceSourceWindowsHWND:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  hinstance: Annotated[POINTER(None), 16]
  hwnd: Annotated[POINTER(None), 24]
@record
class struct_WGPUSurfaceSourceXlibWindow:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  display: Annotated[POINTER(None), 16]
  window: Annotated[uint64_t, 24]
@record
class struct_WGPUSurfaceTexture:
  SIZE = 16
  texture: Annotated[WGPUTexture, 0]
  suboptimal: Annotated[WGPUBool, 8]
  status: Annotated[WGPUSurfaceGetCurrentTextureStatus, 12]
enum_WGPUSurfaceGetCurrentTextureStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUSurfaceGetCurrentTextureStatus_Success = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Success', 1)
WGPUSurfaceGetCurrentTextureStatus_Timeout = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Timeout', 2)
WGPUSurfaceGetCurrentTextureStatus_Outdated = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Outdated', 3)
WGPUSurfaceGetCurrentTextureStatus_Lost = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Lost', 4)
WGPUSurfaceGetCurrentTextureStatus_OutOfMemory = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_OutOfMemory', 5)
WGPUSurfaceGetCurrentTextureStatus_DeviceLost = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_DeviceLost', 6)
WGPUSurfaceGetCurrentTextureStatus_Error = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Error', 7)
WGPUSurfaceGetCurrentTextureStatus_Force32 = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Force32', 2147483647)

WGPUSurfaceGetCurrentTextureStatus: TypeAlias = enum_WGPUSurfaceGetCurrentTextureStatus
@record
class struct_WGPUTextureBindingLayout:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  sampleType: Annotated[WGPUTextureSampleType, 8]
  viewDimension: Annotated[WGPUTextureViewDimension, 12]
  multisampled: Annotated[WGPUBool, 16]
enum_WGPUTextureSampleType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureSampleType_BindingNotUsed = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_BindingNotUsed', 0)
WGPUTextureSampleType_Float = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Float', 1)
WGPUTextureSampleType_UnfilterableFloat = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_UnfilterableFloat', 2)
WGPUTextureSampleType_Depth = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Depth', 3)
WGPUTextureSampleType_Sint = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Sint', 4)
WGPUTextureSampleType_Uint = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Uint', 5)
WGPUTextureSampleType_Force32 = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Force32', 2147483647)

WGPUTextureSampleType: TypeAlias = enum_WGPUTextureSampleType
@record
class struct_WGPUTextureBindingViewDimensionDescriptor:
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  textureBindingViewDimension: Annotated[WGPUTextureViewDimension, 16]
@record
class struct_WGPUTextureDataLayout:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  offset: Annotated[uint64_t, 8]
  bytesPerRow: Annotated[uint32_t, 16]
  rowsPerImage: Annotated[uint32_t, 20]
@record
class struct_WGPUUncapturedErrorCallbackInfo:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  callback: Annotated[WGPUErrorCallback, 8]
  userdata: Annotated[POINTER(None), 16]
@record
class struct_WGPUVertexAttribute:
  SIZE = 24
  format: Annotated[WGPUVertexFormat, 0]
  offset: Annotated[uint64_t, 8]
  shaderLocation: Annotated[uint32_t, 16]
enum_WGPUVertexFormat = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUVertexFormat: TypeAlias = enum_WGPUVertexFormat
@record
class struct_WGPUYCbCrVkDescriptor:
  SIZE = 72
  chain: Annotated[WGPUChainedStruct, 0]
  vkFormat: Annotated[uint32_t, 16]
  vkYCbCrModel: Annotated[uint32_t, 20]
  vkYCbCrRange: Annotated[uint32_t, 24]
  vkComponentSwizzleRed: Annotated[uint32_t, 28]
  vkComponentSwizzleGreen: Annotated[uint32_t, 32]
  vkComponentSwizzleBlue: Annotated[uint32_t, 36]
  vkComponentSwizzleAlpha: Annotated[uint32_t, 40]
  vkXChromaOffset: Annotated[uint32_t, 44]
  vkYChromaOffset: Annotated[uint32_t, 48]
  vkChromaFilter: Annotated[WGPUFilterMode, 52]
  forceExplicitReconstruction: Annotated[WGPUBool, 56]
  externalFormat: Annotated[uint64_t, 64]
enum_WGPUFilterMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUFilterMode_Undefined = enum_WGPUFilterMode.define('WGPUFilterMode_Undefined', 0)
WGPUFilterMode_Nearest = enum_WGPUFilterMode.define('WGPUFilterMode_Nearest', 1)
WGPUFilterMode_Linear = enum_WGPUFilterMode.define('WGPUFilterMode_Linear', 2)
WGPUFilterMode_Force32 = enum_WGPUFilterMode.define('WGPUFilterMode_Force32', 2147483647)

WGPUFilterMode: TypeAlias = enum_WGPUFilterMode
@record
class struct_WGPUAHardwareBufferProperties:
  SIZE = 72
  yCbCrInfo: Annotated[WGPUYCbCrVkDescriptor, 0]
WGPUYCbCrVkDescriptor: TypeAlias = struct_WGPUYCbCrVkDescriptor
@record
class struct_WGPUAdapterInfo:
  SIZE = 96
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
  vendor: Annotated[WGPUStringView, 8]
  architecture: Annotated[WGPUStringView, 24]
  device: Annotated[WGPUStringView, 40]
  description: Annotated[WGPUStringView, 56]
  backendType: Annotated[WGPUBackendType, 72]
  adapterType: Annotated[WGPUAdapterType, 76]
  vendorID: Annotated[uint32_t, 80]
  deviceID: Annotated[uint32_t, 84]
  compatibilityMode: Annotated[WGPUBool, 88]
enum_WGPUAdapterType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUAdapterType_DiscreteGPU = enum_WGPUAdapterType.define('WGPUAdapterType_DiscreteGPU', 1)
WGPUAdapterType_IntegratedGPU = enum_WGPUAdapterType.define('WGPUAdapterType_IntegratedGPU', 2)
WGPUAdapterType_CPU = enum_WGPUAdapterType.define('WGPUAdapterType_CPU', 3)
WGPUAdapterType_Unknown = enum_WGPUAdapterType.define('WGPUAdapterType_Unknown', 4)
WGPUAdapterType_Force32 = enum_WGPUAdapterType.define('WGPUAdapterType_Force32', 2147483647)

WGPUAdapterType: TypeAlias = enum_WGPUAdapterType
@record
class struct_WGPUAdapterPropertiesMemoryHeaps:
  SIZE = 32
  chain: Annotated[WGPUChainedStructOut, 0]
  heapCount: Annotated[size_t, 16]
  heapInfo: Annotated[POINTER(WGPUMemoryHeapInfo), 24]
WGPUMemoryHeapInfo: TypeAlias = struct_WGPUMemoryHeapInfo
@record
class struct_WGPUBindGroupDescriptor:
  SIZE = 48
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  layout: Annotated[WGPUBindGroupLayout, 24]
  entryCount: Annotated[size_t, 32]
  entries: Annotated[POINTER(WGPUBindGroupEntry), 40]
WGPUBindGroupEntry: TypeAlias = struct_WGPUBindGroupEntry
@record
class struct_WGPUBindGroupLayoutEntry:
  SIZE = 112
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  binding: Annotated[uint32_t, 8]
  visibility: Annotated[WGPUShaderStage, 16]
  buffer: Annotated[WGPUBufferBindingLayout, 24]
  sampler: Annotated[WGPUSamplerBindingLayout, 48]
  texture: Annotated[WGPUTextureBindingLayout, 64]
  storageTexture: Annotated[WGPUStorageTextureBindingLayout, 88]
WGPUShaderStage: TypeAlias = Annotated[int, ctypes.c_uint64]
WGPUBufferBindingLayout: TypeAlias = struct_WGPUBufferBindingLayout
WGPUSamplerBindingLayout: TypeAlias = struct_WGPUSamplerBindingLayout
WGPUTextureBindingLayout: TypeAlias = struct_WGPUTextureBindingLayout
WGPUStorageTextureBindingLayout: TypeAlias = struct_WGPUStorageTextureBindingLayout
@record
class struct_WGPUBlendState:
  SIZE = 24
  color: Annotated[WGPUBlendComponent, 0]
  alpha: Annotated[WGPUBlendComponent, 12]
WGPUBlendComponent: TypeAlias = struct_WGPUBlendComponent
@record
class struct_WGPUBufferDescriptor:
  SIZE = 48
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  usage: Annotated[WGPUBufferUsage, 24]
  size: Annotated[uint64_t, 32]
  mappedAtCreation: Annotated[WGPUBool, 40]
@record
class struct_WGPUCommandBufferDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPUCommandEncoderDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPUComputePassDescriptor:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  timestampWrites: Annotated[POINTER(WGPUComputePassTimestampWrites), 24]
WGPUComputePassTimestampWrites: TypeAlias = struct_WGPUComputePassTimestampWrites
@record
class struct_WGPUConstantEntry:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  key: Annotated[WGPUStringView, 8]
  value: Annotated[Annotated[float, ctypes.c_double], 24]
@record
class struct_WGPUDawnCacheDeviceDescriptor:
  SIZE = 56
  chain: Annotated[WGPUChainedStruct, 0]
  isolationKey: Annotated[WGPUStringView, 16]
  loadDataFunction: Annotated[WGPUDawnLoadCacheDataFunction, 32]
  storeDataFunction: Annotated[WGPUDawnStoreCacheDataFunction, 40]
  functionUserdata: Annotated[POINTER(None), 48]
WGPUDawnLoadCacheDataFunction: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint64], POINTER(None), Annotated[int, ctypes.c_uint64], POINTER(None), Annotated[int, ctypes.c_uint64], POINTER(None))
WGPUDawnStoreCacheDataFunction: TypeAlias = CFUNCTYPE(None, POINTER(None), Annotated[int, ctypes.c_uint64], POINTER(None), Annotated[int, ctypes.c_uint64], POINTER(None))
@record
class struct_WGPUDepthStencilState:
  SIZE = 72
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  format: Annotated[WGPUTextureFormat, 8]
  depthWriteEnabled: Annotated[WGPUOptionalBool, 12]
  depthCompare: Annotated[WGPUCompareFunction, 16]
  stencilFront: Annotated[WGPUStencilFaceState, 20]
  stencilBack: Annotated[WGPUStencilFaceState, 36]
  stencilReadMask: Annotated[uint32_t, 52]
  stencilWriteMask: Annotated[uint32_t, 56]
  depthBias: Annotated[int32_t, 60]
  depthBiasSlopeScale: Annotated[Annotated[float, ctypes.c_float], 64]
  depthBiasClamp: Annotated[Annotated[float, ctypes.c_float], 68]
enum_WGPUOptionalBool = CEnum(Annotated[int, ctypes.c_uint32])
WGPUOptionalBool_False = enum_WGPUOptionalBool.define('WGPUOptionalBool_False', 0)
WGPUOptionalBool_True = enum_WGPUOptionalBool.define('WGPUOptionalBool_True', 1)
WGPUOptionalBool_Undefined = enum_WGPUOptionalBool.define('WGPUOptionalBool_Undefined', 2)
WGPUOptionalBool_Force32 = enum_WGPUOptionalBool.define('WGPUOptionalBool_Force32', 2147483647)

WGPUOptionalBool: TypeAlias = enum_WGPUOptionalBool
WGPUStencilFaceState: TypeAlias = struct_WGPUStencilFaceState
@record
class struct_WGPUDrmFormatCapabilities:
  SIZE = 32
  chain: Annotated[WGPUChainedStructOut, 0]
  propertiesCount: Annotated[size_t, 16]
  properties: Annotated[POINTER(WGPUDrmFormatProperties), 24]
WGPUDrmFormatProperties: TypeAlias = struct_WGPUDrmFormatProperties
@record
class struct_WGPUExternalTextureDescriptor:
  SIZE = 112
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  plane0: Annotated[WGPUTextureView, 24]
  plane1: Annotated[WGPUTextureView, 32]
  cropOrigin: Annotated[WGPUOrigin2D, 40]
  cropSize: Annotated[WGPUExtent2D, 48]
  apparentSize: Annotated[WGPUExtent2D, 56]
  doYuvToRgbConversionOnly: Annotated[WGPUBool, 64]
  yuvToRgbConversionMatrix: Annotated[POINTER(Annotated[float, ctypes.c_float]), 72]
  srcTransferFunctionParameters: Annotated[POINTER(Annotated[float, ctypes.c_float]), 80]
  dstTransferFunctionParameters: Annotated[POINTER(Annotated[float, ctypes.c_float]), 88]
  gamutConversionMatrix: Annotated[POINTER(Annotated[float, ctypes.c_float]), 96]
  mirrored: Annotated[WGPUBool, 104]
  rotation: Annotated[WGPUExternalTextureRotation, 108]
WGPUOrigin2D: TypeAlias = struct_WGPUOrigin2D
WGPUExtent2D: TypeAlias = struct_WGPUExtent2D
enum_WGPUExternalTextureRotation = CEnum(Annotated[int, ctypes.c_uint32])
WGPUExternalTextureRotation_Rotate0Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate0Degrees', 1)
WGPUExternalTextureRotation_Rotate90Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate90Degrees', 2)
WGPUExternalTextureRotation_Rotate180Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate180Degrees', 3)
WGPUExternalTextureRotation_Rotate270Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate270Degrees', 4)
WGPUExternalTextureRotation_Force32 = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Force32', 2147483647)

WGPUExternalTextureRotation: TypeAlias = enum_WGPUExternalTextureRotation
@record
class struct_WGPUFutureWaitInfo:
  SIZE = 16
  future: Annotated[WGPUFuture, 0]
  completed: Annotated[WGPUBool, 8]
WGPUFuture: TypeAlias = struct_WGPUFuture
@record
class struct_WGPUImageCopyBuffer:
  SIZE = 32
  layout: Annotated[WGPUTextureDataLayout, 0]
  buffer: Annotated[WGPUBuffer, 24]
WGPUTextureDataLayout: TypeAlias = struct_WGPUTextureDataLayout
@record
class struct_WGPUImageCopyExternalTexture:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  externalTexture: Annotated[WGPUExternalTexture, 8]
  origin: Annotated[WGPUOrigin3D, 16]
  naturalSize: Annotated[WGPUExtent2D, 28]
WGPUOrigin3D: TypeAlias = struct_WGPUOrigin3D
@record
class struct_WGPUImageCopyTexture:
  SIZE = 32
  texture: Annotated[WGPUTexture, 0]
  mipLevel: Annotated[uint32_t, 8]
  origin: Annotated[WGPUOrigin3D, 12]
  aspect: Annotated[WGPUTextureAspect, 24]
enum_WGPUTextureAspect = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureAspect_Undefined = enum_WGPUTextureAspect.define('WGPUTextureAspect_Undefined', 0)
WGPUTextureAspect_All = enum_WGPUTextureAspect.define('WGPUTextureAspect_All', 1)
WGPUTextureAspect_StencilOnly = enum_WGPUTextureAspect.define('WGPUTextureAspect_StencilOnly', 2)
WGPUTextureAspect_DepthOnly = enum_WGPUTextureAspect.define('WGPUTextureAspect_DepthOnly', 3)
WGPUTextureAspect_Plane0Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane0Only', 327680)
WGPUTextureAspect_Plane1Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane1Only', 327681)
WGPUTextureAspect_Plane2Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane2Only', 327682)
WGPUTextureAspect_Force32 = enum_WGPUTextureAspect.define('WGPUTextureAspect_Force32', 2147483647)

WGPUTextureAspect: TypeAlias = enum_WGPUTextureAspect
@record
class struct_WGPUInstanceDescriptor:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  features: Annotated[WGPUInstanceFeatures, 8]
WGPUInstanceFeatures: TypeAlias = struct_WGPUInstanceFeatures
@record
class struct_WGPUPipelineLayoutDescriptor:
  SIZE = 48
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  bindGroupLayoutCount: Annotated[size_t, 24]
  bindGroupLayouts: Annotated[POINTER(WGPUBindGroupLayout), 32]
  immediateDataRangeByteSize: Annotated[uint32_t, 40]
@record
class struct_WGPUPipelineLayoutPixelLocalStorage:
  SIZE = 40
  chain: Annotated[WGPUChainedStruct, 0]
  totalPixelLocalStorageSize: Annotated[uint64_t, 16]
  storageAttachmentCount: Annotated[size_t, 24]
  storageAttachments: Annotated[POINTER(WGPUPipelineLayoutStorageAttachment), 32]
WGPUPipelineLayoutStorageAttachment: TypeAlias = struct_WGPUPipelineLayoutStorageAttachment
@record
class struct_WGPUQuerySetDescriptor:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  type: Annotated[WGPUQueryType, 24]
  count: Annotated[uint32_t, 28]
enum_WGPUQueryType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUQueryType_Occlusion = enum_WGPUQueryType.define('WGPUQueryType_Occlusion', 1)
WGPUQueryType_Timestamp = enum_WGPUQueryType.define('WGPUQueryType_Timestamp', 2)
WGPUQueryType_Force32 = enum_WGPUQueryType.define('WGPUQueryType_Force32', 2147483647)

WGPUQueryType: TypeAlias = enum_WGPUQueryType
@record
class struct_WGPUQueueDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPURenderBundleDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPURenderBundleEncoderDescriptor:
  SIZE = 56
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  colorFormatCount: Annotated[size_t, 24]
  colorFormats: Annotated[POINTER(WGPUTextureFormat), 32]
  depthStencilFormat: Annotated[WGPUTextureFormat, 40]
  sampleCount: Annotated[uint32_t, 44]
  depthReadOnly: Annotated[WGPUBool, 48]
  stencilReadOnly: Annotated[WGPUBool, 52]
@record
class struct_WGPURenderPassColorAttachment:
  SIZE = 72
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  view: Annotated[WGPUTextureView, 8]
  depthSlice: Annotated[uint32_t, 16]
  resolveTarget: Annotated[WGPUTextureView, 24]
  loadOp: Annotated[WGPULoadOp, 32]
  storeOp: Annotated[WGPUStoreOp, 36]
  clearValue: Annotated[WGPUColor, 40]
WGPUColor: TypeAlias = struct_WGPUColor
@record
class struct_WGPURenderPassStorageAttachment:
  SIZE = 64
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  offset: Annotated[uint64_t, 8]
  storage: Annotated[WGPUTextureView, 16]
  loadOp: Annotated[WGPULoadOp, 24]
  storeOp: Annotated[WGPUStoreOp, 28]
  clearValue: Annotated[WGPUColor, 32]
@record
class struct_WGPURequiredLimits:
  SIZE = 168
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  limits: Annotated[WGPULimits, 8]
WGPULimits: TypeAlias = struct_WGPULimits
@record
class struct_WGPUSamplerDescriptor:
  SIZE = 64
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  addressModeU: Annotated[WGPUAddressMode, 24]
  addressModeV: Annotated[WGPUAddressMode, 28]
  addressModeW: Annotated[WGPUAddressMode, 32]
  magFilter: Annotated[WGPUFilterMode, 36]
  minFilter: Annotated[WGPUFilterMode, 40]
  mipmapFilter: Annotated[WGPUMipmapFilterMode, 44]
  lodMinClamp: Annotated[Annotated[float, ctypes.c_float], 48]
  lodMaxClamp: Annotated[Annotated[float, ctypes.c_float], 52]
  compare: Annotated[WGPUCompareFunction, 56]
  maxAnisotropy: Annotated[uint16_t, 60]
enum_WGPUAddressMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUAddressMode_Undefined = enum_WGPUAddressMode.define('WGPUAddressMode_Undefined', 0)
WGPUAddressMode_ClampToEdge = enum_WGPUAddressMode.define('WGPUAddressMode_ClampToEdge', 1)
WGPUAddressMode_Repeat = enum_WGPUAddressMode.define('WGPUAddressMode_Repeat', 2)
WGPUAddressMode_MirrorRepeat = enum_WGPUAddressMode.define('WGPUAddressMode_MirrorRepeat', 3)
WGPUAddressMode_Force32 = enum_WGPUAddressMode.define('WGPUAddressMode_Force32', 2147483647)

WGPUAddressMode: TypeAlias = enum_WGPUAddressMode
enum_WGPUMipmapFilterMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUMipmapFilterMode_Undefined = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Undefined', 0)
WGPUMipmapFilterMode_Nearest = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Nearest', 1)
WGPUMipmapFilterMode_Linear = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Linear', 2)
WGPUMipmapFilterMode_Force32 = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Force32', 2147483647)

WGPUMipmapFilterMode: TypeAlias = enum_WGPUMipmapFilterMode
uint16_t: TypeAlias = Annotated[int, ctypes.c_uint16]
@record
class struct_WGPUShaderModuleDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPUShaderSourceWGSL:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  code: Annotated[WGPUStringView, 16]
@record
class struct_WGPUSharedBufferMemoryDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPUSharedFenceDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPUSharedTextureMemoryAHardwareBufferProperties:
  SIZE = 88
  chain: Annotated[WGPUChainedStructOut, 0]
  yCbCrInfo: Annotated[WGPUYCbCrVkDescriptor, 16]
@record
class struct_WGPUSharedTextureMemoryDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPUSharedTextureMemoryDmaBufDescriptor:
  SIZE = 56
  chain: Annotated[WGPUChainedStruct, 0]
  size: Annotated[WGPUExtent3D, 16]
  drmFormat: Annotated[uint32_t, 28]
  drmModifier: Annotated[uint64_t, 32]
  planeCount: Annotated[size_t, 40]
  planes: Annotated[POINTER(WGPUSharedTextureMemoryDmaBufPlane), 48]
WGPUExtent3D: TypeAlias = struct_WGPUExtent3D
WGPUSharedTextureMemoryDmaBufPlane: TypeAlias = struct_WGPUSharedTextureMemoryDmaBufPlane
@record
class struct_WGPUSharedTextureMemoryProperties:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
  usage: Annotated[WGPUTextureUsage, 8]
  size: Annotated[WGPUExtent3D, 16]
  format: Annotated[WGPUTextureFormat, 28]
@record
class struct_WGPUSupportedLimits:
  SIZE = 168
  nextInChain: Annotated[POINTER(WGPUChainedStructOut), 0]
  limits: Annotated[WGPULimits, 8]
@record
class struct_WGPUSurfaceDescriptor:
  SIZE = 24
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
@record
class struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten:
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  selector: Annotated[WGPUStringView, 16]
@record
class struct_WGPUTextureDescriptor:
  SIZE = 80
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  usage: Annotated[WGPUTextureUsage, 24]
  dimension: Annotated[WGPUTextureDimension, 32]
  size: Annotated[WGPUExtent3D, 36]
  format: Annotated[WGPUTextureFormat, 48]
  mipLevelCount: Annotated[uint32_t, 52]
  sampleCount: Annotated[uint32_t, 56]
  viewFormatCount: Annotated[size_t, 64]
  viewFormats: Annotated[POINTER(WGPUTextureFormat), 72]
enum_WGPUTextureDimension = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureDimension_Undefined = enum_WGPUTextureDimension.define('WGPUTextureDimension_Undefined', 0)
WGPUTextureDimension_1D = enum_WGPUTextureDimension.define('WGPUTextureDimension_1D', 1)
WGPUTextureDimension_2D = enum_WGPUTextureDimension.define('WGPUTextureDimension_2D', 2)
WGPUTextureDimension_3D = enum_WGPUTextureDimension.define('WGPUTextureDimension_3D', 3)
WGPUTextureDimension_Force32 = enum_WGPUTextureDimension.define('WGPUTextureDimension_Force32', 2147483647)

WGPUTextureDimension: TypeAlias = enum_WGPUTextureDimension
@record
class struct_WGPUTextureViewDescriptor:
  SIZE = 64
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  format: Annotated[WGPUTextureFormat, 24]
  dimension: Annotated[WGPUTextureViewDimension, 28]
  baseMipLevel: Annotated[uint32_t, 32]
  mipLevelCount: Annotated[uint32_t, 36]
  baseArrayLayer: Annotated[uint32_t, 40]
  arrayLayerCount: Annotated[uint32_t, 44]
  aspect: Annotated[WGPUTextureAspect, 48]
  usage: Annotated[WGPUTextureUsage, 56]
@record
class struct_WGPUVertexBufferLayout:
  SIZE = 32
  arrayStride: Annotated[uint64_t, 0]
  stepMode: Annotated[WGPUVertexStepMode, 8]
  attributeCount: Annotated[size_t, 16]
  attributes: Annotated[POINTER(WGPUVertexAttribute), 24]
enum_WGPUVertexStepMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUVertexStepMode_Undefined = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Undefined', 0)
WGPUVertexStepMode_Vertex = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Vertex', 1)
WGPUVertexStepMode_Instance = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Instance', 2)
WGPUVertexStepMode_Force32 = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Force32', 2147483647)

WGPUVertexStepMode: TypeAlias = enum_WGPUVertexStepMode
WGPUVertexAttribute: TypeAlias = struct_WGPUVertexAttribute
@record
class struct_WGPUBindGroupLayoutDescriptor:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  entryCount: Annotated[size_t, 24]
  entries: Annotated[POINTER(WGPUBindGroupLayoutEntry), 32]
WGPUBindGroupLayoutEntry: TypeAlias = struct_WGPUBindGroupLayoutEntry
@record
class struct_WGPUColorTargetState:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  format: Annotated[WGPUTextureFormat, 8]
  blend: Annotated[POINTER(WGPUBlendState), 16]
  writeMask: Annotated[WGPUColorWriteMask, 24]
WGPUBlendState: TypeAlias = struct_WGPUBlendState
WGPUColorWriteMask: TypeAlias = Annotated[int, ctypes.c_uint64]
@record
class struct_WGPUComputeState:
  SIZE = 48
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  module: Annotated[WGPUShaderModule, 8]
  entryPoint: Annotated[WGPUStringView, 16]
  constantCount: Annotated[size_t, 32]
  constants: Annotated[POINTER(WGPUConstantEntry), 40]
WGPUConstantEntry: TypeAlias = struct_WGPUConstantEntry
@record
class struct_WGPUDeviceDescriptor:
  SIZE = 144
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  requiredFeatureCount: Annotated[size_t, 24]
  requiredFeatures: Annotated[POINTER(WGPUFeatureName), 32]
  requiredLimits: Annotated[POINTER(WGPURequiredLimits), 40]
  defaultQueue: Annotated[WGPUQueueDescriptor, 48]
  deviceLostCallbackInfo2: Annotated[WGPUDeviceLostCallbackInfo2, 72]
  uncapturedErrorCallbackInfo2: Annotated[WGPUUncapturedErrorCallbackInfo2, 112]
WGPURequiredLimits: TypeAlias = struct_WGPURequiredLimits
WGPUQueueDescriptor: TypeAlias = struct_WGPUQueueDescriptor
@record
class struct_WGPUDeviceLostCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUDeviceLostCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPUDeviceLostCallbackInfo2: TypeAlias = struct_WGPUDeviceLostCallbackInfo2
WGPUDeviceLostCallback2: TypeAlias = CFUNCTYPE(None, POINTER(POINTER(struct_WGPUDeviceImpl)), enum_WGPUDeviceLostReason, struct_WGPUStringView, POINTER(None), POINTER(None))
@record
class struct_WGPUUncapturedErrorCallbackInfo2:
  SIZE = 32
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  callback: Annotated[WGPUUncapturedErrorCallback, 8]
  userdata1: Annotated[POINTER(None), 16]
  userdata2: Annotated[POINTER(None), 24]
WGPUUncapturedErrorCallbackInfo2: TypeAlias = struct_WGPUUncapturedErrorCallbackInfo2
WGPUUncapturedErrorCallback: TypeAlias = CFUNCTYPE(None, POINTER(POINTER(struct_WGPUDeviceImpl)), enum_WGPUErrorType, struct_WGPUStringView, POINTER(None), POINTER(None))
@record
class struct_WGPURenderPassDescriptor:
  SIZE = 64
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  colorAttachmentCount: Annotated[size_t, 24]
  colorAttachments: Annotated[POINTER(WGPURenderPassColorAttachment), 32]
  depthStencilAttachment: Annotated[POINTER(WGPURenderPassDepthStencilAttachment), 40]
  occlusionQuerySet: Annotated[WGPUQuerySet, 48]
  timestampWrites: Annotated[POINTER(WGPURenderPassTimestampWrites), 56]
WGPURenderPassColorAttachment: TypeAlias = struct_WGPURenderPassColorAttachment
WGPURenderPassDepthStencilAttachment: TypeAlias = struct_WGPURenderPassDepthStencilAttachment
WGPURenderPassTimestampWrites: TypeAlias = struct_WGPURenderPassTimestampWrites
@record
class struct_WGPURenderPassPixelLocalStorage:
  SIZE = 40
  chain: Annotated[WGPUChainedStruct, 0]
  totalPixelLocalStorageSize: Annotated[uint64_t, 16]
  storageAttachmentCount: Annotated[size_t, 24]
  storageAttachments: Annotated[POINTER(WGPURenderPassStorageAttachment), 32]
WGPURenderPassStorageAttachment: TypeAlias = struct_WGPURenderPassStorageAttachment
@record
class struct_WGPUVertexState:
  SIZE = 64
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  module: Annotated[WGPUShaderModule, 8]
  entryPoint: Annotated[WGPUStringView, 16]
  constantCount: Annotated[size_t, 32]
  constants: Annotated[POINTER(WGPUConstantEntry), 40]
  bufferCount: Annotated[size_t, 48]
  buffers: Annotated[POINTER(WGPUVertexBufferLayout), 56]
WGPUVertexBufferLayout: TypeAlias = struct_WGPUVertexBufferLayout
@record
class struct_WGPUComputePipelineDescriptor:
  SIZE = 80
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  layout: Annotated[WGPUPipelineLayout, 24]
  compute: Annotated[WGPUComputeState, 32]
WGPUComputeState: TypeAlias = struct_WGPUComputeState
@record
class struct_WGPUFragmentState:
  SIZE = 64
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  module: Annotated[WGPUShaderModule, 8]
  entryPoint: Annotated[WGPUStringView, 16]
  constantCount: Annotated[size_t, 32]
  constants: Annotated[POINTER(WGPUConstantEntry), 40]
  targetCount: Annotated[size_t, 48]
  targets: Annotated[POINTER(WGPUColorTargetState), 56]
WGPUColorTargetState: TypeAlias = struct_WGPUColorTargetState
@record
class struct_WGPURenderPipelineDescriptor:
  SIZE = 168
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  label: Annotated[WGPUStringView, 8]
  layout: Annotated[WGPUPipelineLayout, 24]
  vertex: Annotated[WGPUVertexState, 32]
  primitive: Annotated[WGPUPrimitiveState, 96]
  depthStencil: Annotated[POINTER(WGPUDepthStencilState), 128]
  multisample: Annotated[WGPUMultisampleState, 136]
  fragment: Annotated[POINTER(WGPUFragmentState), 160]
WGPUVertexState: TypeAlias = struct_WGPUVertexState
WGPUPrimitiveState: TypeAlias = struct_WGPUPrimitiveState
WGPUDepthStencilState: TypeAlias = struct_WGPUDepthStencilState
WGPUMultisampleState: TypeAlias = struct_WGPUMultisampleState
WGPUFragmentState: TypeAlias = struct_WGPUFragmentState
enum_WGPUWGSLFeatureName = CEnum(Annotated[int, ctypes.c_uint32])
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

WGPUWGSLFeatureName: TypeAlias = enum_WGPUWGSLFeatureName
WGPUBufferMapAsyncStatus: TypeAlias = enum_WGPUBufferMapAsyncStatus
enum_WGPUBufferMapState = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBufferMapState_Unmapped = enum_WGPUBufferMapState.define('WGPUBufferMapState_Unmapped', 1)
WGPUBufferMapState_Pending = enum_WGPUBufferMapState.define('WGPUBufferMapState_Pending', 2)
WGPUBufferMapState_Mapped = enum_WGPUBufferMapState.define('WGPUBufferMapState_Mapped', 3)
WGPUBufferMapState_Force32 = enum_WGPUBufferMapState.define('WGPUBufferMapState_Force32', 2147483647)

WGPUBufferMapState: TypeAlias = enum_WGPUBufferMapState
WGPUCompilationInfoRequestStatus: TypeAlias = enum_WGPUCompilationInfoRequestStatus
WGPUCreatePipelineAsyncStatus: TypeAlias = enum_WGPUCreatePipelineAsyncStatus
WGPUDeviceLostReason: TypeAlias = enum_WGPUDeviceLostReason
enum_WGPUErrorFilter = CEnum(Annotated[int, ctypes.c_uint32])
WGPUErrorFilter_Validation = enum_WGPUErrorFilter.define('WGPUErrorFilter_Validation', 1)
WGPUErrorFilter_OutOfMemory = enum_WGPUErrorFilter.define('WGPUErrorFilter_OutOfMemory', 2)
WGPUErrorFilter_Internal = enum_WGPUErrorFilter.define('WGPUErrorFilter_Internal', 3)
WGPUErrorFilter_Force32 = enum_WGPUErrorFilter.define('WGPUErrorFilter_Force32', 2147483647)

WGPUErrorFilter: TypeAlias = enum_WGPUErrorFilter
WGPUErrorType: TypeAlias = enum_WGPUErrorType
enum_WGPULoggingType = CEnum(Annotated[int, ctypes.c_uint32])
WGPULoggingType_Verbose = enum_WGPULoggingType.define('WGPULoggingType_Verbose', 1)
WGPULoggingType_Info = enum_WGPULoggingType.define('WGPULoggingType_Info', 2)
WGPULoggingType_Warning = enum_WGPULoggingType.define('WGPULoggingType_Warning', 3)
WGPULoggingType_Error = enum_WGPULoggingType.define('WGPULoggingType_Error', 4)
WGPULoggingType_Force32 = enum_WGPULoggingType.define('WGPULoggingType_Force32', 2147483647)

WGPULoggingType: TypeAlias = enum_WGPULoggingType
enum_WGPUMapAsyncStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUMapAsyncStatus_Success = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Success', 1)
WGPUMapAsyncStatus_InstanceDropped = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_InstanceDropped', 2)
WGPUMapAsyncStatus_Error = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Error', 3)
WGPUMapAsyncStatus_Aborted = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Aborted', 4)
WGPUMapAsyncStatus_Unknown = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Unknown', 5)
WGPUMapAsyncStatus_Force32 = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Force32', 2147483647)

WGPUMapAsyncStatus: TypeAlias = enum_WGPUMapAsyncStatus
WGPUPopErrorScopeStatus: TypeAlias = enum_WGPUPopErrorScopeStatus
WGPUQueueWorkDoneStatus: TypeAlias = enum_WGPUQueueWorkDoneStatus
WGPURequestAdapterStatus: TypeAlias = enum_WGPURequestAdapterStatus
WGPURequestDeviceStatus: TypeAlias = enum_WGPURequestDeviceStatus
enum_WGPUStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUStatus_Success = enum_WGPUStatus.define('WGPUStatus_Success', 1)
WGPUStatus_Error = enum_WGPUStatus.define('WGPUStatus_Error', 2)
WGPUStatus_Force32 = enum_WGPUStatus.define('WGPUStatus_Force32', 2147483647)

WGPUStatus: TypeAlias = enum_WGPUStatus
enum_WGPUWaitStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUWaitStatus_Success = enum_WGPUWaitStatus.define('WGPUWaitStatus_Success', 1)
WGPUWaitStatus_TimedOut = enum_WGPUWaitStatus.define('WGPUWaitStatus_TimedOut', 2)
WGPUWaitStatus_UnsupportedTimeout = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedTimeout', 3)
WGPUWaitStatus_UnsupportedCount = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedCount', 4)
WGPUWaitStatus_UnsupportedMixedSources = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedMixedSources', 5)
WGPUWaitStatus_Unknown = enum_WGPUWaitStatus.define('WGPUWaitStatus_Unknown', 6)
WGPUWaitStatus_Force32 = enum_WGPUWaitStatus.define('WGPUWaitStatus_Force32', 2147483647)

WGPUWaitStatus: TypeAlias = enum_WGPUWaitStatus
WGPUMapMode: TypeAlias = Annotated[int, ctypes.c_uint64]
WGPUDeviceLostCallback: TypeAlias = CFUNCTYPE(None, enum_WGPUDeviceLostReason, struct_WGPUStringView, POINTER(None))
WGPULoggingCallback: TypeAlias = CFUNCTYPE(None, enum_WGPULoggingType, struct_WGPUStringView, POINTER(None))
WGPUProc: TypeAlias = CFUNCTYPE(None, )
WGPUBufferMapCallback2: TypeAlias = CFUNCTYPE(None, enum_WGPUMapAsyncStatus, struct_WGPUStringView, POINTER(None), POINTER(None))
WGPUCompilationInfoCallback2: TypeAlias = CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, POINTER(struct_WGPUCompilationInfo), POINTER(None), POINTER(None))
WGPUCreateComputePipelineAsyncCallback2: TypeAlias = CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, POINTER(struct_WGPUComputePipelineImpl), struct_WGPUStringView, POINTER(None), POINTER(None))
WGPUCreateRenderPipelineAsyncCallback2: TypeAlias = CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, POINTER(struct_WGPURenderPipelineImpl), struct_WGPUStringView, POINTER(None), POINTER(None))
WGPUPopErrorScopeCallback2: TypeAlias = CFUNCTYPE(None, enum_WGPUPopErrorScopeStatus, enum_WGPUErrorType, struct_WGPUStringView, POINTER(None), POINTER(None))
WGPUQueueWorkDoneCallback2: TypeAlias = CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, POINTER(None), POINTER(None))
WGPURequestAdapterCallback2: TypeAlias = CFUNCTYPE(None, enum_WGPURequestAdapterStatus, POINTER(struct_WGPUAdapterImpl), struct_WGPUStringView, POINTER(None), POINTER(None))
WGPURequestDeviceCallback2: TypeAlias = CFUNCTYPE(None, enum_WGPURequestDeviceStatus, POINTER(struct_WGPUDeviceImpl), struct_WGPUStringView, POINTER(None), POINTER(None))
@record
class struct_WGPUBufferMapCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUBufferMapCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPUBufferMapCallbackInfo2: TypeAlias = struct_WGPUBufferMapCallbackInfo2
@record
class struct_WGPUCompilationInfoCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCompilationInfoCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPUCompilationInfoCallbackInfo2: TypeAlias = struct_WGPUCompilationInfoCallbackInfo2
@record
class struct_WGPUCreateComputePipelineAsyncCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCreateComputePipelineAsyncCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPUCreateComputePipelineAsyncCallbackInfo2: TypeAlias = struct_WGPUCreateComputePipelineAsyncCallbackInfo2
@record
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCreateRenderPipelineAsyncCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPUCreateRenderPipelineAsyncCallbackInfo2: TypeAlias = struct_WGPUCreateRenderPipelineAsyncCallbackInfo2
@record
class struct_WGPUPopErrorScopeCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUPopErrorScopeCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPUPopErrorScopeCallbackInfo2: TypeAlias = struct_WGPUPopErrorScopeCallbackInfo2
@record
class struct_WGPUQueueWorkDoneCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUQueueWorkDoneCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPUQueueWorkDoneCallbackInfo2: TypeAlias = struct_WGPUQueueWorkDoneCallbackInfo2
@record
class struct_WGPURequestAdapterCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPURequestAdapterCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPURequestAdapterCallbackInfo2: TypeAlias = struct_WGPURequestAdapterCallbackInfo2
@record
class struct_WGPURequestDeviceCallbackInfo2:
  SIZE = 40
  nextInChain: Annotated[POINTER(WGPUChainedStruct), 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPURequestDeviceCallback2, 16]
  userdata1: Annotated[POINTER(None), 24]
  userdata2: Annotated[POINTER(None), 32]
WGPURequestDeviceCallbackInfo2: TypeAlias = struct_WGPURequestDeviceCallbackInfo2
WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER: TypeAlias = struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER
WGPUAdapterPropertiesD3D: TypeAlias = struct_WGPUAdapterPropertiesD3D
WGPUAdapterPropertiesSubgroups: TypeAlias = struct_WGPUAdapterPropertiesSubgroups
WGPUAdapterPropertiesVk: TypeAlias = struct_WGPUAdapterPropertiesVk
WGPUBufferHostMappedPointer: TypeAlias = struct_WGPUBufferHostMappedPointer
WGPUBufferMapCallbackInfo: TypeAlias = struct_WGPUBufferMapCallbackInfo
WGPUColorTargetStateExpandResolveTextureDawn: TypeAlias = struct_WGPUColorTargetStateExpandResolveTextureDawn
WGPUCompilationInfoCallbackInfo: TypeAlias = struct_WGPUCompilationInfoCallbackInfo
WGPUCopyTextureForBrowserOptions: TypeAlias = struct_WGPUCopyTextureForBrowserOptions
WGPUCreateComputePipelineAsyncCallbackInfo: TypeAlias = struct_WGPUCreateComputePipelineAsyncCallbackInfo
WGPUCreateRenderPipelineAsyncCallbackInfo: TypeAlias = struct_WGPUCreateRenderPipelineAsyncCallbackInfo
WGPUDawnWGSLBlocklist: TypeAlias = struct_WGPUDawnWGSLBlocklist
WGPUDawnAdapterPropertiesPowerPreference: TypeAlias = struct_WGPUDawnAdapterPropertiesPowerPreference
WGPUDawnBufferDescriptorErrorInfoFromWireClient: TypeAlias = struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient
WGPUDawnEncoderInternalUsageDescriptor: TypeAlias = struct_WGPUDawnEncoderInternalUsageDescriptor
WGPUDawnExperimentalImmediateDataLimits: TypeAlias = struct_WGPUDawnExperimentalImmediateDataLimits
WGPUDawnExperimentalSubgroupLimits: TypeAlias = struct_WGPUDawnExperimentalSubgroupLimits
WGPUDawnRenderPassColorAttachmentRenderToSingleSampled: TypeAlias = struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled
WGPUDawnShaderModuleSPIRVOptionsDescriptor: TypeAlias = struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor
WGPUDawnTexelCopyBufferRowAlignmentLimits: TypeAlias = struct_WGPUDawnTexelCopyBufferRowAlignmentLimits
WGPUDawnTextureInternalUsageDescriptor: TypeAlias = struct_WGPUDawnTextureInternalUsageDescriptor
WGPUDawnTogglesDescriptor: TypeAlias = struct_WGPUDawnTogglesDescriptor
WGPUDawnWireWGSLControl: TypeAlias = struct_WGPUDawnWireWGSLControl
WGPUDeviceLostCallbackInfo: TypeAlias = struct_WGPUDeviceLostCallbackInfo
WGPUExternalTextureBindingEntry: TypeAlias = struct_WGPUExternalTextureBindingEntry
WGPUExternalTextureBindingLayout: TypeAlias = struct_WGPUExternalTextureBindingLayout
WGPUFormatCapabilities: TypeAlias = struct_WGPUFormatCapabilities
WGPUPopErrorScopeCallbackInfo: TypeAlias = struct_WGPUPopErrorScopeCallbackInfo
WGPUQueueWorkDoneCallbackInfo: TypeAlias = struct_WGPUQueueWorkDoneCallbackInfo
WGPURenderPassDescriptorExpandResolveRect: TypeAlias = struct_WGPURenderPassDescriptorExpandResolveRect
WGPURenderPassMaxDrawCount: TypeAlias = struct_WGPURenderPassMaxDrawCount
WGPURequestAdapterCallbackInfo: TypeAlias = struct_WGPURequestAdapterCallbackInfo
WGPURequestAdapterOptions: TypeAlias = struct_WGPURequestAdapterOptions
WGPURequestDeviceCallbackInfo: TypeAlias = struct_WGPURequestDeviceCallbackInfo
WGPUShaderModuleCompilationOptions: TypeAlias = struct_WGPUShaderModuleCompilationOptions
WGPUShaderSourceSPIRV: TypeAlias = struct_WGPUShaderSourceSPIRV
WGPUSharedBufferMemoryBeginAccessDescriptor: TypeAlias = struct_WGPUSharedBufferMemoryBeginAccessDescriptor
WGPUSharedBufferMemoryEndAccessState: TypeAlias = struct_WGPUSharedBufferMemoryEndAccessState
WGPUSharedBufferMemoryProperties: TypeAlias = struct_WGPUSharedBufferMemoryProperties
WGPUSharedFenceDXGISharedHandleDescriptor: TypeAlias = struct_WGPUSharedFenceDXGISharedHandleDescriptor
WGPUSharedFenceDXGISharedHandleExportInfo: TypeAlias = struct_WGPUSharedFenceDXGISharedHandleExportInfo
WGPUSharedFenceMTLSharedEventDescriptor: TypeAlias = struct_WGPUSharedFenceMTLSharedEventDescriptor
WGPUSharedFenceMTLSharedEventExportInfo: TypeAlias = struct_WGPUSharedFenceMTLSharedEventExportInfo
WGPUSharedFenceExportInfo: TypeAlias = struct_WGPUSharedFenceExportInfo
WGPUSharedFenceSyncFDDescriptor: TypeAlias = struct_WGPUSharedFenceSyncFDDescriptor
WGPUSharedFenceSyncFDExportInfo: TypeAlias = struct_WGPUSharedFenceSyncFDExportInfo
WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor: TypeAlias = struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor
WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo: TypeAlias = struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo
WGPUSharedFenceVkSemaphoreZirconHandleDescriptor: TypeAlias = struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor
WGPUSharedFenceVkSemaphoreZirconHandleExportInfo: TypeAlias = struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo
WGPUSharedTextureMemoryD3DSwapchainBeginState: TypeAlias = struct_WGPUSharedTextureMemoryD3DSwapchainBeginState
WGPUSharedTextureMemoryDXGISharedHandleDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor
WGPUSharedTextureMemoryEGLImageDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryEGLImageDescriptor
WGPUSharedTextureMemoryIOSurfaceDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryIOSurfaceDescriptor
WGPUSharedTextureMemoryAHardwareBufferDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor
WGPUSharedTextureMemoryBeginAccessDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryBeginAccessDescriptor
WGPUSharedTextureMemoryEndAccessState: TypeAlias = struct_WGPUSharedTextureMemoryEndAccessState
WGPUSharedTextureMemoryOpaqueFDDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryOpaqueFDDescriptor
WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor
WGPUSharedTextureMemoryVkImageLayoutBeginState: TypeAlias = struct_WGPUSharedTextureMemoryVkImageLayoutBeginState
WGPUSharedTextureMemoryVkImageLayoutEndState: TypeAlias = struct_WGPUSharedTextureMemoryVkImageLayoutEndState
WGPUSharedTextureMemoryZirconHandleDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryZirconHandleDescriptor
WGPUStaticSamplerBindingLayout: TypeAlias = struct_WGPUStaticSamplerBindingLayout
WGPUSupportedFeatures: TypeAlias = struct_WGPUSupportedFeatures
WGPUSurfaceCapabilities: TypeAlias = struct_WGPUSurfaceCapabilities
WGPUSurfaceConfiguration: TypeAlias = struct_WGPUSurfaceConfiguration
WGPUSurfaceDescriptorFromWindowsCoreWindow: TypeAlias = struct_WGPUSurfaceDescriptorFromWindowsCoreWindow
WGPUSurfaceDescriptorFromWindowsSwapChainPanel: TypeAlias = struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel
WGPUSurfaceSourceXCBWindow: TypeAlias = struct_WGPUSurfaceSourceXCBWindow
WGPUSurfaceSourceAndroidNativeWindow: TypeAlias = struct_WGPUSurfaceSourceAndroidNativeWindow
WGPUSurfaceSourceMetalLayer: TypeAlias = struct_WGPUSurfaceSourceMetalLayer
WGPUSurfaceSourceWaylandSurface: TypeAlias = struct_WGPUSurfaceSourceWaylandSurface
WGPUSurfaceSourceWindowsHWND: TypeAlias = struct_WGPUSurfaceSourceWindowsHWND
WGPUSurfaceSourceXlibWindow: TypeAlias = struct_WGPUSurfaceSourceXlibWindow
WGPUSurfaceTexture: TypeAlias = struct_WGPUSurfaceTexture
WGPUTextureBindingViewDimensionDescriptor: TypeAlias = struct_WGPUTextureBindingViewDimensionDescriptor
WGPUUncapturedErrorCallbackInfo: TypeAlias = struct_WGPUUncapturedErrorCallbackInfo
WGPUAHardwareBufferProperties: TypeAlias = struct_WGPUAHardwareBufferProperties
WGPUAdapterInfo: TypeAlias = struct_WGPUAdapterInfo
WGPUAdapterPropertiesMemoryHeaps: TypeAlias = struct_WGPUAdapterPropertiesMemoryHeaps
WGPUBindGroupDescriptor: TypeAlias = struct_WGPUBindGroupDescriptor
WGPUBufferDescriptor: TypeAlias = struct_WGPUBufferDescriptor
WGPUCommandBufferDescriptor: TypeAlias = struct_WGPUCommandBufferDescriptor
WGPUCommandEncoderDescriptor: TypeAlias = struct_WGPUCommandEncoderDescriptor
WGPUComputePassDescriptor: TypeAlias = struct_WGPUComputePassDescriptor
WGPUDawnCacheDeviceDescriptor: TypeAlias = struct_WGPUDawnCacheDeviceDescriptor
WGPUDrmFormatCapabilities: TypeAlias = struct_WGPUDrmFormatCapabilities
WGPUExternalTextureDescriptor: TypeAlias = struct_WGPUExternalTextureDescriptor
WGPUFutureWaitInfo: TypeAlias = struct_WGPUFutureWaitInfo
WGPUImageCopyBuffer: TypeAlias = struct_WGPUImageCopyBuffer
WGPUImageCopyExternalTexture: TypeAlias = struct_WGPUImageCopyExternalTexture
WGPUImageCopyTexture: TypeAlias = struct_WGPUImageCopyTexture
WGPUInstanceDescriptor: TypeAlias = struct_WGPUInstanceDescriptor
WGPUPipelineLayoutDescriptor: TypeAlias = struct_WGPUPipelineLayoutDescriptor
WGPUPipelineLayoutPixelLocalStorage: TypeAlias = struct_WGPUPipelineLayoutPixelLocalStorage
WGPUQuerySetDescriptor: TypeAlias = struct_WGPUQuerySetDescriptor
WGPURenderBundleDescriptor: TypeAlias = struct_WGPURenderBundleDescriptor
WGPURenderBundleEncoderDescriptor: TypeAlias = struct_WGPURenderBundleEncoderDescriptor
WGPUSamplerDescriptor: TypeAlias = struct_WGPUSamplerDescriptor
WGPUShaderModuleDescriptor: TypeAlias = struct_WGPUShaderModuleDescriptor
WGPUShaderSourceWGSL: TypeAlias = struct_WGPUShaderSourceWGSL
WGPUSharedBufferMemoryDescriptor: TypeAlias = struct_WGPUSharedBufferMemoryDescriptor
WGPUSharedFenceDescriptor: TypeAlias = struct_WGPUSharedFenceDescriptor
WGPUSharedTextureMemoryAHardwareBufferProperties: TypeAlias = struct_WGPUSharedTextureMemoryAHardwareBufferProperties
WGPUSharedTextureMemoryDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryDescriptor
WGPUSharedTextureMemoryDmaBufDescriptor: TypeAlias = struct_WGPUSharedTextureMemoryDmaBufDescriptor
WGPUSharedTextureMemoryProperties: TypeAlias = struct_WGPUSharedTextureMemoryProperties
WGPUSupportedLimits: TypeAlias = struct_WGPUSupportedLimits
WGPUSurfaceDescriptor: TypeAlias = struct_WGPUSurfaceDescriptor
WGPUSurfaceSourceCanvasHTMLSelector_Emscripten: TypeAlias = struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten
WGPUTextureDescriptor: TypeAlias = struct_WGPUTextureDescriptor
WGPUTextureViewDescriptor: TypeAlias = struct_WGPUTextureViewDescriptor
WGPUBindGroupLayoutDescriptor: TypeAlias = struct_WGPUBindGroupLayoutDescriptor
WGPUCompilationInfo: TypeAlias = struct_WGPUCompilationInfo
WGPUDeviceDescriptor: TypeAlias = struct_WGPUDeviceDescriptor
WGPURenderPassDescriptor: TypeAlias = struct_WGPURenderPassDescriptor
WGPURenderPassPixelLocalStorage: TypeAlias = struct_WGPURenderPassPixelLocalStorage
WGPUComputePipelineDescriptor: TypeAlias = struct_WGPUComputePipelineDescriptor
WGPURenderPipelineDescriptor: TypeAlias = struct_WGPURenderPipelineDescriptor
WGPURenderPassDescriptorMaxDrawCount: TypeAlias = struct_WGPURenderPassMaxDrawCount
WGPUShaderModuleSPIRVDescriptor: TypeAlias = struct_WGPUShaderSourceSPIRV
WGPUShaderModuleWGSLDescriptor: TypeAlias = struct_WGPUShaderSourceWGSL
WGPUSurfaceDescriptorFromAndroidNativeWindow: TypeAlias = struct_WGPUSurfaceSourceAndroidNativeWindow
WGPUSurfaceDescriptorFromCanvasHTMLSelector: TypeAlias = struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten
WGPUSurfaceDescriptorFromMetalLayer: TypeAlias = struct_WGPUSurfaceSourceMetalLayer
WGPUSurfaceDescriptorFromWaylandSurface: TypeAlias = struct_WGPUSurfaceSourceWaylandSurface
WGPUSurfaceDescriptorFromWindowsHWND: TypeAlias = struct_WGPUSurfaceSourceWindowsHWND
WGPUSurfaceDescriptorFromXcbWindow: TypeAlias = struct_WGPUSurfaceSourceXCBWindow
WGPUSurfaceDescriptorFromXlibWindow: TypeAlias = struct_WGPUSurfaceSourceXlibWindow
WGPUProcAdapterInfoFreeMembers: TypeAlias = CFUNCTYPE(None, struct_WGPUAdapterInfo)
WGPUProcAdapterPropertiesMemoryHeapsFreeMembers: TypeAlias = CFUNCTYPE(None, struct_WGPUAdapterPropertiesMemoryHeaps)
WGPUProcCreateInstance: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUInstanceImpl), POINTER(struct_WGPUInstanceDescriptor))
WGPUProcDrmFormatCapabilitiesFreeMembers: TypeAlias = CFUNCTYPE(None, struct_WGPUDrmFormatCapabilities)
WGPUProcGetInstanceFeatures: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUInstanceFeatures))
WGPUProcGetProcAddress: TypeAlias = CFUNCTYPE(CFUNCTYPE(None, ), struct_WGPUStringView)
WGPUProcSharedBufferMemoryEndAccessStateFreeMembers: TypeAlias = CFUNCTYPE(None, struct_WGPUSharedBufferMemoryEndAccessState)
WGPUProcSharedTextureMemoryEndAccessStateFreeMembers: TypeAlias = CFUNCTYPE(None, struct_WGPUSharedTextureMemoryEndAccessState)
WGPUProcSupportedFeaturesFreeMembers: TypeAlias = CFUNCTYPE(None, struct_WGPUSupportedFeatures)
WGPUProcSurfaceCapabilitiesFreeMembers: TypeAlias = CFUNCTYPE(None, struct_WGPUSurfaceCapabilities)
WGPUProcAdapterCreateDevice: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUDeviceDescriptor))
WGPUProcAdapterGetFeatures: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUSupportedFeatures))
WGPUProcAdapterGetFormatCapabilities: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUAdapterImpl), enum_WGPUTextureFormat, POINTER(struct_WGPUFormatCapabilities))
WGPUProcAdapterGetInfo: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUAdapterInfo))
WGPUProcAdapterGetInstance: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUInstanceImpl), POINTER(struct_WGPUAdapterImpl))
WGPUProcAdapterGetLimits: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUSupportedLimits))
WGPUProcAdapterHasFeature: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUAdapterImpl), enum_WGPUFeatureName)
WGPUProcAdapterRequestDevice: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUDeviceDescriptor), CFUNCTYPE(None, enum_WGPURequestDeviceStatus, POINTER(struct_WGPUDeviceImpl), struct_WGPUStringView, POINTER(None)), POINTER(None))
WGPUProcAdapterRequestDevice2: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUDeviceDescriptor), struct_WGPURequestDeviceCallbackInfo2)
WGPUProcAdapterRequestDeviceF: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUDeviceDescriptor), struct_WGPURequestDeviceCallbackInfo)
WGPUProcAdapterAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUAdapterImpl))
WGPUProcAdapterRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUAdapterImpl))
WGPUProcBindGroupSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBindGroupImpl), struct_WGPUStringView)
WGPUProcBindGroupAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBindGroupImpl))
WGPUProcBindGroupRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBindGroupImpl))
WGPUProcBindGroupLayoutSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBindGroupLayoutImpl), struct_WGPUStringView)
WGPUProcBindGroupLayoutAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBindGroupLayoutImpl))
WGPUProcBindGroupLayoutRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBindGroupLayoutImpl))
WGPUProcBufferDestroy: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBufferImpl))
WGPUProcBufferGetConstMappedRange: TypeAlias = CFUNCTYPE(POINTER(None), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcBufferGetMapState: TypeAlias = CFUNCTYPE(enum_WGPUBufferMapState, POINTER(struct_WGPUBufferImpl))
WGPUProcBufferGetMappedRange: TypeAlias = CFUNCTYPE(POINTER(None), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcBufferGetSize: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint64], POINTER(struct_WGPUBufferImpl))
WGPUProcBufferGetUsage: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint64], POINTER(struct_WGPUBufferImpl))
WGPUProcBufferMapAsync: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], CFUNCTYPE(None, enum_WGPUBufferMapAsyncStatus, POINTER(None)), POINTER(None))
WGPUProcBufferMapAsync2: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], struct_WGPUBufferMapCallbackInfo2)
WGPUProcBufferMapAsyncF: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], struct_WGPUBufferMapCallbackInfo)
WGPUProcBufferSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBufferImpl), struct_WGPUStringView)
WGPUProcBufferUnmap: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBufferImpl))
WGPUProcBufferAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBufferImpl))
WGPUProcBufferRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUBufferImpl))
WGPUProcCommandBufferSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandBufferImpl), struct_WGPUStringView)
WGPUProcCommandBufferAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandBufferImpl))
WGPUProcCommandBufferRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandBufferImpl))
WGPUProcCommandEncoderBeginComputePass: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUComputePassEncoderImpl), POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUComputePassDescriptor))
WGPUProcCommandEncoderBeginRenderPass: TypeAlias = CFUNCTYPE(POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPURenderPassDescriptor))
WGPUProcCommandEncoderClearBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcCommandEncoderCopyBufferToBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcCommandEncoderCopyBufferToTexture: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUImageCopyBuffer), POINTER(struct_WGPUImageCopyTexture), POINTER(struct_WGPUExtent3D))
WGPUProcCommandEncoderCopyTextureToBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUImageCopyTexture), POINTER(struct_WGPUImageCopyBuffer), POINTER(struct_WGPUExtent3D))
WGPUProcCommandEncoderCopyTextureToTexture: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUImageCopyTexture), POINTER(struct_WGPUImageCopyTexture), POINTER(struct_WGPUExtent3D))
WGPUProcCommandEncoderFinish: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUCommandBufferImpl), POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUCommandBufferDescriptor))
WGPUProcCommandEncoderInjectValidationError: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderInsertDebugMarker: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderPopDebugGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl))
WGPUProcCommandEncoderPushDebugGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderResolveQuerySet: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUQuerySetImpl), Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64])
WGPUProcCommandEncoderSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), struct_WGPUStringView)
WGPUProcCommandEncoderWriteBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], POINTER(Annotated[int, ctypes.c_ubyte]), Annotated[int, ctypes.c_uint64])
WGPUProcCommandEncoderWriteTimestamp: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUQuerySetImpl), Annotated[int, ctypes.c_uint32])
WGPUProcCommandEncoderAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl))
WGPUProcCommandEncoderRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUCommandEncoderImpl))
WGPUProcComputePassEncoderDispatchWorkgroups: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl), Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32])
WGPUProcComputePassEncoderDispatchWorkgroupsIndirect: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64])
WGPUProcComputePassEncoderEnd: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderInsertDebugMarker: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderPopDebugGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderPushDebugGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderSetBindGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl), Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUBindGroupImpl), Annotated[int, ctypes.c_uint64], POINTER(Annotated[int, ctypes.c_uint32]))
WGPUProcComputePassEncoderSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl), struct_WGPUStringView)
WGPUProcComputePassEncoderSetPipeline: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl), POINTER(struct_WGPUComputePipelineImpl))
WGPUProcComputePassEncoderWriteTimestamp: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl), POINTER(struct_WGPUQuerySetImpl), Annotated[int, ctypes.c_uint32])
WGPUProcComputePassEncoderAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePassEncoderRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePassEncoderImpl))
WGPUProcComputePipelineGetBindGroupLayout: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUBindGroupLayoutImpl), POINTER(struct_WGPUComputePipelineImpl), Annotated[int, ctypes.c_uint32])
WGPUProcComputePipelineSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePipelineImpl), struct_WGPUStringView)
WGPUProcComputePipelineAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePipelineImpl))
WGPUProcComputePipelineRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUComputePipelineImpl))
WGPUProcDeviceCreateBindGroup: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUBindGroupImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUBindGroupDescriptor))
WGPUProcDeviceCreateBindGroupLayout: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUBindGroupLayoutImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUBindGroupLayoutDescriptor))
WGPUProcDeviceCreateBuffer: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUBufferImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUBufferDescriptor))
WGPUProcDeviceCreateCommandEncoder: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUCommandEncoderImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUCommandEncoderDescriptor))
WGPUProcDeviceCreateComputePipeline: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUComputePipelineImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUComputePipelineDescriptor))
WGPUProcDeviceCreateComputePipelineAsync: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUComputePipelineDescriptor), CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, POINTER(struct_WGPUComputePipelineImpl), struct_WGPUStringView, POINTER(None)), POINTER(None))
WGPUProcDeviceCreateComputePipelineAsync2: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUComputePipelineDescriptor), struct_WGPUCreateComputePipelineAsyncCallbackInfo2)
WGPUProcDeviceCreateComputePipelineAsyncF: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUComputePipelineDescriptor), struct_WGPUCreateComputePipelineAsyncCallbackInfo)
WGPUProcDeviceCreateErrorBuffer: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUBufferImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUBufferDescriptor))
WGPUProcDeviceCreateErrorExternalTexture: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUExternalTextureImpl), POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceCreateErrorShaderModule: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUShaderModuleImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUShaderModuleDescriptor), struct_WGPUStringView)
WGPUProcDeviceCreateErrorTexture: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUTextureImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUTextureDescriptor))
WGPUProcDeviceCreateExternalTexture: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUExternalTextureImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUExternalTextureDescriptor))
WGPUProcDeviceCreatePipelineLayout: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUPipelineLayoutImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUPipelineLayoutDescriptor))
WGPUProcDeviceCreateQuerySet: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUQuerySetImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUQuerySetDescriptor))
WGPUProcDeviceCreateRenderBundleEncoder: TypeAlias = CFUNCTYPE(POINTER(struct_WGPURenderBundleEncoderImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPURenderBundleEncoderDescriptor))
WGPUProcDeviceCreateRenderPipeline: TypeAlias = CFUNCTYPE(POINTER(struct_WGPURenderPipelineImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPURenderPipelineDescriptor))
WGPUProcDeviceCreateRenderPipelineAsync: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPURenderPipelineDescriptor), CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, POINTER(struct_WGPURenderPipelineImpl), struct_WGPUStringView, POINTER(None)), POINTER(None))
WGPUProcDeviceCreateRenderPipelineAsync2: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPURenderPipelineDescriptor), struct_WGPUCreateRenderPipelineAsyncCallbackInfo2)
WGPUProcDeviceCreateRenderPipelineAsyncF: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPURenderPipelineDescriptor), struct_WGPUCreateRenderPipelineAsyncCallbackInfo)
WGPUProcDeviceCreateSampler: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUSamplerImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUSamplerDescriptor))
WGPUProcDeviceCreateShaderModule: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUShaderModuleImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUShaderModuleDescriptor))
WGPUProcDeviceCreateTexture: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUTextureImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUTextureDescriptor))
WGPUProcDeviceDestroy: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceForceLoss: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), enum_WGPUDeviceLostReason, struct_WGPUStringView)
WGPUProcDeviceGetAHardwareBufferProperties: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUDeviceImpl), POINTER(None), POINTER(struct_WGPUAHardwareBufferProperties))
WGPUProcDeviceGetAdapter: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceGetAdapterInfo: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUAdapterInfo))
WGPUProcDeviceGetFeatures: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUSupportedFeatures))
WGPUProcDeviceGetLimits: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUSupportedLimits))
WGPUProcDeviceGetLostFuture: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceGetQueue: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUQueueImpl), POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceHasFeature: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUDeviceImpl), enum_WGPUFeatureName)
WGPUProcDeviceImportSharedBufferMemory: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUSharedBufferMemoryImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUSharedBufferMemoryDescriptor))
WGPUProcDeviceImportSharedFence: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUSharedFenceImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUSharedFenceDescriptor))
WGPUProcDeviceImportSharedTextureMemory: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUSharedTextureMemoryImpl), POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUSharedTextureMemoryDescriptor))
WGPUProcDeviceInjectError: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), enum_WGPUErrorType, struct_WGPUStringView)
WGPUProcDevicePopErrorScope: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), CFUNCTYPE(None, enum_WGPUErrorType, struct_WGPUStringView, POINTER(None)), POINTER(None))
WGPUProcDevicePopErrorScope2: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUDeviceImpl), struct_WGPUPopErrorScopeCallbackInfo2)
WGPUProcDevicePopErrorScopeF: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUDeviceImpl), struct_WGPUPopErrorScopeCallbackInfo)
WGPUProcDevicePushErrorScope: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), enum_WGPUErrorFilter)
WGPUProcDeviceSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), struct_WGPUStringView)
WGPUProcDeviceSetLoggingCallback: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), CFUNCTYPE(None, enum_WGPULoggingType, struct_WGPUStringView, POINTER(None)), POINTER(None))
WGPUProcDeviceTick: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceValidateTextureDescriptor: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl), POINTER(struct_WGPUTextureDescriptor))
WGPUProcDeviceAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl))
WGPUProcDeviceRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUDeviceImpl))
WGPUProcExternalTextureDestroy: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureExpire: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureRefresh: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUExternalTextureImpl), struct_WGPUStringView)
WGPUProcExternalTextureAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUExternalTextureImpl))
WGPUProcExternalTextureRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUExternalTextureImpl))
WGPUProcInstanceCreateSurface: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUSurfaceImpl), POINTER(struct_WGPUInstanceImpl), POINTER(struct_WGPUSurfaceDescriptor))
WGPUProcInstanceEnumerateWGSLLanguageFeatures: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint64], POINTER(struct_WGPUInstanceImpl), POINTER(enum_WGPUWGSLFeatureName))
WGPUProcInstanceHasWGSLLanguageFeature: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUInstanceImpl), enum_WGPUWGSLFeatureName)
WGPUProcInstanceProcessEvents: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUInstanceImpl))
WGPUProcInstanceRequestAdapter: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUInstanceImpl), POINTER(struct_WGPURequestAdapterOptions), CFUNCTYPE(None, enum_WGPURequestAdapterStatus, POINTER(struct_WGPUAdapterImpl), struct_WGPUStringView, POINTER(None)), POINTER(None))
WGPUProcInstanceRequestAdapter2: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUInstanceImpl), POINTER(struct_WGPURequestAdapterOptions), struct_WGPURequestAdapterCallbackInfo2)
WGPUProcInstanceRequestAdapterF: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUInstanceImpl), POINTER(struct_WGPURequestAdapterOptions), struct_WGPURequestAdapterCallbackInfo)
WGPUProcInstanceWaitAny: TypeAlias = CFUNCTYPE(enum_WGPUWaitStatus, POINTER(struct_WGPUInstanceImpl), Annotated[int, ctypes.c_uint64], POINTER(struct_WGPUFutureWaitInfo), Annotated[int, ctypes.c_uint64])
WGPUProcInstanceAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUInstanceImpl))
WGPUProcInstanceRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUInstanceImpl))
WGPUProcPipelineLayoutSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUPipelineLayoutImpl), struct_WGPUStringView)
WGPUProcPipelineLayoutAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUPipelineLayoutImpl))
WGPUProcPipelineLayoutRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUPipelineLayoutImpl))
WGPUProcQuerySetDestroy: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQuerySetImpl))
WGPUProcQuerySetGetCount: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUQuerySetImpl))
WGPUProcQuerySetGetType: TypeAlias = CFUNCTYPE(enum_WGPUQueryType, POINTER(struct_WGPUQuerySetImpl))
WGPUProcQuerySetSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQuerySetImpl), struct_WGPUStringView)
WGPUProcQuerySetAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQuerySetImpl))
WGPUProcQuerySetRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQuerySetImpl))
WGPUProcQueueCopyExternalTextureForBrowser: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl), POINTER(struct_WGPUImageCopyExternalTexture), POINTER(struct_WGPUImageCopyTexture), POINTER(struct_WGPUExtent3D), POINTER(struct_WGPUCopyTextureForBrowserOptions))
WGPUProcQueueCopyTextureForBrowser: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl), POINTER(struct_WGPUImageCopyTexture), POINTER(struct_WGPUImageCopyTexture), POINTER(struct_WGPUExtent3D), POINTER(struct_WGPUCopyTextureForBrowserOptions))
WGPUProcQueueOnSubmittedWorkDone: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl), CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, POINTER(None)), POINTER(None))
WGPUProcQueueOnSubmittedWorkDone2: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUQueueImpl), struct_WGPUQueueWorkDoneCallbackInfo2)
WGPUProcQueueOnSubmittedWorkDoneF: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUQueueImpl), struct_WGPUQueueWorkDoneCallbackInfo)
WGPUProcQueueSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl), struct_WGPUStringView)
WGPUProcQueueSubmit: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl), Annotated[int, ctypes.c_uint64], POINTER(POINTER(struct_WGPUCommandBufferImpl)))
WGPUProcQueueWriteBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], POINTER(None), Annotated[int, ctypes.c_uint64])
WGPUProcQueueWriteTexture: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl), POINTER(struct_WGPUImageCopyTexture), POINTER(None), Annotated[int, ctypes.c_uint64], POINTER(struct_WGPUTextureDataLayout), POINTER(struct_WGPUExtent3D))
WGPUProcQueueAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl))
WGPUProcQueueRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUQueueImpl))
WGPUProcRenderBundleSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleImpl), struct_WGPUStringView)
WGPUProcRenderBundleAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleImpl))
WGPUProcRenderBundleRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleImpl))
WGPUProcRenderBundleEncoderDraw: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderBundleEncoderDrawIndexed: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_int32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderBundleEncoderDrawIndexedIndirect: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64])
WGPUProcRenderBundleEncoderDrawIndirect: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64])
WGPUProcRenderBundleEncoderFinish: TypeAlias = CFUNCTYPE(POINTER(struct_WGPURenderBundleImpl), POINTER(struct_WGPURenderBundleEncoderImpl), POINTER(struct_WGPURenderBundleDescriptor))
WGPUProcRenderBundleEncoderInsertDebugMarker: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderPopDebugGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderBundleEncoderPushDebugGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderSetBindGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUBindGroupImpl), Annotated[int, ctypes.c_uint64], POINTER(Annotated[int, ctypes.c_uint32]))
WGPUProcRenderBundleEncoderSetIndexBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), POINTER(struct_WGPUBufferImpl), enum_WGPUIndexFormat, Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcRenderBundleEncoderSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), struct_WGPUStringView)
WGPUProcRenderBundleEncoderSetPipeline: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), POINTER(struct_WGPURenderPipelineImpl))
WGPUProcRenderBundleEncoderSetVertexBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl), Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcRenderBundleEncoderAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderBundleEncoderRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderBundleEncoderImpl))
WGPUProcRenderPassEncoderBeginOcclusionQuery: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderDraw: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderDrawIndexed: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_int32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderDrawIndexedIndirect: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderDrawIndirect: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderEnd: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderEndOcclusionQuery: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderExecuteBundles: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[int, ctypes.c_uint64], POINTER(POINTER(struct_WGPURenderBundleImpl)))
WGPUProcRenderPassEncoderInsertDebugMarker: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderMultiDrawIndexedIndirect: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderMultiDrawIndirect: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderPixelLocalStorageBarrier: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderPopDebugGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderPushDebugGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderSetBindGroup: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUBindGroupImpl), Annotated[int, ctypes.c_uint64], POINTER(Annotated[int, ctypes.c_uint32]))
WGPUProcRenderPassEncoderSetBlendConstant: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPUColor))
WGPUProcRenderPassEncoderSetIndexBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPUBufferImpl), enum_WGPUIndexFormat, Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), struct_WGPUStringView)
WGPUProcRenderPassEncoderSetPipeline: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPURenderPipelineImpl))
WGPUProcRenderPassEncoderSetScissorRect: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderSetStencilReference: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderSetVertexBuffer: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUBufferImpl), Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderSetViewport: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float])
WGPUProcRenderPassEncoderWriteTimestamp: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl), POINTER(struct_WGPUQuerySetImpl), Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPassEncoderRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPassEncoderImpl))
WGPUProcRenderPipelineGetBindGroupLayout: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUBindGroupLayoutImpl), POINTER(struct_WGPURenderPipelineImpl), Annotated[int, ctypes.c_uint32])
WGPUProcRenderPipelineSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPipelineImpl), struct_WGPUStringView)
WGPUProcRenderPipelineAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPipelineImpl))
WGPUProcRenderPipelineRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPURenderPipelineImpl))
WGPUProcSamplerSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSamplerImpl), struct_WGPUStringView)
WGPUProcSamplerAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSamplerImpl))
WGPUProcSamplerRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSamplerImpl))
WGPUProcShaderModuleGetCompilationInfo: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUShaderModuleImpl), CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, POINTER(struct_WGPUCompilationInfo), POINTER(None)), POINTER(None))
WGPUProcShaderModuleGetCompilationInfo2: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUShaderModuleImpl), struct_WGPUCompilationInfoCallbackInfo2)
WGPUProcShaderModuleGetCompilationInfoF: TypeAlias = CFUNCTYPE(struct_WGPUFuture, POINTER(struct_WGPUShaderModuleImpl), struct_WGPUCompilationInfoCallbackInfo)
WGPUProcShaderModuleSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUShaderModuleImpl), struct_WGPUStringView)
WGPUProcShaderModuleAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUShaderModuleImpl))
WGPUProcShaderModuleRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUShaderModuleImpl))
WGPUProcSharedBufferMemoryBeginAccess: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUSharedBufferMemoryImpl), POINTER(struct_WGPUBufferImpl), POINTER(struct_WGPUSharedBufferMemoryBeginAccessDescriptor))
WGPUProcSharedBufferMemoryCreateBuffer: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUBufferImpl), POINTER(struct_WGPUSharedBufferMemoryImpl), POINTER(struct_WGPUBufferDescriptor))
WGPUProcSharedBufferMemoryEndAccess: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUSharedBufferMemoryImpl), POINTER(struct_WGPUBufferImpl), POINTER(struct_WGPUSharedBufferMemoryEndAccessState))
WGPUProcSharedBufferMemoryGetProperties: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUSharedBufferMemoryImpl), POINTER(struct_WGPUSharedBufferMemoryProperties))
WGPUProcSharedBufferMemoryIsDeviceLost: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedBufferMemorySetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedBufferMemoryImpl), struct_WGPUStringView)
WGPUProcSharedBufferMemoryAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedBufferMemoryRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedBufferMemoryImpl))
WGPUProcSharedFenceExportInfo: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedFenceImpl), POINTER(struct_WGPUSharedFenceExportInfo))
WGPUProcSharedFenceAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedFenceImpl))
WGPUProcSharedFenceRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedFenceImpl))
WGPUProcSharedTextureMemoryBeginAccess: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUSharedTextureMemoryImpl), POINTER(struct_WGPUTextureImpl), POINTER(struct_WGPUSharedTextureMemoryBeginAccessDescriptor))
WGPUProcSharedTextureMemoryCreateTexture: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUTextureImpl), POINTER(struct_WGPUSharedTextureMemoryImpl), POINTER(struct_WGPUTextureDescriptor))
WGPUProcSharedTextureMemoryEndAccess: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUSharedTextureMemoryImpl), POINTER(struct_WGPUTextureImpl), POINTER(struct_WGPUSharedTextureMemoryEndAccessState))
WGPUProcSharedTextureMemoryGetProperties: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUSharedTextureMemoryImpl), POINTER(struct_WGPUSharedTextureMemoryProperties))
WGPUProcSharedTextureMemoryIsDeviceLost: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUSharedTextureMemoryImpl))
WGPUProcSharedTextureMemorySetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedTextureMemoryImpl), struct_WGPUStringView)
WGPUProcSharedTextureMemoryAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedTextureMemoryImpl))
WGPUProcSharedTextureMemoryRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSharedTextureMemoryImpl))
WGPUProcSurfaceConfigure: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSurfaceImpl), POINTER(struct_WGPUSurfaceConfiguration))
WGPUProcSurfaceGetCapabilities: TypeAlias = CFUNCTYPE(enum_WGPUStatus, POINTER(struct_WGPUSurfaceImpl), POINTER(struct_WGPUAdapterImpl), POINTER(struct_WGPUSurfaceCapabilities))
WGPUProcSurfaceGetCurrentTexture: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSurfaceImpl), POINTER(struct_WGPUSurfaceTexture))
WGPUProcSurfacePresent: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSurfaceImpl))
WGPUProcSurfaceSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSurfaceImpl), struct_WGPUStringView)
WGPUProcSurfaceUnconfigure: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSurfaceImpl))
WGPUProcSurfaceAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSurfaceImpl))
WGPUProcSurfaceRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUSurfaceImpl))
WGPUProcTextureCreateErrorView: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUTextureViewImpl), POINTER(struct_WGPUTextureImpl), POINTER(struct_WGPUTextureViewDescriptor))
WGPUProcTextureCreateView: TypeAlias = CFUNCTYPE(POINTER(struct_WGPUTextureViewImpl), POINTER(struct_WGPUTextureImpl), POINTER(struct_WGPUTextureViewDescriptor))
WGPUProcTextureDestroy: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetDepthOrArrayLayers: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetDimension: TypeAlias = CFUNCTYPE(enum_WGPUTextureDimension, POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetFormat: TypeAlias = CFUNCTYPE(enum_WGPUTextureFormat, POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetHeight: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetMipLevelCount: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetSampleCount: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetUsage: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint64], POINTER(struct_WGPUTextureImpl))
WGPUProcTextureGetWidth: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint32], POINTER(struct_WGPUTextureImpl))
WGPUProcTextureSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUTextureImpl), struct_WGPUStringView)
WGPUProcTextureAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUTextureImpl))
WGPUProcTextureRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUTextureImpl))
WGPUProcTextureViewSetLabel: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUTextureViewImpl), struct_WGPUStringView)
WGPUProcTextureViewAddRef: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUTextureViewImpl))
WGPUProcTextureViewRelease: TypeAlias = CFUNCTYPE(None, POINTER(struct_WGPUTextureViewImpl))
@dll.bind
def wgpuAdapterInfoFreeMembers(value:WGPUAdapterInfo) -> None: ...
@dll.bind
def wgpuAdapterPropertiesMemoryHeapsFreeMembers(value:WGPUAdapterPropertiesMemoryHeaps) -> None: ...
@dll.bind
def wgpuCreateInstance(descriptor:POINTER(WGPUInstanceDescriptor)) -> WGPUInstance: ...
@dll.bind
def wgpuDrmFormatCapabilitiesFreeMembers(value:WGPUDrmFormatCapabilities) -> None: ...
@dll.bind
def wgpuGetInstanceFeatures(features:POINTER(WGPUInstanceFeatures)) -> WGPUStatus: ...
@dll.bind
def wgpuGetProcAddress(procName:WGPUStringView) -> WGPUProc: ...
@dll.bind
def wgpuSharedBufferMemoryEndAccessStateFreeMembers(value:WGPUSharedBufferMemoryEndAccessState) -> None: ...
@dll.bind
def wgpuSharedTextureMemoryEndAccessStateFreeMembers(value:WGPUSharedTextureMemoryEndAccessState) -> None: ...
@dll.bind
def wgpuSupportedFeaturesFreeMembers(value:WGPUSupportedFeatures) -> None: ...
@dll.bind
def wgpuSurfaceCapabilitiesFreeMembers(value:WGPUSurfaceCapabilities) -> None: ...
@dll.bind
def wgpuAdapterCreateDevice(adapter:WGPUAdapter, descriptor:POINTER(WGPUDeviceDescriptor)) -> WGPUDevice: ...
@dll.bind
def wgpuAdapterGetFeatures(adapter:WGPUAdapter, features:POINTER(WGPUSupportedFeatures)) -> None: ...
@dll.bind
def wgpuAdapterGetFormatCapabilities(adapter:WGPUAdapter, format:WGPUTextureFormat, capabilities:POINTER(WGPUFormatCapabilities)) -> WGPUStatus: ...
@dll.bind
def wgpuAdapterGetInfo(adapter:WGPUAdapter, info:POINTER(WGPUAdapterInfo)) -> WGPUStatus: ...
@dll.bind
def wgpuAdapterGetInstance(adapter:WGPUAdapter) -> WGPUInstance: ...
@dll.bind
def wgpuAdapterGetLimits(adapter:WGPUAdapter, limits:POINTER(WGPUSupportedLimits)) -> WGPUStatus: ...
@dll.bind
def wgpuAdapterHasFeature(adapter:WGPUAdapter, feature:WGPUFeatureName) -> WGPUBool: ...
@dll.bind
def wgpuAdapterRequestDevice(adapter:WGPUAdapter, descriptor:POINTER(WGPUDeviceDescriptor), callback:WGPURequestDeviceCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuAdapterRequestDevice2(adapter:WGPUAdapter, options:POINTER(WGPUDeviceDescriptor), callbackInfo:WGPURequestDeviceCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuAdapterRequestDeviceF(adapter:WGPUAdapter, options:POINTER(WGPUDeviceDescriptor), callbackInfo:WGPURequestDeviceCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuAdapterAddRef(adapter:WGPUAdapter) -> None: ...
@dll.bind
def wgpuAdapterRelease(adapter:WGPUAdapter) -> None: ...
@dll.bind
def wgpuBindGroupSetLabel(bindGroup:WGPUBindGroup, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuBindGroupAddRef(bindGroup:WGPUBindGroup) -> None: ...
@dll.bind
def wgpuBindGroupRelease(bindGroup:WGPUBindGroup) -> None: ...
@dll.bind
def wgpuBindGroupLayoutSetLabel(bindGroupLayout:WGPUBindGroupLayout, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuBindGroupLayoutAddRef(bindGroupLayout:WGPUBindGroupLayout) -> None: ...
@dll.bind
def wgpuBindGroupLayoutRelease(bindGroupLayout:WGPUBindGroupLayout) -> None: ...
@dll.bind
def wgpuBufferDestroy(buffer:WGPUBuffer) -> None: ...
@dll.bind
def wgpuBufferGetConstMappedRange(buffer:WGPUBuffer, offset:size_t, size:size_t) -> POINTER(None): ...
@dll.bind
def wgpuBufferGetMapState(buffer:WGPUBuffer) -> WGPUBufferMapState: ...
@dll.bind
def wgpuBufferGetMappedRange(buffer:WGPUBuffer, offset:size_t, size:size_t) -> POINTER(None): ...
@dll.bind
def wgpuBufferGetSize(buffer:WGPUBuffer) -> uint64_t: ...
@dll.bind
def wgpuBufferGetUsage(buffer:WGPUBuffer) -> WGPUBufferUsage: ...
@dll.bind
def wgpuBufferMapAsync(buffer:WGPUBuffer, mode:WGPUMapMode, offset:size_t, size:size_t, callback:WGPUBufferMapCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuBufferMapAsync2(buffer:WGPUBuffer, mode:WGPUMapMode, offset:size_t, size:size_t, callbackInfo:WGPUBufferMapCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuBufferMapAsyncF(buffer:WGPUBuffer, mode:WGPUMapMode, offset:size_t, size:size_t, callbackInfo:WGPUBufferMapCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuBufferSetLabel(buffer:WGPUBuffer, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuBufferUnmap(buffer:WGPUBuffer) -> None: ...
@dll.bind
def wgpuBufferAddRef(buffer:WGPUBuffer) -> None: ...
@dll.bind
def wgpuBufferRelease(buffer:WGPUBuffer) -> None: ...
@dll.bind
def wgpuCommandBufferSetLabel(commandBuffer:WGPUCommandBuffer, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuCommandBufferAddRef(commandBuffer:WGPUCommandBuffer) -> None: ...
@dll.bind
def wgpuCommandBufferRelease(commandBuffer:WGPUCommandBuffer) -> None: ...
@dll.bind
def wgpuCommandEncoderBeginComputePass(commandEncoder:WGPUCommandEncoder, descriptor:POINTER(WGPUComputePassDescriptor)) -> WGPUComputePassEncoder: ...
@dll.bind
def wgpuCommandEncoderBeginRenderPass(commandEncoder:WGPUCommandEncoder, descriptor:POINTER(WGPURenderPassDescriptor)) -> WGPURenderPassEncoder: ...
@dll.bind
def wgpuCommandEncoderClearBuffer(commandEncoder:WGPUCommandEncoder, buffer:WGPUBuffer, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind
def wgpuCommandEncoderCopyBufferToBuffer(commandEncoder:WGPUCommandEncoder, source:WGPUBuffer, sourceOffset:uint64_t, destination:WGPUBuffer, destinationOffset:uint64_t, size:uint64_t) -> None: ...
@dll.bind
def wgpuCommandEncoderCopyBufferToTexture(commandEncoder:WGPUCommandEncoder, source:POINTER(WGPUImageCopyBuffer), destination:POINTER(WGPUImageCopyTexture), copySize:POINTER(WGPUExtent3D)) -> None: ...
@dll.bind
def wgpuCommandEncoderCopyTextureToBuffer(commandEncoder:WGPUCommandEncoder, source:POINTER(WGPUImageCopyTexture), destination:POINTER(WGPUImageCopyBuffer), copySize:POINTER(WGPUExtent3D)) -> None: ...
@dll.bind
def wgpuCommandEncoderCopyTextureToTexture(commandEncoder:WGPUCommandEncoder, source:POINTER(WGPUImageCopyTexture), destination:POINTER(WGPUImageCopyTexture), copySize:POINTER(WGPUExtent3D)) -> None: ...
@dll.bind
def wgpuCommandEncoderFinish(commandEncoder:WGPUCommandEncoder, descriptor:POINTER(WGPUCommandBufferDescriptor)) -> WGPUCommandBuffer: ...
@dll.bind
def wgpuCommandEncoderInjectValidationError(commandEncoder:WGPUCommandEncoder, message:WGPUStringView) -> None: ...
@dll.bind
def wgpuCommandEncoderInsertDebugMarker(commandEncoder:WGPUCommandEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuCommandEncoderPopDebugGroup(commandEncoder:WGPUCommandEncoder) -> None: ...
@dll.bind
def wgpuCommandEncoderPushDebugGroup(commandEncoder:WGPUCommandEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuCommandEncoderResolveQuerySet(commandEncoder:WGPUCommandEncoder, querySet:WGPUQuerySet, firstQuery:uint32_t, queryCount:uint32_t, destination:WGPUBuffer, destinationOffset:uint64_t) -> None: ...
@dll.bind
def wgpuCommandEncoderSetLabel(commandEncoder:WGPUCommandEncoder, label:WGPUStringView) -> None: ...
uint8_t: TypeAlias = Annotated[int, ctypes.c_ubyte]
@dll.bind
def wgpuCommandEncoderWriteBuffer(commandEncoder:WGPUCommandEncoder, buffer:WGPUBuffer, bufferOffset:uint64_t, data:POINTER(uint8_t), size:uint64_t) -> None: ...
@dll.bind
def wgpuCommandEncoderWriteTimestamp(commandEncoder:WGPUCommandEncoder, querySet:WGPUQuerySet, queryIndex:uint32_t) -> None: ...
@dll.bind
def wgpuCommandEncoderAddRef(commandEncoder:WGPUCommandEncoder) -> None: ...
@dll.bind
def wgpuCommandEncoderRelease(commandEncoder:WGPUCommandEncoder) -> None: ...
@dll.bind
def wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder:WGPUComputePassEncoder, workgroupCountX:uint32_t, workgroupCountY:uint32_t, workgroupCountZ:uint32_t) -> None: ...
@dll.bind
def wgpuComputePassEncoderDispatchWorkgroupsIndirect(computePassEncoder:WGPUComputePassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind
def wgpuComputePassEncoderEnd(computePassEncoder:WGPUComputePassEncoder) -> None: ...
@dll.bind
def wgpuComputePassEncoderInsertDebugMarker(computePassEncoder:WGPUComputePassEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuComputePassEncoderPopDebugGroup(computePassEncoder:WGPUComputePassEncoder) -> None: ...
@dll.bind
def wgpuComputePassEncoderPushDebugGroup(computePassEncoder:WGPUComputePassEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuComputePassEncoderSetBindGroup(computePassEncoder:WGPUComputePassEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:POINTER(uint32_t)) -> None: ...
@dll.bind
def wgpuComputePassEncoderSetLabel(computePassEncoder:WGPUComputePassEncoder, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuComputePassEncoderSetPipeline(computePassEncoder:WGPUComputePassEncoder, pipeline:WGPUComputePipeline) -> None: ...
@dll.bind
def wgpuComputePassEncoderWriteTimestamp(computePassEncoder:WGPUComputePassEncoder, querySet:WGPUQuerySet, queryIndex:uint32_t) -> None: ...
@dll.bind
def wgpuComputePassEncoderAddRef(computePassEncoder:WGPUComputePassEncoder) -> None: ...
@dll.bind
def wgpuComputePassEncoderRelease(computePassEncoder:WGPUComputePassEncoder) -> None: ...
@dll.bind
def wgpuComputePipelineGetBindGroupLayout(computePipeline:WGPUComputePipeline, groupIndex:uint32_t) -> WGPUBindGroupLayout: ...
@dll.bind
def wgpuComputePipelineSetLabel(computePipeline:WGPUComputePipeline, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuComputePipelineAddRef(computePipeline:WGPUComputePipeline) -> None: ...
@dll.bind
def wgpuComputePipelineRelease(computePipeline:WGPUComputePipeline) -> None: ...
@dll.bind
def wgpuDeviceCreateBindGroup(device:WGPUDevice, descriptor:POINTER(WGPUBindGroupDescriptor)) -> WGPUBindGroup: ...
@dll.bind
def wgpuDeviceCreateBindGroupLayout(device:WGPUDevice, descriptor:POINTER(WGPUBindGroupLayoutDescriptor)) -> WGPUBindGroupLayout: ...
@dll.bind
def wgpuDeviceCreateBuffer(device:WGPUDevice, descriptor:POINTER(WGPUBufferDescriptor)) -> WGPUBuffer: ...
@dll.bind
def wgpuDeviceCreateCommandEncoder(device:WGPUDevice, descriptor:POINTER(WGPUCommandEncoderDescriptor)) -> WGPUCommandEncoder: ...
@dll.bind
def wgpuDeviceCreateComputePipeline(device:WGPUDevice, descriptor:POINTER(WGPUComputePipelineDescriptor)) -> WGPUComputePipeline: ...
@dll.bind
def wgpuDeviceCreateComputePipelineAsync(device:WGPUDevice, descriptor:POINTER(WGPUComputePipelineDescriptor), callback:WGPUCreateComputePipelineAsyncCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuDeviceCreateComputePipelineAsync2(device:WGPUDevice, descriptor:POINTER(WGPUComputePipelineDescriptor), callbackInfo:WGPUCreateComputePipelineAsyncCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceCreateComputePipelineAsyncF(device:WGPUDevice, descriptor:POINTER(WGPUComputePipelineDescriptor), callbackInfo:WGPUCreateComputePipelineAsyncCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceCreateErrorBuffer(device:WGPUDevice, descriptor:POINTER(WGPUBufferDescriptor)) -> WGPUBuffer: ...
@dll.bind
def wgpuDeviceCreateErrorExternalTexture(device:WGPUDevice) -> WGPUExternalTexture: ...
@dll.bind
def wgpuDeviceCreateErrorShaderModule(device:WGPUDevice, descriptor:POINTER(WGPUShaderModuleDescriptor), errorMessage:WGPUStringView) -> WGPUShaderModule: ...
@dll.bind
def wgpuDeviceCreateErrorTexture(device:WGPUDevice, descriptor:POINTER(WGPUTextureDescriptor)) -> WGPUTexture: ...
@dll.bind
def wgpuDeviceCreateExternalTexture(device:WGPUDevice, externalTextureDescriptor:POINTER(WGPUExternalTextureDescriptor)) -> WGPUExternalTexture: ...
@dll.bind
def wgpuDeviceCreatePipelineLayout(device:WGPUDevice, descriptor:POINTER(WGPUPipelineLayoutDescriptor)) -> WGPUPipelineLayout: ...
@dll.bind
def wgpuDeviceCreateQuerySet(device:WGPUDevice, descriptor:POINTER(WGPUQuerySetDescriptor)) -> WGPUQuerySet: ...
@dll.bind
def wgpuDeviceCreateRenderBundleEncoder(device:WGPUDevice, descriptor:POINTER(WGPURenderBundleEncoderDescriptor)) -> WGPURenderBundleEncoder: ...
@dll.bind
def wgpuDeviceCreateRenderPipeline(device:WGPUDevice, descriptor:POINTER(WGPURenderPipelineDescriptor)) -> WGPURenderPipeline: ...
@dll.bind
def wgpuDeviceCreateRenderPipelineAsync(device:WGPUDevice, descriptor:POINTER(WGPURenderPipelineDescriptor), callback:WGPUCreateRenderPipelineAsyncCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuDeviceCreateRenderPipelineAsync2(device:WGPUDevice, descriptor:POINTER(WGPURenderPipelineDescriptor), callbackInfo:WGPUCreateRenderPipelineAsyncCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceCreateRenderPipelineAsyncF(device:WGPUDevice, descriptor:POINTER(WGPURenderPipelineDescriptor), callbackInfo:WGPUCreateRenderPipelineAsyncCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceCreateSampler(device:WGPUDevice, descriptor:POINTER(WGPUSamplerDescriptor)) -> WGPUSampler: ...
@dll.bind
def wgpuDeviceCreateShaderModule(device:WGPUDevice, descriptor:POINTER(WGPUShaderModuleDescriptor)) -> WGPUShaderModule: ...
@dll.bind
def wgpuDeviceCreateTexture(device:WGPUDevice, descriptor:POINTER(WGPUTextureDescriptor)) -> WGPUTexture: ...
@dll.bind
def wgpuDeviceDestroy(device:WGPUDevice) -> None: ...
@dll.bind
def wgpuDeviceForceLoss(device:WGPUDevice, type:WGPUDeviceLostReason, message:WGPUStringView) -> None: ...
@dll.bind
def wgpuDeviceGetAHardwareBufferProperties(device:WGPUDevice, handle:POINTER(None), properties:POINTER(WGPUAHardwareBufferProperties)) -> WGPUStatus: ...
@dll.bind
def wgpuDeviceGetAdapter(device:WGPUDevice) -> WGPUAdapter: ...
@dll.bind
def wgpuDeviceGetAdapterInfo(device:WGPUDevice, adapterInfo:POINTER(WGPUAdapterInfo)) -> WGPUStatus: ...
@dll.bind
def wgpuDeviceGetFeatures(device:WGPUDevice, features:POINTER(WGPUSupportedFeatures)) -> None: ...
@dll.bind
def wgpuDeviceGetLimits(device:WGPUDevice, limits:POINTER(WGPUSupportedLimits)) -> WGPUStatus: ...
@dll.bind
def wgpuDeviceGetLostFuture(device:WGPUDevice) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceGetQueue(device:WGPUDevice) -> WGPUQueue: ...
@dll.bind
def wgpuDeviceHasFeature(device:WGPUDevice, feature:WGPUFeatureName) -> WGPUBool: ...
@dll.bind
def wgpuDeviceImportSharedBufferMemory(device:WGPUDevice, descriptor:POINTER(WGPUSharedBufferMemoryDescriptor)) -> WGPUSharedBufferMemory: ...
@dll.bind
def wgpuDeviceImportSharedFence(device:WGPUDevice, descriptor:POINTER(WGPUSharedFenceDescriptor)) -> WGPUSharedFence: ...
@dll.bind
def wgpuDeviceImportSharedTextureMemory(device:WGPUDevice, descriptor:POINTER(WGPUSharedTextureMemoryDescriptor)) -> WGPUSharedTextureMemory: ...
@dll.bind
def wgpuDeviceInjectError(device:WGPUDevice, type:WGPUErrorType, message:WGPUStringView) -> None: ...
@dll.bind
def wgpuDevicePopErrorScope(device:WGPUDevice, oldCallback:WGPUErrorCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuDevicePopErrorScope2(device:WGPUDevice, callbackInfo:WGPUPopErrorScopeCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuDevicePopErrorScopeF(device:WGPUDevice, callbackInfo:WGPUPopErrorScopeCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuDevicePushErrorScope(device:WGPUDevice, filter:WGPUErrorFilter) -> None: ...
@dll.bind
def wgpuDeviceSetLabel(device:WGPUDevice, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuDeviceSetLoggingCallback(device:WGPUDevice, callback:WGPULoggingCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuDeviceTick(device:WGPUDevice) -> None: ...
@dll.bind
def wgpuDeviceValidateTextureDescriptor(device:WGPUDevice, descriptor:POINTER(WGPUTextureDescriptor)) -> None: ...
@dll.bind
def wgpuDeviceAddRef(device:WGPUDevice) -> None: ...
@dll.bind
def wgpuDeviceRelease(device:WGPUDevice) -> None: ...
@dll.bind
def wgpuExternalTextureDestroy(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind
def wgpuExternalTextureExpire(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind
def wgpuExternalTextureRefresh(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind
def wgpuExternalTextureSetLabel(externalTexture:WGPUExternalTexture, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuExternalTextureAddRef(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind
def wgpuExternalTextureRelease(externalTexture:WGPUExternalTexture) -> None: ...
@dll.bind
def wgpuInstanceCreateSurface(instance:WGPUInstance, descriptor:POINTER(WGPUSurfaceDescriptor)) -> WGPUSurface: ...
@dll.bind
def wgpuInstanceEnumerateWGSLLanguageFeatures(instance:WGPUInstance, features:POINTER(WGPUWGSLFeatureName)) -> size_t: ...
@dll.bind
def wgpuInstanceHasWGSLLanguageFeature(instance:WGPUInstance, feature:WGPUWGSLFeatureName) -> WGPUBool: ...
@dll.bind
def wgpuInstanceProcessEvents(instance:WGPUInstance) -> None: ...
@dll.bind
def wgpuInstanceRequestAdapter(instance:WGPUInstance, options:POINTER(WGPURequestAdapterOptions), callback:WGPURequestAdapterCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuInstanceRequestAdapter2(instance:WGPUInstance, options:POINTER(WGPURequestAdapterOptions), callbackInfo:WGPURequestAdapterCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuInstanceRequestAdapterF(instance:WGPUInstance, options:POINTER(WGPURequestAdapterOptions), callbackInfo:WGPURequestAdapterCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuInstanceWaitAny(instance:WGPUInstance, futureCount:size_t, futures:POINTER(WGPUFutureWaitInfo), timeoutNS:uint64_t) -> WGPUWaitStatus: ...
@dll.bind
def wgpuInstanceAddRef(instance:WGPUInstance) -> None: ...
@dll.bind
def wgpuInstanceRelease(instance:WGPUInstance) -> None: ...
@dll.bind
def wgpuPipelineLayoutSetLabel(pipelineLayout:WGPUPipelineLayout, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuPipelineLayoutAddRef(pipelineLayout:WGPUPipelineLayout) -> None: ...
@dll.bind
def wgpuPipelineLayoutRelease(pipelineLayout:WGPUPipelineLayout) -> None: ...
@dll.bind
def wgpuQuerySetDestroy(querySet:WGPUQuerySet) -> None: ...
@dll.bind
def wgpuQuerySetGetCount(querySet:WGPUQuerySet) -> uint32_t: ...
@dll.bind
def wgpuQuerySetGetType(querySet:WGPUQuerySet) -> WGPUQueryType: ...
@dll.bind
def wgpuQuerySetSetLabel(querySet:WGPUQuerySet, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuQuerySetAddRef(querySet:WGPUQuerySet) -> None: ...
@dll.bind
def wgpuQuerySetRelease(querySet:WGPUQuerySet) -> None: ...
@dll.bind
def wgpuQueueCopyExternalTextureForBrowser(queue:WGPUQueue, source:POINTER(WGPUImageCopyExternalTexture), destination:POINTER(WGPUImageCopyTexture), copySize:POINTER(WGPUExtent3D), options:POINTER(WGPUCopyTextureForBrowserOptions)) -> None: ...
@dll.bind
def wgpuQueueCopyTextureForBrowser(queue:WGPUQueue, source:POINTER(WGPUImageCopyTexture), destination:POINTER(WGPUImageCopyTexture), copySize:POINTER(WGPUExtent3D), options:POINTER(WGPUCopyTextureForBrowserOptions)) -> None: ...
@dll.bind
def wgpuQueueOnSubmittedWorkDone(queue:WGPUQueue, callback:WGPUQueueWorkDoneCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuQueueOnSubmittedWorkDone2(queue:WGPUQueue, callbackInfo:WGPUQueueWorkDoneCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuQueueOnSubmittedWorkDoneF(queue:WGPUQueue, callbackInfo:WGPUQueueWorkDoneCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuQueueSetLabel(queue:WGPUQueue, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuQueueSubmit(queue:WGPUQueue, commandCount:size_t, commands:POINTER(WGPUCommandBuffer)) -> None: ...
@dll.bind
def wgpuQueueWriteBuffer(queue:WGPUQueue, buffer:WGPUBuffer, bufferOffset:uint64_t, data:POINTER(None), size:size_t) -> None: ...
@dll.bind
def wgpuQueueWriteTexture(queue:WGPUQueue, destination:POINTER(WGPUImageCopyTexture), data:POINTER(None), dataSize:size_t, dataLayout:POINTER(WGPUTextureDataLayout), writeSize:POINTER(WGPUExtent3D)) -> None: ...
@dll.bind
def wgpuQueueAddRef(queue:WGPUQueue) -> None: ...
@dll.bind
def wgpuQueueRelease(queue:WGPUQueue) -> None: ...
@dll.bind
def wgpuRenderBundleSetLabel(renderBundle:WGPURenderBundle, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderBundleAddRef(renderBundle:WGPURenderBundle) -> None: ...
@dll.bind
def wgpuRenderBundleRelease(renderBundle:WGPURenderBundle) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderDraw(renderBundleEncoder:WGPURenderBundleEncoder, vertexCount:uint32_t, instanceCount:uint32_t, firstVertex:uint32_t, firstInstance:uint32_t) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderDrawIndexed(renderBundleEncoder:WGPURenderBundleEncoder, indexCount:uint32_t, instanceCount:uint32_t, firstIndex:uint32_t, baseVertex:int32_t, firstInstance:uint32_t) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderDrawIndexedIndirect(renderBundleEncoder:WGPURenderBundleEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderDrawIndirect(renderBundleEncoder:WGPURenderBundleEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderFinish(renderBundleEncoder:WGPURenderBundleEncoder, descriptor:POINTER(WGPURenderBundleDescriptor)) -> WGPURenderBundle: ...
@dll.bind
def wgpuRenderBundleEncoderInsertDebugMarker(renderBundleEncoder:WGPURenderBundleEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderPopDebugGroup(renderBundleEncoder:WGPURenderBundleEncoder) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderPushDebugGroup(renderBundleEncoder:WGPURenderBundleEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderSetBindGroup(renderBundleEncoder:WGPURenderBundleEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:POINTER(uint32_t)) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderSetIndexBuffer(renderBundleEncoder:WGPURenderBundleEncoder, buffer:WGPUBuffer, format:WGPUIndexFormat, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderSetLabel(renderBundleEncoder:WGPURenderBundleEncoder, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderSetPipeline(renderBundleEncoder:WGPURenderBundleEncoder, pipeline:WGPURenderPipeline) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderSetVertexBuffer(renderBundleEncoder:WGPURenderBundleEncoder, slot:uint32_t, buffer:WGPUBuffer, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderAddRef(renderBundleEncoder:WGPURenderBundleEncoder) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderRelease(renderBundleEncoder:WGPURenderBundleEncoder) -> None: ...
@dll.bind
def wgpuRenderPassEncoderBeginOcclusionQuery(renderPassEncoder:WGPURenderPassEncoder, queryIndex:uint32_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderDraw(renderPassEncoder:WGPURenderPassEncoder, vertexCount:uint32_t, instanceCount:uint32_t, firstVertex:uint32_t, firstInstance:uint32_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderDrawIndexed(renderPassEncoder:WGPURenderPassEncoder, indexCount:uint32_t, instanceCount:uint32_t, firstIndex:uint32_t, baseVertex:int32_t, firstInstance:uint32_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderDrawIndexedIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderDrawIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderEnd(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind
def wgpuRenderPassEncoderEndOcclusionQuery(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind
def wgpuRenderPassEncoderExecuteBundles(renderPassEncoder:WGPURenderPassEncoder, bundleCount:size_t, bundles:POINTER(WGPURenderBundle)) -> None: ...
@dll.bind
def wgpuRenderPassEncoderInsertDebugMarker(renderPassEncoder:WGPURenderPassEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderPassEncoderMultiDrawIndexedIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t, maxDrawCount:uint32_t, drawCountBuffer:WGPUBuffer, drawCountBufferOffset:uint64_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderMultiDrawIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:uint64_t, maxDrawCount:uint32_t, drawCountBuffer:WGPUBuffer, drawCountBufferOffset:uint64_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderPixelLocalStorageBarrier(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind
def wgpuRenderPassEncoderPopDebugGroup(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind
def wgpuRenderPassEncoderPushDebugGroup(renderPassEncoder:WGPURenderPassEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetBindGroup(renderPassEncoder:WGPURenderPassEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:POINTER(uint32_t)) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetBlendConstant(renderPassEncoder:WGPURenderPassEncoder, color:POINTER(WGPUColor)) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder:WGPURenderPassEncoder, buffer:WGPUBuffer, format:WGPUIndexFormat, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetLabel(renderPassEncoder:WGPURenderPassEncoder, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetPipeline(renderPassEncoder:WGPURenderPassEncoder, pipeline:WGPURenderPipeline) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetScissorRect(renderPassEncoder:WGPURenderPassEncoder, x:uint32_t, y:uint32_t, width:uint32_t, height:uint32_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetStencilReference(renderPassEncoder:WGPURenderPassEncoder, reference:uint32_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder:WGPURenderPassEncoder, slot:uint32_t, buffer:WGPUBuffer, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetViewport(renderPassEncoder:WGPURenderPassEncoder, x:Annotated[float, ctypes.c_float], y:Annotated[float, ctypes.c_float], width:Annotated[float, ctypes.c_float], height:Annotated[float, ctypes.c_float], minDepth:Annotated[float, ctypes.c_float], maxDepth:Annotated[float, ctypes.c_float]) -> None: ...
@dll.bind
def wgpuRenderPassEncoderWriteTimestamp(renderPassEncoder:WGPURenderPassEncoder, querySet:WGPUQuerySet, queryIndex:uint32_t) -> None: ...
@dll.bind
def wgpuRenderPassEncoderAddRef(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind
def wgpuRenderPassEncoderRelease(renderPassEncoder:WGPURenderPassEncoder) -> None: ...
@dll.bind
def wgpuRenderPipelineGetBindGroupLayout(renderPipeline:WGPURenderPipeline, groupIndex:uint32_t) -> WGPUBindGroupLayout: ...
@dll.bind
def wgpuRenderPipelineSetLabel(renderPipeline:WGPURenderPipeline, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderPipelineAddRef(renderPipeline:WGPURenderPipeline) -> None: ...
@dll.bind
def wgpuRenderPipelineRelease(renderPipeline:WGPURenderPipeline) -> None: ...
@dll.bind
def wgpuSamplerSetLabel(sampler:WGPUSampler, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuSamplerAddRef(sampler:WGPUSampler) -> None: ...
@dll.bind
def wgpuSamplerRelease(sampler:WGPUSampler) -> None: ...
@dll.bind
def wgpuShaderModuleGetCompilationInfo(shaderModule:WGPUShaderModule, callback:WGPUCompilationInfoCallback, userdata:POINTER(None)) -> None: ...
@dll.bind
def wgpuShaderModuleGetCompilationInfo2(shaderModule:WGPUShaderModule, callbackInfo:WGPUCompilationInfoCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuShaderModuleGetCompilationInfoF(shaderModule:WGPUShaderModule, callbackInfo:WGPUCompilationInfoCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuShaderModuleSetLabel(shaderModule:WGPUShaderModule, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuShaderModuleAddRef(shaderModule:WGPUShaderModule) -> None: ...
@dll.bind
def wgpuShaderModuleRelease(shaderModule:WGPUShaderModule) -> None: ...
@dll.bind
def wgpuSharedBufferMemoryBeginAccess(sharedBufferMemory:WGPUSharedBufferMemory, buffer:WGPUBuffer, descriptor:POINTER(WGPUSharedBufferMemoryBeginAccessDescriptor)) -> WGPUStatus: ...
@dll.bind
def wgpuSharedBufferMemoryCreateBuffer(sharedBufferMemory:WGPUSharedBufferMemory, descriptor:POINTER(WGPUBufferDescriptor)) -> WGPUBuffer: ...
@dll.bind
def wgpuSharedBufferMemoryEndAccess(sharedBufferMemory:WGPUSharedBufferMemory, buffer:WGPUBuffer, descriptor:POINTER(WGPUSharedBufferMemoryEndAccessState)) -> WGPUStatus: ...
@dll.bind
def wgpuSharedBufferMemoryGetProperties(sharedBufferMemory:WGPUSharedBufferMemory, properties:POINTER(WGPUSharedBufferMemoryProperties)) -> WGPUStatus: ...
@dll.bind
def wgpuSharedBufferMemoryIsDeviceLost(sharedBufferMemory:WGPUSharedBufferMemory) -> WGPUBool: ...
@dll.bind
def wgpuSharedBufferMemorySetLabel(sharedBufferMemory:WGPUSharedBufferMemory, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuSharedBufferMemoryAddRef(sharedBufferMemory:WGPUSharedBufferMemory) -> None: ...
@dll.bind
def wgpuSharedBufferMemoryRelease(sharedBufferMemory:WGPUSharedBufferMemory) -> None: ...
@dll.bind
def wgpuSharedFenceExportInfo(sharedFence:WGPUSharedFence, info:POINTER(WGPUSharedFenceExportInfo)) -> None: ...
@dll.bind
def wgpuSharedFenceAddRef(sharedFence:WGPUSharedFence) -> None: ...
@dll.bind
def wgpuSharedFenceRelease(sharedFence:WGPUSharedFence) -> None: ...
@dll.bind
def wgpuSharedTextureMemoryBeginAccess(sharedTextureMemory:WGPUSharedTextureMemory, texture:WGPUTexture, descriptor:POINTER(WGPUSharedTextureMemoryBeginAccessDescriptor)) -> WGPUStatus: ...
@dll.bind
def wgpuSharedTextureMemoryCreateTexture(sharedTextureMemory:WGPUSharedTextureMemory, descriptor:POINTER(WGPUTextureDescriptor)) -> WGPUTexture: ...
@dll.bind
def wgpuSharedTextureMemoryEndAccess(sharedTextureMemory:WGPUSharedTextureMemory, texture:WGPUTexture, descriptor:POINTER(WGPUSharedTextureMemoryEndAccessState)) -> WGPUStatus: ...
@dll.bind
def wgpuSharedTextureMemoryGetProperties(sharedTextureMemory:WGPUSharedTextureMemory, properties:POINTER(WGPUSharedTextureMemoryProperties)) -> WGPUStatus: ...
@dll.bind
def wgpuSharedTextureMemoryIsDeviceLost(sharedTextureMemory:WGPUSharedTextureMemory) -> WGPUBool: ...
@dll.bind
def wgpuSharedTextureMemorySetLabel(sharedTextureMemory:WGPUSharedTextureMemory, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuSharedTextureMemoryAddRef(sharedTextureMemory:WGPUSharedTextureMemory) -> None: ...
@dll.bind
def wgpuSharedTextureMemoryRelease(sharedTextureMemory:WGPUSharedTextureMemory) -> None: ...
@dll.bind
def wgpuSurfaceConfigure(surface:WGPUSurface, config:POINTER(WGPUSurfaceConfiguration)) -> None: ...
@dll.bind
def wgpuSurfaceGetCapabilities(surface:WGPUSurface, adapter:WGPUAdapter, capabilities:POINTER(WGPUSurfaceCapabilities)) -> WGPUStatus: ...
@dll.bind
def wgpuSurfaceGetCurrentTexture(surface:WGPUSurface, surfaceTexture:POINTER(WGPUSurfaceTexture)) -> None: ...
@dll.bind
def wgpuSurfacePresent(surface:WGPUSurface) -> None: ...
@dll.bind
def wgpuSurfaceSetLabel(surface:WGPUSurface, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuSurfaceUnconfigure(surface:WGPUSurface) -> None: ...
@dll.bind
def wgpuSurfaceAddRef(surface:WGPUSurface) -> None: ...
@dll.bind
def wgpuSurfaceRelease(surface:WGPUSurface) -> None: ...
@dll.bind
def wgpuTextureCreateErrorView(texture:WGPUTexture, descriptor:POINTER(WGPUTextureViewDescriptor)) -> WGPUTextureView: ...
@dll.bind
def wgpuTextureCreateView(texture:WGPUTexture, descriptor:POINTER(WGPUTextureViewDescriptor)) -> WGPUTextureView: ...
@dll.bind
def wgpuTextureDestroy(texture:WGPUTexture) -> None: ...
@dll.bind
def wgpuTextureGetDepthOrArrayLayers(texture:WGPUTexture) -> uint32_t: ...
@dll.bind
def wgpuTextureGetDimension(texture:WGPUTexture) -> WGPUTextureDimension: ...
@dll.bind
def wgpuTextureGetFormat(texture:WGPUTexture) -> WGPUTextureFormat: ...
@dll.bind
def wgpuTextureGetHeight(texture:WGPUTexture) -> uint32_t: ...
@dll.bind
def wgpuTextureGetMipLevelCount(texture:WGPUTexture) -> uint32_t: ...
@dll.bind
def wgpuTextureGetSampleCount(texture:WGPUTexture) -> uint32_t: ...
@dll.bind
def wgpuTextureGetUsage(texture:WGPUTexture) -> WGPUTextureUsage: ...
@dll.bind
def wgpuTextureGetWidth(texture:WGPUTexture) -> uint32_t: ...
@dll.bind
def wgpuTextureSetLabel(texture:WGPUTexture, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuTextureAddRef(texture:WGPUTexture) -> None: ...
@dll.bind
def wgpuTextureRelease(texture:WGPUTexture) -> None: ...
@dll.bind
def wgpuTextureViewSetLabel(textureView:WGPUTextureView, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuTextureViewAddRef(textureView:WGPUTextureView) -> None: ...
@dll.bind
def wgpuTextureViewRelease(textureView:WGPUTextureView) -> None: ...
init_records()
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