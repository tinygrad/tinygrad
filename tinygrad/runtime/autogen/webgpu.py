# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
from tinygrad.helpers import WIN, OSX
import sysconfig, os
dll = c.DLL('webgpu', os.path.join(sysconfig.get_paths()['purelib'], 'pydawn', 'lib', 'libwebgpu_dawn.dll') if WIN else 'webgpu_dawn')
WGPUFlags = Annotated[int, ctypes.c_uint64]
WGPUBool = Annotated[int, ctypes.c_uint32]
class struct_WGPUAdapterImpl(ctypes.Structure): pass
WGPUAdapter = c.POINTER[struct_WGPUAdapterImpl]
class struct_WGPUBindGroupImpl(ctypes.Structure): pass
WGPUBindGroup = c.POINTER[struct_WGPUBindGroupImpl]
class struct_WGPUBindGroupLayoutImpl(ctypes.Structure): pass
WGPUBindGroupLayout = c.POINTER[struct_WGPUBindGroupLayoutImpl]
class struct_WGPUBufferImpl(ctypes.Structure): pass
WGPUBuffer = c.POINTER[struct_WGPUBufferImpl]
class struct_WGPUCommandBufferImpl(ctypes.Structure): pass
WGPUCommandBuffer = c.POINTER[struct_WGPUCommandBufferImpl]
class struct_WGPUCommandEncoderImpl(ctypes.Structure): pass
WGPUCommandEncoder = c.POINTER[struct_WGPUCommandEncoderImpl]
class struct_WGPUComputePassEncoderImpl(ctypes.Structure): pass
WGPUComputePassEncoder = c.POINTER[struct_WGPUComputePassEncoderImpl]
class struct_WGPUComputePipelineImpl(ctypes.Structure): pass
WGPUComputePipeline = c.POINTER[struct_WGPUComputePipelineImpl]
class struct_WGPUDeviceImpl(ctypes.Structure): pass
WGPUDevice = c.POINTER[struct_WGPUDeviceImpl]
class struct_WGPUExternalTextureImpl(ctypes.Structure): pass
WGPUExternalTexture = c.POINTER[struct_WGPUExternalTextureImpl]
class struct_WGPUInstanceImpl(ctypes.Structure): pass
WGPUInstance = c.POINTER[struct_WGPUInstanceImpl]
class struct_WGPUPipelineLayoutImpl(ctypes.Structure): pass
WGPUPipelineLayout = c.POINTER[struct_WGPUPipelineLayoutImpl]
class struct_WGPUQuerySetImpl(ctypes.Structure): pass
WGPUQuerySet = c.POINTER[struct_WGPUQuerySetImpl]
class struct_WGPUQueueImpl(ctypes.Structure): pass
WGPUQueue = c.POINTER[struct_WGPUQueueImpl]
class struct_WGPURenderBundleImpl(ctypes.Structure): pass
WGPURenderBundle = c.POINTER[struct_WGPURenderBundleImpl]
class struct_WGPURenderBundleEncoderImpl(ctypes.Structure): pass
WGPURenderBundleEncoder = c.POINTER[struct_WGPURenderBundleEncoderImpl]
class struct_WGPURenderPassEncoderImpl(ctypes.Structure): pass
WGPURenderPassEncoder = c.POINTER[struct_WGPURenderPassEncoderImpl]
class struct_WGPURenderPipelineImpl(ctypes.Structure): pass
WGPURenderPipeline = c.POINTER[struct_WGPURenderPipelineImpl]
class struct_WGPUSamplerImpl(ctypes.Structure): pass
WGPUSampler = c.POINTER[struct_WGPUSamplerImpl]
class struct_WGPUShaderModuleImpl(ctypes.Structure): pass
WGPUShaderModule = c.POINTER[struct_WGPUShaderModuleImpl]
class struct_WGPUSharedBufferMemoryImpl(ctypes.Structure): pass
WGPUSharedBufferMemory = c.POINTER[struct_WGPUSharedBufferMemoryImpl]
class struct_WGPUSharedFenceImpl(ctypes.Structure): pass
WGPUSharedFence = c.POINTER[struct_WGPUSharedFenceImpl]
class struct_WGPUSharedTextureMemoryImpl(ctypes.Structure): pass
WGPUSharedTextureMemory = c.POINTER[struct_WGPUSharedTextureMemoryImpl]
class struct_WGPUSurfaceImpl(ctypes.Structure): pass
WGPUSurface = c.POINTER[struct_WGPUSurfaceImpl]
class struct_WGPUTextureImpl(ctypes.Structure): pass
WGPUTexture = c.POINTER[struct_WGPUTextureImpl]
class struct_WGPUTextureViewImpl(ctypes.Structure): pass
WGPUTextureView = c.POINTER[struct_WGPUTextureViewImpl]
@c.record
class struct_WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER(c.Struct):
  SIZE = 4
  unused: Annotated[WGPUBool, 0]
@c.record
class struct_WGPUAdapterPropertiesD3D(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  shaderModel: Annotated[uint32_t, 16]
@c.record
class struct_WGPUChainedStructOut(c.Struct):
  SIZE = 16
  next: Annotated[c.POINTER[struct_WGPUChainedStructOut], 0]
  sType: Annotated[WGPUSType, 8]
WGPUChainedStructOut: TypeAlias = struct_WGPUChainedStructOut
enum_WGPUSType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUSType_ShaderSourceSPIRV = enum_WGPUSType.define('WGPUSType_ShaderSourceSPIRV', 1) # type: ignore
WGPUSType_ShaderSourceWGSL = enum_WGPUSType.define('WGPUSType_ShaderSourceWGSL', 2) # type: ignore
WGPUSType_RenderPassMaxDrawCount = enum_WGPUSType.define('WGPUSType_RenderPassMaxDrawCount', 3) # type: ignore
WGPUSType_SurfaceSourceMetalLayer = enum_WGPUSType.define('WGPUSType_SurfaceSourceMetalLayer', 4) # type: ignore
WGPUSType_SurfaceSourceWindowsHWND = enum_WGPUSType.define('WGPUSType_SurfaceSourceWindowsHWND', 5) # type: ignore
WGPUSType_SurfaceSourceXlibWindow = enum_WGPUSType.define('WGPUSType_SurfaceSourceXlibWindow', 6) # type: ignore
WGPUSType_SurfaceSourceWaylandSurface = enum_WGPUSType.define('WGPUSType_SurfaceSourceWaylandSurface', 7) # type: ignore
WGPUSType_SurfaceSourceAndroidNativeWindow = enum_WGPUSType.define('WGPUSType_SurfaceSourceAndroidNativeWindow', 8) # type: ignore
WGPUSType_SurfaceSourceXCBWindow = enum_WGPUSType.define('WGPUSType_SurfaceSourceXCBWindow', 9) # type: ignore
WGPUSType_AdapterPropertiesSubgroups = enum_WGPUSType.define('WGPUSType_AdapterPropertiesSubgroups', 10) # type: ignore
WGPUSType_TextureBindingViewDimensionDescriptor = enum_WGPUSType.define('WGPUSType_TextureBindingViewDimensionDescriptor', 131072) # type: ignore
WGPUSType_SurfaceSourceCanvasHTMLSelector_Emscripten = enum_WGPUSType.define('WGPUSType_SurfaceSourceCanvasHTMLSelector_Emscripten', 262144) # type: ignore
WGPUSType_SurfaceDescriptorFromWindowsCoreWindow = enum_WGPUSType.define('WGPUSType_SurfaceDescriptorFromWindowsCoreWindow', 327680) # type: ignore
WGPUSType_ExternalTextureBindingEntry = enum_WGPUSType.define('WGPUSType_ExternalTextureBindingEntry', 327681) # type: ignore
WGPUSType_ExternalTextureBindingLayout = enum_WGPUSType.define('WGPUSType_ExternalTextureBindingLayout', 327682) # type: ignore
WGPUSType_SurfaceDescriptorFromWindowsSwapChainPanel = enum_WGPUSType.define('WGPUSType_SurfaceDescriptorFromWindowsSwapChainPanel', 327683) # type: ignore
WGPUSType_DawnTextureInternalUsageDescriptor = enum_WGPUSType.define('WGPUSType_DawnTextureInternalUsageDescriptor', 327684) # type: ignore
WGPUSType_DawnEncoderInternalUsageDescriptor = enum_WGPUSType.define('WGPUSType_DawnEncoderInternalUsageDescriptor', 327685) # type: ignore
WGPUSType_DawnInstanceDescriptor = enum_WGPUSType.define('WGPUSType_DawnInstanceDescriptor', 327686) # type: ignore
WGPUSType_DawnCacheDeviceDescriptor = enum_WGPUSType.define('WGPUSType_DawnCacheDeviceDescriptor', 327687) # type: ignore
WGPUSType_DawnAdapterPropertiesPowerPreference = enum_WGPUSType.define('WGPUSType_DawnAdapterPropertiesPowerPreference', 327688) # type: ignore
WGPUSType_DawnBufferDescriptorErrorInfoFromWireClient = enum_WGPUSType.define('WGPUSType_DawnBufferDescriptorErrorInfoFromWireClient', 327689) # type: ignore
WGPUSType_DawnTogglesDescriptor = enum_WGPUSType.define('WGPUSType_DawnTogglesDescriptor', 327690) # type: ignore
WGPUSType_DawnShaderModuleSPIRVOptionsDescriptor = enum_WGPUSType.define('WGPUSType_DawnShaderModuleSPIRVOptionsDescriptor', 327691) # type: ignore
WGPUSType_RequestAdapterOptionsLUID = enum_WGPUSType.define('WGPUSType_RequestAdapterOptionsLUID', 327692) # type: ignore
WGPUSType_RequestAdapterOptionsGetGLProc = enum_WGPUSType.define('WGPUSType_RequestAdapterOptionsGetGLProc', 327693) # type: ignore
WGPUSType_RequestAdapterOptionsD3D11Device = enum_WGPUSType.define('WGPUSType_RequestAdapterOptionsD3D11Device', 327694) # type: ignore
WGPUSType_DawnRenderPassColorAttachmentRenderToSingleSampled = enum_WGPUSType.define('WGPUSType_DawnRenderPassColorAttachmentRenderToSingleSampled', 327695) # type: ignore
WGPUSType_RenderPassPixelLocalStorage = enum_WGPUSType.define('WGPUSType_RenderPassPixelLocalStorage', 327696) # type: ignore
WGPUSType_PipelineLayoutPixelLocalStorage = enum_WGPUSType.define('WGPUSType_PipelineLayoutPixelLocalStorage', 327697) # type: ignore
WGPUSType_BufferHostMappedPointer = enum_WGPUSType.define('WGPUSType_BufferHostMappedPointer', 327698) # type: ignore
WGPUSType_DawnExperimentalSubgroupLimits = enum_WGPUSType.define('WGPUSType_DawnExperimentalSubgroupLimits', 327699) # type: ignore
WGPUSType_AdapterPropertiesMemoryHeaps = enum_WGPUSType.define('WGPUSType_AdapterPropertiesMemoryHeaps', 327700) # type: ignore
WGPUSType_AdapterPropertiesD3D = enum_WGPUSType.define('WGPUSType_AdapterPropertiesD3D', 327701) # type: ignore
WGPUSType_AdapterPropertiesVk = enum_WGPUSType.define('WGPUSType_AdapterPropertiesVk', 327702) # type: ignore
WGPUSType_DawnWireWGSLControl = enum_WGPUSType.define('WGPUSType_DawnWireWGSLControl', 327703) # type: ignore
WGPUSType_DawnWGSLBlocklist = enum_WGPUSType.define('WGPUSType_DawnWGSLBlocklist', 327704) # type: ignore
WGPUSType_DrmFormatCapabilities = enum_WGPUSType.define('WGPUSType_DrmFormatCapabilities', 327705) # type: ignore
WGPUSType_ShaderModuleCompilationOptions = enum_WGPUSType.define('WGPUSType_ShaderModuleCompilationOptions', 327706) # type: ignore
WGPUSType_ColorTargetStateExpandResolveTextureDawn = enum_WGPUSType.define('WGPUSType_ColorTargetStateExpandResolveTextureDawn', 327707) # type: ignore
WGPUSType_RenderPassDescriptorExpandResolveRect = enum_WGPUSType.define('WGPUSType_RenderPassDescriptorExpandResolveRect', 327708) # type: ignore
WGPUSType_SharedTextureMemoryVkDedicatedAllocationDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryVkDedicatedAllocationDescriptor', 327709) # type: ignore
WGPUSType_SharedTextureMemoryAHardwareBufferDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryAHardwareBufferDescriptor', 327710) # type: ignore
WGPUSType_SharedTextureMemoryDmaBufDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryDmaBufDescriptor', 327711) # type: ignore
WGPUSType_SharedTextureMemoryOpaqueFDDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryOpaqueFDDescriptor', 327712) # type: ignore
WGPUSType_SharedTextureMemoryZirconHandleDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryZirconHandleDescriptor', 327713) # type: ignore
WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor', 327714) # type: ignore
WGPUSType_SharedTextureMemoryD3D11Texture2DDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryD3D11Texture2DDescriptor', 327715) # type: ignore
WGPUSType_SharedTextureMemoryIOSurfaceDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryIOSurfaceDescriptor', 327716) # type: ignore
WGPUSType_SharedTextureMemoryEGLImageDescriptor = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryEGLImageDescriptor', 327717) # type: ignore
WGPUSType_SharedTextureMemoryInitializedBeginState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryInitializedBeginState', 327718) # type: ignore
WGPUSType_SharedTextureMemoryInitializedEndState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryInitializedEndState', 327719) # type: ignore
WGPUSType_SharedTextureMemoryVkImageLayoutBeginState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryVkImageLayoutBeginState', 327720) # type: ignore
WGPUSType_SharedTextureMemoryVkImageLayoutEndState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryVkImageLayoutEndState', 327721) # type: ignore
WGPUSType_SharedTextureMemoryD3DSwapchainBeginState = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryD3DSwapchainBeginState', 327722) # type: ignore
WGPUSType_SharedFenceVkSemaphoreOpaqueFDDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceVkSemaphoreOpaqueFDDescriptor', 327723) # type: ignore
WGPUSType_SharedFenceVkSemaphoreOpaqueFDExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceVkSemaphoreOpaqueFDExportInfo', 327724) # type: ignore
WGPUSType_SharedFenceSyncFDDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceSyncFDDescriptor', 327725) # type: ignore
WGPUSType_SharedFenceSyncFDExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceSyncFDExportInfo', 327726) # type: ignore
WGPUSType_SharedFenceVkSemaphoreZirconHandleDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceVkSemaphoreZirconHandleDescriptor', 327727) # type: ignore
WGPUSType_SharedFenceVkSemaphoreZirconHandleExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceVkSemaphoreZirconHandleExportInfo', 327728) # type: ignore
WGPUSType_SharedFenceDXGISharedHandleDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceDXGISharedHandleDescriptor', 327729) # type: ignore
WGPUSType_SharedFenceDXGISharedHandleExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceDXGISharedHandleExportInfo', 327730) # type: ignore
WGPUSType_SharedFenceMTLSharedEventDescriptor = enum_WGPUSType.define('WGPUSType_SharedFenceMTLSharedEventDescriptor', 327731) # type: ignore
WGPUSType_SharedFenceMTLSharedEventExportInfo = enum_WGPUSType.define('WGPUSType_SharedFenceMTLSharedEventExportInfo', 327732) # type: ignore
WGPUSType_SharedBufferMemoryD3D12ResourceDescriptor = enum_WGPUSType.define('WGPUSType_SharedBufferMemoryD3D12ResourceDescriptor', 327733) # type: ignore
WGPUSType_StaticSamplerBindingLayout = enum_WGPUSType.define('WGPUSType_StaticSamplerBindingLayout', 327734) # type: ignore
WGPUSType_YCbCrVkDescriptor = enum_WGPUSType.define('WGPUSType_YCbCrVkDescriptor', 327735) # type: ignore
WGPUSType_SharedTextureMemoryAHardwareBufferProperties = enum_WGPUSType.define('WGPUSType_SharedTextureMemoryAHardwareBufferProperties', 327736) # type: ignore
WGPUSType_AHardwareBufferProperties = enum_WGPUSType.define('WGPUSType_AHardwareBufferProperties', 327737) # type: ignore
WGPUSType_DawnExperimentalImmediateDataLimits = enum_WGPUSType.define('WGPUSType_DawnExperimentalImmediateDataLimits', 327738) # type: ignore
WGPUSType_DawnTexelCopyBufferRowAlignmentLimits = enum_WGPUSType.define('WGPUSType_DawnTexelCopyBufferRowAlignmentLimits', 327739) # type: ignore
WGPUSType_Force32 = enum_WGPUSType.define('WGPUSType_Force32', 2147483647) # type: ignore

WGPUSType: TypeAlias = enum_WGPUSType
uint32_t = Annotated[int, ctypes.c_uint32]
@c.record
class struct_WGPUAdapterPropertiesSubgroups(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  subgroupMinSize: Annotated[uint32_t, 16]
  subgroupMaxSize: Annotated[uint32_t, 20]
@c.record
class struct_WGPUAdapterPropertiesVk(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  driverVersion: Annotated[uint32_t, 16]
@c.record
class struct_WGPUBindGroupEntry(c.Struct):
  SIZE = 56
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  binding: Annotated[uint32_t, 8]
  buffer: Annotated[WGPUBuffer, 16]
  offset: Annotated[uint64_t, 24]
  size: Annotated[uint64_t, 32]
  sampler: Annotated[WGPUSampler, 40]
  textureView: Annotated[WGPUTextureView, 48]
@c.record
class struct_WGPUChainedStruct(c.Struct):
  SIZE = 16
  next: Annotated[c.POINTER[struct_WGPUChainedStruct], 0]
  sType: Annotated[WGPUSType, 8]
WGPUChainedStruct: TypeAlias = struct_WGPUChainedStruct
uint64_t = Annotated[int, ctypes.c_uint64]
@c.record
class struct_WGPUBlendComponent(c.Struct):
  SIZE = 12
  operation: Annotated[WGPUBlendOperation, 0]
  srcFactor: Annotated[WGPUBlendFactor, 4]
  dstFactor: Annotated[WGPUBlendFactor, 8]
enum_WGPUBlendOperation = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBlendOperation_Undefined = enum_WGPUBlendOperation.define('WGPUBlendOperation_Undefined', 0) # type: ignore
WGPUBlendOperation_Add = enum_WGPUBlendOperation.define('WGPUBlendOperation_Add', 1) # type: ignore
WGPUBlendOperation_Subtract = enum_WGPUBlendOperation.define('WGPUBlendOperation_Subtract', 2) # type: ignore
WGPUBlendOperation_ReverseSubtract = enum_WGPUBlendOperation.define('WGPUBlendOperation_ReverseSubtract', 3) # type: ignore
WGPUBlendOperation_Min = enum_WGPUBlendOperation.define('WGPUBlendOperation_Min', 4) # type: ignore
WGPUBlendOperation_Max = enum_WGPUBlendOperation.define('WGPUBlendOperation_Max', 5) # type: ignore
WGPUBlendOperation_Force32 = enum_WGPUBlendOperation.define('WGPUBlendOperation_Force32', 2147483647) # type: ignore

WGPUBlendOperation: TypeAlias = enum_WGPUBlendOperation
enum_WGPUBlendFactor = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBlendFactor_Undefined = enum_WGPUBlendFactor.define('WGPUBlendFactor_Undefined', 0) # type: ignore
WGPUBlendFactor_Zero = enum_WGPUBlendFactor.define('WGPUBlendFactor_Zero', 1) # type: ignore
WGPUBlendFactor_One = enum_WGPUBlendFactor.define('WGPUBlendFactor_One', 2) # type: ignore
WGPUBlendFactor_Src = enum_WGPUBlendFactor.define('WGPUBlendFactor_Src', 3) # type: ignore
WGPUBlendFactor_OneMinusSrc = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusSrc', 4) # type: ignore
WGPUBlendFactor_SrcAlpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_SrcAlpha', 5) # type: ignore
WGPUBlendFactor_OneMinusSrcAlpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusSrcAlpha', 6) # type: ignore
WGPUBlendFactor_Dst = enum_WGPUBlendFactor.define('WGPUBlendFactor_Dst', 7) # type: ignore
WGPUBlendFactor_OneMinusDst = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusDst', 8) # type: ignore
WGPUBlendFactor_DstAlpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_DstAlpha', 9) # type: ignore
WGPUBlendFactor_OneMinusDstAlpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusDstAlpha', 10) # type: ignore
WGPUBlendFactor_SrcAlphaSaturated = enum_WGPUBlendFactor.define('WGPUBlendFactor_SrcAlphaSaturated', 11) # type: ignore
WGPUBlendFactor_Constant = enum_WGPUBlendFactor.define('WGPUBlendFactor_Constant', 12) # type: ignore
WGPUBlendFactor_OneMinusConstant = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusConstant', 13) # type: ignore
WGPUBlendFactor_Src1 = enum_WGPUBlendFactor.define('WGPUBlendFactor_Src1', 14) # type: ignore
WGPUBlendFactor_OneMinusSrc1 = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusSrc1', 15) # type: ignore
WGPUBlendFactor_Src1Alpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_Src1Alpha', 16) # type: ignore
WGPUBlendFactor_OneMinusSrc1Alpha = enum_WGPUBlendFactor.define('WGPUBlendFactor_OneMinusSrc1Alpha', 17) # type: ignore
WGPUBlendFactor_Force32 = enum_WGPUBlendFactor.define('WGPUBlendFactor_Force32', 2147483647) # type: ignore

WGPUBlendFactor: TypeAlias = enum_WGPUBlendFactor
@c.record
class struct_WGPUBufferBindingLayout(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  type: Annotated[WGPUBufferBindingType, 8]
  hasDynamicOffset: Annotated[WGPUBool, 12]
  minBindingSize: Annotated[uint64_t, 16]
enum_WGPUBufferBindingType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBufferBindingType_BindingNotUsed = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_BindingNotUsed', 0) # type: ignore
WGPUBufferBindingType_Uniform = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Uniform', 1) # type: ignore
WGPUBufferBindingType_Storage = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Storage', 2) # type: ignore
WGPUBufferBindingType_ReadOnlyStorage = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_ReadOnlyStorage', 3) # type: ignore
WGPUBufferBindingType_Force32 = enum_WGPUBufferBindingType.define('WGPUBufferBindingType_Force32', 2147483647) # type: ignore

WGPUBufferBindingType: TypeAlias = enum_WGPUBufferBindingType
@c.record
class struct_WGPUBufferHostMappedPointer(c.Struct):
  SIZE = 40
  chain: Annotated[WGPUChainedStruct, 0]
  pointer: Annotated[c.POINTER[None], 16]
  disposeCallback: Annotated[WGPUCallback, 24]
  userdata: Annotated[c.POINTER[None], 32]
WGPUCallback = c.CFUNCTYPE(None, c.POINTER[None])
@c.record
class struct_WGPUBufferMapCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUBufferMapCallback, 16]
  userdata: Annotated[c.POINTER[None], 24]
enum_WGPUCallbackMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCallbackMode_WaitAnyOnly = enum_WGPUCallbackMode.define('WGPUCallbackMode_WaitAnyOnly', 1) # type: ignore
WGPUCallbackMode_AllowProcessEvents = enum_WGPUCallbackMode.define('WGPUCallbackMode_AllowProcessEvents', 2) # type: ignore
WGPUCallbackMode_AllowSpontaneous = enum_WGPUCallbackMode.define('WGPUCallbackMode_AllowSpontaneous', 3) # type: ignore
WGPUCallbackMode_Force32 = enum_WGPUCallbackMode.define('WGPUCallbackMode_Force32', 2147483647) # type: ignore

WGPUCallbackMode: TypeAlias = enum_WGPUCallbackMode
enum_WGPUBufferMapAsyncStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBufferMapAsyncStatus_Success = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_Success', 1) # type: ignore
WGPUBufferMapAsyncStatus_InstanceDropped = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_InstanceDropped', 2) # type: ignore
WGPUBufferMapAsyncStatus_ValidationError = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_ValidationError', 3) # type: ignore
WGPUBufferMapAsyncStatus_Unknown = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_Unknown', 4) # type: ignore
WGPUBufferMapAsyncStatus_DeviceLost = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_DeviceLost', 5) # type: ignore
WGPUBufferMapAsyncStatus_DestroyedBeforeCallback = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_DestroyedBeforeCallback', 6) # type: ignore
WGPUBufferMapAsyncStatus_UnmappedBeforeCallback = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_UnmappedBeforeCallback', 7) # type: ignore
WGPUBufferMapAsyncStatus_MappingAlreadyPending = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_MappingAlreadyPending', 8) # type: ignore
WGPUBufferMapAsyncStatus_OffsetOutOfRange = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_OffsetOutOfRange', 9) # type: ignore
WGPUBufferMapAsyncStatus_SizeOutOfRange = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_SizeOutOfRange', 10) # type: ignore
WGPUBufferMapAsyncStatus_Force32 = enum_WGPUBufferMapAsyncStatus.define('WGPUBufferMapAsyncStatus_Force32', 2147483647) # type: ignore

WGPUBufferMapCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPUBufferMapAsyncStatus, c.POINTER[None])
@c.record
class struct_WGPUColor(c.Struct):
  SIZE = 32
  r: Annotated[Annotated[float, ctypes.c_double], 0]
  g: Annotated[Annotated[float, ctypes.c_double], 8]
  b: Annotated[Annotated[float, ctypes.c_double], 16]
  a: Annotated[Annotated[float, ctypes.c_double], 24]
@c.record
class struct_WGPUColorTargetStateExpandResolveTextureDawn(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  enabled: Annotated[WGPUBool, 16]
@c.record
class struct_WGPUCompilationInfoCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCompilationInfoCallback, 16]
  userdata: Annotated[c.POINTER[None], 24]
enum_WGPUCompilationInfoRequestStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCompilationInfoRequestStatus_Success = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Success', 1) # type: ignore
WGPUCompilationInfoRequestStatus_InstanceDropped = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_InstanceDropped', 2) # type: ignore
WGPUCompilationInfoRequestStatus_Error = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Error', 3) # type: ignore
WGPUCompilationInfoRequestStatus_DeviceLost = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_DeviceLost', 4) # type: ignore
WGPUCompilationInfoRequestStatus_Unknown = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Unknown', 5) # type: ignore
WGPUCompilationInfoRequestStatus_Force32 = enum_WGPUCompilationInfoRequestStatus.define('WGPUCompilationInfoRequestStatus_Force32', 2147483647) # type: ignore

@c.record
class struct_WGPUCompilationInfo(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  messageCount: Annotated[size_t, 8]
  messages: Annotated[c.POINTER[WGPUCompilationMessage], 16]
size_t = Annotated[int, ctypes.c_uint64]
@c.record
class struct_WGPUCompilationMessage(c.Struct):
  SIZE = 88
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
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
@c.record
class struct_WGPUStringView(c.Struct):
  SIZE = 16
  data: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 0]
  length: Annotated[size_t, 8]
WGPUStringView: TypeAlias = struct_WGPUStringView
enum_WGPUCompilationMessageType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCompilationMessageType_Error = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Error', 1) # type: ignore
WGPUCompilationMessageType_Warning = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Warning', 2) # type: ignore
WGPUCompilationMessageType_Info = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Info', 3) # type: ignore
WGPUCompilationMessageType_Force32 = enum_WGPUCompilationMessageType.define('WGPUCompilationMessageType_Force32', 2147483647) # type: ignore

WGPUCompilationMessageType: TypeAlias = enum_WGPUCompilationMessageType
WGPUCompilationInfoCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, c.POINTER[struct_WGPUCompilationInfo], c.POINTER[None])
@c.record
class struct_WGPUComputePassTimestampWrites(c.Struct):
  SIZE = 16
  querySet: Annotated[WGPUQuerySet, 0]
  beginningOfPassWriteIndex: Annotated[uint32_t, 8]
  endOfPassWriteIndex: Annotated[uint32_t, 12]
@c.record
class struct_WGPUCopyTextureForBrowserOptions(c.Struct):
  SIZE = 56
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  flipY: Annotated[WGPUBool, 8]
  needsColorSpaceConversion: Annotated[WGPUBool, 12]
  srcAlphaMode: Annotated[WGPUAlphaMode, 16]
  srcTransferFunctionParameters: Annotated[c.POINTER[Annotated[float, ctypes.c_float]], 24]
  conversionMatrix: Annotated[c.POINTER[Annotated[float, ctypes.c_float]], 32]
  dstTransferFunctionParameters: Annotated[c.POINTER[Annotated[float, ctypes.c_float]], 40]
  dstAlphaMode: Annotated[WGPUAlphaMode, 48]
  internalUsage: Annotated[WGPUBool, 52]
enum_WGPUAlphaMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUAlphaMode_Opaque = enum_WGPUAlphaMode.define('WGPUAlphaMode_Opaque', 1) # type: ignore
WGPUAlphaMode_Premultiplied = enum_WGPUAlphaMode.define('WGPUAlphaMode_Premultiplied', 2) # type: ignore
WGPUAlphaMode_Unpremultiplied = enum_WGPUAlphaMode.define('WGPUAlphaMode_Unpremultiplied', 3) # type: ignore
WGPUAlphaMode_Force32 = enum_WGPUAlphaMode.define('WGPUAlphaMode_Force32', 2147483647) # type: ignore

WGPUAlphaMode: TypeAlias = enum_WGPUAlphaMode
@c.record
class struct_WGPUCreateComputePipelineAsyncCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCreateComputePipelineAsyncCallback, 16]
  userdata: Annotated[c.POINTER[None], 24]
enum_WGPUCreatePipelineAsyncStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCreatePipelineAsyncStatus_Success = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Success', 1) # type: ignore
WGPUCreatePipelineAsyncStatus_InstanceDropped = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_InstanceDropped', 2) # type: ignore
WGPUCreatePipelineAsyncStatus_ValidationError = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_ValidationError', 3) # type: ignore
WGPUCreatePipelineAsyncStatus_InternalError = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_InternalError', 4) # type: ignore
WGPUCreatePipelineAsyncStatus_DeviceLost = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_DeviceLost', 5) # type: ignore
WGPUCreatePipelineAsyncStatus_DeviceDestroyed = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_DeviceDestroyed', 6) # type: ignore
WGPUCreatePipelineAsyncStatus_Unknown = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Unknown', 7) # type: ignore
WGPUCreatePipelineAsyncStatus_Force32 = enum_WGPUCreatePipelineAsyncStatus.define('WGPUCreatePipelineAsyncStatus_Force32', 2147483647) # type: ignore

WGPUCreateComputePipelineAsyncCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView, c.POINTER[None])
@c.record
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCreateRenderPipelineAsyncCallback, 16]
  userdata: Annotated[c.POINTER[None], 24]
WGPUCreateRenderPipelineAsyncCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView, c.POINTER[None])
@c.record
class struct_WGPUDawnWGSLBlocklist(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  blocklistedFeatureCount: Annotated[size_t, 16]
  blocklistedFeatures: Annotated[c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], 24]
@c.record
class struct_WGPUDawnAdapterPropertiesPowerPreference(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  powerPreference: Annotated[WGPUPowerPreference, 16]
enum_WGPUPowerPreference = CEnum(Annotated[int, ctypes.c_uint32])
WGPUPowerPreference_Undefined = enum_WGPUPowerPreference.define('WGPUPowerPreference_Undefined', 0) # type: ignore
WGPUPowerPreference_LowPower = enum_WGPUPowerPreference.define('WGPUPowerPreference_LowPower', 1) # type: ignore
WGPUPowerPreference_HighPerformance = enum_WGPUPowerPreference.define('WGPUPowerPreference_HighPerformance', 2) # type: ignore
WGPUPowerPreference_Force32 = enum_WGPUPowerPreference.define('WGPUPowerPreference_Force32', 2147483647) # type: ignore

WGPUPowerPreference: TypeAlias = enum_WGPUPowerPreference
@c.record
class struct_WGPUDawnBufferDescriptorErrorInfoFromWireClient(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  outOfMemory: Annotated[WGPUBool, 16]
@c.record
class struct_WGPUDawnEncoderInternalUsageDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  useInternalUsages: Annotated[WGPUBool, 16]
@c.record
class struct_WGPUDawnExperimentalImmediateDataLimits(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  maxImmediateDataRangeByteSize: Annotated[uint32_t, 16]
@c.record
class struct_WGPUDawnExperimentalSubgroupLimits(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  minSubgroupSize: Annotated[uint32_t, 16]
  maxSubgroupSize: Annotated[uint32_t, 20]
@c.record
class struct_WGPUDawnRenderPassColorAttachmentRenderToSingleSampled(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  implicitSampleCount: Annotated[uint32_t, 16]
@c.record
class struct_WGPUDawnShaderModuleSPIRVOptionsDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  allowNonUniformDerivatives: Annotated[WGPUBool, 16]
@c.record
class struct_WGPUDawnTexelCopyBufferRowAlignmentLimits(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  minTexelCopyBufferRowAlignment: Annotated[uint32_t, 16]
@c.record
class struct_WGPUDawnTextureInternalUsageDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  internalUsage: Annotated[WGPUTextureUsage, 16]
WGPUTextureUsage = Annotated[int, ctypes.c_uint64]
@c.record
class struct_WGPUDawnTogglesDescriptor(c.Struct):
  SIZE = 48
  chain: Annotated[WGPUChainedStruct, 0]
  enabledToggleCount: Annotated[size_t, 16]
  enabledToggles: Annotated[c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], 24]
  disabledToggleCount: Annotated[size_t, 32]
  disabledToggles: Annotated[c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], 40]
@c.record
class struct_WGPUDawnWireWGSLControl(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  enableExperimental: Annotated[WGPUBool, 16]
  enableUnsafe: Annotated[WGPUBool, 20]
  enableTesting: Annotated[WGPUBool, 24]
@c.record
class struct_WGPUDeviceLostCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUDeviceLostCallbackNew, 16]
  userdata: Annotated[c.POINTER[None], 24]
enum_WGPUDeviceLostReason = CEnum(Annotated[int, ctypes.c_uint32])
WGPUDeviceLostReason_Unknown = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Unknown', 1) # type: ignore
WGPUDeviceLostReason_Destroyed = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Destroyed', 2) # type: ignore
WGPUDeviceLostReason_InstanceDropped = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_InstanceDropped', 3) # type: ignore
WGPUDeviceLostReason_FailedCreation = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_FailedCreation', 4) # type: ignore
WGPUDeviceLostReason_Force32 = enum_WGPUDeviceLostReason.define('WGPUDeviceLostReason_Force32', 2147483647) # type: ignore

WGPUDeviceLostCallbackNew: TypeAlias = c.CFUNCTYPE(None, c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], enum_WGPUDeviceLostReason, struct_WGPUStringView, c.POINTER[None])
@c.record
class struct_WGPUDrmFormatProperties(c.Struct):
  SIZE = 16
  modifier: Annotated[uint64_t, 0]
  modifierPlaneCount: Annotated[uint32_t, 8]
@c.record
class struct_WGPUExtent2D(c.Struct):
  SIZE = 8
  width: Annotated[uint32_t, 0]
  height: Annotated[uint32_t, 4]
@c.record
class struct_WGPUExtent3D(c.Struct):
  SIZE = 12
  width: Annotated[uint32_t, 0]
  height: Annotated[uint32_t, 4]
  depthOrArrayLayers: Annotated[uint32_t, 8]
@c.record
class struct_WGPUExternalTextureBindingEntry(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  externalTexture: Annotated[WGPUExternalTexture, 16]
@c.record
class struct_WGPUExternalTextureBindingLayout(c.Struct):
  SIZE = 16
  chain: Annotated[WGPUChainedStruct, 0]
@c.record
class struct_WGPUFormatCapabilities(c.Struct):
  SIZE = 8
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
@c.record
class struct_WGPUFuture(c.Struct):
  SIZE = 8
  id: Annotated[uint64_t, 0]
@c.record
class struct_WGPUInstanceFeatures(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  timedWaitAnyEnable: Annotated[WGPUBool, 8]
  timedWaitAnyMaxCount: Annotated[size_t, 16]
@c.record
class struct_WGPULimits(c.Struct):
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
@c.record
class struct_WGPUMemoryHeapInfo(c.Struct):
  SIZE = 16
  properties: Annotated[WGPUHeapProperty, 0]
  size: Annotated[uint64_t, 8]
WGPUHeapProperty = Annotated[int, ctypes.c_uint64]
@c.record
class struct_WGPUMultisampleState(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  count: Annotated[uint32_t, 8]
  mask: Annotated[uint32_t, 12]
  alphaToCoverageEnabled: Annotated[WGPUBool, 16]
@c.record
class struct_WGPUOrigin2D(c.Struct):
  SIZE = 8
  x: Annotated[uint32_t, 0]
  y: Annotated[uint32_t, 4]
@c.record
class struct_WGPUOrigin3D(c.Struct):
  SIZE = 12
  x: Annotated[uint32_t, 0]
  y: Annotated[uint32_t, 4]
  z: Annotated[uint32_t, 8]
@c.record
class struct_WGPUPipelineLayoutStorageAttachment(c.Struct):
  SIZE = 16
  offset: Annotated[uint64_t, 0]
  format: Annotated[WGPUTextureFormat, 8]
enum_WGPUTextureFormat = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureFormat_Undefined = enum_WGPUTextureFormat.define('WGPUTextureFormat_Undefined', 0) # type: ignore
WGPUTextureFormat_R8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8Unorm', 1) # type: ignore
WGPUTextureFormat_R8Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8Snorm', 2) # type: ignore
WGPUTextureFormat_R8Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8Uint', 3) # type: ignore
WGPUTextureFormat_R8Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8Sint', 4) # type: ignore
WGPUTextureFormat_R16Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Uint', 5) # type: ignore
WGPUTextureFormat_R16Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Sint', 6) # type: ignore
WGPUTextureFormat_R16Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Float', 7) # type: ignore
WGPUTextureFormat_RG8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG8Unorm', 8) # type: ignore
WGPUTextureFormat_RG8Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG8Snorm', 9) # type: ignore
WGPUTextureFormat_RG8Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG8Uint', 10) # type: ignore
WGPUTextureFormat_RG8Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG8Sint', 11) # type: ignore
WGPUTextureFormat_R32Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_R32Float', 12) # type: ignore
WGPUTextureFormat_R32Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R32Uint', 13) # type: ignore
WGPUTextureFormat_R32Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_R32Sint', 14) # type: ignore
WGPUTextureFormat_RG16Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Uint', 15) # type: ignore
WGPUTextureFormat_RG16Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Sint', 16) # type: ignore
WGPUTextureFormat_RG16Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Float', 17) # type: ignore
WGPUTextureFormat_RGBA8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8Unorm', 18) # type: ignore
WGPUTextureFormat_RGBA8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8UnormSrgb', 19) # type: ignore
WGPUTextureFormat_RGBA8Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8Snorm', 20) # type: ignore
WGPUTextureFormat_RGBA8Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8Uint', 21) # type: ignore
WGPUTextureFormat_RGBA8Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA8Sint', 22) # type: ignore
WGPUTextureFormat_BGRA8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BGRA8Unorm', 23) # type: ignore
WGPUTextureFormat_BGRA8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BGRA8UnormSrgb', 24) # type: ignore
WGPUTextureFormat_RGB10A2Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGB10A2Uint', 25) # type: ignore
WGPUTextureFormat_RGB10A2Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGB10A2Unorm', 26) # type: ignore
WGPUTextureFormat_RG11B10Ufloat = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG11B10Ufloat', 27) # type: ignore
WGPUTextureFormat_RGB9E5Ufloat = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGB9E5Ufloat', 28) # type: ignore
WGPUTextureFormat_RG32Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG32Float', 29) # type: ignore
WGPUTextureFormat_RG32Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG32Uint', 30) # type: ignore
WGPUTextureFormat_RG32Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG32Sint', 31) # type: ignore
WGPUTextureFormat_RGBA16Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Uint', 32) # type: ignore
WGPUTextureFormat_RGBA16Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Sint', 33) # type: ignore
WGPUTextureFormat_RGBA16Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Float', 34) # type: ignore
WGPUTextureFormat_RGBA32Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA32Float', 35) # type: ignore
WGPUTextureFormat_RGBA32Uint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA32Uint', 36) # type: ignore
WGPUTextureFormat_RGBA32Sint = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA32Sint', 37) # type: ignore
WGPUTextureFormat_Stencil8 = enum_WGPUTextureFormat.define('WGPUTextureFormat_Stencil8', 38) # type: ignore
WGPUTextureFormat_Depth16Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth16Unorm', 39) # type: ignore
WGPUTextureFormat_Depth24Plus = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth24Plus', 40) # type: ignore
WGPUTextureFormat_Depth24PlusStencil8 = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth24PlusStencil8', 41) # type: ignore
WGPUTextureFormat_Depth32Float = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth32Float', 42) # type: ignore
WGPUTextureFormat_Depth32FloatStencil8 = enum_WGPUTextureFormat.define('WGPUTextureFormat_Depth32FloatStencil8', 43) # type: ignore
WGPUTextureFormat_BC1RGBAUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC1RGBAUnorm', 44) # type: ignore
WGPUTextureFormat_BC1RGBAUnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC1RGBAUnormSrgb', 45) # type: ignore
WGPUTextureFormat_BC2RGBAUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC2RGBAUnorm', 46) # type: ignore
WGPUTextureFormat_BC2RGBAUnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC2RGBAUnormSrgb', 47) # type: ignore
WGPUTextureFormat_BC3RGBAUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC3RGBAUnorm', 48) # type: ignore
WGPUTextureFormat_BC3RGBAUnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC3RGBAUnormSrgb', 49) # type: ignore
WGPUTextureFormat_BC4RUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC4RUnorm', 50) # type: ignore
WGPUTextureFormat_BC4RSnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC4RSnorm', 51) # type: ignore
WGPUTextureFormat_BC5RGUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC5RGUnorm', 52) # type: ignore
WGPUTextureFormat_BC5RGSnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC5RGSnorm', 53) # type: ignore
WGPUTextureFormat_BC6HRGBUfloat = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC6HRGBUfloat', 54) # type: ignore
WGPUTextureFormat_BC6HRGBFloat = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC6HRGBFloat', 55) # type: ignore
WGPUTextureFormat_BC7RGBAUnorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC7RGBAUnorm', 56) # type: ignore
WGPUTextureFormat_BC7RGBAUnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_BC7RGBAUnormSrgb', 57) # type: ignore
WGPUTextureFormat_ETC2RGB8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGB8Unorm', 58) # type: ignore
WGPUTextureFormat_ETC2RGB8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGB8UnormSrgb', 59) # type: ignore
WGPUTextureFormat_ETC2RGB8A1Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGB8A1Unorm', 60) # type: ignore
WGPUTextureFormat_ETC2RGB8A1UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGB8A1UnormSrgb', 61) # type: ignore
WGPUTextureFormat_ETC2RGBA8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGBA8Unorm', 62) # type: ignore
WGPUTextureFormat_ETC2RGBA8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ETC2RGBA8UnormSrgb', 63) # type: ignore
WGPUTextureFormat_EACR11Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_EACR11Unorm', 64) # type: ignore
WGPUTextureFormat_EACR11Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_EACR11Snorm', 65) # type: ignore
WGPUTextureFormat_EACRG11Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_EACRG11Unorm', 66) # type: ignore
WGPUTextureFormat_EACRG11Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_EACRG11Snorm', 67) # type: ignore
WGPUTextureFormat_ASTC4x4Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC4x4Unorm', 68) # type: ignore
WGPUTextureFormat_ASTC4x4UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC4x4UnormSrgb', 69) # type: ignore
WGPUTextureFormat_ASTC5x4Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC5x4Unorm', 70) # type: ignore
WGPUTextureFormat_ASTC5x4UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC5x4UnormSrgb', 71) # type: ignore
WGPUTextureFormat_ASTC5x5Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC5x5Unorm', 72) # type: ignore
WGPUTextureFormat_ASTC5x5UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC5x5UnormSrgb', 73) # type: ignore
WGPUTextureFormat_ASTC6x5Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC6x5Unorm', 74) # type: ignore
WGPUTextureFormat_ASTC6x5UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC6x5UnormSrgb', 75) # type: ignore
WGPUTextureFormat_ASTC6x6Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC6x6Unorm', 76) # type: ignore
WGPUTextureFormat_ASTC6x6UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC6x6UnormSrgb', 77) # type: ignore
WGPUTextureFormat_ASTC8x5Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x5Unorm', 78) # type: ignore
WGPUTextureFormat_ASTC8x5UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x5UnormSrgb', 79) # type: ignore
WGPUTextureFormat_ASTC8x6Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x6Unorm', 80) # type: ignore
WGPUTextureFormat_ASTC8x6UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x6UnormSrgb', 81) # type: ignore
WGPUTextureFormat_ASTC8x8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x8Unorm', 82) # type: ignore
WGPUTextureFormat_ASTC8x8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC8x8UnormSrgb', 83) # type: ignore
WGPUTextureFormat_ASTC10x5Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x5Unorm', 84) # type: ignore
WGPUTextureFormat_ASTC10x5UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x5UnormSrgb', 85) # type: ignore
WGPUTextureFormat_ASTC10x6Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x6Unorm', 86) # type: ignore
WGPUTextureFormat_ASTC10x6UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x6UnormSrgb', 87) # type: ignore
WGPUTextureFormat_ASTC10x8Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x8Unorm', 88) # type: ignore
WGPUTextureFormat_ASTC10x8UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x8UnormSrgb', 89) # type: ignore
WGPUTextureFormat_ASTC10x10Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x10Unorm', 90) # type: ignore
WGPUTextureFormat_ASTC10x10UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC10x10UnormSrgb', 91) # type: ignore
WGPUTextureFormat_ASTC12x10Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC12x10Unorm', 92) # type: ignore
WGPUTextureFormat_ASTC12x10UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC12x10UnormSrgb', 93) # type: ignore
WGPUTextureFormat_ASTC12x12Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC12x12Unorm', 94) # type: ignore
WGPUTextureFormat_ASTC12x12UnormSrgb = enum_WGPUTextureFormat.define('WGPUTextureFormat_ASTC12x12UnormSrgb', 95) # type: ignore
WGPUTextureFormat_R16Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Unorm', 327680) # type: ignore
WGPUTextureFormat_RG16Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Unorm', 327681) # type: ignore
WGPUTextureFormat_RGBA16Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Unorm', 327682) # type: ignore
WGPUTextureFormat_R16Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R16Snorm', 327683) # type: ignore
WGPUTextureFormat_RG16Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RG16Snorm', 327684) # type: ignore
WGPUTextureFormat_RGBA16Snorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_RGBA16Snorm', 327685) # type: ignore
WGPUTextureFormat_R8BG8Biplanar420Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8BG8Biplanar420Unorm', 327686) # type: ignore
WGPUTextureFormat_R10X6BG10X6Biplanar420Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R10X6BG10X6Biplanar420Unorm', 327687) # type: ignore
WGPUTextureFormat_R8BG8A8Triplanar420Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8BG8A8Triplanar420Unorm', 327688) # type: ignore
WGPUTextureFormat_R8BG8Biplanar422Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8BG8Biplanar422Unorm', 327689) # type: ignore
WGPUTextureFormat_R8BG8Biplanar444Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R8BG8Biplanar444Unorm', 327690) # type: ignore
WGPUTextureFormat_R10X6BG10X6Biplanar422Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R10X6BG10X6Biplanar422Unorm', 327691) # type: ignore
WGPUTextureFormat_R10X6BG10X6Biplanar444Unorm = enum_WGPUTextureFormat.define('WGPUTextureFormat_R10X6BG10X6Biplanar444Unorm', 327692) # type: ignore
WGPUTextureFormat_External = enum_WGPUTextureFormat.define('WGPUTextureFormat_External', 327693) # type: ignore
WGPUTextureFormat_Force32 = enum_WGPUTextureFormat.define('WGPUTextureFormat_Force32', 2147483647) # type: ignore

WGPUTextureFormat: TypeAlias = enum_WGPUTextureFormat
@c.record
class struct_WGPUPopErrorScopeCallbackInfo(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUPopErrorScopeCallback, 16]
  oldCallback: Annotated[WGPUErrorCallback, 24]
  userdata: Annotated[c.POINTER[None], 32]
enum_WGPUPopErrorScopeStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUPopErrorScopeStatus_Success = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_Success', 1) # type: ignore
WGPUPopErrorScopeStatus_InstanceDropped = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_InstanceDropped', 2) # type: ignore
WGPUPopErrorScopeStatus_Force32 = enum_WGPUPopErrorScopeStatus.define('WGPUPopErrorScopeStatus_Force32', 2147483647) # type: ignore

enum_WGPUErrorType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUErrorType_NoError = enum_WGPUErrorType.define('WGPUErrorType_NoError', 1) # type: ignore
WGPUErrorType_Validation = enum_WGPUErrorType.define('WGPUErrorType_Validation', 2) # type: ignore
WGPUErrorType_OutOfMemory = enum_WGPUErrorType.define('WGPUErrorType_OutOfMemory', 3) # type: ignore
WGPUErrorType_Internal = enum_WGPUErrorType.define('WGPUErrorType_Internal', 4) # type: ignore
WGPUErrorType_Unknown = enum_WGPUErrorType.define('WGPUErrorType_Unknown', 5) # type: ignore
WGPUErrorType_DeviceLost = enum_WGPUErrorType.define('WGPUErrorType_DeviceLost', 6) # type: ignore
WGPUErrorType_Force32 = enum_WGPUErrorType.define('WGPUErrorType_Force32', 2147483647) # type: ignore

WGPUPopErrorScopeCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPUPopErrorScopeStatus, enum_WGPUErrorType, struct_WGPUStringView, c.POINTER[None])
WGPUErrorCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPUErrorType, struct_WGPUStringView, c.POINTER[None])
@c.record
class struct_WGPUPrimitiveState(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  topology: Annotated[WGPUPrimitiveTopology, 8]
  stripIndexFormat: Annotated[WGPUIndexFormat, 12]
  frontFace: Annotated[WGPUFrontFace, 16]
  cullMode: Annotated[WGPUCullMode, 20]
  unclippedDepth: Annotated[WGPUBool, 24]
enum_WGPUPrimitiveTopology = CEnum(Annotated[int, ctypes.c_uint32])
WGPUPrimitiveTopology_Undefined = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_Undefined', 0) # type: ignore
WGPUPrimitiveTopology_PointList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_PointList', 1) # type: ignore
WGPUPrimitiveTopology_LineList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_LineList', 2) # type: ignore
WGPUPrimitiveTopology_LineStrip = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_LineStrip', 3) # type: ignore
WGPUPrimitiveTopology_TriangleList = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_TriangleList', 4) # type: ignore
WGPUPrimitiveTopology_TriangleStrip = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_TriangleStrip', 5) # type: ignore
WGPUPrimitiveTopology_Force32 = enum_WGPUPrimitiveTopology.define('WGPUPrimitiveTopology_Force32', 2147483647) # type: ignore

WGPUPrimitiveTopology: TypeAlias = enum_WGPUPrimitiveTopology
enum_WGPUIndexFormat = CEnum(Annotated[int, ctypes.c_uint32])
WGPUIndexFormat_Undefined = enum_WGPUIndexFormat.define('WGPUIndexFormat_Undefined', 0) # type: ignore
WGPUIndexFormat_Uint16 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Uint16', 1) # type: ignore
WGPUIndexFormat_Uint32 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Uint32', 2) # type: ignore
WGPUIndexFormat_Force32 = enum_WGPUIndexFormat.define('WGPUIndexFormat_Force32', 2147483647) # type: ignore

WGPUIndexFormat: TypeAlias = enum_WGPUIndexFormat
enum_WGPUFrontFace = CEnum(Annotated[int, ctypes.c_uint32])
WGPUFrontFace_Undefined = enum_WGPUFrontFace.define('WGPUFrontFace_Undefined', 0) # type: ignore
WGPUFrontFace_CCW = enum_WGPUFrontFace.define('WGPUFrontFace_CCW', 1) # type: ignore
WGPUFrontFace_CW = enum_WGPUFrontFace.define('WGPUFrontFace_CW', 2) # type: ignore
WGPUFrontFace_Force32 = enum_WGPUFrontFace.define('WGPUFrontFace_Force32', 2147483647) # type: ignore

WGPUFrontFace: TypeAlias = enum_WGPUFrontFace
enum_WGPUCullMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCullMode_Undefined = enum_WGPUCullMode.define('WGPUCullMode_Undefined', 0) # type: ignore
WGPUCullMode_None = enum_WGPUCullMode.define('WGPUCullMode_None', 1) # type: ignore
WGPUCullMode_Front = enum_WGPUCullMode.define('WGPUCullMode_Front', 2) # type: ignore
WGPUCullMode_Back = enum_WGPUCullMode.define('WGPUCullMode_Back', 3) # type: ignore
WGPUCullMode_Force32 = enum_WGPUCullMode.define('WGPUCullMode_Force32', 2147483647) # type: ignore

WGPUCullMode: TypeAlias = enum_WGPUCullMode
@c.record
class struct_WGPUQueueWorkDoneCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUQueueWorkDoneCallback, 16]
  userdata: Annotated[c.POINTER[None], 24]
enum_WGPUQueueWorkDoneStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUQueueWorkDoneStatus_Success = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Success', 1) # type: ignore
WGPUQueueWorkDoneStatus_InstanceDropped = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_InstanceDropped', 2) # type: ignore
WGPUQueueWorkDoneStatus_Error = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Error', 3) # type: ignore
WGPUQueueWorkDoneStatus_Unknown = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Unknown', 4) # type: ignore
WGPUQueueWorkDoneStatus_DeviceLost = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_DeviceLost', 5) # type: ignore
WGPUQueueWorkDoneStatus_Force32 = enum_WGPUQueueWorkDoneStatus.define('WGPUQueueWorkDoneStatus_Force32', 2147483647) # type: ignore

WGPUQueueWorkDoneCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, c.POINTER[None])
@c.record
class struct_WGPURenderPassDepthStencilAttachment(c.Struct):
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
WGPULoadOp_Undefined = enum_WGPULoadOp.define('WGPULoadOp_Undefined', 0) # type: ignore
WGPULoadOp_Load = enum_WGPULoadOp.define('WGPULoadOp_Load', 1) # type: ignore
WGPULoadOp_Clear = enum_WGPULoadOp.define('WGPULoadOp_Clear', 2) # type: ignore
WGPULoadOp_ExpandResolveTexture = enum_WGPULoadOp.define('WGPULoadOp_ExpandResolveTexture', 327683) # type: ignore
WGPULoadOp_Force32 = enum_WGPULoadOp.define('WGPULoadOp_Force32', 2147483647) # type: ignore

WGPULoadOp: TypeAlias = enum_WGPULoadOp
enum_WGPUStoreOp = CEnum(Annotated[int, ctypes.c_uint32])
WGPUStoreOp_Undefined = enum_WGPUStoreOp.define('WGPUStoreOp_Undefined', 0) # type: ignore
WGPUStoreOp_Store = enum_WGPUStoreOp.define('WGPUStoreOp_Store', 1) # type: ignore
WGPUStoreOp_Discard = enum_WGPUStoreOp.define('WGPUStoreOp_Discard', 2) # type: ignore
WGPUStoreOp_Force32 = enum_WGPUStoreOp.define('WGPUStoreOp_Force32', 2147483647) # type: ignore

WGPUStoreOp: TypeAlias = enum_WGPUStoreOp
@c.record
class struct_WGPURenderPassDescriptorExpandResolveRect(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  x: Annotated[uint32_t, 16]
  y: Annotated[uint32_t, 20]
  width: Annotated[uint32_t, 24]
  height: Annotated[uint32_t, 28]
@c.record
class struct_WGPURenderPassMaxDrawCount(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  maxDrawCount: Annotated[uint64_t, 16]
@c.record
class struct_WGPURenderPassTimestampWrites(c.Struct):
  SIZE = 16
  querySet: Annotated[WGPUQuerySet, 0]
  beginningOfPassWriteIndex: Annotated[uint32_t, 8]
  endOfPassWriteIndex: Annotated[uint32_t, 12]
@c.record
class struct_WGPURequestAdapterCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPURequestAdapterCallback, 16]
  userdata: Annotated[c.POINTER[None], 24]
enum_WGPURequestAdapterStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPURequestAdapterStatus_Success = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Success', 1) # type: ignore
WGPURequestAdapterStatus_InstanceDropped = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_InstanceDropped', 2) # type: ignore
WGPURequestAdapterStatus_Unavailable = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Unavailable', 3) # type: ignore
WGPURequestAdapterStatus_Error = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Error', 4) # type: ignore
WGPURequestAdapterStatus_Unknown = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Unknown', 5) # type: ignore
WGPURequestAdapterStatus_Force32 = enum_WGPURequestAdapterStatus.define('WGPURequestAdapterStatus_Force32', 2147483647) # type: ignore

WGPURequestAdapterCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, c.POINTER[struct_WGPUAdapterImpl], struct_WGPUStringView, c.POINTER[None])
@c.record
class struct_WGPURequestAdapterOptions(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  compatibleSurface: Annotated[WGPUSurface, 8]
  featureLevel: Annotated[WGPUFeatureLevel, 16]
  powerPreference: Annotated[WGPUPowerPreference, 20]
  backendType: Annotated[WGPUBackendType, 24]
  forceFallbackAdapter: Annotated[WGPUBool, 28]
  compatibilityMode: Annotated[WGPUBool, 32]
enum_WGPUFeatureLevel = CEnum(Annotated[int, ctypes.c_uint32])
WGPUFeatureLevel_Undefined = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Undefined', 0) # type: ignore
WGPUFeatureLevel_Compatibility = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Compatibility', 1) # type: ignore
WGPUFeatureLevel_Core = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Core', 2) # type: ignore
WGPUFeatureLevel_Force32 = enum_WGPUFeatureLevel.define('WGPUFeatureLevel_Force32', 2147483647) # type: ignore

WGPUFeatureLevel: TypeAlias = enum_WGPUFeatureLevel
enum_WGPUBackendType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBackendType_Undefined = enum_WGPUBackendType.define('WGPUBackendType_Undefined', 0) # type: ignore
WGPUBackendType_Null = enum_WGPUBackendType.define('WGPUBackendType_Null', 1) # type: ignore
WGPUBackendType_WebGPU = enum_WGPUBackendType.define('WGPUBackendType_WebGPU', 2) # type: ignore
WGPUBackendType_D3D11 = enum_WGPUBackendType.define('WGPUBackendType_D3D11', 3) # type: ignore
WGPUBackendType_D3D12 = enum_WGPUBackendType.define('WGPUBackendType_D3D12', 4) # type: ignore
WGPUBackendType_Metal = enum_WGPUBackendType.define('WGPUBackendType_Metal', 5) # type: ignore
WGPUBackendType_Vulkan = enum_WGPUBackendType.define('WGPUBackendType_Vulkan', 6) # type: ignore
WGPUBackendType_OpenGL = enum_WGPUBackendType.define('WGPUBackendType_OpenGL', 7) # type: ignore
WGPUBackendType_OpenGLES = enum_WGPUBackendType.define('WGPUBackendType_OpenGLES', 8) # type: ignore
WGPUBackendType_Force32 = enum_WGPUBackendType.define('WGPUBackendType_Force32', 2147483647) # type: ignore

WGPUBackendType: TypeAlias = enum_WGPUBackendType
@c.record
class struct_WGPURequestDeviceCallbackInfo(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPURequestDeviceCallback, 16]
  userdata: Annotated[c.POINTER[None], 24]
enum_WGPURequestDeviceStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPURequestDeviceStatus_Success = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Success', 1) # type: ignore
WGPURequestDeviceStatus_InstanceDropped = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_InstanceDropped', 2) # type: ignore
WGPURequestDeviceStatus_Error = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Error', 3) # type: ignore
WGPURequestDeviceStatus_Unknown = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Unknown', 4) # type: ignore
WGPURequestDeviceStatus_Force32 = enum_WGPURequestDeviceStatus.define('WGPURequestDeviceStatus_Force32', 2147483647) # type: ignore

WGPURequestDeviceCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView, c.POINTER[None])
@c.record
class struct_WGPUSamplerBindingLayout(c.Struct):
  SIZE = 16
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  type: Annotated[WGPUSamplerBindingType, 8]
enum_WGPUSamplerBindingType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUSamplerBindingType_BindingNotUsed = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_BindingNotUsed', 0) # type: ignore
WGPUSamplerBindingType_Filtering = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Filtering', 1) # type: ignore
WGPUSamplerBindingType_NonFiltering = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_NonFiltering', 2) # type: ignore
WGPUSamplerBindingType_Comparison = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Comparison', 3) # type: ignore
WGPUSamplerBindingType_Force32 = enum_WGPUSamplerBindingType.define('WGPUSamplerBindingType_Force32', 2147483647) # type: ignore

WGPUSamplerBindingType: TypeAlias = enum_WGPUSamplerBindingType
@c.record
class struct_WGPUShaderModuleCompilationOptions(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  strictMath: Annotated[WGPUBool, 16]
@c.record
class struct_WGPUShaderSourceSPIRV(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  codeSize: Annotated[uint32_t, 16]
  code: Annotated[c.POINTER[uint32_t], 24]
@c.record
class struct_WGPUSharedBufferMemoryBeginAccessDescriptor(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  initialized: Annotated[WGPUBool, 8]
  fenceCount: Annotated[size_t, 16]
  fences: Annotated[c.POINTER[WGPUSharedFence], 24]
  signaledValues: Annotated[c.POINTER[uint64_t], 32]
@c.record
class struct_WGPUSharedBufferMemoryEndAccessState(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
  initialized: Annotated[WGPUBool, 8]
  fenceCount: Annotated[size_t, 16]
  fences: Annotated[c.POINTER[WGPUSharedFence], 24]
  signaledValues: Annotated[c.POINTER[uint64_t], 32]
@c.record
class struct_WGPUSharedBufferMemoryProperties(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
  usage: Annotated[WGPUBufferUsage, 8]
  size: Annotated[uint64_t, 16]
WGPUBufferUsage = Annotated[int, ctypes.c_uint64]
@c.record
class struct_WGPUSharedFenceDXGISharedHandleDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSharedFenceDXGISharedHandleExportInfo(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  handle: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSharedFenceMTLSharedEventDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  sharedEvent: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSharedFenceMTLSharedEventExportInfo(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  sharedEvent: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSharedFenceExportInfo(c.Struct):
  SIZE = 16
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
  type: Annotated[WGPUSharedFenceType, 8]
enum_WGPUSharedFenceType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUSharedFenceType_VkSemaphoreOpaqueFD = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_VkSemaphoreOpaqueFD', 1) # type: ignore
WGPUSharedFenceType_SyncFD = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_SyncFD', 2) # type: ignore
WGPUSharedFenceType_VkSemaphoreZirconHandle = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_VkSemaphoreZirconHandle', 3) # type: ignore
WGPUSharedFenceType_DXGISharedHandle = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_DXGISharedHandle', 4) # type: ignore
WGPUSharedFenceType_MTLSharedEvent = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_MTLSharedEvent', 5) # type: ignore
WGPUSharedFenceType_Force32 = enum_WGPUSharedFenceType.define('WGPUSharedFenceType_Force32', 2147483647) # type: ignore

WGPUSharedFenceType: TypeAlias = enum_WGPUSharedFenceType
@c.record
class struct_WGPUSharedFenceSyncFDDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[Annotated[int, ctypes.c_int32], 16]
@c.record
class struct_WGPUSharedFenceSyncFDExportInfo(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  handle: Annotated[Annotated[int, ctypes.c_int32], 16]
@c.record
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[Annotated[int, ctypes.c_int32], 16]
@c.record
class struct_WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  handle: Annotated[Annotated[int, ctypes.c_int32], 16]
@c.record
class struct_WGPUSharedFenceVkSemaphoreZirconHandleDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[uint32_t, 16]
@c.record
class struct_WGPUSharedFenceVkSemaphoreZirconHandleExportInfo(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  handle: Annotated[uint32_t, 16]
@c.record
class struct_WGPUSharedTextureMemoryD3DSwapchainBeginState(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  isSwapchain: Annotated[WGPUBool, 16]
@c.record
class struct_WGPUSharedTextureMemoryDXGISharedHandleDescriptor(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[c.POINTER[None], 16]
  useKeyedMutex: Annotated[WGPUBool, 24]
@c.record
class struct_WGPUSharedTextureMemoryEGLImageDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  image: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSharedTextureMemoryIOSurfaceDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  ioSurface: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSharedTextureMemoryAHardwareBufferDescriptor(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  handle: Annotated[c.POINTER[None], 16]
  useExternalFormat: Annotated[WGPUBool, 24]
@c.record
class struct_WGPUSharedTextureMemoryBeginAccessDescriptor(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  concurrentRead: Annotated[WGPUBool, 8]
  initialized: Annotated[WGPUBool, 12]
  fenceCount: Annotated[size_t, 16]
  fences: Annotated[c.POINTER[WGPUSharedFence], 24]
  signaledValues: Annotated[c.POINTER[uint64_t], 32]
@c.record
class struct_WGPUSharedTextureMemoryDmaBufPlane(c.Struct):
  SIZE = 24
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  offset: Annotated[uint64_t, 8]
  stride: Annotated[uint32_t, 16]
@c.record
class struct_WGPUSharedTextureMemoryEndAccessState(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
  initialized: Annotated[WGPUBool, 8]
  fenceCount: Annotated[size_t, 16]
  fences: Annotated[c.POINTER[WGPUSharedFence], 24]
  signaledValues: Annotated[c.POINTER[uint64_t], 32]
@c.record
class struct_WGPUSharedTextureMemoryOpaqueFDDescriptor(c.Struct):
  SIZE = 48
  chain: Annotated[WGPUChainedStruct, 0]
  vkImageCreateInfo: Annotated[c.POINTER[None], 16]
  memoryFD: Annotated[Annotated[int, ctypes.c_int32], 24]
  memoryTypeIndex: Annotated[uint32_t, 28]
  allocationSize: Annotated[uint64_t, 32]
  dedicatedAllocation: Annotated[WGPUBool, 40]
@c.record
class struct_WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  dedicatedAllocation: Annotated[WGPUBool, 16]
@c.record
class struct_WGPUSharedTextureMemoryVkImageLayoutBeginState(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  oldLayout: Annotated[int32_t, 16]
  newLayout: Annotated[int32_t, 20]
int32_t = Annotated[int, ctypes.c_int32]
@c.record
class struct_WGPUSharedTextureMemoryVkImageLayoutEndState(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStructOut, 0]
  oldLayout: Annotated[int32_t, 16]
  newLayout: Annotated[int32_t, 20]
@c.record
class struct_WGPUSharedTextureMemoryZirconHandleDescriptor(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  memoryFD: Annotated[uint32_t, 16]
  allocationSize: Annotated[uint64_t, 24]
@c.record
class struct_WGPUStaticSamplerBindingLayout(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  sampler: Annotated[WGPUSampler, 16]
  sampledTextureBinding: Annotated[uint32_t, 24]
@c.record
class struct_WGPUStencilFaceState(c.Struct):
  SIZE = 16
  compare: Annotated[WGPUCompareFunction, 0]
  failOp: Annotated[WGPUStencilOperation, 4]
  depthFailOp: Annotated[WGPUStencilOperation, 8]
  passOp: Annotated[WGPUStencilOperation, 12]
enum_WGPUCompareFunction = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCompareFunction_Undefined = enum_WGPUCompareFunction.define('WGPUCompareFunction_Undefined', 0) # type: ignore
WGPUCompareFunction_Never = enum_WGPUCompareFunction.define('WGPUCompareFunction_Never', 1) # type: ignore
WGPUCompareFunction_Less = enum_WGPUCompareFunction.define('WGPUCompareFunction_Less', 2) # type: ignore
WGPUCompareFunction_Equal = enum_WGPUCompareFunction.define('WGPUCompareFunction_Equal', 3) # type: ignore
WGPUCompareFunction_LessEqual = enum_WGPUCompareFunction.define('WGPUCompareFunction_LessEqual', 4) # type: ignore
WGPUCompareFunction_Greater = enum_WGPUCompareFunction.define('WGPUCompareFunction_Greater', 5) # type: ignore
WGPUCompareFunction_NotEqual = enum_WGPUCompareFunction.define('WGPUCompareFunction_NotEqual', 6) # type: ignore
WGPUCompareFunction_GreaterEqual = enum_WGPUCompareFunction.define('WGPUCompareFunction_GreaterEqual', 7) # type: ignore
WGPUCompareFunction_Always = enum_WGPUCompareFunction.define('WGPUCompareFunction_Always', 8) # type: ignore
WGPUCompareFunction_Force32 = enum_WGPUCompareFunction.define('WGPUCompareFunction_Force32', 2147483647) # type: ignore

WGPUCompareFunction: TypeAlias = enum_WGPUCompareFunction
enum_WGPUStencilOperation = CEnum(Annotated[int, ctypes.c_uint32])
WGPUStencilOperation_Undefined = enum_WGPUStencilOperation.define('WGPUStencilOperation_Undefined', 0) # type: ignore
WGPUStencilOperation_Keep = enum_WGPUStencilOperation.define('WGPUStencilOperation_Keep', 1) # type: ignore
WGPUStencilOperation_Zero = enum_WGPUStencilOperation.define('WGPUStencilOperation_Zero', 2) # type: ignore
WGPUStencilOperation_Replace = enum_WGPUStencilOperation.define('WGPUStencilOperation_Replace', 3) # type: ignore
WGPUStencilOperation_Invert = enum_WGPUStencilOperation.define('WGPUStencilOperation_Invert', 4) # type: ignore
WGPUStencilOperation_IncrementClamp = enum_WGPUStencilOperation.define('WGPUStencilOperation_IncrementClamp', 5) # type: ignore
WGPUStencilOperation_DecrementClamp = enum_WGPUStencilOperation.define('WGPUStencilOperation_DecrementClamp', 6) # type: ignore
WGPUStencilOperation_IncrementWrap = enum_WGPUStencilOperation.define('WGPUStencilOperation_IncrementWrap', 7) # type: ignore
WGPUStencilOperation_DecrementWrap = enum_WGPUStencilOperation.define('WGPUStencilOperation_DecrementWrap', 8) # type: ignore
WGPUStencilOperation_Force32 = enum_WGPUStencilOperation.define('WGPUStencilOperation_Force32', 2147483647) # type: ignore

WGPUStencilOperation: TypeAlias = enum_WGPUStencilOperation
@c.record
class struct_WGPUStorageTextureBindingLayout(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  access: Annotated[WGPUStorageTextureAccess, 8]
  format: Annotated[WGPUTextureFormat, 12]
  viewDimension: Annotated[WGPUTextureViewDimension, 16]
enum_WGPUStorageTextureAccess = CEnum(Annotated[int, ctypes.c_uint32])
WGPUStorageTextureAccess_BindingNotUsed = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_BindingNotUsed', 0) # type: ignore
WGPUStorageTextureAccess_WriteOnly = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_WriteOnly', 1) # type: ignore
WGPUStorageTextureAccess_ReadOnly = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_ReadOnly', 2) # type: ignore
WGPUStorageTextureAccess_ReadWrite = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_ReadWrite', 3) # type: ignore
WGPUStorageTextureAccess_Force32 = enum_WGPUStorageTextureAccess.define('WGPUStorageTextureAccess_Force32', 2147483647) # type: ignore

WGPUStorageTextureAccess: TypeAlias = enum_WGPUStorageTextureAccess
enum_WGPUTextureViewDimension = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureViewDimension_Undefined = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Undefined', 0) # type: ignore
WGPUTextureViewDimension_1D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_1D', 1) # type: ignore
WGPUTextureViewDimension_2D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_2D', 2) # type: ignore
WGPUTextureViewDimension_2DArray = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_2DArray', 3) # type: ignore
WGPUTextureViewDimension_Cube = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Cube', 4) # type: ignore
WGPUTextureViewDimension_CubeArray = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_CubeArray', 5) # type: ignore
WGPUTextureViewDimension_3D = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_3D', 6) # type: ignore
WGPUTextureViewDimension_Force32 = enum_WGPUTextureViewDimension.define('WGPUTextureViewDimension_Force32', 2147483647) # type: ignore

WGPUTextureViewDimension: TypeAlias = enum_WGPUTextureViewDimension
@c.record
class struct_WGPUSupportedFeatures(c.Struct):
  SIZE = 16
  featureCount: Annotated[size_t, 0]
  features: Annotated[c.POINTER[WGPUFeatureName], 8]
enum_WGPUFeatureName = CEnum(Annotated[int, ctypes.c_uint32])
WGPUFeatureName_DepthClipControl = enum_WGPUFeatureName.define('WGPUFeatureName_DepthClipControl', 1) # type: ignore
WGPUFeatureName_Depth32FloatStencil8 = enum_WGPUFeatureName.define('WGPUFeatureName_Depth32FloatStencil8', 2) # type: ignore
WGPUFeatureName_TimestampQuery = enum_WGPUFeatureName.define('WGPUFeatureName_TimestampQuery', 3) # type: ignore
WGPUFeatureName_TextureCompressionBC = enum_WGPUFeatureName.define('WGPUFeatureName_TextureCompressionBC', 4) # type: ignore
WGPUFeatureName_TextureCompressionETC2 = enum_WGPUFeatureName.define('WGPUFeatureName_TextureCompressionETC2', 5) # type: ignore
WGPUFeatureName_TextureCompressionASTC = enum_WGPUFeatureName.define('WGPUFeatureName_TextureCompressionASTC', 6) # type: ignore
WGPUFeatureName_IndirectFirstInstance = enum_WGPUFeatureName.define('WGPUFeatureName_IndirectFirstInstance', 7) # type: ignore
WGPUFeatureName_ShaderF16 = enum_WGPUFeatureName.define('WGPUFeatureName_ShaderF16', 8) # type: ignore
WGPUFeatureName_RG11B10UfloatRenderable = enum_WGPUFeatureName.define('WGPUFeatureName_RG11B10UfloatRenderable', 9) # type: ignore
WGPUFeatureName_BGRA8UnormStorage = enum_WGPUFeatureName.define('WGPUFeatureName_BGRA8UnormStorage', 10) # type: ignore
WGPUFeatureName_Float32Filterable = enum_WGPUFeatureName.define('WGPUFeatureName_Float32Filterable', 11) # type: ignore
WGPUFeatureName_Float32Blendable = enum_WGPUFeatureName.define('WGPUFeatureName_Float32Blendable', 12) # type: ignore
WGPUFeatureName_Subgroups = enum_WGPUFeatureName.define('WGPUFeatureName_Subgroups', 13) # type: ignore
WGPUFeatureName_SubgroupsF16 = enum_WGPUFeatureName.define('WGPUFeatureName_SubgroupsF16', 14) # type: ignore
WGPUFeatureName_DawnInternalUsages = enum_WGPUFeatureName.define('WGPUFeatureName_DawnInternalUsages', 327680) # type: ignore
WGPUFeatureName_DawnMultiPlanarFormats = enum_WGPUFeatureName.define('WGPUFeatureName_DawnMultiPlanarFormats', 327681) # type: ignore
WGPUFeatureName_DawnNative = enum_WGPUFeatureName.define('WGPUFeatureName_DawnNative', 327682) # type: ignore
WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses = enum_WGPUFeatureName.define('WGPUFeatureName_ChromiumExperimentalTimestampQueryInsidePasses', 327683) # type: ignore
WGPUFeatureName_ImplicitDeviceSynchronization = enum_WGPUFeatureName.define('WGPUFeatureName_ImplicitDeviceSynchronization', 327684) # type: ignore
WGPUFeatureName_ChromiumExperimentalImmediateData = enum_WGPUFeatureName.define('WGPUFeatureName_ChromiumExperimentalImmediateData', 327685) # type: ignore
WGPUFeatureName_TransientAttachments = enum_WGPUFeatureName.define('WGPUFeatureName_TransientAttachments', 327686) # type: ignore
WGPUFeatureName_MSAARenderToSingleSampled = enum_WGPUFeatureName.define('WGPUFeatureName_MSAARenderToSingleSampled', 327687) # type: ignore
WGPUFeatureName_DualSourceBlending = enum_WGPUFeatureName.define('WGPUFeatureName_DualSourceBlending', 327688) # type: ignore
WGPUFeatureName_D3D11MultithreadProtected = enum_WGPUFeatureName.define('WGPUFeatureName_D3D11MultithreadProtected', 327689) # type: ignore
WGPUFeatureName_ANGLETextureSharing = enum_WGPUFeatureName.define('WGPUFeatureName_ANGLETextureSharing', 327690) # type: ignore
WGPUFeatureName_PixelLocalStorageCoherent = enum_WGPUFeatureName.define('WGPUFeatureName_PixelLocalStorageCoherent', 327691) # type: ignore
WGPUFeatureName_PixelLocalStorageNonCoherent = enum_WGPUFeatureName.define('WGPUFeatureName_PixelLocalStorageNonCoherent', 327692) # type: ignore
WGPUFeatureName_Unorm16TextureFormats = enum_WGPUFeatureName.define('WGPUFeatureName_Unorm16TextureFormats', 327693) # type: ignore
WGPUFeatureName_Snorm16TextureFormats = enum_WGPUFeatureName.define('WGPUFeatureName_Snorm16TextureFormats', 327694) # type: ignore
WGPUFeatureName_MultiPlanarFormatExtendedUsages = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatExtendedUsages', 327695) # type: ignore
WGPUFeatureName_MultiPlanarFormatP010 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatP010', 327696) # type: ignore
WGPUFeatureName_HostMappedPointer = enum_WGPUFeatureName.define('WGPUFeatureName_HostMappedPointer', 327697) # type: ignore
WGPUFeatureName_MultiPlanarRenderTargets = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarRenderTargets', 327698) # type: ignore
WGPUFeatureName_MultiPlanarFormatNv12a = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatNv12a', 327699) # type: ignore
WGPUFeatureName_FramebufferFetch = enum_WGPUFeatureName.define('WGPUFeatureName_FramebufferFetch', 327700) # type: ignore
WGPUFeatureName_BufferMapExtendedUsages = enum_WGPUFeatureName.define('WGPUFeatureName_BufferMapExtendedUsages', 327701) # type: ignore
WGPUFeatureName_AdapterPropertiesMemoryHeaps = enum_WGPUFeatureName.define('WGPUFeatureName_AdapterPropertiesMemoryHeaps', 327702) # type: ignore
WGPUFeatureName_AdapterPropertiesD3D = enum_WGPUFeatureName.define('WGPUFeatureName_AdapterPropertiesD3D', 327703) # type: ignore
WGPUFeatureName_AdapterPropertiesVk = enum_WGPUFeatureName.define('WGPUFeatureName_AdapterPropertiesVk', 327704) # type: ignore
WGPUFeatureName_R8UnormStorage = enum_WGPUFeatureName.define('WGPUFeatureName_R8UnormStorage', 327705) # type: ignore
WGPUFeatureName_FormatCapabilities = enum_WGPUFeatureName.define('WGPUFeatureName_FormatCapabilities', 327706) # type: ignore
WGPUFeatureName_DrmFormatCapabilities = enum_WGPUFeatureName.define('WGPUFeatureName_DrmFormatCapabilities', 327707) # type: ignore
WGPUFeatureName_Norm16TextureFormats = enum_WGPUFeatureName.define('WGPUFeatureName_Norm16TextureFormats', 327708) # type: ignore
WGPUFeatureName_MultiPlanarFormatNv16 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatNv16', 327709) # type: ignore
WGPUFeatureName_MultiPlanarFormatNv24 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatNv24', 327710) # type: ignore
WGPUFeatureName_MultiPlanarFormatP210 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatP210', 327711) # type: ignore
WGPUFeatureName_MultiPlanarFormatP410 = enum_WGPUFeatureName.define('WGPUFeatureName_MultiPlanarFormatP410', 327712) # type: ignore
WGPUFeatureName_SharedTextureMemoryVkDedicatedAllocation = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryVkDedicatedAllocation', 327713) # type: ignore
WGPUFeatureName_SharedTextureMemoryAHardwareBuffer = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryAHardwareBuffer', 327714) # type: ignore
WGPUFeatureName_SharedTextureMemoryDmaBuf = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryDmaBuf', 327715) # type: ignore
WGPUFeatureName_SharedTextureMemoryOpaqueFD = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryOpaqueFD', 327716) # type: ignore
WGPUFeatureName_SharedTextureMemoryZirconHandle = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryZirconHandle', 327717) # type: ignore
WGPUFeatureName_SharedTextureMemoryDXGISharedHandle = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryDXGISharedHandle', 327718) # type: ignore
WGPUFeatureName_SharedTextureMemoryD3D11Texture2D = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryD3D11Texture2D', 327719) # type: ignore
WGPUFeatureName_SharedTextureMemoryIOSurface = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryIOSurface', 327720) # type: ignore
WGPUFeatureName_SharedTextureMemoryEGLImage = enum_WGPUFeatureName.define('WGPUFeatureName_SharedTextureMemoryEGLImage', 327721) # type: ignore
WGPUFeatureName_SharedFenceVkSemaphoreOpaqueFD = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceVkSemaphoreOpaqueFD', 327722) # type: ignore
WGPUFeatureName_SharedFenceSyncFD = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceSyncFD', 327723) # type: ignore
WGPUFeatureName_SharedFenceVkSemaphoreZirconHandle = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceVkSemaphoreZirconHandle', 327724) # type: ignore
WGPUFeatureName_SharedFenceDXGISharedHandle = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceDXGISharedHandle', 327725) # type: ignore
WGPUFeatureName_SharedFenceMTLSharedEvent = enum_WGPUFeatureName.define('WGPUFeatureName_SharedFenceMTLSharedEvent', 327726) # type: ignore
WGPUFeatureName_SharedBufferMemoryD3D12Resource = enum_WGPUFeatureName.define('WGPUFeatureName_SharedBufferMemoryD3D12Resource', 327727) # type: ignore
WGPUFeatureName_StaticSamplers = enum_WGPUFeatureName.define('WGPUFeatureName_StaticSamplers', 327728) # type: ignore
WGPUFeatureName_YCbCrVulkanSamplers = enum_WGPUFeatureName.define('WGPUFeatureName_YCbCrVulkanSamplers', 327729) # type: ignore
WGPUFeatureName_ShaderModuleCompilationOptions = enum_WGPUFeatureName.define('WGPUFeatureName_ShaderModuleCompilationOptions', 327730) # type: ignore
WGPUFeatureName_DawnLoadResolveTexture = enum_WGPUFeatureName.define('WGPUFeatureName_DawnLoadResolveTexture', 327731) # type: ignore
WGPUFeatureName_DawnPartialLoadResolveTexture = enum_WGPUFeatureName.define('WGPUFeatureName_DawnPartialLoadResolveTexture', 327732) # type: ignore
WGPUFeatureName_MultiDrawIndirect = enum_WGPUFeatureName.define('WGPUFeatureName_MultiDrawIndirect', 327733) # type: ignore
WGPUFeatureName_ClipDistances = enum_WGPUFeatureName.define('WGPUFeatureName_ClipDistances', 327734) # type: ignore
WGPUFeatureName_DawnTexelCopyBufferRowAlignment = enum_WGPUFeatureName.define('WGPUFeatureName_DawnTexelCopyBufferRowAlignment', 327735) # type: ignore
WGPUFeatureName_FlexibleTextureViews = enum_WGPUFeatureName.define('WGPUFeatureName_FlexibleTextureViews', 327736) # type: ignore
WGPUFeatureName_Force32 = enum_WGPUFeatureName.define('WGPUFeatureName_Force32', 2147483647) # type: ignore

WGPUFeatureName: TypeAlias = enum_WGPUFeatureName
@c.record
class struct_WGPUSurfaceCapabilities(c.Struct):
  SIZE = 64
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
  usages: Annotated[WGPUTextureUsage, 8]
  formatCount: Annotated[size_t, 16]
  formats: Annotated[c.POINTER[WGPUTextureFormat], 24]
  presentModeCount: Annotated[size_t, 32]
  presentModes: Annotated[c.POINTER[WGPUPresentMode], 40]
  alphaModeCount: Annotated[size_t, 48]
  alphaModes: Annotated[c.POINTER[WGPUCompositeAlphaMode], 56]
enum_WGPUPresentMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUPresentMode_Fifo = enum_WGPUPresentMode.define('WGPUPresentMode_Fifo', 1) # type: ignore
WGPUPresentMode_FifoRelaxed = enum_WGPUPresentMode.define('WGPUPresentMode_FifoRelaxed', 2) # type: ignore
WGPUPresentMode_Immediate = enum_WGPUPresentMode.define('WGPUPresentMode_Immediate', 3) # type: ignore
WGPUPresentMode_Mailbox = enum_WGPUPresentMode.define('WGPUPresentMode_Mailbox', 4) # type: ignore
WGPUPresentMode_Force32 = enum_WGPUPresentMode.define('WGPUPresentMode_Force32', 2147483647) # type: ignore

WGPUPresentMode: TypeAlias = enum_WGPUPresentMode
enum_WGPUCompositeAlphaMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUCompositeAlphaMode_Auto = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Auto', 0) # type: ignore
WGPUCompositeAlphaMode_Opaque = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Opaque', 1) # type: ignore
WGPUCompositeAlphaMode_Premultiplied = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Premultiplied', 2) # type: ignore
WGPUCompositeAlphaMode_Unpremultiplied = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Unpremultiplied', 3) # type: ignore
WGPUCompositeAlphaMode_Inherit = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Inherit', 4) # type: ignore
WGPUCompositeAlphaMode_Force32 = enum_WGPUCompositeAlphaMode.define('WGPUCompositeAlphaMode_Force32', 2147483647) # type: ignore

WGPUCompositeAlphaMode: TypeAlias = enum_WGPUCompositeAlphaMode
@c.record
class struct_WGPUSurfaceConfiguration(c.Struct):
  SIZE = 64
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  device: Annotated[WGPUDevice, 8]
  format: Annotated[WGPUTextureFormat, 16]
  usage: Annotated[WGPUTextureUsage, 24]
  viewFormatCount: Annotated[size_t, 32]
  viewFormats: Annotated[c.POINTER[WGPUTextureFormat], 40]
  alphaMode: Annotated[WGPUCompositeAlphaMode, 48]
  width: Annotated[uint32_t, 52]
  height: Annotated[uint32_t, 56]
  presentMode: Annotated[WGPUPresentMode, 60]
@c.record
class struct_WGPUSurfaceDescriptorFromWindowsCoreWindow(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  coreWindow: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSurfaceDescriptorFromWindowsSwapChainPanel(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  swapChainPanel: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSurfaceSourceXCBWindow(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  connection: Annotated[c.POINTER[None], 16]
  window: Annotated[uint32_t, 24]
@c.record
class struct_WGPUSurfaceSourceAndroidNativeWindow(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  window: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSurfaceSourceMetalLayer(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  layer: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUSurfaceSourceWaylandSurface(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  display: Annotated[c.POINTER[None], 16]
  surface: Annotated[c.POINTER[None], 24]
@c.record
class struct_WGPUSurfaceSourceWindowsHWND(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  hinstance: Annotated[c.POINTER[None], 16]
  hwnd: Annotated[c.POINTER[None], 24]
@c.record
class struct_WGPUSurfaceSourceXlibWindow(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  display: Annotated[c.POINTER[None], 16]
  window: Annotated[uint64_t, 24]
@c.record
class struct_WGPUSurfaceTexture(c.Struct):
  SIZE = 16
  texture: Annotated[WGPUTexture, 0]
  suboptimal: Annotated[WGPUBool, 8]
  status: Annotated[WGPUSurfaceGetCurrentTextureStatus, 12]
enum_WGPUSurfaceGetCurrentTextureStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUSurfaceGetCurrentTextureStatus_Success = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Success', 1) # type: ignore
WGPUSurfaceGetCurrentTextureStatus_Timeout = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Timeout', 2) # type: ignore
WGPUSurfaceGetCurrentTextureStatus_Outdated = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Outdated', 3) # type: ignore
WGPUSurfaceGetCurrentTextureStatus_Lost = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Lost', 4) # type: ignore
WGPUSurfaceGetCurrentTextureStatus_OutOfMemory = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_OutOfMemory', 5) # type: ignore
WGPUSurfaceGetCurrentTextureStatus_DeviceLost = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_DeviceLost', 6) # type: ignore
WGPUSurfaceGetCurrentTextureStatus_Error = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Error', 7) # type: ignore
WGPUSurfaceGetCurrentTextureStatus_Force32 = enum_WGPUSurfaceGetCurrentTextureStatus.define('WGPUSurfaceGetCurrentTextureStatus_Force32', 2147483647) # type: ignore

WGPUSurfaceGetCurrentTextureStatus: TypeAlias = enum_WGPUSurfaceGetCurrentTextureStatus
@c.record
class struct_WGPUTextureBindingLayout(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  sampleType: Annotated[WGPUTextureSampleType, 8]
  viewDimension: Annotated[WGPUTextureViewDimension, 12]
  multisampled: Annotated[WGPUBool, 16]
enum_WGPUTextureSampleType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureSampleType_BindingNotUsed = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_BindingNotUsed', 0) # type: ignore
WGPUTextureSampleType_Float = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Float', 1) # type: ignore
WGPUTextureSampleType_UnfilterableFloat = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_UnfilterableFloat', 2) # type: ignore
WGPUTextureSampleType_Depth = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Depth', 3) # type: ignore
WGPUTextureSampleType_Sint = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Sint', 4) # type: ignore
WGPUTextureSampleType_Uint = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Uint', 5) # type: ignore
WGPUTextureSampleType_Force32 = enum_WGPUTextureSampleType.define('WGPUTextureSampleType_Force32', 2147483647) # type: ignore

WGPUTextureSampleType: TypeAlias = enum_WGPUTextureSampleType
@c.record
class struct_WGPUTextureBindingViewDimensionDescriptor(c.Struct):
  SIZE = 24
  chain: Annotated[WGPUChainedStruct, 0]
  textureBindingViewDimension: Annotated[WGPUTextureViewDimension, 16]
@c.record
class struct_WGPUTextureDataLayout(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  offset: Annotated[uint64_t, 8]
  bytesPerRow: Annotated[uint32_t, 16]
  rowsPerImage: Annotated[uint32_t, 20]
@c.record
class struct_WGPUUncapturedErrorCallbackInfo(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  callback: Annotated[WGPUErrorCallback, 8]
  userdata: Annotated[c.POINTER[None], 16]
@c.record
class struct_WGPUVertexAttribute(c.Struct):
  SIZE = 24
  format: Annotated[WGPUVertexFormat, 0]
  offset: Annotated[uint64_t, 8]
  shaderLocation: Annotated[uint32_t, 16]
enum_WGPUVertexFormat = CEnum(Annotated[int, ctypes.c_uint32])
WGPUVertexFormat_Uint8 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint8', 1) # type: ignore
WGPUVertexFormat_Uint8x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint8x2', 2) # type: ignore
WGPUVertexFormat_Uint8x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint8x4', 3) # type: ignore
WGPUVertexFormat_Sint8 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint8', 4) # type: ignore
WGPUVertexFormat_Sint8x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint8x2', 5) # type: ignore
WGPUVertexFormat_Sint8x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint8x4', 6) # type: ignore
WGPUVertexFormat_Unorm8 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm8', 7) # type: ignore
WGPUVertexFormat_Unorm8x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm8x2', 8) # type: ignore
WGPUVertexFormat_Unorm8x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm8x4', 9) # type: ignore
WGPUVertexFormat_Snorm8 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm8', 10) # type: ignore
WGPUVertexFormat_Snorm8x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm8x2', 11) # type: ignore
WGPUVertexFormat_Snorm8x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm8x4', 12) # type: ignore
WGPUVertexFormat_Uint16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint16', 13) # type: ignore
WGPUVertexFormat_Uint16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint16x2', 14) # type: ignore
WGPUVertexFormat_Uint16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint16x4', 15) # type: ignore
WGPUVertexFormat_Sint16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint16', 16) # type: ignore
WGPUVertexFormat_Sint16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint16x2', 17) # type: ignore
WGPUVertexFormat_Sint16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint16x4', 18) # type: ignore
WGPUVertexFormat_Unorm16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm16', 19) # type: ignore
WGPUVertexFormat_Unorm16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm16x2', 20) # type: ignore
WGPUVertexFormat_Unorm16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm16x4', 21) # type: ignore
WGPUVertexFormat_Snorm16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm16', 22) # type: ignore
WGPUVertexFormat_Snorm16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm16x2', 23) # type: ignore
WGPUVertexFormat_Snorm16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Snorm16x4', 24) # type: ignore
WGPUVertexFormat_Float16 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float16', 25) # type: ignore
WGPUVertexFormat_Float16x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float16x2', 26) # type: ignore
WGPUVertexFormat_Float16x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float16x4', 27) # type: ignore
WGPUVertexFormat_Float32 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float32', 28) # type: ignore
WGPUVertexFormat_Float32x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float32x2', 29) # type: ignore
WGPUVertexFormat_Float32x3 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float32x3', 30) # type: ignore
WGPUVertexFormat_Float32x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Float32x4', 31) # type: ignore
WGPUVertexFormat_Uint32 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint32', 32) # type: ignore
WGPUVertexFormat_Uint32x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint32x2', 33) # type: ignore
WGPUVertexFormat_Uint32x3 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint32x3', 34) # type: ignore
WGPUVertexFormat_Uint32x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Uint32x4', 35) # type: ignore
WGPUVertexFormat_Sint32 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint32', 36) # type: ignore
WGPUVertexFormat_Sint32x2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint32x2', 37) # type: ignore
WGPUVertexFormat_Sint32x3 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint32x3', 38) # type: ignore
WGPUVertexFormat_Sint32x4 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Sint32x4', 39) # type: ignore
WGPUVertexFormat_Unorm10_10_10_2 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm10_10_10_2', 40) # type: ignore
WGPUVertexFormat_Unorm8x4BGRA = enum_WGPUVertexFormat.define('WGPUVertexFormat_Unorm8x4BGRA', 41) # type: ignore
WGPUVertexFormat_Force32 = enum_WGPUVertexFormat.define('WGPUVertexFormat_Force32', 2147483647) # type: ignore

WGPUVertexFormat: TypeAlias = enum_WGPUVertexFormat
@c.record
class struct_WGPUYCbCrVkDescriptor(c.Struct):
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
WGPUFilterMode_Undefined = enum_WGPUFilterMode.define('WGPUFilterMode_Undefined', 0) # type: ignore
WGPUFilterMode_Nearest = enum_WGPUFilterMode.define('WGPUFilterMode_Nearest', 1) # type: ignore
WGPUFilterMode_Linear = enum_WGPUFilterMode.define('WGPUFilterMode_Linear', 2) # type: ignore
WGPUFilterMode_Force32 = enum_WGPUFilterMode.define('WGPUFilterMode_Force32', 2147483647) # type: ignore

WGPUFilterMode: TypeAlias = enum_WGPUFilterMode
@c.record
class struct_WGPUAHardwareBufferProperties(c.Struct):
  SIZE = 72
  yCbCrInfo: Annotated[WGPUYCbCrVkDescriptor, 0]
WGPUYCbCrVkDescriptor = struct_WGPUYCbCrVkDescriptor
@c.record
class struct_WGPUAdapterInfo(c.Struct):
  SIZE = 96
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
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
WGPUAdapterType_DiscreteGPU = enum_WGPUAdapterType.define('WGPUAdapterType_DiscreteGPU', 1) # type: ignore
WGPUAdapterType_IntegratedGPU = enum_WGPUAdapterType.define('WGPUAdapterType_IntegratedGPU', 2) # type: ignore
WGPUAdapterType_CPU = enum_WGPUAdapterType.define('WGPUAdapterType_CPU', 3) # type: ignore
WGPUAdapterType_Unknown = enum_WGPUAdapterType.define('WGPUAdapterType_Unknown', 4) # type: ignore
WGPUAdapterType_Force32 = enum_WGPUAdapterType.define('WGPUAdapterType_Force32', 2147483647) # type: ignore

WGPUAdapterType: TypeAlias = enum_WGPUAdapterType
@c.record
class struct_WGPUAdapterPropertiesMemoryHeaps(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStructOut, 0]
  heapCount: Annotated[size_t, 16]
  heapInfo: Annotated[c.POINTER[WGPUMemoryHeapInfo], 24]
WGPUMemoryHeapInfo = struct_WGPUMemoryHeapInfo
@c.record
class struct_WGPUBindGroupDescriptor(c.Struct):
  SIZE = 48
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  layout: Annotated[WGPUBindGroupLayout, 24]
  entryCount: Annotated[size_t, 32]
  entries: Annotated[c.POINTER[WGPUBindGroupEntry], 40]
WGPUBindGroupEntry = struct_WGPUBindGroupEntry
@c.record
class struct_WGPUBindGroupLayoutEntry(c.Struct):
  SIZE = 112
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  binding: Annotated[uint32_t, 8]
  visibility: Annotated[WGPUShaderStage, 16]
  buffer: Annotated[WGPUBufferBindingLayout, 24]
  sampler: Annotated[WGPUSamplerBindingLayout, 48]
  texture: Annotated[WGPUTextureBindingLayout, 64]
  storageTexture: Annotated[WGPUStorageTextureBindingLayout, 88]
WGPUShaderStage = Annotated[int, ctypes.c_uint64]
WGPUBufferBindingLayout = struct_WGPUBufferBindingLayout
WGPUSamplerBindingLayout = struct_WGPUSamplerBindingLayout
WGPUTextureBindingLayout = struct_WGPUTextureBindingLayout
WGPUStorageTextureBindingLayout = struct_WGPUStorageTextureBindingLayout
@c.record
class struct_WGPUBlendState(c.Struct):
  SIZE = 24
  color: Annotated[WGPUBlendComponent, 0]
  alpha: Annotated[WGPUBlendComponent, 12]
WGPUBlendComponent = struct_WGPUBlendComponent
@c.record
class struct_WGPUBufferDescriptor(c.Struct):
  SIZE = 48
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  usage: Annotated[WGPUBufferUsage, 24]
  size: Annotated[uint64_t, 32]
  mappedAtCreation: Annotated[WGPUBool, 40]
@c.record
class struct_WGPUCommandBufferDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPUCommandEncoderDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPUComputePassDescriptor(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  timestampWrites: Annotated[c.POINTER[WGPUComputePassTimestampWrites], 24]
WGPUComputePassTimestampWrites = struct_WGPUComputePassTimestampWrites
@c.record
class struct_WGPUConstantEntry(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  key: Annotated[WGPUStringView, 8]
  value: Annotated[Annotated[float, ctypes.c_double], 24]
@c.record
class struct_WGPUDawnCacheDeviceDescriptor(c.Struct):
  SIZE = 56
  chain: Annotated[WGPUChainedStruct, 0]
  isolationKey: Annotated[WGPUStringView, 16]
  loadDataFunction: Annotated[WGPUDawnLoadCacheDataFunction, 32]
  storeDataFunction: Annotated[WGPUDawnStoreCacheDataFunction, 40]
  functionUserdata: Annotated[c.POINTER[None], 48]
WGPUDawnLoadCacheDataFunction = c.CFUNCTYPE(Annotated[int, ctypes.c_uint64], c.POINTER[None], Annotated[int, ctypes.c_uint64], c.POINTER[None], Annotated[int, ctypes.c_uint64], c.POINTER[None])
WGPUDawnStoreCacheDataFunction = c.CFUNCTYPE(None, c.POINTER[None], Annotated[int, ctypes.c_uint64], c.POINTER[None], Annotated[int, ctypes.c_uint64], c.POINTER[None])
@c.record
class struct_WGPUDepthStencilState(c.Struct):
  SIZE = 72
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
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
WGPUOptionalBool_False = enum_WGPUOptionalBool.define('WGPUOptionalBool_False', 0) # type: ignore
WGPUOptionalBool_True = enum_WGPUOptionalBool.define('WGPUOptionalBool_True', 1) # type: ignore
WGPUOptionalBool_Undefined = enum_WGPUOptionalBool.define('WGPUOptionalBool_Undefined', 2) # type: ignore
WGPUOptionalBool_Force32 = enum_WGPUOptionalBool.define('WGPUOptionalBool_Force32', 2147483647) # type: ignore

WGPUOptionalBool: TypeAlias = enum_WGPUOptionalBool
WGPUStencilFaceState = struct_WGPUStencilFaceState
@c.record
class struct_WGPUDrmFormatCapabilities(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStructOut, 0]
  propertiesCount: Annotated[size_t, 16]
  properties: Annotated[c.POINTER[WGPUDrmFormatProperties], 24]
WGPUDrmFormatProperties = struct_WGPUDrmFormatProperties
@c.record
class struct_WGPUExternalTextureDescriptor(c.Struct):
  SIZE = 112
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  plane0: Annotated[WGPUTextureView, 24]
  plane1: Annotated[WGPUTextureView, 32]
  cropOrigin: Annotated[WGPUOrigin2D, 40]
  cropSize: Annotated[WGPUExtent2D, 48]
  apparentSize: Annotated[WGPUExtent2D, 56]
  doYuvToRgbConversionOnly: Annotated[WGPUBool, 64]
  yuvToRgbConversionMatrix: Annotated[c.POINTER[Annotated[float, ctypes.c_float]], 72]
  srcTransferFunctionParameters: Annotated[c.POINTER[Annotated[float, ctypes.c_float]], 80]
  dstTransferFunctionParameters: Annotated[c.POINTER[Annotated[float, ctypes.c_float]], 88]
  gamutConversionMatrix: Annotated[c.POINTER[Annotated[float, ctypes.c_float]], 96]
  mirrored: Annotated[WGPUBool, 104]
  rotation: Annotated[WGPUExternalTextureRotation, 108]
WGPUOrigin2D = struct_WGPUOrigin2D
WGPUExtent2D = struct_WGPUExtent2D
enum_WGPUExternalTextureRotation = CEnum(Annotated[int, ctypes.c_uint32])
WGPUExternalTextureRotation_Rotate0Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate0Degrees', 1) # type: ignore
WGPUExternalTextureRotation_Rotate90Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate90Degrees', 2) # type: ignore
WGPUExternalTextureRotation_Rotate180Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate180Degrees', 3) # type: ignore
WGPUExternalTextureRotation_Rotate270Degrees = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Rotate270Degrees', 4) # type: ignore
WGPUExternalTextureRotation_Force32 = enum_WGPUExternalTextureRotation.define('WGPUExternalTextureRotation_Force32', 2147483647) # type: ignore

WGPUExternalTextureRotation: TypeAlias = enum_WGPUExternalTextureRotation
@c.record
class struct_WGPUFutureWaitInfo(c.Struct):
  SIZE = 16
  future: Annotated[WGPUFuture, 0]
  completed: Annotated[WGPUBool, 8]
WGPUFuture = struct_WGPUFuture
@c.record
class struct_WGPUImageCopyBuffer(c.Struct):
  SIZE = 32
  layout: Annotated[WGPUTextureDataLayout, 0]
  buffer: Annotated[WGPUBuffer, 24]
WGPUTextureDataLayout = struct_WGPUTextureDataLayout
@c.record
class struct_WGPUImageCopyExternalTexture(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  externalTexture: Annotated[WGPUExternalTexture, 8]
  origin: Annotated[WGPUOrigin3D, 16]
  naturalSize: Annotated[WGPUExtent2D, 28]
WGPUOrigin3D = struct_WGPUOrigin3D
@c.record
class struct_WGPUImageCopyTexture(c.Struct):
  SIZE = 32
  texture: Annotated[WGPUTexture, 0]
  mipLevel: Annotated[uint32_t, 8]
  origin: Annotated[WGPUOrigin3D, 12]
  aspect: Annotated[WGPUTextureAspect, 24]
enum_WGPUTextureAspect = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureAspect_Undefined = enum_WGPUTextureAspect.define('WGPUTextureAspect_Undefined', 0) # type: ignore
WGPUTextureAspect_All = enum_WGPUTextureAspect.define('WGPUTextureAspect_All', 1) # type: ignore
WGPUTextureAspect_StencilOnly = enum_WGPUTextureAspect.define('WGPUTextureAspect_StencilOnly', 2) # type: ignore
WGPUTextureAspect_DepthOnly = enum_WGPUTextureAspect.define('WGPUTextureAspect_DepthOnly', 3) # type: ignore
WGPUTextureAspect_Plane0Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane0Only', 327680) # type: ignore
WGPUTextureAspect_Plane1Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane1Only', 327681) # type: ignore
WGPUTextureAspect_Plane2Only = enum_WGPUTextureAspect.define('WGPUTextureAspect_Plane2Only', 327682) # type: ignore
WGPUTextureAspect_Force32 = enum_WGPUTextureAspect.define('WGPUTextureAspect_Force32', 2147483647) # type: ignore

WGPUTextureAspect: TypeAlias = enum_WGPUTextureAspect
@c.record
class struct_WGPUInstanceDescriptor(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  features: Annotated[WGPUInstanceFeatures, 8]
WGPUInstanceFeatures = struct_WGPUInstanceFeatures
@c.record
class struct_WGPUPipelineLayoutDescriptor(c.Struct):
  SIZE = 48
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  bindGroupLayoutCount: Annotated[size_t, 24]
  bindGroupLayouts: Annotated[c.POINTER[WGPUBindGroupLayout], 32]
  immediateDataRangeByteSize: Annotated[uint32_t, 40]
@c.record
class struct_WGPUPipelineLayoutPixelLocalStorage(c.Struct):
  SIZE = 40
  chain: Annotated[WGPUChainedStruct, 0]
  totalPixelLocalStorageSize: Annotated[uint64_t, 16]
  storageAttachmentCount: Annotated[size_t, 24]
  storageAttachments: Annotated[c.POINTER[WGPUPipelineLayoutStorageAttachment], 32]
WGPUPipelineLayoutStorageAttachment = struct_WGPUPipelineLayoutStorageAttachment
@c.record
class struct_WGPUQuerySetDescriptor(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  type: Annotated[WGPUQueryType, 24]
  count: Annotated[uint32_t, 28]
enum_WGPUQueryType = CEnum(Annotated[int, ctypes.c_uint32])
WGPUQueryType_Occlusion = enum_WGPUQueryType.define('WGPUQueryType_Occlusion', 1) # type: ignore
WGPUQueryType_Timestamp = enum_WGPUQueryType.define('WGPUQueryType_Timestamp', 2) # type: ignore
WGPUQueryType_Force32 = enum_WGPUQueryType.define('WGPUQueryType_Force32', 2147483647) # type: ignore

WGPUQueryType: TypeAlias = enum_WGPUQueryType
@c.record
class struct_WGPUQueueDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPURenderBundleDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPURenderBundleEncoderDescriptor(c.Struct):
  SIZE = 56
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  colorFormatCount: Annotated[size_t, 24]
  colorFormats: Annotated[c.POINTER[WGPUTextureFormat], 32]
  depthStencilFormat: Annotated[WGPUTextureFormat, 40]
  sampleCount: Annotated[uint32_t, 44]
  depthReadOnly: Annotated[WGPUBool, 48]
  stencilReadOnly: Annotated[WGPUBool, 52]
@c.record
class struct_WGPURenderPassColorAttachment(c.Struct):
  SIZE = 72
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  view: Annotated[WGPUTextureView, 8]
  depthSlice: Annotated[uint32_t, 16]
  resolveTarget: Annotated[WGPUTextureView, 24]
  loadOp: Annotated[WGPULoadOp, 32]
  storeOp: Annotated[WGPUStoreOp, 36]
  clearValue: Annotated[WGPUColor, 40]
WGPUColor = struct_WGPUColor
@c.record
class struct_WGPURenderPassStorageAttachment(c.Struct):
  SIZE = 64
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  offset: Annotated[uint64_t, 8]
  storage: Annotated[WGPUTextureView, 16]
  loadOp: Annotated[WGPULoadOp, 24]
  storeOp: Annotated[WGPUStoreOp, 28]
  clearValue: Annotated[WGPUColor, 32]
@c.record
class struct_WGPURequiredLimits(c.Struct):
  SIZE = 168
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  limits: Annotated[WGPULimits, 8]
WGPULimits = struct_WGPULimits
@c.record
class struct_WGPUSamplerDescriptor(c.Struct):
  SIZE = 64
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
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
WGPUAddressMode_Undefined = enum_WGPUAddressMode.define('WGPUAddressMode_Undefined', 0) # type: ignore
WGPUAddressMode_ClampToEdge = enum_WGPUAddressMode.define('WGPUAddressMode_ClampToEdge', 1) # type: ignore
WGPUAddressMode_Repeat = enum_WGPUAddressMode.define('WGPUAddressMode_Repeat', 2) # type: ignore
WGPUAddressMode_MirrorRepeat = enum_WGPUAddressMode.define('WGPUAddressMode_MirrorRepeat', 3) # type: ignore
WGPUAddressMode_Force32 = enum_WGPUAddressMode.define('WGPUAddressMode_Force32', 2147483647) # type: ignore

WGPUAddressMode: TypeAlias = enum_WGPUAddressMode
enum_WGPUMipmapFilterMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUMipmapFilterMode_Undefined = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Undefined', 0) # type: ignore
WGPUMipmapFilterMode_Nearest = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Nearest', 1) # type: ignore
WGPUMipmapFilterMode_Linear = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Linear', 2) # type: ignore
WGPUMipmapFilterMode_Force32 = enum_WGPUMipmapFilterMode.define('WGPUMipmapFilterMode_Force32', 2147483647) # type: ignore

WGPUMipmapFilterMode: TypeAlias = enum_WGPUMipmapFilterMode
uint16_t = Annotated[int, ctypes.c_uint16]
@c.record
class struct_WGPUShaderModuleDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPUShaderSourceWGSL(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  code: Annotated[WGPUStringView, 16]
@c.record
class struct_WGPUSharedBufferMemoryDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPUSharedFenceDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPUSharedTextureMemoryAHardwareBufferProperties(c.Struct):
  SIZE = 88
  chain: Annotated[WGPUChainedStructOut, 0]
  yCbCrInfo: Annotated[WGPUYCbCrVkDescriptor, 16]
@c.record
class struct_WGPUSharedTextureMemoryDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPUSharedTextureMemoryDmaBufDescriptor(c.Struct):
  SIZE = 56
  chain: Annotated[WGPUChainedStruct, 0]
  size: Annotated[WGPUExtent3D, 16]
  drmFormat: Annotated[uint32_t, 28]
  drmModifier: Annotated[uint64_t, 32]
  planeCount: Annotated[size_t, 40]
  planes: Annotated[c.POINTER[WGPUSharedTextureMemoryDmaBufPlane], 48]
WGPUExtent3D = struct_WGPUExtent3D
WGPUSharedTextureMemoryDmaBufPlane = struct_WGPUSharedTextureMemoryDmaBufPlane
@c.record
class struct_WGPUSharedTextureMemoryProperties(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
  usage: Annotated[WGPUTextureUsage, 8]
  size: Annotated[WGPUExtent3D, 16]
  format: Annotated[WGPUTextureFormat, 28]
@c.record
class struct_WGPUSupportedLimits(c.Struct):
  SIZE = 168
  nextInChain: Annotated[c.POINTER[WGPUChainedStructOut], 0]
  limits: Annotated[WGPULimits, 8]
@c.record
class struct_WGPUSurfaceDescriptor(c.Struct):
  SIZE = 24
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
@c.record
class struct_WGPUSurfaceSourceCanvasHTMLSelector_Emscripten(c.Struct):
  SIZE = 32
  chain: Annotated[WGPUChainedStruct, 0]
  selector: Annotated[WGPUStringView, 16]
@c.record
class struct_WGPUTextureDescriptor(c.Struct):
  SIZE = 80
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  usage: Annotated[WGPUTextureUsage, 24]
  dimension: Annotated[WGPUTextureDimension, 32]
  size: Annotated[WGPUExtent3D, 36]
  format: Annotated[WGPUTextureFormat, 48]
  mipLevelCount: Annotated[uint32_t, 52]
  sampleCount: Annotated[uint32_t, 56]
  viewFormatCount: Annotated[size_t, 64]
  viewFormats: Annotated[c.POINTER[WGPUTextureFormat], 72]
enum_WGPUTextureDimension = CEnum(Annotated[int, ctypes.c_uint32])
WGPUTextureDimension_Undefined = enum_WGPUTextureDimension.define('WGPUTextureDimension_Undefined', 0) # type: ignore
WGPUTextureDimension_1D = enum_WGPUTextureDimension.define('WGPUTextureDimension_1D', 1) # type: ignore
WGPUTextureDimension_2D = enum_WGPUTextureDimension.define('WGPUTextureDimension_2D', 2) # type: ignore
WGPUTextureDimension_3D = enum_WGPUTextureDimension.define('WGPUTextureDimension_3D', 3) # type: ignore
WGPUTextureDimension_Force32 = enum_WGPUTextureDimension.define('WGPUTextureDimension_Force32', 2147483647) # type: ignore

WGPUTextureDimension: TypeAlias = enum_WGPUTextureDimension
@c.record
class struct_WGPUTextureViewDescriptor(c.Struct):
  SIZE = 64
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  format: Annotated[WGPUTextureFormat, 24]
  dimension: Annotated[WGPUTextureViewDimension, 28]
  baseMipLevel: Annotated[uint32_t, 32]
  mipLevelCount: Annotated[uint32_t, 36]
  baseArrayLayer: Annotated[uint32_t, 40]
  arrayLayerCount: Annotated[uint32_t, 44]
  aspect: Annotated[WGPUTextureAspect, 48]
  usage: Annotated[WGPUTextureUsage, 56]
@c.record
class struct_WGPUVertexBufferLayout(c.Struct):
  SIZE = 32
  arrayStride: Annotated[uint64_t, 0]
  stepMode: Annotated[WGPUVertexStepMode, 8]
  attributeCount: Annotated[size_t, 16]
  attributes: Annotated[c.POINTER[WGPUVertexAttribute], 24]
enum_WGPUVertexStepMode = CEnum(Annotated[int, ctypes.c_uint32])
WGPUVertexStepMode_Undefined = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Undefined', 0) # type: ignore
WGPUVertexStepMode_Vertex = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Vertex', 1) # type: ignore
WGPUVertexStepMode_Instance = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Instance', 2) # type: ignore
WGPUVertexStepMode_Force32 = enum_WGPUVertexStepMode.define('WGPUVertexStepMode_Force32', 2147483647) # type: ignore

WGPUVertexStepMode: TypeAlias = enum_WGPUVertexStepMode
WGPUVertexAttribute = struct_WGPUVertexAttribute
@c.record
class struct_WGPUBindGroupLayoutDescriptor(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  entryCount: Annotated[size_t, 24]
  entries: Annotated[c.POINTER[WGPUBindGroupLayoutEntry], 32]
WGPUBindGroupLayoutEntry = struct_WGPUBindGroupLayoutEntry
@c.record
class struct_WGPUColorTargetState(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  format: Annotated[WGPUTextureFormat, 8]
  blend: Annotated[c.POINTER[WGPUBlendState], 16]
  writeMask: Annotated[WGPUColorWriteMask, 24]
WGPUBlendState = struct_WGPUBlendState
WGPUColorWriteMask = Annotated[int, ctypes.c_uint64]
@c.record
class struct_WGPUComputeState(c.Struct):
  SIZE = 48
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  module: Annotated[WGPUShaderModule, 8]
  entryPoint: Annotated[WGPUStringView, 16]
  constantCount: Annotated[size_t, 32]
  constants: Annotated[c.POINTER[WGPUConstantEntry], 40]
WGPUConstantEntry = struct_WGPUConstantEntry
@c.record
class struct_WGPUDeviceDescriptor(c.Struct):
  SIZE = 144
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  requiredFeatureCount: Annotated[size_t, 24]
  requiredFeatures: Annotated[c.POINTER[WGPUFeatureName], 32]
  requiredLimits: Annotated[c.POINTER[WGPURequiredLimits], 40]
  defaultQueue: Annotated[WGPUQueueDescriptor, 48]
  deviceLostCallbackInfo2: Annotated[WGPUDeviceLostCallbackInfo2, 72]
  uncapturedErrorCallbackInfo2: Annotated[WGPUUncapturedErrorCallbackInfo2, 112]
WGPURequiredLimits = struct_WGPURequiredLimits
WGPUQueueDescriptor = struct_WGPUQueueDescriptor
@c.record
class struct_WGPUDeviceLostCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUDeviceLostCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
WGPUDeviceLostCallbackInfo2: TypeAlias = struct_WGPUDeviceLostCallbackInfo2
WGPUDeviceLostCallback2: TypeAlias = c.CFUNCTYPE(None, c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], enum_WGPUDeviceLostReason, struct_WGPUStringView, c.POINTER[None], c.POINTER[None])
@c.record
class struct_WGPUUncapturedErrorCallbackInfo2(c.Struct):
  SIZE = 32
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  callback: Annotated[WGPUUncapturedErrorCallback, 8]
  userdata1: Annotated[c.POINTER[None], 16]
  userdata2: Annotated[c.POINTER[None], 24]
WGPUUncapturedErrorCallbackInfo2: TypeAlias = struct_WGPUUncapturedErrorCallbackInfo2
WGPUUncapturedErrorCallback: TypeAlias = c.CFUNCTYPE(None, c.POINTER[c.POINTER[struct_WGPUDeviceImpl]], enum_WGPUErrorType, struct_WGPUStringView, c.POINTER[None], c.POINTER[None])
@c.record
class struct_WGPURenderPassDescriptor(c.Struct):
  SIZE = 64
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  colorAttachmentCount: Annotated[size_t, 24]
  colorAttachments: Annotated[c.POINTER[WGPURenderPassColorAttachment], 32]
  depthStencilAttachment: Annotated[c.POINTER[WGPURenderPassDepthStencilAttachment], 40]
  occlusionQuerySet: Annotated[WGPUQuerySet, 48]
  timestampWrites: Annotated[c.POINTER[WGPURenderPassTimestampWrites], 56]
WGPURenderPassColorAttachment = struct_WGPURenderPassColorAttachment
WGPURenderPassDepthStencilAttachment = struct_WGPURenderPassDepthStencilAttachment
WGPURenderPassTimestampWrites = struct_WGPURenderPassTimestampWrites
@c.record
class struct_WGPURenderPassPixelLocalStorage(c.Struct):
  SIZE = 40
  chain: Annotated[WGPUChainedStruct, 0]
  totalPixelLocalStorageSize: Annotated[uint64_t, 16]
  storageAttachmentCount: Annotated[size_t, 24]
  storageAttachments: Annotated[c.POINTER[WGPURenderPassStorageAttachment], 32]
WGPURenderPassStorageAttachment = struct_WGPURenderPassStorageAttachment
@c.record
class struct_WGPUVertexState(c.Struct):
  SIZE = 64
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  module: Annotated[WGPUShaderModule, 8]
  entryPoint: Annotated[WGPUStringView, 16]
  constantCount: Annotated[size_t, 32]
  constants: Annotated[c.POINTER[WGPUConstantEntry], 40]
  bufferCount: Annotated[size_t, 48]
  buffers: Annotated[c.POINTER[WGPUVertexBufferLayout], 56]
WGPUVertexBufferLayout = struct_WGPUVertexBufferLayout
@c.record
class struct_WGPUComputePipelineDescriptor(c.Struct):
  SIZE = 80
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  layout: Annotated[WGPUPipelineLayout, 24]
  compute: Annotated[WGPUComputeState, 32]
WGPUComputeState = struct_WGPUComputeState
@c.record
class struct_WGPUFragmentState(c.Struct):
  SIZE = 64
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  module: Annotated[WGPUShaderModule, 8]
  entryPoint: Annotated[WGPUStringView, 16]
  constantCount: Annotated[size_t, 32]
  constants: Annotated[c.POINTER[WGPUConstantEntry], 40]
  targetCount: Annotated[size_t, 48]
  targets: Annotated[c.POINTER[WGPUColorTargetState], 56]
WGPUColorTargetState = struct_WGPUColorTargetState
@c.record
class struct_WGPURenderPipelineDescriptor(c.Struct):
  SIZE = 168
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  label: Annotated[WGPUStringView, 8]
  layout: Annotated[WGPUPipelineLayout, 24]
  vertex: Annotated[WGPUVertexState, 32]
  primitive: Annotated[WGPUPrimitiveState, 96]
  depthStencil: Annotated[c.POINTER[WGPUDepthStencilState], 128]
  multisample: Annotated[WGPUMultisampleState, 136]
  fragment: Annotated[c.POINTER[WGPUFragmentState], 160]
WGPUVertexState = struct_WGPUVertexState
WGPUPrimitiveState = struct_WGPUPrimitiveState
WGPUDepthStencilState = struct_WGPUDepthStencilState
WGPUMultisampleState = struct_WGPUMultisampleState
WGPUFragmentState = struct_WGPUFragmentState
enum_WGPUWGSLFeatureName = CEnum(Annotated[int, ctypes.c_uint32])
WGPUWGSLFeatureName_ReadonlyAndReadwriteStorageTextures = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ReadonlyAndReadwriteStorageTextures', 1) # type: ignore
WGPUWGSLFeatureName_Packed4x8IntegerDotProduct = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_Packed4x8IntegerDotProduct', 2) # type: ignore
WGPUWGSLFeatureName_UnrestrictedPointerParameters = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_UnrestrictedPointerParameters', 3) # type: ignore
WGPUWGSLFeatureName_PointerCompositeAccess = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_PointerCompositeAccess', 4) # type: ignore
WGPUWGSLFeatureName_ChromiumTestingUnimplemented = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingUnimplemented', 327680) # type: ignore
WGPUWGSLFeatureName_ChromiumTestingUnsafeExperimental = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingUnsafeExperimental', 327681) # type: ignore
WGPUWGSLFeatureName_ChromiumTestingExperimental = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingExperimental', 327682) # type: ignore
WGPUWGSLFeatureName_ChromiumTestingShippedWithKillswitch = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingShippedWithKillswitch', 327683) # type: ignore
WGPUWGSLFeatureName_ChromiumTestingShipped = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_ChromiumTestingShipped', 327684) # type: ignore
WGPUWGSLFeatureName_Force32 = enum_WGPUWGSLFeatureName.define('WGPUWGSLFeatureName_Force32', 2147483647) # type: ignore

WGPUWGSLFeatureName: TypeAlias = enum_WGPUWGSLFeatureName
WGPUBufferMapAsyncStatus: TypeAlias = enum_WGPUBufferMapAsyncStatus
enum_WGPUBufferMapState = CEnum(Annotated[int, ctypes.c_uint32])
WGPUBufferMapState_Unmapped = enum_WGPUBufferMapState.define('WGPUBufferMapState_Unmapped', 1) # type: ignore
WGPUBufferMapState_Pending = enum_WGPUBufferMapState.define('WGPUBufferMapState_Pending', 2) # type: ignore
WGPUBufferMapState_Mapped = enum_WGPUBufferMapState.define('WGPUBufferMapState_Mapped', 3) # type: ignore
WGPUBufferMapState_Force32 = enum_WGPUBufferMapState.define('WGPUBufferMapState_Force32', 2147483647) # type: ignore

WGPUBufferMapState: TypeAlias = enum_WGPUBufferMapState
WGPUCompilationInfoRequestStatus: TypeAlias = enum_WGPUCompilationInfoRequestStatus
WGPUCreatePipelineAsyncStatus: TypeAlias = enum_WGPUCreatePipelineAsyncStatus
WGPUDeviceLostReason: TypeAlias = enum_WGPUDeviceLostReason
enum_WGPUErrorFilter = CEnum(Annotated[int, ctypes.c_uint32])
WGPUErrorFilter_Validation = enum_WGPUErrorFilter.define('WGPUErrorFilter_Validation', 1) # type: ignore
WGPUErrorFilter_OutOfMemory = enum_WGPUErrorFilter.define('WGPUErrorFilter_OutOfMemory', 2) # type: ignore
WGPUErrorFilter_Internal = enum_WGPUErrorFilter.define('WGPUErrorFilter_Internal', 3) # type: ignore
WGPUErrorFilter_Force32 = enum_WGPUErrorFilter.define('WGPUErrorFilter_Force32', 2147483647) # type: ignore

WGPUErrorFilter: TypeAlias = enum_WGPUErrorFilter
WGPUErrorType: TypeAlias = enum_WGPUErrorType
enum_WGPULoggingType = CEnum(Annotated[int, ctypes.c_uint32])
WGPULoggingType_Verbose = enum_WGPULoggingType.define('WGPULoggingType_Verbose', 1) # type: ignore
WGPULoggingType_Info = enum_WGPULoggingType.define('WGPULoggingType_Info', 2) # type: ignore
WGPULoggingType_Warning = enum_WGPULoggingType.define('WGPULoggingType_Warning', 3) # type: ignore
WGPULoggingType_Error = enum_WGPULoggingType.define('WGPULoggingType_Error', 4) # type: ignore
WGPULoggingType_Force32 = enum_WGPULoggingType.define('WGPULoggingType_Force32', 2147483647) # type: ignore

WGPULoggingType: TypeAlias = enum_WGPULoggingType
enum_WGPUMapAsyncStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUMapAsyncStatus_Success = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Success', 1) # type: ignore
WGPUMapAsyncStatus_InstanceDropped = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_InstanceDropped', 2) # type: ignore
WGPUMapAsyncStatus_Error = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Error', 3) # type: ignore
WGPUMapAsyncStatus_Aborted = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Aborted', 4) # type: ignore
WGPUMapAsyncStatus_Unknown = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Unknown', 5) # type: ignore
WGPUMapAsyncStatus_Force32 = enum_WGPUMapAsyncStatus.define('WGPUMapAsyncStatus_Force32', 2147483647) # type: ignore

WGPUMapAsyncStatus: TypeAlias = enum_WGPUMapAsyncStatus
WGPUPopErrorScopeStatus: TypeAlias = enum_WGPUPopErrorScopeStatus
WGPUQueueWorkDoneStatus: TypeAlias = enum_WGPUQueueWorkDoneStatus
WGPURequestAdapterStatus: TypeAlias = enum_WGPURequestAdapterStatus
WGPURequestDeviceStatus: TypeAlias = enum_WGPURequestDeviceStatus
enum_WGPUStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUStatus_Success = enum_WGPUStatus.define('WGPUStatus_Success', 1) # type: ignore
WGPUStatus_Error = enum_WGPUStatus.define('WGPUStatus_Error', 2) # type: ignore
WGPUStatus_Force32 = enum_WGPUStatus.define('WGPUStatus_Force32', 2147483647) # type: ignore

WGPUStatus: TypeAlias = enum_WGPUStatus
enum_WGPUWaitStatus = CEnum(Annotated[int, ctypes.c_uint32])
WGPUWaitStatus_Success = enum_WGPUWaitStatus.define('WGPUWaitStatus_Success', 1) # type: ignore
WGPUWaitStatus_TimedOut = enum_WGPUWaitStatus.define('WGPUWaitStatus_TimedOut', 2) # type: ignore
WGPUWaitStatus_UnsupportedTimeout = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedTimeout', 3) # type: ignore
WGPUWaitStatus_UnsupportedCount = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedCount', 4) # type: ignore
WGPUWaitStatus_UnsupportedMixedSources = enum_WGPUWaitStatus.define('WGPUWaitStatus_UnsupportedMixedSources', 5) # type: ignore
WGPUWaitStatus_Unknown = enum_WGPUWaitStatus.define('WGPUWaitStatus_Unknown', 6) # type: ignore
WGPUWaitStatus_Force32 = enum_WGPUWaitStatus.define('WGPUWaitStatus_Force32', 2147483647) # type: ignore

WGPUWaitStatus: TypeAlias = enum_WGPUWaitStatus
WGPUMapMode = Annotated[int, ctypes.c_uint64]
WGPUDeviceLostCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPUDeviceLostReason, struct_WGPUStringView, c.POINTER[None])
WGPULoggingCallback: TypeAlias = c.CFUNCTYPE(None, enum_WGPULoggingType, struct_WGPUStringView, c.POINTER[None])
WGPUProc = c.CFUNCTYPE(None, )
WGPUBufferMapCallback2: TypeAlias = c.CFUNCTYPE(None, enum_WGPUMapAsyncStatus, struct_WGPUStringView, c.POINTER[None], c.POINTER[None])
WGPUCompilationInfoCallback2: TypeAlias = c.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, c.POINTER[struct_WGPUCompilationInfo], c.POINTER[None], c.POINTER[None])
WGPUCreateComputePipelineAsyncCallback2: TypeAlias = c.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView, c.POINTER[None], c.POINTER[None])
WGPUCreateRenderPipelineAsyncCallback2: TypeAlias = c.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView, c.POINTER[None], c.POINTER[None])
WGPUPopErrorScopeCallback2: TypeAlias = c.CFUNCTYPE(None, enum_WGPUPopErrorScopeStatus, enum_WGPUErrorType, struct_WGPUStringView, c.POINTER[None], c.POINTER[None])
WGPUQueueWorkDoneCallback2: TypeAlias = c.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, c.POINTER[None], c.POINTER[None])
WGPURequestAdapterCallback2: TypeAlias = c.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, c.POINTER[struct_WGPUAdapterImpl], struct_WGPUStringView, c.POINTER[None], c.POINTER[None])
WGPURequestDeviceCallback2: TypeAlias = c.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView, c.POINTER[None], c.POINTER[None])
@c.record
class struct_WGPUBufferMapCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUBufferMapCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
WGPUBufferMapCallbackInfo2 = struct_WGPUBufferMapCallbackInfo2
@c.record
class struct_WGPUCompilationInfoCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCompilationInfoCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
WGPUCompilationInfoCallbackInfo2 = struct_WGPUCompilationInfoCallbackInfo2
@c.record
class struct_WGPUCreateComputePipelineAsyncCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCreateComputePipelineAsyncCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
WGPUCreateComputePipelineAsyncCallbackInfo2 = struct_WGPUCreateComputePipelineAsyncCallbackInfo2
@c.record
class struct_WGPUCreateRenderPipelineAsyncCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUCreateRenderPipelineAsyncCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
WGPUCreateRenderPipelineAsyncCallbackInfo2 = struct_WGPUCreateRenderPipelineAsyncCallbackInfo2
@c.record
class struct_WGPUPopErrorScopeCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUPopErrorScopeCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
WGPUPopErrorScopeCallbackInfo2 = struct_WGPUPopErrorScopeCallbackInfo2
@c.record
class struct_WGPUQueueWorkDoneCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPUQueueWorkDoneCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
WGPUQueueWorkDoneCallbackInfo2 = struct_WGPUQueueWorkDoneCallbackInfo2
@c.record
class struct_WGPURequestAdapterCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPURequestAdapterCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
WGPURequestAdapterCallbackInfo2 = struct_WGPURequestAdapterCallbackInfo2
@c.record
class struct_WGPURequestDeviceCallbackInfo2(c.Struct):
  SIZE = 40
  nextInChain: Annotated[c.POINTER[WGPUChainedStruct], 0]
  mode: Annotated[WGPUCallbackMode, 8]
  callback: Annotated[WGPURequestDeviceCallback2, 16]
  userdata1: Annotated[c.POINTER[None], 24]
  userdata2: Annotated[c.POINTER[None], 32]
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
WGPUProcAdapterInfoFreeMembers = c.CFUNCTYPE(None, struct_WGPUAdapterInfo)
WGPUProcAdapterPropertiesMemoryHeapsFreeMembers = c.CFUNCTYPE(None, struct_WGPUAdapterPropertiesMemoryHeaps)
WGPUProcCreateInstance = c.CFUNCTYPE(c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPUInstanceDescriptor])
WGPUProcDrmFormatCapabilitiesFreeMembers = c.CFUNCTYPE(None, struct_WGPUDrmFormatCapabilities)
WGPUProcGetInstanceFeatures: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUInstanceFeatures])
WGPUProcGetProcAddress = c.CFUNCTYPE(c.CFUNCTYPE(None, ), struct_WGPUStringView)
WGPUProcSharedBufferMemoryEndAccessStateFreeMembers = c.CFUNCTYPE(None, struct_WGPUSharedBufferMemoryEndAccessState)
WGPUProcSharedTextureMemoryEndAccessStateFreeMembers = c.CFUNCTYPE(None, struct_WGPUSharedTextureMemoryEndAccessState)
WGPUProcSupportedFeaturesFreeMembers = c.CFUNCTYPE(None, struct_WGPUSupportedFeatures)
WGPUProcSurfaceCapabilitiesFreeMembers = c.CFUNCTYPE(None, struct_WGPUSurfaceCapabilities)
WGPUProcAdapterCreateDevice = c.CFUNCTYPE(c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceDescriptor])
WGPUProcAdapterGetFeatures = c.CFUNCTYPE(None, c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUSupportedFeatures])
WGPUProcAdapterGetFormatCapabilities: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUAdapterImpl], enum_WGPUTextureFormat, c.POINTER[struct_WGPUFormatCapabilities])
WGPUProcAdapterGetInfo: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUAdapterInfo])
WGPUProcAdapterGetInstance = c.CFUNCTYPE(c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPUAdapterImpl])
WGPUProcAdapterGetLimits: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUSupportedLimits])
WGPUProcAdapterHasFeature: TypeAlias = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUAdapterImpl], enum_WGPUFeatureName)
WGPUProcAdapterRequestDevice: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceDescriptor], c.CFUNCTYPE(None, enum_WGPURequestDeviceStatus, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView, c.POINTER[None]), c.POINTER[None])
WGPUProcAdapterRequestDevice2 = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceDescriptor], struct_WGPURequestDeviceCallbackInfo2)
WGPUProcAdapterRequestDeviceF = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceDescriptor], struct_WGPURequestDeviceCallbackInfo)
WGPUProcAdapterAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUAdapterImpl])
WGPUProcAdapterRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUAdapterImpl])
WGPUProcBindGroupSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBindGroupImpl], struct_WGPUStringView)
WGPUProcBindGroupAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBindGroupImpl])
WGPUProcBindGroupRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBindGroupImpl])
WGPUProcBindGroupLayoutSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBindGroupLayoutImpl], struct_WGPUStringView)
WGPUProcBindGroupLayoutAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBindGroupLayoutImpl])
WGPUProcBindGroupLayoutRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBindGroupLayoutImpl])
WGPUProcBufferDestroy = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBufferImpl])
WGPUProcBufferGetConstMappedRange = c.CFUNCTYPE(c.POINTER[None], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcBufferGetMapState: TypeAlias = c.CFUNCTYPE(enum_WGPUBufferMapState, c.POINTER[struct_WGPUBufferImpl])
WGPUProcBufferGetMappedRange = c.CFUNCTYPE(c.POINTER[None], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcBufferGetSize = c.CFUNCTYPE(Annotated[int, ctypes.c_uint64], c.POINTER[struct_WGPUBufferImpl])
WGPUProcBufferGetUsage = c.CFUNCTYPE(Annotated[int, ctypes.c_uint64], c.POINTER[struct_WGPUBufferImpl])
WGPUProcBufferMapAsync: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], c.CFUNCTYPE(None, enum_WGPUBufferMapAsyncStatus, c.POINTER[None]), c.POINTER[None])
WGPUProcBufferMapAsync2 = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], struct_WGPUBufferMapCallbackInfo2)
WGPUProcBufferMapAsyncF = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64], struct_WGPUBufferMapCallbackInfo)
WGPUProcBufferSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBufferImpl], struct_WGPUStringView)
WGPUProcBufferUnmap = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBufferImpl])
WGPUProcBufferAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBufferImpl])
WGPUProcBufferRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUBufferImpl])
WGPUProcCommandBufferSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandBufferImpl], struct_WGPUStringView)
WGPUProcCommandBufferAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandBufferImpl])
WGPUProcCommandBufferRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandBufferImpl])
WGPUProcCommandEncoderBeginComputePass = c.CFUNCTYPE(c.POINTER[struct_WGPUComputePassEncoderImpl], c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUComputePassDescriptor])
WGPUProcCommandEncoderBeginRenderPass = c.CFUNCTYPE(c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPURenderPassDescriptor])
WGPUProcCommandEncoderClearBuffer = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcCommandEncoderCopyBufferToBuffer = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcCommandEncoderCopyBufferToTexture = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUImageCopyBuffer], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUExtent3D])
WGPUProcCommandEncoderCopyTextureToBuffer = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUImageCopyBuffer], c.POINTER[struct_WGPUExtent3D])
WGPUProcCommandEncoderCopyTextureToTexture = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUExtent3D])
WGPUProcCommandEncoderFinish = c.CFUNCTYPE(c.POINTER[struct_WGPUCommandBufferImpl], c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUCommandBufferDescriptor])
WGPUProcCommandEncoderInjectValidationError = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], struct_WGPUStringView)
WGPUProcCommandEncoderInsertDebugMarker = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], struct_WGPUStringView)
WGPUProcCommandEncoderPopDebugGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl])
WGPUProcCommandEncoderPushDebugGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], struct_WGPUStringView)
WGPUProcCommandEncoderResolveQuerySet = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUQuerySetImpl], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64])
WGPUProcCommandEncoderSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], struct_WGPUStringView)
WGPUProcCommandEncoderWriteBuffer = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], c.POINTER[Annotated[int, ctypes.c_ubyte]], Annotated[int, ctypes.c_uint64])
WGPUProcCommandEncoderWriteTimestamp = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUQuerySetImpl], Annotated[int, ctypes.c_uint32])
WGPUProcCommandEncoderAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl])
WGPUProcCommandEncoderRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUCommandEncoderImpl])
WGPUProcComputePassEncoderDispatchWorkgroups = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32])
WGPUProcComputePassEncoderDispatchWorkgroupsIndirect = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64])
WGPUProcComputePassEncoderEnd = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl])
WGPUProcComputePassEncoderInsertDebugMarker = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl], struct_WGPUStringView)
WGPUProcComputePassEncoderPopDebugGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl])
WGPUProcComputePassEncoderPushDebugGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl], struct_WGPUStringView)
WGPUProcComputePassEncoderSetBindGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl], Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUBindGroupImpl], Annotated[int, ctypes.c_uint64], c.POINTER[Annotated[int, ctypes.c_uint32]])
WGPUProcComputePassEncoderSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl], struct_WGPUStringView)
WGPUProcComputePassEncoderSetPipeline = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl], c.POINTER[struct_WGPUComputePipelineImpl])
WGPUProcComputePassEncoderWriteTimestamp = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl], c.POINTER[struct_WGPUQuerySetImpl], Annotated[int, ctypes.c_uint32])
WGPUProcComputePassEncoderAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl])
WGPUProcComputePassEncoderRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePassEncoderImpl])
WGPUProcComputePipelineGetBindGroupLayout = c.CFUNCTYPE(c.POINTER[struct_WGPUBindGroupLayoutImpl], c.POINTER[struct_WGPUComputePipelineImpl], Annotated[int, ctypes.c_uint32])
WGPUProcComputePipelineSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView)
WGPUProcComputePipelineAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePipelineImpl])
WGPUProcComputePipelineRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUComputePipelineImpl])
WGPUProcDeviceCreateBindGroup = c.CFUNCTYPE(c.POINTER[struct_WGPUBindGroupImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUBindGroupDescriptor])
WGPUProcDeviceCreateBindGroupLayout = c.CFUNCTYPE(c.POINTER[struct_WGPUBindGroupLayoutImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUBindGroupLayoutDescriptor])
WGPUProcDeviceCreateBuffer = c.CFUNCTYPE(c.POINTER[struct_WGPUBufferImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUBufferDescriptor])
WGPUProcDeviceCreateCommandEncoder = c.CFUNCTYPE(c.POINTER[struct_WGPUCommandEncoderImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUCommandEncoderDescriptor])
WGPUProcDeviceCreateComputePipeline = c.CFUNCTYPE(c.POINTER[struct_WGPUComputePipelineImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUComputePipelineDescriptor])
WGPUProcDeviceCreateComputePipelineAsync: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUComputePipelineDescriptor], c.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, c.POINTER[struct_WGPUComputePipelineImpl], struct_WGPUStringView, c.POINTER[None]), c.POINTER[None])
WGPUProcDeviceCreateComputePipelineAsync2 = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUComputePipelineDescriptor], struct_WGPUCreateComputePipelineAsyncCallbackInfo2)
WGPUProcDeviceCreateComputePipelineAsyncF = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUComputePipelineDescriptor], struct_WGPUCreateComputePipelineAsyncCallbackInfo)
WGPUProcDeviceCreateErrorBuffer = c.CFUNCTYPE(c.POINTER[struct_WGPUBufferImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUBufferDescriptor])
WGPUProcDeviceCreateErrorExternalTexture = c.CFUNCTYPE(c.POINTER[struct_WGPUExternalTextureImpl], c.POINTER[struct_WGPUDeviceImpl])
WGPUProcDeviceCreateErrorShaderModule = c.CFUNCTYPE(c.POINTER[struct_WGPUShaderModuleImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUShaderModuleDescriptor], struct_WGPUStringView)
WGPUProcDeviceCreateErrorTexture = c.CFUNCTYPE(c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUTextureDescriptor])
WGPUProcDeviceCreateExternalTexture = c.CFUNCTYPE(c.POINTER[struct_WGPUExternalTextureImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUExternalTextureDescriptor])
WGPUProcDeviceCreatePipelineLayout = c.CFUNCTYPE(c.POINTER[struct_WGPUPipelineLayoutImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUPipelineLayoutDescriptor])
WGPUProcDeviceCreateQuerySet = c.CFUNCTYPE(c.POINTER[struct_WGPUQuerySetImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUQuerySetDescriptor])
WGPUProcDeviceCreateRenderBundleEncoder = c.CFUNCTYPE(c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderBundleEncoderDescriptor])
WGPUProcDeviceCreateRenderPipeline = c.CFUNCTYPE(c.POINTER[struct_WGPURenderPipelineImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderPipelineDescriptor])
WGPUProcDeviceCreateRenderPipelineAsync: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderPipelineDescriptor], c.CFUNCTYPE(None, enum_WGPUCreatePipelineAsyncStatus, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView, c.POINTER[None]), c.POINTER[None])
WGPUProcDeviceCreateRenderPipelineAsync2 = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderPipelineDescriptor], struct_WGPUCreateRenderPipelineAsyncCallbackInfo2)
WGPUProcDeviceCreateRenderPipelineAsyncF = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPURenderPipelineDescriptor], struct_WGPUCreateRenderPipelineAsyncCallbackInfo)
WGPUProcDeviceCreateSampler = c.CFUNCTYPE(c.POINTER[struct_WGPUSamplerImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSamplerDescriptor])
WGPUProcDeviceCreateShaderModule = c.CFUNCTYPE(c.POINTER[struct_WGPUShaderModuleImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUShaderModuleDescriptor])
WGPUProcDeviceCreateTexture = c.CFUNCTYPE(c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUTextureDescriptor])
WGPUProcDeviceDestroy = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl])
WGPUProcDeviceForceLoss: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], enum_WGPUDeviceLostReason, struct_WGPUStringView)
WGPUProcDeviceGetAHardwareBufferProperties: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[None], c.POINTER[struct_WGPUAHardwareBufferProperties])
WGPUProcDeviceGetAdapter = c.CFUNCTYPE(c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUDeviceImpl])
WGPUProcDeviceGetAdapterInfo: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUAdapterInfo])
WGPUProcDeviceGetFeatures = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSupportedFeatures])
WGPUProcDeviceGetLimits: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSupportedLimits])
WGPUProcDeviceGetLostFuture = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUDeviceImpl])
WGPUProcDeviceGetQueue = c.CFUNCTYPE(c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUDeviceImpl])
WGPUProcDeviceHasFeature: TypeAlias = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUDeviceImpl], enum_WGPUFeatureName)
WGPUProcDeviceImportSharedBufferMemory = c.CFUNCTYPE(c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSharedBufferMemoryDescriptor])
WGPUProcDeviceImportSharedFence = c.CFUNCTYPE(c.POINTER[struct_WGPUSharedFenceImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSharedFenceDescriptor])
WGPUProcDeviceImportSharedTextureMemory = c.CFUNCTYPE(c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUSharedTextureMemoryDescriptor])
WGPUProcDeviceInjectError: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], enum_WGPUErrorType, struct_WGPUStringView)
WGPUProcDevicePopErrorScope: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], c.CFUNCTYPE(None, enum_WGPUErrorType, struct_WGPUStringView, c.POINTER[None]), c.POINTER[None])
WGPUProcDevicePopErrorScope2 = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUPopErrorScopeCallbackInfo2)
WGPUProcDevicePopErrorScopeF = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUPopErrorScopeCallbackInfo)
WGPUProcDevicePushErrorScope: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], enum_WGPUErrorFilter)
WGPUProcDeviceSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], struct_WGPUStringView)
WGPUProcDeviceSetLoggingCallback: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], c.CFUNCTYPE(None, enum_WGPULoggingType, struct_WGPUStringView, c.POINTER[None]), c.POINTER[None])
WGPUProcDeviceTick = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl])
WGPUProcDeviceValidateTextureDescriptor = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl], c.POINTER[struct_WGPUTextureDescriptor])
WGPUProcDeviceAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl])
WGPUProcDeviceRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUDeviceImpl])
WGPUProcExternalTextureDestroy = c.CFUNCTYPE(None, c.POINTER[struct_WGPUExternalTextureImpl])
WGPUProcExternalTextureExpire = c.CFUNCTYPE(None, c.POINTER[struct_WGPUExternalTextureImpl])
WGPUProcExternalTextureRefresh = c.CFUNCTYPE(None, c.POINTER[struct_WGPUExternalTextureImpl])
WGPUProcExternalTextureSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUExternalTextureImpl], struct_WGPUStringView)
WGPUProcExternalTextureAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUExternalTextureImpl])
WGPUProcExternalTextureRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUExternalTextureImpl])
WGPUProcInstanceCreateSurface = c.CFUNCTYPE(c.POINTER[struct_WGPUSurfaceImpl], c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPUSurfaceDescriptor])
WGPUProcInstanceEnumerateWGSLLanguageFeatures: TypeAlias = c.CFUNCTYPE(Annotated[int, ctypes.c_uint64], c.POINTER[struct_WGPUInstanceImpl], c.POINTER[enum_WGPUWGSLFeatureName])
WGPUProcInstanceHasWGSLLanguageFeature: TypeAlias = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUInstanceImpl], enum_WGPUWGSLFeatureName)
WGPUProcInstanceProcessEvents = c.CFUNCTYPE(None, c.POINTER[struct_WGPUInstanceImpl])
WGPUProcInstanceRequestAdapter: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPURequestAdapterOptions], c.CFUNCTYPE(None, enum_WGPURequestAdapterStatus, c.POINTER[struct_WGPUAdapterImpl], struct_WGPUStringView, c.POINTER[None]), c.POINTER[None])
WGPUProcInstanceRequestAdapter2 = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPURequestAdapterOptions], struct_WGPURequestAdapterCallbackInfo2)
WGPUProcInstanceRequestAdapterF = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUInstanceImpl], c.POINTER[struct_WGPURequestAdapterOptions], struct_WGPURequestAdapterCallbackInfo)
WGPUProcInstanceWaitAny: TypeAlias = c.CFUNCTYPE(enum_WGPUWaitStatus, c.POINTER[struct_WGPUInstanceImpl], Annotated[int, ctypes.c_uint64], c.POINTER[struct_WGPUFutureWaitInfo], Annotated[int, ctypes.c_uint64])
WGPUProcInstanceAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUInstanceImpl])
WGPUProcInstanceRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUInstanceImpl])
WGPUProcPipelineLayoutSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUPipelineLayoutImpl], struct_WGPUStringView)
WGPUProcPipelineLayoutAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUPipelineLayoutImpl])
WGPUProcPipelineLayoutRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUPipelineLayoutImpl])
WGPUProcQuerySetDestroy = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQuerySetImpl])
WGPUProcQuerySetGetCount = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUQuerySetImpl])
WGPUProcQuerySetGetType: TypeAlias = c.CFUNCTYPE(enum_WGPUQueryType, c.POINTER[struct_WGPUQuerySetImpl])
WGPUProcQuerySetSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQuerySetImpl], struct_WGPUStringView)
WGPUProcQuerySetAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQuerySetImpl])
WGPUProcQuerySetRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQuerySetImpl])
WGPUProcQueueCopyExternalTextureForBrowser = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUImageCopyExternalTexture], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUExtent3D], c.POINTER[struct_WGPUCopyTextureForBrowserOptions])
WGPUProcQueueCopyTextureForBrowser = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[struct_WGPUExtent3D], c.POINTER[struct_WGPUCopyTextureForBrowserOptions])
WGPUProcQueueOnSubmittedWorkDone: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl], c.CFUNCTYPE(None, enum_WGPUQueueWorkDoneStatus, c.POINTER[None]), c.POINTER[None])
WGPUProcQueueOnSubmittedWorkDone2 = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUQueueImpl], struct_WGPUQueueWorkDoneCallbackInfo2)
WGPUProcQueueOnSubmittedWorkDoneF = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUQueueImpl], struct_WGPUQueueWorkDoneCallbackInfo)
WGPUProcQueueSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl], struct_WGPUStringView)
WGPUProcQueueSubmit = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl], Annotated[int, ctypes.c_uint64], c.POINTER[c.POINTER[struct_WGPUCommandBufferImpl]])
WGPUProcQueueWriteBuffer = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], c.POINTER[None], Annotated[int, ctypes.c_uint64])
WGPUProcQueueWriteTexture = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl], c.POINTER[struct_WGPUImageCopyTexture], c.POINTER[None], Annotated[int, ctypes.c_uint64], c.POINTER[struct_WGPUTextureDataLayout], c.POINTER[struct_WGPUExtent3D])
WGPUProcQueueAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl])
WGPUProcQueueRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUQueueImpl])
WGPUProcRenderBundleSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleImpl], struct_WGPUStringView)
WGPUProcRenderBundleAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleImpl])
WGPUProcRenderBundleRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleImpl])
WGPUProcRenderBundleEncoderDraw = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderBundleEncoderDrawIndexed = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_int32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderBundleEncoderDrawIndexedIndirect = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64])
WGPUProcRenderBundleEncoderDrawIndirect = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64])
WGPUProcRenderBundleEncoderFinish = c.CFUNCTYPE(c.POINTER[struct_WGPURenderBundleImpl], c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPURenderBundleDescriptor])
WGPUProcRenderBundleEncoderInsertDebugMarker = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], struct_WGPUStringView)
WGPUProcRenderBundleEncoderPopDebugGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl])
WGPUProcRenderBundleEncoderPushDebugGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], struct_WGPUStringView)
WGPUProcRenderBundleEncoderSetBindGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUBindGroupImpl], Annotated[int, ctypes.c_uint64], c.POINTER[Annotated[int, ctypes.c_uint32]])
WGPUProcRenderBundleEncoderSetIndexBuffer: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPUBufferImpl], enum_WGPUIndexFormat, Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcRenderBundleEncoderSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], struct_WGPUStringView)
WGPUProcRenderBundleEncoderSetPipeline = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], c.POINTER[struct_WGPURenderPipelineImpl])
WGPUProcRenderBundleEncoderSetVertexBuffer = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl], Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcRenderBundleEncoderAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl])
WGPUProcRenderBundleEncoderRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderBundleEncoderImpl])
WGPUProcRenderPassEncoderBeginOcclusionQuery = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderDraw = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderDrawIndexed = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_int32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderDrawIndexedIndirect = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderDrawIndirect = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderEnd = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl])
WGPUProcRenderPassEncoderEndOcclusionQuery = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl])
WGPUProcRenderPassEncoderExecuteBundles = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[int, ctypes.c_uint64], c.POINTER[c.POINTER[struct_WGPURenderBundleImpl]])
WGPUProcRenderPassEncoderInsertDebugMarker = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], struct_WGPUStringView)
WGPUProcRenderPassEncoderMultiDrawIndexedIndirect = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderMultiDrawIndirect = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderPixelLocalStorageBarrier = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl])
WGPUProcRenderPassEncoderPopDebugGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl])
WGPUProcRenderPassEncoderPushDebugGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], struct_WGPUStringView)
WGPUProcRenderPassEncoderSetBindGroup = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUBindGroupImpl], Annotated[int, ctypes.c_uint64], c.POINTER[Annotated[int, ctypes.c_uint32]])
WGPUProcRenderPassEncoderSetBlendConstant = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUColor])
WGPUProcRenderPassEncoderSetIndexBuffer: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUBufferImpl], enum_WGPUIndexFormat, Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], struct_WGPUStringView)
WGPUProcRenderPassEncoderSetPipeline = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPURenderPipelineImpl])
WGPUProcRenderPassEncoderSetScissorRect = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderSetStencilReference = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderSetVertexBuffer = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUBufferImpl], Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_uint64])
WGPUProcRenderPassEncoderSetViewport = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float])
WGPUProcRenderPassEncoderWriteTimestamp = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl], c.POINTER[struct_WGPUQuerySetImpl], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPassEncoderAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl])
WGPUProcRenderPassEncoderRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPassEncoderImpl])
WGPUProcRenderPipelineGetBindGroupLayout = c.CFUNCTYPE(c.POINTER[struct_WGPUBindGroupLayoutImpl], c.POINTER[struct_WGPURenderPipelineImpl], Annotated[int, ctypes.c_uint32])
WGPUProcRenderPipelineSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPipelineImpl], struct_WGPUStringView)
WGPUProcRenderPipelineAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPipelineImpl])
WGPUProcRenderPipelineRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPURenderPipelineImpl])
WGPUProcSamplerSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSamplerImpl], struct_WGPUStringView)
WGPUProcSamplerAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSamplerImpl])
WGPUProcSamplerRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSamplerImpl])
WGPUProcShaderModuleGetCompilationInfo: TypeAlias = c.CFUNCTYPE(None, c.POINTER[struct_WGPUShaderModuleImpl], c.CFUNCTYPE(None, enum_WGPUCompilationInfoRequestStatus, c.POINTER[struct_WGPUCompilationInfo], c.POINTER[None]), c.POINTER[None])
WGPUProcShaderModuleGetCompilationInfo2 = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUShaderModuleImpl], struct_WGPUCompilationInfoCallbackInfo2)
WGPUProcShaderModuleGetCompilationInfoF = c.CFUNCTYPE(struct_WGPUFuture, c.POINTER[struct_WGPUShaderModuleImpl], struct_WGPUCompilationInfoCallbackInfo)
WGPUProcShaderModuleSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUShaderModuleImpl], struct_WGPUStringView)
WGPUProcShaderModuleAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUShaderModuleImpl])
WGPUProcShaderModuleRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUShaderModuleImpl])
WGPUProcSharedBufferMemoryBeginAccess: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUBufferImpl], c.POINTER[struct_WGPUSharedBufferMemoryBeginAccessDescriptor])
WGPUProcSharedBufferMemoryCreateBuffer = c.CFUNCTYPE(c.POINTER[struct_WGPUBufferImpl], c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUBufferDescriptor])
WGPUProcSharedBufferMemoryEndAccess: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUBufferImpl], c.POINTER[struct_WGPUSharedBufferMemoryEndAccessState])
WGPUProcSharedBufferMemoryGetProperties: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUSharedBufferMemoryImpl], c.POINTER[struct_WGPUSharedBufferMemoryProperties])
WGPUProcSharedBufferMemoryIsDeviceLost = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUSharedBufferMemoryImpl])
WGPUProcSharedBufferMemorySetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedBufferMemoryImpl], struct_WGPUStringView)
WGPUProcSharedBufferMemoryAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedBufferMemoryImpl])
WGPUProcSharedBufferMemoryRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedBufferMemoryImpl])
WGPUProcSharedFenceExportInfo = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedFenceImpl], c.POINTER[struct_WGPUSharedFenceExportInfo])
WGPUProcSharedFenceAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedFenceImpl])
WGPUProcSharedFenceRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedFenceImpl])
WGPUProcSharedTextureMemoryBeginAccess: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUSharedTextureMemoryBeginAccessDescriptor])
WGPUProcSharedTextureMemoryCreateTexture = c.CFUNCTYPE(c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUTextureDescriptor])
WGPUProcSharedTextureMemoryEndAccess: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUSharedTextureMemoryEndAccessState])
WGPUProcSharedTextureMemoryGetProperties: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUSharedTextureMemoryImpl], c.POINTER[struct_WGPUSharedTextureMemoryProperties])
WGPUProcSharedTextureMemoryIsDeviceLost = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUSharedTextureMemoryImpl])
WGPUProcSharedTextureMemorySetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedTextureMemoryImpl], struct_WGPUStringView)
WGPUProcSharedTextureMemoryAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedTextureMemoryImpl])
WGPUProcSharedTextureMemoryRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSharedTextureMemoryImpl])
WGPUProcSurfaceConfigure = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSurfaceImpl], c.POINTER[struct_WGPUSurfaceConfiguration])
WGPUProcSurfaceGetCapabilities: TypeAlias = c.CFUNCTYPE(enum_WGPUStatus, c.POINTER[struct_WGPUSurfaceImpl], c.POINTER[struct_WGPUAdapterImpl], c.POINTER[struct_WGPUSurfaceCapabilities])
WGPUProcSurfaceGetCurrentTexture = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSurfaceImpl], c.POINTER[struct_WGPUSurfaceTexture])
WGPUProcSurfacePresent = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSurfaceImpl])
WGPUProcSurfaceSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSurfaceImpl], struct_WGPUStringView)
WGPUProcSurfaceUnconfigure = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSurfaceImpl])
WGPUProcSurfaceAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSurfaceImpl])
WGPUProcSurfaceRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUSurfaceImpl])
WGPUProcTextureCreateErrorView = c.CFUNCTYPE(c.POINTER[struct_WGPUTextureViewImpl], c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUTextureViewDescriptor])
WGPUProcTextureCreateView = c.CFUNCTYPE(c.POINTER[struct_WGPUTextureViewImpl], c.POINTER[struct_WGPUTextureImpl], c.POINTER[struct_WGPUTextureViewDescriptor])
WGPUProcTextureDestroy = c.CFUNCTYPE(None, c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureGetDepthOrArrayLayers = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureGetDimension: TypeAlias = c.CFUNCTYPE(enum_WGPUTextureDimension, c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureGetFormat: TypeAlias = c.CFUNCTYPE(enum_WGPUTextureFormat, c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureGetHeight = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureGetMipLevelCount = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureGetSampleCount = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureGetUsage = c.CFUNCTYPE(Annotated[int, ctypes.c_uint64], c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureGetWidth = c.CFUNCTYPE(Annotated[int, ctypes.c_uint32], c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUTextureImpl], struct_WGPUStringView)
WGPUProcTextureAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUTextureImpl])
WGPUProcTextureViewSetLabel = c.CFUNCTYPE(None, c.POINTER[struct_WGPUTextureViewImpl], struct_WGPUStringView)
WGPUProcTextureViewAddRef = c.CFUNCTYPE(None, c.POINTER[struct_WGPUTextureViewImpl])
WGPUProcTextureViewRelease = c.CFUNCTYPE(None, c.POINTER[struct_WGPUTextureViewImpl])
@dll.bind
def wgpuAdapterInfoFreeMembers(value:WGPUAdapterInfo) -> None: ...
@dll.bind
def wgpuAdapterPropertiesMemoryHeapsFreeMembers(value:WGPUAdapterPropertiesMemoryHeaps) -> None: ...
@dll.bind
def wgpuCreateInstance(descriptor:c.POINTER[WGPUInstanceDescriptor]) -> WGPUInstance: ...
@dll.bind
def wgpuDrmFormatCapabilitiesFreeMembers(value:WGPUDrmFormatCapabilities) -> None: ...
@dll.bind
def wgpuGetInstanceFeatures(features:c.POINTER[WGPUInstanceFeatures]) -> WGPUStatus: ...
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
def wgpuAdapterCreateDevice(adapter:WGPUAdapter, descriptor:c.POINTER[WGPUDeviceDescriptor]) -> WGPUDevice: ...
@dll.bind
def wgpuAdapterGetFeatures(adapter:WGPUAdapter, features:c.POINTER[WGPUSupportedFeatures]) -> None: ...
@dll.bind
def wgpuAdapterGetFormatCapabilities(adapter:WGPUAdapter, format:WGPUTextureFormat, capabilities:c.POINTER[WGPUFormatCapabilities]) -> WGPUStatus: ...
@dll.bind
def wgpuAdapterGetInfo(adapter:WGPUAdapter, info:c.POINTER[WGPUAdapterInfo]) -> WGPUStatus: ...
@dll.bind
def wgpuAdapterGetInstance(adapter:WGPUAdapter) -> WGPUInstance: ...
@dll.bind
def wgpuAdapterGetLimits(adapter:WGPUAdapter, limits:c.POINTER[WGPUSupportedLimits]) -> WGPUStatus: ...
@dll.bind
def wgpuAdapterHasFeature(adapter:WGPUAdapter, feature:WGPUFeatureName) -> WGPUBool: ...
@dll.bind
def wgpuAdapterRequestDevice(adapter:WGPUAdapter, descriptor:c.POINTER[WGPUDeviceDescriptor], callback:WGPURequestDeviceCallback, userdata:c.POINTER[None]) -> None: ...
@dll.bind
def wgpuAdapterRequestDevice2(adapter:WGPUAdapter, options:c.POINTER[WGPUDeviceDescriptor], callbackInfo:WGPURequestDeviceCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuAdapterRequestDeviceF(adapter:WGPUAdapter, options:c.POINTER[WGPUDeviceDescriptor], callbackInfo:WGPURequestDeviceCallbackInfo) -> WGPUFuture: ...
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
def wgpuBufferGetConstMappedRange(buffer:WGPUBuffer, offset:size_t, size:size_t) -> c.POINTER[None]: ...
@dll.bind
def wgpuBufferGetMapState(buffer:WGPUBuffer) -> WGPUBufferMapState: ...
@dll.bind
def wgpuBufferGetMappedRange(buffer:WGPUBuffer, offset:size_t, size:size_t) -> c.POINTER[None]: ...
@dll.bind
def wgpuBufferGetSize(buffer:WGPUBuffer) -> uint64_t: ...
@dll.bind
def wgpuBufferGetUsage(buffer:WGPUBuffer) -> WGPUBufferUsage: ...
@dll.bind
def wgpuBufferMapAsync(buffer:WGPUBuffer, mode:WGPUMapMode, offset:size_t, size:size_t, callback:WGPUBufferMapCallback, userdata:c.POINTER[None]) -> None: ...
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
def wgpuCommandEncoderBeginComputePass(commandEncoder:WGPUCommandEncoder, descriptor:c.POINTER[WGPUComputePassDescriptor]) -> WGPUComputePassEncoder: ...
@dll.bind
def wgpuCommandEncoderBeginRenderPass(commandEncoder:WGPUCommandEncoder, descriptor:c.POINTER[WGPURenderPassDescriptor]) -> WGPURenderPassEncoder: ...
@dll.bind
def wgpuCommandEncoderClearBuffer(commandEncoder:WGPUCommandEncoder, buffer:WGPUBuffer, offset:uint64_t, size:uint64_t) -> None: ...
@dll.bind
def wgpuCommandEncoderCopyBufferToBuffer(commandEncoder:WGPUCommandEncoder, source:WGPUBuffer, sourceOffset:uint64_t, destination:WGPUBuffer, destinationOffset:uint64_t, size:uint64_t) -> None: ...
@dll.bind
def wgpuCommandEncoderCopyBufferToTexture(commandEncoder:WGPUCommandEncoder, source:c.POINTER[WGPUImageCopyBuffer], destination:c.POINTER[WGPUImageCopyTexture], copySize:c.POINTER[WGPUExtent3D]) -> None: ...
@dll.bind
def wgpuCommandEncoderCopyTextureToBuffer(commandEncoder:WGPUCommandEncoder, source:c.POINTER[WGPUImageCopyTexture], destination:c.POINTER[WGPUImageCopyBuffer], copySize:c.POINTER[WGPUExtent3D]) -> None: ...
@dll.bind
def wgpuCommandEncoderCopyTextureToTexture(commandEncoder:WGPUCommandEncoder, source:c.POINTER[WGPUImageCopyTexture], destination:c.POINTER[WGPUImageCopyTexture], copySize:c.POINTER[WGPUExtent3D]) -> None: ...
@dll.bind
def wgpuCommandEncoderFinish(commandEncoder:WGPUCommandEncoder, descriptor:c.POINTER[WGPUCommandBufferDescriptor]) -> WGPUCommandBuffer: ...
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
uint8_t = Annotated[int, ctypes.c_ubyte]
@dll.bind
def wgpuCommandEncoderWriteBuffer(commandEncoder:WGPUCommandEncoder, buffer:WGPUBuffer, bufferOffset:uint64_t, data:c.POINTER[uint8_t], size:uint64_t) -> None: ...
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
def wgpuComputePassEncoderSetBindGroup(computePassEncoder:WGPUComputePassEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:c.POINTER[uint32_t]) -> None: ...
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
def wgpuDeviceCreateBindGroup(device:WGPUDevice, descriptor:c.POINTER[WGPUBindGroupDescriptor]) -> WGPUBindGroup: ...
@dll.bind
def wgpuDeviceCreateBindGroupLayout(device:WGPUDevice, descriptor:c.POINTER[WGPUBindGroupLayoutDescriptor]) -> WGPUBindGroupLayout: ...
@dll.bind
def wgpuDeviceCreateBuffer(device:WGPUDevice, descriptor:c.POINTER[WGPUBufferDescriptor]) -> WGPUBuffer: ...
@dll.bind
def wgpuDeviceCreateCommandEncoder(device:WGPUDevice, descriptor:c.POINTER[WGPUCommandEncoderDescriptor]) -> WGPUCommandEncoder: ...
@dll.bind
def wgpuDeviceCreateComputePipeline(device:WGPUDevice, descriptor:c.POINTER[WGPUComputePipelineDescriptor]) -> WGPUComputePipeline: ...
@dll.bind
def wgpuDeviceCreateComputePipelineAsync(device:WGPUDevice, descriptor:c.POINTER[WGPUComputePipelineDescriptor], callback:WGPUCreateComputePipelineAsyncCallback, userdata:c.POINTER[None]) -> None: ...
@dll.bind
def wgpuDeviceCreateComputePipelineAsync2(device:WGPUDevice, descriptor:c.POINTER[WGPUComputePipelineDescriptor], callbackInfo:WGPUCreateComputePipelineAsyncCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceCreateComputePipelineAsyncF(device:WGPUDevice, descriptor:c.POINTER[WGPUComputePipelineDescriptor], callbackInfo:WGPUCreateComputePipelineAsyncCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceCreateErrorBuffer(device:WGPUDevice, descriptor:c.POINTER[WGPUBufferDescriptor]) -> WGPUBuffer: ...
@dll.bind
def wgpuDeviceCreateErrorExternalTexture(device:WGPUDevice) -> WGPUExternalTexture: ...
@dll.bind
def wgpuDeviceCreateErrorShaderModule(device:WGPUDevice, descriptor:c.POINTER[WGPUShaderModuleDescriptor], errorMessage:WGPUStringView) -> WGPUShaderModule: ...
@dll.bind
def wgpuDeviceCreateErrorTexture(device:WGPUDevice, descriptor:c.POINTER[WGPUTextureDescriptor]) -> WGPUTexture: ...
@dll.bind
def wgpuDeviceCreateExternalTexture(device:WGPUDevice, externalTextureDescriptor:c.POINTER[WGPUExternalTextureDescriptor]) -> WGPUExternalTexture: ...
@dll.bind
def wgpuDeviceCreatePipelineLayout(device:WGPUDevice, descriptor:c.POINTER[WGPUPipelineLayoutDescriptor]) -> WGPUPipelineLayout: ...
@dll.bind
def wgpuDeviceCreateQuerySet(device:WGPUDevice, descriptor:c.POINTER[WGPUQuerySetDescriptor]) -> WGPUQuerySet: ...
@dll.bind
def wgpuDeviceCreateRenderBundleEncoder(device:WGPUDevice, descriptor:c.POINTER[WGPURenderBundleEncoderDescriptor]) -> WGPURenderBundleEncoder: ...
@dll.bind
def wgpuDeviceCreateRenderPipeline(device:WGPUDevice, descriptor:c.POINTER[WGPURenderPipelineDescriptor]) -> WGPURenderPipeline: ...
@dll.bind
def wgpuDeviceCreateRenderPipelineAsync(device:WGPUDevice, descriptor:c.POINTER[WGPURenderPipelineDescriptor], callback:WGPUCreateRenderPipelineAsyncCallback, userdata:c.POINTER[None]) -> None: ...
@dll.bind
def wgpuDeviceCreateRenderPipelineAsync2(device:WGPUDevice, descriptor:c.POINTER[WGPURenderPipelineDescriptor], callbackInfo:WGPUCreateRenderPipelineAsyncCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceCreateRenderPipelineAsyncF(device:WGPUDevice, descriptor:c.POINTER[WGPURenderPipelineDescriptor], callbackInfo:WGPUCreateRenderPipelineAsyncCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceCreateSampler(device:WGPUDevice, descriptor:c.POINTER[WGPUSamplerDescriptor]) -> WGPUSampler: ...
@dll.bind
def wgpuDeviceCreateShaderModule(device:WGPUDevice, descriptor:c.POINTER[WGPUShaderModuleDescriptor]) -> WGPUShaderModule: ...
@dll.bind
def wgpuDeviceCreateTexture(device:WGPUDevice, descriptor:c.POINTER[WGPUTextureDescriptor]) -> WGPUTexture: ...
@dll.bind
def wgpuDeviceDestroy(device:WGPUDevice) -> None: ...
@dll.bind
def wgpuDeviceForceLoss(device:WGPUDevice, type:WGPUDeviceLostReason, message:WGPUStringView) -> None: ...
@dll.bind
def wgpuDeviceGetAHardwareBufferProperties(device:WGPUDevice, handle:c.POINTER[None], properties:c.POINTER[WGPUAHardwareBufferProperties]) -> WGPUStatus: ...
@dll.bind
def wgpuDeviceGetAdapter(device:WGPUDevice) -> WGPUAdapter: ...
@dll.bind
def wgpuDeviceGetAdapterInfo(device:WGPUDevice, adapterInfo:c.POINTER[WGPUAdapterInfo]) -> WGPUStatus: ...
@dll.bind
def wgpuDeviceGetFeatures(device:WGPUDevice, features:c.POINTER[WGPUSupportedFeatures]) -> None: ...
@dll.bind
def wgpuDeviceGetLimits(device:WGPUDevice, limits:c.POINTER[WGPUSupportedLimits]) -> WGPUStatus: ...
@dll.bind
def wgpuDeviceGetLostFuture(device:WGPUDevice) -> WGPUFuture: ...
@dll.bind
def wgpuDeviceGetQueue(device:WGPUDevice) -> WGPUQueue: ...
@dll.bind
def wgpuDeviceHasFeature(device:WGPUDevice, feature:WGPUFeatureName) -> WGPUBool: ...
@dll.bind
def wgpuDeviceImportSharedBufferMemory(device:WGPUDevice, descriptor:c.POINTER[WGPUSharedBufferMemoryDescriptor]) -> WGPUSharedBufferMemory: ...
@dll.bind
def wgpuDeviceImportSharedFence(device:WGPUDevice, descriptor:c.POINTER[WGPUSharedFenceDescriptor]) -> WGPUSharedFence: ...
@dll.bind
def wgpuDeviceImportSharedTextureMemory(device:WGPUDevice, descriptor:c.POINTER[WGPUSharedTextureMemoryDescriptor]) -> WGPUSharedTextureMemory: ...
@dll.bind
def wgpuDeviceInjectError(device:WGPUDevice, type:WGPUErrorType, message:WGPUStringView) -> None: ...
@dll.bind
def wgpuDevicePopErrorScope(device:WGPUDevice, oldCallback:WGPUErrorCallback, userdata:c.POINTER[None]) -> None: ...
@dll.bind
def wgpuDevicePopErrorScope2(device:WGPUDevice, callbackInfo:WGPUPopErrorScopeCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuDevicePopErrorScopeF(device:WGPUDevice, callbackInfo:WGPUPopErrorScopeCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuDevicePushErrorScope(device:WGPUDevice, filter:WGPUErrorFilter) -> None: ...
@dll.bind
def wgpuDeviceSetLabel(device:WGPUDevice, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuDeviceSetLoggingCallback(device:WGPUDevice, callback:WGPULoggingCallback, userdata:c.POINTER[None]) -> None: ...
@dll.bind
def wgpuDeviceTick(device:WGPUDevice) -> None: ...
@dll.bind
def wgpuDeviceValidateTextureDescriptor(device:WGPUDevice, descriptor:c.POINTER[WGPUTextureDescriptor]) -> None: ...
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
def wgpuInstanceCreateSurface(instance:WGPUInstance, descriptor:c.POINTER[WGPUSurfaceDescriptor]) -> WGPUSurface: ...
@dll.bind
def wgpuInstanceEnumerateWGSLLanguageFeatures(instance:WGPUInstance, features:c.POINTER[WGPUWGSLFeatureName]) -> size_t: ...
@dll.bind
def wgpuInstanceHasWGSLLanguageFeature(instance:WGPUInstance, feature:WGPUWGSLFeatureName) -> WGPUBool: ...
@dll.bind
def wgpuInstanceProcessEvents(instance:WGPUInstance) -> None: ...
@dll.bind
def wgpuInstanceRequestAdapter(instance:WGPUInstance, options:c.POINTER[WGPURequestAdapterOptions], callback:WGPURequestAdapterCallback, userdata:c.POINTER[None]) -> None: ...
@dll.bind
def wgpuInstanceRequestAdapter2(instance:WGPUInstance, options:c.POINTER[WGPURequestAdapterOptions], callbackInfo:WGPURequestAdapterCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuInstanceRequestAdapterF(instance:WGPUInstance, options:c.POINTER[WGPURequestAdapterOptions], callbackInfo:WGPURequestAdapterCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuInstanceWaitAny(instance:WGPUInstance, futureCount:size_t, futures:c.POINTER[WGPUFutureWaitInfo], timeoutNS:uint64_t) -> WGPUWaitStatus: ...
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
def wgpuQueueCopyExternalTextureForBrowser(queue:WGPUQueue, source:c.POINTER[WGPUImageCopyExternalTexture], destination:c.POINTER[WGPUImageCopyTexture], copySize:c.POINTER[WGPUExtent3D], options:c.POINTER[WGPUCopyTextureForBrowserOptions]) -> None: ...
@dll.bind
def wgpuQueueCopyTextureForBrowser(queue:WGPUQueue, source:c.POINTER[WGPUImageCopyTexture], destination:c.POINTER[WGPUImageCopyTexture], copySize:c.POINTER[WGPUExtent3D], options:c.POINTER[WGPUCopyTextureForBrowserOptions]) -> None: ...
@dll.bind
def wgpuQueueOnSubmittedWorkDone(queue:WGPUQueue, callback:WGPUQueueWorkDoneCallback, userdata:c.POINTER[None]) -> None: ...
@dll.bind
def wgpuQueueOnSubmittedWorkDone2(queue:WGPUQueue, callbackInfo:WGPUQueueWorkDoneCallbackInfo2) -> WGPUFuture: ...
@dll.bind
def wgpuQueueOnSubmittedWorkDoneF(queue:WGPUQueue, callbackInfo:WGPUQueueWorkDoneCallbackInfo) -> WGPUFuture: ...
@dll.bind
def wgpuQueueSetLabel(queue:WGPUQueue, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuQueueSubmit(queue:WGPUQueue, commandCount:size_t, commands:c.POINTER[WGPUCommandBuffer]) -> None: ...
@dll.bind
def wgpuQueueWriteBuffer(queue:WGPUQueue, buffer:WGPUBuffer, bufferOffset:uint64_t, data:c.POINTER[None], size:size_t) -> None: ...
@dll.bind
def wgpuQueueWriteTexture(queue:WGPUQueue, destination:c.POINTER[WGPUImageCopyTexture], data:c.POINTER[None], dataSize:size_t, dataLayout:c.POINTER[WGPUTextureDataLayout], writeSize:c.POINTER[WGPUExtent3D]) -> None: ...
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
def wgpuRenderBundleEncoderFinish(renderBundleEncoder:WGPURenderBundleEncoder, descriptor:c.POINTER[WGPURenderBundleDescriptor]) -> WGPURenderBundle: ...
@dll.bind
def wgpuRenderBundleEncoderInsertDebugMarker(renderBundleEncoder:WGPURenderBundleEncoder, markerLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderPopDebugGroup(renderBundleEncoder:WGPURenderBundleEncoder) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderPushDebugGroup(renderBundleEncoder:WGPURenderBundleEncoder, groupLabel:WGPUStringView) -> None: ...
@dll.bind
def wgpuRenderBundleEncoderSetBindGroup(renderBundleEncoder:WGPURenderBundleEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:c.POINTER[uint32_t]) -> None: ...
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
def wgpuRenderPassEncoderExecuteBundles(renderPassEncoder:WGPURenderPassEncoder, bundleCount:size_t, bundles:c.POINTER[WGPURenderBundle]) -> None: ...
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
def wgpuRenderPassEncoderSetBindGroup(renderPassEncoder:WGPURenderPassEncoder, groupIndex:uint32_t, group:WGPUBindGroup, dynamicOffsetCount:size_t, dynamicOffsets:c.POINTER[uint32_t]) -> None: ...
@dll.bind
def wgpuRenderPassEncoderSetBlendConstant(renderPassEncoder:WGPURenderPassEncoder, color:c.POINTER[WGPUColor]) -> None: ...
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
def wgpuShaderModuleGetCompilationInfo(shaderModule:WGPUShaderModule, callback:WGPUCompilationInfoCallback, userdata:c.POINTER[None]) -> None: ...
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
def wgpuSharedBufferMemoryBeginAccess(sharedBufferMemory:WGPUSharedBufferMemory, buffer:WGPUBuffer, descriptor:c.POINTER[WGPUSharedBufferMemoryBeginAccessDescriptor]) -> WGPUStatus: ...
@dll.bind
def wgpuSharedBufferMemoryCreateBuffer(sharedBufferMemory:WGPUSharedBufferMemory, descriptor:c.POINTER[WGPUBufferDescriptor]) -> WGPUBuffer: ...
@dll.bind
def wgpuSharedBufferMemoryEndAccess(sharedBufferMemory:WGPUSharedBufferMemory, buffer:WGPUBuffer, descriptor:c.POINTER[WGPUSharedBufferMemoryEndAccessState]) -> WGPUStatus: ...
@dll.bind
def wgpuSharedBufferMemoryGetProperties(sharedBufferMemory:WGPUSharedBufferMemory, properties:c.POINTER[WGPUSharedBufferMemoryProperties]) -> WGPUStatus: ...
@dll.bind
def wgpuSharedBufferMemoryIsDeviceLost(sharedBufferMemory:WGPUSharedBufferMemory) -> WGPUBool: ...
@dll.bind
def wgpuSharedBufferMemorySetLabel(sharedBufferMemory:WGPUSharedBufferMemory, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuSharedBufferMemoryAddRef(sharedBufferMemory:WGPUSharedBufferMemory) -> None: ...
@dll.bind
def wgpuSharedBufferMemoryRelease(sharedBufferMemory:WGPUSharedBufferMemory) -> None: ...
@dll.bind
def wgpuSharedFenceExportInfo(sharedFence:WGPUSharedFence, info:c.POINTER[WGPUSharedFenceExportInfo]) -> None: ...
@dll.bind
def wgpuSharedFenceAddRef(sharedFence:WGPUSharedFence) -> None: ...
@dll.bind
def wgpuSharedFenceRelease(sharedFence:WGPUSharedFence) -> None: ...
@dll.bind
def wgpuSharedTextureMemoryBeginAccess(sharedTextureMemory:WGPUSharedTextureMemory, texture:WGPUTexture, descriptor:c.POINTER[WGPUSharedTextureMemoryBeginAccessDescriptor]) -> WGPUStatus: ...
@dll.bind
def wgpuSharedTextureMemoryCreateTexture(sharedTextureMemory:WGPUSharedTextureMemory, descriptor:c.POINTER[WGPUTextureDescriptor]) -> WGPUTexture: ...
@dll.bind
def wgpuSharedTextureMemoryEndAccess(sharedTextureMemory:WGPUSharedTextureMemory, texture:WGPUTexture, descriptor:c.POINTER[WGPUSharedTextureMemoryEndAccessState]) -> WGPUStatus: ...
@dll.bind
def wgpuSharedTextureMemoryGetProperties(sharedTextureMemory:WGPUSharedTextureMemory, properties:c.POINTER[WGPUSharedTextureMemoryProperties]) -> WGPUStatus: ...
@dll.bind
def wgpuSharedTextureMemoryIsDeviceLost(sharedTextureMemory:WGPUSharedTextureMemory) -> WGPUBool: ...
@dll.bind
def wgpuSharedTextureMemorySetLabel(sharedTextureMemory:WGPUSharedTextureMemory, label:WGPUStringView) -> None: ...
@dll.bind
def wgpuSharedTextureMemoryAddRef(sharedTextureMemory:WGPUSharedTextureMemory) -> None: ...
@dll.bind
def wgpuSharedTextureMemoryRelease(sharedTextureMemory:WGPUSharedTextureMemory) -> None: ...
@dll.bind
def wgpuSurfaceConfigure(surface:WGPUSurface, config:c.POINTER[WGPUSurfaceConfiguration]) -> None: ...
@dll.bind
def wgpuSurfaceGetCapabilities(surface:WGPUSurface, adapter:WGPUAdapter, capabilities:c.POINTER[WGPUSurfaceCapabilities]) -> WGPUStatus: ...
@dll.bind
def wgpuSurfaceGetCurrentTexture(surface:WGPUSurface, surfaceTexture:c.POINTER[WGPUSurfaceTexture]) -> None: ...
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
def wgpuTextureCreateErrorView(texture:WGPUTexture, descriptor:c.POINTER[WGPUTextureViewDescriptor]) -> WGPUTextureView: ...
@dll.bind
def wgpuTextureCreateView(texture:WGPUTexture, descriptor:c.POINTER[WGPUTextureViewDescriptor]) -> WGPUTextureView: ...
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
c.init_records()
WGPUBufferUsage_None = 0x0000000000000000 # type: ignore
WGPUBufferUsage_MapRead = 0x0000000000000001 # type: ignore
WGPUBufferUsage_MapWrite = 0x0000000000000002 # type: ignore
WGPUBufferUsage_CopySrc = 0x0000000000000004 # type: ignore
WGPUBufferUsage_CopyDst = 0x0000000000000008 # type: ignore
WGPUBufferUsage_Index = 0x0000000000000010 # type: ignore
WGPUBufferUsage_Vertex = 0x0000000000000020 # type: ignore
WGPUBufferUsage_Uniform = 0x0000000000000040 # type: ignore
WGPUBufferUsage_Storage = 0x0000000000000080 # type: ignore
WGPUBufferUsage_Indirect = 0x0000000000000100 # type: ignore
WGPUBufferUsage_QueryResolve = 0x0000000000000200 # type: ignore
WGPUColorWriteMask_None = 0x0000000000000000 # type: ignore
WGPUColorWriteMask_Red = 0x0000000000000001 # type: ignore
WGPUColorWriteMask_Green = 0x0000000000000002 # type: ignore
WGPUColorWriteMask_Blue = 0x0000000000000004 # type: ignore
WGPUColorWriteMask_Alpha = 0x0000000000000008 # type: ignore
WGPUColorWriteMask_All = 0x000000000000000F # type: ignore
WGPUHeapProperty_DeviceLocal = 0x0000000000000001 # type: ignore
WGPUHeapProperty_HostVisible = 0x0000000000000002 # type: ignore
WGPUHeapProperty_HostCoherent = 0x0000000000000004 # type: ignore
WGPUHeapProperty_HostUncached = 0x0000000000000008 # type: ignore
WGPUHeapProperty_HostCached = 0x0000000000000010 # type: ignore
WGPUMapMode_None = 0x0000000000000000 # type: ignore
WGPUMapMode_Read = 0x0000000000000001 # type: ignore
WGPUMapMode_Write = 0x0000000000000002 # type: ignore
WGPUShaderStage_None = 0x0000000000000000 # type: ignore
WGPUShaderStage_Vertex = 0x0000000000000001 # type: ignore
WGPUShaderStage_Fragment = 0x0000000000000002 # type: ignore
WGPUShaderStage_Compute = 0x0000000000000004 # type: ignore
WGPUTextureUsage_None = 0x0000000000000000 # type: ignore
WGPUTextureUsage_CopySrc = 0x0000000000000001 # type: ignore
WGPUTextureUsage_CopyDst = 0x0000000000000002 # type: ignore
WGPUTextureUsage_TextureBinding = 0x0000000000000004 # type: ignore
WGPUTextureUsage_StorageBinding = 0x0000000000000008 # type: ignore
WGPUTextureUsage_RenderAttachment = 0x0000000000000010 # type: ignore
WGPUTextureUsage_TransientAttachment = 0x0000000000000020 # type: ignore
WGPUTextureUsage_StorageAttachment = 0x0000000000000040 # type: ignore