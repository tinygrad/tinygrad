# mypy: ignore-errors
import ctypes
from tinygrad.helpers import Struct, CEnum, objc_id, objc_instance, _IO, _IOW, _IOR, _IOWR, unwrap
from ctypes.util import find_library
def dll():
  try: return ctypes.CDLL(unwrap(find_library('Metal')))
  except: pass
  return None
dll = dll()

NSInteger = ctypes.c_long
enum_MTLIOCompressionMethod = CEnum(NSInteger)
MTLIOCompressionMethodZlib = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodZlib', 0)
MTLIOCompressionMethodLZFSE = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZFSE', 1)
MTLIOCompressionMethodLZ4 = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZ4', 2)
MTLIOCompressionMethodLZMA = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZMA', 3)
MTLIOCompressionMethodLZBitmap = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZBitmap', 4)

MTLIOCompressionMethod = enum_MTLIOCompressionMethod
# __attribute__((visibility("default"))) extern id<MTLDevice>  _Nullable MTLCreateSystemDefaultDevice(void) __attribute__((ns_returns_retained)) __attribute__((availability(macos, introduced=10.11))) __attribute__((availability(ios, introduced=8.0)))
try: (MTLCreateSystemDefaultDevice:=dll.MTLCreateSystemDefaultDevice).restype, MTLCreateSystemDefaultDevice.argtypes = objc_instance, []
except AttributeError: pass

# __attribute__((visibility("default"))) extern NSArray<id<MTLDevice>> * _Nonnull MTLCopyAllDevices(void) __attribute__((ns_returns_retained)) __attribute__((availability(macos, introduced=10.11))) __attribute__((availability(maccatalyst, introduced=13.0))) __attribute__((availability(ios, unavailable)))
try: (MTLCopyAllDevices:=dll.MTLCopyAllDevices).restype, MTLCopyAllDevices.argtypes = objc_instance, []
except AttributeError: pass

MTLDeviceNotificationName = objc_id
try: MTLDeviceWasAddedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasAddedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceRemovalRequestedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceRemovalRequestedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceWasRemovedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasRemovedNotification')
except (ValueError,AttributeError): pass
# __attribute__((visibility("default"))) extern void MTLRemoveDeviceObserver(id<NSObject>  _Nonnull observer) __attribute__((availability(macos, introduced=10.13))) __attribute__((availability(ios, unavailable)))
try: (MTLRemoveDeviceObserver:=dll.MTLRemoveDeviceObserver).restype, MTLRemoveDeviceObserver.argtypes = None, [objc_id]
except AttributeError: pass

NSUInteger = ctypes.c_ulong
enum_MTLFeatureSet = CEnum(NSUInteger)
MTLFeatureSet_iOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v1', 0)
MTLFeatureSet_iOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v1', 1)
MTLFeatureSet_iOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v2', 2)
MTLFeatureSet_iOS_GPUFamily2_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v2', 3)
MTLFeatureSet_iOS_GPUFamily3_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v1', 4)
MTLFeatureSet_iOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v3', 5)
MTLFeatureSet_iOS_GPUFamily2_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v3', 6)
MTLFeatureSet_iOS_GPUFamily3_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v2', 7)
MTLFeatureSet_iOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v4', 8)
MTLFeatureSet_iOS_GPUFamily2_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v4', 9)
MTLFeatureSet_iOS_GPUFamily3_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v3', 10)
MTLFeatureSet_iOS_GPUFamily4_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily4_v1', 11)
MTLFeatureSet_iOS_GPUFamily1_v5 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v5', 12)
MTLFeatureSet_iOS_GPUFamily2_v5 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v5', 13)
MTLFeatureSet_iOS_GPUFamily3_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v4', 14)
MTLFeatureSet_iOS_GPUFamily4_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily4_v2', 15)
MTLFeatureSet_iOS_GPUFamily5_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily5_v1', 16)
MTLFeatureSet_macOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v1', 10000)
MTLFeatureSet_OSX_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_GPUFamily1_v1', 10000)
MTLFeatureSet_macOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v2', 10001)
MTLFeatureSet_OSX_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_GPUFamily1_v2', 10001)
MTLFeatureSet_macOS_ReadWriteTextureTier2 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_ReadWriteTextureTier2', 10002)
MTLFeatureSet_OSX_ReadWriteTextureTier2 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_ReadWriteTextureTier2', 10002)
MTLFeatureSet_macOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v3', 10003)
MTLFeatureSet_macOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v4', 10004)
MTLFeatureSet_macOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily2_v1', 10005)
MTLFeatureSet_tvOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v1', 30000)
MTLFeatureSet_TVOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_TVOS_GPUFamily1_v1', 30000)
MTLFeatureSet_tvOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v2', 30001)
MTLFeatureSet_tvOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v3', 30002)
MTLFeatureSet_tvOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily2_v1', 30003)
MTLFeatureSet_tvOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v4', 30004)
MTLFeatureSet_tvOS_GPUFamily2_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily2_v2', 30005)

MTLFeatureSet = enum_MTLFeatureSet
enum_MTLGPUFamily = CEnum(NSInteger)
MTLGPUFamilyApple1 = enum_MTLGPUFamily.define('MTLGPUFamilyApple1', 1001)
MTLGPUFamilyApple2 = enum_MTLGPUFamily.define('MTLGPUFamilyApple2', 1002)
MTLGPUFamilyApple3 = enum_MTLGPUFamily.define('MTLGPUFamilyApple3', 1003)
MTLGPUFamilyApple4 = enum_MTLGPUFamily.define('MTLGPUFamilyApple4', 1004)
MTLGPUFamilyApple5 = enum_MTLGPUFamily.define('MTLGPUFamilyApple5', 1005)
MTLGPUFamilyApple6 = enum_MTLGPUFamily.define('MTLGPUFamilyApple6', 1006)
MTLGPUFamilyApple7 = enum_MTLGPUFamily.define('MTLGPUFamilyApple7', 1007)
MTLGPUFamilyApple8 = enum_MTLGPUFamily.define('MTLGPUFamilyApple8', 1008)
MTLGPUFamilyApple9 = enum_MTLGPUFamily.define('MTLGPUFamilyApple9', 1009)
MTLGPUFamilyMac1 = enum_MTLGPUFamily.define('MTLGPUFamilyMac1', 2001)
MTLGPUFamilyMac2 = enum_MTLGPUFamily.define('MTLGPUFamilyMac2', 2002)
MTLGPUFamilyCommon1 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon1', 3001)
MTLGPUFamilyCommon2 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon2', 3002)
MTLGPUFamilyCommon3 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon3', 3003)
MTLGPUFamilyMacCatalyst1 = enum_MTLGPUFamily.define('MTLGPUFamilyMacCatalyst1', 4001)
MTLGPUFamilyMacCatalyst2 = enum_MTLGPUFamily.define('MTLGPUFamilyMacCatalyst2', 4002)
MTLGPUFamilyMetal3 = enum_MTLGPUFamily.define('MTLGPUFamilyMetal3', 5001)

MTLGPUFamily = enum_MTLGPUFamily
enum_MTLDeviceLocation = CEnum(NSUInteger)
MTLDeviceLocationBuiltIn = enum_MTLDeviceLocation.define('MTLDeviceLocationBuiltIn', 0)
MTLDeviceLocationSlot = enum_MTLDeviceLocation.define('MTLDeviceLocationSlot', 1)
MTLDeviceLocationExternal = enum_MTLDeviceLocation.define('MTLDeviceLocationExternal', 2)
MTLDeviceLocationUnspecified = enum_MTLDeviceLocation.define('MTLDeviceLocationUnspecified', -1)

MTLDeviceLocation = enum_MTLDeviceLocation
enum_MTLPipelineOption = CEnum(NSUInteger)
MTLPipelineOptionNone = enum_MTLPipelineOption.define('MTLPipelineOptionNone', 0)
MTLPipelineOptionArgumentInfo = enum_MTLPipelineOption.define('MTLPipelineOptionArgumentInfo', 1)
MTLPipelineOptionBufferTypeInfo = enum_MTLPipelineOption.define('MTLPipelineOptionBufferTypeInfo', 2)
MTLPipelineOptionFailOnBinaryArchiveMiss = enum_MTLPipelineOption.define('MTLPipelineOptionFailOnBinaryArchiveMiss', 4)

MTLPipelineOption = enum_MTLPipelineOption
enum_MTLReadWriteTextureTier = CEnum(NSUInteger)
MTLReadWriteTextureTierNone = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTierNone', 0)
MTLReadWriteTextureTier1 = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTier1', 1)
MTLReadWriteTextureTier2 = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTier2', 2)

MTLReadWriteTextureTier = enum_MTLReadWriteTextureTier
enum_MTLArgumentBuffersTier = CEnum(NSUInteger)
MTLArgumentBuffersTier1 = enum_MTLArgumentBuffersTier.define('MTLArgumentBuffersTier1', 0)
MTLArgumentBuffersTier2 = enum_MTLArgumentBuffersTier.define('MTLArgumentBuffersTier2', 1)

MTLArgumentBuffersTier = enum_MTLArgumentBuffersTier
enum_MTLSparseTextureRegionAlignmentMode = CEnum(NSUInteger)
MTLSparseTextureRegionAlignmentModeOutward = enum_MTLSparseTextureRegionAlignmentMode.define('MTLSparseTextureRegionAlignmentModeOutward', 0)
MTLSparseTextureRegionAlignmentModeInward = enum_MTLSparseTextureRegionAlignmentMode.define('MTLSparseTextureRegionAlignmentModeInward', 1)

MTLSparseTextureRegionAlignmentMode = enum_MTLSparseTextureRegionAlignmentMode
enum_MTLSparsePageSize = CEnum(NSInteger)
MTLSparsePageSize16 = enum_MTLSparsePageSize.define('MTLSparsePageSize16', 101)
MTLSparsePageSize64 = enum_MTLSparsePageSize.define('MTLSparsePageSize64', 102)
MTLSparsePageSize256 = enum_MTLSparsePageSize.define('MTLSparsePageSize256', 103)

MTLSparsePageSize = enum_MTLSparsePageSize
class MTLAccelerationStructureSizes(Struct): pass
MTLAccelerationStructureSizes._fields_ = [
  ('accelerationStructureSize', NSUInteger),
  ('buildScratchBufferSize', NSUInteger),
  ('refitScratchBufferSize', NSUInteger),
]
enum_MTLCounterSamplingPoint = CEnum(NSUInteger)
MTLCounterSamplingPointAtStageBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtStageBoundary', 0)
MTLCounterSamplingPointAtDrawBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtDrawBoundary', 1)
MTLCounterSamplingPointAtDispatchBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtDispatchBoundary', 2)
MTLCounterSamplingPointAtTileDispatchBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtTileDispatchBoundary', 3)
MTLCounterSamplingPointAtBlitBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtBlitBoundary', 4)

MTLCounterSamplingPoint = enum_MTLCounterSamplingPoint
class MTLSizeAndAlign(Struct): pass
MTLSizeAndAlign._fields_ = [
  ('size', NSUInteger),
  ('align', NSUInteger),
]
MTLTimestamp = ctypes.c_ulonglong
enum_MTLPurgeableState = CEnum(NSUInteger)
MTLPurgeableStateKeepCurrent = enum_MTLPurgeableState.define('MTLPurgeableStateKeepCurrent', 1)
MTLPurgeableStateNonVolatile = enum_MTLPurgeableState.define('MTLPurgeableStateNonVolatile', 2)
MTLPurgeableStateVolatile = enum_MTLPurgeableState.define('MTLPurgeableStateVolatile', 3)
MTLPurgeableStateEmpty = enum_MTLPurgeableState.define('MTLPurgeableStateEmpty', 4)

MTLPurgeableState = enum_MTLPurgeableState
enum_MTLCPUCacheMode = CEnum(NSUInteger)
MTLCPUCacheModeDefaultCache = enum_MTLCPUCacheMode.define('MTLCPUCacheModeDefaultCache', 0)
MTLCPUCacheModeWriteCombined = enum_MTLCPUCacheMode.define('MTLCPUCacheModeWriteCombined', 1)

MTLCPUCacheMode = enum_MTLCPUCacheMode
enum_MTLStorageMode = CEnum(NSUInteger)
MTLStorageModeShared = enum_MTLStorageMode.define('MTLStorageModeShared', 0)
MTLStorageModeManaged = enum_MTLStorageMode.define('MTLStorageModeManaged', 1)
MTLStorageModePrivate = enum_MTLStorageMode.define('MTLStorageModePrivate', 2)
MTLStorageModeMemoryless = enum_MTLStorageMode.define('MTLStorageModeMemoryless', 3)

MTLStorageMode = enum_MTLStorageMode
enum_MTLHazardTrackingMode = CEnum(NSUInteger)
MTLHazardTrackingModeDefault = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeDefault', 0)
MTLHazardTrackingModeUntracked = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeUntracked', 1)
MTLHazardTrackingModeTracked = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeTracked', 2)

MTLHazardTrackingMode = enum_MTLHazardTrackingMode
enum_MTLResourceOptions = CEnum(NSUInteger)
MTLResourceCPUCacheModeDefaultCache = enum_MTLResourceOptions.define('MTLResourceCPUCacheModeDefaultCache', 0)
MTLResourceCPUCacheModeWriteCombined = enum_MTLResourceOptions.define('MTLResourceCPUCacheModeWriteCombined', 1)
MTLResourceStorageModeShared = enum_MTLResourceOptions.define('MTLResourceStorageModeShared', 0)
MTLResourceStorageModeManaged = enum_MTLResourceOptions.define('MTLResourceStorageModeManaged', 16)
MTLResourceStorageModePrivate = enum_MTLResourceOptions.define('MTLResourceStorageModePrivate', 32)
MTLResourceStorageModeMemoryless = enum_MTLResourceOptions.define('MTLResourceStorageModeMemoryless', 48)
MTLResourceHazardTrackingModeDefault = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeDefault', 0)
MTLResourceHazardTrackingModeUntracked = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeUntracked', 256)
MTLResourceHazardTrackingModeTracked = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeTracked', 512)
MTLResourceOptionCPUCacheModeDefault = enum_MTLResourceOptions.define('MTLResourceOptionCPUCacheModeDefault', 0)
MTLResourceOptionCPUCacheModeWriteCombined = enum_MTLResourceOptions.define('MTLResourceOptionCPUCacheModeWriteCombined', 1)

MTLResourceOptions = enum_MTLResourceOptions
enum_MTLIndirectCommandType = CEnum(NSUInteger)
MTLIndirectCommandTypeDraw = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDraw', 1)
MTLIndirectCommandTypeDrawIndexed = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawIndexed', 2)
MTLIndirectCommandTypeDrawPatches = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawPatches', 4)
MTLIndirectCommandTypeDrawIndexedPatches = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawIndexedPatches', 8)
MTLIndirectCommandTypeConcurrentDispatch = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeConcurrentDispatch', 32)
MTLIndirectCommandTypeConcurrentDispatchThreads = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeConcurrentDispatchThreads', 64)

MTLIndirectCommandType = enum_MTLIndirectCommandType
class MTLIndirectCommandBufferExecutionRange(Struct): pass
uint32_t = ctypes.c_uint
MTLIndirectCommandBufferExecutionRange._fields_ = [
  ('location', uint32_t),
  ('length', uint32_t),
]
enum_MTLResourceUsage = CEnum(NSUInteger)
MTLResourceUsageRead = enum_MTLResourceUsage.define('MTLResourceUsageRead', 1)
MTLResourceUsageWrite = enum_MTLResourceUsage.define('MTLResourceUsageWrite', 2)
MTLResourceUsageSample = enum_MTLResourceUsage.define('MTLResourceUsageSample', 4)

MTLResourceUsage = enum_MTLResourceUsage
enum_MTLBarrierScope = CEnum(NSUInteger)
MTLBarrierScopeBuffers = enum_MTLBarrierScope.define('MTLBarrierScopeBuffers', 1)
MTLBarrierScopeTextures = enum_MTLBarrierScope.define('MTLBarrierScopeTextures', 2)
MTLBarrierScopeRenderTargets = enum_MTLBarrierScope.define('MTLBarrierScopeRenderTargets', 4)

MTLBarrierScope = enum_MTLBarrierScope
MTLResourceCPUCacheModeShift = 0
MTLResourceCPUCacheModeMask = (0xf << MTLResourceCPUCacheModeShift)
MTLResourceStorageModeShift = 4
MTLResourceStorageModeMask = (0xf << MTLResourceStorageModeShift)
MTLResourceHazardTrackingModeShift = 8
MTLResourceHazardTrackingModeMask = (0x3 << MTLResourceHazardTrackingModeShift)
