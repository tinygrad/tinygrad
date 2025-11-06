# mypy: ignore-errors
import ctypes
from tinygrad.helpers import Struct, CEnum, _IO, _IOW, _IOR, _IOWR, unwrap
from ctypes.util import find_library
from tinygrad.runtime.support import objc
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
try: (MTLCreateSystemDefaultDevice:=dll.MTLCreateSystemDefaultDevice).restype, MTLCreateSystemDefaultDevice.argtypes = objc.id_, []
except AttributeError: pass

MTLCreateSystemDefaultDevice = objc.returns_retained(MTLCreateSystemDefaultDevice)
# __attribute__((visibility("default"))) extern NSArray<id<MTLDevice>> * _Nonnull MTLCopyAllDevices(void) __attribute__((ns_returns_retained)) __attribute__((availability(macos, introduced=10.11))) __attribute__((availability(maccatalyst, introduced=13.0))) __attribute__((availability(ios, unavailable)))
try: (MTLCopyAllDevices:=dll.MTLCopyAllDevices).restype, MTLCopyAllDevices.argtypes = objc.id_, []
except AttributeError: pass

MTLCopyAllDevices = objc.returns_retained(MTLCopyAllDevices)
class NSObject(objc.Spec): pass
instancetype = objc.id_
class struct__NSZone(Struct): pass
BOOL = ctypes.c_int
class Protocol(objc.Spec): pass
IMP = ctypes.CFUNCTYPE(None, )
class NSInvocation(NSObject): pass
class NSMethodSignature(NSObject): pass
NSUInteger = ctypes.c_ulong
NSMethodSignature._methods_ = [
  ('getArgumentTypeAtIndex:', ctypes.POINTER(ctypes.c_char), [NSUInteger]),
  ('isOneway', BOOL, []),
  ('numberOfArguments', NSUInteger, []),
  ('frameLength', NSUInteger, []),
  ('methodReturnType', ctypes.POINTER(ctypes.c_char), []),
  ('methodReturnLength', NSUInteger, []),
]
NSMethodSignature._classmethods_ = [
  ('signatureWithObjCTypes:', NSMethodSignature, [ctypes.POINTER(ctypes.c_char)]),
]
NSInvocation._methods_ = [
  ('retainArguments', None, []),
  ('getReturnValue:', None, [ctypes.c_void_p]),
  ('setReturnValue:', None, [ctypes.c_void_p]),
  ('getArgument:atIndex:', None, [ctypes.c_void_p, NSInteger]),
  ('setArgument:atIndex:', None, [ctypes.c_void_p, NSInteger]),
  ('invoke', None, []),
  ('invokeWithTarget:', None, [objc.id_]),
  ('invokeUsingIMP:', None, [IMP]),
  ('methodSignature', NSMethodSignature, []),
  ('argumentsRetained', BOOL, []),
  ('target', objc.id_, []),
  ('setTarget:', None, [objc.id_]),
  ('selector', objc.id_, []),
  ('setSelector:', None, [objc.id_]),
]
NSInvocation._classmethods_ = [
  ('invocationWithMethodSignature:', NSInvocation, [NSMethodSignature]),
]
class NSString(NSObject): pass
unichar = ctypes.c_ushort
class NSCoder(NSObject): pass
class NSData(NSObject): pass
NSData._methods_ = [
  ('length', NSUInteger, []),
  ('bytes', ctypes.c_void_p, []),
]
NSCoder._methods_ = [
  ('encodeValueOfObjCType:at:', None, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]),
  ('encodeDataObject:', None, [NSData]),
  ('decodeDataObject', NSData, []),
  ('decodeValueOfObjCType:at:size:', None, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, NSUInteger]),
  ('versionForClassName:', NSInteger, [NSString]),
]
NSString._methods_ = [
  ('characterAtIndex:', unichar, [NSUInteger]),
  ('init', instancetype, []),
  ('initWithCoder:', instancetype, [NSCoder]),
  ('length', NSUInteger, []),
]
NSObject._methods_ = [
  ('init', instancetype, []),
  ('dealloc', None, []),
  ('finalize', None, []),
  ('copy', objc.id_, []),
  ('mutableCopy', objc.id_, []),
  ('methodForSelector:', IMP, [objc.id_]),
  ('doesNotRecognizeSelector:', None, [objc.id_]),
  ('forwardingTargetForSelector:', objc.id_, [objc.id_]),
  ('forwardInvocation:', None, [NSInvocation]),
  ('methodSignatureForSelector:', NSMethodSignature, [objc.id_]),
  ('allowsWeakReference', BOOL, []),
  ('retainWeakReference', BOOL, []),
]
NSObject._classmethods_ = [
  ('load', None, []),
  ('initialize', None, []),
  ('new', instancetype, []),
  ('allocWithZone:', instancetype, [ctypes.POINTER(struct__NSZone)]),
  ('alloc', instancetype, []),
  ('copyWithZone:', objc.id_, [ctypes.POINTER(struct__NSZone)]),
  ('mutableCopyWithZone:', objc.id_, [ctypes.POINTER(struct__NSZone)]),
  ('instancesRespondToSelector:', BOOL, [objc.id_]),
  ('conformsToProtocol:', BOOL, [Protocol]),
  ('instanceMethodForSelector:', IMP, [objc.id_]),
  ('instanceMethodSignatureForSelector:', NSMethodSignature, [objc.id_]),
  ('resolveClassMethod:', BOOL, [objc.id_]),
  ('resolveInstanceMethod:', BOOL, [objc.id_]),
  ('hash', NSUInteger, []),
  ('description', NSString, []),
  ('debugDescription', NSString, []),
]
class NSString(NSObject): pass
NSString._methods_ = [
  ('characterAtIndex:', unichar, [NSUInteger]),
  ('init', instancetype, []),
  ('initWithCoder:', instancetype, [NSCoder]),
  ('length', NSUInteger, []),
]
MTLDeviceNotificationName = NSString
try: MTLDeviceWasAddedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasAddedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceRemovalRequestedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceRemovalRequestedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceWasRemovedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasRemovedNotification')
except (ValueError,AttributeError): pass
# __attribute__((visibility("default"))) extern void MTLRemoveDeviceObserver(id<NSObject>  _Nonnull observer) __attribute__((availability(macos, introduced=10.13))) __attribute__((availability(ios, unavailable)))
try: (MTLRemoveDeviceObserver:=dll.MTLRemoveDeviceObserver).restype, MTLRemoveDeviceObserver.argtypes = None, [objc.id_]
except AttributeError: pass

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
class MTLRenderPipelineReflection(objc.Spec): pass
class MTLComputePipelineReflection(objc.Spec): pass
class MTLArgumentDescriptor(NSObject): pass
enum_MTLDataType = CEnum(NSUInteger)
MTLDataTypeNone = enum_MTLDataType.define('MTLDataTypeNone', 0)
MTLDataTypeStruct = enum_MTLDataType.define('MTLDataTypeStruct', 1)
MTLDataTypeArray = enum_MTLDataType.define('MTLDataTypeArray', 2)
MTLDataTypeFloat = enum_MTLDataType.define('MTLDataTypeFloat', 3)
MTLDataTypeFloat2 = enum_MTLDataType.define('MTLDataTypeFloat2', 4)
MTLDataTypeFloat3 = enum_MTLDataType.define('MTLDataTypeFloat3', 5)
MTLDataTypeFloat4 = enum_MTLDataType.define('MTLDataTypeFloat4', 6)
MTLDataTypeFloat2x2 = enum_MTLDataType.define('MTLDataTypeFloat2x2', 7)
MTLDataTypeFloat2x3 = enum_MTLDataType.define('MTLDataTypeFloat2x3', 8)
MTLDataTypeFloat2x4 = enum_MTLDataType.define('MTLDataTypeFloat2x4', 9)
MTLDataTypeFloat3x2 = enum_MTLDataType.define('MTLDataTypeFloat3x2', 10)
MTLDataTypeFloat3x3 = enum_MTLDataType.define('MTLDataTypeFloat3x3', 11)
MTLDataTypeFloat3x4 = enum_MTLDataType.define('MTLDataTypeFloat3x4', 12)
MTLDataTypeFloat4x2 = enum_MTLDataType.define('MTLDataTypeFloat4x2', 13)
MTLDataTypeFloat4x3 = enum_MTLDataType.define('MTLDataTypeFloat4x3', 14)
MTLDataTypeFloat4x4 = enum_MTLDataType.define('MTLDataTypeFloat4x4', 15)
MTLDataTypeHalf = enum_MTLDataType.define('MTLDataTypeHalf', 16)
MTLDataTypeHalf2 = enum_MTLDataType.define('MTLDataTypeHalf2', 17)
MTLDataTypeHalf3 = enum_MTLDataType.define('MTLDataTypeHalf3', 18)
MTLDataTypeHalf4 = enum_MTLDataType.define('MTLDataTypeHalf4', 19)
MTLDataTypeHalf2x2 = enum_MTLDataType.define('MTLDataTypeHalf2x2', 20)
MTLDataTypeHalf2x3 = enum_MTLDataType.define('MTLDataTypeHalf2x3', 21)
MTLDataTypeHalf2x4 = enum_MTLDataType.define('MTLDataTypeHalf2x4', 22)
MTLDataTypeHalf3x2 = enum_MTLDataType.define('MTLDataTypeHalf3x2', 23)
MTLDataTypeHalf3x3 = enum_MTLDataType.define('MTLDataTypeHalf3x3', 24)
MTLDataTypeHalf3x4 = enum_MTLDataType.define('MTLDataTypeHalf3x4', 25)
MTLDataTypeHalf4x2 = enum_MTLDataType.define('MTLDataTypeHalf4x2', 26)
MTLDataTypeHalf4x3 = enum_MTLDataType.define('MTLDataTypeHalf4x3', 27)
MTLDataTypeHalf4x4 = enum_MTLDataType.define('MTLDataTypeHalf4x4', 28)
MTLDataTypeInt = enum_MTLDataType.define('MTLDataTypeInt', 29)
MTLDataTypeInt2 = enum_MTLDataType.define('MTLDataTypeInt2', 30)
MTLDataTypeInt3 = enum_MTLDataType.define('MTLDataTypeInt3', 31)
MTLDataTypeInt4 = enum_MTLDataType.define('MTLDataTypeInt4', 32)
MTLDataTypeUInt = enum_MTLDataType.define('MTLDataTypeUInt', 33)
MTLDataTypeUInt2 = enum_MTLDataType.define('MTLDataTypeUInt2', 34)
MTLDataTypeUInt3 = enum_MTLDataType.define('MTLDataTypeUInt3', 35)
MTLDataTypeUInt4 = enum_MTLDataType.define('MTLDataTypeUInt4', 36)
MTLDataTypeShort = enum_MTLDataType.define('MTLDataTypeShort', 37)
MTLDataTypeShort2 = enum_MTLDataType.define('MTLDataTypeShort2', 38)
MTLDataTypeShort3 = enum_MTLDataType.define('MTLDataTypeShort3', 39)
MTLDataTypeShort4 = enum_MTLDataType.define('MTLDataTypeShort4', 40)
MTLDataTypeUShort = enum_MTLDataType.define('MTLDataTypeUShort', 41)
MTLDataTypeUShort2 = enum_MTLDataType.define('MTLDataTypeUShort2', 42)
MTLDataTypeUShort3 = enum_MTLDataType.define('MTLDataTypeUShort3', 43)
MTLDataTypeUShort4 = enum_MTLDataType.define('MTLDataTypeUShort4', 44)
MTLDataTypeChar = enum_MTLDataType.define('MTLDataTypeChar', 45)
MTLDataTypeChar2 = enum_MTLDataType.define('MTLDataTypeChar2', 46)
MTLDataTypeChar3 = enum_MTLDataType.define('MTLDataTypeChar3', 47)
MTLDataTypeChar4 = enum_MTLDataType.define('MTLDataTypeChar4', 48)
MTLDataTypeUChar = enum_MTLDataType.define('MTLDataTypeUChar', 49)
MTLDataTypeUChar2 = enum_MTLDataType.define('MTLDataTypeUChar2', 50)
MTLDataTypeUChar3 = enum_MTLDataType.define('MTLDataTypeUChar3', 51)
MTLDataTypeUChar4 = enum_MTLDataType.define('MTLDataTypeUChar4', 52)
MTLDataTypeBool = enum_MTLDataType.define('MTLDataTypeBool', 53)
MTLDataTypeBool2 = enum_MTLDataType.define('MTLDataTypeBool2', 54)
MTLDataTypeBool3 = enum_MTLDataType.define('MTLDataTypeBool3', 55)
MTLDataTypeBool4 = enum_MTLDataType.define('MTLDataTypeBool4', 56)
MTLDataTypeTexture = enum_MTLDataType.define('MTLDataTypeTexture', 58)
MTLDataTypeSampler = enum_MTLDataType.define('MTLDataTypeSampler', 59)
MTLDataTypePointer = enum_MTLDataType.define('MTLDataTypePointer', 60)
MTLDataTypeR8Unorm = enum_MTLDataType.define('MTLDataTypeR8Unorm', 62)
MTLDataTypeR8Snorm = enum_MTLDataType.define('MTLDataTypeR8Snorm', 63)
MTLDataTypeR16Unorm = enum_MTLDataType.define('MTLDataTypeR16Unorm', 64)
MTLDataTypeR16Snorm = enum_MTLDataType.define('MTLDataTypeR16Snorm', 65)
MTLDataTypeRG8Unorm = enum_MTLDataType.define('MTLDataTypeRG8Unorm', 66)
MTLDataTypeRG8Snorm = enum_MTLDataType.define('MTLDataTypeRG8Snorm', 67)
MTLDataTypeRG16Unorm = enum_MTLDataType.define('MTLDataTypeRG16Unorm', 68)
MTLDataTypeRG16Snorm = enum_MTLDataType.define('MTLDataTypeRG16Snorm', 69)
MTLDataTypeRGBA8Unorm = enum_MTLDataType.define('MTLDataTypeRGBA8Unorm', 70)
MTLDataTypeRGBA8Unorm_sRGB = enum_MTLDataType.define('MTLDataTypeRGBA8Unorm_sRGB', 71)
MTLDataTypeRGBA8Snorm = enum_MTLDataType.define('MTLDataTypeRGBA8Snorm', 72)
MTLDataTypeRGBA16Unorm = enum_MTLDataType.define('MTLDataTypeRGBA16Unorm', 73)
MTLDataTypeRGBA16Snorm = enum_MTLDataType.define('MTLDataTypeRGBA16Snorm', 74)
MTLDataTypeRGB10A2Unorm = enum_MTLDataType.define('MTLDataTypeRGB10A2Unorm', 75)
MTLDataTypeRG11B10Float = enum_MTLDataType.define('MTLDataTypeRG11B10Float', 76)
MTLDataTypeRGB9E5Float = enum_MTLDataType.define('MTLDataTypeRGB9E5Float', 77)
MTLDataTypeRenderPipeline = enum_MTLDataType.define('MTLDataTypeRenderPipeline', 78)
MTLDataTypeComputePipeline = enum_MTLDataType.define('MTLDataTypeComputePipeline', 79)
MTLDataTypeIndirectCommandBuffer = enum_MTLDataType.define('MTLDataTypeIndirectCommandBuffer', 80)
MTLDataTypeLong = enum_MTLDataType.define('MTLDataTypeLong', 81)
MTLDataTypeLong2 = enum_MTLDataType.define('MTLDataTypeLong2', 82)
MTLDataTypeLong3 = enum_MTLDataType.define('MTLDataTypeLong3', 83)
MTLDataTypeLong4 = enum_MTLDataType.define('MTLDataTypeLong4', 84)
MTLDataTypeULong = enum_MTLDataType.define('MTLDataTypeULong', 85)
MTLDataTypeULong2 = enum_MTLDataType.define('MTLDataTypeULong2', 86)
MTLDataTypeULong3 = enum_MTLDataType.define('MTLDataTypeULong3', 87)
MTLDataTypeULong4 = enum_MTLDataType.define('MTLDataTypeULong4', 88)
MTLDataTypeVisibleFunctionTable = enum_MTLDataType.define('MTLDataTypeVisibleFunctionTable', 115)
MTLDataTypeIntersectionFunctionTable = enum_MTLDataType.define('MTLDataTypeIntersectionFunctionTable', 116)
MTLDataTypePrimitiveAccelerationStructure = enum_MTLDataType.define('MTLDataTypePrimitiveAccelerationStructure', 117)
MTLDataTypeInstanceAccelerationStructure = enum_MTLDataType.define('MTLDataTypeInstanceAccelerationStructure', 118)
MTLDataTypeBFloat = enum_MTLDataType.define('MTLDataTypeBFloat', 121)
MTLDataTypeBFloat2 = enum_MTLDataType.define('MTLDataTypeBFloat2', 122)
MTLDataTypeBFloat3 = enum_MTLDataType.define('MTLDataTypeBFloat3', 123)
MTLDataTypeBFloat4 = enum_MTLDataType.define('MTLDataTypeBFloat4', 124)

MTLDataType = enum_MTLDataType
enum_MTLBindingAccess = CEnum(NSUInteger)
MTLBindingAccessReadOnly = enum_MTLBindingAccess.define('MTLBindingAccessReadOnly', 0)
MTLBindingAccessReadWrite = enum_MTLBindingAccess.define('MTLBindingAccessReadWrite', 1)
MTLBindingAccessWriteOnly = enum_MTLBindingAccess.define('MTLBindingAccessWriteOnly', 2)
MTLArgumentAccessReadOnly = enum_MTLBindingAccess.define('MTLArgumentAccessReadOnly', 0)
MTLArgumentAccessReadWrite = enum_MTLBindingAccess.define('MTLArgumentAccessReadWrite', 1)
MTLArgumentAccessWriteOnly = enum_MTLBindingAccess.define('MTLArgumentAccessWriteOnly', 2)

MTLBindingAccess = enum_MTLBindingAccess
enum_MTLTextureType = CEnum(NSUInteger)
MTLTextureType1D = enum_MTLTextureType.define('MTLTextureType1D', 0)
MTLTextureType1DArray = enum_MTLTextureType.define('MTLTextureType1DArray', 1)
MTLTextureType2D = enum_MTLTextureType.define('MTLTextureType2D', 2)
MTLTextureType2DArray = enum_MTLTextureType.define('MTLTextureType2DArray', 3)
MTLTextureType2DMultisample = enum_MTLTextureType.define('MTLTextureType2DMultisample', 4)
MTLTextureTypeCube = enum_MTLTextureType.define('MTLTextureTypeCube', 5)
MTLTextureTypeCubeArray = enum_MTLTextureType.define('MTLTextureTypeCubeArray', 6)
MTLTextureType3D = enum_MTLTextureType.define('MTLTextureType3D', 7)
MTLTextureType2DMultisampleArray = enum_MTLTextureType.define('MTLTextureType2DMultisampleArray', 8)
MTLTextureTypeTextureBuffer = enum_MTLTextureType.define('MTLTextureTypeTextureBuffer', 9)

MTLTextureType = enum_MTLTextureType
MTLArgumentDescriptor._methods_ = [
  ('dataType', MTLDataType, []),
  ('setDataType:', None, [MTLDataType]),
  ('index', NSUInteger, []),
  ('setIndex:', None, [NSUInteger]),
  ('arrayLength', NSUInteger, []),
  ('setArrayLength:', None, [NSUInteger]),
  ('access', MTLBindingAccess, []),
  ('setAccess:', None, [MTLBindingAccess]),
  ('textureType', MTLTextureType, []),
  ('setTextureType:', None, [MTLTextureType]),
  ('constantBlockAlignment', NSUInteger, []),
  ('setConstantBlockAlignment:', None, [NSUInteger]),
]
MTLArgumentDescriptor._classmethods_ = [
  ('argumentDescriptor', MTLArgumentDescriptor, []),
]
class MTLArchitecture(NSObject): pass
MTLArchitecture._methods_ = [
  ('name', NSString, []),
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
class MTLIndirectCommandBufferDescriptor(NSObject): pass
MTLIndirectCommandBufferDescriptor._methods_ = [
  ('commandTypes', MTLIndirectCommandType, []),
  ('setCommandTypes:', None, [MTLIndirectCommandType]),
  ('inheritPipelineState', BOOL, []),
  ('setInheritPipelineState:', None, [BOOL]),
  ('inheritBuffers', BOOL, []),
  ('setInheritBuffers:', None, [BOOL]),
  ('maxVertexBufferBindCount', NSUInteger, []),
  ('setMaxVertexBufferBindCount:', None, [NSUInteger]),
  ('maxFragmentBufferBindCount', NSUInteger, []),
  ('setMaxFragmentBufferBindCount:', None, [NSUInteger]),
  ('maxKernelBufferBindCount', NSUInteger, []),
  ('setMaxKernelBufferBindCount:', None, [NSUInteger]),
  ('maxKernelThreadgroupMemoryBindCount', NSUInteger, []),
  ('setMaxKernelThreadgroupMemoryBindCount:', None, [NSUInteger]),
  ('supportRayTracing', BOOL, []),
  ('setSupportRayTracing:', None, [BOOL]),
  ('supportDynamicAttributeStride', BOOL, []),
  ('setSupportDynamicAttributeStride:', None, [BOOL]),
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
