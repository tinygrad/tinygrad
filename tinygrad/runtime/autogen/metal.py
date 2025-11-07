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
class _anondynamic0(objc.Spec): pass
class _anondynamic1(objc.Spec): pass
NSUInteger = ctypes.c_ulong
class MTLSizeAndAlign(Struct): pass
MTLSizeAndAlign._fields_ = [
  ('size', NSUInteger),
  ('align', NSUInteger),
]
class NSObject(objc.Spec): pass
IMP = ctypes.CFUNCTYPE(None, )
class NSInvocation(NSObject): pass
class NSMethodSignature(NSObject): pass
BOOL = ctypes.c_int
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
class struct__NSZone(Struct): pass
class Protocol(objc.Spec): pass
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
  ('init', NSString, []),
  ('initWithCoder:', NSString, [NSCoder]),
  ('length', NSUInteger, []),
]
NSObject._methods_ = [
  ('init', NSObject, []),
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
  ('new', NSObject, []),
  ('allocWithZone:', NSObject, [ctypes.POINTER(struct__NSZone)]),
  ('alloc', NSObject, []),
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
class MTLTextureDescriptor(NSObject): pass
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
enum_MTLPixelFormat = CEnum(NSUInteger)
MTLPixelFormatInvalid = enum_MTLPixelFormat.define('MTLPixelFormatInvalid', 0)
MTLPixelFormatA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatA8Unorm', 1)
MTLPixelFormatR8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatR8Unorm', 10)
MTLPixelFormatR8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatR8Unorm_sRGB', 11)
MTLPixelFormatR8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatR8Snorm', 12)
MTLPixelFormatR8Uint = enum_MTLPixelFormat.define('MTLPixelFormatR8Uint', 13)
MTLPixelFormatR8Sint = enum_MTLPixelFormat.define('MTLPixelFormatR8Sint', 14)
MTLPixelFormatR16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatR16Unorm', 20)
MTLPixelFormatR16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatR16Snorm', 22)
MTLPixelFormatR16Uint = enum_MTLPixelFormat.define('MTLPixelFormatR16Uint', 23)
MTLPixelFormatR16Sint = enum_MTLPixelFormat.define('MTLPixelFormatR16Sint', 24)
MTLPixelFormatR16Float = enum_MTLPixelFormat.define('MTLPixelFormatR16Float', 25)
MTLPixelFormatRG8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRG8Unorm', 30)
MTLPixelFormatRG8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatRG8Unorm_sRGB', 31)
MTLPixelFormatRG8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRG8Snorm', 32)
MTLPixelFormatRG8Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG8Uint', 33)
MTLPixelFormatRG8Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG8Sint', 34)
MTLPixelFormatB5G6R5Unorm = enum_MTLPixelFormat.define('MTLPixelFormatB5G6R5Unorm', 40)
MTLPixelFormatA1BGR5Unorm = enum_MTLPixelFormat.define('MTLPixelFormatA1BGR5Unorm', 41)
MTLPixelFormatABGR4Unorm = enum_MTLPixelFormat.define('MTLPixelFormatABGR4Unorm', 42)
MTLPixelFormatBGR5A1Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGR5A1Unorm', 43)
MTLPixelFormatR32Uint = enum_MTLPixelFormat.define('MTLPixelFormatR32Uint', 53)
MTLPixelFormatR32Sint = enum_MTLPixelFormat.define('MTLPixelFormatR32Sint', 54)
MTLPixelFormatR32Float = enum_MTLPixelFormat.define('MTLPixelFormatR32Float', 55)
MTLPixelFormatRG16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRG16Unorm', 60)
MTLPixelFormatRG16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRG16Snorm', 62)
MTLPixelFormatRG16Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG16Uint', 63)
MTLPixelFormatRG16Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG16Sint', 64)
MTLPixelFormatRG16Float = enum_MTLPixelFormat.define('MTLPixelFormatRG16Float', 65)
MTLPixelFormatRGBA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Unorm', 70)
MTLPixelFormatRGBA8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Unorm_sRGB', 71)
MTLPixelFormatRGBA8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Snorm', 72)
MTLPixelFormatRGBA8Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Uint', 73)
MTLPixelFormatRGBA8Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Sint', 74)
MTLPixelFormatBGRA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGRA8Unorm', 80)
MTLPixelFormatBGRA8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGRA8Unorm_sRGB', 81)
MTLPixelFormatRGB10A2Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGB10A2Unorm', 90)
MTLPixelFormatRGB10A2Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGB10A2Uint', 91)
MTLPixelFormatRG11B10Float = enum_MTLPixelFormat.define('MTLPixelFormatRG11B10Float', 92)
MTLPixelFormatRGB9E5Float = enum_MTLPixelFormat.define('MTLPixelFormatRGB9E5Float', 93)
MTLPixelFormatBGR10A2Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGR10A2Unorm', 94)
MTLPixelFormatBGR10_XR = enum_MTLPixelFormat.define('MTLPixelFormatBGR10_XR', 554)
MTLPixelFormatBGR10_XR_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGR10_XR_sRGB', 555)
MTLPixelFormatRG32Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG32Uint', 103)
MTLPixelFormatRG32Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG32Sint', 104)
MTLPixelFormatRG32Float = enum_MTLPixelFormat.define('MTLPixelFormatRG32Float', 105)
MTLPixelFormatRGBA16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Unorm', 110)
MTLPixelFormatRGBA16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Snorm', 112)
MTLPixelFormatRGBA16Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Uint', 113)
MTLPixelFormatRGBA16Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Sint', 114)
MTLPixelFormatRGBA16Float = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Float', 115)
MTLPixelFormatBGRA10_XR = enum_MTLPixelFormat.define('MTLPixelFormatBGRA10_XR', 552)
MTLPixelFormatBGRA10_XR_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGRA10_XR_sRGB', 553)
MTLPixelFormatRGBA32Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Uint', 123)
MTLPixelFormatRGBA32Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Sint', 124)
MTLPixelFormatRGBA32Float = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Float', 125)
MTLPixelFormatBC1_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC1_RGBA', 130)
MTLPixelFormatBC1_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC1_RGBA_sRGB', 131)
MTLPixelFormatBC2_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC2_RGBA', 132)
MTLPixelFormatBC2_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC2_RGBA_sRGB', 133)
MTLPixelFormatBC3_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC3_RGBA', 134)
MTLPixelFormatBC3_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC3_RGBA_sRGB', 135)
MTLPixelFormatBC4_RUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC4_RUnorm', 140)
MTLPixelFormatBC4_RSnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC4_RSnorm', 141)
MTLPixelFormatBC5_RGUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC5_RGUnorm', 142)
MTLPixelFormatBC5_RGSnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC5_RGSnorm', 143)
MTLPixelFormatBC6H_RGBFloat = enum_MTLPixelFormat.define('MTLPixelFormatBC6H_RGBFloat', 150)
MTLPixelFormatBC6H_RGBUfloat = enum_MTLPixelFormat.define('MTLPixelFormatBC6H_RGBUfloat', 151)
MTLPixelFormatBC7_RGBAUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC7_RGBAUnorm', 152)
MTLPixelFormatBC7_RGBAUnorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC7_RGBAUnorm_sRGB', 153)
MTLPixelFormatPVRTC_RGB_2BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_2BPP', 160)
MTLPixelFormatPVRTC_RGB_2BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_2BPP_sRGB', 161)
MTLPixelFormatPVRTC_RGB_4BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_4BPP', 162)
MTLPixelFormatPVRTC_RGB_4BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_4BPP_sRGB', 163)
MTLPixelFormatPVRTC_RGBA_2BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_2BPP', 164)
MTLPixelFormatPVRTC_RGBA_2BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_2BPP_sRGB', 165)
MTLPixelFormatPVRTC_RGBA_4BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_4BPP', 166)
MTLPixelFormatPVRTC_RGBA_4BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_4BPP_sRGB', 167)
MTLPixelFormatEAC_R11Unorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_R11Unorm', 170)
MTLPixelFormatEAC_R11Snorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_R11Snorm', 172)
MTLPixelFormatEAC_RG11Unorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RG11Unorm', 174)
MTLPixelFormatEAC_RG11Snorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RG11Snorm', 176)
MTLPixelFormatEAC_RGBA8 = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RGBA8', 178)
MTLPixelFormatEAC_RGBA8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RGBA8_sRGB', 179)
MTLPixelFormatETC2_RGB8 = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8', 180)
MTLPixelFormatETC2_RGB8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8_sRGB', 181)
MTLPixelFormatETC2_RGB8A1 = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8A1', 182)
MTLPixelFormatETC2_RGB8A1_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8A1_sRGB', 183)
MTLPixelFormatASTC_4x4_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_sRGB', 186)
MTLPixelFormatASTC_5x4_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_sRGB', 187)
MTLPixelFormatASTC_5x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_sRGB', 188)
MTLPixelFormatASTC_6x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_sRGB', 189)
MTLPixelFormatASTC_6x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_sRGB', 190)
MTLPixelFormatASTC_8x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_sRGB', 192)
MTLPixelFormatASTC_8x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_sRGB', 193)
MTLPixelFormatASTC_8x8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_sRGB', 194)
MTLPixelFormatASTC_10x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_sRGB', 195)
MTLPixelFormatASTC_10x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_sRGB', 196)
MTLPixelFormatASTC_10x8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_sRGB', 197)
MTLPixelFormatASTC_10x10_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_sRGB', 198)
MTLPixelFormatASTC_12x10_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_sRGB', 199)
MTLPixelFormatASTC_12x12_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_sRGB', 200)
MTLPixelFormatASTC_4x4_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_LDR', 204)
MTLPixelFormatASTC_5x4_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_LDR', 205)
MTLPixelFormatASTC_5x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_LDR', 206)
MTLPixelFormatASTC_6x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_LDR', 207)
MTLPixelFormatASTC_6x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_LDR', 208)
MTLPixelFormatASTC_8x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_LDR', 210)
MTLPixelFormatASTC_8x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_LDR', 211)
MTLPixelFormatASTC_8x8_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_LDR', 212)
MTLPixelFormatASTC_10x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_LDR', 213)
MTLPixelFormatASTC_10x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_LDR', 214)
MTLPixelFormatASTC_10x8_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_LDR', 215)
MTLPixelFormatASTC_10x10_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_LDR', 216)
MTLPixelFormatASTC_12x10_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_LDR', 217)
MTLPixelFormatASTC_12x12_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_LDR', 218)
MTLPixelFormatASTC_4x4_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_HDR', 222)
MTLPixelFormatASTC_5x4_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_HDR', 223)
MTLPixelFormatASTC_5x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_HDR', 224)
MTLPixelFormatASTC_6x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_HDR', 225)
MTLPixelFormatASTC_6x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_HDR', 226)
MTLPixelFormatASTC_8x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_HDR', 228)
MTLPixelFormatASTC_8x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_HDR', 229)
MTLPixelFormatASTC_8x8_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_HDR', 230)
MTLPixelFormatASTC_10x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_HDR', 231)
MTLPixelFormatASTC_10x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_HDR', 232)
MTLPixelFormatASTC_10x8_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_HDR', 233)
MTLPixelFormatASTC_10x10_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_HDR', 234)
MTLPixelFormatASTC_12x10_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_HDR', 235)
MTLPixelFormatASTC_12x12_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_HDR', 236)
MTLPixelFormatGBGR422 = enum_MTLPixelFormat.define('MTLPixelFormatGBGR422', 240)
MTLPixelFormatBGRG422 = enum_MTLPixelFormat.define('MTLPixelFormatBGRG422', 241)
MTLPixelFormatDepth16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatDepth16Unorm', 250)
MTLPixelFormatDepth32Float = enum_MTLPixelFormat.define('MTLPixelFormatDepth32Float', 252)
MTLPixelFormatStencil8 = enum_MTLPixelFormat.define('MTLPixelFormatStencil8', 253)
MTLPixelFormatDepth24Unorm_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatDepth24Unorm_Stencil8', 255)
MTLPixelFormatDepth32Float_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatDepth32Float_Stencil8', 260)
MTLPixelFormatX32_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatX32_Stencil8', 261)
MTLPixelFormatX24_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatX24_Stencil8', 262)

MTLPixelFormat = enum_MTLPixelFormat
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
enum_MTLTextureUsage = CEnum(NSUInteger)
MTLTextureUsageUnknown = enum_MTLTextureUsage.define('MTLTextureUsageUnknown', 0)
MTLTextureUsageShaderRead = enum_MTLTextureUsage.define('MTLTextureUsageShaderRead', 1)
MTLTextureUsageShaderWrite = enum_MTLTextureUsage.define('MTLTextureUsageShaderWrite', 2)
MTLTextureUsageRenderTarget = enum_MTLTextureUsage.define('MTLTextureUsageRenderTarget', 4)
MTLTextureUsagePixelFormatView = enum_MTLTextureUsage.define('MTLTextureUsagePixelFormatView', 16)
MTLTextureUsageShaderAtomic = enum_MTLTextureUsage.define('MTLTextureUsageShaderAtomic', 32)

MTLTextureUsage = enum_MTLTextureUsage
enum_MTLTextureCompressionType = CEnum(NSInteger)
MTLTextureCompressionTypeLossless = enum_MTLTextureCompressionType.define('MTLTextureCompressionTypeLossless', 0)
MTLTextureCompressionTypeLossy = enum_MTLTextureCompressionType.define('MTLTextureCompressionTypeLossy', 1)

MTLTextureCompressionType = enum_MTLTextureCompressionType
class MTLTextureSwizzleChannels(Struct): pass
uint8_t = ctypes.c_ubyte
enum_MTLTextureSwizzle = CEnum(uint8_t)
MTLTextureSwizzleZero = enum_MTLTextureSwizzle.define('MTLTextureSwizzleZero', 0)
MTLTextureSwizzleOne = enum_MTLTextureSwizzle.define('MTLTextureSwizzleOne', 1)
MTLTextureSwizzleRed = enum_MTLTextureSwizzle.define('MTLTextureSwizzleRed', 2)
MTLTextureSwizzleGreen = enum_MTLTextureSwizzle.define('MTLTextureSwizzleGreen', 3)
MTLTextureSwizzleBlue = enum_MTLTextureSwizzle.define('MTLTextureSwizzleBlue', 4)
MTLTextureSwizzleAlpha = enum_MTLTextureSwizzle.define('MTLTextureSwizzleAlpha', 5)

MTLTextureSwizzle = enum_MTLTextureSwizzle
MTLTextureSwizzleChannels._fields_ = [
  ('red', MTLTextureSwizzle),
  ('green', MTLTextureSwizzle),
  ('blue', MTLTextureSwizzle),
  ('alpha', MTLTextureSwizzle),
]
MTLTextureDescriptor._methods_ = [
  ('textureType', MTLTextureType, []),
  ('setTextureType:', None, [MTLTextureType]),
  ('pixelFormat', MTLPixelFormat, []),
  ('setPixelFormat:', None, [MTLPixelFormat]),
  ('width', NSUInteger, []),
  ('setWidth:', None, [NSUInteger]),
  ('height', NSUInteger, []),
  ('setHeight:', None, [NSUInteger]),
  ('depth', NSUInteger, []),
  ('setDepth:', None, [NSUInteger]),
  ('mipmapLevelCount', NSUInteger, []),
  ('setMipmapLevelCount:', None, [NSUInteger]),
  ('sampleCount', NSUInteger, []),
  ('setSampleCount:', None, [NSUInteger]),
  ('arrayLength', NSUInteger, []),
  ('setArrayLength:', None, [NSUInteger]),
  ('resourceOptions', MTLResourceOptions, []),
  ('setResourceOptions:', None, [MTLResourceOptions]),
  ('cpuCacheMode', MTLCPUCacheMode, []),
  ('setCpuCacheMode:', None, [MTLCPUCacheMode]),
  ('storageMode', MTLStorageMode, []),
  ('setStorageMode:', None, [MTLStorageMode]),
  ('hazardTrackingMode', MTLHazardTrackingMode, []),
  ('setHazardTrackingMode:', None, [MTLHazardTrackingMode]),
  ('usage', MTLTextureUsage, []),
  ('setUsage:', None, [MTLTextureUsage]),
  ('allowGPUOptimizedContents', BOOL, []),
  ('setAllowGPUOptimizedContents:', None, [BOOL]),
  ('compressionType', MTLTextureCompressionType, []),
  ('setCompressionType:', None, [MTLTextureCompressionType]),
  ('swizzle', MTLTextureSwizzleChannels, []),
  ('setSwizzle:', None, [MTLTextureSwizzleChannels]),
]
MTLTextureDescriptor._classmethods_ = [
  ('texture2DDescriptorWithPixelFormat:width:height:mipmapped:', MTLTextureDescriptor, [MTLPixelFormat, NSUInteger, NSUInteger, BOOL]),
  ('textureCubeDescriptorWithPixelFormat:size:mipmapped:', MTLTextureDescriptor, [MTLPixelFormat, NSUInteger, BOOL]),
  ('textureBufferDescriptorWithPixelFormat:width:resourceOptions:usage:', MTLTextureDescriptor, [MTLPixelFormat, NSUInteger, MTLResourceOptions, MTLTextureUsage]),
]
class _anondynamic2(objc.Spec): pass
class MTLHeapDescriptor(objc.Spec): pass
class _anondynamic3(objc.Spec): pass
class struct__NSRange(Struct): pass
NSRange = struct__NSRange
struct__NSRange._fields_ = [
  ('location', NSUInteger),
  ('length', NSUInteger),
]
class _anondynamic4(objc.Spec): pass
class MTLRegion(Struct): pass
class MTLOrigin(Struct): pass
MTLOrigin._fields_ = [
  ('x', NSUInteger),
  ('y', NSUInteger),
  ('z', NSUInteger),
]
class MTLSize(Struct): pass
MTLSize._fields_ = [
  ('width', NSUInteger),
  ('height', NSUInteger),
  ('depth', NSUInteger),
]
MTLRegion._fields_ = [
  ('origin', MTLOrigin),
  ('size', MTLSize),
]
class MTLSharedTextureHandle(NSObject): pass
MTLSharedTextureHandle._methods_ = [
  ('device', _anondynamic0, []),
  ('label', NSString, []),
]
class _anondynamic5(objc.Spec): pass
enum_MTLPurgeableState = CEnum(NSUInteger)
MTLPurgeableStateKeepCurrent = enum_MTLPurgeableState.define('MTLPurgeableStateKeepCurrent', 1)
MTLPurgeableStateNonVolatile = enum_MTLPurgeableState.define('MTLPurgeableStateNonVolatile', 2)
MTLPurgeableStateVolatile = enum_MTLPurgeableState.define('MTLPurgeableStateVolatile', 3)
MTLPurgeableStateEmpty = enum_MTLPurgeableState.define('MTLPurgeableStateEmpty', 4)

MTLPurgeableState = enum_MTLPurgeableState
_anondynamic5._methods_ = [
  ('setPurgeableState:', MTLPurgeableState, [MTLPurgeableState]),
  ('makeAliasable', None, []),
  ('isAliasable', BOOL, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', _anondynamic0, []),
  ('cpuCacheMode', MTLCPUCacheMode, []),
  ('storageMode', MTLStorageMode, []),
  ('hazardTrackingMode', MTLHazardTrackingMode, []),
  ('resourceOptions', MTLResourceOptions, []),
  ('heap', _anondynamic2, []),
  ('heapOffset', NSUInteger, []),
  ('allocatedSize', NSUInteger, []),
]
class struct___IOSurface(Struct): pass
IOSurfaceRef = ctypes.POINTER(struct___IOSurface)
class struct_MTLResourceID(Struct): pass
MTLResourceID = struct_MTLResourceID
uint64_t = ctypes.c_ulonglong
struct_MTLResourceID._fields_ = [
  ('_impl', uint64_t),
]
_anondynamic4._methods_ = [
  ('getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:', None, [ctypes.c_void_p, NSUInteger, NSUInteger, MTLRegion, NSUInteger, NSUInteger]),
  ('replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:', None, [MTLRegion, NSUInteger, NSUInteger, ctypes.c_void_p, NSUInteger, NSUInteger]),
  ('getBytes:bytesPerRow:fromRegion:mipmapLevel:', None, [ctypes.c_void_p, NSUInteger, MTLRegion, NSUInteger]),
  ('replaceRegion:mipmapLevel:withBytes:bytesPerRow:', None, [MTLRegion, NSUInteger, ctypes.c_void_p, NSUInteger]),
  ('newTextureViewWithPixelFormat:', _anondynamic4, [MTLPixelFormat]),
  ('newTextureViewWithPixelFormat:textureType:levels:slices:', _anondynamic4, [MTLPixelFormat, MTLTextureType, NSRange, NSRange]),
  ('newSharedTextureHandle', MTLSharedTextureHandle, []),
  ('newRemoteTextureViewForDevice:', _anondynamic4, [_anondynamic0]),
  ('newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:', _anondynamic4, [MTLPixelFormat, MTLTextureType, NSRange, NSRange, MTLTextureSwizzleChannels]),
  ('rootResource', _anondynamic5, []),
  ('parentTexture', _anondynamic4, []),
  ('parentRelativeLevel', NSUInteger, []),
  ('parentRelativeSlice', NSUInteger, []),
  ('buffer', _anondynamic3, []),
  ('bufferOffset', NSUInteger, []),
  ('bufferBytesPerRow', NSUInteger, []),
  ('iosurface', IOSurfaceRef, []),
  ('iosurfacePlane', NSUInteger, []),
  ('textureType', MTLTextureType, []),
  ('pixelFormat', MTLPixelFormat, []),
  ('width', NSUInteger, []),
  ('height', NSUInteger, []),
  ('depth', NSUInteger, []),
  ('mipmapLevelCount', NSUInteger, []),
  ('sampleCount', NSUInteger, []),
  ('arrayLength', NSUInteger, []),
  ('usage', MTLTextureUsage, []),
  ('isShareable', BOOL, []),
  ('isFramebufferOnly', BOOL, []),
  ('firstMipmapInTail', NSUInteger, []),
  ('tailSizeInBytes', NSUInteger, []),
  ('isSparse', BOOL, []),
  ('allowGPUOptimizedContents', BOOL, []),
  ('compressionType', MTLTextureCompressionType, []),
  ('gpuResourceID', MTLResourceID, []),
  ('remoteStorageTexture', _anondynamic4, []),
  ('swizzle', MTLTextureSwizzleChannels, []),
]
_anondynamic3._methods_ = [
  ('contents', ctypes.c_void_p, []),
  ('didModifyRange:', None, [NSRange]),
  ('newTextureWithDescriptor:offset:bytesPerRow:', _anondynamic4, [MTLTextureDescriptor, NSUInteger, NSUInteger]),
  ('addDebugMarker:range:', None, [NSString, NSRange]),
  ('removeAllDebugMarkers', None, []),
  ('newRemoteBufferViewForDevice:', _anondynamic3, [_anondynamic0]),
  ('length', NSUInteger, []),
  ('remoteStorageBuffer', _anondynamic3, []),
  ('gpuAddress', uint64_t, []),
]
class _anondynamic6(objc.Spec): pass
class MTLDepthStencilDescriptor(objc.Spec): pass
class _anondynamic7(objc.Spec): pass
class MTLSamplerDescriptor(objc.Spec): pass
class _anondynamic8(objc.Spec): pass
class _anondynamic9(objc.Spec): pass
class _anondynamic10(objc.Spec): pass
class MTLArgument(NSObject): pass
enum_MTLArgumentType = CEnum(NSUInteger)
MTLArgumentTypeBuffer = enum_MTLArgumentType.define('MTLArgumentTypeBuffer', 0)
MTLArgumentTypeThreadgroupMemory = enum_MTLArgumentType.define('MTLArgumentTypeThreadgroupMemory', 1)
MTLArgumentTypeTexture = enum_MTLArgumentType.define('MTLArgumentTypeTexture', 2)
MTLArgumentTypeSampler = enum_MTLArgumentType.define('MTLArgumentTypeSampler', 3)
MTLArgumentTypeImageblockData = enum_MTLArgumentType.define('MTLArgumentTypeImageblockData', 16)
MTLArgumentTypeImageblock = enum_MTLArgumentType.define('MTLArgumentTypeImageblock', 17)
MTLArgumentTypeVisibleFunctionTable = enum_MTLArgumentType.define('MTLArgumentTypeVisibleFunctionTable', 24)
MTLArgumentTypePrimitiveAccelerationStructure = enum_MTLArgumentType.define('MTLArgumentTypePrimitiveAccelerationStructure', 25)
MTLArgumentTypeInstanceAccelerationStructure = enum_MTLArgumentType.define('MTLArgumentTypeInstanceAccelerationStructure', 26)
MTLArgumentTypeIntersectionFunctionTable = enum_MTLArgumentType.define('MTLArgumentTypeIntersectionFunctionTable', 27)

MTLArgumentType = enum_MTLArgumentType
enum_MTLBindingAccess = CEnum(NSUInteger)
MTLBindingAccessReadOnly = enum_MTLBindingAccess.define('MTLBindingAccessReadOnly', 0)
MTLBindingAccessReadWrite = enum_MTLBindingAccess.define('MTLBindingAccessReadWrite', 1)
MTLBindingAccessWriteOnly = enum_MTLBindingAccess.define('MTLBindingAccessWriteOnly', 2)
MTLArgumentAccessReadOnly = enum_MTLBindingAccess.define('MTLArgumentAccessReadOnly', 0)
MTLArgumentAccessReadWrite = enum_MTLBindingAccess.define('MTLArgumentAccessReadWrite', 1)
MTLArgumentAccessWriteOnly = enum_MTLBindingAccess.define('MTLArgumentAccessWriteOnly', 2)

MTLBindingAccess = enum_MTLBindingAccess
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
class MTLType(NSObject): pass
MTLType._methods_ = [
  ('dataType', MTLDataType, []),
]
class MTLStructType(MTLType): pass
class MTLStructMember(NSObject): pass
class MTLArrayType(MTLType): pass
class MTLTextureReferenceType(MTLType): pass
MTLTextureReferenceType._methods_ = [
  ('textureDataType', MTLDataType, []),
  ('textureType', MTLTextureType, []),
  ('access', MTLBindingAccess, []),
  ('isDepthTexture', BOOL, []),
]
class MTLPointerType(MTLType): pass
MTLPointerType._methods_ = [
  ('elementStructType', MTLStructType, []),
  ('elementArrayType', MTLArrayType, []),
  ('elementType', MTLDataType, []),
  ('access', MTLBindingAccess, []),
  ('alignment', NSUInteger, []),
  ('dataSize', NSUInteger, []),
  ('elementIsArgumentBuffer', BOOL, []),
]
MTLArrayType._methods_ = [
  ('elementStructType', MTLStructType, []),
  ('elementArrayType', MTLArrayType, []),
  ('elementTextureReferenceType', MTLTextureReferenceType, []),
  ('elementPointerType', MTLPointerType, []),
  ('elementType', MTLDataType, []),
  ('arrayLength', NSUInteger, []),
  ('stride', NSUInteger, []),
  ('argumentIndexStride', NSUInteger, []),
]
MTLStructMember._methods_ = [
  ('structType', MTLStructType, []),
  ('arrayType', MTLArrayType, []),
  ('textureReferenceType', MTLTextureReferenceType, []),
  ('pointerType', MTLPointerType, []),
  ('name', NSString, []),
  ('offset', NSUInteger, []),
  ('dataType', MTLDataType, []),
  ('argumentIndex', NSUInteger, []),
]
MTLStructType._methods_ = [
  ('memberByName:', MTLStructMember, [NSString]),
]
MTLArgument._methods_ = [
  ('name', NSString, []),
  ('type', MTLArgumentType, []),
  ('access', MTLBindingAccess, []),
  ('index', NSUInteger, []),
  ('isActive', BOOL, []),
  ('bufferAlignment', NSUInteger, []),
  ('bufferDataSize', NSUInteger, []),
  ('bufferDataType', MTLDataType, []),
  ('bufferStructType', MTLStructType, []),
  ('bufferPointerType', MTLPointerType, []),
  ('threadgroupMemoryAlignment', NSUInteger, []),
  ('threadgroupMemoryDataSize', NSUInteger, []),
  ('textureType', MTLTextureType, []),
  ('textureDataType', MTLDataType, []),
  ('isDepthTexture', BOOL, []),
  ('arrayLength', NSUInteger, []),
]
enum_MTLFunctionType = CEnum(NSUInteger)
MTLFunctionTypeVertex = enum_MTLFunctionType.define('MTLFunctionTypeVertex', 1)
MTLFunctionTypeFragment = enum_MTLFunctionType.define('MTLFunctionTypeFragment', 2)
MTLFunctionTypeKernel = enum_MTLFunctionType.define('MTLFunctionTypeKernel', 3)
MTLFunctionTypeVisible = enum_MTLFunctionType.define('MTLFunctionTypeVisible', 5)
MTLFunctionTypeIntersection = enum_MTLFunctionType.define('MTLFunctionTypeIntersection', 6)
MTLFunctionTypeMesh = enum_MTLFunctionType.define('MTLFunctionTypeMesh', 7)
MTLFunctionTypeObject = enum_MTLFunctionType.define('MTLFunctionTypeObject', 8)

MTLFunctionType = enum_MTLFunctionType
enum_MTLPatchType = CEnum(NSUInteger)
MTLPatchTypeNone = enum_MTLPatchType.define('MTLPatchTypeNone', 0)
MTLPatchTypeTriangle = enum_MTLPatchType.define('MTLPatchTypeTriangle', 1)
MTLPatchTypeQuad = enum_MTLPatchType.define('MTLPatchTypeQuad', 2)

MTLPatchType = enum_MTLPatchType
enum_MTLFunctionOptions = CEnum(NSUInteger)
MTLFunctionOptionNone = enum_MTLFunctionOptions.define('MTLFunctionOptionNone', 0)
MTLFunctionOptionCompileToBinary = enum_MTLFunctionOptions.define('MTLFunctionOptionCompileToBinary', 1)
MTLFunctionOptionStoreFunctionInMetalScript = enum_MTLFunctionOptions.define('MTLFunctionOptionStoreFunctionInMetalScript', 2)

MTLFunctionOptions = enum_MTLFunctionOptions
_anondynamic9._methods_ = [
  ('newArgumentEncoderWithBufferIndex:', _anondynamic10, [NSUInteger]),
  ('newArgumentEncoderWithBufferIndex:reflection:', _anondynamic10, [NSUInteger, ctypes.POINTER(MTLArgument)]),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', _anondynamic0, []),
  ('functionType', MTLFunctionType, []),
  ('patchType', MTLPatchType, []),
  ('patchControlPointCount', NSInteger, []),
  ('name', NSString, []),
  ('options', MTLFunctionOptions, []),
]
class MTLFunctionConstantValues(NSObject): pass
MTLFunctionConstantValues._methods_ = [
  ('setConstantValue:type:atIndex:', None, [ctypes.c_void_p, MTLDataType, NSUInteger]),
  ('setConstantValues:type:withRange:', None, [ctypes.c_void_p, MTLDataType, NSRange]),
  ('setConstantValue:type:withName:', None, [ctypes.c_void_p, MTLDataType, NSString]),
  ('reset', None, []),
]
class NSError(NSObject): pass
NSErrorDomain = NSString
NSError._methods_ = [
  ('domain', NSErrorDomain, []),
  ('code', NSInteger, []),
  ('localizedDescription', NSString, []),
  ('localizedFailureReason', NSString, []),
  ('localizedRecoverySuggestion', NSString, []),
  ('recoveryAttempter', objc.id_, []),
  ('helpAnchor', NSString, []),
]
class MTLFunctionDescriptor(NSObject): pass
MTLFunctionDescriptor._methods_ = [
  ('name', NSString, []),
  ('setName:', None, [NSString]),
  ('specializedName', NSString, []),
  ('setSpecializedName:', None, [NSString]),
  ('constantValues', MTLFunctionConstantValues, []),
  ('setConstantValues:', None, [MTLFunctionConstantValues]),
  ('options', MTLFunctionOptions, []),
  ('setOptions:', None, [MTLFunctionOptions]),
]
MTLFunctionDescriptor._classmethods_ = [
  ('functionDescriptor', MTLFunctionDescriptor, []),
]
class MTLIntersectionFunctionDescriptor(MTLFunctionDescriptor): pass
enum_MTLLibraryType = CEnum(NSInteger)
MTLLibraryTypeExecutable = enum_MTLLibraryType.define('MTLLibraryTypeExecutable', 0)
MTLLibraryTypeDynamic = enum_MTLLibraryType.define('MTLLibraryTypeDynamic', 1)

MTLLibraryType = enum_MTLLibraryType
_anondynamic8._methods_ = [
  ('newFunctionWithName:', _anondynamic9, [NSString]),
  ('newFunctionWithName:constantValues:error:', _anondynamic9, [NSString, MTLFunctionConstantValues, ctypes.POINTER(NSError)]),
  ('newFunctionWithDescriptor:error:', _anondynamic9, [MTLFunctionDescriptor, ctypes.POINTER(NSError)]),
  ('newIntersectionFunctionWithDescriptor:error:', _anondynamic9, [MTLIntersectionFunctionDescriptor, ctypes.POINTER(NSError)]),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', _anondynamic0, []),
  ('type', MTLLibraryType, []),
  ('installName', NSString, []),
]
class NSBundle(NSObject): pass
class NSURL(NSObject): pass
NSURLResourceKey = NSString
enum_NSURLBookmarkCreationOptions = CEnum(NSUInteger)
NSURLBookmarkCreationPreferFileIDResolution = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationPreferFileIDResolution', 256)
NSURLBookmarkCreationMinimalBookmark = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationMinimalBookmark', 512)
NSURLBookmarkCreationSuitableForBookmarkFile = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationSuitableForBookmarkFile', 1024)
NSURLBookmarkCreationWithSecurityScope = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationWithSecurityScope', 2048)
NSURLBookmarkCreationSecurityScopeAllowOnlyReadAccess = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationSecurityScopeAllowOnlyReadAccess', 4096)
NSURLBookmarkCreationWithoutImplicitSecurityScope = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationWithoutImplicitSecurityScope', 536870912)

NSURLBookmarkCreationOptions = enum_NSURLBookmarkCreationOptions
enum_NSURLBookmarkResolutionOptions = CEnum(NSUInteger)
NSURLBookmarkResolutionWithoutUI = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutUI', 256)
NSURLBookmarkResolutionWithoutMounting = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutMounting', 512)
NSURLBookmarkResolutionWithSecurityScope = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithSecurityScope', 1024)
NSURLBookmarkResolutionWithoutImplicitStartAccessing = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutImplicitStartAccessing', 32768)

NSURLBookmarkResolutionOptions = enum_NSURLBookmarkResolutionOptions
class NSValue(NSObject): pass
NSValue._methods_ = [
  ('getValue:size:', None, [ctypes.c_void_p, NSUInteger]),
  ('initWithBytes:objCType:', NSValue, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]),
  ('initWithCoder:', NSValue, [NSCoder]),
  ('objCType', ctypes.POINTER(ctypes.c_char), []),
]
class NSNumber(NSValue): pass
enum_NSComparisonResult = CEnum(NSInteger)
NSOrderedAscending = enum_NSComparisonResult.define('NSOrderedAscending', -1)
NSOrderedSame = enum_NSComparisonResult.define('NSOrderedSame', 0)
NSOrderedDescending = enum_NSComparisonResult.define('NSOrderedDescending', 1)

NSComparisonResult = enum_NSComparisonResult
NSNumber._methods_ = [
  ('initWithCoder:', NSNumber, [NSCoder]),
  ('initWithChar:', NSNumber, [ctypes.c_char]),
  ('initWithUnsignedChar:', NSNumber, [ctypes.c_ubyte]),
  ('initWithShort:', NSNumber, [ctypes.c_short]),
  ('initWithUnsignedShort:', NSNumber, [ctypes.c_ushort]),
  ('initWithInt:', NSNumber, [ctypes.c_int]),
  ('initWithUnsignedInt:', NSNumber, [ctypes.c_uint]),
  ('initWithLong:', NSNumber, [ctypes.c_long]),
  ('initWithUnsignedLong:', NSNumber, [ctypes.c_ulong]),
  ('initWithLongLong:', NSNumber, [ctypes.c_longlong]),
  ('initWithUnsignedLongLong:', NSNumber, [ctypes.c_ulonglong]),
  ('initWithFloat:', NSNumber, [ctypes.c_float]),
  ('initWithDouble:', NSNumber, [ctypes.c_double]),
  ('initWithBool:', NSNumber, [BOOL]),
  ('initWithInteger:', NSNumber, [NSInteger]),
  ('initWithUnsignedInteger:', NSNumber, [NSUInteger]),
  ('compare:', NSComparisonResult, [NSNumber]),
  ('isEqualToNumber:', BOOL, [NSNumber]),
  ('descriptionWithLocale:', NSString, [objc.id_]),
  ('charValue', ctypes.c_char, []),
  ('unsignedCharValue', ctypes.c_ubyte, []),
  ('shortValue', ctypes.c_short, []),
  ('unsignedShortValue', ctypes.c_ushort, []),
  ('intValue', ctypes.c_int, []),
  ('unsignedIntValue', ctypes.c_uint, []),
  ('longValue', ctypes.c_long, []),
  ('unsignedLongValue', ctypes.c_ulong, []),
  ('longLongValue', ctypes.c_longlong, []),
  ('unsignedLongLongValue', ctypes.c_ulonglong, []),
  ('floatValue', ctypes.c_float, []),
  ('doubleValue', ctypes.c_double, []),
  ('boolValue', BOOL, []),
  ('integerValue', NSInteger, []),
  ('unsignedIntegerValue', NSUInteger, []),
  ('stringValue', NSString, []),
]
NSURLBookmarkFileCreationOptions = ctypes.c_ulong
NSURL._methods_ = [
  ('initWithScheme:host:path:', NSURL, [NSString, NSString, NSString]),
  ('initFileURLWithPath:isDirectory:relativeToURL:', NSURL, [NSString, BOOL, NSURL]),
  ('initFileURLWithPath:relativeToURL:', NSURL, [NSString, NSURL]),
  ('initFileURLWithPath:isDirectory:', NSURL, [NSString, BOOL]),
  ('initFileURLWithPath:', NSURL, [NSString]),
  ('initFileURLWithFileSystemRepresentation:isDirectory:relativeToURL:', NSURL, [ctypes.POINTER(ctypes.c_char), BOOL, NSURL]),
  ('initWithString:', NSURL, [NSString]),
  ('initWithString:relativeToURL:', NSURL, [NSString, NSURL]),
  ('initWithString:encodingInvalidCharacters:', NSURL, [NSString, BOOL]),
  ('initWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('initAbsoluteURLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('getFileSystemRepresentation:maxLength:', BOOL, [ctypes.POINTER(ctypes.c_char), NSUInteger]),
  ('checkResourceIsReachableAndReturnError:', BOOL, [ctypes.POINTER(NSError)]),
  ('isFileReferenceURL', BOOL, []),
  ('fileReferenceURL', NSURL, []),
  ('getResourceValue:forKey:error:', BOOL, [ctypes.POINTER(objc.id_), NSURLResourceKey, ctypes.POINTER(NSError)]),
  ('setResourceValue:forKey:error:', BOOL, [objc.id_, NSURLResourceKey, ctypes.POINTER(NSError)]),
  ('removeCachedResourceValueForKey:', None, [NSURLResourceKey]),
  ('removeAllCachedResourceValues', None, []),
  ('setTemporaryResourceValue:forKey:', None, [objc.id_, NSURLResourceKey]),
  ('initByResolvingBookmarkData:options:relativeToURL:bookmarkDataIsStale:error:', NSURL, [NSData, NSURLBookmarkResolutionOptions, NSURL, ctypes.POINTER(BOOL), ctypes.POINTER(NSError)]),
  ('startAccessingSecurityScopedResource', BOOL, []),
  ('stopAccessingSecurityScopedResource', None, []),
  ('dataRepresentation', NSData, []),
  ('absoluteString', NSString, []),
  ('relativeString', NSString, []),
  ('baseURL', NSURL, []),
  ('absoluteURL', NSURL, []),
  ('scheme', NSString, []),
  ('resourceSpecifier', NSString, []),
  ('host', NSString, []),
  ('port', NSNumber, []),
  ('user', NSString, []),
  ('password', NSString, []),
  ('path', NSString, []),
  ('fragment', NSString, []),
  ('parameterString', NSString, []),
  ('query', NSString, []),
  ('relativePath', NSString, []),
  ('hasDirectoryPath', BOOL, []),
  ('fileSystemRepresentation', ctypes.POINTER(ctypes.c_char), []),
  ('isFileURL', BOOL, []),
  ('standardizedURL', NSURL, []),
  ('filePathURL', NSURL, []),
]
NSURL._classmethods_ = [
  ('fileURLWithPath:isDirectory:relativeToURL:', NSURL, [NSString, BOOL, NSURL]),
  ('fileURLWithPath:relativeToURL:', NSURL, [NSString, NSURL]),
  ('fileURLWithPath:isDirectory:', NSURL, [NSString, BOOL]),
  ('fileURLWithPath:', NSURL, [NSString]),
  ('fileURLWithFileSystemRepresentation:isDirectory:relativeToURL:', NSURL, [ctypes.POINTER(ctypes.c_char), BOOL, NSURL]),
  ('URLWithString:', NSURL, [NSString]),
  ('URLWithString:relativeToURL:', NSURL, [NSString, NSURL]),
  ('URLWithString:encodingInvalidCharacters:', NSURL, [NSString, BOOL]),
  ('URLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('absoluteURLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('URLByResolvingBookmarkData:options:relativeToURL:bookmarkDataIsStale:error:', NSURL, [NSData, NSURLBookmarkResolutionOptions, NSURL, ctypes.POINTER(BOOL), ctypes.POINTER(NSError)]),
  ('writeBookmarkData:toURL:options:error:', BOOL, [NSData, NSURL, NSURLBookmarkFileCreationOptions, ctypes.POINTER(NSError)]),
  ('bookmarkDataWithContentsOfURL:error:', NSData, [NSURL, ctypes.POINTER(NSError)]),
  ('URLByResolvingAliasFileAtURL:options:error:', NSURL, [NSURL, NSURLBookmarkResolutionOptions, ctypes.POINTER(NSError)]),
]
class NSAttributedString(NSObject): pass
NSAttributedString._methods_ = [
  ('string', NSString, []),
]
NSBundle._methods_ = [
  ('initWithPath:', NSBundle, [NSString]),
  ('initWithURL:', NSBundle, [NSURL]),
  ('load', BOOL, []),
  ('unload', BOOL, []),
  ('preflightAndReturnError:', BOOL, [ctypes.POINTER(NSError)]),
  ('loadAndReturnError:', BOOL, [ctypes.POINTER(NSError)]),
  ('URLForAuxiliaryExecutable:', NSURL, [NSString]),
  ('pathForAuxiliaryExecutable:', NSString, [NSString]),
  ('URLForResource:withExtension:', NSURL, [NSString, NSString]),
  ('URLForResource:withExtension:subdirectory:', NSURL, [NSString, NSString, NSString]),
  ('URLForResource:withExtension:subdirectory:localization:', NSURL, [NSString, NSString, NSString, NSString]),
  ('pathForResource:ofType:', NSString, [NSString, NSString]),
  ('pathForResource:ofType:inDirectory:', NSString, [NSString, NSString, NSString]),
  ('pathForResource:ofType:inDirectory:forLocalization:', NSString, [NSString, NSString, NSString, NSString]),
  ('localizedStringForKey:value:table:', NSString, [NSString, NSString, NSString]),
  ('localizedAttributedStringForKey:value:table:', NSAttributedString, [NSString, NSString, NSString]),
  ('objectForInfoDictionaryKey:', objc.id_, [NSString]),
  ('isLoaded', BOOL, []),
  ('bundleURL', NSURL, []),
  ('resourceURL', NSURL, []),
  ('executableURL', NSURL, []),
  ('privateFrameworksURL', NSURL, []),
  ('sharedFrameworksURL', NSURL, []),
  ('sharedSupportURL', NSURL, []),
  ('builtInPlugInsURL', NSURL, []),
  ('appStoreReceiptURL', NSURL, []),
  ('bundlePath', NSString, []),
  ('resourcePath', NSString, []),
  ('executablePath', NSString, []),
  ('privateFrameworksPath', NSString, []),
  ('sharedFrameworksPath', NSString, []),
  ('sharedSupportPath', NSString, []),
  ('builtInPlugInsPath', NSString, []),
  ('bundleIdentifier', NSString, []),
  ('developmentLocalization', NSString, []),
]
NSBundle._classmethods_ = [
  ('bundleWithPath:', NSBundle, [NSString]),
  ('bundleWithURL:', NSBundle, [NSURL]),
  ('bundleWithIdentifier:', NSBundle, [NSString]),
  ('URLForResource:withExtension:subdirectory:inBundleWithURL:', NSURL, [NSString, NSString, NSString, NSURL]),
  ('pathForResource:ofType:inDirectory:', NSString, [NSString, NSString, NSString]),
  ('mainBundle', NSBundle, []),
]
class MTLCompileOptions(NSObject): pass
enum_MTLLanguageVersion = CEnum(NSUInteger)
MTLLanguageVersion1_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_0', 65536)
MTLLanguageVersion1_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_1', 65537)
MTLLanguageVersion1_2 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_2', 65538)
MTLLanguageVersion2_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_0', 131072)
MTLLanguageVersion2_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_1', 131073)
MTLLanguageVersion2_2 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_2', 131074)
MTLLanguageVersion2_3 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_3', 131075)
MTLLanguageVersion2_4 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_4', 131076)
MTLLanguageVersion3_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion3_0', 196608)
MTLLanguageVersion3_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion3_1', 196609)

MTLLanguageVersion = enum_MTLLanguageVersion
enum_MTLLibraryOptimizationLevel = CEnum(NSInteger)
MTLLibraryOptimizationLevelDefault = enum_MTLLibraryOptimizationLevel.define('MTLLibraryOptimizationLevelDefault', 0)
MTLLibraryOptimizationLevelSize = enum_MTLLibraryOptimizationLevel.define('MTLLibraryOptimizationLevelSize', 1)

MTLLibraryOptimizationLevel = enum_MTLLibraryOptimizationLevel
enum_MTLCompileSymbolVisibility = CEnum(NSInteger)
MTLCompileSymbolVisibilityDefault = enum_MTLCompileSymbolVisibility.define('MTLCompileSymbolVisibilityDefault', 0)
MTLCompileSymbolVisibilityHidden = enum_MTLCompileSymbolVisibility.define('MTLCompileSymbolVisibilityHidden', 1)

MTLCompileSymbolVisibility = enum_MTLCompileSymbolVisibility
MTLCompileOptions._methods_ = [
  ('fastMathEnabled', BOOL, []),
  ('setFastMathEnabled:', None, [BOOL]),
  ('languageVersion', MTLLanguageVersion, []),
  ('setLanguageVersion:', None, [MTLLanguageVersion]),
  ('libraryType', MTLLibraryType, []),
  ('setLibraryType:', None, [MTLLibraryType]),
  ('installName', NSString, []),
  ('setInstallName:', None, [NSString]),
  ('preserveInvariance', BOOL, []),
  ('setPreserveInvariance:', None, [BOOL]),
  ('optimizationLevel', MTLLibraryOptimizationLevel, []),
  ('setOptimizationLevel:', None, [MTLLibraryOptimizationLevel]),
  ('compileSymbolVisibility', MTLCompileSymbolVisibility, []),
  ('setCompileSymbolVisibility:', None, [MTLCompileSymbolVisibility]),
  ('allowReferencingUndefinedSymbols', BOOL, []),
  ('setAllowReferencingUndefinedSymbols:', None, [BOOL]),
  ('maxTotalThreadsPerThreadgroup', NSUInteger, []),
  ('setMaxTotalThreadsPerThreadgroup:', None, [NSUInteger]),
]
class MTLStitchedLibraryDescriptor(objc.Spec): pass
class _anondynamic11(objc.Spec): pass
class MTLRenderPipelineDescriptor(objc.Spec): pass
enum_MTLPipelineOption = CEnum(NSUInteger)
MTLPipelineOptionNone = enum_MTLPipelineOption.define('MTLPipelineOptionNone', 0)
MTLPipelineOptionArgumentInfo = enum_MTLPipelineOption.define('MTLPipelineOptionArgumentInfo', 1)
MTLPipelineOptionBufferTypeInfo = enum_MTLPipelineOption.define('MTLPipelineOptionBufferTypeInfo', 2)
MTLPipelineOptionFailOnBinaryArchiveMiss = enum_MTLPipelineOption.define('MTLPipelineOptionFailOnBinaryArchiveMiss', 4)

MTLPipelineOption = enum_MTLPipelineOption
class MTLRenderPipelineReflection(objc.Spec): pass
class _anondynamic12(objc.Spec): pass
class MTLComputePipelineReflection(objc.Spec): pass
class MTLComputePipelineDescriptor(objc.Spec): pass
class _anondynamic13(objc.Spec): pass
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
class MTLTileRenderPipelineDescriptor(objc.Spec): pass
class MTLMeshRenderPipelineDescriptor(objc.Spec): pass
class MTLSamplePosition(Struct): pass
MTLSamplePosition._fields_ = [
  ('x', ctypes.c_float),
  ('y', ctypes.c_float),
]
class _anondynamic14(objc.Spec): pass
class MTLRasterizationRateMapDescriptor(objc.Spec): pass
class _anondynamic15(objc.Spec): pass
class MTLIndirectCommandBufferDescriptor(objc.Spec): pass
class _anondynamic16(objc.Spec): pass
class _anondynamic17(objc.Spec): pass
class MTLSharedEventHandle(objc.Spec): pass
class _anondynamic18(objc.Spec): pass
class _anondynamic19(objc.Spec): pass
class MTLIOCommandQueueDescriptor(objc.Spec): pass
enum_MTLSparseTextureRegionAlignmentMode = CEnum(NSUInteger)
MTLSparseTextureRegionAlignmentModeOutward = enum_MTLSparseTextureRegionAlignmentMode.define('MTLSparseTextureRegionAlignmentModeOutward', 0)
MTLSparseTextureRegionAlignmentModeInward = enum_MTLSparseTextureRegionAlignmentMode.define('MTLSparseTextureRegionAlignmentModeInward', 1)

MTLSparseTextureRegionAlignmentMode = enum_MTLSparseTextureRegionAlignmentMode
enum_MTLSparsePageSize = CEnum(NSInteger)
MTLSparsePageSize16 = enum_MTLSparsePageSize.define('MTLSparsePageSize16', 101)
MTLSparsePageSize64 = enum_MTLSparsePageSize.define('MTLSparsePageSize64', 102)
MTLSparsePageSize256 = enum_MTLSparsePageSize.define('MTLSparsePageSize256', 103)

MTLSparsePageSize = enum_MTLSparsePageSize
class _anondynamic20(objc.Spec): pass
_anondynamic20._methods_ = [
  ('resolveCounterRange:', NSData, [NSRange]),
  ('device', _anondynamic0, []),
  ('label', NSString, []),
  ('sampleCount', NSUInteger, []),
]
class MTLCounterSampleBufferDescriptor(NSObject): pass
class _anondynamic21(objc.Spec): pass
_anondynamic21._methods_ = [
  ('name', NSString, []),
]
MTLCounterSampleBufferDescriptor._methods_ = [
  ('counterSet', _anondynamic21, []),
  ('setCounterSet:', None, [_anondynamic21]),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('storageMode', MTLStorageMode, []),
  ('setStorageMode:', None, [MTLStorageMode]),
  ('sampleCount', NSUInteger, []),
  ('setSampleCount:', None, [NSUInteger]),
]
MTLTimestamp = ctypes.c_ulonglong
class _anondynamic22(objc.Spec): pass
_anondynamic22._methods_ = [
  ('bufferAlignment', NSUInteger, []),
  ('bufferDataSize', NSUInteger, []),
  ('bufferDataType', MTLDataType, []),
  ('bufferStructType', MTLStructType, []),
  ('bufferPointerType', MTLPointerType, []),
]
enum_MTLCounterSamplingPoint = CEnum(NSUInteger)
MTLCounterSamplingPointAtStageBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtStageBoundary', 0)
MTLCounterSamplingPointAtDrawBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtDrawBoundary', 1)
MTLCounterSamplingPointAtDispatchBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtDispatchBoundary', 2)
MTLCounterSamplingPointAtTileDispatchBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtTileDispatchBoundary', 3)
MTLCounterSamplingPointAtBlitBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtBlitBoundary', 4)

MTLCounterSamplingPoint = enum_MTLCounterSamplingPoint
class _anondynamic23(objc.Spec): pass
class _anondynamic24(objc.Spec): pass
class MTLBinaryArchiveDescriptor(objc.Spec): pass
class MTLAccelerationStructureSizes(Struct): pass
MTLAccelerationStructureSizes._fields_ = [
  ('accelerationStructureSize', NSUInteger),
  ('buildScratchBufferSize', NSUInteger),
  ('refitScratchBufferSize', NSUInteger),
]
class MTLAccelerationStructureDescriptor(objc.Spec): pass
class _anondynamic25(objc.Spec): pass
class MTLArchitecture(NSObject): pass
MTLArchitecture._methods_ = [
  ('name', NSString, []),
]
enum_MTLDeviceLocation = CEnum(NSUInteger)
MTLDeviceLocationBuiltIn = enum_MTLDeviceLocation.define('MTLDeviceLocationBuiltIn', 0)
MTLDeviceLocationSlot = enum_MTLDeviceLocation.define('MTLDeviceLocationSlot', 1)
MTLDeviceLocationExternal = enum_MTLDeviceLocation.define('MTLDeviceLocationExternal', 2)
MTLDeviceLocationUnspecified = enum_MTLDeviceLocation.define('MTLDeviceLocationUnspecified', -1)

MTLDeviceLocation = enum_MTLDeviceLocation
enum_MTLReadWriteTextureTier = CEnum(NSUInteger)
MTLReadWriteTextureTierNone = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTierNone', 0)
MTLReadWriteTextureTier1 = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTier1', 1)
MTLReadWriteTextureTier2 = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTier2', 2)

MTLReadWriteTextureTier = enum_MTLReadWriteTextureTier
enum_MTLArgumentBuffersTier = CEnum(NSUInteger)
MTLArgumentBuffersTier1 = enum_MTLArgumentBuffersTier.define('MTLArgumentBuffersTier1', 0)
MTLArgumentBuffersTier2 = enum_MTLArgumentBuffersTier.define('MTLArgumentBuffersTier2', 1)

MTLArgumentBuffersTier = enum_MTLArgumentBuffersTier
uint32_t = ctypes.c_uint
_anondynamic0._methods_ = [
  ('newCommandQueue', _anondynamic1, []),
  ('newCommandQueueWithMaxCommandBufferCount:', _anondynamic1, [NSUInteger]),
  ('heapTextureSizeAndAlignWithDescriptor:', MTLSizeAndAlign, [MTLTextureDescriptor]),
  ('heapBufferSizeAndAlignWithLength:options:', MTLSizeAndAlign, [NSUInteger, MTLResourceOptions]),
  ('newHeapWithDescriptor:', _anondynamic2, [MTLHeapDescriptor]),
  ('newBufferWithLength:options:', _anondynamic3, [NSUInteger, MTLResourceOptions]),
  ('newBufferWithBytes:length:options:', _anondynamic3, [ctypes.c_void_p, NSUInteger, MTLResourceOptions]),
  ('newDepthStencilStateWithDescriptor:', _anondynamic6, [MTLDepthStencilDescriptor]),
  ('newTextureWithDescriptor:', _anondynamic4, [MTLTextureDescriptor]),
  ('newTextureWithDescriptor:iosurface:plane:', _anondynamic4, [MTLTextureDescriptor, IOSurfaceRef, NSUInteger]),
  ('newSharedTextureWithDescriptor:', _anondynamic4, [MTLTextureDescriptor]),
  ('newSharedTextureWithHandle:', _anondynamic4, [MTLSharedTextureHandle]),
  ('newSamplerStateWithDescriptor:', _anondynamic7, [MTLSamplerDescriptor]),
  ('newDefaultLibrary', _anondynamic8, []),
  ('newDefaultLibraryWithBundle:error:', _anondynamic8, [NSBundle, ctypes.POINTER(NSError)]),
  ('newLibraryWithFile:error:', _anondynamic8, [NSString, ctypes.POINTER(NSError)]),
  ('newLibraryWithURL:error:', _anondynamic8, [NSURL, ctypes.POINTER(NSError)]),
  ('newLibraryWithSource:options:error:', _anondynamic8, [NSString, MTLCompileOptions, ctypes.POINTER(NSError)]),
  ('newLibraryWithStitchedDescriptor:error:', _anondynamic8, [MTLStitchedLibraryDescriptor, ctypes.POINTER(NSError)]),
  ('newRenderPipelineStateWithDescriptor:error:', _anondynamic11, [MTLRenderPipelineDescriptor, ctypes.POINTER(NSError)]),
  ('newRenderPipelineStateWithDescriptor:options:reflection:error:', _anondynamic11, [MTLRenderPipelineDescriptor, MTLPipelineOption, ctypes.POINTER(MTLRenderPipelineReflection), ctypes.POINTER(NSError)]),
  ('newComputePipelineStateWithFunction:error:', _anondynamic12, [_anondynamic9, ctypes.POINTER(NSError)]),
  ('newComputePipelineStateWithFunction:options:reflection:error:', _anondynamic12, [_anondynamic9, MTLPipelineOption, ctypes.POINTER(MTLComputePipelineReflection), ctypes.POINTER(NSError)]),
  ('newComputePipelineStateWithDescriptor:options:reflection:error:', _anondynamic12, [MTLComputePipelineDescriptor, MTLPipelineOption, ctypes.POINTER(MTLComputePipelineReflection), ctypes.POINTER(NSError)]),
  ('newFence', _anondynamic13, []),
  ('supportsFeatureSet:', BOOL, [MTLFeatureSet]),
  ('supportsFamily:', BOOL, [MTLGPUFamily]),
  ('supportsTextureSampleCount:', BOOL, [NSUInteger]),
  ('minimumLinearTextureAlignmentForPixelFormat:', NSUInteger, [MTLPixelFormat]),
  ('minimumTextureBufferAlignmentForPixelFormat:', NSUInteger, [MTLPixelFormat]),
  ('newRenderPipelineStateWithTileDescriptor:options:reflection:error:', _anondynamic11, [MTLTileRenderPipelineDescriptor, MTLPipelineOption, ctypes.POINTER(MTLRenderPipelineReflection), ctypes.POINTER(NSError)]),
  ('newRenderPipelineStateWithMeshDescriptor:options:reflection:error:', _anondynamic11, [MTLMeshRenderPipelineDescriptor, MTLPipelineOption, ctypes.POINTER(MTLRenderPipelineReflection), ctypes.POINTER(NSError)]),
  ('getDefaultSamplePositions:count:', None, [ctypes.POINTER(MTLSamplePosition), NSUInteger]),
  ('supportsRasterizationRateMapWithLayerCount:', BOOL, [NSUInteger]),
  ('newRasterizationRateMapWithDescriptor:', _anondynamic14, [MTLRasterizationRateMapDescriptor]),
  ('newIndirectCommandBufferWithDescriptor:maxCommandCount:options:', _anondynamic15, [MTLIndirectCommandBufferDescriptor, NSUInteger, MTLResourceOptions]),
  ('newEvent', _anondynamic16, []),
  ('newSharedEvent', _anondynamic17, []),
  ('newSharedEventWithHandle:', _anondynamic17, [MTLSharedEventHandle]),
  ('newIOHandleWithURL:error:', _anondynamic18, [NSURL, ctypes.POINTER(NSError)]),
  ('newIOCommandQueueWithDescriptor:error:', _anondynamic19, [MTLIOCommandQueueDescriptor, ctypes.POINTER(NSError)]),
  ('newIOHandleWithURL:compressionMethod:error:', _anondynamic18, [NSURL, MTLIOCompressionMethod, ctypes.POINTER(NSError)]),
  ('newIOFileHandleWithURL:error:', _anondynamic18, [NSURL, ctypes.POINTER(NSError)]),
  ('newIOFileHandleWithURL:compressionMethod:error:', _anondynamic18, [NSURL, MTLIOCompressionMethod, ctypes.POINTER(NSError)]),
  ('sparseTileSizeWithTextureType:pixelFormat:sampleCount:', MTLSize, [MTLTextureType, MTLPixelFormat, NSUInteger]),
  ('convertSparsePixelRegions:toTileRegions:withTileSize:alignmentMode:numRegions:', None, [ctypes.POINTER(MTLRegion), ctypes.POINTER(MTLRegion), MTLSize, MTLSparseTextureRegionAlignmentMode, NSUInteger]),
  ('convertSparseTileRegions:toPixelRegions:withTileSize:numRegions:', None, [ctypes.POINTER(MTLRegion), ctypes.POINTER(MTLRegion), MTLSize, NSUInteger]),
  ('sparseTileSizeInBytesForSparsePageSize:', NSUInteger, [MTLSparsePageSize]),
  ('sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:', MTLSize, [MTLTextureType, MTLPixelFormat, NSUInteger, MTLSparsePageSize]),
  ('newCounterSampleBufferWithDescriptor:error:', _anondynamic20, [MTLCounterSampleBufferDescriptor, ctypes.POINTER(NSError)]),
  ('sampleTimestamps:gpuTimestamp:', None, [ctypes.POINTER(MTLTimestamp), ctypes.POINTER(MTLTimestamp)]),
  ('newArgumentEncoderWithBufferBinding:', _anondynamic10, [_anondynamic22]),
  ('supportsCounterSampling:', BOOL, [MTLCounterSamplingPoint]),
  ('supportsVertexAmplificationCount:', BOOL, [NSUInteger]),
  ('newDynamicLibrary:error:', _anondynamic23, [_anondynamic8, ctypes.POINTER(NSError)]),
  ('newDynamicLibraryWithURL:error:', _anondynamic23, [NSURL, ctypes.POINTER(NSError)]),
  ('newBinaryArchiveWithDescriptor:error:', _anondynamic24, [MTLBinaryArchiveDescriptor, ctypes.POINTER(NSError)]),
  ('accelerationStructureSizesWithDescriptor:', MTLAccelerationStructureSizes, [MTLAccelerationStructureDescriptor]),
  ('newAccelerationStructureWithSize:', _anondynamic25, [NSUInteger]),
  ('newAccelerationStructureWithDescriptor:', _anondynamic25, [MTLAccelerationStructureDescriptor]),
  ('heapAccelerationStructureSizeAndAlignWithSize:', MTLSizeAndAlign, [NSUInteger]),
  ('heapAccelerationStructureSizeAndAlignWithDescriptor:', MTLSizeAndAlign, [MTLAccelerationStructureDescriptor]),
  ('name', NSString, []),
  ('registryID', uint64_t, []),
  ('architecture', MTLArchitecture, []),
  ('maxThreadsPerThreadgroup', MTLSize, []),
  ('isLowPower', BOOL, []),
  ('isHeadless', BOOL, []),
  ('isRemovable', BOOL, []),
  ('hasUnifiedMemory', BOOL, []),
  ('recommendedMaxWorkingSetSize', uint64_t, []),
  ('location', MTLDeviceLocation, []),
  ('locationNumber', NSUInteger, []),
  ('maxTransferRate', uint64_t, []),
  ('isDepth24Stencil8PixelFormatSupported', BOOL, []),
  ('readWriteTextureSupport', MTLReadWriteTextureTier, []),
  ('argumentBuffersSupport', MTLArgumentBuffersTier, []),
  ('areRasterOrderGroupsSupported', BOOL, []),
  ('supports32BitFloatFiltering', BOOL, []),
  ('supports32BitMSAA', BOOL, []),
  ('supportsQueryTextureLOD', BOOL, []),
  ('supportsBCTextureCompression', BOOL, []),
  ('supportsPullModelInterpolation', BOOL, []),
  ('areBarycentricCoordsSupported', BOOL, []),
  ('supportsShaderBarycentricCoordinates', BOOL, []),
  ('currentAllocatedSize', NSUInteger, []),
  ('maxThreadgroupMemoryLength', NSUInteger, []),
  ('maxArgumentBufferSamplerCount', NSUInteger, []),
  ('areProgrammableSamplePositionsSupported', BOOL, []),
  ('peerGroupID', uint64_t, []),
  ('peerIndex', uint32_t, []),
  ('peerCount', uint32_t, []),
  ('sparseTileSizeInBytes', NSUInteger, []),
  ('maxBufferLength', NSUInteger, []),
  ('supportsDynamicLibraries', BOOL, []),
  ('supportsRenderDynamicLibraries', BOOL, []),
  ('supportsRaytracing', BOOL, []),
  ('supportsFunctionPointers', BOOL, []),
  ('supportsFunctionPointersFromRender', BOOL, []),
  ('supportsRaytracingFromRender', BOOL, []),
  ('supportsPrimitiveMotionBlur', BOOL, []),
  ('shouldMaximizeConcurrentCompilation', BOOL, []),
  ('setShouldMaximizeConcurrentCompilation:', None, [BOOL]),
  ('maximumConcurrentCompilationTaskCount', NSUInteger, []),
]
# __attribute__((visibility("default"))) extern id<MTLDevice>  _Nullable MTLCreateSystemDefaultDevice(void) __attribute__((ns_returns_retained)) __attribute__((availability(macos, introduced=10.11))) __attribute__((availability(ios, introduced=8.0)))
try: (MTLCreateSystemDefaultDevice:=dll.MTLCreateSystemDefaultDevice).restype, MTLCreateSystemDefaultDevice.argtypes = _anondynamic0, []
except AttributeError: pass

MTLCreateSystemDefaultDevice = objc.returns_retained(MTLCreateSystemDefaultDevice)
MTLDeviceNotificationName = NSString
try: MTLDeviceWasAddedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasAddedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceRemovalRequestedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceRemovalRequestedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceWasRemovedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasRemovedNotification')
except (ValueError,AttributeError): pass
class _anondynamic26(objc.Spec): pass
_anondynamic26._methods_ = [
  ('isEqual:', BOOL, [objc.id_]),
  ('self', _anondynamic26, []),
  ('performSelector:', objc.id_, [objc.id_]),
  ('performSelector:withObject:', objc.id_, [objc.id_, objc.id_]),
  ('performSelector:withObject:withObject:', objc.id_, [objc.id_, objc.id_, objc.id_]),
  ('isProxy', BOOL, []),
  ('conformsToProtocol:', BOOL, [Protocol]),
  ('respondsToSelector:', BOOL, [objc.id_]),
  ('retain', _anondynamic26, []),
  ('release', None, []),
  ('autorelease', _anondynamic26, []),
  ('retainCount', NSUInteger, []),
  ('zone', ctypes.POINTER(struct__NSZone), []),
  ('hash', NSUInteger, []),
  ('description', NSString, []),
  ('debugDescription', NSString, []),
]
# __attribute__((visibility("default"))) extern void MTLRemoveDeviceObserver(id<NSObject>  _Nonnull observer) __attribute__((availability(macos, introduced=10.13))) __attribute__((availability(ios, unavailable)))
try: (MTLRemoveDeviceObserver:=dll.MTLRemoveDeviceObserver).restype, MTLRemoveDeviceObserver.argtypes = None, [_anondynamic26]
except AttributeError: pass

class MTLArgumentDescriptor(NSObject): pass
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
enum_MTLIndirectCommandType = CEnum(NSUInteger)
MTLIndirectCommandTypeDraw = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDraw', 1)
MTLIndirectCommandTypeDrawIndexed = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawIndexed', 2)
MTLIndirectCommandTypeDrawPatches = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawPatches', 4)
MTLIndirectCommandTypeDrawIndexedPatches = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawIndexedPatches', 8)
MTLIndirectCommandTypeConcurrentDispatch = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeConcurrentDispatch', 32)
MTLIndirectCommandTypeConcurrentDispatchThreads = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeConcurrentDispatchThreads', 64)

MTLIndirectCommandType = enum_MTLIndirectCommandType
class MTLIndirectCommandBufferExecutionRange(Struct): pass
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
