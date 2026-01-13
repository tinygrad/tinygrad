# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
from tinygrad.runtime.support import objc
dll = c.DLL('metal', 'Metal')
@c.record
class MTLDispatchThreadgroupsIndirectArguments(c.Struct):
  SIZE = 12
  threadgroupsPerGrid: Annotated[c.Array[uint32_t, Literal[3]], 0]
uint32_t = Annotated[int, ctypes.c_uint32]
@c.record
class MTLStageInRegionIndirectArguments(c.Struct):
  SIZE = 24
  stageInOrigin: Annotated[c.Array[uint32_t, Literal[3]], 0]
  stageInSize: Annotated[c.Array[uint32_t, Literal[3]], 12]
class MTLComputeCommandEncoder(objc.Spec): pass
class MTLCommandEncoder(objc.Spec): pass
class MTLComputePipelineState(objc.Spec): pass
NSUInteger = Annotated[int, ctypes.c_uint64]
class MTLBuffer(objc.Spec): pass
class MTLResource(objc.Spec): pass
@c.record
class struct__NSRange(c.Struct):
  SIZE = 16
  location: Annotated[NSUInteger, 0]
  length: Annotated[NSUInteger, 8]
NSRange: TypeAlias = struct__NSRange
class MTLTexture(objc.Spec): pass
class MTLTextureDescriptor(objc.Spec): pass
enum_MTLTextureType = CEnum(NSUInteger)
MTLTextureType1D = enum_MTLTextureType.define('MTLTextureType1D', 0) # type: ignore
MTLTextureType1DArray = enum_MTLTextureType.define('MTLTextureType1DArray', 1) # type: ignore
MTLTextureType2D = enum_MTLTextureType.define('MTLTextureType2D', 2) # type: ignore
MTLTextureType2DArray = enum_MTLTextureType.define('MTLTextureType2DArray', 3) # type: ignore
MTLTextureType2DMultisample = enum_MTLTextureType.define('MTLTextureType2DMultisample', 4) # type: ignore
MTLTextureTypeCube = enum_MTLTextureType.define('MTLTextureTypeCube', 5) # type: ignore
MTLTextureTypeCubeArray = enum_MTLTextureType.define('MTLTextureTypeCubeArray', 6) # type: ignore
MTLTextureType3D = enum_MTLTextureType.define('MTLTextureType3D', 7) # type: ignore
MTLTextureType2DMultisampleArray = enum_MTLTextureType.define('MTLTextureType2DMultisampleArray', 8) # type: ignore
MTLTextureTypeTextureBuffer = enum_MTLTextureType.define('MTLTextureTypeTextureBuffer', 9) # type: ignore

MTLTextureType: TypeAlias = enum_MTLTextureType
enum_MTLPixelFormat = CEnum(NSUInteger)
MTLPixelFormatInvalid = enum_MTLPixelFormat.define('MTLPixelFormatInvalid', 0) # type: ignore
MTLPixelFormatA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatA8Unorm', 1) # type: ignore
MTLPixelFormatR8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatR8Unorm', 10) # type: ignore
MTLPixelFormatR8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatR8Unorm_sRGB', 11) # type: ignore
MTLPixelFormatR8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatR8Snorm', 12) # type: ignore
MTLPixelFormatR8Uint = enum_MTLPixelFormat.define('MTLPixelFormatR8Uint', 13) # type: ignore
MTLPixelFormatR8Sint = enum_MTLPixelFormat.define('MTLPixelFormatR8Sint', 14) # type: ignore
MTLPixelFormatR16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatR16Unorm', 20) # type: ignore
MTLPixelFormatR16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatR16Snorm', 22) # type: ignore
MTLPixelFormatR16Uint = enum_MTLPixelFormat.define('MTLPixelFormatR16Uint', 23) # type: ignore
MTLPixelFormatR16Sint = enum_MTLPixelFormat.define('MTLPixelFormatR16Sint', 24) # type: ignore
MTLPixelFormatR16Float = enum_MTLPixelFormat.define('MTLPixelFormatR16Float', 25) # type: ignore
MTLPixelFormatRG8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRG8Unorm', 30) # type: ignore
MTLPixelFormatRG8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatRG8Unorm_sRGB', 31) # type: ignore
MTLPixelFormatRG8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRG8Snorm', 32) # type: ignore
MTLPixelFormatRG8Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG8Uint', 33) # type: ignore
MTLPixelFormatRG8Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG8Sint', 34) # type: ignore
MTLPixelFormatB5G6R5Unorm = enum_MTLPixelFormat.define('MTLPixelFormatB5G6R5Unorm', 40) # type: ignore
MTLPixelFormatA1BGR5Unorm = enum_MTLPixelFormat.define('MTLPixelFormatA1BGR5Unorm', 41) # type: ignore
MTLPixelFormatABGR4Unorm = enum_MTLPixelFormat.define('MTLPixelFormatABGR4Unorm', 42) # type: ignore
MTLPixelFormatBGR5A1Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGR5A1Unorm', 43) # type: ignore
MTLPixelFormatR32Uint = enum_MTLPixelFormat.define('MTLPixelFormatR32Uint', 53) # type: ignore
MTLPixelFormatR32Sint = enum_MTLPixelFormat.define('MTLPixelFormatR32Sint', 54) # type: ignore
MTLPixelFormatR32Float = enum_MTLPixelFormat.define('MTLPixelFormatR32Float', 55) # type: ignore
MTLPixelFormatRG16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRG16Unorm', 60) # type: ignore
MTLPixelFormatRG16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRG16Snorm', 62) # type: ignore
MTLPixelFormatRG16Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG16Uint', 63) # type: ignore
MTLPixelFormatRG16Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG16Sint', 64) # type: ignore
MTLPixelFormatRG16Float = enum_MTLPixelFormat.define('MTLPixelFormatRG16Float', 65) # type: ignore
MTLPixelFormatRGBA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Unorm', 70) # type: ignore
MTLPixelFormatRGBA8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Unorm_sRGB', 71) # type: ignore
MTLPixelFormatRGBA8Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Snorm', 72) # type: ignore
MTLPixelFormatRGBA8Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Uint', 73) # type: ignore
MTLPixelFormatRGBA8Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA8Sint', 74) # type: ignore
MTLPixelFormatBGRA8Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGRA8Unorm', 80) # type: ignore
MTLPixelFormatBGRA8Unorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGRA8Unorm_sRGB', 81) # type: ignore
MTLPixelFormatRGB10A2Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGB10A2Unorm', 90) # type: ignore
MTLPixelFormatRGB10A2Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGB10A2Uint', 91) # type: ignore
MTLPixelFormatRG11B10Float = enum_MTLPixelFormat.define('MTLPixelFormatRG11B10Float', 92) # type: ignore
MTLPixelFormatRGB9E5Float = enum_MTLPixelFormat.define('MTLPixelFormatRGB9E5Float', 93) # type: ignore
MTLPixelFormatBGR10A2Unorm = enum_MTLPixelFormat.define('MTLPixelFormatBGR10A2Unorm', 94) # type: ignore
MTLPixelFormatBGR10_XR = enum_MTLPixelFormat.define('MTLPixelFormatBGR10_XR', 554) # type: ignore
MTLPixelFormatBGR10_XR_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGR10_XR_sRGB', 555) # type: ignore
MTLPixelFormatRG32Uint = enum_MTLPixelFormat.define('MTLPixelFormatRG32Uint', 103) # type: ignore
MTLPixelFormatRG32Sint = enum_MTLPixelFormat.define('MTLPixelFormatRG32Sint', 104) # type: ignore
MTLPixelFormatRG32Float = enum_MTLPixelFormat.define('MTLPixelFormatRG32Float', 105) # type: ignore
MTLPixelFormatRGBA16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Unorm', 110) # type: ignore
MTLPixelFormatRGBA16Snorm = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Snorm', 112) # type: ignore
MTLPixelFormatRGBA16Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Uint', 113) # type: ignore
MTLPixelFormatRGBA16Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Sint', 114) # type: ignore
MTLPixelFormatRGBA16Float = enum_MTLPixelFormat.define('MTLPixelFormatRGBA16Float', 115) # type: ignore
MTLPixelFormatBGRA10_XR = enum_MTLPixelFormat.define('MTLPixelFormatBGRA10_XR', 552) # type: ignore
MTLPixelFormatBGRA10_XR_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBGRA10_XR_sRGB', 553) # type: ignore
MTLPixelFormatRGBA32Uint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Uint', 123) # type: ignore
MTLPixelFormatRGBA32Sint = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Sint', 124) # type: ignore
MTLPixelFormatRGBA32Float = enum_MTLPixelFormat.define('MTLPixelFormatRGBA32Float', 125) # type: ignore
MTLPixelFormatBC1_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC1_RGBA', 130) # type: ignore
MTLPixelFormatBC1_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC1_RGBA_sRGB', 131) # type: ignore
MTLPixelFormatBC2_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC2_RGBA', 132) # type: ignore
MTLPixelFormatBC2_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC2_RGBA_sRGB', 133) # type: ignore
MTLPixelFormatBC3_RGBA = enum_MTLPixelFormat.define('MTLPixelFormatBC3_RGBA', 134) # type: ignore
MTLPixelFormatBC3_RGBA_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC3_RGBA_sRGB', 135) # type: ignore
MTLPixelFormatBC4_RUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC4_RUnorm', 140) # type: ignore
MTLPixelFormatBC4_RSnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC4_RSnorm', 141) # type: ignore
MTLPixelFormatBC5_RGUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC5_RGUnorm', 142) # type: ignore
MTLPixelFormatBC5_RGSnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC5_RGSnorm', 143) # type: ignore
MTLPixelFormatBC6H_RGBFloat = enum_MTLPixelFormat.define('MTLPixelFormatBC6H_RGBFloat', 150) # type: ignore
MTLPixelFormatBC6H_RGBUfloat = enum_MTLPixelFormat.define('MTLPixelFormatBC6H_RGBUfloat', 151) # type: ignore
MTLPixelFormatBC7_RGBAUnorm = enum_MTLPixelFormat.define('MTLPixelFormatBC7_RGBAUnorm', 152) # type: ignore
MTLPixelFormatBC7_RGBAUnorm_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatBC7_RGBAUnorm_sRGB', 153) # type: ignore
MTLPixelFormatPVRTC_RGB_2BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_2BPP', 160) # type: ignore
MTLPixelFormatPVRTC_RGB_2BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_2BPP_sRGB', 161) # type: ignore
MTLPixelFormatPVRTC_RGB_4BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_4BPP', 162) # type: ignore
MTLPixelFormatPVRTC_RGB_4BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGB_4BPP_sRGB', 163) # type: ignore
MTLPixelFormatPVRTC_RGBA_2BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_2BPP', 164) # type: ignore
MTLPixelFormatPVRTC_RGBA_2BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_2BPP_sRGB', 165) # type: ignore
MTLPixelFormatPVRTC_RGBA_4BPP = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_4BPP', 166) # type: ignore
MTLPixelFormatPVRTC_RGBA_4BPP_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatPVRTC_RGBA_4BPP_sRGB', 167) # type: ignore
MTLPixelFormatEAC_R11Unorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_R11Unorm', 170) # type: ignore
MTLPixelFormatEAC_R11Snorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_R11Snorm', 172) # type: ignore
MTLPixelFormatEAC_RG11Unorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RG11Unorm', 174) # type: ignore
MTLPixelFormatEAC_RG11Snorm = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RG11Snorm', 176) # type: ignore
MTLPixelFormatEAC_RGBA8 = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RGBA8', 178) # type: ignore
MTLPixelFormatEAC_RGBA8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatEAC_RGBA8_sRGB', 179) # type: ignore
MTLPixelFormatETC2_RGB8 = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8', 180) # type: ignore
MTLPixelFormatETC2_RGB8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8_sRGB', 181) # type: ignore
MTLPixelFormatETC2_RGB8A1 = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8A1', 182) # type: ignore
MTLPixelFormatETC2_RGB8A1_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatETC2_RGB8A1_sRGB', 183) # type: ignore
MTLPixelFormatASTC_4x4_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_sRGB', 186) # type: ignore
MTLPixelFormatASTC_5x4_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_sRGB', 187) # type: ignore
MTLPixelFormatASTC_5x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_sRGB', 188) # type: ignore
MTLPixelFormatASTC_6x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_sRGB', 189) # type: ignore
MTLPixelFormatASTC_6x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_sRGB', 190) # type: ignore
MTLPixelFormatASTC_8x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_sRGB', 192) # type: ignore
MTLPixelFormatASTC_8x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_sRGB', 193) # type: ignore
MTLPixelFormatASTC_8x8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_sRGB', 194) # type: ignore
MTLPixelFormatASTC_10x5_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_sRGB', 195) # type: ignore
MTLPixelFormatASTC_10x6_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_sRGB', 196) # type: ignore
MTLPixelFormatASTC_10x8_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_sRGB', 197) # type: ignore
MTLPixelFormatASTC_10x10_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_sRGB', 198) # type: ignore
MTLPixelFormatASTC_12x10_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_sRGB', 199) # type: ignore
MTLPixelFormatASTC_12x12_sRGB = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_sRGB', 200) # type: ignore
MTLPixelFormatASTC_4x4_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_LDR', 204) # type: ignore
MTLPixelFormatASTC_5x4_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_LDR', 205) # type: ignore
MTLPixelFormatASTC_5x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_LDR', 206) # type: ignore
MTLPixelFormatASTC_6x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_LDR', 207) # type: ignore
MTLPixelFormatASTC_6x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_LDR', 208) # type: ignore
MTLPixelFormatASTC_8x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_LDR', 210) # type: ignore
MTLPixelFormatASTC_8x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_LDR', 211) # type: ignore
MTLPixelFormatASTC_8x8_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_LDR', 212) # type: ignore
MTLPixelFormatASTC_10x5_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_LDR', 213) # type: ignore
MTLPixelFormatASTC_10x6_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_LDR', 214) # type: ignore
MTLPixelFormatASTC_10x8_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_LDR', 215) # type: ignore
MTLPixelFormatASTC_10x10_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_LDR', 216) # type: ignore
MTLPixelFormatASTC_12x10_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_LDR', 217) # type: ignore
MTLPixelFormatASTC_12x12_LDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_LDR', 218) # type: ignore
MTLPixelFormatASTC_4x4_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_4x4_HDR', 222) # type: ignore
MTLPixelFormatASTC_5x4_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x4_HDR', 223) # type: ignore
MTLPixelFormatASTC_5x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_5x5_HDR', 224) # type: ignore
MTLPixelFormatASTC_6x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x5_HDR', 225) # type: ignore
MTLPixelFormatASTC_6x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_6x6_HDR', 226) # type: ignore
MTLPixelFormatASTC_8x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x5_HDR', 228) # type: ignore
MTLPixelFormatASTC_8x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x6_HDR', 229) # type: ignore
MTLPixelFormatASTC_8x8_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_8x8_HDR', 230) # type: ignore
MTLPixelFormatASTC_10x5_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x5_HDR', 231) # type: ignore
MTLPixelFormatASTC_10x6_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x6_HDR', 232) # type: ignore
MTLPixelFormatASTC_10x8_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x8_HDR', 233) # type: ignore
MTLPixelFormatASTC_10x10_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_10x10_HDR', 234) # type: ignore
MTLPixelFormatASTC_12x10_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x10_HDR', 235) # type: ignore
MTLPixelFormatASTC_12x12_HDR = enum_MTLPixelFormat.define('MTLPixelFormatASTC_12x12_HDR', 236) # type: ignore
MTLPixelFormatGBGR422 = enum_MTLPixelFormat.define('MTLPixelFormatGBGR422', 240) # type: ignore
MTLPixelFormatBGRG422 = enum_MTLPixelFormat.define('MTLPixelFormatBGRG422', 241) # type: ignore
MTLPixelFormatDepth16Unorm = enum_MTLPixelFormat.define('MTLPixelFormatDepth16Unorm', 250) # type: ignore
MTLPixelFormatDepth32Float = enum_MTLPixelFormat.define('MTLPixelFormatDepth32Float', 252) # type: ignore
MTLPixelFormatStencil8 = enum_MTLPixelFormat.define('MTLPixelFormatStencil8', 253) # type: ignore
MTLPixelFormatDepth24Unorm_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatDepth24Unorm_Stencil8', 255) # type: ignore
MTLPixelFormatDepth32Float_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatDepth32Float_Stencil8', 260) # type: ignore
MTLPixelFormatX32_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatX32_Stencil8', 261) # type: ignore
MTLPixelFormatX24_Stencil8 = enum_MTLPixelFormat.define('MTLPixelFormatX24_Stencil8', 262) # type: ignore

MTLPixelFormat: TypeAlias = enum_MTLPixelFormat
enum_MTLResourceOptions = CEnum(NSUInteger)
MTLResourceCPUCacheModeDefaultCache = enum_MTLResourceOptions.define('MTLResourceCPUCacheModeDefaultCache', 0) # type: ignore
MTLResourceCPUCacheModeWriteCombined = enum_MTLResourceOptions.define('MTLResourceCPUCacheModeWriteCombined', 1) # type: ignore
MTLResourceStorageModeShared = enum_MTLResourceOptions.define('MTLResourceStorageModeShared', 0) # type: ignore
MTLResourceStorageModeManaged = enum_MTLResourceOptions.define('MTLResourceStorageModeManaged', 16) # type: ignore
MTLResourceStorageModePrivate = enum_MTLResourceOptions.define('MTLResourceStorageModePrivate', 32) # type: ignore
MTLResourceStorageModeMemoryless = enum_MTLResourceOptions.define('MTLResourceStorageModeMemoryless', 48) # type: ignore
MTLResourceHazardTrackingModeDefault = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeDefault', 0) # type: ignore
MTLResourceHazardTrackingModeUntracked = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeUntracked', 256) # type: ignore
MTLResourceHazardTrackingModeTracked = enum_MTLResourceOptions.define('MTLResourceHazardTrackingModeTracked', 512) # type: ignore
MTLResourceOptionCPUCacheModeDefault = enum_MTLResourceOptions.define('MTLResourceOptionCPUCacheModeDefault', 0) # type: ignore
MTLResourceOptionCPUCacheModeWriteCombined = enum_MTLResourceOptions.define('MTLResourceOptionCPUCacheModeWriteCombined', 1) # type: ignore

MTLResourceOptions: TypeAlias = enum_MTLResourceOptions
enum_MTLCPUCacheMode = CEnum(NSUInteger)
MTLCPUCacheModeDefaultCache = enum_MTLCPUCacheMode.define('MTLCPUCacheModeDefaultCache', 0) # type: ignore
MTLCPUCacheModeWriteCombined = enum_MTLCPUCacheMode.define('MTLCPUCacheModeWriteCombined', 1) # type: ignore

MTLCPUCacheMode: TypeAlias = enum_MTLCPUCacheMode
enum_MTLStorageMode = CEnum(NSUInteger)
MTLStorageModeShared = enum_MTLStorageMode.define('MTLStorageModeShared', 0) # type: ignore
MTLStorageModeManaged = enum_MTLStorageMode.define('MTLStorageModeManaged', 1) # type: ignore
MTLStorageModePrivate = enum_MTLStorageMode.define('MTLStorageModePrivate', 2) # type: ignore
MTLStorageModeMemoryless = enum_MTLStorageMode.define('MTLStorageModeMemoryless', 3) # type: ignore

MTLStorageMode: TypeAlias = enum_MTLStorageMode
enum_MTLHazardTrackingMode = CEnum(NSUInteger)
MTLHazardTrackingModeDefault = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeDefault', 0) # type: ignore
MTLHazardTrackingModeUntracked = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeUntracked', 1) # type: ignore
MTLHazardTrackingModeTracked = enum_MTLHazardTrackingMode.define('MTLHazardTrackingModeTracked', 2) # type: ignore

MTLHazardTrackingMode: TypeAlias = enum_MTLHazardTrackingMode
enum_MTLTextureUsage = CEnum(NSUInteger)
MTLTextureUsageUnknown = enum_MTLTextureUsage.define('MTLTextureUsageUnknown', 0) # type: ignore
MTLTextureUsageShaderRead = enum_MTLTextureUsage.define('MTLTextureUsageShaderRead', 1) # type: ignore
MTLTextureUsageShaderWrite = enum_MTLTextureUsage.define('MTLTextureUsageShaderWrite', 2) # type: ignore
MTLTextureUsageRenderTarget = enum_MTLTextureUsage.define('MTLTextureUsageRenderTarget', 4) # type: ignore
MTLTextureUsagePixelFormatView = enum_MTLTextureUsage.define('MTLTextureUsagePixelFormatView', 16) # type: ignore
MTLTextureUsageShaderAtomic = enum_MTLTextureUsage.define('MTLTextureUsageShaderAtomic', 32) # type: ignore

MTLTextureUsage: TypeAlias = enum_MTLTextureUsage
BOOL = Annotated[int, ctypes.c_int32]
NSInteger = Annotated[int, ctypes.c_int64]
enum_MTLTextureCompressionType = CEnum(NSInteger)
MTLTextureCompressionTypeLossless = enum_MTLTextureCompressionType.define('MTLTextureCompressionTypeLossless', 0) # type: ignore
MTLTextureCompressionTypeLossy = enum_MTLTextureCompressionType.define('MTLTextureCompressionTypeLossy', 1) # type: ignore

MTLTextureCompressionType: TypeAlias = enum_MTLTextureCompressionType
@c.record
class MTLTextureSwizzleChannels(c.Struct):
  SIZE = 4
  red: Annotated[MTLTextureSwizzle, 0]
  green: Annotated[MTLTextureSwizzle, 1]
  blue: Annotated[MTLTextureSwizzle, 2]
  alpha: Annotated[MTLTextureSwizzle, 3]
uint8_t = Annotated[int, ctypes.c_ubyte]
enum_MTLTextureSwizzle = CEnum(uint8_t)
MTLTextureSwizzleZero = enum_MTLTextureSwizzle.define('MTLTextureSwizzleZero', 0) # type: ignore
MTLTextureSwizzleOne = enum_MTLTextureSwizzle.define('MTLTextureSwizzleOne', 1) # type: ignore
MTLTextureSwizzleRed = enum_MTLTextureSwizzle.define('MTLTextureSwizzleRed', 2) # type: ignore
MTLTextureSwizzleGreen = enum_MTLTextureSwizzle.define('MTLTextureSwizzleGreen', 3) # type: ignore
MTLTextureSwizzleBlue = enum_MTLTextureSwizzle.define('MTLTextureSwizzleBlue', 4) # type: ignore
MTLTextureSwizzleAlpha = enum_MTLTextureSwizzle.define('MTLTextureSwizzleAlpha', 5) # type: ignore

MTLTextureSwizzle: TypeAlias = enum_MTLTextureSwizzle
class NSObject(objc.Spec): pass
IMP = c.CFUNCTYPE(None, )
class NSInvocation(objc.Spec): pass
class NSMethodSignature(objc.Spec): pass
NSMethodSignature._bases_ = [NSObject]
NSMethodSignature._methods_ = [
  ('getArgumentTypeAtIndex:', c.POINTER[Annotated[bytes, ctypes.c_char]], [NSUInteger]),
  ('isOneway', BOOL, []),
  ('numberOfArguments', NSUInteger, []),
  ('frameLength', NSUInteger, []),
  ('methodReturnType', c.POINTER[Annotated[bytes, ctypes.c_char]], []),
  ('methodReturnLength', NSUInteger, []),
]
NSMethodSignature._classmethods_ = [
  ('signatureWithObjCTypes:', NSMethodSignature, [c.POINTER[Annotated[bytes, ctypes.c_char]]]),
]
NSInvocation._bases_ = [NSObject]
NSInvocation._methods_ = [
  ('retainArguments', None, []),
  ('getReturnValue:', None, [c.POINTER[None]]),
  ('setReturnValue:', None, [c.POINTER[None]]),
  ('getArgument:atIndex:', None, [c.POINTER[None], NSInteger]),
  ('setArgument:atIndex:', None, [c.POINTER[None], NSInteger]),
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
class struct__NSZone(ctypes.Structure): pass
class Protocol(objc.Spec): pass
class NSString(objc.Spec): pass
unichar = Annotated[int, ctypes.c_uint16]
class NSCoder(objc.Spec): pass
class NSData(objc.Spec): pass
NSData._bases_ = [NSObject]
NSData._methods_ = [
  ('length', NSUInteger, []),
  ('bytes', c.POINTER[None], []),
]
NSCoder._bases_ = [NSObject]
NSCoder._methods_ = [
  ('encodeValueOfObjCType:at:', None, [c.POINTER[Annotated[bytes, ctypes.c_char]], c.POINTER[None]]),
  ('encodeDataObject:', None, [NSData]),
  ('decodeDataObject', NSData, []),
  ('decodeValueOfObjCType:at:size:', None, [c.POINTER[Annotated[bytes, ctypes.c_char]], c.POINTER[None], NSUInteger]),
  ('versionForClassName:', NSInteger, [NSString]),
]
NSString._bases_ = [NSObject]
NSString._methods_ = [
  ('characterAtIndex:', unichar, [NSUInteger]),
  ('init', 'instancetype', []),
  ('initWithCoder:', 'instancetype', [NSCoder]),
  ('length', NSUInteger, []),
]
NSObject._methods_ = [
  ('init', 'instancetype', []),
  ('dealloc', None, []),
  ('finalize', None, []),
  ('copy', objc.id_, [], True),
  ('mutableCopy', objc.id_, [], True),
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
  ('new', 'instancetype', [], True),
  ('allocWithZone:', 'instancetype', [c.POINTER[struct__NSZone]], True),
  ('alloc', 'instancetype', [], True),
  ('copyWithZone:', objc.id_, [c.POINTER[struct__NSZone]], True),
  ('mutableCopyWithZone:', objc.id_, [c.POINTER[struct__NSZone]], True),
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
MTLTextureDescriptor._bases_ = [NSObject]
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
class MTLDevice(objc.Spec): pass
uint64_t = Annotated[int, ctypes.c_uint64]
MTLBuffer._bases_ = [MTLResource]
MTLBuffer._methods_ = [
  ('contents', c.POINTER[None], []),
  ('didModifyRange:', None, [NSRange]),
  ('newTextureWithDescriptor:offset:bytesPerRow:', MTLTexture, [MTLTextureDescriptor, NSUInteger, NSUInteger], True),
  ('addDebugMarker:range:', None, [NSString, NSRange]),
  ('removeAllDebugMarkers', None, []),
  ('newRemoteBufferViewForDevice:', MTLBuffer, [MTLDevice], True),
  ('length', NSUInteger, []),
  ('remoteStorageBuffer', MTLBuffer, []),
  ('gpuAddress', uint64_t, []),
]
class MTLVisibleFunctionTable(objc.Spec): pass
class MTLIntersectionFunctionTable(objc.Spec): pass
class MTLAccelerationStructure(objc.Spec): pass
class MTLSamplerState(objc.Spec): pass
@c.record
class MTLRegion(c.Struct):
  SIZE = 48
  origin: Annotated[MTLOrigin, 0]
  size: Annotated[MTLSize, 24]
@c.record
class MTLOrigin(c.Struct):
  SIZE = 24
  x: Annotated[NSUInteger, 0]
  y: Annotated[NSUInteger, 8]
  z: Annotated[NSUInteger, 16]
@c.record
class MTLSize(c.Struct):
  SIZE = 24
  width: Annotated[NSUInteger, 0]
  height: Annotated[NSUInteger, 8]
  depth: Annotated[NSUInteger, 16]
class MTLFence(objc.Spec): pass
MTLFence._bases_ = [NSObject]
MTLFence._methods_ = [
  ('device', MTLDevice, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
]
enum_MTLPurgeableState = CEnum(NSUInteger)
MTLPurgeableStateKeepCurrent = enum_MTLPurgeableState.define('MTLPurgeableStateKeepCurrent', 1) # type: ignore
MTLPurgeableStateNonVolatile = enum_MTLPurgeableState.define('MTLPurgeableStateNonVolatile', 2) # type: ignore
MTLPurgeableStateVolatile = enum_MTLPurgeableState.define('MTLPurgeableStateVolatile', 3) # type: ignore
MTLPurgeableStateEmpty = enum_MTLPurgeableState.define('MTLPurgeableStateEmpty', 4) # type: ignore

MTLPurgeableState: TypeAlias = enum_MTLPurgeableState
kern_return_t = Annotated[int, ctypes.c_int32]
task_id_token_t = Annotated[int, ctypes.c_uint32]
class MTLHeap(objc.Spec): pass
MTLResource._bases_ = [NSObject]
MTLResource._methods_ = [
  ('setPurgeableState:', MTLPurgeableState, [MTLPurgeableState]),
  ('makeAliasable', None, []),
  ('isAliasable', BOOL, []),
  ('setOwnerWithIdentity:', kern_return_t, [task_id_token_t]),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
  ('cpuCacheMode', MTLCPUCacheMode, []),
  ('storageMode', MTLStorageMode, []),
  ('hazardTrackingMode', MTLHazardTrackingMode, []),
  ('resourceOptions', MTLResourceOptions, []),
  ('heap', MTLHeap, []),
  ('heapOffset', NSUInteger, []),
  ('allocatedSize', NSUInteger, [], True),
]
enum_MTLResourceUsage = CEnum(NSUInteger)
MTLResourceUsageRead = enum_MTLResourceUsage.define('MTLResourceUsageRead', 1) # type: ignore
MTLResourceUsageWrite = enum_MTLResourceUsage.define('MTLResourceUsageWrite', 2) # type: ignore
MTLResourceUsageSample = enum_MTLResourceUsage.define('MTLResourceUsageSample', 4) # type: ignore

MTLResourceUsage: TypeAlias = enum_MTLResourceUsage
class MTLIndirectCommandBuffer(objc.Spec): pass
enum_MTLBarrierScope = CEnum(NSUInteger)
MTLBarrierScopeBuffers = enum_MTLBarrierScope.define('MTLBarrierScopeBuffers', 1) # type: ignore
MTLBarrierScopeTextures = enum_MTLBarrierScope.define('MTLBarrierScopeTextures', 2) # type: ignore
MTLBarrierScopeRenderTargets = enum_MTLBarrierScope.define('MTLBarrierScopeRenderTargets', 4) # type: ignore

MTLBarrierScope: TypeAlias = enum_MTLBarrierScope
class MTLCounterSampleBuffer(objc.Spec): pass
MTLCounterSampleBuffer._bases_ = [NSObject]
MTLCounterSampleBuffer._methods_ = [
  ('resolveCounterRange:', NSData, [NSRange]),
  ('device', MTLDevice, []),
  ('label', NSString, []),
  ('sampleCount', NSUInteger, []),
]
enum_MTLDispatchType = CEnum(NSUInteger)
MTLDispatchTypeSerial = enum_MTLDispatchType.define('MTLDispatchTypeSerial', 0) # type: ignore
MTLDispatchTypeConcurrent = enum_MTLDispatchType.define('MTLDispatchTypeConcurrent', 1) # type: ignore

MTLDispatchType: TypeAlias = enum_MTLDispatchType
MTLComputeCommandEncoder._bases_ = [MTLCommandEncoder]
MTLComputeCommandEncoder._methods_ = [
  ('setComputePipelineState:', None, [MTLComputePipelineState]),
  ('setBytes:length:atIndex:', None, [c.POINTER[None], NSUInteger, NSUInteger]),
  ('setBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setBufferOffset:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setBuffers:offsets:withRange:', None, [c.POINTER[MTLBuffer], c.POINTER[NSUInteger], NSRange]),
  ('setBuffer:offset:attributeStride:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('setBuffers:offsets:attributeStrides:withRange:', None, [c.POINTER[MTLBuffer], c.POINTER[NSUInteger], c.POINTER[NSUInteger], NSRange]),
  ('setBufferOffset:attributeStride:atIndex:', None, [NSUInteger, NSUInteger, NSUInteger]),
  ('setBytes:length:attributeStride:atIndex:', None, [c.POINTER[None], NSUInteger, NSUInteger, NSUInteger]),
  ('setVisibleFunctionTable:atBufferIndex:', None, [MTLVisibleFunctionTable, NSUInteger]),
  ('setVisibleFunctionTables:withBufferRange:', None, [c.POINTER[MTLVisibleFunctionTable], NSRange]),
  ('setIntersectionFunctionTable:atBufferIndex:', None, [MTLIntersectionFunctionTable, NSUInteger]),
  ('setIntersectionFunctionTables:withBufferRange:', None, [c.POINTER[MTLIntersectionFunctionTable], NSRange]),
  ('setAccelerationStructure:atBufferIndex:', None, [MTLAccelerationStructure, NSUInteger]),
  ('setTexture:atIndex:', None, [MTLTexture, NSUInteger]),
  ('setTextures:withRange:', None, [c.POINTER[MTLTexture], NSRange]),
  ('setSamplerState:atIndex:', None, [MTLSamplerState, NSUInteger]),
  ('setSamplerStates:withRange:', None, [c.POINTER[MTLSamplerState], NSRange]),
  ('setSamplerState:lodMinClamp:lodMaxClamp:atIndex:', None, [MTLSamplerState, Annotated[float, ctypes.c_float], Annotated[float, ctypes.c_float], NSUInteger]),
  ('setSamplerStates:lodMinClamps:lodMaxClamps:withRange:', None, [c.POINTER[MTLSamplerState], c.POINTER[Annotated[float, ctypes.c_float]], c.POINTER[Annotated[float, ctypes.c_float]], NSRange]),
  ('setThreadgroupMemoryLength:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setImageblockWidth:height:', None, [NSUInteger, NSUInteger]),
  ('setStageInRegion:', None, [MTLRegion]),
  ('setStageInRegionWithIndirectBuffer:indirectBufferOffset:', None, [MTLBuffer, NSUInteger]),
  ('dispatchThreadgroups:threadsPerThreadgroup:', None, [MTLSize, MTLSize]),
  ('dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:', None, [MTLBuffer, NSUInteger, MTLSize]),
  ('dispatchThreads:threadsPerThreadgroup:', None, [MTLSize, MTLSize]),
  ('updateFence:', None, [MTLFence]),
  ('waitForFence:', None, [MTLFence]),
  ('useResource:usage:', None, [MTLResource, MTLResourceUsage]),
  ('useResources:count:usage:', None, [c.POINTER[MTLResource], NSUInteger, MTLResourceUsage]),
  ('useHeap:', None, [MTLHeap]),
  ('useHeaps:count:', None, [c.POINTER[MTLHeap], NSUInteger]),
  ('executeCommandsInBuffer:withRange:', None, [MTLIndirectCommandBuffer, NSRange]),
  ('executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:', None, [MTLIndirectCommandBuffer, MTLBuffer, NSUInteger]),
  ('memoryBarrierWithScope:', None, [MTLBarrierScope]),
  ('memoryBarrierWithResources:count:', None, [c.POINTER[MTLResource], NSUInteger]),
  ('sampleCountersInBuffer:atSampleIndex:withBarrier:', None, [MTLCounterSampleBuffer, NSUInteger, BOOL]),
  ('dispatchType', MTLDispatchType, []),
]
class MTLComputePipelineReflection(objc.Spec): pass
MTLComputePipelineReflection._bases_ = [NSObject]
class MTLComputePipelineDescriptor(objc.Spec): pass
class MTLFunction(objc.Spec): pass
class MTLArgumentEncoder(objc.Spec): pass
class MTLArgument(objc.Spec): pass
enum_MTLArgumentType = CEnum(NSUInteger)
MTLArgumentTypeBuffer = enum_MTLArgumentType.define('MTLArgumentTypeBuffer', 0) # type: ignore
MTLArgumentTypeThreadgroupMemory = enum_MTLArgumentType.define('MTLArgumentTypeThreadgroupMemory', 1) # type: ignore
MTLArgumentTypeTexture = enum_MTLArgumentType.define('MTLArgumentTypeTexture', 2) # type: ignore
MTLArgumentTypeSampler = enum_MTLArgumentType.define('MTLArgumentTypeSampler', 3) # type: ignore
MTLArgumentTypeImageblockData = enum_MTLArgumentType.define('MTLArgumentTypeImageblockData', 16) # type: ignore
MTLArgumentTypeImageblock = enum_MTLArgumentType.define('MTLArgumentTypeImageblock', 17) # type: ignore
MTLArgumentTypeVisibleFunctionTable = enum_MTLArgumentType.define('MTLArgumentTypeVisibleFunctionTable', 24) # type: ignore
MTLArgumentTypePrimitiveAccelerationStructure = enum_MTLArgumentType.define('MTLArgumentTypePrimitiveAccelerationStructure', 25) # type: ignore
MTLArgumentTypeInstanceAccelerationStructure = enum_MTLArgumentType.define('MTLArgumentTypeInstanceAccelerationStructure', 26) # type: ignore
MTLArgumentTypeIntersectionFunctionTable = enum_MTLArgumentType.define('MTLArgumentTypeIntersectionFunctionTable', 27) # type: ignore

MTLArgumentType: TypeAlias = enum_MTLArgumentType
enum_MTLBindingAccess = CEnum(NSUInteger)
MTLBindingAccessReadOnly = enum_MTLBindingAccess.define('MTLBindingAccessReadOnly', 0) # type: ignore
MTLBindingAccessReadWrite = enum_MTLBindingAccess.define('MTLBindingAccessReadWrite', 1) # type: ignore
MTLBindingAccessWriteOnly = enum_MTLBindingAccess.define('MTLBindingAccessWriteOnly', 2) # type: ignore
MTLArgumentAccessReadOnly = enum_MTLBindingAccess.define('MTLArgumentAccessReadOnly', 0) # type: ignore
MTLArgumentAccessReadWrite = enum_MTLBindingAccess.define('MTLArgumentAccessReadWrite', 1) # type: ignore
MTLArgumentAccessWriteOnly = enum_MTLBindingAccess.define('MTLArgumentAccessWriteOnly', 2) # type: ignore

MTLBindingAccess: TypeAlias = enum_MTLBindingAccess
enum_MTLDataType = CEnum(NSUInteger)
MTLDataTypeNone = enum_MTLDataType.define('MTLDataTypeNone', 0) # type: ignore
MTLDataTypeStruct = enum_MTLDataType.define('MTLDataTypeStruct', 1) # type: ignore
MTLDataTypeArray = enum_MTLDataType.define('MTLDataTypeArray', 2) # type: ignore
MTLDataTypeFloat = enum_MTLDataType.define('MTLDataTypeFloat', 3) # type: ignore
MTLDataTypeFloat2 = enum_MTLDataType.define('MTLDataTypeFloat2', 4) # type: ignore
MTLDataTypeFloat3 = enum_MTLDataType.define('MTLDataTypeFloat3', 5) # type: ignore
MTLDataTypeFloat4 = enum_MTLDataType.define('MTLDataTypeFloat4', 6) # type: ignore
MTLDataTypeFloat2x2 = enum_MTLDataType.define('MTLDataTypeFloat2x2', 7) # type: ignore
MTLDataTypeFloat2x3 = enum_MTLDataType.define('MTLDataTypeFloat2x3', 8) # type: ignore
MTLDataTypeFloat2x4 = enum_MTLDataType.define('MTLDataTypeFloat2x4', 9) # type: ignore
MTLDataTypeFloat3x2 = enum_MTLDataType.define('MTLDataTypeFloat3x2', 10) # type: ignore
MTLDataTypeFloat3x3 = enum_MTLDataType.define('MTLDataTypeFloat3x3', 11) # type: ignore
MTLDataTypeFloat3x4 = enum_MTLDataType.define('MTLDataTypeFloat3x4', 12) # type: ignore
MTLDataTypeFloat4x2 = enum_MTLDataType.define('MTLDataTypeFloat4x2', 13) # type: ignore
MTLDataTypeFloat4x3 = enum_MTLDataType.define('MTLDataTypeFloat4x3', 14) # type: ignore
MTLDataTypeFloat4x4 = enum_MTLDataType.define('MTLDataTypeFloat4x4', 15) # type: ignore
MTLDataTypeHalf = enum_MTLDataType.define('MTLDataTypeHalf', 16) # type: ignore
MTLDataTypeHalf2 = enum_MTLDataType.define('MTLDataTypeHalf2', 17) # type: ignore
MTLDataTypeHalf3 = enum_MTLDataType.define('MTLDataTypeHalf3', 18) # type: ignore
MTLDataTypeHalf4 = enum_MTLDataType.define('MTLDataTypeHalf4', 19) # type: ignore
MTLDataTypeHalf2x2 = enum_MTLDataType.define('MTLDataTypeHalf2x2', 20) # type: ignore
MTLDataTypeHalf2x3 = enum_MTLDataType.define('MTLDataTypeHalf2x3', 21) # type: ignore
MTLDataTypeHalf2x4 = enum_MTLDataType.define('MTLDataTypeHalf2x4', 22) # type: ignore
MTLDataTypeHalf3x2 = enum_MTLDataType.define('MTLDataTypeHalf3x2', 23) # type: ignore
MTLDataTypeHalf3x3 = enum_MTLDataType.define('MTLDataTypeHalf3x3', 24) # type: ignore
MTLDataTypeHalf3x4 = enum_MTLDataType.define('MTLDataTypeHalf3x4', 25) # type: ignore
MTLDataTypeHalf4x2 = enum_MTLDataType.define('MTLDataTypeHalf4x2', 26) # type: ignore
MTLDataTypeHalf4x3 = enum_MTLDataType.define('MTLDataTypeHalf4x3', 27) # type: ignore
MTLDataTypeHalf4x4 = enum_MTLDataType.define('MTLDataTypeHalf4x4', 28) # type: ignore
MTLDataTypeInt = enum_MTLDataType.define('MTLDataTypeInt', 29) # type: ignore
MTLDataTypeInt2 = enum_MTLDataType.define('MTLDataTypeInt2', 30) # type: ignore
MTLDataTypeInt3 = enum_MTLDataType.define('MTLDataTypeInt3', 31) # type: ignore
MTLDataTypeInt4 = enum_MTLDataType.define('MTLDataTypeInt4', 32) # type: ignore
MTLDataTypeUInt = enum_MTLDataType.define('MTLDataTypeUInt', 33) # type: ignore
MTLDataTypeUInt2 = enum_MTLDataType.define('MTLDataTypeUInt2', 34) # type: ignore
MTLDataTypeUInt3 = enum_MTLDataType.define('MTLDataTypeUInt3', 35) # type: ignore
MTLDataTypeUInt4 = enum_MTLDataType.define('MTLDataTypeUInt4', 36) # type: ignore
MTLDataTypeShort = enum_MTLDataType.define('MTLDataTypeShort', 37) # type: ignore
MTLDataTypeShort2 = enum_MTLDataType.define('MTLDataTypeShort2', 38) # type: ignore
MTLDataTypeShort3 = enum_MTLDataType.define('MTLDataTypeShort3', 39) # type: ignore
MTLDataTypeShort4 = enum_MTLDataType.define('MTLDataTypeShort4', 40) # type: ignore
MTLDataTypeUShort = enum_MTLDataType.define('MTLDataTypeUShort', 41) # type: ignore
MTLDataTypeUShort2 = enum_MTLDataType.define('MTLDataTypeUShort2', 42) # type: ignore
MTLDataTypeUShort3 = enum_MTLDataType.define('MTLDataTypeUShort3', 43) # type: ignore
MTLDataTypeUShort4 = enum_MTLDataType.define('MTLDataTypeUShort4', 44) # type: ignore
MTLDataTypeChar = enum_MTLDataType.define('MTLDataTypeChar', 45) # type: ignore
MTLDataTypeChar2 = enum_MTLDataType.define('MTLDataTypeChar2', 46) # type: ignore
MTLDataTypeChar3 = enum_MTLDataType.define('MTLDataTypeChar3', 47) # type: ignore
MTLDataTypeChar4 = enum_MTLDataType.define('MTLDataTypeChar4', 48) # type: ignore
MTLDataTypeUChar = enum_MTLDataType.define('MTLDataTypeUChar', 49) # type: ignore
MTLDataTypeUChar2 = enum_MTLDataType.define('MTLDataTypeUChar2', 50) # type: ignore
MTLDataTypeUChar3 = enum_MTLDataType.define('MTLDataTypeUChar3', 51) # type: ignore
MTLDataTypeUChar4 = enum_MTLDataType.define('MTLDataTypeUChar4', 52) # type: ignore
MTLDataTypeBool = enum_MTLDataType.define('MTLDataTypeBool', 53) # type: ignore
MTLDataTypeBool2 = enum_MTLDataType.define('MTLDataTypeBool2', 54) # type: ignore
MTLDataTypeBool3 = enum_MTLDataType.define('MTLDataTypeBool3', 55) # type: ignore
MTLDataTypeBool4 = enum_MTLDataType.define('MTLDataTypeBool4', 56) # type: ignore
MTLDataTypeTexture = enum_MTLDataType.define('MTLDataTypeTexture', 58) # type: ignore
MTLDataTypeSampler = enum_MTLDataType.define('MTLDataTypeSampler', 59) # type: ignore
MTLDataTypePointer = enum_MTLDataType.define('MTLDataTypePointer', 60) # type: ignore
MTLDataTypeR8Unorm = enum_MTLDataType.define('MTLDataTypeR8Unorm', 62) # type: ignore
MTLDataTypeR8Snorm = enum_MTLDataType.define('MTLDataTypeR8Snorm', 63) # type: ignore
MTLDataTypeR16Unorm = enum_MTLDataType.define('MTLDataTypeR16Unorm', 64) # type: ignore
MTLDataTypeR16Snorm = enum_MTLDataType.define('MTLDataTypeR16Snorm', 65) # type: ignore
MTLDataTypeRG8Unorm = enum_MTLDataType.define('MTLDataTypeRG8Unorm', 66) # type: ignore
MTLDataTypeRG8Snorm = enum_MTLDataType.define('MTLDataTypeRG8Snorm', 67) # type: ignore
MTLDataTypeRG16Unorm = enum_MTLDataType.define('MTLDataTypeRG16Unorm', 68) # type: ignore
MTLDataTypeRG16Snorm = enum_MTLDataType.define('MTLDataTypeRG16Snorm', 69) # type: ignore
MTLDataTypeRGBA8Unorm = enum_MTLDataType.define('MTLDataTypeRGBA8Unorm', 70) # type: ignore
MTLDataTypeRGBA8Unorm_sRGB = enum_MTLDataType.define('MTLDataTypeRGBA8Unorm_sRGB', 71) # type: ignore
MTLDataTypeRGBA8Snorm = enum_MTLDataType.define('MTLDataTypeRGBA8Snorm', 72) # type: ignore
MTLDataTypeRGBA16Unorm = enum_MTLDataType.define('MTLDataTypeRGBA16Unorm', 73) # type: ignore
MTLDataTypeRGBA16Snorm = enum_MTLDataType.define('MTLDataTypeRGBA16Snorm', 74) # type: ignore
MTLDataTypeRGB10A2Unorm = enum_MTLDataType.define('MTLDataTypeRGB10A2Unorm', 75) # type: ignore
MTLDataTypeRG11B10Float = enum_MTLDataType.define('MTLDataTypeRG11B10Float', 76) # type: ignore
MTLDataTypeRGB9E5Float = enum_MTLDataType.define('MTLDataTypeRGB9E5Float', 77) # type: ignore
MTLDataTypeRenderPipeline = enum_MTLDataType.define('MTLDataTypeRenderPipeline', 78) # type: ignore
MTLDataTypeComputePipeline = enum_MTLDataType.define('MTLDataTypeComputePipeline', 79) # type: ignore
MTLDataTypeIndirectCommandBuffer = enum_MTLDataType.define('MTLDataTypeIndirectCommandBuffer', 80) # type: ignore
MTLDataTypeLong = enum_MTLDataType.define('MTLDataTypeLong', 81) # type: ignore
MTLDataTypeLong2 = enum_MTLDataType.define('MTLDataTypeLong2', 82) # type: ignore
MTLDataTypeLong3 = enum_MTLDataType.define('MTLDataTypeLong3', 83) # type: ignore
MTLDataTypeLong4 = enum_MTLDataType.define('MTLDataTypeLong4', 84) # type: ignore
MTLDataTypeULong = enum_MTLDataType.define('MTLDataTypeULong', 85) # type: ignore
MTLDataTypeULong2 = enum_MTLDataType.define('MTLDataTypeULong2', 86) # type: ignore
MTLDataTypeULong3 = enum_MTLDataType.define('MTLDataTypeULong3', 87) # type: ignore
MTLDataTypeULong4 = enum_MTLDataType.define('MTLDataTypeULong4', 88) # type: ignore
MTLDataTypeVisibleFunctionTable = enum_MTLDataType.define('MTLDataTypeVisibleFunctionTable', 115) # type: ignore
MTLDataTypeIntersectionFunctionTable = enum_MTLDataType.define('MTLDataTypeIntersectionFunctionTable', 116) # type: ignore
MTLDataTypePrimitiveAccelerationStructure = enum_MTLDataType.define('MTLDataTypePrimitiveAccelerationStructure', 117) # type: ignore
MTLDataTypeInstanceAccelerationStructure = enum_MTLDataType.define('MTLDataTypeInstanceAccelerationStructure', 118) # type: ignore
MTLDataTypeBFloat = enum_MTLDataType.define('MTLDataTypeBFloat', 121) # type: ignore
MTLDataTypeBFloat2 = enum_MTLDataType.define('MTLDataTypeBFloat2', 122) # type: ignore
MTLDataTypeBFloat3 = enum_MTLDataType.define('MTLDataTypeBFloat3', 123) # type: ignore
MTLDataTypeBFloat4 = enum_MTLDataType.define('MTLDataTypeBFloat4', 124) # type: ignore

MTLDataType: TypeAlias = enum_MTLDataType
class MTLStructType(objc.Spec): pass
class MTLStructMember(objc.Spec): pass
class MTLArrayType(objc.Spec): pass
class MTLTextureReferenceType(objc.Spec): pass
class MTLType(objc.Spec): pass
MTLType._bases_ = [NSObject]
MTLType._methods_ = [
  ('dataType', MTLDataType, []),
]
MTLTextureReferenceType._bases_ = [MTLType]
MTLTextureReferenceType._methods_ = [
  ('textureDataType', MTLDataType, []),
  ('textureType', MTLTextureType, []),
  ('access', MTLBindingAccess, []),
  ('isDepthTexture', BOOL, []),
]
class MTLPointerType(objc.Spec): pass
MTLPointerType._bases_ = [MTLType]
MTLPointerType._methods_ = [
  ('elementStructType', MTLStructType, []),
  ('elementArrayType', MTLArrayType, []),
  ('elementType', MTLDataType, []),
  ('access', MTLBindingAccess, []),
  ('alignment', NSUInteger, []),
  ('dataSize', NSUInteger, []),
  ('elementIsArgumentBuffer', BOOL, []),
]
MTLArrayType._bases_ = [MTLType]
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
MTLStructMember._bases_ = [NSObject]
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
MTLStructType._bases_ = [MTLType]
MTLStructType._methods_ = [
  ('memberByName:', MTLStructMember, [NSString]),
]
MTLArgument._bases_ = [NSObject]
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
MTLFunctionTypeVertex = enum_MTLFunctionType.define('MTLFunctionTypeVertex', 1) # type: ignore
MTLFunctionTypeFragment = enum_MTLFunctionType.define('MTLFunctionTypeFragment', 2) # type: ignore
MTLFunctionTypeKernel = enum_MTLFunctionType.define('MTLFunctionTypeKernel', 3) # type: ignore
MTLFunctionTypeVisible = enum_MTLFunctionType.define('MTLFunctionTypeVisible', 5) # type: ignore
MTLFunctionTypeIntersection = enum_MTLFunctionType.define('MTLFunctionTypeIntersection', 6) # type: ignore
MTLFunctionTypeMesh = enum_MTLFunctionType.define('MTLFunctionTypeMesh', 7) # type: ignore
MTLFunctionTypeObject = enum_MTLFunctionType.define('MTLFunctionTypeObject', 8) # type: ignore

MTLFunctionType: TypeAlias = enum_MTLFunctionType
enum_MTLPatchType = CEnum(NSUInteger)
MTLPatchTypeNone = enum_MTLPatchType.define('MTLPatchTypeNone', 0) # type: ignore
MTLPatchTypeTriangle = enum_MTLPatchType.define('MTLPatchTypeTriangle', 1) # type: ignore
MTLPatchTypeQuad = enum_MTLPatchType.define('MTLPatchTypeQuad', 2) # type: ignore

MTLPatchType: TypeAlias = enum_MTLPatchType
enum_MTLFunctionOptions = CEnum(NSUInteger)
MTLFunctionOptionNone = enum_MTLFunctionOptions.define('MTLFunctionOptionNone', 0) # type: ignore
MTLFunctionOptionCompileToBinary = enum_MTLFunctionOptions.define('MTLFunctionOptionCompileToBinary', 1) # type: ignore
MTLFunctionOptionStoreFunctionInMetalScript = enum_MTLFunctionOptions.define('MTLFunctionOptionStoreFunctionInMetalScript', 2) # type: ignore

MTLFunctionOptions: TypeAlias = enum_MTLFunctionOptions
MTLFunction._bases_ = [NSObject]
MTLFunction._methods_ = [
  ('newArgumentEncoderWithBufferIndex:', MTLArgumentEncoder, [NSUInteger], True),
  ('newArgumentEncoderWithBufferIndex:reflection:', MTLArgumentEncoder, [NSUInteger, c.POINTER[MTLArgument]], True),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
  ('functionType', MTLFunctionType, []),
  ('patchType', MTLPatchType, []),
  ('patchControlPointCount', NSInteger, []),
  ('name', NSString, []),
  ('options', MTLFunctionOptions, []),
]
class MTLStageInputOutputDescriptor(objc.Spec): pass
class MTLBufferLayoutDescriptorArray(objc.Spec): pass
class MTLBufferLayoutDescriptor(objc.Spec): pass
enum_MTLStepFunction = CEnum(NSUInteger)
MTLStepFunctionConstant = enum_MTLStepFunction.define('MTLStepFunctionConstant', 0) # type: ignore
MTLStepFunctionPerVertex = enum_MTLStepFunction.define('MTLStepFunctionPerVertex', 1) # type: ignore
MTLStepFunctionPerInstance = enum_MTLStepFunction.define('MTLStepFunctionPerInstance', 2) # type: ignore
MTLStepFunctionPerPatch = enum_MTLStepFunction.define('MTLStepFunctionPerPatch', 3) # type: ignore
MTLStepFunctionPerPatchControlPoint = enum_MTLStepFunction.define('MTLStepFunctionPerPatchControlPoint', 4) # type: ignore
MTLStepFunctionThreadPositionInGridX = enum_MTLStepFunction.define('MTLStepFunctionThreadPositionInGridX', 5) # type: ignore
MTLStepFunctionThreadPositionInGridY = enum_MTLStepFunction.define('MTLStepFunctionThreadPositionInGridY', 6) # type: ignore
MTLStepFunctionThreadPositionInGridXIndexed = enum_MTLStepFunction.define('MTLStepFunctionThreadPositionInGridXIndexed', 7) # type: ignore
MTLStepFunctionThreadPositionInGridYIndexed = enum_MTLStepFunction.define('MTLStepFunctionThreadPositionInGridYIndexed', 8) # type: ignore

MTLStepFunction: TypeAlias = enum_MTLStepFunction
MTLBufferLayoutDescriptor._bases_ = [NSObject]
MTLBufferLayoutDescriptor._methods_ = [
  ('stride', NSUInteger, []),
  ('setStride:', None, [NSUInteger]),
  ('stepFunction', MTLStepFunction, []),
  ('setStepFunction:', None, [MTLStepFunction]),
  ('stepRate', NSUInteger, []),
  ('setStepRate:', None, [NSUInteger]),
]
MTLBufferLayoutDescriptorArray._bases_ = [NSObject]
MTLBufferLayoutDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLBufferLayoutDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLBufferLayoutDescriptor, NSUInteger]),
]
class MTLAttributeDescriptorArray(objc.Spec): pass
class MTLAttributeDescriptor(objc.Spec): pass
enum_MTLAttributeFormat = CEnum(NSUInteger)
MTLAttributeFormatInvalid = enum_MTLAttributeFormat.define('MTLAttributeFormatInvalid', 0) # type: ignore
MTLAttributeFormatUChar2 = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar2', 1) # type: ignore
MTLAttributeFormatUChar3 = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar3', 2) # type: ignore
MTLAttributeFormatUChar4 = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar4', 3) # type: ignore
MTLAttributeFormatChar2 = enum_MTLAttributeFormat.define('MTLAttributeFormatChar2', 4) # type: ignore
MTLAttributeFormatChar3 = enum_MTLAttributeFormat.define('MTLAttributeFormatChar3', 5) # type: ignore
MTLAttributeFormatChar4 = enum_MTLAttributeFormat.define('MTLAttributeFormatChar4', 6) # type: ignore
MTLAttributeFormatUChar2Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar2Normalized', 7) # type: ignore
MTLAttributeFormatUChar3Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar3Normalized', 8) # type: ignore
MTLAttributeFormatUChar4Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar4Normalized', 9) # type: ignore
MTLAttributeFormatChar2Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatChar2Normalized', 10) # type: ignore
MTLAttributeFormatChar3Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatChar3Normalized', 11) # type: ignore
MTLAttributeFormatChar4Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatChar4Normalized', 12) # type: ignore
MTLAttributeFormatUShort2 = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort2', 13) # type: ignore
MTLAttributeFormatUShort3 = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort3', 14) # type: ignore
MTLAttributeFormatUShort4 = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort4', 15) # type: ignore
MTLAttributeFormatShort2 = enum_MTLAttributeFormat.define('MTLAttributeFormatShort2', 16) # type: ignore
MTLAttributeFormatShort3 = enum_MTLAttributeFormat.define('MTLAttributeFormatShort3', 17) # type: ignore
MTLAttributeFormatShort4 = enum_MTLAttributeFormat.define('MTLAttributeFormatShort4', 18) # type: ignore
MTLAttributeFormatUShort2Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort2Normalized', 19) # type: ignore
MTLAttributeFormatUShort3Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort3Normalized', 20) # type: ignore
MTLAttributeFormatUShort4Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort4Normalized', 21) # type: ignore
MTLAttributeFormatShort2Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatShort2Normalized', 22) # type: ignore
MTLAttributeFormatShort3Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatShort3Normalized', 23) # type: ignore
MTLAttributeFormatShort4Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatShort4Normalized', 24) # type: ignore
MTLAttributeFormatHalf2 = enum_MTLAttributeFormat.define('MTLAttributeFormatHalf2', 25) # type: ignore
MTLAttributeFormatHalf3 = enum_MTLAttributeFormat.define('MTLAttributeFormatHalf3', 26) # type: ignore
MTLAttributeFormatHalf4 = enum_MTLAttributeFormat.define('MTLAttributeFormatHalf4', 27) # type: ignore
MTLAttributeFormatFloat = enum_MTLAttributeFormat.define('MTLAttributeFormatFloat', 28) # type: ignore
MTLAttributeFormatFloat2 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloat2', 29) # type: ignore
MTLAttributeFormatFloat3 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloat3', 30) # type: ignore
MTLAttributeFormatFloat4 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloat4', 31) # type: ignore
MTLAttributeFormatInt = enum_MTLAttributeFormat.define('MTLAttributeFormatInt', 32) # type: ignore
MTLAttributeFormatInt2 = enum_MTLAttributeFormat.define('MTLAttributeFormatInt2', 33) # type: ignore
MTLAttributeFormatInt3 = enum_MTLAttributeFormat.define('MTLAttributeFormatInt3', 34) # type: ignore
MTLAttributeFormatInt4 = enum_MTLAttributeFormat.define('MTLAttributeFormatInt4', 35) # type: ignore
MTLAttributeFormatUInt = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt', 36) # type: ignore
MTLAttributeFormatUInt2 = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt2', 37) # type: ignore
MTLAttributeFormatUInt3 = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt3', 38) # type: ignore
MTLAttributeFormatUInt4 = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt4', 39) # type: ignore
MTLAttributeFormatInt1010102Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatInt1010102Normalized', 40) # type: ignore
MTLAttributeFormatUInt1010102Normalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUInt1010102Normalized', 41) # type: ignore
MTLAttributeFormatUChar4Normalized_BGRA = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar4Normalized_BGRA', 42) # type: ignore
MTLAttributeFormatUChar = enum_MTLAttributeFormat.define('MTLAttributeFormatUChar', 45) # type: ignore
MTLAttributeFormatChar = enum_MTLAttributeFormat.define('MTLAttributeFormatChar', 46) # type: ignore
MTLAttributeFormatUCharNormalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUCharNormalized', 47) # type: ignore
MTLAttributeFormatCharNormalized = enum_MTLAttributeFormat.define('MTLAttributeFormatCharNormalized', 48) # type: ignore
MTLAttributeFormatUShort = enum_MTLAttributeFormat.define('MTLAttributeFormatUShort', 49) # type: ignore
MTLAttributeFormatShort = enum_MTLAttributeFormat.define('MTLAttributeFormatShort', 50) # type: ignore
MTLAttributeFormatUShortNormalized = enum_MTLAttributeFormat.define('MTLAttributeFormatUShortNormalized', 51) # type: ignore
MTLAttributeFormatShortNormalized = enum_MTLAttributeFormat.define('MTLAttributeFormatShortNormalized', 52) # type: ignore
MTLAttributeFormatHalf = enum_MTLAttributeFormat.define('MTLAttributeFormatHalf', 53) # type: ignore
MTLAttributeFormatFloatRG11B10 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloatRG11B10', 54) # type: ignore
MTLAttributeFormatFloatRGB9E5 = enum_MTLAttributeFormat.define('MTLAttributeFormatFloatRGB9E5', 55) # type: ignore

MTLAttributeFormat: TypeAlias = enum_MTLAttributeFormat
MTLAttributeDescriptor._bases_ = [NSObject]
MTLAttributeDescriptor._methods_ = [
  ('format', MTLAttributeFormat, []),
  ('setFormat:', None, [MTLAttributeFormat]),
  ('offset', NSUInteger, []),
  ('setOffset:', None, [NSUInteger]),
  ('bufferIndex', NSUInteger, []),
  ('setBufferIndex:', None, [NSUInteger]),
]
MTLAttributeDescriptorArray._bases_ = [NSObject]
MTLAttributeDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLAttributeDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLAttributeDescriptor, NSUInteger]),
]
enum_MTLIndexType = CEnum(NSUInteger)
MTLIndexTypeUInt16 = enum_MTLIndexType.define('MTLIndexTypeUInt16', 0) # type: ignore
MTLIndexTypeUInt32 = enum_MTLIndexType.define('MTLIndexTypeUInt32', 1) # type: ignore

MTLIndexType: TypeAlias = enum_MTLIndexType
MTLStageInputOutputDescriptor._bases_ = [NSObject]
MTLStageInputOutputDescriptor._methods_ = [
  ('reset', None, []),
  ('layouts', MTLBufferLayoutDescriptorArray, []),
  ('attributes', MTLAttributeDescriptorArray, []),
  ('indexType', MTLIndexType, []),
  ('setIndexType:', None, [MTLIndexType]),
  ('indexBufferIndex', NSUInteger, []),
  ('setIndexBufferIndex:', None, [NSUInteger]),
]
MTLStageInputOutputDescriptor._classmethods_ = [
  ('stageInputOutputDescriptor', MTLStageInputOutputDescriptor, []),
]
class MTLPipelineBufferDescriptorArray(objc.Spec): pass
class MTLPipelineBufferDescriptor(objc.Spec): pass
enum_MTLMutability = CEnum(NSUInteger)
MTLMutabilityDefault = enum_MTLMutability.define('MTLMutabilityDefault', 0) # type: ignore
MTLMutabilityMutable = enum_MTLMutability.define('MTLMutabilityMutable', 1) # type: ignore
MTLMutabilityImmutable = enum_MTLMutability.define('MTLMutabilityImmutable', 2) # type: ignore

MTLMutability: TypeAlias = enum_MTLMutability
MTLPipelineBufferDescriptor._bases_ = [NSObject]
MTLPipelineBufferDescriptor._methods_ = [
  ('mutability', MTLMutability, []),
  ('setMutability:', None, [MTLMutability]),
]
MTLPipelineBufferDescriptorArray._bases_ = [NSObject]
MTLPipelineBufferDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLPipelineBufferDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLPipelineBufferDescriptor, NSUInteger]),
]
class MTLLinkedFunctions(objc.Spec): pass
MTLLinkedFunctions._bases_ = [NSObject]
MTLLinkedFunctions._classmethods_ = [
  ('linkedFunctions', MTLLinkedFunctions, []),
]
MTLComputePipelineDescriptor._bases_ = [NSObject]
MTLComputePipelineDescriptor._methods_ = [
  ('reset', None, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('computeFunction', MTLFunction, []),
  ('setComputeFunction:', None, [MTLFunction]),
  ('threadGroupSizeIsMultipleOfThreadExecutionWidth', BOOL, []),
  ('setThreadGroupSizeIsMultipleOfThreadExecutionWidth:', None, [BOOL]),
  ('maxTotalThreadsPerThreadgroup', NSUInteger, []),
  ('setMaxTotalThreadsPerThreadgroup:', None, [NSUInteger]),
  ('stageInputDescriptor', MTLStageInputOutputDescriptor, []),
  ('setStageInputDescriptor:', None, [MTLStageInputOutputDescriptor]),
  ('buffers', MTLPipelineBufferDescriptorArray, []),
  ('supportIndirectCommandBuffers', BOOL, []),
  ('setSupportIndirectCommandBuffers:', None, [BOOL]),
  ('linkedFunctions', MTLLinkedFunctions, []),
  ('setLinkedFunctions:', None, [MTLLinkedFunctions]),
  ('supportAddingBinaryFunctions', BOOL, []),
  ('setSupportAddingBinaryFunctions:', None, [BOOL]),
  ('maxCallStackDepth', NSUInteger, []),
  ('setMaxCallStackDepth:', None, [NSUInteger]),
]
class MTLFunctionHandle(objc.Spec): pass
class MTLVisibleFunctionTableDescriptor(objc.Spec): pass
class MTLIntersectionFunctionTableDescriptor(objc.Spec): pass
@c.record
class struct_MTLResourceID(c.Struct):
  SIZE = 8
  _impl: Annotated[uint64_t, 0]
MTLResourceID: TypeAlias = struct_MTLResourceID
MTLComputePipelineState._bases_ = [NSObject]
MTLComputePipelineState._methods_ = [
  ('imageblockMemoryLengthForDimensions:', NSUInteger, [MTLSize]),
  ('functionHandleWithFunction:', MTLFunctionHandle, [MTLFunction]),
  ('newVisibleFunctionTableWithDescriptor:', MTLVisibleFunctionTable, [MTLVisibleFunctionTableDescriptor], True),
  ('newIntersectionFunctionTableWithDescriptor:', MTLIntersectionFunctionTable, [MTLIntersectionFunctionTableDescriptor], True),
  ('label', NSString, []),
  ('device', MTLDevice, []),
  ('maxTotalThreadsPerThreadgroup', NSUInteger, []),
  ('threadExecutionWidth', NSUInteger, []),
  ('staticThreadgroupMemoryLength', NSUInteger, []),
  ('supportIndirectCommandBuffers', BOOL, []),
  ('gpuResourceID', MTLResourceID, []),
]
class MTLCommandQueue(objc.Spec): pass
class MTLCommandBuffer(objc.Spec): pass
class MTLDrawable(objc.Spec): pass
CFTimeInterval = Annotated[float, ctypes.c_double]
class MTLBlitCommandEncoder(objc.Spec): pass
enum_MTLBlitOption = CEnum(NSUInteger)
MTLBlitOptionNone = enum_MTLBlitOption.define('MTLBlitOptionNone', 0) # type: ignore
MTLBlitOptionDepthFromDepthStencil = enum_MTLBlitOption.define('MTLBlitOptionDepthFromDepthStencil', 1) # type: ignore
MTLBlitOptionStencilFromDepthStencil = enum_MTLBlitOption.define('MTLBlitOptionStencilFromDepthStencil', 2) # type: ignore
MTLBlitOptionRowLinearPVRTC = enum_MTLBlitOption.define('MTLBlitOptionRowLinearPVRTC', 4) # type: ignore

MTLBlitOption: TypeAlias = enum_MTLBlitOption
MTLBlitCommandEncoder._bases_ = [MTLCommandEncoder]
MTLBlitCommandEncoder._methods_ = [
  ('synchronizeResource:', None, [MTLResource]),
  ('synchronizeTexture:slice:level:', None, [MTLTexture, NSUInteger, NSUInteger]),
  ('copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:', None, [MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLSize, MTLTexture, NSUInteger, NSUInteger, MTLOrigin]),
  ('copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLSize, MTLTexture, NSUInteger, NSUInteger, MTLOrigin]),
  ('copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLSize, MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLBlitOption]),
  ('copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:', None, [MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLSize, MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:', None, [MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLSize, MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLBlitOption]),
  ('generateMipmapsForTexture:', None, [MTLTexture]),
  ('fillBuffer:range:value:', None, [MTLBuffer, NSRange, uint8_t]),
  ('copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:', None, [MTLTexture, NSUInteger, NSUInteger, MTLTexture, NSUInteger, NSUInteger, NSUInteger, NSUInteger]),
  ('copyFromTexture:toTexture:', None, [MTLTexture, MTLTexture]),
  ('copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:', None, [MTLBuffer, NSUInteger, MTLBuffer, NSUInteger, NSUInteger]),
  ('updateFence:', None, [MTLFence]),
  ('waitForFence:', None, [MTLFence]),
  ('getTextureAccessCounters:region:mipLevel:slice:resetCounters:countersBuffer:countersBufferOffset:', None, [MTLTexture, MTLRegion, NSUInteger, NSUInteger, BOOL, MTLBuffer, NSUInteger]),
  ('resetTextureAccessCounters:region:mipLevel:slice:', None, [MTLTexture, MTLRegion, NSUInteger, NSUInteger]),
  ('optimizeContentsForGPUAccess:', None, [MTLTexture]),
  ('optimizeContentsForGPUAccess:slice:level:', None, [MTLTexture, NSUInteger, NSUInteger]),
  ('optimizeContentsForCPUAccess:', None, [MTLTexture]),
  ('optimizeContentsForCPUAccess:slice:level:', None, [MTLTexture, NSUInteger, NSUInteger]),
  ('resetCommandsInBuffer:withRange:', None, [MTLIndirectCommandBuffer, NSRange]),
  ('copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:', None, [MTLIndirectCommandBuffer, NSRange, MTLIndirectCommandBuffer, NSUInteger]),
  ('optimizeIndirectCommandBuffer:withRange:', None, [MTLIndirectCommandBuffer, NSRange]),
  ('sampleCountersInBuffer:atSampleIndex:withBarrier:', None, [MTLCounterSampleBuffer, NSUInteger, BOOL]),
  ('resolveCounters:inRange:destinationBuffer:destinationOffset:', None, [MTLCounterSampleBuffer, NSRange, MTLBuffer, NSUInteger]),
]
class MTLRenderCommandEncoder(objc.Spec): pass
class MTLRenderPassDescriptor(objc.Spec): pass
@c.record
class MTLSamplePosition(c.Struct):
  SIZE = 8
  x: Annotated[Annotated[float, ctypes.c_float], 0]
  y: Annotated[Annotated[float, ctypes.c_float], 4]
class MTLRenderPassColorAttachmentDescriptorArray(objc.Spec): pass
class MTLRenderPassColorAttachmentDescriptor(objc.Spec): pass
@c.record
class MTLClearColor(c.Struct):
  SIZE = 32
  red: Annotated[Annotated[float, ctypes.c_double], 0]
  green: Annotated[Annotated[float, ctypes.c_double], 8]
  blue: Annotated[Annotated[float, ctypes.c_double], 16]
  alpha: Annotated[Annotated[float, ctypes.c_double], 24]
class MTLRenderPassAttachmentDescriptor(objc.Spec): pass
enum_MTLLoadAction = CEnum(NSUInteger)
MTLLoadActionDontCare = enum_MTLLoadAction.define('MTLLoadActionDontCare', 0) # type: ignore
MTLLoadActionLoad = enum_MTLLoadAction.define('MTLLoadActionLoad', 1) # type: ignore
MTLLoadActionClear = enum_MTLLoadAction.define('MTLLoadActionClear', 2) # type: ignore

MTLLoadAction: TypeAlias = enum_MTLLoadAction
enum_MTLStoreAction = CEnum(NSUInteger)
MTLStoreActionDontCare = enum_MTLStoreAction.define('MTLStoreActionDontCare', 0) # type: ignore
MTLStoreActionStore = enum_MTLStoreAction.define('MTLStoreActionStore', 1) # type: ignore
MTLStoreActionMultisampleResolve = enum_MTLStoreAction.define('MTLStoreActionMultisampleResolve', 2) # type: ignore
MTLStoreActionStoreAndMultisampleResolve = enum_MTLStoreAction.define('MTLStoreActionStoreAndMultisampleResolve', 3) # type: ignore
MTLStoreActionUnknown = enum_MTLStoreAction.define('MTLStoreActionUnknown', 4) # type: ignore
MTLStoreActionCustomSampleDepthStore = enum_MTLStoreAction.define('MTLStoreActionCustomSampleDepthStore', 5) # type: ignore

MTLStoreAction: TypeAlias = enum_MTLStoreAction
enum_MTLStoreActionOptions = CEnum(NSUInteger)
MTLStoreActionOptionNone = enum_MTLStoreActionOptions.define('MTLStoreActionOptionNone', 0) # type: ignore
MTLStoreActionOptionCustomSamplePositions = enum_MTLStoreActionOptions.define('MTLStoreActionOptionCustomSamplePositions', 1) # type: ignore

MTLStoreActionOptions: TypeAlias = enum_MTLStoreActionOptions
MTLRenderPassAttachmentDescriptor._bases_ = [NSObject]
MTLRenderPassAttachmentDescriptor._methods_ = [
  ('texture', MTLTexture, []),
  ('setTexture:', None, [MTLTexture]),
  ('level', NSUInteger, []),
  ('setLevel:', None, [NSUInteger]),
  ('slice', NSUInteger, []),
  ('setSlice:', None, [NSUInteger]),
  ('depthPlane', NSUInteger, []),
  ('setDepthPlane:', None, [NSUInteger]),
  ('resolveTexture', MTLTexture, []),
  ('setResolveTexture:', None, [MTLTexture]),
  ('resolveLevel', NSUInteger, []),
  ('setResolveLevel:', None, [NSUInteger]),
  ('resolveSlice', NSUInteger, []),
  ('setResolveSlice:', None, [NSUInteger]),
  ('resolveDepthPlane', NSUInteger, []),
  ('setResolveDepthPlane:', None, [NSUInteger]),
  ('loadAction', MTLLoadAction, []),
  ('setLoadAction:', None, [MTLLoadAction]),
  ('storeAction', MTLStoreAction, []),
  ('setStoreAction:', None, [MTLStoreAction]),
  ('storeActionOptions', MTLStoreActionOptions, []),
  ('setStoreActionOptions:', None, [MTLStoreActionOptions]),
]
MTLRenderPassColorAttachmentDescriptor._bases_ = [MTLRenderPassAttachmentDescriptor]
MTLRenderPassColorAttachmentDescriptor._methods_ = [
  ('clearColor', MTLClearColor, []),
  ('setClearColor:', None, [MTLClearColor]),
]
MTLRenderPassColorAttachmentDescriptorArray._bases_ = [NSObject]
MTLRenderPassColorAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLRenderPassColorAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLRenderPassColorAttachmentDescriptor, NSUInteger]),
]
class MTLRenderPassDepthAttachmentDescriptor(objc.Spec): pass
enum_MTLMultisampleDepthResolveFilter = CEnum(NSUInteger)
MTLMultisampleDepthResolveFilterSample0 = enum_MTLMultisampleDepthResolveFilter.define('MTLMultisampleDepthResolveFilterSample0', 0) # type: ignore
MTLMultisampleDepthResolveFilterMin = enum_MTLMultisampleDepthResolveFilter.define('MTLMultisampleDepthResolveFilterMin', 1) # type: ignore
MTLMultisampleDepthResolveFilterMax = enum_MTLMultisampleDepthResolveFilter.define('MTLMultisampleDepthResolveFilterMax', 2) # type: ignore

MTLMultisampleDepthResolveFilter: TypeAlias = enum_MTLMultisampleDepthResolveFilter
MTLRenderPassDepthAttachmentDescriptor._bases_ = [MTLRenderPassAttachmentDescriptor]
MTLRenderPassDepthAttachmentDescriptor._methods_ = [
  ('clearDepth', Annotated[float, ctypes.c_double], []),
  ('setClearDepth:', None, [Annotated[float, ctypes.c_double]]),
  ('depthResolveFilter', MTLMultisampleDepthResolveFilter, []),
  ('setDepthResolveFilter:', None, [MTLMultisampleDepthResolveFilter]),
]
class MTLRenderPassStencilAttachmentDescriptor(objc.Spec): pass
enum_MTLMultisampleStencilResolveFilter = CEnum(NSUInteger)
MTLMultisampleStencilResolveFilterSample0 = enum_MTLMultisampleStencilResolveFilter.define('MTLMultisampleStencilResolveFilterSample0', 0) # type: ignore
MTLMultisampleStencilResolveFilterDepthResolvedSample = enum_MTLMultisampleStencilResolveFilter.define('MTLMultisampleStencilResolveFilterDepthResolvedSample', 1) # type: ignore

MTLMultisampleStencilResolveFilter: TypeAlias = enum_MTLMultisampleStencilResolveFilter
MTLRenderPassStencilAttachmentDescriptor._bases_ = [MTLRenderPassAttachmentDescriptor]
MTLRenderPassStencilAttachmentDescriptor._methods_ = [
  ('clearStencil', uint32_t, []),
  ('setClearStencil:', None, [uint32_t]),
  ('stencilResolveFilter', MTLMultisampleStencilResolveFilter, []),
  ('setStencilResolveFilter:', None, [MTLMultisampleStencilResolveFilter]),
]
class MTLRasterizationRateMap(objc.Spec): pass
class MTLRenderPassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLRenderPassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLRenderPassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLRenderPassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfVertexSampleIndex', NSUInteger, []),
  ('setStartOfVertexSampleIndex:', None, [NSUInteger]),
  ('endOfVertexSampleIndex', NSUInteger, []),
  ('setEndOfVertexSampleIndex:', None, [NSUInteger]),
  ('startOfFragmentSampleIndex', NSUInteger, []),
  ('setStartOfFragmentSampleIndex:', None, [NSUInteger]),
  ('endOfFragmentSampleIndex', NSUInteger, []),
  ('setEndOfFragmentSampleIndex:', None, [NSUInteger]),
]
MTLRenderPassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLRenderPassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLRenderPassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLRenderPassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLRenderPassDescriptor._bases_ = [NSObject]
MTLRenderPassDescriptor._methods_ = [
  ('setSamplePositions:count:', None, [c.POINTER[MTLSamplePosition], NSUInteger]),
  ('getSamplePositions:count:', NSUInteger, [c.POINTER[MTLSamplePosition], NSUInteger]),
  ('colorAttachments', MTLRenderPassColorAttachmentDescriptorArray, []),
  ('depthAttachment', MTLRenderPassDepthAttachmentDescriptor, []),
  ('setDepthAttachment:', None, [MTLRenderPassDepthAttachmentDescriptor]),
  ('stencilAttachment', MTLRenderPassStencilAttachmentDescriptor, []),
  ('setStencilAttachment:', None, [MTLRenderPassStencilAttachmentDescriptor]),
  ('visibilityResultBuffer', MTLBuffer, []),
  ('setVisibilityResultBuffer:', None, [MTLBuffer]),
  ('renderTargetArrayLength', NSUInteger, []),
  ('setRenderTargetArrayLength:', None, [NSUInteger]),
  ('imageblockSampleLength', NSUInteger, []),
  ('setImageblockSampleLength:', None, [NSUInteger]),
  ('threadgroupMemoryLength', NSUInteger, []),
  ('setThreadgroupMemoryLength:', None, [NSUInteger]),
  ('tileWidth', NSUInteger, []),
  ('setTileWidth:', None, [NSUInteger]),
  ('tileHeight', NSUInteger, []),
  ('setTileHeight:', None, [NSUInteger]),
  ('defaultRasterSampleCount', NSUInteger, []),
  ('setDefaultRasterSampleCount:', None, [NSUInteger]),
  ('renderTargetWidth', NSUInteger, []),
  ('setRenderTargetWidth:', None, [NSUInteger]),
  ('renderTargetHeight', NSUInteger, []),
  ('setRenderTargetHeight:', None, [NSUInteger]),
  ('rasterizationRateMap', MTLRasterizationRateMap, []),
  ('setRasterizationRateMap:', None, [MTLRasterizationRateMap]),
  ('sampleBufferAttachments', MTLRenderPassSampleBufferAttachmentDescriptorArray, []),
]
MTLRenderPassDescriptor._classmethods_ = [
  ('renderPassDescriptor', MTLRenderPassDescriptor, []),
]
class MTLComputePassDescriptor(objc.Spec): pass
class MTLComputePassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLComputePassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLComputePassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLComputePassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfEncoderSampleIndex', NSUInteger, []),
  ('setStartOfEncoderSampleIndex:', None, [NSUInteger]),
  ('endOfEncoderSampleIndex', NSUInteger, []),
  ('setEndOfEncoderSampleIndex:', None, [NSUInteger]),
]
MTLComputePassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLComputePassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLComputePassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLComputePassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLComputePassDescriptor._bases_ = [NSObject]
MTLComputePassDescriptor._methods_ = [
  ('dispatchType', MTLDispatchType, []),
  ('setDispatchType:', None, [MTLDispatchType]),
  ('sampleBufferAttachments', MTLComputePassSampleBufferAttachmentDescriptorArray, []),
]
MTLComputePassDescriptor._classmethods_ = [
  ('computePassDescriptor', MTLComputePassDescriptor, []),
]
class MTLBlitPassDescriptor(objc.Spec): pass
class MTLBlitPassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLBlitPassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLBlitPassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLBlitPassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfEncoderSampleIndex', NSUInteger, []),
  ('setStartOfEncoderSampleIndex:', None, [NSUInteger]),
  ('endOfEncoderSampleIndex', NSUInteger, []),
  ('setEndOfEncoderSampleIndex:', None, [NSUInteger]),
]
MTLBlitPassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLBlitPassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLBlitPassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLBlitPassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLBlitPassDescriptor._bases_ = [NSObject]
MTLBlitPassDescriptor._methods_ = [
  ('sampleBufferAttachments', MTLBlitPassSampleBufferAttachmentDescriptorArray, []),
]
MTLBlitPassDescriptor._classmethods_ = [
  ('blitPassDescriptor', MTLBlitPassDescriptor, []),
]
class MTLEvent(objc.Spec): pass
class MTLParallelRenderCommandEncoder(objc.Spec): pass
class MTLResourceStateCommandEncoder(objc.Spec): pass
enum_MTLSparseTextureMappingMode = CEnum(NSUInteger)
MTLSparseTextureMappingModeMap = enum_MTLSparseTextureMappingMode.define('MTLSparseTextureMappingModeMap', 0) # type: ignore
MTLSparseTextureMappingModeUnmap = enum_MTLSparseTextureMappingMode.define('MTLSparseTextureMappingModeUnmap', 1) # type: ignore

MTLSparseTextureMappingMode: TypeAlias = enum_MTLSparseTextureMappingMode
MTLResourceStateCommandEncoder._bases_ = [MTLCommandEncoder]
MTLResourceStateCommandEncoder._methods_ = [
  ('updateTextureMappings:mode:regions:mipLevels:slices:numRegions:', None, [MTLTexture, MTLSparseTextureMappingMode, c.POINTER[MTLRegion], c.POINTER[NSUInteger], c.POINTER[NSUInteger], NSUInteger]),
  ('updateTextureMapping:mode:region:mipLevel:slice:', None, [MTLTexture, MTLSparseTextureMappingMode, MTLRegion, NSUInteger, NSUInteger]),
  ('updateTextureMapping:mode:indirectBuffer:indirectBufferOffset:', None, [MTLTexture, MTLSparseTextureMappingMode, MTLBuffer, NSUInteger]),
  ('updateFence:', None, [MTLFence]),
  ('waitForFence:', None, [MTLFence]),
  ('moveTextureMappingsFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:', None, [MTLTexture, NSUInteger, NSUInteger, MTLOrigin, MTLSize, MTLTexture, NSUInteger, NSUInteger, MTLOrigin]),
]
class MTLResourceStatePassDescriptor(objc.Spec): pass
class MTLResourceStatePassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLResourceStatePassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLResourceStatePassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLResourceStatePassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfEncoderSampleIndex', NSUInteger, []),
  ('setStartOfEncoderSampleIndex:', None, [NSUInteger]),
  ('endOfEncoderSampleIndex', NSUInteger, []),
  ('setEndOfEncoderSampleIndex:', None, [NSUInteger]),
]
MTLResourceStatePassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLResourceStatePassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLResourceStatePassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLResourceStatePassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLResourceStatePassDescriptor._bases_ = [NSObject]
MTLResourceStatePassDescriptor._methods_ = [
  ('sampleBufferAttachments', MTLResourceStatePassSampleBufferAttachmentDescriptorArray, []),
]
MTLResourceStatePassDescriptor._classmethods_ = [
  ('resourceStatePassDescriptor', MTLResourceStatePassDescriptor, []),
]
class MTLAccelerationStructureCommandEncoder(objc.Spec): pass
class MTLAccelerationStructurePassDescriptor(objc.Spec): pass
class MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray(objc.Spec): pass
class MTLAccelerationStructurePassSampleBufferAttachmentDescriptor(objc.Spec): pass
MTLAccelerationStructurePassSampleBufferAttachmentDescriptor._bases_ = [NSObject]
MTLAccelerationStructurePassSampleBufferAttachmentDescriptor._methods_ = [
  ('sampleBuffer', MTLCounterSampleBuffer, []),
  ('setSampleBuffer:', None, [MTLCounterSampleBuffer]),
  ('startOfEncoderSampleIndex', NSUInteger, []),
  ('setStartOfEncoderSampleIndex:', None, [NSUInteger]),
  ('endOfEncoderSampleIndex', NSUInteger, []),
  ('setEndOfEncoderSampleIndex:', None, [NSUInteger]),
]
MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray._bases_ = [NSObject]
MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray._methods_ = [
  ('objectAtIndexedSubscript:', MTLAccelerationStructurePassSampleBufferAttachmentDescriptor, [NSUInteger]),
  ('setObject:atIndexedSubscript:', None, [MTLAccelerationStructurePassSampleBufferAttachmentDescriptor, NSUInteger]),
]
MTLAccelerationStructurePassDescriptor._bases_ = [NSObject]
MTLAccelerationStructurePassDescriptor._methods_ = [
  ('sampleBufferAttachments', MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray, []),
]
MTLAccelerationStructurePassDescriptor._classmethods_ = [
  ('accelerationStructurePassDescriptor', MTLAccelerationStructurePassDescriptor, []),
]
enum_MTLCommandBufferErrorOption = CEnum(NSUInteger)
MTLCommandBufferErrorOptionNone = enum_MTLCommandBufferErrorOption.define('MTLCommandBufferErrorOptionNone', 0) # type: ignore
MTLCommandBufferErrorOptionEncoderExecutionStatus = enum_MTLCommandBufferErrorOption.define('MTLCommandBufferErrorOptionEncoderExecutionStatus', 1) # type: ignore

MTLCommandBufferErrorOption: TypeAlias = enum_MTLCommandBufferErrorOption
class MTLLogContainer(objc.Spec): pass
enum_MTLCommandBufferStatus = CEnum(NSUInteger)
MTLCommandBufferStatusNotEnqueued = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusNotEnqueued', 0) # type: ignore
MTLCommandBufferStatusEnqueued = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusEnqueued', 1) # type: ignore
MTLCommandBufferStatusCommitted = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusCommitted', 2) # type: ignore
MTLCommandBufferStatusScheduled = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusScheduled', 3) # type: ignore
MTLCommandBufferStatusCompleted = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusCompleted', 4) # type: ignore
MTLCommandBufferStatusError = enum_MTLCommandBufferStatus.define('MTLCommandBufferStatusError', 5) # type: ignore

MTLCommandBufferStatus: TypeAlias = enum_MTLCommandBufferStatus
class NSError(objc.Spec): pass
NSErrorDomain = NSString
NSError._bases_ = [NSObject]
NSError._methods_ = [
  ('domain', NSErrorDomain, []),
  ('code', NSInteger, []),
  ('localizedDescription', NSString, []),
  ('localizedFailureReason', NSString, []),
  ('localizedRecoverySuggestion', NSString, []),
  ('recoveryAttempter', objc.id_, []),
  ('helpAnchor', NSString, []),
]
MTLCommandBuffer._bases_ = [NSObject]
MTLCommandBuffer._methods_ = [
  ('enqueue', None, []),
  ('commit', None, []),
  ('presentDrawable:', None, [MTLDrawable]),
  ('presentDrawable:atTime:', None, [MTLDrawable, CFTimeInterval]),
  ('presentDrawable:afterMinimumDuration:', None, [MTLDrawable, CFTimeInterval]),
  ('waitUntilScheduled', None, []),
  ('waitUntilCompleted', None, []),
  ('blitCommandEncoder', MTLBlitCommandEncoder, []),
  ('renderCommandEncoderWithDescriptor:', MTLRenderCommandEncoder, [MTLRenderPassDescriptor]),
  ('computeCommandEncoderWithDescriptor:', MTLComputeCommandEncoder, [MTLComputePassDescriptor]),
  ('blitCommandEncoderWithDescriptor:', MTLBlitCommandEncoder, [MTLBlitPassDescriptor]),
  ('computeCommandEncoder', MTLComputeCommandEncoder, []),
  ('computeCommandEncoderWithDispatchType:', MTLComputeCommandEncoder, [MTLDispatchType]),
  ('encodeWaitForEvent:value:', None, [MTLEvent, uint64_t]),
  ('encodeSignalEvent:value:', None, [MTLEvent, uint64_t]),
  ('parallelRenderCommandEncoderWithDescriptor:', MTLParallelRenderCommandEncoder, [MTLRenderPassDescriptor]),
  ('resourceStateCommandEncoder', MTLResourceStateCommandEncoder, []),
  ('resourceStateCommandEncoderWithDescriptor:', MTLResourceStateCommandEncoder, [MTLResourceStatePassDescriptor]),
  ('accelerationStructureCommandEncoder', MTLAccelerationStructureCommandEncoder, []),
  ('accelerationStructureCommandEncoderWithDescriptor:', MTLAccelerationStructureCommandEncoder, [MTLAccelerationStructurePassDescriptor]),
  ('pushDebugGroup:', None, [NSString]),
  ('popDebugGroup', None, []),
  ('device', MTLDevice, []),
  ('commandQueue', MTLCommandQueue, []),
  ('retainedReferences', BOOL, []),
  ('errorOptions', MTLCommandBufferErrorOption, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('kernelStartTime', CFTimeInterval, []),
  ('kernelEndTime', CFTimeInterval, []),
  ('logs', MTLLogContainer, []),
  ('GPUStartTime', CFTimeInterval, []),
  ('GPUEndTime', CFTimeInterval, []),
  ('status', MTLCommandBufferStatus, []),
  ('error', NSError, []),
]
class MTLCommandBufferDescriptor(objc.Spec): pass
MTLCommandBufferDescriptor._bases_ = [NSObject]
MTLCommandBufferDescriptor._methods_ = [
  ('retainedReferences', BOOL, []),
  ('setRetainedReferences:', None, [BOOL]),
  ('errorOptions', MTLCommandBufferErrorOption, []),
  ('setErrorOptions:', None, [MTLCommandBufferErrorOption]),
]
MTLCommandQueue._bases_ = [NSObject]
MTLCommandQueue._methods_ = [
  ('commandBuffer', MTLCommandBuffer, []),
  ('commandBufferWithDescriptor:', MTLCommandBuffer, [MTLCommandBufferDescriptor]),
  ('commandBufferWithUnretainedReferences', MTLCommandBuffer, []),
  ('insertDebugCaptureBoundary', None, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
]
enum_MTLIOCompressionMethod = CEnum(NSInteger)
MTLIOCompressionMethodZlib = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodZlib', 0) # type: ignore
MTLIOCompressionMethodLZFSE = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZFSE', 1) # type: ignore
MTLIOCompressionMethodLZ4 = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZ4', 2) # type: ignore
MTLIOCompressionMethodLZMA = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZMA', 3) # type: ignore
MTLIOCompressionMethodLZBitmap = enum_MTLIOCompressionMethod.define('MTLIOCompressionMethodLZBitmap', 4) # type: ignore

MTLIOCompressionMethod: TypeAlias = enum_MTLIOCompressionMethod
@dll.bind
def MTLCreateSystemDefaultDevice() -> MTLDevice: ...
MTLCreateSystemDefaultDevice = objc.returns_retained(MTLCreateSystemDefaultDevice)
MTLDeviceNotificationName = NSString
try: MTLDeviceWasAddedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasAddedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceRemovalRequestedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceRemovalRequestedNotification')
except (ValueError,AttributeError): pass
try: MTLDeviceWasRemovedNotification = MTLDeviceNotificationName.in_dll(dll, 'MTLDeviceWasRemovedNotification')
except (ValueError,AttributeError): pass
@dll.bind
def MTLRemoveDeviceObserver(observer:NSObject) -> None: ...
enum_MTLFeatureSet = CEnum(NSUInteger)
MTLFeatureSet_iOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v1', 0) # type: ignore
MTLFeatureSet_iOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v1', 1) # type: ignore
MTLFeatureSet_iOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v2', 2) # type: ignore
MTLFeatureSet_iOS_GPUFamily2_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v2', 3) # type: ignore
MTLFeatureSet_iOS_GPUFamily3_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v1', 4) # type: ignore
MTLFeatureSet_iOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v3', 5) # type: ignore
MTLFeatureSet_iOS_GPUFamily2_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v3', 6) # type: ignore
MTLFeatureSet_iOS_GPUFamily3_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v2', 7) # type: ignore
MTLFeatureSet_iOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v4', 8) # type: ignore
MTLFeatureSet_iOS_GPUFamily2_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v4', 9) # type: ignore
MTLFeatureSet_iOS_GPUFamily3_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v3', 10) # type: ignore
MTLFeatureSet_iOS_GPUFamily4_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily4_v1', 11) # type: ignore
MTLFeatureSet_iOS_GPUFamily1_v5 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily1_v5', 12) # type: ignore
MTLFeatureSet_iOS_GPUFamily2_v5 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily2_v5', 13) # type: ignore
MTLFeatureSet_iOS_GPUFamily3_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily3_v4', 14) # type: ignore
MTLFeatureSet_iOS_GPUFamily4_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily4_v2', 15) # type: ignore
MTLFeatureSet_iOS_GPUFamily5_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_iOS_GPUFamily5_v1', 16) # type: ignore
MTLFeatureSet_macOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v1', 10000) # type: ignore
MTLFeatureSet_OSX_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_GPUFamily1_v1', 10000) # type: ignore
MTLFeatureSet_macOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v2', 10001) # type: ignore
MTLFeatureSet_OSX_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_GPUFamily1_v2', 10001) # type: ignore
MTLFeatureSet_macOS_ReadWriteTextureTier2 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_ReadWriteTextureTier2', 10002) # type: ignore
MTLFeatureSet_OSX_ReadWriteTextureTier2 = enum_MTLFeatureSet.define('MTLFeatureSet_OSX_ReadWriteTextureTier2', 10002) # type: ignore
MTLFeatureSet_macOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v3', 10003) # type: ignore
MTLFeatureSet_macOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily1_v4', 10004) # type: ignore
MTLFeatureSet_macOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_macOS_GPUFamily2_v1', 10005) # type: ignore
MTLFeatureSet_tvOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v1', 30000) # type: ignore
MTLFeatureSet_TVOS_GPUFamily1_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_TVOS_GPUFamily1_v1', 30000) # type: ignore
MTLFeatureSet_tvOS_GPUFamily1_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v2', 30001) # type: ignore
MTLFeatureSet_tvOS_GPUFamily1_v3 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v3', 30002) # type: ignore
MTLFeatureSet_tvOS_GPUFamily2_v1 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily2_v1', 30003) # type: ignore
MTLFeatureSet_tvOS_GPUFamily1_v4 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily1_v4', 30004) # type: ignore
MTLFeatureSet_tvOS_GPUFamily2_v2 = enum_MTLFeatureSet.define('MTLFeatureSet_tvOS_GPUFamily2_v2', 30005) # type: ignore

MTLFeatureSet: TypeAlias = enum_MTLFeatureSet
enum_MTLGPUFamily = CEnum(NSInteger)
MTLGPUFamilyApple1 = enum_MTLGPUFamily.define('MTLGPUFamilyApple1', 1001) # type: ignore
MTLGPUFamilyApple2 = enum_MTLGPUFamily.define('MTLGPUFamilyApple2', 1002) # type: ignore
MTLGPUFamilyApple3 = enum_MTLGPUFamily.define('MTLGPUFamilyApple3', 1003) # type: ignore
MTLGPUFamilyApple4 = enum_MTLGPUFamily.define('MTLGPUFamilyApple4', 1004) # type: ignore
MTLGPUFamilyApple5 = enum_MTLGPUFamily.define('MTLGPUFamilyApple5', 1005) # type: ignore
MTLGPUFamilyApple6 = enum_MTLGPUFamily.define('MTLGPUFamilyApple6', 1006) # type: ignore
MTLGPUFamilyApple7 = enum_MTLGPUFamily.define('MTLGPUFamilyApple7', 1007) # type: ignore
MTLGPUFamilyApple8 = enum_MTLGPUFamily.define('MTLGPUFamilyApple8', 1008) # type: ignore
MTLGPUFamilyApple9 = enum_MTLGPUFamily.define('MTLGPUFamilyApple9', 1009) # type: ignore
MTLGPUFamilyMac1 = enum_MTLGPUFamily.define('MTLGPUFamilyMac1', 2001) # type: ignore
MTLGPUFamilyMac2 = enum_MTLGPUFamily.define('MTLGPUFamilyMac2', 2002) # type: ignore
MTLGPUFamilyCommon1 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon1', 3001) # type: ignore
MTLGPUFamilyCommon2 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon2', 3002) # type: ignore
MTLGPUFamilyCommon3 = enum_MTLGPUFamily.define('MTLGPUFamilyCommon3', 3003) # type: ignore
MTLGPUFamilyMacCatalyst1 = enum_MTLGPUFamily.define('MTLGPUFamilyMacCatalyst1', 4001) # type: ignore
MTLGPUFamilyMacCatalyst2 = enum_MTLGPUFamily.define('MTLGPUFamilyMacCatalyst2', 4002) # type: ignore
MTLGPUFamilyMetal3 = enum_MTLGPUFamily.define('MTLGPUFamilyMetal3', 5001) # type: ignore

MTLGPUFamily: TypeAlias = enum_MTLGPUFamily
enum_MTLDeviceLocation = CEnum(NSUInteger)
MTLDeviceLocationBuiltIn = enum_MTLDeviceLocation.define('MTLDeviceLocationBuiltIn', 0) # type: ignore
MTLDeviceLocationSlot = enum_MTLDeviceLocation.define('MTLDeviceLocationSlot', 1) # type: ignore
MTLDeviceLocationExternal = enum_MTLDeviceLocation.define('MTLDeviceLocationExternal', 2) # type: ignore
MTLDeviceLocationUnspecified = enum_MTLDeviceLocation.define('MTLDeviceLocationUnspecified', -1) # type: ignore

MTLDeviceLocation: TypeAlias = enum_MTLDeviceLocation
enum_MTLPipelineOption = CEnum(NSUInteger)
MTLPipelineOptionNone = enum_MTLPipelineOption.define('MTLPipelineOptionNone', 0) # type: ignore
MTLPipelineOptionArgumentInfo = enum_MTLPipelineOption.define('MTLPipelineOptionArgumentInfo', 1) # type: ignore
MTLPipelineOptionBufferTypeInfo = enum_MTLPipelineOption.define('MTLPipelineOptionBufferTypeInfo', 2) # type: ignore
MTLPipelineOptionFailOnBinaryArchiveMiss = enum_MTLPipelineOption.define('MTLPipelineOptionFailOnBinaryArchiveMiss', 4) # type: ignore

MTLPipelineOption: TypeAlias = enum_MTLPipelineOption
enum_MTLReadWriteTextureTier = CEnum(NSUInteger)
MTLReadWriteTextureTierNone = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTierNone', 0) # type: ignore
MTLReadWriteTextureTier1 = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTier1', 1) # type: ignore
MTLReadWriteTextureTier2 = enum_MTLReadWriteTextureTier.define('MTLReadWriteTextureTier2', 2) # type: ignore

MTLReadWriteTextureTier: TypeAlias = enum_MTLReadWriteTextureTier
enum_MTLArgumentBuffersTier = CEnum(NSUInteger)
MTLArgumentBuffersTier1 = enum_MTLArgumentBuffersTier.define('MTLArgumentBuffersTier1', 0) # type: ignore
MTLArgumentBuffersTier2 = enum_MTLArgumentBuffersTier.define('MTLArgumentBuffersTier2', 1) # type: ignore

MTLArgumentBuffersTier: TypeAlias = enum_MTLArgumentBuffersTier
enum_MTLSparseTextureRegionAlignmentMode = CEnum(NSUInteger)
MTLSparseTextureRegionAlignmentModeOutward = enum_MTLSparseTextureRegionAlignmentMode.define('MTLSparseTextureRegionAlignmentModeOutward', 0) # type: ignore
MTLSparseTextureRegionAlignmentModeInward = enum_MTLSparseTextureRegionAlignmentMode.define('MTLSparseTextureRegionAlignmentModeInward', 1) # type: ignore

MTLSparseTextureRegionAlignmentMode: TypeAlias = enum_MTLSparseTextureRegionAlignmentMode
enum_MTLSparsePageSize = CEnum(NSInteger)
MTLSparsePageSize16 = enum_MTLSparsePageSize.define('MTLSparsePageSize16', 101) # type: ignore
MTLSparsePageSize64 = enum_MTLSparsePageSize.define('MTLSparsePageSize64', 102) # type: ignore
MTLSparsePageSize256 = enum_MTLSparsePageSize.define('MTLSparsePageSize256', 103) # type: ignore

MTLSparsePageSize: TypeAlias = enum_MTLSparsePageSize
@c.record
class MTLAccelerationStructureSizes(c.Struct):
  SIZE = 24
  accelerationStructureSize: Annotated[NSUInteger, 0]
  buildScratchBufferSize: Annotated[NSUInteger, 8]
  refitScratchBufferSize: Annotated[NSUInteger, 16]
enum_MTLCounterSamplingPoint = CEnum(NSUInteger)
MTLCounterSamplingPointAtStageBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtStageBoundary', 0) # type: ignore
MTLCounterSamplingPointAtDrawBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtDrawBoundary', 1) # type: ignore
MTLCounterSamplingPointAtDispatchBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtDispatchBoundary', 2) # type: ignore
MTLCounterSamplingPointAtTileDispatchBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtTileDispatchBoundary', 3) # type: ignore
MTLCounterSamplingPointAtBlitBoundary = enum_MTLCounterSamplingPoint.define('MTLCounterSamplingPointAtBlitBoundary', 4) # type: ignore

MTLCounterSamplingPoint: TypeAlias = enum_MTLCounterSamplingPoint
@c.record
class MTLSizeAndAlign(c.Struct):
  SIZE = 16
  size: Annotated[NSUInteger, 0]
  align: Annotated[NSUInteger, 8]
class MTLRenderPipelineReflection(objc.Spec): pass
class MTLArgumentDescriptor(objc.Spec): pass
MTLArgumentDescriptor._bases_ = [NSObject]
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
class MTLArchitecture(objc.Spec): pass
MTLArchitecture._bases_ = [NSObject]
MTLArchitecture._methods_ = [
  ('name', NSString, []),
]
class MTLHeapDescriptor(objc.Spec): pass
class MTLDepthStencilState(objc.Spec): pass
class MTLDepthStencilDescriptor(objc.Spec): pass
class struct___IOSurface(ctypes.Structure): pass
IOSurfaceRef = c.POINTER[struct___IOSurface]
class MTLSharedTextureHandle(objc.Spec): pass
MTLSharedTextureHandle._bases_ = [NSObject]
MTLSharedTextureHandle._methods_ = [
  ('device', MTLDevice, []),
  ('label', NSString, []),
]
class MTLSamplerDescriptor(objc.Spec): pass
class MTLLibrary(objc.Spec): pass
class MTLFunctionConstantValues(objc.Spec): pass
MTLFunctionConstantValues._bases_ = [NSObject]
MTLFunctionConstantValues._methods_ = [
  ('setConstantValue:type:atIndex:', None, [c.POINTER[None], MTLDataType, NSUInteger]),
  ('setConstantValues:type:withRange:', None, [c.POINTER[None], MTLDataType, NSRange]),
  ('setConstantValue:type:withName:', None, [c.POINTER[None], MTLDataType, NSString]),
  ('reset', None, []),
]
class MTLFunctionDescriptor(objc.Spec): pass
MTLFunctionDescriptor._bases_ = [NSObject]
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
class MTLIntersectionFunctionDescriptor(objc.Spec): pass
enum_MTLLibraryType = CEnum(NSInteger)
MTLLibraryTypeExecutable = enum_MTLLibraryType.define('MTLLibraryTypeExecutable', 0) # type: ignore
MTLLibraryTypeDynamic = enum_MTLLibraryType.define('MTLLibraryTypeDynamic', 1) # type: ignore

MTLLibraryType: TypeAlias = enum_MTLLibraryType
MTLLibrary._bases_ = [NSObject]
MTLLibrary._methods_ = [
  ('newFunctionWithName:', MTLFunction, [NSString], True),
  ('newFunctionWithName:constantValues:error:', MTLFunction, [NSString, MTLFunctionConstantValues, c.POINTER[NSError]], True),
  ('newFunctionWithDescriptor:error:', MTLFunction, [MTLFunctionDescriptor, c.POINTER[NSError]], True),
  ('newIntersectionFunctionWithDescriptor:error:', MTLFunction, [MTLIntersectionFunctionDescriptor, c.POINTER[NSError]], True),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('device', MTLDevice, []),
  ('type', MTLLibraryType, []),
  ('installName', NSString, []),
]
class NSBundle(objc.Spec): pass
class NSURL(objc.Spec): pass
NSURLResourceKey = NSString
enum_NSURLBookmarkCreationOptions = CEnum(NSUInteger)
NSURLBookmarkCreationPreferFileIDResolution = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationPreferFileIDResolution', 256) # type: ignore
NSURLBookmarkCreationMinimalBookmark = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationMinimalBookmark', 512) # type: ignore
NSURLBookmarkCreationSuitableForBookmarkFile = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationSuitableForBookmarkFile', 1024) # type: ignore
NSURLBookmarkCreationWithSecurityScope = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationWithSecurityScope', 2048) # type: ignore
NSURLBookmarkCreationSecurityScopeAllowOnlyReadAccess = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationSecurityScopeAllowOnlyReadAccess', 4096) # type: ignore
NSURLBookmarkCreationWithoutImplicitSecurityScope = enum_NSURLBookmarkCreationOptions.define('NSURLBookmarkCreationWithoutImplicitSecurityScope', 536870912) # type: ignore

NSURLBookmarkCreationOptions: TypeAlias = enum_NSURLBookmarkCreationOptions
enum_NSURLBookmarkResolutionOptions = CEnum(NSUInteger)
NSURLBookmarkResolutionWithoutUI = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutUI', 256) # type: ignore
NSURLBookmarkResolutionWithoutMounting = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutMounting', 512) # type: ignore
NSURLBookmarkResolutionWithSecurityScope = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithSecurityScope', 1024) # type: ignore
NSURLBookmarkResolutionWithoutImplicitStartAccessing = enum_NSURLBookmarkResolutionOptions.define('NSURLBookmarkResolutionWithoutImplicitStartAccessing', 32768) # type: ignore

NSURLBookmarkResolutionOptions: TypeAlias = enum_NSURLBookmarkResolutionOptions
class NSNumber(objc.Spec): pass
enum_NSComparisonResult = CEnum(NSInteger)
NSOrderedAscending = enum_NSComparisonResult.define('NSOrderedAscending', -1) # type: ignore
NSOrderedSame = enum_NSComparisonResult.define('NSOrderedSame', 0) # type: ignore
NSOrderedDescending = enum_NSComparisonResult.define('NSOrderedDescending', 1) # type: ignore

NSComparisonResult: TypeAlias = enum_NSComparisonResult
class NSValue(objc.Spec): pass
NSValue._bases_ = [NSObject]
NSValue._methods_ = [
  ('getValue:size:', None, [c.POINTER[None], NSUInteger]),
  ('initWithBytes:objCType:', 'instancetype', [c.POINTER[None], c.POINTER[Annotated[bytes, ctypes.c_char]]]),
  ('initWithCoder:', 'instancetype', [NSCoder]),
  ('objCType', c.POINTER[Annotated[bytes, ctypes.c_char]], []),
]
NSNumber._bases_ = [NSValue]
NSNumber._methods_ = [
  ('initWithCoder:', 'instancetype', [NSCoder]),
  ('initWithChar:', NSNumber, [Annotated[bytes, ctypes.c_char]]),
  ('initWithUnsignedChar:', NSNumber, [Annotated[int, ctypes.c_ubyte]]),
  ('initWithShort:', NSNumber, [Annotated[int, ctypes.c_int16]]),
  ('initWithUnsignedShort:', NSNumber, [Annotated[int, ctypes.c_uint16]]),
  ('initWithInt:', NSNumber, [Annotated[int, ctypes.c_int32]]),
  ('initWithUnsignedInt:', NSNumber, [Annotated[int, ctypes.c_uint32]]),
  ('initWithLong:', NSNumber, [Annotated[int, ctypes.c_int64]]),
  ('initWithUnsignedLong:', NSNumber, [Annotated[int, ctypes.c_uint64]]),
  ('initWithLongLong:', NSNumber, [Annotated[int, ctypes.c_int64]]),
  ('initWithUnsignedLongLong:', NSNumber, [Annotated[int, ctypes.c_uint64]]),
  ('initWithFloat:', NSNumber, [Annotated[float, ctypes.c_float]]),
  ('initWithDouble:', NSNumber, [Annotated[float, ctypes.c_double]]),
  ('initWithBool:', NSNumber, [BOOL]),
  ('initWithInteger:', NSNumber, [NSInteger]),
  ('initWithUnsignedInteger:', NSNumber, [NSUInteger]),
  ('compare:', NSComparisonResult, [NSNumber]),
  ('isEqualToNumber:', BOOL, [NSNumber]),
  ('descriptionWithLocale:', NSString, [objc.id_]),
  ('charValue', Annotated[bytes, ctypes.c_char], []),
  ('unsignedCharValue', Annotated[int, ctypes.c_ubyte], []),
  ('shortValue', Annotated[int, ctypes.c_int16], []),
  ('unsignedShortValue', Annotated[int, ctypes.c_uint16], []),
  ('intValue', Annotated[int, ctypes.c_int32], []),
  ('unsignedIntValue', Annotated[int, ctypes.c_uint32], []),
  ('longValue', Annotated[int, ctypes.c_int64], []),
  ('unsignedLongValue', Annotated[int, ctypes.c_uint64], []),
  ('longLongValue', Annotated[int, ctypes.c_int64], []),
  ('unsignedLongLongValue', Annotated[int, ctypes.c_uint64], []),
  ('floatValue', Annotated[float, ctypes.c_float], []),
  ('doubleValue', Annotated[float, ctypes.c_double], []),
  ('boolValue', BOOL, []),
  ('integerValue', NSInteger, []),
  ('unsignedIntegerValue', NSUInteger, []),
  ('stringValue', NSString, []),
]
NSURLBookmarkFileCreationOptions = Annotated[int, ctypes.c_uint64]
NSURL._bases_ = [NSObject]
NSURL._methods_ = [
  ('initWithScheme:host:path:', 'instancetype', [NSString, NSString, NSString]),
  ('initFileURLWithPath:isDirectory:relativeToURL:', 'instancetype', [NSString, BOOL, NSURL]),
  ('initFileURLWithPath:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('initFileURLWithPath:isDirectory:', 'instancetype', [NSString, BOOL]),
  ('initFileURLWithPath:', 'instancetype', [NSString]),
  ('initFileURLWithFileSystemRepresentation:isDirectory:relativeToURL:', 'instancetype', [c.POINTER[Annotated[bytes, ctypes.c_char]], BOOL, NSURL]),
  ('initWithString:', 'instancetype', [NSString]),
  ('initWithString:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('initWithString:encodingInvalidCharacters:', 'instancetype', [NSString, BOOL]),
  ('initWithDataRepresentation:relativeToURL:', 'instancetype', [NSData, NSURL]),
  ('initAbsoluteURLWithDataRepresentation:relativeToURL:', 'instancetype', [NSData, NSURL]),
  ('getFileSystemRepresentation:maxLength:', BOOL, [c.POINTER[Annotated[bytes, ctypes.c_char]], NSUInteger]),
  ('isFileReferenceURL', BOOL, []),
  ('fileReferenceURL', NSURL, []),
  ('getResourceValue:forKey:error:', BOOL, [c.POINTER[objc.id_], NSURLResourceKey, c.POINTER[NSError]]),
  ('setResourceValue:forKey:error:', BOOL, [objc.id_, NSURLResourceKey, c.POINTER[NSError]]),
  ('removeCachedResourceValueForKey:', None, [NSURLResourceKey]),
  ('removeAllCachedResourceValues', None, []),
  ('setTemporaryResourceValue:forKey:', None, [objc.id_, NSURLResourceKey]),
  ('initByResolvingBookmarkData:options:relativeToURL:bookmarkDataIsStale:error:', 'instancetype', [NSData, NSURLBookmarkResolutionOptions, NSURL, c.POINTER[BOOL], c.POINTER[NSError]]),
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
  ('fileSystemRepresentation', c.POINTER[Annotated[bytes, ctypes.c_char]], []),
  ('isFileURL', BOOL, []),
  ('standardizedURL', NSURL, []),
  ('filePathURL', NSURL, []),
]
NSURL._classmethods_ = [
  ('fileURLWithPath:isDirectory:relativeToURL:', NSURL, [NSString, BOOL, NSURL]),
  ('fileURLWithPath:relativeToURL:', NSURL, [NSString, NSURL]),
  ('fileURLWithPath:isDirectory:', NSURL, [NSString, BOOL]),
  ('fileURLWithPath:', NSURL, [NSString]),
  ('fileURLWithFileSystemRepresentation:isDirectory:relativeToURL:', NSURL, [c.POINTER[Annotated[bytes, ctypes.c_char]], BOOL, NSURL]),
  ('URLWithString:', 'instancetype', [NSString]),
  ('URLWithString:relativeToURL:', 'instancetype', [NSString, NSURL]),
  ('URLWithString:encodingInvalidCharacters:', 'instancetype', [NSString, BOOL]),
  ('URLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('absoluteURLWithDataRepresentation:relativeToURL:', NSURL, [NSData, NSURL]),
  ('URLByResolvingBookmarkData:options:relativeToURL:bookmarkDataIsStale:error:', 'instancetype', [NSData, NSURLBookmarkResolutionOptions, NSURL, c.POINTER[BOOL], c.POINTER[NSError]]),
  ('writeBookmarkData:toURL:options:error:', BOOL, [NSData, NSURL, NSURLBookmarkFileCreationOptions, c.POINTER[NSError]]),
  ('bookmarkDataWithContentsOfURL:error:', NSData, [NSURL, c.POINTER[NSError]]),
  ('URLByResolvingAliasFileAtURL:options:error:', 'instancetype', [NSURL, NSURLBookmarkResolutionOptions, c.POINTER[NSError]]),
]
class NSAttributedString(objc.Spec): pass
NSAttributedString._bases_ = [NSObject]
NSAttributedString._methods_ = [
  ('string', NSString, []),
]
NSBundle._bases_ = [NSObject]
NSBundle._methods_ = [
  ('initWithPath:', 'instancetype', [NSString]),
  ('initWithURL:', 'instancetype', [NSURL]),
  ('load', BOOL, []),
  ('unload', BOOL, []),
  ('preflightAndReturnError:', BOOL, [c.POINTER[NSError]]),
  ('loadAndReturnError:', BOOL, [c.POINTER[NSError]]),
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
  ('bundleWithPath:', 'instancetype', [NSString]),
  ('bundleWithURL:', 'instancetype', [NSURL]),
  ('bundleWithIdentifier:', NSBundle, [NSString]),
  ('URLForResource:withExtension:subdirectory:inBundleWithURL:', NSURL, [NSString, NSString, NSString, NSURL]),
  ('pathForResource:ofType:inDirectory:', NSString, [NSString, NSString, NSString]),
  ('mainBundle', NSBundle, []),
]
class MTLCompileOptions(objc.Spec): pass
enum_MTLLanguageVersion = CEnum(NSUInteger)
MTLLanguageVersion1_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_0', 65536) # type: ignore
MTLLanguageVersion1_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_1', 65537) # type: ignore
MTLLanguageVersion1_2 = enum_MTLLanguageVersion.define('MTLLanguageVersion1_2', 65538) # type: ignore
MTLLanguageVersion2_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_0', 131072) # type: ignore
MTLLanguageVersion2_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_1', 131073) # type: ignore
MTLLanguageVersion2_2 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_2', 131074) # type: ignore
MTLLanguageVersion2_3 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_3', 131075) # type: ignore
MTLLanguageVersion2_4 = enum_MTLLanguageVersion.define('MTLLanguageVersion2_4', 131076) # type: ignore
MTLLanguageVersion3_0 = enum_MTLLanguageVersion.define('MTLLanguageVersion3_0', 196608) # type: ignore
MTLLanguageVersion3_1 = enum_MTLLanguageVersion.define('MTLLanguageVersion3_1', 196609) # type: ignore

MTLLanguageVersion: TypeAlias = enum_MTLLanguageVersion
enum_MTLLibraryOptimizationLevel = CEnum(NSInteger)
MTLLibraryOptimizationLevelDefault = enum_MTLLibraryOptimizationLevel.define('MTLLibraryOptimizationLevelDefault', 0) # type: ignore
MTLLibraryOptimizationLevelSize = enum_MTLLibraryOptimizationLevel.define('MTLLibraryOptimizationLevelSize', 1) # type: ignore

MTLLibraryOptimizationLevel: TypeAlias = enum_MTLLibraryOptimizationLevel
enum_MTLCompileSymbolVisibility = CEnum(NSInteger)
MTLCompileSymbolVisibilityDefault = enum_MTLCompileSymbolVisibility.define('MTLCompileSymbolVisibilityDefault', 0) # type: ignore
MTLCompileSymbolVisibilityHidden = enum_MTLCompileSymbolVisibility.define('MTLCompileSymbolVisibilityHidden', 1) # type: ignore

MTLCompileSymbolVisibility: TypeAlias = enum_MTLCompileSymbolVisibility
MTLCompileOptions._bases_ = [NSObject]
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
class MTLRenderPipelineState(objc.Spec): pass
class MTLRenderPipelineDescriptor(objc.Spec): pass
class MTLTileRenderPipelineDescriptor(objc.Spec): pass
class MTLMeshRenderPipelineDescriptor(objc.Spec): pass
class MTLRasterizationRateMapDescriptor(objc.Spec): pass
class MTLIndirectCommandBufferDescriptor(objc.Spec): pass
class MTLSharedEvent(objc.Spec): pass
class MTLSharedEventHandle(objc.Spec): pass
class MTLIOFileHandle(objc.Spec): pass
class MTLIOCommandQueue(objc.Spec): pass
class MTLIOCommandQueueDescriptor(objc.Spec): pass
class MTLCounterSampleBufferDescriptor(objc.Spec): pass
class MTLCounterSet(objc.Spec): pass
MTLCounterSet._bases_ = [NSObject]
MTLCounterSet._methods_ = [
  ('name', NSString, []),
]
MTLCounterSampleBufferDescriptor._bases_ = [NSObject]
MTLCounterSampleBufferDescriptor._methods_ = [
  ('counterSet', MTLCounterSet, []),
  ('setCounterSet:', None, [MTLCounterSet]),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
  ('storageMode', MTLStorageMode, []),
  ('setStorageMode:', None, [MTLStorageMode]),
  ('sampleCount', NSUInteger, []),
  ('setSampleCount:', None, [NSUInteger]),
]
MTLTimestamp = Annotated[int, ctypes.c_uint64]
class MTLBufferBinding(objc.Spec): pass
class MTLBinding(objc.Spec): pass
MTLBufferBinding._bases_ = [MTLBinding]
MTLBufferBinding._methods_ = [
  ('bufferAlignment', NSUInteger, []),
  ('bufferDataSize', NSUInteger, []),
  ('bufferDataType', MTLDataType, []),
  ('bufferStructType', MTLStructType, []),
  ('bufferPointerType', MTLPointerType, []),
]
class MTLDynamicLibrary(objc.Spec): pass
class MTLBinaryArchive(objc.Spec): pass
class MTLBinaryArchiveDescriptor(objc.Spec): pass
class MTLAccelerationStructureDescriptor(objc.Spec): pass
MTLDevice._bases_ = [NSObject]
MTLDevice._methods_ = [
  ('newCommandQueue', MTLCommandQueue, [], True),
  ('newCommandQueueWithMaxCommandBufferCount:', MTLCommandQueue, [NSUInteger], True),
  ('heapTextureSizeAndAlignWithDescriptor:', MTLSizeAndAlign, [MTLTextureDescriptor]),
  ('heapBufferSizeAndAlignWithLength:options:', MTLSizeAndAlign, [NSUInteger, MTLResourceOptions]),
  ('newHeapWithDescriptor:', MTLHeap, [MTLHeapDescriptor], True),
  ('newBufferWithLength:options:', MTLBuffer, [NSUInteger, MTLResourceOptions], True),
  ('newBufferWithBytes:length:options:', MTLBuffer, [c.POINTER[None], NSUInteger, MTLResourceOptions], True),
  ('newDepthStencilStateWithDescriptor:', MTLDepthStencilState, [MTLDepthStencilDescriptor], True),
  ('newTextureWithDescriptor:', MTLTexture, [MTLTextureDescriptor], True),
  ('newTextureWithDescriptor:iosurface:plane:', MTLTexture, [MTLTextureDescriptor, IOSurfaceRef, NSUInteger], True),
  ('newSharedTextureWithDescriptor:', MTLTexture, [MTLTextureDescriptor], True),
  ('newSharedTextureWithHandle:', MTLTexture, [MTLSharedTextureHandle], True),
  ('newSamplerStateWithDescriptor:', MTLSamplerState, [MTLSamplerDescriptor], True),
  ('newDefaultLibrary', MTLLibrary, [], True),
  ('newDefaultLibraryWithBundle:error:', MTLLibrary, [NSBundle, c.POINTER[NSError]], True),
  ('newLibraryWithFile:error:', MTLLibrary, [NSString, c.POINTER[NSError]], True),
  ('newLibraryWithURL:error:', MTLLibrary, [NSURL, c.POINTER[NSError]], True),
  ('newLibraryWithData:error:', MTLLibrary, [objc.id_, c.POINTER[NSError]], True),
  ('newLibraryWithSource:options:error:', MTLLibrary, [NSString, MTLCompileOptions, c.POINTER[NSError]], True),
  ('newLibraryWithStitchedDescriptor:error:', MTLLibrary, [MTLStitchedLibraryDescriptor, c.POINTER[NSError]], True),
  ('newRenderPipelineStateWithDescriptor:error:', MTLRenderPipelineState, [MTLRenderPipelineDescriptor, c.POINTER[NSError]], True),
  ('newRenderPipelineStateWithDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLRenderPipelineDescriptor, MTLPipelineOption, c.POINTER[MTLRenderPipelineReflection], c.POINTER[NSError]], True),
  ('newComputePipelineStateWithFunction:error:', MTLComputePipelineState, [MTLFunction, c.POINTER[NSError]], True),
  ('newComputePipelineStateWithFunction:options:reflection:error:', MTLComputePipelineState, [MTLFunction, MTLPipelineOption, c.POINTER[MTLComputePipelineReflection], c.POINTER[NSError]], True),
  ('newComputePipelineStateWithDescriptor:options:reflection:error:', MTLComputePipelineState, [MTLComputePipelineDescriptor, MTLPipelineOption, c.POINTER[MTLComputePipelineReflection], c.POINTER[NSError]], True),
  ('newFence', MTLFence, [], True),
  ('supportsFeatureSet:', BOOL, [MTLFeatureSet]),
  ('supportsFamily:', BOOL, [MTLGPUFamily]),
  ('supportsTextureSampleCount:', BOOL, [NSUInteger]),
  ('minimumLinearTextureAlignmentForPixelFormat:', NSUInteger, [MTLPixelFormat]),
  ('minimumTextureBufferAlignmentForPixelFormat:', NSUInteger, [MTLPixelFormat]),
  ('newRenderPipelineStateWithTileDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLTileRenderPipelineDescriptor, MTLPipelineOption, c.POINTER[MTLRenderPipelineReflection], c.POINTER[NSError]], True),
  ('newRenderPipelineStateWithMeshDescriptor:options:reflection:error:', MTLRenderPipelineState, [MTLMeshRenderPipelineDescriptor, MTLPipelineOption, c.POINTER[MTLRenderPipelineReflection], c.POINTER[NSError]], True),
  ('getDefaultSamplePositions:count:', None, [c.POINTER[MTLSamplePosition], NSUInteger]),
  ('supportsRasterizationRateMapWithLayerCount:', BOOL, [NSUInteger]),
  ('newRasterizationRateMapWithDescriptor:', MTLRasterizationRateMap, [MTLRasterizationRateMapDescriptor], True),
  ('newIndirectCommandBufferWithDescriptor:maxCommandCount:options:', MTLIndirectCommandBuffer, [MTLIndirectCommandBufferDescriptor, NSUInteger, MTLResourceOptions], True),
  ('newEvent', MTLEvent, [], True),
  ('newSharedEvent', MTLSharedEvent, [], True),
  ('newSharedEventWithHandle:', MTLSharedEvent, [MTLSharedEventHandle], True),
  ('newIOHandleWithURL:error:', MTLIOFileHandle, [NSURL, c.POINTER[NSError]], True),
  ('newIOCommandQueueWithDescriptor:error:', MTLIOCommandQueue, [MTLIOCommandQueueDescriptor, c.POINTER[NSError]], True),
  ('newIOHandleWithURL:compressionMethod:error:', MTLIOFileHandle, [NSURL, MTLIOCompressionMethod, c.POINTER[NSError]], True),
  ('newIOFileHandleWithURL:error:', MTLIOFileHandle, [NSURL, c.POINTER[NSError]], True),
  ('newIOFileHandleWithURL:compressionMethod:error:', MTLIOFileHandle, [NSURL, MTLIOCompressionMethod, c.POINTER[NSError]], True),
  ('sparseTileSizeWithTextureType:pixelFormat:sampleCount:', MTLSize, [MTLTextureType, MTLPixelFormat, NSUInteger]),
  ('convertSparsePixelRegions:toTileRegions:withTileSize:alignmentMode:numRegions:', None, [c.POINTER[MTLRegion], c.POINTER[MTLRegion], MTLSize, MTLSparseTextureRegionAlignmentMode, NSUInteger]),
  ('convertSparseTileRegions:toPixelRegions:withTileSize:numRegions:', None, [c.POINTER[MTLRegion], c.POINTER[MTLRegion], MTLSize, NSUInteger]),
  ('sparseTileSizeInBytesForSparsePageSize:', NSUInteger, [MTLSparsePageSize]),
  ('sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:', MTLSize, [MTLTextureType, MTLPixelFormat, NSUInteger, MTLSparsePageSize]),
  ('newCounterSampleBufferWithDescriptor:error:', MTLCounterSampleBuffer, [MTLCounterSampleBufferDescriptor, c.POINTER[NSError]], True),
  ('sampleTimestamps:gpuTimestamp:', None, [c.POINTER[MTLTimestamp], c.POINTER[MTLTimestamp]]),
  ('newArgumentEncoderWithBufferBinding:', MTLArgumentEncoder, [MTLBufferBinding], True),
  ('supportsCounterSampling:', BOOL, [MTLCounterSamplingPoint]),
  ('supportsVertexAmplificationCount:', BOOL, [NSUInteger]),
  ('newDynamicLibrary:error:', MTLDynamicLibrary, [MTLLibrary, c.POINTER[NSError]], True),
  ('newDynamicLibraryWithURL:error:', MTLDynamicLibrary, [NSURL, c.POINTER[NSError]], True),
  ('newBinaryArchiveWithDescriptor:error:', MTLBinaryArchive, [MTLBinaryArchiveDescriptor, c.POINTER[NSError]], True),
  ('accelerationStructureSizesWithDescriptor:', MTLAccelerationStructureSizes, [MTLAccelerationStructureDescriptor]),
  ('newAccelerationStructureWithSize:', MTLAccelerationStructure, [NSUInteger], True),
  ('newAccelerationStructureWithDescriptor:', MTLAccelerationStructure, [MTLAccelerationStructureDescriptor], True),
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
enum_MTLIndirectCommandType = CEnum(NSUInteger)
MTLIndirectCommandTypeDraw = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDraw', 1) # type: ignore
MTLIndirectCommandTypeDrawIndexed = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawIndexed', 2) # type: ignore
MTLIndirectCommandTypeDrawPatches = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawPatches', 4) # type: ignore
MTLIndirectCommandTypeDrawIndexedPatches = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawIndexedPatches', 8) # type: ignore
MTLIndirectCommandTypeConcurrentDispatch = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeConcurrentDispatch', 32) # type: ignore
MTLIndirectCommandTypeConcurrentDispatchThreads = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeConcurrentDispatchThreads', 64) # type: ignore
MTLIndirectCommandTypeDrawMeshThreadgroups = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawMeshThreadgroups', 128) # type: ignore
MTLIndirectCommandTypeDrawMeshThreads = enum_MTLIndirectCommandType.define('MTLIndirectCommandTypeDrawMeshThreads', 256) # type: ignore

MTLIndirectCommandType: TypeAlias = enum_MTLIndirectCommandType
@c.record
class MTLIndirectCommandBufferExecutionRange(c.Struct):
  SIZE = 8
  location: Annotated[uint32_t, 0]
  length: Annotated[uint32_t, 4]
MTLIndirectCommandBufferDescriptor._bases_ = [NSObject]
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
  ('maxObjectBufferBindCount', NSUInteger, []),
  ('setMaxObjectBufferBindCount:', None, [NSUInteger]),
  ('maxMeshBufferBindCount', NSUInteger, []),
  ('setMaxMeshBufferBindCount:', None, [NSUInteger]),
  ('maxObjectThreadgroupMemoryBindCount', NSUInteger, []),
  ('setMaxObjectThreadgroupMemoryBindCount:', None, [NSUInteger]),
  ('supportRayTracing', BOOL, []),
  ('setSupportRayTracing:', None, [BOOL]),
  ('supportDynamicAttributeStride', BOOL, []),
  ('setSupportDynamicAttributeStride:', None, [BOOL]),
]
class MTLIndirectRenderCommand(objc.Spec): pass
enum_MTLPrimitiveType = CEnum(NSUInteger)
MTLPrimitiveTypePoint = enum_MTLPrimitiveType.define('MTLPrimitiveTypePoint', 0) # type: ignore
MTLPrimitiveTypeLine = enum_MTLPrimitiveType.define('MTLPrimitiveTypeLine', 1) # type: ignore
MTLPrimitiveTypeLineStrip = enum_MTLPrimitiveType.define('MTLPrimitiveTypeLineStrip', 2) # type: ignore
MTLPrimitiveTypeTriangle = enum_MTLPrimitiveType.define('MTLPrimitiveTypeTriangle', 3) # type: ignore
MTLPrimitiveTypeTriangleStrip = enum_MTLPrimitiveType.define('MTLPrimitiveTypeTriangleStrip', 4) # type: ignore

MTLPrimitiveType: TypeAlias = enum_MTLPrimitiveType
MTLIndirectRenderCommand._bases_ = [NSObject]
MTLIndirectRenderCommand._methods_ = [
  ('setRenderPipelineState:', None, [MTLRenderPipelineState]),
  ('setVertexBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setFragmentBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setVertexBuffer:offset:attributeStride:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:', None, [NSUInteger, NSUInteger, NSUInteger, MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLBuffer, NSUInteger, NSUInteger]),
  ('drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:', None, [NSUInteger, NSUInteger, NSUInteger, MTLBuffer, NSUInteger, MTLBuffer, NSUInteger, NSUInteger, NSUInteger, MTLBuffer, NSUInteger, NSUInteger]),
  ('drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:', None, [MTLPrimitiveType, NSUInteger, NSUInteger, NSUInteger, NSUInteger]),
  ('drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:', None, [MTLPrimitiveType, NSUInteger, MTLIndexType, MTLBuffer, NSUInteger, NSUInteger, NSInteger, NSUInteger]),
  ('setObjectThreadgroupMemoryLength:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setObjectBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setMeshBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:', None, [MTLSize, MTLSize, MTLSize]),
  ('drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:', None, [MTLSize, MTLSize, MTLSize]),
  ('setBarrier', None, []),
  ('clearBarrier', None, []),
  ('reset', None, []),
]
class MTLIndirectComputeCommand(objc.Spec): pass
MTLIndirectComputeCommand._bases_ = [NSObject]
MTLIndirectComputeCommand._methods_ = [
  ('setComputePipelineState:', None, [MTLComputePipelineState]),
  ('setKernelBuffer:offset:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger]),
  ('setKernelBuffer:offset:attributeStride:atIndex:', None, [MTLBuffer, NSUInteger, NSUInteger, NSUInteger]),
  ('concurrentDispatchThreadgroups:threadsPerThreadgroup:', None, [MTLSize, MTLSize]),
  ('concurrentDispatchThreads:threadsPerThreadgroup:', None, [MTLSize, MTLSize]),
  ('setBarrier', None, []),
  ('clearBarrier', None, []),
  ('setImageblockWidth:height:', None, [NSUInteger, NSUInteger]),
  ('reset', None, []),
  ('setThreadgroupMemoryLength:atIndex:', None, [NSUInteger, NSUInteger]),
  ('setStageInRegion:', None, [MTLRegion]),
]
MTLIndirectCommandBuffer._bases_ = [MTLResource]
MTLIndirectCommandBuffer._methods_ = [
  ('resetWithRange:', None, [NSRange]),
  ('indirectRenderCommandAtIndex:', MTLIndirectRenderCommand, [NSUInteger]),
  ('indirectComputeCommandAtIndex:', MTLIndirectComputeCommand, [NSUInteger]),
  ('size', NSUInteger, []),
  ('gpuResourceID', MTLResourceID, []),
]
MTLCommandEncoder._bases_ = [NSObject]
MTLCommandEncoder._methods_ = [
  ('endEncoding', None, []),
  ('insertDebugSignpost:', None, [NSString]),
  ('pushDebugGroup:', None, [NSString]),
  ('popDebugGroup', None, []),
  ('device', MTLDevice, []),
  ('label', NSString, []),
  ('setLabel:', None, [NSString]),
]
c.init_records()
MTLResourceCPUCacheModeShift = 0 # type: ignore
MTLResourceCPUCacheModeMask = (0xf << MTLResourceCPUCacheModeShift) # type: ignore
MTLResourceStorageModeShift = 4 # type: ignore
MTLResourceStorageModeMask = (0xf << MTLResourceStorageModeShift) # type: ignore
MTLResourceHazardTrackingModeShift = 8 # type: ignore
MTLResourceHazardTrackingModeMask = (0x3 << MTLResourceHazardTrackingModeShift) # type: ignore