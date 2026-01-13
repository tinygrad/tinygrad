# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
import os
dll = c.DLL('hsa', [os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libhsa-runtime64.so', 'hsa-runtime64'])
enum_SQ_RSRC_BUF_TYPE = CEnum(Annotated[int, ctypes.c_uint32])
SQ_RSRC_BUF = enum_SQ_RSRC_BUF_TYPE.define('SQ_RSRC_BUF', 0) # type: ignore
SQ_RSRC_BUF_RSVD_1 = enum_SQ_RSRC_BUF_TYPE.define('SQ_RSRC_BUF_RSVD_1', 1) # type: ignore
SQ_RSRC_BUF_RSVD_2 = enum_SQ_RSRC_BUF_TYPE.define('SQ_RSRC_BUF_RSVD_2', 2) # type: ignore
SQ_RSRC_BUF_RSVD_3 = enum_SQ_RSRC_BUF_TYPE.define('SQ_RSRC_BUF_RSVD_3', 3) # type: ignore

SQ_RSRC_BUF_TYPE: TypeAlias = enum_SQ_RSRC_BUF_TYPE
enum_BUF_DATA_FORMAT = CEnum(Annotated[int, ctypes.c_uint32])
BUF_DATA_FORMAT_INVALID = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_INVALID', 0) # type: ignore
BUF_DATA_FORMAT_8 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_8', 1) # type: ignore
BUF_DATA_FORMAT_16 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_16', 2) # type: ignore
BUF_DATA_FORMAT_8_8 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_8_8', 3) # type: ignore
BUF_DATA_FORMAT_32 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_32', 4) # type: ignore
BUF_DATA_FORMAT_16_16 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_16_16', 5) # type: ignore
BUF_DATA_FORMAT_10_11_11 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_10_11_11', 6) # type: ignore
BUF_DATA_FORMAT_11_11_10 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_11_11_10', 7) # type: ignore
BUF_DATA_FORMAT_10_10_10_2 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_10_10_10_2', 8) # type: ignore
BUF_DATA_FORMAT_2_10_10_10 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_2_10_10_10', 9) # type: ignore
BUF_DATA_FORMAT_8_8_8_8 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_8_8_8_8', 10) # type: ignore
BUF_DATA_FORMAT_32_32 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_32_32', 11) # type: ignore
BUF_DATA_FORMAT_16_16_16_16 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_16_16_16_16', 12) # type: ignore
BUF_DATA_FORMAT_32_32_32 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_32_32_32', 13) # type: ignore
BUF_DATA_FORMAT_32_32_32_32 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_32_32_32_32', 14) # type: ignore
BUF_DATA_FORMAT_RESERVED_15 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_RESERVED_15', 15) # type: ignore

BUF_DATA_FORMAT: TypeAlias = enum_BUF_DATA_FORMAT
enum_BUF_NUM_FORMAT = CEnum(Annotated[int, ctypes.c_uint32])
BUF_NUM_FORMAT_UNORM = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_UNORM', 0) # type: ignore
BUF_NUM_FORMAT_SNORM = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_SNORM', 1) # type: ignore
BUF_NUM_FORMAT_USCALED = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_USCALED', 2) # type: ignore
BUF_NUM_FORMAT_SSCALED = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_SSCALED', 3) # type: ignore
BUF_NUM_FORMAT_UINT = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_UINT', 4) # type: ignore
BUF_NUM_FORMAT_SINT = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_SINT', 5) # type: ignore
BUF_NUM_FORMAT_SNORM_OGL__SI__CI = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_SNORM_OGL__SI__CI', 6) # type: ignore
BUF_NUM_FORMAT_RESERVED_6__VI = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_RESERVED_6__VI', 6) # type: ignore
BUF_NUM_FORMAT_FLOAT = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_FLOAT', 7) # type: ignore

BUF_NUM_FORMAT: TypeAlias = enum_BUF_NUM_FORMAT
enum_BUF_FORMAT = CEnum(Annotated[int, ctypes.c_uint32])
BUF_FORMAT_32_UINT = enum_BUF_FORMAT.define('BUF_FORMAT_32_UINT', 20) # type: ignore

BUF_FORMAT: TypeAlias = enum_BUF_FORMAT
enum_SQ_SEL_XYZW01 = CEnum(Annotated[int, ctypes.c_uint32])
SQ_SEL_0 = enum_SQ_SEL_XYZW01.define('SQ_SEL_0', 0) # type: ignore
SQ_SEL_1 = enum_SQ_SEL_XYZW01.define('SQ_SEL_1', 1) # type: ignore
SQ_SEL_RESERVED_0 = enum_SQ_SEL_XYZW01.define('SQ_SEL_RESERVED_0', 2) # type: ignore
SQ_SEL_RESERVED_1 = enum_SQ_SEL_XYZW01.define('SQ_SEL_RESERVED_1', 3) # type: ignore
SQ_SEL_X = enum_SQ_SEL_XYZW01.define('SQ_SEL_X', 4) # type: ignore
SQ_SEL_Y = enum_SQ_SEL_XYZW01.define('SQ_SEL_Y', 5) # type: ignore
SQ_SEL_Z = enum_SQ_SEL_XYZW01.define('SQ_SEL_Z', 6) # type: ignore
SQ_SEL_W = enum_SQ_SEL_XYZW01.define('SQ_SEL_W', 7) # type: ignore

SQ_SEL_XYZW01: TypeAlias = enum_SQ_SEL_XYZW01
@c.record
class union_COMPUTE_TMPRING_SIZE(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_COMPUTE_TMPRING_SIZE_bitfields, 0]
  bits: Annotated[union_COMPUTE_TMPRING_SIZE_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_COMPUTE_TMPRING_SIZE_bitfields(c.Struct):
  SIZE = 4
  WAVES: Annotated[Annotated[int, ctypes.c_uint32], 0, 12, 0]
  WAVESIZE: Annotated[Annotated[int, ctypes.c_uint32], 1, 13, 4]
@c.record
class union_COMPUTE_TMPRING_SIZE_GFX11(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_COMPUTE_TMPRING_SIZE_GFX11_bitfields, 0]
  bits: Annotated[union_COMPUTE_TMPRING_SIZE_GFX11_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_COMPUTE_TMPRING_SIZE_GFX11_bitfields(c.Struct):
  SIZE = 4
  WAVES: Annotated[Annotated[int, ctypes.c_uint32], 0, 12, 0]
  WAVESIZE: Annotated[Annotated[int, ctypes.c_uint32], 1, 15, 4]
@c.record
class union_COMPUTE_TMPRING_SIZE_GFX12(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_COMPUTE_TMPRING_SIZE_GFX12_bitfields, 0]
  bits: Annotated[union_COMPUTE_TMPRING_SIZE_GFX12_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_COMPUTE_TMPRING_SIZE_GFX12_bitfields(c.Struct):
  SIZE = 4
  WAVES: Annotated[Annotated[int, ctypes.c_uint32], 0, 12, 0]
  WAVESIZE: Annotated[Annotated[int, ctypes.c_uint32], 1, 18, 4]
@c.record
class union_SQ_BUF_RSRC_WORD0(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_SQ_BUF_RSRC_WORD0_bitfields, 0]
  bits: Annotated[union_SQ_BUF_RSRC_WORD0_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_SQ_BUF_RSRC_WORD0_bitfields(c.Struct):
  SIZE = 4
  BASE_ADDRESS: Annotated[Annotated[int, ctypes.c_uint32], 0, 32, 0]
@c.record
class union_SQ_BUF_RSRC_WORD1(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_SQ_BUF_RSRC_WORD1_bitfields, 0]
  bits: Annotated[union_SQ_BUF_RSRC_WORD1_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_SQ_BUF_RSRC_WORD1_bitfields(c.Struct):
  SIZE = 4
  BASE_ADDRESS_HI: Annotated[Annotated[int, ctypes.c_uint32], 0, 16, 0]
  STRIDE: Annotated[Annotated[int, ctypes.c_uint32], 2, 14, 0]
  CACHE_SWIZZLE: Annotated[Annotated[int, ctypes.c_uint32], 3, 1, 6]
  SWIZZLE_ENABLE: Annotated[Annotated[int, ctypes.c_uint32], 3, 1, 7]
@c.record
class union_SQ_BUF_RSRC_WORD1_GFX11(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_SQ_BUF_RSRC_WORD1_GFX11_bitfields, 0]
  bits: Annotated[union_SQ_BUF_RSRC_WORD1_GFX11_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_SQ_BUF_RSRC_WORD1_GFX11_bitfields(c.Struct):
  SIZE = 4
  BASE_ADDRESS_HI: Annotated[Annotated[int, ctypes.c_uint32], 0, 16, 0]
  STRIDE: Annotated[Annotated[int, ctypes.c_uint32], 2, 14, 0]
  SWIZZLE_ENABLE: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 6]
@c.record
class union_SQ_BUF_RSRC_WORD2(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_SQ_BUF_RSRC_WORD2_bitfields, 0]
  bits: Annotated[union_SQ_BUF_RSRC_WORD2_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_SQ_BUF_RSRC_WORD2_bitfields(c.Struct):
  SIZE = 4
  NUM_RECORDS: Annotated[Annotated[int, ctypes.c_uint32], 0, 32, 0]
@c.record
class union_SQ_BUF_RSRC_WORD3(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_SQ_BUF_RSRC_WORD3_bitfields, 0]
  bits: Annotated[union_SQ_BUF_RSRC_WORD3_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_SQ_BUF_RSRC_WORD3_bitfields(c.Struct):
  SIZE = 4
  DST_SEL_X: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 0]
  DST_SEL_Y: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 3]
  DST_SEL_Z: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 6]
  DST_SEL_W: Annotated[Annotated[int, ctypes.c_uint32], 1, 3, 1]
  NUM_FORMAT: Annotated[Annotated[int, ctypes.c_uint32], 1, 3, 4]
  DATA_FORMAT: Annotated[Annotated[int, ctypes.c_uint32], 1, 4, 7]
  ELEMENT_SIZE: Annotated[Annotated[int, ctypes.c_uint32], 2, 2, 3]
  INDEX_STRIDE: Annotated[Annotated[int, ctypes.c_uint32], 2, 2, 5]
  ADD_TID_ENABLE: Annotated[Annotated[int, ctypes.c_uint32], 2, 1, 7]
  ATC__CI__VI: Annotated[Annotated[int, ctypes.c_uint32], 3, 1, 0]
  HASH_ENABLE: Annotated[Annotated[int, ctypes.c_uint32], 3, 1, 1]
  HEAP: Annotated[Annotated[int, ctypes.c_uint32], 3, 1, 2]
  MTYPE__CI__VI: Annotated[Annotated[int, ctypes.c_uint32], 3, 3, 3]
  TYPE: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 6]
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX10(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_SQ_BUF_RSRC_WORD3_GFX10_bitfields, 0]
  bits: Annotated[union_SQ_BUF_RSRC_WORD3_GFX10_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX10_bitfields(c.Struct):
  SIZE = 4
  DST_SEL_X: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 0]
  DST_SEL_Y: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 3]
  DST_SEL_Z: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 6]
  DST_SEL_W: Annotated[Annotated[int, ctypes.c_uint32], 1, 3, 1]
  FORMAT: Annotated[Annotated[int, ctypes.c_uint32], 1, 7, 4]
  RESERVED1: Annotated[Annotated[int, ctypes.c_uint32], 2, 2, 3]
  INDEX_STRIDE: Annotated[Annotated[int, ctypes.c_uint32], 2, 2, 5]
  ADD_TID_ENABLE: Annotated[Annotated[int, ctypes.c_uint32], 2, 1, 7]
  RESOURCE_LEVEL: Annotated[Annotated[int, ctypes.c_uint32], 3, 1, 0]
  RESERVED2: Annotated[Annotated[int, ctypes.c_uint32], 3, 3, 1]
  OOB_SELECT: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 4]
  TYPE: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 6]
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX11(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_SQ_BUF_RSRC_WORD3_GFX11_bitfields, 0]
  bits: Annotated[union_SQ_BUF_RSRC_WORD3_GFX11_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX11_bitfields(c.Struct):
  SIZE = 4
  DST_SEL_X: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 0]
  DST_SEL_Y: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 3]
  DST_SEL_Z: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 6]
  DST_SEL_W: Annotated[Annotated[int, ctypes.c_uint32], 1, 3, 1]
  FORMAT: Annotated[Annotated[int, ctypes.c_uint32], 1, 6, 4]
  RESERVED1: Annotated[Annotated[int, ctypes.c_uint32], 2, 3, 2]
  INDEX_STRIDE: Annotated[Annotated[int, ctypes.c_uint32], 2, 2, 5]
  ADD_TID_ENABLE: Annotated[Annotated[int, ctypes.c_uint32], 2, 1, 7]
  RESERVED2: Annotated[Annotated[int, ctypes.c_uint32], 3, 4, 0]
  OOB_SELECT: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 4]
  TYPE: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 6]
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX12(c.Struct):
  SIZE = 4
  bitfields: Annotated[union_SQ_BUF_RSRC_WORD3_GFX12_bitfields, 0]
  bits: Annotated[union_SQ_BUF_RSRC_WORD3_GFX12_bitfields, 0]
  u32All: Annotated[Annotated[int, ctypes.c_uint32], 0]
  i32All: Annotated[Annotated[int, ctypes.c_int32], 0]
  f32All: Annotated[Annotated[float, ctypes.c_float], 0]
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX12_bitfields(c.Struct):
  SIZE = 4
  DST_SEL_X: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 0]
  DST_SEL_Y: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 3]
  DST_SEL_Z: Annotated[Annotated[int, ctypes.c_uint32], 0, 3, 6]
  DST_SEL_W: Annotated[Annotated[int, ctypes.c_uint32], 1, 3, 1]
  FORMAT: Annotated[Annotated[int, ctypes.c_uint32], 1, 6, 4]
  RESERVED1: Annotated[Annotated[int, ctypes.c_uint32], 2, 3, 2]
  INDEX_STRIDE: Annotated[Annotated[int, ctypes.c_uint32], 2, 2, 5]
  ADD_TID_ENABLE: Annotated[Annotated[int, ctypes.c_uint32], 2, 1, 7]
  WRITE_COMPRESS_ENABLE: Annotated[Annotated[int, ctypes.c_uint32], 3, 1, 0]
  COMPRESSION_EN: Annotated[Annotated[int, ctypes.c_uint32], 3, 1, 1]
  COMPRESSION_ACCESS_MODE: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 2]
  OOB_SELECT: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 4]
  TYPE: Annotated[Annotated[int, ctypes.c_uint32], 3, 2, 6]
hsa_status_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_STATUS_SUCCESS = hsa_status_t.define('HSA_STATUS_SUCCESS', 0) # type: ignore
HSA_STATUS_INFO_BREAK = hsa_status_t.define('HSA_STATUS_INFO_BREAK', 1) # type: ignore
HSA_STATUS_ERROR = hsa_status_t.define('HSA_STATUS_ERROR', 4096) # type: ignore
HSA_STATUS_ERROR_INVALID_ARGUMENT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_ARGUMENT', 4097) # type: ignore
HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_QUEUE_CREATION', 4098) # type: ignore
HSA_STATUS_ERROR_INVALID_ALLOCATION = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_ALLOCATION', 4099) # type: ignore
HSA_STATUS_ERROR_INVALID_AGENT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_AGENT', 4100) # type: ignore
HSA_STATUS_ERROR_INVALID_REGION = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_REGION', 4101) # type: ignore
HSA_STATUS_ERROR_INVALID_SIGNAL = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_SIGNAL', 4102) # type: ignore
HSA_STATUS_ERROR_INVALID_QUEUE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_QUEUE', 4103) # type: ignore
HSA_STATUS_ERROR_OUT_OF_RESOURCES = hsa_status_t.define('HSA_STATUS_ERROR_OUT_OF_RESOURCES', 4104) # type: ignore
HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_PACKET_FORMAT', 4105) # type: ignore
HSA_STATUS_ERROR_RESOURCE_FREE = hsa_status_t.define('HSA_STATUS_ERROR_RESOURCE_FREE', 4106) # type: ignore
HSA_STATUS_ERROR_NOT_INITIALIZED = hsa_status_t.define('HSA_STATUS_ERROR_NOT_INITIALIZED', 4107) # type: ignore
HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = hsa_status_t.define('HSA_STATUS_ERROR_REFCOUNT_OVERFLOW', 4108) # type: ignore
HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = hsa_status_t.define('HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS', 4109) # type: ignore
HSA_STATUS_ERROR_INVALID_INDEX = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_INDEX', 4110) # type: ignore
HSA_STATUS_ERROR_INVALID_ISA = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_ISA', 4111) # type: ignore
HSA_STATUS_ERROR_INVALID_ISA_NAME = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_ISA_NAME', 4119) # type: ignore
HSA_STATUS_ERROR_INVALID_CODE_OBJECT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_CODE_OBJECT', 4112) # type: ignore
HSA_STATUS_ERROR_INVALID_EXECUTABLE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_EXECUTABLE', 4113) # type: ignore
HSA_STATUS_ERROR_FROZEN_EXECUTABLE = hsa_status_t.define('HSA_STATUS_ERROR_FROZEN_EXECUTABLE', 4114) # type: ignore
HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_SYMBOL_NAME', 4115) # type: ignore
HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = hsa_status_t.define('HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED', 4116) # type: ignore
HSA_STATUS_ERROR_VARIABLE_UNDEFINED = hsa_status_t.define('HSA_STATUS_ERROR_VARIABLE_UNDEFINED', 4117) # type: ignore
HSA_STATUS_ERROR_EXCEPTION = hsa_status_t.define('HSA_STATUS_ERROR_EXCEPTION', 4118) # type: ignore
HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_CODE_SYMBOL', 4120) # type: ignore
HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL', 4121) # type: ignore
HSA_STATUS_ERROR_INVALID_FILE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_FILE', 4128) # type: ignore
HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER', 4129) # type: ignore
HSA_STATUS_ERROR_INVALID_CACHE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_CACHE', 4130) # type: ignore
HSA_STATUS_ERROR_INVALID_WAVEFRONT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_WAVEFRONT', 4131) # type: ignore
HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP', 4132) # type: ignore
HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_RUNTIME_STATE', 4133) # type: ignore
HSA_STATUS_ERROR_FATAL = hsa_status_t.define('HSA_STATUS_ERROR_FATAL', 4134) # type: ignore

@dll.bind
def hsa_status_string(status:hsa_status_t, status_string:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> hsa_status_t: ...
@c.record
class struct_hsa_dim3_s(c.Struct):
  SIZE = 12
  x: Annotated[uint32_t, 0]
  y: Annotated[uint32_t, 4]
  z: Annotated[uint32_t, 8]
uint32_t = Annotated[int, ctypes.c_uint32]
hsa_dim3_t = struct_hsa_dim3_s
hsa_access_permission_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_ACCESS_PERMISSION_NONE = hsa_access_permission_t.define('HSA_ACCESS_PERMISSION_NONE', 0) # type: ignore
HSA_ACCESS_PERMISSION_RO = hsa_access_permission_t.define('HSA_ACCESS_PERMISSION_RO', 1) # type: ignore
HSA_ACCESS_PERMISSION_WO = hsa_access_permission_t.define('HSA_ACCESS_PERMISSION_WO', 2) # type: ignore
HSA_ACCESS_PERMISSION_RW = hsa_access_permission_t.define('HSA_ACCESS_PERMISSION_RW', 3) # type: ignore

hsa_file_t = Annotated[int, ctypes.c_int32]
@dll.bind
def hsa_init() -> hsa_status_t: ...
@dll.bind
def hsa_shut_down() -> hsa_status_t: ...
hsa_endianness_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_ENDIANNESS_LITTLE = hsa_endianness_t.define('HSA_ENDIANNESS_LITTLE', 0) # type: ignore
HSA_ENDIANNESS_BIG = hsa_endianness_t.define('HSA_ENDIANNESS_BIG', 1) # type: ignore

hsa_machine_model_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_MACHINE_MODEL_SMALL = hsa_machine_model_t.define('HSA_MACHINE_MODEL_SMALL', 0) # type: ignore
HSA_MACHINE_MODEL_LARGE = hsa_machine_model_t.define('HSA_MACHINE_MODEL_LARGE', 1) # type: ignore

hsa_profile_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_PROFILE_BASE = hsa_profile_t.define('HSA_PROFILE_BASE', 0) # type: ignore
HSA_PROFILE_FULL = hsa_profile_t.define('HSA_PROFILE_FULL', 1) # type: ignore

hsa_system_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_SYSTEM_INFO_VERSION_MAJOR = hsa_system_info_t.define('HSA_SYSTEM_INFO_VERSION_MAJOR', 0) # type: ignore
HSA_SYSTEM_INFO_VERSION_MINOR = hsa_system_info_t.define('HSA_SYSTEM_INFO_VERSION_MINOR', 1) # type: ignore
HSA_SYSTEM_INFO_TIMESTAMP = hsa_system_info_t.define('HSA_SYSTEM_INFO_TIMESTAMP', 2) # type: ignore
HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY = hsa_system_info_t.define('HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY', 3) # type: ignore
HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT = hsa_system_info_t.define('HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT', 4) # type: ignore
HSA_SYSTEM_INFO_ENDIANNESS = hsa_system_info_t.define('HSA_SYSTEM_INFO_ENDIANNESS', 5) # type: ignore
HSA_SYSTEM_INFO_MACHINE_MODEL = hsa_system_info_t.define('HSA_SYSTEM_INFO_MACHINE_MODEL', 6) # type: ignore
HSA_SYSTEM_INFO_EXTENSIONS = hsa_system_info_t.define('HSA_SYSTEM_INFO_EXTENSIONS', 7) # type: ignore
HSA_AMD_SYSTEM_INFO_BUILD_VERSION = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_BUILD_VERSION', 512) # type: ignore
HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED', 513) # type: ignore
HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT', 514) # type: ignore
HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED', 515) # type: ignore
HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED', 516) # type: ignore
HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED', 517) # type: ignore
HSA_AMD_SYSTEM_INFO_XNACK_ENABLED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_XNACK_ENABLED', 518) # type: ignore
HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR', 519) # type: ignore
HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR', 520) # type: ignore

@dll.bind
def hsa_system_get_info(attribute:hsa_system_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
hsa_extension_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXTENSION_FINALIZER = hsa_extension_t.define('HSA_EXTENSION_FINALIZER', 0) # type: ignore
HSA_EXTENSION_IMAGES = hsa_extension_t.define('HSA_EXTENSION_IMAGES', 1) # type: ignore
HSA_EXTENSION_PERFORMANCE_COUNTERS = hsa_extension_t.define('HSA_EXTENSION_PERFORMANCE_COUNTERS', 2) # type: ignore
HSA_EXTENSION_PROFILING_EVENTS = hsa_extension_t.define('HSA_EXTENSION_PROFILING_EVENTS', 3) # type: ignore
HSA_EXTENSION_STD_LAST = hsa_extension_t.define('HSA_EXTENSION_STD_LAST', 3) # type: ignore
HSA_AMD_FIRST_EXTENSION = hsa_extension_t.define('HSA_AMD_FIRST_EXTENSION', 512) # type: ignore
HSA_EXTENSION_AMD_PROFILER = hsa_extension_t.define('HSA_EXTENSION_AMD_PROFILER', 512) # type: ignore
HSA_EXTENSION_AMD_LOADER = hsa_extension_t.define('HSA_EXTENSION_AMD_LOADER', 513) # type: ignore
HSA_EXTENSION_AMD_AQLPROFILE = hsa_extension_t.define('HSA_EXTENSION_AMD_AQLPROFILE', 514) # type: ignore
HSA_EXTENSION_AMD_PC_SAMPLING = hsa_extension_t.define('HSA_EXTENSION_AMD_PC_SAMPLING', 515) # type: ignore
HSA_AMD_LAST_EXTENSION = hsa_extension_t.define('HSA_AMD_LAST_EXTENSION', 515) # type: ignore

uint16_t = Annotated[int, ctypes.c_uint16]
@dll.bind
def hsa_extension_get_name(extension:uint16_t, name:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> hsa_status_t: ...
@dll.bind
def hsa_system_extension_supported(extension:uint16_t, version_major:uint16_t, version_minor:uint16_t, result:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> hsa_status_t: ...
@dll.bind
def hsa_system_major_extension_supported(extension:uint16_t, version_major:uint16_t, version_minor:c.POINTER[uint16_t], result:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> hsa_status_t: ...
@dll.bind
def hsa_system_get_extension_table(extension:uint16_t, version_major:uint16_t, version_minor:uint16_t, table:c.POINTER[None]) -> hsa_status_t: ...
size_t = Annotated[int, ctypes.c_uint64]
@dll.bind
def hsa_system_get_major_extension_table(extension:uint16_t, version_major:uint16_t, table_length:size_t, table:c.POINTER[None]) -> hsa_status_t: ...
@c.record
class struct_hsa_agent_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
uint64_t = Annotated[int, ctypes.c_uint64]
hsa_agent_t = struct_hsa_agent_s
hsa_agent_feature_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AGENT_FEATURE_KERNEL_DISPATCH = hsa_agent_feature_t.define('HSA_AGENT_FEATURE_KERNEL_DISPATCH', 1) # type: ignore
HSA_AGENT_FEATURE_AGENT_DISPATCH = hsa_agent_feature_t.define('HSA_AGENT_FEATURE_AGENT_DISPATCH', 2) # type: ignore

hsa_device_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_DEVICE_TYPE_CPU = hsa_device_type_t.define('HSA_DEVICE_TYPE_CPU', 0) # type: ignore
HSA_DEVICE_TYPE_GPU = hsa_device_type_t.define('HSA_DEVICE_TYPE_GPU', 1) # type: ignore
HSA_DEVICE_TYPE_DSP = hsa_device_type_t.define('HSA_DEVICE_TYPE_DSP', 2) # type: ignore
HSA_DEVICE_TYPE_AIE = hsa_device_type_t.define('HSA_DEVICE_TYPE_AIE', 3) # type: ignore

hsa_default_float_rounding_mode_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT = hsa_default_float_rounding_mode_t.define('HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT', 0) # type: ignore
HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO = hsa_default_float_rounding_mode_t.define('HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO', 1) # type: ignore
HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR = hsa_default_float_rounding_mode_t.define('HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR', 2) # type: ignore

hsa_agent_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AGENT_INFO_NAME = hsa_agent_info_t.define('HSA_AGENT_INFO_NAME', 0) # type: ignore
HSA_AGENT_INFO_VENDOR_NAME = hsa_agent_info_t.define('HSA_AGENT_INFO_VENDOR_NAME', 1) # type: ignore
HSA_AGENT_INFO_FEATURE = hsa_agent_info_t.define('HSA_AGENT_INFO_FEATURE', 2) # type: ignore
HSA_AGENT_INFO_MACHINE_MODEL = hsa_agent_info_t.define('HSA_AGENT_INFO_MACHINE_MODEL', 3) # type: ignore
HSA_AGENT_INFO_PROFILE = hsa_agent_info_t.define('HSA_AGENT_INFO_PROFILE', 4) # type: ignore
HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = hsa_agent_info_t.define('HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE', 5) # type: ignore
HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = hsa_agent_info_t.define('HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES', 23) # type: ignore
HSA_AGENT_INFO_FAST_F16_OPERATION = hsa_agent_info_t.define('HSA_AGENT_INFO_FAST_F16_OPERATION', 24) # type: ignore
HSA_AGENT_INFO_WAVEFRONT_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_WAVEFRONT_SIZE', 6) # type: ignore
HSA_AGENT_INFO_WORKGROUP_MAX_DIM = hsa_agent_info_t.define('HSA_AGENT_INFO_WORKGROUP_MAX_DIM', 7) # type: ignore
HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_WORKGROUP_MAX_SIZE', 8) # type: ignore
HSA_AGENT_INFO_GRID_MAX_DIM = hsa_agent_info_t.define('HSA_AGENT_INFO_GRID_MAX_DIM', 9) # type: ignore
HSA_AGENT_INFO_GRID_MAX_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_GRID_MAX_SIZE', 10) # type: ignore
HSA_AGENT_INFO_FBARRIER_MAX_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_FBARRIER_MAX_SIZE', 11) # type: ignore
HSA_AGENT_INFO_QUEUES_MAX = hsa_agent_info_t.define('HSA_AGENT_INFO_QUEUES_MAX', 12) # type: ignore
HSA_AGENT_INFO_QUEUE_MIN_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_QUEUE_MIN_SIZE', 13) # type: ignore
HSA_AGENT_INFO_QUEUE_MAX_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_QUEUE_MAX_SIZE', 14) # type: ignore
HSA_AGENT_INFO_QUEUE_TYPE = hsa_agent_info_t.define('HSA_AGENT_INFO_QUEUE_TYPE', 15) # type: ignore
HSA_AGENT_INFO_NODE = hsa_agent_info_t.define('HSA_AGENT_INFO_NODE', 16) # type: ignore
HSA_AGENT_INFO_DEVICE = hsa_agent_info_t.define('HSA_AGENT_INFO_DEVICE', 17) # type: ignore
HSA_AGENT_INFO_CACHE_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_CACHE_SIZE', 18) # type: ignore
HSA_AGENT_INFO_ISA = hsa_agent_info_t.define('HSA_AGENT_INFO_ISA', 19) # type: ignore
HSA_AGENT_INFO_EXTENSIONS = hsa_agent_info_t.define('HSA_AGENT_INFO_EXTENSIONS', 20) # type: ignore
HSA_AGENT_INFO_VERSION_MAJOR = hsa_agent_info_t.define('HSA_AGENT_INFO_VERSION_MAJOR', 21) # type: ignore
HSA_AGENT_INFO_VERSION_MINOR = hsa_agent_info_t.define('HSA_AGENT_INFO_VERSION_MINOR', 22) # type: ignore
HSA_AGENT_INFO_LAST = hsa_agent_info_t.define('HSA_AGENT_INFO_LAST', 2147483647) # type: ignore

@dll.bind
def hsa_agent_get_info(agent:hsa_agent_t, attribute:hsa_agent_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_iterate_agents(callback:c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
hsa_exception_policy_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXCEPTION_POLICY_BREAK = hsa_exception_policy_t.define('HSA_EXCEPTION_POLICY_BREAK', 1) # type: ignore
HSA_EXCEPTION_POLICY_DETECT = hsa_exception_policy_t.define('HSA_EXCEPTION_POLICY_DETECT', 2) # type: ignore

@dll.bind
def hsa_agent_get_exception_policies(agent:hsa_agent_t, profile:hsa_profile_t, mask:c.POINTER[uint16_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_cache_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_cache_t = struct_hsa_cache_s
hsa_cache_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_CACHE_INFO_NAME_LENGTH = hsa_cache_info_t.define('HSA_CACHE_INFO_NAME_LENGTH', 0) # type: ignore
HSA_CACHE_INFO_NAME = hsa_cache_info_t.define('HSA_CACHE_INFO_NAME', 1) # type: ignore
HSA_CACHE_INFO_LEVEL = hsa_cache_info_t.define('HSA_CACHE_INFO_LEVEL', 2) # type: ignore
HSA_CACHE_INFO_SIZE = hsa_cache_info_t.define('HSA_CACHE_INFO_SIZE', 3) # type: ignore

@dll.bind
def hsa_cache_get_info(cache:hsa_cache_t, attribute:hsa_cache_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_agent_iterate_caches(agent:hsa_agent_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_cache_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_agent_extension_supported(extension:uint16_t, agent:hsa_agent_t, version_major:uint16_t, version_minor:uint16_t, result:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> hsa_status_t: ...
@dll.bind
def hsa_agent_major_extension_supported(extension:uint16_t, agent:hsa_agent_t, version_major:uint16_t, version_minor:c.POINTER[uint16_t], result:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> hsa_status_t: ...
@c.record
class struct_hsa_signal_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_signal_t = struct_hsa_signal_s
hsa_signal_value_t = Annotated[int, ctypes.c_int64]
@dll.bind
def hsa_signal_create(initial_value:hsa_signal_value_t, num_consumers:uint32_t, consumers:c.POINTER[hsa_agent_t], signal:c.POINTER[hsa_signal_t]) -> hsa_status_t: ...
@dll.bind
def hsa_signal_destroy(signal:hsa_signal_t) -> hsa_status_t: ...
@dll.bind
def hsa_signal_load_scacquire(signal:hsa_signal_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_load_relaxed(signal:hsa_signal_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_load_acquire(signal:hsa_signal_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_store_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_store_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_store_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_silent_store_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_silent_store_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_exchange_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_exchange_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_exchange_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_exchange_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_exchange_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_exchange_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_exchange_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_cas_scacq_screl(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_cas_acq_rel(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_cas_scacquire(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_cas_acquire(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_cas_relaxed(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_cas_screlease(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_cas_release(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_add_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_add_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_add_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_add_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_add_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_add_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_add_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_subtract_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_subtract_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_subtract_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_subtract_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_subtract_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_subtract_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_subtract_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_and_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_and_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_and_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_and_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_and_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_and_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_and_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_or_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_or_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_or_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_or_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_or_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_or_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_or_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_xor_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_xor_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_xor_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_xor_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_xor_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_xor_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind
def hsa_signal_xor_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
hsa_signal_condition_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_SIGNAL_CONDITION_EQ = hsa_signal_condition_t.define('HSA_SIGNAL_CONDITION_EQ', 0) # type: ignore
HSA_SIGNAL_CONDITION_NE = hsa_signal_condition_t.define('HSA_SIGNAL_CONDITION_NE', 1) # type: ignore
HSA_SIGNAL_CONDITION_LT = hsa_signal_condition_t.define('HSA_SIGNAL_CONDITION_LT', 2) # type: ignore
HSA_SIGNAL_CONDITION_GTE = hsa_signal_condition_t.define('HSA_SIGNAL_CONDITION_GTE', 3) # type: ignore

hsa_wait_state_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_WAIT_STATE_BLOCKED = hsa_wait_state_t.define('HSA_WAIT_STATE_BLOCKED', 0) # type: ignore
HSA_WAIT_STATE_ACTIVE = hsa_wait_state_t.define('HSA_WAIT_STATE_ACTIVE', 1) # type: ignore

@dll.bind
def hsa_signal_wait_scacquire(signal:hsa_signal_t, condition:hsa_signal_condition_t, compare_value:hsa_signal_value_t, timeout_hint:uint64_t, wait_state_hint:hsa_wait_state_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_wait_relaxed(signal:hsa_signal_t, condition:hsa_signal_condition_t, compare_value:hsa_signal_value_t, timeout_hint:uint64_t, wait_state_hint:hsa_wait_state_t) -> hsa_signal_value_t: ...
@dll.bind
def hsa_signal_wait_acquire(signal:hsa_signal_t, condition:hsa_signal_condition_t, compare_value:hsa_signal_value_t, timeout_hint:uint64_t, wait_state_hint:hsa_wait_state_t) -> hsa_signal_value_t: ...
@c.record
class struct_hsa_signal_group_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_signal_group_t = struct_hsa_signal_group_s
@dll.bind
def hsa_signal_group_create(num_signals:uint32_t, signals:c.POINTER[hsa_signal_t], num_consumers:uint32_t, consumers:c.POINTER[hsa_agent_t], signal_group:c.POINTER[hsa_signal_group_t]) -> hsa_status_t: ...
@dll.bind
def hsa_signal_group_destroy(signal_group:hsa_signal_group_t) -> hsa_status_t: ...
@dll.bind
def hsa_signal_group_wait_any_scacquire(signal_group:hsa_signal_group_t, conditions:c.POINTER[hsa_signal_condition_t], compare_values:c.POINTER[hsa_signal_value_t], wait_state_hint:hsa_wait_state_t, signal:c.POINTER[hsa_signal_t], value:c.POINTER[hsa_signal_value_t]) -> hsa_status_t: ...
@dll.bind
def hsa_signal_group_wait_any_relaxed(signal_group:hsa_signal_group_t, conditions:c.POINTER[hsa_signal_condition_t], compare_values:c.POINTER[hsa_signal_value_t], wait_state_hint:hsa_wait_state_t, signal:c.POINTER[hsa_signal_t], value:c.POINTER[hsa_signal_value_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_region_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_region_t = struct_hsa_region_s
hsa_queue_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_QUEUE_TYPE_MULTI = hsa_queue_type_t.define('HSA_QUEUE_TYPE_MULTI', 0) # type: ignore
HSA_QUEUE_TYPE_SINGLE = hsa_queue_type_t.define('HSA_QUEUE_TYPE_SINGLE', 1) # type: ignore
HSA_QUEUE_TYPE_COOPERATIVE = hsa_queue_type_t.define('HSA_QUEUE_TYPE_COOPERATIVE', 2) # type: ignore

hsa_queue_type32_t = Annotated[int, ctypes.c_uint32]
hsa_queue_feature_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_QUEUE_FEATURE_KERNEL_DISPATCH = hsa_queue_feature_t.define('HSA_QUEUE_FEATURE_KERNEL_DISPATCH', 1) # type: ignore
HSA_QUEUE_FEATURE_AGENT_DISPATCH = hsa_queue_feature_t.define('HSA_QUEUE_FEATURE_AGENT_DISPATCH', 2) # type: ignore

@c.record
class struct_hsa_queue_s(c.Struct):
  SIZE = 40
  type: Annotated[hsa_queue_type32_t, 0]
  features: Annotated[uint32_t, 4]
  base_address: Annotated[c.POINTER[None], 8]
  doorbell_signal: Annotated[hsa_signal_t, 16]
  size: Annotated[uint32_t, 24]
  reserved1: Annotated[uint32_t, 28]
  id: Annotated[uint64_t, 32]
hsa_queue_t = struct_hsa_queue_s
@dll.bind
def hsa_queue_create(agent:hsa_agent_t, size:uint32_t, type:hsa_queue_type32_t, callback:c.CFUNCTYPE(None, hsa_status_t, c.POINTER[hsa_queue_t], c.POINTER[None]), data:c.POINTER[None], private_segment_size:uint32_t, group_segment_size:uint32_t, queue:c.POINTER[c.POINTER[hsa_queue_t]]) -> hsa_status_t: ...
@dll.bind
def hsa_soft_queue_create(region:hsa_region_t, size:uint32_t, type:hsa_queue_type32_t, features:uint32_t, doorbell_signal:hsa_signal_t, queue:c.POINTER[c.POINTER[hsa_queue_t]]) -> hsa_status_t: ...
@dll.bind
def hsa_queue_destroy(queue:c.POINTER[hsa_queue_t]) -> hsa_status_t: ...
@dll.bind
def hsa_queue_inactivate(queue:c.POINTER[hsa_queue_t]) -> hsa_status_t: ...
@dll.bind
def hsa_queue_load_read_index_acquire(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind
def hsa_queue_load_read_index_scacquire(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind
def hsa_queue_load_read_index_relaxed(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind
def hsa_queue_load_write_index_acquire(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind
def hsa_queue_load_write_index_scacquire(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind
def hsa_queue_load_write_index_relaxed(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind
def hsa_queue_store_write_index_relaxed(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind
def hsa_queue_store_write_index_release(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind
def hsa_queue_store_write_index_screlease(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind
def hsa_queue_cas_write_index_acq_rel(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_cas_write_index_scacq_screl(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_cas_write_index_acquire(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_cas_write_index_scacquire(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_cas_write_index_relaxed(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_cas_write_index_release(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_cas_write_index_screlease(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_add_write_index_acq_rel(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_add_write_index_scacq_screl(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_add_write_index_acquire(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_add_write_index_scacquire(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_add_write_index_relaxed(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_add_write_index_release(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_add_write_index_screlease(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind
def hsa_queue_store_read_index_relaxed(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind
def hsa_queue_store_read_index_release(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind
def hsa_queue_store_read_index_screlease(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
hsa_packet_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_PACKET_TYPE_VENDOR_SPECIFIC = hsa_packet_type_t.define('HSA_PACKET_TYPE_VENDOR_SPECIFIC', 0) # type: ignore
HSA_PACKET_TYPE_INVALID = hsa_packet_type_t.define('HSA_PACKET_TYPE_INVALID', 1) # type: ignore
HSA_PACKET_TYPE_KERNEL_DISPATCH = hsa_packet_type_t.define('HSA_PACKET_TYPE_KERNEL_DISPATCH', 2) # type: ignore
HSA_PACKET_TYPE_BARRIER_AND = hsa_packet_type_t.define('HSA_PACKET_TYPE_BARRIER_AND', 3) # type: ignore
HSA_PACKET_TYPE_AGENT_DISPATCH = hsa_packet_type_t.define('HSA_PACKET_TYPE_AGENT_DISPATCH', 4) # type: ignore
HSA_PACKET_TYPE_BARRIER_OR = hsa_packet_type_t.define('HSA_PACKET_TYPE_BARRIER_OR', 5) # type: ignore

hsa_fence_scope_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_FENCE_SCOPE_NONE = hsa_fence_scope_t.define('HSA_FENCE_SCOPE_NONE', 0) # type: ignore
HSA_FENCE_SCOPE_AGENT = hsa_fence_scope_t.define('HSA_FENCE_SCOPE_AGENT', 1) # type: ignore
HSA_FENCE_SCOPE_SYSTEM = hsa_fence_scope_t.define('HSA_FENCE_SCOPE_SYSTEM', 2) # type: ignore

hsa_packet_header_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_PACKET_HEADER_TYPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_TYPE', 0) # type: ignore
HSA_PACKET_HEADER_BARRIER = hsa_packet_header_t.define('HSA_PACKET_HEADER_BARRIER', 8) # type: ignore
HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE', 9) # type: ignore
HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE', 9) # type: ignore
HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE', 11) # type: ignore
HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE', 11) # type: ignore

hsa_packet_header_width_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_PACKET_HEADER_WIDTH_TYPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_TYPE', 8) # type: ignore
HSA_PACKET_HEADER_WIDTH_BARRIER = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_BARRIER', 1) # type: ignore
HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE', 2) # type: ignore
HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE', 2) # type: ignore
HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE', 2) # type: ignore
HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE', 2) # type: ignore

hsa_kernel_dispatch_packet_setup_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = hsa_kernel_dispatch_packet_setup_t.define('HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS', 0) # type: ignore

hsa_kernel_dispatch_packet_setup_width_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS = hsa_kernel_dispatch_packet_setup_width_t.define('HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS', 2) # type: ignore

@c.record
class struct_hsa_kernel_dispatch_packet_s(c.Struct):
  SIZE = 64
  header: Annotated[uint16_t, 0]
  setup: Annotated[uint16_t, 2]
  full_header: Annotated[uint32_t, 0]
  workgroup_size_x: Annotated[uint16_t, 4]
  workgroup_size_y: Annotated[uint16_t, 6]
  workgroup_size_z: Annotated[uint16_t, 8]
  reserved0: Annotated[uint16_t, 10]
  grid_size_x: Annotated[uint32_t, 12]
  grid_size_y: Annotated[uint32_t, 16]
  grid_size_z: Annotated[uint32_t, 20]
  private_segment_size: Annotated[uint32_t, 24]
  group_segment_size: Annotated[uint32_t, 28]
  kernel_object: Annotated[uint64_t, 32]
  kernarg_address: Annotated[c.POINTER[None], 40]
  reserved2: Annotated[uint64_t, 48]
  completion_signal: Annotated[hsa_signal_t, 56]
hsa_kernel_dispatch_packet_t = struct_hsa_kernel_dispatch_packet_s
@c.record
class struct_hsa_agent_dispatch_packet_s(c.Struct):
  SIZE = 64
  header: Annotated[uint16_t, 0]
  type: Annotated[uint16_t, 2]
  reserved0: Annotated[uint32_t, 4]
  return_address: Annotated[c.POINTER[None], 8]
  arg: Annotated[c.Array[uint64_t, Literal[4]], 16]
  reserved2: Annotated[uint64_t, 48]
  completion_signal: Annotated[hsa_signal_t, 56]
hsa_agent_dispatch_packet_t = struct_hsa_agent_dispatch_packet_s
@c.record
class struct_hsa_barrier_and_packet_s(c.Struct):
  SIZE = 64
  header: Annotated[uint16_t, 0]
  reserved0: Annotated[uint16_t, 2]
  reserved1: Annotated[uint32_t, 4]
  dep_signal: Annotated[c.Array[hsa_signal_t, Literal[5]], 8]
  reserved2: Annotated[uint64_t, 48]
  completion_signal: Annotated[hsa_signal_t, 56]
hsa_barrier_and_packet_t = struct_hsa_barrier_and_packet_s
@c.record
class struct_hsa_barrier_or_packet_s(c.Struct):
  SIZE = 64
  header: Annotated[uint16_t, 0]
  reserved0: Annotated[uint16_t, 2]
  reserved1: Annotated[uint32_t, 4]
  dep_signal: Annotated[c.Array[hsa_signal_t, Literal[5]], 8]
  reserved2: Annotated[uint64_t, 48]
  completion_signal: Annotated[hsa_signal_t, 56]
hsa_barrier_or_packet_t = struct_hsa_barrier_or_packet_s
hsa_region_segment_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_REGION_SEGMENT_GLOBAL = hsa_region_segment_t.define('HSA_REGION_SEGMENT_GLOBAL', 0) # type: ignore
HSA_REGION_SEGMENT_READONLY = hsa_region_segment_t.define('HSA_REGION_SEGMENT_READONLY', 1) # type: ignore
HSA_REGION_SEGMENT_PRIVATE = hsa_region_segment_t.define('HSA_REGION_SEGMENT_PRIVATE', 2) # type: ignore
HSA_REGION_SEGMENT_GROUP = hsa_region_segment_t.define('HSA_REGION_SEGMENT_GROUP', 3) # type: ignore
HSA_REGION_SEGMENT_KERNARG = hsa_region_segment_t.define('HSA_REGION_SEGMENT_KERNARG', 4) # type: ignore

hsa_region_global_flag_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_REGION_GLOBAL_FLAG_KERNARG = hsa_region_global_flag_t.define('HSA_REGION_GLOBAL_FLAG_KERNARG', 1) # type: ignore
HSA_REGION_GLOBAL_FLAG_FINE_GRAINED = hsa_region_global_flag_t.define('HSA_REGION_GLOBAL_FLAG_FINE_GRAINED', 2) # type: ignore
HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED = hsa_region_global_flag_t.define('HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED', 4) # type: ignore
HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = hsa_region_global_flag_t.define('HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED', 8) # type: ignore

hsa_region_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_REGION_INFO_SEGMENT = hsa_region_info_t.define('HSA_REGION_INFO_SEGMENT', 0) # type: ignore
HSA_REGION_INFO_GLOBAL_FLAGS = hsa_region_info_t.define('HSA_REGION_INFO_GLOBAL_FLAGS', 1) # type: ignore
HSA_REGION_INFO_SIZE = hsa_region_info_t.define('HSA_REGION_INFO_SIZE', 2) # type: ignore
HSA_REGION_INFO_ALLOC_MAX_SIZE = hsa_region_info_t.define('HSA_REGION_INFO_ALLOC_MAX_SIZE', 4) # type: ignore
HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE = hsa_region_info_t.define('HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE', 8) # type: ignore
HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED = hsa_region_info_t.define('HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED', 5) # type: ignore
HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE = hsa_region_info_t.define('HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE', 6) # type: ignore
HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT = hsa_region_info_t.define('HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT', 7) # type: ignore

@dll.bind
def hsa_region_get_info(region:hsa_region_t, attribute:hsa_region_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_agent_iterate_regions(agent:hsa_agent_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_region_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_memory_allocate(region:hsa_region_t, size:size_t, ptr:c.POINTER[c.POINTER[None]]) -> hsa_status_t: ...
@dll.bind
def hsa_memory_free(ptr:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_memory_copy(dst:c.POINTER[None], src:c.POINTER[None], size:size_t) -> hsa_status_t: ...
@dll.bind
def hsa_memory_assign_agent(ptr:c.POINTER[None], agent:hsa_agent_t, access:hsa_access_permission_t) -> hsa_status_t: ...
@dll.bind
def hsa_memory_register(ptr:c.POINTER[None], size:size_t) -> hsa_status_t: ...
@dll.bind
def hsa_memory_deregister(ptr:c.POINTER[None], size:size_t) -> hsa_status_t: ...
@c.record
class struct_hsa_isa_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_isa_t = struct_hsa_isa_s
@dll.bind
def hsa_isa_from_name(name:c.POINTER[Annotated[bytes, ctypes.c_char]], isa:c.POINTER[hsa_isa_t]) -> hsa_status_t: ...
@dll.bind
def hsa_agent_iterate_isas(agent:hsa_agent_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_isa_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
hsa_isa_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_ISA_INFO_NAME_LENGTH = hsa_isa_info_t.define('HSA_ISA_INFO_NAME_LENGTH', 0) # type: ignore
HSA_ISA_INFO_NAME = hsa_isa_info_t.define('HSA_ISA_INFO_NAME', 1) # type: ignore
HSA_ISA_INFO_CALL_CONVENTION_COUNT = hsa_isa_info_t.define('HSA_ISA_INFO_CALL_CONVENTION_COUNT', 2) # type: ignore
HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE = hsa_isa_info_t.define('HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE', 3) # type: ignore
HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT = hsa_isa_info_t.define('HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT', 4) # type: ignore
HSA_ISA_INFO_MACHINE_MODELS = hsa_isa_info_t.define('HSA_ISA_INFO_MACHINE_MODELS', 5) # type: ignore
HSA_ISA_INFO_PROFILES = hsa_isa_info_t.define('HSA_ISA_INFO_PROFILES', 6) # type: ignore
HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES = hsa_isa_info_t.define('HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES', 7) # type: ignore
HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = hsa_isa_info_t.define('HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES', 8) # type: ignore
HSA_ISA_INFO_FAST_F16_OPERATION = hsa_isa_info_t.define('HSA_ISA_INFO_FAST_F16_OPERATION', 9) # type: ignore
HSA_ISA_INFO_WORKGROUP_MAX_DIM = hsa_isa_info_t.define('HSA_ISA_INFO_WORKGROUP_MAX_DIM', 12) # type: ignore
HSA_ISA_INFO_WORKGROUP_MAX_SIZE = hsa_isa_info_t.define('HSA_ISA_INFO_WORKGROUP_MAX_SIZE', 13) # type: ignore
HSA_ISA_INFO_GRID_MAX_DIM = hsa_isa_info_t.define('HSA_ISA_INFO_GRID_MAX_DIM', 14) # type: ignore
HSA_ISA_INFO_GRID_MAX_SIZE = hsa_isa_info_t.define('HSA_ISA_INFO_GRID_MAX_SIZE', 16) # type: ignore
HSA_ISA_INFO_FBARRIER_MAX_SIZE = hsa_isa_info_t.define('HSA_ISA_INFO_FBARRIER_MAX_SIZE', 17) # type: ignore

@dll.bind
def hsa_isa_get_info(isa:hsa_isa_t, attribute:hsa_isa_info_t, index:uint32_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_isa_get_info_alt(isa:hsa_isa_t, attribute:hsa_isa_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_isa_get_exception_policies(isa:hsa_isa_t, profile:hsa_profile_t, mask:c.POINTER[uint16_t]) -> hsa_status_t: ...
hsa_fp_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_FP_TYPE_16 = hsa_fp_type_t.define('HSA_FP_TYPE_16', 1) # type: ignore
HSA_FP_TYPE_32 = hsa_fp_type_t.define('HSA_FP_TYPE_32', 2) # type: ignore
HSA_FP_TYPE_64 = hsa_fp_type_t.define('HSA_FP_TYPE_64', 4) # type: ignore

hsa_flush_mode_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_FLUSH_MODE_FTZ = hsa_flush_mode_t.define('HSA_FLUSH_MODE_FTZ', 1) # type: ignore
HSA_FLUSH_MODE_NON_FTZ = hsa_flush_mode_t.define('HSA_FLUSH_MODE_NON_FTZ', 2) # type: ignore

hsa_round_method_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_ROUND_METHOD_SINGLE = hsa_round_method_t.define('HSA_ROUND_METHOD_SINGLE', 1) # type: ignore
HSA_ROUND_METHOD_DOUBLE = hsa_round_method_t.define('HSA_ROUND_METHOD_DOUBLE', 2) # type: ignore

@dll.bind
def hsa_isa_get_round_method(isa:hsa_isa_t, fp_type:hsa_fp_type_t, flush_mode:hsa_flush_mode_t, round_method:c.POINTER[hsa_round_method_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_wavefront_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_wavefront_t = struct_hsa_wavefront_s
hsa_wavefront_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_WAVEFRONT_INFO_SIZE = hsa_wavefront_info_t.define('HSA_WAVEFRONT_INFO_SIZE', 0) # type: ignore

@dll.bind
def hsa_wavefront_get_info(wavefront:hsa_wavefront_t, attribute:hsa_wavefront_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_isa_iterate_wavefronts(isa:hsa_isa_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_wavefront_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_isa_compatible(code_object_isa:hsa_isa_t, agent_isa:hsa_isa_t, result:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> hsa_status_t: ...
@c.record
class struct_hsa_code_object_reader_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_code_object_reader_t = struct_hsa_code_object_reader_s
@dll.bind
def hsa_code_object_reader_create_from_file(file:hsa_file_t, code_object_reader:c.POINTER[hsa_code_object_reader_t]) -> hsa_status_t: ...
@dll.bind
def hsa_code_object_reader_create_from_memory(code_object:c.POINTER[None], size:size_t, code_object_reader:c.POINTER[hsa_code_object_reader_t]) -> hsa_status_t: ...
@dll.bind
def hsa_code_object_reader_destroy(code_object_reader:hsa_code_object_reader_t) -> hsa_status_t: ...
@c.record
class struct_hsa_executable_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_executable_t = struct_hsa_executable_s
hsa_executable_state_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXECUTABLE_STATE_UNFROZEN = hsa_executable_state_t.define('HSA_EXECUTABLE_STATE_UNFROZEN', 0) # type: ignore
HSA_EXECUTABLE_STATE_FROZEN = hsa_executable_state_t.define('HSA_EXECUTABLE_STATE_FROZEN', 1) # type: ignore

@dll.bind
def hsa_executable_create(profile:hsa_profile_t, executable_state:hsa_executable_state_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], executable:c.POINTER[hsa_executable_t]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_create_alt(profile:hsa_profile_t, default_float_rounding_mode:hsa_default_float_rounding_mode_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], executable:c.POINTER[hsa_executable_t]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_destroy(executable:hsa_executable_t) -> hsa_status_t: ...
@c.record
class struct_hsa_loaded_code_object_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_loaded_code_object_t = struct_hsa_loaded_code_object_s
@dll.bind
def hsa_executable_load_program_code_object(executable:hsa_executable_t, code_object_reader:hsa_code_object_reader_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], loaded_code_object:c.POINTER[hsa_loaded_code_object_t]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_load_agent_code_object(executable:hsa_executable_t, agent:hsa_agent_t, code_object_reader:hsa_code_object_reader_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], loaded_code_object:c.POINTER[hsa_loaded_code_object_t]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_freeze(executable:hsa_executable_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hsa_status_t: ...
hsa_executable_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXECUTABLE_INFO_PROFILE = hsa_executable_info_t.define('HSA_EXECUTABLE_INFO_PROFILE', 1) # type: ignore
HSA_EXECUTABLE_INFO_STATE = hsa_executable_info_t.define('HSA_EXECUTABLE_INFO_STATE', 2) # type: ignore
HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE = hsa_executable_info_t.define('HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE', 3) # type: ignore

@dll.bind
def hsa_executable_get_info(executable:hsa_executable_t, attribute:hsa_executable_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_global_variable_define(executable:hsa_executable_t, variable_name:c.POINTER[Annotated[bytes, ctypes.c_char]], address:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_agent_global_variable_define(executable:hsa_executable_t, agent:hsa_agent_t, variable_name:c.POINTER[Annotated[bytes, ctypes.c_char]], address:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_readonly_variable_define(executable:hsa_executable_t, agent:hsa_agent_t, variable_name:c.POINTER[Annotated[bytes, ctypes.c_char]], address:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_validate(executable:hsa_executable_t, result:c.POINTER[uint32_t]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_validate_alt(executable:hsa_executable_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], result:c.POINTER[uint32_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_executable_symbol_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_executable_symbol_t = struct_hsa_executable_symbol_s
int32_t = Annotated[int, ctypes.c_int32]
@dll.bind
def hsa_executable_get_symbol(executable:hsa_executable_t, module_name:c.POINTER[Annotated[bytes, ctypes.c_char]], symbol_name:c.POINTER[Annotated[bytes, ctypes.c_char]], agent:hsa_agent_t, call_convention:int32_t, symbol:c.POINTER[hsa_executable_symbol_t]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_get_symbol_by_name(executable:hsa_executable_t, symbol_name:c.POINTER[Annotated[bytes, ctypes.c_char]], agent:c.POINTER[hsa_agent_t], symbol:c.POINTER[hsa_executable_symbol_t]) -> hsa_status_t: ...
hsa_symbol_kind_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_SYMBOL_KIND_VARIABLE = hsa_symbol_kind_t.define('HSA_SYMBOL_KIND_VARIABLE', 0) # type: ignore
HSA_SYMBOL_KIND_KERNEL = hsa_symbol_kind_t.define('HSA_SYMBOL_KIND_KERNEL', 1) # type: ignore
HSA_SYMBOL_KIND_INDIRECT_FUNCTION = hsa_symbol_kind_t.define('HSA_SYMBOL_KIND_INDIRECT_FUNCTION', 2) # type: ignore

hsa_symbol_linkage_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_SYMBOL_LINKAGE_MODULE = hsa_symbol_linkage_t.define('HSA_SYMBOL_LINKAGE_MODULE', 0) # type: ignore
HSA_SYMBOL_LINKAGE_PROGRAM = hsa_symbol_linkage_t.define('HSA_SYMBOL_LINKAGE_PROGRAM', 1) # type: ignore

hsa_variable_allocation_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_VARIABLE_ALLOCATION_AGENT = hsa_variable_allocation_t.define('HSA_VARIABLE_ALLOCATION_AGENT', 0) # type: ignore
HSA_VARIABLE_ALLOCATION_PROGRAM = hsa_variable_allocation_t.define('HSA_VARIABLE_ALLOCATION_PROGRAM', 1) # type: ignore

hsa_variable_segment_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_VARIABLE_SEGMENT_GLOBAL = hsa_variable_segment_t.define('HSA_VARIABLE_SEGMENT_GLOBAL', 0) # type: ignore
HSA_VARIABLE_SEGMENT_READONLY = hsa_variable_segment_t.define('HSA_VARIABLE_SEGMENT_READONLY', 1) # type: ignore

hsa_executable_symbol_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXECUTABLE_SYMBOL_INFO_TYPE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_TYPE', 0) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH', 1) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_NAME = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_NAME', 2) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH', 3) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME', 4) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_AGENT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_AGENT', 20) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS', 21) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE', 5) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION', 17) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION', 6) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT', 7) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT', 8) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE', 9) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST', 10) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT', 22) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE', 11) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT', 12) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE', 13) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE', 14) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK', 15) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION', 18) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT', 23) # type: ignore
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION', 16) # type: ignore

@dll.bind
def hsa_executable_symbol_get_info(executable_symbol:hsa_executable_symbol_t, attribute:hsa_executable_symbol_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_iterate_symbols(executable:hsa_executable_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_executable_t, hsa_executable_symbol_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_iterate_agent_symbols(executable:hsa_executable_t, agent:hsa_agent_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_iterate_program_symbols(executable:hsa_executable_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_executable_t, hsa_executable_symbol_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
@c.record
class struct_hsa_code_object_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_code_object_t = struct_hsa_code_object_s
@c.record
class struct_hsa_callback_data_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_callback_data_t = struct_hsa_callback_data_s
@dll.bind
def hsa_code_object_serialize(code_object:hsa_code_object_t, alloc_callback:c.CFUNCTYPE(hsa_status_t, size_t, hsa_callback_data_t, c.POINTER[c.POINTER[None]]), callback_data:hsa_callback_data_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], serialized_code_object:c.POINTER[c.POINTER[None]], serialized_code_object_size:c.POINTER[size_t]) -> hsa_status_t: ...
@dll.bind
def hsa_code_object_deserialize(serialized_code_object:c.POINTER[None], serialized_code_object_size:size_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], code_object:c.POINTER[hsa_code_object_t]) -> hsa_status_t: ...
@dll.bind
def hsa_code_object_destroy(code_object:hsa_code_object_t) -> hsa_status_t: ...
hsa_code_object_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_CODE_OBJECT_TYPE_PROGRAM = hsa_code_object_type_t.define('HSA_CODE_OBJECT_TYPE_PROGRAM', 0) # type: ignore

hsa_code_object_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_CODE_OBJECT_INFO_VERSION = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_VERSION', 0) # type: ignore
HSA_CODE_OBJECT_INFO_TYPE = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_TYPE', 1) # type: ignore
HSA_CODE_OBJECT_INFO_ISA = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_ISA', 2) # type: ignore
HSA_CODE_OBJECT_INFO_MACHINE_MODEL = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_MACHINE_MODEL', 3) # type: ignore
HSA_CODE_OBJECT_INFO_PROFILE = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_PROFILE', 4) # type: ignore
HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE', 5) # type: ignore

@dll.bind
def hsa_code_object_get_info(code_object:hsa_code_object_t, attribute:hsa_code_object_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_executable_load_code_object(executable:hsa_executable_t, agent:hsa_agent_t, code_object:hsa_code_object_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> hsa_status_t: ...
@c.record
class struct_hsa_code_symbol_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_code_symbol_t = struct_hsa_code_symbol_s
@dll.bind
def hsa_code_object_get_symbol(code_object:hsa_code_object_t, symbol_name:c.POINTER[Annotated[bytes, ctypes.c_char]], symbol:c.POINTER[hsa_code_symbol_t]) -> hsa_status_t: ...
@dll.bind
def hsa_code_object_get_symbol_from_name(code_object:hsa_code_object_t, module_name:c.POINTER[Annotated[bytes, ctypes.c_char]], symbol_name:c.POINTER[Annotated[bytes, ctypes.c_char]], symbol:c.POINTER[hsa_code_symbol_t]) -> hsa_status_t: ...
hsa_code_symbol_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_CODE_SYMBOL_INFO_TYPE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_TYPE', 0) # type: ignore
HSA_CODE_SYMBOL_INFO_NAME_LENGTH = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_NAME_LENGTH', 1) # type: ignore
HSA_CODE_SYMBOL_INFO_NAME = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_NAME', 2) # type: ignore
HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH', 3) # type: ignore
HSA_CODE_SYMBOL_INFO_MODULE_NAME = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_MODULE_NAME', 4) # type: ignore
HSA_CODE_SYMBOL_INFO_LINKAGE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_LINKAGE', 5) # type: ignore
HSA_CODE_SYMBOL_INFO_IS_DEFINITION = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_IS_DEFINITION', 17) # type: ignore
HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION', 6) # type: ignore
HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT', 7) # type: ignore
HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT', 8) # type: ignore
HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE', 9) # type: ignore
HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST', 10) # type: ignore
HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE', 11) # type: ignore
HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT', 12) # type: ignore
HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE', 13) # type: ignore
HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE', 14) # type: ignore
HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK', 15) # type: ignore
HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION', 18) # type: ignore
HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION', 16) # type: ignore
HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE', 19) # type: ignore

@dll.bind
def hsa_code_symbol_get_info(code_symbol:hsa_code_symbol_t, attribute:hsa_code_symbol_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_code_object_iterate_symbols(code_object:hsa_code_object_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_code_object_t, hsa_code_symbol_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
hsa_signal_condition32_t = Annotated[int, ctypes.c_uint32]
hsa_amd_packet_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_PACKET_TYPE_BARRIER_VALUE = hsa_amd_packet_type_t.define('HSA_AMD_PACKET_TYPE_BARRIER_VALUE', 2) # type: ignore
HSA_AMD_PACKET_TYPE_AIE_ERT = hsa_amd_packet_type_t.define('HSA_AMD_PACKET_TYPE_AIE_ERT', 3) # type: ignore

hsa_amd_packet_type8_t = Annotated[int, ctypes.c_ubyte]
@c.record
class struct_hsa_amd_packet_header_s(c.Struct):
  SIZE = 4
  header: Annotated[uint16_t, 0]
  AmdFormat: Annotated[hsa_amd_packet_type8_t, 2]
  reserved: Annotated[uint8_t, 3]
uint8_t = Annotated[int, ctypes.c_ubyte]
hsa_amd_vendor_packet_header_t = struct_hsa_amd_packet_header_s
@c.record
class struct_hsa_amd_barrier_value_packet_s(c.Struct):
  SIZE = 64
  header: Annotated[hsa_amd_vendor_packet_header_t, 0]
  reserved0: Annotated[uint32_t, 4]
  signal: Annotated[hsa_signal_t, 8]
  value: Annotated[hsa_signal_value_t, 16]
  mask: Annotated[hsa_signal_value_t, 24]
  cond: Annotated[hsa_signal_condition32_t, 32]
  reserved1: Annotated[uint32_t, 36]
  reserved2: Annotated[uint64_t, 40]
  reserved3: Annotated[uint64_t, 48]
  completion_signal: Annotated[hsa_signal_t, 56]
hsa_amd_barrier_value_packet_t = struct_hsa_amd_barrier_value_packet_s
hsa_amd_aie_ert_state = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_AIE_ERT_STATE_NEW = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_NEW', 1) # type: ignore
HSA_AMD_AIE_ERT_STATE_QUEUED = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_QUEUED', 2) # type: ignore
HSA_AMD_AIE_ERT_STATE_RUNNING = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_RUNNING', 3) # type: ignore
HSA_AMD_AIE_ERT_STATE_COMPLETED = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_COMPLETED', 4) # type: ignore
HSA_AMD_AIE_ERT_STATE_ERROR = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_ERROR', 5) # type: ignore
HSA_AMD_AIE_ERT_STATE_ABORT = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_ABORT', 6) # type: ignore
HSA_AMD_AIE_ERT_STATE_SUBMITTED = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_SUBMITTED', 7) # type: ignore
HSA_AMD_AIE_ERT_STATE_TIMEOUT = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_TIMEOUT', 8) # type: ignore
HSA_AMD_AIE_ERT_STATE_NORESPONSE = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_NORESPONSE', 9) # type: ignore
HSA_AMD_AIE_ERT_STATE_SKERROR = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_SKERROR', 10) # type: ignore
HSA_AMD_AIE_ERT_STATE_SKCRASHED = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_SKCRASHED', 11) # type: ignore
HSA_AMD_AIE_ERT_STATE_MAX = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_MAX', 12) # type: ignore

hsa_amd_aie_ert_cmd_opcode_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_AIE_ERT_START_CU = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_CU', 0) # type: ignore
HSA_AMD_AIE_ERT_START_KERNEL = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_KERNEL', 0) # type: ignore
HSA_AMD_AIE_ERT_CONFIGURE = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_CONFIGURE', 2) # type: ignore
HSA_AMD_AIE_ERT_EXIT = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_EXIT', 3) # type: ignore
HSA_AMD_AIE_ERT_ABORT = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_ABORT', 4) # type: ignore
HSA_AMD_AIE_ERT_EXEC_WRITE = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_EXEC_WRITE', 5) # type: ignore
HSA_AMD_AIE_ERT_CU_STAT = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_CU_STAT', 6) # type: ignore
HSA_AMD_AIE_ERT_START_COPYBO = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_COPYBO', 7) # type: ignore
HSA_AMD_AIE_ERT_SK_CONFIG = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_SK_CONFIG', 8) # type: ignore
HSA_AMD_AIE_ERT_SK_START = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_SK_START', 9) # type: ignore
HSA_AMD_AIE_ERT_SK_UNCONFIG = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_SK_UNCONFIG', 10) # type: ignore
HSA_AMD_AIE_ERT_INIT_CU = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_INIT_CU', 11) # type: ignore
HSA_AMD_AIE_ERT_START_FA = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_FA', 12) # type: ignore
HSA_AMD_AIE_ERT_CLK_CALIB = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_CLK_CALIB', 13) # type: ignore
HSA_AMD_AIE_ERT_MB_VALIDATE = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_MB_VALIDATE', 14) # type: ignore
HSA_AMD_AIE_ERT_START_KEY_VAL = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_KEY_VAL', 15) # type: ignore
HSA_AMD_AIE_ERT_ACCESS_TEST_C = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_ACCESS_TEST_C', 16) # type: ignore
HSA_AMD_AIE_ERT_ACCESS_TEST = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_ACCESS_TEST', 17) # type: ignore
HSA_AMD_AIE_ERT_START_DPU = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_DPU', 18) # type: ignore
HSA_AMD_AIE_ERT_CMD_CHAIN = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_CMD_CHAIN', 19) # type: ignore
HSA_AMD_AIE_ERT_START_NPU = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_NPU', 20) # type: ignore
HSA_AMD_AIE_ERT_START_NPU_PREEMPT = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_NPU_PREEMPT', 21) # type: ignore

@c.record
class struct_hsa_amd_aie_ert_start_kernel_data_s(c.Struct):
  SIZE = 8
  pdi_addr: Annotated[c.POINTER[None], 0]
  data: Annotated[c.Array[uint32_t, Literal[0]], 8]
hsa_amd_aie_ert_start_kernel_data_t = struct_hsa_amd_aie_ert_start_kernel_data_s
@c.record
class struct_hsa_amd_aie_ert_packet_s(c.Struct):
  SIZE = 64
  header: Annotated[hsa_amd_vendor_packet_header_t, 0]
  state: Annotated[uint32_t, 4, 4, 0]
  custom: Annotated[uint32_t, 4, 8, 4]
  count: Annotated[uint32_t, 5, 11, 4]
  opcode: Annotated[uint32_t, 6, 5, 7]
  type: Annotated[uint32_t, 7, 4, 4]
  reserved0: Annotated[uint64_t, 8]
  reserved1: Annotated[uint64_t, 16]
  reserved2: Annotated[uint64_t, 24]
  reserved3: Annotated[uint64_t, 32]
  reserved4: Annotated[uint64_t, 40]
  reserved5: Annotated[uint64_t, 48]
  payload_data: Annotated[uint64_t, 56]
hsa_amd_aie_ert_packet_t = struct_hsa_amd_aie_ert_packet_s
_anonenum0 = CEnum(Annotated[int, ctypes.c_uint32])
HSA_STATUS_ERROR_INVALID_MEMORY_POOL = _anonenum0.define('HSA_STATUS_ERROR_INVALID_MEMORY_POOL', 40) # type: ignore
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION = _anonenum0.define('HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION', 41) # type: ignore
HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION = _anonenum0.define('HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION', 42) # type: ignore
HSA_STATUS_ERROR_MEMORY_FAULT = _anonenum0.define('HSA_STATUS_ERROR_MEMORY_FAULT', 43) # type: ignore
HSA_STATUS_CU_MASK_REDUCED = _anonenum0.define('HSA_STATUS_CU_MASK_REDUCED', 44) # type: ignore
HSA_STATUS_ERROR_OUT_OF_REGISTERS = _anonenum0.define('HSA_STATUS_ERROR_OUT_OF_REGISTERS', 45) # type: ignore
HSA_STATUS_ERROR_RESOURCE_BUSY = _anonenum0.define('HSA_STATUS_ERROR_RESOURCE_BUSY', 46) # type: ignore
HSA_STATUS_ERROR_NOT_SUPPORTED = _anonenum0.define('HSA_STATUS_ERROR_NOT_SUPPORTED', 47) # type: ignore

hsa_amd_iommu_version_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_IOMMU_SUPPORT_NONE = hsa_amd_iommu_version_t.define('HSA_IOMMU_SUPPORT_NONE', 0) # type: ignore
HSA_IOMMU_SUPPORT_V2 = hsa_amd_iommu_version_t.define('HSA_IOMMU_SUPPORT_V2', 1) # type: ignore

@c.record
class struct_hsa_amd_clock_counters_s(c.Struct):
  SIZE = 32
  gpu_clock_counter: Annotated[uint64_t, 0]
  cpu_clock_counter: Annotated[uint64_t, 8]
  system_clock_counter: Annotated[uint64_t, 16]
  system_clock_frequency: Annotated[uint64_t, 24]
hsa_amd_clock_counters_t = struct_hsa_amd_clock_counters_s
enum_hsa_amd_agent_info_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_AGENT_INFO_CHIP_ID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_CHIP_ID', 40960) # type: ignore
HSA_AMD_AGENT_INFO_CACHELINE_SIZE = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_CACHELINE_SIZE', 40961) # type: ignore
HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT', 40962) # type: ignore
HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY', 40963) # type: ignore
HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_DRIVER_NODE_ID', 40964) # type: ignore
HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS', 40965) # type: ignore
HSA_AMD_AGENT_INFO_BDFID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_BDFID', 40966) # type: ignore
HSA_AMD_AGENT_INFO_MEMORY_WIDTH = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MEMORY_WIDTH', 40967) # type: ignore
HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY', 40968) # type: ignore
HSA_AMD_AGENT_INFO_PRODUCT_NAME = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_PRODUCT_NAME', 40969) # type: ignore
HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU', 40970) # type: ignore
HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU', 40971) # type: ignore
HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES', 40972) # type: ignore
HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE', 40973) # type: ignore
HSA_AMD_AGENT_INFO_HDP_FLUSH = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_HDP_FLUSH', 40974) # type: ignore
HSA_AMD_AGENT_INFO_DOMAIN = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_DOMAIN', 40975) # type: ignore
HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES', 40976) # type: ignore
HSA_AMD_AGENT_INFO_UUID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_UUID', 40977) # type: ignore
HSA_AMD_AGENT_INFO_ASIC_REVISION = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_ASIC_REVISION', 40978) # type: ignore
HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS', 40979) # type: ignore
HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT', 40980) # type: ignore
HSA_AMD_AGENT_INFO_MEMORY_AVAIL = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MEMORY_AVAIL', 40981) # type: ignore
HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY', 40982) # type: ignore
HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID', 41223) # type: ignore
HSA_AMD_AGENT_INFO_UCODE_VERSION = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_UCODE_VERSION', 41224) # type: ignore
HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION', 41225) # type: ignore
HSA_AMD_AGENT_INFO_NUM_SDMA_ENG = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SDMA_ENG', 41226) # type: ignore
HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG', 41227) # type: ignore
HSA_AMD_AGENT_INFO_IOMMU_SUPPORT = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_IOMMU_SUPPORT', 41232) # type: ignore
HSA_AMD_AGENT_INFO_NUM_XCC = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_XCC', 41233) # type: ignore
HSA_AMD_AGENT_INFO_DRIVER_UID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_DRIVER_UID', 41234) # type: ignore
HSA_AMD_AGENT_INFO_NEAREST_CPU = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NEAREST_CPU', 41235) # type: ignore
HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES', 41236) # type: ignore
HSA_AMD_AGENT_INFO_AQL_EXTENSIONS = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_AQL_EXTENSIONS', 41237) # type: ignore
HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX', 41238) # type: ignore
HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT', 41239) # type: ignore
HSA_AMD_AGENT_INFO_CLOCK_COUNTERS = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_CLOCK_COUNTERS', 41240) # type: ignore

hsa_amd_agent_info_t: TypeAlias = enum_hsa_amd_agent_info_s
enum_hsa_amd_agent_memory_properties_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU = enum_hsa_amd_agent_memory_properties_s.define('HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU', 1) # type: ignore

hsa_amd_agent_memory_properties_t: TypeAlias = enum_hsa_amd_agent_memory_properties_s
enum_hsa_amd_sdma_engine_id = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_SDMA_ENGINE_0 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_0', 1) # type: ignore
HSA_AMD_SDMA_ENGINE_1 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_1', 2) # type: ignore
HSA_AMD_SDMA_ENGINE_2 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_2', 4) # type: ignore
HSA_AMD_SDMA_ENGINE_3 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_3', 8) # type: ignore
HSA_AMD_SDMA_ENGINE_4 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_4', 16) # type: ignore
HSA_AMD_SDMA_ENGINE_5 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_5', 32) # type: ignore
HSA_AMD_SDMA_ENGINE_6 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_6', 64) # type: ignore
HSA_AMD_SDMA_ENGINE_7 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_7', 128) # type: ignore
HSA_AMD_SDMA_ENGINE_8 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_8', 256) # type: ignore
HSA_AMD_SDMA_ENGINE_9 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_9', 512) # type: ignore
HSA_AMD_SDMA_ENGINE_10 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_10', 1024) # type: ignore
HSA_AMD_SDMA_ENGINE_11 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_11', 2048) # type: ignore
HSA_AMD_SDMA_ENGINE_12 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_12', 4096) # type: ignore
HSA_AMD_SDMA_ENGINE_13 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_13', 8192) # type: ignore
HSA_AMD_SDMA_ENGINE_14 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_14', 16384) # type: ignore
HSA_AMD_SDMA_ENGINE_15 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_15', 32768) # type: ignore

hsa_amd_sdma_engine_id_t: TypeAlias = enum_hsa_amd_sdma_engine_id
@c.record
class struct_hsa_amd_hdp_flush_s(c.Struct):
  SIZE = 16
  HDP_MEM_FLUSH_CNTL: Annotated[c.POINTER[uint32_t], 0]
  HDP_REG_FLUSH_CNTL: Annotated[c.POINTER[uint32_t], 8]
hsa_amd_hdp_flush_t = struct_hsa_amd_hdp_flush_s
enum_hsa_amd_region_info_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_REGION_INFO_HOST_ACCESSIBLE = enum_hsa_amd_region_info_s.define('HSA_AMD_REGION_INFO_HOST_ACCESSIBLE', 40960) # type: ignore
HSA_AMD_REGION_INFO_BASE = enum_hsa_amd_region_info_s.define('HSA_AMD_REGION_INFO_BASE', 40961) # type: ignore
HSA_AMD_REGION_INFO_BUS_WIDTH = enum_hsa_amd_region_info_s.define('HSA_AMD_REGION_INFO_BUS_WIDTH', 40962) # type: ignore
HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY = enum_hsa_amd_region_info_s.define('HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY', 40963) # type: ignore

hsa_amd_region_info_t: TypeAlias = enum_hsa_amd_region_info_s
enum_hsa_amd_coherency_type_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_COHERENCY_TYPE_COHERENT = enum_hsa_amd_coherency_type_s.define('HSA_AMD_COHERENCY_TYPE_COHERENT', 0) # type: ignore
HSA_AMD_COHERENCY_TYPE_NONCOHERENT = enum_hsa_amd_coherency_type_s.define('HSA_AMD_COHERENCY_TYPE_NONCOHERENT', 1) # type: ignore

hsa_amd_coherency_type_t: TypeAlias = enum_hsa_amd_coherency_type_s
enum_hsa_amd_dma_buf_mapping_type_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_DMABUF_MAPPING_TYPE_NONE = enum_hsa_amd_dma_buf_mapping_type_s.define('HSA_AMD_DMABUF_MAPPING_TYPE_NONE', 0) # type: ignore
HSA_AMD_DMABUF_MAPPING_TYPE_PCIE = enum_hsa_amd_dma_buf_mapping_type_s.define('HSA_AMD_DMABUF_MAPPING_TYPE_PCIE', 1) # type: ignore

hsa_amd_dma_buf_mapping_type_t: TypeAlias = enum_hsa_amd_dma_buf_mapping_type_s
@dll.bind
def hsa_amd_coherency_get_type(agent:hsa_agent_t, type:c.POINTER[hsa_amd_coherency_type_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_coherency_set_type(agent:hsa_agent_t, type:hsa_amd_coherency_type_t) -> hsa_status_t: ...
@c.record
class struct_hsa_amd_profiling_dispatch_time_s(c.Struct):
  SIZE = 16
  start: Annotated[uint64_t, 0]
  end: Annotated[uint64_t, 8]
hsa_amd_profiling_dispatch_time_t = struct_hsa_amd_profiling_dispatch_time_s
@c.record
class struct_hsa_amd_profiling_async_copy_time_s(c.Struct):
  SIZE = 16
  start: Annotated[uint64_t, 0]
  end: Annotated[uint64_t, 8]
hsa_amd_profiling_async_copy_time_t = struct_hsa_amd_profiling_async_copy_time_s
@dll.bind
def hsa_amd_profiling_set_profiler_enabled(queue:c.POINTER[hsa_queue_t], enable:Annotated[int, ctypes.c_int32]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_profiling_async_copy_enable(enable:Annotated[bool, ctypes.c_bool]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_profiling_get_dispatch_time(agent:hsa_agent_t, signal:hsa_signal_t, time:c.POINTER[hsa_amd_profiling_dispatch_time_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_profiling_get_async_copy_time(signal:hsa_signal_t, time:c.POINTER[hsa_amd_profiling_async_copy_time_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_profiling_convert_tick_to_system_domain(agent:hsa_agent_t, agent_tick:uint64_t, system_tick:c.POINTER[uint64_t]) -> hsa_status_t: ...
hsa_amd_signal_attribute_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_SIGNAL_AMD_GPU_ONLY = hsa_amd_signal_attribute_t.define('HSA_AMD_SIGNAL_AMD_GPU_ONLY', 1) # type: ignore
HSA_AMD_SIGNAL_IPC = hsa_amd_signal_attribute_t.define('HSA_AMD_SIGNAL_IPC', 2) # type: ignore

@dll.bind
def hsa_amd_signal_create(initial_value:hsa_signal_value_t, num_consumers:uint32_t, consumers:c.POINTER[hsa_agent_t], attributes:uint64_t, signal:c.POINTER[hsa_signal_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_signal_value_pointer(signal:hsa_signal_t, value_ptr:c.POINTER[c.POINTER[hsa_signal_value_t]]) -> hsa_status_t: ...
hsa_amd_signal_handler = c.CFUNCTYPE(Annotated[bool, ctypes.c_bool], Annotated[int, ctypes.c_int64], c.POINTER[None])
@dll.bind
def hsa_amd_signal_async_handler(signal:hsa_signal_t, cond:hsa_signal_condition_t, value:hsa_signal_value_t, handler:hsa_amd_signal_handler, arg:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_signal_wait_all(signal_count:uint32_t, signals:c.POINTER[hsa_signal_t], conds:c.POINTER[hsa_signal_condition_t], values:c.POINTER[hsa_signal_value_t], timeout_hint:uint64_t, wait_hint:hsa_wait_state_t, satisfying_values:c.POINTER[hsa_signal_value_t]) -> uint32_t: ...
@dll.bind
def hsa_amd_signal_wait_any(signal_count:uint32_t, signals:c.POINTER[hsa_signal_t], conds:c.POINTER[hsa_signal_condition_t], values:c.POINTER[hsa_signal_value_t], timeout_hint:uint64_t, wait_hint:hsa_wait_state_t, satisfying_value:c.POINTER[hsa_signal_value_t]) -> uint32_t: ...
@dll.bind
def hsa_amd_async_function(callback:c.CFUNCTYPE(None, c.POINTER[None]), arg:c.POINTER[None]) -> hsa_status_t: ...
@c.record
class struct_hsa_amd_image_descriptor_s(c.Struct):
  SIZE = 12
  version: Annotated[uint32_t, 0]
  deviceID: Annotated[uint32_t, 4]
  data: Annotated[c.Array[uint32_t, Literal[1]], 8]
hsa_amd_image_descriptor_t = struct_hsa_amd_image_descriptor_s
@c.record
class struct_hsa_ext_image_descriptor_s(c.Struct):
  SIZE = 48
  geometry: Annotated[hsa_ext_image_geometry_t, 0]
  width: Annotated[size_t, 8]
  height: Annotated[size_t, 16]
  depth: Annotated[size_t, 24]
  array_size: Annotated[size_t, 32]
  format: Annotated[hsa_ext_image_format_t, 40]
hsa_ext_image_descriptor_t: TypeAlias = struct_hsa_ext_image_descriptor_s
hsa_ext_image_geometry_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_IMAGE_GEOMETRY_1D = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_1D', 0) # type: ignore
HSA_EXT_IMAGE_GEOMETRY_2D = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_2D', 1) # type: ignore
HSA_EXT_IMAGE_GEOMETRY_3D = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_3D', 2) # type: ignore
HSA_EXT_IMAGE_GEOMETRY_1DA = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_1DA', 3) # type: ignore
HSA_EXT_IMAGE_GEOMETRY_2DA = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_2DA', 4) # type: ignore
HSA_EXT_IMAGE_GEOMETRY_1DB = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_1DB', 5) # type: ignore
HSA_EXT_IMAGE_GEOMETRY_2DDEPTH = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_2DDEPTH', 6) # type: ignore
HSA_EXT_IMAGE_GEOMETRY_2DADEPTH = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_2DADEPTH', 7) # type: ignore

@c.record
class struct_hsa_ext_image_format_s(c.Struct):
  SIZE = 8
  channel_type: Annotated[hsa_ext_image_channel_type32_t, 0]
  channel_order: Annotated[hsa_ext_image_channel_order32_t, 4]
hsa_ext_image_format_t: TypeAlias = struct_hsa_ext_image_format_s
hsa_ext_image_channel_type32_t = Annotated[int, ctypes.c_uint32]
hsa_ext_image_channel_order32_t = Annotated[int, ctypes.c_uint32]
@c.record
class struct_hsa_ext_image_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_ext_image_t: TypeAlias = struct_hsa_ext_image_s
@dll.bind
def hsa_amd_image_create(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], image_layout:c.POINTER[hsa_amd_image_descriptor_t], image_data:c.POINTER[None], access_permission:hsa_access_permission_t, image:c.POINTER[hsa_ext_image_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_image_get_info_max_dim(agent:hsa_agent_t, attribute:hsa_agent_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_queue_cu_set_mask(queue:c.POINTER[hsa_queue_t], num_cu_mask_count:uint32_t, cu_mask:c.POINTER[uint32_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_queue_cu_get_mask(queue:c.POINTER[hsa_queue_t], num_cu_mask_count:uint32_t, cu_mask:c.POINTER[uint32_t]) -> hsa_status_t: ...
hsa_amd_segment_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_SEGMENT_GLOBAL = hsa_amd_segment_t.define('HSA_AMD_SEGMENT_GLOBAL', 0) # type: ignore
HSA_AMD_SEGMENT_READONLY = hsa_amd_segment_t.define('HSA_AMD_SEGMENT_READONLY', 1) # type: ignore
HSA_AMD_SEGMENT_PRIVATE = hsa_amd_segment_t.define('HSA_AMD_SEGMENT_PRIVATE', 2) # type: ignore
HSA_AMD_SEGMENT_GROUP = hsa_amd_segment_t.define('HSA_AMD_SEGMENT_GROUP', 3) # type: ignore

@c.record
class struct_hsa_amd_memory_pool_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_amd_memory_pool_t = struct_hsa_amd_memory_pool_s
enum_hsa_amd_memory_pool_global_flag_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = enum_hsa_amd_memory_pool_global_flag_s.define('HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT', 1) # type: ignore
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = enum_hsa_amd_memory_pool_global_flag_s.define('HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED', 2) # type: ignore
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = enum_hsa_amd_memory_pool_global_flag_s.define('HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED', 4) # type: ignore
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = enum_hsa_amd_memory_pool_global_flag_s.define('HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED', 8) # type: ignore

hsa_amd_memory_pool_global_flag_t: TypeAlias = enum_hsa_amd_memory_pool_global_flag_s
enum_hsa_amd_memory_pool_location_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_MEMORY_POOL_LOCATION_CPU = enum_hsa_amd_memory_pool_location_s.define('HSA_AMD_MEMORY_POOL_LOCATION_CPU', 0) # type: ignore
HSA_AMD_MEMORY_POOL_LOCATION_GPU = enum_hsa_amd_memory_pool_location_s.define('HSA_AMD_MEMORY_POOL_LOCATION_GPU', 1) # type: ignore

hsa_amd_memory_pool_location_t: TypeAlias = enum_hsa_amd_memory_pool_location_s
hsa_amd_memory_pool_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_MEMORY_POOL_INFO_SEGMENT = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_SEGMENT', 0) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS', 1) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_SIZE = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_SIZE', 2) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED', 5) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE', 6) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT', 7) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL', 15) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE', 16) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_LOCATION = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_LOCATION', 17) # type: ignore
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE', 18) # type: ignore

enum_hsa_amd_memory_pool_flag_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_MEMORY_POOL_STANDARD_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_STANDARD_FLAG', 0) # type: ignore
HSA_AMD_MEMORY_POOL_PCIE_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_PCIE_FLAG', 1) # type: ignore
HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG', 2) # type: ignore
HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG', 4) # type: ignore
HSA_AMD_MEMORY_POOL_UNCACHED_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_UNCACHED_FLAG', 8) # type: ignore

hsa_amd_memory_pool_flag_t: TypeAlias = enum_hsa_amd_memory_pool_flag_s
@dll.bind
def hsa_amd_memory_pool_get_info(memory_pool:hsa_amd_memory_pool_t, attribute:hsa_amd_memory_pool_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_agent_iterate_memory_pools(agent:hsa_agent_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_amd_memory_pool_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_pool_allocate(memory_pool:hsa_amd_memory_pool_t, size:size_t, flags:uint32_t, ptr:c.POINTER[c.POINTER[None]]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_pool_free(ptr:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_async_copy(dst:c.POINTER[None], dst_agent:hsa_agent_t, src:c.POINTER[None], src_agent:hsa_agent_t, size:size_t, num_dep_signals:uint32_t, dep_signals:c.POINTER[hsa_signal_t], completion_signal:hsa_signal_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_async_copy_on_engine(dst:c.POINTER[None], dst_agent:hsa_agent_t, src:c.POINTER[None], src_agent:hsa_agent_t, size:size_t, num_dep_signals:uint32_t, dep_signals:c.POINTER[hsa_signal_t], completion_signal:hsa_signal_t, engine_id:hsa_amd_sdma_engine_id_t, force_copy_on_sdma:Annotated[bool, ctypes.c_bool]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_copy_engine_status(dst_agent:hsa_agent_t, src_agent:hsa_agent_t, engine_ids_mask:c.POINTER[uint32_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_get_preferred_copy_engine(dst_agent:hsa_agent_t, src_agent:hsa_agent_t, recommended_ids_mask:c.POINTER[uint32_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_pitched_ptr_s(c.Struct):
  SIZE = 24
  base: Annotated[c.POINTER[None], 0]
  pitch: Annotated[size_t, 8]
  slice: Annotated[size_t, 16]
hsa_pitched_ptr_t = struct_hsa_pitched_ptr_s
hsa_amd_copy_direction_t = CEnum(Annotated[int, ctypes.c_uint32])
hsaHostToHost = hsa_amd_copy_direction_t.define('hsaHostToHost', 0) # type: ignore
hsaHostToDevice = hsa_amd_copy_direction_t.define('hsaHostToDevice', 1) # type: ignore
hsaDeviceToHost = hsa_amd_copy_direction_t.define('hsaDeviceToHost', 2) # type: ignore
hsaDeviceToDevice = hsa_amd_copy_direction_t.define('hsaDeviceToDevice', 3) # type: ignore

@dll.bind
def hsa_amd_memory_async_copy_rect(dst:c.POINTER[hsa_pitched_ptr_t], dst_offset:c.POINTER[hsa_dim3_t], src:c.POINTER[hsa_pitched_ptr_t], src_offset:c.POINTER[hsa_dim3_t], range:c.POINTER[hsa_dim3_t], copy_agent:hsa_agent_t, dir:hsa_amd_copy_direction_t, num_dep_signals:uint32_t, dep_signals:c.POINTER[hsa_signal_t], completion_signal:hsa_signal_t) -> hsa_status_t: ...
hsa_amd_memory_pool_access_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = hsa_amd_memory_pool_access_t.define('HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED', 0) # type: ignore
HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT = hsa_amd_memory_pool_access_t.define('HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT', 1) # type: ignore
HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT = hsa_amd_memory_pool_access_t.define('HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT', 2) # type: ignore

hsa_amd_link_info_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT', 0) # type: ignore
HSA_AMD_LINK_INFO_TYPE_QPI = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_QPI', 1) # type: ignore
HSA_AMD_LINK_INFO_TYPE_PCIE = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_PCIE', 2) # type: ignore
HSA_AMD_LINK_INFO_TYPE_INFINBAND = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_INFINBAND', 3) # type: ignore
HSA_AMD_LINK_INFO_TYPE_XGMI = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_XGMI', 4) # type: ignore

@c.record
class struct_hsa_amd_memory_pool_link_info_s(c.Struct):
  SIZE = 28
  min_latency: Annotated[uint32_t, 0]
  max_latency: Annotated[uint32_t, 4]
  min_bandwidth: Annotated[uint32_t, 8]
  max_bandwidth: Annotated[uint32_t, 12]
  atomic_support_32bit: Annotated[Annotated[bool, ctypes.c_bool], 16]
  atomic_support_64bit: Annotated[Annotated[bool, ctypes.c_bool], 17]
  coherent_support: Annotated[Annotated[bool, ctypes.c_bool], 18]
  link_type: Annotated[hsa_amd_link_info_type_t, 20]
  numa_distance: Annotated[uint32_t, 24]
hsa_amd_memory_pool_link_info_t = struct_hsa_amd_memory_pool_link_info_s
hsa_amd_agent_memory_pool_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = hsa_amd_agent_memory_pool_info_t.define('HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS', 0) # type: ignore
HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS = hsa_amd_agent_memory_pool_info_t.define('HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS', 1) # type: ignore
HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO = hsa_amd_agent_memory_pool_info_t.define('HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO', 2) # type: ignore

@dll.bind
def hsa_amd_agent_memory_pool_get_info(agent:hsa_agent_t, memory_pool:hsa_amd_memory_pool_t, attribute:hsa_amd_agent_memory_pool_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_agents_allow_access(num_agents:uint32_t, agents:c.POINTER[hsa_agent_t], flags:c.POINTER[uint32_t], ptr:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_pool_can_migrate(src_memory_pool:hsa_amd_memory_pool_t, dst_memory_pool:hsa_amd_memory_pool_t, result:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_migrate(ptr:c.POINTER[None], memory_pool:hsa_amd_memory_pool_t, flags:uint32_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_lock(host_ptr:c.POINTER[None], size:size_t, agents:c.POINTER[hsa_agent_t], num_agent:Annotated[int, ctypes.c_int32], agent_ptr:c.POINTER[c.POINTER[None]]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_lock_to_pool(host_ptr:c.POINTER[None], size:size_t, agents:c.POINTER[hsa_agent_t], num_agent:Annotated[int, ctypes.c_int32], pool:hsa_amd_memory_pool_t, flags:uint32_t, agent_ptr:c.POINTER[c.POINTER[None]]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_unlock(host_ptr:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_memory_fill(ptr:c.POINTER[None], value:uint32_t, count:size_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_interop_map_buffer(num_agents:uint32_t, agents:c.POINTER[hsa_agent_t], interop_handle:Annotated[int, ctypes.c_int32], flags:uint32_t, size:c.POINTER[size_t], ptr:c.POINTER[c.POINTER[None]], metadata_size:c.POINTER[size_t], metadata:c.POINTER[c.POINTER[None]]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_interop_unmap_buffer(ptr:c.POINTER[None]) -> hsa_status_t: ...
hsa_amd_pointer_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_POINTER_TYPE_UNKNOWN = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_UNKNOWN', 0) # type: ignore
HSA_EXT_POINTER_TYPE_HSA = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_HSA', 1) # type: ignore
HSA_EXT_POINTER_TYPE_LOCKED = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_LOCKED', 2) # type: ignore
HSA_EXT_POINTER_TYPE_GRAPHICS = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_GRAPHICS', 3) # type: ignore
HSA_EXT_POINTER_TYPE_IPC = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_IPC', 4) # type: ignore
HSA_EXT_POINTER_TYPE_RESERVED_ADDR = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_RESERVED_ADDR', 5) # type: ignore
HSA_EXT_POINTER_TYPE_HSA_VMEM = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_HSA_VMEM', 6) # type: ignore

@c.record
class struct_hsa_amd_pointer_info_s(c.Struct):
  SIZE = 56
  size: Annotated[uint32_t, 0]
  type: Annotated[hsa_amd_pointer_type_t, 4]
  agentBaseAddress: Annotated[c.POINTER[None], 8]
  hostBaseAddress: Annotated[c.POINTER[None], 16]
  sizeInBytes: Annotated[size_t, 24]
  userData: Annotated[c.POINTER[None], 32]
  agentOwner: Annotated[hsa_agent_t, 40]
  global_flags: Annotated[uint32_t, 48]
  registered: Annotated[Annotated[bool, ctypes.c_bool], 52]
hsa_amd_pointer_info_t = struct_hsa_amd_pointer_info_s
@dll.bind
def hsa_amd_pointer_info(ptr:c.POINTER[None], info:c.POINTER[hsa_amd_pointer_info_t], alloc:c.CFUNCTYPE(c.POINTER[None], size_t), num_agents_accessible:c.POINTER[uint32_t], accessible:c.POINTER[c.POINTER[hsa_agent_t]]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_pointer_info_set_userdata(ptr:c.POINTER[None], userdata:c.POINTER[None]) -> hsa_status_t: ...
@c.record
class struct_hsa_amd_ipc_memory_s(c.Struct):
  SIZE = 32
  handle: Annotated[c.Array[uint32_t, Literal[8]], 0]
hsa_amd_ipc_memory_t = struct_hsa_amd_ipc_memory_s
@dll.bind
def hsa_amd_ipc_memory_create(ptr:c.POINTER[None], len:size_t, handle:c.POINTER[hsa_amd_ipc_memory_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_ipc_memory_attach(handle:c.POINTER[hsa_amd_ipc_memory_t], len:size_t, num_agents:uint32_t, mapping_agents:c.POINTER[hsa_agent_t], mapped_ptr:c.POINTER[c.POINTER[None]]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_ipc_memory_detach(mapped_ptr:c.POINTER[None]) -> hsa_status_t: ...
hsa_amd_ipc_signal_t = struct_hsa_amd_ipc_memory_s
@dll.bind
def hsa_amd_ipc_signal_create(signal:hsa_signal_t, handle:c.POINTER[hsa_amd_ipc_signal_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_ipc_signal_attach(handle:c.POINTER[hsa_amd_ipc_signal_t], signal:c.POINTER[hsa_signal_t]) -> hsa_status_t: ...
enum_hsa_amd_event_type_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_GPU_MEMORY_FAULT_EVENT = enum_hsa_amd_event_type_s.define('HSA_AMD_GPU_MEMORY_FAULT_EVENT', 0) # type: ignore
HSA_AMD_GPU_HW_EXCEPTION_EVENT = enum_hsa_amd_event_type_s.define('HSA_AMD_GPU_HW_EXCEPTION_EVENT', 1) # type: ignore
HSA_AMD_GPU_MEMORY_ERROR_EVENT = enum_hsa_amd_event_type_s.define('HSA_AMD_GPU_MEMORY_ERROR_EVENT', 2) # type: ignore

hsa_amd_event_type_t: TypeAlias = enum_hsa_amd_event_type_s
hsa_amd_memory_fault_reason_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT', 1) # type: ignore
HSA_AMD_MEMORY_FAULT_READ_ONLY = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_READ_ONLY', 2) # type: ignore
HSA_AMD_MEMORY_FAULT_NX = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_NX', 4) # type: ignore
HSA_AMD_MEMORY_FAULT_HOST_ONLY = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_HOST_ONLY', 8) # type: ignore
HSA_AMD_MEMORY_FAULT_DRAMECC = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_DRAMECC', 16) # type: ignore
HSA_AMD_MEMORY_FAULT_IMPRECISE = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_IMPRECISE', 32) # type: ignore
HSA_AMD_MEMORY_FAULT_SRAMECC = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_SRAMECC', 64) # type: ignore
HSA_AMD_MEMORY_FAULT_HANG = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_HANG', 2147483648) # type: ignore

@c.record
class struct_hsa_amd_gpu_memory_fault_info_s(c.Struct):
  SIZE = 24
  agent: Annotated[hsa_agent_t, 0]
  virtual_address: Annotated[uint64_t, 8]
  fault_reason_mask: Annotated[uint32_t, 16]
hsa_amd_gpu_memory_fault_info_t = struct_hsa_amd_gpu_memory_fault_info_s
hsa_amd_memory_error_reason_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_MEMORY_ERROR_MEMORY_IN_USE = hsa_amd_memory_error_reason_t.define('HSA_AMD_MEMORY_ERROR_MEMORY_IN_USE', 1) # type: ignore

@c.record
class struct_hsa_amd_gpu_memory_error_info_s(c.Struct):
  SIZE = 24
  agent: Annotated[hsa_agent_t, 0]
  virtual_address: Annotated[uint64_t, 8]
  error_reason_mask: Annotated[uint32_t, 16]
hsa_amd_gpu_memory_error_info_t = struct_hsa_amd_gpu_memory_error_info_s
hsa_amd_hw_exception_reset_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER = hsa_amd_hw_exception_reset_type_t.define('HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER', 1) # type: ignore

hsa_amd_hw_exception_reset_cause_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG = hsa_amd_hw_exception_reset_cause_t.define('HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG', 1) # type: ignore
HSA_AMD_HW_EXCEPTION_CAUSE_ECC = hsa_amd_hw_exception_reset_cause_t.define('HSA_AMD_HW_EXCEPTION_CAUSE_ECC', 2) # type: ignore

@c.record
class struct_hsa_amd_gpu_hw_exception_info_s(c.Struct):
  SIZE = 16
  agent: Annotated[hsa_agent_t, 0]
  reset_type: Annotated[hsa_amd_hw_exception_reset_type_t, 8]
  reset_cause: Annotated[hsa_amd_hw_exception_reset_cause_t, 12]
hsa_amd_gpu_hw_exception_info_t = struct_hsa_amd_gpu_hw_exception_info_s
@c.record
class struct_hsa_amd_event_s(c.Struct):
  SIZE = 32
  event_type: Annotated[hsa_amd_event_type_t, 0]
  memory_fault: Annotated[hsa_amd_gpu_memory_fault_info_t, 8]
  hw_exception: Annotated[hsa_amd_gpu_hw_exception_info_t, 8]
  memory_error: Annotated[hsa_amd_gpu_memory_error_info_t, 8]
hsa_amd_event_t = struct_hsa_amd_event_s
hsa_amd_system_event_callback_t = c.CFUNCTYPE(hsa_status_t, c.POINTER[struct_hsa_amd_event_s], c.POINTER[None])
@dll.bind
def hsa_amd_register_system_event_handler(callback:hsa_amd_system_event_callback_t, data:c.POINTER[None]) -> hsa_status_t: ...
enum_hsa_amd_queue_priority_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_QUEUE_PRIORITY_LOW = enum_hsa_amd_queue_priority_s.define('HSA_AMD_QUEUE_PRIORITY_LOW', 0) # type: ignore
HSA_AMD_QUEUE_PRIORITY_NORMAL = enum_hsa_amd_queue_priority_s.define('HSA_AMD_QUEUE_PRIORITY_NORMAL', 1) # type: ignore
HSA_AMD_QUEUE_PRIORITY_HIGH = enum_hsa_amd_queue_priority_s.define('HSA_AMD_QUEUE_PRIORITY_HIGH', 2) # type: ignore

hsa_amd_queue_priority_t: TypeAlias = enum_hsa_amd_queue_priority_s
@dll.bind
def hsa_amd_queue_set_priority(queue:c.POINTER[hsa_queue_t], priority:hsa_amd_queue_priority_t) -> hsa_status_t: ...
hsa_amd_queue_create_flag_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_QUEUE_CREATE_SYSTEM_MEM = hsa_amd_queue_create_flag_t.define('HSA_AMD_QUEUE_CREATE_SYSTEM_MEM', 0) # type: ignore
HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF = hsa_amd_queue_create_flag_t.define('HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF', 1) # type: ignore
HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR = hsa_amd_queue_create_flag_t.define('HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR', 2) # type: ignore

hsa_amd_deallocation_callback_t = c.CFUNCTYPE(None, c.POINTER[None], c.POINTER[None])
@dll.bind
def hsa_amd_register_deallocation_callback(ptr:c.POINTER[None], callback:hsa_amd_deallocation_callback_t, user_data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_deregister_deallocation_callback(ptr:c.POINTER[None], callback:hsa_amd_deallocation_callback_t) -> hsa_status_t: ...
enum_hsa_amd_svm_model_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED = enum_hsa_amd_svm_model_s.define('HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED', 0) # type: ignore
HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED = enum_hsa_amd_svm_model_s.define('HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED', 1) # type: ignore
HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE = enum_hsa_amd_svm_model_s.define('HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE', 2) # type: ignore

hsa_amd_svm_model_t: TypeAlias = enum_hsa_amd_svm_model_s
enum_hsa_amd_svm_attribute_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG', 0) # type: ignore
HSA_AMD_SVM_ATTRIB_READ_ONLY = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_READ_ONLY', 1) # type: ignore
HSA_AMD_SVM_ATTRIB_HIVE_LOCAL = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_HIVE_LOCAL', 2) # type: ignore
HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY', 3) # type: ignore
HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION', 4) # type: ignore
HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION', 5) # type: ignore
HSA_AMD_SVM_ATTRIB_READ_MOSTLY = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_READ_MOSTLY', 6) # type: ignore
HSA_AMD_SVM_ATTRIB_GPU_EXEC = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_GPU_EXEC', 7) # type: ignore
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE', 512) # type: ignore
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE', 513) # type: ignore
HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS', 514) # type: ignore
HSA_AMD_SVM_ATTRIB_ACCESS_QUERY = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_ACCESS_QUERY', 515) # type: ignore

hsa_amd_svm_attribute_t: TypeAlias = enum_hsa_amd_svm_attribute_s
@c.record
class struct_hsa_amd_svm_attribute_pair_s(c.Struct):
  SIZE = 16
  attribute: Annotated[uint64_t, 0]
  value: Annotated[uint64_t, 8]
hsa_amd_svm_attribute_pair_t = struct_hsa_amd_svm_attribute_pair_s
@dll.bind
def hsa_amd_svm_attributes_set(ptr:c.POINTER[None], size:size_t, attribute_list:c.POINTER[hsa_amd_svm_attribute_pair_t], attribute_count:size_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_svm_attributes_get(ptr:c.POINTER[None], size:size_t, attribute_list:c.POINTER[hsa_amd_svm_attribute_pair_t], attribute_count:size_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_svm_prefetch_async(ptr:c.POINTER[None], size:size_t, agent:hsa_agent_t, num_dep_signals:uint32_t, dep_signals:c.POINTER[hsa_signal_t], completion_signal:hsa_signal_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_spm_acquire(preferred_agent:hsa_agent_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_spm_release(preferred_agent:hsa_agent_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_spm_set_dest_buffer(preferred_agent:hsa_agent_t, size_in_bytes:size_t, timeout:c.POINTER[uint32_t], size_copied:c.POINTER[uint32_t], dest:c.POINTER[None], is_data_loss:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_portable_export_dmabuf(ptr:c.POINTER[None], size:size_t, dmabuf:c.POINTER[Annotated[int, ctypes.c_int32]], offset:c.POINTER[uint64_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_portable_export_dmabuf_v2(ptr:c.POINTER[None], size:size_t, dmabuf:c.POINTER[Annotated[int, ctypes.c_int32]], offset:c.POINTER[uint64_t], flags:uint64_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_portable_close_dmabuf(dmabuf:Annotated[int, ctypes.c_int32]) -> hsa_status_t: ...
enum_hsa_amd_vmem_address_reserve_flag_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_VMEM_ADDRESS_NO_REGISTER = enum_hsa_amd_vmem_address_reserve_flag_s.define('HSA_AMD_VMEM_ADDRESS_NO_REGISTER', 1) # type: ignore

hsa_amd_vmem_address_reserve_flag_t: TypeAlias = enum_hsa_amd_vmem_address_reserve_flag_s
@dll.bind
def hsa_amd_vmem_address_reserve(va:c.POINTER[c.POINTER[None]], size:size_t, address:uint64_t, flags:uint64_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_address_reserve_align(va:c.POINTER[c.POINTER[None]], size:size_t, address:uint64_t, alignment:uint64_t, flags:uint64_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_address_free(va:c.POINTER[None], size:size_t) -> hsa_status_t: ...
@c.record
class struct_hsa_amd_vmem_alloc_handle_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_amd_vmem_alloc_handle_t = struct_hsa_amd_vmem_alloc_handle_s
hsa_amd_memory_type_t = CEnum(Annotated[int, ctypes.c_uint32])
MEMORY_TYPE_NONE = hsa_amd_memory_type_t.define('MEMORY_TYPE_NONE', 0) # type: ignore
MEMORY_TYPE_PINNED = hsa_amd_memory_type_t.define('MEMORY_TYPE_PINNED', 1) # type: ignore

@dll.bind
def hsa_amd_vmem_handle_create(pool:hsa_amd_memory_pool_t, size:size_t, type:hsa_amd_memory_type_t, flags:uint64_t, memory_handle:c.POINTER[hsa_amd_vmem_alloc_handle_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_handle_release(memory_handle:hsa_amd_vmem_alloc_handle_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_map(va:c.POINTER[None], size:size_t, in_offset:size_t, memory_handle:hsa_amd_vmem_alloc_handle_t, flags:uint64_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_unmap(va:c.POINTER[None], size:size_t) -> hsa_status_t: ...
@c.record
class struct_hsa_amd_memory_access_desc_s(c.Struct):
  SIZE = 16
  permissions: Annotated[hsa_access_permission_t, 0]
  agent_handle: Annotated[hsa_agent_t, 8]
hsa_amd_memory_access_desc_t = struct_hsa_amd_memory_access_desc_s
@dll.bind
def hsa_amd_vmem_set_access(va:c.POINTER[None], size:size_t, desc:c.POINTER[hsa_amd_memory_access_desc_t], desc_cnt:size_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_get_access(va:c.POINTER[None], perms:c.POINTER[hsa_access_permission_t], agent_handle:hsa_agent_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_export_shareable_handle(dmabuf_fd:c.POINTER[Annotated[int, ctypes.c_int32]], handle:hsa_amd_vmem_alloc_handle_t, flags:uint64_t) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_import_shareable_handle(dmabuf_fd:Annotated[int, ctypes.c_int32], handle:c.POINTER[hsa_amd_vmem_alloc_handle_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_retain_alloc_handle(memory_handle:c.POINTER[hsa_amd_vmem_alloc_handle_t], addr:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_vmem_get_alloc_properties_from_handle(memory_handle:hsa_amd_vmem_alloc_handle_t, pool:c.POINTER[hsa_amd_memory_pool_t], type:c.POINTER[hsa_amd_memory_type_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_agent_set_async_scratch_limit(agent:hsa_agent_t, threshold:size_t) -> hsa_status_t: ...
hsa_queue_info_attribute_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_QUEUE_INFO_AGENT = hsa_queue_info_attribute_t.define('HSA_AMD_QUEUE_INFO_AGENT', 0) # type: ignore
HSA_AMD_QUEUE_INFO_DOORBELL_ID = hsa_queue_info_attribute_t.define('HSA_AMD_QUEUE_INFO_DOORBELL_ID', 1) # type: ignore

@dll.bind
def hsa_amd_queue_get_info(queue:c.POINTER[hsa_queue_t], attribute:hsa_queue_info_attribute_t, value:c.POINTER[None]) -> hsa_status_t: ...
@c.record
class struct_hsa_amd_ais_file_handle_s(c.Struct):
  SIZE = 8
  handle: Annotated[c.POINTER[None], 0]
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  pad: Annotated[c.Array[uint8_t, Literal[8]], 0]
hsa_amd_ais_file_handle_t = struct_hsa_amd_ais_file_handle_s
int64_t = Annotated[int, ctypes.c_int64]
@dll.bind
def hsa_amd_ais_file_write(handle:hsa_amd_ais_file_handle_t, devicePtr:c.POINTER[None], size:uint64_t, file_offset:int64_t, size_copied:c.POINTER[uint64_t], status:c.POINTER[int32_t]) -> hsa_status_t: ...
@dll.bind
def hsa_amd_ais_file_read(handle:hsa_amd_ais_file_handle_t, devicePtr:c.POINTER[None], size:uint64_t, file_offset:int64_t, size_copied:c.POINTER[uint64_t], status:c.POINTER[int32_t]) -> hsa_status_t: ...
enum_hsa_amd_log_flag_s = CEnum(Annotated[int, ctypes.c_uint32])
HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS = enum_hsa_amd_log_flag_s.define('HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS', 0) # type: ignore
HSA_AMD_LOG_FLAG_AQL = enum_hsa_amd_log_flag_s.define('HSA_AMD_LOG_FLAG_AQL', 0) # type: ignore
HSA_AMD_LOG_FLAG_SDMA = enum_hsa_amd_log_flag_s.define('HSA_AMD_LOG_FLAG_SDMA', 1) # type: ignore
HSA_AMD_LOG_FLAG_INFO = enum_hsa_amd_log_flag_s.define('HSA_AMD_LOG_FLAG_INFO', 2) # type: ignore

hsa_amd_log_flag_t: TypeAlias = enum_hsa_amd_log_flag_s
@dll.bind
def hsa_amd_enable_logging(flags:c.POINTER[uint8_t], file:c.POINTER[None]) -> hsa_status_t: ...
amd_signal_kind64_t = Annotated[int, ctypes.c_int64]
enum_amd_signal_kind_t = CEnum(Annotated[int, ctypes.c_int32])
AMD_SIGNAL_KIND_INVALID = enum_amd_signal_kind_t.define('AMD_SIGNAL_KIND_INVALID', 0) # type: ignore
AMD_SIGNAL_KIND_USER = enum_amd_signal_kind_t.define('AMD_SIGNAL_KIND_USER', 1) # type: ignore
AMD_SIGNAL_KIND_DOORBELL = enum_amd_signal_kind_t.define('AMD_SIGNAL_KIND_DOORBELL', -1) # type: ignore
AMD_SIGNAL_KIND_LEGACY_DOORBELL = enum_amd_signal_kind_t.define('AMD_SIGNAL_KIND_LEGACY_DOORBELL', -2) # type: ignore

@c.record
class struct_amd_signal_s(c.Struct):
  SIZE = 64
  kind: Annotated[amd_signal_kind64_t, 0]
  value: Annotated[int64_t, 8]
  hardware_doorbell_ptr: Annotated[c.POINTER[uint64_t], 8]
  event_mailbox_ptr: Annotated[uint64_t, 16]
  event_id: Annotated[uint32_t, 24]
  reserved1: Annotated[uint32_t, 28]
  start_ts: Annotated[uint64_t, 32]
  end_ts: Annotated[uint64_t, 40]
  queue_ptr: Annotated[c.POINTER[amd_queue_v2_t], 48]
  reserved2: Annotated[uint64_t, 48]
  reserved3: Annotated[c.Array[uint32_t, Literal[2]], 56]
@c.record
class struct_amd_queue_v2_s(c.Struct):
  SIZE = 2304
  hsa_queue: Annotated[hsa_queue_t, 0]
  caps: Annotated[uint32_t, 40]
  reserved1: Annotated[c.Array[uint32_t, Literal[3]], 44]
  write_dispatch_id: Annotated[uint64_t, 56]
  group_segment_aperture_base_hi: Annotated[uint32_t, 64]
  private_segment_aperture_base_hi: Annotated[uint32_t, 68]
  max_cu_id: Annotated[uint32_t, 72]
  max_wave_id: Annotated[uint32_t, 76]
  max_legacy_doorbell_dispatch_id_plus_1: Annotated[uint64_t, 80]
  legacy_doorbell_lock: Annotated[uint32_t, 88]
  reserved2: Annotated[c.Array[uint32_t, Literal[9]], 92]
  read_dispatch_id: Annotated[uint64_t, 128]
  read_dispatch_id_field_base_byte_offset: Annotated[uint32_t, 136]
  compute_tmpring_size: Annotated[uint32_t, 140]
  scratch_resource_descriptor: Annotated[c.Array[uint32_t, Literal[4]], 144]
  scratch_backing_memory_location: Annotated[uint64_t, 160]
  scratch_backing_memory_byte_size: Annotated[uint64_t, 168]
  scratch_wave64_lane_byte_size: Annotated[uint32_t, 176]
  queue_properties: Annotated[amd_queue_properties32_t, 180]
  scratch_max_use_index: Annotated[uint64_t, 184]
  queue_inactive_signal: Annotated[hsa_signal_t, 192]
  alt_scratch_max_use_index: Annotated[uint64_t, 200]
  alt_scratch_resource_descriptor: Annotated[c.Array[uint32_t, Literal[4]], 208]
  alt_scratch_backing_memory_location: Annotated[uint64_t, 224]
  alt_scratch_dispatch_limit_x: Annotated[uint32_t, 232]
  alt_scratch_dispatch_limit_y: Annotated[uint32_t, 236]
  alt_scratch_dispatch_limit_z: Annotated[uint32_t, 240]
  alt_scratch_wave64_lane_byte_size: Annotated[uint32_t, 244]
  alt_compute_tmpring_size: Annotated[uint32_t, 248]
  reserved5: Annotated[uint32_t, 252]
  scratch_last_used_index: Annotated[c.Array[scratch_last_used_index_xcc_t, Literal[128]], 256]
amd_queue_v2_t: TypeAlias = struct_amd_queue_v2_s
amd_queue_properties32_t = Annotated[int, ctypes.c_uint32]
@c.record
class struct_scratch_last_used_index_xcc_s(c.Struct):
  SIZE = 16
  main: Annotated[uint64_t, 0]
  alt: Annotated[uint64_t, 8]
scratch_last_used_index_xcc_t: TypeAlias = struct_scratch_last_used_index_xcc_s
amd_signal_t = struct_amd_signal_s
enum_amd_queue_properties_t = CEnum(Annotated[int, ctypes.c_int32])
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_SHIFT', 0) # type: ignore
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_WIDTH', 1) # type: ignore
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER', 1) # type: ignore
AMD_QUEUE_PROPERTIES_IS_PTR64_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_IS_PTR64_SHIFT', 1) # type: ignore
AMD_QUEUE_PROPERTIES_IS_PTR64_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_IS_PTR64_WIDTH', 1) # type: ignore
AMD_QUEUE_PROPERTIES_IS_PTR64 = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_IS_PTR64', 2) # type: ignore
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_SHIFT', 2) # type: ignore
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_WIDTH', 1) # type: ignore
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS', 4) # type: ignore
AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_SHIFT', 3) # type: ignore
AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_WIDTH', 1) # type: ignore
AMD_QUEUE_PROPERTIES_ENABLE_PROFILING = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_PROFILING', 8) # type: ignore
AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_SHIFT', 4) # type: ignore
AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_WIDTH', 1) # type: ignore
AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE', 16) # type: ignore
AMD_QUEUE_PROPERTIES_RESERVED1_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_RESERVED1_SHIFT', 5) # type: ignore
AMD_QUEUE_PROPERTIES_RESERVED1_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_RESERVED1_WIDTH', 27) # type: ignore
AMD_QUEUE_PROPERTIES_RESERVED1 = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_RESERVED1', -32) # type: ignore

amd_queue_capabilities32_t = Annotated[int, ctypes.c_uint32]
enum_amd_queue_capabilities_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_SHIFT = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_SHIFT', 0) # type: ignore
AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_WIDTH = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_WIDTH', 1) # type: ignore
AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM', 1) # type: ignore
AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_SHIFT = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_SHIFT', 1) # type: ignore
AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_WIDTH = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_WIDTH', 1) # type: ignore
AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM', 2) # type: ignore

@c.record
class struct_amd_queue_s(c.Struct):
  SIZE = 256
  hsa_queue: Annotated[hsa_queue_t, 0]
  caps: Annotated[uint32_t, 40]
  reserved1: Annotated[c.Array[uint32_t, Literal[3]], 44]
  write_dispatch_id: Annotated[uint64_t, 56]
  group_segment_aperture_base_hi: Annotated[uint32_t, 64]
  private_segment_aperture_base_hi: Annotated[uint32_t, 68]
  max_cu_id: Annotated[uint32_t, 72]
  max_wave_id: Annotated[uint32_t, 76]
  max_legacy_doorbell_dispatch_id_plus_1: Annotated[uint64_t, 80]
  legacy_doorbell_lock: Annotated[uint32_t, 88]
  reserved2: Annotated[c.Array[uint32_t, Literal[9]], 92]
  read_dispatch_id: Annotated[uint64_t, 128]
  read_dispatch_id_field_base_byte_offset: Annotated[uint32_t, 136]
  compute_tmpring_size: Annotated[uint32_t, 140]
  scratch_resource_descriptor: Annotated[c.Array[uint32_t, Literal[4]], 144]
  scratch_backing_memory_location: Annotated[uint64_t, 160]
  reserved3: Annotated[c.Array[uint32_t, Literal[2]], 168]
  scratch_wave64_lane_byte_size: Annotated[uint32_t, 176]
  queue_properties: Annotated[amd_queue_properties32_t, 180]
  reserved4: Annotated[c.Array[uint32_t, Literal[2]], 184]
  queue_inactive_signal: Annotated[hsa_signal_t, 192]
  reserved5: Annotated[c.Array[uint32_t, Literal[14]], 200]
amd_queue_t = struct_amd_queue_s
amd_kernel_code_version32_t = Annotated[int, ctypes.c_uint32]
enum_amd_kernel_code_version_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_KERNEL_CODE_VERSION_MAJOR = enum_amd_kernel_code_version_t.define('AMD_KERNEL_CODE_VERSION_MAJOR', 1) # type: ignore
AMD_KERNEL_CODE_VERSION_MINOR = enum_amd_kernel_code_version_t.define('AMD_KERNEL_CODE_VERSION_MINOR', 1) # type: ignore

amd_machine_kind16_t = Annotated[int, ctypes.c_uint16]
enum_amd_machine_kind_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_MACHINE_KIND_UNDEFINED = enum_amd_machine_kind_t.define('AMD_MACHINE_KIND_UNDEFINED', 0) # type: ignore
AMD_MACHINE_KIND_AMDGPU = enum_amd_machine_kind_t.define('AMD_MACHINE_KIND_AMDGPU', 1) # type: ignore

amd_machine_version16_t = Annotated[int, ctypes.c_uint16]
enum_amd_float_round_mode_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_FLOAT_ROUND_MODE_NEAREST_EVEN = enum_amd_float_round_mode_t.define('AMD_FLOAT_ROUND_MODE_NEAREST_EVEN', 0) # type: ignore
AMD_FLOAT_ROUND_MODE_PLUS_INFINITY = enum_amd_float_round_mode_t.define('AMD_FLOAT_ROUND_MODE_PLUS_INFINITY', 1) # type: ignore
AMD_FLOAT_ROUND_MODE_MINUS_INFINITY = enum_amd_float_round_mode_t.define('AMD_FLOAT_ROUND_MODE_MINUS_INFINITY', 2) # type: ignore
AMD_FLOAT_ROUND_MODE_ZERO = enum_amd_float_round_mode_t.define('AMD_FLOAT_ROUND_MODE_ZERO', 3) # type: ignore

enum_amd_float_denorm_mode_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT = enum_amd_float_denorm_mode_t.define('AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT', 0) # type: ignore
AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT = enum_amd_float_denorm_mode_t.define('AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT', 1) # type: ignore
AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE = enum_amd_float_denorm_mode_t.define('AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE', 2) # type: ignore
AMD_FLOAT_DENORM_MODE_NO_FLUSH = enum_amd_float_denorm_mode_t.define('AMD_FLOAT_DENORM_MODE_NO_FLUSH', 3) # type: ignore

amd_compute_pgm_rsrc_one32_t = Annotated[int, ctypes.c_uint32]
enum_amd_compute_pgm_rsrc_one_t = CEnum(Annotated[int, ctypes.c_int32])
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT', 0) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH', 6) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT', 63) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT', 6) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH', 4) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT', 960) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT', 10) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH', 2) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY', 3072) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT', 12) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH', 2) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32', 12288) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT', 14) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH', 2) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64', 49152) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT', 16) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH', 2) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32', 196608) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT', 18) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH', 2) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64', 786432) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT', 20) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_PRIV = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIV', 1048576) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT', 21) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP', 2097152) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT', 22) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE', 4194304) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT', 23) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE', 8388608) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT', 24) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_BULKY = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_BULKY', 16777216) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT', 25) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER', 33554432) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT', 26) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH', 6) # type: ignore
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1', -67108864) # type: ignore

enum_amd_system_vgpr_workitem_id_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_SYSTEM_VGPR_WORKITEM_ID_X = enum_amd_system_vgpr_workitem_id_t.define('AMD_SYSTEM_VGPR_WORKITEM_ID_X', 0) # type: ignore
AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y = enum_amd_system_vgpr_workitem_id_t.define('AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y', 1) # type: ignore
AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z = enum_amd_system_vgpr_workitem_id_t.define('AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z', 2) # type: ignore
AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED = enum_amd_system_vgpr_workitem_id_t.define('AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED', 3) # type: ignore

amd_compute_pgm_rsrc_two32_t = Annotated[int, ctypes.c_uint32]
enum_amd_compute_pgm_rsrc_two_t = CEnum(Annotated[int, ctypes.c_int32])
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT', 0) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH', 5) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT', 62) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT', 6) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER', 64) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT', 7) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X', 128) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT', 8) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y', 256) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT', 9) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z', 512) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT', 10) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO', 1024) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT', 11) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH', 2) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID', 6144) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT', 13) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH', 8192) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT', 14) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION', 16384) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT', 15) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH', 9) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE', 16744448) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT', 24) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION', 16777216) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT', 25) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE', 33554432) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT', 26) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO', 67108864) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT', 27) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW', 134217728) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT', 28) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW', 268435456) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT', 29) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT', 536870912) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT', 30) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO', 1073741824) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT', 31) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH', 1) # type: ignore
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1 = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1', -2147483648) # type: ignore

enum_amd_element_byte_size_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_ELEMENT_BYTE_SIZE_2 = enum_amd_element_byte_size_t.define('AMD_ELEMENT_BYTE_SIZE_2', 0) # type: ignore
AMD_ELEMENT_BYTE_SIZE_4 = enum_amd_element_byte_size_t.define('AMD_ELEMENT_BYTE_SIZE_4', 1) # type: ignore
AMD_ELEMENT_BYTE_SIZE_8 = enum_amd_element_byte_size_t.define('AMD_ELEMENT_BYTE_SIZE_8', 2) # type: ignore
AMD_ELEMENT_BYTE_SIZE_16 = enum_amd_element_byte_size_t.define('AMD_ELEMENT_BYTE_SIZE_16', 3) # type: ignore

amd_kernel_code_properties32_t = Annotated[int, ctypes.c_uint32]
enum_amd_kernel_code_properties_t = CEnum(Annotated[int, ctypes.c_int32])
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT', 0) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR', 2) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT', 2) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR', 4) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT', 3) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR', 8) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT', 4) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID', 16) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT', 5) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT', 32) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT', 6) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE', 64) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT', 7) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X', 128) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT', 8) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y', 256) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT', 9) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z', 512) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_SHIFT', 10) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32 = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32', 1024) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT', 11) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH', 5) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_RESERVED1 = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED1', 63488) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT', 16) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS', 65536) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT', 17) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH', 2) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE', 393216) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT', 19) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64 = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_PTR64', 524288) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT', 20) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK', 1048576) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT', 21) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED', 2097152) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT', 22) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH', 1) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED', 4194304) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT', 23) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH', 9) # type: ignore
AMD_KERNEL_CODE_PROPERTIES_RESERVED2 = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED2', -8388608) # type: ignore

amd_powertwo8_t = Annotated[int, ctypes.c_ubyte]
enum_amd_powertwo_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_POWERTWO_1 = enum_amd_powertwo_t.define('AMD_POWERTWO_1', 0) # type: ignore
AMD_POWERTWO_2 = enum_amd_powertwo_t.define('AMD_POWERTWO_2', 1) # type: ignore
AMD_POWERTWO_4 = enum_amd_powertwo_t.define('AMD_POWERTWO_4', 2) # type: ignore
AMD_POWERTWO_8 = enum_amd_powertwo_t.define('AMD_POWERTWO_8', 3) # type: ignore
AMD_POWERTWO_16 = enum_amd_powertwo_t.define('AMD_POWERTWO_16', 4) # type: ignore
AMD_POWERTWO_32 = enum_amd_powertwo_t.define('AMD_POWERTWO_32', 5) # type: ignore
AMD_POWERTWO_64 = enum_amd_powertwo_t.define('AMD_POWERTWO_64', 6) # type: ignore
AMD_POWERTWO_128 = enum_amd_powertwo_t.define('AMD_POWERTWO_128', 7) # type: ignore
AMD_POWERTWO_256 = enum_amd_powertwo_t.define('AMD_POWERTWO_256', 8) # type: ignore

amd_enabled_control_directive64_t = Annotated[int, ctypes.c_uint64]
enum_amd_enabled_control_directive_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS', 1) # type: ignore
AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS', 2) # type: ignore
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE', 4) # type: ignore
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE', 8) # type: ignore
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE', 16) # type: ignore
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM', 32) # type: ignore
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE', 64) # type: ignore
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE', 128) # type: ignore
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS', 256) # type: ignore

amd_exception_kind16_t = Annotated[int, ctypes.c_uint16]
enum_amd_exception_kind_t = CEnum(Annotated[int, ctypes.c_uint32])
AMD_EXCEPTION_KIND_INVALID_OPERATION = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_INVALID_OPERATION', 1) # type: ignore
AMD_EXCEPTION_KIND_DIVISION_BY_ZERO = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_DIVISION_BY_ZERO', 2) # type: ignore
AMD_EXCEPTION_KIND_OVERFLOW = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_OVERFLOW', 4) # type: ignore
AMD_EXCEPTION_KIND_UNDERFLOW = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_UNDERFLOW', 8) # type: ignore
AMD_EXCEPTION_KIND_INEXACT = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_INEXACT', 16) # type: ignore

@c.record
class struct_amd_control_directives_s(c.Struct):
  SIZE = 128
  enabled_control_directives: Annotated[amd_enabled_control_directive64_t, 0]
  enable_break_exceptions: Annotated[uint16_t, 8]
  enable_detect_exceptions: Annotated[uint16_t, 10]
  max_dynamic_group_size: Annotated[uint32_t, 12]
  max_flat_grid_size: Annotated[uint64_t, 16]
  max_flat_workgroup_size: Annotated[uint32_t, 24]
  required_dim: Annotated[uint8_t, 28]
  reserved1: Annotated[c.Array[uint8_t, Literal[3]], 29]
  required_grid_size: Annotated[c.Array[uint64_t, Literal[3]], 32]
  required_workgroup_size: Annotated[c.Array[uint32_t, Literal[3]], 56]
  reserved2: Annotated[c.Array[uint8_t, Literal[60]], 68]
amd_control_directives_t = struct_amd_control_directives_s
@c.record
class struct_amd_kernel_code_s(c.Struct):
  SIZE = 256
  amd_kernel_code_version_major: Annotated[amd_kernel_code_version32_t, 0]
  amd_kernel_code_version_minor: Annotated[amd_kernel_code_version32_t, 4]
  amd_machine_kind: Annotated[amd_machine_kind16_t, 8]
  amd_machine_version_major: Annotated[amd_machine_version16_t, 10]
  amd_machine_version_minor: Annotated[amd_machine_version16_t, 12]
  amd_machine_version_stepping: Annotated[amd_machine_version16_t, 14]
  kernel_code_entry_byte_offset: Annotated[int64_t, 16]
  kernel_code_prefetch_byte_offset: Annotated[int64_t, 24]
  kernel_code_prefetch_byte_size: Annotated[uint64_t, 32]
  max_scratch_backing_memory_byte_size: Annotated[uint64_t, 40]
  compute_pgm_rsrc1: Annotated[amd_compute_pgm_rsrc_one32_t, 48]
  compute_pgm_rsrc2: Annotated[amd_compute_pgm_rsrc_two32_t, 52]
  kernel_code_properties: Annotated[amd_kernel_code_properties32_t, 56]
  workitem_private_segment_byte_size: Annotated[uint32_t, 60]
  workgroup_group_segment_byte_size: Annotated[uint32_t, 64]
  gds_segment_byte_size: Annotated[uint32_t, 68]
  kernarg_segment_byte_size: Annotated[uint64_t, 72]
  workgroup_fbarrier_count: Annotated[uint32_t, 80]
  wavefront_sgpr_count: Annotated[uint16_t, 84]
  workitem_vgpr_count: Annotated[uint16_t, 86]
  reserved_vgpr_first: Annotated[uint16_t, 88]
  reserved_vgpr_count: Annotated[uint16_t, 90]
  reserved_sgpr_first: Annotated[uint16_t, 92]
  reserved_sgpr_count: Annotated[uint16_t, 94]
  debug_wavefront_private_segment_offset_sgpr: Annotated[uint16_t, 96]
  debug_private_segment_buffer_sgpr: Annotated[uint16_t, 98]
  kernarg_segment_alignment: Annotated[amd_powertwo8_t, 100]
  group_segment_alignment: Annotated[amd_powertwo8_t, 101]
  private_segment_alignment: Annotated[amd_powertwo8_t, 102]
  wavefront_size: Annotated[amd_powertwo8_t, 103]
  call_convention: Annotated[int32_t, 104]
  reserved1: Annotated[c.Array[uint8_t, Literal[12]], 108]
  runtime_loader_kernel_symbol: Annotated[uint64_t, 120]
  control_directives: Annotated[amd_control_directives_t, 128]
amd_kernel_code_t = struct_amd_kernel_code_s
@c.record
class struct_amd_runtime_loader_debug_info_s(c.Struct):
  SIZE = 32
  elf_raw: Annotated[c.POINTER[None], 0]
  elf_size: Annotated[size_t, 8]
  kernel_name: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 16]
  owning_segment: Annotated[c.POINTER[None], 24]
amd_runtime_loader_debug_info_t = struct_amd_runtime_loader_debug_info_s
class struct_BrigModuleHeader(ctypes.Structure): pass
BrigModule_t = c.POINTER[struct_BrigModuleHeader]
_anonenum1 = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_STATUS_ERROR_INVALID_PROGRAM = _anonenum1.define('HSA_EXT_STATUS_ERROR_INVALID_PROGRAM', 8192) # type: ignore
HSA_EXT_STATUS_ERROR_INVALID_MODULE = _anonenum1.define('HSA_EXT_STATUS_ERROR_INVALID_MODULE', 8193) # type: ignore
HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE = _anonenum1.define('HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE', 8194) # type: ignore
HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED = _anonenum1.define('HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED', 8195) # type: ignore
HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH = _anonenum1.define('HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH', 8196) # type: ignore
HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED = _anonenum1.define('HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED', 8197) # type: ignore
HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH = _anonenum1.define('HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH', 8198) # type: ignore

hsa_ext_module_t = c.POINTER[struct_BrigModuleHeader]
@c.record
class struct_hsa_ext_program_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_ext_program_t = struct_hsa_ext_program_s
@dll.bind
def hsa_ext_program_create(machine_model:hsa_machine_model_t, profile:hsa_profile_t, default_float_rounding_mode:hsa_default_float_rounding_mode_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], program:c.POINTER[hsa_ext_program_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_program_destroy(program:hsa_ext_program_t) -> hsa_status_t: ...
@dll.bind
def hsa_ext_program_add_module(program:hsa_ext_program_t, module:hsa_ext_module_t) -> hsa_status_t: ...
@dll.bind
def hsa_ext_program_iterate_modules(program:hsa_ext_program_t, callback:c.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_ext_module_t, c.POINTER[None]), data:c.POINTER[None]) -> hsa_status_t: ...
hsa_ext_program_info_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_PROGRAM_INFO_MACHINE_MODEL = hsa_ext_program_info_t.define('HSA_EXT_PROGRAM_INFO_MACHINE_MODEL', 0) # type: ignore
HSA_EXT_PROGRAM_INFO_PROFILE = hsa_ext_program_info_t.define('HSA_EXT_PROGRAM_INFO_PROFILE', 1) # type: ignore
HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE = hsa_ext_program_info_t.define('HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE', 2) # type: ignore

@dll.bind
def hsa_ext_program_get_info(program:hsa_ext_program_t, attribute:hsa_ext_program_info_t, value:c.POINTER[None]) -> hsa_status_t: ...
hsa_ext_finalizer_call_convention_t = CEnum(Annotated[int, ctypes.c_int32])
HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO = hsa_ext_finalizer_call_convention_t.define('HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO', -1) # type: ignore

@c.record
class struct_hsa_ext_control_directives_s(c.Struct):
  SIZE = 144
  control_directives_mask: Annotated[uint64_t, 0]
  break_exceptions_mask: Annotated[uint16_t, 8]
  detect_exceptions_mask: Annotated[uint16_t, 10]
  max_dynamic_group_size: Annotated[uint32_t, 12]
  max_flat_grid_size: Annotated[uint64_t, 16]
  max_flat_workgroup_size: Annotated[uint32_t, 24]
  reserved1: Annotated[uint32_t, 28]
  required_grid_size: Annotated[c.Array[uint64_t, Literal[3]], 32]
  required_workgroup_size: Annotated[hsa_dim3_t, 56]
  required_dim: Annotated[uint8_t, 68]
  reserved2: Annotated[c.Array[uint8_t, Literal[75]], 69]
hsa_ext_control_directives_t = struct_hsa_ext_control_directives_s
@dll.bind
def hsa_ext_program_finalize(program:hsa_ext_program_t, isa:hsa_isa_t, call_convention:int32_t, control_directives:hsa_ext_control_directives_t, options:c.POINTER[Annotated[bytes, ctypes.c_char]], code_object_type:hsa_code_object_type_t, code_object:c.POINTER[hsa_code_object_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_ext_finalizer_1_00_pfn_s(c.Struct):
  SIZE = 48
  hsa_ext_program_create: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_machine_model_t, hsa_profile_t, hsa_default_float_rounding_mode_t, c.POINTER[Annotated[bytes, ctypes.c_char]], c.POINTER[hsa_ext_program_t]), 0]
  hsa_ext_program_destroy: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_ext_program_t), 8]
  hsa_ext_program_add_module: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_ext_module_t), 16]
  hsa_ext_program_iterate_modules: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, c.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_ext_module_t, c.POINTER[None]), c.POINTER[None]), 24]
  hsa_ext_program_get_info: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_ext_program_info_t, c.POINTER[None]), 32]
  hsa_ext_program_finalize: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_isa_t, int32_t, hsa_ext_control_directives_t, c.POINTER[Annotated[bytes, ctypes.c_char]], hsa_code_object_type_t, c.POINTER[hsa_code_object_t]), 40]
hsa_ext_finalizer_1_00_pfn_t = struct_hsa_ext_finalizer_1_00_pfn_s
_anonenum2 = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED = _anonenum2.define('HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED', 12288) # type: ignore
HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED = _anonenum2.define('HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED', 12289) # type: ignore
HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED = _anonenum2.define('HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED', 12290) # type: ignore
HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED = _anonenum2.define('HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED', 12291) # type: ignore

_anonenum3 = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS', 12288) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS', 12289) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS', 12290) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS', 12291) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS', 12292) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS', 12293) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS', 12294) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS', 12295) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS', 12296) # type: ignore
HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES = _anonenum3.define('HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES', 12297) # type: ignore
HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES = _anonenum3.define('HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES', 12298) # type: ignore
HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS = _anonenum3.define('HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS', 12299) # type: ignore
HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT', 12300) # type: ignore

hsa_ext_image_channel_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8', 0) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16', 1) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8', 2) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16', 3) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24', 4) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555', 5) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565', 6) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010', 7) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8', 8) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16', 9) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32', 10) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8', 11) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16', 12) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32', 13) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT', 14) # type: ignore
HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT', 15) # type: ignore

hsa_ext_image_channel_order_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_IMAGE_CHANNEL_ORDER_A = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_A', 0) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_R = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_R', 1) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_RX = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RX', 2) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_RG = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RG', 3) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_RGX = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RGX', 4) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_RA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RA', 5) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_RGB = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RGB', 6) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX', 7) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA', 8) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA', 9) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB', 10) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR', 11) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB', 12) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX', 13) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA', 14) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA', 15) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY', 16) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE', 17) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH', 18) # type: ignore
HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL', 19) # type: ignore

hsa_ext_image_capability_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED', 0) # type: ignore
HSA_EXT_IMAGE_CAPABILITY_READ_ONLY = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_READ_ONLY', 1) # type: ignore
HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY', 2) # type: ignore
HSA_EXT_IMAGE_CAPABILITY_READ_WRITE = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_READ_WRITE', 4) # type: ignore
HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE', 8) # type: ignore
HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT', 16) # type: ignore

hsa_ext_image_data_layout_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE = hsa_ext_image_data_layout_t.define('HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE', 0) # type: ignore
HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR = hsa_ext_image_data_layout_t.define('HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR', 1) # type: ignore

@dll.bind
def hsa_ext_image_get_capability(agent:hsa_agent_t, geometry:hsa_ext_image_geometry_t, image_format:c.POINTER[hsa_ext_image_format_t], capability_mask:c.POINTER[uint32_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_image_get_capability_with_layout(agent:hsa_agent_t, geometry:hsa_ext_image_geometry_t, image_format:c.POINTER[hsa_ext_image_format_t], image_data_layout:hsa_ext_image_data_layout_t, capability_mask:c.POINTER[uint32_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_ext_image_data_info_s(c.Struct):
  SIZE = 16
  size: Annotated[size_t, 0]
  alignment: Annotated[size_t, 8]
hsa_ext_image_data_info_t = struct_hsa_ext_image_data_info_s
@dll.bind
def hsa_ext_image_data_get_info(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], access_permission:hsa_access_permission_t, image_data_info:c.POINTER[hsa_ext_image_data_info_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_image_data_get_info_with_layout(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], access_permission:hsa_access_permission_t, image_data_layout:hsa_ext_image_data_layout_t, image_data_row_pitch:size_t, image_data_slice_pitch:size_t, image_data_info:c.POINTER[hsa_ext_image_data_info_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_image_create(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], image_data:c.POINTER[None], access_permission:hsa_access_permission_t, image:c.POINTER[hsa_ext_image_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_image_create_with_layout(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], image_data:c.POINTER[None], access_permission:hsa_access_permission_t, image_data_layout:hsa_ext_image_data_layout_t, image_data_row_pitch:size_t, image_data_slice_pitch:size_t, image:c.POINTER[hsa_ext_image_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_image_destroy(agent:hsa_agent_t, image:hsa_ext_image_t) -> hsa_status_t: ...
@dll.bind
def hsa_ext_image_copy(agent:hsa_agent_t, src_image:hsa_ext_image_t, src_offset:c.POINTER[hsa_dim3_t], dst_image:hsa_ext_image_t, dst_offset:c.POINTER[hsa_dim3_t], range:c.POINTER[hsa_dim3_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_ext_image_region_s(c.Struct):
  SIZE = 24
  offset: Annotated[hsa_dim3_t, 0]
  range: Annotated[hsa_dim3_t, 12]
hsa_ext_image_region_t = struct_hsa_ext_image_region_s
@dll.bind
def hsa_ext_image_import(agent:hsa_agent_t, src_memory:c.POINTER[None], src_row_pitch:size_t, src_slice_pitch:size_t, dst_image:hsa_ext_image_t, image_region:c.POINTER[hsa_ext_image_region_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_image_export(agent:hsa_agent_t, src_image:hsa_ext_image_t, dst_memory:c.POINTER[None], dst_row_pitch:size_t, dst_slice_pitch:size_t, image_region:c.POINTER[hsa_ext_image_region_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_image_clear(agent:hsa_agent_t, image:hsa_ext_image_t, data:c.POINTER[None], image_region:c.POINTER[hsa_ext_image_region_t]) -> hsa_status_t: ...
@c.record
class struct_hsa_ext_sampler_s(c.Struct):
  SIZE = 8
  handle: Annotated[uint64_t, 0]
hsa_ext_sampler_t = struct_hsa_ext_sampler_s
hsa_ext_sampler_addressing_mode_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED', 0) # type: ignore
HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE', 1) # type: ignore
HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER', 2) # type: ignore
HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT', 3) # type: ignore
HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT', 4) # type: ignore

hsa_ext_sampler_addressing_mode32_t = Annotated[int, ctypes.c_uint32]
hsa_ext_sampler_coordinate_mode_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED = hsa_ext_sampler_coordinate_mode_t.define('HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED', 0) # type: ignore
HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED = hsa_ext_sampler_coordinate_mode_t.define('HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED', 1) # type: ignore

hsa_ext_sampler_coordinate_mode32_t = Annotated[int, ctypes.c_uint32]
hsa_ext_sampler_filter_mode_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_EXT_SAMPLER_FILTER_MODE_NEAREST = hsa_ext_sampler_filter_mode_t.define('HSA_EXT_SAMPLER_FILTER_MODE_NEAREST', 0) # type: ignore
HSA_EXT_SAMPLER_FILTER_MODE_LINEAR = hsa_ext_sampler_filter_mode_t.define('HSA_EXT_SAMPLER_FILTER_MODE_LINEAR', 1) # type: ignore

hsa_ext_sampler_filter_mode32_t = Annotated[int, ctypes.c_uint32]
@c.record
class struct_hsa_ext_sampler_descriptor_s(c.Struct):
  SIZE = 12
  coordinate_mode: Annotated[hsa_ext_sampler_coordinate_mode32_t, 0]
  filter_mode: Annotated[hsa_ext_sampler_filter_mode32_t, 4]
  address_mode: Annotated[hsa_ext_sampler_addressing_mode32_t, 8]
hsa_ext_sampler_descriptor_t = struct_hsa_ext_sampler_descriptor_s
@c.record
class struct_hsa_ext_sampler_descriptor_v2_s(c.Struct):
  SIZE = 20
  coordinate_mode: Annotated[hsa_ext_sampler_coordinate_mode32_t, 0]
  filter_mode: Annotated[hsa_ext_sampler_filter_mode32_t, 4]
  address_modes: Annotated[c.Array[hsa_ext_sampler_addressing_mode32_t, Literal[3]], 8]
hsa_ext_sampler_descriptor_v2_t = struct_hsa_ext_sampler_descriptor_v2_s
@dll.bind
def hsa_ext_sampler_create(agent:hsa_agent_t, sampler_descriptor:c.POINTER[hsa_ext_sampler_descriptor_t], sampler:c.POINTER[hsa_ext_sampler_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_sampler_create_v2(agent:hsa_agent_t, sampler_descriptor:c.POINTER[hsa_ext_sampler_descriptor_v2_t], sampler:c.POINTER[hsa_ext_sampler_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ext_sampler_destroy(agent:hsa_agent_t, sampler:hsa_ext_sampler_t) -> hsa_status_t: ...
@c.record
class struct_hsa_ext_images_1_00_pfn_s(c.Struct):
  SIZE = 80
  hsa_ext_image_get_capability: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_geometry_t, c.POINTER[hsa_ext_image_format_t], c.POINTER[uint32_t]), 0]
  hsa_ext_image_data_get_info: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], hsa_access_permission_t, c.POINTER[hsa_ext_image_data_info_t]), 8]
  hsa_ext_image_create: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], c.POINTER[None], hsa_access_permission_t, c.POINTER[hsa_ext_image_t]), 16]
  hsa_ext_image_destroy: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t), 24]
  hsa_ext_image_copy: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, c.POINTER[hsa_dim3_t], hsa_ext_image_t, c.POINTER[hsa_dim3_t], c.POINTER[hsa_dim3_t]), 32]
  hsa_ext_image_import: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[None], size_t, size_t, hsa_ext_image_t, c.POINTER[hsa_ext_image_region_t]), 40]
  hsa_ext_image_export: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, c.POINTER[None], size_t, size_t, c.POINTER[hsa_ext_image_region_t]), 48]
  hsa_ext_image_clear: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, c.POINTER[None], c.POINTER[hsa_ext_image_region_t]), 56]
  hsa_ext_sampler_create: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_sampler_descriptor_t], c.POINTER[hsa_ext_sampler_t]), 64]
  hsa_ext_sampler_destroy: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_sampler_t), 72]
hsa_ext_images_1_00_pfn_t = struct_hsa_ext_images_1_00_pfn_s
@c.record
class struct_hsa_ext_images_1_pfn_s(c.Struct):
  SIZE = 112
  hsa_ext_image_get_capability: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_geometry_t, c.POINTER[hsa_ext_image_format_t], c.POINTER[uint32_t]), 0]
  hsa_ext_image_data_get_info: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], hsa_access_permission_t, c.POINTER[hsa_ext_image_data_info_t]), 8]
  hsa_ext_image_create: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], c.POINTER[None], hsa_access_permission_t, c.POINTER[hsa_ext_image_t]), 16]
  hsa_ext_image_destroy: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t), 24]
  hsa_ext_image_copy: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, c.POINTER[hsa_dim3_t], hsa_ext_image_t, c.POINTER[hsa_dim3_t], c.POINTER[hsa_dim3_t]), 32]
  hsa_ext_image_import: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[None], size_t, size_t, hsa_ext_image_t, c.POINTER[hsa_ext_image_region_t]), 40]
  hsa_ext_image_export: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, c.POINTER[None], size_t, size_t, c.POINTER[hsa_ext_image_region_t]), 48]
  hsa_ext_image_clear: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, c.POINTER[None], c.POINTER[hsa_ext_image_region_t]), 56]
  hsa_ext_sampler_create: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_sampler_descriptor_t], c.POINTER[hsa_ext_sampler_t]), 64]
  hsa_ext_sampler_destroy: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_sampler_t), 72]
  hsa_ext_image_get_capability_with_layout: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_geometry_t, c.POINTER[hsa_ext_image_format_t], hsa_ext_image_data_layout_t, c.POINTER[uint32_t]), 80]
  hsa_ext_image_data_get_info_with_layout: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, c.POINTER[hsa_ext_image_data_info_t]), 88]
  hsa_ext_image_create_with_layout: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], c.POINTER[None], hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, c.POINTER[hsa_ext_image_t]), 96]
  hsa_ext_sampler_create_v2: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ext_sampler_descriptor_v2_t], c.POINTER[hsa_ext_sampler_t]), 104]
hsa_ext_images_1_pfn_t = struct_hsa_ext_images_1_pfn_s
@dll.bind
def hsa_ven_amd_aqlprofile_version_major() -> uint32_t: ...
@dll.bind
def hsa_ven_amd_aqlprofile_version_minor() -> uint32_t: ...
hsa_ven_amd_aqlprofile_event_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC = hsa_ven_amd_aqlprofile_event_type_t.define('HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC', 0) # type: ignore
HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE = hsa_ven_amd_aqlprofile_event_type_t.define('HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE', 1) # type: ignore

hsa_ven_amd_aqlprofile_block_name_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC', 0) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF', 1) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS', 2) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM', 3) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE', 4) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI', 5) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ', 6) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS', 7) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM', 8) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX', 9) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA', 10) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA', 11) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC', 12) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP', 13) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD', 14) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB', 15) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB', 16) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM', 17) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ', 18) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2 = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2', 19) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR', 20) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC', 21) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2 = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2', 22) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA', 23) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB', 24) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA', 25) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A', 26) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C', 27) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A', 28) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C', 29) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR', 30) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS', 31) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC', 32) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MMEA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MMEA', 33) # type: ignore
HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER', 34) # type: ignore

@c.record
class hsa_ven_amd_aqlprofile_event_t(c.Struct):
  SIZE = 12
  block_name: Annotated[hsa_ven_amd_aqlprofile_block_name_t, 0]
  block_index: Annotated[uint32_t, 4]
  counter_id: Annotated[uint32_t, 8]
@dll.bind
def hsa_ven_amd_aqlprofile_validate_event(agent:hsa_agent_t, event:c.POINTER[hsa_ven_amd_aqlprofile_event_t], result:c.POINTER[Annotated[bool, ctypes.c_bool]]) -> hsa_status_t: ...
hsa_ven_amd_aqlprofile_parameter_name_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET', 0) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK', 1) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK', 2) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK', 3) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2 = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2', 4) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK', 5) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE', 6) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT', 7) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION', 8) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE', 9) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE', 10) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_MASK', 240) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL', 241) # type: ignore
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME', 242) # type: ignore

@c.record
class hsa_ven_amd_aqlprofile_parameter_t(c.Struct):
  SIZE = 8
  parameter_name: Annotated[hsa_ven_amd_aqlprofile_parameter_name_t, 0]
  value: Annotated[uint32_t, 4]
hsa_ven_amd_aqlprofile_att_marker_channel_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_0 = hsa_ven_amd_aqlprofile_att_marker_channel_t.define('HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_0', 0) # type: ignore
HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_1 = hsa_ven_amd_aqlprofile_att_marker_channel_t.define('HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_1', 1) # type: ignore
HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_2 = hsa_ven_amd_aqlprofile_att_marker_channel_t.define('HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_2', 2) # type: ignore
HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_3 = hsa_ven_amd_aqlprofile_att_marker_channel_t.define('HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_3', 3) # type: ignore

@c.record
class hsa_ven_amd_aqlprofile_descriptor_t(c.Struct):
  SIZE = 16
  ptr: Annotated[c.POINTER[None], 0]
  size: Annotated[uint32_t, 8]
@c.record
class hsa_ven_amd_aqlprofile_profile_t(c.Struct):
  SIZE = 80
  agent: Annotated[hsa_agent_t, 0]
  type: Annotated[hsa_ven_amd_aqlprofile_event_type_t, 8]
  events: Annotated[c.POINTER[hsa_ven_amd_aqlprofile_event_t], 16]
  event_count: Annotated[uint32_t, 24]
  parameters: Annotated[c.POINTER[hsa_ven_amd_aqlprofile_parameter_t], 32]
  parameter_count: Annotated[uint32_t, 40]
  output_buffer: Annotated[hsa_ven_amd_aqlprofile_descriptor_t, 48]
  command_buffer: Annotated[hsa_ven_amd_aqlprofile_descriptor_t, 64]
@c.record
class hsa_ext_amd_aql_pm4_packet_t(c.Struct):
  SIZE = 64
  header: Annotated[uint16_t, 0]
  pm4_command: Annotated[c.Array[uint16_t, Literal[27]], 2]
  completion_signal: Annotated[hsa_signal_t, 56]
@dll.bind
def hsa_ven_amd_aqlprofile_start(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], aql_start_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ven_amd_aqlprofile_stop(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], aql_stop_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t]) -> hsa_status_t: ...
@dll.bind
def hsa_ven_amd_aqlprofile_read(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], aql_read_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t]) -> hsa_status_t: ...
try: HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE = Annotated[int, ctypes.c_uint32].in_dll(dll, 'HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE')
except (ValueError,AttributeError): pass
@dll.bind
def hsa_ven_amd_aqlprofile_legacy_get_pm4(aql_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t], data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_ven_amd_aqlprofile_att_marker(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], aql_marker_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t], data:uint32_t, channel:hsa_ven_amd_aqlprofile_att_marker_channel_t) -> hsa_status_t: ...
@c.record
class hsa_ven_amd_aqlprofile_info_data_t(c.Struct):
  SIZE = 32
  sample_id: Annotated[uint32_t, 0]
  pmc_data: Annotated[hsa_ven_amd_aqlprofile_info_data_t_pmc_data, 8]
  trace_data: Annotated[hsa_ven_amd_aqlprofile_descriptor_t, 8]
@c.record
class hsa_ven_amd_aqlprofile_info_data_t_pmc_data(c.Struct):
  SIZE = 24
  event: Annotated[hsa_ven_amd_aqlprofile_event_t, 0]
  result: Annotated[uint64_t, 16]
@c.record
class hsa_ven_amd_aqlprofile_id_query_t(c.Struct):
  SIZE = 16
  name: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 0]
  id: Annotated[uint32_t, 8]
  instance_count: Annotated[uint32_t, 12]
hsa_ven_amd_aqlprofile_info_type_t = CEnum(Annotated[int, ctypes.c_uint32])
HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE', 0) # type: ignore
HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE', 1) # type: ignore
HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA', 2) # type: ignore
HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA', 3) # type: ignore
HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS', 4) # type: ignore
HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID', 5) # type: ignore
HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD', 6) # type: ignore
HSA_VEN_AMD_AQLPROFILE_INFO_DISABLE_CMD = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_DISABLE_CMD', 7) # type: ignore

hsa_ven_amd_aqlprofile_data_callback_t = c.CFUNCTYPE(hsa_status_t, hsa_ven_amd_aqlprofile_info_type_t, c.POINTER[hsa_ven_amd_aqlprofile_info_data_t], c.POINTER[None])
@dll.bind
def hsa_ven_amd_aqlprofile_get_info(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], attribute:hsa_ven_amd_aqlprofile_info_type_t, value:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_ven_amd_aqlprofile_iterate_data(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], callback:hsa_ven_amd_aqlprofile_data_callback_t, data:c.POINTER[None]) -> hsa_status_t: ...
@dll.bind
def hsa_ven_amd_aqlprofile_error_string(str:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> hsa_status_t: ...
hsa_ven_amd_aqlprofile_eventname_callback_t = c.CFUNCTYPE(hsa_status_t, Annotated[int, ctypes.c_int32], c.POINTER[Annotated[bytes, ctypes.c_char]])
@dll.bind
def hsa_ven_amd_aqlprofile_iterate_event_ids(_0:hsa_ven_amd_aqlprofile_eventname_callback_t) -> hsa_status_t: ...
hsa_ven_amd_aqlprofile_coordinate_callback_t = c.CFUNCTYPE(hsa_status_t, Annotated[int, ctypes.c_int32], Annotated[int, ctypes.c_int32], Annotated[int, ctypes.c_int32], Annotated[int, ctypes.c_int32], c.POINTER[Annotated[bytes, ctypes.c_char]], c.POINTER[None])
@dll.bind
def hsa_ven_amd_aqlprofile_iterate_event_coord(agent:hsa_agent_t, event:hsa_ven_amd_aqlprofile_event_t, sample_id:uint32_t, callback:hsa_ven_amd_aqlprofile_coordinate_callback_t, userdata:c.POINTER[None]) -> hsa_status_t: ...
@c.record
class struct_hsa_ven_amd_aqlprofile_1_00_pfn_s(c.Struct):
  SIZE = 104
  hsa_ven_amd_aqlprofile_version_major: Annotated[c.CFUNCTYPE(uint32_t), 0]
  hsa_ven_amd_aqlprofile_version_minor: Annotated[c.CFUNCTYPE(uint32_t), 8]
  hsa_ven_amd_aqlprofile_error_string: Annotated[c.CFUNCTYPE(hsa_status_t, c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]), 16]
  hsa_ven_amd_aqlprofile_validate_event: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, c.POINTER[hsa_ven_amd_aqlprofile_event_t], c.POINTER[Annotated[bool, ctypes.c_bool]]), 24]
  hsa_ven_amd_aqlprofile_start: Annotated[c.CFUNCTYPE(hsa_status_t, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]), 32]
  hsa_ven_amd_aqlprofile_stop: Annotated[c.CFUNCTYPE(hsa_status_t, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]), 40]
  hsa_ven_amd_aqlprofile_read: Annotated[c.CFUNCTYPE(hsa_status_t, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]), 48]
  hsa_ven_amd_aqlprofile_legacy_get_pm4: Annotated[c.CFUNCTYPE(hsa_status_t, c.POINTER[hsa_ext_amd_aql_pm4_packet_t], c.POINTER[None]), 56]
  hsa_ven_amd_aqlprofile_get_info: Annotated[c.CFUNCTYPE(hsa_status_t, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], hsa_ven_amd_aqlprofile_info_type_t, c.POINTER[None]), 64]
  hsa_ven_amd_aqlprofile_iterate_data: Annotated[c.CFUNCTYPE(hsa_status_t, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], hsa_ven_amd_aqlprofile_data_callback_t, c.POINTER[None]), 72]
  hsa_ven_amd_aqlprofile_iterate_event_ids: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_ven_amd_aqlprofile_eventname_callback_t), 80]
  hsa_ven_amd_aqlprofile_iterate_event_coord: Annotated[c.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ven_amd_aqlprofile_event_t, uint32_t, hsa_ven_amd_aqlprofile_coordinate_callback_t, c.POINTER[None]), 88]
  hsa_ven_amd_aqlprofile_att_marker: Annotated[c.CFUNCTYPE(hsa_status_t, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t], uint32_t, hsa_ven_amd_aqlprofile_att_marker_channel_t), 96]
hsa_ven_amd_aqlprofile_1_00_pfn_t = struct_hsa_ven_amd_aqlprofile_1_00_pfn_s
hsa_ven_amd_aqlprofile_pfn_t = struct_hsa_ven_amd_aqlprofile_1_00_pfn_s
c.init_records()
HSA_VERSION_1_0 = 1 # type: ignore
HSA_AMD_INTERFACE_VERSION_MAJOR = 1 # type: ignore
HSA_AMD_INTERFACE_VERSION_MINOR = 14 # type: ignore
AMD_SIGNAL_ALIGN_BYTES = 64 # type: ignore
AMD_QUEUE_ALIGN_BYTES = 64 # type: ignore
MAX_NUM_XCC = 128 # type: ignore
AMD_CONTROL_DIRECTIVES_ALIGN_BYTES = 64 # type: ignore
AMD_ISA_ALIGN_BYTES = 256 # type: ignore
AMD_KERNEL_CODE_ALIGN_BYTES = 64 # type: ignore
HSA_AQLPROFILE_VERSION_MAJOR = 2 # type: ignore
HSA_AQLPROFILE_VERSION_MINOR = 0 # type: ignore
hsa_ven_amd_aqlprofile_VERSION_MAJOR = 1 # type: ignore