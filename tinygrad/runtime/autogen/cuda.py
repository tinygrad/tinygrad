from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import DLL, record, Array, POINTER, CFUNCTYPE, CEnum, _IO, _IOW, _IOR, _IOWR, init_records
dll = DLL('cuda', 'cuda')
cuuint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
cuuint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
CUdeviceptr_v2: TypeAlias = Annotated[int, ctypes.c_uint64]
CUdeviceptr: TypeAlias = Annotated[int, ctypes.c_uint64]
CUdevice_v1: TypeAlias = Annotated[int, ctypes.c_int32]
CUdevice: TypeAlias = Annotated[int, ctypes.c_int32]
class struct_CUctx_st(ctypes.Structure): pass
CUcontext: TypeAlias = POINTER(struct_CUctx_st)
class struct_CUmod_st(ctypes.Structure): pass
CUmodule: TypeAlias = POINTER(struct_CUmod_st)
class struct_CUfunc_st(ctypes.Structure): pass
CUfunction: TypeAlias = POINTER(struct_CUfunc_st)
class struct_CUlib_st(ctypes.Structure): pass
CUlibrary: TypeAlias = POINTER(struct_CUlib_st)
class struct_CUkern_st(ctypes.Structure): pass
CUkernel: TypeAlias = POINTER(struct_CUkern_st)
class struct_CUarray_st(ctypes.Structure): pass
CUarray: TypeAlias = POINTER(struct_CUarray_st)
class struct_CUmipmappedArray_st(ctypes.Structure): pass
CUmipmappedArray: TypeAlias = POINTER(struct_CUmipmappedArray_st)
class struct_CUtexref_st(ctypes.Structure): pass
CUtexref: TypeAlias = POINTER(struct_CUtexref_st)
class struct_CUsurfref_st(ctypes.Structure): pass
CUsurfref: TypeAlias = POINTER(struct_CUsurfref_st)
class struct_CUevent_st(ctypes.Structure): pass
CUevent: TypeAlias = POINTER(struct_CUevent_st)
class struct_CUstream_st(ctypes.Structure): pass
CUstream: TypeAlias = POINTER(struct_CUstream_st)
class struct_CUgraphicsResource_st(ctypes.Structure): pass
CUgraphicsResource: TypeAlias = POINTER(struct_CUgraphicsResource_st)
CUtexObject_v1: TypeAlias = Annotated[int, ctypes.c_uint64]
CUtexObject: TypeAlias = Annotated[int, ctypes.c_uint64]
CUsurfObject_v1: TypeAlias = Annotated[int, ctypes.c_uint64]
CUsurfObject: TypeAlias = Annotated[int, ctypes.c_uint64]
class struct_CUextMemory_st(ctypes.Structure): pass
CUexternalMemory: TypeAlias = POINTER(struct_CUextMemory_st)
class struct_CUextSemaphore_st(ctypes.Structure): pass
CUexternalSemaphore: TypeAlias = POINTER(struct_CUextSemaphore_st)
class struct_CUgraph_st(ctypes.Structure): pass
CUgraph: TypeAlias = POINTER(struct_CUgraph_st)
class struct_CUgraphNode_st(ctypes.Structure): pass
CUgraphNode: TypeAlias = POINTER(struct_CUgraphNode_st)
class struct_CUgraphExec_st(ctypes.Structure): pass
CUgraphExec: TypeAlias = POINTER(struct_CUgraphExec_st)
class struct_CUmemPoolHandle_st(ctypes.Structure): pass
CUmemoryPool: TypeAlias = POINTER(struct_CUmemPoolHandle_st)
class struct_CUuserObject_st(ctypes.Structure): pass
CUuserObject: TypeAlias = POINTER(struct_CUuserObject_st)
@record
class struct_CUuuid_st:
  SIZE = 16
  bytes: Annotated[Array[Annotated[bytes, ctypes.c_char], Literal[16]], 0]
CUuuid: TypeAlias = struct_CUuuid_st
@record
class struct_CUipcEventHandle_st:
  SIZE = 64
  reserved: Annotated[Array[Annotated[bytes, ctypes.c_char], Literal[64]], 0]
CUipcEventHandle_v1: TypeAlias = struct_CUipcEventHandle_st
CUipcEventHandle: TypeAlias = struct_CUipcEventHandle_st
@record
class struct_CUipcMemHandle_st:
  SIZE = 64
  reserved: Annotated[Array[Annotated[bytes, ctypes.c_char], Literal[64]], 0]
CUipcMemHandle_v1: TypeAlias = struct_CUipcMemHandle_st
CUipcMemHandle: TypeAlias = struct_CUipcMemHandle_st
enum_CUipcMem_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = enum_CUipcMem_flags_enum.define('CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS', 1)

CUipcMem_flags: TypeAlias = enum_CUipcMem_flags_enum
enum_CUmemAttach_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_ATTACH_GLOBAL = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_GLOBAL', 1)
CU_MEM_ATTACH_HOST = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_HOST', 2)
CU_MEM_ATTACH_SINGLE = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_SINGLE', 4)

CUmemAttach_flags: TypeAlias = enum_CUmemAttach_flags_enum
enum_CUctx_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_CTX_SCHED_AUTO = enum_CUctx_flags_enum.define('CU_CTX_SCHED_AUTO', 0)
CU_CTX_SCHED_SPIN = enum_CUctx_flags_enum.define('CU_CTX_SCHED_SPIN', 1)
CU_CTX_SCHED_YIELD = enum_CUctx_flags_enum.define('CU_CTX_SCHED_YIELD', 2)
CU_CTX_SCHED_BLOCKING_SYNC = enum_CUctx_flags_enum.define('CU_CTX_SCHED_BLOCKING_SYNC', 4)
CU_CTX_BLOCKING_SYNC = enum_CUctx_flags_enum.define('CU_CTX_BLOCKING_SYNC', 4)
CU_CTX_SCHED_MASK = enum_CUctx_flags_enum.define('CU_CTX_SCHED_MASK', 7)
CU_CTX_MAP_HOST = enum_CUctx_flags_enum.define('CU_CTX_MAP_HOST', 8)
CU_CTX_LMEM_RESIZE_TO_MAX = enum_CUctx_flags_enum.define('CU_CTX_LMEM_RESIZE_TO_MAX', 16)
CU_CTX_FLAGS_MASK = enum_CUctx_flags_enum.define('CU_CTX_FLAGS_MASK', 31)

CUctx_flags: TypeAlias = enum_CUctx_flags_enum
enum_CUevent_sched_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_EVENT_SCHED_AUTO = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_AUTO', 0)
CU_EVENT_SCHED_SPIN = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_SPIN', 1)
CU_EVENT_SCHED_YIELD = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_YIELD', 2)
CU_EVENT_SCHED_BLOCKING_SYNC = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_BLOCKING_SYNC', 4)

CUevent_sched_flags: TypeAlias = enum_CUevent_sched_flags_enum
enum_cl_event_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
NVCL_EVENT_SCHED_AUTO = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_AUTO', 0)
NVCL_EVENT_SCHED_SPIN = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_SPIN', 1)
NVCL_EVENT_SCHED_YIELD = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_YIELD', 2)
NVCL_EVENT_SCHED_BLOCKING_SYNC = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_BLOCKING_SYNC', 4)

cl_event_flags: TypeAlias = enum_cl_event_flags_enum
enum_cl_context_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
NVCL_CTX_SCHED_AUTO = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_AUTO', 0)
NVCL_CTX_SCHED_SPIN = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_SPIN', 1)
NVCL_CTX_SCHED_YIELD = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_YIELD', 2)
NVCL_CTX_SCHED_BLOCKING_SYNC = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_BLOCKING_SYNC', 4)

cl_context_flags: TypeAlias = enum_cl_context_flags_enum
enum_CUstream_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_STREAM_DEFAULT = enum_CUstream_flags_enum.define('CU_STREAM_DEFAULT', 0)
CU_STREAM_NON_BLOCKING = enum_CUstream_flags_enum.define('CU_STREAM_NON_BLOCKING', 1)

CUstream_flags: TypeAlias = enum_CUstream_flags_enum
enum_CUevent_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_EVENT_DEFAULT = enum_CUevent_flags_enum.define('CU_EVENT_DEFAULT', 0)
CU_EVENT_BLOCKING_SYNC = enum_CUevent_flags_enum.define('CU_EVENT_BLOCKING_SYNC', 1)
CU_EVENT_DISABLE_TIMING = enum_CUevent_flags_enum.define('CU_EVENT_DISABLE_TIMING', 2)
CU_EVENT_INTERPROCESS = enum_CUevent_flags_enum.define('CU_EVENT_INTERPROCESS', 4)

CUevent_flags: TypeAlias = enum_CUevent_flags_enum
enum_CUevent_record_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_EVENT_RECORD_DEFAULT = enum_CUevent_record_flags_enum.define('CU_EVENT_RECORD_DEFAULT', 0)
CU_EVENT_RECORD_EXTERNAL = enum_CUevent_record_flags_enum.define('CU_EVENT_RECORD_EXTERNAL', 1)

CUevent_record_flags: TypeAlias = enum_CUevent_record_flags_enum
enum_CUevent_wait_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_EVENT_WAIT_DEFAULT = enum_CUevent_wait_flags_enum.define('CU_EVENT_WAIT_DEFAULT', 0)
CU_EVENT_WAIT_EXTERNAL = enum_CUevent_wait_flags_enum.define('CU_EVENT_WAIT_EXTERNAL', 1)

CUevent_wait_flags: TypeAlias = enum_CUevent_wait_flags_enum
enum_CUstreamWaitValue_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_STREAM_WAIT_VALUE_GEQ = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_GEQ', 0)
CU_STREAM_WAIT_VALUE_EQ = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_EQ', 1)
CU_STREAM_WAIT_VALUE_AND = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_AND', 2)
CU_STREAM_WAIT_VALUE_NOR = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_NOR', 3)
CU_STREAM_WAIT_VALUE_FLUSH = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_FLUSH', 1073741824)

CUstreamWaitValue_flags: TypeAlias = enum_CUstreamWaitValue_flags_enum
enum_CUstreamWriteValue_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_STREAM_WRITE_VALUE_DEFAULT = enum_CUstreamWriteValue_flags_enum.define('CU_STREAM_WRITE_VALUE_DEFAULT', 0)
CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = enum_CUstreamWriteValue_flags_enum.define('CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER', 1)

CUstreamWriteValue_flags: TypeAlias = enum_CUstreamWriteValue_flags_enum
enum_CUstreamBatchMemOpType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_STREAM_MEM_OP_WAIT_VALUE_32 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WAIT_VALUE_32', 1)
CU_STREAM_MEM_OP_WRITE_VALUE_32 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WRITE_VALUE_32', 2)
CU_STREAM_MEM_OP_WAIT_VALUE_64 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WAIT_VALUE_64', 4)
CU_STREAM_MEM_OP_WRITE_VALUE_64 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WRITE_VALUE_64', 5)
CU_STREAM_MEM_OP_BARRIER = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_BARRIER', 6)
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES', 3)

CUstreamBatchMemOpType: TypeAlias = enum_CUstreamBatchMemOpType_enum
enum_CUstreamMemoryBarrier_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_STREAM_MEMORY_BARRIER_TYPE_SYS = enum_CUstreamMemoryBarrier_flags_enum.define('CU_STREAM_MEMORY_BARRIER_TYPE_SYS', 0)
CU_STREAM_MEMORY_BARRIER_TYPE_GPU = enum_CUstreamMemoryBarrier_flags_enum.define('CU_STREAM_MEMORY_BARRIER_TYPE_GPU', 1)

CUstreamMemoryBarrier_flags: TypeAlias = enum_CUstreamMemoryBarrier_flags_enum
@record
class union_CUstreamBatchMemOpParams_union:
  SIZE = 48
  operation: Annotated[CUstreamBatchMemOpType, 0]
  waitValue: Annotated[struct_CUstreamMemOpWaitValueParams_st, 0]
  writeValue: Annotated[struct_CUstreamMemOpWriteValueParams_st, 0]
  flushRemoteWrites: Annotated[struct_CUstreamMemOpFlushRemoteWritesParams_st, 0]
  memoryBarrier: Annotated[struct_CUstreamMemOpMemoryBarrierParams_st, 0]
  pad: Annotated[Array[cuuint64_t, Literal[6]], 0]
@record
class struct_CUstreamMemOpWaitValueParams_st:
  SIZE = 40
  operation: Annotated[CUstreamBatchMemOpType, 0]
  address: Annotated[CUdeviceptr, 8]
  value: Annotated[cuuint32_t, 16]
  value64: Annotated[cuuint64_t, 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  alias: Annotated[CUdeviceptr, 32]
@record
class struct_CUstreamMemOpWriteValueParams_st:
  SIZE = 40
  operation: Annotated[CUstreamBatchMemOpType, 0]
  address: Annotated[CUdeviceptr, 8]
  value: Annotated[cuuint32_t, 16]
  value64: Annotated[cuuint64_t, 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  alias: Annotated[CUdeviceptr, 32]
@record
class struct_CUstreamMemOpFlushRemoteWritesParams_st:
  SIZE = 8
  operation: Annotated[CUstreamBatchMemOpType, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
@record
class struct_CUstreamMemOpMemoryBarrierParams_st:
  SIZE = 8
  operation: Annotated[CUstreamBatchMemOpType, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
CUstreamBatchMemOpParams_v1: TypeAlias = union_CUstreamBatchMemOpParams_union
CUstreamBatchMemOpParams: TypeAlias = union_CUstreamBatchMemOpParams_union
@record
class struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st:
  SIZE = 32
  ctx: Annotated[CUcontext, 0]
  count: Annotated[Annotated[int, ctypes.c_uint32], 8]
  paramArray: Annotated[POINTER(CUstreamBatchMemOpParams), 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
CUDA_BATCH_MEM_OP_NODE_PARAMS: TypeAlias = struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st
enum_CUoccupancy_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_OCCUPANCY_DEFAULT = enum_CUoccupancy_flags_enum.define('CU_OCCUPANCY_DEFAULT', 0)
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = enum_CUoccupancy_flags_enum.define('CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE', 1)

CUoccupancy_flags: TypeAlias = enum_CUoccupancy_flags_enum
enum_CUstreamUpdateCaptureDependencies_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_STREAM_ADD_CAPTURE_DEPENDENCIES = enum_CUstreamUpdateCaptureDependencies_flags_enum.define('CU_STREAM_ADD_CAPTURE_DEPENDENCIES', 0)
CU_STREAM_SET_CAPTURE_DEPENDENCIES = enum_CUstreamUpdateCaptureDependencies_flags_enum.define('CU_STREAM_SET_CAPTURE_DEPENDENCIES', 1)

CUstreamUpdateCaptureDependencies_flags: TypeAlias = enum_CUstreamUpdateCaptureDependencies_flags_enum
enum_CUarray_format_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_AD_FORMAT_UNSIGNED_INT8 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNSIGNED_INT8', 1)
CU_AD_FORMAT_UNSIGNED_INT16 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNSIGNED_INT16', 2)
CU_AD_FORMAT_UNSIGNED_INT32 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNSIGNED_INT32', 3)
CU_AD_FORMAT_SIGNED_INT8 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SIGNED_INT8', 8)
CU_AD_FORMAT_SIGNED_INT16 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SIGNED_INT16', 9)
CU_AD_FORMAT_SIGNED_INT32 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SIGNED_INT32', 10)
CU_AD_FORMAT_HALF = enum_CUarray_format_enum.define('CU_AD_FORMAT_HALF', 16)
CU_AD_FORMAT_FLOAT = enum_CUarray_format_enum.define('CU_AD_FORMAT_FLOAT', 32)
CU_AD_FORMAT_NV12 = enum_CUarray_format_enum.define('CU_AD_FORMAT_NV12', 176)
CU_AD_FORMAT_UNORM_INT8X1 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNORM_INT8X1', 192)
CU_AD_FORMAT_UNORM_INT8X2 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNORM_INT8X2', 193)
CU_AD_FORMAT_UNORM_INT8X4 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNORM_INT8X4', 194)
CU_AD_FORMAT_UNORM_INT16X1 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNORM_INT16X1', 195)
CU_AD_FORMAT_UNORM_INT16X2 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNORM_INT16X2', 196)
CU_AD_FORMAT_UNORM_INT16X4 = enum_CUarray_format_enum.define('CU_AD_FORMAT_UNORM_INT16X4', 197)
CU_AD_FORMAT_SNORM_INT8X1 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SNORM_INT8X1', 198)
CU_AD_FORMAT_SNORM_INT8X2 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SNORM_INT8X2', 199)
CU_AD_FORMAT_SNORM_INT8X4 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SNORM_INT8X4', 200)
CU_AD_FORMAT_SNORM_INT16X1 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SNORM_INT16X1', 201)
CU_AD_FORMAT_SNORM_INT16X2 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SNORM_INT16X2', 202)
CU_AD_FORMAT_SNORM_INT16X4 = enum_CUarray_format_enum.define('CU_AD_FORMAT_SNORM_INT16X4', 203)
CU_AD_FORMAT_BC1_UNORM = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC1_UNORM', 145)
CU_AD_FORMAT_BC1_UNORM_SRGB = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC1_UNORM_SRGB', 146)
CU_AD_FORMAT_BC2_UNORM = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC2_UNORM', 147)
CU_AD_FORMAT_BC2_UNORM_SRGB = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC2_UNORM_SRGB', 148)
CU_AD_FORMAT_BC3_UNORM = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC3_UNORM', 149)
CU_AD_FORMAT_BC3_UNORM_SRGB = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC3_UNORM_SRGB', 150)
CU_AD_FORMAT_BC4_UNORM = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC4_UNORM', 151)
CU_AD_FORMAT_BC4_SNORM = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC4_SNORM', 152)
CU_AD_FORMAT_BC5_UNORM = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC5_UNORM', 153)
CU_AD_FORMAT_BC5_SNORM = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC5_SNORM', 154)
CU_AD_FORMAT_BC6H_UF16 = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC6H_UF16', 155)
CU_AD_FORMAT_BC6H_SF16 = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC6H_SF16', 156)
CU_AD_FORMAT_BC7_UNORM = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC7_UNORM', 157)
CU_AD_FORMAT_BC7_UNORM_SRGB = enum_CUarray_format_enum.define('CU_AD_FORMAT_BC7_UNORM_SRGB', 158)

CUarray_format: TypeAlias = enum_CUarray_format_enum
enum_CUaddress_mode_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_TR_ADDRESS_MODE_WRAP = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_WRAP', 0)
CU_TR_ADDRESS_MODE_CLAMP = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_CLAMP', 1)
CU_TR_ADDRESS_MODE_MIRROR = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_MIRROR', 2)
CU_TR_ADDRESS_MODE_BORDER = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_BORDER', 3)

CUaddress_mode: TypeAlias = enum_CUaddress_mode_enum
enum_CUfilter_mode_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_TR_FILTER_MODE_POINT = enum_CUfilter_mode_enum.define('CU_TR_FILTER_MODE_POINT', 0)
CU_TR_FILTER_MODE_LINEAR = enum_CUfilter_mode_enum.define('CU_TR_FILTER_MODE_LINEAR', 1)

CUfilter_mode: TypeAlias = enum_CUfilter_mode_enum
enum_CUdevice_attribute_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK', 1)
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X', 2)
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y', 3)
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z', 4)
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X', 5)
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y', 6)
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z', 7)
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK', 8)
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK', 8)
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY', 9)
CU_DEVICE_ATTRIBUTE_WARP_SIZE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_WARP_SIZE', 10)
CU_DEVICE_ATTRIBUTE_MAX_PITCH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_PITCH', 11)
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK', 12)
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK', 12)
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CLOCK_RATE', 13)
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT', 14)
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_GPU_OVERLAP', 15)
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT', 16)
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT', 17)
CU_DEVICE_ATTRIBUTE_INTEGRATED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_INTEGRATED', 18)
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY', 19)
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_COMPUTE_MODE', 20)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH', 21)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH', 22)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT', 23)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH', 24)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT', 25)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH', 26)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH', 27)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT', 28)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS', 29)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH', 27)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT', 28)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES', 29)
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT', 30)
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS', 31)
CU_DEVICE_ATTRIBUTE_ECC_ENABLED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_ECC_ENABLED', 32)
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_PCI_BUS_ID', 33)
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID', 34)
CU_DEVICE_ATTRIBUTE_TCC_DRIVER = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_TCC_DRIVER', 35)
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE', 36)
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH', 37)
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE', 38)
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR', 39)
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT', 40)
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING', 41)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH', 42)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS', 43)
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER', 44)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH', 45)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT', 46)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE', 47)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE', 48)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE', 49)
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID', 50)
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT', 51)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH', 52)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH', 53)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS', 54)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH', 55)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH', 56)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT', 57)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH', 58)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT', 59)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH', 60)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH', 61)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS', 62)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH', 63)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT', 64)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS', 65)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH', 66)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH', 67)
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS', 68)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH', 69)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH', 70)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT', 71)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH', 72)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH', 73)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT', 74)
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR', 75)
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR', 76)
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH', 77)
CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED', 78)
CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED', 79)
CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED', 80)
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', 81)
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR', 82)
CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY', 83)
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD', 84)
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID', 85)
CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED', 86)
CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO', 87)
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS', 88)
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS', 89)
CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED', 90)
CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM', 91)
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1', 92)
CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1', 93)
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1', 94)
CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH', 95)
CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH', 96)
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN', 97)
CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES', 98)
CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED', 99)
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES', 100)
CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST', 101)
CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED', 102)
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED', 102)
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED', 103)
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED', 104)
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED', 105)
CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR', 106)
CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED', 107)
CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE', 108)
CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE', 109)
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED', 110)
CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK', 111)
CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED', 112)
CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED', 113)
CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED', 114)
CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED', 115)
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED', 116)
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS', 117)
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING', 118)
CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES', 119)
CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH', 120)
CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED', 121)
CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS', 122)
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR', 123)
CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED', 124)
CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED', 125)
CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT', 126)
CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED', 127)
CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS', 129)
CU_DEVICE_ATTRIBUTE_MAX = enum_CUdevice_attribute_enum.define('CU_DEVICE_ATTRIBUTE_MAX', 130)

CUdevice_attribute: TypeAlias = enum_CUdevice_attribute_enum
@record
class struct_CUdevprop_st:
  SIZE = 56
  maxThreadsPerBlock: Annotated[Annotated[int, ctypes.c_int32], 0]
  maxThreadsDim: Annotated[Array[Annotated[int, ctypes.c_int32], Literal[3]], 4]
  maxGridSize: Annotated[Array[Annotated[int, ctypes.c_int32], Literal[3]], 16]
  sharedMemPerBlock: Annotated[Annotated[int, ctypes.c_int32], 28]
  totalConstantMemory: Annotated[Annotated[int, ctypes.c_int32], 32]
  SIMDWidth: Annotated[Annotated[int, ctypes.c_int32], 36]
  memPitch: Annotated[Annotated[int, ctypes.c_int32], 40]
  regsPerBlock: Annotated[Annotated[int, ctypes.c_int32], 44]
  clockRate: Annotated[Annotated[int, ctypes.c_int32], 48]
  textureAlign: Annotated[Annotated[int, ctypes.c_int32], 52]
CUdevprop_v1: TypeAlias = struct_CUdevprop_st
CUdevprop: TypeAlias = struct_CUdevprop_st
enum_CUpointer_attribute_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_POINTER_ATTRIBUTE_CONTEXT = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_CONTEXT', 1)
CU_POINTER_ATTRIBUTE_MEMORY_TYPE = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_MEMORY_TYPE', 2)
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_DEVICE_POINTER', 3)
CU_POINTER_ATTRIBUTE_HOST_POINTER = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_HOST_POINTER', 4)
CU_POINTER_ATTRIBUTE_P2P_TOKENS = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_P2P_TOKENS', 5)
CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_SYNC_MEMOPS', 6)
CU_POINTER_ATTRIBUTE_BUFFER_ID = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_BUFFER_ID', 7)
CU_POINTER_ATTRIBUTE_IS_MANAGED = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_IS_MANAGED', 8)
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL', 9)
CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE', 10)
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_RANGE_START_ADDR', 11)
CU_POINTER_ATTRIBUTE_RANGE_SIZE = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_RANGE_SIZE', 12)
CU_POINTER_ATTRIBUTE_MAPPED = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_MAPPED', 13)
CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES', 14)
CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE', 15)
CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAGS', 16)
CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE', 17)
CU_POINTER_ATTRIBUTE_MAPPING_SIZE = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_MAPPING_SIZE', 18)
CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR', 19)
CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = enum_CUpointer_attribute_enum.define('CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID', 20)

CUpointer_attribute: TypeAlias = enum_CUpointer_attribute_enum
enum_CUfunction_attribute_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK', 0)
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES', 1)
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES', 2)
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', 3)
CU_FUNC_ATTRIBUTE_NUM_REGS = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_NUM_REGS', 4)
CU_FUNC_ATTRIBUTE_PTX_VERSION = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_PTX_VERSION', 5)
CU_FUNC_ATTRIBUTE_BINARY_VERSION = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_BINARY_VERSION', 6)
CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_CACHE_MODE_CA', 7)
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES', 8)
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT', 9)
CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET', 10)
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH', 11)
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT', 12)
CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH', 13)
CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED', 14)
CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE', 15)
CU_FUNC_ATTRIBUTE_MAX = enum_CUfunction_attribute_enum.define('CU_FUNC_ATTRIBUTE_MAX', 16)

CUfunction_attribute: TypeAlias = enum_CUfunction_attribute_enum
enum_CUfunc_cache_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_FUNC_CACHE_PREFER_NONE = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_NONE', 0)
CU_FUNC_CACHE_PREFER_SHARED = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_SHARED', 1)
CU_FUNC_CACHE_PREFER_L1 = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_L1', 2)
CU_FUNC_CACHE_PREFER_EQUAL = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_EQUAL', 3)

CUfunc_cache: TypeAlias = enum_CUfunc_cache_enum
enum_CUsharedconfig_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE', 0)
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE', 1)
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE', 2)

CUsharedconfig: TypeAlias = enum_CUsharedconfig_enum
enum_CUshared_carveout_enum = CEnum(Annotated[int, ctypes.c_int32])
CU_SHAREDMEM_CARVEOUT_DEFAULT = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_DEFAULT', -1)
CU_SHAREDMEM_CARVEOUT_MAX_SHARED = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_MAX_SHARED', 100)
CU_SHAREDMEM_CARVEOUT_MAX_L1 = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_MAX_L1', 0)

CUshared_carveout: TypeAlias = enum_CUshared_carveout_enum
enum_CUmemorytype_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEMORYTYPE_HOST = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_HOST', 1)
CU_MEMORYTYPE_DEVICE = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_DEVICE', 2)
CU_MEMORYTYPE_ARRAY = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_ARRAY', 3)
CU_MEMORYTYPE_UNIFIED = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_UNIFIED', 4)

CUmemorytype: TypeAlias = enum_CUmemorytype_enum
enum_CUcomputemode_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_COMPUTEMODE_DEFAULT = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_DEFAULT', 0)
CU_COMPUTEMODE_PROHIBITED = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_PROHIBITED', 2)
CU_COMPUTEMODE_EXCLUSIVE_PROCESS = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_EXCLUSIVE_PROCESS', 3)

CUcomputemode: TypeAlias = enum_CUcomputemode_enum
enum_CUmem_advise_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_ADVISE_SET_READ_MOSTLY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_READ_MOSTLY', 1)
CU_MEM_ADVISE_UNSET_READ_MOSTLY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_READ_MOSTLY', 2)
CU_MEM_ADVISE_SET_PREFERRED_LOCATION = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_PREFERRED_LOCATION', 3)
CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION', 4)
CU_MEM_ADVISE_SET_ACCESSED_BY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_ACCESSED_BY', 5)
CU_MEM_ADVISE_UNSET_ACCESSED_BY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_ACCESSED_BY', 6)

CUmem_advise: TypeAlias = enum_CUmem_advise_enum
enum_CUmem_range_attribute_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY', 1)
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION', 2)
CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY', 3)
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION', 4)

CUmem_range_attribute: TypeAlias = enum_CUmem_range_attribute_enum
enum_CUjit_option_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_JIT_MAX_REGISTERS = enum_CUjit_option_enum.define('CU_JIT_MAX_REGISTERS', 0)
CU_JIT_THREADS_PER_BLOCK = enum_CUjit_option_enum.define('CU_JIT_THREADS_PER_BLOCK', 1)
CU_JIT_WALL_TIME = enum_CUjit_option_enum.define('CU_JIT_WALL_TIME', 2)
CU_JIT_INFO_LOG_BUFFER = enum_CUjit_option_enum.define('CU_JIT_INFO_LOG_BUFFER', 3)
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = enum_CUjit_option_enum.define('CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES', 4)
CU_JIT_ERROR_LOG_BUFFER = enum_CUjit_option_enum.define('CU_JIT_ERROR_LOG_BUFFER', 5)
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = enum_CUjit_option_enum.define('CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES', 6)
CU_JIT_OPTIMIZATION_LEVEL = enum_CUjit_option_enum.define('CU_JIT_OPTIMIZATION_LEVEL', 7)
CU_JIT_TARGET_FROM_CUCONTEXT = enum_CUjit_option_enum.define('CU_JIT_TARGET_FROM_CUCONTEXT', 8)
CU_JIT_TARGET = enum_CUjit_option_enum.define('CU_JIT_TARGET', 9)
CU_JIT_FALLBACK_STRATEGY = enum_CUjit_option_enum.define('CU_JIT_FALLBACK_STRATEGY', 10)
CU_JIT_GENERATE_DEBUG_INFO = enum_CUjit_option_enum.define('CU_JIT_GENERATE_DEBUG_INFO', 11)
CU_JIT_LOG_VERBOSE = enum_CUjit_option_enum.define('CU_JIT_LOG_VERBOSE', 12)
CU_JIT_GENERATE_LINE_INFO = enum_CUjit_option_enum.define('CU_JIT_GENERATE_LINE_INFO', 13)
CU_JIT_CACHE_MODE = enum_CUjit_option_enum.define('CU_JIT_CACHE_MODE', 14)
CU_JIT_NEW_SM3X_OPT = enum_CUjit_option_enum.define('CU_JIT_NEW_SM3X_OPT', 15)
CU_JIT_FAST_COMPILE = enum_CUjit_option_enum.define('CU_JIT_FAST_COMPILE', 16)
CU_JIT_GLOBAL_SYMBOL_NAMES = enum_CUjit_option_enum.define('CU_JIT_GLOBAL_SYMBOL_NAMES', 17)
CU_JIT_GLOBAL_SYMBOL_ADDRESSES = enum_CUjit_option_enum.define('CU_JIT_GLOBAL_SYMBOL_ADDRESSES', 18)
CU_JIT_GLOBAL_SYMBOL_COUNT = enum_CUjit_option_enum.define('CU_JIT_GLOBAL_SYMBOL_COUNT', 19)
CU_JIT_LTO = enum_CUjit_option_enum.define('CU_JIT_LTO', 20)
CU_JIT_FTZ = enum_CUjit_option_enum.define('CU_JIT_FTZ', 21)
CU_JIT_PREC_DIV = enum_CUjit_option_enum.define('CU_JIT_PREC_DIV', 22)
CU_JIT_PREC_SQRT = enum_CUjit_option_enum.define('CU_JIT_PREC_SQRT', 23)
CU_JIT_FMA = enum_CUjit_option_enum.define('CU_JIT_FMA', 24)
CU_JIT_REFERENCED_KERNEL_NAMES = enum_CUjit_option_enum.define('CU_JIT_REFERENCED_KERNEL_NAMES', 25)
CU_JIT_REFERENCED_KERNEL_COUNT = enum_CUjit_option_enum.define('CU_JIT_REFERENCED_KERNEL_COUNT', 26)
CU_JIT_REFERENCED_VARIABLE_NAMES = enum_CUjit_option_enum.define('CU_JIT_REFERENCED_VARIABLE_NAMES', 27)
CU_JIT_REFERENCED_VARIABLE_COUNT = enum_CUjit_option_enum.define('CU_JIT_REFERENCED_VARIABLE_COUNT', 28)
CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = enum_CUjit_option_enum.define('CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES', 29)
CU_JIT_POSITION_INDEPENDENT_CODE = enum_CUjit_option_enum.define('CU_JIT_POSITION_INDEPENDENT_CODE', 30)
CU_JIT_NUM_OPTIONS = enum_CUjit_option_enum.define('CU_JIT_NUM_OPTIONS', 31)

CUjit_option: TypeAlias = enum_CUjit_option_enum
enum_CUjit_target_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_TARGET_COMPUTE_30 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_30', 30)
CU_TARGET_COMPUTE_32 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_32', 32)
CU_TARGET_COMPUTE_35 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_35', 35)
CU_TARGET_COMPUTE_37 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_37', 37)
CU_TARGET_COMPUTE_50 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_50', 50)
CU_TARGET_COMPUTE_52 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_52', 52)
CU_TARGET_COMPUTE_53 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_53', 53)
CU_TARGET_COMPUTE_60 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_60', 60)
CU_TARGET_COMPUTE_61 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_61', 61)
CU_TARGET_COMPUTE_62 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_62', 62)
CU_TARGET_COMPUTE_70 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_70', 70)
CU_TARGET_COMPUTE_72 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_72', 72)
CU_TARGET_COMPUTE_75 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_75', 75)
CU_TARGET_COMPUTE_80 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_80', 80)
CU_TARGET_COMPUTE_86 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_86', 86)
CU_TARGET_COMPUTE_87 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_87', 87)
CU_TARGET_COMPUTE_89 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_89', 89)
CU_TARGET_COMPUTE_90 = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_90', 90)
CU_TARGET_COMPUTE_90A = enum_CUjit_target_enum.define('CU_TARGET_COMPUTE_90A', 65626)

CUjit_target: TypeAlias = enum_CUjit_target_enum
enum_CUjit_fallback_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_PREFER_PTX = enum_CUjit_fallback_enum.define('CU_PREFER_PTX', 0)
CU_PREFER_BINARY = enum_CUjit_fallback_enum.define('CU_PREFER_BINARY', 1)

CUjit_fallback: TypeAlias = enum_CUjit_fallback_enum
enum_CUjit_cacheMode_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_JIT_CACHE_OPTION_NONE = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_NONE', 0)
CU_JIT_CACHE_OPTION_CG = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_CG', 1)
CU_JIT_CACHE_OPTION_CA = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_CA', 2)

CUjit_cacheMode: TypeAlias = enum_CUjit_cacheMode_enum
enum_CUjitInputType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_JIT_INPUT_CUBIN = enum_CUjitInputType_enum.define('CU_JIT_INPUT_CUBIN', 0)
CU_JIT_INPUT_PTX = enum_CUjitInputType_enum.define('CU_JIT_INPUT_PTX', 1)
CU_JIT_INPUT_FATBINARY = enum_CUjitInputType_enum.define('CU_JIT_INPUT_FATBINARY', 2)
CU_JIT_INPUT_OBJECT = enum_CUjitInputType_enum.define('CU_JIT_INPUT_OBJECT', 3)
CU_JIT_INPUT_LIBRARY = enum_CUjitInputType_enum.define('CU_JIT_INPUT_LIBRARY', 4)
CU_JIT_INPUT_NVVM = enum_CUjitInputType_enum.define('CU_JIT_INPUT_NVVM', 5)
CU_JIT_NUM_INPUT_TYPES = enum_CUjitInputType_enum.define('CU_JIT_NUM_INPUT_TYPES', 6)

CUjitInputType: TypeAlias = enum_CUjitInputType_enum
class struct_CUlinkState_st(ctypes.Structure): pass
CUlinkState: TypeAlias = POINTER(struct_CUlinkState_st)
enum_CUgraphicsRegisterFlags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GRAPHICS_REGISTER_FLAGS_NONE = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_NONE', 0)
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY', 1)
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD', 2)
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST', 4)
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER', 8)

CUgraphicsRegisterFlags: TypeAlias = enum_CUgraphicsRegisterFlags_enum
enum_CUgraphicsMapResourceFlags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE', 0)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY', 1)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD', 2)

CUgraphicsMapResourceFlags: TypeAlias = enum_CUgraphicsMapResourceFlags_enum
enum_CUarray_cubemap_face_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_CUBEMAP_FACE_POSITIVE_X = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_X', 0)
CU_CUBEMAP_FACE_NEGATIVE_X = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_X', 1)
CU_CUBEMAP_FACE_POSITIVE_Y = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_Y', 2)
CU_CUBEMAP_FACE_NEGATIVE_Y = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_Y', 3)
CU_CUBEMAP_FACE_POSITIVE_Z = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_Z', 4)
CU_CUBEMAP_FACE_NEGATIVE_Z = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_Z', 5)

CUarray_cubemap_face: TypeAlias = enum_CUarray_cubemap_face_enum
enum_CUlimit_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_LIMIT_STACK_SIZE = enum_CUlimit_enum.define('CU_LIMIT_STACK_SIZE', 0)
CU_LIMIT_PRINTF_FIFO_SIZE = enum_CUlimit_enum.define('CU_LIMIT_PRINTF_FIFO_SIZE', 1)
CU_LIMIT_MALLOC_HEAP_SIZE = enum_CUlimit_enum.define('CU_LIMIT_MALLOC_HEAP_SIZE', 2)
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = enum_CUlimit_enum.define('CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH', 3)
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = enum_CUlimit_enum.define('CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT', 4)
CU_LIMIT_MAX_L2_FETCH_GRANULARITY = enum_CUlimit_enum.define('CU_LIMIT_MAX_L2_FETCH_GRANULARITY', 5)
CU_LIMIT_PERSISTING_L2_CACHE_SIZE = enum_CUlimit_enum.define('CU_LIMIT_PERSISTING_L2_CACHE_SIZE', 6)
CU_LIMIT_MAX = enum_CUlimit_enum.define('CU_LIMIT_MAX', 7)

CUlimit: TypeAlias = enum_CUlimit_enum
enum_CUresourcetype_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_RESOURCE_TYPE_ARRAY = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_ARRAY', 0)
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_MIPMAPPED_ARRAY', 1)
CU_RESOURCE_TYPE_LINEAR = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_LINEAR', 2)
CU_RESOURCE_TYPE_PITCH2D = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_PITCH2D', 3)

CUresourcetype: TypeAlias = enum_CUresourcetype_enum
CUhostFn: TypeAlias = CFUNCTYPE(None, POINTER(None))
enum_CUaccessProperty_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_ACCESS_PROPERTY_NORMAL = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_NORMAL', 0)
CU_ACCESS_PROPERTY_STREAMING = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_STREAMING', 1)
CU_ACCESS_PROPERTY_PERSISTING = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_PERSISTING', 2)

CUaccessProperty: TypeAlias = enum_CUaccessProperty_enum
@record
class struct_CUaccessPolicyWindow_st:
  SIZE = 32
  base_ptr: Annotated[POINTER(None), 0]
  num_bytes: Annotated[size_t, 8]
  hitRatio: Annotated[Annotated[float, ctypes.c_float], 16]
  hitProp: Annotated[CUaccessProperty, 20]
  missProp: Annotated[CUaccessProperty, 24]
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
CUaccessPolicyWindow_v1: TypeAlias = struct_CUaccessPolicyWindow_st
CUaccessPolicyWindow: TypeAlias = struct_CUaccessPolicyWindow_st
@record
class struct_CUDA_KERNEL_NODE_PARAMS_st:
  SIZE = 56
  func: Annotated[CUfunction, 0]
  gridDimX: Annotated[Annotated[int, ctypes.c_uint32], 8]
  gridDimY: Annotated[Annotated[int, ctypes.c_uint32], 12]
  gridDimZ: Annotated[Annotated[int, ctypes.c_uint32], 16]
  blockDimX: Annotated[Annotated[int, ctypes.c_uint32], 20]
  blockDimY: Annotated[Annotated[int, ctypes.c_uint32], 24]
  blockDimZ: Annotated[Annotated[int, ctypes.c_uint32], 28]
  sharedMemBytes: Annotated[Annotated[int, ctypes.c_uint32], 32]
  kernelParams: Annotated[POINTER(POINTER(None)), 40]
  extra: Annotated[POINTER(POINTER(None)), 48]
CUDA_KERNEL_NODE_PARAMS_v1: TypeAlias = struct_CUDA_KERNEL_NODE_PARAMS_st
@record
class struct_CUDA_KERNEL_NODE_PARAMS_v2_st:
  SIZE = 72
  func: Annotated[CUfunction, 0]
  gridDimX: Annotated[Annotated[int, ctypes.c_uint32], 8]
  gridDimY: Annotated[Annotated[int, ctypes.c_uint32], 12]
  gridDimZ: Annotated[Annotated[int, ctypes.c_uint32], 16]
  blockDimX: Annotated[Annotated[int, ctypes.c_uint32], 20]
  blockDimY: Annotated[Annotated[int, ctypes.c_uint32], 24]
  blockDimZ: Annotated[Annotated[int, ctypes.c_uint32], 28]
  sharedMemBytes: Annotated[Annotated[int, ctypes.c_uint32], 32]
  kernelParams: Annotated[POINTER(POINTER(None)), 40]
  extra: Annotated[POINTER(POINTER(None)), 48]
  kern: Annotated[CUkernel, 56]
  ctx: Annotated[CUcontext, 64]
CUDA_KERNEL_NODE_PARAMS_v2: TypeAlias = struct_CUDA_KERNEL_NODE_PARAMS_v2_st
CUDA_KERNEL_NODE_PARAMS: TypeAlias = struct_CUDA_KERNEL_NODE_PARAMS_v2_st
@record
class struct_CUDA_MEMSET_NODE_PARAMS_st:
  SIZE = 40
  dst: Annotated[CUdeviceptr, 0]
  pitch: Annotated[size_t, 8]
  value: Annotated[Annotated[int, ctypes.c_uint32], 16]
  elementSize: Annotated[Annotated[int, ctypes.c_uint32], 20]
  width: Annotated[size_t, 24]
  height: Annotated[size_t, 32]
CUDA_MEMSET_NODE_PARAMS_v1: TypeAlias = struct_CUDA_MEMSET_NODE_PARAMS_st
CUDA_MEMSET_NODE_PARAMS: TypeAlias = struct_CUDA_MEMSET_NODE_PARAMS_st
@record
class struct_CUDA_HOST_NODE_PARAMS_st:
  SIZE = 16
  fn: Annotated[CUhostFn, 0]
  userData: Annotated[POINTER(None), 8]
CUDA_HOST_NODE_PARAMS_v1: TypeAlias = struct_CUDA_HOST_NODE_PARAMS_st
CUDA_HOST_NODE_PARAMS: TypeAlias = struct_CUDA_HOST_NODE_PARAMS_st
enum_CUgraphNodeType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GRAPH_NODE_TYPE_KERNEL = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_KERNEL', 0)
CU_GRAPH_NODE_TYPE_MEMCPY = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_MEMCPY', 1)
CU_GRAPH_NODE_TYPE_MEMSET = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_MEMSET', 2)
CU_GRAPH_NODE_TYPE_HOST = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_HOST', 3)
CU_GRAPH_NODE_TYPE_GRAPH = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_GRAPH', 4)
CU_GRAPH_NODE_TYPE_EMPTY = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_EMPTY', 5)
CU_GRAPH_NODE_TYPE_WAIT_EVENT = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_WAIT_EVENT', 6)
CU_GRAPH_NODE_TYPE_EVENT_RECORD = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_EVENT_RECORD', 7)
CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL', 8)
CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT', 9)
CU_GRAPH_NODE_TYPE_MEM_ALLOC = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_MEM_ALLOC', 10)
CU_GRAPH_NODE_TYPE_MEM_FREE = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_MEM_FREE', 11)
CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = enum_CUgraphNodeType_enum.define('CU_GRAPH_NODE_TYPE_BATCH_MEM_OP', 12)

CUgraphNodeType: TypeAlias = enum_CUgraphNodeType_enum
enum_CUgraphInstantiateResult_enum = CEnum(Annotated[int, ctypes.c_uint32])
CUDA_GRAPH_INSTANTIATE_SUCCESS = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_SUCCESS', 0)
CUDA_GRAPH_INSTANTIATE_ERROR = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_ERROR', 1)
CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE', 2)
CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED', 3)
CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED', 4)

CUgraphInstantiateResult: TypeAlias = enum_CUgraphInstantiateResult_enum
@record
class struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st:
  SIZE = 32
  flags: Annotated[cuuint64_t, 0]
  hUploadStream: Annotated[CUstream, 8]
  hErrNode_out: Annotated[CUgraphNode, 16]
  result_out: Annotated[CUgraphInstantiateResult, 24]
CUDA_GRAPH_INSTANTIATE_PARAMS: TypeAlias = struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st
enum_CUsynchronizationPolicy_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_SYNC_POLICY_AUTO = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_AUTO', 1)
CU_SYNC_POLICY_SPIN = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_SPIN', 2)
CU_SYNC_POLICY_YIELD = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_YIELD', 3)
CU_SYNC_POLICY_BLOCKING_SYNC = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_BLOCKING_SYNC', 4)

CUsynchronizationPolicy: TypeAlias = enum_CUsynchronizationPolicy_enum
enum_CUclusterSchedulingPolicy_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_DEFAULT', 0)
CU_CLUSTER_SCHEDULING_POLICY_SPREAD = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_SPREAD', 1)
CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING', 2)

CUclusterSchedulingPolicy: TypeAlias = enum_CUclusterSchedulingPolicy_enum
enum_CUlaunchMemSyncDomain_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = enum_CUlaunchMemSyncDomain_enum.define('CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT', 0)
CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = enum_CUlaunchMemSyncDomain_enum.define('CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE', 1)

CUlaunchMemSyncDomain: TypeAlias = enum_CUlaunchMemSyncDomain_enum
@record
class struct_CUlaunchMemSyncDomainMap_st:
  SIZE = 2
  default_: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  remote: Annotated[Annotated[int, ctypes.c_ubyte], 1]
CUlaunchMemSyncDomainMap: TypeAlias = struct_CUlaunchMemSyncDomainMap_st
enum_CUlaunchAttributeID_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_LAUNCH_ATTRIBUTE_IGNORE = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_IGNORE', 0)
CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW', 1)
CU_LAUNCH_ATTRIBUTE_COOPERATIVE = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_COOPERATIVE', 2)
CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY', 3)
CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION', 4)
CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE', 5)
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION', 6)
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT', 7)
CU_LAUNCH_ATTRIBUTE_PRIORITY = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_PRIORITY', 8)
CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP', 9)
CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = enum_CUlaunchAttributeID_enum.define('CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN', 10)

CUlaunchAttributeID: TypeAlias = enum_CUlaunchAttributeID_enum
@record
class union_CUlaunchAttributeValue_union:
  SIZE = 64
  pad: Annotated[Array[Annotated[bytes, ctypes.c_char], Literal[64]], 0]
  accessPolicyWindow: Annotated[CUaccessPolicyWindow, 0]
  cooperative: Annotated[Annotated[int, ctypes.c_int32], 0]
  syncPolicy: Annotated[CUsynchronizationPolicy, 0]
  clusterDim: Annotated[_anonstruct0, 0]
  clusterSchedulingPolicyPreference: Annotated[CUclusterSchedulingPolicy, 0]
  programmaticStreamSerializationAllowed: Annotated[Annotated[int, ctypes.c_int32], 0]
  programmaticEvent: Annotated[_anonstruct1, 0]
  priority: Annotated[Annotated[int, ctypes.c_int32], 0]
  memSyncDomainMap: Annotated[CUlaunchMemSyncDomainMap, 0]
  memSyncDomain: Annotated[CUlaunchMemSyncDomain, 0]
@record
class _anonstruct0:
  SIZE = 12
  x: Annotated[Annotated[int, ctypes.c_uint32], 0]
  y: Annotated[Annotated[int, ctypes.c_uint32], 4]
  z: Annotated[Annotated[int, ctypes.c_uint32], 8]
@record
class _anonstruct1:
  SIZE = 16
  event: Annotated[CUevent, 0]
  flags: Annotated[Annotated[int, ctypes.c_int32], 8]
  triggerAtBlockStart: Annotated[Annotated[int, ctypes.c_int32], 12]
CUlaunchAttributeValue: TypeAlias = union_CUlaunchAttributeValue_union
@record
class struct_CUlaunchAttribute_st:
  SIZE = 72
  id: Annotated[CUlaunchAttributeID, 0]
  pad: Annotated[Array[Annotated[bytes, ctypes.c_char], Literal[4]], 4]
  value: Annotated[CUlaunchAttributeValue, 8]
CUlaunchAttribute: TypeAlias = struct_CUlaunchAttribute_st
@record
class struct_CUlaunchConfig_st:
  SIZE = 56
  gridDimX: Annotated[Annotated[int, ctypes.c_uint32], 0]
  gridDimY: Annotated[Annotated[int, ctypes.c_uint32], 4]
  gridDimZ: Annotated[Annotated[int, ctypes.c_uint32], 8]
  blockDimX: Annotated[Annotated[int, ctypes.c_uint32], 12]
  blockDimY: Annotated[Annotated[int, ctypes.c_uint32], 16]
  blockDimZ: Annotated[Annotated[int, ctypes.c_uint32], 20]
  sharedMemBytes: Annotated[Annotated[int, ctypes.c_uint32], 24]
  hStream: Annotated[CUstream, 32]
  attrs: Annotated[POINTER(CUlaunchAttribute), 40]
  numAttrs: Annotated[Annotated[int, ctypes.c_uint32], 48]
CUlaunchConfig: TypeAlias = struct_CUlaunchConfig_st
CUkernelNodeAttrID: TypeAlias = enum_CUlaunchAttributeID_enum
CUkernelNodeAttrValue_v1: TypeAlias = union_CUlaunchAttributeValue_union
CUkernelNodeAttrValue: TypeAlias = union_CUlaunchAttributeValue_union
enum_CUstreamCaptureStatus_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_STREAM_CAPTURE_STATUS_NONE = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_NONE', 0)
CU_STREAM_CAPTURE_STATUS_ACTIVE = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_ACTIVE', 1)
CU_STREAM_CAPTURE_STATUS_INVALIDATED = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_INVALIDATED', 2)

CUstreamCaptureStatus: TypeAlias = enum_CUstreamCaptureStatus_enum
enum_CUstreamCaptureMode_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_STREAM_CAPTURE_MODE_GLOBAL = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_GLOBAL', 0)
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_THREAD_LOCAL', 1)
CU_STREAM_CAPTURE_MODE_RELAXED = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_RELAXED', 2)

CUstreamCaptureMode: TypeAlias = enum_CUstreamCaptureMode_enum
CUstreamAttrID: TypeAlias = enum_CUlaunchAttributeID_enum
CUstreamAttrValue_v1: TypeAlias = union_CUlaunchAttributeValue_union
CUstreamAttrValue: TypeAlias = union_CUlaunchAttributeValue_union
enum_CUdriverProcAddress_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GET_PROC_ADDRESS_DEFAULT = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_DEFAULT', 0)
CU_GET_PROC_ADDRESS_LEGACY_STREAM = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_LEGACY_STREAM', 1)
CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM', 2)

CUdriverProcAddress_flags: TypeAlias = enum_CUdriverProcAddress_flags_enum
enum_CUdriverProcAddressQueryResult_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GET_PROC_ADDRESS_SUCCESS = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_SUCCESS', 0)
CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND', 1)
CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT', 2)

CUdriverProcAddressQueryResult: TypeAlias = enum_CUdriverProcAddressQueryResult_enum
enum_CUexecAffinityType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_EXEC_AFFINITY_TYPE_SM_COUNT = enum_CUexecAffinityType_enum.define('CU_EXEC_AFFINITY_TYPE_SM_COUNT', 0)
CU_EXEC_AFFINITY_TYPE_MAX = enum_CUexecAffinityType_enum.define('CU_EXEC_AFFINITY_TYPE_MAX', 1)

CUexecAffinityType: TypeAlias = enum_CUexecAffinityType_enum
@record
class struct_CUexecAffinitySmCount_st:
  SIZE = 4
  val: Annotated[Annotated[int, ctypes.c_uint32], 0]
CUexecAffinitySmCount_v1: TypeAlias = struct_CUexecAffinitySmCount_st
CUexecAffinitySmCount: TypeAlias = struct_CUexecAffinitySmCount_st
@record
class struct_CUexecAffinityParam_st:
  SIZE = 8
  type: Annotated[CUexecAffinityType, 0]
  param: Annotated[_anonunion2, 4]
@record
class _anonunion2:
  SIZE = 4
  smCount: Annotated[CUexecAffinitySmCount, 0]
CUexecAffinityParam_v1: TypeAlias = struct_CUexecAffinityParam_st
CUexecAffinityParam: TypeAlias = struct_CUexecAffinityParam_st
enum_CUlibraryOption_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = enum_CUlibraryOption_enum.define('CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE', 0)
CU_LIBRARY_BINARY_IS_PRESERVED = enum_CUlibraryOption_enum.define('CU_LIBRARY_BINARY_IS_PRESERVED', 1)
CU_LIBRARY_NUM_OPTIONS = enum_CUlibraryOption_enum.define('CU_LIBRARY_NUM_OPTIONS', 2)

CUlibraryOption: TypeAlias = enum_CUlibraryOption_enum
@record
class struct_CUlibraryHostUniversalFunctionAndDataTable_st:
  SIZE = 32
  functionTable: Annotated[POINTER(None), 0]
  functionWindowSize: Annotated[size_t, 8]
  dataTable: Annotated[POINTER(None), 16]
  dataWindowSize: Annotated[size_t, 24]
CUlibraryHostUniversalFunctionAndDataTable: TypeAlias = struct_CUlibraryHostUniversalFunctionAndDataTable_st
enum_cudaError_enum = CEnum(Annotated[int, ctypes.c_uint32])
CUDA_SUCCESS = enum_cudaError_enum.define('CUDA_SUCCESS', 0)
CUDA_ERROR_INVALID_VALUE = enum_cudaError_enum.define('CUDA_ERROR_INVALID_VALUE', 1)
CUDA_ERROR_OUT_OF_MEMORY = enum_cudaError_enum.define('CUDA_ERROR_OUT_OF_MEMORY', 2)
CUDA_ERROR_NOT_INITIALIZED = enum_cudaError_enum.define('CUDA_ERROR_NOT_INITIALIZED', 3)
CUDA_ERROR_DEINITIALIZED = enum_cudaError_enum.define('CUDA_ERROR_DEINITIALIZED', 4)
CUDA_ERROR_PROFILER_DISABLED = enum_cudaError_enum.define('CUDA_ERROR_PROFILER_DISABLED', 5)
CUDA_ERROR_PROFILER_NOT_INITIALIZED = enum_cudaError_enum.define('CUDA_ERROR_PROFILER_NOT_INITIALIZED', 6)
CUDA_ERROR_PROFILER_ALREADY_STARTED = enum_cudaError_enum.define('CUDA_ERROR_PROFILER_ALREADY_STARTED', 7)
CUDA_ERROR_PROFILER_ALREADY_STOPPED = enum_cudaError_enum.define('CUDA_ERROR_PROFILER_ALREADY_STOPPED', 8)
CUDA_ERROR_STUB_LIBRARY = enum_cudaError_enum.define('CUDA_ERROR_STUB_LIBRARY', 34)
CUDA_ERROR_DEVICE_UNAVAILABLE = enum_cudaError_enum.define('CUDA_ERROR_DEVICE_UNAVAILABLE', 46)
CUDA_ERROR_NO_DEVICE = enum_cudaError_enum.define('CUDA_ERROR_NO_DEVICE', 100)
CUDA_ERROR_INVALID_DEVICE = enum_cudaError_enum.define('CUDA_ERROR_INVALID_DEVICE', 101)
CUDA_ERROR_DEVICE_NOT_LICENSED = enum_cudaError_enum.define('CUDA_ERROR_DEVICE_NOT_LICENSED', 102)
CUDA_ERROR_INVALID_IMAGE = enum_cudaError_enum.define('CUDA_ERROR_INVALID_IMAGE', 200)
CUDA_ERROR_INVALID_CONTEXT = enum_cudaError_enum.define('CUDA_ERROR_INVALID_CONTEXT', 201)
CUDA_ERROR_CONTEXT_ALREADY_CURRENT = enum_cudaError_enum.define('CUDA_ERROR_CONTEXT_ALREADY_CURRENT', 202)
CUDA_ERROR_MAP_FAILED = enum_cudaError_enum.define('CUDA_ERROR_MAP_FAILED', 205)
CUDA_ERROR_UNMAP_FAILED = enum_cudaError_enum.define('CUDA_ERROR_UNMAP_FAILED', 206)
CUDA_ERROR_ARRAY_IS_MAPPED = enum_cudaError_enum.define('CUDA_ERROR_ARRAY_IS_MAPPED', 207)
CUDA_ERROR_ALREADY_MAPPED = enum_cudaError_enum.define('CUDA_ERROR_ALREADY_MAPPED', 208)
CUDA_ERROR_NO_BINARY_FOR_GPU = enum_cudaError_enum.define('CUDA_ERROR_NO_BINARY_FOR_GPU', 209)
CUDA_ERROR_ALREADY_ACQUIRED = enum_cudaError_enum.define('CUDA_ERROR_ALREADY_ACQUIRED', 210)
CUDA_ERROR_NOT_MAPPED = enum_cudaError_enum.define('CUDA_ERROR_NOT_MAPPED', 211)
CUDA_ERROR_NOT_MAPPED_AS_ARRAY = enum_cudaError_enum.define('CUDA_ERROR_NOT_MAPPED_AS_ARRAY', 212)
CUDA_ERROR_NOT_MAPPED_AS_POINTER = enum_cudaError_enum.define('CUDA_ERROR_NOT_MAPPED_AS_POINTER', 213)
CUDA_ERROR_ECC_UNCORRECTABLE = enum_cudaError_enum.define('CUDA_ERROR_ECC_UNCORRECTABLE', 214)
CUDA_ERROR_UNSUPPORTED_LIMIT = enum_cudaError_enum.define('CUDA_ERROR_UNSUPPORTED_LIMIT', 215)
CUDA_ERROR_CONTEXT_ALREADY_IN_USE = enum_cudaError_enum.define('CUDA_ERROR_CONTEXT_ALREADY_IN_USE', 216)
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = enum_cudaError_enum.define('CUDA_ERROR_PEER_ACCESS_UNSUPPORTED', 217)
CUDA_ERROR_INVALID_PTX = enum_cudaError_enum.define('CUDA_ERROR_INVALID_PTX', 218)
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = enum_cudaError_enum.define('CUDA_ERROR_INVALID_GRAPHICS_CONTEXT', 219)
CUDA_ERROR_NVLINK_UNCORRECTABLE = enum_cudaError_enum.define('CUDA_ERROR_NVLINK_UNCORRECTABLE', 220)
CUDA_ERROR_JIT_COMPILER_NOT_FOUND = enum_cudaError_enum.define('CUDA_ERROR_JIT_COMPILER_NOT_FOUND', 221)
CUDA_ERROR_UNSUPPORTED_PTX_VERSION = enum_cudaError_enum.define('CUDA_ERROR_UNSUPPORTED_PTX_VERSION', 222)
CUDA_ERROR_JIT_COMPILATION_DISABLED = enum_cudaError_enum.define('CUDA_ERROR_JIT_COMPILATION_DISABLED', 223)
CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = enum_cudaError_enum.define('CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY', 224)
CUDA_ERROR_INVALID_SOURCE = enum_cudaError_enum.define('CUDA_ERROR_INVALID_SOURCE', 300)
CUDA_ERROR_FILE_NOT_FOUND = enum_cudaError_enum.define('CUDA_ERROR_FILE_NOT_FOUND', 301)
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = enum_cudaError_enum.define('CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND', 302)
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = enum_cudaError_enum.define('CUDA_ERROR_SHARED_OBJECT_INIT_FAILED', 303)
CUDA_ERROR_OPERATING_SYSTEM = enum_cudaError_enum.define('CUDA_ERROR_OPERATING_SYSTEM', 304)
CUDA_ERROR_INVALID_HANDLE = enum_cudaError_enum.define('CUDA_ERROR_INVALID_HANDLE', 400)
CUDA_ERROR_ILLEGAL_STATE = enum_cudaError_enum.define('CUDA_ERROR_ILLEGAL_STATE', 401)
CUDA_ERROR_NOT_FOUND = enum_cudaError_enum.define('CUDA_ERROR_NOT_FOUND', 500)
CUDA_ERROR_NOT_READY = enum_cudaError_enum.define('CUDA_ERROR_NOT_READY', 600)
CUDA_ERROR_ILLEGAL_ADDRESS = enum_cudaError_enum.define('CUDA_ERROR_ILLEGAL_ADDRESS', 700)
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = enum_cudaError_enum.define('CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES', 701)
CUDA_ERROR_LAUNCH_TIMEOUT = enum_cudaError_enum.define('CUDA_ERROR_LAUNCH_TIMEOUT', 702)
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = enum_cudaError_enum.define('CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING', 703)
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = enum_cudaError_enum.define('CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED', 704)
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = enum_cudaError_enum.define('CUDA_ERROR_PEER_ACCESS_NOT_ENABLED', 705)
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = enum_cudaError_enum.define('CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE', 708)
CUDA_ERROR_CONTEXT_IS_DESTROYED = enum_cudaError_enum.define('CUDA_ERROR_CONTEXT_IS_DESTROYED', 709)
CUDA_ERROR_ASSERT = enum_cudaError_enum.define('CUDA_ERROR_ASSERT', 710)
CUDA_ERROR_TOO_MANY_PEERS = enum_cudaError_enum.define('CUDA_ERROR_TOO_MANY_PEERS', 711)
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = enum_cudaError_enum.define('CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED', 712)
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = enum_cudaError_enum.define('CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED', 713)
CUDA_ERROR_HARDWARE_STACK_ERROR = enum_cudaError_enum.define('CUDA_ERROR_HARDWARE_STACK_ERROR', 714)
CUDA_ERROR_ILLEGAL_INSTRUCTION = enum_cudaError_enum.define('CUDA_ERROR_ILLEGAL_INSTRUCTION', 715)
CUDA_ERROR_MISALIGNED_ADDRESS = enum_cudaError_enum.define('CUDA_ERROR_MISALIGNED_ADDRESS', 716)
CUDA_ERROR_INVALID_ADDRESS_SPACE = enum_cudaError_enum.define('CUDA_ERROR_INVALID_ADDRESS_SPACE', 717)
CUDA_ERROR_INVALID_PC = enum_cudaError_enum.define('CUDA_ERROR_INVALID_PC', 718)
CUDA_ERROR_LAUNCH_FAILED = enum_cudaError_enum.define('CUDA_ERROR_LAUNCH_FAILED', 719)
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = enum_cudaError_enum.define('CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE', 720)
CUDA_ERROR_NOT_PERMITTED = enum_cudaError_enum.define('CUDA_ERROR_NOT_PERMITTED', 800)
CUDA_ERROR_NOT_SUPPORTED = enum_cudaError_enum.define('CUDA_ERROR_NOT_SUPPORTED', 801)
CUDA_ERROR_SYSTEM_NOT_READY = enum_cudaError_enum.define('CUDA_ERROR_SYSTEM_NOT_READY', 802)
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = enum_cudaError_enum.define('CUDA_ERROR_SYSTEM_DRIVER_MISMATCH', 803)
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = enum_cudaError_enum.define('CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE', 804)
CUDA_ERROR_MPS_CONNECTION_FAILED = enum_cudaError_enum.define('CUDA_ERROR_MPS_CONNECTION_FAILED', 805)
CUDA_ERROR_MPS_RPC_FAILURE = enum_cudaError_enum.define('CUDA_ERROR_MPS_RPC_FAILURE', 806)
CUDA_ERROR_MPS_SERVER_NOT_READY = enum_cudaError_enum.define('CUDA_ERROR_MPS_SERVER_NOT_READY', 807)
CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = enum_cudaError_enum.define('CUDA_ERROR_MPS_MAX_CLIENTS_REACHED', 808)
CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = enum_cudaError_enum.define('CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED', 809)
CUDA_ERROR_MPS_CLIENT_TERMINATED = enum_cudaError_enum.define('CUDA_ERROR_MPS_CLIENT_TERMINATED', 810)
CUDA_ERROR_CDP_NOT_SUPPORTED = enum_cudaError_enum.define('CUDA_ERROR_CDP_NOT_SUPPORTED', 811)
CUDA_ERROR_CDP_VERSION_MISMATCH = enum_cudaError_enum.define('CUDA_ERROR_CDP_VERSION_MISMATCH', 812)
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = enum_cudaError_enum.define('CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED', 900)
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = enum_cudaError_enum.define('CUDA_ERROR_STREAM_CAPTURE_INVALIDATED', 901)
CUDA_ERROR_STREAM_CAPTURE_MERGE = enum_cudaError_enum.define('CUDA_ERROR_STREAM_CAPTURE_MERGE', 902)
CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = enum_cudaError_enum.define('CUDA_ERROR_STREAM_CAPTURE_UNMATCHED', 903)
CUDA_ERROR_STREAM_CAPTURE_UNJOINED = enum_cudaError_enum.define('CUDA_ERROR_STREAM_CAPTURE_UNJOINED', 904)
CUDA_ERROR_STREAM_CAPTURE_ISOLATION = enum_cudaError_enum.define('CUDA_ERROR_STREAM_CAPTURE_ISOLATION', 905)
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = enum_cudaError_enum.define('CUDA_ERROR_STREAM_CAPTURE_IMPLICIT', 906)
CUDA_ERROR_CAPTURED_EVENT = enum_cudaError_enum.define('CUDA_ERROR_CAPTURED_EVENT', 907)
CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = enum_cudaError_enum.define('CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD', 908)
CUDA_ERROR_TIMEOUT = enum_cudaError_enum.define('CUDA_ERROR_TIMEOUT', 909)
CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = enum_cudaError_enum.define('CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE', 910)
CUDA_ERROR_EXTERNAL_DEVICE = enum_cudaError_enum.define('CUDA_ERROR_EXTERNAL_DEVICE', 911)
CUDA_ERROR_INVALID_CLUSTER_SIZE = enum_cudaError_enum.define('CUDA_ERROR_INVALID_CLUSTER_SIZE', 912)
CUDA_ERROR_UNKNOWN = enum_cudaError_enum.define('CUDA_ERROR_UNKNOWN', 999)

CUresult: TypeAlias = enum_cudaError_enum
enum_CUdevice_P2PAttribute_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK', 1)
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED', 2)
CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED', 3)
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED', 4)
CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED', 4)

CUdevice_P2PAttribute: TypeAlias = enum_CUdevice_P2PAttribute_enum
CUstreamCallback: TypeAlias = CFUNCTYPE(None, POINTER(struct_CUstream_st), enum_cudaError_enum, POINTER(None))
CUoccupancyB2DSize: TypeAlias = CFUNCTYPE(Annotated[int, ctypes.c_uint64], Annotated[int, ctypes.c_int32])
@record
class struct_CUDA_MEMCPY2D_st:
  SIZE = 128
  srcXInBytes: Annotated[size_t, 0]
  srcY: Annotated[size_t, 8]
  srcMemoryType: Annotated[CUmemorytype, 16]
  srcHost: Annotated[POINTER(None), 24]
  srcDevice: Annotated[CUdeviceptr, 32]
  srcArray: Annotated[CUarray, 40]
  srcPitch: Annotated[size_t, 48]
  dstXInBytes: Annotated[size_t, 56]
  dstY: Annotated[size_t, 64]
  dstMemoryType: Annotated[CUmemorytype, 72]
  dstHost: Annotated[POINTER(None), 80]
  dstDevice: Annotated[CUdeviceptr, 88]
  dstArray: Annotated[CUarray, 96]
  dstPitch: Annotated[size_t, 104]
  WidthInBytes: Annotated[size_t, 112]
  Height: Annotated[size_t, 120]
CUDA_MEMCPY2D_v2: TypeAlias = struct_CUDA_MEMCPY2D_st
CUDA_MEMCPY2D: TypeAlias = struct_CUDA_MEMCPY2D_st
@record
class struct_CUDA_MEMCPY3D_st:
  SIZE = 200
  srcXInBytes: Annotated[size_t, 0]
  srcY: Annotated[size_t, 8]
  srcZ: Annotated[size_t, 16]
  srcLOD: Annotated[size_t, 24]
  srcMemoryType: Annotated[CUmemorytype, 32]
  srcHost: Annotated[POINTER(None), 40]
  srcDevice: Annotated[CUdeviceptr, 48]
  srcArray: Annotated[CUarray, 56]
  reserved0: Annotated[POINTER(None), 64]
  srcPitch: Annotated[size_t, 72]
  srcHeight: Annotated[size_t, 80]
  dstXInBytes: Annotated[size_t, 88]
  dstY: Annotated[size_t, 96]
  dstZ: Annotated[size_t, 104]
  dstLOD: Annotated[size_t, 112]
  dstMemoryType: Annotated[CUmemorytype, 120]
  dstHost: Annotated[POINTER(None), 128]
  dstDevice: Annotated[CUdeviceptr, 136]
  dstArray: Annotated[CUarray, 144]
  reserved1: Annotated[POINTER(None), 152]
  dstPitch: Annotated[size_t, 160]
  dstHeight: Annotated[size_t, 168]
  WidthInBytes: Annotated[size_t, 176]
  Height: Annotated[size_t, 184]
  Depth: Annotated[size_t, 192]
CUDA_MEMCPY3D_v2: TypeAlias = struct_CUDA_MEMCPY3D_st
CUDA_MEMCPY3D: TypeAlias = struct_CUDA_MEMCPY3D_st
@record
class struct_CUDA_MEMCPY3D_PEER_st:
  SIZE = 200
  srcXInBytes: Annotated[size_t, 0]
  srcY: Annotated[size_t, 8]
  srcZ: Annotated[size_t, 16]
  srcLOD: Annotated[size_t, 24]
  srcMemoryType: Annotated[CUmemorytype, 32]
  srcHost: Annotated[POINTER(None), 40]
  srcDevice: Annotated[CUdeviceptr, 48]
  srcArray: Annotated[CUarray, 56]
  srcContext: Annotated[CUcontext, 64]
  srcPitch: Annotated[size_t, 72]
  srcHeight: Annotated[size_t, 80]
  dstXInBytes: Annotated[size_t, 88]
  dstY: Annotated[size_t, 96]
  dstZ: Annotated[size_t, 104]
  dstLOD: Annotated[size_t, 112]
  dstMemoryType: Annotated[CUmemorytype, 120]
  dstHost: Annotated[POINTER(None), 128]
  dstDevice: Annotated[CUdeviceptr, 136]
  dstArray: Annotated[CUarray, 144]
  dstContext: Annotated[CUcontext, 152]
  dstPitch: Annotated[size_t, 160]
  dstHeight: Annotated[size_t, 168]
  WidthInBytes: Annotated[size_t, 176]
  Height: Annotated[size_t, 184]
  Depth: Annotated[size_t, 192]
CUDA_MEMCPY3D_PEER_v1: TypeAlias = struct_CUDA_MEMCPY3D_PEER_st
CUDA_MEMCPY3D_PEER: TypeAlias = struct_CUDA_MEMCPY3D_PEER_st
@record
class struct_CUDA_ARRAY_DESCRIPTOR_st:
  SIZE = 24
  Width: Annotated[size_t, 0]
  Height: Annotated[size_t, 8]
  Format: Annotated[CUarray_format, 16]
  NumChannels: Annotated[Annotated[int, ctypes.c_uint32], 20]
CUDA_ARRAY_DESCRIPTOR_v2: TypeAlias = struct_CUDA_ARRAY_DESCRIPTOR_st
CUDA_ARRAY_DESCRIPTOR: TypeAlias = struct_CUDA_ARRAY_DESCRIPTOR_st
@record
class struct_CUDA_ARRAY3D_DESCRIPTOR_st:
  SIZE = 40
  Width: Annotated[size_t, 0]
  Height: Annotated[size_t, 8]
  Depth: Annotated[size_t, 16]
  Format: Annotated[CUarray_format, 24]
  NumChannels: Annotated[Annotated[int, ctypes.c_uint32], 28]
  Flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
CUDA_ARRAY3D_DESCRIPTOR_v2: TypeAlias = struct_CUDA_ARRAY3D_DESCRIPTOR_st
CUDA_ARRAY3D_DESCRIPTOR: TypeAlias = struct_CUDA_ARRAY3D_DESCRIPTOR_st
@record
class struct_CUDA_ARRAY_SPARSE_PROPERTIES_st:
  SIZE = 48
  tileExtent: Annotated[_anonstruct3, 0]
  miptailFirstLevel: Annotated[Annotated[int, ctypes.c_uint32], 12]
  miptailSize: Annotated[Annotated[int, ctypes.c_uint64], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[4]], 28]
@record
class _anonstruct3:
  SIZE = 12
  width: Annotated[Annotated[int, ctypes.c_uint32], 0]
  height: Annotated[Annotated[int, ctypes.c_uint32], 4]
  depth: Annotated[Annotated[int, ctypes.c_uint32], 8]
CUDA_ARRAY_SPARSE_PROPERTIES_v1: TypeAlias = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
CUDA_ARRAY_SPARSE_PROPERTIES: TypeAlias = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
@record
class struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st:
  SIZE = 32
  size: Annotated[size_t, 0]
  alignment: Annotated[size_t, 8]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[4]], 16]
CUDA_ARRAY_MEMORY_REQUIREMENTS_v1: TypeAlias = struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st
CUDA_ARRAY_MEMORY_REQUIREMENTS: TypeAlias = struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st
@record
class struct_CUDA_RESOURCE_DESC_st:
  SIZE = 144
  resType: Annotated[CUresourcetype, 0]
  res: Annotated[_anonunion4, 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 136]
@record
class _anonunion4:
  SIZE = 128
  array: Annotated[_anonstruct5, 0]
  mipmap: Annotated[_anonstruct6, 0]
  linear: Annotated[_anonstruct7, 0]
  pitch2D: Annotated[_anonstruct8, 0]
  reserved: Annotated[_anonstruct9, 0]
@record
class _anonstruct5:
  SIZE = 8
  hArray: Annotated[CUarray, 0]
@record
class _anonstruct6:
  SIZE = 8
  hMipmappedArray: Annotated[CUmipmappedArray, 0]
@record
class _anonstruct7:
  SIZE = 24
  devPtr: Annotated[CUdeviceptr, 0]
  format: Annotated[CUarray_format, 8]
  numChannels: Annotated[Annotated[int, ctypes.c_uint32], 12]
  sizeInBytes: Annotated[size_t, 16]
@record
class _anonstruct8:
  SIZE = 40
  devPtr: Annotated[CUdeviceptr, 0]
  format: Annotated[CUarray_format, 8]
  numChannels: Annotated[Annotated[int, ctypes.c_uint32], 12]
  width: Annotated[size_t, 16]
  height: Annotated[size_t, 24]
  pitchInBytes: Annotated[size_t, 32]
@record
class _anonstruct9:
  SIZE = 128
  reserved: Annotated[Array[Annotated[int, ctypes.c_int32], Literal[32]], 0]
CUDA_RESOURCE_DESC_v1: TypeAlias = struct_CUDA_RESOURCE_DESC_st
CUDA_RESOURCE_DESC: TypeAlias = struct_CUDA_RESOURCE_DESC_st
@record
class struct_CUDA_TEXTURE_DESC_st:
  SIZE = 104
  addressMode: Annotated[Array[CUaddress_mode, Literal[3]], 0]
  filterMode: Annotated[CUfilter_mode, 12]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  maxAnisotropy: Annotated[Annotated[int, ctypes.c_uint32], 20]
  mipmapFilterMode: Annotated[CUfilter_mode, 24]
  mipmapLevelBias: Annotated[Annotated[float, ctypes.c_float], 28]
  minMipmapLevelClamp: Annotated[Annotated[float, ctypes.c_float], 32]
  maxMipmapLevelClamp: Annotated[Annotated[float, ctypes.c_float], 36]
  borderColor: Annotated[Array[Annotated[float, ctypes.c_float], Literal[4]], 40]
  reserved: Annotated[Array[Annotated[int, ctypes.c_int32], Literal[12]], 56]
CUDA_TEXTURE_DESC_v1: TypeAlias = struct_CUDA_TEXTURE_DESC_st
CUDA_TEXTURE_DESC: TypeAlias = struct_CUDA_TEXTURE_DESC_st
enum_CUresourceViewFormat_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_RES_VIEW_FORMAT_NONE = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_NONE', 0)
CU_RES_VIEW_FORMAT_UINT_1X8 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_1X8', 1)
CU_RES_VIEW_FORMAT_UINT_2X8 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_2X8', 2)
CU_RES_VIEW_FORMAT_UINT_4X8 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_4X8', 3)
CU_RES_VIEW_FORMAT_SINT_1X8 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_1X8', 4)
CU_RES_VIEW_FORMAT_SINT_2X8 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_2X8', 5)
CU_RES_VIEW_FORMAT_SINT_4X8 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_4X8', 6)
CU_RES_VIEW_FORMAT_UINT_1X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_1X16', 7)
CU_RES_VIEW_FORMAT_UINT_2X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_2X16', 8)
CU_RES_VIEW_FORMAT_UINT_4X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_4X16', 9)
CU_RES_VIEW_FORMAT_SINT_1X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_1X16', 10)
CU_RES_VIEW_FORMAT_SINT_2X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_2X16', 11)
CU_RES_VIEW_FORMAT_SINT_4X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_4X16', 12)
CU_RES_VIEW_FORMAT_UINT_1X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_1X32', 13)
CU_RES_VIEW_FORMAT_UINT_2X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_2X32', 14)
CU_RES_VIEW_FORMAT_UINT_4X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UINT_4X32', 15)
CU_RES_VIEW_FORMAT_SINT_1X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_1X32', 16)
CU_RES_VIEW_FORMAT_SINT_2X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_2X32', 17)
CU_RES_VIEW_FORMAT_SINT_4X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SINT_4X32', 18)
CU_RES_VIEW_FORMAT_FLOAT_1X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_FLOAT_1X16', 19)
CU_RES_VIEW_FORMAT_FLOAT_2X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_FLOAT_2X16', 20)
CU_RES_VIEW_FORMAT_FLOAT_4X16 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_FLOAT_4X16', 21)
CU_RES_VIEW_FORMAT_FLOAT_1X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_FLOAT_1X32', 22)
CU_RES_VIEW_FORMAT_FLOAT_2X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_FLOAT_2X32', 23)
CU_RES_VIEW_FORMAT_FLOAT_4X32 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_FLOAT_4X32', 24)
CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UNSIGNED_BC1', 25)
CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UNSIGNED_BC2', 26)
CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UNSIGNED_BC3', 27)
CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UNSIGNED_BC4', 28)
CU_RES_VIEW_FORMAT_SIGNED_BC4 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SIGNED_BC4', 29)
CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UNSIGNED_BC5', 30)
CU_RES_VIEW_FORMAT_SIGNED_BC5 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SIGNED_BC5', 31)
CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UNSIGNED_BC6H', 32)
CU_RES_VIEW_FORMAT_SIGNED_BC6H = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_SIGNED_BC6H', 33)
CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = enum_CUresourceViewFormat_enum.define('CU_RES_VIEW_FORMAT_UNSIGNED_BC7', 34)

CUresourceViewFormat: TypeAlias = enum_CUresourceViewFormat_enum
@record
class struct_CUDA_RESOURCE_VIEW_DESC_st:
  SIZE = 112
  format: Annotated[CUresourceViewFormat, 0]
  width: Annotated[size_t, 8]
  height: Annotated[size_t, 16]
  depth: Annotated[size_t, 24]
  firstMipmapLevel: Annotated[Annotated[int, ctypes.c_uint32], 32]
  lastMipmapLevel: Annotated[Annotated[int, ctypes.c_uint32], 36]
  firstLayer: Annotated[Annotated[int, ctypes.c_uint32], 40]
  lastLayer: Annotated[Annotated[int, ctypes.c_uint32], 44]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[16]], 48]
CUDA_RESOURCE_VIEW_DESC_v1: TypeAlias = struct_CUDA_RESOURCE_VIEW_DESC_st
CUDA_RESOURCE_VIEW_DESC: TypeAlias = struct_CUDA_RESOURCE_VIEW_DESC_st
@record
class struct_CUtensorMap_st:
  SIZE = 128
  opaque: Annotated[Array[cuuint64_t, Literal[16]], 0]
CUtensorMap: TypeAlias = struct_CUtensorMap_st
enum_CUtensorMapDataType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_TENSOR_MAP_DATA_TYPE_UINT8 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_UINT8', 0)
CU_TENSOR_MAP_DATA_TYPE_UINT16 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_UINT16', 1)
CU_TENSOR_MAP_DATA_TYPE_UINT32 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_UINT32', 2)
CU_TENSOR_MAP_DATA_TYPE_INT32 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_INT32', 3)
CU_TENSOR_MAP_DATA_TYPE_UINT64 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_UINT64', 4)
CU_TENSOR_MAP_DATA_TYPE_INT64 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_INT64', 5)
CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_FLOAT16', 6)
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_FLOAT32', 7)
CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_FLOAT64', 8)
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_BFLOAT16', 9)
CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ', 10)
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_TFLOAT32', 11)
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = enum_CUtensorMapDataType_enum.define('CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ', 12)

CUtensorMapDataType: TypeAlias = enum_CUtensorMapDataType_enum
enum_CUtensorMapInterleave_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_TENSOR_MAP_INTERLEAVE_NONE = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_NONE', 0)
CU_TENSOR_MAP_INTERLEAVE_16B = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_16B', 1)
CU_TENSOR_MAP_INTERLEAVE_32B = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_32B', 2)

CUtensorMapInterleave: TypeAlias = enum_CUtensorMapInterleave_enum
enum_CUtensorMapSwizzle_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_TENSOR_MAP_SWIZZLE_NONE = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_NONE', 0)
CU_TENSOR_MAP_SWIZZLE_32B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_32B', 1)
CU_TENSOR_MAP_SWIZZLE_64B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_64B', 2)
CU_TENSOR_MAP_SWIZZLE_128B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_128B', 3)

CUtensorMapSwizzle: TypeAlias = enum_CUtensorMapSwizzle_enum
enum_CUtensorMapL2promotion_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_TENSOR_MAP_L2_PROMOTION_NONE = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_NONE', 0)
CU_TENSOR_MAP_L2_PROMOTION_L2_64B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_64B', 1)
CU_TENSOR_MAP_L2_PROMOTION_L2_128B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_128B', 2)
CU_TENSOR_MAP_L2_PROMOTION_L2_256B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_256B', 3)

CUtensorMapL2promotion: TypeAlias = enum_CUtensorMapL2promotion_enum
enum_CUtensorMapFloatOOBfill_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = enum_CUtensorMapFloatOOBfill_enum.define('CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE', 0)
CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA = enum_CUtensorMapFloatOOBfill_enum.define('CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA', 1)

CUtensorMapFloatOOBfill: TypeAlias = enum_CUtensorMapFloatOOBfill_enum
@record
class struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st:
  SIZE = 16
  p2pToken: Annotated[Annotated[int, ctypes.c_uint64], 0]
  vaSpaceToken: Annotated[Annotated[int, ctypes.c_uint32], 8]
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1: TypeAlias = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS: TypeAlias = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE', 0)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ', 1)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE', 3)

CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS: TypeAlias = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum
@record
class struct_CUDA_LAUNCH_PARAMS_st:
  SIZE = 56
  function: Annotated[CUfunction, 0]
  gridDimX: Annotated[Annotated[int, ctypes.c_uint32], 8]
  gridDimY: Annotated[Annotated[int, ctypes.c_uint32], 12]
  gridDimZ: Annotated[Annotated[int, ctypes.c_uint32], 16]
  blockDimX: Annotated[Annotated[int, ctypes.c_uint32], 20]
  blockDimY: Annotated[Annotated[int, ctypes.c_uint32], 24]
  blockDimZ: Annotated[Annotated[int, ctypes.c_uint32], 28]
  sharedMemBytes: Annotated[Annotated[int, ctypes.c_uint32], 32]
  hStream: Annotated[CUstream, 40]
  kernelParams: Annotated[POINTER(POINTER(None)), 48]
CUDA_LAUNCH_PARAMS_v1: TypeAlias = struct_CUDA_LAUNCH_PARAMS_st
CUDA_LAUNCH_PARAMS: TypeAlias = struct_CUDA_LAUNCH_PARAMS_st
enum_CUexternalMemoryHandleType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD', 1)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32', 2)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT', 3)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP', 4)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE', 5)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE', 6)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT', 7)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF', 8)

CUexternalMemoryHandleType: TypeAlias = enum_CUexternalMemoryHandleType_enum
@record
class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st:
  SIZE = 104
  type: Annotated[CUexternalMemoryHandleType, 0]
  handle: Annotated[_anonunion10, 8]
  size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[16]], 36]
@record
class _anonunion10:
  SIZE = 16
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  win32: Annotated[_anonstruct11, 0]
  nvSciBufObject: Annotated[POINTER(None), 0]
@record
class _anonstruct11:
  SIZE = 16
  handle: Annotated[POINTER(None), 0]
  name: Annotated[POINTER(None), 8]
CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1: TypeAlias = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
CUDA_EXTERNAL_MEMORY_HANDLE_DESC: TypeAlias = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
@record
class struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st:
  SIZE = 88
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[16]], 20]
CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1: TypeAlias = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
CUDA_EXTERNAL_MEMORY_BUFFER_DESC: TypeAlias = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
@record
class struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st:
  SIZE = 120
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  arrayDesc: Annotated[CUDA_ARRAY3D_DESCRIPTOR, 8]
  numLevels: Annotated[Annotated[int, ctypes.c_uint32], 48]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[16]], 52]
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1: TypeAlias = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC: TypeAlias = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
enum_CUexternalSemaphoreHandleType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD', 1)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32', 2)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT', 3)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE', 4)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE', 5)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC', 6)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX', 7)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT', 8)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD', 9)
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = enum_CUexternalSemaphoreHandleType_enum.define('CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32', 10)

CUexternalSemaphoreHandleType: TypeAlias = enum_CUexternalSemaphoreHandleType_enum
@record
class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st:
  SIZE = 96
  type: Annotated[CUexternalSemaphoreHandleType, 0]
  handle: Annotated[_anonunion12, 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[16]], 28]
@record
class _anonunion12:
  SIZE = 16
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  win32: Annotated[_anonstruct13, 0]
  nvSciSyncObj: Annotated[POINTER(None), 0]
@record
class _anonstruct13:
  SIZE = 16
  handle: Annotated[POINTER(None), 0]
  name: Annotated[POINTER(None), 8]
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1: TypeAlias = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC: TypeAlias = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
@record
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st:
  SIZE = 144
  params: Annotated[_anonstruct14, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 72]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[16]], 76]
@record
class _anonstruct14:
  SIZE = 72
  fence: Annotated[_anonstruct15, 0]
  nvSciSync: Annotated[_anonunion16, 8]
  keyedMutex: Annotated[_anonstruct17, 16]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[12]], 24]
@record
class _anonstruct15:
  SIZE = 8
  value: Annotated[Annotated[int, ctypes.c_uint64], 0]
@record
class _anonunion16:
  SIZE = 8
  fence: Annotated[POINTER(None), 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint64], 0]
@record
class _anonstruct17:
  SIZE = 8
  key: Annotated[Annotated[int, ctypes.c_uint64], 0]
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1: TypeAlias = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS: TypeAlias = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
@record
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st:
  SIZE = 144
  params: Annotated[_anonstruct18, 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 72]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[16]], 76]
@record
class _anonstruct18:
  SIZE = 72
  fence: Annotated[_anonstruct19, 0]
  nvSciSync: Annotated[_anonunion20, 8]
  keyedMutex: Annotated[_anonstruct21, 16]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[10]], 32]
@record
class _anonstruct19:
  SIZE = 8
  value: Annotated[Annotated[int, ctypes.c_uint64], 0]
@record
class _anonunion20:
  SIZE = 8
  fence: Annotated[POINTER(None), 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint64], 0]
@record
class _anonstruct21:
  SIZE = 16
  key: Annotated[Annotated[int, ctypes.c_uint64], 0]
  timeoutMs: Annotated[Annotated[int, ctypes.c_uint32], 8]
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1: TypeAlias = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS: TypeAlias = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
@record
class struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st:
  SIZE = 24
  extSemArray: Annotated[POINTER(CUexternalSemaphore), 0]
  paramsArray: Annotated[POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), 8]
  numExtSems: Annotated[Annotated[int, ctypes.c_uint32], 16]
CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1: TypeAlias = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
CUDA_EXT_SEM_SIGNAL_NODE_PARAMS: TypeAlias = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
@record
class struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st:
  SIZE = 24
  extSemArray: Annotated[POINTER(CUexternalSemaphore), 0]
  paramsArray: Annotated[POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS), 8]
  numExtSems: Annotated[Annotated[int, ctypes.c_uint32], 16]
CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1: TypeAlias = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUDA_EXT_SEM_WAIT_NODE_PARAMS: TypeAlias = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUmemGenericAllocationHandle_v1: TypeAlias = Annotated[int, ctypes.c_uint64]
CUmemGenericAllocationHandle: TypeAlias = Annotated[int, ctypes.c_uint64]
enum_CUmemAllocationHandleType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_HANDLE_TYPE_NONE = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_NONE', 0)
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR', 1)
CU_MEM_HANDLE_TYPE_WIN32 = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_WIN32', 2)
CU_MEM_HANDLE_TYPE_WIN32_KMT = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_WIN32_KMT', 4)
CU_MEM_HANDLE_TYPE_MAX = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_MAX', 2147483647)

CUmemAllocationHandleType: TypeAlias = enum_CUmemAllocationHandleType_enum
enum_CUmemAccess_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_ACCESS_FLAGS_PROT_NONE = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_NONE', 0)
CU_MEM_ACCESS_FLAGS_PROT_READ = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_READ', 1)
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_READWRITE', 3)
CU_MEM_ACCESS_FLAGS_PROT_MAX = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_MAX', 2147483647)

CUmemAccess_flags: TypeAlias = enum_CUmemAccess_flags_enum
enum_CUmemLocationType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_LOCATION_TYPE_INVALID = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_INVALID', 0)
CU_MEM_LOCATION_TYPE_DEVICE = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_DEVICE', 1)
CU_MEM_LOCATION_TYPE_MAX = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_MAX', 2147483647)

CUmemLocationType: TypeAlias = enum_CUmemLocationType_enum
enum_CUmemAllocationType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_ALLOCATION_TYPE_INVALID = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_INVALID', 0)
CU_MEM_ALLOCATION_TYPE_PINNED = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_PINNED', 1)
CU_MEM_ALLOCATION_TYPE_MAX = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_MAX', 2147483647)

CUmemAllocationType: TypeAlias = enum_CUmemAllocationType_enum
enum_CUmemAllocationGranularity_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_ALLOC_GRANULARITY_MINIMUM = enum_CUmemAllocationGranularity_flags_enum.define('CU_MEM_ALLOC_GRANULARITY_MINIMUM', 0)
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = enum_CUmemAllocationGranularity_flags_enum.define('CU_MEM_ALLOC_GRANULARITY_RECOMMENDED', 1)

CUmemAllocationGranularity_flags: TypeAlias = enum_CUmemAllocationGranularity_flags_enum
enum_CUmemRangeHandleType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = enum_CUmemRangeHandleType_enum.define('CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD', 1)
CU_MEM_RANGE_HANDLE_TYPE_MAX = enum_CUmemRangeHandleType_enum.define('CU_MEM_RANGE_HANDLE_TYPE_MAX', 2147483647)

CUmemRangeHandleType: TypeAlias = enum_CUmemRangeHandleType_enum
enum_CUarraySparseSubresourceType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = enum_CUarraySparseSubresourceType_enum.define('CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL', 0)
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = enum_CUarraySparseSubresourceType_enum.define('CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL', 1)

CUarraySparseSubresourceType: TypeAlias = enum_CUarraySparseSubresourceType_enum
enum_CUmemOperationType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_OPERATION_TYPE_MAP = enum_CUmemOperationType_enum.define('CU_MEM_OPERATION_TYPE_MAP', 1)
CU_MEM_OPERATION_TYPE_UNMAP = enum_CUmemOperationType_enum.define('CU_MEM_OPERATION_TYPE_UNMAP', 2)

CUmemOperationType: TypeAlias = enum_CUmemOperationType_enum
enum_CUmemHandleType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_HANDLE_TYPE_GENERIC = enum_CUmemHandleType_enum.define('CU_MEM_HANDLE_TYPE_GENERIC', 0)

CUmemHandleType: TypeAlias = enum_CUmemHandleType_enum
@record
class struct_CUarrayMapInfo_st:
  SIZE = 96
  resourceType: Annotated[CUresourcetype, 0]
  resource: Annotated[_anonunion22, 8]
  subresourceType: Annotated[CUarraySparseSubresourceType, 16]
  subresource: Annotated[_anonunion23, 24]
  memOperationType: Annotated[CUmemOperationType, 56]
  memHandleType: Annotated[CUmemHandleType, 60]
  memHandle: Annotated[_anonunion26, 64]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 72]
  deviceBitMask: Annotated[Annotated[int, ctypes.c_uint32], 80]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 84]
  reserved: Annotated[Array[Annotated[int, ctypes.c_uint32], Literal[2]], 88]
@record
class _anonunion22:
  SIZE = 8
  mipmap: Annotated[CUmipmappedArray, 0]
  array: Annotated[CUarray, 0]
@record
class _anonunion23:
  SIZE = 32
  sparseLevel: Annotated[_anonstruct24, 0]
  miptail: Annotated[_anonstruct25, 0]
@record
class _anonstruct24:
  SIZE = 32
  level: Annotated[Annotated[int, ctypes.c_uint32], 0]
  layer: Annotated[Annotated[int, ctypes.c_uint32], 4]
  offsetX: Annotated[Annotated[int, ctypes.c_uint32], 8]
  offsetY: Annotated[Annotated[int, ctypes.c_uint32], 12]
  offsetZ: Annotated[Annotated[int, ctypes.c_uint32], 16]
  extentWidth: Annotated[Annotated[int, ctypes.c_uint32], 20]
  extentHeight: Annotated[Annotated[int, ctypes.c_uint32], 24]
  extentDepth: Annotated[Annotated[int, ctypes.c_uint32], 28]
@record
class _anonstruct25:
  SIZE = 24
  layer: Annotated[Annotated[int, ctypes.c_uint32], 0]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
  size: Annotated[Annotated[int, ctypes.c_uint64], 16]
@record
class _anonunion26:
  SIZE = 8
  memHandle: Annotated[CUmemGenericAllocationHandle, 0]
CUarrayMapInfo_v1: TypeAlias = struct_CUarrayMapInfo_st
CUarrayMapInfo: TypeAlias = struct_CUarrayMapInfo_st
@record
class struct_CUmemLocation_st:
  SIZE = 8
  type: Annotated[CUmemLocationType, 0]
  id: Annotated[Annotated[int, ctypes.c_int32], 4]
CUmemLocation_v1: TypeAlias = struct_CUmemLocation_st
CUmemLocation: TypeAlias = struct_CUmemLocation_st
enum_CUmemAllocationCompType_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEM_ALLOCATION_COMP_NONE = enum_CUmemAllocationCompType_enum.define('CU_MEM_ALLOCATION_COMP_NONE', 0)
CU_MEM_ALLOCATION_COMP_GENERIC = enum_CUmemAllocationCompType_enum.define('CU_MEM_ALLOCATION_COMP_GENERIC', 1)

CUmemAllocationCompType: TypeAlias = enum_CUmemAllocationCompType_enum
@record
class struct_CUmemAllocationProp_st:
  SIZE = 32
  type: Annotated[CUmemAllocationType, 0]
  requestedHandleTypes: Annotated[CUmemAllocationHandleType, 4]
  location: Annotated[CUmemLocation, 8]
  win32HandleMetaData: Annotated[POINTER(None), 16]
  allocFlags: Annotated[_anonstruct27, 24]
@record
class _anonstruct27:
  SIZE = 8
  compressionType: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  gpuDirectRDMACapable: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  usage: Annotated[Annotated[int, ctypes.c_uint16], 2]
  reserved: Annotated[Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
CUmemAllocationProp_v1: TypeAlias = struct_CUmemAllocationProp_st
CUmemAllocationProp: TypeAlias = struct_CUmemAllocationProp_st
@record
class struct_CUmemAccessDesc_st:
  SIZE = 12
  location: Annotated[CUmemLocation, 0]
  flags: Annotated[CUmemAccess_flags, 8]
CUmemAccessDesc_v1: TypeAlias = struct_CUmemAccessDesc_st
CUmemAccessDesc: TypeAlias = struct_CUmemAccessDesc_st
enum_CUgraphExecUpdateResult_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GRAPH_EXEC_UPDATE_SUCCESS = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_SUCCESS', 0)
CU_GRAPH_EXEC_UPDATE_ERROR = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR', 1)
CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED', 2)
CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED', 3)
CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED', 4)
CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED', 5)
CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED', 6)
CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE', 7)
CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED', 8)

CUgraphExecUpdateResult: TypeAlias = enum_CUgraphExecUpdateResult_enum
@record
class struct_CUgraphExecUpdateResultInfo_st:
  SIZE = 24
  result: Annotated[CUgraphExecUpdateResult, 0]
  errorNode: Annotated[CUgraphNode, 8]
  errorFromNode: Annotated[CUgraphNode, 16]
CUgraphExecUpdateResultInfo_v1: TypeAlias = struct_CUgraphExecUpdateResultInfo_st
CUgraphExecUpdateResultInfo: TypeAlias = struct_CUgraphExecUpdateResultInfo_st
enum_CUmemPool_attribute_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES', 1)
CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC', 2)
CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES', 3)
CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RELEASE_THRESHOLD', 4)
CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT', 5)
CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH', 6)
CU_MEMPOOL_ATTR_USED_MEM_CURRENT = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_USED_MEM_CURRENT', 7)
CU_MEMPOOL_ATTR_USED_MEM_HIGH = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_USED_MEM_HIGH', 8)

CUmemPool_attribute: TypeAlias = enum_CUmemPool_attribute_enum
@record
class struct_CUmemPoolProps_st:
  SIZE = 88
  allocType: Annotated[CUmemAllocationType, 0]
  handleTypes: Annotated[CUmemAllocationHandleType, 4]
  location: Annotated[CUmemLocation, 8]
  win32SecurityAttributes: Annotated[POINTER(None), 16]
  reserved: Annotated[Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 24]
CUmemPoolProps_v1: TypeAlias = struct_CUmemPoolProps_st
CUmemPoolProps: TypeAlias = struct_CUmemPoolProps_st
@record
class struct_CUmemPoolPtrExportData_st:
  SIZE = 64
  reserved: Annotated[Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
CUmemPoolPtrExportData_v1: TypeAlias = struct_CUmemPoolPtrExportData_st
CUmemPoolPtrExportData: TypeAlias = struct_CUmemPoolPtrExportData_st
@record
class struct_CUDA_MEM_ALLOC_NODE_PARAMS_st:
  SIZE = 120
  poolProps: Annotated[CUmemPoolProps, 0]
  accessDescs: Annotated[POINTER(CUmemAccessDesc), 88]
  accessDescCount: Annotated[size_t, 96]
  bytesize: Annotated[size_t, 104]
  dptr: Annotated[CUdeviceptr, 112]
CUDA_MEM_ALLOC_NODE_PARAMS: TypeAlias = struct_CUDA_MEM_ALLOC_NODE_PARAMS_st
enum_CUgraphMem_attribute_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT', 0)
CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_USED_MEM_HIGH', 1)
CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT', 2)
CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH', 3)

CUgraphMem_attribute: TypeAlias = enum_CUgraphMem_attribute_enum
enum_CUflushGPUDirectRDMAWritesOptions_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = enum_CUflushGPUDirectRDMAWritesOptions_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST', 1)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = enum_CUflushGPUDirectRDMAWritesOptions_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS', 2)

CUflushGPUDirectRDMAWritesOptions: TypeAlias = enum_CUflushGPUDirectRDMAWritesOptions_enum
enum_CUGPUDirectRDMAWritesOrdering_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE', 0)
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER', 100)
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES', 200)

CUGPUDirectRDMAWritesOrdering: TypeAlias = enum_CUGPUDirectRDMAWritesOrdering_enum
enum_CUflushGPUDirectRDMAWritesScope_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = enum_CUflushGPUDirectRDMAWritesScope_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER', 100)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = enum_CUflushGPUDirectRDMAWritesScope_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES', 200)

CUflushGPUDirectRDMAWritesScope: TypeAlias = enum_CUflushGPUDirectRDMAWritesScope_enum
enum_CUflushGPUDirectRDMAWritesTarget_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = enum_CUflushGPUDirectRDMAWritesTarget_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX', 0)

CUflushGPUDirectRDMAWritesTarget: TypeAlias = enum_CUflushGPUDirectRDMAWritesTarget_enum
enum_CUgraphDebugDot_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE', 1)
CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES', 2)
CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS', 4)
CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS', 8)
CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS', 16)
CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS', 32)
CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS', 64)
CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS', 128)
CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS', 256)
CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES', 512)
CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES', 1024)
CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS', 2048)
CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS', 4096)
CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS', 8192)
CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO = enum_CUgraphDebugDot_flags_enum.define('CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO', 16384)

CUgraphDebugDot_flags: TypeAlias = enum_CUgraphDebugDot_flags_enum
enum_CUuserObject_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = enum_CUuserObject_flags_enum.define('CU_USER_OBJECT_NO_DESTRUCTOR_SYNC', 1)

CUuserObject_flags: TypeAlias = enum_CUuserObject_flags_enum
enum_CUuserObjectRetain_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_GRAPH_USER_OBJECT_MOVE = enum_CUuserObjectRetain_flags_enum.define('CU_GRAPH_USER_OBJECT_MOVE', 1)

CUuserObjectRetain_flags: TypeAlias = enum_CUuserObjectRetain_flags_enum
enum_CUgraphInstantiate_flags_enum = CEnum(Annotated[int, ctypes.c_uint32])
CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH', 1)
CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD', 2)
CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH', 4)
CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY', 8)

CUgraphInstantiate_flags: TypeAlias = enum_CUgraphInstantiate_flags_enum
@dll.bind
def cuGetErrorString(error:CUresult, pStr:POINTER(POINTER(Annotated[bytes, ctypes.c_char]))) -> CUresult: ...
@dll.bind
def cuGetErrorName(error:CUresult, pStr:POINTER(POINTER(Annotated[bytes, ctypes.c_char]))) -> CUresult: ...
@dll.bind
def cuInit(Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuDriverGetVersion(driverVersion:POINTER(Annotated[int, ctypes.c_int32])) -> CUresult: ...
@dll.bind
def cuDeviceGet(device:POINTER(CUdevice), ordinal:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuDeviceGetCount(count:POINTER(Annotated[int, ctypes.c_int32])) -> CUresult: ...
@dll.bind
def cuDeviceGetName(name:POINTER(Annotated[bytes, ctypes.c_char]), len:Annotated[int, ctypes.c_int32], dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetUuid(uuid:POINTER(CUuuid), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetUuid_v2(uuid:POINTER(CUuuid), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetLuid(luid:POINTER(Annotated[bytes, ctypes.c_char]), deviceNodeMask:POINTER(Annotated[int, ctypes.c_uint32]), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceTotalMem_v2(bytes:POINTER(size_t), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements:POINTER(size_t), format:CUarray_format, numChannels:Annotated[int, ctypes.c_uint32], dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetAttribute(pi:POINTER(Annotated[int, ctypes.c_int32]), attrib:CUdevice_attribute, dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList:POINTER(None), dev:CUdevice, flags:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuDeviceSetMemPool(dev:CUdevice, pool:CUmemoryPool) -> CUresult: ...
@dll.bind
def cuDeviceGetMemPool(pool:POINTER(CUmemoryPool), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetDefaultMemPool(pool_out:POINTER(CUmemoryPool), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetExecAffinitySupport(pi:POINTER(Annotated[int, ctypes.c_int32]), type:CUexecAffinityType, dev:CUdevice) -> CUresult: ...
@dll.bind
def cuFlushGPUDirectRDMAWrites(target:CUflushGPUDirectRDMAWritesTarget, scope:CUflushGPUDirectRDMAWritesScope) -> CUresult: ...
@dll.bind
def cuDeviceGetProperties(prop:POINTER(CUdevprop), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceComputeCapability(major:POINTER(Annotated[int, ctypes.c_int32]), minor:POINTER(Annotated[int, ctypes.c_int32]), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDevicePrimaryCtxRetain(pctx:POINTER(CUcontext), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDevicePrimaryCtxRelease_v2(dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDevicePrimaryCtxSetFlags_v2(dev:CUdevice, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuDevicePrimaryCtxGetState(dev:CUdevice, flags:POINTER(Annotated[int, ctypes.c_uint32]), active:POINTER(Annotated[int, ctypes.c_int32])) -> CUresult: ...
@dll.bind
def cuDevicePrimaryCtxReset_v2(dev:CUdevice) -> CUresult: ...
@dll.bind
def cuCtxCreate_v2(pctx:POINTER(CUcontext), flags:Annotated[int, ctypes.c_uint32], dev:CUdevice) -> CUresult: ...
@dll.bind
def cuCtxCreate_v3(pctx:POINTER(CUcontext), paramsArray:POINTER(CUexecAffinityParam), numParams:Annotated[int, ctypes.c_int32], flags:Annotated[int, ctypes.c_uint32], dev:CUdevice) -> CUresult: ...
@dll.bind
def cuCtxDestroy_v2(ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuCtxPushCurrent_v2(ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuCtxPopCurrent_v2(pctx:POINTER(CUcontext)) -> CUresult: ...
@dll.bind
def cuCtxSetCurrent(ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuCtxGetCurrent(pctx:POINTER(CUcontext)) -> CUresult: ...
@dll.bind
def cuCtxGetDevice(device:POINTER(CUdevice)) -> CUresult: ...
@dll.bind
def cuCtxGetFlags(flags:POINTER(Annotated[int, ctypes.c_uint32])) -> CUresult: ...
@dll.bind
def cuCtxGetId(ctx:CUcontext, ctxId:POINTER(Annotated[int, ctypes.c_uint64])) -> CUresult: ...
@dll.bind
def cuCtxSynchronize() -> CUresult: ...
@dll.bind
def cuCtxSetLimit(limit:CUlimit, value:size_t) -> CUresult: ...
@dll.bind
def cuCtxGetLimit(pvalue:POINTER(size_t), limit:CUlimit) -> CUresult: ...
@dll.bind
def cuCtxGetCacheConfig(pconfig:POINTER(CUfunc_cache)) -> CUresult: ...
@dll.bind
def cuCtxSetCacheConfig(config:CUfunc_cache) -> CUresult: ...
@dll.bind
def cuCtxGetSharedMemConfig(pConfig:POINTER(CUsharedconfig)) -> CUresult: ...
@dll.bind
def cuCtxSetSharedMemConfig(config:CUsharedconfig) -> CUresult: ...
@dll.bind
def cuCtxGetApiVersion(ctx:CUcontext, version:POINTER(Annotated[int, ctypes.c_uint32])) -> CUresult: ...
@dll.bind
def cuCtxGetStreamPriorityRange(leastPriority:POINTER(Annotated[int, ctypes.c_int32]), greatestPriority:POINTER(Annotated[int, ctypes.c_int32])) -> CUresult: ...
@dll.bind
def cuCtxResetPersistingL2Cache() -> CUresult: ...
@dll.bind
def cuCtxGetExecAffinity(pExecAffinity:POINTER(CUexecAffinityParam), type:CUexecAffinityType) -> CUresult: ...
@dll.bind
def cuCtxAttach(pctx:POINTER(CUcontext), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuCtxDetach(ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuModuleLoad(module:POINTER(CUmodule), fname:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuModuleLoadData(module:POINTER(CUmodule), image:POINTER(None)) -> CUresult: ...
@dll.bind
def cuModuleLoadDataEx(module:POINTER(CUmodule), image:POINTER(None), numOptions:Annotated[int, ctypes.c_uint32], options:POINTER(CUjit_option), optionValues:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuModuleLoadFatBinary(module:POINTER(CUmodule), fatCubin:POINTER(None)) -> CUresult: ...
@dll.bind
def cuModuleUnload(hmod:CUmodule) -> CUresult: ...
enum_CUmoduleLoadingMode_enum = CEnum(Annotated[int, ctypes.c_uint32])
CU_MODULE_EAGER_LOADING = enum_CUmoduleLoadingMode_enum.define('CU_MODULE_EAGER_LOADING', 1)
CU_MODULE_LAZY_LOADING = enum_CUmoduleLoadingMode_enum.define('CU_MODULE_LAZY_LOADING', 2)

CUmoduleLoadingMode: TypeAlias = enum_CUmoduleLoadingMode_enum
@dll.bind
def cuModuleGetLoadingMode(mode:POINTER(CUmoduleLoadingMode)) -> CUresult: ...
@dll.bind
def cuModuleGetFunction(hfunc:POINTER(CUfunction), hmod:CUmodule, name:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuModuleGetGlobal_v2(dptr:POINTER(CUdeviceptr), bytes:POINTER(size_t), hmod:CUmodule, name:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuLinkCreate_v2(numOptions:Annotated[int, ctypes.c_uint32], options:POINTER(CUjit_option), optionValues:POINTER(POINTER(None)), stateOut:POINTER(CUlinkState)) -> CUresult: ...
@dll.bind
def cuLinkAddData_v2(state:CUlinkState, type:CUjitInputType, data:POINTER(None), size:size_t, name:POINTER(Annotated[bytes, ctypes.c_char]), numOptions:Annotated[int, ctypes.c_uint32], options:POINTER(CUjit_option), optionValues:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuLinkAddFile_v2(state:CUlinkState, type:CUjitInputType, path:POINTER(Annotated[bytes, ctypes.c_char]), numOptions:Annotated[int, ctypes.c_uint32], options:POINTER(CUjit_option), optionValues:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuLinkComplete(state:CUlinkState, cubinOut:POINTER(POINTER(None)), sizeOut:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuLinkDestroy(state:CUlinkState) -> CUresult: ...
@dll.bind
def cuModuleGetTexRef(pTexRef:POINTER(CUtexref), hmod:CUmodule, name:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuModuleGetSurfRef(pSurfRef:POINTER(CUsurfref), hmod:CUmodule, name:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuLibraryLoadData(library:POINTER(CUlibrary), code:POINTER(None), jitOptions:POINTER(CUjit_option), jitOptionsValues:POINTER(POINTER(None)), numJitOptions:Annotated[int, ctypes.c_uint32], libraryOptions:POINTER(CUlibraryOption), libraryOptionValues:POINTER(POINTER(None)), numLibraryOptions:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuLibraryLoadFromFile(library:POINTER(CUlibrary), fileName:POINTER(Annotated[bytes, ctypes.c_char]), jitOptions:POINTER(CUjit_option), jitOptionsValues:POINTER(POINTER(None)), numJitOptions:Annotated[int, ctypes.c_uint32], libraryOptions:POINTER(CUlibraryOption), libraryOptionValues:POINTER(POINTER(None)), numLibraryOptions:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuLibraryUnload(library:CUlibrary) -> CUresult: ...
@dll.bind
def cuLibraryGetKernel(pKernel:POINTER(CUkernel), library:CUlibrary, name:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuLibraryGetModule(pMod:POINTER(CUmodule), library:CUlibrary) -> CUresult: ...
@dll.bind
def cuKernelGetFunction(pFunc:POINTER(CUfunction), kernel:CUkernel) -> CUresult: ...
@dll.bind
def cuLibraryGetGlobal(dptr:POINTER(CUdeviceptr), bytes:POINTER(size_t), library:CUlibrary, name:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuLibraryGetManaged(dptr:POINTER(CUdeviceptr), bytes:POINTER(size_t), library:CUlibrary, name:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuLibraryGetUnifiedFunction(fptr:POINTER(POINTER(None)), library:CUlibrary, symbol:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuKernelGetAttribute(pi:POINTER(Annotated[int, ctypes.c_int32]), attrib:CUfunction_attribute, kernel:CUkernel, dev:CUdevice) -> CUresult: ...
@dll.bind
def cuKernelSetAttribute(attrib:CUfunction_attribute, val:Annotated[int, ctypes.c_int32], kernel:CUkernel, dev:CUdevice) -> CUresult: ...
@dll.bind
def cuKernelSetCacheConfig(kernel:CUkernel, config:CUfunc_cache, dev:CUdevice) -> CUresult: ...
@dll.bind
def cuMemGetInfo_v2(free:POINTER(size_t), total:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuMemAlloc_v2(dptr:POINTER(CUdeviceptr), bytesize:size_t) -> CUresult: ...
@dll.bind
def cuMemAllocPitch_v2(dptr:POINTER(CUdeviceptr), pPitch:POINTER(size_t), WidthInBytes:size_t, Height:size_t, ElementSizeBytes:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemFree_v2(dptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuMemGetAddressRange_v2(pbase:POINTER(CUdeviceptr), psize:POINTER(size_t), dptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuMemAllocHost_v2(pp:POINTER(POINTER(None)), bytesize:size_t) -> CUresult: ...
@dll.bind
def cuMemFreeHost(p:POINTER(None)) -> CUresult: ...
@dll.bind
def cuMemHostAlloc(pp:POINTER(POINTER(None)), bytesize:size_t, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemHostGetDevicePointer_v2(pdptr:POINTER(CUdeviceptr), p:POINTER(None), Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemHostGetFlags(pFlags:POINTER(Annotated[int, ctypes.c_uint32]), p:POINTER(None)) -> CUresult: ...
@dll.bind
def cuMemAllocManaged(dptr:POINTER(CUdeviceptr), bytesize:size_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuDeviceGetByPCIBusId(dev:POINTER(CUdevice), pciBusId:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuDeviceGetPCIBusId(pciBusId:POINTER(Annotated[bytes, ctypes.c_char]), len:Annotated[int, ctypes.c_int32], dev:CUdevice) -> CUresult: ...
@dll.bind
def cuIpcGetEventHandle(pHandle:POINTER(CUipcEventHandle), event:CUevent) -> CUresult: ...
@dll.bind
def cuIpcOpenEventHandle(phEvent:POINTER(CUevent), handle:CUipcEventHandle) -> CUresult: ...
@dll.bind
def cuIpcGetMemHandle(pHandle:POINTER(CUipcMemHandle), dptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuIpcOpenMemHandle_v2(pdptr:POINTER(CUdeviceptr), handle:CUipcMemHandle, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuIpcCloseMemHandle(dptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuMemHostRegister_v2(p:POINTER(None), bytesize:size_t, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemHostUnregister(p:POINTER(None)) -> CUresult: ...
@dll.bind
def cuMemcpy_ptds(dst:CUdeviceptr, src:CUdeviceptr, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyPeer_ptds(dstDevice:CUdeviceptr, dstContext:CUcontext, srcDevice:CUdeviceptr, srcContext:CUcontext, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyHtoD_v2_ptds(dstDevice:CUdeviceptr, srcHost:POINTER(None), ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyDtoH_v2_ptds(dstHost:POINTER(None), srcDevice:CUdeviceptr, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyDtoD_v2_ptds(dstDevice:CUdeviceptr, srcDevice:CUdeviceptr, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyDtoA_v2_ptds(dstArray:CUarray, dstOffset:size_t, srcDevice:CUdeviceptr, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyAtoD_v2_ptds(dstDevice:CUdeviceptr, srcArray:CUarray, srcOffset:size_t, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyHtoA_v2_ptds(dstArray:CUarray, dstOffset:size_t, srcHost:POINTER(None), ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyAtoH_v2_ptds(dstHost:POINTER(None), srcArray:CUarray, srcOffset:size_t, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyAtoA_v2_ptds(dstArray:CUarray, dstOffset:size_t, srcArray:CUarray, srcOffset:size_t, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpy2D_v2_ptds(pCopy:POINTER(CUDA_MEMCPY2D)) -> CUresult: ...
@dll.bind
def cuMemcpy2DUnaligned_v2_ptds(pCopy:POINTER(CUDA_MEMCPY2D)) -> CUresult: ...
@dll.bind
def cuMemcpy3D_v2_ptds(pCopy:POINTER(CUDA_MEMCPY3D)) -> CUresult: ...
@dll.bind
def cuMemcpy3DPeer_ptds(pCopy:POINTER(CUDA_MEMCPY3D_PEER)) -> CUresult: ...
@dll.bind
def cuMemcpyAsync_ptsz(dst:CUdeviceptr, src:CUdeviceptr, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyPeerAsync_ptsz(dstDevice:CUdeviceptr, dstContext:CUcontext, srcDevice:CUdeviceptr, srcContext:CUcontext, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyHtoDAsync_v2_ptsz(dstDevice:CUdeviceptr, srcHost:POINTER(None), ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyDtoHAsync_v2_ptsz(dstHost:POINTER(None), srcDevice:CUdeviceptr, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyDtoDAsync_v2_ptsz(dstDevice:CUdeviceptr, srcDevice:CUdeviceptr, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyHtoAAsync_v2_ptsz(dstArray:CUarray, dstOffset:size_t, srcHost:POINTER(None), ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyAtoHAsync_v2_ptsz(dstHost:POINTER(None), srcArray:CUarray, srcOffset:size_t, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy2DAsync_v2_ptsz(pCopy:POINTER(CUDA_MEMCPY2D), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy3DAsync_v2_ptsz(pCopy:POINTER(CUDA_MEMCPY3D), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy3DPeerAsync_ptsz(pCopy:POINTER(CUDA_MEMCPY3D_PEER), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD8_v2_ptds(dstDevice:CUdeviceptr, uc:Annotated[int, ctypes.c_ubyte], N:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD16_v2_ptds(dstDevice:CUdeviceptr, us:Annotated[int, ctypes.c_uint16], N:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD32_v2_ptds(dstDevice:CUdeviceptr, ui:Annotated[int, ctypes.c_uint32], N:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD2D8_v2_ptds(dstDevice:CUdeviceptr, dstPitch:size_t, uc:Annotated[int, ctypes.c_ubyte], Width:size_t, Height:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD2D16_v2_ptds(dstDevice:CUdeviceptr, dstPitch:size_t, us:Annotated[int, ctypes.c_uint16], Width:size_t, Height:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD2D32_v2_ptds(dstDevice:CUdeviceptr, dstPitch:size_t, ui:Annotated[int, ctypes.c_uint32], Width:size_t, Height:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD8Async_ptsz(dstDevice:CUdeviceptr, uc:Annotated[int, ctypes.c_ubyte], N:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD16Async_ptsz(dstDevice:CUdeviceptr, us:Annotated[int, ctypes.c_uint16], N:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD32Async_ptsz(dstDevice:CUdeviceptr, ui:Annotated[int, ctypes.c_uint32], N:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD2D8Async_ptsz(dstDevice:CUdeviceptr, dstPitch:size_t, uc:Annotated[int, ctypes.c_ubyte], Width:size_t, Height:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD2D16Async_ptsz(dstDevice:CUdeviceptr, dstPitch:size_t, us:Annotated[int, ctypes.c_uint16], Width:size_t, Height:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD2D32Async_ptsz(dstDevice:CUdeviceptr, dstPitch:size_t, ui:Annotated[int, ctypes.c_uint32], Width:size_t, Height:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuArrayCreate_v2(pHandle:POINTER(CUarray), pAllocateArray:POINTER(CUDA_ARRAY_DESCRIPTOR)) -> CUresult: ...
@dll.bind
def cuArrayGetDescriptor_v2(pArrayDescriptor:POINTER(CUDA_ARRAY_DESCRIPTOR), hArray:CUarray) -> CUresult: ...
@dll.bind
def cuArrayGetSparseProperties(sparseProperties:POINTER(CUDA_ARRAY_SPARSE_PROPERTIES), array:CUarray) -> CUresult: ...
@dll.bind
def cuMipmappedArrayGetSparseProperties(sparseProperties:POINTER(CUDA_ARRAY_SPARSE_PROPERTIES), mipmap:CUmipmappedArray) -> CUresult: ...
@dll.bind
def cuArrayGetMemoryRequirements(memoryRequirements:POINTER(CUDA_ARRAY_MEMORY_REQUIREMENTS), array:CUarray, device:CUdevice) -> CUresult: ...
@dll.bind
def cuMipmappedArrayGetMemoryRequirements(memoryRequirements:POINTER(CUDA_ARRAY_MEMORY_REQUIREMENTS), mipmap:CUmipmappedArray, device:CUdevice) -> CUresult: ...
@dll.bind
def cuArrayGetPlane(pPlaneArray:POINTER(CUarray), hArray:CUarray, planeIdx:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuArrayDestroy(hArray:CUarray) -> CUresult: ...
@dll.bind
def cuArray3DCreate_v2(pHandle:POINTER(CUarray), pAllocateArray:POINTER(CUDA_ARRAY3D_DESCRIPTOR)) -> CUresult: ...
@dll.bind
def cuArray3DGetDescriptor_v2(pArrayDescriptor:POINTER(CUDA_ARRAY3D_DESCRIPTOR), hArray:CUarray) -> CUresult: ...
@dll.bind
def cuMipmappedArrayCreate(pHandle:POINTER(CUmipmappedArray), pMipmappedArrayDesc:POINTER(CUDA_ARRAY3D_DESCRIPTOR), numMipmapLevels:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMipmappedArrayGetLevel(pLevelArray:POINTER(CUarray), hMipmappedArray:CUmipmappedArray, level:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMipmappedArrayDestroy(hMipmappedArray:CUmipmappedArray) -> CUresult: ...
@dll.bind
def cuMemGetHandleForAddressRange(handle:POINTER(None), dptr:CUdeviceptr, size:size_t, handleType:CUmemRangeHandleType, flags:Annotated[int, ctypes.c_uint64]) -> CUresult: ...
@dll.bind
def cuMemAddressReserve(ptr:POINTER(CUdeviceptr), size:size_t, alignment:size_t, addr:CUdeviceptr, flags:Annotated[int, ctypes.c_uint64]) -> CUresult: ...
@dll.bind
def cuMemAddressFree(ptr:CUdeviceptr, size:size_t) -> CUresult: ...
@dll.bind
def cuMemCreate(handle:POINTER(CUmemGenericAllocationHandle), size:size_t, prop:POINTER(CUmemAllocationProp), flags:Annotated[int, ctypes.c_uint64]) -> CUresult: ...
@dll.bind
def cuMemRelease(handle:CUmemGenericAllocationHandle) -> CUresult: ...
@dll.bind
def cuMemMap(ptr:CUdeviceptr, size:size_t, offset:size_t, handle:CUmemGenericAllocationHandle, flags:Annotated[int, ctypes.c_uint64]) -> CUresult: ...
@dll.bind
def cuMemMapArrayAsync_ptsz(mapInfoList:POINTER(CUarrayMapInfo), count:Annotated[int, ctypes.c_uint32], hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemUnmap(ptr:CUdeviceptr, size:size_t) -> CUresult: ...
@dll.bind
def cuMemSetAccess(ptr:CUdeviceptr, size:size_t, desc:POINTER(CUmemAccessDesc), count:size_t) -> CUresult: ...
@dll.bind
def cuMemGetAccess(flags:POINTER(Annotated[int, ctypes.c_uint64]), location:POINTER(CUmemLocation), ptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuMemExportToShareableHandle(shareableHandle:POINTER(None), handle:CUmemGenericAllocationHandle, handleType:CUmemAllocationHandleType, flags:Annotated[int, ctypes.c_uint64]) -> CUresult: ...
@dll.bind
def cuMemImportFromShareableHandle(handle:POINTER(CUmemGenericAllocationHandle), osHandle:POINTER(None), shHandleType:CUmemAllocationHandleType) -> CUresult: ...
@dll.bind
def cuMemGetAllocationGranularity(granularity:POINTER(size_t), prop:POINTER(CUmemAllocationProp), option:CUmemAllocationGranularity_flags) -> CUresult: ...
@dll.bind
def cuMemGetAllocationPropertiesFromHandle(prop:POINTER(CUmemAllocationProp), handle:CUmemGenericAllocationHandle) -> CUresult: ...
@dll.bind
def cuMemRetainAllocationHandle(handle:POINTER(CUmemGenericAllocationHandle), addr:POINTER(None)) -> CUresult: ...
@dll.bind
def cuMemFreeAsync_ptsz(dptr:CUdeviceptr, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemAllocAsync_ptsz(dptr:POINTER(CUdeviceptr), bytesize:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemPoolTrimTo(pool:CUmemoryPool, minBytesToKeep:size_t) -> CUresult: ...
@dll.bind
def cuMemPoolSetAttribute(pool:CUmemoryPool, attr:CUmemPool_attribute, value:POINTER(None)) -> CUresult: ...
@dll.bind
def cuMemPoolGetAttribute(pool:CUmemoryPool, attr:CUmemPool_attribute, value:POINTER(None)) -> CUresult: ...
@dll.bind
def cuMemPoolSetAccess(pool:CUmemoryPool, map:POINTER(CUmemAccessDesc), count:size_t) -> CUresult: ...
@dll.bind
def cuMemPoolGetAccess(flags:POINTER(CUmemAccess_flags), memPool:CUmemoryPool, location:POINTER(CUmemLocation)) -> CUresult: ...
@dll.bind
def cuMemPoolCreate(pool:POINTER(CUmemoryPool), poolProps:POINTER(CUmemPoolProps)) -> CUresult: ...
@dll.bind
def cuMemPoolDestroy(pool:CUmemoryPool) -> CUresult: ...
@dll.bind
def cuMemAllocFromPoolAsync_ptsz(dptr:POINTER(CUdeviceptr), bytesize:size_t, pool:CUmemoryPool, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemPoolExportToShareableHandle(handle_out:POINTER(None), pool:CUmemoryPool, handleType:CUmemAllocationHandleType, flags:Annotated[int, ctypes.c_uint64]) -> CUresult: ...
@dll.bind
def cuMemPoolImportFromShareableHandle(pool_out:POINTER(CUmemoryPool), handle:POINTER(None), handleType:CUmemAllocationHandleType, flags:Annotated[int, ctypes.c_uint64]) -> CUresult: ...
@dll.bind
def cuMemPoolExportPointer(shareData_out:POINTER(CUmemPoolPtrExportData), ptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuMemPoolImportPointer(ptr_out:POINTER(CUdeviceptr), pool:CUmemoryPool, shareData:POINTER(CUmemPoolPtrExportData)) -> CUresult: ...
@dll.bind
def cuPointerGetAttribute(data:POINTER(None), attribute:CUpointer_attribute, ptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuMemPrefetchAsync_ptsz(devPtr:CUdeviceptr, count:size_t, dstDevice:CUdevice, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemAdvise(devPtr:CUdeviceptr, count:size_t, advice:CUmem_advise, device:CUdevice) -> CUresult: ...
@dll.bind
def cuMemRangeGetAttribute(data:POINTER(None), dataSize:size_t, attribute:CUmem_range_attribute, devPtr:CUdeviceptr, count:size_t) -> CUresult: ...
@dll.bind
def cuMemRangeGetAttributes(data:POINTER(POINTER(None)), dataSizes:POINTER(size_t), attributes:POINTER(CUmem_range_attribute), numAttributes:size_t, devPtr:CUdeviceptr, count:size_t) -> CUresult: ...
@dll.bind
def cuPointerSetAttribute(value:POINTER(None), attribute:CUpointer_attribute, ptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuPointerGetAttributes(numAttributes:Annotated[int, ctypes.c_uint32], attributes:POINTER(CUpointer_attribute), data:POINTER(POINTER(None)), ptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuStreamCreate(phStream:POINTER(CUstream), Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamCreateWithPriority(phStream:POINTER(CUstream), flags:Annotated[int, ctypes.c_uint32], priority:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuStreamGetPriority_ptsz(hStream:CUstream, priority:POINTER(Annotated[int, ctypes.c_int32])) -> CUresult: ...
@dll.bind
def cuStreamGetFlags_ptsz(hStream:CUstream, flags:POINTER(Annotated[int, ctypes.c_uint32])) -> CUresult: ...
@dll.bind
def cuStreamGetId_ptsz(hStream:CUstream, streamId:POINTER(Annotated[int, ctypes.c_uint64])) -> CUresult: ...
@dll.bind
def cuStreamGetCtx_ptsz(hStream:CUstream, pctx:POINTER(CUcontext)) -> CUresult: ...
@dll.bind
def cuStreamWaitEvent_ptsz(hStream:CUstream, hEvent:CUevent, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamAddCallback_ptsz(hStream:CUstream, callback:CUstreamCallback, userData:POINTER(None), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamBeginCapture_v2_ptsz(hStream:CUstream, mode:CUstreamCaptureMode) -> CUresult: ...
@dll.bind
def cuThreadExchangeStreamCaptureMode(mode:POINTER(CUstreamCaptureMode)) -> CUresult: ...
@dll.bind
def cuStreamEndCapture_ptsz(hStream:CUstream, phGraph:POINTER(CUgraph)) -> CUresult: ...
@dll.bind
def cuStreamIsCapturing_ptsz(hStream:CUstream, captureStatus:POINTER(CUstreamCaptureStatus)) -> CUresult: ...
@dll.bind
def cuStreamGetCaptureInfo_v2_ptsz(hStream:CUstream, captureStatus_out:POINTER(CUstreamCaptureStatus), id_out:POINTER(cuuint64_t), graph_out:POINTER(CUgraph), dependencies_out:POINTER(POINTER(CUgraphNode)), numDependencies_out:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuStreamUpdateCaptureDependencies_ptsz(hStream:CUstream, dependencies:POINTER(CUgraphNode), numDependencies:size_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamAttachMemAsync_ptsz(hStream:CUstream, dptr:CUdeviceptr, length:size_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamQuery_ptsz(hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamSynchronize_ptsz(hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamDestroy_v2(hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamCopyAttributes_ptsz(dst:CUstream, src:CUstream) -> CUresult: ...
@dll.bind
def cuStreamGetAttribute_ptsz(hStream:CUstream, attr:CUstreamAttrID, value_out:POINTER(CUstreamAttrValue)) -> CUresult: ...
@dll.bind
def cuStreamSetAttribute_ptsz(hStream:CUstream, attr:CUstreamAttrID, value:POINTER(CUstreamAttrValue)) -> CUresult: ...
@dll.bind
def cuEventCreate(phEvent:POINTER(CUevent), Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuEventRecord_ptsz(hEvent:CUevent, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuEventRecordWithFlags_ptsz(hEvent:CUevent, hStream:CUstream, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuEventQuery(hEvent:CUevent) -> CUresult: ...
@dll.bind
def cuEventSynchronize(hEvent:CUevent) -> CUresult: ...
@dll.bind
def cuEventDestroy_v2(hEvent:CUevent) -> CUresult: ...
@dll.bind
def cuEventElapsedTime(pMilliseconds:POINTER(Annotated[float, ctypes.c_float]), hStart:CUevent, hEnd:CUevent) -> CUresult: ...
@dll.bind
def cuImportExternalMemory(extMem_out:POINTER(CUexternalMemory), memHandleDesc:POINTER(CUDA_EXTERNAL_MEMORY_HANDLE_DESC)) -> CUresult: ...
@dll.bind
def cuExternalMemoryGetMappedBuffer(devPtr:POINTER(CUdeviceptr), extMem:CUexternalMemory, bufferDesc:POINTER(CUDA_EXTERNAL_MEMORY_BUFFER_DESC)) -> CUresult: ...
@dll.bind
def cuExternalMemoryGetMappedMipmappedArray(mipmap:POINTER(CUmipmappedArray), extMem:CUexternalMemory, mipmapDesc:POINTER(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC)) -> CUresult: ...
@dll.bind
def cuDestroyExternalMemory(extMem:CUexternalMemory) -> CUresult: ...
@dll.bind
def cuImportExternalSemaphore(extSem_out:POINTER(CUexternalSemaphore), semHandleDesc:POINTER(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC)) -> CUresult: ...
@dll.bind
def cuSignalExternalSemaphoresAsync_ptsz(extSemArray:POINTER(CUexternalSemaphore), paramsArray:POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), numExtSems:Annotated[int, ctypes.c_uint32], stream:CUstream) -> CUresult: ...
@dll.bind
def cuWaitExternalSemaphoresAsync_ptsz(extSemArray:POINTER(CUexternalSemaphore), paramsArray:POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS), numExtSems:Annotated[int, ctypes.c_uint32], stream:CUstream) -> CUresult: ...
@dll.bind
def cuDestroyExternalSemaphore(extSem:CUexternalSemaphore) -> CUresult: ...
@dll.bind
def cuStreamWaitValue32_v2_ptsz(stream:CUstream, addr:CUdeviceptr, value:cuuint32_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWaitValue64_v2_ptsz(stream:CUstream, addr:CUdeviceptr, value:cuuint64_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWriteValue32_v2_ptsz(stream:CUstream, addr:CUdeviceptr, value:cuuint32_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWriteValue64_v2_ptsz(stream:CUstream, addr:CUdeviceptr, value:cuuint64_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamBatchMemOp_v2_ptsz(stream:CUstream, count:Annotated[int, ctypes.c_uint32], paramArray:POINTER(CUstreamBatchMemOpParams), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuFuncGetAttribute(pi:POINTER(Annotated[int, ctypes.c_int32]), attrib:CUfunction_attribute, hfunc:CUfunction) -> CUresult: ...
@dll.bind
def cuFuncSetAttribute(hfunc:CUfunction, attrib:CUfunction_attribute, value:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuFuncSetCacheConfig(hfunc:CUfunction, config:CUfunc_cache) -> CUresult: ...
@dll.bind
def cuFuncSetSharedMemConfig(hfunc:CUfunction, config:CUsharedconfig) -> CUresult: ...
@dll.bind
def cuFuncGetModule(hmod:POINTER(CUmodule), hfunc:CUfunction) -> CUresult: ...
@dll.bind
def cuLaunchKernel_ptsz(f:CUfunction, gridDimX:Annotated[int, ctypes.c_uint32], gridDimY:Annotated[int, ctypes.c_uint32], gridDimZ:Annotated[int, ctypes.c_uint32], blockDimX:Annotated[int, ctypes.c_uint32], blockDimY:Annotated[int, ctypes.c_uint32], blockDimZ:Annotated[int, ctypes.c_uint32], sharedMemBytes:Annotated[int, ctypes.c_uint32], hStream:CUstream, kernelParams:POINTER(POINTER(None)), extra:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuLaunchKernelEx_ptsz(config:POINTER(CUlaunchConfig), f:CUfunction, kernelParams:POINTER(POINTER(None)), extra:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuLaunchCooperativeKernel_ptsz(f:CUfunction, gridDimX:Annotated[int, ctypes.c_uint32], gridDimY:Annotated[int, ctypes.c_uint32], gridDimZ:Annotated[int, ctypes.c_uint32], blockDimX:Annotated[int, ctypes.c_uint32], blockDimY:Annotated[int, ctypes.c_uint32], blockDimZ:Annotated[int, ctypes.c_uint32], sharedMemBytes:Annotated[int, ctypes.c_uint32], hStream:CUstream, kernelParams:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuLaunchCooperativeKernelMultiDevice(launchParamsList:POINTER(CUDA_LAUNCH_PARAMS), numDevices:Annotated[int, ctypes.c_uint32], flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuLaunchHostFunc_ptsz(hStream:CUstream, fn:CUhostFn, userData:POINTER(None)) -> CUresult: ...
@dll.bind
def cuFuncSetBlockShape(hfunc:CUfunction, x:Annotated[int, ctypes.c_int32], y:Annotated[int, ctypes.c_int32], z:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuFuncSetSharedSize(hfunc:CUfunction, bytes:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuParamSetSize(hfunc:CUfunction, numbytes:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuParamSeti(hfunc:CUfunction, offset:Annotated[int, ctypes.c_int32], value:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuParamSetf(hfunc:CUfunction, offset:Annotated[int, ctypes.c_int32], value:Annotated[float, ctypes.c_float]) -> CUresult: ...
@dll.bind
def cuParamSetv(hfunc:CUfunction, offset:Annotated[int, ctypes.c_int32], ptr:POINTER(None), numbytes:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuLaunch(f:CUfunction) -> CUresult: ...
@dll.bind
def cuLaunchGrid(f:CUfunction, grid_width:Annotated[int, ctypes.c_int32], grid_height:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuLaunchGridAsync(f:CUfunction, grid_width:Annotated[int, ctypes.c_int32], grid_height:Annotated[int, ctypes.c_int32], hStream:CUstream) -> CUresult: ...
@dll.bind
def cuParamSetTexRef(hfunc:CUfunction, texunit:Annotated[int, ctypes.c_int32], hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuGraphCreate(phGraph:POINTER(CUgraph), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGraphAddKernelNode_v2(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, nodeParams:POINTER(CUDA_KERNEL_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphKernelNodeGetParams_v2(hNode:CUgraphNode, nodeParams:POINTER(CUDA_KERNEL_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphKernelNodeSetParams_v2(hNode:CUgraphNode, nodeParams:POINTER(CUDA_KERNEL_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphAddMemcpyNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, copyParams:POINTER(CUDA_MEMCPY3D), ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuGraphMemcpyNodeGetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_MEMCPY3D)) -> CUresult: ...
@dll.bind
def cuGraphMemcpyNodeSetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_MEMCPY3D)) -> CUresult: ...
@dll.bind
def cuGraphAddMemsetNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, memsetParams:POINTER(CUDA_MEMSET_NODE_PARAMS), ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuGraphMemsetNodeGetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_MEMSET_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphMemsetNodeSetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_MEMSET_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphAddHostNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, nodeParams:POINTER(CUDA_HOST_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphHostNodeGetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_HOST_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphHostNodeSetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_HOST_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphAddChildGraphNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, childGraph:CUgraph) -> CUresult: ...
@dll.bind
def cuGraphChildGraphNodeGetGraph(hNode:CUgraphNode, phGraph:POINTER(CUgraph)) -> CUresult: ...
@dll.bind
def cuGraphAddEmptyNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t) -> CUresult: ...
@dll.bind
def cuGraphAddEventRecordNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, event:CUevent) -> CUresult: ...
@dll.bind
def cuGraphEventRecordNodeGetEvent(hNode:CUgraphNode, event_out:POINTER(CUevent)) -> CUresult: ...
@dll.bind
def cuGraphEventRecordNodeSetEvent(hNode:CUgraphNode, event:CUevent) -> CUresult: ...
@dll.bind
def cuGraphAddEventWaitNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, event:CUevent) -> CUresult: ...
@dll.bind
def cuGraphEventWaitNodeGetEvent(hNode:CUgraphNode, event_out:POINTER(CUevent)) -> CUresult: ...
@dll.bind
def cuGraphEventWaitNodeSetEvent(hNode:CUgraphNode, event:CUevent) -> CUresult: ...
@dll.bind
def cuGraphAddExternalSemaphoresSignalNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, nodeParams:POINTER(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExternalSemaphoresSignalNodeGetParams(hNode:CUgraphNode, params_out:POINTER(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExternalSemaphoresSignalNodeSetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphAddExternalSemaphoresWaitNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, nodeParams:POINTER(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExternalSemaphoresWaitNodeGetParams(hNode:CUgraphNode, params_out:POINTER(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExternalSemaphoresWaitNodeSetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphAddBatchMemOpNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, nodeParams:POINTER(CUDA_BATCH_MEM_OP_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphBatchMemOpNodeGetParams(hNode:CUgraphNode, nodeParams_out:POINTER(CUDA_BATCH_MEM_OP_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphBatchMemOpNodeSetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_BATCH_MEM_OP_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExecBatchMemOpNodeSetParams(hGraphExec:CUgraphExec, hNode:CUgraphNode, nodeParams:POINTER(CUDA_BATCH_MEM_OP_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphAddMemAllocNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, nodeParams:POINTER(CUDA_MEM_ALLOC_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphMemAllocNodeGetParams(hNode:CUgraphNode, params_out:POINTER(CUDA_MEM_ALLOC_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphAddMemFreeNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, dptr:CUdeviceptr) -> CUresult: ...
@dll.bind
def cuGraphMemFreeNodeGetParams(hNode:CUgraphNode, dptr_out:POINTER(CUdeviceptr)) -> CUresult: ...
@dll.bind
def cuDeviceGraphMemTrim(device:CUdevice) -> CUresult: ...
@dll.bind
def cuDeviceGetGraphMemAttribute(device:CUdevice, attr:CUgraphMem_attribute, value:POINTER(None)) -> CUresult: ...
@dll.bind
def cuDeviceSetGraphMemAttribute(device:CUdevice, attr:CUgraphMem_attribute, value:POINTER(None)) -> CUresult: ...
@dll.bind
def cuGraphClone(phGraphClone:POINTER(CUgraph), originalGraph:CUgraph) -> CUresult: ...
@dll.bind
def cuGraphNodeFindInClone(phNode:POINTER(CUgraphNode), hOriginalNode:CUgraphNode, hClonedGraph:CUgraph) -> CUresult: ...
@dll.bind
def cuGraphNodeGetType(hNode:CUgraphNode, type:POINTER(CUgraphNodeType)) -> CUresult: ...
@dll.bind
def cuGraphGetNodes(hGraph:CUgraph, nodes:POINTER(CUgraphNode), numNodes:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuGraphGetRootNodes(hGraph:CUgraph, rootNodes:POINTER(CUgraphNode), numRootNodes:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuGraphGetEdges(hGraph:CUgraph, _from:POINTER(CUgraphNode), to:POINTER(CUgraphNode), numEdges:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuGraphNodeGetDependencies(hNode:CUgraphNode, dependencies:POINTER(CUgraphNode), numDependencies:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuGraphNodeGetDependentNodes(hNode:CUgraphNode, dependentNodes:POINTER(CUgraphNode), numDependentNodes:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuGraphAddDependencies(hGraph:CUgraph, _from:POINTER(CUgraphNode), to:POINTER(CUgraphNode), numDependencies:size_t) -> CUresult: ...
@dll.bind
def cuGraphRemoveDependencies(hGraph:CUgraph, _from:POINTER(CUgraphNode), to:POINTER(CUgraphNode), numDependencies:size_t) -> CUresult: ...
@dll.bind
def cuGraphDestroyNode(hNode:CUgraphNode) -> CUresult: ...
@dll.bind
def cuGraphInstantiateWithFlags(phGraphExec:POINTER(CUgraphExec), hGraph:CUgraph, flags:Annotated[int, ctypes.c_uint64]) -> CUresult: ...
@dll.bind
def cuGraphInstantiateWithParams_ptsz(phGraphExec:POINTER(CUgraphExec), hGraph:CUgraph, instantiateParams:POINTER(CUDA_GRAPH_INSTANTIATE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExecGetFlags(hGraphExec:CUgraphExec, flags:POINTER(cuuint64_t)) -> CUresult: ...
@dll.bind
def cuGraphExecKernelNodeSetParams_v2(hGraphExec:CUgraphExec, hNode:CUgraphNode, nodeParams:POINTER(CUDA_KERNEL_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExecMemcpyNodeSetParams(hGraphExec:CUgraphExec, hNode:CUgraphNode, copyParams:POINTER(CUDA_MEMCPY3D), ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuGraphExecMemsetNodeSetParams(hGraphExec:CUgraphExec, hNode:CUgraphNode, memsetParams:POINTER(CUDA_MEMSET_NODE_PARAMS), ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuGraphExecHostNodeSetParams(hGraphExec:CUgraphExec, hNode:CUgraphNode, nodeParams:POINTER(CUDA_HOST_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExecChildGraphNodeSetParams(hGraphExec:CUgraphExec, hNode:CUgraphNode, childGraph:CUgraph) -> CUresult: ...
@dll.bind
def cuGraphExecEventRecordNodeSetEvent(hGraphExec:CUgraphExec, hNode:CUgraphNode, event:CUevent) -> CUresult: ...
@dll.bind
def cuGraphExecEventWaitNodeSetEvent(hGraphExec:CUgraphExec, hNode:CUgraphNode, event:CUevent) -> CUresult: ...
@dll.bind
def cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec:CUgraphExec, hNode:CUgraphNode, nodeParams:POINTER(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec:CUgraphExec, hNode:CUgraphNode, nodeParams:POINTER(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphNodeSetEnabled(hGraphExec:CUgraphExec, hNode:CUgraphNode, isEnabled:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGraphNodeGetEnabled(hGraphExec:CUgraphExec, hNode:CUgraphNode, isEnabled:POINTER(Annotated[int, ctypes.c_uint32])) -> CUresult: ...
@dll.bind
def cuGraphUpload_ptsz(hGraphExec:CUgraphExec, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuGraphLaunch_ptsz(hGraphExec:CUgraphExec, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuGraphExecDestroy(hGraphExec:CUgraphExec) -> CUresult: ...
@dll.bind
def cuGraphDestroy(hGraph:CUgraph) -> CUresult: ...
@dll.bind
def cuGraphExecUpdate_v2(hGraphExec:CUgraphExec, hGraph:CUgraph, resultInfo:POINTER(CUgraphExecUpdateResultInfo)) -> CUresult: ...
@dll.bind
def cuGraphKernelNodeCopyAttributes(dst:CUgraphNode, src:CUgraphNode) -> CUresult: ...
@dll.bind
def cuGraphKernelNodeGetAttribute(hNode:CUgraphNode, attr:CUkernelNodeAttrID, value_out:POINTER(CUkernelNodeAttrValue)) -> CUresult: ...
@dll.bind
def cuGraphKernelNodeSetAttribute(hNode:CUgraphNode, attr:CUkernelNodeAttrID, value:POINTER(CUkernelNodeAttrValue)) -> CUresult: ...
@dll.bind
def cuGraphDebugDotPrint(hGraph:CUgraph, path:POINTER(Annotated[bytes, ctypes.c_char]), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuUserObjectCreate(object_out:POINTER(CUuserObject), ptr:POINTER(None), destroy:CUhostFn, initialRefcount:Annotated[int, ctypes.c_uint32], flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuUserObjectRetain(object:CUuserObject, count:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuUserObjectRelease(object:CUuserObject, count:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGraphRetainUserObject(graph:CUgraph, object:CUuserObject, count:Annotated[int, ctypes.c_uint32], flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGraphReleaseUserObject(graph:CUgraph, object:CUuserObject, count:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks:POINTER(Annotated[int, ctypes.c_int32]), func:CUfunction, blockSize:Annotated[int, ctypes.c_int32], dynamicSMemSize:size_t) -> CUresult: ...
@dll.bind
def cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks:POINTER(Annotated[int, ctypes.c_int32]), func:CUfunction, blockSize:Annotated[int, ctypes.c_int32], dynamicSMemSize:size_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuOccupancyMaxPotentialBlockSize(minGridSize:POINTER(Annotated[int, ctypes.c_int32]), blockSize:POINTER(Annotated[int, ctypes.c_int32]), func:CUfunction, blockSizeToDynamicSMemSize:CUoccupancyB2DSize, dynamicSMemSize:size_t, blockSizeLimit:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize:POINTER(Annotated[int, ctypes.c_int32]), blockSize:POINTER(Annotated[int, ctypes.c_int32]), func:CUfunction, blockSizeToDynamicSMemSize:CUoccupancyB2DSize, dynamicSMemSize:size_t, blockSizeLimit:Annotated[int, ctypes.c_int32], flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize:POINTER(size_t), func:CUfunction, numBlocks:Annotated[int, ctypes.c_int32], blockSize:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuOccupancyMaxPotentialClusterSize(clusterSize:POINTER(Annotated[int, ctypes.c_int32]), func:CUfunction, config:POINTER(CUlaunchConfig)) -> CUresult: ...
@dll.bind
def cuOccupancyMaxActiveClusters(numClusters:POINTER(Annotated[int, ctypes.c_int32]), func:CUfunction, config:POINTER(CUlaunchConfig)) -> CUresult: ...
@dll.bind
def cuTexRefSetArray(hTexRef:CUtexref, hArray:CUarray, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuTexRefSetMipmappedArray(hTexRef:CUtexref, hMipmappedArray:CUmipmappedArray, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuTexRefSetAddress_v2(ByteOffset:POINTER(size_t), hTexRef:CUtexref, dptr:CUdeviceptr, bytes:size_t) -> CUresult: ...
@dll.bind
def cuTexRefSetAddress2D_v3(hTexRef:CUtexref, desc:POINTER(CUDA_ARRAY_DESCRIPTOR), dptr:CUdeviceptr, Pitch:size_t) -> CUresult: ...
@dll.bind
def cuTexRefSetFormat(hTexRef:CUtexref, fmt:CUarray_format, NumPackedComponents:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuTexRefSetAddressMode(hTexRef:CUtexref, dim:Annotated[int, ctypes.c_int32], am:CUaddress_mode) -> CUresult: ...
@dll.bind
def cuTexRefSetFilterMode(hTexRef:CUtexref, fm:CUfilter_mode) -> CUresult: ...
@dll.bind
def cuTexRefSetMipmapFilterMode(hTexRef:CUtexref, fm:CUfilter_mode) -> CUresult: ...
@dll.bind
def cuTexRefSetMipmapLevelBias(hTexRef:CUtexref, bias:Annotated[float, ctypes.c_float]) -> CUresult: ...
@dll.bind
def cuTexRefSetMipmapLevelClamp(hTexRef:CUtexref, minMipmapLevelClamp:Annotated[float, ctypes.c_float], maxMipmapLevelClamp:Annotated[float, ctypes.c_float]) -> CUresult: ...
@dll.bind
def cuTexRefSetMaxAnisotropy(hTexRef:CUtexref, maxAniso:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuTexRefSetBorderColor(hTexRef:CUtexref, pBorderColor:POINTER(Annotated[float, ctypes.c_float])) -> CUresult: ...
@dll.bind
def cuTexRefSetFlags(hTexRef:CUtexref, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuTexRefGetAddress_v2(pdptr:POINTER(CUdeviceptr), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetArray(phArray:POINTER(CUarray), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetMipmappedArray(phMipmappedArray:POINTER(CUmipmappedArray), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetAddressMode(pam:POINTER(CUaddress_mode), hTexRef:CUtexref, dim:Annotated[int, ctypes.c_int32]) -> CUresult: ...
@dll.bind
def cuTexRefGetFilterMode(pfm:POINTER(CUfilter_mode), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetFormat(pFormat:POINTER(CUarray_format), pNumChannels:POINTER(Annotated[int, ctypes.c_int32]), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetMipmapFilterMode(pfm:POINTER(CUfilter_mode), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetMipmapLevelBias(pbias:POINTER(Annotated[float, ctypes.c_float]), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp:POINTER(Annotated[float, ctypes.c_float]), pmaxMipmapLevelClamp:POINTER(Annotated[float, ctypes.c_float]), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetMaxAnisotropy(pmaxAniso:POINTER(Annotated[int, ctypes.c_int32]), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetBorderColor(pBorderColor:POINTER(Annotated[float, ctypes.c_float]), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefGetFlags(pFlags:POINTER(Annotated[int, ctypes.c_uint32]), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuTexRefCreate(pTexRef:POINTER(CUtexref)) -> CUresult: ...
@dll.bind
def cuTexRefDestroy(hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuSurfRefSetArray(hSurfRef:CUsurfref, hArray:CUarray, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuSurfRefGetArray(phArray:POINTER(CUarray), hSurfRef:CUsurfref) -> CUresult: ...
@dll.bind
def cuTexObjectCreate(pTexObject:POINTER(CUtexObject), pResDesc:POINTER(CUDA_RESOURCE_DESC), pTexDesc:POINTER(CUDA_TEXTURE_DESC), pResViewDesc:POINTER(CUDA_RESOURCE_VIEW_DESC)) -> CUresult: ...
@dll.bind
def cuTexObjectDestroy(texObject:CUtexObject) -> CUresult: ...
@dll.bind
def cuTexObjectGetResourceDesc(pResDesc:POINTER(CUDA_RESOURCE_DESC), texObject:CUtexObject) -> CUresult: ...
@dll.bind
def cuTexObjectGetTextureDesc(pTexDesc:POINTER(CUDA_TEXTURE_DESC), texObject:CUtexObject) -> CUresult: ...
@dll.bind
def cuTexObjectGetResourceViewDesc(pResViewDesc:POINTER(CUDA_RESOURCE_VIEW_DESC), texObject:CUtexObject) -> CUresult: ...
@dll.bind
def cuSurfObjectCreate(pSurfObject:POINTER(CUsurfObject), pResDesc:POINTER(CUDA_RESOURCE_DESC)) -> CUresult: ...
@dll.bind
def cuSurfObjectDestroy(surfObject:CUsurfObject) -> CUresult: ...
@dll.bind
def cuSurfObjectGetResourceDesc(pResDesc:POINTER(CUDA_RESOURCE_DESC), surfObject:CUsurfObject) -> CUresult: ...
@dll.bind
def cuTensorMapEncodeTiled(tensorMap:POINTER(CUtensorMap), tensorDataType:CUtensorMapDataType, tensorRank:cuuint32_t, globalAddress:POINTER(None), globalDim:POINTER(cuuint64_t), globalStrides:POINTER(cuuint64_t), boxDim:POINTER(cuuint32_t), elementStrides:POINTER(cuuint32_t), interleave:CUtensorMapInterleave, swizzle:CUtensorMapSwizzle, l2Promotion:CUtensorMapL2promotion, oobFill:CUtensorMapFloatOOBfill) -> CUresult: ...
@dll.bind
def cuTensorMapEncodeIm2col(tensorMap:POINTER(CUtensorMap), tensorDataType:CUtensorMapDataType, tensorRank:cuuint32_t, globalAddress:POINTER(None), globalDim:POINTER(cuuint64_t), globalStrides:POINTER(cuuint64_t), pixelBoxLowerCorner:POINTER(Annotated[int, ctypes.c_int32]), pixelBoxUpperCorner:POINTER(Annotated[int, ctypes.c_int32]), channelsPerPixel:cuuint32_t, pixelsPerColumn:cuuint32_t, elementStrides:POINTER(cuuint32_t), interleave:CUtensorMapInterleave, swizzle:CUtensorMapSwizzle, l2Promotion:CUtensorMapL2promotion, oobFill:CUtensorMapFloatOOBfill) -> CUresult: ...
@dll.bind
def cuTensorMapReplaceAddress(tensorMap:POINTER(CUtensorMap), globalAddress:POINTER(None)) -> CUresult: ...
@dll.bind
def cuDeviceCanAccessPeer(canAccessPeer:POINTER(Annotated[int, ctypes.c_int32]), dev:CUdevice, peerDev:CUdevice) -> CUresult: ...
@dll.bind
def cuCtxEnablePeerAccess(peerContext:CUcontext, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuCtxDisablePeerAccess(peerContext:CUcontext) -> CUresult: ...
@dll.bind
def cuDeviceGetP2PAttribute(value:POINTER(Annotated[int, ctypes.c_int32]), attrib:CUdevice_P2PAttribute, srcDevice:CUdevice, dstDevice:CUdevice) -> CUresult: ...
@dll.bind
def cuGraphicsUnregisterResource(resource:CUgraphicsResource) -> CUresult: ...
@dll.bind
def cuGraphicsSubResourceGetMappedArray(pArray:POINTER(CUarray), resource:CUgraphicsResource, arrayIndex:Annotated[int, ctypes.c_uint32], mipLevel:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray:POINTER(CUmipmappedArray), resource:CUgraphicsResource) -> CUresult: ...
@dll.bind
def cuGraphicsResourceGetMappedPointer_v2(pDevPtr:POINTER(CUdeviceptr), pSize:POINTER(size_t), resource:CUgraphicsResource) -> CUresult: ...
@dll.bind
def cuGraphicsResourceSetMapFlags_v2(resource:CUgraphicsResource, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGraphicsMapResources_ptsz(count:Annotated[int, ctypes.c_uint32], resources:POINTER(CUgraphicsResource), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuGraphicsUnmapResources_ptsz(count:Annotated[int, ctypes.c_uint32], resources:POINTER(CUgraphicsResource), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuGetProcAddress_v2(symbol:POINTER(Annotated[bytes, ctypes.c_char]), pfn:POINTER(POINTER(None)), cudaVersion:Annotated[int, ctypes.c_int32], flags:cuuint64_t, symbolStatus:POINTER(CUdriverProcAddressQueryResult)) -> CUresult: ...
@dll.bind
def cuGetExportTable(ppExportTable:POINTER(POINTER(None)), pExportTableId:POINTER(CUuuid)) -> CUresult: ...
@dll.bind
def cuMemHostRegister(p:POINTER(None), bytesize:size_t, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGraphicsResourceSetMapFlags(resource:CUgraphicsResource, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuLinkCreate(numOptions:Annotated[int, ctypes.c_uint32], options:POINTER(CUjit_option), optionValues:POINTER(POINTER(None)), stateOut:POINTER(CUlinkState)) -> CUresult: ...
@dll.bind
def cuLinkAddData(state:CUlinkState, type:CUjitInputType, data:POINTER(None), size:size_t, name:POINTER(Annotated[bytes, ctypes.c_char]), numOptions:Annotated[int, ctypes.c_uint32], options:POINTER(CUjit_option), optionValues:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuLinkAddFile(state:CUlinkState, type:CUjitInputType, path:POINTER(Annotated[bytes, ctypes.c_char]), numOptions:Annotated[int, ctypes.c_uint32], options:POINTER(CUjit_option), optionValues:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuTexRefSetAddress2D_v2(hTexRef:CUtexref, desc:POINTER(CUDA_ARRAY_DESCRIPTOR), dptr:CUdeviceptr, Pitch:size_t) -> CUresult: ...
CUdeviceptr_v1: TypeAlias = Annotated[int, ctypes.c_uint32]
@record
class struct_CUDA_MEMCPY2D_v1_st:
  SIZE = 96
  srcXInBytes: Annotated[Annotated[int, ctypes.c_uint32], 0]
  srcY: Annotated[Annotated[int, ctypes.c_uint32], 4]
  srcMemoryType: Annotated[CUmemorytype, 8]
  srcHost: Annotated[POINTER(None), 16]
  srcDevice: Annotated[CUdeviceptr_v1, 24]
  srcArray: Annotated[CUarray, 32]
  srcPitch: Annotated[Annotated[int, ctypes.c_uint32], 40]
  dstXInBytes: Annotated[Annotated[int, ctypes.c_uint32], 44]
  dstY: Annotated[Annotated[int, ctypes.c_uint32], 48]
  dstMemoryType: Annotated[CUmemorytype, 52]
  dstHost: Annotated[POINTER(None), 56]
  dstDevice: Annotated[CUdeviceptr_v1, 64]
  dstArray: Annotated[CUarray, 72]
  dstPitch: Annotated[Annotated[int, ctypes.c_uint32], 80]
  WidthInBytes: Annotated[Annotated[int, ctypes.c_uint32], 84]
  Height: Annotated[Annotated[int, ctypes.c_uint32], 88]
CUDA_MEMCPY2D_v1: TypeAlias = struct_CUDA_MEMCPY2D_v1_st
@record
class struct_CUDA_MEMCPY3D_v1_st:
  SIZE = 144
  srcXInBytes: Annotated[Annotated[int, ctypes.c_uint32], 0]
  srcY: Annotated[Annotated[int, ctypes.c_uint32], 4]
  srcZ: Annotated[Annotated[int, ctypes.c_uint32], 8]
  srcLOD: Annotated[Annotated[int, ctypes.c_uint32], 12]
  srcMemoryType: Annotated[CUmemorytype, 16]
  srcHost: Annotated[POINTER(None), 24]
  srcDevice: Annotated[CUdeviceptr_v1, 32]
  srcArray: Annotated[CUarray, 40]
  reserved0: Annotated[POINTER(None), 48]
  srcPitch: Annotated[Annotated[int, ctypes.c_uint32], 56]
  srcHeight: Annotated[Annotated[int, ctypes.c_uint32], 60]
  dstXInBytes: Annotated[Annotated[int, ctypes.c_uint32], 64]
  dstY: Annotated[Annotated[int, ctypes.c_uint32], 68]
  dstZ: Annotated[Annotated[int, ctypes.c_uint32], 72]
  dstLOD: Annotated[Annotated[int, ctypes.c_uint32], 76]
  dstMemoryType: Annotated[CUmemorytype, 80]
  dstHost: Annotated[POINTER(None), 88]
  dstDevice: Annotated[CUdeviceptr_v1, 96]
  dstArray: Annotated[CUarray, 104]
  reserved1: Annotated[POINTER(None), 112]
  dstPitch: Annotated[Annotated[int, ctypes.c_uint32], 120]
  dstHeight: Annotated[Annotated[int, ctypes.c_uint32], 124]
  WidthInBytes: Annotated[Annotated[int, ctypes.c_uint32], 128]
  Height: Annotated[Annotated[int, ctypes.c_uint32], 132]
  Depth: Annotated[Annotated[int, ctypes.c_uint32], 136]
CUDA_MEMCPY3D_v1: TypeAlias = struct_CUDA_MEMCPY3D_v1_st
@record
class struct_CUDA_ARRAY_DESCRIPTOR_v1_st:
  SIZE = 16
  Width: Annotated[Annotated[int, ctypes.c_uint32], 0]
  Height: Annotated[Annotated[int, ctypes.c_uint32], 4]
  Format: Annotated[CUarray_format, 8]
  NumChannels: Annotated[Annotated[int, ctypes.c_uint32], 12]
CUDA_ARRAY_DESCRIPTOR_v1: TypeAlias = struct_CUDA_ARRAY_DESCRIPTOR_v1_st
@record
class struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st:
  SIZE = 24
  Width: Annotated[Annotated[int, ctypes.c_uint32], 0]
  Height: Annotated[Annotated[int, ctypes.c_uint32], 4]
  Depth: Annotated[Annotated[int, ctypes.c_uint32], 8]
  Format: Annotated[CUarray_format, 12]
  NumChannels: Annotated[Annotated[int, ctypes.c_uint32], 16]
  Flags: Annotated[Annotated[int, ctypes.c_uint32], 20]
CUDA_ARRAY3D_DESCRIPTOR_v1: TypeAlias = struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st
@dll.bind
def cuDeviceTotalMem(bytes:POINTER(Annotated[int, ctypes.c_uint32]), dev:CUdevice) -> CUresult: ...
@dll.bind
def cuCtxCreate(pctx:POINTER(CUcontext), flags:Annotated[int, ctypes.c_uint32], dev:CUdevice) -> CUresult: ...
@dll.bind
def cuModuleGetGlobal(dptr:POINTER(CUdeviceptr_v1), bytes:POINTER(Annotated[int, ctypes.c_uint32]), hmod:CUmodule, name:POINTER(Annotated[bytes, ctypes.c_char])) -> CUresult: ...
@dll.bind
def cuMemGetInfo(free:POINTER(Annotated[int, ctypes.c_uint32]), total:POINTER(Annotated[int, ctypes.c_uint32])) -> CUresult: ...
@dll.bind
def cuMemAlloc(dptr:POINTER(CUdeviceptr_v1), bytesize:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemAllocPitch(dptr:POINTER(CUdeviceptr_v1), pPitch:POINTER(Annotated[int, ctypes.c_uint32]), WidthInBytes:Annotated[int, ctypes.c_uint32], Height:Annotated[int, ctypes.c_uint32], ElementSizeBytes:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemFree(dptr:CUdeviceptr_v1) -> CUresult: ...
@dll.bind
def cuMemGetAddressRange(pbase:POINTER(CUdeviceptr_v1), psize:POINTER(Annotated[int, ctypes.c_uint32]), dptr:CUdeviceptr_v1) -> CUresult: ...
@dll.bind
def cuMemAllocHost(pp:POINTER(POINTER(None)), bytesize:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemHostGetDevicePointer(pdptr:POINTER(CUdeviceptr_v1), p:POINTER(None), Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyHtoD(dstDevice:CUdeviceptr_v1, srcHost:POINTER(None), ByteCount:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyDtoH(dstHost:POINTER(None), srcDevice:CUdeviceptr_v1, ByteCount:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyDtoD(dstDevice:CUdeviceptr_v1, srcDevice:CUdeviceptr_v1, ByteCount:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyDtoA(dstArray:CUarray, dstOffset:Annotated[int, ctypes.c_uint32], srcDevice:CUdeviceptr_v1, ByteCount:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyAtoD(dstDevice:CUdeviceptr_v1, srcArray:CUarray, srcOffset:Annotated[int, ctypes.c_uint32], ByteCount:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyHtoA(dstArray:CUarray, dstOffset:Annotated[int, ctypes.c_uint32], srcHost:POINTER(None), ByteCount:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyAtoH(dstHost:POINTER(None), srcArray:CUarray, srcOffset:Annotated[int, ctypes.c_uint32], ByteCount:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyAtoA(dstArray:CUarray, dstOffset:Annotated[int, ctypes.c_uint32], srcArray:CUarray, srcOffset:Annotated[int, ctypes.c_uint32], ByteCount:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyHtoAAsync(dstArray:CUarray, dstOffset:Annotated[int, ctypes.c_uint32], srcHost:POINTER(None), ByteCount:Annotated[int, ctypes.c_uint32], hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyAtoHAsync(dstHost:POINTER(None), srcArray:CUarray, srcOffset:Annotated[int, ctypes.c_uint32], ByteCount:Annotated[int, ctypes.c_uint32], hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy2D(pCopy:POINTER(CUDA_MEMCPY2D_v1)) -> CUresult: ...
@dll.bind
def cuMemcpy2DUnaligned(pCopy:POINTER(CUDA_MEMCPY2D_v1)) -> CUresult: ...
@dll.bind
def cuMemcpy3D(pCopy:POINTER(CUDA_MEMCPY3D_v1)) -> CUresult: ...
@dll.bind
def cuMemcpyHtoDAsync(dstDevice:CUdeviceptr_v1, srcHost:POINTER(None), ByteCount:Annotated[int, ctypes.c_uint32], hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyDtoHAsync(dstHost:POINTER(None), srcDevice:CUdeviceptr_v1, ByteCount:Annotated[int, ctypes.c_uint32], hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyDtoDAsync(dstDevice:CUdeviceptr_v1, srcDevice:CUdeviceptr_v1, ByteCount:Annotated[int, ctypes.c_uint32], hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy2DAsync(pCopy:POINTER(CUDA_MEMCPY2D_v1), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy3DAsync(pCopy:POINTER(CUDA_MEMCPY3D_v1), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD8(dstDevice:CUdeviceptr_v1, uc:Annotated[int, ctypes.c_ubyte], N:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemsetD16(dstDevice:CUdeviceptr_v1, us:Annotated[int, ctypes.c_uint16], N:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemsetD32(dstDevice:CUdeviceptr_v1, ui:Annotated[int, ctypes.c_uint32], N:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemsetD2D8(dstDevice:CUdeviceptr_v1, dstPitch:Annotated[int, ctypes.c_uint32], uc:Annotated[int, ctypes.c_ubyte], Width:Annotated[int, ctypes.c_uint32], Height:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemsetD2D16(dstDevice:CUdeviceptr_v1, dstPitch:Annotated[int, ctypes.c_uint32], us:Annotated[int, ctypes.c_uint16], Width:Annotated[int, ctypes.c_uint32], Height:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemsetD2D32(dstDevice:CUdeviceptr_v1, dstPitch:Annotated[int, ctypes.c_uint32], ui:Annotated[int, ctypes.c_uint32], Width:Annotated[int, ctypes.c_uint32], Height:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuArrayCreate(pHandle:POINTER(CUarray), pAllocateArray:POINTER(CUDA_ARRAY_DESCRIPTOR_v1)) -> CUresult: ...
@dll.bind
def cuArrayGetDescriptor(pArrayDescriptor:POINTER(CUDA_ARRAY_DESCRIPTOR_v1), hArray:CUarray) -> CUresult: ...
@dll.bind
def cuArray3DCreate(pHandle:POINTER(CUarray), pAllocateArray:POINTER(CUDA_ARRAY3D_DESCRIPTOR_v1)) -> CUresult: ...
@dll.bind
def cuArray3DGetDescriptor(pArrayDescriptor:POINTER(CUDA_ARRAY3D_DESCRIPTOR_v1), hArray:CUarray) -> CUresult: ...
@dll.bind
def cuTexRefSetAddress(ByteOffset:POINTER(Annotated[int, ctypes.c_uint32]), hTexRef:CUtexref, dptr:CUdeviceptr_v1, bytes:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuTexRefSetAddress2D(hTexRef:CUtexref, desc:POINTER(CUDA_ARRAY_DESCRIPTOR_v1), dptr:CUdeviceptr_v1, Pitch:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuTexRefGetAddress(pdptr:POINTER(CUdeviceptr_v1), hTexRef:CUtexref) -> CUresult: ...
@dll.bind
def cuGraphicsResourceGetMappedPointer(pDevPtr:POINTER(CUdeviceptr_v1), pSize:POINTER(Annotated[int, ctypes.c_uint32]), resource:CUgraphicsResource) -> CUresult: ...
@dll.bind
def cuCtxDestroy(ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuCtxPopCurrent(pctx:POINTER(CUcontext)) -> CUresult: ...
@dll.bind
def cuCtxPushCurrent(ctx:CUcontext) -> CUresult: ...
@dll.bind
def cuStreamDestroy(hStream:CUstream) -> CUresult: ...
@dll.bind
def cuEventDestroy(hEvent:CUevent) -> CUresult: ...
@dll.bind
def cuDevicePrimaryCtxRelease(dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDevicePrimaryCtxReset(dev:CUdevice) -> CUresult: ...
@dll.bind
def cuDevicePrimaryCtxSetFlags(dev:CUdevice, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemcpyHtoD_v2(dstDevice:CUdeviceptr, srcHost:POINTER(None), ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyDtoH_v2(dstHost:POINTER(None), srcDevice:CUdeviceptr, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyDtoD_v2(dstDevice:CUdeviceptr, srcDevice:CUdeviceptr, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyDtoA_v2(dstArray:CUarray, dstOffset:size_t, srcDevice:CUdeviceptr, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyAtoD_v2(dstDevice:CUdeviceptr, srcArray:CUarray, srcOffset:size_t, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyHtoA_v2(dstArray:CUarray, dstOffset:size_t, srcHost:POINTER(None), ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyAtoH_v2(dstHost:POINTER(None), srcArray:CUarray, srcOffset:size_t, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyAtoA_v2(dstArray:CUarray, dstOffset:size_t, srcArray:CUarray, srcOffset:size_t, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyHtoAAsync_v2(dstArray:CUarray, dstOffset:size_t, srcHost:POINTER(None), ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyAtoHAsync_v2(dstHost:POINTER(None), srcArray:CUarray, srcOffset:size_t, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy2D_v2(pCopy:POINTER(CUDA_MEMCPY2D)) -> CUresult: ...
@dll.bind
def cuMemcpy2DUnaligned_v2(pCopy:POINTER(CUDA_MEMCPY2D)) -> CUresult: ...
@dll.bind
def cuMemcpy3D_v2(pCopy:POINTER(CUDA_MEMCPY3D)) -> CUresult: ...
@dll.bind
def cuMemcpyHtoDAsync_v2(dstDevice:CUdeviceptr, srcHost:POINTER(None), ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyDtoHAsync_v2(dstHost:POINTER(None), srcDevice:CUdeviceptr, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyDtoDAsync_v2(dstDevice:CUdeviceptr, srcDevice:CUdeviceptr, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy2DAsync_v2(pCopy:POINTER(CUDA_MEMCPY2D), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy3DAsync_v2(pCopy:POINTER(CUDA_MEMCPY3D), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD8_v2(dstDevice:CUdeviceptr, uc:Annotated[int, ctypes.c_ubyte], N:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD16_v2(dstDevice:CUdeviceptr, us:Annotated[int, ctypes.c_uint16], N:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD32_v2(dstDevice:CUdeviceptr, ui:Annotated[int, ctypes.c_uint32], N:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD2D8_v2(dstDevice:CUdeviceptr, dstPitch:size_t, uc:Annotated[int, ctypes.c_ubyte], Width:size_t, Height:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD2D16_v2(dstDevice:CUdeviceptr, dstPitch:size_t, us:Annotated[int, ctypes.c_uint16], Width:size_t, Height:size_t) -> CUresult: ...
@dll.bind
def cuMemsetD2D32_v2(dstDevice:CUdeviceptr, dstPitch:size_t, ui:Annotated[int, ctypes.c_uint32], Width:size_t, Height:size_t) -> CUresult: ...
@dll.bind
def cuMemcpy(dst:CUdeviceptr, src:CUdeviceptr, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyAsync(dst:CUdeviceptr, src:CUdeviceptr, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpyPeer(dstDevice:CUdeviceptr, dstContext:CUcontext, srcDevice:CUdeviceptr, srcContext:CUcontext, ByteCount:size_t) -> CUresult: ...
@dll.bind
def cuMemcpyPeerAsync(dstDevice:CUdeviceptr, dstContext:CUcontext, srcDevice:CUdeviceptr, srcContext:CUcontext, ByteCount:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemcpy3DPeer(pCopy:POINTER(CUDA_MEMCPY3D_PEER)) -> CUresult: ...
@dll.bind
def cuMemcpy3DPeerAsync(pCopy:POINTER(CUDA_MEMCPY3D_PEER), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD8Async(dstDevice:CUdeviceptr, uc:Annotated[int, ctypes.c_ubyte], N:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD16Async(dstDevice:CUdeviceptr, us:Annotated[int, ctypes.c_uint16], N:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD32Async(dstDevice:CUdeviceptr, ui:Annotated[int, ctypes.c_uint32], N:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD2D8Async(dstDevice:CUdeviceptr, dstPitch:size_t, uc:Annotated[int, ctypes.c_ubyte], Width:size_t, Height:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD2D16Async(dstDevice:CUdeviceptr, dstPitch:size_t, us:Annotated[int, ctypes.c_uint16], Width:size_t, Height:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemsetD2D32Async(dstDevice:CUdeviceptr, dstPitch:size_t, ui:Annotated[int, ctypes.c_uint32], Width:size_t, Height:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamGetPriority(hStream:CUstream, priority:POINTER(Annotated[int, ctypes.c_int32])) -> CUresult: ...
@dll.bind
def cuStreamGetId(hStream:CUstream, streamId:POINTER(Annotated[int, ctypes.c_uint64])) -> CUresult: ...
@dll.bind
def cuStreamGetFlags(hStream:CUstream, flags:POINTER(Annotated[int, ctypes.c_uint32])) -> CUresult: ...
@dll.bind
def cuStreamGetCtx(hStream:CUstream, pctx:POINTER(CUcontext)) -> CUresult: ...
@dll.bind
def cuStreamWaitEvent(hStream:CUstream, hEvent:CUevent, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamAddCallback(hStream:CUstream, callback:CUstreamCallback, userData:POINTER(None), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamAttachMemAsync(hStream:CUstream, dptr:CUdeviceptr, length:size_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamQuery(hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamSynchronize(hStream:CUstream) -> CUresult: ...
@dll.bind
def cuEventRecord(hEvent:CUevent, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuEventRecordWithFlags(hEvent:CUevent, hStream:CUstream, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuLaunchKernel(f:CUfunction, gridDimX:Annotated[int, ctypes.c_uint32], gridDimY:Annotated[int, ctypes.c_uint32], gridDimZ:Annotated[int, ctypes.c_uint32], blockDimX:Annotated[int, ctypes.c_uint32], blockDimY:Annotated[int, ctypes.c_uint32], blockDimZ:Annotated[int, ctypes.c_uint32], sharedMemBytes:Annotated[int, ctypes.c_uint32], hStream:CUstream, kernelParams:POINTER(POINTER(None)), extra:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuLaunchKernelEx(config:POINTER(CUlaunchConfig), f:CUfunction, kernelParams:POINTER(POINTER(None)), extra:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuLaunchHostFunc(hStream:CUstream, fn:CUhostFn, userData:POINTER(None)) -> CUresult: ...
@dll.bind
def cuGraphicsMapResources(count:Annotated[int, ctypes.c_uint32], resources:POINTER(CUgraphicsResource), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuGraphicsUnmapResources(count:Annotated[int, ctypes.c_uint32], resources:POINTER(CUgraphicsResource), hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamWriteValue32(stream:CUstream, addr:CUdeviceptr, value:cuuint32_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWaitValue32(stream:CUstream, addr:CUdeviceptr, value:cuuint32_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWriteValue64(stream:CUstream, addr:CUdeviceptr, value:cuuint64_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWaitValue64(stream:CUstream, addr:CUdeviceptr, value:cuuint64_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamBatchMemOp(stream:CUstream, count:Annotated[int, ctypes.c_uint32], paramArray:POINTER(CUstreamBatchMemOpParams), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWriteValue32_ptsz(stream:CUstream, addr:CUdeviceptr, value:cuuint32_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWaitValue32_ptsz(stream:CUstream, addr:CUdeviceptr, value:cuuint32_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWriteValue64_ptsz(stream:CUstream, addr:CUdeviceptr, value:cuuint64_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWaitValue64_ptsz(stream:CUstream, addr:CUdeviceptr, value:cuuint64_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamBatchMemOp_ptsz(stream:CUstream, count:Annotated[int, ctypes.c_uint32], paramArray:POINTER(CUstreamBatchMemOpParams), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWriteValue32_v2(stream:CUstream, addr:CUdeviceptr, value:cuuint32_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWaitValue32_v2(stream:CUstream, addr:CUdeviceptr, value:cuuint32_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWriteValue64_v2(stream:CUstream, addr:CUdeviceptr, value:cuuint64_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamWaitValue64_v2(stream:CUstream, addr:CUdeviceptr, value:cuuint64_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuStreamBatchMemOp_v2(stream:CUstream, count:Annotated[int, ctypes.c_uint32], paramArray:POINTER(CUstreamBatchMemOpParams), flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuMemPrefetchAsync(devPtr:CUdeviceptr, count:size_t, dstDevice:CUdevice, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuLaunchCooperativeKernel(f:CUfunction, gridDimX:Annotated[int, ctypes.c_uint32], gridDimY:Annotated[int, ctypes.c_uint32], gridDimZ:Annotated[int, ctypes.c_uint32], blockDimX:Annotated[int, ctypes.c_uint32], blockDimY:Annotated[int, ctypes.c_uint32], blockDimZ:Annotated[int, ctypes.c_uint32], sharedMemBytes:Annotated[int, ctypes.c_uint32], hStream:CUstream, kernelParams:POINTER(POINTER(None))) -> CUresult: ...
@dll.bind
def cuSignalExternalSemaphoresAsync(extSemArray:POINTER(CUexternalSemaphore), paramsArray:POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), numExtSems:Annotated[int, ctypes.c_uint32], stream:CUstream) -> CUresult: ...
@dll.bind
def cuWaitExternalSemaphoresAsync(extSemArray:POINTER(CUexternalSemaphore), paramsArray:POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS), numExtSems:Annotated[int, ctypes.c_uint32], stream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamBeginCapture(hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamBeginCapture_ptsz(hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamBeginCapture_v2(hStream:CUstream, mode:CUstreamCaptureMode) -> CUresult: ...
@dll.bind
def cuStreamEndCapture(hStream:CUstream, phGraph:POINTER(CUgraph)) -> CUresult: ...
@dll.bind
def cuStreamIsCapturing(hStream:CUstream, captureStatus:POINTER(CUstreamCaptureStatus)) -> CUresult: ...
@dll.bind
def cuStreamGetCaptureInfo(hStream:CUstream, captureStatus_out:POINTER(CUstreamCaptureStatus), id_out:POINTER(cuuint64_t)) -> CUresult: ...
@dll.bind
def cuStreamGetCaptureInfo_ptsz(hStream:CUstream, captureStatus_out:POINTER(CUstreamCaptureStatus), id_out:POINTER(cuuint64_t)) -> CUresult: ...
@dll.bind
def cuStreamGetCaptureInfo_v2(hStream:CUstream, captureStatus_out:POINTER(CUstreamCaptureStatus), id_out:POINTER(cuuint64_t), graph_out:POINTER(CUgraph), dependencies_out:POINTER(POINTER(CUgraphNode)), numDependencies_out:POINTER(size_t)) -> CUresult: ...
@dll.bind
def cuGraphAddKernelNode(phGraphNode:POINTER(CUgraphNode), hGraph:CUgraph, dependencies:POINTER(CUgraphNode), numDependencies:size_t, nodeParams:POINTER(CUDA_KERNEL_NODE_PARAMS_v1)) -> CUresult: ...
@dll.bind
def cuGraphKernelNodeGetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_KERNEL_NODE_PARAMS_v1)) -> CUresult: ...
@dll.bind
def cuGraphKernelNodeSetParams(hNode:CUgraphNode, nodeParams:POINTER(CUDA_KERNEL_NODE_PARAMS_v1)) -> CUresult: ...
@dll.bind
def cuGraphExecKernelNodeSetParams(hGraphExec:CUgraphExec, hNode:CUgraphNode, nodeParams:POINTER(CUDA_KERNEL_NODE_PARAMS_v1)) -> CUresult: ...
@dll.bind
def cuGraphInstantiateWithParams(phGraphExec:POINTER(CUgraphExec), hGraph:CUgraph, instantiateParams:POINTER(CUDA_GRAPH_INSTANTIATE_PARAMS)) -> CUresult: ...
@dll.bind
def cuGraphExecUpdate(hGraphExec:CUgraphExec, hGraph:CUgraph, hErrorNode_out:POINTER(CUgraphNode), updateResult_out:POINTER(CUgraphExecUpdateResult)) -> CUresult: ...
@dll.bind
def cuGraphUpload(hGraph:CUgraphExec, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuGraphLaunch(hGraph:CUgraphExec, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamCopyAttributes(dstStream:CUstream, srcStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamGetAttribute(hStream:CUstream, attr:CUstreamAttrID, value:POINTER(CUstreamAttrValue)) -> CUresult: ...
@dll.bind
def cuStreamSetAttribute(hStream:CUstream, attr:CUstreamAttrID, param:POINTER(CUstreamAttrValue)) -> CUresult: ...
@dll.bind
def cuIpcOpenMemHandle(pdptr:POINTER(CUdeviceptr), handle:CUipcMemHandle, Flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGraphInstantiate(phGraphExec:POINTER(CUgraphExec), hGraph:CUgraph, phErrorNode:POINTER(CUgraphNode), logBuffer:POINTER(Annotated[bytes, ctypes.c_char]), bufferSize:size_t) -> CUresult: ...
@dll.bind
def cuGraphInstantiate_v2(phGraphExec:POINTER(CUgraphExec), hGraph:CUgraph, phErrorNode:POINTER(CUgraphNode), logBuffer:POINTER(Annotated[bytes, ctypes.c_char]), bufferSize:size_t) -> CUresult: ...
@dll.bind
def cuMemMapArrayAsync(mapInfoList:POINTER(CUarrayMapInfo), count:Annotated[int, ctypes.c_uint32], hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemFreeAsync(dptr:CUdeviceptr, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemAllocAsync(dptr:POINTER(CUdeviceptr), bytesize:size_t, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuMemAllocFromPoolAsync(dptr:POINTER(CUdeviceptr), bytesize:size_t, pool:CUmemoryPool, hStream:CUstream) -> CUresult: ...
@dll.bind
def cuStreamUpdateCaptureDependencies(hStream:CUstream, dependencies:POINTER(CUgraphNode), numDependencies:size_t, flags:Annotated[int, ctypes.c_uint32]) -> CUresult: ...
@dll.bind
def cuGetProcAddress(symbol:POINTER(Annotated[bytes, ctypes.c_char]), pfn:POINTER(POINTER(None)), cudaVersion:Annotated[int, ctypes.c_int32], flags:cuuint64_t) -> CUresult: ...
init_records()
