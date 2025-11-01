# mypy: ignore-errors
import ctypes
from ctypes.util import find_library
from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR
def dll():
  try: return ctypes.CDLL(find_library('cuda'))
  except: pass
  return None
dll = dll()
cuuint32_t = ctypes.c_uint
cuuint64_t = ctypes.c_ulong
CUdeviceptr_v2 = ctypes.c_ulonglong
CUdeviceptr = ctypes.c_ulonglong
CUdevice_v1 = ctypes.c_int
CUdevice = ctypes.c_int
class struct_CUctx_st(ctypes.Structure): pass
struct_CUctx_st._fields_ = []
CUcontext = ctypes.POINTER(struct_CUctx_st)
class struct_CUmod_st(ctypes.Structure): pass
struct_CUmod_st._fields_ = []
CUmodule = ctypes.POINTER(struct_CUmod_st)
class struct_CUfunc_st(ctypes.Structure): pass
struct_CUfunc_st._fields_ = []
CUfunction = ctypes.POINTER(struct_CUfunc_st)
class struct_CUlib_st(ctypes.Structure): pass
struct_CUlib_st._fields_ = []
CUlibrary = ctypes.POINTER(struct_CUlib_st)
class struct_CUkern_st(ctypes.Structure): pass
struct_CUkern_st._fields_ = []
CUkernel = ctypes.POINTER(struct_CUkern_st)
class struct_CUarray_st(ctypes.Structure): pass
struct_CUarray_st._fields_ = []
CUarray = ctypes.POINTER(struct_CUarray_st)
class struct_CUmipmappedArray_st(ctypes.Structure): pass
struct_CUmipmappedArray_st._fields_ = []
CUmipmappedArray = ctypes.POINTER(struct_CUmipmappedArray_st)
class struct_CUtexref_st(ctypes.Structure): pass
struct_CUtexref_st._fields_ = []
CUtexref = ctypes.POINTER(struct_CUtexref_st)
class struct_CUsurfref_st(ctypes.Structure): pass
struct_CUsurfref_st._fields_ = []
CUsurfref = ctypes.POINTER(struct_CUsurfref_st)
class struct_CUevent_st(ctypes.Structure): pass
struct_CUevent_st._fields_ = []
CUevent = ctypes.POINTER(struct_CUevent_st)
class struct_CUstream_st(ctypes.Structure): pass
struct_CUstream_st._fields_ = []
CUstream = ctypes.POINTER(struct_CUstream_st)
class struct_CUgraphicsResource_st(ctypes.Structure): pass
struct_CUgraphicsResource_st._fields_ = []
CUgraphicsResource = ctypes.POINTER(struct_CUgraphicsResource_st)
CUtexObject_v1 = ctypes.c_ulonglong
CUtexObject = ctypes.c_ulonglong
CUsurfObject_v1 = ctypes.c_ulonglong
CUsurfObject = ctypes.c_ulonglong
class struct_CUextMemory_st(ctypes.Structure): pass
struct_CUextMemory_st._fields_ = []
CUexternalMemory = ctypes.POINTER(struct_CUextMemory_st)
class struct_CUextSemaphore_st(ctypes.Structure): pass
struct_CUextSemaphore_st._fields_ = []
CUexternalSemaphore = ctypes.POINTER(struct_CUextSemaphore_st)
class struct_CUgraph_st(ctypes.Structure): pass
struct_CUgraph_st._fields_ = []
CUgraph = ctypes.POINTER(struct_CUgraph_st)
class struct_CUgraphNode_st(ctypes.Structure): pass
struct_CUgraphNode_st._fields_ = []
CUgraphNode = ctypes.POINTER(struct_CUgraphNode_st)
class struct_CUgraphExec_st(ctypes.Structure): pass
struct_CUgraphExec_st._fields_ = []
CUgraphExec = ctypes.POINTER(struct_CUgraphExec_st)
class struct_CUmemPoolHandle_st(ctypes.Structure): pass
struct_CUmemPoolHandle_st._fields_ = []
CUmemoryPool = ctypes.POINTER(struct_CUmemPoolHandle_st)
class struct_CUuserObject_st(ctypes.Structure): pass
struct_CUuserObject_st._fields_ = []
CUuserObject = ctypes.POINTER(struct_CUuserObject_st)
class struct_CUuuid_st(ctypes.Structure): pass
struct_CUuuid_st._fields_ = [
  ('bytes', (ctypes.c_char * 16)),
]
CUuuid = struct_CUuuid_st
class struct_CUipcEventHandle_st(ctypes.Structure): pass
struct_CUipcEventHandle_st._fields_ = [
  ('reserved', (ctypes.c_char * 64)),
]
CUipcEventHandle_v1 = struct_CUipcEventHandle_st
CUipcEventHandle = struct_CUipcEventHandle_st
class struct_CUipcMemHandle_st(ctypes.Structure): pass
struct_CUipcMemHandle_st._fields_ = [
  ('reserved', (ctypes.c_char * 64)),
]
CUipcMemHandle_v1 = struct_CUipcMemHandle_st
CUipcMemHandle = struct_CUipcMemHandle_st
enum_CUipcMem_flags_enum = CEnum(ctypes.c_uint)
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = enum_CUipcMem_flags_enum.define('CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS', 1)

CUipcMem_flags = enum_CUipcMem_flags_enum
enum_CUmemAttach_flags_enum = CEnum(ctypes.c_uint)
CU_MEM_ATTACH_GLOBAL = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_GLOBAL', 1)
CU_MEM_ATTACH_HOST = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_HOST', 2)
CU_MEM_ATTACH_SINGLE = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_SINGLE', 4)

CUmemAttach_flags = enum_CUmemAttach_flags_enum
enum_CUctx_flags_enum = CEnum(ctypes.c_uint)
CU_CTX_SCHED_AUTO = enum_CUctx_flags_enum.define('CU_CTX_SCHED_AUTO', 0)
CU_CTX_SCHED_SPIN = enum_CUctx_flags_enum.define('CU_CTX_SCHED_SPIN', 1)
CU_CTX_SCHED_YIELD = enum_CUctx_flags_enum.define('CU_CTX_SCHED_YIELD', 2)
CU_CTX_SCHED_BLOCKING_SYNC = enum_CUctx_flags_enum.define('CU_CTX_SCHED_BLOCKING_SYNC', 4)
CU_CTX_BLOCKING_SYNC = enum_CUctx_flags_enum.define('CU_CTX_BLOCKING_SYNC', 4)
CU_CTX_SCHED_MASK = enum_CUctx_flags_enum.define('CU_CTX_SCHED_MASK', 7)
CU_CTX_MAP_HOST = enum_CUctx_flags_enum.define('CU_CTX_MAP_HOST', 8)
CU_CTX_LMEM_RESIZE_TO_MAX = enum_CUctx_flags_enum.define('CU_CTX_LMEM_RESIZE_TO_MAX', 16)
CU_CTX_FLAGS_MASK = enum_CUctx_flags_enum.define('CU_CTX_FLAGS_MASK', 31)

CUctx_flags = enum_CUctx_flags_enum
enum_CUevent_sched_flags_enum = CEnum(ctypes.c_uint)
CU_EVENT_SCHED_AUTO = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_AUTO', 0)
CU_EVENT_SCHED_SPIN = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_SPIN', 1)
CU_EVENT_SCHED_YIELD = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_YIELD', 2)
CU_EVENT_SCHED_BLOCKING_SYNC = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_BLOCKING_SYNC', 4)

CUevent_sched_flags = enum_CUevent_sched_flags_enum
enum_cl_event_flags_enum = CEnum(ctypes.c_uint)
NVCL_EVENT_SCHED_AUTO = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_AUTO', 0)
NVCL_EVENT_SCHED_SPIN = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_SPIN', 1)
NVCL_EVENT_SCHED_YIELD = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_YIELD', 2)
NVCL_EVENT_SCHED_BLOCKING_SYNC = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_BLOCKING_SYNC', 4)

cl_event_flags = enum_cl_event_flags_enum
enum_cl_context_flags_enum = CEnum(ctypes.c_uint)
NVCL_CTX_SCHED_AUTO = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_AUTO', 0)
NVCL_CTX_SCHED_SPIN = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_SPIN', 1)
NVCL_CTX_SCHED_YIELD = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_YIELD', 2)
NVCL_CTX_SCHED_BLOCKING_SYNC = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_BLOCKING_SYNC', 4)

cl_context_flags = enum_cl_context_flags_enum
enum_CUstream_flags_enum = CEnum(ctypes.c_uint)
CU_STREAM_DEFAULT = enum_CUstream_flags_enum.define('CU_STREAM_DEFAULT', 0)
CU_STREAM_NON_BLOCKING = enum_CUstream_flags_enum.define('CU_STREAM_NON_BLOCKING', 1)

CUstream_flags = enum_CUstream_flags_enum
enum_CUevent_flags_enum = CEnum(ctypes.c_uint)
CU_EVENT_DEFAULT = enum_CUevent_flags_enum.define('CU_EVENT_DEFAULT', 0)
CU_EVENT_BLOCKING_SYNC = enum_CUevent_flags_enum.define('CU_EVENT_BLOCKING_SYNC', 1)
CU_EVENT_DISABLE_TIMING = enum_CUevent_flags_enum.define('CU_EVENT_DISABLE_TIMING', 2)
CU_EVENT_INTERPROCESS = enum_CUevent_flags_enum.define('CU_EVENT_INTERPROCESS', 4)

CUevent_flags = enum_CUevent_flags_enum
enum_CUevent_record_flags_enum = CEnum(ctypes.c_uint)
CU_EVENT_RECORD_DEFAULT = enum_CUevent_record_flags_enum.define('CU_EVENT_RECORD_DEFAULT', 0)
CU_EVENT_RECORD_EXTERNAL = enum_CUevent_record_flags_enum.define('CU_EVENT_RECORD_EXTERNAL', 1)

CUevent_record_flags = enum_CUevent_record_flags_enum
enum_CUevent_wait_flags_enum = CEnum(ctypes.c_uint)
CU_EVENT_WAIT_DEFAULT = enum_CUevent_wait_flags_enum.define('CU_EVENT_WAIT_DEFAULT', 0)
CU_EVENT_WAIT_EXTERNAL = enum_CUevent_wait_flags_enum.define('CU_EVENT_WAIT_EXTERNAL', 1)

CUevent_wait_flags = enum_CUevent_wait_flags_enum
enum_CUstreamWaitValue_flags_enum = CEnum(ctypes.c_uint)
CU_STREAM_WAIT_VALUE_GEQ = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_GEQ', 0)
CU_STREAM_WAIT_VALUE_EQ = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_EQ', 1)
CU_STREAM_WAIT_VALUE_AND = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_AND', 2)
CU_STREAM_WAIT_VALUE_NOR = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_NOR', 3)
CU_STREAM_WAIT_VALUE_FLUSH = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_FLUSH', 1073741824)

CUstreamWaitValue_flags = enum_CUstreamWaitValue_flags_enum
enum_CUstreamWriteValue_flags_enum = CEnum(ctypes.c_uint)
CU_STREAM_WRITE_VALUE_DEFAULT = enum_CUstreamWriteValue_flags_enum.define('CU_STREAM_WRITE_VALUE_DEFAULT', 0)
CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = enum_CUstreamWriteValue_flags_enum.define('CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER', 1)

CUstreamWriteValue_flags = enum_CUstreamWriteValue_flags_enum
enum_CUstreamBatchMemOpType_enum = CEnum(ctypes.c_uint)
CU_STREAM_MEM_OP_WAIT_VALUE_32 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WAIT_VALUE_32', 1)
CU_STREAM_MEM_OP_WRITE_VALUE_32 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WRITE_VALUE_32', 2)
CU_STREAM_MEM_OP_WAIT_VALUE_64 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WAIT_VALUE_64', 4)
CU_STREAM_MEM_OP_WRITE_VALUE_64 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WRITE_VALUE_64', 5)
CU_STREAM_MEM_OP_BARRIER = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_BARRIER', 6)
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES', 3)

CUstreamBatchMemOpType = enum_CUstreamBatchMemOpType_enum
enum_CUstreamMemoryBarrier_flags_enum = CEnum(ctypes.c_uint)
CU_STREAM_MEMORY_BARRIER_TYPE_SYS = enum_CUstreamMemoryBarrier_flags_enum.define('CU_STREAM_MEMORY_BARRIER_TYPE_SYS', 0)
CU_STREAM_MEMORY_BARRIER_TYPE_GPU = enum_CUstreamMemoryBarrier_flags_enum.define('CU_STREAM_MEMORY_BARRIER_TYPE_GPU', 1)

CUstreamMemoryBarrier_flags = enum_CUstreamMemoryBarrier_flags_enum
class union_CUstreamBatchMemOpParams_union(ctypes.Union): pass
class struct_CUstreamMemOpWaitValueParams_st(ctypes.Structure): pass
class _anonunion0(ctypes.Union): pass
_anonunion0._fields_ = [
  ('value', cuuint32_t),
  ('value64', cuuint64_t),
]
struct_CUstreamMemOpWaitValueParams_st._anonymous_ = ['_0']
struct_CUstreamMemOpWaitValueParams_st._fields_ = [
  ('operation', CUstreamBatchMemOpType),
  ('address', CUdeviceptr),
  ('_0', _anonunion0),
  ('flags', ctypes.c_uint),
  ('alias', CUdeviceptr),
]
class struct_CUstreamMemOpWriteValueParams_st(ctypes.Structure): pass
class _anonunion1(ctypes.Union): pass
_anonunion1._fields_ = [
  ('value', cuuint32_t),
  ('value64', cuuint64_t),
]
struct_CUstreamMemOpWriteValueParams_st._anonymous_ = ['_0']
struct_CUstreamMemOpWriteValueParams_st._fields_ = [
  ('operation', CUstreamBatchMemOpType),
  ('address', CUdeviceptr),
  ('_0', _anonunion1),
  ('flags', ctypes.c_uint),
  ('alias', CUdeviceptr),
]
class struct_CUstreamMemOpFlushRemoteWritesParams_st(ctypes.Structure): pass
struct_CUstreamMemOpFlushRemoteWritesParams_st._fields_ = [
  ('operation', CUstreamBatchMemOpType),
  ('flags', ctypes.c_uint),
]
class struct_CUstreamMemOpMemoryBarrierParams_st(ctypes.Structure): pass
struct_CUstreamMemOpMemoryBarrierParams_st._fields_ = [
  ('operation', CUstreamBatchMemOpType),
  ('flags', ctypes.c_uint),
]
union_CUstreamBatchMemOpParams_union._fields_ = [
  ('operation', CUstreamBatchMemOpType),
  ('waitValue', struct_CUstreamMemOpWaitValueParams_st),
  ('writeValue', struct_CUstreamMemOpWriteValueParams_st),
  ('flushRemoteWrites', struct_CUstreamMemOpFlushRemoteWritesParams_st),
  ('memoryBarrier', struct_CUstreamMemOpMemoryBarrierParams_st),
  ('pad', (cuuint64_t * 6)),
]
CUstreamBatchMemOpParams_v1 = union_CUstreamBatchMemOpParams_union
CUstreamBatchMemOpParams = union_CUstreamBatchMemOpParams_union
class struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st(ctypes.Structure): pass
struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st._fields_ = [
  ('ctx', CUcontext),
  ('count', ctypes.c_uint),
  ('paramArray', ctypes.POINTER(CUstreamBatchMemOpParams)),
  ('flags', ctypes.c_uint),
]
CUDA_BATCH_MEM_OP_NODE_PARAMS = struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st
enum_CUoccupancy_flags_enum = CEnum(ctypes.c_uint)
CU_OCCUPANCY_DEFAULT = enum_CUoccupancy_flags_enum.define('CU_OCCUPANCY_DEFAULT', 0)
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = enum_CUoccupancy_flags_enum.define('CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE', 1)

CUoccupancy_flags = enum_CUoccupancy_flags_enum
enum_CUstreamUpdateCaptureDependencies_flags_enum = CEnum(ctypes.c_uint)
CU_STREAM_ADD_CAPTURE_DEPENDENCIES = enum_CUstreamUpdateCaptureDependencies_flags_enum.define('CU_STREAM_ADD_CAPTURE_DEPENDENCIES', 0)
CU_STREAM_SET_CAPTURE_DEPENDENCIES = enum_CUstreamUpdateCaptureDependencies_flags_enum.define('CU_STREAM_SET_CAPTURE_DEPENDENCIES', 1)

CUstreamUpdateCaptureDependencies_flags = enum_CUstreamUpdateCaptureDependencies_flags_enum
enum_CUarray_format_enum = CEnum(ctypes.c_uint)
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

CUarray_format = enum_CUarray_format_enum
enum_CUaddress_mode_enum = CEnum(ctypes.c_uint)
CU_TR_ADDRESS_MODE_WRAP = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_WRAP', 0)
CU_TR_ADDRESS_MODE_CLAMP = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_CLAMP', 1)
CU_TR_ADDRESS_MODE_MIRROR = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_MIRROR', 2)
CU_TR_ADDRESS_MODE_BORDER = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_BORDER', 3)

CUaddress_mode = enum_CUaddress_mode_enum
enum_CUfilter_mode_enum = CEnum(ctypes.c_uint)
CU_TR_FILTER_MODE_POINT = enum_CUfilter_mode_enum.define('CU_TR_FILTER_MODE_POINT', 0)
CU_TR_FILTER_MODE_LINEAR = enum_CUfilter_mode_enum.define('CU_TR_FILTER_MODE_LINEAR', 1)

CUfilter_mode = enum_CUfilter_mode_enum
enum_CUdevice_attribute_enum = CEnum(ctypes.c_uint)
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

CUdevice_attribute = enum_CUdevice_attribute_enum
class struct_CUdevprop_st(ctypes.Structure): pass
struct_CUdevprop_st._fields_ = [
  ('maxThreadsPerBlock', ctypes.c_int),
  ('maxThreadsDim', (ctypes.c_int * 3)),
  ('maxGridSize', (ctypes.c_int * 3)),
  ('sharedMemPerBlock', ctypes.c_int),
  ('totalConstantMemory', ctypes.c_int),
  ('SIMDWidth', ctypes.c_int),
  ('memPitch', ctypes.c_int),
  ('regsPerBlock', ctypes.c_int),
  ('clockRate', ctypes.c_int),
  ('textureAlign', ctypes.c_int),
]
CUdevprop_v1 = struct_CUdevprop_st
CUdevprop = struct_CUdevprop_st
enum_CUpointer_attribute_enum = CEnum(ctypes.c_uint)
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

CUpointer_attribute = enum_CUpointer_attribute_enum
enum_CUfunction_attribute_enum = CEnum(ctypes.c_uint)
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

CUfunction_attribute = enum_CUfunction_attribute_enum
enum_CUfunc_cache_enum = CEnum(ctypes.c_uint)
CU_FUNC_CACHE_PREFER_NONE = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_NONE', 0)
CU_FUNC_CACHE_PREFER_SHARED = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_SHARED', 1)
CU_FUNC_CACHE_PREFER_L1 = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_L1', 2)
CU_FUNC_CACHE_PREFER_EQUAL = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_EQUAL', 3)

CUfunc_cache = enum_CUfunc_cache_enum
enum_CUsharedconfig_enum = CEnum(ctypes.c_uint)
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE', 0)
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE', 1)
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE', 2)

CUsharedconfig = enum_CUsharedconfig_enum
enum_CUshared_carveout_enum = CEnum(ctypes.c_int)
CU_SHAREDMEM_CARVEOUT_DEFAULT = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_DEFAULT', -1)
CU_SHAREDMEM_CARVEOUT_MAX_SHARED = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_MAX_SHARED', 100)
CU_SHAREDMEM_CARVEOUT_MAX_L1 = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_MAX_L1', 0)

CUshared_carveout = enum_CUshared_carveout_enum
enum_CUmemorytype_enum = CEnum(ctypes.c_uint)
CU_MEMORYTYPE_HOST = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_HOST', 1)
CU_MEMORYTYPE_DEVICE = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_DEVICE', 2)
CU_MEMORYTYPE_ARRAY = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_ARRAY', 3)
CU_MEMORYTYPE_UNIFIED = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_UNIFIED', 4)

CUmemorytype = enum_CUmemorytype_enum
enum_CUcomputemode_enum = CEnum(ctypes.c_uint)
CU_COMPUTEMODE_DEFAULT = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_DEFAULT', 0)
CU_COMPUTEMODE_PROHIBITED = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_PROHIBITED', 2)
CU_COMPUTEMODE_EXCLUSIVE_PROCESS = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_EXCLUSIVE_PROCESS', 3)

CUcomputemode = enum_CUcomputemode_enum
enum_CUmem_advise_enum = CEnum(ctypes.c_uint)
CU_MEM_ADVISE_SET_READ_MOSTLY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_READ_MOSTLY', 1)
CU_MEM_ADVISE_UNSET_READ_MOSTLY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_READ_MOSTLY', 2)
CU_MEM_ADVISE_SET_PREFERRED_LOCATION = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_PREFERRED_LOCATION', 3)
CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION', 4)
CU_MEM_ADVISE_SET_ACCESSED_BY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_ACCESSED_BY', 5)
CU_MEM_ADVISE_UNSET_ACCESSED_BY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_ACCESSED_BY', 6)

CUmem_advise = enum_CUmem_advise_enum
enum_CUmem_range_attribute_enum = CEnum(ctypes.c_uint)
CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY', 1)
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION', 2)
CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY', 3)
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION', 4)

CUmem_range_attribute = enum_CUmem_range_attribute_enum
enum_CUjit_option_enum = CEnum(ctypes.c_uint)
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

CUjit_option = enum_CUjit_option_enum
enum_CUjit_target_enum = CEnum(ctypes.c_uint)
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

CUjit_target = enum_CUjit_target_enum
enum_CUjit_fallback_enum = CEnum(ctypes.c_uint)
CU_PREFER_PTX = enum_CUjit_fallback_enum.define('CU_PREFER_PTX', 0)
CU_PREFER_BINARY = enum_CUjit_fallback_enum.define('CU_PREFER_BINARY', 1)

CUjit_fallback = enum_CUjit_fallback_enum
enum_CUjit_cacheMode_enum = CEnum(ctypes.c_uint)
CU_JIT_CACHE_OPTION_NONE = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_NONE', 0)
CU_JIT_CACHE_OPTION_CG = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_CG', 1)
CU_JIT_CACHE_OPTION_CA = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_CA', 2)

CUjit_cacheMode = enum_CUjit_cacheMode_enum
enum_CUjitInputType_enum = CEnum(ctypes.c_uint)
CU_JIT_INPUT_CUBIN = enum_CUjitInputType_enum.define('CU_JIT_INPUT_CUBIN', 0)
CU_JIT_INPUT_PTX = enum_CUjitInputType_enum.define('CU_JIT_INPUT_PTX', 1)
CU_JIT_INPUT_FATBINARY = enum_CUjitInputType_enum.define('CU_JIT_INPUT_FATBINARY', 2)
CU_JIT_INPUT_OBJECT = enum_CUjitInputType_enum.define('CU_JIT_INPUT_OBJECT', 3)
CU_JIT_INPUT_LIBRARY = enum_CUjitInputType_enum.define('CU_JIT_INPUT_LIBRARY', 4)
CU_JIT_INPUT_NVVM = enum_CUjitInputType_enum.define('CU_JIT_INPUT_NVVM', 5)
CU_JIT_NUM_INPUT_TYPES = enum_CUjitInputType_enum.define('CU_JIT_NUM_INPUT_TYPES', 6)

CUjitInputType = enum_CUjitInputType_enum
class struct_CUlinkState_st(ctypes.Structure): pass
struct_CUlinkState_st._fields_ = []
CUlinkState = ctypes.POINTER(struct_CUlinkState_st)
enum_CUgraphicsRegisterFlags_enum = CEnum(ctypes.c_uint)
CU_GRAPHICS_REGISTER_FLAGS_NONE = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_NONE', 0)
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY', 1)
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD', 2)
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST', 4)
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER', 8)

CUgraphicsRegisterFlags = enum_CUgraphicsRegisterFlags_enum
enum_CUgraphicsMapResourceFlags_enum = CEnum(ctypes.c_uint)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE', 0)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY', 1)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD', 2)

CUgraphicsMapResourceFlags = enum_CUgraphicsMapResourceFlags_enum
enum_CUarray_cubemap_face_enum = CEnum(ctypes.c_uint)
CU_CUBEMAP_FACE_POSITIVE_X = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_X', 0)
CU_CUBEMAP_FACE_NEGATIVE_X = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_X', 1)
CU_CUBEMAP_FACE_POSITIVE_Y = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_Y', 2)
CU_CUBEMAP_FACE_NEGATIVE_Y = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_Y', 3)
CU_CUBEMAP_FACE_POSITIVE_Z = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_Z', 4)
CU_CUBEMAP_FACE_NEGATIVE_Z = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_Z', 5)

CUarray_cubemap_face = enum_CUarray_cubemap_face_enum
enum_CUlimit_enum = CEnum(ctypes.c_uint)
CU_LIMIT_STACK_SIZE = enum_CUlimit_enum.define('CU_LIMIT_STACK_SIZE', 0)
CU_LIMIT_PRINTF_FIFO_SIZE = enum_CUlimit_enum.define('CU_LIMIT_PRINTF_FIFO_SIZE', 1)
CU_LIMIT_MALLOC_HEAP_SIZE = enum_CUlimit_enum.define('CU_LIMIT_MALLOC_HEAP_SIZE', 2)
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = enum_CUlimit_enum.define('CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH', 3)
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = enum_CUlimit_enum.define('CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT', 4)
CU_LIMIT_MAX_L2_FETCH_GRANULARITY = enum_CUlimit_enum.define('CU_LIMIT_MAX_L2_FETCH_GRANULARITY', 5)
CU_LIMIT_PERSISTING_L2_CACHE_SIZE = enum_CUlimit_enum.define('CU_LIMIT_PERSISTING_L2_CACHE_SIZE', 6)
CU_LIMIT_MAX = enum_CUlimit_enum.define('CU_LIMIT_MAX', 7)

CUlimit = enum_CUlimit_enum
enum_CUresourcetype_enum = CEnum(ctypes.c_uint)
CU_RESOURCE_TYPE_ARRAY = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_ARRAY', 0)
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_MIPMAPPED_ARRAY', 1)
CU_RESOURCE_TYPE_LINEAR = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_LINEAR', 2)
CU_RESOURCE_TYPE_PITCH2D = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_PITCH2D', 3)

CUresourcetype = enum_CUresourcetype_enum
CUhostFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
enum_CUaccessProperty_enum = CEnum(ctypes.c_uint)
CU_ACCESS_PROPERTY_NORMAL = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_NORMAL', 0)
CU_ACCESS_PROPERTY_STREAMING = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_STREAMING', 1)
CU_ACCESS_PROPERTY_PERSISTING = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_PERSISTING', 2)

CUaccessProperty = enum_CUaccessProperty_enum
class struct_CUaccessPolicyWindow_st(ctypes.Structure): pass
size_t = ctypes.c_ulong
struct_CUaccessPolicyWindow_st._fields_ = [
  ('base_ptr', ctypes.c_void_p),
  ('num_bytes', size_t),
  ('hitRatio', ctypes.c_float),
  ('hitProp', CUaccessProperty),
  ('missProp', CUaccessProperty),
]
CUaccessPolicyWindow_v1 = struct_CUaccessPolicyWindow_st
CUaccessPolicyWindow = struct_CUaccessPolicyWindow_st
class struct_CUDA_KERNEL_NODE_PARAMS_st(ctypes.Structure): pass
struct_CUDA_KERNEL_NODE_PARAMS_st._fields_ = [
  ('func', CUfunction),
  ('gridDimX', ctypes.c_uint),
  ('gridDimY', ctypes.c_uint),
  ('gridDimZ', ctypes.c_uint),
  ('blockDimX', ctypes.c_uint),
  ('blockDimY', ctypes.c_uint),
  ('blockDimZ', ctypes.c_uint),
  ('sharedMemBytes', ctypes.c_uint),
  ('kernelParams', ctypes.POINTER(ctypes.c_void_p)),
  ('extra', ctypes.POINTER(ctypes.c_void_p)),
]
CUDA_KERNEL_NODE_PARAMS_v1 = struct_CUDA_KERNEL_NODE_PARAMS_st
class struct_CUDA_KERNEL_NODE_PARAMS_v2_st(ctypes.Structure): pass
struct_CUDA_KERNEL_NODE_PARAMS_v2_st._fields_ = [
  ('func', CUfunction),
  ('gridDimX', ctypes.c_uint),
  ('gridDimY', ctypes.c_uint),
  ('gridDimZ', ctypes.c_uint),
  ('blockDimX', ctypes.c_uint),
  ('blockDimY', ctypes.c_uint),
  ('blockDimZ', ctypes.c_uint),
  ('sharedMemBytes', ctypes.c_uint),
  ('kernelParams', ctypes.POINTER(ctypes.c_void_p)),
  ('extra', ctypes.POINTER(ctypes.c_void_p)),
  ('kern', CUkernel),
  ('ctx', CUcontext),
]
CUDA_KERNEL_NODE_PARAMS_v2 = struct_CUDA_KERNEL_NODE_PARAMS_v2_st
CUDA_KERNEL_NODE_PARAMS = struct_CUDA_KERNEL_NODE_PARAMS_v2_st
class struct_CUDA_MEMSET_NODE_PARAMS_st(ctypes.Structure): pass
struct_CUDA_MEMSET_NODE_PARAMS_st._fields_ = [
  ('dst', CUdeviceptr),
  ('pitch', size_t),
  ('value', ctypes.c_uint),
  ('elementSize', ctypes.c_uint),
  ('width', size_t),
  ('height', size_t),
]
CUDA_MEMSET_NODE_PARAMS_v1 = struct_CUDA_MEMSET_NODE_PARAMS_st
CUDA_MEMSET_NODE_PARAMS = struct_CUDA_MEMSET_NODE_PARAMS_st
class struct_CUDA_HOST_NODE_PARAMS_st(ctypes.Structure): pass
struct_CUDA_HOST_NODE_PARAMS_st._fields_ = [
  ('fn', CUhostFn),
  ('userData', ctypes.c_void_p),
]
CUDA_HOST_NODE_PARAMS_v1 = struct_CUDA_HOST_NODE_PARAMS_st
CUDA_HOST_NODE_PARAMS = struct_CUDA_HOST_NODE_PARAMS_st
enum_CUgraphNodeType_enum = CEnum(ctypes.c_uint)
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

CUgraphNodeType = enum_CUgraphNodeType_enum
enum_CUgraphInstantiateResult_enum = CEnum(ctypes.c_uint)
CUDA_GRAPH_INSTANTIATE_SUCCESS = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_SUCCESS', 0)
CUDA_GRAPH_INSTANTIATE_ERROR = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_ERROR', 1)
CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE', 2)
CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED', 3)
CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED', 4)

CUgraphInstantiateResult = enum_CUgraphInstantiateResult_enum
class struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st(ctypes.Structure): pass
struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st._fields_ = [
  ('flags', cuuint64_t),
  ('hUploadStream', CUstream),
  ('hErrNode_out', CUgraphNode),
  ('result_out', CUgraphInstantiateResult),
]
CUDA_GRAPH_INSTANTIATE_PARAMS = struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st
enum_CUsynchronizationPolicy_enum = CEnum(ctypes.c_uint)
CU_SYNC_POLICY_AUTO = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_AUTO', 1)
CU_SYNC_POLICY_SPIN = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_SPIN', 2)
CU_SYNC_POLICY_YIELD = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_YIELD', 3)
CU_SYNC_POLICY_BLOCKING_SYNC = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_BLOCKING_SYNC', 4)

CUsynchronizationPolicy = enum_CUsynchronizationPolicy_enum
enum_CUclusterSchedulingPolicy_enum = CEnum(ctypes.c_uint)
CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_DEFAULT', 0)
CU_CLUSTER_SCHEDULING_POLICY_SPREAD = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_SPREAD', 1)
CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING', 2)

CUclusterSchedulingPolicy = enum_CUclusterSchedulingPolicy_enum
enum_CUlaunchMemSyncDomain_enum = CEnum(ctypes.c_uint)
CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = enum_CUlaunchMemSyncDomain_enum.define('CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT', 0)
CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = enum_CUlaunchMemSyncDomain_enum.define('CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE', 1)

CUlaunchMemSyncDomain = enum_CUlaunchMemSyncDomain_enum
class struct_CUlaunchMemSyncDomainMap_st(ctypes.Structure): pass
struct_CUlaunchMemSyncDomainMap_st._fields_ = [
  ('default_', ctypes.c_ubyte),
  ('remote', ctypes.c_ubyte),
]
CUlaunchMemSyncDomainMap = struct_CUlaunchMemSyncDomainMap_st
enum_CUlaunchAttributeID_enum = CEnum(ctypes.c_uint)
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

CUlaunchAttributeID = enum_CUlaunchAttributeID_enum
class union_CUlaunchAttributeValue_union(ctypes.Union): pass
class union_CUlaunchAttributeValue_union_clusterDim(ctypes.Structure): pass
union_CUlaunchAttributeValue_union_clusterDim._fields_ = [
  ('x', ctypes.c_uint),
  ('y', ctypes.c_uint),
  ('z', ctypes.c_uint),
]
class union_CUlaunchAttributeValue_union_programmaticEvent(ctypes.Structure): pass
union_CUlaunchAttributeValue_union_programmaticEvent._fields_ = [
  ('event', CUevent),
  ('flags', ctypes.c_int),
  ('triggerAtBlockStart', ctypes.c_int),
]
union_CUlaunchAttributeValue_union._fields_ = [
  ('pad', (ctypes.c_char * 64)),
  ('accessPolicyWindow', CUaccessPolicyWindow),
  ('cooperative', ctypes.c_int),
  ('syncPolicy', CUsynchronizationPolicy),
  ('clusterDim', union_CUlaunchAttributeValue_union_clusterDim),
  ('clusterSchedulingPolicyPreference', CUclusterSchedulingPolicy),
  ('programmaticStreamSerializationAllowed', ctypes.c_int),
  ('programmaticEvent', union_CUlaunchAttributeValue_union_programmaticEvent),
  ('priority', ctypes.c_int),
  ('memSyncDomainMap', CUlaunchMemSyncDomainMap),
  ('memSyncDomain', CUlaunchMemSyncDomain),
]
CUlaunchAttributeValue = union_CUlaunchAttributeValue_union
class struct_CUlaunchAttribute_st(ctypes.Structure): pass
struct_CUlaunchAttribute_st._fields_ = [
  ('id', CUlaunchAttributeID),
  ('pad', (ctypes.c_char * 4)),
  ('value', CUlaunchAttributeValue),
]
CUlaunchAttribute = struct_CUlaunchAttribute_st
class struct_CUlaunchConfig_st(ctypes.Structure): pass
struct_CUlaunchConfig_st._fields_ = [
  ('gridDimX', ctypes.c_uint),
  ('gridDimY', ctypes.c_uint),
  ('gridDimZ', ctypes.c_uint),
  ('blockDimX', ctypes.c_uint),
  ('blockDimY', ctypes.c_uint),
  ('blockDimZ', ctypes.c_uint),
  ('sharedMemBytes', ctypes.c_uint),
  ('hStream', CUstream),
  ('attrs', ctypes.POINTER(CUlaunchAttribute)),
  ('numAttrs', ctypes.c_uint),
]
CUlaunchConfig = struct_CUlaunchConfig_st
CUkernelNodeAttrID = enum_CUlaunchAttributeID_enum
CUkernelNodeAttrValue_v1 = union_CUlaunchAttributeValue_union
CUkernelNodeAttrValue = union_CUlaunchAttributeValue_union
enum_CUstreamCaptureStatus_enum = CEnum(ctypes.c_uint)
CU_STREAM_CAPTURE_STATUS_NONE = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_NONE', 0)
CU_STREAM_CAPTURE_STATUS_ACTIVE = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_ACTIVE', 1)
CU_STREAM_CAPTURE_STATUS_INVALIDATED = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_INVALIDATED', 2)

CUstreamCaptureStatus = enum_CUstreamCaptureStatus_enum
enum_CUstreamCaptureMode_enum = CEnum(ctypes.c_uint)
CU_STREAM_CAPTURE_MODE_GLOBAL = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_GLOBAL', 0)
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_THREAD_LOCAL', 1)
CU_STREAM_CAPTURE_MODE_RELAXED = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_RELAXED', 2)

CUstreamCaptureMode = enum_CUstreamCaptureMode_enum
CUstreamAttrID = enum_CUlaunchAttributeID_enum
CUstreamAttrValue_v1 = union_CUlaunchAttributeValue_union
CUstreamAttrValue = union_CUlaunchAttributeValue_union
enum_CUdriverProcAddress_flags_enum = CEnum(ctypes.c_uint)
CU_GET_PROC_ADDRESS_DEFAULT = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_DEFAULT', 0)
CU_GET_PROC_ADDRESS_LEGACY_STREAM = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_LEGACY_STREAM', 1)
CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM', 2)

CUdriverProcAddress_flags = enum_CUdriverProcAddress_flags_enum
enum_CUdriverProcAddressQueryResult_enum = CEnum(ctypes.c_uint)
CU_GET_PROC_ADDRESS_SUCCESS = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_SUCCESS', 0)
CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND', 1)
CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT', 2)

CUdriverProcAddressQueryResult = enum_CUdriverProcAddressQueryResult_enum
enum_CUexecAffinityType_enum = CEnum(ctypes.c_uint)
CU_EXEC_AFFINITY_TYPE_SM_COUNT = enum_CUexecAffinityType_enum.define('CU_EXEC_AFFINITY_TYPE_SM_COUNT', 0)
CU_EXEC_AFFINITY_TYPE_MAX = enum_CUexecAffinityType_enum.define('CU_EXEC_AFFINITY_TYPE_MAX', 1)

CUexecAffinityType = enum_CUexecAffinityType_enum
class struct_CUexecAffinitySmCount_st(ctypes.Structure): pass
struct_CUexecAffinitySmCount_st._fields_ = [
  ('val', ctypes.c_uint),
]
CUexecAffinitySmCount_v1 = struct_CUexecAffinitySmCount_st
CUexecAffinitySmCount = struct_CUexecAffinitySmCount_st
class struct_CUexecAffinityParam_st(ctypes.Structure): pass
class struct_CUexecAffinityParam_st_param(ctypes.Union): pass
struct_CUexecAffinityParam_st_param._fields_ = [
  ('smCount', CUexecAffinitySmCount),
]
struct_CUexecAffinityParam_st._fields_ = [
  ('type', CUexecAffinityType),
  ('param', struct_CUexecAffinityParam_st_param),
]
CUexecAffinityParam_v1 = struct_CUexecAffinityParam_st
CUexecAffinityParam = struct_CUexecAffinityParam_st
enum_CUlibraryOption_enum = CEnum(ctypes.c_uint)
CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = enum_CUlibraryOption_enum.define('CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE', 0)
CU_LIBRARY_BINARY_IS_PRESERVED = enum_CUlibraryOption_enum.define('CU_LIBRARY_BINARY_IS_PRESERVED', 1)
CU_LIBRARY_NUM_OPTIONS = enum_CUlibraryOption_enum.define('CU_LIBRARY_NUM_OPTIONS', 2)

CUlibraryOption = enum_CUlibraryOption_enum
class struct_CUlibraryHostUniversalFunctionAndDataTable_st(ctypes.Structure): pass
struct_CUlibraryHostUniversalFunctionAndDataTable_st._fields_ = [
  ('functionTable', ctypes.c_void_p),
  ('functionWindowSize', size_t),
  ('dataTable', ctypes.c_void_p),
  ('dataWindowSize', size_t),
]
CUlibraryHostUniversalFunctionAndDataTable = struct_CUlibraryHostUniversalFunctionAndDataTable_st
enum_cudaError_enum = CEnum(ctypes.c_uint)
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

CUresult = enum_cudaError_enum
enum_CUdevice_P2PAttribute_enum = CEnum(ctypes.c_uint)
CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK', 1)
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED', 2)
CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED', 3)
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED', 4)
CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED', 4)

CUdevice_P2PAttribute = enum_CUdevice_P2PAttribute_enum
CUstreamCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_CUstream_st), enum_cudaError_enum, ctypes.c_void_p)
CUoccupancyB2DSize = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_int)
class struct_CUDA_MEMCPY2D_st(ctypes.Structure): pass
struct_CUDA_MEMCPY2D_st._fields_ = [
  ('srcXInBytes', size_t),
  ('srcY', size_t),
  ('srcMemoryType', CUmemorytype),
  ('srcHost', ctypes.c_void_p),
  ('srcDevice', CUdeviceptr),
  ('srcArray', CUarray),
  ('srcPitch', size_t),
  ('dstXInBytes', size_t),
  ('dstY', size_t),
  ('dstMemoryType', CUmemorytype),
  ('dstHost', ctypes.c_void_p),
  ('dstDevice', CUdeviceptr),
  ('dstArray', CUarray),
  ('dstPitch', size_t),
  ('WidthInBytes', size_t),
  ('Height', size_t),
]
CUDA_MEMCPY2D_v2 = struct_CUDA_MEMCPY2D_st
CUDA_MEMCPY2D = struct_CUDA_MEMCPY2D_st
class struct_CUDA_MEMCPY3D_st(ctypes.Structure): pass
struct_CUDA_MEMCPY3D_st._fields_ = [
  ('srcXInBytes', size_t),
  ('srcY', size_t),
  ('srcZ', size_t),
  ('srcLOD', size_t),
  ('srcMemoryType', CUmemorytype),
  ('srcHost', ctypes.c_void_p),
  ('srcDevice', CUdeviceptr),
  ('srcArray', CUarray),
  ('reserved0', ctypes.c_void_p),
  ('srcPitch', size_t),
  ('srcHeight', size_t),
  ('dstXInBytes', size_t),
  ('dstY', size_t),
  ('dstZ', size_t),
  ('dstLOD', size_t),
  ('dstMemoryType', CUmemorytype),
  ('dstHost', ctypes.c_void_p),
  ('dstDevice', CUdeviceptr),
  ('dstArray', CUarray),
  ('reserved1', ctypes.c_void_p),
  ('dstPitch', size_t),
  ('dstHeight', size_t),
  ('WidthInBytes', size_t),
  ('Height', size_t),
  ('Depth', size_t),
]
CUDA_MEMCPY3D_v2 = struct_CUDA_MEMCPY3D_st
CUDA_MEMCPY3D = struct_CUDA_MEMCPY3D_st
class struct_CUDA_MEMCPY3D_PEER_st(ctypes.Structure): pass
struct_CUDA_MEMCPY3D_PEER_st._fields_ = [
  ('srcXInBytes', size_t),
  ('srcY', size_t),
  ('srcZ', size_t),
  ('srcLOD', size_t),
  ('srcMemoryType', CUmemorytype),
  ('srcHost', ctypes.c_void_p),
  ('srcDevice', CUdeviceptr),
  ('srcArray', CUarray),
  ('srcContext', CUcontext),
  ('srcPitch', size_t),
  ('srcHeight', size_t),
  ('dstXInBytes', size_t),
  ('dstY', size_t),
  ('dstZ', size_t),
  ('dstLOD', size_t),
  ('dstMemoryType', CUmemorytype),
  ('dstHost', ctypes.c_void_p),
  ('dstDevice', CUdeviceptr),
  ('dstArray', CUarray),
  ('dstContext', CUcontext),
  ('dstPitch', size_t),
  ('dstHeight', size_t),
  ('WidthInBytes', size_t),
  ('Height', size_t),
  ('Depth', size_t),
]
CUDA_MEMCPY3D_PEER_v1 = struct_CUDA_MEMCPY3D_PEER_st
CUDA_MEMCPY3D_PEER = struct_CUDA_MEMCPY3D_PEER_st
class struct_CUDA_ARRAY_DESCRIPTOR_st(ctypes.Structure): pass
struct_CUDA_ARRAY_DESCRIPTOR_st._fields_ = [
  ('Width', size_t),
  ('Height', size_t),
  ('Format', CUarray_format),
  ('NumChannels', ctypes.c_uint),
]
CUDA_ARRAY_DESCRIPTOR_v2 = struct_CUDA_ARRAY_DESCRIPTOR_st
CUDA_ARRAY_DESCRIPTOR = struct_CUDA_ARRAY_DESCRIPTOR_st
class struct_CUDA_ARRAY3D_DESCRIPTOR_st(ctypes.Structure): pass
struct_CUDA_ARRAY3D_DESCRIPTOR_st._fields_ = [
  ('Width', size_t),
  ('Height', size_t),
  ('Depth', size_t),
  ('Format', CUarray_format),
  ('NumChannels', ctypes.c_uint),
  ('Flags', ctypes.c_uint),
]
CUDA_ARRAY3D_DESCRIPTOR_v2 = struct_CUDA_ARRAY3D_DESCRIPTOR_st
CUDA_ARRAY3D_DESCRIPTOR = struct_CUDA_ARRAY3D_DESCRIPTOR_st
class struct_CUDA_ARRAY_SPARSE_PROPERTIES_st(ctypes.Structure): pass
class struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent(ctypes.Structure): pass
struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent._fields_ = [
  ('width', ctypes.c_uint),
  ('height', ctypes.c_uint),
  ('depth', ctypes.c_uint),
]
struct_CUDA_ARRAY_SPARSE_PROPERTIES_st._fields_ = [
  ('tileExtent', struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent),
  ('miptailFirstLevel', ctypes.c_uint),
  ('miptailSize', ctypes.c_ulonglong),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 4)),
]
CUDA_ARRAY_SPARSE_PROPERTIES_v1 = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
CUDA_ARRAY_SPARSE_PROPERTIES = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
class struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st(ctypes.Structure): pass
struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st._fields_ = [
  ('size', size_t),
  ('alignment', size_t),
  ('reserved', (ctypes.c_uint * 4)),
]
CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 = struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st
CUDA_ARRAY_MEMORY_REQUIREMENTS = struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st
class struct_CUDA_RESOURCE_DESC_st(ctypes.Structure): pass
class struct_CUDA_RESOURCE_DESC_st_res(ctypes.Union): pass
class struct_CUDA_RESOURCE_DESC_st_res_array(ctypes.Structure): pass
struct_CUDA_RESOURCE_DESC_st_res_array._fields_ = [
  ('hArray', CUarray),
]
class struct_CUDA_RESOURCE_DESC_st_res_mipmap(ctypes.Structure): pass
struct_CUDA_RESOURCE_DESC_st_res_mipmap._fields_ = [
  ('hMipmappedArray', CUmipmappedArray),
]
class struct_CUDA_RESOURCE_DESC_st_res_linear(ctypes.Structure): pass
struct_CUDA_RESOURCE_DESC_st_res_linear._fields_ = [
  ('devPtr', CUdeviceptr),
  ('format', CUarray_format),
  ('numChannels', ctypes.c_uint),
  ('sizeInBytes', size_t),
]
class struct_CUDA_RESOURCE_DESC_st_res_pitch2D(ctypes.Structure): pass
struct_CUDA_RESOURCE_DESC_st_res_pitch2D._fields_ = [
  ('devPtr', CUdeviceptr),
  ('format', CUarray_format),
  ('numChannels', ctypes.c_uint),
  ('width', size_t),
  ('height', size_t),
  ('pitchInBytes', size_t),
]
class struct_CUDA_RESOURCE_DESC_st_res_reserved(ctypes.Structure): pass
struct_CUDA_RESOURCE_DESC_st_res_reserved._fields_ = [
  ('reserved', (ctypes.c_int * 32)),
]
struct_CUDA_RESOURCE_DESC_st_res._fields_ = [
  ('array', struct_CUDA_RESOURCE_DESC_st_res_array),
  ('mipmap', struct_CUDA_RESOURCE_DESC_st_res_mipmap),
  ('linear', struct_CUDA_RESOURCE_DESC_st_res_linear),
  ('pitch2D', struct_CUDA_RESOURCE_DESC_st_res_pitch2D),
  ('reserved', struct_CUDA_RESOURCE_DESC_st_res_reserved),
]
struct_CUDA_RESOURCE_DESC_st._fields_ = [
  ('resType', CUresourcetype),
  ('res', struct_CUDA_RESOURCE_DESC_st_res),
  ('flags', ctypes.c_uint),
]
CUDA_RESOURCE_DESC_v1 = struct_CUDA_RESOURCE_DESC_st
CUDA_RESOURCE_DESC = struct_CUDA_RESOURCE_DESC_st
class struct_CUDA_TEXTURE_DESC_st(ctypes.Structure): pass
struct_CUDA_TEXTURE_DESC_st._fields_ = [
  ('addressMode', (CUaddress_mode * 3)),
  ('filterMode', CUfilter_mode),
  ('flags', ctypes.c_uint),
  ('maxAnisotropy', ctypes.c_uint),
  ('mipmapFilterMode', CUfilter_mode),
  ('mipmapLevelBias', ctypes.c_float),
  ('minMipmapLevelClamp', ctypes.c_float),
  ('maxMipmapLevelClamp', ctypes.c_float),
  ('borderColor', (ctypes.c_float * 4)),
  ('reserved', (ctypes.c_int * 12)),
]
CUDA_TEXTURE_DESC_v1 = struct_CUDA_TEXTURE_DESC_st
CUDA_TEXTURE_DESC = struct_CUDA_TEXTURE_DESC_st
enum_CUresourceViewFormat_enum = CEnum(ctypes.c_uint)
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

CUresourceViewFormat = enum_CUresourceViewFormat_enum
class struct_CUDA_RESOURCE_VIEW_DESC_st(ctypes.Structure): pass
struct_CUDA_RESOURCE_VIEW_DESC_st._fields_ = [
  ('format', CUresourceViewFormat),
  ('width', size_t),
  ('height', size_t),
  ('depth', size_t),
  ('firstMipmapLevel', ctypes.c_uint),
  ('lastMipmapLevel', ctypes.c_uint),
  ('firstLayer', ctypes.c_uint),
  ('lastLayer', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
CUDA_RESOURCE_VIEW_DESC_v1 = struct_CUDA_RESOURCE_VIEW_DESC_st
CUDA_RESOURCE_VIEW_DESC = struct_CUDA_RESOURCE_VIEW_DESC_st
class struct_CUtensorMap_st(ctypes.Structure): pass
struct_CUtensorMap_st._fields_ = [
  ('opaque', (cuuint64_t * 16)),
]
CUtensorMap = struct_CUtensorMap_st
enum_CUtensorMapDataType_enum = CEnum(ctypes.c_uint)
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

CUtensorMapDataType = enum_CUtensorMapDataType_enum
enum_CUtensorMapInterleave_enum = CEnum(ctypes.c_uint)
CU_TENSOR_MAP_INTERLEAVE_NONE = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_NONE', 0)
CU_TENSOR_MAP_INTERLEAVE_16B = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_16B', 1)
CU_TENSOR_MAP_INTERLEAVE_32B = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_32B', 2)

CUtensorMapInterleave = enum_CUtensorMapInterleave_enum
enum_CUtensorMapSwizzle_enum = CEnum(ctypes.c_uint)
CU_TENSOR_MAP_SWIZZLE_NONE = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_NONE', 0)
CU_TENSOR_MAP_SWIZZLE_32B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_32B', 1)
CU_TENSOR_MAP_SWIZZLE_64B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_64B', 2)
CU_TENSOR_MAP_SWIZZLE_128B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_128B', 3)

CUtensorMapSwizzle = enum_CUtensorMapSwizzle_enum
enum_CUtensorMapL2promotion_enum = CEnum(ctypes.c_uint)
CU_TENSOR_MAP_L2_PROMOTION_NONE = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_NONE', 0)
CU_TENSOR_MAP_L2_PROMOTION_L2_64B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_64B', 1)
CU_TENSOR_MAP_L2_PROMOTION_L2_128B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_128B', 2)
CU_TENSOR_MAP_L2_PROMOTION_L2_256B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_256B', 3)

CUtensorMapL2promotion = enum_CUtensorMapL2promotion_enum
enum_CUtensorMapFloatOOBfill_enum = CEnum(ctypes.c_uint)
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = enum_CUtensorMapFloatOOBfill_enum.define('CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE', 0)
CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA = enum_CUtensorMapFloatOOBfill_enum.define('CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA', 1)

CUtensorMapFloatOOBfill = enum_CUtensorMapFloatOOBfill_enum
class struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st(ctypes.Structure): pass
struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st._fields_ = [
  ('p2pToken', ctypes.c_ulonglong),
  ('vaSpaceToken', ctypes.c_uint),
]
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum = CEnum(ctypes.c_uint)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE', 0)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ', 1)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE', 3)

CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum
class struct_CUDA_LAUNCH_PARAMS_st(ctypes.Structure): pass
struct_CUDA_LAUNCH_PARAMS_st._fields_ = [
  ('function', CUfunction),
  ('gridDimX', ctypes.c_uint),
  ('gridDimY', ctypes.c_uint),
  ('gridDimZ', ctypes.c_uint),
  ('blockDimX', ctypes.c_uint),
  ('blockDimY', ctypes.c_uint),
  ('blockDimZ', ctypes.c_uint),
  ('sharedMemBytes', ctypes.c_uint),
  ('hStream', CUstream),
  ('kernelParams', ctypes.POINTER(ctypes.c_void_p)),
]
CUDA_LAUNCH_PARAMS_v1 = struct_CUDA_LAUNCH_PARAMS_st
CUDA_LAUNCH_PARAMS = struct_CUDA_LAUNCH_PARAMS_st
enum_CUexternalMemoryHandleType_enum = CEnum(ctypes.c_uint)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD', 1)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32', 2)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT', 3)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP', 4)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE', 5)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE', 6)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT', 7)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF', 8)

CUexternalMemoryHandleType = enum_CUexternalMemoryHandleType_enum
class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(ctypes.Structure): pass
class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle(ctypes.Union): pass
class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_win32(ctypes.Structure): pass
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_win32._fields_ = [
  ('handle', ctypes.c_void_p),
  ('name', ctypes.c_void_p),
]
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle._fields_ = [
  ('fd', ctypes.c_int),
  ('win32', struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_win32),
  ('nvSciBufObject', ctypes.c_void_p),
]
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st._fields_ = [
  ('type', CUexternalMemoryHandleType),
  ('handle', struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle),
  ('size', ctypes.c_ulonglong),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
CUDA_EXTERNAL_MEMORY_HANDLE_DESC = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
class struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(ctypes.Structure): pass
struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st._fields_ = [
  ('offset', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
CUDA_EXTERNAL_MEMORY_BUFFER_DESC = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
class struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st(ctypes.Structure): pass
struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st._fields_ = [
  ('offset', ctypes.c_ulonglong),
  ('arrayDesc', CUDA_ARRAY3D_DESCRIPTOR),
  ('numLevels', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
enum_CUexternalSemaphoreHandleType_enum = CEnum(ctypes.c_uint)
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

CUexternalSemaphoreHandleType = enum_CUexternalSemaphoreHandleType_enum
class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(ctypes.Structure): pass
class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle(ctypes.Union): pass
class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_win32(ctypes.Structure): pass
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_win32._fields_ = [
  ('handle', ctypes.c_void_p),
  ('name', ctypes.c_void_p),
]
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle._fields_ = [
  ('fd', ctypes.c_int),
  ('win32', struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_win32),
  ('nvSciSyncObj', ctypes.c_void_p),
]
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st._fields_ = [
  ('type', CUexternalSemaphoreHandleType),
  ('handle', struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(ctypes.Structure): pass
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params(ctypes.Structure): pass
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_fence(ctypes.Structure): pass
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_fence._fields_ = [
  ('value', ctypes.c_ulonglong),
]
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_nvSciSync(ctypes.Union): pass
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_nvSciSync._fields_ = [
  ('fence', ctypes.c_void_p),
  ('reserved', ctypes.c_ulonglong),
]
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_keyedMutex(ctypes.Structure): pass
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_keyedMutex._fields_ = [
  ('key', ctypes.c_ulonglong),
]
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params._fields_ = [
  ('fence', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_fence),
  ('nvSciSync', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_nvSciSync),
  ('keyedMutex', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_keyedMutex),
  ('reserved', (ctypes.c_uint * 12)),
]
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st._fields_ = [
  ('params', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(ctypes.Structure): pass
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params(ctypes.Structure): pass
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_fence(ctypes.Structure): pass
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_fence._fields_ = [
  ('value', ctypes.c_ulonglong),
]
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_nvSciSync(ctypes.Union): pass
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_nvSciSync._fields_ = [
  ('fence', ctypes.c_void_p),
  ('reserved', ctypes.c_ulonglong),
]
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_keyedMutex(ctypes.Structure): pass
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_keyedMutex._fields_ = [
  ('key', ctypes.c_ulonglong),
  ('timeoutMs', ctypes.c_uint),
]
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params._fields_ = [
  ('fence', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_fence),
  ('nvSciSync', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_nvSciSync),
  ('keyedMutex', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_keyedMutex),
  ('reserved', (ctypes.c_uint * 10)),
]
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st._fields_ = [
  ('params', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 16)),
]
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
class struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st(ctypes.Structure): pass
struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st._fields_ = [
  ('extSemArray', ctypes.POINTER(CUexternalSemaphore)),
  ('paramsArray', ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS)),
  ('numExtSems', ctypes.c_uint),
]
CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
class struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st(ctypes.Structure): pass
struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st._fields_ = [
  ('extSemArray', ctypes.POINTER(CUexternalSemaphore)),
  ('paramsArray', ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS)),
  ('numExtSems', ctypes.c_uint),
]
CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUDA_EXT_SEM_WAIT_NODE_PARAMS = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUmemGenericAllocationHandle_v1 = ctypes.c_ulonglong
CUmemGenericAllocationHandle = ctypes.c_ulonglong
enum_CUmemAllocationHandleType_enum = CEnum(ctypes.c_uint)
CU_MEM_HANDLE_TYPE_NONE = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_NONE', 0)
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR', 1)
CU_MEM_HANDLE_TYPE_WIN32 = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_WIN32', 2)
CU_MEM_HANDLE_TYPE_WIN32_KMT = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_WIN32_KMT', 4)
CU_MEM_HANDLE_TYPE_MAX = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_MAX', 2147483647)

CUmemAllocationHandleType = enum_CUmemAllocationHandleType_enum
enum_CUmemAccess_flags_enum = CEnum(ctypes.c_uint)
CU_MEM_ACCESS_FLAGS_PROT_NONE = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_NONE', 0)
CU_MEM_ACCESS_FLAGS_PROT_READ = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_READ', 1)
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_READWRITE', 3)
CU_MEM_ACCESS_FLAGS_PROT_MAX = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_MAX', 2147483647)

CUmemAccess_flags = enum_CUmemAccess_flags_enum
enum_CUmemLocationType_enum = CEnum(ctypes.c_uint)
CU_MEM_LOCATION_TYPE_INVALID = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_INVALID', 0)
CU_MEM_LOCATION_TYPE_DEVICE = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_DEVICE', 1)
CU_MEM_LOCATION_TYPE_MAX = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_MAX', 2147483647)

CUmemLocationType = enum_CUmemLocationType_enum
enum_CUmemAllocationType_enum = CEnum(ctypes.c_uint)
CU_MEM_ALLOCATION_TYPE_INVALID = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_INVALID', 0)
CU_MEM_ALLOCATION_TYPE_PINNED = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_PINNED', 1)
CU_MEM_ALLOCATION_TYPE_MAX = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_MAX', 2147483647)

CUmemAllocationType = enum_CUmemAllocationType_enum
enum_CUmemAllocationGranularity_flags_enum = CEnum(ctypes.c_uint)
CU_MEM_ALLOC_GRANULARITY_MINIMUM = enum_CUmemAllocationGranularity_flags_enum.define('CU_MEM_ALLOC_GRANULARITY_MINIMUM', 0)
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = enum_CUmemAllocationGranularity_flags_enum.define('CU_MEM_ALLOC_GRANULARITY_RECOMMENDED', 1)

CUmemAllocationGranularity_flags = enum_CUmemAllocationGranularity_flags_enum
enum_CUmemRangeHandleType_enum = CEnum(ctypes.c_uint)
CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = enum_CUmemRangeHandleType_enum.define('CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD', 1)
CU_MEM_RANGE_HANDLE_TYPE_MAX = enum_CUmemRangeHandleType_enum.define('CU_MEM_RANGE_HANDLE_TYPE_MAX', 2147483647)

CUmemRangeHandleType = enum_CUmemRangeHandleType_enum
enum_CUarraySparseSubresourceType_enum = CEnum(ctypes.c_uint)
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = enum_CUarraySparseSubresourceType_enum.define('CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL', 0)
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = enum_CUarraySparseSubresourceType_enum.define('CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL', 1)

CUarraySparseSubresourceType = enum_CUarraySparseSubresourceType_enum
enum_CUmemOperationType_enum = CEnum(ctypes.c_uint)
CU_MEM_OPERATION_TYPE_MAP = enum_CUmemOperationType_enum.define('CU_MEM_OPERATION_TYPE_MAP', 1)
CU_MEM_OPERATION_TYPE_UNMAP = enum_CUmemOperationType_enum.define('CU_MEM_OPERATION_TYPE_UNMAP', 2)

CUmemOperationType = enum_CUmemOperationType_enum
enum_CUmemHandleType_enum = CEnum(ctypes.c_uint)
CU_MEM_HANDLE_TYPE_GENERIC = enum_CUmemHandleType_enum.define('CU_MEM_HANDLE_TYPE_GENERIC', 0)

CUmemHandleType = enum_CUmemHandleType_enum
class struct_CUarrayMapInfo_st(ctypes.Structure): pass
class struct_CUarrayMapInfo_st_resource(ctypes.Union): pass
struct_CUarrayMapInfo_st_resource._fields_ = [
  ('mipmap', CUmipmappedArray),
  ('array', CUarray),
]
class struct_CUarrayMapInfo_st_subresource(ctypes.Union): pass
class struct_CUarrayMapInfo_st_subresource_sparseLevel(ctypes.Structure): pass
struct_CUarrayMapInfo_st_subresource_sparseLevel._fields_ = [
  ('level', ctypes.c_uint),
  ('layer', ctypes.c_uint),
  ('offsetX', ctypes.c_uint),
  ('offsetY', ctypes.c_uint),
  ('offsetZ', ctypes.c_uint),
  ('extentWidth', ctypes.c_uint),
  ('extentHeight', ctypes.c_uint),
  ('extentDepth', ctypes.c_uint),
]
class struct_CUarrayMapInfo_st_subresource_miptail(ctypes.Structure): pass
struct_CUarrayMapInfo_st_subresource_miptail._fields_ = [
  ('layer', ctypes.c_uint),
  ('offset', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
]
struct_CUarrayMapInfo_st_subresource._fields_ = [
  ('sparseLevel', struct_CUarrayMapInfo_st_subresource_sparseLevel),
  ('miptail', struct_CUarrayMapInfo_st_subresource_miptail),
]
class struct_CUarrayMapInfo_st_memHandle(ctypes.Union): pass
struct_CUarrayMapInfo_st_memHandle._fields_ = [
  ('memHandle', CUmemGenericAllocationHandle),
]
struct_CUarrayMapInfo_st._fields_ = [
  ('resourceType', CUresourcetype),
  ('resource', struct_CUarrayMapInfo_st_resource),
  ('subresourceType', CUarraySparseSubresourceType),
  ('subresource', struct_CUarrayMapInfo_st_subresource),
  ('memOperationType', CUmemOperationType),
  ('memHandleType', CUmemHandleType),
  ('memHandle', struct_CUarrayMapInfo_st_memHandle),
  ('offset', ctypes.c_ulonglong),
  ('deviceBitMask', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('reserved', (ctypes.c_uint * 2)),
]
CUarrayMapInfo_v1 = struct_CUarrayMapInfo_st
CUarrayMapInfo = struct_CUarrayMapInfo_st
class struct_CUmemLocation_st(ctypes.Structure): pass
struct_CUmemLocation_st._fields_ = [
  ('type', CUmemLocationType),
  ('id', ctypes.c_int),
]
CUmemLocation_v1 = struct_CUmemLocation_st
CUmemLocation = struct_CUmemLocation_st
enum_CUmemAllocationCompType_enum = CEnum(ctypes.c_uint)
CU_MEM_ALLOCATION_COMP_NONE = enum_CUmemAllocationCompType_enum.define('CU_MEM_ALLOCATION_COMP_NONE', 0)
CU_MEM_ALLOCATION_COMP_GENERIC = enum_CUmemAllocationCompType_enum.define('CU_MEM_ALLOCATION_COMP_GENERIC', 1)

CUmemAllocationCompType = enum_CUmemAllocationCompType_enum
class struct_CUmemAllocationProp_st(ctypes.Structure): pass
class struct_CUmemAllocationProp_st_allocFlags(ctypes.Structure): pass
struct_CUmemAllocationProp_st_allocFlags._fields_ = [
  ('compressionType', ctypes.c_ubyte),
  ('gpuDirectRDMACapable', ctypes.c_ubyte),
  ('usage', ctypes.c_ushort),
  ('reserved', (ctypes.c_ubyte * 4)),
]
struct_CUmemAllocationProp_st._fields_ = [
  ('type', CUmemAllocationType),
  ('requestedHandleTypes', CUmemAllocationHandleType),
  ('location', CUmemLocation),
  ('win32HandleMetaData', ctypes.c_void_p),
  ('allocFlags', struct_CUmemAllocationProp_st_allocFlags),
]
CUmemAllocationProp_v1 = struct_CUmemAllocationProp_st
CUmemAllocationProp = struct_CUmemAllocationProp_st
class struct_CUmemAccessDesc_st(ctypes.Structure): pass
struct_CUmemAccessDesc_st._fields_ = [
  ('location', CUmemLocation),
  ('flags', CUmemAccess_flags),
]
CUmemAccessDesc_v1 = struct_CUmemAccessDesc_st
CUmemAccessDesc = struct_CUmemAccessDesc_st
enum_CUgraphExecUpdateResult_enum = CEnum(ctypes.c_uint)
CU_GRAPH_EXEC_UPDATE_SUCCESS = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_SUCCESS', 0)
CU_GRAPH_EXEC_UPDATE_ERROR = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR', 1)
CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED', 2)
CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED', 3)
CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED', 4)
CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED', 5)
CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED', 6)
CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE', 7)
CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = enum_CUgraphExecUpdateResult_enum.define('CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED', 8)

CUgraphExecUpdateResult = enum_CUgraphExecUpdateResult_enum
class struct_CUgraphExecUpdateResultInfo_st(ctypes.Structure): pass
struct_CUgraphExecUpdateResultInfo_st._fields_ = [
  ('result', CUgraphExecUpdateResult),
  ('errorNode', CUgraphNode),
  ('errorFromNode', CUgraphNode),
]
CUgraphExecUpdateResultInfo_v1 = struct_CUgraphExecUpdateResultInfo_st
CUgraphExecUpdateResultInfo = struct_CUgraphExecUpdateResultInfo_st
enum_CUmemPool_attribute_enum = CEnum(ctypes.c_uint)
CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES', 1)
CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC', 2)
CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES', 3)
CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RELEASE_THRESHOLD', 4)
CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT', 5)
CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH', 6)
CU_MEMPOOL_ATTR_USED_MEM_CURRENT = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_USED_MEM_CURRENT', 7)
CU_MEMPOOL_ATTR_USED_MEM_HIGH = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_USED_MEM_HIGH', 8)

CUmemPool_attribute = enum_CUmemPool_attribute_enum
class struct_CUmemPoolProps_st(ctypes.Structure): pass
struct_CUmemPoolProps_st._fields_ = [
  ('allocType', CUmemAllocationType),
  ('handleTypes', CUmemAllocationHandleType),
  ('location', CUmemLocation),
  ('win32SecurityAttributes', ctypes.c_void_p),
  ('reserved', (ctypes.c_ubyte * 64)),
]
CUmemPoolProps_v1 = struct_CUmemPoolProps_st
CUmemPoolProps = struct_CUmemPoolProps_st
class struct_CUmemPoolPtrExportData_st(ctypes.Structure): pass
struct_CUmemPoolPtrExportData_st._fields_ = [
  ('reserved', (ctypes.c_ubyte * 64)),
]
CUmemPoolPtrExportData_v1 = struct_CUmemPoolPtrExportData_st
CUmemPoolPtrExportData = struct_CUmemPoolPtrExportData_st
class struct_CUDA_MEM_ALLOC_NODE_PARAMS_st(ctypes.Structure): pass
struct_CUDA_MEM_ALLOC_NODE_PARAMS_st._fields_ = [
  ('poolProps', CUmemPoolProps),
  ('accessDescs', ctypes.POINTER(CUmemAccessDesc)),
  ('accessDescCount', size_t),
  ('bytesize', size_t),
  ('dptr', CUdeviceptr),
]
CUDA_MEM_ALLOC_NODE_PARAMS = struct_CUDA_MEM_ALLOC_NODE_PARAMS_st
enum_CUgraphMem_attribute_enum = CEnum(ctypes.c_uint)
CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT', 0)
CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_USED_MEM_HIGH', 1)
CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT', 2)
CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH', 3)

CUgraphMem_attribute = enum_CUgraphMem_attribute_enum
enum_CUflushGPUDirectRDMAWritesOptions_enum = CEnum(ctypes.c_uint)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = enum_CUflushGPUDirectRDMAWritesOptions_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST', 1)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = enum_CUflushGPUDirectRDMAWritesOptions_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS', 2)

CUflushGPUDirectRDMAWritesOptions = enum_CUflushGPUDirectRDMAWritesOptions_enum
enum_CUGPUDirectRDMAWritesOrdering_enum = CEnum(ctypes.c_uint)
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE', 0)
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER', 100)
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES', 200)

CUGPUDirectRDMAWritesOrdering = enum_CUGPUDirectRDMAWritesOrdering_enum
enum_CUflushGPUDirectRDMAWritesScope_enum = CEnum(ctypes.c_uint)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = enum_CUflushGPUDirectRDMAWritesScope_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER', 100)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = enum_CUflushGPUDirectRDMAWritesScope_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES', 200)

CUflushGPUDirectRDMAWritesScope = enum_CUflushGPUDirectRDMAWritesScope_enum
enum_CUflushGPUDirectRDMAWritesTarget_enum = CEnum(ctypes.c_uint)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = enum_CUflushGPUDirectRDMAWritesTarget_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX', 0)

CUflushGPUDirectRDMAWritesTarget = enum_CUflushGPUDirectRDMAWritesTarget_enum
enum_CUgraphDebugDot_flags_enum = CEnum(ctypes.c_uint)
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

CUgraphDebugDot_flags = enum_CUgraphDebugDot_flags_enum
enum_CUuserObject_flags_enum = CEnum(ctypes.c_uint)
CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = enum_CUuserObject_flags_enum.define('CU_USER_OBJECT_NO_DESTRUCTOR_SYNC', 1)

CUuserObject_flags = enum_CUuserObject_flags_enum
enum_CUuserObjectRetain_flags_enum = CEnum(ctypes.c_uint)
CU_GRAPH_USER_OBJECT_MOVE = enum_CUuserObjectRetain_flags_enum.define('CU_GRAPH_USER_OBJECT_MOVE', 1)

CUuserObjectRetain_flags = enum_CUuserObjectRetain_flags_enum
enum_CUgraphInstantiate_flags_enum = CEnum(ctypes.c_uint)
CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH', 1)
CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD', 2)
CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH', 4)
CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY', 8)

CUgraphInstantiate_flags = enum_CUgraphInstantiate_flags_enum
# CUresult cuGetErrorString(CUresult error, const char **pStr)
try: (cuGetErrorString:=dll.cuGetErrorString).restype, cuGetErrorString.argtypes = CUresult, [CUresult, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# CUresult cuGetErrorName(CUresult error, const char **pStr)
try: (cuGetErrorName:=dll.cuGetErrorName).restype, cuGetErrorName.argtypes = CUresult, [CUresult, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# CUresult cuInit(unsigned int Flags)
try: (cuInit:=dll.cuInit).restype, cuInit.argtypes = CUresult, [ctypes.c_uint]
except AttributeError: pass

# CUresult cuDriverGetVersion(int *driverVersion)
try: (cuDriverGetVersion:=dll.cuDriverGetVersion).restype, cuDriverGetVersion.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# CUresult cuDeviceGet(CUdevice *device, int ordinal)
try: (cuDeviceGet:=dll.cuDeviceGet).restype, cuDeviceGet.argtypes = CUresult, [ctypes.POINTER(CUdevice), ctypes.c_int]
except AttributeError: pass

# CUresult cuDeviceGetCount(int *count)
try: (cuDeviceGetCount:=dll.cuDeviceGetCount).restype, cuDeviceGetCount.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
try: (cuDeviceGetName:=dll.cuDeviceGetName).restype, cuDeviceGetName.argtypes = CUresult, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev)
try: (cuDeviceGetUuid:=dll.cuDeviceGetUuid).restype, cuDeviceGetUuid.argtypes = CUresult, [ctypes.POINTER(CUuuid), CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev)
try: (cuDeviceGetUuid_v2:=dll.cuDeviceGetUuid_v2).restype, cuDeviceGetUuid_v2.argtypes = CUresult, [ctypes.POINTER(CUuuid), CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev)
try: (cuDeviceGetLuid:=dll.cuDeviceGetLuid).restype, cuDeviceGetLuid.argtypes = CUresult, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint), CUdevice]
except AttributeError: pass

# CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
try: (cuDeviceTotalMem_v2:=dll.cuDeviceTotalMem_v2).restype, cuDeviceTotalMem_v2.argtypes = CUresult, [ctypes.POINTER(size_t), CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned int numChannels, CUdevice dev)
try: (cuDeviceGetTexture1DLinearMaxWidth:=dll.cuDeviceGetTexture1DLinearMaxWidth).restype, cuDeviceGetTexture1DLinearMaxWidth.argtypes = CUresult, [ctypes.POINTER(size_t), CUarray_format, ctypes.c_uint, CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
try: (cuDeviceGetAttribute:=dll.cuDeviceGetAttribute).restype, cuDeviceGetAttribute.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUdevice_attribute, CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags)
try: (cuDeviceGetNvSciSyncAttributes:=dll.cuDeviceGetNvSciSyncAttributes).restype, cuDeviceGetNvSciSyncAttributes.argtypes = CUresult, [ctypes.c_void_p, CUdevice, ctypes.c_int]
except AttributeError: pass

# CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool)
try: (cuDeviceSetMemPool:=dll.cuDeviceSetMemPool).restype, cuDeviceSetMemPool.argtypes = CUresult, [CUdevice, CUmemoryPool]
except AttributeError: pass

# CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev)
try: (cuDeviceGetMemPool:=dll.cuDeviceGetMemPool).restype, cuDeviceGetMemPool.argtypes = CUresult, [ctypes.POINTER(CUmemoryPool), CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev)
try: (cuDeviceGetDefaultMemPool:=dll.cuDeviceGetDefaultMemPool).restype, cuDeviceGetDefaultMemPool.argtypes = CUresult, [ctypes.POINTER(CUmemoryPool), CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type, CUdevice dev)
try: (cuDeviceGetExecAffinitySupport:=dll.cuDeviceGetExecAffinitySupport).restype, cuDeviceGetExecAffinitySupport.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUexecAffinityType, CUdevice]
except AttributeError: pass

# CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope)
try: (cuFlushGPUDirectRDMAWrites:=dll.cuFlushGPUDirectRDMAWrites).restype, cuFlushGPUDirectRDMAWrites.argtypes = CUresult, [CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope]
except AttributeError: pass

# CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
try: (cuDeviceGetProperties:=dll.cuDeviceGetProperties).restype, cuDeviceGetProperties.argtypes = CUresult, [ctypes.POINTER(CUdevprop), CUdevice]
except AttributeError: pass

# CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
try: (cuDeviceComputeCapability:=dll.cuDeviceComputeCapability).restype, cuDeviceComputeCapability.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), CUdevice]
except AttributeError: pass

# CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
try: (cuDevicePrimaryCtxRetain:=dll.cuDevicePrimaryCtxRetain).restype, cuDevicePrimaryCtxRetain.argtypes = CUresult, [ctypes.POINTER(CUcontext), CUdevice]
except AttributeError: pass

# CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev)
try: (cuDevicePrimaryCtxRelease_v2:=dll.cuDevicePrimaryCtxRelease_v2).restype, cuDevicePrimaryCtxRelease_v2.argtypes = CUresult, [CUdevice]
except AttributeError: pass

# CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags)
try: (cuDevicePrimaryCtxSetFlags_v2:=dll.cuDevicePrimaryCtxSetFlags_v2).restype, cuDevicePrimaryCtxSetFlags_v2.argtypes = CUresult, [CUdevice, ctypes.c_uint]
except AttributeError: pass

# CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active)
try: (cuDevicePrimaryCtxGetState:=dll.cuDevicePrimaryCtxGetState).restype, cuDevicePrimaryCtxGetState.argtypes = CUresult, [CUdevice, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev)
try: (cuDevicePrimaryCtxReset_v2:=dll.cuDevicePrimaryCtxReset_v2).restype, cuDevicePrimaryCtxReset_v2.argtypes = CUresult, [CUdevice]
except AttributeError: pass

# CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
try: (cuCtxCreate_v2:=dll.cuCtxCreate_v2).restype, cuCtxCreate_v2.argtypes = CUresult, [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
except AttributeError: pass

# CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev)
try: (cuCtxCreate_v3:=dll.cuCtxCreate_v3).restype, cuCtxCreate_v3.argtypes = CUresult, [ctypes.POINTER(CUcontext), ctypes.POINTER(CUexecAffinityParam), ctypes.c_int, ctypes.c_uint, CUdevice]
except AttributeError: pass

# CUresult cuCtxDestroy_v2(CUcontext ctx)
try: (cuCtxDestroy_v2:=dll.cuCtxDestroy_v2).restype, cuCtxDestroy_v2.argtypes = CUresult, [CUcontext]
except AttributeError: pass

# CUresult cuCtxPushCurrent_v2(CUcontext ctx)
try: (cuCtxPushCurrent_v2:=dll.cuCtxPushCurrent_v2).restype, cuCtxPushCurrent_v2.argtypes = CUresult, [CUcontext]
except AttributeError: pass

# CUresult cuCtxPopCurrent_v2(CUcontext *pctx)
try: (cuCtxPopCurrent_v2:=dll.cuCtxPopCurrent_v2).restype, cuCtxPopCurrent_v2.argtypes = CUresult, [ctypes.POINTER(CUcontext)]
except AttributeError: pass

# CUresult cuCtxSetCurrent(CUcontext ctx)
try: (cuCtxSetCurrent:=dll.cuCtxSetCurrent).restype, cuCtxSetCurrent.argtypes = CUresult, [CUcontext]
except AttributeError: pass

# CUresult cuCtxGetCurrent(CUcontext *pctx)
try: (cuCtxGetCurrent:=dll.cuCtxGetCurrent).restype, cuCtxGetCurrent.argtypes = CUresult, [ctypes.POINTER(CUcontext)]
except AttributeError: pass

# CUresult cuCtxGetDevice(CUdevice *device)
try: (cuCtxGetDevice:=dll.cuCtxGetDevice).restype, cuCtxGetDevice.argtypes = CUresult, [ctypes.POINTER(CUdevice)]
except AttributeError: pass

# CUresult cuCtxGetFlags(unsigned int *flags)
try: (cuCtxGetFlags:=dll.cuCtxGetFlags).restype, cuCtxGetFlags.argtypes = CUresult, [ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# CUresult cuCtxGetId(CUcontext ctx, unsigned long long *ctxId)
try: (cuCtxGetId:=dll.cuCtxGetId).restype, cuCtxGetId.argtypes = CUresult, [CUcontext, ctypes.POINTER(ctypes.c_ulonglong)]
except AttributeError: pass

# CUresult cuCtxSynchronize(void)
try: (cuCtxSynchronize:=dll.cuCtxSynchronize).restype, cuCtxSynchronize.argtypes = CUresult, []
except AttributeError: pass

# CUresult cuCtxSetLimit(CUlimit limit, size_t value)
try: (cuCtxSetLimit:=dll.cuCtxSetLimit).restype, cuCtxSetLimit.argtypes = CUresult, [CUlimit, size_t]
except AttributeError: pass

# CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit)
try: (cuCtxGetLimit:=dll.cuCtxGetLimit).restype, cuCtxGetLimit.argtypes = CUresult, [ctypes.POINTER(size_t), CUlimit]
except AttributeError: pass

# CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig)
try: (cuCtxGetCacheConfig:=dll.cuCtxGetCacheConfig).restype, cuCtxGetCacheConfig.argtypes = CUresult, [ctypes.POINTER(CUfunc_cache)]
except AttributeError: pass

# CUresult cuCtxSetCacheConfig(CUfunc_cache config)
try: (cuCtxSetCacheConfig:=dll.cuCtxSetCacheConfig).restype, cuCtxSetCacheConfig.argtypes = CUresult, [CUfunc_cache]
except AttributeError: pass

# CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig)
try: (cuCtxGetSharedMemConfig:=dll.cuCtxGetSharedMemConfig).restype, cuCtxGetSharedMemConfig.argtypes = CUresult, [ctypes.POINTER(CUsharedconfig)]
except AttributeError: pass

# CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)
try: (cuCtxSetSharedMemConfig:=dll.cuCtxSetSharedMemConfig).restype, cuCtxSetSharedMemConfig.argtypes = CUresult, [CUsharedconfig]
except AttributeError: pass

# CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
try: (cuCtxGetApiVersion:=dll.cuCtxGetApiVersion).restype, cuCtxGetApiVersion.argtypes = CUresult, [CUcontext, ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
try: (cuCtxGetStreamPriorityRange:=dll.cuCtxGetStreamPriorityRange).restype, cuCtxGetStreamPriorityRange.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# CUresult cuCtxResetPersistingL2Cache(void)
try: (cuCtxResetPersistingL2Cache:=dll.cuCtxResetPersistingL2Cache).restype, cuCtxResetPersistingL2Cache.argtypes = CUresult, []
except AttributeError: pass

# CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type)
try: (cuCtxGetExecAffinity:=dll.cuCtxGetExecAffinity).restype, cuCtxGetExecAffinity.argtypes = CUresult, [ctypes.POINTER(CUexecAffinityParam), CUexecAffinityType]
except AttributeError: pass

# CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags)
try: (cuCtxAttach:=dll.cuCtxAttach).restype, cuCtxAttach.argtypes = CUresult, [ctypes.POINTER(CUcontext), ctypes.c_uint]
except AttributeError: pass

# CUresult cuCtxDetach(CUcontext ctx)
try: (cuCtxDetach:=dll.cuCtxDetach).restype, cuCtxDetach.argtypes = CUresult, [CUcontext]
except AttributeError: pass

# CUresult cuModuleLoad(CUmodule *module, const char *fname)
try: (cuModuleLoad:=dll.cuModuleLoad).restype, cuModuleLoad.argtypes = CUresult, [ctypes.POINTER(CUmodule), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuModuleLoadData(CUmodule *module, const void *image)
try: (cuModuleLoadData:=dll.cuModuleLoadData).restype, cuModuleLoadData.argtypes = CUresult, [ctypes.POINTER(CUmodule), ctypes.c_void_p]
except AttributeError: pass

# CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
try: (cuModuleLoadDataEx:=dll.cuModuleLoadDataEx).restype, cuModuleLoadDataEx.argtypes = CUresult, [ctypes.POINTER(CUmodule), ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
try: (cuModuleLoadFatBinary:=dll.cuModuleLoadFatBinary).restype, cuModuleLoadFatBinary.argtypes = CUresult, [ctypes.POINTER(CUmodule), ctypes.c_void_p]
except AttributeError: pass

# CUresult cuModuleUnload(CUmodule hmod)
try: (cuModuleUnload:=dll.cuModuleUnload).restype, cuModuleUnload.argtypes = CUresult, [CUmodule]
except AttributeError: pass

enum_CUmoduleLoadingMode_enum = CEnum(ctypes.c_uint)
CU_MODULE_EAGER_LOADING = enum_CUmoduleLoadingMode_enum.define('CU_MODULE_EAGER_LOADING', 1)
CU_MODULE_LAZY_LOADING = enum_CUmoduleLoadingMode_enum.define('CU_MODULE_LAZY_LOADING', 2)

CUmoduleLoadingMode = enum_CUmoduleLoadingMode_enum
# CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode)
try: (cuModuleGetLoadingMode:=dll.cuModuleGetLoadingMode).restype, cuModuleGetLoadingMode.argtypes = CUresult, [ctypes.POINTER(CUmoduleLoadingMode)]
except AttributeError: pass

# CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
try: (cuModuleGetFunction:=dll.cuModuleGetFunction).restype, cuModuleGetFunction.argtypes = CUresult, [ctypes.POINTER(CUfunction), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name)
try: (cuModuleGetGlobal_v2:=dll.cuModuleGetGlobal_v2).restype, cuModuleGetGlobal_v2.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), ctypes.POINTER(size_t), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut)
try: (cuLinkCreate_v2:=dll.cuLinkCreate_v2).restype, cuLinkCreate_v2.argtypes = CUresult, [ctypes.c_uint, ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUlinkState)]
except AttributeError: pass

# CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues)
try: (cuLinkAddData_v2:=dll.cuLinkAddData_v2).restype, cuLinkAddData_v2.argtypes = CUresult, [CUlinkState, CUjitInputType, ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint, ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues)
try: (cuLinkAddFile_v2:=dll.cuLinkAddFile_v2).restype, cuLinkAddFile_v2.argtypes = CUresult, [CUlinkState, CUjitInputType, ctypes.POINTER(ctypes.c_char), ctypes.c_uint, ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut)
try: (cuLinkComplete:=dll.cuLinkComplete).restype, cuLinkComplete.argtypes = CUresult, [CUlinkState, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuLinkDestroy(CUlinkState state)
try: (cuLinkDestroy:=dll.cuLinkDestroy).restype, cuLinkDestroy.argtypes = CUresult, [CUlinkState]
except AttributeError: pass

# CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
try: (cuModuleGetTexRef:=dll.cuModuleGetTexRef).restype, cuModuleGetTexRef.argtypes = CUresult, [ctypes.POINTER(CUtexref), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name)
try: (cuModuleGetSurfRef:=dll.cuModuleGetSurfRef).restype, cuModuleGetSurfRef.argtypes = CUresult, [ctypes.POINTER(CUsurfref), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions)
try: (cuLibraryLoadData:=dll.cuLibraryLoadData).restype, cuLibraryLoadData.argtypes = CUresult, [ctypes.POINTER(CUlibrary), ctypes.c_void_p, ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.POINTER(CUlibraryOption), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
except AttributeError: pass

# CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions)
try: (cuLibraryLoadFromFile:=dll.cuLibraryLoadFromFile).restype, cuLibraryLoadFromFile.argtypes = CUresult, [ctypes.POINTER(CUlibrary), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.POINTER(CUlibraryOption), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
except AttributeError: pass

# CUresult cuLibraryUnload(CUlibrary library)
try: (cuLibraryUnload:=dll.cuLibraryUnload).restype, cuLibraryUnload.argtypes = CUresult, [CUlibrary]
except AttributeError: pass

# CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name)
try: (cuLibraryGetKernel:=dll.cuLibraryGetKernel).restype, cuLibraryGetKernel.argtypes = CUresult, [ctypes.POINTER(CUkernel), CUlibrary, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library)
try: (cuLibraryGetModule:=dll.cuLibraryGetModule).restype, cuLibraryGetModule.argtypes = CUresult, [ctypes.POINTER(CUmodule), CUlibrary]
except AttributeError: pass

# CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel)
try: (cuKernelGetFunction:=dll.cuKernelGetFunction).restype, cuKernelGetFunction.argtypes = CUresult, [ctypes.POINTER(CUfunction), CUkernel]
except AttributeError: pass

# CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name)
try: (cuLibraryGetGlobal:=dll.cuLibraryGetGlobal).restype, cuLibraryGetGlobal.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), ctypes.POINTER(size_t), CUlibrary, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name)
try: (cuLibraryGetManaged:=dll.cuLibraryGetManaged).restype, cuLibraryGetManaged.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), ctypes.POINTER(size_t), CUlibrary, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library, const char *symbol)
try: (cuLibraryGetUnifiedFunction:=dll.cuLibraryGetUnifiedFunction).restype, cuLibraryGetUnifiedFunction.argtypes = CUresult, [ctypes.POINTER(ctypes.c_void_p), CUlibrary, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev)
try: (cuKernelGetAttribute:=dll.cuKernelGetAttribute).restype, cuKernelGetAttribute.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUfunction_attribute, CUkernel, CUdevice]
except AttributeError: pass

# CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev)
try: (cuKernelSetAttribute:=dll.cuKernelSetAttribute).restype, cuKernelSetAttribute.argtypes = CUresult, [CUfunction_attribute, ctypes.c_int, CUkernel, CUdevice]
except AttributeError: pass

# CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev)
try: (cuKernelSetCacheConfig:=dll.cuKernelSetCacheConfig).restype, cuKernelSetCacheConfig.argtypes = CUresult, [CUkernel, CUfunc_cache, CUdevice]
except AttributeError: pass

# CUresult cuMemGetInfo_v2(size_t *free, size_t *total)
try: (cuMemGetInfo_v2:=dll.cuMemGetInfo_v2).restype, cuMemGetInfo_v2.argtypes = CUresult, [ctypes.POINTER(size_t), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
try: (cuMemAlloc_v2:=dll.cuMemAlloc_v2).restype, cuMemAlloc_v2.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), size_t]
except AttributeError: pass

# CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
try: (cuMemAllocPitch_v2:=dll.cuMemAllocPitch_v2).restype, cuMemAllocPitch_v2.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), ctypes.POINTER(size_t), size_t, size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemFree_v2(CUdeviceptr dptr)
try: (cuMemFree_v2:=dll.cuMemFree_v2).restype, cuMemFree_v2.argtypes = CUresult, [CUdeviceptr]
except AttributeError: pass

# CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr)
try: (cuMemGetAddressRange_v2:=dll.cuMemGetAddressRange_v2).restype, cuMemGetAddressRange_v2.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), ctypes.POINTER(size_t), CUdeviceptr]
except AttributeError: pass

# CUresult cuMemAllocHost_v2(void **pp, size_t bytesize)
try: (cuMemAllocHost_v2:=dll.cuMemAllocHost_v2).restype, cuMemAllocHost_v2.argtypes = CUresult, [ctypes.POINTER(ctypes.c_void_p), size_t]
except AttributeError: pass

# CUresult cuMemFreeHost(void *p)
try: (cuMemFreeHost:=dll.cuMemFreeHost).restype, cuMemFreeHost.argtypes = CUresult, [ctypes.c_void_p]
except AttributeError: pass

# CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags)
try: (cuMemHostAlloc:=dll.cuMemHostAlloc).restype, cuMemHostAlloc.argtypes = CUresult, [ctypes.POINTER(ctypes.c_void_p), size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags)
try: (cuMemHostGetDevicePointer_v2:=dll.cuMemHostGetDevicePointer_v2).restype, cuMemHostGetDevicePointer_v2.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p)
try: (cuMemHostGetFlags:=dll.cuMemHostGetFlags).restype, cuMemHostGetFlags.argtypes = CUresult, [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
except AttributeError: pass

# CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
try: (cuMemAllocManaged:=dll.cuMemAllocManaged).restype, cuMemAllocManaged.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId)
try: (cuDeviceGetByPCIBusId:=dll.cuDeviceGetByPCIBusId).restype, cuDeviceGetByPCIBusId.argtypes = CUresult, [ctypes.POINTER(CUdevice), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev)
try: (cuDeviceGetPCIBusId:=dll.cuDeviceGetPCIBusId).restype, cuDeviceGetPCIBusId.argtypes = CUresult, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, CUdevice]
except AttributeError: pass

# CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event)
try: (cuIpcGetEventHandle:=dll.cuIpcGetEventHandle).restype, cuIpcGetEventHandle.argtypes = CUresult, [ctypes.POINTER(CUipcEventHandle), CUevent]
except AttributeError: pass

# CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle)
try: (cuIpcOpenEventHandle:=dll.cuIpcOpenEventHandle).restype, cuIpcOpenEventHandle.argtypes = CUresult, [ctypes.POINTER(CUevent), CUipcEventHandle]
except AttributeError: pass

# CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr)
try: (cuIpcGetMemHandle:=dll.cuIpcGetMemHandle).restype, cuIpcGetMemHandle.argtypes = CUresult, [ctypes.POINTER(CUipcMemHandle), CUdeviceptr]
except AttributeError: pass

# CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags)
try: (cuIpcOpenMemHandle_v2:=dll.cuIpcOpenMemHandle_v2).restype, cuIpcOpenMemHandle_v2.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), CUipcMemHandle, ctypes.c_uint]
except AttributeError: pass

# CUresult cuIpcCloseMemHandle(CUdeviceptr dptr)
try: (cuIpcCloseMemHandle:=dll.cuIpcCloseMemHandle).restype, cuIpcCloseMemHandle.argtypes = CUresult, [CUdeviceptr]
except AttributeError: pass

# CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags)
try: (cuMemHostRegister_v2:=dll.cuMemHostRegister_v2).restype, cuMemHostRegister_v2.argtypes = CUresult, [ctypes.c_void_p, size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemHostUnregister(void *p)
try: (cuMemHostUnregister:=dll.cuMemHostUnregister).restype, cuMemHostUnregister.argtypes = CUresult, [ctypes.c_void_p]
except AttributeError: pass

# CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
try: (cuMemcpy_ptds:=dll.cuMemcpy_ptds).restype, cuMemcpy_ptds.argtypes = CUresult, [CUdeviceptr, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemcpyPeer_ptds(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
try: (cuMemcpyPeer_ptds:=dll.cuMemcpyPeer_ptds).restype, cuMemcpyPeer_ptds.argtypes = CUresult, [CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t]
except AttributeError: pass

# CUresult cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
try: (cuMemcpyHtoD_v2_ptds:=dll.cuMemcpyHtoD_v2_ptds).restype, cuMemcpyHtoD_v2_ptds.argtypes = CUresult, [CUdeviceptr, ctypes.c_void_p, size_t]
except AttributeError: pass

# CUresult cuMemcpyDtoH_v2_ptds(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
try: (cuMemcpyDtoH_v2_ptds:=dll.cuMemcpyDtoH_v2_ptds).restype, cuMemcpyDtoH_v2_ptds.argtypes = CUresult, [ctypes.c_void_p, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
try: (cuMemcpyDtoD_v2_ptds:=dll.cuMemcpyDtoD_v2_ptds).restype, cuMemcpyDtoD_v2_ptds.argtypes = CUresult, [CUdeviceptr, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemcpyDtoA_v2_ptds(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount)
try: (cuMemcpyDtoA_v2_ptds:=dll.cuMemcpyDtoA_v2_ptds).restype, cuMemcpyDtoA_v2_ptds.argtypes = CUresult, [CUarray, size_t, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemcpyAtoD_v2_ptds(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
try: (cuMemcpyAtoD_v2_ptds:=dll.cuMemcpyAtoD_v2_ptds).restype, cuMemcpyAtoD_v2_ptds.argtypes = CUresult, [CUdeviceptr, CUarray, size_t, size_t]
except AttributeError: pass

# CUresult cuMemcpyHtoA_v2_ptds(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount)
try: (cuMemcpyHtoA_v2_ptds:=dll.cuMemcpyHtoA_v2_ptds).restype, cuMemcpyHtoA_v2_ptds.argtypes = CUresult, [CUarray, size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

# CUresult cuMemcpyAtoH_v2_ptds(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
try: (cuMemcpyAtoH_v2_ptds:=dll.cuMemcpyAtoH_v2_ptds).restype, cuMemcpyAtoH_v2_ptds.argtypes = CUresult, [ctypes.c_void_p, CUarray, size_t, size_t]
except AttributeError: pass

# CUresult cuMemcpyAtoA_v2_ptds(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount)
try: (cuMemcpyAtoA_v2_ptds:=dll.cuMemcpyAtoA_v2_ptds).restype, cuMemcpyAtoA_v2_ptds.argtypes = CUresult, [CUarray, size_t, CUarray, size_t, size_t]
except AttributeError: pass

# CUresult cuMemcpy2D_v2_ptds(const CUDA_MEMCPY2D *pCopy)
try: (cuMemcpy2D_v2_ptds:=dll.cuMemcpy2D_v2_ptds).restype, cuMemcpy2D_v2_ptds.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D)]
except AttributeError: pass

# CUresult cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *pCopy)
try: (cuMemcpy2DUnaligned_v2_ptds:=dll.cuMemcpy2DUnaligned_v2_ptds).restype, cuMemcpy2DUnaligned_v2_ptds.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D)]
except AttributeError: pass

# CUresult cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *pCopy)
try: (cuMemcpy3D_v2_ptds:=dll.cuMemcpy3D_v2_ptds).restype, cuMemcpy3D_v2_ptds.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D)]
except AttributeError: pass

# CUresult cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy)
try: (cuMemcpy3DPeer_ptds:=dll.cuMemcpy3DPeer_ptds).restype, cuMemcpy3DPeer_ptds.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D_PEER)]
except AttributeError: pass

# CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
try: (cuMemcpyAsync_ptsz:=dll.cuMemcpyAsync_ptsz).restype, cuMemcpyAsync_ptsz.argtypes = CUresult, [CUdeviceptr, CUdeviceptr, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyPeerAsync_ptsz(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
try: (cuMemcpyPeerAsync_ptsz:=dll.cuMemcpyPeerAsync_ptsz).restype, cuMemcpyPeerAsync_ptsz.argtypes = CUresult, [CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
try: (cuMemcpyHtoDAsync_v2_ptsz:=dll.cuMemcpyHtoDAsync_v2_ptsz).restype, cuMemcpyHtoDAsync_v2_ptsz.argtypes = CUresult, [CUdeviceptr, ctypes.c_void_p, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyDtoHAsync_v2_ptsz(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
try: (cuMemcpyDtoHAsync_v2_ptsz:=dll.cuMemcpyDtoHAsync_v2_ptsz).restype, cuMemcpyDtoHAsync_v2_ptsz.argtypes = CUresult, [ctypes.c_void_p, CUdeviceptr, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
try: (cuMemcpyDtoDAsync_v2_ptsz:=dll.cuMemcpyDtoDAsync_v2_ptsz).restype, cuMemcpyDtoDAsync_v2_ptsz.argtypes = CUresult, [CUdeviceptr, CUdeviceptr, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyHtoAAsync_v2_ptsz(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream)
try: (cuMemcpyHtoAAsync_v2_ptsz:=dll.cuMemcpyHtoAAsync_v2_ptsz).restype, cuMemcpyHtoAAsync_v2_ptsz.argtypes = CUresult, [CUarray, size_t, ctypes.c_void_p, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyAtoHAsync_v2_ptsz(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream)
try: (cuMemcpyAtoHAsync_v2_ptsz:=dll.cuMemcpyAtoHAsync_v2_ptsz).restype, cuMemcpyAtoHAsync_v2_ptsz.argtypes = CUresult, [ctypes.c_void_p, CUarray, size_t, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
try: (cuMemcpy2DAsync_v2_ptsz:=dll.cuMemcpy2DAsync_v2_ptsz).restype, cuMemcpy2DAsync_v2_ptsz.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D), CUstream]
except AttributeError: pass

# CUresult cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
try: (cuMemcpy3DAsync_v2_ptsz:=dll.cuMemcpy3DAsync_v2_ptsz).restype, cuMemcpy3DAsync_v2_ptsz.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D), CUstream]
except AttributeError: pass

# CUresult cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream)
try: (cuMemcpy3DPeerAsync_ptsz:=dll.cuMemcpy3DPeerAsync_ptsz).restype, cuMemcpy3DPeerAsync_ptsz.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D_PEER), CUstream]
except AttributeError: pass

# CUresult cuMemsetD8_v2_ptds(CUdeviceptr dstDevice, unsigned char uc, size_t N)
try: (cuMemsetD8_v2_ptds:=dll.cuMemsetD8_v2_ptds).restype, cuMemsetD8_v2_ptds.argtypes = CUresult, [CUdeviceptr, ctypes.c_ubyte, size_t]
except AttributeError: pass

# CUresult cuMemsetD16_v2_ptds(CUdeviceptr dstDevice, unsigned short us, size_t N)
try: (cuMemsetD16_v2_ptds:=dll.cuMemsetD16_v2_ptds).restype, cuMemsetD16_v2_ptds.argtypes = CUresult, [CUdeviceptr, ctypes.c_ushort, size_t]
except AttributeError: pass

# CUresult cuMemsetD32_v2_ptds(CUdeviceptr dstDevice, unsigned int ui, size_t N)
try: (cuMemsetD32_v2_ptds:=dll.cuMemsetD32_v2_ptds).restype, cuMemsetD32_v2_ptds.argtypes = CUresult, [CUdeviceptr, ctypes.c_uint, size_t]
except AttributeError: pass

# CUresult cuMemsetD2D8_v2_ptds(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
try: (cuMemsetD2D8_v2_ptds:=dll.cuMemsetD2D8_v2_ptds).restype, cuMemsetD2D8_v2_ptds.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t]
except AttributeError: pass

# CUresult cuMemsetD2D16_v2_ptds(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
try: (cuMemsetD2D16_v2_ptds:=dll.cuMemsetD2D16_v2_ptds).restype, cuMemsetD2D16_v2_ptds.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_ushort, size_t, size_t]
except AttributeError: pass

# CUresult cuMemsetD2D32_v2_ptds(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
try: (cuMemsetD2D32_v2_ptds:=dll.cuMemsetD2D32_v2_ptds).restype, cuMemsetD2D32_v2_ptds.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_uint, size_t, size_t]
except AttributeError: pass

# CUresult cuMemsetD8Async_ptsz(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
try: (cuMemsetD8Async_ptsz:=dll.cuMemsetD8Async_ptsz).restype, cuMemsetD8Async_ptsz.argtypes = CUresult, [CUdeviceptr, ctypes.c_ubyte, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD16Async_ptsz(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
try: (cuMemsetD16Async_ptsz:=dll.cuMemsetD16Async_ptsz).restype, cuMemsetD16Async_ptsz.argtypes = CUresult, [CUdeviceptr, ctypes.c_ushort, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD32Async_ptsz(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
try: (cuMemsetD32Async_ptsz:=dll.cuMemsetD32Async_ptsz).restype, cuMemsetD32Async_ptsz.argtypes = CUresult, [CUdeviceptr, ctypes.c_uint, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD2D8Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
try: (cuMemsetD2D8Async_ptsz:=dll.cuMemsetD2D8Async_ptsz).restype, cuMemsetD2D8Async_ptsz.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD2D16Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
try: (cuMemsetD2D16Async_ptsz:=dll.cuMemsetD2D16Async_ptsz).restype, cuMemsetD2D16Async_ptsz.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_ushort, size_t, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD2D32Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
try: (cuMemsetD2D32Async_ptsz:=dll.cuMemsetD2D32Async_ptsz).restype, cuMemsetD2D32Async_ptsz.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_uint, size_t, size_t, CUstream]
except AttributeError: pass

# CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
try: (cuArrayCreate_v2:=dll.cuArrayCreate_v2).restype, cuArrayCreate_v2.argtypes = CUresult, [ctypes.POINTER(CUarray), ctypes.POINTER(CUDA_ARRAY_DESCRIPTOR)]
except AttributeError: pass

# CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
try: (cuArrayGetDescriptor_v2:=dll.cuArrayGetDescriptor_v2).restype, cuArrayGetDescriptor_v2.argtypes = CUresult, [ctypes.POINTER(CUDA_ARRAY_DESCRIPTOR), CUarray]
except AttributeError: pass

# CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUarray array)
try: (cuArrayGetSparseProperties:=dll.cuArrayGetSparseProperties).restype, cuArrayGetSparseProperties.argtypes = CUresult, [ctypes.POINTER(CUDA_ARRAY_SPARSE_PROPERTIES), CUarray]
except AttributeError: pass

# CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap)
try: (cuMipmappedArrayGetSparseProperties:=dll.cuMipmappedArrayGetSparseProperties).restype, cuMipmappedArrayGetSparseProperties.argtypes = CUresult, [ctypes.POINTER(CUDA_ARRAY_SPARSE_PROPERTIES), CUmipmappedArray]
except AttributeError: pass

# CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUarray array, CUdevice device)
try: (cuArrayGetMemoryRequirements:=dll.cuArrayGetMemoryRequirements).restype, cuArrayGetMemoryRequirements.argtypes = CUresult, [ctypes.POINTER(CUDA_ARRAY_MEMORY_REQUIREMENTS), CUarray, CUdevice]
except AttributeError: pass

# CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap, CUdevice device)
try: (cuMipmappedArrayGetMemoryRequirements:=dll.cuMipmappedArrayGetMemoryRequirements).restype, cuMipmappedArrayGetMemoryRequirements.argtypes = CUresult, [ctypes.POINTER(CUDA_ARRAY_MEMORY_REQUIREMENTS), CUmipmappedArray, CUdevice]
except AttributeError: pass

# CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx)
try: (cuArrayGetPlane:=dll.cuArrayGetPlane).restype, cuArrayGetPlane.argtypes = CUresult, [ctypes.POINTER(CUarray), CUarray, ctypes.c_uint]
except AttributeError: pass

# CUresult cuArrayDestroy(CUarray hArray)
try: (cuArrayDestroy:=dll.cuArrayDestroy).restype, cuArrayDestroy.argtypes = CUresult, [CUarray]
except AttributeError: pass

# CUresult cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
try: (cuArray3DCreate_v2:=dll.cuArray3DCreate_v2).restype, cuArray3DCreate_v2.argtypes = CUresult, [ctypes.POINTER(CUarray), ctypes.POINTER(CUDA_ARRAY3D_DESCRIPTOR)]
except AttributeError: pass

# CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray)
try: (cuArray3DGetDescriptor_v2:=dll.cuArray3DGetDescriptor_v2).restype, cuArray3DGetDescriptor_v2.argtypes = CUresult, [ctypes.POINTER(CUDA_ARRAY3D_DESCRIPTOR), CUarray]
except AttributeError: pass

# CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels)
try: (cuMipmappedArrayCreate:=dll.cuMipmappedArrayCreate).restype, cuMipmappedArrayCreate.argtypes = CUresult, [ctypes.POINTER(CUmipmappedArray), ctypes.POINTER(CUDA_ARRAY3D_DESCRIPTOR), ctypes.c_uint]
except AttributeError: pass

# CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level)
try: (cuMipmappedArrayGetLevel:=dll.cuMipmappedArrayGetLevel).restype, cuMipmappedArrayGetLevel.argtypes = CUresult, [ctypes.POINTER(CUarray), CUmipmappedArray, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
try: (cuMipmappedArrayDestroy:=dll.cuMipmappedArrayDestroy).restype, cuMipmappedArrayDestroy.argtypes = CUresult, [CUmipmappedArray]
except AttributeError: pass

# CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags)
try: (cuMemGetHandleForAddressRange:=dll.cuMemGetHandleForAddressRange).restype, cuMemGetHandleForAddressRange.argtypes = CUresult, [ctypes.c_void_p, CUdeviceptr, size_t, CUmemRangeHandleType, ctypes.c_ulonglong]
except AttributeError: pass

# CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)
try: (cuMemAddressReserve:=dll.cuMemAddressReserve).restype, cuMemAddressReserve.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), size_t, size_t, CUdeviceptr, ctypes.c_ulonglong]
except AttributeError: pass

# CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
try: (cuMemAddressFree:=dll.cuMemAddressFree).restype, cuMemAddressFree.argtypes = CUresult, [CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags)
try: (cuMemCreate:=dll.cuMemCreate).restype, cuMemCreate.argtypes = CUresult, [ctypes.POINTER(CUmemGenericAllocationHandle), size_t, ctypes.POINTER(CUmemAllocationProp), ctypes.c_ulonglong]
except AttributeError: pass

# CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
try: (cuMemRelease:=dll.cuMemRelease).restype, cuMemRelease.argtypes = CUresult, [CUmemGenericAllocationHandle]
except AttributeError: pass

# CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)
try: (cuMemMap:=dll.cuMemMap).restype, cuMemMap.argtypes = CUresult, [CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, ctypes.c_ulonglong]
except AttributeError: pass

# CUresult cuMemMapArrayAsync_ptsz(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream)
try: (cuMemMapArrayAsync_ptsz:=dll.cuMemMapArrayAsync_ptsz).restype, cuMemMapArrayAsync_ptsz.argtypes = CUresult, [ctypes.POINTER(CUarrayMapInfo), ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
try: (cuMemUnmap:=dll.cuMemUnmap).restype, cuMemUnmap.argtypes = CUresult, [CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count)
try: (cuMemSetAccess:=dll.cuMemSetAccess).restype, cuMemSetAccess.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.POINTER(CUmemAccessDesc), size_t]
except AttributeError: pass

# CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr)
try: (cuMemGetAccess:=dll.cuMemGetAccess).restype, cuMemGetAccess.argtypes = CUresult, [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(CUmemLocation), CUdeviceptr]
except AttributeError: pass

# CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags)
try: (cuMemExportToShareableHandle:=dll.cuMemExportToShareableHandle).restype, cuMemExportToShareableHandle.argtypes = CUresult, [ctypes.c_void_p, CUmemGenericAllocationHandle, CUmemAllocationHandleType, ctypes.c_ulonglong]
except AttributeError: pass

# CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType)
try: (cuMemImportFromShareableHandle:=dll.cuMemImportFromShareableHandle).restype, cuMemImportFromShareableHandle.argtypes = CUresult, [ctypes.POINTER(CUmemGenericAllocationHandle), ctypes.c_void_p, CUmemAllocationHandleType]
except AttributeError: pass

# CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option)
try: (cuMemGetAllocationGranularity:=dll.cuMemGetAllocationGranularity).restype, cuMemGetAllocationGranularity.argtypes = CUresult, [ctypes.POINTER(size_t), ctypes.POINTER(CUmemAllocationProp), CUmemAllocationGranularity_flags]
except AttributeError: pass

# CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle)
try: (cuMemGetAllocationPropertiesFromHandle:=dll.cuMemGetAllocationPropertiesFromHandle).restype, cuMemGetAllocationPropertiesFromHandle.argtypes = CUresult, [ctypes.POINTER(CUmemAllocationProp), CUmemGenericAllocationHandle]
except AttributeError: pass

# CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr)
try: (cuMemRetainAllocationHandle:=dll.cuMemRetainAllocationHandle).restype, cuMemRetainAllocationHandle.argtypes = CUresult, [ctypes.POINTER(CUmemGenericAllocationHandle), ctypes.c_void_p]
except AttributeError: pass

# CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream)
try: (cuMemFreeAsync_ptsz:=dll.cuMemFreeAsync_ptsz).restype, cuMemFreeAsync_ptsz.argtypes = CUresult, [CUdeviceptr, CUstream]
except AttributeError: pass

# CUresult cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUstream hStream)
try: (cuMemAllocAsync_ptsz:=dll.cuMemAllocAsync_ptsz).restype, cuMemAllocAsync_ptsz.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), size_t, CUstream]
except AttributeError: pass

# CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep)
try: (cuMemPoolTrimTo:=dll.cuMemPoolTrimTo).restype, cuMemPoolTrimTo.argtypes = CUresult, [CUmemoryPool, size_t]
except AttributeError: pass

# CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value)
try: (cuMemPoolSetAttribute:=dll.cuMemPoolSetAttribute).restype, cuMemPoolSetAttribute.argtypes = CUresult, [CUmemoryPool, CUmemPool_attribute, ctypes.c_void_p]
except AttributeError: pass

# CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value)
try: (cuMemPoolGetAttribute:=dll.cuMemPoolGetAttribute).restype, cuMemPoolGetAttribute.argtypes = CUresult, [CUmemoryPool, CUmemPool_attribute, ctypes.c_void_p]
except AttributeError: pass

# CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count)
try: (cuMemPoolSetAccess:=dll.cuMemPoolSetAccess).restype, cuMemPoolSetAccess.argtypes = CUresult, [CUmemoryPool, ctypes.POINTER(CUmemAccessDesc), size_t]
except AttributeError: pass

# CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location)
try: (cuMemPoolGetAccess:=dll.cuMemPoolGetAccess).restype, cuMemPoolGetAccess.argtypes = CUresult, [ctypes.POINTER(CUmemAccess_flags), CUmemoryPool, ctypes.POINTER(CUmemLocation)]
except AttributeError: pass

# CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps)
try: (cuMemPoolCreate:=dll.cuMemPoolCreate).restype, cuMemPoolCreate.argtypes = CUresult, [ctypes.POINTER(CUmemoryPool), ctypes.POINTER(CUmemPoolProps)]
except AttributeError: pass

# CUresult cuMemPoolDestroy(CUmemoryPool pool)
try: (cuMemPoolDestroy:=dll.cuMemPoolDestroy).restype, cuMemPoolDestroy.argtypes = CUresult, [CUmemoryPool]
except AttributeError: pass

# CUresult cuMemAllocFromPoolAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)
try: (cuMemAllocFromPoolAsync_ptsz:=dll.cuMemAllocFromPoolAsync_ptsz).restype, cuMemAllocFromPoolAsync_ptsz.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), size_t, CUmemoryPool, CUstream]
except AttributeError: pass

# CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags)
try: (cuMemPoolExportToShareableHandle:=dll.cuMemPoolExportToShareableHandle).restype, cuMemPoolExportToShareableHandle.argtypes = CUresult, [ctypes.c_void_p, CUmemoryPool, CUmemAllocationHandleType, ctypes.c_ulonglong]
except AttributeError: pass

# CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle, CUmemAllocationHandleType handleType, unsigned long long flags)
try: (cuMemPoolImportFromShareableHandle:=dll.cuMemPoolImportFromShareableHandle).restype, cuMemPoolImportFromShareableHandle.argtypes = CUresult, [ctypes.POINTER(CUmemoryPool), ctypes.c_void_p, CUmemAllocationHandleType, ctypes.c_ulonglong]
except AttributeError: pass

# CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr)
try: (cuMemPoolExportPointer:=dll.cuMemPoolExportPointer).restype, cuMemPoolExportPointer.argtypes = CUresult, [ctypes.POINTER(CUmemPoolPtrExportData), CUdeviceptr]
except AttributeError: pass

# CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData)
try: (cuMemPoolImportPointer:=dll.cuMemPoolImportPointer).restype, cuMemPoolImportPointer.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), CUmemoryPool, ctypes.POINTER(CUmemPoolPtrExportData)]
except AttributeError: pass

# CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr)
try: (cuPointerGetAttribute:=dll.cuPointerGetAttribute).restype, cuPointerGetAttribute.argtypes = CUresult, [ctypes.c_void_p, CUpointer_attribute, CUdeviceptr]
except AttributeError: pass

# CUresult cuMemPrefetchAsync_ptsz(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream)
try: (cuMemPrefetchAsync_ptsz:=dll.cuMemPrefetchAsync_ptsz).restype, cuMemPrefetchAsync_ptsz.argtypes = CUresult, [CUdeviceptr, size_t, CUdevice, CUstream]
except AttributeError: pass

# CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device)
try: (cuMemAdvise:=dll.cuMemAdvise).restype, cuMemAdvise.argtypes = CUresult, [CUdeviceptr, size_t, CUmem_advise, CUdevice]
except AttributeError: pass

# CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count)
try: (cuMemRangeGetAttribute:=dll.cuMemRangeGetAttribute).restype, cuMemRangeGetAttribute.argtypes = CUresult, [ctypes.c_void_p, size_t, CUmem_range_attribute, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count)
try: (cuMemRangeGetAttributes:=dll.cuMemRangeGetAttributes).restype, cuMemRangeGetAttributes.argtypes = CUresult, [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), ctypes.POINTER(CUmem_range_attribute), size_t, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr)
try: (cuPointerSetAttribute:=dll.cuPointerSetAttribute).restype, cuPointerSetAttribute.argtypes = CUresult, [ctypes.c_void_p, CUpointer_attribute, CUdeviceptr]
except AttributeError: pass

# CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr)
try: (cuPointerGetAttributes:=dll.cuPointerGetAttributes).restype, cuPointerGetAttributes.argtypes = CUresult, [ctypes.c_uint, ctypes.POINTER(CUpointer_attribute), ctypes.POINTER(ctypes.c_void_p), CUdeviceptr]
except AttributeError: pass

# CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags)
try: (cuStreamCreate:=dll.cuStreamCreate).restype, cuStreamCreate.argtypes = CUresult, [ctypes.POINTER(CUstream), ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority)
try: (cuStreamCreateWithPriority:=dll.cuStreamCreateWithPriority).restype, cuStreamCreateWithPriority.argtypes = CUresult, [ctypes.POINTER(CUstream), ctypes.c_uint, ctypes.c_int]
except AttributeError: pass

# CUresult cuStreamGetPriority_ptsz(CUstream hStream, int *priority)
try: (cuStreamGetPriority_ptsz:=dll.cuStreamGetPriority_ptsz).restype, cuStreamGetPriority_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# CUresult cuStreamGetFlags_ptsz(CUstream hStream, unsigned int *flags)
try: (cuStreamGetFlags_ptsz:=dll.cuStreamGetFlags_ptsz).restype, cuStreamGetFlags_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# CUresult cuStreamGetId_ptsz(CUstream hStream, unsigned long long *streamId)
try: (cuStreamGetId_ptsz:=dll.cuStreamGetId_ptsz).restype, cuStreamGetId_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(ctypes.c_ulonglong)]
except AttributeError: pass

# CUresult cuStreamGetCtx_ptsz(CUstream hStream, CUcontext *pctx)
try: (cuStreamGetCtx_ptsz:=dll.cuStreamGetCtx_ptsz).restype, cuStreamGetCtx_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(CUcontext)]
except AttributeError: pass

# CUresult cuStreamWaitEvent_ptsz(CUstream hStream, CUevent hEvent, unsigned int Flags)
try: (cuStreamWaitEvent_ptsz:=dll.cuStreamWaitEvent_ptsz).restype, cuStreamWaitEvent_ptsz.argtypes = CUresult, [CUstream, CUevent, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamAddCallback_ptsz(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags)
try: (cuStreamAddCallback_ptsz:=dll.cuStreamAddCallback_ptsz).restype, cuStreamAddCallback_ptsz.argtypes = CUresult, [CUstream, CUstreamCallback, ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamBeginCapture_v2_ptsz(CUstream hStream, CUstreamCaptureMode mode)
try: (cuStreamBeginCapture_v2_ptsz:=dll.cuStreamBeginCapture_v2_ptsz).restype, cuStreamBeginCapture_v2_ptsz.argtypes = CUresult, [CUstream, CUstreamCaptureMode]
except AttributeError: pass

# CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode)
try: (cuThreadExchangeStreamCaptureMode:=dll.cuThreadExchangeStreamCaptureMode).restype, cuThreadExchangeStreamCaptureMode.argtypes = CUresult, [ctypes.POINTER(CUstreamCaptureMode)]
except AttributeError: pass

# CUresult cuStreamEndCapture_ptsz(CUstream hStream, CUgraph *phGraph)
try: (cuStreamEndCapture_ptsz:=dll.cuStreamEndCapture_ptsz).restype, cuStreamEndCapture_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(CUgraph)]
except AttributeError: pass

# CUresult cuStreamIsCapturing_ptsz(CUstream hStream, CUstreamCaptureStatus *captureStatus)
try: (cuStreamIsCapturing_ptsz:=dll.cuStreamIsCapturing_ptsz).restype, cuStreamIsCapturing_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(CUstreamCaptureStatus)]
except AttributeError: pass

# CUresult cuStreamGetCaptureInfo_v2_ptsz(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out)
try: (cuStreamGetCaptureInfo_v2_ptsz:=dll.cuStreamGetCaptureInfo_v2_ptsz).restype, cuStreamGetCaptureInfo_v2_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(CUstreamCaptureStatus), ctypes.POINTER(cuuint64_t), ctypes.POINTER(CUgraph), ctypes.POINTER(ctypes.POINTER(CUgraphNode)), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuStreamUpdateCaptureDependencies_ptsz(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags)
try: (cuStreamUpdateCaptureDependencies_ptsz:=dll.cuStreamUpdateCaptureDependencies_ptsz).restype, cuStreamUpdateCaptureDependencies_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(CUgraphNode), size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamAttachMemAsync_ptsz(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)
try: (cuStreamAttachMemAsync_ptsz:=dll.cuStreamAttachMemAsync_ptsz).restype, cuStreamAttachMemAsync_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamQuery_ptsz(CUstream hStream)
try: (cuStreamQuery_ptsz:=dll.cuStreamQuery_ptsz).restype, cuStreamQuery_ptsz.argtypes = CUresult, [CUstream]
except AttributeError: pass

# CUresult cuStreamSynchronize_ptsz(CUstream hStream)
try: (cuStreamSynchronize_ptsz:=dll.cuStreamSynchronize_ptsz).restype, cuStreamSynchronize_ptsz.argtypes = CUresult, [CUstream]
except AttributeError: pass

# CUresult cuStreamDestroy_v2(CUstream hStream)
try: (cuStreamDestroy_v2:=dll.cuStreamDestroy_v2).restype, cuStreamDestroy_v2.argtypes = CUresult, [CUstream]
except AttributeError: pass

# CUresult cuStreamCopyAttributes_ptsz(CUstream dst, CUstream src)
try: (cuStreamCopyAttributes_ptsz:=dll.cuStreamCopyAttributes_ptsz).restype, cuStreamCopyAttributes_ptsz.argtypes = CUresult, [CUstream, CUstream]
except AttributeError: pass

# CUresult cuStreamGetAttribute_ptsz(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue *value_out)
try: (cuStreamGetAttribute_ptsz:=dll.cuStreamGetAttribute_ptsz).restype, cuStreamGetAttribute_ptsz.argtypes = CUresult, [CUstream, CUstreamAttrID, ctypes.POINTER(CUstreamAttrValue)]
except AttributeError: pass

# CUresult cuStreamSetAttribute_ptsz(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *value)
try: (cuStreamSetAttribute_ptsz:=dll.cuStreamSetAttribute_ptsz).restype, cuStreamSetAttribute_ptsz.argtypes = CUresult, [CUstream, CUstreamAttrID, ctypes.POINTER(CUstreamAttrValue)]
except AttributeError: pass

# CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags)
try: (cuEventCreate:=dll.cuEventCreate).restype, cuEventCreate.argtypes = CUresult, [ctypes.POINTER(CUevent), ctypes.c_uint]
except AttributeError: pass

# CUresult cuEventRecord_ptsz(CUevent hEvent, CUstream hStream)
try: (cuEventRecord_ptsz:=dll.cuEventRecord_ptsz).restype, cuEventRecord_ptsz.argtypes = CUresult, [CUevent, CUstream]
except AttributeError: pass

# CUresult cuEventRecordWithFlags_ptsz(CUevent hEvent, CUstream hStream, unsigned int flags)
try: (cuEventRecordWithFlags_ptsz:=dll.cuEventRecordWithFlags_ptsz).restype, cuEventRecordWithFlags_ptsz.argtypes = CUresult, [CUevent, CUstream, ctypes.c_uint]
except AttributeError: pass

# CUresult cuEventQuery(CUevent hEvent)
try: (cuEventQuery:=dll.cuEventQuery).restype, cuEventQuery.argtypes = CUresult, [CUevent]
except AttributeError: pass

# CUresult cuEventSynchronize(CUevent hEvent)
try: (cuEventSynchronize:=dll.cuEventSynchronize).restype, cuEventSynchronize.argtypes = CUresult, [CUevent]
except AttributeError: pass

# CUresult cuEventDestroy_v2(CUevent hEvent)
try: (cuEventDestroy_v2:=dll.cuEventDestroy_v2).restype, cuEventDestroy_v2.argtypes = CUresult, [CUevent]
except AttributeError: pass

# CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
try: (cuEventElapsedTime:=dll.cuEventElapsedTime).restype, cuEventElapsedTime.argtypes = CUresult, [ctypes.POINTER(ctypes.c_float), CUevent, CUevent]
except AttributeError: pass

# CUresult cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc)
try: (cuImportExternalMemory:=dll.cuImportExternalMemory).restype, cuImportExternalMemory.argtypes = CUresult, [ctypes.POINTER(CUexternalMemory), ctypes.POINTER(CUDA_EXTERNAL_MEMORY_HANDLE_DESC)]
except AttributeError: pass

# CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc)
try: (cuExternalMemoryGetMappedBuffer:=dll.cuExternalMemoryGetMappedBuffer).restype, cuExternalMemoryGetMappedBuffer.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), CUexternalMemory, ctypes.POINTER(CUDA_EXTERNAL_MEMORY_BUFFER_DESC)]
except AttributeError: pass

# CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc)
try: (cuExternalMemoryGetMappedMipmappedArray:=dll.cuExternalMemoryGetMappedMipmappedArray).restype, cuExternalMemoryGetMappedMipmappedArray.argtypes = CUresult, [ctypes.POINTER(CUmipmappedArray), CUexternalMemory, ctypes.POINTER(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC)]
except AttributeError: pass

# CUresult cuDestroyExternalMemory(CUexternalMemory extMem)
try: (cuDestroyExternalMemory:=dll.cuDestroyExternalMemory).restype, cuDestroyExternalMemory.argtypes = CUresult, [CUexternalMemory]
except AttributeError: pass

# CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc)
try: (cuImportExternalSemaphore:=dll.cuImportExternalSemaphore).restype, cuImportExternalSemaphore.argtypes = CUresult, [ctypes.POINTER(CUexternalSemaphore), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC)]
except AttributeError: pass

# CUresult cuSignalExternalSemaphoresAsync_ptsz(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream)
try: (cuSignalExternalSemaphoresAsync_ptsz:=dll.cuSignalExternalSemaphoresAsync_ptsz).restype, cuSignalExternalSemaphoresAsync_ptsz.argtypes = CUresult, [ctypes.POINTER(CUexternalSemaphore), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuWaitExternalSemaphoresAsync_ptsz(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream)
try: (cuWaitExternalSemaphoresAsync_ptsz:=dll.cuWaitExternalSemaphoresAsync_ptsz).restype, cuWaitExternalSemaphoresAsync_ptsz.argtypes = CUresult, [ctypes.POINTER(CUexternalSemaphore), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS), ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem)
try: (cuDestroyExternalSemaphore:=dll.cuDestroyExternalSemaphore).restype, cuDestroyExternalSemaphore.argtypes = CUresult, [CUexternalSemaphore]
except AttributeError: pass

# CUresult cuStreamWaitValue32_v2_ptsz(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
try: (cuStreamWaitValue32_v2_ptsz:=dll.cuStreamWaitValue32_v2_ptsz).restype, cuStreamWaitValue32_v2_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWaitValue64_v2_ptsz(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
try: (cuStreamWaitValue64_v2_ptsz:=dll.cuStreamWaitValue64_v2_ptsz).restype, cuStreamWaitValue64_v2_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWriteValue32_v2_ptsz(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
try: (cuStreamWriteValue32_v2_ptsz:=dll.cuStreamWriteValue32_v2_ptsz).restype, cuStreamWriteValue32_v2_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWriteValue64_v2_ptsz(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
try: (cuStreamWriteValue64_v2_ptsz:=dll.cuStreamWriteValue64_v2_ptsz).restype, cuStreamWriteValue64_v2_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamBatchMemOp_v2_ptsz(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags)
try: (cuStreamBatchMemOp_v2_ptsz:=dll.cuStreamBatchMemOp_v2_ptsz).restype, cuStreamBatchMemOp_v2_ptsz.argtypes = CUresult, [CUstream, ctypes.c_uint, ctypes.POINTER(CUstreamBatchMemOpParams), ctypes.c_uint]
except AttributeError: pass

# CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc)
try: (cuFuncGetAttribute:=dll.cuFuncGetAttribute).restype, cuFuncGetAttribute.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUfunction_attribute, CUfunction]
except AttributeError: pass

# CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value)
try: (cuFuncSetAttribute:=dll.cuFuncSetAttribute).restype, cuFuncSetAttribute.argtypes = CUresult, [CUfunction, CUfunction_attribute, ctypes.c_int]
except AttributeError: pass

# CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
try: (cuFuncSetCacheConfig:=dll.cuFuncSetCacheConfig).restype, cuFuncSetCacheConfig.argtypes = CUresult, [CUfunction, CUfunc_cache]
except AttributeError: pass

# CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config)
try: (cuFuncSetSharedMemConfig:=dll.cuFuncSetSharedMemConfig).restype, cuFuncSetSharedMemConfig.argtypes = CUresult, [CUfunction, CUsharedconfig]
except AttributeError: pass

# CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc)
try: (cuFuncGetModule:=dll.cuFuncGetModule).restype, cuFuncGetModule.argtypes = CUresult, [ctypes.POINTER(CUmodule), CUfunction]
except AttributeError: pass

# CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
try: (cuLaunchKernel_ptsz:=dll.cuLaunchKernel_ptsz).restype, cuLaunchKernel_ptsz.argtypes = CUresult, [CUfunction, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, CUstream, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra)
try: (cuLaunchKernelEx_ptsz:=dll.cuLaunchKernelEx_ptsz).restype, cuLaunchKernelEx_ptsz.argtypes = CUresult, [ctypes.POINTER(CUlaunchConfig), CUfunction, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuLaunchCooperativeKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams)
try: (cuLaunchCooperativeKernel_ptsz:=dll.cuLaunchCooperativeKernel_ptsz).restype, cuLaunchCooperativeKernel_ptsz.argtypes = CUresult, [CUfunction, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, CUstream, ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags)
try: (cuLaunchCooperativeKernelMultiDevice:=dll.cuLaunchCooperativeKernelMultiDevice).restype, cuLaunchCooperativeKernelMultiDevice.argtypes = CUresult, [ctypes.POINTER(CUDA_LAUNCH_PARAMS), ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuLaunchHostFunc_ptsz(CUstream hStream, CUhostFn fn, void *userData)
try: (cuLaunchHostFunc_ptsz:=dll.cuLaunchHostFunc_ptsz).restype, cuLaunchHostFunc_ptsz.argtypes = CUresult, [CUstream, CUhostFn, ctypes.c_void_p]
except AttributeError: pass

# CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
try: (cuFuncSetBlockShape:=dll.cuFuncSetBlockShape).restype, cuFuncSetBlockShape.argtypes = CUresult, [CUfunction, ctypes.c_int, ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes)
try: (cuFuncSetSharedSize:=dll.cuFuncSetSharedSize).restype, cuFuncSetSharedSize.argtypes = CUresult, [CUfunction, ctypes.c_uint]
except AttributeError: pass

# CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
try: (cuParamSetSize:=dll.cuParamSetSize).restype, cuParamSetSize.argtypes = CUresult, [CUfunction, ctypes.c_uint]
except AttributeError: pass

# CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value)
try: (cuParamSeti:=dll.cuParamSeti).restype, cuParamSeti.argtypes = CUresult, [CUfunction, ctypes.c_int, ctypes.c_uint]
except AttributeError: pass

# CUresult cuParamSetf(CUfunction hfunc, int offset, float value)
try: (cuParamSetf:=dll.cuParamSetf).restype, cuParamSetf.argtypes = CUresult, [CUfunction, ctypes.c_int, ctypes.c_float]
except AttributeError: pass

# CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
try: (cuParamSetv:=dll.cuParamSetv).restype, cuParamSetv.argtypes = CUresult, [CUfunction, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# CUresult cuLaunch(CUfunction f)
try: (cuLaunch:=dll.cuLaunch).restype, cuLaunch.argtypes = CUresult, [CUfunction]
except AttributeError: pass

# CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
try: (cuLaunchGrid:=dll.cuLaunchGrid).restype, cuLaunchGrid.argtypes = CUresult, [CUfunction, ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream)
try: (cuLaunchGridAsync:=dll.cuLaunchGridAsync).restype, cuLaunchGridAsync.argtypes = CUresult, [CUfunction, ctypes.c_int, ctypes.c_int, CUstream]
except AttributeError: pass

# CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
try: (cuParamSetTexRef:=dll.cuParamSetTexRef).restype, cuParamSetTexRef.argtypes = CUresult, [CUfunction, ctypes.c_int, CUtexref]
except AttributeError: pass

# CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags)
try: (cuGraphCreate:=dll.cuGraphCreate).restype, cuGraphCreate.argtypes = CUresult, [ctypes.POINTER(CUgraph), ctypes.c_uint]
except AttributeError: pass

# CUresult cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams)
try: (cuGraphAddKernelNode_v2:=dll.cuGraphAddKernelNode_v2).restype, cuGraphAddKernelNode_v2.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_KERNEL_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams)
try: (cuGraphKernelNodeGetParams_v2:=dll.cuGraphKernelNodeGetParams_v2).restype, cuGraphKernelNodeGetParams_v2.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_KERNEL_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams)
try: (cuGraphKernelNodeSetParams_v2:=dll.cuGraphKernelNodeSetParams_v2).restype, cuGraphKernelNodeSetParams_v2.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_KERNEL_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx)
try: (cuGraphAddMemcpyNode:=dll.cuGraphAddMemcpyNode).restype, cuGraphAddMemcpyNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_MEMCPY3D), CUcontext]
except AttributeError: pass

# CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams)
try: (cuGraphMemcpyNodeGetParams:=dll.cuGraphMemcpyNodeGetParams).restype, cuGraphMemcpyNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_MEMCPY3D)]
except AttributeError: pass

# CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams)
try: (cuGraphMemcpyNodeSetParams:=dll.cuGraphMemcpyNodeSetParams).restype, cuGraphMemcpyNodeSetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_MEMCPY3D)]
except AttributeError: pass

# CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx)
try: (cuGraphAddMemsetNode:=dll.cuGraphAddMemsetNode).restype, cuGraphAddMemsetNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_MEMSET_NODE_PARAMS), CUcontext]
except AttributeError: pass

# CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams)
try: (cuGraphMemsetNodeGetParams:=dll.cuGraphMemsetNodeGetParams).restype, cuGraphMemsetNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_MEMSET_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams)
try: (cuGraphMemsetNodeSetParams:=dll.cuGraphMemsetNodeSetParams).restype, cuGraphMemsetNodeSetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_MEMSET_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams)
try: (cuGraphAddHostNode:=dll.cuGraphAddHostNode).restype, cuGraphAddHostNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_HOST_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams)
try: (cuGraphHostNodeGetParams:=dll.cuGraphHostNodeGetParams).restype, cuGraphHostNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_HOST_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams)
try: (cuGraphHostNodeSetParams:=dll.cuGraphHostNodeSetParams).restype, cuGraphHostNodeSetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_HOST_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph)
try: (cuGraphAddChildGraphNode:=dll.cuGraphAddChildGraphNode).restype, cuGraphAddChildGraphNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, CUgraph]
except AttributeError: pass

# CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph)
try: (cuGraphChildGraphNodeGetGraph:=dll.cuGraphChildGraphNodeGetGraph).restype, cuGraphChildGraphNodeGetGraph.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUgraph)]
except AttributeError: pass

# CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies)
try: (cuGraphAddEmptyNode:=dll.cuGraphAddEmptyNode).restype, cuGraphAddEmptyNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t]
except AttributeError: pass

# CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event)
try: (cuGraphAddEventRecordNode:=dll.cuGraphAddEventRecordNode).restype, cuGraphAddEventRecordNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, CUevent]
except AttributeError: pass

# CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out)
try: (cuGraphEventRecordNodeGetEvent:=dll.cuGraphEventRecordNodeGetEvent).restype, cuGraphEventRecordNodeGetEvent.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUevent)]
except AttributeError: pass

# CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event)
try: (cuGraphEventRecordNodeSetEvent:=dll.cuGraphEventRecordNodeSetEvent).restype, cuGraphEventRecordNodeSetEvent.argtypes = CUresult, [CUgraphNode, CUevent]
except AttributeError: pass

# CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event)
try: (cuGraphAddEventWaitNode:=dll.cuGraphAddEventWaitNode).restype, cuGraphAddEventWaitNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, CUevent]
except AttributeError: pass

# CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out)
try: (cuGraphEventWaitNodeGetEvent:=dll.cuGraphEventWaitNodeGetEvent).restype, cuGraphEventWaitNodeGetEvent.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUevent)]
except AttributeError: pass

# CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event)
try: (cuGraphEventWaitNodeSetEvent:=dll.cuGraphEventWaitNodeSetEvent).restype, cuGraphEventWaitNodeSetEvent.argtypes = CUresult, [CUgraphNode, CUevent]
except AttributeError: pass

# CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams)
try: (cuGraphAddExternalSemaphoresSignalNode:=dll.cuGraphAddExternalSemaphoresSignalNode).restype, cuGraphAddExternalSemaphoresSignalNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out)
try: (cuGraphExternalSemaphoresSignalNodeGetParams:=dll.cuGraphExternalSemaphoresSignalNodeGetParams).restype, cuGraphExternalSemaphoresSignalNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams)
try: (cuGraphExternalSemaphoresSignalNodeSetParams:=dll.cuGraphExternalSemaphoresSignalNodeSetParams).restype, cuGraphExternalSemaphoresSignalNodeSetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams)
try: (cuGraphAddExternalSemaphoresWaitNode:=dll.cuGraphAddExternalSemaphoresWaitNode).restype, cuGraphAddExternalSemaphoresWaitNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_EXT_SEM_WAIT_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out)
try: (cuGraphExternalSemaphoresWaitNodeGetParams:=dll.cuGraphExternalSemaphoresWaitNodeGetParams).restype, cuGraphExternalSemaphoresWaitNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_EXT_SEM_WAIT_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams)
try: (cuGraphExternalSemaphoresWaitNodeSetParams:=dll.cuGraphExternalSemaphoresWaitNodeSetParams).restype, cuGraphExternalSemaphoresWaitNodeSetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_EXT_SEM_WAIT_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphAddBatchMemOpNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams)
try: (cuGraphAddBatchMemOpNode:=dll.cuGraphAddBatchMemOpNode).restype, cuGraphAddBatchMemOpNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_BATCH_MEM_OP_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams_out)
try: (cuGraphBatchMemOpNodeGetParams:=dll.cuGraphBatchMemOpNodeGetParams).restype, cuGraphBatchMemOpNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_BATCH_MEM_OP_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams)
try: (cuGraphBatchMemOpNodeSetParams:=dll.cuGraphBatchMemOpNodeSetParams).restype, cuGraphBatchMemOpNodeSetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_BATCH_MEM_OP_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams)
try: (cuGraphExecBatchMemOpNodeSetParams:=dll.cuGraphExecBatchMemOpNodeSetParams).restype, cuGraphExecBatchMemOpNodeSetParams.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(CUDA_BATCH_MEM_OP_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams)
try: (cuGraphAddMemAllocNode:=dll.cuGraphAddMemAllocNode).restype, cuGraphAddMemAllocNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_MEM_ALLOC_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out)
try: (cuGraphMemAllocNodeGetParams:=dll.cuGraphMemAllocNodeGetParams).restype, cuGraphMemAllocNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_MEM_ALLOC_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr)
try: (cuGraphAddMemFreeNode:=dll.cuGraphAddMemFreeNode).restype, cuGraphAddMemFreeNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, CUdeviceptr]
except AttributeError: pass

# CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out)
try: (cuGraphMemFreeNodeGetParams:=dll.cuGraphMemFreeNodeGetParams).restype, cuGraphMemFreeNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUdeviceptr)]
except AttributeError: pass

# CUresult cuDeviceGraphMemTrim(CUdevice device)
try: (cuDeviceGraphMemTrim:=dll.cuDeviceGraphMemTrim).restype, cuDeviceGraphMemTrim.argtypes = CUresult, [CUdevice]
except AttributeError: pass

# CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value)
try: (cuDeviceGetGraphMemAttribute:=dll.cuDeviceGetGraphMemAttribute).restype, cuDeviceGetGraphMemAttribute.argtypes = CUresult, [CUdevice, CUgraphMem_attribute, ctypes.c_void_p]
except AttributeError: pass

# CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value)
try: (cuDeviceSetGraphMemAttribute:=dll.cuDeviceSetGraphMemAttribute).restype, cuDeviceSetGraphMemAttribute.argtypes = CUresult, [CUdevice, CUgraphMem_attribute, ctypes.c_void_p]
except AttributeError: pass

# CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph)
try: (cuGraphClone:=dll.cuGraphClone).restype, cuGraphClone.argtypes = CUresult, [ctypes.POINTER(CUgraph), CUgraph]
except AttributeError: pass

# CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph)
try: (cuGraphNodeFindInClone:=dll.cuGraphNodeFindInClone).restype, cuGraphNodeFindInClone.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraphNode, CUgraph]
except AttributeError: pass

# CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type)
try: (cuGraphNodeGetType:=dll.cuGraphNodeGetType).restype, cuGraphNodeGetType.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUgraphNodeType)]
except AttributeError: pass

# CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes)
try: (cuGraphGetNodes:=dll.cuGraphGetNodes).restype, cuGraphGetNodes.argtypes = CUresult, [CUgraph, ctypes.POINTER(CUgraphNode), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes)
try: (cuGraphGetRootNodes:=dll.cuGraphGetRootNodes).restype, cuGraphGetRootNodes.argtypes = CUresult, [CUgraph, ctypes.POINTER(CUgraphNode), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges)
try: (cuGraphGetEdges:=dll.cuGraphGetEdges).restype, cuGraphGetEdges.argtypes = CUresult, [CUgraph, ctypes.POINTER(CUgraphNode), ctypes.POINTER(CUgraphNode), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies)
try: (cuGraphNodeGetDependencies:=dll.cuGraphNodeGetDependencies).restype, cuGraphNodeGetDependencies.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUgraphNode), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes)
try: (cuGraphNodeGetDependentNodes:=dll.cuGraphNodeGetDependentNodes).restype, cuGraphNodeGetDependentNodes.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUgraphNode), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies)
try: (cuGraphAddDependencies:=dll.cuGraphAddDependencies).restype, cuGraphAddDependencies.argtypes = CUresult, [CUgraph, ctypes.POINTER(CUgraphNode), ctypes.POINTER(CUgraphNode), size_t]
except AttributeError: pass

# CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies)
try: (cuGraphRemoveDependencies:=dll.cuGraphRemoveDependencies).restype, cuGraphRemoveDependencies.argtypes = CUresult, [CUgraph, ctypes.POINTER(CUgraphNode), ctypes.POINTER(CUgraphNode), size_t]
except AttributeError: pass

# CUresult cuGraphDestroyNode(CUgraphNode hNode)
try: (cuGraphDestroyNode:=dll.cuGraphDestroyNode).restype, cuGraphDestroyNode.argtypes = CUresult, [CUgraphNode]
except AttributeError: pass

# CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags)
try: (cuGraphInstantiateWithFlags:=dll.cuGraphInstantiateWithFlags).restype, cuGraphInstantiateWithFlags.argtypes = CUresult, [ctypes.POINTER(CUgraphExec), CUgraph, ctypes.c_ulonglong]
except AttributeError: pass

# CUresult cuGraphInstantiateWithParams_ptsz(CUgraphExec *phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams)
try: (cuGraphInstantiateWithParams_ptsz:=dll.cuGraphInstantiateWithParams_ptsz).restype, cuGraphInstantiateWithParams_ptsz.argtypes = CUresult, [ctypes.POINTER(CUgraphExec), CUgraph, ctypes.POINTER(CUDA_GRAPH_INSTANTIATE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t *flags)
try: (cuGraphExecGetFlags:=dll.cuGraphExecGetFlags).restype, cuGraphExecGetFlags.argtypes = CUresult, [CUgraphExec, ctypes.POINTER(cuuint64_t)]
except AttributeError: pass

# CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams)
try: (cuGraphExecKernelNodeSetParams_v2:=dll.cuGraphExecKernelNodeSetParams_v2).restype, cuGraphExecKernelNodeSetParams_v2.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(CUDA_KERNEL_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx)
try: (cuGraphExecMemcpyNodeSetParams:=dll.cuGraphExecMemcpyNodeSetParams).restype, cuGraphExecMemcpyNodeSetParams.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(CUDA_MEMCPY3D), CUcontext]
except AttributeError: pass

# CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx)
try: (cuGraphExecMemsetNodeSetParams:=dll.cuGraphExecMemsetNodeSetParams).restype, cuGraphExecMemsetNodeSetParams.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(CUDA_MEMSET_NODE_PARAMS), CUcontext]
except AttributeError: pass

# CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams)
try: (cuGraphExecHostNodeSetParams:=dll.cuGraphExecHostNodeSetParams).restype, cuGraphExecHostNodeSetParams.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(CUDA_HOST_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph)
try: (cuGraphExecChildGraphNodeSetParams:=dll.cuGraphExecChildGraphNodeSetParams).restype, cuGraphExecChildGraphNodeSetParams.argtypes = CUresult, [CUgraphExec, CUgraphNode, CUgraph]
except AttributeError: pass

# CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
try: (cuGraphExecEventRecordNodeSetEvent:=dll.cuGraphExecEventRecordNodeSetEvent).restype, cuGraphExecEventRecordNodeSetEvent.argtypes = CUresult, [CUgraphExec, CUgraphNode, CUevent]
except AttributeError: pass

# CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
try: (cuGraphExecEventWaitNodeSetEvent:=dll.cuGraphExecEventWaitNodeSetEvent).restype, cuGraphExecEventWaitNodeSetEvent.argtypes = CUresult, [CUgraphExec, CUgraphNode, CUevent]
except AttributeError: pass

# CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams)
try: (cuGraphExecExternalSemaphoresSignalNodeSetParams:=dll.cuGraphExecExternalSemaphoresSignalNodeSetParams).restype, cuGraphExecExternalSemaphoresSignalNodeSetParams.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams)
try: (cuGraphExecExternalSemaphoresWaitNodeSetParams:=dll.cuGraphExecExternalSemaphoresWaitNodeSetParams).restype, cuGraphExecExternalSemaphoresWaitNodeSetParams.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(CUDA_EXT_SEM_WAIT_NODE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled)
try: (cuGraphNodeSetEnabled:=dll.cuGraphNodeSetEnabled).restype, cuGraphNodeSetEnabled.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.c_uint]
except AttributeError: pass

# CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int *isEnabled)
try: (cuGraphNodeGetEnabled:=dll.cuGraphNodeGetEnabled).restype, cuGraphNodeGetEnabled.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# CUresult cuGraphUpload_ptsz(CUgraphExec hGraphExec, CUstream hStream)
try: (cuGraphUpload_ptsz:=dll.cuGraphUpload_ptsz).restype, cuGraphUpload_ptsz.argtypes = CUresult, [CUgraphExec, CUstream]
except AttributeError: pass

# CUresult cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream)
try: (cuGraphLaunch_ptsz:=dll.cuGraphLaunch_ptsz).restype, cuGraphLaunch_ptsz.argtypes = CUresult, [CUgraphExec, CUstream]
except AttributeError: pass

# CUresult cuGraphExecDestroy(CUgraphExec hGraphExec)
try: (cuGraphExecDestroy:=dll.cuGraphExecDestroy).restype, cuGraphExecDestroy.argtypes = CUresult, [CUgraphExec]
except AttributeError: pass

# CUresult cuGraphDestroy(CUgraph hGraph)
try: (cuGraphDestroy:=dll.cuGraphDestroy).restype, cuGraphDestroy.argtypes = CUresult, [CUgraph]
except AttributeError: pass

# CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo *resultInfo)
try: (cuGraphExecUpdate_v2:=dll.cuGraphExecUpdate_v2).restype, cuGraphExecUpdate_v2.argtypes = CUresult, [CUgraphExec, CUgraph, ctypes.POINTER(CUgraphExecUpdateResultInfo)]
except AttributeError: pass

# CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src)
try: (cuGraphKernelNodeCopyAttributes:=dll.cuGraphKernelNodeCopyAttributes).restype, cuGraphKernelNodeCopyAttributes.argtypes = CUresult, [CUgraphNode, CUgraphNode]
except AttributeError: pass

# CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue *value_out)
try: (cuGraphKernelNodeGetAttribute:=dll.cuGraphKernelNodeGetAttribute).restype, cuGraphKernelNodeGetAttribute.argtypes = CUresult, [CUgraphNode, CUkernelNodeAttrID, ctypes.POINTER(CUkernelNodeAttrValue)]
except AttributeError: pass

# CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue *value)
try: (cuGraphKernelNodeSetAttribute:=dll.cuGraphKernelNodeSetAttribute).restype, cuGraphKernelNodeSetAttribute.argtypes = CUresult, [CUgraphNode, CUkernelNodeAttrID, ctypes.POINTER(CUkernelNodeAttrValue)]
except AttributeError: pass

# CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags)
try: (cuGraphDebugDotPrint:=dll.cuGraphDebugDotPrint).restype, cuGraphDebugDotPrint.argtypes = CUresult, [CUgraph, ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
except AttributeError: pass

# CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags)
try: (cuUserObjectCreate:=dll.cuUserObjectCreate).restype, cuUserObjectCreate.argtypes = CUresult, [ctypes.POINTER(CUuserObject), ctypes.c_void_p, CUhostFn, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuUserObjectRetain(CUuserObject object, unsigned int count)
try: (cuUserObjectRetain:=dll.cuUserObjectRetain).restype, cuUserObjectRetain.argtypes = CUresult, [CUuserObject, ctypes.c_uint]
except AttributeError: pass

# CUresult cuUserObjectRelease(CUuserObject object, unsigned int count)
try: (cuUserObjectRelease:=dll.cuUserObjectRelease).restype, cuUserObjectRelease.argtypes = CUresult, [CUuserObject, ctypes.c_uint]
except AttributeError: pass

# CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags)
try: (cuGraphRetainUserObject:=dll.cuGraphRetainUserObject).restype, cuGraphRetainUserObject.argtypes = CUresult, [CUgraph, CUuserObject, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count)
try: (cuGraphReleaseUserObject:=dll.cuGraphReleaseUserObject).restype, cuGraphReleaseUserObject.argtypes = CUresult, [CUgraph, CUuserObject, ctypes.c_uint]
except AttributeError: pass

# CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize)
try: (cuOccupancyMaxActiveBlocksPerMultiprocessor:=dll.cuOccupancyMaxActiveBlocksPerMultiprocessor).restype, cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUfunction, ctypes.c_int, size_t]
except AttributeError: pass

# CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
try: (cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:=dll.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags).restype, cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUfunction, ctypes.c_int, size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit)
try: (cuOccupancyMaxPotentialBlockSize:=dll.cuOccupancyMaxPotentialBlockSize).restype, cuOccupancyMaxPotentialBlockSize.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), CUfunction, CUoccupancyB2DSize, size_t, ctypes.c_int]
except AttributeError: pass

# CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags)
try: (cuOccupancyMaxPotentialBlockSizeWithFlags:=dll.cuOccupancyMaxPotentialBlockSizeWithFlags).restype, cuOccupancyMaxPotentialBlockSizeWithFlags.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), CUfunction, CUoccupancyB2DSize, size_t, ctypes.c_int, ctypes.c_uint]
except AttributeError: pass

# CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize)
try: (cuOccupancyAvailableDynamicSMemPerBlock:=dll.cuOccupancyAvailableDynamicSMemPerBlock).restype, cuOccupancyAvailableDynamicSMemPerBlock.argtypes = CUresult, [ctypes.POINTER(size_t), CUfunction, ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# CUresult cuOccupancyMaxPotentialClusterSize(int *clusterSize, CUfunction func, const CUlaunchConfig *config)
try: (cuOccupancyMaxPotentialClusterSize:=dll.cuOccupancyMaxPotentialClusterSize).restype, cuOccupancyMaxPotentialClusterSize.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUfunction, ctypes.POINTER(CUlaunchConfig)]
except AttributeError: pass

# CUresult cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func, const CUlaunchConfig *config)
try: (cuOccupancyMaxActiveClusters:=dll.cuOccupancyMaxActiveClusters).restype, cuOccupancyMaxActiveClusters.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUfunction, ctypes.POINTER(CUlaunchConfig)]
except AttributeError: pass

# CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags)
try: (cuTexRefSetArray:=dll.cuTexRefSetArray).restype, cuTexRefSetArray.argtypes = CUresult, [CUtexref, CUarray, ctypes.c_uint]
except AttributeError: pass

# CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags)
try: (cuTexRefSetMipmappedArray:=dll.cuTexRefSetMipmappedArray).restype, cuTexRefSetMipmappedArray.argtypes = CUresult, [CUtexref, CUmipmappedArray, ctypes.c_uint]
except AttributeError: pass

# CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)
try: (cuTexRefSetAddress_v2:=dll.cuTexRefSetAddress_v2).restype, cuTexRefSetAddress_v2.argtypes = CUresult, [ctypes.POINTER(size_t), CUtexref, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch)
try: (cuTexRefSetAddress2D_v3:=dll.cuTexRefSetAddress2D_v3).restype, cuTexRefSetAddress2D_v3.argtypes = CUresult, [CUtexref, ctypes.POINTER(CUDA_ARRAY_DESCRIPTOR), CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents)
try: (cuTexRefSetFormat:=dll.cuTexRefSetFormat).restype, cuTexRefSetFormat.argtypes = CUresult, [CUtexref, CUarray_format, ctypes.c_int]
except AttributeError: pass

# CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am)
try: (cuTexRefSetAddressMode:=dll.cuTexRefSetAddressMode).restype, cuTexRefSetAddressMode.argtypes = CUresult, [CUtexref, ctypes.c_int, CUaddress_mode]
except AttributeError: pass

# CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm)
try: (cuTexRefSetFilterMode:=dll.cuTexRefSetFilterMode).restype, cuTexRefSetFilterMode.argtypes = CUresult, [CUtexref, CUfilter_mode]
except AttributeError: pass

# CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm)
try: (cuTexRefSetMipmapFilterMode:=dll.cuTexRefSetMipmapFilterMode).restype, cuTexRefSetMipmapFilterMode.argtypes = CUresult, [CUtexref, CUfilter_mode]
except AttributeError: pass

# CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias)
try: (cuTexRefSetMipmapLevelBias:=dll.cuTexRefSetMipmapLevelBias).restype, cuTexRefSetMipmapLevelBias.argtypes = CUresult, [CUtexref, ctypes.c_float]
except AttributeError: pass

# CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp)
try: (cuTexRefSetMipmapLevelClamp:=dll.cuTexRefSetMipmapLevelClamp).restype, cuTexRefSetMipmapLevelClamp.argtypes = CUresult, [CUtexref, ctypes.c_float, ctypes.c_float]
except AttributeError: pass

# CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso)
try: (cuTexRefSetMaxAnisotropy:=dll.cuTexRefSetMaxAnisotropy).restype, cuTexRefSetMaxAnisotropy.argtypes = CUresult, [CUtexref, ctypes.c_uint]
except AttributeError: pass

# CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor)
try: (cuTexRefSetBorderColor:=dll.cuTexRefSetBorderColor).restype, cuTexRefSetBorderColor.argtypes = CUresult, [CUtexref, ctypes.POINTER(ctypes.c_float)]
except AttributeError: pass

# CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags)
try: (cuTexRefSetFlags:=dll.cuTexRefSetFlags).restype, cuTexRefSetFlags.argtypes = CUresult, [CUtexref, ctypes.c_uint]
except AttributeError: pass

# CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef)
try: (cuTexRefGetAddress_v2:=dll.cuTexRefGetAddress_v2).restype, cuTexRefGetAddress_v2.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef)
try: (cuTexRefGetArray:=dll.cuTexRefGetArray).restype, cuTexRefGetArray.argtypes = CUresult, [ctypes.POINTER(CUarray), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef)
try: (cuTexRefGetMipmappedArray:=dll.cuTexRefGetMipmappedArray).restype, cuTexRefGetMipmappedArray.argtypes = CUresult, [ctypes.POINTER(CUmipmappedArray), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim)
try: (cuTexRefGetAddressMode:=dll.cuTexRefGetAddressMode).restype, cuTexRefGetAddressMode.argtypes = CUresult, [ctypes.POINTER(CUaddress_mode), CUtexref, ctypes.c_int]
except AttributeError: pass

# CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef)
try: (cuTexRefGetFilterMode:=dll.cuTexRefGetFilterMode).restype, cuTexRefGetFilterMode.argtypes = CUresult, [ctypes.POINTER(CUfilter_mode), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef)
try: (cuTexRefGetFormat:=dll.cuTexRefGetFormat).restype, cuTexRefGetFormat.argtypes = CUresult, [ctypes.POINTER(CUarray_format), ctypes.POINTER(ctypes.c_int), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef)
try: (cuTexRefGetMipmapFilterMode:=dll.cuTexRefGetMipmapFilterMode).restype, cuTexRefGetMipmapFilterMode.argtypes = CUresult, [ctypes.POINTER(CUfilter_mode), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef)
try: (cuTexRefGetMipmapLevelBias:=dll.cuTexRefGetMipmapLevelBias).restype, cuTexRefGetMipmapLevelBias.argtypes = CUresult, [ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef)
try: (cuTexRefGetMipmapLevelClamp:=dll.cuTexRefGetMipmapLevelClamp).restype, cuTexRefGetMipmapLevelClamp.argtypes = CUresult, [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef)
try: (cuTexRefGetMaxAnisotropy:=dll.cuTexRefGetMaxAnisotropy).restype, cuTexRefGetMaxAnisotropy.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef)
try: (cuTexRefGetBorderColor:=dll.cuTexRefGetBorderColor).restype, cuTexRefGetBorderColor.argtypes = CUresult, [ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError: pass

# CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef)
try: (cuTexRefGetFlags:=dll.cuTexRefGetFlags).restype, cuTexRefGetFlags.argtypes = CUresult, [ctypes.POINTER(ctypes.c_uint), CUtexref]
except AttributeError: pass

# CUresult cuTexRefCreate(CUtexref *pTexRef)
try: (cuTexRefCreate:=dll.cuTexRefCreate).restype, cuTexRefCreate.argtypes = CUresult, [ctypes.POINTER(CUtexref)]
except AttributeError: pass

# CUresult cuTexRefDestroy(CUtexref hTexRef)
try: (cuTexRefDestroy:=dll.cuTexRefDestroy).restype, cuTexRefDestroy.argtypes = CUresult, [CUtexref]
except AttributeError: pass

# CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags)
try: (cuSurfRefSetArray:=dll.cuSurfRefSetArray).restype, cuSurfRefSetArray.argtypes = CUresult, [CUsurfref, CUarray, ctypes.c_uint]
except AttributeError: pass

# CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef)
try: (cuSurfRefGetArray:=dll.cuSurfRefGetArray).restype, cuSurfRefGetArray.argtypes = CUresult, [ctypes.POINTER(CUarray), CUsurfref]
except AttributeError: pass

# CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc)
try: (cuTexObjectCreate:=dll.cuTexObjectCreate).restype, cuTexObjectCreate.argtypes = CUresult, [ctypes.POINTER(CUtexObject), ctypes.POINTER(CUDA_RESOURCE_DESC), ctypes.POINTER(CUDA_TEXTURE_DESC), ctypes.POINTER(CUDA_RESOURCE_VIEW_DESC)]
except AttributeError: pass

# CUresult cuTexObjectDestroy(CUtexObject texObject)
try: (cuTexObjectDestroy:=dll.cuTexObjectDestroy).restype, cuTexObjectDestroy.argtypes = CUresult, [CUtexObject]
except AttributeError: pass

# CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject)
try: (cuTexObjectGetResourceDesc:=dll.cuTexObjectGetResourceDesc).restype, cuTexObjectGetResourceDesc.argtypes = CUresult, [ctypes.POINTER(CUDA_RESOURCE_DESC), CUtexObject]
except AttributeError: pass

# CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject)
try: (cuTexObjectGetTextureDesc:=dll.cuTexObjectGetTextureDesc).restype, cuTexObjectGetTextureDesc.argtypes = CUresult, [ctypes.POINTER(CUDA_TEXTURE_DESC), CUtexObject]
except AttributeError: pass

# CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject)
try: (cuTexObjectGetResourceViewDesc:=dll.cuTexObjectGetResourceViewDesc).restype, cuTexObjectGetResourceViewDesc.argtypes = CUresult, [ctypes.POINTER(CUDA_RESOURCE_VIEW_DESC), CUtexObject]
except AttributeError: pass

# CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc)
try: (cuSurfObjectCreate:=dll.cuSurfObjectCreate).restype, cuSurfObjectCreate.argtypes = CUresult, [ctypes.POINTER(CUsurfObject), ctypes.POINTER(CUDA_RESOURCE_DESC)]
except AttributeError: pass

# CUresult cuSurfObjectDestroy(CUsurfObject surfObject)
try: (cuSurfObjectDestroy:=dll.cuSurfObjectDestroy).restype, cuSurfObjectDestroy.argtypes = CUresult, [CUsurfObject]
except AttributeError: pass

# CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject)
try: (cuSurfObjectGetResourceDesc:=dll.cuSurfObjectGetResourceDesc).restype, cuSurfObjectGetResourceDesc.argtypes = CUresult, [ctypes.POINTER(CUDA_RESOURCE_DESC), CUsurfObject]
except AttributeError: pass

# CUresult cuTensorMapEncodeTiled(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const cuuint32_t *boxDim, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)
try: (cuTensorMapEncodeTiled:=dll.cuTensorMapEncodeTiled).restype, cuTensorMapEncodeTiled.argtypes = CUresult, [ctypes.POINTER(CUtensorMap), CUtensorMapDataType, cuuint32_t, ctypes.c_void_p, ctypes.POINTER(cuuint64_t), ctypes.POINTER(cuuint64_t), ctypes.POINTER(cuuint32_t), ctypes.POINTER(cuuint32_t), CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill]
except AttributeError: pass

# CUresult cuTensorMapEncodeIm2col(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner, const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)
try: (cuTensorMapEncodeIm2col:=dll.cuTensorMapEncodeIm2col).restype, cuTensorMapEncodeIm2col.argtypes = CUresult, [ctypes.POINTER(CUtensorMap), CUtensorMapDataType, cuuint32_t, ctypes.c_void_p, ctypes.POINTER(cuuint64_t), ctypes.POINTER(cuuint64_t), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), cuuint32_t, cuuint32_t, ctypes.POINTER(cuuint32_t), CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill]
except AttributeError: pass

# CUresult cuTensorMapReplaceAddress(CUtensorMap *tensorMap, void *globalAddress)
try: (cuTensorMapReplaceAddress:=dll.cuTensorMapReplaceAddress).restype, cuTensorMapReplaceAddress.argtypes = CUresult, [ctypes.POINTER(CUtensorMap), ctypes.c_void_p]
except AttributeError: pass

# CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev)
try: (cuDeviceCanAccessPeer:=dll.cuDeviceCanAccessPeer).restype, cuDeviceCanAccessPeer.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUdevice, CUdevice]
except AttributeError: pass

# CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags)
try: (cuCtxEnablePeerAccess:=dll.cuCtxEnablePeerAccess).restype, cuCtxEnablePeerAccess.argtypes = CUresult, [CUcontext, ctypes.c_uint]
except AttributeError: pass

# CUresult cuCtxDisablePeerAccess(CUcontext peerContext)
try: (cuCtxDisablePeerAccess:=dll.cuCtxDisablePeerAccess).restype, cuCtxDisablePeerAccess.argtypes = CUresult, [CUcontext]
except AttributeError: pass

# CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice)
try: (cuDeviceGetP2PAttribute:=dll.cuDeviceGetP2PAttribute).restype, cuDeviceGetP2PAttribute.argtypes = CUresult, [ctypes.POINTER(ctypes.c_int), CUdevice_P2PAttribute, CUdevice, CUdevice]
except AttributeError: pass

# CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource)
try: (cuGraphicsUnregisterResource:=dll.cuGraphicsUnregisterResource).restype, cuGraphicsUnregisterResource.argtypes = CUresult, [CUgraphicsResource]
except AttributeError: pass

# CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel)
try: (cuGraphicsSubResourceGetMappedArray:=dll.cuGraphicsSubResourceGetMappedArray).restype, cuGraphicsSubResourceGetMappedArray.argtypes = CUresult, [ctypes.POINTER(CUarray), CUgraphicsResource, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource)
try: (cuGraphicsResourceGetMappedMipmappedArray:=dll.cuGraphicsResourceGetMappedMipmappedArray).restype, cuGraphicsResourceGetMappedMipmappedArray.argtypes = CUresult, [ctypes.POINTER(CUmipmappedArray), CUgraphicsResource]
except AttributeError: pass

# CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource)
try: (cuGraphicsResourceGetMappedPointer_v2:=dll.cuGraphicsResourceGetMappedPointer_v2).restype, cuGraphicsResourceGetMappedPointer_v2.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), ctypes.POINTER(size_t), CUgraphicsResource]
except AttributeError: pass

# CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags)
try: (cuGraphicsResourceSetMapFlags_v2:=dll.cuGraphicsResourceSetMapFlags_v2).restype, cuGraphicsResourceSetMapFlags_v2.argtypes = CUresult, [CUgraphicsResource, ctypes.c_uint]
except AttributeError: pass

# CUresult cuGraphicsMapResources_ptsz(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
try: (cuGraphicsMapResources_ptsz:=dll.cuGraphicsMapResources_ptsz).restype, cuGraphicsMapResources_ptsz.argtypes = CUresult, [ctypes.c_uint, ctypes.POINTER(CUgraphicsResource), CUstream]
except AttributeError: pass

# CUresult cuGraphicsUnmapResources_ptsz(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
try: (cuGraphicsUnmapResources_ptsz:=dll.cuGraphicsUnmapResources_ptsz).restype, cuGraphicsUnmapResources_ptsz.argtypes = CUresult, [ctypes.c_uint, ctypes.POINTER(CUgraphicsResource), CUstream]
except AttributeError: pass

# CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
try: (cuGetProcAddress_v2:=dll.cuGetProcAddress_v2).restype, cuGetProcAddress_v2.argtypes = CUresult, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, cuuint64_t, ctypes.POINTER(CUdriverProcAddressQueryResult)]
except AttributeError: pass

# CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId)
try: (cuGetExportTable:=dll.cuGetExportTable).restype, cuGetExportTable.argtypes = CUresult, [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUuuid)]
except AttributeError: pass

# CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags)
try: (cuMemHostRegister:=dll.cuMemHostRegister).restype, cuMemHostRegister.argtypes = CUresult, [ctypes.c_void_p, size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags)
try: (cuGraphicsResourceSetMapFlags:=dll.cuGraphicsResourceSetMapFlags).restype, cuGraphicsResourceSetMapFlags.argtypes = CUresult, [CUgraphicsResource, ctypes.c_uint]
except AttributeError: pass

# CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut)
try: (cuLinkCreate:=dll.cuLinkCreate).restype, cuLinkCreate.argtypes = CUresult, [ctypes.c_uint, ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUlinkState)]
except AttributeError: pass

# CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues)
try: (cuLinkAddData:=dll.cuLinkAddData).restype, cuLinkAddData.argtypes = CUresult, [CUlinkState, CUjitInputType, ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint, ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues)
try: (cuLinkAddFile:=dll.cuLinkAddFile).restype, cuLinkAddFile.argtypes = CUresult, [CUlinkState, CUjitInputType, ctypes.POINTER(ctypes.c_char), ctypes.c_uint, ctypes.POINTER(CUjit_option), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuTexRefSetAddress2D_v2(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch)
try: (cuTexRefSetAddress2D_v2:=dll.cuTexRefSetAddress2D_v2).restype, cuTexRefSetAddress2D_v2.argtypes = CUresult, [CUtexref, ctypes.POINTER(CUDA_ARRAY_DESCRIPTOR), CUdeviceptr, size_t]
except AttributeError: pass

CUdeviceptr_v1 = ctypes.c_uint
class struct_CUDA_MEMCPY2D_v1_st(ctypes.Structure): pass
struct_CUDA_MEMCPY2D_v1_st._fields_ = [
  ('srcXInBytes', ctypes.c_uint),
  ('srcY', ctypes.c_uint),
  ('srcMemoryType', CUmemorytype),
  ('srcHost', ctypes.c_void_p),
  ('srcDevice', CUdeviceptr_v1),
  ('srcArray', CUarray),
  ('srcPitch', ctypes.c_uint),
  ('dstXInBytes', ctypes.c_uint),
  ('dstY', ctypes.c_uint),
  ('dstMemoryType', CUmemorytype),
  ('dstHost', ctypes.c_void_p),
  ('dstDevice', CUdeviceptr_v1),
  ('dstArray', CUarray),
  ('dstPitch', ctypes.c_uint),
  ('WidthInBytes', ctypes.c_uint),
  ('Height', ctypes.c_uint),
]
CUDA_MEMCPY2D_v1 = struct_CUDA_MEMCPY2D_v1_st
class struct_CUDA_MEMCPY3D_v1_st(ctypes.Structure): pass
struct_CUDA_MEMCPY3D_v1_st._fields_ = [
  ('srcXInBytes', ctypes.c_uint),
  ('srcY', ctypes.c_uint),
  ('srcZ', ctypes.c_uint),
  ('srcLOD', ctypes.c_uint),
  ('srcMemoryType', CUmemorytype),
  ('srcHost', ctypes.c_void_p),
  ('srcDevice', CUdeviceptr_v1),
  ('srcArray', CUarray),
  ('reserved0', ctypes.c_void_p),
  ('srcPitch', ctypes.c_uint),
  ('srcHeight', ctypes.c_uint),
  ('dstXInBytes', ctypes.c_uint),
  ('dstY', ctypes.c_uint),
  ('dstZ', ctypes.c_uint),
  ('dstLOD', ctypes.c_uint),
  ('dstMemoryType', CUmemorytype),
  ('dstHost', ctypes.c_void_p),
  ('dstDevice', CUdeviceptr_v1),
  ('dstArray', CUarray),
  ('reserved1', ctypes.c_void_p),
  ('dstPitch', ctypes.c_uint),
  ('dstHeight', ctypes.c_uint),
  ('WidthInBytes', ctypes.c_uint),
  ('Height', ctypes.c_uint),
  ('Depth', ctypes.c_uint),
]
CUDA_MEMCPY3D_v1 = struct_CUDA_MEMCPY3D_v1_st
class struct_CUDA_ARRAY_DESCRIPTOR_v1_st(ctypes.Structure): pass
struct_CUDA_ARRAY_DESCRIPTOR_v1_st._fields_ = [
  ('Width', ctypes.c_uint),
  ('Height', ctypes.c_uint),
  ('Format', CUarray_format),
  ('NumChannels', ctypes.c_uint),
]
CUDA_ARRAY_DESCRIPTOR_v1 = struct_CUDA_ARRAY_DESCRIPTOR_v1_st
class struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st(ctypes.Structure): pass
struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st._fields_ = [
  ('Width', ctypes.c_uint),
  ('Height', ctypes.c_uint),
  ('Depth', ctypes.c_uint),
  ('Format', CUarray_format),
  ('NumChannels', ctypes.c_uint),
  ('Flags', ctypes.c_uint),
]
CUDA_ARRAY3D_DESCRIPTOR_v1 = struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st
# CUresult cuDeviceTotalMem(unsigned int *bytes, CUdevice dev)
try: (cuDeviceTotalMem:=dll.cuDeviceTotalMem).restype, cuDeviceTotalMem.argtypes = CUresult, [ctypes.POINTER(ctypes.c_uint), CUdevice]
except AttributeError: pass

# CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
try: (cuCtxCreate:=dll.cuCtxCreate).restype, cuCtxCreate.argtypes = CUresult, [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
except AttributeError: pass

# CUresult cuModuleGetGlobal(CUdeviceptr_v1 *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
try: (cuModuleGetGlobal:=dll.cuModuleGetGlobal).restype, cuModuleGetGlobal.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr_v1), ctypes.POINTER(ctypes.c_uint), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# CUresult cuMemGetInfo(unsigned int *free, unsigned int *total)
try: (cuMemGetInfo:=dll.cuMemGetInfo).restype, cuMemGetInfo.argtypes = CUresult, [ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# CUresult cuMemAlloc(CUdeviceptr_v1 *dptr, unsigned int bytesize)
try: (cuMemAlloc:=dll.cuMemAlloc).restype, cuMemAlloc.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr_v1), ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemAllocPitch(CUdeviceptr_v1 *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
try: (cuMemAllocPitch:=dll.cuMemAllocPitch).restype, cuMemAllocPitch.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr_v1), ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemFree(CUdeviceptr_v1 dptr)
try: (cuMemFree:=dll.cuMemFree).restype, cuMemFree.argtypes = CUresult, [CUdeviceptr_v1]
except AttributeError: pass

# CUresult cuMemGetAddressRange(CUdeviceptr_v1 *pbase, unsigned int *psize, CUdeviceptr_v1 dptr)
try: (cuMemGetAddressRange:=dll.cuMemGetAddressRange).restype, cuMemGetAddressRange.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr_v1), ctypes.POINTER(ctypes.c_uint), CUdeviceptr_v1]
except AttributeError: pass

# CUresult cuMemAllocHost(void **pp, unsigned int bytesize)
try: (cuMemAllocHost:=dll.cuMemAllocHost).restype, cuMemAllocHost.argtypes = CUresult, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemHostGetDevicePointer(CUdeviceptr_v1 *pdptr, void *p, unsigned int Flags)
try: (cuMemHostGetDevicePointer:=dll.cuMemHostGetDevicePointer).restype, cuMemHostGetDevicePointer.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr_v1), ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyHtoD(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount)
try: (cuMemcpyHtoD:=dll.cuMemcpyHtoD).restype, cuMemcpyHtoD.argtypes = CUresult, [CUdeviceptr_v1, ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount)
try: (cuMemcpyDtoH:=dll.cuMemcpyDtoH).restype, cuMemcpyDtoH.argtypes = CUresult, [ctypes.c_void_p, CUdeviceptr_v1, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyDtoD(CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount)
try: (cuMemcpyDtoD:=dll.cuMemcpyDtoD).restype, cuMemcpyDtoD.argtypes = CUresult, [CUdeviceptr_v1, CUdeviceptr_v1, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyDtoA(CUarray dstArray, unsigned int dstOffset, CUdeviceptr_v1 srcDevice, unsigned int ByteCount)
try: (cuMemcpyDtoA:=dll.cuMemcpyDtoA).restype, cuMemcpyDtoA.argtypes = CUresult, [CUarray, ctypes.c_uint, CUdeviceptr_v1, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyAtoD(CUdeviceptr_v1 dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount)
try: (cuMemcpyAtoD:=dll.cuMemcpyAtoD).restype, cuMemcpyAtoD.argtypes = CUresult, [CUdeviceptr_v1, CUarray, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyHtoA(CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount)
try: (cuMemcpyHtoA:=dll.cuMemcpyHtoA).restype, cuMemcpyHtoA.argtypes = CUresult, [CUarray, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyAtoH(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount)
try: (cuMemcpyAtoH:=dll.cuMemcpyAtoH).restype, cuMemcpyAtoH.argtypes = CUresult, [ctypes.c_void_p, CUarray, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyAtoA(CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount)
try: (cuMemcpyAtoA:=dll.cuMemcpyAtoA).restype, cuMemcpyAtoA.argtypes = CUresult, [CUarray, ctypes.c_uint, CUarray, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyHtoAAsync(CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream)
try: (cuMemcpyHtoAAsync:=dll.cuMemcpyHtoAAsync).restype, cuMemcpyHtoAAsync.argtypes = CUresult, [CUarray, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream)
try: (cuMemcpyAtoHAsync:=dll.cuMemcpyAtoHAsync).restype, cuMemcpyAtoHAsync.argtypes = CUresult, [ctypes.c_void_p, CUarray, ctypes.c_uint, ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuMemcpy2D(const CUDA_MEMCPY2D_v1 *pCopy)
try: (cuMemcpy2D:=dll.cuMemcpy2D).restype, cuMemcpy2D.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D_v1)]
except AttributeError: pass

# CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D_v1 *pCopy)
try: (cuMemcpy2DUnaligned:=dll.cuMemcpy2DUnaligned).restype, cuMemcpy2DUnaligned.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D_v1)]
except AttributeError: pass

# CUresult cuMemcpy3D(const CUDA_MEMCPY3D_v1 *pCopy)
try: (cuMemcpy3D:=dll.cuMemcpy3D).restype, cuMemcpy3D.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D_v1)]
except AttributeError: pass

# CUresult cuMemcpyHtoDAsync(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream)
try: (cuMemcpyHtoDAsync:=dll.cuMemcpyHtoDAsync).restype, cuMemcpyHtoDAsync.argtypes = CUresult, [CUdeviceptr_v1, ctypes.c_void_p, ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream)
try: (cuMemcpyDtoHAsync:=dll.cuMemcpyDtoHAsync).restype, cuMemcpyDtoHAsync.argtypes = CUresult, [ctypes.c_void_p, CUdeviceptr_v1, ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuMemcpyDtoDAsync(CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream)
try: (cuMemcpyDtoDAsync:=dll.cuMemcpyDtoDAsync).restype, cuMemcpyDtoDAsync.argtypes = CUresult, [CUdeviceptr_v1, CUdeviceptr_v1, ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D_v1 *pCopy, CUstream hStream)
try: (cuMemcpy2DAsync:=dll.cuMemcpy2DAsync).restype, cuMemcpy2DAsync.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D_v1), CUstream]
except AttributeError: pass

# CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D_v1 *pCopy, CUstream hStream)
try: (cuMemcpy3DAsync:=dll.cuMemcpy3DAsync).restype, cuMemcpy3DAsync.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D_v1), CUstream]
except AttributeError: pass

# CUresult cuMemsetD8(CUdeviceptr_v1 dstDevice, unsigned char uc, unsigned int N)
try: (cuMemsetD8:=dll.cuMemsetD8).restype, cuMemsetD8.argtypes = CUresult, [CUdeviceptr_v1, ctypes.c_ubyte, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemsetD16(CUdeviceptr_v1 dstDevice, unsigned short us, unsigned int N)
try: (cuMemsetD16:=dll.cuMemsetD16).restype, cuMemsetD16.argtypes = CUresult, [CUdeviceptr_v1, ctypes.c_ushort, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemsetD32(CUdeviceptr_v1 dstDevice, unsigned int ui, unsigned int N)
try: (cuMemsetD32:=dll.cuMemsetD32).restype, cuMemsetD32.argtypes = CUresult, [CUdeviceptr_v1, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemsetD2D8(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height)
try: (cuMemsetD2D8:=dll.cuMemsetD2D8).restype, cuMemsetD2D8.argtypes = CUresult, [CUdeviceptr_v1, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemsetD2D16(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height)
try: (cuMemsetD2D16:=dll.cuMemsetD2D16).restype, cuMemsetD2D16.argtypes = CUresult, [CUdeviceptr_v1, ctypes.c_uint, ctypes.c_ushort, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemsetD2D32(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height)
try: (cuMemsetD2D32:=dll.cuMemsetD2D32).restype, cuMemsetD2D32.argtypes = CUresult, [CUdeviceptr_v1, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
except AttributeError: pass

# CUresult cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR_v1 *pAllocateArray)
try: (cuArrayCreate:=dll.cuArrayCreate).restype, cuArrayCreate.argtypes = CUresult, [ctypes.POINTER(CUarray), ctypes.POINTER(CUDA_ARRAY_DESCRIPTOR_v1)]
except AttributeError: pass

# CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR_v1 *pArrayDescriptor, CUarray hArray)
try: (cuArrayGetDescriptor:=dll.cuArrayGetDescriptor).restype, cuArrayGetDescriptor.argtypes = CUresult, [ctypes.POINTER(CUDA_ARRAY_DESCRIPTOR_v1), CUarray]
except AttributeError: pass

# CUresult cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR_v1 *pAllocateArray)
try: (cuArray3DCreate:=dll.cuArray3DCreate).restype, cuArray3DCreate.argtypes = CUresult, [ctypes.POINTER(CUarray), ctypes.POINTER(CUDA_ARRAY3D_DESCRIPTOR_v1)]
except AttributeError: pass

# CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR_v1 *pArrayDescriptor, CUarray hArray)
try: (cuArray3DGetDescriptor:=dll.cuArray3DGetDescriptor).restype, cuArray3DGetDescriptor.argtypes = CUresult, [ctypes.POINTER(CUDA_ARRAY3D_DESCRIPTOR_v1), CUarray]
except AttributeError: pass

# CUresult cuTexRefSetAddress(unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr_v1 dptr, unsigned int bytes)
try: (cuTexRefSetAddress:=dll.cuTexRefSetAddress).restype, cuTexRefSetAddress.argtypes = CUresult, [ctypes.POINTER(ctypes.c_uint), CUtexref, CUdeviceptr_v1, ctypes.c_uint]
except AttributeError: pass

# CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR_v1 *desc, CUdeviceptr_v1 dptr, unsigned int Pitch)
try: (cuTexRefSetAddress2D:=dll.cuTexRefSetAddress2D).restype, cuTexRefSetAddress2D.argtypes = CUresult, [CUtexref, ctypes.POINTER(CUDA_ARRAY_DESCRIPTOR_v1), CUdeviceptr_v1, ctypes.c_uint]
except AttributeError: pass

# CUresult cuTexRefGetAddress(CUdeviceptr_v1 *pdptr, CUtexref hTexRef)
try: (cuTexRefGetAddress:=dll.cuTexRefGetAddress).restype, cuTexRefGetAddress.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr_v1), CUtexref]
except AttributeError: pass

# CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr_v1 *pDevPtr, unsigned int *pSize, CUgraphicsResource resource)
try: (cuGraphicsResourceGetMappedPointer:=dll.cuGraphicsResourceGetMappedPointer).restype, cuGraphicsResourceGetMappedPointer.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr_v1), ctypes.POINTER(ctypes.c_uint), CUgraphicsResource]
except AttributeError: pass

# CUresult cuCtxDestroy(CUcontext ctx)
try: (cuCtxDestroy:=dll.cuCtxDestroy).restype, cuCtxDestroy.argtypes = CUresult, [CUcontext]
except AttributeError: pass

# CUresult cuCtxPopCurrent(CUcontext *pctx)
try: (cuCtxPopCurrent:=dll.cuCtxPopCurrent).restype, cuCtxPopCurrent.argtypes = CUresult, [ctypes.POINTER(CUcontext)]
except AttributeError: pass

# CUresult cuCtxPushCurrent(CUcontext ctx)
try: (cuCtxPushCurrent:=dll.cuCtxPushCurrent).restype, cuCtxPushCurrent.argtypes = CUresult, [CUcontext]
except AttributeError: pass

# CUresult cuStreamDestroy(CUstream hStream)
try: (cuStreamDestroy:=dll.cuStreamDestroy).restype, cuStreamDestroy.argtypes = CUresult, [CUstream]
except AttributeError: pass

# CUresult cuEventDestroy(CUevent hEvent)
try: (cuEventDestroy:=dll.cuEventDestroy).restype, cuEventDestroy.argtypes = CUresult, [CUevent]
except AttributeError: pass

# CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
try: (cuDevicePrimaryCtxRelease:=dll.cuDevicePrimaryCtxRelease).restype, cuDevicePrimaryCtxRelease.argtypes = CUresult, [CUdevice]
except AttributeError: pass

# CUresult cuDevicePrimaryCtxReset(CUdevice dev)
try: (cuDevicePrimaryCtxReset:=dll.cuDevicePrimaryCtxReset).restype, cuDevicePrimaryCtxReset.argtypes = CUresult, [CUdevice]
except AttributeError: pass

# CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags)
try: (cuDevicePrimaryCtxSetFlags:=dll.cuDevicePrimaryCtxSetFlags).restype, cuDevicePrimaryCtxSetFlags.argtypes = CUresult, [CUdevice, ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
try: (cuMemcpyHtoD_v2:=dll.cuMemcpyHtoD_v2).restype, cuMemcpyHtoD_v2.argtypes = CUresult, [CUdeviceptr, ctypes.c_void_p, size_t]
except AttributeError: pass

# CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
try: (cuMemcpyDtoH_v2:=dll.cuMemcpyDtoH_v2).restype, cuMemcpyDtoH_v2.argtypes = CUresult, [ctypes.c_void_p, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
try: (cuMemcpyDtoD_v2:=dll.cuMemcpyDtoD_v2).restype, cuMemcpyDtoD_v2.argtypes = CUresult, [CUdeviceptr, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount)
try: (cuMemcpyDtoA_v2:=dll.cuMemcpyDtoA_v2).restype, cuMemcpyDtoA_v2.argtypes = CUresult, [CUarray, size_t, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
try: (cuMemcpyAtoD_v2:=dll.cuMemcpyAtoD_v2).restype, cuMemcpyAtoD_v2.argtypes = CUresult, [CUdeviceptr, CUarray, size_t, size_t]
except AttributeError: pass

# CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount)
try: (cuMemcpyHtoA_v2:=dll.cuMemcpyHtoA_v2).restype, cuMemcpyHtoA_v2.argtypes = CUresult, [CUarray, size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

# CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
try: (cuMemcpyAtoH_v2:=dll.cuMemcpyAtoH_v2).restype, cuMemcpyAtoH_v2.argtypes = CUresult, [ctypes.c_void_p, CUarray, size_t, size_t]
except AttributeError: pass

# CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount)
try: (cuMemcpyAtoA_v2:=dll.cuMemcpyAtoA_v2).restype, cuMemcpyAtoA_v2.argtypes = CUresult, [CUarray, size_t, CUarray, size_t, size_t]
except AttributeError: pass

# CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream)
try: (cuMemcpyHtoAAsync_v2:=dll.cuMemcpyHtoAAsync_v2).restype, cuMemcpyHtoAAsync_v2.argtypes = CUresult, [CUarray, size_t, ctypes.c_void_p, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream)
try: (cuMemcpyAtoHAsync_v2:=dll.cuMemcpyAtoHAsync_v2).restype, cuMemcpyAtoHAsync_v2.argtypes = CUresult, [ctypes.c_void_p, CUarray, size_t, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy)
try: (cuMemcpy2D_v2:=dll.cuMemcpy2D_v2).restype, cuMemcpy2D_v2.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D)]
except AttributeError: pass

# CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy)
try: (cuMemcpy2DUnaligned_v2:=dll.cuMemcpy2DUnaligned_v2).restype, cuMemcpy2DUnaligned_v2.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D)]
except AttributeError: pass

# CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy)
try: (cuMemcpy3D_v2:=dll.cuMemcpy3D_v2).restype, cuMemcpy3D_v2.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D)]
except AttributeError: pass

# CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
try: (cuMemcpyHtoDAsync_v2:=dll.cuMemcpyHtoDAsync_v2).restype, cuMemcpyHtoDAsync_v2.argtypes = CUresult, [CUdeviceptr, ctypes.c_void_p, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
try: (cuMemcpyDtoHAsync_v2:=dll.cuMemcpyDtoHAsync_v2).restype, cuMemcpyDtoHAsync_v2.argtypes = CUresult, [ctypes.c_void_p, CUdeviceptr, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
try: (cuMemcpyDtoDAsync_v2:=dll.cuMemcpyDtoDAsync_v2).restype, cuMemcpyDtoDAsync_v2.argtypes = CUresult, [CUdeviceptr, CUdeviceptr, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
try: (cuMemcpy2DAsync_v2:=dll.cuMemcpy2DAsync_v2).restype, cuMemcpy2DAsync_v2.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY2D), CUstream]
except AttributeError: pass

# CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
try: (cuMemcpy3DAsync_v2:=dll.cuMemcpy3DAsync_v2).restype, cuMemcpy3DAsync_v2.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D), CUstream]
except AttributeError: pass

# CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
try: (cuMemsetD8_v2:=dll.cuMemsetD8_v2).restype, cuMemsetD8_v2.argtypes = CUresult, [CUdeviceptr, ctypes.c_ubyte, size_t]
except AttributeError: pass

# CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N)
try: (cuMemsetD16_v2:=dll.cuMemsetD16_v2).restype, cuMemsetD16_v2.argtypes = CUresult, [CUdeviceptr, ctypes.c_ushort, size_t]
except AttributeError: pass

# CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
try: (cuMemsetD32_v2:=dll.cuMemsetD32_v2).restype, cuMemsetD32_v2.argtypes = CUresult, [CUdeviceptr, ctypes.c_uint, size_t]
except AttributeError: pass

# CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
try: (cuMemsetD2D8_v2:=dll.cuMemsetD2D8_v2).restype, cuMemsetD2D8_v2.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t]
except AttributeError: pass

# CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
try: (cuMemsetD2D16_v2:=dll.cuMemsetD2D16_v2).restype, cuMemsetD2D16_v2.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_ushort, size_t, size_t]
except AttributeError: pass

# CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
try: (cuMemsetD2D32_v2:=dll.cuMemsetD2D32_v2).restype, cuMemsetD2D32_v2.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_uint, size_t, size_t]
except AttributeError: pass

# CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
try: (cuMemcpy:=dll.cuMemcpy).restype, cuMemcpy.argtypes = CUresult, [CUdeviceptr, CUdeviceptr, size_t]
except AttributeError: pass

# CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
try: (cuMemcpyAsync:=dll.cuMemcpyAsync).restype, cuMemcpyAsync.argtypes = CUresult, [CUdeviceptr, CUdeviceptr, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
try: (cuMemcpyPeer:=dll.cuMemcpyPeer).restype, cuMemcpyPeer.argtypes = CUresult, [CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t]
except AttributeError: pass

# CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
try: (cuMemcpyPeerAsync:=dll.cuMemcpyPeerAsync).restype, cuMemcpyPeerAsync.argtypes = CUresult, [CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy)
try: (cuMemcpy3DPeer:=dll.cuMemcpy3DPeer).restype, cuMemcpy3DPeer.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D_PEER)]
except AttributeError: pass

# CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream)
try: (cuMemcpy3DPeerAsync:=dll.cuMemcpy3DPeerAsync).restype, cuMemcpy3DPeerAsync.argtypes = CUresult, [ctypes.POINTER(CUDA_MEMCPY3D_PEER), CUstream]
except AttributeError: pass

# CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
try: (cuMemsetD8Async:=dll.cuMemsetD8Async).restype, cuMemsetD8Async.argtypes = CUresult, [CUdeviceptr, ctypes.c_ubyte, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
try: (cuMemsetD16Async:=dll.cuMemsetD16Async).restype, cuMemsetD16Async.argtypes = CUresult, [CUdeviceptr, ctypes.c_ushort, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
try: (cuMemsetD32Async:=dll.cuMemsetD32Async).restype, cuMemsetD32Async.argtypes = CUresult, [CUdeviceptr, ctypes.c_uint, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
try: (cuMemsetD2D8Async:=dll.cuMemsetD2D8Async).restype, cuMemsetD2D8Async.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
try: (cuMemsetD2D16Async:=dll.cuMemsetD2D16Async).restype, cuMemsetD2D16Async.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_ushort, size_t, size_t, CUstream]
except AttributeError: pass

# CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
try: (cuMemsetD2D32Async:=dll.cuMemsetD2D32Async).restype, cuMemsetD2D32Async.argtypes = CUresult, [CUdeviceptr, size_t, ctypes.c_uint, size_t, size_t, CUstream]
except AttributeError: pass

# CUresult cuStreamGetPriority(CUstream hStream, int *priority)
try: (cuStreamGetPriority:=dll.cuStreamGetPriority).restype, cuStreamGetPriority.argtypes = CUresult, [CUstream, ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId)
try: (cuStreamGetId:=dll.cuStreamGetId).restype, cuStreamGetId.argtypes = CUresult, [CUstream, ctypes.POINTER(ctypes.c_ulonglong)]
except AttributeError: pass

# CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags)
try: (cuStreamGetFlags:=dll.cuStreamGetFlags).restype, cuStreamGetFlags.argtypes = CUresult, [CUstream, ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx)
try: (cuStreamGetCtx:=dll.cuStreamGetCtx).restype, cuStreamGetCtx.argtypes = CUresult, [CUstream, ctypes.POINTER(CUcontext)]
except AttributeError: pass

# CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
try: (cuStreamWaitEvent:=dll.cuStreamWaitEvent).restype, cuStreamWaitEvent.argtypes = CUresult, [CUstream, CUevent, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags)
try: (cuStreamAddCallback:=dll.cuStreamAddCallback).restype, cuStreamAddCallback.argtypes = CUresult, [CUstream, CUstreamCallback, ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)
try: (cuStreamAttachMemAsync:=dll.cuStreamAttachMemAsync).restype, cuStreamAttachMemAsync.argtypes = CUresult, [CUstream, CUdeviceptr, size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamQuery(CUstream hStream)
try: (cuStreamQuery:=dll.cuStreamQuery).restype, cuStreamQuery.argtypes = CUresult, [CUstream]
except AttributeError: pass

# CUresult cuStreamSynchronize(CUstream hStream)
try: (cuStreamSynchronize:=dll.cuStreamSynchronize).restype, cuStreamSynchronize.argtypes = CUresult, [CUstream]
except AttributeError: pass

# CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
try: (cuEventRecord:=dll.cuEventRecord).restype, cuEventRecord.argtypes = CUresult, [CUevent, CUstream]
except AttributeError: pass

# CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
try: (cuEventRecordWithFlags:=dll.cuEventRecordWithFlags).restype, cuEventRecordWithFlags.argtypes = CUresult, [CUevent, CUstream, ctypes.c_uint]
except AttributeError: pass

# CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
try: (cuLaunchKernel:=dll.cuLaunchKernel).restype, cuLaunchKernel.argtypes = CUresult, [CUfunction, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, CUstream, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra)
try: (cuLaunchKernelEx:=dll.cuLaunchKernelEx).restype, cuLaunchKernelEx.argtypes = CUresult, [ctypes.POINTER(CUlaunchConfig), CUfunction, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData)
try: (cuLaunchHostFunc:=dll.cuLaunchHostFunc).restype, cuLaunchHostFunc.argtypes = CUresult, [CUstream, CUhostFn, ctypes.c_void_p]
except AttributeError: pass

# CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
try: (cuGraphicsMapResources:=dll.cuGraphicsMapResources).restype, cuGraphicsMapResources.argtypes = CUresult, [ctypes.c_uint, ctypes.POINTER(CUgraphicsResource), CUstream]
except AttributeError: pass

# CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream)
try: (cuGraphicsUnmapResources:=dll.cuGraphicsUnmapResources).restype, cuGraphicsUnmapResources.argtypes = CUresult, [ctypes.c_uint, ctypes.POINTER(CUgraphicsResource), CUstream]
except AttributeError: pass

# CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
try: (cuStreamWriteValue32:=dll.cuStreamWriteValue32).restype, cuStreamWriteValue32.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
try: (cuStreamWaitValue32:=dll.cuStreamWaitValue32).restype, cuStreamWaitValue32.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
try: (cuStreamWriteValue64:=dll.cuStreamWriteValue64).restype, cuStreamWriteValue64.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
try: (cuStreamWaitValue64:=dll.cuStreamWaitValue64).restype, cuStreamWaitValue64.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags)
try: (cuStreamBatchMemOp:=dll.cuStreamBatchMemOp).restype, cuStreamBatchMemOp.argtypes = CUresult, [CUstream, ctypes.c_uint, ctypes.POINTER(CUstreamBatchMemOpParams), ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWriteValue32_ptsz(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
try: (cuStreamWriteValue32_ptsz:=dll.cuStreamWriteValue32_ptsz).restype, cuStreamWriteValue32_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWaitValue32_ptsz(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
try: (cuStreamWaitValue32_ptsz:=dll.cuStreamWaitValue32_ptsz).restype, cuStreamWaitValue32_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWriteValue64_ptsz(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
try: (cuStreamWriteValue64_ptsz:=dll.cuStreamWriteValue64_ptsz).restype, cuStreamWriteValue64_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWaitValue64_ptsz(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
try: (cuStreamWaitValue64_ptsz:=dll.cuStreamWaitValue64_ptsz).restype, cuStreamWaitValue64_ptsz.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamBatchMemOp_ptsz(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags)
try: (cuStreamBatchMemOp_ptsz:=dll.cuStreamBatchMemOp_ptsz).restype, cuStreamBatchMemOp_ptsz.argtypes = CUresult, [CUstream, ctypes.c_uint, ctypes.POINTER(CUstreamBatchMemOpParams), ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
try: (cuStreamWriteValue32_v2:=dll.cuStreamWriteValue32_v2).restype, cuStreamWriteValue32_v2.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
try: (cuStreamWaitValue32_v2:=dll.cuStreamWaitValue32_v2).restype, cuStreamWaitValue32_v2.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
try: (cuStreamWriteValue64_v2:=dll.cuStreamWriteValue64_v2).restype, cuStreamWriteValue64_v2.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
try: (cuStreamWaitValue64_v2:=dll.cuStreamWaitValue64_v2).restype, cuStreamWaitValue64_v2.argtypes = CUresult, [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags)
try: (cuStreamBatchMemOp_v2:=dll.cuStreamBatchMemOp_v2).restype, cuStreamBatchMemOp_v2.argtypes = CUresult, [CUstream, ctypes.c_uint, ctypes.POINTER(CUstreamBatchMemOpParams), ctypes.c_uint]
except AttributeError: pass

# CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream)
try: (cuMemPrefetchAsync:=dll.cuMemPrefetchAsync).restype, cuMemPrefetchAsync.argtypes = CUresult, [CUdeviceptr, size_t, CUdevice, CUstream]
except AttributeError: pass

# CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams)
try: (cuLaunchCooperativeKernel:=dll.cuLaunchCooperativeKernel).restype, cuLaunchCooperativeKernel.argtypes = CUresult, [CUfunction, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, CUstream, ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

# CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream)
try: (cuSignalExternalSemaphoresAsync:=dll.cuSignalExternalSemaphoresAsync).restype, cuSignalExternalSemaphoresAsync.argtypes = CUresult, [ctypes.POINTER(CUexternalSemaphore), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream)
try: (cuWaitExternalSemaphoresAsync:=dll.cuWaitExternalSemaphoresAsync).restype, cuWaitExternalSemaphoresAsync.argtypes = CUresult, [ctypes.POINTER(CUexternalSemaphore), ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS), ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuStreamBeginCapture(CUstream hStream)
try: (cuStreamBeginCapture:=dll.cuStreamBeginCapture).restype, cuStreamBeginCapture.argtypes = CUresult, [CUstream]
except AttributeError: pass

# CUresult cuStreamBeginCapture_ptsz(CUstream hStream)
try: (cuStreamBeginCapture_ptsz:=dll.cuStreamBeginCapture_ptsz).restype, cuStreamBeginCapture_ptsz.argtypes = CUresult, [CUstream]
except AttributeError: pass

# CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode)
try: (cuStreamBeginCapture_v2:=dll.cuStreamBeginCapture_v2).restype, cuStreamBeginCapture_v2.argtypes = CUresult, [CUstream, CUstreamCaptureMode]
except AttributeError: pass

# CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph)
try: (cuStreamEndCapture:=dll.cuStreamEndCapture).restype, cuStreamEndCapture.argtypes = CUresult, [CUstream, ctypes.POINTER(CUgraph)]
except AttributeError: pass

# CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus)
try: (cuStreamIsCapturing:=dll.cuStreamIsCapturing).restype, cuStreamIsCapturing.argtypes = CUresult, [CUstream, ctypes.POINTER(CUstreamCaptureStatus)]
except AttributeError: pass

# CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out)
try: (cuStreamGetCaptureInfo:=dll.cuStreamGetCaptureInfo).restype, cuStreamGetCaptureInfo.argtypes = CUresult, [CUstream, ctypes.POINTER(CUstreamCaptureStatus), ctypes.POINTER(cuuint64_t)]
except AttributeError: pass

# CUresult cuStreamGetCaptureInfo_ptsz(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out)
try: (cuStreamGetCaptureInfo_ptsz:=dll.cuStreamGetCaptureInfo_ptsz).restype, cuStreamGetCaptureInfo_ptsz.argtypes = CUresult, [CUstream, ctypes.POINTER(CUstreamCaptureStatus), ctypes.POINTER(cuuint64_t)]
except AttributeError: pass

# CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out)
try: (cuStreamGetCaptureInfo_v2:=dll.cuStreamGetCaptureInfo_v2).restype, cuStreamGetCaptureInfo_v2.argtypes = CUresult, [CUstream, ctypes.POINTER(CUstreamCaptureStatus), ctypes.POINTER(cuuint64_t), ctypes.POINTER(CUgraph), ctypes.POINTER(ctypes.POINTER(CUgraphNode)), ctypes.POINTER(size_t)]
except AttributeError: pass

# CUresult cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
try: (cuGraphAddKernelNode:=dll.cuGraphAddKernelNode).restype, cuGraphAddKernelNode.argtypes = CUresult, [ctypes.POINTER(CUgraphNode), CUgraph, ctypes.POINTER(CUgraphNode), size_t, ctypes.POINTER(CUDA_KERNEL_NODE_PARAMS_v1)]
except AttributeError: pass

# CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
try: (cuGraphKernelNodeGetParams:=dll.cuGraphKernelNodeGetParams).restype, cuGraphKernelNodeGetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_KERNEL_NODE_PARAMS_v1)]
except AttributeError: pass

# CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
try: (cuGraphKernelNodeSetParams:=dll.cuGraphKernelNodeSetParams).restype, cuGraphKernelNodeSetParams.argtypes = CUresult, [CUgraphNode, ctypes.POINTER(CUDA_KERNEL_NODE_PARAMS_v1)]
except AttributeError: pass

# CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams)
try: (cuGraphExecKernelNodeSetParams:=dll.cuGraphExecKernelNodeSetParams).restype, cuGraphExecKernelNodeSetParams.argtypes = CUresult, [CUgraphExec, CUgraphNode, ctypes.POINTER(CUDA_KERNEL_NODE_PARAMS_v1)]
except AttributeError: pass

# CUresult cuGraphInstantiateWithParams(CUgraphExec *phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams)
try: (cuGraphInstantiateWithParams:=dll.cuGraphInstantiateWithParams).restype, cuGraphInstantiateWithParams.argtypes = CUresult, [ctypes.POINTER(CUgraphExec), CUgraph, ctypes.POINTER(CUDA_GRAPH_INSTANTIATE_PARAMS)]
except AttributeError: pass

# CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, CUgraphExecUpdateResult *updateResult_out)
try: (cuGraphExecUpdate:=dll.cuGraphExecUpdate).restype, cuGraphExecUpdate.argtypes = CUresult, [CUgraphExec, CUgraph, ctypes.POINTER(CUgraphNode), ctypes.POINTER(CUgraphExecUpdateResult)]
except AttributeError: pass

# CUresult cuGraphUpload(CUgraphExec hGraph, CUstream hStream)
try: (cuGraphUpload:=dll.cuGraphUpload).restype, cuGraphUpload.argtypes = CUresult, [CUgraphExec, CUstream]
except AttributeError: pass

# CUresult cuGraphLaunch(CUgraphExec hGraph, CUstream hStream)
try: (cuGraphLaunch:=dll.cuGraphLaunch).restype, cuGraphLaunch.argtypes = CUresult, [CUgraphExec, CUstream]
except AttributeError: pass

# CUresult cuStreamCopyAttributes(CUstream dstStream, CUstream srcStream)
try: (cuStreamCopyAttributes:=dll.cuStreamCopyAttributes).restype, cuStreamCopyAttributes.argtypes = CUresult, [CUstream, CUstream]
except AttributeError: pass

# CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue *value)
try: (cuStreamGetAttribute:=dll.cuStreamGetAttribute).restype, cuStreamGetAttribute.argtypes = CUresult, [CUstream, CUstreamAttrID, ctypes.POINTER(CUstreamAttrValue)]
except AttributeError: pass

# CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *param)
try: (cuStreamSetAttribute:=dll.cuStreamSetAttribute).restype, cuStreamSetAttribute.argtypes = CUresult, [CUstream, CUstreamAttrID, ctypes.POINTER(CUstreamAttrValue)]
except AttributeError: pass

# CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags)
try: (cuIpcOpenMemHandle:=dll.cuIpcOpenMemHandle).restype, cuIpcOpenMemHandle.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), CUipcMemHandle, ctypes.c_uint]
except AttributeError: pass

# CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize)
try: (cuGraphInstantiate:=dll.cuGraphInstantiate).restype, cuGraphInstantiate.argtypes = CUresult, [ctypes.POINTER(CUgraphExec), CUgraph, ctypes.POINTER(CUgraphNode), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize)
try: (cuGraphInstantiate_v2:=dll.cuGraphInstantiate_v2).restype, cuGraphInstantiate_v2.argtypes = CUresult, [ctypes.POINTER(CUgraphExec), CUgraph, ctypes.POINTER(CUgraphNode), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream)
try: (cuMemMapArrayAsync:=dll.cuMemMapArrayAsync).restype, cuMemMapArrayAsync.argtypes = CUresult, [ctypes.POINTER(CUarrayMapInfo), ctypes.c_uint, CUstream]
except AttributeError: pass

# CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
try: (cuMemFreeAsync:=dll.cuMemFreeAsync).restype, cuMemFreeAsync.argtypes = CUresult, [CUdeviceptr, CUstream]
except AttributeError: pass

# CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream)
try: (cuMemAllocAsync:=dll.cuMemAllocAsync).restype, cuMemAllocAsync.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), size_t, CUstream]
except AttributeError: pass

# CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)
try: (cuMemAllocFromPoolAsync:=dll.cuMemAllocFromPoolAsync).restype, cuMemAllocFromPoolAsync.argtypes = CUresult, [ctypes.POINTER(CUdeviceptr), size_t, CUmemoryPool, CUstream]
except AttributeError: pass

# CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags)
try: (cuStreamUpdateCaptureDependencies:=dll.cuStreamUpdateCaptureDependencies).restype, cuStreamUpdateCaptureDependencies.argtypes = CUresult, [CUstream, ctypes.POINTER(CUgraphNode), size_t, ctypes.c_uint]
except AttributeError: pass

# CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
try: (cuGetProcAddress:=dll.cuGetProcAddress).restype, cuGetProcAddress.argtypes = CUresult, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, cuuint64_t]
except AttributeError: pass

__CUDA_API_PTDS = lambda api: api ## _ptds
__CUDA_API_PTSZ = lambda api: api ## _ptsz
cuDeviceTotalMem = cuDeviceTotalMem_v2
cuCtxCreate = cuCtxCreate_v2
cuCtxCreate_v3 = cuCtxCreate_v3
cuModuleGetGlobal = cuModuleGetGlobal_v2
cuMemGetInfo = cuMemGetInfo_v2
cuMemAlloc = cuMemAlloc_v2
cuMemAllocPitch = cuMemAllocPitch_v2
cuMemFree = cuMemFree_v2
cuMemGetAddressRange = cuMemGetAddressRange_v2
cuMemAllocHost = cuMemAllocHost_v2
cuMemHostGetDevicePointer = cuMemHostGetDevicePointer_v2
cuMemcpyHtoD = __CUDA_API_PTDS(cuMemcpyHtoD_v2)
cuMemcpyDtoH = __CUDA_API_PTDS(cuMemcpyDtoH_v2)
cuMemcpyDtoD = __CUDA_API_PTDS(cuMemcpyDtoD_v2)
cuMemcpyDtoA = __CUDA_API_PTDS(cuMemcpyDtoA_v2)
cuMemcpyAtoD = __CUDA_API_PTDS(cuMemcpyAtoD_v2)
cuMemcpyHtoA = __CUDA_API_PTDS(cuMemcpyHtoA_v2)
cuMemcpyAtoH = __CUDA_API_PTDS(cuMemcpyAtoH_v2)
cuMemcpyAtoA = __CUDA_API_PTDS(cuMemcpyAtoA_v2)
cuMemcpyHtoAAsync = __CUDA_API_PTSZ(cuMemcpyHtoAAsync_v2)
cuMemcpyAtoHAsync = __CUDA_API_PTSZ(cuMemcpyAtoHAsync_v2)
cuMemcpy2D = __CUDA_API_PTDS(cuMemcpy2D_v2)
cuMemcpy2DUnaligned = __CUDA_API_PTDS(cuMemcpy2DUnaligned_v2)
cuMemcpy3D = __CUDA_API_PTDS(cuMemcpy3D_v2)
cuMemcpyHtoDAsync = __CUDA_API_PTSZ(cuMemcpyHtoDAsync_v2)
cuMemcpyDtoHAsync = __CUDA_API_PTSZ(cuMemcpyDtoHAsync_v2)
cuMemcpyDtoDAsync = __CUDA_API_PTSZ(cuMemcpyDtoDAsync_v2)
cuMemcpy2DAsync = __CUDA_API_PTSZ(cuMemcpy2DAsync_v2)
cuMemcpy3DAsync = __CUDA_API_PTSZ(cuMemcpy3DAsync_v2)
cuMemsetD8 = __CUDA_API_PTDS(cuMemsetD8_v2)
cuMemsetD16 = __CUDA_API_PTDS(cuMemsetD16_v2)
cuMemsetD32 = __CUDA_API_PTDS(cuMemsetD32_v2)
cuMemsetD2D8 = __CUDA_API_PTDS(cuMemsetD2D8_v2)
cuMemsetD2D16 = __CUDA_API_PTDS(cuMemsetD2D16_v2)
cuMemsetD2D32 = __CUDA_API_PTDS(cuMemsetD2D32_v2)
cuArrayCreate = cuArrayCreate_v2
cuArrayGetDescriptor = cuArrayGetDescriptor_v2
cuArray3DCreate = cuArray3DCreate_v2
cuArray3DGetDescriptor = cuArray3DGetDescriptor_v2
cuTexRefSetAddress = cuTexRefSetAddress_v2
cuTexRefGetAddress = cuTexRefGetAddress_v2
cuGraphicsResourceGetMappedPointer = cuGraphicsResourceGetMappedPointer_v2
cuCtxDestroy = cuCtxDestroy_v2
cuCtxPopCurrent = cuCtxPopCurrent_v2
cuCtxPushCurrent = cuCtxPushCurrent_v2
cuStreamDestroy = cuStreamDestroy_v2
cuEventDestroy = cuEventDestroy_v2
cuTexRefSetAddress2D = cuTexRefSetAddress2D_v3
cuLinkCreate = cuLinkCreate_v2
cuLinkAddData = cuLinkAddData_v2
cuLinkAddFile = cuLinkAddFile_v2
cuMemHostRegister = cuMemHostRegister_v2
cuGraphicsResourceSetMapFlags = cuGraphicsResourceSetMapFlags_v2
cuStreamBeginCapture = __CUDA_API_PTSZ(cuStreamBeginCapture_v2)
cuDevicePrimaryCtxRelease = cuDevicePrimaryCtxRelease_v2
cuDevicePrimaryCtxReset = cuDevicePrimaryCtxReset_v2
cuDevicePrimaryCtxSetFlags = cuDevicePrimaryCtxSetFlags_v2
cuDeviceGetUuid_v2 = cuDeviceGetUuid_v2
cuIpcOpenMemHandle = cuIpcOpenMemHandle_v2
cuGraphInstantiate = cuGraphInstantiateWithFlags
cuGraphExecUpdate = cuGraphExecUpdate_v2
cuGetProcAddress = cuGetProcAddress_v2
cuGraphAddKernelNode = cuGraphAddKernelNode_v2
cuGraphKernelNodeGetParams = cuGraphKernelNodeGetParams_v2
cuGraphKernelNodeSetParams = cuGraphKernelNodeSetParams_v2
cuGraphExecKernelNodeSetParams = cuGraphExecKernelNodeSetParams_v2
cuStreamWriteValue32 = __CUDA_API_PTSZ(cuStreamWriteValue32_v2)
cuStreamWaitValue32 = __CUDA_API_PTSZ(cuStreamWaitValue32_v2)
cuStreamWriteValue64 = __CUDA_API_PTSZ(cuStreamWriteValue64_v2)
cuStreamWaitValue64 = __CUDA_API_PTSZ(cuStreamWaitValue64_v2)
cuStreamBatchMemOp = __CUDA_API_PTSZ(cuStreamBatchMemOp_v2)
cuStreamGetCaptureInfo = __CUDA_API_PTSZ(cuStreamGetCaptureInfo_v2)
cuStreamGetCaptureInfo_v2 = __CUDA_API_PTSZ(cuStreamGetCaptureInfo_v2)
cuMemcpy = __CUDA_API_PTDS(cuMemcpy)
cuMemcpyAsync = __CUDA_API_PTSZ(cuMemcpyAsync)
cuMemcpyPeer = __CUDA_API_PTDS(cuMemcpyPeer)
cuMemcpyPeerAsync = __CUDA_API_PTSZ(cuMemcpyPeerAsync)
cuMemcpy3DPeer = __CUDA_API_PTDS(cuMemcpy3DPeer)
cuMemcpy3DPeerAsync = __CUDA_API_PTSZ(cuMemcpy3DPeerAsync)
cuMemPrefetchAsync = __CUDA_API_PTSZ(cuMemPrefetchAsync)
cuMemsetD8Async = __CUDA_API_PTSZ(cuMemsetD8Async)
cuMemsetD16Async = __CUDA_API_PTSZ(cuMemsetD16Async)
cuMemsetD32Async = __CUDA_API_PTSZ(cuMemsetD32Async)
cuMemsetD2D8Async = __CUDA_API_PTSZ(cuMemsetD2D8Async)
cuMemsetD2D16Async = __CUDA_API_PTSZ(cuMemsetD2D16Async)
cuMemsetD2D32Async = __CUDA_API_PTSZ(cuMemsetD2D32Async)
cuStreamGetPriority = __CUDA_API_PTSZ(cuStreamGetPriority)
cuStreamGetId = __CUDA_API_PTSZ(cuStreamGetId)
cuStreamGetFlags = __CUDA_API_PTSZ(cuStreamGetFlags)
cuStreamGetCtx = __CUDA_API_PTSZ(cuStreamGetCtx)
cuStreamWaitEvent = __CUDA_API_PTSZ(cuStreamWaitEvent)
cuStreamEndCapture = __CUDA_API_PTSZ(cuStreamEndCapture)
cuStreamIsCapturing = __CUDA_API_PTSZ(cuStreamIsCapturing)
cuStreamUpdateCaptureDependencies = __CUDA_API_PTSZ(cuStreamUpdateCaptureDependencies)
cuStreamAddCallback = __CUDA_API_PTSZ(cuStreamAddCallback)
cuStreamAttachMemAsync = __CUDA_API_PTSZ(cuStreamAttachMemAsync)
cuStreamQuery = __CUDA_API_PTSZ(cuStreamQuery)
cuStreamSynchronize = __CUDA_API_PTSZ(cuStreamSynchronize)
cuEventRecord = __CUDA_API_PTSZ(cuEventRecord)
cuEventRecordWithFlags = __CUDA_API_PTSZ(cuEventRecordWithFlags)
cuLaunchKernel = __CUDA_API_PTSZ(cuLaunchKernel)
cuLaunchKernelEx = __CUDA_API_PTSZ(cuLaunchKernelEx)
cuLaunchHostFunc = __CUDA_API_PTSZ(cuLaunchHostFunc)
cuGraphicsMapResources = __CUDA_API_PTSZ(cuGraphicsMapResources)
cuGraphicsUnmapResources = __CUDA_API_PTSZ(cuGraphicsUnmapResources)
cuLaunchCooperativeKernel = __CUDA_API_PTSZ(cuLaunchCooperativeKernel)
cuSignalExternalSemaphoresAsync = __CUDA_API_PTSZ(cuSignalExternalSemaphoresAsync)
cuWaitExternalSemaphoresAsync = __CUDA_API_PTSZ(cuWaitExternalSemaphoresAsync)
cuGraphInstantiateWithParams = __CUDA_API_PTSZ(cuGraphInstantiateWithParams)
cuGraphUpload = __CUDA_API_PTSZ(cuGraphUpload)
cuGraphLaunch = __CUDA_API_PTSZ(cuGraphLaunch)
cuStreamCopyAttributes = __CUDA_API_PTSZ(cuStreamCopyAttributes)
cuStreamGetAttribute = __CUDA_API_PTSZ(cuStreamGetAttribute)
cuStreamSetAttribute = __CUDA_API_PTSZ(cuStreamSetAttribute)
cuMemMapArrayAsync = __CUDA_API_PTSZ(cuMemMapArrayAsync)
cuMemFreeAsync = __CUDA_API_PTSZ(cuMemFreeAsync)
cuMemAllocAsync = __CUDA_API_PTSZ(cuMemAllocAsync)
cuMemAllocFromPoolAsync = __CUDA_API_PTSZ(cuMemAllocFromPoolAsync)
CUDA_VERSION = 12000
CU_IPC_HANDLE_SIZE = 64
CU_COMPUTE_ACCELERATED_TARGET_BASE = 0x10000
CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = CU_LAUNCH_ATTRIBUTE_COOPERATIVE
CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
CU_KERNEL_NODE_ATTRIBUTE_PRIORITY = CU_LAUNCH_ATTRIBUTE_PRIORITY
CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP
CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN
CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY
CU_STREAM_ATTRIBUTE_PRIORITY = CU_LAUNCH_ATTRIBUTE_PRIORITY
CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP
CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN
CU_MEMHOSTALLOC_PORTABLE = 0x01
CU_MEMHOSTALLOC_DEVICEMAP = 0x02
CU_MEMHOSTALLOC_WRITECOMBINED = 0x04
CU_MEMHOSTREGISTER_PORTABLE = 0x01
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
CU_MEMHOSTREGISTER_IOMEMORY = 0x04
CU_MEMHOSTREGISTER_READ_ONLY = 0x08
CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL = 0x1
CU_TENSOR_MAP_NUM_QWORDS = 16
CUDA_EXTERNAL_MEMORY_DEDICATED = 0x1
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC = 0x01
CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC = 0x02
CUDA_NVSCISYNC_ATTR_SIGNAL = 0x1
CUDA_NVSCISYNC_ATTR_WAIT = 0x2
CU_MEM_CREATE_USAGE_TILE_POOL = 0x1
CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = 0x01
CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = 0x02
CUDA_ARRAY3D_LAYERED = 0x01
CUDA_ARRAY3D_2DARRAY = 0x01
CUDA_ARRAY3D_SURFACE_LDST = 0x02
CUDA_ARRAY3D_CUBEMAP = 0x04
CUDA_ARRAY3D_TEXTURE_GATHER = 0x08
CUDA_ARRAY3D_DEPTH_TEXTURE = 0x10
CUDA_ARRAY3D_COLOR_ATTACHMENT = 0x20
CUDA_ARRAY3D_SPARSE = 0x40
CUDA_ARRAY3D_DEFERRED_MAPPING = 0x80
CU_TRSA_OVERRIDE_FORMAT = 0x01
CU_TRSF_READ_AS_INTEGER = 0x01
CU_TRSF_NORMALIZED_COORDINATES = 0x02
CU_TRSF_SRGB = 0x10
CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION = 0x20
CU_TRSF_SEAMLESS_CUBEMAP = 0x40
CU_LAUNCH_PARAM_END_AS_INT = 0x00
CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT = 0x01
CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT = 0x02
CU_PARAM_TR_DEFAULT = -1