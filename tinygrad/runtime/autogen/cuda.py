# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('cuda', 'cuda')
cuuint32_t = ctypes.c_uint32
cuuint64_t = ctypes.c_uint64
CUdeviceptr_v2 = ctypes.c_uint64
CUdeviceptr = ctypes.c_uint64
CUdevice_v1 = ctypes.c_int32
CUdevice = ctypes.c_int32
class struct_CUctx_st(Struct): pass
CUcontext = Pointer(struct_CUctx_st)
class struct_CUmod_st(Struct): pass
CUmodule = Pointer(struct_CUmod_st)
class struct_CUfunc_st(Struct): pass
CUfunction = Pointer(struct_CUfunc_st)
class struct_CUlib_st(Struct): pass
CUlibrary = Pointer(struct_CUlib_st)
class struct_CUkern_st(Struct): pass
CUkernel = Pointer(struct_CUkern_st)
class struct_CUarray_st(Struct): pass
CUarray = Pointer(struct_CUarray_st)
class struct_CUmipmappedArray_st(Struct): pass
CUmipmappedArray = Pointer(struct_CUmipmappedArray_st)
class struct_CUtexref_st(Struct): pass
CUtexref = Pointer(struct_CUtexref_st)
class struct_CUsurfref_st(Struct): pass
CUsurfref = Pointer(struct_CUsurfref_st)
class struct_CUevent_st(Struct): pass
CUevent = Pointer(struct_CUevent_st)
class struct_CUstream_st(Struct): pass
CUstream = Pointer(struct_CUstream_st)
class struct_CUgraphicsResource_st(Struct): pass
CUgraphicsResource = Pointer(struct_CUgraphicsResource_st)
CUtexObject_v1 = ctypes.c_uint64
CUtexObject = ctypes.c_uint64
CUsurfObject_v1 = ctypes.c_uint64
CUsurfObject = ctypes.c_uint64
class struct_CUextMemory_st(Struct): pass
CUexternalMemory = Pointer(struct_CUextMemory_st)
class struct_CUextSemaphore_st(Struct): pass
CUexternalSemaphore = Pointer(struct_CUextSemaphore_st)
class struct_CUgraph_st(Struct): pass
CUgraph = Pointer(struct_CUgraph_st)
class struct_CUgraphNode_st(Struct): pass
CUgraphNode = Pointer(struct_CUgraphNode_st)
class struct_CUgraphExec_st(Struct): pass
CUgraphExec = Pointer(struct_CUgraphExec_st)
class struct_CUmemPoolHandle_st(Struct): pass
CUmemoryPool = Pointer(struct_CUmemPoolHandle_st)
class struct_CUuserObject_st(Struct): pass
CUuserObject = Pointer(struct_CUuserObject_st)
class struct_CUuuid_st(Struct): pass
struct_CUuuid_st.SIZE = 16
struct_CUuuid_st._fields_ = ['bytes']
setattr(struct_CUuuid_st, 'bytes', field(0, Array(ctypes.c_char, 16)))
CUuuid = struct_CUuuid_st
class struct_CUipcEventHandle_st(Struct): pass
struct_CUipcEventHandle_st.SIZE = 64
struct_CUipcEventHandle_st._fields_ = ['reserved']
setattr(struct_CUipcEventHandle_st, 'reserved', field(0, Array(ctypes.c_char, 64)))
CUipcEventHandle_v1 = struct_CUipcEventHandle_st
CUipcEventHandle = struct_CUipcEventHandle_st
class struct_CUipcMemHandle_st(Struct): pass
struct_CUipcMemHandle_st.SIZE = 64
struct_CUipcMemHandle_st._fields_ = ['reserved']
setattr(struct_CUipcMemHandle_st, 'reserved', field(0, Array(ctypes.c_char, 64)))
CUipcMemHandle_v1 = struct_CUipcMemHandle_st
CUipcMemHandle = struct_CUipcMemHandle_st
enum_CUipcMem_flags_enum = CEnum(ctypes.c_uint32)
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = enum_CUipcMem_flags_enum.define('CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS', 1)

CUipcMem_flags = enum_CUipcMem_flags_enum
enum_CUmemAttach_flags_enum = CEnum(ctypes.c_uint32)
CU_MEM_ATTACH_GLOBAL = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_GLOBAL', 1)
CU_MEM_ATTACH_HOST = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_HOST', 2)
CU_MEM_ATTACH_SINGLE = enum_CUmemAttach_flags_enum.define('CU_MEM_ATTACH_SINGLE', 4)

CUmemAttach_flags = enum_CUmemAttach_flags_enum
enum_CUctx_flags_enum = CEnum(ctypes.c_uint32)
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
enum_CUevent_sched_flags_enum = CEnum(ctypes.c_uint32)
CU_EVENT_SCHED_AUTO = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_AUTO', 0)
CU_EVENT_SCHED_SPIN = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_SPIN', 1)
CU_EVENT_SCHED_YIELD = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_YIELD', 2)
CU_EVENT_SCHED_BLOCKING_SYNC = enum_CUevent_sched_flags_enum.define('CU_EVENT_SCHED_BLOCKING_SYNC', 4)

CUevent_sched_flags = enum_CUevent_sched_flags_enum
enum_cl_event_flags_enum = CEnum(ctypes.c_uint32)
NVCL_EVENT_SCHED_AUTO = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_AUTO', 0)
NVCL_EVENT_SCHED_SPIN = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_SPIN', 1)
NVCL_EVENT_SCHED_YIELD = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_YIELD', 2)
NVCL_EVENT_SCHED_BLOCKING_SYNC = enum_cl_event_flags_enum.define('NVCL_EVENT_SCHED_BLOCKING_SYNC', 4)

cl_event_flags = enum_cl_event_flags_enum
enum_cl_context_flags_enum = CEnum(ctypes.c_uint32)
NVCL_CTX_SCHED_AUTO = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_AUTO', 0)
NVCL_CTX_SCHED_SPIN = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_SPIN', 1)
NVCL_CTX_SCHED_YIELD = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_YIELD', 2)
NVCL_CTX_SCHED_BLOCKING_SYNC = enum_cl_context_flags_enum.define('NVCL_CTX_SCHED_BLOCKING_SYNC', 4)

cl_context_flags = enum_cl_context_flags_enum
enum_CUstream_flags_enum = CEnum(ctypes.c_uint32)
CU_STREAM_DEFAULT = enum_CUstream_flags_enum.define('CU_STREAM_DEFAULT', 0)
CU_STREAM_NON_BLOCKING = enum_CUstream_flags_enum.define('CU_STREAM_NON_BLOCKING', 1)

CUstream_flags = enum_CUstream_flags_enum
enum_CUevent_flags_enum = CEnum(ctypes.c_uint32)
CU_EVENT_DEFAULT = enum_CUevent_flags_enum.define('CU_EVENT_DEFAULT', 0)
CU_EVENT_BLOCKING_SYNC = enum_CUevent_flags_enum.define('CU_EVENT_BLOCKING_SYNC', 1)
CU_EVENT_DISABLE_TIMING = enum_CUevent_flags_enum.define('CU_EVENT_DISABLE_TIMING', 2)
CU_EVENT_INTERPROCESS = enum_CUevent_flags_enum.define('CU_EVENT_INTERPROCESS', 4)

CUevent_flags = enum_CUevent_flags_enum
enum_CUevent_record_flags_enum = CEnum(ctypes.c_uint32)
CU_EVENT_RECORD_DEFAULT = enum_CUevent_record_flags_enum.define('CU_EVENT_RECORD_DEFAULT', 0)
CU_EVENT_RECORD_EXTERNAL = enum_CUevent_record_flags_enum.define('CU_EVENT_RECORD_EXTERNAL', 1)

CUevent_record_flags = enum_CUevent_record_flags_enum
enum_CUevent_wait_flags_enum = CEnum(ctypes.c_uint32)
CU_EVENT_WAIT_DEFAULT = enum_CUevent_wait_flags_enum.define('CU_EVENT_WAIT_DEFAULT', 0)
CU_EVENT_WAIT_EXTERNAL = enum_CUevent_wait_flags_enum.define('CU_EVENT_WAIT_EXTERNAL', 1)

CUevent_wait_flags = enum_CUevent_wait_flags_enum
enum_CUstreamWaitValue_flags_enum = CEnum(ctypes.c_uint32)
CU_STREAM_WAIT_VALUE_GEQ = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_GEQ', 0)
CU_STREAM_WAIT_VALUE_EQ = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_EQ', 1)
CU_STREAM_WAIT_VALUE_AND = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_AND', 2)
CU_STREAM_WAIT_VALUE_NOR = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_NOR', 3)
CU_STREAM_WAIT_VALUE_FLUSH = enum_CUstreamWaitValue_flags_enum.define('CU_STREAM_WAIT_VALUE_FLUSH', 1073741824)

CUstreamWaitValue_flags = enum_CUstreamWaitValue_flags_enum
enum_CUstreamWriteValue_flags_enum = CEnum(ctypes.c_uint32)
CU_STREAM_WRITE_VALUE_DEFAULT = enum_CUstreamWriteValue_flags_enum.define('CU_STREAM_WRITE_VALUE_DEFAULT', 0)
CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = enum_CUstreamWriteValue_flags_enum.define('CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER', 1)

CUstreamWriteValue_flags = enum_CUstreamWriteValue_flags_enum
enum_CUstreamBatchMemOpType_enum = CEnum(ctypes.c_uint32)
CU_STREAM_MEM_OP_WAIT_VALUE_32 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WAIT_VALUE_32', 1)
CU_STREAM_MEM_OP_WRITE_VALUE_32 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WRITE_VALUE_32', 2)
CU_STREAM_MEM_OP_WAIT_VALUE_64 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WAIT_VALUE_64', 4)
CU_STREAM_MEM_OP_WRITE_VALUE_64 = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_WRITE_VALUE_64', 5)
CU_STREAM_MEM_OP_BARRIER = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_BARRIER', 6)
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = enum_CUstreamBatchMemOpType_enum.define('CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES', 3)

CUstreamBatchMemOpType = enum_CUstreamBatchMemOpType_enum
enum_CUstreamMemoryBarrier_flags_enum = CEnum(ctypes.c_uint32)
CU_STREAM_MEMORY_BARRIER_TYPE_SYS = enum_CUstreamMemoryBarrier_flags_enum.define('CU_STREAM_MEMORY_BARRIER_TYPE_SYS', 0)
CU_STREAM_MEMORY_BARRIER_TYPE_GPU = enum_CUstreamMemoryBarrier_flags_enum.define('CU_STREAM_MEMORY_BARRIER_TYPE_GPU', 1)

CUstreamMemoryBarrier_flags = enum_CUstreamMemoryBarrier_flags_enum
class union_CUstreamBatchMemOpParams_union(Union): pass
class struct_CUstreamMemOpWaitValueParams_st(Struct): pass
struct_CUstreamMemOpWaitValueParams_st.SIZE = 40
struct_CUstreamMemOpWaitValueParams_st._fields_ = ['operation', 'address', 'value', 'value64', 'flags', 'alias']
setattr(struct_CUstreamMemOpWaitValueParams_st, 'operation', field(0, CUstreamBatchMemOpType))
setattr(struct_CUstreamMemOpWaitValueParams_st, 'address', field(8, CUdeviceptr))
setattr(struct_CUstreamMemOpWaitValueParams_st, 'value', field(16, cuuint32_t))
setattr(struct_CUstreamMemOpWaitValueParams_st, 'value64', field(16, cuuint64_t))
setattr(struct_CUstreamMemOpWaitValueParams_st, 'flags', field(24, ctypes.c_uint32))
setattr(struct_CUstreamMemOpWaitValueParams_st, 'alias', field(32, CUdeviceptr))
class struct_CUstreamMemOpWriteValueParams_st(Struct): pass
struct_CUstreamMemOpWriteValueParams_st.SIZE = 40
struct_CUstreamMemOpWriteValueParams_st._fields_ = ['operation', 'address', 'value', 'value64', 'flags', 'alias']
setattr(struct_CUstreamMemOpWriteValueParams_st, 'operation', field(0, CUstreamBatchMemOpType))
setattr(struct_CUstreamMemOpWriteValueParams_st, 'address', field(8, CUdeviceptr))
setattr(struct_CUstreamMemOpWriteValueParams_st, 'value', field(16, cuuint32_t))
setattr(struct_CUstreamMemOpWriteValueParams_st, 'value64', field(16, cuuint64_t))
setattr(struct_CUstreamMemOpWriteValueParams_st, 'flags', field(24, ctypes.c_uint32))
setattr(struct_CUstreamMemOpWriteValueParams_st, 'alias', field(32, CUdeviceptr))
class struct_CUstreamMemOpFlushRemoteWritesParams_st(Struct): pass
struct_CUstreamMemOpFlushRemoteWritesParams_st.SIZE = 8
struct_CUstreamMemOpFlushRemoteWritesParams_st._fields_ = ['operation', 'flags']
setattr(struct_CUstreamMemOpFlushRemoteWritesParams_st, 'operation', field(0, CUstreamBatchMemOpType))
setattr(struct_CUstreamMemOpFlushRemoteWritesParams_st, 'flags', field(4, ctypes.c_uint32))
class struct_CUstreamMemOpMemoryBarrierParams_st(Struct): pass
struct_CUstreamMemOpMemoryBarrierParams_st.SIZE = 8
struct_CUstreamMemOpMemoryBarrierParams_st._fields_ = ['operation', 'flags']
setattr(struct_CUstreamMemOpMemoryBarrierParams_st, 'operation', field(0, CUstreamBatchMemOpType))
setattr(struct_CUstreamMemOpMemoryBarrierParams_st, 'flags', field(4, ctypes.c_uint32))
union_CUstreamBatchMemOpParams_union.SIZE = 48
union_CUstreamBatchMemOpParams_union._fields_ = ['operation', 'waitValue', 'writeValue', 'flushRemoteWrites', 'memoryBarrier', 'pad']
setattr(union_CUstreamBatchMemOpParams_union, 'operation', field(0, CUstreamBatchMemOpType))
setattr(union_CUstreamBatchMemOpParams_union, 'waitValue', field(0, struct_CUstreamMemOpWaitValueParams_st))
setattr(union_CUstreamBatchMemOpParams_union, 'writeValue', field(0, struct_CUstreamMemOpWriteValueParams_st))
setattr(union_CUstreamBatchMemOpParams_union, 'flushRemoteWrites', field(0, struct_CUstreamMemOpFlushRemoteWritesParams_st))
setattr(union_CUstreamBatchMemOpParams_union, 'memoryBarrier', field(0, struct_CUstreamMemOpMemoryBarrierParams_st))
setattr(union_CUstreamBatchMemOpParams_union, 'pad', field(0, Array(cuuint64_t, 6)))
CUstreamBatchMemOpParams_v1 = union_CUstreamBatchMemOpParams_union
CUstreamBatchMemOpParams = union_CUstreamBatchMemOpParams_union
class struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st(Struct): pass
struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st.SIZE = 32
struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st._fields_ = ['ctx', 'count', 'paramArray', 'flags']
setattr(struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st, 'ctx', field(0, CUcontext))
setattr(struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st, 'count', field(8, ctypes.c_uint32))
setattr(struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st, 'paramArray', field(16, Pointer(CUstreamBatchMemOpParams)))
setattr(struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st, 'flags', field(24, ctypes.c_uint32))
CUDA_BATCH_MEM_OP_NODE_PARAMS = struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_st
enum_CUoccupancy_flags_enum = CEnum(ctypes.c_uint32)
CU_OCCUPANCY_DEFAULT = enum_CUoccupancy_flags_enum.define('CU_OCCUPANCY_DEFAULT', 0)
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = enum_CUoccupancy_flags_enum.define('CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE', 1)

CUoccupancy_flags = enum_CUoccupancy_flags_enum
enum_CUstreamUpdateCaptureDependencies_flags_enum = CEnum(ctypes.c_uint32)
CU_STREAM_ADD_CAPTURE_DEPENDENCIES = enum_CUstreamUpdateCaptureDependencies_flags_enum.define('CU_STREAM_ADD_CAPTURE_DEPENDENCIES', 0)
CU_STREAM_SET_CAPTURE_DEPENDENCIES = enum_CUstreamUpdateCaptureDependencies_flags_enum.define('CU_STREAM_SET_CAPTURE_DEPENDENCIES', 1)

CUstreamUpdateCaptureDependencies_flags = enum_CUstreamUpdateCaptureDependencies_flags_enum
enum_CUarray_format_enum = CEnum(ctypes.c_uint32)
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
enum_CUaddress_mode_enum = CEnum(ctypes.c_uint32)
CU_TR_ADDRESS_MODE_WRAP = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_WRAP', 0)
CU_TR_ADDRESS_MODE_CLAMP = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_CLAMP', 1)
CU_TR_ADDRESS_MODE_MIRROR = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_MIRROR', 2)
CU_TR_ADDRESS_MODE_BORDER = enum_CUaddress_mode_enum.define('CU_TR_ADDRESS_MODE_BORDER', 3)

CUaddress_mode = enum_CUaddress_mode_enum
enum_CUfilter_mode_enum = CEnum(ctypes.c_uint32)
CU_TR_FILTER_MODE_POINT = enum_CUfilter_mode_enum.define('CU_TR_FILTER_MODE_POINT', 0)
CU_TR_FILTER_MODE_LINEAR = enum_CUfilter_mode_enum.define('CU_TR_FILTER_MODE_LINEAR', 1)

CUfilter_mode = enum_CUfilter_mode_enum
enum_CUdevice_attribute_enum = CEnum(ctypes.c_uint32)
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
class struct_CUdevprop_st(Struct): pass
struct_CUdevprop_st.SIZE = 56
struct_CUdevprop_st._fields_ = ['maxThreadsPerBlock', 'maxThreadsDim', 'maxGridSize', 'sharedMemPerBlock', 'totalConstantMemory', 'SIMDWidth', 'memPitch', 'regsPerBlock', 'clockRate', 'textureAlign']
setattr(struct_CUdevprop_st, 'maxThreadsPerBlock', field(0, ctypes.c_int32))
setattr(struct_CUdevprop_st, 'maxThreadsDim', field(4, Array(ctypes.c_int32, 3)))
setattr(struct_CUdevprop_st, 'maxGridSize', field(16, Array(ctypes.c_int32, 3)))
setattr(struct_CUdevprop_st, 'sharedMemPerBlock', field(28, ctypes.c_int32))
setattr(struct_CUdevprop_st, 'totalConstantMemory', field(32, ctypes.c_int32))
setattr(struct_CUdevprop_st, 'SIMDWidth', field(36, ctypes.c_int32))
setattr(struct_CUdevprop_st, 'memPitch', field(40, ctypes.c_int32))
setattr(struct_CUdevprop_st, 'regsPerBlock', field(44, ctypes.c_int32))
setattr(struct_CUdevprop_st, 'clockRate', field(48, ctypes.c_int32))
setattr(struct_CUdevprop_st, 'textureAlign', field(52, ctypes.c_int32))
CUdevprop_v1 = struct_CUdevprop_st
CUdevprop = struct_CUdevprop_st
enum_CUpointer_attribute_enum = CEnum(ctypes.c_uint32)
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
enum_CUfunction_attribute_enum = CEnum(ctypes.c_uint32)
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
enum_CUfunc_cache_enum = CEnum(ctypes.c_uint32)
CU_FUNC_CACHE_PREFER_NONE = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_NONE', 0)
CU_FUNC_CACHE_PREFER_SHARED = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_SHARED', 1)
CU_FUNC_CACHE_PREFER_L1 = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_L1', 2)
CU_FUNC_CACHE_PREFER_EQUAL = enum_CUfunc_cache_enum.define('CU_FUNC_CACHE_PREFER_EQUAL', 3)

CUfunc_cache = enum_CUfunc_cache_enum
enum_CUsharedconfig_enum = CEnum(ctypes.c_uint32)
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE', 0)
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE', 1)
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = enum_CUsharedconfig_enum.define('CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE', 2)

CUsharedconfig = enum_CUsharedconfig_enum
enum_CUshared_carveout_enum = CEnum(ctypes.c_int32)
CU_SHAREDMEM_CARVEOUT_DEFAULT = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_DEFAULT', -1)
CU_SHAREDMEM_CARVEOUT_MAX_SHARED = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_MAX_SHARED', 100)
CU_SHAREDMEM_CARVEOUT_MAX_L1 = enum_CUshared_carveout_enum.define('CU_SHAREDMEM_CARVEOUT_MAX_L1', 0)

CUshared_carveout = enum_CUshared_carveout_enum
enum_CUmemorytype_enum = CEnum(ctypes.c_uint32)
CU_MEMORYTYPE_HOST = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_HOST', 1)
CU_MEMORYTYPE_DEVICE = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_DEVICE', 2)
CU_MEMORYTYPE_ARRAY = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_ARRAY', 3)
CU_MEMORYTYPE_UNIFIED = enum_CUmemorytype_enum.define('CU_MEMORYTYPE_UNIFIED', 4)

CUmemorytype = enum_CUmemorytype_enum
enum_CUcomputemode_enum = CEnum(ctypes.c_uint32)
CU_COMPUTEMODE_DEFAULT = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_DEFAULT', 0)
CU_COMPUTEMODE_PROHIBITED = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_PROHIBITED', 2)
CU_COMPUTEMODE_EXCLUSIVE_PROCESS = enum_CUcomputemode_enum.define('CU_COMPUTEMODE_EXCLUSIVE_PROCESS', 3)

CUcomputemode = enum_CUcomputemode_enum
enum_CUmem_advise_enum = CEnum(ctypes.c_uint32)
CU_MEM_ADVISE_SET_READ_MOSTLY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_READ_MOSTLY', 1)
CU_MEM_ADVISE_UNSET_READ_MOSTLY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_READ_MOSTLY', 2)
CU_MEM_ADVISE_SET_PREFERRED_LOCATION = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_PREFERRED_LOCATION', 3)
CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION', 4)
CU_MEM_ADVISE_SET_ACCESSED_BY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_SET_ACCESSED_BY', 5)
CU_MEM_ADVISE_UNSET_ACCESSED_BY = enum_CUmem_advise_enum.define('CU_MEM_ADVISE_UNSET_ACCESSED_BY', 6)

CUmem_advise = enum_CUmem_advise_enum
enum_CUmem_range_attribute_enum = CEnum(ctypes.c_uint32)
CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY', 1)
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION', 2)
CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY', 3)
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = enum_CUmem_range_attribute_enum.define('CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION', 4)

CUmem_range_attribute = enum_CUmem_range_attribute_enum
enum_CUjit_option_enum = CEnum(ctypes.c_uint32)
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
enum_CUjit_target_enum = CEnum(ctypes.c_uint32)
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
enum_CUjit_fallback_enum = CEnum(ctypes.c_uint32)
CU_PREFER_PTX = enum_CUjit_fallback_enum.define('CU_PREFER_PTX', 0)
CU_PREFER_BINARY = enum_CUjit_fallback_enum.define('CU_PREFER_BINARY', 1)

CUjit_fallback = enum_CUjit_fallback_enum
enum_CUjit_cacheMode_enum = CEnum(ctypes.c_uint32)
CU_JIT_CACHE_OPTION_NONE = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_NONE', 0)
CU_JIT_CACHE_OPTION_CG = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_CG', 1)
CU_JIT_CACHE_OPTION_CA = enum_CUjit_cacheMode_enum.define('CU_JIT_CACHE_OPTION_CA', 2)

CUjit_cacheMode = enum_CUjit_cacheMode_enum
enum_CUjitInputType_enum = CEnum(ctypes.c_uint32)
CU_JIT_INPUT_CUBIN = enum_CUjitInputType_enum.define('CU_JIT_INPUT_CUBIN', 0)
CU_JIT_INPUT_PTX = enum_CUjitInputType_enum.define('CU_JIT_INPUT_PTX', 1)
CU_JIT_INPUT_FATBINARY = enum_CUjitInputType_enum.define('CU_JIT_INPUT_FATBINARY', 2)
CU_JIT_INPUT_OBJECT = enum_CUjitInputType_enum.define('CU_JIT_INPUT_OBJECT', 3)
CU_JIT_INPUT_LIBRARY = enum_CUjitInputType_enum.define('CU_JIT_INPUT_LIBRARY', 4)
CU_JIT_INPUT_NVVM = enum_CUjitInputType_enum.define('CU_JIT_INPUT_NVVM', 5)
CU_JIT_NUM_INPUT_TYPES = enum_CUjitInputType_enum.define('CU_JIT_NUM_INPUT_TYPES', 6)

CUjitInputType = enum_CUjitInputType_enum
class struct_CUlinkState_st(Struct): pass
CUlinkState = Pointer(struct_CUlinkState_st)
enum_CUgraphicsRegisterFlags_enum = CEnum(ctypes.c_uint32)
CU_GRAPHICS_REGISTER_FLAGS_NONE = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_NONE', 0)
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY', 1)
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD', 2)
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST', 4)
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = enum_CUgraphicsRegisterFlags_enum.define('CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER', 8)

CUgraphicsRegisterFlags = enum_CUgraphicsRegisterFlags_enum
enum_CUgraphicsMapResourceFlags_enum = CEnum(ctypes.c_uint32)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE', 0)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY', 1)
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = enum_CUgraphicsMapResourceFlags_enum.define('CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD', 2)

CUgraphicsMapResourceFlags = enum_CUgraphicsMapResourceFlags_enum
enum_CUarray_cubemap_face_enum = CEnum(ctypes.c_uint32)
CU_CUBEMAP_FACE_POSITIVE_X = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_X', 0)
CU_CUBEMAP_FACE_NEGATIVE_X = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_X', 1)
CU_CUBEMAP_FACE_POSITIVE_Y = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_Y', 2)
CU_CUBEMAP_FACE_NEGATIVE_Y = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_Y', 3)
CU_CUBEMAP_FACE_POSITIVE_Z = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_POSITIVE_Z', 4)
CU_CUBEMAP_FACE_NEGATIVE_Z = enum_CUarray_cubemap_face_enum.define('CU_CUBEMAP_FACE_NEGATIVE_Z', 5)

CUarray_cubemap_face = enum_CUarray_cubemap_face_enum
enum_CUlimit_enum = CEnum(ctypes.c_uint32)
CU_LIMIT_STACK_SIZE = enum_CUlimit_enum.define('CU_LIMIT_STACK_SIZE', 0)
CU_LIMIT_PRINTF_FIFO_SIZE = enum_CUlimit_enum.define('CU_LIMIT_PRINTF_FIFO_SIZE', 1)
CU_LIMIT_MALLOC_HEAP_SIZE = enum_CUlimit_enum.define('CU_LIMIT_MALLOC_HEAP_SIZE', 2)
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = enum_CUlimit_enum.define('CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH', 3)
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = enum_CUlimit_enum.define('CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT', 4)
CU_LIMIT_MAX_L2_FETCH_GRANULARITY = enum_CUlimit_enum.define('CU_LIMIT_MAX_L2_FETCH_GRANULARITY', 5)
CU_LIMIT_PERSISTING_L2_CACHE_SIZE = enum_CUlimit_enum.define('CU_LIMIT_PERSISTING_L2_CACHE_SIZE', 6)
CU_LIMIT_MAX = enum_CUlimit_enum.define('CU_LIMIT_MAX', 7)

CUlimit = enum_CUlimit_enum
enum_CUresourcetype_enum = CEnum(ctypes.c_uint32)
CU_RESOURCE_TYPE_ARRAY = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_ARRAY', 0)
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_MIPMAPPED_ARRAY', 1)
CU_RESOURCE_TYPE_LINEAR = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_LINEAR', 2)
CU_RESOURCE_TYPE_PITCH2D = enum_CUresourcetype_enum.define('CU_RESOURCE_TYPE_PITCH2D', 3)

CUresourcetype = enum_CUresourcetype_enum
CUhostFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
enum_CUaccessProperty_enum = CEnum(ctypes.c_uint32)
CU_ACCESS_PROPERTY_NORMAL = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_NORMAL', 0)
CU_ACCESS_PROPERTY_STREAMING = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_STREAMING', 1)
CU_ACCESS_PROPERTY_PERSISTING = enum_CUaccessProperty_enum.define('CU_ACCESS_PROPERTY_PERSISTING', 2)

CUaccessProperty = enum_CUaccessProperty_enum
class struct_CUaccessPolicyWindow_st(Struct): pass
size_t = ctypes.c_uint64
struct_CUaccessPolicyWindow_st.SIZE = 32
struct_CUaccessPolicyWindow_st._fields_ = ['base_ptr', 'num_bytes', 'hitRatio', 'hitProp', 'missProp']
setattr(struct_CUaccessPolicyWindow_st, 'base_ptr', field(0, ctypes.c_void_p))
setattr(struct_CUaccessPolicyWindow_st, 'num_bytes', field(8, size_t))
setattr(struct_CUaccessPolicyWindow_st, 'hitRatio', field(16, ctypes.c_float))
setattr(struct_CUaccessPolicyWindow_st, 'hitProp', field(20, CUaccessProperty))
setattr(struct_CUaccessPolicyWindow_st, 'missProp', field(24, CUaccessProperty))
CUaccessPolicyWindow_v1 = struct_CUaccessPolicyWindow_st
CUaccessPolicyWindow = struct_CUaccessPolicyWindow_st
class struct_CUDA_KERNEL_NODE_PARAMS_st(Struct): pass
struct_CUDA_KERNEL_NODE_PARAMS_st.SIZE = 56
struct_CUDA_KERNEL_NODE_PARAMS_st._fields_ = ['func', 'gridDimX', 'gridDimY', 'gridDimZ', 'blockDimX', 'blockDimY', 'blockDimZ', 'sharedMemBytes', 'kernelParams', 'extra']
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'func', field(0, CUfunction))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'gridDimX', field(8, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'gridDimY', field(12, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'gridDimZ', field(16, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'blockDimX', field(20, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'blockDimY', field(24, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'blockDimZ', field(28, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'sharedMemBytes', field(32, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'kernelParams', field(40, Pointer(ctypes.c_void_p)))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_st, 'extra', field(48, Pointer(ctypes.c_void_p)))
CUDA_KERNEL_NODE_PARAMS_v1 = struct_CUDA_KERNEL_NODE_PARAMS_st
class struct_CUDA_KERNEL_NODE_PARAMS_v2_st(Struct): pass
struct_CUDA_KERNEL_NODE_PARAMS_v2_st.SIZE = 72
struct_CUDA_KERNEL_NODE_PARAMS_v2_st._fields_ = ['func', 'gridDimX', 'gridDimY', 'gridDimZ', 'blockDimX', 'blockDimY', 'blockDimZ', 'sharedMemBytes', 'kernelParams', 'extra', 'kern', 'ctx']
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'func', field(0, CUfunction))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'gridDimX', field(8, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'gridDimY', field(12, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'gridDimZ', field(16, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'blockDimX', field(20, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'blockDimY', field(24, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'blockDimZ', field(28, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'sharedMemBytes', field(32, ctypes.c_uint32))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'kernelParams', field(40, Pointer(ctypes.c_void_p)))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'extra', field(48, Pointer(ctypes.c_void_p)))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'kern', field(56, CUkernel))
setattr(struct_CUDA_KERNEL_NODE_PARAMS_v2_st, 'ctx', field(64, CUcontext))
CUDA_KERNEL_NODE_PARAMS_v2 = struct_CUDA_KERNEL_NODE_PARAMS_v2_st
CUDA_KERNEL_NODE_PARAMS = struct_CUDA_KERNEL_NODE_PARAMS_v2_st
class struct_CUDA_MEMSET_NODE_PARAMS_st(Struct): pass
struct_CUDA_MEMSET_NODE_PARAMS_st.SIZE = 40
struct_CUDA_MEMSET_NODE_PARAMS_st._fields_ = ['dst', 'pitch', 'value', 'elementSize', 'width', 'height']
setattr(struct_CUDA_MEMSET_NODE_PARAMS_st, 'dst', field(0, CUdeviceptr))
setattr(struct_CUDA_MEMSET_NODE_PARAMS_st, 'pitch', field(8, size_t))
setattr(struct_CUDA_MEMSET_NODE_PARAMS_st, 'value', field(16, ctypes.c_uint32))
setattr(struct_CUDA_MEMSET_NODE_PARAMS_st, 'elementSize', field(20, ctypes.c_uint32))
setattr(struct_CUDA_MEMSET_NODE_PARAMS_st, 'width', field(24, size_t))
setattr(struct_CUDA_MEMSET_NODE_PARAMS_st, 'height', field(32, size_t))
CUDA_MEMSET_NODE_PARAMS_v1 = struct_CUDA_MEMSET_NODE_PARAMS_st
CUDA_MEMSET_NODE_PARAMS = struct_CUDA_MEMSET_NODE_PARAMS_st
class struct_CUDA_HOST_NODE_PARAMS_st(Struct): pass
struct_CUDA_HOST_NODE_PARAMS_st.SIZE = 16
struct_CUDA_HOST_NODE_PARAMS_st._fields_ = ['fn', 'userData']
setattr(struct_CUDA_HOST_NODE_PARAMS_st, 'fn', field(0, CUhostFn))
setattr(struct_CUDA_HOST_NODE_PARAMS_st, 'userData', field(8, ctypes.c_void_p))
CUDA_HOST_NODE_PARAMS_v1 = struct_CUDA_HOST_NODE_PARAMS_st
CUDA_HOST_NODE_PARAMS = struct_CUDA_HOST_NODE_PARAMS_st
enum_CUgraphNodeType_enum = CEnum(ctypes.c_uint32)
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
enum_CUgraphInstantiateResult_enum = CEnum(ctypes.c_uint32)
CUDA_GRAPH_INSTANTIATE_SUCCESS = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_SUCCESS', 0)
CUDA_GRAPH_INSTANTIATE_ERROR = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_ERROR', 1)
CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE', 2)
CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED', 3)
CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = enum_CUgraphInstantiateResult_enum.define('CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED', 4)

CUgraphInstantiateResult = enum_CUgraphInstantiateResult_enum
class struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st(Struct): pass
struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st.SIZE = 32
struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st._fields_ = ['flags', 'hUploadStream', 'hErrNode_out', 'result_out']
setattr(struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st, 'flags', field(0, cuuint64_t))
setattr(struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st, 'hUploadStream', field(8, CUstream))
setattr(struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st, 'hErrNode_out', field(16, CUgraphNode))
setattr(struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st, 'result_out', field(24, CUgraphInstantiateResult))
CUDA_GRAPH_INSTANTIATE_PARAMS = struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st
enum_CUsynchronizationPolicy_enum = CEnum(ctypes.c_uint32)
CU_SYNC_POLICY_AUTO = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_AUTO', 1)
CU_SYNC_POLICY_SPIN = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_SPIN', 2)
CU_SYNC_POLICY_YIELD = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_YIELD', 3)
CU_SYNC_POLICY_BLOCKING_SYNC = enum_CUsynchronizationPolicy_enum.define('CU_SYNC_POLICY_BLOCKING_SYNC', 4)

CUsynchronizationPolicy = enum_CUsynchronizationPolicy_enum
enum_CUclusterSchedulingPolicy_enum = CEnum(ctypes.c_uint32)
CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_DEFAULT', 0)
CU_CLUSTER_SCHEDULING_POLICY_SPREAD = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_SPREAD', 1)
CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = enum_CUclusterSchedulingPolicy_enum.define('CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING', 2)

CUclusterSchedulingPolicy = enum_CUclusterSchedulingPolicy_enum
enum_CUlaunchMemSyncDomain_enum = CEnum(ctypes.c_uint32)
CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = enum_CUlaunchMemSyncDomain_enum.define('CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT', 0)
CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = enum_CUlaunchMemSyncDomain_enum.define('CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE', 1)

CUlaunchMemSyncDomain = enum_CUlaunchMemSyncDomain_enum
class struct_CUlaunchMemSyncDomainMap_st(Struct): pass
struct_CUlaunchMemSyncDomainMap_st.SIZE = 2
struct_CUlaunchMemSyncDomainMap_st._fields_ = ['default_', 'remote']
setattr(struct_CUlaunchMemSyncDomainMap_st, 'default_', field(0, ctypes.c_ubyte))
setattr(struct_CUlaunchMemSyncDomainMap_st, 'remote', field(1, ctypes.c_ubyte))
CUlaunchMemSyncDomainMap = struct_CUlaunchMemSyncDomainMap_st
enum_CUlaunchAttributeID_enum = CEnum(ctypes.c_uint32)
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
class union_CUlaunchAttributeValue_union(Union): pass
class _anonstruct0(Struct): pass
_anonstruct0.SIZE = 12
_anonstruct0._fields_ = ['x', 'y', 'z']
setattr(_anonstruct0, 'x', field(0, ctypes.c_uint32))
setattr(_anonstruct0, 'y', field(4, ctypes.c_uint32))
setattr(_anonstruct0, 'z', field(8, ctypes.c_uint32))
class _anonstruct1(Struct): pass
_anonstruct1.SIZE = 16
_anonstruct1._fields_ = ['event', 'flags', 'triggerAtBlockStart']
setattr(_anonstruct1, 'event', field(0, CUevent))
setattr(_anonstruct1, 'flags', field(8, ctypes.c_int32))
setattr(_anonstruct1, 'triggerAtBlockStart', field(12, ctypes.c_int32))
union_CUlaunchAttributeValue_union.SIZE = 64
union_CUlaunchAttributeValue_union._fields_ = ['pad', 'accessPolicyWindow', 'cooperative', 'syncPolicy', 'clusterDim', 'clusterSchedulingPolicyPreference', 'programmaticStreamSerializationAllowed', 'programmaticEvent', 'priority', 'memSyncDomainMap', 'memSyncDomain']
setattr(union_CUlaunchAttributeValue_union, 'pad', field(0, Array(ctypes.c_char, 64)))
setattr(union_CUlaunchAttributeValue_union, 'accessPolicyWindow', field(0, CUaccessPolicyWindow))
setattr(union_CUlaunchAttributeValue_union, 'cooperative', field(0, ctypes.c_int32))
setattr(union_CUlaunchAttributeValue_union, 'syncPolicy', field(0, CUsynchronizationPolicy))
setattr(union_CUlaunchAttributeValue_union, 'clusterDim', field(0, _anonstruct0))
setattr(union_CUlaunchAttributeValue_union, 'clusterSchedulingPolicyPreference', field(0, CUclusterSchedulingPolicy))
setattr(union_CUlaunchAttributeValue_union, 'programmaticStreamSerializationAllowed', field(0, ctypes.c_int32))
setattr(union_CUlaunchAttributeValue_union, 'programmaticEvent', field(0, _anonstruct1))
setattr(union_CUlaunchAttributeValue_union, 'priority', field(0, ctypes.c_int32))
setattr(union_CUlaunchAttributeValue_union, 'memSyncDomainMap', field(0, CUlaunchMemSyncDomainMap))
setattr(union_CUlaunchAttributeValue_union, 'memSyncDomain', field(0, CUlaunchMemSyncDomain))
CUlaunchAttributeValue = union_CUlaunchAttributeValue_union
class struct_CUlaunchAttribute_st(Struct): pass
struct_CUlaunchAttribute_st.SIZE = 72
struct_CUlaunchAttribute_st._fields_ = ['id', 'pad', 'value']
setattr(struct_CUlaunchAttribute_st, 'id', field(0, CUlaunchAttributeID))
setattr(struct_CUlaunchAttribute_st, 'pad', field(4, Array(ctypes.c_char, 4)))
setattr(struct_CUlaunchAttribute_st, 'value', field(8, CUlaunchAttributeValue))
CUlaunchAttribute = struct_CUlaunchAttribute_st
class struct_CUlaunchConfig_st(Struct): pass
struct_CUlaunchConfig_st.SIZE = 56
struct_CUlaunchConfig_st._fields_ = ['gridDimX', 'gridDimY', 'gridDimZ', 'blockDimX', 'blockDimY', 'blockDimZ', 'sharedMemBytes', 'hStream', 'attrs', 'numAttrs']
setattr(struct_CUlaunchConfig_st, 'gridDimX', field(0, ctypes.c_uint32))
setattr(struct_CUlaunchConfig_st, 'gridDimY', field(4, ctypes.c_uint32))
setattr(struct_CUlaunchConfig_st, 'gridDimZ', field(8, ctypes.c_uint32))
setattr(struct_CUlaunchConfig_st, 'blockDimX', field(12, ctypes.c_uint32))
setattr(struct_CUlaunchConfig_st, 'blockDimY', field(16, ctypes.c_uint32))
setattr(struct_CUlaunchConfig_st, 'blockDimZ', field(20, ctypes.c_uint32))
setattr(struct_CUlaunchConfig_st, 'sharedMemBytes', field(24, ctypes.c_uint32))
setattr(struct_CUlaunchConfig_st, 'hStream', field(32, CUstream))
setattr(struct_CUlaunchConfig_st, 'attrs', field(40, Pointer(CUlaunchAttribute)))
setattr(struct_CUlaunchConfig_st, 'numAttrs', field(48, ctypes.c_uint32))
CUlaunchConfig = struct_CUlaunchConfig_st
CUkernelNodeAttrID = enum_CUlaunchAttributeID_enum
CUkernelNodeAttrValue_v1 = union_CUlaunchAttributeValue_union
CUkernelNodeAttrValue = union_CUlaunchAttributeValue_union
enum_CUstreamCaptureStatus_enum = CEnum(ctypes.c_uint32)
CU_STREAM_CAPTURE_STATUS_NONE = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_NONE', 0)
CU_STREAM_CAPTURE_STATUS_ACTIVE = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_ACTIVE', 1)
CU_STREAM_CAPTURE_STATUS_INVALIDATED = enum_CUstreamCaptureStatus_enum.define('CU_STREAM_CAPTURE_STATUS_INVALIDATED', 2)

CUstreamCaptureStatus = enum_CUstreamCaptureStatus_enum
enum_CUstreamCaptureMode_enum = CEnum(ctypes.c_uint32)
CU_STREAM_CAPTURE_MODE_GLOBAL = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_GLOBAL', 0)
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_THREAD_LOCAL', 1)
CU_STREAM_CAPTURE_MODE_RELAXED = enum_CUstreamCaptureMode_enum.define('CU_STREAM_CAPTURE_MODE_RELAXED', 2)

CUstreamCaptureMode = enum_CUstreamCaptureMode_enum
CUstreamAttrID = enum_CUlaunchAttributeID_enum
CUstreamAttrValue_v1 = union_CUlaunchAttributeValue_union
CUstreamAttrValue = union_CUlaunchAttributeValue_union
enum_CUdriverProcAddress_flags_enum = CEnum(ctypes.c_uint32)
CU_GET_PROC_ADDRESS_DEFAULT = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_DEFAULT', 0)
CU_GET_PROC_ADDRESS_LEGACY_STREAM = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_LEGACY_STREAM', 1)
CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = enum_CUdriverProcAddress_flags_enum.define('CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM', 2)

CUdriverProcAddress_flags = enum_CUdriverProcAddress_flags_enum
enum_CUdriverProcAddressQueryResult_enum = CEnum(ctypes.c_uint32)
CU_GET_PROC_ADDRESS_SUCCESS = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_SUCCESS', 0)
CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND', 1)
CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = enum_CUdriverProcAddressQueryResult_enum.define('CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT', 2)

CUdriverProcAddressQueryResult = enum_CUdriverProcAddressQueryResult_enum
enum_CUexecAffinityType_enum = CEnum(ctypes.c_uint32)
CU_EXEC_AFFINITY_TYPE_SM_COUNT = enum_CUexecAffinityType_enum.define('CU_EXEC_AFFINITY_TYPE_SM_COUNT', 0)
CU_EXEC_AFFINITY_TYPE_MAX = enum_CUexecAffinityType_enum.define('CU_EXEC_AFFINITY_TYPE_MAX', 1)

CUexecAffinityType = enum_CUexecAffinityType_enum
class struct_CUexecAffinitySmCount_st(Struct): pass
struct_CUexecAffinitySmCount_st.SIZE = 4
struct_CUexecAffinitySmCount_st._fields_ = ['val']
setattr(struct_CUexecAffinitySmCount_st, 'val', field(0, ctypes.c_uint32))
CUexecAffinitySmCount_v1 = struct_CUexecAffinitySmCount_st
CUexecAffinitySmCount = struct_CUexecAffinitySmCount_st
class struct_CUexecAffinityParam_st(Struct): pass
class _anonunion2(Union): pass
_anonunion2.SIZE = 4
_anonunion2._fields_ = ['smCount']
setattr(_anonunion2, 'smCount', field(0, CUexecAffinitySmCount))
struct_CUexecAffinityParam_st.SIZE = 8
struct_CUexecAffinityParam_st._fields_ = ['type', 'param']
setattr(struct_CUexecAffinityParam_st, 'type', field(0, CUexecAffinityType))
setattr(struct_CUexecAffinityParam_st, 'param', field(4, _anonunion2))
CUexecAffinityParam_v1 = struct_CUexecAffinityParam_st
CUexecAffinityParam = struct_CUexecAffinityParam_st
enum_CUlibraryOption_enum = CEnum(ctypes.c_uint32)
CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = enum_CUlibraryOption_enum.define('CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE', 0)
CU_LIBRARY_BINARY_IS_PRESERVED = enum_CUlibraryOption_enum.define('CU_LIBRARY_BINARY_IS_PRESERVED', 1)
CU_LIBRARY_NUM_OPTIONS = enum_CUlibraryOption_enum.define('CU_LIBRARY_NUM_OPTIONS', 2)

CUlibraryOption = enum_CUlibraryOption_enum
class struct_CUlibraryHostUniversalFunctionAndDataTable_st(Struct): pass
struct_CUlibraryHostUniversalFunctionAndDataTable_st.SIZE = 32
struct_CUlibraryHostUniversalFunctionAndDataTable_st._fields_ = ['functionTable', 'functionWindowSize', 'dataTable', 'dataWindowSize']
setattr(struct_CUlibraryHostUniversalFunctionAndDataTable_st, 'functionTable', field(0, ctypes.c_void_p))
setattr(struct_CUlibraryHostUniversalFunctionAndDataTable_st, 'functionWindowSize', field(8, size_t))
setattr(struct_CUlibraryHostUniversalFunctionAndDataTable_st, 'dataTable', field(16, ctypes.c_void_p))
setattr(struct_CUlibraryHostUniversalFunctionAndDataTable_st, 'dataWindowSize', field(24, size_t))
CUlibraryHostUniversalFunctionAndDataTable = struct_CUlibraryHostUniversalFunctionAndDataTable_st
enum_cudaError_enum = CEnum(ctypes.c_uint32)
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
enum_CUdevice_P2PAttribute_enum = CEnum(ctypes.c_uint32)
CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK', 1)
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED', 2)
CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED', 3)
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED', 4)
CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = enum_CUdevice_P2PAttribute_enum.define('CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED', 4)

CUdevice_P2PAttribute = enum_CUdevice_P2PAttribute_enum
CUstreamCallback = ctypes.CFUNCTYPE(None, Pointer(struct_CUstream_st), enum_cudaError_enum, ctypes.c_void_p)
CUoccupancyB2DSize = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_int32)
class struct_CUDA_MEMCPY2D_st(Struct): pass
struct_CUDA_MEMCPY2D_st.SIZE = 128
struct_CUDA_MEMCPY2D_st._fields_ = ['srcXInBytes', 'srcY', 'srcMemoryType', 'srcHost', 'srcDevice', 'srcArray', 'srcPitch', 'dstXInBytes', 'dstY', 'dstMemoryType', 'dstHost', 'dstDevice', 'dstArray', 'dstPitch', 'WidthInBytes', 'Height']
setattr(struct_CUDA_MEMCPY2D_st, 'srcXInBytes', field(0, size_t))
setattr(struct_CUDA_MEMCPY2D_st, 'srcY', field(8, size_t))
setattr(struct_CUDA_MEMCPY2D_st, 'srcMemoryType', field(16, CUmemorytype))
setattr(struct_CUDA_MEMCPY2D_st, 'srcHost', field(24, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY2D_st, 'srcDevice', field(32, CUdeviceptr))
setattr(struct_CUDA_MEMCPY2D_st, 'srcArray', field(40, CUarray))
setattr(struct_CUDA_MEMCPY2D_st, 'srcPitch', field(48, size_t))
setattr(struct_CUDA_MEMCPY2D_st, 'dstXInBytes', field(56, size_t))
setattr(struct_CUDA_MEMCPY2D_st, 'dstY', field(64, size_t))
setattr(struct_CUDA_MEMCPY2D_st, 'dstMemoryType', field(72, CUmemorytype))
setattr(struct_CUDA_MEMCPY2D_st, 'dstHost', field(80, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY2D_st, 'dstDevice', field(88, CUdeviceptr))
setattr(struct_CUDA_MEMCPY2D_st, 'dstArray', field(96, CUarray))
setattr(struct_CUDA_MEMCPY2D_st, 'dstPitch', field(104, size_t))
setattr(struct_CUDA_MEMCPY2D_st, 'WidthInBytes', field(112, size_t))
setattr(struct_CUDA_MEMCPY2D_st, 'Height', field(120, size_t))
CUDA_MEMCPY2D_v2 = struct_CUDA_MEMCPY2D_st
CUDA_MEMCPY2D = struct_CUDA_MEMCPY2D_st
class struct_CUDA_MEMCPY3D_st(Struct): pass
struct_CUDA_MEMCPY3D_st.SIZE = 200
struct_CUDA_MEMCPY3D_st._fields_ = ['srcXInBytes', 'srcY', 'srcZ', 'srcLOD', 'srcMemoryType', 'srcHost', 'srcDevice', 'srcArray', 'reserved0', 'srcPitch', 'srcHeight', 'dstXInBytes', 'dstY', 'dstZ', 'dstLOD', 'dstMemoryType', 'dstHost', 'dstDevice', 'dstArray', 'reserved1', 'dstPitch', 'dstHeight', 'WidthInBytes', 'Height', 'Depth']
setattr(struct_CUDA_MEMCPY3D_st, 'srcXInBytes', field(0, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'srcY', field(8, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'srcZ', field(16, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'srcLOD', field(24, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'srcMemoryType', field(32, CUmemorytype))
setattr(struct_CUDA_MEMCPY3D_st, 'srcHost', field(40, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_st, 'srcDevice', field(48, CUdeviceptr))
setattr(struct_CUDA_MEMCPY3D_st, 'srcArray', field(56, CUarray))
setattr(struct_CUDA_MEMCPY3D_st, 'reserved0', field(64, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_st, 'srcPitch', field(72, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'srcHeight', field(80, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'dstXInBytes', field(88, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'dstY', field(96, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'dstZ', field(104, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'dstLOD', field(112, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'dstMemoryType', field(120, CUmemorytype))
setattr(struct_CUDA_MEMCPY3D_st, 'dstHost', field(128, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_st, 'dstDevice', field(136, CUdeviceptr))
setattr(struct_CUDA_MEMCPY3D_st, 'dstArray', field(144, CUarray))
setattr(struct_CUDA_MEMCPY3D_st, 'reserved1', field(152, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_st, 'dstPitch', field(160, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'dstHeight', field(168, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'WidthInBytes', field(176, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'Height', field(184, size_t))
setattr(struct_CUDA_MEMCPY3D_st, 'Depth', field(192, size_t))
CUDA_MEMCPY3D_v2 = struct_CUDA_MEMCPY3D_st
CUDA_MEMCPY3D = struct_CUDA_MEMCPY3D_st
class struct_CUDA_MEMCPY3D_PEER_st(Struct): pass
struct_CUDA_MEMCPY3D_PEER_st.SIZE = 200
struct_CUDA_MEMCPY3D_PEER_st._fields_ = ['srcXInBytes', 'srcY', 'srcZ', 'srcLOD', 'srcMemoryType', 'srcHost', 'srcDevice', 'srcArray', 'srcContext', 'srcPitch', 'srcHeight', 'dstXInBytes', 'dstY', 'dstZ', 'dstLOD', 'dstMemoryType', 'dstHost', 'dstDevice', 'dstArray', 'dstContext', 'dstPitch', 'dstHeight', 'WidthInBytes', 'Height', 'Depth']
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcXInBytes', field(0, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcY', field(8, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcZ', field(16, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcLOD', field(24, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcMemoryType', field(32, CUmemorytype))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcHost', field(40, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcDevice', field(48, CUdeviceptr))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcArray', field(56, CUarray))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcContext', field(64, CUcontext))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcPitch', field(72, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'srcHeight', field(80, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstXInBytes', field(88, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstY', field(96, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstZ', field(104, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstLOD', field(112, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstMemoryType', field(120, CUmemorytype))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstHost', field(128, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstDevice', field(136, CUdeviceptr))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstArray', field(144, CUarray))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstContext', field(152, CUcontext))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstPitch', field(160, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'dstHeight', field(168, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'WidthInBytes', field(176, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'Height', field(184, size_t))
setattr(struct_CUDA_MEMCPY3D_PEER_st, 'Depth', field(192, size_t))
CUDA_MEMCPY3D_PEER_v1 = struct_CUDA_MEMCPY3D_PEER_st
CUDA_MEMCPY3D_PEER = struct_CUDA_MEMCPY3D_PEER_st
class struct_CUDA_ARRAY_DESCRIPTOR_st(Struct): pass
struct_CUDA_ARRAY_DESCRIPTOR_st.SIZE = 24
struct_CUDA_ARRAY_DESCRIPTOR_st._fields_ = ['Width', 'Height', 'Format', 'NumChannels']
setattr(struct_CUDA_ARRAY_DESCRIPTOR_st, 'Width', field(0, size_t))
setattr(struct_CUDA_ARRAY_DESCRIPTOR_st, 'Height', field(8, size_t))
setattr(struct_CUDA_ARRAY_DESCRIPTOR_st, 'Format', field(16, CUarray_format))
setattr(struct_CUDA_ARRAY_DESCRIPTOR_st, 'NumChannels', field(20, ctypes.c_uint32))
CUDA_ARRAY_DESCRIPTOR_v2 = struct_CUDA_ARRAY_DESCRIPTOR_st
CUDA_ARRAY_DESCRIPTOR = struct_CUDA_ARRAY_DESCRIPTOR_st
class struct_CUDA_ARRAY3D_DESCRIPTOR_st(Struct): pass
struct_CUDA_ARRAY3D_DESCRIPTOR_st.SIZE = 40
struct_CUDA_ARRAY3D_DESCRIPTOR_st._fields_ = ['Width', 'Height', 'Depth', 'Format', 'NumChannels', 'Flags']
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_st, 'Width', field(0, size_t))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_st, 'Height', field(8, size_t))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_st, 'Depth', field(16, size_t))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_st, 'Format', field(24, CUarray_format))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_st, 'NumChannels', field(28, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_st, 'Flags', field(32, ctypes.c_uint32))
CUDA_ARRAY3D_DESCRIPTOR_v2 = struct_CUDA_ARRAY3D_DESCRIPTOR_st
CUDA_ARRAY3D_DESCRIPTOR = struct_CUDA_ARRAY3D_DESCRIPTOR_st
class struct_CUDA_ARRAY_SPARSE_PROPERTIES_st(Struct): pass
class _anonstruct3(Struct): pass
_anonstruct3.SIZE = 12
_anonstruct3._fields_ = ['width', 'height', 'depth']
setattr(_anonstruct3, 'width', field(0, ctypes.c_uint32))
setattr(_anonstruct3, 'height', field(4, ctypes.c_uint32))
setattr(_anonstruct3, 'depth', field(8, ctypes.c_uint32))
struct_CUDA_ARRAY_SPARSE_PROPERTIES_st.SIZE = 48
struct_CUDA_ARRAY_SPARSE_PROPERTIES_st._fields_ = ['tileExtent', 'miptailFirstLevel', 'miptailSize', 'flags', 'reserved']
setattr(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st, 'tileExtent', field(0, _anonstruct3))
setattr(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st, 'miptailFirstLevel', field(12, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st, 'miptailSize', field(16, ctypes.c_uint64))
setattr(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st, 'flags', field(24, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st, 'reserved', field(28, Array(ctypes.c_uint32, 4)))
CUDA_ARRAY_SPARSE_PROPERTIES_v1 = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
CUDA_ARRAY_SPARSE_PROPERTIES = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
class struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st(Struct): pass
struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st.SIZE = 32
struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st._fields_ = ['size', 'alignment', 'reserved']
setattr(struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st, 'size', field(0, size_t))
setattr(struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st, 'alignment', field(8, size_t))
setattr(struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st, 'reserved', field(16, Array(ctypes.c_uint32, 4)))
CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 = struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st
CUDA_ARRAY_MEMORY_REQUIREMENTS = struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st
class struct_CUDA_RESOURCE_DESC_st(Struct): pass
class _anonunion4(Union): pass
class _anonstruct5(Struct): pass
_anonstruct5.SIZE = 8
_anonstruct5._fields_ = ['hArray']
setattr(_anonstruct5, 'hArray', field(0, CUarray))
class _anonstruct6(Struct): pass
_anonstruct6.SIZE = 8
_anonstruct6._fields_ = ['hMipmappedArray']
setattr(_anonstruct6, 'hMipmappedArray', field(0, CUmipmappedArray))
class _anonstruct7(Struct): pass
_anonstruct7.SIZE = 24
_anonstruct7._fields_ = ['devPtr', 'format', 'numChannels', 'sizeInBytes']
setattr(_anonstruct7, 'devPtr', field(0, CUdeviceptr))
setattr(_anonstruct7, 'format', field(8, CUarray_format))
setattr(_anonstruct7, 'numChannels', field(12, ctypes.c_uint32))
setattr(_anonstruct7, 'sizeInBytes', field(16, size_t))
class _anonstruct8(Struct): pass
_anonstruct8.SIZE = 40
_anonstruct8._fields_ = ['devPtr', 'format', 'numChannels', 'width', 'height', 'pitchInBytes']
setattr(_anonstruct8, 'devPtr', field(0, CUdeviceptr))
setattr(_anonstruct8, 'format', field(8, CUarray_format))
setattr(_anonstruct8, 'numChannels', field(12, ctypes.c_uint32))
setattr(_anonstruct8, 'width', field(16, size_t))
setattr(_anonstruct8, 'height', field(24, size_t))
setattr(_anonstruct8, 'pitchInBytes', field(32, size_t))
class _anonstruct9(Struct): pass
_anonstruct9.SIZE = 128
_anonstruct9._fields_ = ['reserved']
setattr(_anonstruct9, 'reserved', field(0, Array(ctypes.c_int32, 32)))
_anonunion4.SIZE = 128
_anonunion4._fields_ = ['array', 'mipmap', 'linear', 'pitch2D', 'reserved']
setattr(_anonunion4, 'array', field(0, _anonstruct5))
setattr(_anonunion4, 'mipmap', field(0, _anonstruct6))
setattr(_anonunion4, 'linear', field(0, _anonstruct7))
setattr(_anonunion4, 'pitch2D', field(0, _anonstruct8))
setattr(_anonunion4, 'reserved', field(0, _anonstruct9))
struct_CUDA_RESOURCE_DESC_st.SIZE = 144
struct_CUDA_RESOURCE_DESC_st._fields_ = ['resType', 'res', 'flags']
setattr(struct_CUDA_RESOURCE_DESC_st, 'resType', field(0, CUresourcetype))
setattr(struct_CUDA_RESOURCE_DESC_st, 'res', field(8, _anonunion4))
setattr(struct_CUDA_RESOURCE_DESC_st, 'flags', field(136, ctypes.c_uint32))
CUDA_RESOURCE_DESC_v1 = struct_CUDA_RESOURCE_DESC_st
CUDA_RESOURCE_DESC = struct_CUDA_RESOURCE_DESC_st
class struct_CUDA_TEXTURE_DESC_st(Struct): pass
struct_CUDA_TEXTURE_DESC_st.SIZE = 104
struct_CUDA_TEXTURE_DESC_st._fields_ = ['addressMode', 'filterMode', 'flags', 'maxAnisotropy', 'mipmapFilterMode', 'mipmapLevelBias', 'minMipmapLevelClamp', 'maxMipmapLevelClamp', 'borderColor', 'reserved']
setattr(struct_CUDA_TEXTURE_DESC_st, 'addressMode', field(0, Array(CUaddress_mode, 3)))
setattr(struct_CUDA_TEXTURE_DESC_st, 'filterMode', field(12, CUfilter_mode))
setattr(struct_CUDA_TEXTURE_DESC_st, 'flags', field(16, ctypes.c_uint32))
setattr(struct_CUDA_TEXTURE_DESC_st, 'maxAnisotropy', field(20, ctypes.c_uint32))
setattr(struct_CUDA_TEXTURE_DESC_st, 'mipmapFilterMode', field(24, CUfilter_mode))
setattr(struct_CUDA_TEXTURE_DESC_st, 'mipmapLevelBias', field(28, ctypes.c_float))
setattr(struct_CUDA_TEXTURE_DESC_st, 'minMipmapLevelClamp', field(32, ctypes.c_float))
setattr(struct_CUDA_TEXTURE_DESC_st, 'maxMipmapLevelClamp', field(36, ctypes.c_float))
setattr(struct_CUDA_TEXTURE_DESC_st, 'borderColor', field(40, Array(ctypes.c_float, 4)))
setattr(struct_CUDA_TEXTURE_DESC_st, 'reserved', field(56, Array(ctypes.c_int32, 12)))
CUDA_TEXTURE_DESC_v1 = struct_CUDA_TEXTURE_DESC_st
CUDA_TEXTURE_DESC = struct_CUDA_TEXTURE_DESC_st
enum_CUresourceViewFormat_enum = CEnum(ctypes.c_uint32)
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
class struct_CUDA_RESOURCE_VIEW_DESC_st(Struct): pass
struct_CUDA_RESOURCE_VIEW_DESC_st.SIZE = 112
struct_CUDA_RESOURCE_VIEW_DESC_st._fields_ = ['format', 'width', 'height', 'depth', 'firstMipmapLevel', 'lastMipmapLevel', 'firstLayer', 'lastLayer', 'reserved']
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'format', field(0, CUresourceViewFormat))
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'width', field(8, size_t))
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'height', field(16, size_t))
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'depth', field(24, size_t))
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'firstMipmapLevel', field(32, ctypes.c_uint32))
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'lastMipmapLevel', field(36, ctypes.c_uint32))
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'firstLayer', field(40, ctypes.c_uint32))
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'lastLayer', field(44, ctypes.c_uint32))
setattr(struct_CUDA_RESOURCE_VIEW_DESC_st, 'reserved', field(48, Array(ctypes.c_uint32, 16)))
CUDA_RESOURCE_VIEW_DESC_v1 = struct_CUDA_RESOURCE_VIEW_DESC_st
CUDA_RESOURCE_VIEW_DESC = struct_CUDA_RESOURCE_VIEW_DESC_st
class struct_CUtensorMap_st(Struct): pass
struct_CUtensorMap_st.SIZE = 128
struct_CUtensorMap_st._fields_ = ['opaque']
setattr(struct_CUtensorMap_st, 'opaque', field(0, Array(cuuint64_t, 16)))
CUtensorMap = struct_CUtensorMap_st
enum_CUtensorMapDataType_enum = CEnum(ctypes.c_uint32)
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
enum_CUtensorMapInterleave_enum = CEnum(ctypes.c_uint32)
CU_TENSOR_MAP_INTERLEAVE_NONE = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_NONE', 0)
CU_TENSOR_MAP_INTERLEAVE_16B = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_16B', 1)
CU_TENSOR_MAP_INTERLEAVE_32B = enum_CUtensorMapInterleave_enum.define('CU_TENSOR_MAP_INTERLEAVE_32B', 2)

CUtensorMapInterleave = enum_CUtensorMapInterleave_enum
enum_CUtensorMapSwizzle_enum = CEnum(ctypes.c_uint32)
CU_TENSOR_MAP_SWIZZLE_NONE = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_NONE', 0)
CU_TENSOR_MAP_SWIZZLE_32B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_32B', 1)
CU_TENSOR_MAP_SWIZZLE_64B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_64B', 2)
CU_TENSOR_MAP_SWIZZLE_128B = enum_CUtensorMapSwizzle_enum.define('CU_TENSOR_MAP_SWIZZLE_128B', 3)

CUtensorMapSwizzle = enum_CUtensorMapSwizzle_enum
enum_CUtensorMapL2promotion_enum = CEnum(ctypes.c_uint32)
CU_TENSOR_MAP_L2_PROMOTION_NONE = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_NONE', 0)
CU_TENSOR_MAP_L2_PROMOTION_L2_64B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_64B', 1)
CU_TENSOR_MAP_L2_PROMOTION_L2_128B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_128B', 2)
CU_TENSOR_MAP_L2_PROMOTION_L2_256B = enum_CUtensorMapL2promotion_enum.define('CU_TENSOR_MAP_L2_PROMOTION_L2_256B', 3)

CUtensorMapL2promotion = enum_CUtensorMapL2promotion_enum
enum_CUtensorMapFloatOOBfill_enum = CEnum(ctypes.c_uint32)
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = enum_CUtensorMapFloatOOBfill_enum.define('CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE', 0)
CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA = enum_CUtensorMapFloatOOBfill_enum.define('CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA', 1)

CUtensorMapFloatOOBfill = enum_CUtensorMapFloatOOBfill_enum
class struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st(Struct): pass
struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st.SIZE = 16
struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st._fields_ = ['p2pToken', 'vaSpaceToken']
setattr(struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st, 'p2pToken', field(0, ctypes.c_uint64))
setattr(struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st, 'vaSpaceToken', field(8, ctypes.c_uint32))
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum = CEnum(ctypes.c_uint32)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE', 0)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ', 1)
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum.define('CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE', 3)

CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum
class struct_CUDA_LAUNCH_PARAMS_st(Struct): pass
struct_CUDA_LAUNCH_PARAMS_st.SIZE = 56
struct_CUDA_LAUNCH_PARAMS_st._fields_ = ['function', 'gridDimX', 'gridDimY', 'gridDimZ', 'blockDimX', 'blockDimY', 'blockDimZ', 'sharedMemBytes', 'hStream', 'kernelParams']
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'function', field(0, CUfunction))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'gridDimX', field(8, ctypes.c_uint32))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'gridDimY', field(12, ctypes.c_uint32))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'gridDimZ', field(16, ctypes.c_uint32))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'blockDimX', field(20, ctypes.c_uint32))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'blockDimY', field(24, ctypes.c_uint32))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'blockDimZ', field(28, ctypes.c_uint32))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'sharedMemBytes', field(32, ctypes.c_uint32))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'hStream', field(40, CUstream))
setattr(struct_CUDA_LAUNCH_PARAMS_st, 'kernelParams', field(48, Pointer(ctypes.c_void_p)))
CUDA_LAUNCH_PARAMS_v1 = struct_CUDA_LAUNCH_PARAMS_st
CUDA_LAUNCH_PARAMS = struct_CUDA_LAUNCH_PARAMS_st
enum_CUexternalMemoryHandleType_enum = CEnum(ctypes.c_uint32)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD', 1)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32', 2)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT', 3)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP', 4)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE', 5)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE', 6)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT', 7)
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = enum_CUexternalMemoryHandleType_enum.define('CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF', 8)

CUexternalMemoryHandleType = enum_CUexternalMemoryHandleType_enum
class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(Struct): pass
class _anonunion10(Union): pass
class _anonstruct11(Struct): pass
_anonstruct11.SIZE = 16
_anonstruct11._fields_ = ['handle', 'name']
setattr(_anonstruct11, 'handle', field(0, ctypes.c_void_p))
setattr(_anonstruct11, 'name', field(8, ctypes.c_void_p))
_anonunion10.SIZE = 16
_anonunion10._fields_ = ['fd', 'win32', 'nvSciBufObject']
setattr(_anonunion10, 'fd', field(0, ctypes.c_int32))
setattr(_anonunion10, 'win32', field(0, _anonstruct11))
setattr(_anonunion10, 'nvSciBufObject', field(0, ctypes.c_void_p))
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st.SIZE = 104
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st._fields_ = ['type', 'handle', 'size', 'flags', 'reserved']
setattr(struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st, 'type', field(0, CUexternalMemoryHandleType))
setattr(struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st, 'handle', field(8, _anonunion10))
setattr(struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st, 'size', field(24, ctypes.c_uint64))
setattr(struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st, 'flags', field(32, ctypes.c_uint32))
setattr(struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st, 'reserved', field(36, Array(ctypes.c_uint32, 16)))
CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
CUDA_EXTERNAL_MEMORY_HANDLE_DESC = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
class struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(Struct): pass
struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st.SIZE = 88
struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st._fields_ = ['offset', 'size', 'flags', 'reserved']
setattr(struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st, 'offset', field(0, ctypes.c_uint64))
setattr(struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st, 'size', field(8, ctypes.c_uint64))
setattr(struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st, 'flags', field(16, ctypes.c_uint32))
setattr(struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st, 'reserved', field(20, Array(ctypes.c_uint32, 16)))
CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
CUDA_EXTERNAL_MEMORY_BUFFER_DESC = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
class struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st(Struct): pass
struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st.SIZE = 120
struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st._fields_ = ['offset', 'arrayDesc', 'numLevels', 'reserved']
setattr(struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st, 'offset', field(0, ctypes.c_uint64))
setattr(struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st, 'arrayDesc', field(8, CUDA_ARRAY3D_DESCRIPTOR))
setattr(struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st, 'numLevels', field(48, ctypes.c_uint32))
setattr(struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st, 'reserved', field(52, Array(ctypes.c_uint32, 16)))
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
enum_CUexternalSemaphoreHandleType_enum = CEnum(ctypes.c_uint32)
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
class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(Struct): pass
class _anonunion12(Union): pass
class _anonstruct13(Struct): pass
_anonstruct13.SIZE = 16
_anonstruct13._fields_ = ['handle', 'name']
setattr(_anonstruct13, 'handle', field(0, ctypes.c_void_p))
setattr(_anonstruct13, 'name', field(8, ctypes.c_void_p))
_anonunion12.SIZE = 16
_anonunion12._fields_ = ['fd', 'win32', 'nvSciSyncObj']
setattr(_anonunion12, 'fd', field(0, ctypes.c_int32))
setattr(_anonunion12, 'win32', field(0, _anonstruct13))
setattr(_anonunion12, 'nvSciSyncObj', field(0, ctypes.c_void_p))
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st.SIZE = 96
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st._fields_ = ['type', 'handle', 'flags', 'reserved']
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st, 'type', field(0, CUexternalSemaphoreHandleType))
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st, 'handle', field(8, _anonunion12))
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st, 'flags', field(24, ctypes.c_uint32))
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st, 'reserved', field(28, Array(ctypes.c_uint32, 16)))
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(Struct): pass
class _anonstruct14(Struct): pass
class _anonstruct15(Struct): pass
_anonstruct15.SIZE = 8
_anonstruct15._fields_ = ['value']
setattr(_anonstruct15, 'value', field(0, ctypes.c_uint64))
class _anonunion16(Union): pass
_anonunion16.SIZE = 8
_anonunion16._fields_ = ['fence', 'reserved']
setattr(_anonunion16, 'fence', field(0, ctypes.c_void_p))
setattr(_anonunion16, 'reserved', field(0, ctypes.c_uint64))
class _anonstruct17(Struct): pass
_anonstruct17.SIZE = 8
_anonstruct17._fields_ = ['key']
setattr(_anonstruct17, 'key', field(0, ctypes.c_uint64))
_anonstruct14.SIZE = 72
_anonstruct14._fields_ = ['fence', 'nvSciSync', 'keyedMutex', 'reserved']
setattr(_anonstruct14, 'fence', field(0, _anonstruct15))
setattr(_anonstruct14, 'nvSciSync', field(8, _anonunion16))
setattr(_anonstruct14, 'keyedMutex', field(16, _anonstruct17))
setattr(_anonstruct14, 'reserved', field(24, Array(ctypes.c_uint32, 12)))
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st.SIZE = 144
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st._fields_ = ['params', 'flags', 'reserved']
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st, 'params', field(0, _anonstruct14))
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st, 'flags', field(72, ctypes.c_uint32))
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st, 'reserved', field(76, Array(ctypes.c_uint32, 16)))
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(Struct): pass
class _anonstruct18(Struct): pass
class _anonstruct19(Struct): pass
_anonstruct19.SIZE = 8
_anonstruct19._fields_ = ['value']
setattr(_anonstruct19, 'value', field(0, ctypes.c_uint64))
class _anonunion20(Union): pass
_anonunion20.SIZE = 8
_anonunion20._fields_ = ['fence', 'reserved']
setattr(_anonunion20, 'fence', field(0, ctypes.c_void_p))
setattr(_anonunion20, 'reserved', field(0, ctypes.c_uint64))
class _anonstruct21(Struct): pass
_anonstruct21.SIZE = 16
_anonstruct21._fields_ = ['key', 'timeoutMs']
setattr(_anonstruct21, 'key', field(0, ctypes.c_uint64))
setattr(_anonstruct21, 'timeoutMs', field(8, ctypes.c_uint32))
_anonstruct18.SIZE = 72
_anonstruct18._fields_ = ['fence', 'nvSciSync', 'keyedMutex', 'reserved']
setattr(_anonstruct18, 'fence', field(0, _anonstruct19))
setattr(_anonstruct18, 'nvSciSync', field(8, _anonunion20))
setattr(_anonstruct18, 'keyedMutex', field(16, _anonstruct21))
setattr(_anonstruct18, 'reserved', field(32, Array(ctypes.c_uint32, 10)))
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st.SIZE = 144
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st._fields_ = ['params', 'flags', 'reserved']
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st, 'params', field(0, _anonstruct18))
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st, 'flags', field(72, ctypes.c_uint32))
setattr(struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st, 'reserved', field(76, Array(ctypes.c_uint32, 16)))
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
class struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st(Struct): pass
struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st.SIZE = 24
struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st._fields_ = ['extSemArray', 'paramsArray', 'numExtSems']
setattr(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st, 'extSemArray', field(0, Pointer(CUexternalSemaphore)))
setattr(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st, 'paramsArray', field(8, Pointer(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS)))
setattr(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st, 'numExtSems', field(16, ctypes.c_uint32))
CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
class struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st(Struct): pass
struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st.SIZE = 24
struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st._fields_ = ['extSemArray', 'paramsArray', 'numExtSems']
setattr(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st, 'extSemArray', field(0, Pointer(CUexternalSemaphore)))
setattr(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st, 'paramsArray', field(8, Pointer(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS)))
setattr(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st, 'numExtSems', field(16, ctypes.c_uint32))
CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUDA_EXT_SEM_WAIT_NODE_PARAMS = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUmemGenericAllocationHandle_v1 = ctypes.c_uint64
CUmemGenericAllocationHandle = ctypes.c_uint64
enum_CUmemAllocationHandleType_enum = CEnum(ctypes.c_uint32)
CU_MEM_HANDLE_TYPE_NONE = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_NONE', 0)
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR', 1)
CU_MEM_HANDLE_TYPE_WIN32 = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_WIN32', 2)
CU_MEM_HANDLE_TYPE_WIN32_KMT = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_WIN32_KMT', 4)
CU_MEM_HANDLE_TYPE_MAX = enum_CUmemAllocationHandleType_enum.define('CU_MEM_HANDLE_TYPE_MAX', 2147483647)

CUmemAllocationHandleType = enum_CUmemAllocationHandleType_enum
enum_CUmemAccess_flags_enum = CEnum(ctypes.c_uint32)
CU_MEM_ACCESS_FLAGS_PROT_NONE = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_NONE', 0)
CU_MEM_ACCESS_FLAGS_PROT_READ = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_READ', 1)
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_READWRITE', 3)
CU_MEM_ACCESS_FLAGS_PROT_MAX = enum_CUmemAccess_flags_enum.define('CU_MEM_ACCESS_FLAGS_PROT_MAX', 2147483647)

CUmemAccess_flags = enum_CUmemAccess_flags_enum
enum_CUmemLocationType_enum = CEnum(ctypes.c_uint32)
CU_MEM_LOCATION_TYPE_INVALID = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_INVALID', 0)
CU_MEM_LOCATION_TYPE_DEVICE = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_DEVICE', 1)
CU_MEM_LOCATION_TYPE_MAX = enum_CUmemLocationType_enum.define('CU_MEM_LOCATION_TYPE_MAX', 2147483647)

CUmemLocationType = enum_CUmemLocationType_enum
enum_CUmemAllocationType_enum = CEnum(ctypes.c_uint32)
CU_MEM_ALLOCATION_TYPE_INVALID = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_INVALID', 0)
CU_MEM_ALLOCATION_TYPE_PINNED = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_PINNED', 1)
CU_MEM_ALLOCATION_TYPE_MAX = enum_CUmemAllocationType_enum.define('CU_MEM_ALLOCATION_TYPE_MAX', 2147483647)

CUmemAllocationType = enum_CUmemAllocationType_enum
enum_CUmemAllocationGranularity_flags_enum = CEnum(ctypes.c_uint32)
CU_MEM_ALLOC_GRANULARITY_MINIMUM = enum_CUmemAllocationGranularity_flags_enum.define('CU_MEM_ALLOC_GRANULARITY_MINIMUM', 0)
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = enum_CUmemAllocationGranularity_flags_enum.define('CU_MEM_ALLOC_GRANULARITY_RECOMMENDED', 1)

CUmemAllocationGranularity_flags = enum_CUmemAllocationGranularity_flags_enum
enum_CUmemRangeHandleType_enum = CEnum(ctypes.c_uint32)
CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = enum_CUmemRangeHandleType_enum.define('CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD', 1)
CU_MEM_RANGE_HANDLE_TYPE_MAX = enum_CUmemRangeHandleType_enum.define('CU_MEM_RANGE_HANDLE_TYPE_MAX', 2147483647)

CUmemRangeHandleType = enum_CUmemRangeHandleType_enum
enum_CUarraySparseSubresourceType_enum = CEnum(ctypes.c_uint32)
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = enum_CUarraySparseSubresourceType_enum.define('CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL', 0)
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = enum_CUarraySparseSubresourceType_enum.define('CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL', 1)

CUarraySparseSubresourceType = enum_CUarraySparseSubresourceType_enum
enum_CUmemOperationType_enum = CEnum(ctypes.c_uint32)
CU_MEM_OPERATION_TYPE_MAP = enum_CUmemOperationType_enum.define('CU_MEM_OPERATION_TYPE_MAP', 1)
CU_MEM_OPERATION_TYPE_UNMAP = enum_CUmemOperationType_enum.define('CU_MEM_OPERATION_TYPE_UNMAP', 2)

CUmemOperationType = enum_CUmemOperationType_enum
enum_CUmemHandleType_enum = CEnum(ctypes.c_uint32)
CU_MEM_HANDLE_TYPE_GENERIC = enum_CUmemHandleType_enum.define('CU_MEM_HANDLE_TYPE_GENERIC', 0)

CUmemHandleType = enum_CUmemHandleType_enum
class struct_CUarrayMapInfo_st(Struct): pass
class _anonunion22(Union): pass
_anonunion22.SIZE = 8
_anonunion22._fields_ = ['mipmap', 'array']
setattr(_anonunion22, 'mipmap', field(0, CUmipmappedArray))
setattr(_anonunion22, 'array', field(0, CUarray))
class _anonunion23(Union): pass
class _anonstruct24(Struct): pass
_anonstruct24.SIZE = 32
_anonstruct24._fields_ = ['level', 'layer', 'offsetX', 'offsetY', 'offsetZ', 'extentWidth', 'extentHeight', 'extentDepth']
setattr(_anonstruct24, 'level', field(0, ctypes.c_uint32))
setattr(_anonstruct24, 'layer', field(4, ctypes.c_uint32))
setattr(_anonstruct24, 'offsetX', field(8, ctypes.c_uint32))
setattr(_anonstruct24, 'offsetY', field(12, ctypes.c_uint32))
setattr(_anonstruct24, 'offsetZ', field(16, ctypes.c_uint32))
setattr(_anonstruct24, 'extentWidth', field(20, ctypes.c_uint32))
setattr(_anonstruct24, 'extentHeight', field(24, ctypes.c_uint32))
setattr(_anonstruct24, 'extentDepth', field(28, ctypes.c_uint32))
class _anonstruct25(Struct): pass
_anonstruct25.SIZE = 24
_anonstruct25._fields_ = ['layer', 'offset', 'size']
setattr(_anonstruct25, 'layer', field(0, ctypes.c_uint32))
setattr(_anonstruct25, 'offset', field(8, ctypes.c_uint64))
setattr(_anonstruct25, 'size', field(16, ctypes.c_uint64))
_anonunion23.SIZE = 32
_anonunion23._fields_ = ['sparseLevel', 'miptail']
setattr(_anonunion23, 'sparseLevel', field(0, _anonstruct24))
setattr(_anonunion23, 'miptail', field(0, _anonstruct25))
class _anonunion26(Union): pass
_anonunion26.SIZE = 8
_anonunion26._fields_ = ['memHandle']
setattr(_anonunion26, 'memHandle', field(0, CUmemGenericAllocationHandle))
struct_CUarrayMapInfo_st.SIZE = 96
struct_CUarrayMapInfo_st._fields_ = ['resourceType', 'resource', 'subresourceType', 'subresource', 'memOperationType', 'memHandleType', 'memHandle', 'offset', 'deviceBitMask', 'flags', 'reserved']
setattr(struct_CUarrayMapInfo_st, 'resourceType', field(0, CUresourcetype))
setattr(struct_CUarrayMapInfo_st, 'resource', field(8, _anonunion22))
setattr(struct_CUarrayMapInfo_st, 'subresourceType', field(16, CUarraySparseSubresourceType))
setattr(struct_CUarrayMapInfo_st, 'subresource', field(24, _anonunion23))
setattr(struct_CUarrayMapInfo_st, 'memOperationType', field(56, CUmemOperationType))
setattr(struct_CUarrayMapInfo_st, 'memHandleType', field(60, CUmemHandleType))
setattr(struct_CUarrayMapInfo_st, 'memHandle', field(64, _anonunion26))
setattr(struct_CUarrayMapInfo_st, 'offset', field(72, ctypes.c_uint64))
setattr(struct_CUarrayMapInfo_st, 'deviceBitMask', field(80, ctypes.c_uint32))
setattr(struct_CUarrayMapInfo_st, 'flags', field(84, ctypes.c_uint32))
setattr(struct_CUarrayMapInfo_st, 'reserved', field(88, Array(ctypes.c_uint32, 2)))
CUarrayMapInfo_v1 = struct_CUarrayMapInfo_st
CUarrayMapInfo = struct_CUarrayMapInfo_st
class struct_CUmemLocation_st(Struct): pass
struct_CUmemLocation_st.SIZE = 8
struct_CUmemLocation_st._fields_ = ['type', 'id']
setattr(struct_CUmemLocation_st, 'type', field(0, CUmemLocationType))
setattr(struct_CUmemLocation_st, 'id', field(4, ctypes.c_int32))
CUmemLocation_v1 = struct_CUmemLocation_st
CUmemLocation = struct_CUmemLocation_st
enum_CUmemAllocationCompType_enum = CEnum(ctypes.c_uint32)
CU_MEM_ALLOCATION_COMP_NONE = enum_CUmemAllocationCompType_enum.define('CU_MEM_ALLOCATION_COMP_NONE', 0)
CU_MEM_ALLOCATION_COMP_GENERIC = enum_CUmemAllocationCompType_enum.define('CU_MEM_ALLOCATION_COMP_GENERIC', 1)

CUmemAllocationCompType = enum_CUmemAllocationCompType_enum
class struct_CUmemAllocationProp_st(Struct): pass
class _anonstruct27(Struct): pass
_anonstruct27.SIZE = 8
_anonstruct27._fields_ = ['compressionType', 'gpuDirectRDMACapable', 'usage', 'reserved']
setattr(_anonstruct27, 'compressionType', field(0, ctypes.c_ubyte))
setattr(_anonstruct27, 'gpuDirectRDMACapable', field(1, ctypes.c_ubyte))
setattr(_anonstruct27, 'usage', field(2, ctypes.c_uint16))
setattr(_anonstruct27, 'reserved', field(4, Array(ctypes.c_ubyte, 4)))
struct_CUmemAllocationProp_st.SIZE = 32
struct_CUmemAllocationProp_st._fields_ = ['type', 'requestedHandleTypes', 'location', 'win32HandleMetaData', 'allocFlags']
setattr(struct_CUmemAllocationProp_st, 'type', field(0, CUmemAllocationType))
setattr(struct_CUmemAllocationProp_st, 'requestedHandleTypes', field(4, CUmemAllocationHandleType))
setattr(struct_CUmemAllocationProp_st, 'location', field(8, CUmemLocation))
setattr(struct_CUmemAllocationProp_st, 'win32HandleMetaData', field(16, ctypes.c_void_p))
setattr(struct_CUmemAllocationProp_st, 'allocFlags', field(24, _anonstruct27))
CUmemAllocationProp_v1 = struct_CUmemAllocationProp_st
CUmemAllocationProp = struct_CUmemAllocationProp_st
class struct_CUmemAccessDesc_st(Struct): pass
struct_CUmemAccessDesc_st.SIZE = 12
struct_CUmemAccessDesc_st._fields_ = ['location', 'flags']
setattr(struct_CUmemAccessDesc_st, 'location', field(0, CUmemLocation))
setattr(struct_CUmemAccessDesc_st, 'flags', field(8, CUmemAccess_flags))
CUmemAccessDesc_v1 = struct_CUmemAccessDesc_st
CUmemAccessDesc = struct_CUmemAccessDesc_st
enum_CUgraphExecUpdateResult_enum = CEnum(ctypes.c_uint32)
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
class struct_CUgraphExecUpdateResultInfo_st(Struct): pass
struct_CUgraphExecUpdateResultInfo_st.SIZE = 24
struct_CUgraphExecUpdateResultInfo_st._fields_ = ['result', 'errorNode', 'errorFromNode']
setattr(struct_CUgraphExecUpdateResultInfo_st, 'result', field(0, CUgraphExecUpdateResult))
setattr(struct_CUgraphExecUpdateResultInfo_st, 'errorNode', field(8, CUgraphNode))
setattr(struct_CUgraphExecUpdateResultInfo_st, 'errorFromNode', field(16, CUgraphNode))
CUgraphExecUpdateResultInfo_v1 = struct_CUgraphExecUpdateResultInfo_st
CUgraphExecUpdateResultInfo = struct_CUgraphExecUpdateResultInfo_st
enum_CUmemPool_attribute_enum = CEnum(ctypes.c_uint32)
CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES', 1)
CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC', 2)
CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES', 3)
CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RELEASE_THRESHOLD', 4)
CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT', 5)
CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH', 6)
CU_MEMPOOL_ATTR_USED_MEM_CURRENT = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_USED_MEM_CURRENT', 7)
CU_MEMPOOL_ATTR_USED_MEM_HIGH = enum_CUmemPool_attribute_enum.define('CU_MEMPOOL_ATTR_USED_MEM_HIGH', 8)

CUmemPool_attribute = enum_CUmemPool_attribute_enum
class struct_CUmemPoolProps_st(Struct): pass
struct_CUmemPoolProps_st.SIZE = 88
struct_CUmemPoolProps_st._fields_ = ['allocType', 'handleTypes', 'location', 'win32SecurityAttributes', 'reserved']
setattr(struct_CUmemPoolProps_st, 'allocType', field(0, CUmemAllocationType))
setattr(struct_CUmemPoolProps_st, 'handleTypes', field(4, CUmemAllocationHandleType))
setattr(struct_CUmemPoolProps_st, 'location', field(8, CUmemLocation))
setattr(struct_CUmemPoolProps_st, 'win32SecurityAttributes', field(16, ctypes.c_void_p))
setattr(struct_CUmemPoolProps_st, 'reserved', field(24, Array(ctypes.c_ubyte, 64)))
CUmemPoolProps_v1 = struct_CUmemPoolProps_st
CUmemPoolProps = struct_CUmemPoolProps_st
class struct_CUmemPoolPtrExportData_st(Struct): pass
struct_CUmemPoolPtrExportData_st.SIZE = 64
struct_CUmemPoolPtrExportData_st._fields_ = ['reserved']
setattr(struct_CUmemPoolPtrExportData_st, 'reserved', field(0, Array(ctypes.c_ubyte, 64)))
CUmemPoolPtrExportData_v1 = struct_CUmemPoolPtrExportData_st
CUmemPoolPtrExportData = struct_CUmemPoolPtrExportData_st
class struct_CUDA_MEM_ALLOC_NODE_PARAMS_st(Struct): pass
struct_CUDA_MEM_ALLOC_NODE_PARAMS_st.SIZE = 120
struct_CUDA_MEM_ALLOC_NODE_PARAMS_st._fields_ = ['poolProps', 'accessDescs', 'accessDescCount', 'bytesize', 'dptr']
setattr(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st, 'poolProps', field(0, CUmemPoolProps))
setattr(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st, 'accessDescs', field(88, Pointer(CUmemAccessDesc)))
setattr(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st, 'accessDescCount', field(96, size_t))
setattr(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st, 'bytesize', field(104, size_t))
setattr(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st, 'dptr', field(112, CUdeviceptr))
CUDA_MEM_ALLOC_NODE_PARAMS = struct_CUDA_MEM_ALLOC_NODE_PARAMS_st
enum_CUgraphMem_attribute_enum = CEnum(ctypes.c_uint32)
CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT', 0)
CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_USED_MEM_HIGH', 1)
CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT', 2)
CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = enum_CUgraphMem_attribute_enum.define('CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH', 3)

CUgraphMem_attribute = enum_CUgraphMem_attribute_enum
enum_CUflushGPUDirectRDMAWritesOptions_enum = CEnum(ctypes.c_uint32)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = enum_CUflushGPUDirectRDMAWritesOptions_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST', 1)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = enum_CUflushGPUDirectRDMAWritesOptions_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS', 2)

CUflushGPUDirectRDMAWritesOptions = enum_CUflushGPUDirectRDMAWritesOptions_enum
enum_CUGPUDirectRDMAWritesOrdering_enum = CEnum(ctypes.c_uint32)
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE', 0)
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER', 100)
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = enum_CUGPUDirectRDMAWritesOrdering_enum.define('CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES', 200)

CUGPUDirectRDMAWritesOrdering = enum_CUGPUDirectRDMAWritesOrdering_enum
enum_CUflushGPUDirectRDMAWritesScope_enum = CEnum(ctypes.c_uint32)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = enum_CUflushGPUDirectRDMAWritesScope_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER', 100)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = enum_CUflushGPUDirectRDMAWritesScope_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES', 200)

CUflushGPUDirectRDMAWritesScope = enum_CUflushGPUDirectRDMAWritesScope_enum
enum_CUflushGPUDirectRDMAWritesTarget_enum = CEnum(ctypes.c_uint32)
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = enum_CUflushGPUDirectRDMAWritesTarget_enum.define('CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX', 0)

CUflushGPUDirectRDMAWritesTarget = enum_CUflushGPUDirectRDMAWritesTarget_enum
enum_CUgraphDebugDot_flags_enum = CEnum(ctypes.c_uint32)
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
enum_CUuserObject_flags_enum = CEnum(ctypes.c_uint32)
CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = enum_CUuserObject_flags_enum.define('CU_USER_OBJECT_NO_DESTRUCTOR_SYNC', 1)

CUuserObject_flags = enum_CUuserObject_flags_enum
enum_CUuserObjectRetain_flags_enum = CEnum(ctypes.c_uint32)
CU_GRAPH_USER_OBJECT_MOVE = enum_CUuserObjectRetain_flags_enum.define('CU_GRAPH_USER_OBJECT_MOVE', 1)

CUuserObjectRetain_flags = enum_CUuserObjectRetain_flags_enum
enum_CUgraphInstantiate_flags_enum = CEnum(ctypes.c_uint32)
CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH', 1)
CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD', 2)
CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH', 4)
CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = enum_CUgraphInstantiate_flags_enum.define('CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY', 8)

CUgraphInstantiate_flags = enum_CUgraphInstantiate_flags_enum
@dll.bind((CUresult, Pointer(Pointer(ctypes.c_char)),), CUresult)
def cuGetErrorString(error, pStr): ...
@dll.bind((CUresult, Pointer(Pointer(ctypes.c_char)),), CUresult)
def cuGetErrorName(error, pStr): ...
@dll.bind((ctypes.c_uint32,), CUresult)
def cuInit(Flags): ...
@dll.bind((Pointer(ctypes.c_int32),), CUresult)
def cuDriverGetVersion(driverVersion): ...
@dll.bind((Pointer(CUdevice), ctypes.c_int32,), CUresult)
def cuDeviceGet(device, ordinal): ...
@dll.bind((Pointer(ctypes.c_int32),), CUresult)
def cuDeviceGetCount(count): ...
@dll.bind((Pointer(ctypes.c_char), ctypes.c_int32, CUdevice,), CUresult)
def cuDeviceGetName(name, len, dev): ...
@dll.bind((Pointer(CUuuid), CUdevice,), CUresult)
def cuDeviceGetUuid(uuid, dev): ...
@dll.bind((Pointer(CUuuid), CUdevice,), CUresult)
def cuDeviceGetUuid_v2(uuid, dev): ...
@dll.bind((Pointer(ctypes.c_char), Pointer(ctypes.c_uint32), CUdevice,), CUresult)
def cuDeviceGetLuid(luid, deviceNodeMask, dev): ...
@dll.bind((Pointer(size_t), CUdevice,), CUresult)
def cuDeviceTotalMem_v2(bytes, dev): ...
@dll.bind((Pointer(size_t), CUarray_format, ctypes.c_uint32, CUdevice,), CUresult)
def cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev): ...
@dll.bind((Pointer(ctypes.c_int32), CUdevice_attribute, CUdevice,), CUresult)
def cuDeviceGetAttribute(pi, attrib, dev): ...
@dll.bind((ctypes.c_void_p, CUdevice, ctypes.c_int32,), CUresult)
def cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags): ...
@dll.bind((CUdevice, CUmemoryPool,), CUresult)
def cuDeviceSetMemPool(dev, pool): ...
@dll.bind((Pointer(CUmemoryPool), CUdevice,), CUresult)
def cuDeviceGetMemPool(pool, dev): ...
@dll.bind((Pointer(CUmemoryPool), CUdevice,), CUresult)
def cuDeviceGetDefaultMemPool(pool_out, dev): ...
@dll.bind((Pointer(ctypes.c_int32), CUexecAffinityType, CUdevice,), CUresult)
def cuDeviceGetExecAffinitySupport(pi, type, dev): ...
@dll.bind((CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope,), CUresult)
def cuFlushGPUDirectRDMAWrites(target, scope): ...
@dll.bind((Pointer(CUdevprop), CUdevice,), CUresult)
def cuDeviceGetProperties(prop, dev): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), CUdevice,), CUresult)
def cuDeviceComputeCapability(major, minor, dev): ...
@dll.bind((Pointer(CUcontext), CUdevice,), CUresult)
def cuDevicePrimaryCtxRetain(pctx, dev): ...
@dll.bind((CUdevice,), CUresult)
def cuDevicePrimaryCtxRelease_v2(dev): ...
@dll.bind((CUdevice, ctypes.c_uint32,), CUresult)
def cuDevicePrimaryCtxSetFlags_v2(dev, flags): ...
@dll.bind((CUdevice, Pointer(ctypes.c_uint32), Pointer(ctypes.c_int32),), CUresult)
def cuDevicePrimaryCtxGetState(dev, flags, active): ...
@dll.bind((CUdevice,), CUresult)
def cuDevicePrimaryCtxReset_v2(dev): ...
@dll.bind((Pointer(CUcontext), ctypes.c_uint32, CUdevice,), CUresult)
def cuCtxCreate_v2(pctx, flags, dev): ...
@dll.bind((Pointer(CUcontext), Pointer(CUexecAffinityParam), ctypes.c_int32, ctypes.c_uint32, CUdevice,), CUresult)
def cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev): ...
@dll.bind((CUcontext,), CUresult)
def cuCtxDestroy_v2(ctx): ...
@dll.bind((CUcontext,), CUresult)
def cuCtxPushCurrent_v2(ctx): ...
@dll.bind((Pointer(CUcontext),), CUresult)
def cuCtxPopCurrent_v2(pctx): ...
@dll.bind((CUcontext,), CUresult)
def cuCtxSetCurrent(ctx): ...
@dll.bind((Pointer(CUcontext),), CUresult)
def cuCtxGetCurrent(pctx): ...
@dll.bind((Pointer(CUdevice),), CUresult)
def cuCtxGetDevice(device): ...
@dll.bind((Pointer(ctypes.c_uint32),), CUresult)
def cuCtxGetFlags(flags): ...
@dll.bind((CUcontext, Pointer(ctypes.c_uint64),), CUresult)
def cuCtxGetId(ctx, ctxId): ...
@dll.bind((), CUresult)
def cuCtxSynchronize(): ...
@dll.bind((CUlimit, size_t,), CUresult)
def cuCtxSetLimit(limit, value): ...
@dll.bind((Pointer(size_t), CUlimit,), CUresult)
def cuCtxGetLimit(pvalue, limit): ...
@dll.bind((Pointer(CUfunc_cache),), CUresult)
def cuCtxGetCacheConfig(pconfig): ...
@dll.bind((CUfunc_cache,), CUresult)
def cuCtxSetCacheConfig(config): ...
@dll.bind((Pointer(CUsharedconfig),), CUresult)
def cuCtxGetSharedMemConfig(pConfig): ...
@dll.bind((CUsharedconfig,), CUresult)
def cuCtxSetSharedMemConfig(config): ...
@dll.bind((CUcontext, Pointer(ctypes.c_uint32),), CUresult)
def cuCtxGetApiVersion(ctx, version): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32),), CUresult)
def cuCtxGetStreamPriorityRange(leastPriority, greatestPriority): ...
@dll.bind((), CUresult)
def cuCtxResetPersistingL2Cache(): ...
@dll.bind((Pointer(CUexecAffinityParam), CUexecAffinityType,), CUresult)
def cuCtxGetExecAffinity(pExecAffinity, type): ...
@dll.bind((Pointer(CUcontext), ctypes.c_uint32,), CUresult)
def cuCtxAttach(pctx, flags): ...
@dll.bind((CUcontext,), CUresult)
def cuCtxDetach(ctx): ...
@dll.bind((Pointer(CUmodule), Pointer(ctypes.c_char),), CUresult)
def cuModuleLoad(module, fname): ...
@dll.bind((Pointer(CUmodule), ctypes.c_void_p,), CUresult)
def cuModuleLoadData(module, image): ...
@dll.bind((Pointer(CUmodule), ctypes.c_void_p, ctypes.c_uint32, Pointer(CUjit_option), Pointer(ctypes.c_void_p),), CUresult)
def cuModuleLoadDataEx(module, image, numOptions, options, optionValues): ...
@dll.bind((Pointer(CUmodule), ctypes.c_void_p,), CUresult)
def cuModuleLoadFatBinary(module, fatCubin): ...
@dll.bind((CUmodule,), CUresult)
def cuModuleUnload(hmod): ...
enum_CUmoduleLoadingMode_enum = CEnum(ctypes.c_uint32)
CU_MODULE_EAGER_LOADING = enum_CUmoduleLoadingMode_enum.define('CU_MODULE_EAGER_LOADING', 1)
CU_MODULE_LAZY_LOADING = enum_CUmoduleLoadingMode_enum.define('CU_MODULE_LAZY_LOADING', 2)

CUmoduleLoadingMode = enum_CUmoduleLoadingMode_enum
@dll.bind((Pointer(CUmoduleLoadingMode),), CUresult)
def cuModuleGetLoadingMode(mode): ...
@dll.bind((Pointer(CUfunction), CUmodule, Pointer(ctypes.c_char),), CUresult)
def cuModuleGetFunction(hfunc, hmod, name): ...
@dll.bind((Pointer(CUdeviceptr), Pointer(size_t), CUmodule, Pointer(ctypes.c_char),), CUresult)
def cuModuleGetGlobal_v2(dptr, bytes, hmod, name): ...
@dll.bind((ctypes.c_uint32, Pointer(CUjit_option), Pointer(ctypes.c_void_p), Pointer(CUlinkState),), CUresult)
def cuLinkCreate_v2(numOptions, options, optionValues, stateOut): ...
@dll.bind((CUlinkState, CUjitInputType, ctypes.c_void_p, size_t, Pointer(ctypes.c_char), ctypes.c_uint32, Pointer(CUjit_option), Pointer(ctypes.c_void_p),), CUresult)
def cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues): ...
@dll.bind((CUlinkState, CUjitInputType, Pointer(ctypes.c_char), ctypes.c_uint32, Pointer(CUjit_option), Pointer(ctypes.c_void_p),), CUresult)
def cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues): ...
@dll.bind((CUlinkState, Pointer(ctypes.c_void_p), Pointer(size_t),), CUresult)
def cuLinkComplete(state, cubinOut, sizeOut): ...
@dll.bind((CUlinkState,), CUresult)
def cuLinkDestroy(state): ...
@dll.bind((Pointer(CUtexref), CUmodule, Pointer(ctypes.c_char),), CUresult)
def cuModuleGetTexRef(pTexRef, hmod, name): ...
@dll.bind((Pointer(CUsurfref), CUmodule, Pointer(ctypes.c_char),), CUresult)
def cuModuleGetSurfRef(pSurfRef, hmod, name): ...
@dll.bind((Pointer(CUlibrary), ctypes.c_void_p, Pointer(CUjit_option), Pointer(ctypes.c_void_p), ctypes.c_uint32, Pointer(CUlibraryOption), Pointer(ctypes.c_void_p), ctypes.c_uint32,), CUresult)
def cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions): ...
@dll.bind((Pointer(CUlibrary), Pointer(ctypes.c_char), Pointer(CUjit_option), Pointer(ctypes.c_void_p), ctypes.c_uint32, Pointer(CUlibraryOption), Pointer(ctypes.c_void_p), ctypes.c_uint32,), CUresult)
def cuLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions): ...
@dll.bind((CUlibrary,), CUresult)
def cuLibraryUnload(library): ...
@dll.bind((Pointer(CUkernel), CUlibrary, Pointer(ctypes.c_char),), CUresult)
def cuLibraryGetKernel(pKernel, library, name): ...
@dll.bind((Pointer(CUmodule), CUlibrary,), CUresult)
def cuLibraryGetModule(pMod, library): ...
@dll.bind((Pointer(CUfunction), CUkernel,), CUresult)
def cuKernelGetFunction(pFunc, kernel): ...
@dll.bind((Pointer(CUdeviceptr), Pointer(size_t), CUlibrary, Pointer(ctypes.c_char),), CUresult)
def cuLibraryGetGlobal(dptr, bytes, library, name): ...
@dll.bind((Pointer(CUdeviceptr), Pointer(size_t), CUlibrary, Pointer(ctypes.c_char),), CUresult)
def cuLibraryGetManaged(dptr, bytes, library, name): ...
@dll.bind((Pointer(ctypes.c_void_p), CUlibrary, Pointer(ctypes.c_char),), CUresult)
def cuLibraryGetUnifiedFunction(fptr, library, symbol): ...
@dll.bind((Pointer(ctypes.c_int32), CUfunction_attribute, CUkernel, CUdevice,), CUresult)
def cuKernelGetAttribute(pi, attrib, kernel, dev): ...
@dll.bind((CUfunction_attribute, ctypes.c_int32, CUkernel, CUdevice,), CUresult)
def cuKernelSetAttribute(attrib, val, kernel, dev): ...
@dll.bind((CUkernel, CUfunc_cache, CUdevice,), CUresult)
def cuKernelSetCacheConfig(kernel, config, dev): ...
@dll.bind((Pointer(size_t), Pointer(size_t),), CUresult)
def cuMemGetInfo_v2(free, total): ...
@dll.bind((Pointer(CUdeviceptr), size_t,), CUresult)
def cuMemAlloc_v2(dptr, bytesize): ...
@dll.bind((Pointer(CUdeviceptr), Pointer(size_t), size_t, size_t, ctypes.c_uint32,), CUresult)
def cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes): ...
@dll.bind((CUdeviceptr,), CUresult)
def cuMemFree_v2(dptr): ...
@dll.bind((Pointer(CUdeviceptr), Pointer(size_t), CUdeviceptr,), CUresult)
def cuMemGetAddressRange_v2(pbase, psize, dptr): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t,), CUresult)
def cuMemAllocHost_v2(pp, bytesize): ...
@dll.bind((ctypes.c_void_p,), CUresult)
def cuMemFreeHost(p): ...
@dll.bind((Pointer(ctypes.c_void_p), size_t, ctypes.c_uint32,), CUresult)
def cuMemHostAlloc(pp, bytesize, Flags): ...
@dll.bind((Pointer(CUdeviceptr), ctypes.c_void_p, ctypes.c_uint32,), CUresult)
def cuMemHostGetDevicePointer_v2(pdptr, p, Flags): ...
@dll.bind((Pointer(ctypes.c_uint32), ctypes.c_void_p,), CUresult)
def cuMemHostGetFlags(pFlags, p): ...
@dll.bind((Pointer(CUdeviceptr), size_t, ctypes.c_uint32,), CUresult)
def cuMemAllocManaged(dptr, bytesize, flags): ...
@dll.bind((Pointer(CUdevice), Pointer(ctypes.c_char),), CUresult)
def cuDeviceGetByPCIBusId(dev, pciBusId): ...
@dll.bind((Pointer(ctypes.c_char), ctypes.c_int32, CUdevice,), CUresult)
def cuDeviceGetPCIBusId(pciBusId, len, dev): ...
@dll.bind((Pointer(CUipcEventHandle), CUevent,), CUresult)
def cuIpcGetEventHandle(pHandle, event): ...
@dll.bind((Pointer(CUevent), CUipcEventHandle,), CUresult)
def cuIpcOpenEventHandle(phEvent, handle): ...
@dll.bind((Pointer(CUipcMemHandle), CUdeviceptr,), CUresult)
def cuIpcGetMemHandle(pHandle, dptr): ...
@dll.bind((Pointer(CUdeviceptr), CUipcMemHandle, ctypes.c_uint32,), CUresult)
def cuIpcOpenMemHandle_v2(pdptr, handle, Flags): ...
@dll.bind((CUdeviceptr,), CUresult)
def cuIpcCloseMemHandle(dptr): ...
@dll.bind((ctypes.c_void_p, size_t, ctypes.c_uint32,), CUresult)
def cuMemHostRegister_v2(p, bytesize, Flags): ...
@dll.bind((ctypes.c_void_p,), CUresult)
def cuMemHostUnregister(p): ...
@dll.bind((CUdeviceptr, CUdeviceptr, size_t,), CUresult)
def cuMemcpy_ptds(dst, src, ByteCount): ...
@dll.bind((CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t,), CUresult)
def cuMemcpyPeer_ptds(dstDevice, dstContext, srcDevice, srcContext, ByteCount): ...
@dll.bind((CUdeviceptr, ctypes.c_void_p, size_t,), CUresult)
def cuMemcpyHtoD_v2_ptds(dstDevice, srcHost, ByteCount): ...
@dll.bind((ctypes.c_void_p, CUdeviceptr, size_t,), CUresult)
def cuMemcpyDtoH_v2_ptds(dstHost, srcDevice, ByteCount): ...
@dll.bind((CUdeviceptr, CUdeviceptr, size_t,), CUresult)
def cuMemcpyDtoD_v2_ptds(dstDevice, srcDevice, ByteCount): ...
@dll.bind((CUarray, size_t, CUdeviceptr, size_t,), CUresult)
def cuMemcpyDtoA_v2_ptds(dstArray, dstOffset, srcDevice, ByteCount): ...
@dll.bind((CUdeviceptr, CUarray, size_t, size_t,), CUresult)
def cuMemcpyAtoD_v2_ptds(dstDevice, srcArray, srcOffset, ByteCount): ...
@dll.bind((CUarray, size_t, ctypes.c_void_p, size_t,), CUresult)
def cuMemcpyHtoA_v2_ptds(dstArray, dstOffset, srcHost, ByteCount): ...
@dll.bind((ctypes.c_void_p, CUarray, size_t, size_t,), CUresult)
def cuMemcpyAtoH_v2_ptds(dstHost, srcArray, srcOffset, ByteCount): ...
@dll.bind((CUarray, size_t, CUarray, size_t, size_t,), CUresult)
def cuMemcpyAtoA_v2_ptds(dstArray, dstOffset, srcArray, srcOffset, ByteCount): ...
@dll.bind((Pointer(CUDA_MEMCPY2D),), CUresult)
def cuMemcpy2D_v2_ptds(pCopy): ...
@dll.bind((Pointer(CUDA_MEMCPY2D),), CUresult)
def cuMemcpy2DUnaligned_v2_ptds(pCopy): ...
@dll.bind((Pointer(CUDA_MEMCPY3D),), CUresult)
def cuMemcpy3D_v2_ptds(pCopy): ...
@dll.bind((Pointer(CUDA_MEMCPY3D_PEER),), CUresult)
def cuMemcpy3DPeer_ptds(pCopy): ...
@dll.bind((CUdeviceptr, CUdeviceptr, size_t, CUstream,), CUresult)
def cuMemcpyAsync_ptsz(dst, src, ByteCount, hStream): ...
@dll.bind((CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream,), CUresult)
def cuMemcpyPeerAsync_ptsz(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream): ...
@dll.bind((CUdeviceptr, ctypes.c_void_p, size_t, CUstream,), CUresult)
def cuMemcpyHtoDAsync_v2_ptsz(dstDevice, srcHost, ByteCount, hStream): ...
@dll.bind((ctypes.c_void_p, CUdeviceptr, size_t, CUstream,), CUresult)
def cuMemcpyDtoHAsync_v2_ptsz(dstHost, srcDevice, ByteCount, hStream): ...
@dll.bind((CUdeviceptr, CUdeviceptr, size_t, CUstream,), CUresult)
def cuMemcpyDtoDAsync_v2_ptsz(dstDevice, srcDevice, ByteCount, hStream): ...
@dll.bind((CUarray, size_t, ctypes.c_void_p, size_t, CUstream,), CUresult)
def cuMemcpyHtoAAsync_v2_ptsz(dstArray, dstOffset, srcHost, ByteCount, hStream): ...
@dll.bind((ctypes.c_void_p, CUarray, size_t, size_t, CUstream,), CUresult)
def cuMemcpyAtoHAsync_v2_ptsz(dstHost, srcArray, srcOffset, ByteCount, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY2D), CUstream,), CUresult)
def cuMemcpy2DAsync_v2_ptsz(pCopy, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY3D), CUstream,), CUresult)
def cuMemcpy3DAsync_v2_ptsz(pCopy, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY3D_PEER), CUstream,), CUresult)
def cuMemcpy3DPeerAsync_ptsz(pCopy, hStream): ...
@dll.bind((CUdeviceptr, ctypes.c_ubyte, size_t,), CUresult)
def cuMemsetD8_v2_ptds(dstDevice, uc, N): ...
@dll.bind((CUdeviceptr, ctypes.c_uint16, size_t,), CUresult)
def cuMemsetD16_v2_ptds(dstDevice, us, N): ...
@dll.bind((CUdeviceptr, ctypes.c_uint32, size_t,), CUresult)
def cuMemsetD32_v2_ptds(dstDevice, ui, N): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t,), CUresult)
def cuMemsetD2D8_v2_ptds(dstDevice, dstPitch, uc, Width, Height): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_uint16, size_t, size_t,), CUresult)
def cuMemsetD2D16_v2_ptds(dstDevice, dstPitch, us, Width, Height): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_uint32, size_t, size_t,), CUresult)
def cuMemsetD2D32_v2_ptds(dstDevice, dstPitch, ui, Width, Height): ...
@dll.bind((CUdeviceptr, ctypes.c_ubyte, size_t, CUstream,), CUresult)
def cuMemsetD8Async_ptsz(dstDevice, uc, N, hStream): ...
@dll.bind((CUdeviceptr, ctypes.c_uint16, size_t, CUstream,), CUresult)
def cuMemsetD16Async_ptsz(dstDevice, us, N, hStream): ...
@dll.bind((CUdeviceptr, ctypes.c_uint32, size_t, CUstream,), CUresult)
def cuMemsetD32Async_ptsz(dstDevice, ui, N, hStream): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t, CUstream,), CUresult)
def cuMemsetD2D8Async_ptsz(dstDevice, dstPitch, uc, Width, Height, hStream): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_uint16, size_t, size_t, CUstream,), CUresult)
def cuMemsetD2D16Async_ptsz(dstDevice, dstPitch, us, Width, Height, hStream): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_uint32, size_t, size_t, CUstream,), CUresult)
def cuMemsetD2D32Async_ptsz(dstDevice, dstPitch, ui, Width, Height, hStream): ...
@dll.bind((Pointer(CUarray), Pointer(CUDA_ARRAY_DESCRIPTOR),), CUresult)
def cuArrayCreate_v2(pHandle, pAllocateArray): ...
@dll.bind((Pointer(CUDA_ARRAY_DESCRIPTOR), CUarray,), CUresult)
def cuArrayGetDescriptor_v2(pArrayDescriptor, hArray): ...
@dll.bind((Pointer(CUDA_ARRAY_SPARSE_PROPERTIES), CUarray,), CUresult)
def cuArrayGetSparseProperties(sparseProperties, array): ...
@dll.bind((Pointer(CUDA_ARRAY_SPARSE_PROPERTIES), CUmipmappedArray,), CUresult)
def cuMipmappedArrayGetSparseProperties(sparseProperties, mipmap): ...
@dll.bind((Pointer(CUDA_ARRAY_MEMORY_REQUIREMENTS), CUarray, CUdevice,), CUresult)
def cuArrayGetMemoryRequirements(memoryRequirements, array, device): ...
@dll.bind((Pointer(CUDA_ARRAY_MEMORY_REQUIREMENTS), CUmipmappedArray, CUdevice,), CUresult)
def cuMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device): ...
@dll.bind((Pointer(CUarray), CUarray, ctypes.c_uint32,), CUresult)
def cuArrayGetPlane(pPlaneArray, hArray, planeIdx): ...
@dll.bind((CUarray,), CUresult)
def cuArrayDestroy(hArray): ...
@dll.bind((Pointer(CUarray), Pointer(CUDA_ARRAY3D_DESCRIPTOR),), CUresult)
def cuArray3DCreate_v2(pHandle, pAllocateArray): ...
@dll.bind((Pointer(CUDA_ARRAY3D_DESCRIPTOR), CUarray,), CUresult)
def cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray): ...
@dll.bind((Pointer(CUmipmappedArray), Pointer(CUDA_ARRAY3D_DESCRIPTOR), ctypes.c_uint32,), CUresult)
def cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels): ...
@dll.bind((Pointer(CUarray), CUmipmappedArray, ctypes.c_uint32,), CUresult)
def cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level): ...
@dll.bind((CUmipmappedArray,), CUresult)
def cuMipmappedArrayDestroy(hMipmappedArray): ...
@dll.bind((ctypes.c_void_p, CUdeviceptr, size_t, CUmemRangeHandleType, ctypes.c_uint64,), CUresult)
def cuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags): ...
@dll.bind((Pointer(CUdeviceptr), size_t, size_t, CUdeviceptr, ctypes.c_uint64,), CUresult)
def cuMemAddressReserve(ptr, size, alignment, addr, flags): ...
@dll.bind((CUdeviceptr, size_t,), CUresult)
def cuMemAddressFree(ptr, size): ...
@dll.bind((Pointer(CUmemGenericAllocationHandle), size_t, Pointer(CUmemAllocationProp), ctypes.c_uint64,), CUresult)
def cuMemCreate(handle, size, prop, flags): ...
@dll.bind((CUmemGenericAllocationHandle,), CUresult)
def cuMemRelease(handle): ...
@dll.bind((CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, ctypes.c_uint64,), CUresult)
def cuMemMap(ptr, size, offset, handle, flags): ...
@dll.bind((Pointer(CUarrayMapInfo), ctypes.c_uint32, CUstream,), CUresult)
def cuMemMapArrayAsync_ptsz(mapInfoList, count, hStream): ...
@dll.bind((CUdeviceptr, size_t,), CUresult)
def cuMemUnmap(ptr, size): ...
@dll.bind((CUdeviceptr, size_t, Pointer(CUmemAccessDesc), size_t,), CUresult)
def cuMemSetAccess(ptr, size, desc, count): ...
@dll.bind((Pointer(ctypes.c_uint64), Pointer(CUmemLocation), CUdeviceptr,), CUresult)
def cuMemGetAccess(flags, location, ptr): ...
@dll.bind((ctypes.c_void_p, CUmemGenericAllocationHandle, CUmemAllocationHandleType, ctypes.c_uint64,), CUresult)
def cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags): ...
@dll.bind((Pointer(CUmemGenericAllocationHandle), ctypes.c_void_p, CUmemAllocationHandleType,), CUresult)
def cuMemImportFromShareableHandle(handle, osHandle, shHandleType): ...
@dll.bind((Pointer(size_t), Pointer(CUmemAllocationProp), CUmemAllocationGranularity_flags,), CUresult)
def cuMemGetAllocationGranularity(granularity, prop, option): ...
@dll.bind((Pointer(CUmemAllocationProp), CUmemGenericAllocationHandle,), CUresult)
def cuMemGetAllocationPropertiesFromHandle(prop, handle): ...
@dll.bind((Pointer(CUmemGenericAllocationHandle), ctypes.c_void_p,), CUresult)
def cuMemRetainAllocationHandle(handle, addr): ...
@dll.bind((CUdeviceptr, CUstream,), CUresult)
def cuMemFreeAsync_ptsz(dptr, hStream): ...
@dll.bind((Pointer(CUdeviceptr), size_t, CUstream,), CUresult)
def cuMemAllocAsync_ptsz(dptr, bytesize, hStream): ...
@dll.bind((CUmemoryPool, size_t,), CUresult)
def cuMemPoolTrimTo(pool, minBytesToKeep): ...
@dll.bind((CUmemoryPool, CUmemPool_attribute, ctypes.c_void_p,), CUresult)
def cuMemPoolSetAttribute(pool, attr, value): ...
@dll.bind((CUmemoryPool, CUmemPool_attribute, ctypes.c_void_p,), CUresult)
def cuMemPoolGetAttribute(pool, attr, value): ...
@dll.bind((CUmemoryPool, Pointer(CUmemAccessDesc), size_t,), CUresult)
def cuMemPoolSetAccess(pool, map, count): ...
@dll.bind((Pointer(CUmemAccess_flags), CUmemoryPool, Pointer(CUmemLocation),), CUresult)
def cuMemPoolGetAccess(flags, memPool, location): ...
@dll.bind((Pointer(CUmemoryPool), Pointer(CUmemPoolProps),), CUresult)
def cuMemPoolCreate(pool, poolProps): ...
@dll.bind((CUmemoryPool,), CUresult)
def cuMemPoolDestroy(pool): ...
@dll.bind((Pointer(CUdeviceptr), size_t, CUmemoryPool, CUstream,), CUresult)
def cuMemAllocFromPoolAsync_ptsz(dptr, bytesize, pool, hStream): ...
@dll.bind((ctypes.c_void_p, CUmemoryPool, CUmemAllocationHandleType, ctypes.c_uint64,), CUresult)
def cuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags): ...
@dll.bind((Pointer(CUmemoryPool), ctypes.c_void_p, CUmemAllocationHandleType, ctypes.c_uint64,), CUresult)
def cuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags): ...
@dll.bind((Pointer(CUmemPoolPtrExportData), CUdeviceptr,), CUresult)
def cuMemPoolExportPointer(shareData_out, ptr): ...
@dll.bind((Pointer(CUdeviceptr), CUmemoryPool, Pointer(CUmemPoolPtrExportData),), CUresult)
def cuMemPoolImportPointer(ptr_out, pool, shareData): ...
@dll.bind((ctypes.c_void_p, CUpointer_attribute, CUdeviceptr,), CUresult)
def cuPointerGetAttribute(data, attribute, ptr): ...
@dll.bind((CUdeviceptr, size_t, CUdevice, CUstream,), CUresult)
def cuMemPrefetchAsync_ptsz(devPtr, count, dstDevice, hStream): ...
@dll.bind((CUdeviceptr, size_t, CUmem_advise, CUdevice,), CUresult)
def cuMemAdvise(devPtr, count, advice, device): ...
@dll.bind((ctypes.c_void_p, size_t, CUmem_range_attribute, CUdeviceptr, size_t,), CUresult)
def cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count): ...
@dll.bind((Pointer(ctypes.c_void_p), Pointer(size_t), Pointer(CUmem_range_attribute), size_t, CUdeviceptr, size_t,), CUresult)
def cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count): ...
@dll.bind((ctypes.c_void_p, CUpointer_attribute, CUdeviceptr,), CUresult)
def cuPointerSetAttribute(value, attribute, ptr): ...
@dll.bind((ctypes.c_uint32, Pointer(CUpointer_attribute), Pointer(ctypes.c_void_p), CUdeviceptr,), CUresult)
def cuPointerGetAttributes(numAttributes, attributes, data, ptr): ...
@dll.bind((Pointer(CUstream), ctypes.c_uint32,), CUresult)
def cuStreamCreate(phStream, Flags): ...
@dll.bind((Pointer(CUstream), ctypes.c_uint32, ctypes.c_int32,), CUresult)
def cuStreamCreateWithPriority(phStream, flags, priority): ...
@dll.bind((CUstream, Pointer(ctypes.c_int32),), CUresult)
def cuStreamGetPriority_ptsz(hStream, priority): ...
@dll.bind((CUstream, Pointer(ctypes.c_uint32),), CUresult)
def cuStreamGetFlags_ptsz(hStream, flags): ...
@dll.bind((CUstream, Pointer(ctypes.c_uint64),), CUresult)
def cuStreamGetId_ptsz(hStream, streamId): ...
@dll.bind((CUstream, Pointer(CUcontext),), CUresult)
def cuStreamGetCtx_ptsz(hStream, pctx): ...
@dll.bind((CUstream, CUevent, ctypes.c_uint32,), CUresult)
def cuStreamWaitEvent_ptsz(hStream, hEvent, Flags): ...
@dll.bind((CUstream, CUstreamCallback, ctypes.c_void_p, ctypes.c_uint32,), CUresult)
def cuStreamAddCallback_ptsz(hStream, callback, userData, flags): ...
@dll.bind((CUstream, CUstreamCaptureMode,), CUresult)
def cuStreamBeginCapture_v2_ptsz(hStream, mode): ...
@dll.bind((Pointer(CUstreamCaptureMode),), CUresult)
def cuThreadExchangeStreamCaptureMode(mode): ...
@dll.bind((CUstream, Pointer(CUgraph),), CUresult)
def cuStreamEndCapture_ptsz(hStream, phGraph): ...
@dll.bind((CUstream, Pointer(CUstreamCaptureStatus),), CUresult)
def cuStreamIsCapturing_ptsz(hStream, captureStatus): ...
@dll.bind((CUstream, Pointer(CUstreamCaptureStatus), Pointer(cuuint64_t), Pointer(CUgraph), Pointer(Pointer(CUgraphNode)), Pointer(size_t),), CUresult)
def cuStreamGetCaptureInfo_v2_ptsz(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out): ...
@dll.bind((CUstream, Pointer(CUgraphNode), size_t, ctypes.c_uint32,), CUresult)
def cuStreamUpdateCaptureDependencies_ptsz(hStream, dependencies, numDependencies, flags): ...
@dll.bind((CUstream, CUdeviceptr, size_t, ctypes.c_uint32,), CUresult)
def cuStreamAttachMemAsync_ptsz(hStream, dptr, length, flags): ...
@dll.bind((CUstream,), CUresult)
def cuStreamQuery_ptsz(hStream): ...
@dll.bind((CUstream,), CUresult)
def cuStreamSynchronize_ptsz(hStream): ...
@dll.bind((CUstream,), CUresult)
def cuStreamDestroy_v2(hStream): ...
@dll.bind((CUstream, CUstream,), CUresult)
def cuStreamCopyAttributes_ptsz(dst, src): ...
@dll.bind((CUstream, CUstreamAttrID, Pointer(CUstreamAttrValue),), CUresult)
def cuStreamGetAttribute_ptsz(hStream, attr, value_out): ...
@dll.bind((CUstream, CUstreamAttrID, Pointer(CUstreamAttrValue),), CUresult)
def cuStreamSetAttribute_ptsz(hStream, attr, value): ...
@dll.bind((Pointer(CUevent), ctypes.c_uint32,), CUresult)
def cuEventCreate(phEvent, Flags): ...
@dll.bind((CUevent, CUstream,), CUresult)
def cuEventRecord_ptsz(hEvent, hStream): ...
@dll.bind((CUevent, CUstream, ctypes.c_uint32,), CUresult)
def cuEventRecordWithFlags_ptsz(hEvent, hStream, flags): ...
@dll.bind((CUevent,), CUresult)
def cuEventQuery(hEvent): ...
@dll.bind((CUevent,), CUresult)
def cuEventSynchronize(hEvent): ...
@dll.bind((CUevent,), CUresult)
def cuEventDestroy_v2(hEvent): ...
@dll.bind((Pointer(ctypes.c_float), CUevent, CUevent,), CUresult)
def cuEventElapsedTime(pMilliseconds, hStart, hEnd): ...
@dll.bind((Pointer(CUexternalMemory), Pointer(CUDA_EXTERNAL_MEMORY_HANDLE_DESC),), CUresult)
def cuImportExternalMemory(extMem_out, memHandleDesc): ...
@dll.bind((Pointer(CUdeviceptr), CUexternalMemory, Pointer(CUDA_EXTERNAL_MEMORY_BUFFER_DESC),), CUresult)
def cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc): ...
@dll.bind((Pointer(CUmipmappedArray), CUexternalMemory, Pointer(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC),), CUresult)
def cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc): ...
@dll.bind((CUexternalMemory,), CUresult)
def cuDestroyExternalMemory(extMem): ...
@dll.bind((Pointer(CUexternalSemaphore), Pointer(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC),), CUresult)
def cuImportExternalSemaphore(extSem_out, semHandleDesc): ...
@dll.bind((Pointer(CUexternalSemaphore), Pointer(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), ctypes.c_uint32, CUstream,), CUresult)
def cuSignalExternalSemaphoresAsync_ptsz(extSemArray, paramsArray, numExtSems, stream): ...
@dll.bind((Pointer(CUexternalSemaphore), Pointer(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS), ctypes.c_uint32, CUstream,), CUresult)
def cuWaitExternalSemaphoresAsync_ptsz(extSemArray, paramsArray, numExtSems, stream): ...
@dll.bind((CUexternalSemaphore,), CUresult)
def cuDestroyExternalSemaphore(extSem): ...
@dll.bind((CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32,), CUresult)
def cuStreamWaitValue32_v2_ptsz(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32,), CUresult)
def cuStreamWaitValue64_v2_ptsz(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32,), CUresult)
def cuStreamWriteValue32_v2_ptsz(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32,), CUresult)
def cuStreamWriteValue64_v2_ptsz(stream, addr, value, flags): ...
@dll.bind((CUstream, ctypes.c_uint32, Pointer(CUstreamBatchMemOpParams), ctypes.c_uint32,), CUresult)
def cuStreamBatchMemOp_v2_ptsz(stream, count, paramArray, flags): ...
@dll.bind((Pointer(ctypes.c_int32), CUfunction_attribute, CUfunction,), CUresult)
def cuFuncGetAttribute(pi, attrib, hfunc): ...
@dll.bind((CUfunction, CUfunction_attribute, ctypes.c_int32,), CUresult)
def cuFuncSetAttribute(hfunc, attrib, value): ...
@dll.bind((CUfunction, CUfunc_cache,), CUresult)
def cuFuncSetCacheConfig(hfunc, config): ...
@dll.bind((CUfunction, CUsharedconfig,), CUresult)
def cuFuncSetSharedMemConfig(hfunc, config): ...
@dll.bind((Pointer(CUmodule), CUfunction,), CUresult)
def cuFuncGetModule(hmod, hfunc): ...
@dll.bind((CUfunction, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, CUstream, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p),), CUresult)
def cuLaunchKernel_ptsz(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra): ...
@dll.bind((Pointer(CUlaunchConfig), CUfunction, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p),), CUresult)
def cuLaunchKernelEx_ptsz(config, f, kernelParams, extra): ...
@dll.bind((CUfunction, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, CUstream, Pointer(ctypes.c_void_p),), CUresult)
def cuLaunchCooperativeKernel_ptsz(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams): ...
@dll.bind((Pointer(CUDA_LAUNCH_PARAMS), ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags): ...
@dll.bind((CUstream, CUhostFn, ctypes.c_void_p,), CUresult)
def cuLaunchHostFunc_ptsz(hStream, fn, userData): ...
@dll.bind((CUfunction, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,), CUresult)
def cuFuncSetBlockShape(hfunc, x, y, z): ...
@dll.bind((CUfunction, ctypes.c_uint32,), CUresult)
def cuFuncSetSharedSize(hfunc, bytes): ...
@dll.bind((CUfunction, ctypes.c_uint32,), CUresult)
def cuParamSetSize(hfunc, numbytes): ...
@dll.bind((CUfunction, ctypes.c_int32, ctypes.c_uint32,), CUresult)
def cuParamSeti(hfunc, offset, value): ...
@dll.bind((CUfunction, ctypes.c_int32, ctypes.c_float,), CUresult)
def cuParamSetf(hfunc, offset, value): ...
@dll.bind((CUfunction, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32,), CUresult)
def cuParamSetv(hfunc, offset, ptr, numbytes): ...
@dll.bind((CUfunction,), CUresult)
def cuLaunch(f): ...
@dll.bind((CUfunction, ctypes.c_int32, ctypes.c_int32,), CUresult)
def cuLaunchGrid(f, grid_width, grid_height): ...
@dll.bind((CUfunction, ctypes.c_int32, ctypes.c_int32, CUstream,), CUresult)
def cuLaunchGridAsync(f, grid_width, grid_height, hStream): ...
@dll.bind((CUfunction, ctypes.c_int32, CUtexref,), CUresult)
def cuParamSetTexRef(hfunc, texunit, hTexRef): ...
@dll.bind((Pointer(CUgraph), ctypes.c_uint32,), CUresult)
def cuGraphCreate(phGraph, flags): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_KERNEL_NODE_PARAMS),), CUresult)
def cuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies, numDependencies, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_KERNEL_NODE_PARAMS),), CUresult)
def cuGraphKernelNodeGetParams_v2(hNode, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_KERNEL_NODE_PARAMS),), CUresult)
def cuGraphKernelNodeSetParams_v2(hNode, nodeParams): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_MEMCPY3D), CUcontext,), CUresult)
def cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx): ...
@dll.bind((CUgraphNode, Pointer(CUDA_MEMCPY3D),), CUresult)
def cuGraphMemcpyNodeGetParams(hNode, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_MEMCPY3D),), CUresult)
def cuGraphMemcpyNodeSetParams(hNode, nodeParams): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_MEMSET_NODE_PARAMS), CUcontext,), CUresult)
def cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx): ...
@dll.bind((CUgraphNode, Pointer(CUDA_MEMSET_NODE_PARAMS),), CUresult)
def cuGraphMemsetNodeGetParams(hNode, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_MEMSET_NODE_PARAMS),), CUresult)
def cuGraphMemsetNodeSetParams(hNode, nodeParams): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_HOST_NODE_PARAMS),), CUresult)
def cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_HOST_NODE_PARAMS),), CUresult)
def cuGraphHostNodeGetParams(hNode, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_HOST_NODE_PARAMS),), CUresult)
def cuGraphHostNodeSetParams(hNode, nodeParams): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, CUgraph,), CUresult)
def cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph): ...
@dll.bind((CUgraphNode, Pointer(CUgraph),), CUresult)
def cuGraphChildGraphNodeGetGraph(hNode, phGraph): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t,), CUresult)
def cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, CUevent,), CUresult)
def cuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event): ...
@dll.bind((CUgraphNode, Pointer(CUevent),), CUresult)
def cuGraphEventRecordNodeGetEvent(hNode, event_out): ...
@dll.bind((CUgraphNode, CUevent,), CUresult)
def cuGraphEventRecordNodeSetEvent(hNode, event): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, CUevent,), CUresult)
def cuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event): ...
@dll.bind((CUgraphNode, Pointer(CUevent),), CUresult)
def cuGraphEventWaitNodeGetEvent(hNode, event_out): ...
@dll.bind((CUgraphNode, CUevent,), CUresult)
def cuGraphEventWaitNodeSetEvent(hNode, event): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS),), CUresult)
def cuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS),), CUresult)
def cuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out): ...
@dll.bind((CUgraphNode, Pointer(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS),), CUresult)
def cuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_EXT_SEM_WAIT_NODE_PARAMS),), CUresult)
def cuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_EXT_SEM_WAIT_NODE_PARAMS),), CUresult)
def cuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out): ...
@dll.bind((CUgraphNode, Pointer(CUDA_EXT_SEM_WAIT_NODE_PARAMS),), CUresult)
def cuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_BATCH_MEM_OP_NODE_PARAMS),), CUresult)
def cuGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_BATCH_MEM_OP_NODE_PARAMS),), CUresult)
def cuGraphBatchMemOpNodeGetParams(hNode, nodeParams_out): ...
@dll.bind((CUgraphNode, Pointer(CUDA_BATCH_MEM_OP_NODE_PARAMS),), CUresult)
def cuGraphBatchMemOpNodeSetParams(hNode, nodeParams): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(CUDA_BATCH_MEM_OP_NODE_PARAMS),), CUresult)
def cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_MEM_ALLOC_NODE_PARAMS),), CUresult)
def cuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_MEM_ALLOC_NODE_PARAMS),), CUresult)
def cuGraphMemAllocNodeGetParams(hNode, params_out): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, CUdeviceptr,), CUresult)
def cuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr): ...
@dll.bind((CUgraphNode, Pointer(CUdeviceptr),), CUresult)
def cuGraphMemFreeNodeGetParams(hNode, dptr_out): ...
@dll.bind((CUdevice,), CUresult)
def cuDeviceGraphMemTrim(device): ...
@dll.bind((CUdevice, CUgraphMem_attribute, ctypes.c_void_p,), CUresult)
def cuDeviceGetGraphMemAttribute(device, attr, value): ...
@dll.bind((CUdevice, CUgraphMem_attribute, ctypes.c_void_p,), CUresult)
def cuDeviceSetGraphMemAttribute(device, attr, value): ...
@dll.bind((Pointer(CUgraph), CUgraph,), CUresult)
def cuGraphClone(phGraphClone, originalGraph): ...
@dll.bind((Pointer(CUgraphNode), CUgraphNode, CUgraph,), CUresult)
def cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph): ...
@dll.bind((CUgraphNode, Pointer(CUgraphNodeType),), CUresult)
def cuGraphNodeGetType(hNode, type): ...
@dll.bind((CUgraph, Pointer(CUgraphNode), Pointer(size_t),), CUresult)
def cuGraphGetNodes(hGraph, nodes, numNodes): ...
@dll.bind((CUgraph, Pointer(CUgraphNode), Pointer(size_t),), CUresult)
def cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes): ...
@dll.bind((CUgraph, Pointer(CUgraphNode), Pointer(CUgraphNode), Pointer(size_t),), CUresult)
def cuGraphGetEdges(hGraph, _nfrom, to, numEdges): ...
@dll.bind((CUgraphNode, Pointer(CUgraphNode), Pointer(size_t),), CUresult)
def cuGraphNodeGetDependencies(hNode, dependencies, numDependencies): ...
@dll.bind((CUgraphNode, Pointer(CUgraphNode), Pointer(size_t),), CUresult)
def cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes): ...
@dll.bind((CUgraph, Pointer(CUgraphNode), Pointer(CUgraphNode), size_t,), CUresult)
def cuGraphAddDependencies(hGraph, _nfrom, to, numDependencies): ...
@dll.bind((CUgraph, Pointer(CUgraphNode), Pointer(CUgraphNode), size_t,), CUresult)
def cuGraphRemoveDependencies(hGraph, _nfrom, to, numDependencies): ...
@dll.bind((CUgraphNode,), CUresult)
def cuGraphDestroyNode(hNode): ...
@dll.bind((Pointer(CUgraphExec), CUgraph, ctypes.c_uint64,), CUresult)
def cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags): ...
@dll.bind((Pointer(CUgraphExec), CUgraph, Pointer(CUDA_GRAPH_INSTANTIATE_PARAMS),), CUresult)
def cuGraphInstantiateWithParams_ptsz(phGraphExec, hGraph, instantiateParams): ...
@dll.bind((CUgraphExec, Pointer(cuuint64_t),), CUresult)
def cuGraphExecGetFlags(hGraphExec, flags): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(CUDA_KERNEL_NODE_PARAMS),), CUresult)
def cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, nodeParams): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(CUDA_MEMCPY3D), CUcontext,), CUresult)
def cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(CUDA_MEMSET_NODE_PARAMS), CUcontext,), CUresult)
def cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(CUDA_HOST_NODE_PARAMS),), CUresult)
def cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams): ...
@dll.bind((CUgraphExec, CUgraphNode, CUgraph,), CUresult)
def cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph): ...
@dll.bind((CUgraphExec, CUgraphNode, CUevent,), CUresult)
def cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event): ...
@dll.bind((CUgraphExec, CUgraphNode, CUevent,), CUresult)
def cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS),), CUresult)
def cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(CUDA_EXT_SEM_WAIT_NODE_PARAMS),), CUresult)
def cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams): ...
@dll.bind((CUgraphExec, CUgraphNode, ctypes.c_uint32,), CUresult)
def cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(ctypes.c_uint32),), CUresult)
def cuGraphNodeGetEnabled(hGraphExec, hNode, isEnabled): ...
@dll.bind((CUgraphExec, CUstream,), CUresult)
def cuGraphUpload_ptsz(hGraphExec, hStream): ...
@dll.bind((CUgraphExec, CUstream,), CUresult)
def cuGraphLaunch_ptsz(hGraphExec, hStream): ...
@dll.bind((CUgraphExec,), CUresult)
def cuGraphExecDestroy(hGraphExec): ...
@dll.bind((CUgraph,), CUresult)
def cuGraphDestroy(hGraph): ...
@dll.bind((CUgraphExec, CUgraph, Pointer(CUgraphExecUpdateResultInfo),), CUresult)
def cuGraphExecUpdate_v2(hGraphExec, hGraph, resultInfo): ...
@dll.bind((CUgraphNode, CUgraphNode,), CUresult)
def cuGraphKernelNodeCopyAttributes(dst, src): ...
@dll.bind((CUgraphNode, CUkernelNodeAttrID, Pointer(CUkernelNodeAttrValue),), CUresult)
def cuGraphKernelNodeGetAttribute(hNode, attr, value_out): ...
@dll.bind((CUgraphNode, CUkernelNodeAttrID, Pointer(CUkernelNodeAttrValue),), CUresult)
def cuGraphKernelNodeSetAttribute(hNode, attr, value): ...
@dll.bind((CUgraph, Pointer(ctypes.c_char), ctypes.c_uint32,), CUresult)
def cuGraphDebugDotPrint(hGraph, path, flags): ...
@dll.bind((Pointer(CUuserObject), ctypes.c_void_p, CUhostFn, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags): ...
@dll.bind((CUuserObject, ctypes.c_uint32,), CUresult)
def cuUserObjectRetain(object, count): ...
@dll.bind((CUuserObject, ctypes.c_uint32,), CUresult)
def cuUserObjectRelease(object, count): ...
@dll.bind((CUgraph, CUuserObject, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuGraphRetainUserObject(graph, object, count, flags): ...
@dll.bind((CUgraph, CUuserObject, ctypes.c_uint32,), CUresult)
def cuGraphReleaseUserObject(graph, object, count): ...
@dll.bind((Pointer(ctypes.c_int32), CUfunction, ctypes.c_int32, size_t,), CUresult)
def cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize): ...
@dll.bind((Pointer(ctypes.c_int32), CUfunction, ctypes.c_int32, size_t, ctypes.c_uint32,), CUresult)
def cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), CUfunction, CUoccupancyB2DSize, size_t, ctypes.c_int32,), CUresult)
def cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit): ...
@dll.bind((Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), CUfunction, CUoccupancyB2DSize, size_t, ctypes.c_int32, ctypes.c_uint32,), CUresult)
def cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags): ...
@dll.bind((Pointer(size_t), CUfunction, ctypes.c_int32, ctypes.c_int32,), CUresult)
def cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize): ...
@dll.bind((Pointer(ctypes.c_int32), CUfunction, Pointer(CUlaunchConfig),), CUresult)
def cuOccupancyMaxPotentialClusterSize(clusterSize, func, config): ...
@dll.bind((Pointer(ctypes.c_int32), CUfunction, Pointer(CUlaunchConfig),), CUresult)
def cuOccupancyMaxActiveClusters(numClusters, func, config): ...
@dll.bind((CUtexref, CUarray, ctypes.c_uint32,), CUresult)
def cuTexRefSetArray(hTexRef, hArray, Flags): ...
@dll.bind((CUtexref, CUmipmappedArray, ctypes.c_uint32,), CUresult)
def cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags): ...
@dll.bind((Pointer(size_t), CUtexref, CUdeviceptr, size_t,), CUresult)
def cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes): ...
@dll.bind((CUtexref, Pointer(CUDA_ARRAY_DESCRIPTOR), CUdeviceptr, size_t,), CUresult)
def cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch): ...
@dll.bind((CUtexref, CUarray_format, ctypes.c_int32,), CUresult)
def cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents): ...
@dll.bind((CUtexref, ctypes.c_int32, CUaddress_mode,), CUresult)
def cuTexRefSetAddressMode(hTexRef, dim, am): ...
@dll.bind((CUtexref, CUfilter_mode,), CUresult)
def cuTexRefSetFilterMode(hTexRef, fm): ...
@dll.bind((CUtexref, CUfilter_mode,), CUresult)
def cuTexRefSetMipmapFilterMode(hTexRef, fm): ...
@dll.bind((CUtexref, ctypes.c_float,), CUresult)
def cuTexRefSetMipmapLevelBias(hTexRef, bias): ...
@dll.bind((CUtexref, ctypes.c_float, ctypes.c_float,), CUresult)
def cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp): ...
@dll.bind((CUtexref, ctypes.c_uint32,), CUresult)
def cuTexRefSetMaxAnisotropy(hTexRef, maxAniso): ...
@dll.bind((CUtexref, Pointer(ctypes.c_float),), CUresult)
def cuTexRefSetBorderColor(hTexRef, pBorderColor): ...
@dll.bind((CUtexref, ctypes.c_uint32,), CUresult)
def cuTexRefSetFlags(hTexRef, Flags): ...
@dll.bind((Pointer(CUdeviceptr), CUtexref,), CUresult)
def cuTexRefGetAddress_v2(pdptr, hTexRef): ...
@dll.bind((Pointer(CUarray), CUtexref,), CUresult)
def cuTexRefGetArray(phArray, hTexRef): ...
@dll.bind((Pointer(CUmipmappedArray), CUtexref,), CUresult)
def cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef): ...
@dll.bind((Pointer(CUaddress_mode), CUtexref, ctypes.c_int32,), CUresult)
def cuTexRefGetAddressMode(pam, hTexRef, dim): ...
@dll.bind((Pointer(CUfilter_mode), CUtexref,), CUresult)
def cuTexRefGetFilterMode(pfm, hTexRef): ...
@dll.bind((Pointer(CUarray_format), Pointer(ctypes.c_int32), CUtexref,), CUresult)
def cuTexRefGetFormat(pFormat, pNumChannels, hTexRef): ...
@dll.bind((Pointer(CUfilter_mode), CUtexref,), CUresult)
def cuTexRefGetMipmapFilterMode(pfm, hTexRef): ...
@dll.bind((Pointer(ctypes.c_float), CUtexref,), CUresult)
def cuTexRefGetMipmapLevelBias(pbias, hTexRef): ...
@dll.bind((Pointer(ctypes.c_float), Pointer(ctypes.c_float), CUtexref,), CUresult)
def cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef): ...
@dll.bind((Pointer(ctypes.c_int32), CUtexref,), CUresult)
def cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef): ...
@dll.bind((Pointer(ctypes.c_float), CUtexref,), CUresult)
def cuTexRefGetBorderColor(pBorderColor, hTexRef): ...
@dll.bind((Pointer(ctypes.c_uint32), CUtexref,), CUresult)
def cuTexRefGetFlags(pFlags, hTexRef): ...
@dll.bind((Pointer(CUtexref),), CUresult)
def cuTexRefCreate(pTexRef): ...
@dll.bind((CUtexref,), CUresult)
def cuTexRefDestroy(hTexRef): ...
@dll.bind((CUsurfref, CUarray, ctypes.c_uint32,), CUresult)
def cuSurfRefSetArray(hSurfRef, hArray, Flags): ...
@dll.bind((Pointer(CUarray), CUsurfref,), CUresult)
def cuSurfRefGetArray(phArray, hSurfRef): ...
@dll.bind((Pointer(CUtexObject), Pointer(CUDA_RESOURCE_DESC), Pointer(CUDA_TEXTURE_DESC), Pointer(CUDA_RESOURCE_VIEW_DESC),), CUresult)
def cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc): ...
@dll.bind((CUtexObject,), CUresult)
def cuTexObjectDestroy(texObject): ...
@dll.bind((Pointer(CUDA_RESOURCE_DESC), CUtexObject,), CUresult)
def cuTexObjectGetResourceDesc(pResDesc, texObject): ...
@dll.bind((Pointer(CUDA_TEXTURE_DESC), CUtexObject,), CUresult)
def cuTexObjectGetTextureDesc(pTexDesc, texObject): ...
@dll.bind((Pointer(CUDA_RESOURCE_VIEW_DESC), CUtexObject,), CUresult)
def cuTexObjectGetResourceViewDesc(pResViewDesc, texObject): ...
@dll.bind((Pointer(CUsurfObject), Pointer(CUDA_RESOURCE_DESC),), CUresult)
def cuSurfObjectCreate(pSurfObject, pResDesc): ...
@dll.bind((CUsurfObject,), CUresult)
def cuSurfObjectDestroy(surfObject): ...
@dll.bind((Pointer(CUDA_RESOURCE_DESC), CUsurfObject,), CUresult)
def cuSurfObjectGetResourceDesc(pResDesc, surfObject): ...
@dll.bind((Pointer(CUtensorMap), CUtensorMapDataType, cuuint32_t, ctypes.c_void_p, Pointer(cuuint64_t), Pointer(cuuint64_t), Pointer(cuuint32_t), Pointer(cuuint32_t), CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill,), CUresult)
def cuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill): ...
@dll.bind((Pointer(CUtensorMap), CUtensorMapDataType, cuuint32_t, ctypes.c_void_p, Pointer(cuuint64_t), Pointer(cuuint64_t), Pointer(ctypes.c_int32), Pointer(ctypes.c_int32), cuuint32_t, cuuint32_t, Pointer(cuuint32_t), CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill,), CUresult)
def cuTensorMapEncodeIm2col(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill): ...
@dll.bind((Pointer(CUtensorMap), ctypes.c_void_p,), CUresult)
def cuTensorMapReplaceAddress(tensorMap, globalAddress): ...
@dll.bind((Pointer(ctypes.c_int32), CUdevice, CUdevice,), CUresult)
def cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev): ...
@dll.bind((CUcontext, ctypes.c_uint32,), CUresult)
def cuCtxEnablePeerAccess(peerContext, Flags): ...
@dll.bind((CUcontext,), CUresult)
def cuCtxDisablePeerAccess(peerContext): ...
@dll.bind((Pointer(ctypes.c_int32), CUdevice_P2PAttribute, CUdevice, CUdevice,), CUresult)
def cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice): ...
@dll.bind((CUgraphicsResource,), CUresult)
def cuGraphicsUnregisterResource(resource): ...
@dll.bind((Pointer(CUarray), CUgraphicsResource, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel): ...
@dll.bind((Pointer(CUmipmappedArray), CUgraphicsResource,), CUresult)
def cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource): ...
@dll.bind((Pointer(CUdeviceptr), Pointer(size_t), CUgraphicsResource,), CUresult)
def cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource): ...
@dll.bind((CUgraphicsResource, ctypes.c_uint32,), CUresult)
def cuGraphicsResourceSetMapFlags_v2(resource, flags): ...
@dll.bind((ctypes.c_uint32, Pointer(CUgraphicsResource), CUstream,), CUresult)
def cuGraphicsMapResources_ptsz(count, resources, hStream): ...
@dll.bind((ctypes.c_uint32, Pointer(CUgraphicsResource), CUstream,), CUresult)
def cuGraphicsUnmapResources_ptsz(count, resources, hStream): ...
@dll.bind((Pointer(ctypes.c_char), Pointer(ctypes.c_void_p), ctypes.c_int32, cuuint64_t, Pointer(CUdriverProcAddressQueryResult),), CUresult)
def cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus): ...
@dll.bind((Pointer(ctypes.c_void_p), Pointer(CUuuid),), CUresult)
def cuGetExportTable(ppExportTable, pExportTableId): ...
@dll.bind((ctypes.c_void_p, size_t, ctypes.c_uint32,), CUresult)
def cuMemHostRegister(p, bytesize, Flags): ...
@dll.bind((CUgraphicsResource, ctypes.c_uint32,), CUresult)
def cuGraphicsResourceSetMapFlags(resource, flags): ...
@dll.bind((ctypes.c_uint32, Pointer(CUjit_option), Pointer(ctypes.c_void_p), Pointer(CUlinkState),), CUresult)
def cuLinkCreate(numOptions, options, optionValues, stateOut): ...
@dll.bind((CUlinkState, CUjitInputType, ctypes.c_void_p, size_t, Pointer(ctypes.c_char), ctypes.c_uint32, Pointer(CUjit_option), Pointer(ctypes.c_void_p),), CUresult)
def cuLinkAddData(state, type, data, size, name, numOptions, options, optionValues): ...
@dll.bind((CUlinkState, CUjitInputType, Pointer(ctypes.c_char), ctypes.c_uint32, Pointer(CUjit_option), Pointer(ctypes.c_void_p),), CUresult)
def cuLinkAddFile(state, type, path, numOptions, options, optionValues): ...
@dll.bind((CUtexref, Pointer(CUDA_ARRAY_DESCRIPTOR), CUdeviceptr, size_t,), CUresult)
def cuTexRefSetAddress2D_v2(hTexRef, desc, dptr, Pitch): ...
CUdeviceptr_v1 = ctypes.c_uint32
class struct_CUDA_MEMCPY2D_v1_st(Struct): pass
struct_CUDA_MEMCPY2D_v1_st.SIZE = 96
struct_CUDA_MEMCPY2D_v1_st._fields_ = ['srcXInBytes', 'srcY', 'srcMemoryType', 'srcHost', 'srcDevice', 'srcArray', 'srcPitch', 'dstXInBytes', 'dstY', 'dstMemoryType', 'dstHost', 'dstDevice', 'dstArray', 'dstPitch', 'WidthInBytes', 'Height']
setattr(struct_CUDA_MEMCPY2D_v1_st, 'srcXInBytes', field(0, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'srcY', field(4, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'srcMemoryType', field(8, CUmemorytype))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'srcHost', field(16, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'srcDevice', field(24, CUdeviceptr_v1))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'srcArray', field(32, CUarray))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'srcPitch', field(40, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'dstXInBytes', field(44, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'dstY', field(48, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'dstMemoryType', field(52, CUmemorytype))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'dstHost', field(56, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'dstDevice', field(64, CUdeviceptr_v1))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'dstArray', field(72, CUarray))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'dstPitch', field(80, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'WidthInBytes', field(84, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY2D_v1_st, 'Height', field(88, ctypes.c_uint32))
CUDA_MEMCPY2D_v1 = struct_CUDA_MEMCPY2D_v1_st
class struct_CUDA_MEMCPY3D_v1_st(Struct): pass
struct_CUDA_MEMCPY3D_v1_st.SIZE = 144
struct_CUDA_MEMCPY3D_v1_st._fields_ = ['srcXInBytes', 'srcY', 'srcZ', 'srcLOD', 'srcMemoryType', 'srcHost', 'srcDevice', 'srcArray', 'reserved0', 'srcPitch', 'srcHeight', 'dstXInBytes', 'dstY', 'dstZ', 'dstLOD', 'dstMemoryType', 'dstHost', 'dstDevice', 'dstArray', 'reserved1', 'dstPitch', 'dstHeight', 'WidthInBytes', 'Height', 'Depth']
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcXInBytes', field(0, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcY', field(4, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcZ', field(8, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcLOD', field(12, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcMemoryType', field(16, CUmemorytype))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcHost', field(24, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcDevice', field(32, CUdeviceptr_v1))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcArray', field(40, CUarray))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'reserved0', field(48, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcPitch', field(56, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'srcHeight', field(60, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstXInBytes', field(64, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstY', field(68, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstZ', field(72, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstLOD', field(76, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstMemoryType', field(80, CUmemorytype))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstHost', field(88, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstDevice', field(96, CUdeviceptr_v1))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstArray', field(104, CUarray))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'reserved1', field(112, ctypes.c_void_p))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstPitch', field(120, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'dstHeight', field(124, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'WidthInBytes', field(128, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'Height', field(132, ctypes.c_uint32))
setattr(struct_CUDA_MEMCPY3D_v1_st, 'Depth', field(136, ctypes.c_uint32))
CUDA_MEMCPY3D_v1 = struct_CUDA_MEMCPY3D_v1_st
class struct_CUDA_ARRAY_DESCRIPTOR_v1_st(Struct): pass
struct_CUDA_ARRAY_DESCRIPTOR_v1_st.SIZE = 16
struct_CUDA_ARRAY_DESCRIPTOR_v1_st._fields_ = ['Width', 'Height', 'Format', 'NumChannels']
setattr(struct_CUDA_ARRAY_DESCRIPTOR_v1_st, 'Width', field(0, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY_DESCRIPTOR_v1_st, 'Height', field(4, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY_DESCRIPTOR_v1_st, 'Format', field(8, CUarray_format))
setattr(struct_CUDA_ARRAY_DESCRIPTOR_v1_st, 'NumChannels', field(12, ctypes.c_uint32))
CUDA_ARRAY_DESCRIPTOR_v1 = struct_CUDA_ARRAY_DESCRIPTOR_v1_st
class struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st(Struct): pass
struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st.SIZE = 24
struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st._fields_ = ['Width', 'Height', 'Depth', 'Format', 'NumChannels', 'Flags']
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st, 'Width', field(0, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st, 'Height', field(4, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st, 'Depth', field(8, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st, 'Format', field(12, CUarray_format))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st, 'NumChannels', field(16, ctypes.c_uint32))
setattr(struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st, 'Flags', field(20, ctypes.c_uint32))
CUDA_ARRAY3D_DESCRIPTOR_v1 = struct_CUDA_ARRAY3D_DESCRIPTOR_v1_st
@dll.bind((Pointer(ctypes.c_uint32), CUdevice,), CUresult)
def cuDeviceTotalMem(bytes, dev): ...
@dll.bind((Pointer(CUcontext), ctypes.c_uint32, CUdevice,), CUresult)
def cuCtxCreate(pctx, flags, dev): ...
@dll.bind((Pointer(CUdeviceptr_v1), Pointer(ctypes.c_uint32), CUmodule, Pointer(ctypes.c_char),), CUresult)
def cuModuleGetGlobal(dptr, bytes, hmod, name): ...
@dll.bind((Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32),), CUresult)
def cuMemGetInfo(free, total): ...
@dll.bind((Pointer(CUdeviceptr_v1), ctypes.c_uint32,), CUresult)
def cuMemAlloc(dptr, bytesize): ...
@dll.bind((Pointer(CUdeviceptr_v1), Pointer(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes): ...
@dll.bind((CUdeviceptr_v1,), CUresult)
def cuMemFree(dptr): ...
@dll.bind((Pointer(CUdeviceptr_v1), Pointer(ctypes.c_uint32), CUdeviceptr_v1,), CUresult)
def cuMemGetAddressRange(pbase, psize, dptr): ...
@dll.bind((Pointer(ctypes.c_void_p), ctypes.c_uint32,), CUresult)
def cuMemAllocHost(pp, bytesize): ...
@dll.bind((Pointer(CUdeviceptr_v1), ctypes.c_void_p, ctypes.c_uint32,), CUresult)
def cuMemHostGetDevicePointer(pdptr, p, Flags): ...
@dll.bind((CUdeviceptr_v1, ctypes.c_void_p, ctypes.c_uint32,), CUresult)
def cuMemcpyHtoD(dstDevice, srcHost, ByteCount): ...
@dll.bind((ctypes.c_void_p, CUdeviceptr_v1, ctypes.c_uint32,), CUresult)
def cuMemcpyDtoH(dstHost, srcDevice, ByteCount): ...
@dll.bind((CUdeviceptr_v1, CUdeviceptr_v1, ctypes.c_uint32,), CUresult)
def cuMemcpyDtoD(dstDevice, srcDevice, ByteCount): ...
@dll.bind((CUarray, ctypes.c_uint32, CUdeviceptr_v1, ctypes.c_uint32,), CUresult)
def cuMemcpyDtoA(dstArray, dstOffset, srcDevice, ByteCount): ...
@dll.bind((CUdeviceptr_v1, CUarray, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuMemcpyAtoD(dstDevice, srcArray, srcOffset, ByteCount): ...
@dll.bind((CUarray, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,), CUresult)
def cuMemcpyHtoA(dstArray, dstOffset, srcHost, ByteCount): ...
@dll.bind((ctypes.c_void_p, CUarray, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuMemcpyAtoH(dstHost, srcArray, srcOffset, ByteCount): ...
@dll.bind((CUarray, ctypes.c_uint32, CUarray, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuMemcpyAtoA(dstArray, dstOffset, srcArray, srcOffset, ByteCount): ...
@dll.bind((CUarray, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, CUstream,), CUresult)
def cuMemcpyHtoAAsync(dstArray, dstOffset, srcHost, ByteCount, hStream): ...
@dll.bind((ctypes.c_void_p, CUarray, ctypes.c_uint32, ctypes.c_uint32, CUstream,), CUresult)
def cuMemcpyAtoHAsync(dstHost, srcArray, srcOffset, ByteCount, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY2D_v1),), CUresult)
def cuMemcpy2D(pCopy): ...
@dll.bind((Pointer(CUDA_MEMCPY2D_v1),), CUresult)
def cuMemcpy2DUnaligned(pCopy): ...
@dll.bind((Pointer(CUDA_MEMCPY3D_v1),), CUresult)
def cuMemcpy3D(pCopy): ...
@dll.bind((CUdeviceptr_v1, ctypes.c_void_p, ctypes.c_uint32, CUstream,), CUresult)
def cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream): ...
@dll.bind((ctypes.c_void_p, CUdeviceptr_v1, ctypes.c_uint32, CUstream,), CUresult)
def cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream): ...
@dll.bind((CUdeviceptr_v1, CUdeviceptr_v1, ctypes.c_uint32, CUstream,), CUresult)
def cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY2D_v1), CUstream,), CUresult)
def cuMemcpy2DAsync(pCopy, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY3D_v1), CUstream,), CUresult)
def cuMemcpy3DAsync(pCopy, hStream): ...
@dll.bind((CUdeviceptr_v1, ctypes.c_ubyte, ctypes.c_uint32,), CUresult)
def cuMemsetD8(dstDevice, uc, N): ...
@dll.bind((CUdeviceptr_v1, ctypes.c_uint16, ctypes.c_uint32,), CUresult)
def cuMemsetD16(dstDevice, us, N): ...
@dll.bind((CUdeviceptr_v1, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuMemsetD32(dstDevice, ui, N): ...
@dll.bind((CUdeviceptr_v1, ctypes.c_uint32, ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuMemsetD2D8(dstDevice, dstPitch, uc, Width, Height): ...
@dll.bind((CUdeviceptr_v1, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuMemsetD2D16(dstDevice, dstPitch, us, Width, Height): ...
@dll.bind((CUdeviceptr_v1, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,), CUresult)
def cuMemsetD2D32(dstDevice, dstPitch, ui, Width, Height): ...
@dll.bind((Pointer(CUarray), Pointer(CUDA_ARRAY_DESCRIPTOR_v1),), CUresult)
def cuArrayCreate(pHandle, pAllocateArray): ...
@dll.bind((Pointer(CUDA_ARRAY_DESCRIPTOR_v1), CUarray,), CUresult)
def cuArrayGetDescriptor(pArrayDescriptor, hArray): ...
@dll.bind((Pointer(CUarray), Pointer(CUDA_ARRAY3D_DESCRIPTOR_v1),), CUresult)
def cuArray3DCreate(pHandle, pAllocateArray): ...
@dll.bind((Pointer(CUDA_ARRAY3D_DESCRIPTOR_v1), CUarray,), CUresult)
def cuArray3DGetDescriptor(pArrayDescriptor, hArray): ...
@dll.bind((Pointer(ctypes.c_uint32), CUtexref, CUdeviceptr_v1, ctypes.c_uint32,), CUresult)
def cuTexRefSetAddress(ByteOffset, hTexRef, dptr, bytes): ...
@dll.bind((CUtexref, Pointer(CUDA_ARRAY_DESCRIPTOR_v1), CUdeviceptr_v1, ctypes.c_uint32,), CUresult)
def cuTexRefSetAddress2D(hTexRef, desc, dptr, Pitch): ...
@dll.bind((Pointer(CUdeviceptr_v1), CUtexref,), CUresult)
def cuTexRefGetAddress(pdptr, hTexRef): ...
@dll.bind((Pointer(CUdeviceptr_v1), Pointer(ctypes.c_uint32), CUgraphicsResource,), CUresult)
def cuGraphicsResourceGetMappedPointer(pDevPtr, pSize, resource): ...
@dll.bind((CUcontext,), CUresult)
def cuCtxDestroy(ctx): ...
@dll.bind((Pointer(CUcontext),), CUresult)
def cuCtxPopCurrent(pctx): ...
@dll.bind((CUcontext,), CUresult)
def cuCtxPushCurrent(ctx): ...
@dll.bind((CUstream,), CUresult)
def cuStreamDestroy(hStream): ...
@dll.bind((CUevent,), CUresult)
def cuEventDestroy(hEvent): ...
@dll.bind((CUdevice,), CUresult)
def cuDevicePrimaryCtxRelease(dev): ...
@dll.bind((CUdevice,), CUresult)
def cuDevicePrimaryCtxReset(dev): ...
@dll.bind((CUdevice, ctypes.c_uint32,), CUresult)
def cuDevicePrimaryCtxSetFlags(dev, flags): ...
@dll.bind((CUdeviceptr, ctypes.c_void_p, size_t,), CUresult)
def cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount): ...
@dll.bind((ctypes.c_void_p, CUdeviceptr, size_t,), CUresult)
def cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount): ...
@dll.bind((CUdeviceptr, CUdeviceptr, size_t,), CUresult)
def cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount): ...
@dll.bind((CUarray, size_t, CUdeviceptr, size_t,), CUresult)
def cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount): ...
@dll.bind((CUdeviceptr, CUarray, size_t, size_t,), CUresult)
def cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount): ...
@dll.bind((CUarray, size_t, ctypes.c_void_p, size_t,), CUresult)
def cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount): ...
@dll.bind((ctypes.c_void_p, CUarray, size_t, size_t,), CUresult)
def cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount): ...
@dll.bind((CUarray, size_t, CUarray, size_t, size_t,), CUresult)
def cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount): ...
@dll.bind((CUarray, size_t, ctypes.c_void_p, size_t, CUstream,), CUresult)
def cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream): ...
@dll.bind((ctypes.c_void_p, CUarray, size_t, size_t, CUstream,), CUresult)
def cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY2D),), CUresult)
def cuMemcpy2D_v2(pCopy): ...
@dll.bind((Pointer(CUDA_MEMCPY2D),), CUresult)
def cuMemcpy2DUnaligned_v2(pCopy): ...
@dll.bind((Pointer(CUDA_MEMCPY3D),), CUresult)
def cuMemcpy3D_v2(pCopy): ...
@dll.bind((CUdeviceptr, ctypes.c_void_p, size_t, CUstream,), CUresult)
def cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream): ...
@dll.bind((ctypes.c_void_p, CUdeviceptr, size_t, CUstream,), CUresult)
def cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream): ...
@dll.bind((CUdeviceptr, CUdeviceptr, size_t, CUstream,), CUresult)
def cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY2D), CUstream,), CUresult)
def cuMemcpy2DAsync_v2(pCopy, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY3D), CUstream,), CUresult)
def cuMemcpy3DAsync_v2(pCopy, hStream): ...
@dll.bind((CUdeviceptr, ctypes.c_ubyte, size_t,), CUresult)
def cuMemsetD8_v2(dstDevice, uc, N): ...
@dll.bind((CUdeviceptr, ctypes.c_uint16, size_t,), CUresult)
def cuMemsetD16_v2(dstDevice, us, N): ...
@dll.bind((CUdeviceptr, ctypes.c_uint32, size_t,), CUresult)
def cuMemsetD32_v2(dstDevice, ui, N): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t,), CUresult)
def cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_uint16, size_t, size_t,), CUresult)
def cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_uint32, size_t, size_t,), CUresult)
def cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height): ...
@dll.bind((CUdeviceptr, CUdeviceptr, size_t,), CUresult)
def cuMemcpy(dst, src, ByteCount): ...
@dll.bind((CUdeviceptr, CUdeviceptr, size_t, CUstream,), CUresult)
def cuMemcpyAsync(dst, src, ByteCount, hStream): ...
@dll.bind((CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t,), CUresult)
def cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount): ...
@dll.bind((CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream,), CUresult)
def cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream): ...
@dll.bind((Pointer(CUDA_MEMCPY3D_PEER),), CUresult)
def cuMemcpy3DPeer(pCopy): ...
@dll.bind((Pointer(CUDA_MEMCPY3D_PEER), CUstream,), CUresult)
def cuMemcpy3DPeerAsync(pCopy, hStream): ...
@dll.bind((CUdeviceptr, ctypes.c_ubyte, size_t, CUstream,), CUresult)
def cuMemsetD8Async(dstDevice, uc, N, hStream): ...
@dll.bind((CUdeviceptr, ctypes.c_uint16, size_t, CUstream,), CUresult)
def cuMemsetD16Async(dstDevice, us, N, hStream): ...
@dll.bind((CUdeviceptr, ctypes.c_uint32, size_t, CUstream,), CUresult)
def cuMemsetD32Async(dstDevice, ui, N, hStream): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t, CUstream,), CUresult)
def cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_uint16, size_t, size_t, CUstream,), CUresult)
def cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream): ...
@dll.bind((CUdeviceptr, size_t, ctypes.c_uint32, size_t, size_t, CUstream,), CUresult)
def cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream): ...
@dll.bind((CUstream, Pointer(ctypes.c_int32),), CUresult)
def cuStreamGetPriority(hStream, priority): ...
@dll.bind((CUstream, Pointer(ctypes.c_uint64),), CUresult)
def cuStreamGetId(hStream, streamId): ...
@dll.bind((CUstream, Pointer(ctypes.c_uint32),), CUresult)
def cuStreamGetFlags(hStream, flags): ...
@dll.bind((CUstream, Pointer(CUcontext),), CUresult)
def cuStreamGetCtx(hStream, pctx): ...
@dll.bind((CUstream, CUevent, ctypes.c_uint32,), CUresult)
def cuStreamWaitEvent(hStream, hEvent, Flags): ...
@dll.bind((CUstream, CUstreamCallback, ctypes.c_void_p, ctypes.c_uint32,), CUresult)
def cuStreamAddCallback(hStream, callback, userData, flags): ...
@dll.bind((CUstream, CUdeviceptr, size_t, ctypes.c_uint32,), CUresult)
def cuStreamAttachMemAsync(hStream, dptr, length, flags): ...
@dll.bind((CUstream,), CUresult)
def cuStreamQuery(hStream): ...
@dll.bind((CUstream,), CUresult)
def cuStreamSynchronize(hStream): ...
@dll.bind((CUevent, CUstream,), CUresult)
def cuEventRecord(hEvent, hStream): ...
@dll.bind((CUevent, CUstream, ctypes.c_uint32,), CUresult)
def cuEventRecordWithFlags(hEvent, hStream, flags): ...
@dll.bind((CUfunction, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, CUstream, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p),), CUresult)
def cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra): ...
@dll.bind((Pointer(CUlaunchConfig), CUfunction, Pointer(ctypes.c_void_p), Pointer(ctypes.c_void_p),), CUresult)
def cuLaunchKernelEx(config, f, kernelParams, extra): ...
@dll.bind((CUstream, CUhostFn, ctypes.c_void_p,), CUresult)
def cuLaunchHostFunc(hStream, fn, userData): ...
@dll.bind((ctypes.c_uint32, Pointer(CUgraphicsResource), CUstream,), CUresult)
def cuGraphicsMapResources(count, resources, hStream): ...
@dll.bind((ctypes.c_uint32, Pointer(CUgraphicsResource), CUstream,), CUresult)
def cuGraphicsUnmapResources(count, resources, hStream): ...
@dll.bind((CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32,), CUresult)
def cuStreamWriteValue32(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32,), CUresult)
def cuStreamWaitValue32(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32,), CUresult)
def cuStreamWriteValue64(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32,), CUresult)
def cuStreamWaitValue64(stream, addr, value, flags): ...
@dll.bind((CUstream, ctypes.c_uint32, Pointer(CUstreamBatchMemOpParams), ctypes.c_uint32,), CUresult)
def cuStreamBatchMemOp(stream, count, paramArray, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32,), CUresult)
def cuStreamWriteValue32_ptsz(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32,), CUresult)
def cuStreamWaitValue32_ptsz(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32,), CUresult)
def cuStreamWriteValue64_ptsz(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32,), CUresult)
def cuStreamWaitValue64_ptsz(stream, addr, value, flags): ...
@dll.bind((CUstream, ctypes.c_uint32, Pointer(CUstreamBatchMemOpParams), ctypes.c_uint32,), CUresult)
def cuStreamBatchMemOp_ptsz(stream, count, paramArray, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32,), CUresult)
def cuStreamWriteValue32_v2(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32,), CUresult)
def cuStreamWaitValue32_v2(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32,), CUresult)
def cuStreamWriteValue64_v2(stream, addr, value, flags): ...
@dll.bind((CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32,), CUresult)
def cuStreamWaitValue64_v2(stream, addr, value, flags): ...
@dll.bind((CUstream, ctypes.c_uint32, Pointer(CUstreamBatchMemOpParams), ctypes.c_uint32,), CUresult)
def cuStreamBatchMemOp_v2(stream, count, paramArray, flags): ...
@dll.bind((CUdeviceptr, size_t, CUdevice, CUstream,), CUresult)
def cuMemPrefetchAsync(devPtr, count, dstDevice, hStream): ...
@dll.bind((CUfunction, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, CUstream, Pointer(ctypes.c_void_p),), CUresult)
def cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams): ...
@dll.bind((Pointer(CUexternalSemaphore), Pointer(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS), ctypes.c_uint32, CUstream,), CUresult)
def cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream): ...
@dll.bind((Pointer(CUexternalSemaphore), Pointer(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS), ctypes.c_uint32, CUstream,), CUresult)
def cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream): ...
@dll.bind((CUstream,), CUresult)
def cuStreamBeginCapture(hStream): ...
@dll.bind((CUstream,), CUresult)
def cuStreamBeginCapture_ptsz(hStream): ...
@dll.bind((CUstream, CUstreamCaptureMode,), CUresult)
def cuStreamBeginCapture_v2(hStream, mode): ...
@dll.bind((CUstream, Pointer(CUgraph),), CUresult)
def cuStreamEndCapture(hStream, phGraph): ...
@dll.bind((CUstream, Pointer(CUstreamCaptureStatus),), CUresult)
def cuStreamIsCapturing(hStream, captureStatus): ...
@dll.bind((CUstream, Pointer(CUstreamCaptureStatus), Pointer(cuuint64_t),), CUresult)
def cuStreamGetCaptureInfo(hStream, captureStatus_out, id_out): ...
@dll.bind((CUstream, Pointer(CUstreamCaptureStatus), Pointer(cuuint64_t),), CUresult)
def cuStreamGetCaptureInfo_ptsz(hStream, captureStatus_out, id_out): ...
@dll.bind((CUstream, Pointer(CUstreamCaptureStatus), Pointer(cuuint64_t), Pointer(CUgraph), Pointer(Pointer(CUgraphNode)), Pointer(size_t),), CUresult)
def cuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out): ...
@dll.bind((Pointer(CUgraphNode), CUgraph, Pointer(CUgraphNode), size_t, Pointer(CUDA_KERNEL_NODE_PARAMS_v1),), CUresult)
def cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_KERNEL_NODE_PARAMS_v1),), CUresult)
def cuGraphKernelNodeGetParams(hNode, nodeParams): ...
@dll.bind((CUgraphNode, Pointer(CUDA_KERNEL_NODE_PARAMS_v1),), CUresult)
def cuGraphKernelNodeSetParams(hNode, nodeParams): ...
@dll.bind((CUgraphExec, CUgraphNode, Pointer(CUDA_KERNEL_NODE_PARAMS_v1),), CUresult)
def cuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams): ...
@dll.bind((Pointer(CUgraphExec), CUgraph, Pointer(CUDA_GRAPH_INSTANTIATE_PARAMS),), CUresult)
def cuGraphInstantiateWithParams(phGraphExec, hGraph, instantiateParams): ...
@dll.bind((CUgraphExec, CUgraph, Pointer(CUgraphNode), Pointer(CUgraphExecUpdateResult),), CUresult)
def cuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out): ...
@dll.bind((CUgraphExec, CUstream,), CUresult)
def cuGraphUpload(hGraph, hStream): ...
@dll.bind((CUgraphExec, CUstream,), CUresult)
def cuGraphLaunch(hGraph, hStream): ...
@dll.bind((CUstream, CUstream,), CUresult)
def cuStreamCopyAttributes(dstStream, srcStream): ...
@dll.bind((CUstream, CUstreamAttrID, Pointer(CUstreamAttrValue),), CUresult)
def cuStreamGetAttribute(hStream, attr, value): ...
@dll.bind((CUstream, CUstreamAttrID, Pointer(CUstreamAttrValue),), CUresult)
def cuStreamSetAttribute(hStream, attr, param): ...
@dll.bind((Pointer(CUdeviceptr), CUipcMemHandle, ctypes.c_uint32,), CUresult)
def cuIpcOpenMemHandle(pdptr, handle, Flags): ...
@dll.bind((Pointer(CUgraphExec), CUgraph, Pointer(CUgraphNode), Pointer(ctypes.c_char), size_t,), CUresult)
def cuGraphInstantiate(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize): ...
@dll.bind((Pointer(CUgraphExec), CUgraph, Pointer(CUgraphNode), Pointer(ctypes.c_char), size_t,), CUresult)
def cuGraphInstantiate_v2(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize): ...
@dll.bind((Pointer(CUarrayMapInfo), ctypes.c_uint32, CUstream,), CUresult)
def cuMemMapArrayAsync(mapInfoList, count, hStream): ...
@dll.bind((CUdeviceptr, CUstream,), CUresult)
def cuMemFreeAsync(dptr, hStream): ...
@dll.bind((Pointer(CUdeviceptr), size_t, CUstream,), CUresult)
def cuMemAllocAsync(dptr, bytesize, hStream): ...
@dll.bind((Pointer(CUdeviceptr), size_t, CUmemoryPool, CUstream,), CUresult)
def cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream): ...
@dll.bind((CUstream, Pointer(CUgraphNode), size_t, ctypes.c_uint32,), CUresult)
def cuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags): ...
@dll.bind((Pointer(ctypes.c_char), Pointer(ctypes.c_void_p), ctypes.c_int32, cuuint64_t,), CUresult)
def cuGetProcAddress(symbol, pfn, cudaVersion, flags): ...
