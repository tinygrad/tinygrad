# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
enum_kgsl_user_mem_type = CEnum(ctypes.c_uint32)
KGSL_USER_MEM_TYPE_PMEM = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_PMEM', 0)
KGSL_USER_MEM_TYPE_ASHMEM = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ASHMEM', 1)
KGSL_USER_MEM_TYPE_ADDR = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ADDR', 2)
KGSL_USER_MEM_TYPE_ION = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ION', 3)
KGSL_USER_MEM_TYPE_DMABUF = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_DMABUF', 3)
KGSL_USER_MEM_TYPE_MAX = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_MAX', 7)

enum_kgsl_ctx_reset_stat = CEnum(ctypes.c_uint32)
KGSL_CTX_STAT_NO_ERROR = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_NO_ERROR', 0)
KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT', 1)
KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT', 2)
KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT', 3)

enum_kgsl_deviceid = CEnum(ctypes.c_uint32)
KGSL_DEVICE_3D0 = enum_kgsl_deviceid.define('KGSL_DEVICE_3D0', 0)
KGSL_DEVICE_MAX = enum_kgsl_deviceid.define('KGSL_DEVICE_MAX', 1)

class struct_kgsl_devinfo(Struct): pass
struct_kgsl_devinfo.SIZE = 40
struct_kgsl_devinfo._fields_ = ['device_id', 'chip_id', 'mmu_enabled', 'gmem_gpubaseaddr', 'gpu_id', 'gmem_sizebytes']
setattr(struct_kgsl_devinfo, 'device_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_devinfo, 'chip_id', field(4, ctypes.c_uint32))
setattr(struct_kgsl_devinfo, 'mmu_enabled', field(8, ctypes.c_uint32))
setattr(struct_kgsl_devinfo, 'gmem_gpubaseaddr', field(16, ctypes.c_uint64))
setattr(struct_kgsl_devinfo, 'gpu_id', field(24, ctypes.c_uint32))
setattr(struct_kgsl_devinfo, 'gmem_sizebytes', field(32, ctypes.c_uint64))
class struct_kgsl_devmemstore(Struct): pass
struct_kgsl_devmemstore.SIZE = 40
struct_kgsl_devmemstore._fields_ = ['soptimestamp', 'sbz', 'eoptimestamp', 'sbz2', 'preempted', 'sbz3', 'ref_wait_ts', 'sbz4', 'current_context', 'sbz5']
setattr(struct_kgsl_devmemstore, 'soptimestamp', field(0, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'sbz', field(4, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'eoptimestamp', field(8, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'sbz2', field(12, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'preempted', field(16, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'sbz3', field(20, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'ref_wait_ts', field(24, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'sbz4', field(28, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'current_context', field(32, ctypes.c_uint32))
setattr(struct_kgsl_devmemstore, 'sbz5', field(36, ctypes.c_uint32))
enum_kgsl_timestamp_type = CEnum(ctypes.c_uint32)
KGSL_TIMESTAMP_CONSUMED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_CONSUMED', 1)
KGSL_TIMESTAMP_RETIRED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_RETIRED', 2)
KGSL_TIMESTAMP_QUEUED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_QUEUED', 3)

class struct_kgsl_shadowprop(Struct): pass
struct_kgsl_shadowprop.SIZE = 24
struct_kgsl_shadowprop._fields_ = ['gpuaddr', 'size', 'flags']
setattr(struct_kgsl_shadowprop, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_shadowprop, 'size', field(8, ctypes.c_uint64))
setattr(struct_kgsl_shadowprop, 'flags', field(16, ctypes.c_uint32))
class struct_kgsl_version(Struct): pass
struct_kgsl_version.SIZE = 16
struct_kgsl_version._fields_ = ['drv_major', 'drv_minor', 'dev_major', 'dev_minor']
setattr(struct_kgsl_version, 'drv_major', field(0, ctypes.c_uint32))
setattr(struct_kgsl_version, 'drv_minor', field(4, ctypes.c_uint32))
setattr(struct_kgsl_version, 'dev_major', field(8, ctypes.c_uint32))
setattr(struct_kgsl_version, 'dev_minor', field(12, ctypes.c_uint32))
class struct_kgsl_sp_generic_mem(Struct): pass
struct_kgsl_sp_generic_mem.SIZE = 16
struct_kgsl_sp_generic_mem._fields_ = ['local', 'pvt']
setattr(struct_kgsl_sp_generic_mem, 'local', field(0, ctypes.c_uint64))
setattr(struct_kgsl_sp_generic_mem, 'pvt', field(8, ctypes.c_uint64))
class struct_kgsl_ucode_version(Struct): pass
struct_kgsl_ucode_version.SIZE = 8
struct_kgsl_ucode_version._fields_ = ['pfp', 'pm4']
setattr(struct_kgsl_ucode_version, 'pfp', field(0, ctypes.c_uint32))
setattr(struct_kgsl_ucode_version, 'pm4', field(4, ctypes.c_uint32))
class struct_kgsl_gpmu_version(Struct): pass
struct_kgsl_gpmu_version.SIZE = 12
struct_kgsl_gpmu_version._fields_ = ['major', 'minor', 'features']
setattr(struct_kgsl_gpmu_version, 'major', field(0, ctypes.c_uint32))
setattr(struct_kgsl_gpmu_version, 'minor', field(4, ctypes.c_uint32))
setattr(struct_kgsl_gpmu_version, 'features', field(8, ctypes.c_uint32))
class struct_kgsl_ibdesc(Struct): pass
struct_kgsl_ibdesc.SIZE = 32
struct_kgsl_ibdesc._fields_ = ['gpuaddr', '__pad', 'sizedwords', 'ctrl']
setattr(struct_kgsl_ibdesc, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_ibdesc, '__pad', field(8, ctypes.c_uint64))
setattr(struct_kgsl_ibdesc, 'sizedwords', field(16, ctypes.c_uint64))
setattr(struct_kgsl_ibdesc, 'ctrl', field(24, ctypes.c_uint32))
class struct_kgsl_cmdbatch_profiling_buffer(Struct): pass
struct_kgsl_cmdbatch_profiling_buffer.SIZE = 40
struct_kgsl_cmdbatch_profiling_buffer._fields_ = ['wall_clock_s', 'wall_clock_ns', 'gpu_ticks_queued', 'gpu_ticks_submitted', 'gpu_ticks_retired']
setattr(struct_kgsl_cmdbatch_profiling_buffer, 'wall_clock_s', field(0, ctypes.c_uint64))
setattr(struct_kgsl_cmdbatch_profiling_buffer, 'wall_clock_ns', field(8, ctypes.c_uint64))
setattr(struct_kgsl_cmdbatch_profiling_buffer, 'gpu_ticks_queued', field(16, ctypes.c_uint64))
setattr(struct_kgsl_cmdbatch_profiling_buffer, 'gpu_ticks_submitted', field(24, ctypes.c_uint64))
setattr(struct_kgsl_cmdbatch_profiling_buffer, 'gpu_ticks_retired', field(32, ctypes.c_uint64))
class struct_kgsl_device_getproperty(Struct): pass
struct_kgsl_device_getproperty.SIZE = 24
struct_kgsl_device_getproperty._fields_ = ['type', 'value', 'sizebytes']
setattr(struct_kgsl_device_getproperty, 'type', field(0, ctypes.c_uint32))
setattr(struct_kgsl_device_getproperty, 'value', field(8, ctypes.c_void_p))
setattr(struct_kgsl_device_getproperty, 'sizebytes', field(16, ctypes.c_uint64))
class struct_kgsl_device_waittimestamp(Struct): pass
struct_kgsl_device_waittimestamp.SIZE = 8
struct_kgsl_device_waittimestamp._fields_ = ['timestamp', 'timeout']
setattr(struct_kgsl_device_waittimestamp, 'timestamp', field(0, ctypes.c_uint32))
setattr(struct_kgsl_device_waittimestamp, 'timeout', field(4, ctypes.c_uint32))
class struct_kgsl_device_waittimestamp_ctxtid(Struct): pass
struct_kgsl_device_waittimestamp_ctxtid.SIZE = 12
struct_kgsl_device_waittimestamp_ctxtid._fields_ = ['context_id', 'timestamp', 'timeout']
setattr(struct_kgsl_device_waittimestamp_ctxtid, 'context_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_device_waittimestamp_ctxtid, 'timestamp', field(4, ctypes.c_uint32))
setattr(struct_kgsl_device_waittimestamp_ctxtid, 'timeout', field(8, ctypes.c_uint32))
class struct_kgsl_ringbuffer_issueibcmds(Struct): pass
struct_kgsl_ringbuffer_issueibcmds.SIZE = 32
struct_kgsl_ringbuffer_issueibcmds._fields_ = ['drawctxt_id', 'ibdesc_addr', 'numibs', 'timestamp', 'flags']
setattr(struct_kgsl_ringbuffer_issueibcmds, 'drawctxt_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_ringbuffer_issueibcmds, 'ibdesc_addr', field(8, ctypes.c_uint64))
setattr(struct_kgsl_ringbuffer_issueibcmds, 'numibs', field(16, ctypes.c_uint32))
setattr(struct_kgsl_ringbuffer_issueibcmds, 'timestamp', field(20, ctypes.c_uint32))
setattr(struct_kgsl_ringbuffer_issueibcmds, 'flags', field(24, ctypes.c_uint32))
class struct_kgsl_cmdstream_readtimestamp(Struct): pass
struct_kgsl_cmdstream_readtimestamp.SIZE = 8
struct_kgsl_cmdstream_readtimestamp._fields_ = ['type', 'timestamp']
setattr(struct_kgsl_cmdstream_readtimestamp, 'type', field(0, ctypes.c_uint32))
setattr(struct_kgsl_cmdstream_readtimestamp, 'timestamp', field(4, ctypes.c_uint32))
class struct_kgsl_cmdstream_freememontimestamp(Struct): pass
struct_kgsl_cmdstream_freememontimestamp.SIZE = 16
struct_kgsl_cmdstream_freememontimestamp._fields_ = ['gpuaddr', 'type', 'timestamp']
setattr(struct_kgsl_cmdstream_freememontimestamp, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_cmdstream_freememontimestamp, 'type', field(8, ctypes.c_uint32))
setattr(struct_kgsl_cmdstream_freememontimestamp, 'timestamp', field(12, ctypes.c_uint32))
class struct_kgsl_drawctxt_create(Struct): pass
struct_kgsl_drawctxt_create.SIZE = 8
struct_kgsl_drawctxt_create._fields_ = ['flags', 'drawctxt_id']
setattr(struct_kgsl_drawctxt_create, 'flags', field(0, ctypes.c_uint32))
setattr(struct_kgsl_drawctxt_create, 'drawctxt_id', field(4, ctypes.c_uint32))
class struct_kgsl_drawctxt_destroy(Struct): pass
struct_kgsl_drawctxt_destroy.SIZE = 4
struct_kgsl_drawctxt_destroy._fields_ = ['drawctxt_id']
setattr(struct_kgsl_drawctxt_destroy, 'drawctxt_id', field(0, ctypes.c_uint32))
class struct_kgsl_map_user_mem(Struct): pass
struct_kgsl_map_user_mem.SIZE = 48
struct_kgsl_map_user_mem._fields_ = ['fd', 'gpuaddr', 'len', 'offset', 'hostptr', 'memtype', 'flags']
setattr(struct_kgsl_map_user_mem, 'fd', field(0, ctypes.c_int32))
setattr(struct_kgsl_map_user_mem, 'gpuaddr', field(8, ctypes.c_uint64))
setattr(struct_kgsl_map_user_mem, 'len', field(16, ctypes.c_uint64))
setattr(struct_kgsl_map_user_mem, 'offset', field(24, ctypes.c_uint64))
setattr(struct_kgsl_map_user_mem, 'hostptr', field(32, ctypes.c_uint64))
setattr(struct_kgsl_map_user_mem, 'memtype', field(40, enum_kgsl_user_mem_type))
setattr(struct_kgsl_map_user_mem, 'flags', field(44, ctypes.c_uint32))
class struct_kgsl_cmdstream_readtimestamp_ctxtid(Struct): pass
struct_kgsl_cmdstream_readtimestamp_ctxtid.SIZE = 12
struct_kgsl_cmdstream_readtimestamp_ctxtid._fields_ = ['context_id', 'type', 'timestamp']
setattr(struct_kgsl_cmdstream_readtimestamp_ctxtid, 'context_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_cmdstream_readtimestamp_ctxtid, 'type', field(4, ctypes.c_uint32))
setattr(struct_kgsl_cmdstream_readtimestamp_ctxtid, 'timestamp', field(8, ctypes.c_uint32))
class struct_kgsl_cmdstream_freememontimestamp_ctxtid(Struct): pass
struct_kgsl_cmdstream_freememontimestamp_ctxtid.SIZE = 24
struct_kgsl_cmdstream_freememontimestamp_ctxtid._fields_ = ['context_id', 'gpuaddr', 'type', 'timestamp']
setattr(struct_kgsl_cmdstream_freememontimestamp_ctxtid, 'context_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_cmdstream_freememontimestamp_ctxtid, 'gpuaddr', field(8, ctypes.c_uint64))
setattr(struct_kgsl_cmdstream_freememontimestamp_ctxtid, 'type', field(16, ctypes.c_uint32))
setattr(struct_kgsl_cmdstream_freememontimestamp_ctxtid, 'timestamp', field(20, ctypes.c_uint32))
class struct_kgsl_sharedmem_from_pmem(Struct): pass
struct_kgsl_sharedmem_from_pmem.SIZE = 24
struct_kgsl_sharedmem_from_pmem._fields_ = ['pmem_fd', 'gpuaddr', 'len', 'offset']
setattr(struct_kgsl_sharedmem_from_pmem, 'pmem_fd', field(0, ctypes.c_int32))
setattr(struct_kgsl_sharedmem_from_pmem, 'gpuaddr', field(8, ctypes.c_uint64))
setattr(struct_kgsl_sharedmem_from_pmem, 'len', field(16, ctypes.c_uint32))
setattr(struct_kgsl_sharedmem_from_pmem, 'offset', field(20, ctypes.c_uint32))
class struct_kgsl_sharedmem_free(Struct): pass
struct_kgsl_sharedmem_free.SIZE = 8
struct_kgsl_sharedmem_free._fields_ = ['gpuaddr']
setattr(struct_kgsl_sharedmem_free, 'gpuaddr', field(0, ctypes.c_uint64))
class struct_kgsl_cff_user_event(Struct): pass
struct_kgsl_cff_user_event.SIZE = 32
struct_kgsl_cff_user_event._fields_ = ['cff_opcode', 'op1', 'op2', 'op3', 'op4', 'op5', '__pad']
setattr(struct_kgsl_cff_user_event, 'cff_opcode', field(0, ctypes.c_ubyte))
setattr(struct_kgsl_cff_user_event, 'op1', field(4, ctypes.c_uint32))
setattr(struct_kgsl_cff_user_event, 'op2', field(8, ctypes.c_uint32))
setattr(struct_kgsl_cff_user_event, 'op3', field(12, ctypes.c_uint32))
setattr(struct_kgsl_cff_user_event, 'op4', field(16, ctypes.c_uint32))
setattr(struct_kgsl_cff_user_event, 'op5', field(20, ctypes.c_uint32))
setattr(struct_kgsl_cff_user_event, '__pad', field(24, Array(ctypes.c_uint32, 2)))
class struct_kgsl_gmem_desc(Struct): pass
struct_kgsl_gmem_desc.SIZE = 20
struct_kgsl_gmem_desc._fields_ = ['x', 'y', 'width', 'height', 'pitch']
setattr(struct_kgsl_gmem_desc, 'x', field(0, ctypes.c_uint32))
setattr(struct_kgsl_gmem_desc, 'y', field(4, ctypes.c_uint32))
setattr(struct_kgsl_gmem_desc, 'width', field(8, ctypes.c_uint32))
setattr(struct_kgsl_gmem_desc, 'height', field(12, ctypes.c_uint32))
setattr(struct_kgsl_gmem_desc, 'pitch', field(16, ctypes.c_uint32))
class struct_kgsl_buffer_desc(Struct): pass
struct_kgsl_buffer_desc.SIZE = 32
struct_kgsl_buffer_desc._fields_ = ['hostptr', 'gpuaddr', 'size', 'format', 'pitch', 'enabled']
setattr(struct_kgsl_buffer_desc, 'hostptr', field(0, ctypes.c_void_p))
setattr(struct_kgsl_buffer_desc, 'gpuaddr', field(8, ctypes.c_uint64))
setattr(struct_kgsl_buffer_desc, 'size', field(16, ctypes.c_int32))
setattr(struct_kgsl_buffer_desc, 'format', field(20, ctypes.c_uint32))
setattr(struct_kgsl_buffer_desc, 'pitch', field(24, ctypes.c_uint32))
setattr(struct_kgsl_buffer_desc, 'enabled', field(28, ctypes.c_uint32))
class struct_kgsl_bind_gmem_shadow(Struct): pass
struct_kgsl_bind_gmem_shadow.SIZE = 72
struct_kgsl_bind_gmem_shadow._fields_ = ['drawctxt_id', 'gmem_desc', 'shadow_x', 'shadow_y', 'shadow_buffer', 'buffer_id']
setattr(struct_kgsl_bind_gmem_shadow, 'drawctxt_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_bind_gmem_shadow, 'gmem_desc', field(4, struct_kgsl_gmem_desc))
setattr(struct_kgsl_bind_gmem_shadow, 'shadow_x', field(24, ctypes.c_uint32))
setattr(struct_kgsl_bind_gmem_shadow, 'shadow_y', field(28, ctypes.c_uint32))
setattr(struct_kgsl_bind_gmem_shadow, 'shadow_buffer', field(32, struct_kgsl_buffer_desc))
setattr(struct_kgsl_bind_gmem_shadow, 'buffer_id', field(64, ctypes.c_uint32))
class struct_kgsl_sharedmem_from_vmalloc(Struct): pass
struct_kgsl_sharedmem_from_vmalloc.SIZE = 16
struct_kgsl_sharedmem_from_vmalloc._fields_ = ['gpuaddr', 'hostptr', 'flags']
setattr(struct_kgsl_sharedmem_from_vmalloc, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_sharedmem_from_vmalloc, 'hostptr', field(8, ctypes.c_uint32))
setattr(struct_kgsl_sharedmem_from_vmalloc, 'flags', field(12, ctypes.c_uint32))
class struct_kgsl_drawctxt_set_bin_base_offset(Struct): pass
struct_kgsl_drawctxt_set_bin_base_offset.SIZE = 8
struct_kgsl_drawctxt_set_bin_base_offset._fields_ = ['drawctxt_id', 'offset']
setattr(struct_kgsl_drawctxt_set_bin_base_offset, 'drawctxt_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_drawctxt_set_bin_base_offset, 'offset', field(4, ctypes.c_uint32))
enum_kgsl_cmdwindow_type = CEnum(ctypes.c_uint32)
KGSL_CMDWINDOW_MIN = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MIN', 0)
KGSL_CMDWINDOW_2D = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_2D', 0)
KGSL_CMDWINDOW_3D = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_3D', 1)
KGSL_CMDWINDOW_MMU = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MMU', 2)
KGSL_CMDWINDOW_ARBITER = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_ARBITER', 255)
KGSL_CMDWINDOW_MAX = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MAX', 255)

class struct_kgsl_cmdwindow_write(Struct): pass
struct_kgsl_cmdwindow_write.SIZE = 12
struct_kgsl_cmdwindow_write._fields_ = ['target', 'addr', 'data']
setattr(struct_kgsl_cmdwindow_write, 'target', field(0, enum_kgsl_cmdwindow_type))
setattr(struct_kgsl_cmdwindow_write, 'addr', field(4, ctypes.c_uint32))
setattr(struct_kgsl_cmdwindow_write, 'data', field(8, ctypes.c_uint32))
class struct_kgsl_gpumem_alloc(Struct): pass
struct_kgsl_gpumem_alloc.SIZE = 24
struct_kgsl_gpumem_alloc._fields_ = ['gpuaddr', 'size', 'flags']
setattr(struct_kgsl_gpumem_alloc, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_alloc, 'size', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_alloc, 'flags', field(16, ctypes.c_uint32))
class struct_kgsl_cff_syncmem(Struct): pass
struct_kgsl_cff_syncmem.SIZE = 24
struct_kgsl_cff_syncmem._fields_ = ['gpuaddr', 'len', '__pad']
setattr(struct_kgsl_cff_syncmem, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_cff_syncmem, 'len', field(8, ctypes.c_uint64))
setattr(struct_kgsl_cff_syncmem, '__pad', field(16, Array(ctypes.c_uint32, 2)))
class struct_kgsl_timestamp_event(Struct): pass
struct_kgsl_timestamp_event.SIZE = 32
struct_kgsl_timestamp_event._fields_ = ['type', 'timestamp', 'context_id', 'priv', 'len']
setattr(struct_kgsl_timestamp_event, 'type', field(0, ctypes.c_int32))
setattr(struct_kgsl_timestamp_event, 'timestamp', field(4, ctypes.c_uint32))
setattr(struct_kgsl_timestamp_event, 'context_id', field(8, ctypes.c_uint32))
setattr(struct_kgsl_timestamp_event, 'priv', field(16, ctypes.c_void_p))
setattr(struct_kgsl_timestamp_event, 'len', field(24, ctypes.c_uint64))
class struct_kgsl_timestamp_event_genlock(Struct): pass
struct_kgsl_timestamp_event_genlock.SIZE = 4
struct_kgsl_timestamp_event_genlock._fields_ = ['handle']
setattr(struct_kgsl_timestamp_event_genlock, 'handle', field(0, ctypes.c_int32))
class struct_kgsl_timestamp_event_fence(Struct): pass
struct_kgsl_timestamp_event_fence.SIZE = 4
struct_kgsl_timestamp_event_fence._fields_ = ['fence_fd']
setattr(struct_kgsl_timestamp_event_fence, 'fence_fd', field(0, ctypes.c_int32))
class struct_kgsl_gpumem_alloc_id(Struct): pass
struct_kgsl_gpumem_alloc_id.SIZE = 48
struct_kgsl_gpumem_alloc_id._fields_ = ['id', 'flags', 'size', 'mmapsize', 'gpuaddr', '__pad']
setattr(struct_kgsl_gpumem_alloc_id, 'id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_alloc_id, 'flags', field(4, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_alloc_id, 'size', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_alloc_id, 'mmapsize', field(16, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_alloc_id, 'gpuaddr', field(24, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_alloc_id, '__pad', field(32, Array(ctypes.c_uint64, 2)))
class struct_kgsl_gpumem_free_id(Struct): pass
struct_kgsl_gpumem_free_id.SIZE = 8
struct_kgsl_gpumem_free_id._fields_ = ['id', '__pad']
setattr(struct_kgsl_gpumem_free_id, 'id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_free_id, '__pad', field(4, ctypes.c_uint32))
class struct_kgsl_gpumem_get_info(Struct): pass
struct_kgsl_gpumem_get_info.SIZE = 72
struct_kgsl_gpumem_get_info._fields_ = ['gpuaddr', 'id', 'flags', 'size', 'mmapsize', 'useraddr', '__pad']
setattr(struct_kgsl_gpumem_get_info, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_get_info, 'id', field(8, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_get_info, 'flags', field(12, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_get_info, 'size', field(16, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_get_info, 'mmapsize', field(24, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_get_info, 'useraddr', field(32, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_get_info, '__pad', field(40, Array(ctypes.c_uint64, 4)))
class struct_kgsl_gpumem_sync_cache(Struct): pass
struct_kgsl_gpumem_sync_cache.SIZE = 32
struct_kgsl_gpumem_sync_cache._fields_ = ['gpuaddr', 'id', 'op', 'offset', 'length']
setattr(struct_kgsl_gpumem_sync_cache, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_sync_cache, 'id', field(8, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_sync_cache, 'op', field(12, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_sync_cache, 'offset', field(16, ctypes.c_uint64))
setattr(struct_kgsl_gpumem_sync_cache, 'length', field(24, ctypes.c_uint64))
class struct_kgsl_perfcounter_get(Struct): pass
struct_kgsl_perfcounter_get.SIZE = 20
struct_kgsl_perfcounter_get._fields_ = ['groupid', 'countable', 'offset', 'offset_hi', '__pad']
setattr(struct_kgsl_perfcounter_get, 'groupid', field(0, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_get, 'countable', field(4, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_get, 'offset', field(8, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_get, 'offset_hi', field(12, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_get, '__pad', field(16, ctypes.c_uint32))
class struct_kgsl_perfcounter_put(Struct): pass
struct_kgsl_perfcounter_put.SIZE = 16
struct_kgsl_perfcounter_put._fields_ = ['groupid', 'countable', '__pad']
setattr(struct_kgsl_perfcounter_put, 'groupid', field(0, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_put, 'countable', field(4, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_put, '__pad', field(8, Array(ctypes.c_uint32, 2)))
class struct_kgsl_perfcounter_query(Struct): pass
struct_kgsl_perfcounter_query.SIZE = 32
struct_kgsl_perfcounter_query._fields_ = ['groupid', 'countables', 'count', 'max_counters', '__pad']
setattr(struct_kgsl_perfcounter_query, 'groupid', field(0, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_query, 'countables', field(8, Pointer(ctypes.c_uint32)))
setattr(struct_kgsl_perfcounter_query, 'count', field(16, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_query, 'max_counters', field(20, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_query, '__pad', field(24, Array(ctypes.c_uint32, 2)))
class struct_kgsl_perfcounter_read_group(Struct): pass
struct_kgsl_perfcounter_read_group.SIZE = 16
struct_kgsl_perfcounter_read_group._fields_ = ['groupid', 'countable', 'value']
setattr(struct_kgsl_perfcounter_read_group, 'groupid', field(0, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_read_group, 'countable', field(4, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_read_group, 'value', field(8, ctypes.c_uint64))
class struct_kgsl_perfcounter_read(Struct): pass
struct_kgsl_perfcounter_read.SIZE = 24
struct_kgsl_perfcounter_read._fields_ = ['reads', 'count', '__pad']
setattr(struct_kgsl_perfcounter_read, 'reads', field(0, Pointer(struct_kgsl_perfcounter_read_group)))
setattr(struct_kgsl_perfcounter_read, 'count', field(8, ctypes.c_uint32))
setattr(struct_kgsl_perfcounter_read, '__pad', field(12, Array(ctypes.c_uint32, 2)))
class struct_kgsl_gpumem_sync_cache_bulk(Struct): pass
struct_kgsl_gpumem_sync_cache_bulk.SIZE = 24
struct_kgsl_gpumem_sync_cache_bulk._fields_ = ['id_list', 'count', 'op', '__pad']
setattr(struct_kgsl_gpumem_sync_cache_bulk, 'id_list', field(0, Pointer(ctypes.c_uint32)))
setattr(struct_kgsl_gpumem_sync_cache_bulk, 'count', field(8, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_sync_cache_bulk, 'op', field(12, ctypes.c_uint32))
setattr(struct_kgsl_gpumem_sync_cache_bulk, '__pad', field(16, Array(ctypes.c_uint32, 2)))
class struct_kgsl_cmd_syncpoint_timestamp(Struct): pass
struct_kgsl_cmd_syncpoint_timestamp.SIZE = 8
struct_kgsl_cmd_syncpoint_timestamp._fields_ = ['context_id', 'timestamp']
setattr(struct_kgsl_cmd_syncpoint_timestamp, 'context_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_cmd_syncpoint_timestamp, 'timestamp', field(4, ctypes.c_uint32))
class struct_kgsl_cmd_syncpoint_fence(Struct): pass
struct_kgsl_cmd_syncpoint_fence.SIZE = 4
struct_kgsl_cmd_syncpoint_fence._fields_ = ['fd']
setattr(struct_kgsl_cmd_syncpoint_fence, 'fd', field(0, ctypes.c_int32))
class struct_kgsl_cmd_syncpoint(Struct): pass
struct_kgsl_cmd_syncpoint.SIZE = 24
struct_kgsl_cmd_syncpoint._fields_ = ['type', 'priv', 'size']
setattr(struct_kgsl_cmd_syncpoint, 'type', field(0, ctypes.c_int32))
setattr(struct_kgsl_cmd_syncpoint, 'priv', field(8, ctypes.c_void_p))
setattr(struct_kgsl_cmd_syncpoint, 'size', field(16, ctypes.c_uint64))
class struct_kgsl_submit_commands(Struct): pass
struct_kgsl_submit_commands.SIZE = 56
struct_kgsl_submit_commands._fields_ = ['context_id', 'flags', 'cmdlist', 'numcmds', 'synclist', 'numsyncs', 'timestamp', '__pad']
setattr(struct_kgsl_submit_commands, 'context_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_submit_commands, 'flags', field(4, ctypes.c_uint32))
setattr(struct_kgsl_submit_commands, 'cmdlist', field(8, Pointer(struct_kgsl_ibdesc)))
setattr(struct_kgsl_submit_commands, 'numcmds', field(16, ctypes.c_uint32))
setattr(struct_kgsl_submit_commands, 'synclist', field(24, Pointer(struct_kgsl_cmd_syncpoint)))
setattr(struct_kgsl_submit_commands, 'numsyncs', field(32, ctypes.c_uint32))
setattr(struct_kgsl_submit_commands, 'timestamp', field(36, ctypes.c_uint32))
setattr(struct_kgsl_submit_commands, '__pad', field(40, Array(ctypes.c_uint32, 4)))
class struct_kgsl_device_constraint(Struct): pass
struct_kgsl_device_constraint.SIZE = 24
struct_kgsl_device_constraint._fields_ = ['type', 'context_id', 'data', 'size']
setattr(struct_kgsl_device_constraint, 'type', field(0, ctypes.c_uint32))
setattr(struct_kgsl_device_constraint, 'context_id', field(4, ctypes.c_uint32))
setattr(struct_kgsl_device_constraint, 'data', field(8, ctypes.c_void_p))
setattr(struct_kgsl_device_constraint, 'size', field(16, ctypes.c_uint64))
class struct_kgsl_device_constraint_pwrlevel(Struct): pass
struct_kgsl_device_constraint_pwrlevel.SIZE = 4
struct_kgsl_device_constraint_pwrlevel._fields_ = ['level']
setattr(struct_kgsl_device_constraint_pwrlevel, 'level', field(0, ctypes.c_uint32))
class struct_kgsl_syncsource_create(Struct): pass
struct_kgsl_syncsource_create.SIZE = 16
struct_kgsl_syncsource_create._fields_ = ['id', '__pad']
setattr(struct_kgsl_syncsource_create, 'id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_syncsource_create, '__pad', field(4, Array(ctypes.c_uint32, 3)))
class struct_kgsl_syncsource_destroy(Struct): pass
struct_kgsl_syncsource_destroy.SIZE = 16
struct_kgsl_syncsource_destroy._fields_ = ['id', '__pad']
setattr(struct_kgsl_syncsource_destroy, 'id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_syncsource_destroy, '__pad', field(4, Array(ctypes.c_uint32, 3)))
class struct_kgsl_syncsource_create_fence(Struct): pass
struct_kgsl_syncsource_create_fence.SIZE = 24
struct_kgsl_syncsource_create_fence._fields_ = ['id', 'fence_fd', '__pad']
setattr(struct_kgsl_syncsource_create_fence, 'id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_syncsource_create_fence, 'fence_fd', field(4, ctypes.c_int32))
setattr(struct_kgsl_syncsource_create_fence, '__pad', field(8, Array(ctypes.c_uint32, 4)))
class struct_kgsl_syncsource_signal_fence(Struct): pass
struct_kgsl_syncsource_signal_fence.SIZE = 24
struct_kgsl_syncsource_signal_fence._fields_ = ['id', 'fence_fd', '__pad']
setattr(struct_kgsl_syncsource_signal_fence, 'id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_syncsource_signal_fence, 'fence_fd', field(4, ctypes.c_int32))
setattr(struct_kgsl_syncsource_signal_fence, '__pad', field(8, Array(ctypes.c_uint32, 4)))
class struct_kgsl_cff_sync_gpuobj(Struct): pass
struct_kgsl_cff_sync_gpuobj.SIZE = 24
struct_kgsl_cff_sync_gpuobj._fields_ = ['offset', 'length', 'id']
setattr(struct_kgsl_cff_sync_gpuobj, 'offset', field(0, ctypes.c_uint64))
setattr(struct_kgsl_cff_sync_gpuobj, 'length', field(8, ctypes.c_uint64))
setattr(struct_kgsl_cff_sync_gpuobj, 'id', field(16, ctypes.c_uint32))
class struct_kgsl_gpuobj_alloc(Struct): pass
struct_kgsl_gpuobj_alloc.SIZE = 48
struct_kgsl_gpuobj_alloc._fields_ = ['size', 'flags', 'va_len', 'mmapsize', 'id', 'metadata_len', 'metadata']
setattr(struct_kgsl_gpuobj_alloc, 'size', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_alloc, 'flags', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_alloc, 'va_len', field(16, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_alloc, 'mmapsize', field(24, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_alloc, 'id', field(32, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_alloc, 'metadata_len', field(36, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_alloc, 'metadata', field(40, ctypes.c_uint64))
class struct_kgsl_gpuobj_free(Struct): pass
struct_kgsl_gpuobj_free.SIZE = 32
struct_kgsl_gpuobj_free._fields_ = ['flags', 'priv', 'id', 'type', 'len']
setattr(struct_kgsl_gpuobj_free, 'flags', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_free, 'priv', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_free, 'id', field(16, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_free, 'type', field(20, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_free, 'len', field(24, ctypes.c_uint32))
class struct_kgsl_gpu_event_timestamp(Struct): pass
struct_kgsl_gpu_event_timestamp.SIZE = 8
struct_kgsl_gpu_event_timestamp._fields_ = ['context_id', 'timestamp']
setattr(struct_kgsl_gpu_event_timestamp, 'context_id', field(0, ctypes.c_uint32))
setattr(struct_kgsl_gpu_event_timestamp, 'timestamp', field(4, ctypes.c_uint32))
class struct_kgsl_gpu_event_fence(Struct): pass
struct_kgsl_gpu_event_fence.SIZE = 4
struct_kgsl_gpu_event_fence._fields_ = ['fd']
setattr(struct_kgsl_gpu_event_fence, 'fd', field(0, ctypes.c_int32))
class struct_kgsl_gpuobj_info(Struct): pass
struct_kgsl_gpuobj_info.SIZE = 48
struct_kgsl_gpuobj_info._fields_ = ['gpuaddr', 'flags', 'size', 'va_len', 'va_addr', 'id']
setattr(struct_kgsl_gpuobj_info, 'gpuaddr', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_info, 'flags', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_info, 'size', field(16, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_info, 'va_len', field(24, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_info, 'va_addr', field(32, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_info, 'id', field(40, ctypes.c_uint32))
class struct_kgsl_gpuobj_import(Struct): pass
struct_kgsl_gpuobj_import.SIZE = 32
struct_kgsl_gpuobj_import._fields_ = ['priv', 'priv_len', 'flags', 'type', 'id']
setattr(struct_kgsl_gpuobj_import, 'priv', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_import, 'priv_len', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_import, 'flags', field(16, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_import, 'type', field(24, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_import, 'id', field(28, ctypes.c_uint32))
class struct_kgsl_gpuobj_import_dma_buf(Struct): pass
struct_kgsl_gpuobj_import_dma_buf.SIZE = 4
struct_kgsl_gpuobj_import_dma_buf._fields_ = ['fd']
setattr(struct_kgsl_gpuobj_import_dma_buf, 'fd', field(0, ctypes.c_int32))
class struct_kgsl_gpuobj_import_useraddr(Struct): pass
struct_kgsl_gpuobj_import_useraddr.SIZE = 8
struct_kgsl_gpuobj_import_useraddr._fields_ = ['virtaddr']
setattr(struct_kgsl_gpuobj_import_useraddr, 'virtaddr', field(0, ctypes.c_uint64))
class struct_kgsl_gpuobj_sync_obj(Struct): pass
struct_kgsl_gpuobj_sync_obj.SIZE = 24
struct_kgsl_gpuobj_sync_obj._fields_ = ['offset', 'length', 'id', 'op']
setattr(struct_kgsl_gpuobj_sync_obj, 'offset', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_sync_obj, 'length', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_sync_obj, 'id', field(16, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_sync_obj, 'op', field(20, ctypes.c_uint32))
class struct_kgsl_gpuobj_sync(Struct): pass
struct_kgsl_gpuobj_sync.SIZE = 16
struct_kgsl_gpuobj_sync._fields_ = ['objs', 'obj_len', 'count']
setattr(struct_kgsl_gpuobj_sync, 'objs', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_sync, 'obj_len', field(8, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_sync, 'count', field(12, ctypes.c_uint32))
class struct_kgsl_command_object(Struct): pass
struct_kgsl_command_object.SIZE = 32
struct_kgsl_command_object._fields_ = ['offset', 'gpuaddr', 'size', 'flags', 'id']
setattr(struct_kgsl_command_object, 'offset', field(0, ctypes.c_uint64))
setattr(struct_kgsl_command_object, 'gpuaddr', field(8, ctypes.c_uint64))
setattr(struct_kgsl_command_object, 'size', field(16, ctypes.c_uint64))
setattr(struct_kgsl_command_object, 'flags', field(24, ctypes.c_uint32))
setattr(struct_kgsl_command_object, 'id', field(28, ctypes.c_uint32))
class struct_kgsl_command_syncpoint(Struct): pass
struct_kgsl_command_syncpoint.SIZE = 24
struct_kgsl_command_syncpoint._fields_ = ['priv', 'size', 'type']
setattr(struct_kgsl_command_syncpoint, 'priv', field(0, ctypes.c_uint64))
setattr(struct_kgsl_command_syncpoint, 'size', field(8, ctypes.c_uint64))
setattr(struct_kgsl_command_syncpoint, 'type', field(16, ctypes.c_uint32))
class struct_kgsl_gpu_command(Struct): pass
struct_kgsl_gpu_command.SIZE = 64
struct_kgsl_gpu_command._fields_ = ['flags', 'cmdlist', 'cmdsize', 'numcmds', 'objlist', 'objsize', 'numobjs', 'synclist', 'syncsize', 'numsyncs', 'context_id', 'timestamp']
setattr(struct_kgsl_gpu_command, 'flags', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpu_command, 'cmdlist', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpu_command, 'cmdsize', field(16, ctypes.c_uint32))
setattr(struct_kgsl_gpu_command, 'numcmds', field(20, ctypes.c_uint32))
setattr(struct_kgsl_gpu_command, 'objlist', field(24, ctypes.c_uint64))
setattr(struct_kgsl_gpu_command, 'objsize', field(32, ctypes.c_uint32))
setattr(struct_kgsl_gpu_command, 'numobjs', field(36, ctypes.c_uint32))
setattr(struct_kgsl_gpu_command, 'synclist', field(40, ctypes.c_uint64))
setattr(struct_kgsl_gpu_command, 'syncsize', field(48, ctypes.c_uint32))
setattr(struct_kgsl_gpu_command, 'numsyncs', field(52, ctypes.c_uint32))
setattr(struct_kgsl_gpu_command, 'context_id', field(56, ctypes.c_uint32))
setattr(struct_kgsl_gpu_command, 'timestamp', field(60, ctypes.c_uint32))
class struct_kgsl_preemption_counters_query(Struct): pass
struct_kgsl_preemption_counters_query.SIZE = 24
struct_kgsl_preemption_counters_query._fields_ = ['counters', 'size_user', 'size_priority_level', 'max_priority_level']
setattr(struct_kgsl_preemption_counters_query, 'counters', field(0, ctypes.c_uint64))
setattr(struct_kgsl_preemption_counters_query, 'size_user', field(8, ctypes.c_uint32))
setattr(struct_kgsl_preemption_counters_query, 'size_priority_level', field(12, ctypes.c_uint32))
setattr(struct_kgsl_preemption_counters_query, 'max_priority_level', field(16, ctypes.c_uint32))
class struct_kgsl_gpuobj_set_info(Struct): pass
struct_kgsl_gpuobj_set_info.SIZE = 32
struct_kgsl_gpuobj_set_info._fields_ = ['flags', 'metadata', 'id', 'metadata_len', 'type']
setattr(struct_kgsl_gpuobj_set_info, 'flags', field(0, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_set_info, 'metadata', field(8, ctypes.c_uint64))
setattr(struct_kgsl_gpuobj_set_info, 'id', field(16, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_set_info, 'metadata_len', field(20, ctypes.c_uint32))
setattr(struct_kgsl_gpuobj_set_info, 'type', field(24, ctypes.c_uint32))
KGSL_VERSION_MAJOR = 3
KGSL_VERSION_MINOR = 14
KGSL_CONTEXT_SAVE_GMEM = 0x00000001
KGSL_CONTEXT_NO_GMEM_ALLOC = 0x00000002
KGSL_CONTEXT_SUBMIT_IB_LIST = 0x00000004
KGSL_CONTEXT_CTX_SWITCH = 0x00000008
KGSL_CONTEXT_PREAMBLE = 0x00000010
KGSL_CONTEXT_TRASH_STATE = 0x00000020
KGSL_CONTEXT_PER_CONTEXT_TS = 0x00000040
KGSL_CONTEXT_USER_GENERATED_TS = 0x00000080
KGSL_CONTEXT_END_OF_FRAME = 0x00000100
KGSL_CONTEXT_NO_FAULT_TOLERANCE = 0x00000200
KGSL_CONTEXT_SYNC = 0x00000400
KGSL_CONTEXT_PWR_CONSTRAINT = 0x00000800
KGSL_CONTEXT_PRIORITY_MASK = 0x0000F000
KGSL_CONTEXT_PRIORITY_SHIFT = 12
KGSL_CONTEXT_PRIORITY_UNDEF = 0
KGSL_CONTEXT_IFH_NOP = 0x00010000
KGSL_CONTEXT_SECURE = 0x00020000
KGSL_CONTEXT_PREEMPT_STYLE_MASK = 0x0E000000
KGSL_CONTEXT_PREEMPT_STYLE_SHIFT = 25
KGSL_CONTEXT_PREEMPT_STYLE_DEFAULT = 0x0
KGSL_CONTEXT_PREEMPT_STYLE_RINGBUFFER = 0x1
KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN = 0x2
KGSL_CONTEXT_TYPE_MASK = 0x01F00000
KGSL_CONTEXT_TYPE_SHIFT = 20
KGSL_CONTEXT_TYPE_ANY = 0
KGSL_CONTEXT_TYPE_GL = 1
KGSL_CONTEXT_TYPE_CL = 2
KGSL_CONTEXT_TYPE_C2D = 3
KGSL_CONTEXT_TYPE_RS = 4
KGSL_CONTEXT_TYPE_UNKNOWN = 0x1E
KGSL_CONTEXT_INVALID = 0xffffffff
KGSL_CMDBATCH_MEMLIST = 0x00000001
KGSL_CMDBATCH_MARKER = 0x00000002
KGSL_CMDBATCH_SUBMIT_IB_LIST = KGSL_CONTEXT_SUBMIT_IB_LIST
KGSL_CMDBATCH_CTX_SWITCH = KGSL_CONTEXT_CTX_SWITCH
KGSL_CMDBATCH_PROFILING = 0x00000010
KGSL_CMDBATCH_PROFILING_KTIME = 0x00000020
KGSL_CMDBATCH_END_OF_FRAME = KGSL_CONTEXT_END_OF_FRAME
KGSL_CMDBATCH_SYNC = KGSL_CONTEXT_SYNC
KGSL_CMDBATCH_PWR_CONSTRAINT = KGSL_CONTEXT_PWR_CONSTRAINT
KGSL_CMDLIST_IB = 0x00000001
KGSL_CMDLIST_CTXTSWITCH_PREAMBLE = 0x00000002
KGSL_CMDLIST_IB_PREAMBLE = 0x00000004
KGSL_OBJLIST_MEMOBJ = 0x00000008
KGSL_OBJLIST_PROFILE = 0x00000010
KGSL_CMD_SYNCPOINT_TYPE_TIMESTAMP = 0
KGSL_CMD_SYNCPOINT_TYPE_FENCE = 1
KGSL_MEMFLAGS_SECURE = 0x00000008
KGSL_MEMFLAGS_GPUREADONLY = 0x01000000
KGSL_MEMFLAGS_GPUWRITEONLY = 0x02000000
KGSL_MEMFLAGS_FORCE_32BIT = 0x100000000
KGSL_CACHEMODE_MASK = 0x0C000000
KGSL_CACHEMODE_SHIFT = 26
KGSL_CACHEMODE_WRITECOMBINE = 0
KGSL_CACHEMODE_UNCACHED = 1
KGSL_CACHEMODE_WRITETHROUGH = 2
KGSL_CACHEMODE_WRITEBACK = 3
KGSL_MEMFLAGS_USE_CPU_MAP = 0x10000000
KGSL_MEMTYPE_MASK = 0x0000FF00
KGSL_MEMTYPE_SHIFT = 8
KGSL_MEMTYPE_OBJECTANY = 0
KGSL_MEMTYPE_FRAMEBUFFER = 1
KGSL_MEMTYPE_RENDERBUFFER = 2
KGSL_MEMTYPE_ARRAYBUFFER = 3
KGSL_MEMTYPE_ELEMENTARRAYBUFFER = 4
KGSL_MEMTYPE_VERTEXARRAYBUFFER = 5
KGSL_MEMTYPE_TEXTURE = 6
KGSL_MEMTYPE_SURFACE = 7
KGSL_MEMTYPE_EGL_SURFACE = 8
KGSL_MEMTYPE_GL = 9
KGSL_MEMTYPE_CL = 10
KGSL_MEMTYPE_CL_BUFFER_MAP = 11
KGSL_MEMTYPE_CL_BUFFER_NOMAP = 12
KGSL_MEMTYPE_CL_IMAGE_MAP = 13
KGSL_MEMTYPE_CL_IMAGE_NOMAP = 14
KGSL_MEMTYPE_CL_KERNEL_STACK = 15
KGSL_MEMTYPE_COMMAND = 16
KGSL_MEMTYPE_2D = 17
KGSL_MEMTYPE_EGL_IMAGE = 18
KGSL_MEMTYPE_EGL_SHADOW = 19
KGSL_MEMTYPE_MULTISAMPLE = 20
KGSL_MEMTYPE_KERNEL = 255
KGSL_MEMALIGN_MASK = 0x00FF0000
KGSL_MEMALIGN_SHIFT = 16
KGSL_MEMFLAGS_USERMEM_MASK = 0x000000e0
KGSL_MEMFLAGS_USERMEM_SHIFT = 5
KGSL_USERMEM_FLAG = lambda x: (((x) + 1) << KGSL_MEMFLAGS_USERMEM_SHIFT)
KGSL_MEMFLAGS_NOT_USERMEM = 0
KGSL_MEMFLAGS_USERMEM_PMEM = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_PMEM)
KGSL_MEMFLAGS_USERMEM_ASHMEM = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ASHMEM)
KGSL_MEMFLAGS_USERMEM_ADDR = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ADDR)
KGSL_MEMFLAGS_USERMEM_ION = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ION)
KGSL_FLAGS_NORMALMODE = 0x00000000
KGSL_FLAGS_SAFEMODE = 0x00000001
KGSL_FLAGS_INITIALIZED0 = 0x00000002
KGSL_FLAGS_INITIALIZED = 0x00000004
KGSL_FLAGS_STARTED = 0x00000008
KGSL_FLAGS_ACTIVE = 0x00000010
KGSL_FLAGS_RESERVED0 = 0x00000020
KGSL_FLAGS_RESERVED1 = 0x00000040
KGSL_FLAGS_RESERVED2 = 0x00000080
KGSL_FLAGS_SOFT_RESET = 0x00000100
KGSL_FLAGS_PER_CONTEXT_TIMESTAMPS = 0x00000200
KGSL_SYNCOBJ_SERVER_TIMEOUT = 2000
KGSL_CONVERT_TO_MBPS = lambda val: (val*1000*1000)
KGSL_MEMSTORE_OFFSET = lambda ctxt_id,field: ((ctxt_id)*sizeof(struct_kgsl_devmemstore) + offsetof(struct_kgsl_devmemstore, field))
KGSL_PROP_DEVICE_INFO = 0x1
KGSL_PROP_DEVICE_SHADOW = 0x2
KGSL_PROP_DEVICE_POWER = 0x3
KGSL_PROP_SHMEM = 0x4
KGSL_PROP_SHMEM_APERTURES = 0x5
KGSL_PROP_MMU_ENABLE = 0x6
KGSL_PROP_INTERRUPT_WAITS = 0x7
KGSL_PROP_VERSION = 0x8
KGSL_PROP_GPU_RESET_STAT = 0x9
KGSL_PROP_PWRCTRL = 0xE
KGSL_PROP_PWR_CONSTRAINT = 0x12
KGSL_PROP_UCHE_GMEM_VADDR = 0x13
KGSL_PROP_SP_GENERIC_MEM = 0x14
KGSL_PROP_UCODE_VERSION = 0x15
KGSL_PROP_GPMU_VERSION = 0x16
KGSL_PROP_DEVICE_BITNESS = 0x18
KGSL_PERFCOUNTER_GROUP_CP = 0x0
KGSL_PERFCOUNTER_GROUP_RBBM = 0x1
KGSL_PERFCOUNTER_GROUP_PC = 0x2
KGSL_PERFCOUNTER_GROUP_VFD = 0x3
KGSL_PERFCOUNTER_GROUP_HLSQ = 0x4
KGSL_PERFCOUNTER_GROUP_VPC = 0x5
KGSL_PERFCOUNTER_GROUP_TSE = 0x6
KGSL_PERFCOUNTER_GROUP_RAS = 0x7
KGSL_PERFCOUNTER_GROUP_UCHE = 0x8
KGSL_PERFCOUNTER_GROUP_TP = 0x9
KGSL_PERFCOUNTER_GROUP_SP = 0xA
KGSL_PERFCOUNTER_GROUP_RB = 0xB
KGSL_PERFCOUNTER_GROUP_PWR = 0xC
KGSL_PERFCOUNTER_GROUP_VBIF = 0xD
KGSL_PERFCOUNTER_GROUP_VBIF_PWR = 0xE
KGSL_PERFCOUNTER_GROUP_MH = 0xF
KGSL_PERFCOUNTER_GROUP_PA_SU = 0x10
KGSL_PERFCOUNTER_GROUP_SQ = 0x11
KGSL_PERFCOUNTER_GROUP_SX = 0x12
KGSL_PERFCOUNTER_GROUP_TCF = 0x13
KGSL_PERFCOUNTER_GROUP_TCM = 0x14
KGSL_PERFCOUNTER_GROUP_TCR = 0x15
KGSL_PERFCOUNTER_GROUP_L2 = 0x16
KGSL_PERFCOUNTER_GROUP_VSC = 0x17
KGSL_PERFCOUNTER_GROUP_CCU = 0x18
KGSL_PERFCOUNTER_GROUP_LRZ = 0x19
KGSL_PERFCOUNTER_GROUP_CMP = 0x1A
KGSL_PERFCOUNTER_GROUP_ALWAYSON = 0x1B
KGSL_PERFCOUNTER_GROUP_SP_PWR = 0x1C
KGSL_PERFCOUNTER_GROUP_TP_PWR = 0x1D
KGSL_PERFCOUNTER_GROUP_RB_PWR = 0x1E
KGSL_PERFCOUNTER_GROUP_CCU_PWR = 0x1F
KGSL_PERFCOUNTER_GROUP_UCHE_PWR = 0x20
KGSL_PERFCOUNTER_GROUP_CP_PWR = 0x21
KGSL_PERFCOUNTER_GROUP_GPMU_PWR = 0x22
KGSL_PERFCOUNTER_GROUP_ALWAYSON_PWR = 0x23
KGSL_PERFCOUNTER_GROUP_MAX = 0x24
KGSL_PERFCOUNTER_NOT_USED = 0xFFFFFFFF
KGSL_PERFCOUNTER_BROKEN = 0xFFFFFFFE
KGSL_IOC_TYPE = 0x09
IOCTL_KGSL_DEVICE_GETPROPERTY = _IOWR(KGSL_IOC_TYPE, 0x2, struct_kgsl_device_getproperty)
IOCTL_KGSL_DEVICE_WAITTIMESTAMP = _IOW(KGSL_IOC_TYPE, 0x6, struct_kgsl_device_waittimestamp)
IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID = _IOW(KGSL_IOC_TYPE, 0x7, struct_kgsl_device_waittimestamp_ctxtid)
IOCTL_KGSL_RINGBUFFER_ISSUEIBCMDS = _IOWR(KGSL_IOC_TYPE, 0x10, struct_kgsl_ringbuffer_issueibcmds)
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_OLD = _IOR(KGSL_IOC_TYPE, 0x11, struct_kgsl_cmdstream_readtimestamp)
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP = _IOWR(KGSL_IOC_TYPE, 0x11, struct_kgsl_cmdstream_readtimestamp)
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP = _IOW(KGSL_IOC_TYPE, 0x12, struct_kgsl_cmdstream_freememontimestamp)
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_OLD = _IOR(KGSL_IOC_TYPE, 0x12, struct_kgsl_cmdstream_freememontimestamp)
IOCTL_KGSL_DRAWCTXT_CREATE = _IOWR(KGSL_IOC_TYPE, 0x13, struct_kgsl_drawctxt_create)
IOCTL_KGSL_DRAWCTXT_DESTROY = _IOW(KGSL_IOC_TYPE, 0x14, struct_kgsl_drawctxt_destroy)
IOCTL_KGSL_MAP_USER_MEM = _IOWR(KGSL_IOC_TYPE, 0x15, struct_kgsl_map_user_mem)
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID = _IOWR(KGSL_IOC_TYPE, 0x16, struct_kgsl_cmdstream_readtimestamp_ctxtid)
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_CTXTID = _IOW(KGSL_IOC_TYPE, 0x17, struct_kgsl_cmdstream_freememontimestamp_ctxtid)
IOCTL_KGSL_SHAREDMEM_FROM_PMEM = _IOWR(KGSL_IOC_TYPE, 0x20, struct_kgsl_sharedmem_from_pmem)
IOCTL_KGSL_SHAREDMEM_FREE = _IOW(KGSL_IOC_TYPE, 0x21, struct_kgsl_sharedmem_free)
IOCTL_KGSL_CFF_USER_EVENT = _IOW(KGSL_IOC_TYPE, 0x31, struct_kgsl_cff_user_event)
IOCTL_KGSL_DRAWCTXT_BIND_GMEM_SHADOW = _IOW(KGSL_IOC_TYPE, 0x22, struct_kgsl_bind_gmem_shadow)
IOCTL_KGSL_SHAREDMEM_FROM_VMALLOC = _IOWR(KGSL_IOC_TYPE, 0x23, struct_kgsl_sharedmem_from_vmalloc)
IOCTL_KGSL_SHAREDMEM_FLUSH_CACHE = _IOW(KGSL_IOC_TYPE, 0x24, struct_kgsl_sharedmem_free)
IOCTL_KGSL_DRAWCTXT_SET_BIN_BASE_OFFSET = _IOW(KGSL_IOC_TYPE, 0x25, struct_kgsl_drawctxt_set_bin_base_offset)
IOCTL_KGSL_CMDWINDOW_WRITE = _IOW(KGSL_IOC_TYPE, 0x2e, struct_kgsl_cmdwindow_write)
IOCTL_KGSL_GPUMEM_ALLOC = _IOWR(KGSL_IOC_TYPE, 0x2f, struct_kgsl_gpumem_alloc)
IOCTL_KGSL_CFF_SYNCMEM = _IOW(KGSL_IOC_TYPE, 0x30, struct_kgsl_cff_syncmem)
IOCTL_KGSL_TIMESTAMP_EVENT_OLD = _IOW(KGSL_IOC_TYPE, 0x31, struct_kgsl_timestamp_event)
KGSL_TIMESTAMP_EVENT_GENLOCK = 1
KGSL_TIMESTAMP_EVENT_FENCE = 2
IOCTL_KGSL_SETPROPERTY = _IOW(KGSL_IOC_TYPE, 0x32, struct_kgsl_device_getproperty)
IOCTL_KGSL_TIMESTAMP_EVENT = _IOWR(KGSL_IOC_TYPE, 0x33, struct_kgsl_timestamp_event)
IOCTL_KGSL_GPUMEM_ALLOC_ID = _IOWR(KGSL_IOC_TYPE, 0x34, struct_kgsl_gpumem_alloc_id)
IOCTL_KGSL_GPUMEM_FREE_ID = _IOWR(KGSL_IOC_TYPE, 0x35, struct_kgsl_gpumem_free_id)
IOCTL_KGSL_GPUMEM_GET_INFO = _IOWR(KGSL_IOC_TYPE, 0x36, struct_kgsl_gpumem_get_info)
KGSL_GPUMEM_CACHE_CLEAN = (1 << 0)
KGSL_GPUMEM_CACHE_TO_GPU = KGSL_GPUMEM_CACHE_CLEAN
KGSL_GPUMEM_CACHE_INV = (1 << 1)
KGSL_GPUMEM_CACHE_FROM_GPU = KGSL_GPUMEM_CACHE_INV
KGSL_GPUMEM_CACHE_FLUSH = (KGSL_GPUMEM_CACHE_CLEAN | KGSL_GPUMEM_CACHE_INV)
KGSL_GPUMEM_CACHE_RANGE = (1 << 31)
IOCTL_KGSL_GPUMEM_SYNC_CACHE = _IOW(KGSL_IOC_TYPE, 0x37, struct_kgsl_gpumem_sync_cache)
IOCTL_KGSL_PERFCOUNTER_GET = _IOWR(KGSL_IOC_TYPE, 0x38, struct_kgsl_perfcounter_get)
IOCTL_KGSL_PERFCOUNTER_PUT = _IOW(KGSL_IOC_TYPE, 0x39, struct_kgsl_perfcounter_put)
IOCTL_KGSL_PERFCOUNTER_QUERY = _IOWR(KGSL_IOC_TYPE, 0x3A, struct_kgsl_perfcounter_query)
IOCTL_KGSL_PERFCOUNTER_READ = _IOWR(KGSL_IOC_TYPE, 0x3B, struct_kgsl_perfcounter_read)
IOCTL_KGSL_GPUMEM_SYNC_CACHE_BULK = _IOWR(KGSL_IOC_TYPE, 0x3C, struct_kgsl_gpumem_sync_cache_bulk)
KGSL_IBDESC_MEMLIST = 0x1
KGSL_IBDESC_PROFILING_BUFFER = 0x2
IOCTL_KGSL_SUBMIT_COMMANDS = _IOWR(KGSL_IOC_TYPE, 0x3D, struct_kgsl_submit_commands)
KGSL_CONSTRAINT_NONE = 0
KGSL_CONSTRAINT_PWRLEVEL = 1
KGSL_CONSTRAINT_PWR_MIN = 0
KGSL_CONSTRAINT_PWR_MAX = 1
IOCTL_KGSL_SYNCSOURCE_CREATE = _IOWR(KGSL_IOC_TYPE, 0x40, struct_kgsl_syncsource_create)
IOCTL_KGSL_SYNCSOURCE_DESTROY = _IOWR(KGSL_IOC_TYPE, 0x41, struct_kgsl_syncsource_destroy)
IOCTL_KGSL_SYNCSOURCE_CREATE_FENCE = _IOWR(KGSL_IOC_TYPE, 0x42, struct_kgsl_syncsource_create_fence)
IOCTL_KGSL_SYNCSOURCE_SIGNAL_FENCE = _IOWR(KGSL_IOC_TYPE, 0x43, struct_kgsl_syncsource_signal_fence)
IOCTL_KGSL_CFF_SYNC_GPUOBJ = _IOW(KGSL_IOC_TYPE, 0x44, struct_kgsl_cff_sync_gpuobj)
KGSL_GPUOBJ_ALLOC_METADATA_MAX = 64
IOCTL_KGSL_GPUOBJ_ALLOC = _IOWR(KGSL_IOC_TYPE, 0x45, struct_kgsl_gpuobj_alloc)
KGSL_GPUOBJ_FREE_ON_EVENT = 1
KGSL_GPU_EVENT_TIMESTAMP = 1
KGSL_GPU_EVENT_FENCE = 2
IOCTL_KGSL_GPUOBJ_FREE = _IOW(KGSL_IOC_TYPE, 0x46, struct_kgsl_gpuobj_free)
IOCTL_KGSL_GPUOBJ_INFO = _IOWR(KGSL_IOC_TYPE, 0x47, struct_kgsl_gpuobj_info)
IOCTL_KGSL_GPUOBJ_IMPORT = _IOWR(KGSL_IOC_TYPE, 0x48, struct_kgsl_gpuobj_import)
IOCTL_KGSL_GPUOBJ_SYNC = _IOW(KGSL_IOC_TYPE, 0x49, struct_kgsl_gpuobj_sync)
IOCTL_KGSL_GPU_COMMAND = _IOWR(KGSL_IOC_TYPE, 0x4A, struct_kgsl_gpu_command)
IOCTL_KGSL_PREEMPTIONCOUNTER_QUERY = _IOWR(KGSL_IOC_TYPE, 0x4B, struct_kgsl_preemption_counters_query)
KGSL_GPUOBJ_SET_INFO_METADATA = (1 << 0)
KGSL_GPUOBJ_SET_INFO_TYPE = (1 << 1)
IOCTL_KGSL_GPUOBJ_SET_INFO = _IOW(KGSL_IOC_TYPE, 0x4C, struct_kgsl_gpuobj_set_info)