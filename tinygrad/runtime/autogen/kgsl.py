# mypy: disable-error-code="empty-body"
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
class enum_kgsl_user_mem_type(ctypes.c_uint32, c.Enum): pass
KGSL_USER_MEM_TYPE_PMEM = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_PMEM', 0)
KGSL_USER_MEM_TYPE_ASHMEM = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ASHMEM', 1)
KGSL_USER_MEM_TYPE_ADDR = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ADDR', 2)
KGSL_USER_MEM_TYPE_ION = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ION', 3)
KGSL_USER_MEM_TYPE_DMABUF = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_DMABUF', 3)
KGSL_USER_MEM_TYPE_MAX = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_MAX', 7)

class enum_kgsl_ctx_reset_stat(ctypes.c_uint32, c.Enum): pass
KGSL_CTX_STAT_NO_ERROR = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_NO_ERROR', 0)
KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT', 1)
KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT', 2)
KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT', 3)

class enum_kgsl_deviceid(ctypes.c_uint32, c.Enum): pass
KGSL_DEVICE_3D0 = enum_kgsl_deviceid.define('KGSL_DEVICE_3D0', 0)
KGSL_DEVICE_MAX = enum_kgsl_deviceid.define('KGSL_DEVICE_MAX', 1)

@c.record
class struct_kgsl_devinfo(c.Struct):
  SIZE = 40
  device_id: 'ctypes.c_uint32'
  chip_id: 'ctypes.c_uint32'
  mmu_enabled: 'ctypes.c_uint32'
  gmem_gpubaseaddr: 'ctypes.c_uint64'
  gpu_id: 'ctypes.c_uint32'
  gmem_sizebytes: 'ctypes.c_uint64'
struct_kgsl_devinfo.register_fields([('device_id', ctypes.c_uint32, 0), ('chip_id', ctypes.c_uint32, 4), ('mmu_enabled', ctypes.c_uint32, 8), ('gmem_gpubaseaddr', ctypes.c_uint64, 16), ('gpu_id', ctypes.c_uint32, 24), ('gmem_sizebytes', ctypes.c_uint64, 32)])
@c.record
class struct_kgsl_devmemstore(c.Struct):
  SIZE = 40
  soptimestamp: 'ctypes.c_uint32'
  sbz: 'ctypes.c_uint32'
  eoptimestamp: 'ctypes.c_uint32'
  sbz2: 'ctypes.c_uint32'
  preempted: 'ctypes.c_uint32'
  sbz3: 'ctypes.c_uint32'
  ref_wait_ts: 'ctypes.c_uint32'
  sbz4: 'ctypes.c_uint32'
  current_context: 'ctypes.c_uint32'
  sbz5: 'ctypes.c_uint32'
struct_kgsl_devmemstore.register_fields([('soptimestamp', ctypes.c_uint32, 0), ('sbz', ctypes.c_uint32, 4), ('eoptimestamp', ctypes.c_uint32, 8), ('sbz2', ctypes.c_uint32, 12), ('preempted', ctypes.c_uint32, 16), ('sbz3', ctypes.c_uint32, 20), ('ref_wait_ts', ctypes.c_uint32, 24), ('sbz4', ctypes.c_uint32, 28), ('current_context', ctypes.c_uint32, 32), ('sbz5', ctypes.c_uint32, 36)])
class enum_kgsl_timestamp_type(ctypes.c_uint32, c.Enum): pass
KGSL_TIMESTAMP_CONSUMED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_CONSUMED', 1)
KGSL_TIMESTAMP_RETIRED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_RETIRED', 2)
KGSL_TIMESTAMP_QUEUED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_QUEUED', 3)

@c.record
class struct_kgsl_shadowprop(c.Struct):
  SIZE = 24
  gpuaddr: 'ctypes.c_uint64'
  size: 'ctypes.c_uint64'
  flags: 'ctypes.c_uint32'
struct_kgsl_shadowprop.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_version(c.Struct):
  SIZE = 16
  drv_major: 'ctypes.c_uint32'
  drv_minor: 'ctypes.c_uint32'
  dev_major: 'ctypes.c_uint32'
  dev_minor: 'ctypes.c_uint32'
struct_kgsl_version.register_fields([('drv_major', ctypes.c_uint32, 0), ('drv_minor', ctypes.c_uint32, 4), ('dev_major', ctypes.c_uint32, 8), ('dev_minor', ctypes.c_uint32, 12)])
@c.record
class struct_kgsl_sp_generic_mem(c.Struct):
  SIZE = 16
  local: 'ctypes.c_uint64'
  pvt: 'ctypes.c_uint64'
struct_kgsl_sp_generic_mem.register_fields([('local', ctypes.c_uint64, 0), ('pvt', ctypes.c_uint64, 8)])
@c.record
class struct_kgsl_ucode_version(c.Struct):
  SIZE = 8
  pfp: 'ctypes.c_uint32'
  pm4: 'ctypes.c_uint32'
struct_kgsl_ucode_version.register_fields([('pfp', ctypes.c_uint32, 0), ('pm4', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_gpmu_version(c.Struct):
  SIZE = 12
  major: 'ctypes.c_uint32'
  minor: 'ctypes.c_uint32'
  features: 'ctypes.c_uint32'
struct_kgsl_gpmu_version.register_fields([('major', ctypes.c_uint32, 0), ('minor', ctypes.c_uint32, 4), ('features', ctypes.c_uint32, 8)])
@c.record
class struct_kgsl_ibdesc(c.Struct):
  SIZE = 32
  gpuaddr: 'ctypes.c_uint64'
  __pad: 'ctypes.c_uint64'
  sizedwords: 'ctypes.c_uint64'
  ctrl: 'ctypes.c_uint32'
struct_kgsl_ibdesc.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('__pad', ctypes.c_uint64, 8), ('sizedwords', ctypes.c_uint64, 16), ('ctrl', ctypes.c_uint32, 24)])
@c.record
class struct_kgsl_cmdbatch_profiling_buffer(c.Struct):
  SIZE = 40
  wall_clock_s: 'ctypes.c_uint64'
  wall_clock_ns: 'ctypes.c_uint64'
  gpu_ticks_queued: 'ctypes.c_uint64'
  gpu_ticks_submitted: 'ctypes.c_uint64'
  gpu_ticks_retired: 'ctypes.c_uint64'
struct_kgsl_cmdbatch_profiling_buffer.register_fields([('wall_clock_s', ctypes.c_uint64, 0), ('wall_clock_ns', ctypes.c_uint64, 8), ('gpu_ticks_queued', ctypes.c_uint64, 16), ('gpu_ticks_submitted', ctypes.c_uint64, 24), ('gpu_ticks_retired', ctypes.c_uint64, 32)])
@c.record
class struct_kgsl_device_getproperty(c.Struct):
  SIZE = 24
  type: 'ctypes.c_uint32'
  value: 'ctypes.c_void_p'
  sizebytes: 'ctypes.c_uint64'
struct_kgsl_device_getproperty.register_fields([('type', ctypes.c_uint32, 0), ('value', ctypes.c_void_p, 8), ('sizebytes', ctypes.c_uint64, 16)])
@c.record
class struct_kgsl_device_waittimestamp(c.Struct):
  SIZE = 8
  timestamp: 'ctypes.c_uint32'
  timeout: 'ctypes.c_uint32'
struct_kgsl_device_waittimestamp.register_fields([('timestamp', ctypes.c_uint32, 0), ('timeout', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_device_waittimestamp_ctxtid(c.Struct):
  SIZE = 12
  context_id: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
  timeout: 'ctypes.c_uint32'
struct_kgsl_device_waittimestamp_ctxtid.register_fields([('context_id', ctypes.c_uint32, 0), ('timestamp', ctypes.c_uint32, 4), ('timeout', ctypes.c_uint32, 8)])
@c.record
class struct_kgsl_ringbuffer_issueibcmds(c.Struct):
  SIZE = 32
  drawctxt_id: 'ctypes.c_uint32'
  ibdesc_addr: 'ctypes.c_uint64'
  numibs: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
  flags: 'ctypes.c_uint32'
struct_kgsl_ringbuffer_issueibcmds.register_fields([('drawctxt_id', ctypes.c_uint32, 0), ('ibdesc_addr', ctypes.c_uint64, 8), ('numibs', ctypes.c_uint32, 16), ('timestamp', ctypes.c_uint32, 20), ('flags', ctypes.c_uint32, 24)])
@c.record
class struct_kgsl_cmdstream_readtimestamp(c.Struct):
  SIZE = 8
  type: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
struct_kgsl_cmdstream_readtimestamp.register_fields([('type', ctypes.c_uint32, 0), ('timestamp', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_cmdstream_freememontimestamp(c.Struct):
  SIZE = 16
  gpuaddr: 'ctypes.c_uint64'
  type: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
struct_kgsl_cmdstream_freememontimestamp.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('type', ctypes.c_uint32, 8), ('timestamp', ctypes.c_uint32, 12)])
@c.record
class struct_kgsl_drawctxt_create(c.Struct):
  SIZE = 8
  flags: 'ctypes.c_uint32'
  drawctxt_id: 'ctypes.c_uint32'
struct_kgsl_drawctxt_create.register_fields([('flags', ctypes.c_uint32, 0), ('drawctxt_id', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_drawctxt_destroy(c.Struct):
  SIZE = 4
  drawctxt_id: 'ctypes.c_uint32'
struct_kgsl_drawctxt_destroy.register_fields([('drawctxt_id', ctypes.c_uint32, 0)])
@c.record
class struct_kgsl_map_user_mem(c.Struct):
  SIZE = 48
  fd: 'ctypes.c_int32'
  gpuaddr: 'ctypes.c_uint64'
  len: 'ctypes.c_uint64'
  offset: 'ctypes.c_uint64'
  hostptr: 'ctypes.c_uint64'
  memtype: 'enum_kgsl_user_mem_type'
  flags: 'ctypes.c_uint32'
struct_kgsl_map_user_mem.register_fields([('fd', ctypes.c_int32, 0), ('gpuaddr', ctypes.c_uint64, 8), ('len', ctypes.c_uint64, 16), ('offset', ctypes.c_uint64, 24), ('hostptr', ctypes.c_uint64, 32), ('memtype', enum_kgsl_user_mem_type, 40), ('flags', ctypes.c_uint32, 44)])
@c.record
class struct_kgsl_cmdstream_readtimestamp_ctxtid(c.Struct):
  SIZE = 12
  context_id: 'ctypes.c_uint32'
  type: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
struct_kgsl_cmdstream_readtimestamp_ctxtid.register_fields([('context_id', ctypes.c_uint32, 0), ('type', ctypes.c_uint32, 4), ('timestamp', ctypes.c_uint32, 8)])
@c.record
class struct_kgsl_cmdstream_freememontimestamp_ctxtid(c.Struct):
  SIZE = 24
  context_id: 'ctypes.c_uint32'
  gpuaddr: 'ctypes.c_uint64'
  type: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
struct_kgsl_cmdstream_freememontimestamp_ctxtid.register_fields([('context_id', ctypes.c_uint32, 0), ('gpuaddr', ctypes.c_uint64, 8), ('type', ctypes.c_uint32, 16), ('timestamp', ctypes.c_uint32, 20)])
@c.record
class struct_kgsl_sharedmem_from_pmem(c.Struct):
  SIZE = 24
  pmem_fd: 'ctypes.c_int32'
  gpuaddr: 'ctypes.c_uint64'
  len: 'ctypes.c_uint32'
  offset: 'ctypes.c_uint32'
struct_kgsl_sharedmem_from_pmem.register_fields([('pmem_fd', ctypes.c_int32, 0), ('gpuaddr', ctypes.c_uint64, 8), ('len', ctypes.c_uint32, 16), ('offset', ctypes.c_uint32, 20)])
@c.record
class struct_kgsl_sharedmem_free(c.Struct):
  SIZE = 8
  gpuaddr: 'ctypes.c_uint64'
struct_kgsl_sharedmem_free.register_fields([('gpuaddr', ctypes.c_uint64, 0)])
@c.record
class struct_kgsl_cff_user_event(c.Struct):
  SIZE = 32
  cff_opcode: 'ctypes.c_ubyte'
  op1: 'ctypes.c_uint32'
  op2: 'ctypes.c_uint32'
  op3: 'ctypes.c_uint32'
  op4: 'ctypes.c_uint32'
  op5: 'ctypes.c_uint32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[2]]'
struct_kgsl_cff_user_event.register_fields([('cff_opcode', ctypes.c_ubyte, 0), ('op1', ctypes.c_uint32, 4), ('op2', ctypes.c_uint32, 8), ('op3', ctypes.c_uint32, 12), ('op4', ctypes.c_uint32, 16), ('op5', ctypes.c_uint32, 20), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 24)])
@c.record
class struct_kgsl_gmem_desc(c.Struct):
  SIZE = 20
  x: 'ctypes.c_uint32'
  y: 'ctypes.c_uint32'
  width: 'ctypes.c_uint32'
  height: 'ctypes.c_uint32'
  pitch: 'ctypes.c_uint32'
struct_kgsl_gmem_desc.register_fields([('x', ctypes.c_uint32, 0), ('y', ctypes.c_uint32, 4), ('width', ctypes.c_uint32, 8), ('height', ctypes.c_uint32, 12), ('pitch', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_buffer_desc(c.Struct):
  SIZE = 32
  hostptr: 'ctypes.c_void_p'
  gpuaddr: 'ctypes.c_uint64'
  size: 'ctypes.c_int32'
  format: 'ctypes.c_uint32'
  pitch: 'ctypes.c_uint32'
  enabled: 'ctypes.c_uint32'
struct_kgsl_buffer_desc.register_fields([('hostptr', ctypes.c_void_p, 0), ('gpuaddr', ctypes.c_uint64, 8), ('size', ctypes.c_int32, 16), ('format', ctypes.c_uint32, 20), ('pitch', ctypes.c_uint32, 24), ('enabled', ctypes.c_uint32, 28)])
@c.record
class struct_kgsl_bind_gmem_shadow(c.Struct):
  SIZE = 72
  drawctxt_id: 'ctypes.c_uint32'
  gmem_desc: 'struct_kgsl_gmem_desc'
  shadow_x: 'ctypes.c_uint32'
  shadow_y: 'ctypes.c_uint32'
  shadow_buffer: 'struct_kgsl_buffer_desc'
  buffer_id: 'ctypes.c_uint32'
struct_kgsl_bind_gmem_shadow.register_fields([('drawctxt_id', ctypes.c_uint32, 0), ('gmem_desc', struct_kgsl_gmem_desc, 4), ('shadow_x', ctypes.c_uint32, 24), ('shadow_y', ctypes.c_uint32, 28), ('shadow_buffer', struct_kgsl_buffer_desc, 32), ('buffer_id', ctypes.c_uint32, 64)])
@c.record
class struct_kgsl_sharedmem_from_vmalloc(c.Struct):
  SIZE = 16
  gpuaddr: 'ctypes.c_uint64'
  hostptr: 'ctypes.c_uint32'
  flags: 'ctypes.c_uint32'
struct_kgsl_sharedmem_from_vmalloc.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('hostptr', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12)])
@c.record
class struct_kgsl_drawctxt_set_bin_base_offset(c.Struct):
  SIZE = 8
  drawctxt_id: 'ctypes.c_uint32'
  offset: 'ctypes.c_uint32'
struct_kgsl_drawctxt_set_bin_base_offset.register_fields([('drawctxt_id', ctypes.c_uint32, 0), ('offset', ctypes.c_uint32, 4)])
class enum_kgsl_cmdwindow_type(ctypes.c_uint32, c.Enum): pass
KGSL_CMDWINDOW_MIN = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MIN', 0)
KGSL_CMDWINDOW_2D = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_2D', 0)
KGSL_CMDWINDOW_3D = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_3D', 1)
KGSL_CMDWINDOW_MMU = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MMU', 2)
KGSL_CMDWINDOW_ARBITER = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_ARBITER', 255)
KGSL_CMDWINDOW_MAX = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MAX', 255)

@c.record
class struct_kgsl_cmdwindow_write(c.Struct):
  SIZE = 12
  target: 'enum_kgsl_cmdwindow_type'
  addr: 'ctypes.c_uint32'
  data: 'ctypes.c_uint32'
struct_kgsl_cmdwindow_write.register_fields([('target', enum_kgsl_cmdwindow_type, 0), ('addr', ctypes.c_uint32, 4), ('data', ctypes.c_uint32, 8)])
@c.record
class struct_kgsl_gpumem_alloc(c.Struct):
  SIZE = 24
  gpuaddr: 'ctypes.c_uint64'
  size: 'ctypes.c_uint64'
  flags: 'ctypes.c_uint32'
struct_kgsl_gpumem_alloc.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_cff_syncmem(c.Struct):
  SIZE = 24
  gpuaddr: 'ctypes.c_uint64'
  len: 'ctypes.c_uint64'
  __pad: 'c.Array[ctypes.c_uint32, Literal[2]]'
struct_kgsl_cff_syncmem.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('len', ctypes.c_uint64, 8), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 16)])
@c.record
class struct_kgsl_timestamp_event(c.Struct):
  SIZE = 32
  type: 'ctypes.c_int32'
  timestamp: 'ctypes.c_uint32'
  context_id: 'ctypes.c_uint32'
  priv: 'ctypes.c_void_p'
  len: 'ctypes.c_uint64'
struct_kgsl_timestamp_event.register_fields([('type', ctypes.c_int32, 0), ('timestamp', ctypes.c_uint32, 4), ('context_id', ctypes.c_uint32, 8), ('priv', ctypes.c_void_p, 16), ('len', ctypes.c_uint64, 24)])
@c.record
class struct_kgsl_timestamp_event_genlock(c.Struct):
  SIZE = 4
  handle: 'ctypes.c_int32'
struct_kgsl_timestamp_event_genlock.register_fields([('handle', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_timestamp_event_fence(c.Struct):
  SIZE = 4
  fence_fd: 'ctypes.c_int32'
struct_kgsl_timestamp_event_fence.register_fields([('fence_fd', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_gpumem_alloc_id(c.Struct):
  SIZE = 48
  id: 'ctypes.c_uint32'
  flags: 'ctypes.c_uint32'
  size: 'ctypes.c_uint64'
  mmapsize: 'ctypes.c_uint64'
  gpuaddr: 'ctypes.c_uint64'
  __pad: 'c.Array[ctypes.c_uint64, Literal[2]]'
struct_kgsl_gpumem_alloc_id.register_fields([('id', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('size', ctypes.c_uint64, 8), ('mmapsize', ctypes.c_uint64, 16), ('gpuaddr', ctypes.c_uint64, 24), ('__pad', c.Array[ctypes.c_uint64, Literal[2]], 32)])
@c.record
class struct_kgsl_gpumem_free_id(c.Struct):
  SIZE = 8
  id: 'ctypes.c_uint32'
  __pad: 'ctypes.c_uint32'
struct_kgsl_gpumem_free_id.register_fields([('id', ctypes.c_uint32, 0), ('__pad', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_gpumem_get_info(c.Struct):
  SIZE = 72
  gpuaddr: 'ctypes.c_uint64'
  id: 'ctypes.c_uint32'
  flags: 'ctypes.c_uint32'
  size: 'ctypes.c_uint64'
  mmapsize: 'ctypes.c_uint64'
  useraddr: 'ctypes.c_uint64'
  __pad: 'c.Array[ctypes.c_uint64, Literal[4]]'
struct_kgsl_gpumem_get_info.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('id', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12), ('size', ctypes.c_uint64, 16), ('mmapsize', ctypes.c_uint64, 24), ('useraddr', ctypes.c_uint64, 32), ('__pad', c.Array[ctypes.c_uint64, Literal[4]], 40)])
@c.record
class struct_kgsl_gpumem_sync_cache(c.Struct):
  SIZE = 32
  gpuaddr: 'ctypes.c_uint64'
  id: 'ctypes.c_uint32'
  op: 'ctypes.c_uint32'
  offset: 'ctypes.c_uint64'
  length: 'ctypes.c_uint64'
struct_kgsl_gpumem_sync_cache.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('id', ctypes.c_uint32, 8), ('op', ctypes.c_uint32, 12), ('offset', ctypes.c_uint64, 16), ('length', ctypes.c_uint64, 24)])
@c.record
class struct_kgsl_perfcounter_get(c.Struct):
  SIZE = 20
  groupid: 'ctypes.c_uint32'
  countable: 'ctypes.c_uint32'
  offset: 'ctypes.c_uint32'
  offset_hi: 'ctypes.c_uint32'
  __pad: 'ctypes.c_uint32'
struct_kgsl_perfcounter_get.register_fields([('groupid', ctypes.c_uint32, 0), ('countable', ctypes.c_uint32, 4), ('offset', ctypes.c_uint32, 8), ('offset_hi', ctypes.c_uint32, 12), ('__pad', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_perfcounter_put(c.Struct):
  SIZE = 16
  groupid: 'ctypes.c_uint32'
  countable: 'ctypes.c_uint32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[2]]'
struct_kgsl_perfcounter_put.register_fields([('groupid', ctypes.c_uint32, 0), ('countable', ctypes.c_uint32, 4), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_kgsl_perfcounter_query(c.Struct):
  SIZE = 32
  groupid: 'ctypes.c_uint32'
  countables: 'c.POINTER[ctypes.c_uint32]'
  count: 'ctypes.c_uint32'
  max_counters: 'ctypes.c_uint32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[2]]'
struct_kgsl_perfcounter_query.register_fields([('groupid', ctypes.c_uint32, 0), ('countables', c.POINTER[ctypes.c_uint32], 8), ('count', ctypes.c_uint32, 16), ('max_counters', ctypes.c_uint32, 20), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 24)])
@c.record
class struct_kgsl_perfcounter_read_group(c.Struct):
  SIZE = 16
  groupid: 'ctypes.c_uint32'
  countable: 'ctypes.c_uint32'
  value: 'ctypes.c_uint64'
struct_kgsl_perfcounter_read_group.register_fields([('groupid', ctypes.c_uint32, 0), ('countable', ctypes.c_uint32, 4), ('value', ctypes.c_uint64, 8)])
@c.record
class struct_kgsl_perfcounter_read(c.Struct):
  SIZE = 24
  reads: 'c.POINTER[struct_kgsl_perfcounter_read_group]'
  count: 'ctypes.c_uint32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[2]]'
struct_kgsl_perfcounter_read.register_fields([('reads', c.POINTER[struct_kgsl_perfcounter_read_group], 0), ('count', ctypes.c_uint32, 8), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 12)])
@c.record
class struct_kgsl_gpumem_sync_cache_bulk(c.Struct):
  SIZE = 24
  id_list: 'c.POINTER[ctypes.c_uint32]'
  count: 'ctypes.c_uint32'
  op: 'ctypes.c_uint32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[2]]'
struct_kgsl_gpumem_sync_cache_bulk.register_fields([('id_list', c.POINTER[ctypes.c_uint32], 0), ('count', ctypes.c_uint32, 8), ('op', ctypes.c_uint32, 12), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 16)])
@c.record
class struct_kgsl_cmd_syncpoint_timestamp(c.Struct):
  SIZE = 8
  context_id: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
struct_kgsl_cmd_syncpoint_timestamp.register_fields([('context_id', ctypes.c_uint32, 0), ('timestamp', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_cmd_syncpoint_fence(c.Struct):
  SIZE = 4
  fd: 'ctypes.c_int32'
struct_kgsl_cmd_syncpoint_fence.register_fields([('fd', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_cmd_syncpoint(c.Struct):
  SIZE = 24
  type: 'ctypes.c_int32'
  priv: 'ctypes.c_void_p'
  size: 'ctypes.c_uint64'
struct_kgsl_cmd_syncpoint.register_fields([('type', ctypes.c_int32, 0), ('priv', ctypes.c_void_p, 8), ('size', ctypes.c_uint64, 16)])
@c.record
class struct_kgsl_submit_commands(c.Struct):
  SIZE = 56
  context_id: 'ctypes.c_uint32'
  flags: 'ctypes.c_uint32'
  cmdlist: 'c.POINTER[struct_kgsl_ibdesc]'
  numcmds: 'ctypes.c_uint32'
  synclist: 'c.POINTER[struct_kgsl_cmd_syncpoint]'
  numsyncs: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[4]]'
struct_kgsl_submit_commands.register_fields([('context_id', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('cmdlist', c.POINTER[struct_kgsl_ibdesc], 8), ('numcmds', ctypes.c_uint32, 16), ('synclist', c.POINTER[struct_kgsl_cmd_syncpoint], 24), ('numsyncs', ctypes.c_uint32, 32), ('timestamp', ctypes.c_uint32, 36), ('__pad', c.Array[ctypes.c_uint32, Literal[4]], 40)])
@c.record
class struct_kgsl_device_constraint(c.Struct):
  SIZE = 24
  type: 'ctypes.c_uint32'
  context_id: 'ctypes.c_uint32'
  data: 'ctypes.c_void_p'
  size: 'ctypes.c_uint64'
struct_kgsl_device_constraint.register_fields([('type', ctypes.c_uint32, 0), ('context_id', ctypes.c_uint32, 4), ('data', ctypes.c_void_p, 8), ('size', ctypes.c_uint64, 16)])
@c.record
class struct_kgsl_device_constraint_pwrlevel(c.Struct):
  SIZE = 4
  level: 'ctypes.c_uint32'
struct_kgsl_device_constraint_pwrlevel.register_fields([('level', ctypes.c_uint32, 0)])
@c.record
class struct_kgsl_syncsource_create(c.Struct):
  SIZE = 16
  id: 'ctypes.c_uint32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[3]]'
struct_kgsl_syncsource_create.register_fields([('id', ctypes.c_uint32, 0), ('__pad', c.Array[ctypes.c_uint32, Literal[3]], 4)])
@c.record
class struct_kgsl_syncsource_destroy(c.Struct):
  SIZE = 16
  id: 'ctypes.c_uint32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[3]]'
struct_kgsl_syncsource_destroy.register_fields([('id', ctypes.c_uint32, 0), ('__pad', c.Array[ctypes.c_uint32, Literal[3]], 4)])
@c.record
class struct_kgsl_syncsource_create_fence(c.Struct):
  SIZE = 24
  id: 'ctypes.c_uint32'
  fence_fd: 'ctypes.c_int32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[4]]'
struct_kgsl_syncsource_create_fence.register_fields([('id', ctypes.c_uint32, 0), ('fence_fd', ctypes.c_int32, 4), ('__pad', c.Array[ctypes.c_uint32, Literal[4]], 8)])
@c.record
class struct_kgsl_syncsource_signal_fence(c.Struct):
  SIZE = 24
  id: 'ctypes.c_uint32'
  fence_fd: 'ctypes.c_int32'
  __pad: 'c.Array[ctypes.c_uint32, Literal[4]]'
struct_kgsl_syncsource_signal_fence.register_fields([('id', ctypes.c_uint32, 0), ('fence_fd', ctypes.c_int32, 4), ('__pad', c.Array[ctypes.c_uint32, Literal[4]], 8)])
@c.record
class struct_kgsl_cff_sync_gpuobj(c.Struct):
  SIZE = 24
  offset: 'ctypes.c_uint64'
  length: 'ctypes.c_uint64'
  id: 'ctypes.c_uint32'
struct_kgsl_cff_sync_gpuobj.register_fields([('offset', ctypes.c_uint64, 0), ('length', ctypes.c_uint64, 8), ('id', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_gpuobj_alloc(c.Struct):
  SIZE = 48
  size: 'ctypes.c_uint64'
  flags: 'ctypes.c_uint64'
  va_len: 'ctypes.c_uint64'
  mmapsize: 'ctypes.c_uint64'
  id: 'ctypes.c_uint32'
  metadata_len: 'ctypes.c_uint32'
  metadata: 'ctypes.c_uint64'
struct_kgsl_gpuobj_alloc.register_fields([('size', ctypes.c_uint64, 0), ('flags', ctypes.c_uint64, 8), ('va_len', ctypes.c_uint64, 16), ('mmapsize', ctypes.c_uint64, 24), ('id', ctypes.c_uint32, 32), ('metadata_len', ctypes.c_uint32, 36), ('metadata', ctypes.c_uint64, 40)])
@c.record
class struct_kgsl_gpuobj_free(c.Struct):
  SIZE = 32
  flags: 'ctypes.c_uint64'
  priv: 'ctypes.c_uint64'
  id: 'ctypes.c_uint32'
  type: 'ctypes.c_uint32'
  len: 'ctypes.c_uint32'
struct_kgsl_gpuobj_free.register_fields([('flags', ctypes.c_uint64, 0), ('priv', ctypes.c_uint64, 8), ('id', ctypes.c_uint32, 16), ('type', ctypes.c_uint32, 20), ('len', ctypes.c_uint32, 24)])
@c.record
class struct_kgsl_gpu_event_timestamp(c.Struct):
  SIZE = 8
  context_id: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
struct_kgsl_gpu_event_timestamp.register_fields([('context_id', ctypes.c_uint32, 0), ('timestamp', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_gpu_event_fence(c.Struct):
  SIZE = 4
  fd: 'ctypes.c_int32'
struct_kgsl_gpu_event_fence.register_fields([('fd', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_gpuobj_info(c.Struct):
  SIZE = 48
  gpuaddr: 'ctypes.c_uint64'
  flags: 'ctypes.c_uint64'
  size: 'ctypes.c_uint64'
  va_len: 'ctypes.c_uint64'
  va_addr: 'ctypes.c_uint64'
  id: 'ctypes.c_uint32'
struct_kgsl_gpuobj_info.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('flags', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16), ('va_len', ctypes.c_uint64, 24), ('va_addr', ctypes.c_uint64, 32), ('id', ctypes.c_uint32, 40)])
@c.record
class struct_kgsl_gpuobj_import(c.Struct):
  SIZE = 32
  priv: 'ctypes.c_uint64'
  priv_len: 'ctypes.c_uint64'
  flags: 'ctypes.c_uint64'
  type: 'ctypes.c_uint32'
  id: 'ctypes.c_uint32'
struct_kgsl_gpuobj_import.register_fields([('priv', ctypes.c_uint64, 0), ('priv_len', ctypes.c_uint64, 8), ('flags', ctypes.c_uint64, 16), ('type', ctypes.c_uint32, 24), ('id', ctypes.c_uint32, 28)])
@c.record
class struct_kgsl_gpuobj_import_dma_buf(c.Struct):
  SIZE = 4
  fd: 'ctypes.c_int32'
struct_kgsl_gpuobj_import_dma_buf.register_fields([('fd', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_gpuobj_import_useraddr(c.Struct):
  SIZE = 8
  virtaddr: 'ctypes.c_uint64'
struct_kgsl_gpuobj_import_useraddr.register_fields([('virtaddr', ctypes.c_uint64, 0)])
@c.record
class struct_kgsl_gpuobj_sync_obj(c.Struct):
  SIZE = 24
  offset: 'ctypes.c_uint64'
  length: 'ctypes.c_uint64'
  id: 'ctypes.c_uint32'
  op: 'ctypes.c_uint32'
struct_kgsl_gpuobj_sync_obj.register_fields([('offset', ctypes.c_uint64, 0), ('length', ctypes.c_uint64, 8), ('id', ctypes.c_uint32, 16), ('op', ctypes.c_uint32, 20)])
@c.record
class struct_kgsl_gpuobj_sync(c.Struct):
  SIZE = 16
  objs: 'ctypes.c_uint64'
  obj_len: 'ctypes.c_uint32'
  count: 'ctypes.c_uint32'
struct_kgsl_gpuobj_sync.register_fields([('objs', ctypes.c_uint64, 0), ('obj_len', ctypes.c_uint32, 8), ('count', ctypes.c_uint32, 12)])
@c.record
class struct_kgsl_command_object(c.Struct):
  SIZE = 32
  offset: 'ctypes.c_uint64'
  gpuaddr: 'ctypes.c_uint64'
  size: 'ctypes.c_uint64'
  flags: 'ctypes.c_uint32'
  id: 'ctypes.c_uint32'
struct_kgsl_command_object.register_fields([('offset', ctypes.c_uint64, 0), ('gpuaddr', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16), ('flags', ctypes.c_uint32, 24), ('id', ctypes.c_uint32, 28)])
@c.record
class struct_kgsl_command_syncpoint(c.Struct):
  SIZE = 24
  priv: 'ctypes.c_uint64'
  size: 'ctypes.c_uint64'
  type: 'ctypes.c_uint32'
struct_kgsl_command_syncpoint.register_fields([('priv', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('type', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_gpu_command(c.Struct):
  SIZE = 64
  flags: 'ctypes.c_uint64'
  cmdlist: 'ctypes.c_uint64'
  cmdsize: 'ctypes.c_uint32'
  numcmds: 'ctypes.c_uint32'
  objlist: 'ctypes.c_uint64'
  objsize: 'ctypes.c_uint32'
  numobjs: 'ctypes.c_uint32'
  synclist: 'ctypes.c_uint64'
  syncsize: 'ctypes.c_uint32'
  numsyncs: 'ctypes.c_uint32'
  context_id: 'ctypes.c_uint32'
  timestamp: 'ctypes.c_uint32'
struct_kgsl_gpu_command.register_fields([('flags', ctypes.c_uint64, 0), ('cmdlist', ctypes.c_uint64, 8), ('cmdsize', ctypes.c_uint32, 16), ('numcmds', ctypes.c_uint32, 20), ('objlist', ctypes.c_uint64, 24), ('objsize', ctypes.c_uint32, 32), ('numobjs', ctypes.c_uint32, 36), ('synclist', ctypes.c_uint64, 40), ('syncsize', ctypes.c_uint32, 48), ('numsyncs', ctypes.c_uint32, 52), ('context_id', ctypes.c_uint32, 56), ('timestamp', ctypes.c_uint32, 60)])
@c.record
class struct_kgsl_preemption_counters_query(c.Struct):
  SIZE = 24
  counters: 'ctypes.c_uint64'
  size_user: 'ctypes.c_uint32'
  size_priority_level: 'ctypes.c_uint32'
  max_priority_level: 'ctypes.c_uint32'
struct_kgsl_preemption_counters_query.register_fields([('counters', ctypes.c_uint64, 0), ('size_user', ctypes.c_uint32, 8), ('size_priority_level', ctypes.c_uint32, 12), ('max_priority_level', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_gpuobj_set_info(c.Struct):
  SIZE = 32
  flags: 'ctypes.c_uint64'
  metadata: 'ctypes.c_uint64'
  id: 'ctypes.c_uint32'
  metadata_len: 'ctypes.c_uint32'
  type: 'ctypes.c_uint32'
struct_kgsl_gpuobj_set_info.register_fields([('flags', ctypes.c_uint64, 0), ('metadata', ctypes.c_uint64, 8), ('id', ctypes.c_uint32, 16), ('metadata_len', ctypes.c_uint32, 20), ('type', ctypes.c_uint32, 24)])
KGSL_VERSION_MAJOR = 3 # type: ignore
KGSL_VERSION_MINOR = 14 # type: ignore
KGSL_CONTEXT_SAVE_GMEM = 0x00000001 # type: ignore
KGSL_CONTEXT_NO_GMEM_ALLOC = 0x00000002 # type: ignore
KGSL_CONTEXT_SUBMIT_IB_LIST = 0x00000004 # type: ignore
KGSL_CONTEXT_CTX_SWITCH = 0x00000008 # type: ignore
KGSL_CONTEXT_PREAMBLE = 0x00000010 # type: ignore
KGSL_CONTEXT_TRASH_STATE = 0x00000020 # type: ignore
KGSL_CONTEXT_PER_CONTEXT_TS = 0x00000040 # type: ignore
KGSL_CONTEXT_USER_GENERATED_TS = 0x00000080 # type: ignore
KGSL_CONTEXT_END_OF_FRAME = 0x00000100 # type: ignore
KGSL_CONTEXT_NO_FAULT_TOLERANCE = 0x00000200 # type: ignore
KGSL_CONTEXT_SYNC = 0x00000400 # type: ignore
KGSL_CONTEXT_PWR_CONSTRAINT = 0x00000800 # type: ignore
KGSL_CONTEXT_PRIORITY_MASK = 0x0000F000 # type: ignore
KGSL_CONTEXT_PRIORITY_SHIFT = 12 # type: ignore
KGSL_CONTEXT_PRIORITY_UNDEF = 0 # type: ignore
KGSL_CONTEXT_IFH_NOP = 0x00010000 # type: ignore
KGSL_CONTEXT_SECURE = 0x00020000 # type: ignore
KGSL_CONTEXT_PREEMPT_STYLE_MASK = 0x0E000000 # type: ignore
KGSL_CONTEXT_PREEMPT_STYLE_SHIFT = 25 # type: ignore
KGSL_CONTEXT_PREEMPT_STYLE_DEFAULT = 0x0 # type: ignore
KGSL_CONTEXT_PREEMPT_STYLE_RINGBUFFER = 0x1 # type: ignore
KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN = 0x2 # type: ignore
KGSL_CONTEXT_TYPE_MASK = 0x01F00000 # type: ignore
KGSL_CONTEXT_TYPE_SHIFT = 20 # type: ignore
KGSL_CONTEXT_TYPE_ANY = 0 # type: ignore
KGSL_CONTEXT_TYPE_GL = 1 # type: ignore
KGSL_CONTEXT_TYPE_CL = 2 # type: ignore
KGSL_CONTEXT_TYPE_C2D = 3 # type: ignore
KGSL_CONTEXT_TYPE_RS = 4 # type: ignore
KGSL_CONTEXT_TYPE_UNKNOWN = 0x1E # type: ignore
KGSL_CONTEXT_INVALID = 0xffffffff # type: ignore
KGSL_CMDBATCH_MEMLIST = 0x00000001 # type: ignore
KGSL_CMDBATCH_MARKER = 0x00000002 # type: ignore
KGSL_CMDBATCH_SUBMIT_IB_LIST = KGSL_CONTEXT_SUBMIT_IB_LIST # type: ignore
KGSL_CMDBATCH_CTX_SWITCH = KGSL_CONTEXT_CTX_SWITCH # type: ignore
KGSL_CMDBATCH_PROFILING = 0x00000010 # type: ignore
KGSL_CMDBATCH_PROFILING_KTIME = 0x00000020 # type: ignore
KGSL_CMDBATCH_END_OF_FRAME = KGSL_CONTEXT_END_OF_FRAME # type: ignore
KGSL_CMDBATCH_SYNC = KGSL_CONTEXT_SYNC # type: ignore
KGSL_CMDBATCH_PWR_CONSTRAINT = KGSL_CONTEXT_PWR_CONSTRAINT # type: ignore
KGSL_CMDLIST_IB = 0x00000001 # type: ignore
KGSL_CMDLIST_CTXTSWITCH_PREAMBLE = 0x00000002 # type: ignore
KGSL_CMDLIST_IB_PREAMBLE = 0x00000004 # type: ignore
KGSL_OBJLIST_MEMOBJ = 0x00000008 # type: ignore
KGSL_OBJLIST_PROFILE = 0x00000010 # type: ignore
KGSL_CMD_SYNCPOINT_TYPE_TIMESTAMP = 0 # type: ignore
KGSL_CMD_SYNCPOINT_TYPE_FENCE = 1 # type: ignore
KGSL_MEMFLAGS_SECURE = 0x00000008 # type: ignore
KGSL_MEMFLAGS_GPUREADONLY = 0x01000000 # type: ignore
KGSL_MEMFLAGS_GPUWRITEONLY = 0x02000000 # type: ignore
KGSL_MEMFLAGS_FORCE_32BIT = 0x100000000 # type: ignore
KGSL_CACHEMODE_MASK = 0x0C000000 # type: ignore
KGSL_CACHEMODE_SHIFT = 26 # type: ignore
KGSL_CACHEMODE_WRITECOMBINE = 0 # type: ignore
KGSL_CACHEMODE_UNCACHED = 1 # type: ignore
KGSL_CACHEMODE_WRITETHROUGH = 2 # type: ignore
KGSL_CACHEMODE_WRITEBACK = 3 # type: ignore
KGSL_MEMFLAGS_USE_CPU_MAP = 0x10000000 # type: ignore
KGSL_MEMTYPE_MASK = 0x0000FF00 # type: ignore
KGSL_MEMTYPE_SHIFT = 8 # type: ignore
KGSL_MEMTYPE_OBJECTANY = 0 # type: ignore
KGSL_MEMTYPE_FRAMEBUFFER = 1 # type: ignore
KGSL_MEMTYPE_RENDERBUFFER = 2 # type: ignore
KGSL_MEMTYPE_ARRAYBUFFER = 3 # type: ignore
KGSL_MEMTYPE_ELEMENTARRAYBUFFER = 4 # type: ignore
KGSL_MEMTYPE_VERTEXARRAYBUFFER = 5 # type: ignore
KGSL_MEMTYPE_TEXTURE = 6 # type: ignore
KGSL_MEMTYPE_SURFACE = 7 # type: ignore
KGSL_MEMTYPE_EGL_SURFACE = 8 # type: ignore
KGSL_MEMTYPE_GL = 9 # type: ignore
KGSL_MEMTYPE_CL = 10 # type: ignore
KGSL_MEMTYPE_CL_BUFFER_MAP = 11 # type: ignore
KGSL_MEMTYPE_CL_BUFFER_NOMAP = 12 # type: ignore
KGSL_MEMTYPE_CL_IMAGE_MAP = 13 # type: ignore
KGSL_MEMTYPE_CL_IMAGE_NOMAP = 14 # type: ignore
KGSL_MEMTYPE_CL_KERNEL_STACK = 15 # type: ignore
KGSL_MEMTYPE_COMMAND = 16 # type: ignore
KGSL_MEMTYPE_2D = 17 # type: ignore
KGSL_MEMTYPE_EGL_IMAGE = 18 # type: ignore
KGSL_MEMTYPE_EGL_SHADOW = 19 # type: ignore
KGSL_MEMTYPE_MULTISAMPLE = 20 # type: ignore
KGSL_MEMTYPE_KERNEL = 255 # type: ignore
KGSL_MEMALIGN_MASK = 0x00FF0000 # type: ignore
KGSL_MEMALIGN_SHIFT = 16 # type: ignore
KGSL_MEMFLAGS_USERMEM_MASK = 0x000000e0 # type: ignore
KGSL_MEMFLAGS_USERMEM_SHIFT = 5 # type: ignore
KGSL_USERMEM_FLAG = lambda x: (((x) + 1) << KGSL_MEMFLAGS_USERMEM_SHIFT) # type: ignore
KGSL_MEMFLAGS_NOT_USERMEM = 0 # type: ignore
KGSL_MEMFLAGS_USERMEM_PMEM = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_PMEM) # type: ignore
KGSL_MEMFLAGS_USERMEM_ASHMEM = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ASHMEM) # type: ignore
KGSL_MEMFLAGS_USERMEM_ADDR = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ADDR) # type: ignore
KGSL_MEMFLAGS_USERMEM_ION = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ION) # type: ignore
KGSL_FLAGS_NORMALMODE = 0x00000000 # type: ignore
KGSL_FLAGS_SAFEMODE = 0x00000001 # type: ignore
KGSL_FLAGS_INITIALIZED0 = 0x00000002 # type: ignore
KGSL_FLAGS_INITIALIZED = 0x00000004 # type: ignore
KGSL_FLAGS_STARTED = 0x00000008 # type: ignore
KGSL_FLAGS_ACTIVE = 0x00000010 # type: ignore
KGSL_FLAGS_RESERVED0 = 0x00000020 # type: ignore
KGSL_FLAGS_RESERVED1 = 0x00000040 # type: ignore
KGSL_FLAGS_RESERVED2 = 0x00000080 # type: ignore
KGSL_FLAGS_SOFT_RESET = 0x00000100 # type: ignore
KGSL_FLAGS_PER_CONTEXT_TIMESTAMPS = 0x00000200 # type: ignore
KGSL_SYNCOBJ_SERVER_TIMEOUT = 2000 # type: ignore
KGSL_CONVERT_TO_MBPS = lambda val: (val*1000*1000) # type: ignore
KGSL_MEMSTORE_OFFSET = lambda ctxt_id,field: ((ctxt_id)*sizeof(struct_kgsl_devmemstore) + offsetof(struct_kgsl_devmemstore, field)) # type: ignore
KGSL_PROP_DEVICE_INFO = 0x1 # type: ignore
KGSL_PROP_DEVICE_SHADOW = 0x2 # type: ignore
KGSL_PROP_DEVICE_POWER = 0x3 # type: ignore
KGSL_PROP_SHMEM = 0x4 # type: ignore
KGSL_PROP_SHMEM_APERTURES = 0x5 # type: ignore
KGSL_PROP_MMU_ENABLE = 0x6 # type: ignore
KGSL_PROP_INTERRUPT_WAITS = 0x7 # type: ignore
KGSL_PROP_VERSION = 0x8 # type: ignore
KGSL_PROP_GPU_RESET_STAT = 0x9 # type: ignore
KGSL_PROP_PWRCTRL = 0xE # type: ignore
KGSL_PROP_PWR_CONSTRAINT = 0x12 # type: ignore
KGSL_PROP_UCHE_GMEM_VADDR = 0x13 # type: ignore
KGSL_PROP_SP_GENERIC_MEM = 0x14 # type: ignore
KGSL_PROP_UCODE_VERSION = 0x15 # type: ignore
KGSL_PROP_GPMU_VERSION = 0x16 # type: ignore
KGSL_PROP_DEVICE_BITNESS = 0x18 # type: ignore
KGSL_PERFCOUNTER_GROUP_CP = 0x0 # type: ignore
KGSL_PERFCOUNTER_GROUP_RBBM = 0x1 # type: ignore
KGSL_PERFCOUNTER_GROUP_PC = 0x2 # type: ignore
KGSL_PERFCOUNTER_GROUP_VFD = 0x3 # type: ignore
KGSL_PERFCOUNTER_GROUP_HLSQ = 0x4 # type: ignore
KGSL_PERFCOUNTER_GROUP_VPC = 0x5 # type: ignore
KGSL_PERFCOUNTER_GROUP_TSE = 0x6 # type: ignore
KGSL_PERFCOUNTER_GROUP_RAS = 0x7 # type: ignore
KGSL_PERFCOUNTER_GROUP_UCHE = 0x8 # type: ignore
KGSL_PERFCOUNTER_GROUP_TP = 0x9 # type: ignore
KGSL_PERFCOUNTER_GROUP_SP = 0xA # type: ignore
KGSL_PERFCOUNTER_GROUP_RB = 0xB # type: ignore
KGSL_PERFCOUNTER_GROUP_PWR = 0xC # type: ignore
KGSL_PERFCOUNTER_GROUP_VBIF = 0xD # type: ignore
KGSL_PERFCOUNTER_GROUP_VBIF_PWR = 0xE # type: ignore
KGSL_PERFCOUNTER_GROUP_MH = 0xF # type: ignore
KGSL_PERFCOUNTER_GROUP_PA_SU = 0x10 # type: ignore
KGSL_PERFCOUNTER_GROUP_SQ = 0x11 # type: ignore
KGSL_PERFCOUNTER_GROUP_SX = 0x12 # type: ignore
KGSL_PERFCOUNTER_GROUP_TCF = 0x13 # type: ignore
KGSL_PERFCOUNTER_GROUP_TCM = 0x14 # type: ignore
KGSL_PERFCOUNTER_GROUP_TCR = 0x15 # type: ignore
KGSL_PERFCOUNTER_GROUP_L2 = 0x16 # type: ignore
KGSL_PERFCOUNTER_GROUP_VSC = 0x17 # type: ignore
KGSL_PERFCOUNTER_GROUP_CCU = 0x18 # type: ignore
KGSL_PERFCOUNTER_GROUP_LRZ = 0x19 # type: ignore
KGSL_PERFCOUNTER_GROUP_CMP = 0x1A # type: ignore
KGSL_PERFCOUNTER_GROUP_ALWAYSON = 0x1B # type: ignore
KGSL_PERFCOUNTER_GROUP_SP_PWR = 0x1C # type: ignore
KGSL_PERFCOUNTER_GROUP_TP_PWR = 0x1D # type: ignore
KGSL_PERFCOUNTER_GROUP_RB_PWR = 0x1E # type: ignore
KGSL_PERFCOUNTER_GROUP_CCU_PWR = 0x1F # type: ignore
KGSL_PERFCOUNTER_GROUP_UCHE_PWR = 0x20 # type: ignore
KGSL_PERFCOUNTER_GROUP_CP_PWR = 0x21 # type: ignore
KGSL_PERFCOUNTER_GROUP_GPMU_PWR = 0x22 # type: ignore
KGSL_PERFCOUNTER_GROUP_ALWAYSON_PWR = 0x23 # type: ignore
KGSL_PERFCOUNTER_GROUP_MAX = 0x24 # type: ignore
KGSL_PERFCOUNTER_NOT_USED = 0xFFFFFFFF # type: ignore
KGSL_PERFCOUNTER_BROKEN = 0xFFFFFFFE # type: ignore
KGSL_IOC_TYPE = 0x09 # type: ignore
IOCTL_KGSL_DEVICE_GETPROPERTY = _IOWR(KGSL_IOC_TYPE, 0x2, struct_kgsl_device_getproperty) # type: ignore
IOCTL_KGSL_DEVICE_WAITTIMESTAMP = _IOW(KGSL_IOC_TYPE, 0x6, struct_kgsl_device_waittimestamp) # type: ignore
IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID = _IOW(KGSL_IOC_TYPE, 0x7, struct_kgsl_device_waittimestamp_ctxtid) # type: ignore
IOCTL_KGSL_RINGBUFFER_ISSUEIBCMDS = _IOWR(KGSL_IOC_TYPE, 0x10, struct_kgsl_ringbuffer_issueibcmds) # type: ignore
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_OLD = _IOR(KGSL_IOC_TYPE, 0x11, struct_kgsl_cmdstream_readtimestamp) # type: ignore
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP = _IOWR(KGSL_IOC_TYPE, 0x11, struct_kgsl_cmdstream_readtimestamp) # type: ignore
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP = _IOW(KGSL_IOC_TYPE, 0x12, struct_kgsl_cmdstream_freememontimestamp) # type: ignore
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_OLD = _IOR(KGSL_IOC_TYPE, 0x12, struct_kgsl_cmdstream_freememontimestamp) # type: ignore
IOCTL_KGSL_DRAWCTXT_CREATE = _IOWR(KGSL_IOC_TYPE, 0x13, struct_kgsl_drawctxt_create) # type: ignore
IOCTL_KGSL_DRAWCTXT_DESTROY = _IOW(KGSL_IOC_TYPE, 0x14, struct_kgsl_drawctxt_destroy) # type: ignore
IOCTL_KGSL_MAP_USER_MEM = _IOWR(KGSL_IOC_TYPE, 0x15, struct_kgsl_map_user_mem) # type: ignore
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID = _IOWR(KGSL_IOC_TYPE, 0x16, struct_kgsl_cmdstream_readtimestamp_ctxtid) # type: ignore
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_CTXTID = _IOW(KGSL_IOC_TYPE, 0x17, struct_kgsl_cmdstream_freememontimestamp_ctxtid) # type: ignore
IOCTL_KGSL_SHAREDMEM_FROM_PMEM = _IOWR(KGSL_IOC_TYPE, 0x20, struct_kgsl_sharedmem_from_pmem) # type: ignore
IOCTL_KGSL_SHAREDMEM_FREE = _IOW(KGSL_IOC_TYPE, 0x21, struct_kgsl_sharedmem_free) # type: ignore
IOCTL_KGSL_CFF_USER_EVENT = _IOW(KGSL_IOC_TYPE, 0x31, struct_kgsl_cff_user_event) # type: ignore
IOCTL_KGSL_DRAWCTXT_BIND_GMEM_SHADOW = _IOW(KGSL_IOC_TYPE, 0x22, struct_kgsl_bind_gmem_shadow) # type: ignore
IOCTL_KGSL_SHAREDMEM_FROM_VMALLOC = _IOWR(KGSL_IOC_TYPE, 0x23, struct_kgsl_sharedmem_from_vmalloc) # type: ignore
IOCTL_KGSL_SHAREDMEM_FLUSH_CACHE = _IOW(KGSL_IOC_TYPE, 0x24, struct_kgsl_sharedmem_free) # type: ignore
IOCTL_KGSL_DRAWCTXT_SET_BIN_BASE_OFFSET = _IOW(KGSL_IOC_TYPE, 0x25, struct_kgsl_drawctxt_set_bin_base_offset) # type: ignore
IOCTL_KGSL_CMDWINDOW_WRITE = _IOW(KGSL_IOC_TYPE, 0x2e, struct_kgsl_cmdwindow_write) # type: ignore
IOCTL_KGSL_GPUMEM_ALLOC = _IOWR(KGSL_IOC_TYPE, 0x2f, struct_kgsl_gpumem_alloc) # type: ignore
IOCTL_KGSL_CFF_SYNCMEM = _IOW(KGSL_IOC_TYPE, 0x30, struct_kgsl_cff_syncmem) # type: ignore
IOCTL_KGSL_TIMESTAMP_EVENT_OLD = _IOW(KGSL_IOC_TYPE, 0x31, struct_kgsl_timestamp_event) # type: ignore
KGSL_TIMESTAMP_EVENT_GENLOCK = 1 # type: ignore
KGSL_TIMESTAMP_EVENT_FENCE = 2 # type: ignore
IOCTL_KGSL_SETPROPERTY = _IOW(KGSL_IOC_TYPE, 0x32, struct_kgsl_device_getproperty) # type: ignore
IOCTL_KGSL_TIMESTAMP_EVENT = _IOWR(KGSL_IOC_TYPE, 0x33, struct_kgsl_timestamp_event) # type: ignore
IOCTL_KGSL_GPUMEM_ALLOC_ID = _IOWR(KGSL_IOC_TYPE, 0x34, struct_kgsl_gpumem_alloc_id) # type: ignore
IOCTL_KGSL_GPUMEM_FREE_ID = _IOWR(KGSL_IOC_TYPE, 0x35, struct_kgsl_gpumem_free_id) # type: ignore
IOCTL_KGSL_GPUMEM_GET_INFO = _IOWR(KGSL_IOC_TYPE, 0x36, struct_kgsl_gpumem_get_info) # type: ignore
KGSL_GPUMEM_CACHE_CLEAN = (1 << 0) # type: ignore
KGSL_GPUMEM_CACHE_TO_GPU = KGSL_GPUMEM_CACHE_CLEAN # type: ignore
KGSL_GPUMEM_CACHE_INV = (1 << 1) # type: ignore
KGSL_GPUMEM_CACHE_FROM_GPU = KGSL_GPUMEM_CACHE_INV # type: ignore
KGSL_GPUMEM_CACHE_FLUSH = (KGSL_GPUMEM_CACHE_CLEAN | KGSL_GPUMEM_CACHE_INV) # type: ignore
KGSL_GPUMEM_CACHE_RANGE = (1 << 31) # type: ignore
IOCTL_KGSL_GPUMEM_SYNC_CACHE = _IOW(KGSL_IOC_TYPE, 0x37, struct_kgsl_gpumem_sync_cache) # type: ignore
IOCTL_KGSL_PERFCOUNTER_GET = _IOWR(KGSL_IOC_TYPE, 0x38, struct_kgsl_perfcounter_get) # type: ignore
IOCTL_KGSL_PERFCOUNTER_PUT = _IOW(KGSL_IOC_TYPE, 0x39, struct_kgsl_perfcounter_put) # type: ignore
IOCTL_KGSL_PERFCOUNTER_QUERY = _IOWR(KGSL_IOC_TYPE, 0x3A, struct_kgsl_perfcounter_query) # type: ignore
IOCTL_KGSL_PERFCOUNTER_READ = _IOWR(KGSL_IOC_TYPE, 0x3B, struct_kgsl_perfcounter_read) # type: ignore
IOCTL_KGSL_GPUMEM_SYNC_CACHE_BULK = _IOWR(KGSL_IOC_TYPE, 0x3C, struct_kgsl_gpumem_sync_cache_bulk) # type: ignore
KGSL_IBDESC_MEMLIST = 0x1 # type: ignore
KGSL_IBDESC_PROFILING_BUFFER = 0x2 # type: ignore
IOCTL_KGSL_SUBMIT_COMMANDS = _IOWR(KGSL_IOC_TYPE, 0x3D, struct_kgsl_submit_commands) # type: ignore
KGSL_CONSTRAINT_NONE = 0 # type: ignore
KGSL_CONSTRAINT_PWRLEVEL = 1 # type: ignore
KGSL_CONSTRAINT_PWR_MIN = 0 # type: ignore
KGSL_CONSTRAINT_PWR_MAX = 1 # type: ignore
IOCTL_KGSL_SYNCSOURCE_CREATE = _IOWR(KGSL_IOC_TYPE, 0x40, struct_kgsl_syncsource_create) # type: ignore
IOCTL_KGSL_SYNCSOURCE_DESTROY = _IOWR(KGSL_IOC_TYPE, 0x41, struct_kgsl_syncsource_destroy) # type: ignore
IOCTL_KGSL_SYNCSOURCE_CREATE_FENCE = _IOWR(KGSL_IOC_TYPE, 0x42, struct_kgsl_syncsource_create_fence) # type: ignore
IOCTL_KGSL_SYNCSOURCE_SIGNAL_FENCE = _IOWR(KGSL_IOC_TYPE, 0x43, struct_kgsl_syncsource_signal_fence) # type: ignore
IOCTL_KGSL_CFF_SYNC_GPUOBJ = _IOW(KGSL_IOC_TYPE, 0x44, struct_kgsl_cff_sync_gpuobj) # type: ignore
KGSL_GPUOBJ_ALLOC_METADATA_MAX = 64 # type: ignore
IOCTL_KGSL_GPUOBJ_ALLOC = _IOWR(KGSL_IOC_TYPE, 0x45, struct_kgsl_gpuobj_alloc) # type: ignore
KGSL_GPUOBJ_FREE_ON_EVENT = 1 # type: ignore
KGSL_GPU_EVENT_TIMESTAMP = 1 # type: ignore
KGSL_GPU_EVENT_FENCE = 2 # type: ignore
IOCTL_KGSL_GPUOBJ_FREE = _IOW(KGSL_IOC_TYPE, 0x46, struct_kgsl_gpuobj_free) # type: ignore
IOCTL_KGSL_GPUOBJ_INFO = _IOWR(KGSL_IOC_TYPE, 0x47, struct_kgsl_gpuobj_info) # type: ignore
IOCTL_KGSL_GPUOBJ_IMPORT = _IOWR(KGSL_IOC_TYPE, 0x48, struct_kgsl_gpuobj_import) # type: ignore
IOCTL_KGSL_GPUOBJ_SYNC = _IOW(KGSL_IOC_TYPE, 0x49, struct_kgsl_gpuobj_sync) # type: ignore
IOCTL_KGSL_GPU_COMMAND = _IOWR(KGSL_IOC_TYPE, 0x4A, struct_kgsl_gpu_command) # type: ignore
IOCTL_KGSL_PREEMPTIONCOUNTER_QUERY = _IOWR(KGSL_IOC_TYPE, 0x4B, struct_kgsl_preemption_counters_query) # type: ignore
KGSL_GPUOBJ_SET_INFO_METADATA = (1 << 0) # type: ignore
KGSL_GPUOBJ_SET_INFO_TYPE = (1 << 1) # type: ignore
IOCTL_KGSL_GPUOBJ_SET_INFO = _IOW(KGSL_IOC_TYPE, 0x4C, struct_kgsl_gpuobj_set_info) # type: ignore