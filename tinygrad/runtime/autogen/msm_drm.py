# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
drm_handle_t: TypeAlias = Annotated[int, ctypes.c_uint32]
drm_context_t: TypeAlias = Annotated[int, ctypes.c_uint32]
drm_drawable_t: TypeAlias = Annotated[int, ctypes.c_uint32]
drm_magic_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_drm_clip_rect(c.Struct):
  SIZE = 8
  x1: Annotated[Annotated[int, ctypes.c_uint16], 0]
  y1: Annotated[Annotated[int, ctypes.c_uint16], 2]
  x2: Annotated[Annotated[int, ctypes.c_uint16], 4]
  y2: Annotated[Annotated[int, ctypes.c_uint16], 6]
@c.record
class struct_drm_drawable_info(c.Struct):
  SIZE = 16
  num_rects: Annotated[Annotated[int, ctypes.c_uint32], 0]
  rects: Annotated[c.POINTER[struct_drm_clip_rect], 8]
@c.record
class struct_drm_tex_region(c.Struct):
  SIZE = 8
  next: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  prev: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  in_use: Annotated[Annotated[int, ctypes.c_ubyte], 2]
  padding: Annotated[Annotated[int, ctypes.c_ubyte], 3]
  age: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_hw_lock(c.Struct):
  SIZE = 64
  lock: Annotated[Annotated[int, ctypes.c_uint32], 0]
  padding: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[60]], 4]
@c.record
class struct_drm_version(c.Struct):
  SIZE = 64
  version_major: Annotated[Annotated[int, ctypes.c_int32], 0]
  version_minor: Annotated[Annotated[int, ctypes.c_int32], 4]
  version_patchlevel: Annotated[Annotated[int, ctypes.c_int32], 8]
  name_len: Annotated[Annotated[int, ctypes.c_uint64], 16]
  name: Annotated[c.POINTER[Annotated[int, ctypes.c_ubyte]], 24]
  date_len: Annotated[Annotated[int, ctypes.c_uint64], 32]
  date: Annotated[c.POINTER[Annotated[int, ctypes.c_ubyte]], 40]
  desc_len: Annotated[Annotated[int, ctypes.c_uint64], 48]
  desc: Annotated[c.POINTER[Annotated[int, ctypes.c_ubyte]], 56]
__kernel_size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_drm_unique(c.Struct):
  SIZE = 16
  unique_len: Annotated[Annotated[int, ctypes.c_uint64], 0]
  unique: Annotated[c.POINTER[Annotated[int, ctypes.c_ubyte]], 8]
@c.record
class struct_drm_list(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  version: Annotated[c.POINTER[struct_drm_version], 8]
@c.record
class struct_drm_block(c.Struct):
  SIZE = 4
  unused: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_drm_control(c.Struct):
  SIZE = 8
  func: Annotated[struct_drm_control_func, 0]
  irq: Annotated[Annotated[int, ctypes.c_int32], 4]
class struct_drm_control_func(Annotated[int, ctypes.c_uint32], c.Enum): pass
DRM_ADD_COMMAND = struct_drm_control_func.define('DRM_ADD_COMMAND', 0)
DRM_RM_COMMAND = struct_drm_control_func.define('DRM_RM_COMMAND', 1)
DRM_INST_HANDLER = struct_drm_control_func.define('DRM_INST_HANDLER', 2)
DRM_UNINST_HANDLER = struct_drm_control_func.define('DRM_UNINST_HANDLER', 3)

class enum_drm_map_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_FRAME_BUFFER = enum_drm_map_type.define('_DRM_FRAME_BUFFER', 0)
_DRM_REGISTERS = enum_drm_map_type.define('_DRM_REGISTERS', 1)
_DRM_SHM = enum_drm_map_type.define('_DRM_SHM', 2)
_DRM_AGP = enum_drm_map_type.define('_DRM_AGP', 3)
_DRM_SCATTER_GATHER = enum_drm_map_type.define('_DRM_SCATTER_GATHER', 4)
_DRM_CONSISTENT = enum_drm_map_type.define('_DRM_CONSISTENT', 5)

class enum_drm_map_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_RESTRICTED = enum_drm_map_flags.define('_DRM_RESTRICTED', 1)
_DRM_READ_ONLY = enum_drm_map_flags.define('_DRM_READ_ONLY', 2)
_DRM_LOCKED = enum_drm_map_flags.define('_DRM_LOCKED', 4)
_DRM_KERNEL = enum_drm_map_flags.define('_DRM_KERNEL', 8)
_DRM_WRITE_COMBINING = enum_drm_map_flags.define('_DRM_WRITE_COMBINING', 16)
_DRM_CONTAINS_LOCK = enum_drm_map_flags.define('_DRM_CONTAINS_LOCK', 32)
_DRM_REMOVABLE = enum_drm_map_flags.define('_DRM_REMOVABLE', 64)
_DRM_DRIVER = enum_drm_map_flags.define('_DRM_DRIVER', 128)

@c.record
class struct_drm_ctx_priv_map(c.Struct):
  SIZE = 16
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  handle: Annotated[ctypes.c_void_p, 8]
@c.record
class struct_drm_map(c.Struct):
  SIZE = 40
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  type: Annotated[enum_drm_map_type, 16]
  flags: Annotated[enum_drm_map_flags, 20]
  handle: Annotated[ctypes.c_void_p, 24]
  mtrr: Annotated[Annotated[int, ctypes.c_int32], 32]
@c.record
class struct_drm_client(c.Struct):
  SIZE = 40
  idx: Annotated[Annotated[int, ctypes.c_int32], 0]
  auth: Annotated[Annotated[int, ctypes.c_int32], 4]
  pid: Annotated[Annotated[int, ctypes.c_uint64], 8]
  uid: Annotated[Annotated[int, ctypes.c_uint64], 16]
  magic: Annotated[Annotated[int, ctypes.c_uint64], 24]
  iocs: Annotated[Annotated[int, ctypes.c_uint64], 32]
class enum_drm_stat_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_STAT_LOCK = enum_drm_stat_type.define('_DRM_STAT_LOCK', 0)
_DRM_STAT_OPENS = enum_drm_stat_type.define('_DRM_STAT_OPENS', 1)
_DRM_STAT_CLOSES = enum_drm_stat_type.define('_DRM_STAT_CLOSES', 2)
_DRM_STAT_IOCTLS = enum_drm_stat_type.define('_DRM_STAT_IOCTLS', 3)
_DRM_STAT_LOCKS = enum_drm_stat_type.define('_DRM_STAT_LOCKS', 4)
_DRM_STAT_UNLOCKS = enum_drm_stat_type.define('_DRM_STAT_UNLOCKS', 5)
_DRM_STAT_VALUE = enum_drm_stat_type.define('_DRM_STAT_VALUE', 6)
_DRM_STAT_BYTE = enum_drm_stat_type.define('_DRM_STAT_BYTE', 7)
_DRM_STAT_COUNT = enum_drm_stat_type.define('_DRM_STAT_COUNT', 8)
_DRM_STAT_IRQ = enum_drm_stat_type.define('_DRM_STAT_IRQ', 9)
_DRM_STAT_PRIMARY = enum_drm_stat_type.define('_DRM_STAT_PRIMARY', 10)
_DRM_STAT_SECONDARY = enum_drm_stat_type.define('_DRM_STAT_SECONDARY', 11)
_DRM_STAT_DMA = enum_drm_stat_type.define('_DRM_STAT_DMA', 12)
_DRM_STAT_SPECIAL = enum_drm_stat_type.define('_DRM_STAT_SPECIAL', 13)
_DRM_STAT_MISSED = enum_drm_stat_type.define('_DRM_STAT_MISSED', 14)

@c.record
class struct_drm_stats(c.Struct):
  SIZE = 248
  count: Annotated[Annotated[int, ctypes.c_uint64], 0]
  data: Annotated[c.Array[struct_drm_stats_data, Literal[15]], 8]
@c.record
class struct_drm_stats_data(c.Struct):
  SIZE = 16
  value: Annotated[Annotated[int, ctypes.c_uint64], 0]
  type: Annotated[enum_drm_stat_type, 8]
class enum_drm_lock_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_LOCK_READY = enum_drm_lock_flags.define('_DRM_LOCK_READY', 1)
_DRM_LOCK_QUIESCENT = enum_drm_lock_flags.define('_DRM_LOCK_QUIESCENT', 2)
_DRM_LOCK_FLUSH = enum_drm_lock_flags.define('_DRM_LOCK_FLUSH', 4)
_DRM_LOCK_FLUSH_ALL = enum_drm_lock_flags.define('_DRM_LOCK_FLUSH_ALL', 8)
_DRM_HALT_ALL_QUEUES = enum_drm_lock_flags.define('_DRM_HALT_ALL_QUEUES', 16)
_DRM_HALT_CUR_QUEUES = enum_drm_lock_flags.define('_DRM_HALT_CUR_QUEUES', 32)

@c.record
class struct_drm_lock(c.Struct):
  SIZE = 8
  context: Annotated[Annotated[int, ctypes.c_int32], 0]
  flags: Annotated[enum_drm_lock_flags, 4]
class enum_drm_dma_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_DMA_BLOCK = enum_drm_dma_flags.define('_DRM_DMA_BLOCK', 1)
_DRM_DMA_WHILE_LOCKED = enum_drm_dma_flags.define('_DRM_DMA_WHILE_LOCKED', 2)
_DRM_DMA_PRIORITY = enum_drm_dma_flags.define('_DRM_DMA_PRIORITY', 4)
_DRM_DMA_WAIT = enum_drm_dma_flags.define('_DRM_DMA_WAIT', 16)
_DRM_DMA_SMALLER_OK = enum_drm_dma_flags.define('_DRM_DMA_SMALLER_OK', 32)
_DRM_DMA_LARGER_OK = enum_drm_dma_flags.define('_DRM_DMA_LARGER_OK', 64)

@c.record
class struct_drm_buf_desc(c.Struct):
  SIZE = 32
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  size: Annotated[Annotated[int, ctypes.c_int32], 4]
  low_mark: Annotated[Annotated[int, ctypes.c_int32], 8]
  high_mark: Annotated[Annotated[int, ctypes.c_int32], 12]
  flags: Annotated[struct_drm_buf_desc_flags, 16]
  agp_start: Annotated[Annotated[int, ctypes.c_uint64], 24]
class struct_drm_buf_desc_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_PAGE_ALIGN = struct_drm_buf_desc_flags.define('_DRM_PAGE_ALIGN', 1)
_DRM_AGP_BUFFER = struct_drm_buf_desc_flags.define('_DRM_AGP_BUFFER', 2)
_DRM_SG_BUFFER = struct_drm_buf_desc_flags.define('_DRM_SG_BUFFER', 4)
_DRM_FB_BUFFER = struct_drm_buf_desc_flags.define('_DRM_FB_BUFFER', 8)
_DRM_PCI_BUFFER_RO = struct_drm_buf_desc_flags.define('_DRM_PCI_BUFFER_RO', 16)

@c.record
class struct_drm_buf_info(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  list: Annotated[c.POINTER[struct_drm_buf_desc], 8]
@c.record
class struct_drm_buf_free(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  list: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 8]
@c.record
class struct_drm_buf_pub(c.Struct):
  SIZE = 24
  idx: Annotated[Annotated[int, ctypes.c_int32], 0]
  total: Annotated[Annotated[int, ctypes.c_int32], 4]
  used: Annotated[Annotated[int, ctypes.c_int32], 8]
  address: Annotated[ctypes.c_void_p, 16]
@c.record
class struct_drm_buf_map(c.Struct):
  SIZE = 24
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  virtual: Annotated[ctypes.c_void_p, 8]
  list: Annotated[c.POINTER[struct_drm_buf_pub], 16]
@c.record
class struct_drm_dma(c.Struct):
  SIZE = 64
  context: Annotated[Annotated[int, ctypes.c_int32], 0]
  send_count: Annotated[Annotated[int, ctypes.c_int32], 4]
  send_indices: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 8]
  send_sizes: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 16]
  flags: Annotated[enum_drm_dma_flags, 24]
  request_count: Annotated[Annotated[int, ctypes.c_int32], 28]
  request_size: Annotated[Annotated[int, ctypes.c_int32], 32]
  request_indices: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 40]
  request_sizes: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 48]
  granted_count: Annotated[Annotated[int, ctypes.c_int32], 56]
class enum_drm_ctx_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_CONTEXT_PRESERVED = enum_drm_ctx_flags.define('_DRM_CONTEXT_PRESERVED', 1)
_DRM_CONTEXT_2DONLY = enum_drm_ctx_flags.define('_DRM_CONTEXT_2DONLY', 2)

@c.record
class struct_drm_ctx(c.Struct):
  SIZE = 8
  handle: Annotated[drm_context_t, 0]
  flags: Annotated[enum_drm_ctx_flags, 4]
@c.record
class struct_drm_ctx_res(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  contexts: Annotated[c.POINTER[struct_drm_ctx], 8]
@c.record
class struct_drm_draw(c.Struct):
  SIZE = 4
  handle: Annotated[drm_drawable_t, 0]
class drm_drawable_info_type_t(Annotated[int, ctypes.c_uint32], c.Enum): pass
DRM_DRAWABLE_CLIPRECTS = drm_drawable_info_type_t.define('DRM_DRAWABLE_CLIPRECTS', 0)

@c.record
class struct_drm_update_draw(c.Struct):
  SIZE = 24
  handle: Annotated[drm_drawable_t, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 4]
  num: Annotated[Annotated[int, ctypes.c_uint32], 8]
  data: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_auth(c.Struct):
  SIZE = 4
  magic: Annotated[drm_magic_t, 0]
@c.record
class struct_drm_irq_busid(c.Struct):
  SIZE = 16
  irq: Annotated[Annotated[int, ctypes.c_int32], 0]
  busnum: Annotated[Annotated[int, ctypes.c_int32], 4]
  devnum: Annotated[Annotated[int, ctypes.c_int32], 8]
  funcnum: Annotated[Annotated[int, ctypes.c_int32], 12]
class enum_drm_vblank_seq_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_VBLANK_ABSOLUTE = enum_drm_vblank_seq_type.define('_DRM_VBLANK_ABSOLUTE', 0)
_DRM_VBLANK_RELATIVE = enum_drm_vblank_seq_type.define('_DRM_VBLANK_RELATIVE', 1)
_DRM_VBLANK_HIGH_CRTC_MASK = enum_drm_vblank_seq_type.define('_DRM_VBLANK_HIGH_CRTC_MASK', 62)
_DRM_VBLANK_EVENT = enum_drm_vblank_seq_type.define('_DRM_VBLANK_EVENT', 67108864)
_DRM_VBLANK_FLIP = enum_drm_vblank_seq_type.define('_DRM_VBLANK_FLIP', 134217728)
_DRM_VBLANK_NEXTONMISS = enum_drm_vblank_seq_type.define('_DRM_VBLANK_NEXTONMISS', 268435456)
_DRM_VBLANK_SECONDARY = enum_drm_vblank_seq_type.define('_DRM_VBLANK_SECONDARY', 536870912)
_DRM_VBLANK_SIGNAL = enum_drm_vblank_seq_type.define('_DRM_VBLANK_SIGNAL', 1073741824)

@c.record
class struct_drm_wait_vblank_request(c.Struct):
  SIZE = 16
  type: Annotated[enum_drm_vblank_seq_type, 0]
  sequence: Annotated[Annotated[int, ctypes.c_uint32], 4]
  signal: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_wait_vblank_reply(c.Struct):
  SIZE = 24
  type: Annotated[enum_drm_vblank_seq_type, 0]
  sequence: Annotated[Annotated[int, ctypes.c_uint32], 4]
  tval_sec: Annotated[Annotated[int, ctypes.c_int64], 8]
  tval_usec: Annotated[Annotated[int, ctypes.c_int64], 16]
@c.record
class union_drm_wait_vblank(c.Struct):
  SIZE = 24
  request: Annotated[struct_drm_wait_vblank_request, 0]
  reply: Annotated[struct_drm_wait_vblank_reply, 0]
@c.record
class struct_drm_modeset_ctl(c.Struct):
  SIZE = 8
  crtc: Annotated[Annotated[int, ctypes.c_uint32], 0]
  cmd: Annotated[Annotated[int, ctypes.c_uint32], 4]
__u32: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_drm_agp_mode(c.Struct):
  SIZE = 8
  mode: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_drm_agp_buffer(c.Struct):
  SIZE = 32
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  type: Annotated[Annotated[int, ctypes.c_uint64], 16]
  physical: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_drm_agp_binding(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_agp_info(c.Struct):
  SIZE = 56
  agp_version_major: Annotated[Annotated[int, ctypes.c_int32], 0]
  agp_version_minor: Annotated[Annotated[int, ctypes.c_int32], 4]
  mode: Annotated[Annotated[int, ctypes.c_uint64], 8]
  aperture_base: Annotated[Annotated[int, ctypes.c_uint64], 16]
  aperture_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  memory_allowed: Annotated[Annotated[int, ctypes.c_uint64], 32]
  memory_used: Annotated[Annotated[int, ctypes.c_uint64], 40]
  id_vendor: Annotated[Annotated[int, ctypes.c_uint16], 48]
  id_device: Annotated[Annotated[int, ctypes.c_uint16], 50]
@c.record
class struct_drm_scatter_gather(c.Struct):
  SIZE = 16
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_set_version(c.Struct):
  SIZE = 16
  drm_di_major: Annotated[Annotated[int, ctypes.c_int32], 0]
  drm_di_minor: Annotated[Annotated[int, ctypes.c_int32], 4]
  drm_dd_major: Annotated[Annotated[int, ctypes.c_int32], 8]
  drm_dd_minor: Annotated[Annotated[int, ctypes.c_int32], 12]
@c.record
class struct_drm_gem_close(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_gem_flink(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  name: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_gem_open(c.Struct):
  SIZE = 16
  name: Annotated[Annotated[int, ctypes.c_uint32], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
__u64: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_drm_get_cap(c.Struct):
  SIZE = 16
  capability: Annotated[Annotated[int, ctypes.c_uint64], 0]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_set_client_cap(c.Struct):
  SIZE = 16
  capability: Annotated[Annotated[int, ctypes.c_uint64], 0]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_prime_handle(c.Struct):
  SIZE = 12
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  fd: Annotated[Annotated[int, ctypes.c_int32], 8]
__s32: TypeAlias = Annotated[int, ctypes.c_int32]
@c.record
class struct_drm_syncobj_create(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_syncobj_destroy(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_syncobj_handle(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  fd: Annotated[Annotated[int, ctypes.c_int32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_syncobj_transfer(c.Struct):
  SIZE = 32
  src_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  dst_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  src_point: Annotated[Annotated[int, ctypes.c_uint64], 8]
  dst_point: Annotated[Annotated[int, ctypes.c_uint64], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_drm_syncobj_wait(c.Struct):
  SIZE = 40
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  timeout_nsec: Annotated[Annotated[int, ctypes.c_int64], 8]
  count_handles: Annotated[Annotated[int, ctypes.c_uint32], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 20]
  first_signaled: Annotated[Annotated[int, ctypes.c_uint32], 24]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 28]
  deadline_nsec: Annotated[Annotated[int, ctypes.c_uint64], 32]
__s64: TypeAlias = Annotated[int, ctypes.c_int64]
@c.record
class struct_drm_syncobj_timeline_wait(c.Struct):
  SIZE = 48
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  points: Annotated[Annotated[int, ctypes.c_uint64], 8]
  timeout_nsec: Annotated[Annotated[int, ctypes.c_int64], 16]
  count_handles: Annotated[Annotated[int, ctypes.c_uint32], 24]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  first_signaled: Annotated[Annotated[int, ctypes.c_uint32], 32]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 36]
  deadline_nsec: Annotated[Annotated[int, ctypes.c_uint64], 40]
@c.record
class struct_drm_syncobj_eventfd(c.Struct):
  SIZE = 24
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  point: Annotated[Annotated[int, ctypes.c_uint64], 8]
  fd: Annotated[Annotated[int, ctypes.c_int32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_syncobj_array(c.Struct):
  SIZE = 16
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  count_handles: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_syncobj_timeline_array(c.Struct):
  SIZE = 24
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  points: Annotated[Annotated[int, ctypes.c_uint64], 8]
  count_handles: Annotated[Annotated[int, ctypes.c_uint32], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_crtc_get_sequence(c.Struct):
  SIZE = 24
  crtc_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  active: Annotated[Annotated[int, ctypes.c_uint32], 4]
  sequence: Annotated[Annotated[int, ctypes.c_uint64], 8]
  sequence_ns: Annotated[Annotated[int, ctypes.c_int64], 16]
@c.record
class struct_drm_crtc_queue_sequence(c.Struct):
  SIZE = 24
  crtc_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  sequence: Annotated[Annotated[int, ctypes.c_uint64], 8]
  user_data: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_set_client_name(c.Struct):
  SIZE = 16
  name_len: Annotated[Annotated[int, ctypes.c_uint64], 0]
  name: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_event(c.Struct):
  SIZE = 8
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  length: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_event_vblank(c.Struct):
  SIZE = 32
  base: Annotated[struct_drm_event, 0]
  user_data: Annotated[Annotated[int, ctypes.c_uint64], 8]
  tv_sec: Annotated[Annotated[int, ctypes.c_uint32], 16]
  tv_usec: Annotated[Annotated[int, ctypes.c_uint32], 20]
  sequence: Annotated[Annotated[int, ctypes.c_uint32], 24]
  crtc_id: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_drm_event_crtc_sequence(c.Struct):
  SIZE = 32
  base: Annotated[struct_drm_event, 0]
  user_data: Annotated[Annotated[int, ctypes.c_uint64], 8]
  time_ns: Annotated[Annotated[int, ctypes.c_int64], 16]
  sequence: Annotated[Annotated[int, ctypes.c_uint64], 24]
drm_clip_rect_t: TypeAlias = struct_drm_clip_rect
drm_drawable_info_t: TypeAlias = struct_drm_drawable_info
drm_tex_region_t: TypeAlias = struct_drm_tex_region
drm_hw_lock_t: TypeAlias = struct_drm_hw_lock
drm_version_t: TypeAlias = struct_drm_version
drm_unique_t: TypeAlias = struct_drm_unique
drm_list_t: TypeAlias = struct_drm_list
drm_block_t: TypeAlias = struct_drm_block
drm_control_t: TypeAlias = struct_drm_control
drm_map_type_t: TypeAlias = enum_drm_map_type
drm_map_flags_t: TypeAlias = enum_drm_map_flags
drm_ctx_priv_map_t: TypeAlias = struct_drm_ctx_priv_map
drm_map_t: TypeAlias = struct_drm_map
drm_client_t: TypeAlias = struct_drm_client
drm_stat_type_t: TypeAlias = enum_drm_stat_type
drm_stats_t: TypeAlias = struct_drm_stats
drm_lock_flags_t: TypeAlias = enum_drm_lock_flags
drm_lock_t: TypeAlias = struct_drm_lock
drm_dma_flags_t: TypeAlias = enum_drm_dma_flags
drm_buf_desc_t: TypeAlias = struct_drm_buf_desc
drm_buf_info_t: TypeAlias = struct_drm_buf_info
drm_buf_free_t: TypeAlias = struct_drm_buf_free
drm_buf_pub_t: TypeAlias = struct_drm_buf_pub
drm_buf_map_t: TypeAlias = struct_drm_buf_map
drm_dma_t: TypeAlias = struct_drm_dma
drm_wait_vblank_t: TypeAlias = union_drm_wait_vblank
drm_agp_mode_t: TypeAlias = struct_drm_agp_mode
drm_ctx_flags_t: TypeAlias = enum_drm_ctx_flags
drm_ctx_t: TypeAlias = struct_drm_ctx
drm_ctx_res_t: TypeAlias = struct_drm_ctx_res
drm_draw_t: TypeAlias = struct_drm_draw
drm_update_draw_t: TypeAlias = struct_drm_update_draw
drm_auth_t: TypeAlias = struct_drm_auth
drm_irq_busid_t: TypeAlias = struct_drm_irq_busid
drm_vblank_seq_type_t: TypeAlias = enum_drm_vblank_seq_type
drm_agp_buffer_t: TypeAlias = struct_drm_agp_buffer
drm_agp_binding_t: TypeAlias = struct_drm_agp_binding
drm_agp_info_t: TypeAlias = struct_drm_agp_info
drm_scatter_gather_t: TypeAlias = struct_drm_scatter_gather
drm_set_version_t: TypeAlias = struct_drm_set_version
@c.record
class struct_drm_msm_timespec(c.Struct):
  SIZE = 16
  tv_sec: Annotated[Annotated[int, ctypes.c_int64], 0]
  tv_nsec: Annotated[Annotated[int, ctypes.c_int64], 8]
@c.record
class struct_drm_msm_param(c.Struct):
  SIZE = 24
  pipe: Annotated[Annotated[int, ctypes.c_uint32], 0]
  param: Annotated[Annotated[int, ctypes.c_uint32], 4]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
  len: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_msm_gem_new(c.Struct):
  SIZE = 16
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_msm_gem_info(c.Struct):
  SIZE = 24
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  info: Annotated[Annotated[int, ctypes.c_uint32], 4]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
  len: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_msm_gem_cpu_prep(c.Struct):
  SIZE = 24
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  op: Annotated[Annotated[int, ctypes.c_uint32], 4]
  timeout: Annotated[struct_drm_msm_timespec, 8]
@c.record
class struct_drm_msm_gem_cpu_fini(c.Struct):
  SIZE = 4
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_drm_msm_gem_submit_reloc(c.Struct):
  SIZE = 24
  submit_offset: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _or: Annotated[Annotated[int, ctypes.c_uint32], 4]
  shift: Annotated[Annotated[int, ctypes.c_int32], 8]
  reloc_idx: Annotated[Annotated[int, ctypes.c_uint32], 12]
  reloc_offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_msm_gem_submit_cmd(c.Struct):
  SIZE = 32
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  submit_idx: Annotated[Annotated[int, ctypes.c_uint32], 4]
  submit_offset: Annotated[Annotated[int, ctypes.c_uint32], 8]
  size: Annotated[Annotated[int, ctypes.c_uint32], 12]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 16]
  nr_relocs: Annotated[Annotated[int, ctypes.c_uint32], 20]
  relocs: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_drm_msm_gem_submit_bo(c.Struct):
  SIZE = 16
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  presumed: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_msm_gem_submit_syncobj(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  point: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_msm_gem_submit(c.Struct):
  SIZE = 72
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  fence: Annotated[Annotated[int, ctypes.c_uint32], 4]
  nr_bos: Annotated[Annotated[int, ctypes.c_uint32], 8]
  nr_cmds: Annotated[Annotated[int, ctypes.c_uint32], 12]
  bos: Annotated[Annotated[int, ctypes.c_uint64], 16]
  cmds: Annotated[Annotated[int, ctypes.c_uint64], 24]
  fence_fd: Annotated[Annotated[int, ctypes.c_int32], 32]
  queueid: Annotated[Annotated[int, ctypes.c_uint32], 36]
  in_syncobjs: Annotated[Annotated[int, ctypes.c_uint64], 40]
  out_syncobjs: Annotated[Annotated[int, ctypes.c_uint64], 48]
  nr_in_syncobjs: Annotated[Annotated[int, ctypes.c_uint32], 56]
  nr_out_syncobjs: Annotated[Annotated[int, ctypes.c_uint32], 60]
  syncobj_stride: Annotated[Annotated[int, ctypes.c_uint32], 64]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 68]
@c.record
class struct_drm_msm_wait_fence(c.Struct):
  SIZE = 32
  fence: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  timeout: Annotated[struct_drm_msm_timespec, 8]
  queueid: Annotated[Annotated[int, ctypes.c_uint32], 24]
@c.record
class struct_drm_msm_gem_madvise(c.Struct):
  SIZE = 12
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  madv: Annotated[Annotated[int, ctypes.c_uint32], 4]
  retained: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class struct_drm_msm_submitqueue(c.Struct):
  SIZE = 12
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  prio: Annotated[Annotated[int, ctypes.c_uint32], 4]
  id: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class struct_drm_msm_submitqueue_query(c.Struct):
  SIZE = 24
  data: Annotated[Annotated[int, ctypes.c_uint64], 0]
  id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  param: Annotated[Annotated[int, ctypes.c_uint32], 12]
  len: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
c.init_records()
DRM_NAME = "drm" # type: ignore
DRM_MIN_ORDER = 5 # type: ignore
DRM_MAX_ORDER = 22 # type: ignore
DRM_RAM_PERCENT = 10 # type: ignore
_DRM_LOCK_HELD = 0x80000000 # type: ignore
_DRM_LOCK_CONT = 0x40000000 # type: ignore
_DRM_LOCK_IS_HELD = lambda lock: ((lock) & _DRM_LOCK_HELD) # type: ignore
_DRM_LOCK_IS_CONT = lambda lock: ((lock) & _DRM_LOCK_CONT) # type: ignore
_DRM_LOCKING_CONTEXT = lambda lock: ((lock) & ~(_DRM_LOCK_HELD|_DRM_LOCK_CONT)) # type: ignore
_DRM_VBLANK_HIGH_CRTC_SHIFT = 1 # type: ignore
_DRM_VBLANK_TYPES_MASK = (_DRM_VBLANK_ABSOLUTE | _DRM_VBLANK_RELATIVE) # type: ignore
_DRM_VBLANK_FLAGS_MASK = (_DRM_VBLANK_EVENT | _DRM_VBLANK_SIGNAL | _DRM_VBLANK_SECONDARY | _DRM_VBLANK_NEXTONMISS) # type: ignore
_DRM_PRE_MODESET = 1 # type: ignore
_DRM_POST_MODESET = 2 # type: ignore
DRM_CAP_DUMB_BUFFER = 0x1 # type: ignore
DRM_CAP_VBLANK_HIGH_CRTC = 0x2 # type: ignore
DRM_CAP_DUMB_PREFERRED_DEPTH = 0x3 # type: ignore
DRM_CAP_DUMB_PREFER_SHADOW = 0x4 # type: ignore
DRM_CAP_PRIME = 0x5 # type: ignore
DRM_PRIME_CAP_IMPORT = 0x1 # type: ignore
DRM_PRIME_CAP_EXPORT = 0x2 # type: ignore
DRM_CAP_TIMESTAMP_MONOTONIC = 0x6 # type: ignore
DRM_CAP_ASYNC_PAGE_FLIP = 0x7 # type: ignore
DRM_CAP_CURSOR_WIDTH = 0x8 # type: ignore
DRM_CAP_CURSOR_HEIGHT = 0x9 # type: ignore
DRM_CAP_ADDFB2_MODIFIERS = 0x10 # type: ignore
DRM_CAP_PAGE_FLIP_TARGET = 0x11 # type: ignore
DRM_CAP_CRTC_IN_VBLANK_EVENT = 0x12 # type: ignore
DRM_CAP_SYNCOBJ = 0x13 # type: ignore
DRM_CAP_SYNCOBJ_TIMELINE = 0x14 # type: ignore
DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP = 0x15 # type: ignore
DRM_CLIENT_CAP_STEREO_3D = 1 # type: ignore
DRM_CLIENT_CAP_UNIVERSAL_PLANES = 2 # type: ignore
DRM_CLIENT_CAP_ATOMIC = 3 # type: ignore
DRM_CLIENT_CAP_ASPECT_RATIO = 4 # type: ignore
DRM_CLIENT_CAP_WRITEBACK_CONNECTORS = 5 # type: ignore
DRM_CLIENT_CAP_CURSOR_PLANE_HOTSPOT = 6 # type: ignore
DRM_SYNCOBJ_CREATE_SIGNALED = (1 << 0) # type: ignore
DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE = (1 << 0) # type: ignore
DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE = (1 << 0) # type: ignore
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL = (1 << 0) # type: ignore
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT = (1 << 1) # type: ignore
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE = (1 << 2) # type: ignore
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE = (1 << 3) # type: ignore
DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED = (1 << 0) # type: ignore
DRM_CRTC_SEQUENCE_RELATIVE = 0x00000001 # type: ignore
DRM_CRTC_SEQUENCE_NEXT_ON_MISS = 0x00000002 # type: ignore
DRM_CLIENT_NAME_MAX_LEN = 64 # type: ignore
DRM_IOCTL_BASE = 'd' # type: ignore
DRM_IO = lambda nr: _IO(DRM_IOCTL_BASE,nr) # type: ignore
DRM_IOR = lambda nr,type: _IOR(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOW = lambda nr,type: _IOW(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOWR = lambda nr,type: _IOWR(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOCTL_VERSION = DRM_IOWR(0x00, struct_drm_version) # type: ignore
DRM_IOCTL_GET_UNIQUE = DRM_IOWR(0x01, struct_drm_unique) # type: ignore
DRM_IOCTL_GET_MAGIC = DRM_IOR( 0x02, struct_drm_auth) # type: ignore
DRM_IOCTL_IRQ_BUSID = DRM_IOWR(0x03, struct_drm_irq_busid) # type: ignore
DRM_IOCTL_GET_MAP = DRM_IOWR(0x04, struct_drm_map) # type: ignore
DRM_IOCTL_GET_CLIENT = DRM_IOWR(0x05, struct_drm_client) # type: ignore
DRM_IOCTL_GET_STATS = DRM_IOR( 0x06, struct_drm_stats) # type: ignore
DRM_IOCTL_SET_VERSION = DRM_IOWR(0x07, struct_drm_set_version) # type: ignore
DRM_IOCTL_MODESET_CTL = DRM_IOW(0x08, struct_drm_modeset_ctl) # type: ignore
DRM_IOCTL_GEM_CLOSE = DRM_IOW (0x09, struct_drm_gem_close) # type: ignore
DRM_IOCTL_GEM_FLINK = DRM_IOWR(0x0a, struct_drm_gem_flink) # type: ignore
DRM_IOCTL_GEM_OPEN = DRM_IOWR(0x0b, struct_drm_gem_open) # type: ignore
DRM_IOCTL_GET_CAP = DRM_IOWR(0x0c, struct_drm_get_cap) # type: ignore
DRM_IOCTL_SET_CLIENT_CAP = DRM_IOW( 0x0d, struct_drm_set_client_cap) # type: ignore
DRM_IOCTL_SET_UNIQUE = DRM_IOW( 0x10, struct_drm_unique) # type: ignore
DRM_IOCTL_AUTH_MAGIC = DRM_IOW( 0x11, struct_drm_auth) # type: ignore
DRM_IOCTL_BLOCK = DRM_IOWR(0x12, struct_drm_block) # type: ignore
DRM_IOCTL_UNBLOCK = DRM_IOWR(0x13, struct_drm_block) # type: ignore
DRM_IOCTL_CONTROL = DRM_IOW( 0x14, struct_drm_control) # type: ignore
DRM_IOCTL_ADD_MAP = DRM_IOWR(0x15, struct_drm_map) # type: ignore
DRM_IOCTL_ADD_BUFS = DRM_IOWR(0x16, struct_drm_buf_desc) # type: ignore
DRM_IOCTL_MARK_BUFS = DRM_IOW( 0x17, struct_drm_buf_desc) # type: ignore
DRM_IOCTL_INFO_BUFS = DRM_IOWR(0x18, struct_drm_buf_info) # type: ignore
DRM_IOCTL_MAP_BUFS = DRM_IOWR(0x19, struct_drm_buf_map) # type: ignore
DRM_IOCTL_FREE_BUFS = DRM_IOW( 0x1a, struct_drm_buf_free) # type: ignore
DRM_IOCTL_RM_MAP = DRM_IOW( 0x1b, struct_drm_map) # type: ignore
DRM_IOCTL_SET_SAREA_CTX = DRM_IOW( 0x1c, struct_drm_ctx_priv_map) # type: ignore
DRM_IOCTL_GET_SAREA_CTX = DRM_IOWR(0x1d, struct_drm_ctx_priv_map) # type: ignore
DRM_IOCTL_SET_MASTER = DRM_IO(0x1e) # type: ignore
DRM_IOCTL_DROP_MASTER = DRM_IO(0x1f) # type: ignore
DRM_IOCTL_ADD_CTX = DRM_IOWR(0x20, struct_drm_ctx) # type: ignore
DRM_IOCTL_RM_CTX = DRM_IOWR(0x21, struct_drm_ctx) # type: ignore
DRM_IOCTL_MOD_CTX = DRM_IOW( 0x22, struct_drm_ctx) # type: ignore
DRM_IOCTL_GET_CTX = DRM_IOWR(0x23, struct_drm_ctx) # type: ignore
DRM_IOCTL_SWITCH_CTX = DRM_IOW( 0x24, struct_drm_ctx) # type: ignore
DRM_IOCTL_NEW_CTX = DRM_IOW( 0x25, struct_drm_ctx) # type: ignore
DRM_IOCTL_RES_CTX = DRM_IOWR(0x26, struct_drm_ctx_res) # type: ignore
DRM_IOCTL_ADD_DRAW = DRM_IOWR(0x27, struct_drm_draw) # type: ignore
DRM_IOCTL_RM_DRAW = DRM_IOWR(0x28, struct_drm_draw) # type: ignore
DRM_IOCTL_DMA = DRM_IOWR(0x29, struct_drm_dma) # type: ignore
DRM_IOCTL_LOCK = DRM_IOW( 0x2a, struct_drm_lock) # type: ignore
DRM_IOCTL_UNLOCK = DRM_IOW( 0x2b, struct_drm_lock) # type: ignore
DRM_IOCTL_FINISH = DRM_IOW( 0x2c, struct_drm_lock) # type: ignore
DRM_IOCTL_PRIME_HANDLE_TO_FD = DRM_IOWR(0x2d, struct_drm_prime_handle) # type: ignore
DRM_IOCTL_PRIME_FD_TO_HANDLE = DRM_IOWR(0x2e, struct_drm_prime_handle) # type: ignore
DRM_IOCTL_AGP_ACQUIRE = DRM_IO(  0x30) # type: ignore
DRM_IOCTL_AGP_RELEASE = DRM_IO(  0x31) # type: ignore
DRM_IOCTL_AGP_ENABLE = DRM_IOW( 0x32, struct_drm_agp_mode) # type: ignore
DRM_IOCTL_AGP_INFO = DRM_IOR( 0x33, struct_drm_agp_info) # type: ignore
DRM_IOCTL_AGP_ALLOC = DRM_IOWR(0x34, struct_drm_agp_buffer) # type: ignore
DRM_IOCTL_AGP_FREE = DRM_IOW( 0x35, struct_drm_agp_buffer) # type: ignore
DRM_IOCTL_AGP_BIND = DRM_IOW( 0x36, struct_drm_agp_binding) # type: ignore
DRM_IOCTL_AGP_UNBIND = DRM_IOW( 0x37, struct_drm_agp_binding) # type: ignore
DRM_IOCTL_SG_ALLOC = DRM_IOWR(0x38, struct_drm_scatter_gather) # type: ignore
DRM_IOCTL_SG_FREE = DRM_IOW( 0x39, struct_drm_scatter_gather) # type: ignore
DRM_IOCTL_WAIT_VBLANK = DRM_IOWR(0x3a, union_drm_wait_vblank) # type: ignore
DRM_IOCTL_CRTC_GET_SEQUENCE = DRM_IOWR(0x3b, struct_drm_crtc_get_sequence) # type: ignore
DRM_IOCTL_CRTC_QUEUE_SEQUENCE = DRM_IOWR(0x3c, struct_drm_crtc_queue_sequence) # type: ignore
DRM_IOCTL_UPDATE_DRAW = DRM_IOW(0x3f, struct_drm_update_draw) # type: ignore
DRM_IOCTL_SYNCOBJ_CREATE = DRM_IOWR(0xBF, struct_drm_syncobj_create) # type: ignore
DRM_IOCTL_SYNCOBJ_DESTROY = DRM_IOWR(0xC0, struct_drm_syncobj_destroy) # type: ignore
DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD = DRM_IOWR(0xC1, struct_drm_syncobj_handle) # type: ignore
DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE = DRM_IOWR(0xC2, struct_drm_syncobj_handle) # type: ignore
DRM_IOCTL_SYNCOBJ_WAIT = DRM_IOWR(0xC3, struct_drm_syncobj_wait) # type: ignore
DRM_IOCTL_SYNCOBJ_RESET = DRM_IOWR(0xC4, struct_drm_syncobj_array) # type: ignore
DRM_IOCTL_SYNCOBJ_SIGNAL = DRM_IOWR(0xC5, struct_drm_syncobj_array) # type: ignore
DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT = DRM_IOWR(0xCA, struct_drm_syncobj_timeline_wait) # type: ignore
DRM_IOCTL_SYNCOBJ_QUERY = DRM_IOWR(0xCB, struct_drm_syncobj_timeline_array) # type: ignore
DRM_IOCTL_SYNCOBJ_TRANSFER = DRM_IOWR(0xCC, struct_drm_syncobj_transfer) # type: ignore
DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL = DRM_IOWR(0xCD, struct_drm_syncobj_timeline_array) # type: ignore
DRM_IOCTL_SYNCOBJ_EVENTFD = DRM_IOWR(0xCF, struct_drm_syncobj_eventfd) # type: ignore
DRM_IOCTL_SET_CLIENT_NAME = DRM_IOWR(0xD1, struct_drm_set_client_name) # type: ignore
DRM_COMMAND_BASE = 0x40 # type: ignore
DRM_COMMAND_END = 0xA0 # type: ignore
DRM_EVENT_VBLANK = 0x01 # type: ignore
DRM_EVENT_FLIP_COMPLETE = 0x02 # type: ignore
DRM_EVENT_CRTC_SEQUENCE = 0x03 # type: ignore
MSM_PIPE_NONE = 0x00 # type: ignore
MSM_PIPE_2D0 = 0x01 # type: ignore
MSM_PIPE_2D1 = 0x02 # type: ignore
MSM_PIPE_3D0 = 0x10 # type: ignore
MSM_PIPE_ID_MASK = 0xffff # type: ignore
MSM_PIPE_ID = lambda x: ((x) & MSM_PIPE_ID_MASK) # type: ignore
MSM_PIPE_FLAGS = lambda x: ((x) & ~MSM_PIPE_ID_MASK) # type: ignore
MSM_PARAM_GPU_ID = 0x01 # type: ignore
MSM_PARAM_GMEM_SIZE = 0x02 # type: ignore
MSM_PARAM_CHIP_ID = 0x03 # type: ignore
MSM_PARAM_MAX_FREQ = 0x04 # type: ignore
MSM_PARAM_TIMESTAMP = 0x05 # type: ignore
MSM_PARAM_GMEM_BASE = 0x06 # type: ignore
MSM_PARAM_PRIORITIES = 0x07 # type: ignore
MSM_PARAM_PP_PGTABLE = 0x08 # type: ignore
MSM_PARAM_FAULTS = 0x09 # type: ignore
MSM_PARAM_SUSPENDS = 0x0a # type: ignore
MSM_PARAM_SYSPROF = 0x0b # type: ignore
MSM_PARAM_COMM = 0x0c # type: ignore
MSM_PARAM_CMDLINE = 0x0d # type: ignore
MSM_PARAM_VA_START = 0x0e # type: ignore
MSM_PARAM_VA_SIZE = 0x0f # type: ignore
MSM_PARAM_HIGHEST_BANK_BIT = 0x10 # type: ignore
MSM_PARAM_NR_RINGS = MSM_PARAM_PRIORITIES # type: ignore
MSM_BO_SCANOUT = 0x00000001 # type: ignore
MSM_BO_GPU_READONLY = 0x00000002 # type: ignore
MSM_BO_CACHE_MASK = 0x000f0000 # type: ignore
MSM_BO_CACHED = 0x00010000 # type: ignore
MSM_BO_WC = 0x00020000 # type: ignore
MSM_BO_UNCACHED = 0x00040000 # type: ignore
MSM_BO_CACHED_COHERENT = 0x080000 # type: ignore
MSM_BO_FLAGS = (MSM_BO_SCANOUT | MSM_BO_GPU_READONLY | MSM_BO_CACHE_MASK) # type: ignore
MSM_INFO_GET_OFFSET = 0x00 # type: ignore
MSM_INFO_GET_IOVA = 0x01 # type: ignore
MSM_INFO_SET_NAME = 0x02 # type: ignore
MSM_INFO_GET_NAME = 0x03 # type: ignore
MSM_INFO_SET_IOVA = 0x04 # type: ignore
MSM_INFO_GET_FLAGS = 0x05 # type: ignore
MSM_INFO_SET_METADATA = 0x06 # type: ignore
MSM_INFO_GET_METADATA = 0x07 # type: ignore
MSM_PREP_READ = 0x01 # type: ignore
MSM_PREP_WRITE = 0x02 # type: ignore
MSM_PREP_NOSYNC = 0x04 # type: ignore
MSM_PREP_BOOST = 0x08 # type: ignore
MSM_PREP_FLAGS = (MSM_PREP_READ | MSM_PREP_WRITE | MSM_PREP_NOSYNC | MSM_PREP_BOOST | 0) # type: ignore
MSM_SUBMIT_CMD_BUF = 0x0001 # type: ignore
MSM_SUBMIT_CMD_IB_TARGET_BUF = 0x0002 # type: ignore
MSM_SUBMIT_CMD_CTX_RESTORE_BUF = 0x0003 # type: ignore
MSM_SUBMIT_BO_READ = 0x0001 # type: ignore
MSM_SUBMIT_BO_WRITE = 0x0002 # type: ignore
MSM_SUBMIT_BO_DUMP = 0x0004 # type: ignore
MSM_SUBMIT_BO_NO_IMPLICIT = 0x0008 # type: ignore
MSM_SUBMIT_BO_FLAGS = (MSM_SUBMIT_BO_READ | MSM_SUBMIT_BO_WRITE | MSM_SUBMIT_BO_DUMP | MSM_SUBMIT_BO_NO_IMPLICIT) # type: ignore
MSM_SUBMIT_NO_IMPLICIT = 0x80000000 # type: ignore
MSM_SUBMIT_FENCE_FD_IN = 0x40000000 # type: ignore
MSM_SUBMIT_FENCE_FD_OUT = 0x20000000 # type: ignore
MSM_SUBMIT_SUDO = 0x10000000 # type: ignore
MSM_SUBMIT_SYNCOBJ_IN = 0x08000000 # type: ignore
MSM_SUBMIT_SYNCOBJ_OUT = 0x04000000 # type: ignore
MSM_SUBMIT_FENCE_SN_IN = 0x02000000 # type: ignore
MSM_SUBMIT_FLAGS = ( MSM_SUBMIT_NO_IMPLICIT   | MSM_SUBMIT_FENCE_FD_IN   | MSM_SUBMIT_FENCE_FD_OUT  | MSM_SUBMIT_SUDO          | MSM_SUBMIT_SYNCOBJ_IN    | MSM_SUBMIT_SYNCOBJ_OUT   | MSM_SUBMIT_FENCE_SN_IN   | 0) # type: ignore
MSM_SUBMIT_SYNCOBJ_RESET = 0x00000001 # type: ignore
MSM_SUBMIT_SYNCOBJ_FLAGS = ( MSM_SUBMIT_SYNCOBJ_RESET | 0) # type: ignore
MSM_WAIT_FENCE_BOOST = 0x00000001 # type: ignore
MSM_WAIT_FENCE_FLAGS = ( MSM_WAIT_FENCE_BOOST | 0) # type: ignore
MSM_MADV_WILLNEED = 0 # type: ignore
MSM_MADV_DONTNEED = 1 # type: ignore
__MSM_MADV_PURGED = 2 # type: ignore
MSM_SUBMITQUEUE_FLAGS = (0) # type: ignore
MSM_SUBMITQUEUE_PARAM_FAULTS = 0 # type: ignore
DRM_MSM_GET_PARAM = 0x00 # type: ignore
DRM_MSM_SET_PARAM = 0x01 # type: ignore
DRM_MSM_GEM_NEW = 0x02 # type: ignore
DRM_MSM_GEM_INFO = 0x03 # type: ignore
DRM_MSM_GEM_CPU_PREP = 0x04 # type: ignore
DRM_MSM_GEM_CPU_FINI = 0x05 # type: ignore
DRM_MSM_GEM_SUBMIT = 0x06 # type: ignore
DRM_MSM_WAIT_FENCE = 0x07 # type: ignore
DRM_MSM_GEM_MADVISE = 0x08 # type: ignore
DRM_MSM_SUBMITQUEUE_NEW = 0x0A # type: ignore
DRM_MSM_SUBMITQUEUE_CLOSE = 0x0B # type: ignore
DRM_MSM_SUBMITQUEUE_QUERY = 0x0C # type: ignore
DRM_IOCTL_MSM_GET_PARAM = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GET_PARAM, struct_drm_msm_param) # type: ignore
DRM_IOCTL_MSM_SET_PARAM = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_SET_PARAM, struct_drm_msm_param) # type: ignore
DRM_IOCTL_MSM_GEM_NEW = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GEM_NEW, struct_drm_msm_gem_new) # type: ignore
DRM_IOCTL_MSM_GEM_INFO = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GEM_INFO, struct_drm_msm_gem_info) # type: ignore
DRM_IOCTL_MSM_GEM_CPU_PREP = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_GEM_CPU_PREP, struct_drm_msm_gem_cpu_prep) # type: ignore
DRM_IOCTL_MSM_GEM_CPU_FINI = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_GEM_CPU_FINI, struct_drm_msm_gem_cpu_fini) # type: ignore
DRM_IOCTL_MSM_GEM_SUBMIT = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GEM_SUBMIT, struct_drm_msm_gem_submit) # type: ignore
DRM_IOCTL_MSM_WAIT_FENCE = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_WAIT_FENCE, struct_drm_msm_wait_fence) # type: ignore
DRM_IOCTL_MSM_GEM_MADVISE = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GEM_MADVISE, struct_drm_msm_gem_madvise) # type: ignore
DRM_IOCTL_MSM_SUBMITQUEUE_NEW = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_SUBMITQUEUE_NEW, struct_drm_msm_submitqueue) # type: ignore
DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_SUBMITQUEUE_CLOSE, __u32) # type: ignore
DRM_IOCTL_MSM_SUBMITQUEUE_QUERY = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_SUBMITQUEUE_QUERY, struct_drm_msm_submitqueue_query) # type: ignore