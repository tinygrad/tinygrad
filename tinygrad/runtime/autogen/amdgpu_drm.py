# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
drm_handle_t = ctypes.c_uint32
drm_context_t = ctypes.c_uint32
drm_drawable_t = ctypes.c_uint32
drm_magic_t = ctypes.c_uint32
class struct_drm_clip_rect(Struct): pass
struct_drm_clip_rect._fields_ = [
  ('x1', ctypes.c_uint16),
  ('y1', ctypes.c_uint16),
  ('x2', ctypes.c_uint16),
  ('y2', ctypes.c_uint16),
]
class struct_drm_drawable_info(Struct): pass
struct_drm_drawable_info._fields_ = [
  ('num_rects', ctypes.c_uint32),
  ('rects', ctypes.POINTER(struct_drm_clip_rect)),
]
class struct_drm_tex_region(Struct): pass
struct_drm_tex_region._fields_ = [
  ('next', ctypes.c_ubyte),
  ('prev', ctypes.c_ubyte),
  ('in_use', ctypes.c_ubyte),
  ('padding', ctypes.c_ubyte),
  ('age', ctypes.c_uint32),
]
class struct_drm_hw_lock(Struct): pass
struct_drm_hw_lock._fields_ = [
  ('lock', ctypes.c_uint32),
  ('padding', (ctypes.c_char * 60)),
]
class struct_drm_version(Struct): pass
__kernel_size_t = ctypes.c_uint64
struct_drm_version._fields_ = [
  ('version_major', ctypes.c_int32),
  ('version_minor', ctypes.c_int32),
  ('version_patchlevel', ctypes.c_int32),
  ('name_len', ctypes.c_uint64),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('date_len', ctypes.c_uint64),
  ('date', ctypes.POINTER(ctypes.c_char)),
  ('desc_len', ctypes.c_uint64),
  ('desc', ctypes.POINTER(ctypes.c_char)),
]
class struct_drm_unique(Struct): pass
struct_drm_unique._fields_ = [
  ('unique_len', ctypes.c_uint64),
  ('unique', ctypes.POINTER(ctypes.c_char)),
]
class struct_drm_list(Struct): pass
struct_drm_list._fields_ = [
  ('count', ctypes.c_int32),
  ('version', ctypes.POINTER(struct_drm_version)),
]
class struct_drm_block(Struct): pass
struct_drm_block._fields_ = [
  ('unused', ctypes.c_int32),
]
class struct_drm_control(Struct): pass
struct_drm_control_func = CEnum(ctypes.c_uint32)
DRM_ADD_COMMAND = struct_drm_control_func.define('DRM_ADD_COMMAND', 0)
DRM_RM_COMMAND = struct_drm_control_func.define('DRM_RM_COMMAND', 1)
DRM_INST_HANDLER = struct_drm_control_func.define('DRM_INST_HANDLER', 2)
DRM_UNINST_HANDLER = struct_drm_control_func.define('DRM_UNINST_HANDLER', 3)

struct_drm_control._fields_ = [
  ('func', struct_drm_control_func),
  ('irq', ctypes.c_int32),
]
enum_drm_map_type = CEnum(ctypes.c_uint32)
_DRM_FRAME_BUFFER = enum_drm_map_type.define('_DRM_FRAME_BUFFER', 0)
_DRM_REGISTERS = enum_drm_map_type.define('_DRM_REGISTERS', 1)
_DRM_SHM = enum_drm_map_type.define('_DRM_SHM', 2)
_DRM_AGP = enum_drm_map_type.define('_DRM_AGP', 3)
_DRM_SCATTER_GATHER = enum_drm_map_type.define('_DRM_SCATTER_GATHER', 4)
_DRM_CONSISTENT = enum_drm_map_type.define('_DRM_CONSISTENT', 5)

enum_drm_map_flags = CEnum(ctypes.c_uint32)
_DRM_RESTRICTED = enum_drm_map_flags.define('_DRM_RESTRICTED', 1)
_DRM_READ_ONLY = enum_drm_map_flags.define('_DRM_READ_ONLY', 2)
_DRM_LOCKED = enum_drm_map_flags.define('_DRM_LOCKED', 4)
_DRM_KERNEL = enum_drm_map_flags.define('_DRM_KERNEL', 8)
_DRM_WRITE_COMBINING = enum_drm_map_flags.define('_DRM_WRITE_COMBINING', 16)
_DRM_CONTAINS_LOCK = enum_drm_map_flags.define('_DRM_CONTAINS_LOCK', 32)
_DRM_REMOVABLE = enum_drm_map_flags.define('_DRM_REMOVABLE', 64)
_DRM_DRIVER = enum_drm_map_flags.define('_DRM_DRIVER', 128)

class struct_drm_ctx_priv_map(Struct): pass
struct_drm_ctx_priv_map._fields_ = [
  ('ctx_id', ctypes.c_uint32),
  ('handle', ctypes.c_void_p),
]
class struct_drm_map(Struct): pass
struct_drm_map._fields_ = [
  ('offset', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('type', enum_drm_map_type),
  ('flags', enum_drm_map_flags),
  ('handle', ctypes.c_void_p),
  ('mtrr', ctypes.c_int32),
]
class struct_drm_client(Struct): pass
struct_drm_client._fields_ = [
  ('idx', ctypes.c_int32),
  ('auth', ctypes.c_int32),
  ('pid', ctypes.c_uint64),
  ('uid', ctypes.c_uint64),
  ('magic', ctypes.c_uint64),
  ('iocs', ctypes.c_uint64),
]
enum_drm_stat_type = CEnum(ctypes.c_uint32)
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

class struct_drm_stats(Struct): pass
class struct_drm_stats_data(Struct): pass
struct_drm_stats_data._fields_ = [
  ('value', ctypes.c_uint64),
  ('type', enum_drm_stat_type),
]
struct_drm_stats._fields_ = [
  ('count', ctypes.c_uint64),
  ('data', (struct_drm_stats_data * 15)),
]
enum_drm_lock_flags = CEnum(ctypes.c_uint32)
_DRM_LOCK_READY = enum_drm_lock_flags.define('_DRM_LOCK_READY', 1)
_DRM_LOCK_QUIESCENT = enum_drm_lock_flags.define('_DRM_LOCK_QUIESCENT', 2)
_DRM_LOCK_FLUSH = enum_drm_lock_flags.define('_DRM_LOCK_FLUSH', 4)
_DRM_LOCK_FLUSH_ALL = enum_drm_lock_flags.define('_DRM_LOCK_FLUSH_ALL', 8)
_DRM_HALT_ALL_QUEUES = enum_drm_lock_flags.define('_DRM_HALT_ALL_QUEUES', 16)
_DRM_HALT_CUR_QUEUES = enum_drm_lock_flags.define('_DRM_HALT_CUR_QUEUES', 32)

class struct_drm_lock(Struct): pass
struct_drm_lock._fields_ = [
  ('context', ctypes.c_int32),
  ('flags', enum_drm_lock_flags),
]
enum_drm_dma_flags = CEnum(ctypes.c_uint32)
_DRM_DMA_BLOCK = enum_drm_dma_flags.define('_DRM_DMA_BLOCK', 1)
_DRM_DMA_WHILE_LOCKED = enum_drm_dma_flags.define('_DRM_DMA_WHILE_LOCKED', 2)
_DRM_DMA_PRIORITY = enum_drm_dma_flags.define('_DRM_DMA_PRIORITY', 4)
_DRM_DMA_WAIT = enum_drm_dma_flags.define('_DRM_DMA_WAIT', 16)
_DRM_DMA_SMALLER_OK = enum_drm_dma_flags.define('_DRM_DMA_SMALLER_OK', 32)
_DRM_DMA_LARGER_OK = enum_drm_dma_flags.define('_DRM_DMA_LARGER_OK', 64)

class struct_drm_buf_desc(Struct): pass
struct_drm_buf_desc_flags = CEnum(ctypes.c_uint32)
_DRM_PAGE_ALIGN = struct_drm_buf_desc_flags.define('_DRM_PAGE_ALIGN', 1)
_DRM_AGP_BUFFER = struct_drm_buf_desc_flags.define('_DRM_AGP_BUFFER', 2)
_DRM_SG_BUFFER = struct_drm_buf_desc_flags.define('_DRM_SG_BUFFER', 4)
_DRM_FB_BUFFER = struct_drm_buf_desc_flags.define('_DRM_FB_BUFFER', 8)
_DRM_PCI_BUFFER_RO = struct_drm_buf_desc_flags.define('_DRM_PCI_BUFFER_RO', 16)

struct_drm_buf_desc._fields_ = [
  ('count', ctypes.c_int32),
  ('size', ctypes.c_int32),
  ('low_mark', ctypes.c_int32),
  ('high_mark', ctypes.c_int32),
  ('flags', struct_drm_buf_desc_flags),
  ('agp_start', ctypes.c_uint64),
]
class struct_drm_buf_info(Struct): pass
struct_drm_buf_info._fields_ = [
  ('count', ctypes.c_int32),
  ('list', ctypes.POINTER(struct_drm_buf_desc)),
]
class struct_drm_buf_free(Struct): pass
struct_drm_buf_free._fields_ = [
  ('count', ctypes.c_int32),
  ('list', ctypes.POINTER(ctypes.c_int32)),
]
class struct_drm_buf_pub(Struct): pass
struct_drm_buf_pub._fields_ = [
  ('idx', ctypes.c_int32),
  ('total', ctypes.c_int32),
  ('used', ctypes.c_int32),
  ('address', ctypes.c_void_p),
]
class struct_drm_buf_map(Struct): pass
struct_drm_buf_map._fields_ = [
  ('count', ctypes.c_int32),
  ('virtual', ctypes.c_void_p),
  ('list', ctypes.POINTER(struct_drm_buf_pub)),
]
class struct_drm_dma(Struct): pass
struct_drm_dma._fields_ = [
  ('context', ctypes.c_int32),
  ('send_count', ctypes.c_int32),
  ('send_indices', ctypes.POINTER(ctypes.c_int32)),
  ('send_sizes', ctypes.POINTER(ctypes.c_int32)),
  ('flags', enum_drm_dma_flags),
  ('request_count', ctypes.c_int32),
  ('request_size', ctypes.c_int32),
  ('request_indices', ctypes.POINTER(ctypes.c_int32)),
  ('request_sizes', ctypes.POINTER(ctypes.c_int32)),
  ('granted_count', ctypes.c_int32),
]
enum_drm_ctx_flags = CEnum(ctypes.c_uint32)
_DRM_CONTEXT_PRESERVED = enum_drm_ctx_flags.define('_DRM_CONTEXT_PRESERVED', 1)
_DRM_CONTEXT_2DONLY = enum_drm_ctx_flags.define('_DRM_CONTEXT_2DONLY', 2)

class struct_drm_ctx(Struct): pass
struct_drm_ctx._fields_ = [
  ('handle', drm_context_t),
  ('flags', enum_drm_ctx_flags),
]
class struct_drm_ctx_res(Struct): pass
struct_drm_ctx_res._fields_ = [
  ('count', ctypes.c_int32),
  ('contexts', ctypes.POINTER(struct_drm_ctx)),
]
class struct_drm_draw(Struct): pass
struct_drm_draw._fields_ = [
  ('handle', drm_drawable_t),
]
drm_drawable_info_type_t = CEnum(ctypes.c_uint32)
DRM_DRAWABLE_CLIPRECTS = drm_drawable_info_type_t.define('DRM_DRAWABLE_CLIPRECTS', 0)

class struct_drm_update_draw(Struct): pass
struct_drm_update_draw._fields_ = [
  ('handle', drm_drawable_t),
  ('type', ctypes.c_uint32),
  ('num', ctypes.c_uint32),
  ('data', ctypes.c_uint64),
]
class struct_drm_auth(Struct): pass
struct_drm_auth._fields_ = [
  ('magic', drm_magic_t),
]
class struct_drm_irq_busid(Struct): pass
struct_drm_irq_busid._fields_ = [
  ('irq', ctypes.c_int32),
  ('busnum', ctypes.c_int32),
  ('devnum', ctypes.c_int32),
  ('funcnum', ctypes.c_int32),
]
enum_drm_vblank_seq_type = CEnum(ctypes.c_uint32)
_DRM_VBLANK_ABSOLUTE = enum_drm_vblank_seq_type.define('_DRM_VBLANK_ABSOLUTE', 0)
_DRM_VBLANK_RELATIVE = enum_drm_vblank_seq_type.define('_DRM_VBLANK_RELATIVE', 1)
_DRM_VBLANK_HIGH_CRTC_MASK = enum_drm_vblank_seq_type.define('_DRM_VBLANK_HIGH_CRTC_MASK', 62)
_DRM_VBLANK_EVENT = enum_drm_vblank_seq_type.define('_DRM_VBLANK_EVENT', 67108864)
_DRM_VBLANK_FLIP = enum_drm_vblank_seq_type.define('_DRM_VBLANK_FLIP', 134217728)
_DRM_VBLANK_NEXTONMISS = enum_drm_vblank_seq_type.define('_DRM_VBLANK_NEXTONMISS', 268435456)
_DRM_VBLANK_SECONDARY = enum_drm_vblank_seq_type.define('_DRM_VBLANK_SECONDARY', 536870912)
_DRM_VBLANK_SIGNAL = enum_drm_vblank_seq_type.define('_DRM_VBLANK_SIGNAL', 1073741824)

class struct_drm_wait_vblank_request(Struct): pass
struct_drm_wait_vblank_request._fields_ = [
  ('type', enum_drm_vblank_seq_type),
  ('sequence', ctypes.c_uint32),
  ('signal', ctypes.c_uint64),
]
class struct_drm_wait_vblank_reply(Struct): pass
struct_drm_wait_vblank_reply._fields_ = [
  ('type', enum_drm_vblank_seq_type),
  ('sequence', ctypes.c_uint32),
  ('tval_sec', ctypes.c_int64),
  ('tval_usec', ctypes.c_int64),
]
class union_drm_wait_vblank(ctypes.Union): pass
union_drm_wait_vblank._fields_ = [
  ('request', struct_drm_wait_vblank_request),
  ('reply', struct_drm_wait_vblank_reply),
]
class struct_drm_modeset_ctl(Struct): pass
__u32 = ctypes.c_uint32
struct_drm_modeset_ctl._fields_ = [
  ('crtc', ctypes.c_uint32),
  ('cmd', ctypes.c_uint32),
]
class struct_drm_agp_mode(Struct): pass
struct_drm_agp_mode._fields_ = [
  ('mode', ctypes.c_uint64),
]
class struct_drm_agp_buffer(Struct): pass
struct_drm_agp_buffer._fields_ = [
  ('size', ctypes.c_uint64),
  ('handle', ctypes.c_uint64),
  ('type', ctypes.c_uint64),
  ('physical', ctypes.c_uint64),
]
class struct_drm_agp_binding(Struct): pass
struct_drm_agp_binding._fields_ = [
  ('handle', ctypes.c_uint64),
  ('offset', ctypes.c_uint64),
]
class struct_drm_agp_info(Struct): pass
struct_drm_agp_info._fields_ = [
  ('agp_version_major', ctypes.c_int32),
  ('agp_version_minor', ctypes.c_int32),
  ('mode', ctypes.c_uint64),
  ('aperture_base', ctypes.c_uint64),
  ('aperture_size', ctypes.c_uint64),
  ('memory_allowed', ctypes.c_uint64),
  ('memory_used', ctypes.c_uint64),
  ('id_vendor', ctypes.c_uint16),
  ('id_device', ctypes.c_uint16),
]
class struct_drm_scatter_gather(Struct): pass
struct_drm_scatter_gather._fields_ = [
  ('size', ctypes.c_uint64),
  ('handle', ctypes.c_uint64),
]
class struct_drm_set_version(Struct): pass
struct_drm_set_version._fields_ = [
  ('drm_di_major', ctypes.c_int32),
  ('drm_di_minor', ctypes.c_int32),
  ('drm_dd_major', ctypes.c_int32),
  ('drm_dd_minor', ctypes.c_int32),
]
class struct_drm_gem_close(Struct): pass
struct_drm_gem_close._fields_ = [
  ('handle', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_gem_flink(Struct): pass
struct_drm_gem_flink._fields_ = [
  ('handle', ctypes.c_uint32),
  ('name', ctypes.c_uint32),
]
class struct_drm_gem_open(Struct): pass
__u64 = ctypes.c_uint64
struct_drm_gem_open._fields_ = [
  ('name', ctypes.c_uint32),
  ('handle', ctypes.c_uint32),
  ('size', ctypes.c_uint64),
]
class struct_drm_get_cap(Struct): pass
struct_drm_get_cap._fields_ = [
  ('capability', ctypes.c_uint64),
  ('value', ctypes.c_uint64),
]
class struct_drm_set_client_cap(Struct): pass
struct_drm_set_client_cap._fields_ = [
  ('capability', ctypes.c_uint64),
  ('value', ctypes.c_uint64),
]
class struct_drm_prime_handle(Struct): pass
__s32 = ctypes.c_int32
struct_drm_prime_handle._fields_ = [
  ('handle', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('fd', ctypes.c_int32),
]
class struct_drm_syncobj_create(Struct): pass
struct_drm_syncobj_create._fields_ = [
  ('handle', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
]
class struct_drm_syncobj_destroy(Struct): pass
struct_drm_syncobj_destroy._fields_ = [
  ('handle', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_syncobj_handle(Struct): pass
struct_drm_syncobj_handle._fields_ = [
  ('handle', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('fd', ctypes.c_int32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_syncobj_transfer(Struct): pass
struct_drm_syncobj_transfer._fields_ = [
  ('src_handle', ctypes.c_uint32),
  ('dst_handle', ctypes.c_uint32),
  ('src_point', ctypes.c_uint64),
  ('dst_point', ctypes.c_uint64),
  ('flags', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_syncobj_wait(Struct): pass
__s64 = ctypes.c_int64
struct_drm_syncobj_wait._fields_ = [
  ('handles', ctypes.c_uint64),
  ('timeout_nsec', ctypes.c_int64),
  ('count_handles', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('first_signaled', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
  ('deadline_nsec', ctypes.c_uint64),
]
class struct_drm_syncobj_timeline_wait(Struct): pass
struct_drm_syncobj_timeline_wait._fields_ = [
  ('handles', ctypes.c_uint64),
  ('points', ctypes.c_uint64),
  ('timeout_nsec', ctypes.c_int64),
  ('count_handles', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('first_signaled', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
  ('deadline_nsec', ctypes.c_uint64),
]
class struct_drm_syncobj_eventfd(Struct): pass
struct_drm_syncobj_eventfd._fields_ = [
  ('handle', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('point', ctypes.c_uint64),
  ('fd', ctypes.c_int32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_syncobj_array(Struct): pass
struct_drm_syncobj_array._fields_ = [
  ('handles', ctypes.c_uint64),
  ('count_handles', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_syncobj_timeline_array(Struct): pass
struct_drm_syncobj_timeline_array._fields_ = [
  ('handles', ctypes.c_uint64),
  ('points', ctypes.c_uint64),
  ('count_handles', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
]
class struct_drm_crtc_get_sequence(Struct): pass
struct_drm_crtc_get_sequence._fields_ = [
  ('crtc_id', ctypes.c_uint32),
  ('active', ctypes.c_uint32),
  ('sequence', ctypes.c_uint64),
  ('sequence_ns', ctypes.c_int64),
]
class struct_drm_crtc_queue_sequence(Struct): pass
struct_drm_crtc_queue_sequence._fields_ = [
  ('crtc_id', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('sequence', ctypes.c_uint64),
  ('user_data', ctypes.c_uint64),
]
class struct_drm_event(Struct): pass
struct_drm_event._fields_ = [
  ('type', ctypes.c_uint32),
  ('length', ctypes.c_uint32),
]
class struct_drm_event_vblank(Struct): pass
struct_drm_event_vblank._fields_ = [
  ('base', struct_drm_event),
  ('user_data', ctypes.c_uint64),
  ('tv_sec', ctypes.c_uint32),
  ('tv_usec', ctypes.c_uint32),
  ('sequence', ctypes.c_uint32),
  ('crtc_id', ctypes.c_uint32),
]
class struct_drm_event_crtc_sequence(Struct): pass
struct_drm_event_crtc_sequence._fields_ = [
  ('base', struct_drm_event),
  ('user_data', ctypes.c_uint64),
  ('time_ns', ctypes.c_int64),
  ('sequence', ctypes.c_uint64),
]
drm_clip_rect_t = struct_drm_clip_rect
drm_drawable_info_t = struct_drm_drawable_info
drm_tex_region_t = struct_drm_tex_region
drm_hw_lock_t = struct_drm_hw_lock
drm_version_t = struct_drm_version
drm_unique_t = struct_drm_unique
drm_list_t = struct_drm_list
drm_block_t = struct_drm_block
drm_control_t = struct_drm_control
drm_map_type_t = enum_drm_map_type
drm_map_flags_t = enum_drm_map_flags
drm_ctx_priv_map_t = struct_drm_ctx_priv_map
drm_map_t = struct_drm_map
drm_client_t = struct_drm_client
drm_stat_type_t = enum_drm_stat_type
drm_stats_t = struct_drm_stats
drm_lock_flags_t = enum_drm_lock_flags
drm_lock_t = struct_drm_lock
drm_dma_flags_t = enum_drm_dma_flags
drm_buf_desc_t = struct_drm_buf_desc
drm_buf_info_t = struct_drm_buf_info
drm_buf_free_t = struct_drm_buf_free
drm_buf_pub_t = struct_drm_buf_pub
drm_buf_map_t = struct_drm_buf_map
drm_dma_t = struct_drm_dma
drm_wait_vblank_t = union_drm_wait_vblank
drm_agp_mode_t = struct_drm_agp_mode
drm_ctx_flags_t = enum_drm_ctx_flags
drm_ctx_t = struct_drm_ctx
drm_ctx_res_t = struct_drm_ctx_res
drm_draw_t = struct_drm_draw
drm_update_draw_t = struct_drm_update_draw
drm_auth_t = struct_drm_auth
drm_irq_busid_t = struct_drm_irq_busid
drm_vblank_seq_type_t = enum_drm_vblank_seq_type
drm_agp_buffer_t = struct_drm_agp_buffer
drm_agp_binding_t = struct_drm_agp_binding
drm_agp_info_t = struct_drm_agp_info
drm_scatter_gather_t = struct_drm_scatter_gather
drm_set_version_t = struct_drm_set_version
class struct_drm_amdgpu_gem_create_in(Struct): pass
struct_drm_amdgpu_gem_create_in._fields_ = [
  ('bo_size', ctypes.c_uint64),
  ('alignment', ctypes.c_uint64),
  ('domains', ctypes.c_uint64),
  ('domain_flags', ctypes.c_uint64),
]
class struct_drm_amdgpu_gem_create_out(Struct): pass
struct_drm_amdgpu_gem_create_out._fields_ = [
  ('handle', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
]
class union_drm_amdgpu_gem_create(ctypes.Union): pass
union_drm_amdgpu_gem_create._fields_ = [
  ('in', struct_drm_amdgpu_gem_create_in),
  ('out', struct_drm_amdgpu_gem_create_out),
]
class struct_drm_amdgpu_bo_list_in(Struct): pass
struct_drm_amdgpu_bo_list_in._fields_ = [
  ('operation', ctypes.c_uint32),
  ('list_handle', ctypes.c_uint32),
  ('bo_number', ctypes.c_uint32),
  ('bo_info_size', ctypes.c_uint32),
  ('bo_info_ptr', ctypes.c_uint64),
]
class struct_drm_amdgpu_bo_list_entry(Struct): pass
struct_drm_amdgpu_bo_list_entry._fields_ = [
  ('bo_handle', ctypes.c_uint32),
  ('bo_priority', ctypes.c_uint32),
]
class struct_drm_amdgpu_bo_list_out(Struct): pass
struct_drm_amdgpu_bo_list_out._fields_ = [
  ('list_handle', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
]
class union_drm_amdgpu_bo_list(ctypes.Union): pass
union_drm_amdgpu_bo_list._fields_ = [
  ('in', struct_drm_amdgpu_bo_list_in),
  ('out', struct_drm_amdgpu_bo_list_out),
]
class struct_drm_amdgpu_ctx_in(Struct): pass
struct_drm_amdgpu_ctx_in._fields_ = [
  ('op', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('ctx_id', ctypes.c_uint32),
  ('priority', ctypes.c_int32),
]
class union_drm_amdgpu_ctx_out(ctypes.Union): pass
class union_drm_amdgpu_ctx_out_alloc(Struct): pass
union_drm_amdgpu_ctx_out_alloc._fields_ = [
  ('ctx_id', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
]
class union_drm_amdgpu_ctx_out_state(Struct): pass
union_drm_amdgpu_ctx_out_state._fields_ = [
  ('flags', ctypes.c_uint64),
  ('hangs', ctypes.c_uint32),
  ('reset_status', ctypes.c_uint32),
]
class union_drm_amdgpu_ctx_out_pstate(Struct): pass
union_drm_amdgpu_ctx_out_pstate._fields_ = [
  ('flags', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
]
union_drm_amdgpu_ctx_out._fields_ = [
  ('alloc', union_drm_amdgpu_ctx_out_alloc),
  ('state', union_drm_amdgpu_ctx_out_state),
  ('pstate', union_drm_amdgpu_ctx_out_pstate),
]
class union_drm_amdgpu_ctx(ctypes.Union): pass
union_drm_amdgpu_ctx._fields_ = [
  ('in', struct_drm_amdgpu_ctx_in),
  ('out', union_drm_amdgpu_ctx_out),
]
class struct_drm_amdgpu_userq_in(Struct): pass
struct_drm_amdgpu_userq_in._fields_ = [
  ('op', ctypes.c_uint32),
  ('queue_id', ctypes.c_uint32),
  ('ip_type', ctypes.c_uint32),
  ('doorbell_handle', ctypes.c_uint32),
  ('doorbell_offset', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('queue_va', ctypes.c_uint64),
  ('queue_size', ctypes.c_uint64),
  ('rptr_va', ctypes.c_uint64),
  ('wptr_va', ctypes.c_uint64),
  ('mqd', ctypes.c_uint64),
  ('mqd_size', ctypes.c_uint64),
]
class struct_drm_amdgpu_userq_out(Struct): pass
struct_drm_amdgpu_userq_out._fields_ = [
  ('queue_id', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
]
class union_drm_amdgpu_userq(ctypes.Union): pass
union_drm_amdgpu_userq._fields_ = [
  ('in', struct_drm_amdgpu_userq_in),
  ('out', struct_drm_amdgpu_userq_out),
]
class struct_drm_amdgpu_userq_mqd_gfx11(Struct): pass
struct_drm_amdgpu_userq_mqd_gfx11._fields_ = [
  ('shadow_va', ctypes.c_uint64),
  ('csa_va', ctypes.c_uint64),
]
class struct_drm_amdgpu_userq_mqd_sdma_gfx11(Struct): pass
struct_drm_amdgpu_userq_mqd_sdma_gfx11._fields_ = [
  ('csa_va', ctypes.c_uint64),
]
class struct_drm_amdgpu_userq_mqd_compute_gfx11(Struct): pass
struct_drm_amdgpu_userq_mqd_compute_gfx11._fields_ = [
  ('eop_va', ctypes.c_uint64),
]
class struct_drm_amdgpu_userq_signal(Struct): pass
struct_drm_amdgpu_userq_signal._fields_ = [
  ('queue_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
  ('syncobj_handles', ctypes.c_uint64),
  ('num_syncobj_handles', ctypes.c_uint64),
  ('bo_read_handles', ctypes.c_uint64),
  ('bo_write_handles', ctypes.c_uint64),
  ('num_bo_read_handles', ctypes.c_uint32),
  ('num_bo_write_handles', ctypes.c_uint32),
]
class struct_drm_amdgpu_userq_fence_info(Struct): pass
struct_drm_amdgpu_userq_fence_info._fields_ = [
  ('va', ctypes.c_uint64),
  ('value', ctypes.c_uint64),
]
class struct_drm_amdgpu_userq_wait(Struct): pass
__u16 = ctypes.c_uint16
struct_drm_amdgpu_userq_wait._fields_ = [
  ('waitq_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
  ('syncobj_handles', ctypes.c_uint64),
  ('syncobj_timeline_handles', ctypes.c_uint64),
  ('syncobj_timeline_points', ctypes.c_uint64),
  ('bo_read_handles', ctypes.c_uint64),
  ('bo_write_handles', ctypes.c_uint64),
  ('num_syncobj_timeline_handles', ctypes.c_uint16),
  ('num_fences', ctypes.c_uint16),
  ('num_syncobj_handles', ctypes.c_uint32),
  ('num_bo_read_handles', ctypes.c_uint32),
  ('num_bo_write_handles', ctypes.c_uint32),
  ('out_fences', ctypes.c_uint64),
]
class struct_drm_amdgpu_sem_in(Struct): pass
class union_drm_amdgpu_sem_out(ctypes.Union): pass
class union_drm_amdgpu_sem(ctypes.Union): pass
class struct_drm_amdgpu_vm_in(Struct): pass
struct_drm_amdgpu_vm_in._fields_ = [
  ('op', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
]
class struct_drm_amdgpu_vm_out(Struct): pass
struct_drm_amdgpu_vm_out._fields_ = [
  ('flags', ctypes.c_uint64),
]
class union_drm_amdgpu_vm(ctypes.Union): pass
union_drm_amdgpu_vm._fields_ = [
  ('in', struct_drm_amdgpu_vm_in),
  ('out', struct_drm_amdgpu_vm_out),
]
class struct_drm_amdgpu_sched_in(Struct): pass
struct_drm_amdgpu_sched_in._fields_ = [
  ('op', ctypes.c_uint32),
  ('fd', ctypes.c_uint32),
  ('priority', ctypes.c_int32),
  ('ctx_id', ctypes.c_uint32),
]
class union_drm_amdgpu_sched(ctypes.Union): pass
union_drm_amdgpu_sched._fields_ = [
  ('in', struct_drm_amdgpu_sched_in),
]
class struct_drm_amdgpu_gem_userptr(Struct): pass
struct_drm_amdgpu_gem_userptr._fields_ = [
  ('addr', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('flags', ctypes.c_uint32),
  ('handle', ctypes.c_uint32),
]
class struct_drm_amdgpu_gem_dgma(Struct): pass
struct_drm_amdgpu_gem_dgma._fields_ = [
  ('addr', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('op', ctypes.c_uint32),
  ('handle', ctypes.c_uint32),
]
class struct_drm_amdgpu_gem_metadata(Struct): pass
class struct_drm_amdgpu_gem_metadata_data(Struct): pass
struct_drm_amdgpu_gem_metadata_data._fields_ = [
  ('flags', ctypes.c_uint64),
  ('tiling_info', ctypes.c_uint64),
  ('data_size_bytes', ctypes.c_uint32),
  ('data', (ctypes.c_uint32 * 64)),
]
struct_drm_amdgpu_gem_metadata._fields_ = [
  ('handle', ctypes.c_uint32),
  ('op', ctypes.c_uint32),
  ('data', struct_drm_amdgpu_gem_metadata_data),
]
class struct_drm_amdgpu_gem_mmap_in(Struct): pass
struct_drm_amdgpu_gem_mmap_in._fields_ = [
  ('handle', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
]
class struct_drm_amdgpu_gem_mmap_out(Struct): pass
struct_drm_amdgpu_gem_mmap_out._fields_ = [
  ('addr_ptr', ctypes.c_uint64),
]
class union_drm_amdgpu_gem_mmap(ctypes.Union): pass
union_drm_amdgpu_gem_mmap._fields_ = [
  ('in', struct_drm_amdgpu_gem_mmap_in),
  ('out', struct_drm_amdgpu_gem_mmap_out),
]
class struct_drm_amdgpu_gem_wait_idle_in(Struct): pass
struct_drm_amdgpu_gem_wait_idle_in._fields_ = [
  ('handle', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('timeout', ctypes.c_uint64),
]
class struct_drm_amdgpu_gem_wait_idle_out(Struct): pass
struct_drm_amdgpu_gem_wait_idle_out._fields_ = [
  ('status', ctypes.c_uint32),
  ('domain', ctypes.c_uint32),
]
class union_drm_amdgpu_gem_wait_idle(ctypes.Union): pass
union_drm_amdgpu_gem_wait_idle._fields_ = [
  ('in', struct_drm_amdgpu_gem_wait_idle_in),
  ('out', struct_drm_amdgpu_gem_wait_idle_out),
]
class struct_drm_amdgpu_wait_cs_in(Struct): pass
struct_drm_amdgpu_wait_cs_in._fields_ = [
  ('handle', ctypes.c_uint64),
  ('timeout', ctypes.c_uint64),
  ('ip_type', ctypes.c_uint32),
  ('ip_instance', ctypes.c_uint32),
  ('ring', ctypes.c_uint32),
  ('ctx_id', ctypes.c_uint32),
]
class struct_drm_amdgpu_wait_cs_out(Struct): pass
struct_drm_amdgpu_wait_cs_out._fields_ = [
  ('status', ctypes.c_uint64),
]
class union_drm_amdgpu_wait_cs(ctypes.Union): pass
union_drm_amdgpu_wait_cs._fields_ = [
  ('in', struct_drm_amdgpu_wait_cs_in),
  ('out', struct_drm_amdgpu_wait_cs_out),
]
class struct_drm_amdgpu_fence(Struct): pass
struct_drm_amdgpu_fence._fields_ = [
  ('ctx_id', ctypes.c_uint32),
  ('ip_type', ctypes.c_uint32),
  ('ip_instance', ctypes.c_uint32),
  ('ring', ctypes.c_uint32),
  ('seq_no', ctypes.c_uint64),
]
class struct_drm_amdgpu_wait_fences_in(Struct): pass
struct_drm_amdgpu_wait_fences_in._fields_ = [
  ('fences', ctypes.c_uint64),
  ('fence_count', ctypes.c_uint32),
  ('wait_all', ctypes.c_uint32),
  ('timeout_ns', ctypes.c_uint64),
]
class struct_drm_amdgpu_wait_fences_out(Struct): pass
struct_drm_amdgpu_wait_fences_out._fields_ = [
  ('status', ctypes.c_uint32),
  ('first_signaled', ctypes.c_uint32),
]
class union_drm_amdgpu_wait_fences(ctypes.Union): pass
union_drm_amdgpu_wait_fences._fields_ = [
  ('in', struct_drm_amdgpu_wait_fences_in),
  ('out', struct_drm_amdgpu_wait_fences_out),
]
class struct_drm_amdgpu_gem_op(Struct): pass
struct_drm_amdgpu_gem_op._fields_ = [
  ('handle', ctypes.c_uint32),
  ('op', ctypes.c_uint32),
  ('value', ctypes.c_uint64),
]
class struct_drm_amdgpu_gem_va(Struct): pass
struct_drm_amdgpu_gem_va._fields_ = [
  ('handle', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
  ('operation', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('va_address', ctypes.c_uint64),
  ('offset_in_bo', ctypes.c_uint64),
  ('map_size', ctypes.c_uint64),
  ('vm_timeline_point', ctypes.c_uint64),
  ('vm_timeline_syncobj_out', ctypes.c_uint32),
  ('num_syncobj_handles', ctypes.c_uint32),
  ('input_fence_syncobj_handles', ctypes.c_uint64),
]
class struct_drm_amdgpu_cs_chunk(Struct): pass
struct_drm_amdgpu_cs_chunk._fields_ = [
  ('chunk_id', ctypes.c_uint32),
  ('length_dw', ctypes.c_uint32),
  ('chunk_data', ctypes.c_uint64),
]
class struct_drm_amdgpu_cs_in(Struct): pass
struct_drm_amdgpu_cs_in._fields_ = [
  ('ctx_id', ctypes.c_uint32),
  ('bo_list_handle', ctypes.c_uint32),
  ('num_chunks', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('chunks', ctypes.c_uint64),
]
class struct_drm_amdgpu_cs_out(Struct): pass
struct_drm_amdgpu_cs_out._fields_ = [
  ('handle', ctypes.c_uint64),
]
class union_drm_amdgpu_cs(ctypes.Union): pass
union_drm_amdgpu_cs._fields_ = [
  ('in', struct_drm_amdgpu_cs_in),
  ('out', struct_drm_amdgpu_cs_out),
]
class struct_drm_amdgpu_cs_chunk_ib(Struct): pass
struct_drm_amdgpu_cs_chunk_ib._fields_ = [
  ('_pad', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('va_start', ctypes.c_uint64),
  ('ib_bytes', ctypes.c_uint32),
  ('ip_type', ctypes.c_uint32),
  ('ip_instance', ctypes.c_uint32),
  ('ring', ctypes.c_uint32),
]
class struct_drm_amdgpu_cs_chunk_dep(Struct): pass
struct_drm_amdgpu_cs_chunk_dep._fields_ = [
  ('ip_type', ctypes.c_uint32),
  ('ip_instance', ctypes.c_uint32),
  ('ring', ctypes.c_uint32),
  ('ctx_id', ctypes.c_uint32),
  ('handle', ctypes.c_uint64),
]
class struct_drm_amdgpu_cs_chunk_fence(Struct): pass
struct_drm_amdgpu_cs_chunk_fence._fields_ = [
  ('handle', ctypes.c_uint32),
  ('offset', ctypes.c_uint32),
]
class struct_drm_amdgpu_cs_chunk_sem(Struct): pass
struct_drm_amdgpu_cs_chunk_sem._fields_ = [
  ('handle', ctypes.c_uint32),
]
class struct_drm_amdgpu_cs_chunk_syncobj(Struct): pass
struct_drm_amdgpu_cs_chunk_syncobj._fields_ = [
  ('handle', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('point', ctypes.c_uint64),
]
class union_drm_amdgpu_fence_to_handle(ctypes.Union): pass
class union_drm_amdgpu_fence_to_handle_in(Struct): pass
union_drm_amdgpu_fence_to_handle_in._fields_ = [
  ('fence', struct_drm_amdgpu_fence),
  ('what', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class union_drm_amdgpu_fence_to_handle_out(Struct): pass
union_drm_amdgpu_fence_to_handle_out._fields_ = [
  ('handle', ctypes.c_uint32),
]
union_drm_amdgpu_fence_to_handle._fields_ = [
  ('in', union_drm_amdgpu_fence_to_handle_in),
  ('out', union_drm_amdgpu_fence_to_handle_out),
]
class struct_drm_amdgpu_cs_chunk_data(Struct): pass
class struct_drm_amdgpu_cs_chunk_data_0(ctypes.Union): pass
struct_drm_amdgpu_cs_chunk_data_0._fields_ = [
  ('ib_data', struct_drm_amdgpu_cs_chunk_ib),
  ('fence_data', struct_drm_amdgpu_cs_chunk_fence),
]
struct_drm_amdgpu_cs_chunk_data._anonymous_ = ['_0']
struct_drm_amdgpu_cs_chunk_data._fields_ = [
  ('_0', struct_drm_amdgpu_cs_chunk_data_0),
]
class struct_drm_amdgpu_cs_chunk_cp_gfx_shadow(Struct): pass
struct_drm_amdgpu_cs_chunk_cp_gfx_shadow._fields_ = [
  ('shadow_va', ctypes.c_uint64),
  ('csa_va', ctypes.c_uint64),
  ('gds_va', ctypes.c_uint64),
  ('flags', ctypes.c_uint64),
]
class struct_drm_amdgpu_query_fw(Struct): pass
struct_drm_amdgpu_query_fw._fields_ = [
  ('fw_type', ctypes.c_uint32),
  ('ip_instance', ctypes.c_uint32),
  ('index', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
]
class struct_drm_amdgpu_info(Struct): pass
struct_drm_amdgpu_info._fields_ = [
  ('return_pointer', ctypes.c_uint64),
  ('return_size', ctypes.c_uint32),
  ('query', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_gds(Struct): pass
struct_drm_amdgpu_info_gds._fields_ = [
  ('gds_gfx_partition_size', ctypes.c_uint32),
  ('compute_partition_size', ctypes.c_uint32),
  ('gds_total_size', ctypes.c_uint32),
  ('gws_per_gfx_partition', ctypes.c_uint32),
  ('gws_per_compute_partition', ctypes.c_uint32),
  ('oa_per_gfx_partition', ctypes.c_uint32),
  ('oa_per_compute_partition', ctypes.c_uint32),
  ('_pad', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_vram_gtt(Struct): pass
struct_drm_amdgpu_info_vram_gtt._fields_ = [
  ('vram_size', ctypes.c_uint64),
  ('vram_cpu_accessible_size', ctypes.c_uint64),
  ('gtt_size', ctypes.c_uint64),
]
class struct_drm_amdgpu_heap_info(Struct): pass
struct_drm_amdgpu_heap_info._fields_ = [
  ('total_heap_size', ctypes.c_uint64),
  ('usable_heap_size', ctypes.c_uint64),
  ('heap_usage', ctypes.c_uint64),
  ('max_allocation', ctypes.c_uint64),
]
class struct_drm_amdgpu_memory_info(Struct): pass
struct_drm_amdgpu_memory_info._fields_ = [
  ('vram', struct_drm_amdgpu_heap_info),
  ('cpu_accessible_vram', struct_drm_amdgpu_heap_info),
  ('gtt', struct_drm_amdgpu_heap_info),
]
class struct_drm_amdgpu_info_firmware(Struct): pass
struct_drm_amdgpu_info_firmware._fields_ = [
  ('ver', ctypes.c_uint32),
  ('feature', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_vbios(Struct): pass
__u8 = ctypes.c_ubyte
struct_drm_amdgpu_info_vbios._fields_ = [
  ('name', (ctypes.c_ubyte * 64)),
  ('vbios_pn', (ctypes.c_ubyte * 64)),
  ('version', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
  ('vbios_ver_str', (ctypes.c_ubyte * 32)),
  ('date', (ctypes.c_ubyte * 32)),
]
class struct_drm_amdgpu_info_device(Struct): pass
struct_drm_amdgpu_info_device._fields_ = [
  ('device_id', ctypes.c_uint32),
  ('chip_rev', ctypes.c_uint32),
  ('external_rev', ctypes.c_uint32),
  ('pci_rev', ctypes.c_uint32),
  ('family', ctypes.c_uint32),
  ('num_shader_engines', ctypes.c_uint32),
  ('num_shader_arrays_per_engine', ctypes.c_uint32),
  ('gpu_counter_freq', ctypes.c_uint32),
  ('max_engine_clock', ctypes.c_uint64),
  ('max_memory_clock', ctypes.c_uint64),
  ('cu_active_number', ctypes.c_uint32),
  ('cu_ao_mask', ctypes.c_uint32),
  ('cu_bitmap', ((ctypes.c_uint32 * 4) * 4)),
  ('enabled_rb_pipes_mask', ctypes.c_uint32),
  ('num_rb_pipes', ctypes.c_uint32),
  ('num_hw_gfx_contexts', ctypes.c_uint32),
  ('pcie_gen', ctypes.c_uint32),
  ('ids_flags', ctypes.c_uint64),
  ('virtual_address_offset', ctypes.c_uint64),
  ('virtual_address_max', ctypes.c_uint64),
  ('virtual_address_alignment', ctypes.c_uint32),
  ('pte_fragment_size', ctypes.c_uint32),
  ('gart_page_size', ctypes.c_uint32),
  ('ce_ram_size', ctypes.c_uint32),
  ('vram_type', ctypes.c_uint32),
  ('vram_bit_width', ctypes.c_uint32),
  ('vce_harvest_config', ctypes.c_uint32),
  ('gc_double_offchip_lds_buf', ctypes.c_uint32),
  ('prim_buf_gpu_addr', ctypes.c_uint64),
  ('pos_buf_gpu_addr', ctypes.c_uint64),
  ('cntl_sb_buf_gpu_addr', ctypes.c_uint64),
  ('param_buf_gpu_addr', ctypes.c_uint64),
  ('prim_buf_size', ctypes.c_uint32),
  ('pos_buf_size', ctypes.c_uint32),
  ('cntl_sb_buf_size', ctypes.c_uint32),
  ('param_buf_size', ctypes.c_uint32),
  ('wave_front_size', ctypes.c_uint32),
  ('num_shader_visible_vgprs', ctypes.c_uint32),
  ('num_cu_per_sh', ctypes.c_uint32),
  ('num_tcc_blocks', ctypes.c_uint32),
  ('gs_vgt_table_depth', ctypes.c_uint32),
  ('gs_prim_buffer_depth', ctypes.c_uint32),
  ('max_gs_waves_per_vgt', ctypes.c_uint32),
  ('pcie_num_lanes', ctypes.c_uint32),
  ('cu_ao_bitmap', ((ctypes.c_uint32 * 4) * 4)),
  ('high_va_offset', ctypes.c_uint64),
  ('high_va_max', ctypes.c_uint64),
  ('pa_sc_tile_steering_override', ctypes.c_uint32),
  ('tcc_disabled_mask', ctypes.c_uint64),
  ('min_engine_clock', ctypes.c_uint64),
  ('min_memory_clock', ctypes.c_uint64),
  ('tcp_cache_size', ctypes.c_uint32),
  ('num_sqc_per_wgp', ctypes.c_uint32),
  ('sqc_data_cache_size', ctypes.c_uint32),
  ('sqc_inst_cache_size', ctypes.c_uint32),
  ('gl1c_cache_size', ctypes.c_uint32),
  ('gl2c_cache_size', ctypes.c_uint32),
  ('mall_size', ctypes.c_uint64),
  ('enabled_rb_pipes_mask_hi', ctypes.c_uint32),
  ('shadow_size', ctypes.c_uint32),
  ('shadow_alignment', ctypes.c_uint32),
  ('csa_size', ctypes.c_uint32),
  ('csa_alignment', ctypes.c_uint32),
  ('userq_ip_mask', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_hw_ip(Struct): pass
struct_drm_amdgpu_info_hw_ip._fields_ = [
  ('hw_ip_version_major', ctypes.c_uint32),
  ('hw_ip_version_minor', ctypes.c_uint32),
  ('capabilities_flags', ctypes.c_uint64),
  ('ib_start_alignment', ctypes.c_uint32),
  ('ib_size_alignment', ctypes.c_uint32),
  ('available_rings', ctypes.c_uint32),
  ('ip_discovery_version', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_uq_fw_areas_gfx(Struct): pass
struct_drm_amdgpu_info_uq_fw_areas_gfx._fields_ = [
  ('shadow_size', ctypes.c_uint32),
  ('shadow_alignment', ctypes.c_uint32),
  ('csa_size', ctypes.c_uint32),
  ('csa_alignment', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_uq_fw_areas(Struct): pass
class struct_drm_amdgpu_info_uq_fw_areas_0(ctypes.Union): pass
struct_drm_amdgpu_info_uq_fw_areas_0._fields_ = [
  ('gfx', struct_drm_amdgpu_info_uq_fw_areas_gfx),
]
struct_drm_amdgpu_info_uq_fw_areas._anonymous_ = ['_0']
struct_drm_amdgpu_info_uq_fw_areas._fields_ = [
  ('_0', struct_drm_amdgpu_info_uq_fw_areas_0),
]
class struct_drm_amdgpu_info_num_handles(Struct): pass
struct_drm_amdgpu_info_num_handles._fields_ = [
  ('uvd_max_handles', ctypes.c_uint32),
  ('uvd_used_handles', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_vce_clock_table_entry(Struct): pass
struct_drm_amdgpu_info_vce_clock_table_entry._fields_ = [
  ('sclk', ctypes.c_uint32),
  ('mclk', ctypes.c_uint32),
  ('eclk', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_vce_clock_table(Struct): pass
struct_drm_amdgpu_info_vce_clock_table._fields_ = [
  ('entries', (struct_drm_amdgpu_info_vce_clock_table_entry * 6)),
  ('num_valid_entries', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_video_codec_info(Struct): pass
struct_drm_amdgpu_info_video_codec_info._fields_ = [
  ('valid', ctypes.c_uint32),
  ('max_width', ctypes.c_uint32),
  ('max_height', ctypes.c_uint32),
  ('max_pixels_per_frame', ctypes.c_uint32),
  ('max_level', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_video_caps(Struct): pass
struct_drm_amdgpu_info_video_caps._fields_ = [
  ('codec_info', (struct_drm_amdgpu_info_video_codec_info * 8)),
]
class struct_drm_amdgpu_info_gpuvm_fault(Struct): pass
struct_drm_amdgpu_info_gpuvm_fault._fields_ = [
  ('addr', ctypes.c_uint64),
  ('status', ctypes.c_uint32),
  ('vmhub', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_uq_metadata_gfx(Struct): pass
struct_drm_amdgpu_info_uq_metadata_gfx._fields_ = [
  ('shadow_size', ctypes.c_uint32),
  ('shadow_alignment', ctypes.c_uint32),
  ('csa_size', ctypes.c_uint32),
  ('csa_alignment', ctypes.c_uint32),
]
class struct_drm_amdgpu_info_uq_metadata(Struct): pass
class struct_drm_amdgpu_info_uq_metadata_0(ctypes.Union): pass
struct_drm_amdgpu_info_uq_metadata_0._fields_ = [
  ('gfx', struct_drm_amdgpu_info_uq_metadata_gfx),
]
struct_drm_amdgpu_info_uq_metadata._anonymous_ = ['_0']
struct_drm_amdgpu_info_uq_metadata._fields_ = [
  ('_0', struct_drm_amdgpu_info_uq_metadata_0),
]
class _anonstruct0(Struct): pass
class struct_drm_amdgpu_virtual_range(Struct): pass
class struct_drm_amdgpu_capability(Struct): pass
struct_drm_amdgpu_capability._fields_ = [
  ('flag', ctypes.c_uint32),
  ('direct_gma_size', ctypes.c_uint32),
]
class struct_drm_amdgpu_freesync(Struct): pass
struct_drm_amdgpu_freesync._fields_ = [
  ('op', ctypes.c_uint32),
  ('spare', (ctypes.c_uint32 * 7)),
]
DRM_NAME = "drm"
DRM_MIN_ORDER = 5
DRM_MAX_ORDER = 22
DRM_RAM_PERCENT = 10
_DRM_LOCK_HELD = 0x80000000
_DRM_LOCK_CONT = 0x40000000
_DRM_LOCK_IS_HELD = lambda lock: ((lock) & _DRM_LOCK_HELD)
_DRM_LOCK_IS_CONT = lambda lock: ((lock) & _DRM_LOCK_CONT)
_DRM_LOCKING_CONTEXT = lambda lock: ((lock) & ~(_DRM_LOCK_HELD|_DRM_LOCK_CONT))
_DRM_VBLANK_HIGH_CRTC_SHIFT = 1
_DRM_VBLANK_TYPES_MASK = (_DRM_VBLANK_ABSOLUTE | _DRM_VBLANK_RELATIVE)
_DRM_VBLANK_FLAGS_MASK = (_DRM_VBLANK_EVENT | _DRM_VBLANK_SIGNAL | _DRM_VBLANK_SECONDARY | _DRM_VBLANK_NEXTONMISS)
_DRM_PRE_MODESET = 1
_DRM_POST_MODESET = 2
DRM_CAP_DUMB_BUFFER = 0x1
DRM_CAP_VBLANK_HIGH_CRTC = 0x2
DRM_CAP_DUMB_PREFERRED_DEPTH = 0x3
DRM_CAP_DUMB_PREFER_SHADOW = 0x4
DRM_CAP_PRIME = 0x5
DRM_PRIME_CAP_IMPORT = 0x1
DRM_PRIME_CAP_EXPORT = 0x2
DRM_CAP_TIMESTAMP_MONOTONIC = 0x6
DRM_CAP_ASYNC_PAGE_FLIP = 0x7
DRM_CAP_CURSOR_WIDTH = 0x8
DRM_CAP_CURSOR_HEIGHT = 0x9
DRM_CAP_ADDFB2_MODIFIERS = 0x10
DRM_CAP_PAGE_FLIP_TARGET = 0x11
DRM_CAP_CRTC_IN_VBLANK_EVENT = 0x12
DRM_CAP_SYNCOBJ = 0x13
DRM_CAP_SYNCOBJ_TIMELINE = 0x14
DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP = 0x15
DRM_CLIENT_CAP_STEREO_3D = 1
DRM_CLIENT_CAP_UNIVERSAL_PLANES = 2
DRM_CLIENT_CAP_ATOMIC = 3
DRM_CLIENT_CAP_ASPECT_RATIO = 4
DRM_CLIENT_CAP_WRITEBACK_CONNECTORS = 5
DRM_CLIENT_CAP_CURSOR_PLANE_HOTSPOT = 6
DRM_SYNCOBJ_CREATE_SIGNALED = (1 << 0)
DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE = (1 << 0)
DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE = (1 << 0)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL = (1 << 0)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT = (1 << 1)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE = (1 << 2)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE = (1 << 3)
DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED = (1 << 0)
DRM_CRTC_SEQUENCE_RELATIVE = 0x00000001
DRM_CRTC_SEQUENCE_NEXT_ON_MISS = 0x00000002
DRM_IOCTL_BASE = 'd'
DRM_IO = lambda nr: _IO(DRM_IOCTL_BASE,nr)
DRM_IOR = lambda nr,type: _IOR(DRM_IOCTL_BASE,nr,type)
DRM_IOW = lambda nr,type: _IOW(DRM_IOCTL_BASE,nr,type)
DRM_IOWR = lambda nr,type: _IOWR(DRM_IOCTL_BASE,nr,type)
DRM_IOCTL_VERSION = DRM_IOWR(0x00, struct_drm_version)
DRM_IOCTL_GET_UNIQUE = DRM_IOWR(0x01, struct_drm_unique)
DRM_IOCTL_GET_MAGIC = DRM_IOR( 0x02, struct_drm_auth)
DRM_IOCTL_IRQ_BUSID = DRM_IOWR(0x03, struct_drm_irq_busid)
DRM_IOCTL_GET_MAP = DRM_IOWR(0x04, struct_drm_map)
DRM_IOCTL_GET_CLIENT = DRM_IOWR(0x05, struct_drm_client)
DRM_IOCTL_GET_STATS = DRM_IOR( 0x06, struct_drm_stats)
DRM_IOCTL_SET_VERSION = DRM_IOWR(0x07, struct_drm_set_version)
DRM_IOCTL_MODESET_CTL = DRM_IOW(0x08, struct_drm_modeset_ctl)
DRM_IOCTL_GEM_CLOSE = DRM_IOW (0x09, struct_drm_gem_close)
DRM_IOCTL_GEM_FLINK = DRM_IOWR(0x0a, struct_drm_gem_flink)
DRM_IOCTL_GEM_OPEN = DRM_IOWR(0x0b, struct_drm_gem_open)
DRM_IOCTL_GET_CAP = DRM_IOWR(0x0c, struct_drm_get_cap)
DRM_IOCTL_SET_CLIENT_CAP = DRM_IOW( 0x0d, struct_drm_set_client_cap)
DRM_IOCTL_SET_UNIQUE = DRM_IOW( 0x10, struct_drm_unique)
DRM_IOCTL_AUTH_MAGIC = DRM_IOW( 0x11, struct_drm_auth)
DRM_IOCTL_BLOCK = DRM_IOWR(0x12, struct_drm_block)
DRM_IOCTL_UNBLOCK = DRM_IOWR(0x13, struct_drm_block)
DRM_IOCTL_CONTROL = DRM_IOW( 0x14, struct_drm_control)
DRM_IOCTL_ADD_MAP = DRM_IOWR(0x15, struct_drm_map)
DRM_IOCTL_ADD_BUFS = DRM_IOWR(0x16, struct_drm_buf_desc)
DRM_IOCTL_MARK_BUFS = DRM_IOW( 0x17, struct_drm_buf_desc)
DRM_IOCTL_INFO_BUFS = DRM_IOWR(0x18, struct_drm_buf_info)
DRM_IOCTL_MAP_BUFS = DRM_IOWR(0x19, struct_drm_buf_map)
DRM_IOCTL_FREE_BUFS = DRM_IOW( 0x1a, struct_drm_buf_free)
DRM_IOCTL_RM_MAP = DRM_IOW( 0x1b, struct_drm_map)
DRM_IOCTL_SET_SAREA_CTX = DRM_IOW( 0x1c, struct_drm_ctx_priv_map)
DRM_IOCTL_GET_SAREA_CTX = DRM_IOWR(0x1d, struct_drm_ctx_priv_map)
DRM_IOCTL_SET_MASTER = DRM_IO(0x1e)
DRM_IOCTL_DROP_MASTER = DRM_IO(0x1f)
DRM_IOCTL_ADD_CTX = DRM_IOWR(0x20, struct_drm_ctx)
DRM_IOCTL_RM_CTX = DRM_IOWR(0x21, struct_drm_ctx)
DRM_IOCTL_MOD_CTX = DRM_IOW( 0x22, struct_drm_ctx)
DRM_IOCTL_GET_CTX = DRM_IOWR(0x23, struct_drm_ctx)
DRM_IOCTL_SWITCH_CTX = DRM_IOW( 0x24, struct_drm_ctx)
DRM_IOCTL_NEW_CTX = DRM_IOW( 0x25, struct_drm_ctx)
DRM_IOCTL_RES_CTX = DRM_IOWR(0x26, struct_drm_ctx_res)
DRM_IOCTL_ADD_DRAW = DRM_IOWR(0x27, struct_drm_draw)
DRM_IOCTL_RM_DRAW = DRM_IOWR(0x28, struct_drm_draw)
DRM_IOCTL_DMA = DRM_IOWR(0x29, struct_drm_dma)
DRM_IOCTL_LOCK = DRM_IOW( 0x2a, struct_drm_lock)
DRM_IOCTL_UNLOCK = DRM_IOW( 0x2b, struct_drm_lock)
DRM_IOCTL_FINISH = DRM_IOW( 0x2c, struct_drm_lock)
DRM_IOCTL_PRIME_HANDLE_TO_FD = DRM_IOWR(0x2d, struct_drm_prime_handle)
DRM_IOCTL_PRIME_FD_TO_HANDLE = DRM_IOWR(0x2e, struct_drm_prime_handle)
DRM_IOCTL_AGP_ACQUIRE = DRM_IO(  0x30)
DRM_IOCTL_AGP_RELEASE = DRM_IO(  0x31)
DRM_IOCTL_AGP_ENABLE = DRM_IOW( 0x32, struct_drm_agp_mode)
DRM_IOCTL_AGP_INFO = DRM_IOR( 0x33, struct_drm_agp_info)
DRM_IOCTL_AGP_ALLOC = DRM_IOWR(0x34, struct_drm_agp_buffer)
DRM_IOCTL_AGP_FREE = DRM_IOW( 0x35, struct_drm_agp_buffer)
DRM_IOCTL_AGP_BIND = DRM_IOW( 0x36, struct_drm_agp_binding)
DRM_IOCTL_AGP_UNBIND = DRM_IOW( 0x37, struct_drm_agp_binding)
DRM_IOCTL_SG_ALLOC = DRM_IOWR(0x38, struct_drm_scatter_gather)
DRM_IOCTL_SG_FREE = DRM_IOW( 0x39, struct_drm_scatter_gather)
DRM_IOCTL_WAIT_VBLANK = DRM_IOWR(0x3a, union_drm_wait_vblank)
DRM_IOCTL_CRTC_GET_SEQUENCE = DRM_IOWR(0x3b, struct_drm_crtc_get_sequence)
DRM_IOCTL_CRTC_QUEUE_SEQUENCE = DRM_IOWR(0x3c, struct_drm_crtc_queue_sequence)
DRM_IOCTL_UPDATE_DRAW = DRM_IOW(0x3f, struct_drm_update_draw)
DRM_IOCTL_SYNCOBJ_CREATE = DRM_IOWR(0xBF, struct_drm_syncobj_create)
DRM_IOCTL_SYNCOBJ_DESTROY = DRM_IOWR(0xC0, struct_drm_syncobj_destroy)
DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD = DRM_IOWR(0xC1, struct_drm_syncobj_handle)
DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE = DRM_IOWR(0xC2, struct_drm_syncobj_handle)
DRM_IOCTL_SYNCOBJ_WAIT = DRM_IOWR(0xC3, struct_drm_syncobj_wait)
DRM_IOCTL_SYNCOBJ_RESET = DRM_IOWR(0xC4, struct_drm_syncobj_array)
DRM_IOCTL_SYNCOBJ_SIGNAL = DRM_IOWR(0xC5, struct_drm_syncobj_array)
DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT = DRM_IOWR(0xCA, struct_drm_syncobj_timeline_wait)
DRM_IOCTL_SYNCOBJ_QUERY = DRM_IOWR(0xCB, struct_drm_syncobj_timeline_array)
DRM_IOCTL_SYNCOBJ_TRANSFER = DRM_IOWR(0xCC, struct_drm_syncobj_transfer)
DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL = DRM_IOWR(0xCD, struct_drm_syncobj_timeline_array)
DRM_IOCTL_SYNCOBJ_EVENTFD = DRM_IOWR(0xCF, struct_drm_syncobj_eventfd)
DRM_COMMAND_BASE = 0x40
DRM_COMMAND_END = 0xA0
DRM_EVENT_VBLANK = 0x01
DRM_EVENT_FLIP_COMPLETE = 0x02
DRM_EVENT_CRTC_SEQUENCE = 0x03
DRM_AMDGPU_GEM_CREATE = 0x00
DRM_AMDGPU_GEM_MMAP = 0x01
DRM_AMDGPU_CTX = 0x02
DRM_AMDGPU_BO_LIST = 0x03
DRM_AMDGPU_CS = 0x04
DRM_AMDGPU_INFO = 0x05
DRM_AMDGPU_GEM_METADATA = 0x06
DRM_AMDGPU_GEM_WAIT_IDLE = 0x07
DRM_AMDGPU_GEM_VA = 0x08
DRM_AMDGPU_WAIT_CS = 0x09
DRM_AMDGPU_GEM_OP = 0x10
DRM_AMDGPU_GEM_USERPTR = 0x11
DRM_AMDGPU_WAIT_FENCES = 0x12
DRM_AMDGPU_VM = 0x13
DRM_AMDGPU_FENCE_TO_HANDLE = 0x14
DRM_AMDGPU_SCHED = 0x15
DRM_AMDGPU_USERQ = 0x16
DRM_AMDGPU_USERQ_SIGNAL = 0x17
DRM_AMDGPU_USERQ_WAIT = 0x18
DRM_AMDGPU_GEM_DGMA = 0x5c
DRM_AMDGPU_SEM = 0x5b
DRM_IOCTL_AMDGPU_GEM_CREATE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE, union_drm_amdgpu_gem_create)
DRM_IOCTL_AMDGPU_GEM_MMAP = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_MMAP, union_drm_amdgpu_gem_mmap)
DRM_IOCTL_AMDGPU_CTX = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union_drm_amdgpu_ctx)
DRM_IOCTL_AMDGPU_BO_LIST = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_BO_LIST, union_drm_amdgpu_bo_list)
DRM_IOCTL_AMDGPU_CS = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CS, union_drm_amdgpu_cs)
DRM_IOCTL_AMDGPU_INFO = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_INFO, struct_drm_amdgpu_info)
DRM_IOCTL_AMDGPU_GEM_METADATA = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_METADATA, struct_drm_amdgpu_gem_metadata)
DRM_IOCTL_AMDGPU_GEM_WAIT_IDLE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_WAIT_IDLE, union_drm_amdgpu_gem_wait_idle)
DRM_IOCTL_AMDGPU_GEM_VA = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA, struct_drm_amdgpu_gem_va)
DRM_IOCTL_AMDGPU_WAIT_CS = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_CS, union_drm_amdgpu_wait_cs)
DRM_IOCTL_AMDGPU_GEM_OP = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_OP, struct_drm_amdgpu_gem_op)
DRM_IOCTL_AMDGPU_GEM_USERPTR = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_USERPTR, struct_drm_amdgpu_gem_userptr)
DRM_IOCTL_AMDGPU_WAIT_FENCES = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_FENCES, union_drm_amdgpu_wait_fences)
DRM_IOCTL_AMDGPU_VM = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_VM, union_drm_amdgpu_vm)
DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_FENCE_TO_HANDLE, union_drm_amdgpu_fence_to_handle)
DRM_IOCTL_AMDGPU_SCHED = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_SCHED, union_drm_amdgpu_sched)
DRM_IOCTL_AMDGPU_USERQ = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ, union_drm_amdgpu_userq)
DRM_IOCTL_AMDGPU_USERQ_SIGNAL = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ_SIGNAL, struct_drm_amdgpu_userq_signal)
DRM_IOCTL_AMDGPU_USERQ_WAIT = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ_WAIT, struct_drm_amdgpu_userq_wait)
DRM_IOCTL_AMDGPU_GEM_DGMA = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_DGMA, struct_drm_amdgpu_gem_dgma)
DRM_IOCTL_AMDGPU_SEM = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_SEM, union_drm_amdgpu_sem)
AMDGPU_GEM_DOMAIN_CPU = 0x1
AMDGPU_GEM_DOMAIN_GTT = 0x2
AMDGPU_GEM_DOMAIN_VRAM = 0x4
AMDGPU_GEM_DOMAIN_GDS = 0x8
AMDGPU_GEM_DOMAIN_GWS = 0x10
AMDGPU_GEM_DOMAIN_OA = 0x20
AMDGPU_GEM_DOMAIN_DOORBELL = 0x40
AMDGPU_GEM_DOMAIN_DGMA = 0x400
AMDGPU_GEM_DOMAIN_DGMA_IMPORT = 0x800
AMDGPU_GEM_DOMAIN_MASK = (AMDGPU_GEM_DOMAIN_CPU | AMDGPU_GEM_DOMAIN_GTT | AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GDS | AMDGPU_GEM_DOMAIN_GWS | AMDGPU_GEM_DOMAIN_OA | AMDGPU_GEM_DOMAIN_DOORBELL | AMDGPU_GEM_DOMAIN_DGMA | AMDGPU_GEM_DOMAIN_DGMA_IMPORT)
AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED = (1 << 0)
AMDGPU_GEM_CREATE_NO_CPU_ACCESS = (1 << 1)
AMDGPU_GEM_CREATE_CPU_GTT_USWC = (1 << 2)
AMDGPU_GEM_CREATE_VRAM_CLEARED = (1 << 3)
AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS = (1 << 5)
AMDGPU_GEM_CREATE_VM_ALWAYS_VALID = (1 << 6)
AMDGPU_GEM_CREATE_EXPLICIT_SYNC = (1 << 7)
AMDGPU_GEM_CREATE_CP_MQD_GFX9 = (1 << 8)
AMDGPU_GEM_CREATE_VRAM_WIPE_ON_RELEASE = (1 << 9)
AMDGPU_GEM_CREATE_ENCRYPTED = (1 << 10)
AMDGPU_GEM_CREATE_PREEMPTIBLE = (1 << 11)
AMDGPU_GEM_CREATE_DISCARDABLE = (1 << 12)
AMDGPU_GEM_CREATE_COHERENT = (1 << 13)
AMDGPU_GEM_CREATE_UNCACHED = (1 << 14)
AMDGPU_GEM_CREATE_EXT_COHERENT = (1 << 15)
AMDGPU_GEM_CREATE_GFX12_DCC = (1 << 16)
AMDGPU_GEM_CREATE_SPARSE = (1 << 29)
AMDGPU_GEM_CREATE_TOP_DOWN = (1 << 30)
AMDGPU_GEM_CREATE_NO_EVICT = (1 << 31)
AMDGPU_BO_LIST_OP_CREATE = 0
AMDGPU_BO_LIST_OP_DESTROY = 1
AMDGPU_BO_LIST_OP_UPDATE = 2
AMDGPU_CTX_OP_ALLOC_CTX = 1
AMDGPU_CTX_OP_FREE_CTX = 2
AMDGPU_CTX_OP_QUERY_STATE = 3
AMDGPU_CTX_OP_QUERY_STATE2 = 4
AMDGPU_CTX_OP_GET_STABLE_PSTATE = 5
AMDGPU_CTX_OP_SET_STABLE_PSTATE = 6
AMDGPU_CTX_NO_RESET = 0
AMDGPU_CTX_GUILTY_RESET = 1
AMDGPU_CTX_INNOCENT_RESET = 2
AMDGPU_CTX_UNKNOWN_RESET = 3
AMDGPU_CTX_QUERY2_FLAGS_RESET = (1<<0)
AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST = (1<<1)
AMDGPU_CTX_QUERY2_FLAGS_GUILTY = (1<<2)
AMDGPU_CTX_QUERY2_FLAGS_RAS_CE = (1<<3)
AMDGPU_CTX_QUERY2_FLAGS_RAS_UE = (1<<4)
AMDGPU_CTX_QUERY2_FLAGS_RESET_IN_PROGRESS = (1<<5)
AMDGPU_CTX_PRIORITY_UNSET = -2048
AMDGPU_CTX_PRIORITY_VERY_LOW = -1023
AMDGPU_CTX_PRIORITY_LOW = -512
AMDGPU_CTX_PRIORITY_NORMAL = 0
AMDGPU_CTX_PRIORITY_HIGH = 512
AMDGPU_CTX_PRIORITY_VERY_HIGH = 1023
AMDGPU_CTX_STABLE_PSTATE_FLAGS_MASK = 0xf
AMDGPU_CTX_STABLE_PSTATE_NONE = 0
AMDGPU_CTX_STABLE_PSTATE_STANDARD = 1
AMDGPU_CTX_STABLE_PSTATE_MIN_SCLK = 2
AMDGPU_CTX_STABLE_PSTATE_MIN_MCLK = 3
AMDGPU_CTX_STABLE_PSTATE_PEAK = 4
AMDGPU_USERQ_OP_CREATE = 1
AMDGPU_USERQ_OP_FREE = 2
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_MASK = 0x3
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_SHIFT = 0
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_NORMAL_LOW = 0
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_LOW = 1
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_NORMAL_HIGH = 2
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_HIGH = 3
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_SECURE = (1 << 2)
AMDGPU_SEM_OP_CREATE_SEM = 1
AMDGPU_SEM_OP_WAIT_SEM = 2
AMDGPU_SEM_OP_SIGNAL_SEM = 3
AMDGPU_SEM_OP_DESTROY_SEM = 4
AMDGPU_SEM_OP_IMPORT_SEM = 5
AMDGPU_SEM_OP_EXPORT_SEM = 6
AMDGPU_VM_OP_RESERVE_VMID = 1
AMDGPU_VM_OP_UNRESERVE_VMID = 2
AMDGPU_SCHED_OP_PROCESS_PRIORITY_OVERRIDE = 1
AMDGPU_SCHED_OP_CONTEXT_PRIORITY_OVERRIDE = 2
AMDGPU_GEM_USERPTR_READONLY = (1 << 0)
AMDGPU_GEM_USERPTR_ANONONLY = (1 << 1)
AMDGPU_GEM_USERPTR_VALIDATE = (1 << 2)
AMDGPU_GEM_USERPTR_REGISTER = (1 << 3)
AMDGPU_GEM_DGMA_IMPORT = 0
AMDGPU_GEM_DGMA_QUERY_PHYS_ADDR = 1
AMDGPU_TILING_ARRAY_MODE_SHIFT = 0
AMDGPU_TILING_ARRAY_MODE_MASK = 0xf
AMDGPU_TILING_PIPE_CONFIG_SHIFT = 4
AMDGPU_TILING_PIPE_CONFIG_MASK = 0x1f
AMDGPU_TILING_TILE_SPLIT_SHIFT = 9
AMDGPU_TILING_TILE_SPLIT_MASK = 0x7
AMDGPU_TILING_MICRO_TILE_MODE_SHIFT = 12
AMDGPU_TILING_MICRO_TILE_MODE_MASK = 0x7
AMDGPU_TILING_BANK_WIDTH_SHIFT = 15
AMDGPU_TILING_BANK_WIDTH_MASK = 0x3
AMDGPU_TILING_BANK_HEIGHT_SHIFT = 17
AMDGPU_TILING_BANK_HEIGHT_MASK = 0x3
AMDGPU_TILING_MACRO_TILE_ASPECT_SHIFT = 19
AMDGPU_TILING_MACRO_TILE_ASPECT_MASK = 0x3
AMDGPU_TILING_NUM_BANKS_SHIFT = 21
AMDGPU_TILING_NUM_BANKS_MASK = 0x3
AMDGPU_TILING_SWIZZLE_MODE_SHIFT = 0
AMDGPU_TILING_SWIZZLE_MODE_MASK = 0x1f
AMDGPU_TILING_DCC_OFFSET_256B_SHIFT = 5
AMDGPU_TILING_DCC_OFFSET_256B_MASK = 0xFFFFFF
AMDGPU_TILING_DCC_PITCH_MAX_SHIFT = 29
AMDGPU_TILING_DCC_PITCH_MAX_MASK = 0x3FFF
AMDGPU_TILING_DCC_INDEPENDENT_64B_SHIFT = 43
AMDGPU_TILING_DCC_INDEPENDENT_64B_MASK = 0x1
AMDGPU_TILING_DCC_INDEPENDENT_128B_SHIFT = 44
AMDGPU_TILING_DCC_INDEPENDENT_128B_MASK = 0x1
AMDGPU_TILING_SCANOUT_SHIFT = 63
AMDGPU_TILING_SCANOUT_MASK = 0x1
AMDGPU_TILING_GFX12_SWIZZLE_MODE_SHIFT = 0
AMDGPU_TILING_GFX12_SWIZZLE_MODE_MASK = 0x7
AMDGPU_TILING_GFX12_DCC_MAX_COMPRESSED_BLOCK_SHIFT = 3
AMDGPU_TILING_GFX12_DCC_MAX_COMPRESSED_BLOCK_MASK = 0x3
AMDGPU_TILING_GFX12_DCC_NUMBER_TYPE_SHIFT = 5
AMDGPU_TILING_GFX12_DCC_NUMBER_TYPE_MASK = 0x7
AMDGPU_TILING_GFX12_DCC_DATA_FORMAT_SHIFT = 8
AMDGPU_TILING_GFX12_DCC_DATA_FORMAT_MASK = 0x3f
AMDGPU_TILING_GFX12_DCC_WRITE_COMPRESS_DISABLE_SHIFT = 14
AMDGPU_TILING_GFX12_DCC_WRITE_COMPRESS_DISABLE_MASK = 0x1
AMDGPU_TILING_GFX12_SCANOUT_SHIFT = 63
AMDGPU_TILING_GFX12_SCANOUT_MASK = 0x1
AMDGPU_GEM_METADATA_OP_SET_METADATA = 1
AMDGPU_GEM_METADATA_OP_GET_METADATA = 2
AMDGPU_GEM_OP_GET_GEM_CREATE_INFO = 0
AMDGPU_GEM_OP_SET_PLACEMENT = 1
AMDGPU_VA_OP_MAP = 1
AMDGPU_VA_OP_UNMAP = 2
AMDGPU_VA_OP_CLEAR = 3
AMDGPU_VA_OP_REPLACE = 4
AMDGPU_VM_DELAY_UPDATE = (1 << 0)
AMDGPU_VM_PAGE_READABLE = (1 << 1)
AMDGPU_VM_PAGE_WRITEABLE = (1 << 2)
AMDGPU_VM_PAGE_EXECUTABLE = (1 << 3)
AMDGPU_VM_PAGE_PRT = (1 << 4)
AMDGPU_VM_MTYPE_MASK = (0xf << 5)
AMDGPU_VM_MTYPE_DEFAULT = (0 << 5)
AMDGPU_VM_MTYPE_NC = (1 << 5)
AMDGPU_VM_MTYPE_WC = (2 << 5)
AMDGPU_VM_MTYPE_CC = (3 << 5)
AMDGPU_VM_MTYPE_UC = (4 << 5)
AMDGPU_VM_MTYPE_RW = (5 << 5)
AMDGPU_VM_PAGE_NOALLOC = (1 << 9)
AMDGPU_HW_IP_GFX = 0
AMDGPU_HW_IP_COMPUTE = 1
AMDGPU_HW_IP_DMA = 2
AMDGPU_HW_IP_UVD = 3
AMDGPU_HW_IP_VCE = 4
AMDGPU_HW_IP_UVD_ENC = 5
AMDGPU_HW_IP_VCN_DEC = 6
AMDGPU_HW_IP_VCN_ENC = 7
AMDGPU_HW_IP_VCN_JPEG = 8
AMDGPU_HW_IP_VPE = 9
AMDGPU_HW_IP_NUM = 10
AMDGPU_HW_IP_INSTANCE_MAX_COUNT = 1
AMDGPU_CHUNK_ID_IB = 0x01
AMDGPU_CHUNK_ID_FENCE = 0x02
AMDGPU_CHUNK_ID_DEPENDENCIES = 0x03
AMDGPU_CHUNK_ID_SYNCOBJ_IN = 0x04
AMDGPU_CHUNK_ID_SYNCOBJ_OUT = 0x05
AMDGPU_CHUNK_ID_BO_HANDLES = 0x06
AMDGPU_CHUNK_ID_SCHEDULED_DEPENDENCIES = 0x07
AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT = 0x08
AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_SIGNAL = 0x09
AMDGPU_CHUNK_ID_CP_GFX_SHADOW = 0x0a
AMDGPU_IB_FLAG_CE = (1<<0)
AMDGPU_IB_FLAG_PREAMBLE = (1<<1)
AMDGPU_IB_FLAG_PREEMPT = (1<<2)
AMDGPU_IB_FLAG_TC_WB_NOT_INVALIDATE = (1 << 3)
AMDGPU_IB_FLAG_RESET_GDS_MAX_WAVE_ID = (1 << 4)
AMDGPU_IB_FLAGS_SECURE = (1 << 5)
AMDGPU_IB_FLAG_EMIT_MEM_SYNC = (1 << 6)
AMDGPU_FENCE_TO_HANDLE_GET_SYNCOBJ = 0
AMDGPU_FENCE_TO_HANDLE_GET_SYNCOBJ_FD = 1
AMDGPU_FENCE_TO_HANDLE_GET_SYNC_FILE_FD = 2
AMDGPU_CS_CHUNK_CP_GFX_SHADOW_FLAGS_INIT_SHADOW = 0x1
AMDGPU_IDS_FLAGS_FUSION = 0x1
AMDGPU_IDS_FLAGS_PREEMPTION = 0x2
AMDGPU_IDS_FLAGS_TMZ = 0x4
AMDGPU_IDS_FLAGS_CONFORMANT_TRUNC_COORD = 0x8
AMDGPU_IDS_FLAGS_MODE_MASK = 0x300
AMDGPU_IDS_FLAGS_MODE_SHIFT = 0x8
AMDGPU_IDS_FLAGS_MODE_PF = 0x0
AMDGPU_IDS_FLAGS_MODE_VF = 0x1
AMDGPU_IDS_FLAGS_MODE_PT = 0x2
AMDGPU_INFO_ACCEL_WORKING = 0x00
AMDGPU_INFO_CRTC_FROM_ID = 0x01
AMDGPU_INFO_HW_IP_INFO = 0x02
AMDGPU_INFO_HW_IP_COUNT = 0x03
AMDGPU_INFO_TIMESTAMP = 0x05
AMDGPU_INFO_FW_VERSION = 0x0e
AMDGPU_INFO_FW_VCE = 0x1
AMDGPU_INFO_FW_UVD = 0x2
AMDGPU_INFO_FW_GMC = 0x03
AMDGPU_INFO_FW_GFX_ME = 0x04
AMDGPU_INFO_FW_GFX_PFP = 0x05
AMDGPU_INFO_FW_GFX_CE = 0x06
AMDGPU_INFO_FW_GFX_RLC = 0x07
AMDGPU_INFO_FW_GFX_MEC = 0x08
AMDGPU_INFO_FW_SMC = 0x0a
AMDGPU_INFO_FW_SDMA = 0x0b
AMDGPU_INFO_FW_SOS = 0x0c
AMDGPU_INFO_FW_ASD = 0x0d
AMDGPU_INFO_FW_VCN = 0x0e
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_CNTL = 0x0f
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_GPM_MEM = 0x10
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_SRM_MEM = 0x11
AMDGPU_INFO_FW_DMCU = 0x12
AMDGPU_INFO_FW_TA = 0x13
AMDGPU_INFO_FW_DMCUB = 0x14
AMDGPU_INFO_FW_TOC = 0x15
AMDGPU_INFO_FW_CAP = 0x16
AMDGPU_INFO_FW_GFX_RLCP = 0x17
AMDGPU_INFO_FW_GFX_RLCV = 0x18
AMDGPU_INFO_FW_MES_KIQ = 0x19
AMDGPU_INFO_FW_MES = 0x1a
AMDGPU_INFO_FW_IMU = 0x1b
AMDGPU_INFO_FW_VPE = 0x1c
AMDGPU_INFO_NUM_BYTES_MOVED = 0x0f
AMDGPU_INFO_VRAM_USAGE = 0x10
AMDGPU_INFO_GTT_USAGE = 0x11
AMDGPU_INFO_GDS_CONFIG = 0x13
AMDGPU_INFO_VRAM_GTT = 0x14
AMDGPU_INFO_READ_MMR_REG = 0x15
AMDGPU_INFO_DEV_INFO = 0x16
AMDGPU_INFO_VIS_VRAM_USAGE = 0x17
AMDGPU_INFO_NUM_EVICTIONS = 0x18
AMDGPU_INFO_MEMORY = 0x19
AMDGPU_INFO_VCE_CLOCK_TABLE = 0x1A
AMDGPU_INFO_VBIOS = 0x1B
AMDGPU_INFO_VBIOS_SIZE = 0x1
AMDGPU_INFO_VBIOS_IMAGE = 0x2
AMDGPU_INFO_VBIOS_INFO = 0x3
AMDGPU_INFO_NUM_HANDLES = 0x1C
AMDGPU_INFO_SENSOR = 0x1D
AMDGPU_INFO_SENSOR_GFX_SCLK = 0x1
AMDGPU_INFO_SENSOR_GFX_MCLK = 0x2
AMDGPU_INFO_SENSOR_GPU_TEMP = 0x3
AMDGPU_INFO_SENSOR_GPU_LOAD = 0x4
AMDGPU_INFO_SENSOR_GPU_AVG_POWER = 0x5
AMDGPU_INFO_SENSOR_VDDNB = 0x6
AMDGPU_INFO_SENSOR_VDDGFX = 0x7
AMDGPU_INFO_SENSOR_STABLE_PSTATE_GFX_SCLK = 0x8
AMDGPU_INFO_SENSOR_STABLE_PSTATE_GFX_MCLK = 0x9
AMDGPU_INFO_SENSOR_PEAK_PSTATE_GFX_SCLK = 0xa
AMDGPU_INFO_SENSOR_PEAK_PSTATE_GFX_MCLK = 0xb
AMDGPU_INFO_SENSOR_GPU_INPUT_POWER = 0xc
AMDGPU_INFO_NUM_VRAM_CPU_PAGE_FAULTS = 0x1E
AMDGPU_INFO_VRAM_LOST_COUNTER = 0x1F
AMDGPU_INFO_RAS_ENABLED_FEATURES = 0x20
AMDGPU_INFO_RAS_ENABLED_UMC = (1 << 0)
AMDGPU_INFO_RAS_ENABLED_SDMA = (1 << 1)
AMDGPU_INFO_RAS_ENABLED_GFX = (1 << 2)
AMDGPU_INFO_RAS_ENABLED_MMHUB = (1 << 3)
AMDGPU_INFO_RAS_ENABLED_ATHUB = (1 << 4)
AMDGPU_INFO_RAS_ENABLED_PCIE = (1 << 5)
AMDGPU_INFO_RAS_ENABLED_HDP = (1 << 6)
AMDGPU_INFO_RAS_ENABLED_XGMI = (1 << 7)
AMDGPU_INFO_RAS_ENABLED_DF = (1 << 8)
AMDGPU_INFO_RAS_ENABLED_SMN = (1 << 9)
AMDGPU_INFO_RAS_ENABLED_SEM = (1 << 10)
AMDGPU_INFO_RAS_ENABLED_MP0 = (1 << 11)
AMDGPU_INFO_RAS_ENABLED_MP1 = (1 << 12)
AMDGPU_INFO_RAS_ENABLED_FUSE = (1 << 13)
AMDGPU_INFO_VIDEO_CAPS = 0x21
AMDGPU_INFO_VIDEO_CAPS_DECODE = 0
AMDGPU_INFO_VIDEO_CAPS_ENCODE = 1
AMDGPU_INFO_MAX_IBS = 0x22
AMDGPU_INFO_GPUVM_FAULT = 0x23
AMDGPU_INFO_UQ_FW_AREAS = 0x24
AMDGPU_INFO_CAPABILITY = 0x50
AMDGPU_INFO_VIRTUAL_RANGE = 0x51
AMDGPU_CAPABILITY_PIN_MEM_FLAG = (1 << 0)
AMDGPU_CAPABILITY_DIRECT_GMA_FLAG = (1 << 1)
AMDGPU_INFO_MMR_SE_INDEX_SHIFT = 0
AMDGPU_INFO_MMR_SE_INDEX_MASK = 0xff
AMDGPU_INFO_MMR_SH_INDEX_SHIFT = 8
AMDGPU_INFO_MMR_SH_INDEX_MASK = 0xff
AMDGPU_VRAM_TYPE_UNKNOWN = 0
AMDGPU_VRAM_TYPE_GDDR1 = 1
AMDGPU_VRAM_TYPE_DDR2 = 2
AMDGPU_VRAM_TYPE_GDDR3 = 3
AMDGPU_VRAM_TYPE_GDDR4 = 4
AMDGPU_VRAM_TYPE_GDDR5 = 5
AMDGPU_VRAM_TYPE_HBM = 6
AMDGPU_VRAM_TYPE_DDR3 = 7
AMDGPU_VRAM_TYPE_DDR4 = 8
AMDGPU_VRAM_TYPE_GDDR6 = 9
AMDGPU_VRAM_TYPE_DDR5 = 10
AMDGPU_VRAM_TYPE_LPDDR4 = 11
AMDGPU_VRAM_TYPE_LPDDR5 = 12
AMDGPU_VRAM_TYPE_HBM3E = 13
AMDGPU_VRAM_TYPE_HBM_WIDTH = 4096
AMDGPU_VCE_CLOCK_TABLE_ENTRIES = 6
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG2 = 0
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG4 = 1
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_VC1 = 2
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG4_AVC = 3
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_HEVC = 4
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_JPEG = 5
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_VP9 = 6
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_AV1 = 7
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_COUNT = 8
AMDGPU_VMHUB_TYPE_MASK = 0xff
AMDGPU_VMHUB_TYPE_SHIFT = 0
AMDGPU_VMHUB_TYPE_GFX = 0
AMDGPU_VMHUB_TYPE_MM0 = 1
AMDGPU_VMHUB_TYPE_MM1 = 2
AMDGPU_VMHUB_IDX_MASK = 0xff00
AMDGPU_VMHUB_IDX_SHIFT = 8
AMDGPU_FAMILY_UNKNOWN = 0
AMDGPU_FAMILY_SI = 110
AMDGPU_FAMILY_CI = 120
AMDGPU_FAMILY_KV = 125
AMDGPU_FAMILY_VI = 130
AMDGPU_FAMILY_CZ = 135
AMDGPU_FAMILY_AI = 141
AMDGPU_FAMILY_RV = 142
AMDGPU_FAMILY_NV = 143
AMDGPU_FAMILY_VGH = 144
AMDGPU_FAMILY_GC_11_0_0 = 145
AMDGPU_FAMILY_YC = 146
AMDGPU_FAMILY_GC_11_0_1 = 148
AMDGPU_FAMILY_GC_10_3_6 = 149
AMDGPU_FAMILY_GC_10_3_7 = 151
AMDGPU_FAMILY_GC_11_5_0 = 150
AMDGPU_FAMILY_GC_12_0_0 = 152
AMDGPU_SUA_APERTURE_PRIVATE = 1
AMDGPU_SUA_APERTURE_SHARED = 2
AMDGPU_FREESYNC_FULLSCREEN_ENTER = 1
AMDGPU_FREESYNC_FULLSCREEN_EXIT = 2