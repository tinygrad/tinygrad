# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal
from tinygrad.runtime.support import c
from tinygrad.runtime.support.c import record

@record
class struct_drm_panthor_obj_array(c.Struct): SIZE = 16
struct_drm_panthor_obj_array.register_fields([('stride', ctypes.c_uint32, 0), ('count', ctypes.c_uint32, 4), ('array', ctypes.c_uint64, 8)])
@record
class struct_drm_panthor_gpu_info(c.Struct): SIZE = 112
struct_drm_panthor_gpu_info.register_fields([
  ('gpu_id', ctypes.c_uint32, 0), ('gpu_rev', ctypes.c_uint32, 4), ('csf_id', ctypes.c_uint32, 8),
  ('l2_features', ctypes.c_uint32, 12), ('tiler_features', ctypes.c_uint32, 16), ('mem_features', ctypes.c_uint32, 20),
  ('mmu_features', ctypes.c_uint32, 24), ('thread_features', ctypes.c_uint32, 28), ('max_threads', ctypes.c_uint32, 32),
  ('thread_max_workgroup_size', ctypes.c_uint32, 36), ('thread_max_barrier_size', ctypes.c_uint32, 40),
  ('coherency_features', ctypes.c_uint32, 44), ('texture_features', c.Array[ctypes.c_uint32, Literal[4]], 48),
  ('as_present', ctypes.c_uint32, 64), ('selected_coherency', ctypes.c_uint32, 68),
  ('shader_present', ctypes.c_uint64, 72), ('l2_present', ctypes.c_uint64, 80), ('tiler_present', ctypes.c_uint64, 88),
  ('core_features', ctypes.c_uint32, 96), ('pad', ctypes.c_uint32, 100), ('gpu_features', ctypes.c_uint64, 104)])
@record
class struct_drm_panthor_csif_info(c.Struct): SIZE = 24
struct_drm_panthor_csif_info.register_fields([
  ('csg_slot_count', ctypes.c_uint32, 0), ('cs_slot_count', ctypes.c_uint32, 4), ('cs_reg_count', ctypes.c_uint32, 8),
  ('scoreboard_slot_count', ctypes.c_uint32, 12), ('unpreserved_cs_reg_count', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@record
class struct_drm_panthor_dev_query(c.Struct): SIZE = 16
struct_drm_panthor_dev_query.register_fields([('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint32, 4), ('pointer', ctypes.c_uint64, 8)])
@record
class struct_drm_panthor_vm_create(c.Struct): SIZE = 16
struct_drm_panthor_vm_create.register_fields([('flags', ctypes.c_uint32, 0), ('id', ctypes.c_uint32, 4), ('user_va_range', ctypes.c_uint64, 8)])
@record
class struct_drm_panthor_vm_destroy(c.Struct): SIZE = 8
struct_drm_panthor_vm_destroy.register_fields([('id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@record
class struct_drm_panthor_sync_op(c.Struct): SIZE = 16
struct_drm_panthor_sync_op.register_fields([('flags', ctypes.c_uint32, 0), ('handle', ctypes.c_uint32, 4), ('timeline_value', ctypes.c_uint64, 8)])
@record
class struct_drm_panthor_vm_bind_op(c.Struct): SIZE = 48
struct_drm_panthor_vm_bind_op.register_fields([
  ('flags', ctypes.c_uint32, 0), ('bo_handle', ctypes.c_uint32, 4), ('bo_offset', ctypes.c_uint64, 8),
  ('va', ctypes.c_uint64, 16), ('size', ctypes.c_uint64, 24), ('syncs', struct_drm_panthor_obj_array, 32)])
@record
class struct_drm_panthor_vm_bind(c.Struct): SIZE = 24
struct_drm_panthor_vm_bind.register_fields([('vm_id', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4),
  ('ops', struct_drm_panthor_obj_array, 8)])
@record
class struct_drm_panthor_bo_create(c.Struct): SIZE = 24
struct_drm_panthor_bo_create.register_fields([('size', ctypes.c_uint64, 0), ('flags', ctypes.c_uint32, 8),
  ('exclusive_vm_id', ctypes.c_uint32, 12), ('handle', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@record
class struct_drm_panthor_bo_mmap_offset(c.Struct): SIZE = 16
struct_drm_panthor_bo_mmap_offset.register_fields([('handle', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4), ('offset', ctypes.c_uint64, 8)])
@record
class struct_drm_panthor_bo_sync_op(c.Struct): SIZE = 24
struct_drm_panthor_bo_sync_op.register_fields([('handle', ctypes.c_uint32, 0), ('type', ctypes.c_uint32, 4),
  ('offset', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16)])
@record
class struct_drm_panthor_bo_sync(c.Struct): SIZE = 16
struct_drm_panthor_bo_sync.register_fields([('ops', struct_drm_panthor_obj_array, 0)])
@record
class struct_drm_gem_close(c.Struct): SIZE = 8
struct_drm_gem_close.register_fields([('handle', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@record
class struct_drm_panthor_queue_create(c.Struct): SIZE = 8
struct_drm_panthor_queue_create.register_fields([('priority', ctypes.c_uint8, 0), ('pad', c.Array[ctypes.c_uint8, Literal[3]], 1),
  ('ringbuf_size', ctypes.c_uint32, 4)])
@record
class struct_drm_panthor_group_create(c.Struct): SIZE = 56
struct_drm_panthor_group_create.register_fields([
  ('queues', struct_drm_panthor_obj_array, 0), ('max_compute_cores', ctypes.c_uint8, 16), ('max_fragment_cores', ctypes.c_uint8, 17),
  ('max_tiler_cores', ctypes.c_uint8, 18), ('priority', ctypes.c_uint8, 19), ('pad', ctypes.c_uint32, 20),
  ('compute_core_mask', ctypes.c_uint64, 24), ('fragment_core_mask', ctypes.c_uint64, 32), ('tiler_core_mask', ctypes.c_uint64, 40),
  ('vm_id', ctypes.c_uint32, 48), ('group_handle', ctypes.c_uint32, 52)])
@record
class struct_drm_panthor_group_destroy(c.Struct): SIZE = 8
struct_drm_panthor_group_destroy.register_fields([('group_handle', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@record
class struct_drm_panthor_queue_submit(c.Struct): SIZE = 40
struct_drm_panthor_queue_submit.register_fields([
  ('queue_index', ctypes.c_uint32, 0), ('stream_size', ctypes.c_uint32, 4), ('stream_addr', ctypes.c_uint64, 8),
  ('latest_flush', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20), ('syncs', struct_drm_panthor_obj_array, 24)])
@record
class struct_drm_panthor_group_submit(c.Struct): SIZE = 24
struct_drm_panthor_group_submit.register_fields([('group_handle', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4),
  ('queue_submits', struct_drm_panthor_obj_array, 8)])
@record
class struct_drm_panthor_group_get_state(c.Struct): SIZE = 16
struct_drm_panthor_group_get_state.register_fields([('group_handle', ctypes.c_uint32, 0), ('state', ctypes.c_uint32, 4),
  ('fatal_queues', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@record
class struct_drm_syncobj_create(c.Struct): SIZE = 8
struct_drm_syncobj_create.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4)])
@record
class struct_drm_syncobj_destroy(c.Struct): SIZE = 8
struct_drm_syncobj_destroy.register_fields([('handle', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@record
class struct_drm_syncobj_array(c.Struct): SIZE = 16
struct_drm_syncobj_array.register_fields([('handles', ctypes.c_uint64, 0), ('count_handles', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@record
class struct_drm_syncobj_wait(c.Struct): SIZE = 40
struct_drm_syncobj_wait.register_fields([('handles', ctypes.c_uint64, 0), ('timeout_nsec', ctypes.c_int64, 8),
  ('count_handles', ctypes.c_uint32, 16), ('flags', ctypes.c_uint32, 20), ('first_signaled', ctypes.c_uint32, 24),
  ('pad', ctypes.c_uint32, 28), ('deadline_nsec', ctypes.c_uint64, 32)])
@record
class struct_drm_syncobj_timeline_wait(c.Struct): SIZE = 48
struct_drm_syncobj_timeline_wait.register_fields([
  ('handles', ctypes.c_uint64, 0), ('points', ctypes.c_uint64, 8), ('timeout_nsec', ctypes.c_int64, 16),
  ('count_handles', ctypes.c_uint32, 24), ('flags', ctypes.c_uint32, 28), ('first_signaled', ctypes.c_uint32, 32),
  ('pad', ctypes.c_uint32, 36), ('deadline_nsec', ctypes.c_uint64, 40)])

def iow(nr:int, size:int) -> int: return (1 << 30) | (size << 16) | (ord('d') << 8) | nr
def iowr(nr:int, size:int) -> int: return (3 << 30) | (size << 16) | (ord('d') << 8) | nr
DRM_IOCTL_PANTHOR_DEV_QUERY = iowr(0x40, 16)
DRM_IOCTL_PANTHOR_VM_CREATE = iowr(0x41, 16)
DRM_IOCTL_PANTHOR_VM_DESTROY = iowr(0x42, 8)
DRM_IOCTL_PANTHOR_VM_BIND = iowr(0x43, 24)
DRM_IOCTL_PANTHOR_BO_CREATE = iowr(0x45, 24)
DRM_IOCTL_PANTHOR_BO_MMAP_OFFSET = iowr(0x46, 16)
DRM_IOCTL_PANTHOR_BO_SYNC = iowr(0x4f, 16)
DRM_IOCTL_PANTHOR_GROUP_CREATE = iowr(0x47, 56)
DRM_IOCTL_PANTHOR_GROUP_DESTROY = iowr(0x48, 8)
DRM_IOCTL_PANTHOR_GROUP_SUBMIT = iowr(0x49, 24)
DRM_IOCTL_PANTHOR_GROUP_GET_STATE = iowr(0x4a, 16)
DRM_IOCTL_GEM_CLOSE = iow(0x09, 8)
DRM_IOCTL_SYNCOBJ_CREATE = iowr(0xbf, 8)
DRM_IOCTL_SYNCOBJ_DESTROY = iowr(0xc0, 8)
DRM_IOCTL_SYNCOBJ_WAIT = iowr(0xc3, 40)
DRM_IOCTL_SYNCOBJ_RESET = iowr(0xc4, 16)
DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT = iowr(0xca, 48)
DRM_PANTHOR_DEV_QUERY_GPU_INFO = 0
DRM_PANTHOR_VM_BIND_OP_TYPE_UNMAP = 1 << 28
DRM_PANTHOR_SYNC_OP_SIGNAL = 1 << 31
DRM_PANTHOR_SYNC_OP_HANDLE_TYPE_SYNCOBJ = 0
DRM_PANTHOR_SYNC_OP_HANDLE_TYPE_TIMELINE_SYNCOBJ = 1
DRM_PANTHOR_BO_SYNC_CPU_CACHE_FLUSH = 0
DRM_PANTHOR_BO_SYNC_CPU_CACHE_FLUSH_AND_INVALIDATE = 1
