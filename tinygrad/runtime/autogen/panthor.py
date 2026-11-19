# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes, os
from typing import Literal
from tinygrad.runtime.support import c
from tinygrad.runtime.support.c import record

dll = c.DLL('panthor', os.getenv('MESA_PATH', 'tinymesa'))
@dll.bind(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32)
def panthor_group_submit(ctx:ctypes.c_void_p, fd:int, group:int, stream_addr:int, stream_size:int, sync_handle:int) -> int: ...
@dll.bind(ctypes.c_uint64, ctypes.c_void_p)
def panthor_context_point(ctx:ctypes.c_void_p) -> int: ...
@dll.bind(None, ctypes.c_void_p, ctypes.c_uint32)
def panthor_set_sync_handle(ctx:ctypes.c_void_p, handle:int) -> None: ...
@dll.bind(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32)
def panthor_write(ctx:ctypes.c_void_p, address:int, value:int, size:int) -> int: ...

@record
class struct_drm_panthor_obj_array(c.Struct): SIZE = 16
struct_drm_panthor_obj_array.register_fields([('stride', ctypes.c_uint32, 0), ('count', ctypes.c_uint32, 4), ('array', ctypes.c_uint64, 8)])
@record
class struct_drm_panthor_queue_create(c.Struct): SIZE = 8
struct_drm_panthor_queue_create.register_fields([('priority', ctypes.c_uint8, 0), ('pad', c.Array[ctypes.c_uint8, Literal[3]], 1), ('ringbuf_size', ctypes.c_uint32, 4)])
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
class struct_drm_panthor_sync_op(c.Struct): SIZE = 16
struct_drm_panthor_sync_op.register_fields([('flags', ctypes.c_uint32, 0), ('handle', ctypes.c_uint32, 4), ('timeline_value', ctypes.c_uint64, 8)])
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
class struct_drm_syncobj_timeline_wait(c.Struct): SIZE = 48
struct_drm_syncobj_timeline_wait.register_fields([
  ('handles', ctypes.c_uint64, 0), ('points', ctypes.c_uint64, 8), ('timeout_nsec', ctypes.c_int64, 16),
  ('count_handles', ctypes.c_uint32, 24), ('flags', ctypes.c_uint32, 28), ('first_signaled', ctypes.c_uint32, 32),
  ('pad', ctypes.c_uint32, 36), ('deadline_nsec', ctypes.c_uint64, 40)])

def iowr(nr:int, size:int) -> int: return (3 << 30) | (size << 16) | (ord('d') << 8) | nr
DRM_IOCTL_PANTHOR_GROUP_CREATE = iowr(0x47, 56)
DRM_IOCTL_PANTHOR_GROUP_DESTROY = iowr(0x48, 8)
DRM_IOCTL_PANTHOR_GROUP_SUBMIT = iowr(0x49, 24)
DRM_IOCTL_PANTHOR_GROUP_GET_STATE = iowr(0x4a, 16)
DRM_IOCTL_SYNCOBJ_CREATE = iowr(0xbf, 8)
DRM_IOCTL_SYNCOBJ_DESTROY = iowr(0xc0, 8)
DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT = iowr(0xca, 48)
DRM_PANTHOR_SYNC_OP_SIGNAL = 1 << 31
DRM_PANTHOR_SYNC_OP_HANDLE_TYPE_TIMELINE_SYNCOBJ = 1
