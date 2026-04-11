# mypy: disable-error-code="empty-body"
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_io_uring_sq(c.Struct):
  SIZE = 104
  khead: 'ctypes._Pointer[int]'
  ktail: 'ctypes._Pointer[int]'
  kring_mask: 'ctypes._Pointer[int]'
  kring_entries: 'ctypes._Pointer[int]'
  kflags: 'ctypes._Pointer[int]'
  kdropped: 'ctypes._Pointer[int]'
  array: 'ctypes._Pointer[int]'
  sqes: 'ctypes._Pointer[struct_io_uring_sqe]'
  sqe_head: 'int'
  sqe_tail: 'int'
  ring_sz: 'int'
  ring_ptr: 'ctypes.c_void_p'
  ring_mask: 'int'
  ring_entries: 'int'
  sqes_sz: 'int'
  pad: 'int'
@c.record
class struct_io_uring_sqe(c.Struct):
  SIZE = 64
  opcode: 'int'
  flags: 'int'
  ioprio: 'int'
  fd: 'int'
  off: 'int'
  addr2: 'int'
  cmd_op: 'int'
  __pad1: 'int'
  addr: 'int'
  splice_off_in: 'int'
  len: 'int'
  rw_flags: 'int'
  fsync_flags: 'int'
  poll_events: 'int'
  poll32_events: 'int'
  sync_range_flags: 'int'
  msg_flags: 'int'
  timeout_flags: 'int'
  accept_flags: 'int'
  cancel_flags: 'int'
  open_flags: 'int'
  statx_flags: 'int'
  fadvise_advice: 'int'
  splice_flags: 'int'
  rename_flags: 'int'
  unlink_flags: 'int'
  hardlink_flags: 'int'
  xattr_flags: 'int'
  msg_ring_flags: 'int'
  uring_cmd_flags: 'int'
  user_data: 'int'
  buf_index: 'int'
  buf_group: 'int'
  personality: 'int'
  splice_fd_in: 'int'
  file_index: 'int'
  addr_len: 'int'
  __pad3: 'list[int]'
  addr3: 'int'
  __pad2: 'list[int]'
  cmd: 'list[int]'
__u8: TypeAlias = ctypes.c_ubyte
__u16: TypeAlias = ctypes.c_uint16
__s32: TypeAlias = ctypes.c_int32
__u64: TypeAlias = ctypes.c_uint64
__u32: TypeAlias = ctypes.c_uint32
__kernel_rwf_t: TypeAlias = ctypes.c_int32
struct_io_uring_sqe.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('ioprio', ctypes.c_uint16, 2), ('fd', ctypes.c_int32, 4), ('off', ctypes.c_uint64, 8), ('addr2', ctypes.c_uint64, 8), ('cmd_op', ctypes.c_uint32, 8), ('__pad1', ctypes.c_uint32, 12), ('addr', ctypes.c_uint64, 16), ('splice_off_in', ctypes.c_uint64, 16), ('len', ctypes.c_uint32, 24), ('rw_flags', ctypes.c_int32, 28), ('fsync_flags', ctypes.c_uint32, 28), ('poll_events', ctypes.c_uint16, 28), ('poll32_events', ctypes.c_uint32, 28), ('sync_range_flags', ctypes.c_uint32, 28), ('msg_flags', ctypes.c_uint32, 28), ('timeout_flags', ctypes.c_uint32, 28), ('accept_flags', ctypes.c_uint32, 28), ('cancel_flags', ctypes.c_uint32, 28), ('open_flags', ctypes.c_uint32, 28), ('statx_flags', ctypes.c_uint32, 28), ('fadvise_advice', ctypes.c_uint32, 28), ('splice_flags', ctypes.c_uint32, 28), ('rename_flags', ctypes.c_uint32, 28), ('unlink_flags', ctypes.c_uint32, 28), ('hardlink_flags', ctypes.c_uint32, 28), ('xattr_flags', ctypes.c_uint32, 28), ('msg_ring_flags', ctypes.c_uint32, 28), ('uring_cmd_flags', ctypes.c_uint32, 28), ('user_data', ctypes.c_uint64, 32), ('buf_index', ctypes.c_uint16, 40), ('buf_group', ctypes.c_uint16, 40), ('personality', ctypes.c_uint16, 42), ('splice_fd_in', ctypes.c_int32, 44), ('file_index', ctypes.c_uint32, 44), ('addr_len', ctypes.c_uint16, 44), ('__pad3', (ctypes.c_uint16 * 1), 46), ('addr3', ctypes.c_uint64, 48), ('__pad2', (ctypes.c_uint64 * 1), 56), ('cmd', (ctypes.c_ubyte * 0), 48)])
size_t: TypeAlias = ctypes.c_uint64
struct_io_uring_sq.register_fields([('khead', ctypes.POINTER(ctypes.c_uint32), 0), ('ktail', ctypes.POINTER(ctypes.c_uint32), 8), ('kring_mask', ctypes.POINTER(ctypes.c_uint32), 16), ('kring_entries', ctypes.POINTER(ctypes.c_uint32), 24), ('kflags', ctypes.POINTER(ctypes.c_uint32), 32), ('kdropped', ctypes.POINTER(ctypes.c_uint32), 40), ('array', ctypes.POINTER(ctypes.c_uint32), 48), ('sqes', ctypes.POINTER(struct_io_uring_sqe), 56), ('sqe_head', ctypes.c_uint32, 64), ('sqe_tail', ctypes.c_uint32, 68), ('ring_sz', size_t, 72), ('ring_ptr', ctypes.c_void_p, 80), ('ring_mask', ctypes.c_uint32, 88), ('ring_entries', ctypes.c_uint32, 92), ('sqes_sz', ctypes.c_uint32, 96), ('pad', ctypes.c_uint32, 100)])
@c.record
class struct_io_uring_cq(c.Struct):
  SIZE = 88
  khead: 'ctypes._Pointer[int]'
  ktail: 'ctypes._Pointer[int]'
  kring_mask: 'ctypes._Pointer[int]'
  kring_entries: 'ctypes._Pointer[int]'
  kflags: 'ctypes._Pointer[int]'
  koverflow: 'ctypes._Pointer[int]'
  cqes: 'ctypes._Pointer[struct_io_uring_cqe]'
  ring_sz: 'int'
  ring_ptr: 'ctypes.c_void_p'
  ring_mask: 'int'
  ring_entries: 'int'
  pad: 'list[int]'
@c.record
class struct_io_uring_cqe(c.Struct):
  SIZE = 16
  user_data: 'int'
  res: 'int'
  flags: 'int'
  big_cqe: 'list[int]'
struct_io_uring_cqe.register_fields([('user_data', ctypes.c_uint64, 0), ('res', ctypes.c_int32, 8), ('flags', ctypes.c_uint32, 12), ('big_cqe', (ctypes.c_uint64 * 0), 16)])
struct_io_uring_cq.register_fields([('khead', ctypes.POINTER(ctypes.c_uint32), 0), ('ktail', ctypes.POINTER(ctypes.c_uint32), 8), ('kring_mask', ctypes.POINTER(ctypes.c_uint32), 16), ('kring_entries', ctypes.POINTER(ctypes.c_uint32), 24), ('kflags', ctypes.POINTER(ctypes.c_uint32), 32), ('koverflow', ctypes.POINTER(ctypes.c_uint32), 40), ('cqes', ctypes.POINTER(struct_io_uring_cqe), 48), ('ring_sz', size_t, 56), ('ring_ptr', ctypes.c_void_p, 64), ('ring_mask', ctypes.c_uint32, 72), ('ring_entries', ctypes.c_uint32, 76), ('pad', (ctypes.c_uint32 * 2), 80)])
@c.record
class struct_io_uring(c.Struct):
  SIZE = 216
  sq: 'struct_io_uring_sq'
  cq: 'struct_io_uring_cq'
  flags: 'int'
  ring_fd: 'int'
  features: 'int'
  enter_ring_fd: 'int'
  int_flags: 'int'
  pad: 'list[int]'
  pad2: 'int'
struct_io_uring.register_fields([('sq', struct_io_uring_sq, 0), ('cq', struct_io_uring_cq, 104), ('flags', ctypes.c_uint32, 192), ('ring_fd', ctypes.c_int32, 196), ('features', ctypes.c_uint32, 200), ('enter_ring_fd', ctypes.c_int32, 204), ('int_flags', ctypes.c_ubyte, 208), ('pad', (ctypes.c_ubyte * 3), 209), ('pad2', ctypes.c_uint32, 212)])
@c.record
class struct_io_uring_zcrx_rq(c.Struct):
  SIZE = 40
  khead: 'ctypes._Pointer[int]'
  ktail: 'ctypes._Pointer[int]'
  rq_tail: 'int'
  ring_entries: 'int'
  rqes: 'ctypes._Pointer[struct_io_uring_zcrx_rqe]'
  ring_ptr: 'ctypes.c_void_p'
@c.record
class struct_io_uring_zcrx_rqe(c.Struct):
  SIZE = 16
  off: 'int'
  len: 'int'
  __pad: 'int'
struct_io_uring_zcrx_rq.register_fields([('khead', ctypes.POINTER(ctypes.c_uint32), 0), ('ktail', ctypes.POINTER(ctypes.c_uint32), 8), ('rq_tail', ctypes.c_uint32, 16), ('ring_entries', ctypes.c_uint32, 20), ('rqes', ctypes.POINTER(struct_io_uring_zcrx_rqe), 24), ('ring_ptr', ctypes.c_void_p, 32)])
@c.record
class struct_io_uring_cqe_iter(c.Struct):
  SIZE = 24
  cqes: 'ctypes._Pointer[struct_io_uring_cqe]'
  mask: 'int'
  shift: 'int'
  head: 'int'
  tail: 'int'
struct_io_uring_cqe_iter.register_fields([('cqes', ctypes.POINTER(struct_io_uring_cqe), 0), ('mask', ctypes.c_uint32, 8), ('shift', ctypes.c_uint32, 12), ('head', ctypes.c_uint32, 16), ('tail', ctypes.c_uint32, 20)])
class struct_epoll_event(c.Struct): pass
class struct_statx(c.Struct): pass
class struct_futex_waitv(c.Struct): pass
@c.record
class struct_io_uring_attr_pi(c.Struct):
  SIZE = 32
  flags: 'int'
  app_tag: 'int'
  len: 'int'
  addr: 'int'
  seed: 'int'
  rsvd: 'int'
struct_io_uring_attr_pi.register_fields([('flags', ctypes.c_uint16, 0), ('app_tag', ctypes.c_uint16, 2), ('len', ctypes.c_uint32, 4), ('addr', ctypes.c_uint64, 8), ('seed', ctypes.c_uint64, 16), ('rsvd', ctypes.c_uint64, 24)])
enum_io_uring_sqe_flags_bit: dict[int, str] = {(IOSQE_FIXED_FILE_BIT:=0): 'IOSQE_FIXED_FILE_BIT', (IOSQE_IO_DRAIN_BIT:=1): 'IOSQE_IO_DRAIN_BIT', (IOSQE_IO_LINK_BIT:=2): 'IOSQE_IO_LINK_BIT', (IOSQE_IO_HARDLINK_BIT:=3): 'IOSQE_IO_HARDLINK_BIT', (IOSQE_ASYNC_BIT:=4): 'IOSQE_ASYNC_BIT', (IOSQE_BUFFER_SELECT_BIT:=5): 'IOSQE_BUFFER_SELECT_BIT', (IOSQE_CQE_SKIP_SUCCESS_BIT:=6): 'IOSQE_CQE_SKIP_SUCCESS_BIT'}
enum_io_uring_op: dict[int, str] = {(IORING_OP_NOP:=0): 'IORING_OP_NOP', (IORING_OP_READV:=1): 'IORING_OP_READV', (IORING_OP_WRITEV:=2): 'IORING_OP_WRITEV', (IORING_OP_FSYNC:=3): 'IORING_OP_FSYNC', (IORING_OP_READ_FIXED:=4): 'IORING_OP_READ_FIXED', (IORING_OP_WRITE_FIXED:=5): 'IORING_OP_WRITE_FIXED', (IORING_OP_POLL_ADD:=6): 'IORING_OP_POLL_ADD', (IORING_OP_POLL_REMOVE:=7): 'IORING_OP_POLL_REMOVE', (IORING_OP_SYNC_FILE_RANGE:=8): 'IORING_OP_SYNC_FILE_RANGE', (IORING_OP_SENDMSG:=9): 'IORING_OP_SENDMSG', (IORING_OP_RECVMSG:=10): 'IORING_OP_RECVMSG', (IORING_OP_TIMEOUT:=11): 'IORING_OP_TIMEOUT', (IORING_OP_TIMEOUT_REMOVE:=12): 'IORING_OP_TIMEOUT_REMOVE', (IORING_OP_ACCEPT:=13): 'IORING_OP_ACCEPT', (IORING_OP_ASYNC_CANCEL:=14): 'IORING_OP_ASYNC_CANCEL', (IORING_OP_LINK_TIMEOUT:=15): 'IORING_OP_LINK_TIMEOUT', (IORING_OP_CONNECT:=16): 'IORING_OP_CONNECT', (IORING_OP_FALLOCATE:=17): 'IORING_OP_FALLOCATE', (IORING_OP_OPENAT:=18): 'IORING_OP_OPENAT', (IORING_OP_CLOSE:=19): 'IORING_OP_CLOSE', (IORING_OP_FILES_UPDATE:=20): 'IORING_OP_FILES_UPDATE', (IORING_OP_STATX:=21): 'IORING_OP_STATX', (IORING_OP_READ:=22): 'IORING_OP_READ', (IORING_OP_WRITE:=23): 'IORING_OP_WRITE', (IORING_OP_FADVISE:=24): 'IORING_OP_FADVISE', (IORING_OP_MADVISE:=25): 'IORING_OP_MADVISE', (IORING_OP_SEND:=26): 'IORING_OP_SEND', (IORING_OP_RECV:=27): 'IORING_OP_RECV', (IORING_OP_OPENAT2:=28): 'IORING_OP_OPENAT2', (IORING_OP_EPOLL_CTL:=29): 'IORING_OP_EPOLL_CTL', (IORING_OP_SPLICE:=30): 'IORING_OP_SPLICE', (IORING_OP_PROVIDE_BUFFERS:=31): 'IORING_OP_PROVIDE_BUFFERS', (IORING_OP_REMOVE_BUFFERS:=32): 'IORING_OP_REMOVE_BUFFERS', (IORING_OP_TEE:=33): 'IORING_OP_TEE', (IORING_OP_SHUTDOWN:=34): 'IORING_OP_SHUTDOWN', (IORING_OP_RENAMEAT:=35): 'IORING_OP_RENAMEAT', (IORING_OP_UNLINKAT:=36): 'IORING_OP_UNLINKAT', (IORING_OP_MKDIRAT:=37): 'IORING_OP_MKDIRAT', (IORING_OP_SYMLINKAT:=38): 'IORING_OP_SYMLINKAT', (IORING_OP_LINKAT:=39): 'IORING_OP_LINKAT', (IORING_OP_MSG_RING:=40): 'IORING_OP_MSG_RING', (IORING_OP_FSETXATTR:=41): 'IORING_OP_FSETXATTR', (IORING_OP_SETXATTR:=42): 'IORING_OP_SETXATTR', (IORING_OP_FGETXATTR:=43): 'IORING_OP_FGETXATTR', (IORING_OP_GETXATTR:=44): 'IORING_OP_GETXATTR', (IORING_OP_SOCKET:=45): 'IORING_OP_SOCKET', (IORING_OP_URING_CMD:=46): 'IORING_OP_URING_CMD', (IORING_OP_SEND_ZC:=47): 'IORING_OP_SEND_ZC', (IORING_OP_SENDMSG_ZC:=48): 'IORING_OP_SENDMSG_ZC', (IORING_OP_READ_MULTISHOT:=49): 'IORING_OP_READ_MULTISHOT', (IORING_OP_WAITID:=50): 'IORING_OP_WAITID', (IORING_OP_FUTEX_WAIT:=51): 'IORING_OP_FUTEX_WAIT', (IORING_OP_FUTEX_WAKE:=52): 'IORING_OP_FUTEX_WAKE', (IORING_OP_FUTEX_WAITV:=53): 'IORING_OP_FUTEX_WAITV', (IORING_OP_FIXED_FD_INSTALL:=54): 'IORING_OP_FIXED_FD_INSTALL', (IORING_OP_FTRUNCATE:=55): 'IORING_OP_FTRUNCATE', (IORING_OP_BIND:=56): 'IORING_OP_BIND', (IORING_OP_LISTEN:=57): 'IORING_OP_LISTEN', (IORING_OP_RECV_ZC:=58): 'IORING_OP_RECV_ZC', (IORING_OP_EPOLL_WAIT:=59): 'IORING_OP_EPOLL_WAIT', (IORING_OP_READV_FIXED:=60): 'IORING_OP_READV_FIXED', (IORING_OP_WRITEV_FIXED:=61): 'IORING_OP_WRITEV_FIXED', (IORING_OP_PIPE:=62): 'IORING_OP_PIPE', (IORING_OP_LAST:=63): 'IORING_OP_LAST'}
enum_io_uring_msg_ring_flags: dict[int, str] = {(IORING_MSG_DATA:=0): 'IORING_MSG_DATA', (IORING_MSG_SEND_FD:=1): 'IORING_MSG_SEND_FD'}
@c.record
class struct_io_sqring_offsets(c.Struct):
  SIZE = 40
  head: 'int'
  tail: 'int'
  ring_mask: 'int'
  ring_entries: 'int'
  flags: 'int'
  dropped: 'int'
  array: 'int'
  resv1: 'int'
  user_addr: 'int'
struct_io_sqring_offsets.register_fields([('head', ctypes.c_uint32, 0), ('tail', ctypes.c_uint32, 4), ('ring_mask', ctypes.c_uint32, 8), ('ring_entries', ctypes.c_uint32, 12), ('flags', ctypes.c_uint32, 16), ('dropped', ctypes.c_uint32, 20), ('array', ctypes.c_uint32, 24), ('resv1', ctypes.c_uint32, 28), ('user_addr', ctypes.c_uint64, 32)])
@c.record
class struct_io_cqring_offsets(c.Struct):
  SIZE = 40
  head: 'int'
  tail: 'int'
  ring_mask: 'int'
  ring_entries: 'int'
  overflow: 'int'
  cqes: 'int'
  flags: 'int'
  resv1: 'int'
  user_addr: 'int'
struct_io_cqring_offsets.register_fields([('head', ctypes.c_uint32, 0), ('tail', ctypes.c_uint32, 4), ('ring_mask', ctypes.c_uint32, 8), ('ring_entries', ctypes.c_uint32, 12), ('overflow', ctypes.c_uint32, 16), ('cqes', ctypes.c_uint32, 20), ('flags', ctypes.c_uint32, 24), ('resv1', ctypes.c_uint32, 28), ('user_addr', ctypes.c_uint64, 32)])
@c.record
class struct_io_uring_params(c.Struct):
  SIZE = 120
  sq_entries: 'int'
  cq_entries: 'int'
  flags: 'int'
  sq_thread_cpu: 'int'
  sq_thread_idle: 'int'
  features: 'int'
  wq_fd: 'int'
  resv: 'list[int]'
  sq_off: 'struct_io_sqring_offsets'
  cq_off: 'struct_io_cqring_offsets'
struct_io_uring_params.register_fields([('sq_entries', ctypes.c_uint32, 0), ('cq_entries', ctypes.c_uint32, 4), ('flags', ctypes.c_uint32, 8), ('sq_thread_cpu', ctypes.c_uint32, 12), ('sq_thread_idle', ctypes.c_uint32, 16), ('features', ctypes.c_uint32, 20), ('wq_fd', ctypes.c_uint32, 24), ('resv', (ctypes.c_uint32 * 3), 28), ('sq_off', struct_io_sqring_offsets, 40), ('cq_off', struct_io_cqring_offsets, 80)])
enum_io_uring_register_op: dict[int, str] = {(IORING_REGISTER_BUFFERS:=0): 'IORING_REGISTER_BUFFERS', (IORING_UNREGISTER_BUFFERS:=1): 'IORING_UNREGISTER_BUFFERS', (IORING_REGISTER_FILES:=2): 'IORING_REGISTER_FILES', (IORING_UNREGISTER_FILES:=3): 'IORING_UNREGISTER_FILES', (IORING_REGISTER_EVENTFD:=4): 'IORING_REGISTER_EVENTFD', (IORING_UNREGISTER_EVENTFD:=5): 'IORING_UNREGISTER_EVENTFD', (IORING_REGISTER_FILES_UPDATE:=6): 'IORING_REGISTER_FILES_UPDATE', (IORING_REGISTER_EVENTFD_ASYNC:=7): 'IORING_REGISTER_EVENTFD_ASYNC', (IORING_REGISTER_PROBE:=8): 'IORING_REGISTER_PROBE', (IORING_REGISTER_PERSONALITY:=9): 'IORING_REGISTER_PERSONALITY', (IORING_UNREGISTER_PERSONALITY:=10): 'IORING_UNREGISTER_PERSONALITY', (IORING_REGISTER_RESTRICTIONS:=11): 'IORING_REGISTER_RESTRICTIONS', (IORING_REGISTER_ENABLE_RINGS:=12): 'IORING_REGISTER_ENABLE_RINGS', (IORING_REGISTER_FILES2:=13): 'IORING_REGISTER_FILES2', (IORING_REGISTER_FILES_UPDATE2:=14): 'IORING_REGISTER_FILES_UPDATE2', (IORING_REGISTER_BUFFERS2:=15): 'IORING_REGISTER_BUFFERS2', (IORING_REGISTER_BUFFERS_UPDATE:=16): 'IORING_REGISTER_BUFFERS_UPDATE', (IORING_REGISTER_IOWQ_AFF:=17): 'IORING_REGISTER_IOWQ_AFF', (IORING_UNREGISTER_IOWQ_AFF:=18): 'IORING_UNREGISTER_IOWQ_AFF', (IORING_REGISTER_IOWQ_MAX_WORKERS:=19): 'IORING_REGISTER_IOWQ_MAX_WORKERS', (IORING_REGISTER_RING_FDS:=20): 'IORING_REGISTER_RING_FDS', (IORING_UNREGISTER_RING_FDS:=21): 'IORING_UNREGISTER_RING_FDS', (IORING_REGISTER_PBUF_RING:=22): 'IORING_REGISTER_PBUF_RING', (IORING_UNREGISTER_PBUF_RING:=23): 'IORING_UNREGISTER_PBUF_RING', (IORING_REGISTER_SYNC_CANCEL:=24): 'IORING_REGISTER_SYNC_CANCEL', (IORING_REGISTER_FILE_ALLOC_RANGE:=25): 'IORING_REGISTER_FILE_ALLOC_RANGE', (IORING_REGISTER_PBUF_STATUS:=26): 'IORING_REGISTER_PBUF_STATUS', (IORING_REGISTER_NAPI:=27): 'IORING_REGISTER_NAPI', (IORING_UNREGISTER_NAPI:=28): 'IORING_UNREGISTER_NAPI', (IORING_REGISTER_CLOCK:=29): 'IORING_REGISTER_CLOCK', (IORING_REGISTER_CLONE_BUFFERS:=30): 'IORING_REGISTER_CLONE_BUFFERS', (IORING_REGISTER_SEND_MSG_RING:=31): 'IORING_REGISTER_SEND_MSG_RING', (IORING_REGISTER_ZCRX_IFQ:=32): 'IORING_REGISTER_ZCRX_IFQ', (IORING_REGISTER_RESIZE_RINGS:=33): 'IORING_REGISTER_RESIZE_RINGS', (IORING_REGISTER_MEM_REGION:=34): 'IORING_REGISTER_MEM_REGION', (IORING_REGISTER_QUERY:=35): 'IORING_REGISTER_QUERY', (IORING_REGISTER_LAST:=36): 'IORING_REGISTER_LAST', (IORING_REGISTER_USE_REGISTERED_RING:=2147483648): 'IORING_REGISTER_USE_REGISTERED_RING'}
enum_io_wq_type: dict[int, str] = {(IO_WQ_BOUND:=0): 'IO_WQ_BOUND', (IO_WQ_UNBOUND:=1): 'IO_WQ_UNBOUND'}
@c.record
class struct_io_uring_files_update(c.Struct):
  SIZE = 16
  offset: 'int'
  resv: 'int'
  fds: 'int'
struct_io_uring_files_update.register_fields([('offset', ctypes.c_uint32, 0), ('resv', ctypes.c_uint32, 4), ('fds', ctypes.c_uint64, 8)])
_anonenum0: dict[int, str] = {(IORING_MEM_REGION_TYPE_USER:=1): 'IORING_MEM_REGION_TYPE_USER'}
@c.record
class struct_io_uring_region_desc(c.Struct):
  SIZE = 64
  user_addr: 'int'
  size: 'int'
  flags: 'int'
  id: 'int'
  mmap_offset: 'int'
  __resv: 'list[int]'
struct_io_uring_region_desc.register_fields([('user_addr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('id', ctypes.c_uint32, 20), ('mmap_offset', ctypes.c_uint64, 24), ('__resv', (ctypes.c_uint64 * 4), 32)])
_anonenum1: dict[int, str] = {(IORING_MEM_REGION_REG_WAIT_ARG:=1): 'IORING_MEM_REGION_REG_WAIT_ARG'}
@c.record
class struct_io_uring_mem_region_reg(c.Struct):
  SIZE = 32
  region_uptr: 'int'
  flags: 'int'
  __resv: 'list[int]'
struct_io_uring_mem_region_reg.register_fields([('region_uptr', ctypes.c_uint64, 0), ('flags', ctypes.c_uint64, 8), ('__resv', (ctypes.c_uint64 * 2), 16)])
@c.record
class struct_io_uring_rsrc_register(c.Struct):
  SIZE = 32
  nr: 'int'
  flags: 'int'
  resv2: 'int'
  data: 'int'
  tags: 'int'
struct_io_uring_rsrc_register.register_fields([('nr', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('resv2', ctypes.c_uint64, 8), ('data', ctypes.c_uint64, 16), ('tags', ctypes.c_uint64, 24)])
@c.record
class struct_io_uring_rsrc_update(c.Struct):
  SIZE = 16
  offset: 'int'
  resv: 'int'
  data: 'int'
struct_io_uring_rsrc_update.register_fields([('offset', ctypes.c_uint32, 0), ('resv', ctypes.c_uint32, 4), ('data', ctypes.c_uint64, 8)])
@c.record
class struct_io_uring_rsrc_update2(c.Struct):
  SIZE = 32
  offset: 'int'
  resv: 'int'
  data: 'int'
  tags: 'int'
  nr: 'int'
  resv2: 'int'
struct_io_uring_rsrc_update2.register_fields([('offset', ctypes.c_uint32, 0), ('resv', ctypes.c_uint32, 4), ('data', ctypes.c_uint64, 8), ('tags', ctypes.c_uint64, 16), ('nr', ctypes.c_uint32, 24), ('resv2', ctypes.c_uint32, 28)])
@c.record
class struct_io_uring_probe_op(c.Struct):
  SIZE = 8
  op: 'int'
  resv: 'int'
  flags: 'int'
  resv2: 'int'
struct_io_uring_probe_op.register_fields([('op', ctypes.c_ubyte, 0), ('resv', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('resv2', ctypes.c_uint32, 4)])
@c.record
class struct_io_uring_probe(c.Struct):
  SIZE = 16
  last_op: 'int'
  ops_len: 'int'
  resv: 'int'
  resv2: 'list[int]'
  ops: 'list[struct_io_uring_probe_op]'
struct_io_uring_probe.register_fields([('last_op', ctypes.c_ubyte, 0), ('ops_len', ctypes.c_ubyte, 1), ('resv', ctypes.c_uint16, 2), ('resv2', (ctypes.c_uint32 * 3), 4), ('ops', (struct_io_uring_probe_op * 0), 16)])
@c.record
class struct_io_uring_restriction(c.Struct):
  SIZE = 16
  opcode: 'int'
  register_op: 'int'
  sqe_op: 'int'
  sqe_flags: 'int'
  resv: 'int'
  resv2: 'list[int]'
struct_io_uring_restriction.register_fields([('opcode', ctypes.c_uint16, 0), ('register_op', ctypes.c_ubyte, 2), ('sqe_op', ctypes.c_ubyte, 2), ('sqe_flags', ctypes.c_ubyte, 2), ('resv', ctypes.c_ubyte, 3), ('resv2', (ctypes.c_uint32 * 3), 4)])
@c.record
class struct_io_uring_clock_register(c.Struct):
  SIZE = 16
  clockid: 'int'
  __resv: 'list[int]'
struct_io_uring_clock_register.register_fields([('clockid', ctypes.c_uint32, 0), ('__resv', (ctypes.c_uint32 * 3), 4)])
_anonenum2: dict[int, str] = {(IORING_REGISTER_SRC_REGISTERED:=1): 'IORING_REGISTER_SRC_REGISTERED', (IORING_REGISTER_DST_REPLACE:=2): 'IORING_REGISTER_DST_REPLACE'}
@c.record
class struct_io_uring_clone_buffers(c.Struct):
  SIZE = 32
  src_fd: 'int'
  flags: 'int'
  src_off: 'int'
  dst_off: 'int'
  nr: 'int'
  pad: 'list[int]'
struct_io_uring_clone_buffers.register_fields([('src_fd', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('src_off', ctypes.c_uint32, 8), ('dst_off', ctypes.c_uint32, 12), ('nr', ctypes.c_uint32, 16), ('pad', (ctypes.c_uint32 * 3), 20)])
@c.record
class struct_io_uring_buf(c.Struct):
  SIZE = 16
  addr: 'int'
  len: 'int'
  bid: 'int'
  resv: 'int'
struct_io_uring_buf.register_fields([('addr', ctypes.c_uint64, 0), ('len', ctypes.c_uint32, 8), ('bid', ctypes.c_uint16, 12), ('resv', ctypes.c_uint16, 14)])
@c.record
class struct_io_uring_buf_ring(c.Struct):
  SIZE = 16
  resv1: 'int'
  resv2: 'int'
  resv3: 'int'
  tail: 'int'
  __empty_bufs: 'struct_io_uring_buf_ring___empty_bufs'
  bufs: 'list[struct_io_uring_buf]'
class struct_io_uring_buf_ring___empty_bufs(c.Struct): pass
struct_io_uring_buf_ring.register_fields([('resv1', ctypes.c_uint64, 0), ('resv2', ctypes.c_uint32, 8), ('resv3', ctypes.c_uint16, 12), ('tail', ctypes.c_uint16, 14), ('__empty_bufs', struct_io_uring_buf_ring___empty_bufs, 0), ('bufs', (struct_io_uring_buf * 0), 0)])
enum_io_uring_register_pbuf_ring_flags: dict[int, str] = {(IOU_PBUF_RING_MMAP:=1): 'IOU_PBUF_RING_MMAP', (IOU_PBUF_RING_INC:=2): 'IOU_PBUF_RING_INC'}
@c.record
class struct_io_uring_buf_reg(c.Struct):
  SIZE = 40
  ring_addr: 'int'
  ring_entries: 'int'
  bgid: 'int'
  flags: 'int'
  resv: 'list[int]'
struct_io_uring_buf_reg.register_fields([('ring_addr', ctypes.c_uint64, 0), ('ring_entries', ctypes.c_uint32, 8), ('bgid', ctypes.c_uint16, 12), ('flags', ctypes.c_uint16, 14), ('resv', (ctypes.c_uint64 * 3), 16)])
@c.record
class struct_io_uring_buf_status(c.Struct):
  SIZE = 40
  buf_group: 'int'
  head: 'int'
  resv: 'list[int]'
struct_io_uring_buf_status.register_fields([('buf_group', ctypes.c_uint32, 0), ('head', ctypes.c_uint32, 4), ('resv', (ctypes.c_uint32 * 8), 8)])
enum_io_uring_napi_op: dict[int, str] = {(IO_URING_NAPI_REGISTER_OP:=0): 'IO_URING_NAPI_REGISTER_OP', (IO_URING_NAPI_STATIC_ADD_ID:=1): 'IO_URING_NAPI_STATIC_ADD_ID', (IO_URING_NAPI_STATIC_DEL_ID:=2): 'IO_URING_NAPI_STATIC_DEL_ID'}
enum_io_uring_napi_tracking_strategy: dict[int, str] = {(IO_URING_NAPI_TRACKING_DYNAMIC:=0): 'IO_URING_NAPI_TRACKING_DYNAMIC', (IO_URING_NAPI_TRACKING_STATIC:=1): 'IO_URING_NAPI_TRACKING_STATIC', (IO_URING_NAPI_TRACKING_INACTIVE:=255): 'IO_URING_NAPI_TRACKING_INACTIVE'}
@c.record
class struct_io_uring_napi(c.Struct):
  SIZE = 16
  busy_poll_to: 'int'
  prefer_busy_poll: 'int'
  opcode: 'int'
  pad: 'list[int]'
  op_param: 'int'
  resv: 'int'
struct_io_uring_napi.register_fields([('busy_poll_to', ctypes.c_uint32, 0), ('prefer_busy_poll', ctypes.c_ubyte, 4), ('opcode', ctypes.c_ubyte, 5), ('pad', (ctypes.c_ubyte * 2), 6), ('op_param', ctypes.c_uint32, 8), ('resv', ctypes.c_uint32, 12)])
enum_io_uring_register_restriction_op: dict[int, str] = {(IORING_RESTRICTION_REGISTER_OP:=0): 'IORING_RESTRICTION_REGISTER_OP', (IORING_RESTRICTION_SQE_OP:=1): 'IORING_RESTRICTION_SQE_OP', (IORING_RESTRICTION_SQE_FLAGS_ALLOWED:=2): 'IORING_RESTRICTION_SQE_FLAGS_ALLOWED', (IORING_RESTRICTION_SQE_FLAGS_REQUIRED:=3): 'IORING_RESTRICTION_SQE_FLAGS_REQUIRED', (IORING_RESTRICTION_LAST:=4): 'IORING_RESTRICTION_LAST'}
_anonenum3: dict[int, str] = {(IORING_REG_WAIT_TS:=1): 'IORING_REG_WAIT_TS'}
@c.record
class struct_io_uring_reg_wait(c.Struct):
  SIZE = 64
  ts: 'struct___kernel_timespec'
  min_wait_usec: 'int'
  flags: 'int'
  sigmask: 'int'
  sigmask_sz: 'int'
  pad: 'list[int]'
  pad2: 'list[int]'
@c.record
class struct___kernel_timespec(c.Struct):
  SIZE = 16
  tv_sec: 'int'
  tv_nsec: 'int'
__kernel_time64_t: TypeAlias = ctypes.c_int64
struct___kernel_timespec.register_fields([('tv_sec', ctypes.c_int64, 0), ('tv_nsec', ctypes.c_int64, 8)])
struct_io_uring_reg_wait.register_fields([('ts', struct___kernel_timespec, 0), ('min_wait_usec', ctypes.c_uint32, 16), ('flags', ctypes.c_uint32, 20), ('sigmask', ctypes.c_uint64, 24), ('sigmask_sz', ctypes.c_uint32, 32), ('pad', (ctypes.c_uint32 * 3), 36), ('pad2', (ctypes.c_uint64 * 2), 48)])
@c.record
class struct_io_uring_getevents_arg(c.Struct):
  SIZE = 24
  sigmask: 'int'
  sigmask_sz: 'int'
  min_wait_usec: 'int'
  ts: 'int'
struct_io_uring_getevents_arg.register_fields([('sigmask', ctypes.c_uint64, 0), ('sigmask_sz', ctypes.c_uint32, 8), ('min_wait_usec', ctypes.c_uint32, 12), ('ts', ctypes.c_uint64, 16)])
@c.record
class struct_io_uring_sync_cancel_reg(c.Struct):
  SIZE = 64
  addr: 'int'
  fd: 'int'
  flags: 'int'
  timeout: 'struct___kernel_timespec'
  opcode: 'int'
  pad: 'list[int]'
  pad2: 'list[int]'
struct_io_uring_sync_cancel_reg.register_fields([('addr', ctypes.c_uint64, 0), ('fd', ctypes.c_int32, 8), ('flags', ctypes.c_uint32, 12), ('timeout', struct___kernel_timespec, 16), ('opcode', ctypes.c_ubyte, 32), ('pad', (ctypes.c_ubyte * 7), 33), ('pad2', (ctypes.c_uint64 * 3), 40)])
@c.record
class struct_io_uring_file_index_range(c.Struct):
  SIZE = 16
  off: 'int'
  len: 'int'
  resv: 'int'
struct_io_uring_file_index_range.register_fields([('off', ctypes.c_uint32, 0), ('len', ctypes.c_uint32, 4), ('resv', ctypes.c_uint64, 8)])
@c.record
class struct_io_uring_recvmsg_out(c.Struct):
  SIZE = 16
  namelen: 'int'
  controllen: 'int'
  payloadlen: 'int'
  flags: 'int'
struct_io_uring_recvmsg_out.register_fields([('namelen', ctypes.c_uint32, 0), ('controllen', ctypes.c_uint32, 4), ('payloadlen', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12)])
enum_io_uring_socket_op: dict[int, str] = {(SOCKET_URING_OP_SIOCINQ:=0): 'SOCKET_URING_OP_SIOCINQ', (SOCKET_URING_OP_SIOCOUTQ:=1): 'SOCKET_URING_OP_SIOCOUTQ', (SOCKET_URING_OP_GETSOCKOPT:=2): 'SOCKET_URING_OP_GETSOCKOPT', (SOCKET_URING_OP_SETSOCKOPT:=3): 'SOCKET_URING_OP_SETSOCKOPT', (SOCKET_URING_OP_TX_TIMESTAMP:=4): 'SOCKET_URING_OP_TX_TIMESTAMP'}
@c.record
class struct_io_timespec(c.Struct):
  SIZE = 16
  tv_sec: 'int'
  tv_nsec: 'int'
struct_io_timespec.register_fields([('tv_sec', ctypes.c_uint64, 0), ('tv_nsec', ctypes.c_uint64, 8)])
struct_io_uring_zcrx_rqe.register_fields([('off', ctypes.c_uint64, 0), ('len', ctypes.c_uint32, 8), ('__pad', ctypes.c_uint32, 12)])
@c.record
class struct_io_uring_zcrx_cqe(c.Struct):
  SIZE = 16
  off: 'int'
  __pad: 'int'
struct_io_uring_zcrx_cqe.register_fields([('off', ctypes.c_uint64, 0), ('__pad', ctypes.c_uint64, 8)])
@c.record
class struct_io_uring_zcrx_offsets(c.Struct):
  SIZE = 32
  head: 'int'
  tail: 'int'
  rqes: 'int'
  __resv2: 'int'
  __resv: 'list[int]'
struct_io_uring_zcrx_offsets.register_fields([('head', ctypes.c_uint32, 0), ('tail', ctypes.c_uint32, 4), ('rqes', ctypes.c_uint32, 8), ('__resv2', ctypes.c_uint32, 12), ('__resv', (ctypes.c_uint64 * 2), 16)])
enum_io_uring_zcrx_area_flags: dict[int, str] = {(IORING_ZCRX_AREA_DMABUF:=1): 'IORING_ZCRX_AREA_DMABUF'}
@c.record
class struct_io_uring_zcrx_area_reg(c.Struct):
  SIZE = 48
  addr: 'int'
  len: 'int'
  rq_area_token: 'int'
  flags: 'int'
  dmabuf_fd: 'int'
  __resv2: 'list[int]'
struct_io_uring_zcrx_area_reg.register_fields([('addr', ctypes.c_uint64, 0), ('len', ctypes.c_uint64, 8), ('rq_area_token', ctypes.c_uint64, 16), ('flags', ctypes.c_uint32, 24), ('dmabuf_fd', ctypes.c_uint32, 28), ('__resv2', (ctypes.c_uint64 * 2), 32)])
@c.record
class struct_io_uring_zcrx_ifq_reg(c.Struct):
  SIZE = 96
  if_idx: 'int'
  if_rxq: 'int'
  rq_entries: 'int'
  flags: 'int'
  area_ptr: 'int'
  region_ptr: 'int'
  offsets: 'struct_io_uring_zcrx_offsets'
  zcrx_id: 'int'
  __resv2: 'int'
  __resv: 'list[int]'
struct_io_uring_zcrx_ifq_reg.register_fields([('if_idx', ctypes.c_uint32, 0), ('if_rxq', ctypes.c_uint32, 4), ('rq_entries', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12), ('area_ptr', ctypes.c_uint64, 16), ('region_ptr', ctypes.c_uint64, 24), ('offsets', struct_io_uring_zcrx_offsets, 32), ('zcrx_id', ctypes.c_uint32, 64), ('__resv2', ctypes.c_uint32, 68), ('__resv', (ctypes.c_uint64 * 3), 72)])
uring_unlikely = lambda cond: __builtin_expect( not  not (cond), 0) # type: ignore
uring_likely = lambda cond: __builtin_expect( not  not (cond), 1) # type: ignore
NR_io_uring_setup = 425 # type: ignore
NR_io_uring_enter = 426 # type: ignore
NR_io_uring_register = 427 # type: ignore
IO_URING_CHECK_VERSION = lambda major,minor: (major > IO_URING_VERSION_MAJOR or (major == IO_URING_VERSION_MAJOR and minor > IO_URING_VERSION_MINOR)) # type: ignore
IORING_RW_ATTR_FLAG_PI = (1 << 0) # type: ignore
IORING_FILE_INDEX_ALLOC = (~0) # type: ignore
IOSQE_FIXED_FILE = (1 << IOSQE_FIXED_FILE_BIT) # type: ignore
IOSQE_IO_DRAIN = (1 << IOSQE_IO_DRAIN_BIT) # type: ignore
IOSQE_IO_LINK = (1 << IOSQE_IO_LINK_BIT) # type: ignore
IOSQE_IO_HARDLINK = (1 << IOSQE_IO_HARDLINK_BIT) # type: ignore
IOSQE_ASYNC = (1 << IOSQE_ASYNC_BIT) # type: ignore
IOSQE_BUFFER_SELECT = (1 << IOSQE_BUFFER_SELECT_BIT) # type: ignore
IOSQE_CQE_SKIP_SUCCESS = (1 << IOSQE_CQE_SKIP_SUCCESS_BIT) # type: ignore
IORING_SETUP_IOPOLL = (1 << 0) # type: ignore
IORING_SETUP_SQPOLL = (1 << 1) # type: ignore
IORING_SETUP_SQ_AFF = (1 << 2) # type: ignore
IORING_SETUP_CQSIZE = (1 << 3) # type: ignore
IORING_SETUP_CLAMP = (1 << 4) # type: ignore
IORING_SETUP_ATTACH_WQ = (1 << 5) # type: ignore
IORING_SETUP_R_DISABLED = (1 << 6) # type: ignore
IORING_SETUP_SUBMIT_ALL = (1 << 7) # type: ignore
IORING_SETUP_COOP_TASKRUN = (1 << 8) # type: ignore
IORING_SETUP_TASKRUN_FLAG = (1 << 9) # type: ignore
IORING_SETUP_SQE128 = (1 << 10) # type: ignore
IORING_SETUP_CQE32 = (1 << 11) # type: ignore
IORING_SETUP_SINGLE_ISSUER = (1 << 12) # type: ignore
IORING_SETUP_DEFER_TASKRUN = (1 << 13) # type: ignore
IORING_SETUP_NO_MMAP = (1 << 14) # type: ignore
IORING_SETUP_REGISTERED_FD_ONLY = (1 << 15) # type: ignore
IORING_SETUP_NO_SQARRAY = (1 << 16) # type: ignore
IORING_SETUP_HYBRID_IOPOLL = (1 << 17) # type: ignore
IORING_SETUP_CQE_MIXED = (1 << 18) # type: ignore
IORING_URING_CMD_FIXED = (1 << 0) # type: ignore
IORING_URING_CMD_MULTISHOT = (1 << 1) # type: ignore
IORING_URING_CMD_MASK = (IORING_URING_CMD_FIXED | IORING_URING_CMD_MULTISHOT) # type: ignore
IORING_FSYNC_DATASYNC = (1 << 0) # type: ignore
IORING_TIMEOUT_ABS = (1 << 0) # type: ignore
IORING_TIMEOUT_UPDATE = (1 << 1) # type: ignore
IORING_TIMEOUT_BOOTTIME = (1 << 2) # type: ignore
IORING_TIMEOUT_REALTIME = (1 << 3) # type: ignore
IORING_LINK_TIMEOUT_UPDATE = (1 << 4) # type: ignore
IORING_TIMEOUT_ETIME_SUCCESS = (1 << 5) # type: ignore
IORING_TIMEOUT_MULTISHOT = (1 << 6) # type: ignore
IORING_TIMEOUT_CLOCK_MASK = (IORING_TIMEOUT_BOOTTIME | IORING_TIMEOUT_REALTIME) # type: ignore
IORING_TIMEOUT_UPDATE_MASK = (IORING_TIMEOUT_UPDATE | IORING_LINK_TIMEOUT_UPDATE) # type: ignore
SPLICE_F_FD_IN_FIXED = (1 << 31) # type: ignore
IORING_POLL_ADD_MULTI = (1 << 0) # type: ignore
IORING_POLL_UPDATE_EVENTS = (1 << 1) # type: ignore
IORING_POLL_UPDATE_USER_DATA = (1 << 2) # type: ignore
IORING_POLL_ADD_LEVEL = (1 << 3) # type: ignore
IORING_ASYNC_CANCEL_ALL = (1 << 0) # type: ignore
IORING_ASYNC_CANCEL_FD = (1 << 1) # type: ignore
IORING_ASYNC_CANCEL_ANY = (1 << 2) # type: ignore
IORING_ASYNC_CANCEL_FD_FIXED = (1 << 3) # type: ignore
IORING_ASYNC_CANCEL_USERDATA = (1 << 4) # type: ignore
IORING_ASYNC_CANCEL_OP = (1 << 5) # type: ignore
IORING_RECVSEND_POLL_FIRST = (1 << 0) # type: ignore
IORING_RECV_MULTISHOT = (1 << 1) # type: ignore
IORING_RECVSEND_FIXED_BUF = (1 << 2) # type: ignore
IORING_SEND_ZC_REPORT_USAGE = (1 << 3) # type: ignore
IORING_RECVSEND_BUNDLE = (1 << 4) # type: ignore
IORING_SEND_VECTORIZED = (1 << 5) # type: ignore
IORING_NOTIF_USAGE_ZC_COPIED = (1 << 31) # type: ignore
IORING_ACCEPT_MULTISHOT = (1 << 0) # type: ignore
IORING_ACCEPT_DONTWAIT = (1 << 1) # type: ignore
IORING_ACCEPT_POLL_FIRST = (1 << 2) # type: ignore
IORING_MSG_RING_CQE_SKIP = (1 << 0) # type: ignore
IORING_MSG_RING_FLAGS_PASS = (1 << 1) # type: ignore
IORING_FIXED_FD_NO_CLOEXEC = (1 << 0) # type: ignore
IORING_NOP_INJECT_RESULT = (1 << 0) # type: ignore
IORING_NOP_FILE = (1 << 1) # type: ignore
IORING_NOP_FIXED_FILE = (1 << 2) # type: ignore
IORING_NOP_FIXED_BUFFER = (1 << 3) # type: ignore
IORING_NOP_TW = (1 << 4) # type: ignore
IORING_NOP_CQE32 = (1 << 5) # type: ignore
IORING_CQE_F_BUFFER = (1 << 0) # type: ignore
IORING_CQE_F_MORE = (1 << 1) # type: ignore
IORING_CQE_F_SOCK_NONEMPTY = (1 << 2) # type: ignore
IORING_CQE_F_NOTIF = (1 << 3) # type: ignore
IORING_CQE_F_BUF_MORE = (1 << 4) # type: ignore
IORING_CQE_F_SKIP = (1 << 5) # type: ignore
IORING_CQE_F_32 = (1 << 15) # type: ignore
IORING_CQE_BUFFER_SHIFT = 16 # type: ignore
IORING_OFF_SQ_RING = 0 # type: ignore
IORING_OFF_CQ_RING = 0x8000000 # type: ignore
IORING_OFF_SQES = 0x10000000 # type: ignore
IORING_OFF_PBUF_RING = 0x80000000 # type: ignore
IORING_OFF_PBUF_SHIFT = 16 # type: ignore
IORING_OFF_MMAP_MASK = 0xf8000000 # type: ignore
IORING_SQ_NEED_WAKEUP = (1 << 0) # type: ignore
IORING_SQ_CQ_OVERFLOW = (1 << 1) # type: ignore
IORING_SQ_TASKRUN = (1 << 2) # type: ignore
IORING_CQ_EVENTFD_DISABLED = (1 << 0) # type: ignore
IORING_ENTER_GETEVENTS = (1 << 0) # type: ignore
IORING_ENTER_SQ_WAKEUP = (1 << 1) # type: ignore
IORING_ENTER_SQ_WAIT = (1 << 2) # type: ignore
IORING_ENTER_EXT_ARG = (1 << 3) # type: ignore
IORING_ENTER_REGISTERED_RING = (1 << 4) # type: ignore
IORING_ENTER_ABS_TIMER = (1 << 5) # type: ignore
IORING_ENTER_EXT_ARG_REG = (1 << 6) # type: ignore
IORING_ENTER_NO_IOWAIT = (1 << 7) # type: ignore
IORING_FEAT_SINGLE_MMAP = (1 << 0) # type: ignore
IORING_FEAT_NODROP = (1 << 1) # type: ignore
IORING_FEAT_SUBMIT_STABLE = (1 << 2) # type: ignore
IORING_FEAT_RW_CUR_POS = (1 << 3) # type: ignore
IORING_FEAT_CUR_PERSONALITY = (1 << 4) # type: ignore
IORING_FEAT_FAST_POLL = (1 << 5) # type: ignore
IORING_FEAT_POLL_32BITS = (1 << 6) # type: ignore
IORING_FEAT_SQPOLL_NONFIXED = (1 << 7) # type: ignore
IORING_FEAT_EXT_ARG = (1 << 8) # type: ignore
IORING_FEAT_NATIVE_WORKERS = (1 << 9) # type: ignore
IORING_FEAT_RSRC_TAGS = (1 << 10) # type: ignore
IORING_FEAT_CQE_SKIP = (1 << 11) # type: ignore
IORING_FEAT_LINKED_FILE = (1 << 12) # type: ignore
IORING_FEAT_REG_REG_RING = (1 << 13) # type: ignore
IORING_FEAT_RECVSEND_BUNDLE = (1 << 14) # type: ignore
IORING_FEAT_MIN_TIMEOUT = (1 << 15) # type: ignore
IORING_FEAT_RW_ATTR = (1 << 16) # type: ignore
IORING_FEAT_NO_IOWAIT = (1 << 17) # type: ignore
IORING_RSRC_REGISTER_SPARSE = (1 << 0) # type: ignore
IORING_REGISTER_FILES_SKIP = (-2) # type: ignore
IO_URING_OP_SUPPORTED = (1 << 0) # type: ignore
IORING_TIMESTAMP_HW_SHIFT = 16 # type: ignore
IORING_TIMESTAMP_TYPE_SHIFT = (IORING_TIMESTAMP_HW_SHIFT + 1) # type: ignore
IORING_ZCRX_AREA_SHIFT = 48 # type: ignore
__SC_3264 = lambda _nr,_32,_64: __SYSCALL(_nr, _64) # type: ignore
__SC_COMP = lambda _nr,_sys,_comp: __SYSCALL(_nr, _sys) # type: ignore
__SC_COMP_3264 = lambda _nr,_32,_64,_comp: __SC_3264(_nr, _32, _64) # type: ignore
NR_io_setup = 0 # type: ignore
NR_io_destroy = 1 # type: ignore
NR_io_submit = 2 # type: ignore
NR_io_cancel = 3 # type: ignore
NR_io_getevents = 4 # type: ignore
NR_setxattr = 5 # type: ignore
NR_lsetxattr = 6 # type: ignore
NR_fsetxattr = 7 # type: ignore
NR_getxattr = 8 # type: ignore
NR_lgetxattr = 9 # type: ignore
NR_fgetxattr = 10 # type: ignore
NR_listxattr = 11 # type: ignore
NR_llistxattr = 12 # type: ignore
NR_flistxattr = 13 # type: ignore
NR_removexattr = 14 # type: ignore
NR_lremovexattr = 15 # type: ignore
NR_fremovexattr = 16 # type: ignore
NR_getcwd = 17 # type: ignore
NR_lookup_dcookie = 18 # type: ignore
NR_eventfd2 = 19 # type: ignore
NR_epoll_create1 = 20 # type: ignore
NR_epoll_ctl = 21 # type: ignore
NR_epoll_pwait = 22 # type: ignore
NR_dup = 23 # type: ignore
NR_dup3 = 24 # type: ignore
NR3264_fcntl = 25 # type: ignore
NR_inotify_init1 = 26 # type: ignore
NR_inotify_add_watch = 27 # type: ignore
NR_inotify_rm_watch = 28 # type: ignore
NR_ioctl = 29 # type: ignore
NR_ioprio_set = 30 # type: ignore
NR_ioprio_get = 31 # type: ignore
NR_flock = 32 # type: ignore
NR_mknodat = 33 # type: ignore
NR_mkdirat = 34 # type: ignore
NR_unlinkat = 35 # type: ignore
NR_symlinkat = 36 # type: ignore
NR_linkat = 37 # type: ignore
NR_umount2 = 39 # type: ignore
NR_mount = 40 # type: ignore
NR_pivot_root = 41 # type: ignore
NR_nfsservctl = 42 # type: ignore
NR3264_statfs = 43 # type: ignore
NR3264_fstatfs = 44 # type: ignore
NR3264_truncate = 45 # type: ignore
NR3264_ftruncate = 46 # type: ignore
NR_fallocate = 47 # type: ignore
NR_faccessat = 48 # type: ignore
NR_chdir = 49 # type: ignore
NR_fchdir = 50 # type: ignore
NR_chroot = 51 # type: ignore
NR_fchmod = 52 # type: ignore
NR_fchmodat = 53 # type: ignore
NR_fchownat = 54 # type: ignore
NR_fchown = 55 # type: ignore
NR_openat = 56 # type: ignore
NR_close = 57 # type: ignore
NR_vhangup = 58 # type: ignore
NR_pipe2 = 59 # type: ignore
NR_quotactl = 60 # type: ignore
NR_getdents64 = 61 # type: ignore
NR3264_lseek = 62 # type: ignore
NR_read = 63 # type: ignore
NR_write = 64 # type: ignore
NR_readv = 65 # type: ignore
NR_writev = 66 # type: ignore
NR_pread64 = 67 # type: ignore
NR_pwrite64 = 68 # type: ignore
NR_preadv = 69 # type: ignore
NR_pwritev = 70 # type: ignore
NR3264_sendfile = 71 # type: ignore
NR_pselect6 = 72 # type: ignore
NR_ppoll = 73 # type: ignore
NR_signalfd4 = 74 # type: ignore
NR_vmsplice = 75 # type: ignore
NR_splice = 76 # type: ignore
NR_tee = 77 # type: ignore
NR_readlinkat = 78 # type: ignore
NR_sync = 81 # type: ignore
NR_fsync = 82 # type: ignore
NR_fdatasync = 83 # type: ignore
NR_sync_file_range = 84 # type: ignore
NR_timerfd_create = 85 # type: ignore
NR_timerfd_settime = 86 # type: ignore
NR_timerfd_gettime = 87 # type: ignore
NR_utimensat = 88 # type: ignore
NR_acct = 89 # type: ignore
NR_capget = 90 # type: ignore
NR_capset = 91 # type: ignore
NR_personality = 92 # type: ignore
NR_exit = 93 # type: ignore
NR_exit_group = 94 # type: ignore
NR_waitid = 95 # type: ignore
NR_set_tid_address = 96 # type: ignore
NR_unshare = 97 # type: ignore
NR_futex = 98 # type: ignore
NR_set_robust_list = 99 # type: ignore
NR_get_robust_list = 100 # type: ignore
NR_nanosleep = 101 # type: ignore
NR_getitimer = 102 # type: ignore
NR_setitimer = 103 # type: ignore
NR_kexec_load = 104 # type: ignore
NR_init_module = 105 # type: ignore
NR_delete_module = 106 # type: ignore
NR_timer_create = 107 # type: ignore
NR_timer_gettime = 108 # type: ignore
NR_timer_getoverrun = 109 # type: ignore
NR_timer_settime = 110 # type: ignore
NR_timer_delete = 111 # type: ignore
NR_clock_settime = 112 # type: ignore
NR_clock_gettime = 113 # type: ignore
NR_clock_getres = 114 # type: ignore
NR_clock_nanosleep = 115 # type: ignore
NR_syslog = 116 # type: ignore
NR_ptrace = 117 # type: ignore
NR_sched_setparam = 118 # type: ignore
NR_sched_setscheduler = 119 # type: ignore
NR_sched_getscheduler = 120 # type: ignore
NR_sched_getparam = 121 # type: ignore
NR_sched_setaffinity = 122 # type: ignore
NR_sched_getaffinity = 123 # type: ignore
NR_sched_yield = 124 # type: ignore
NR_sched_get_priority_max = 125 # type: ignore
NR_sched_get_priority_min = 126 # type: ignore
NR_sched_rr_get_interval = 127 # type: ignore
NR_restart_syscall = 128 # type: ignore
NR_kill = 129 # type: ignore
NR_tkill = 130 # type: ignore
NR_tgkill = 131 # type: ignore
NR_sigaltstack = 132 # type: ignore
NR_rt_sigsuspend = 133 # type: ignore
NR_rt_sigaction = 134 # type: ignore
NR_rt_sigprocmask = 135 # type: ignore
NR_rt_sigpending = 136 # type: ignore
NR_rt_sigtimedwait = 137 # type: ignore
NR_rt_sigqueueinfo = 138 # type: ignore
NR_rt_sigreturn = 139 # type: ignore
NR_setpriority = 140 # type: ignore
NR_getpriority = 141 # type: ignore
NR_reboot = 142 # type: ignore
NR_setregid = 143 # type: ignore
NR_setgid = 144 # type: ignore
NR_setreuid = 145 # type: ignore
NR_setuid = 146 # type: ignore
NR_setresuid = 147 # type: ignore
NR_getresuid = 148 # type: ignore
NR_setresgid = 149 # type: ignore
NR_getresgid = 150 # type: ignore
NR_setfsuid = 151 # type: ignore
NR_setfsgid = 152 # type: ignore
NR_times = 153 # type: ignore
NR_setpgid = 154 # type: ignore
NR_getpgid = 155 # type: ignore
NR_getsid = 156 # type: ignore
NR_setsid = 157 # type: ignore
NR_getgroups = 158 # type: ignore
NR_setgroups = 159 # type: ignore
NR_uname = 160 # type: ignore
NR_sethostname = 161 # type: ignore
NR_setdomainname = 162 # type: ignore
NR_getrusage = 165 # type: ignore
NR_umask = 166 # type: ignore
NR_prctl = 167 # type: ignore
NR_getcpu = 168 # type: ignore
NR_gettimeofday = 169 # type: ignore
NR_settimeofday = 170 # type: ignore
NR_adjtimex = 171 # type: ignore
NR_getpid = 172 # type: ignore
NR_getppid = 173 # type: ignore
NR_getuid = 174 # type: ignore
NR_geteuid = 175 # type: ignore
NR_getgid = 176 # type: ignore
NR_getegid = 177 # type: ignore
NR_gettid = 178 # type: ignore
NR_sysinfo = 179 # type: ignore
NR_mq_open = 180 # type: ignore
NR_mq_unlink = 181 # type: ignore
NR_mq_timedsend = 182 # type: ignore
NR_mq_timedreceive = 183 # type: ignore
NR_mq_notify = 184 # type: ignore
NR_mq_getsetattr = 185 # type: ignore
NR_msgget = 186 # type: ignore
NR_msgctl = 187 # type: ignore
NR_msgrcv = 188 # type: ignore
NR_msgsnd = 189 # type: ignore
NR_semget = 190 # type: ignore
NR_semctl = 191 # type: ignore
NR_semtimedop = 192 # type: ignore
NR_semop = 193 # type: ignore
NR_shmget = 194 # type: ignore
NR_shmctl = 195 # type: ignore
NR_shmat = 196 # type: ignore
NR_shmdt = 197 # type: ignore
NR_socket = 198 # type: ignore
NR_socketpair = 199 # type: ignore
NR_bind = 200 # type: ignore
NR_listen = 201 # type: ignore
NR_accept = 202 # type: ignore
NR_connect = 203 # type: ignore
NR_getsockname = 204 # type: ignore
NR_getpeername = 205 # type: ignore
NR_sendto = 206 # type: ignore
NR_recvfrom = 207 # type: ignore
NR_setsockopt = 208 # type: ignore
NR_getsockopt = 209 # type: ignore
NR_shutdown = 210 # type: ignore
NR_sendmsg = 211 # type: ignore
NR_recvmsg = 212 # type: ignore
NR_readahead = 213 # type: ignore
NR_brk = 214 # type: ignore
NR_munmap = 215 # type: ignore
NR_mremap = 216 # type: ignore
NR_add_key = 217 # type: ignore
NR_request_key = 218 # type: ignore
NR_keyctl = 219 # type: ignore
NR_clone = 220 # type: ignore
NR_execve = 221 # type: ignore
NR3264_mmap = 222 # type: ignore
NR3264_fadvise64 = 223 # type: ignore
NR_swapon = 224 # type: ignore
NR_swapoff = 225 # type: ignore
NR_mprotect = 226 # type: ignore
NR_msync = 227 # type: ignore
NR_mlock = 228 # type: ignore
NR_munlock = 229 # type: ignore
NR_mlockall = 230 # type: ignore
NR_munlockall = 231 # type: ignore
NR_mincore = 232 # type: ignore
NR_madvise = 233 # type: ignore
NR_remap_file_pages = 234 # type: ignore
NR_mbind = 235 # type: ignore
NR_get_mempolicy = 236 # type: ignore
NR_set_mempolicy = 237 # type: ignore
NR_migrate_pages = 238 # type: ignore
NR_move_pages = 239 # type: ignore
NR_rt_tgsigqueueinfo = 240 # type: ignore
NR_perf_event_open = 241 # type: ignore
NR_accept4 = 242 # type: ignore
NR_recvmmsg = 243 # type: ignore
NR_arch_specific_syscall = 244 # type: ignore
NR_wait4 = 260 # type: ignore
NR_prlimit64 = 261 # type: ignore
NR_fanotify_init = 262 # type: ignore
NR_fanotify_mark = 263 # type: ignore
NR_name_to_handle_at = 264 # type: ignore
NR_open_by_handle_at = 265 # type: ignore
NR_clock_adjtime = 266 # type: ignore
NR_syncfs = 267 # type: ignore
NR_setns = 268 # type: ignore
NR_sendmmsg = 269 # type: ignore
NR_process_vm_readv = 270 # type: ignore
NR_process_vm_writev = 271 # type: ignore
NR_kcmp = 272 # type: ignore
NR_finit_module = 273 # type: ignore
NR_sched_setattr = 274 # type: ignore
NR_sched_getattr = 275 # type: ignore
NR_renameat2 = 276 # type: ignore
NR_seccomp = 277 # type: ignore
NR_getrandom = 278 # type: ignore
NR_memfd_create = 279 # type: ignore
NR_bpf = 280 # type: ignore
NR_execveat = 281 # type: ignore
NR_userfaultfd = 282 # type: ignore
NR_membarrier = 283 # type: ignore
NR_mlock2 = 284 # type: ignore
NR_copy_file_range = 285 # type: ignore
NR_preadv2 = 286 # type: ignore
NR_pwritev2 = 287 # type: ignore
NR_pkey_mprotect = 288 # type: ignore
NR_pkey_alloc = 289 # type: ignore
NR_pkey_free = 290 # type: ignore
NR_statx = 291 # type: ignore
NR_io_pgetevents = 292 # type: ignore
NR_rseq = 293 # type: ignore
NR_kexec_file_load = 294 # type: ignore
NR_pidfd_send_signal = 424 # type: ignore
NR_io_uring_setup = 425 # type: ignore
NR_io_uring_enter = 426 # type: ignore
NR_io_uring_register = 427 # type: ignore
NR_open_tree = 428 # type: ignore
NR_move_mount = 429 # type: ignore
NR_fsopen = 430 # type: ignore
NR_fsconfig = 431 # type: ignore
NR_fsmount = 432 # type: ignore
NR_fspick = 433 # type: ignore
NR_pidfd_open = 434 # type: ignore
NR_clone3 = 435 # type: ignore
NR_close_range = 436 # type: ignore
NR_openat2 = 437 # type: ignore
NR_pidfd_getfd = 438 # type: ignore
NR_faccessat2 = 439 # type: ignore
NR_process_madvise = 440 # type: ignore
NR_epoll_pwait2 = 441 # type: ignore
NR_mount_setattr = 442 # type: ignore
NR_quotactl_fd = 443 # type: ignore
NR_landlock_create_ruleset = 444 # type: ignore
NR_landlock_add_rule = 445 # type: ignore
NR_landlock_restrict_self = 446 # type: ignore
NR_process_mrelease = 448 # type: ignore
NR_futex_waitv = 449 # type: ignore
NR_set_mempolicy_home_node = 450 # type: ignore
NR_cachestat = 451 # type: ignore
NR_fchmodat2 = 452 # type: ignore
NR_map_shadow_stack = 453 # type: ignore
NR_futex_wake = 454 # type: ignore
NR_futex_wait = 455 # type: ignore
NR_futex_requeue = 456 # type: ignore
NR_statmount = 457 # type: ignore
NR_listmount = 458 # type: ignore
NR_lsm_get_self_attr = 459 # type: ignore
NR_lsm_set_self_attr = 460 # type: ignore
NR_lsm_list_modules = 461 # type: ignore
NR_mseal = 462 # type: ignore
NR_setxattrat = 463 # type: ignore
NR_getxattrat = 464 # type: ignore
NR_listxattrat = 465 # type: ignore
NR_removexattrat = 466 # type: ignore
NR_open_tree_attr = 467 # type: ignore
NR_file_getattr = 468 # type: ignore
NR_file_setattr = 469 # type: ignore
NR_syscalls = 470 # type: ignore
NR_fcntl = NR3264_fcntl # type: ignore
NR_statfs = NR3264_statfs # type: ignore
NR_fstatfs = NR3264_fstatfs # type: ignore
NR_truncate = NR3264_truncate # type: ignore
NR_ftruncate = NR3264_ftruncate # type: ignore
NR_lseek = NR3264_lseek # type: ignore
NR_sendfile = NR3264_sendfile # type: ignore
NR_mmap = NR3264_mmap # type: ignore
NR_fadvise64 = NR3264_fadvise64 # type: ignore