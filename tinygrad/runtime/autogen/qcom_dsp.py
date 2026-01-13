from __future__ import annotations
import ctypes
from typing import Annotated, Literal
from tinygrad.runtime.support.c import DLL, record, Array, CEnum, _IO, _IOW, _IOR, _IOWR, init_records
ion_user_handle_t = ctypes.c_int32
enum_ion_heap_type = CEnum(ctypes.c_uint32)
ION_HEAP_TYPE_SYSTEM = enum_ion_heap_type.define('ION_HEAP_TYPE_SYSTEM', 0)
ION_HEAP_TYPE_SYSTEM_CONTIG = enum_ion_heap_type.define('ION_HEAP_TYPE_SYSTEM_CONTIG', 1)
ION_HEAP_TYPE_CARVEOUT = enum_ion_heap_type.define('ION_HEAP_TYPE_CARVEOUT', 2)
ION_HEAP_TYPE_CHUNK = enum_ion_heap_type.define('ION_HEAP_TYPE_CHUNK', 3)
ION_HEAP_TYPE_DMA = enum_ion_heap_type.define('ION_HEAP_TYPE_DMA', 4)
ION_HEAP_TYPE_CUSTOM = enum_ion_heap_type.define('ION_HEAP_TYPE_CUSTOM', 5)
ION_NUM_HEAPS = enum_ion_heap_type.define('ION_NUM_HEAPS', 16)

@record
class struct_ion_allocation_data:
  SIZE = 32
  len: Annotated[size_t, 0]
  align: Annotated[size_t, 8]
  heap_id_mask: Annotated[ctypes.c_uint32, 16]
  flags: Annotated[ctypes.c_uint32, 20]
  handle: Annotated[ion_user_handle_t, 24]
size_t = ctypes.c_uint64
@record
class struct_ion_fd_data:
  SIZE = 8
  handle: Annotated[ion_user_handle_t, 0]
  fd: Annotated[ctypes.c_int32, 4]
@record
class struct_ion_handle_data:
  SIZE = 4
  handle: Annotated[ion_user_handle_t, 0]
@record
class struct_ion_custom_data:
  SIZE = 16
  cmd: Annotated[ctypes.c_uint32, 0]
  arg: Annotated[ctypes.c_uint64, 8]
enum_msm_ion_heap_types = CEnum(ctypes.c_uint32)
ION_HEAP_TYPE_MSM_START = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_MSM_START', 6)
ION_HEAP_TYPE_SECURE_DMA = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_SECURE_DMA', 6)
ION_HEAP_TYPE_SYSTEM_SECURE = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_SYSTEM_SECURE', 7)
ION_HEAP_TYPE_HYP_CMA = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_HYP_CMA', 8)

enum_ion_heap_ids = CEnum(ctypes.c_int32)
INVALID_HEAP_ID = enum_ion_heap_ids.define('INVALID_HEAP_ID', -1)
ION_CP_MM_HEAP_ID = enum_ion_heap_ids.define('ION_CP_MM_HEAP_ID', 8)
ION_SECURE_HEAP_ID = enum_ion_heap_ids.define('ION_SECURE_HEAP_ID', 9)
ION_SECURE_DISPLAY_HEAP_ID = enum_ion_heap_ids.define('ION_SECURE_DISPLAY_HEAP_ID', 10)
ION_CP_MFC_HEAP_ID = enum_ion_heap_ids.define('ION_CP_MFC_HEAP_ID', 12)
ION_CP_WB_HEAP_ID = enum_ion_heap_ids.define('ION_CP_WB_HEAP_ID', 16)
ION_CAMERA_HEAP_ID = enum_ion_heap_ids.define('ION_CAMERA_HEAP_ID', 20)
ION_SYSTEM_CONTIG_HEAP_ID = enum_ion_heap_ids.define('ION_SYSTEM_CONTIG_HEAP_ID', 21)
ION_ADSP_HEAP_ID = enum_ion_heap_ids.define('ION_ADSP_HEAP_ID', 22)
ION_PIL1_HEAP_ID = enum_ion_heap_ids.define('ION_PIL1_HEAP_ID', 23)
ION_SF_HEAP_ID = enum_ion_heap_ids.define('ION_SF_HEAP_ID', 24)
ION_SYSTEM_HEAP_ID = enum_ion_heap_ids.define('ION_SYSTEM_HEAP_ID', 25)
ION_PIL2_HEAP_ID = enum_ion_heap_ids.define('ION_PIL2_HEAP_ID', 26)
ION_QSECOM_HEAP_ID = enum_ion_heap_ids.define('ION_QSECOM_HEAP_ID', 27)
ION_AUDIO_HEAP_ID = enum_ion_heap_ids.define('ION_AUDIO_HEAP_ID', 28)
ION_MM_FIRMWARE_HEAP_ID = enum_ion_heap_ids.define('ION_MM_FIRMWARE_HEAP_ID', 29)
ION_HEAP_ID_RESERVED = enum_ion_heap_ids.define('ION_HEAP_ID_RESERVED', 31)

enum_ion_fixed_position = CEnum(ctypes.c_uint32)
NOT_FIXED = enum_ion_fixed_position.define('NOT_FIXED', 0)
FIXED_LOW = enum_ion_fixed_position.define('FIXED_LOW', 1)
FIXED_MIDDLE = enum_ion_fixed_position.define('FIXED_MIDDLE', 2)
FIXED_HIGH = enum_ion_fixed_position.define('FIXED_HIGH', 3)

enum_cp_mem_usage = CEnum(ctypes.c_uint32)
VIDEO_BITSTREAM = enum_cp_mem_usage.define('VIDEO_BITSTREAM', 1)
VIDEO_PIXEL = enum_cp_mem_usage.define('VIDEO_PIXEL', 2)
VIDEO_NONPIXEL = enum_cp_mem_usage.define('VIDEO_NONPIXEL', 3)
DISPLAY_SECURE_CP_USAGE = enum_cp_mem_usage.define('DISPLAY_SECURE_CP_USAGE', 4)
CAMERA_SECURE_CP_USAGE = enum_cp_mem_usage.define('CAMERA_SECURE_CP_USAGE', 5)
MAX_USAGE = enum_cp_mem_usage.define('MAX_USAGE', 6)
UNKNOWN = enum_cp_mem_usage.define('UNKNOWN', 2147483647)

@record
class struct_ion_flush_data:
  SIZE = 24
  handle: Annotated[ion_user_handle_t, 0]
  fd: Annotated[ctypes.c_int32, 4]
  vaddr: Annotated[ctypes.POINTER(None), 8]
  offset: Annotated[ctypes.c_uint32, 16]
  length: Annotated[ctypes.c_uint32, 20]
@record
class struct_ion_prefetch_regions:
  SIZE = 24
  vmid: Annotated[ctypes.c_uint32, 0]
  sizes: Annotated[ctypes.POINTER(size_t), 8]
  nr_sizes: Annotated[ctypes.c_uint32, 16]
@record
class struct_ion_prefetch_data:
  SIZE = 32
  heap_id: Annotated[ctypes.c_int32, 0]
  len: Annotated[ctypes.c_uint64, 8]
  regions: Annotated[ctypes.POINTER(struct_ion_prefetch_regions), 16]
  nr_regions: Annotated[ctypes.c_uint32, 24]
@record
class struct_remote_buf64:
  SIZE = 16
  pv: Annotated[uint64_t, 0]
  len: Annotated[uint64_t, 8]
uint64_t = ctypes.c_uint64
@record
class struct_remote_dma_handle64:
  SIZE = 12
  fd: Annotated[ctypes.c_int32, 0]
  offset: Annotated[uint32_t, 4]
  len: Annotated[uint32_t, 8]
uint32_t = ctypes.c_uint32
@record
class union_remote_arg64:
  SIZE = 16
  buf: Annotated[struct_remote_buf64, 0]
  dma: Annotated[struct_remote_dma_handle64, 0]
  h: Annotated[uint32_t, 0]
@record
class struct_remote_buf:
  SIZE = 16
  pv: Annotated[ctypes.POINTER(None), 0]
  len: Annotated[size_t, 8]
@record
class struct_remote_dma_handle:
  SIZE = 8
  fd: Annotated[ctypes.c_int32, 0]
  offset: Annotated[uint32_t, 4]
@record
class union_remote_arg:
  SIZE = 16
  buf: Annotated[struct_remote_buf, 0]
  dma: Annotated[struct_remote_dma_handle, 0]
  h: Annotated[uint32_t, 0]
@record
class struct_fastrpc_ioctl_invoke:
  SIZE = 16
  handle: Annotated[uint32_t, 0]
  sc: Annotated[uint32_t, 4]
  pra: Annotated[ctypes.POINTER(union_remote_arg), 8]
@record
class struct_fastrpc_ioctl_invoke_fd:
  SIZE = 24
  inv: Annotated[struct_fastrpc_ioctl_invoke, 0]
  fds: Annotated[ctypes.POINTER(ctypes.c_int32), 16]
@record
class struct_fastrpc_ioctl_invoke_attrs:
  SIZE = 32
  inv: Annotated[struct_fastrpc_ioctl_invoke, 0]
  fds: Annotated[ctypes.POINTER(ctypes.c_int32), 16]
  attrs: Annotated[ctypes.POINTER(ctypes.c_uint32), 24]
@record
class struct_fastrpc_ioctl_invoke_crc:
  SIZE = 40
  inv: Annotated[struct_fastrpc_ioctl_invoke, 0]
  fds: Annotated[ctypes.POINTER(ctypes.c_int32), 16]
  attrs: Annotated[ctypes.POINTER(ctypes.c_uint32), 24]
  crc: Annotated[ctypes.POINTER(ctypes.c_uint32), 32]
@record
class struct_fastrpc_ioctl_init:
  SIZE = 40
  flags: Annotated[uint32_t, 0]
  file: Annotated[uintptr_t, 8]
  filelen: Annotated[uint32_t, 16]
  filefd: Annotated[int32_t, 20]
  mem: Annotated[uintptr_t, 24]
  memlen: Annotated[uint32_t, 32]
  memfd: Annotated[int32_t, 36]
uintptr_t = ctypes.c_uint64
int32_t = ctypes.c_int32
@record
class struct_fastrpc_ioctl_init_attrs:
  SIZE = 48
  init: Annotated[struct_fastrpc_ioctl_init, 0]
  attrs: Annotated[ctypes.c_int32, 40]
  siglen: Annotated[ctypes.c_uint32, 44]
@record
class struct_fastrpc_ioctl_munmap:
  SIZE = 16
  vaddrout: Annotated[uintptr_t, 0]
  size: Annotated[size_t, 8]
@record
class struct_fastrpc_ioctl_munmap_64:
  SIZE = 16
  vaddrout: Annotated[uint64_t, 0]
  size: Annotated[size_t, 8]
@record
class struct_fastrpc_ioctl_mmap:
  SIZE = 32
  fd: Annotated[ctypes.c_int32, 0]
  flags: Annotated[uint32_t, 4]
  vaddrin: Annotated[uintptr_t, 8]
  size: Annotated[size_t, 16]
  vaddrout: Annotated[uintptr_t, 24]
@record
class struct_fastrpc_ioctl_mmap_64:
  SIZE = 32
  fd: Annotated[ctypes.c_int32, 0]
  flags: Annotated[uint32_t, 4]
  vaddrin: Annotated[uint64_t, 8]
  size: Annotated[size_t, 16]
  vaddrout: Annotated[uint64_t, 24]
@record
class struct_fastrpc_ioctl_munmap_fd:
  SIZE = 24
  fd: Annotated[ctypes.c_int32, 0]
  flags: Annotated[uint32_t, 4]
  va: Annotated[uintptr_t, 8]
  len: Annotated[ssize_t, 16]
ssize_t = ctypes.c_int64
@record
class struct_fastrpc_ioctl_perf:
  SIZE = 24
  data: Annotated[uintptr_t, 0]
  numkeys: Annotated[uint32_t, 8]
  keys: Annotated[uintptr_t, 16]
@record
class struct_fastrpc_ctrl_latency:
  SIZE = 8
  enable: Annotated[uint32_t, 0]
  level: Annotated[uint32_t, 4]
@record
class struct_fastrpc_ctrl_smmu:
  SIZE = 4
  sharedcb: Annotated[uint32_t, 0]
@record
class struct_fastrpc_ctrl_kalloc:
  SIZE = 4
  kalloc_support: Annotated[uint32_t, 0]
@record
class struct_fastrpc_ioctl_control:
  SIZE = 12
  req: Annotated[uint32_t, 0]
  lp: Annotated[struct_fastrpc_ctrl_latency, 4]
  smmu: Annotated[struct_fastrpc_ctrl_smmu, 4]
  kalloc: Annotated[struct_fastrpc_ctrl_kalloc, 4]
@record
class struct_smq_null_invoke:
  SIZE = 16
  ctx: Annotated[uint64_t, 0]
  handle: Annotated[uint32_t, 8]
  sc: Annotated[uint32_t, 12]
@record
class struct_smq_phy_page:
  SIZE = 16
  addr: Annotated[uint64_t, 0]
  size: Annotated[uint64_t, 8]
@record
class struct_smq_invoke_buf:
  SIZE = 8
  num: Annotated[ctypes.c_int32, 0]
  pgidx: Annotated[ctypes.c_int32, 4]
@record
class struct_smq_invoke:
  SIZE = 32
  header: Annotated[struct_smq_null_invoke, 0]
  page: Annotated[struct_smq_phy_page, 16]
@record
class struct_smq_msg:
  SIZE = 40
  pid: Annotated[uint32_t, 0]
  tid: Annotated[uint32_t, 4]
  invoke: Annotated[struct_smq_invoke, 8]
@record
class struct_smq_invoke_rsp:
  SIZE = 16
  ctx: Annotated[uint64_t, 0]
  retval: Annotated[ctypes.c_int32, 8]
remote_handle = ctypes.c_uint32
remote_handle64 = ctypes.c_uint64
fastrpc_async_jobid = ctypes.c_uint64
@record
class remote_buf:
  SIZE = 16
  pv: Annotated[ctypes.POINTER(None), 0]
  nLen: Annotated[size_t, 8]
@record
class remote_dma_handle:
  SIZE = 8
  fd: Annotated[int32_t, 0]
  offset: Annotated[uint32_t, 4]
@record
class remote_arg:
  SIZE = 16
  buf: Annotated[remote_buf, 0]
  h: Annotated[remote_handle, 0]
  h64: Annotated[remote_handle64, 0]
  dma: Annotated[remote_dma_handle, 0]
enum_fastrpc_async_notify_type = CEnum(ctypes.c_uint32)
FASTRPC_ASYNC_NO_SYNC = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_NO_SYNC', 0)
FASTRPC_ASYNC_CALLBACK = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_CALLBACK', 1)
FASTRPC_ASYNC_POLL = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_POLL', 2)
FASTRPC_ASYNC_TYPE_MAX = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_TYPE_MAX', 3)

@record
class struct_fastrpc_async_callback:
  SIZE = 16
  fn: Annotated[ctypes.CFUNCTYPE(None, fastrpc_async_jobid, ctypes.POINTER(None), ctypes.c_int32), 0]
  context: Annotated[ctypes.POINTER(None), 8]
fastrpc_async_callback_t = struct_fastrpc_async_callback
@record
class struct_fastrpc_async_descriptor:
  SIZE = 32
  type: Annotated[enum_fastrpc_async_notify_type, 0]
  jobid: Annotated[fastrpc_async_jobid, 8]
  cb: Annotated[fastrpc_async_callback_t, 16]
fastrpc_async_descriptor_t = struct_fastrpc_async_descriptor
enum_fastrpc_process_type = CEnum(ctypes.c_uint32)
PROCESS_TYPE_SIGNED = enum_fastrpc_process_type.define('PROCESS_TYPE_SIGNED', 0)
PROCESS_TYPE_UNSIGNED = enum_fastrpc_process_type.define('PROCESS_TYPE_UNSIGNED', 1)

enum_handle_control_req_id = CEnum(ctypes.c_uint32)
DSPRPC_CONTROL_LATENCY = enum_handle_control_req_id.define('DSPRPC_CONTROL_LATENCY', 1)
DSPRPC_GET_DSP_INFO = enum_handle_control_req_id.define('DSPRPC_GET_DSP_INFO', 2)
DSPRPC_CONTROL_WAKELOCK = enum_handle_control_req_id.define('DSPRPC_CONTROL_WAKELOCK', 3)
DSPRPC_GET_DOMAIN = enum_handle_control_req_id.define('DSPRPC_GET_DOMAIN', 4)

enum_remote_rpc_latency_flags = CEnum(ctypes.c_uint32)
RPC_DISABLE_QOS = enum_remote_rpc_latency_flags.define('RPC_DISABLE_QOS', 0)
RPC_PM_QOS = enum_remote_rpc_latency_flags.define('RPC_PM_QOS', 1)
RPC_ADAPTIVE_QOS = enum_remote_rpc_latency_flags.define('RPC_ADAPTIVE_QOS', 2)
RPC_POLL_QOS = enum_remote_rpc_latency_flags.define('RPC_POLL_QOS', 3)

remote_rpc_control_latency_t = enum_remote_rpc_latency_flags
@record
class struct_remote_rpc_control_latency:
  SIZE = 8
  enable: Annotated[remote_rpc_control_latency_t, 0]
  latency: Annotated[uint32_t, 4]
enum_remote_dsp_attributes = CEnum(ctypes.c_uint32)
DOMAIN_SUPPORT = enum_remote_dsp_attributes.define('DOMAIN_SUPPORT', 0)
UNSIGNED_PD_SUPPORT = enum_remote_dsp_attributes.define('UNSIGNED_PD_SUPPORT', 1)
HVX_SUPPORT_64B = enum_remote_dsp_attributes.define('HVX_SUPPORT_64B', 2)
HVX_SUPPORT_128B = enum_remote_dsp_attributes.define('HVX_SUPPORT_128B', 3)
VTCM_PAGE = enum_remote_dsp_attributes.define('VTCM_PAGE', 4)
VTCM_COUNT = enum_remote_dsp_attributes.define('VTCM_COUNT', 5)
ARCH_VER = enum_remote_dsp_attributes.define('ARCH_VER', 6)
HMX_SUPPORT_DEPTH = enum_remote_dsp_attributes.define('HMX_SUPPORT_DEPTH', 7)
HMX_SUPPORT_SPATIAL = enum_remote_dsp_attributes.define('HMX_SUPPORT_SPATIAL', 8)
ASYNC_FASTRPC_SUPPORT = enum_remote_dsp_attributes.define('ASYNC_FASTRPC_SUPPORT', 9)
STATUS_NOTIFICATION_SUPPORT = enum_remote_dsp_attributes.define('STATUS_NOTIFICATION_SUPPORT', 10)
FASTRPC_MAX_DSP_ATTRIBUTES = enum_remote_dsp_attributes.define('FASTRPC_MAX_DSP_ATTRIBUTES', 11)

@record
class struct_remote_dsp_capability:
  SIZE = 12
  domain: Annotated[uint32_t, 0]
  attribute_ID: Annotated[uint32_t, 4]
  capability: Annotated[uint32_t, 8]
fastrpc_capability = struct_remote_dsp_capability
@record
class struct_remote_rpc_control_wakelock:
  SIZE = 4
  enable: Annotated[uint32_t, 0]
@record
class struct_remote_rpc_get_domain:
  SIZE = 4
  domain: Annotated[ctypes.c_int32, 0]
remote_rpc_get_domain_t = struct_remote_rpc_get_domain
enum_session_control_req_id = CEnum(ctypes.c_uint32)
FASTRPC_THREAD_PARAMS = enum_session_control_req_id.define('FASTRPC_THREAD_PARAMS', 1)
DSPRPC_CONTROL_UNSIGNED_MODULE = enum_session_control_req_id.define('DSPRPC_CONTROL_UNSIGNED_MODULE', 2)
FASTRPC_RELATIVE_THREAD_PRIORITY = enum_session_control_req_id.define('FASTRPC_RELATIVE_THREAD_PRIORITY', 4)
FASTRPC_REMOTE_PROCESS_KILL = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_KILL', 6)
FASTRPC_SESSION_CLOSE = enum_session_control_req_id.define('FASTRPC_SESSION_CLOSE', 7)
FASTRPC_CONTROL_PD_DUMP = enum_session_control_req_id.define('FASTRPC_CONTROL_PD_DUMP', 8)
FASTRPC_REMOTE_PROCESS_EXCEPTION = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_EXCEPTION', 9)
FASTRPC_REMOTE_PROCESS_TYPE = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_TYPE', 10)
FASTRPC_REGISTER_STATUS_NOTIFICATIONS = enum_session_control_req_id.define('FASTRPC_REGISTER_STATUS_NOTIFICATIONS', 11)

@record
class struct_remote_rpc_thread_params:
  SIZE = 12
  domain: Annotated[ctypes.c_int32, 0]
  prio: Annotated[ctypes.c_int32, 4]
  stack_size: Annotated[ctypes.c_int32, 8]
@record
class struct_remote_rpc_control_unsigned_module:
  SIZE = 8
  domain: Annotated[ctypes.c_int32, 0]
  enable: Annotated[ctypes.c_int32, 4]
@record
class struct_remote_rpc_relative_thread_priority:
  SIZE = 8
  domain: Annotated[ctypes.c_int32, 0]
  relative_thread_priority: Annotated[ctypes.c_int32, 4]
@record
class struct_remote_rpc_process_clean_params:
  SIZE = 4
  domain: Annotated[ctypes.c_int32, 0]
@record
class struct_remote_rpc_session_close:
  SIZE = 4
  domain: Annotated[ctypes.c_int32, 0]
@record
class struct_remote_rpc_control_pd_dump:
  SIZE = 8
  domain: Annotated[ctypes.c_int32, 0]
  enable: Annotated[ctypes.c_int32, 4]
@record
class struct_remote_process_type:
  SIZE = 8
  domain: Annotated[ctypes.c_int32, 0]
  process_type: Annotated[ctypes.c_int32, 4]
remote_rpc_process_exception = struct_remote_rpc_process_clean_params
enum_remote_rpc_status_flags = CEnum(ctypes.c_uint32)
FASTRPC_USER_PD_UP = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_UP', 0)
FASTRPC_USER_PD_EXIT = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_EXIT', 1)
FASTRPC_USER_PD_FORCE_KILL = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_FORCE_KILL', 2)
FASTRPC_USER_PD_EXCEPTION = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_EXCEPTION', 3)
FASTRPC_DSP_SSR = enum_remote_rpc_status_flags.define('FASTRPC_DSP_SSR', 4)

remote_rpc_status_flags_t = enum_remote_rpc_status_flags
fastrpc_notif_fn_t = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.c_int32, ctypes.c_int32, enum_remote_rpc_status_flags)
@record
class struct_remote_rpc_notif_register:
  SIZE = 24
  context: Annotated[ctypes.POINTER(None), 0]
  domain: Annotated[ctypes.c_int32, 8]
  notifier_fn: Annotated[fastrpc_notif_fn_t, 16]
remote_rpc_notif_register_t = struct_remote_rpc_notif_register
enum_remote_mem_map_flags = CEnum(ctypes.c_uint32)
REMOTE_MAP_MEM_STATIC = enum_remote_mem_map_flags.define('REMOTE_MAP_MEM_STATIC', 0)
REMOTE_MAP_MAX_FLAG = enum_remote_mem_map_flags.define('REMOTE_MAP_MAX_FLAG', 1)

enum_remote_buf_attributes = CEnum(ctypes.c_uint32)
FASTRPC_ATTR_NON_COHERENT = enum_remote_buf_attributes.define('FASTRPC_ATTR_NON_COHERENT', 2)
FASTRPC_ATTR_COHERENT = enum_remote_buf_attributes.define('FASTRPC_ATTR_COHERENT', 4)
FASTRPC_ATTR_KEEP_MAP = enum_remote_buf_attributes.define('FASTRPC_ATTR_KEEP_MAP', 8)
FASTRPC_ATTR_NOMAP = enum_remote_buf_attributes.define('FASTRPC_ATTR_NOMAP', 16)
FASTRPC_ATTR_FORCE_NOFLUSH = enum_remote_buf_attributes.define('FASTRPC_ATTR_FORCE_NOFLUSH', 32)
FASTRPC_ATTR_FORCE_NOINVALIDATE = enum_remote_buf_attributes.define('FASTRPC_ATTR_FORCE_NOINVALIDATE', 64)
FASTRPC_ATTR_TRY_MAP_STATIC = enum_remote_buf_attributes.define('FASTRPC_ATTR_TRY_MAP_STATIC', 128)

enum_fastrpc_map_flags = CEnum(ctypes.c_uint32)
FASTRPC_MAP_STATIC = enum_fastrpc_map_flags.define('FASTRPC_MAP_STATIC', 0)
FASTRPC_MAP_RESERVED = enum_fastrpc_map_flags.define('FASTRPC_MAP_RESERVED', 1)
FASTRPC_MAP_FD = enum_fastrpc_map_flags.define('FASTRPC_MAP_FD', 2)
FASTRPC_MAP_FD_DELAYED = enum_fastrpc_map_flags.define('FASTRPC_MAP_FD_DELAYED', 3)
FASTRPC_MAP_MAX = enum_fastrpc_map_flags.define('FASTRPC_MAP_MAX', 4)

@record
class struct__cstring1_s:
  SIZE = 16
  data: Annotated[ctypes.POINTER(ctypes.c_char), 0]
  dataLen: Annotated[ctypes.c_int32, 8]
_cstring1_t = struct__cstring1_s
apps_std_FILE = ctypes.c_int32
enum_apps_std_SEEK = CEnum(ctypes.c_uint32)
APPS_STD_SEEK_SET = enum_apps_std_SEEK.define('APPS_STD_SEEK_SET', 0)
APPS_STD_SEEK_CUR = enum_apps_std_SEEK.define('APPS_STD_SEEK_CUR', 1)
APPS_STD_SEEK_END = enum_apps_std_SEEK.define('APPS_STD_SEEK_END', 2)
_32BIT_PLACEHOLDER_apps_std_SEEK = enum_apps_std_SEEK.define('_32BIT_PLACEHOLDER_apps_std_SEEK', 2147483647)

apps_std_SEEK = enum_apps_std_SEEK
@record
class struct_apps_std_DIR:
  SIZE = 8
  handle: Annotated[uint64, 0]
uint64 = ctypes.c_uint64
apps_std_DIR = struct_apps_std_DIR
@record
class struct_apps_std_DIRENT:
  SIZE = 260
  ino: Annotated[ctypes.c_int32, 0]
  name: Annotated[Array[ctypes.c_char, Literal[255]], 4]
apps_std_DIRENT = struct_apps_std_DIRENT
@record
class struct_apps_std_STAT:
  SIZE = 96
  tsz: Annotated[uint64, 0]
  dev: Annotated[uint64, 8]
  ino: Annotated[uint64, 16]
  mode: Annotated[uint32, 24]
  nlink: Annotated[uint32, 28]
  rdev: Annotated[uint64, 32]
  size: Annotated[uint64, 40]
  atime: Annotated[int64, 48]
  atimensec: Annotated[int64, 56]
  mtime: Annotated[int64, 64]
  mtimensec: Annotated[int64, 72]
  ctime: Annotated[int64, 80]
  ctimensec: Annotated[int64, 88]
uint32 = ctypes.c_uint32
int64 = ctypes.c_int64
apps_std_STAT = struct_apps_std_STAT
init_records()
ION_HEAP_SYSTEM_MASK = ((1 << ION_HEAP_TYPE_SYSTEM))
ION_HEAP_SYSTEM_CONTIG_MASK = ((1 << ION_HEAP_TYPE_SYSTEM_CONTIG))
ION_HEAP_CARVEOUT_MASK = ((1 << ION_HEAP_TYPE_CARVEOUT))
ION_HEAP_TYPE_DMA_MASK = ((1 << ION_HEAP_TYPE_DMA))
ION_FLAG_CACHED = 1
ION_FLAG_CACHED_NEEDS_SYNC = 2
ION_IOC_MAGIC = 'I'
ION_IOC_ALLOC = _IOWR(ION_IOC_MAGIC, 0, struct_ion_allocation_data)
ION_IOC_FREE = _IOWR(ION_IOC_MAGIC, 1, struct_ion_handle_data)
ION_IOC_MAP = _IOWR(ION_IOC_MAGIC, 2, struct_ion_fd_data)
ION_IOC_SHARE = _IOWR(ION_IOC_MAGIC, 4, struct_ion_fd_data)
ION_IOC_IMPORT = _IOWR(ION_IOC_MAGIC, 5, struct_ion_fd_data)
ION_IOC_SYNC = _IOWR(ION_IOC_MAGIC, 7, struct_ion_fd_data)
ION_IOC_CUSTOM = _IOWR(ION_IOC_MAGIC, 6, struct_ion_custom_data)
ION_IOMMU_HEAP_ID = ION_SYSTEM_HEAP_ID
ION_HEAP_TYPE_IOMMU = ION_HEAP_TYPE_SYSTEM
ION_FLAG_CP_TOUCH = (1 << 17)
ION_FLAG_CP_BITSTREAM = (1 << 18)
ION_FLAG_CP_PIXEL = (1 << 19)
ION_FLAG_CP_NON_PIXEL = (1 << 20)
ION_FLAG_CP_CAMERA = (1 << 21)
ION_FLAG_CP_HLOS = (1 << 22)
ION_FLAG_CP_HLOS_FREE = (1 << 23)
ION_FLAG_CP_SEC_DISPLAY = (1 << 25)
ION_FLAG_CP_APP = (1 << 26)
ION_FLAG_ALLOW_NON_CONTIG = (1 << 24)
ION_FLAG_SECURE = (1 << ION_HEAP_ID_RESERVED)
ION_FLAG_FORCE_CONTIGUOUS = (1 << 30)
ION_FLAG_POOL_FORCE_ALLOC = (1 << 16)
ION_FLAG_POOL_PREFETCH = (1 << 27)
ION_SECURE = ION_FLAG_SECURE
ION_FORCE_CONTIGUOUS = ION_FLAG_FORCE_CONTIGUOUS
ION_HEAP = lambda bit: (1 << (bit))
ION_ADSP_HEAP_NAME = "adsp"
ION_SYSTEM_HEAP_NAME = "system"
ION_VMALLOC_HEAP_NAME = ION_SYSTEM_HEAP_NAME
ION_KMALLOC_HEAP_NAME = "kmalloc"
ION_AUDIO_HEAP_NAME = "audio"
ION_SF_HEAP_NAME = "sf"
ION_MM_HEAP_NAME = "mm"
ION_CAMERA_HEAP_NAME = "camera_preview"
ION_IOMMU_HEAP_NAME = "iommu"
ION_MFC_HEAP_NAME = "mfc"
ION_WB_HEAP_NAME = "wb"
ION_MM_FIRMWARE_HEAP_NAME = "mm_fw"
ION_PIL1_HEAP_NAME = "pil_1"
ION_PIL2_HEAP_NAME = "pil_2"
ION_QSECOM_HEAP_NAME = "qsecom"
ION_SECURE_HEAP_NAME = "secure_heap"
ION_SECURE_DISPLAY_HEAP_NAME = "secure_display"
ION_SET_CACHED = lambda __cache: (__cache | ION_FLAG_CACHED)
ION_SET_UNCACHED = lambda __cache: (__cache & ~ION_FLAG_CACHED)
ION_IS_CACHED = lambda __flags: ((__flags) & ION_FLAG_CACHED)
ION_IOC_MSM_MAGIC = 'M'
ION_IOC_CLEAN_CACHES = _IOWR(ION_IOC_MSM_MAGIC, 0, struct_ion_flush_data)
ION_IOC_INV_CACHES = _IOWR(ION_IOC_MSM_MAGIC, 1, struct_ion_flush_data)
ION_IOC_CLEAN_INV_CACHES = _IOWR(ION_IOC_MSM_MAGIC, 2, struct_ion_flush_data)
ION_IOC_PREFETCH = _IOWR(ION_IOC_MSM_MAGIC, 3, struct_ion_prefetch_data)
ION_IOC_DRAIN = _IOWR(ION_IOC_MSM_MAGIC, 4, struct_ion_prefetch_data)
FASTRPC_IOCTL_INVOKE = _IOWR('R', 1, struct_fastrpc_ioctl_invoke)
FASTRPC_IOCTL_MMAP = _IOWR('R', 2, struct_fastrpc_ioctl_mmap)
FASTRPC_IOCTL_MUNMAP = _IOWR('R', 3, struct_fastrpc_ioctl_munmap)
FASTRPC_IOCTL_MMAP_64 = _IOWR('R', 14, struct_fastrpc_ioctl_mmap_64)
FASTRPC_IOCTL_MUNMAP_64 = _IOWR('R', 15, struct_fastrpc_ioctl_munmap_64)
FASTRPC_IOCTL_INVOKE_FD = _IOWR('R', 4, struct_fastrpc_ioctl_invoke_fd)
FASTRPC_IOCTL_SETMODE = _IOWR('R', 5, uint32_t)
FASTRPC_IOCTL_INIT = _IOWR('R', 6, struct_fastrpc_ioctl_init)
FASTRPC_IOCTL_INVOKE_ATTRS = _IOWR('R', 7, struct_fastrpc_ioctl_invoke_attrs)
FASTRPC_IOCTL_GETINFO = _IOWR('R', 8, uint32_t)
FASTRPC_IOCTL_GETPERF = _IOWR('R', 9, struct_fastrpc_ioctl_perf)
FASTRPC_IOCTL_INIT_ATTRS = _IOWR('R', 10, struct_fastrpc_ioctl_init_attrs)
FASTRPC_IOCTL_INVOKE_CRC = _IOWR('R', 11, struct_fastrpc_ioctl_invoke_crc)
FASTRPC_IOCTL_CONTROL = _IOWR('R', 12, struct_fastrpc_ioctl_control)
FASTRPC_IOCTL_MUNMAP_FD = _IOWR('R', 13, struct_fastrpc_ioctl_munmap_fd)
FASTRPC_GLINK_GUID = "fastrpcglink-apps-dsp"
FASTRPC_SMD_GUID = "fastrpcsmd-apps-dsp"
DEVICE_NAME = "adsprpc-smd"
FASTRPC_ATTR_NOVA = 0x1
FASTRPC_ATTR_NON_COHERENT = 0x2
FASTRPC_ATTR_COHERENT = 0x4
FASTRPC_ATTR_KEEP_MAP = 0x8
FASTRPC_ATTR_NOMAP = (16)
FASTRPC_MODE_PARALLEL = 0
FASTRPC_MODE_SERIAL = 1
FASTRPC_MODE_PROFILE = 2
FASTRPC_MODE_SESSION = 4
FASTRPC_INIT_ATTACH = 0
FASTRPC_INIT_CREATE = 1
FASTRPC_INIT_CREATE_STATIC = 2
FASTRPC_INIT_ATTACH_SENSORS = 3
REMOTE_SCALARS_INBUFS = lambda sc: (((sc) >> 16) & 0x0ff)
REMOTE_SCALARS_OUTBUFS = lambda sc: (((sc) >> 8) & 0x0ff)
REMOTE_SCALARS_INHANDLES = lambda sc: (((sc) >> 4) & 0x0f)
REMOTE_SCALARS_OUTHANDLES = lambda sc: ((sc) & 0x0f)
REMOTE_SCALARS_LENGTH = lambda sc: (REMOTE_SCALARS_INBUFS(sc) + REMOTE_SCALARS_OUTBUFS(sc) + REMOTE_SCALARS_INHANDLES(sc) + REMOTE_SCALARS_OUTHANDLES(sc))
__TOSTR__ = lambda x: __STR__(x)
remote_arg64_t = union_remote_arg64
remote_arg_t = union_remote_arg
FASTRPC_CONTROL_LATENCY = (1)
FASTRPC_CONTROL_SMMU = (2)
FASTRPC_CONTROL_KALLOC = (3)
REMOTE_SCALARS_METHOD_ATTR = lambda dwScalars: (((dwScalars) >> 29) & 0x7)
REMOTE_SCALARS_METHOD = lambda dwScalars: (((dwScalars) >> 24) & 0x1f)
REMOTE_SCALARS_INBUFS = lambda dwScalars: (((dwScalars) >> 16) & 0x0ff)
REMOTE_SCALARS_OUTBUFS = lambda dwScalars: (((dwScalars) >> 8) & 0x0ff)
REMOTE_SCALARS_INHANDLES = lambda dwScalars: (((dwScalars) >> 4) & 0x0f)
REMOTE_SCALARS_OUTHANDLES = lambda dwScalars: ((dwScalars) & 0x0f)
REMOTE_SCALARS_MAKEX = lambda nAttr,nMethod,nIn,nOut,noIn,noOut: ((((uint32_t)   (nAttr) &  0x7) << 29) | (((uint32_t) (nMethod) & 0x1f) << 24) | (((uint32_t)     (nIn) & 0xff) << 16) | (((uint32_t)    (nOut) & 0xff) <<  8) | (((uint32_t)    (noIn) & 0x0f) <<  4) | ((uint32_t)   (noOut) & 0x0f))
REMOTE_SCALARS_MAKE = lambda nMethod,nIn,nOut: REMOTE_SCALARS_MAKEX(0,nMethod,nIn,nOut,0,0)
REMOTE_SCALARS_LENGTH = lambda sc: (REMOTE_SCALARS_INBUFS(sc) + REMOTE_SCALARS_OUTBUFS(sc) + REMOTE_SCALARS_INHANDLES(sc) + REMOTE_SCALARS_OUTHANDLES(sc))
__QAIC_REMOTE = lambda ff: ff
NUM_DOMAINS = 4
NUM_SESSIONS = 2
DOMAIN_ID_MASK = 3
DEFAULT_DOMAIN_ID = 0
ADSP_DOMAIN_ID = 0
MDSP_DOMAIN_ID = 1
SDSP_DOMAIN_ID = 2
CDSP_DOMAIN_ID = 3
ADSP_DOMAIN = "&_dom=adsp"
MDSP_DOMAIN = "&_dom=mdsp"
SDSP_DOMAIN = "&_dom=sdsp"
CDSP_DOMAIN = "&_dom=cdsp"
FASTRPC_WAKELOCK_CONTROL_SUPPORTED = 1
REMOTE_MODE_PARALLEL = 0
REMOTE_MODE_SERIAL = 1
ITRANSPORT_PREFIX = "'\":;./\\"
__QAIC_HEADER = lambda ff: ff
__QAIC_IMPL = lambda ff: ff