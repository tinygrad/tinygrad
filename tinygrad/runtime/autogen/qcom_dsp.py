# mypy: ignore-errors
import ctypes
from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR
ion_user_handle_t = ctypes.c_int
enum_ion_heap_type = CEnum(ctypes.c_uint)
ION_HEAP_TYPE_SYSTEM = enum_ion_heap_type.define('ION_HEAP_TYPE_SYSTEM', 0)
ION_HEAP_TYPE_SYSTEM_CONTIG = enum_ion_heap_type.define('ION_HEAP_TYPE_SYSTEM_CONTIG', 1)
ION_HEAP_TYPE_CARVEOUT = enum_ion_heap_type.define('ION_HEAP_TYPE_CARVEOUT', 2)
ION_HEAP_TYPE_CHUNK = enum_ion_heap_type.define('ION_HEAP_TYPE_CHUNK', 3)
ION_HEAP_TYPE_DMA = enum_ion_heap_type.define('ION_HEAP_TYPE_DMA', 4)
ION_HEAP_TYPE_CUSTOM = enum_ion_heap_type.define('ION_HEAP_TYPE_CUSTOM', 5)
ION_NUM_HEAPS = enum_ion_heap_type.define('ION_NUM_HEAPS', 16)

class struct_ion_allocation_data(ctypes.Structure): pass
size_t = ctypes.c_ulong
struct_ion_allocation_data._fields_ = [
  ('len', size_t),
  ('align', size_t),
  ('heap_id_mask', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('handle', ion_user_handle_t),
]
class struct_ion_fd_data(ctypes.Structure): pass
struct_ion_fd_data._fields_ = [
  ('handle', ion_user_handle_t),
  ('fd', ctypes.c_int),
]
class struct_ion_handle_data(ctypes.Structure): pass
struct_ion_handle_data._fields_ = [
  ('handle', ion_user_handle_t),
]
class struct_ion_custom_data(ctypes.Structure): pass
struct_ion_custom_data._fields_ = [
  ('cmd', ctypes.c_uint),
  ('arg', ctypes.c_ulong),
]
enum_msm_ion_heap_types = CEnum(ctypes.c_uint)
ION_HEAP_TYPE_MSM_START = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_MSM_START', 6)
ION_HEAP_TYPE_SECURE_DMA = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_SECURE_DMA', 6)
ION_HEAP_TYPE_SYSTEM_SECURE = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_SYSTEM_SECURE', 7)
ION_HEAP_TYPE_HYP_CMA = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_HYP_CMA', 8)

enum_ion_heap_ids = CEnum(ctypes.c_int)
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

enum_ion_fixed_position = CEnum(ctypes.c_uint)
NOT_FIXED = enum_ion_fixed_position.define('NOT_FIXED', 0)
FIXED_LOW = enum_ion_fixed_position.define('FIXED_LOW', 1)
FIXED_MIDDLE = enum_ion_fixed_position.define('FIXED_MIDDLE', 2)
FIXED_HIGH = enum_ion_fixed_position.define('FIXED_HIGH', 3)

enum_cp_mem_usage = CEnum(ctypes.c_uint)
VIDEO_BITSTREAM = enum_cp_mem_usage.define('VIDEO_BITSTREAM', 1)
VIDEO_PIXEL = enum_cp_mem_usage.define('VIDEO_PIXEL', 2)
VIDEO_NONPIXEL = enum_cp_mem_usage.define('VIDEO_NONPIXEL', 3)
DISPLAY_SECURE_CP_USAGE = enum_cp_mem_usage.define('DISPLAY_SECURE_CP_USAGE', 4)
CAMERA_SECURE_CP_USAGE = enum_cp_mem_usage.define('CAMERA_SECURE_CP_USAGE', 5)
MAX_USAGE = enum_cp_mem_usage.define('MAX_USAGE', 6)
UNKNOWN = enum_cp_mem_usage.define('UNKNOWN', 2147483647)

class struct_ion_flush_data(ctypes.Structure): pass
struct_ion_flush_data._fields_ = [
  ('handle', ion_user_handle_t),
  ('fd', ctypes.c_int),
  ('vaddr', ctypes.c_void_p),
  ('offset', ctypes.c_uint),
  ('length', ctypes.c_uint),
]
class struct_ion_prefetch_regions(ctypes.Structure): pass
struct_ion_prefetch_regions._fields_ = [
  ('vmid', ctypes.c_uint),
  ('sizes', ctypes.POINTER(size_t)),
  ('nr_sizes', ctypes.c_uint),
]
class struct_ion_prefetch_data(ctypes.Structure): pass
struct_ion_prefetch_data._fields_ = [
  ('heap_id', ctypes.c_int),
  ('len', ctypes.c_ulong),
  ('regions', ctypes.POINTER(struct_ion_prefetch_regions)),
  ('nr_regions', ctypes.c_uint),
]
class struct_remote_buf64(ctypes.Structure): pass
uint64_t = ctypes.c_ulong
struct_remote_buf64._fields_ = [
  ('pv', uint64_t),
  ('len', uint64_t),
]
class struct_remote_dma_handle64(ctypes.Structure): pass
uint32_t = ctypes.c_uint
struct_remote_dma_handle64._fields_ = [
  ('fd', ctypes.c_int),
  ('offset', uint32_t),
  ('len', uint32_t),
]
class union_remote_arg64(ctypes.Union): pass
union_remote_arg64._fields_ = [
  ('buf', struct_remote_buf64),
  ('dma', struct_remote_dma_handle64),
  ('h', uint32_t),
]
class struct_remote_buf(ctypes.Structure): pass
struct_remote_buf._fields_ = [
  ('pv', ctypes.c_void_p),
  ('len', size_t),
]
class struct_remote_dma_handle(ctypes.Structure): pass
struct_remote_dma_handle._fields_ = [
  ('fd', ctypes.c_int),
  ('offset', uint32_t),
]
class union_remote_arg(ctypes.Union): pass
union_remote_arg._fields_ = [
  ('buf', struct_remote_buf),
  ('dma', struct_remote_dma_handle),
  ('h', uint32_t),
]
class struct_fastrpc_ioctl_invoke(ctypes.Structure): pass
struct_fastrpc_ioctl_invoke._fields_ = [
  ('handle', uint32_t),
  ('sc', uint32_t),
  ('pra', ctypes.POINTER(union_remote_arg)),
]
class struct_fastrpc_ioctl_invoke_fd(ctypes.Structure): pass
struct_fastrpc_ioctl_invoke_fd._fields_ = [
  ('inv', struct_fastrpc_ioctl_invoke),
  ('fds', ctypes.POINTER(ctypes.c_int)),
]
class struct_fastrpc_ioctl_invoke_attrs(ctypes.Structure): pass
struct_fastrpc_ioctl_invoke_attrs._fields_ = [
  ('inv', struct_fastrpc_ioctl_invoke),
  ('fds', ctypes.POINTER(ctypes.c_int)),
  ('attrs', ctypes.POINTER(ctypes.c_uint)),
]
class struct_fastrpc_ioctl_invoke_crc(ctypes.Structure): pass
struct_fastrpc_ioctl_invoke_crc._fields_ = [
  ('inv', struct_fastrpc_ioctl_invoke),
  ('fds', ctypes.POINTER(ctypes.c_int)),
  ('attrs', ctypes.POINTER(ctypes.c_uint)),
  ('crc', ctypes.POINTER(ctypes.c_uint)),
]
class struct_fastrpc_ioctl_init(ctypes.Structure): pass
uintptr_t = ctypes.c_ulong
int32_t = ctypes.c_int
struct_fastrpc_ioctl_init._fields_ = [
  ('flags', uint32_t),
  ('file', uintptr_t),
  ('filelen', uint32_t),
  ('filefd', int32_t),
  ('mem', uintptr_t),
  ('memlen', uint32_t),
  ('memfd', int32_t),
]
class struct_fastrpc_ioctl_init_attrs(ctypes.Structure): pass
struct_fastrpc_ioctl_init_attrs._fields_ = [
  ('init', struct_fastrpc_ioctl_init),
  ('attrs', ctypes.c_int),
  ('siglen', ctypes.c_uint),
]
class struct_fastrpc_ioctl_munmap(ctypes.Structure): pass
struct_fastrpc_ioctl_munmap._fields_ = [
  ('vaddrout', uintptr_t),
  ('size', size_t),
]
class struct_fastrpc_ioctl_munmap_64(ctypes.Structure): pass
struct_fastrpc_ioctl_munmap_64._fields_ = [
  ('vaddrout', uint64_t),
  ('size', size_t),
]
class struct_fastrpc_ioctl_mmap(ctypes.Structure): pass
struct_fastrpc_ioctl_mmap._fields_ = [
  ('fd', ctypes.c_int),
  ('flags', uint32_t),
  ('vaddrin', uintptr_t),
  ('size', size_t),
  ('vaddrout', uintptr_t),
]
class struct_fastrpc_ioctl_mmap_64(ctypes.Structure): pass
struct_fastrpc_ioctl_mmap_64._fields_ = [
  ('fd', ctypes.c_int),
  ('flags', uint32_t),
  ('vaddrin', uint64_t),
  ('size', size_t),
  ('vaddrout', uint64_t),
]
class struct_fastrpc_ioctl_munmap_fd(ctypes.Structure): pass
ssize_t = ctypes.c_long
struct_fastrpc_ioctl_munmap_fd._fields_ = [
  ('fd', ctypes.c_int),
  ('flags', uint32_t),
  ('va', uintptr_t),
  ('len', ssize_t),
]
class struct_fastrpc_ioctl_perf(ctypes.Structure): pass
struct_fastrpc_ioctl_perf._fields_ = [
  ('data', uintptr_t),
  ('numkeys', uint32_t),
  ('keys', uintptr_t),
]
class struct_fastrpc_ctrl_latency(ctypes.Structure): pass
struct_fastrpc_ctrl_latency._fields_ = [
  ('enable', uint32_t),
  ('level', uint32_t),
]
class struct_fastrpc_ctrl_smmu(ctypes.Structure): pass
struct_fastrpc_ctrl_smmu._fields_ = [
  ('sharedcb', uint32_t),
]
class struct_fastrpc_ctrl_kalloc(ctypes.Structure): pass
struct_fastrpc_ctrl_kalloc._fields_ = [
  ('kalloc_support', uint32_t),
]
class struct_fastrpc_ioctl_control(ctypes.Structure): pass
class _anonunion0(ctypes.Union): pass
_anonunion0._fields_ = [
  ('lp', struct_fastrpc_ctrl_latency),
  ('smmu', struct_fastrpc_ctrl_smmu),
  ('kalloc', struct_fastrpc_ctrl_kalloc),
]
struct_fastrpc_ioctl_control._anonymous_ = ['_0']
struct_fastrpc_ioctl_control._fields_ = [
  ('req', uint32_t),
  ('_0', _anonunion0),
]
class struct_smq_null_invoke(ctypes.Structure): pass
struct_smq_null_invoke._fields_ = [
  ('ctx', uint64_t),
  ('handle', uint32_t),
  ('sc', uint32_t),
]
class struct_smq_phy_page(ctypes.Structure): pass
struct_smq_phy_page._fields_ = [
  ('addr', uint64_t),
  ('size', uint64_t),
]
class struct_smq_invoke_buf(ctypes.Structure): pass
struct_smq_invoke_buf._fields_ = [
  ('num', ctypes.c_int),
  ('pgidx', ctypes.c_int),
]
class struct_smq_invoke(ctypes.Structure): pass
struct_smq_invoke._fields_ = [
  ('header', struct_smq_null_invoke),
  ('page', struct_smq_phy_page),
]
class struct_smq_msg(ctypes.Structure): pass
struct_smq_msg._fields_ = [
  ('pid', uint32_t),
  ('tid', uint32_t),
  ('invoke', struct_smq_invoke),
]
class struct_smq_invoke_rsp(ctypes.Structure): pass
struct_smq_invoke_rsp._fields_ = [
  ('ctx', uint64_t),
  ('retval', ctypes.c_int),
]
remote_handle = ctypes.c_uint
remote_handle64 = ctypes.c_ulong
fastrpc_async_jobid = ctypes.c_ulong
class remote_buf(ctypes.Structure): pass
remote_buf._fields_ = [
  ('pv', ctypes.c_void_p),
  ('nLen', size_t),
]
class remote_dma_handle(ctypes.Structure): pass
remote_dma_handle._fields_ = [
  ('fd', int32_t),
  ('offset', uint32_t),
]
class remote_arg(ctypes.Union): pass
remote_arg._fields_ = [
  ('buf', remote_buf),
  ('h', remote_handle),
  ('h64', remote_handle64),
  ('dma', remote_dma_handle),
]
enum_fastrpc_async_notify_type = CEnum(ctypes.c_uint)
FASTRPC_ASYNC_NO_SYNC = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_NO_SYNC', 0)
FASTRPC_ASYNC_CALLBACK = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_CALLBACK', 1)
FASTRPC_ASYNC_POLL = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_POLL', 2)
FASTRPC_ASYNC_TYPE_MAX = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_TYPE_MAX', 3)

class struct_fastrpc_async_callback(ctypes.Structure): pass
struct_fastrpc_async_callback._fields_ = [
  ('fn', ctypes.CFUNCTYPE(None, fastrpc_async_jobid, ctypes.c_void_p, ctypes.c_int)),
  ('context', ctypes.c_void_p),
]
fastrpc_async_callback_t = struct_fastrpc_async_callback
class struct_fastrpc_async_descriptor(ctypes.Structure): pass
class _anonunion1(ctypes.Union): pass
_anonunion1._fields_ = [
  ('cb', fastrpc_async_callback_t),
]
struct_fastrpc_async_descriptor._anonymous_ = ['_0']
struct_fastrpc_async_descriptor._fields_ = [
  ('type', enum_fastrpc_async_notify_type),
  ('jobid', fastrpc_async_jobid),
  ('_0', _anonunion1),
]
fastrpc_async_descriptor_t = struct_fastrpc_async_descriptor
enum_fastrpc_process_type = CEnum(ctypes.c_uint)
PROCESS_TYPE_SIGNED = enum_fastrpc_process_type.define('PROCESS_TYPE_SIGNED', 0)
PROCESS_TYPE_UNSIGNED = enum_fastrpc_process_type.define('PROCESS_TYPE_UNSIGNED', 1)

enum_handle_control_req_id = CEnum(ctypes.c_uint)
DSPRPC_CONTROL_LATENCY = enum_handle_control_req_id.define('DSPRPC_CONTROL_LATENCY', 1)
DSPRPC_GET_DSP_INFO = enum_handle_control_req_id.define('DSPRPC_GET_DSP_INFO', 2)
DSPRPC_CONTROL_WAKELOCK = enum_handle_control_req_id.define('DSPRPC_CONTROL_WAKELOCK', 3)
DSPRPC_GET_DOMAIN = enum_handle_control_req_id.define('DSPRPC_GET_DOMAIN', 4)

enum_remote_rpc_latency_flags = CEnum(ctypes.c_uint)
RPC_DISABLE_QOS = enum_remote_rpc_latency_flags.define('RPC_DISABLE_QOS', 0)
RPC_PM_QOS = enum_remote_rpc_latency_flags.define('RPC_PM_QOS', 1)
RPC_ADAPTIVE_QOS = enum_remote_rpc_latency_flags.define('RPC_ADAPTIVE_QOS', 2)
RPC_POLL_QOS = enum_remote_rpc_latency_flags.define('RPC_POLL_QOS', 3)

remote_rpc_control_latency_t = enum_remote_rpc_latency_flags
class struct_remote_rpc_control_latency(ctypes.Structure): pass
struct_remote_rpc_control_latency._fields_ = [
  ('enable', remote_rpc_control_latency_t),
  ('latency', uint32_t),
]
enum_remote_dsp_attributes = CEnum(ctypes.c_uint)
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

class struct_remote_dsp_capability(ctypes.Structure): pass
struct_remote_dsp_capability._fields_ = [
  ('domain', uint32_t),
  ('attribute_ID', uint32_t),
  ('capability', uint32_t),
]
fastrpc_capability = struct_remote_dsp_capability
class struct_remote_rpc_control_wakelock(ctypes.Structure): pass
struct_remote_rpc_control_wakelock._fields_ = [
  ('enable', uint32_t),
]
class struct_remote_rpc_get_domain(ctypes.Structure): pass
struct_remote_rpc_get_domain._fields_ = [
  ('domain', ctypes.c_int),
]
remote_rpc_get_domain_t = struct_remote_rpc_get_domain
enum_session_control_req_id = CEnum(ctypes.c_uint)
FASTRPC_THREAD_PARAMS = enum_session_control_req_id.define('FASTRPC_THREAD_PARAMS', 1)
DSPRPC_CONTROL_UNSIGNED_MODULE = enum_session_control_req_id.define('DSPRPC_CONTROL_UNSIGNED_MODULE', 2)
FASTRPC_RELATIVE_THREAD_PRIORITY = enum_session_control_req_id.define('FASTRPC_RELATIVE_THREAD_PRIORITY', 4)
FASTRPC_REMOTE_PROCESS_KILL = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_KILL', 6)
FASTRPC_SESSION_CLOSE = enum_session_control_req_id.define('FASTRPC_SESSION_CLOSE', 7)
FASTRPC_CONTROL_PD_DUMP = enum_session_control_req_id.define('FASTRPC_CONTROL_PD_DUMP', 8)
FASTRPC_REMOTE_PROCESS_EXCEPTION = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_EXCEPTION', 9)
FASTRPC_REMOTE_PROCESS_TYPE = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_TYPE', 10)
FASTRPC_REGISTER_STATUS_NOTIFICATIONS = enum_session_control_req_id.define('FASTRPC_REGISTER_STATUS_NOTIFICATIONS', 11)

class struct_remote_rpc_thread_params(ctypes.Structure): pass
struct_remote_rpc_thread_params._fields_ = [
  ('domain', ctypes.c_int),
  ('prio', ctypes.c_int),
  ('stack_size', ctypes.c_int),
]
class struct_remote_rpc_control_unsigned_module(ctypes.Structure): pass
struct_remote_rpc_control_unsigned_module._fields_ = [
  ('domain', ctypes.c_int),
  ('enable', ctypes.c_int),
]
class struct_remote_rpc_relative_thread_priority(ctypes.Structure): pass
struct_remote_rpc_relative_thread_priority._fields_ = [
  ('domain', ctypes.c_int),
  ('relative_thread_priority', ctypes.c_int),
]
class struct_remote_rpc_process_clean_params(ctypes.Structure): pass
struct_remote_rpc_process_clean_params._fields_ = [
  ('domain', ctypes.c_int),
]
class struct_remote_rpc_session_close(ctypes.Structure): pass
struct_remote_rpc_session_close._fields_ = [
  ('domain', ctypes.c_int),
]
class struct_remote_rpc_control_pd_dump(ctypes.Structure): pass
struct_remote_rpc_control_pd_dump._fields_ = [
  ('domain', ctypes.c_int),
  ('enable', ctypes.c_int),
]
class struct_remote_process_type(ctypes.Structure): pass
struct_remote_process_type._fields_ = [
  ('domain', ctypes.c_int),
  ('process_type', ctypes.c_int),
]
remote_rpc_process_exception = struct_remote_rpc_process_clean_params
enum_remote_rpc_status_flags = CEnum(ctypes.c_uint)
FASTRPC_USER_PD_UP = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_UP', 0)
FASTRPC_USER_PD_EXIT = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_EXIT', 1)
FASTRPC_USER_PD_FORCE_KILL = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_FORCE_KILL', 2)
FASTRPC_USER_PD_EXCEPTION = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_EXCEPTION', 3)
FASTRPC_DSP_SSR = enum_remote_rpc_status_flags.define('FASTRPC_DSP_SSR', 4)

remote_rpc_status_flags_t = enum_remote_rpc_status_flags
fastrpc_notif_fn_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, enum_remote_rpc_status_flags)
class struct_remote_rpc_notif_register(ctypes.Structure): pass
struct_remote_rpc_notif_register._fields_ = [
  ('context', ctypes.c_void_p),
  ('domain', ctypes.c_int),
  ('notifier_fn', fastrpc_notif_fn_t),
]
remote_rpc_notif_register_t = struct_remote_rpc_notif_register
enum_remote_mem_map_flags = CEnum(ctypes.c_uint)
REMOTE_MAP_MEM_STATIC = enum_remote_mem_map_flags.define('REMOTE_MAP_MEM_STATIC', 0)
REMOTE_MAP_MAX_FLAG = enum_remote_mem_map_flags.define('REMOTE_MAP_MAX_FLAG', 1)

enum_remote_buf_attributes = CEnum(ctypes.c_uint)
FASTRPC_ATTR_NON_COHERENT = enum_remote_buf_attributes.define('FASTRPC_ATTR_NON_COHERENT', 2)
FASTRPC_ATTR_COHERENT = enum_remote_buf_attributes.define('FASTRPC_ATTR_COHERENT', 4)
FASTRPC_ATTR_KEEP_MAP = enum_remote_buf_attributes.define('FASTRPC_ATTR_KEEP_MAP', 8)
FASTRPC_ATTR_NOMAP = enum_remote_buf_attributes.define('FASTRPC_ATTR_NOMAP', 16)
FASTRPC_ATTR_FORCE_NOFLUSH = enum_remote_buf_attributes.define('FASTRPC_ATTR_FORCE_NOFLUSH', 32)
FASTRPC_ATTR_FORCE_NOINVALIDATE = enum_remote_buf_attributes.define('FASTRPC_ATTR_FORCE_NOINVALIDATE', 64)
FASTRPC_ATTR_TRY_MAP_STATIC = enum_remote_buf_attributes.define('FASTRPC_ATTR_TRY_MAP_STATIC', 128)

enum_fastrpc_map_flags = CEnum(ctypes.c_uint)
FASTRPC_MAP_STATIC = enum_fastrpc_map_flags.define('FASTRPC_MAP_STATIC', 0)
FASTRPC_MAP_RESERVED = enum_fastrpc_map_flags.define('FASTRPC_MAP_RESERVED', 1)
FASTRPC_MAP_FD = enum_fastrpc_map_flags.define('FASTRPC_MAP_FD', 2)
FASTRPC_MAP_FD_DELAYED = enum_fastrpc_map_flags.define('FASTRPC_MAP_FD_DELAYED', 3)
FASTRPC_MAP_MAX = enum_fastrpc_map_flags.define('FASTRPC_MAP_MAX', 4)

class struct__cstring1_s(ctypes.Structure): pass
struct__cstring1_s._fields_ = [
  ('data', ctypes.POINTER(ctypes.c_char)),
  ('dataLen', ctypes.c_int),
]
_cstring1_t = struct__cstring1_s
apps_std_FILE = ctypes.c_int
enum_apps_std_SEEK = CEnum(ctypes.c_uint)
APPS_STD_SEEK_SET = enum_apps_std_SEEK.define('APPS_STD_SEEK_SET', 0)
APPS_STD_SEEK_CUR = enum_apps_std_SEEK.define('APPS_STD_SEEK_CUR', 1)
APPS_STD_SEEK_END = enum_apps_std_SEEK.define('APPS_STD_SEEK_END', 2)
_32BIT_PLACEHOLDER_apps_std_SEEK = enum_apps_std_SEEK.define('_32BIT_PLACEHOLDER_apps_std_SEEK', 2147483647)

apps_std_SEEK = enum_apps_std_SEEK
class struct_apps_std_DIR(ctypes.Structure): pass
uint64 = ctypes.c_ulonglong
struct_apps_std_DIR._fields_ = [
  ('handle', uint64),
]
apps_std_DIR = struct_apps_std_DIR
class struct_apps_std_DIRENT(ctypes.Structure): pass
struct_apps_std_DIRENT._fields_ = [
  ('ino', ctypes.c_int),
  ('name', (ctypes.c_char * 255)),
]
apps_std_DIRENT = struct_apps_std_DIRENT
class struct_apps_std_STAT(ctypes.Structure): pass
uint32 = ctypes.c_uint
int64 = ctypes.c_longlong
struct_apps_std_STAT._fields_ = [
  ('tsz', uint64),
  ('dev', uint64),
  ('ino', uint64),
  ('mode', uint32),
  ('nlink', uint32),
  ('rdev', uint64),
  ('size', uint64),
  ('atime', int64),
  ('atimensec', int64),
  ('mtime', int64),
  ('mtimensec', int64),
  ('ctime', int64),
  ('ctimensec', int64),
]
apps_std_STAT = struct_apps_std_STAT
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