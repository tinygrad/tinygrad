# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('ib', 'ibverbs', use_errno=True)
class union_ibv_gid(Union): pass
uint8_t = ctypes.c_ubyte
class _anonstruct0(Struct): pass
__be64 = ctypes.c_uint64
_anonstruct0.SIZE = 16
_anonstruct0._fields_ = ['subnet_prefix', 'interface_id']
setattr(_anonstruct0, 'subnet_prefix', field(0, ctypes.c_uint64))
setattr(_anonstruct0, 'interface_id', field(8, ctypes.c_uint64))
union_ibv_gid.SIZE = 16
union_ibv_gid._fields_ = ['raw', 'global']
setattr(union_ibv_gid, 'raw', field(0, Array(uint8_t, 16)))
setattr(union_ibv_gid, 'global', field(0, _anonstruct0))
enum_ibv_gid_type = CEnum(ctypes.c_uint32)
IBV_GID_TYPE_IB = enum_ibv_gid_type.define('IBV_GID_TYPE_IB', 0)
IBV_GID_TYPE_ROCE_V1 = enum_ibv_gid_type.define('IBV_GID_TYPE_ROCE_V1', 1)
IBV_GID_TYPE_ROCE_V2 = enum_ibv_gid_type.define('IBV_GID_TYPE_ROCE_V2', 2)

class struct_ibv_gid_entry(Struct): pass
uint32_t = ctypes.c_uint32
struct_ibv_gid_entry.SIZE = 32
struct_ibv_gid_entry._fields_ = ['gid', 'gid_index', 'port_num', 'gid_type', 'ndev_ifindex']
setattr(struct_ibv_gid_entry, 'gid', field(0, union_ibv_gid))
setattr(struct_ibv_gid_entry, 'gid_index', field(16, uint32_t))
setattr(struct_ibv_gid_entry, 'port_num', field(20, uint32_t))
setattr(struct_ibv_gid_entry, 'gid_type', field(24, uint32_t))
setattr(struct_ibv_gid_entry, 'ndev_ifindex', field(28, uint32_t))
enum_ibv_node_type = CEnum(ctypes.c_int32)
IBV_NODE_UNKNOWN = enum_ibv_node_type.define('IBV_NODE_UNKNOWN', -1)
IBV_NODE_CA = enum_ibv_node_type.define('IBV_NODE_CA', 1)
IBV_NODE_SWITCH = enum_ibv_node_type.define('IBV_NODE_SWITCH', 2)
IBV_NODE_ROUTER = enum_ibv_node_type.define('IBV_NODE_ROUTER', 3)
IBV_NODE_RNIC = enum_ibv_node_type.define('IBV_NODE_RNIC', 4)
IBV_NODE_USNIC = enum_ibv_node_type.define('IBV_NODE_USNIC', 5)
IBV_NODE_USNIC_UDP = enum_ibv_node_type.define('IBV_NODE_USNIC_UDP', 6)
IBV_NODE_UNSPECIFIED = enum_ibv_node_type.define('IBV_NODE_UNSPECIFIED', 7)

enum_ibv_transport_type = CEnum(ctypes.c_int32)
IBV_TRANSPORT_UNKNOWN = enum_ibv_transport_type.define('IBV_TRANSPORT_UNKNOWN', -1)
IBV_TRANSPORT_IB = enum_ibv_transport_type.define('IBV_TRANSPORT_IB', 0)
IBV_TRANSPORT_IWARP = enum_ibv_transport_type.define('IBV_TRANSPORT_IWARP', 1)
IBV_TRANSPORT_USNIC = enum_ibv_transport_type.define('IBV_TRANSPORT_USNIC', 2)
IBV_TRANSPORT_USNIC_UDP = enum_ibv_transport_type.define('IBV_TRANSPORT_USNIC_UDP', 3)
IBV_TRANSPORT_UNSPECIFIED = enum_ibv_transport_type.define('IBV_TRANSPORT_UNSPECIFIED', 4)

enum_ibv_device_cap_flags = CEnum(ctypes.c_uint32)
IBV_DEVICE_RESIZE_MAX_WR = enum_ibv_device_cap_flags.define('IBV_DEVICE_RESIZE_MAX_WR', 1)
IBV_DEVICE_BAD_PKEY_CNTR = enum_ibv_device_cap_flags.define('IBV_DEVICE_BAD_PKEY_CNTR', 2)
IBV_DEVICE_BAD_QKEY_CNTR = enum_ibv_device_cap_flags.define('IBV_DEVICE_BAD_QKEY_CNTR', 4)
IBV_DEVICE_RAW_MULTI = enum_ibv_device_cap_flags.define('IBV_DEVICE_RAW_MULTI', 8)
IBV_DEVICE_AUTO_PATH_MIG = enum_ibv_device_cap_flags.define('IBV_DEVICE_AUTO_PATH_MIG', 16)
IBV_DEVICE_CHANGE_PHY_PORT = enum_ibv_device_cap_flags.define('IBV_DEVICE_CHANGE_PHY_PORT', 32)
IBV_DEVICE_UD_AV_PORT_ENFORCE = enum_ibv_device_cap_flags.define('IBV_DEVICE_UD_AV_PORT_ENFORCE', 64)
IBV_DEVICE_CURR_QP_STATE_MOD = enum_ibv_device_cap_flags.define('IBV_DEVICE_CURR_QP_STATE_MOD', 128)
IBV_DEVICE_SHUTDOWN_PORT = enum_ibv_device_cap_flags.define('IBV_DEVICE_SHUTDOWN_PORT', 256)
IBV_DEVICE_INIT_TYPE = enum_ibv_device_cap_flags.define('IBV_DEVICE_INIT_TYPE', 512)
IBV_DEVICE_PORT_ACTIVE_EVENT = enum_ibv_device_cap_flags.define('IBV_DEVICE_PORT_ACTIVE_EVENT', 1024)
IBV_DEVICE_SYS_IMAGE_GUID = enum_ibv_device_cap_flags.define('IBV_DEVICE_SYS_IMAGE_GUID', 2048)
IBV_DEVICE_RC_RNR_NAK_GEN = enum_ibv_device_cap_flags.define('IBV_DEVICE_RC_RNR_NAK_GEN', 4096)
IBV_DEVICE_SRQ_RESIZE = enum_ibv_device_cap_flags.define('IBV_DEVICE_SRQ_RESIZE', 8192)
IBV_DEVICE_N_NOTIFY_CQ = enum_ibv_device_cap_flags.define('IBV_DEVICE_N_NOTIFY_CQ', 16384)
IBV_DEVICE_MEM_WINDOW = enum_ibv_device_cap_flags.define('IBV_DEVICE_MEM_WINDOW', 131072)
IBV_DEVICE_UD_IP_CSUM = enum_ibv_device_cap_flags.define('IBV_DEVICE_UD_IP_CSUM', 262144)
IBV_DEVICE_XRC = enum_ibv_device_cap_flags.define('IBV_DEVICE_XRC', 1048576)
IBV_DEVICE_MEM_MGT_EXTENSIONS = enum_ibv_device_cap_flags.define('IBV_DEVICE_MEM_MGT_EXTENSIONS', 2097152)
IBV_DEVICE_MEM_WINDOW_TYPE_2A = enum_ibv_device_cap_flags.define('IBV_DEVICE_MEM_WINDOW_TYPE_2A', 8388608)
IBV_DEVICE_MEM_WINDOW_TYPE_2B = enum_ibv_device_cap_flags.define('IBV_DEVICE_MEM_WINDOW_TYPE_2B', 16777216)
IBV_DEVICE_RC_IP_CSUM = enum_ibv_device_cap_flags.define('IBV_DEVICE_RC_IP_CSUM', 33554432)
IBV_DEVICE_RAW_IP_CSUM = enum_ibv_device_cap_flags.define('IBV_DEVICE_RAW_IP_CSUM', 67108864)
IBV_DEVICE_MANAGED_FLOW_STEERING = enum_ibv_device_cap_flags.define('IBV_DEVICE_MANAGED_FLOW_STEERING', 536870912)

enum_ibv_fork_status = CEnum(ctypes.c_uint32)
IBV_FORK_DISABLED = enum_ibv_fork_status.define('IBV_FORK_DISABLED', 0)
IBV_FORK_ENABLED = enum_ibv_fork_status.define('IBV_FORK_ENABLED', 1)
IBV_FORK_UNNEEDED = enum_ibv_fork_status.define('IBV_FORK_UNNEEDED', 2)

enum_ibv_atomic_cap = CEnum(ctypes.c_uint32)
IBV_ATOMIC_NONE = enum_ibv_atomic_cap.define('IBV_ATOMIC_NONE', 0)
IBV_ATOMIC_HCA = enum_ibv_atomic_cap.define('IBV_ATOMIC_HCA', 1)
IBV_ATOMIC_GLOB = enum_ibv_atomic_cap.define('IBV_ATOMIC_GLOB', 2)

class struct_ibv_alloc_dm_attr(Struct): pass
size_t = ctypes.c_uint64
struct_ibv_alloc_dm_attr.SIZE = 16
struct_ibv_alloc_dm_attr._fields_ = ['length', 'log_align_req', 'comp_mask']
setattr(struct_ibv_alloc_dm_attr, 'length', field(0, size_t))
setattr(struct_ibv_alloc_dm_attr, 'log_align_req', field(8, uint32_t))
setattr(struct_ibv_alloc_dm_attr, 'comp_mask', field(12, uint32_t))
enum_ibv_dm_mask = CEnum(ctypes.c_uint32)
IBV_DM_MASK_HANDLE = enum_ibv_dm_mask.define('IBV_DM_MASK_HANDLE', 1)

class struct_ibv_dm(Struct): pass
class struct_ibv_context(Struct): pass
class struct_ibv_device(Struct): pass
class struct__ibv_device_ops(Struct): pass
struct__ibv_device_ops.SIZE = 16
struct__ibv_device_ops._fields_ = ['_dummy1', '_dummy2']
setattr(struct__ibv_device_ops, '_dummy1', field(0, ctypes.CFUNCTYPE(Pointer(struct_ibv_context), Pointer(struct_ibv_device), ctypes.c_int32)))
setattr(struct__ibv_device_ops, '_dummy2', field(8, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_context))))
struct_ibv_device.SIZE = 664
struct_ibv_device._fields_ = ['_ops', 'node_type', 'transport_type', 'name', 'dev_name', 'dev_path', 'ibdev_path']
setattr(struct_ibv_device, '_ops', field(0, struct__ibv_device_ops))
setattr(struct_ibv_device, 'node_type', field(16, enum_ibv_node_type))
setattr(struct_ibv_device, 'transport_type', field(20, enum_ibv_transport_type))
setattr(struct_ibv_device, 'name', field(24, Array(ctypes.c_char, 64)))
setattr(struct_ibv_device, 'dev_name', field(88, Array(ctypes.c_char, 64)))
setattr(struct_ibv_device, 'dev_path', field(152, Array(ctypes.c_char, 256)))
setattr(struct_ibv_device, 'ibdev_path', field(408, Array(ctypes.c_char, 256)))
class struct_ibv_context_ops(Struct): pass
class struct_ibv_device_attr(Struct): pass
uint64_t = ctypes.c_uint64
uint16_t = ctypes.c_uint16
struct_ibv_device_attr.SIZE = 232
struct_ibv_device_attr._fields_ = ['fw_ver', 'node_guid', 'sys_image_guid', 'max_mr_size', 'page_size_cap', 'vendor_id', 'vendor_part_id', 'hw_ver', 'max_qp', 'max_qp_wr', 'device_cap_flags', 'max_sge', 'max_sge_rd', 'max_cq', 'max_cqe', 'max_mr', 'max_pd', 'max_qp_rd_atom', 'max_ee_rd_atom', 'max_res_rd_atom', 'max_qp_init_rd_atom', 'max_ee_init_rd_atom', 'atomic_cap', 'max_ee', 'max_rdd', 'max_mw', 'max_raw_ipv6_qp', 'max_raw_ethy_qp', 'max_mcast_grp', 'max_mcast_qp_attach', 'max_total_mcast_qp_attach', 'max_ah', 'max_fmr', 'max_map_per_fmr', 'max_srq', 'max_srq_wr', 'max_srq_sge', 'max_pkeys', 'local_ca_ack_delay', 'phys_port_cnt']
setattr(struct_ibv_device_attr, 'fw_ver', field(0, Array(ctypes.c_char, 64)))
setattr(struct_ibv_device_attr, 'node_guid', field(64, ctypes.c_uint64))
setattr(struct_ibv_device_attr, 'sys_image_guid', field(72, ctypes.c_uint64))
setattr(struct_ibv_device_attr, 'max_mr_size', field(80, uint64_t))
setattr(struct_ibv_device_attr, 'page_size_cap', field(88, uint64_t))
setattr(struct_ibv_device_attr, 'vendor_id', field(96, uint32_t))
setattr(struct_ibv_device_attr, 'vendor_part_id', field(100, uint32_t))
setattr(struct_ibv_device_attr, 'hw_ver', field(104, uint32_t))
setattr(struct_ibv_device_attr, 'max_qp', field(108, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_qp_wr', field(112, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'device_cap_flags', field(116, ctypes.c_uint32))
setattr(struct_ibv_device_attr, 'max_sge', field(120, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_sge_rd', field(124, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_cq', field(128, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_cqe', field(132, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_mr', field(136, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_pd', field(140, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_qp_rd_atom', field(144, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_ee_rd_atom', field(148, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_res_rd_atom', field(152, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_qp_init_rd_atom', field(156, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_ee_init_rd_atom', field(160, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'atomic_cap', field(164, enum_ibv_atomic_cap))
setattr(struct_ibv_device_attr, 'max_ee', field(168, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_rdd', field(172, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_mw', field(176, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_raw_ipv6_qp', field(180, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_raw_ethy_qp', field(184, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_mcast_grp', field(188, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_mcast_qp_attach', field(192, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_total_mcast_qp_attach', field(196, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_ah', field(200, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_fmr', field(204, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_map_per_fmr', field(208, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_srq', field(212, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_srq_wr', field(216, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_srq_sge', field(220, ctypes.c_int32))
setattr(struct_ibv_device_attr, 'max_pkeys', field(224, uint16_t))
setattr(struct_ibv_device_attr, 'local_ca_ack_delay', field(226, uint8_t))
setattr(struct_ibv_device_attr, 'phys_port_cnt', field(227, uint8_t))
class struct__compat_ibv_port_attr(Struct): pass
class struct_ibv_mw(Struct): pass
class struct_ibv_pd(Struct): pass
struct_ibv_pd.SIZE = 16
struct_ibv_pd._fields_ = ['context', 'handle']
setattr(struct_ibv_pd, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_pd, 'handle', field(8, uint32_t))
enum_ibv_mw_type = CEnum(ctypes.c_uint32)
IBV_MW_TYPE_1 = enum_ibv_mw_type.define('IBV_MW_TYPE_1', 1)
IBV_MW_TYPE_2 = enum_ibv_mw_type.define('IBV_MW_TYPE_2', 2)

struct_ibv_mw.SIZE = 32
struct_ibv_mw._fields_ = ['context', 'pd', 'rkey', 'handle', 'type']
setattr(struct_ibv_mw, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_mw, 'pd', field(8, Pointer(struct_ibv_pd)))
setattr(struct_ibv_mw, 'rkey', field(16, uint32_t))
setattr(struct_ibv_mw, 'handle', field(20, uint32_t))
setattr(struct_ibv_mw, 'type', field(24, enum_ibv_mw_type))
class struct_ibv_qp(Struct): pass
class struct_ibv_cq(Struct): pass
class struct_ibv_comp_channel(Struct): pass
struct_ibv_comp_channel.SIZE = 16
struct_ibv_comp_channel._fields_ = ['context', 'fd', 'refcnt']
setattr(struct_ibv_comp_channel, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_comp_channel, 'fd', field(8, ctypes.c_int32))
setattr(struct_ibv_comp_channel, 'refcnt', field(12, ctypes.c_int32))
class pthread_mutex_t(Union): pass
class struct___pthread_mutex_s(Struct): pass
class struct___pthread_internal_list(Struct): pass
__pthread_list_t = struct___pthread_internal_list
struct___pthread_internal_list.SIZE = 16
struct___pthread_internal_list._fields_ = ['__prev', '__next']
setattr(struct___pthread_internal_list, '__prev', field(0, Pointer(struct___pthread_internal_list)))
setattr(struct___pthread_internal_list, '__next', field(8, Pointer(struct___pthread_internal_list)))
struct___pthread_mutex_s.SIZE = 40
struct___pthread_mutex_s._fields_ = ['__lock', '__count', '__owner', '__nusers', '__kind', '__spins', '__elision', '__list']
setattr(struct___pthread_mutex_s, '__lock', field(0, ctypes.c_int32))
setattr(struct___pthread_mutex_s, '__count', field(4, ctypes.c_uint32))
setattr(struct___pthread_mutex_s, '__owner', field(8, ctypes.c_int32))
setattr(struct___pthread_mutex_s, '__nusers', field(12, ctypes.c_uint32))
setattr(struct___pthread_mutex_s, '__kind', field(16, ctypes.c_int32))
setattr(struct___pthread_mutex_s, '__spins', field(20, ctypes.c_int16))
setattr(struct___pthread_mutex_s, '__elision', field(22, ctypes.c_int16))
setattr(struct___pthread_mutex_s, '__list', field(24, struct___pthread_internal_list))
pthread_mutex_t.SIZE = 40
pthread_mutex_t._fields_ = ['__data', '__size', '__align']
setattr(pthread_mutex_t, '__data', field(0, struct___pthread_mutex_s))
setattr(pthread_mutex_t, '__size', field(0, Array(ctypes.c_char, 40)))
setattr(pthread_mutex_t, '__align', field(0, ctypes.c_int64))
class pthread_cond_t(Union): pass
class struct___pthread_cond_s(Struct): pass
class __atomic_wide_counter(Union): pass
class _anonstruct1(Struct): pass
_anonstruct1.SIZE = 8
_anonstruct1._fields_ = ['__low', '__high']
setattr(_anonstruct1, '__low', field(0, ctypes.c_uint32))
setattr(_anonstruct1, '__high', field(4, ctypes.c_uint32))
__atomic_wide_counter.SIZE = 8
__atomic_wide_counter._fields_ = ['__value64', '__value32']
setattr(__atomic_wide_counter, '__value64', field(0, ctypes.c_uint64))
setattr(__atomic_wide_counter, '__value32', field(0, _anonstruct1))
struct___pthread_cond_s.SIZE = 48
struct___pthread_cond_s._fields_ = ['__wseq', '__g1_start', '__g_refs', '__g_size', '__g1_orig_size', '__wrefs', '__g_signals']
setattr(struct___pthread_cond_s, '__wseq', field(0, __atomic_wide_counter))
setattr(struct___pthread_cond_s, '__g1_start', field(8, __atomic_wide_counter))
setattr(struct___pthread_cond_s, '__g_refs', field(16, Array(ctypes.c_uint32, 2)))
setattr(struct___pthread_cond_s, '__g_size', field(24, Array(ctypes.c_uint32, 2)))
setattr(struct___pthread_cond_s, '__g1_orig_size', field(32, ctypes.c_uint32))
setattr(struct___pthread_cond_s, '__wrefs', field(36, ctypes.c_uint32))
setattr(struct___pthread_cond_s, '__g_signals', field(40, Array(ctypes.c_uint32, 2)))
pthread_cond_t.SIZE = 48
pthread_cond_t._fields_ = ['__data', '__size', '__align']
setattr(pthread_cond_t, '__data', field(0, struct___pthread_cond_s))
setattr(pthread_cond_t, '__size', field(0, Array(ctypes.c_char, 48)))
setattr(pthread_cond_t, '__align', field(0, ctypes.c_int64))
struct_ibv_cq.SIZE = 128
struct_ibv_cq._fields_ = ['context', 'channel', 'cq_context', 'handle', 'cqe', 'mutex', 'cond', 'comp_events_completed', 'async_events_completed']
setattr(struct_ibv_cq, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_cq, 'channel', field(8, Pointer(struct_ibv_comp_channel)))
setattr(struct_ibv_cq, 'cq_context', field(16, ctypes.c_void_p))
setattr(struct_ibv_cq, 'handle', field(24, uint32_t))
setattr(struct_ibv_cq, 'cqe', field(28, ctypes.c_int32))
setattr(struct_ibv_cq, 'mutex', field(32, pthread_mutex_t))
setattr(struct_ibv_cq, 'cond', field(72, pthread_cond_t))
setattr(struct_ibv_cq, 'comp_events_completed', field(120, uint32_t))
setattr(struct_ibv_cq, 'async_events_completed', field(124, uint32_t))
class struct_ibv_srq(Struct): pass
struct_ibv_srq.SIZE = 128
struct_ibv_srq._fields_ = ['context', 'srq_context', 'pd', 'handle', 'mutex', 'cond', 'events_completed']
setattr(struct_ibv_srq, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_srq, 'srq_context', field(8, ctypes.c_void_p))
setattr(struct_ibv_srq, 'pd', field(16, Pointer(struct_ibv_pd)))
setattr(struct_ibv_srq, 'handle', field(24, uint32_t))
setattr(struct_ibv_srq, 'mutex', field(32, pthread_mutex_t))
setattr(struct_ibv_srq, 'cond', field(72, pthread_cond_t))
setattr(struct_ibv_srq, 'events_completed', field(120, uint32_t))
enum_ibv_qp_state = CEnum(ctypes.c_uint32)
IBV_QPS_RESET = enum_ibv_qp_state.define('IBV_QPS_RESET', 0)
IBV_QPS_INIT = enum_ibv_qp_state.define('IBV_QPS_INIT', 1)
IBV_QPS_RTR = enum_ibv_qp_state.define('IBV_QPS_RTR', 2)
IBV_QPS_RTS = enum_ibv_qp_state.define('IBV_QPS_RTS', 3)
IBV_QPS_SQD = enum_ibv_qp_state.define('IBV_QPS_SQD', 4)
IBV_QPS_SQE = enum_ibv_qp_state.define('IBV_QPS_SQE', 5)
IBV_QPS_ERR = enum_ibv_qp_state.define('IBV_QPS_ERR', 6)
IBV_QPS_UNKNOWN = enum_ibv_qp_state.define('IBV_QPS_UNKNOWN', 7)

enum_ibv_qp_type = CEnum(ctypes.c_uint32)
IBV_QPT_RC = enum_ibv_qp_type.define('IBV_QPT_RC', 2)
IBV_QPT_UC = enum_ibv_qp_type.define('IBV_QPT_UC', 3)
IBV_QPT_UD = enum_ibv_qp_type.define('IBV_QPT_UD', 4)
IBV_QPT_RAW_PACKET = enum_ibv_qp_type.define('IBV_QPT_RAW_PACKET', 8)
IBV_QPT_XRC_SEND = enum_ibv_qp_type.define('IBV_QPT_XRC_SEND', 9)
IBV_QPT_XRC_RECV = enum_ibv_qp_type.define('IBV_QPT_XRC_RECV', 10)
IBV_QPT_DRIVER = enum_ibv_qp_type.define('IBV_QPT_DRIVER', 255)

struct_ibv_qp.SIZE = 160
struct_ibv_qp._fields_ = ['context', 'qp_context', 'pd', 'send_cq', 'recv_cq', 'srq', 'handle', 'qp_num', 'state', 'qp_type', 'mutex', 'cond', 'events_completed']
setattr(struct_ibv_qp, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_qp, 'qp_context', field(8, ctypes.c_void_p))
setattr(struct_ibv_qp, 'pd', field(16, Pointer(struct_ibv_pd)))
setattr(struct_ibv_qp, 'send_cq', field(24, Pointer(struct_ibv_cq)))
setattr(struct_ibv_qp, 'recv_cq', field(32, Pointer(struct_ibv_cq)))
setattr(struct_ibv_qp, 'srq', field(40, Pointer(struct_ibv_srq)))
setattr(struct_ibv_qp, 'handle', field(48, uint32_t))
setattr(struct_ibv_qp, 'qp_num', field(52, uint32_t))
setattr(struct_ibv_qp, 'state', field(56, enum_ibv_qp_state))
setattr(struct_ibv_qp, 'qp_type', field(60, enum_ibv_qp_type))
setattr(struct_ibv_qp, 'mutex', field(64, pthread_mutex_t))
setattr(struct_ibv_qp, 'cond', field(104, pthread_cond_t))
setattr(struct_ibv_qp, 'events_completed', field(152, uint32_t))
class struct_ibv_mw_bind(Struct): pass
class struct_ibv_mw_bind_info(Struct): pass
class struct_ibv_mr(Struct): pass
struct_ibv_mr.SIZE = 48
struct_ibv_mr._fields_ = ['context', 'pd', 'addr', 'length', 'handle', 'lkey', 'rkey']
setattr(struct_ibv_mr, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_mr, 'pd', field(8, Pointer(struct_ibv_pd)))
setattr(struct_ibv_mr, 'addr', field(16, ctypes.c_void_p))
setattr(struct_ibv_mr, 'length', field(24, size_t))
setattr(struct_ibv_mr, 'handle', field(32, uint32_t))
setattr(struct_ibv_mr, 'lkey', field(36, uint32_t))
setattr(struct_ibv_mr, 'rkey', field(40, uint32_t))
struct_ibv_mw_bind_info.SIZE = 32
struct_ibv_mw_bind_info._fields_ = ['mr', 'addr', 'length', 'mw_access_flags']
setattr(struct_ibv_mw_bind_info, 'mr', field(0, Pointer(struct_ibv_mr)))
setattr(struct_ibv_mw_bind_info, 'addr', field(8, uint64_t))
setattr(struct_ibv_mw_bind_info, 'length', field(16, uint64_t))
setattr(struct_ibv_mw_bind_info, 'mw_access_flags', field(24, ctypes.c_uint32))
struct_ibv_mw_bind.SIZE = 48
struct_ibv_mw_bind._fields_ = ['wr_id', 'send_flags', 'bind_info']
setattr(struct_ibv_mw_bind, 'wr_id', field(0, uint64_t))
setattr(struct_ibv_mw_bind, 'send_flags', field(8, ctypes.c_uint32))
setattr(struct_ibv_mw_bind, 'bind_info', field(16, struct_ibv_mw_bind_info))
class struct_ibv_wc(Struct): pass
enum_ibv_wc_status = CEnum(ctypes.c_uint32)
IBV_WC_SUCCESS = enum_ibv_wc_status.define('IBV_WC_SUCCESS', 0)
IBV_WC_LOC_LEN_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_LEN_ERR', 1)
IBV_WC_LOC_QP_OP_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_QP_OP_ERR', 2)
IBV_WC_LOC_EEC_OP_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_EEC_OP_ERR', 3)
IBV_WC_LOC_PROT_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_PROT_ERR', 4)
IBV_WC_WR_FLUSH_ERR = enum_ibv_wc_status.define('IBV_WC_WR_FLUSH_ERR', 5)
IBV_WC_MW_BIND_ERR = enum_ibv_wc_status.define('IBV_WC_MW_BIND_ERR', 6)
IBV_WC_BAD_RESP_ERR = enum_ibv_wc_status.define('IBV_WC_BAD_RESP_ERR', 7)
IBV_WC_LOC_ACCESS_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_ACCESS_ERR', 8)
IBV_WC_REM_INV_REQ_ERR = enum_ibv_wc_status.define('IBV_WC_REM_INV_REQ_ERR', 9)
IBV_WC_REM_ACCESS_ERR = enum_ibv_wc_status.define('IBV_WC_REM_ACCESS_ERR', 10)
IBV_WC_REM_OP_ERR = enum_ibv_wc_status.define('IBV_WC_REM_OP_ERR', 11)
IBV_WC_RETRY_EXC_ERR = enum_ibv_wc_status.define('IBV_WC_RETRY_EXC_ERR', 12)
IBV_WC_RNR_RETRY_EXC_ERR = enum_ibv_wc_status.define('IBV_WC_RNR_RETRY_EXC_ERR', 13)
IBV_WC_LOC_RDD_VIOL_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_RDD_VIOL_ERR', 14)
IBV_WC_REM_INV_RD_REQ_ERR = enum_ibv_wc_status.define('IBV_WC_REM_INV_RD_REQ_ERR', 15)
IBV_WC_REM_ABORT_ERR = enum_ibv_wc_status.define('IBV_WC_REM_ABORT_ERR', 16)
IBV_WC_INV_EECN_ERR = enum_ibv_wc_status.define('IBV_WC_INV_EECN_ERR', 17)
IBV_WC_INV_EEC_STATE_ERR = enum_ibv_wc_status.define('IBV_WC_INV_EEC_STATE_ERR', 18)
IBV_WC_FATAL_ERR = enum_ibv_wc_status.define('IBV_WC_FATAL_ERR', 19)
IBV_WC_RESP_TIMEOUT_ERR = enum_ibv_wc_status.define('IBV_WC_RESP_TIMEOUT_ERR', 20)
IBV_WC_GENERAL_ERR = enum_ibv_wc_status.define('IBV_WC_GENERAL_ERR', 21)
IBV_WC_TM_ERR = enum_ibv_wc_status.define('IBV_WC_TM_ERR', 22)
IBV_WC_TM_RNDV_INCOMPLETE = enum_ibv_wc_status.define('IBV_WC_TM_RNDV_INCOMPLETE', 23)

enum_ibv_wc_opcode = CEnum(ctypes.c_uint32)
IBV_WC_SEND = enum_ibv_wc_opcode.define('IBV_WC_SEND', 0)
IBV_WC_RDMA_WRITE = enum_ibv_wc_opcode.define('IBV_WC_RDMA_WRITE', 1)
IBV_WC_RDMA_READ = enum_ibv_wc_opcode.define('IBV_WC_RDMA_READ', 2)
IBV_WC_COMP_SWAP = enum_ibv_wc_opcode.define('IBV_WC_COMP_SWAP', 3)
IBV_WC_FETCH_ADD = enum_ibv_wc_opcode.define('IBV_WC_FETCH_ADD', 4)
IBV_WC_BIND_MW = enum_ibv_wc_opcode.define('IBV_WC_BIND_MW', 5)
IBV_WC_LOCAL_INV = enum_ibv_wc_opcode.define('IBV_WC_LOCAL_INV', 6)
IBV_WC_TSO = enum_ibv_wc_opcode.define('IBV_WC_TSO', 7)
IBV_WC_FLUSH = enum_ibv_wc_opcode.define('IBV_WC_FLUSH', 8)
IBV_WC_ATOMIC_WRITE = enum_ibv_wc_opcode.define('IBV_WC_ATOMIC_WRITE', 9)
IBV_WC_RECV = enum_ibv_wc_opcode.define('IBV_WC_RECV', 128)
IBV_WC_RECV_RDMA_WITH_IMM = enum_ibv_wc_opcode.define('IBV_WC_RECV_RDMA_WITH_IMM', 129)
IBV_WC_TM_ADD = enum_ibv_wc_opcode.define('IBV_WC_TM_ADD', 130)
IBV_WC_TM_DEL = enum_ibv_wc_opcode.define('IBV_WC_TM_DEL', 131)
IBV_WC_TM_SYNC = enum_ibv_wc_opcode.define('IBV_WC_TM_SYNC', 132)
IBV_WC_TM_RECV = enum_ibv_wc_opcode.define('IBV_WC_TM_RECV', 133)
IBV_WC_TM_NO_TAG = enum_ibv_wc_opcode.define('IBV_WC_TM_NO_TAG', 134)
IBV_WC_DRIVER1 = enum_ibv_wc_opcode.define('IBV_WC_DRIVER1', 135)
IBV_WC_DRIVER2 = enum_ibv_wc_opcode.define('IBV_WC_DRIVER2', 136)
IBV_WC_DRIVER3 = enum_ibv_wc_opcode.define('IBV_WC_DRIVER3', 137)

__be32 = ctypes.c_uint32
struct_ibv_wc.SIZE = 48
struct_ibv_wc._fields_ = ['wr_id', 'status', 'opcode', 'vendor_err', 'byte_len', 'imm_data', 'invalidated_rkey', 'qp_num', 'src_qp', 'wc_flags', 'pkey_index', 'slid', 'sl', 'dlid_path_bits']
setattr(struct_ibv_wc, 'wr_id', field(0, uint64_t))
setattr(struct_ibv_wc, 'status', field(8, enum_ibv_wc_status))
setattr(struct_ibv_wc, 'opcode', field(12, enum_ibv_wc_opcode))
setattr(struct_ibv_wc, 'vendor_err', field(16, uint32_t))
setattr(struct_ibv_wc, 'byte_len', field(20, uint32_t))
setattr(struct_ibv_wc, 'imm_data', field(24, ctypes.c_uint32))
setattr(struct_ibv_wc, 'invalidated_rkey', field(24, uint32_t))
setattr(struct_ibv_wc, 'qp_num', field(28, uint32_t))
setattr(struct_ibv_wc, 'src_qp', field(32, uint32_t))
setattr(struct_ibv_wc, 'wc_flags', field(36, ctypes.c_uint32))
setattr(struct_ibv_wc, 'pkey_index', field(40, uint16_t))
setattr(struct_ibv_wc, 'slid', field(42, uint16_t))
setattr(struct_ibv_wc, 'sl', field(44, uint8_t))
setattr(struct_ibv_wc, 'dlid_path_bits', field(45, uint8_t))
class struct_ibv_recv_wr(Struct): pass
class struct_ibv_sge(Struct): pass
struct_ibv_sge.SIZE = 16
struct_ibv_sge._fields_ = ['addr', 'length', 'lkey']
setattr(struct_ibv_sge, 'addr', field(0, uint64_t))
setattr(struct_ibv_sge, 'length', field(8, uint32_t))
setattr(struct_ibv_sge, 'lkey', field(12, uint32_t))
struct_ibv_recv_wr.SIZE = 32
struct_ibv_recv_wr._fields_ = ['wr_id', 'next', 'sg_list', 'num_sge']
setattr(struct_ibv_recv_wr, 'wr_id', field(0, uint64_t))
setattr(struct_ibv_recv_wr, 'next', field(8, Pointer(struct_ibv_recv_wr)))
setattr(struct_ibv_recv_wr, 'sg_list', field(16, Pointer(struct_ibv_sge)))
setattr(struct_ibv_recv_wr, 'num_sge', field(24, ctypes.c_int32))
class struct_ibv_send_wr(Struct): pass
enum_ibv_wr_opcode = CEnum(ctypes.c_uint32)
IBV_WR_RDMA_WRITE = enum_ibv_wr_opcode.define('IBV_WR_RDMA_WRITE', 0)
IBV_WR_RDMA_WRITE_WITH_IMM = enum_ibv_wr_opcode.define('IBV_WR_RDMA_WRITE_WITH_IMM', 1)
IBV_WR_SEND = enum_ibv_wr_opcode.define('IBV_WR_SEND', 2)
IBV_WR_SEND_WITH_IMM = enum_ibv_wr_opcode.define('IBV_WR_SEND_WITH_IMM', 3)
IBV_WR_RDMA_READ = enum_ibv_wr_opcode.define('IBV_WR_RDMA_READ', 4)
IBV_WR_ATOMIC_CMP_AND_SWP = enum_ibv_wr_opcode.define('IBV_WR_ATOMIC_CMP_AND_SWP', 5)
IBV_WR_ATOMIC_FETCH_AND_ADD = enum_ibv_wr_opcode.define('IBV_WR_ATOMIC_FETCH_AND_ADD', 6)
IBV_WR_LOCAL_INV = enum_ibv_wr_opcode.define('IBV_WR_LOCAL_INV', 7)
IBV_WR_BIND_MW = enum_ibv_wr_opcode.define('IBV_WR_BIND_MW', 8)
IBV_WR_SEND_WITH_INV = enum_ibv_wr_opcode.define('IBV_WR_SEND_WITH_INV', 9)
IBV_WR_TSO = enum_ibv_wr_opcode.define('IBV_WR_TSO', 10)
IBV_WR_DRIVER1 = enum_ibv_wr_opcode.define('IBV_WR_DRIVER1', 11)
IBV_WR_FLUSH = enum_ibv_wr_opcode.define('IBV_WR_FLUSH', 14)
IBV_WR_ATOMIC_WRITE = enum_ibv_wr_opcode.define('IBV_WR_ATOMIC_WRITE', 15)

class _anonunion2(Union): pass
class _anonstruct3(Struct): pass
_anonstruct3.SIZE = 16
_anonstruct3._fields_ = ['remote_addr', 'rkey']
setattr(_anonstruct3, 'remote_addr', field(0, uint64_t))
setattr(_anonstruct3, 'rkey', field(8, uint32_t))
class _anonstruct4(Struct): pass
_anonstruct4.SIZE = 32
_anonstruct4._fields_ = ['remote_addr', 'compare_add', 'swap', 'rkey']
setattr(_anonstruct4, 'remote_addr', field(0, uint64_t))
setattr(_anonstruct4, 'compare_add', field(8, uint64_t))
setattr(_anonstruct4, 'swap', field(16, uint64_t))
setattr(_anonstruct4, 'rkey', field(24, uint32_t))
class _anonstruct5(Struct): pass
class struct_ibv_ah(Struct): pass
struct_ibv_ah.SIZE = 24
struct_ibv_ah._fields_ = ['context', 'pd', 'handle']
setattr(struct_ibv_ah, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_ah, 'pd', field(8, Pointer(struct_ibv_pd)))
setattr(struct_ibv_ah, 'handle', field(16, uint32_t))
_anonstruct5.SIZE = 16
_anonstruct5._fields_ = ['ah', 'remote_qpn', 'remote_qkey']
setattr(_anonstruct5, 'ah', field(0, Pointer(struct_ibv_ah)))
setattr(_anonstruct5, 'remote_qpn', field(8, uint32_t))
setattr(_anonstruct5, 'remote_qkey', field(12, uint32_t))
_anonunion2.SIZE = 32
_anonunion2._fields_ = ['rdma', 'atomic', 'ud']
setattr(_anonunion2, 'rdma', field(0, _anonstruct3))
setattr(_anonunion2, 'atomic', field(0, _anonstruct4))
setattr(_anonunion2, 'ud', field(0, _anonstruct5))
class _anonunion6(Union): pass
class _anonstruct7(Struct): pass
_anonstruct7.SIZE = 4
_anonstruct7._fields_ = ['remote_srqn']
setattr(_anonstruct7, 'remote_srqn', field(0, uint32_t))
_anonunion6.SIZE = 4
_anonunion6._fields_ = ['xrc']
setattr(_anonunion6, 'xrc', field(0, _anonstruct7))
class _anonstruct8(Struct): pass
_anonstruct8.SIZE = 48
_anonstruct8._fields_ = ['mw', 'rkey', 'bind_info']
setattr(_anonstruct8, 'mw', field(0, Pointer(struct_ibv_mw)))
setattr(_anonstruct8, 'rkey', field(8, uint32_t))
setattr(_anonstruct8, 'bind_info', field(16, struct_ibv_mw_bind_info))
class _anonstruct9(Struct): pass
_anonstruct9.SIZE = 16
_anonstruct9._fields_ = ['hdr', 'hdr_sz', 'mss']
setattr(_anonstruct9, 'hdr', field(0, ctypes.c_void_p))
setattr(_anonstruct9, 'hdr_sz', field(8, uint16_t))
setattr(_anonstruct9, 'mss', field(10, uint16_t))
struct_ibv_send_wr.SIZE = 128
struct_ibv_send_wr._fields_ = ['wr_id', 'next', 'sg_list', 'num_sge', 'opcode', 'send_flags', 'imm_data', 'invalidate_rkey', 'wr', 'qp_type', 'bind_mw', 'tso']
setattr(struct_ibv_send_wr, 'wr_id', field(0, uint64_t))
setattr(struct_ibv_send_wr, 'next', field(8, Pointer(struct_ibv_send_wr)))
setattr(struct_ibv_send_wr, 'sg_list', field(16, Pointer(struct_ibv_sge)))
setattr(struct_ibv_send_wr, 'num_sge', field(24, ctypes.c_int32))
setattr(struct_ibv_send_wr, 'opcode', field(28, enum_ibv_wr_opcode))
setattr(struct_ibv_send_wr, 'send_flags', field(32, ctypes.c_uint32))
setattr(struct_ibv_send_wr, 'imm_data', field(36, ctypes.c_uint32))
setattr(struct_ibv_send_wr, 'invalidate_rkey', field(36, uint32_t))
setattr(struct_ibv_send_wr, 'wr', field(40, _anonunion2))
setattr(struct_ibv_send_wr, 'qp_type', field(72, _anonunion6))
setattr(struct_ibv_send_wr, 'bind_mw', field(80, _anonstruct8))
setattr(struct_ibv_send_wr, 'tso', field(80, _anonstruct9))
struct_ibv_context_ops.SIZE = 256
struct_ibv_context_ops._fields_ = ['_compat_query_device', '_compat_query_port', '_compat_alloc_pd', '_compat_dealloc_pd', '_compat_reg_mr', '_compat_rereg_mr', '_compat_dereg_mr', 'alloc_mw', 'bind_mw', 'dealloc_mw', '_compat_create_cq', 'poll_cq', 'req_notify_cq', '_compat_cq_event', '_compat_resize_cq', '_compat_destroy_cq', '_compat_create_srq', '_compat_modify_srq', '_compat_query_srq', '_compat_destroy_srq', 'post_srq_recv', '_compat_create_qp', '_compat_query_qp', '_compat_modify_qp', '_compat_destroy_qp', 'post_send', 'post_recv', '_compat_create_ah', '_compat_destroy_ah', '_compat_attach_mcast', '_compat_detach_mcast', '_compat_async_event']
setattr(struct_ibv_context_ops, '_compat_query_device', field(0, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_context), Pointer(struct_ibv_device_attr))))
setattr(struct_ibv_context_ops, '_compat_query_port', field(8, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_context), uint8_t, Pointer(struct__compat_ibv_port_attr))))
setattr(struct_ibv_context_ops, '_compat_alloc_pd', field(16, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_dealloc_pd', field(24, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_reg_mr', field(32, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_rereg_mr', field(40, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_dereg_mr', field(48, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, 'alloc_mw', field(56, ctypes.CFUNCTYPE(Pointer(struct_ibv_mw), Pointer(struct_ibv_pd), enum_ibv_mw_type)))
setattr(struct_ibv_context_ops, 'bind_mw', field(64, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_qp), Pointer(struct_ibv_mw), Pointer(struct_ibv_mw_bind))))
setattr(struct_ibv_context_ops, 'dealloc_mw', field(72, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_mw))))
setattr(struct_ibv_context_ops, '_compat_create_cq', field(80, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, 'poll_cq', field(88, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_cq), ctypes.c_int32, Pointer(struct_ibv_wc))))
setattr(struct_ibv_context_ops, 'req_notify_cq', field(96, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_cq), ctypes.c_int32)))
setattr(struct_ibv_context_ops, '_compat_cq_event', field(104, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_resize_cq', field(112, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_destroy_cq', field(120, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_create_srq', field(128, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_modify_srq', field(136, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_query_srq', field(144, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_destroy_srq', field(152, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, 'post_srq_recv', field(160, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_srq), Pointer(struct_ibv_recv_wr), Pointer(Pointer(struct_ibv_recv_wr)))))
setattr(struct_ibv_context_ops, '_compat_create_qp', field(168, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_query_qp', field(176, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_modify_qp', field(184, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_destroy_qp', field(192, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, 'post_send', field(200, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_qp), Pointer(struct_ibv_send_wr), Pointer(Pointer(struct_ibv_send_wr)))))
setattr(struct_ibv_context_ops, 'post_recv', field(208, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_qp), Pointer(struct_ibv_recv_wr), Pointer(Pointer(struct_ibv_recv_wr)))))
setattr(struct_ibv_context_ops, '_compat_create_ah', field(216, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_destroy_ah', field(224, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_attach_mcast', field(232, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_detach_mcast', field(240, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
setattr(struct_ibv_context_ops, '_compat_async_event', field(248, ctypes.CFUNCTYPE(ctypes.c_void_p, )))
struct_ibv_context.SIZE = 328
struct_ibv_context._fields_ = ['device', 'ops', 'cmd_fd', 'async_fd', 'num_comp_vectors', 'mutex', 'abi_compat']
setattr(struct_ibv_context, 'device', field(0, Pointer(struct_ibv_device)))
setattr(struct_ibv_context, 'ops', field(8, struct_ibv_context_ops))
setattr(struct_ibv_context, 'cmd_fd', field(264, ctypes.c_int32))
setattr(struct_ibv_context, 'async_fd', field(268, ctypes.c_int32))
setattr(struct_ibv_context, 'num_comp_vectors', field(272, ctypes.c_int32))
setattr(struct_ibv_context, 'mutex', field(280, pthread_mutex_t))
setattr(struct_ibv_context, 'abi_compat', field(320, ctypes.c_void_p))
struct_ibv_dm.SIZE = 32
struct_ibv_dm._fields_ = ['context', 'memcpy_to_dm', 'memcpy_from_dm', 'comp_mask', 'handle']
setattr(struct_ibv_dm, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_dm, 'memcpy_to_dm', field(8, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_dm), uint64_t, ctypes.c_void_p, size_t)))
setattr(struct_ibv_dm, 'memcpy_from_dm', field(16, ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, Pointer(struct_ibv_dm), uint64_t, size_t)))
setattr(struct_ibv_dm, 'comp_mask', field(24, uint32_t))
setattr(struct_ibv_dm, 'handle', field(28, uint32_t))
class struct_ibv_query_device_ex_input(Struct): pass
struct_ibv_query_device_ex_input.SIZE = 4
struct_ibv_query_device_ex_input._fields_ = ['comp_mask']
setattr(struct_ibv_query_device_ex_input, 'comp_mask', field(0, uint32_t))
enum_ibv_odp_transport_cap_bits = CEnum(ctypes.c_uint32)
IBV_ODP_SUPPORT_SEND = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_SEND', 1)
IBV_ODP_SUPPORT_RECV = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_RECV', 2)
IBV_ODP_SUPPORT_WRITE = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_WRITE', 4)
IBV_ODP_SUPPORT_READ = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_READ', 8)
IBV_ODP_SUPPORT_ATOMIC = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_ATOMIC', 16)
IBV_ODP_SUPPORT_SRQ_RECV = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_SRQ_RECV', 32)

class struct_ibv_odp_caps(Struct): pass
class _anonstruct10(Struct): pass
_anonstruct10.SIZE = 12
_anonstruct10._fields_ = ['rc_odp_caps', 'uc_odp_caps', 'ud_odp_caps']
setattr(_anonstruct10, 'rc_odp_caps', field(0, uint32_t))
setattr(_anonstruct10, 'uc_odp_caps', field(4, uint32_t))
setattr(_anonstruct10, 'ud_odp_caps', field(8, uint32_t))
struct_ibv_odp_caps.SIZE = 24
struct_ibv_odp_caps._fields_ = ['general_caps', 'per_transport_caps']
setattr(struct_ibv_odp_caps, 'general_caps', field(0, uint64_t))
setattr(struct_ibv_odp_caps, 'per_transport_caps', field(8, _anonstruct10))
enum_ibv_odp_general_caps = CEnum(ctypes.c_uint32)
IBV_ODP_SUPPORT = enum_ibv_odp_general_caps.define('IBV_ODP_SUPPORT', 1)
IBV_ODP_SUPPORT_IMPLICIT = enum_ibv_odp_general_caps.define('IBV_ODP_SUPPORT_IMPLICIT', 2)

class struct_ibv_tso_caps(Struct): pass
struct_ibv_tso_caps.SIZE = 8
struct_ibv_tso_caps._fields_ = ['max_tso', 'supported_qpts']
setattr(struct_ibv_tso_caps, 'max_tso', field(0, uint32_t))
setattr(struct_ibv_tso_caps, 'supported_qpts', field(4, uint32_t))
enum_ibv_rx_hash_function_flags = CEnum(ctypes.c_uint32)
IBV_RX_HASH_FUNC_TOEPLITZ = enum_ibv_rx_hash_function_flags.define('IBV_RX_HASH_FUNC_TOEPLITZ', 1)

enum_ibv_rx_hash_fields = CEnum(ctypes.c_uint32)
IBV_RX_HASH_SRC_IPV4 = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_SRC_IPV4', 1)
IBV_RX_HASH_DST_IPV4 = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_DST_IPV4', 2)
IBV_RX_HASH_SRC_IPV6 = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_SRC_IPV6', 4)
IBV_RX_HASH_DST_IPV6 = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_DST_IPV6', 8)
IBV_RX_HASH_SRC_PORT_TCP = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_SRC_PORT_TCP', 16)
IBV_RX_HASH_DST_PORT_TCP = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_DST_PORT_TCP', 32)
IBV_RX_HASH_SRC_PORT_UDP = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_SRC_PORT_UDP', 64)
IBV_RX_HASH_DST_PORT_UDP = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_DST_PORT_UDP', 128)
IBV_RX_HASH_IPSEC_SPI = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_IPSEC_SPI', 256)
IBV_RX_HASH_INNER = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_INNER', 2147483648)

class struct_ibv_rss_caps(Struct): pass
struct_ibv_rss_caps.SIZE = 32
struct_ibv_rss_caps._fields_ = ['supported_qpts', 'max_rwq_indirection_tables', 'max_rwq_indirection_table_size', 'rx_hash_fields_mask', 'rx_hash_function']
setattr(struct_ibv_rss_caps, 'supported_qpts', field(0, uint32_t))
setattr(struct_ibv_rss_caps, 'max_rwq_indirection_tables', field(4, uint32_t))
setattr(struct_ibv_rss_caps, 'max_rwq_indirection_table_size', field(8, uint32_t))
setattr(struct_ibv_rss_caps, 'rx_hash_fields_mask', field(16, uint64_t))
setattr(struct_ibv_rss_caps, 'rx_hash_function', field(24, uint8_t))
class struct_ibv_packet_pacing_caps(Struct): pass
struct_ibv_packet_pacing_caps.SIZE = 12
struct_ibv_packet_pacing_caps._fields_ = ['qp_rate_limit_min', 'qp_rate_limit_max', 'supported_qpts']
setattr(struct_ibv_packet_pacing_caps, 'qp_rate_limit_min', field(0, uint32_t))
setattr(struct_ibv_packet_pacing_caps, 'qp_rate_limit_max', field(4, uint32_t))
setattr(struct_ibv_packet_pacing_caps, 'supported_qpts', field(8, uint32_t))
enum_ibv_raw_packet_caps = CEnum(ctypes.c_uint32)
IBV_RAW_PACKET_CAP_CVLAN_STRIPPING = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_CVLAN_STRIPPING', 1)
IBV_RAW_PACKET_CAP_SCATTER_FCS = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_SCATTER_FCS', 2)
IBV_RAW_PACKET_CAP_IP_CSUM = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_IP_CSUM', 4)
IBV_RAW_PACKET_CAP_DELAY_DROP = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_DELAY_DROP', 8)

enum_ibv_tm_cap_flags = CEnum(ctypes.c_uint32)
IBV_TM_CAP_RC = enum_ibv_tm_cap_flags.define('IBV_TM_CAP_RC', 1)

class struct_ibv_tm_caps(Struct): pass
struct_ibv_tm_caps.SIZE = 20
struct_ibv_tm_caps._fields_ = ['max_rndv_hdr_size', 'max_num_tags', 'flags', 'max_ops', 'max_sge']
setattr(struct_ibv_tm_caps, 'max_rndv_hdr_size', field(0, uint32_t))
setattr(struct_ibv_tm_caps, 'max_num_tags', field(4, uint32_t))
setattr(struct_ibv_tm_caps, 'flags', field(8, uint32_t))
setattr(struct_ibv_tm_caps, 'max_ops', field(12, uint32_t))
setattr(struct_ibv_tm_caps, 'max_sge', field(16, uint32_t))
class struct_ibv_cq_moderation_caps(Struct): pass
struct_ibv_cq_moderation_caps.SIZE = 4
struct_ibv_cq_moderation_caps._fields_ = ['max_cq_count', 'max_cq_period']
setattr(struct_ibv_cq_moderation_caps, 'max_cq_count', field(0, uint16_t))
setattr(struct_ibv_cq_moderation_caps, 'max_cq_period', field(2, uint16_t))
enum_ibv_pci_atomic_op_size = CEnum(ctypes.c_uint32)
IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP', 1)
IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP', 2)
IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP', 4)

class struct_ibv_pci_atomic_caps(Struct): pass
struct_ibv_pci_atomic_caps.SIZE = 6
struct_ibv_pci_atomic_caps._fields_ = ['fetch_add', 'swap', 'compare_swap']
setattr(struct_ibv_pci_atomic_caps, 'fetch_add', field(0, uint16_t))
setattr(struct_ibv_pci_atomic_caps, 'swap', field(2, uint16_t))
setattr(struct_ibv_pci_atomic_caps, 'compare_swap', field(4, uint16_t))
class struct_ibv_device_attr_ex(Struct): pass
struct_ibv_device_attr_ex.SIZE = 400
struct_ibv_device_attr_ex._fields_ = ['orig_attr', 'comp_mask', 'odp_caps', 'completion_timestamp_mask', 'hca_core_clock', 'device_cap_flags_ex', 'tso_caps', 'rss_caps', 'max_wq_type_rq', 'packet_pacing_caps', 'raw_packet_caps', 'tm_caps', 'cq_mod_caps', 'max_dm_size', 'pci_atomic_caps', 'xrc_odp_caps', 'phys_port_cnt_ex']
setattr(struct_ibv_device_attr_ex, 'orig_attr', field(0, struct_ibv_device_attr))
setattr(struct_ibv_device_attr_ex, 'comp_mask', field(232, uint32_t))
setattr(struct_ibv_device_attr_ex, 'odp_caps', field(240, struct_ibv_odp_caps))
setattr(struct_ibv_device_attr_ex, 'completion_timestamp_mask', field(264, uint64_t))
setattr(struct_ibv_device_attr_ex, 'hca_core_clock', field(272, uint64_t))
setattr(struct_ibv_device_attr_ex, 'device_cap_flags_ex', field(280, uint64_t))
setattr(struct_ibv_device_attr_ex, 'tso_caps', field(288, struct_ibv_tso_caps))
setattr(struct_ibv_device_attr_ex, 'rss_caps', field(296, struct_ibv_rss_caps))
setattr(struct_ibv_device_attr_ex, 'max_wq_type_rq', field(328, uint32_t))
setattr(struct_ibv_device_attr_ex, 'packet_pacing_caps', field(332, struct_ibv_packet_pacing_caps))
setattr(struct_ibv_device_attr_ex, 'raw_packet_caps', field(344, uint32_t))
setattr(struct_ibv_device_attr_ex, 'tm_caps', field(348, struct_ibv_tm_caps))
setattr(struct_ibv_device_attr_ex, 'cq_mod_caps', field(368, struct_ibv_cq_moderation_caps))
setattr(struct_ibv_device_attr_ex, 'max_dm_size', field(376, uint64_t))
setattr(struct_ibv_device_attr_ex, 'pci_atomic_caps', field(384, struct_ibv_pci_atomic_caps))
setattr(struct_ibv_device_attr_ex, 'xrc_odp_caps', field(392, uint32_t))
setattr(struct_ibv_device_attr_ex, 'phys_port_cnt_ex', field(396, uint32_t))
enum_ibv_mtu = CEnum(ctypes.c_uint32)
IBV_MTU_256 = enum_ibv_mtu.define('IBV_MTU_256', 1)
IBV_MTU_512 = enum_ibv_mtu.define('IBV_MTU_512', 2)
IBV_MTU_1024 = enum_ibv_mtu.define('IBV_MTU_1024', 3)
IBV_MTU_2048 = enum_ibv_mtu.define('IBV_MTU_2048', 4)
IBV_MTU_4096 = enum_ibv_mtu.define('IBV_MTU_4096', 5)

enum_ibv_port_state = CEnum(ctypes.c_uint32)
IBV_PORT_NOP = enum_ibv_port_state.define('IBV_PORT_NOP', 0)
IBV_PORT_DOWN = enum_ibv_port_state.define('IBV_PORT_DOWN', 1)
IBV_PORT_INIT = enum_ibv_port_state.define('IBV_PORT_INIT', 2)
IBV_PORT_ARMED = enum_ibv_port_state.define('IBV_PORT_ARMED', 3)
IBV_PORT_ACTIVE = enum_ibv_port_state.define('IBV_PORT_ACTIVE', 4)
IBV_PORT_ACTIVE_DEFER = enum_ibv_port_state.define('IBV_PORT_ACTIVE_DEFER', 5)

_anonenum11 = CEnum(ctypes.c_uint32)
IBV_LINK_LAYER_UNSPECIFIED = _anonenum11.define('IBV_LINK_LAYER_UNSPECIFIED', 0)
IBV_LINK_LAYER_INFINIBAND = _anonenum11.define('IBV_LINK_LAYER_INFINIBAND', 1)
IBV_LINK_LAYER_ETHERNET = _anonenum11.define('IBV_LINK_LAYER_ETHERNET', 2)

enum_ibv_port_cap_flags = CEnum(ctypes.c_uint32)
IBV_PORT_SM = enum_ibv_port_cap_flags.define('IBV_PORT_SM', 2)
IBV_PORT_NOTICE_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_NOTICE_SUP', 4)
IBV_PORT_TRAP_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_TRAP_SUP', 8)
IBV_PORT_OPT_IPD_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_OPT_IPD_SUP', 16)
IBV_PORT_AUTO_MIGR_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_AUTO_MIGR_SUP', 32)
IBV_PORT_SL_MAP_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_SL_MAP_SUP', 64)
IBV_PORT_MKEY_NVRAM = enum_ibv_port_cap_flags.define('IBV_PORT_MKEY_NVRAM', 128)
IBV_PORT_PKEY_NVRAM = enum_ibv_port_cap_flags.define('IBV_PORT_PKEY_NVRAM', 256)
IBV_PORT_LED_INFO_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_LED_INFO_SUP', 512)
IBV_PORT_SYS_IMAGE_GUID_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_SYS_IMAGE_GUID_SUP', 2048)
IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP', 4096)
IBV_PORT_EXTENDED_SPEEDS_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_EXTENDED_SPEEDS_SUP', 16384)
IBV_PORT_CAP_MASK2_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_CAP_MASK2_SUP', 32768)
IBV_PORT_CM_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_CM_SUP', 65536)
IBV_PORT_SNMP_TUNNEL_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_SNMP_TUNNEL_SUP', 131072)
IBV_PORT_REINIT_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_REINIT_SUP', 262144)
IBV_PORT_DEVICE_MGMT_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_DEVICE_MGMT_SUP', 524288)
IBV_PORT_VENDOR_CLASS_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_VENDOR_CLASS_SUP', 1048576)
IBV_PORT_DR_NOTICE_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_DR_NOTICE_SUP', 2097152)
IBV_PORT_CAP_MASK_NOTICE_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_CAP_MASK_NOTICE_SUP', 4194304)
IBV_PORT_BOOT_MGMT_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_BOOT_MGMT_SUP', 8388608)
IBV_PORT_LINK_LATENCY_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_LINK_LATENCY_SUP', 16777216)
IBV_PORT_CLIENT_REG_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_CLIENT_REG_SUP', 33554432)
IBV_PORT_IP_BASED_GIDS = enum_ibv_port_cap_flags.define('IBV_PORT_IP_BASED_GIDS', 67108864)

enum_ibv_port_cap_flags2 = CEnum(ctypes.c_uint32)
IBV_PORT_SET_NODE_DESC_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_SET_NODE_DESC_SUP', 1)
IBV_PORT_INFO_EXT_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_INFO_EXT_SUP', 2)
IBV_PORT_VIRT_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_VIRT_SUP', 4)
IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP', 8)
IBV_PORT_LINK_WIDTH_2X_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_WIDTH_2X_SUP', 16)
IBV_PORT_LINK_SPEED_HDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_HDR_SUP', 32)
IBV_PORT_LINK_SPEED_NDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_NDR_SUP', 1024)
IBV_PORT_LINK_SPEED_XDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_XDR_SUP', 4096)

class struct_ibv_port_attr(Struct): pass
struct_ibv_port_attr.SIZE = 56
struct_ibv_port_attr._fields_ = ['state', 'max_mtu', 'active_mtu', 'gid_tbl_len', 'port_cap_flags', 'max_msg_sz', 'bad_pkey_cntr', 'qkey_viol_cntr', 'pkey_tbl_len', 'lid', 'sm_lid', 'lmc', 'max_vl_num', 'sm_sl', 'subnet_timeout', 'init_type_reply', 'active_width', 'active_speed', 'phys_state', 'link_layer', 'flags', 'port_cap_flags2', 'active_speed_ex']
setattr(struct_ibv_port_attr, 'state', field(0, enum_ibv_port_state))
setattr(struct_ibv_port_attr, 'max_mtu', field(4, enum_ibv_mtu))
setattr(struct_ibv_port_attr, 'active_mtu', field(8, enum_ibv_mtu))
setattr(struct_ibv_port_attr, 'gid_tbl_len', field(12, ctypes.c_int32))
setattr(struct_ibv_port_attr, 'port_cap_flags', field(16, uint32_t))
setattr(struct_ibv_port_attr, 'max_msg_sz', field(20, uint32_t))
setattr(struct_ibv_port_attr, 'bad_pkey_cntr', field(24, uint32_t))
setattr(struct_ibv_port_attr, 'qkey_viol_cntr', field(28, uint32_t))
setattr(struct_ibv_port_attr, 'pkey_tbl_len', field(32, uint16_t))
setattr(struct_ibv_port_attr, 'lid', field(34, uint16_t))
setattr(struct_ibv_port_attr, 'sm_lid', field(36, uint16_t))
setattr(struct_ibv_port_attr, 'lmc', field(38, uint8_t))
setattr(struct_ibv_port_attr, 'max_vl_num', field(39, uint8_t))
setattr(struct_ibv_port_attr, 'sm_sl', field(40, uint8_t))
setattr(struct_ibv_port_attr, 'subnet_timeout', field(41, uint8_t))
setattr(struct_ibv_port_attr, 'init_type_reply', field(42, uint8_t))
setattr(struct_ibv_port_attr, 'active_width', field(43, uint8_t))
setattr(struct_ibv_port_attr, 'active_speed', field(44, uint8_t))
setattr(struct_ibv_port_attr, 'phys_state', field(45, uint8_t))
setattr(struct_ibv_port_attr, 'link_layer', field(46, uint8_t))
setattr(struct_ibv_port_attr, 'flags', field(47, uint8_t))
setattr(struct_ibv_port_attr, 'port_cap_flags2', field(48, uint16_t))
setattr(struct_ibv_port_attr, 'active_speed_ex', field(52, uint32_t))
enum_ibv_event_type = CEnum(ctypes.c_uint32)
IBV_EVENT_CQ_ERR = enum_ibv_event_type.define('IBV_EVENT_CQ_ERR', 0)
IBV_EVENT_QP_FATAL = enum_ibv_event_type.define('IBV_EVENT_QP_FATAL', 1)
IBV_EVENT_QP_REQ_ERR = enum_ibv_event_type.define('IBV_EVENT_QP_REQ_ERR', 2)
IBV_EVENT_QP_ACCESS_ERR = enum_ibv_event_type.define('IBV_EVENT_QP_ACCESS_ERR', 3)
IBV_EVENT_COMM_EST = enum_ibv_event_type.define('IBV_EVENT_COMM_EST', 4)
IBV_EVENT_SQ_DRAINED = enum_ibv_event_type.define('IBV_EVENT_SQ_DRAINED', 5)
IBV_EVENT_PATH_MIG = enum_ibv_event_type.define('IBV_EVENT_PATH_MIG', 6)
IBV_EVENT_PATH_MIG_ERR = enum_ibv_event_type.define('IBV_EVENT_PATH_MIG_ERR', 7)
IBV_EVENT_DEVICE_FATAL = enum_ibv_event_type.define('IBV_EVENT_DEVICE_FATAL', 8)
IBV_EVENT_PORT_ACTIVE = enum_ibv_event_type.define('IBV_EVENT_PORT_ACTIVE', 9)
IBV_EVENT_PORT_ERR = enum_ibv_event_type.define('IBV_EVENT_PORT_ERR', 10)
IBV_EVENT_LID_CHANGE = enum_ibv_event_type.define('IBV_EVENT_LID_CHANGE', 11)
IBV_EVENT_PKEY_CHANGE = enum_ibv_event_type.define('IBV_EVENT_PKEY_CHANGE', 12)
IBV_EVENT_SM_CHANGE = enum_ibv_event_type.define('IBV_EVENT_SM_CHANGE', 13)
IBV_EVENT_SRQ_ERR = enum_ibv_event_type.define('IBV_EVENT_SRQ_ERR', 14)
IBV_EVENT_SRQ_LIMIT_REACHED = enum_ibv_event_type.define('IBV_EVENT_SRQ_LIMIT_REACHED', 15)
IBV_EVENT_QP_LAST_WQE_REACHED = enum_ibv_event_type.define('IBV_EVENT_QP_LAST_WQE_REACHED', 16)
IBV_EVENT_CLIENT_REREGISTER = enum_ibv_event_type.define('IBV_EVENT_CLIENT_REREGISTER', 17)
IBV_EVENT_GID_CHANGE = enum_ibv_event_type.define('IBV_EVENT_GID_CHANGE', 18)
IBV_EVENT_WQ_FATAL = enum_ibv_event_type.define('IBV_EVENT_WQ_FATAL', 19)

class struct_ibv_async_event(Struct): pass
class _anonunion12(Union): pass
class struct_ibv_wq(Struct): pass
enum_ibv_wq_state = CEnum(ctypes.c_uint32)
IBV_WQS_RESET = enum_ibv_wq_state.define('IBV_WQS_RESET', 0)
IBV_WQS_RDY = enum_ibv_wq_state.define('IBV_WQS_RDY', 1)
IBV_WQS_ERR = enum_ibv_wq_state.define('IBV_WQS_ERR', 2)
IBV_WQS_UNKNOWN = enum_ibv_wq_state.define('IBV_WQS_UNKNOWN', 3)

enum_ibv_wq_type = CEnum(ctypes.c_uint32)
IBV_WQT_RQ = enum_ibv_wq_type.define('IBV_WQT_RQ', 0)

struct_ibv_wq.SIZE = 152
struct_ibv_wq._fields_ = ['context', 'wq_context', 'pd', 'cq', 'wq_num', 'handle', 'state', 'wq_type', 'post_recv', 'mutex', 'cond', 'events_completed', 'comp_mask']
setattr(struct_ibv_wq, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_wq, 'wq_context', field(8, ctypes.c_void_p))
setattr(struct_ibv_wq, 'pd', field(16, Pointer(struct_ibv_pd)))
setattr(struct_ibv_wq, 'cq', field(24, Pointer(struct_ibv_cq)))
setattr(struct_ibv_wq, 'wq_num', field(32, uint32_t))
setattr(struct_ibv_wq, 'handle', field(36, uint32_t))
setattr(struct_ibv_wq, 'state', field(40, enum_ibv_wq_state))
setattr(struct_ibv_wq, 'wq_type', field(44, enum_ibv_wq_type))
setattr(struct_ibv_wq, 'post_recv', field(48, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_wq), Pointer(struct_ibv_recv_wr), Pointer(Pointer(struct_ibv_recv_wr)))))
setattr(struct_ibv_wq, 'mutex', field(56, pthread_mutex_t))
setattr(struct_ibv_wq, 'cond', field(96, pthread_cond_t))
setattr(struct_ibv_wq, 'events_completed', field(144, uint32_t))
setattr(struct_ibv_wq, 'comp_mask', field(148, uint32_t))
_anonunion12.SIZE = 8
_anonunion12._fields_ = ['cq', 'qp', 'srq', 'wq', 'port_num']
setattr(_anonunion12, 'cq', field(0, Pointer(struct_ibv_cq)))
setattr(_anonunion12, 'qp', field(0, Pointer(struct_ibv_qp)))
setattr(_anonunion12, 'srq', field(0, Pointer(struct_ibv_srq)))
setattr(_anonunion12, 'wq', field(0, Pointer(struct_ibv_wq)))
setattr(_anonunion12, 'port_num', field(0, ctypes.c_int32))
struct_ibv_async_event.SIZE = 16
struct_ibv_async_event._fields_ = ['element', 'event_type']
setattr(struct_ibv_async_event, 'element', field(0, _anonunion12))
setattr(struct_ibv_async_event, 'event_type', field(8, enum_ibv_event_type))
@dll.bind((enum_ibv_wc_status), Pointer(ctypes.c_char))
def ibv_wc_status_str(status): ...
_anonenum13 = CEnum(ctypes.c_uint32)
IBV_WC_IP_CSUM_OK_SHIFT = _anonenum13.define('IBV_WC_IP_CSUM_OK_SHIFT', 2)

enum_ibv_create_cq_wc_flags = CEnum(ctypes.c_uint32)
IBV_WC_EX_WITH_BYTE_LEN = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_BYTE_LEN', 1)
IBV_WC_EX_WITH_IMM = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_IMM', 2)
IBV_WC_EX_WITH_QP_NUM = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_QP_NUM', 4)
IBV_WC_EX_WITH_SRC_QP = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_SRC_QP', 8)
IBV_WC_EX_WITH_SLID = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_SLID', 16)
IBV_WC_EX_WITH_SL = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_SL', 32)
IBV_WC_EX_WITH_DLID_PATH_BITS = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_DLID_PATH_BITS', 64)
IBV_WC_EX_WITH_COMPLETION_TIMESTAMP = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_COMPLETION_TIMESTAMP', 128)
IBV_WC_EX_WITH_CVLAN = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_CVLAN', 256)
IBV_WC_EX_WITH_FLOW_TAG = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_FLOW_TAG', 512)
IBV_WC_EX_WITH_TM_INFO = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_TM_INFO', 1024)
IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK', 2048)

_anonenum14 = CEnum(ctypes.c_uint32)
IBV_WC_STANDARD_FLAGS = _anonenum14.define('IBV_WC_STANDARD_FLAGS', 127)

_anonenum15 = CEnum(ctypes.c_uint32)
IBV_CREATE_CQ_SUP_WC_FLAGS = _anonenum15.define('IBV_CREATE_CQ_SUP_WC_FLAGS', 4095)

enum_ibv_wc_flags = CEnum(ctypes.c_uint32)
IBV_WC_GRH = enum_ibv_wc_flags.define('IBV_WC_GRH', 1)
IBV_WC_WITH_IMM = enum_ibv_wc_flags.define('IBV_WC_WITH_IMM', 2)
IBV_WC_IP_CSUM_OK = enum_ibv_wc_flags.define('IBV_WC_IP_CSUM_OK', 4)
IBV_WC_WITH_INV = enum_ibv_wc_flags.define('IBV_WC_WITH_INV', 8)
IBV_WC_TM_SYNC_REQ = enum_ibv_wc_flags.define('IBV_WC_TM_SYNC_REQ', 16)
IBV_WC_TM_MATCH = enum_ibv_wc_flags.define('IBV_WC_TM_MATCH', 32)
IBV_WC_TM_DATA_VALID = enum_ibv_wc_flags.define('IBV_WC_TM_DATA_VALID', 64)

enum_ibv_access_flags = CEnum(ctypes.c_uint32)
IBV_ACCESS_LOCAL_WRITE = enum_ibv_access_flags.define('IBV_ACCESS_LOCAL_WRITE', 1)
IBV_ACCESS_REMOTE_WRITE = enum_ibv_access_flags.define('IBV_ACCESS_REMOTE_WRITE', 2)
IBV_ACCESS_REMOTE_READ = enum_ibv_access_flags.define('IBV_ACCESS_REMOTE_READ', 4)
IBV_ACCESS_REMOTE_ATOMIC = enum_ibv_access_flags.define('IBV_ACCESS_REMOTE_ATOMIC', 8)
IBV_ACCESS_MW_BIND = enum_ibv_access_flags.define('IBV_ACCESS_MW_BIND', 16)
IBV_ACCESS_ZERO_BASED = enum_ibv_access_flags.define('IBV_ACCESS_ZERO_BASED', 32)
IBV_ACCESS_ON_DEMAND = enum_ibv_access_flags.define('IBV_ACCESS_ON_DEMAND', 64)
IBV_ACCESS_HUGETLB = enum_ibv_access_flags.define('IBV_ACCESS_HUGETLB', 128)
IBV_ACCESS_FLUSH_GLOBAL = enum_ibv_access_flags.define('IBV_ACCESS_FLUSH_GLOBAL', 256)
IBV_ACCESS_FLUSH_PERSISTENT = enum_ibv_access_flags.define('IBV_ACCESS_FLUSH_PERSISTENT', 512)
IBV_ACCESS_RELAXED_ORDERING = enum_ibv_access_flags.define('IBV_ACCESS_RELAXED_ORDERING', 1048576)

class struct_ibv_td_init_attr(Struct): pass
struct_ibv_td_init_attr.SIZE = 4
struct_ibv_td_init_attr._fields_ = ['comp_mask']
setattr(struct_ibv_td_init_attr, 'comp_mask', field(0, uint32_t))
class struct_ibv_td(Struct): pass
struct_ibv_td.SIZE = 8
struct_ibv_td._fields_ = ['context']
setattr(struct_ibv_td, 'context', field(0, Pointer(struct_ibv_context)))
enum_ibv_xrcd_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_XRCD_INIT_ATTR_FD = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_FD', 1)
IBV_XRCD_INIT_ATTR_OFLAGS = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_OFLAGS', 2)
IBV_XRCD_INIT_ATTR_RESERVED = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_RESERVED', 4)

class struct_ibv_xrcd_init_attr(Struct): pass
struct_ibv_xrcd_init_attr.SIZE = 12
struct_ibv_xrcd_init_attr._fields_ = ['comp_mask', 'fd', 'oflags']
setattr(struct_ibv_xrcd_init_attr, 'comp_mask', field(0, uint32_t))
setattr(struct_ibv_xrcd_init_attr, 'fd', field(4, ctypes.c_int32))
setattr(struct_ibv_xrcd_init_attr, 'oflags', field(8, ctypes.c_int32))
class struct_ibv_xrcd(Struct): pass
struct_ibv_xrcd.SIZE = 8
struct_ibv_xrcd._fields_ = ['context']
setattr(struct_ibv_xrcd, 'context', field(0, Pointer(struct_ibv_context)))
enum_ibv_rereg_mr_flags = CEnum(ctypes.c_uint32)
IBV_REREG_MR_CHANGE_TRANSLATION = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_TRANSLATION', 1)
IBV_REREG_MR_CHANGE_PD = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_PD', 2)
IBV_REREG_MR_CHANGE_ACCESS = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_ACCESS', 4)
IBV_REREG_MR_FLAGS_SUPPORTED = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_FLAGS_SUPPORTED', 7)

class struct_ibv_global_route(Struct): pass
struct_ibv_global_route.SIZE = 24
struct_ibv_global_route._fields_ = ['dgid', 'flow_label', 'sgid_index', 'hop_limit', 'traffic_class']
setattr(struct_ibv_global_route, 'dgid', field(0, union_ibv_gid))
setattr(struct_ibv_global_route, 'flow_label', field(16, uint32_t))
setattr(struct_ibv_global_route, 'sgid_index', field(20, uint8_t))
setattr(struct_ibv_global_route, 'hop_limit', field(21, uint8_t))
setattr(struct_ibv_global_route, 'traffic_class', field(22, uint8_t))
class struct_ibv_grh(Struct): pass
__be16 = ctypes.c_uint16
struct_ibv_grh.SIZE = 40
struct_ibv_grh._fields_ = ['version_tclass_flow', 'paylen', 'next_hdr', 'hop_limit', 'sgid', 'dgid']
setattr(struct_ibv_grh, 'version_tclass_flow', field(0, ctypes.c_uint32))
setattr(struct_ibv_grh, 'paylen', field(4, ctypes.c_uint16))
setattr(struct_ibv_grh, 'next_hdr', field(6, uint8_t))
setattr(struct_ibv_grh, 'hop_limit', field(7, uint8_t))
setattr(struct_ibv_grh, 'sgid', field(8, union_ibv_gid))
setattr(struct_ibv_grh, 'dgid', field(24, union_ibv_gid))
enum_ibv_rate = CEnum(ctypes.c_uint32)
IBV_RATE_MAX = enum_ibv_rate.define('IBV_RATE_MAX', 0)
IBV_RATE_2_5_GBPS = enum_ibv_rate.define('IBV_RATE_2_5_GBPS', 2)
IBV_RATE_5_GBPS = enum_ibv_rate.define('IBV_RATE_5_GBPS', 5)
IBV_RATE_10_GBPS = enum_ibv_rate.define('IBV_RATE_10_GBPS', 3)
IBV_RATE_20_GBPS = enum_ibv_rate.define('IBV_RATE_20_GBPS', 6)
IBV_RATE_30_GBPS = enum_ibv_rate.define('IBV_RATE_30_GBPS', 4)
IBV_RATE_40_GBPS = enum_ibv_rate.define('IBV_RATE_40_GBPS', 7)
IBV_RATE_60_GBPS = enum_ibv_rate.define('IBV_RATE_60_GBPS', 8)
IBV_RATE_80_GBPS = enum_ibv_rate.define('IBV_RATE_80_GBPS', 9)
IBV_RATE_120_GBPS = enum_ibv_rate.define('IBV_RATE_120_GBPS', 10)
IBV_RATE_14_GBPS = enum_ibv_rate.define('IBV_RATE_14_GBPS', 11)
IBV_RATE_56_GBPS = enum_ibv_rate.define('IBV_RATE_56_GBPS', 12)
IBV_RATE_112_GBPS = enum_ibv_rate.define('IBV_RATE_112_GBPS', 13)
IBV_RATE_168_GBPS = enum_ibv_rate.define('IBV_RATE_168_GBPS', 14)
IBV_RATE_25_GBPS = enum_ibv_rate.define('IBV_RATE_25_GBPS', 15)
IBV_RATE_100_GBPS = enum_ibv_rate.define('IBV_RATE_100_GBPS', 16)
IBV_RATE_200_GBPS = enum_ibv_rate.define('IBV_RATE_200_GBPS', 17)
IBV_RATE_300_GBPS = enum_ibv_rate.define('IBV_RATE_300_GBPS', 18)
IBV_RATE_28_GBPS = enum_ibv_rate.define('IBV_RATE_28_GBPS', 19)
IBV_RATE_50_GBPS = enum_ibv_rate.define('IBV_RATE_50_GBPS', 20)
IBV_RATE_400_GBPS = enum_ibv_rate.define('IBV_RATE_400_GBPS', 21)
IBV_RATE_600_GBPS = enum_ibv_rate.define('IBV_RATE_600_GBPS', 22)
IBV_RATE_800_GBPS = enum_ibv_rate.define('IBV_RATE_800_GBPS', 23)
IBV_RATE_1200_GBPS = enum_ibv_rate.define('IBV_RATE_1200_GBPS', 24)

@dll.bind((enum_ibv_rate), ctypes.c_int32)
def ibv_rate_to_mult(rate): ...
@dll.bind((ctypes.c_int32), enum_ibv_rate)
def mult_to_ibv_rate(mult): ...
@dll.bind((enum_ibv_rate), ctypes.c_int32)
def ibv_rate_to_mbps(rate): ...
@dll.bind((ctypes.c_int32), enum_ibv_rate)
def mbps_to_ibv_rate(mbps): ...
class struct_ibv_ah_attr(Struct): pass
struct_ibv_ah_attr.SIZE = 32
struct_ibv_ah_attr._fields_ = ['grh', 'dlid', 'sl', 'src_path_bits', 'static_rate', 'is_global', 'port_num']
setattr(struct_ibv_ah_attr, 'grh', field(0, struct_ibv_global_route))
setattr(struct_ibv_ah_attr, 'dlid', field(24, uint16_t))
setattr(struct_ibv_ah_attr, 'sl', field(26, uint8_t))
setattr(struct_ibv_ah_attr, 'src_path_bits', field(27, uint8_t))
setattr(struct_ibv_ah_attr, 'static_rate', field(28, uint8_t))
setattr(struct_ibv_ah_attr, 'is_global', field(29, uint8_t))
setattr(struct_ibv_ah_attr, 'port_num', field(30, uint8_t))
enum_ibv_srq_attr_mask = CEnum(ctypes.c_uint32)
IBV_SRQ_MAX_WR = enum_ibv_srq_attr_mask.define('IBV_SRQ_MAX_WR', 1)
IBV_SRQ_LIMIT = enum_ibv_srq_attr_mask.define('IBV_SRQ_LIMIT', 2)

class struct_ibv_srq_attr(Struct): pass
struct_ibv_srq_attr.SIZE = 12
struct_ibv_srq_attr._fields_ = ['max_wr', 'max_sge', 'srq_limit']
setattr(struct_ibv_srq_attr, 'max_wr', field(0, uint32_t))
setattr(struct_ibv_srq_attr, 'max_sge', field(4, uint32_t))
setattr(struct_ibv_srq_attr, 'srq_limit', field(8, uint32_t))
class struct_ibv_srq_init_attr(Struct): pass
struct_ibv_srq_init_attr.SIZE = 24
struct_ibv_srq_init_attr._fields_ = ['srq_context', 'attr']
setattr(struct_ibv_srq_init_attr, 'srq_context', field(0, ctypes.c_void_p))
setattr(struct_ibv_srq_init_attr, 'attr', field(8, struct_ibv_srq_attr))
enum_ibv_srq_type = CEnum(ctypes.c_uint32)
IBV_SRQT_BASIC = enum_ibv_srq_type.define('IBV_SRQT_BASIC', 0)
IBV_SRQT_XRC = enum_ibv_srq_type.define('IBV_SRQT_XRC', 1)
IBV_SRQT_TM = enum_ibv_srq_type.define('IBV_SRQT_TM', 2)

enum_ibv_srq_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_SRQ_INIT_ATTR_TYPE = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_TYPE', 1)
IBV_SRQ_INIT_ATTR_PD = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_PD', 2)
IBV_SRQ_INIT_ATTR_XRCD = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_XRCD', 4)
IBV_SRQ_INIT_ATTR_CQ = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_CQ', 8)
IBV_SRQ_INIT_ATTR_TM = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_TM', 16)
IBV_SRQ_INIT_ATTR_RESERVED = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_RESERVED', 32)

class struct_ibv_tm_cap(Struct): pass
struct_ibv_tm_cap.SIZE = 8
struct_ibv_tm_cap._fields_ = ['max_num_tags', 'max_ops']
setattr(struct_ibv_tm_cap, 'max_num_tags', field(0, uint32_t))
setattr(struct_ibv_tm_cap, 'max_ops', field(4, uint32_t))
class struct_ibv_srq_init_attr_ex(Struct): pass
struct_ibv_srq_init_attr_ex.SIZE = 64
struct_ibv_srq_init_attr_ex._fields_ = ['srq_context', 'attr', 'comp_mask', 'srq_type', 'pd', 'xrcd', 'cq', 'tm_cap']
setattr(struct_ibv_srq_init_attr_ex, 'srq_context', field(0, ctypes.c_void_p))
setattr(struct_ibv_srq_init_attr_ex, 'attr', field(8, struct_ibv_srq_attr))
setattr(struct_ibv_srq_init_attr_ex, 'comp_mask', field(20, uint32_t))
setattr(struct_ibv_srq_init_attr_ex, 'srq_type', field(24, enum_ibv_srq_type))
setattr(struct_ibv_srq_init_attr_ex, 'pd', field(32, Pointer(struct_ibv_pd)))
setattr(struct_ibv_srq_init_attr_ex, 'xrcd', field(40, Pointer(struct_ibv_xrcd)))
setattr(struct_ibv_srq_init_attr_ex, 'cq', field(48, Pointer(struct_ibv_cq)))
setattr(struct_ibv_srq_init_attr_ex, 'tm_cap', field(56, struct_ibv_tm_cap))
enum_ibv_wq_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_WQ_INIT_ATTR_FLAGS = enum_ibv_wq_init_attr_mask.define('IBV_WQ_INIT_ATTR_FLAGS', 1)
IBV_WQ_INIT_ATTR_RESERVED = enum_ibv_wq_init_attr_mask.define('IBV_WQ_INIT_ATTR_RESERVED', 2)

enum_ibv_wq_flags = CEnum(ctypes.c_uint32)
IBV_WQ_FLAGS_CVLAN_STRIPPING = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_CVLAN_STRIPPING', 1)
IBV_WQ_FLAGS_SCATTER_FCS = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_SCATTER_FCS', 2)
IBV_WQ_FLAGS_DELAY_DROP = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_DELAY_DROP', 4)
IBV_WQ_FLAGS_PCI_WRITE_END_PADDING = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_PCI_WRITE_END_PADDING', 8)
IBV_WQ_FLAGS_RESERVED = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_RESERVED', 16)

class struct_ibv_wq_init_attr(Struct): pass
struct_ibv_wq_init_attr.SIZE = 48
struct_ibv_wq_init_attr._fields_ = ['wq_context', 'wq_type', 'max_wr', 'max_sge', 'pd', 'cq', 'comp_mask', 'create_flags']
setattr(struct_ibv_wq_init_attr, 'wq_context', field(0, ctypes.c_void_p))
setattr(struct_ibv_wq_init_attr, 'wq_type', field(8, enum_ibv_wq_type))
setattr(struct_ibv_wq_init_attr, 'max_wr', field(12, uint32_t))
setattr(struct_ibv_wq_init_attr, 'max_sge', field(16, uint32_t))
setattr(struct_ibv_wq_init_attr, 'pd', field(24, Pointer(struct_ibv_pd)))
setattr(struct_ibv_wq_init_attr, 'cq', field(32, Pointer(struct_ibv_cq)))
setattr(struct_ibv_wq_init_attr, 'comp_mask', field(40, uint32_t))
setattr(struct_ibv_wq_init_attr, 'create_flags', field(44, uint32_t))
enum_ibv_wq_attr_mask = CEnum(ctypes.c_uint32)
IBV_WQ_ATTR_STATE = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_STATE', 1)
IBV_WQ_ATTR_CURR_STATE = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_CURR_STATE', 2)
IBV_WQ_ATTR_FLAGS = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_FLAGS', 4)
IBV_WQ_ATTR_RESERVED = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_RESERVED', 8)

class struct_ibv_wq_attr(Struct): pass
struct_ibv_wq_attr.SIZE = 20
struct_ibv_wq_attr._fields_ = ['attr_mask', 'wq_state', 'curr_wq_state', 'flags', 'flags_mask']
setattr(struct_ibv_wq_attr, 'attr_mask', field(0, uint32_t))
setattr(struct_ibv_wq_attr, 'wq_state', field(4, enum_ibv_wq_state))
setattr(struct_ibv_wq_attr, 'curr_wq_state', field(8, enum_ibv_wq_state))
setattr(struct_ibv_wq_attr, 'flags', field(12, uint32_t))
setattr(struct_ibv_wq_attr, 'flags_mask', field(16, uint32_t))
class struct_ibv_rwq_ind_table(Struct): pass
struct_ibv_rwq_ind_table.SIZE = 24
struct_ibv_rwq_ind_table._fields_ = ['context', 'ind_tbl_handle', 'ind_tbl_num', 'comp_mask']
setattr(struct_ibv_rwq_ind_table, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_rwq_ind_table, 'ind_tbl_handle', field(8, ctypes.c_int32))
setattr(struct_ibv_rwq_ind_table, 'ind_tbl_num', field(12, ctypes.c_int32))
setattr(struct_ibv_rwq_ind_table, 'comp_mask', field(16, uint32_t))
enum_ibv_ind_table_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_CREATE_IND_TABLE_RESERVED = enum_ibv_ind_table_init_attr_mask.define('IBV_CREATE_IND_TABLE_RESERVED', 1)

class struct_ibv_rwq_ind_table_init_attr(Struct): pass
struct_ibv_rwq_ind_table_init_attr.SIZE = 24
struct_ibv_rwq_ind_table_init_attr._fields_ = ['log_ind_tbl_size', 'ind_tbl', 'comp_mask']
setattr(struct_ibv_rwq_ind_table_init_attr, 'log_ind_tbl_size', field(0, uint32_t))
setattr(struct_ibv_rwq_ind_table_init_attr, 'ind_tbl', field(8, Pointer(Pointer(struct_ibv_wq))))
setattr(struct_ibv_rwq_ind_table_init_attr, 'comp_mask', field(16, uint32_t))
class struct_ibv_qp_cap(Struct): pass
struct_ibv_qp_cap.SIZE = 20
struct_ibv_qp_cap._fields_ = ['max_send_wr', 'max_recv_wr', 'max_send_sge', 'max_recv_sge', 'max_inline_data']
setattr(struct_ibv_qp_cap, 'max_send_wr', field(0, uint32_t))
setattr(struct_ibv_qp_cap, 'max_recv_wr', field(4, uint32_t))
setattr(struct_ibv_qp_cap, 'max_send_sge', field(8, uint32_t))
setattr(struct_ibv_qp_cap, 'max_recv_sge', field(12, uint32_t))
setattr(struct_ibv_qp_cap, 'max_inline_data', field(16, uint32_t))
class struct_ibv_qp_init_attr(Struct): pass
struct_ibv_qp_init_attr.SIZE = 64
struct_ibv_qp_init_attr._fields_ = ['qp_context', 'send_cq', 'recv_cq', 'srq', 'cap', 'qp_type', 'sq_sig_all']
setattr(struct_ibv_qp_init_attr, 'qp_context', field(0, ctypes.c_void_p))
setattr(struct_ibv_qp_init_attr, 'send_cq', field(8, Pointer(struct_ibv_cq)))
setattr(struct_ibv_qp_init_attr, 'recv_cq', field(16, Pointer(struct_ibv_cq)))
setattr(struct_ibv_qp_init_attr, 'srq', field(24, Pointer(struct_ibv_srq)))
setattr(struct_ibv_qp_init_attr, 'cap', field(32, struct_ibv_qp_cap))
setattr(struct_ibv_qp_init_attr, 'qp_type', field(52, enum_ibv_qp_type))
setattr(struct_ibv_qp_init_attr, 'sq_sig_all', field(56, ctypes.c_int32))
enum_ibv_qp_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_QP_INIT_ATTR_PD = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_PD', 1)
IBV_QP_INIT_ATTR_XRCD = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_XRCD', 2)
IBV_QP_INIT_ATTR_CREATE_FLAGS = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_CREATE_FLAGS', 4)
IBV_QP_INIT_ATTR_MAX_TSO_HEADER = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_MAX_TSO_HEADER', 8)
IBV_QP_INIT_ATTR_IND_TABLE = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_IND_TABLE', 16)
IBV_QP_INIT_ATTR_RX_HASH = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_RX_HASH', 32)
IBV_QP_INIT_ATTR_SEND_OPS_FLAGS = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_SEND_OPS_FLAGS', 64)

enum_ibv_qp_create_flags = CEnum(ctypes.c_uint32)
IBV_QP_CREATE_BLOCK_SELF_MCAST_LB = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_BLOCK_SELF_MCAST_LB', 2)
IBV_QP_CREATE_SCATTER_FCS = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_SCATTER_FCS', 256)
IBV_QP_CREATE_CVLAN_STRIPPING = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_CVLAN_STRIPPING', 512)
IBV_QP_CREATE_SOURCE_QPN = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_SOURCE_QPN', 1024)
IBV_QP_CREATE_PCI_WRITE_END_PADDING = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_PCI_WRITE_END_PADDING', 2048)

enum_ibv_qp_create_send_ops_flags = CEnum(ctypes.c_uint32)
IBV_QP_EX_WITH_RDMA_WRITE = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_RDMA_WRITE', 1)
IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM', 2)
IBV_QP_EX_WITH_SEND = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_SEND', 4)
IBV_QP_EX_WITH_SEND_WITH_IMM = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_SEND_WITH_IMM', 8)
IBV_QP_EX_WITH_RDMA_READ = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_RDMA_READ', 16)
IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP', 32)
IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD', 64)
IBV_QP_EX_WITH_LOCAL_INV = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_LOCAL_INV', 128)
IBV_QP_EX_WITH_BIND_MW = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_BIND_MW', 256)
IBV_QP_EX_WITH_SEND_WITH_INV = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_SEND_WITH_INV', 512)
IBV_QP_EX_WITH_TSO = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_TSO', 1024)
IBV_QP_EX_WITH_FLUSH = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_FLUSH', 2048)
IBV_QP_EX_WITH_ATOMIC_WRITE = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_ATOMIC_WRITE', 4096)

class struct_ibv_rx_hash_conf(Struct): pass
struct_ibv_rx_hash_conf.SIZE = 24
struct_ibv_rx_hash_conf._fields_ = ['rx_hash_function', 'rx_hash_key_len', 'rx_hash_key', 'rx_hash_fields_mask']
setattr(struct_ibv_rx_hash_conf, 'rx_hash_function', field(0, uint8_t))
setattr(struct_ibv_rx_hash_conf, 'rx_hash_key_len', field(1, uint8_t))
setattr(struct_ibv_rx_hash_conf, 'rx_hash_key', field(8, Pointer(uint8_t)))
setattr(struct_ibv_rx_hash_conf, 'rx_hash_fields_mask', field(16, uint64_t))
class struct_ibv_qp_init_attr_ex(Struct): pass
struct_ibv_qp_init_attr_ex.SIZE = 136
struct_ibv_qp_init_attr_ex._fields_ = ['qp_context', 'send_cq', 'recv_cq', 'srq', 'cap', 'qp_type', 'sq_sig_all', 'comp_mask', 'pd', 'xrcd', 'create_flags', 'max_tso_header', 'rwq_ind_tbl', 'rx_hash_conf', 'source_qpn', 'send_ops_flags']
setattr(struct_ibv_qp_init_attr_ex, 'qp_context', field(0, ctypes.c_void_p))
setattr(struct_ibv_qp_init_attr_ex, 'send_cq', field(8, Pointer(struct_ibv_cq)))
setattr(struct_ibv_qp_init_attr_ex, 'recv_cq', field(16, Pointer(struct_ibv_cq)))
setattr(struct_ibv_qp_init_attr_ex, 'srq', field(24, Pointer(struct_ibv_srq)))
setattr(struct_ibv_qp_init_attr_ex, 'cap', field(32, struct_ibv_qp_cap))
setattr(struct_ibv_qp_init_attr_ex, 'qp_type', field(52, enum_ibv_qp_type))
setattr(struct_ibv_qp_init_attr_ex, 'sq_sig_all', field(56, ctypes.c_int32))
setattr(struct_ibv_qp_init_attr_ex, 'comp_mask', field(60, uint32_t))
setattr(struct_ibv_qp_init_attr_ex, 'pd', field(64, Pointer(struct_ibv_pd)))
setattr(struct_ibv_qp_init_attr_ex, 'xrcd', field(72, Pointer(struct_ibv_xrcd)))
setattr(struct_ibv_qp_init_attr_ex, 'create_flags', field(80, uint32_t))
setattr(struct_ibv_qp_init_attr_ex, 'max_tso_header', field(84, uint16_t))
setattr(struct_ibv_qp_init_attr_ex, 'rwq_ind_tbl', field(88, Pointer(struct_ibv_rwq_ind_table)))
setattr(struct_ibv_qp_init_attr_ex, 'rx_hash_conf', field(96, struct_ibv_rx_hash_conf))
setattr(struct_ibv_qp_init_attr_ex, 'source_qpn', field(120, uint32_t))
setattr(struct_ibv_qp_init_attr_ex, 'send_ops_flags', field(128, uint64_t))
enum_ibv_qp_open_attr_mask = CEnum(ctypes.c_uint32)
IBV_QP_OPEN_ATTR_NUM = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_NUM', 1)
IBV_QP_OPEN_ATTR_XRCD = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_XRCD', 2)
IBV_QP_OPEN_ATTR_CONTEXT = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_CONTEXT', 4)
IBV_QP_OPEN_ATTR_TYPE = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_TYPE', 8)
IBV_QP_OPEN_ATTR_RESERVED = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_RESERVED', 16)

class struct_ibv_qp_open_attr(Struct): pass
struct_ibv_qp_open_attr.SIZE = 32
struct_ibv_qp_open_attr._fields_ = ['comp_mask', 'qp_num', 'xrcd', 'qp_context', 'qp_type']
setattr(struct_ibv_qp_open_attr, 'comp_mask', field(0, uint32_t))
setattr(struct_ibv_qp_open_attr, 'qp_num', field(4, uint32_t))
setattr(struct_ibv_qp_open_attr, 'xrcd', field(8, Pointer(struct_ibv_xrcd)))
setattr(struct_ibv_qp_open_attr, 'qp_context', field(16, ctypes.c_void_p))
setattr(struct_ibv_qp_open_attr, 'qp_type', field(24, enum_ibv_qp_type))
enum_ibv_qp_attr_mask = CEnum(ctypes.c_uint32)
IBV_QP_STATE = enum_ibv_qp_attr_mask.define('IBV_QP_STATE', 1)
IBV_QP_CUR_STATE = enum_ibv_qp_attr_mask.define('IBV_QP_CUR_STATE', 2)
IBV_QP_EN_SQD_ASYNC_NOTIFY = enum_ibv_qp_attr_mask.define('IBV_QP_EN_SQD_ASYNC_NOTIFY', 4)
IBV_QP_ACCESS_FLAGS = enum_ibv_qp_attr_mask.define('IBV_QP_ACCESS_FLAGS', 8)
IBV_QP_PKEY_INDEX = enum_ibv_qp_attr_mask.define('IBV_QP_PKEY_INDEX', 16)
IBV_QP_PORT = enum_ibv_qp_attr_mask.define('IBV_QP_PORT', 32)
IBV_QP_QKEY = enum_ibv_qp_attr_mask.define('IBV_QP_QKEY', 64)
IBV_QP_AV = enum_ibv_qp_attr_mask.define('IBV_QP_AV', 128)
IBV_QP_PATH_MTU = enum_ibv_qp_attr_mask.define('IBV_QP_PATH_MTU', 256)
IBV_QP_TIMEOUT = enum_ibv_qp_attr_mask.define('IBV_QP_TIMEOUT', 512)
IBV_QP_RETRY_CNT = enum_ibv_qp_attr_mask.define('IBV_QP_RETRY_CNT', 1024)
IBV_QP_RNR_RETRY = enum_ibv_qp_attr_mask.define('IBV_QP_RNR_RETRY', 2048)
IBV_QP_RQ_PSN = enum_ibv_qp_attr_mask.define('IBV_QP_RQ_PSN', 4096)
IBV_QP_MAX_QP_RD_ATOMIC = enum_ibv_qp_attr_mask.define('IBV_QP_MAX_QP_RD_ATOMIC', 8192)
IBV_QP_ALT_PATH = enum_ibv_qp_attr_mask.define('IBV_QP_ALT_PATH', 16384)
IBV_QP_MIN_RNR_TIMER = enum_ibv_qp_attr_mask.define('IBV_QP_MIN_RNR_TIMER', 32768)
IBV_QP_SQ_PSN = enum_ibv_qp_attr_mask.define('IBV_QP_SQ_PSN', 65536)
IBV_QP_MAX_DEST_RD_ATOMIC = enum_ibv_qp_attr_mask.define('IBV_QP_MAX_DEST_RD_ATOMIC', 131072)
IBV_QP_PATH_MIG_STATE = enum_ibv_qp_attr_mask.define('IBV_QP_PATH_MIG_STATE', 262144)
IBV_QP_CAP = enum_ibv_qp_attr_mask.define('IBV_QP_CAP', 524288)
IBV_QP_DEST_QPN = enum_ibv_qp_attr_mask.define('IBV_QP_DEST_QPN', 1048576)
IBV_QP_RATE_LIMIT = enum_ibv_qp_attr_mask.define('IBV_QP_RATE_LIMIT', 33554432)

enum_ibv_query_qp_data_in_order_flags = CEnum(ctypes.c_uint32)
IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS = enum_ibv_query_qp_data_in_order_flags.define('IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS', 1)

enum_ibv_query_qp_data_in_order_caps = CEnum(ctypes.c_uint32)
IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG = enum_ibv_query_qp_data_in_order_caps.define('IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG', 1)
IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES = enum_ibv_query_qp_data_in_order_caps.define('IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES', 2)

enum_ibv_mig_state = CEnum(ctypes.c_uint32)
IBV_MIG_MIGRATED = enum_ibv_mig_state.define('IBV_MIG_MIGRATED', 0)
IBV_MIG_REARM = enum_ibv_mig_state.define('IBV_MIG_REARM', 1)
IBV_MIG_ARMED = enum_ibv_mig_state.define('IBV_MIG_ARMED', 2)

class struct_ibv_qp_attr(Struct): pass
struct_ibv_qp_attr.SIZE = 144
struct_ibv_qp_attr._fields_ = ['qp_state', 'cur_qp_state', 'path_mtu', 'path_mig_state', 'qkey', 'rq_psn', 'sq_psn', 'dest_qp_num', 'qp_access_flags', 'cap', 'ah_attr', 'alt_ah_attr', 'pkey_index', 'alt_pkey_index', 'en_sqd_async_notify', 'sq_draining', 'max_rd_atomic', 'max_dest_rd_atomic', 'min_rnr_timer', 'port_num', 'timeout', 'retry_cnt', 'rnr_retry', 'alt_port_num', 'alt_timeout', 'rate_limit']
setattr(struct_ibv_qp_attr, 'qp_state', field(0, enum_ibv_qp_state))
setattr(struct_ibv_qp_attr, 'cur_qp_state', field(4, enum_ibv_qp_state))
setattr(struct_ibv_qp_attr, 'path_mtu', field(8, enum_ibv_mtu))
setattr(struct_ibv_qp_attr, 'path_mig_state', field(12, enum_ibv_mig_state))
setattr(struct_ibv_qp_attr, 'qkey', field(16, uint32_t))
setattr(struct_ibv_qp_attr, 'rq_psn', field(20, uint32_t))
setattr(struct_ibv_qp_attr, 'sq_psn', field(24, uint32_t))
setattr(struct_ibv_qp_attr, 'dest_qp_num', field(28, uint32_t))
setattr(struct_ibv_qp_attr, 'qp_access_flags', field(32, ctypes.c_uint32))
setattr(struct_ibv_qp_attr, 'cap', field(36, struct_ibv_qp_cap))
setattr(struct_ibv_qp_attr, 'ah_attr', field(56, struct_ibv_ah_attr))
setattr(struct_ibv_qp_attr, 'alt_ah_attr', field(88, struct_ibv_ah_attr))
setattr(struct_ibv_qp_attr, 'pkey_index', field(120, uint16_t))
setattr(struct_ibv_qp_attr, 'alt_pkey_index', field(122, uint16_t))
setattr(struct_ibv_qp_attr, 'en_sqd_async_notify', field(124, uint8_t))
setattr(struct_ibv_qp_attr, 'sq_draining', field(125, uint8_t))
setattr(struct_ibv_qp_attr, 'max_rd_atomic', field(126, uint8_t))
setattr(struct_ibv_qp_attr, 'max_dest_rd_atomic', field(127, uint8_t))
setattr(struct_ibv_qp_attr, 'min_rnr_timer', field(128, uint8_t))
setattr(struct_ibv_qp_attr, 'port_num', field(129, uint8_t))
setattr(struct_ibv_qp_attr, 'timeout', field(130, uint8_t))
setattr(struct_ibv_qp_attr, 'retry_cnt', field(131, uint8_t))
setattr(struct_ibv_qp_attr, 'rnr_retry', field(132, uint8_t))
setattr(struct_ibv_qp_attr, 'alt_port_num', field(133, uint8_t))
setattr(struct_ibv_qp_attr, 'alt_timeout', field(134, uint8_t))
setattr(struct_ibv_qp_attr, 'rate_limit', field(136, uint32_t))
class struct_ibv_qp_rate_limit_attr(Struct): pass
struct_ibv_qp_rate_limit_attr.SIZE = 16
struct_ibv_qp_rate_limit_attr._fields_ = ['rate_limit', 'max_burst_sz', 'typical_pkt_sz', 'comp_mask']
setattr(struct_ibv_qp_rate_limit_attr, 'rate_limit', field(0, uint32_t))
setattr(struct_ibv_qp_rate_limit_attr, 'max_burst_sz', field(4, uint32_t))
setattr(struct_ibv_qp_rate_limit_attr, 'typical_pkt_sz', field(8, uint16_t))
setattr(struct_ibv_qp_rate_limit_attr, 'comp_mask', field(12, uint32_t))
@dll.bind((enum_ibv_wr_opcode), Pointer(ctypes.c_char))
def ibv_wr_opcode_str(opcode): ...
enum_ibv_send_flags = CEnum(ctypes.c_uint32)
IBV_SEND_FENCE = enum_ibv_send_flags.define('IBV_SEND_FENCE', 1)
IBV_SEND_SIGNALED = enum_ibv_send_flags.define('IBV_SEND_SIGNALED', 2)
IBV_SEND_SOLICITED = enum_ibv_send_flags.define('IBV_SEND_SOLICITED', 4)
IBV_SEND_INLINE = enum_ibv_send_flags.define('IBV_SEND_INLINE', 8)
IBV_SEND_IP_CSUM = enum_ibv_send_flags.define('IBV_SEND_IP_CSUM', 16)

enum_ibv_placement_type = CEnum(ctypes.c_uint32)
IBV_FLUSH_GLOBAL = enum_ibv_placement_type.define('IBV_FLUSH_GLOBAL', 1)
IBV_FLUSH_PERSISTENT = enum_ibv_placement_type.define('IBV_FLUSH_PERSISTENT', 2)

enum_ibv_selectivity_level = CEnum(ctypes.c_uint32)
IBV_FLUSH_RANGE = enum_ibv_selectivity_level.define('IBV_FLUSH_RANGE', 0)
IBV_FLUSH_MR = enum_ibv_selectivity_level.define('IBV_FLUSH_MR', 1)

class struct_ibv_data_buf(Struct): pass
struct_ibv_data_buf.SIZE = 16
struct_ibv_data_buf._fields_ = ['addr', 'length']
setattr(struct_ibv_data_buf, 'addr', field(0, ctypes.c_void_p))
setattr(struct_ibv_data_buf, 'length', field(8, size_t))
enum_ibv_ops_wr_opcode = CEnum(ctypes.c_uint32)
IBV_WR_TAG_ADD = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_ADD', 0)
IBV_WR_TAG_DEL = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_DEL', 1)
IBV_WR_TAG_SYNC = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_SYNC', 2)

enum_ibv_ops_flags = CEnum(ctypes.c_uint32)
IBV_OPS_SIGNALED = enum_ibv_ops_flags.define('IBV_OPS_SIGNALED', 1)
IBV_OPS_TM_SYNC = enum_ibv_ops_flags.define('IBV_OPS_TM_SYNC', 2)

class struct_ibv_ops_wr(Struct): pass
class _anonstruct16(Struct): pass
class _anonstruct17(Struct): pass
_anonstruct17.SIZE = 40
_anonstruct17._fields_ = ['recv_wr_id', 'sg_list', 'num_sge', 'tag', 'mask']
setattr(_anonstruct17, 'recv_wr_id', field(0, uint64_t))
setattr(_anonstruct17, 'sg_list', field(8, Pointer(struct_ibv_sge)))
setattr(_anonstruct17, 'num_sge', field(16, ctypes.c_int32))
setattr(_anonstruct17, 'tag', field(24, uint64_t))
setattr(_anonstruct17, 'mask', field(32, uint64_t))
_anonstruct16.SIZE = 48
_anonstruct16._fields_ = ['unexpected_cnt', 'handle', 'add']
setattr(_anonstruct16, 'unexpected_cnt', field(0, uint32_t))
setattr(_anonstruct16, 'handle', field(4, uint32_t))
setattr(_anonstruct16, 'add', field(8, _anonstruct17))
struct_ibv_ops_wr.SIZE = 72
struct_ibv_ops_wr._fields_ = ['wr_id', 'next', 'opcode', 'flags', 'tm']
setattr(struct_ibv_ops_wr, 'wr_id', field(0, uint64_t))
setattr(struct_ibv_ops_wr, 'next', field(8, Pointer(struct_ibv_ops_wr)))
setattr(struct_ibv_ops_wr, 'opcode', field(16, enum_ibv_ops_wr_opcode))
setattr(struct_ibv_ops_wr, 'flags', field(20, ctypes.c_int32))
setattr(struct_ibv_ops_wr, 'tm', field(24, _anonstruct16))
class struct_ibv_qp_ex(Struct): pass
struct_ibv_qp_ex.SIZE = 360
struct_ibv_qp_ex._fields_ = ['qp_base', 'comp_mask', 'wr_id', 'wr_flags', 'wr_atomic_cmp_swp', 'wr_atomic_fetch_add', 'wr_bind_mw', 'wr_local_inv', 'wr_rdma_read', 'wr_rdma_write', 'wr_rdma_write_imm', 'wr_send', 'wr_send_imm', 'wr_send_inv', 'wr_send_tso', 'wr_set_ud_addr', 'wr_set_xrc_srqn', 'wr_set_inline_data', 'wr_set_inline_data_list', 'wr_set_sge', 'wr_set_sge_list', 'wr_start', 'wr_complete', 'wr_abort', 'wr_atomic_write', 'wr_flush']
setattr(struct_ibv_qp_ex, 'qp_base', field(0, struct_ibv_qp))
setattr(struct_ibv_qp_ex, 'comp_mask', field(160, uint64_t))
setattr(struct_ibv_qp_ex, 'wr_id', field(168, uint64_t))
setattr(struct_ibv_qp_ex, 'wr_flags', field(176, ctypes.c_uint32))
setattr(struct_ibv_qp_ex, 'wr_atomic_cmp_swp', field(184, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t, uint64_t, uint64_t, uint64_t)))
setattr(struct_ibv_qp_ex, 'wr_atomic_fetch_add', field(192, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t, uint64_t, uint64_t)))
setattr(struct_ibv_qp_ex, 'wr_bind_mw', field(200, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), Pointer(struct_ibv_mw), uint32_t, Pointer(struct_ibv_mw_bind_info))))
setattr(struct_ibv_qp_ex, 'wr_local_inv', field(208, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t)))
setattr(struct_ibv_qp_ex, 'wr_rdma_read', field(216, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t, uint64_t)))
setattr(struct_ibv_qp_ex, 'wr_rdma_write', field(224, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t, uint64_t)))
setattr(struct_ibv_qp_ex, 'wr_rdma_write_imm', field(232, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t, uint64_t, ctypes.c_uint32)))
setattr(struct_ibv_qp_ex, 'wr_send', field(240, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex))))
setattr(struct_ibv_qp_ex, 'wr_send_imm', field(248, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), ctypes.c_uint32)))
setattr(struct_ibv_qp_ex, 'wr_send_inv', field(256, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t)))
setattr(struct_ibv_qp_ex, 'wr_send_tso', field(264, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), ctypes.c_void_p, uint16_t, uint16_t)))
setattr(struct_ibv_qp_ex, 'wr_set_ud_addr', field(272, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), Pointer(struct_ibv_ah), uint32_t, uint32_t)))
setattr(struct_ibv_qp_ex, 'wr_set_xrc_srqn', field(280, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t)))
setattr(struct_ibv_qp_ex, 'wr_set_inline_data', field(288, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), ctypes.c_void_p, size_t)))
setattr(struct_ibv_qp_ex, 'wr_set_inline_data_list', field(296, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), size_t, Pointer(struct_ibv_data_buf))))
setattr(struct_ibv_qp_ex, 'wr_set_sge', field(304, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t, uint64_t, uint32_t)))
setattr(struct_ibv_qp_ex, 'wr_set_sge_list', field(312, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), size_t, Pointer(struct_ibv_sge))))
setattr(struct_ibv_qp_ex, 'wr_start', field(320, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex))))
setattr(struct_ibv_qp_ex, 'wr_complete', field(328, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_qp_ex))))
setattr(struct_ibv_qp_ex, 'wr_abort', field(336, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex))))
setattr(struct_ibv_qp_ex, 'wr_atomic_write', field(344, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t, uint64_t, ctypes.c_void_p)))
setattr(struct_ibv_qp_ex, 'wr_flush', field(352, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_qp_ex), uint32_t, uint64_t, size_t, uint8_t, uint8_t)))
@dll.bind((Pointer(struct_ibv_qp)), Pointer(struct_ibv_qp_ex))
def ibv_qp_to_qp_ex(qp): ...
class struct_ibv_ece(Struct): pass
struct_ibv_ece.SIZE = 12
struct_ibv_ece._fields_ = ['vendor_id', 'options', 'comp_mask']
setattr(struct_ibv_ece, 'vendor_id', field(0, uint32_t))
setattr(struct_ibv_ece, 'options', field(4, uint32_t))
setattr(struct_ibv_ece, 'comp_mask', field(8, uint32_t))
class struct_ibv_poll_cq_attr(Struct): pass
struct_ibv_poll_cq_attr.SIZE = 4
struct_ibv_poll_cq_attr._fields_ = ['comp_mask']
setattr(struct_ibv_poll_cq_attr, 'comp_mask', field(0, uint32_t))
class struct_ibv_wc_tm_info(Struct): pass
struct_ibv_wc_tm_info.SIZE = 16
struct_ibv_wc_tm_info._fields_ = ['tag', 'priv']
setattr(struct_ibv_wc_tm_info, 'tag', field(0, uint64_t))
setattr(struct_ibv_wc_tm_info, 'priv', field(8, uint32_t))
class struct_ibv_cq_ex(Struct): pass
struct_ibv_cq_ex.SIZE = 288
struct_ibv_cq_ex._fields_ = ['context', 'channel', 'cq_context', 'handle', 'cqe', 'mutex', 'cond', 'comp_events_completed', 'async_events_completed', 'comp_mask', 'status', 'wr_id', 'start_poll', 'next_poll', 'end_poll', 'read_opcode', 'read_vendor_err', 'read_byte_len', 'read_imm_data', 'read_qp_num', 'read_src_qp', 'read_wc_flags', 'read_slid', 'read_sl', 'read_dlid_path_bits', 'read_completion_ts', 'read_cvlan', 'read_flow_tag', 'read_tm_info', 'read_completion_wallclock_ns']
setattr(struct_ibv_cq_ex, 'context', field(0, Pointer(struct_ibv_context)))
setattr(struct_ibv_cq_ex, 'channel', field(8, Pointer(struct_ibv_comp_channel)))
setattr(struct_ibv_cq_ex, 'cq_context', field(16, ctypes.c_void_p))
setattr(struct_ibv_cq_ex, 'handle', field(24, uint32_t))
setattr(struct_ibv_cq_ex, 'cqe', field(28, ctypes.c_int32))
setattr(struct_ibv_cq_ex, 'mutex', field(32, pthread_mutex_t))
setattr(struct_ibv_cq_ex, 'cond', field(72, pthread_cond_t))
setattr(struct_ibv_cq_ex, 'comp_events_completed', field(120, uint32_t))
setattr(struct_ibv_cq_ex, 'async_events_completed', field(124, uint32_t))
setattr(struct_ibv_cq_ex, 'comp_mask', field(128, uint32_t))
setattr(struct_ibv_cq_ex, 'status', field(132, enum_ibv_wc_status))
setattr(struct_ibv_cq_ex, 'wr_id', field(136, uint64_t))
setattr(struct_ibv_cq_ex, 'start_poll', field(144, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_cq_ex), Pointer(struct_ibv_poll_cq_attr))))
setattr(struct_ibv_cq_ex, 'next_poll', field(152, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'end_poll', field(160, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_opcode', field(168, ctypes.CFUNCTYPE(enum_ibv_wc_opcode, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_vendor_err', field(176, ctypes.CFUNCTYPE(uint32_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_byte_len', field(184, ctypes.CFUNCTYPE(uint32_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_imm_data', field(192, ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_qp_num', field(200, ctypes.CFUNCTYPE(uint32_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_src_qp', field(208, ctypes.CFUNCTYPE(uint32_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_wc_flags', field(216, ctypes.CFUNCTYPE(ctypes.c_uint32, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_slid', field(224, ctypes.CFUNCTYPE(uint32_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_sl', field(232, ctypes.CFUNCTYPE(uint8_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_dlid_path_bits', field(240, ctypes.CFUNCTYPE(uint8_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_completion_ts', field(248, ctypes.CFUNCTYPE(uint64_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_cvlan', field(256, ctypes.CFUNCTYPE(uint16_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_flow_tag', field(264, ctypes.CFUNCTYPE(uint32_t, Pointer(struct_ibv_cq_ex))))
setattr(struct_ibv_cq_ex, 'read_tm_info', field(272, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_cq_ex), Pointer(struct_ibv_wc_tm_info))))
setattr(struct_ibv_cq_ex, 'read_completion_wallclock_ns', field(280, ctypes.CFUNCTYPE(uint64_t, Pointer(struct_ibv_cq_ex))))
enum_ibv_cq_attr_mask = CEnum(ctypes.c_uint32)
IBV_CQ_ATTR_MODERATE = enum_ibv_cq_attr_mask.define('IBV_CQ_ATTR_MODERATE', 1)
IBV_CQ_ATTR_RESERVED = enum_ibv_cq_attr_mask.define('IBV_CQ_ATTR_RESERVED', 2)

class struct_ibv_moderate_cq(Struct): pass
struct_ibv_moderate_cq.SIZE = 4
struct_ibv_moderate_cq._fields_ = ['cq_count', 'cq_period']
setattr(struct_ibv_moderate_cq, 'cq_count', field(0, uint16_t))
setattr(struct_ibv_moderate_cq, 'cq_period', field(2, uint16_t))
class struct_ibv_modify_cq_attr(Struct): pass
struct_ibv_modify_cq_attr.SIZE = 8
struct_ibv_modify_cq_attr._fields_ = ['attr_mask', 'moderate']
setattr(struct_ibv_modify_cq_attr, 'attr_mask', field(0, uint32_t))
setattr(struct_ibv_modify_cq_attr, 'moderate', field(4, struct_ibv_moderate_cq))
enum_ibv_flow_flags = CEnum(ctypes.c_uint32)
IBV_FLOW_ATTR_FLAGS_DONT_TRAP = enum_ibv_flow_flags.define('IBV_FLOW_ATTR_FLAGS_DONT_TRAP', 2)
IBV_FLOW_ATTR_FLAGS_EGRESS = enum_ibv_flow_flags.define('IBV_FLOW_ATTR_FLAGS_EGRESS', 4)

enum_ibv_flow_attr_type = CEnum(ctypes.c_uint32)
IBV_FLOW_ATTR_NORMAL = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_NORMAL', 0)
IBV_FLOW_ATTR_ALL_DEFAULT = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_ALL_DEFAULT', 1)
IBV_FLOW_ATTR_MC_DEFAULT = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_MC_DEFAULT', 2)
IBV_FLOW_ATTR_SNIFFER = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_SNIFFER', 3)

enum_ibv_flow_spec_type = CEnum(ctypes.c_uint32)
IBV_FLOW_SPEC_ETH = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ETH', 32)
IBV_FLOW_SPEC_IPV4 = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_IPV4', 48)
IBV_FLOW_SPEC_IPV6 = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_IPV6', 49)
IBV_FLOW_SPEC_IPV4_EXT = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_IPV4_EXT', 50)
IBV_FLOW_SPEC_ESP = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ESP', 52)
IBV_FLOW_SPEC_TCP = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_TCP', 64)
IBV_FLOW_SPEC_UDP = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_UDP', 65)
IBV_FLOW_SPEC_VXLAN_TUNNEL = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_VXLAN_TUNNEL', 80)
IBV_FLOW_SPEC_GRE = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_GRE', 81)
IBV_FLOW_SPEC_MPLS = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_MPLS', 96)
IBV_FLOW_SPEC_INNER = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_INNER', 256)
IBV_FLOW_SPEC_ACTION_TAG = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ACTION_TAG', 4096)
IBV_FLOW_SPEC_ACTION_DROP = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ACTION_DROP', 4097)
IBV_FLOW_SPEC_ACTION_HANDLE = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ACTION_HANDLE', 4098)
IBV_FLOW_SPEC_ACTION_COUNT = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ACTION_COUNT', 4099)

class struct_ibv_flow_eth_filter(Struct): pass
struct_ibv_flow_eth_filter.SIZE = 16
struct_ibv_flow_eth_filter._fields_ = ['dst_mac', 'src_mac', 'ether_type', 'vlan_tag']
setattr(struct_ibv_flow_eth_filter, 'dst_mac', field(0, Array(uint8_t, 6)))
setattr(struct_ibv_flow_eth_filter, 'src_mac', field(6, Array(uint8_t, 6)))
setattr(struct_ibv_flow_eth_filter, 'ether_type', field(12, uint16_t))
setattr(struct_ibv_flow_eth_filter, 'vlan_tag', field(14, uint16_t))
class struct_ibv_flow_spec_eth(Struct): pass
struct_ibv_flow_spec_eth.SIZE = 40
struct_ibv_flow_spec_eth._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_eth, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_eth, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_eth, 'val', field(6, struct_ibv_flow_eth_filter))
setattr(struct_ibv_flow_spec_eth, 'mask', field(22, struct_ibv_flow_eth_filter))
class struct_ibv_flow_ipv4_filter(Struct): pass
struct_ibv_flow_ipv4_filter.SIZE = 8
struct_ibv_flow_ipv4_filter._fields_ = ['src_ip', 'dst_ip']
setattr(struct_ibv_flow_ipv4_filter, 'src_ip', field(0, uint32_t))
setattr(struct_ibv_flow_ipv4_filter, 'dst_ip', field(4, uint32_t))
class struct_ibv_flow_spec_ipv4(Struct): pass
struct_ibv_flow_spec_ipv4.SIZE = 24
struct_ibv_flow_spec_ipv4._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_ipv4, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_ipv4, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_ipv4, 'val', field(8, struct_ibv_flow_ipv4_filter))
setattr(struct_ibv_flow_spec_ipv4, 'mask', field(16, struct_ibv_flow_ipv4_filter))
class struct_ibv_flow_ipv4_ext_filter(Struct): pass
struct_ibv_flow_ipv4_ext_filter.SIZE = 12
struct_ibv_flow_ipv4_ext_filter._fields_ = ['src_ip', 'dst_ip', 'proto', 'tos', 'ttl', 'flags']
setattr(struct_ibv_flow_ipv4_ext_filter, 'src_ip', field(0, uint32_t))
setattr(struct_ibv_flow_ipv4_ext_filter, 'dst_ip', field(4, uint32_t))
setattr(struct_ibv_flow_ipv4_ext_filter, 'proto', field(8, uint8_t))
setattr(struct_ibv_flow_ipv4_ext_filter, 'tos', field(9, uint8_t))
setattr(struct_ibv_flow_ipv4_ext_filter, 'ttl', field(10, uint8_t))
setattr(struct_ibv_flow_ipv4_ext_filter, 'flags', field(11, uint8_t))
class struct_ibv_flow_spec_ipv4_ext(Struct): pass
struct_ibv_flow_spec_ipv4_ext.SIZE = 32
struct_ibv_flow_spec_ipv4_ext._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_ipv4_ext, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_ipv4_ext, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_ipv4_ext, 'val', field(8, struct_ibv_flow_ipv4_ext_filter))
setattr(struct_ibv_flow_spec_ipv4_ext, 'mask', field(20, struct_ibv_flow_ipv4_ext_filter))
class struct_ibv_flow_ipv6_filter(Struct): pass
struct_ibv_flow_ipv6_filter.SIZE = 40
struct_ibv_flow_ipv6_filter._fields_ = ['src_ip', 'dst_ip', 'flow_label', 'next_hdr', 'traffic_class', 'hop_limit']
setattr(struct_ibv_flow_ipv6_filter, 'src_ip', field(0, Array(uint8_t, 16)))
setattr(struct_ibv_flow_ipv6_filter, 'dst_ip', field(16, Array(uint8_t, 16)))
setattr(struct_ibv_flow_ipv6_filter, 'flow_label', field(32, uint32_t))
setattr(struct_ibv_flow_ipv6_filter, 'next_hdr', field(36, uint8_t))
setattr(struct_ibv_flow_ipv6_filter, 'traffic_class', field(37, uint8_t))
setattr(struct_ibv_flow_ipv6_filter, 'hop_limit', field(38, uint8_t))
class struct_ibv_flow_spec_ipv6(Struct): pass
struct_ibv_flow_spec_ipv6.SIZE = 88
struct_ibv_flow_spec_ipv6._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_ipv6, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_ipv6, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_ipv6, 'val', field(8, struct_ibv_flow_ipv6_filter))
setattr(struct_ibv_flow_spec_ipv6, 'mask', field(48, struct_ibv_flow_ipv6_filter))
class struct_ibv_flow_esp_filter(Struct): pass
struct_ibv_flow_esp_filter.SIZE = 8
struct_ibv_flow_esp_filter._fields_ = ['spi', 'seq']
setattr(struct_ibv_flow_esp_filter, 'spi', field(0, uint32_t))
setattr(struct_ibv_flow_esp_filter, 'seq', field(4, uint32_t))
class struct_ibv_flow_spec_esp(Struct): pass
struct_ibv_flow_spec_esp.SIZE = 24
struct_ibv_flow_spec_esp._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_esp, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_esp, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_esp, 'val', field(8, struct_ibv_flow_esp_filter))
setattr(struct_ibv_flow_spec_esp, 'mask', field(16, struct_ibv_flow_esp_filter))
class struct_ibv_flow_tcp_udp_filter(Struct): pass
struct_ibv_flow_tcp_udp_filter.SIZE = 4
struct_ibv_flow_tcp_udp_filter._fields_ = ['dst_port', 'src_port']
setattr(struct_ibv_flow_tcp_udp_filter, 'dst_port', field(0, uint16_t))
setattr(struct_ibv_flow_tcp_udp_filter, 'src_port', field(2, uint16_t))
class struct_ibv_flow_spec_tcp_udp(Struct): pass
struct_ibv_flow_spec_tcp_udp.SIZE = 16
struct_ibv_flow_spec_tcp_udp._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_tcp_udp, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_tcp_udp, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_tcp_udp, 'val', field(6, struct_ibv_flow_tcp_udp_filter))
setattr(struct_ibv_flow_spec_tcp_udp, 'mask', field(10, struct_ibv_flow_tcp_udp_filter))
class struct_ibv_flow_gre_filter(Struct): pass
struct_ibv_flow_gre_filter.SIZE = 8
struct_ibv_flow_gre_filter._fields_ = ['c_ks_res0_ver', 'protocol', 'key']
setattr(struct_ibv_flow_gre_filter, 'c_ks_res0_ver', field(0, uint16_t))
setattr(struct_ibv_flow_gre_filter, 'protocol', field(2, uint16_t))
setattr(struct_ibv_flow_gre_filter, 'key', field(4, uint32_t))
class struct_ibv_flow_spec_gre(Struct): pass
struct_ibv_flow_spec_gre.SIZE = 24
struct_ibv_flow_spec_gre._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_gre, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_gre, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_gre, 'val', field(8, struct_ibv_flow_gre_filter))
setattr(struct_ibv_flow_spec_gre, 'mask', field(16, struct_ibv_flow_gre_filter))
class struct_ibv_flow_mpls_filter(Struct): pass
struct_ibv_flow_mpls_filter.SIZE = 4
struct_ibv_flow_mpls_filter._fields_ = ['label']
setattr(struct_ibv_flow_mpls_filter, 'label', field(0, uint32_t))
class struct_ibv_flow_spec_mpls(Struct): pass
struct_ibv_flow_spec_mpls.SIZE = 16
struct_ibv_flow_spec_mpls._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_mpls, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_mpls, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_mpls, 'val', field(8, struct_ibv_flow_mpls_filter))
setattr(struct_ibv_flow_spec_mpls, 'mask', field(12, struct_ibv_flow_mpls_filter))
class struct_ibv_flow_tunnel_filter(Struct): pass
struct_ibv_flow_tunnel_filter.SIZE = 4
struct_ibv_flow_tunnel_filter._fields_ = ['tunnel_id']
setattr(struct_ibv_flow_tunnel_filter, 'tunnel_id', field(0, uint32_t))
class struct_ibv_flow_spec_tunnel(Struct): pass
struct_ibv_flow_spec_tunnel.SIZE = 16
struct_ibv_flow_spec_tunnel._fields_ = ['type', 'size', 'val', 'mask']
setattr(struct_ibv_flow_spec_tunnel, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_tunnel, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_tunnel, 'val', field(8, struct_ibv_flow_tunnel_filter))
setattr(struct_ibv_flow_spec_tunnel, 'mask', field(12, struct_ibv_flow_tunnel_filter))
class struct_ibv_flow_spec_action_tag(Struct): pass
struct_ibv_flow_spec_action_tag.SIZE = 12
struct_ibv_flow_spec_action_tag._fields_ = ['type', 'size', 'tag_id']
setattr(struct_ibv_flow_spec_action_tag, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_action_tag, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_action_tag, 'tag_id', field(8, uint32_t))
class struct_ibv_flow_spec_action_drop(Struct): pass
struct_ibv_flow_spec_action_drop.SIZE = 8
struct_ibv_flow_spec_action_drop._fields_ = ['type', 'size']
setattr(struct_ibv_flow_spec_action_drop, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_action_drop, 'size', field(4, uint16_t))
class struct_ibv_flow_spec_action_handle(Struct): pass
class struct_ibv_flow_action(Struct): pass
struct_ibv_flow_action.SIZE = 8
struct_ibv_flow_action._fields_ = ['context']
setattr(struct_ibv_flow_action, 'context', field(0, Pointer(struct_ibv_context)))
struct_ibv_flow_spec_action_handle.SIZE = 16
struct_ibv_flow_spec_action_handle._fields_ = ['type', 'size', 'action']
setattr(struct_ibv_flow_spec_action_handle, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_action_handle, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_action_handle, 'action', field(8, Pointer(struct_ibv_flow_action)))
class struct_ibv_flow_spec_counter_action(Struct): pass
class struct_ibv_counters(Struct): pass
struct_ibv_counters.SIZE = 8
struct_ibv_counters._fields_ = ['context']
setattr(struct_ibv_counters, 'context', field(0, Pointer(struct_ibv_context)))
struct_ibv_flow_spec_counter_action.SIZE = 16
struct_ibv_flow_spec_counter_action._fields_ = ['type', 'size', 'counters']
setattr(struct_ibv_flow_spec_counter_action, 'type', field(0, enum_ibv_flow_spec_type))
setattr(struct_ibv_flow_spec_counter_action, 'size', field(4, uint16_t))
setattr(struct_ibv_flow_spec_counter_action, 'counters', field(8, Pointer(struct_ibv_counters)))
class struct_ibv_flow_spec(Struct): pass
class _anonstruct18(Struct): pass
_anonstruct18.SIZE = 8
_anonstruct18._fields_ = ['type', 'size']
setattr(_anonstruct18, 'type', field(0, enum_ibv_flow_spec_type))
setattr(_anonstruct18, 'size', field(4, uint16_t))
struct_ibv_flow_spec.SIZE = 88
struct_ibv_flow_spec._fields_ = ['hdr', 'eth', 'ipv4', 'tcp_udp', 'ipv4_ext', 'ipv6', 'esp', 'tunnel', 'gre', 'mpls', 'flow_tag', 'drop', 'handle', 'flow_count']
setattr(struct_ibv_flow_spec, 'hdr', field(0, _anonstruct18))
setattr(struct_ibv_flow_spec, 'eth', field(0, struct_ibv_flow_spec_eth))
setattr(struct_ibv_flow_spec, 'ipv4', field(0, struct_ibv_flow_spec_ipv4))
setattr(struct_ibv_flow_spec, 'tcp_udp', field(0, struct_ibv_flow_spec_tcp_udp))
setattr(struct_ibv_flow_spec, 'ipv4_ext', field(0, struct_ibv_flow_spec_ipv4_ext))
setattr(struct_ibv_flow_spec, 'ipv6', field(0, struct_ibv_flow_spec_ipv6))
setattr(struct_ibv_flow_spec, 'esp', field(0, struct_ibv_flow_spec_esp))
setattr(struct_ibv_flow_spec, 'tunnel', field(0, struct_ibv_flow_spec_tunnel))
setattr(struct_ibv_flow_spec, 'gre', field(0, struct_ibv_flow_spec_gre))
setattr(struct_ibv_flow_spec, 'mpls', field(0, struct_ibv_flow_spec_mpls))
setattr(struct_ibv_flow_spec, 'flow_tag', field(0, struct_ibv_flow_spec_action_tag))
setattr(struct_ibv_flow_spec, 'drop', field(0, struct_ibv_flow_spec_action_drop))
setattr(struct_ibv_flow_spec, 'handle', field(0, struct_ibv_flow_spec_action_handle))
setattr(struct_ibv_flow_spec, 'flow_count', field(0, struct_ibv_flow_spec_counter_action))
class struct_ibv_flow_attr(Struct): pass
struct_ibv_flow_attr.SIZE = 20
struct_ibv_flow_attr._fields_ = ['comp_mask', 'type', 'size', 'priority', 'num_of_specs', 'port', 'flags']
setattr(struct_ibv_flow_attr, 'comp_mask', field(0, uint32_t))
setattr(struct_ibv_flow_attr, 'type', field(4, enum_ibv_flow_attr_type))
setattr(struct_ibv_flow_attr, 'size', field(8, uint16_t))
setattr(struct_ibv_flow_attr, 'priority', field(10, uint16_t))
setattr(struct_ibv_flow_attr, 'num_of_specs', field(12, uint8_t))
setattr(struct_ibv_flow_attr, 'port', field(13, uint8_t))
setattr(struct_ibv_flow_attr, 'flags', field(16, uint32_t))
class struct_ibv_flow(Struct): pass
struct_ibv_flow.SIZE = 24
struct_ibv_flow._fields_ = ['comp_mask', 'context', 'handle']
setattr(struct_ibv_flow, 'comp_mask', field(0, uint32_t))
setattr(struct_ibv_flow, 'context', field(8, Pointer(struct_ibv_context)))
setattr(struct_ibv_flow, 'handle', field(16, uint32_t))
enum_ibv_flow_action_esp_mask = CEnum(ctypes.c_uint32)
IBV_FLOW_ACTION_ESP_MASK_ESN = enum_ibv_flow_action_esp_mask.define('IBV_FLOW_ACTION_ESP_MASK_ESN', 1)

class struct_ibv_flow_action_esp_attr(Struct): pass
class struct_ib_uverbs_flow_action_esp(Struct): pass
__u32 = ctypes.c_uint32
__u64 = ctypes.c_uint64
struct_ib_uverbs_flow_action_esp.SIZE = 24
struct_ib_uverbs_flow_action_esp._fields_ = ['spi', 'seq', 'tfc_pad', 'flags', 'hard_limit_pkts']
setattr(struct_ib_uverbs_flow_action_esp, 'spi', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_action_esp, 'seq', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_action_esp, 'tfc_pad', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_action_esp, 'flags', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_action_esp, 'hard_limit_pkts', field(16, ctypes.c_uint64))
enum_ib_uverbs_flow_action_esp_keymat = CEnum(ctypes.c_uint32)
IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM = enum_ib_uverbs_flow_action_esp_keymat.define('IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM', 0)

enum_ib_uverbs_flow_action_esp_replay = CEnum(ctypes.c_uint32)
IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE = enum_ib_uverbs_flow_action_esp_replay.define('IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE', 0)
IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP = enum_ib_uverbs_flow_action_esp_replay.define('IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP', 1)

class struct_ib_uverbs_flow_action_esp_encap(Struct): pass
__u16 = ctypes.c_uint16
struct_ib_uverbs_flow_action_esp_encap.SIZE = 24
struct_ib_uverbs_flow_action_esp_encap._fields_ = ['val_ptr', 'val_ptr_data_u64', 'next_ptr', 'next_ptr_data_u64', 'len', 'type']
setattr(struct_ib_uverbs_flow_action_esp_encap, 'val_ptr', field(0, ctypes.c_void_p))
setattr(struct_ib_uverbs_flow_action_esp_encap, 'val_ptr_data_u64', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_flow_action_esp_encap, 'next_ptr', field(8, Pointer(struct_ib_uverbs_flow_action_esp_encap)))
setattr(struct_ib_uverbs_flow_action_esp_encap, 'next_ptr_data_u64', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_flow_action_esp_encap, 'len', field(16, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_action_esp_encap, 'type', field(18, ctypes.c_uint16))
struct_ibv_flow_action_esp_attr.SIZE = 56
struct_ibv_flow_action_esp_attr._fields_ = ['esp_attr', 'keymat_proto', 'keymat_len', 'keymat_ptr', 'replay_proto', 'replay_len', 'replay_ptr', 'esp_encap', 'comp_mask', 'esn']
setattr(struct_ibv_flow_action_esp_attr, 'esp_attr', field(0, Pointer(struct_ib_uverbs_flow_action_esp)))
setattr(struct_ibv_flow_action_esp_attr, 'keymat_proto', field(8, enum_ib_uverbs_flow_action_esp_keymat))
setattr(struct_ibv_flow_action_esp_attr, 'keymat_len', field(12, uint16_t))
setattr(struct_ibv_flow_action_esp_attr, 'keymat_ptr', field(16, ctypes.c_void_p))
setattr(struct_ibv_flow_action_esp_attr, 'replay_proto', field(24, enum_ib_uverbs_flow_action_esp_replay))
setattr(struct_ibv_flow_action_esp_attr, 'replay_len', field(28, uint16_t))
setattr(struct_ibv_flow_action_esp_attr, 'replay_ptr', field(32, ctypes.c_void_p))
setattr(struct_ibv_flow_action_esp_attr, 'esp_encap', field(40, Pointer(struct_ib_uverbs_flow_action_esp_encap)))
setattr(struct_ibv_flow_action_esp_attr, 'comp_mask', field(48, uint32_t))
setattr(struct_ibv_flow_action_esp_attr, 'esn', field(52, uint32_t))
_anonenum19 = CEnum(ctypes.c_uint32)
IBV_SYSFS_NAME_MAX = _anonenum19.define('IBV_SYSFS_NAME_MAX', 64)
IBV_SYSFS_PATH_MAX = _anonenum19.define('IBV_SYSFS_PATH_MAX', 256)

enum_ibv_cq_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_CQ_INIT_ATTR_MASK_FLAGS = enum_ibv_cq_init_attr_mask.define('IBV_CQ_INIT_ATTR_MASK_FLAGS', 1)
IBV_CQ_INIT_ATTR_MASK_PD = enum_ibv_cq_init_attr_mask.define('IBV_CQ_INIT_ATTR_MASK_PD', 2)

enum_ibv_create_cq_attr_flags = CEnum(ctypes.c_uint32)
IBV_CREATE_CQ_ATTR_SINGLE_THREADED = enum_ibv_create_cq_attr_flags.define('IBV_CREATE_CQ_ATTR_SINGLE_THREADED', 1)
IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN = enum_ibv_create_cq_attr_flags.define('IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN', 2)

class struct_ibv_cq_init_attr_ex(Struct): pass
struct_ibv_cq_init_attr_ex.SIZE = 56
struct_ibv_cq_init_attr_ex._fields_ = ['cqe', 'cq_context', 'channel', 'comp_vector', 'wc_flags', 'comp_mask', 'flags', 'parent_domain']
setattr(struct_ibv_cq_init_attr_ex, 'cqe', field(0, uint32_t))
setattr(struct_ibv_cq_init_attr_ex, 'cq_context', field(8, ctypes.c_void_p))
setattr(struct_ibv_cq_init_attr_ex, 'channel', field(16, Pointer(struct_ibv_comp_channel)))
setattr(struct_ibv_cq_init_attr_ex, 'comp_vector', field(24, uint32_t))
setattr(struct_ibv_cq_init_attr_ex, 'wc_flags', field(32, uint64_t))
setattr(struct_ibv_cq_init_attr_ex, 'comp_mask', field(40, uint32_t))
setattr(struct_ibv_cq_init_attr_ex, 'flags', field(44, uint32_t))
setattr(struct_ibv_cq_init_attr_ex, 'parent_domain', field(48, Pointer(struct_ibv_pd)))
enum_ibv_parent_domain_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS = enum_ibv_parent_domain_init_attr_mask.define('IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS', 1)
IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT = enum_ibv_parent_domain_init_attr_mask.define('IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT', 2)

class struct_ibv_parent_domain_init_attr(Struct): pass
struct_ibv_parent_domain_init_attr.SIZE = 48
struct_ibv_parent_domain_init_attr._fields_ = ['pd', 'td', 'comp_mask', 'alloc', 'free', 'pd_context']
setattr(struct_ibv_parent_domain_init_attr, 'pd', field(0, Pointer(struct_ibv_pd)))
setattr(struct_ibv_parent_domain_init_attr, 'td', field(8, Pointer(struct_ibv_td)))
setattr(struct_ibv_parent_domain_init_attr, 'comp_mask', field(16, uint32_t))
setattr(struct_ibv_parent_domain_init_attr, 'alloc', field(24, ctypes.CFUNCTYPE(ctypes.c_void_p, Pointer(struct_ibv_pd), ctypes.c_void_p, size_t, size_t, uint64_t)))
setattr(struct_ibv_parent_domain_init_attr, 'free', field(32, ctypes.CFUNCTYPE(None, Pointer(struct_ibv_pd), ctypes.c_void_p, ctypes.c_void_p, uint64_t)))
setattr(struct_ibv_parent_domain_init_attr, 'pd_context', field(40, ctypes.c_void_p))
class struct_ibv_counters_init_attr(Struct): pass
struct_ibv_counters_init_attr.SIZE = 4
struct_ibv_counters_init_attr._fields_ = ['comp_mask']
setattr(struct_ibv_counters_init_attr, 'comp_mask', field(0, uint32_t))
enum_ibv_counter_description = CEnum(ctypes.c_uint32)
IBV_COUNTER_PACKETS = enum_ibv_counter_description.define('IBV_COUNTER_PACKETS', 0)
IBV_COUNTER_BYTES = enum_ibv_counter_description.define('IBV_COUNTER_BYTES', 1)

class struct_ibv_counter_attach_attr(Struct): pass
struct_ibv_counter_attach_attr.SIZE = 12
struct_ibv_counter_attach_attr._fields_ = ['counter_desc', 'index', 'comp_mask']
setattr(struct_ibv_counter_attach_attr, 'counter_desc', field(0, enum_ibv_counter_description))
setattr(struct_ibv_counter_attach_attr, 'index', field(4, uint32_t))
setattr(struct_ibv_counter_attach_attr, 'comp_mask', field(8, uint32_t))
enum_ibv_read_counters_flags = CEnum(ctypes.c_uint32)
IBV_READ_COUNTERS_ATTR_PREFER_CACHED = enum_ibv_read_counters_flags.define('IBV_READ_COUNTERS_ATTR_PREFER_CACHED', 1)

enum_ibv_values_mask = CEnum(ctypes.c_uint32)
IBV_VALUES_MASK_RAW_CLOCK = enum_ibv_values_mask.define('IBV_VALUES_MASK_RAW_CLOCK', 1)
IBV_VALUES_MASK_RESERVED = enum_ibv_values_mask.define('IBV_VALUES_MASK_RESERVED', 2)

class struct_ibv_values_ex(Struct): pass
class struct_timespec(Struct): pass
__time_t = ctypes.c_int64
__syscall_slong_t = ctypes.c_int64
struct_timespec.SIZE = 16
struct_timespec._fields_ = ['tv_sec', 'tv_nsec']
setattr(struct_timespec, 'tv_sec', field(0, ctypes.c_int64))
setattr(struct_timespec, 'tv_nsec', field(8, ctypes.c_int64))
struct_ibv_values_ex.SIZE = 24
struct_ibv_values_ex._fields_ = ['comp_mask', 'raw_clock']
setattr(struct_ibv_values_ex, 'comp_mask', field(0, uint32_t))
setattr(struct_ibv_values_ex, 'raw_clock', field(8, struct_timespec))
class struct_verbs_context(Struct): pass
enum_ib_uverbs_advise_mr_advice = CEnum(ctypes.c_uint32)
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH', 0)
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE', 1)
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT', 2)

class struct_verbs_ex_private(Struct): pass
struct_verbs_context.SIZE = 648
struct_verbs_context._fields_ = ['query_port', 'advise_mr', 'alloc_null_mr', 'read_counters', 'attach_counters_point_flow', 'create_counters', 'destroy_counters', 'reg_dm_mr', 'alloc_dm', 'free_dm', 'modify_flow_action_esp', 'destroy_flow_action', 'create_flow_action_esp', 'modify_qp_rate_limit', 'alloc_parent_domain', 'dealloc_td', 'alloc_td', 'modify_cq', 'post_srq_ops', 'destroy_rwq_ind_table', 'create_rwq_ind_table', 'destroy_wq', 'modify_wq', 'create_wq', 'query_rt_values', 'create_cq_ex', 'priv', 'query_device_ex', 'ibv_destroy_flow', 'ABI_placeholder2', 'ibv_create_flow', 'ABI_placeholder1', 'open_qp', 'create_qp_ex', 'get_srq_num', 'create_srq_ex', 'open_xrcd', 'close_xrcd', '_ABI_placeholder3', 'sz', 'context']
setattr(struct_verbs_context, 'query_port', field(0, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_context), uint8_t, Pointer(struct_ibv_port_attr), size_t)))
setattr(struct_verbs_context, 'advise_mr', field(8, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_pd), enum_ib_uverbs_advise_mr_advice, uint32_t, Pointer(struct_ibv_sge), uint32_t)))
setattr(struct_verbs_context, 'alloc_null_mr', field(16, ctypes.CFUNCTYPE(Pointer(struct_ibv_mr), Pointer(struct_ibv_pd))))
setattr(struct_verbs_context, 'read_counters', field(24, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_counters), Pointer(uint64_t), uint32_t, uint32_t)))
setattr(struct_verbs_context, 'attach_counters_point_flow', field(32, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_counters), Pointer(struct_ibv_counter_attach_attr), Pointer(struct_ibv_flow))))
setattr(struct_verbs_context, 'create_counters', field(40, ctypes.CFUNCTYPE(Pointer(struct_ibv_counters), Pointer(struct_ibv_context), Pointer(struct_ibv_counters_init_attr))))
setattr(struct_verbs_context, 'destroy_counters', field(48, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_counters))))
setattr(struct_verbs_context, 'reg_dm_mr', field(56, ctypes.CFUNCTYPE(Pointer(struct_ibv_mr), Pointer(struct_ibv_pd), Pointer(struct_ibv_dm), uint64_t, size_t, ctypes.c_uint32)))
setattr(struct_verbs_context, 'alloc_dm', field(64, ctypes.CFUNCTYPE(Pointer(struct_ibv_dm), Pointer(struct_ibv_context), Pointer(struct_ibv_alloc_dm_attr))))
setattr(struct_verbs_context, 'free_dm', field(72, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_dm))))
setattr(struct_verbs_context, 'modify_flow_action_esp', field(80, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_flow_action), Pointer(struct_ibv_flow_action_esp_attr))))
setattr(struct_verbs_context, 'destroy_flow_action', field(88, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_flow_action))))
setattr(struct_verbs_context, 'create_flow_action_esp', field(96, ctypes.CFUNCTYPE(Pointer(struct_ibv_flow_action), Pointer(struct_ibv_context), Pointer(struct_ibv_flow_action_esp_attr))))
setattr(struct_verbs_context, 'modify_qp_rate_limit', field(104, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_qp), Pointer(struct_ibv_qp_rate_limit_attr))))
setattr(struct_verbs_context, 'alloc_parent_domain', field(112, ctypes.CFUNCTYPE(Pointer(struct_ibv_pd), Pointer(struct_ibv_context), Pointer(struct_ibv_parent_domain_init_attr))))
setattr(struct_verbs_context, 'dealloc_td', field(120, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_td))))
setattr(struct_verbs_context, 'alloc_td', field(128, ctypes.CFUNCTYPE(Pointer(struct_ibv_td), Pointer(struct_ibv_context), Pointer(struct_ibv_td_init_attr))))
setattr(struct_verbs_context, 'modify_cq', field(136, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_cq), Pointer(struct_ibv_modify_cq_attr))))
setattr(struct_verbs_context, 'post_srq_ops', field(144, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_srq), Pointer(struct_ibv_ops_wr), Pointer(Pointer(struct_ibv_ops_wr)))))
setattr(struct_verbs_context, 'destroy_rwq_ind_table', field(152, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_rwq_ind_table))))
setattr(struct_verbs_context, 'create_rwq_ind_table', field(160, ctypes.CFUNCTYPE(Pointer(struct_ibv_rwq_ind_table), Pointer(struct_ibv_context), Pointer(struct_ibv_rwq_ind_table_init_attr))))
setattr(struct_verbs_context, 'destroy_wq', field(168, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_wq))))
setattr(struct_verbs_context, 'modify_wq', field(176, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_wq), Pointer(struct_ibv_wq_attr))))
setattr(struct_verbs_context, 'create_wq', field(184, ctypes.CFUNCTYPE(Pointer(struct_ibv_wq), Pointer(struct_ibv_context), Pointer(struct_ibv_wq_init_attr))))
setattr(struct_verbs_context, 'query_rt_values', field(192, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_context), Pointer(struct_ibv_values_ex))))
setattr(struct_verbs_context, 'create_cq_ex', field(200, ctypes.CFUNCTYPE(Pointer(struct_ibv_cq_ex), Pointer(struct_ibv_context), Pointer(struct_ibv_cq_init_attr_ex))))
setattr(struct_verbs_context, 'priv', field(208, Pointer(struct_verbs_ex_private)))
setattr(struct_verbs_context, 'query_device_ex', field(216, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_context), Pointer(struct_ibv_query_device_ex_input), Pointer(struct_ibv_device_attr_ex), size_t)))
setattr(struct_verbs_context, 'ibv_destroy_flow', field(224, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_flow))))
setattr(struct_verbs_context, 'ABI_placeholder2', field(232, ctypes.CFUNCTYPE(None, )))
setattr(struct_verbs_context, 'ibv_create_flow', field(240, ctypes.CFUNCTYPE(Pointer(struct_ibv_flow), Pointer(struct_ibv_qp), Pointer(struct_ibv_flow_attr))))
setattr(struct_verbs_context, 'ABI_placeholder1', field(248, ctypes.CFUNCTYPE(None, )))
setattr(struct_verbs_context, 'open_qp', field(256, ctypes.CFUNCTYPE(Pointer(struct_ibv_qp), Pointer(struct_ibv_context), Pointer(struct_ibv_qp_open_attr))))
setattr(struct_verbs_context, 'create_qp_ex', field(264, ctypes.CFUNCTYPE(Pointer(struct_ibv_qp), Pointer(struct_ibv_context), Pointer(struct_ibv_qp_init_attr_ex))))
setattr(struct_verbs_context, 'get_srq_num', field(272, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_srq), Pointer(uint32_t))))
setattr(struct_verbs_context, 'create_srq_ex', field(280, ctypes.CFUNCTYPE(Pointer(struct_ibv_srq), Pointer(struct_ibv_context), Pointer(struct_ibv_srq_init_attr_ex))))
setattr(struct_verbs_context, 'open_xrcd', field(288, ctypes.CFUNCTYPE(Pointer(struct_ibv_xrcd), Pointer(struct_ibv_context), Pointer(struct_ibv_xrcd_init_attr))))
setattr(struct_verbs_context, 'close_xrcd', field(296, ctypes.CFUNCTYPE(ctypes.c_int32, Pointer(struct_ibv_xrcd))))
setattr(struct_verbs_context, '_ABI_placeholder3', field(304, uint64_t))
setattr(struct_verbs_context, 'sz', field(312, size_t))
setattr(struct_verbs_context, 'context', field(320, struct_ibv_context))
@dll.bind((Pointer(ctypes.c_int32)), Pointer(Pointer(struct_ibv_device)))
def ibv_get_device_list(num_devices): ...
@dll.bind((Pointer(Pointer(struct_ibv_device))), None)
def ibv_free_device_list(list): ...
@dll.bind((Pointer(struct_ibv_device)), Pointer(ctypes.c_char))
def ibv_get_device_name(device): ...
@dll.bind((Pointer(struct_ibv_device)), ctypes.c_int32)
def ibv_get_device_index(device): ...
@dll.bind((Pointer(struct_ibv_device)), ctypes.c_uint64)
def ibv_get_device_guid(device): ...
@dll.bind((Pointer(struct_ibv_device)), Pointer(struct_ibv_context))
def ibv_open_device(device): ...
@dll.bind((Pointer(struct_ibv_context)), ctypes.c_int32)
def ibv_close_device(context): ...
@dll.bind((ctypes.c_int32), Pointer(struct_ibv_context))
def ibv_import_device(cmd_fd): ...
@dll.bind((Pointer(struct_ibv_context), uint32_t), Pointer(struct_ibv_pd))
def ibv_import_pd(context, pd_handle): ...
@dll.bind((Pointer(struct_ibv_pd)), None)
def ibv_unimport_pd(pd): ...
@dll.bind((Pointer(struct_ibv_pd), uint32_t), Pointer(struct_ibv_mr))
def ibv_import_mr(pd, mr_handle): ...
@dll.bind((Pointer(struct_ibv_mr)), None)
def ibv_unimport_mr(mr): ...
@dll.bind((Pointer(struct_ibv_context), uint32_t), Pointer(struct_ibv_dm))
def ibv_import_dm(context, dm_handle): ...
@dll.bind((Pointer(struct_ibv_dm)), None)
def ibv_unimport_dm(dm): ...
@dll.bind((Pointer(struct_ibv_context), Pointer(struct_ibv_async_event)), ctypes.c_int32)
def ibv_get_async_event(context, event): ...
@dll.bind((Pointer(struct_ibv_async_event)), None)
def ibv_ack_async_event(event): ...
@dll.bind((Pointer(struct_ibv_context), Pointer(struct_ibv_device_attr)), ctypes.c_int32)
def ibv_query_device(context, device_attr): ...
@dll.bind((Pointer(struct_ibv_context), uint8_t, Pointer(struct__compat_ibv_port_attr)), ctypes.c_int32)
def ibv_query_port(context, port_num, port_attr): ...
@dll.bind((Pointer(struct_ibv_context), uint8_t, ctypes.c_int32, Pointer(union_ibv_gid)), ctypes.c_int32)
def ibv_query_gid(context, port_num, index, gid): ...
@dll.bind((Pointer(struct_ibv_context), uint32_t, uint32_t, Pointer(struct_ibv_gid_entry), uint32_t, size_t), ctypes.c_int32)
def _ibv_query_gid_ex(context, port_num, gid_index, entry, flags, entry_size): ...
ssize_t = ctypes.c_int64
@dll.bind((Pointer(struct_ibv_context), Pointer(struct_ibv_gid_entry), size_t, uint32_t, size_t), ssize_t)
def _ibv_query_gid_table(context, entries, max_entries, flags, entry_size): ...
@dll.bind((Pointer(struct_ibv_context), uint8_t, ctypes.c_int32, Pointer(ctypes.c_uint16)), ctypes.c_int32)
def ibv_query_pkey(context, port_num, index, pkey): ...
@dll.bind((Pointer(struct_ibv_context), uint8_t, ctypes.c_uint16), ctypes.c_int32)
def ibv_get_pkey_index(context, port_num, pkey): ...
@dll.bind((Pointer(struct_ibv_context)), Pointer(struct_ibv_pd))
def ibv_alloc_pd(context): ...
@dll.bind((Pointer(struct_ibv_pd)), ctypes.c_int32)
def ibv_dealloc_pd(pd): ...
@dll.bind((Pointer(struct_ibv_pd), ctypes.c_void_p, size_t, uint64_t, ctypes.c_uint32), Pointer(struct_ibv_mr))
def ibv_reg_mr_iova2(pd, addr, length, iova, access): ...
@dll.bind((Pointer(struct_ibv_pd), ctypes.c_void_p, size_t, ctypes.c_int32), Pointer(struct_ibv_mr))
def ibv_reg_mr(pd, addr, length, access): ...
@dll.bind((Pointer(struct_ibv_pd), ctypes.c_void_p, size_t, uint64_t, ctypes.c_int32), Pointer(struct_ibv_mr))
def ibv_reg_mr_iova(pd, addr, length, iova, access): ...
@dll.bind((Pointer(struct_ibv_pd), uint64_t, size_t, uint64_t, ctypes.c_int32, ctypes.c_int32), Pointer(struct_ibv_mr))
def ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access): ...
enum_ibv_rereg_mr_err_code = CEnum(ctypes.c_int32)
IBV_REREG_MR_ERR_INPUT = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_INPUT', -1)
IBV_REREG_MR_ERR_DONT_FORK_NEW = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_DONT_FORK_NEW', -2)
IBV_REREG_MR_ERR_DO_FORK_OLD = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_DO_FORK_OLD', -3)
IBV_REREG_MR_ERR_CMD = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_CMD', -4)
IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW', -5)

@dll.bind((Pointer(struct_ibv_mr), ctypes.c_int32, Pointer(struct_ibv_pd), ctypes.c_void_p, size_t, ctypes.c_int32), ctypes.c_int32)
def ibv_rereg_mr(mr, flags, pd, addr, length, access): ...
@dll.bind((Pointer(struct_ibv_mr)), ctypes.c_int32)
def ibv_dereg_mr(mr): ...
@dll.bind((Pointer(struct_ibv_context)), Pointer(struct_ibv_comp_channel))
def ibv_create_comp_channel(context): ...
@dll.bind((Pointer(struct_ibv_comp_channel)), ctypes.c_int32)
def ibv_destroy_comp_channel(channel): ...
@dll.bind((Pointer(struct_ibv_context), ctypes.c_int32, ctypes.c_void_p, Pointer(struct_ibv_comp_channel), ctypes.c_int32), Pointer(struct_ibv_cq))
def ibv_create_cq(context, cqe, cq_context, channel, comp_vector): ...
@dll.bind((Pointer(struct_ibv_cq), ctypes.c_int32), ctypes.c_int32)
def ibv_resize_cq(cq, cqe): ...
@dll.bind((Pointer(struct_ibv_cq)), ctypes.c_int32)
def ibv_destroy_cq(cq): ...
@dll.bind((Pointer(struct_ibv_comp_channel), Pointer(Pointer(struct_ibv_cq)), Pointer(ctypes.c_void_p)), ctypes.c_int32)
def ibv_get_cq_event(channel, cq, cq_context): ...
@dll.bind((Pointer(struct_ibv_cq), ctypes.c_uint32), None)
def ibv_ack_cq_events(cq, nevents): ...
@dll.bind((Pointer(struct_ibv_pd), Pointer(struct_ibv_srq_init_attr)), Pointer(struct_ibv_srq))
def ibv_create_srq(pd, srq_init_attr): ...
@dll.bind((Pointer(struct_ibv_srq), Pointer(struct_ibv_srq_attr), ctypes.c_int32), ctypes.c_int32)
def ibv_modify_srq(srq, srq_attr, srq_attr_mask): ...
@dll.bind((Pointer(struct_ibv_srq), Pointer(struct_ibv_srq_attr)), ctypes.c_int32)
def ibv_query_srq(srq, srq_attr): ...
@dll.bind((Pointer(struct_ibv_srq)), ctypes.c_int32)
def ibv_destroy_srq(srq): ...
@dll.bind((Pointer(struct_ibv_pd), Pointer(struct_ibv_qp_init_attr)), Pointer(struct_ibv_qp))
def ibv_create_qp(pd, qp_init_attr): ...
@dll.bind((Pointer(struct_ibv_qp), Pointer(struct_ibv_qp_attr), ctypes.c_int32), ctypes.c_int32)
def ibv_modify_qp(qp, attr, attr_mask): ...
@dll.bind((Pointer(struct_ibv_qp), enum_ibv_wr_opcode, uint32_t), ctypes.c_int32)
def ibv_query_qp_data_in_order(qp, op, flags): ...
@dll.bind((Pointer(struct_ibv_qp), Pointer(struct_ibv_qp_attr), ctypes.c_int32, Pointer(struct_ibv_qp_init_attr)), ctypes.c_int32)
def ibv_query_qp(qp, attr, attr_mask, init_attr): ...
@dll.bind((Pointer(struct_ibv_qp)), ctypes.c_int32)
def ibv_destroy_qp(qp): ...
@dll.bind((Pointer(struct_ibv_pd), Pointer(struct_ibv_ah_attr)), Pointer(struct_ibv_ah))
def ibv_create_ah(pd, attr): ...
@dll.bind((Pointer(struct_ibv_context), uint8_t, Pointer(struct_ibv_wc), Pointer(struct_ibv_grh), Pointer(struct_ibv_ah_attr)), ctypes.c_int32)
def ibv_init_ah_from_wc(context, port_num, wc, grh, ah_attr): ...
@dll.bind((Pointer(struct_ibv_pd), Pointer(struct_ibv_wc), Pointer(struct_ibv_grh), uint8_t), Pointer(struct_ibv_ah))
def ibv_create_ah_from_wc(pd, wc, grh, port_num): ...
@dll.bind((Pointer(struct_ibv_ah)), ctypes.c_int32)
def ibv_destroy_ah(ah): ...
@dll.bind((Pointer(struct_ibv_qp), Pointer(union_ibv_gid), uint16_t), ctypes.c_int32)
def ibv_attach_mcast(qp, gid, lid): ...
@dll.bind((Pointer(struct_ibv_qp), Pointer(union_ibv_gid), uint16_t), ctypes.c_int32)
def ibv_detach_mcast(qp, gid, lid): ...
@dll.bind((), ctypes.c_int32)
def ibv_fork_init(): ...
@dll.bind((), enum_ibv_fork_status)
def ibv_is_fork_initialized(): ...
@dll.bind((enum_ibv_node_type), Pointer(ctypes.c_char))
def ibv_node_type_str(node_type): ...
@dll.bind((enum_ibv_port_state), Pointer(ctypes.c_char))
def ibv_port_state_str(port_state): ...
@dll.bind((enum_ibv_event_type), Pointer(ctypes.c_char))
def ibv_event_type_str(event): ...
@dll.bind((Pointer(struct_ibv_context), Pointer(struct_ibv_ah_attr), Array(uint8_t, 6), Pointer(uint16_t)), ctypes.c_int32)
def ibv_resolve_eth_l2_from_gid(context, attr, eth_mac, vid): ...
@dll.bind((Pointer(struct_ibv_qp), Pointer(struct_ibv_ece)), ctypes.c_int32)
def ibv_set_ece(qp, ece): ...
@dll.bind((Pointer(struct_ibv_qp), Pointer(struct_ibv_ece)), ctypes.c_int32)
def ibv_query_ece(qp, ece): ...
enum_ib_uverbs_core_support = CEnum(ctypes.c_uint32)
IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS = enum_ib_uverbs_core_support.define('IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS', 1)

enum_ib_uverbs_access_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_ACCESS_LOCAL_WRITE = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_LOCAL_WRITE', 1)
IB_UVERBS_ACCESS_REMOTE_WRITE = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_REMOTE_WRITE', 2)
IB_UVERBS_ACCESS_REMOTE_READ = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_REMOTE_READ', 4)
IB_UVERBS_ACCESS_REMOTE_ATOMIC = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_REMOTE_ATOMIC', 8)
IB_UVERBS_ACCESS_MW_BIND = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_MW_BIND', 16)
IB_UVERBS_ACCESS_ZERO_BASED = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_ZERO_BASED', 32)
IB_UVERBS_ACCESS_ON_DEMAND = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_ON_DEMAND', 64)
IB_UVERBS_ACCESS_HUGETLB = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_HUGETLB', 128)
IB_UVERBS_ACCESS_FLUSH_GLOBAL = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_FLUSH_GLOBAL', 256)
IB_UVERBS_ACCESS_FLUSH_PERSISTENT = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_FLUSH_PERSISTENT', 512)
IB_UVERBS_ACCESS_RELAXED_ORDERING = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_RELAXED_ORDERING', 1048576)
IB_UVERBS_ACCESS_OPTIONAL_RANGE = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_OPTIONAL_RANGE', 1072693248)

enum_ib_uverbs_srq_type = CEnum(ctypes.c_uint32)
IB_UVERBS_SRQT_BASIC = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_BASIC', 0)
IB_UVERBS_SRQT_XRC = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_XRC', 1)
IB_UVERBS_SRQT_TM = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_TM', 2)

enum_ib_uverbs_wq_type = CEnum(ctypes.c_uint32)
IB_UVERBS_WQT_RQ = enum_ib_uverbs_wq_type.define('IB_UVERBS_WQT_RQ', 0)

enum_ib_uverbs_wq_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING', 1)
IB_UVERBS_WQ_FLAGS_SCATTER_FCS = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_SCATTER_FCS', 2)
IB_UVERBS_WQ_FLAGS_DELAY_DROP = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_DELAY_DROP', 4)
IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING', 8)

enum_ib_uverbs_qp_type = CEnum(ctypes.c_uint32)
IB_UVERBS_QPT_RC = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_RC', 2)
IB_UVERBS_QPT_UC = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_UC', 3)
IB_UVERBS_QPT_UD = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_UD', 4)
IB_UVERBS_QPT_RAW_PACKET = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_RAW_PACKET', 8)
IB_UVERBS_QPT_XRC_INI = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_XRC_INI', 9)
IB_UVERBS_QPT_XRC_TGT = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_XRC_TGT', 10)
IB_UVERBS_QPT_DRIVER = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_DRIVER', 255)

enum_ib_uverbs_qp_create_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK', 2)
IB_UVERBS_QP_CREATE_SCATTER_FCS = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_SCATTER_FCS', 256)
IB_UVERBS_QP_CREATE_CVLAN_STRIPPING = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_CVLAN_STRIPPING', 512)
IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING', 2048)
IB_UVERBS_QP_CREATE_SQ_SIG_ALL = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_SQ_SIG_ALL', 4096)

enum_ib_uverbs_query_port_cap_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_PCF_SM = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SM', 2)
IB_UVERBS_PCF_NOTICE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_NOTICE_SUP', 4)
IB_UVERBS_PCF_TRAP_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_TRAP_SUP', 8)
IB_UVERBS_PCF_OPT_IPD_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_OPT_IPD_SUP', 16)
IB_UVERBS_PCF_AUTO_MIGR_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_AUTO_MIGR_SUP', 32)
IB_UVERBS_PCF_SL_MAP_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SL_MAP_SUP', 64)
IB_UVERBS_PCF_MKEY_NVRAM = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_MKEY_NVRAM', 128)
IB_UVERBS_PCF_PKEY_NVRAM = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_PKEY_NVRAM', 256)
IB_UVERBS_PCF_LED_INFO_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_LED_INFO_SUP', 512)
IB_UVERBS_PCF_SM_DISABLED = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SM_DISABLED', 1024)
IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP', 2048)
IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP', 4096)
IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP', 16384)
IB_UVERBS_PCF_CM_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_CM_SUP', 65536)
IB_UVERBS_PCF_SNMP_TUNNEL_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SNMP_TUNNEL_SUP', 131072)
IB_UVERBS_PCF_REINIT_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_REINIT_SUP', 262144)
IB_UVERBS_PCF_DEVICE_MGMT_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_DEVICE_MGMT_SUP', 524288)
IB_UVERBS_PCF_VENDOR_CLASS_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_VENDOR_CLASS_SUP', 1048576)
IB_UVERBS_PCF_DR_NOTICE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_DR_NOTICE_SUP', 2097152)
IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP', 4194304)
IB_UVERBS_PCF_BOOT_MGMT_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_BOOT_MGMT_SUP', 8388608)
IB_UVERBS_PCF_LINK_LATENCY_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_LINK_LATENCY_SUP', 16777216)
IB_UVERBS_PCF_CLIENT_REG_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_CLIENT_REG_SUP', 33554432)
IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP', 134217728)
IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP', 268435456)
IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP', 536870912)
IB_UVERBS_PCF_MCAST_FDB_TOP_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_MCAST_FDB_TOP_SUP', 1073741824)
IB_UVERBS_PCF_HIERARCHY_INFO_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_HIERARCHY_INFO_SUP', 2147483648)
IB_UVERBS_PCF_IP_BASED_GIDS = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_IP_BASED_GIDS', 67108864)

enum_ib_uverbs_query_port_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_QPF_GRH_REQUIRED = enum_ib_uverbs_query_port_flags.define('IB_UVERBS_QPF_GRH_REQUIRED', 1)

enum_ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo = CEnum(ctypes.c_uint32)
IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ = enum_ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo.define('IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ', 0)

class struct_ib_uverbs_flow_action_esp_keymat_aes_gcm(Struct): pass
struct_ib_uverbs_flow_action_esp_keymat_aes_gcm.SIZE = 56
struct_ib_uverbs_flow_action_esp_keymat_aes_gcm._fields_ = ['iv', 'iv_algo', 'salt', 'icv_len', 'key_len', 'aes_key']
setattr(struct_ib_uverbs_flow_action_esp_keymat_aes_gcm, 'iv', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_flow_action_esp_keymat_aes_gcm, 'iv_algo', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_action_esp_keymat_aes_gcm, 'salt', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_action_esp_keymat_aes_gcm, 'icv_len', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_action_esp_keymat_aes_gcm, 'key_len', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_action_esp_keymat_aes_gcm, 'aes_key', field(24, Array(ctypes.c_uint32, 8)))
class struct_ib_uverbs_flow_action_esp_replay_bmp(Struct): pass
struct_ib_uverbs_flow_action_esp_replay_bmp.SIZE = 4
struct_ib_uverbs_flow_action_esp_replay_bmp._fields_ = ['size']
setattr(struct_ib_uverbs_flow_action_esp_replay_bmp, 'size', field(0, ctypes.c_uint32))
enum_ib_uverbs_flow_action_esp_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD', 1)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT', 2)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT', 4)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW', 8)

enum_ib_uverbs_read_counters_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_READ_COUNTERS_PREFER_CACHED = enum_ib_uverbs_read_counters_flags.define('IB_UVERBS_READ_COUNTERS_PREFER_CACHED', 1)

enum_ib_uverbs_advise_mr_flag = CEnum(ctypes.c_uint32)
IB_UVERBS_ADVISE_MR_FLAG_FLUSH = enum_ib_uverbs_advise_mr_flag.define('IB_UVERBS_ADVISE_MR_FLAG_FLUSH', 1)

class struct_ib_uverbs_query_port_resp_ex(Struct): pass
class struct_ib_uverbs_query_port_resp(Struct): pass
__u8 = ctypes.c_ubyte
struct_ib_uverbs_query_port_resp.SIZE = 40
struct_ib_uverbs_query_port_resp._fields_ = ['port_cap_flags', 'max_msg_sz', 'bad_pkey_cntr', 'qkey_viol_cntr', 'gid_tbl_len', 'pkey_tbl_len', 'lid', 'sm_lid', 'state', 'max_mtu', 'active_mtu', 'lmc', 'max_vl_num', 'sm_sl', 'subnet_timeout', 'init_type_reply', 'active_width', 'active_speed', 'phys_state', 'link_layer', 'flags', 'reserved']
setattr(struct_ib_uverbs_query_port_resp, 'port_cap_flags', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_port_resp, 'max_msg_sz', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_port_resp, 'bad_pkey_cntr', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_port_resp, 'qkey_viol_cntr', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_port_resp, 'gid_tbl_len', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_port_resp, 'pkey_tbl_len', field(20, ctypes.c_uint16))
setattr(struct_ib_uverbs_query_port_resp, 'lid', field(22, ctypes.c_uint16))
setattr(struct_ib_uverbs_query_port_resp, 'sm_lid', field(24, ctypes.c_uint16))
setattr(struct_ib_uverbs_query_port_resp, 'state', field(26, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'max_mtu', field(27, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'active_mtu', field(28, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'lmc', field(29, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'max_vl_num', field(30, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'sm_sl', field(31, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'subnet_timeout', field(32, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'init_type_reply', field(33, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'active_width', field(34, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'active_speed', field(35, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'phys_state', field(36, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'link_layer', field(37, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'flags', field(38, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port_resp, 'reserved', field(39, ctypes.c_ubyte))
struct_ib_uverbs_query_port_resp_ex.SIZE = 48
struct_ib_uverbs_query_port_resp_ex._fields_ = ['legacy_resp', 'port_cap_flags2', 'reserved', 'active_speed_ex']
setattr(struct_ib_uverbs_query_port_resp_ex, 'legacy_resp', field(0, struct_ib_uverbs_query_port_resp))
setattr(struct_ib_uverbs_query_port_resp_ex, 'port_cap_flags2', field(40, ctypes.c_uint16))
setattr(struct_ib_uverbs_query_port_resp_ex, 'reserved', field(42, Array(ctypes.c_ubyte, 2)))
setattr(struct_ib_uverbs_query_port_resp_ex, 'active_speed_ex', field(44, ctypes.c_uint32))
class struct_ib_uverbs_qp_cap(Struct): pass
struct_ib_uverbs_qp_cap.SIZE = 20
struct_ib_uverbs_qp_cap._fields_ = ['max_send_wr', 'max_recv_wr', 'max_send_sge', 'max_recv_sge', 'max_inline_data']
setattr(struct_ib_uverbs_qp_cap, 'max_send_wr', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_cap, 'max_recv_wr', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_cap, 'max_send_sge', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_cap, 'max_recv_sge', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_cap, 'max_inline_data', field(16, ctypes.c_uint32))
enum_rdma_driver_id = CEnum(ctypes.c_uint32)
RDMA_DRIVER_UNKNOWN = enum_rdma_driver_id.define('RDMA_DRIVER_UNKNOWN', 0)
RDMA_DRIVER_MLX5 = enum_rdma_driver_id.define('RDMA_DRIVER_MLX5', 1)
RDMA_DRIVER_MLX4 = enum_rdma_driver_id.define('RDMA_DRIVER_MLX4', 2)
RDMA_DRIVER_CXGB3 = enum_rdma_driver_id.define('RDMA_DRIVER_CXGB3', 3)
RDMA_DRIVER_CXGB4 = enum_rdma_driver_id.define('RDMA_DRIVER_CXGB4', 4)
RDMA_DRIVER_MTHCA = enum_rdma_driver_id.define('RDMA_DRIVER_MTHCA', 5)
RDMA_DRIVER_BNXT_RE = enum_rdma_driver_id.define('RDMA_DRIVER_BNXT_RE', 6)
RDMA_DRIVER_OCRDMA = enum_rdma_driver_id.define('RDMA_DRIVER_OCRDMA', 7)
RDMA_DRIVER_NES = enum_rdma_driver_id.define('RDMA_DRIVER_NES', 8)
RDMA_DRIVER_I40IW = enum_rdma_driver_id.define('RDMA_DRIVER_I40IW', 9)
RDMA_DRIVER_IRDMA = enum_rdma_driver_id.define('RDMA_DRIVER_IRDMA', 9)
RDMA_DRIVER_VMW_PVRDMA = enum_rdma_driver_id.define('RDMA_DRIVER_VMW_PVRDMA', 10)
RDMA_DRIVER_QEDR = enum_rdma_driver_id.define('RDMA_DRIVER_QEDR', 11)
RDMA_DRIVER_HNS = enum_rdma_driver_id.define('RDMA_DRIVER_HNS', 12)
RDMA_DRIVER_USNIC = enum_rdma_driver_id.define('RDMA_DRIVER_USNIC', 13)
RDMA_DRIVER_RXE = enum_rdma_driver_id.define('RDMA_DRIVER_RXE', 14)
RDMA_DRIVER_HFI1 = enum_rdma_driver_id.define('RDMA_DRIVER_HFI1', 15)
RDMA_DRIVER_QIB = enum_rdma_driver_id.define('RDMA_DRIVER_QIB', 16)
RDMA_DRIVER_EFA = enum_rdma_driver_id.define('RDMA_DRIVER_EFA', 17)
RDMA_DRIVER_SIW = enum_rdma_driver_id.define('RDMA_DRIVER_SIW', 18)
RDMA_DRIVER_ERDMA = enum_rdma_driver_id.define('RDMA_DRIVER_ERDMA', 19)
RDMA_DRIVER_MANA = enum_rdma_driver_id.define('RDMA_DRIVER_MANA', 20)

enum_ib_uverbs_gid_type = CEnum(ctypes.c_uint32)
IB_UVERBS_GID_TYPE_IB = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_IB', 0)
IB_UVERBS_GID_TYPE_ROCE_V1 = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_ROCE_V1', 1)
IB_UVERBS_GID_TYPE_ROCE_V2 = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_ROCE_V2', 2)

class struct_ib_uverbs_gid_entry(Struct): pass
struct_ib_uverbs_gid_entry.SIZE = 32
struct_ib_uverbs_gid_entry._fields_ = ['gid', 'gid_index', 'port_num', 'gid_type', 'netdev_ifindex']
setattr(struct_ib_uverbs_gid_entry, 'gid', field(0, Array(ctypes.c_uint64, 2)))
setattr(struct_ib_uverbs_gid_entry, 'gid_index', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_gid_entry, 'port_num', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_gid_entry, 'gid_type', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_gid_entry, 'netdev_ifindex', field(28, ctypes.c_uint32))
enum_ib_uverbs_write_cmds = CEnum(ctypes.c_uint32)
IB_USER_VERBS_CMD_GET_CONTEXT = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_GET_CONTEXT', 0)
IB_USER_VERBS_CMD_QUERY_DEVICE = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_DEVICE', 1)
IB_USER_VERBS_CMD_QUERY_PORT = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_PORT', 2)
IB_USER_VERBS_CMD_ALLOC_PD = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_ALLOC_PD', 3)
IB_USER_VERBS_CMD_DEALLOC_PD = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DEALLOC_PD', 4)
IB_USER_VERBS_CMD_CREATE_AH = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_AH', 5)
IB_USER_VERBS_CMD_MODIFY_AH = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_MODIFY_AH', 6)
IB_USER_VERBS_CMD_QUERY_AH = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_AH', 7)
IB_USER_VERBS_CMD_DESTROY_AH = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DESTROY_AH', 8)
IB_USER_VERBS_CMD_REG_MR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_REG_MR', 9)
IB_USER_VERBS_CMD_REG_SMR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_REG_SMR', 10)
IB_USER_VERBS_CMD_REREG_MR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_REREG_MR', 11)
IB_USER_VERBS_CMD_QUERY_MR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_MR', 12)
IB_USER_VERBS_CMD_DEREG_MR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DEREG_MR', 13)
IB_USER_VERBS_CMD_ALLOC_MW = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_ALLOC_MW', 14)
IB_USER_VERBS_CMD_BIND_MW = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_BIND_MW', 15)
IB_USER_VERBS_CMD_DEALLOC_MW = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DEALLOC_MW', 16)
IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL', 17)
IB_USER_VERBS_CMD_CREATE_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_CQ', 18)
IB_USER_VERBS_CMD_RESIZE_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_RESIZE_CQ', 19)
IB_USER_VERBS_CMD_DESTROY_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DESTROY_CQ', 20)
IB_USER_VERBS_CMD_POLL_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_POLL_CQ', 21)
IB_USER_VERBS_CMD_PEEK_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_PEEK_CQ', 22)
IB_USER_VERBS_CMD_REQ_NOTIFY_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_REQ_NOTIFY_CQ', 23)
IB_USER_VERBS_CMD_CREATE_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_QP', 24)
IB_USER_VERBS_CMD_QUERY_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_QP', 25)
IB_USER_VERBS_CMD_MODIFY_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_MODIFY_QP', 26)
IB_USER_VERBS_CMD_DESTROY_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DESTROY_QP', 27)
IB_USER_VERBS_CMD_POST_SEND = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_POST_SEND', 28)
IB_USER_VERBS_CMD_POST_RECV = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_POST_RECV', 29)
IB_USER_VERBS_CMD_ATTACH_MCAST = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_ATTACH_MCAST', 30)
IB_USER_VERBS_CMD_DETACH_MCAST = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DETACH_MCAST', 31)
IB_USER_VERBS_CMD_CREATE_SRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_SRQ', 32)
IB_USER_VERBS_CMD_MODIFY_SRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_MODIFY_SRQ', 33)
IB_USER_VERBS_CMD_QUERY_SRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_SRQ', 34)
IB_USER_VERBS_CMD_DESTROY_SRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DESTROY_SRQ', 35)
IB_USER_VERBS_CMD_POST_SRQ_RECV = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_POST_SRQ_RECV', 36)
IB_USER_VERBS_CMD_OPEN_XRCD = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_OPEN_XRCD', 37)
IB_USER_VERBS_CMD_CLOSE_XRCD = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CLOSE_XRCD', 38)
IB_USER_VERBS_CMD_CREATE_XSRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_XSRQ', 39)
IB_USER_VERBS_CMD_OPEN_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_OPEN_QP', 40)

_anonenum20 = CEnum(ctypes.c_uint32)
IB_USER_VERBS_EX_CMD_QUERY_DEVICE = _anonenum20.define('IB_USER_VERBS_EX_CMD_QUERY_DEVICE', 1)
IB_USER_VERBS_EX_CMD_CREATE_CQ = _anonenum20.define('IB_USER_VERBS_EX_CMD_CREATE_CQ', 18)
IB_USER_VERBS_EX_CMD_CREATE_QP = _anonenum20.define('IB_USER_VERBS_EX_CMD_CREATE_QP', 24)
IB_USER_VERBS_EX_CMD_MODIFY_QP = _anonenum20.define('IB_USER_VERBS_EX_CMD_MODIFY_QP', 26)
IB_USER_VERBS_EX_CMD_CREATE_FLOW = _anonenum20.define('IB_USER_VERBS_EX_CMD_CREATE_FLOW', 50)
IB_USER_VERBS_EX_CMD_DESTROY_FLOW = _anonenum20.define('IB_USER_VERBS_EX_CMD_DESTROY_FLOW', 51)
IB_USER_VERBS_EX_CMD_CREATE_WQ = _anonenum20.define('IB_USER_VERBS_EX_CMD_CREATE_WQ', 52)
IB_USER_VERBS_EX_CMD_MODIFY_WQ = _anonenum20.define('IB_USER_VERBS_EX_CMD_MODIFY_WQ', 53)
IB_USER_VERBS_EX_CMD_DESTROY_WQ = _anonenum20.define('IB_USER_VERBS_EX_CMD_DESTROY_WQ', 54)
IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL = _anonenum20.define('IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL', 55)
IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL = _anonenum20.define('IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL', 56)
IB_USER_VERBS_EX_CMD_MODIFY_CQ = _anonenum20.define('IB_USER_VERBS_EX_CMD_MODIFY_CQ', 57)

enum_ib_placement_type = CEnum(ctypes.c_uint32)
IB_FLUSH_GLOBAL = enum_ib_placement_type.define('IB_FLUSH_GLOBAL', 1)
IB_FLUSH_PERSISTENT = enum_ib_placement_type.define('IB_FLUSH_PERSISTENT', 2)

enum_ib_selectivity_level = CEnum(ctypes.c_uint32)
IB_FLUSH_RANGE = enum_ib_selectivity_level.define('IB_FLUSH_RANGE', 0)
IB_FLUSH_MR = enum_ib_selectivity_level.define('IB_FLUSH_MR', 1)

class struct_ib_uverbs_async_event_desc(Struct): pass
struct_ib_uverbs_async_event_desc.SIZE = 16
struct_ib_uverbs_async_event_desc._fields_ = ['element', 'event_type', 'reserved']
setattr(struct_ib_uverbs_async_event_desc, 'element', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_async_event_desc, 'event_type', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_async_event_desc, 'reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_comp_event_desc(Struct): pass
struct_ib_uverbs_comp_event_desc.SIZE = 8
struct_ib_uverbs_comp_event_desc._fields_ = ['cq_handle']
setattr(struct_ib_uverbs_comp_event_desc, 'cq_handle', field(0, ctypes.c_uint64))
class struct_ib_uverbs_cq_moderation_caps(Struct): pass
struct_ib_uverbs_cq_moderation_caps.SIZE = 8
struct_ib_uverbs_cq_moderation_caps._fields_ = ['max_cq_moderation_count', 'max_cq_moderation_period', 'reserved']
setattr(struct_ib_uverbs_cq_moderation_caps, 'max_cq_moderation_count', field(0, ctypes.c_uint16))
setattr(struct_ib_uverbs_cq_moderation_caps, 'max_cq_moderation_period', field(2, ctypes.c_uint16))
setattr(struct_ib_uverbs_cq_moderation_caps, 'reserved', field(4, ctypes.c_uint32))
class struct_ib_uverbs_cmd_hdr(Struct): pass
struct_ib_uverbs_cmd_hdr.SIZE = 8
struct_ib_uverbs_cmd_hdr._fields_ = ['command', 'in_words', 'out_words']
setattr(struct_ib_uverbs_cmd_hdr, 'command', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_cmd_hdr, 'in_words', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_cmd_hdr, 'out_words', field(6, ctypes.c_uint16))
class struct_ib_uverbs_ex_cmd_hdr(Struct): pass
struct_ib_uverbs_ex_cmd_hdr.SIZE = 16
struct_ib_uverbs_ex_cmd_hdr._fields_ = ['response', 'provider_in_words', 'provider_out_words', 'cmd_hdr_reserved']
setattr(struct_ib_uverbs_ex_cmd_hdr, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_ex_cmd_hdr, 'provider_in_words', field(8, ctypes.c_uint16))
setattr(struct_ib_uverbs_ex_cmd_hdr, 'provider_out_words', field(10, ctypes.c_uint16))
setattr(struct_ib_uverbs_ex_cmd_hdr, 'cmd_hdr_reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_get_context(Struct): pass
struct_ib_uverbs_get_context.SIZE = 8
struct_ib_uverbs_get_context._fields_ = ['response', 'driver_data']
setattr(struct_ib_uverbs_get_context, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_get_context, 'driver_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_get_context_resp(Struct): pass
struct_ib_uverbs_get_context_resp.SIZE = 8
struct_ib_uverbs_get_context_resp._fields_ = ['async_fd', 'num_comp_vectors', 'driver_data']
setattr(struct_ib_uverbs_get_context_resp, 'async_fd', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_get_context_resp, 'num_comp_vectors', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_get_context_resp, 'driver_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_query_device(Struct): pass
struct_ib_uverbs_query_device.SIZE = 8
struct_ib_uverbs_query_device._fields_ = ['response', 'driver_data']
setattr(struct_ib_uverbs_query_device, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_device, 'driver_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_query_device_resp(Struct): pass
struct_ib_uverbs_query_device_resp.SIZE = 176
struct_ib_uverbs_query_device_resp._fields_ = ['fw_ver', 'node_guid', 'sys_image_guid', 'max_mr_size', 'page_size_cap', 'vendor_id', 'vendor_part_id', 'hw_ver', 'max_qp', 'max_qp_wr', 'device_cap_flags', 'max_sge', 'max_sge_rd', 'max_cq', 'max_cqe', 'max_mr', 'max_pd', 'max_qp_rd_atom', 'max_ee_rd_atom', 'max_res_rd_atom', 'max_qp_init_rd_atom', 'max_ee_init_rd_atom', 'atomic_cap', 'max_ee', 'max_rdd', 'max_mw', 'max_raw_ipv6_qp', 'max_raw_ethy_qp', 'max_mcast_grp', 'max_mcast_qp_attach', 'max_total_mcast_qp_attach', 'max_ah', 'max_fmr', 'max_map_per_fmr', 'max_srq', 'max_srq_wr', 'max_srq_sge', 'max_pkeys', 'local_ca_ack_delay', 'phys_port_cnt', 'reserved']
setattr(struct_ib_uverbs_query_device_resp, 'fw_ver', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_device_resp, 'node_guid', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_device_resp, 'sys_image_guid', field(16, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_device_resp, 'max_mr_size', field(24, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_device_resp, 'page_size_cap', field(32, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_device_resp, 'vendor_id', field(40, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'vendor_part_id', field(44, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'hw_ver', field(48, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_qp', field(52, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_qp_wr', field(56, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'device_cap_flags', field(60, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_sge', field(64, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_sge_rd', field(68, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_cq', field(72, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_cqe', field(76, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_mr', field(80, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_pd', field(84, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_qp_rd_atom', field(88, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_ee_rd_atom', field(92, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_res_rd_atom', field(96, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_qp_init_rd_atom', field(100, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_ee_init_rd_atom', field(104, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'atomic_cap', field(108, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_ee', field(112, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_rdd', field(116, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_mw', field(120, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_raw_ipv6_qp', field(124, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_raw_ethy_qp', field(128, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_mcast_grp', field(132, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_mcast_qp_attach', field(136, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_total_mcast_qp_attach', field(140, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_ah', field(144, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_fmr', field(148, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_map_per_fmr', field(152, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_srq', field(156, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_srq_wr', field(160, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_srq_sge', field(164, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_device_resp, 'max_pkeys', field(168, ctypes.c_uint16))
setattr(struct_ib_uverbs_query_device_resp, 'local_ca_ack_delay', field(170, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_device_resp, 'phys_port_cnt', field(171, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_device_resp, 'reserved', field(172, Array(ctypes.c_ubyte, 4)))
class struct_ib_uverbs_ex_query_device(Struct): pass
struct_ib_uverbs_ex_query_device.SIZE = 8
struct_ib_uverbs_ex_query_device._fields_ = ['comp_mask', 'reserved']
setattr(struct_ib_uverbs_ex_query_device, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_query_device, 'reserved', field(4, ctypes.c_uint32))
class struct_ib_uverbs_odp_caps(Struct): pass
class _anonstruct21(Struct): pass
_anonstruct21.SIZE = 12
_anonstruct21._fields_ = ['rc_odp_caps', 'uc_odp_caps', 'ud_odp_caps']
setattr(_anonstruct21, 'rc_odp_caps', field(0, ctypes.c_uint32))
setattr(_anonstruct21, 'uc_odp_caps', field(4, ctypes.c_uint32))
setattr(_anonstruct21, 'ud_odp_caps', field(8, ctypes.c_uint32))
struct_ib_uverbs_odp_caps.SIZE = 24
struct_ib_uverbs_odp_caps._fields_ = ['general_caps', 'per_transport_caps', 'reserved']
setattr(struct_ib_uverbs_odp_caps, 'general_caps', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_odp_caps, 'per_transport_caps', field(8, _anonstruct21))
setattr(struct_ib_uverbs_odp_caps, 'reserved', field(20, ctypes.c_uint32))
class struct_ib_uverbs_rss_caps(Struct): pass
struct_ib_uverbs_rss_caps.SIZE = 16
struct_ib_uverbs_rss_caps._fields_ = ['supported_qpts', 'max_rwq_indirection_tables', 'max_rwq_indirection_table_size', 'reserved']
setattr(struct_ib_uverbs_rss_caps, 'supported_qpts', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_rss_caps, 'max_rwq_indirection_tables', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_rss_caps, 'max_rwq_indirection_table_size', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_rss_caps, 'reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_tm_caps(Struct): pass
struct_ib_uverbs_tm_caps.SIZE = 24
struct_ib_uverbs_tm_caps._fields_ = ['max_rndv_hdr_size', 'max_num_tags', 'flags', 'max_ops', 'max_sge', 'reserved']
setattr(struct_ib_uverbs_tm_caps, 'max_rndv_hdr_size', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_tm_caps, 'max_num_tags', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_tm_caps, 'flags', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_tm_caps, 'max_ops', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_tm_caps, 'max_sge', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_tm_caps, 'reserved', field(20, ctypes.c_uint32))
class struct_ib_uverbs_ex_query_device_resp(Struct): pass
struct_ib_uverbs_ex_query_device_resp.SIZE = 304
struct_ib_uverbs_ex_query_device_resp._fields_ = ['base', 'comp_mask', 'response_length', 'odp_caps', 'timestamp_mask', 'hca_core_clock', 'device_cap_flags_ex', 'rss_caps', 'max_wq_type_rq', 'raw_packet_caps', 'tm_caps', 'cq_moderation_caps', 'max_dm_size', 'xrc_odp_caps', 'reserved']
setattr(struct_ib_uverbs_ex_query_device_resp, 'base', field(0, struct_ib_uverbs_query_device_resp))
setattr(struct_ib_uverbs_ex_query_device_resp, 'comp_mask', field(176, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_query_device_resp, 'response_length', field(180, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_query_device_resp, 'odp_caps', field(184, struct_ib_uverbs_odp_caps))
setattr(struct_ib_uverbs_ex_query_device_resp, 'timestamp_mask', field(208, ctypes.c_uint64))
setattr(struct_ib_uverbs_ex_query_device_resp, 'hca_core_clock', field(216, ctypes.c_uint64))
setattr(struct_ib_uverbs_ex_query_device_resp, 'device_cap_flags_ex', field(224, ctypes.c_uint64))
setattr(struct_ib_uverbs_ex_query_device_resp, 'rss_caps', field(232, struct_ib_uverbs_rss_caps))
setattr(struct_ib_uverbs_ex_query_device_resp, 'max_wq_type_rq', field(248, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_query_device_resp, 'raw_packet_caps', field(252, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_query_device_resp, 'tm_caps', field(256, struct_ib_uverbs_tm_caps))
setattr(struct_ib_uverbs_ex_query_device_resp, 'cq_moderation_caps', field(280, struct_ib_uverbs_cq_moderation_caps))
setattr(struct_ib_uverbs_ex_query_device_resp, 'max_dm_size', field(288, ctypes.c_uint64))
setattr(struct_ib_uverbs_ex_query_device_resp, 'xrc_odp_caps', field(296, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_query_device_resp, 'reserved', field(300, ctypes.c_uint32))
class struct_ib_uverbs_query_port(Struct): pass
struct_ib_uverbs_query_port.SIZE = 16
struct_ib_uverbs_query_port._fields_ = ['response', 'port_num', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_query_port, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_port, 'port_num', field(8, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_port, 'reserved', field(9, Array(ctypes.c_ubyte, 7)))
setattr(struct_ib_uverbs_query_port, 'driver_data', field(16, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_alloc_pd(Struct): pass
struct_ib_uverbs_alloc_pd.SIZE = 8
struct_ib_uverbs_alloc_pd._fields_ = ['response', 'driver_data']
setattr(struct_ib_uverbs_alloc_pd, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_alloc_pd, 'driver_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_alloc_pd_resp(Struct): pass
struct_ib_uverbs_alloc_pd_resp.SIZE = 4
struct_ib_uverbs_alloc_pd_resp._fields_ = ['pd_handle', 'driver_data']
setattr(struct_ib_uverbs_alloc_pd_resp, 'pd_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_alloc_pd_resp, 'driver_data', field(4, Array(ctypes.c_uint32, 0)))
class struct_ib_uverbs_dealloc_pd(Struct): pass
struct_ib_uverbs_dealloc_pd.SIZE = 4
struct_ib_uverbs_dealloc_pd._fields_ = ['pd_handle']
setattr(struct_ib_uverbs_dealloc_pd, 'pd_handle', field(0, ctypes.c_uint32))
class struct_ib_uverbs_open_xrcd(Struct): pass
struct_ib_uverbs_open_xrcd.SIZE = 16
struct_ib_uverbs_open_xrcd._fields_ = ['response', 'fd', 'oflags', 'driver_data']
setattr(struct_ib_uverbs_open_xrcd, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_open_xrcd, 'fd', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_open_xrcd, 'oflags', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_open_xrcd, 'driver_data', field(16, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_open_xrcd_resp(Struct): pass
struct_ib_uverbs_open_xrcd_resp.SIZE = 4
struct_ib_uverbs_open_xrcd_resp._fields_ = ['xrcd_handle', 'driver_data']
setattr(struct_ib_uverbs_open_xrcd_resp, 'xrcd_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_open_xrcd_resp, 'driver_data', field(4, Array(ctypes.c_uint32, 0)))
class struct_ib_uverbs_close_xrcd(Struct): pass
struct_ib_uverbs_close_xrcd.SIZE = 4
struct_ib_uverbs_close_xrcd._fields_ = ['xrcd_handle']
setattr(struct_ib_uverbs_close_xrcd, 'xrcd_handle', field(0, ctypes.c_uint32))
class struct_ib_uverbs_reg_mr(Struct): pass
struct_ib_uverbs_reg_mr.SIZE = 40
struct_ib_uverbs_reg_mr._fields_ = ['response', 'start', 'length', 'hca_va', 'pd_handle', 'access_flags', 'driver_data']
setattr(struct_ib_uverbs_reg_mr, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_reg_mr, 'start', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_reg_mr, 'length', field(16, ctypes.c_uint64))
setattr(struct_ib_uverbs_reg_mr, 'hca_va', field(24, ctypes.c_uint64))
setattr(struct_ib_uverbs_reg_mr, 'pd_handle', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_reg_mr, 'access_flags', field(36, ctypes.c_uint32))
setattr(struct_ib_uverbs_reg_mr, 'driver_data', field(40, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_reg_mr_resp(Struct): pass
struct_ib_uverbs_reg_mr_resp.SIZE = 12
struct_ib_uverbs_reg_mr_resp._fields_ = ['mr_handle', 'lkey', 'rkey', 'driver_data']
setattr(struct_ib_uverbs_reg_mr_resp, 'mr_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_reg_mr_resp, 'lkey', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_reg_mr_resp, 'rkey', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_reg_mr_resp, 'driver_data', field(12, Array(ctypes.c_uint32, 0)))
class struct_ib_uverbs_rereg_mr(Struct): pass
struct_ib_uverbs_rereg_mr.SIZE = 48
struct_ib_uverbs_rereg_mr._fields_ = ['response', 'mr_handle', 'flags', 'start', 'length', 'hca_va', 'pd_handle', 'access_flags', 'driver_data']
setattr(struct_ib_uverbs_rereg_mr, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_rereg_mr, 'mr_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_rereg_mr, 'flags', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_rereg_mr, 'start', field(16, ctypes.c_uint64))
setattr(struct_ib_uverbs_rereg_mr, 'length', field(24, ctypes.c_uint64))
setattr(struct_ib_uverbs_rereg_mr, 'hca_va', field(32, ctypes.c_uint64))
setattr(struct_ib_uverbs_rereg_mr, 'pd_handle', field(40, ctypes.c_uint32))
setattr(struct_ib_uverbs_rereg_mr, 'access_flags', field(44, ctypes.c_uint32))
setattr(struct_ib_uverbs_rereg_mr, 'driver_data', field(48, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_rereg_mr_resp(Struct): pass
struct_ib_uverbs_rereg_mr_resp.SIZE = 8
struct_ib_uverbs_rereg_mr_resp._fields_ = ['lkey', 'rkey', 'driver_data']
setattr(struct_ib_uverbs_rereg_mr_resp, 'lkey', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_rereg_mr_resp, 'rkey', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_rereg_mr_resp, 'driver_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_dereg_mr(Struct): pass
struct_ib_uverbs_dereg_mr.SIZE = 4
struct_ib_uverbs_dereg_mr._fields_ = ['mr_handle']
setattr(struct_ib_uverbs_dereg_mr, 'mr_handle', field(0, ctypes.c_uint32))
class struct_ib_uverbs_alloc_mw(Struct): pass
struct_ib_uverbs_alloc_mw.SIZE = 16
struct_ib_uverbs_alloc_mw._fields_ = ['response', 'pd_handle', 'mw_type', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_alloc_mw, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_alloc_mw, 'pd_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_alloc_mw, 'mw_type', field(12, ctypes.c_ubyte))
setattr(struct_ib_uverbs_alloc_mw, 'reserved', field(13, Array(ctypes.c_ubyte, 3)))
setattr(struct_ib_uverbs_alloc_mw, 'driver_data', field(16, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_alloc_mw_resp(Struct): pass
struct_ib_uverbs_alloc_mw_resp.SIZE = 8
struct_ib_uverbs_alloc_mw_resp._fields_ = ['mw_handle', 'rkey', 'driver_data']
setattr(struct_ib_uverbs_alloc_mw_resp, 'mw_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_alloc_mw_resp, 'rkey', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_alloc_mw_resp, 'driver_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_dealloc_mw(Struct): pass
struct_ib_uverbs_dealloc_mw.SIZE = 4
struct_ib_uverbs_dealloc_mw._fields_ = ['mw_handle']
setattr(struct_ib_uverbs_dealloc_mw, 'mw_handle', field(0, ctypes.c_uint32))
class struct_ib_uverbs_create_comp_channel(Struct): pass
struct_ib_uverbs_create_comp_channel.SIZE = 8
struct_ib_uverbs_create_comp_channel._fields_ = ['response']
setattr(struct_ib_uverbs_create_comp_channel, 'response', field(0, ctypes.c_uint64))
class struct_ib_uverbs_create_comp_channel_resp(Struct): pass
struct_ib_uverbs_create_comp_channel_resp.SIZE = 4
struct_ib_uverbs_create_comp_channel_resp._fields_ = ['fd']
setattr(struct_ib_uverbs_create_comp_channel_resp, 'fd', field(0, ctypes.c_uint32))
class struct_ib_uverbs_create_cq(Struct): pass
__s32 = ctypes.c_int32
struct_ib_uverbs_create_cq.SIZE = 32
struct_ib_uverbs_create_cq._fields_ = ['response', 'user_handle', 'cqe', 'comp_vector', 'comp_channel', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_create_cq, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_cq, 'user_handle', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_cq, 'cqe', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_cq, 'comp_vector', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_cq, 'comp_channel', field(24, ctypes.c_int32))
setattr(struct_ib_uverbs_create_cq, 'reserved', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_cq, 'driver_data', field(32, Array(ctypes.c_uint64, 0)))
enum_ib_uverbs_ex_create_cq_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION = enum_ib_uverbs_ex_create_cq_flags.define('IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION', 1)
IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN = enum_ib_uverbs_ex_create_cq_flags.define('IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN', 2)

class struct_ib_uverbs_ex_create_cq(Struct): pass
struct_ib_uverbs_ex_create_cq.SIZE = 32
struct_ib_uverbs_ex_create_cq._fields_ = ['user_handle', 'cqe', 'comp_vector', 'comp_channel', 'comp_mask', 'flags', 'reserved']
setattr(struct_ib_uverbs_ex_create_cq, 'user_handle', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_ex_create_cq, 'cqe', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_cq, 'comp_vector', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_cq, 'comp_channel', field(16, ctypes.c_int32))
setattr(struct_ib_uverbs_ex_create_cq, 'comp_mask', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_cq, 'flags', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_cq, 'reserved', field(28, ctypes.c_uint32))
class struct_ib_uverbs_create_cq_resp(Struct): pass
struct_ib_uverbs_create_cq_resp.SIZE = 8
struct_ib_uverbs_create_cq_resp._fields_ = ['cq_handle', 'cqe', 'driver_data']
setattr(struct_ib_uverbs_create_cq_resp, 'cq_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_cq_resp, 'cqe', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_cq_resp, 'driver_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_ex_create_cq_resp(Struct): pass
struct_ib_uverbs_ex_create_cq_resp.SIZE = 16
struct_ib_uverbs_ex_create_cq_resp._fields_ = ['base', 'comp_mask', 'response_length']
setattr(struct_ib_uverbs_ex_create_cq_resp, 'base', field(0, struct_ib_uverbs_create_cq_resp))
setattr(struct_ib_uverbs_ex_create_cq_resp, 'comp_mask', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_cq_resp, 'response_length', field(12, ctypes.c_uint32))
class struct_ib_uverbs_resize_cq(Struct): pass
struct_ib_uverbs_resize_cq.SIZE = 16
struct_ib_uverbs_resize_cq._fields_ = ['response', 'cq_handle', 'cqe', 'driver_data']
setattr(struct_ib_uverbs_resize_cq, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_resize_cq, 'cq_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_resize_cq, 'cqe', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_resize_cq, 'driver_data', field(16, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_resize_cq_resp(Struct): pass
struct_ib_uverbs_resize_cq_resp.SIZE = 8
struct_ib_uverbs_resize_cq_resp._fields_ = ['cqe', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_resize_cq_resp, 'cqe', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_resize_cq_resp, 'reserved', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_resize_cq_resp, 'driver_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_poll_cq(Struct): pass
struct_ib_uverbs_poll_cq.SIZE = 16
struct_ib_uverbs_poll_cq._fields_ = ['response', 'cq_handle', 'ne']
setattr(struct_ib_uverbs_poll_cq, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_poll_cq, 'cq_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_poll_cq, 'ne', field(12, ctypes.c_uint32))
enum_ib_uverbs_wc_opcode = CEnum(ctypes.c_uint32)
IB_UVERBS_WC_SEND = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_SEND', 0)
IB_UVERBS_WC_RDMA_WRITE = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_RDMA_WRITE', 1)
IB_UVERBS_WC_RDMA_READ = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_RDMA_READ', 2)
IB_UVERBS_WC_COMP_SWAP = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_COMP_SWAP', 3)
IB_UVERBS_WC_FETCH_ADD = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_FETCH_ADD', 4)
IB_UVERBS_WC_BIND_MW = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_BIND_MW', 5)
IB_UVERBS_WC_LOCAL_INV = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_LOCAL_INV', 6)
IB_UVERBS_WC_TSO = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_TSO', 7)
IB_UVERBS_WC_FLUSH = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_FLUSH', 8)
IB_UVERBS_WC_ATOMIC_WRITE = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_ATOMIC_WRITE', 9)

class struct_ib_uverbs_wc(Struct): pass
class _anonunion22(Union): pass
_anonunion22.SIZE = 4
_anonunion22._fields_ = ['imm_data', 'invalidate_rkey']
setattr(_anonunion22, 'imm_data', field(0, ctypes.c_uint32))
setattr(_anonunion22, 'invalidate_rkey', field(0, ctypes.c_uint32))
struct_ib_uverbs_wc.SIZE = 48
struct_ib_uverbs_wc._fields_ = ['wr_id', 'status', 'opcode', 'vendor_err', 'byte_len', 'ex', 'qp_num', 'src_qp', 'wc_flags', 'pkey_index', 'slid', 'sl', 'dlid_path_bits', 'port_num', 'reserved']
setattr(struct_ib_uverbs_wc, 'wr_id', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_wc, 'status', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_wc, 'opcode', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_wc, 'vendor_err', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_wc, 'byte_len', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_wc, 'ex', field(24, _anonunion22))
setattr(struct_ib_uverbs_wc, 'qp_num', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_wc, 'src_qp', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_wc, 'wc_flags', field(36, ctypes.c_uint32))
setattr(struct_ib_uverbs_wc, 'pkey_index', field(40, ctypes.c_uint16))
setattr(struct_ib_uverbs_wc, 'slid', field(42, ctypes.c_uint16))
setattr(struct_ib_uverbs_wc, 'sl', field(44, ctypes.c_ubyte))
setattr(struct_ib_uverbs_wc, 'dlid_path_bits', field(45, ctypes.c_ubyte))
setattr(struct_ib_uverbs_wc, 'port_num', field(46, ctypes.c_ubyte))
setattr(struct_ib_uverbs_wc, 'reserved', field(47, ctypes.c_ubyte))
class struct_ib_uverbs_poll_cq_resp(Struct): pass
struct_ib_uverbs_poll_cq_resp.SIZE = 8
struct_ib_uverbs_poll_cq_resp._fields_ = ['count', 'reserved', 'wc']
setattr(struct_ib_uverbs_poll_cq_resp, 'count', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_poll_cq_resp, 'reserved', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_poll_cq_resp, 'wc', field(8, Array(struct_ib_uverbs_wc, 0)))
class struct_ib_uverbs_req_notify_cq(Struct): pass
struct_ib_uverbs_req_notify_cq.SIZE = 8
struct_ib_uverbs_req_notify_cq._fields_ = ['cq_handle', 'solicited_only']
setattr(struct_ib_uverbs_req_notify_cq, 'cq_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_req_notify_cq, 'solicited_only', field(4, ctypes.c_uint32))
class struct_ib_uverbs_destroy_cq(Struct): pass
struct_ib_uverbs_destroy_cq.SIZE = 16
struct_ib_uverbs_destroy_cq._fields_ = ['response', 'cq_handle', 'reserved']
setattr(struct_ib_uverbs_destroy_cq, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_destroy_cq, 'cq_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_destroy_cq, 'reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_destroy_cq_resp(Struct): pass
struct_ib_uverbs_destroy_cq_resp.SIZE = 8
struct_ib_uverbs_destroy_cq_resp._fields_ = ['comp_events_reported', 'async_events_reported']
setattr(struct_ib_uverbs_destroy_cq_resp, 'comp_events_reported', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_destroy_cq_resp, 'async_events_reported', field(4, ctypes.c_uint32))
class struct_ib_uverbs_global_route(Struct): pass
struct_ib_uverbs_global_route.SIZE = 24
struct_ib_uverbs_global_route._fields_ = ['dgid', 'flow_label', 'sgid_index', 'hop_limit', 'traffic_class', 'reserved']
setattr(struct_ib_uverbs_global_route, 'dgid', field(0, Array(ctypes.c_ubyte, 16)))
setattr(struct_ib_uverbs_global_route, 'flow_label', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_global_route, 'sgid_index', field(20, ctypes.c_ubyte))
setattr(struct_ib_uverbs_global_route, 'hop_limit', field(21, ctypes.c_ubyte))
setattr(struct_ib_uverbs_global_route, 'traffic_class', field(22, ctypes.c_ubyte))
setattr(struct_ib_uverbs_global_route, 'reserved', field(23, ctypes.c_ubyte))
class struct_ib_uverbs_ah_attr(Struct): pass
struct_ib_uverbs_ah_attr.SIZE = 32
struct_ib_uverbs_ah_attr._fields_ = ['grh', 'dlid', 'sl', 'src_path_bits', 'static_rate', 'is_global', 'port_num', 'reserved']
setattr(struct_ib_uverbs_ah_attr, 'grh', field(0, struct_ib_uverbs_global_route))
setattr(struct_ib_uverbs_ah_attr, 'dlid', field(24, ctypes.c_uint16))
setattr(struct_ib_uverbs_ah_attr, 'sl', field(26, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ah_attr, 'src_path_bits', field(27, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ah_attr, 'static_rate', field(28, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ah_attr, 'is_global', field(29, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ah_attr, 'port_num', field(30, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ah_attr, 'reserved', field(31, ctypes.c_ubyte))
class struct_ib_uverbs_qp_attr(Struct): pass
struct_ib_uverbs_qp_attr.SIZE = 144
struct_ib_uverbs_qp_attr._fields_ = ['qp_attr_mask', 'qp_state', 'cur_qp_state', 'path_mtu', 'path_mig_state', 'qkey', 'rq_psn', 'sq_psn', 'dest_qp_num', 'qp_access_flags', 'ah_attr', 'alt_ah_attr', 'max_send_wr', 'max_recv_wr', 'max_send_sge', 'max_recv_sge', 'max_inline_data', 'pkey_index', 'alt_pkey_index', 'en_sqd_async_notify', 'sq_draining', 'max_rd_atomic', 'max_dest_rd_atomic', 'min_rnr_timer', 'port_num', 'timeout', 'retry_cnt', 'rnr_retry', 'alt_port_num', 'alt_timeout', 'reserved']
setattr(struct_ib_uverbs_qp_attr, 'qp_attr_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'qp_state', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'cur_qp_state', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'path_mtu', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'path_mig_state', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'qkey', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'rq_psn', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'sq_psn', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'dest_qp_num', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'qp_access_flags', field(36, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'ah_attr', field(40, struct_ib_uverbs_ah_attr))
setattr(struct_ib_uverbs_qp_attr, 'alt_ah_attr', field(72, struct_ib_uverbs_ah_attr))
setattr(struct_ib_uverbs_qp_attr, 'max_send_wr', field(104, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'max_recv_wr', field(108, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'max_send_sge', field(112, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'max_recv_sge', field(116, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'max_inline_data', field(120, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_attr, 'pkey_index', field(124, ctypes.c_uint16))
setattr(struct_ib_uverbs_qp_attr, 'alt_pkey_index', field(126, ctypes.c_uint16))
setattr(struct_ib_uverbs_qp_attr, 'en_sqd_async_notify', field(128, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'sq_draining', field(129, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'max_rd_atomic', field(130, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'max_dest_rd_atomic', field(131, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'min_rnr_timer', field(132, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'port_num', field(133, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'timeout', field(134, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'retry_cnt', field(135, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'rnr_retry', field(136, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'alt_port_num', field(137, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'alt_timeout', field(138, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_attr, 'reserved', field(139, Array(ctypes.c_ubyte, 5)))
class struct_ib_uverbs_create_qp(Struct): pass
struct_ib_uverbs_create_qp.SIZE = 56
struct_ib_uverbs_create_qp._fields_ = ['response', 'user_handle', 'pd_handle', 'send_cq_handle', 'recv_cq_handle', 'srq_handle', 'max_send_wr', 'max_recv_wr', 'max_send_sge', 'max_recv_sge', 'max_inline_data', 'sq_sig_all', 'qp_type', 'is_srq', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_create_qp, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_qp, 'user_handle', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_qp, 'pd_handle', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'send_cq_handle', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'recv_cq_handle', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'srq_handle', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'max_send_wr', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'max_recv_wr', field(36, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'max_send_sge', field(40, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'max_recv_sge', field(44, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'max_inline_data', field(48, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp, 'sq_sig_all', field(52, ctypes.c_ubyte))
setattr(struct_ib_uverbs_create_qp, 'qp_type', field(53, ctypes.c_ubyte))
setattr(struct_ib_uverbs_create_qp, 'is_srq', field(54, ctypes.c_ubyte))
setattr(struct_ib_uverbs_create_qp, 'reserved', field(55, ctypes.c_ubyte))
setattr(struct_ib_uverbs_create_qp, 'driver_data', field(56, Array(ctypes.c_uint64, 0)))
enum_ib_uverbs_create_qp_mask = CEnum(ctypes.c_uint32)
IB_UVERBS_CREATE_QP_MASK_IND_TABLE = enum_ib_uverbs_create_qp_mask.define('IB_UVERBS_CREATE_QP_MASK_IND_TABLE', 1)

_anonenum23 = CEnum(ctypes.c_uint32)
IB_UVERBS_CREATE_QP_SUP_COMP_MASK = _anonenum23.define('IB_UVERBS_CREATE_QP_SUP_COMP_MASK', 1)

class struct_ib_uverbs_ex_create_qp(Struct): pass
struct_ib_uverbs_ex_create_qp.SIZE = 64
struct_ib_uverbs_ex_create_qp._fields_ = ['user_handle', 'pd_handle', 'send_cq_handle', 'recv_cq_handle', 'srq_handle', 'max_send_wr', 'max_recv_wr', 'max_send_sge', 'max_recv_sge', 'max_inline_data', 'sq_sig_all', 'qp_type', 'is_srq', 'reserved', 'comp_mask', 'create_flags', 'rwq_ind_tbl_handle', 'source_qpn']
setattr(struct_ib_uverbs_ex_create_qp, 'user_handle', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_ex_create_qp, 'pd_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'send_cq_handle', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'recv_cq_handle', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'srq_handle', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'max_send_wr', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'max_recv_wr', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'max_send_sge', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'max_recv_sge', field(36, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'max_inline_data', field(40, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'sq_sig_all', field(44, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ex_create_qp, 'qp_type', field(45, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ex_create_qp, 'is_srq', field(46, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ex_create_qp, 'reserved', field(47, ctypes.c_ubyte))
setattr(struct_ib_uverbs_ex_create_qp, 'comp_mask', field(48, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'create_flags', field(52, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'rwq_ind_tbl_handle', field(56, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp, 'source_qpn', field(60, ctypes.c_uint32))
class struct_ib_uverbs_open_qp(Struct): pass
struct_ib_uverbs_open_qp.SIZE = 32
struct_ib_uverbs_open_qp._fields_ = ['response', 'user_handle', 'pd_handle', 'qpn', 'qp_type', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_open_qp, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_open_qp, 'user_handle', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_open_qp, 'pd_handle', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_open_qp, 'qpn', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_open_qp, 'qp_type', field(24, ctypes.c_ubyte))
setattr(struct_ib_uverbs_open_qp, 'reserved', field(25, Array(ctypes.c_ubyte, 7)))
setattr(struct_ib_uverbs_open_qp, 'driver_data', field(32, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_create_qp_resp(Struct): pass
struct_ib_uverbs_create_qp_resp.SIZE = 32
struct_ib_uverbs_create_qp_resp._fields_ = ['qp_handle', 'qpn', 'max_send_wr', 'max_recv_wr', 'max_send_sge', 'max_recv_sge', 'max_inline_data', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_create_qp_resp, 'qp_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp_resp, 'qpn', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp_resp, 'max_send_wr', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp_resp, 'max_recv_wr', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp_resp, 'max_send_sge', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp_resp, 'max_recv_sge', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp_resp, 'max_inline_data', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp_resp, 'reserved', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_qp_resp, 'driver_data', field(32, Array(ctypes.c_uint32, 0)))
class struct_ib_uverbs_ex_create_qp_resp(Struct): pass
struct_ib_uverbs_ex_create_qp_resp.SIZE = 40
struct_ib_uverbs_ex_create_qp_resp._fields_ = ['base', 'comp_mask', 'response_length']
setattr(struct_ib_uverbs_ex_create_qp_resp, 'base', field(0, struct_ib_uverbs_create_qp_resp))
setattr(struct_ib_uverbs_ex_create_qp_resp, 'comp_mask', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_qp_resp, 'response_length', field(36, ctypes.c_uint32))
class struct_ib_uverbs_qp_dest(Struct): pass
struct_ib_uverbs_qp_dest.SIZE = 32
struct_ib_uverbs_qp_dest._fields_ = ['dgid', 'flow_label', 'dlid', 'reserved', 'sgid_index', 'hop_limit', 'traffic_class', 'sl', 'src_path_bits', 'static_rate', 'is_global', 'port_num']
setattr(struct_ib_uverbs_qp_dest, 'dgid', field(0, Array(ctypes.c_ubyte, 16)))
setattr(struct_ib_uverbs_qp_dest, 'flow_label', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_qp_dest, 'dlid', field(20, ctypes.c_uint16))
setattr(struct_ib_uverbs_qp_dest, 'reserved', field(22, ctypes.c_uint16))
setattr(struct_ib_uverbs_qp_dest, 'sgid_index', field(24, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_dest, 'hop_limit', field(25, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_dest, 'traffic_class', field(26, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_dest, 'sl', field(27, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_dest, 'src_path_bits', field(28, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_dest, 'static_rate', field(29, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_dest, 'is_global', field(30, ctypes.c_ubyte))
setattr(struct_ib_uverbs_qp_dest, 'port_num', field(31, ctypes.c_ubyte))
class struct_ib_uverbs_query_qp(Struct): pass
struct_ib_uverbs_query_qp.SIZE = 16
struct_ib_uverbs_query_qp._fields_ = ['response', 'qp_handle', 'attr_mask', 'driver_data']
setattr(struct_ib_uverbs_query_qp, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_qp, 'qp_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp, 'attr_mask', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp, 'driver_data', field(16, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_query_qp_resp(Struct): pass
struct_ib_uverbs_query_qp_resp.SIZE = 128
struct_ib_uverbs_query_qp_resp._fields_ = ['dest', 'alt_dest', 'max_send_wr', 'max_recv_wr', 'max_send_sge', 'max_recv_sge', 'max_inline_data', 'qkey', 'rq_psn', 'sq_psn', 'dest_qp_num', 'qp_access_flags', 'pkey_index', 'alt_pkey_index', 'qp_state', 'cur_qp_state', 'path_mtu', 'path_mig_state', 'sq_draining', 'max_rd_atomic', 'max_dest_rd_atomic', 'min_rnr_timer', 'port_num', 'timeout', 'retry_cnt', 'rnr_retry', 'alt_port_num', 'alt_timeout', 'sq_sig_all', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_query_qp_resp, 'dest', field(0, struct_ib_uverbs_qp_dest))
setattr(struct_ib_uverbs_query_qp_resp, 'alt_dest', field(32, struct_ib_uverbs_qp_dest))
setattr(struct_ib_uverbs_query_qp_resp, 'max_send_wr', field(64, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'max_recv_wr', field(68, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'max_send_sge', field(72, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'max_recv_sge', field(76, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'max_inline_data', field(80, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'qkey', field(84, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'rq_psn', field(88, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'sq_psn', field(92, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'dest_qp_num', field(96, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'qp_access_flags', field(100, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_qp_resp, 'pkey_index', field(104, ctypes.c_uint16))
setattr(struct_ib_uverbs_query_qp_resp, 'alt_pkey_index', field(106, ctypes.c_uint16))
setattr(struct_ib_uverbs_query_qp_resp, 'qp_state', field(108, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'cur_qp_state', field(109, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'path_mtu', field(110, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'path_mig_state', field(111, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'sq_draining', field(112, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'max_rd_atomic', field(113, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'max_dest_rd_atomic', field(114, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'min_rnr_timer', field(115, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'port_num', field(116, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'timeout', field(117, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'retry_cnt', field(118, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'rnr_retry', field(119, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'alt_port_num', field(120, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'alt_timeout', field(121, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'sq_sig_all', field(122, ctypes.c_ubyte))
setattr(struct_ib_uverbs_query_qp_resp, 'reserved', field(123, Array(ctypes.c_ubyte, 5)))
setattr(struct_ib_uverbs_query_qp_resp, 'driver_data', field(128, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_modify_qp(Struct): pass
struct_ib_uverbs_modify_qp.SIZE = 112
struct_ib_uverbs_modify_qp._fields_ = ['dest', 'alt_dest', 'qp_handle', 'attr_mask', 'qkey', 'rq_psn', 'sq_psn', 'dest_qp_num', 'qp_access_flags', 'pkey_index', 'alt_pkey_index', 'qp_state', 'cur_qp_state', 'path_mtu', 'path_mig_state', 'en_sqd_async_notify', 'max_rd_atomic', 'max_dest_rd_atomic', 'min_rnr_timer', 'port_num', 'timeout', 'retry_cnt', 'rnr_retry', 'alt_port_num', 'alt_timeout', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_modify_qp, 'dest', field(0, struct_ib_uverbs_qp_dest))
setattr(struct_ib_uverbs_modify_qp, 'alt_dest', field(32, struct_ib_uverbs_qp_dest))
setattr(struct_ib_uverbs_modify_qp, 'qp_handle', field(64, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_qp, 'attr_mask', field(68, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_qp, 'qkey', field(72, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_qp, 'rq_psn', field(76, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_qp, 'sq_psn', field(80, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_qp, 'dest_qp_num', field(84, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_qp, 'qp_access_flags', field(88, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_qp, 'pkey_index', field(92, ctypes.c_uint16))
setattr(struct_ib_uverbs_modify_qp, 'alt_pkey_index', field(94, ctypes.c_uint16))
setattr(struct_ib_uverbs_modify_qp, 'qp_state', field(96, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'cur_qp_state', field(97, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'path_mtu', field(98, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'path_mig_state', field(99, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'en_sqd_async_notify', field(100, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'max_rd_atomic', field(101, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'max_dest_rd_atomic', field(102, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'min_rnr_timer', field(103, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'port_num', field(104, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'timeout', field(105, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'retry_cnt', field(106, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'rnr_retry', field(107, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'alt_port_num', field(108, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'alt_timeout', field(109, ctypes.c_ubyte))
setattr(struct_ib_uverbs_modify_qp, 'reserved', field(110, Array(ctypes.c_ubyte, 2)))
setattr(struct_ib_uverbs_modify_qp, 'driver_data', field(112, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_ex_modify_qp(Struct): pass
struct_ib_uverbs_ex_modify_qp.SIZE = 120
struct_ib_uverbs_ex_modify_qp._fields_ = ['base', 'rate_limit', 'reserved']
setattr(struct_ib_uverbs_ex_modify_qp, 'base', field(0, struct_ib_uverbs_modify_qp))
setattr(struct_ib_uverbs_ex_modify_qp, 'rate_limit', field(112, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_qp, 'reserved', field(116, ctypes.c_uint32))
class struct_ib_uverbs_ex_modify_qp_resp(Struct): pass
struct_ib_uverbs_ex_modify_qp_resp.SIZE = 8
struct_ib_uverbs_ex_modify_qp_resp._fields_ = ['comp_mask', 'response_length']
setattr(struct_ib_uverbs_ex_modify_qp_resp, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_qp_resp, 'response_length', field(4, ctypes.c_uint32))
class struct_ib_uverbs_destroy_qp(Struct): pass
struct_ib_uverbs_destroy_qp.SIZE = 16
struct_ib_uverbs_destroy_qp._fields_ = ['response', 'qp_handle', 'reserved']
setattr(struct_ib_uverbs_destroy_qp, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_destroy_qp, 'qp_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_destroy_qp, 'reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_destroy_qp_resp(Struct): pass
struct_ib_uverbs_destroy_qp_resp.SIZE = 4
struct_ib_uverbs_destroy_qp_resp._fields_ = ['events_reported']
setattr(struct_ib_uverbs_destroy_qp_resp, 'events_reported', field(0, ctypes.c_uint32))
class struct_ib_uverbs_sge(Struct): pass
struct_ib_uverbs_sge.SIZE = 16
struct_ib_uverbs_sge._fields_ = ['addr', 'length', 'lkey']
setattr(struct_ib_uverbs_sge, 'addr', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_sge, 'length', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_sge, 'lkey', field(12, ctypes.c_uint32))
enum_ib_uverbs_wr_opcode = CEnum(ctypes.c_uint32)
IB_UVERBS_WR_RDMA_WRITE = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_RDMA_WRITE', 0)
IB_UVERBS_WR_RDMA_WRITE_WITH_IMM = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_RDMA_WRITE_WITH_IMM', 1)
IB_UVERBS_WR_SEND = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_SEND', 2)
IB_UVERBS_WR_SEND_WITH_IMM = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_SEND_WITH_IMM', 3)
IB_UVERBS_WR_RDMA_READ = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_RDMA_READ', 4)
IB_UVERBS_WR_ATOMIC_CMP_AND_SWP = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_ATOMIC_CMP_AND_SWP', 5)
IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD', 6)
IB_UVERBS_WR_LOCAL_INV = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_LOCAL_INV', 7)
IB_UVERBS_WR_BIND_MW = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_BIND_MW', 8)
IB_UVERBS_WR_SEND_WITH_INV = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_SEND_WITH_INV', 9)
IB_UVERBS_WR_TSO = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_TSO', 10)
IB_UVERBS_WR_RDMA_READ_WITH_INV = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_RDMA_READ_WITH_INV', 11)
IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP', 12)
IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD', 13)
IB_UVERBS_WR_FLUSH = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_FLUSH', 14)
IB_UVERBS_WR_ATOMIC_WRITE = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_ATOMIC_WRITE', 15)

class struct_ib_uverbs_send_wr(Struct): pass
class _anonunion24(Union): pass
_anonunion24.SIZE = 4
_anonunion24._fields_ = ['imm_data', 'invalidate_rkey']
setattr(_anonunion24, 'imm_data', field(0, ctypes.c_uint32))
setattr(_anonunion24, 'invalidate_rkey', field(0, ctypes.c_uint32))
class _anonunion25(Union): pass
class _anonstruct26(Struct): pass
_anonstruct26.SIZE = 16
_anonstruct26._fields_ = ['remote_addr', 'rkey', 'reserved']
setattr(_anonstruct26, 'remote_addr', field(0, ctypes.c_uint64))
setattr(_anonstruct26, 'rkey', field(8, ctypes.c_uint32))
setattr(_anonstruct26, 'reserved', field(12, ctypes.c_uint32))
class _anonstruct27(Struct): pass
_anonstruct27.SIZE = 32
_anonstruct27._fields_ = ['remote_addr', 'compare_add', 'swap', 'rkey', 'reserved']
setattr(_anonstruct27, 'remote_addr', field(0, ctypes.c_uint64))
setattr(_anonstruct27, 'compare_add', field(8, ctypes.c_uint64))
setattr(_anonstruct27, 'swap', field(16, ctypes.c_uint64))
setattr(_anonstruct27, 'rkey', field(24, ctypes.c_uint32))
setattr(_anonstruct27, 'reserved', field(28, ctypes.c_uint32))
class _anonstruct28(Struct): pass
_anonstruct28.SIZE = 16
_anonstruct28._fields_ = ['ah', 'remote_qpn', 'remote_qkey', 'reserved']
setattr(_anonstruct28, 'ah', field(0, ctypes.c_uint32))
setattr(_anonstruct28, 'remote_qpn', field(4, ctypes.c_uint32))
setattr(_anonstruct28, 'remote_qkey', field(8, ctypes.c_uint32))
setattr(_anonstruct28, 'reserved', field(12, ctypes.c_uint32))
_anonunion25.SIZE = 32
_anonunion25._fields_ = ['rdma', 'atomic', 'ud']
setattr(_anonunion25, 'rdma', field(0, _anonstruct26))
setattr(_anonunion25, 'atomic', field(0, _anonstruct27))
setattr(_anonunion25, 'ud', field(0, _anonstruct28))
struct_ib_uverbs_send_wr.SIZE = 56
struct_ib_uverbs_send_wr._fields_ = ['wr_id', 'num_sge', 'opcode', 'send_flags', 'ex', 'wr']
setattr(struct_ib_uverbs_send_wr, 'wr_id', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_send_wr, 'num_sge', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_send_wr, 'opcode', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_send_wr, 'send_flags', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_send_wr, 'ex', field(20, _anonunion24))
setattr(struct_ib_uverbs_send_wr, 'wr', field(24, _anonunion25))
class struct_ib_uverbs_post_send(Struct): pass
struct_ib_uverbs_post_send.SIZE = 24
struct_ib_uverbs_post_send._fields_ = ['response', 'qp_handle', 'wr_count', 'sge_count', 'wqe_size', 'send_wr']
setattr(struct_ib_uverbs_post_send, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_post_send, 'qp_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_send, 'wr_count', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_send, 'sge_count', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_send, 'wqe_size', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_send, 'send_wr', field(24, Array(struct_ib_uverbs_send_wr, 0)))
class struct_ib_uverbs_post_send_resp(Struct): pass
struct_ib_uverbs_post_send_resp.SIZE = 4
struct_ib_uverbs_post_send_resp._fields_ = ['bad_wr']
setattr(struct_ib_uverbs_post_send_resp, 'bad_wr', field(0, ctypes.c_uint32))
class struct_ib_uverbs_recv_wr(Struct): pass
struct_ib_uverbs_recv_wr.SIZE = 16
struct_ib_uverbs_recv_wr._fields_ = ['wr_id', 'num_sge', 'reserved']
setattr(struct_ib_uverbs_recv_wr, 'wr_id', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_recv_wr, 'num_sge', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_recv_wr, 'reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_post_recv(Struct): pass
struct_ib_uverbs_post_recv.SIZE = 24
struct_ib_uverbs_post_recv._fields_ = ['response', 'qp_handle', 'wr_count', 'sge_count', 'wqe_size', 'recv_wr']
setattr(struct_ib_uverbs_post_recv, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_post_recv, 'qp_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_recv, 'wr_count', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_recv, 'sge_count', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_recv, 'wqe_size', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_recv, 'recv_wr', field(24, Array(struct_ib_uverbs_recv_wr, 0)))
class struct_ib_uverbs_post_recv_resp(Struct): pass
struct_ib_uverbs_post_recv_resp.SIZE = 4
struct_ib_uverbs_post_recv_resp._fields_ = ['bad_wr']
setattr(struct_ib_uverbs_post_recv_resp, 'bad_wr', field(0, ctypes.c_uint32))
class struct_ib_uverbs_post_srq_recv(Struct): pass
struct_ib_uverbs_post_srq_recv.SIZE = 24
struct_ib_uverbs_post_srq_recv._fields_ = ['response', 'srq_handle', 'wr_count', 'sge_count', 'wqe_size', 'recv']
setattr(struct_ib_uverbs_post_srq_recv, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_post_srq_recv, 'srq_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_srq_recv, 'wr_count', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_srq_recv, 'sge_count', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_srq_recv, 'wqe_size', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_post_srq_recv, 'recv', field(24, Array(struct_ib_uverbs_recv_wr, 0)))
class struct_ib_uverbs_post_srq_recv_resp(Struct): pass
struct_ib_uverbs_post_srq_recv_resp.SIZE = 4
struct_ib_uverbs_post_srq_recv_resp._fields_ = ['bad_wr']
setattr(struct_ib_uverbs_post_srq_recv_resp, 'bad_wr', field(0, ctypes.c_uint32))
class struct_ib_uverbs_create_ah(Struct): pass
struct_ib_uverbs_create_ah.SIZE = 56
struct_ib_uverbs_create_ah._fields_ = ['response', 'user_handle', 'pd_handle', 'reserved', 'attr', 'driver_data']
setattr(struct_ib_uverbs_create_ah, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_ah, 'user_handle', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_ah, 'pd_handle', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_ah, 'reserved', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_ah, 'attr', field(24, struct_ib_uverbs_ah_attr))
setattr(struct_ib_uverbs_create_ah, 'driver_data', field(56, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_create_ah_resp(Struct): pass
struct_ib_uverbs_create_ah_resp.SIZE = 4
struct_ib_uverbs_create_ah_resp._fields_ = ['ah_handle', 'driver_data']
setattr(struct_ib_uverbs_create_ah_resp, 'ah_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_ah_resp, 'driver_data', field(4, Array(ctypes.c_uint32, 0)))
class struct_ib_uverbs_destroy_ah(Struct): pass
struct_ib_uverbs_destroy_ah.SIZE = 4
struct_ib_uverbs_destroy_ah._fields_ = ['ah_handle']
setattr(struct_ib_uverbs_destroy_ah, 'ah_handle', field(0, ctypes.c_uint32))
class struct_ib_uverbs_attach_mcast(Struct): pass
struct_ib_uverbs_attach_mcast.SIZE = 24
struct_ib_uverbs_attach_mcast._fields_ = ['gid', 'qp_handle', 'mlid', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_attach_mcast, 'gid', field(0, Array(ctypes.c_ubyte, 16)))
setattr(struct_ib_uverbs_attach_mcast, 'qp_handle', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_attach_mcast, 'mlid', field(20, ctypes.c_uint16))
setattr(struct_ib_uverbs_attach_mcast, 'reserved', field(22, ctypes.c_uint16))
setattr(struct_ib_uverbs_attach_mcast, 'driver_data', field(24, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_detach_mcast(Struct): pass
struct_ib_uverbs_detach_mcast.SIZE = 24
struct_ib_uverbs_detach_mcast._fields_ = ['gid', 'qp_handle', 'mlid', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_detach_mcast, 'gid', field(0, Array(ctypes.c_ubyte, 16)))
setattr(struct_ib_uverbs_detach_mcast, 'qp_handle', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_detach_mcast, 'mlid', field(20, ctypes.c_uint16))
setattr(struct_ib_uverbs_detach_mcast, 'reserved', field(22, ctypes.c_uint16))
setattr(struct_ib_uverbs_detach_mcast, 'driver_data', field(24, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_flow_spec_hdr(Struct): pass
struct_ib_uverbs_flow_spec_hdr.SIZE = 8
struct_ib_uverbs_flow_spec_hdr._fields_ = ['type', 'size', 'reserved', 'flow_spec_data']
setattr(struct_ib_uverbs_flow_spec_hdr, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_hdr, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_hdr, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_hdr, 'flow_spec_data', field(8, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_flow_eth_filter(Struct): pass
struct_ib_uverbs_flow_eth_filter.SIZE = 16
struct_ib_uverbs_flow_eth_filter._fields_ = ['dst_mac', 'src_mac', 'ether_type', 'vlan_tag']
setattr(struct_ib_uverbs_flow_eth_filter, 'dst_mac', field(0, Array(ctypes.c_ubyte, 6)))
setattr(struct_ib_uverbs_flow_eth_filter, 'src_mac', field(6, Array(ctypes.c_ubyte, 6)))
setattr(struct_ib_uverbs_flow_eth_filter, 'ether_type', field(12, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_eth_filter, 'vlan_tag', field(14, ctypes.c_uint16))
class struct_ib_uverbs_flow_spec_eth(Struct): pass
struct_ib_uverbs_flow_spec_eth.SIZE = 40
struct_ib_uverbs_flow_spec_eth._fields_ = ['hdr', 'type', 'size', 'reserved', 'val', 'mask']
setattr(struct_ib_uverbs_flow_spec_eth, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_eth, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_eth, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_eth, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_eth, 'val', field(8, struct_ib_uverbs_flow_eth_filter))
setattr(struct_ib_uverbs_flow_spec_eth, 'mask', field(24, struct_ib_uverbs_flow_eth_filter))
class struct_ib_uverbs_flow_ipv4_filter(Struct): pass
struct_ib_uverbs_flow_ipv4_filter.SIZE = 12
struct_ib_uverbs_flow_ipv4_filter._fields_ = ['src_ip', 'dst_ip', 'proto', 'tos', 'ttl', 'flags']
setattr(struct_ib_uverbs_flow_ipv4_filter, 'src_ip', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_ipv4_filter, 'dst_ip', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_ipv4_filter, 'proto', field(8, ctypes.c_ubyte))
setattr(struct_ib_uverbs_flow_ipv4_filter, 'tos', field(9, ctypes.c_ubyte))
setattr(struct_ib_uverbs_flow_ipv4_filter, 'ttl', field(10, ctypes.c_ubyte))
setattr(struct_ib_uverbs_flow_ipv4_filter, 'flags', field(11, ctypes.c_ubyte))
class struct_ib_uverbs_flow_spec_ipv4(Struct): pass
struct_ib_uverbs_flow_spec_ipv4.SIZE = 32
struct_ib_uverbs_flow_spec_ipv4._fields_ = ['hdr', 'type', 'size', 'reserved', 'val', 'mask']
setattr(struct_ib_uverbs_flow_spec_ipv4, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_ipv4, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_ipv4, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_ipv4, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_ipv4, 'val', field(8, struct_ib_uverbs_flow_ipv4_filter))
setattr(struct_ib_uverbs_flow_spec_ipv4, 'mask', field(20, struct_ib_uverbs_flow_ipv4_filter))
class struct_ib_uverbs_flow_tcp_udp_filter(Struct): pass
struct_ib_uverbs_flow_tcp_udp_filter.SIZE = 4
struct_ib_uverbs_flow_tcp_udp_filter._fields_ = ['dst_port', 'src_port']
setattr(struct_ib_uverbs_flow_tcp_udp_filter, 'dst_port', field(0, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_tcp_udp_filter, 'src_port', field(2, ctypes.c_uint16))
class struct_ib_uverbs_flow_spec_tcp_udp(Struct): pass
struct_ib_uverbs_flow_spec_tcp_udp.SIZE = 16
struct_ib_uverbs_flow_spec_tcp_udp._fields_ = ['hdr', 'type', 'size', 'reserved', 'val', 'mask']
setattr(struct_ib_uverbs_flow_spec_tcp_udp, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_tcp_udp, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_tcp_udp, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_tcp_udp, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_tcp_udp, 'val', field(8, struct_ib_uverbs_flow_tcp_udp_filter))
setattr(struct_ib_uverbs_flow_spec_tcp_udp, 'mask', field(12, struct_ib_uverbs_flow_tcp_udp_filter))
class struct_ib_uverbs_flow_ipv6_filter(Struct): pass
struct_ib_uverbs_flow_ipv6_filter.SIZE = 40
struct_ib_uverbs_flow_ipv6_filter._fields_ = ['src_ip', 'dst_ip', 'flow_label', 'next_hdr', 'traffic_class', 'hop_limit', 'reserved']
setattr(struct_ib_uverbs_flow_ipv6_filter, 'src_ip', field(0, Array(ctypes.c_ubyte, 16)))
setattr(struct_ib_uverbs_flow_ipv6_filter, 'dst_ip', field(16, Array(ctypes.c_ubyte, 16)))
setattr(struct_ib_uverbs_flow_ipv6_filter, 'flow_label', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_ipv6_filter, 'next_hdr', field(36, ctypes.c_ubyte))
setattr(struct_ib_uverbs_flow_ipv6_filter, 'traffic_class', field(37, ctypes.c_ubyte))
setattr(struct_ib_uverbs_flow_ipv6_filter, 'hop_limit', field(38, ctypes.c_ubyte))
setattr(struct_ib_uverbs_flow_ipv6_filter, 'reserved', field(39, ctypes.c_ubyte))
class struct_ib_uverbs_flow_spec_ipv6(Struct): pass
struct_ib_uverbs_flow_spec_ipv6.SIZE = 88
struct_ib_uverbs_flow_spec_ipv6._fields_ = ['hdr', 'type', 'size', 'reserved', 'val', 'mask']
setattr(struct_ib_uverbs_flow_spec_ipv6, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_ipv6, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_ipv6, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_ipv6, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_ipv6, 'val', field(8, struct_ib_uverbs_flow_ipv6_filter))
setattr(struct_ib_uverbs_flow_spec_ipv6, 'mask', field(48, struct_ib_uverbs_flow_ipv6_filter))
class struct_ib_uverbs_flow_spec_action_tag(Struct): pass
struct_ib_uverbs_flow_spec_action_tag.SIZE = 16
struct_ib_uverbs_flow_spec_action_tag._fields_ = ['hdr', 'type', 'size', 'reserved', 'tag_id', 'reserved1']
setattr(struct_ib_uverbs_flow_spec_action_tag, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_action_tag, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_action_tag, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_action_tag, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_action_tag, 'tag_id', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_action_tag, 'reserved1', field(12, ctypes.c_uint32))
class struct_ib_uverbs_flow_spec_action_drop(Struct): pass
struct_ib_uverbs_flow_spec_action_drop.SIZE = 8
struct_ib_uverbs_flow_spec_action_drop._fields_ = ['hdr', 'type', 'size', 'reserved']
setattr(struct_ib_uverbs_flow_spec_action_drop, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_action_drop, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_action_drop, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_action_drop, 'reserved', field(6, ctypes.c_uint16))
class struct_ib_uverbs_flow_spec_action_handle(Struct): pass
struct_ib_uverbs_flow_spec_action_handle.SIZE = 16
struct_ib_uverbs_flow_spec_action_handle._fields_ = ['hdr', 'type', 'size', 'reserved', 'handle', 'reserved1']
setattr(struct_ib_uverbs_flow_spec_action_handle, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_action_handle, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_action_handle, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_action_handle, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_action_handle, 'handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_action_handle, 'reserved1', field(12, ctypes.c_uint32))
class struct_ib_uverbs_flow_spec_action_count(Struct): pass
struct_ib_uverbs_flow_spec_action_count.SIZE = 16
struct_ib_uverbs_flow_spec_action_count._fields_ = ['hdr', 'type', 'size', 'reserved', 'handle', 'reserved1']
setattr(struct_ib_uverbs_flow_spec_action_count, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_action_count, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_action_count, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_action_count, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_action_count, 'handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_action_count, 'reserved1', field(12, ctypes.c_uint32))
class struct_ib_uverbs_flow_tunnel_filter(Struct): pass
struct_ib_uverbs_flow_tunnel_filter.SIZE = 4
struct_ib_uverbs_flow_tunnel_filter._fields_ = ['tunnel_id']
setattr(struct_ib_uverbs_flow_tunnel_filter, 'tunnel_id', field(0, ctypes.c_uint32))
class struct_ib_uverbs_flow_spec_tunnel(Struct): pass
struct_ib_uverbs_flow_spec_tunnel.SIZE = 16
struct_ib_uverbs_flow_spec_tunnel._fields_ = ['hdr', 'type', 'size', 'reserved', 'val', 'mask']
setattr(struct_ib_uverbs_flow_spec_tunnel, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_tunnel, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_tunnel, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_tunnel, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_tunnel, 'val', field(8, struct_ib_uverbs_flow_tunnel_filter))
setattr(struct_ib_uverbs_flow_spec_tunnel, 'mask', field(12, struct_ib_uverbs_flow_tunnel_filter))
class struct_ib_uverbs_flow_spec_esp_filter(Struct): pass
struct_ib_uverbs_flow_spec_esp_filter.SIZE = 8
struct_ib_uverbs_flow_spec_esp_filter._fields_ = ['spi', 'seq']
setattr(struct_ib_uverbs_flow_spec_esp_filter, 'spi', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_esp_filter, 'seq', field(4, ctypes.c_uint32))
class struct_ib_uverbs_flow_spec_esp(Struct): pass
struct_ib_uverbs_flow_spec_esp.SIZE = 24
struct_ib_uverbs_flow_spec_esp._fields_ = ['hdr', 'type', 'size', 'reserved', 'val', 'mask']
setattr(struct_ib_uverbs_flow_spec_esp, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_esp, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_esp, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_esp, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_esp, 'val', field(8, struct_ib_uverbs_flow_spec_esp_filter))
setattr(struct_ib_uverbs_flow_spec_esp, 'mask', field(16, struct_ib_uverbs_flow_spec_esp_filter))
class struct_ib_uverbs_flow_gre_filter(Struct): pass
struct_ib_uverbs_flow_gre_filter.SIZE = 8
struct_ib_uverbs_flow_gre_filter._fields_ = ['c_ks_res0_ver', 'protocol', 'key']
setattr(struct_ib_uverbs_flow_gre_filter, 'c_ks_res0_ver', field(0, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_gre_filter, 'protocol', field(2, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_gre_filter, 'key', field(4, ctypes.c_uint32))
class struct_ib_uverbs_flow_spec_gre(Struct): pass
struct_ib_uverbs_flow_spec_gre.SIZE = 24
struct_ib_uverbs_flow_spec_gre._fields_ = ['hdr', 'type', 'size', 'reserved', 'val', 'mask']
setattr(struct_ib_uverbs_flow_spec_gre, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_gre, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_gre, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_gre, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_gre, 'val', field(8, struct_ib_uverbs_flow_gre_filter))
setattr(struct_ib_uverbs_flow_spec_gre, 'mask', field(16, struct_ib_uverbs_flow_gre_filter))
class struct_ib_uverbs_flow_mpls_filter(Struct): pass
struct_ib_uverbs_flow_mpls_filter.SIZE = 4
struct_ib_uverbs_flow_mpls_filter._fields_ = ['label']
setattr(struct_ib_uverbs_flow_mpls_filter, 'label', field(0, ctypes.c_uint32))
class struct_ib_uverbs_flow_spec_mpls(Struct): pass
struct_ib_uverbs_flow_spec_mpls.SIZE = 16
struct_ib_uverbs_flow_spec_mpls._fields_ = ['hdr', 'type', 'size', 'reserved', 'val', 'mask']
setattr(struct_ib_uverbs_flow_spec_mpls, 'hdr', field(0, struct_ib_uverbs_flow_spec_hdr))
setattr(struct_ib_uverbs_flow_spec_mpls, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_spec_mpls, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_mpls, 'reserved', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_spec_mpls, 'val', field(8, struct_ib_uverbs_flow_mpls_filter))
setattr(struct_ib_uverbs_flow_spec_mpls, 'mask', field(12, struct_ib_uverbs_flow_mpls_filter))
class struct_ib_uverbs_flow_attr(Struct): pass
struct_ib_uverbs_flow_attr.SIZE = 16
struct_ib_uverbs_flow_attr._fields_ = ['type', 'size', 'priority', 'num_of_specs', 'reserved', 'port', 'flags', 'flow_specs']
setattr(struct_ib_uverbs_flow_attr, 'type', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_attr, 'size', field(4, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_attr, 'priority', field(6, ctypes.c_uint16))
setattr(struct_ib_uverbs_flow_attr, 'num_of_specs', field(8, ctypes.c_ubyte))
setattr(struct_ib_uverbs_flow_attr, 'reserved', field(9, Array(ctypes.c_ubyte, 2)))
setattr(struct_ib_uverbs_flow_attr, 'port', field(11, ctypes.c_ubyte))
setattr(struct_ib_uverbs_flow_attr, 'flags', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_flow_attr, 'flow_specs', field(16, Array(struct_ib_uverbs_flow_spec_hdr, 0)))
class struct_ib_uverbs_create_flow(Struct): pass
struct_ib_uverbs_create_flow.SIZE = 24
struct_ib_uverbs_create_flow._fields_ = ['comp_mask', 'qp_handle', 'flow_attr']
setattr(struct_ib_uverbs_create_flow, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_flow, 'qp_handle', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_flow, 'flow_attr', field(8, struct_ib_uverbs_flow_attr))
class struct_ib_uverbs_create_flow_resp(Struct): pass
struct_ib_uverbs_create_flow_resp.SIZE = 8
struct_ib_uverbs_create_flow_resp._fields_ = ['comp_mask', 'flow_handle']
setattr(struct_ib_uverbs_create_flow_resp, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_flow_resp, 'flow_handle', field(4, ctypes.c_uint32))
class struct_ib_uverbs_destroy_flow(Struct): pass
struct_ib_uverbs_destroy_flow.SIZE = 8
struct_ib_uverbs_destroy_flow._fields_ = ['comp_mask', 'flow_handle']
setattr(struct_ib_uverbs_destroy_flow, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_destroy_flow, 'flow_handle', field(4, ctypes.c_uint32))
class struct_ib_uverbs_create_srq(Struct): pass
struct_ib_uverbs_create_srq.SIZE = 32
struct_ib_uverbs_create_srq._fields_ = ['response', 'user_handle', 'pd_handle', 'max_wr', 'max_sge', 'srq_limit', 'driver_data']
setattr(struct_ib_uverbs_create_srq, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_srq, 'user_handle', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_srq, 'pd_handle', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_srq, 'max_wr', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_srq, 'max_sge', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_srq, 'srq_limit', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_srq, 'driver_data', field(32, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_create_xsrq(Struct): pass
struct_ib_uverbs_create_xsrq.SIZE = 48
struct_ib_uverbs_create_xsrq._fields_ = ['response', 'user_handle', 'srq_type', 'pd_handle', 'max_wr', 'max_sge', 'srq_limit', 'max_num_tags', 'xrcd_handle', 'cq_handle', 'driver_data']
setattr(struct_ib_uverbs_create_xsrq, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_xsrq, 'user_handle', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_create_xsrq, 'srq_type', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_xsrq, 'pd_handle', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_xsrq, 'max_wr', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_xsrq, 'max_sge', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_xsrq, 'srq_limit', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_xsrq, 'max_num_tags', field(36, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_xsrq, 'xrcd_handle', field(40, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_xsrq, 'cq_handle', field(44, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_xsrq, 'driver_data', field(48, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_create_srq_resp(Struct): pass
struct_ib_uverbs_create_srq_resp.SIZE = 16
struct_ib_uverbs_create_srq_resp._fields_ = ['srq_handle', 'max_wr', 'max_sge', 'srqn', 'driver_data']
setattr(struct_ib_uverbs_create_srq_resp, 'srq_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_srq_resp, 'max_wr', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_srq_resp, 'max_sge', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_srq_resp, 'srqn', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_create_srq_resp, 'driver_data', field(16, Array(ctypes.c_uint32, 0)))
class struct_ib_uverbs_modify_srq(Struct): pass
struct_ib_uverbs_modify_srq.SIZE = 16
struct_ib_uverbs_modify_srq._fields_ = ['srq_handle', 'attr_mask', 'max_wr', 'srq_limit', 'driver_data']
setattr(struct_ib_uverbs_modify_srq, 'srq_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_srq, 'attr_mask', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_srq, 'max_wr', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_srq, 'srq_limit', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_modify_srq, 'driver_data', field(16, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_query_srq(Struct): pass
struct_ib_uverbs_query_srq.SIZE = 16
struct_ib_uverbs_query_srq._fields_ = ['response', 'srq_handle', 'reserved', 'driver_data']
setattr(struct_ib_uverbs_query_srq, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_query_srq, 'srq_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_srq, 'reserved', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_srq, 'driver_data', field(16, Array(ctypes.c_uint64, 0)))
class struct_ib_uverbs_query_srq_resp(Struct): pass
struct_ib_uverbs_query_srq_resp.SIZE = 16
struct_ib_uverbs_query_srq_resp._fields_ = ['max_wr', 'max_sge', 'srq_limit', 'reserved']
setattr(struct_ib_uverbs_query_srq_resp, 'max_wr', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_srq_resp, 'max_sge', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_srq_resp, 'srq_limit', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_query_srq_resp, 'reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_destroy_srq(Struct): pass
struct_ib_uverbs_destroy_srq.SIZE = 16
struct_ib_uverbs_destroy_srq._fields_ = ['response', 'srq_handle', 'reserved']
setattr(struct_ib_uverbs_destroy_srq, 'response', field(0, ctypes.c_uint64))
setattr(struct_ib_uverbs_destroy_srq, 'srq_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_destroy_srq, 'reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_destroy_srq_resp(Struct): pass
struct_ib_uverbs_destroy_srq_resp.SIZE = 4
struct_ib_uverbs_destroy_srq_resp._fields_ = ['events_reported']
setattr(struct_ib_uverbs_destroy_srq_resp, 'events_reported', field(0, ctypes.c_uint32))
class struct_ib_uverbs_ex_create_wq(Struct): pass
struct_ib_uverbs_ex_create_wq.SIZE = 40
struct_ib_uverbs_ex_create_wq._fields_ = ['comp_mask', 'wq_type', 'user_handle', 'pd_handle', 'cq_handle', 'max_wr', 'max_sge', 'create_flags', 'reserved']
setattr(struct_ib_uverbs_ex_create_wq, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq, 'wq_type', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq, 'user_handle', field(8, ctypes.c_uint64))
setattr(struct_ib_uverbs_ex_create_wq, 'pd_handle', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq, 'cq_handle', field(20, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq, 'max_wr', field(24, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq, 'max_sge', field(28, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq, 'create_flags', field(32, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq, 'reserved', field(36, ctypes.c_uint32))
class struct_ib_uverbs_ex_create_wq_resp(Struct): pass
struct_ib_uverbs_ex_create_wq_resp.SIZE = 24
struct_ib_uverbs_ex_create_wq_resp._fields_ = ['comp_mask', 'response_length', 'wq_handle', 'max_wr', 'max_sge', 'wqn']
setattr(struct_ib_uverbs_ex_create_wq_resp, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq_resp, 'response_length', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq_resp, 'wq_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq_resp, 'max_wr', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq_resp, 'max_sge', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_wq_resp, 'wqn', field(20, ctypes.c_uint32))
class struct_ib_uverbs_ex_destroy_wq(Struct): pass
struct_ib_uverbs_ex_destroy_wq.SIZE = 8
struct_ib_uverbs_ex_destroy_wq._fields_ = ['comp_mask', 'wq_handle']
setattr(struct_ib_uverbs_ex_destroy_wq, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_destroy_wq, 'wq_handle', field(4, ctypes.c_uint32))
class struct_ib_uverbs_ex_destroy_wq_resp(Struct): pass
struct_ib_uverbs_ex_destroy_wq_resp.SIZE = 16
struct_ib_uverbs_ex_destroy_wq_resp._fields_ = ['comp_mask', 'response_length', 'events_reported', 'reserved']
setattr(struct_ib_uverbs_ex_destroy_wq_resp, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_destroy_wq_resp, 'response_length', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_destroy_wq_resp, 'events_reported', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_destroy_wq_resp, 'reserved', field(12, ctypes.c_uint32))
class struct_ib_uverbs_ex_modify_wq(Struct): pass
struct_ib_uverbs_ex_modify_wq.SIZE = 24
struct_ib_uverbs_ex_modify_wq._fields_ = ['attr_mask', 'wq_handle', 'wq_state', 'curr_wq_state', 'flags', 'flags_mask']
setattr(struct_ib_uverbs_ex_modify_wq, 'attr_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_wq, 'wq_handle', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_wq, 'wq_state', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_wq, 'curr_wq_state', field(12, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_wq, 'flags', field(16, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_wq, 'flags_mask', field(20, ctypes.c_uint32))
class struct_ib_uverbs_ex_create_rwq_ind_table(Struct): pass
struct_ib_uverbs_ex_create_rwq_ind_table.SIZE = 8
struct_ib_uverbs_ex_create_rwq_ind_table._fields_ = ['comp_mask', 'log_ind_tbl_size', 'wq_handles']
setattr(struct_ib_uverbs_ex_create_rwq_ind_table, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_rwq_ind_table, 'log_ind_tbl_size', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_rwq_ind_table, 'wq_handles', field(8, Array(ctypes.c_uint32, 0)))
class struct_ib_uverbs_ex_create_rwq_ind_table_resp(Struct): pass
struct_ib_uverbs_ex_create_rwq_ind_table_resp.SIZE = 16
struct_ib_uverbs_ex_create_rwq_ind_table_resp._fields_ = ['comp_mask', 'response_length', 'ind_tbl_handle', 'ind_tbl_num']
setattr(struct_ib_uverbs_ex_create_rwq_ind_table_resp, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_rwq_ind_table_resp, 'response_length', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_rwq_ind_table_resp, 'ind_tbl_handle', field(8, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_create_rwq_ind_table_resp, 'ind_tbl_num', field(12, ctypes.c_uint32))
class struct_ib_uverbs_ex_destroy_rwq_ind_table(Struct): pass
struct_ib_uverbs_ex_destroy_rwq_ind_table.SIZE = 8
struct_ib_uverbs_ex_destroy_rwq_ind_table._fields_ = ['comp_mask', 'ind_tbl_handle']
setattr(struct_ib_uverbs_ex_destroy_rwq_ind_table, 'comp_mask', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_destroy_rwq_ind_table, 'ind_tbl_handle', field(4, ctypes.c_uint32))
class struct_ib_uverbs_cq_moderation(Struct): pass
struct_ib_uverbs_cq_moderation.SIZE = 4
struct_ib_uverbs_cq_moderation._fields_ = ['cq_count', 'cq_period']
setattr(struct_ib_uverbs_cq_moderation, 'cq_count', field(0, ctypes.c_uint16))
setattr(struct_ib_uverbs_cq_moderation, 'cq_period', field(2, ctypes.c_uint16))
class struct_ib_uverbs_ex_modify_cq(Struct): pass
struct_ib_uverbs_ex_modify_cq.SIZE = 16
struct_ib_uverbs_ex_modify_cq._fields_ = ['cq_handle', 'attr_mask', 'attr', 'reserved']
setattr(struct_ib_uverbs_ex_modify_cq, 'cq_handle', field(0, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_cq, 'attr_mask', field(4, ctypes.c_uint32))
setattr(struct_ib_uverbs_ex_modify_cq, 'attr', field(8, struct_ib_uverbs_cq_moderation))
setattr(struct_ib_uverbs_ex_modify_cq, 'reserved', field(12, ctypes.c_uint32))
enum_ib_uverbs_device_cap_flags = CEnum(ctypes.c_uint64)
IB_UVERBS_DEVICE_RESIZE_MAX_WR = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RESIZE_MAX_WR', 1)
IB_UVERBS_DEVICE_BAD_PKEY_CNTR = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_BAD_PKEY_CNTR', 2)
IB_UVERBS_DEVICE_BAD_QKEY_CNTR = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_BAD_QKEY_CNTR', 4)
IB_UVERBS_DEVICE_RAW_MULTI = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RAW_MULTI', 8)
IB_UVERBS_DEVICE_AUTO_PATH_MIG = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_AUTO_PATH_MIG', 16)
IB_UVERBS_DEVICE_CHANGE_PHY_PORT = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_CHANGE_PHY_PORT', 32)
IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE', 64)
IB_UVERBS_DEVICE_CURR_QP_STATE_MOD = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_CURR_QP_STATE_MOD', 128)
IB_UVERBS_DEVICE_SHUTDOWN_PORT = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_SHUTDOWN_PORT', 256)
IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT', 1024)
IB_UVERBS_DEVICE_SYS_IMAGE_GUID = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_SYS_IMAGE_GUID', 2048)
IB_UVERBS_DEVICE_RC_RNR_NAK_GEN = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RC_RNR_NAK_GEN', 4096)
IB_UVERBS_DEVICE_SRQ_RESIZE = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_SRQ_RESIZE', 8192)
IB_UVERBS_DEVICE_N_NOTIFY_CQ = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_N_NOTIFY_CQ', 16384)
IB_UVERBS_DEVICE_MEM_WINDOW = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MEM_WINDOW', 131072)
IB_UVERBS_DEVICE_UD_IP_CSUM = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_UD_IP_CSUM', 262144)
IB_UVERBS_DEVICE_XRC = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_XRC', 1048576)
IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS', 2097152)
IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A', 8388608)
IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B', 16777216)
IB_UVERBS_DEVICE_RC_IP_CSUM = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RC_IP_CSUM', 33554432)
IB_UVERBS_DEVICE_RAW_IP_CSUM = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RAW_IP_CSUM', 67108864)
IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING', 536870912)
IB_UVERBS_DEVICE_RAW_SCATTER_FCS = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RAW_SCATTER_FCS', 17179869184)
IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING', 68719476736)
IB_UVERBS_DEVICE_FLUSH_GLOBAL = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_FLUSH_GLOBAL', 274877906944)
IB_UVERBS_DEVICE_FLUSH_PERSISTENT = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_FLUSH_PERSISTENT', 549755813888)
IB_UVERBS_DEVICE_ATOMIC_WRITE = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_ATOMIC_WRITE', 1099511627776)

enum_ib_uverbs_raw_packet_caps = CEnum(ctypes.c_uint32)
IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING', 1)
IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS', 2)
IB_UVERBS_RAW_PACKET_CAP_IP_CSUM = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_IP_CSUM', 4)
IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP', 8)

vext_field_avail = lambda type,fld,sz: (offsetof(type, fld) < (sz))
IBV_DEVICE_RAW_SCATTER_FCS = (1 << 34)
IBV_DEVICE_PCI_WRITE_END_PADDING = (1 << 36)
ibv_query_port = lambda context,port_num,port_attr: ___ibv_query_port(context, port_num, port_attr)
ibv_reg_mr = lambda pd,addr,length,access: __ibv_reg_mr(pd, addr, length, access, __builtin_constant_p( ((int)(access) & IBV_ACCESS_OPTIONAL_RANGE) == 0))
ibv_reg_mr_iova = lambda pd,addr,length,iova,access: __ibv_reg_mr_iova(pd, addr, length, iova, access, __builtin_constant_p( ((access) & IBV_ACCESS_OPTIONAL_RANGE) == 0))
ETHERNET_LL_SIZE = 6
IB_ROCE_UDP_ENCAP_VALID_PORT_MIN = (0xC000)
IB_ROCE_UDP_ENCAP_VALID_PORT_MAX = (0xFFFF)
IB_GRH_FLOWLABEL_MASK = (0x000FFFFF)
IBV_FLOW_ACTION_ESP_KEYMAT_AES_GCM = IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM
IBV_FLOW_ACTION_IV_ALGO_SEQ = IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ
IBV_FLOW_ACTION_ESP_REPLAY_NONE = IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE
IBV_FLOW_ACTION_ESP_REPLAY_BMP = IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP
IBV_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO
IBV_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD
IBV_FLOW_ACTION_ESP_FLAGS_TUNNEL = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL
IBV_FLOW_ACTION_ESP_FLAGS_TRANSPORT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT
IBV_FLOW_ACTION_ESP_FLAGS_DECRYPT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT
IBV_FLOW_ACTION_ESP_FLAGS_ENCRYPT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT
IBV_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW
IBV_ADVISE_MR_ADVICE_PREFETCH = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH
IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE
IBV_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT
IBV_ADVISE_MR_FLAG_FLUSH = IB_UVERBS_ADVISE_MR_FLAG_FLUSH
IBV_QPF_GRH_REQUIRED = IB_UVERBS_QPF_GRH_REQUIRED
IBV_ACCESS_OPTIONAL_RANGE = IB_UVERBS_ACCESS_OPTIONAL_RANGE
IB_UVERBS_ACCESS_OPTIONAL_FIRST = (1 << 20)
IB_UVERBS_ACCESS_OPTIONAL_LAST = (1 << 29)
IB_USER_VERBS_ABI_VERSION = 6
IB_USER_VERBS_CMD_THRESHOLD = 50
IB_USER_VERBS_CMD_COMMAND_MASK = 0xff
IB_USER_VERBS_CMD_FLAG_EXTENDED = 0x80000000
IB_USER_VERBS_MAX_LOG_IND_TBL_SIZE = 0x0d
IB_DEVICE_NAME_MAX = 64