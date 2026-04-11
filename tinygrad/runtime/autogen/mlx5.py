# mypy: disable-error-code="empty-body"
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
__u8: TypeAlias = ctypes.c_ubyte
@c.record
class struct_mlx5_cmd_layout(c.Struct):
  SIZE = 64
  type: 'int'
  rsvd0: 'list[int]'
  inlen: 'int'
  in_ptr: 'int'
  _in: 'list[int]'
  out: 'list[int]'
  out_ptr: 'int'
  outlen: 'int'
  token: 'int'
  sig: 'int'
  rsvd1: 'int'
  status_own: 'int'
struct_mlx5_cmd_layout.register_fields([('type', ctypes.c_ubyte, 0), ('rsvd0', (ctypes.c_ubyte * 3), 1), ('inlen', ctypes.c_uint32, 4), ('in_ptr', ctypes.c_uint64, 8), ('_in', (ctypes.c_uint32 * 4), 16), ('out', (ctypes.c_uint32 * 4), 32), ('out_ptr', ctypes.c_uint64, 48), ('outlen', ctypes.c_uint32, 56), ('token', ctypes.c_ubyte, 60), ('sig', ctypes.c_ubyte, 61), ('rsvd1', ctypes.c_ubyte, 62), ('status_own', ctypes.c_ubyte, 63)])
@c.record
class struct_mlx5_cmd_prot_block(c.Struct):
  SIZE = 576
  data: 'list[int]'
  rsvd0: 'list[int]'
  next: 'int'
  block_num: 'int'
  rsvd1: 'int'
  token: 'int'
  ctrl_sig: 'int'
  sig: 'int'
struct_mlx5_cmd_prot_block.register_fields([('data', (ctypes.c_ubyte * 512), 0), ('rsvd0', (ctypes.c_ubyte * 48), 512), ('next', ctypes.c_uint64, 560), ('block_num', ctypes.c_uint32, 568), ('rsvd1', ctypes.c_ubyte, 572), ('token', ctypes.c_ubyte, 573), ('ctrl_sig', ctypes.c_ubyte, 574), ('sig', ctypes.c_ubyte, 575)])
@c.record
class struct_mlx5_init_seg(c.Struct):
  SIZE = 512
  fw_rev: 'int'
  cmdif_rev_fw_sub: 'int'
  rsvd0: 'list[int]'
  cmdq_addr_h: 'int'
  cmdq_addr_l_sz: 'int'
  cmd_dbell: 'int'
  rsvd1: 'list[int]'
  initializing: 'int'
struct_mlx5_init_seg.register_fields([('fw_rev', ctypes.c_uint32, 0), ('cmdif_rev_fw_sub', ctypes.c_uint32, 4), ('rsvd0', (ctypes.c_uint32 * 2), 8), ('cmdq_addr_h', ctypes.c_uint32, 16), ('cmdq_addr_l_sz', ctypes.c_uint32, 20), ('cmd_dbell', ctypes.c_uint32, 24), ('rsvd1', (ctypes.c_uint32 * 120), 28), ('initializing', ctypes.c_uint32, 508)])
_anonenum0: dict[int, str] = {(MLX5_EVENT_TYPE_CODING_COMPLETION_EVENTS:=0): 'MLX5_EVENT_TYPE_CODING_COMPLETION_EVENTS', (MLX5_EVENT_TYPE_CODING_PATH_MIGRATED_SUCCEEDED:=1): 'MLX5_EVENT_TYPE_CODING_PATH_MIGRATED_SUCCEEDED', (MLX5_EVENT_TYPE_CODING_COMMUNICATION_ESTABLISHED:=2): 'MLX5_EVENT_TYPE_CODING_COMMUNICATION_ESTABLISHED', (MLX5_EVENT_TYPE_CODING_SEND_QUEUE_DRAINED:=3): 'MLX5_EVENT_TYPE_CODING_SEND_QUEUE_DRAINED', (MLX5_EVENT_TYPE_CODING_LAST_WQE_REACHED:=19): 'MLX5_EVENT_TYPE_CODING_LAST_WQE_REACHED', (MLX5_EVENT_TYPE_CODING_SRQ_LIMIT:=20): 'MLX5_EVENT_TYPE_CODING_SRQ_LIMIT', (MLX5_EVENT_TYPE_CODING_DCT_ALL_CONNECTIONS_CLOSED:=28): 'MLX5_EVENT_TYPE_CODING_DCT_ALL_CONNECTIONS_CLOSED', (MLX5_EVENT_TYPE_CODING_DCT_ACCESS_KEY_VIOLATION:=29): 'MLX5_EVENT_TYPE_CODING_DCT_ACCESS_KEY_VIOLATION', (MLX5_EVENT_TYPE_CODING_CQ_ERROR:=4): 'MLX5_EVENT_TYPE_CODING_CQ_ERROR', (MLX5_EVENT_TYPE_CODING_LOCAL_WQ_CATASTROPHIC_ERROR:=5): 'MLX5_EVENT_TYPE_CODING_LOCAL_WQ_CATASTROPHIC_ERROR', (MLX5_EVENT_TYPE_CODING_PATH_MIGRATION_FAILED:=7): 'MLX5_EVENT_TYPE_CODING_PATH_MIGRATION_FAILED', (MLX5_EVENT_TYPE_CODING_PAGE_FAULT_EVENT:=12): 'MLX5_EVENT_TYPE_CODING_PAGE_FAULT_EVENT', (MLX5_EVENT_TYPE_CODING_INVALID_REQUEST_LOCAL_WQ_ERROR:=16): 'MLX5_EVENT_TYPE_CODING_INVALID_REQUEST_LOCAL_WQ_ERROR', (MLX5_EVENT_TYPE_CODING_LOCAL_ACCESS_VIOLATION_WQ_ERROR:=17): 'MLX5_EVENT_TYPE_CODING_LOCAL_ACCESS_VIOLATION_WQ_ERROR', (MLX5_EVENT_TYPE_CODING_LOCAL_SRQ_CATASTROPHIC_ERROR:=18): 'MLX5_EVENT_TYPE_CODING_LOCAL_SRQ_CATASTROPHIC_ERROR', (MLX5_EVENT_TYPE_CODING_INTERNAL_ERROR:=8): 'MLX5_EVENT_TYPE_CODING_INTERNAL_ERROR', (MLX5_EVENT_TYPE_CODING_PORT_STATE_CHANGE:=9): 'MLX5_EVENT_TYPE_CODING_PORT_STATE_CHANGE', (MLX5_EVENT_TYPE_CODING_GPIO_EVENT:=21): 'MLX5_EVENT_TYPE_CODING_GPIO_EVENT', (MLX5_EVENT_TYPE_CODING_REMOTE_CONFIGURATION_PROTOCOL_EVENT:=25): 'MLX5_EVENT_TYPE_CODING_REMOTE_CONFIGURATION_PROTOCOL_EVENT', (MLX5_EVENT_TYPE_CODING_DOORBELL_BLUEFLAME_CONGESTION_EVENT:=26): 'MLX5_EVENT_TYPE_CODING_DOORBELL_BLUEFLAME_CONGESTION_EVENT', (MLX5_EVENT_TYPE_CODING_STALL_VL_EVENT:=27): 'MLX5_EVENT_TYPE_CODING_STALL_VL_EVENT', (MLX5_EVENT_TYPE_CODING_DROPPED_PACKET_LOGGED_EVENT:=31): 'MLX5_EVENT_TYPE_CODING_DROPPED_PACKET_LOGGED_EVENT', (MLX5_EVENT_TYPE_CODING_COMMAND_INTERFACE_COMPLETION:=10): 'MLX5_EVENT_TYPE_CODING_COMMAND_INTERFACE_COMPLETION', (MLX5_EVENT_TYPE_CODING_PAGE_REQUEST:=11): 'MLX5_EVENT_TYPE_CODING_PAGE_REQUEST', (MLX5_EVENT_TYPE_CODING_FPGA_ERROR:=32): 'MLX5_EVENT_TYPE_CODING_FPGA_ERROR', (MLX5_EVENT_TYPE_CODING_FPGA_QP_ERROR:=33): 'MLX5_EVENT_TYPE_CODING_FPGA_QP_ERROR'}
_anonenum1: dict[int, str] = {(MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE:=0): 'MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE', (MLX5_SET_HCA_CAP_OP_MOD_ETHERNET_OFFLOADS:=1): 'MLX5_SET_HCA_CAP_OP_MOD_ETHERNET_OFFLOADS', (MLX5_SET_HCA_CAP_OP_MOD_ODP:=2): 'MLX5_SET_HCA_CAP_OP_MOD_ODP', (MLX5_SET_HCA_CAP_OP_MOD_ATOMIC:=3): 'MLX5_SET_HCA_CAP_OP_MOD_ATOMIC', (MLX5_SET_HCA_CAP_OP_MOD_ROCE:=4): 'MLX5_SET_HCA_CAP_OP_MOD_ROCE', (MLX5_SET_HCA_CAP_OP_MOD_IPSEC:=21): 'MLX5_SET_HCA_CAP_OP_MOD_IPSEC', (MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE2:=32): 'MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE2', (MLX5_SET_HCA_CAP_OP_MOD_PORT_SELECTION:=37): 'MLX5_SET_HCA_CAP_OP_MOD_PORT_SELECTION'}
_anonenum2: dict[int, str] = {(MLX5_SHARED_RESOURCE_UID:=65535): 'MLX5_SHARED_RESOURCE_UID'}
_anonenum3: dict[int, str] = {(MLX5_OBJ_TYPE_SW_ICM:=8): 'MLX5_OBJ_TYPE_SW_ICM', (MLX5_OBJ_TYPE_GENEVE_TLV_OPT:=11): 'MLX5_OBJ_TYPE_GENEVE_TLV_OPT', (MLX5_OBJ_TYPE_VIRTIO_NET_Q:=13): 'MLX5_OBJ_TYPE_VIRTIO_NET_Q', (MLX5_OBJ_TYPE_VIRTIO_Q_COUNTERS:=28): 'MLX5_OBJ_TYPE_VIRTIO_Q_COUNTERS', (MLX5_OBJ_TYPE_MATCH_DEFINER:=24): 'MLX5_OBJ_TYPE_MATCH_DEFINER', (MLX5_OBJ_TYPE_HEADER_MODIFY_ARGUMENT:=35): 'MLX5_OBJ_TYPE_HEADER_MODIFY_ARGUMENT', (MLX5_OBJ_TYPE_STC:=64): 'MLX5_OBJ_TYPE_STC', (MLX5_OBJ_TYPE_RTC:=65): 'MLX5_OBJ_TYPE_RTC', (MLX5_OBJ_TYPE_STE:=66): 'MLX5_OBJ_TYPE_STE', (MLX5_OBJ_TYPE_MODIFY_HDR_PATTERN:=67): 'MLX5_OBJ_TYPE_MODIFY_HDR_PATTERN', (MLX5_OBJ_TYPE_PAGE_TRACK:=70): 'MLX5_OBJ_TYPE_PAGE_TRACK', (MLX5_OBJ_TYPE_MKEY:=65281): 'MLX5_OBJ_TYPE_MKEY', (MLX5_OBJ_TYPE_QP:=65282): 'MLX5_OBJ_TYPE_QP', (MLX5_OBJ_TYPE_PSV:=65283): 'MLX5_OBJ_TYPE_PSV', (MLX5_OBJ_TYPE_RMP:=65284): 'MLX5_OBJ_TYPE_RMP', (MLX5_OBJ_TYPE_XRC_SRQ:=65285): 'MLX5_OBJ_TYPE_XRC_SRQ', (MLX5_OBJ_TYPE_RQ:=65286): 'MLX5_OBJ_TYPE_RQ', (MLX5_OBJ_TYPE_SQ:=65287): 'MLX5_OBJ_TYPE_SQ', (MLX5_OBJ_TYPE_TIR:=65288): 'MLX5_OBJ_TYPE_TIR', (MLX5_OBJ_TYPE_TIS:=65289): 'MLX5_OBJ_TYPE_TIS', (MLX5_OBJ_TYPE_DCT:=65290): 'MLX5_OBJ_TYPE_DCT', (MLX5_OBJ_TYPE_XRQ:=65291): 'MLX5_OBJ_TYPE_XRQ', (MLX5_OBJ_TYPE_RQT:=65294): 'MLX5_OBJ_TYPE_RQT', (MLX5_OBJ_TYPE_FLOW_COUNTER:=65295): 'MLX5_OBJ_TYPE_FLOW_COUNTER', (MLX5_OBJ_TYPE_CQ:=65296): 'MLX5_OBJ_TYPE_CQ', (MLX5_OBJ_TYPE_FT_ALIAS:=65301): 'MLX5_OBJ_TYPE_FT_ALIAS'}
_anonenum4: dict[int, str] = {(MLX5_GENERAL_OBJ_TYPES_CAP_SW_ICM:=256): 'MLX5_GENERAL_OBJ_TYPES_CAP_SW_ICM', (MLX5_GENERAL_OBJ_TYPES_CAP_GENEVE_TLV_OPT:=2048): 'MLX5_GENERAL_OBJ_TYPES_CAP_GENEVE_TLV_OPT', (MLX5_GENERAL_OBJ_TYPES_CAP_VIRTIO_NET_Q:=8192): 'MLX5_GENERAL_OBJ_TYPES_CAP_VIRTIO_NET_Q', (MLX5_GENERAL_OBJ_TYPES_CAP_HEADER_MODIFY_ARGUMENT:=34359738368): 'MLX5_GENERAL_OBJ_TYPES_CAP_HEADER_MODIFY_ARGUMENT', (MLX5_GENERAL_OBJ_TYPES_CAP_MACSEC_OFFLOAD:=549755813888): 'MLX5_GENERAL_OBJ_TYPES_CAP_MACSEC_OFFLOAD'}
_anonenum5: dict[int, str] = {(MLX5_CMD_OP_QUERY_HCA_CAP:=256): 'MLX5_CMD_OP_QUERY_HCA_CAP', (MLX5_CMD_OP_QUERY_ADAPTER:=257): 'MLX5_CMD_OP_QUERY_ADAPTER', (MLX5_CMD_OP_INIT_HCA:=258): 'MLX5_CMD_OP_INIT_HCA', (MLX5_CMD_OP_TEARDOWN_HCA:=259): 'MLX5_CMD_OP_TEARDOWN_HCA', (MLX5_CMD_OP_ENABLE_HCA:=260): 'MLX5_CMD_OP_ENABLE_HCA', (MLX5_CMD_OP_DISABLE_HCA:=261): 'MLX5_CMD_OP_DISABLE_HCA', (MLX5_CMD_OP_QUERY_PAGES:=263): 'MLX5_CMD_OP_QUERY_PAGES', (MLX5_CMD_OP_MANAGE_PAGES:=264): 'MLX5_CMD_OP_MANAGE_PAGES', (MLX5_CMD_OP_SET_HCA_CAP:=265): 'MLX5_CMD_OP_SET_HCA_CAP', (MLX5_CMD_OP_QUERY_ISSI:=266): 'MLX5_CMD_OP_QUERY_ISSI', (MLX5_CMD_OP_SET_ISSI:=267): 'MLX5_CMD_OP_SET_ISSI', (MLX5_CMD_OP_SET_DRIVER_VERSION:=269): 'MLX5_CMD_OP_SET_DRIVER_VERSION', (MLX5_CMD_OP_QUERY_SF_PARTITION:=273): 'MLX5_CMD_OP_QUERY_SF_PARTITION', (MLX5_CMD_OP_ALLOC_SF:=275): 'MLX5_CMD_OP_ALLOC_SF', (MLX5_CMD_OP_DEALLOC_SF:=276): 'MLX5_CMD_OP_DEALLOC_SF', (MLX5_CMD_OP_SUSPEND_VHCA:=277): 'MLX5_CMD_OP_SUSPEND_VHCA', (MLX5_CMD_OP_RESUME_VHCA:=278): 'MLX5_CMD_OP_RESUME_VHCA', (MLX5_CMD_OP_QUERY_VHCA_MIGRATION_STATE:=279): 'MLX5_CMD_OP_QUERY_VHCA_MIGRATION_STATE', (MLX5_CMD_OP_SAVE_VHCA_STATE:=280): 'MLX5_CMD_OP_SAVE_VHCA_STATE', (MLX5_CMD_OP_LOAD_VHCA_STATE:=281): 'MLX5_CMD_OP_LOAD_VHCA_STATE', (MLX5_CMD_OP_CREATE_MKEY:=512): 'MLX5_CMD_OP_CREATE_MKEY', (MLX5_CMD_OP_QUERY_MKEY:=513): 'MLX5_CMD_OP_QUERY_MKEY', (MLX5_CMD_OP_DESTROY_MKEY:=514): 'MLX5_CMD_OP_DESTROY_MKEY', (MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS:=515): 'MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS', (MLX5_CMD_OP_PAGE_FAULT_RESUME:=516): 'MLX5_CMD_OP_PAGE_FAULT_RESUME', (MLX5_CMD_OP_ALLOC_MEMIC:=517): 'MLX5_CMD_OP_ALLOC_MEMIC', (MLX5_CMD_OP_DEALLOC_MEMIC:=518): 'MLX5_CMD_OP_DEALLOC_MEMIC', (MLX5_CMD_OP_MODIFY_MEMIC:=519): 'MLX5_CMD_OP_MODIFY_MEMIC', (MLX5_CMD_OP_CREATE_EQ:=769): 'MLX5_CMD_OP_CREATE_EQ', (MLX5_CMD_OP_DESTROY_EQ:=770): 'MLX5_CMD_OP_DESTROY_EQ', (MLX5_CMD_OP_QUERY_EQ:=771): 'MLX5_CMD_OP_QUERY_EQ', (MLX5_CMD_OP_GEN_EQE:=772): 'MLX5_CMD_OP_GEN_EQE', (MLX5_CMD_OP_CREATE_CQ:=1024): 'MLX5_CMD_OP_CREATE_CQ', (MLX5_CMD_OP_DESTROY_CQ:=1025): 'MLX5_CMD_OP_DESTROY_CQ', (MLX5_CMD_OP_QUERY_CQ:=1026): 'MLX5_CMD_OP_QUERY_CQ', (MLX5_CMD_OP_MODIFY_CQ:=1027): 'MLX5_CMD_OP_MODIFY_CQ', (MLX5_CMD_OP_CREATE_QP:=1280): 'MLX5_CMD_OP_CREATE_QP', (MLX5_CMD_OP_DESTROY_QP:=1281): 'MLX5_CMD_OP_DESTROY_QP', (MLX5_CMD_OP_RST2INIT_QP:=1282): 'MLX5_CMD_OP_RST2INIT_QP', (MLX5_CMD_OP_INIT2RTR_QP:=1283): 'MLX5_CMD_OP_INIT2RTR_QP', (MLX5_CMD_OP_RTR2RTS_QP:=1284): 'MLX5_CMD_OP_RTR2RTS_QP', (MLX5_CMD_OP_RTS2RTS_QP:=1285): 'MLX5_CMD_OP_RTS2RTS_QP', (MLX5_CMD_OP_SQERR2RTS_QP:=1286): 'MLX5_CMD_OP_SQERR2RTS_QP', (MLX5_CMD_OP_2ERR_QP:=1287): 'MLX5_CMD_OP_2ERR_QP', (MLX5_CMD_OP_2RST_QP:=1290): 'MLX5_CMD_OP_2RST_QP', (MLX5_CMD_OP_QUERY_QP:=1291): 'MLX5_CMD_OP_QUERY_QP', (MLX5_CMD_OP_SQD_RTS_QP:=1292): 'MLX5_CMD_OP_SQD_RTS_QP', (MLX5_CMD_OP_INIT2INIT_QP:=1294): 'MLX5_CMD_OP_INIT2INIT_QP', (MLX5_CMD_OP_CREATE_PSV:=1536): 'MLX5_CMD_OP_CREATE_PSV', (MLX5_CMD_OP_DESTROY_PSV:=1537): 'MLX5_CMD_OP_DESTROY_PSV', (MLX5_CMD_OP_CREATE_SRQ:=1792): 'MLX5_CMD_OP_CREATE_SRQ', (MLX5_CMD_OP_DESTROY_SRQ:=1793): 'MLX5_CMD_OP_DESTROY_SRQ', (MLX5_CMD_OP_QUERY_SRQ:=1794): 'MLX5_CMD_OP_QUERY_SRQ', (MLX5_CMD_OP_ARM_RQ:=1795): 'MLX5_CMD_OP_ARM_RQ', (MLX5_CMD_OP_CREATE_XRC_SRQ:=1797): 'MLX5_CMD_OP_CREATE_XRC_SRQ', (MLX5_CMD_OP_DESTROY_XRC_SRQ:=1798): 'MLX5_CMD_OP_DESTROY_XRC_SRQ', (MLX5_CMD_OP_QUERY_XRC_SRQ:=1799): 'MLX5_CMD_OP_QUERY_XRC_SRQ', (MLX5_CMD_OP_ARM_XRC_SRQ:=1800): 'MLX5_CMD_OP_ARM_XRC_SRQ', (MLX5_CMD_OP_CREATE_DCT:=1808): 'MLX5_CMD_OP_CREATE_DCT', (MLX5_CMD_OP_DESTROY_DCT:=1809): 'MLX5_CMD_OP_DESTROY_DCT', (MLX5_CMD_OP_DRAIN_DCT:=1810): 'MLX5_CMD_OP_DRAIN_DCT', (MLX5_CMD_OP_QUERY_DCT:=1811): 'MLX5_CMD_OP_QUERY_DCT', (MLX5_CMD_OP_ARM_DCT_FOR_KEY_VIOLATION:=1812): 'MLX5_CMD_OP_ARM_DCT_FOR_KEY_VIOLATION', (MLX5_CMD_OP_CREATE_XRQ:=1815): 'MLX5_CMD_OP_CREATE_XRQ', (MLX5_CMD_OP_DESTROY_XRQ:=1816): 'MLX5_CMD_OP_DESTROY_XRQ', (MLX5_CMD_OP_QUERY_XRQ:=1817): 'MLX5_CMD_OP_QUERY_XRQ', (MLX5_CMD_OP_ARM_XRQ:=1818): 'MLX5_CMD_OP_ARM_XRQ', (MLX5_CMD_OP_QUERY_XRQ_DC_PARAMS_ENTRY:=1829): 'MLX5_CMD_OP_QUERY_XRQ_DC_PARAMS_ENTRY', (MLX5_CMD_OP_SET_XRQ_DC_PARAMS_ENTRY:=1830): 'MLX5_CMD_OP_SET_XRQ_DC_PARAMS_ENTRY', (MLX5_CMD_OP_QUERY_XRQ_ERROR_PARAMS:=1831): 'MLX5_CMD_OP_QUERY_XRQ_ERROR_PARAMS', (MLX5_CMD_OP_RELEASE_XRQ_ERROR:=1833): 'MLX5_CMD_OP_RELEASE_XRQ_ERROR', (MLX5_CMD_OP_MODIFY_XRQ:=1834): 'MLX5_CMD_OP_MODIFY_XRQ', (MLX5_CMD_OPCODE_QUERY_DELEGATED_VHCA:=1842): 'MLX5_CMD_OPCODE_QUERY_DELEGATED_VHCA', (MLX5_CMD_OPCODE_CREATE_ESW_VPORT:=1843): 'MLX5_CMD_OPCODE_CREATE_ESW_VPORT', (MLX5_CMD_OPCODE_DESTROY_ESW_VPORT:=1844): 'MLX5_CMD_OPCODE_DESTROY_ESW_VPORT', (MLX5_CMD_OP_QUERY_ESW_FUNCTIONS:=1856): 'MLX5_CMD_OP_QUERY_ESW_FUNCTIONS', (MLX5_CMD_OP_QUERY_VPORT_STATE:=1872): 'MLX5_CMD_OP_QUERY_VPORT_STATE', (MLX5_CMD_OP_MODIFY_VPORT_STATE:=1873): 'MLX5_CMD_OP_MODIFY_VPORT_STATE', (MLX5_CMD_OP_QUERY_ESW_VPORT_CONTEXT:=1874): 'MLX5_CMD_OP_QUERY_ESW_VPORT_CONTEXT', (MLX5_CMD_OP_MODIFY_ESW_VPORT_CONTEXT:=1875): 'MLX5_CMD_OP_MODIFY_ESW_VPORT_CONTEXT', (MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT:=1876): 'MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT', (MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT:=1877): 'MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT', (MLX5_CMD_OP_QUERY_ROCE_ADDRESS:=1888): 'MLX5_CMD_OP_QUERY_ROCE_ADDRESS', (MLX5_CMD_OP_SET_ROCE_ADDRESS:=1889): 'MLX5_CMD_OP_SET_ROCE_ADDRESS', (MLX5_CMD_OP_QUERY_HCA_VPORT_CONTEXT:=1890): 'MLX5_CMD_OP_QUERY_HCA_VPORT_CONTEXT', (MLX5_CMD_OP_MODIFY_HCA_VPORT_CONTEXT:=1891): 'MLX5_CMD_OP_MODIFY_HCA_VPORT_CONTEXT', (MLX5_CMD_OP_QUERY_HCA_VPORT_GID:=1892): 'MLX5_CMD_OP_QUERY_HCA_VPORT_GID', (MLX5_CMD_OP_QUERY_HCA_VPORT_PKEY:=1893): 'MLX5_CMD_OP_QUERY_HCA_VPORT_PKEY', (MLX5_CMD_OP_QUERY_VNIC_ENV:=1903): 'MLX5_CMD_OP_QUERY_VNIC_ENV', (MLX5_CMD_OP_QUERY_VPORT_COUNTER:=1904): 'MLX5_CMD_OP_QUERY_VPORT_COUNTER', (MLX5_CMD_OP_ALLOC_Q_COUNTER:=1905): 'MLX5_CMD_OP_ALLOC_Q_COUNTER', (MLX5_CMD_OP_DEALLOC_Q_COUNTER:=1906): 'MLX5_CMD_OP_DEALLOC_Q_COUNTER', (MLX5_CMD_OP_QUERY_Q_COUNTER:=1907): 'MLX5_CMD_OP_QUERY_Q_COUNTER', (MLX5_CMD_OP_SET_MONITOR_COUNTER:=1908): 'MLX5_CMD_OP_SET_MONITOR_COUNTER', (MLX5_CMD_OP_ARM_MONITOR_COUNTER:=1909): 'MLX5_CMD_OP_ARM_MONITOR_COUNTER', (MLX5_CMD_OP_SET_PP_RATE_LIMIT:=1920): 'MLX5_CMD_OP_SET_PP_RATE_LIMIT', (MLX5_CMD_OP_QUERY_RATE_LIMIT:=1921): 'MLX5_CMD_OP_QUERY_RATE_LIMIT', (MLX5_CMD_OP_CREATE_SCHEDULING_ELEMENT:=1922): 'MLX5_CMD_OP_CREATE_SCHEDULING_ELEMENT', (MLX5_CMD_OP_DESTROY_SCHEDULING_ELEMENT:=1923): 'MLX5_CMD_OP_DESTROY_SCHEDULING_ELEMENT', (MLX5_CMD_OP_QUERY_SCHEDULING_ELEMENT:=1924): 'MLX5_CMD_OP_QUERY_SCHEDULING_ELEMENT', (MLX5_CMD_OP_MODIFY_SCHEDULING_ELEMENT:=1925): 'MLX5_CMD_OP_MODIFY_SCHEDULING_ELEMENT', (MLX5_CMD_OP_CREATE_QOS_PARA_VPORT:=1926): 'MLX5_CMD_OP_CREATE_QOS_PARA_VPORT', (MLX5_CMD_OP_DESTROY_QOS_PARA_VPORT:=1927): 'MLX5_CMD_OP_DESTROY_QOS_PARA_VPORT', (MLX5_CMD_OP_ALLOC_PD:=2048): 'MLX5_CMD_OP_ALLOC_PD', (MLX5_CMD_OP_DEALLOC_PD:=2049): 'MLX5_CMD_OP_DEALLOC_PD', (MLX5_CMD_OP_ALLOC_UAR:=2050): 'MLX5_CMD_OP_ALLOC_UAR', (MLX5_CMD_OP_DEALLOC_UAR:=2051): 'MLX5_CMD_OP_DEALLOC_UAR', (MLX5_CMD_OP_CONFIG_INT_MODERATION:=2052): 'MLX5_CMD_OP_CONFIG_INT_MODERATION', (MLX5_CMD_OP_ACCESS_REG:=2053): 'MLX5_CMD_OP_ACCESS_REG', (MLX5_CMD_OP_ATTACH_TO_MCG:=2054): 'MLX5_CMD_OP_ATTACH_TO_MCG', (MLX5_CMD_OP_DETACH_FROM_MCG:=2055): 'MLX5_CMD_OP_DETACH_FROM_MCG', (MLX5_CMD_OP_GET_DROPPED_PACKET_LOG:=2058): 'MLX5_CMD_OP_GET_DROPPED_PACKET_LOG', (MLX5_CMD_OP_MAD_IFC:=1293): 'MLX5_CMD_OP_MAD_IFC', (MLX5_CMD_OP_QUERY_MAD_DEMUX:=2059): 'MLX5_CMD_OP_QUERY_MAD_DEMUX', (MLX5_CMD_OP_SET_MAD_DEMUX:=2060): 'MLX5_CMD_OP_SET_MAD_DEMUX', (MLX5_CMD_OP_NOP:=2061): 'MLX5_CMD_OP_NOP', (MLX5_CMD_OP_ALLOC_XRCD:=2062): 'MLX5_CMD_OP_ALLOC_XRCD', (MLX5_CMD_OP_DEALLOC_XRCD:=2063): 'MLX5_CMD_OP_DEALLOC_XRCD', (MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN:=2070): 'MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN', (MLX5_CMD_OP_DEALLOC_TRANSPORT_DOMAIN:=2071): 'MLX5_CMD_OP_DEALLOC_TRANSPORT_DOMAIN', (MLX5_CMD_OP_QUERY_CONG_STATUS:=2082): 'MLX5_CMD_OP_QUERY_CONG_STATUS', (MLX5_CMD_OP_MODIFY_CONG_STATUS:=2083): 'MLX5_CMD_OP_MODIFY_CONG_STATUS', (MLX5_CMD_OP_QUERY_CONG_PARAMS:=2084): 'MLX5_CMD_OP_QUERY_CONG_PARAMS', (MLX5_CMD_OP_MODIFY_CONG_PARAMS:=2085): 'MLX5_CMD_OP_MODIFY_CONG_PARAMS', (MLX5_CMD_OP_QUERY_CONG_STATISTICS:=2086): 'MLX5_CMD_OP_QUERY_CONG_STATISTICS', (MLX5_CMD_OP_ADD_VXLAN_UDP_DPORT:=2087): 'MLX5_CMD_OP_ADD_VXLAN_UDP_DPORT', (MLX5_CMD_OP_DELETE_VXLAN_UDP_DPORT:=2088): 'MLX5_CMD_OP_DELETE_VXLAN_UDP_DPORT', (MLX5_CMD_OP_SET_L2_TABLE_ENTRY:=2089): 'MLX5_CMD_OP_SET_L2_TABLE_ENTRY', (MLX5_CMD_OP_QUERY_L2_TABLE_ENTRY:=2090): 'MLX5_CMD_OP_QUERY_L2_TABLE_ENTRY', (MLX5_CMD_OP_DELETE_L2_TABLE_ENTRY:=2091): 'MLX5_CMD_OP_DELETE_L2_TABLE_ENTRY', (MLX5_CMD_OP_SET_WOL_ROL:=2096): 'MLX5_CMD_OP_SET_WOL_ROL', (MLX5_CMD_OP_QUERY_WOL_ROL:=2097): 'MLX5_CMD_OP_QUERY_WOL_ROL', (MLX5_CMD_OP_CREATE_LAG:=2112): 'MLX5_CMD_OP_CREATE_LAG', (MLX5_CMD_OP_MODIFY_LAG:=2113): 'MLX5_CMD_OP_MODIFY_LAG', (MLX5_CMD_OP_QUERY_LAG:=2114): 'MLX5_CMD_OP_QUERY_LAG', (MLX5_CMD_OP_DESTROY_LAG:=2115): 'MLX5_CMD_OP_DESTROY_LAG', (MLX5_CMD_OP_CREATE_VPORT_LAG:=2116): 'MLX5_CMD_OP_CREATE_VPORT_LAG', (MLX5_CMD_OP_DESTROY_VPORT_LAG:=2117): 'MLX5_CMD_OP_DESTROY_VPORT_LAG', (MLX5_CMD_OP_CREATE_TIR:=2304): 'MLX5_CMD_OP_CREATE_TIR', (MLX5_CMD_OP_MODIFY_TIR:=2305): 'MLX5_CMD_OP_MODIFY_TIR', (MLX5_CMD_OP_DESTROY_TIR:=2306): 'MLX5_CMD_OP_DESTROY_TIR', (MLX5_CMD_OP_QUERY_TIR:=2307): 'MLX5_CMD_OP_QUERY_TIR', (MLX5_CMD_OP_CREATE_SQ:=2308): 'MLX5_CMD_OP_CREATE_SQ', (MLX5_CMD_OP_MODIFY_SQ:=2309): 'MLX5_CMD_OP_MODIFY_SQ', (MLX5_CMD_OP_DESTROY_SQ:=2310): 'MLX5_CMD_OP_DESTROY_SQ', (MLX5_CMD_OP_QUERY_SQ:=2311): 'MLX5_CMD_OP_QUERY_SQ', (MLX5_CMD_OP_CREATE_RQ:=2312): 'MLX5_CMD_OP_CREATE_RQ', (MLX5_CMD_OP_MODIFY_RQ:=2313): 'MLX5_CMD_OP_MODIFY_RQ', (MLX5_CMD_OP_SET_DELAY_DROP_PARAMS:=2320): 'MLX5_CMD_OP_SET_DELAY_DROP_PARAMS', (MLX5_CMD_OP_DESTROY_RQ:=2314): 'MLX5_CMD_OP_DESTROY_RQ', (MLX5_CMD_OP_QUERY_RQ:=2315): 'MLX5_CMD_OP_QUERY_RQ', (MLX5_CMD_OP_CREATE_RMP:=2316): 'MLX5_CMD_OP_CREATE_RMP', (MLX5_CMD_OP_MODIFY_RMP:=2317): 'MLX5_CMD_OP_MODIFY_RMP', (MLX5_CMD_OP_DESTROY_RMP:=2318): 'MLX5_CMD_OP_DESTROY_RMP', (MLX5_CMD_OP_QUERY_RMP:=2319): 'MLX5_CMD_OP_QUERY_RMP', (MLX5_CMD_OP_CREATE_TIS:=2322): 'MLX5_CMD_OP_CREATE_TIS', (MLX5_CMD_OP_MODIFY_TIS:=2323): 'MLX5_CMD_OP_MODIFY_TIS', (MLX5_CMD_OP_DESTROY_TIS:=2324): 'MLX5_CMD_OP_DESTROY_TIS', (MLX5_CMD_OP_QUERY_TIS:=2325): 'MLX5_CMD_OP_QUERY_TIS', (MLX5_CMD_OP_CREATE_RQT:=2326): 'MLX5_CMD_OP_CREATE_RQT', (MLX5_CMD_OP_MODIFY_RQT:=2327): 'MLX5_CMD_OP_MODIFY_RQT', (MLX5_CMD_OP_DESTROY_RQT:=2328): 'MLX5_CMD_OP_DESTROY_RQT', (MLX5_CMD_OP_QUERY_RQT:=2329): 'MLX5_CMD_OP_QUERY_RQT', (MLX5_CMD_OP_SET_FLOW_TABLE_ROOT:=2351): 'MLX5_CMD_OP_SET_FLOW_TABLE_ROOT', (MLX5_CMD_OP_CREATE_FLOW_TABLE:=2352): 'MLX5_CMD_OP_CREATE_FLOW_TABLE', (MLX5_CMD_OP_DESTROY_FLOW_TABLE:=2353): 'MLX5_CMD_OP_DESTROY_FLOW_TABLE', (MLX5_CMD_OP_QUERY_FLOW_TABLE:=2354): 'MLX5_CMD_OP_QUERY_FLOW_TABLE', (MLX5_CMD_OP_CREATE_FLOW_GROUP:=2355): 'MLX5_CMD_OP_CREATE_FLOW_GROUP', (MLX5_CMD_OP_DESTROY_FLOW_GROUP:=2356): 'MLX5_CMD_OP_DESTROY_FLOW_GROUP', (MLX5_CMD_OP_QUERY_FLOW_GROUP:=2357): 'MLX5_CMD_OP_QUERY_FLOW_GROUP', (MLX5_CMD_OP_SET_FLOW_TABLE_ENTRY:=2358): 'MLX5_CMD_OP_SET_FLOW_TABLE_ENTRY', (MLX5_CMD_OP_QUERY_FLOW_TABLE_ENTRY:=2359): 'MLX5_CMD_OP_QUERY_FLOW_TABLE_ENTRY', (MLX5_CMD_OP_DELETE_FLOW_TABLE_ENTRY:=2360): 'MLX5_CMD_OP_DELETE_FLOW_TABLE_ENTRY', (MLX5_CMD_OP_ALLOC_FLOW_COUNTER:=2361): 'MLX5_CMD_OP_ALLOC_FLOW_COUNTER', (MLX5_CMD_OP_DEALLOC_FLOW_COUNTER:=2362): 'MLX5_CMD_OP_DEALLOC_FLOW_COUNTER', (MLX5_CMD_OP_QUERY_FLOW_COUNTER:=2363): 'MLX5_CMD_OP_QUERY_FLOW_COUNTER', (MLX5_CMD_OP_MODIFY_FLOW_TABLE:=2364): 'MLX5_CMD_OP_MODIFY_FLOW_TABLE', (MLX5_CMD_OP_ALLOC_PACKET_REFORMAT_CONTEXT:=2365): 'MLX5_CMD_OP_ALLOC_PACKET_REFORMAT_CONTEXT', (MLX5_CMD_OP_DEALLOC_PACKET_REFORMAT_CONTEXT:=2366): 'MLX5_CMD_OP_DEALLOC_PACKET_REFORMAT_CONTEXT', (MLX5_CMD_OP_QUERY_PACKET_REFORMAT_CONTEXT:=2367): 'MLX5_CMD_OP_QUERY_PACKET_REFORMAT_CONTEXT', (MLX5_CMD_OP_ALLOC_MODIFY_HEADER_CONTEXT:=2368): 'MLX5_CMD_OP_ALLOC_MODIFY_HEADER_CONTEXT', (MLX5_CMD_OP_DEALLOC_MODIFY_HEADER_CONTEXT:=2369): 'MLX5_CMD_OP_DEALLOC_MODIFY_HEADER_CONTEXT', (MLX5_CMD_OP_QUERY_MODIFY_HEADER_CONTEXT:=2370): 'MLX5_CMD_OP_QUERY_MODIFY_HEADER_CONTEXT', (MLX5_CMD_OP_FPGA_CREATE_QP:=2400): 'MLX5_CMD_OP_FPGA_CREATE_QP', (MLX5_CMD_OP_FPGA_MODIFY_QP:=2401): 'MLX5_CMD_OP_FPGA_MODIFY_QP', (MLX5_CMD_OP_FPGA_QUERY_QP:=2402): 'MLX5_CMD_OP_FPGA_QUERY_QP', (MLX5_CMD_OP_FPGA_DESTROY_QP:=2403): 'MLX5_CMD_OP_FPGA_DESTROY_QP', (MLX5_CMD_OP_FPGA_QUERY_QP_COUNTERS:=2404): 'MLX5_CMD_OP_FPGA_QUERY_QP_COUNTERS', (MLX5_CMD_OP_CREATE_GENERAL_OBJECT:=2560): 'MLX5_CMD_OP_CREATE_GENERAL_OBJECT', (MLX5_CMD_OP_MODIFY_GENERAL_OBJECT:=2561): 'MLX5_CMD_OP_MODIFY_GENERAL_OBJECT', (MLX5_CMD_OP_QUERY_GENERAL_OBJECT:=2562): 'MLX5_CMD_OP_QUERY_GENERAL_OBJECT', (MLX5_CMD_OP_DESTROY_GENERAL_OBJECT:=2563): 'MLX5_CMD_OP_DESTROY_GENERAL_OBJECT', (MLX5_CMD_OP_CREATE_UCTX:=2564): 'MLX5_CMD_OP_CREATE_UCTX', (MLX5_CMD_OP_DESTROY_UCTX:=2566): 'MLX5_CMD_OP_DESTROY_UCTX', (MLX5_CMD_OP_CREATE_UMEM:=2568): 'MLX5_CMD_OP_CREATE_UMEM', (MLX5_CMD_OP_DESTROY_UMEM:=2570): 'MLX5_CMD_OP_DESTROY_UMEM', (MLX5_CMD_OP_SYNC_STEERING:=2816): 'MLX5_CMD_OP_SYNC_STEERING', (MLX5_CMD_OP_PSP_GEN_SPI:=2832): 'MLX5_CMD_OP_PSP_GEN_SPI', (MLX5_CMD_OP_PSP_ROTATE_KEY:=2833): 'MLX5_CMD_OP_PSP_ROTATE_KEY', (MLX5_CMD_OP_QUERY_VHCA_STATE:=2829): 'MLX5_CMD_OP_QUERY_VHCA_STATE', (MLX5_CMD_OP_MODIFY_VHCA_STATE:=2830): 'MLX5_CMD_OP_MODIFY_VHCA_STATE', (MLX5_CMD_OP_SYNC_CRYPTO:=2834): 'MLX5_CMD_OP_SYNC_CRYPTO', (MLX5_CMD_OP_ALLOW_OTHER_VHCA_ACCESS:=2838): 'MLX5_CMD_OP_ALLOW_OTHER_VHCA_ACCESS', (MLX5_CMD_OP_GENERATE_WQE:=2839): 'MLX5_CMD_OP_GENERATE_WQE', (MLX5_CMD_OPCODE_QUERY_VUID:=2850): 'MLX5_CMD_OPCODE_QUERY_VUID', (MLX5_CMD_OP_MAX:=2851): 'MLX5_CMD_OP_MAX'}
_anonenum6: dict[int, str] = {(MLX5_CMD_OP_GENERAL_START:=2816): 'MLX5_CMD_OP_GENERAL_START', (MLX5_CMD_OP_GENERAL_END:=3328): 'MLX5_CMD_OP_GENERAL_END'}
_anonenum7: dict[int, str] = {(MLX5_FT_NIC_RX_2_NIC_RX_RDMA:=0): 'MLX5_FT_NIC_RX_2_NIC_RX_RDMA', (MLX5_FT_NIC_TX_RDMA_2_NIC_TX:=1): 'MLX5_FT_NIC_TX_RDMA_2_NIC_TX'}
_anonenum8: dict[int, str] = {(MLX5_CMD_OP_MOD_UPDATE_HEADER_MODIFY_ARGUMENT:=1): 'MLX5_CMD_OP_MOD_UPDATE_HEADER_MODIFY_ARGUMENT'}
@c.record
class struct_mlx5_ifc_flow_table_fields_supported_bits(c.Struct):
  SIZE = 128
  outer_dmac: 'list[int]'
  outer_smac: 'list[int]'
  outer_ether_type: 'list[int]'
  outer_ip_version: 'list[int]'
  outer_first_prio: 'list[int]'
  outer_first_cfi: 'list[int]'
  outer_first_vid: 'list[int]'
  outer_ipv4_ttl: 'list[int]'
  outer_second_prio: 'list[int]'
  outer_second_cfi: 'list[int]'
  outer_second_vid: 'list[int]'
  reserved_at_b: 'list[int]'
  outer_sip: 'list[int]'
  outer_dip: 'list[int]'
  outer_frag: 'list[int]'
  outer_ip_protocol: 'list[int]'
  outer_ip_ecn: 'list[int]'
  outer_ip_dscp: 'list[int]'
  outer_udp_sport: 'list[int]'
  outer_udp_dport: 'list[int]'
  outer_tcp_sport: 'list[int]'
  outer_tcp_dport: 'list[int]'
  outer_tcp_flags: 'list[int]'
  outer_gre_protocol: 'list[int]'
  outer_gre_key: 'list[int]'
  outer_vxlan_vni: 'list[int]'
  outer_geneve_vni: 'list[int]'
  outer_geneve_oam: 'list[int]'
  outer_geneve_protocol_type: 'list[int]'
  outer_geneve_opt_len: 'list[int]'
  source_vhca_port: 'list[int]'
  source_eswitch_port: 'list[int]'
  inner_dmac: 'list[int]'
  inner_smac: 'list[int]'
  inner_ether_type: 'list[int]'
  inner_ip_version: 'list[int]'
  inner_first_prio: 'list[int]'
  inner_first_cfi: 'list[int]'
  inner_first_vid: 'list[int]'
  reserved_at_27: 'list[int]'
  inner_second_prio: 'list[int]'
  inner_second_cfi: 'list[int]'
  inner_second_vid: 'list[int]'
  reserved_at_2b: 'list[int]'
  inner_sip: 'list[int]'
  inner_dip: 'list[int]'
  inner_frag: 'list[int]'
  inner_ip_protocol: 'list[int]'
  inner_ip_ecn: 'list[int]'
  inner_ip_dscp: 'list[int]'
  inner_udp_sport: 'list[int]'
  inner_udp_dport: 'list[int]'
  inner_tcp_sport: 'list[int]'
  inner_tcp_dport: 'list[int]'
  inner_tcp_flags: 'list[int]'
  reserved_at_37: 'list[int]'
  geneve_tlv_option_0_data: 'list[int]'
  geneve_tlv_option_0_exist: 'list[int]'
  reserved_at_42: 'list[int]'
  outer_first_mpls_over_udp: 'list[int]'
  outer_first_mpls_over_gre: 'list[int]'
  inner_first_mpls: 'list[int]'
  outer_first_mpls: 'list[int]'
  reserved_at_55: 'list[int]'
  outer_esp_spi: 'list[int]'
  reserved_at_58: 'list[int]'
  bth_dst_qp: 'list[int]'
  reserved_at_5b: 'list[int]'
  reserved_at_60: 'list[int]'
  metadata_reg_c_7: 'list[int]'
  metadata_reg_c_6: 'list[int]'
  metadata_reg_c_5: 'list[int]'
  metadata_reg_c_4: 'list[int]'
  metadata_reg_c_3: 'list[int]'
  metadata_reg_c_2: 'list[int]'
  metadata_reg_c_1: 'list[int]'
  metadata_reg_c_0: 'list[int]'
struct_mlx5_ifc_flow_table_fields_supported_bits.register_fields([('outer_dmac', (ctypes.c_ubyte * 1), 0), ('outer_smac', (ctypes.c_ubyte * 1), 1), ('outer_ether_type', (ctypes.c_ubyte * 1), 2), ('outer_ip_version', (ctypes.c_ubyte * 1), 3), ('outer_first_prio', (ctypes.c_ubyte * 1), 4), ('outer_first_cfi', (ctypes.c_ubyte * 1), 5), ('outer_first_vid', (ctypes.c_ubyte * 1), 6), ('outer_ipv4_ttl', (ctypes.c_ubyte * 1), 7), ('outer_second_prio', (ctypes.c_ubyte * 1), 8), ('outer_second_cfi', (ctypes.c_ubyte * 1), 9), ('outer_second_vid', (ctypes.c_ubyte * 1), 10), ('reserved_at_b', (ctypes.c_ubyte * 1), 11), ('outer_sip', (ctypes.c_ubyte * 1), 12), ('outer_dip', (ctypes.c_ubyte * 1), 13), ('outer_frag', (ctypes.c_ubyte * 1), 14), ('outer_ip_protocol', (ctypes.c_ubyte * 1), 15), ('outer_ip_ecn', (ctypes.c_ubyte * 1), 16), ('outer_ip_dscp', (ctypes.c_ubyte * 1), 17), ('outer_udp_sport', (ctypes.c_ubyte * 1), 18), ('outer_udp_dport', (ctypes.c_ubyte * 1), 19), ('outer_tcp_sport', (ctypes.c_ubyte * 1), 20), ('outer_tcp_dport', (ctypes.c_ubyte * 1), 21), ('outer_tcp_flags', (ctypes.c_ubyte * 1), 22), ('outer_gre_protocol', (ctypes.c_ubyte * 1), 23), ('outer_gre_key', (ctypes.c_ubyte * 1), 24), ('outer_vxlan_vni', (ctypes.c_ubyte * 1), 25), ('outer_geneve_vni', (ctypes.c_ubyte * 1), 26), ('outer_geneve_oam', (ctypes.c_ubyte * 1), 27), ('outer_geneve_protocol_type', (ctypes.c_ubyte * 1), 28), ('outer_geneve_opt_len', (ctypes.c_ubyte * 1), 29), ('source_vhca_port', (ctypes.c_ubyte * 1), 30), ('source_eswitch_port', (ctypes.c_ubyte * 1), 31), ('inner_dmac', (ctypes.c_ubyte * 1), 32), ('inner_smac', (ctypes.c_ubyte * 1), 33), ('inner_ether_type', (ctypes.c_ubyte * 1), 34), ('inner_ip_version', (ctypes.c_ubyte * 1), 35), ('inner_first_prio', (ctypes.c_ubyte * 1), 36), ('inner_first_cfi', (ctypes.c_ubyte * 1), 37), ('inner_first_vid', (ctypes.c_ubyte * 1), 38), ('reserved_at_27', (ctypes.c_ubyte * 1), 39), ('inner_second_prio', (ctypes.c_ubyte * 1), 40), ('inner_second_cfi', (ctypes.c_ubyte * 1), 41), ('inner_second_vid', (ctypes.c_ubyte * 1), 42), ('reserved_at_2b', (ctypes.c_ubyte * 1), 43), ('inner_sip', (ctypes.c_ubyte * 1), 44), ('inner_dip', (ctypes.c_ubyte * 1), 45), ('inner_frag', (ctypes.c_ubyte * 1), 46), ('inner_ip_protocol', (ctypes.c_ubyte * 1), 47), ('inner_ip_ecn', (ctypes.c_ubyte * 1), 48), ('inner_ip_dscp', (ctypes.c_ubyte * 1), 49), ('inner_udp_sport', (ctypes.c_ubyte * 1), 50), ('inner_udp_dport', (ctypes.c_ubyte * 1), 51), ('inner_tcp_sport', (ctypes.c_ubyte * 1), 52), ('inner_tcp_dport', (ctypes.c_ubyte * 1), 53), ('inner_tcp_flags', (ctypes.c_ubyte * 1), 54), ('reserved_at_37', (ctypes.c_ubyte * 9), 55), ('geneve_tlv_option_0_data', (ctypes.c_ubyte * 1), 64), ('geneve_tlv_option_0_exist', (ctypes.c_ubyte * 1), 65), ('reserved_at_42', (ctypes.c_ubyte * 3), 66), ('outer_first_mpls_over_udp', (ctypes.c_ubyte * 4), 69), ('outer_first_mpls_over_gre', (ctypes.c_ubyte * 4), 73), ('inner_first_mpls', (ctypes.c_ubyte * 4), 77), ('outer_first_mpls', (ctypes.c_ubyte * 4), 81), ('reserved_at_55', (ctypes.c_ubyte * 2), 85), ('outer_esp_spi', (ctypes.c_ubyte * 1), 87), ('reserved_at_58', (ctypes.c_ubyte * 2), 88), ('bth_dst_qp', (ctypes.c_ubyte * 1), 90), ('reserved_at_5b', (ctypes.c_ubyte * 5), 91), ('reserved_at_60', (ctypes.c_ubyte * 24), 96), ('metadata_reg_c_7', (ctypes.c_ubyte * 1), 120), ('metadata_reg_c_6', (ctypes.c_ubyte * 1), 121), ('metadata_reg_c_5', (ctypes.c_ubyte * 1), 122), ('metadata_reg_c_4', (ctypes.c_ubyte * 1), 123), ('metadata_reg_c_3', (ctypes.c_ubyte * 1), 124), ('metadata_reg_c_2', (ctypes.c_ubyte * 1), 125), ('metadata_reg_c_1', (ctypes.c_ubyte * 1), 126), ('metadata_reg_c_0', (ctypes.c_ubyte * 1), 127)])
@c.record
class struct_mlx5_ifc_flow_table_fields_supported_2_bits(c.Struct):
  SIZE = 128
  inner_l4_type_ext: 'list[int]'
  outer_l4_type_ext: 'list[int]'
  inner_l4_type: 'list[int]'
  outer_l4_type: 'list[int]'
  reserved_at_4: 'list[int]'
  bth_opcode: 'list[int]'
  reserved_at_f: 'list[int]'
  tunnel_header_0_1: 'list[int]'
  reserved_at_11: 'list[int]'
  reserved_at_20: 'list[int]'
  ipsec_next_header: 'list[int]'
  reserved_at_30: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_flow_table_fields_supported_2_bits.register_fields([('inner_l4_type_ext', (ctypes.c_ubyte * 1), 0), ('outer_l4_type_ext', (ctypes.c_ubyte * 1), 1), ('inner_l4_type', (ctypes.c_ubyte * 1), 2), ('outer_l4_type', (ctypes.c_ubyte * 1), 3), ('reserved_at_4', (ctypes.c_ubyte * 10), 4), ('bth_opcode', (ctypes.c_ubyte * 1), 14), ('reserved_at_f', (ctypes.c_ubyte * 1), 15), ('tunnel_header_0_1', (ctypes.c_ubyte * 1), 16), ('reserved_at_11', (ctypes.c_ubyte * 15), 17), ('reserved_at_20', (ctypes.c_ubyte * 15), 32), ('ipsec_next_header', (ctypes.c_ubyte * 1), 47), ('reserved_at_30', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_flow_table_prop_layout_bits(c.Struct):
  SIZE = 512
  ft_support: 'list[int]'
  reserved_at_1: 'list[int]'
  flow_counter: 'list[int]'
  flow_modify_en: 'list[int]'
  modify_root: 'list[int]'
  identified_miss_table_mode: 'list[int]'
  flow_table_modify: 'list[int]'
  reformat: 'list[int]'
  decap: 'list[int]'
  reset_root_to_default: 'list[int]'
  pop_vlan: 'list[int]'
  push_vlan: 'list[int]'
  reserved_at_c: 'list[int]'
  pop_vlan_2: 'list[int]'
  push_vlan_2: 'list[int]'
  reformat_and_vlan_action: 'list[int]'
  reserved_at_10: 'list[int]'
  sw_owner: 'list[int]'
  reformat_l3_tunnel_to_l2: 'list[int]'
  reformat_l2_to_l3_tunnel: 'list[int]'
  reformat_and_modify_action: 'list[int]'
  ignore_flow_level: 'list[int]'
  reserved_at_16: 'list[int]'
  table_miss_action_domain: 'list[int]'
  termination_table: 'list[int]'
  reformat_and_fwd_to_table: 'list[int]'
  reserved_at_1a: 'list[int]'
  ipsec_encrypt: 'list[int]'
  ipsec_decrypt: 'list[int]'
  sw_owner_v2: 'list[int]'
  reserved_at_1f: 'list[int]'
  termination_table_raw_traffic: 'list[int]'
  reserved_at_21: 'list[int]'
  log_max_ft_size: 'list[int]'
  log_max_modify_header_context: 'list[int]'
  max_modify_header_actions: 'list[int]'
  max_ft_level: 'list[int]'
  reformat_add_esp_trasport: 'list[int]'
  reformat_l2_to_l3_esp_tunnel: 'list[int]'
  reformat_add_esp_transport_over_udp: 'list[int]'
  reformat_del_esp_trasport: 'list[int]'
  reformat_l3_esp_tunnel_to_l2: 'list[int]'
  reformat_del_esp_transport_over_udp: 'list[int]'
  execute_aso: 'list[int]'
  reserved_at_47: 'list[int]'
  reformat_l2_to_l3_psp_tunnel: 'list[int]'
  reformat_l3_psp_tunnel_to_l2: 'list[int]'
  reformat_insert: 'list[int]'
  reformat_remove: 'list[int]'
  macsec_encrypt: 'list[int]'
  macsec_decrypt: 'list[int]'
  psp_encrypt: 'list[int]'
  psp_decrypt: 'list[int]'
  reformat_add_macsec: 'list[int]'
  reformat_remove_macsec: 'list[int]'
  reparse: 'list[int]'
  reserved_at_6b: 'list[int]'
  cross_vhca_object: 'list[int]'
  reformat_l2_to_l3_audp_tunnel: 'list[int]'
  reformat_l3_audp_tunnel_to_l2: 'list[int]'
  ignore_flow_level_rtc_valid: 'list[int]'
  reserved_at_70: 'list[int]'
  log_max_ft_num: 'list[int]'
  reserved_at_80: 'list[int]'
  log_max_flow_counter: 'list[int]'
  log_max_destination: 'list[int]'
  reserved_at_a0: 'list[int]'
  log_max_flow: 'list[int]'
  reserved_at_c0: 'list[int]'
  ft_field_support: 'struct_mlx5_ifc_flow_table_fields_supported_bits'
  ft_field_bitmask_support: 'struct_mlx5_ifc_flow_table_fields_supported_bits'
struct_mlx5_ifc_flow_table_prop_layout_bits.register_fields([('ft_support', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 1), 1), ('flow_counter', (ctypes.c_ubyte * 1), 2), ('flow_modify_en', (ctypes.c_ubyte * 1), 3), ('modify_root', (ctypes.c_ubyte * 1), 4), ('identified_miss_table_mode', (ctypes.c_ubyte * 1), 5), ('flow_table_modify', (ctypes.c_ubyte * 1), 6), ('reformat', (ctypes.c_ubyte * 1), 7), ('decap', (ctypes.c_ubyte * 1), 8), ('reset_root_to_default', (ctypes.c_ubyte * 1), 9), ('pop_vlan', (ctypes.c_ubyte * 1), 10), ('push_vlan', (ctypes.c_ubyte * 1), 11), ('reserved_at_c', (ctypes.c_ubyte * 1), 12), ('pop_vlan_2', (ctypes.c_ubyte * 1), 13), ('push_vlan_2', (ctypes.c_ubyte * 1), 14), ('reformat_and_vlan_action', (ctypes.c_ubyte * 1), 15), ('reserved_at_10', (ctypes.c_ubyte * 1), 16), ('sw_owner', (ctypes.c_ubyte * 1), 17), ('reformat_l3_tunnel_to_l2', (ctypes.c_ubyte * 1), 18), ('reformat_l2_to_l3_tunnel', (ctypes.c_ubyte * 1), 19), ('reformat_and_modify_action', (ctypes.c_ubyte * 1), 20), ('ignore_flow_level', (ctypes.c_ubyte * 1), 21), ('reserved_at_16', (ctypes.c_ubyte * 1), 22), ('table_miss_action_domain', (ctypes.c_ubyte * 1), 23), ('termination_table', (ctypes.c_ubyte * 1), 24), ('reformat_and_fwd_to_table', (ctypes.c_ubyte * 1), 25), ('reserved_at_1a', (ctypes.c_ubyte * 2), 26), ('ipsec_encrypt', (ctypes.c_ubyte * 1), 28), ('ipsec_decrypt', (ctypes.c_ubyte * 1), 29), ('sw_owner_v2', (ctypes.c_ubyte * 1), 30), ('reserved_at_1f', (ctypes.c_ubyte * 1), 31), ('termination_table_raw_traffic', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 1), 33), ('log_max_ft_size', (ctypes.c_ubyte * 6), 34), ('log_max_modify_header_context', (ctypes.c_ubyte * 8), 40), ('max_modify_header_actions', (ctypes.c_ubyte * 8), 48), ('max_ft_level', (ctypes.c_ubyte * 8), 56), ('reformat_add_esp_trasport', (ctypes.c_ubyte * 1), 64), ('reformat_l2_to_l3_esp_tunnel', (ctypes.c_ubyte * 1), 65), ('reformat_add_esp_transport_over_udp', (ctypes.c_ubyte * 1), 66), ('reformat_del_esp_trasport', (ctypes.c_ubyte * 1), 67), ('reformat_l3_esp_tunnel_to_l2', (ctypes.c_ubyte * 1), 68), ('reformat_del_esp_transport_over_udp', (ctypes.c_ubyte * 1), 69), ('execute_aso', (ctypes.c_ubyte * 1), 70), ('reserved_at_47', (ctypes.c_ubyte * 25), 71), ('reformat_l2_to_l3_psp_tunnel', (ctypes.c_ubyte * 1), 96), ('reformat_l3_psp_tunnel_to_l2', (ctypes.c_ubyte * 1), 97), ('reformat_insert', (ctypes.c_ubyte * 1), 98), ('reformat_remove', (ctypes.c_ubyte * 1), 99), ('macsec_encrypt', (ctypes.c_ubyte * 1), 100), ('macsec_decrypt', (ctypes.c_ubyte * 1), 101), ('psp_encrypt', (ctypes.c_ubyte * 1), 102), ('psp_decrypt', (ctypes.c_ubyte * 1), 103), ('reformat_add_macsec', (ctypes.c_ubyte * 1), 104), ('reformat_remove_macsec', (ctypes.c_ubyte * 1), 105), ('reparse', (ctypes.c_ubyte * 1), 106), ('reserved_at_6b', (ctypes.c_ubyte * 1), 107), ('cross_vhca_object', (ctypes.c_ubyte * 1), 108), ('reformat_l2_to_l3_audp_tunnel', (ctypes.c_ubyte * 1), 109), ('reformat_l3_audp_tunnel_to_l2', (ctypes.c_ubyte * 1), 110), ('ignore_flow_level_rtc_valid', (ctypes.c_ubyte * 1), 111), ('reserved_at_70', (ctypes.c_ubyte * 8), 112), ('log_max_ft_num', (ctypes.c_ubyte * 8), 120), ('reserved_at_80', (ctypes.c_ubyte * 16), 128), ('log_max_flow_counter', (ctypes.c_ubyte * 8), 144), ('log_max_destination', (ctypes.c_ubyte * 8), 152), ('reserved_at_a0', (ctypes.c_ubyte * 24), 160), ('log_max_flow', (ctypes.c_ubyte * 8), 184), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('ft_field_support', struct_mlx5_ifc_flow_table_fields_supported_bits, 256), ('ft_field_bitmask_support', struct_mlx5_ifc_flow_table_fields_supported_bits, 384)])
@c.record
class struct_mlx5_ifc_odp_per_transport_service_cap_bits(c.Struct):
  SIZE = 32
  send: 'list[int]'
  receive: 'list[int]'
  write: 'list[int]'
  read: 'list[int]'
  atomic: 'list[int]'
  srq_receive: 'list[int]'
  reserved_at_6: 'list[int]'
struct_mlx5_ifc_odp_per_transport_service_cap_bits.register_fields([('send', (ctypes.c_ubyte * 1), 0), ('receive', (ctypes.c_ubyte * 1), 1), ('write', (ctypes.c_ubyte * 1), 2), ('read', (ctypes.c_ubyte * 1), 3), ('atomic', (ctypes.c_ubyte * 1), 4), ('srq_receive', (ctypes.c_ubyte * 1), 5), ('reserved_at_6', (ctypes.c_ubyte * 26), 6)])
@c.record
class struct_mlx5_ifc_ipv4_layout_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  ipv4: 'list[int]'
struct_mlx5_ifc_ipv4_layout_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 96), 0), ('ipv4', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_ipv6_layout_bits(c.Struct):
  SIZE = 128
  ipv6: 'list[list[int]]'
struct_mlx5_ifc_ipv6_layout_bits.register_fields([('ipv6', ((ctypes.c_ubyte * 8) * 16), 0)])
@c.record
class struct_mlx5_ifc_ipv6_simple_layout_bits(c.Struct):
  SIZE = 128
  ipv6_127_96: 'list[int]'
  ipv6_95_64: 'list[int]'
  ipv6_63_32: 'list[int]'
  ipv6_31_0: 'list[int]'
struct_mlx5_ifc_ipv6_simple_layout_bits.register_fields([('ipv6_127_96', (ctypes.c_ubyte * 32), 0), ('ipv6_95_64', (ctypes.c_ubyte * 32), 32), ('ipv6_63_32', (ctypes.c_ubyte * 32), 64), ('ipv6_31_0', (ctypes.c_ubyte * 32), 96)])
@c.record
class union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits(c.Struct):
  SIZE = 128
  ipv6_simple_layout: 'struct_mlx5_ifc_ipv6_simple_layout_bits'
  ipv6_layout: 'struct_mlx5_ifc_ipv6_layout_bits'
  ipv4_layout: 'struct_mlx5_ifc_ipv4_layout_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits.register_fields([('ipv6_simple_layout', struct_mlx5_ifc_ipv6_simple_layout_bits, 0), ('ipv6_layout', struct_mlx5_ifc_ipv6_layout_bits, 0), ('ipv4_layout', struct_mlx5_ifc_ipv4_layout_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 128), 0)])
_anonenum9: dict[int, str] = {(MLX5_PACKET_L4_TYPE_NONE:=0): 'MLX5_PACKET_L4_TYPE_NONE', (MLX5_PACKET_L4_TYPE_TCP:=1): 'MLX5_PACKET_L4_TYPE_TCP', (MLX5_PACKET_L4_TYPE_UDP:=2): 'MLX5_PACKET_L4_TYPE_UDP'}
_anonenum10: dict[int, str] = {(MLX5_PACKET_L4_TYPE_EXT_NONE:=0): 'MLX5_PACKET_L4_TYPE_EXT_NONE', (MLX5_PACKET_L4_TYPE_EXT_TCP:=1): 'MLX5_PACKET_L4_TYPE_EXT_TCP', (MLX5_PACKET_L4_TYPE_EXT_UDP:=2): 'MLX5_PACKET_L4_TYPE_EXT_UDP', (MLX5_PACKET_L4_TYPE_EXT_ICMP:=3): 'MLX5_PACKET_L4_TYPE_EXT_ICMP'}
@c.record
class struct_mlx5_ifc_fte_match_set_lyr_2_4_bits(c.Struct):
  SIZE = 512
  smac_47_16: 'list[int]'
  smac_15_0: 'list[int]'
  ethertype: 'list[int]'
  dmac_47_16: 'list[int]'
  dmac_15_0: 'list[int]'
  first_prio: 'list[int]'
  first_cfi: 'list[int]'
  first_vid: 'list[int]'
  ip_protocol: 'list[int]'
  ip_dscp: 'list[int]'
  ip_ecn: 'list[int]'
  cvlan_tag: 'list[int]'
  svlan_tag: 'list[int]'
  frag: 'list[int]'
  ip_version: 'list[int]'
  tcp_flags: 'list[int]'
  tcp_sport: 'list[int]'
  tcp_dport: 'list[int]'
  l4_type: 'list[int]'
  l4_type_ext: 'list[int]'
  reserved_at_c6: 'list[int]'
  ipv4_ihl: 'list[int]'
  reserved_at_d4: 'list[int]'
  ttl_hoplimit: 'list[int]'
  udp_sport: 'list[int]'
  udp_dport: 'list[int]'
  src_ipv4_src_ipv6: 'union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits'
  dst_ipv4_dst_ipv6: 'union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits'
struct_mlx5_ifc_fte_match_set_lyr_2_4_bits.register_fields([('smac_47_16', (ctypes.c_ubyte * 32), 0), ('smac_15_0', (ctypes.c_ubyte * 16), 32), ('ethertype', (ctypes.c_ubyte * 16), 48), ('dmac_47_16', (ctypes.c_ubyte * 32), 64), ('dmac_15_0', (ctypes.c_ubyte * 16), 96), ('first_prio', (ctypes.c_ubyte * 3), 112), ('first_cfi', (ctypes.c_ubyte * 1), 115), ('first_vid', (ctypes.c_ubyte * 12), 116), ('ip_protocol', (ctypes.c_ubyte * 8), 128), ('ip_dscp', (ctypes.c_ubyte * 6), 136), ('ip_ecn', (ctypes.c_ubyte * 2), 142), ('cvlan_tag', (ctypes.c_ubyte * 1), 144), ('svlan_tag', (ctypes.c_ubyte * 1), 145), ('frag', (ctypes.c_ubyte * 1), 146), ('ip_version', (ctypes.c_ubyte * 4), 147), ('tcp_flags', (ctypes.c_ubyte * 9), 151), ('tcp_sport', (ctypes.c_ubyte * 16), 160), ('tcp_dport', (ctypes.c_ubyte * 16), 176), ('l4_type', (ctypes.c_ubyte * 2), 192), ('l4_type_ext', (ctypes.c_ubyte * 4), 194), ('reserved_at_c6', (ctypes.c_ubyte * 10), 198), ('ipv4_ihl', (ctypes.c_ubyte * 4), 208), ('reserved_at_d4', (ctypes.c_ubyte * 4), 212), ('ttl_hoplimit', (ctypes.c_ubyte * 8), 216), ('udp_sport', (ctypes.c_ubyte * 16), 224), ('udp_dport', (ctypes.c_ubyte * 16), 240), ('src_ipv4_src_ipv6', union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits, 256), ('dst_ipv4_dst_ipv6', union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits, 384)])
@c.record
class struct_mlx5_ifc_nvgre_key_bits(c.Struct):
  SIZE = 32
  hi: 'list[int]'
  lo: 'list[int]'
struct_mlx5_ifc_nvgre_key_bits.register_fields([('hi', (ctypes.c_ubyte * 24), 0), ('lo', (ctypes.c_ubyte * 8), 24)])
@c.record
class union_mlx5_ifc_gre_key_bits(c.Struct):
  SIZE = 32
  nvgre: 'struct_mlx5_ifc_nvgre_key_bits'
  key: 'list[int]'
union_mlx5_ifc_gre_key_bits.register_fields([('nvgre', struct_mlx5_ifc_nvgre_key_bits, 0), ('key', (ctypes.c_ubyte * 32), 0)])
@c.record
class struct_mlx5_ifc_fte_match_set_misc_bits(c.Struct):
  SIZE = 512
  gre_c_present: 'list[int]'
  reserved_at_1: 'list[int]'
  gre_k_present: 'list[int]'
  gre_s_present: 'list[int]'
  source_vhca_port: 'list[int]'
  source_sqn: 'list[int]'
  source_eswitch_owner_vhca_id: 'list[int]'
  source_port: 'list[int]'
  outer_second_prio: 'list[int]'
  outer_second_cfi: 'list[int]'
  outer_second_vid: 'list[int]'
  inner_second_prio: 'list[int]'
  inner_second_cfi: 'list[int]'
  inner_second_vid: 'list[int]'
  outer_second_cvlan_tag: 'list[int]'
  inner_second_cvlan_tag: 'list[int]'
  outer_second_svlan_tag: 'list[int]'
  inner_second_svlan_tag: 'list[int]'
  reserved_at_64: 'list[int]'
  gre_protocol: 'list[int]'
  gre_key: 'union_mlx5_ifc_gre_key_bits'
  vxlan_vni: 'list[int]'
  bth_opcode: 'list[int]'
  geneve_vni: 'list[int]'
  reserved_at_d8: 'list[int]'
  geneve_tlv_option_0_exist: 'list[int]'
  geneve_oam: 'list[int]'
  reserved_at_e0: 'list[int]'
  outer_ipv6_flow_label: 'list[int]'
  reserved_at_100: 'list[int]'
  inner_ipv6_flow_label: 'list[int]'
  reserved_at_120: 'list[int]'
  geneve_opt_len: 'list[int]'
  geneve_protocol_type: 'list[int]'
  reserved_at_140: 'list[int]'
  bth_dst_qp: 'list[int]'
  inner_esp_spi: 'list[int]'
  outer_esp_spi: 'list[int]'
  reserved_at_1a0: 'list[int]'
struct_mlx5_ifc_fte_match_set_misc_bits.register_fields([('gre_c_present', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 1), 1), ('gre_k_present', (ctypes.c_ubyte * 1), 2), ('gre_s_present', (ctypes.c_ubyte * 1), 3), ('source_vhca_port', (ctypes.c_ubyte * 4), 4), ('source_sqn', (ctypes.c_ubyte * 24), 8), ('source_eswitch_owner_vhca_id', (ctypes.c_ubyte * 16), 32), ('source_port', (ctypes.c_ubyte * 16), 48), ('outer_second_prio', (ctypes.c_ubyte * 3), 64), ('outer_second_cfi', (ctypes.c_ubyte * 1), 67), ('outer_second_vid', (ctypes.c_ubyte * 12), 68), ('inner_second_prio', (ctypes.c_ubyte * 3), 80), ('inner_second_cfi', (ctypes.c_ubyte * 1), 83), ('inner_second_vid', (ctypes.c_ubyte * 12), 84), ('outer_second_cvlan_tag', (ctypes.c_ubyte * 1), 96), ('inner_second_cvlan_tag', (ctypes.c_ubyte * 1), 97), ('outer_second_svlan_tag', (ctypes.c_ubyte * 1), 98), ('inner_second_svlan_tag', (ctypes.c_ubyte * 1), 99), ('reserved_at_64', (ctypes.c_ubyte * 12), 100), ('gre_protocol', (ctypes.c_ubyte * 16), 112), ('gre_key', union_mlx5_ifc_gre_key_bits, 128), ('vxlan_vni', (ctypes.c_ubyte * 24), 160), ('bth_opcode', (ctypes.c_ubyte * 8), 184), ('geneve_vni', (ctypes.c_ubyte * 24), 192), ('reserved_at_d8', (ctypes.c_ubyte * 6), 216), ('geneve_tlv_option_0_exist', (ctypes.c_ubyte * 1), 222), ('geneve_oam', (ctypes.c_ubyte * 1), 223), ('reserved_at_e0', (ctypes.c_ubyte * 12), 224), ('outer_ipv6_flow_label', (ctypes.c_ubyte * 20), 236), ('reserved_at_100', (ctypes.c_ubyte * 12), 256), ('inner_ipv6_flow_label', (ctypes.c_ubyte * 20), 268), ('reserved_at_120', (ctypes.c_ubyte * 10), 288), ('geneve_opt_len', (ctypes.c_ubyte * 6), 298), ('geneve_protocol_type', (ctypes.c_ubyte * 16), 304), ('reserved_at_140', (ctypes.c_ubyte * 8), 320), ('bth_dst_qp', (ctypes.c_ubyte * 24), 328), ('inner_esp_spi', (ctypes.c_ubyte * 32), 352), ('outer_esp_spi', (ctypes.c_ubyte * 32), 384), ('reserved_at_1a0', (ctypes.c_ubyte * 96), 416)])
@c.record
class struct_mlx5_ifc_fte_match_mpls_bits(c.Struct):
  SIZE = 32
  mpls_label: 'list[int]'
  mpls_exp: 'list[int]'
  mpls_s_bos: 'list[int]'
  mpls_ttl: 'list[int]'
struct_mlx5_ifc_fte_match_mpls_bits.register_fields([('mpls_label', (ctypes.c_ubyte * 20), 0), ('mpls_exp', (ctypes.c_ubyte * 3), 20), ('mpls_s_bos', (ctypes.c_ubyte * 1), 23), ('mpls_ttl', (ctypes.c_ubyte * 8), 24)])
@c.record
class struct_mlx5_ifc_fte_match_set_misc2_bits(c.Struct):
  SIZE = 512
  outer_first_mpls: 'struct_mlx5_ifc_fte_match_mpls_bits'
  inner_first_mpls: 'struct_mlx5_ifc_fte_match_mpls_bits'
  outer_first_mpls_over_gre: 'struct_mlx5_ifc_fte_match_mpls_bits'
  outer_first_mpls_over_udp: 'struct_mlx5_ifc_fte_match_mpls_bits'
  metadata_reg_c_7: 'list[int]'
  metadata_reg_c_6: 'list[int]'
  metadata_reg_c_5: 'list[int]'
  metadata_reg_c_4: 'list[int]'
  metadata_reg_c_3: 'list[int]'
  metadata_reg_c_2: 'list[int]'
  metadata_reg_c_1: 'list[int]'
  metadata_reg_c_0: 'list[int]'
  metadata_reg_a: 'list[int]'
  psp_syndrome: 'list[int]'
  macsec_syndrome: 'list[int]'
  ipsec_syndrome: 'list[int]'
  ipsec_next_header: 'list[int]'
  reserved_at_1c0: 'list[int]'
struct_mlx5_ifc_fte_match_set_misc2_bits.register_fields([('outer_first_mpls', struct_mlx5_ifc_fte_match_mpls_bits, 0), ('inner_first_mpls', struct_mlx5_ifc_fte_match_mpls_bits, 32), ('outer_first_mpls_over_gre', struct_mlx5_ifc_fte_match_mpls_bits, 64), ('outer_first_mpls_over_udp', struct_mlx5_ifc_fte_match_mpls_bits, 96), ('metadata_reg_c_7', (ctypes.c_ubyte * 32), 128), ('metadata_reg_c_6', (ctypes.c_ubyte * 32), 160), ('metadata_reg_c_5', (ctypes.c_ubyte * 32), 192), ('metadata_reg_c_4', (ctypes.c_ubyte * 32), 224), ('metadata_reg_c_3', (ctypes.c_ubyte * 32), 256), ('metadata_reg_c_2', (ctypes.c_ubyte * 32), 288), ('metadata_reg_c_1', (ctypes.c_ubyte * 32), 320), ('metadata_reg_c_0', (ctypes.c_ubyte * 32), 352), ('metadata_reg_a', (ctypes.c_ubyte * 32), 384), ('psp_syndrome', (ctypes.c_ubyte * 8), 416), ('macsec_syndrome', (ctypes.c_ubyte * 8), 424), ('ipsec_syndrome', (ctypes.c_ubyte * 8), 432), ('ipsec_next_header', (ctypes.c_ubyte * 8), 440), ('reserved_at_1c0', (ctypes.c_ubyte * 64), 448)])
@c.record
class struct_mlx5_ifc_fte_match_set_misc3_bits(c.Struct):
  SIZE = 512
  inner_tcp_seq_num: 'list[int]'
  outer_tcp_seq_num: 'list[int]'
  inner_tcp_ack_num: 'list[int]'
  outer_tcp_ack_num: 'list[int]'
  reserved_at_80: 'list[int]'
  outer_vxlan_gpe_vni: 'list[int]'
  outer_vxlan_gpe_next_protocol: 'list[int]'
  outer_vxlan_gpe_flags: 'list[int]'
  reserved_at_b0: 'list[int]'
  icmp_header_data: 'list[int]'
  icmpv6_header_data: 'list[int]'
  icmp_type: 'list[int]'
  icmp_code: 'list[int]'
  icmpv6_type: 'list[int]'
  icmpv6_code: 'list[int]'
  geneve_tlv_option_0_data: 'list[int]'
  gtpu_teid: 'list[int]'
  gtpu_msg_type: 'list[int]'
  gtpu_msg_flags: 'list[int]'
  reserved_at_170: 'list[int]'
  gtpu_dw_2: 'list[int]'
  gtpu_first_ext_dw_0: 'list[int]'
  gtpu_dw_0: 'list[int]'
  reserved_at_1e0: 'list[int]'
struct_mlx5_ifc_fte_match_set_misc3_bits.register_fields([('inner_tcp_seq_num', (ctypes.c_ubyte * 32), 0), ('outer_tcp_seq_num', (ctypes.c_ubyte * 32), 32), ('inner_tcp_ack_num', (ctypes.c_ubyte * 32), 64), ('outer_tcp_ack_num', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('outer_vxlan_gpe_vni', (ctypes.c_ubyte * 24), 136), ('outer_vxlan_gpe_next_protocol', (ctypes.c_ubyte * 8), 160), ('outer_vxlan_gpe_flags', (ctypes.c_ubyte * 8), 168), ('reserved_at_b0', (ctypes.c_ubyte * 16), 176), ('icmp_header_data', (ctypes.c_ubyte * 32), 192), ('icmpv6_header_data', (ctypes.c_ubyte * 32), 224), ('icmp_type', (ctypes.c_ubyte * 8), 256), ('icmp_code', (ctypes.c_ubyte * 8), 264), ('icmpv6_type', (ctypes.c_ubyte * 8), 272), ('icmpv6_code', (ctypes.c_ubyte * 8), 280), ('geneve_tlv_option_0_data', (ctypes.c_ubyte * 32), 288), ('gtpu_teid', (ctypes.c_ubyte * 32), 320), ('gtpu_msg_type', (ctypes.c_ubyte * 8), 352), ('gtpu_msg_flags', (ctypes.c_ubyte * 8), 360), ('reserved_at_170', (ctypes.c_ubyte * 16), 368), ('gtpu_dw_2', (ctypes.c_ubyte * 32), 384), ('gtpu_first_ext_dw_0', (ctypes.c_ubyte * 32), 416), ('gtpu_dw_0', (ctypes.c_ubyte * 32), 448), ('reserved_at_1e0', (ctypes.c_ubyte * 32), 480)])
@c.record
class struct_mlx5_ifc_fte_match_set_misc4_bits(c.Struct):
  SIZE = 512
  prog_sample_field_value_0: 'list[int]'
  prog_sample_field_id_0: 'list[int]'
  prog_sample_field_value_1: 'list[int]'
  prog_sample_field_id_1: 'list[int]'
  prog_sample_field_value_2: 'list[int]'
  prog_sample_field_id_2: 'list[int]'
  prog_sample_field_value_3: 'list[int]'
  prog_sample_field_id_3: 'list[int]'
  reserved_at_100: 'list[int]'
struct_mlx5_ifc_fte_match_set_misc4_bits.register_fields([('prog_sample_field_value_0', (ctypes.c_ubyte * 32), 0), ('prog_sample_field_id_0', (ctypes.c_ubyte * 32), 32), ('prog_sample_field_value_1', (ctypes.c_ubyte * 32), 64), ('prog_sample_field_id_1', (ctypes.c_ubyte * 32), 96), ('prog_sample_field_value_2', (ctypes.c_ubyte * 32), 128), ('prog_sample_field_id_2', (ctypes.c_ubyte * 32), 160), ('prog_sample_field_value_3', (ctypes.c_ubyte * 32), 192), ('prog_sample_field_id_3', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 256), 256)])
@c.record
class struct_mlx5_ifc_fte_match_set_misc5_bits(c.Struct):
  SIZE = 512
  macsec_tag_0: 'list[int]'
  macsec_tag_1: 'list[int]'
  macsec_tag_2: 'list[int]'
  macsec_tag_3: 'list[int]'
  tunnel_header_0: 'list[int]'
  tunnel_header_1: 'list[int]'
  tunnel_header_2: 'list[int]'
  tunnel_header_3: 'list[int]'
  reserved_at_100: 'list[int]'
struct_mlx5_ifc_fte_match_set_misc5_bits.register_fields([('macsec_tag_0', (ctypes.c_ubyte * 32), 0), ('macsec_tag_1', (ctypes.c_ubyte * 32), 32), ('macsec_tag_2', (ctypes.c_ubyte * 32), 64), ('macsec_tag_3', (ctypes.c_ubyte * 32), 96), ('tunnel_header_0', (ctypes.c_ubyte * 32), 128), ('tunnel_header_1', (ctypes.c_ubyte * 32), 160), ('tunnel_header_2', (ctypes.c_ubyte * 32), 192), ('tunnel_header_3', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 256), 256)])
@c.record
class struct_mlx5_ifc_cmd_pas_bits(c.Struct):
  SIZE = 64
  pa_h: 'list[int]'
  pa_l: 'list[int]'
  reserved_at_34: 'list[int]'
struct_mlx5_ifc_cmd_pas_bits.register_fields([('pa_h', (ctypes.c_ubyte * 32), 0), ('pa_l', (ctypes.c_ubyte * 20), 32), ('reserved_at_34', (ctypes.c_ubyte * 12), 52)])
@c.record
class struct_mlx5_ifc_uint64_bits(c.Struct):
  SIZE = 64
  hi: 'list[int]'
  lo: 'list[int]'
struct_mlx5_ifc_uint64_bits.register_fields([('hi', (ctypes.c_ubyte * 32), 0), ('lo', (ctypes.c_ubyte * 32), 32)])
_anonenum11: dict[int, str] = {(MLX5_ADS_STAT_RATE_NO_LIMIT:=0): 'MLX5_ADS_STAT_RATE_NO_LIMIT', (MLX5_ADS_STAT_RATE_2_5GBPS:=7): 'MLX5_ADS_STAT_RATE_2_5GBPS', (MLX5_ADS_STAT_RATE_10GBPS:=8): 'MLX5_ADS_STAT_RATE_10GBPS', (MLX5_ADS_STAT_RATE_30GBPS:=9): 'MLX5_ADS_STAT_RATE_30GBPS', (MLX5_ADS_STAT_RATE_5GBPS:=10): 'MLX5_ADS_STAT_RATE_5GBPS', (MLX5_ADS_STAT_RATE_20GBPS:=11): 'MLX5_ADS_STAT_RATE_20GBPS', (MLX5_ADS_STAT_RATE_40GBPS:=12): 'MLX5_ADS_STAT_RATE_40GBPS', (MLX5_ADS_STAT_RATE_60GBPS:=13): 'MLX5_ADS_STAT_RATE_60GBPS', (MLX5_ADS_STAT_RATE_80GBPS:=14): 'MLX5_ADS_STAT_RATE_80GBPS', (MLX5_ADS_STAT_RATE_120GBPS:=15): 'MLX5_ADS_STAT_RATE_120GBPS'}
@c.record
class struct_mlx5_ifc_ads_bits(c.Struct):
  SIZE = 352
  fl: 'list[int]'
  free_ar: 'list[int]'
  reserved_at_2: 'list[int]'
  pkey_index: 'list[int]'
  plane_index: 'list[int]'
  grh: 'list[int]'
  mlid: 'list[int]'
  rlid: 'list[int]'
  ack_timeout: 'list[int]'
  reserved_at_45: 'list[int]'
  src_addr_index: 'list[int]'
  reserved_at_50: 'list[int]'
  stat_rate: 'list[int]'
  hop_limit: 'list[int]'
  reserved_at_60: 'list[int]'
  tclass: 'list[int]'
  flow_label: 'list[int]'
  rgid_rip: 'list[list[int]]'
  reserved_at_100: 'list[int]'
  f_dscp: 'list[int]'
  f_ecn: 'list[int]'
  reserved_at_106: 'list[int]'
  f_eth_prio: 'list[int]'
  ecn: 'list[int]'
  dscp: 'list[int]'
  udp_sport: 'list[int]'
  dei_cfi: 'list[int]'
  eth_prio: 'list[int]'
  sl: 'list[int]'
  vhca_port_num: 'list[int]'
  rmac_47_32: 'list[int]'
  rmac_31_0: 'list[int]'
struct_mlx5_ifc_ads_bits.register_fields([('fl', (ctypes.c_ubyte * 1), 0), ('free_ar', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 14), 2), ('pkey_index', (ctypes.c_ubyte * 16), 16), ('plane_index', (ctypes.c_ubyte * 8), 32), ('grh', (ctypes.c_ubyte * 1), 40), ('mlid', (ctypes.c_ubyte * 7), 41), ('rlid', (ctypes.c_ubyte * 16), 48), ('ack_timeout', (ctypes.c_ubyte * 5), 64), ('reserved_at_45', (ctypes.c_ubyte * 3), 69), ('src_addr_index', (ctypes.c_ubyte * 8), 72), ('reserved_at_50', (ctypes.c_ubyte * 4), 80), ('stat_rate', (ctypes.c_ubyte * 4), 84), ('hop_limit', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 4), 96), ('tclass', (ctypes.c_ubyte * 8), 100), ('flow_label', (ctypes.c_ubyte * 20), 108), ('rgid_rip', ((ctypes.c_ubyte * 8) * 16), 128), ('reserved_at_100', (ctypes.c_ubyte * 4), 256), ('f_dscp', (ctypes.c_ubyte * 1), 260), ('f_ecn', (ctypes.c_ubyte * 1), 261), ('reserved_at_106', (ctypes.c_ubyte * 1), 262), ('f_eth_prio', (ctypes.c_ubyte * 1), 263), ('ecn', (ctypes.c_ubyte * 2), 264), ('dscp', (ctypes.c_ubyte * 6), 266), ('udp_sport', (ctypes.c_ubyte * 16), 272), ('dei_cfi', (ctypes.c_ubyte * 1), 288), ('eth_prio', (ctypes.c_ubyte * 3), 289), ('sl', (ctypes.c_ubyte * 4), 292), ('vhca_port_num', (ctypes.c_ubyte * 8), 296), ('rmac_47_32', (ctypes.c_ubyte * 16), 304), ('rmac_31_0', (ctypes.c_ubyte * 32), 320)])
@c.record
class struct_mlx5_ifc_flow_table_nic_cap_bits(c.Struct):
  SIZE = 32768
  nic_rx_multi_path_tirs: 'list[int]'
  nic_rx_multi_path_tirs_fts: 'list[int]'
  allow_sniffer_and_nic_rx_shared_tir: 'list[int]'
  reserved_at_3: 'list[int]'
  sw_owner_reformat_supported: 'list[int]'
  reserved_at_8: 'list[int]'
  encap_general_header: 'list[int]'
  reserved_at_21: 'list[int]'
  log_max_packet_reformat_context: 'list[int]'
  reserved_at_30: 'list[int]'
  max_encap_header_size: 'list[int]'
  reserved_at_40: 'list[int]'
  flow_table_properties_nic_receive: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  flow_table_properties_nic_receive_rdma: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  flow_table_properties_nic_receive_sniffer: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  flow_table_properties_nic_transmit: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  flow_table_properties_nic_transmit_rdma: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  flow_table_properties_nic_transmit_sniffer: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  reserved_at_e00: 'list[int]'
  ft_field_support_2_nic_receive: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  reserved_at_1480: 'list[int]'
  ft_field_support_2_nic_receive_rdma: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  reserved_at_1580: 'list[int]'
  ft_field_support_2_nic_transmit_rdma: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  reserved_at_1880: 'list[int]'
  sw_steering_nic_rx_action_drop_icm_address: 'list[int]'
  sw_steering_nic_tx_action_drop_icm_address: 'list[int]'
  sw_steering_nic_tx_action_allow_icm_address: 'list[int]'
  reserved_at_20c0: 'list[int]'
struct_mlx5_ifc_flow_table_nic_cap_bits.register_fields([('nic_rx_multi_path_tirs', (ctypes.c_ubyte * 1), 0), ('nic_rx_multi_path_tirs_fts', (ctypes.c_ubyte * 1), 1), ('allow_sniffer_and_nic_rx_shared_tir', (ctypes.c_ubyte * 1), 2), ('reserved_at_3', (ctypes.c_ubyte * 4), 3), ('sw_owner_reformat_supported', (ctypes.c_ubyte * 1), 7), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('encap_general_header', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 10), 33), ('log_max_packet_reformat_context', (ctypes.c_ubyte * 5), 43), ('reserved_at_30', (ctypes.c_ubyte * 6), 48), ('max_encap_header_size', (ctypes.c_ubyte * 10), 54), ('reserved_at_40', (ctypes.c_ubyte * 448), 64), ('flow_table_properties_nic_receive', struct_mlx5_ifc_flow_table_prop_layout_bits, 512), ('flow_table_properties_nic_receive_rdma', struct_mlx5_ifc_flow_table_prop_layout_bits, 1024), ('flow_table_properties_nic_receive_sniffer', struct_mlx5_ifc_flow_table_prop_layout_bits, 1536), ('flow_table_properties_nic_transmit', struct_mlx5_ifc_flow_table_prop_layout_bits, 2048), ('flow_table_properties_nic_transmit_rdma', struct_mlx5_ifc_flow_table_prop_layout_bits, 2560), ('flow_table_properties_nic_transmit_sniffer', struct_mlx5_ifc_flow_table_prop_layout_bits, 3072), ('reserved_at_e00', (ctypes.c_ubyte * 1536), 3584), ('ft_field_support_2_nic_receive', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 5120), ('reserved_at_1480', (ctypes.c_ubyte * 128), 5248), ('ft_field_support_2_nic_receive_rdma', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 5376), ('reserved_at_1580', (ctypes.c_ubyte * 640), 5504), ('ft_field_support_2_nic_transmit_rdma', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 6144), ('reserved_at_1880', (ctypes.c_ubyte * 1920), 6272), ('sw_steering_nic_rx_action_drop_icm_address', (ctypes.c_ubyte * 64), 8192), ('sw_steering_nic_tx_action_drop_icm_address', (ctypes.c_ubyte * 64), 8256), ('sw_steering_nic_tx_action_allow_icm_address', (ctypes.c_ubyte * 64), 8320), ('reserved_at_20c0', (ctypes.c_ubyte * 24384), 8384)])
@c.record
class struct_mlx5_ifc_port_selection_cap_bits(c.Struct):
  SIZE = 32768
  reserved_at_0: 'list[int]'
  port_select_flow_table: 'list[int]'
  reserved_at_11: 'list[int]'
  port_select_flow_table_bypass: 'list[int]'
  reserved_at_13: 'list[int]'
  reserved_at_20: 'list[int]'
  flow_table_properties_port_selection: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  ft_field_support_2_port_selection: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  reserved_at_480: 'list[int]'
struct_mlx5_ifc_port_selection_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('port_select_flow_table', (ctypes.c_ubyte * 1), 16), ('reserved_at_11', (ctypes.c_ubyte * 1), 17), ('port_select_flow_table_bypass', (ctypes.c_ubyte * 1), 18), ('reserved_at_13', (ctypes.c_ubyte * 13), 19), ('reserved_at_20', (ctypes.c_ubyte * 480), 32), ('flow_table_properties_port_selection', struct_mlx5_ifc_flow_table_prop_layout_bits, 512), ('ft_field_support_2_port_selection', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1024), ('reserved_at_480', (ctypes.c_ubyte * 31616), 1152)])
_anonenum12: dict[int, str] = {(MLX5_FDB_TO_VPORT_REG_C_0:=1): 'MLX5_FDB_TO_VPORT_REG_C_0', (MLX5_FDB_TO_VPORT_REG_C_1:=2): 'MLX5_FDB_TO_VPORT_REG_C_1', (MLX5_FDB_TO_VPORT_REG_C_2:=4): 'MLX5_FDB_TO_VPORT_REG_C_2', (MLX5_FDB_TO_VPORT_REG_C_3:=8): 'MLX5_FDB_TO_VPORT_REG_C_3', (MLX5_FDB_TO_VPORT_REG_C_4:=16): 'MLX5_FDB_TO_VPORT_REG_C_4', (MLX5_FDB_TO_VPORT_REG_C_5:=32): 'MLX5_FDB_TO_VPORT_REG_C_5', (MLX5_FDB_TO_VPORT_REG_C_6:=64): 'MLX5_FDB_TO_VPORT_REG_C_6', (MLX5_FDB_TO_VPORT_REG_C_7:=128): 'MLX5_FDB_TO_VPORT_REG_C_7'}
@c.record
class struct_mlx5_ifc_flow_table_eswitch_cap_bits(c.Struct):
  SIZE = 32768
  fdb_to_vport_reg_c_id: 'list[int]'
  reserved_at_8: 'list[int]'
  fdb_uplink_hairpin: 'list[int]'
  fdb_multi_path_any_table_limit_regc: 'list[int]'
  reserved_at_f: 'list[int]'
  fdb_dynamic_tunnel: 'list[int]'
  reserved_at_11: 'list[int]'
  fdb_multi_path_any_table: 'list[int]'
  reserved_at_13: 'list[int]'
  fdb_modify_header_fwd_to_table: 'list[int]'
  fdb_ipv4_ttl_modify: 'list[int]'
  flow_source: 'list[int]'
  reserved_at_18: 'list[int]'
  multi_fdb_encap: 'list[int]'
  egress_acl_forward_to_vport: 'list[int]'
  fdb_multi_path_to_table: 'list[int]'
  reserved_at_1d: 'list[int]'
  reserved_at_20: 'list[int]'
  flow_table_properties_nic_esw_fdb: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  flow_table_properties_esw_acl_ingress: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  flow_table_properties_esw_acl_egress: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  reserved_at_800: 'list[int]'
  ft_field_support_2_esw_fdb: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  ft_field_bitmask_support_2_esw_fdb: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  reserved_at_1500: 'list[int]'
  sw_steering_fdb_action_drop_icm_address_rx: 'list[int]'
  sw_steering_fdb_action_drop_icm_address_tx: 'list[int]'
  sw_steering_uplink_icm_address_rx: 'list[int]'
  sw_steering_uplink_icm_address_tx: 'list[int]'
  reserved_at_1900: 'list[int]'
struct_mlx5_ifc_flow_table_eswitch_cap_bits.register_fields([('fdb_to_vport_reg_c_id', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 5), 8), ('fdb_uplink_hairpin', (ctypes.c_ubyte * 1), 13), ('fdb_multi_path_any_table_limit_regc', (ctypes.c_ubyte * 1), 14), ('reserved_at_f', (ctypes.c_ubyte * 1), 15), ('fdb_dynamic_tunnel', (ctypes.c_ubyte * 1), 16), ('reserved_at_11', (ctypes.c_ubyte * 1), 17), ('fdb_multi_path_any_table', (ctypes.c_ubyte * 1), 18), ('reserved_at_13', (ctypes.c_ubyte * 2), 19), ('fdb_modify_header_fwd_to_table', (ctypes.c_ubyte * 1), 21), ('fdb_ipv4_ttl_modify', (ctypes.c_ubyte * 1), 22), ('flow_source', (ctypes.c_ubyte * 1), 23), ('reserved_at_18', (ctypes.c_ubyte * 2), 24), ('multi_fdb_encap', (ctypes.c_ubyte * 1), 26), ('egress_acl_forward_to_vport', (ctypes.c_ubyte * 1), 27), ('fdb_multi_path_to_table', (ctypes.c_ubyte * 1), 28), ('reserved_at_1d', (ctypes.c_ubyte * 3), 29), ('reserved_at_20', (ctypes.c_ubyte * 480), 32), ('flow_table_properties_nic_esw_fdb', struct_mlx5_ifc_flow_table_prop_layout_bits, 512), ('flow_table_properties_esw_acl_ingress', struct_mlx5_ifc_flow_table_prop_layout_bits, 1024), ('flow_table_properties_esw_acl_egress', struct_mlx5_ifc_flow_table_prop_layout_bits, 1536), ('reserved_at_800', (ctypes.c_ubyte * 3072), 2048), ('ft_field_support_2_esw_fdb', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 5120), ('ft_field_bitmask_support_2_esw_fdb', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 5248), ('reserved_at_1500', (ctypes.c_ubyte * 768), 5376), ('sw_steering_fdb_action_drop_icm_address_rx', (ctypes.c_ubyte * 64), 6144), ('sw_steering_fdb_action_drop_icm_address_tx', (ctypes.c_ubyte * 64), 6208), ('sw_steering_uplink_icm_address_rx', (ctypes.c_ubyte * 64), 6272), ('sw_steering_uplink_icm_address_tx', (ctypes.c_ubyte * 64), 6336), ('reserved_at_1900', (ctypes.c_ubyte * 26368), 6400)])
@c.record
class struct_mlx5_ifc_wqe_based_flow_table_cap_bits(c.Struct):
  SIZE = 480
  reserved_at_0: 'list[int]'
  log_max_num_ste: 'list[int]'
  reserved_at_8: 'list[int]'
  log_max_num_stc: 'list[int]'
  reserved_at_10: 'list[int]'
  log_max_num_rtc: 'list[int]'
  reserved_at_18: 'list[int]'
  log_max_num_header_modify_pattern: 'list[int]'
  rtc_hash_split_table: 'list[int]'
  rtc_linear_lookup_table: 'list[int]'
  reserved_at_22: 'list[int]'
  stc_alloc_log_granularity: 'list[int]'
  reserved_at_28: 'list[int]'
  stc_alloc_log_max: 'list[int]'
  reserved_at_30: 'list[int]'
  ste_alloc_log_granularity: 'list[int]'
  reserved_at_38: 'list[int]'
  ste_alloc_log_max: 'list[int]'
  reserved_at_40: 'list[int]'
  rtc_reparse_mode: 'list[int]'
  reserved_at_50: 'list[int]'
  rtc_index_mode: 'list[int]'
  reserved_at_58: 'list[int]'
  rtc_log_depth_max: 'list[int]'
  reserved_at_60: 'list[int]'
  ste_format: 'list[int]'
  stc_action_type: 'list[int]'
  header_insert_type: 'list[int]'
  header_remove_type: 'list[int]'
  trivial_match_definer: 'list[int]'
  reserved_at_140: 'list[int]'
  rtc_max_num_hash_definer_gen_wqe: 'list[int]'
  reserved_at_160: 'list[int]'
  access_index_mode: 'list[int]'
  reserved_at_180: 'list[int]'
  ste_format_gen_wqe: 'list[int]'
  linear_match_definer_reg_c3: 'list[int]'
  fdb_jump_to_tir_stc: 'list[int]'
  reserved_at_1c1: 'list[int]'
struct_mlx5_ifc_wqe_based_flow_table_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 3), 0), ('log_max_num_ste', (ctypes.c_ubyte * 5), 3), ('reserved_at_8', (ctypes.c_ubyte * 3), 8), ('log_max_num_stc', (ctypes.c_ubyte * 5), 11), ('reserved_at_10', (ctypes.c_ubyte * 3), 16), ('log_max_num_rtc', (ctypes.c_ubyte * 5), 19), ('reserved_at_18', (ctypes.c_ubyte * 3), 24), ('log_max_num_header_modify_pattern', (ctypes.c_ubyte * 5), 27), ('rtc_hash_split_table', (ctypes.c_ubyte * 1), 32), ('rtc_linear_lookup_table', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 1), 34), ('stc_alloc_log_granularity', (ctypes.c_ubyte * 5), 35), ('reserved_at_28', (ctypes.c_ubyte * 3), 40), ('stc_alloc_log_max', (ctypes.c_ubyte * 5), 43), ('reserved_at_30', (ctypes.c_ubyte * 3), 48), ('ste_alloc_log_granularity', (ctypes.c_ubyte * 5), 51), ('reserved_at_38', (ctypes.c_ubyte * 3), 56), ('ste_alloc_log_max', (ctypes.c_ubyte * 5), 59), ('reserved_at_40', (ctypes.c_ubyte * 11), 64), ('rtc_reparse_mode', (ctypes.c_ubyte * 5), 75), ('reserved_at_50', (ctypes.c_ubyte * 3), 80), ('rtc_index_mode', (ctypes.c_ubyte * 5), 83), ('reserved_at_58', (ctypes.c_ubyte * 3), 88), ('rtc_log_depth_max', (ctypes.c_ubyte * 5), 91), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('ste_format', (ctypes.c_ubyte * 16), 112), ('stc_action_type', (ctypes.c_ubyte * 128), 128), ('header_insert_type', (ctypes.c_ubyte * 16), 256), ('header_remove_type', (ctypes.c_ubyte * 16), 272), ('trivial_match_definer', (ctypes.c_ubyte * 32), 288), ('reserved_at_140', (ctypes.c_ubyte * 27), 320), ('rtc_max_num_hash_definer_gen_wqe', (ctypes.c_ubyte * 5), 347), ('reserved_at_160', (ctypes.c_ubyte * 24), 352), ('access_index_mode', (ctypes.c_ubyte * 8), 376), ('reserved_at_180', (ctypes.c_ubyte * 16), 384), ('ste_format_gen_wqe', (ctypes.c_ubyte * 16), 400), ('linear_match_definer_reg_c3', (ctypes.c_ubyte * 32), 416), ('fdb_jump_to_tir_stc', (ctypes.c_ubyte * 1), 448), ('reserved_at_1c1', (ctypes.c_ubyte * 31), 449)])
@c.record
class struct_mlx5_ifc_esw_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  merged_eswitch: 'list[int]'
  reserved_at_1e: 'list[int]'
  reserved_at_20: 'list[int]'
  esw_manager_vport_number_valid: 'list[int]'
  reserved_at_61: 'list[int]'
  esw_manager_vport_number: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_esw_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 29), 0), ('merged_eswitch', (ctypes.c_ubyte * 1), 29), ('reserved_at_1e', (ctypes.c_ubyte * 2), 30), ('reserved_at_20', (ctypes.c_ubyte * 64), 32), ('esw_manager_vport_number_valid', (ctypes.c_ubyte * 1), 96), ('reserved_at_61', (ctypes.c_ubyte * 15), 97), ('esw_manager_vport_number', (ctypes.c_ubyte * 16), 112), ('reserved_at_80', (ctypes.c_ubyte * 1920), 128)])
_anonenum13: dict[int, str] = {(MLX5_COUNTER_SOURCE_ESWITCH:=0): 'MLX5_COUNTER_SOURCE_ESWITCH', (MLX5_COUNTER_FLOW_ESWITCH:=1): 'MLX5_COUNTER_FLOW_ESWITCH'}
@c.record
class struct_mlx5_ifc_e_switch_cap_bits(c.Struct):
  SIZE = 2048
  vport_svlan_strip: 'list[int]'
  vport_cvlan_strip: 'list[int]'
  vport_svlan_insert: 'list[int]'
  vport_cvlan_insert_if_not_exist: 'list[int]'
  vport_cvlan_insert_overwrite: 'list[int]'
  reserved_at_5: 'list[int]'
  vport_cvlan_insert_always: 'list[int]'
  esw_shared_ingress_acl: 'list[int]'
  esw_uplink_ingress_acl: 'list[int]'
  root_ft_on_other_esw: 'list[int]'
  reserved_at_a: 'list[int]'
  esw_functions_changed: 'list[int]'
  reserved_at_1a: 'list[int]'
  ecpf_vport_exists: 'list[int]'
  counter_eswitch_affinity: 'list[int]'
  merged_eswitch: 'list[int]'
  nic_vport_node_guid_modify: 'list[int]'
  nic_vport_port_guid_modify: 'list[int]'
  vxlan_encap_decap: 'list[int]'
  nvgre_encap_decap: 'list[int]'
  reserved_at_22: 'list[int]'
  log_max_fdb_encap_uplink: 'list[int]'
  reserved_at_21: 'list[int]'
  log_max_packet_reformat_context: 'list[int]'
  reserved_2b: 'list[int]'
  max_encap_header_size: 'list[int]'
  reserved_at_40: 'list[int]'
  log_max_esw_sf: 'list[int]'
  esw_sf_base_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_e_switch_cap_bits.register_fields([('vport_svlan_strip', (ctypes.c_ubyte * 1), 0), ('vport_cvlan_strip', (ctypes.c_ubyte * 1), 1), ('vport_svlan_insert', (ctypes.c_ubyte * 1), 2), ('vport_cvlan_insert_if_not_exist', (ctypes.c_ubyte * 1), 3), ('vport_cvlan_insert_overwrite', (ctypes.c_ubyte * 1), 4), ('reserved_at_5', (ctypes.c_ubyte * 1), 5), ('vport_cvlan_insert_always', (ctypes.c_ubyte * 1), 6), ('esw_shared_ingress_acl', (ctypes.c_ubyte * 1), 7), ('esw_uplink_ingress_acl', (ctypes.c_ubyte * 1), 8), ('root_ft_on_other_esw', (ctypes.c_ubyte * 1), 9), ('reserved_at_a', (ctypes.c_ubyte * 15), 10), ('esw_functions_changed', (ctypes.c_ubyte * 1), 25), ('reserved_at_1a', (ctypes.c_ubyte * 1), 26), ('ecpf_vport_exists', (ctypes.c_ubyte * 1), 27), ('counter_eswitch_affinity', (ctypes.c_ubyte * 1), 28), ('merged_eswitch', (ctypes.c_ubyte * 1), 29), ('nic_vport_node_guid_modify', (ctypes.c_ubyte * 1), 30), ('nic_vport_port_guid_modify', (ctypes.c_ubyte * 1), 31), ('vxlan_encap_decap', (ctypes.c_ubyte * 1), 32), ('nvgre_encap_decap', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 1), 34), ('log_max_fdb_encap_uplink', (ctypes.c_ubyte * 5), 35), ('reserved_at_21', (ctypes.c_ubyte * 3), 40), ('log_max_packet_reformat_context', (ctypes.c_ubyte * 5), 43), ('reserved_2b', (ctypes.c_ubyte * 6), 48), ('max_encap_header_size', (ctypes.c_ubyte * 10), 54), ('reserved_at_40', (ctypes.c_ubyte * 11), 64), ('log_max_esw_sf', (ctypes.c_ubyte * 5), 75), ('esw_sf_base_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 1952), 96)])
@c.record
class struct_mlx5_ifc_qos_cap_bits(c.Struct):
  SIZE = 2048
  packet_pacing: 'list[int]'
  esw_scheduling: 'list[int]'
  esw_bw_share: 'list[int]'
  esw_rate_limit: 'list[int]'
  reserved_at_4: 'list[int]'
  packet_pacing_burst_bound: 'list[int]'
  packet_pacing_typical_size: 'list[int]'
  reserved_at_7: 'list[int]'
  nic_sq_scheduling: 'list[int]'
  nic_bw_share: 'list[int]'
  nic_rate_limit: 'list[int]'
  packet_pacing_uid: 'list[int]'
  log_esw_max_sched_depth: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  esw_cross_esw_sched: 'list[int]'
  reserved_at_2a: 'list[int]'
  log_max_qos_nic_queue_group: 'list[int]'
  reserved_at_30: 'list[int]'
  packet_pacing_max_rate: 'list[int]'
  packet_pacing_min_rate: 'list[int]'
  reserved_at_80: 'list[int]'
  log_esw_max_rate_limit: 'list[int]'
  packet_pacing_rate_table_size: 'list[int]'
  esw_element_type: 'list[int]'
  esw_tsar_type: 'list[int]'
  reserved_at_c0: 'list[int]'
  max_qos_para_vport: 'list[int]'
  max_tsar_bw_share: 'list[int]'
  nic_element_type: 'list[int]'
  nic_tsar_type: 'list[int]'
  reserved_at_120: 'list[int]'
  log_meter_aso_granularity: 'list[int]'
  reserved_at_128: 'list[int]'
  log_meter_aso_max_alloc: 'list[int]'
  reserved_at_130: 'list[int]'
  log_max_num_meter_aso: 'list[int]'
  reserved_at_138: 'list[int]'
  reserved_at_140: 'list[int]'
struct_mlx5_ifc_qos_cap_bits.register_fields([('packet_pacing', (ctypes.c_ubyte * 1), 0), ('esw_scheduling', (ctypes.c_ubyte * 1), 1), ('esw_bw_share', (ctypes.c_ubyte * 1), 2), ('esw_rate_limit', (ctypes.c_ubyte * 1), 3), ('reserved_at_4', (ctypes.c_ubyte * 1), 4), ('packet_pacing_burst_bound', (ctypes.c_ubyte * 1), 5), ('packet_pacing_typical_size', (ctypes.c_ubyte * 1), 6), ('reserved_at_7', (ctypes.c_ubyte * 1), 7), ('nic_sq_scheduling', (ctypes.c_ubyte * 1), 8), ('nic_bw_share', (ctypes.c_ubyte * 1), 9), ('nic_rate_limit', (ctypes.c_ubyte * 1), 10), ('packet_pacing_uid', (ctypes.c_ubyte * 1), 11), ('log_esw_max_sched_depth', (ctypes.c_ubyte * 4), 12), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 9), 32), ('esw_cross_esw_sched', (ctypes.c_ubyte * 1), 41), ('reserved_at_2a', (ctypes.c_ubyte * 1), 42), ('log_max_qos_nic_queue_group', (ctypes.c_ubyte * 5), 43), ('reserved_at_30', (ctypes.c_ubyte * 16), 48), ('packet_pacing_max_rate', (ctypes.c_ubyte * 32), 64), ('packet_pacing_min_rate', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 11), 128), ('log_esw_max_rate_limit', (ctypes.c_ubyte * 5), 139), ('packet_pacing_rate_table_size', (ctypes.c_ubyte * 16), 144), ('esw_element_type', (ctypes.c_ubyte * 16), 160), ('esw_tsar_type', (ctypes.c_ubyte * 16), 176), ('reserved_at_c0', (ctypes.c_ubyte * 16), 192), ('max_qos_para_vport', (ctypes.c_ubyte * 16), 208), ('max_tsar_bw_share', (ctypes.c_ubyte * 32), 224), ('nic_element_type', (ctypes.c_ubyte * 16), 256), ('nic_tsar_type', (ctypes.c_ubyte * 16), 272), ('reserved_at_120', (ctypes.c_ubyte * 3), 288), ('log_meter_aso_granularity', (ctypes.c_ubyte * 5), 291), ('reserved_at_128', (ctypes.c_ubyte * 3), 296), ('log_meter_aso_max_alloc', (ctypes.c_ubyte * 5), 299), ('reserved_at_130', (ctypes.c_ubyte * 3), 304), ('log_max_num_meter_aso', (ctypes.c_ubyte * 5), 307), ('reserved_at_138', (ctypes.c_ubyte * 8), 312), ('reserved_at_140', (ctypes.c_ubyte * 1728), 320)])
@c.record
class struct_mlx5_ifc_debug_cap_bits(c.Struct):
  SIZE = 2048
  core_dump_general: 'list[int]'
  core_dump_qp: 'list[int]'
  reserved_at_2: 'list[int]'
  resource_dump: 'list[int]'
  reserved_at_a: 'list[int]'
  reserved_at_20: 'list[int]'
  stall_detect: 'list[int]'
  reserved_at_23: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_debug_cap_bits.register_fields([('core_dump_general', (ctypes.c_ubyte * 1), 0), ('core_dump_qp', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 7), 2), ('resource_dump', (ctypes.c_ubyte * 1), 9), ('reserved_at_a', (ctypes.c_ubyte * 22), 10), ('reserved_at_20', (ctypes.c_ubyte * 2), 32), ('stall_detect', (ctypes.c_ubyte * 1), 34), ('reserved_at_23', (ctypes.c_ubyte * 29), 35), ('reserved_at_40', (ctypes.c_ubyte * 1984), 64)])
@c.record
class struct_mlx5_ifc_per_protocol_networking_offload_caps_bits(c.Struct):
  SIZE = 2048
  csum_cap: 'list[int]'
  vlan_cap: 'list[int]'
  lro_cap: 'list[int]'
  lro_psh_flag: 'list[int]'
  lro_time_stamp: 'list[int]'
  reserved_at_5: 'list[int]'
  wqe_vlan_insert: 'list[int]'
  self_lb_en_modifiable: 'list[int]'
  reserved_at_9: 'list[int]'
  max_lso_cap: 'list[int]'
  multi_pkt_send_wqe: 'list[int]'
  wqe_inline_mode: 'list[int]'
  rss_ind_tbl_cap: 'list[int]'
  reg_umr_sq: 'list[int]'
  scatter_fcs: 'list[int]'
  enhanced_multi_pkt_send_wqe: 'list[int]'
  tunnel_lso_const_out_ip_id: 'list[int]'
  tunnel_lro_gre: 'list[int]'
  tunnel_lro_vxlan: 'list[int]'
  tunnel_stateless_gre: 'list[int]'
  tunnel_stateless_vxlan: 'list[int]'
  swp: 'list[int]'
  swp_csum: 'list[int]'
  swp_lso: 'list[int]'
  cqe_checksum_full: 'list[int]'
  tunnel_stateless_geneve_tx: 'list[int]'
  tunnel_stateless_mpls_over_udp: 'list[int]'
  tunnel_stateless_mpls_over_gre: 'list[int]'
  tunnel_stateless_vxlan_gpe: 'list[int]'
  tunnel_stateless_ipv4_over_vxlan: 'list[int]'
  tunnel_stateless_ip_over_ip: 'list[int]'
  insert_trailer: 'list[int]'
  reserved_at_2b: 'list[int]'
  tunnel_stateless_ip_over_ip_rx: 'list[int]'
  tunnel_stateless_ip_over_ip_tx: 'list[int]'
  reserved_at_2e: 'list[int]'
  max_vxlan_udp_ports: 'list[int]'
  swp_csum_l4_partial: 'list[int]'
  reserved_at_39: 'list[int]'
  max_geneve_opt_len: 'list[int]'
  tunnel_stateless_geneve_rx: 'list[int]'
  reserved_at_40: 'list[int]'
  lro_min_mss_size: 'list[int]'
  reserved_at_60: 'list[int]'
  lro_timer_supported_periods: 'list[list[int]]'
  reserved_at_200: 'list[int]'
struct_mlx5_ifc_per_protocol_networking_offload_caps_bits.register_fields([('csum_cap', (ctypes.c_ubyte * 1), 0), ('vlan_cap', (ctypes.c_ubyte * 1), 1), ('lro_cap', (ctypes.c_ubyte * 1), 2), ('lro_psh_flag', (ctypes.c_ubyte * 1), 3), ('lro_time_stamp', (ctypes.c_ubyte * 1), 4), ('reserved_at_5', (ctypes.c_ubyte * 2), 5), ('wqe_vlan_insert', (ctypes.c_ubyte * 1), 7), ('self_lb_en_modifiable', (ctypes.c_ubyte * 1), 8), ('reserved_at_9', (ctypes.c_ubyte * 2), 9), ('max_lso_cap', (ctypes.c_ubyte * 5), 11), ('multi_pkt_send_wqe', (ctypes.c_ubyte * 2), 16), ('wqe_inline_mode', (ctypes.c_ubyte * 2), 18), ('rss_ind_tbl_cap', (ctypes.c_ubyte * 4), 20), ('reg_umr_sq', (ctypes.c_ubyte * 1), 24), ('scatter_fcs', (ctypes.c_ubyte * 1), 25), ('enhanced_multi_pkt_send_wqe', (ctypes.c_ubyte * 1), 26), ('tunnel_lso_const_out_ip_id', (ctypes.c_ubyte * 1), 27), ('tunnel_lro_gre', (ctypes.c_ubyte * 1), 28), ('tunnel_lro_vxlan', (ctypes.c_ubyte * 1), 29), ('tunnel_stateless_gre', (ctypes.c_ubyte * 1), 30), ('tunnel_stateless_vxlan', (ctypes.c_ubyte * 1), 31), ('swp', (ctypes.c_ubyte * 1), 32), ('swp_csum', (ctypes.c_ubyte * 1), 33), ('swp_lso', (ctypes.c_ubyte * 1), 34), ('cqe_checksum_full', (ctypes.c_ubyte * 1), 35), ('tunnel_stateless_geneve_tx', (ctypes.c_ubyte * 1), 36), ('tunnel_stateless_mpls_over_udp', (ctypes.c_ubyte * 1), 37), ('tunnel_stateless_mpls_over_gre', (ctypes.c_ubyte * 1), 38), ('tunnel_stateless_vxlan_gpe', (ctypes.c_ubyte * 1), 39), ('tunnel_stateless_ipv4_over_vxlan', (ctypes.c_ubyte * 1), 40), ('tunnel_stateless_ip_over_ip', (ctypes.c_ubyte * 1), 41), ('insert_trailer', (ctypes.c_ubyte * 1), 42), ('reserved_at_2b', (ctypes.c_ubyte * 1), 43), ('tunnel_stateless_ip_over_ip_rx', (ctypes.c_ubyte * 1), 44), ('tunnel_stateless_ip_over_ip_tx', (ctypes.c_ubyte * 1), 45), ('reserved_at_2e', (ctypes.c_ubyte * 2), 46), ('max_vxlan_udp_ports', (ctypes.c_ubyte * 8), 48), ('swp_csum_l4_partial', (ctypes.c_ubyte * 1), 56), ('reserved_at_39', (ctypes.c_ubyte * 5), 57), ('max_geneve_opt_len', (ctypes.c_ubyte * 1), 62), ('tunnel_stateless_geneve_rx', (ctypes.c_ubyte * 1), 63), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('lro_min_mss_size', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 288), 96), ('lro_timer_supported_periods', ((ctypes.c_ubyte * 32) * 4), 384), ('reserved_at_200', (ctypes.c_ubyte * 1536), 512)])
_anonenum14: dict[int, str] = {(MLX5_TIMESTAMP_FORMAT_CAP_FREE_RUNNING:=0): 'MLX5_TIMESTAMP_FORMAT_CAP_FREE_RUNNING', (MLX5_TIMESTAMP_FORMAT_CAP_REAL_TIME:=1): 'MLX5_TIMESTAMP_FORMAT_CAP_REAL_TIME', (MLX5_TIMESTAMP_FORMAT_CAP_FREE_RUNNING_AND_REAL_TIME:=2): 'MLX5_TIMESTAMP_FORMAT_CAP_FREE_RUNNING_AND_REAL_TIME'}
@c.record
class struct_mlx5_ifc_roce_cap_bits(c.Struct):
  SIZE = 2048
  roce_apm: 'list[int]'
  reserved_at_1: 'list[int]'
  sw_r_roce_src_udp_port: 'list[int]'
  fl_rc_qp_when_roce_disabled: 'list[int]'
  fl_rc_qp_when_roce_enabled: 'list[int]'
  roce_cc_general: 'list[int]'
  qp_ooo_transmit_default: 'list[int]'
  reserved_at_9: 'list[int]'
  qp_ts_format: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_80: 'list[int]'
  l3_type: 'list[int]'
  reserved_at_90: 'list[int]'
  roce_version: 'list[int]'
  reserved_at_a0: 'list[int]'
  r_roce_dest_udp_port: 'list[int]'
  r_roce_max_src_udp_port: 'list[int]'
  r_roce_min_src_udp_port: 'list[int]'
  reserved_at_e0: 'list[int]'
  roce_address_table_size: 'list[int]'
  reserved_at_100: 'list[int]'
struct_mlx5_ifc_roce_cap_bits.register_fields([('roce_apm', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 3), 1), ('sw_r_roce_src_udp_port', (ctypes.c_ubyte * 1), 4), ('fl_rc_qp_when_roce_disabled', (ctypes.c_ubyte * 1), 5), ('fl_rc_qp_when_roce_enabled', (ctypes.c_ubyte * 1), 6), ('roce_cc_general', (ctypes.c_ubyte * 1), 7), ('qp_ooo_transmit_default', (ctypes.c_ubyte * 1), 8), ('reserved_at_9', (ctypes.c_ubyte * 21), 9), ('qp_ts_format', (ctypes.c_ubyte * 2), 30), ('reserved_at_20', (ctypes.c_ubyte * 96), 32), ('reserved_at_80', (ctypes.c_ubyte * 12), 128), ('l3_type', (ctypes.c_ubyte * 4), 140), ('reserved_at_90', (ctypes.c_ubyte * 8), 144), ('roce_version', (ctypes.c_ubyte * 8), 152), ('reserved_at_a0', (ctypes.c_ubyte * 16), 160), ('r_roce_dest_udp_port', (ctypes.c_ubyte * 16), 176), ('r_roce_max_src_udp_port', (ctypes.c_ubyte * 16), 192), ('r_roce_min_src_udp_port', (ctypes.c_ubyte * 16), 208), ('reserved_at_e0', (ctypes.c_ubyte * 16), 224), ('roce_address_table_size', (ctypes.c_ubyte * 16), 240), ('reserved_at_100', (ctypes.c_ubyte * 1792), 256)])
@c.record
class struct_mlx5_ifc_sync_steering_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_sync_steering_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 192), 64)])
@c.record
class struct_mlx5_ifc_sync_steering_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_sync_steering_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_sync_crypto_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  crypto_type: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_sync_crypto_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('crypto_type', (ctypes.c_ubyte * 16), 112), ('reserved_at_80', (ctypes.c_ubyte * 128), 128)])
@c.record
class struct_mlx5_ifc_sync_crypto_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_sync_crypto_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_device_mem_cap_bits(c.Struct):
  SIZE = 2048
  memic: 'list[int]'
  reserved_at_1: 'list[int]'
  reserved_at_20: 'list[int]'
  log_min_memic_alloc_size: 'list[int]'
  reserved_at_30: 'list[int]'
  log_max_memic_addr_alignment: 'list[int]'
  memic_bar_start_addr: 'list[int]'
  memic_bar_size: 'list[int]'
  max_memic_size: 'list[int]'
  steering_sw_icm_start_address: 'list[int]'
  reserved_at_100: 'list[int]'
  log_header_modify_sw_icm_size: 'list[int]'
  reserved_at_110: 'list[int]'
  log_sw_icm_alloc_granularity: 'list[int]'
  log_steering_sw_icm_size: 'list[int]'
  log_indirect_encap_sw_icm_size: 'list[int]'
  reserved_at_128: 'list[int]'
  log_header_modify_pattern_sw_icm_size: 'list[int]'
  header_modify_sw_icm_start_address: 'list[int]'
  reserved_at_180: 'list[int]'
  header_modify_pattern_sw_icm_start_address: 'list[int]'
  memic_operations: 'list[int]'
  reserved_at_220: 'list[int]'
  indirect_encap_sw_icm_start_address: 'list[int]'
  reserved_at_280: 'list[int]'
struct_mlx5_ifc_device_mem_cap_bits.register_fields([('memic', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 31), 1), ('reserved_at_20', (ctypes.c_ubyte * 11), 32), ('log_min_memic_alloc_size', (ctypes.c_ubyte * 5), 43), ('reserved_at_30', (ctypes.c_ubyte * 8), 48), ('log_max_memic_addr_alignment', (ctypes.c_ubyte * 8), 56), ('memic_bar_start_addr', (ctypes.c_ubyte * 64), 64), ('memic_bar_size', (ctypes.c_ubyte * 32), 128), ('max_memic_size', (ctypes.c_ubyte * 32), 160), ('steering_sw_icm_start_address', (ctypes.c_ubyte * 64), 192), ('reserved_at_100', (ctypes.c_ubyte * 8), 256), ('log_header_modify_sw_icm_size', (ctypes.c_ubyte * 8), 264), ('reserved_at_110', (ctypes.c_ubyte * 2), 272), ('log_sw_icm_alloc_granularity', (ctypes.c_ubyte * 6), 274), ('log_steering_sw_icm_size', (ctypes.c_ubyte * 8), 280), ('log_indirect_encap_sw_icm_size', (ctypes.c_ubyte * 8), 288), ('reserved_at_128', (ctypes.c_ubyte * 16), 296), ('log_header_modify_pattern_sw_icm_size', (ctypes.c_ubyte * 8), 312), ('header_modify_sw_icm_start_address', (ctypes.c_ubyte * 64), 320), ('reserved_at_180', (ctypes.c_ubyte * 64), 384), ('header_modify_pattern_sw_icm_start_address', (ctypes.c_ubyte * 64), 448), ('memic_operations', (ctypes.c_ubyte * 32), 512), ('reserved_at_220', (ctypes.c_ubyte * 32), 544), ('indirect_encap_sw_icm_start_address', (ctypes.c_ubyte * 64), 576), ('reserved_at_280', (ctypes.c_ubyte * 1408), 640)])
@c.record
class struct_mlx5_ifc_device_event_cap_bits(c.Struct):
  SIZE = 512
  user_affiliated_events: 'list[list[int]]'
  user_unaffiliated_events: 'list[list[int]]'
struct_mlx5_ifc_device_event_cap_bits.register_fields([('user_affiliated_events', ((ctypes.c_ubyte * 64) * 4), 0), ('user_unaffiliated_events', ((ctypes.c_ubyte * 64) * 4), 256)])
@c.record
class struct_mlx5_ifc_virtio_emulation_cap_bits(c.Struct):
  SIZE = 2048
  desc_tunnel_offload_type: 'list[int]'
  eth_frame_offload_type: 'list[int]'
  virtio_version_1_0: 'list[int]'
  device_features_bits_mask: 'list[int]'
  event_mode: 'list[int]'
  virtio_queue_type: 'list[int]'
  max_tunnel_desc: 'list[int]'
  reserved_at_30: 'list[int]'
  log_doorbell_stride: 'list[int]'
  reserved_at_38: 'list[int]'
  log_doorbell_bar_size: 'list[int]'
  doorbell_bar_offset: 'list[int]'
  max_emulated_devices: 'list[int]'
  max_num_virtio_queues: 'list[int]'
  reserved_at_a0: 'list[int]'
  reserved_at_c0: 'list[int]'
  desc_group_mkey_supported: 'list[int]'
  freeze_to_rdy_supported: 'list[int]'
  reserved_at_d5: 'list[int]'
  reserved_at_e0: 'list[int]'
  umem_1_buffer_param_a: 'list[int]'
  umem_1_buffer_param_b: 'list[int]'
  umem_2_buffer_param_a: 'list[int]'
  umem_2_buffer_param_b: 'list[int]'
  umem_3_buffer_param_a: 'list[int]'
  umem_3_buffer_param_b: 'list[int]'
  reserved_at_1c0: 'list[int]'
struct_mlx5_ifc_virtio_emulation_cap_bits.register_fields([('desc_tunnel_offload_type', (ctypes.c_ubyte * 1), 0), ('eth_frame_offload_type', (ctypes.c_ubyte * 1), 1), ('virtio_version_1_0', (ctypes.c_ubyte * 1), 2), ('device_features_bits_mask', (ctypes.c_ubyte * 13), 3), ('event_mode', (ctypes.c_ubyte * 8), 16), ('virtio_queue_type', (ctypes.c_ubyte * 8), 24), ('max_tunnel_desc', (ctypes.c_ubyte * 16), 32), ('reserved_at_30', (ctypes.c_ubyte * 3), 48), ('log_doorbell_stride', (ctypes.c_ubyte * 5), 51), ('reserved_at_38', (ctypes.c_ubyte * 3), 56), ('log_doorbell_bar_size', (ctypes.c_ubyte * 5), 59), ('doorbell_bar_offset', (ctypes.c_ubyte * 64), 64), ('max_emulated_devices', (ctypes.c_ubyte * 8), 128), ('max_num_virtio_queues', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 19), 192), ('desc_group_mkey_supported', (ctypes.c_ubyte * 1), 211), ('freeze_to_rdy_supported', (ctypes.c_ubyte * 1), 212), ('reserved_at_d5', (ctypes.c_ubyte * 11), 213), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224), ('umem_1_buffer_param_a', (ctypes.c_ubyte * 32), 256), ('umem_1_buffer_param_b', (ctypes.c_ubyte * 32), 288), ('umem_2_buffer_param_a', (ctypes.c_ubyte * 32), 320), ('umem_2_buffer_param_b', (ctypes.c_ubyte * 32), 352), ('umem_3_buffer_param_a', (ctypes.c_ubyte * 32), 384), ('umem_3_buffer_param_b', (ctypes.c_ubyte * 32), 416), ('reserved_at_1c0', (ctypes.c_ubyte * 1600), 448)])
_anonenum15: dict[int, str] = {(MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_1_BYTE:=0): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_1_BYTE', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_2_BYTES:=2): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_2_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_4_BYTES:=4): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_4_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_8_BYTES:=8): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_8_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_16_BYTES:=16): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_16_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_32_BYTES:=32): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_32_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_64_BYTES:=64): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_64_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_128_BYTES:=128): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_128_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_256_BYTES:=256): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_256_BYTES'}
_anonenum16: dict[int, str] = {(MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_1_BYTE:=1): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_1_BYTE', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_2_BYTES:=2): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_2_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_4_BYTES:=4): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_4_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_8_BYTES:=8): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_8_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_16_BYTES:=16): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_16_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_32_BYTES:=32): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_32_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_64_BYTES:=64): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_64_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_128_BYTES:=128): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_128_BYTES', (MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_256_BYTES:=256): 'MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_256_BYTES'}
@c.record
class struct_mlx5_ifc_atomic_caps_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  atomic_req_8B_endianness_mode: 'list[int]'
  reserved_at_42: 'list[int]'
  supported_atomic_req_8B_endianness_mode_1: 'list[int]'
  reserved_at_47: 'list[int]'
  reserved_at_60: 'list[int]'
  reserved_at_80: 'list[int]'
  atomic_operations: 'list[int]'
  reserved_at_a0: 'list[int]'
  atomic_size_qp: 'list[int]'
  reserved_at_c0: 'list[int]'
  atomic_size_dc: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_atomic_caps_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 64), 0), ('atomic_req_8B_endianness_mode', (ctypes.c_ubyte * 2), 64), ('reserved_at_42', (ctypes.c_ubyte * 4), 66), ('supported_atomic_req_8B_endianness_mode_1', (ctypes.c_ubyte * 1), 70), ('reserved_at_47', (ctypes.c_ubyte * 25), 71), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 16), 128), ('atomic_operations', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 16), 160), ('atomic_size_qp', (ctypes.c_ubyte * 16), 176), ('reserved_at_c0', (ctypes.c_ubyte * 16), 192), ('atomic_size_dc', (ctypes.c_ubyte * 16), 208), ('reserved_at_e0', (ctypes.c_ubyte * 1824), 224)])
@c.record
class struct_mlx5_ifc_odp_scheme_cap_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  sig: 'list[int]'
  reserved_at_41: 'list[int]'
  page_prefetch: 'list[int]'
  reserved_at_46: 'list[int]'
  reserved_at_60: 'list[int]'
  rc_odp_caps: 'struct_mlx5_ifc_odp_per_transport_service_cap_bits'
  uc_odp_caps: 'struct_mlx5_ifc_odp_per_transport_service_cap_bits'
  ud_odp_caps: 'struct_mlx5_ifc_odp_per_transport_service_cap_bits'
  xrc_odp_caps: 'struct_mlx5_ifc_odp_per_transport_service_cap_bits'
  dc_odp_caps: 'struct_mlx5_ifc_odp_per_transport_service_cap_bits'
  reserved_at_120: 'list[int]'
struct_mlx5_ifc_odp_scheme_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 64), 0), ('sig', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 4), 65), ('page_prefetch', (ctypes.c_ubyte * 1), 69), ('reserved_at_46', (ctypes.c_ubyte * 26), 70), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('rc_odp_caps', struct_mlx5_ifc_odp_per_transport_service_cap_bits, 128), ('uc_odp_caps', struct_mlx5_ifc_odp_per_transport_service_cap_bits, 160), ('ud_odp_caps', struct_mlx5_ifc_odp_per_transport_service_cap_bits, 192), ('xrc_odp_caps', struct_mlx5_ifc_odp_per_transport_service_cap_bits, 224), ('dc_odp_caps', struct_mlx5_ifc_odp_per_transport_service_cap_bits, 256), ('reserved_at_120', (ctypes.c_ubyte * 224), 288)])
@c.record
class struct_mlx5_ifc_odp_cap_bits(c.Struct):
  SIZE = 2048
  transport_page_fault_scheme_cap: 'struct_mlx5_ifc_odp_scheme_cap_bits'
  memory_page_fault_scheme_cap: 'struct_mlx5_ifc_odp_scheme_cap_bits'
  reserved_at_400: 'list[int]'
  mem_page_fault: 'list[int]'
  reserved_at_601: 'list[int]'
  reserved_at_620: 'list[int]'
struct_mlx5_ifc_odp_cap_bits.register_fields([('transport_page_fault_scheme_cap', struct_mlx5_ifc_odp_scheme_cap_bits, 0), ('memory_page_fault_scheme_cap', struct_mlx5_ifc_odp_scheme_cap_bits, 512), ('reserved_at_400', (ctypes.c_ubyte * 512), 1024), ('mem_page_fault', (ctypes.c_ubyte * 1), 1536), ('reserved_at_601', (ctypes.c_ubyte * 31), 1537), ('reserved_at_620', (ctypes.c_ubyte * 480), 1568)])
@c.record
class struct_mlx5_ifc_tls_cap_bits(c.Struct):
  SIZE = 2048
  tls_1_2_aes_gcm_128: 'list[int]'
  tls_1_3_aes_gcm_128: 'list[int]'
  tls_1_2_aes_gcm_256: 'list[int]'
  tls_1_3_aes_gcm_256: 'list[int]'
  reserved_at_4: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_tls_cap_bits.register_fields([('tls_1_2_aes_gcm_128', (ctypes.c_ubyte * 1), 0), ('tls_1_3_aes_gcm_128', (ctypes.c_ubyte * 1), 1), ('tls_1_2_aes_gcm_256', (ctypes.c_ubyte * 1), 2), ('tls_1_3_aes_gcm_256', (ctypes.c_ubyte * 1), 3), ('reserved_at_4', (ctypes.c_ubyte * 28), 4), ('reserved_at_20', (ctypes.c_ubyte * 2016), 32)])
@c.record
class struct_mlx5_ifc_ipsec_cap_bits(c.Struct):
  SIZE = 2048
  ipsec_full_offload: 'list[int]'
  ipsec_crypto_offload: 'list[int]'
  ipsec_esn: 'list[int]'
  ipsec_crypto_esp_aes_gcm_256_encrypt: 'list[int]'
  ipsec_crypto_esp_aes_gcm_128_encrypt: 'list[int]'
  ipsec_crypto_esp_aes_gcm_256_decrypt: 'list[int]'
  ipsec_crypto_esp_aes_gcm_128_decrypt: 'list[int]'
  reserved_at_7: 'list[int]'
  log_max_ipsec_offload: 'list[int]'
  reserved_at_10: 'list[int]'
  min_log_ipsec_full_replay_window: 'list[int]'
  max_log_ipsec_full_replay_window: 'list[int]'
  reserved_at_30: 'list[int]'
struct_mlx5_ifc_ipsec_cap_bits.register_fields([('ipsec_full_offload', (ctypes.c_ubyte * 1), 0), ('ipsec_crypto_offload', (ctypes.c_ubyte * 1), 1), ('ipsec_esn', (ctypes.c_ubyte * 1), 2), ('ipsec_crypto_esp_aes_gcm_256_encrypt', (ctypes.c_ubyte * 1), 3), ('ipsec_crypto_esp_aes_gcm_128_encrypt', (ctypes.c_ubyte * 1), 4), ('ipsec_crypto_esp_aes_gcm_256_decrypt', (ctypes.c_ubyte * 1), 5), ('ipsec_crypto_esp_aes_gcm_128_decrypt', (ctypes.c_ubyte * 1), 6), ('reserved_at_7', (ctypes.c_ubyte * 4), 7), ('log_max_ipsec_offload', (ctypes.c_ubyte * 5), 11), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('min_log_ipsec_full_replay_window', (ctypes.c_ubyte * 8), 32), ('max_log_ipsec_full_replay_window', (ctypes.c_ubyte * 8), 40), ('reserved_at_30', (ctypes.c_ubyte * 2000), 48)])
@c.record
class struct_mlx5_ifc_macsec_cap_bits(c.Struct):
  SIZE = 2048
  macsec_epn: 'list[int]'
  reserved_at_1: 'list[int]'
  macsec_crypto_esp_aes_gcm_256_encrypt: 'list[int]'
  macsec_crypto_esp_aes_gcm_128_encrypt: 'list[int]'
  macsec_crypto_esp_aes_gcm_256_decrypt: 'list[int]'
  macsec_crypto_esp_aes_gcm_128_decrypt: 'list[int]'
  reserved_at_7: 'list[int]'
  log_max_macsec_offload: 'list[int]'
  reserved_at_10: 'list[int]'
  min_log_macsec_full_replay_window: 'list[int]'
  max_log_macsec_full_replay_window: 'list[int]'
  reserved_at_30: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_macsec_cap_bits.register_fields([('macsec_epn', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 2), 1), ('macsec_crypto_esp_aes_gcm_256_encrypt', (ctypes.c_ubyte * 1), 3), ('macsec_crypto_esp_aes_gcm_128_encrypt', (ctypes.c_ubyte * 1), 4), ('macsec_crypto_esp_aes_gcm_256_decrypt', (ctypes.c_ubyte * 1), 5), ('macsec_crypto_esp_aes_gcm_128_decrypt', (ctypes.c_ubyte * 1), 6), ('reserved_at_7', (ctypes.c_ubyte * 4), 7), ('log_max_macsec_offload', (ctypes.c_ubyte * 5), 11), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('min_log_macsec_full_replay_window', (ctypes.c_ubyte * 8), 32), ('max_log_macsec_full_replay_window', (ctypes.c_ubyte * 8), 40), ('reserved_at_30', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 1984), 64)])
@c.record
class struct_mlx5_ifc_psp_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  psp_crypto_offload: 'list[int]'
  reserved_at_2: 'list[int]'
  psp_crypto_esp_aes_gcm_256_encrypt: 'list[int]'
  psp_crypto_esp_aes_gcm_128_encrypt: 'list[int]'
  psp_crypto_esp_aes_gcm_256_decrypt: 'list[int]'
  psp_crypto_esp_aes_gcm_128_decrypt: 'list[int]'
  reserved_at_7: 'list[int]'
  log_max_num_of_psp_spi: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_psp_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 1), 0), ('psp_crypto_offload', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 1), 2), ('psp_crypto_esp_aes_gcm_256_encrypt', (ctypes.c_ubyte * 1), 3), ('psp_crypto_esp_aes_gcm_128_encrypt', (ctypes.c_ubyte * 1), 4), ('psp_crypto_esp_aes_gcm_256_decrypt', (ctypes.c_ubyte * 1), 5), ('psp_crypto_esp_aes_gcm_128_decrypt', (ctypes.c_ubyte * 1), 6), ('reserved_at_7', (ctypes.c_ubyte * 4), 7), ('log_max_num_of_psp_spi', (ctypes.c_ubyte * 5), 11), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 2016), 32)])
_anonenum17: dict[int, str] = {(MLX5_WQ_TYPE_LINKED_LIST:=0): 'MLX5_WQ_TYPE_LINKED_LIST', (MLX5_WQ_TYPE_CYCLIC:=1): 'MLX5_WQ_TYPE_CYCLIC', (MLX5_WQ_TYPE_LINKED_LIST_STRIDING_RQ:=2): 'MLX5_WQ_TYPE_LINKED_LIST_STRIDING_RQ', (MLX5_WQ_TYPE_CYCLIC_STRIDING_RQ:=3): 'MLX5_WQ_TYPE_CYCLIC_STRIDING_RQ'}
_anonenum18: dict[int, str] = {(MLX5_WQ_END_PAD_MODE_NONE:=0): 'MLX5_WQ_END_PAD_MODE_NONE', (MLX5_WQ_END_PAD_MODE_ALIGN:=1): 'MLX5_WQ_END_PAD_MODE_ALIGN'}
_anonenum19: dict[int, str] = {(MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_8_GID_ENTRIES:=0): 'MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_8_GID_ENTRIES', (MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_16_GID_ENTRIES:=1): 'MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_16_GID_ENTRIES', (MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_32_GID_ENTRIES:=2): 'MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_32_GID_ENTRIES', (MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_64_GID_ENTRIES:=3): 'MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_64_GID_ENTRIES', (MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_128_GID_ENTRIES:=4): 'MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_128_GID_ENTRIES'}
_anonenum20: dict[int, str] = {(MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_128_ENTRIES:=0): 'MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_128_ENTRIES', (MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_256_ENTRIES:=1): 'MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_256_ENTRIES', (MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_512_ENTRIES:=2): 'MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_512_ENTRIES', (MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_1K_ENTRIES:=3): 'MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_1K_ENTRIES', (MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_2K_ENTRIES:=4): 'MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_2K_ENTRIES', (MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_4K_ENTRIES:=5): 'MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_4K_ENTRIES'}
_anonenum21: dict[int, str] = {(MLX5_CMD_HCA_CAP_PORT_TYPE_IB:=0): 'MLX5_CMD_HCA_CAP_PORT_TYPE_IB', (MLX5_CMD_HCA_CAP_PORT_TYPE_ETHERNET:=1): 'MLX5_CMD_HCA_CAP_PORT_TYPE_ETHERNET'}
_anonenum22: dict[int, str] = {(MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_DISABLED:=0): 'MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_DISABLED', (MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_INITIAL_STATE:=1): 'MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_INITIAL_STATE', (MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_ENABLED:=3): 'MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_ENABLED'}
_anonenum23: dict[int, str] = {(MLX5_CAP_PORT_TYPE_IB:=0): 'MLX5_CAP_PORT_TYPE_IB', (MLX5_CAP_PORT_TYPE_ETH:=1): 'MLX5_CAP_PORT_TYPE_ETH'}
_anonenum24: dict[int, str] = {(MLX5_CAP_UMR_FENCE_STRONG:=0): 'MLX5_CAP_UMR_FENCE_STRONG', (MLX5_CAP_UMR_FENCE_SMALL:=1): 'MLX5_CAP_UMR_FENCE_SMALL', (MLX5_CAP_UMR_FENCE_NONE:=2): 'MLX5_CAP_UMR_FENCE_NONE'}
_anonenum25: dict[int, str] = {(MLX5_FLEX_IPV4_OVER_VXLAN_ENABLED:=1): 'MLX5_FLEX_IPV4_OVER_VXLAN_ENABLED', (MLX5_FLEX_IPV6_OVER_VXLAN_ENABLED:=2): 'MLX5_FLEX_IPV6_OVER_VXLAN_ENABLED', (MLX5_FLEX_IPV6_OVER_IP_ENABLED:=4): 'MLX5_FLEX_IPV6_OVER_IP_ENABLED', (MLX5_FLEX_PARSER_GENEVE_ENABLED:=8): 'MLX5_FLEX_PARSER_GENEVE_ENABLED', (MLX5_FLEX_PARSER_MPLS_OVER_GRE_ENABLED:=16): 'MLX5_FLEX_PARSER_MPLS_OVER_GRE_ENABLED', (MLX5_FLEX_PARSER_MPLS_OVER_UDP_ENABLED:=32): 'MLX5_FLEX_PARSER_MPLS_OVER_UDP_ENABLED', (MLX5_FLEX_P_BIT_VXLAN_GPE_ENABLED:=64): 'MLX5_FLEX_P_BIT_VXLAN_GPE_ENABLED', (MLX5_FLEX_PARSER_VXLAN_GPE_ENABLED:=128): 'MLX5_FLEX_PARSER_VXLAN_GPE_ENABLED', (MLX5_FLEX_PARSER_ICMP_V4_ENABLED:=256): 'MLX5_FLEX_PARSER_ICMP_V4_ENABLED', (MLX5_FLEX_PARSER_ICMP_V6_ENABLED:=512): 'MLX5_FLEX_PARSER_ICMP_V6_ENABLED', (MLX5_FLEX_PARSER_GENEVE_TLV_OPTION_0_ENABLED:=1024): 'MLX5_FLEX_PARSER_GENEVE_TLV_OPTION_0_ENABLED', (MLX5_FLEX_PARSER_GTPU_ENABLED:=2048): 'MLX5_FLEX_PARSER_GTPU_ENABLED', (MLX5_FLEX_PARSER_GTPU_DW_2_ENABLED:=65536): 'MLX5_FLEX_PARSER_GTPU_DW_2_ENABLED', (MLX5_FLEX_PARSER_GTPU_FIRST_EXT_DW_0_ENABLED:=131072): 'MLX5_FLEX_PARSER_GTPU_FIRST_EXT_DW_0_ENABLED', (MLX5_FLEX_PARSER_GTPU_DW_0_ENABLED:=262144): 'MLX5_FLEX_PARSER_GTPU_DW_0_ENABLED', (MLX5_FLEX_PARSER_GTPU_TEID_ENABLED:=524288): 'MLX5_FLEX_PARSER_GTPU_TEID_ENABLED'}
_anonenum26: dict[int, str] = {(MLX5_UCTX_CAP_RAW_TX:=1): 'MLX5_UCTX_CAP_RAW_TX', (MLX5_UCTX_CAP_INTERNAL_DEV_RES:=2): 'MLX5_UCTX_CAP_INTERNAL_DEV_RES', (MLX5_UCTX_CAP_RDMA_CTRL:=8): 'MLX5_UCTX_CAP_RDMA_CTRL', (MLX5_UCTX_CAP_RDMA_CTRL_OTHER_VHCA:=16): 'MLX5_UCTX_CAP_RDMA_CTRL_OTHER_VHCA'}
enum_mlx5_fc_bulk_alloc_bitmask: dict[int, str] = {(MLX5_FC_BULK_128:=1): 'MLX5_FC_BULK_128', (MLX5_FC_BULK_256:=2): 'MLX5_FC_BULK_256', (MLX5_FC_BULK_512:=4): 'MLX5_FC_BULK_512', (MLX5_FC_BULK_1024:=8): 'MLX5_FC_BULK_1024', (MLX5_FC_BULK_2048:=16): 'MLX5_FC_BULK_2048', (MLX5_FC_BULK_4096:=32): 'MLX5_FC_BULK_4096', (MLX5_FC_BULK_8192:=64): 'MLX5_FC_BULK_8192', (MLX5_FC_BULK_16384:=128): 'MLX5_FC_BULK_16384'}
_anonenum27: dict[int, str] = {(MLX5_STEERING_FORMAT_CONNECTX_5:=0): 'MLX5_STEERING_FORMAT_CONNECTX_5', (MLX5_STEERING_FORMAT_CONNECTX_6DX:=1): 'MLX5_STEERING_FORMAT_CONNECTX_6DX', (MLX5_STEERING_FORMAT_CONNECTX_7:=2): 'MLX5_STEERING_FORMAT_CONNECTX_7', (MLX5_STEERING_FORMAT_CONNECTX_8:=3): 'MLX5_STEERING_FORMAT_CONNECTX_8'}
@c.record
class struct_mlx5_ifc_cmd_hca_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  page_request_disable: 'list[int]'
  abs_native_port_num: 'list[int]'
  reserved_at_8: 'list[int]'
  shared_object_to_user_object_allowed: 'list[int]'
  reserved_at_13: 'list[int]'
  vhca_resource_manager: 'list[int]'
  hca_cap_2: 'list[int]'
  create_lag_when_not_master_up: 'list[int]'
  dtor: 'list[int]'
  event_on_vhca_state_teardown_request: 'list[int]'
  event_on_vhca_state_in_use: 'list[int]'
  event_on_vhca_state_active: 'list[int]'
  event_on_vhca_state_allocated: 'list[int]'
  event_on_vhca_state_invalid: 'list[int]'
  reserved_at_28: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_40: 'list[int]'
  log_max_srq_sz: 'list[int]'
  log_max_qp_sz: 'list[int]'
  event_cap: 'list[int]'
  reserved_at_91: 'list[int]'
  isolate_vl_tc_new: 'list[int]'
  reserved_at_94: 'list[int]'
  prio_tag_required: 'list[int]'
  reserved_at_99: 'list[int]'
  log_max_qp: 'list[int]'
  reserved_at_a0: 'list[int]'
  ece_support: 'list[int]'
  reserved_at_a4: 'list[int]'
  reg_c_preserve: 'list[int]'
  reserved_at_aa: 'list[int]'
  log_max_srq: 'list[int]'
  reserved_at_b0: 'list[int]'
  uplink_follow: 'list[int]'
  ts_cqe_to_dest_cqn: 'list[int]'
  reserved_at_b3: 'list[int]'
  go_back_n: 'list[int]'
  reserved_at_ba: 'list[int]'
  max_sgl_for_optimized_performance: 'list[int]'
  log_max_cq_sz: 'list[int]'
  relaxed_ordering_write_umr: 'list[int]'
  relaxed_ordering_read_umr: 'list[int]'
  reserved_at_d2: 'list[int]'
  virtio_net_device_emualtion_manager: 'list[int]'
  virtio_blk_device_emualtion_manager: 'list[int]'
  log_max_cq: 'list[int]'
  log_max_eq_sz: 'list[int]'
  relaxed_ordering_write: 'list[int]'
  relaxed_ordering_read_pci_enabled: 'list[int]'
  log_max_mkey: 'list[int]'
  reserved_at_f0: 'list[int]'
  terminate_scatter_list_mkey: 'list[int]'
  repeated_mkey: 'list[int]'
  dump_fill_mkey: 'list[int]'
  reserved_at_f9: 'list[int]'
  fast_teardown: 'list[int]'
  log_max_eq: 'list[int]'
  max_indirection: 'list[int]'
  fixed_buffer_size: 'list[int]'
  log_max_mrw_sz: 'list[int]'
  force_teardown: 'list[int]'
  reserved_at_111: 'list[int]'
  log_max_bsf_list_size: 'list[int]'
  umr_extended_translation_offset: 'list[int]'
  null_mkey: 'list[int]'
  log_max_klm_list_size: 'list[int]'
  reserved_at_120: 'list[int]'
  qpc_extension: 'list[int]'
  reserved_at_123: 'list[int]'
  log_max_ra_req_dc: 'list[int]'
  reserved_at_130: 'list[int]'
  eth_wqe_too_small: 'list[int]'
  reserved_at_133: 'list[int]'
  vnic_env_cq_overrun: 'list[int]'
  log_max_ra_res_dc: 'list[int]'
  reserved_at_140: 'list[int]'
  release_all_pages: 'list[int]'
  must_not_use: 'list[int]'
  reserved_at_147: 'list[int]'
  roce_accl: 'list[int]'
  log_max_ra_req_qp: 'list[int]'
  reserved_at_150: 'list[int]'
  log_max_ra_res_qp: 'list[int]'
  end_pad: 'list[int]'
  cc_query_allowed: 'list[int]'
  cc_modify_allowed: 'list[int]'
  start_pad: 'list[int]'
  cache_line_128byte: 'list[int]'
  reserved_at_165: 'list[int]'
  rts2rts_qp_counters_set_id: 'list[int]'
  reserved_at_16a: 'list[int]'
  vnic_env_int_rq_oob: 'list[int]'
  sbcam_reg: 'list[int]'
  reserved_at_16e: 'list[int]'
  qcam_reg: 'list[int]'
  gid_table_size: 'list[int]'
  out_of_seq_cnt: 'list[int]'
  vport_counters: 'list[int]'
  retransmission_q_counters: 'list[int]'
  debug: 'list[int]'
  modify_rq_counter_set_id: 'list[int]'
  rq_delay_drop: 'list[int]'
  max_qp_cnt: 'list[int]'
  pkey_table_size: 'list[int]'
  vport_group_manager: 'list[int]'
  vhca_group_manager: 'list[int]'
  ib_virt: 'list[int]'
  eth_virt: 'list[int]'
  vnic_env_queue_counters: 'list[int]'
  ets: 'list[int]'
  nic_flow_table: 'list[int]'
  eswitch_manager: 'list[int]'
  device_memory: 'list[int]'
  mcam_reg: 'list[int]'
  pcam_reg: 'list[int]'
  local_ca_ack_delay: 'list[int]'
  port_module_event: 'list[int]'
  enhanced_error_q_counters: 'list[int]'
  ports_check: 'list[int]'
  reserved_at_1b3: 'list[int]'
  disable_link_up: 'list[int]'
  beacon_led: 'list[int]'
  port_type: 'list[int]'
  num_ports: 'list[int]'
  reserved_at_1c0: 'list[int]'
  pps: 'list[int]'
  pps_modify: 'list[int]'
  log_max_msg: 'list[int]'
  reserved_at_1c8: 'list[int]'
  max_tc: 'list[int]'
  temp_warn_event: 'list[int]'
  dcbx: 'list[int]'
  general_notification_event: 'list[int]'
  reserved_at_1d3: 'list[int]'
  fpga: 'list[int]'
  rol_s: 'list[int]'
  rol_g: 'list[int]'
  reserved_at_1d8: 'list[int]'
  wol_s: 'list[int]'
  wol_g: 'list[int]'
  wol_a: 'list[int]'
  wol_b: 'list[int]'
  wol_m: 'list[int]'
  wol_u: 'list[int]'
  wol_p: 'list[int]'
  stat_rate_support: 'list[int]'
  reserved_at_1f0: 'list[int]'
  pci_sync_for_fw_update_event: 'list[int]'
  reserved_at_1f2: 'list[int]'
  init2_lag_tx_port_affinity: 'list[int]'
  reserved_at_1fa: 'list[int]'
  wqe_based_flow_table_update_cap: 'list[int]'
  cqe_version: 'list[int]'
  compact_address_vector: 'list[int]'
  striding_rq: 'list[int]'
  reserved_at_202: 'list[int]'
  ipoib_enhanced_offloads: 'list[int]'
  ipoib_basic_offloads: 'list[int]'
  reserved_at_205: 'list[int]'
  repeated_block_disabled: 'list[int]'
  umr_modify_entity_size_disabled: 'list[int]'
  umr_modify_atomic_disabled: 'list[int]'
  umr_indirect_mkey_disabled: 'list[int]'
  umr_fence: 'list[int]'
  dc_req_scat_data_cqe: 'list[int]'
  reserved_at_20d: 'list[int]'
  drain_sigerr: 'list[int]'
  cmdif_checksum: 'list[int]'
  sigerr_cqe: 'list[int]'
  reserved_at_213: 'list[int]'
  wq_signature: 'list[int]'
  sctr_data_cqe: 'list[int]'
  reserved_at_216: 'list[int]'
  sho: 'list[int]'
  tph: 'list[int]'
  rf: 'list[int]'
  dct: 'list[int]'
  qos: 'list[int]'
  eth_net_offloads: 'list[int]'
  roce: 'list[int]'
  atomic: 'list[int]'
  reserved_at_21f: 'list[int]'
  cq_oi: 'list[int]'
  cq_resize: 'list[int]'
  cq_moderation: 'list[int]'
  cq_period_mode_modify: 'list[int]'
  reserved_at_224: 'list[int]'
  cq_eq_remap: 'list[int]'
  pg: 'list[int]'
  block_lb_mc: 'list[int]'
  reserved_at_229: 'list[int]'
  scqe_break_moderation: 'list[int]'
  cq_period_start_from_cqe: 'list[int]'
  cd: 'list[int]'
  reserved_at_22d: 'list[int]'
  apm: 'list[int]'
  vector_calc: 'list[int]'
  umr_ptr_rlky: 'list[int]'
  imaicl: 'list[int]'
  qp_packet_based: 'list[int]'
  reserved_at_233: 'list[int]'
  qkv: 'list[int]'
  pkv: 'list[int]'
  set_deth_sqpn: 'list[int]'
  reserved_at_239: 'list[int]'
  xrc: 'list[int]'
  ud: 'list[int]'
  uc: 'list[int]'
  rc: 'list[int]'
  uar_4k: 'list[int]'
  reserved_at_241: 'list[int]'
  fl_rc_qp_when_roce_disabled: 'list[int]'
  regexp_params: 'list[int]'
  uar_sz: 'list[int]'
  port_selection_cap: 'list[int]'
  nic_cap_reg: 'list[int]'
  umem_uid_0: 'list[int]'
  reserved_at_253: 'list[int]'
  log_pg_sz: 'list[int]'
  bf: 'list[int]'
  driver_version: 'list[int]'
  pad_tx_eth_packet: 'list[int]'
  reserved_at_263: 'list[int]'
  mkey_by_name: 'list[int]'
  reserved_at_267: 'list[int]'
  log_bf_reg_size: 'list[int]'
  disciplined_fr_counter: 'list[int]'
  reserved_at_271: 'list[int]'
  qp_error_syndrome: 'list[int]'
  reserved_at_274: 'list[int]'
  lag_dct: 'list[int]'
  lag_tx_port_affinity: 'list[int]'
  lag_native_fdb_selection: 'list[int]'
  reserved_at_27a: 'list[int]'
  lag_master: 'list[int]'
  num_lag_ports: 'list[int]'
  reserved_at_280: 'list[int]'
  max_wqe_sz_sq: 'list[int]'
  reserved_at_2a0: 'list[int]'
  mkey_pcie_tph: 'list[int]'
  reserved_at_2a8: 'list[int]'
  tis_tir_td_order: 'list[int]'
  psp: 'list[int]'
  shampo: 'list[int]'
  reserved_at_2ac: 'list[int]'
  max_wqe_sz_rq: 'list[int]'
  max_flow_counter_31_16: 'list[int]'
  max_wqe_sz_sq_dc: 'list[int]'
  reserved_at_2e0: 'list[int]'
  max_qp_mcg: 'list[int]'
  reserved_at_300: 'list[int]'
  flow_counter_bulk_alloc: 'list[int]'
  log_max_mcg: 'list[int]'
  reserved_at_320: 'list[int]'
  log_max_transport_domain: 'list[int]'
  reserved_at_328: 'list[int]'
  relaxed_ordering_read: 'list[int]'
  log_max_pd: 'list[int]'
  dp_ordering_ooo_all_ud: 'list[int]'
  dp_ordering_ooo_all_uc: 'list[int]'
  dp_ordering_ooo_all_xrc: 'list[int]'
  dp_ordering_ooo_all_dc: 'list[int]'
  dp_ordering_ooo_all_rc: 'list[int]'
  pcie_reset_using_hotreset_method: 'list[int]'
  pci_sync_for_fw_update_with_driver_unload: 'list[int]'
  vnic_env_cnt_steering_fail: 'list[int]'
  vport_counter_local_loopback: 'list[int]'
  q_counter_aggregation: 'list[int]'
  q_counter_other_vport: 'list[int]'
  log_max_xrcd: 'list[int]'
  nic_receive_steering_discard: 'list[int]'
  receive_discard_vport_down: 'list[int]'
  transmit_discard_vport_down: 'list[int]'
  eq_overrun_count: 'list[int]'
  reserved_at_344: 'list[int]'
  invalid_command_count: 'list[int]'
  quota_exceeded_count: 'list[int]'
  reserved_at_347: 'list[int]'
  log_max_flow_counter_bulk: 'list[int]'
  max_flow_counter_15_0: 'list[int]'
  reserved_at_360: 'list[int]'
  log_max_rq: 'list[int]'
  reserved_at_368: 'list[int]'
  log_max_sq: 'list[int]'
  reserved_at_370: 'list[int]'
  log_max_tir: 'list[int]'
  reserved_at_378: 'list[int]'
  log_max_tis: 'list[int]'
  basic_cyclic_rcv_wqe: 'list[int]'
  reserved_at_381: 'list[int]'
  log_max_rmp: 'list[int]'
  reserved_at_388: 'list[int]'
  log_max_rqt: 'list[int]'
  reserved_at_390: 'list[int]'
  log_max_rqt_size: 'list[int]'
  reserved_at_398: 'list[int]'
  vnic_env_cnt_bar_uar_access: 'list[int]'
  vnic_env_cnt_odp_page_fault: 'list[int]'
  log_max_tis_per_sq: 'list[int]'
  ext_stride_num_range: 'list[int]'
  roce_rw_supported: 'list[int]'
  log_max_current_uc_list_wr_supported: 'list[int]'
  log_max_stride_sz_rq: 'list[int]'
  reserved_at_3a8: 'list[int]'
  log_min_stride_sz_rq: 'list[int]'
  reserved_at_3b0: 'list[int]'
  log_max_stride_sz_sq: 'list[int]'
  reserved_at_3b8: 'list[int]'
  log_min_stride_sz_sq: 'list[int]'
  hairpin: 'list[int]'
  reserved_at_3c1: 'list[int]'
  log_max_hairpin_queues: 'list[int]'
  reserved_at_3c8: 'list[int]'
  log_max_hairpin_wq_data_sz: 'list[int]'
  reserved_at_3d0: 'list[int]'
  log_max_hairpin_num_packets: 'list[int]'
  reserved_at_3d8: 'list[int]'
  log_max_wq_sz: 'list[int]'
  nic_vport_change_event: 'list[int]'
  disable_local_lb_uc: 'list[int]'
  disable_local_lb_mc: 'list[int]'
  log_min_hairpin_wq_data_sz: 'list[int]'
  reserved_at_3e8: 'list[int]'
  silent_mode: 'list[int]'
  vhca_state: 'list[int]'
  log_max_vlan_list: 'list[int]'
  reserved_at_3f0: 'list[int]'
  log_max_current_mc_list: 'list[int]'
  reserved_at_3f8: 'list[int]'
  log_max_current_uc_list: 'list[int]'
  general_obj_types: 'list[int]'
  sq_ts_format: 'list[int]'
  rq_ts_format: 'list[int]'
  steering_format_version: 'list[int]'
  create_qp_start_hint: 'list[int]'
  reserved_at_460: 'list[int]'
  ats: 'list[int]'
  cross_vhca_rqt: 'list[int]'
  log_max_uctx: 'list[int]'
  reserved_at_468: 'list[int]'
  crypto: 'list[int]'
  ipsec_offload: 'list[int]'
  log_max_umem: 'list[int]'
  max_num_eqs: 'list[int]'
  reserved_at_480: 'list[int]'
  tls_tx: 'list[int]'
  tls_rx: 'list[int]'
  log_max_l2_table: 'list[int]'
  reserved_at_488: 'list[int]'
  log_uar_page_sz: 'list[int]'
  reserved_at_4a0: 'list[int]'
  device_frequency_mhz: 'list[int]'
  device_frequency_khz: 'list[int]'
  reserved_at_500: 'list[int]'
  num_of_uars_per_page: 'list[int]'
  flex_parser_protocols: 'list[int]'
  max_geneve_tlv_options: 'list[int]'
  reserved_at_568: 'list[int]'
  max_geneve_tlv_option_data_len: 'list[int]'
  reserved_at_570: 'list[int]'
  adv_rdma: 'list[int]'
  reserved_at_572: 'list[int]'
  adv_virtualization: 'list[int]'
  reserved_at_57a: 'list[int]'
  reserved_at_580: 'list[int]'
  log_max_dci_stream_channels: 'list[int]'
  reserved_at_590: 'list[int]'
  log_max_dci_errored_streams: 'list[int]'
  reserved_at_598: 'list[int]'
  reserved_at_5a0: 'list[int]'
  enhanced_cqe_compression: 'list[int]'
  reserved_at_5b1: 'list[int]'
  crossing_vhca_mkey: 'list[int]'
  log_max_dek: 'list[int]'
  reserved_at_5b8: 'list[int]'
  mini_cqe_resp_stride_index: 'list[int]'
  cqe_128_always: 'list[int]'
  cqe_compression_128: 'list[int]'
  cqe_compression: 'list[int]'
  cqe_compression_timeout: 'list[int]'
  cqe_compression_max_num: 'list[int]'
  reserved_at_5e0: 'list[int]'
  flex_parser_id_gtpu_dw_0: 'list[int]'
  reserved_at_5ec: 'list[int]'
  tag_matching: 'list[int]'
  rndv_offload_rc: 'list[int]'
  rndv_offload_dc: 'list[int]'
  log_tag_matching_list_sz: 'list[int]'
  reserved_at_5f8: 'list[int]'
  log_max_xrq: 'list[int]'
  affiliate_nic_vport_criteria: 'list[int]'
  native_port_num: 'list[int]'
  num_vhca_ports: 'list[int]'
  flex_parser_id_gtpu_teid: 'list[int]'
  reserved_at_61c: 'list[int]'
  sw_owner_id: 'list[int]'
  reserved_at_61f: 'list[int]'
  max_num_of_monitor_counters: 'list[int]'
  num_ppcnt_monitor_counters: 'list[int]'
  max_num_sf: 'list[int]'
  num_q_monitor_counters: 'list[int]'
  reserved_at_660: 'list[int]'
  sf: 'list[int]'
  sf_set_partition: 'list[int]'
  reserved_at_682: 'list[int]'
  log_max_sf: 'list[int]'
  apu: 'list[int]'
  reserved_at_689: 'list[int]'
  migration: 'list[int]'
  reserved_at_68e: 'list[int]'
  log_min_sf_size: 'list[int]'
  max_num_sf_partitions: 'list[int]'
  uctx_cap: 'list[int]'
  reserved_at_6c0: 'list[int]'
  flex_parser_id_geneve_tlv_option_0: 'list[int]'
  flex_parser_id_icmp_dw1: 'list[int]'
  flex_parser_id_icmp_dw0: 'list[int]'
  flex_parser_id_icmpv6_dw1: 'list[int]'
  flex_parser_id_icmpv6_dw0: 'list[int]'
  flex_parser_id_outer_first_mpls_over_gre: 'list[int]'
  flex_parser_id_outer_first_mpls_over_udp_label: 'list[int]'
  max_num_match_definer: 'list[int]'
  sf_base_id: 'list[int]'
  flex_parser_id_gtpu_dw_2: 'list[int]'
  flex_parser_id_gtpu_first_ext_dw_0: 'list[int]'
  num_total_dynamic_vf_msix: 'list[int]'
  reserved_at_720: 'list[int]'
  dynamic_msix_table_size: 'list[int]'
  reserved_at_740: 'list[int]'
  min_dynamic_vf_msix_table_size: 'list[int]'
  reserved_at_750: 'list[int]'
  data_direct: 'list[int]'
  reserved_at_753: 'list[int]'
  max_dynamic_vf_msix_table_size: 'list[int]'
  reserved_at_760: 'list[int]'
  log_max_num_header_modify_argument: 'list[int]'
  log_header_modify_argument_granularity_offset: 'list[int]'
  log_header_modify_argument_granularity: 'list[int]'
  reserved_at_770: 'list[int]'
  log_header_modify_argument_max_alloc: 'list[int]'
  reserved_at_778: 'list[int]'
  vhca_tunnel_commands: 'list[int]'
  match_definer_format_supported: 'list[int]'
struct_mlx5_ifc_cmd_hca_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 6), 0), ('page_request_disable', (ctypes.c_ubyte * 1), 6), ('abs_native_port_num', (ctypes.c_ubyte * 1), 7), ('reserved_at_8', (ctypes.c_ubyte * 8), 8), ('shared_object_to_user_object_allowed', (ctypes.c_ubyte * 1), 16), ('reserved_at_13', (ctypes.c_ubyte * 14), 17), ('vhca_resource_manager', (ctypes.c_ubyte * 1), 31), ('hca_cap_2', (ctypes.c_ubyte * 1), 32), ('create_lag_when_not_master_up', (ctypes.c_ubyte * 1), 33), ('dtor', (ctypes.c_ubyte * 1), 34), ('event_on_vhca_state_teardown_request', (ctypes.c_ubyte * 1), 35), ('event_on_vhca_state_in_use', (ctypes.c_ubyte * 1), 36), ('event_on_vhca_state_active', (ctypes.c_ubyte * 1), 37), ('event_on_vhca_state_allocated', (ctypes.c_ubyte * 1), 38), ('event_on_vhca_state_invalid', (ctypes.c_ubyte * 1), 39), ('reserved_at_28', (ctypes.c_ubyte * 8), 40), ('vhca_id', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('log_max_srq_sz', (ctypes.c_ubyte * 8), 128), ('log_max_qp_sz', (ctypes.c_ubyte * 8), 136), ('event_cap', (ctypes.c_ubyte * 1), 144), ('reserved_at_91', (ctypes.c_ubyte * 2), 145), ('isolate_vl_tc_new', (ctypes.c_ubyte * 1), 147), ('reserved_at_94', (ctypes.c_ubyte * 4), 148), ('prio_tag_required', (ctypes.c_ubyte * 1), 152), ('reserved_at_99', (ctypes.c_ubyte * 2), 153), ('log_max_qp', (ctypes.c_ubyte * 5), 155), ('reserved_at_a0', (ctypes.c_ubyte * 3), 160), ('ece_support', (ctypes.c_ubyte * 1), 163), ('reserved_at_a4', (ctypes.c_ubyte * 5), 164), ('reg_c_preserve', (ctypes.c_ubyte * 1), 169), ('reserved_at_aa', (ctypes.c_ubyte * 1), 170), ('log_max_srq', (ctypes.c_ubyte * 5), 171), ('reserved_at_b0', (ctypes.c_ubyte * 1), 176), ('uplink_follow', (ctypes.c_ubyte * 1), 177), ('ts_cqe_to_dest_cqn', (ctypes.c_ubyte * 1), 178), ('reserved_at_b3', (ctypes.c_ubyte * 6), 179), ('go_back_n', (ctypes.c_ubyte * 1), 185), ('reserved_at_ba', (ctypes.c_ubyte * 6), 186), ('max_sgl_for_optimized_performance', (ctypes.c_ubyte * 8), 192), ('log_max_cq_sz', (ctypes.c_ubyte * 8), 200), ('relaxed_ordering_write_umr', (ctypes.c_ubyte * 1), 208), ('relaxed_ordering_read_umr', (ctypes.c_ubyte * 1), 209), ('reserved_at_d2', (ctypes.c_ubyte * 7), 210), ('virtio_net_device_emualtion_manager', (ctypes.c_ubyte * 1), 217), ('virtio_blk_device_emualtion_manager', (ctypes.c_ubyte * 1), 218), ('log_max_cq', (ctypes.c_ubyte * 5), 219), ('log_max_eq_sz', (ctypes.c_ubyte * 8), 224), ('relaxed_ordering_write', (ctypes.c_ubyte * 1), 232), ('relaxed_ordering_read_pci_enabled', (ctypes.c_ubyte * 1), 233), ('log_max_mkey', (ctypes.c_ubyte * 6), 234), ('reserved_at_f0', (ctypes.c_ubyte * 6), 240), ('terminate_scatter_list_mkey', (ctypes.c_ubyte * 1), 246), ('repeated_mkey', (ctypes.c_ubyte * 1), 247), ('dump_fill_mkey', (ctypes.c_ubyte * 1), 248), ('reserved_at_f9', (ctypes.c_ubyte * 2), 249), ('fast_teardown', (ctypes.c_ubyte * 1), 251), ('log_max_eq', (ctypes.c_ubyte * 4), 252), ('max_indirection', (ctypes.c_ubyte * 8), 256), ('fixed_buffer_size', (ctypes.c_ubyte * 1), 264), ('log_max_mrw_sz', (ctypes.c_ubyte * 7), 265), ('force_teardown', (ctypes.c_ubyte * 1), 272), ('reserved_at_111', (ctypes.c_ubyte * 1), 273), ('log_max_bsf_list_size', (ctypes.c_ubyte * 6), 274), ('umr_extended_translation_offset', (ctypes.c_ubyte * 1), 280), ('null_mkey', (ctypes.c_ubyte * 1), 281), ('log_max_klm_list_size', (ctypes.c_ubyte * 6), 282), ('reserved_at_120', (ctypes.c_ubyte * 2), 288), ('qpc_extension', (ctypes.c_ubyte * 1), 290), ('reserved_at_123', (ctypes.c_ubyte * 7), 291), ('log_max_ra_req_dc', (ctypes.c_ubyte * 6), 298), ('reserved_at_130', (ctypes.c_ubyte * 2), 304), ('eth_wqe_too_small', (ctypes.c_ubyte * 1), 306), ('reserved_at_133', (ctypes.c_ubyte * 6), 307), ('vnic_env_cq_overrun', (ctypes.c_ubyte * 1), 313), ('log_max_ra_res_dc', (ctypes.c_ubyte * 6), 314), ('reserved_at_140', (ctypes.c_ubyte * 5), 320), ('release_all_pages', (ctypes.c_ubyte * 1), 325), ('must_not_use', (ctypes.c_ubyte * 1), 326), ('reserved_at_147', (ctypes.c_ubyte * 2), 327), ('roce_accl', (ctypes.c_ubyte * 1), 329), ('log_max_ra_req_qp', (ctypes.c_ubyte * 6), 330), ('reserved_at_150', (ctypes.c_ubyte * 10), 336), ('log_max_ra_res_qp', (ctypes.c_ubyte * 6), 346), ('end_pad', (ctypes.c_ubyte * 1), 352), ('cc_query_allowed', (ctypes.c_ubyte * 1), 353), ('cc_modify_allowed', (ctypes.c_ubyte * 1), 354), ('start_pad', (ctypes.c_ubyte * 1), 355), ('cache_line_128byte', (ctypes.c_ubyte * 1), 356), ('reserved_at_165', (ctypes.c_ubyte * 4), 357), ('rts2rts_qp_counters_set_id', (ctypes.c_ubyte * 1), 361), ('reserved_at_16a', (ctypes.c_ubyte * 2), 362), ('vnic_env_int_rq_oob', (ctypes.c_ubyte * 1), 364), ('sbcam_reg', (ctypes.c_ubyte * 1), 365), ('reserved_at_16e', (ctypes.c_ubyte * 1), 366), ('qcam_reg', (ctypes.c_ubyte * 1), 367), ('gid_table_size', (ctypes.c_ubyte * 16), 368), ('out_of_seq_cnt', (ctypes.c_ubyte * 1), 384), ('vport_counters', (ctypes.c_ubyte * 1), 385), ('retransmission_q_counters', (ctypes.c_ubyte * 1), 386), ('debug', (ctypes.c_ubyte * 1), 387), ('modify_rq_counter_set_id', (ctypes.c_ubyte * 1), 388), ('rq_delay_drop', (ctypes.c_ubyte * 1), 389), ('max_qp_cnt', (ctypes.c_ubyte * 10), 390), ('pkey_table_size', (ctypes.c_ubyte * 16), 400), ('vport_group_manager', (ctypes.c_ubyte * 1), 416), ('vhca_group_manager', (ctypes.c_ubyte * 1), 417), ('ib_virt', (ctypes.c_ubyte * 1), 418), ('eth_virt', (ctypes.c_ubyte * 1), 419), ('vnic_env_queue_counters', (ctypes.c_ubyte * 1), 420), ('ets', (ctypes.c_ubyte * 1), 421), ('nic_flow_table', (ctypes.c_ubyte * 1), 422), ('eswitch_manager', (ctypes.c_ubyte * 1), 423), ('device_memory', (ctypes.c_ubyte * 1), 424), ('mcam_reg', (ctypes.c_ubyte * 1), 425), ('pcam_reg', (ctypes.c_ubyte * 1), 426), ('local_ca_ack_delay', (ctypes.c_ubyte * 5), 427), ('port_module_event', (ctypes.c_ubyte * 1), 432), ('enhanced_error_q_counters', (ctypes.c_ubyte * 1), 433), ('ports_check', (ctypes.c_ubyte * 1), 434), ('reserved_at_1b3', (ctypes.c_ubyte * 1), 435), ('disable_link_up', (ctypes.c_ubyte * 1), 436), ('beacon_led', (ctypes.c_ubyte * 1), 437), ('port_type', (ctypes.c_ubyte * 2), 438), ('num_ports', (ctypes.c_ubyte * 8), 440), ('reserved_at_1c0', (ctypes.c_ubyte * 1), 448), ('pps', (ctypes.c_ubyte * 1), 449), ('pps_modify', (ctypes.c_ubyte * 1), 450), ('log_max_msg', (ctypes.c_ubyte * 5), 451), ('reserved_at_1c8', (ctypes.c_ubyte * 4), 456), ('max_tc', (ctypes.c_ubyte * 4), 460), ('temp_warn_event', (ctypes.c_ubyte * 1), 464), ('dcbx', (ctypes.c_ubyte * 1), 465), ('general_notification_event', (ctypes.c_ubyte * 1), 466), ('reserved_at_1d3', (ctypes.c_ubyte * 2), 467), ('fpga', (ctypes.c_ubyte * 1), 469), ('rol_s', (ctypes.c_ubyte * 1), 470), ('rol_g', (ctypes.c_ubyte * 1), 471), ('reserved_at_1d8', (ctypes.c_ubyte * 1), 472), ('wol_s', (ctypes.c_ubyte * 1), 473), ('wol_g', (ctypes.c_ubyte * 1), 474), ('wol_a', (ctypes.c_ubyte * 1), 475), ('wol_b', (ctypes.c_ubyte * 1), 476), ('wol_m', (ctypes.c_ubyte * 1), 477), ('wol_u', (ctypes.c_ubyte * 1), 478), ('wol_p', (ctypes.c_ubyte * 1), 479), ('stat_rate_support', (ctypes.c_ubyte * 16), 480), ('reserved_at_1f0', (ctypes.c_ubyte * 1), 496), ('pci_sync_for_fw_update_event', (ctypes.c_ubyte * 1), 497), ('reserved_at_1f2', (ctypes.c_ubyte * 6), 498), ('init2_lag_tx_port_affinity', (ctypes.c_ubyte * 1), 504), ('reserved_at_1fa', (ctypes.c_ubyte * 2), 505), ('wqe_based_flow_table_update_cap', (ctypes.c_ubyte * 1), 507), ('cqe_version', (ctypes.c_ubyte * 4), 508), ('compact_address_vector', (ctypes.c_ubyte * 1), 512), ('striding_rq', (ctypes.c_ubyte * 1), 513), ('reserved_at_202', (ctypes.c_ubyte * 1), 514), ('ipoib_enhanced_offloads', (ctypes.c_ubyte * 1), 515), ('ipoib_basic_offloads', (ctypes.c_ubyte * 1), 516), ('reserved_at_205', (ctypes.c_ubyte * 1), 517), ('repeated_block_disabled', (ctypes.c_ubyte * 1), 518), ('umr_modify_entity_size_disabled', (ctypes.c_ubyte * 1), 519), ('umr_modify_atomic_disabled', (ctypes.c_ubyte * 1), 520), ('umr_indirect_mkey_disabled', (ctypes.c_ubyte * 1), 521), ('umr_fence', (ctypes.c_ubyte * 2), 522), ('dc_req_scat_data_cqe', (ctypes.c_ubyte * 1), 524), ('reserved_at_20d', (ctypes.c_ubyte * 2), 525), ('drain_sigerr', (ctypes.c_ubyte * 1), 527), ('cmdif_checksum', (ctypes.c_ubyte * 2), 528), ('sigerr_cqe', (ctypes.c_ubyte * 1), 530), ('reserved_at_213', (ctypes.c_ubyte * 1), 531), ('wq_signature', (ctypes.c_ubyte * 1), 532), ('sctr_data_cqe', (ctypes.c_ubyte * 1), 533), ('reserved_at_216', (ctypes.c_ubyte * 1), 534), ('sho', (ctypes.c_ubyte * 1), 535), ('tph', (ctypes.c_ubyte * 1), 536), ('rf', (ctypes.c_ubyte * 1), 537), ('dct', (ctypes.c_ubyte * 1), 538), ('qos', (ctypes.c_ubyte * 1), 539), ('eth_net_offloads', (ctypes.c_ubyte * 1), 540), ('roce', (ctypes.c_ubyte * 1), 541), ('atomic', (ctypes.c_ubyte * 1), 542), ('reserved_at_21f', (ctypes.c_ubyte * 1), 543), ('cq_oi', (ctypes.c_ubyte * 1), 544), ('cq_resize', (ctypes.c_ubyte * 1), 545), ('cq_moderation', (ctypes.c_ubyte * 1), 546), ('cq_period_mode_modify', (ctypes.c_ubyte * 1), 547), ('reserved_at_224', (ctypes.c_ubyte * 2), 548), ('cq_eq_remap', (ctypes.c_ubyte * 1), 550), ('pg', (ctypes.c_ubyte * 1), 551), ('block_lb_mc', (ctypes.c_ubyte * 1), 552), ('reserved_at_229', (ctypes.c_ubyte * 1), 553), ('scqe_break_moderation', (ctypes.c_ubyte * 1), 554), ('cq_period_start_from_cqe', (ctypes.c_ubyte * 1), 555), ('cd', (ctypes.c_ubyte * 1), 556), ('reserved_at_22d', (ctypes.c_ubyte * 1), 557), ('apm', (ctypes.c_ubyte * 1), 558), ('vector_calc', (ctypes.c_ubyte * 1), 559), ('umr_ptr_rlky', (ctypes.c_ubyte * 1), 560), ('imaicl', (ctypes.c_ubyte * 1), 561), ('qp_packet_based', (ctypes.c_ubyte * 1), 562), ('reserved_at_233', (ctypes.c_ubyte * 3), 563), ('qkv', (ctypes.c_ubyte * 1), 566), ('pkv', (ctypes.c_ubyte * 1), 567), ('set_deth_sqpn', (ctypes.c_ubyte * 1), 568), ('reserved_at_239', (ctypes.c_ubyte * 3), 569), ('xrc', (ctypes.c_ubyte * 1), 572), ('ud', (ctypes.c_ubyte * 1), 573), ('uc', (ctypes.c_ubyte * 1), 574), ('rc', (ctypes.c_ubyte * 1), 575), ('uar_4k', (ctypes.c_ubyte * 1), 576), ('reserved_at_241', (ctypes.c_ubyte * 7), 577), ('fl_rc_qp_when_roce_disabled', (ctypes.c_ubyte * 1), 584), ('regexp_params', (ctypes.c_ubyte * 1), 585), ('uar_sz', (ctypes.c_ubyte * 6), 586), ('port_selection_cap', (ctypes.c_ubyte * 1), 592), ('nic_cap_reg', (ctypes.c_ubyte * 1), 593), ('umem_uid_0', (ctypes.c_ubyte * 1), 594), ('reserved_at_253', (ctypes.c_ubyte * 5), 595), ('log_pg_sz', (ctypes.c_ubyte * 8), 600), ('bf', (ctypes.c_ubyte * 1), 608), ('driver_version', (ctypes.c_ubyte * 1), 609), ('pad_tx_eth_packet', (ctypes.c_ubyte * 1), 610), ('reserved_at_263', (ctypes.c_ubyte * 3), 611), ('mkey_by_name', (ctypes.c_ubyte * 1), 614), ('reserved_at_267', (ctypes.c_ubyte * 4), 615), ('log_bf_reg_size', (ctypes.c_ubyte * 5), 619), ('disciplined_fr_counter', (ctypes.c_ubyte * 1), 624), ('reserved_at_271', (ctypes.c_ubyte * 2), 625), ('qp_error_syndrome', (ctypes.c_ubyte * 1), 627), ('reserved_at_274', (ctypes.c_ubyte * 2), 628), ('lag_dct', (ctypes.c_ubyte * 2), 630), ('lag_tx_port_affinity', (ctypes.c_ubyte * 1), 632), ('lag_native_fdb_selection', (ctypes.c_ubyte * 1), 633), ('reserved_at_27a', (ctypes.c_ubyte * 1), 634), ('lag_master', (ctypes.c_ubyte * 1), 635), ('num_lag_ports', (ctypes.c_ubyte * 4), 636), ('reserved_at_280', (ctypes.c_ubyte * 16), 640), ('max_wqe_sz_sq', (ctypes.c_ubyte * 16), 656), ('reserved_at_2a0', (ctypes.c_ubyte * 7), 672), ('mkey_pcie_tph', (ctypes.c_ubyte * 1), 679), ('reserved_at_2a8', (ctypes.c_ubyte * 1), 680), ('tis_tir_td_order', (ctypes.c_ubyte * 1), 681), ('psp', (ctypes.c_ubyte * 1), 682), ('shampo', (ctypes.c_ubyte * 1), 683), ('reserved_at_2ac', (ctypes.c_ubyte * 4), 684), ('max_wqe_sz_rq', (ctypes.c_ubyte * 16), 688), ('max_flow_counter_31_16', (ctypes.c_ubyte * 16), 704), ('max_wqe_sz_sq_dc', (ctypes.c_ubyte * 16), 720), ('reserved_at_2e0', (ctypes.c_ubyte * 7), 736), ('max_qp_mcg', (ctypes.c_ubyte * 25), 743), ('reserved_at_300', (ctypes.c_ubyte * 16), 768), ('flow_counter_bulk_alloc', (ctypes.c_ubyte * 8), 784), ('log_max_mcg', (ctypes.c_ubyte * 8), 792), ('reserved_at_320', (ctypes.c_ubyte * 3), 800), ('log_max_transport_domain', (ctypes.c_ubyte * 5), 803), ('reserved_at_328', (ctypes.c_ubyte * 2), 808), ('relaxed_ordering_read', (ctypes.c_ubyte * 1), 810), ('log_max_pd', (ctypes.c_ubyte * 5), 811), ('dp_ordering_ooo_all_ud', (ctypes.c_ubyte * 1), 816), ('dp_ordering_ooo_all_uc', (ctypes.c_ubyte * 1), 817), ('dp_ordering_ooo_all_xrc', (ctypes.c_ubyte * 1), 818), ('dp_ordering_ooo_all_dc', (ctypes.c_ubyte * 1), 819), ('dp_ordering_ooo_all_rc', (ctypes.c_ubyte * 1), 820), ('pcie_reset_using_hotreset_method', (ctypes.c_ubyte * 1), 821), ('pci_sync_for_fw_update_with_driver_unload', (ctypes.c_ubyte * 1), 822), ('vnic_env_cnt_steering_fail', (ctypes.c_ubyte * 1), 823), ('vport_counter_local_loopback', (ctypes.c_ubyte * 1), 824), ('q_counter_aggregation', (ctypes.c_ubyte * 1), 825), ('q_counter_other_vport', (ctypes.c_ubyte * 1), 826), ('log_max_xrcd', (ctypes.c_ubyte * 5), 827), ('nic_receive_steering_discard', (ctypes.c_ubyte * 1), 832), ('receive_discard_vport_down', (ctypes.c_ubyte * 1), 833), ('transmit_discard_vport_down', (ctypes.c_ubyte * 1), 834), ('eq_overrun_count', (ctypes.c_ubyte * 1), 835), ('reserved_at_344', (ctypes.c_ubyte * 1), 836), ('invalid_command_count', (ctypes.c_ubyte * 1), 837), ('quota_exceeded_count', (ctypes.c_ubyte * 1), 838), ('reserved_at_347', (ctypes.c_ubyte * 1), 839), ('log_max_flow_counter_bulk', (ctypes.c_ubyte * 8), 840), ('max_flow_counter_15_0', (ctypes.c_ubyte * 16), 848), ('reserved_at_360', (ctypes.c_ubyte * 3), 864), ('log_max_rq', (ctypes.c_ubyte * 5), 867), ('reserved_at_368', (ctypes.c_ubyte * 3), 872), ('log_max_sq', (ctypes.c_ubyte * 5), 875), ('reserved_at_370', (ctypes.c_ubyte * 3), 880), ('log_max_tir', (ctypes.c_ubyte * 5), 883), ('reserved_at_378', (ctypes.c_ubyte * 3), 888), ('log_max_tis', (ctypes.c_ubyte * 5), 891), ('basic_cyclic_rcv_wqe', (ctypes.c_ubyte * 1), 896), ('reserved_at_381', (ctypes.c_ubyte * 2), 897), ('log_max_rmp', (ctypes.c_ubyte * 5), 899), ('reserved_at_388', (ctypes.c_ubyte * 3), 904), ('log_max_rqt', (ctypes.c_ubyte * 5), 907), ('reserved_at_390', (ctypes.c_ubyte * 3), 912), ('log_max_rqt_size', (ctypes.c_ubyte * 5), 915), ('reserved_at_398', (ctypes.c_ubyte * 1), 920), ('vnic_env_cnt_bar_uar_access', (ctypes.c_ubyte * 1), 921), ('vnic_env_cnt_odp_page_fault', (ctypes.c_ubyte * 1), 922), ('log_max_tis_per_sq', (ctypes.c_ubyte * 5), 923), ('ext_stride_num_range', (ctypes.c_ubyte * 1), 928), ('roce_rw_supported', (ctypes.c_ubyte * 1), 929), ('log_max_current_uc_list_wr_supported', (ctypes.c_ubyte * 1), 930), ('log_max_stride_sz_rq', (ctypes.c_ubyte * 5), 931), ('reserved_at_3a8', (ctypes.c_ubyte * 3), 936), ('log_min_stride_sz_rq', (ctypes.c_ubyte * 5), 939), ('reserved_at_3b0', (ctypes.c_ubyte * 3), 944), ('log_max_stride_sz_sq', (ctypes.c_ubyte * 5), 947), ('reserved_at_3b8', (ctypes.c_ubyte * 3), 952), ('log_min_stride_sz_sq', (ctypes.c_ubyte * 5), 955), ('hairpin', (ctypes.c_ubyte * 1), 960), ('reserved_at_3c1', (ctypes.c_ubyte * 2), 961), ('log_max_hairpin_queues', (ctypes.c_ubyte * 5), 963), ('reserved_at_3c8', (ctypes.c_ubyte * 3), 968), ('log_max_hairpin_wq_data_sz', (ctypes.c_ubyte * 5), 971), ('reserved_at_3d0', (ctypes.c_ubyte * 3), 976), ('log_max_hairpin_num_packets', (ctypes.c_ubyte * 5), 979), ('reserved_at_3d8', (ctypes.c_ubyte * 3), 984), ('log_max_wq_sz', (ctypes.c_ubyte * 5), 987), ('nic_vport_change_event', (ctypes.c_ubyte * 1), 992), ('disable_local_lb_uc', (ctypes.c_ubyte * 1), 993), ('disable_local_lb_mc', (ctypes.c_ubyte * 1), 994), ('log_min_hairpin_wq_data_sz', (ctypes.c_ubyte * 5), 995), ('reserved_at_3e8', (ctypes.c_ubyte * 1), 1000), ('silent_mode', (ctypes.c_ubyte * 1), 1001), ('vhca_state', (ctypes.c_ubyte * 1), 1002), ('log_max_vlan_list', (ctypes.c_ubyte * 5), 1003), ('reserved_at_3f0', (ctypes.c_ubyte * 3), 1008), ('log_max_current_mc_list', (ctypes.c_ubyte * 5), 1011), ('reserved_at_3f8', (ctypes.c_ubyte * 3), 1016), ('log_max_current_uc_list', (ctypes.c_ubyte * 5), 1019), ('general_obj_types', (ctypes.c_ubyte * 64), 1024), ('sq_ts_format', (ctypes.c_ubyte * 2), 1088), ('rq_ts_format', (ctypes.c_ubyte * 2), 1090), ('steering_format_version', (ctypes.c_ubyte * 4), 1092), ('create_qp_start_hint', (ctypes.c_ubyte * 24), 1096), ('reserved_at_460', (ctypes.c_ubyte * 1), 1120), ('ats', (ctypes.c_ubyte * 1), 1121), ('cross_vhca_rqt', (ctypes.c_ubyte * 1), 1122), ('log_max_uctx', (ctypes.c_ubyte * 5), 1123), ('reserved_at_468', (ctypes.c_ubyte * 1), 1128), ('crypto', (ctypes.c_ubyte * 1), 1129), ('ipsec_offload', (ctypes.c_ubyte * 1), 1130), ('log_max_umem', (ctypes.c_ubyte * 5), 1131), ('max_num_eqs', (ctypes.c_ubyte * 16), 1136), ('reserved_at_480', (ctypes.c_ubyte * 1), 1152), ('tls_tx', (ctypes.c_ubyte * 1), 1153), ('tls_rx', (ctypes.c_ubyte * 1), 1154), ('log_max_l2_table', (ctypes.c_ubyte * 5), 1155), ('reserved_at_488', (ctypes.c_ubyte * 8), 1160), ('log_uar_page_sz', (ctypes.c_ubyte * 16), 1168), ('reserved_at_4a0', (ctypes.c_ubyte * 32), 1184), ('device_frequency_mhz', (ctypes.c_ubyte * 32), 1216), ('device_frequency_khz', (ctypes.c_ubyte * 32), 1248), ('reserved_at_500', (ctypes.c_ubyte * 32), 1280), ('num_of_uars_per_page', (ctypes.c_ubyte * 32), 1312), ('flex_parser_protocols', (ctypes.c_ubyte * 32), 1344), ('max_geneve_tlv_options', (ctypes.c_ubyte * 8), 1376), ('reserved_at_568', (ctypes.c_ubyte * 3), 1384), ('max_geneve_tlv_option_data_len', (ctypes.c_ubyte * 5), 1387), ('reserved_at_570', (ctypes.c_ubyte * 1), 1392), ('adv_rdma', (ctypes.c_ubyte * 1), 1393), ('reserved_at_572', (ctypes.c_ubyte * 7), 1394), ('adv_virtualization', (ctypes.c_ubyte * 1), 1401), ('reserved_at_57a', (ctypes.c_ubyte * 6), 1402), ('reserved_at_580', (ctypes.c_ubyte * 11), 1408), ('log_max_dci_stream_channels', (ctypes.c_ubyte * 5), 1419), ('reserved_at_590', (ctypes.c_ubyte * 3), 1424), ('log_max_dci_errored_streams', (ctypes.c_ubyte * 5), 1427), ('reserved_at_598', (ctypes.c_ubyte * 8), 1432), ('reserved_at_5a0', (ctypes.c_ubyte * 16), 1440), ('enhanced_cqe_compression', (ctypes.c_ubyte * 1), 1456), ('reserved_at_5b1', (ctypes.c_ubyte * 1), 1457), ('crossing_vhca_mkey', (ctypes.c_ubyte * 1), 1458), ('log_max_dek', (ctypes.c_ubyte * 5), 1459), ('reserved_at_5b8', (ctypes.c_ubyte * 4), 1464), ('mini_cqe_resp_stride_index', (ctypes.c_ubyte * 1), 1468), ('cqe_128_always', (ctypes.c_ubyte * 1), 1469), ('cqe_compression_128', (ctypes.c_ubyte * 1), 1470), ('cqe_compression', (ctypes.c_ubyte * 1), 1471), ('cqe_compression_timeout', (ctypes.c_ubyte * 16), 1472), ('cqe_compression_max_num', (ctypes.c_ubyte * 16), 1488), ('reserved_at_5e0', (ctypes.c_ubyte * 8), 1504), ('flex_parser_id_gtpu_dw_0', (ctypes.c_ubyte * 4), 1512), ('reserved_at_5ec', (ctypes.c_ubyte * 4), 1516), ('tag_matching', (ctypes.c_ubyte * 1), 1520), ('rndv_offload_rc', (ctypes.c_ubyte * 1), 1521), ('rndv_offload_dc', (ctypes.c_ubyte * 1), 1522), ('log_tag_matching_list_sz', (ctypes.c_ubyte * 5), 1523), ('reserved_at_5f8', (ctypes.c_ubyte * 3), 1528), ('log_max_xrq', (ctypes.c_ubyte * 5), 1531), ('affiliate_nic_vport_criteria', (ctypes.c_ubyte * 8), 1536), ('native_port_num', (ctypes.c_ubyte * 8), 1544), ('num_vhca_ports', (ctypes.c_ubyte * 8), 1552), ('flex_parser_id_gtpu_teid', (ctypes.c_ubyte * 4), 1560), ('reserved_at_61c', (ctypes.c_ubyte * 2), 1564), ('sw_owner_id', (ctypes.c_ubyte * 1), 1566), ('reserved_at_61f', (ctypes.c_ubyte * 1), 1567), ('max_num_of_monitor_counters', (ctypes.c_ubyte * 16), 1568), ('num_ppcnt_monitor_counters', (ctypes.c_ubyte * 16), 1584), ('max_num_sf', (ctypes.c_ubyte * 16), 1600), ('num_q_monitor_counters', (ctypes.c_ubyte * 16), 1616), ('reserved_at_660', (ctypes.c_ubyte * 32), 1632), ('sf', (ctypes.c_ubyte * 1), 1664), ('sf_set_partition', (ctypes.c_ubyte * 1), 1665), ('reserved_at_682', (ctypes.c_ubyte * 1), 1666), ('log_max_sf', (ctypes.c_ubyte * 5), 1667), ('apu', (ctypes.c_ubyte * 1), 1672), ('reserved_at_689', (ctypes.c_ubyte * 4), 1673), ('migration', (ctypes.c_ubyte * 1), 1677), ('reserved_at_68e', (ctypes.c_ubyte * 2), 1678), ('log_min_sf_size', (ctypes.c_ubyte * 8), 1680), ('max_num_sf_partitions', (ctypes.c_ubyte * 8), 1688), ('uctx_cap', (ctypes.c_ubyte * 32), 1696), ('reserved_at_6c0', (ctypes.c_ubyte * 4), 1728), ('flex_parser_id_geneve_tlv_option_0', (ctypes.c_ubyte * 4), 1732), ('flex_parser_id_icmp_dw1', (ctypes.c_ubyte * 4), 1736), ('flex_parser_id_icmp_dw0', (ctypes.c_ubyte * 4), 1740), ('flex_parser_id_icmpv6_dw1', (ctypes.c_ubyte * 4), 1744), ('flex_parser_id_icmpv6_dw0', (ctypes.c_ubyte * 4), 1748), ('flex_parser_id_outer_first_mpls_over_gre', (ctypes.c_ubyte * 4), 1752), ('flex_parser_id_outer_first_mpls_over_udp_label', (ctypes.c_ubyte * 4), 1756), ('max_num_match_definer', (ctypes.c_ubyte * 16), 1760), ('sf_base_id', (ctypes.c_ubyte * 16), 1776), ('flex_parser_id_gtpu_dw_2', (ctypes.c_ubyte * 4), 1792), ('flex_parser_id_gtpu_first_ext_dw_0', (ctypes.c_ubyte * 4), 1796), ('num_total_dynamic_vf_msix', (ctypes.c_ubyte * 24), 1800), ('reserved_at_720', (ctypes.c_ubyte * 20), 1824), ('dynamic_msix_table_size', (ctypes.c_ubyte * 12), 1844), ('reserved_at_740', (ctypes.c_ubyte * 12), 1856), ('min_dynamic_vf_msix_table_size', (ctypes.c_ubyte * 4), 1868), ('reserved_at_750', (ctypes.c_ubyte * 2), 1872), ('data_direct', (ctypes.c_ubyte * 1), 1874), ('reserved_at_753', (ctypes.c_ubyte * 1), 1875), ('max_dynamic_vf_msix_table_size', (ctypes.c_ubyte * 12), 1876), ('reserved_at_760', (ctypes.c_ubyte * 3), 1888), ('log_max_num_header_modify_argument', (ctypes.c_ubyte * 5), 1891), ('log_header_modify_argument_granularity_offset', (ctypes.c_ubyte * 4), 1896), ('log_header_modify_argument_granularity', (ctypes.c_ubyte * 4), 1900), ('reserved_at_770', (ctypes.c_ubyte * 3), 1904), ('log_header_modify_argument_max_alloc', (ctypes.c_ubyte * 5), 1907), ('reserved_at_778', (ctypes.c_ubyte * 8), 1912), ('vhca_tunnel_commands', (ctypes.c_ubyte * 64), 1920), ('match_definer_format_supported', (ctypes.c_ubyte * 64), 1984)])
_anonenum28: dict[int, str] = {(MLX5_CROSS_VHCA_OBJ_TO_OBJ_SUPPORTED_LOCAL_FLOW_TABLE_TO_REMOTE_FLOW_TABLE_MISS:=524288): 'MLX5_CROSS_VHCA_OBJ_TO_OBJ_SUPPORTED_LOCAL_FLOW_TABLE_TO_REMOTE_FLOW_TABLE_MISS', (MLX5_CROSS_VHCA_OBJ_TO_OBJ_SUPPORTED_LOCAL_FLOW_TABLE_ROOT_TO_REMOTE_FLOW_TABLE:=1048576): 'MLX5_CROSS_VHCA_OBJ_TO_OBJ_SUPPORTED_LOCAL_FLOW_TABLE_ROOT_TO_REMOTE_FLOW_TABLE'}
_anonenum29: dict[int, str] = {(MLX5_ALLOWED_OBJ_FOR_OTHER_VHCA_ACCESS_FLOW_TABLE:=512): 'MLX5_ALLOWED_OBJ_FOR_OTHER_VHCA_ACCESS_FLOW_TABLE'}
@c.record
class struct_mlx5_ifc_cmd_hca_cap_2_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  migratable: 'list[int]'
  reserved_at_81: 'list[int]'
  dp_ordering_force: 'list[int]'
  reserved_at_89: 'list[int]'
  query_vuid: 'list[int]'
  reserved_at_93: 'list[int]'
  umr_log_entity_size_5: 'list[int]'
  reserved_at_99: 'list[int]'
  max_reformat_insert_size: 'list[int]'
  max_reformat_insert_offset: 'list[int]'
  max_reformat_remove_size: 'list[int]'
  max_reformat_remove_offset: 'list[int]'
  reserved_at_c0: 'list[int]'
  migration_multi_load: 'list[int]'
  migration_tracking_state: 'list[int]'
  multiplane_qp_ud: 'list[int]'
  reserved_at_cb: 'list[int]'
  migration_in_chunks: 'list[int]'
  reserved_at_d1: 'list[int]'
  sf_eq_usage: 'list[int]'
  reserved_at_d3: 'list[int]'
  multiplane: 'list[int]'
  reserved_at_d9: 'list[int]'
  cross_vhca_object_to_object_supported: 'list[int]'
  allowed_object_for_other_vhca_access: 'list[int]'
  reserved_at_140: 'list[int]'
  flow_table_type_2_type: 'list[int]'
  reserved_at_1a8: 'list[int]'
  format_select_dw_8_6_ext: 'list[int]'
  log_min_mkey_entity_size: 'list[int]'
  reserved_at_1b0: 'list[int]'
  general_obj_types_127_64: 'list[int]'
  reserved_at_200: 'list[int]'
  reserved_at_220: 'list[int]'
  sw_vhca_id_valid: 'list[int]'
  sw_vhca_id: 'list[int]'
  reserved_at_230: 'list[int]'
  reserved_at_240: 'list[int]'
  ts_cqe_metadata_size2wqe_counter: 'list[int]'
  reserved_at_250: 'list[int]'
  reserved_at_260: 'list[int]'
  format_select_dw_gtpu_dw_0: 'list[int]'
  format_select_dw_gtpu_dw_1: 'list[int]'
  format_select_dw_gtpu_dw_2: 'list[int]'
  format_select_dw_gtpu_first_ext_dw_0: 'list[int]'
  generate_wqe_type: 'list[int]'
  reserved_at_2c0: 'list[int]'
  reserved_at_380: 'list[int]'
  min_mkey_log_entity_size_fixed_buffer: 'list[int]'
  ec_vf_vport_base: 'list[int]'
  reserved_at_3a0: 'list[int]'
  max_mkey_log_entity_size_fixed_buffer: 'list[int]'
  reserved_at_3a8: 'list[int]'
  max_mkey_log_entity_size_mtt: 'list[int]'
  max_rqt_vhca_id: 'list[int]'
  reserved_at_3c0: 'list[int]'
  reserved_at_3e0: 'list[int]'
  pcc_ifa2: 'list[int]'
  reserved_at_3f1: 'list[int]'
  reserved_at_400: 'list[int]'
  min_mkey_log_entity_size_fixed_buffer_valid: 'list[int]'
  reserved_at_402: 'list[int]'
  return_reg_id: 'list[int]'
  reserved_at_420: 'list[int]'
  flow_table_hash_type: 'list[int]'
  reserved_at_440: 'list[int]'
  max_num_eqs_24b: 'list[int]'
  reserved_at_460: 'list[int]'
  load_balance_id: 'list[int]'
  reserved_at_5a8: 'list[int]'
  query_adjacent_functions_id: 'list[int]'
  ingress_egress_esw_vport_connect: 'list[int]'
  function_id_type_vhca_id: 'list[int]'
  reserved_at_5c3: 'list[int]'
  lag_per_mp_group: 'list[int]'
  reserved_at_5c5: 'list[int]'
  delegate_vhca_management_profiles: 'list[int]'
  delegated_vhca_max: 'list[int]'
  delegate_vhca_max: 'list[int]'
  reserved_at_600: 'list[int]'
struct_mlx5_ifc_cmd_hca_cap_2_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 128), 0), ('migratable', (ctypes.c_ubyte * 1), 128), ('reserved_at_81', (ctypes.c_ubyte * 7), 129), ('dp_ordering_force', (ctypes.c_ubyte * 1), 136), ('reserved_at_89', (ctypes.c_ubyte * 9), 137), ('query_vuid', (ctypes.c_ubyte * 1), 146), ('reserved_at_93', (ctypes.c_ubyte * 5), 147), ('umr_log_entity_size_5', (ctypes.c_ubyte * 1), 152), ('reserved_at_99', (ctypes.c_ubyte * 7), 153), ('max_reformat_insert_size', (ctypes.c_ubyte * 8), 160), ('max_reformat_insert_offset', (ctypes.c_ubyte * 8), 168), ('max_reformat_remove_size', (ctypes.c_ubyte * 8), 176), ('max_reformat_remove_offset', (ctypes.c_ubyte * 8), 184), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('migration_multi_load', (ctypes.c_ubyte * 1), 200), ('migration_tracking_state', (ctypes.c_ubyte * 1), 201), ('multiplane_qp_ud', (ctypes.c_ubyte * 1), 202), ('reserved_at_cb', (ctypes.c_ubyte * 5), 203), ('migration_in_chunks', (ctypes.c_ubyte * 1), 208), ('reserved_at_d1', (ctypes.c_ubyte * 1), 209), ('sf_eq_usage', (ctypes.c_ubyte * 1), 210), ('reserved_at_d3', (ctypes.c_ubyte * 5), 211), ('multiplane', (ctypes.c_ubyte * 1), 216), ('reserved_at_d9', (ctypes.c_ubyte * 7), 217), ('cross_vhca_object_to_object_supported', (ctypes.c_ubyte * 32), 224), ('allowed_object_for_other_vhca_access', (ctypes.c_ubyte * 64), 256), ('reserved_at_140', (ctypes.c_ubyte * 96), 320), ('flow_table_type_2_type', (ctypes.c_ubyte * 8), 416), ('reserved_at_1a8', (ctypes.c_ubyte * 2), 424), ('format_select_dw_8_6_ext', (ctypes.c_ubyte * 1), 426), ('log_min_mkey_entity_size', (ctypes.c_ubyte * 5), 427), ('reserved_at_1b0', (ctypes.c_ubyte * 16), 432), ('general_obj_types_127_64', (ctypes.c_ubyte * 64), 448), ('reserved_at_200', (ctypes.c_ubyte * 32), 512), ('reserved_at_220', (ctypes.c_ubyte * 1), 544), ('sw_vhca_id_valid', (ctypes.c_ubyte * 1), 545), ('sw_vhca_id', (ctypes.c_ubyte * 14), 546), ('reserved_at_230', (ctypes.c_ubyte * 16), 560), ('reserved_at_240', (ctypes.c_ubyte * 11), 576), ('ts_cqe_metadata_size2wqe_counter', (ctypes.c_ubyte * 5), 587), ('reserved_at_250', (ctypes.c_ubyte * 16), 592), ('reserved_at_260', (ctypes.c_ubyte * 32), 608), ('format_select_dw_gtpu_dw_0', (ctypes.c_ubyte * 8), 640), ('format_select_dw_gtpu_dw_1', (ctypes.c_ubyte * 8), 648), ('format_select_dw_gtpu_dw_2', (ctypes.c_ubyte * 8), 656), ('format_select_dw_gtpu_first_ext_dw_0', (ctypes.c_ubyte * 8), 664), ('generate_wqe_type', (ctypes.c_ubyte * 32), 672), ('reserved_at_2c0', (ctypes.c_ubyte * 192), 704), ('reserved_at_380', (ctypes.c_ubyte * 11), 896), ('min_mkey_log_entity_size_fixed_buffer', (ctypes.c_ubyte * 5), 907), ('ec_vf_vport_base', (ctypes.c_ubyte * 16), 912), ('reserved_at_3a0', (ctypes.c_ubyte * 2), 928), ('max_mkey_log_entity_size_fixed_buffer', (ctypes.c_ubyte * 6), 930), ('reserved_at_3a8', (ctypes.c_ubyte * 2), 936), ('max_mkey_log_entity_size_mtt', (ctypes.c_ubyte * 6), 938), ('max_rqt_vhca_id', (ctypes.c_ubyte * 16), 944), ('reserved_at_3c0', (ctypes.c_ubyte * 32), 960), ('reserved_at_3e0', (ctypes.c_ubyte * 16), 992), ('pcc_ifa2', (ctypes.c_ubyte * 1), 1008), ('reserved_at_3f1', (ctypes.c_ubyte * 15), 1009), ('reserved_at_400', (ctypes.c_ubyte * 1), 1024), ('min_mkey_log_entity_size_fixed_buffer_valid', (ctypes.c_ubyte * 1), 1025), ('reserved_at_402', (ctypes.c_ubyte * 14), 1026), ('return_reg_id', (ctypes.c_ubyte * 16), 1040), ('reserved_at_420', (ctypes.c_ubyte * 28), 1056), ('flow_table_hash_type', (ctypes.c_ubyte * 4), 1084), ('reserved_at_440', (ctypes.c_ubyte * 8), 1088), ('max_num_eqs_24b', (ctypes.c_ubyte * 24), 1096), ('reserved_at_460', (ctypes.c_ubyte * 324), 1120), ('load_balance_id', (ctypes.c_ubyte * 4), 1444), ('reserved_at_5a8', (ctypes.c_ubyte * 24), 1448), ('query_adjacent_functions_id', (ctypes.c_ubyte * 1), 1472), ('ingress_egress_esw_vport_connect', (ctypes.c_ubyte * 1), 1473), ('function_id_type_vhca_id', (ctypes.c_ubyte * 1), 1474), ('reserved_at_5c3', (ctypes.c_ubyte * 1), 1475), ('lag_per_mp_group', (ctypes.c_ubyte * 1), 1476), ('reserved_at_5c5', (ctypes.c_ubyte * 11), 1477), ('delegate_vhca_management_profiles', (ctypes.c_ubyte * 16), 1488), ('delegated_vhca_max', (ctypes.c_ubyte * 16), 1504), ('delegate_vhca_max', (ctypes.c_ubyte * 16), 1520), ('reserved_at_600', (ctypes.c_ubyte * 512), 1536)])
enum_mlx5_ifc_flow_destination_type: dict[int, str] = {(MLX5_IFC_FLOW_DESTINATION_TYPE_VPORT:=0): 'MLX5_IFC_FLOW_DESTINATION_TYPE_VPORT', (MLX5_IFC_FLOW_DESTINATION_TYPE_FLOW_TABLE:=1): 'MLX5_IFC_FLOW_DESTINATION_TYPE_FLOW_TABLE', (MLX5_IFC_FLOW_DESTINATION_TYPE_TIR:=2): 'MLX5_IFC_FLOW_DESTINATION_TYPE_TIR', (MLX5_IFC_FLOW_DESTINATION_TYPE_FLOW_SAMPLER:=6): 'MLX5_IFC_FLOW_DESTINATION_TYPE_FLOW_SAMPLER', (MLX5_IFC_FLOW_DESTINATION_TYPE_UPLINK:=8): 'MLX5_IFC_FLOW_DESTINATION_TYPE_UPLINK', (MLX5_IFC_FLOW_DESTINATION_TYPE_TABLE_TYPE:=10): 'MLX5_IFC_FLOW_DESTINATION_TYPE_TABLE_TYPE'}
enum_mlx5_flow_table_miss_action: dict[int, str] = {(MLX5_FLOW_TABLE_MISS_ACTION_DEF:=0): 'MLX5_FLOW_TABLE_MISS_ACTION_DEF', (MLX5_FLOW_TABLE_MISS_ACTION_FWD:=1): 'MLX5_FLOW_TABLE_MISS_ACTION_FWD', (MLX5_FLOW_TABLE_MISS_ACTION_SWITCH_DOMAIN:=2): 'MLX5_FLOW_TABLE_MISS_ACTION_SWITCH_DOMAIN'}
@c.record
class struct_mlx5_ifc_dest_format_struct_bits(c.Struct):
  SIZE = 64
  destination_type: 'list[int]'
  destination_id: 'list[int]'
  destination_eswitch_owner_vhca_id_valid: 'list[int]'
  packet_reformat: 'list[int]'
  reserved_at_22: 'list[int]'
  destination_table_type: 'list[int]'
  destination_eswitch_owner_vhca_id: 'list[int]'
struct_mlx5_ifc_dest_format_struct_bits.register_fields([('destination_type', (ctypes.c_ubyte * 8), 0), ('destination_id', (ctypes.c_ubyte * 24), 8), ('destination_eswitch_owner_vhca_id_valid', (ctypes.c_ubyte * 1), 32), ('packet_reformat', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 6), 34), ('destination_table_type', (ctypes.c_ubyte * 8), 40), ('destination_eswitch_owner_vhca_id', (ctypes.c_ubyte * 16), 48)])
@c.record
class struct_mlx5_ifc_flow_counter_list_bits(c.Struct):
  SIZE = 64
  flow_counter_id: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_flow_counter_list_bits.register_fields([('flow_counter_id', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_extended_dest_format_bits(c.Struct):
  SIZE = 128
  destination_entry: 'struct_mlx5_ifc_dest_format_struct_bits'
  packet_reformat_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_extended_dest_format_bits.register_fields([('destination_entry', struct_mlx5_ifc_dest_format_struct_bits, 0), ('packet_reformat_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class union_mlx5_ifc_dest_format_flow_counter_list_auto_bits(c.Struct):
  SIZE = 128
  extended_dest_format: 'struct_mlx5_ifc_extended_dest_format_bits'
  flow_counter_list: 'struct_mlx5_ifc_flow_counter_list_bits'
union_mlx5_ifc_dest_format_flow_counter_list_auto_bits.register_fields([('extended_dest_format', struct_mlx5_ifc_extended_dest_format_bits, 0), ('flow_counter_list', struct_mlx5_ifc_flow_counter_list_bits, 0)])
@c.record
class struct_mlx5_ifc_fte_match_param_bits(c.Struct):
  SIZE = 4096
  outer_headers: 'struct_mlx5_ifc_fte_match_set_lyr_2_4_bits'
  misc_parameters: 'struct_mlx5_ifc_fte_match_set_misc_bits'
  inner_headers: 'struct_mlx5_ifc_fte_match_set_lyr_2_4_bits'
  misc_parameters_2: 'struct_mlx5_ifc_fte_match_set_misc2_bits'
  misc_parameters_3: 'struct_mlx5_ifc_fte_match_set_misc3_bits'
  misc_parameters_4: 'struct_mlx5_ifc_fte_match_set_misc4_bits'
  misc_parameters_5: 'struct_mlx5_ifc_fte_match_set_misc5_bits'
  reserved_at_e00: 'list[int]'
struct_mlx5_ifc_fte_match_param_bits.register_fields([('outer_headers', struct_mlx5_ifc_fte_match_set_lyr_2_4_bits, 0), ('misc_parameters', struct_mlx5_ifc_fte_match_set_misc_bits, 512), ('inner_headers', struct_mlx5_ifc_fte_match_set_lyr_2_4_bits, 1024), ('misc_parameters_2', struct_mlx5_ifc_fte_match_set_misc2_bits, 1536), ('misc_parameters_3', struct_mlx5_ifc_fte_match_set_misc3_bits, 2048), ('misc_parameters_4', struct_mlx5_ifc_fte_match_set_misc4_bits, 2560), ('misc_parameters_5', struct_mlx5_ifc_fte_match_set_misc5_bits, 3072), ('reserved_at_e00', (ctypes.c_ubyte * 512), 3584)])
_anonenum30: dict[int, str] = {(MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_SRC_IP:=0): 'MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_SRC_IP', (MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_DST_IP:=1): 'MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_DST_IP', (MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_L4_SPORT:=2): 'MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_L4_SPORT', (MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_L4_DPORT:=3): 'MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_L4_DPORT', (MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_IPSEC_SPI:=4): 'MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_IPSEC_SPI'}
@c.record
class struct_mlx5_ifc_rx_hash_field_select_bits(c.Struct):
  SIZE = 32
  l3_prot_type: 'list[int]'
  l4_prot_type: 'list[int]'
  selected_fields: 'list[int]'
struct_mlx5_ifc_rx_hash_field_select_bits.register_fields([('l3_prot_type', (ctypes.c_ubyte * 1), 0), ('l4_prot_type', (ctypes.c_ubyte * 1), 1), ('selected_fields', (ctypes.c_ubyte * 30), 2)])
_anonenum31: dict[int, str] = {(MLX5_WQ_WQ_TYPE_WQ_LINKED_LIST:=0): 'MLX5_WQ_WQ_TYPE_WQ_LINKED_LIST', (MLX5_WQ_WQ_TYPE_WQ_CYCLIC:=1): 'MLX5_WQ_WQ_TYPE_WQ_CYCLIC'}
_anonenum32: dict[int, str] = {(MLX5_WQ_END_PADDING_MODE_END_PAD_NONE:=0): 'MLX5_WQ_END_PADDING_MODE_END_PAD_NONE', (MLX5_WQ_END_PADDING_MODE_END_PAD_ALIGN:=1): 'MLX5_WQ_END_PADDING_MODE_END_PAD_ALIGN'}
@c.record
class struct_mlx5_ifc_wq_bits(c.Struct):
  SIZE = 1536
  wq_type: 'list[int]'
  wq_signature: 'list[int]'
  end_padding_mode: 'list[int]'
  cd_slave: 'list[int]'
  reserved_at_8: 'list[int]'
  hds_skip_first_sge: 'list[int]'
  log2_hds_buf_size: 'list[int]'
  reserved_at_24: 'list[int]'
  page_offset: 'list[int]'
  lwm: 'list[int]'
  reserved_at_40: 'list[int]'
  pd: 'list[int]'
  reserved_at_60: 'list[int]'
  uar_page: 'list[int]'
  dbr_addr: 'list[int]'
  hw_counter: 'list[int]'
  sw_counter: 'list[int]'
  reserved_at_100: 'list[int]'
  log_wq_stride: 'list[int]'
  reserved_at_110: 'list[int]'
  log_wq_pg_sz: 'list[int]'
  reserved_at_118: 'list[int]'
  log_wq_sz: 'list[int]'
  dbr_umem_valid: 'list[int]'
  wq_umem_valid: 'list[int]'
  reserved_at_122: 'list[int]'
  log_hairpin_num_packets: 'list[int]'
  reserved_at_128: 'list[int]'
  log_hairpin_data_sz: 'list[int]'
  reserved_at_130: 'list[int]'
  log_wqe_num_of_strides: 'list[int]'
  two_byte_shift_en: 'list[int]'
  reserved_at_139: 'list[int]'
  log_wqe_stride_size: 'list[int]'
  dbr_umem_id: 'list[int]'
  wq_umem_id: 'list[int]'
  wq_umem_offset: 'list[int]'
  headers_mkey: 'list[int]'
  shampo_enable: 'list[int]'
  reserved_at_1e1: 'list[int]'
  shampo_mode: 'list[int]'
  reserved_at_1e4: 'list[int]'
  log_reservation_size: 'list[int]'
  reserved_at_1e8: 'list[int]'
  log_max_num_of_packets_per_reservation: 'list[int]'
  reserved_at_1f0: 'list[int]'
  log_headers_entry_size: 'list[int]'
  reserved_at_1f8: 'list[int]'
  log_headers_buffer_entry_num: 'list[int]'
  reserved_at_200: 'list[int]'
  pas: 'list[struct_mlx5_ifc_cmd_pas_bits]'
struct_mlx5_ifc_wq_bits.register_fields([('wq_type', (ctypes.c_ubyte * 4), 0), ('wq_signature', (ctypes.c_ubyte * 1), 4), ('end_padding_mode', (ctypes.c_ubyte * 2), 5), ('cd_slave', (ctypes.c_ubyte * 1), 7), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('hds_skip_first_sge', (ctypes.c_ubyte * 1), 32), ('log2_hds_buf_size', (ctypes.c_ubyte * 3), 33), ('reserved_at_24', (ctypes.c_ubyte * 7), 36), ('page_offset', (ctypes.c_ubyte * 5), 43), ('lwm', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('pd', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 8), 96), ('uar_page', (ctypes.c_ubyte * 24), 104), ('dbr_addr', (ctypes.c_ubyte * 64), 128), ('hw_counter', (ctypes.c_ubyte * 32), 192), ('sw_counter', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 12), 256), ('log_wq_stride', (ctypes.c_ubyte * 4), 268), ('reserved_at_110', (ctypes.c_ubyte * 3), 272), ('log_wq_pg_sz', (ctypes.c_ubyte * 5), 275), ('reserved_at_118', (ctypes.c_ubyte * 3), 280), ('log_wq_sz', (ctypes.c_ubyte * 5), 283), ('dbr_umem_valid', (ctypes.c_ubyte * 1), 288), ('wq_umem_valid', (ctypes.c_ubyte * 1), 289), ('reserved_at_122', (ctypes.c_ubyte * 1), 290), ('log_hairpin_num_packets', (ctypes.c_ubyte * 5), 291), ('reserved_at_128', (ctypes.c_ubyte * 3), 296), ('log_hairpin_data_sz', (ctypes.c_ubyte * 5), 299), ('reserved_at_130', (ctypes.c_ubyte * 4), 304), ('log_wqe_num_of_strides', (ctypes.c_ubyte * 4), 308), ('two_byte_shift_en', (ctypes.c_ubyte * 1), 312), ('reserved_at_139', (ctypes.c_ubyte * 4), 313), ('log_wqe_stride_size', (ctypes.c_ubyte * 3), 317), ('dbr_umem_id', (ctypes.c_ubyte * 32), 320), ('wq_umem_id', (ctypes.c_ubyte * 32), 352), ('wq_umem_offset', (ctypes.c_ubyte * 64), 384), ('headers_mkey', (ctypes.c_ubyte * 32), 448), ('shampo_enable', (ctypes.c_ubyte * 1), 480), ('reserved_at_1e1', (ctypes.c_ubyte * 1), 481), ('shampo_mode', (ctypes.c_ubyte * 2), 482), ('reserved_at_1e4', (ctypes.c_ubyte * 1), 484), ('log_reservation_size', (ctypes.c_ubyte * 3), 485), ('reserved_at_1e8', (ctypes.c_ubyte * 5), 488), ('log_max_num_of_packets_per_reservation', (ctypes.c_ubyte * 3), 493), ('reserved_at_1f0', (ctypes.c_ubyte * 6), 496), ('log_headers_entry_size', (ctypes.c_ubyte * 2), 502), ('reserved_at_1f8', (ctypes.c_ubyte * 4), 504), ('log_headers_buffer_entry_num', (ctypes.c_ubyte * 4), 508), ('reserved_at_200', (ctypes.c_ubyte * 1024), 512), ('pas', (struct_mlx5_ifc_cmd_pas_bits * 0), 1536)])
@c.record
class struct_mlx5_ifc_rq_num_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  rq_num: 'list[int]'
struct_mlx5_ifc_rq_num_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('rq_num', (ctypes.c_ubyte * 24), 8)])
@c.record
class struct_mlx5_ifc_rq_vhca_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  rq_num: 'list[int]'
  reserved_at_20: 'list[int]'
  rq_vhca_id: 'list[int]'
struct_mlx5_ifc_rq_vhca_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('rq_num', (ctypes.c_ubyte * 24), 8), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('rq_vhca_id', (ctypes.c_ubyte * 16), 48)])
@c.record
class struct_mlx5_ifc_mac_address_layout_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  mac_addr_47_32: 'list[int]'
  mac_addr_31_0: 'list[int]'
struct_mlx5_ifc_mac_address_layout_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('mac_addr_47_32', (ctypes.c_ubyte * 16), 16), ('mac_addr_31_0', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_vlan_layout_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  vlan: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_vlan_layout_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 20), 0), ('vlan', (ctypes.c_ubyte * 12), 20), ('reserved_at_20', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_cong_control_r_roce_ecn_np_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  min_time_between_cnps: 'list[int]'
  reserved_at_c0: 'list[int]'
  cnp_dscp: 'list[int]'
  reserved_at_d8: 'list[int]'
  cnp_prio_mode: 'list[int]'
  cnp_802p_prio: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_cong_control_r_roce_ecn_np_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 160), 0), ('min_time_between_cnps', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 18), 192), ('cnp_dscp', (ctypes.c_ubyte * 6), 210), ('reserved_at_d8', (ctypes.c_ubyte * 4), 216), ('cnp_prio_mode', (ctypes.c_ubyte * 1), 220), ('cnp_802p_prio', (ctypes.c_ubyte * 3), 221), ('reserved_at_e0', (ctypes.c_ubyte * 1824), 224)])
@c.record
class struct_mlx5_ifc_cong_control_r_roce_ecn_rp_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  reserved_at_60: 'list[int]'
  clamp_tgt_rate: 'list[int]'
  reserved_at_65: 'list[int]'
  clamp_tgt_rate_after_time_inc: 'list[int]'
  reserved_at_69: 'list[int]'
  reserved_at_80: 'list[int]'
  rpg_time_reset: 'list[int]'
  rpg_byte_reset: 'list[int]'
  rpg_threshold: 'list[int]'
  rpg_max_rate: 'list[int]'
  rpg_ai_rate: 'list[int]'
  rpg_hai_rate: 'list[int]'
  rpg_gd: 'list[int]'
  rpg_min_dec_fac: 'list[int]'
  rpg_min_rate: 'list[int]'
  reserved_at_1c0: 'list[int]'
  rate_to_set_on_first_cnp: 'list[int]'
  dce_tcp_g: 'list[int]'
  dce_tcp_rtt: 'list[int]'
  rate_reduce_monitor_period: 'list[int]'
  reserved_at_320: 'list[int]'
  initial_alpha_value: 'list[int]'
  reserved_at_360: 'list[int]'
struct_mlx5_ifc_cong_control_r_roce_ecn_rp_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 96), 0), ('reserved_at_60', (ctypes.c_ubyte * 4), 96), ('clamp_tgt_rate', (ctypes.c_ubyte * 1), 100), ('reserved_at_65', (ctypes.c_ubyte * 3), 101), ('clamp_tgt_rate_after_time_inc', (ctypes.c_ubyte * 1), 104), ('reserved_at_69', (ctypes.c_ubyte * 23), 105), ('reserved_at_80', (ctypes.c_ubyte * 32), 128), ('rpg_time_reset', (ctypes.c_ubyte * 32), 160), ('rpg_byte_reset', (ctypes.c_ubyte * 32), 192), ('rpg_threshold', (ctypes.c_ubyte * 32), 224), ('rpg_max_rate', (ctypes.c_ubyte * 32), 256), ('rpg_ai_rate', (ctypes.c_ubyte * 32), 288), ('rpg_hai_rate', (ctypes.c_ubyte * 32), 320), ('rpg_gd', (ctypes.c_ubyte * 32), 352), ('rpg_min_dec_fac', (ctypes.c_ubyte * 32), 384), ('rpg_min_rate', (ctypes.c_ubyte * 32), 416), ('reserved_at_1c0', (ctypes.c_ubyte * 224), 448), ('rate_to_set_on_first_cnp', (ctypes.c_ubyte * 32), 672), ('dce_tcp_g', (ctypes.c_ubyte * 32), 704), ('dce_tcp_rtt', (ctypes.c_ubyte * 32), 736), ('rate_reduce_monitor_period', (ctypes.c_ubyte * 32), 768), ('reserved_at_320', (ctypes.c_ubyte * 32), 800), ('initial_alpha_value', (ctypes.c_ubyte * 32), 832), ('reserved_at_360', (ctypes.c_ubyte * 1184), 864)])
@c.record
class struct_mlx5_ifc_cong_control_r_roce_general_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  reserved_at_80: 'list[int]'
  rtt_resp_dscp_valid: 'list[int]'
  reserved_at_91: 'list[int]'
  rtt_resp_dscp: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_cong_control_r_roce_general_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 128), 0), ('reserved_at_80', (ctypes.c_ubyte * 16), 128), ('rtt_resp_dscp_valid', (ctypes.c_ubyte * 1), 144), ('reserved_at_91', (ctypes.c_ubyte * 9), 145), ('rtt_resp_dscp', (ctypes.c_ubyte * 6), 154), ('reserved_at_a0', (ctypes.c_ubyte * 1888), 160)])
@c.record
class struct_mlx5_ifc_cong_control_802_1qau_rp_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  rppp_max_rps: 'list[int]'
  rpg_time_reset: 'list[int]'
  rpg_byte_reset: 'list[int]'
  rpg_threshold: 'list[int]'
  rpg_max_rate: 'list[int]'
  rpg_ai_rate: 'list[int]'
  rpg_hai_rate: 'list[int]'
  rpg_gd: 'list[int]'
  rpg_min_dec_fac: 'list[int]'
  rpg_min_rate: 'list[int]'
  reserved_at_1c0: 'list[int]'
struct_mlx5_ifc_cong_control_802_1qau_rp_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 128), 0), ('rppp_max_rps', (ctypes.c_ubyte * 32), 128), ('rpg_time_reset', (ctypes.c_ubyte * 32), 160), ('rpg_byte_reset', (ctypes.c_ubyte * 32), 192), ('rpg_threshold', (ctypes.c_ubyte * 32), 224), ('rpg_max_rate', (ctypes.c_ubyte * 32), 256), ('rpg_ai_rate', (ctypes.c_ubyte * 32), 288), ('rpg_hai_rate', (ctypes.c_ubyte * 32), 320), ('rpg_gd', (ctypes.c_ubyte * 32), 352), ('rpg_min_dec_fac', (ctypes.c_ubyte * 32), 384), ('rpg_min_rate', (ctypes.c_ubyte * 32), 416), ('reserved_at_1c0', (ctypes.c_ubyte * 1600), 448)])
_anonenum33: dict[int, str] = {(MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_LOG_CQ_SIZE:=1): 'MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_LOG_CQ_SIZE', (MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_PAGE_OFFSET:=2): 'MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_PAGE_OFFSET', (MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_LOG_PAGE_SIZE:=4): 'MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_LOG_PAGE_SIZE'}
@c.record
class struct_mlx5_ifc_resize_field_select_bits(c.Struct):
  SIZE = 32
  resize_field_select: 'list[int]'
struct_mlx5_ifc_resize_field_select_bits.register_fields([('resize_field_select', (ctypes.c_ubyte * 32), 0)])
@c.record
class struct_mlx5_ifc_resource_dump_bits(c.Struct):
  SIZE = 2048
  more_dump: 'list[int]'
  inline_dump: 'list[int]'
  reserved_at_2: 'list[int]'
  seq_num: 'list[int]'
  segment_type: 'list[int]'
  reserved_at_20: 'list[int]'
  vhca_id: 'list[int]'
  index1: 'list[int]'
  index2: 'list[int]'
  num_of_obj1: 'list[int]'
  num_of_obj2: 'list[int]'
  reserved_at_a0: 'list[int]'
  device_opaque: 'list[int]'
  mkey: 'list[int]'
  size: 'list[int]'
  address: 'list[int]'
  inline_data: 'list[list[int]]'
struct_mlx5_ifc_resource_dump_bits.register_fields([('more_dump', (ctypes.c_ubyte * 1), 0), ('inline_dump', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 10), 2), ('seq_num', (ctypes.c_ubyte * 4), 12), ('segment_type', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('vhca_id', (ctypes.c_ubyte * 16), 48), ('index1', (ctypes.c_ubyte * 32), 64), ('index2', (ctypes.c_ubyte * 32), 96), ('num_of_obj1', (ctypes.c_ubyte * 16), 128), ('num_of_obj2', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('device_opaque', (ctypes.c_ubyte * 64), 192), ('mkey', (ctypes.c_ubyte * 32), 256), ('size', (ctypes.c_ubyte * 32), 288), ('address', (ctypes.c_ubyte * 64), 320), ('inline_data', ((ctypes.c_ubyte * 32) * 52), 384)])
@c.record
class struct_mlx5_ifc_resource_dump_menu_record_bits(c.Struct):
  SIZE = 416
  reserved_at_0: 'list[int]'
  num_of_obj2_supports_active: 'list[int]'
  num_of_obj2_supports_all: 'list[int]'
  must_have_num_of_obj2: 'list[int]'
  support_num_of_obj2: 'list[int]'
  num_of_obj1_supports_active: 'list[int]'
  num_of_obj1_supports_all: 'list[int]'
  must_have_num_of_obj1: 'list[int]'
  support_num_of_obj1: 'list[int]'
  must_have_index2: 'list[int]'
  support_index2: 'list[int]'
  must_have_index1: 'list[int]'
  support_index1: 'list[int]'
  segment_type: 'list[int]'
  segment_name: 'list[list[int]]'
  index1_name: 'list[list[int]]'
  index2_name: 'list[list[int]]'
struct_mlx5_ifc_resource_dump_menu_record_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('num_of_obj2_supports_active', (ctypes.c_ubyte * 1), 4), ('num_of_obj2_supports_all', (ctypes.c_ubyte * 1), 5), ('must_have_num_of_obj2', (ctypes.c_ubyte * 1), 6), ('support_num_of_obj2', (ctypes.c_ubyte * 1), 7), ('num_of_obj1_supports_active', (ctypes.c_ubyte * 1), 8), ('num_of_obj1_supports_all', (ctypes.c_ubyte * 1), 9), ('must_have_num_of_obj1', (ctypes.c_ubyte * 1), 10), ('support_num_of_obj1', (ctypes.c_ubyte * 1), 11), ('must_have_index2', (ctypes.c_ubyte * 1), 12), ('support_index2', (ctypes.c_ubyte * 1), 13), ('must_have_index1', (ctypes.c_ubyte * 1), 14), ('support_index1', (ctypes.c_ubyte * 1), 15), ('segment_type', (ctypes.c_ubyte * 16), 16), ('segment_name', ((ctypes.c_ubyte * 32) * 4), 32), ('index1_name', ((ctypes.c_ubyte * 32) * 4), 160), ('index2_name', ((ctypes.c_ubyte * 32) * 4), 288)])
@c.record
class struct_mlx5_ifc_resource_dump_segment_header_bits(c.Struct):
  SIZE = 32
  length_dw: 'list[int]'
  segment_type: 'list[int]'
struct_mlx5_ifc_resource_dump_segment_header_bits.register_fields([('length_dw', (ctypes.c_ubyte * 16), 0), ('segment_type', (ctypes.c_ubyte * 16), 16)])
@c.record
class struct_mlx5_ifc_resource_dump_command_segment_bits(c.Struct):
  SIZE = 160
  segment_header: 'struct_mlx5_ifc_resource_dump_segment_header_bits'
  segment_called: 'list[int]'
  vhca_id: 'list[int]'
  index1: 'list[int]'
  index2: 'list[int]'
  num_of_obj1: 'list[int]'
  num_of_obj2: 'list[int]'
struct_mlx5_ifc_resource_dump_command_segment_bits.register_fields([('segment_header', struct_mlx5_ifc_resource_dump_segment_header_bits, 0), ('segment_called', (ctypes.c_ubyte * 16), 32), ('vhca_id', (ctypes.c_ubyte * 16), 48), ('index1', (ctypes.c_ubyte * 32), 64), ('index2', (ctypes.c_ubyte * 32), 96), ('num_of_obj1', (ctypes.c_ubyte * 16), 128), ('num_of_obj2', (ctypes.c_ubyte * 16), 144)])
@c.record
class struct_mlx5_ifc_resource_dump_error_segment_bits(c.Struct):
  SIZE = 384
  segment_header: 'struct_mlx5_ifc_resource_dump_segment_header_bits'
  reserved_at_20: 'list[int]'
  syndrome_id: 'list[int]'
  reserved_at_40: 'list[int]'
  error: 'list[list[int]]'
struct_mlx5_ifc_resource_dump_error_segment_bits.register_fields([('segment_header', struct_mlx5_ifc_resource_dump_segment_header_bits, 0), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('syndrome_id', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('error', ((ctypes.c_ubyte * 32) * 8), 128)])
@c.record
class struct_mlx5_ifc_resource_dump_info_segment_bits(c.Struct):
  SIZE = 128
  segment_header: 'struct_mlx5_ifc_resource_dump_segment_header_bits'
  reserved_at_20: 'list[int]'
  dump_version: 'list[int]'
  hw_version: 'list[int]'
  fw_version: 'list[int]'
struct_mlx5_ifc_resource_dump_info_segment_bits.register_fields([('segment_header', struct_mlx5_ifc_resource_dump_segment_header_bits, 0), ('reserved_at_20', (ctypes.c_ubyte * 24), 32), ('dump_version', (ctypes.c_ubyte * 8), 56), ('hw_version', (ctypes.c_ubyte * 32), 64), ('fw_version', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_resource_dump_menu_segment_bits(c.Struct):
  SIZE = 64
  segment_header: 'struct_mlx5_ifc_resource_dump_segment_header_bits'
  reserved_at_20: 'list[int]'
  num_of_records: 'list[int]'
  record: 'list[struct_mlx5_ifc_resource_dump_menu_record_bits]'
struct_mlx5_ifc_resource_dump_menu_segment_bits.register_fields([('segment_header', struct_mlx5_ifc_resource_dump_segment_header_bits, 0), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('num_of_records', (ctypes.c_ubyte * 16), 48), ('record', (struct_mlx5_ifc_resource_dump_menu_record_bits * 0), 64)])
@c.record
class struct_mlx5_ifc_resource_dump_resource_segment_bits(c.Struct):
  SIZE = 128
  segment_header: 'struct_mlx5_ifc_resource_dump_segment_header_bits'
  reserved_at_20: 'list[int]'
  index1: 'list[int]'
  index2: 'list[int]'
  payload: 'list[list[int]]'
struct_mlx5_ifc_resource_dump_resource_segment_bits.register_fields([('segment_header', struct_mlx5_ifc_resource_dump_segment_header_bits, 0), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('index1', (ctypes.c_ubyte * 32), 64), ('index2', (ctypes.c_ubyte * 32), 96), ('payload', ((ctypes.c_ubyte * 32) * 0), 128)])
@c.record
class struct_mlx5_ifc_resource_dump_terminate_segment_bits(c.Struct):
  SIZE = 32
  segment_header: 'struct_mlx5_ifc_resource_dump_segment_header_bits'
struct_mlx5_ifc_resource_dump_terminate_segment_bits.register_fields([('segment_header', struct_mlx5_ifc_resource_dump_segment_header_bits, 0)])
@c.record
class struct_mlx5_ifc_menu_resource_dump_response_bits(c.Struct):
  SIZE = 384
  info: 'struct_mlx5_ifc_resource_dump_info_segment_bits'
  cmd: 'struct_mlx5_ifc_resource_dump_command_segment_bits'
  menu: 'struct_mlx5_ifc_resource_dump_menu_segment_bits'
  terminate: 'struct_mlx5_ifc_resource_dump_terminate_segment_bits'
struct_mlx5_ifc_menu_resource_dump_response_bits.register_fields([('info', struct_mlx5_ifc_resource_dump_info_segment_bits, 0), ('cmd', struct_mlx5_ifc_resource_dump_command_segment_bits, 128), ('menu', struct_mlx5_ifc_resource_dump_menu_segment_bits, 288), ('terminate', struct_mlx5_ifc_resource_dump_terminate_segment_bits, 352)])
_anonenum34: dict[int, str] = {(MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_CQ_PERIOD:=1): 'MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_CQ_PERIOD', (MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_CQ_MAX_COUNT:=2): 'MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_CQ_MAX_COUNT', (MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_OI:=4): 'MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_OI', (MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_C_EQN:=8): 'MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_C_EQN'}
@c.record
class struct_mlx5_ifc_modify_field_select_bits(c.Struct):
  SIZE = 32
  modify_field_select: 'list[int]'
struct_mlx5_ifc_modify_field_select_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 32), 0)])
@c.record
class struct_mlx5_ifc_field_select_r_roce_np_bits(c.Struct):
  SIZE = 32
  field_select_r_roce_np: 'list[int]'
struct_mlx5_ifc_field_select_r_roce_np_bits.register_fields([('field_select_r_roce_np', (ctypes.c_ubyte * 32), 0)])
@c.record
class struct_mlx5_ifc_field_select_r_roce_rp_bits(c.Struct):
  SIZE = 32
  field_select_r_roce_rp: 'list[int]'
struct_mlx5_ifc_field_select_r_roce_rp_bits.register_fields([('field_select_r_roce_rp', (ctypes.c_ubyte * 32), 0)])
_anonenum35: dict[int, str] = {(MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPPP_MAX_RPS:=4): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPPP_MAX_RPS', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_TIME_RESET:=8): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_TIME_RESET', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_BYTE_RESET:=16): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_BYTE_RESET', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_THRESHOLD:=32): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_THRESHOLD', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MAX_RATE:=64): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MAX_RATE', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_AI_RATE:=128): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_AI_RATE', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_HAI_RATE:=256): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_HAI_RATE', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_GD:=512): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_GD', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MIN_DEC_FAC:=1024): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MIN_DEC_FAC', (MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MIN_RATE:=2048): 'MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MIN_RATE'}
@c.record
class struct_mlx5_ifc_field_select_802_1qau_rp_bits(c.Struct):
  SIZE = 32
  field_select_8021qaurp: 'list[int]'
struct_mlx5_ifc_field_select_802_1qau_rp_bits.register_fields([('field_select_8021qaurp', (ctypes.c_ubyte * 32), 0)])
@c.record
class struct_mlx5_ifc_phys_layer_recovery_cntrs_bits(c.Struct):
  SIZE = 1984
  total_successful_recovery_events: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_phys_layer_recovery_cntrs_bits.register_fields([('total_successful_recovery_events', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 1952), 32)])
@c.record
class struct_mlx5_ifc_phys_layer_cntrs_bits(c.Struct):
  SIZE = 1984
  time_since_last_clear_high: 'list[int]'
  time_since_last_clear_low: 'list[int]'
  symbol_errors_high: 'list[int]'
  symbol_errors_low: 'list[int]'
  sync_headers_errors_high: 'list[int]'
  sync_headers_errors_low: 'list[int]'
  edpl_bip_errors_lane0_high: 'list[int]'
  edpl_bip_errors_lane0_low: 'list[int]'
  edpl_bip_errors_lane1_high: 'list[int]'
  edpl_bip_errors_lane1_low: 'list[int]'
  edpl_bip_errors_lane2_high: 'list[int]'
  edpl_bip_errors_lane2_low: 'list[int]'
  edpl_bip_errors_lane3_high: 'list[int]'
  edpl_bip_errors_lane3_low: 'list[int]'
  fc_fec_corrected_blocks_lane0_high: 'list[int]'
  fc_fec_corrected_blocks_lane0_low: 'list[int]'
  fc_fec_corrected_blocks_lane1_high: 'list[int]'
  fc_fec_corrected_blocks_lane1_low: 'list[int]'
  fc_fec_corrected_blocks_lane2_high: 'list[int]'
  fc_fec_corrected_blocks_lane2_low: 'list[int]'
  fc_fec_corrected_blocks_lane3_high: 'list[int]'
  fc_fec_corrected_blocks_lane3_low: 'list[int]'
  fc_fec_uncorrectable_blocks_lane0_high: 'list[int]'
  fc_fec_uncorrectable_blocks_lane0_low: 'list[int]'
  fc_fec_uncorrectable_blocks_lane1_high: 'list[int]'
  fc_fec_uncorrectable_blocks_lane1_low: 'list[int]'
  fc_fec_uncorrectable_blocks_lane2_high: 'list[int]'
  fc_fec_uncorrectable_blocks_lane2_low: 'list[int]'
  fc_fec_uncorrectable_blocks_lane3_high: 'list[int]'
  fc_fec_uncorrectable_blocks_lane3_low: 'list[int]'
  rs_fec_corrected_blocks_high: 'list[int]'
  rs_fec_corrected_blocks_low: 'list[int]'
  rs_fec_uncorrectable_blocks_high: 'list[int]'
  rs_fec_uncorrectable_blocks_low: 'list[int]'
  rs_fec_no_errors_blocks_high: 'list[int]'
  rs_fec_no_errors_blocks_low: 'list[int]'
  rs_fec_single_error_blocks_high: 'list[int]'
  rs_fec_single_error_blocks_low: 'list[int]'
  rs_fec_corrected_symbols_total_high: 'list[int]'
  rs_fec_corrected_symbols_total_low: 'list[int]'
  rs_fec_corrected_symbols_lane0_high: 'list[int]'
  rs_fec_corrected_symbols_lane0_low: 'list[int]'
  rs_fec_corrected_symbols_lane1_high: 'list[int]'
  rs_fec_corrected_symbols_lane1_low: 'list[int]'
  rs_fec_corrected_symbols_lane2_high: 'list[int]'
  rs_fec_corrected_symbols_lane2_low: 'list[int]'
  rs_fec_corrected_symbols_lane3_high: 'list[int]'
  rs_fec_corrected_symbols_lane3_low: 'list[int]'
  link_down_events: 'list[int]'
  successful_recovery_events: 'list[int]'
  reserved_at_640: 'list[int]'
struct_mlx5_ifc_phys_layer_cntrs_bits.register_fields([('time_since_last_clear_high', (ctypes.c_ubyte * 32), 0), ('time_since_last_clear_low', (ctypes.c_ubyte * 32), 32), ('symbol_errors_high', (ctypes.c_ubyte * 32), 64), ('symbol_errors_low', (ctypes.c_ubyte * 32), 96), ('sync_headers_errors_high', (ctypes.c_ubyte * 32), 128), ('sync_headers_errors_low', (ctypes.c_ubyte * 32), 160), ('edpl_bip_errors_lane0_high', (ctypes.c_ubyte * 32), 192), ('edpl_bip_errors_lane0_low', (ctypes.c_ubyte * 32), 224), ('edpl_bip_errors_lane1_high', (ctypes.c_ubyte * 32), 256), ('edpl_bip_errors_lane1_low', (ctypes.c_ubyte * 32), 288), ('edpl_bip_errors_lane2_high', (ctypes.c_ubyte * 32), 320), ('edpl_bip_errors_lane2_low', (ctypes.c_ubyte * 32), 352), ('edpl_bip_errors_lane3_high', (ctypes.c_ubyte * 32), 384), ('edpl_bip_errors_lane3_low', (ctypes.c_ubyte * 32), 416), ('fc_fec_corrected_blocks_lane0_high', (ctypes.c_ubyte * 32), 448), ('fc_fec_corrected_blocks_lane0_low', (ctypes.c_ubyte * 32), 480), ('fc_fec_corrected_blocks_lane1_high', (ctypes.c_ubyte * 32), 512), ('fc_fec_corrected_blocks_lane1_low', (ctypes.c_ubyte * 32), 544), ('fc_fec_corrected_blocks_lane2_high', (ctypes.c_ubyte * 32), 576), ('fc_fec_corrected_blocks_lane2_low', (ctypes.c_ubyte * 32), 608), ('fc_fec_corrected_blocks_lane3_high', (ctypes.c_ubyte * 32), 640), ('fc_fec_corrected_blocks_lane3_low', (ctypes.c_ubyte * 32), 672), ('fc_fec_uncorrectable_blocks_lane0_high', (ctypes.c_ubyte * 32), 704), ('fc_fec_uncorrectable_blocks_lane0_low', (ctypes.c_ubyte * 32), 736), ('fc_fec_uncorrectable_blocks_lane1_high', (ctypes.c_ubyte * 32), 768), ('fc_fec_uncorrectable_blocks_lane1_low', (ctypes.c_ubyte * 32), 800), ('fc_fec_uncorrectable_blocks_lane2_high', (ctypes.c_ubyte * 32), 832), ('fc_fec_uncorrectable_blocks_lane2_low', (ctypes.c_ubyte * 32), 864), ('fc_fec_uncorrectable_blocks_lane3_high', (ctypes.c_ubyte * 32), 896), ('fc_fec_uncorrectable_blocks_lane3_low', (ctypes.c_ubyte * 32), 928), ('rs_fec_corrected_blocks_high', (ctypes.c_ubyte * 32), 960), ('rs_fec_corrected_blocks_low', (ctypes.c_ubyte * 32), 992), ('rs_fec_uncorrectable_blocks_high', (ctypes.c_ubyte * 32), 1024), ('rs_fec_uncorrectable_blocks_low', (ctypes.c_ubyte * 32), 1056), ('rs_fec_no_errors_blocks_high', (ctypes.c_ubyte * 32), 1088), ('rs_fec_no_errors_blocks_low', (ctypes.c_ubyte * 32), 1120), ('rs_fec_single_error_blocks_high', (ctypes.c_ubyte * 32), 1152), ('rs_fec_single_error_blocks_low', (ctypes.c_ubyte * 32), 1184), ('rs_fec_corrected_symbols_total_high', (ctypes.c_ubyte * 32), 1216), ('rs_fec_corrected_symbols_total_low', (ctypes.c_ubyte * 32), 1248), ('rs_fec_corrected_symbols_lane0_high', (ctypes.c_ubyte * 32), 1280), ('rs_fec_corrected_symbols_lane0_low', (ctypes.c_ubyte * 32), 1312), ('rs_fec_corrected_symbols_lane1_high', (ctypes.c_ubyte * 32), 1344), ('rs_fec_corrected_symbols_lane1_low', (ctypes.c_ubyte * 32), 1376), ('rs_fec_corrected_symbols_lane2_high', (ctypes.c_ubyte * 32), 1408), ('rs_fec_corrected_symbols_lane2_low', (ctypes.c_ubyte * 32), 1440), ('rs_fec_corrected_symbols_lane3_high', (ctypes.c_ubyte * 32), 1472), ('rs_fec_corrected_symbols_lane3_low', (ctypes.c_ubyte * 32), 1504), ('link_down_events', (ctypes.c_ubyte * 32), 1536), ('successful_recovery_events', (ctypes.c_ubyte * 32), 1568), ('reserved_at_640', (ctypes.c_ubyte * 384), 1600)])
@c.record
class struct_mlx5_ifc_phys_layer_statistical_cntrs_bits(c.Struct):
  SIZE = 1984
  time_since_last_clear_high: 'list[int]'
  time_since_last_clear_low: 'list[int]'
  phy_received_bits_high: 'list[int]'
  phy_received_bits_low: 'list[int]'
  phy_symbol_errors_high: 'list[int]'
  phy_symbol_errors_low: 'list[int]'
  phy_corrected_bits_high: 'list[int]'
  phy_corrected_bits_low: 'list[int]'
  phy_corrected_bits_lane0_high: 'list[int]'
  phy_corrected_bits_lane0_low: 'list[int]'
  phy_corrected_bits_lane1_high: 'list[int]'
  phy_corrected_bits_lane1_low: 'list[int]'
  phy_corrected_bits_lane2_high: 'list[int]'
  phy_corrected_bits_lane2_low: 'list[int]'
  phy_corrected_bits_lane3_high: 'list[int]'
  phy_corrected_bits_lane3_low: 'list[int]'
  reserved_at_200: 'list[int]'
struct_mlx5_ifc_phys_layer_statistical_cntrs_bits.register_fields([('time_since_last_clear_high', (ctypes.c_ubyte * 32), 0), ('time_since_last_clear_low', (ctypes.c_ubyte * 32), 32), ('phy_received_bits_high', (ctypes.c_ubyte * 32), 64), ('phy_received_bits_low', (ctypes.c_ubyte * 32), 96), ('phy_symbol_errors_high', (ctypes.c_ubyte * 32), 128), ('phy_symbol_errors_low', (ctypes.c_ubyte * 32), 160), ('phy_corrected_bits_high', (ctypes.c_ubyte * 32), 192), ('phy_corrected_bits_low', (ctypes.c_ubyte * 32), 224), ('phy_corrected_bits_lane0_high', (ctypes.c_ubyte * 32), 256), ('phy_corrected_bits_lane0_low', (ctypes.c_ubyte * 32), 288), ('phy_corrected_bits_lane1_high', (ctypes.c_ubyte * 32), 320), ('phy_corrected_bits_lane1_low', (ctypes.c_ubyte * 32), 352), ('phy_corrected_bits_lane2_high', (ctypes.c_ubyte * 32), 384), ('phy_corrected_bits_lane2_low', (ctypes.c_ubyte * 32), 416), ('phy_corrected_bits_lane3_high', (ctypes.c_ubyte * 32), 448), ('phy_corrected_bits_lane3_low', (ctypes.c_ubyte * 32), 480), ('reserved_at_200', (ctypes.c_ubyte * 1472), 512)])
@c.record
class struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 320
  symbol_error_counter: 'list[int]'
  link_error_recovery_counter: 'list[int]'
  link_downed_counter: 'list[int]'
  port_rcv_errors: 'list[int]'
  port_rcv_remote_physical_errors: 'list[int]'
  port_rcv_switch_relay_errors: 'list[int]'
  port_xmit_discards: 'list[int]'
  port_xmit_constraint_errors: 'list[int]'
  port_rcv_constraint_errors: 'list[int]'
  reserved_at_70: 'list[int]'
  link_overrun_errors: 'list[int]'
  reserved_at_80: 'list[int]'
  vl_15_dropped: 'list[int]'
  reserved_at_a0: 'list[int]'
  port_xmit_wait: 'list[int]'
struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits.register_fields([('symbol_error_counter', (ctypes.c_ubyte * 16), 0), ('link_error_recovery_counter', (ctypes.c_ubyte * 8), 16), ('link_downed_counter', (ctypes.c_ubyte * 8), 24), ('port_rcv_errors', (ctypes.c_ubyte * 16), 32), ('port_rcv_remote_physical_errors', (ctypes.c_ubyte * 16), 48), ('port_rcv_switch_relay_errors', (ctypes.c_ubyte * 16), 64), ('port_xmit_discards', (ctypes.c_ubyte * 16), 80), ('port_xmit_constraint_errors', (ctypes.c_ubyte * 8), 96), ('port_rcv_constraint_errors', (ctypes.c_ubyte * 8), 104), ('reserved_at_70', (ctypes.c_ubyte * 8), 112), ('link_overrun_errors', (ctypes.c_ubyte * 8), 120), ('reserved_at_80', (ctypes.c_ubyte * 16), 128), ('vl_15_dropped', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 128), 160), ('port_xmit_wait', (ctypes.c_ubyte * 32), 288)])
@c.record
class struct_mlx5_ifc_ib_ext_port_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  reserved_at_0: 'list[int]'
  port_xmit_data_high: 'list[int]'
  port_xmit_data_low: 'list[int]'
  port_rcv_data_high: 'list[int]'
  port_rcv_data_low: 'list[int]'
  port_xmit_pkts_high: 'list[int]'
  port_xmit_pkts_low: 'list[int]'
  port_rcv_pkts_high: 'list[int]'
  port_rcv_pkts_low: 'list[int]'
  reserved_at_400: 'list[int]'
  port_unicast_xmit_pkts_high: 'list[int]'
  port_unicast_xmit_pkts_low: 'list[int]'
  port_multicast_xmit_pkts_high: 'list[int]'
  port_multicast_xmit_pkts_low: 'list[int]'
  port_unicast_rcv_pkts_high: 'list[int]'
  port_unicast_rcv_pkts_low: 'list[int]'
  port_multicast_rcv_pkts_high: 'list[int]'
  port_multicast_rcv_pkts_low: 'list[int]'
  reserved_at_580: 'list[int]'
struct_mlx5_ifc_ib_ext_port_cntrs_grp_data_layout_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 768), 0), ('port_xmit_data_high', (ctypes.c_ubyte * 32), 768), ('port_xmit_data_low', (ctypes.c_ubyte * 32), 800), ('port_rcv_data_high', (ctypes.c_ubyte * 32), 832), ('port_rcv_data_low', (ctypes.c_ubyte * 32), 864), ('port_xmit_pkts_high', (ctypes.c_ubyte * 32), 896), ('port_xmit_pkts_low', (ctypes.c_ubyte * 32), 928), ('port_rcv_pkts_high', (ctypes.c_ubyte * 32), 960), ('port_rcv_pkts_low', (ctypes.c_ubyte * 32), 992), ('reserved_at_400', (ctypes.c_ubyte * 128), 1024), ('port_unicast_xmit_pkts_high', (ctypes.c_ubyte * 32), 1152), ('port_unicast_xmit_pkts_low', (ctypes.c_ubyte * 32), 1184), ('port_multicast_xmit_pkts_high', (ctypes.c_ubyte * 32), 1216), ('port_multicast_xmit_pkts_low', (ctypes.c_ubyte * 32), 1248), ('port_unicast_rcv_pkts_high', (ctypes.c_ubyte * 32), 1280), ('port_unicast_rcv_pkts_low', (ctypes.c_ubyte * 32), 1312), ('port_multicast_rcv_pkts_high', (ctypes.c_ubyte * 32), 1344), ('port_multicast_rcv_pkts_low', (ctypes.c_ubyte * 32), 1376), ('reserved_at_580', (ctypes.c_ubyte * 576), 1408)])
@c.record
class struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  transmit_queue_high: 'list[int]'
  transmit_queue_low: 'list[int]'
  no_buffer_discard_uc_high: 'list[int]'
  no_buffer_discard_uc_low: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits.register_fields([('transmit_queue_high', (ctypes.c_ubyte * 32), 0), ('transmit_queue_low', (ctypes.c_ubyte * 32), 32), ('no_buffer_discard_uc_high', (ctypes.c_ubyte * 32), 64), ('no_buffer_discard_uc_low', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 1856), 128)])
@c.record
class struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  wred_discard_high: 'list[int]'
  wred_discard_low: 'list[int]'
  ecn_marked_tc_high: 'list[int]'
  ecn_marked_tc_low: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits.register_fields([('wred_discard_high', (ctypes.c_ubyte * 32), 0), ('wred_discard_low', (ctypes.c_ubyte * 32), 32), ('ecn_marked_tc_high', (ctypes.c_ubyte * 32), 64), ('ecn_marked_tc_low', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 1856), 128)])
@c.record
class struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  rx_octets_high: 'list[int]'
  rx_octets_low: 'list[int]'
  reserved_at_40: 'list[int]'
  rx_frames_high: 'list[int]'
  rx_frames_low: 'list[int]'
  tx_octets_high: 'list[int]'
  tx_octets_low: 'list[int]'
  reserved_at_180: 'list[int]'
  tx_frames_high: 'list[int]'
  tx_frames_low: 'list[int]'
  rx_pause_high: 'list[int]'
  rx_pause_low: 'list[int]'
  rx_pause_duration_high: 'list[int]'
  rx_pause_duration_low: 'list[int]'
  tx_pause_high: 'list[int]'
  tx_pause_low: 'list[int]'
  tx_pause_duration_high: 'list[int]'
  tx_pause_duration_low: 'list[int]'
  rx_pause_transition_high: 'list[int]'
  rx_pause_transition_low: 'list[int]'
  rx_discards_high: 'list[int]'
  rx_discards_low: 'list[int]'
  device_stall_minor_watermark_cnt_high: 'list[int]'
  device_stall_minor_watermark_cnt_low: 'list[int]'
  device_stall_critical_watermark_cnt_high: 'list[int]'
  device_stall_critical_watermark_cnt_low: 'list[int]'
  reserved_at_480: 'list[int]'
struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits.register_fields([('rx_octets_high', (ctypes.c_ubyte * 32), 0), ('rx_octets_low', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('rx_frames_high', (ctypes.c_ubyte * 32), 256), ('rx_frames_low', (ctypes.c_ubyte * 32), 288), ('tx_octets_high', (ctypes.c_ubyte * 32), 320), ('tx_octets_low', (ctypes.c_ubyte * 32), 352), ('reserved_at_180', (ctypes.c_ubyte * 192), 384), ('tx_frames_high', (ctypes.c_ubyte * 32), 576), ('tx_frames_low', (ctypes.c_ubyte * 32), 608), ('rx_pause_high', (ctypes.c_ubyte * 32), 640), ('rx_pause_low', (ctypes.c_ubyte * 32), 672), ('rx_pause_duration_high', (ctypes.c_ubyte * 32), 704), ('rx_pause_duration_low', (ctypes.c_ubyte * 32), 736), ('tx_pause_high', (ctypes.c_ubyte * 32), 768), ('tx_pause_low', (ctypes.c_ubyte * 32), 800), ('tx_pause_duration_high', (ctypes.c_ubyte * 32), 832), ('tx_pause_duration_low', (ctypes.c_ubyte * 32), 864), ('rx_pause_transition_high', (ctypes.c_ubyte * 32), 896), ('rx_pause_transition_low', (ctypes.c_ubyte * 32), 928), ('rx_discards_high', (ctypes.c_ubyte * 32), 960), ('rx_discards_low', (ctypes.c_ubyte * 32), 992), ('device_stall_minor_watermark_cnt_high', (ctypes.c_ubyte * 32), 1024), ('device_stall_minor_watermark_cnt_low', (ctypes.c_ubyte * 32), 1056), ('device_stall_critical_watermark_cnt_high', (ctypes.c_ubyte * 32), 1088), ('device_stall_critical_watermark_cnt_low', (ctypes.c_ubyte * 32), 1120), ('reserved_at_480', (ctypes.c_ubyte * 832), 1152)])
@c.record
class struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  port_transmit_wait_high: 'list[int]'
  port_transmit_wait_low: 'list[int]'
  reserved_at_40: 'list[int]'
  rx_buffer_almost_full_high: 'list[int]'
  rx_buffer_almost_full_low: 'list[int]'
  rx_buffer_full_high: 'list[int]'
  rx_buffer_full_low: 'list[int]'
  rx_icrc_encapsulated_high: 'list[int]'
  rx_icrc_encapsulated_low: 'list[int]'
  reserved_at_200: 'list[int]'
struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits.register_fields([('port_transmit_wait_high', (ctypes.c_ubyte * 32), 0), ('port_transmit_wait_low', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 256), 64), ('rx_buffer_almost_full_high', (ctypes.c_ubyte * 32), 320), ('rx_buffer_almost_full_low', (ctypes.c_ubyte * 32), 352), ('rx_buffer_full_high', (ctypes.c_ubyte * 32), 384), ('rx_buffer_full_low', (ctypes.c_ubyte * 32), 416), ('rx_icrc_encapsulated_high', (ctypes.c_ubyte * 32), 448), ('rx_icrc_encapsulated_low', (ctypes.c_ubyte * 32), 480), ('reserved_at_200', (ctypes.c_ubyte * 1472), 512)])
@c.record
class struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  dot3stats_alignment_errors_high: 'list[int]'
  dot3stats_alignment_errors_low: 'list[int]'
  dot3stats_fcs_errors_high: 'list[int]'
  dot3stats_fcs_errors_low: 'list[int]'
  dot3stats_single_collision_frames_high: 'list[int]'
  dot3stats_single_collision_frames_low: 'list[int]'
  dot3stats_multiple_collision_frames_high: 'list[int]'
  dot3stats_multiple_collision_frames_low: 'list[int]'
  dot3stats_sqe_test_errors_high: 'list[int]'
  dot3stats_sqe_test_errors_low: 'list[int]'
  dot3stats_deferred_transmissions_high: 'list[int]'
  dot3stats_deferred_transmissions_low: 'list[int]'
  dot3stats_late_collisions_high: 'list[int]'
  dot3stats_late_collisions_low: 'list[int]'
  dot3stats_excessive_collisions_high: 'list[int]'
  dot3stats_excessive_collisions_low: 'list[int]'
  dot3stats_internal_mac_transmit_errors_high: 'list[int]'
  dot3stats_internal_mac_transmit_errors_low: 'list[int]'
  dot3stats_carrier_sense_errors_high: 'list[int]'
  dot3stats_carrier_sense_errors_low: 'list[int]'
  dot3stats_frame_too_longs_high: 'list[int]'
  dot3stats_frame_too_longs_low: 'list[int]'
  dot3stats_internal_mac_receive_errors_high: 'list[int]'
  dot3stats_internal_mac_receive_errors_low: 'list[int]'
  dot3stats_symbol_errors_high: 'list[int]'
  dot3stats_symbol_errors_low: 'list[int]'
  dot3control_in_unknown_opcodes_high: 'list[int]'
  dot3control_in_unknown_opcodes_low: 'list[int]'
  dot3in_pause_frames_high: 'list[int]'
  dot3in_pause_frames_low: 'list[int]'
  dot3out_pause_frames_high: 'list[int]'
  dot3out_pause_frames_low: 'list[int]'
  reserved_at_400: 'list[int]'
struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits.register_fields([('dot3stats_alignment_errors_high', (ctypes.c_ubyte * 32), 0), ('dot3stats_alignment_errors_low', (ctypes.c_ubyte * 32), 32), ('dot3stats_fcs_errors_high', (ctypes.c_ubyte * 32), 64), ('dot3stats_fcs_errors_low', (ctypes.c_ubyte * 32), 96), ('dot3stats_single_collision_frames_high', (ctypes.c_ubyte * 32), 128), ('dot3stats_single_collision_frames_low', (ctypes.c_ubyte * 32), 160), ('dot3stats_multiple_collision_frames_high', (ctypes.c_ubyte * 32), 192), ('dot3stats_multiple_collision_frames_low', (ctypes.c_ubyte * 32), 224), ('dot3stats_sqe_test_errors_high', (ctypes.c_ubyte * 32), 256), ('dot3stats_sqe_test_errors_low', (ctypes.c_ubyte * 32), 288), ('dot3stats_deferred_transmissions_high', (ctypes.c_ubyte * 32), 320), ('dot3stats_deferred_transmissions_low', (ctypes.c_ubyte * 32), 352), ('dot3stats_late_collisions_high', (ctypes.c_ubyte * 32), 384), ('dot3stats_late_collisions_low', (ctypes.c_ubyte * 32), 416), ('dot3stats_excessive_collisions_high', (ctypes.c_ubyte * 32), 448), ('dot3stats_excessive_collisions_low', (ctypes.c_ubyte * 32), 480), ('dot3stats_internal_mac_transmit_errors_high', (ctypes.c_ubyte * 32), 512), ('dot3stats_internal_mac_transmit_errors_low', (ctypes.c_ubyte * 32), 544), ('dot3stats_carrier_sense_errors_high', (ctypes.c_ubyte * 32), 576), ('dot3stats_carrier_sense_errors_low', (ctypes.c_ubyte * 32), 608), ('dot3stats_frame_too_longs_high', (ctypes.c_ubyte * 32), 640), ('dot3stats_frame_too_longs_low', (ctypes.c_ubyte * 32), 672), ('dot3stats_internal_mac_receive_errors_high', (ctypes.c_ubyte * 32), 704), ('dot3stats_internal_mac_receive_errors_low', (ctypes.c_ubyte * 32), 736), ('dot3stats_symbol_errors_high', (ctypes.c_ubyte * 32), 768), ('dot3stats_symbol_errors_low', (ctypes.c_ubyte * 32), 800), ('dot3control_in_unknown_opcodes_high', (ctypes.c_ubyte * 32), 832), ('dot3control_in_unknown_opcodes_low', (ctypes.c_ubyte * 32), 864), ('dot3in_pause_frames_high', (ctypes.c_ubyte * 32), 896), ('dot3in_pause_frames_low', (ctypes.c_ubyte * 32), 928), ('dot3out_pause_frames_high', (ctypes.c_ubyte * 32), 960), ('dot3out_pause_frames_low', (ctypes.c_ubyte * 32), 992), ('reserved_at_400', (ctypes.c_ubyte * 960), 1024)])
@c.record
class struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  ether_stats_drop_events_high: 'list[int]'
  ether_stats_drop_events_low: 'list[int]'
  ether_stats_octets_high: 'list[int]'
  ether_stats_octets_low: 'list[int]'
  ether_stats_pkts_high: 'list[int]'
  ether_stats_pkts_low: 'list[int]'
  ether_stats_broadcast_pkts_high: 'list[int]'
  ether_stats_broadcast_pkts_low: 'list[int]'
  ether_stats_multicast_pkts_high: 'list[int]'
  ether_stats_multicast_pkts_low: 'list[int]'
  ether_stats_crc_align_errors_high: 'list[int]'
  ether_stats_crc_align_errors_low: 'list[int]'
  ether_stats_undersize_pkts_high: 'list[int]'
  ether_stats_undersize_pkts_low: 'list[int]'
  ether_stats_oversize_pkts_high: 'list[int]'
  ether_stats_oversize_pkts_low: 'list[int]'
  ether_stats_fragments_high: 'list[int]'
  ether_stats_fragments_low: 'list[int]'
  ether_stats_jabbers_high: 'list[int]'
  ether_stats_jabbers_low: 'list[int]'
  ether_stats_collisions_high: 'list[int]'
  ether_stats_collisions_low: 'list[int]'
  ether_stats_pkts64octets_high: 'list[int]'
  ether_stats_pkts64octets_low: 'list[int]'
  ether_stats_pkts65to127octets_high: 'list[int]'
  ether_stats_pkts65to127octets_low: 'list[int]'
  ether_stats_pkts128to255octets_high: 'list[int]'
  ether_stats_pkts128to255octets_low: 'list[int]'
  ether_stats_pkts256to511octets_high: 'list[int]'
  ether_stats_pkts256to511octets_low: 'list[int]'
  ether_stats_pkts512to1023octets_high: 'list[int]'
  ether_stats_pkts512to1023octets_low: 'list[int]'
  ether_stats_pkts1024to1518octets_high: 'list[int]'
  ether_stats_pkts1024to1518octets_low: 'list[int]'
  ether_stats_pkts1519to2047octets_high: 'list[int]'
  ether_stats_pkts1519to2047octets_low: 'list[int]'
  ether_stats_pkts2048to4095octets_high: 'list[int]'
  ether_stats_pkts2048to4095octets_low: 'list[int]'
  ether_stats_pkts4096to8191octets_high: 'list[int]'
  ether_stats_pkts4096to8191octets_low: 'list[int]'
  ether_stats_pkts8192to10239octets_high: 'list[int]'
  ether_stats_pkts8192to10239octets_low: 'list[int]'
  reserved_at_540: 'list[int]'
struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits.register_fields([('ether_stats_drop_events_high', (ctypes.c_ubyte * 32), 0), ('ether_stats_drop_events_low', (ctypes.c_ubyte * 32), 32), ('ether_stats_octets_high', (ctypes.c_ubyte * 32), 64), ('ether_stats_octets_low', (ctypes.c_ubyte * 32), 96), ('ether_stats_pkts_high', (ctypes.c_ubyte * 32), 128), ('ether_stats_pkts_low', (ctypes.c_ubyte * 32), 160), ('ether_stats_broadcast_pkts_high', (ctypes.c_ubyte * 32), 192), ('ether_stats_broadcast_pkts_low', (ctypes.c_ubyte * 32), 224), ('ether_stats_multicast_pkts_high', (ctypes.c_ubyte * 32), 256), ('ether_stats_multicast_pkts_low', (ctypes.c_ubyte * 32), 288), ('ether_stats_crc_align_errors_high', (ctypes.c_ubyte * 32), 320), ('ether_stats_crc_align_errors_low', (ctypes.c_ubyte * 32), 352), ('ether_stats_undersize_pkts_high', (ctypes.c_ubyte * 32), 384), ('ether_stats_undersize_pkts_low', (ctypes.c_ubyte * 32), 416), ('ether_stats_oversize_pkts_high', (ctypes.c_ubyte * 32), 448), ('ether_stats_oversize_pkts_low', (ctypes.c_ubyte * 32), 480), ('ether_stats_fragments_high', (ctypes.c_ubyte * 32), 512), ('ether_stats_fragments_low', (ctypes.c_ubyte * 32), 544), ('ether_stats_jabbers_high', (ctypes.c_ubyte * 32), 576), ('ether_stats_jabbers_low', (ctypes.c_ubyte * 32), 608), ('ether_stats_collisions_high', (ctypes.c_ubyte * 32), 640), ('ether_stats_collisions_low', (ctypes.c_ubyte * 32), 672), ('ether_stats_pkts64octets_high', (ctypes.c_ubyte * 32), 704), ('ether_stats_pkts64octets_low', (ctypes.c_ubyte * 32), 736), ('ether_stats_pkts65to127octets_high', (ctypes.c_ubyte * 32), 768), ('ether_stats_pkts65to127octets_low', (ctypes.c_ubyte * 32), 800), ('ether_stats_pkts128to255octets_high', (ctypes.c_ubyte * 32), 832), ('ether_stats_pkts128to255octets_low', (ctypes.c_ubyte * 32), 864), ('ether_stats_pkts256to511octets_high', (ctypes.c_ubyte * 32), 896), ('ether_stats_pkts256to511octets_low', (ctypes.c_ubyte * 32), 928), ('ether_stats_pkts512to1023octets_high', (ctypes.c_ubyte * 32), 960), ('ether_stats_pkts512to1023octets_low', (ctypes.c_ubyte * 32), 992), ('ether_stats_pkts1024to1518octets_high', (ctypes.c_ubyte * 32), 1024), ('ether_stats_pkts1024to1518octets_low', (ctypes.c_ubyte * 32), 1056), ('ether_stats_pkts1519to2047octets_high', (ctypes.c_ubyte * 32), 1088), ('ether_stats_pkts1519to2047octets_low', (ctypes.c_ubyte * 32), 1120), ('ether_stats_pkts2048to4095octets_high', (ctypes.c_ubyte * 32), 1152), ('ether_stats_pkts2048to4095octets_low', (ctypes.c_ubyte * 32), 1184), ('ether_stats_pkts4096to8191octets_high', (ctypes.c_ubyte * 32), 1216), ('ether_stats_pkts4096to8191octets_low', (ctypes.c_ubyte * 32), 1248), ('ether_stats_pkts8192to10239octets_high', (ctypes.c_ubyte * 32), 1280), ('ether_stats_pkts8192to10239octets_low', (ctypes.c_ubyte * 32), 1312), ('reserved_at_540', (ctypes.c_ubyte * 640), 1344)])
@c.record
class struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  if_in_octets_high: 'list[int]'
  if_in_octets_low: 'list[int]'
  if_in_ucast_pkts_high: 'list[int]'
  if_in_ucast_pkts_low: 'list[int]'
  if_in_discards_high: 'list[int]'
  if_in_discards_low: 'list[int]'
  if_in_errors_high: 'list[int]'
  if_in_errors_low: 'list[int]'
  if_in_unknown_protos_high: 'list[int]'
  if_in_unknown_protos_low: 'list[int]'
  if_out_octets_high: 'list[int]'
  if_out_octets_low: 'list[int]'
  if_out_ucast_pkts_high: 'list[int]'
  if_out_ucast_pkts_low: 'list[int]'
  if_out_discards_high: 'list[int]'
  if_out_discards_low: 'list[int]'
  if_out_errors_high: 'list[int]'
  if_out_errors_low: 'list[int]'
  if_in_multicast_pkts_high: 'list[int]'
  if_in_multicast_pkts_low: 'list[int]'
  if_in_broadcast_pkts_high: 'list[int]'
  if_in_broadcast_pkts_low: 'list[int]'
  if_out_multicast_pkts_high: 'list[int]'
  if_out_multicast_pkts_low: 'list[int]'
  if_out_broadcast_pkts_high: 'list[int]'
  if_out_broadcast_pkts_low: 'list[int]'
  reserved_at_340: 'list[int]'
struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits.register_fields([('if_in_octets_high', (ctypes.c_ubyte * 32), 0), ('if_in_octets_low', (ctypes.c_ubyte * 32), 32), ('if_in_ucast_pkts_high', (ctypes.c_ubyte * 32), 64), ('if_in_ucast_pkts_low', (ctypes.c_ubyte * 32), 96), ('if_in_discards_high', (ctypes.c_ubyte * 32), 128), ('if_in_discards_low', (ctypes.c_ubyte * 32), 160), ('if_in_errors_high', (ctypes.c_ubyte * 32), 192), ('if_in_errors_low', (ctypes.c_ubyte * 32), 224), ('if_in_unknown_protos_high', (ctypes.c_ubyte * 32), 256), ('if_in_unknown_protos_low', (ctypes.c_ubyte * 32), 288), ('if_out_octets_high', (ctypes.c_ubyte * 32), 320), ('if_out_octets_low', (ctypes.c_ubyte * 32), 352), ('if_out_ucast_pkts_high', (ctypes.c_ubyte * 32), 384), ('if_out_ucast_pkts_low', (ctypes.c_ubyte * 32), 416), ('if_out_discards_high', (ctypes.c_ubyte * 32), 448), ('if_out_discards_low', (ctypes.c_ubyte * 32), 480), ('if_out_errors_high', (ctypes.c_ubyte * 32), 512), ('if_out_errors_low', (ctypes.c_ubyte * 32), 544), ('if_in_multicast_pkts_high', (ctypes.c_ubyte * 32), 576), ('if_in_multicast_pkts_low', (ctypes.c_ubyte * 32), 608), ('if_in_broadcast_pkts_high', (ctypes.c_ubyte * 32), 640), ('if_in_broadcast_pkts_low', (ctypes.c_ubyte * 32), 672), ('if_out_multicast_pkts_high', (ctypes.c_ubyte * 32), 704), ('if_out_multicast_pkts_low', (ctypes.c_ubyte * 32), 736), ('if_out_broadcast_pkts_high', (ctypes.c_ubyte * 32), 768), ('if_out_broadcast_pkts_low', (ctypes.c_ubyte * 32), 800), ('reserved_at_340', (ctypes.c_ubyte * 1152), 832)])
@c.record
class struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  a_frames_transmitted_ok_high: 'list[int]'
  a_frames_transmitted_ok_low: 'list[int]'
  a_frames_received_ok_high: 'list[int]'
  a_frames_received_ok_low: 'list[int]'
  a_frame_check_sequence_errors_high: 'list[int]'
  a_frame_check_sequence_errors_low: 'list[int]'
  a_alignment_errors_high: 'list[int]'
  a_alignment_errors_low: 'list[int]'
  a_octets_transmitted_ok_high: 'list[int]'
  a_octets_transmitted_ok_low: 'list[int]'
  a_octets_received_ok_high: 'list[int]'
  a_octets_received_ok_low: 'list[int]'
  a_multicast_frames_xmitted_ok_high: 'list[int]'
  a_multicast_frames_xmitted_ok_low: 'list[int]'
  a_broadcast_frames_xmitted_ok_high: 'list[int]'
  a_broadcast_frames_xmitted_ok_low: 'list[int]'
  a_multicast_frames_received_ok_high: 'list[int]'
  a_multicast_frames_received_ok_low: 'list[int]'
  a_broadcast_frames_received_ok_high: 'list[int]'
  a_broadcast_frames_received_ok_low: 'list[int]'
  a_in_range_length_errors_high: 'list[int]'
  a_in_range_length_errors_low: 'list[int]'
  a_out_of_range_length_field_high: 'list[int]'
  a_out_of_range_length_field_low: 'list[int]'
  a_frame_too_long_errors_high: 'list[int]'
  a_frame_too_long_errors_low: 'list[int]'
  a_symbol_error_during_carrier_high: 'list[int]'
  a_symbol_error_during_carrier_low: 'list[int]'
  a_mac_control_frames_transmitted_high: 'list[int]'
  a_mac_control_frames_transmitted_low: 'list[int]'
  a_mac_control_frames_received_high: 'list[int]'
  a_mac_control_frames_received_low: 'list[int]'
  a_unsupported_opcodes_received_high: 'list[int]'
  a_unsupported_opcodes_received_low: 'list[int]'
  a_pause_mac_ctrl_frames_received_high: 'list[int]'
  a_pause_mac_ctrl_frames_received_low: 'list[int]'
  a_pause_mac_ctrl_frames_transmitted_high: 'list[int]'
  a_pause_mac_ctrl_frames_transmitted_low: 'list[int]'
  reserved_at_4c0: 'list[int]'
struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits.register_fields([('a_frames_transmitted_ok_high', (ctypes.c_ubyte * 32), 0), ('a_frames_transmitted_ok_low', (ctypes.c_ubyte * 32), 32), ('a_frames_received_ok_high', (ctypes.c_ubyte * 32), 64), ('a_frames_received_ok_low', (ctypes.c_ubyte * 32), 96), ('a_frame_check_sequence_errors_high', (ctypes.c_ubyte * 32), 128), ('a_frame_check_sequence_errors_low', (ctypes.c_ubyte * 32), 160), ('a_alignment_errors_high', (ctypes.c_ubyte * 32), 192), ('a_alignment_errors_low', (ctypes.c_ubyte * 32), 224), ('a_octets_transmitted_ok_high', (ctypes.c_ubyte * 32), 256), ('a_octets_transmitted_ok_low', (ctypes.c_ubyte * 32), 288), ('a_octets_received_ok_high', (ctypes.c_ubyte * 32), 320), ('a_octets_received_ok_low', (ctypes.c_ubyte * 32), 352), ('a_multicast_frames_xmitted_ok_high', (ctypes.c_ubyte * 32), 384), ('a_multicast_frames_xmitted_ok_low', (ctypes.c_ubyte * 32), 416), ('a_broadcast_frames_xmitted_ok_high', (ctypes.c_ubyte * 32), 448), ('a_broadcast_frames_xmitted_ok_low', (ctypes.c_ubyte * 32), 480), ('a_multicast_frames_received_ok_high', (ctypes.c_ubyte * 32), 512), ('a_multicast_frames_received_ok_low', (ctypes.c_ubyte * 32), 544), ('a_broadcast_frames_received_ok_high', (ctypes.c_ubyte * 32), 576), ('a_broadcast_frames_received_ok_low', (ctypes.c_ubyte * 32), 608), ('a_in_range_length_errors_high', (ctypes.c_ubyte * 32), 640), ('a_in_range_length_errors_low', (ctypes.c_ubyte * 32), 672), ('a_out_of_range_length_field_high', (ctypes.c_ubyte * 32), 704), ('a_out_of_range_length_field_low', (ctypes.c_ubyte * 32), 736), ('a_frame_too_long_errors_high', (ctypes.c_ubyte * 32), 768), ('a_frame_too_long_errors_low', (ctypes.c_ubyte * 32), 800), ('a_symbol_error_during_carrier_high', (ctypes.c_ubyte * 32), 832), ('a_symbol_error_during_carrier_low', (ctypes.c_ubyte * 32), 864), ('a_mac_control_frames_transmitted_high', (ctypes.c_ubyte * 32), 896), ('a_mac_control_frames_transmitted_low', (ctypes.c_ubyte * 32), 928), ('a_mac_control_frames_received_high', (ctypes.c_ubyte * 32), 960), ('a_mac_control_frames_received_low', (ctypes.c_ubyte * 32), 992), ('a_unsupported_opcodes_received_high', (ctypes.c_ubyte * 32), 1024), ('a_unsupported_opcodes_received_low', (ctypes.c_ubyte * 32), 1056), ('a_pause_mac_ctrl_frames_received_high', (ctypes.c_ubyte * 32), 1088), ('a_pause_mac_ctrl_frames_received_low', (ctypes.c_ubyte * 32), 1120), ('a_pause_mac_ctrl_frames_transmitted_high', (ctypes.c_ubyte * 32), 1152), ('a_pause_mac_ctrl_frames_transmitted_low', (ctypes.c_ubyte * 32), 1184), ('reserved_at_4c0', (ctypes.c_ubyte * 768), 1216)])
@c.record
class struct_mlx5_ifc_pcie_perf_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  life_time_counter_high: 'list[int]'
  life_time_counter_low: 'list[int]'
  rx_errors: 'list[int]'
  tx_errors: 'list[int]'
  l0_to_recovery_eieos: 'list[int]'
  l0_to_recovery_ts: 'list[int]'
  l0_to_recovery_framing: 'list[int]'
  l0_to_recovery_retrain: 'list[int]'
  crc_error_dllp: 'list[int]'
  crc_error_tlp: 'list[int]'
  tx_overflow_buffer_pkt_high: 'list[int]'
  tx_overflow_buffer_pkt_low: 'list[int]'
  outbound_stalled_reads: 'list[int]'
  outbound_stalled_writes: 'list[int]'
  outbound_stalled_reads_events: 'list[int]'
  outbound_stalled_writes_events: 'list[int]'
  reserved_at_200: 'list[int]'
struct_mlx5_ifc_pcie_perf_cntrs_grp_data_layout_bits.register_fields([('life_time_counter_high', (ctypes.c_ubyte * 32), 0), ('life_time_counter_low', (ctypes.c_ubyte * 32), 32), ('rx_errors', (ctypes.c_ubyte * 32), 64), ('tx_errors', (ctypes.c_ubyte * 32), 96), ('l0_to_recovery_eieos', (ctypes.c_ubyte * 32), 128), ('l0_to_recovery_ts', (ctypes.c_ubyte * 32), 160), ('l0_to_recovery_framing', (ctypes.c_ubyte * 32), 192), ('l0_to_recovery_retrain', (ctypes.c_ubyte * 32), 224), ('crc_error_dllp', (ctypes.c_ubyte * 32), 256), ('crc_error_tlp', (ctypes.c_ubyte * 32), 288), ('tx_overflow_buffer_pkt_high', (ctypes.c_ubyte * 32), 320), ('tx_overflow_buffer_pkt_low', (ctypes.c_ubyte * 32), 352), ('outbound_stalled_reads', (ctypes.c_ubyte * 32), 384), ('outbound_stalled_writes', (ctypes.c_ubyte * 32), 416), ('outbound_stalled_reads_events', (ctypes.c_ubyte * 32), 448), ('outbound_stalled_writes_events', (ctypes.c_ubyte * 32), 480), ('reserved_at_200', (ctypes.c_ubyte * 1472), 512)])
@c.record
class struct_mlx5_ifc_cmd_inter_comp_event_bits(c.Struct):
  SIZE = 224
  command_completion_vector: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_cmd_inter_comp_event_bits.register_fields([('command_completion_vector', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 192), 32)])
@c.record
class struct_mlx5_ifc_stall_vl_event_bits(c.Struct):
  SIZE = 192
  reserved_at_0: 'list[int]'
  port_num: 'list[int]'
  reserved_at_19: 'list[int]'
  vl: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_stall_vl_event_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 24), 0), ('port_num', (ctypes.c_ubyte * 1), 24), ('reserved_at_19', (ctypes.c_ubyte * 3), 25), ('vl', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 160), 32)])
@c.record
class struct_mlx5_ifc_db_bf_congestion_event_bits(c.Struct):
  SIZE = 192
  event_subtype: 'list[int]'
  reserved_at_8: 'list[int]'
  congestion_level: 'list[int]'
  reserved_at_18: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_db_bf_congestion_event_bits.register_fields([('event_subtype', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 8), 8), ('congestion_level', (ctypes.c_ubyte * 8), 16), ('reserved_at_18', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 160), 32)])
@c.record
class struct_mlx5_ifc_gpio_event_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
  gpio_event_hi: 'list[int]'
  gpio_event_lo: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_gpio_event_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 96), 0), ('gpio_event_hi', (ctypes.c_ubyte * 32), 96), ('gpio_event_lo', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 64), 160)])
@c.record
class struct_mlx5_ifc_port_state_change_event_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
  port_num: 'list[int]'
  reserved_at_44: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_port_state_change_event_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 64), 0), ('port_num', (ctypes.c_ubyte * 4), 64), ('reserved_at_44', (ctypes.c_ubyte * 28), 68), ('reserved_at_60', (ctypes.c_ubyte * 128), 96)])
@c.record
class struct_mlx5_ifc_dropped_packet_logged_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
struct_mlx5_ifc_dropped_packet_logged_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 224), 0)])
@c.record
class struct_mlx5_ifc_nic_cap_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  vhca_icm_ctrl: 'list[int]'
  reserved_at_1b: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_nic_cap_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 26), 0), ('vhca_icm_ctrl', (ctypes.c_ubyte * 1), 26), ('reserved_at_1b', (ctypes.c_ubyte * 5), 27), ('reserved_at_20', (ctypes.c_ubyte * 96), 32)])
@c.record
class struct_mlx5_ifc_default_timeout_bits(c.Struct):
  SIZE = 32
  to_multiplier: 'list[int]'
  reserved_at_3: 'list[int]'
  to_value: 'list[int]'
struct_mlx5_ifc_default_timeout_bits.register_fields([('to_multiplier', (ctypes.c_ubyte * 3), 0), ('reserved_at_3', (ctypes.c_ubyte * 9), 3), ('to_value', (ctypes.c_ubyte * 20), 12)])
@c.record
class struct_mlx5_ifc_dtor_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  pcie_toggle_to: 'struct_mlx5_ifc_default_timeout_bits'
  reserved_at_40: 'list[int]'
  health_poll_to: 'struct_mlx5_ifc_default_timeout_bits'
  full_crdump_to: 'struct_mlx5_ifc_default_timeout_bits'
  fw_reset_to: 'struct_mlx5_ifc_default_timeout_bits'
  flush_on_err_to: 'struct_mlx5_ifc_default_timeout_bits'
  pci_sync_update_to: 'struct_mlx5_ifc_default_timeout_bits'
  tear_down_to: 'struct_mlx5_ifc_default_timeout_bits'
  fsm_reactivate_to: 'struct_mlx5_ifc_default_timeout_bits'
  reclaim_pages_to: 'struct_mlx5_ifc_default_timeout_bits'
  reclaim_vfs_pages_to: 'struct_mlx5_ifc_default_timeout_bits'
  reset_unload_to: 'struct_mlx5_ifc_default_timeout_bits'
  reserved_at_1c0: 'list[int]'
struct_mlx5_ifc_dtor_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('pcie_toggle_to', struct_mlx5_ifc_default_timeout_bits, 32), ('reserved_at_40', (ctypes.c_ubyte * 96), 64), ('health_poll_to', struct_mlx5_ifc_default_timeout_bits, 160), ('full_crdump_to', struct_mlx5_ifc_default_timeout_bits, 192), ('fw_reset_to', struct_mlx5_ifc_default_timeout_bits, 224), ('flush_on_err_to', struct_mlx5_ifc_default_timeout_bits, 256), ('pci_sync_update_to', struct_mlx5_ifc_default_timeout_bits, 288), ('tear_down_to', struct_mlx5_ifc_default_timeout_bits, 320), ('fsm_reactivate_to', struct_mlx5_ifc_default_timeout_bits, 352), ('reclaim_pages_to', struct_mlx5_ifc_default_timeout_bits, 384), ('reclaim_vfs_pages_to', struct_mlx5_ifc_default_timeout_bits, 416), ('reset_unload_to', struct_mlx5_ifc_default_timeout_bits, 448), ('reserved_at_1c0', (ctypes.c_ubyte * 32), 480)])
@c.record
class struct_mlx5_ifc_vhca_icm_ctrl_reg_bits(c.Struct):
  SIZE = 512
  vhca_id_valid: 'list[int]'
  reserved_at_1: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_20: 'list[int]'
  cur_alloc_icm: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_vhca_icm_ctrl_reg_bits.register_fields([('vhca_id_valid', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 15), 1), ('vhca_id', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 160), 32), ('cur_alloc_icm', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 288), 224)])
_anonenum36: dict[int, str] = {(MLX5_CQ_ERROR_SYNDROME_CQ_OVERRUN:=1): 'MLX5_CQ_ERROR_SYNDROME_CQ_OVERRUN', (MLX5_CQ_ERROR_SYNDROME_CQ_ACCESS_VIOLATION_ERROR:=2): 'MLX5_CQ_ERROR_SYNDROME_CQ_ACCESS_VIOLATION_ERROR'}
@c.record
class struct_mlx5_ifc_cq_error_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
  cqn: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_40: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_cq_error_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('cqn', (ctypes.c_ubyte * 24), 8), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('syndrome', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 128), 96)])
@c.record
class struct_mlx5_ifc_rdma_page_fault_event_bits(c.Struct):
  SIZE = 224
  bytes_committed: 'list[int]'
  r_key: 'list[int]'
  reserved_at_40: 'list[int]'
  packet_len: 'list[int]'
  rdma_op_len: 'list[int]'
  rdma_va: 'list[int]'
  reserved_at_c0: 'list[int]'
  rdma: 'list[int]'
  write: 'list[int]'
  requestor: 'list[int]'
  qp_number: 'list[int]'
struct_mlx5_ifc_rdma_page_fault_event_bits.register_fields([('bytes_committed', (ctypes.c_ubyte * 32), 0), ('r_key', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('packet_len', (ctypes.c_ubyte * 16), 80), ('rdma_op_len', (ctypes.c_ubyte * 32), 96), ('rdma_va', (ctypes.c_ubyte * 64), 128), ('reserved_at_c0', (ctypes.c_ubyte * 5), 192), ('rdma', (ctypes.c_ubyte * 1), 197), ('write', (ctypes.c_ubyte * 1), 198), ('requestor', (ctypes.c_ubyte * 1), 199), ('qp_number', (ctypes.c_ubyte * 24), 200)])
@c.record
class struct_mlx5_ifc_wqe_associated_page_fault_event_bits(c.Struct):
  SIZE = 224
  bytes_committed: 'list[int]'
  reserved_at_20: 'list[int]'
  wqe_index: 'list[int]'
  reserved_at_40: 'list[int]'
  len: 'list[int]'
  reserved_at_60: 'list[int]'
  reserved_at_c0: 'list[int]'
  rdma: 'list[int]'
  write_read: 'list[int]'
  requestor: 'list[int]'
  qpn: 'list[int]'
struct_mlx5_ifc_wqe_associated_page_fault_event_bits.register_fields([('bytes_committed', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('wqe_index', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('len', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 96), 96), ('reserved_at_c0', (ctypes.c_ubyte * 5), 192), ('rdma', (ctypes.c_ubyte * 1), 197), ('write_read', (ctypes.c_ubyte * 1), 198), ('requestor', (ctypes.c_ubyte * 1), 199), ('qpn', (ctypes.c_ubyte * 24), 200)])
@c.record
class struct_mlx5_ifc_qp_events_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
  type: 'list[int]'
  reserved_at_a8: 'list[int]'
  reserved_at_c0: 'list[int]'
  qpn_rqn_sqn: 'list[int]'
struct_mlx5_ifc_qp_events_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 160), 0), ('type', (ctypes.c_ubyte * 8), 160), ('reserved_at_a8', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('qpn_rqn_sqn', (ctypes.c_ubyte * 24), 200)])
@c.record
class struct_mlx5_ifc_dct_events_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
  reserved_at_c0: 'list[int]'
  dct_number: 'list[int]'
struct_mlx5_ifc_dct_events_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 192), 0), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('dct_number', (ctypes.c_ubyte * 24), 200)])
@c.record
class struct_mlx5_ifc_comp_event_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
  reserved_at_c0: 'list[int]'
  cq_number: 'list[int]'
struct_mlx5_ifc_comp_event_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 192), 0), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('cq_number', (ctypes.c_ubyte * 24), 200)])
_anonenum37: dict[int, str] = {(MLX5_QPC_STATE_RST:=0): 'MLX5_QPC_STATE_RST', (MLX5_QPC_STATE_INIT:=1): 'MLX5_QPC_STATE_INIT', (MLX5_QPC_STATE_RTR:=2): 'MLX5_QPC_STATE_RTR', (MLX5_QPC_STATE_RTS:=3): 'MLX5_QPC_STATE_RTS', (MLX5_QPC_STATE_SQER:=4): 'MLX5_QPC_STATE_SQER', (MLX5_QPC_STATE_ERR:=6): 'MLX5_QPC_STATE_ERR', (MLX5_QPC_STATE_SQD:=7): 'MLX5_QPC_STATE_SQD', (MLX5_QPC_STATE_SUSPENDED:=9): 'MLX5_QPC_STATE_SUSPENDED'}
_anonenum38: dict[int, str] = {(MLX5_QPC_ST_RC:=0): 'MLX5_QPC_ST_RC', (MLX5_QPC_ST_UC:=1): 'MLX5_QPC_ST_UC', (MLX5_QPC_ST_UD:=2): 'MLX5_QPC_ST_UD', (MLX5_QPC_ST_XRC:=3): 'MLX5_QPC_ST_XRC', (MLX5_QPC_ST_DCI:=5): 'MLX5_QPC_ST_DCI', (MLX5_QPC_ST_QP0:=7): 'MLX5_QPC_ST_QP0', (MLX5_QPC_ST_QP1:=8): 'MLX5_QPC_ST_QP1', (MLX5_QPC_ST_RAW_DATAGRAM:=9): 'MLX5_QPC_ST_RAW_DATAGRAM', (MLX5_QPC_ST_REG_UMR:=12): 'MLX5_QPC_ST_REG_UMR'}
_anonenum39: dict[int, str] = {(MLX5_QPC_PM_STATE_ARMED:=0): 'MLX5_QPC_PM_STATE_ARMED', (MLX5_QPC_PM_STATE_REARM:=1): 'MLX5_QPC_PM_STATE_REARM', (MLX5_QPC_PM_STATE_RESERVED:=2): 'MLX5_QPC_PM_STATE_RESERVED', (MLX5_QPC_PM_STATE_MIGRATED:=3): 'MLX5_QPC_PM_STATE_MIGRATED'}
_anonenum40: dict[int, str] = {(MLX5_QPC_OFFLOAD_TYPE_RNDV:=1): 'MLX5_QPC_OFFLOAD_TYPE_RNDV'}
_anonenum41: dict[int, str] = {(MLX5_QPC_END_PADDING_MODE_SCATTER_AS_IS:=0): 'MLX5_QPC_END_PADDING_MODE_SCATTER_AS_IS', (MLX5_QPC_END_PADDING_MODE_PAD_TO_CACHE_LINE_ALIGNMENT:=1): 'MLX5_QPC_END_PADDING_MODE_PAD_TO_CACHE_LINE_ALIGNMENT'}
_anonenum42: dict[int, str] = {(MLX5_QPC_MTU_256_BYTES:=1): 'MLX5_QPC_MTU_256_BYTES', (MLX5_QPC_MTU_512_BYTES:=2): 'MLX5_QPC_MTU_512_BYTES', (MLX5_QPC_MTU_1K_BYTES:=3): 'MLX5_QPC_MTU_1K_BYTES', (MLX5_QPC_MTU_2K_BYTES:=4): 'MLX5_QPC_MTU_2K_BYTES', (MLX5_QPC_MTU_4K_BYTES:=5): 'MLX5_QPC_MTU_4K_BYTES', (MLX5_QPC_MTU_RAW_ETHERNET_QP:=7): 'MLX5_QPC_MTU_RAW_ETHERNET_QP'}
_anonenum43: dict[int, str] = {(MLX5_QPC_ATOMIC_MODE_IB_SPEC:=1): 'MLX5_QPC_ATOMIC_MODE_IB_SPEC', (MLX5_QPC_ATOMIC_MODE_ONLY_8B:=2): 'MLX5_QPC_ATOMIC_MODE_ONLY_8B', (MLX5_QPC_ATOMIC_MODE_UP_TO_8B:=3): 'MLX5_QPC_ATOMIC_MODE_UP_TO_8B', (MLX5_QPC_ATOMIC_MODE_UP_TO_16B:=4): 'MLX5_QPC_ATOMIC_MODE_UP_TO_16B', (MLX5_QPC_ATOMIC_MODE_UP_TO_32B:=5): 'MLX5_QPC_ATOMIC_MODE_UP_TO_32B', (MLX5_QPC_ATOMIC_MODE_UP_TO_64B:=6): 'MLX5_QPC_ATOMIC_MODE_UP_TO_64B', (MLX5_QPC_ATOMIC_MODE_UP_TO_128B:=7): 'MLX5_QPC_ATOMIC_MODE_UP_TO_128B', (MLX5_QPC_ATOMIC_MODE_UP_TO_256B:=8): 'MLX5_QPC_ATOMIC_MODE_UP_TO_256B'}
_anonenum44: dict[int, str] = {(MLX5_QPC_CS_REQ_DISABLE:=0): 'MLX5_QPC_CS_REQ_DISABLE', (MLX5_QPC_CS_REQ_UP_TO_32B:=17): 'MLX5_QPC_CS_REQ_UP_TO_32B', (MLX5_QPC_CS_REQ_UP_TO_64B:=34): 'MLX5_QPC_CS_REQ_UP_TO_64B'}
_anonenum45: dict[int, str] = {(MLX5_QPC_CS_RES_DISABLE:=0): 'MLX5_QPC_CS_RES_DISABLE', (MLX5_QPC_CS_RES_UP_TO_32B:=1): 'MLX5_QPC_CS_RES_UP_TO_32B', (MLX5_QPC_CS_RES_UP_TO_64B:=2): 'MLX5_QPC_CS_RES_UP_TO_64B'}
_anonenum46: dict[int, str] = {(MLX5_TIMESTAMP_FORMAT_FREE_RUNNING:=0): 'MLX5_TIMESTAMP_FORMAT_FREE_RUNNING', (MLX5_TIMESTAMP_FORMAT_DEFAULT:=1): 'MLX5_TIMESTAMP_FORMAT_DEFAULT', (MLX5_TIMESTAMP_FORMAT_REAL_TIME:=2): 'MLX5_TIMESTAMP_FORMAT_REAL_TIME'}
@c.record
class struct_mlx5_ifc_qpc_bits(c.Struct):
  SIZE = 1856
  state: 'list[int]'
  lag_tx_port_affinity: 'list[int]'
  st: 'list[int]'
  reserved_at_10: 'list[int]'
  isolate_vl_tc: 'list[int]'
  pm_state: 'list[int]'
  reserved_at_15: 'list[int]'
  req_e2e_credit_mode: 'list[int]'
  offload_type: 'list[int]'
  end_padding_mode: 'list[int]'
  reserved_at_1e: 'list[int]'
  wq_signature: 'list[int]'
  block_lb_mc: 'list[int]'
  atomic_like_write_en: 'list[int]'
  latency_sensitive: 'list[int]'
  reserved_at_24: 'list[int]'
  drain_sigerr: 'list[int]'
  reserved_at_26: 'list[int]'
  dp_ordering_force: 'list[int]'
  pd: 'list[int]'
  mtu: 'list[int]'
  log_msg_max: 'list[int]'
  reserved_at_48: 'list[int]'
  log_rq_size: 'list[int]'
  log_rq_stride: 'list[int]'
  no_sq: 'list[int]'
  log_sq_size: 'list[int]'
  reserved_at_55: 'list[int]'
  retry_mode: 'list[int]'
  ts_format: 'list[int]'
  reserved_at_5a: 'list[int]'
  rlky: 'list[int]'
  ulp_stateless_offload_mode: 'list[int]'
  counter_set_id: 'list[int]'
  uar_page: 'list[int]'
  reserved_at_80: 'list[int]'
  user_index: 'list[int]'
  reserved_at_a0: 'list[int]'
  log_page_size: 'list[int]'
  remote_qpn: 'list[int]'
  primary_address_path: 'struct_mlx5_ifc_ads_bits'
  secondary_address_path: 'struct_mlx5_ifc_ads_bits'
  log_ack_req_freq: 'list[int]'
  reserved_at_384: 'list[int]'
  log_sra_max: 'list[int]'
  reserved_at_38b: 'list[int]'
  retry_count: 'list[int]'
  rnr_retry: 'list[int]'
  reserved_at_393: 'list[int]'
  fre: 'list[int]'
  cur_rnr_retry: 'list[int]'
  cur_retry_count: 'list[int]'
  reserved_at_39b: 'list[int]'
  reserved_at_3a0: 'list[int]'
  reserved_at_3c0: 'list[int]'
  next_send_psn: 'list[int]'
  reserved_at_3e0: 'list[int]'
  log_num_dci_stream_channels: 'list[int]'
  cqn_snd: 'list[int]'
  reserved_at_400: 'list[int]'
  log_num_dci_errored_streams: 'list[int]'
  deth_sqpn: 'list[int]'
  reserved_at_420: 'list[int]'
  reserved_at_440: 'list[int]'
  last_acked_psn: 'list[int]'
  reserved_at_460: 'list[int]'
  ssn: 'list[int]'
  reserved_at_480: 'list[int]'
  log_rra_max: 'list[int]'
  reserved_at_48b: 'list[int]'
  atomic_mode: 'list[int]'
  rre: 'list[int]'
  rwe: 'list[int]'
  rae: 'list[int]'
  reserved_at_493: 'list[int]'
  page_offset: 'list[int]'
  reserved_at_49a: 'list[int]'
  dp_ordering_1: 'list[int]'
  cd_slave_receive: 'list[int]'
  cd_slave_send: 'list[int]'
  cd_master: 'list[int]'
  reserved_at_4a0: 'list[int]'
  min_rnr_nak: 'list[int]'
  next_rcv_psn: 'list[int]'
  reserved_at_4c0: 'list[int]'
  xrcd: 'list[int]'
  reserved_at_4e0: 'list[int]'
  cqn_rcv: 'list[int]'
  dbr_addr: 'list[int]'
  q_key: 'list[int]'
  reserved_at_560: 'list[int]'
  rq_type: 'list[int]'
  srqn_rmpn_xrqn: 'list[int]'
  reserved_at_580: 'list[int]'
  rmsn: 'list[int]'
  hw_sq_wqebb_counter: 'list[int]'
  sw_sq_wqebb_counter: 'list[int]'
  hw_rq_counter: 'list[int]'
  sw_rq_counter: 'list[int]'
  reserved_at_600: 'list[int]'
  reserved_at_620: 'list[int]'
  cgs: 'list[int]'
  cs_req: 'list[int]'
  cs_res: 'list[int]'
  dc_access_key: 'list[int]'
  reserved_at_680: 'list[int]'
  dbr_umem_valid: 'list[int]'
  reserved_at_684: 'list[int]'
struct_mlx5_ifc_qpc_bits.register_fields([('state', (ctypes.c_ubyte * 4), 0), ('lag_tx_port_affinity', (ctypes.c_ubyte * 4), 4), ('st', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 2), 16), ('isolate_vl_tc', (ctypes.c_ubyte * 1), 18), ('pm_state', (ctypes.c_ubyte * 2), 19), ('reserved_at_15', (ctypes.c_ubyte * 1), 21), ('req_e2e_credit_mode', (ctypes.c_ubyte * 2), 22), ('offload_type', (ctypes.c_ubyte * 4), 24), ('end_padding_mode', (ctypes.c_ubyte * 2), 28), ('reserved_at_1e', (ctypes.c_ubyte * 2), 30), ('wq_signature', (ctypes.c_ubyte * 1), 32), ('block_lb_mc', (ctypes.c_ubyte * 1), 33), ('atomic_like_write_en', (ctypes.c_ubyte * 1), 34), ('latency_sensitive', (ctypes.c_ubyte * 1), 35), ('reserved_at_24', (ctypes.c_ubyte * 1), 36), ('drain_sigerr', (ctypes.c_ubyte * 1), 37), ('reserved_at_26', (ctypes.c_ubyte * 1), 38), ('dp_ordering_force', (ctypes.c_ubyte * 1), 39), ('pd', (ctypes.c_ubyte * 24), 40), ('mtu', (ctypes.c_ubyte * 3), 64), ('log_msg_max', (ctypes.c_ubyte * 5), 67), ('reserved_at_48', (ctypes.c_ubyte * 1), 72), ('log_rq_size', (ctypes.c_ubyte * 4), 73), ('log_rq_stride', (ctypes.c_ubyte * 3), 77), ('no_sq', (ctypes.c_ubyte * 1), 80), ('log_sq_size', (ctypes.c_ubyte * 4), 81), ('reserved_at_55', (ctypes.c_ubyte * 1), 85), ('retry_mode', (ctypes.c_ubyte * 2), 86), ('ts_format', (ctypes.c_ubyte * 2), 88), ('reserved_at_5a', (ctypes.c_ubyte * 1), 90), ('rlky', (ctypes.c_ubyte * 1), 91), ('ulp_stateless_offload_mode', (ctypes.c_ubyte * 4), 92), ('counter_set_id', (ctypes.c_ubyte * 8), 96), ('uar_page', (ctypes.c_ubyte * 24), 104), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('user_index', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 3), 160), ('log_page_size', (ctypes.c_ubyte * 5), 163), ('remote_qpn', (ctypes.c_ubyte * 24), 168), ('primary_address_path', struct_mlx5_ifc_ads_bits, 192), ('secondary_address_path', struct_mlx5_ifc_ads_bits, 544), ('log_ack_req_freq', (ctypes.c_ubyte * 4), 896), ('reserved_at_384', (ctypes.c_ubyte * 4), 900), ('log_sra_max', (ctypes.c_ubyte * 3), 904), ('reserved_at_38b', (ctypes.c_ubyte * 2), 907), ('retry_count', (ctypes.c_ubyte * 3), 909), ('rnr_retry', (ctypes.c_ubyte * 3), 912), ('reserved_at_393', (ctypes.c_ubyte * 1), 915), ('fre', (ctypes.c_ubyte * 1), 916), ('cur_rnr_retry', (ctypes.c_ubyte * 3), 917), ('cur_retry_count', (ctypes.c_ubyte * 3), 920), ('reserved_at_39b', (ctypes.c_ubyte * 5), 923), ('reserved_at_3a0', (ctypes.c_ubyte * 32), 928), ('reserved_at_3c0', (ctypes.c_ubyte * 8), 960), ('next_send_psn', (ctypes.c_ubyte * 24), 968), ('reserved_at_3e0', (ctypes.c_ubyte * 3), 992), ('log_num_dci_stream_channels', (ctypes.c_ubyte * 5), 995), ('cqn_snd', (ctypes.c_ubyte * 24), 1000), ('reserved_at_400', (ctypes.c_ubyte * 3), 1024), ('log_num_dci_errored_streams', (ctypes.c_ubyte * 5), 1027), ('deth_sqpn', (ctypes.c_ubyte * 24), 1032), ('reserved_at_420', (ctypes.c_ubyte * 32), 1056), ('reserved_at_440', (ctypes.c_ubyte * 8), 1088), ('last_acked_psn', (ctypes.c_ubyte * 24), 1096), ('reserved_at_460', (ctypes.c_ubyte * 8), 1120), ('ssn', (ctypes.c_ubyte * 24), 1128), ('reserved_at_480', (ctypes.c_ubyte * 8), 1152), ('log_rra_max', (ctypes.c_ubyte * 3), 1160), ('reserved_at_48b', (ctypes.c_ubyte * 1), 1163), ('atomic_mode', (ctypes.c_ubyte * 4), 1164), ('rre', (ctypes.c_ubyte * 1), 1168), ('rwe', (ctypes.c_ubyte * 1), 1169), ('rae', (ctypes.c_ubyte * 1), 1170), ('reserved_at_493', (ctypes.c_ubyte * 1), 1171), ('page_offset', (ctypes.c_ubyte * 6), 1172), ('reserved_at_49a', (ctypes.c_ubyte * 2), 1178), ('dp_ordering_1', (ctypes.c_ubyte * 1), 1180), ('cd_slave_receive', (ctypes.c_ubyte * 1), 1181), ('cd_slave_send', (ctypes.c_ubyte * 1), 1182), ('cd_master', (ctypes.c_ubyte * 1), 1183), ('reserved_at_4a0', (ctypes.c_ubyte * 3), 1184), ('min_rnr_nak', (ctypes.c_ubyte * 5), 1187), ('next_rcv_psn', (ctypes.c_ubyte * 24), 1192), ('reserved_at_4c0', (ctypes.c_ubyte * 8), 1216), ('xrcd', (ctypes.c_ubyte * 24), 1224), ('reserved_at_4e0', (ctypes.c_ubyte * 8), 1248), ('cqn_rcv', (ctypes.c_ubyte * 24), 1256), ('dbr_addr', (ctypes.c_ubyte * 64), 1280), ('q_key', (ctypes.c_ubyte * 32), 1344), ('reserved_at_560', (ctypes.c_ubyte * 5), 1376), ('rq_type', (ctypes.c_ubyte * 3), 1381), ('srqn_rmpn_xrqn', (ctypes.c_ubyte * 24), 1384), ('reserved_at_580', (ctypes.c_ubyte * 8), 1408), ('rmsn', (ctypes.c_ubyte * 24), 1416), ('hw_sq_wqebb_counter', (ctypes.c_ubyte * 16), 1440), ('sw_sq_wqebb_counter', (ctypes.c_ubyte * 16), 1456), ('hw_rq_counter', (ctypes.c_ubyte * 32), 1472), ('sw_rq_counter', (ctypes.c_ubyte * 32), 1504), ('reserved_at_600', (ctypes.c_ubyte * 32), 1536), ('reserved_at_620', (ctypes.c_ubyte * 15), 1568), ('cgs', (ctypes.c_ubyte * 1), 1583), ('cs_req', (ctypes.c_ubyte * 8), 1584), ('cs_res', (ctypes.c_ubyte * 8), 1592), ('dc_access_key', (ctypes.c_ubyte * 64), 1600), ('reserved_at_680', (ctypes.c_ubyte * 3), 1664), ('dbr_umem_valid', (ctypes.c_ubyte * 1), 1667), ('reserved_at_684', (ctypes.c_ubyte * 188), 1668)])
@c.record
class struct_mlx5_ifc_roce_addr_layout_bits(c.Struct):
  SIZE = 256
  source_l3_address: 'list[list[int]]'
  reserved_at_80: 'list[int]'
  vlan_valid: 'list[int]'
  vlan_id: 'list[int]'
  source_mac_47_32: 'list[int]'
  source_mac_31_0: 'list[int]'
  reserved_at_c0: 'list[int]'
  roce_l3_type: 'list[int]'
  roce_version: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_roce_addr_layout_bits.register_fields([('source_l3_address', ((ctypes.c_ubyte * 8) * 16), 0), ('reserved_at_80', (ctypes.c_ubyte * 3), 128), ('vlan_valid', (ctypes.c_ubyte * 1), 131), ('vlan_id', (ctypes.c_ubyte * 12), 132), ('source_mac_47_32', (ctypes.c_ubyte * 16), 144), ('source_mac_31_0', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 20), 192), ('roce_l3_type', (ctypes.c_ubyte * 4), 212), ('roce_version', (ctypes.c_ubyte * 8), 216), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_crypto_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  synchronize_dek: 'list[int]'
  int_kek_manual: 'list[int]'
  int_kek_auto: 'list[int]'
  reserved_at_6: 'list[int]'
  reserved_at_20: 'list[int]'
  log_dek_max_alloc: 'list[int]'
  reserved_at_28: 'list[int]'
  log_max_num_deks: 'list[int]'
  reserved_at_30: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  log_dek_granularity: 'list[int]'
  reserved_at_68: 'list[int]'
  log_max_num_int_kek: 'list[int]'
  sw_wrapped_dek: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_crypto_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 3), 0), ('synchronize_dek', (ctypes.c_ubyte * 1), 3), ('int_kek_manual', (ctypes.c_ubyte * 1), 4), ('int_kek_auto', (ctypes.c_ubyte * 1), 5), ('reserved_at_6', (ctypes.c_ubyte * 26), 6), ('reserved_at_20', (ctypes.c_ubyte * 3), 32), ('log_dek_max_alloc', (ctypes.c_ubyte * 5), 35), ('reserved_at_28', (ctypes.c_ubyte * 3), 40), ('log_max_num_deks', (ctypes.c_ubyte * 5), 43), ('reserved_at_30', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 3), 96), ('log_dek_granularity', (ctypes.c_ubyte * 5), 99), ('reserved_at_68', (ctypes.c_ubyte * 3), 104), ('log_max_num_int_kek', (ctypes.c_ubyte * 5), 107), ('sw_wrapped_dek', (ctypes.c_ubyte * 16), 112), ('reserved_at_80', (ctypes.c_ubyte * 1920), 128)])
@c.record
class struct_mlx5_ifc_shampo_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  shampo_log_max_reservation_size: 'list[int]'
  reserved_at_8: 'list[int]'
  shampo_log_min_reservation_size: 'list[int]'
  shampo_min_mss_size: 'list[int]'
  shampo_header_split: 'list[int]'
  shampo_header_split_data_merge: 'list[int]'
  reserved_at_22: 'list[int]'
  shampo_log_max_headers_entry_size: 'list[int]'
  reserved_at_28: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_shampo_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 3), 0), ('shampo_log_max_reservation_size', (ctypes.c_ubyte * 5), 3), ('reserved_at_8', (ctypes.c_ubyte * 3), 8), ('shampo_log_min_reservation_size', (ctypes.c_ubyte * 5), 11), ('shampo_min_mss_size', (ctypes.c_ubyte * 16), 16), ('shampo_header_split', (ctypes.c_ubyte * 1), 32), ('shampo_header_split_data_merge', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 1), 34), ('shampo_log_max_headers_entry_size', (ctypes.c_ubyte * 5), 35), ('reserved_at_28', (ctypes.c_ubyte * 24), 40), ('reserved_at_40', (ctypes.c_ubyte * 1984), 64)])
@c.record
class union_mlx5_ifc_hca_cap_union_bits(c.Struct):
  SIZE = 32768
  cmd_hca_cap: 'struct_mlx5_ifc_cmd_hca_cap_bits'
  cmd_hca_cap_2: 'struct_mlx5_ifc_cmd_hca_cap_2_bits'
  odp_cap: 'struct_mlx5_ifc_odp_cap_bits'
  atomic_caps: 'struct_mlx5_ifc_atomic_caps_bits'
  roce_cap: 'struct_mlx5_ifc_roce_cap_bits'
  per_protocol_networking_offload_caps: 'struct_mlx5_ifc_per_protocol_networking_offload_caps_bits'
  flow_table_nic_cap: 'struct_mlx5_ifc_flow_table_nic_cap_bits'
  flow_table_eswitch_cap: 'struct_mlx5_ifc_flow_table_eswitch_cap_bits'
  wqe_based_flow_table_cap: 'struct_mlx5_ifc_wqe_based_flow_table_cap_bits'
  esw_cap: 'struct_mlx5_ifc_esw_cap_bits'
  e_switch_cap: 'struct_mlx5_ifc_e_switch_cap_bits'
  port_selection_cap: 'struct_mlx5_ifc_port_selection_cap_bits'
  qos_cap: 'struct_mlx5_ifc_qos_cap_bits'
  debug_cap: 'struct_mlx5_ifc_debug_cap_bits'
  fpga_cap: 'struct_mlx5_ifc_fpga_cap_bits'
  tls_cap: 'struct_mlx5_ifc_tls_cap_bits'
  device_mem_cap: 'struct_mlx5_ifc_device_mem_cap_bits'
  virtio_emulation_cap: 'struct_mlx5_ifc_virtio_emulation_cap_bits'
  macsec_cap: 'struct_mlx5_ifc_macsec_cap_bits'
  crypto_cap: 'struct_mlx5_ifc_crypto_cap_bits'
  ipsec_cap: 'struct_mlx5_ifc_ipsec_cap_bits'
  psp_cap: 'struct_mlx5_ifc_psp_cap_bits'
  reserved_at_0: 'list[int]'
@c.record
class struct_mlx5_ifc_fpga_cap_bits(c.Struct):
  SIZE = 2048
  fpga_id: 'list[int]'
  fpga_device: 'list[int]'
  register_file_ver: 'list[int]'
  fpga_ctrl_modify: 'list[int]'
  reserved_at_41: 'list[int]'
  access_reg_query_mode: 'list[int]'
  reserved_at_48: 'list[int]'
  access_reg_modify_mode: 'list[int]'
  reserved_at_50: 'list[int]'
  reserved_at_60: 'list[int]'
  image_version: 'list[int]'
  image_date: 'list[int]'
  image_time: 'list[int]'
  shell_version: 'list[int]'
  reserved_at_100: 'list[int]'
  shell_caps: 'struct_mlx5_ifc_fpga_shell_caps_bits'
  reserved_at_380: 'list[int]'
  ieee_vendor_id: 'list[int]'
  sandbox_product_version: 'list[int]'
  sandbox_product_id: 'list[int]'
  sandbox_basic_caps: 'list[int]'
  reserved_at_3e0: 'list[int]'
  sandbox_extended_caps_len: 'list[int]'
  sandbox_extended_caps_addr: 'list[int]'
  fpga_ddr_start_addr: 'list[int]'
  fpga_cr_space_start_addr: 'list[int]'
  fpga_ddr_size: 'list[int]'
  fpga_cr_space_size: 'list[int]'
  reserved_at_500: 'list[int]'
@c.record
class struct_mlx5_ifc_fpga_shell_caps_bits(c.Struct):
  SIZE = 512
  max_num_qps: 'list[int]'
  reserved_at_10: 'list[int]'
  total_rcv_credits: 'list[int]'
  reserved_at_20: 'list[int]'
  qp_type: 'list[int]'
  reserved_at_30: 'list[int]'
  rae: 'list[int]'
  rwe: 'list[int]'
  rre: 'list[int]'
  reserved_at_38: 'list[int]'
  dc: 'list[int]'
  ud: 'list[int]'
  uc: 'list[int]'
  rc: 'list[int]'
  reserved_at_40: 'list[int]'
  log_ddr_size: 'list[int]'
  max_fpga_qp_msg_size: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_fpga_shell_caps_bits.register_fields([('max_num_qps', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('total_rcv_credits', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 14), 32), ('qp_type', (ctypes.c_ubyte * 2), 46), ('reserved_at_30', (ctypes.c_ubyte * 5), 48), ('rae', (ctypes.c_ubyte * 1), 53), ('rwe', (ctypes.c_ubyte * 1), 54), ('rre', (ctypes.c_ubyte * 1), 55), ('reserved_at_38', (ctypes.c_ubyte * 4), 56), ('dc', (ctypes.c_ubyte * 1), 60), ('ud', (ctypes.c_ubyte * 1), 61), ('uc', (ctypes.c_ubyte * 1), 62), ('rc', (ctypes.c_ubyte * 1), 63), ('reserved_at_40', (ctypes.c_ubyte * 26), 64), ('log_ddr_size', (ctypes.c_ubyte * 6), 90), ('max_fpga_qp_msg_size', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
struct_mlx5_ifc_fpga_cap_bits.register_fields([('fpga_id', (ctypes.c_ubyte * 8), 0), ('fpga_device', (ctypes.c_ubyte * 24), 8), ('register_file_ver', (ctypes.c_ubyte * 32), 32), ('fpga_ctrl_modify', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 5), 65), ('access_reg_query_mode', (ctypes.c_ubyte * 2), 70), ('reserved_at_48', (ctypes.c_ubyte * 6), 72), ('access_reg_modify_mode', (ctypes.c_ubyte * 2), 78), ('reserved_at_50', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('image_version', (ctypes.c_ubyte * 32), 128), ('image_date', (ctypes.c_ubyte * 32), 160), ('image_time', (ctypes.c_ubyte * 32), 192), ('shell_version', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 128), 256), ('shell_caps', struct_mlx5_ifc_fpga_shell_caps_bits, 384), ('reserved_at_380', (ctypes.c_ubyte * 8), 896), ('ieee_vendor_id', (ctypes.c_ubyte * 24), 904), ('sandbox_product_version', (ctypes.c_ubyte * 16), 928), ('sandbox_product_id', (ctypes.c_ubyte * 16), 944), ('sandbox_basic_caps', (ctypes.c_ubyte * 32), 960), ('reserved_at_3e0', (ctypes.c_ubyte * 16), 992), ('sandbox_extended_caps_len', (ctypes.c_ubyte * 16), 1008), ('sandbox_extended_caps_addr', (ctypes.c_ubyte * 64), 1024), ('fpga_ddr_start_addr', (ctypes.c_ubyte * 64), 1088), ('fpga_cr_space_start_addr', (ctypes.c_ubyte * 64), 1152), ('fpga_ddr_size', (ctypes.c_ubyte * 32), 1216), ('fpga_cr_space_size', (ctypes.c_ubyte * 32), 1248), ('reserved_at_500', (ctypes.c_ubyte * 768), 1280)])
union_mlx5_ifc_hca_cap_union_bits.register_fields([('cmd_hca_cap', struct_mlx5_ifc_cmd_hca_cap_bits, 0), ('cmd_hca_cap_2', struct_mlx5_ifc_cmd_hca_cap_2_bits, 0), ('odp_cap', struct_mlx5_ifc_odp_cap_bits, 0), ('atomic_caps', struct_mlx5_ifc_atomic_caps_bits, 0), ('roce_cap', struct_mlx5_ifc_roce_cap_bits, 0), ('per_protocol_networking_offload_caps', struct_mlx5_ifc_per_protocol_networking_offload_caps_bits, 0), ('flow_table_nic_cap', struct_mlx5_ifc_flow_table_nic_cap_bits, 0), ('flow_table_eswitch_cap', struct_mlx5_ifc_flow_table_eswitch_cap_bits, 0), ('wqe_based_flow_table_cap', struct_mlx5_ifc_wqe_based_flow_table_cap_bits, 0), ('esw_cap', struct_mlx5_ifc_esw_cap_bits, 0), ('e_switch_cap', struct_mlx5_ifc_e_switch_cap_bits, 0), ('port_selection_cap', struct_mlx5_ifc_port_selection_cap_bits, 0), ('qos_cap', struct_mlx5_ifc_qos_cap_bits, 0), ('debug_cap', struct_mlx5_ifc_debug_cap_bits, 0), ('fpga_cap', struct_mlx5_ifc_fpga_cap_bits, 0), ('tls_cap', struct_mlx5_ifc_tls_cap_bits, 0), ('device_mem_cap', struct_mlx5_ifc_device_mem_cap_bits, 0), ('virtio_emulation_cap', struct_mlx5_ifc_virtio_emulation_cap_bits, 0), ('macsec_cap', struct_mlx5_ifc_macsec_cap_bits, 0), ('crypto_cap', struct_mlx5_ifc_crypto_cap_bits, 0), ('ipsec_cap', struct_mlx5_ifc_ipsec_cap_bits, 0), ('psp_cap', struct_mlx5_ifc_psp_cap_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 32768), 0)])
_anonenum47: dict[int, str] = {(MLX5_FLOW_CONTEXT_ACTION_ALLOW:=1): 'MLX5_FLOW_CONTEXT_ACTION_ALLOW', (MLX5_FLOW_CONTEXT_ACTION_DROP:=2): 'MLX5_FLOW_CONTEXT_ACTION_DROP', (MLX5_FLOW_CONTEXT_ACTION_FWD_DEST:=4): 'MLX5_FLOW_CONTEXT_ACTION_FWD_DEST', (MLX5_FLOW_CONTEXT_ACTION_COUNT:=8): 'MLX5_FLOW_CONTEXT_ACTION_COUNT', (MLX5_FLOW_CONTEXT_ACTION_PACKET_REFORMAT:=16): 'MLX5_FLOW_CONTEXT_ACTION_PACKET_REFORMAT', (MLX5_FLOW_CONTEXT_ACTION_DECAP:=32): 'MLX5_FLOW_CONTEXT_ACTION_DECAP', (MLX5_FLOW_CONTEXT_ACTION_MOD_HDR:=64): 'MLX5_FLOW_CONTEXT_ACTION_MOD_HDR', (MLX5_FLOW_CONTEXT_ACTION_VLAN_POP:=128): 'MLX5_FLOW_CONTEXT_ACTION_VLAN_POP', (MLX5_FLOW_CONTEXT_ACTION_VLAN_PUSH:=256): 'MLX5_FLOW_CONTEXT_ACTION_VLAN_PUSH', (MLX5_FLOW_CONTEXT_ACTION_VLAN_POP_2:=1024): 'MLX5_FLOW_CONTEXT_ACTION_VLAN_POP_2', (MLX5_FLOW_CONTEXT_ACTION_VLAN_PUSH_2:=2048): 'MLX5_FLOW_CONTEXT_ACTION_VLAN_PUSH_2', (MLX5_FLOW_CONTEXT_ACTION_CRYPTO_DECRYPT:=4096): 'MLX5_FLOW_CONTEXT_ACTION_CRYPTO_DECRYPT', (MLX5_FLOW_CONTEXT_ACTION_CRYPTO_ENCRYPT:=8192): 'MLX5_FLOW_CONTEXT_ACTION_CRYPTO_ENCRYPT', (MLX5_FLOW_CONTEXT_ACTION_EXECUTE_ASO:=16384): 'MLX5_FLOW_CONTEXT_ACTION_EXECUTE_ASO'}
_anonenum48: dict[int, str] = {(MLX5_FLOW_CONTEXT_FLOW_SOURCE_ANY_VPORT:=0): 'MLX5_FLOW_CONTEXT_FLOW_SOURCE_ANY_VPORT', (MLX5_FLOW_CONTEXT_FLOW_SOURCE_UPLINK:=1): 'MLX5_FLOW_CONTEXT_FLOW_SOURCE_UPLINK', (MLX5_FLOW_CONTEXT_FLOW_SOURCE_LOCAL_VPORT:=2): 'MLX5_FLOW_CONTEXT_FLOW_SOURCE_LOCAL_VPORT'}
_anonenum49: dict[int, str] = {(MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_IPSEC:=0): 'MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_IPSEC', (MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_MACSEC:=1): 'MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_MACSEC', (MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_PSP:=2): 'MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_PSP'}
@c.record
class struct_mlx5_ifc_vlan_bits(c.Struct):
  SIZE = 32
  ethtype: 'list[int]'
  prio: 'list[int]'
  cfi: 'list[int]'
  vid: 'list[int]'
struct_mlx5_ifc_vlan_bits.register_fields([('ethtype', (ctypes.c_ubyte * 16), 0), ('prio', (ctypes.c_ubyte * 3), 16), ('cfi', (ctypes.c_ubyte * 1), 19), ('vid', (ctypes.c_ubyte * 12), 20)])
_anonenum50: dict[int, str] = {(MLX5_FLOW_METER_COLOR_RED:=0): 'MLX5_FLOW_METER_COLOR_RED', (MLX5_FLOW_METER_COLOR_YELLOW:=1): 'MLX5_FLOW_METER_COLOR_YELLOW', (MLX5_FLOW_METER_COLOR_GREEN:=2): 'MLX5_FLOW_METER_COLOR_GREEN', (MLX5_FLOW_METER_COLOR_UNDEFINED:=3): 'MLX5_FLOW_METER_COLOR_UNDEFINED'}
_anonenum51: dict[int, str] = {(MLX5_EXE_ASO_FLOW_METER:=2): 'MLX5_EXE_ASO_FLOW_METER'}
@c.record
class struct_mlx5_ifc_exe_aso_ctrl_flow_meter_bits(c.Struct):
  SIZE = 32
  return_reg_id: 'list[int]'
  aso_type: 'list[int]'
  reserved_at_8: 'list[int]'
  action: 'list[int]'
  init_color: 'list[int]'
  meter_id: 'list[int]'
struct_mlx5_ifc_exe_aso_ctrl_flow_meter_bits.register_fields([('return_reg_id', (ctypes.c_ubyte * 4), 0), ('aso_type', (ctypes.c_ubyte * 4), 4), ('reserved_at_8', (ctypes.c_ubyte * 20), 8), ('action', (ctypes.c_ubyte * 1), 28), ('init_color', (ctypes.c_ubyte * 2), 29), ('meter_id', (ctypes.c_ubyte * 1), 31)])
@c.record
class union_mlx5_ifc_exe_aso_ctrl(c.Struct):
  SIZE = 32
  exe_aso_ctrl_flow_meter: 'struct_mlx5_ifc_exe_aso_ctrl_flow_meter_bits'
union_mlx5_ifc_exe_aso_ctrl.register_fields([('exe_aso_ctrl_flow_meter', struct_mlx5_ifc_exe_aso_ctrl_flow_meter_bits, 0)])
@c.record
class struct_mlx5_ifc_execute_aso_bits(c.Struct):
  SIZE = 64
  valid: 'list[int]'
  reserved_at_1: 'list[int]'
  aso_object_id: 'list[int]'
  exe_aso_ctrl: 'union_mlx5_ifc_exe_aso_ctrl'
struct_mlx5_ifc_execute_aso_bits.register_fields([('valid', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 7), 1), ('aso_object_id', (ctypes.c_ubyte * 24), 8), ('exe_aso_ctrl', union_mlx5_ifc_exe_aso_ctrl, 32)])
@c.record
class struct_mlx5_ifc_flow_context_bits(c.Struct):
  SIZE = 6144
  push_vlan: 'struct_mlx5_ifc_vlan_bits'
  group_id: 'list[int]'
  reserved_at_40: 'list[int]'
  flow_tag: 'list[int]'
  reserved_at_60: 'list[int]'
  action: 'list[int]'
  extended_destination: 'list[int]'
  uplink_hairpin_en: 'list[int]'
  flow_source: 'list[int]'
  encrypt_decrypt_type: 'list[int]'
  destination_list_size: 'list[int]'
  reserved_at_a0: 'list[int]'
  flow_counter_list_size: 'list[int]'
  packet_reformat_id: 'list[int]'
  modify_header_id: 'list[int]'
  push_vlan_2: 'struct_mlx5_ifc_vlan_bits'
  encrypt_decrypt_obj_id: 'list[int]'
  reserved_at_140: 'list[int]'
  match_value: 'struct_mlx5_ifc_fte_match_param_bits'
  execute_aso: 'list[struct_mlx5_ifc_execute_aso_bits]'
  reserved_at_1300: 'list[int]'
  destination: 'list[union_mlx5_ifc_dest_format_flow_counter_list_auto_bits]'
struct_mlx5_ifc_flow_context_bits.register_fields([('push_vlan', struct_mlx5_ifc_vlan_bits, 0), ('group_id', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('flow_tag', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('action', (ctypes.c_ubyte * 16), 112), ('extended_destination', (ctypes.c_ubyte * 1), 128), ('uplink_hairpin_en', (ctypes.c_ubyte * 1), 129), ('flow_source', (ctypes.c_ubyte * 2), 130), ('encrypt_decrypt_type', (ctypes.c_ubyte * 4), 132), ('destination_list_size', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('flow_counter_list_size', (ctypes.c_ubyte * 24), 168), ('packet_reformat_id', (ctypes.c_ubyte * 32), 192), ('modify_header_id', (ctypes.c_ubyte * 32), 224), ('push_vlan_2', struct_mlx5_ifc_vlan_bits, 256), ('encrypt_decrypt_obj_id', (ctypes.c_ubyte * 32), 288), ('reserved_at_140', (ctypes.c_ubyte * 192), 320), ('match_value', struct_mlx5_ifc_fte_match_param_bits, 512), ('execute_aso', (struct_mlx5_ifc_execute_aso_bits * 4), 4608), ('reserved_at_1300', (ctypes.c_ubyte * 1280), 4864), ('destination', (union_mlx5_ifc_dest_format_flow_counter_list_auto_bits * 0), 6144)])
_anonenum52: dict[int, str] = {(MLX5_XRC_SRQC_STATE_GOOD:=0): 'MLX5_XRC_SRQC_STATE_GOOD', (MLX5_XRC_SRQC_STATE_ERROR:=1): 'MLX5_XRC_SRQC_STATE_ERROR'}
@c.record
class struct_mlx5_ifc_xrc_srqc_bits(c.Struct):
  SIZE = 512
  state: 'list[int]'
  log_xrc_srq_size: 'list[int]'
  reserved_at_8: 'list[int]'
  wq_signature: 'list[int]'
  cont_srq: 'list[int]'
  reserved_at_22: 'list[int]'
  rlky: 'list[int]'
  basic_cyclic_rcv_wqe: 'list[int]'
  log_rq_stride: 'list[int]'
  xrcd: 'list[int]'
  page_offset: 'list[int]'
  reserved_at_46: 'list[int]'
  dbr_umem_valid: 'list[int]'
  cqn: 'list[int]'
  reserved_at_60: 'list[int]'
  user_index_equal_xrc_srqn: 'list[int]'
  reserved_at_81: 'list[int]'
  log_page_size: 'list[int]'
  user_index: 'list[int]'
  reserved_at_a0: 'list[int]'
  reserved_at_c0: 'list[int]'
  pd: 'list[int]'
  lwm: 'list[int]'
  wqe_cnt: 'list[int]'
  reserved_at_100: 'list[int]'
  db_record_addr_h: 'list[int]'
  db_record_addr_l: 'list[int]'
  reserved_at_17e: 'list[int]'
  reserved_at_180: 'list[int]'
struct_mlx5_ifc_xrc_srqc_bits.register_fields([('state', (ctypes.c_ubyte * 4), 0), ('log_xrc_srq_size', (ctypes.c_ubyte * 4), 4), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('wq_signature', (ctypes.c_ubyte * 1), 32), ('cont_srq', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 1), 34), ('rlky', (ctypes.c_ubyte * 1), 35), ('basic_cyclic_rcv_wqe', (ctypes.c_ubyte * 1), 36), ('log_rq_stride', (ctypes.c_ubyte * 3), 37), ('xrcd', (ctypes.c_ubyte * 24), 40), ('page_offset', (ctypes.c_ubyte * 6), 64), ('reserved_at_46', (ctypes.c_ubyte * 1), 70), ('dbr_umem_valid', (ctypes.c_ubyte * 1), 71), ('cqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('user_index_equal_xrc_srqn', (ctypes.c_ubyte * 1), 128), ('reserved_at_81', (ctypes.c_ubyte * 1), 129), ('log_page_size', (ctypes.c_ubyte * 6), 130), ('user_index', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('pd', (ctypes.c_ubyte * 24), 200), ('lwm', (ctypes.c_ubyte * 16), 224), ('wqe_cnt', (ctypes.c_ubyte * 16), 240), ('reserved_at_100', (ctypes.c_ubyte * 64), 256), ('db_record_addr_h', (ctypes.c_ubyte * 32), 320), ('db_record_addr_l', (ctypes.c_ubyte * 30), 352), ('reserved_at_17e', (ctypes.c_ubyte * 2), 382), ('reserved_at_180', (ctypes.c_ubyte * 128), 384)])
@c.record
class struct_mlx5_ifc_vnic_diagnostic_statistics_bits(c.Struct):
  SIZE = 4096
  counter_error_queues: 'list[int]'
  total_error_queues: 'list[int]'
  send_queue_priority_update_flow: 'list[int]'
  reserved_at_60: 'list[int]'
  nic_receive_steering_discard: 'list[int]'
  receive_discard_vport_down: 'list[int]'
  transmit_discard_vport_down: 'list[int]'
  async_eq_overrun: 'list[int]'
  comp_eq_overrun: 'list[int]'
  reserved_at_180: 'list[int]'
  invalid_command: 'list[int]'
  quota_exceeded_command: 'list[int]'
  internal_rq_out_of_buffer: 'list[int]'
  cq_overrun: 'list[int]'
  eth_wqe_too_small: 'list[int]'
  reserved_at_220: 'list[int]'
  generated_pkt_steering_fail: 'list[int]'
  handled_pkt_steering_fail: 'list[int]'
  bar_uar_access: 'list[int]'
  odp_local_triggered_page_fault: 'list[int]'
  odp_remote_triggered_page_fault: 'list[int]'
  reserved_at_3c0: 'list[int]'
struct_mlx5_ifc_vnic_diagnostic_statistics_bits.register_fields([('counter_error_queues', (ctypes.c_ubyte * 32), 0), ('total_error_queues', (ctypes.c_ubyte * 32), 32), ('send_queue_priority_update_flow', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('nic_receive_steering_discard', (ctypes.c_ubyte * 64), 128), ('receive_discard_vport_down', (ctypes.c_ubyte * 64), 192), ('transmit_discard_vport_down', (ctypes.c_ubyte * 64), 256), ('async_eq_overrun', (ctypes.c_ubyte * 32), 320), ('comp_eq_overrun', (ctypes.c_ubyte * 32), 352), ('reserved_at_180', (ctypes.c_ubyte * 32), 384), ('invalid_command', (ctypes.c_ubyte * 32), 416), ('quota_exceeded_command', (ctypes.c_ubyte * 32), 448), ('internal_rq_out_of_buffer', (ctypes.c_ubyte * 32), 480), ('cq_overrun', (ctypes.c_ubyte * 32), 512), ('eth_wqe_too_small', (ctypes.c_ubyte * 32), 544), ('reserved_at_220', (ctypes.c_ubyte * 192), 576), ('generated_pkt_steering_fail', (ctypes.c_ubyte * 64), 768), ('handled_pkt_steering_fail', (ctypes.c_ubyte * 64), 832), ('bar_uar_access', (ctypes.c_ubyte * 32), 896), ('odp_local_triggered_page_fault', (ctypes.c_ubyte * 32), 928), ('odp_remote_triggered_page_fault', (ctypes.c_ubyte * 32), 960), ('reserved_at_3c0', (ctypes.c_ubyte * 3104), 992)])
@c.record
class struct_mlx5_ifc_traffic_counter_bits(c.Struct):
  SIZE = 128
  packets: 'list[int]'
  octets: 'list[int]'
struct_mlx5_ifc_traffic_counter_bits.register_fields([('packets', (ctypes.c_ubyte * 64), 0), ('octets', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_tisc_bits(c.Struct):
  SIZE = 1280
  strict_lag_tx_port_affinity: 'list[int]'
  tls_en: 'list[int]'
  reserved_at_2: 'list[int]'
  lag_tx_port_affinity: 'list[int]'
  reserved_at_8: 'list[int]'
  prio: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_120: 'list[int]'
  transport_domain: 'list[int]'
  reserved_at_140: 'list[int]'
  underlay_qpn: 'list[int]'
  reserved_at_160: 'list[int]'
  pd: 'list[int]'
  reserved_at_180: 'list[int]'
struct_mlx5_ifc_tisc_bits.register_fields([('strict_lag_tx_port_affinity', (ctypes.c_ubyte * 1), 0), ('tls_en', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 2), 2), ('lag_tx_port_affinity', (ctypes.c_ubyte * 4), 4), ('reserved_at_8', (ctypes.c_ubyte * 4), 8), ('prio', (ctypes.c_ubyte * 4), 12), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 256), 32), ('reserved_at_120', (ctypes.c_ubyte * 8), 288), ('transport_domain', (ctypes.c_ubyte * 24), 296), ('reserved_at_140', (ctypes.c_ubyte * 8), 320), ('underlay_qpn', (ctypes.c_ubyte * 24), 328), ('reserved_at_160', (ctypes.c_ubyte * 8), 352), ('pd', (ctypes.c_ubyte * 24), 360), ('reserved_at_180', (ctypes.c_ubyte * 896), 384)])
_anonenum53: dict[int, str] = {(MLX5_TIRC_DISP_TYPE_DIRECT:=0): 'MLX5_TIRC_DISP_TYPE_DIRECT', (MLX5_TIRC_DISP_TYPE_INDIRECT:=1): 'MLX5_TIRC_DISP_TYPE_INDIRECT'}
_anonenum54: dict[int, str] = {(MLX5_TIRC_PACKET_MERGE_MASK_IPV4_LRO:=0): 'MLX5_TIRC_PACKET_MERGE_MASK_IPV4_LRO', (MLX5_TIRC_PACKET_MERGE_MASK_IPV6_LRO:=1): 'MLX5_TIRC_PACKET_MERGE_MASK_IPV6_LRO'}
_anonenum55: dict[int, str] = {(MLX5_RX_HASH_FN_NONE:=0): 'MLX5_RX_HASH_FN_NONE', (MLX5_RX_HASH_FN_INVERTED_XOR8:=1): 'MLX5_RX_HASH_FN_INVERTED_XOR8', (MLX5_RX_HASH_FN_TOEPLITZ:=2): 'MLX5_RX_HASH_FN_TOEPLITZ'}
_anonenum56: dict[int, str] = {(MLX5_TIRC_SELF_LB_BLOCK_BLOCK_UNICAST:=1): 'MLX5_TIRC_SELF_LB_BLOCK_BLOCK_UNICAST', (MLX5_TIRC_SELF_LB_BLOCK_BLOCK_MULTICAST:=2): 'MLX5_TIRC_SELF_LB_BLOCK_BLOCK_MULTICAST'}
@c.record
class struct_mlx5_ifc_tirc_bits(c.Struct):
  SIZE = 1920
  reserved_at_0: 'list[int]'
  disp_type: 'list[int]'
  tls_en: 'list[int]'
  reserved_at_25: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_80: 'list[int]'
  lro_timeout_period_usecs: 'list[int]'
  packet_merge_mask: 'list[int]'
  lro_max_ip_payload_size: 'list[int]'
  reserved_at_a0: 'list[int]'
  reserved_at_e0: 'list[int]'
  inline_rqn: 'list[int]'
  rx_hash_symmetric: 'list[int]'
  reserved_at_101: 'list[int]'
  tunneled_offload_en: 'list[int]'
  reserved_at_103: 'list[int]'
  indirect_table: 'list[int]'
  rx_hash_fn: 'list[int]'
  reserved_at_124: 'list[int]'
  self_lb_block: 'list[int]'
  transport_domain: 'list[int]'
  rx_hash_toeplitz_key: 'list[list[int]]'
  rx_hash_field_selector_outer: 'struct_mlx5_ifc_rx_hash_field_select_bits'
  rx_hash_field_selector_inner: 'struct_mlx5_ifc_rx_hash_field_select_bits'
  reserved_at_2c0: 'list[int]'
struct_mlx5_ifc_tirc_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('disp_type', (ctypes.c_ubyte * 4), 32), ('tls_en', (ctypes.c_ubyte * 1), 36), ('reserved_at_25', (ctypes.c_ubyte * 27), 37), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('reserved_at_80', (ctypes.c_ubyte * 4), 128), ('lro_timeout_period_usecs', (ctypes.c_ubyte * 16), 132), ('packet_merge_mask', (ctypes.c_ubyte * 4), 148), ('lro_max_ip_payload_size', (ctypes.c_ubyte * 8), 152), ('reserved_at_a0', (ctypes.c_ubyte * 64), 160), ('reserved_at_e0', (ctypes.c_ubyte * 8), 224), ('inline_rqn', (ctypes.c_ubyte * 24), 232), ('rx_hash_symmetric', (ctypes.c_ubyte * 1), 256), ('reserved_at_101', (ctypes.c_ubyte * 1), 257), ('tunneled_offload_en', (ctypes.c_ubyte * 1), 258), ('reserved_at_103', (ctypes.c_ubyte * 5), 259), ('indirect_table', (ctypes.c_ubyte * 24), 264), ('rx_hash_fn', (ctypes.c_ubyte * 4), 288), ('reserved_at_124', (ctypes.c_ubyte * 2), 292), ('self_lb_block', (ctypes.c_ubyte * 2), 294), ('transport_domain', (ctypes.c_ubyte * 24), 296), ('rx_hash_toeplitz_key', ((ctypes.c_ubyte * 32) * 10), 320), ('rx_hash_field_selector_outer', struct_mlx5_ifc_rx_hash_field_select_bits, 640), ('rx_hash_field_selector_inner', struct_mlx5_ifc_rx_hash_field_select_bits, 672), ('reserved_at_2c0', (ctypes.c_ubyte * 1216), 704)])
_anonenum57: dict[int, str] = {(MLX5_SRQC_STATE_GOOD:=0): 'MLX5_SRQC_STATE_GOOD', (MLX5_SRQC_STATE_ERROR:=1): 'MLX5_SRQC_STATE_ERROR'}
@c.record
class struct_mlx5_ifc_srqc_bits(c.Struct):
  SIZE = 512
  state: 'list[int]'
  log_srq_size: 'list[int]'
  reserved_at_8: 'list[int]'
  wq_signature: 'list[int]'
  cont_srq: 'list[int]'
  reserved_at_22: 'list[int]'
  rlky: 'list[int]'
  reserved_at_24: 'list[int]'
  log_rq_stride: 'list[int]'
  xrcd: 'list[int]'
  page_offset: 'list[int]'
  reserved_at_46: 'list[int]'
  cqn: 'list[int]'
  reserved_at_60: 'list[int]'
  reserved_at_80: 'list[int]'
  log_page_size: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  reserved_at_c0: 'list[int]'
  pd: 'list[int]'
  lwm: 'list[int]'
  wqe_cnt: 'list[int]'
  reserved_at_100: 'list[int]'
  dbr_addr: 'list[int]'
  reserved_at_180: 'list[int]'
struct_mlx5_ifc_srqc_bits.register_fields([('state', (ctypes.c_ubyte * 4), 0), ('log_srq_size', (ctypes.c_ubyte * 4), 4), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('wq_signature', (ctypes.c_ubyte * 1), 32), ('cont_srq', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 1), 34), ('rlky', (ctypes.c_ubyte * 1), 35), ('reserved_at_24', (ctypes.c_ubyte * 1), 36), ('log_rq_stride', (ctypes.c_ubyte * 3), 37), ('xrcd', (ctypes.c_ubyte * 24), 40), ('page_offset', (ctypes.c_ubyte * 6), 64), ('reserved_at_46', (ctypes.c_ubyte * 2), 70), ('cqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 2), 128), ('log_page_size', (ctypes.c_ubyte * 6), 130), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('pd', (ctypes.c_ubyte * 24), 200), ('lwm', (ctypes.c_ubyte * 16), 224), ('wqe_cnt', (ctypes.c_ubyte * 16), 240), ('reserved_at_100', (ctypes.c_ubyte * 64), 256), ('dbr_addr', (ctypes.c_ubyte * 64), 320), ('reserved_at_180', (ctypes.c_ubyte * 128), 384)])
_anonenum58: dict[int, str] = {(MLX5_SQC_STATE_RST:=0): 'MLX5_SQC_STATE_RST', (MLX5_SQC_STATE_RDY:=1): 'MLX5_SQC_STATE_RDY', (MLX5_SQC_STATE_ERR:=3): 'MLX5_SQC_STATE_ERR'}
@c.record
class struct_mlx5_ifc_sqc_bits(c.Struct):
  SIZE = 1920
  rlky: 'list[int]'
  cd_master: 'list[int]'
  fre: 'list[int]'
  flush_in_error_en: 'list[int]'
  allow_multi_pkt_send_wqe: 'list[int]'
  min_wqe_inline_mode: 'list[int]'
  state: 'list[int]'
  reg_umr: 'list[int]'
  allow_swp: 'list[int]'
  hairpin: 'list[int]'
  non_wire: 'list[int]'
  reserved_at_10: 'list[int]'
  ts_format: 'list[int]'
  reserved_at_1c: 'list[int]'
  reserved_at_20: 'list[int]'
  user_index: 'list[int]'
  reserved_at_40: 'list[int]'
  cqn: 'list[int]'
  reserved_at_60: 'list[int]'
  hairpin_peer_rq: 'list[int]'
  reserved_at_80: 'list[int]'
  hairpin_peer_vhca: 'list[int]'
  reserved_at_a0: 'list[int]'
  reserved_at_c0: 'list[int]'
  ts_cqe_to_dest_cqn: 'list[int]'
  reserved_at_e0: 'list[int]'
  packet_pacing_rate_limit_index: 'list[int]'
  tis_lst_sz: 'list[int]'
  qos_queue_group_id: 'list[int]'
  reserved_at_120: 'list[int]'
  reserved_at_160: 'list[int]'
  tis_num_0: 'list[int]'
  wq: 'struct_mlx5_ifc_wq_bits'
struct_mlx5_ifc_sqc_bits.register_fields([('rlky', (ctypes.c_ubyte * 1), 0), ('cd_master', (ctypes.c_ubyte * 1), 1), ('fre', (ctypes.c_ubyte * 1), 2), ('flush_in_error_en', (ctypes.c_ubyte * 1), 3), ('allow_multi_pkt_send_wqe', (ctypes.c_ubyte * 1), 4), ('min_wqe_inline_mode', (ctypes.c_ubyte * 3), 5), ('state', (ctypes.c_ubyte * 4), 8), ('reg_umr', (ctypes.c_ubyte * 1), 12), ('allow_swp', (ctypes.c_ubyte * 1), 13), ('hairpin', (ctypes.c_ubyte * 1), 14), ('non_wire', (ctypes.c_ubyte * 1), 15), ('reserved_at_10', (ctypes.c_ubyte * 10), 16), ('ts_format', (ctypes.c_ubyte * 2), 26), ('reserved_at_1c', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('user_index', (ctypes.c_ubyte * 24), 40), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('cqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 8), 96), ('hairpin_peer_rq', (ctypes.c_ubyte * 24), 104), ('reserved_at_80', (ctypes.c_ubyte * 16), 128), ('hairpin_peer_vhca', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('ts_cqe_to_dest_cqn', (ctypes.c_ubyte * 24), 200), ('reserved_at_e0', (ctypes.c_ubyte * 16), 224), ('packet_pacing_rate_limit_index', (ctypes.c_ubyte * 16), 240), ('tis_lst_sz', (ctypes.c_ubyte * 16), 256), ('qos_queue_group_id', (ctypes.c_ubyte * 16), 272), ('reserved_at_120', (ctypes.c_ubyte * 64), 288), ('reserved_at_160', (ctypes.c_ubyte * 8), 352), ('tis_num_0', (ctypes.c_ubyte * 24), 360), ('wq', struct_mlx5_ifc_wq_bits, 384)])
_anonenum59: dict[int, str] = {(SCHEDULING_CONTEXT_ELEMENT_TYPE_TSAR:=0): 'SCHEDULING_CONTEXT_ELEMENT_TYPE_TSAR', (SCHEDULING_CONTEXT_ELEMENT_TYPE_VPORT:=1): 'SCHEDULING_CONTEXT_ELEMENT_TYPE_VPORT', (SCHEDULING_CONTEXT_ELEMENT_TYPE_VPORT_TC:=2): 'SCHEDULING_CONTEXT_ELEMENT_TYPE_VPORT_TC', (SCHEDULING_CONTEXT_ELEMENT_TYPE_PARA_VPORT_TC:=3): 'SCHEDULING_CONTEXT_ELEMENT_TYPE_PARA_VPORT_TC', (SCHEDULING_CONTEXT_ELEMENT_TYPE_QUEUE_GROUP:=4): 'SCHEDULING_CONTEXT_ELEMENT_TYPE_QUEUE_GROUP', (SCHEDULING_CONTEXT_ELEMENT_TYPE_RATE_LIMIT:=5): 'SCHEDULING_CONTEXT_ELEMENT_TYPE_RATE_LIMIT'}
_anonenum60: dict[int, str] = {(ELEMENT_TYPE_CAP_MASK_TSAR:=1): 'ELEMENT_TYPE_CAP_MASK_TSAR', (ELEMENT_TYPE_CAP_MASK_VPORT:=2): 'ELEMENT_TYPE_CAP_MASK_VPORT', (ELEMENT_TYPE_CAP_MASK_VPORT_TC:=4): 'ELEMENT_TYPE_CAP_MASK_VPORT_TC', (ELEMENT_TYPE_CAP_MASK_PARA_VPORT_TC:=8): 'ELEMENT_TYPE_CAP_MASK_PARA_VPORT_TC', (ELEMENT_TYPE_CAP_MASK_QUEUE_GROUP:=16): 'ELEMENT_TYPE_CAP_MASK_QUEUE_GROUP', (ELEMENT_TYPE_CAP_MASK_RATE_LIMIT:=32): 'ELEMENT_TYPE_CAP_MASK_RATE_LIMIT'}
_anonenum61: dict[int, str] = {(TSAR_ELEMENT_TSAR_TYPE_DWRR:=0): 'TSAR_ELEMENT_TSAR_TYPE_DWRR', (TSAR_ELEMENT_TSAR_TYPE_ROUND_ROBIN:=1): 'TSAR_ELEMENT_TSAR_TYPE_ROUND_ROBIN', (TSAR_ELEMENT_TSAR_TYPE_ETS:=2): 'TSAR_ELEMENT_TSAR_TYPE_ETS', (TSAR_ELEMENT_TSAR_TYPE_TC_ARB:=3): 'TSAR_ELEMENT_TSAR_TYPE_TC_ARB'}
_anonenum62: dict[int, str] = {(TSAR_TYPE_CAP_MASK_DWRR:=1): 'TSAR_TYPE_CAP_MASK_DWRR', (TSAR_TYPE_CAP_MASK_ROUND_ROBIN:=2): 'TSAR_TYPE_CAP_MASK_ROUND_ROBIN', (TSAR_TYPE_CAP_MASK_ETS:=4): 'TSAR_TYPE_CAP_MASK_ETS', (TSAR_TYPE_CAP_MASK_TC_ARB:=8): 'TSAR_TYPE_CAP_MASK_TC_ARB'}
@c.record
class struct_mlx5_ifc_tsar_element_bits(c.Struct):
  SIZE = 32
  traffic_class: 'list[int]'
  reserved_at_4: 'list[int]'
  tsar_type: 'list[int]'
  reserved_at_10: 'list[int]'
struct_mlx5_ifc_tsar_element_bits.register_fields([('traffic_class', (ctypes.c_ubyte * 4), 0), ('reserved_at_4', (ctypes.c_ubyte * 4), 4), ('tsar_type', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16)])
@c.record
class struct_mlx5_ifc_vport_element_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  eswitch_owner_vhca_id_valid: 'list[int]'
  eswitch_owner_vhca_id: 'list[int]'
  vport_number: 'list[int]'
struct_mlx5_ifc_vport_element_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('eswitch_owner_vhca_id_valid', (ctypes.c_ubyte * 1), 4), ('eswitch_owner_vhca_id', (ctypes.c_ubyte * 11), 5), ('vport_number', (ctypes.c_ubyte * 16), 16)])
@c.record
class struct_mlx5_ifc_vport_tc_element_bits(c.Struct):
  SIZE = 32
  traffic_class: 'list[int]'
  eswitch_owner_vhca_id_valid: 'list[int]'
  eswitch_owner_vhca_id: 'list[int]'
  vport_number: 'list[int]'
struct_mlx5_ifc_vport_tc_element_bits.register_fields([('traffic_class', (ctypes.c_ubyte * 4), 0), ('eswitch_owner_vhca_id_valid', (ctypes.c_ubyte * 1), 4), ('eswitch_owner_vhca_id', (ctypes.c_ubyte * 11), 5), ('vport_number', (ctypes.c_ubyte * 16), 16)])
@c.record
class union_mlx5_ifc_element_attributes_bits(c.Struct):
  SIZE = 32
  tsar: 'struct_mlx5_ifc_tsar_element_bits'
  vport: 'struct_mlx5_ifc_vport_element_bits'
  vport_tc: 'struct_mlx5_ifc_vport_tc_element_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_element_attributes_bits.register_fields([('tsar', struct_mlx5_ifc_tsar_element_bits, 0), ('vport', struct_mlx5_ifc_vport_element_bits, 0), ('vport_tc', struct_mlx5_ifc_vport_tc_element_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 32), 0)])
@c.record
class struct_mlx5_ifc_scheduling_context_bits(c.Struct):
  SIZE = 512
  element_type: 'list[int]'
  reserved_at_8: 'list[int]'
  element_attributes: 'union_mlx5_ifc_element_attributes_bits'
  parent_element_id: 'list[int]'
  reserved_at_60: 'list[int]'
  bw_share: 'list[int]'
  max_average_bw: 'list[int]'
  max_bw_obj_id: 'list[int]'
  reserved_at_100: 'list[int]'
struct_mlx5_ifc_scheduling_context_bits.register_fields([('element_type', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('element_attributes', union_mlx5_ifc_element_attributes_bits, 32), ('parent_element_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 64), 96), ('bw_share', (ctypes.c_ubyte * 32), 160), ('max_average_bw', (ctypes.c_ubyte * 32), 192), ('max_bw_obj_id', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 256), 256)])
@c.record
class struct_mlx5_ifc_rqtc_bits(c.Struct):
  SIZE = 1920
  reserved_at_0: 'list[int]'
  reserved_at_a0: 'list[int]'
  list_q_type: 'list[int]'
  reserved_at_a8: 'list[int]'
  rqt_max_size: 'list[int]'
  rq_vhca_id_format: 'list[int]'
  reserved_at_c1: 'list[int]'
  rqt_actual_size: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_rqtc_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 160), 0), ('reserved_at_a0', (ctypes.c_ubyte * 5), 160), ('list_q_type', (ctypes.c_ubyte * 3), 165), ('reserved_at_a8', (ctypes.c_ubyte * 8), 168), ('rqt_max_size', (ctypes.c_ubyte * 16), 176), ('rq_vhca_id_format', (ctypes.c_ubyte * 1), 192), ('reserved_at_c1', (ctypes.c_ubyte * 15), 193), ('rqt_actual_size', (ctypes.c_ubyte * 16), 208), ('reserved_at_e0', (ctypes.c_ubyte * 1696), 224)])
_anonenum63: dict[int, str] = {(MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_INLINE:=0): 'MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_INLINE', (MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_RMP:=1): 'MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_RMP'}
_anonenum64: dict[int, str] = {(MLX5_RQC_STATE_RST:=0): 'MLX5_RQC_STATE_RST', (MLX5_RQC_STATE_RDY:=1): 'MLX5_RQC_STATE_RDY', (MLX5_RQC_STATE_ERR:=3): 'MLX5_RQC_STATE_ERR'}
_anonenum65: dict[int, str] = {(MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_BYTE:=0): 'MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_BYTE', (MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_STRIDE:=1): 'MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_STRIDE', (MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_PAGE:=2): 'MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_PAGE'}
_anonenum66: dict[int, str] = {(MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_NO_MATCH:=0): 'MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_NO_MATCH', (MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_EXTENDED:=1): 'MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_EXTENDED', (MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_FIVE_TUPLE:=2): 'MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_FIVE_TUPLE'}
@c.record
class struct_mlx5_ifc_rqc_bits(c.Struct):
  SIZE = 1920
  rlky: 'list[int]'
  delay_drop_en: 'list[int]'
  scatter_fcs: 'list[int]'
  vsd: 'list[int]'
  mem_rq_type: 'list[int]'
  state: 'list[int]'
  reserved_at_c: 'list[int]'
  flush_in_error_en: 'list[int]'
  hairpin: 'list[int]'
  reserved_at_f: 'list[int]'
  ts_format: 'list[int]'
  reserved_at_1c: 'list[int]'
  reserved_at_20: 'list[int]'
  user_index: 'list[int]'
  reserved_at_40: 'list[int]'
  cqn: 'list[int]'
  counter_set_id: 'list[int]'
  reserved_at_68: 'list[int]'
  reserved_at_80: 'list[int]'
  rmpn: 'list[int]'
  reserved_at_a0: 'list[int]'
  hairpin_peer_sq: 'list[int]'
  reserved_at_c0: 'list[int]'
  hairpin_peer_vhca: 'list[int]'
  reserved_at_e0: 'list[int]'
  shampo_no_match_alignment_granularity: 'list[int]'
  reserved_at_128: 'list[int]'
  shampo_match_criteria_type: 'list[int]'
  reservation_timeout: 'list[int]'
  reserved_at_140: 'list[int]'
  wq: 'struct_mlx5_ifc_wq_bits'
struct_mlx5_ifc_rqc_bits.register_fields([('rlky', (ctypes.c_ubyte * 1), 0), ('delay_drop_en', (ctypes.c_ubyte * 1), 1), ('scatter_fcs', (ctypes.c_ubyte * 1), 2), ('vsd', (ctypes.c_ubyte * 1), 3), ('mem_rq_type', (ctypes.c_ubyte * 4), 4), ('state', (ctypes.c_ubyte * 4), 8), ('reserved_at_c', (ctypes.c_ubyte * 1), 12), ('flush_in_error_en', (ctypes.c_ubyte * 1), 13), ('hairpin', (ctypes.c_ubyte * 1), 14), ('reserved_at_f', (ctypes.c_ubyte * 11), 15), ('ts_format', (ctypes.c_ubyte * 2), 26), ('reserved_at_1c', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('user_index', (ctypes.c_ubyte * 24), 40), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('cqn', (ctypes.c_ubyte * 24), 72), ('counter_set_id', (ctypes.c_ubyte * 8), 96), ('reserved_at_68', (ctypes.c_ubyte * 24), 104), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('rmpn', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('hairpin_peer_sq', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 16), 192), ('hairpin_peer_vhca', (ctypes.c_ubyte * 16), 208), ('reserved_at_e0', (ctypes.c_ubyte * 70), 224), ('shampo_no_match_alignment_granularity', (ctypes.c_ubyte * 2), 294), ('reserved_at_128', (ctypes.c_ubyte * 6), 296), ('shampo_match_criteria_type', (ctypes.c_ubyte * 2), 302), ('reservation_timeout', (ctypes.c_ubyte * 16), 304), ('reserved_at_140', (ctypes.c_ubyte * 64), 320), ('wq', struct_mlx5_ifc_wq_bits, 384)])
_anonenum67: dict[int, str] = {(MLX5_RMPC_STATE_RDY:=1): 'MLX5_RMPC_STATE_RDY', (MLX5_RMPC_STATE_ERR:=3): 'MLX5_RMPC_STATE_ERR'}
@c.record
class struct_mlx5_ifc_rmpc_bits(c.Struct):
  SIZE = 1920
  reserved_at_0: 'list[int]'
  state: 'list[int]'
  reserved_at_c: 'list[int]'
  basic_cyclic_rcv_wqe: 'list[int]'
  reserved_at_21: 'list[int]'
  reserved_at_40: 'list[int]'
  wq: 'struct_mlx5_ifc_wq_bits'
struct_mlx5_ifc_rmpc_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('state', (ctypes.c_ubyte * 4), 8), ('reserved_at_c', (ctypes.c_ubyte * 20), 12), ('basic_cyclic_rcv_wqe', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 31), 33), ('reserved_at_40', (ctypes.c_ubyte * 320), 64), ('wq', struct_mlx5_ifc_wq_bits, 384)])
_anonenum68: dict[int, str] = {(VHCA_ID_TYPE_HW:=0): 'VHCA_ID_TYPE_HW', (VHCA_ID_TYPE_SW:=1): 'VHCA_ID_TYPE_SW'}
@c.record
class struct_mlx5_ifc_nic_vport_context_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  min_wqe_inline_mode: 'list[int]'
  reserved_at_8: 'list[int]'
  disable_mc_local_lb: 'list[int]'
  disable_uc_local_lb: 'list[int]'
  roce_en: 'list[int]'
  arm_change_event: 'list[int]'
  reserved_at_21: 'list[int]'
  event_on_mtu: 'list[int]'
  event_on_promisc_change: 'list[int]'
  event_on_vlan_change: 'list[int]'
  event_on_mc_address_change: 'list[int]'
  event_on_uc_address_change: 'list[int]'
  vhca_id_type: 'list[int]'
  reserved_at_41: 'list[int]'
  affiliation_criteria: 'list[int]'
  affiliated_vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
  reserved_at_100: 'list[int]'
  sd_group: 'list[int]'
  reserved_at_104: 'list[int]'
  reserved_at_120: 'list[int]'
  mtu: 'list[int]'
  system_image_guid: 'list[int]'
  port_guid: 'list[int]'
  node_guid: 'list[int]'
  reserved_at_200: 'list[int]'
  qkey_violation_counter: 'list[int]'
  reserved_at_350: 'list[int]'
  promisc_uc: 'list[int]'
  promisc_mc: 'list[int]'
  promisc_all: 'list[int]'
  reserved_at_783: 'list[int]'
  allowed_list_type: 'list[int]'
  reserved_at_788: 'list[int]'
  allowed_list_size: 'list[int]'
  permanent_address: 'struct_mlx5_ifc_mac_address_layout_bits'
  reserved_at_7e0: 'list[int]'
  current_uc_mac_address: 'list[list[int]]'
struct_mlx5_ifc_nic_vport_context_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 5), 0), ('min_wqe_inline_mode', (ctypes.c_ubyte * 3), 5), ('reserved_at_8', (ctypes.c_ubyte * 21), 8), ('disable_mc_local_lb', (ctypes.c_ubyte * 1), 29), ('disable_uc_local_lb', (ctypes.c_ubyte * 1), 30), ('roce_en', (ctypes.c_ubyte * 1), 31), ('arm_change_event', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 26), 33), ('event_on_mtu', (ctypes.c_ubyte * 1), 59), ('event_on_promisc_change', (ctypes.c_ubyte * 1), 60), ('event_on_vlan_change', (ctypes.c_ubyte * 1), 61), ('event_on_mc_address_change', (ctypes.c_ubyte * 1), 62), ('event_on_uc_address_change', (ctypes.c_ubyte * 1), 63), ('vhca_id_type', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 11), 65), ('affiliation_criteria', (ctypes.c_ubyte * 4), 76), ('affiliated_vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 160), 96), ('reserved_at_100', (ctypes.c_ubyte * 1), 256), ('sd_group', (ctypes.c_ubyte * 3), 257), ('reserved_at_104', (ctypes.c_ubyte * 28), 260), ('reserved_at_120', (ctypes.c_ubyte * 16), 288), ('mtu', (ctypes.c_ubyte * 16), 304), ('system_image_guid', (ctypes.c_ubyte * 64), 320), ('port_guid', (ctypes.c_ubyte * 64), 384), ('node_guid', (ctypes.c_ubyte * 64), 448), ('reserved_at_200', (ctypes.c_ubyte * 320), 512), ('qkey_violation_counter', (ctypes.c_ubyte * 16), 832), ('reserved_at_350', (ctypes.c_ubyte * 1072), 848), ('promisc_uc', (ctypes.c_ubyte * 1), 1920), ('promisc_mc', (ctypes.c_ubyte * 1), 1921), ('promisc_all', (ctypes.c_ubyte * 1), 1922), ('reserved_at_783', (ctypes.c_ubyte * 2), 1923), ('allowed_list_type', (ctypes.c_ubyte * 3), 1925), ('reserved_at_788', (ctypes.c_ubyte * 12), 1928), ('allowed_list_size', (ctypes.c_ubyte * 12), 1940), ('permanent_address', struct_mlx5_ifc_mac_address_layout_bits, 1952), ('reserved_at_7e0', (ctypes.c_ubyte * 32), 2016), ('current_uc_mac_address', ((ctypes.c_ubyte * 64) * 0), 2048)])
_anonenum69: dict[int, str] = {(MLX5_MKC_ACCESS_MODE_PA:=0): 'MLX5_MKC_ACCESS_MODE_PA', (MLX5_MKC_ACCESS_MODE_MTT:=1): 'MLX5_MKC_ACCESS_MODE_MTT', (MLX5_MKC_ACCESS_MODE_KLMS:=2): 'MLX5_MKC_ACCESS_MODE_KLMS', (MLX5_MKC_ACCESS_MODE_KSM:=3): 'MLX5_MKC_ACCESS_MODE_KSM', (MLX5_MKC_ACCESS_MODE_SW_ICM:=4): 'MLX5_MKC_ACCESS_MODE_SW_ICM', (MLX5_MKC_ACCESS_MODE_MEMIC:=5): 'MLX5_MKC_ACCESS_MODE_MEMIC', (MLX5_MKC_ACCESS_MODE_CROSSING:=6): 'MLX5_MKC_ACCESS_MODE_CROSSING'}
_anonenum70: dict[int, str] = {(MLX5_MKC_PCIE_TPH_NO_STEERING_TAG_INDEX:=0): 'MLX5_MKC_PCIE_TPH_NO_STEERING_TAG_INDEX'}
@c.record
class struct_mlx5_ifc_mkc_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  free: 'list[int]'
  reserved_at_2: 'list[int]'
  access_mode_4_2: 'list[int]'
  reserved_at_6: 'list[int]'
  relaxed_ordering_write: 'list[int]'
  reserved_at_e: 'list[int]'
  small_fence_on_rdma_read_response: 'list[int]'
  umr_en: 'list[int]'
  a: 'list[int]'
  rw: 'list[int]'
  rr: 'list[int]'
  lw: 'list[int]'
  lr: 'list[int]'
  access_mode_1_0: 'list[int]'
  reserved_at_18: 'list[int]'
  ma_translation_mode: 'list[int]'
  reserved_at_1c: 'list[int]'
  qpn: 'list[int]'
  mkey_7_0: 'list[int]'
  reserved_at_40: 'list[int]'
  length64: 'list[int]'
  bsf_en: 'list[int]'
  sync_umr: 'list[int]'
  reserved_at_63: 'list[int]'
  expected_sigerr_count: 'list[int]'
  reserved_at_66: 'list[int]'
  en_rinval: 'list[int]'
  pd: 'list[int]'
  start_addr: 'list[int]'
  len: 'list[int]'
  bsf_octword_size: 'list[int]'
  reserved_at_120: 'list[int]'
  crossing_target_vhca_id: 'list[int]'
  reserved_at_190: 'list[int]'
  translations_octword_size: 'list[int]'
  reserved_at_1c0: 'list[int]'
  relaxed_ordering_read: 'list[int]'
  log_page_size: 'list[int]'
  reserved_at_1e0: 'list[int]'
  pcie_tph_en: 'list[int]'
  pcie_tph_ph: 'list[int]'
  pcie_tph_steering_tag_index: 'list[int]'
  reserved_at_1f0: 'list[int]'
struct_mlx5_ifc_mkc_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 1), 0), ('free', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 1), 2), ('access_mode_4_2', (ctypes.c_ubyte * 3), 3), ('reserved_at_6', (ctypes.c_ubyte * 7), 6), ('relaxed_ordering_write', (ctypes.c_ubyte * 1), 13), ('reserved_at_e', (ctypes.c_ubyte * 1), 14), ('small_fence_on_rdma_read_response', (ctypes.c_ubyte * 1), 15), ('umr_en', (ctypes.c_ubyte * 1), 16), ('a', (ctypes.c_ubyte * 1), 17), ('rw', (ctypes.c_ubyte * 1), 18), ('rr', (ctypes.c_ubyte * 1), 19), ('lw', (ctypes.c_ubyte * 1), 20), ('lr', (ctypes.c_ubyte * 1), 21), ('access_mode_1_0', (ctypes.c_ubyte * 2), 22), ('reserved_at_18', (ctypes.c_ubyte * 2), 24), ('ma_translation_mode', (ctypes.c_ubyte * 2), 26), ('reserved_at_1c', (ctypes.c_ubyte * 4), 28), ('qpn', (ctypes.c_ubyte * 24), 32), ('mkey_7_0', (ctypes.c_ubyte * 8), 56), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('length64', (ctypes.c_ubyte * 1), 96), ('bsf_en', (ctypes.c_ubyte * 1), 97), ('sync_umr', (ctypes.c_ubyte * 1), 98), ('reserved_at_63', (ctypes.c_ubyte * 2), 99), ('expected_sigerr_count', (ctypes.c_ubyte * 1), 101), ('reserved_at_66', (ctypes.c_ubyte * 1), 102), ('en_rinval', (ctypes.c_ubyte * 1), 103), ('pd', (ctypes.c_ubyte * 24), 104), ('start_addr', (ctypes.c_ubyte * 64), 128), ('len', (ctypes.c_ubyte * 64), 192), ('bsf_octword_size', (ctypes.c_ubyte * 32), 256), ('reserved_at_120', (ctypes.c_ubyte * 96), 288), ('crossing_target_vhca_id', (ctypes.c_ubyte * 16), 384), ('reserved_at_190', (ctypes.c_ubyte * 16), 400), ('translations_octword_size', (ctypes.c_ubyte * 32), 416), ('reserved_at_1c0', (ctypes.c_ubyte * 25), 448), ('relaxed_ordering_read', (ctypes.c_ubyte * 1), 473), ('log_page_size', (ctypes.c_ubyte * 6), 474), ('reserved_at_1e0', (ctypes.c_ubyte * 5), 480), ('pcie_tph_en', (ctypes.c_ubyte * 1), 485), ('pcie_tph_ph', (ctypes.c_ubyte * 2), 486), ('pcie_tph_steering_tag_index', (ctypes.c_ubyte * 8), 488), ('reserved_at_1f0', (ctypes.c_ubyte * 16), 496)])
@c.record
class struct_mlx5_ifc_pkey_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  pkey: 'list[int]'
struct_mlx5_ifc_pkey_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('pkey', (ctypes.c_ubyte * 16), 16)])
@c.record
class struct_mlx5_ifc_array128_auto_bits(c.Struct):
  SIZE = 128
  array128_auto: 'list[list[int]]'
struct_mlx5_ifc_array128_auto_bits.register_fields([('array128_auto', ((ctypes.c_ubyte * 8) * 16), 0)])
@c.record
class struct_mlx5_ifc_hca_vport_context_bits(c.Struct):
  SIZE = 4096
  field_select: 'list[int]'
  reserved_at_20: 'list[int]'
  sm_virt_aware: 'list[int]'
  has_smi: 'list[int]'
  has_raw: 'list[int]'
  grh_required: 'list[int]'
  reserved_at_104: 'list[int]'
  num_port_plane: 'list[int]'
  port_physical_state: 'list[int]'
  vport_state_policy: 'list[int]'
  port_state: 'list[int]'
  vport_state: 'list[int]'
  reserved_at_120: 'list[int]'
  system_image_guid: 'list[int]'
  port_guid: 'list[int]'
  node_guid: 'list[int]'
  cap_mask1: 'list[int]'
  cap_mask1_field_select: 'list[int]'
  cap_mask2: 'list[int]'
  cap_mask2_field_select: 'list[int]'
  reserved_at_280: 'list[int]'
  lid: 'list[int]'
  reserved_at_310: 'list[int]'
  init_type_reply: 'list[int]'
  lmc: 'list[int]'
  subnet_timeout: 'list[int]'
  sm_lid: 'list[int]'
  sm_sl: 'list[int]'
  reserved_at_334: 'list[int]'
  qkey_violation_counter: 'list[int]'
  pkey_violation_counter: 'list[int]'
  reserved_at_360: 'list[int]'
struct_mlx5_ifc_hca_vport_context_bits.register_fields([('field_select', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 224), 32), ('sm_virt_aware', (ctypes.c_ubyte * 1), 256), ('has_smi', (ctypes.c_ubyte * 1), 257), ('has_raw', (ctypes.c_ubyte * 1), 258), ('grh_required', (ctypes.c_ubyte * 1), 259), ('reserved_at_104', (ctypes.c_ubyte * 4), 260), ('num_port_plane', (ctypes.c_ubyte * 8), 264), ('port_physical_state', (ctypes.c_ubyte * 4), 272), ('vport_state_policy', (ctypes.c_ubyte * 4), 276), ('port_state', (ctypes.c_ubyte * 4), 280), ('vport_state', (ctypes.c_ubyte * 4), 284), ('reserved_at_120', (ctypes.c_ubyte * 32), 288), ('system_image_guid', (ctypes.c_ubyte * 64), 320), ('port_guid', (ctypes.c_ubyte * 64), 384), ('node_guid', (ctypes.c_ubyte * 64), 448), ('cap_mask1', (ctypes.c_ubyte * 32), 512), ('cap_mask1_field_select', (ctypes.c_ubyte * 32), 544), ('cap_mask2', (ctypes.c_ubyte * 32), 576), ('cap_mask2_field_select', (ctypes.c_ubyte * 32), 608), ('reserved_at_280', (ctypes.c_ubyte * 128), 640), ('lid', (ctypes.c_ubyte * 16), 768), ('reserved_at_310', (ctypes.c_ubyte * 4), 784), ('init_type_reply', (ctypes.c_ubyte * 4), 788), ('lmc', (ctypes.c_ubyte * 3), 792), ('subnet_timeout', (ctypes.c_ubyte * 5), 795), ('sm_lid', (ctypes.c_ubyte * 16), 800), ('sm_sl', (ctypes.c_ubyte * 4), 816), ('reserved_at_334', (ctypes.c_ubyte * 12), 820), ('qkey_violation_counter', (ctypes.c_ubyte * 16), 832), ('pkey_violation_counter', (ctypes.c_ubyte * 16), 848), ('reserved_at_360', (ctypes.c_ubyte * 3232), 864)])
@c.record
class struct_mlx5_ifc_esw_vport_context_bits(c.Struct):
  SIZE = 2048
  fdb_to_vport_reg_c: 'list[int]'
  reserved_at_1: 'list[int]'
  vport_svlan_strip: 'list[int]'
  vport_cvlan_strip: 'list[int]'
  vport_svlan_insert: 'list[int]'
  vport_cvlan_insert: 'list[int]'
  fdb_to_vport_reg_c_id: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  svlan_cfi: 'list[int]'
  svlan_pcp: 'list[int]'
  svlan_id: 'list[int]'
  cvlan_cfi: 'list[int]'
  cvlan_pcp: 'list[int]'
  cvlan_id: 'list[int]'
  reserved_at_60: 'list[int]'
  sw_steering_vport_icm_address_rx: 'list[int]'
  sw_steering_vport_icm_address_tx: 'list[int]'
struct_mlx5_ifc_esw_vport_context_bits.register_fields([('fdb_to_vport_reg_c', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 2), 1), ('vport_svlan_strip', (ctypes.c_ubyte * 1), 3), ('vport_cvlan_strip', (ctypes.c_ubyte * 1), 4), ('vport_svlan_insert', (ctypes.c_ubyte * 1), 5), ('vport_cvlan_insert', (ctypes.c_ubyte * 2), 6), ('fdb_to_vport_reg_c_id', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('svlan_cfi', (ctypes.c_ubyte * 1), 64), ('svlan_pcp', (ctypes.c_ubyte * 3), 65), ('svlan_id', (ctypes.c_ubyte * 12), 68), ('cvlan_cfi', (ctypes.c_ubyte * 1), 80), ('cvlan_pcp', (ctypes.c_ubyte * 3), 81), ('cvlan_id', (ctypes.c_ubyte * 12), 84), ('reserved_at_60', (ctypes.c_ubyte * 1824), 96), ('sw_steering_vport_icm_address_rx', (ctypes.c_ubyte * 64), 1920), ('sw_steering_vport_icm_address_tx', (ctypes.c_ubyte * 64), 1984)])
_anonenum71: dict[int, str] = {(MLX5_EQC_STATUS_OK:=0): 'MLX5_EQC_STATUS_OK', (MLX5_EQC_STATUS_EQ_WRITE_FAILURE:=10): 'MLX5_EQC_STATUS_EQ_WRITE_FAILURE'}
_anonenum72: dict[int, str] = {(MLX5_EQC_ST_ARMED:=9): 'MLX5_EQC_ST_ARMED', (MLX5_EQC_ST_FIRED:=10): 'MLX5_EQC_ST_FIRED'}
@c.record
class struct_mlx5_ifc_eqc_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_4: 'list[int]'
  ec: 'list[int]'
  oi: 'list[int]'
  reserved_at_f: 'list[int]'
  st: 'list[int]'
  reserved_at_18: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_40: 'list[int]'
  page_offset: 'list[int]'
  reserved_at_5a: 'list[int]'
  reserved_at_60: 'list[int]'
  log_eq_size: 'list[int]'
  uar_page: 'list[int]'
  reserved_at_80: 'list[int]'
  reserved_at_a0: 'list[int]'
  intr: 'list[int]'
  reserved_at_c0: 'list[int]'
  log_page_size: 'list[int]'
  reserved_at_c8: 'list[int]'
  reserved_at_e0: 'list[int]'
  reserved_at_140: 'list[int]'
  consumer_counter: 'list[int]'
  reserved_at_160: 'list[int]'
  producer_counter: 'list[int]'
  reserved_at_180: 'list[int]'
struct_mlx5_ifc_eqc_bits.register_fields([('status', (ctypes.c_ubyte * 4), 0), ('reserved_at_4', (ctypes.c_ubyte * 9), 4), ('ec', (ctypes.c_ubyte * 1), 13), ('oi', (ctypes.c_ubyte * 1), 14), ('reserved_at_f', (ctypes.c_ubyte * 5), 15), ('st', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 20), 64), ('page_offset', (ctypes.c_ubyte * 6), 84), ('reserved_at_5a', (ctypes.c_ubyte * 6), 90), ('reserved_at_60', (ctypes.c_ubyte * 3), 96), ('log_eq_size', (ctypes.c_ubyte * 5), 99), ('uar_page', (ctypes.c_ubyte * 24), 104), ('reserved_at_80', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 20), 160), ('intr', (ctypes.c_ubyte * 12), 180), ('reserved_at_c0', (ctypes.c_ubyte * 3), 192), ('log_page_size', (ctypes.c_ubyte * 5), 195), ('reserved_at_c8', (ctypes.c_ubyte * 24), 200), ('reserved_at_e0', (ctypes.c_ubyte * 96), 224), ('reserved_at_140', (ctypes.c_ubyte * 8), 320), ('consumer_counter', (ctypes.c_ubyte * 24), 328), ('reserved_at_160', (ctypes.c_ubyte * 8), 352), ('producer_counter', (ctypes.c_ubyte * 24), 360), ('reserved_at_180', (ctypes.c_ubyte * 128), 384)])
_anonenum73: dict[int, str] = {(MLX5_DCTC_STATE_ACTIVE:=0): 'MLX5_DCTC_STATE_ACTIVE', (MLX5_DCTC_STATE_DRAINING:=1): 'MLX5_DCTC_STATE_DRAINING', (MLX5_DCTC_STATE_DRAINED:=2): 'MLX5_DCTC_STATE_DRAINED'}
_anonenum74: dict[int, str] = {(MLX5_DCTC_CS_RES_DISABLE:=0): 'MLX5_DCTC_CS_RES_DISABLE', (MLX5_DCTC_CS_RES_NA:=1): 'MLX5_DCTC_CS_RES_NA', (MLX5_DCTC_CS_RES_UP_TO_64B:=2): 'MLX5_DCTC_CS_RES_UP_TO_64B'}
_anonenum75: dict[int, str] = {(MLX5_DCTC_MTU_256_BYTES:=1): 'MLX5_DCTC_MTU_256_BYTES', (MLX5_DCTC_MTU_512_BYTES:=2): 'MLX5_DCTC_MTU_512_BYTES', (MLX5_DCTC_MTU_1K_BYTES:=3): 'MLX5_DCTC_MTU_1K_BYTES', (MLX5_DCTC_MTU_2K_BYTES:=4): 'MLX5_DCTC_MTU_2K_BYTES', (MLX5_DCTC_MTU_4K_BYTES:=5): 'MLX5_DCTC_MTU_4K_BYTES'}
@c.record
class struct_mlx5_ifc_dctc_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  state: 'list[int]'
  reserved_at_8: 'list[int]'
  reserved_at_20: 'list[int]'
  dp_ordering_force: 'list[int]'
  user_index: 'list[int]'
  reserved_at_40: 'list[int]'
  cqn: 'list[int]'
  counter_set_id: 'list[int]'
  atomic_mode: 'list[int]'
  rre: 'list[int]'
  rwe: 'list[int]'
  rae: 'list[int]'
  atomic_like_write_en: 'list[int]'
  latency_sensitive: 'list[int]'
  rlky: 'list[int]'
  free_ar: 'list[int]'
  reserved_at_73: 'list[int]'
  dp_ordering_1: 'list[int]'
  reserved_at_75: 'list[int]'
  reserved_at_80: 'list[int]'
  cs_res: 'list[int]'
  reserved_at_90: 'list[int]'
  min_rnr_nak: 'list[int]'
  reserved_at_98: 'list[int]'
  reserved_at_a0: 'list[int]'
  srqn_xrqn: 'list[int]'
  reserved_at_c0: 'list[int]'
  pd: 'list[int]'
  tclass: 'list[int]'
  reserved_at_e8: 'list[int]'
  flow_label: 'list[int]'
  dc_access_key: 'list[int]'
  reserved_at_140: 'list[int]'
  mtu: 'list[int]'
  port: 'list[int]'
  pkey_index: 'list[int]'
  reserved_at_160: 'list[int]'
  my_addr_index: 'list[int]'
  reserved_at_170: 'list[int]'
  hop_limit: 'list[int]'
  dc_access_key_violation_count: 'list[int]'
  reserved_at_1a0: 'list[int]'
  dei_cfi: 'list[int]'
  eth_prio: 'list[int]'
  ecn: 'list[int]'
  dscp: 'list[int]'
  reserved_at_1c0: 'list[int]'
  ece: 'list[int]'
struct_mlx5_ifc_dctc_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('state', (ctypes.c_ubyte * 4), 4), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('reserved_at_20', (ctypes.c_ubyte * 7), 32), ('dp_ordering_force', (ctypes.c_ubyte * 1), 39), ('user_index', (ctypes.c_ubyte * 24), 40), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('cqn', (ctypes.c_ubyte * 24), 72), ('counter_set_id', (ctypes.c_ubyte * 8), 96), ('atomic_mode', (ctypes.c_ubyte * 4), 104), ('rre', (ctypes.c_ubyte * 1), 108), ('rwe', (ctypes.c_ubyte * 1), 109), ('rae', (ctypes.c_ubyte * 1), 110), ('atomic_like_write_en', (ctypes.c_ubyte * 1), 111), ('latency_sensitive', (ctypes.c_ubyte * 1), 112), ('rlky', (ctypes.c_ubyte * 1), 113), ('free_ar', (ctypes.c_ubyte * 1), 114), ('reserved_at_73', (ctypes.c_ubyte * 1), 115), ('dp_ordering_1', (ctypes.c_ubyte * 1), 116), ('reserved_at_75', (ctypes.c_ubyte * 11), 117), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('cs_res', (ctypes.c_ubyte * 8), 136), ('reserved_at_90', (ctypes.c_ubyte * 3), 144), ('min_rnr_nak', (ctypes.c_ubyte * 5), 147), ('reserved_at_98', (ctypes.c_ubyte * 8), 152), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('srqn_xrqn', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('pd', (ctypes.c_ubyte * 24), 200), ('tclass', (ctypes.c_ubyte * 8), 224), ('reserved_at_e8', (ctypes.c_ubyte * 4), 232), ('flow_label', (ctypes.c_ubyte * 20), 236), ('dc_access_key', (ctypes.c_ubyte * 64), 256), ('reserved_at_140', (ctypes.c_ubyte * 5), 320), ('mtu', (ctypes.c_ubyte * 3), 325), ('port', (ctypes.c_ubyte * 8), 328), ('pkey_index', (ctypes.c_ubyte * 16), 336), ('reserved_at_160', (ctypes.c_ubyte * 8), 352), ('my_addr_index', (ctypes.c_ubyte * 8), 360), ('reserved_at_170', (ctypes.c_ubyte * 8), 368), ('hop_limit', (ctypes.c_ubyte * 8), 376), ('dc_access_key_violation_count', (ctypes.c_ubyte * 32), 384), ('reserved_at_1a0', (ctypes.c_ubyte * 20), 416), ('dei_cfi', (ctypes.c_ubyte * 1), 436), ('eth_prio', (ctypes.c_ubyte * 3), 437), ('ecn', (ctypes.c_ubyte * 2), 440), ('dscp', (ctypes.c_ubyte * 6), 442), ('reserved_at_1c0', (ctypes.c_ubyte * 32), 448), ('ece', (ctypes.c_ubyte * 32), 480)])
_anonenum76: dict[int, str] = {(MLX5_CQC_STATUS_OK:=0): 'MLX5_CQC_STATUS_OK', (MLX5_CQC_STATUS_CQ_OVERFLOW:=9): 'MLX5_CQC_STATUS_CQ_OVERFLOW', (MLX5_CQC_STATUS_CQ_WRITE_FAIL:=10): 'MLX5_CQC_STATUS_CQ_WRITE_FAIL'}
_anonenum77: dict[int, str] = {(MLX5_CQC_CQE_SZ_64_BYTES:=0): 'MLX5_CQC_CQE_SZ_64_BYTES', (MLX5_CQC_CQE_SZ_128_BYTES:=1): 'MLX5_CQC_CQE_SZ_128_BYTES'}
_anonenum78: dict[int, str] = {(MLX5_CQC_ST_SOLICITED_NOTIFICATION_REQUEST_ARMED:=6): 'MLX5_CQC_ST_SOLICITED_NOTIFICATION_REQUEST_ARMED', (MLX5_CQC_ST_NOTIFICATION_REQUEST_ARMED:=9): 'MLX5_CQC_ST_NOTIFICATION_REQUEST_ARMED', (MLX5_CQC_ST_FIRED:=10): 'MLX5_CQC_ST_FIRED'}
enum_mlx5_cq_period_mode: dict[int, str] = {(MLX5_CQ_PERIOD_MODE_START_FROM_EQE:=0): 'MLX5_CQ_PERIOD_MODE_START_FROM_EQE', (MLX5_CQ_PERIOD_MODE_START_FROM_CQE:=1): 'MLX5_CQ_PERIOD_MODE_START_FROM_CQE', (MLX5_CQ_PERIOD_NUM_MODES:=2): 'MLX5_CQ_PERIOD_NUM_MODES'}
@c.record
class struct_mlx5_ifc_cqc_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_4: 'list[int]'
  dbr_umem_valid: 'list[int]'
  apu_cq: 'list[int]'
  cqe_sz: 'list[int]'
  cc: 'list[int]'
  reserved_at_c: 'list[int]'
  scqe_break_moderation_en: 'list[int]'
  oi: 'list[int]'
  cq_period_mode: 'list[int]'
  cqe_comp_en: 'list[int]'
  mini_cqe_res_format: 'list[int]'
  st: 'list[int]'
  reserved_at_18: 'list[int]'
  cqe_compression_layout: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_40: 'list[int]'
  page_offset: 'list[int]'
  reserved_at_5a: 'list[int]'
  reserved_at_60: 'list[int]'
  log_cq_size: 'list[int]'
  uar_page: 'list[int]'
  reserved_at_80: 'list[int]'
  cq_period: 'list[int]'
  cq_max_count: 'list[int]'
  c_eqn_or_apu_element: 'list[int]'
  reserved_at_c0: 'list[int]'
  log_page_size: 'list[int]'
  reserved_at_c8: 'list[int]'
  reserved_at_e0: 'list[int]'
  reserved_at_100: 'list[int]'
  last_notified_index: 'list[int]'
  reserved_at_120: 'list[int]'
  last_solicit_index: 'list[int]'
  reserved_at_140: 'list[int]'
  consumer_counter: 'list[int]'
  reserved_at_160: 'list[int]'
  producer_counter: 'list[int]'
  reserved_at_180: 'list[int]'
  dbr_addr: 'list[int]'
struct_mlx5_ifc_cqc_bits.register_fields([('status', (ctypes.c_ubyte * 4), 0), ('reserved_at_4', (ctypes.c_ubyte * 2), 4), ('dbr_umem_valid', (ctypes.c_ubyte * 1), 6), ('apu_cq', (ctypes.c_ubyte * 1), 7), ('cqe_sz', (ctypes.c_ubyte * 3), 8), ('cc', (ctypes.c_ubyte * 1), 11), ('reserved_at_c', (ctypes.c_ubyte * 1), 12), ('scqe_break_moderation_en', (ctypes.c_ubyte * 1), 13), ('oi', (ctypes.c_ubyte * 1), 14), ('cq_period_mode', (ctypes.c_ubyte * 2), 15), ('cqe_comp_en', (ctypes.c_ubyte * 1), 17), ('mini_cqe_res_format', (ctypes.c_ubyte * 2), 18), ('st', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 6), 24), ('cqe_compression_layout', (ctypes.c_ubyte * 2), 30), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 20), 64), ('page_offset', (ctypes.c_ubyte * 6), 84), ('reserved_at_5a', (ctypes.c_ubyte * 6), 90), ('reserved_at_60', (ctypes.c_ubyte * 3), 96), ('log_cq_size', (ctypes.c_ubyte * 5), 99), ('uar_page', (ctypes.c_ubyte * 24), 104), ('reserved_at_80', (ctypes.c_ubyte * 4), 128), ('cq_period', (ctypes.c_ubyte * 12), 132), ('cq_max_count', (ctypes.c_ubyte * 16), 144), ('c_eqn_or_apu_element', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 3), 192), ('log_page_size', (ctypes.c_ubyte * 5), 195), ('reserved_at_c8', (ctypes.c_ubyte * 24), 200), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 8), 256), ('last_notified_index', (ctypes.c_ubyte * 24), 264), ('reserved_at_120', (ctypes.c_ubyte * 8), 288), ('last_solicit_index', (ctypes.c_ubyte * 24), 296), ('reserved_at_140', (ctypes.c_ubyte * 8), 320), ('consumer_counter', (ctypes.c_ubyte * 24), 328), ('reserved_at_160', (ctypes.c_ubyte * 8), 352), ('producer_counter', (ctypes.c_ubyte * 24), 360), ('reserved_at_180', (ctypes.c_ubyte * 64), 384), ('dbr_addr', (ctypes.c_ubyte * 64), 448)])
@c.record
class union_mlx5_ifc_cong_control_roce_ecn_auto_bits(c.Struct):
  SIZE = 2048
  cong_control_802_1qau_rp: 'struct_mlx5_ifc_cong_control_802_1qau_rp_bits'
  cong_control_r_roce_ecn_rp: 'struct_mlx5_ifc_cong_control_r_roce_ecn_rp_bits'
  cong_control_r_roce_ecn_np: 'struct_mlx5_ifc_cong_control_r_roce_ecn_np_bits'
  cong_control_r_roce_general: 'struct_mlx5_ifc_cong_control_r_roce_general_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_cong_control_roce_ecn_auto_bits.register_fields([('cong_control_802_1qau_rp', struct_mlx5_ifc_cong_control_802_1qau_rp_bits, 0), ('cong_control_r_roce_ecn_rp', struct_mlx5_ifc_cong_control_r_roce_ecn_rp_bits, 0), ('cong_control_r_roce_ecn_np', struct_mlx5_ifc_cong_control_r_roce_ecn_np_bits, 0), ('cong_control_r_roce_general', struct_mlx5_ifc_cong_control_r_roce_general_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 2048), 0)])
@c.record
class struct_mlx5_ifc_query_adapter_param_block_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  reserved_at_c0: 'list[int]'
  ieee_vendor_id: 'list[int]'
  reserved_at_e0: 'list[int]'
  vsd_vendor_id: 'list[int]'
  vsd: 'list[list[int]]'
  vsd_contd_psid: 'list[list[int]]'
struct_mlx5_ifc_query_adapter_param_block_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 192), 0), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('ieee_vendor_id', (ctypes.c_ubyte * 24), 200), ('reserved_at_e0', (ctypes.c_ubyte * 16), 224), ('vsd_vendor_id', (ctypes.c_ubyte * 16), 240), ('vsd', ((ctypes.c_ubyte * 8) * 208), 256), ('vsd_contd_psid', ((ctypes.c_ubyte * 8) * 16), 1920)])
_anonenum79: dict[int, str] = {(MLX5_XRQC_STATE_GOOD:=0): 'MLX5_XRQC_STATE_GOOD', (MLX5_XRQC_STATE_ERROR:=1): 'MLX5_XRQC_STATE_ERROR'}
_anonenum80: dict[int, str] = {(MLX5_XRQC_TOPOLOGY_NO_SPECIAL_TOPOLOGY:=0): 'MLX5_XRQC_TOPOLOGY_NO_SPECIAL_TOPOLOGY', (MLX5_XRQC_TOPOLOGY_TAG_MATCHING:=1): 'MLX5_XRQC_TOPOLOGY_TAG_MATCHING'}
_anonenum81: dict[int, str] = {(MLX5_XRQC_OFFLOAD_RNDV:=1): 'MLX5_XRQC_OFFLOAD_RNDV'}
@c.record
class struct_mlx5_ifc_tag_matching_topology_context_bits(c.Struct):
  SIZE = 128
  log_matching_list_sz: 'list[int]'
  reserved_at_4: 'list[int]'
  append_next_index: 'list[int]'
  sw_phase_cnt: 'list[int]'
  hw_phase_cnt: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_tag_matching_topology_context_bits.register_fields([('log_matching_list_sz', (ctypes.c_ubyte * 4), 0), ('reserved_at_4', (ctypes.c_ubyte * 12), 4), ('append_next_index', (ctypes.c_ubyte * 16), 16), ('sw_phase_cnt', (ctypes.c_ubyte * 16), 32), ('hw_phase_cnt', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_xrqc_bits(c.Struct):
  SIZE = 2560
  state: 'list[int]'
  rlkey: 'list[int]'
  reserved_at_5: 'list[int]'
  topology: 'list[int]'
  reserved_at_18: 'list[int]'
  offload: 'list[int]'
  reserved_at_20: 'list[int]'
  user_index: 'list[int]'
  reserved_at_40: 'list[int]'
  cqn: 'list[int]'
  reserved_at_60: 'list[int]'
  tag_matching_topology_context: 'struct_mlx5_ifc_tag_matching_topology_context_bits'
  reserved_at_180: 'list[int]'
  wq: 'struct_mlx5_ifc_wq_bits'
struct_mlx5_ifc_xrqc_bits.register_fields([('state', (ctypes.c_ubyte * 4), 0), ('rlkey', (ctypes.c_ubyte * 1), 4), ('reserved_at_5', (ctypes.c_ubyte * 15), 5), ('topology', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 4), 24), ('offload', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('user_index', (ctypes.c_ubyte * 24), 40), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('cqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 160), 96), ('tag_matching_topology_context', struct_mlx5_ifc_tag_matching_topology_context_bits, 256), ('reserved_at_180', (ctypes.c_ubyte * 640), 384), ('wq', struct_mlx5_ifc_wq_bits, 1024)])
@c.record
class union_mlx5_ifc_modify_field_select_resize_field_select_auto_bits(c.Struct):
  SIZE = 32
  modify_field_select: 'struct_mlx5_ifc_modify_field_select_bits'
  resize_field_select: 'struct_mlx5_ifc_resize_field_select_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_modify_field_select_resize_field_select_auto_bits.register_fields([('modify_field_select', struct_mlx5_ifc_modify_field_select_bits, 0), ('resize_field_select', struct_mlx5_ifc_resize_field_select_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 32), 0)])
@c.record
class union_mlx5_ifc_field_select_802_1_r_roce_auto_bits(c.Struct):
  SIZE = 32
  field_select_802_1qau_rp: 'struct_mlx5_ifc_field_select_802_1qau_rp_bits'
  field_select_r_roce_rp: 'struct_mlx5_ifc_field_select_r_roce_rp_bits'
  field_select_r_roce_np: 'struct_mlx5_ifc_field_select_r_roce_np_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_field_select_802_1_r_roce_auto_bits.register_fields([('field_select_802_1qau_rp', struct_mlx5_ifc_field_select_802_1qau_rp_bits, 0), ('field_select_r_roce_rp', struct_mlx5_ifc_field_select_r_roce_rp_bits, 0), ('field_select_r_roce_np', struct_mlx5_ifc_field_select_r_roce_np_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 32), 0)])
@c.record
class struct_mlx5_ifc_rs_histogram_cntrs_bits(c.Struct):
  SIZE = 1728
  hist: 'list[list[int]]'
  reserved_at_400: 'list[int]'
struct_mlx5_ifc_rs_histogram_cntrs_bits.register_fields([('hist', ((ctypes.c_ubyte * 64) * 16), 0), ('reserved_at_400', (ctypes.c_ubyte * 704), 1024)])
@c.record
class union_mlx5_ifc_eth_cntrs_grp_data_layout_auto_bits(c.Struct):
  SIZE = 1984
  eth_802_3_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits'
  eth_2863_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits'
  eth_2819_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits'
  eth_3635_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits'
  eth_extended_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits'
  eth_per_prio_grp_data_layout: 'struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits'
  eth_per_tc_prio_grp_data_layout: 'struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits'
  eth_per_tc_congest_prio_grp_data_layout: 'struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits'
  ib_port_cntrs_grp_data_layout: 'struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits'
  ib_ext_port_cntrs_grp_data_layout: 'struct_mlx5_ifc_ib_ext_port_cntrs_grp_data_layout_bits'
  phys_layer_cntrs: 'struct_mlx5_ifc_phys_layer_cntrs_bits'
  phys_layer_statistical_cntrs: 'struct_mlx5_ifc_phys_layer_statistical_cntrs_bits'
  phys_layer_recovery_cntrs: 'struct_mlx5_ifc_phys_layer_recovery_cntrs_bits'
  rs_histogram_cntrs: 'struct_mlx5_ifc_rs_histogram_cntrs_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_eth_cntrs_grp_data_layout_auto_bits.register_fields([('eth_802_3_cntrs_grp_data_layout', struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits, 0), ('eth_2863_cntrs_grp_data_layout', struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits, 0), ('eth_2819_cntrs_grp_data_layout', struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits, 0), ('eth_3635_cntrs_grp_data_layout', struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits, 0), ('eth_extended_cntrs_grp_data_layout', struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits, 0), ('eth_per_prio_grp_data_layout', struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits, 0), ('eth_per_tc_prio_grp_data_layout', struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits, 0), ('eth_per_tc_congest_prio_grp_data_layout', struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits, 0), ('ib_port_cntrs_grp_data_layout', struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits, 0), ('ib_ext_port_cntrs_grp_data_layout', struct_mlx5_ifc_ib_ext_port_cntrs_grp_data_layout_bits, 0), ('phys_layer_cntrs', struct_mlx5_ifc_phys_layer_cntrs_bits, 0), ('phys_layer_statistical_cntrs', struct_mlx5_ifc_phys_layer_statistical_cntrs_bits, 0), ('phys_layer_recovery_cntrs', struct_mlx5_ifc_phys_layer_recovery_cntrs_bits, 0), ('rs_histogram_cntrs', struct_mlx5_ifc_rs_histogram_cntrs_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 1984), 0)])
@c.record
class union_mlx5_ifc_pcie_cntrs_grp_data_layout_auto_bits(c.Struct):
  SIZE = 1984
  pcie_perf_cntrs_grp_data_layout: 'struct_mlx5_ifc_pcie_perf_cntrs_grp_data_layout_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_pcie_cntrs_grp_data_layout_auto_bits.register_fields([('pcie_perf_cntrs_grp_data_layout', struct_mlx5_ifc_pcie_perf_cntrs_grp_data_layout_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 1984), 0)])
@c.record
class union_mlx5_ifc_event_auto_bits(c.Struct):
  SIZE = 224
  comp_event: 'struct_mlx5_ifc_comp_event_bits'
  dct_events: 'struct_mlx5_ifc_dct_events_bits'
  qp_events: 'struct_mlx5_ifc_qp_events_bits'
  wqe_associated_page_fault_event: 'struct_mlx5_ifc_wqe_associated_page_fault_event_bits'
  rdma_page_fault_event: 'struct_mlx5_ifc_rdma_page_fault_event_bits'
  cq_error: 'struct_mlx5_ifc_cq_error_bits'
  dropped_packet_logged: 'struct_mlx5_ifc_dropped_packet_logged_bits'
  port_state_change_event: 'struct_mlx5_ifc_port_state_change_event_bits'
  gpio_event: 'struct_mlx5_ifc_gpio_event_bits'
  db_bf_congestion_event: 'struct_mlx5_ifc_db_bf_congestion_event_bits'
  stall_vl_event: 'struct_mlx5_ifc_stall_vl_event_bits'
  cmd_inter_comp_event: 'struct_mlx5_ifc_cmd_inter_comp_event_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_event_auto_bits.register_fields([('comp_event', struct_mlx5_ifc_comp_event_bits, 0), ('dct_events', struct_mlx5_ifc_dct_events_bits, 0), ('qp_events', struct_mlx5_ifc_qp_events_bits, 0), ('wqe_associated_page_fault_event', struct_mlx5_ifc_wqe_associated_page_fault_event_bits, 0), ('rdma_page_fault_event', struct_mlx5_ifc_rdma_page_fault_event_bits, 0), ('cq_error', struct_mlx5_ifc_cq_error_bits, 0), ('dropped_packet_logged', struct_mlx5_ifc_dropped_packet_logged_bits, 0), ('port_state_change_event', struct_mlx5_ifc_port_state_change_event_bits, 0), ('gpio_event', struct_mlx5_ifc_gpio_event_bits, 0), ('db_bf_congestion_event', struct_mlx5_ifc_db_bf_congestion_event_bits, 0), ('stall_vl_event', struct_mlx5_ifc_stall_vl_event_bits, 0), ('cmd_inter_comp_event', struct_mlx5_ifc_cmd_inter_comp_event_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 224), 0)])
@c.record
class struct_mlx5_ifc_health_buffer_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  assert_existptr: 'list[int]'
  assert_callra: 'list[int]'
  reserved_at_140: 'list[int]'
  time: 'list[int]'
  fw_version: 'list[int]'
  hw_id: 'list[int]'
  rfr: 'list[int]'
  reserved_at_1c1: 'list[int]'
  valid: 'list[int]'
  severity: 'list[int]'
  reserved_at_1c8: 'list[int]'
  irisc_index: 'list[int]'
  synd: 'list[int]'
  ext_synd: 'list[int]'
struct_mlx5_ifc_health_buffer_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 256), 0), ('assert_existptr', (ctypes.c_ubyte * 32), 256), ('assert_callra', (ctypes.c_ubyte * 32), 288), ('reserved_at_140', (ctypes.c_ubyte * 32), 320), ('time', (ctypes.c_ubyte * 32), 352), ('fw_version', (ctypes.c_ubyte * 32), 384), ('hw_id', (ctypes.c_ubyte * 32), 416), ('rfr', (ctypes.c_ubyte * 1), 448), ('reserved_at_1c1', (ctypes.c_ubyte * 3), 449), ('valid', (ctypes.c_ubyte * 1), 452), ('severity', (ctypes.c_ubyte * 3), 453), ('reserved_at_1c8', (ctypes.c_ubyte * 24), 456), ('irisc_index', (ctypes.c_ubyte * 8), 480), ('synd', (ctypes.c_ubyte * 8), 488), ('ext_synd', (ctypes.c_ubyte * 16), 496)])
@c.record
class struct_mlx5_ifc_register_loopback_control_bits(c.Struct):
  SIZE = 128
  no_lb: 'list[int]'
  reserved_at_1: 'list[int]'
  port: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_register_loopback_control_bits.register_fields([('no_lb', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 7), 1), ('port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 96), 32)])
_anonenum82: dict[int, str] = {(MLX5_TEARDOWN_HCA_OUT_FORCE_STATE_SUCCESS:=0): 'MLX5_TEARDOWN_HCA_OUT_FORCE_STATE_SUCCESS', (MLX5_TEARDOWN_HCA_OUT_FORCE_STATE_FAIL:=1): 'MLX5_TEARDOWN_HCA_OUT_FORCE_STATE_FAIL'}
@c.record
class struct_mlx5_ifc_teardown_hca_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  state: 'list[int]'
struct_mlx5_ifc_teardown_hca_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 63), 64), ('state', (ctypes.c_ubyte * 1), 127)])
_anonenum83: dict[int, str] = {(MLX5_TEARDOWN_HCA_IN_PROFILE_GRACEFUL_CLOSE:=0): 'MLX5_TEARDOWN_HCA_IN_PROFILE_GRACEFUL_CLOSE', (MLX5_TEARDOWN_HCA_IN_PROFILE_FORCE_CLOSE:=1): 'MLX5_TEARDOWN_HCA_IN_PROFILE_FORCE_CLOSE', (MLX5_TEARDOWN_HCA_IN_PROFILE_PREPARE_FAST_TEARDOWN:=2): 'MLX5_TEARDOWN_HCA_IN_PROFILE_PREPARE_FAST_TEARDOWN'}
@c.record
class struct_mlx5_ifc_teardown_hca_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  profile: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_teardown_hca_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('profile', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_sqerr2rts_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_sqerr2rts_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_sqerr2rts_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  opt_param_mask: 'list[int]'
  reserved_at_a0: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  reserved_at_800: 'list[int]'
struct_mlx5_ifc_sqerr2rts_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('reserved_at_800', (ctypes.c_ubyte * 128), 2048)])
@c.record
class struct_mlx5_ifc_sqd2rts_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_sqd2rts_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_sqd2rts_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  opt_param_mask: 'list[int]'
  reserved_at_a0: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  reserved_at_800: 'list[int]'
struct_mlx5_ifc_sqd2rts_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('reserved_at_800', (ctypes.c_ubyte * 128), 2048)])
@c.record
class struct_mlx5_ifc_set_roce_address_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_roce_address_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_roce_address_in_bits(c.Struct):
  SIZE = 384
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  roce_address_index: 'list[int]'
  reserved_at_50: 'list[int]'
  vhca_port_num: 'list[int]'
  reserved_at_60: 'list[int]'
  roce_address: 'struct_mlx5_ifc_roce_addr_layout_bits'
struct_mlx5_ifc_set_roce_address_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('roce_address_index', (ctypes.c_ubyte * 16), 64), ('reserved_at_50', (ctypes.c_ubyte * 12), 80), ('vhca_port_num', (ctypes.c_ubyte * 4), 92), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('roce_address', struct_mlx5_ifc_roce_addr_layout_bits, 128)])
@c.record
class struct_mlx5_ifc_set_mad_demux_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_mad_demux_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
_anonenum84: dict[int, str] = {(MLX5_SET_MAD_DEMUX_IN_DEMUX_MODE_PASS_ALL:=0): 'MLX5_SET_MAD_DEMUX_IN_DEMUX_MODE_PASS_ALL', (MLX5_SET_MAD_DEMUX_IN_DEMUX_MODE_SELECTIVE:=2): 'MLX5_SET_MAD_DEMUX_IN_DEMUX_MODE_SELECTIVE'}
@c.record
class struct_mlx5_ifc_set_mad_demux_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  demux_mode: 'list[int]'
  reserved_at_68: 'list[int]'
struct_mlx5_ifc_set_mad_demux_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 6), 96), ('demux_mode', (ctypes.c_ubyte * 2), 102), ('reserved_at_68', (ctypes.c_ubyte * 24), 104)])
@c.record
class struct_mlx5_ifc_set_l2_table_entry_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_l2_table_entry_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_l2_table_entry_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_index: 'list[int]'
  reserved_at_c0: 'list[int]'
  reserved_at_e0: 'list[int]'
  silent_mode_valid: 'list[int]'
  silent_mode: 'list[int]'
  reserved_at_f2: 'list[int]'
  vlan_valid: 'list[int]'
  vlan: 'list[int]'
  mac_address: 'struct_mlx5_ifc_mac_address_layout_bits'
  reserved_at_140: 'list[int]'
struct_mlx5_ifc_set_l2_table_entry_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 96), 64), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_index', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 16), 224), ('silent_mode_valid', (ctypes.c_ubyte * 1), 240), ('silent_mode', (ctypes.c_ubyte * 1), 241), ('reserved_at_f2', (ctypes.c_ubyte * 1), 242), ('vlan_valid', (ctypes.c_ubyte * 1), 243), ('vlan', (ctypes.c_ubyte * 12), 244), ('mac_address', struct_mlx5_ifc_mac_address_layout_bits, 256), ('reserved_at_140', (ctypes.c_ubyte * 192), 320)])
@c.record
class struct_mlx5_ifc_set_issi_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_issi_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_issi_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  current_issi: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_set_issi_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('current_issi', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_set_hca_cap_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_hca_cap_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_hca_cap_in_bits(c.Struct):
  SIZE = 32896
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_function: 'list[int]'
  ec_vf_function: 'list[int]'
  reserved_at_42: 'list[int]'
  function_id_type: 'list[int]'
  reserved_at_44: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
  capability: 'union_mlx5_ifc_hca_cap_union_bits'
struct_mlx5_ifc_set_hca_cap_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_function', (ctypes.c_ubyte * 1), 64), ('ec_vf_function', (ctypes.c_ubyte * 1), 65), ('reserved_at_42', (ctypes.c_ubyte * 1), 66), ('function_id_type', (ctypes.c_ubyte * 1), 67), ('reserved_at_44', (ctypes.c_ubyte * 12), 68), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('capability', union_mlx5_ifc_hca_cap_union_bits, 128)])
_anonenum85: dict[int, str] = {(MLX5_SET_FTE_MODIFY_ENABLE_MASK_ACTION:=0): 'MLX5_SET_FTE_MODIFY_ENABLE_MASK_ACTION', (MLX5_SET_FTE_MODIFY_ENABLE_MASK_FLOW_TAG:=1): 'MLX5_SET_FTE_MODIFY_ENABLE_MASK_FLOW_TAG', (MLX5_SET_FTE_MODIFY_ENABLE_MASK_DESTINATION_LIST:=2): 'MLX5_SET_FTE_MODIFY_ENABLE_MASK_DESTINATION_LIST', (MLX5_SET_FTE_MODIFY_ENABLE_MASK_FLOW_COUNTERS:=3): 'MLX5_SET_FTE_MODIFY_ENABLE_MASK_FLOW_COUNTERS', (MLX5_SET_FTE_MODIFY_ENABLE_MASK_IPSEC_OBJ_ID:=4): 'MLX5_SET_FTE_MODIFY_ENABLE_MASK_IPSEC_OBJ_ID'}
@c.record
class struct_mlx5_ifc_set_fte_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_fte_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_fte_in_bits(c.Struct):
  SIZE = 6656
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  ignore_flow_level: 'list[int]'
  reserved_at_c1: 'list[int]'
  modify_enable_mask: 'list[int]'
  reserved_at_e0: 'list[int]'
  flow_index: 'list[int]'
  reserved_at_120: 'list[int]'
  flow_context: 'struct_mlx5_ifc_flow_context_bits'
struct_mlx5_ifc_set_fte_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('ignore_flow_level', (ctypes.c_ubyte * 1), 192), ('reserved_at_c1', (ctypes.c_ubyte * 23), 193), ('modify_enable_mask', (ctypes.c_ubyte * 8), 216), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224), ('flow_index', (ctypes.c_ubyte * 32), 256), ('reserved_at_120', (ctypes.c_ubyte * 224), 288), ('flow_context', struct_mlx5_ifc_flow_context_bits, 512)])
@c.record
class struct_mlx5_ifc_dest_format_bits(c.Struct):
  SIZE = 64
  destination_type: 'list[int]'
  destination_id: 'list[int]'
  destination_eswitch_owner_vhca_id_valid: 'list[int]'
  packet_reformat: 'list[int]'
  reserved_at_22: 'list[int]'
  destination_eswitch_owner_vhca_id: 'list[int]'
struct_mlx5_ifc_dest_format_bits.register_fields([('destination_type', (ctypes.c_ubyte * 8), 0), ('destination_id', (ctypes.c_ubyte * 24), 8), ('destination_eswitch_owner_vhca_id_valid', (ctypes.c_ubyte * 1), 32), ('packet_reformat', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 14), 34), ('destination_eswitch_owner_vhca_id', (ctypes.c_ubyte * 16), 48)])
@c.record
class struct_mlx5_ifc_rts2rts_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  ece: 'list[int]'
struct_mlx5_ifc_rts2rts_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('ece', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_rts2rts_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  opt_param_mask: 'list[int]'
  ece: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  reserved_at_800: 'list[int]'
struct_mlx5_ifc_rts2rts_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('ece', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('reserved_at_800', (ctypes.c_ubyte * 128), 2048)])
@c.record
class struct_mlx5_ifc_rtr2rts_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  ece: 'list[int]'
struct_mlx5_ifc_rtr2rts_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('ece', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_rtr2rts_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  opt_param_mask: 'list[int]'
  ece: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  reserved_at_800: 'list[int]'
struct_mlx5_ifc_rtr2rts_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('ece', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('reserved_at_800', (ctypes.c_ubyte * 128), 2048)])
@c.record
class struct_mlx5_ifc_rst2init_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  ece: 'list[int]'
struct_mlx5_ifc_rst2init_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('ece', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_rst2init_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  opt_param_mask: 'list[int]'
  ece: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  reserved_at_800: 'list[int]'
struct_mlx5_ifc_rst2init_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('ece', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('reserved_at_800', (ctypes.c_ubyte * 128), 2048)])
@c.record
class struct_mlx5_ifc_query_xrq_out_bits(c.Struct):
  SIZE = 2688
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  xrq_context: 'struct_mlx5_ifc_xrqc_bits'
struct_mlx5_ifc_query_xrq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('xrq_context', struct_mlx5_ifc_xrqc_bits, 128)])
@c.record
class struct_mlx5_ifc_query_xrq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_xrq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_xrc_srq_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  xrc_srq_context_entry: 'struct_mlx5_ifc_xrc_srqc_bits'
  reserved_at_280: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_query_xrc_srq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('xrc_srq_context_entry', struct_mlx5_ifc_xrc_srqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 1536), 640), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_query_xrc_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrc_srqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_xrc_srq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrc_srqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
_anonenum86: dict[int, str] = {(MLX5_QUERY_VPORT_STATE_OUT_STATE_DOWN:=0): 'MLX5_QUERY_VPORT_STATE_OUT_STATE_DOWN', (MLX5_QUERY_VPORT_STATE_OUT_STATE_UP:=1): 'MLX5_QUERY_VPORT_STATE_OUT_STATE_UP'}
@c.record
class struct_mlx5_ifc_query_vport_state_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  admin_state: 'list[int]'
  state: 'list[int]'
struct_mlx5_ifc_query_vport_state_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 24), 96), ('admin_state', (ctypes.c_ubyte * 4), 120), ('state', (ctypes.c_ubyte * 4), 124)])
@c.record
class struct_mlx5_ifc_array1024_auto_bits(c.Struct):
  SIZE = 1024
  array1024_auto: 'list[list[int]]'
struct_mlx5_ifc_array1024_auto_bits.register_fields([('array1024_auto', ((ctypes.c_ubyte * 32) * 32), 0)])
@c.record
class struct_mlx5_ifc_query_vuid_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  query_vfs_vuid: 'list[int]'
  data_direct: 'list[int]'
  reserved_at_62: 'list[int]'
  vhca_id: 'list[int]'
struct_mlx5_ifc_query_vuid_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 64), 32), ('query_vfs_vuid', (ctypes.c_ubyte * 1), 96), ('data_direct', (ctypes.c_ubyte * 1), 97), ('reserved_at_62', (ctypes.c_ubyte * 14), 98), ('vhca_id', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_query_vuid_out_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_1e0: 'list[int]'
  num_of_entries: 'list[int]'
  vuid: 'list[struct_mlx5_ifc_array1024_auto_bits]'
struct_mlx5_ifc_query_vuid_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 416), 64), ('reserved_at_1e0', (ctypes.c_ubyte * 16), 480), ('num_of_entries', (ctypes.c_ubyte * 16), 496), ('vuid', (struct_mlx5_ifc_array1024_auto_bits * 0), 512)])
_anonenum87: dict[int, str] = {(MLX5_VPORT_STATE_OP_MOD_VNIC_VPORT:=0): 'MLX5_VPORT_STATE_OP_MOD_VNIC_VPORT', (MLX5_VPORT_STATE_OP_MOD_ESW_VPORT:=1): 'MLX5_VPORT_STATE_OP_MOD_ESW_VPORT', (MLX5_VPORT_STATE_OP_MOD_UPLINK:=2): 'MLX5_VPORT_STATE_OP_MOD_UPLINK'}
@c.record
class struct_mlx5_ifc_arm_monitor_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_arm_monitor_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_arm_monitor_counter_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_arm_monitor_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
_anonenum88: dict[int, str] = {(MLX5_QUERY_MONITOR_CNT_TYPE_PPCNT:=0): 'MLX5_QUERY_MONITOR_CNT_TYPE_PPCNT', (MLX5_QUERY_MONITOR_CNT_TYPE_Q_COUNTER:=1): 'MLX5_QUERY_MONITOR_CNT_TYPE_Q_COUNTER'}
enum_mlx5_monitor_counter_ppcnt: dict[int, str] = {(MLX5_QUERY_MONITOR_PPCNT_IN_RANGE_LENGTH_ERRORS:=0): 'MLX5_QUERY_MONITOR_PPCNT_IN_RANGE_LENGTH_ERRORS', (MLX5_QUERY_MONITOR_PPCNT_OUT_OF_RANGE_LENGTH_FIELD:=1): 'MLX5_QUERY_MONITOR_PPCNT_OUT_OF_RANGE_LENGTH_FIELD', (MLX5_QUERY_MONITOR_PPCNT_FRAME_TOO_LONG_ERRORS:=2): 'MLX5_QUERY_MONITOR_PPCNT_FRAME_TOO_LONG_ERRORS', (MLX5_QUERY_MONITOR_PPCNT_FRAME_CHECK_SEQUENCE_ERRORS:=3): 'MLX5_QUERY_MONITOR_PPCNT_FRAME_CHECK_SEQUENCE_ERRORS', (MLX5_QUERY_MONITOR_PPCNT_ALIGNMENT_ERRORS:=4): 'MLX5_QUERY_MONITOR_PPCNT_ALIGNMENT_ERRORS', (MLX5_QUERY_MONITOR_PPCNT_IF_OUT_DISCARDS:=5): 'MLX5_QUERY_MONITOR_PPCNT_IF_OUT_DISCARDS'}
_anonenum89: dict[int, str] = {(MLX5_QUERY_MONITOR_Q_COUNTER_RX_OUT_OF_BUFFER:=4): 'MLX5_QUERY_MONITOR_Q_COUNTER_RX_OUT_OF_BUFFER'}
@c.record
class struct_mlx5_ifc_monitor_counter_output_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  type: 'list[int]'
  reserved_at_8: 'list[int]'
  counter: 'list[int]'
  counter_group_id: 'list[int]'
struct_mlx5_ifc_monitor_counter_output_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('type', (ctypes.c_ubyte * 4), 4), ('reserved_at_8', (ctypes.c_ubyte * 8), 8), ('counter', (ctypes.c_ubyte * 16), 16), ('counter_group_id', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_set_monitor_counter_in_bits(c.Struct):
  SIZE = 576
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  num_of_counters: 'list[int]'
  reserved_at_60: 'list[int]'
  monitor_counter: 'list[struct_mlx5_ifc_monitor_counter_output_bits]'
struct_mlx5_ifc_set_monitor_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('num_of_counters', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('monitor_counter', (struct_mlx5_ifc_monitor_counter_output_bits * 7), 128)])
@c.record
class struct_mlx5_ifc_set_monitor_counter_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_monitor_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_query_vport_state_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_vport_state_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_vnic_env_out_bits(c.Struct):
  SIZE = 4224
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  vport_env: 'struct_mlx5_ifc_vnic_diagnostic_statistics_bits'
struct_mlx5_ifc_query_vnic_env_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('vport_env', struct_mlx5_ifc_vnic_diagnostic_statistics_bits, 128)])
_anonenum90: dict[int, str] = {(MLX5_QUERY_VNIC_ENV_IN_OP_MOD_VPORT_DIAG_STATISTICS:=0): 'MLX5_QUERY_VNIC_ENV_IN_OP_MOD_VPORT_DIAG_STATISTICS'}
@c.record
class struct_mlx5_ifc_query_vnic_env_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_vnic_env_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_vport_counter_out_bits(c.Struct):
  SIZE = 4224
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  received_errors: 'struct_mlx5_ifc_traffic_counter_bits'
  transmit_errors: 'struct_mlx5_ifc_traffic_counter_bits'
  received_ib_unicast: 'struct_mlx5_ifc_traffic_counter_bits'
  transmitted_ib_unicast: 'struct_mlx5_ifc_traffic_counter_bits'
  received_ib_multicast: 'struct_mlx5_ifc_traffic_counter_bits'
  transmitted_ib_multicast: 'struct_mlx5_ifc_traffic_counter_bits'
  received_eth_broadcast: 'struct_mlx5_ifc_traffic_counter_bits'
  transmitted_eth_broadcast: 'struct_mlx5_ifc_traffic_counter_bits'
  received_eth_unicast: 'struct_mlx5_ifc_traffic_counter_bits'
  transmitted_eth_unicast: 'struct_mlx5_ifc_traffic_counter_bits'
  received_eth_multicast: 'struct_mlx5_ifc_traffic_counter_bits'
  transmitted_eth_multicast: 'struct_mlx5_ifc_traffic_counter_bits'
  local_loopback: 'struct_mlx5_ifc_traffic_counter_bits'
  reserved_at_700: 'list[int]'
struct_mlx5_ifc_query_vport_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('received_errors', struct_mlx5_ifc_traffic_counter_bits, 128), ('transmit_errors', struct_mlx5_ifc_traffic_counter_bits, 256), ('received_ib_unicast', struct_mlx5_ifc_traffic_counter_bits, 384), ('transmitted_ib_unicast', struct_mlx5_ifc_traffic_counter_bits, 512), ('received_ib_multicast', struct_mlx5_ifc_traffic_counter_bits, 640), ('transmitted_ib_multicast', struct_mlx5_ifc_traffic_counter_bits, 768), ('received_eth_broadcast', struct_mlx5_ifc_traffic_counter_bits, 896), ('transmitted_eth_broadcast', struct_mlx5_ifc_traffic_counter_bits, 1024), ('received_eth_unicast', struct_mlx5_ifc_traffic_counter_bits, 1152), ('transmitted_eth_unicast', struct_mlx5_ifc_traffic_counter_bits, 1280), ('received_eth_multicast', struct_mlx5_ifc_traffic_counter_bits, 1408), ('transmitted_eth_multicast', struct_mlx5_ifc_traffic_counter_bits, 1536), ('local_loopback', struct_mlx5_ifc_traffic_counter_bits, 1664), ('reserved_at_700', (ctypes.c_ubyte * 2432), 1792)])
_anonenum91: dict[int, str] = {(MLX5_QUERY_VPORT_COUNTER_IN_OP_MOD_VPORT_COUNTERS:=0): 'MLX5_QUERY_VPORT_COUNTER_IN_OP_MOD_VPORT_COUNTERS'}
@c.record
class struct_mlx5_ifc_query_vport_counter_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  port_num: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  clear: 'list[int]'
  reserved_at_c1: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_query_vport_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 11), 65), ('port_num', (ctypes.c_ubyte * 4), 76), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 96), 96), ('clear', (ctypes.c_ubyte * 1), 192), ('reserved_at_c1', (ctypes.c_ubyte * 31), 193), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_query_tis_out_bits(c.Struct):
  SIZE = 1408
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  tis_context: 'struct_mlx5_ifc_tisc_bits'
struct_mlx5_ifc_query_tis_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('tis_context', struct_mlx5_ifc_tisc_bits, 128)])
@c.record
class struct_mlx5_ifc_query_tis_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  tisn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_tis_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('tisn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_tir_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  tir_context: 'struct_mlx5_ifc_tirc_bits'
struct_mlx5_ifc_query_tir_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('tir_context', struct_mlx5_ifc_tirc_bits, 256)])
@c.record
class struct_mlx5_ifc_query_tir_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  tirn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_tir_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('tirn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_srq_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  srq_context_entry: 'struct_mlx5_ifc_srqc_bits'
  reserved_at_280: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_query_srq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('srq_context_entry', struct_mlx5_ifc_srqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 1536), 640), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_query_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  srqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_srq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('srqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_sq_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  sq_context: 'struct_mlx5_ifc_sqc_bits'
struct_mlx5_ifc_query_sq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('sq_context', struct_mlx5_ifc_sqc_bits, 256)])
@c.record
class struct_mlx5_ifc_query_sq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  sqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_sq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('sqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_special_contexts_out_bits(c.Struct):
  SIZE = 256
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  dump_fill_mkey: 'list[int]'
  resd_lkey: 'list[int]'
  null_mkey: 'list[int]'
  terminate_scatter_list_mkey: 'list[int]'
  repeated_mkey: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_query_special_contexts_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('dump_fill_mkey', (ctypes.c_ubyte * 32), 64), ('resd_lkey', (ctypes.c_ubyte * 32), 96), ('null_mkey', (ctypes.c_ubyte * 32), 128), ('terminate_scatter_list_mkey', (ctypes.c_ubyte * 32), 160), ('repeated_mkey', (ctypes.c_ubyte * 32), 192), ('reserved_at_a0', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_query_special_contexts_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_special_contexts_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_query_scheduling_element_out_bits(c.Struct):
  SIZE = 1024
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  scheduling_context: 'struct_mlx5_ifc_scheduling_context_bits'
  reserved_at_300: 'list[int]'
struct_mlx5_ifc_query_scheduling_element_out_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('scheduling_context', struct_mlx5_ifc_scheduling_context_bits, 256), ('reserved_at_300', (ctypes.c_ubyte * 256), 768)])
_anonenum92: dict[int, str] = {(SCHEDULING_HIERARCHY_E_SWITCH:=2): 'SCHEDULING_HIERARCHY_E_SWITCH', (SCHEDULING_HIERARCHY_NIC:=3): 'SCHEDULING_HIERARCHY_NIC'}
@c.record
class struct_mlx5_ifc_query_scheduling_element_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  scheduling_hierarchy: 'list[int]'
  reserved_at_48: 'list[int]'
  scheduling_element_id: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_query_scheduling_element_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('scheduling_hierarchy', (ctypes.c_ubyte * 8), 64), ('reserved_at_48', (ctypes.c_ubyte * 24), 72), ('scheduling_element_id', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
@c.record
class struct_mlx5_ifc_query_rqt_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rqt_context: 'struct_mlx5_ifc_rqtc_bits'
struct_mlx5_ifc_query_rqt_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('rqt_context', struct_mlx5_ifc_rqtc_bits, 256)])
@c.record
class struct_mlx5_ifc_query_rqt_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rqtn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_rqt_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rqtn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_rq_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rq_context: 'struct_mlx5_ifc_rqc_bits'
struct_mlx5_ifc_query_rq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('rq_context', struct_mlx5_ifc_rqc_bits, 256)])
@c.record
class struct_mlx5_ifc_query_rq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_rq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_roce_address_out_bits(c.Struct):
  SIZE = 384
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  roce_address: 'struct_mlx5_ifc_roce_addr_layout_bits'
struct_mlx5_ifc_query_roce_address_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('roce_address', struct_mlx5_ifc_roce_addr_layout_bits, 128)])
@c.record
class struct_mlx5_ifc_query_roce_address_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  roce_address_index: 'list[int]'
  reserved_at_50: 'list[int]'
  vhca_port_num: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_roce_address_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('roce_address_index', (ctypes.c_ubyte * 16), 64), ('reserved_at_50', (ctypes.c_ubyte * 12), 80), ('vhca_port_num', (ctypes.c_ubyte * 4), 92), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_rmp_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rmp_context: 'struct_mlx5_ifc_rmpc_bits'
struct_mlx5_ifc_query_rmp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('rmp_context', struct_mlx5_ifc_rmpc_bits, 256)])
@c.record
class struct_mlx5_ifc_query_rmp_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rmpn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_rmp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rmpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_cqe_error_syndrome_bits(c.Struct):
  SIZE = 32
  hw_error_syndrome: 'list[int]'
  hw_syndrome_type: 'list[int]'
  reserved_at_c: 'list[int]'
  vendor_error_syndrome: 'list[int]'
  syndrome: 'list[int]'
struct_mlx5_ifc_cqe_error_syndrome_bits.register_fields([('hw_error_syndrome', (ctypes.c_ubyte * 8), 0), ('hw_syndrome_type', (ctypes.c_ubyte * 4), 8), ('reserved_at_c', (ctypes.c_ubyte * 4), 12), ('vendor_error_syndrome', (ctypes.c_ubyte * 8), 16), ('syndrome', (ctypes.c_ubyte * 8), 24)])
@c.record
class struct_mlx5_ifc_qp_context_extension_bits(c.Struct):
  SIZE = 1536
  reserved_at_0: 'list[int]'
  error_syndrome: 'struct_mlx5_ifc_cqe_error_syndrome_bits'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_qp_context_extension_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 96), 0), ('error_syndrome', struct_mlx5_ifc_cqe_error_syndrome_bits, 96), ('reserved_at_80', (ctypes.c_ubyte * 1408), 128)])
@c.record
class struct_mlx5_ifc_qpc_extension_and_pas_list_in_bits(c.Struct):
  SIZE = 1536
  qpc_data_extension: 'struct_mlx5_ifc_qp_context_extension_bits'
  pas: 'list[list[int]]'
struct_mlx5_ifc_qpc_extension_and_pas_list_in_bits.register_fields([('qpc_data_extension', struct_mlx5_ifc_qp_context_extension_bits, 0), ('pas', ((ctypes.c_ubyte * 64) * 0), 1536)])
@c.record
class struct_mlx5_ifc_qp_pas_list_in_bits(c.Struct):
  SIZE = 0
  pas: 'list[struct_mlx5_ifc_cmd_pas_bits]'
struct_mlx5_ifc_qp_pas_list_in_bits.register_fields([('pas', (struct_mlx5_ifc_cmd_pas_bits * 0), 0)])
@c.record
class union_mlx5_ifc_qp_pas_or_qpc_ext_and_pas_bits(c.Struct):
  SIZE = 1536
  qp_pas_list: 'struct_mlx5_ifc_qp_pas_list_in_bits'
  qpc_ext_and_pas_list: 'struct_mlx5_ifc_qpc_extension_and_pas_list_in_bits'
union_mlx5_ifc_qp_pas_or_qpc_ext_and_pas_bits.register_fields([('qp_pas_list', struct_mlx5_ifc_qp_pas_list_in_bits, 0), ('qpc_ext_and_pas_list', struct_mlx5_ifc_qpc_extension_and_pas_list_in_bits, 0)])
@c.record
class struct_mlx5_ifc_query_qp_out_bits(c.Struct):
  SIZE = 3712
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  opt_param_mask: 'list[int]'
  ece: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  reserved_at_800: 'list[int]'
  qp_pas_or_qpc_ext_and_pas: 'union_mlx5_ifc_qp_pas_or_qpc_ext_and_pas_bits'
struct_mlx5_ifc_query_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('ece', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('reserved_at_800', (ctypes.c_ubyte * 128), 2048), ('qp_pas_or_qpc_ext_and_pas', union_mlx5_ifc_qp_pas_or_qpc_ext_and_pas_bits, 2176)])
@c.record
class struct_mlx5_ifc_query_qp_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  qpc_ext: 'list[int]'
  reserved_at_41: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('qpc_ext', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 7), 65), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_q_counter_out_bits(c.Struct):
  SIZE = 2048
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rx_write_requests: 'list[int]'
  reserved_at_a0: 'list[int]'
  rx_read_requests: 'list[int]'
  reserved_at_e0: 'list[int]'
  rx_atomic_requests: 'list[int]'
  reserved_at_120: 'list[int]'
  rx_dct_connect: 'list[int]'
  reserved_at_160: 'list[int]'
  out_of_buffer: 'list[int]'
  reserved_at_1a0: 'list[int]'
  out_of_sequence: 'list[int]'
  reserved_at_1e0: 'list[int]'
  duplicate_request: 'list[int]'
  reserved_at_220: 'list[int]'
  rnr_nak_retry_err: 'list[int]'
  reserved_at_260: 'list[int]'
  packet_seq_err: 'list[int]'
  reserved_at_2a0: 'list[int]'
  implied_nak_seq_err: 'list[int]'
  reserved_at_2e0: 'list[int]'
  local_ack_timeout_err: 'list[int]'
  reserved_at_320: 'list[int]'
  req_rnr_retries_exceeded: 'list[int]'
  reserved_at_3a0: 'list[int]'
  resp_local_length_error: 'list[int]'
  req_local_length_error: 'list[int]'
  resp_local_qp_error: 'list[int]'
  local_operation_error: 'list[int]'
  resp_local_protection: 'list[int]'
  req_local_protection: 'list[int]'
  resp_cqe_error: 'list[int]'
  req_cqe_error: 'list[int]'
  req_mw_binding: 'list[int]'
  req_bad_response: 'list[int]'
  req_remote_invalid_request: 'list[int]'
  resp_remote_invalid_request: 'list[int]'
  req_remote_access_errors: 'list[int]'
  resp_remote_access_errors: 'list[int]'
  req_remote_operation_errors: 'list[int]'
  req_transport_retries_exceeded: 'list[int]'
  cq_overflow: 'list[int]'
  resp_cqe_flush_error: 'list[int]'
  req_cqe_flush_error: 'list[int]'
  reserved_at_620: 'list[int]'
  roce_adp_retrans: 'list[int]'
  roce_adp_retrans_to: 'list[int]'
  roce_slow_restart: 'list[int]'
  roce_slow_restart_cnps: 'list[int]'
  roce_slow_restart_trans: 'list[int]'
  reserved_at_6e0: 'list[int]'
struct_mlx5_ifc_query_q_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('rx_write_requests', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('rx_read_requests', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224), ('rx_atomic_requests', (ctypes.c_ubyte * 32), 256), ('reserved_at_120', (ctypes.c_ubyte * 32), 288), ('rx_dct_connect', (ctypes.c_ubyte * 32), 320), ('reserved_at_160', (ctypes.c_ubyte * 32), 352), ('out_of_buffer', (ctypes.c_ubyte * 32), 384), ('reserved_at_1a0', (ctypes.c_ubyte * 32), 416), ('out_of_sequence', (ctypes.c_ubyte * 32), 448), ('reserved_at_1e0', (ctypes.c_ubyte * 32), 480), ('duplicate_request', (ctypes.c_ubyte * 32), 512), ('reserved_at_220', (ctypes.c_ubyte * 32), 544), ('rnr_nak_retry_err', (ctypes.c_ubyte * 32), 576), ('reserved_at_260', (ctypes.c_ubyte * 32), 608), ('packet_seq_err', (ctypes.c_ubyte * 32), 640), ('reserved_at_2a0', (ctypes.c_ubyte * 32), 672), ('implied_nak_seq_err', (ctypes.c_ubyte * 32), 704), ('reserved_at_2e0', (ctypes.c_ubyte * 32), 736), ('local_ack_timeout_err', (ctypes.c_ubyte * 32), 768), ('reserved_at_320', (ctypes.c_ubyte * 96), 800), ('req_rnr_retries_exceeded', (ctypes.c_ubyte * 32), 896), ('reserved_at_3a0', (ctypes.c_ubyte * 32), 928), ('resp_local_length_error', (ctypes.c_ubyte * 32), 960), ('req_local_length_error', (ctypes.c_ubyte * 32), 992), ('resp_local_qp_error', (ctypes.c_ubyte * 32), 1024), ('local_operation_error', (ctypes.c_ubyte * 32), 1056), ('resp_local_protection', (ctypes.c_ubyte * 32), 1088), ('req_local_protection', (ctypes.c_ubyte * 32), 1120), ('resp_cqe_error', (ctypes.c_ubyte * 32), 1152), ('req_cqe_error', (ctypes.c_ubyte * 32), 1184), ('req_mw_binding', (ctypes.c_ubyte * 32), 1216), ('req_bad_response', (ctypes.c_ubyte * 32), 1248), ('req_remote_invalid_request', (ctypes.c_ubyte * 32), 1280), ('resp_remote_invalid_request', (ctypes.c_ubyte * 32), 1312), ('req_remote_access_errors', (ctypes.c_ubyte * 32), 1344), ('resp_remote_access_errors', (ctypes.c_ubyte * 32), 1376), ('req_remote_operation_errors', (ctypes.c_ubyte * 32), 1408), ('req_transport_retries_exceeded', (ctypes.c_ubyte * 32), 1440), ('cq_overflow', (ctypes.c_ubyte * 32), 1472), ('resp_cqe_flush_error', (ctypes.c_ubyte * 32), 1504), ('req_cqe_flush_error', (ctypes.c_ubyte * 32), 1536), ('reserved_at_620', (ctypes.c_ubyte * 32), 1568), ('roce_adp_retrans', (ctypes.c_ubyte * 32), 1600), ('roce_adp_retrans_to', (ctypes.c_ubyte * 32), 1632), ('roce_slow_restart', (ctypes.c_ubyte * 32), 1664), ('roce_slow_restart_cnps', (ctypes.c_ubyte * 32), 1696), ('roce_slow_restart_trans', (ctypes.c_ubyte * 32), 1728), ('reserved_at_6e0', (ctypes.c_ubyte * 288), 1760)])
@c.record
class struct_mlx5_ifc_query_q_counter_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  clear: 'list[int]'
  aggregate: 'list[int]'
  reserved_at_c2: 'list[int]'
  reserved_at_e0: 'list[int]'
  counter_set_id: 'list[int]'
struct_mlx5_ifc_query_q_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 96), 96), ('clear', (ctypes.c_ubyte * 1), 192), ('aggregate', (ctypes.c_ubyte * 1), 193), ('reserved_at_c2', (ctypes.c_ubyte * 30), 194), ('reserved_at_e0', (ctypes.c_ubyte * 24), 224), ('counter_set_id', (ctypes.c_ubyte * 8), 248)])
@c.record
class struct_mlx5_ifc_query_pages_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  embedded_cpu_function: 'list[int]'
  reserved_at_41: 'list[int]'
  function_id: 'list[int]'
  num_pages: 'list[int]'
struct_mlx5_ifc_query_pages_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('embedded_cpu_function', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('function_id', (ctypes.c_ubyte * 16), 80), ('num_pages', (ctypes.c_ubyte * 32), 96)])
_anonenum93: dict[int, str] = {(MLX5_QUERY_PAGES_IN_OP_MOD_BOOT_PAGES:=1): 'MLX5_QUERY_PAGES_IN_OP_MOD_BOOT_PAGES', (MLX5_QUERY_PAGES_IN_OP_MOD_INIT_PAGES:=2): 'MLX5_QUERY_PAGES_IN_OP_MOD_INIT_PAGES', (MLX5_QUERY_PAGES_IN_OP_MOD_REGULAR_PAGES:=3): 'MLX5_QUERY_PAGES_IN_OP_MOD_REGULAR_PAGES'}
@c.record
class struct_mlx5_ifc_query_pages_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  embedded_cpu_function: 'list[int]'
  reserved_at_41: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_pages_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('embedded_cpu_function', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_nic_vport_context_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  nic_vport_context: 'struct_mlx5_ifc_nic_vport_context_bits'
struct_mlx5_ifc_query_nic_vport_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('nic_vport_context', struct_mlx5_ifc_nic_vport_context_bits, 128)])
@c.record
class struct_mlx5_ifc_query_nic_vport_context_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  allowed_list_type: 'list[int]'
  reserved_at_68: 'list[int]'
struct_mlx5_ifc_query_nic_vport_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 5), 96), ('allowed_list_type', (ctypes.c_ubyte * 3), 101), ('reserved_at_68', (ctypes.c_ubyte * 24), 104)])
@c.record
class struct_mlx5_ifc_query_mkey_out_bits(c.Struct):
  SIZE = 2432
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  memory_key_mkey_entry: 'struct_mlx5_ifc_mkc_bits'
  reserved_at_280: 'list[int]'
  bsf0_klm0_pas_mtt0_1: 'list[list[int]]'
  bsf1_klm1_pas_mtt2_3: 'list[list[int]]'
struct_mlx5_ifc_query_mkey_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('memory_key_mkey_entry', struct_mlx5_ifc_mkc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 1536), 640), ('bsf0_klm0_pas_mtt0_1', ((ctypes.c_ubyte * 8) * 16), 2176), ('bsf1_klm1_pas_mtt2_3', ((ctypes.c_ubyte * 8) * 16), 2304)])
@c.record
class struct_mlx5_ifc_query_mkey_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  mkey_index: 'list[int]'
  pg_access: 'list[int]'
  reserved_at_61: 'list[int]'
struct_mlx5_ifc_query_mkey_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('mkey_index', (ctypes.c_ubyte * 24), 72), ('pg_access', (ctypes.c_ubyte * 1), 96), ('reserved_at_61', (ctypes.c_ubyte * 31), 97)])
@c.record
class struct_mlx5_ifc_query_mad_demux_out_bits(c.Struct):
  SIZE = 160
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  mad_dumux_parameters_block: 'list[int]'
struct_mlx5_ifc_query_mad_demux_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('mad_dumux_parameters_block', (ctypes.c_ubyte * 32), 128)])
@c.record
class struct_mlx5_ifc_query_mad_demux_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_mad_demux_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_query_l2_table_entry_out_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_e0: 'list[int]'
  vlan_valid: 'list[int]'
  vlan: 'list[int]'
  mac_address: 'struct_mlx5_ifc_mac_address_layout_bits'
  reserved_at_140: 'list[int]'
struct_mlx5_ifc_query_l2_table_entry_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 160), 64), ('reserved_at_e0', (ctypes.c_ubyte * 19), 224), ('vlan_valid', (ctypes.c_ubyte * 1), 243), ('vlan', (ctypes.c_ubyte * 12), 244), ('mac_address', struct_mlx5_ifc_mac_address_layout_bits, 256), ('reserved_at_140', (ctypes.c_ubyte * 192), 320)])
@c.record
class struct_mlx5_ifc_query_l2_table_entry_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_index: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_query_l2_table_entry_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 96), 64), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_index', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 320), 192)])
@c.record
class struct_mlx5_ifc_query_issi_out_bits(c.Struct):
  SIZE = 896
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  current_issi: 'list[int]'
  reserved_at_60: 'list[int]'
  reserved_at_100: 'list[list[int]]'
  supported_issi_dw0: 'list[int]'
struct_mlx5_ifc_query_issi_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('current_issi', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 160), 96), ('reserved_at_100', ((ctypes.c_ubyte * 8) * 76), 256), ('supported_issi_dw0', (ctypes.c_ubyte * 32), 864)])
@c.record
class struct_mlx5_ifc_query_issi_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_issi_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_driver_version_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_0: 'list[int]'
  syndrome: 'list[int]'
  reserved_1: 'list[int]'
struct_mlx5_ifc_set_driver_version_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_0', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_1', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_driver_version_in_bits(c.Struct):
  SIZE = 640
  opcode: 'list[int]'
  reserved_0: 'list[int]'
  reserved_1: 'list[int]'
  op_mod: 'list[int]'
  reserved_2: 'list[int]'
  driver_version: 'list[list[int]]'
struct_mlx5_ifc_set_driver_version_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_0', (ctypes.c_ubyte * 16), 16), ('reserved_1', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_2', (ctypes.c_ubyte * 64), 64), ('driver_version', ((ctypes.c_ubyte * 8) * 64), 128)])
@c.record
class struct_mlx5_ifc_query_hca_vport_pkey_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  pkey: 'list[struct_mlx5_ifc_pkey_bits]'
struct_mlx5_ifc_query_hca_vport_pkey_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('pkey', (struct_mlx5_ifc_pkey_bits * 0), 128)])
@c.record
class struct_mlx5_ifc_query_hca_vport_pkey_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  port_num: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  pkey_index: 'list[int]'
struct_mlx5_ifc_query_hca_vport_pkey_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 11), 65), ('port_num', (ctypes.c_ubyte * 4), 76), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('pkey_index', (ctypes.c_ubyte * 16), 112)])
_anonenum94: dict[int, str] = {(MLX5_HCA_VPORT_SEL_PORT_GUID:=1): 'MLX5_HCA_VPORT_SEL_PORT_GUID', (MLX5_HCA_VPORT_SEL_NODE_GUID:=2): 'MLX5_HCA_VPORT_SEL_NODE_GUID', (MLX5_HCA_VPORT_SEL_STATE_POLICY:=4): 'MLX5_HCA_VPORT_SEL_STATE_POLICY'}
@c.record
class struct_mlx5_ifc_query_hca_vport_gid_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  gids_num: 'list[int]'
  reserved_at_70: 'list[int]'
  gid: 'list[struct_mlx5_ifc_array128_auto_bits]'
struct_mlx5_ifc_query_hca_vport_gid_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('gids_num', (ctypes.c_ubyte * 16), 96), ('reserved_at_70', (ctypes.c_ubyte * 16), 112), ('gid', (struct_mlx5_ifc_array128_auto_bits * 0), 128)])
@c.record
class struct_mlx5_ifc_query_hca_vport_gid_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  port_num: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  gid_index: 'list[int]'
struct_mlx5_ifc_query_hca_vport_gid_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 11), 65), ('port_num', (ctypes.c_ubyte * 4), 76), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('gid_index', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_query_hca_vport_context_out_bits(c.Struct):
  SIZE = 4224
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  hca_vport_context: 'struct_mlx5_ifc_hca_vport_context_bits'
struct_mlx5_ifc_query_hca_vport_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('hca_vport_context', struct_mlx5_ifc_hca_vport_context_bits, 128)])
@c.record
class struct_mlx5_ifc_query_hca_vport_context_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  port_num: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_hca_vport_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 11), 65), ('port_num', (ctypes.c_ubyte * 4), 76), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_hca_cap_out_bits(c.Struct):
  SIZE = 32896
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  capability: 'union_mlx5_ifc_hca_cap_union_bits'
struct_mlx5_ifc_query_hca_cap_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('capability', union_mlx5_ifc_hca_cap_union_bits, 128)])
@c.record
class struct_mlx5_ifc_query_hca_cap_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_function: 'list[int]'
  ec_vf_function: 'list[int]'
  reserved_at_42: 'list[int]'
  function_id_type: 'list[int]'
  reserved_at_44: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_hca_cap_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_function', (ctypes.c_ubyte * 1), 64), ('ec_vf_function', (ctypes.c_ubyte * 1), 65), ('reserved_at_42', (ctypes.c_ubyte * 1), 66), ('function_id_type', (ctypes.c_ubyte * 1), 67), ('reserved_at_44', (ctypes.c_ubyte * 12), 68), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_other_hca_cap_bits(c.Struct):
  SIZE = 640
  roce: 'list[int]'
  reserved_at_1: 'list[int]'
struct_mlx5_ifc_other_hca_cap_bits.register_fields([('roce', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 639), 1)])
@c.record
class struct_mlx5_ifc_query_other_hca_cap_out_bits(c.Struct):
  SIZE = 768
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  other_capability: 'struct_mlx5_ifc_other_hca_cap_bits'
struct_mlx5_ifc_query_other_hca_cap_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('other_capability', struct_mlx5_ifc_other_hca_cap_bits, 128)])
@c.record
class struct_mlx5_ifc_query_other_hca_cap_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_other_hca_cap_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_modify_other_hca_cap_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_other_hca_cap_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_other_hca_cap_in_bits(c.Struct):
  SIZE = 768
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  function_id: 'list[int]'
  field_select: 'list[int]'
  other_capability: 'struct_mlx5_ifc_other_hca_cap_bits'
struct_mlx5_ifc_modify_other_hca_cap_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('function_id', (ctypes.c_ubyte * 16), 80), ('field_select', (ctypes.c_ubyte * 32), 96), ('other_capability', struct_mlx5_ifc_other_hca_cap_bits, 128)])
@c.record
class struct_mlx5_ifc_sw_owner_icm_root_params_bits(c.Struct):
  SIZE = 128
  sw_owner_icm_root_1: 'list[int]'
  sw_owner_icm_root_0: 'list[int]'
struct_mlx5_ifc_sw_owner_icm_root_params_bits.register_fields([('sw_owner_icm_root_1', (ctypes.c_ubyte * 64), 0), ('sw_owner_icm_root_0', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_rtc_params_bits(c.Struct):
  SIZE = 128
  rtc_id_0: 'list[int]'
  rtc_id_1: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_rtc_params_bits.register_fields([('rtc_id_0', (ctypes.c_ubyte * 32), 0), ('rtc_id_1', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_flow_table_context_bits(c.Struct):
  SIZE = 320
  reformat_en: 'list[int]'
  decap_en: 'list[int]'
  sw_owner: 'list[int]'
  termination_table: 'list[int]'
  table_miss_action: 'list[int]'
  level: 'list[int]'
  rtc_valid: 'list[int]'
  reserved_at_11: 'list[int]'
  log_size: 'list[int]'
  reserved_at_20: 'list[int]'
  table_miss_id: 'list[int]'
  reserved_at_40: 'list[int]'
  lag_master_next_table_id: 'list[int]'
  reserved_at_60: 'list[int]'
  sws: 'struct_mlx5_ifc_sw_owner_icm_root_params_bits'
  hws: 'struct_mlx5_ifc_rtc_params_bits'
struct_mlx5_ifc_flow_table_context_bits.register_fields([('reformat_en', (ctypes.c_ubyte * 1), 0), ('decap_en', (ctypes.c_ubyte * 1), 1), ('sw_owner', (ctypes.c_ubyte * 1), 2), ('termination_table', (ctypes.c_ubyte * 1), 3), ('table_miss_action', (ctypes.c_ubyte * 4), 4), ('level', (ctypes.c_ubyte * 8), 8), ('rtc_valid', (ctypes.c_ubyte * 1), 16), ('reserved_at_11', (ctypes.c_ubyte * 7), 17), ('log_size', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('table_miss_id', (ctypes.c_ubyte * 24), 40), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('lag_master_next_table_id', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 96), 96), ('sws', struct_mlx5_ifc_sw_owner_icm_root_params_bits, 192), ('hws', struct_mlx5_ifc_rtc_params_bits, 192)])
@c.record
class struct_mlx5_ifc_query_flow_table_out_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  flow_table_context: 'struct_mlx5_ifc_flow_table_context_bits'
struct_mlx5_ifc_query_flow_table_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 128), 64), ('flow_table_context', struct_mlx5_ifc_flow_table_context_bits, 192)])
@c.record
class struct_mlx5_ifc_query_flow_table_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_query_flow_table_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 320), 192)])
@c.record
class struct_mlx5_ifc_query_fte_out_bits(c.Struct):
  SIZE = 6656
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  flow_context: 'struct_mlx5_ifc_flow_context_bits'
struct_mlx5_ifc_query_fte_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 448), 64), ('flow_context', struct_mlx5_ifc_flow_context_bits, 512)])
@c.record
class struct_mlx5_ifc_query_fte_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  reserved_at_c0: 'list[int]'
  flow_index: 'list[int]'
  reserved_at_120: 'list[int]'
struct_mlx5_ifc_query_fte_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('flow_index', (ctypes.c_ubyte * 32), 256), ('reserved_at_120', (ctypes.c_ubyte * 224), 288)])
@c.record
class struct_mlx5_ifc_match_definer_format_0_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  metadata_reg_c_0: 'list[int]'
  metadata_reg_c_1: 'list[int]'
  outer_dmac_47_16: 'list[int]'
  outer_dmac_15_0: 'list[int]'
  outer_ethertype: 'list[int]'
  reserved_at_180: 'list[int]'
  sx_sniffer: 'list[int]'
  functional_lb: 'list[int]'
  outer_ip_frag: 'list[int]'
  outer_qp_type: 'list[int]'
  outer_encap_type: 'list[int]'
  port_number: 'list[int]'
  outer_l3_type: 'list[int]'
  outer_l4_type: 'list[int]'
  outer_first_vlan_type: 'list[int]'
  outer_first_vlan_prio: 'list[int]'
  outer_first_vlan_cfi: 'list[int]'
  outer_first_vlan_vid: 'list[int]'
  outer_l4_type_ext: 'list[int]'
  reserved_at_1a4: 'list[int]'
  outer_ipsec_layer: 'list[int]'
  outer_l2_type: 'list[int]'
  force_lb: 'list[int]'
  outer_l2_ok: 'list[int]'
  outer_l3_ok: 'list[int]'
  outer_l4_ok: 'list[int]'
  outer_second_vlan_type: 'list[int]'
  outer_second_vlan_prio: 'list[int]'
  outer_second_vlan_cfi: 'list[int]'
  outer_second_vlan_vid: 'list[int]'
  outer_smac_47_16: 'list[int]'
  outer_smac_15_0: 'list[int]'
  inner_ipv4_checksum_ok: 'list[int]'
  inner_l4_checksum_ok: 'list[int]'
  outer_ipv4_checksum_ok: 'list[int]'
  outer_l4_checksum_ok: 'list[int]'
  inner_l3_ok: 'list[int]'
  inner_l4_ok: 'list[int]'
  outer_l3_ok_duplicate: 'list[int]'
  outer_l4_ok_duplicate: 'list[int]'
  outer_tcp_cwr: 'list[int]'
  outer_tcp_ece: 'list[int]'
  outer_tcp_urg: 'list[int]'
  outer_tcp_ack: 'list[int]'
  outer_tcp_psh: 'list[int]'
  outer_tcp_rst: 'list[int]'
  outer_tcp_syn: 'list[int]'
  outer_tcp_fin: 'list[int]'
struct_mlx5_ifc_match_definer_format_0_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 256), 0), ('metadata_reg_c_0', (ctypes.c_ubyte * 32), 256), ('metadata_reg_c_1', (ctypes.c_ubyte * 32), 288), ('outer_dmac_47_16', (ctypes.c_ubyte * 32), 320), ('outer_dmac_15_0', (ctypes.c_ubyte * 16), 352), ('outer_ethertype', (ctypes.c_ubyte * 16), 368), ('reserved_at_180', (ctypes.c_ubyte * 1), 384), ('sx_sniffer', (ctypes.c_ubyte * 1), 385), ('functional_lb', (ctypes.c_ubyte * 1), 386), ('outer_ip_frag', (ctypes.c_ubyte * 1), 387), ('outer_qp_type', (ctypes.c_ubyte * 2), 388), ('outer_encap_type', (ctypes.c_ubyte * 2), 390), ('port_number', (ctypes.c_ubyte * 2), 392), ('outer_l3_type', (ctypes.c_ubyte * 2), 394), ('outer_l4_type', (ctypes.c_ubyte * 2), 396), ('outer_first_vlan_type', (ctypes.c_ubyte * 2), 398), ('outer_first_vlan_prio', (ctypes.c_ubyte * 3), 400), ('outer_first_vlan_cfi', (ctypes.c_ubyte * 1), 403), ('outer_first_vlan_vid', (ctypes.c_ubyte * 12), 404), ('outer_l4_type_ext', (ctypes.c_ubyte * 4), 416), ('reserved_at_1a4', (ctypes.c_ubyte * 2), 420), ('outer_ipsec_layer', (ctypes.c_ubyte * 2), 422), ('outer_l2_type', (ctypes.c_ubyte * 2), 424), ('force_lb', (ctypes.c_ubyte * 1), 426), ('outer_l2_ok', (ctypes.c_ubyte * 1), 427), ('outer_l3_ok', (ctypes.c_ubyte * 1), 428), ('outer_l4_ok', (ctypes.c_ubyte * 1), 429), ('outer_second_vlan_type', (ctypes.c_ubyte * 2), 430), ('outer_second_vlan_prio', (ctypes.c_ubyte * 3), 432), ('outer_second_vlan_cfi', (ctypes.c_ubyte * 1), 435), ('outer_second_vlan_vid', (ctypes.c_ubyte * 12), 436), ('outer_smac_47_16', (ctypes.c_ubyte * 32), 448), ('outer_smac_15_0', (ctypes.c_ubyte * 16), 480), ('inner_ipv4_checksum_ok', (ctypes.c_ubyte * 1), 496), ('inner_l4_checksum_ok', (ctypes.c_ubyte * 1), 497), ('outer_ipv4_checksum_ok', (ctypes.c_ubyte * 1), 498), ('outer_l4_checksum_ok', (ctypes.c_ubyte * 1), 499), ('inner_l3_ok', (ctypes.c_ubyte * 1), 500), ('inner_l4_ok', (ctypes.c_ubyte * 1), 501), ('outer_l3_ok_duplicate', (ctypes.c_ubyte * 1), 502), ('outer_l4_ok_duplicate', (ctypes.c_ubyte * 1), 503), ('outer_tcp_cwr', (ctypes.c_ubyte * 1), 504), ('outer_tcp_ece', (ctypes.c_ubyte * 1), 505), ('outer_tcp_urg', (ctypes.c_ubyte * 1), 506), ('outer_tcp_ack', (ctypes.c_ubyte * 1), 507), ('outer_tcp_psh', (ctypes.c_ubyte * 1), 508), ('outer_tcp_rst', (ctypes.c_ubyte * 1), 509), ('outer_tcp_syn', (ctypes.c_ubyte * 1), 510), ('outer_tcp_fin', (ctypes.c_ubyte * 1), 511)])
@c.record
class struct_mlx5_ifc_match_definer_format_22_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  outer_ip_src_addr: 'list[int]'
  outer_ip_dest_addr: 'list[int]'
  outer_l4_sport: 'list[int]'
  outer_l4_dport: 'list[int]'
  reserved_at_160: 'list[int]'
  sx_sniffer: 'list[int]'
  functional_lb: 'list[int]'
  outer_ip_frag: 'list[int]'
  outer_qp_type: 'list[int]'
  outer_encap_type: 'list[int]'
  port_number: 'list[int]'
  outer_l3_type: 'list[int]'
  outer_l4_type: 'list[int]'
  outer_first_vlan_type: 'list[int]'
  outer_first_vlan_prio: 'list[int]'
  outer_first_vlan_cfi: 'list[int]'
  outer_first_vlan_vid: 'list[int]'
  metadata_reg_c_0: 'list[int]'
  outer_dmac_47_16: 'list[int]'
  outer_smac_47_16: 'list[int]'
  outer_smac_15_0: 'list[int]'
  outer_dmac_15_0: 'list[int]'
struct_mlx5_ifc_match_definer_format_22_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 256), 0), ('outer_ip_src_addr', (ctypes.c_ubyte * 32), 256), ('outer_ip_dest_addr', (ctypes.c_ubyte * 32), 288), ('outer_l4_sport', (ctypes.c_ubyte * 16), 320), ('outer_l4_dport', (ctypes.c_ubyte * 16), 336), ('reserved_at_160', (ctypes.c_ubyte * 1), 352), ('sx_sniffer', (ctypes.c_ubyte * 1), 353), ('functional_lb', (ctypes.c_ubyte * 1), 354), ('outer_ip_frag', (ctypes.c_ubyte * 1), 355), ('outer_qp_type', (ctypes.c_ubyte * 2), 356), ('outer_encap_type', (ctypes.c_ubyte * 2), 358), ('port_number', (ctypes.c_ubyte * 2), 360), ('outer_l3_type', (ctypes.c_ubyte * 2), 362), ('outer_l4_type', (ctypes.c_ubyte * 2), 364), ('outer_first_vlan_type', (ctypes.c_ubyte * 2), 366), ('outer_first_vlan_prio', (ctypes.c_ubyte * 3), 368), ('outer_first_vlan_cfi', (ctypes.c_ubyte * 1), 371), ('outer_first_vlan_vid', (ctypes.c_ubyte * 12), 372), ('metadata_reg_c_0', (ctypes.c_ubyte * 32), 384), ('outer_dmac_47_16', (ctypes.c_ubyte * 32), 416), ('outer_smac_47_16', (ctypes.c_ubyte * 32), 448), ('outer_smac_15_0', (ctypes.c_ubyte * 16), 480), ('outer_dmac_15_0', (ctypes.c_ubyte * 16), 496)])
@c.record
class struct_mlx5_ifc_match_definer_format_23_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  inner_ip_src_addr: 'list[int]'
  inner_ip_dest_addr: 'list[int]'
  inner_l4_sport: 'list[int]'
  inner_l4_dport: 'list[int]'
  reserved_at_160: 'list[int]'
  sx_sniffer: 'list[int]'
  functional_lb: 'list[int]'
  inner_ip_frag: 'list[int]'
  inner_qp_type: 'list[int]'
  inner_encap_type: 'list[int]'
  port_number: 'list[int]'
  inner_l3_type: 'list[int]'
  inner_l4_type: 'list[int]'
  inner_first_vlan_type: 'list[int]'
  inner_first_vlan_prio: 'list[int]'
  inner_first_vlan_cfi: 'list[int]'
  inner_first_vlan_vid: 'list[int]'
  tunnel_header_0: 'list[int]'
  inner_dmac_47_16: 'list[int]'
  inner_smac_47_16: 'list[int]'
  inner_smac_15_0: 'list[int]'
  inner_dmac_15_0: 'list[int]'
struct_mlx5_ifc_match_definer_format_23_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 256), 0), ('inner_ip_src_addr', (ctypes.c_ubyte * 32), 256), ('inner_ip_dest_addr', (ctypes.c_ubyte * 32), 288), ('inner_l4_sport', (ctypes.c_ubyte * 16), 320), ('inner_l4_dport', (ctypes.c_ubyte * 16), 336), ('reserved_at_160', (ctypes.c_ubyte * 1), 352), ('sx_sniffer', (ctypes.c_ubyte * 1), 353), ('functional_lb', (ctypes.c_ubyte * 1), 354), ('inner_ip_frag', (ctypes.c_ubyte * 1), 355), ('inner_qp_type', (ctypes.c_ubyte * 2), 356), ('inner_encap_type', (ctypes.c_ubyte * 2), 358), ('port_number', (ctypes.c_ubyte * 2), 360), ('inner_l3_type', (ctypes.c_ubyte * 2), 362), ('inner_l4_type', (ctypes.c_ubyte * 2), 364), ('inner_first_vlan_type', (ctypes.c_ubyte * 2), 366), ('inner_first_vlan_prio', (ctypes.c_ubyte * 3), 368), ('inner_first_vlan_cfi', (ctypes.c_ubyte * 1), 371), ('inner_first_vlan_vid', (ctypes.c_ubyte * 12), 372), ('tunnel_header_0', (ctypes.c_ubyte * 32), 384), ('inner_dmac_47_16', (ctypes.c_ubyte * 32), 416), ('inner_smac_47_16', (ctypes.c_ubyte * 32), 448), ('inner_smac_15_0', (ctypes.c_ubyte * 16), 480), ('inner_dmac_15_0', (ctypes.c_ubyte * 16), 496)])
@c.record
class struct_mlx5_ifc_match_definer_format_29_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  outer_ip_dest_addr: 'list[int]'
  outer_ip_src_addr: 'list[int]'
  outer_l4_sport: 'list[int]'
  outer_l4_dport: 'list[int]'
  reserved_at_1e0: 'list[int]'
struct_mlx5_ifc_match_definer_format_29_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 192), 0), ('outer_ip_dest_addr', (ctypes.c_ubyte * 128), 192), ('outer_ip_src_addr', (ctypes.c_ubyte * 128), 320), ('outer_l4_sport', (ctypes.c_ubyte * 16), 448), ('outer_l4_dport', (ctypes.c_ubyte * 16), 464), ('reserved_at_1e0', (ctypes.c_ubyte * 32), 480)])
@c.record
class struct_mlx5_ifc_match_definer_format_30_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  outer_ip_dest_addr: 'list[int]'
  outer_ip_src_addr: 'list[int]'
  outer_dmac_47_16: 'list[int]'
  outer_smac_47_16: 'list[int]'
  outer_smac_15_0: 'list[int]'
  outer_dmac_15_0: 'list[int]'
struct_mlx5_ifc_match_definer_format_30_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 160), 0), ('outer_ip_dest_addr', (ctypes.c_ubyte * 128), 160), ('outer_ip_src_addr', (ctypes.c_ubyte * 128), 288), ('outer_dmac_47_16', (ctypes.c_ubyte * 32), 416), ('outer_smac_47_16', (ctypes.c_ubyte * 32), 448), ('outer_smac_15_0', (ctypes.c_ubyte * 16), 480), ('outer_dmac_15_0', (ctypes.c_ubyte * 16), 496)])
@c.record
class struct_mlx5_ifc_match_definer_format_31_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  inner_ip_dest_addr: 'list[int]'
  inner_ip_src_addr: 'list[int]'
  inner_l4_sport: 'list[int]'
  inner_l4_dport: 'list[int]'
  reserved_at_1e0: 'list[int]'
struct_mlx5_ifc_match_definer_format_31_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 192), 0), ('inner_ip_dest_addr', (ctypes.c_ubyte * 128), 192), ('inner_ip_src_addr', (ctypes.c_ubyte * 128), 320), ('inner_l4_sport', (ctypes.c_ubyte * 16), 448), ('inner_l4_dport', (ctypes.c_ubyte * 16), 464), ('reserved_at_1e0', (ctypes.c_ubyte * 32), 480)])
@c.record
class struct_mlx5_ifc_match_definer_format_32_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  inner_ip_dest_addr: 'list[int]'
  inner_ip_src_addr: 'list[int]'
  inner_dmac_47_16: 'list[int]'
  inner_smac_47_16: 'list[int]'
  inner_smac_15_0: 'list[int]'
  inner_dmac_15_0: 'list[int]'
struct_mlx5_ifc_match_definer_format_32_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 160), 0), ('inner_ip_dest_addr', (ctypes.c_ubyte * 128), 160), ('inner_ip_src_addr', (ctypes.c_ubyte * 128), 288), ('inner_dmac_47_16', (ctypes.c_ubyte * 32), 416), ('inner_smac_47_16', (ctypes.c_ubyte * 32), 448), ('inner_smac_15_0', (ctypes.c_ubyte * 16), 480), ('inner_dmac_15_0', (ctypes.c_ubyte * 16), 496)])
_anonenum95: dict[int, str] = {(MLX5_IFC_DEFINER_FORMAT_ID_SELECT:=61): 'MLX5_IFC_DEFINER_FORMAT_ID_SELECT'}
@c.record
class struct_mlx5_ifc_match_definer_match_mask_bits(c.Struct):
  SIZE = 512
  reserved_at_1c0: 'list[list[int]]'
  match_dw_8: 'list[int]'
  match_dw_7: 'list[int]'
  match_dw_6: 'list[int]'
  match_dw_5: 'list[int]'
  match_dw_4: 'list[int]'
  match_dw_3: 'list[int]'
  match_dw_2: 'list[int]'
  match_dw_1: 'list[int]'
  match_dw_0: 'list[int]'
  match_byte_7: 'list[int]'
  match_byte_6: 'list[int]'
  match_byte_5: 'list[int]'
  match_byte_4: 'list[int]'
  match_byte_3: 'list[int]'
  match_byte_2: 'list[int]'
  match_byte_1: 'list[int]'
  match_byte_0: 'list[int]'
struct_mlx5_ifc_match_definer_match_mask_bits.register_fields([('reserved_at_1c0', ((ctypes.c_ubyte * 32) * 5), 0), ('match_dw_8', (ctypes.c_ubyte * 32), 160), ('match_dw_7', (ctypes.c_ubyte * 32), 192), ('match_dw_6', (ctypes.c_ubyte * 32), 224), ('match_dw_5', (ctypes.c_ubyte * 32), 256), ('match_dw_4', (ctypes.c_ubyte * 32), 288), ('match_dw_3', (ctypes.c_ubyte * 32), 320), ('match_dw_2', (ctypes.c_ubyte * 32), 352), ('match_dw_1', (ctypes.c_ubyte * 32), 384), ('match_dw_0', (ctypes.c_ubyte * 32), 416), ('match_byte_7', (ctypes.c_ubyte * 8), 448), ('match_byte_6', (ctypes.c_ubyte * 8), 456), ('match_byte_5', (ctypes.c_ubyte * 8), 464), ('match_byte_4', (ctypes.c_ubyte * 8), 472), ('match_byte_3', (ctypes.c_ubyte * 8), 480), ('match_byte_2', (ctypes.c_ubyte * 8), 488), ('match_byte_1', (ctypes.c_ubyte * 8), 496), ('match_byte_0', (ctypes.c_ubyte * 8), 504)])
@c.record
class struct_mlx5_ifc_match_definer_bits(c.Struct):
  SIZE = 1024
  modify_field_select: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_80: 'list[int]'
  format_id: 'list[int]'
  reserved_at_a0: 'list[int]'
  format_select_dw3: 'list[int]'
  format_select_dw2: 'list[int]'
  format_select_dw1: 'list[int]'
  format_select_dw0: 'list[int]'
  format_select_dw7: 'list[int]'
  format_select_dw6: 'list[int]'
  format_select_dw5: 'list[int]'
  format_select_dw4: 'list[int]'
  reserved_at_100: 'list[int]'
  format_select_dw8: 'list[int]'
  reserved_at_120: 'list[int]'
  format_select_byte3: 'list[int]'
  format_select_byte2: 'list[int]'
  format_select_byte1: 'list[int]'
  format_select_byte0: 'list[int]'
  format_select_byte7: 'list[int]'
  format_select_byte6: 'list[int]'
  format_select_byte5: 'list[int]'
  format_select_byte4: 'list[int]'
  reserved_at_180: 'list[int]'
  match_mask: 'list[list[int]]'
  match_mask_format: 'struct_mlx5_ifc_match_definer_match_mask_bits'
struct_mlx5_ifc_match_definer_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('reserved_at_80', (ctypes.c_ubyte * 16), 128), ('format_id', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 96), 160), ('format_select_dw3', (ctypes.c_ubyte * 8), 256), ('format_select_dw2', (ctypes.c_ubyte * 8), 264), ('format_select_dw1', (ctypes.c_ubyte * 8), 272), ('format_select_dw0', (ctypes.c_ubyte * 8), 280), ('format_select_dw7', (ctypes.c_ubyte * 8), 288), ('format_select_dw6', (ctypes.c_ubyte * 8), 296), ('format_select_dw5', (ctypes.c_ubyte * 8), 304), ('format_select_dw4', (ctypes.c_ubyte * 8), 312), ('reserved_at_100', (ctypes.c_ubyte * 24), 320), ('format_select_dw8', (ctypes.c_ubyte * 8), 344), ('reserved_at_120', (ctypes.c_ubyte * 32), 352), ('format_select_byte3', (ctypes.c_ubyte * 8), 384), ('format_select_byte2', (ctypes.c_ubyte * 8), 392), ('format_select_byte1', (ctypes.c_ubyte * 8), 400), ('format_select_byte0', (ctypes.c_ubyte * 8), 408), ('format_select_byte7', (ctypes.c_ubyte * 8), 416), ('format_select_byte6', (ctypes.c_ubyte * 8), 424), ('format_select_byte5', (ctypes.c_ubyte * 8), 432), ('format_select_byte4', (ctypes.c_ubyte * 8), 440), ('reserved_at_180', (ctypes.c_ubyte * 64), 448), ('match_mask', ((ctypes.c_ubyte * 32) * 16), 512), ('match_mask_format', struct_mlx5_ifc_match_definer_match_mask_bits, 512)])
@c.record
class struct_mlx5_ifc_general_obj_create_param_bits(c.Struct):
  SIZE = 32
  alias_object: 'list[int]'
  reserved_at_1: 'list[int]'
  log_obj_range: 'list[int]'
  reserved_at_8: 'list[int]'
struct_mlx5_ifc_general_obj_create_param_bits.register_fields([('alias_object', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 2), 1), ('log_obj_range', (ctypes.c_ubyte * 5), 3), ('reserved_at_8', (ctypes.c_ubyte * 24), 8)])
@c.record
class struct_mlx5_ifc_general_obj_query_param_bits(c.Struct):
  SIZE = 32
  alias_object: 'list[int]'
  obj_offset: 'list[int]'
struct_mlx5_ifc_general_obj_query_param_bits.register_fields([('alias_object', (ctypes.c_ubyte * 1), 0), ('obj_offset', (ctypes.c_ubyte * 31), 1)])
@c.record
class struct_mlx5_ifc_general_obj_in_cmd_hdr_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  vhca_tunnel_id: 'list[int]'
  obj_type: 'list[int]'
  obj_id: 'list[int]'
  op_param: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits_op_param'
@c.record
class struct_mlx5_ifc_general_obj_in_cmd_hdr_bits_op_param(c.Struct):
  SIZE = 32
  create: 'struct_mlx5_ifc_general_obj_create_param_bits'
  query: 'struct_mlx5_ifc_general_obj_query_param_bits'
struct_mlx5_ifc_general_obj_in_cmd_hdr_bits_op_param.register_fields([('create', struct_mlx5_ifc_general_obj_create_param_bits, 0), ('query', struct_mlx5_ifc_general_obj_query_param_bits, 0)])
struct_mlx5_ifc_general_obj_in_cmd_hdr_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('vhca_tunnel_id', (ctypes.c_ubyte * 16), 32), ('obj_type', (ctypes.c_ubyte * 16), 48), ('obj_id', (ctypes.c_ubyte * 32), 64), ('op_param', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits_op_param, 96)])
@c.record
class struct_mlx5_ifc_general_obj_out_cmd_hdr_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  obj_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_general_obj_out_cmd_hdr_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('obj_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_allow_other_vhca_access_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  object_type_to_be_accessed: 'list[int]'
  object_id_to_be_accessed: 'list[int]'
  reserved_at_c0: 'list[int]'
  access_key_raw: 'list[int]'
  access_key: 'list[list[int]]'
struct_mlx5_ifc_allow_other_vhca_access_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 80), 64), ('object_type_to_be_accessed', (ctypes.c_ubyte * 16), 144), ('object_id_to_be_accessed', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('access_key_raw', (ctypes.c_ubyte * 256), 256), ('access_key', ((ctypes.c_ubyte * 32) * 8), 256)])
@c.record
class struct_mlx5_ifc_allow_other_vhca_access_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_allow_other_vhca_access_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_header_arg_bits(c.Struct):
  SIZE = 160
  reserved_at_0: 'list[int]'
  reserved_at_80: 'list[int]'
  access_pd: 'list[int]'
struct_mlx5_ifc_modify_header_arg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 128), 0), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('access_pd', (ctypes.c_ubyte * 24), 136)])
@c.record
class struct_mlx5_ifc_create_modify_header_arg_in_bits(c.Struct):
  SIZE = 288
  hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  arg: 'struct_mlx5_ifc_modify_header_arg_bits'
struct_mlx5_ifc_create_modify_header_arg_in_bits.register_fields([('hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('arg', struct_mlx5_ifc_modify_header_arg_bits, 128)])
@c.record
class struct_mlx5_ifc_create_match_definer_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  obj_context: 'struct_mlx5_ifc_match_definer_bits'
struct_mlx5_ifc_create_match_definer_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('obj_context', struct_mlx5_ifc_match_definer_bits, 128)])
@c.record
class struct_mlx5_ifc_create_match_definer_out_bits(c.Struct):
  SIZE = 128
  general_obj_out_cmd_hdr: 'struct_mlx5_ifc_general_obj_out_cmd_hdr_bits'
struct_mlx5_ifc_create_match_definer_out_bits.register_fields([('general_obj_out_cmd_hdr', struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0)])
@c.record
class struct_mlx5_ifc_alias_context_bits(c.Struct):
  SIZE = 512
  vhca_id_to_be_accessed: 'list[int]'
  reserved_at_10: 'list[int]'
  status: 'list[int]'
  object_id_to_be_accessed: 'list[int]'
  reserved_at_40: 'list[int]'
  access_key_raw: 'list[int]'
  access_key: 'list[list[int]]'
  metadata: 'list[int]'
struct_mlx5_ifc_alias_context_bits.register_fields([('vhca_id_to_be_accessed', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 13), 16), ('status', (ctypes.c_ubyte * 3), 29), ('object_id_to_be_accessed', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('access_key_raw', (ctypes.c_ubyte * 256), 128), ('access_key', ((ctypes.c_ubyte * 32) * 8), 128), ('metadata', (ctypes.c_ubyte * 128), 384)])
@c.record
class struct_mlx5_ifc_create_alias_obj_in_bits(c.Struct):
  SIZE = 640
  hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  alias_ctx: 'struct_mlx5_ifc_alias_context_bits'
struct_mlx5_ifc_create_alias_obj_in_bits.register_fields([('hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('alias_ctx', struct_mlx5_ifc_alias_context_bits, 128)])
_anonenum96: dict[int, str] = {(MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_OUTER_HEADERS:=0): 'MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_OUTER_HEADERS', (MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS:=1): 'MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS', (MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_INNER_HEADERS:=2): 'MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_INNER_HEADERS', (MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_2:=3): 'MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_2', (MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_3:=4): 'MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_3', (MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_4:=5): 'MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_4', (MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_5:=6): 'MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_5'}
@c.record
class struct_mlx5_ifc_query_flow_group_out_bits(c.Struct):
  SIZE = 8192
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  start_flow_index: 'list[int]'
  reserved_at_100: 'list[int]'
  end_flow_index: 'list[int]'
  reserved_at_140: 'list[int]'
  reserved_at_1e0: 'list[int]'
  match_criteria_enable: 'list[int]'
  match_criteria: 'struct_mlx5_ifc_fte_match_param_bits'
  reserved_at_1200: 'list[int]'
struct_mlx5_ifc_query_flow_group_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 160), 64), ('start_flow_index', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 32), 256), ('end_flow_index', (ctypes.c_ubyte * 32), 288), ('reserved_at_140', (ctypes.c_ubyte * 160), 320), ('reserved_at_1e0', (ctypes.c_ubyte * 24), 480), ('match_criteria_enable', (ctypes.c_ubyte * 8), 504), ('match_criteria', struct_mlx5_ifc_fte_match_param_bits, 512), ('reserved_at_1200', (ctypes.c_ubyte * 3584), 4608)])
@c.record
class struct_mlx5_ifc_query_flow_group_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  group_id: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_query_flow_group_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('group_id', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 288), 224)])
@c.record
class struct_mlx5_ifc_query_flow_counter_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  flow_statistics: 'list[struct_mlx5_ifc_traffic_counter_bits]'
struct_mlx5_ifc_query_flow_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('flow_statistics', (struct_mlx5_ifc_traffic_counter_bits * 0), 128)])
@c.record
class struct_mlx5_ifc_query_flow_counter_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  clear: 'list[int]'
  reserved_at_c1: 'list[int]'
  num_of_counters: 'list[int]'
  flow_counter_id: 'list[int]'
struct_mlx5_ifc_query_flow_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 128), 64), ('clear', (ctypes.c_ubyte * 1), 192), ('reserved_at_c1', (ctypes.c_ubyte * 15), 193), ('num_of_counters', (ctypes.c_ubyte * 16), 208), ('flow_counter_id', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_query_esw_vport_context_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  esw_vport_context: 'struct_mlx5_ifc_esw_vport_context_bits'
struct_mlx5_ifc_query_esw_vport_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('esw_vport_context', struct_mlx5_ifc_esw_vport_context_bits, 128)])
@c.record
class struct_mlx5_ifc_query_esw_vport_context_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_esw_vport_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_esw_vport_out_bits(c.Struct):
  SIZE = 96
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_esw_vport_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64)])
@c.record
class struct_mlx5_ifc_destroy_esw_vport_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  vport_num: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_esw_vport_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('vport_num', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_modify_esw_vport_context_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_esw_vport_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_esw_vport_context_fields_select_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  fdb_to_vport_reg_c_id: 'list[int]'
  vport_cvlan_insert: 'list[int]'
  vport_svlan_insert: 'list[int]'
  vport_cvlan_strip: 'list[int]'
  vport_svlan_strip: 'list[int]'
struct_mlx5_ifc_esw_vport_context_fields_select_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 27), 0), ('fdb_to_vport_reg_c_id', (ctypes.c_ubyte * 1), 27), ('vport_cvlan_insert', (ctypes.c_ubyte * 1), 28), ('vport_svlan_insert', (ctypes.c_ubyte * 1), 29), ('vport_cvlan_strip', (ctypes.c_ubyte * 1), 30), ('vport_svlan_strip', (ctypes.c_ubyte * 1), 31)])
@c.record
class struct_mlx5_ifc_modify_esw_vport_context_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  field_select: 'struct_mlx5_ifc_esw_vport_context_fields_select_bits'
  esw_vport_context: 'struct_mlx5_ifc_esw_vport_context_bits'
struct_mlx5_ifc_modify_esw_vport_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('field_select', struct_mlx5_ifc_esw_vport_context_fields_select_bits, 96), ('esw_vport_context', struct_mlx5_ifc_esw_vport_context_bits, 128)])
@c.record
class struct_mlx5_ifc_query_eq_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  eq_context_entry: 'struct_mlx5_ifc_eqc_bits'
  reserved_at_280: 'list[int]'
  event_bitmask: 'list[int]'
  reserved_at_300: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_query_eq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('eq_context_entry', struct_mlx5_ifc_eqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 64), 640), ('event_bitmask', (ctypes.c_ubyte * 64), 704), ('reserved_at_300', (ctypes.c_ubyte * 1408), 768), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_query_eq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  eq_number: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_eq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('eq_number', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_packet_reformat_context_in_bits(c.Struct):
  SIZE = 64
  reformat_type: 'list[int]'
  reserved_at_8: 'list[int]'
  reformat_param_0: 'list[int]'
  reserved_at_10: 'list[int]'
  reformat_data_size: 'list[int]'
  reformat_param_1: 'list[int]'
  reserved_at_28: 'list[int]'
  reformat_data: 'list[list[int]]'
  more_reformat_data: 'list[list[int]]'
struct_mlx5_ifc_packet_reformat_context_in_bits.register_fields([('reformat_type', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 4), 8), ('reformat_param_0', (ctypes.c_ubyte * 4), 12), ('reserved_at_10', (ctypes.c_ubyte * 6), 16), ('reformat_data_size', (ctypes.c_ubyte * 10), 22), ('reformat_param_1', (ctypes.c_ubyte * 8), 32), ('reserved_at_28', (ctypes.c_ubyte * 8), 40), ('reformat_data', ((ctypes.c_ubyte * 8) * 2), 48), ('more_reformat_data', ((ctypes.c_ubyte * 8) * 0), 64)])
@c.record
class struct_mlx5_ifc_query_packet_reformat_context_out_bits(c.Struct):
  SIZE = 224
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  packet_reformat_context: 'list[struct_mlx5_ifc_packet_reformat_context_in_bits]'
struct_mlx5_ifc_query_packet_reformat_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 160), 64), ('packet_reformat_context', (struct_mlx5_ifc_packet_reformat_context_in_bits * 0), 224)])
@c.record
class struct_mlx5_ifc_query_packet_reformat_context_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  packet_reformat_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_packet_reformat_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('packet_reformat_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 160), 96)])
@c.record
class struct_mlx5_ifc_alloc_packet_reformat_context_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  packet_reformat_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_packet_reformat_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('packet_reformat_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
_anonenum97: dict[int, str] = {(MLX5_REFORMAT_CONTEXT_ANCHOR_MAC_START:=1): 'MLX5_REFORMAT_CONTEXT_ANCHOR_MAC_START', (MLX5_REFORMAT_CONTEXT_ANCHOR_VLAN_START:=2): 'MLX5_REFORMAT_CONTEXT_ANCHOR_VLAN_START', (MLX5_REFORMAT_CONTEXT_ANCHOR_IP_START:=7): 'MLX5_REFORMAT_CONTEXT_ANCHOR_IP_START', (MLX5_REFORMAT_CONTEXT_ANCHOR_TCP_UDP_START:=9): 'MLX5_REFORMAT_CONTEXT_ANCHOR_TCP_UDP_START'}
enum_mlx5_reformat_ctx_type: dict[int, str] = {(MLX5_REFORMAT_TYPE_L2_TO_VXLAN:=0): 'MLX5_REFORMAT_TYPE_L2_TO_VXLAN', (MLX5_REFORMAT_TYPE_L2_TO_NVGRE:=1): 'MLX5_REFORMAT_TYPE_L2_TO_NVGRE', (MLX5_REFORMAT_TYPE_L2_TO_L2_TUNNEL:=2): 'MLX5_REFORMAT_TYPE_L2_TO_L2_TUNNEL', (MLX5_REFORMAT_TYPE_L3_TUNNEL_TO_L2:=3): 'MLX5_REFORMAT_TYPE_L3_TUNNEL_TO_L2', (MLX5_REFORMAT_TYPE_L2_TO_L3_TUNNEL:=4): 'MLX5_REFORMAT_TYPE_L2_TO_L3_TUNNEL', (MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_IPV4:=5): 'MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_IPV4', (MLX5_REFORMAT_TYPE_L2_TO_L3_ESP_TUNNEL:=6): 'MLX5_REFORMAT_TYPE_L2_TO_L3_ESP_TUNNEL', (MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_UDPV4:=7): 'MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_UDPV4', (MLX5_REFORMAT_TYPE_DEL_ESP_TRANSPORT:=8): 'MLX5_REFORMAT_TYPE_DEL_ESP_TRANSPORT', (MLX5_REFORMAT_TYPE_L3_ESP_TUNNEL_TO_L2:=9): 'MLX5_REFORMAT_TYPE_L3_ESP_TUNNEL_TO_L2', (MLX5_REFORMAT_TYPE_DEL_ESP_TRANSPORT_OVER_UDP:=10): 'MLX5_REFORMAT_TYPE_DEL_ESP_TRANSPORT_OVER_UDP', (MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_IPV6:=11): 'MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_IPV6', (MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_UDPV6:=12): 'MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_UDPV6', (MLX5_REFORMAT_TYPE_ADD_PSP_TUNNEL:=13): 'MLX5_REFORMAT_TYPE_ADD_PSP_TUNNEL', (MLX5_REFORMAT_TYPE_DEL_PSP_TUNNEL:=14): 'MLX5_REFORMAT_TYPE_DEL_PSP_TUNNEL', (MLX5_REFORMAT_TYPE_INSERT_HDR:=15): 'MLX5_REFORMAT_TYPE_INSERT_HDR', (MLX5_REFORMAT_TYPE_REMOVE_HDR:=16): 'MLX5_REFORMAT_TYPE_REMOVE_HDR', (MLX5_REFORMAT_TYPE_ADD_MACSEC:=17): 'MLX5_REFORMAT_TYPE_ADD_MACSEC', (MLX5_REFORMAT_TYPE_DEL_MACSEC:=18): 'MLX5_REFORMAT_TYPE_DEL_MACSEC'}
@c.record
class struct_mlx5_ifc_alloc_packet_reformat_context_in_bits(c.Struct):
  SIZE = 288
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  packet_reformat_context: 'struct_mlx5_ifc_packet_reformat_context_in_bits'
struct_mlx5_ifc_alloc_packet_reformat_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 160), 64), ('packet_reformat_context', struct_mlx5_ifc_packet_reformat_context_in_bits, 224)])
@c.record
class struct_mlx5_ifc_dealloc_packet_reformat_context_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_packet_reformat_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_packet_reformat_context_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_20: 'list[int]'
  op_mod: 'list[int]'
  packet_reformat_id: 'list[int]'
  reserved_60: 'list[int]'
struct_mlx5_ifc_dealloc_packet_reformat_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('packet_reformat_id', (ctypes.c_ubyte * 32), 64), ('reserved_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_set_action_in_bits(c.Struct):
  SIZE = 64
  action_type: 'list[int]'
  field: 'list[int]'
  reserved_at_10: 'list[int]'
  offset: 'list[int]'
  reserved_at_18: 'list[int]'
  length: 'list[int]'
  data: 'list[int]'
struct_mlx5_ifc_set_action_in_bits.register_fields([('action_type', (ctypes.c_ubyte * 4), 0), ('field', (ctypes.c_ubyte * 12), 4), ('reserved_at_10', (ctypes.c_ubyte * 3), 16), ('offset', (ctypes.c_ubyte * 5), 19), ('reserved_at_18', (ctypes.c_ubyte * 3), 24), ('length', (ctypes.c_ubyte * 5), 27), ('data', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_add_action_in_bits(c.Struct):
  SIZE = 64
  action_type: 'list[int]'
  field: 'list[int]'
  reserved_at_10: 'list[int]'
  data: 'list[int]'
struct_mlx5_ifc_add_action_in_bits.register_fields([('action_type', (ctypes.c_ubyte * 4), 0), ('field', (ctypes.c_ubyte * 12), 4), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('data', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_copy_action_in_bits(c.Struct):
  SIZE = 64
  action_type: 'list[int]'
  src_field: 'list[int]'
  reserved_at_10: 'list[int]'
  src_offset: 'list[int]'
  reserved_at_18: 'list[int]'
  length: 'list[int]'
  reserved_at_20: 'list[int]'
  dst_field: 'list[int]'
  reserved_at_30: 'list[int]'
  dst_offset: 'list[int]'
  reserved_at_38: 'list[int]'
struct_mlx5_ifc_copy_action_in_bits.register_fields([('action_type', (ctypes.c_ubyte * 4), 0), ('src_field', (ctypes.c_ubyte * 12), 4), ('reserved_at_10', (ctypes.c_ubyte * 3), 16), ('src_offset', (ctypes.c_ubyte * 5), 19), ('reserved_at_18', (ctypes.c_ubyte * 3), 24), ('length', (ctypes.c_ubyte * 5), 27), ('reserved_at_20', (ctypes.c_ubyte * 4), 32), ('dst_field', (ctypes.c_ubyte * 12), 36), ('reserved_at_30', (ctypes.c_ubyte * 3), 48), ('dst_offset', (ctypes.c_ubyte * 5), 51), ('reserved_at_38', (ctypes.c_ubyte * 8), 56)])
@c.record
class union_mlx5_ifc_set_add_copy_action_in_auto_bits(c.Struct):
  SIZE = 64
  set_action_in: 'struct_mlx5_ifc_set_action_in_bits'
  add_action_in: 'struct_mlx5_ifc_add_action_in_bits'
  copy_action_in: 'struct_mlx5_ifc_copy_action_in_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_set_add_copy_action_in_auto_bits.register_fields([('set_action_in', struct_mlx5_ifc_set_action_in_bits, 0), ('add_action_in', struct_mlx5_ifc_add_action_in_bits, 0), ('copy_action_in', struct_mlx5_ifc_copy_action_in_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 64), 0)])
_anonenum98: dict[int, str] = {(MLX5_ACTION_TYPE_SET:=1): 'MLX5_ACTION_TYPE_SET', (MLX5_ACTION_TYPE_ADD:=2): 'MLX5_ACTION_TYPE_ADD', (MLX5_ACTION_TYPE_COPY:=3): 'MLX5_ACTION_TYPE_COPY'}
_anonenum99: dict[int, str] = {(MLX5_ACTION_IN_FIELD_OUT_SMAC_47_16:=1): 'MLX5_ACTION_IN_FIELD_OUT_SMAC_47_16', (MLX5_ACTION_IN_FIELD_OUT_SMAC_15_0:=2): 'MLX5_ACTION_IN_FIELD_OUT_SMAC_15_0', (MLX5_ACTION_IN_FIELD_OUT_ETHERTYPE:=3): 'MLX5_ACTION_IN_FIELD_OUT_ETHERTYPE', (MLX5_ACTION_IN_FIELD_OUT_DMAC_47_16:=4): 'MLX5_ACTION_IN_FIELD_OUT_DMAC_47_16', (MLX5_ACTION_IN_FIELD_OUT_DMAC_15_0:=5): 'MLX5_ACTION_IN_FIELD_OUT_DMAC_15_0', (MLX5_ACTION_IN_FIELD_OUT_IP_DSCP:=6): 'MLX5_ACTION_IN_FIELD_OUT_IP_DSCP', (MLX5_ACTION_IN_FIELD_OUT_TCP_FLAGS:=7): 'MLX5_ACTION_IN_FIELD_OUT_TCP_FLAGS', (MLX5_ACTION_IN_FIELD_OUT_TCP_SPORT:=8): 'MLX5_ACTION_IN_FIELD_OUT_TCP_SPORT', (MLX5_ACTION_IN_FIELD_OUT_TCP_DPORT:=9): 'MLX5_ACTION_IN_FIELD_OUT_TCP_DPORT', (MLX5_ACTION_IN_FIELD_OUT_IP_TTL:=10): 'MLX5_ACTION_IN_FIELD_OUT_IP_TTL', (MLX5_ACTION_IN_FIELD_OUT_UDP_SPORT:=11): 'MLX5_ACTION_IN_FIELD_OUT_UDP_SPORT', (MLX5_ACTION_IN_FIELD_OUT_UDP_DPORT:=12): 'MLX5_ACTION_IN_FIELD_OUT_UDP_DPORT', (MLX5_ACTION_IN_FIELD_OUT_SIPV6_127_96:=13): 'MLX5_ACTION_IN_FIELD_OUT_SIPV6_127_96', (MLX5_ACTION_IN_FIELD_OUT_SIPV6_95_64:=14): 'MLX5_ACTION_IN_FIELD_OUT_SIPV6_95_64', (MLX5_ACTION_IN_FIELD_OUT_SIPV6_63_32:=15): 'MLX5_ACTION_IN_FIELD_OUT_SIPV6_63_32', (MLX5_ACTION_IN_FIELD_OUT_SIPV6_31_0:=16): 'MLX5_ACTION_IN_FIELD_OUT_SIPV6_31_0', (MLX5_ACTION_IN_FIELD_OUT_DIPV6_127_96:=17): 'MLX5_ACTION_IN_FIELD_OUT_DIPV6_127_96', (MLX5_ACTION_IN_FIELD_OUT_DIPV6_95_64:=18): 'MLX5_ACTION_IN_FIELD_OUT_DIPV6_95_64', (MLX5_ACTION_IN_FIELD_OUT_DIPV6_63_32:=19): 'MLX5_ACTION_IN_FIELD_OUT_DIPV6_63_32', (MLX5_ACTION_IN_FIELD_OUT_DIPV6_31_0:=20): 'MLX5_ACTION_IN_FIELD_OUT_DIPV6_31_0', (MLX5_ACTION_IN_FIELD_OUT_SIPV4:=21): 'MLX5_ACTION_IN_FIELD_OUT_SIPV4', (MLX5_ACTION_IN_FIELD_OUT_DIPV4:=22): 'MLX5_ACTION_IN_FIELD_OUT_DIPV4', (MLX5_ACTION_IN_FIELD_OUT_FIRST_VID:=23): 'MLX5_ACTION_IN_FIELD_OUT_FIRST_VID', (MLX5_ACTION_IN_FIELD_OUT_IPV6_HOPLIMIT:=71): 'MLX5_ACTION_IN_FIELD_OUT_IPV6_HOPLIMIT', (MLX5_ACTION_IN_FIELD_METADATA_REG_A:=73): 'MLX5_ACTION_IN_FIELD_METADATA_REG_A', (MLX5_ACTION_IN_FIELD_METADATA_REG_B:=80): 'MLX5_ACTION_IN_FIELD_METADATA_REG_B', (MLX5_ACTION_IN_FIELD_METADATA_REG_C_0:=81): 'MLX5_ACTION_IN_FIELD_METADATA_REG_C_0', (MLX5_ACTION_IN_FIELD_METADATA_REG_C_1:=82): 'MLX5_ACTION_IN_FIELD_METADATA_REG_C_1', (MLX5_ACTION_IN_FIELD_METADATA_REG_C_2:=83): 'MLX5_ACTION_IN_FIELD_METADATA_REG_C_2', (MLX5_ACTION_IN_FIELD_METADATA_REG_C_3:=84): 'MLX5_ACTION_IN_FIELD_METADATA_REG_C_3', (MLX5_ACTION_IN_FIELD_METADATA_REG_C_4:=85): 'MLX5_ACTION_IN_FIELD_METADATA_REG_C_4', (MLX5_ACTION_IN_FIELD_METADATA_REG_C_5:=86): 'MLX5_ACTION_IN_FIELD_METADATA_REG_C_5', (MLX5_ACTION_IN_FIELD_METADATA_REG_C_6:=87): 'MLX5_ACTION_IN_FIELD_METADATA_REG_C_6', (MLX5_ACTION_IN_FIELD_METADATA_REG_C_7:=88): 'MLX5_ACTION_IN_FIELD_METADATA_REG_C_7', (MLX5_ACTION_IN_FIELD_OUT_TCP_SEQ_NUM:=89): 'MLX5_ACTION_IN_FIELD_OUT_TCP_SEQ_NUM', (MLX5_ACTION_IN_FIELD_OUT_TCP_ACK_NUM:=91): 'MLX5_ACTION_IN_FIELD_OUT_TCP_ACK_NUM', (MLX5_ACTION_IN_FIELD_IPSEC_SYNDROME:=93): 'MLX5_ACTION_IN_FIELD_IPSEC_SYNDROME', (MLX5_ACTION_IN_FIELD_OUT_EMD_47_32:=111): 'MLX5_ACTION_IN_FIELD_OUT_EMD_47_32', (MLX5_ACTION_IN_FIELD_OUT_EMD_31_0:=112): 'MLX5_ACTION_IN_FIELD_OUT_EMD_31_0', (MLX5_ACTION_IN_FIELD_PSP_SYNDROME:=113): 'MLX5_ACTION_IN_FIELD_PSP_SYNDROME'}
@c.record
class struct_mlx5_ifc_alloc_modify_header_context_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  modify_header_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_modify_header_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('modify_header_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_modify_header_context_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  table_type: 'list[int]'
  reserved_at_68: 'list[int]'
  num_of_actions: 'list[int]'
  actions: 'list[union_mlx5_ifc_set_add_copy_action_in_auto_bits]'
struct_mlx5_ifc_alloc_modify_header_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('table_type', (ctypes.c_ubyte * 8), 96), ('reserved_at_68', (ctypes.c_ubyte * 16), 104), ('num_of_actions', (ctypes.c_ubyte * 8), 120), ('actions', (union_mlx5_ifc_set_add_copy_action_in_auto_bits * 0), 128)])
@c.record
class struct_mlx5_ifc_dealloc_modify_header_context_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_modify_header_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_modify_header_context_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  modify_header_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_dealloc_modify_header_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('modify_header_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_modify_header_context_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  modify_header_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_modify_header_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('modify_header_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 160), 96)])
@c.record
class struct_mlx5_ifc_query_dct_out_bits(c.Struct):
  SIZE = 1024
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  dct_context_entry: 'struct_mlx5_ifc_dctc_bits'
  reserved_at_280: 'list[int]'
struct_mlx5_ifc_query_dct_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('dct_context_entry', struct_mlx5_ifc_dctc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 384), 640)])
@c.record
class struct_mlx5_ifc_query_dct_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  dctn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_dct_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('dctn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_cq_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  cq_context: 'struct_mlx5_ifc_cqc_bits'
  reserved_at_280: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_query_cq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('cq_context', struct_mlx5_ifc_cqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 1536), 640), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_query_cq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  cqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_cq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('cqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_cong_status_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  enable: 'list[int]'
  tag_enable: 'list[int]'
  reserved_at_62: 'list[int]'
struct_mlx5_ifc_query_cong_status_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('enable', (ctypes.c_ubyte * 1), 96), ('tag_enable', (ctypes.c_ubyte * 1), 97), ('reserved_at_62', (ctypes.c_ubyte * 30), 98)])
@c.record
class struct_mlx5_ifc_query_cong_status_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  priority: 'list[int]'
  cong_protocol: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_cong_status_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('priority', (ctypes.c_ubyte * 4), 88), ('cong_protocol', (ctypes.c_ubyte * 4), 92), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_cong_statistics_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rp_cur_flows: 'list[int]'
  sum_flows: 'list[int]'
  rp_cnp_ignored_high: 'list[int]'
  rp_cnp_ignored_low: 'list[int]'
  rp_cnp_handled_high: 'list[int]'
  rp_cnp_handled_low: 'list[int]'
  reserved_at_140: 'list[int]'
  time_stamp_high: 'list[int]'
  time_stamp_low: 'list[int]'
  accumulators_period: 'list[int]'
  np_ecn_marked_roce_packets_high: 'list[int]'
  np_ecn_marked_roce_packets_low: 'list[int]'
  np_cnp_sent_high: 'list[int]'
  np_cnp_sent_low: 'list[int]'
  reserved_at_320: 'list[int]'
struct_mlx5_ifc_query_cong_statistics_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('rp_cur_flows', (ctypes.c_ubyte * 32), 128), ('sum_flows', (ctypes.c_ubyte * 32), 160), ('rp_cnp_ignored_high', (ctypes.c_ubyte * 32), 192), ('rp_cnp_ignored_low', (ctypes.c_ubyte * 32), 224), ('rp_cnp_handled_high', (ctypes.c_ubyte * 32), 256), ('rp_cnp_handled_low', (ctypes.c_ubyte * 32), 288), ('reserved_at_140', (ctypes.c_ubyte * 256), 320), ('time_stamp_high', (ctypes.c_ubyte * 32), 576), ('time_stamp_low', (ctypes.c_ubyte * 32), 608), ('accumulators_period', (ctypes.c_ubyte * 32), 640), ('np_ecn_marked_roce_packets_high', (ctypes.c_ubyte * 32), 672), ('np_ecn_marked_roce_packets_low', (ctypes.c_ubyte * 32), 704), ('np_cnp_sent_high', (ctypes.c_ubyte * 32), 736), ('np_cnp_sent_low', (ctypes.c_ubyte * 32), 768), ('reserved_at_320', (ctypes.c_ubyte * 1376), 800)])
@c.record
class struct_mlx5_ifc_query_cong_statistics_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  clear: 'list[int]'
  reserved_at_41: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_cong_statistics_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('clear', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 31), 65), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_cong_params_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  congestion_parameters: 'union_mlx5_ifc_cong_control_roce_ecn_auto_bits'
struct_mlx5_ifc_query_cong_params_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('congestion_parameters', union_mlx5_ifc_cong_control_roce_ecn_auto_bits, 128)])
@c.record
class struct_mlx5_ifc_query_cong_params_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  cong_protocol: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_cong_params_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 28), 64), ('cong_protocol', (ctypes.c_ubyte * 4), 92), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_adapter_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  query_adapter_struct: 'struct_mlx5_ifc_query_adapter_param_block_bits'
struct_mlx5_ifc_query_adapter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('query_adapter_struct', struct_mlx5_ifc_query_adapter_param_block_bits, 128)])
@c.record
class struct_mlx5_ifc_query_adapter_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_adapter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_function_vhca_rid_info_reg_bits(c.Struct):
  SIZE = 128
  host_number: 'list[int]'
  host_pci_device_function: 'list[int]'
  host_pci_bus: 'list[int]'
  reserved_at_18: 'list[int]'
  pci_bus_assigned: 'list[int]'
  function_type: 'list[int]'
  parent_pci_device_function: 'list[int]'
  parent_pci_bus: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_40: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_function_vhca_rid_info_reg_bits.register_fields([('host_number', (ctypes.c_ubyte * 8), 0), ('host_pci_device_function', (ctypes.c_ubyte * 8), 8), ('host_pci_bus', (ctypes.c_ubyte * 8), 16), ('reserved_at_18', (ctypes.c_ubyte * 3), 24), ('pci_bus_assigned', (ctypes.c_ubyte * 1), 27), ('function_type', (ctypes.c_ubyte * 4), 28), ('parent_pci_device_function', (ctypes.c_ubyte * 8), 32), ('parent_pci_bus', (ctypes.c_ubyte * 8), 40), ('vhca_id', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_delegated_function_vhca_rid_info_bits(c.Struct):
  SIZE = 256
  function_vhca_rid_info: 'struct_mlx5_ifc_function_vhca_rid_info_reg_bits'
  reserved_at_80: 'list[int]'
  manage_profile: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_delegated_function_vhca_rid_info_bits.register_fields([('function_vhca_rid_info', struct_mlx5_ifc_function_vhca_rid_info_reg_bits, 0), ('reserved_at_80', (ctypes.c_ubyte * 24), 128), ('manage_profile', (ctypes.c_ubyte * 8), 152), ('reserved_at_a0', (ctypes.c_ubyte * 96), 160)])
@c.record
class struct_mlx5_ifc_query_delegated_vhca_out_bits(c.Struct):
  SIZE = 256
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  functions_count: 'list[int]'
  reserved_at_80: 'list[int]'
  delegated_function_vhca_rid_info: 'list[struct_mlx5_ifc_delegated_function_vhca_rid_info_bits]'
struct_mlx5_ifc_query_delegated_vhca_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('functions_count', (ctypes.c_ubyte * 16), 112), ('reserved_at_80', (ctypes.c_ubyte * 128), 128), ('delegated_function_vhca_rid_info', (struct_mlx5_ifc_delegated_function_vhca_rid_info_bits * 0), 256)])
@c.record
class struct_mlx5_ifc_query_delegated_vhca_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_delegated_vhca_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_create_esw_vport_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  vport_num: 'list[int]'
struct_mlx5_ifc_create_esw_vport_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('vport_num', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_create_esw_vport_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  managed_vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_esw_vport_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('managed_vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_qp_2rst_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_qp_2rst_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_qp_2rst_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_qp_2rst_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_qp_2err_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_qp_2err_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_qp_2err_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_qp_2err_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_trans_page_fault_info_bits(c.Struct):
  SIZE = 64
  error: 'list[int]'
  reserved_at_1: 'list[int]'
  page_fault_type: 'list[int]'
  wq_number: 'list[int]'
  reserved_at_20: 'list[int]'
  fault_token: 'list[int]'
struct_mlx5_ifc_trans_page_fault_info_bits.register_fields([('error', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 4), 1), ('page_fault_type', (ctypes.c_ubyte * 3), 5), ('wq_number', (ctypes.c_ubyte * 24), 8), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('fault_token', (ctypes.c_ubyte * 24), 40)])
@c.record
class struct_mlx5_ifc_mem_page_fault_info_bits(c.Struct):
  SIZE = 64
  error: 'list[int]'
  reserved_at_1: 'list[int]'
  fault_token_47_32: 'list[int]'
  fault_token_31_0: 'list[int]'
struct_mlx5_ifc_mem_page_fault_info_bits.register_fields([('error', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 15), 1), ('fault_token_47_32', (ctypes.c_ubyte * 16), 16), ('fault_token_31_0', (ctypes.c_ubyte * 32), 32)])
@c.record
class union_mlx5_ifc_page_fault_resume_in_page_fault_info_auto_bits(c.Struct):
  SIZE = 64
  trans_page_fault_info: 'struct_mlx5_ifc_trans_page_fault_info_bits'
  mem_page_fault_info: 'struct_mlx5_ifc_mem_page_fault_info_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_page_fault_resume_in_page_fault_info_auto_bits.register_fields([('trans_page_fault_info', struct_mlx5_ifc_trans_page_fault_info_bits, 0), ('mem_page_fault_info', struct_mlx5_ifc_mem_page_fault_info_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 64), 0)])
@c.record
class struct_mlx5_ifc_page_fault_resume_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_page_fault_resume_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_page_fault_resume_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  page_fault_info: 'union_mlx5_ifc_page_fault_resume_in_page_fault_info_auto_bits'
struct_mlx5_ifc_page_fault_resume_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('page_fault_info', union_mlx5_ifc_page_fault_resume_in_page_fault_info_auto_bits, 64)])
@c.record
class struct_mlx5_ifc_nop_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_nop_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_nop_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_nop_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_vport_state_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_vport_state_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_vport_state_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  ingress_connect: 'list[int]'
  egress_connect: 'list[int]'
  ingress_connect_valid: 'list[int]'
  egress_connect_valid: 'list[int]'
  reserved_at_74: 'list[int]'
  admin_state: 'list[int]'
  reserved_at_7c: 'list[int]'
struct_mlx5_ifc_modify_vport_state_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('ingress_connect', (ctypes.c_ubyte * 1), 112), ('egress_connect', (ctypes.c_ubyte * 1), 113), ('ingress_connect_valid', (ctypes.c_ubyte * 1), 114), ('egress_connect_valid', (ctypes.c_ubyte * 1), 115), ('reserved_at_74', (ctypes.c_ubyte * 4), 116), ('admin_state', (ctypes.c_ubyte * 4), 120), ('reserved_at_7c', (ctypes.c_ubyte * 4), 124)])
@c.record
class struct_mlx5_ifc_modify_tis_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_tis_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_tis_bitmask_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  reserved_at_20: 'list[int]'
  lag_tx_port_affinity: 'list[int]'
  strict_lag_tx_port_affinity: 'list[int]'
  prio: 'list[int]'
struct_mlx5_ifc_modify_tis_bitmask_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 29), 32), ('lag_tx_port_affinity', (ctypes.c_ubyte * 1), 61), ('strict_lag_tx_port_affinity', (ctypes.c_ubyte * 1), 62), ('prio', (ctypes.c_ubyte * 1), 63)])
@c.record
class struct_mlx5_ifc_modify_tis_in_bits(c.Struct):
  SIZE = 1536
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  tisn: 'list[int]'
  reserved_at_60: 'list[int]'
  bitmask: 'struct_mlx5_ifc_modify_tis_bitmask_bits'
  reserved_at_c0: 'list[int]'
  ctx: 'struct_mlx5_ifc_tisc_bits'
struct_mlx5_ifc_modify_tis_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('tisn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('bitmask', struct_mlx5_ifc_modify_tis_bitmask_bits, 128), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('ctx', struct_mlx5_ifc_tisc_bits, 256)])
@c.record
class struct_mlx5_ifc_modify_tir_bitmask_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  reserved_at_20: 'list[int]'
  self_lb_en: 'list[int]'
  reserved_at_3c: 'list[int]'
  hash: 'list[int]'
  reserved_at_3e: 'list[int]'
  packet_merge: 'list[int]'
struct_mlx5_ifc_modify_tir_bitmask_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 27), 32), ('self_lb_en', (ctypes.c_ubyte * 1), 59), ('reserved_at_3c', (ctypes.c_ubyte * 1), 60), ('hash', (ctypes.c_ubyte * 1), 61), ('reserved_at_3e', (ctypes.c_ubyte * 1), 62), ('packet_merge', (ctypes.c_ubyte * 1), 63)])
@c.record
class struct_mlx5_ifc_modify_tir_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_tir_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_tir_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  tirn: 'list[int]'
  reserved_at_60: 'list[int]'
  bitmask: 'struct_mlx5_ifc_modify_tir_bitmask_bits'
  reserved_at_c0: 'list[int]'
  ctx: 'struct_mlx5_ifc_tirc_bits'
struct_mlx5_ifc_modify_tir_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('tirn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('bitmask', struct_mlx5_ifc_modify_tir_bitmask_bits, 128), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('ctx', struct_mlx5_ifc_tirc_bits, 256)])
@c.record
class struct_mlx5_ifc_modify_sq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_sq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_sq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  sq_state: 'list[int]'
  reserved_at_44: 'list[int]'
  sqn: 'list[int]'
  reserved_at_60: 'list[int]'
  modify_bitmask: 'list[int]'
  reserved_at_c0: 'list[int]'
  ctx: 'struct_mlx5_ifc_sqc_bits'
struct_mlx5_ifc_modify_sq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('sq_state', (ctypes.c_ubyte * 4), 64), ('reserved_at_44', (ctypes.c_ubyte * 4), 68), ('sqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('modify_bitmask', (ctypes.c_ubyte * 64), 128), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('ctx', struct_mlx5_ifc_sqc_bits, 256)])
@c.record
class struct_mlx5_ifc_modify_scheduling_element_out_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_scheduling_element_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 448), 64)])
_anonenum100: dict[int, str] = {(MODIFY_SCHEDULING_ELEMENT_IN_MODIFY_BITMASK_BW_SHARE:=1): 'MODIFY_SCHEDULING_ELEMENT_IN_MODIFY_BITMASK_BW_SHARE', (MODIFY_SCHEDULING_ELEMENT_IN_MODIFY_BITMASK_MAX_AVERAGE_BW:=2): 'MODIFY_SCHEDULING_ELEMENT_IN_MODIFY_BITMASK_MAX_AVERAGE_BW'}
@c.record
class struct_mlx5_ifc_modify_scheduling_element_in_bits(c.Struct):
  SIZE = 1024
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  scheduling_hierarchy: 'list[int]'
  reserved_at_48: 'list[int]'
  scheduling_element_id: 'list[int]'
  reserved_at_80: 'list[int]'
  modify_bitmask: 'list[int]'
  reserved_at_c0: 'list[int]'
  scheduling_context: 'struct_mlx5_ifc_scheduling_context_bits'
  reserved_at_300: 'list[int]'
struct_mlx5_ifc_modify_scheduling_element_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('scheduling_hierarchy', (ctypes.c_ubyte * 8), 64), ('reserved_at_48', (ctypes.c_ubyte * 24), 72), ('scheduling_element_id', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 32), 128), ('modify_bitmask', (ctypes.c_ubyte * 32), 160), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('scheduling_context', struct_mlx5_ifc_scheduling_context_bits, 256), ('reserved_at_300', (ctypes.c_ubyte * 256), 768)])
@c.record
class struct_mlx5_ifc_modify_rqt_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_rqt_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_rqt_bitmask_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  reserved_at_20: 'list[int]'
  rqn_list: 'list[int]'
struct_mlx5_ifc_rqt_bitmask_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 31), 32), ('rqn_list', (ctypes.c_ubyte * 1), 63)])
@c.record
class struct_mlx5_ifc_modify_rqt_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rqtn: 'list[int]'
  reserved_at_60: 'list[int]'
  bitmask: 'struct_mlx5_ifc_rqt_bitmask_bits'
  reserved_at_c0: 'list[int]'
  ctx: 'struct_mlx5_ifc_rqtc_bits'
struct_mlx5_ifc_modify_rqt_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rqtn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('bitmask', struct_mlx5_ifc_rqt_bitmask_bits, 128), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('ctx', struct_mlx5_ifc_rqtc_bits, 256)])
@c.record
class struct_mlx5_ifc_modify_rq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_rq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
_anonenum101: dict[int, str] = {(MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_VSD:=2): 'MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_VSD', (MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_SCATTER_FCS:=4): 'MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_SCATTER_FCS', (MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_RQ_COUNTER_SET_ID:=8): 'MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_RQ_COUNTER_SET_ID'}
@c.record
class struct_mlx5_ifc_modify_rq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  rq_state: 'list[int]'
  reserved_at_44: 'list[int]'
  rqn: 'list[int]'
  reserved_at_60: 'list[int]'
  modify_bitmask: 'list[int]'
  reserved_at_c0: 'list[int]'
  ctx: 'struct_mlx5_ifc_rqc_bits'
struct_mlx5_ifc_modify_rq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('rq_state', (ctypes.c_ubyte * 4), 64), ('reserved_at_44', (ctypes.c_ubyte * 4), 68), ('rqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('modify_bitmask', (ctypes.c_ubyte * 64), 128), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('ctx', struct_mlx5_ifc_rqc_bits, 256)])
@c.record
class struct_mlx5_ifc_modify_rmp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_rmp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_rmp_bitmask_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  reserved_at_20: 'list[int]'
  lwm: 'list[int]'
struct_mlx5_ifc_rmp_bitmask_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 31), 32), ('lwm', (ctypes.c_ubyte * 1), 63)])
@c.record
class struct_mlx5_ifc_modify_rmp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  rmp_state: 'list[int]'
  reserved_at_44: 'list[int]'
  rmpn: 'list[int]'
  reserved_at_60: 'list[int]'
  bitmask: 'struct_mlx5_ifc_rmp_bitmask_bits'
  reserved_at_c0: 'list[int]'
  ctx: 'struct_mlx5_ifc_rmpc_bits'
struct_mlx5_ifc_modify_rmp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('rmp_state', (ctypes.c_ubyte * 4), 64), ('reserved_at_44', (ctypes.c_ubyte * 4), 68), ('rmpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('bitmask', struct_mlx5_ifc_rmp_bitmask_bits, 128), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('ctx', struct_mlx5_ifc_rmpc_bits, 256)])
@c.record
class struct_mlx5_ifc_modify_nic_vport_context_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_nic_vport_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_nic_vport_field_select_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  affiliation: 'list[int]'
  reserved_at_13: 'list[int]'
  disable_uc_local_lb: 'list[int]'
  disable_mc_local_lb: 'list[int]'
  node_guid: 'list[int]'
  port_guid: 'list[int]'
  min_inline: 'list[int]'
  mtu: 'list[int]'
  change_event: 'list[int]'
  promisc: 'list[int]'
  permanent_address: 'list[int]'
  addresses_list: 'list[int]'
  roce_en: 'list[int]'
  reserved_at_1f: 'list[int]'
struct_mlx5_ifc_modify_nic_vport_field_select_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 18), 0), ('affiliation', (ctypes.c_ubyte * 1), 18), ('reserved_at_13', (ctypes.c_ubyte * 1), 19), ('disable_uc_local_lb', (ctypes.c_ubyte * 1), 20), ('disable_mc_local_lb', (ctypes.c_ubyte * 1), 21), ('node_guid', (ctypes.c_ubyte * 1), 22), ('port_guid', (ctypes.c_ubyte * 1), 23), ('min_inline', (ctypes.c_ubyte * 1), 24), ('mtu', (ctypes.c_ubyte * 1), 25), ('change_event', (ctypes.c_ubyte * 1), 26), ('promisc', (ctypes.c_ubyte * 1), 27), ('permanent_address', (ctypes.c_ubyte * 1), 28), ('addresses_list', (ctypes.c_ubyte * 1), 29), ('roce_en', (ctypes.c_ubyte * 1), 30), ('reserved_at_1f', (ctypes.c_ubyte * 1), 31)])
@c.record
class struct_mlx5_ifc_modify_nic_vport_context_in_bits(c.Struct):
  SIZE = 4096
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  field_select: 'struct_mlx5_ifc_modify_nic_vport_field_select_bits'
  reserved_at_80: 'list[int]'
  nic_vport_context: 'struct_mlx5_ifc_nic_vport_context_bits'
struct_mlx5_ifc_modify_nic_vport_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('field_select', struct_mlx5_ifc_modify_nic_vport_field_select_bits, 96), ('reserved_at_80', (ctypes.c_ubyte * 1920), 128), ('nic_vport_context', struct_mlx5_ifc_nic_vport_context_bits, 2048)])
@c.record
class struct_mlx5_ifc_modify_hca_vport_context_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_hca_vport_context_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_hca_vport_context_in_bits(c.Struct):
  SIZE = 4224
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  port_num: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  hca_vport_context: 'struct_mlx5_ifc_hca_vport_context_bits'
struct_mlx5_ifc_modify_hca_vport_context_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 11), 65), ('port_num', (ctypes.c_ubyte * 4), 76), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('hca_vport_context', struct_mlx5_ifc_hca_vport_context_bits, 128)])
@c.record
class struct_mlx5_ifc_modify_cq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_cq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
_anonenum102: dict[int, str] = {(MLX5_MODIFY_CQ_IN_OP_MOD_MODIFY_CQ:=0): 'MLX5_MODIFY_CQ_IN_OP_MOD_MODIFY_CQ', (MLX5_MODIFY_CQ_IN_OP_MOD_RESIZE_CQ:=1): 'MLX5_MODIFY_CQ_IN_OP_MOD_RESIZE_CQ'}
@c.record
class struct_mlx5_ifc_modify_cq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  cqn: 'list[int]'
  modify_field_select_resize_field_select: 'union_mlx5_ifc_modify_field_select_resize_field_select_auto_bits'
  cq_context: 'struct_mlx5_ifc_cqc_bits'
  reserved_at_280: 'list[int]'
  cq_umem_valid: 'list[int]'
  reserved_at_2e1: 'list[int]'
  reserved_at_300: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_modify_cq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('cqn', (ctypes.c_ubyte * 24), 72), ('modify_field_select_resize_field_select', union_mlx5_ifc_modify_field_select_resize_field_select_auto_bits, 96), ('cq_context', struct_mlx5_ifc_cqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 96), 640), ('cq_umem_valid', (ctypes.c_ubyte * 1), 736), ('reserved_at_2e1', (ctypes.c_ubyte * 31), 737), ('reserved_at_300', (ctypes.c_ubyte * 1408), 768), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_modify_cong_status_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_cong_status_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_cong_status_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  priority: 'list[int]'
  cong_protocol: 'list[int]'
  enable: 'list[int]'
  tag_enable: 'list[int]'
  reserved_at_62: 'list[int]'
struct_mlx5_ifc_modify_cong_status_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('priority', (ctypes.c_ubyte * 4), 88), ('cong_protocol', (ctypes.c_ubyte * 4), 92), ('enable', (ctypes.c_ubyte * 1), 96), ('tag_enable', (ctypes.c_ubyte * 1), 97), ('reserved_at_62', (ctypes.c_ubyte * 30), 98)])
@c.record
class struct_mlx5_ifc_modify_cong_params_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_cong_params_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_cong_params_in_bits(c.Struct):
  SIZE = 2304
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  cong_protocol: 'list[int]'
  field_select: 'union_mlx5_ifc_field_select_802_1_r_roce_auto_bits'
  reserved_at_80: 'list[int]'
  congestion_parameters: 'union_mlx5_ifc_cong_control_roce_ecn_auto_bits'
struct_mlx5_ifc_modify_cong_params_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 28), 64), ('cong_protocol', (ctypes.c_ubyte * 4), 92), ('field_select', union_mlx5_ifc_field_select_802_1_r_roce_auto_bits, 96), ('reserved_at_80', (ctypes.c_ubyte * 128), 128), ('congestion_parameters', union_mlx5_ifc_cong_control_roce_ecn_auto_bits, 256)])
@c.record
class struct_mlx5_ifc_manage_pages_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  output_num_entries: 'list[int]'
  reserved_at_60: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_manage_pages_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('output_num_entries', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('pas', ((ctypes.c_ubyte * 64) * 0), 128)])
_anonenum103: dict[int, str] = {(MLX5_MANAGE_PAGES_IN_OP_MOD_ALLOCATION_FAIL:=0): 'MLX5_MANAGE_PAGES_IN_OP_MOD_ALLOCATION_FAIL', (MLX5_MANAGE_PAGES_IN_OP_MOD_ALLOCATION_SUCCESS:=1): 'MLX5_MANAGE_PAGES_IN_OP_MOD_ALLOCATION_SUCCESS', (MLX5_MANAGE_PAGES_IN_OP_MOD_HCA_RETURN_PAGES:=2): 'MLX5_MANAGE_PAGES_IN_OP_MOD_HCA_RETURN_PAGES'}
@c.record
class struct_mlx5_ifc_manage_pages_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  embedded_cpu_function: 'list[int]'
  reserved_at_41: 'list[int]'
  function_id: 'list[int]'
  input_num_entries: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_manage_pages_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('embedded_cpu_function', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('function_id', (ctypes.c_ubyte * 16), 80), ('input_num_entries', (ctypes.c_ubyte * 32), 96), ('pas', ((ctypes.c_ubyte * 64) * 0), 128)])
@c.record
class struct_mlx5_ifc_mad_ifc_out_bits(c.Struct):
  SIZE = 2176
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  response_mad_packet: 'list[list[int]]'
struct_mlx5_ifc_mad_ifc_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('response_mad_packet', ((ctypes.c_ubyte * 8) * 256), 128)])
@c.record
class struct_mlx5_ifc_mad_ifc_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  remote_lid: 'list[int]'
  plane_index: 'list[int]'
  port: 'list[int]'
  reserved_at_60: 'list[int]'
  mad: 'list[list[int]]'
struct_mlx5_ifc_mad_ifc_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('remote_lid', (ctypes.c_ubyte * 16), 64), ('plane_index', (ctypes.c_ubyte * 8), 80), ('port', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('mad', ((ctypes.c_ubyte * 8) * 256), 128)])
@c.record
class struct_mlx5_ifc_init_hca_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_init_hca_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_init_hca_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  sw_vhca_id: 'list[int]'
  reserved_at_70: 'list[int]'
  sw_owner_id: 'list[list[int]]'
struct_mlx5_ifc_init_hca_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 2), 96), ('sw_vhca_id', (ctypes.c_ubyte * 14), 98), ('reserved_at_70', (ctypes.c_ubyte * 16), 112), ('sw_owner_id', ((ctypes.c_ubyte * 32) * 4), 128)])
@c.record
class struct_mlx5_ifc_init2rtr_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  ece: 'list[int]'
struct_mlx5_ifc_init2rtr_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('ece', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_init2rtr_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  opt_param_mask: 'list[int]'
  ece: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  reserved_at_800: 'list[int]'
struct_mlx5_ifc_init2rtr_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('ece', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('reserved_at_800', (ctypes.c_ubyte * 128), 2048)])
@c.record
class struct_mlx5_ifc_init2init_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  ece: 'list[int]'
struct_mlx5_ifc_init2init_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('ece', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_init2init_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  opt_param_mask: 'list[int]'
  ece: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  reserved_at_800: 'list[int]'
struct_mlx5_ifc_init2init_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('ece', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('reserved_at_800', (ctypes.c_ubyte * 128), 2048)])
@c.record
class struct_mlx5_ifc_get_dropped_packet_log_out_bits(c.Struct):
  SIZE = 1664
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  packet_headers_log: 'list[list[int]]'
  packet_syndrome: 'list[list[int]]'
struct_mlx5_ifc_get_dropped_packet_log_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('packet_headers_log', ((ctypes.c_ubyte * 8) * 128), 128), ('packet_syndrome', ((ctypes.c_ubyte * 8) * 64), 1152)])
@c.record
class struct_mlx5_ifc_get_dropped_packet_log_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_get_dropped_packet_log_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_gen_eqe_in_bits(c.Struct):
  SIZE = 640
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  eq_number: 'list[int]'
  reserved_at_60: 'list[int]'
  eqe: 'list[list[int]]'
struct_mlx5_ifc_gen_eqe_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('eq_number', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('eqe', ((ctypes.c_ubyte * 8) * 64), 128)])
@c.record
class struct_mlx5_ifc_gen_eq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_gen_eq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_enable_hca_out_bits(c.Struct):
  SIZE = 96
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_enable_hca_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64)])
@c.record
class struct_mlx5_ifc_enable_hca_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  embedded_cpu_function: 'list[int]'
  reserved_at_41: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_enable_hca_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('embedded_cpu_function', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_drain_dct_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_drain_dct_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_drain_dct_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  dctn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_drain_dct_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('dctn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_disable_hca_out_bits(c.Struct):
  SIZE = 96
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_disable_hca_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64)])
@c.record
class struct_mlx5_ifc_disable_hca_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  embedded_cpu_function: 'list[int]'
  reserved_at_41: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_disable_hca_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('embedded_cpu_function', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_detach_from_mcg_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_detach_from_mcg_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_detach_from_mcg_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  multicast_gid: 'list[list[int]]'
struct_mlx5_ifc_detach_from_mcg_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('multicast_gid', ((ctypes.c_ubyte * 8) * 16), 128)])
@c.record
class struct_mlx5_ifc_destroy_xrq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_xrq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_xrq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_xrq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_xrc_srq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_xrc_srq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_xrc_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrc_srqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_xrc_srq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrc_srqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_tis_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_tis_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_tis_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  tisn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_tis_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('tisn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_tir_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_tir_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_tir_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  tirn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_tir_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('tirn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_srq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_srq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  srqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_srq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('srqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_sq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_sq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_sq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  sqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_sq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('sqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_scheduling_element_out_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_scheduling_element_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 448), 64)])
@c.record
class struct_mlx5_ifc_destroy_scheduling_element_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  scheduling_hierarchy: 'list[int]'
  reserved_at_48: 'list[int]'
  scheduling_element_id: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_destroy_scheduling_element_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('scheduling_hierarchy', (ctypes.c_ubyte * 8), 64), ('reserved_at_48', (ctypes.c_ubyte * 24), 72), ('scheduling_element_id', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
@c.record
class struct_mlx5_ifc_destroy_rqt_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_rqt_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_rqt_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rqtn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_rqt_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rqtn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_rq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_rq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_rq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_rq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_set_delay_drop_params_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  delay_drop_timeout: 'list[int]'
struct_mlx5_ifc_set_delay_drop_params_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('delay_drop_timeout', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_set_delay_drop_params_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_delay_drop_params_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_rmp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_rmp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_rmp_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rmpn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_rmp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rmpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_qp_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_psv_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_psv_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_psv_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  psvn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_psv_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('psvn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_mkey_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_mkey_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_mkey_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  mkey_index: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_mkey_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('mkey_index', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_flow_table_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_flow_table_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_flow_table_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_destroy_flow_table_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 320), 192)])
@c.record
class struct_mlx5_ifc_destroy_flow_group_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_flow_group_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_flow_group_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  group_id: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_destroy_flow_group_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('group_id', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 288), 224)])
@c.record
class struct_mlx5_ifc_destroy_eq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_eq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_eq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  eq_number: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_eq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('eq_number', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_dct_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_dct_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_dct_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  dctn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_dct_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('dctn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_cq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_cq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_cq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  cqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_cq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('cqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_delete_vxlan_udp_dport_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_delete_vxlan_udp_dport_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_delete_vxlan_udp_dport_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  vxlan_udp_port: 'list[int]'
struct_mlx5_ifc_delete_vxlan_udp_dport_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('vxlan_udp_port', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_delete_l2_table_entry_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_delete_l2_table_entry_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_delete_l2_table_entry_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_index: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_delete_l2_table_entry_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 96), 64), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_index', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 320), 192)])
@c.record
class struct_mlx5_ifc_delete_fte_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_delete_fte_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_delete_fte_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  reserved_at_c0: 'list[int]'
  flow_index: 'list[int]'
  reserved_at_120: 'list[int]'
struct_mlx5_ifc_delete_fte_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('flow_index', (ctypes.c_ubyte * 32), 256), ('reserved_at_120', (ctypes.c_ubyte * 224), 288)])
@c.record
class struct_mlx5_ifc_dealloc_xrcd_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_xrcd_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_xrcd_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrcd: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_dealloc_xrcd_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrcd', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_dealloc_uar_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_uar_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_uar_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  uar: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_dealloc_uar_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('uar', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_dealloc_transport_domain_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_transport_domain_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_transport_domain_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  transport_domain: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_dealloc_transport_domain_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('transport_domain', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_dealloc_q_counter_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_q_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_q_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  counter_set_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_dealloc_q_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('counter_set_id', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_dealloc_pd_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_pd_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_pd_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  pd: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_dealloc_pd_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('pd', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_dealloc_flow_counter_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_flow_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_flow_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  flow_counter_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_dealloc_flow_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('flow_counter_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_xrq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  xrqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_xrq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_xrq_in_bits(c.Struct):
  SIZE = 2688
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrq_context: 'struct_mlx5_ifc_xrqc_bits'
struct_mlx5_ifc_create_xrq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('xrq_context', struct_mlx5_ifc_xrqc_bits, 128)])
@c.record
class struct_mlx5_ifc_create_xrc_srq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  xrc_srqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_xrc_srq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrc_srqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_xrc_srq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrc_srq_context_entry: 'struct_mlx5_ifc_xrc_srqc_bits'
  reserved_at_280: 'list[int]'
  xrc_srq_umem_valid: 'list[int]'
  reserved_at_2e1: 'list[int]'
  reserved_at_300: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_create_xrc_srq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('xrc_srq_context_entry', struct_mlx5_ifc_xrc_srqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 96), 640), ('xrc_srq_umem_valid', (ctypes.c_ubyte * 1), 736), ('reserved_at_2e1', (ctypes.c_ubyte * 31), 737), ('reserved_at_300', (ctypes.c_ubyte * 1408), 768), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_create_tis_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  tisn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_tis_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('tisn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_tis_in_bits(c.Struct):
  SIZE = 1536
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  ctx: 'struct_mlx5_ifc_tisc_bits'
struct_mlx5_ifc_create_tis_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('ctx', struct_mlx5_ifc_tisc_bits, 256)])
@c.record
class struct_mlx5_ifc_create_tir_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  icm_address_63_40: 'list[int]'
  syndrome: 'list[int]'
  icm_address_39_32: 'list[int]'
  tirn: 'list[int]'
  icm_address_31_0: 'list[int]'
struct_mlx5_ifc_create_tir_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('icm_address_63_40', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('icm_address_39_32', (ctypes.c_ubyte * 8), 64), ('tirn', (ctypes.c_ubyte * 24), 72), ('icm_address_31_0', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_tir_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  ctx: 'struct_mlx5_ifc_tirc_bits'
struct_mlx5_ifc_create_tir_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('ctx', struct_mlx5_ifc_tirc_bits, 256)])
@c.record
class struct_mlx5_ifc_create_srq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  srqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_srq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('srqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_srq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  srq_context_entry: 'struct_mlx5_ifc_srqc_bits'
  reserved_at_280: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_create_srq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('srq_context_entry', struct_mlx5_ifc_srqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 1536), 640), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_create_sq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  sqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_sq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('sqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_sq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  ctx: 'struct_mlx5_ifc_sqc_bits'
struct_mlx5_ifc_create_sq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('ctx', struct_mlx5_ifc_sqc_bits, 256)])
@c.record
class struct_mlx5_ifc_create_scheduling_element_out_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  scheduling_element_id: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_create_scheduling_element_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('scheduling_element_id', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 352), 160)])
@c.record
class struct_mlx5_ifc_create_scheduling_element_in_bits(c.Struct):
  SIZE = 1024
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  scheduling_hierarchy: 'list[int]'
  reserved_at_48: 'list[int]'
  reserved_at_60: 'list[int]'
  scheduling_context: 'struct_mlx5_ifc_scheduling_context_bits'
  reserved_at_300: 'list[int]'
struct_mlx5_ifc_create_scheduling_element_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('scheduling_hierarchy', (ctypes.c_ubyte * 8), 64), ('reserved_at_48', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 160), 96), ('scheduling_context', struct_mlx5_ifc_scheduling_context_bits, 256), ('reserved_at_300', (ctypes.c_ubyte * 256), 768)])
@c.record
class struct_mlx5_ifc_create_rqt_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rqtn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_rqt_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rqtn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_rqt_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rqt_context: 'struct_mlx5_ifc_rqtc_bits'
struct_mlx5_ifc_create_rqt_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('rqt_context', struct_mlx5_ifc_rqtc_bits, 256)])
@c.record
class struct_mlx5_ifc_create_rq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_rq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_rq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  ctx: 'struct_mlx5_ifc_rqc_bits'
struct_mlx5_ifc_create_rq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('ctx', struct_mlx5_ifc_rqc_bits, 256)])
@c.record
class struct_mlx5_ifc_create_rmp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rmpn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_rmp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('rmpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_rmp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  ctx: 'struct_mlx5_ifc_rmpc_bits'
struct_mlx5_ifc_create_rmp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 192), 64), ('ctx', struct_mlx5_ifc_rmpc_bits, 256)])
@c.record
class struct_mlx5_ifc_create_qp_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  ece: 'list[int]'
struct_mlx5_ifc_create_qp_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('ece', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  qpc_ext: 'list[int]'
  reserved_at_41: 'list[int]'
  input_qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  opt_param_mask: 'list[int]'
  ece: 'list[int]'
  qpc: 'struct_mlx5_ifc_qpc_bits'
  wq_umem_offset: 'list[int]'
  wq_umem_id: 'list[int]'
  wq_umem_valid: 'list[int]'
  reserved_at_861: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_create_qp_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('qpc_ext', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 7), 65), ('input_qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('opt_param_mask', (ctypes.c_ubyte * 32), 128), ('ece', (ctypes.c_ubyte * 32), 160), ('qpc', struct_mlx5_ifc_qpc_bits, 192), ('wq_umem_offset', (ctypes.c_ubyte * 64), 2048), ('wq_umem_id', (ctypes.c_ubyte * 32), 2112), ('wq_umem_valid', (ctypes.c_ubyte * 1), 2144), ('reserved_at_861', (ctypes.c_ubyte * 31), 2145), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_create_psv_out_bits(c.Struct):
  SIZE = 256
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_80: 'list[int]'
  psv0_index: 'list[int]'
  reserved_at_a0: 'list[int]'
  psv1_index: 'list[int]'
  reserved_at_c0: 'list[int]'
  psv2_index: 'list[int]'
  reserved_at_e0: 'list[int]'
  psv3_index: 'list[int]'
struct_mlx5_ifc_create_psv_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('psv0_index', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('psv1_index', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('psv2_index', (ctypes.c_ubyte * 24), 200), ('reserved_at_e0', (ctypes.c_ubyte * 8), 224), ('psv3_index', (ctypes.c_ubyte * 24), 232)])
@c.record
class struct_mlx5_ifc_create_psv_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  num_psv: 'list[int]'
  reserved_at_44: 'list[int]'
  pd: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_psv_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('num_psv', (ctypes.c_ubyte * 4), 64), ('reserved_at_44', (ctypes.c_ubyte * 4), 68), ('pd', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_mkey_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  mkey_index: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_mkey_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('mkey_index', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_mkey_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  pg_access: 'list[int]'
  mkey_umem_valid: 'list[int]'
  data_direct: 'list[int]'
  reserved_at_63: 'list[int]'
  memory_key_mkey_entry: 'struct_mlx5_ifc_mkc_bits'
  reserved_at_280: 'list[int]'
  translations_octword_actual_size: 'list[int]'
  reserved_at_320: 'list[int]'
  klm_pas_mtt: 'list[list[int]]'
struct_mlx5_ifc_create_mkey_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('pg_access', (ctypes.c_ubyte * 1), 96), ('mkey_umem_valid', (ctypes.c_ubyte * 1), 97), ('data_direct', (ctypes.c_ubyte * 1), 98), ('reserved_at_63', (ctypes.c_ubyte * 29), 99), ('memory_key_mkey_entry', struct_mlx5_ifc_mkc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 128), 640), ('translations_octword_actual_size', (ctypes.c_ubyte * 32), 768), ('reserved_at_320', (ctypes.c_ubyte * 1376), 800), ('klm_pas_mtt', ((ctypes.c_ubyte * 32) * 0), 2176)])
_anonenum104: dict[int, str] = {(MLX5_FLOW_TABLE_TYPE_NIC_RX:=0): 'MLX5_FLOW_TABLE_TYPE_NIC_RX', (MLX5_FLOW_TABLE_TYPE_NIC_TX:=1): 'MLX5_FLOW_TABLE_TYPE_NIC_TX', (MLX5_FLOW_TABLE_TYPE_ESW_EGRESS_ACL:=2): 'MLX5_FLOW_TABLE_TYPE_ESW_EGRESS_ACL', (MLX5_FLOW_TABLE_TYPE_ESW_INGRESS_ACL:=3): 'MLX5_FLOW_TABLE_TYPE_ESW_INGRESS_ACL', (MLX5_FLOW_TABLE_TYPE_FDB:=4): 'MLX5_FLOW_TABLE_TYPE_FDB', (MLX5_FLOW_TABLE_TYPE_SNIFFER_RX:=5): 'MLX5_FLOW_TABLE_TYPE_SNIFFER_RX', (MLX5_FLOW_TABLE_TYPE_SNIFFER_TX:=6): 'MLX5_FLOW_TABLE_TYPE_SNIFFER_TX'}
@c.record
class struct_mlx5_ifc_create_flow_table_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  icm_address_63_40: 'list[int]'
  syndrome: 'list[int]'
  icm_address_39_32: 'list[int]'
  table_id: 'list[int]'
  icm_address_31_0: 'list[int]'
struct_mlx5_ifc_create_flow_table_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('icm_address_63_40', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('icm_address_39_32', (ctypes.c_ubyte * 8), 64), ('table_id', (ctypes.c_ubyte * 24), 72), ('icm_address_31_0', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_flow_table_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  flow_table_context: 'struct_mlx5_ifc_flow_table_context_bits'
struct_mlx5_ifc_create_flow_table_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('flow_table_context', struct_mlx5_ifc_flow_table_context_bits, 192)])
@c.record
class struct_mlx5_ifc_create_flow_group_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  group_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_flow_group_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('group_id', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
_anonenum105: dict[int, str] = {(MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_TCAM_SUBTABLE:=0): 'MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_TCAM_SUBTABLE', (MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_HASH_SPLIT:=1): 'MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_HASH_SPLIT'}
_anonenum106: dict[int, str] = {(MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_OUTER_HEADERS:=0): 'MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_OUTER_HEADERS', (MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS:=1): 'MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS', (MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_INNER_HEADERS:=2): 'MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_INNER_HEADERS', (MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_2:=3): 'MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_2'}
@c.record
class struct_mlx5_ifc_create_flow_group_in_bits(c.Struct):
  SIZE = 8192
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  group_type: 'list[int]'
  reserved_at_90: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  source_eswitch_owner_vhca_id_valid: 'list[int]'
  reserved_at_c1: 'list[int]'
  start_flow_index: 'list[int]'
  reserved_at_100: 'list[int]'
  end_flow_index: 'list[int]'
  reserved_at_140: 'list[int]'
  match_definer_id: 'list[int]'
  reserved_at_160: 'list[int]'
  reserved_at_1e0: 'list[int]'
  match_criteria_enable: 'list[int]'
  match_criteria: 'struct_mlx5_ifc_fte_match_param_bits'
  reserved_at_1200: 'list[int]'
struct_mlx5_ifc_create_flow_group_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 4), 136), ('group_type', (ctypes.c_ubyte * 4), 140), ('reserved_at_90', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('source_eswitch_owner_vhca_id_valid', (ctypes.c_ubyte * 1), 192), ('reserved_at_c1', (ctypes.c_ubyte * 31), 193), ('start_flow_index', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 32), 256), ('end_flow_index', (ctypes.c_ubyte * 32), 288), ('reserved_at_140', (ctypes.c_ubyte * 16), 320), ('match_definer_id', (ctypes.c_ubyte * 16), 336), ('reserved_at_160', (ctypes.c_ubyte * 128), 352), ('reserved_at_1e0', (ctypes.c_ubyte * 24), 480), ('match_criteria_enable', (ctypes.c_ubyte * 8), 504), ('match_criteria', struct_mlx5_ifc_fte_match_param_bits, 512), ('reserved_at_1200', (ctypes.c_ubyte * 3584), 4608)])
@c.record
class struct_mlx5_ifc_create_eq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  eq_number: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_eq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('eq_number', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_eq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  eq_context_entry: 'struct_mlx5_ifc_eqc_bits'
  reserved_at_280: 'list[int]'
  event_bitmask: 'list[list[int]]'
  reserved_at_3c0: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_create_eq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('eq_context_entry', struct_mlx5_ifc_eqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 64), 640), ('event_bitmask', ((ctypes.c_ubyte * 64) * 4), 704), ('reserved_at_3c0', (ctypes.c_ubyte * 1216), 960), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_create_dct_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  dctn: 'list[int]'
  ece: 'list[int]'
struct_mlx5_ifc_create_dct_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('dctn', (ctypes.c_ubyte * 24), 72), ('ece', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_dct_in_bits(c.Struct):
  SIZE = 1024
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  dct_context_entry: 'struct_mlx5_ifc_dctc_bits'
  reserved_at_280: 'list[int]'
struct_mlx5_ifc_create_dct_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('dct_context_entry', struct_mlx5_ifc_dctc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 384), 640)])
@c.record
class struct_mlx5_ifc_create_cq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  cqn: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_cq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('cqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_create_cq_in_bits(c.Struct):
  SIZE = 2176
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  cq_context: 'struct_mlx5_ifc_cqc_bits'
  reserved_at_280: 'list[int]'
  cq_umem_valid: 'list[int]'
  reserved_at_2e1: 'list[int]'
  pas: 'list[list[int]]'
struct_mlx5_ifc_create_cq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('cq_context', struct_mlx5_ifc_cqc_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 96), 640), ('cq_umem_valid', (ctypes.c_ubyte * 1), 736), ('reserved_at_2e1', (ctypes.c_ubyte * 1439), 737), ('pas', ((ctypes.c_ubyte * 64) * 0), 2176)])
@c.record
class struct_mlx5_ifc_config_int_moderation_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  min_delay: 'list[int]'
  int_vector: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_config_int_moderation_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 4), 64), ('min_delay', (ctypes.c_ubyte * 12), 68), ('int_vector', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
_anonenum107: dict[int, str] = {(MLX5_CONFIG_INT_MODERATION_IN_OP_MOD_WRITE:=0): 'MLX5_CONFIG_INT_MODERATION_IN_OP_MOD_WRITE', (MLX5_CONFIG_INT_MODERATION_IN_OP_MOD_READ:=1): 'MLX5_CONFIG_INT_MODERATION_IN_OP_MOD_READ'}
@c.record
class struct_mlx5_ifc_config_int_moderation_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  min_delay: 'list[int]'
  int_vector: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_config_int_moderation_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 4), 64), ('min_delay', (ctypes.c_ubyte * 12), 68), ('int_vector', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_attach_to_mcg_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_attach_to_mcg_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_attach_to_mcg_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  qpn: 'list[int]'
  reserved_at_60: 'list[int]'
  multicast_gid: 'list[list[int]]'
struct_mlx5_ifc_attach_to_mcg_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('qpn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('multicast_gid', ((ctypes.c_ubyte * 8) * 16), 128)])
@c.record
class struct_mlx5_ifc_arm_xrq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_arm_xrq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_arm_xrq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrqn: 'list[int]'
  reserved_at_60: 'list[int]'
  lwm: 'list[int]'
struct_mlx5_ifc_arm_xrq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('lwm', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_arm_xrc_srq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_arm_xrc_srq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
_anonenum108: dict[int, str] = {(MLX5_ARM_XRC_SRQ_IN_OP_MOD_XRC_SRQ:=1): 'MLX5_ARM_XRC_SRQ_IN_OP_MOD_XRC_SRQ'}
@c.record
class struct_mlx5_ifc_arm_xrc_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  xrc_srqn: 'list[int]'
  reserved_at_60: 'list[int]'
  lwm: 'list[int]'
struct_mlx5_ifc_arm_xrc_srq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrc_srqn', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('lwm', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_arm_rq_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_arm_rq_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
_anonenum109: dict[int, str] = {(MLX5_ARM_RQ_IN_OP_MOD_SRQ:=1): 'MLX5_ARM_RQ_IN_OP_MOD_SRQ', (MLX5_ARM_RQ_IN_OP_MOD_XRQ:=2): 'MLX5_ARM_RQ_IN_OP_MOD_XRQ'}
@c.record
class struct_mlx5_ifc_arm_rq_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  srq_number: 'list[int]'
  reserved_at_60: 'list[int]'
  lwm: 'list[int]'
struct_mlx5_ifc_arm_rq_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('srq_number', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('lwm', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_arm_dct_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_arm_dct_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_arm_dct_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  dct_number: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_arm_dct_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('dct_number', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_xrcd_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  xrcd: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_xrcd_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('xrcd', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_xrcd_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_alloc_xrcd_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_alloc_uar_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  uar: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_uar_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('uar', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_uar_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_alloc_uar_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_alloc_transport_domain_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  transport_domain: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_transport_domain_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('transport_domain', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_transport_domain_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_alloc_transport_domain_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_alloc_q_counter_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  counter_set_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_q_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('counter_set_id', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_q_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_alloc_q_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_alloc_pd_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  pd: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_pd_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('pd', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_pd_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_alloc_pd_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_alloc_flow_counter_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  flow_counter_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_flow_counter_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('flow_counter_id', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_flow_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  flow_counter_bulk_log_size: 'list[int]'
  flow_counter_bulk: 'list[int]'
struct_mlx5_ifc_alloc_flow_counter_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 51), 64), ('flow_counter_bulk_log_size', (ctypes.c_ubyte * 5), 115), ('flow_counter_bulk', (ctypes.c_ubyte * 8), 120)])
@c.record
class struct_mlx5_ifc_add_vxlan_udp_dport_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_add_vxlan_udp_dport_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_add_vxlan_udp_dport_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  vxlan_udp_port: 'list[int]'
struct_mlx5_ifc_add_vxlan_udp_dport_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('vxlan_udp_port', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_set_pp_rate_limit_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_pp_rate_limit_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_pp_rate_limit_context_bits(c.Struct):
  SIZE = 384
  rate_limit: 'list[int]'
  burst_upper_bound: 'list[int]'
  reserved_at_40: 'list[int]'
  typical_packet_size: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_set_pp_rate_limit_context_bits.register_fields([('rate_limit', (ctypes.c_ubyte * 32), 0), ('burst_upper_bound', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('typical_packet_size', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 288), 96)])
@c.record
class struct_mlx5_ifc_set_pp_rate_limit_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  rate_limit_index: 'list[int]'
  reserved_at_60: 'list[int]'
  ctx: 'struct_mlx5_ifc_set_pp_rate_limit_context_bits'
struct_mlx5_ifc_set_pp_rate_limit_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('rate_limit_index', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('ctx', struct_mlx5_ifc_set_pp_rate_limit_context_bits, 128)])
@c.record
class struct_mlx5_ifc_access_register_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  register_data: 'list[list[int]]'
struct_mlx5_ifc_access_register_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('register_data', ((ctypes.c_ubyte * 32) * 0), 128)])
_anonenum110: dict[int, str] = {(MLX5_ACCESS_REGISTER_IN_OP_MOD_WRITE:=0): 'MLX5_ACCESS_REGISTER_IN_OP_MOD_WRITE', (MLX5_ACCESS_REGISTER_IN_OP_MOD_READ:=1): 'MLX5_ACCESS_REGISTER_IN_OP_MOD_READ'}
@c.record
class struct_mlx5_ifc_access_register_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  register_id: 'list[int]'
  argument: 'list[int]'
  register_data: 'list[list[int]]'
struct_mlx5_ifc_access_register_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('register_id', (ctypes.c_ubyte * 16), 80), ('argument', (ctypes.c_ubyte * 32), 96), ('register_data', ((ctypes.c_ubyte * 32) * 0), 128)])
@c.record
class struct_mlx5_ifc_sltp_reg_bits(c.Struct):
  SIZE = 160
  status: 'list[int]'
  version: 'list[int]'
  local_port: 'list[int]'
  pnat: 'list[int]'
  reserved_at_12: 'list[int]'
  lane: 'list[int]'
  reserved_at_18: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_40: 'list[int]'
  polarity: 'list[int]'
  ob_tap0: 'list[int]'
  ob_tap1: 'list[int]'
  ob_tap2: 'list[int]'
  reserved_at_60: 'list[int]'
  ob_preemp_mode: 'list[int]'
  ob_reg: 'list[int]'
  ob_bias: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_sltp_reg_bits.register_fields([('status', (ctypes.c_ubyte * 4), 0), ('version', (ctypes.c_ubyte * 4), 4), ('local_port', (ctypes.c_ubyte * 8), 8), ('pnat', (ctypes.c_ubyte * 2), 16), ('reserved_at_12', (ctypes.c_ubyte * 2), 18), ('lane', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 7), 64), ('polarity', (ctypes.c_ubyte * 1), 71), ('ob_tap0', (ctypes.c_ubyte * 8), 72), ('ob_tap1', (ctypes.c_ubyte * 8), 80), ('ob_tap2', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 12), 96), ('ob_preemp_mode', (ctypes.c_ubyte * 4), 108), ('ob_reg', (ctypes.c_ubyte * 8), 112), ('ob_bias', (ctypes.c_ubyte * 8), 120), ('reserved_at_80', (ctypes.c_ubyte * 32), 128)])
@c.record
class struct_mlx5_ifc_slrg_reg_bits(c.Struct):
  SIZE = 320
  status: 'list[int]'
  version: 'list[int]'
  local_port: 'list[int]'
  pnat: 'list[int]'
  reserved_at_12: 'list[int]'
  lane: 'list[int]'
  reserved_at_18: 'list[int]'
  time_to_link_up: 'list[int]'
  reserved_at_30: 'list[int]'
  grade_lane_speed: 'list[int]'
  grade_version: 'list[int]'
  grade: 'list[int]'
  reserved_at_60: 'list[int]'
  height_grade_type: 'list[int]'
  height_grade: 'list[int]'
  height_dz: 'list[int]'
  height_dv: 'list[int]'
  reserved_at_a0: 'list[int]'
  height_sigma: 'list[int]'
  reserved_at_c0: 'list[int]'
  reserved_at_e0: 'list[int]'
  phase_grade_type: 'list[int]'
  phase_grade: 'list[int]'
  reserved_at_100: 'list[int]'
  phase_eo_pos: 'list[int]'
  reserved_at_110: 'list[int]'
  phase_eo_neg: 'list[int]'
  ffe_set_tested: 'list[int]'
  test_errors_per_lane: 'list[int]'
struct_mlx5_ifc_slrg_reg_bits.register_fields([('status', (ctypes.c_ubyte * 4), 0), ('version', (ctypes.c_ubyte * 4), 4), ('local_port', (ctypes.c_ubyte * 8), 8), ('pnat', (ctypes.c_ubyte * 2), 16), ('reserved_at_12', (ctypes.c_ubyte * 2), 18), ('lane', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 8), 24), ('time_to_link_up', (ctypes.c_ubyte * 16), 32), ('reserved_at_30', (ctypes.c_ubyte * 12), 48), ('grade_lane_speed', (ctypes.c_ubyte * 4), 60), ('grade_version', (ctypes.c_ubyte * 8), 64), ('grade', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 4), 96), ('height_grade_type', (ctypes.c_ubyte * 4), 100), ('height_grade', (ctypes.c_ubyte * 24), 104), ('height_dz', (ctypes.c_ubyte * 16), 128), ('height_dv', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 16), 160), ('height_sigma', (ctypes.c_ubyte * 16), 176), ('reserved_at_c0', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 4), 224), ('phase_grade_type', (ctypes.c_ubyte * 4), 228), ('phase_grade', (ctypes.c_ubyte * 24), 232), ('reserved_at_100', (ctypes.c_ubyte * 8), 256), ('phase_eo_pos', (ctypes.c_ubyte * 8), 264), ('reserved_at_110', (ctypes.c_ubyte * 8), 272), ('phase_eo_neg', (ctypes.c_ubyte * 8), 280), ('ffe_set_tested', (ctypes.c_ubyte * 16), 288), ('test_errors_per_lane', (ctypes.c_ubyte * 16), 304)])
@c.record
class struct_mlx5_ifc_pvlc_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  vl_hw_cap: 'list[int]'
  reserved_at_40: 'list[int]'
  vl_admin: 'list[int]'
  reserved_at_60: 'list[int]'
  vl_operational: 'list[int]'
struct_mlx5_ifc_pvlc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 28), 32), ('vl_hw_cap', (ctypes.c_ubyte * 4), 60), ('reserved_at_40', (ctypes.c_ubyte * 28), 64), ('vl_admin', (ctypes.c_ubyte * 4), 92), ('reserved_at_60', (ctypes.c_ubyte * 28), 96), ('vl_operational', (ctypes.c_ubyte * 4), 124)])
@c.record
class struct_mlx5_ifc_pude_reg_bits(c.Struct):
  SIZE = 128
  swid: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  admin_status: 'list[int]'
  reserved_at_18: 'list[int]'
  oper_status: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_pude_reg_bits.register_fields([('swid', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 4), 16), ('admin_status', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 4), 24), ('oper_status', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 96), 32)])
_anonenum111: dict[int, str] = {(MLX5_PTYS_CONNECTOR_TYPE_PORT_DA:=7): 'MLX5_PTYS_CONNECTOR_TYPE_PORT_DA'}
@c.record
class struct_mlx5_ifc_ptys_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  an_disable_admin: 'list[int]'
  an_disable_cap: 'list[int]'
  reserved_at_3: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  plane_ind: 'list[int]'
  reserved_at_1c: 'list[int]'
  proto_mask: 'list[int]'
  an_status: 'list[int]'
  reserved_at_24: 'list[int]'
  data_rate_oper: 'list[int]'
  ext_eth_proto_capability: 'list[int]'
  eth_proto_capability: 'list[int]'
  ib_link_width_capability: 'list[int]'
  ib_proto_capability: 'list[int]'
  ext_eth_proto_admin: 'list[int]'
  eth_proto_admin: 'list[int]'
  ib_link_width_admin: 'list[int]'
  ib_proto_admin: 'list[int]'
  ext_eth_proto_oper: 'list[int]'
  eth_proto_oper: 'list[int]'
  ib_link_width_oper: 'list[int]'
  ib_proto_oper: 'list[int]'
  reserved_at_160: 'list[int]'
  lane_rate_oper: 'list[int]'
  connector_type: 'list[int]'
  eth_proto_lp_advertise: 'list[int]'
  reserved_at_1a0: 'list[int]'
struct_mlx5_ifc_ptys_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 1), 0), ('an_disable_admin', (ctypes.c_ubyte * 1), 1), ('an_disable_cap', (ctypes.c_ubyte * 1), 2), ('reserved_at_3', (ctypes.c_ubyte * 5), 3), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('plane_ind', (ctypes.c_ubyte * 4), 24), ('reserved_at_1c', (ctypes.c_ubyte * 1), 28), ('proto_mask', (ctypes.c_ubyte * 3), 29), ('an_status', (ctypes.c_ubyte * 4), 32), ('reserved_at_24', (ctypes.c_ubyte * 12), 36), ('data_rate_oper', (ctypes.c_ubyte * 16), 48), ('ext_eth_proto_capability', (ctypes.c_ubyte * 32), 64), ('eth_proto_capability', (ctypes.c_ubyte * 32), 96), ('ib_link_width_capability', (ctypes.c_ubyte * 16), 128), ('ib_proto_capability', (ctypes.c_ubyte * 16), 144), ('ext_eth_proto_admin', (ctypes.c_ubyte * 32), 160), ('eth_proto_admin', (ctypes.c_ubyte * 32), 192), ('ib_link_width_admin', (ctypes.c_ubyte * 16), 224), ('ib_proto_admin', (ctypes.c_ubyte * 16), 240), ('ext_eth_proto_oper', (ctypes.c_ubyte * 32), 256), ('eth_proto_oper', (ctypes.c_ubyte * 32), 288), ('ib_link_width_oper', (ctypes.c_ubyte * 16), 320), ('ib_proto_oper', (ctypes.c_ubyte * 16), 336), ('reserved_at_160', (ctypes.c_ubyte * 8), 352), ('lane_rate_oper', (ctypes.c_ubyte * 20), 360), ('connector_type', (ctypes.c_ubyte * 4), 380), ('eth_proto_lp_advertise', (ctypes.c_ubyte * 32), 384), ('reserved_at_1a0', (ctypes.c_ubyte * 96), 416)])
@c.record
class struct_mlx5_ifc_mlcr_reg_bits(c.Struct):
  SIZE = 96
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  beacon_duration: 'list[int]'
  reserved_at_40: 'list[int]'
  beacon_remain: 'list[int]'
struct_mlx5_ifc_mlcr_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 32), 16), ('beacon_duration', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('beacon_remain', (ctypes.c_ubyte * 16), 80)])
@c.record
class struct_mlx5_ifc_ptas_reg_bits(c.Struct):
  SIZE = 352
  reserved_at_0: 'list[int]'
  algorithm_options: 'list[int]'
  reserved_at_30: 'list[int]'
  repetitions_mode: 'list[int]'
  num_of_repetitions: 'list[int]'
  grade_version: 'list[int]'
  height_grade_type: 'list[int]'
  phase_grade_type: 'list[int]'
  height_grade_weight: 'list[int]'
  phase_grade_weight: 'list[int]'
  gisim_measure_bits: 'list[int]'
  adaptive_tap_measure_bits: 'list[int]'
  ber_bath_high_error_threshold: 'list[int]'
  ber_bath_mid_error_threshold: 'list[int]'
  ber_bath_low_error_threshold: 'list[int]'
  one_ratio_high_threshold: 'list[int]'
  one_ratio_high_mid_threshold: 'list[int]'
  one_ratio_low_mid_threshold: 'list[int]'
  one_ratio_low_threshold: 'list[int]'
  ndeo_error_threshold: 'list[int]'
  mixer_offset_step_size: 'list[int]'
  reserved_at_110: 'list[int]'
  mix90_phase_for_voltage_bath: 'list[int]'
  mixer_offset_start: 'list[int]'
  mixer_offset_end: 'list[int]'
  reserved_at_140: 'list[int]'
  ber_test_time: 'list[int]'
struct_mlx5_ifc_ptas_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('algorithm_options', (ctypes.c_ubyte * 16), 32), ('reserved_at_30', (ctypes.c_ubyte * 4), 48), ('repetitions_mode', (ctypes.c_ubyte * 4), 52), ('num_of_repetitions', (ctypes.c_ubyte * 8), 56), ('grade_version', (ctypes.c_ubyte * 8), 64), ('height_grade_type', (ctypes.c_ubyte * 4), 72), ('phase_grade_type', (ctypes.c_ubyte * 4), 76), ('height_grade_weight', (ctypes.c_ubyte * 8), 80), ('phase_grade_weight', (ctypes.c_ubyte * 8), 88), ('gisim_measure_bits', (ctypes.c_ubyte * 16), 96), ('adaptive_tap_measure_bits', (ctypes.c_ubyte * 16), 112), ('ber_bath_high_error_threshold', (ctypes.c_ubyte * 16), 128), ('ber_bath_mid_error_threshold', (ctypes.c_ubyte * 16), 144), ('ber_bath_low_error_threshold', (ctypes.c_ubyte * 16), 160), ('one_ratio_high_threshold', (ctypes.c_ubyte * 16), 176), ('one_ratio_high_mid_threshold', (ctypes.c_ubyte * 16), 192), ('one_ratio_low_mid_threshold', (ctypes.c_ubyte * 16), 208), ('one_ratio_low_threshold', (ctypes.c_ubyte * 16), 224), ('ndeo_error_threshold', (ctypes.c_ubyte * 16), 240), ('mixer_offset_step_size', (ctypes.c_ubyte * 16), 256), ('reserved_at_110', (ctypes.c_ubyte * 8), 272), ('mix90_phase_for_voltage_bath', (ctypes.c_ubyte * 8), 280), ('mixer_offset_start', (ctypes.c_ubyte * 16), 288), ('mixer_offset_end', (ctypes.c_ubyte * 16), 304), ('reserved_at_140', (ctypes.c_ubyte * 21), 320), ('ber_test_time', (ctypes.c_ubyte * 11), 341)])
@c.record
class struct_mlx5_ifc_pspa_reg_bits(c.Struct):
  SIZE = 64
  swid: 'list[int]'
  local_port: 'list[int]'
  sub_port: 'list[int]'
  reserved_at_18: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_pspa_reg_bits.register_fields([('swid', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('sub_port', (ctypes.c_ubyte * 8), 16), ('reserved_at_18', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_pqdr_reg_bits(c.Struct):
  SIZE = 256
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  prio: 'list[int]'
  reserved_at_18: 'list[int]'
  mode: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_40: 'list[int]'
  min_threshold: 'list[int]'
  reserved_at_60: 'list[int]'
  max_threshold: 'list[int]'
  reserved_at_80: 'list[int]'
  mark_probability_denominator: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_pqdr_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 5), 16), ('prio', (ctypes.c_ubyte * 3), 21), ('reserved_at_18', (ctypes.c_ubyte * 6), 24), ('mode', (ctypes.c_ubyte * 2), 30), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('min_threshold', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('max_threshold', (ctypes.c_ubyte * 16), 112), ('reserved_at_80', (ctypes.c_ubyte * 16), 128), ('mark_probability_denominator', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 96), 160)])
@c.record
class struct_mlx5_ifc_ppsc_reg_bits(c.Struct):
  SIZE = 384
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_80: 'list[int]'
  wrps_admin: 'list[int]'
  reserved_at_a0: 'list[int]'
  wrps_status: 'list[int]'
  reserved_at_c0: 'list[int]'
  up_threshold: 'list[int]'
  reserved_at_d0: 'list[int]'
  down_threshold: 'list[int]'
  reserved_at_e0: 'list[int]'
  reserved_at_100: 'list[int]'
  srps_admin: 'list[int]'
  reserved_at_120: 'list[int]'
  srps_status: 'list[int]'
  reserved_at_140: 'list[int]'
struct_mlx5_ifc_ppsc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 96), 32), ('reserved_at_80', (ctypes.c_ubyte * 28), 128), ('wrps_admin', (ctypes.c_ubyte * 4), 156), ('reserved_at_a0', (ctypes.c_ubyte * 28), 160), ('wrps_status', (ctypes.c_ubyte * 4), 188), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('up_threshold', (ctypes.c_ubyte * 8), 200), ('reserved_at_d0', (ctypes.c_ubyte * 8), 208), ('down_threshold', (ctypes.c_ubyte * 8), 216), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224), ('reserved_at_100', (ctypes.c_ubyte * 28), 256), ('srps_admin', (ctypes.c_ubyte * 4), 284), ('reserved_at_120', (ctypes.c_ubyte * 28), 288), ('srps_status', (ctypes.c_ubyte * 4), 316), ('reserved_at_140', (ctypes.c_ubyte * 64), 320)])
@c.record
class struct_mlx5_ifc_pplr_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  lb_cap: 'list[int]'
  reserved_at_30: 'list[int]'
  lb_en: 'list[int]'
struct_mlx5_ifc_pplr_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('lb_cap', (ctypes.c_ubyte * 8), 40), ('reserved_at_30', (ctypes.c_ubyte * 8), 48), ('lb_en', (ctypes.c_ubyte * 8), 56)])
@c.record
class struct_mlx5_ifc_pplm_reg_bits(c.Struct):
  SIZE = 960
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  port_profile_mode: 'list[int]'
  static_port_profile: 'list[int]'
  active_port_profile: 'list[int]'
  reserved_at_58: 'list[int]'
  retransmission_active: 'list[int]'
  fec_mode_active: 'list[int]'
  rs_fec_correction_bypass_cap: 'list[int]'
  reserved_at_84: 'list[int]'
  fec_override_cap_56g: 'list[int]'
  fec_override_cap_100g: 'list[int]'
  fec_override_cap_50g: 'list[int]'
  fec_override_cap_25g: 'list[int]'
  fec_override_cap_10g_40g: 'list[int]'
  rs_fec_correction_bypass_admin: 'list[int]'
  reserved_at_a4: 'list[int]'
  fec_override_admin_56g: 'list[int]'
  fec_override_admin_100g: 'list[int]'
  fec_override_admin_50g: 'list[int]'
  fec_override_admin_25g: 'list[int]'
  fec_override_admin_10g_40g: 'list[int]'
  fec_override_cap_400g_8x: 'list[int]'
  fec_override_cap_200g_4x: 'list[int]'
  fec_override_cap_100g_2x: 'list[int]'
  fec_override_cap_50g_1x: 'list[int]'
  fec_override_admin_400g_8x: 'list[int]'
  fec_override_admin_200g_4x: 'list[int]'
  fec_override_admin_100g_2x: 'list[int]'
  fec_override_admin_50g_1x: 'list[int]'
  fec_override_cap_800g_8x: 'list[int]'
  fec_override_cap_400g_4x: 'list[int]'
  fec_override_cap_200g_2x: 'list[int]'
  fec_override_cap_100g_1x: 'list[int]'
  reserved_at_180: 'list[int]'
  fec_override_admin_800g_8x: 'list[int]'
  fec_override_admin_400g_4x: 'list[int]'
  fec_override_admin_200g_2x: 'list[int]'
  fec_override_admin_100g_1x: 'list[int]'
  reserved_at_260: 'list[int]'
  fec_override_cap_1600g_8x: 'list[int]'
  fec_override_cap_800g_4x: 'list[int]'
  fec_override_cap_400g_2x: 'list[int]'
  fec_override_cap_200g_1x: 'list[int]'
  fec_override_admin_1600g_8x: 'list[int]'
  fec_override_admin_800g_4x: 'list[int]'
  fec_override_admin_400g_2x: 'list[int]'
  fec_override_admin_200g_1x: 'list[int]'
  reserved_at_340: 'list[int]'
struct_mlx5_ifc_pplm_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('port_profile_mode', (ctypes.c_ubyte * 8), 64), ('static_port_profile', (ctypes.c_ubyte * 8), 72), ('active_port_profile', (ctypes.c_ubyte * 8), 80), ('reserved_at_58', (ctypes.c_ubyte * 8), 88), ('retransmission_active', (ctypes.c_ubyte * 8), 96), ('fec_mode_active', (ctypes.c_ubyte * 24), 104), ('rs_fec_correction_bypass_cap', (ctypes.c_ubyte * 4), 128), ('reserved_at_84', (ctypes.c_ubyte * 8), 132), ('fec_override_cap_56g', (ctypes.c_ubyte * 4), 140), ('fec_override_cap_100g', (ctypes.c_ubyte * 4), 144), ('fec_override_cap_50g', (ctypes.c_ubyte * 4), 148), ('fec_override_cap_25g', (ctypes.c_ubyte * 4), 152), ('fec_override_cap_10g_40g', (ctypes.c_ubyte * 4), 156), ('rs_fec_correction_bypass_admin', (ctypes.c_ubyte * 4), 160), ('reserved_at_a4', (ctypes.c_ubyte * 8), 164), ('fec_override_admin_56g', (ctypes.c_ubyte * 4), 172), ('fec_override_admin_100g', (ctypes.c_ubyte * 4), 176), ('fec_override_admin_50g', (ctypes.c_ubyte * 4), 180), ('fec_override_admin_25g', (ctypes.c_ubyte * 4), 184), ('fec_override_admin_10g_40g', (ctypes.c_ubyte * 4), 188), ('fec_override_cap_400g_8x', (ctypes.c_ubyte * 16), 192), ('fec_override_cap_200g_4x', (ctypes.c_ubyte * 16), 208), ('fec_override_cap_100g_2x', (ctypes.c_ubyte * 16), 224), ('fec_override_cap_50g_1x', (ctypes.c_ubyte * 16), 240), ('fec_override_admin_400g_8x', (ctypes.c_ubyte * 16), 256), ('fec_override_admin_200g_4x', (ctypes.c_ubyte * 16), 272), ('fec_override_admin_100g_2x', (ctypes.c_ubyte * 16), 288), ('fec_override_admin_50g_1x', (ctypes.c_ubyte * 16), 304), ('fec_override_cap_800g_8x', (ctypes.c_ubyte * 16), 320), ('fec_override_cap_400g_4x', (ctypes.c_ubyte * 16), 336), ('fec_override_cap_200g_2x', (ctypes.c_ubyte * 16), 352), ('fec_override_cap_100g_1x', (ctypes.c_ubyte * 16), 368), ('reserved_at_180', (ctypes.c_ubyte * 160), 384), ('fec_override_admin_800g_8x', (ctypes.c_ubyte * 16), 544), ('fec_override_admin_400g_4x', (ctypes.c_ubyte * 16), 560), ('fec_override_admin_200g_2x', (ctypes.c_ubyte * 16), 576), ('fec_override_admin_100g_1x', (ctypes.c_ubyte * 16), 592), ('reserved_at_260', (ctypes.c_ubyte * 96), 608), ('fec_override_cap_1600g_8x', (ctypes.c_ubyte * 16), 704), ('fec_override_cap_800g_4x', (ctypes.c_ubyte * 16), 720), ('fec_override_cap_400g_2x', (ctypes.c_ubyte * 16), 736), ('fec_override_cap_200g_1x', (ctypes.c_ubyte * 16), 752), ('fec_override_admin_1600g_8x', (ctypes.c_ubyte * 16), 768), ('fec_override_admin_800g_4x', (ctypes.c_ubyte * 16), 784), ('fec_override_admin_400g_2x', (ctypes.c_ubyte * 16), 800), ('fec_override_admin_200g_1x', (ctypes.c_ubyte * 16), 816), ('reserved_at_340', (ctypes.c_ubyte * 128), 832)])
@c.record
class struct_mlx5_ifc_ppcnt_reg_bits(c.Struct):
  SIZE = 2048
  swid: 'list[int]'
  local_port: 'list[int]'
  pnat: 'list[int]'
  reserved_at_12: 'list[int]'
  grp: 'list[int]'
  clr: 'list[int]'
  reserved_at_21: 'list[int]'
  plane_ind: 'list[int]'
  reserved_at_38: 'list[int]'
  prio_tc: 'list[int]'
  counter_set: 'union_mlx5_ifc_eth_cntrs_grp_data_layout_auto_bits'
struct_mlx5_ifc_ppcnt_reg_bits.register_fields([('swid', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('pnat', (ctypes.c_ubyte * 2), 16), ('reserved_at_12', (ctypes.c_ubyte * 8), 18), ('grp', (ctypes.c_ubyte * 6), 26), ('clr', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 19), 33), ('plane_ind', (ctypes.c_ubyte * 4), 52), ('reserved_at_38', (ctypes.c_ubyte * 3), 56), ('prio_tc', (ctypes.c_ubyte * 5), 59), ('counter_set', union_mlx5_ifc_eth_cntrs_grp_data_layout_auto_bits, 64)])
@c.record
class struct_mlx5_ifc_mpein_reg_bits(c.Struct):
  SIZE = 384
  reserved_at_0: 'list[int]'
  depth: 'list[int]'
  pcie_index: 'list[int]'
  node: 'list[int]'
  reserved_at_18: 'list[int]'
  capability_mask: 'list[int]'
  reserved_at_40: 'list[int]'
  link_width_enabled: 'list[int]'
  link_speed_enabled: 'list[int]'
  lane0_physical_position: 'list[int]'
  link_width_active: 'list[int]'
  link_speed_active: 'list[int]'
  num_of_pfs: 'list[int]'
  num_of_vfs: 'list[int]'
  bdf0: 'list[int]'
  reserved_at_b0: 'list[int]'
  max_read_request_size: 'list[int]'
  max_payload_size: 'list[int]'
  reserved_at_c8: 'list[int]'
  pwr_status: 'list[int]'
  port_type: 'list[int]'
  reserved_at_d4: 'list[int]'
  lane_reversal: 'list[int]'
  reserved_at_e0: 'list[int]'
  pci_power: 'list[int]'
  reserved_at_100: 'list[int]'
  device_status: 'list[int]'
  port_state: 'list[int]'
  reserved_at_138: 'list[int]'
  reserved_at_140: 'list[int]'
  receiver_detect_result: 'list[int]'
  reserved_at_160: 'list[int]'
struct_mlx5_ifc_mpein_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 2), 0), ('depth', (ctypes.c_ubyte * 6), 2), ('pcie_index', (ctypes.c_ubyte * 8), 8), ('node', (ctypes.c_ubyte * 8), 16), ('reserved_at_18', (ctypes.c_ubyte * 8), 24), ('capability_mask', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('link_width_enabled', (ctypes.c_ubyte * 8), 72), ('link_speed_enabled', (ctypes.c_ubyte * 16), 80), ('lane0_physical_position', (ctypes.c_ubyte * 8), 96), ('link_width_active', (ctypes.c_ubyte * 8), 104), ('link_speed_active', (ctypes.c_ubyte * 16), 112), ('num_of_pfs', (ctypes.c_ubyte * 16), 128), ('num_of_vfs', (ctypes.c_ubyte * 16), 144), ('bdf0', (ctypes.c_ubyte * 16), 160), ('reserved_at_b0', (ctypes.c_ubyte * 16), 176), ('max_read_request_size', (ctypes.c_ubyte * 4), 192), ('max_payload_size', (ctypes.c_ubyte * 4), 196), ('reserved_at_c8', (ctypes.c_ubyte * 5), 200), ('pwr_status', (ctypes.c_ubyte * 3), 205), ('port_type', (ctypes.c_ubyte * 4), 208), ('reserved_at_d4', (ctypes.c_ubyte * 11), 212), ('lane_reversal', (ctypes.c_ubyte * 1), 223), ('reserved_at_e0', (ctypes.c_ubyte * 20), 224), ('pci_power', (ctypes.c_ubyte * 12), 244), ('reserved_at_100', (ctypes.c_ubyte * 32), 256), ('device_status', (ctypes.c_ubyte * 16), 288), ('port_state', (ctypes.c_ubyte * 8), 304), ('reserved_at_138', (ctypes.c_ubyte * 8), 312), ('reserved_at_140', (ctypes.c_ubyte * 16), 320), ('receiver_detect_result', (ctypes.c_ubyte * 16), 336), ('reserved_at_160', (ctypes.c_ubyte * 32), 352)])
@c.record
class struct_mlx5_ifc_mpcnt_reg_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  pcie_index: 'list[int]'
  reserved_at_10: 'list[int]'
  grp: 'list[int]'
  clr: 'list[int]'
  reserved_at_21: 'list[int]'
  counter_set: 'union_mlx5_ifc_pcie_cntrs_grp_data_layout_auto_bits'
struct_mlx5_ifc_mpcnt_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('pcie_index', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 10), 16), ('grp', (ctypes.c_ubyte * 6), 26), ('clr', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 31), 33), ('counter_set', union_mlx5_ifc_pcie_cntrs_grp_data_layout_auto_bits, 64)])
@c.record
class struct_mlx5_ifc_ppad_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  single_mac: 'list[int]'
  reserved_at_4: 'list[int]'
  local_port: 'list[int]'
  mac_47_32: 'list[int]'
  mac_31_0: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_ppad_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 3), 0), ('single_mac', (ctypes.c_ubyte * 1), 3), ('reserved_at_4', (ctypes.c_ubyte * 4), 4), ('local_port', (ctypes.c_ubyte * 8), 8), ('mac_47_32', (ctypes.c_ubyte * 16), 16), ('mac_31_0', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_pmtu_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  max_mtu: 'list[int]'
  reserved_at_30: 'list[int]'
  admin_mtu: 'list[int]'
  reserved_at_50: 'list[int]'
  oper_mtu: 'list[int]'
  reserved_at_70: 'list[int]'
struct_mlx5_ifc_pmtu_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('max_mtu', (ctypes.c_ubyte * 16), 32), ('reserved_at_30', (ctypes.c_ubyte * 16), 48), ('admin_mtu', (ctypes.c_ubyte * 16), 64), ('reserved_at_50', (ctypes.c_ubyte * 16), 80), ('oper_mtu', (ctypes.c_ubyte * 16), 96), ('reserved_at_70', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_pmpr_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  module: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  attenuation_5g: 'list[int]'
  reserved_at_40: 'list[int]'
  attenuation_7g: 'list[int]'
  reserved_at_60: 'list[int]'
  attenuation_12g: 'list[int]'
struct_mlx5_ifc_pmpr_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('module', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 24), 32), ('attenuation_5g', (ctypes.c_ubyte * 8), 56), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('attenuation_7g', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 24), 96), ('attenuation_12g', (ctypes.c_ubyte * 8), 120)])
@c.record
class struct_mlx5_ifc_pmpe_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  module: 'list[int]'
  reserved_at_10: 'list[int]'
  module_status: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_pmpe_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('module', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 12), 16), ('module_status', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 96), 32)])
@c.record
class struct_mlx5_ifc_pmpc_reg_bits(c.Struct):
  SIZE = 256
  module_state_updated: 'list[list[int]]'
struct_mlx5_ifc_pmpc_reg_bits.register_fields([('module_state_updated', ((ctypes.c_ubyte * 8) * 32), 0)])
@c.record
class struct_mlx5_ifc_pmlpn_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  mlpn_status: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  e: 'list[int]'
  reserved_at_21: 'list[int]'
struct_mlx5_ifc_pmlpn_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('mlpn_status', (ctypes.c_ubyte * 4), 4), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('e', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 31), 33)])
@c.record
class struct_mlx5_ifc_pmlp_reg_bits(c.Struct):
  SIZE = 512
  rxtx: 'list[int]'
  reserved_at_1: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  width: 'list[int]'
  lane0_module_mapping: 'list[int]'
  lane1_module_mapping: 'list[int]'
  lane2_module_mapping: 'list[int]'
  lane3_module_mapping: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_pmlp_reg_bits.register_fields([('rxtx', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 7), 1), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('width', (ctypes.c_ubyte * 8), 24), ('lane0_module_mapping', (ctypes.c_ubyte * 32), 32), ('lane1_module_mapping', (ctypes.c_ubyte * 32), 64), ('lane2_module_mapping', (ctypes.c_ubyte * 32), 96), ('lane3_module_mapping', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 352), 160)])
@c.record
class struct_mlx5_ifc_pmaos_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  module: 'list[int]'
  reserved_at_10: 'list[int]'
  admin_status: 'list[int]'
  reserved_at_18: 'list[int]'
  oper_status: 'list[int]'
  ase: 'list[int]'
  ee: 'list[int]'
  reserved_at_22: 'list[int]'
  e: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_pmaos_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('module', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 4), 16), ('admin_status', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 4), 24), ('oper_status', (ctypes.c_ubyte * 4), 28), ('ase', (ctypes.c_ubyte * 1), 32), ('ee', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 28), 34), ('e', (ctypes.c_ubyte * 2), 62), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_plpc_reg_bits(c.Struct):
  SIZE = 320
  reserved_at_0: 'list[int]'
  profile_id: 'list[int]'
  reserved_at_10: 'list[int]'
  proto_mask: 'list[int]'
  reserved_at_18: 'list[int]'
  reserved_at_20: 'list[int]'
  lane_speed: 'list[int]'
  reserved_at_40: 'list[int]'
  lpbf: 'list[int]'
  fec_mode_policy: 'list[int]'
  retransmission_capability: 'list[int]'
  fec_mode_capability: 'list[int]'
  retransmission_support_admin: 'list[int]'
  fec_mode_support_admin: 'list[int]'
  retransmission_request_admin: 'list[int]'
  fec_mode_request_admin: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_plpc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('profile_id', (ctypes.c_ubyte * 12), 4), ('reserved_at_10', (ctypes.c_ubyte * 4), 16), ('proto_mask', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('lane_speed', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 23), 64), ('lpbf', (ctypes.c_ubyte * 1), 87), ('fec_mode_policy', (ctypes.c_ubyte * 8), 88), ('retransmission_capability', (ctypes.c_ubyte * 8), 96), ('fec_mode_capability', (ctypes.c_ubyte * 24), 104), ('retransmission_support_admin', (ctypes.c_ubyte * 8), 128), ('fec_mode_support_admin', (ctypes.c_ubyte * 24), 136), ('retransmission_request_admin', (ctypes.c_ubyte * 8), 160), ('fec_mode_request_admin', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 128), 192)])
@c.record
class struct_mlx5_ifc_plib_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  ib_port: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_plib_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('ib_port', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 96), 32)])
@c.record
class struct_mlx5_ifc_plbf_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  lbf_mode: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_plbf_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 13), 16), ('lbf_mode', (ctypes.c_ubyte * 3), 29), ('reserved_at_20', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_pipg_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  dic: 'list[int]'
  reserved_at_21: 'list[int]'
  ipg: 'list[int]'
  reserved_at_3e: 'list[int]'
struct_mlx5_ifc_pipg_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('dic', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 25), 33), ('ipg', (ctypes.c_ubyte * 4), 58), ('reserved_at_3e', (ctypes.c_ubyte * 2), 62)])
@c.record
class struct_mlx5_ifc_pifr_reg_bits(c.Struct):
  SIZE = 768
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  port_filter: 'list[list[int]]'
  port_filter_update_en: 'list[list[int]]'
struct_mlx5_ifc_pifr_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 224), 32), ('port_filter', ((ctypes.c_ubyte * 32) * 8), 256), ('port_filter_update_en', ((ctypes.c_ubyte * 32) * 8), 512)])
_anonenum112: dict[int, str] = {(MLX5_BUF_OWNERSHIP_UNKNOWN:=0): 'MLX5_BUF_OWNERSHIP_UNKNOWN', (MLX5_BUF_OWNERSHIP_FW_OWNED:=1): 'MLX5_BUF_OWNERSHIP_FW_OWNED', (MLX5_BUF_OWNERSHIP_SW_OWNED:=2): 'MLX5_BUF_OWNERSHIP_SW_OWNED'}
@c.record
class struct_mlx5_ifc_pfcc_reg_bits(c.Struct):
  SIZE = 256
  reserved_at_0: 'list[int]'
  buf_ownership: 'list[int]'
  reserved_at_6: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  cable_length_mask: 'list[int]'
  ppan_mask_n: 'list[int]'
  minor_stall_mask: 'list[int]'
  critical_stall_mask: 'list[int]'
  reserved_at_1e: 'list[int]'
  ppan: 'list[int]'
  reserved_at_24: 'list[int]'
  prio_mask_tx: 'list[int]'
  reserved_at_30: 'list[int]'
  prio_mask_rx: 'list[int]'
  pptx: 'list[int]'
  aptx: 'list[int]'
  pptx_mask_n: 'list[int]'
  reserved_at_43: 'list[int]'
  pfctx: 'list[int]'
  reserved_at_50: 'list[int]'
  pprx: 'list[int]'
  aprx: 'list[int]'
  pprx_mask_n: 'list[int]'
  reserved_at_63: 'list[int]'
  pfcrx: 'list[int]'
  reserved_at_70: 'list[int]'
  device_stall_minor_watermark: 'list[int]'
  device_stall_critical_watermark: 'list[int]'
  reserved_at_a0: 'list[int]'
  cable_length: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_pfcc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('buf_ownership', (ctypes.c_ubyte * 2), 4), ('reserved_at_6', (ctypes.c_ubyte * 2), 6), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 10), 16), ('cable_length_mask', (ctypes.c_ubyte * 1), 26), ('ppan_mask_n', (ctypes.c_ubyte * 1), 27), ('minor_stall_mask', (ctypes.c_ubyte * 1), 28), ('critical_stall_mask', (ctypes.c_ubyte * 1), 29), ('reserved_at_1e', (ctypes.c_ubyte * 2), 30), ('ppan', (ctypes.c_ubyte * 4), 32), ('reserved_at_24', (ctypes.c_ubyte * 4), 36), ('prio_mask_tx', (ctypes.c_ubyte * 8), 40), ('reserved_at_30', (ctypes.c_ubyte * 8), 48), ('prio_mask_rx', (ctypes.c_ubyte * 8), 56), ('pptx', (ctypes.c_ubyte * 1), 64), ('aptx', (ctypes.c_ubyte * 1), 65), ('pptx_mask_n', (ctypes.c_ubyte * 1), 66), ('reserved_at_43', (ctypes.c_ubyte * 5), 67), ('pfctx', (ctypes.c_ubyte * 8), 72), ('reserved_at_50', (ctypes.c_ubyte * 16), 80), ('pprx', (ctypes.c_ubyte * 1), 96), ('aprx', (ctypes.c_ubyte * 1), 97), ('pprx_mask_n', (ctypes.c_ubyte * 1), 98), ('reserved_at_63', (ctypes.c_ubyte * 5), 99), ('pfcrx', (ctypes.c_ubyte * 8), 104), ('reserved_at_70', (ctypes.c_ubyte * 16), 112), ('device_stall_minor_watermark', (ctypes.c_ubyte * 16), 128), ('device_stall_critical_watermark', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 24), 160), ('cable_length', (ctypes.c_ubyte * 8), 184), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192)])
@c.record
class struct_mlx5_ifc_pelc_reg_bits(c.Struct):
  SIZE = 448
  op: 'list[int]'
  reserved_at_4: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  op_admin: 'list[int]'
  op_capability: 'list[int]'
  op_request: 'list[int]'
  op_active: 'list[int]'
  admin: 'list[int]'
  capability: 'list[int]'
  request: 'list[int]'
  active: 'list[int]'
  reserved_at_140: 'list[int]'
struct_mlx5_ifc_pelc_reg_bits.register_fields([('op', (ctypes.c_ubyte * 4), 0), ('reserved_at_4', (ctypes.c_ubyte * 4), 4), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('op_admin', (ctypes.c_ubyte * 8), 32), ('op_capability', (ctypes.c_ubyte * 8), 40), ('op_request', (ctypes.c_ubyte * 8), 48), ('op_active', (ctypes.c_ubyte * 8), 56), ('admin', (ctypes.c_ubyte * 64), 64), ('capability', (ctypes.c_ubyte * 64), 128), ('request', (ctypes.c_ubyte * 64), 192), ('active', (ctypes.c_ubyte * 64), 256), ('reserved_at_140', (ctypes.c_ubyte * 128), 320)])
@c.record
class struct_mlx5_ifc_peir_reg_bits(c.Struct):
  SIZE = 96
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  error_count: 'list[int]'
  reserved_at_30: 'list[int]'
  reserved_at_40: 'list[int]'
  lane: 'list[int]'
  reserved_at_50: 'list[int]'
  error_type: 'list[int]'
struct_mlx5_ifc_peir_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 12), 32), ('error_count', (ctypes.c_ubyte * 4), 44), ('reserved_at_30', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 12), 64), ('lane', (ctypes.c_ubyte * 4), 76), ('reserved_at_50', (ctypes.c_ubyte * 8), 80), ('error_type', (ctypes.c_ubyte * 8), 88)])
@c.record
class struct_mlx5_ifc_mpegc_reg_bits(c.Struct):
  SIZE = 352
  reserved_at_0: 'list[int]'
  field_select: 'list[int]'
  tx_overflow_sense: 'list[int]'
  mark_cqe: 'list[int]'
  mark_cnp: 'list[int]'
  reserved_at_43: 'list[int]'
  tx_lossy_overflow_oper: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_mpegc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 48), 0), ('field_select', (ctypes.c_ubyte * 16), 48), ('tx_overflow_sense', (ctypes.c_ubyte * 1), 64), ('mark_cqe', (ctypes.c_ubyte * 1), 65), ('mark_cnp', (ctypes.c_ubyte * 1), 66), ('reserved_at_43', (ctypes.c_ubyte * 27), 67), ('tx_lossy_overflow_oper', (ctypes.c_ubyte * 2), 94), ('reserved_at_60', (ctypes.c_ubyte * 256), 96)])
@c.record
class struct_mlx5_ifc_mpir_reg_bits(c.Struct):
  SIZE = 128
  sdm: 'list[int]'
  reserved_at_1: 'list[int]'
  host_buses: 'list[int]'
  reserved_at_20: 'list[int]'
  local_port: 'list[int]'
  reserved_at_28: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_mpir_reg_bits.register_fields([('sdm', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 27), 1), ('host_buses', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('local_port', (ctypes.c_ubyte * 8), 64), ('reserved_at_28', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
_anonenum113: dict[int, str] = {(MLX5_MTUTC_FREQ_ADJ_UNITS_PPB:=0): 'MLX5_MTUTC_FREQ_ADJ_UNITS_PPB', (MLX5_MTUTC_FREQ_ADJ_UNITS_SCALED_PPM:=1): 'MLX5_MTUTC_FREQ_ADJ_UNITS_SCALED_PPM'}
_anonenum114: dict[int, str] = {(MLX5_MTUTC_OPERATION_SET_TIME_IMMEDIATE:=1): 'MLX5_MTUTC_OPERATION_SET_TIME_IMMEDIATE', (MLX5_MTUTC_OPERATION_ADJUST_TIME:=2): 'MLX5_MTUTC_OPERATION_ADJUST_TIME', (MLX5_MTUTC_OPERATION_ADJUST_FREQ_UTC:=3): 'MLX5_MTUTC_OPERATION_ADJUST_FREQ_UTC'}
@c.record
class struct_mlx5_ifc_mtutc_reg_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
  freq_adj_units: 'list[int]'
  reserved_at_8: 'list[int]'
  log_max_freq_adjustment: 'list[int]'
  reserved_at_10: 'list[int]'
  operation: 'list[int]'
  freq_adjustment: 'list[int]'
  reserved_at_40: 'list[int]'
  utc_sec: 'list[int]'
  reserved_at_a0: 'list[int]'
  utc_nsec: 'list[int]'
  time_adjustment: 'list[int]'
struct_mlx5_ifc_mtutc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 5), 0), ('freq_adj_units', (ctypes.c_ubyte * 3), 5), ('reserved_at_8', (ctypes.c_ubyte * 3), 8), ('log_max_freq_adjustment', (ctypes.c_ubyte * 5), 11), ('reserved_at_10', (ctypes.c_ubyte * 12), 16), ('operation', (ctypes.c_ubyte * 4), 28), ('freq_adjustment', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('utc_sec', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 2), 160), ('utc_nsec', (ctypes.c_ubyte * 30), 162), ('time_adjustment', (ctypes.c_ubyte * 32), 192)])
@c.record
class struct_mlx5_ifc_pcam_enhanced_features_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  ppcnt_recovery_counters: 'list[int]'
  reserved_at_11: 'list[int]'
  cable_length: 'list[int]'
  reserved_at_19: 'list[int]'
  fec_200G_per_lane_in_pplm: 'list[int]'
  reserved_at_1e: 'list[int]'
  fec_100G_per_lane_in_pplm: 'list[int]'
  reserved_at_49: 'list[int]'
  buffer_ownership: 'list[int]'
  resereved_at_54: 'list[int]'
  fec_50G_per_lane_in_pplm: 'list[int]'
  reserved_at_69: 'list[int]'
  rx_icrc_encapsulated_counter: 'list[int]'
  reserved_at_6e: 'list[int]'
  ptys_extended_ethernet: 'list[int]'
  reserved_at_73: 'list[int]'
  pfcc_mask: 'list[int]'
  reserved_at_77: 'list[int]'
  per_lane_error_counters: 'list[int]'
  rx_buffer_fullness_counters: 'list[int]'
  ptys_connector_type: 'list[int]'
  reserved_at_7d: 'list[int]'
  ppcnt_discard_group: 'list[int]'
  ppcnt_statistical_group: 'list[int]'
struct_mlx5_ifc_pcam_enhanced_features_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('ppcnt_recovery_counters', (ctypes.c_ubyte * 1), 16), ('reserved_at_11', (ctypes.c_ubyte * 7), 17), ('cable_length', (ctypes.c_ubyte * 1), 24), ('reserved_at_19', (ctypes.c_ubyte * 4), 25), ('fec_200G_per_lane_in_pplm', (ctypes.c_ubyte * 1), 29), ('reserved_at_1e', (ctypes.c_ubyte * 42), 30), ('fec_100G_per_lane_in_pplm', (ctypes.c_ubyte * 1), 72), ('reserved_at_49', (ctypes.c_ubyte * 10), 73), ('buffer_ownership', (ctypes.c_ubyte * 1), 83), ('resereved_at_54', (ctypes.c_ubyte * 20), 84), ('fec_50G_per_lane_in_pplm', (ctypes.c_ubyte * 1), 104), ('reserved_at_69', (ctypes.c_ubyte * 4), 105), ('rx_icrc_encapsulated_counter', (ctypes.c_ubyte * 1), 109), ('reserved_at_6e', (ctypes.c_ubyte * 4), 110), ('ptys_extended_ethernet', (ctypes.c_ubyte * 1), 114), ('reserved_at_73', (ctypes.c_ubyte * 3), 115), ('pfcc_mask', (ctypes.c_ubyte * 1), 118), ('reserved_at_77', (ctypes.c_ubyte * 3), 119), ('per_lane_error_counters', (ctypes.c_ubyte * 1), 122), ('rx_buffer_fullness_counters', (ctypes.c_ubyte * 1), 123), ('ptys_connector_type', (ctypes.c_ubyte * 1), 124), ('reserved_at_7d', (ctypes.c_ubyte * 1), 125), ('ppcnt_discard_group', (ctypes.c_ubyte * 1), 126), ('ppcnt_statistical_group', (ctypes.c_ubyte * 1), 127)])
@c.record
class struct_mlx5_ifc_pcam_regs_5000_to_507f_bits(c.Struct):
  SIZE = 128
  port_access_reg_cap_mask_127_to_96: 'list[int]'
  port_access_reg_cap_mask_95_to_64: 'list[int]'
  port_access_reg_cap_mask_63: 'list[int]'
  pphcr: 'list[int]'
  port_access_reg_cap_mask_61_to_36: 'list[int]'
  pplm: 'list[int]'
  port_access_reg_cap_mask_34_to_32: 'list[int]'
  port_access_reg_cap_mask_31_to_13: 'list[int]'
  pbmc: 'list[int]'
  pptb: 'list[int]'
  port_access_reg_cap_mask_10_to_09: 'list[int]'
  ppcnt: 'list[int]'
  port_access_reg_cap_mask_07_to_00: 'list[int]'
struct_mlx5_ifc_pcam_regs_5000_to_507f_bits.register_fields([('port_access_reg_cap_mask_127_to_96', (ctypes.c_ubyte * 32), 0), ('port_access_reg_cap_mask_95_to_64', (ctypes.c_ubyte * 32), 32), ('port_access_reg_cap_mask_63', (ctypes.c_ubyte * 1), 64), ('pphcr', (ctypes.c_ubyte * 1), 65), ('port_access_reg_cap_mask_61_to_36', (ctypes.c_ubyte * 26), 66), ('pplm', (ctypes.c_ubyte * 1), 92), ('port_access_reg_cap_mask_34_to_32', (ctypes.c_ubyte * 3), 93), ('port_access_reg_cap_mask_31_to_13', (ctypes.c_ubyte * 19), 96), ('pbmc', (ctypes.c_ubyte * 1), 115), ('pptb', (ctypes.c_ubyte * 1), 116), ('port_access_reg_cap_mask_10_to_09', (ctypes.c_ubyte * 2), 117), ('ppcnt', (ctypes.c_ubyte * 1), 119), ('port_access_reg_cap_mask_07_to_00', (ctypes.c_ubyte * 8), 120)])
@c.record
class struct_mlx5_ifc_pcam_reg_bits(c.Struct):
  SIZE = 640
  reserved_at_0: 'list[int]'
  feature_group: 'list[int]'
  reserved_at_10: 'list[int]'
  access_reg_group: 'list[int]'
  reserved_at_20: 'list[int]'
  port_access_reg_cap_mask: 'struct_mlx5_ifc_pcam_reg_bits_port_access_reg_cap_mask'
  reserved_at_c0: 'list[int]'
  feature_cap_mask: 'struct_mlx5_ifc_pcam_reg_bits_feature_cap_mask'
  reserved_at_1c0: 'list[int]'
@c.record
class struct_mlx5_ifc_pcam_reg_bits_port_access_reg_cap_mask(c.Struct):
  SIZE = 128
  regs_5000_to_507f: 'struct_mlx5_ifc_pcam_regs_5000_to_507f_bits'
  reserved_at_0: 'list[int]'
struct_mlx5_ifc_pcam_reg_bits_port_access_reg_cap_mask.register_fields([('regs_5000_to_507f', struct_mlx5_ifc_pcam_regs_5000_to_507f_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 128), 0)])
@c.record
class struct_mlx5_ifc_pcam_reg_bits_feature_cap_mask(c.Struct):
  SIZE = 128
  enhanced_features: 'struct_mlx5_ifc_pcam_enhanced_features_bits'
  reserved_at_0: 'list[int]'
struct_mlx5_ifc_pcam_reg_bits_feature_cap_mask.register_fields([('enhanced_features', struct_mlx5_ifc_pcam_enhanced_features_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 128), 0)])
struct_mlx5_ifc_pcam_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('feature_group', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('access_reg_group', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('port_access_reg_cap_mask', struct_mlx5_ifc_pcam_reg_bits_port_access_reg_cap_mask, 64), ('reserved_at_c0', (ctypes.c_ubyte * 128), 192), ('feature_cap_mask', struct_mlx5_ifc_pcam_reg_bits_feature_cap_mask, 320), ('reserved_at_1c0', (ctypes.c_ubyte * 192), 448)])
@c.record
class struct_mlx5_ifc_mcam_enhanced_features_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  mtutc_freq_adj_units: 'list[int]'
  mtutc_time_adjustment_extended_range: 'list[int]'
  reserved_at_52: 'list[int]'
  mcia_32dwords: 'list[int]'
  out_pulse_duration_ns: 'list[int]'
  npps_period: 'list[int]'
  reserved_at_60: 'list[int]'
  reset_state: 'list[int]'
  ptpcyc2realtime_modify: 'list[int]'
  reserved_at_6c: 'list[int]'
  pci_status_and_power: 'list[int]'
  reserved_at_6f: 'list[int]'
  mark_tx_action_cnp: 'list[int]'
  mark_tx_action_cqe: 'list[int]'
  dynamic_tx_overflow: 'list[int]'
  reserved_at_77: 'list[int]'
  pcie_outbound_stalled: 'list[int]'
  tx_overflow_buffer_pkt: 'list[int]'
  mtpps_enh_out_per_adj: 'list[int]'
  mtpps_fs: 'list[int]'
  pcie_performance_group: 'list[int]'
struct_mlx5_ifc_mcam_enhanced_features_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 80), 0), ('mtutc_freq_adj_units', (ctypes.c_ubyte * 1), 80), ('mtutc_time_adjustment_extended_range', (ctypes.c_ubyte * 1), 81), ('reserved_at_52', (ctypes.c_ubyte * 11), 82), ('mcia_32dwords', (ctypes.c_ubyte * 1), 93), ('out_pulse_duration_ns', (ctypes.c_ubyte * 1), 94), ('npps_period', (ctypes.c_ubyte * 1), 95), ('reserved_at_60', (ctypes.c_ubyte * 10), 96), ('reset_state', (ctypes.c_ubyte * 1), 106), ('ptpcyc2realtime_modify', (ctypes.c_ubyte * 1), 107), ('reserved_at_6c', (ctypes.c_ubyte * 2), 108), ('pci_status_and_power', (ctypes.c_ubyte * 1), 110), ('reserved_at_6f', (ctypes.c_ubyte * 5), 111), ('mark_tx_action_cnp', (ctypes.c_ubyte * 1), 116), ('mark_tx_action_cqe', (ctypes.c_ubyte * 1), 117), ('dynamic_tx_overflow', (ctypes.c_ubyte * 1), 118), ('reserved_at_77', (ctypes.c_ubyte * 4), 119), ('pcie_outbound_stalled', (ctypes.c_ubyte * 1), 123), ('tx_overflow_buffer_pkt', (ctypes.c_ubyte * 1), 124), ('mtpps_enh_out_per_adj', (ctypes.c_ubyte * 1), 125), ('mtpps_fs', (ctypes.c_ubyte * 1), 126), ('pcie_performance_group', (ctypes.c_ubyte * 1), 127)])
@c.record
class struct_mlx5_ifc_mcam_access_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  mcda: 'list[int]'
  mcc: 'list[int]'
  mcqi: 'list[int]'
  mcqs: 'list[int]'
  regs_95_to_90: 'list[int]'
  mpir: 'list[int]'
  regs_88_to_87: 'list[int]'
  mpegc: 'list[int]'
  mtutc: 'list[int]'
  regs_84_to_68: 'list[int]'
  tracer_registers: 'list[int]'
  regs_63_to_46: 'list[int]'
  mrtc: 'list[int]'
  regs_44_to_41: 'list[int]'
  mfrl: 'list[int]'
  regs_39_to_32: 'list[int]'
  regs_31_to_11: 'list[int]'
  mtmp: 'list[int]'
  regs_9_to_0: 'list[int]'
struct_mlx5_ifc_mcam_access_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 28), 0), ('mcda', (ctypes.c_ubyte * 1), 28), ('mcc', (ctypes.c_ubyte * 1), 29), ('mcqi', (ctypes.c_ubyte * 1), 30), ('mcqs', (ctypes.c_ubyte * 1), 31), ('regs_95_to_90', (ctypes.c_ubyte * 6), 32), ('mpir', (ctypes.c_ubyte * 1), 38), ('regs_88_to_87', (ctypes.c_ubyte * 2), 39), ('mpegc', (ctypes.c_ubyte * 1), 41), ('mtutc', (ctypes.c_ubyte * 1), 42), ('regs_84_to_68', (ctypes.c_ubyte * 17), 43), ('tracer_registers', (ctypes.c_ubyte * 4), 60), ('regs_63_to_46', (ctypes.c_ubyte * 18), 64), ('mrtc', (ctypes.c_ubyte * 1), 82), ('regs_44_to_41', (ctypes.c_ubyte * 4), 83), ('mfrl', (ctypes.c_ubyte * 1), 87), ('regs_39_to_32', (ctypes.c_ubyte * 8), 88), ('regs_31_to_11', (ctypes.c_ubyte * 21), 96), ('mtmp', (ctypes.c_ubyte * 1), 117), ('regs_9_to_0', (ctypes.c_ubyte * 10), 118)])
@c.record
class struct_mlx5_ifc_mcam_access_reg_bits1(c.Struct):
  SIZE = 128
  regs_127_to_96: 'list[int]'
  regs_95_to_64: 'list[int]'
  regs_63_to_32: 'list[int]'
  regs_31_to_0: 'list[int]'
struct_mlx5_ifc_mcam_access_reg_bits1.register_fields([('regs_127_to_96', (ctypes.c_ubyte * 32), 0), ('regs_95_to_64', (ctypes.c_ubyte * 32), 32), ('regs_63_to_32', (ctypes.c_ubyte * 32), 64), ('regs_31_to_0', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_mcam_access_reg_bits2(c.Struct):
  SIZE = 128
  regs_127_to_99: 'list[int]'
  mirc: 'list[int]'
  regs_97_to_96: 'list[int]'
  regs_95_to_87: 'list[int]'
  synce_registers: 'list[int]'
  regs_84_to_64: 'list[int]'
  regs_63_to_32: 'list[int]'
  regs_31_to_0: 'list[int]'
struct_mlx5_ifc_mcam_access_reg_bits2.register_fields([('regs_127_to_99', (ctypes.c_ubyte * 29), 0), ('mirc', (ctypes.c_ubyte * 1), 29), ('regs_97_to_96', (ctypes.c_ubyte * 2), 30), ('regs_95_to_87', (ctypes.c_ubyte * 9), 32), ('synce_registers', (ctypes.c_ubyte * 2), 41), ('regs_84_to_64', (ctypes.c_ubyte * 21), 43), ('regs_63_to_32', (ctypes.c_ubyte * 32), 64), ('regs_31_to_0', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_mcam_access_reg_bits3(c.Struct):
  SIZE = 128
  regs_127_to_96: 'list[int]'
  regs_95_to_64: 'list[int]'
  regs_63_to_32: 'list[int]'
  regs_31_to_3: 'list[int]'
  mrtcq: 'list[int]'
  mtctr: 'list[int]'
  mtptm: 'list[int]'
struct_mlx5_ifc_mcam_access_reg_bits3.register_fields([('regs_127_to_96', (ctypes.c_ubyte * 32), 0), ('regs_95_to_64', (ctypes.c_ubyte * 32), 32), ('regs_63_to_32', (ctypes.c_ubyte * 32), 64), ('regs_31_to_3', (ctypes.c_ubyte * 29), 96), ('mrtcq', (ctypes.c_ubyte * 1), 125), ('mtctr', (ctypes.c_ubyte * 1), 126), ('mtptm', (ctypes.c_ubyte * 1), 127)])
@c.record
class struct_mlx5_ifc_mcam_reg_bits(c.Struct):
  SIZE = 576
  reserved_at_0: 'list[int]'
  feature_group: 'list[int]'
  reserved_at_10: 'list[int]'
  access_reg_group: 'list[int]'
  reserved_at_20: 'list[int]'
  mng_access_reg_cap_mask: 'struct_mlx5_ifc_mcam_reg_bits_mng_access_reg_cap_mask'
  reserved_at_c0: 'list[int]'
  mng_feature_cap_mask: 'struct_mlx5_ifc_mcam_reg_bits_mng_feature_cap_mask'
  reserved_at_1c0: 'list[int]'
@c.record
class struct_mlx5_ifc_mcam_reg_bits_mng_access_reg_cap_mask(c.Struct):
  SIZE = 128
  access_regs: 'struct_mlx5_ifc_mcam_access_reg_bits'
  access_regs1: 'struct_mlx5_ifc_mcam_access_reg_bits1'
  access_regs2: 'struct_mlx5_ifc_mcam_access_reg_bits2'
  access_regs3: 'struct_mlx5_ifc_mcam_access_reg_bits3'
  reserved_at_0: 'list[int]'
struct_mlx5_ifc_mcam_reg_bits_mng_access_reg_cap_mask.register_fields([('access_regs', struct_mlx5_ifc_mcam_access_reg_bits, 0), ('access_regs1', struct_mlx5_ifc_mcam_access_reg_bits1, 0), ('access_regs2', struct_mlx5_ifc_mcam_access_reg_bits2, 0), ('access_regs3', struct_mlx5_ifc_mcam_access_reg_bits3, 0), ('reserved_at_0', (ctypes.c_ubyte * 128), 0)])
@c.record
class struct_mlx5_ifc_mcam_reg_bits_mng_feature_cap_mask(c.Struct):
  SIZE = 128
  enhanced_features: 'struct_mlx5_ifc_mcam_enhanced_features_bits'
  reserved_at_0: 'list[int]'
struct_mlx5_ifc_mcam_reg_bits_mng_feature_cap_mask.register_fields([('enhanced_features', struct_mlx5_ifc_mcam_enhanced_features_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 128), 0)])
struct_mlx5_ifc_mcam_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('feature_group', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('access_reg_group', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('mng_access_reg_cap_mask', struct_mlx5_ifc_mcam_reg_bits_mng_access_reg_cap_mask, 64), ('reserved_at_c0', (ctypes.c_ubyte * 128), 192), ('mng_feature_cap_mask', struct_mlx5_ifc_mcam_reg_bits_mng_feature_cap_mask, 320), ('reserved_at_1c0', (ctypes.c_ubyte * 128), 448)])
@c.record
class struct_mlx5_ifc_qcam_access_reg_cap_mask(c.Struct):
  SIZE = 128
  qcam_access_reg_cap_mask_127_to_20: 'list[int]'
  qpdpm: 'list[int]'
  qcam_access_reg_cap_mask_18_to_4: 'list[int]'
  qdpm: 'list[int]'
  qpts: 'list[int]'
  qcap: 'list[int]'
  qcam_access_reg_cap_mask_0: 'list[int]'
struct_mlx5_ifc_qcam_access_reg_cap_mask.register_fields([('qcam_access_reg_cap_mask_127_to_20', (ctypes.c_ubyte * 108), 0), ('qpdpm', (ctypes.c_ubyte * 1), 108), ('qcam_access_reg_cap_mask_18_to_4', (ctypes.c_ubyte * 15), 109), ('qdpm', (ctypes.c_ubyte * 1), 124), ('qpts', (ctypes.c_ubyte * 1), 125), ('qcap', (ctypes.c_ubyte * 1), 126), ('qcam_access_reg_cap_mask_0', (ctypes.c_ubyte * 1), 127)])
@c.record
class struct_mlx5_ifc_qcam_qos_feature_cap_mask(c.Struct):
  SIZE = 128
  qcam_qos_feature_cap_mask_127_to_1: 'list[int]'
  qpts_trust_both: 'list[int]'
struct_mlx5_ifc_qcam_qos_feature_cap_mask.register_fields([('qcam_qos_feature_cap_mask_127_to_1', (ctypes.c_ubyte * 127), 0), ('qpts_trust_both', (ctypes.c_ubyte * 1), 127)])
@c.record
class struct_mlx5_ifc_qcam_reg_bits(c.Struct):
  SIZE = 576
  reserved_at_0: 'list[int]'
  feature_group: 'list[int]'
  reserved_at_10: 'list[int]'
  access_reg_group: 'list[int]'
  reserved_at_20: 'list[int]'
  qos_access_reg_cap_mask: 'struct_mlx5_ifc_qcam_reg_bits_qos_access_reg_cap_mask'
  reserved_at_c0: 'list[int]'
  qos_feature_cap_mask: 'struct_mlx5_ifc_qcam_reg_bits_qos_feature_cap_mask'
  reserved_at_1c0: 'list[int]'
@c.record
class struct_mlx5_ifc_qcam_reg_bits_qos_access_reg_cap_mask(c.Struct):
  SIZE = 128
  reg_cap: 'struct_mlx5_ifc_qcam_access_reg_cap_mask'
  reserved_at_0: 'list[int]'
struct_mlx5_ifc_qcam_reg_bits_qos_access_reg_cap_mask.register_fields([('reg_cap', struct_mlx5_ifc_qcam_access_reg_cap_mask, 0), ('reserved_at_0', (ctypes.c_ubyte * 128), 0)])
@c.record
class struct_mlx5_ifc_qcam_reg_bits_qos_feature_cap_mask(c.Struct):
  SIZE = 128
  feature_cap: 'struct_mlx5_ifc_qcam_qos_feature_cap_mask'
  reserved_at_0: 'list[int]'
struct_mlx5_ifc_qcam_reg_bits_qos_feature_cap_mask.register_fields([('feature_cap', struct_mlx5_ifc_qcam_qos_feature_cap_mask, 0), ('reserved_at_0', (ctypes.c_ubyte * 128), 0)])
struct_mlx5_ifc_qcam_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('feature_group', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('access_reg_group', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('qos_access_reg_cap_mask', struct_mlx5_ifc_qcam_reg_bits_qos_access_reg_cap_mask, 64), ('reserved_at_c0', (ctypes.c_ubyte * 128), 192), ('qos_feature_cap_mask', struct_mlx5_ifc_qcam_reg_bits_qos_feature_cap_mask, 320), ('reserved_at_1c0', (ctypes.c_ubyte * 128), 448)])
@c.record
class struct_mlx5_ifc_core_dump_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  core_dump_type: 'list[int]'
  reserved_at_20: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
  qpn: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_core_dump_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 24), 0), ('core_dump_type', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 48), 32), ('vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 8), 96), ('qpn', (ctypes.c_ubyte * 24), 104), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
@c.record
class struct_mlx5_ifc_pcap_reg_bits(c.Struct):
  SIZE = 160
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  port_capability_mask: 'list[list[int]]'
struct_mlx5_ifc_pcap_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('port_capability_mask', ((ctypes.c_ubyte * 32) * 4), 32)])
@c.record
class struct_mlx5_ifc_paos_reg_bits(c.Struct):
  SIZE = 128
  swid: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  admin_status: 'list[int]'
  reserved_at_18: 'list[int]'
  oper_status: 'list[int]'
  ase: 'list[int]'
  ee: 'list[int]'
  reserved_at_22: 'list[int]'
  e: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_paos_reg_bits.register_fields([('swid', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 4), 16), ('admin_status', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 4), 24), ('oper_status', (ctypes.c_ubyte * 4), 28), ('ase', (ctypes.c_ubyte * 1), 32), ('ee', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 28), 34), ('e', (ctypes.c_ubyte * 2), 62), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_pamp_reg_bits(c.Struct):
  SIZE = 352
  reserved_at_0: 'list[int]'
  opamp_group: 'list[int]'
  reserved_at_10: 'list[int]'
  opamp_group_type: 'list[int]'
  start_index: 'list[int]'
  reserved_at_30: 'list[int]'
  num_of_indices: 'list[int]'
  index_data: 'list[list[int]]'
struct_mlx5_ifc_pamp_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('opamp_group', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 12), 16), ('opamp_group_type', (ctypes.c_ubyte * 4), 28), ('start_index', (ctypes.c_ubyte * 16), 32), ('reserved_at_30', (ctypes.c_ubyte * 4), 48), ('num_of_indices', (ctypes.c_ubyte * 12), 52), ('index_data', ((ctypes.c_ubyte * 16) * 18), 64)])
@c.record
class struct_mlx5_ifc_pcmr_reg_bits(c.Struct):
  SIZE = 96
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  entropy_force_cap: 'list[int]'
  entropy_calc_cap: 'list[int]'
  entropy_gre_calc_cap: 'list[int]'
  reserved_at_23: 'list[int]'
  rx_ts_over_crc_cap: 'list[int]'
  reserved_at_33: 'list[int]'
  fcs_cap: 'list[int]'
  reserved_at_3f: 'list[int]'
  entropy_force: 'list[int]'
  entropy_calc: 'list[int]'
  entropy_gre_calc: 'list[int]'
  reserved_at_43: 'list[int]'
  rx_ts_over_crc: 'list[int]'
  reserved_at_53: 'list[int]'
  fcs_chk: 'list[int]'
  reserved_at_5f: 'list[int]'
struct_mlx5_ifc_pcmr_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('entropy_force_cap', (ctypes.c_ubyte * 1), 32), ('entropy_calc_cap', (ctypes.c_ubyte * 1), 33), ('entropy_gre_calc_cap', (ctypes.c_ubyte * 1), 34), ('reserved_at_23', (ctypes.c_ubyte * 15), 35), ('rx_ts_over_crc_cap', (ctypes.c_ubyte * 1), 50), ('reserved_at_33', (ctypes.c_ubyte * 11), 51), ('fcs_cap', (ctypes.c_ubyte * 1), 62), ('reserved_at_3f', (ctypes.c_ubyte * 1), 63), ('entropy_force', (ctypes.c_ubyte * 1), 64), ('entropy_calc', (ctypes.c_ubyte * 1), 65), ('entropy_gre_calc', (ctypes.c_ubyte * 1), 66), ('reserved_at_43', (ctypes.c_ubyte * 15), 67), ('rx_ts_over_crc', (ctypes.c_ubyte * 1), 82), ('reserved_at_53', (ctypes.c_ubyte * 11), 83), ('fcs_chk', (ctypes.c_ubyte * 1), 94), ('reserved_at_5f', (ctypes.c_ubyte * 1), 95)])
@c.record
class struct_mlx5_ifc_lane_2_module_mapping_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  rx_lane: 'list[int]'
  reserved_at_8: 'list[int]'
  tx_lane: 'list[int]'
  reserved_at_10: 'list[int]'
  module: 'list[int]'
struct_mlx5_ifc_lane_2_module_mapping_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('rx_lane', (ctypes.c_ubyte * 4), 4), ('reserved_at_8', (ctypes.c_ubyte * 4), 8), ('tx_lane', (ctypes.c_ubyte * 4), 12), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('module', (ctypes.c_ubyte * 8), 24)])
@c.record
class struct_mlx5_ifc_bufferx_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  lossy: 'list[int]'
  epsb: 'list[int]'
  reserved_at_8: 'list[int]'
  size: 'list[int]'
  xoff_threshold: 'list[int]'
  xon_threshold: 'list[int]'
struct_mlx5_ifc_bufferx_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 6), 0), ('lossy', (ctypes.c_ubyte * 1), 6), ('epsb', (ctypes.c_ubyte * 1), 7), ('reserved_at_8', (ctypes.c_ubyte * 8), 8), ('size', (ctypes.c_ubyte * 16), 16), ('xoff_threshold', (ctypes.c_ubyte * 16), 32), ('xon_threshold', (ctypes.c_ubyte * 16), 48)])
@c.record
class struct_mlx5_ifc_set_node_in_bits(c.Struct):
  SIZE = 512
  node_description: 'list[list[int]]'
struct_mlx5_ifc_set_node_in_bits.register_fields([('node_description', ((ctypes.c_ubyte * 8) * 64), 0)])
@c.record
class struct_mlx5_ifc_register_power_settings_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  power_settings_level: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_register_power_settings_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 24), 0), ('power_settings_level', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 96), 32)])
@c.record
class struct_mlx5_ifc_register_host_endianness_bits(c.Struct):
  SIZE = 128
  he: 'list[int]'
  reserved_at_1: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_register_host_endianness_bits.register_fields([('he', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 31), 1), ('reserved_at_20', (ctypes.c_ubyte * 96), 32)])
@c.record
class struct_mlx5_ifc_umr_pointer_desc_argument_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  mkey: 'list[int]'
  addressh_63_32: 'list[int]'
  addressl_31_0: 'list[int]'
struct_mlx5_ifc_umr_pointer_desc_argument_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('mkey', (ctypes.c_ubyte * 32), 32), ('addressh_63_32', (ctypes.c_ubyte * 32), 64), ('addressl_31_0', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_ud_adrs_vector_bits(c.Struct):
  SIZE = 384
  dc_key: 'list[int]'
  ext: 'list[int]'
  reserved_at_41: 'list[int]'
  destination_qp_dct: 'list[int]'
  static_rate: 'list[int]'
  sl_eth_prio: 'list[int]'
  fl: 'list[int]'
  mlid: 'list[int]'
  rlid_udp_sport: 'list[int]'
  reserved_at_80: 'list[int]'
  rmac_47_16: 'list[int]'
  rmac_15_0: 'list[int]'
  tclass: 'list[int]'
  hop_limit: 'list[int]'
  reserved_at_e0: 'list[int]'
  grh: 'list[int]'
  reserved_at_e2: 'list[int]'
  src_addr_index: 'list[int]'
  flow_label: 'list[int]'
  rgid_rip: 'list[list[int]]'
struct_mlx5_ifc_ud_adrs_vector_bits.register_fields([('dc_key', (ctypes.c_ubyte * 64), 0), ('ext', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 7), 65), ('destination_qp_dct', (ctypes.c_ubyte * 24), 72), ('static_rate', (ctypes.c_ubyte * 4), 96), ('sl_eth_prio', (ctypes.c_ubyte * 4), 100), ('fl', (ctypes.c_ubyte * 1), 104), ('mlid', (ctypes.c_ubyte * 7), 105), ('rlid_udp_sport', (ctypes.c_ubyte * 16), 112), ('reserved_at_80', (ctypes.c_ubyte * 32), 128), ('rmac_47_16', (ctypes.c_ubyte * 32), 160), ('rmac_15_0', (ctypes.c_ubyte * 16), 192), ('tclass', (ctypes.c_ubyte * 8), 208), ('hop_limit', (ctypes.c_ubyte * 8), 216), ('reserved_at_e0', (ctypes.c_ubyte * 1), 224), ('grh', (ctypes.c_ubyte * 1), 225), ('reserved_at_e2', (ctypes.c_ubyte * 2), 226), ('src_addr_index', (ctypes.c_ubyte * 8), 228), ('flow_label', (ctypes.c_ubyte * 20), 236), ('rgid_rip', ((ctypes.c_ubyte * 8) * 16), 256)])
@c.record
class struct_mlx5_ifc_pages_req_event_bits(c.Struct):
  SIZE = 224
  reserved_at_0: 'list[int]'
  function_id: 'list[int]'
  num_pages: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_pages_req_event_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('function_id', (ctypes.c_ubyte * 16), 16), ('num_pages', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 160), 64)])
@c.record
class struct_mlx5_ifc_eqe_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  event_type: 'list[int]'
  reserved_at_10: 'list[int]'
  event_sub_type: 'list[int]'
  reserved_at_20: 'list[int]'
  event_data: 'union_mlx5_ifc_event_auto_bits'
  reserved_at_1e0: 'list[int]'
  signature: 'list[int]'
  reserved_at_1f8: 'list[int]'
  owner: 'list[int]'
struct_mlx5_ifc_eqe_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('event_type', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('event_sub_type', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 224), 32), ('event_data', union_mlx5_ifc_event_auto_bits, 256), ('reserved_at_1e0', (ctypes.c_ubyte * 16), 480), ('signature', (ctypes.c_ubyte * 8), 496), ('reserved_at_1f8', (ctypes.c_ubyte * 7), 504), ('owner', (ctypes.c_ubyte * 1), 511)])
_anonenum115: dict[int, str] = {(MLX5_CMD_QUEUE_ENTRY_TYPE_PCIE_CMD_IF_TRANSPORT:=7): 'MLX5_CMD_QUEUE_ENTRY_TYPE_PCIE_CMD_IF_TRANSPORT'}
@c.record
class struct_mlx5_ifc_cmd_queue_entry_bits(c.Struct):
  SIZE = 512
  type: 'list[int]'
  reserved_at_8: 'list[int]'
  input_length: 'list[int]'
  input_mailbox_pointer_63_32: 'list[int]'
  input_mailbox_pointer_31_9: 'list[int]'
  reserved_at_77: 'list[int]'
  command_input_inline_data: 'list[list[int]]'
  command_output_inline_data: 'list[list[int]]'
  output_mailbox_pointer_63_32: 'list[int]'
  output_mailbox_pointer_31_9: 'list[int]'
  reserved_at_1b7: 'list[int]'
  output_length: 'list[int]'
  token: 'list[int]'
  signature: 'list[int]'
  reserved_at_1f0: 'list[int]'
  status: 'list[int]'
  ownership: 'list[int]'
struct_mlx5_ifc_cmd_queue_entry_bits.register_fields([('type', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('input_length', (ctypes.c_ubyte * 32), 32), ('input_mailbox_pointer_63_32', (ctypes.c_ubyte * 32), 64), ('input_mailbox_pointer_31_9', (ctypes.c_ubyte * 23), 96), ('reserved_at_77', (ctypes.c_ubyte * 9), 119), ('command_input_inline_data', ((ctypes.c_ubyte * 8) * 16), 128), ('command_output_inline_data', ((ctypes.c_ubyte * 8) * 16), 256), ('output_mailbox_pointer_63_32', (ctypes.c_ubyte * 32), 384), ('output_mailbox_pointer_31_9', (ctypes.c_ubyte * 23), 416), ('reserved_at_1b7', (ctypes.c_ubyte * 9), 439), ('output_length', (ctypes.c_ubyte * 32), 448), ('token', (ctypes.c_ubyte * 8), 480), ('signature', (ctypes.c_ubyte * 8), 488), ('reserved_at_1f0', (ctypes.c_ubyte * 8), 496), ('status', (ctypes.c_ubyte * 7), 504), ('ownership', (ctypes.c_ubyte * 1), 511)])
@c.record
class struct_mlx5_ifc_cmd_out_bits(c.Struct):
  SIZE = 96
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  command_output: 'list[int]'
struct_mlx5_ifc_cmd_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('command_output', (ctypes.c_ubyte * 32), 64)])
@c.record
class struct_mlx5_ifc_cmd_in_bits(c.Struct):
  SIZE = 64
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  command: 'list[list[int]]'
struct_mlx5_ifc_cmd_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('command', ((ctypes.c_ubyte * 32) * 0), 64)])
@c.record
class struct_mlx5_ifc_cmd_if_box_bits(c.Struct):
  SIZE = 4608
  mailbox_data: 'list[list[int]]'
  reserved_at_1000: 'list[int]'
  next_pointer_63_32: 'list[int]'
  next_pointer_31_10: 'list[int]'
  reserved_at_11b6: 'list[int]'
  block_number: 'list[int]'
  reserved_at_11e0: 'list[int]'
  token: 'list[int]'
  ctrl_signature: 'list[int]'
  signature: 'list[int]'
struct_mlx5_ifc_cmd_if_box_bits.register_fields([('mailbox_data', ((ctypes.c_ubyte * 8) * 512), 0), ('reserved_at_1000', (ctypes.c_ubyte * 384), 4096), ('next_pointer_63_32', (ctypes.c_ubyte * 32), 4480), ('next_pointer_31_10', (ctypes.c_ubyte * 22), 4512), ('reserved_at_11b6', (ctypes.c_ubyte * 10), 4534), ('block_number', (ctypes.c_ubyte * 32), 4544), ('reserved_at_11e0', (ctypes.c_ubyte * 8), 4576), ('token', (ctypes.c_ubyte * 8), 4584), ('ctrl_signature', (ctypes.c_ubyte * 8), 4592), ('signature', (ctypes.c_ubyte * 8), 4600)])
@c.record
class struct_mlx5_ifc_mtt_bits(c.Struct):
  SIZE = 64
  ptag_63_32: 'list[int]'
  ptag_31_8: 'list[int]'
  reserved_at_38: 'list[int]'
  wr_en: 'list[int]'
  rd_en: 'list[int]'
struct_mlx5_ifc_mtt_bits.register_fields([('ptag_63_32', (ctypes.c_ubyte * 32), 0), ('ptag_31_8', (ctypes.c_ubyte * 24), 32), ('reserved_at_38', (ctypes.c_ubyte * 6), 56), ('wr_en', (ctypes.c_ubyte * 1), 62), ('rd_en', (ctypes.c_ubyte * 1), 63)])
@c.record
class struct_mlx5_ifc_query_wol_rol_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  rol_mode: 'list[int]'
  wol_mode: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_wol_rol_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('rol_mode', (ctypes.c_ubyte * 8), 80), ('wol_mode', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_wol_rol_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_wol_rol_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_wol_rol_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_wol_rol_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_wol_rol_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  rol_mode_valid: 'list[int]'
  wol_mode_valid: 'list[int]'
  reserved_at_42: 'list[int]'
  rol_mode: 'list[int]'
  wol_mode: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_set_wol_rol_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('rol_mode_valid', (ctypes.c_ubyte * 1), 64), ('wol_mode_valid', (ctypes.c_ubyte * 1), 65), ('reserved_at_42', (ctypes.c_ubyte * 14), 66), ('rol_mode', (ctypes.c_ubyte * 8), 80), ('wol_mode', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
_anonenum116: dict[int, str] = {(MLX5_INITIAL_SEG_NIC_INTERFACE_FULL_DRIVER:=0): 'MLX5_INITIAL_SEG_NIC_INTERFACE_FULL_DRIVER', (MLX5_INITIAL_SEG_NIC_INTERFACE_DISABLED:=1): 'MLX5_INITIAL_SEG_NIC_INTERFACE_DISABLED', (MLX5_INITIAL_SEG_NIC_INTERFACE_NO_DRAM_NIC:=2): 'MLX5_INITIAL_SEG_NIC_INTERFACE_NO_DRAM_NIC', (MLX5_INITIAL_SEG_NIC_INTERFACE_SW_RESET:=7): 'MLX5_INITIAL_SEG_NIC_INTERFACE_SW_RESET'}
_anonenum117: dict[int, str] = {(MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_FULL_DRIVER:=0): 'MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_FULL_DRIVER', (MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_DISABLED:=1): 'MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_DISABLED', (MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_NO_DRAM_NIC:=2): 'MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_NO_DRAM_NIC'}
_anonenum118: dict[int, str] = {(MLX5_INITIAL_SEG_HEALTH_SYNDROME_FW_INTERNAL_ERR:=1): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_FW_INTERNAL_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_DEAD_IRISC:=7): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_DEAD_IRISC', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_HW_FATAL_ERR:=8): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_HW_FATAL_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_FW_CRC_ERR:=9): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_FW_CRC_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_FETCH_PCI_ERR:=10): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_FETCH_PCI_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_PAGE_ERR:=11): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_PAGE_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_ASYNCHRONOUS_EQ_BUF_OVERRUN:=12): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_ASYNCHRONOUS_EQ_BUF_OVERRUN', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_EQ_IN_ERR:=13): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_EQ_IN_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_EQ_INV:=14): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_EQ_INV', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_FFSER_ERR:=15): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_FFSER_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_HIGH_TEMP_ERR:=16): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_HIGH_TEMP_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_PCI_POISONED_ERR:=18): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_PCI_POISONED_ERR', (MLX5_INITIAL_SEG_HEALTH_SYNDROME_TRUST_LOCKDOWN_ERR:=19): 'MLX5_INITIAL_SEG_HEALTH_SYNDROME_TRUST_LOCKDOWN_ERR'}
@c.record
class struct_mlx5_ifc_initial_seg_bits(c.Struct):
  SIZE = 131168
  fw_rev_minor: 'list[int]'
  fw_rev_major: 'list[int]'
  cmd_interface_rev: 'list[int]'
  fw_rev_subminor: 'list[int]'
  reserved_at_40: 'list[int]'
  cmdq_phy_addr_63_32: 'list[int]'
  cmdq_phy_addr_31_12: 'list[int]'
  reserved_at_b4: 'list[int]'
  nic_interface: 'list[int]'
  log_cmdq_size: 'list[int]'
  log_cmdq_stride: 'list[int]'
  command_doorbell_vector: 'list[int]'
  reserved_at_e0: 'list[int]'
  initializing: 'list[int]'
  reserved_at_fe1: 'list[int]'
  nic_interface_supported: 'list[int]'
  embedded_cpu: 'list[int]'
  reserved_at_fe9: 'list[int]'
  health_buffer: 'struct_mlx5_ifc_health_buffer_bits'
  no_dram_nic_offset: 'list[int]'
  reserved_at_1220: 'list[int]'
  reserved_at_8060: 'list[int]'
  clear_int: 'list[int]'
  health_syndrome: 'list[int]'
  health_counter: 'list[int]'
  reserved_at_80a0: 'list[int]'
struct_mlx5_ifc_initial_seg_bits.register_fields([('fw_rev_minor', (ctypes.c_ubyte * 16), 0), ('fw_rev_major', (ctypes.c_ubyte * 16), 16), ('cmd_interface_rev', (ctypes.c_ubyte * 16), 32), ('fw_rev_subminor', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('cmdq_phy_addr_63_32', (ctypes.c_ubyte * 32), 128), ('cmdq_phy_addr_31_12', (ctypes.c_ubyte * 20), 160), ('reserved_at_b4', (ctypes.c_ubyte * 2), 180), ('nic_interface', (ctypes.c_ubyte * 2), 182), ('log_cmdq_size', (ctypes.c_ubyte * 4), 184), ('log_cmdq_stride', (ctypes.c_ubyte * 4), 188), ('command_doorbell_vector', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 3840), 224), ('initializing', (ctypes.c_ubyte * 1), 4064), ('reserved_at_fe1', (ctypes.c_ubyte * 4), 4065), ('nic_interface_supported', (ctypes.c_ubyte * 3), 4069), ('embedded_cpu', (ctypes.c_ubyte * 1), 4072), ('reserved_at_fe9', (ctypes.c_ubyte * 23), 4073), ('health_buffer', struct_mlx5_ifc_health_buffer_bits, 4096), ('no_dram_nic_offset', (ctypes.c_ubyte * 32), 4608), ('reserved_at_1220', (ctypes.c_ubyte * 28224), 4640), ('reserved_at_8060', (ctypes.c_ubyte * 31), 32864), ('clear_int', (ctypes.c_ubyte * 1), 32895), ('health_syndrome', (ctypes.c_ubyte * 8), 32896), ('health_counter', (ctypes.c_ubyte * 24), 32904), ('reserved_at_80a0', (ctypes.c_ubyte * 98240), 32928)])
@c.record
class struct_mlx5_ifc_mtpps_reg_bits(c.Struct):
  SIZE = 480
  reserved_at_0: 'list[int]'
  cap_number_of_pps_pins: 'list[int]'
  reserved_at_10: 'list[int]'
  cap_max_num_of_pps_in_pins: 'list[int]'
  reserved_at_18: 'list[int]'
  cap_max_num_of_pps_out_pins: 'list[int]'
  reserved_at_20: 'list[int]'
  cap_log_min_npps_period: 'list[int]'
  reserved_at_38: 'list[int]'
  cap_log_min_out_pulse_duration_ns: 'list[int]'
  reserved_at_40: 'list[int]'
  cap_pin_3_mode: 'list[int]'
  reserved_at_48: 'list[int]'
  cap_pin_2_mode: 'list[int]'
  reserved_at_50: 'list[int]'
  cap_pin_1_mode: 'list[int]'
  reserved_at_58: 'list[int]'
  cap_pin_0_mode: 'list[int]'
  reserved_at_60: 'list[int]'
  cap_pin_7_mode: 'list[int]'
  reserved_at_68: 'list[int]'
  cap_pin_6_mode: 'list[int]'
  reserved_at_70: 'list[int]'
  cap_pin_5_mode: 'list[int]'
  reserved_at_78: 'list[int]'
  cap_pin_4_mode: 'list[int]'
  field_select: 'list[int]'
  reserved_at_a0: 'list[int]'
  npps_period: 'list[int]'
  enable: 'list[int]'
  reserved_at_101: 'list[int]'
  pattern: 'list[int]'
  reserved_at_110: 'list[int]'
  pin_mode: 'list[int]'
  pin: 'list[int]'
  reserved_at_120: 'list[int]'
  out_pulse_duration_ns: 'list[int]'
  time_stamp: 'list[int]'
  out_pulse_duration: 'list[int]'
  out_periodic_adjustment: 'list[int]'
  enhanced_out_periodic_adjustment: 'list[int]'
  reserved_at_1c0: 'list[int]'
struct_mlx5_ifc_mtpps_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 12), 0), ('cap_number_of_pps_pins', (ctypes.c_ubyte * 4), 12), ('reserved_at_10', (ctypes.c_ubyte * 4), 16), ('cap_max_num_of_pps_in_pins', (ctypes.c_ubyte * 4), 20), ('reserved_at_18', (ctypes.c_ubyte * 4), 24), ('cap_max_num_of_pps_out_pins', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 19), 32), ('cap_log_min_npps_period', (ctypes.c_ubyte * 5), 51), ('reserved_at_38', (ctypes.c_ubyte * 3), 56), ('cap_log_min_out_pulse_duration_ns', (ctypes.c_ubyte * 5), 59), ('reserved_at_40', (ctypes.c_ubyte * 4), 64), ('cap_pin_3_mode', (ctypes.c_ubyte * 4), 68), ('reserved_at_48', (ctypes.c_ubyte * 4), 72), ('cap_pin_2_mode', (ctypes.c_ubyte * 4), 76), ('reserved_at_50', (ctypes.c_ubyte * 4), 80), ('cap_pin_1_mode', (ctypes.c_ubyte * 4), 84), ('reserved_at_58', (ctypes.c_ubyte * 4), 88), ('cap_pin_0_mode', (ctypes.c_ubyte * 4), 92), ('reserved_at_60', (ctypes.c_ubyte * 4), 96), ('cap_pin_7_mode', (ctypes.c_ubyte * 4), 100), ('reserved_at_68', (ctypes.c_ubyte * 4), 104), ('cap_pin_6_mode', (ctypes.c_ubyte * 4), 108), ('reserved_at_70', (ctypes.c_ubyte * 4), 112), ('cap_pin_5_mode', (ctypes.c_ubyte * 4), 116), ('reserved_at_78', (ctypes.c_ubyte * 4), 120), ('cap_pin_4_mode', (ctypes.c_ubyte * 4), 124), ('field_select', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('npps_period', (ctypes.c_ubyte * 64), 192), ('enable', (ctypes.c_ubyte * 1), 256), ('reserved_at_101', (ctypes.c_ubyte * 11), 257), ('pattern', (ctypes.c_ubyte * 4), 268), ('reserved_at_110', (ctypes.c_ubyte * 4), 272), ('pin_mode', (ctypes.c_ubyte * 4), 276), ('pin', (ctypes.c_ubyte * 8), 280), ('reserved_at_120', (ctypes.c_ubyte * 2), 288), ('out_pulse_duration_ns', (ctypes.c_ubyte * 30), 290), ('time_stamp', (ctypes.c_ubyte * 64), 320), ('out_pulse_duration', (ctypes.c_ubyte * 16), 384), ('out_periodic_adjustment', (ctypes.c_ubyte * 16), 400), ('enhanced_out_periodic_adjustment', (ctypes.c_ubyte * 32), 416), ('reserved_at_1c0', (ctypes.c_ubyte * 32), 448)])
@c.record
class struct_mlx5_ifc_mtppse_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  pin: 'list[int]'
  event_arm: 'list[int]'
  reserved_at_21: 'list[int]'
  event_generation_mode: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_mtppse_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 24), 0), ('pin', (ctypes.c_ubyte * 8), 24), ('event_arm', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 27), 33), ('event_generation_mode', (ctypes.c_ubyte * 4), 60), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_mcqs_reg_bits(c.Struct):
  SIZE = 128
  last_index_flag: 'list[int]'
  reserved_at_1: 'list[int]'
  fw_device: 'list[int]'
  component_index: 'list[int]'
  reserved_at_20: 'list[int]'
  identifier: 'list[int]'
  reserved_at_40: 'list[int]'
  component_status: 'list[int]'
  component_update_state: 'list[int]'
  last_update_state_changer_type: 'list[int]'
  last_update_state_changer_host_id: 'list[int]'
  reserved_at_68: 'list[int]'
struct_mlx5_ifc_mcqs_reg_bits.register_fields([('last_index_flag', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 7), 1), ('fw_device', (ctypes.c_ubyte * 8), 8), ('component_index', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('identifier', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 23), 64), ('component_status', (ctypes.c_ubyte * 5), 87), ('component_update_state', (ctypes.c_ubyte * 4), 92), ('last_update_state_changer_type', (ctypes.c_ubyte * 4), 96), ('last_update_state_changer_host_id', (ctypes.c_ubyte * 4), 100), ('reserved_at_68', (ctypes.c_ubyte * 24), 104)])
@c.record
class struct_mlx5_ifc_mcqi_cap_bits(c.Struct):
  SIZE = 160
  supported_info_bitmask: 'list[int]'
  component_size: 'list[int]'
  max_component_size: 'list[int]'
  log_mcda_word_size: 'list[int]'
  reserved_at_64: 'list[int]'
  mcda_max_write_size: 'list[int]'
  rd_en: 'list[int]'
  reserved_at_81: 'list[int]'
  match_chip_id: 'list[int]'
  match_psid: 'list[int]'
  check_user_timestamp: 'list[int]'
  match_base_guid_mac: 'list[int]'
  reserved_at_86: 'list[int]'
struct_mlx5_ifc_mcqi_cap_bits.register_fields([('supported_info_bitmask', (ctypes.c_ubyte * 32), 0), ('component_size', (ctypes.c_ubyte * 32), 32), ('max_component_size', (ctypes.c_ubyte * 32), 64), ('log_mcda_word_size', (ctypes.c_ubyte * 4), 96), ('reserved_at_64', (ctypes.c_ubyte * 12), 100), ('mcda_max_write_size', (ctypes.c_ubyte * 16), 112), ('rd_en', (ctypes.c_ubyte * 1), 128), ('reserved_at_81', (ctypes.c_ubyte * 1), 129), ('match_chip_id', (ctypes.c_ubyte * 1), 130), ('match_psid', (ctypes.c_ubyte * 1), 131), ('check_user_timestamp', (ctypes.c_ubyte * 1), 132), ('match_base_guid_mac', (ctypes.c_ubyte * 1), 133), ('reserved_at_86', (ctypes.c_ubyte * 26), 134)])
@c.record
class struct_mlx5_ifc_mcqi_version_bits(c.Struct):
  SIZE = 992
  reserved_at_0: 'list[int]'
  build_time_valid: 'list[int]'
  user_defined_time_valid: 'list[int]'
  reserved_at_4: 'list[int]'
  version_string_length: 'list[int]'
  version: 'list[int]'
  build_time: 'list[int]'
  user_defined_time: 'list[int]'
  build_tool_version: 'list[int]'
  reserved_at_e0: 'list[int]'
  version_string: 'list[list[int]]'
struct_mlx5_ifc_mcqi_version_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 2), 0), ('build_time_valid', (ctypes.c_ubyte * 1), 2), ('user_defined_time_valid', (ctypes.c_ubyte * 1), 3), ('reserved_at_4', (ctypes.c_ubyte * 20), 4), ('version_string_length', (ctypes.c_ubyte * 8), 24), ('version', (ctypes.c_ubyte * 32), 32), ('build_time', (ctypes.c_ubyte * 64), 64), ('user_defined_time', (ctypes.c_ubyte * 64), 128), ('build_tool_version', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224), ('version_string', ((ctypes.c_ubyte * 8) * 92), 256)])
@c.record
class struct_mlx5_ifc_mcqi_activation_method_bits(c.Struct):
  SIZE = 32
  pending_server_ac_power_cycle: 'list[int]'
  pending_server_dc_power_cycle: 'list[int]'
  pending_server_reboot: 'list[int]'
  pending_fw_reset: 'list[int]'
  auto_activate: 'list[int]'
  all_hosts_sync: 'list[int]'
  device_hw_reset: 'list[int]'
  reserved_at_7: 'list[int]'
struct_mlx5_ifc_mcqi_activation_method_bits.register_fields([('pending_server_ac_power_cycle', (ctypes.c_ubyte * 1), 0), ('pending_server_dc_power_cycle', (ctypes.c_ubyte * 1), 1), ('pending_server_reboot', (ctypes.c_ubyte * 1), 2), ('pending_fw_reset', (ctypes.c_ubyte * 1), 3), ('auto_activate', (ctypes.c_ubyte * 1), 4), ('all_hosts_sync', (ctypes.c_ubyte * 1), 5), ('device_hw_reset', (ctypes.c_ubyte * 1), 6), ('reserved_at_7', (ctypes.c_ubyte * 25), 7)])
@c.record
class union_mlx5_ifc_mcqi_reg_data_bits(c.Struct):
  SIZE = 992
  mcqi_caps: 'struct_mlx5_ifc_mcqi_cap_bits'
  mcqi_version: 'struct_mlx5_ifc_mcqi_version_bits'
  mcqi_activation_mathod: 'struct_mlx5_ifc_mcqi_activation_method_bits'
union_mlx5_ifc_mcqi_reg_data_bits.register_fields([('mcqi_caps', struct_mlx5_ifc_mcqi_cap_bits, 0), ('mcqi_version', struct_mlx5_ifc_mcqi_version_bits, 0), ('mcqi_activation_mathod', struct_mlx5_ifc_mcqi_activation_method_bits, 0)])
@c.record
class struct_mlx5_ifc_mcqi_reg_bits(c.Struct):
  SIZE = 192
  read_pending_component: 'list[int]'
  reserved_at_1: 'list[int]'
  component_index: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_40: 'list[int]'
  info_type: 'list[int]'
  info_size: 'list[int]'
  offset: 'list[int]'
  reserved_at_a0: 'list[int]'
  data_size: 'list[int]'
  data: 'list[union_mlx5_ifc_mcqi_reg_data_bits]'
struct_mlx5_ifc_mcqi_reg_bits.register_fields([('read_pending_component', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 15), 1), ('component_index', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 27), 64), ('info_type', (ctypes.c_ubyte * 5), 91), ('info_size', (ctypes.c_ubyte * 32), 96), ('offset', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 16), 160), ('data_size', (ctypes.c_ubyte * 16), 176), ('data', (union_mlx5_ifc_mcqi_reg_data_bits * 0), 192)])
@c.record
class struct_mlx5_ifc_mcc_reg_bits(c.Struct):
  SIZE = 256
  reserved_at_0: 'list[int]'
  time_elapsed_since_last_cmd: 'list[int]'
  reserved_at_10: 'list[int]'
  instruction: 'list[int]'
  reserved_at_20: 'list[int]'
  component_index: 'list[int]'
  reserved_at_40: 'list[int]'
  update_handle: 'list[int]'
  handle_owner_type: 'list[int]'
  handle_owner_host_id: 'list[int]'
  reserved_at_68: 'list[int]'
  control_progress: 'list[int]'
  error_code: 'list[int]'
  reserved_at_78: 'list[int]'
  control_state: 'list[int]'
  component_size: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_mcc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 4), 0), ('time_elapsed_since_last_cmd', (ctypes.c_ubyte * 12), 4), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('instruction', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('component_index', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('update_handle', (ctypes.c_ubyte * 24), 72), ('handle_owner_type', (ctypes.c_ubyte * 4), 96), ('handle_owner_host_id', (ctypes.c_ubyte * 4), 100), ('reserved_at_68', (ctypes.c_ubyte * 1), 104), ('control_progress', (ctypes.c_ubyte * 7), 105), ('error_code', (ctypes.c_ubyte * 8), 112), ('reserved_at_78', (ctypes.c_ubyte * 4), 120), ('control_state', (ctypes.c_ubyte * 4), 124), ('component_size', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 96), 160)])
@c.record
class struct_mlx5_ifc_mcda_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  update_handle: 'list[int]'
  offset: 'list[int]'
  reserved_at_40: 'list[int]'
  size: 'list[int]'
  reserved_at_60: 'list[int]'
  data: 'list[list[int]]'
struct_mlx5_ifc_mcda_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('update_handle', (ctypes.c_ubyte * 24), 8), ('offset', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('size', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('data', ((ctypes.c_ubyte * 32) * 0), 128)])
_anonenum119: dict[int, str] = {(MLX5_MFRL_REG_PCI_RESET_METHOD_LINK_TOGGLE:=0): 'MLX5_MFRL_REG_PCI_RESET_METHOD_LINK_TOGGLE', (MLX5_MFRL_REG_PCI_RESET_METHOD_HOT_RESET:=1): 'MLX5_MFRL_REG_PCI_RESET_METHOD_HOT_RESET'}
_anonenum120: dict[int, str] = {(MLX5_MFRL_REG_RESET_STATE_IDLE:=0): 'MLX5_MFRL_REG_RESET_STATE_IDLE', (MLX5_MFRL_REG_RESET_STATE_IN_NEGOTIATION:=1): 'MLX5_MFRL_REG_RESET_STATE_IN_NEGOTIATION', (MLX5_MFRL_REG_RESET_STATE_RESET_IN_PROGRESS:=2): 'MLX5_MFRL_REG_RESET_STATE_RESET_IN_PROGRESS', (MLX5_MFRL_REG_RESET_STATE_NEG_TIMEOUT:=3): 'MLX5_MFRL_REG_RESET_STATE_NEG_TIMEOUT', (MLX5_MFRL_REG_RESET_STATE_NACK:=4): 'MLX5_MFRL_REG_RESET_STATE_NACK', (MLX5_MFRL_REG_RESET_STATE_UNLOAD_TIMEOUT:=5): 'MLX5_MFRL_REG_RESET_STATE_UNLOAD_TIMEOUT'}
_anonenum121: dict[int, str] = {(MLX5_MFRL_REG_RESET_TYPE_FULL_CHIP:=0): 'MLX5_MFRL_REG_RESET_TYPE_FULL_CHIP', (MLX5_MFRL_REG_RESET_TYPE_NET_PORT_ALIVE:=1): 'MLX5_MFRL_REG_RESET_TYPE_NET_PORT_ALIVE'}
_anonenum122: dict[int, str] = {(MLX5_MFRL_REG_RESET_LEVEL0:=0): 'MLX5_MFRL_REG_RESET_LEVEL0', (MLX5_MFRL_REG_RESET_LEVEL3:=1): 'MLX5_MFRL_REG_RESET_LEVEL3', (MLX5_MFRL_REG_RESET_LEVEL6:=2): 'MLX5_MFRL_REG_RESET_LEVEL6'}
@c.record
class struct_mlx5_ifc_mfrl_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  reserved_at_20: 'list[int]'
  pci_sync_for_fw_update_start: 'list[int]'
  pci_sync_for_fw_update_resp: 'list[int]'
  rst_type_sel: 'list[int]'
  pci_reset_req_method: 'list[int]'
  reserved_at_2b: 'list[int]'
  reset_state: 'list[int]'
  reset_type: 'list[int]'
  reset_level: 'list[int]'
struct_mlx5_ifc_mfrl_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 2), 32), ('pci_sync_for_fw_update_start', (ctypes.c_ubyte * 1), 34), ('pci_sync_for_fw_update_resp', (ctypes.c_ubyte * 2), 35), ('rst_type_sel', (ctypes.c_ubyte * 3), 37), ('pci_reset_req_method', (ctypes.c_ubyte * 3), 40), ('reserved_at_2b', (ctypes.c_ubyte * 1), 43), ('reset_state', (ctypes.c_ubyte * 4), 44), ('reset_type', (ctypes.c_ubyte * 8), 48), ('reset_level', (ctypes.c_ubyte * 8), 56)])
@c.record
class struct_mlx5_ifc_mirc_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  status_code: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_mirc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 24), 0), ('status_code', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 32), 32)])
@c.record
class struct_mlx5_ifc_pddr_monitor_opcode_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  monitor_opcode: 'list[int]'
struct_mlx5_ifc_pddr_monitor_opcode_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('monitor_opcode', (ctypes.c_ubyte * 16), 16)])
@c.record
class union_mlx5_ifc_pddr_troubleshooting_page_status_opcode_auto_bits(c.Struct):
  SIZE = 32
  pddr_monitor_opcode: 'struct_mlx5_ifc_pddr_monitor_opcode_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_pddr_troubleshooting_page_status_opcode_auto_bits.register_fields([('pddr_monitor_opcode', struct_mlx5_ifc_pddr_monitor_opcode_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 32), 0)])
_anonenum123: dict[int, str] = {(MLX5_PDDR_REG_TRBLSH_GROUP_OPCODE_MONITOR:=0): 'MLX5_PDDR_REG_TRBLSH_GROUP_OPCODE_MONITOR'}
@c.record
class struct_mlx5_ifc_pddr_troubleshooting_page_bits(c.Struct):
  SIZE = 1984
  reserved_at_0: 'list[int]'
  group_opcode: 'list[int]'
  status_opcode: 'union_mlx5_ifc_pddr_troubleshooting_page_status_opcode_auto_bits'
  reserved_at_40: 'list[int]'
  status_message: 'list[list[int]]'
struct_mlx5_ifc_pddr_troubleshooting_page_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('group_opcode', (ctypes.c_ubyte * 16), 16), ('status_opcode', union_mlx5_ifc_pddr_troubleshooting_page_status_opcode_auto_bits, 32), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('status_message', ((ctypes.c_ubyte * 32) * 59), 96)])
@c.record
class union_mlx5_ifc_pddr_reg_page_data_auto_bits(c.Struct):
  SIZE = 1984
  pddr_troubleshooting_page: 'struct_mlx5_ifc_pddr_troubleshooting_page_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_pddr_reg_page_data_auto_bits.register_fields([('pddr_troubleshooting_page', struct_mlx5_ifc_pddr_troubleshooting_page_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 1984), 0)])
_anonenum124: dict[int, str] = {(MLX5_PDDR_REG_PAGE_SELECT_TROUBLESHOOTING_INFO_PAGE:=1): 'MLX5_PDDR_REG_PAGE_SELECT_TROUBLESHOOTING_INFO_PAGE'}
@c.record
class struct_mlx5_ifc_pddr_reg_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  pnat: 'list[int]'
  reserved_at_12: 'list[int]'
  reserved_at_20: 'list[int]'
  page_select: 'list[int]'
  page_data: 'union_mlx5_ifc_pddr_reg_page_data_auto_bits'
struct_mlx5_ifc_pddr_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('pnat', (ctypes.c_ubyte * 2), 16), ('reserved_at_12', (ctypes.c_ubyte * 14), 18), ('reserved_at_20', (ctypes.c_ubyte * 24), 32), ('page_select', (ctypes.c_ubyte * 8), 56), ('page_data', union_mlx5_ifc_pddr_reg_page_data_auto_bits, 64)])
@c.record
class struct_mlx5_ifc_mrtc_reg_bits(c.Struct):
  SIZE = 128
  time_synced: 'list[int]'
  reserved_at_1: 'list[int]'
  reserved_at_20: 'list[int]'
  time_h: 'list[int]'
  time_l: 'list[int]'
struct_mlx5_ifc_mrtc_reg_bits.register_fields([('time_synced', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 31), 1), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('time_h', (ctypes.c_ubyte * 32), 64), ('time_l', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_mtcap_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  sensor_count: 'list[int]'
  reserved_at_20: 'list[int]'
  sensor_map: 'list[int]'
struct_mlx5_ifc_mtcap_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 25), 0), ('sensor_count', (ctypes.c_ubyte * 7), 25), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('sensor_map', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_mtmp_reg_bits(c.Struct):
  SIZE = 256
  reserved_at_0: 'list[int]'
  sensor_index: 'list[int]'
  reserved_at_20: 'list[int]'
  temperature: 'list[int]'
  mte: 'list[int]'
  mtr: 'list[int]'
  reserved_at_42: 'list[int]'
  max_temperature: 'list[int]'
  tee: 'list[int]'
  reserved_at_62: 'list[int]'
  temp_threshold_hi: 'list[int]'
  reserved_at_80: 'list[int]'
  temp_threshold_lo: 'list[int]'
  reserved_at_a0: 'list[int]'
  sensor_name_hi: 'list[int]'
  sensor_name_lo: 'list[int]'
struct_mlx5_ifc_mtmp_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 20), 0), ('sensor_index', (ctypes.c_ubyte * 12), 20), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('temperature', (ctypes.c_ubyte * 16), 48), ('mte', (ctypes.c_ubyte * 1), 64), ('mtr', (ctypes.c_ubyte * 1), 65), ('reserved_at_42', (ctypes.c_ubyte * 14), 66), ('max_temperature', (ctypes.c_ubyte * 16), 80), ('tee', (ctypes.c_ubyte * 2), 96), ('reserved_at_62', (ctypes.c_ubyte * 14), 98), ('temp_threshold_hi', (ctypes.c_ubyte * 16), 112), ('reserved_at_80', (ctypes.c_ubyte * 16), 128), ('temp_threshold_lo', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('sensor_name_hi', (ctypes.c_ubyte * 32), 192), ('sensor_name_lo', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_mtptm_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  psta: 'list[int]'
  reserved_at_11: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_mtptm_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('psta', (ctypes.c_ubyte * 1), 16), ('reserved_at_11', (ctypes.c_ubyte * 15), 17), ('reserved_at_20', (ctypes.c_ubyte * 96), 32)])
_anonenum125: dict[int, str] = {(MLX5_MTCTR_REQUEST_NOP:=0): 'MLX5_MTCTR_REQUEST_NOP', (MLX5_MTCTR_REQUEST_PTM_ROOT_CLOCK:=1): 'MLX5_MTCTR_REQUEST_PTM_ROOT_CLOCK', (MLX5_MTCTR_REQUEST_FREE_RUNNING_COUNTER:=2): 'MLX5_MTCTR_REQUEST_FREE_RUNNING_COUNTER', (MLX5_MTCTR_REQUEST_REAL_TIME_CLOCK:=3): 'MLX5_MTCTR_REQUEST_REAL_TIME_CLOCK'}
@c.record
class struct_mlx5_ifc_mtctr_reg_bits(c.Struct):
  SIZE = 192
  first_clock_timestamp_request: 'list[int]'
  second_clock_timestamp_request: 'list[int]'
  reserved_at_10: 'list[int]'
  first_clock_valid: 'list[int]'
  second_clock_valid: 'list[int]'
  reserved_at_22: 'list[int]'
  first_clock_timestamp: 'list[int]'
  second_clock_timestamp: 'list[int]'
struct_mlx5_ifc_mtctr_reg_bits.register_fields([('first_clock_timestamp_request', (ctypes.c_ubyte * 8), 0), ('second_clock_timestamp_request', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('first_clock_valid', (ctypes.c_ubyte * 1), 32), ('second_clock_valid', (ctypes.c_ubyte * 1), 33), ('reserved_at_22', (ctypes.c_ubyte * 30), 34), ('first_clock_timestamp', (ctypes.c_ubyte * 64), 64), ('second_clock_timestamp', (ctypes.c_ubyte * 64), 128)])
@c.record
class struct_mlx5_ifc_bin_range_layout_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  high_val: 'list[int]'
  reserved_at_10: 'list[int]'
  low_val: 'list[int]'
struct_mlx5_ifc_bin_range_layout_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 10), 0), ('high_val', (ctypes.c_ubyte * 6), 10), ('reserved_at_10', (ctypes.c_ubyte * 10), 16), ('low_val', (ctypes.c_ubyte * 6), 26)])
@c.record
class struct_mlx5_ifc_pphcr_reg_bits(c.Struct):
  SIZE = 640
  active_hist_type: 'list[int]'
  reserved_at_4: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  num_of_bins: 'list[int]'
  reserved_at_30: 'list[int]'
  reserved_at_40: 'list[int]'
  bin_range: 'list[struct_mlx5_ifc_bin_range_layout_bits]'
struct_mlx5_ifc_pphcr_reg_bits.register_fields([('active_hist_type', (ctypes.c_ubyte * 4), 0), ('reserved_at_4', (ctypes.c_ubyte * 4), 4), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('num_of_bins', (ctypes.c_ubyte * 8), 40), ('reserved_at_30', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('bin_range', (struct_mlx5_ifc_bin_range_layout_bits * 16), 128)])
@c.record
class union_mlx5_ifc_ports_control_registers_document_bits(c.Struct):
  SIZE = 24800
  bufferx_reg: 'struct_mlx5_ifc_bufferx_reg_bits'
  eth_2819_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits'
  eth_2863_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits'
  eth_3635_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits'
  eth_802_3_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits'
  eth_extended_cntrs_grp_data_layout: 'struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits'
  eth_per_prio_grp_data_layout: 'struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits'
  eth_per_tc_prio_grp_data_layout: 'struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits'
  eth_per_tc_congest_prio_grp_data_layout: 'struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits'
  lane_2_module_mapping: 'struct_mlx5_ifc_lane_2_module_mapping_bits'
  pamp_reg: 'struct_mlx5_ifc_pamp_reg_bits'
  paos_reg: 'struct_mlx5_ifc_paos_reg_bits'
  pcap_reg: 'struct_mlx5_ifc_pcap_reg_bits'
  pddr_monitor_opcode: 'struct_mlx5_ifc_pddr_monitor_opcode_bits'
  pddr_reg: 'struct_mlx5_ifc_pddr_reg_bits'
  pddr_troubleshooting_page: 'struct_mlx5_ifc_pddr_troubleshooting_page_bits'
  peir_reg: 'struct_mlx5_ifc_peir_reg_bits'
  pelc_reg: 'struct_mlx5_ifc_pelc_reg_bits'
  pfcc_reg: 'struct_mlx5_ifc_pfcc_reg_bits'
  ib_port_cntrs_grp_data_layout: 'struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits'
  phys_layer_cntrs: 'struct_mlx5_ifc_phys_layer_cntrs_bits'
  pifr_reg: 'struct_mlx5_ifc_pifr_reg_bits'
  pipg_reg: 'struct_mlx5_ifc_pipg_reg_bits'
  plbf_reg: 'struct_mlx5_ifc_plbf_reg_bits'
  plib_reg: 'struct_mlx5_ifc_plib_reg_bits'
  plpc_reg: 'struct_mlx5_ifc_plpc_reg_bits'
  pmaos_reg: 'struct_mlx5_ifc_pmaos_reg_bits'
  pmlp_reg: 'struct_mlx5_ifc_pmlp_reg_bits'
  pmlpn_reg: 'struct_mlx5_ifc_pmlpn_reg_bits'
  pmpc_reg: 'struct_mlx5_ifc_pmpc_reg_bits'
  pmpe_reg: 'struct_mlx5_ifc_pmpe_reg_bits'
  pmpr_reg: 'struct_mlx5_ifc_pmpr_reg_bits'
  pmtu_reg: 'struct_mlx5_ifc_pmtu_reg_bits'
  ppad_reg: 'struct_mlx5_ifc_ppad_reg_bits'
  ppcnt_reg: 'struct_mlx5_ifc_ppcnt_reg_bits'
  mpein_reg: 'struct_mlx5_ifc_mpein_reg_bits'
  mpcnt_reg: 'struct_mlx5_ifc_mpcnt_reg_bits'
  pplm_reg: 'struct_mlx5_ifc_pplm_reg_bits'
  pplr_reg: 'struct_mlx5_ifc_pplr_reg_bits'
  ppsc_reg: 'struct_mlx5_ifc_ppsc_reg_bits'
  pqdr_reg: 'struct_mlx5_ifc_pqdr_reg_bits'
  pspa_reg: 'struct_mlx5_ifc_pspa_reg_bits'
  ptas_reg: 'struct_mlx5_ifc_ptas_reg_bits'
  ptys_reg: 'struct_mlx5_ifc_ptys_reg_bits'
  mlcr_reg: 'struct_mlx5_ifc_mlcr_reg_bits'
  pude_reg: 'struct_mlx5_ifc_pude_reg_bits'
  pvlc_reg: 'struct_mlx5_ifc_pvlc_reg_bits'
  slrg_reg: 'struct_mlx5_ifc_slrg_reg_bits'
  sltp_reg: 'struct_mlx5_ifc_sltp_reg_bits'
  mtpps_reg: 'struct_mlx5_ifc_mtpps_reg_bits'
  mtppse_reg: 'struct_mlx5_ifc_mtppse_reg_bits'
  fpga_access_reg: 'struct_mlx5_ifc_fpga_access_reg_bits'
  fpga_ctrl_bits: 'struct_mlx5_ifc_fpga_ctrl_bits'
  fpga_cap_bits: 'struct_mlx5_ifc_fpga_cap_bits'
  mcqi_reg: 'struct_mlx5_ifc_mcqi_reg_bits'
  mcc_reg: 'struct_mlx5_ifc_mcc_reg_bits'
  mcda_reg: 'struct_mlx5_ifc_mcda_reg_bits'
  mirc_reg: 'struct_mlx5_ifc_mirc_reg_bits'
  mfrl_reg: 'struct_mlx5_ifc_mfrl_reg_bits'
  mtutc_reg: 'struct_mlx5_ifc_mtutc_reg_bits'
  mrtc_reg: 'struct_mlx5_ifc_mrtc_reg_bits'
  mtcap_reg: 'struct_mlx5_ifc_mtcap_reg_bits'
  mtmp_reg: 'struct_mlx5_ifc_mtmp_reg_bits'
  mtptm_reg: 'struct_mlx5_ifc_mtptm_reg_bits'
  mtctr_reg: 'struct_mlx5_ifc_mtctr_reg_bits'
  pphcr_reg: 'struct_mlx5_ifc_pphcr_reg_bits'
  reserved_at_0: 'list[int]'
@c.record
class struct_mlx5_ifc_fpga_access_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  reserved_at_20: 'list[int]'
  size: 'list[int]'
  address: 'list[int]'
  data: 'list[list[int]]'
struct_mlx5_ifc_fpga_access_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('size', (ctypes.c_ubyte * 16), 48), ('address', (ctypes.c_ubyte * 64), 64), ('data', ((ctypes.c_ubyte * 8) * 0), 128)])
@c.record
class struct_mlx5_ifc_fpga_ctrl_bits(c.Struct):
  SIZE = 128
  reserved_at_0: 'list[int]'
  operation: 'list[int]'
  reserved_at_10: 'list[int]'
  status: 'list[int]'
  reserved_at_20: 'list[int]'
  flash_select_admin: 'list[int]'
  reserved_at_30: 'list[int]'
  flash_select_oper: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_fpga_ctrl_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('operation', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('status', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('flash_select_admin', (ctypes.c_ubyte * 8), 40), ('reserved_at_30', (ctypes.c_ubyte * 8), 48), ('flash_select_oper', (ctypes.c_ubyte * 8), 56), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
union_mlx5_ifc_ports_control_registers_document_bits.register_fields([('bufferx_reg', struct_mlx5_ifc_bufferx_reg_bits, 0), ('eth_2819_cntrs_grp_data_layout', struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits, 0), ('eth_2863_cntrs_grp_data_layout', struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits, 0), ('eth_3635_cntrs_grp_data_layout', struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits, 0), ('eth_802_3_cntrs_grp_data_layout', struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits, 0), ('eth_extended_cntrs_grp_data_layout', struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits, 0), ('eth_per_prio_grp_data_layout', struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits, 0), ('eth_per_tc_prio_grp_data_layout', struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits, 0), ('eth_per_tc_congest_prio_grp_data_layout', struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits, 0), ('lane_2_module_mapping', struct_mlx5_ifc_lane_2_module_mapping_bits, 0), ('pamp_reg', struct_mlx5_ifc_pamp_reg_bits, 0), ('paos_reg', struct_mlx5_ifc_paos_reg_bits, 0), ('pcap_reg', struct_mlx5_ifc_pcap_reg_bits, 0), ('pddr_monitor_opcode', struct_mlx5_ifc_pddr_monitor_opcode_bits, 0), ('pddr_reg', struct_mlx5_ifc_pddr_reg_bits, 0), ('pddr_troubleshooting_page', struct_mlx5_ifc_pddr_troubleshooting_page_bits, 0), ('peir_reg', struct_mlx5_ifc_peir_reg_bits, 0), ('pelc_reg', struct_mlx5_ifc_pelc_reg_bits, 0), ('pfcc_reg', struct_mlx5_ifc_pfcc_reg_bits, 0), ('ib_port_cntrs_grp_data_layout', struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits, 0), ('phys_layer_cntrs', struct_mlx5_ifc_phys_layer_cntrs_bits, 0), ('pifr_reg', struct_mlx5_ifc_pifr_reg_bits, 0), ('pipg_reg', struct_mlx5_ifc_pipg_reg_bits, 0), ('plbf_reg', struct_mlx5_ifc_plbf_reg_bits, 0), ('plib_reg', struct_mlx5_ifc_plib_reg_bits, 0), ('plpc_reg', struct_mlx5_ifc_plpc_reg_bits, 0), ('pmaos_reg', struct_mlx5_ifc_pmaos_reg_bits, 0), ('pmlp_reg', struct_mlx5_ifc_pmlp_reg_bits, 0), ('pmlpn_reg', struct_mlx5_ifc_pmlpn_reg_bits, 0), ('pmpc_reg', struct_mlx5_ifc_pmpc_reg_bits, 0), ('pmpe_reg', struct_mlx5_ifc_pmpe_reg_bits, 0), ('pmpr_reg', struct_mlx5_ifc_pmpr_reg_bits, 0), ('pmtu_reg', struct_mlx5_ifc_pmtu_reg_bits, 0), ('ppad_reg', struct_mlx5_ifc_ppad_reg_bits, 0), ('ppcnt_reg', struct_mlx5_ifc_ppcnt_reg_bits, 0), ('mpein_reg', struct_mlx5_ifc_mpein_reg_bits, 0), ('mpcnt_reg', struct_mlx5_ifc_mpcnt_reg_bits, 0), ('pplm_reg', struct_mlx5_ifc_pplm_reg_bits, 0), ('pplr_reg', struct_mlx5_ifc_pplr_reg_bits, 0), ('ppsc_reg', struct_mlx5_ifc_ppsc_reg_bits, 0), ('pqdr_reg', struct_mlx5_ifc_pqdr_reg_bits, 0), ('pspa_reg', struct_mlx5_ifc_pspa_reg_bits, 0), ('ptas_reg', struct_mlx5_ifc_ptas_reg_bits, 0), ('ptys_reg', struct_mlx5_ifc_ptys_reg_bits, 0), ('mlcr_reg', struct_mlx5_ifc_mlcr_reg_bits, 0), ('pude_reg', struct_mlx5_ifc_pude_reg_bits, 0), ('pvlc_reg', struct_mlx5_ifc_pvlc_reg_bits, 0), ('slrg_reg', struct_mlx5_ifc_slrg_reg_bits, 0), ('sltp_reg', struct_mlx5_ifc_sltp_reg_bits, 0), ('mtpps_reg', struct_mlx5_ifc_mtpps_reg_bits, 0), ('mtppse_reg', struct_mlx5_ifc_mtppse_reg_bits, 0), ('fpga_access_reg', struct_mlx5_ifc_fpga_access_reg_bits, 0), ('fpga_ctrl_bits', struct_mlx5_ifc_fpga_ctrl_bits, 0), ('fpga_cap_bits', struct_mlx5_ifc_fpga_cap_bits, 0), ('mcqi_reg', struct_mlx5_ifc_mcqi_reg_bits, 0), ('mcc_reg', struct_mlx5_ifc_mcc_reg_bits, 0), ('mcda_reg', struct_mlx5_ifc_mcda_reg_bits, 0), ('mirc_reg', struct_mlx5_ifc_mirc_reg_bits, 0), ('mfrl_reg', struct_mlx5_ifc_mfrl_reg_bits, 0), ('mtutc_reg', struct_mlx5_ifc_mtutc_reg_bits, 0), ('mrtc_reg', struct_mlx5_ifc_mrtc_reg_bits, 0), ('mtcap_reg', struct_mlx5_ifc_mtcap_reg_bits, 0), ('mtmp_reg', struct_mlx5_ifc_mtmp_reg_bits, 0), ('mtptm_reg', struct_mlx5_ifc_mtptm_reg_bits, 0), ('mtctr_reg', struct_mlx5_ifc_mtctr_reg_bits, 0), ('pphcr_reg', struct_mlx5_ifc_pphcr_reg_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 24800), 0)])
@c.record
class union_mlx5_ifc_debug_enhancements_document_bits(c.Struct):
  SIZE = 512
  health_buffer: 'struct_mlx5_ifc_health_buffer_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_debug_enhancements_document_bits.register_fields([('health_buffer', struct_mlx5_ifc_health_buffer_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 512), 0)])
@c.record
class union_mlx5_ifc_uplink_pci_interface_document_bits(c.Struct):
  SIZE = 131168
  initial_seg: 'struct_mlx5_ifc_initial_seg_bits'
  reserved_at_0: 'list[int]'
union_mlx5_ifc_uplink_pci_interface_document_bits.register_fields([('initial_seg', struct_mlx5_ifc_initial_seg_bits, 0), ('reserved_at_0', (ctypes.c_ubyte * 131168), 0)])
@c.record
class struct_mlx5_ifc_set_flow_table_root_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_set_flow_table_root_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_set_flow_table_root_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  table_of_other_vport: 'list[int]'
  table_vport_number: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  reserved_at_c0: 'list[int]'
  underlay_qpn: 'list[int]'
  table_eswitch_owner_vhca_id_valid: 'list[int]'
  reserved_at_e1: 'list[int]'
  table_eswitch_owner_vhca_id: 'list[int]'
  reserved_at_100: 'list[int]'
struct_mlx5_ifc_set_flow_table_root_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 7), 136), ('table_of_other_vport', (ctypes.c_ubyte * 1), 143), ('table_vport_number', (ctypes.c_ubyte * 16), 144), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('underlay_qpn', (ctypes.c_ubyte * 24), 200), ('table_eswitch_owner_vhca_id_valid', (ctypes.c_ubyte * 1), 224), ('reserved_at_e1', (ctypes.c_ubyte * 15), 225), ('table_eswitch_owner_vhca_id', (ctypes.c_ubyte * 16), 240), ('reserved_at_100', (ctypes.c_ubyte * 256), 256)])
_anonenum126: dict[int, str] = {(MLX5_MODIFY_FLOW_TABLE_MISS_TABLE_ID:=1): 'MLX5_MODIFY_FLOW_TABLE_MISS_TABLE_ID', (MLX5_MODIFY_FLOW_TABLE_LAG_NEXT_TABLE_ID:=32768): 'MLX5_MODIFY_FLOW_TABLE_LAG_NEXT_TABLE_ID'}
@c.record
class struct_mlx5_ifc_modify_flow_table_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_flow_table_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_flow_table_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  other_vport: 'list[int]'
  reserved_at_41: 'list[int]'
  vport_number: 'list[int]'
  reserved_at_60: 'list[int]'
  modify_field_select: 'list[int]'
  table_type: 'list[int]'
  reserved_at_88: 'list[int]'
  reserved_at_a0: 'list[int]'
  table_id: 'list[int]'
  flow_table_context: 'struct_mlx5_ifc_flow_table_context_bits'
struct_mlx5_ifc_modify_flow_table_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('other_vport', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 15), 65), ('vport_number', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('modify_field_select', (ctypes.c_ubyte * 16), 112), ('table_type', (ctypes.c_ubyte * 8), 128), ('reserved_at_88', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('table_id', (ctypes.c_ubyte * 24), 168), ('flow_table_context', struct_mlx5_ifc_flow_table_context_bits, 192)])
@c.record
class struct_mlx5_ifc_ets_tcn_config_reg_bits(c.Struct):
  SIZE = 64
  g: 'list[int]'
  b: 'list[int]'
  r: 'list[int]'
  reserved_at_3: 'list[int]'
  group: 'list[int]'
  reserved_at_10: 'list[int]'
  bw_allocation: 'list[int]'
  reserved_at_20: 'list[int]'
  max_bw_units: 'list[int]'
  reserved_at_30: 'list[int]'
  max_bw_value: 'list[int]'
struct_mlx5_ifc_ets_tcn_config_reg_bits.register_fields([('g', (ctypes.c_ubyte * 1), 0), ('b', (ctypes.c_ubyte * 1), 1), ('r', (ctypes.c_ubyte * 1), 2), ('reserved_at_3', (ctypes.c_ubyte * 9), 3), ('group', (ctypes.c_ubyte * 4), 12), ('reserved_at_10', (ctypes.c_ubyte * 9), 16), ('bw_allocation', (ctypes.c_ubyte * 7), 25), ('reserved_at_20', (ctypes.c_ubyte * 12), 32), ('max_bw_units', (ctypes.c_ubyte * 4), 44), ('reserved_at_30', (ctypes.c_ubyte * 8), 48), ('max_bw_value', (ctypes.c_ubyte * 8), 56)])
@c.record
class struct_mlx5_ifc_ets_global_config_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  r: 'list[int]'
  reserved_at_3: 'list[int]'
  reserved_at_20: 'list[int]'
  max_bw_units: 'list[int]'
  reserved_at_30: 'list[int]'
  max_bw_value: 'list[int]'
struct_mlx5_ifc_ets_global_config_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 2), 0), ('r', (ctypes.c_ubyte * 1), 2), ('reserved_at_3', (ctypes.c_ubyte * 29), 3), ('reserved_at_20', (ctypes.c_ubyte * 12), 32), ('max_bw_units', (ctypes.c_ubyte * 4), 44), ('reserved_at_30', (ctypes.c_ubyte * 8), 48), ('max_bw_value', (ctypes.c_ubyte * 8), 56)])
@c.record
class struct_mlx5_ifc_qetc_reg_bits(c.Struct):
  SIZE = 640
  reserved_at_0: 'list[int]'
  port_number: 'list[int]'
  reserved_at_10: 'list[int]'
  tc_configuration: 'list[struct_mlx5_ifc_ets_tcn_config_reg_bits]'
  global_configuration: 'struct_mlx5_ifc_ets_global_config_reg_bits'
struct_mlx5_ifc_qetc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('port_number', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 48), 16), ('tc_configuration', (struct_mlx5_ifc_ets_tcn_config_reg_bits * 8), 64), ('global_configuration', struct_mlx5_ifc_ets_global_config_reg_bits, 576)])
@c.record
class struct_mlx5_ifc_qpdpm_dscp_reg_bits(c.Struct):
  SIZE = 16
  e: 'list[int]'
  reserved_at_01: 'list[int]'
  prio: 'list[int]'
struct_mlx5_ifc_qpdpm_dscp_reg_bits.register_fields([('e', (ctypes.c_ubyte * 1), 0), ('reserved_at_01', (ctypes.c_ubyte * 11), 1), ('prio', (ctypes.c_ubyte * 4), 12)])
@c.record
class struct_mlx5_ifc_qpdpm_reg_bits(c.Struct):
  SIZE = 1056
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  dscp: 'list[struct_mlx5_ifc_qpdpm_dscp_reg_bits]'
struct_mlx5_ifc_qpdpm_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('dscp', (struct_mlx5_ifc_qpdpm_dscp_reg_bits * 64), 32)])
@c.record
class struct_mlx5_ifc_qpts_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  trust_state: 'list[int]'
struct_mlx5_ifc_qpts_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 45), 16), ('trust_state', (ctypes.c_ubyte * 3), 61)])
@c.record
class struct_mlx5_ifc_pptb_reg_bits(c.Struct):
  SIZE = 96
  reserved_at_0: 'list[int]'
  mm: 'list[int]'
  reserved_at_4: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  cm: 'list[int]'
  um: 'list[int]'
  pm: 'list[int]'
  prio_x_buff: 'list[int]'
  pm_msb: 'list[int]'
  reserved_at_48: 'list[int]'
  ctrl_buff: 'list[int]'
  untagged_buff: 'list[int]'
struct_mlx5_ifc_pptb_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 2), 0), ('mm', (ctypes.c_ubyte * 2), 2), ('reserved_at_4', (ctypes.c_ubyte * 4), 4), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 6), 16), ('cm', (ctypes.c_ubyte * 1), 22), ('um', (ctypes.c_ubyte * 1), 23), ('pm', (ctypes.c_ubyte * 8), 24), ('prio_x_buff', (ctypes.c_ubyte * 32), 32), ('pm_msb', (ctypes.c_ubyte * 8), 64), ('reserved_at_48', (ctypes.c_ubyte * 16), 72), ('ctrl_buff', (ctypes.c_ubyte * 4), 88), ('untagged_buff', (ctypes.c_ubyte * 4), 92)])
@c.record
class struct_mlx5_ifc_sbcam_reg_bits(c.Struct):
  SIZE = 608
  reserved_at_0: 'list[int]'
  feature_group: 'list[int]'
  reserved_at_10: 'list[int]'
  access_reg_group: 'list[int]'
  reserved_at_20: 'list[int]'
  sb_access_reg_cap_mask: 'list[list[int]]'
  reserved_at_c0: 'list[int]'
  sb_feature_cap_mask: 'list[list[int]]'
  reserved_at_1c0: 'list[int]'
  cap_total_buffer_size: 'list[int]'
  cap_cell_size: 'list[int]'
  cap_max_pg_buffers: 'list[int]'
  cap_num_pool_supported: 'list[int]'
  reserved_at_240: 'list[int]'
  cap_sbsr_stat_size: 'list[int]'
  cap_max_tclass_data: 'list[int]'
  cap_max_cpu_ingress_tclass_sb: 'list[int]'
struct_mlx5_ifc_sbcam_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('feature_group', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('access_reg_group', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('sb_access_reg_cap_mask', ((ctypes.c_ubyte * 32) * 4), 64), ('reserved_at_c0', (ctypes.c_ubyte * 128), 192), ('sb_feature_cap_mask', ((ctypes.c_ubyte * 32) * 4), 320), ('reserved_at_1c0', (ctypes.c_ubyte * 64), 448), ('cap_total_buffer_size', (ctypes.c_ubyte * 32), 512), ('cap_cell_size', (ctypes.c_ubyte * 16), 544), ('cap_max_pg_buffers', (ctypes.c_ubyte * 8), 560), ('cap_num_pool_supported', (ctypes.c_ubyte * 8), 568), ('reserved_at_240', (ctypes.c_ubyte * 8), 576), ('cap_sbsr_stat_size', (ctypes.c_ubyte * 8), 584), ('cap_max_tclass_data', (ctypes.c_ubyte * 8), 592), ('cap_max_cpu_ingress_tclass_sb', (ctypes.c_ubyte * 8), 600)])
@c.record
class struct_mlx5_ifc_pbmc_reg_bits(c.Struct):
  SIZE = 864
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  reserved_at_10: 'list[int]'
  xoff_timer_value: 'list[int]'
  xoff_refresh: 'list[int]'
  reserved_at_40: 'list[int]'
  fullness_threshold: 'list[int]'
  port_buffer_size: 'list[int]'
  buffer: 'list[struct_mlx5_ifc_bufferx_reg_bits]'
  reserved_at_2e0: 'list[int]'
struct_mlx5_ifc_pbmc_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('xoff_timer_value', (ctypes.c_ubyte * 16), 32), ('xoff_refresh', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 9), 64), ('fullness_threshold', (ctypes.c_ubyte * 7), 73), ('port_buffer_size', (ctypes.c_ubyte * 16), 80), ('buffer', (struct_mlx5_ifc_bufferx_reg_bits * 10), 96), ('reserved_at_2e0', (ctypes.c_ubyte * 128), 736)])
@c.record
class struct_mlx5_ifc_sbpr_reg_bits(c.Struct):
  SIZE = 192
  desc: 'list[int]'
  snap: 'list[int]'
  reserved_at_2: 'list[int]'
  dir: 'list[int]'
  reserved_at_8: 'list[int]'
  pool: 'list[int]'
  infi_size: 'list[int]'
  reserved_at_21: 'list[int]'
  size: 'list[int]'
  reserved_at_40: 'list[int]'
  mode: 'list[int]'
  reserved_at_60: 'list[int]'
  buff_occupancy: 'list[int]'
  clr: 'list[int]'
  reserved_at_81: 'list[int]'
  max_buff_occupancy: 'list[int]'
  reserved_at_a0: 'list[int]'
  ext_buff_occupancy: 'list[int]'
struct_mlx5_ifc_sbpr_reg_bits.register_fields([('desc', (ctypes.c_ubyte * 1), 0), ('snap', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 4), 2), ('dir', (ctypes.c_ubyte * 2), 6), ('reserved_at_8', (ctypes.c_ubyte * 20), 8), ('pool', (ctypes.c_ubyte * 4), 28), ('infi_size', (ctypes.c_ubyte * 1), 32), ('reserved_at_21', (ctypes.c_ubyte * 7), 33), ('size', (ctypes.c_ubyte * 24), 40), ('reserved_at_40', (ctypes.c_ubyte * 28), 64), ('mode', (ctypes.c_ubyte * 4), 92), ('reserved_at_60', (ctypes.c_ubyte * 8), 96), ('buff_occupancy', (ctypes.c_ubyte * 24), 104), ('clr', (ctypes.c_ubyte * 1), 128), ('reserved_at_81', (ctypes.c_ubyte * 7), 129), ('max_buff_occupancy', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('ext_buff_occupancy', (ctypes.c_ubyte * 24), 168)])
@c.record
class struct_mlx5_ifc_sbcm_reg_bits(c.Struct):
  SIZE = 320
  desc: 'list[int]'
  snap: 'list[int]'
  reserved_at_2: 'list[int]'
  local_port: 'list[int]'
  pnat: 'list[int]'
  pg_buff: 'list[int]'
  reserved_at_18: 'list[int]'
  dir: 'list[int]'
  reserved_at_20: 'list[int]'
  exc: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_80: 'list[int]'
  buff_occupancy: 'list[int]'
  clr: 'list[int]'
  reserved_at_a1: 'list[int]'
  max_buff_occupancy: 'list[int]'
  reserved_at_c0: 'list[int]'
  min_buff: 'list[int]'
  infi_max: 'list[int]'
  reserved_at_e1: 'list[int]'
  max_buff: 'list[int]'
  reserved_at_100: 'list[int]'
  reserved_at_120: 'list[int]'
  pool: 'list[int]'
struct_mlx5_ifc_sbcm_reg_bits.register_fields([('desc', (ctypes.c_ubyte * 1), 0), ('snap', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 6), 2), ('local_port', (ctypes.c_ubyte * 8), 8), ('pnat', (ctypes.c_ubyte * 2), 16), ('pg_buff', (ctypes.c_ubyte * 6), 18), ('reserved_at_18', (ctypes.c_ubyte * 6), 24), ('dir', (ctypes.c_ubyte * 2), 30), ('reserved_at_20', (ctypes.c_ubyte * 31), 32), ('exc', (ctypes.c_ubyte * 1), 63), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('buff_occupancy', (ctypes.c_ubyte * 24), 136), ('clr', (ctypes.c_ubyte * 1), 160), ('reserved_at_a1', (ctypes.c_ubyte * 7), 161), ('max_buff_occupancy', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 8), 192), ('min_buff', (ctypes.c_ubyte * 24), 200), ('infi_max', (ctypes.c_ubyte * 1), 224), ('reserved_at_e1', (ctypes.c_ubyte * 7), 225), ('max_buff', (ctypes.c_ubyte * 24), 232), ('reserved_at_100', (ctypes.c_ubyte * 32), 256), ('reserved_at_120', (ctypes.c_ubyte * 28), 288), ('pool', (ctypes.c_ubyte * 4), 316)])
@c.record
class struct_mlx5_ifc_qtct_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  port_number: 'list[int]'
  reserved_at_10: 'list[int]'
  prio: 'list[int]'
  reserved_at_20: 'list[int]'
  tclass: 'list[int]'
struct_mlx5_ifc_qtct_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('port_number', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 13), 16), ('prio', (ctypes.c_ubyte * 3), 29), ('reserved_at_20', (ctypes.c_ubyte * 29), 32), ('tclass', (ctypes.c_ubyte * 3), 61)])
@c.record
class struct_mlx5_ifc_mcia_reg_bits(c.Struct):
  SIZE = 512
  l: 'list[int]'
  reserved_at_1: 'list[int]'
  module: 'list[int]'
  reserved_at_10: 'list[int]'
  status: 'list[int]'
  i2c_device_address: 'list[int]'
  page_number: 'list[int]'
  device_address: 'list[int]'
  reserved_at_40: 'list[int]'
  size: 'list[int]'
  reserved_at_60: 'list[int]'
  dword_0: 'list[int]'
  dword_1: 'list[int]'
  dword_2: 'list[int]'
  dword_3: 'list[int]'
  dword_4: 'list[int]'
  dword_5: 'list[int]'
  dword_6: 'list[int]'
  dword_7: 'list[int]'
  dword_8: 'list[int]'
  dword_9: 'list[int]'
  dword_10: 'list[int]'
  dword_11: 'list[int]'
struct_mlx5_ifc_mcia_reg_bits.register_fields([('l', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 7), 1), ('module', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 8), 16), ('status', (ctypes.c_ubyte * 8), 24), ('i2c_device_address', (ctypes.c_ubyte * 8), 32), ('page_number', (ctypes.c_ubyte * 8), 40), ('device_address', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('size', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('dword_0', (ctypes.c_ubyte * 32), 128), ('dword_1', (ctypes.c_ubyte * 32), 160), ('dword_2', (ctypes.c_ubyte * 32), 192), ('dword_3', (ctypes.c_ubyte * 32), 224), ('dword_4', (ctypes.c_ubyte * 32), 256), ('dword_5', (ctypes.c_ubyte * 32), 288), ('dword_6', (ctypes.c_ubyte * 32), 320), ('dword_7', (ctypes.c_ubyte * 32), 352), ('dword_8', (ctypes.c_ubyte * 32), 384), ('dword_9', (ctypes.c_ubyte * 32), 416), ('dword_10', (ctypes.c_ubyte * 32), 448), ('dword_11', (ctypes.c_ubyte * 32), 480)])
@c.record
class struct_mlx5_ifc_dcbx_param_bits(c.Struct):
  SIZE = 512
  dcbx_cee_cap: 'list[int]'
  dcbx_ieee_cap: 'list[int]'
  dcbx_standby_cap: 'list[int]'
  reserved_at_3: 'list[int]'
  port_number: 'list[int]'
  reserved_at_10: 'list[int]'
  max_application_table_size: 'list[int]'
  reserved_at_20: 'list[int]'
  version_oper: 'list[int]'
  reserved_at_38: 'list[int]'
  version_admin: 'list[int]'
  willing_admin: 'list[int]'
  reserved_at_41: 'list[int]'
  pfc_cap_oper: 'list[int]'
  reserved_at_48: 'list[int]'
  pfc_cap_admin: 'list[int]'
  reserved_at_50: 'list[int]'
  num_of_tc_oper: 'list[int]'
  reserved_at_58: 'list[int]'
  num_of_tc_admin: 'list[int]'
  remote_willing: 'list[int]'
  reserved_at_61: 'list[int]'
  remote_pfc_cap: 'list[int]'
  reserved_at_68: 'list[int]'
  remote_num_of_tc: 'list[int]'
  reserved_at_80: 'list[int]'
  error: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_dcbx_param_bits.register_fields([('dcbx_cee_cap', (ctypes.c_ubyte * 1), 0), ('dcbx_ieee_cap', (ctypes.c_ubyte * 1), 1), ('dcbx_standby_cap', (ctypes.c_ubyte * 1), 2), ('reserved_at_3', (ctypes.c_ubyte * 5), 3), ('port_number', (ctypes.c_ubyte * 8), 8), ('reserved_at_10', (ctypes.c_ubyte * 10), 16), ('max_application_table_size', (ctypes.c_ubyte * 6), 26), ('reserved_at_20', (ctypes.c_ubyte * 21), 32), ('version_oper', (ctypes.c_ubyte * 3), 53), ('reserved_at_38', (ctypes.c_ubyte * 5), 56), ('version_admin', (ctypes.c_ubyte * 3), 61), ('willing_admin', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 3), 65), ('pfc_cap_oper', (ctypes.c_ubyte * 4), 68), ('reserved_at_48', (ctypes.c_ubyte * 4), 72), ('pfc_cap_admin', (ctypes.c_ubyte * 4), 76), ('reserved_at_50', (ctypes.c_ubyte * 4), 80), ('num_of_tc_oper', (ctypes.c_ubyte * 4), 84), ('reserved_at_58', (ctypes.c_ubyte * 4), 88), ('num_of_tc_admin', (ctypes.c_ubyte * 4), 92), ('remote_willing', (ctypes.c_ubyte * 1), 96), ('reserved_at_61', (ctypes.c_ubyte * 3), 97), ('remote_pfc_cap', (ctypes.c_ubyte * 4), 100), ('reserved_at_68', (ctypes.c_ubyte * 20), 104), ('remote_num_of_tc', (ctypes.c_ubyte * 4), 124), ('reserved_at_80', (ctypes.c_ubyte * 24), 128), ('error', (ctypes.c_ubyte * 8), 152), ('reserved_at_a0', (ctypes.c_ubyte * 352), 160)])
_anonenum127: dict[int, str] = {(MLX5_LAG_PORT_SELECT_MODE_QUEUE_AFFINITY:=0): 'MLX5_LAG_PORT_SELECT_MODE_QUEUE_AFFINITY', (MLX5_LAG_PORT_SELECT_MODE_PORT_SELECT_FT:=1): 'MLX5_LAG_PORT_SELECT_MODE_PORT_SELECT_FT', (MLX5_LAG_PORT_SELECT_MODE_PORT_SELECT_MPESW:=2): 'MLX5_LAG_PORT_SELECT_MODE_PORT_SELECT_MPESW'}
@c.record
class struct_mlx5_ifc_lagc_bits(c.Struct):
  SIZE = 64
  fdb_selection_mode: 'list[int]'
  reserved_at_1: 'list[int]'
  port_select_mode: 'list[int]'
  reserved_at_18: 'list[int]'
  lag_state: 'list[int]'
  reserved_at_20: 'list[int]'
  active_port: 'list[int]'
  reserved_at_30: 'list[int]'
  tx_remap_affinity_2: 'list[int]'
  reserved_at_38: 'list[int]'
  tx_remap_affinity_1: 'list[int]'
struct_mlx5_ifc_lagc_bits.register_fields([('fdb_selection_mode', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 20), 1), ('port_select_mode', (ctypes.c_ubyte * 3), 21), ('reserved_at_18', (ctypes.c_ubyte * 5), 24), ('lag_state', (ctypes.c_ubyte * 3), 29), ('reserved_at_20', (ctypes.c_ubyte * 12), 32), ('active_port', (ctypes.c_ubyte * 4), 44), ('reserved_at_30', (ctypes.c_ubyte * 4), 48), ('tx_remap_affinity_2', (ctypes.c_ubyte * 4), 52), ('reserved_at_38', (ctypes.c_ubyte * 4), 56), ('tx_remap_affinity_1', (ctypes.c_ubyte * 4), 60)])
@c.record
class struct_mlx5_ifc_create_lag_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_create_lag_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_create_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  ctx: 'struct_mlx5_ifc_lagc_bits'
struct_mlx5_ifc_create_lag_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('ctx', struct_mlx5_ifc_lagc_bits, 64)])
@c.record
class struct_mlx5_ifc_modify_lag_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_modify_lag_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_modify_lag_in_bits(c.Struct):
  SIZE = 192
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  field_select: 'list[int]'
  ctx: 'struct_mlx5_ifc_lagc_bits'
struct_mlx5_ifc_modify_lag_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('field_select', (ctypes.c_ubyte * 32), 96), ('ctx', struct_mlx5_ifc_lagc_bits, 128)])
@c.record
class struct_mlx5_ifc_query_lag_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  ctx: 'struct_mlx5_ifc_lagc_bits'
struct_mlx5_ifc_query_lag_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('ctx', struct_mlx5_ifc_lagc_bits, 64)])
@c.record
class struct_mlx5_ifc_query_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_lag_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_lag_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_lag_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_lag_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_create_vport_lag_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_create_vport_lag_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_create_vport_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_create_vport_lag_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_vport_lag_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_vport_lag_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_destroy_vport_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_vport_lag_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
_anonenum128: dict[int, str] = {(MLX5_MODIFY_MEMIC_OP_MOD_ALLOC:=0): 'MLX5_MODIFY_MEMIC_OP_MOD_ALLOC', (MLX5_MODIFY_MEMIC_OP_MOD_DEALLOC:=1): 'MLX5_MODIFY_MEMIC_OP_MOD_DEALLOC'}
@c.record
class struct_mlx5_ifc_modify_memic_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_60: 'list[int]'
  memic_operation_type: 'list[int]'
  memic_start_addr: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_modify_memic_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 24), 96), ('memic_operation_type', (ctypes.c_ubyte * 8), 120), ('memic_start_addr', (ctypes.c_ubyte * 64), 128), ('reserved_at_c0', (ctypes.c_ubyte * 320), 192)])
@c.record
class struct_mlx5_ifc_modify_memic_out_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  memic_operation_addr: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_modify_memic_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('memic_operation_addr', (ctypes.c_ubyte * 64), 128), ('reserved_at_c0', (ctypes.c_ubyte * 320), 192)])
@c.record
class struct_mlx5_ifc_alloc_memic_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_30: 'list[int]'
  reserved_at_40: 'list[int]'
  log_memic_addr_alignment: 'list[int]'
  range_start_addr: 'list[int]'
  range_size: 'list[int]'
  memic_size: 'list[int]'
struct_mlx5_ifc_alloc_memic_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_30', (ctypes.c_ubyte * 32), 64), ('reserved_at_40', (ctypes.c_ubyte * 24), 96), ('log_memic_addr_alignment', (ctypes.c_ubyte * 8), 120), ('range_start_addr', (ctypes.c_ubyte * 64), 128), ('range_size', (ctypes.c_ubyte * 32), 192), ('memic_size', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_alloc_memic_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  memic_start_addr: 'list[int]'
struct_mlx5_ifc_alloc_memic_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('memic_start_addr', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_memic_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  memic_start_addr: 'list[int]'
  memic_size: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_dealloc_memic_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('memic_start_addr', (ctypes.c_ubyte * 64), 128), ('memic_size', (ctypes.c_ubyte * 32), 192), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_dealloc_memic_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_memic_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_umem_bits(c.Struct):
  SIZE = 256
  reserved_at_0: 'list[int]'
  ats: 'list[int]'
  reserved_at_81: 'list[int]'
  log_page_size: 'list[int]'
  page_offset: 'list[int]'
  num_of_mtt: 'list[int]'
  mtt: 'list[struct_mlx5_ifc_mtt_bits]'
struct_mlx5_ifc_umem_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 128), 0), ('ats', (ctypes.c_ubyte * 1), 128), ('reserved_at_81', (ctypes.c_ubyte * 26), 129), ('log_page_size', (ctypes.c_ubyte * 5), 155), ('page_offset', (ctypes.c_ubyte * 32), 160), ('num_of_mtt', (ctypes.c_ubyte * 64), 192), ('mtt', (struct_mlx5_ifc_mtt_bits * 0), 256)])
@c.record
class struct_mlx5_ifc_uctx_bits(c.Struct):
  SIZE = 384
  cap: 'list[int]'
  reserved_at_20: 'list[int]'
struct_mlx5_ifc_uctx_bits.register_fields([('cap', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 352), 32)])
@c.record
class struct_mlx5_ifc_sw_icm_bits(c.Struct):
  SIZE = 512
  modify_field_select: 'list[int]'
  reserved_at_40: 'list[int]'
  log_sw_icm_size: 'list[int]'
  reserved_at_60: 'list[int]'
  sw_icm_start_addr: 'list[int]'
  reserved_at_c0: 'list[int]'
struct_mlx5_ifc_sw_icm_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('log_sw_icm_size', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('sw_icm_start_addr', (ctypes.c_ubyte * 64), 128), ('reserved_at_c0', (ctypes.c_ubyte * 320), 192)])
@c.record
class struct_mlx5_ifc_geneve_tlv_option_bits(c.Struct):
  SIZE = 512
  modify_field_select: 'list[int]'
  reserved_at_40: 'list[int]'
  geneve_option_fte_index: 'list[int]'
  option_class: 'list[int]'
  option_type: 'list[int]'
  reserved_at_78: 'list[int]'
  option_data_length: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_geneve_tlv_option_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('geneve_option_fte_index', (ctypes.c_ubyte * 8), 88), ('option_class', (ctypes.c_ubyte * 16), 96), ('option_type', (ctypes.c_ubyte * 8), 112), ('reserved_at_78', (ctypes.c_ubyte * 3), 120), ('option_data_length', (ctypes.c_ubyte * 5), 123), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
@c.record
class struct_mlx5_ifc_create_umem_in_bits(c.Struct):
  SIZE = 384
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  umem: 'struct_mlx5_ifc_umem_bits'
struct_mlx5_ifc_create_umem_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('umem', struct_mlx5_ifc_umem_bits, 128)])
@c.record
class struct_mlx5_ifc_create_umem_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  umem_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_umem_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('umem_id', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_umem_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  umem_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_umem_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 8), 64), ('umem_id', (ctypes.c_ubyte * 24), 72), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_umem_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_umem_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_create_uctx_in_bits(c.Struct):
  SIZE = 512
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  uctx: 'struct_mlx5_ifc_uctx_bits'
struct_mlx5_ifc_create_uctx_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('uctx', struct_mlx5_ifc_uctx_bits, 128)])
@c.record
class struct_mlx5_ifc_create_uctx_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  uid: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_create_uctx_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('uid', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_uctx_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  uid: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_destroy_uctx_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('uid', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_destroy_uctx_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_destroy_uctx_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_create_sw_icm_in_bits(c.Struct):
  SIZE = 640
  hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  sw_icm: 'struct_mlx5_ifc_sw_icm_bits'
struct_mlx5_ifc_create_sw_icm_in_bits.register_fields([('hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('sw_icm', struct_mlx5_ifc_sw_icm_bits, 128)])
@c.record
class struct_mlx5_ifc_create_geneve_tlv_option_in_bits(c.Struct):
  SIZE = 640
  hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  geneve_tlv_opt: 'struct_mlx5_ifc_geneve_tlv_option_bits'
struct_mlx5_ifc_create_geneve_tlv_option_in_bits.register_fields([('hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('geneve_tlv_opt', struct_mlx5_ifc_geneve_tlv_option_bits, 128)])
@c.record
class struct_mlx5_ifc_mtrc_string_db_param_bits(c.Struct):
  SIZE = 64
  string_db_base_address: 'list[int]'
  reserved_at_20: 'list[int]'
  string_db_size: 'list[int]'
struct_mlx5_ifc_mtrc_string_db_param_bits.register_fields([('string_db_base_address', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 8), 32), ('string_db_size', (ctypes.c_ubyte * 24), 40)])
@c.record
class struct_mlx5_ifc_mtrc_cap_bits(c.Struct):
  SIZE = 1024
  trace_owner: 'list[int]'
  trace_to_memory: 'list[int]'
  reserved_at_2: 'list[int]'
  trc_ver: 'list[int]'
  reserved_at_8: 'list[int]'
  num_string_db: 'list[int]'
  first_string_trace: 'list[int]'
  num_string_trace: 'list[int]'
  reserved_at_30: 'list[int]'
  log_max_trace_buffer_size: 'list[int]'
  reserved_at_60: 'list[int]'
  string_db_param: 'list[struct_mlx5_ifc_mtrc_string_db_param_bits]'
  reserved_at_280: 'list[int]'
struct_mlx5_ifc_mtrc_cap_bits.register_fields([('trace_owner', (ctypes.c_ubyte * 1), 0), ('trace_to_memory', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 4), 2), ('trc_ver', (ctypes.c_ubyte * 2), 6), ('reserved_at_8', (ctypes.c_ubyte * 20), 8), ('num_string_db', (ctypes.c_ubyte * 4), 28), ('first_string_trace', (ctypes.c_ubyte * 8), 32), ('num_string_trace', (ctypes.c_ubyte * 8), 40), ('reserved_at_30', (ctypes.c_ubyte * 40), 48), ('log_max_trace_buffer_size', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('string_db_param', (struct_mlx5_ifc_mtrc_string_db_param_bits * 8), 128), ('reserved_at_280', (ctypes.c_ubyte * 384), 640)])
@c.record
class struct_mlx5_ifc_mtrc_conf_bits(c.Struct):
  SIZE = 1024
  reserved_at_0: 'list[int]'
  trace_mode: 'list[int]'
  reserved_at_20: 'list[int]'
  log_trace_buffer_size: 'list[int]'
  trace_mkey: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_mtrc_conf_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 28), 0), ('trace_mode', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 24), 32), ('log_trace_buffer_size', (ctypes.c_ubyte * 8), 56), ('trace_mkey', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 928), 96)])
@c.record
class struct_mlx5_ifc_mtrc_stdb_bits(c.Struct):
  SIZE = 64
  string_db_index: 'list[int]'
  reserved_at_4: 'list[int]'
  read_size: 'list[int]'
  start_offset: 'list[int]'
  string_db_data: 'list[int]'
struct_mlx5_ifc_mtrc_stdb_bits.register_fields([('string_db_index', (ctypes.c_ubyte * 4), 0), ('reserved_at_4', (ctypes.c_ubyte * 4), 4), ('read_size', (ctypes.c_ubyte * 24), 8), ('start_offset', (ctypes.c_ubyte * 32), 32), ('string_db_data', (ctypes.c_ubyte * 0), 64)])
@c.record
class struct_mlx5_ifc_mtrc_ctrl_bits(c.Struct):
  SIZE = 512
  trace_status: 'list[int]'
  reserved_at_2: 'list[int]'
  arm_event: 'list[int]'
  reserved_at_5: 'list[int]'
  modify_field_select: 'list[int]'
  reserved_at_20: 'list[int]'
  current_timestamp52_32: 'list[int]'
  current_timestamp31_0: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_mtrc_ctrl_bits.register_fields([('trace_status', (ctypes.c_ubyte * 2), 0), ('reserved_at_2', (ctypes.c_ubyte * 2), 2), ('arm_event', (ctypes.c_ubyte * 1), 4), ('reserved_at_5', (ctypes.c_ubyte * 11), 5), ('modify_field_select', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 43), 32), ('current_timestamp52_32', (ctypes.c_ubyte * 21), 75), ('current_timestamp31_0', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
@c.record
class struct_mlx5_ifc_host_params_context_bits(c.Struct):
  SIZE = 512
  host_number: 'list[int]'
  reserved_at_8: 'list[int]'
  host_pf_not_exist: 'list[int]'
  reserved_at_14: 'list[int]'
  host_pf_disabled: 'list[int]'
  host_num_of_vfs: 'list[int]'
  host_total_vfs: 'list[int]'
  host_pci_bus: 'list[int]'
  reserved_at_40: 'list[int]'
  host_pci_device: 'list[int]'
  reserved_at_60: 'list[int]'
  host_pci_function: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_host_params_context_bits.register_fields([('host_number', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 5), 8), ('host_pf_not_exist', (ctypes.c_ubyte * 1), 13), ('reserved_at_14', (ctypes.c_ubyte * 1), 14), ('host_pf_disabled', (ctypes.c_ubyte * 1), 15), ('host_num_of_vfs', (ctypes.c_ubyte * 16), 16), ('host_total_vfs', (ctypes.c_ubyte * 16), 32), ('host_pci_bus', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('host_pci_device', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 16), 96), ('host_pci_function', (ctypes.c_ubyte * 16), 112), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
@c.record
class struct_mlx5_ifc_query_esw_functions_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_esw_functions_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_query_esw_functions_out_bits(c.Struct):
  SIZE = 1024
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  host_params_context: 'struct_mlx5_ifc_host_params_context_bits'
  reserved_at_280: 'list[int]'
  host_sf_enable: 'list[list[int]]'
struct_mlx5_ifc_query_esw_functions_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('host_params_context', struct_mlx5_ifc_host_params_context_bits, 128), ('reserved_at_280', (ctypes.c_ubyte * 384), 640), ('host_sf_enable', ((ctypes.c_ubyte * 64) * 0), 1024)])
@c.record
class struct_mlx5_ifc_sf_partition_bits(c.Struct):
  SIZE = 32
  reserved_at_0: 'list[int]'
  log_num_sf: 'list[int]'
  log_sf_bar_size: 'list[int]'
struct_mlx5_ifc_sf_partition_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('log_num_sf', (ctypes.c_ubyte * 8), 16), ('log_sf_bar_size', (ctypes.c_ubyte * 8), 24)])
@c.record
class struct_mlx5_ifc_query_sf_partitions_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  num_sf_partitions: 'list[int]'
  reserved_at_60: 'list[int]'
  sf_partition: 'list[struct_mlx5_ifc_sf_partition_bits]'
struct_mlx5_ifc_query_sf_partitions_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 24), 64), ('num_sf_partitions', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('sf_partition', (struct_mlx5_ifc_sf_partition_bits * 0), 128)])
@c.record
class struct_mlx5_ifc_query_sf_partitions_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_query_sf_partitions_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_sf_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_dealloc_sf_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_dealloc_sf_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_dealloc_sf_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_alloc_sf_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_alloc_sf_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_alloc_sf_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  reserved_at_10: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  function_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_alloc_sf_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('reserved_at_10', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('function_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_affiliated_event_header_bits(c.Struct):
  SIZE = 64
  reserved_at_0: 'list[int]'
  obj_type: 'list[int]'
  obj_id: 'list[int]'
struct_mlx5_ifc_affiliated_event_header_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 16), 0), ('obj_type', (ctypes.c_ubyte * 16), 16), ('obj_id', (ctypes.c_ubyte * 32), 32)])
_anonenum129: dict[int, str] = {(MLX5_GENERAL_OBJECT_TYPES_ENCRYPTION_KEY:=12): 'MLX5_GENERAL_OBJECT_TYPES_ENCRYPTION_KEY', (MLX5_GENERAL_OBJECT_TYPES_IPSEC:=19): 'MLX5_GENERAL_OBJECT_TYPES_IPSEC', (MLX5_GENERAL_OBJECT_TYPES_SAMPLER:=32): 'MLX5_GENERAL_OBJECT_TYPES_SAMPLER', (MLX5_GENERAL_OBJECT_TYPES_FLOW_METER_ASO:=36): 'MLX5_GENERAL_OBJECT_TYPES_FLOW_METER_ASO', (MLX5_GENERAL_OBJECT_TYPES_MACSEC:=39): 'MLX5_GENERAL_OBJECT_TYPES_MACSEC', (MLX5_GENERAL_OBJECT_TYPES_INT_KEK:=71): 'MLX5_GENERAL_OBJECT_TYPES_INT_KEK', (MLX5_GENERAL_OBJECT_TYPES_RDMA_CTRL:=83): 'MLX5_GENERAL_OBJECT_TYPES_RDMA_CTRL', (MLX5_GENERAL_OBJECT_TYPES_PCIE_CONG_EVENT:=88): 'MLX5_GENERAL_OBJECT_TYPES_PCIE_CONG_EVENT', (MLX5_GENERAL_OBJECT_TYPES_FLOW_TABLE_ALIAS:=65301): 'MLX5_GENERAL_OBJECT_TYPES_FLOW_TABLE_ALIAS'}
_anonenum130: dict[int, str] = {(MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_ENCRYPTION_KEY:=0): 'MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_ENCRYPTION_KEY', (MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_IPSEC:=1): 'MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_IPSEC', (MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_SAMPLER:=2): 'MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_SAMPLER', (MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_FLOW_METER_ASO:=3): 'MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_FLOW_METER_ASO'}
_anonenum131: dict[int, str] = {(MLX5_HCA_CAP_2_GENERAL_OBJECT_TYPES_RDMA_CTRL:=0): 'MLX5_HCA_CAP_2_GENERAL_OBJECT_TYPES_RDMA_CTRL', (MLX5_HCA_CAP_2_GENERAL_OBJECT_TYPES_PCIE_CONG_EVENT:=1): 'MLX5_HCA_CAP_2_GENERAL_OBJECT_TYPES_PCIE_CONG_EVENT'}
_anonenum132: dict[int, str] = {(MLX5_IPSEC_OBJECT_ICV_LEN_16B:=0): 'MLX5_IPSEC_OBJECT_ICV_LEN_16B'}
_anonenum133: dict[int, str] = {(MLX5_IPSEC_ASO_REG_C_0_1:=0): 'MLX5_IPSEC_ASO_REG_C_0_1', (MLX5_IPSEC_ASO_REG_C_2_3:=1): 'MLX5_IPSEC_ASO_REG_C_2_3', (MLX5_IPSEC_ASO_REG_C_4_5:=2): 'MLX5_IPSEC_ASO_REG_C_4_5', (MLX5_IPSEC_ASO_REG_C_6_7:=3): 'MLX5_IPSEC_ASO_REG_C_6_7'}
_anonenum134: dict[int, str] = {(MLX5_IPSEC_ASO_MODE:=0): 'MLX5_IPSEC_ASO_MODE', (MLX5_IPSEC_ASO_REPLAY_PROTECTION:=1): 'MLX5_IPSEC_ASO_REPLAY_PROTECTION', (MLX5_IPSEC_ASO_INC_SN:=2): 'MLX5_IPSEC_ASO_INC_SN'}
_anonenum135: dict[int, str] = {(MLX5_IPSEC_ASO_REPLAY_WIN_32BIT:=0): 'MLX5_IPSEC_ASO_REPLAY_WIN_32BIT', (MLX5_IPSEC_ASO_REPLAY_WIN_64BIT:=1): 'MLX5_IPSEC_ASO_REPLAY_WIN_64BIT', (MLX5_IPSEC_ASO_REPLAY_WIN_128BIT:=2): 'MLX5_IPSEC_ASO_REPLAY_WIN_128BIT', (MLX5_IPSEC_ASO_REPLAY_WIN_256BIT:=3): 'MLX5_IPSEC_ASO_REPLAY_WIN_256BIT'}
@c.record
class struct_mlx5_ifc_ipsec_aso_bits(c.Struct):
  SIZE = 512
  valid: 'list[int]'
  reserved_at_201: 'list[int]'
  mode: 'list[int]'
  window_sz: 'list[int]'
  soft_lft_arm: 'list[int]'
  hard_lft_arm: 'list[int]'
  remove_flow_enable: 'list[int]'
  esn_event_arm: 'list[int]'
  reserved_at_20a: 'list[int]'
  remove_flow_pkt_cnt: 'list[int]'
  remove_flow_soft_lft: 'list[int]'
  reserved_at_260: 'list[int]'
  mode_parameter: 'list[int]'
  replay_protection_window: 'list[int]'
struct_mlx5_ifc_ipsec_aso_bits.register_fields([('valid', (ctypes.c_ubyte * 1), 0), ('reserved_at_201', (ctypes.c_ubyte * 1), 1), ('mode', (ctypes.c_ubyte * 2), 2), ('window_sz', (ctypes.c_ubyte * 2), 4), ('soft_lft_arm', (ctypes.c_ubyte * 1), 6), ('hard_lft_arm', (ctypes.c_ubyte * 1), 7), ('remove_flow_enable', (ctypes.c_ubyte * 1), 8), ('esn_event_arm', (ctypes.c_ubyte * 1), 9), ('reserved_at_20a', (ctypes.c_ubyte * 22), 10), ('remove_flow_pkt_cnt', (ctypes.c_ubyte * 32), 32), ('remove_flow_soft_lft', (ctypes.c_ubyte * 32), 64), ('reserved_at_260', (ctypes.c_ubyte * 128), 96), ('mode_parameter', (ctypes.c_ubyte * 32), 224), ('replay_protection_window', (ctypes.c_ubyte * 256), 256)])
@c.record
class struct_mlx5_ifc_ipsec_obj_bits(c.Struct):
  SIZE = 1024
  modify_field_select: 'list[int]'
  full_offload: 'list[int]'
  reserved_at_41: 'list[int]'
  esn_en: 'list[int]'
  esn_overlap: 'list[int]'
  reserved_at_44: 'list[int]'
  icv_length: 'list[int]'
  reserved_at_48: 'list[int]'
  aso_return_reg: 'list[int]'
  reserved_at_50: 'list[int]'
  esn_msb: 'list[int]'
  reserved_at_80: 'list[int]'
  dekn: 'list[int]'
  salt: 'list[int]'
  implicit_iv: 'list[int]'
  reserved_at_100: 'list[int]'
  ipsec_aso_access_pd: 'list[int]'
  reserved_at_120: 'list[int]'
  ipsec_aso: 'struct_mlx5_ifc_ipsec_aso_bits'
struct_mlx5_ifc_ipsec_obj_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('full_offload', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 1), 65), ('esn_en', (ctypes.c_ubyte * 1), 66), ('esn_overlap', (ctypes.c_ubyte * 1), 67), ('reserved_at_44', (ctypes.c_ubyte * 2), 68), ('icv_length', (ctypes.c_ubyte * 2), 70), ('reserved_at_48', (ctypes.c_ubyte * 4), 72), ('aso_return_reg', (ctypes.c_ubyte * 4), 76), ('reserved_at_50', (ctypes.c_ubyte * 16), 80), ('esn_msb', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('dekn', (ctypes.c_ubyte * 24), 136), ('salt', (ctypes.c_ubyte * 32), 160), ('implicit_iv', (ctypes.c_ubyte * 64), 192), ('reserved_at_100', (ctypes.c_ubyte * 8), 256), ('ipsec_aso_access_pd', (ctypes.c_ubyte * 24), 264), ('reserved_at_120', (ctypes.c_ubyte * 224), 288), ('ipsec_aso', struct_mlx5_ifc_ipsec_aso_bits, 512)])
@c.record
class struct_mlx5_ifc_create_ipsec_obj_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  ipsec_object: 'struct_mlx5_ifc_ipsec_obj_bits'
struct_mlx5_ifc_create_ipsec_obj_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('ipsec_object', struct_mlx5_ifc_ipsec_obj_bits, 128)])
_anonenum136: dict[int, str] = {(MLX5_MODIFY_IPSEC_BITMASK_ESN_OVERLAP:=0): 'MLX5_MODIFY_IPSEC_BITMASK_ESN_OVERLAP', (MLX5_MODIFY_IPSEC_BITMASK_ESN_MSB:=1): 'MLX5_MODIFY_IPSEC_BITMASK_ESN_MSB'}
@c.record
class struct_mlx5_ifc_query_ipsec_obj_out_bits(c.Struct):
  SIZE = 1152
  general_obj_out_cmd_hdr: 'struct_mlx5_ifc_general_obj_out_cmd_hdr_bits'
  ipsec_object: 'struct_mlx5_ifc_ipsec_obj_bits'
struct_mlx5_ifc_query_ipsec_obj_out_bits.register_fields([('general_obj_out_cmd_hdr', struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0), ('ipsec_object', struct_mlx5_ifc_ipsec_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_modify_ipsec_obj_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  ipsec_object: 'struct_mlx5_ifc_ipsec_obj_bits'
struct_mlx5_ifc_modify_ipsec_obj_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('ipsec_object', struct_mlx5_ifc_ipsec_obj_bits, 128)])
_anonenum137: dict[int, str] = {(MLX5_MACSEC_ASO_REPLAY_PROTECTION:=1): 'MLX5_MACSEC_ASO_REPLAY_PROTECTION'}
_anonenum138: dict[int, str] = {(MLX5_MACSEC_ASO_REPLAY_WIN_32BIT:=0): 'MLX5_MACSEC_ASO_REPLAY_WIN_32BIT', (MLX5_MACSEC_ASO_REPLAY_WIN_64BIT:=1): 'MLX5_MACSEC_ASO_REPLAY_WIN_64BIT', (MLX5_MACSEC_ASO_REPLAY_WIN_128BIT:=2): 'MLX5_MACSEC_ASO_REPLAY_WIN_128BIT', (MLX5_MACSEC_ASO_REPLAY_WIN_256BIT:=3): 'MLX5_MACSEC_ASO_REPLAY_WIN_256BIT'}
@c.record
class struct_mlx5_ifc_macsec_aso_bits(c.Struct):
  SIZE = 512
  valid: 'list[int]'
  reserved_at_1: 'list[int]'
  mode: 'list[int]'
  window_size: 'list[int]'
  soft_lifetime_arm: 'list[int]'
  hard_lifetime_arm: 'list[int]'
  remove_flow_enable: 'list[int]'
  epn_event_arm: 'list[int]'
  reserved_at_a: 'list[int]'
  remove_flow_packet_count: 'list[int]'
  remove_flow_soft_lifetime: 'list[int]'
  reserved_at_60: 'list[int]'
  mode_parameter: 'list[int]'
  replay_protection_window: 'list[list[int]]'
struct_mlx5_ifc_macsec_aso_bits.register_fields([('valid', (ctypes.c_ubyte * 1), 0), ('reserved_at_1', (ctypes.c_ubyte * 1), 1), ('mode', (ctypes.c_ubyte * 2), 2), ('window_size', (ctypes.c_ubyte * 2), 4), ('soft_lifetime_arm', (ctypes.c_ubyte * 1), 6), ('hard_lifetime_arm', (ctypes.c_ubyte * 1), 7), ('remove_flow_enable', (ctypes.c_ubyte * 1), 8), ('epn_event_arm', (ctypes.c_ubyte * 1), 9), ('reserved_at_a', (ctypes.c_ubyte * 22), 10), ('remove_flow_packet_count', (ctypes.c_ubyte * 32), 32), ('remove_flow_soft_lifetime', (ctypes.c_ubyte * 32), 64), ('reserved_at_60', (ctypes.c_ubyte * 128), 96), ('mode_parameter', (ctypes.c_ubyte * 32), 224), ('replay_protection_window', ((ctypes.c_ubyte * 32) * 8), 256)])
@c.record
class struct_mlx5_ifc_macsec_offload_obj_bits(c.Struct):
  SIZE = 1024
  modify_field_select: 'list[int]'
  confidentiality_en: 'list[int]'
  reserved_at_41: 'list[int]'
  epn_en: 'list[int]'
  epn_overlap: 'list[int]'
  reserved_at_44: 'list[int]'
  confidentiality_offset: 'list[int]'
  reserved_at_48: 'list[int]'
  aso_return_reg: 'list[int]'
  reserved_at_50: 'list[int]'
  epn_msb: 'list[int]'
  reserved_at_80: 'list[int]'
  dekn: 'list[int]'
  reserved_at_a0: 'list[int]'
  sci: 'list[int]'
  reserved_at_100: 'list[int]'
  macsec_aso_access_pd: 'list[int]'
  reserved_at_120: 'list[int]'
  salt: 'list[list[int]]'
  reserved_at_1e0: 'list[int]'
  macsec_aso: 'struct_mlx5_ifc_macsec_aso_bits'
struct_mlx5_ifc_macsec_offload_obj_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('confidentiality_en', (ctypes.c_ubyte * 1), 64), ('reserved_at_41', (ctypes.c_ubyte * 1), 65), ('epn_en', (ctypes.c_ubyte * 1), 66), ('epn_overlap', (ctypes.c_ubyte * 1), 67), ('reserved_at_44', (ctypes.c_ubyte * 2), 68), ('confidentiality_offset', (ctypes.c_ubyte * 2), 70), ('reserved_at_48', (ctypes.c_ubyte * 4), 72), ('aso_return_reg', (ctypes.c_ubyte * 4), 76), ('reserved_at_50', (ctypes.c_ubyte * 16), 80), ('epn_msb', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('dekn', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('sci', (ctypes.c_ubyte * 64), 192), ('reserved_at_100', (ctypes.c_ubyte * 8), 256), ('macsec_aso_access_pd', (ctypes.c_ubyte * 24), 264), ('reserved_at_120', (ctypes.c_ubyte * 96), 288), ('salt', ((ctypes.c_ubyte * 32) * 3), 384), ('reserved_at_1e0', (ctypes.c_ubyte * 32), 480), ('macsec_aso', struct_mlx5_ifc_macsec_aso_bits, 512)])
@c.record
class struct_mlx5_ifc_create_macsec_obj_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  macsec_object: 'struct_mlx5_ifc_macsec_offload_obj_bits'
struct_mlx5_ifc_create_macsec_obj_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('macsec_object', struct_mlx5_ifc_macsec_offload_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_modify_macsec_obj_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  macsec_object: 'struct_mlx5_ifc_macsec_offload_obj_bits'
struct_mlx5_ifc_modify_macsec_obj_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('macsec_object', struct_mlx5_ifc_macsec_offload_obj_bits, 128)])
_anonenum139: dict[int, str] = {(MLX5_MODIFY_MACSEC_BITMASK_EPN_OVERLAP:=0): 'MLX5_MODIFY_MACSEC_BITMASK_EPN_OVERLAP', (MLX5_MODIFY_MACSEC_BITMASK_EPN_MSB:=1): 'MLX5_MODIFY_MACSEC_BITMASK_EPN_MSB'}
@c.record
class struct_mlx5_ifc_query_macsec_obj_out_bits(c.Struct):
  SIZE = 1152
  general_obj_out_cmd_hdr: 'struct_mlx5_ifc_general_obj_out_cmd_hdr_bits'
  macsec_object: 'struct_mlx5_ifc_macsec_offload_obj_bits'
struct_mlx5_ifc_query_macsec_obj_out_bits.register_fields([('general_obj_out_cmd_hdr', struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0), ('macsec_object', struct_mlx5_ifc_macsec_offload_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_wrapped_dek_bits(c.Struct):
  SIZE = 1024
  gcm_iv: 'list[int]'
  reserved_at_60: 'list[int]'
  const0: 'list[int]'
  key_size: 'list[int]'
  reserved_at_82: 'list[int]'
  key2_invalid: 'list[int]'
  reserved_at_85: 'list[int]'
  pd: 'list[int]'
  key_purpose: 'list[int]'
  reserved_at_a5: 'list[int]'
  kek_id: 'list[int]'
  reserved_at_c0: 'list[int]'
  key1: 'list[list[int]]'
  key2: 'list[list[int]]'
  reserved_at_300: 'list[int]'
  const1: 'list[int]'
  reserved_at_341: 'list[int]'
  reserved_at_360: 'list[int]'
  auth_tag: 'list[int]'
struct_mlx5_ifc_wrapped_dek_bits.register_fields([('gcm_iv', (ctypes.c_ubyte * 96), 0), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('const0', (ctypes.c_ubyte * 1), 128), ('key_size', (ctypes.c_ubyte * 1), 129), ('reserved_at_82', (ctypes.c_ubyte * 2), 130), ('key2_invalid', (ctypes.c_ubyte * 1), 132), ('reserved_at_85', (ctypes.c_ubyte * 3), 133), ('pd', (ctypes.c_ubyte * 24), 136), ('key_purpose', (ctypes.c_ubyte * 5), 160), ('reserved_at_a5', (ctypes.c_ubyte * 19), 165), ('kek_id', (ctypes.c_ubyte * 8), 184), ('reserved_at_c0', (ctypes.c_ubyte * 64), 192), ('key1', ((ctypes.c_ubyte * 32) * 8), 256), ('key2', ((ctypes.c_ubyte * 32) * 8), 512), ('reserved_at_300', (ctypes.c_ubyte * 64), 768), ('const1', (ctypes.c_ubyte * 1), 832), ('reserved_at_341', (ctypes.c_ubyte * 31), 833), ('reserved_at_360', (ctypes.c_ubyte * 32), 864), ('auth_tag', (ctypes.c_ubyte * 128), 896)])
@c.record
class struct_mlx5_ifc_encryption_key_obj_bits(c.Struct):
  SIZE = 4096
  modify_field_select: 'list[int]'
  state: 'list[int]'
  sw_wrapped: 'list[int]'
  reserved_at_49: 'list[int]'
  key_size: 'list[int]'
  reserved_at_58: 'list[int]'
  key_purpose: 'list[int]'
  reserved_at_60: 'list[int]'
  pd: 'list[int]'
  reserved_at_80: 'list[int]'
  opaque: 'list[int]'
  reserved_at_1c0: 'list[int]'
  key: 'list[list[int]]'
  sw_wrapped_dek: 'list[list[int]]'
  reserved_at_a00: 'list[int]'
struct_mlx5_ifc_encryption_key_obj_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('state', (ctypes.c_ubyte * 8), 64), ('sw_wrapped', (ctypes.c_ubyte * 1), 72), ('reserved_at_49', (ctypes.c_ubyte * 11), 73), ('key_size', (ctypes.c_ubyte * 4), 84), ('reserved_at_58', (ctypes.c_ubyte * 4), 88), ('key_purpose', (ctypes.c_ubyte * 4), 92), ('reserved_at_60', (ctypes.c_ubyte * 8), 96), ('pd', (ctypes.c_ubyte * 24), 104), ('reserved_at_80', (ctypes.c_ubyte * 256), 128), ('opaque', (ctypes.c_ubyte * 64), 384), ('reserved_at_1c0', (ctypes.c_ubyte * 64), 448), ('key', ((ctypes.c_ubyte * 128) * 8), 512), ('sw_wrapped_dek', ((ctypes.c_ubyte * 128) * 8), 1536), ('reserved_at_a00', (ctypes.c_ubyte * 1536), 2560)])
@c.record
class struct_mlx5_ifc_create_encryption_key_in_bits(c.Struct):
  SIZE = 4224
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  encryption_key_object: 'struct_mlx5_ifc_encryption_key_obj_bits'
struct_mlx5_ifc_create_encryption_key_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('encryption_key_object', struct_mlx5_ifc_encryption_key_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_modify_encryption_key_in_bits(c.Struct):
  SIZE = 4224
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  encryption_key_object: 'struct_mlx5_ifc_encryption_key_obj_bits'
struct_mlx5_ifc_modify_encryption_key_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('encryption_key_object', struct_mlx5_ifc_encryption_key_obj_bits, 128)])
_anonenum140: dict[int, str] = {(MLX5_FLOW_METER_MODE_BYTES_IP_LENGTH:=0): 'MLX5_FLOW_METER_MODE_BYTES_IP_LENGTH', (MLX5_FLOW_METER_MODE_BYTES_CALC_WITH_L2:=1): 'MLX5_FLOW_METER_MODE_BYTES_CALC_WITH_L2', (MLX5_FLOW_METER_MODE_BYTES_CALC_WITH_L2_IPG:=2): 'MLX5_FLOW_METER_MODE_BYTES_CALC_WITH_L2_IPG', (MLX5_FLOW_METER_MODE_NUM_PACKETS:=3): 'MLX5_FLOW_METER_MODE_NUM_PACKETS'}
@c.record
class struct_mlx5_ifc_flow_meter_parameters_bits(c.Struct):
  SIZE = 256
  valid: 'list[int]'
  bucket_overflow: 'list[int]'
  start_color: 'list[int]'
  both_buckets_on_green: 'list[int]'
  reserved_at_5: 'list[int]'
  meter_mode: 'list[int]'
  reserved_at_8: 'list[int]'
  reserved_at_20: 'list[int]'
  reserved_at_40: 'list[int]'
  cbs_exponent: 'list[int]'
  cbs_mantissa: 'list[int]'
  reserved_at_50: 'list[int]'
  cir_exponent: 'list[int]'
  cir_mantissa: 'list[int]'
  reserved_at_60: 'list[int]'
  reserved_at_80: 'list[int]'
  ebs_exponent: 'list[int]'
  ebs_mantissa: 'list[int]'
  reserved_at_90: 'list[int]'
  eir_exponent: 'list[int]'
  eir_mantissa: 'list[int]'
  reserved_at_a0: 'list[int]'
struct_mlx5_ifc_flow_meter_parameters_bits.register_fields([('valid', (ctypes.c_ubyte * 1), 0), ('bucket_overflow', (ctypes.c_ubyte * 1), 1), ('start_color', (ctypes.c_ubyte * 2), 2), ('both_buckets_on_green', (ctypes.c_ubyte * 1), 4), ('reserved_at_5', (ctypes.c_ubyte * 1), 5), ('meter_mode', (ctypes.c_ubyte * 2), 6), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 3), 64), ('cbs_exponent', (ctypes.c_ubyte * 5), 67), ('cbs_mantissa', (ctypes.c_ubyte * 8), 72), ('reserved_at_50', (ctypes.c_ubyte * 3), 80), ('cir_exponent', (ctypes.c_ubyte * 5), 83), ('cir_mantissa', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 3), 128), ('ebs_exponent', (ctypes.c_ubyte * 5), 131), ('ebs_mantissa', (ctypes.c_ubyte * 8), 136), ('reserved_at_90', (ctypes.c_ubyte * 3), 144), ('eir_exponent', (ctypes.c_ubyte * 5), 147), ('eir_mantissa', (ctypes.c_ubyte * 8), 152), ('reserved_at_a0', (ctypes.c_ubyte * 96), 160)])
@c.record
class struct_mlx5_ifc_flow_meter_aso_obj_bits(c.Struct):
  SIZE = 1024
  modify_field_select: 'list[int]'
  reserved_at_40: 'list[int]'
  reserved_at_80: 'list[int]'
  meter_aso_access_pd: 'list[int]'
  reserved_at_a0: 'list[int]'
  flow_meter_parameters: 'list[struct_mlx5_ifc_flow_meter_parameters_bits]'
struct_mlx5_ifc_flow_meter_aso_obj_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('meter_aso_access_pd', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 352), 160), ('flow_meter_parameters', (struct_mlx5_ifc_flow_meter_parameters_bits * 2), 512)])
@c.record
class struct_mlx5_ifc_create_flow_meter_aso_obj_in_bits(c.Struct):
  SIZE = 1152
  hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  flow_meter_aso_obj: 'struct_mlx5_ifc_flow_meter_aso_obj_bits'
struct_mlx5_ifc_create_flow_meter_aso_obj_in_bits.register_fields([('hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('flow_meter_aso_obj', struct_mlx5_ifc_flow_meter_aso_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_int_kek_obj_bits(c.Struct):
  SIZE = 2048
  modify_field_select: 'list[int]'
  state: 'list[int]'
  auto_gen: 'list[int]'
  reserved_at_49: 'list[int]'
  key_size: 'list[int]'
  reserved_at_58: 'list[int]'
  reserved_at_60: 'list[int]'
  pd: 'list[int]'
  reserved_at_80: 'list[int]'
  key: 'list[list[int]]'
  reserved_at_600: 'list[int]'
struct_mlx5_ifc_int_kek_obj_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('state', (ctypes.c_ubyte * 8), 64), ('auto_gen', (ctypes.c_ubyte * 1), 72), ('reserved_at_49', (ctypes.c_ubyte * 11), 73), ('key_size', (ctypes.c_ubyte * 4), 84), ('reserved_at_58', (ctypes.c_ubyte * 8), 88), ('reserved_at_60', (ctypes.c_ubyte * 8), 96), ('pd', (ctypes.c_ubyte * 24), 104), ('reserved_at_80', (ctypes.c_ubyte * 384), 128), ('key', ((ctypes.c_ubyte * 128) * 8), 512), ('reserved_at_600', (ctypes.c_ubyte * 512), 1536)])
@c.record
class struct_mlx5_ifc_create_int_kek_obj_in_bits(c.Struct):
  SIZE = 2176
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  int_kek_object: 'struct_mlx5_ifc_int_kek_obj_bits'
struct_mlx5_ifc_create_int_kek_obj_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('int_kek_object', struct_mlx5_ifc_int_kek_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_create_int_kek_obj_out_bits(c.Struct):
  SIZE = 2176
  general_obj_out_cmd_hdr: 'struct_mlx5_ifc_general_obj_out_cmd_hdr_bits'
  int_kek_object: 'struct_mlx5_ifc_int_kek_obj_bits'
struct_mlx5_ifc_create_int_kek_obj_out_bits.register_fields([('general_obj_out_cmd_hdr', struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0), ('int_kek_object', struct_mlx5_ifc_int_kek_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_sampler_obj_bits(c.Struct):
  SIZE = 480
  modify_field_select: 'list[int]'
  table_type: 'list[int]'
  level: 'list[int]'
  reserved_at_50: 'list[int]'
  ignore_flow_level: 'list[int]'
  sample_ratio: 'list[int]'
  reserved_at_80: 'list[int]'
  sample_table_id: 'list[int]'
  reserved_at_a0: 'list[int]'
  default_table_id: 'list[int]'
  sw_steering_icm_address_rx: 'list[int]'
  sw_steering_icm_address_tx: 'list[int]'
  reserved_at_140: 'list[int]'
struct_mlx5_ifc_sampler_obj_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('table_type', (ctypes.c_ubyte * 8), 64), ('level', (ctypes.c_ubyte * 8), 72), ('reserved_at_50', (ctypes.c_ubyte * 15), 80), ('ignore_flow_level', (ctypes.c_ubyte * 1), 95), ('sample_ratio', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 8), 128), ('sample_table_id', (ctypes.c_ubyte * 24), 136), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('default_table_id', (ctypes.c_ubyte * 24), 168), ('sw_steering_icm_address_rx', (ctypes.c_ubyte * 64), 192), ('sw_steering_icm_address_tx', (ctypes.c_ubyte * 64), 256), ('reserved_at_140', (ctypes.c_ubyte * 160), 320)])
@c.record
class struct_mlx5_ifc_create_sampler_obj_in_bits(c.Struct):
  SIZE = 608
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  sampler_object: 'struct_mlx5_ifc_sampler_obj_bits'
struct_mlx5_ifc_create_sampler_obj_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('sampler_object', struct_mlx5_ifc_sampler_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_query_sampler_obj_out_bits(c.Struct):
  SIZE = 608
  general_obj_out_cmd_hdr: 'struct_mlx5_ifc_general_obj_out_cmd_hdr_bits'
  sampler_object: 'struct_mlx5_ifc_sampler_obj_bits'
struct_mlx5_ifc_query_sampler_obj_out_bits.register_fields([('general_obj_out_cmd_hdr', struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0), ('sampler_object', struct_mlx5_ifc_sampler_obj_bits, 128)])
_anonenum141: dict[int, str] = {(MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_KEY_SIZE_128:=0): 'MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_KEY_SIZE_128', (MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_KEY_SIZE_256:=1): 'MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_KEY_SIZE_256'}
_anonenum142: dict[int, str] = {(MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_TLS:=1): 'MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_TLS', (MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_IPSEC:=2): 'MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_IPSEC', (MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_MACSEC:=4): 'MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_MACSEC', (MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_PSP:=6): 'MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_PSP'}
@c.record
class struct_mlx5_ifc_tls_static_params_bits(c.Struct):
  SIZE = 512
  const_2: 'list[int]'
  tls_version: 'list[int]'
  const_1: 'list[int]'
  reserved_at_8: 'list[int]'
  encryption_standard: 'list[int]'
  reserved_at_20: 'list[int]'
  initial_record_number: 'list[int]'
  resync_tcp_sn: 'list[int]'
  gcm_iv: 'list[int]'
  implicit_iv: 'list[int]'
  reserved_at_100: 'list[int]'
  dek_index: 'list[int]'
  reserved_at_120: 'list[int]'
struct_mlx5_ifc_tls_static_params_bits.register_fields([('const_2', (ctypes.c_ubyte * 2), 0), ('tls_version', (ctypes.c_ubyte * 4), 2), ('const_1', (ctypes.c_ubyte * 2), 6), ('reserved_at_8', (ctypes.c_ubyte * 20), 8), ('encryption_standard', (ctypes.c_ubyte * 4), 28), ('reserved_at_20', (ctypes.c_ubyte * 32), 32), ('initial_record_number', (ctypes.c_ubyte * 64), 64), ('resync_tcp_sn', (ctypes.c_ubyte * 32), 128), ('gcm_iv', (ctypes.c_ubyte * 32), 160), ('implicit_iv', (ctypes.c_ubyte * 64), 192), ('reserved_at_100', (ctypes.c_ubyte * 8), 256), ('dek_index', (ctypes.c_ubyte * 24), 264), ('reserved_at_120', (ctypes.c_ubyte * 224), 288)])
@c.record
class struct_mlx5_ifc_tls_progress_params_bits(c.Struct):
  SIZE = 96
  next_record_tcp_sn: 'list[int]'
  hw_resync_tcp_sn: 'list[int]'
  record_tracker_state: 'list[int]'
  auth_state: 'list[int]'
  reserved_at_44: 'list[int]'
  hw_offset_record_number: 'list[int]'
struct_mlx5_ifc_tls_progress_params_bits.register_fields([('next_record_tcp_sn', (ctypes.c_ubyte * 32), 0), ('hw_resync_tcp_sn', (ctypes.c_ubyte * 32), 32), ('record_tracker_state', (ctypes.c_ubyte * 2), 64), ('auth_state', (ctypes.c_ubyte * 2), 66), ('reserved_at_44', (ctypes.c_ubyte * 4), 68), ('hw_offset_record_number', (ctypes.c_ubyte * 24), 72)])
_anonenum143: dict[int, str] = {(MLX5_MTT_PERM_READ:=1): 'MLX5_MTT_PERM_READ', (MLX5_MTT_PERM_WRITE:=2): 'MLX5_MTT_PERM_WRITE', (MLX5_MTT_PERM_RW:=3): 'MLX5_MTT_PERM_RW'}
_anonenum144: dict[int, str] = {(MLX5_SUSPEND_VHCA_IN_OP_MOD_SUSPEND_INITIATOR:=0): 'MLX5_SUSPEND_VHCA_IN_OP_MOD_SUSPEND_INITIATOR', (MLX5_SUSPEND_VHCA_IN_OP_MOD_SUSPEND_RESPONDER:=1): 'MLX5_SUSPEND_VHCA_IN_OP_MOD_SUSPEND_RESPONDER'}
@c.record
class struct_mlx5_ifc_suspend_vhca_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_suspend_vhca_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_suspend_vhca_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_suspend_vhca_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
_anonenum145: dict[int, str] = {(MLX5_RESUME_VHCA_IN_OP_MOD_RESUME_RESPONDER:=0): 'MLX5_RESUME_VHCA_IN_OP_MOD_RESUME_RESPONDER', (MLX5_RESUME_VHCA_IN_OP_MOD_RESUME_INITIATOR:=1): 'MLX5_RESUME_VHCA_IN_OP_MOD_RESUME_INITIATOR'}
@c.record
class struct_mlx5_ifc_resume_vhca_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_resume_vhca_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_resume_vhca_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_resume_vhca_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_query_vhca_migration_state_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  incremental: 'list[int]'
  chunk: 'list[int]'
  reserved_at_42: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
struct_mlx5_ifc_query_vhca_migration_state_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('incremental', (ctypes.c_ubyte * 1), 64), ('chunk', (ctypes.c_ubyte * 1), 65), ('reserved_at_42', (ctypes.c_ubyte * 14), 66), ('vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_query_vhca_migration_state_out_bits(c.Struct):
  SIZE = 512
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  required_umem_size: 'list[int]'
  reserved_at_a0: 'list[int]'
  remaining_total_size: 'list[int]'
  reserved_at_100: 'list[int]'
struct_mlx5_ifc_query_vhca_migration_state_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64), ('required_umem_size', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 32), 160), ('remaining_total_size', (ctypes.c_ubyte * 64), 192), ('reserved_at_100', (ctypes.c_ubyte * 256), 256)])
@c.record
class struct_mlx5_ifc_save_vhca_state_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  incremental: 'list[int]'
  set_track: 'list[int]'
  reserved_at_42: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
  va: 'list[int]'
  mkey: 'list[int]'
  size: 'list[int]'
struct_mlx5_ifc_save_vhca_state_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('incremental', (ctypes.c_ubyte * 1), 64), ('set_track', (ctypes.c_ubyte * 1), 65), ('reserved_at_42', (ctypes.c_ubyte * 14), 66), ('vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('va', (ctypes.c_ubyte * 64), 128), ('mkey', (ctypes.c_ubyte * 32), 192), ('size', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_save_vhca_state_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  actual_image_size: 'list[int]'
  next_required_umem_size: 'list[int]'
struct_mlx5_ifc_save_vhca_state_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('actual_image_size', (ctypes.c_ubyte * 32), 64), ('next_required_umem_size', (ctypes.c_ubyte * 32), 96)])
@c.record
class struct_mlx5_ifc_load_vhca_state_in_bits(c.Struct):
  SIZE = 256
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
  va: 'list[int]'
  mkey: 'list[int]'
  size: 'list[int]'
struct_mlx5_ifc_load_vhca_state_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('va', (ctypes.c_ubyte * 64), 128), ('mkey', (ctypes.c_ubyte * 32), 192), ('size', (ctypes.c_ubyte * 32), 224)])
@c.record
class struct_mlx5_ifc_load_vhca_state_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_load_vhca_state_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_adv_rdma_cap_bits(c.Struct):
  SIZE = 16384
  rdma_transport_manager: 'list[int]'
  rdma_transport_manager_other_eswitch: 'list[int]'
  reserved_at_2: 'list[int]'
  rcx_type: 'list[int]'
  reserved_at_28: 'list[int]'
  ps_entry_log_max_value: 'list[int]'
  reserved_at_30: 'list[int]'
  qp_max_ps_num_entry: 'list[int]'
  mp_max_num_queues: 'list[int]'
  ps_user_context_max_log_size: 'list[int]'
  message_based_qp_and_striding_wq: 'list[int]'
  reserved_at_58: 'list[int]'
  max_receive_send_message_size_stride: 'list[int]'
  reserved_at_70: 'list[int]'
  max_receive_send_message_size_byte: 'list[int]'
  reserved_at_a0: 'list[int]'
  rdma_transport_rx_flow_table_properties: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  rdma_transport_tx_flow_table_properties: 'struct_mlx5_ifc_flow_table_prop_layout_bits'
  rdma_transport_rx_ft_field_support_2: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  rdma_transport_tx_ft_field_support_2: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  rdma_transport_rx_ft_field_bitmask_support_2: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  rdma_transport_tx_ft_field_bitmask_support_2: 'struct_mlx5_ifc_flow_table_fields_supported_2_bits'
  reserved_at_800: 'list[int]'
struct_mlx5_ifc_adv_rdma_cap_bits.register_fields([('rdma_transport_manager', (ctypes.c_ubyte * 1), 0), ('rdma_transport_manager_other_eswitch', (ctypes.c_ubyte * 1), 1), ('reserved_at_2', (ctypes.c_ubyte * 30), 2), ('rcx_type', (ctypes.c_ubyte * 8), 32), ('reserved_at_28', (ctypes.c_ubyte * 2), 40), ('ps_entry_log_max_value', (ctypes.c_ubyte * 6), 42), ('reserved_at_30', (ctypes.c_ubyte * 6), 48), ('qp_max_ps_num_entry', (ctypes.c_ubyte * 10), 54), ('mp_max_num_queues', (ctypes.c_ubyte * 8), 64), ('ps_user_context_max_log_size', (ctypes.c_ubyte * 8), 72), ('message_based_qp_and_striding_wq', (ctypes.c_ubyte * 8), 80), ('reserved_at_58', (ctypes.c_ubyte * 8), 88), ('max_receive_send_message_size_stride', (ctypes.c_ubyte * 16), 96), ('reserved_at_70', (ctypes.c_ubyte * 16), 112), ('max_receive_send_message_size_byte', (ctypes.c_ubyte * 32), 128), ('reserved_at_a0', (ctypes.c_ubyte * 352), 160), ('rdma_transport_rx_flow_table_properties', struct_mlx5_ifc_flow_table_prop_layout_bits, 512), ('rdma_transport_tx_flow_table_properties', struct_mlx5_ifc_flow_table_prop_layout_bits, 1024), ('rdma_transport_rx_ft_field_support_2', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1536), ('rdma_transport_tx_ft_field_support_2', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1664), ('rdma_transport_rx_ft_field_bitmask_support_2', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1792), ('rdma_transport_tx_ft_field_bitmask_support_2', struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1920), ('reserved_at_800', (ctypes.c_ubyte * 14336), 2048)])
@c.record
class struct_mlx5_ifc_adv_virtualization_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: 'list[int]'
  pg_track_log_max_num: 'list[int]'
  pg_track_max_num_range: 'list[int]'
  pg_track_log_min_addr_space: 'list[int]'
  pg_track_log_max_addr_space: 'list[int]'
  reserved_at_20: 'list[int]'
  pg_track_log_min_msg_size: 'list[int]'
  reserved_at_28: 'list[int]'
  pg_track_log_max_msg_size: 'list[int]'
  reserved_at_30: 'list[int]'
  pg_track_log_min_page_size: 'list[int]'
  reserved_at_38: 'list[int]'
  pg_track_log_max_page_size: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_adv_virtualization_cap_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 3), 0), ('pg_track_log_max_num', (ctypes.c_ubyte * 5), 3), ('pg_track_max_num_range', (ctypes.c_ubyte * 8), 8), ('pg_track_log_min_addr_space', (ctypes.c_ubyte * 8), 16), ('pg_track_log_max_addr_space', (ctypes.c_ubyte * 8), 24), ('reserved_at_20', (ctypes.c_ubyte * 3), 32), ('pg_track_log_min_msg_size', (ctypes.c_ubyte * 5), 35), ('reserved_at_28', (ctypes.c_ubyte * 3), 40), ('pg_track_log_max_msg_size', (ctypes.c_ubyte * 5), 43), ('reserved_at_30', (ctypes.c_ubyte * 3), 48), ('pg_track_log_min_page_size', (ctypes.c_ubyte * 5), 51), ('reserved_at_38', (ctypes.c_ubyte * 3), 56), ('pg_track_log_max_page_size', (ctypes.c_ubyte * 5), 59), ('reserved_at_40', (ctypes.c_ubyte * 1984), 64)])
@c.record
class struct_mlx5_ifc_page_track_report_entry_bits(c.Struct):
  SIZE = 64
  dirty_address_high: 'list[int]'
  dirty_address_low: 'list[int]'
struct_mlx5_ifc_page_track_report_entry_bits.register_fields([('dirty_address_high', (ctypes.c_ubyte * 32), 0), ('dirty_address_low', (ctypes.c_ubyte * 32), 32)])
_anonenum146: dict[int, str] = {(MLX5_PAGE_TRACK_STATE_TRACKING:=0): 'MLX5_PAGE_TRACK_STATE_TRACKING', (MLX5_PAGE_TRACK_STATE_REPORTING:=1): 'MLX5_PAGE_TRACK_STATE_REPORTING', (MLX5_PAGE_TRACK_STATE_ERROR:=2): 'MLX5_PAGE_TRACK_STATE_ERROR'}
@c.record
class struct_mlx5_ifc_page_track_range_bits(c.Struct):
  SIZE = 128
  start_address: 'list[int]'
  length: 'list[int]'
struct_mlx5_ifc_page_track_range_bits.register_fields([('start_address', (ctypes.c_ubyte * 64), 0), ('length', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_page_track_bits(c.Struct):
  SIZE = 384
  modify_field_select: 'list[int]'
  reserved_at_40: 'list[int]'
  vhca_id: 'list[int]'
  reserved_at_60: 'list[int]'
  state: 'list[int]'
  track_type: 'list[int]'
  log_addr_space_size: 'list[int]'
  reserved_at_90: 'list[int]'
  log_page_size: 'list[int]'
  reserved_at_98: 'list[int]'
  log_msg_size: 'list[int]'
  reserved_at_a0: 'list[int]'
  reporting_qpn: 'list[int]'
  reserved_at_c0: 'list[int]'
  num_ranges: 'list[int]'
  reserved_at_e0: 'list[int]'
  range_start_address: 'list[int]'
  length: 'list[int]'
  track_range: 'list[struct_mlx5_ifc_page_track_range_bits]'
struct_mlx5_ifc_page_track_bits.register_fields([('modify_field_select', (ctypes.c_ubyte * 64), 0), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('vhca_id', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('state', (ctypes.c_ubyte * 4), 128), ('track_type', (ctypes.c_ubyte * 4), 132), ('log_addr_space_size', (ctypes.c_ubyte * 8), 136), ('reserved_at_90', (ctypes.c_ubyte * 3), 144), ('log_page_size', (ctypes.c_ubyte * 5), 147), ('reserved_at_98', (ctypes.c_ubyte * 3), 152), ('log_msg_size', (ctypes.c_ubyte * 5), 155), ('reserved_at_a0', (ctypes.c_ubyte * 8), 160), ('reporting_qpn', (ctypes.c_ubyte * 24), 168), ('reserved_at_c0', (ctypes.c_ubyte * 24), 192), ('num_ranges', (ctypes.c_ubyte * 8), 216), ('reserved_at_e0', (ctypes.c_ubyte * 32), 224), ('range_start_address', (ctypes.c_ubyte * 64), 256), ('length', (ctypes.c_ubyte * 64), 320), ('track_range', (struct_mlx5_ifc_page_track_range_bits * 0), 384)])
@c.record
class struct_mlx5_ifc_create_page_track_obj_in_bits(c.Struct):
  SIZE = 512
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  obj_context: 'struct_mlx5_ifc_page_track_bits'
struct_mlx5_ifc_create_page_track_obj_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('obj_context', struct_mlx5_ifc_page_track_bits, 128)])
@c.record
class struct_mlx5_ifc_modify_page_track_obj_in_bits(c.Struct):
  SIZE = 512
  general_obj_in_cmd_hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  obj_context: 'struct_mlx5_ifc_page_track_bits'
struct_mlx5_ifc_modify_page_track_obj_in_bits.register_fields([('general_obj_in_cmd_hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('obj_context', struct_mlx5_ifc_page_track_bits, 128)])
@c.record
class struct_mlx5_ifc_query_page_track_obj_out_bits(c.Struct):
  SIZE = 512
  general_obj_out_cmd_hdr: 'struct_mlx5_ifc_general_obj_out_cmd_hdr_bits'
  obj_context: 'struct_mlx5_ifc_page_track_bits'
struct_mlx5_ifc_query_page_track_obj_out_bits.register_fields([('general_obj_out_cmd_hdr', struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0), ('obj_context', struct_mlx5_ifc_page_track_bits, 128)])
@c.record
class struct_mlx5_ifc_msecq_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  reserved_at_20: 'list[int]'
  network_option: 'list[int]'
  local_ssm_code: 'list[int]'
  local_enhanced_ssm_code: 'list[int]'
  local_clock_identity: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_msecq_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 18), 32), ('network_option', (ctypes.c_ubyte * 2), 50), ('local_ssm_code', (ctypes.c_ubyte * 4), 52), ('local_enhanced_ssm_code', (ctypes.c_ubyte * 8), 56), ('local_clock_identity', (ctypes.c_ubyte * 64), 64), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
_anonenum147: dict[int, str] = {(MLX5_MSEES_FIELD_SELECT_ENABLE:=0): 'MLX5_MSEES_FIELD_SELECT_ENABLE', (MLX5_MSEES_FIELD_SELECT_ADMIN_STATUS:=1): 'MLX5_MSEES_FIELD_SELECT_ADMIN_STATUS', (MLX5_MSEES_FIELD_SELECT_ADMIN_FREQ_MEASURE:=2): 'MLX5_MSEES_FIELD_SELECT_ADMIN_FREQ_MEASURE'}
enum_mlx5_msees_admin_status: dict[int, str] = {(MLX5_MSEES_ADMIN_STATUS_FREE_RUNNING:=0): 'MLX5_MSEES_ADMIN_STATUS_FREE_RUNNING', (MLX5_MSEES_ADMIN_STATUS_TRACK:=1): 'MLX5_MSEES_ADMIN_STATUS_TRACK'}
enum_mlx5_msees_oper_status: dict[int, str] = {(MLX5_MSEES_OPER_STATUS_FREE_RUNNING:=0): 'MLX5_MSEES_OPER_STATUS_FREE_RUNNING', (MLX5_MSEES_OPER_STATUS_SELF_TRACK:=1): 'MLX5_MSEES_OPER_STATUS_SELF_TRACK', (MLX5_MSEES_OPER_STATUS_OTHER_TRACK:=2): 'MLX5_MSEES_OPER_STATUS_OTHER_TRACK', (MLX5_MSEES_OPER_STATUS_HOLDOVER:=3): 'MLX5_MSEES_OPER_STATUS_HOLDOVER', (MLX5_MSEES_OPER_STATUS_FAIL_HOLDOVER:=4): 'MLX5_MSEES_OPER_STATUS_FAIL_HOLDOVER', (MLX5_MSEES_OPER_STATUS_FAIL_FREE_RUNNING:=5): 'MLX5_MSEES_OPER_STATUS_FAIL_FREE_RUNNING'}
enum_mlx5_msees_failure_reason: dict[int, str] = {(MLX5_MSEES_FAILURE_REASON_UNDEFINED_ERROR:=0): 'MLX5_MSEES_FAILURE_REASON_UNDEFINED_ERROR', (MLX5_MSEES_FAILURE_REASON_PORT_DOWN:=1): 'MLX5_MSEES_FAILURE_REASON_PORT_DOWN', (MLX5_MSEES_FAILURE_REASON_TOO_HIGH_FREQUENCY_DIFF:=2): 'MLX5_MSEES_FAILURE_REASON_TOO_HIGH_FREQUENCY_DIFF', (MLX5_MSEES_FAILURE_REASON_NET_SYNCHRONIZER_DEVICE_ERROR:=3): 'MLX5_MSEES_FAILURE_REASON_NET_SYNCHRONIZER_DEVICE_ERROR', (MLX5_MSEES_FAILURE_REASON_LACK_OF_RESOURCES:=4): 'MLX5_MSEES_FAILURE_REASON_LACK_OF_RESOURCES'}
@c.record
class struct_mlx5_ifc_msees_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  local_port: 'list[int]'
  pnat: 'list[int]'
  lp_msb: 'list[int]'
  reserved_at_14: 'list[int]'
  field_select: 'list[int]'
  admin_status: 'list[int]'
  oper_status: 'list[int]'
  ho_acq: 'list[int]'
  reserved_at_49: 'list[int]'
  admin_freq_measure: 'list[int]'
  oper_freq_measure: 'list[int]'
  failure_reason: 'list[int]'
  frequency_diff: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_msees_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 8), 0), ('local_port', (ctypes.c_ubyte * 8), 8), ('pnat', (ctypes.c_ubyte * 2), 16), ('lp_msb', (ctypes.c_ubyte * 2), 18), ('reserved_at_14', (ctypes.c_ubyte * 12), 20), ('field_select', (ctypes.c_ubyte * 32), 32), ('admin_status', (ctypes.c_ubyte * 4), 64), ('oper_status', (ctypes.c_ubyte * 4), 68), ('ho_acq', (ctypes.c_ubyte * 1), 72), ('reserved_at_49', (ctypes.c_ubyte * 12), 73), ('admin_freq_measure', (ctypes.c_ubyte * 1), 85), ('oper_freq_measure', (ctypes.c_ubyte * 1), 86), ('failure_reason', (ctypes.c_ubyte * 9), 87), ('frequency_diff', (ctypes.c_ubyte * 32), 96), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
@c.record
class struct_mlx5_ifc_mrtcq_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: 'list[int]'
  rt_clock_identity: 'list[int]'
  reserved_at_80: 'list[int]'
struct_mlx5_ifc_mrtcq_reg_bits.register_fields([('reserved_at_0', (ctypes.c_ubyte * 64), 0), ('rt_clock_identity', (ctypes.c_ubyte * 64), 64), ('reserved_at_80', (ctypes.c_ubyte * 384), 128)])
@c.record
class struct_mlx5_ifc_pcie_cong_event_obj_bits(c.Struct):
  SIZE = 1024
  modify_select_field: 'list[int]'
  inbound_event_en: 'list[int]'
  outbound_event_en: 'list[int]'
  reserved_at_42: 'list[int]'
  reserved_at_60: 'list[int]'
  inbound_cong_state: 'list[int]'
  reserved_at_64: 'list[int]'
  outbound_cong_state: 'list[int]'
  reserved_at_68: 'list[int]'
  inbound_cong_low_threshold: 'list[int]'
  inbound_cong_high_threshold: 'list[int]'
  outbound_cong_low_threshold: 'list[int]'
  outbound_cong_high_threshold: 'list[int]'
  reserved_at_e0: 'list[int]'
struct_mlx5_ifc_pcie_cong_event_obj_bits.register_fields([('modify_select_field', (ctypes.c_ubyte * 64), 0), ('inbound_event_en', (ctypes.c_ubyte * 1), 64), ('outbound_event_en', (ctypes.c_ubyte * 1), 65), ('reserved_at_42', (ctypes.c_ubyte * 30), 66), ('reserved_at_60', (ctypes.c_ubyte * 1), 96), ('inbound_cong_state', (ctypes.c_ubyte * 3), 97), ('reserved_at_64', (ctypes.c_ubyte * 1), 100), ('outbound_cong_state', (ctypes.c_ubyte * 3), 101), ('reserved_at_68', (ctypes.c_ubyte * 24), 104), ('inbound_cong_low_threshold', (ctypes.c_ubyte * 16), 128), ('inbound_cong_high_threshold', (ctypes.c_ubyte * 16), 144), ('outbound_cong_low_threshold', (ctypes.c_ubyte * 16), 160), ('outbound_cong_high_threshold', (ctypes.c_ubyte * 16), 176), ('reserved_at_e0', (ctypes.c_ubyte * 832), 192)])
@c.record
class struct_mlx5_ifc_pcie_cong_event_cmd_in_bits(c.Struct):
  SIZE = 1152
  hdr: 'struct_mlx5_ifc_general_obj_in_cmd_hdr_bits'
  cong_obj: 'struct_mlx5_ifc_pcie_cong_event_obj_bits'
struct_mlx5_ifc_pcie_cong_event_cmd_in_bits.register_fields([('hdr', struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0), ('cong_obj', struct_mlx5_ifc_pcie_cong_event_obj_bits, 128)])
@c.record
class struct_mlx5_ifc_pcie_cong_event_cmd_out_bits(c.Struct):
  SIZE = 1152
  hdr: 'struct_mlx5_ifc_general_obj_out_cmd_hdr_bits'
  cong_obj: 'struct_mlx5_ifc_pcie_cong_event_obj_bits'
struct_mlx5_ifc_pcie_cong_event_cmd_out_bits.register_fields([('hdr', struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0), ('cong_obj', struct_mlx5_ifc_pcie_cong_event_obj_bits, 128)])
enum_mlx5e_pcie_cong_event_mod_field: dict[int, str] = {(MLX5_PCIE_CONG_EVENT_MOD_EVENT_EN:=0): 'MLX5_PCIE_CONG_EVENT_MOD_EVENT_EN', (MLX5_PCIE_CONG_EVENT_MOD_THRESH:=1): 'MLX5_PCIE_CONG_EVENT_MOD_THRESH'}
@c.record
class struct_mlx5_ifc_psp_rotate_key_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_psp_rotate_key_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
@c.record
class struct_mlx5_ifc_psp_rotate_key_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
struct_mlx5_ifc_psp_rotate_key_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 64), 64)])
enum_mlx5_psp_gen_spi_in_key_size: dict[int, str] = {(MLX5_PSP_GEN_SPI_IN_KEY_SIZE_128:=0): 'MLX5_PSP_GEN_SPI_IN_KEY_SIZE_128', (MLX5_PSP_GEN_SPI_IN_KEY_SIZE_256:=1): 'MLX5_PSP_GEN_SPI_IN_KEY_SIZE_256'}
@c.record
class struct_mlx5_ifc_key_spi_bits(c.Struct):
  SIZE = 384
  spi: 'list[int]'
  reserved_at_20: 'list[int]'
  key: 'list[list[int]]'
struct_mlx5_ifc_key_spi_bits.register_fields([('spi', (ctypes.c_ubyte * 32), 0), ('reserved_at_20', (ctypes.c_ubyte * 96), 32), ('key', ((ctypes.c_ubyte * 32) * 8), 128)])
@c.record
class struct_mlx5_ifc_psp_gen_spi_in_bits(c.Struct):
  SIZE = 128
  opcode: 'list[int]'
  uid: 'list[int]'
  reserved_at_20: 'list[int]'
  op_mod: 'list[int]'
  reserved_at_40: 'list[int]'
  key_size: 'list[int]'
  reserved_at_62: 'list[int]'
  num_of_spi: 'list[int]'
struct_mlx5_ifc_psp_gen_spi_in_bits.register_fields([('opcode', (ctypes.c_ubyte * 16), 0), ('uid', (ctypes.c_ubyte * 16), 16), ('reserved_at_20', (ctypes.c_ubyte * 16), 32), ('op_mod', (ctypes.c_ubyte * 16), 48), ('reserved_at_40', (ctypes.c_ubyte * 32), 64), ('key_size', (ctypes.c_ubyte * 2), 96), ('reserved_at_62', (ctypes.c_ubyte * 14), 98), ('num_of_spi', (ctypes.c_ubyte * 16), 112)])
@c.record
class struct_mlx5_ifc_psp_gen_spi_out_bits(c.Struct):
  SIZE = 128
  status: 'list[int]'
  reserved_at_8: 'list[int]'
  syndrome: 'list[int]'
  reserved_at_40: 'list[int]'
  num_of_spi: 'list[int]'
  reserved_at_60: 'list[int]'
  key_spi: 'list[struct_mlx5_ifc_key_spi_bits]'
struct_mlx5_ifc_psp_gen_spi_out_bits.register_fields([('status', (ctypes.c_ubyte * 8), 0), ('reserved_at_8', (ctypes.c_ubyte * 24), 8), ('syndrome', (ctypes.c_ubyte * 32), 32), ('reserved_at_40', (ctypes.c_ubyte * 16), 64), ('num_of_spi', (ctypes.c_ubyte * 16), 80), ('reserved_at_60', (ctypes.c_ubyte * 32), 96), ('key_spi', (struct_mlx5_ifc_key_spi_bits * 0), 128)])
MLX5_CMD_OP_QUERY_HCA_CAP = 0x100 # type: ignore
MLX5_CMD_OP_QUERY_ADAPTER = 0x101 # type: ignore
MLX5_CMD_OP_INIT_HCA = 0x102 # type: ignore
MLX5_CMD_OP_TEARDOWN_HCA = 0x103 # type: ignore
MLX5_CMD_OP_ENABLE_HCA = 0x104 # type: ignore
MLX5_CMD_OP_DISABLE_HCA = 0x105 # type: ignore
MLX5_CMD_OP_QUERY_PAGES = 0x107 # type: ignore
MLX5_CMD_OP_MANAGE_PAGES = 0x108 # type: ignore
MLX5_CMD_OP_SET_HCA_CAP = 0x109 # type: ignore
MLX5_CMD_OP_QUERY_ISSI = 0x10a # type: ignore
MLX5_CMD_OP_SET_ISSI = 0x10b # type: ignore
MLX5_CMD_OP_SET_DRIVER_VERSION = 0x10d # type: ignore
MLX5_CMD_OP_CREATE_MKEY = 0x200 # type: ignore
MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS = 0x203 # type: ignore
MLX5_CMD_OP_CREATE_EQ = 0x301 # type: ignore
MLX5_CMD_OP_DESTROY_EQ = 0x302 # type: ignore
MLX5_CMD_OP_CREATE_CQ = 0x400 # type: ignore
MLX5_CMD_OP_DESTROY_CQ = 0x401 # type: ignore
MLX5_CMD_OP_CREATE_QP = 0x500 # type: ignore
MLX5_CMD_OP_DESTROY_QP = 0x501 # type: ignore
MLX5_CMD_OP_RST2INIT_QP = 0x502 # type: ignore
MLX5_CMD_OP_INIT2RTR_QP = 0x503 # type: ignore
MLX5_CMD_OP_RTR2RTS_QP = 0x504 # type: ignore
MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT = 0x754 # type: ignore
MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT = 0x755 # type: ignore
MLX5_CMD_OP_SET_ROCE_ADDRESS = 0x761 # type: ignore
MLX5_CMD_OP_ALLOC_PD = 0x800 # type: ignore
MLX5_CMD_OP_ALLOC_UAR = 0x802 # type: ignore
MLX5_CMD_OP_ACCESS_REG = 0x805 # type: ignore
MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN = 0x816 # type: ignore
MLX5_CMD_STAT_OK = 0x0 # type: ignore
MLX5_CMD_STAT_INT_ERR = 0x1 # type: ignore
MLX5_CMD_STAT_BAD_OP_ERR = 0x2 # type: ignore
MLX5_CMD_STAT_BAD_PARAM_ERR = 0x3 # type: ignore
MLX5_CMD_STAT_BAD_SYS_STATE_ERR = 0x4 # type: ignore
MLX5_CMD_STAT_BAD_RES_ERR = 0x5 # type: ignore
MLX5_CMD_STAT_RES_BUSY = 0x6 # type: ignore
MLX5_CMD_STAT_LIM_ERR = 0x8 # type: ignore
MLX5_CMD_STAT_BAD_RES_STATE_ERR = 0x9 # type: ignore
MLX5_CMD_STAT_NO_RES_ERR = 0xf # type: ignore
MLX5_CMD_STAT_BAD_INP_LEN_ERR = 0x50 # type: ignore
MLX5_CMD_STAT_BAD_OUTP_LEN_ERR = 0x51 # type: ignore
MLX5_CAP_GENERAL = 0x0 # type: ignore
MLX5_CAP_ODP = 0x2 # type: ignore
MLX5_CAP_ATOMIC = 0x3 # type: ignore
MLX5_CAP_ROCE = 0x4 # type: ignore
HCA_CAP_OPMOD_GET_MAX = 0 # type: ignore
HCA_CAP_OPMOD_GET_CUR = 1 # type: ignore
MLX5_PAGES_GIVE = 1 # type: ignore
MLX5_PAGES_TAKE = 2 # type: ignore
MLX5_BOOT_PAGES = 1 # type: ignore
MLX5_INIT_PAGES = 2 # type: ignore
MLX5_REG_HOST_ENDIANNESS = 0x7004 # type: ignore
MLX5_REG_DTOR = 0xC00E # type: ignore
MLX5_PCI_CMD_XPORT = 0x07 # type: ignore
MLX5_CMD_DATA_BLOCK_SIZE = 512 # type: ignore
CMD_OWNER_HW = 0x01 # type: ignore
CAP_GEN_ABS_NATIVE_PORT_NUM = 0x007 # type: ignore
CAP_GEN_HCA_CAP_2 = 0x020 # type: ignore
CAP_GEN_EVENT_ON_VHCA_STATE_ALLOCATED = 0x023 # type: ignore
CAP_GEN_EVENT_ON_VHCA_STATE_ACTIVE = 0x024 # type: ignore
CAP_GEN_EVENT_ON_VHCA_STATE_IN_USE = 0x025 # type: ignore
CAP_GEN_EVENT_ON_VHCA_STATE_TEARDOWN_REQUEST = 0x026 # type: ignore
CAP_GEN_LOG_MAX_QP = 0x09B # type: ignore
CAP_GEN_LOG_MAX_CQ = 0x0DB # type: ignore
CAP_GEN_RELEASE_ALL_PAGES = 0x145 # type: ignore
CAP_GEN_CACHE_LINE_128BYTE = 0x164 # type: ignore
CAP_GEN_NUM_PORTS = 0x1B8 # type: ignore
CAP_GEN_PKEY_TABLE_SIZE = 0x190 # type: ignore
CAP_GEN_PCI_SYNC_FOR_FW_UPDATE_EVENT = 0x1F1 # type: ignore
CAP_GEN_CMDIF_CHECKSUM = 0x210 # type: ignore
CAP_GEN_DCT = 0x21A # type: ignore
CAP_GEN_ROCE = 0x21D # type: ignore
CAP_GEN_ATOMIC = 0x21E # type: ignore
CAP_GEN_ODP = 0x227 # type: ignore
CAP_GEN_MKEY_BY_NAME = 0x266 # type: ignore
CAP_GEN_LOG_MAX_PD = 0x32B # type: ignore
CAP_GEN_PCIE_RESET_USING_HOTRESET = 0x335 # type: ignore
CAP_GEN_PCI_SYNC_FOR_FW_UPDATE_WITH_DRIVER_UNLOAD = 0x336 # type: ignore
CAP_GEN_VHCA_STATE = 0x3EA # type: ignore
CAP_GEN_ROCE_RW_SUPPORTED = 0x3A1 # type: ignore
CAP_GEN_LOG_MAX_CURRENT_UC_LIST = 0x3FB # type: ignore
CAP_GEN_LOG_UAR_PAGE_SZ = 0x490 # type: ignore
CAP_GEN_NUM_VHCA_PORTS = 0x610 # type: ignore
CAP_GEN_SW_OWNER_ID = 0x61E # type: ignore
CAP_GEN_NUM_TOTAL_DYNAMIC_VF_MSIX = 0x708 # type: ignore
MLX5_FC_BULK_SIZE_FACTOR = 128 # type: ignore
MLX5_FC_BULK_NUM_FCS = lambda fc_enum: (MLX5_FC_BULK_SIZE_FACTOR * (fc_enum)) # type: ignore
MLX5_FT_MAX_MULTIPATH_LEVEL = 63 # type: ignore
MLX5_CMD_SET_MONITOR_NUM_PPCNT_COUNTER_SET1 = (6) # type: ignore
MLX5_CMD_SET_MONITOR_NUM_Q_COUNTERS_SET1 = (1) # type: ignore
MLX5_CMD_SET_MONITOR_NUM_COUNTER = (MLX5_CMD_SET_MONITOR_NUM_PPCNT_COUNTER_SET1 + MLX5_CMD_SET_MONITOR_NUM_Q_COUNTERS_SET1) # type: ignore
MLX5_IFC_DEFINER_FORMAT_OFFSET_UNUSED = 0x0 # type: ignore
MLX5_IFC_DEFINER_FORMAT_OFFSET_OUTER_ETH_PKT_LEN = 0x48 # type: ignore
MLX5_IFC_DEFINER_DW_SELECTORS_NUM = 9 # type: ignore
MLX5_IFC_DEFINER_BYTE_SELECTORS_NUM = 8 # type: ignore
MLX5_MACSEC_ASO_INC_SN = 0x2 # type: ignore
MLX5_MACSEC_ASO_REG_C_4_5 = 0x2 # type: ignore