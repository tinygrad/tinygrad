# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
__u8: TypeAlias = Annotated[int, ctypes.c_ubyte]
@c.record
class struct_mlx5_cmd_layout(c.Struct):
  SIZE = 64
  type: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  rsvd0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1]
  inlen: Annotated[Annotated[int, ctypes.c_uint32], 4]
  in_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  _in: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 16]
  out: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 32]
  out_ptr: Annotated[Annotated[int, ctypes.c_uint64], 48]
  outlen: Annotated[Annotated[int, ctypes.c_uint32], 56]
  token: Annotated[Annotated[int, ctypes.c_ubyte], 60]
  sig: Annotated[Annotated[int, ctypes.c_ubyte], 61]
  rsvd1: Annotated[Annotated[int, ctypes.c_ubyte], 62]
  status_own: Annotated[Annotated[int, ctypes.c_ubyte], 63]
@c.record
class struct_mlx5_cmd_prot_block(c.Struct):
  SIZE = 576
  data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[512]], 0]
  rsvd0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[48]], 512]
  next: Annotated[Annotated[int, ctypes.c_uint64], 560]
  block_num: Annotated[Annotated[int, ctypes.c_uint32], 568]
  rsvd1: Annotated[Annotated[int, ctypes.c_ubyte], 572]
  token: Annotated[Annotated[int, ctypes.c_ubyte], 573]
  ctrl_sig: Annotated[Annotated[int, ctypes.c_ubyte], 574]
  sig: Annotated[Annotated[int, ctypes.c_ubyte], 575]
@c.record
class struct_mlx5_init_seg(c.Struct):
  SIZE = 512
  fw_rev: Annotated[Annotated[int, ctypes.c_uint32], 0]
  cmdif_rev_fw_sub: Annotated[Annotated[int, ctypes.c_uint32], 4]
  rsvd0: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 8]
  cmdq_addr_h: Annotated[Annotated[int, ctypes.c_uint32], 16]
  cmdq_addr_l_sz: Annotated[Annotated[int, ctypes.c_uint32], 20]
  cmd_dbell: Annotated[Annotated[int, ctypes.c_uint32], 24]
  rsvd1: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[120]], 28]
  initializing: Annotated[Annotated[int, ctypes.c_uint32], 508]
class _anonenum0(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_EVENT_TYPE_CODING_COMPLETION_EVENTS = _anonenum0.define('MLX5_EVENT_TYPE_CODING_COMPLETION_EVENTS', 0)
MLX5_EVENT_TYPE_CODING_PATH_MIGRATED_SUCCEEDED = _anonenum0.define('MLX5_EVENT_TYPE_CODING_PATH_MIGRATED_SUCCEEDED', 1)
MLX5_EVENT_TYPE_CODING_COMMUNICATION_ESTABLISHED = _anonenum0.define('MLX5_EVENT_TYPE_CODING_COMMUNICATION_ESTABLISHED', 2)
MLX5_EVENT_TYPE_CODING_SEND_QUEUE_DRAINED = _anonenum0.define('MLX5_EVENT_TYPE_CODING_SEND_QUEUE_DRAINED', 3)
MLX5_EVENT_TYPE_CODING_LAST_WQE_REACHED = _anonenum0.define('MLX5_EVENT_TYPE_CODING_LAST_WQE_REACHED', 19)
MLX5_EVENT_TYPE_CODING_SRQ_LIMIT = _anonenum0.define('MLX5_EVENT_TYPE_CODING_SRQ_LIMIT', 20)
MLX5_EVENT_TYPE_CODING_DCT_ALL_CONNECTIONS_CLOSED = _anonenum0.define('MLX5_EVENT_TYPE_CODING_DCT_ALL_CONNECTIONS_CLOSED', 28)
MLX5_EVENT_TYPE_CODING_DCT_ACCESS_KEY_VIOLATION = _anonenum0.define('MLX5_EVENT_TYPE_CODING_DCT_ACCESS_KEY_VIOLATION', 29)
MLX5_EVENT_TYPE_CODING_CQ_ERROR = _anonenum0.define('MLX5_EVENT_TYPE_CODING_CQ_ERROR', 4)
MLX5_EVENT_TYPE_CODING_LOCAL_WQ_CATASTROPHIC_ERROR = _anonenum0.define('MLX5_EVENT_TYPE_CODING_LOCAL_WQ_CATASTROPHIC_ERROR', 5)
MLX5_EVENT_TYPE_CODING_PATH_MIGRATION_FAILED = _anonenum0.define('MLX5_EVENT_TYPE_CODING_PATH_MIGRATION_FAILED', 7)
MLX5_EVENT_TYPE_CODING_PAGE_FAULT_EVENT = _anonenum0.define('MLX5_EVENT_TYPE_CODING_PAGE_FAULT_EVENT', 12)
MLX5_EVENT_TYPE_CODING_INVALID_REQUEST_LOCAL_WQ_ERROR = _anonenum0.define('MLX5_EVENT_TYPE_CODING_INVALID_REQUEST_LOCAL_WQ_ERROR', 16)
MLX5_EVENT_TYPE_CODING_LOCAL_ACCESS_VIOLATION_WQ_ERROR = _anonenum0.define('MLX5_EVENT_TYPE_CODING_LOCAL_ACCESS_VIOLATION_WQ_ERROR', 17)
MLX5_EVENT_TYPE_CODING_LOCAL_SRQ_CATASTROPHIC_ERROR = _anonenum0.define('MLX5_EVENT_TYPE_CODING_LOCAL_SRQ_CATASTROPHIC_ERROR', 18)
MLX5_EVENT_TYPE_CODING_INTERNAL_ERROR = _anonenum0.define('MLX5_EVENT_TYPE_CODING_INTERNAL_ERROR', 8)
MLX5_EVENT_TYPE_CODING_PORT_STATE_CHANGE = _anonenum0.define('MLX5_EVENT_TYPE_CODING_PORT_STATE_CHANGE', 9)
MLX5_EVENT_TYPE_CODING_GPIO_EVENT = _anonenum0.define('MLX5_EVENT_TYPE_CODING_GPIO_EVENT', 21)
MLX5_EVENT_TYPE_CODING_REMOTE_CONFIGURATION_PROTOCOL_EVENT = _anonenum0.define('MLX5_EVENT_TYPE_CODING_REMOTE_CONFIGURATION_PROTOCOL_EVENT', 25)
MLX5_EVENT_TYPE_CODING_DOORBELL_BLUEFLAME_CONGESTION_EVENT = _anonenum0.define('MLX5_EVENT_TYPE_CODING_DOORBELL_BLUEFLAME_CONGESTION_EVENT', 26)
MLX5_EVENT_TYPE_CODING_STALL_VL_EVENT = _anonenum0.define('MLX5_EVENT_TYPE_CODING_STALL_VL_EVENT', 27)
MLX5_EVENT_TYPE_CODING_DROPPED_PACKET_LOGGED_EVENT = _anonenum0.define('MLX5_EVENT_TYPE_CODING_DROPPED_PACKET_LOGGED_EVENT', 31)
MLX5_EVENT_TYPE_CODING_COMMAND_INTERFACE_COMPLETION = _anonenum0.define('MLX5_EVENT_TYPE_CODING_COMMAND_INTERFACE_COMPLETION', 10)
MLX5_EVENT_TYPE_CODING_PAGE_REQUEST = _anonenum0.define('MLX5_EVENT_TYPE_CODING_PAGE_REQUEST', 11)
MLX5_EVENT_TYPE_CODING_FPGA_ERROR = _anonenum0.define('MLX5_EVENT_TYPE_CODING_FPGA_ERROR', 32)
MLX5_EVENT_TYPE_CODING_FPGA_QP_ERROR = _anonenum0.define('MLX5_EVENT_TYPE_CODING_FPGA_QP_ERROR', 33)

class _anonenum1(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE = _anonenum1.define('MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE', 0)
MLX5_SET_HCA_CAP_OP_MOD_ETHERNET_OFFLOADS = _anonenum1.define('MLX5_SET_HCA_CAP_OP_MOD_ETHERNET_OFFLOADS', 1)
MLX5_SET_HCA_CAP_OP_MOD_ODP = _anonenum1.define('MLX5_SET_HCA_CAP_OP_MOD_ODP', 2)
MLX5_SET_HCA_CAP_OP_MOD_ATOMIC = _anonenum1.define('MLX5_SET_HCA_CAP_OP_MOD_ATOMIC', 3)
MLX5_SET_HCA_CAP_OP_MOD_ROCE = _anonenum1.define('MLX5_SET_HCA_CAP_OP_MOD_ROCE', 4)
MLX5_SET_HCA_CAP_OP_MOD_IPSEC = _anonenum1.define('MLX5_SET_HCA_CAP_OP_MOD_IPSEC', 21)
MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE2 = _anonenum1.define('MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE2', 32)
MLX5_SET_HCA_CAP_OP_MOD_PORT_SELECTION = _anonenum1.define('MLX5_SET_HCA_CAP_OP_MOD_PORT_SELECTION', 37)

class _anonenum2(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_SHARED_RESOURCE_UID = _anonenum2.define('MLX5_SHARED_RESOURCE_UID', 65535)

class _anonenum3(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_OBJ_TYPE_SW_ICM = _anonenum3.define('MLX5_OBJ_TYPE_SW_ICM', 8)
MLX5_OBJ_TYPE_GENEVE_TLV_OPT = _anonenum3.define('MLX5_OBJ_TYPE_GENEVE_TLV_OPT', 11)
MLX5_OBJ_TYPE_VIRTIO_NET_Q = _anonenum3.define('MLX5_OBJ_TYPE_VIRTIO_NET_Q', 13)
MLX5_OBJ_TYPE_VIRTIO_Q_COUNTERS = _anonenum3.define('MLX5_OBJ_TYPE_VIRTIO_Q_COUNTERS', 28)
MLX5_OBJ_TYPE_MATCH_DEFINER = _anonenum3.define('MLX5_OBJ_TYPE_MATCH_DEFINER', 24)
MLX5_OBJ_TYPE_HEADER_MODIFY_ARGUMENT = _anonenum3.define('MLX5_OBJ_TYPE_HEADER_MODIFY_ARGUMENT', 35)
MLX5_OBJ_TYPE_STC = _anonenum3.define('MLX5_OBJ_TYPE_STC', 64)
MLX5_OBJ_TYPE_RTC = _anonenum3.define('MLX5_OBJ_TYPE_RTC', 65)
MLX5_OBJ_TYPE_STE = _anonenum3.define('MLX5_OBJ_TYPE_STE', 66)
MLX5_OBJ_TYPE_MODIFY_HDR_PATTERN = _anonenum3.define('MLX5_OBJ_TYPE_MODIFY_HDR_PATTERN', 67)
MLX5_OBJ_TYPE_PAGE_TRACK = _anonenum3.define('MLX5_OBJ_TYPE_PAGE_TRACK', 70)
MLX5_OBJ_TYPE_MKEY = _anonenum3.define('MLX5_OBJ_TYPE_MKEY', 65281)
MLX5_OBJ_TYPE_QP = _anonenum3.define('MLX5_OBJ_TYPE_QP', 65282)
MLX5_OBJ_TYPE_PSV = _anonenum3.define('MLX5_OBJ_TYPE_PSV', 65283)
MLX5_OBJ_TYPE_RMP = _anonenum3.define('MLX5_OBJ_TYPE_RMP', 65284)
MLX5_OBJ_TYPE_XRC_SRQ = _anonenum3.define('MLX5_OBJ_TYPE_XRC_SRQ', 65285)
MLX5_OBJ_TYPE_RQ = _anonenum3.define('MLX5_OBJ_TYPE_RQ', 65286)
MLX5_OBJ_TYPE_SQ = _anonenum3.define('MLX5_OBJ_TYPE_SQ', 65287)
MLX5_OBJ_TYPE_TIR = _anonenum3.define('MLX5_OBJ_TYPE_TIR', 65288)
MLX5_OBJ_TYPE_TIS = _anonenum3.define('MLX5_OBJ_TYPE_TIS', 65289)
MLX5_OBJ_TYPE_DCT = _anonenum3.define('MLX5_OBJ_TYPE_DCT', 65290)
MLX5_OBJ_TYPE_XRQ = _anonenum3.define('MLX5_OBJ_TYPE_XRQ', 65291)
MLX5_OBJ_TYPE_RQT = _anonenum3.define('MLX5_OBJ_TYPE_RQT', 65294)
MLX5_OBJ_TYPE_FLOW_COUNTER = _anonenum3.define('MLX5_OBJ_TYPE_FLOW_COUNTER', 65295)
MLX5_OBJ_TYPE_CQ = _anonenum3.define('MLX5_OBJ_TYPE_CQ', 65296)
MLX5_OBJ_TYPE_FT_ALIAS = _anonenum3.define('MLX5_OBJ_TYPE_FT_ALIAS', 65301)

class _anonenum4(Annotated[int, ctypes.c_uint64], c.Enum): pass
MLX5_GENERAL_OBJ_TYPES_CAP_SW_ICM = _anonenum4.define('MLX5_GENERAL_OBJ_TYPES_CAP_SW_ICM', 256)
MLX5_GENERAL_OBJ_TYPES_CAP_GENEVE_TLV_OPT = _anonenum4.define('MLX5_GENERAL_OBJ_TYPES_CAP_GENEVE_TLV_OPT', 2048)
MLX5_GENERAL_OBJ_TYPES_CAP_VIRTIO_NET_Q = _anonenum4.define('MLX5_GENERAL_OBJ_TYPES_CAP_VIRTIO_NET_Q', 8192)
MLX5_GENERAL_OBJ_TYPES_CAP_HEADER_MODIFY_ARGUMENT = _anonenum4.define('MLX5_GENERAL_OBJ_TYPES_CAP_HEADER_MODIFY_ARGUMENT', 34359738368)
MLX5_GENERAL_OBJ_TYPES_CAP_MACSEC_OFFLOAD = _anonenum4.define('MLX5_GENERAL_OBJ_TYPES_CAP_MACSEC_OFFLOAD', 549755813888)

class _anonenum5(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CMD_OP_QUERY_HCA_CAP = _anonenum5.define('MLX5_CMD_OP_QUERY_HCA_CAP', 256)
MLX5_CMD_OP_QUERY_ADAPTER = _anonenum5.define('MLX5_CMD_OP_QUERY_ADAPTER', 257)
MLX5_CMD_OP_INIT_HCA = _anonenum5.define('MLX5_CMD_OP_INIT_HCA', 258)
MLX5_CMD_OP_TEARDOWN_HCA = _anonenum5.define('MLX5_CMD_OP_TEARDOWN_HCA', 259)
MLX5_CMD_OP_ENABLE_HCA = _anonenum5.define('MLX5_CMD_OP_ENABLE_HCA', 260)
MLX5_CMD_OP_DISABLE_HCA = _anonenum5.define('MLX5_CMD_OP_DISABLE_HCA', 261)
MLX5_CMD_OP_QUERY_PAGES = _anonenum5.define('MLX5_CMD_OP_QUERY_PAGES', 263)
MLX5_CMD_OP_MANAGE_PAGES = _anonenum5.define('MLX5_CMD_OP_MANAGE_PAGES', 264)
MLX5_CMD_OP_SET_HCA_CAP = _anonenum5.define('MLX5_CMD_OP_SET_HCA_CAP', 265)
MLX5_CMD_OP_QUERY_ISSI = _anonenum5.define('MLX5_CMD_OP_QUERY_ISSI', 266)
MLX5_CMD_OP_SET_ISSI = _anonenum5.define('MLX5_CMD_OP_SET_ISSI', 267)
MLX5_CMD_OP_SET_DRIVER_VERSION = _anonenum5.define('MLX5_CMD_OP_SET_DRIVER_VERSION', 269)
MLX5_CMD_OP_QUERY_SF_PARTITION = _anonenum5.define('MLX5_CMD_OP_QUERY_SF_PARTITION', 273)
MLX5_CMD_OP_ALLOC_SF = _anonenum5.define('MLX5_CMD_OP_ALLOC_SF', 275)
MLX5_CMD_OP_DEALLOC_SF = _anonenum5.define('MLX5_CMD_OP_DEALLOC_SF', 276)
MLX5_CMD_OP_SUSPEND_VHCA = _anonenum5.define('MLX5_CMD_OP_SUSPEND_VHCA', 277)
MLX5_CMD_OP_RESUME_VHCA = _anonenum5.define('MLX5_CMD_OP_RESUME_VHCA', 278)
MLX5_CMD_OP_QUERY_VHCA_MIGRATION_STATE = _anonenum5.define('MLX5_CMD_OP_QUERY_VHCA_MIGRATION_STATE', 279)
MLX5_CMD_OP_SAVE_VHCA_STATE = _anonenum5.define('MLX5_CMD_OP_SAVE_VHCA_STATE', 280)
MLX5_CMD_OP_LOAD_VHCA_STATE = _anonenum5.define('MLX5_CMD_OP_LOAD_VHCA_STATE', 281)
MLX5_CMD_OP_CREATE_MKEY = _anonenum5.define('MLX5_CMD_OP_CREATE_MKEY', 512)
MLX5_CMD_OP_QUERY_MKEY = _anonenum5.define('MLX5_CMD_OP_QUERY_MKEY', 513)
MLX5_CMD_OP_DESTROY_MKEY = _anonenum5.define('MLX5_CMD_OP_DESTROY_MKEY', 514)
MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS = _anonenum5.define('MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS', 515)
MLX5_CMD_OP_PAGE_FAULT_RESUME = _anonenum5.define('MLX5_CMD_OP_PAGE_FAULT_RESUME', 516)
MLX5_CMD_OP_ALLOC_MEMIC = _anonenum5.define('MLX5_CMD_OP_ALLOC_MEMIC', 517)
MLX5_CMD_OP_DEALLOC_MEMIC = _anonenum5.define('MLX5_CMD_OP_DEALLOC_MEMIC', 518)
MLX5_CMD_OP_MODIFY_MEMIC = _anonenum5.define('MLX5_CMD_OP_MODIFY_MEMIC', 519)
MLX5_CMD_OP_CREATE_EQ = _anonenum5.define('MLX5_CMD_OP_CREATE_EQ', 769)
MLX5_CMD_OP_DESTROY_EQ = _anonenum5.define('MLX5_CMD_OP_DESTROY_EQ', 770)
MLX5_CMD_OP_QUERY_EQ = _anonenum5.define('MLX5_CMD_OP_QUERY_EQ', 771)
MLX5_CMD_OP_GEN_EQE = _anonenum5.define('MLX5_CMD_OP_GEN_EQE', 772)
MLX5_CMD_OP_CREATE_CQ = _anonenum5.define('MLX5_CMD_OP_CREATE_CQ', 1024)
MLX5_CMD_OP_DESTROY_CQ = _anonenum5.define('MLX5_CMD_OP_DESTROY_CQ', 1025)
MLX5_CMD_OP_QUERY_CQ = _anonenum5.define('MLX5_CMD_OP_QUERY_CQ', 1026)
MLX5_CMD_OP_MODIFY_CQ = _anonenum5.define('MLX5_CMD_OP_MODIFY_CQ', 1027)
MLX5_CMD_OP_CREATE_QP = _anonenum5.define('MLX5_CMD_OP_CREATE_QP', 1280)
MLX5_CMD_OP_DESTROY_QP = _anonenum5.define('MLX5_CMD_OP_DESTROY_QP', 1281)
MLX5_CMD_OP_RST2INIT_QP = _anonenum5.define('MLX5_CMD_OP_RST2INIT_QP', 1282)
MLX5_CMD_OP_INIT2RTR_QP = _anonenum5.define('MLX5_CMD_OP_INIT2RTR_QP', 1283)
MLX5_CMD_OP_RTR2RTS_QP = _anonenum5.define('MLX5_CMD_OP_RTR2RTS_QP', 1284)
MLX5_CMD_OP_RTS2RTS_QP = _anonenum5.define('MLX5_CMD_OP_RTS2RTS_QP', 1285)
MLX5_CMD_OP_SQERR2RTS_QP = _anonenum5.define('MLX5_CMD_OP_SQERR2RTS_QP', 1286)
MLX5_CMD_OP_2ERR_QP = _anonenum5.define('MLX5_CMD_OP_2ERR_QP', 1287)
MLX5_CMD_OP_2RST_QP = _anonenum5.define('MLX5_CMD_OP_2RST_QP', 1290)
MLX5_CMD_OP_QUERY_QP = _anonenum5.define('MLX5_CMD_OP_QUERY_QP', 1291)
MLX5_CMD_OP_SQD_RTS_QP = _anonenum5.define('MLX5_CMD_OP_SQD_RTS_QP', 1292)
MLX5_CMD_OP_INIT2INIT_QP = _anonenum5.define('MLX5_CMD_OP_INIT2INIT_QP', 1294)
MLX5_CMD_OP_CREATE_PSV = _anonenum5.define('MLX5_CMD_OP_CREATE_PSV', 1536)
MLX5_CMD_OP_DESTROY_PSV = _anonenum5.define('MLX5_CMD_OP_DESTROY_PSV', 1537)
MLX5_CMD_OP_CREATE_SRQ = _anonenum5.define('MLX5_CMD_OP_CREATE_SRQ', 1792)
MLX5_CMD_OP_DESTROY_SRQ = _anonenum5.define('MLX5_CMD_OP_DESTROY_SRQ', 1793)
MLX5_CMD_OP_QUERY_SRQ = _anonenum5.define('MLX5_CMD_OP_QUERY_SRQ', 1794)
MLX5_CMD_OP_ARM_RQ = _anonenum5.define('MLX5_CMD_OP_ARM_RQ', 1795)
MLX5_CMD_OP_CREATE_XRC_SRQ = _anonenum5.define('MLX5_CMD_OP_CREATE_XRC_SRQ', 1797)
MLX5_CMD_OP_DESTROY_XRC_SRQ = _anonenum5.define('MLX5_CMD_OP_DESTROY_XRC_SRQ', 1798)
MLX5_CMD_OP_QUERY_XRC_SRQ = _anonenum5.define('MLX5_CMD_OP_QUERY_XRC_SRQ', 1799)
MLX5_CMD_OP_ARM_XRC_SRQ = _anonenum5.define('MLX5_CMD_OP_ARM_XRC_SRQ', 1800)
MLX5_CMD_OP_CREATE_DCT = _anonenum5.define('MLX5_CMD_OP_CREATE_DCT', 1808)
MLX5_CMD_OP_DESTROY_DCT = _anonenum5.define('MLX5_CMD_OP_DESTROY_DCT', 1809)
MLX5_CMD_OP_DRAIN_DCT = _anonenum5.define('MLX5_CMD_OP_DRAIN_DCT', 1810)
MLX5_CMD_OP_QUERY_DCT = _anonenum5.define('MLX5_CMD_OP_QUERY_DCT', 1811)
MLX5_CMD_OP_ARM_DCT_FOR_KEY_VIOLATION = _anonenum5.define('MLX5_CMD_OP_ARM_DCT_FOR_KEY_VIOLATION', 1812)
MLX5_CMD_OP_CREATE_XRQ = _anonenum5.define('MLX5_CMD_OP_CREATE_XRQ', 1815)
MLX5_CMD_OP_DESTROY_XRQ = _anonenum5.define('MLX5_CMD_OP_DESTROY_XRQ', 1816)
MLX5_CMD_OP_QUERY_XRQ = _anonenum5.define('MLX5_CMD_OP_QUERY_XRQ', 1817)
MLX5_CMD_OP_ARM_XRQ = _anonenum5.define('MLX5_CMD_OP_ARM_XRQ', 1818)
MLX5_CMD_OP_QUERY_XRQ_DC_PARAMS_ENTRY = _anonenum5.define('MLX5_CMD_OP_QUERY_XRQ_DC_PARAMS_ENTRY', 1829)
MLX5_CMD_OP_SET_XRQ_DC_PARAMS_ENTRY = _anonenum5.define('MLX5_CMD_OP_SET_XRQ_DC_PARAMS_ENTRY', 1830)
MLX5_CMD_OP_QUERY_XRQ_ERROR_PARAMS = _anonenum5.define('MLX5_CMD_OP_QUERY_XRQ_ERROR_PARAMS', 1831)
MLX5_CMD_OP_RELEASE_XRQ_ERROR = _anonenum5.define('MLX5_CMD_OP_RELEASE_XRQ_ERROR', 1833)
MLX5_CMD_OP_MODIFY_XRQ = _anonenum5.define('MLX5_CMD_OP_MODIFY_XRQ', 1834)
MLX5_CMD_OPCODE_QUERY_DELEGATED_VHCA = _anonenum5.define('MLX5_CMD_OPCODE_QUERY_DELEGATED_VHCA', 1842)
MLX5_CMD_OPCODE_CREATE_ESW_VPORT = _anonenum5.define('MLX5_CMD_OPCODE_CREATE_ESW_VPORT', 1843)
MLX5_CMD_OPCODE_DESTROY_ESW_VPORT = _anonenum5.define('MLX5_CMD_OPCODE_DESTROY_ESW_VPORT', 1844)
MLX5_CMD_OP_QUERY_ESW_FUNCTIONS = _anonenum5.define('MLX5_CMD_OP_QUERY_ESW_FUNCTIONS', 1856)
MLX5_CMD_OP_QUERY_VPORT_STATE = _anonenum5.define('MLX5_CMD_OP_QUERY_VPORT_STATE', 1872)
MLX5_CMD_OP_MODIFY_VPORT_STATE = _anonenum5.define('MLX5_CMD_OP_MODIFY_VPORT_STATE', 1873)
MLX5_CMD_OP_QUERY_ESW_VPORT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_QUERY_ESW_VPORT_CONTEXT', 1874)
MLX5_CMD_OP_MODIFY_ESW_VPORT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_MODIFY_ESW_VPORT_CONTEXT', 1875)
MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT', 1876)
MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT', 1877)
MLX5_CMD_OP_QUERY_ROCE_ADDRESS = _anonenum5.define('MLX5_CMD_OP_QUERY_ROCE_ADDRESS', 1888)
MLX5_CMD_OP_SET_ROCE_ADDRESS = _anonenum5.define('MLX5_CMD_OP_SET_ROCE_ADDRESS', 1889)
MLX5_CMD_OP_QUERY_HCA_VPORT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_QUERY_HCA_VPORT_CONTEXT', 1890)
MLX5_CMD_OP_MODIFY_HCA_VPORT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_MODIFY_HCA_VPORT_CONTEXT', 1891)
MLX5_CMD_OP_QUERY_HCA_VPORT_GID = _anonenum5.define('MLX5_CMD_OP_QUERY_HCA_VPORT_GID', 1892)
MLX5_CMD_OP_QUERY_HCA_VPORT_PKEY = _anonenum5.define('MLX5_CMD_OP_QUERY_HCA_VPORT_PKEY', 1893)
MLX5_CMD_OP_QUERY_VNIC_ENV = _anonenum5.define('MLX5_CMD_OP_QUERY_VNIC_ENV', 1903)
MLX5_CMD_OP_QUERY_VPORT_COUNTER = _anonenum5.define('MLX5_CMD_OP_QUERY_VPORT_COUNTER', 1904)
MLX5_CMD_OP_ALLOC_Q_COUNTER = _anonenum5.define('MLX5_CMD_OP_ALLOC_Q_COUNTER', 1905)
MLX5_CMD_OP_DEALLOC_Q_COUNTER = _anonenum5.define('MLX5_CMD_OP_DEALLOC_Q_COUNTER', 1906)
MLX5_CMD_OP_QUERY_Q_COUNTER = _anonenum5.define('MLX5_CMD_OP_QUERY_Q_COUNTER', 1907)
MLX5_CMD_OP_SET_MONITOR_COUNTER = _anonenum5.define('MLX5_CMD_OP_SET_MONITOR_COUNTER', 1908)
MLX5_CMD_OP_ARM_MONITOR_COUNTER = _anonenum5.define('MLX5_CMD_OP_ARM_MONITOR_COUNTER', 1909)
MLX5_CMD_OP_SET_PP_RATE_LIMIT = _anonenum5.define('MLX5_CMD_OP_SET_PP_RATE_LIMIT', 1920)
MLX5_CMD_OP_QUERY_RATE_LIMIT = _anonenum5.define('MLX5_CMD_OP_QUERY_RATE_LIMIT', 1921)
MLX5_CMD_OP_CREATE_SCHEDULING_ELEMENT = _anonenum5.define('MLX5_CMD_OP_CREATE_SCHEDULING_ELEMENT', 1922)
MLX5_CMD_OP_DESTROY_SCHEDULING_ELEMENT = _anonenum5.define('MLX5_CMD_OP_DESTROY_SCHEDULING_ELEMENT', 1923)
MLX5_CMD_OP_QUERY_SCHEDULING_ELEMENT = _anonenum5.define('MLX5_CMD_OP_QUERY_SCHEDULING_ELEMENT', 1924)
MLX5_CMD_OP_MODIFY_SCHEDULING_ELEMENT = _anonenum5.define('MLX5_CMD_OP_MODIFY_SCHEDULING_ELEMENT', 1925)
MLX5_CMD_OP_CREATE_QOS_PARA_VPORT = _anonenum5.define('MLX5_CMD_OP_CREATE_QOS_PARA_VPORT', 1926)
MLX5_CMD_OP_DESTROY_QOS_PARA_VPORT = _anonenum5.define('MLX5_CMD_OP_DESTROY_QOS_PARA_VPORT', 1927)
MLX5_CMD_OP_ALLOC_PD = _anonenum5.define('MLX5_CMD_OP_ALLOC_PD', 2048)
MLX5_CMD_OP_DEALLOC_PD = _anonenum5.define('MLX5_CMD_OP_DEALLOC_PD', 2049)
MLX5_CMD_OP_ALLOC_UAR = _anonenum5.define('MLX5_CMD_OP_ALLOC_UAR', 2050)
MLX5_CMD_OP_DEALLOC_UAR = _anonenum5.define('MLX5_CMD_OP_DEALLOC_UAR', 2051)
MLX5_CMD_OP_CONFIG_INT_MODERATION = _anonenum5.define('MLX5_CMD_OP_CONFIG_INT_MODERATION', 2052)
MLX5_CMD_OP_ACCESS_REG = _anonenum5.define('MLX5_CMD_OP_ACCESS_REG', 2053)
MLX5_CMD_OP_ATTACH_TO_MCG = _anonenum5.define('MLX5_CMD_OP_ATTACH_TO_MCG', 2054)
MLX5_CMD_OP_DETACH_FROM_MCG = _anonenum5.define('MLX5_CMD_OP_DETACH_FROM_MCG', 2055)
MLX5_CMD_OP_GET_DROPPED_PACKET_LOG = _anonenum5.define('MLX5_CMD_OP_GET_DROPPED_PACKET_LOG', 2058)
MLX5_CMD_OP_MAD_IFC = _anonenum5.define('MLX5_CMD_OP_MAD_IFC', 1293)
MLX5_CMD_OP_QUERY_MAD_DEMUX = _anonenum5.define('MLX5_CMD_OP_QUERY_MAD_DEMUX', 2059)
MLX5_CMD_OP_SET_MAD_DEMUX = _anonenum5.define('MLX5_CMD_OP_SET_MAD_DEMUX', 2060)
MLX5_CMD_OP_NOP = _anonenum5.define('MLX5_CMD_OP_NOP', 2061)
MLX5_CMD_OP_ALLOC_XRCD = _anonenum5.define('MLX5_CMD_OP_ALLOC_XRCD', 2062)
MLX5_CMD_OP_DEALLOC_XRCD = _anonenum5.define('MLX5_CMD_OP_DEALLOC_XRCD', 2063)
MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN = _anonenum5.define('MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN', 2070)
MLX5_CMD_OP_DEALLOC_TRANSPORT_DOMAIN = _anonenum5.define('MLX5_CMD_OP_DEALLOC_TRANSPORT_DOMAIN', 2071)
MLX5_CMD_OP_QUERY_CONG_STATUS = _anonenum5.define('MLX5_CMD_OP_QUERY_CONG_STATUS', 2082)
MLX5_CMD_OP_MODIFY_CONG_STATUS = _anonenum5.define('MLX5_CMD_OP_MODIFY_CONG_STATUS', 2083)
MLX5_CMD_OP_QUERY_CONG_PARAMS = _anonenum5.define('MLX5_CMD_OP_QUERY_CONG_PARAMS', 2084)
MLX5_CMD_OP_MODIFY_CONG_PARAMS = _anonenum5.define('MLX5_CMD_OP_MODIFY_CONG_PARAMS', 2085)
MLX5_CMD_OP_QUERY_CONG_STATISTICS = _anonenum5.define('MLX5_CMD_OP_QUERY_CONG_STATISTICS', 2086)
MLX5_CMD_OP_ADD_VXLAN_UDP_DPORT = _anonenum5.define('MLX5_CMD_OP_ADD_VXLAN_UDP_DPORT', 2087)
MLX5_CMD_OP_DELETE_VXLAN_UDP_DPORT = _anonenum5.define('MLX5_CMD_OP_DELETE_VXLAN_UDP_DPORT', 2088)
MLX5_CMD_OP_SET_L2_TABLE_ENTRY = _anonenum5.define('MLX5_CMD_OP_SET_L2_TABLE_ENTRY', 2089)
MLX5_CMD_OP_QUERY_L2_TABLE_ENTRY = _anonenum5.define('MLX5_CMD_OP_QUERY_L2_TABLE_ENTRY', 2090)
MLX5_CMD_OP_DELETE_L2_TABLE_ENTRY = _anonenum5.define('MLX5_CMD_OP_DELETE_L2_TABLE_ENTRY', 2091)
MLX5_CMD_OP_SET_WOL_ROL = _anonenum5.define('MLX5_CMD_OP_SET_WOL_ROL', 2096)
MLX5_CMD_OP_QUERY_WOL_ROL = _anonenum5.define('MLX5_CMD_OP_QUERY_WOL_ROL', 2097)
MLX5_CMD_OP_CREATE_LAG = _anonenum5.define('MLX5_CMD_OP_CREATE_LAG', 2112)
MLX5_CMD_OP_MODIFY_LAG = _anonenum5.define('MLX5_CMD_OP_MODIFY_LAG', 2113)
MLX5_CMD_OP_QUERY_LAG = _anonenum5.define('MLX5_CMD_OP_QUERY_LAG', 2114)
MLX5_CMD_OP_DESTROY_LAG = _anonenum5.define('MLX5_CMD_OP_DESTROY_LAG', 2115)
MLX5_CMD_OP_CREATE_VPORT_LAG = _anonenum5.define('MLX5_CMD_OP_CREATE_VPORT_LAG', 2116)
MLX5_CMD_OP_DESTROY_VPORT_LAG = _anonenum5.define('MLX5_CMD_OP_DESTROY_VPORT_LAG', 2117)
MLX5_CMD_OP_CREATE_TIR = _anonenum5.define('MLX5_CMD_OP_CREATE_TIR', 2304)
MLX5_CMD_OP_MODIFY_TIR = _anonenum5.define('MLX5_CMD_OP_MODIFY_TIR', 2305)
MLX5_CMD_OP_DESTROY_TIR = _anonenum5.define('MLX5_CMD_OP_DESTROY_TIR', 2306)
MLX5_CMD_OP_QUERY_TIR = _anonenum5.define('MLX5_CMD_OP_QUERY_TIR', 2307)
MLX5_CMD_OP_CREATE_SQ = _anonenum5.define('MLX5_CMD_OP_CREATE_SQ', 2308)
MLX5_CMD_OP_MODIFY_SQ = _anonenum5.define('MLX5_CMD_OP_MODIFY_SQ', 2309)
MLX5_CMD_OP_DESTROY_SQ = _anonenum5.define('MLX5_CMD_OP_DESTROY_SQ', 2310)
MLX5_CMD_OP_QUERY_SQ = _anonenum5.define('MLX5_CMD_OP_QUERY_SQ', 2311)
MLX5_CMD_OP_CREATE_RQ = _anonenum5.define('MLX5_CMD_OP_CREATE_RQ', 2312)
MLX5_CMD_OP_MODIFY_RQ = _anonenum5.define('MLX5_CMD_OP_MODIFY_RQ', 2313)
MLX5_CMD_OP_SET_DELAY_DROP_PARAMS = _anonenum5.define('MLX5_CMD_OP_SET_DELAY_DROP_PARAMS', 2320)
MLX5_CMD_OP_DESTROY_RQ = _anonenum5.define('MLX5_CMD_OP_DESTROY_RQ', 2314)
MLX5_CMD_OP_QUERY_RQ = _anonenum5.define('MLX5_CMD_OP_QUERY_RQ', 2315)
MLX5_CMD_OP_CREATE_RMP = _anonenum5.define('MLX5_CMD_OP_CREATE_RMP', 2316)
MLX5_CMD_OP_MODIFY_RMP = _anonenum5.define('MLX5_CMD_OP_MODIFY_RMP', 2317)
MLX5_CMD_OP_DESTROY_RMP = _anonenum5.define('MLX5_CMD_OP_DESTROY_RMP', 2318)
MLX5_CMD_OP_QUERY_RMP = _anonenum5.define('MLX5_CMD_OP_QUERY_RMP', 2319)
MLX5_CMD_OP_CREATE_TIS = _anonenum5.define('MLX5_CMD_OP_CREATE_TIS', 2322)
MLX5_CMD_OP_MODIFY_TIS = _anonenum5.define('MLX5_CMD_OP_MODIFY_TIS', 2323)
MLX5_CMD_OP_DESTROY_TIS = _anonenum5.define('MLX5_CMD_OP_DESTROY_TIS', 2324)
MLX5_CMD_OP_QUERY_TIS = _anonenum5.define('MLX5_CMD_OP_QUERY_TIS', 2325)
MLX5_CMD_OP_CREATE_RQT = _anonenum5.define('MLX5_CMD_OP_CREATE_RQT', 2326)
MLX5_CMD_OP_MODIFY_RQT = _anonenum5.define('MLX5_CMD_OP_MODIFY_RQT', 2327)
MLX5_CMD_OP_DESTROY_RQT = _anonenum5.define('MLX5_CMD_OP_DESTROY_RQT', 2328)
MLX5_CMD_OP_QUERY_RQT = _anonenum5.define('MLX5_CMD_OP_QUERY_RQT', 2329)
MLX5_CMD_OP_SET_FLOW_TABLE_ROOT = _anonenum5.define('MLX5_CMD_OP_SET_FLOW_TABLE_ROOT', 2351)
MLX5_CMD_OP_CREATE_FLOW_TABLE = _anonenum5.define('MLX5_CMD_OP_CREATE_FLOW_TABLE', 2352)
MLX5_CMD_OP_DESTROY_FLOW_TABLE = _anonenum5.define('MLX5_CMD_OP_DESTROY_FLOW_TABLE', 2353)
MLX5_CMD_OP_QUERY_FLOW_TABLE = _anonenum5.define('MLX5_CMD_OP_QUERY_FLOW_TABLE', 2354)
MLX5_CMD_OP_CREATE_FLOW_GROUP = _anonenum5.define('MLX5_CMD_OP_CREATE_FLOW_GROUP', 2355)
MLX5_CMD_OP_DESTROY_FLOW_GROUP = _anonenum5.define('MLX5_CMD_OP_DESTROY_FLOW_GROUP', 2356)
MLX5_CMD_OP_QUERY_FLOW_GROUP = _anonenum5.define('MLX5_CMD_OP_QUERY_FLOW_GROUP', 2357)
MLX5_CMD_OP_SET_FLOW_TABLE_ENTRY = _anonenum5.define('MLX5_CMD_OP_SET_FLOW_TABLE_ENTRY', 2358)
MLX5_CMD_OP_QUERY_FLOW_TABLE_ENTRY = _anonenum5.define('MLX5_CMD_OP_QUERY_FLOW_TABLE_ENTRY', 2359)
MLX5_CMD_OP_DELETE_FLOW_TABLE_ENTRY = _anonenum5.define('MLX5_CMD_OP_DELETE_FLOW_TABLE_ENTRY', 2360)
MLX5_CMD_OP_ALLOC_FLOW_COUNTER = _anonenum5.define('MLX5_CMD_OP_ALLOC_FLOW_COUNTER', 2361)
MLX5_CMD_OP_DEALLOC_FLOW_COUNTER = _anonenum5.define('MLX5_CMD_OP_DEALLOC_FLOW_COUNTER', 2362)
MLX5_CMD_OP_QUERY_FLOW_COUNTER = _anonenum5.define('MLX5_CMD_OP_QUERY_FLOW_COUNTER', 2363)
MLX5_CMD_OP_MODIFY_FLOW_TABLE = _anonenum5.define('MLX5_CMD_OP_MODIFY_FLOW_TABLE', 2364)
MLX5_CMD_OP_ALLOC_PACKET_REFORMAT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_ALLOC_PACKET_REFORMAT_CONTEXT', 2365)
MLX5_CMD_OP_DEALLOC_PACKET_REFORMAT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_DEALLOC_PACKET_REFORMAT_CONTEXT', 2366)
MLX5_CMD_OP_QUERY_PACKET_REFORMAT_CONTEXT = _anonenum5.define('MLX5_CMD_OP_QUERY_PACKET_REFORMAT_CONTEXT', 2367)
MLX5_CMD_OP_ALLOC_MODIFY_HEADER_CONTEXT = _anonenum5.define('MLX5_CMD_OP_ALLOC_MODIFY_HEADER_CONTEXT', 2368)
MLX5_CMD_OP_DEALLOC_MODIFY_HEADER_CONTEXT = _anonenum5.define('MLX5_CMD_OP_DEALLOC_MODIFY_HEADER_CONTEXT', 2369)
MLX5_CMD_OP_QUERY_MODIFY_HEADER_CONTEXT = _anonenum5.define('MLX5_CMD_OP_QUERY_MODIFY_HEADER_CONTEXT', 2370)
MLX5_CMD_OP_FPGA_CREATE_QP = _anonenum5.define('MLX5_CMD_OP_FPGA_CREATE_QP', 2400)
MLX5_CMD_OP_FPGA_MODIFY_QP = _anonenum5.define('MLX5_CMD_OP_FPGA_MODIFY_QP', 2401)
MLX5_CMD_OP_FPGA_QUERY_QP = _anonenum5.define('MLX5_CMD_OP_FPGA_QUERY_QP', 2402)
MLX5_CMD_OP_FPGA_DESTROY_QP = _anonenum5.define('MLX5_CMD_OP_FPGA_DESTROY_QP', 2403)
MLX5_CMD_OP_FPGA_QUERY_QP_COUNTERS = _anonenum5.define('MLX5_CMD_OP_FPGA_QUERY_QP_COUNTERS', 2404)
MLX5_CMD_OP_CREATE_GENERAL_OBJECT = _anonenum5.define('MLX5_CMD_OP_CREATE_GENERAL_OBJECT', 2560)
MLX5_CMD_OP_MODIFY_GENERAL_OBJECT = _anonenum5.define('MLX5_CMD_OP_MODIFY_GENERAL_OBJECT', 2561)
MLX5_CMD_OP_QUERY_GENERAL_OBJECT = _anonenum5.define('MLX5_CMD_OP_QUERY_GENERAL_OBJECT', 2562)
MLX5_CMD_OP_DESTROY_GENERAL_OBJECT = _anonenum5.define('MLX5_CMD_OP_DESTROY_GENERAL_OBJECT', 2563)
MLX5_CMD_OP_CREATE_UCTX = _anonenum5.define('MLX5_CMD_OP_CREATE_UCTX', 2564)
MLX5_CMD_OP_DESTROY_UCTX = _anonenum5.define('MLX5_CMD_OP_DESTROY_UCTX', 2566)
MLX5_CMD_OP_CREATE_UMEM = _anonenum5.define('MLX5_CMD_OP_CREATE_UMEM', 2568)
MLX5_CMD_OP_DESTROY_UMEM = _anonenum5.define('MLX5_CMD_OP_DESTROY_UMEM', 2570)
MLX5_CMD_OP_SYNC_STEERING = _anonenum5.define('MLX5_CMD_OP_SYNC_STEERING', 2816)
MLX5_CMD_OP_PSP_GEN_SPI = _anonenum5.define('MLX5_CMD_OP_PSP_GEN_SPI', 2832)
MLX5_CMD_OP_PSP_ROTATE_KEY = _anonenum5.define('MLX5_CMD_OP_PSP_ROTATE_KEY', 2833)
MLX5_CMD_OP_QUERY_VHCA_STATE = _anonenum5.define('MLX5_CMD_OP_QUERY_VHCA_STATE', 2829)
MLX5_CMD_OP_MODIFY_VHCA_STATE = _anonenum5.define('MLX5_CMD_OP_MODIFY_VHCA_STATE', 2830)
MLX5_CMD_OP_SYNC_CRYPTO = _anonenum5.define('MLX5_CMD_OP_SYNC_CRYPTO', 2834)
MLX5_CMD_OP_ALLOW_OTHER_VHCA_ACCESS = _anonenum5.define('MLX5_CMD_OP_ALLOW_OTHER_VHCA_ACCESS', 2838)
MLX5_CMD_OP_GENERATE_WQE = _anonenum5.define('MLX5_CMD_OP_GENERATE_WQE', 2839)
MLX5_CMD_OPCODE_QUERY_VUID = _anonenum5.define('MLX5_CMD_OPCODE_QUERY_VUID', 2850)
MLX5_CMD_OP_MAX = _anonenum5.define('MLX5_CMD_OP_MAX', 2851)

class _anonenum6(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CMD_OP_GENERAL_START = _anonenum6.define('MLX5_CMD_OP_GENERAL_START', 2816)
MLX5_CMD_OP_GENERAL_END = _anonenum6.define('MLX5_CMD_OP_GENERAL_END', 3328)

class _anonenum7(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FT_NIC_RX_2_NIC_RX_RDMA = _anonenum7.define('MLX5_FT_NIC_RX_2_NIC_RX_RDMA', 0)
MLX5_FT_NIC_TX_RDMA_2_NIC_TX = _anonenum7.define('MLX5_FT_NIC_TX_RDMA_2_NIC_TX', 1)

class _anonenum8(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CMD_OP_MOD_UPDATE_HEADER_MODIFY_ARGUMENT = _anonenum8.define('MLX5_CMD_OP_MOD_UPDATE_HEADER_MODIFY_ARGUMENT', 1)

@c.record
class struct_mlx5_ifc_flow_table_fields_supported_bits(c.Struct):
  SIZE = 128
  outer_dmac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  outer_smac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  outer_ether_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  outer_ip_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  outer_first_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  outer_first_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  outer_first_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  outer_ipv4_ttl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  outer_second_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  outer_second_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 9]
  outer_second_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 10]
  reserved_at_b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 11]
  outer_sip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 12]
  outer_dip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  outer_frag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  outer_ip_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 15]
  outer_ip_ecn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  outer_ip_dscp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 17]
  outer_udp_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 18]
  outer_udp_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 19]
  outer_tcp_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 20]
  outer_tcp_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 21]
  outer_tcp_flags: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 22]
  outer_gre_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 23]
  outer_gre_key: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 24]
  outer_vxlan_vni: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 25]
  outer_geneve_vni: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 26]
  outer_geneve_oam: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 27]
  outer_geneve_protocol_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  outer_geneve_opt_len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  source_vhca_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 30]
  source_eswitch_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
  inner_dmac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  inner_smac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  inner_ether_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  inner_ip_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 35]
  inner_first_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 36]
  inner_first_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 37]
  inner_first_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 38]
  reserved_at_27: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 39]
  inner_second_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 40]
  inner_second_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 41]
  inner_second_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 42]
  reserved_at_2b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 43]
  inner_sip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 44]
  inner_dip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 45]
  inner_frag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 46]
  inner_ip_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 47]
  inner_ip_ecn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 48]
  inner_ip_dscp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 49]
  inner_udp_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 50]
  inner_udp_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 51]
  inner_tcp_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 52]
  inner_tcp_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 53]
  inner_tcp_flags: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 54]
  reserved_at_37: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 55]
  geneve_tlv_option_0_data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  geneve_tlv_option_0_exist: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 66]
  outer_first_mpls_over_udp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 69]
  outer_first_mpls_over_gre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 73]
  inner_first_mpls: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 77]
  outer_first_mpls: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 81]
  reserved_at_55: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 85]
  outer_esp_spi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 87]
  reserved_at_58: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 88]
  bth_dst_qp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 90]
  reserved_at_5b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 91]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 96]
  metadata_reg_c_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 120]
  metadata_reg_c_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 121]
  metadata_reg_c_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 122]
  metadata_reg_c_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 123]
  metadata_reg_c_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 124]
  metadata_reg_c_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 125]
  metadata_reg_c_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 126]
  metadata_reg_c_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 127]
@c.record
class struct_mlx5_ifc_flow_table_fields_supported_2_bits(c.Struct):
  SIZE = 128
  inner_l4_type_ext: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  outer_l4_type_ext: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  inner_l4_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  outer_l4_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 4]
  bth_opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  reserved_at_f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 15]
  tunnel_header_0_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  reserved_at_11: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 17]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 32]
  ipsec_next_header: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 47]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_flow_table_prop_layout_bits(c.Struct):
  SIZE = 512
  ft_support: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  flow_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  flow_modify_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  modify_root: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  identified_miss_table_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  flow_table_modify: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  reformat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  decap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  reset_root_to_default: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 9]
  pop_vlan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 10]
  push_vlan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 11]
  reserved_at_c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 12]
  pop_vlan_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  push_vlan_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  reformat_and_vlan_action: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 15]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  sw_owner: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 17]
  reformat_l3_tunnel_to_l2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 18]
  reformat_l2_to_l3_tunnel: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 19]
  reformat_and_modify_action: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 20]
  ignore_flow_level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 21]
  reserved_at_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 22]
  table_miss_action_domain: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 23]
  termination_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 24]
  reformat_and_fwd_to_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 25]
  reserved_at_1a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 26]
  ipsec_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  ipsec_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  sw_owner_v2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 30]
  reserved_at_1f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
  termination_table_raw_traffic: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  log_max_ft_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 34]
  log_max_modify_header_context: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  max_modify_header_actions: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  max_ft_level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  reformat_add_esp_trasport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reformat_l2_to_l3_esp_tunnel: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reformat_add_esp_transport_over_udp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 66]
  reformat_del_esp_trasport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 67]
  reformat_l3_esp_tunnel_to_l2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 68]
  reformat_del_esp_transport_over_udp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 69]
  execute_aso: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 70]
  reserved_at_47: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[25]], 71]
  reformat_l2_to_l3_psp_tunnel: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  reformat_l3_psp_tunnel_to_l2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 97]
  reformat_insert: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 98]
  reformat_remove: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 99]
  macsec_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 100]
  macsec_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 101]
  psp_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 102]
  psp_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 103]
  reformat_add_macsec: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 104]
  reformat_remove_macsec: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 105]
  reparse: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 106]
  reserved_at_6b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 107]
  cross_vhca_object: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 108]
  reformat_l2_to_l3_audp_tunnel: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 109]
  reformat_l3_audp_tunnel_to_l2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 110]
  ignore_flow_level_rtc_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 111]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 112]
  log_max_ft_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  log_max_flow_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 144]
  log_max_destination: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 152]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 160]
  log_max_flow: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 184]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  ft_field_support: Annotated[struct_mlx5_ifc_flow_table_fields_supported_bits, 256]
  ft_field_bitmask_support: Annotated[struct_mlx5_ifc_flow_table_fields_supported_bits, 384]
@c.record
class struct_mlx5_ifc_odp_per_transport_service_cap_bits(c.Struct):
  SIZE = 32
  send: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  receive: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  write: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  read: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  atomic: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  srq_receive: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  reserved_at_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 6]
@c.record
class struct_mlx5_ifc_ipv4_layout_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 0]
  ipv4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_ipv6_layout_bits(c.Struct):
  SIZE = 128
  ipv6: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 0]
@c.record
class struct_mlx5_ifc_ipv6_simple_layout_bits(c.Struct):
  SIZE = 128
  ipv6_127_96: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  ipv6_95_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  ipv6_63_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  ipv6_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits(c.Struct):
  SIZE = 128
  ipv6_simple_layout: Annotated[struct_mlx5_ifc_ipv6_simple_layout_bits, 0]
  ipv6_layout: Annotated[struct_mlx5_ifc_ipv6_layout_bits, 0]
  ipv4_layout: Annotated[struct_mlx5_ifc_ipv4_layout_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
class _anonenum9(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_PACKET_L4_TYPE_NONE = _anonenum9.define('MLX5_PACKET_L4_TYPE_NONE', 0)
MLX5_PACKET_L4_TYPE_TCP = _anonenum9.define('MLX5_PACKET_L4_TYPE_TCP', 1)
MLX5_PACKET_L4_TYPE_UDP = _anonenum9.define('MLX5_PACKET_L4_TYPE_UDP', 2)

class _anonenum10(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_PACKET_L4_TYPE_EXT_NONE = _anonenum10.define('MLX5_PACKET_L4_TYPE_EXT_NONE', 0)
MLX5_PACKET_L4_TYPE_EXT_TCP = _anonenum10.define('MLX5_PACKET_L4_TYPE_EXT_TCP', 1)
MLX5_PACKET_L4_TYPE_EXT_UDP = _anonenum10.define('MLX5_PACKET_L4_TYPE_EXT_UDP', 2)
MLX5_PACKET_L4_TYPE_EXT_ICMP = _anonenum10.define('MLX5_PACKET_L4_TYPE_EXT_ICMP', 3)

@c.record
class struct_mlx5_ifc_fte_match_set_lyr_2_4_bits(c.Struct):
  SIZE = 512
  smac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  smac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  ethertype: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  dmac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  dmac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  first_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 112]
  first_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 115]
  first_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 116]
  ip_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  ip_dscp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 136]
  ip_ecn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 142]
  cvlan_tag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 144]
  svlan_tag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 145]
  frag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 146]
  ip_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 147]
  tcp_flags: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 151]
  tcp_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  tcp_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  l4_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 192]
  l4_type_ext: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 194]
  reserved_at_c6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 198]
  ipv4_ihl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 208]
  reserved_at_d4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 212]
  ttl_hoplimit: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 216]
  udp_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  udp_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  src_ipv4_src_ipv6: Annotated[union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits, 256]
  dst_ipv4_dst_ipv6: Annotated[union_mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits, 384]
@c.record
class struct_mlx5_ifc_nvgre_key_bits(c.Struct):
  SIZE = 32
  hi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 0]
  lo: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
@c.record
class union_mlx5_ifc_gre_key_bits(c.Struct):
  SIZE = 32
  nvgre: Annotated[struct_mlx5_ifc_nvgre_key_bits, 0]
  key: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_fte_match_set_misc_bits(c.Struct):
  SIZE = 512
  gre_c_present: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  gre_k_present: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  gre_s_present: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  source_vhca_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  source_sqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  source_eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  source_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  outer_second_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 64]
  outer_second_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 67]
  outer_second_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 68]
  inner_second_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 80]
  inner_second_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 83]
  inner_second_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 84]
  outer_second_cvlan_tag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  inner_second_cvlan_tag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 97]
  outer_second_svlan_tag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 98]
  inner_second_svlan_tag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 99]
  reserved_at_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 100]
  gre_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  gre_key: Annotated[union_mlx5_ifc_gre_key_bits, 128]
  vxlan_vni: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 160]
  bth_opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 184]
  geneve_vni: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 192]
  reserved_at_d8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 216]
  geneve_tlv_option_0_exist: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 222]
  geneve_oam: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 223]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 224]
  outer_ipv6_flow_label: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 236]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 256]
  inner_ipv6_flow_label: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 268]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 288]
  geneve_opt_len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 298]
  geneve_protocol_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 304]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 320]
  bth_dst_qp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 328]
  inner_esp_spi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  outer_esp_spi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  reserved_at_1a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 416]
@c.record
class struct_mlx5_ifc_fte_match_mpls_bits(c.Struct):
  SIZE = 32
  mpls_label: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 0]
  mpls_exp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 20]
  mpls_s_bos: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 23]
  mpls_ttl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
@c.record
class struct_mlx5_ifc_fte_match_set_misc2_bits(c.Struct):
  SIZE = 512
  outer_first_mpls: Annotated[struct_mlx5_ifc_fte_match_mpls_bits, 0]
  inner_first_mpls: Annotated[struct_mlx5_ifc_fte_match_mpls_bits, 32]
  outer_first_mpls_over_gre: Annotated[struct_mlx5_ifc_fte_match_mpls_bits, 64]
  outer_first_mpls_over_udp: Annotated[struct_mlx5_ifc_fte_match_mpls_bits, 96]
  metadata_reg_c_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  metadata_reg_c_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  metadata_reg_c_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  metadata_reg_c_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  metadata_reg_c_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  metadata_reg_c_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  metadata_reg_c_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  metadata_reg_c_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  metadata_reg_a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  psp_syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 416]
  macsec_syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 424]
  ipsec_syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 432]
  ipsec_next_header: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 440]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
@c.record
class struct_mlx5_ifc_fte_match_set_misc3_bits(c.Struct):
  SIZE = 512
  inner_tcp_seq_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  outer_tcp_seq_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  inner_tcp_ack_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  outer_tcp_ack_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  outer_vxlan_gpe_vni: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  outer_vxlan_gpe_next_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  outer_vxlan_gpe_flags: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 168]
  reserved_at_b0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  icmp_header_data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  icmpv6_header_data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  icmp_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  icmp_code: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 264]
  icmpv6_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 272]
  icmpv6_code: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 280]
  geneve_tlv_option_0_data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  gtpu_teid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  gtpu_msg_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 352]
  gtpu_msg_flags: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 360]
  reserved_at_170: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 368]
  gtpu_dw_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  gtpu_first_ext_dw_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  gtpu_dw_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
@c.record
class struct_mlx5_ifc_fte_match_set_misc4_bits(c.Struct):
  SIZE = 512
  prog_sample_field_value_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  prog_sample_field_id_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  prog_sample_field_value_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  prog_sample_field_id_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  prog_sample_field_value_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  prog_sample_field_id_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  prog_sample_field_value_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  prog_sample_field_id_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 256]
@c.record
class struct_mlx5_ifc_fte_match_set_misc5_bits(c.Struct):
  SIZE = 512
  macsec_tag_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  macsec_tag_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  macsec_tag_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  macsec_tag_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  tunnel_header_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  tunnel_header_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  tunnel_header_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  tunnel_header_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 256]
@c.record
class struct_mlx5_ifc_cmd_pas_bits(c.Struct):
  SIZE = 64
  pa_h: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  pa_l: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 32]
  reserved_at_34: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 52]
@c.record
class struct_mlx5_ifc_uint64_bits(c.Struct):
  SIZE = 64
  hi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  lo: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
class _anonenum11(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ADS_STAT_RATE_NO_LIMIT = _anonenum11.define('MLX5_ADS_STAT_RATE_NO_LIMIT', 0)
MLX5_ADS_STAT_RATE_2_5GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_2_5GBPS', 7)
MLX5_ADS_STAT_RATE_10GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_10GBPS', 8)
MLX5_ADS_STAT_RATE_30GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_30GBPS', 9)
MLX5_ADS_STAT_RATE_5GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_5GBPS', 10)
MLX5_ADS_STAT_RATE_20GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_20GBPS', 11)
MLX5_ADS_STAT_RATE_40GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_40GBPS', 12)
MLX5_ADS_STAT_RATE_60GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_60GBPS', 13)
MLX5_ADS_STAT_RATE_80GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_80GBPS', 14)
MLX5_ADS_STAT_RATE_120GBPS = _anonenum11.define('MLX5_ADS_STAT_RATE_120GBPS', 15)

@c.record
class struct_mlx5_ifc_ads_bits(c.Struct):
  SIZE = 352
  fl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  free_ar: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 2]
  pkey_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  plane_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  grh: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 40]
  mlid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 41]
  rlid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  ack_timeout: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 64]
  reserved_at_45: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 69]
  src_addr_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 72]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 80]
  stat_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 84]
  hop_limit: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 96]
  tclass: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 100]
  flow_label: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 108]
  rgid_rip: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 128]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 256]
  f_dscp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 260]
  f_ecn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 261]
  reserved_at_106: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 262]
  f_eth_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 263]
  ecn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 264]
  dscp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 266]
  udp_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 272]
  dei_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 288]
  eth_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 289]
  sl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 292]
  vhca_port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 296]
  rmac_47_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 304]
  rmac_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
@c.record
class struct_mlx5_ifc_flow_table_nic_cap_bits(c.Struct):
  SIZE = 32768
  nic_rx_multi_path_tirs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  nic_rx_multi_path_tirs_fts: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  allow_sniffer_and_nic_rx_shared_tir: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  reserved_at_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 3]
  sw_owner_reformat_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  encap_general_header: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 33]
  log_max_packet_reformat_context: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 43]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 48]
  max_encap_header_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 54]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[448]], 64]
  flow_table_properties_nic_receive: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 512]
  flow_table_properties_nic_receive_rdma: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 1024]
  flow_table_properties_nic_receive_sniffer: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 1536]
  flow_table_properties_nic_transmit: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 2048]
  flow_table_properties_nic_transmit_rdma: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 2560]
  flow_table_properties_nic_transmit_sniffer: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 3072]
  reserved_at_e00: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1536]], 3584]
  ft_field_support_2_nic_receive: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 5120]
  reserved_at_1480: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 5248]
  ft_field_support_2_nic_receive_rdma: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 5376]
  reserved_at_1580: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[640]], 5504]
  ft_field_support_2_nic_transmit_rdma: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 6144]
  reserved_at_1880: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1920]], 6272]
  sw_steering_nic_rx_action_drop_icm_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 8192]
  sw_steering_nic_tx_action_drop_icm_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 8256]
  sw_steering_nic_tx_action_allow_icm_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 8320]
  reserved_at_20c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24384]], 8384]
@c.record
class struct_mlx5_ifc_port_selection_cap_bits(c.Struct):
  SIZE = 32768
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  port_select_flow_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  reserved_at_11: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 17]
  port_select_flow_table_bypass: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 18]
  reserved_at_13: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[13]], 19]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[480]], 32]
  flow_table_properties_port_selection: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 512]
  ft_field_support_2_port_selection: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1024]
  reserved_at_480: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31616]], 1152]
class _anonenum12(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FDB_TO_VPORT_REG_C_0 = _anonenum12.define('MLX5_FDB_TO_VPORT_REG_C_0', 1)
MLX5_FDB_TO_VPORT_REG_C_1 = _anonenum12.define('MLX5_FDB_TO_VPORT_REG_C_1', 2)
MLX5_FDB_TO_VPORT_REG_C_2 = _anonenum12.define('MLX5_FDB_TO_VPORT_REG_C_2', 4)
MLX5_FDB_TO_VPORT_REG_C_3 = _anonenum12.define('MLX5_FDB_TO_VPORT_REG_C_3', 8)
MLX5_FDB_TO_VPORT_REG_C_4 = _anonenum12.define('MLX5_FDB_TO_VPORT_REG_C_4', 16)
MLX5_FDB_TO_VPORT_REG_C_5 = _anonenum12.define('MLX5_FDB_TO_VPORT_REG_C_5', 32)
MLX5_FDB_TO_VPORT_REG_C_6 = _anonenum12.define('MLX5_FDB_TO_VPORT_REG_C_6', 64)
MLX5_FDB_TO_VPORT_REG_C_7 = _anonenum12.define('MLX5_FDB_TO_VPORT_REG_C_7', 128)

@c.record
class struct_mlx5_ifc_flow_table_eswitch_cap_bits(c.Struct):
  SIZE = 32768
  fdb_to_vport_reg_c_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 8]
  fdb_uplink_hairpin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  fdb_multi_path_any_table_limit_regc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  reserved_at_f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 15]
  fdb_dynamic_tunnel: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  reserved_at_11: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 17]
  fdb_multi_path_any_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 18]
  reserved_at_13: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 19]
  fdb_modify_header_fwd_to_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 21]
  fdb_ipv4_ttl_modify: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 22]
  flow_source: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 23]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 24]
  multi_fdb_encap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 26]
  egress_acl_forward_to_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 27]
  fdb_multi_path_to_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  reserved_at_1d: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 29]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[480]], 32]
  flow_table_properties_nic_esw_fdb: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 512]
  flow_table_properties_esw_acl_ingress: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 1024]
  flow_table_properties_esw_acl_egress: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 1536]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3072]], 2048]
  ft_field_support_2_esw_fdb: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 5120]
  ft_field_bitmask_support_2_esw_fdb: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 5248]
  reserved_at_1500: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[768]], 5376]
  sw_steering_fdb_action_drop_icm_address_rx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 6144]
  sw_steering_fdb_action_drop_icm_address_tx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 6208]
  sw_steering_uplink_icm_address_rx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 6272]
  sw_steering_uplink_icm_address_tx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 6336]
  reserved_at_1900: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26368]], 6400]
@c.record
class struct_mlx5_ifc_wqe_based_flow_table_cap_bits(c.Struct):
  SIZE = 480
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 0]
  log_max_num_ste: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 3]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 8]
  log_max_num_stc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 11]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 16]
  log_max_num_rtc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 19]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 24]
  log_max_num_header_modify_pattern: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 27]
  rtc_hash_split_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  rtc_linear_lookup_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  stc_alloc_log_granularity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 35]
  reserved_at_28: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 40]
  stc_alloc_log_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 43]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 48]
  ste_alloc_log_granularity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 51]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 56]
  ste_alloc_log_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 59]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 64]
  rtc_reparse_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 75]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 80]
  rtc_index_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 83]
  reserved_at_58: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 88]
  rtc_log_depth_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 91]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  ste_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  stc_action_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 128]
  header_insert_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 256]
  header_remove_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 272]
  trivial_match_definer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[27]], 320]
  rtc_max_num_hash_definer_gen_wqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 347]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 352]
  access_index_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 376]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 384]
  ste_format_gen_wqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 400]
  linear_match_definer_reg_c3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  fdb_jump_to_tir_stc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 448]
  reserved_at_1c1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 449]
@c.record
class struct_mlx5_ifc_esw_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[29]], 0]
  merged_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  reserved_at_1e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 30]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 32]
  esw_manager_vport_number_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  reserved_at_61: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 97]
  esw_manager_vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1920]], 128]
class _anonenum13(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_COUNTER_SOURCE_ESWITCH = _anonenum13.define('MLX5_COUNTER_SOURCE_ESWITCH', 0)
MLX5_COUNTER_FLOW_ESWITCH = _anonenum13.define('MLX5_COUNTER_FLOW_ESWITCH', 1)

@c.record
class struct_mlx5_ifc_e_switch_cap_bits(c.Struct):
  SIZE = 2048
  vport_svlan_strip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  vport_cvlan_strip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  vport_svlan_insert: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  vport_cvlan_insert_if_not_exist: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  vport_cvlan_insert_overwrite: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  reserved_at_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  vport_cvlan_insert_always: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  esw_shared_ingress_acl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  esw_uplink_ingress_acl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  root_ft_on_other_esw: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 9]
  reserved_at_a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 10]
  esw_vport_state_max_tx_speed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 11]
  reserved_at_c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[13]], 12]
  esw_functions_changed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 25]
  reserved_at_1a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 26]
  ecpf_vport_exists: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 27]
  counter_eswitch_affinity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  merged_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  nic_vport_node_guid_modify: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 30]
  nic_vport_port_guid_modify: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
  vxlan_encap_decap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  nvgre_encap_decap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  log_max_fdb_encap_uplink: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 35]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 40]
  log_max_packet_reformat_context: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 43]
  reserved_2b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 48]
  max_encap_header_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 54]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 64]
  log_max_esw_sf: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 75]
  esw_sf_base_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1952]], 96]
@c.record
class struct_mlx5_ifc_qos_cap_bits(c.Struct):
  SIZE = 2048
  packet_pacing: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  esw_scheduling: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  esw_bw_share: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  esw_rate_limit: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  packet_pacing_burst_bound: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  packet_pacing_typical_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  reserved_at_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  nic_sq_scheduling: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  nic_bw_share: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 9]
  nic_rate_limit: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 10]
  packet_pacing_uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 11]
  log_esw_max_sched_depth: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 32]
  esw_cross_esw_sched: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 41]
  reserved_at_2a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 42]
  log_max_qos_nic_queue_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 43]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  packet_pacing_max_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  packet_pacing_min_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 128]
  log_esw_max_rate_limit: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 139]
  packet_pacing_rate_table_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  esw_element_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  esw_tsar_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 192]
  max_qos_para_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 208]
  max_tsar_bw_share: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  nic_element_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 256]
  nic_tsar_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 272]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 288]
  log_meter_aso_granularity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 291]
  reserved_at_128: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 296]
  log_meter_aso_max_alloc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 299]
  reserved_at_130: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 304]
  log_max_num_meter_aso: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 307]
  reserved_at_138: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 312]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1728]], 320]
@c.record
class struct_mlx5_ifc_debug_cap_bits(c.Struct):
  SIZE = 2048
  core_dump_general: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  core_dump_qp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 2]
  resource_dump: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 9]
  reserved_at_a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[22]], 10]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 32]
  stall_detect: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  reserved_at_23: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[29]], 35]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1984]], 64]
@c.record
class struct_mlx5_ifc_per_protocol_networking_offload_caps_bits(c.Struct):
  SIZE = 2048
  csum_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  vlan_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  lro_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  lro_psh_flag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  lro_time_stamp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  reserved_at_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 5]
  wqe_vlan_insert: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  self_lb_en_modifiable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  reserved_at_9: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 9]
  max_lso_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 11]
  multi_pkt_send_wqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 16]
  wqe_inline_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 18]
  rss_ind_tbl_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reg_umr_sq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 24]
  scatter_fcs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 25]
  enhanced_multi_pkt_send_wqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 26]
  tunnel_lso_const_out_ip_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 27]
  tunnel_lro_gre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  tunnel_lro_vxlan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  tunnel_stateless_gre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 30]
  tunnel_stateless_vxlan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
  swp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  swp_csum: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  swp_lso: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  cqe_checksum_full: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 35]
  tunnel_stateless_geneve_tx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 36]
  tunnel_stateless_mpls_over_udp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 37]
  tunnel_stateless_mpls_over_gre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 38]
  tunnel_stateless_vxlan_gpe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 39]
  tunnel_stateless_ipv4_over_vxlan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 40]
  tunnel_stateless_ip_over_ip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 41]
  insert_trailer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 42]
  reserved_at_2b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 43]
  tunnel_stateless_ip_over_ip_rx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 44]
  tunnel_stateless_ip_over_ip_tx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 45]
  reserved_at_2e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 46]
  max_vxlan_udp_ports: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  swp_csum_l4_partial: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 56]
  reserved_at_39: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 57]
  max_geneve_opt_len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 62]
  tunnel_stateless_geneve_rx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  lro_min_mss_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[288]], 96]
  lro_timer_supported_periods: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[4]], 384]
  reserved_at_200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1536]], 512]
class _anonenum14(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_TIMESTAMP_FORMAT_CAP_FREE_RUNNING = _anonenum14.define('MLX5_TIMESTAMP_FORMAT_CAP_FREE_RUNNING', 0)
MLX5_TIMESTAMP_FORMAT_CAP_REAL_TIME = _anonenum14.define('MLX5_TIMESTAMP_FORMAT_CAP_REAL_TIME', 1)
MLX5_TIMESTAMP_FORMAT_CAP_FREE_RUNNING_AND_REAL_TIME = _anonenum14.define('MLX5_TIMESTAMP_FORMAT_CAP_FREE_RUNNING_AND_REAL_TIME', 2)

@c.record
class struct_mlx5_ifc_roce_cap_bits(c.Struct):
  SIZE = 2048
  roce_apm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1]
  sw_r_roce_src_udp_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  fl_rc_qp_when_roce_disabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  fl_rc_qp_when_roce_enabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  roce_cc_general: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  qp_ooo_transmit_default: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  reserved_at_9: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[21]], 9]
  qp_ts_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 30]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 128]
  l3_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 140]
  reserved_at_90: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 144]
  roce_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 152]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  r_roce_dest_udp_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  r_roce_max_src_udp_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 192]
  r_roce_min_src_udp_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 208]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  roce_address_table_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1792]], 256]
@c.record
class struct_mlx5_ifc_sync_steering_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
@c.record
class struct_mlx5_ifc_sync_steering_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_sync_crypto_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  crypto_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 128]
@c.record
class struct_mlx5_ifc_sync_crypto_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_device_mem_cap_bits(c.Struct):
  SIZE = 2048
  memic: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 1]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 32]
  log_min_memic_alloc_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 43]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  log_max_memic_addr_alignment: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  memic_bar_start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  memic_bar_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  max_memic_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  steering_sw_icm_start_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  log_header_modify_sw_icm_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 264]
  reserved_at_110: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 272]
  log_sw_icm_alloc_granularity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 274]
  log_steering_sw_icm_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 280]
  log_indirect_encap_sw_icm_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 288]
  reserved_at_128: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 296]
  log_header_modify_pattern_sw_icm_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 312]
  header_modify_sw_icm_start_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 384]
  header_modify_pattern_sw_icm_start_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
  memic_operations: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  reserved_at_220: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  indirect_encap_sw_icm_start_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 576]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1408]], 640]
@c.record
class struct_mlx5_ifc_device_event_cap_bits(c.Struct):
  SIZE = 512
  user_affiliated_events: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[4]], 0]
  user_unaffiliated_events: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[4]], 256]
@c.record
class struct_mlx5_ifc_virtio_emulation_cap_bits(c.Struct):
  SIZE = 2048
  desc_tunnel_offload_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  eth_frame_offload_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  virtio_version_1_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  device_features_bits_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[13]], 3]
  event_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  virtio_queue_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  max_tunnel_desc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 48]
  log_doorbell_stride: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 51]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 56]
  log_doorbell_bar_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 59]
  doorbell_bar_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  max_emulated_devices: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  max_num_virtio_queues: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[19]], 192]
  desc_group_mkey_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 211]
  freeze_to_rdy_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 212]
  reserved_at_d5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 213]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  umem_1_buffer_param_a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  umem_1_buffer_param_b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  umem_2_buffer_param_a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  umem_2_buffer_param_b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  umem_3_buffer_param_a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  umem_3_buffer_param_b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1600]], 448]
class _anonenum15(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_1_BYTE = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_1_BYTE', 0)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_2_BYTES = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_2_BYTES', 2)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_4_BYTES = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_4_BYTES', 4)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_8_BYTES = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_8_BYTES', 8)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_16_BYTES = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_16_BYTES', 16)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_32_BYTES = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_32_BYTES', 32)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_64_BYTES = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_64_BYTES', 64)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_128_BYTES = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_128_BYTES', 128)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_256_BYTES = _anonenum15.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_QP_256_BYTES', 256)

class _anonenum16(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_1_BYTE = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_1_BYTE', 1)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_2_BYTES = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_2_BYTES', 2)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_4_BYTES = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_4_BYTES', 4)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_8_BYTES = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_8_BYTES', 8)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_16_BYTES = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_16_BYTES', 16)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_32_BYTES = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_32_BYTES', 32)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_64_BYTES = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_64_BYTES', 64)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_128_BYTES = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_128_BYTES', 128)
MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_256_BYTES = _anonenum16.define('MLX5_ATOMIC_CAPS_ATOMIC_SIZE_DC_256_BYTES', 256)

@c.record
class struct_mlx5_ifc_atomic_caps_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  atomic_req_8B_endianness_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 64]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 66]
  supported_atomic_req_8B_endianness_mode_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 70]
  reserved_at_47: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[25]], 71]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  atomic_operations: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  atomic_size_qp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 192]
  atomic_size_dc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 208]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1824]], 224]
@c.record
class struct_mlx5_ifc_odp_scheme_cap_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  sig: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 65]
  page_prefetch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 69]
  reserved_at_46: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 70]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  rc_odp_caps: Annotated[struct_mlx5_ifc_odp_per_transport_service_cap_bits, 128]
  uc_odp_caps: Annotated[struct_mlx5_ifc_odp_per_transport_service_cap_bits, 160]
  ud_odp_caps: Annotated[struct_mlx5_ifc_odp_per_transport_service_cap_bits, 192]
  xrc_odp_caps: Annotated[struct_mlx5_ifc_odp_per_transport_service_cap_bits, 224]
  dc_odp_caps: Annotated[struct_mlx5_ifc_odp_per_transport_service_cap_bits, 256]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 288]
@c.record
class struct_mlx5_ifc_odp_cap_bits(c.Struct):
  SIZE = 2048
  transport_page_fault_scheme_cap: Annotated[struct_mlx5_ifc_odp_scheme_cap_bits, 0]
  memory_page_fault_scheme_cap: Annotated[struct_mlx5_ifc_odp_scheme_cap_bits, 512]
  reserved_at_400: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[512]], 1024]
  mem_page_fault: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1536]
  reserved_at_601: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 1537]
  reserved_at_620: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[480]], 1568]
@c.record
class struct_mlx5_ifc_tls_cap_bits(c.Struct):
  SIZE = 2048
  tls_1_2_aes_gcm_128: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  tls_1_3_aes_gcm_128: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  tls_1_2_aes_gcm_256: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  tls_1_3_aes_gcm_256: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 4]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2016]], 32]
@c.record
class struct_mlx5_ifc_ipsec_cap_bits(c.Struct):
  SIZE = 2048
  ipsec_full_offload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  ipsec_crypto_offload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  ipsec_esn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  ipsec_crypto_esp_aes_gcm_256_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  ipsec_crypto_esp_aes_gcm_128_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  ipsec_crypto_esp_aes_gcm_256_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  ipsec_crypto_esp_aes_gcm_128_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  reserved_at_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 7]
  log_max_ipsec_offload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 11]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  min_log_ipsec_full_replay_window: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  max_log_ipsec_full_replay_window: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2000]], 48]
@c.record
class struct_mlx5_ifc_macsec_cap_bits(c.Struct):
  SIZE = 2048
  macsec_epn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1]
  macsec_crypto_esp_aes_gcm_256_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  macsec_crypto_esp_aes_gcm_128_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  macsec_crypto_esp_aes_gcm_256_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  macsec_crypto_esp_aes_gcm_128_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  reserved_at_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 7]
  log_max_macsec_offload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 11]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  min_log_macsec_full_replay_window: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  max_log_macsec_full_replay_window: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1984]], 64]
@c.record
class struct_mlx5_ifc_psp_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  psp_crypto_offload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  psp_crypto_esp_aes_gcm_256_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  psp_crypto_esp_aes_gcm_128_encrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  psp_crypto_esp_aes_gcm_256_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  psp_crypto_esp_aes_gcm_128_decrypt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  reserved_at_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 7]
  log_max_num_of_psp_spi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 11]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2016]], 32]
class _anonenum17(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_WQ_TYPE_LINKED_LIST = _anonenum17.define('MLX5_WQ_TYPE_LINKED_LIST', 0)
MLX5_WQ_TYPE_CYCLIC = _anonenum17.define('MLX5_WQ_TYPE_CYCLIC', 1)
MLX5_WQ_TYPE_LINKED_LIST_STRIDING_RQ = _anonenum17.define('MLX5_WQ_TYPE_LINKED_LIST_STRIDING_RQ', 2)
MLX5_WQ_TYPE_CYCLIC_STRIDING_RQ = _anonenum17.define('MLX5_WQ_TYPE_CYCLIC_STRIDING_RQ', 3)

class _anonenum18(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_WQ_END_PAD_MODE_NONE = _anonenum18.define('MLX5_WQ_END_PAD_MODE_NONE', 0)
MLX5_WQ_END_PAD_MODE_ALIGN = _anonenum18.define('MLX5_WQ_END_PAD_MODE_ALIGN', 1)

class _anonenum19(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_8_GID_ENTRIES = _anonenum19.define('MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_8_GID_ENTRIES', 0)
MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_16_GID_ENTRIES = _anonenum19.define('MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_16_GID_ENTRIES', 1)
MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_32_GID_ENTRIES = _anonenum19.define('MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_32_GID_ENTRIES', 2)
MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_64_GID_ENTRIES = _anonenum19.define('MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_64_GID_ENTRIES', 3)
MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_128_GID_ENTRIES = _anonenum19.define('MLX5_CMD_HCA_CAP_GID_TABLE_SIZE_128_GID_ENTRIES', 4)

class _anonenum20(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_128_ENTRIES = _anonenum20.define('MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_128_ENTRIES', 0)
MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_256_ENTRIES = _anonenum20.define('MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_256_ENTRIES', 1)
MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_512_ENTRIES = _anonenum20.define('MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_512_ENTRIES', 2)
MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_1K_ENTRIES = _anonenum20.define('MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_1K_ENTRIES', 3)
MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_2K_ENTRIES = _anonenum20.define('MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_2K_ENTRIES', 4)
MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_4K_ENTRIES = _anonenum20.define('MLX5_CMD_HCA_CAP_PKEY_TABLE_SIZE_4K_ENTRIES', 5)

class _anonenum21(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CMD_HCA_CAP_PORT_TYPE_IB = _anonenum21.define('MLX5_CMD_HCA_CAP_PORT_TYPE_IB', 0)
MLX5_CMD_HCA_CAP_PORT_TYPE_ETHERNET = _anonenum21.define('MLX5_CMD_HCA_CAP_PORT_TYPE_ETHERNET', 1)

class _anonenum22(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_DISABLED = _anonenum22.define('MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_DISABLED', 0)
MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_INITIAL_STATE = _anonenum22.define('MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_INITIAL_STATE', 1)
MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_ENABLED = _anonenum22.define('MLX5_CMD_HCA_CAP_CMDIF_CHECKSUM_ENABLED', 3)

class _anonenum23(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CAP_PORT_TYPE_IB = _anonenum23.define('MLX5_CAP_PORT_TYPE_IB', 0)
MLX5_CAP_PORT_TYPE_ETH = _anonenum23.define('MLX5_CAP_PORT_TYPE_ETH', 1)

class _anonenum24(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CAP_UMR_FENCE_STRONG = _anonenum24.define('MLX5_CAP_UMR_FENCE_STRONG', 0)
MLX5_CAP_UMR_FENCE_SMALL = _anonenum24.define('MLX5_CAP_UMR_FENCE_SMALL', 1)
MLX5_CAP_UMR_FENCE_NONE = _anonenum24.define('MLX5_CAP_UMR_FENCE_NONE', 2)

class _anonenum25(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FLEX_IPV4_OVER_VXLAN_ENABLED = _anonenum25.define('MLX5_FLEX_IPV4_OVER_VXLAN_ENABLED', 1)
MLX5_FLEX_IPV6_OVER_VXLAN_ENABLED = _anonenum25.define('MLX5_FLEX_IPV6_OVER_VXLAN_ENABLED', 2)
MLX5_FLEX_IPV6_OVER_IP_ENABLED = _anonenum25.define('MLX5_FLEX_IPV6_OVER_IP_ENABLED', 4)
MLX5_FLEX_PARSER_GENEVE_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_GENEVE_ENABLED', 8)
MLX5_FLEX_PARSER_MPLS_OVER_GRE_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_MPLS_OVER_GRE_ENABLED', 16)
MLX5_FLEX_PARSER_MPLS_OVER_UDP_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_MPLS_OVER_UDP_ENABLED', 32)
MLX5_FLEX_P_BIT_VXLAN_GPE_ENABLED = _anonenum25.define('MLX5_FLEX_P_BIT_VXLAN_GPE_ENABLED', 64)
MLX5_FLEX_PARSER_VXLAN_GPE_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_VXLAN_GPE_ENABLED', 128)
MLX5_FLEX_PARSER_ICMP_V4_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_ICMP_V4_ENABLED', 256)
MLX5_FLEX_PARSER_ICMP_V6_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_ICMP_V6_ENABLED', 512)
MLX5_FLEX_PARSER_GENEVE_TLV_OPTION_0_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_GENEVE_TLV_OPTION_0_ENABLED', 1024)
MLX5_FLEX_PARSER_GTPU_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_GTPU_ENABLED', 2048)
MLX5_FLEX_PARSER_GTPU_DW_2_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_GTPU_DW_2_ENABLED', 65536)
MLX5_FLEX_PARSER_GTPU_FIRST_EXT_DW_0_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_GTPU_FIRST_EXT_DW_0_ENABLED', 131072)
MLX5_FLEX_PARSER_GTPU_DW_0_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_GTPU_DW_0_ENABLED', 262144)
MLX5_FLEX_PARSER_GTPU_TEID_ENABLED = _anonenum25.define('MLX5_FLEX_PARSER_GTPU_TEID_ENABLED', 524288)

class _anonenum26(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_UCTX_CAP_RAW_TX = _anonenum26.define('MLX5_UCTX_CAP_RAW_TX', 1)
MLX5_UCTX_CAP_INTERNAL_DEV_RES = _anonenum26.define('MLX5_UCTX_CAP_INTERNAL_DEV_RES', 2)
MLX5_UCTX_CAP_RDMA_CTRL = _anonenum26.define('MLX5_UCTX_CAP_RDMA_CTRL', 8)
MLX5_UCTX_CAP_RDMA_CTRL_OTHER_VHCA = _anonenum26.define('MLX5_UCTX_CAP_RDMA_CTRL_OTHER_VHCA', 16)

class enum_mlx5_fc_bulk_alloc_bitmask(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FC_BULK_128 = enum_mlx5_fc_bulk_alloc_bitmask.define('MLX5_FC_BULK_128', 1)
MLX5_FC_BULK_256 = enum_mlx5_fc_bulk_alloc_bitmask.define('MLX5_FC_BULK_256', 2)
MLX5_FC_BULK_512 = enum_mlx5_fc_bulk_alloc_bitmask.define('MLX5_FC_BULK_512', 4)
MLX5_FC_BULK_1024 = enum_mlx5_fc_bulk_alloc_bitmask.define('MLX5_FC_BULK_1024', 8)
MLX5_FC_BULK_2048 = enum_mlx5_fc_bulk_alloc_bitmask.define('MLX5_FC_BULK_2048', 16)
MLX5_FC_BULK_4096 = enum_mlx5_fc_bulk_alloc_bitmask.define('MLX5_FC_BULK_4096', 32)
MLX5_FC_BULK_8192 = enum_mlx5_fc_bulk_alloc_bitmask.define('MLX5_FC_BULK_8192', 64)
MLX5_FC_BULK_16384 = enum_mlx5_fc_bulk_alloc_bitmask.define('MLX5_FC_BULK_16384', 128)

class _anonenum27(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_STEERING_FORMAT_CONNECTX_5 = _anonenum27.define('MLX5_STEERING_FORMAT_CONNECTX_5', 0)
MLX5_STEERING_FORMAT_CONNECTX_6DX = _anonenum27.define('MLX5_STEERING_FORMAT_CONNECTX_6DX', 1)
MLX5_STEERING_FORMAT_CONNECTX_7 = _anonenum27.define('MLX5_STEERING_FORMAT_CONNECTX_7', 2)
MLX5_STEERING_FORMAT_CONNECTX_8 = _anonenum27.define('MLX5_STEERING_FORMAT_CONNECTX_8', 3)

@c.record
class struct_mlx5_ifc_cmd_hca_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 0]
  page_request_disable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  abs_native_port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  shared_object_to_user_object_allowed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  reserved_at_13: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 17]
  vhca_resource_manager: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
  hca_cap_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  create_lag_when_not_master_up: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  dtor: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  event_on_vhca_state_teardown_request: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 35]
  event_on_vhca_state_in_use: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 36]
  event_on_vhca_state_active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 37]
  event_on_vhca_state_allocated: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 38]
  event_on_vhca_state_invalid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 39]
  reserved_at_28: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  log_max_srq_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  log_max_qp_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  event_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 144]
  reserved_at_91: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 145]
  isolate_vl_tc_new: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 147]
  reserved_at_94: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 148]
  prio_tag_required: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 152]
  reserved_at_99: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 153]
  log_max_qp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 155]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 160]
  ece_support: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 163]
  reserved_at_a4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 164]
  reg_c_preserve: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 169]
  reserved_at_aa: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 170]
  log_max_srq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 171]
  reserved_at_b0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 176]
  uplink_follow: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 177]
  ts_cqe_to_dest_cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 178]
  reserved_at_b3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 179]
  go_back_n: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 185]
  reserved_at_ba: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 186]
  max_sgl_for_optimized_performance: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  log_max_cq_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 200]
  relaxed_ordering_write_umr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 208]
  relaxed_ordering_read_umr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 209]
  reserved_at_d2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 210]
  virtio_net_device_emualtion_manager: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 217]
  virtio_blk_device_emualtion_manager: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 218]
  log_max_cq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 219]
  log_max_eq_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 224]
  relaxed_ordering_write: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 232]
  relaxed_ordering_read_pci_enabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 233]
  log_max_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 234]
  reserved_at_f0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 240]
  terminate_scatter_list_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 246]
  repeated_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 247]
  dump_fill_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 248]
  reserved_at_f9: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 249]
  fast_teardown: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 251]
  log_max_eq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 252]
  max_indirection: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  fixed_buffer_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 264]
  log_max_mrw_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 265]
  force_teardown: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 272]
  reserved_at_111: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 273]
  log_max_bsf_list_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 274]
  umr_extended_translation_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 280]
  null_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 281]
  log_max_klm_list_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 282]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 288]
  qpc_extension: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 290]
  reserved_at_123: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 291]
  log_max_ra_req_dc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 298]
  reserved_at_130: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 304]
  eth_wqe_too_small: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 306]
  reserved_at_133: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 307]
  vnic_env_cq_overrun: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 313]
  log_max_ra_res_dc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 314]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 320]
  release_all_pages: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 325]
  must_not_use: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 326]
  reserved_at_147: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 327]
  roce_accl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 329]
  log_max_ra_req_qp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 330]
  reserved_at_150: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 336]
  log_max_ra_res_qp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 346]
  end_pad: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 352]
  cc_query_allowed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 353]
  cc_modify_allowed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 354]
  start_pad: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 355]
  cache_line_128byte: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 356]
  reserved_at_165: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 357]
  rts2rts_qp_counters_set_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 361]
  reserved_at_16a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 362]
  vnic_env_int_rq_oob: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 364]
  sbcam_reg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 365]
  reserved_at_16e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 366]
  qcam_reg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 367]
  gid_table_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 368]
  out_of_seq_cnt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 384]
  vport_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 385]
  retransmission_q_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 386]
  debug: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 387]
  modify_rq_counter_set_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 388]
  rq_delay_drop: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 389]
  max_qp_cnt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 390]
  pkey_table_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 400]
  vport_group_manager: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 416]
  vhca_group_manager: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 417]
  ib_virt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 418]
  eth_virt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 419]
  vnic_env_queue_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 420]
  ets: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 421]
  nic_flow_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 422]
  eswitch_manager: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 423]
  device_memory: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 424]
  mcam_reg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 425]
  pcam_reg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 426]
  local_ca_ack_delay: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 427]
  port_module_event: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 432]
  enhanced_error_q_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 433]
  ports_check: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 434]
  reserved_at_1b3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 435]
  disable_link_up: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 436]
  beacon_led: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 437]
  port_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 438]
  num_ports: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 440]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 448]
  pps: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 449]
  pps_modify: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 450]
  log_max_msg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 451]
  reserved_at_1c8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 456]
  max_tc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 460]
  temp_warn_event: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 464]
  dcbx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 465]
  general_notification_event: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 466]
  reserved_at_1d3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 467]
  fpga: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 469]
  rol_s: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 470]
  rol_g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 471]
  reserved_at_1d8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 472]
  wol_s: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 473]
  wol_g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 474]
  wol_a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 475]
  wol_b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 476]
  wol_m: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 477]
  wol_u: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 478]
  wol_p: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 479]
  stat_rate_support: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 480]
  reserved_at_1f0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 496]
  pci_sync_for_fw_update_event: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 497]
  reserved_at_1f2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 498]
  init2_lag_tx_port_affinity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 504]
  reserved_at_1fa: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 505]
  wqe_based_flow_table_update_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 507]
  cqe_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 508]
  compact_address_vector: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 512]
  striding_rq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 513]
  reserved_at_202: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 514]
  ipoib_enhanced_offloads: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 515]
  ipoib_basic_offloads: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 516]
  reserved_at_205: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 517]
  repeated_block_disabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 518]
  umr_modify_entity_size_disabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 519]
  umr_modify_atomic_disabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 520]
  umr_indirect_mkey_disabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 521]
  umr_fence: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 522]
  dc_req_scat_data_cqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 524]
  reserved_at_20d: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 525]
  drain_sigerr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 527]
  cmdif_checksum: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 528]
  sigerr_cqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 530]
  reserved_at_213: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 531]
  wq_signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 532]
  sctr_data_cqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 533]
  reserved_at_216: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 534]
  sho: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 535]
  tph: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 536]
  rf: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 537]
  dct: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 538]
  qos: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 539]
  eth_net_offloads: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 540]
  roce: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 541]
  atomic: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 542]
  reserved_at_21f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 543]
  cq_oi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 544]
  cq_resize: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 545]
  cq_moderation: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 546]
  cq_period_mode_modify: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 547]
  reserved_at_224: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 548]
  cq_eq_remap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 550]
  pg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 551]
  block_lb_mc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 552]
  reserved_at_229: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 553]
  scqe_break_moderation: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 554]
  cq_period_start_from_cqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 555]
  cd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 556]
  reserved_at_22d: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 557]
  apm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 558]
  vector_calc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 559]
  umr_ptr_rlky: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 560]
  imaicl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 561]
  qp_packet_based: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 562]
  reserved_at_233: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 563]
  qkv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 566]
  pkv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 567]
  set_deth_sqpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 568]
  reserved_at_239: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 569]
  xrc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 572]
  ud: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 573]
  uc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 574]
  rc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 575]
  uar_4k: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 576]
  reserved_at_241: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 577]
  fl_rc_qp_when_roce_disabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 584]
  regexp_params: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 585]
  uar_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 586]
  port_selection_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 592]
  nic_cap_reg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 593]
  umem_uid_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 594]
  reserved_at_253: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 595]
  log_pg_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 600]
  bf: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 608]
  driver_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 609]
  pad_tx_eth_packet: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 610]
  reserved_at_263: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 611]
  mkey_by_name: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 614]
  reserved_at_267: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 615]
  log_bf_reg_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 619]
  disciplined_fr_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 624]
  reserved_at_271: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 625]
  qp_error_syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 627]
  reserved_at_274: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 628]
  lag_dct: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 630]
  lag_tx_port_affinity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 632]
  lag_native_fdb_selection: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 633]
  reserved_at_27a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 634]
  lag_master: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 635]
  num_lag_ports: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 636]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 640]
  max_wqe_sz_sq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 656]
  reserved_at_2a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 672]
  mkey_pcie_tph: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 679]
  reserved_at_2a8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 680]
  tis_tir_td_order: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 681]
  psp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 682]
  shampo: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 683]
  reserved_at_2ac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 684]
  max_wqe_sz_rq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 688]
  max_flow_counter_31_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 704]
  max_wqe_sz_sq_dc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 720]
  reserved_at_2e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 736]
  max_qp_mcg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[25]], 743]
  reserved_at_300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 768]
  flow_counter_bulk_alloc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 784]
  log_max_mcg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 792]
  reserved_at_320: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 800]
  log_max_transport_domain: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 803]
  reserved_at_328: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 808]
  relaxed_ordering_read: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 810]
  log_max_pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 811]
  dp_ordering_ooo_all_ud: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 816]
  dp_ordering_ooo_all_uc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 817]
  dp_ordering_ooo_all_xrc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 818]
  dp_ordering_ooo_all_dc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 819]
  dp_ordering_ooo_all_rc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 820]
  pcie_reset_using_hotreset_method: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 821]
  pci_sync_for_fw_update_with_driver_unload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 822]
  vnic_env_cnt_steering_fail: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 823]
  vport_counter_local_loopback: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 824]
  q_counter_aggregation: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 825]
  q_counter_other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 826]
  log_max_xrcd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 827]
  nic_receive_steering_discard: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 832]
  receive_discard_vport_down: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 833]
  transmit_discard_vport_down: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 834]
  eq_overrun_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 835]
  reserved_at_344: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 836]
  invalid_command_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 837]
  quota_exceeded_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 838]
  reserved_at_347: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 839]
  log_max_flow_counter_bulk: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 840]
  max_flow_counter_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 848]
  reserved_at_360: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 864]
  log_max_rq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 867]
  reserved_at_368: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 872]
  log_max_sq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 875]
  reserved_at_370: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 880]
  log_max_tir: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 883]
  reserved_at_378: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 888]
  log_max_tis: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 891]
  basic_cyclic_rcv_wqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 896]
  reserved_at_381: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 897]
  log_max_rmp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 899]
  reserved_at_388: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 904]
  log_max_rqt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 907]
  reserved_at_390: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 912]
  log_max_rqt_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 915]
  reserved_at_398: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 920]
  vnic_env_cnt_bar_uar_access: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 921]
  vnic_env_cnt_odp_page_fault: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 922]
  log_max_tis_per_sq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 923]
  ext_stride_num_range: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 928]
  roce_rw_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 929]
  log_max_current_uc_list_wr_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 930]
  log_max_stride_sz_rq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 931]
  reserved_at_3a8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 936]
  log_min_stride_sz_rq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 939]
  reserved_at_3b0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 944]
  log_max_stride_sz_sq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 947]
  reserved_at_3b8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 952]
  log_min_stride_sz_sq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 955]
  hairpin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 960]
  reserved_at_3c1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 961]
  log_max_hairpin_queues: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 963]
  reserved_at_3c8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 968]
  log_max_hairpin_wq_data_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 971]
  reserved_at_3d0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 976]
  log_max_hairpin_num_packets: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 979]
  reserved_at_3d8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 984]
  log_max_wq_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 987]
  nic_vport_change_event: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 992]
  disable_local_lb_uc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 993]
  disable_local_lb_mc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 994]
  log_min_hairpin_wq_data_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 995]
  reserved_at_3e8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1000]
  silent_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1001]
  vhca_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1002]
  log_max_vlan_list: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1003]
  reserved_at_3f0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1008]
  log_max_current_mc_list: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1011]
  reserved_at_3f8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1016]
  log_max_current_uc_list: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1019]
  general_obj_types: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1024]
  sq_ts_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1088]
  rq_ts_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1090]
  steering_format_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1092]
  create_qp_start_hint: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1096]
  reserved_at_460: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1120]
  ats: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1121]
  cross_vhca_rqt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1122]
  log_max_uctx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1123]
  reserved_at_468: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1128]
  crypto: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1129]
  ipsec_offload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1130]
  log_max_umem: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1131]
  max_num_eqs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1136]
  reserved_at_480: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1152]
  tls_tx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1153]
  tls_rx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1154]
  log_max_l2_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1155]
  reserved_at_488: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1160]
  log_uar_page_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1168]
  reserved_at_4a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1184]
  device_frequency_mhz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1216]
  device_frequency_khz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1248]
  reserved_at_500: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1280]
  num_of_uars_per_page: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1312]
  flex_parser_protocols: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1344]
  max_geneve_tlv_options: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1376]
  reserved_at_568: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1384]
  max_geneve_tlv_option_data_len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1387]
  reserved_at_570: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1392]
  adv_rdma: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1393]
  reserved_at_572: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 1394]
  adv_virtualization: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1401]
  reserved_at_57a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 1402]
  reserved_at_580: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 1408]
  log_max_dci_stream_channels: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1419]
  reserved_at_590: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1424]
  log_max_dci_errored_streams: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1427]
  reserved_at_598: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1432]
  reserved_at_5a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1440]
  enhanced_cqe_compression: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1456]
  reserved_at_5b1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1457]
  crossing_vhca_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1458]
  log_max_dek: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1459]
  reserved_at_5b8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1464]
  mini_cqe_resp_stride_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1468]
  cqe_128_always: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1469]
  cqe_compression_128: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1470]
  cqe_compression: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1471]
  cqe_compression_timeout: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1472]
  cqe_compression_max_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1488]
  reserved_at_5e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1504]
  flex_parser_id_gtpu_dw_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1512]
  reserved_at_5ec: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1516]
  tag_matching: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1520]
  rndv_offload_rc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1521]
  rndv_offload_dc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1522]
  log_tag_matching_list_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1523]
  reserved_at_5f8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1528]
  log_max_xrq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1531]
  affiliate_nic_vport_criteria: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1536]
  native_port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1544]
  num_vhca_ports: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1552]
  flex_parser_id_gtpu_teid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1560]
  reserved_at_61c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1564]
  sw_owner_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1566]
  reserved_at_61f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1567]
  max_num_of_monitor_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1568]
  num_ppcnt_monitor_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1584]
  max_num_sf: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1600]
  num_q_monitor_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1616]
  reserved_at_660: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1632]
  sf: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1664]
  sf_set_partition: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1665]
  reserved_at_682: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1666]
  log_max_sf: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1667]
  apu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1672]
  reserved_at_689: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1673]
  migration: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1677]
  reserved_at_68e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1678]
  log_min_sf_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1680]
  max_num_sf_partitions: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1688]
  uctx_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1696]
  reserved_at_6c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1728]
  flex_parser_id_geneve_tlv_option_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1732]
  flex_parser_id_icmp_dw1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1736]
  flex_parser_id_icmp_dw0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1740]
  flex_parser_id_icmpv6_dw1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1744]
  flex_parser_id_icmpv6_dw0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1748]
  flex_parser_id_outer_first_mpls_over_gre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1752]
  flex_parser_id_outer_first_mpls_over_udp_label: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1756]
  max_num_match_definer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1760]
  sf_base_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1776]
  flex_parser_id_gtpu_dw_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1792]
  flex_parser_id_gtpu_first_ext_dw_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1796]
  num_total_dynamic_vf_msix: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1800]
  reserved_at_720: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 1824]
  dynamic_msix_table_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 1844]
  reserved_at_740: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 1856]
  min_dynamic_vf_msix_table_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1868]
  reserved_at_750: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1872]
  data_direct: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1874]
  reserved_at_753: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1875]
  max_dynamic_vf_msix_table_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 1876]
  reserved_at_760: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1888]
  log_max_num_header_modify_argument: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1891]
  log_header_modify_argument_granularity_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1896]
  log_header_modify_argument_granularity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1900]
  reserved_at_770: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1904]
  log_header_modify_argument_max_alloc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1907]
  reserved_at_778: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1912]
  vhca_tunnel_commands: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1920]
  match_definer_format_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1984]
class _anonenum28(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CROSS_VHCA_OBJ_TO_OBJ_SUPPORTED_LOCAL_FLOW_TABLE_TO_REMOTE_FLOW_TABLE_MISS = _anonenum28.define('MLX5_CROSS_VHCA_OBJ_TO_OBJ_SUPPORTED_LOCAL_FLOW_TABLE_TO_REMOTE_FLOW_TABLE_MISS', 524288)
MLX5_CROSS_VHCA_OBJ_TO_OBJ_SUPPORTED_LOCAL_FLOW_TABLE_ROOT_TO_REMOTE_FLOW_TABLE = _anonenum28.define('MLX5_CROSS_VHCA_OBJ_TO_OBJ_SUPPORTED_LOCAL_FLOW_TABLE_ROOT_TO_REMOTE_FLOW_TABLE', 1048576)

class _anonenum29(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ALLOWED_OBJ_FOR_OTHER_VHCA_ACCESS_FLOW_TABLE = _anonenum29.define('MLX5_ALLOWED_OBJ_FOR_OTHER_VHCA_ACCESS_FLOW_TABLE', 512)

@c.record
class struct_mlx5_ifc_cmd_hca_cap_2_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
  migratable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 128]
  reserved_at_81: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 129]
  dp_ordering_force: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 136]
  reserved_at_89: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 137]
  query_vuid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 146]
  reserved_at_93: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 147]
  umr_log_entity_size_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 152]
  reserved_at_99: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 153]
  max_reformat_insert_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  max_reformat_insert_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 168]
  max_reformat_remove_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 176]
  max_reformat_remove_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 184]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  migration_multi_load: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 200]
  migration_tracking_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 201]
  multiplane_qp_ud: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 202]
  reserved_at_cb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 203]
  migration_in_chunks: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 208]
  reserved_at_d1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 209]
  sf_eq_usage: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 210]
  reserved_at_d3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 211]
  multiplane: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 216]
  reserved_at_d9: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 217]
  cross_vhca_object_to_object_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  allowed_object_for_other_vhca_access: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 256]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 320]
  flow_table_type_2_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 416]
  reserved_at_1a8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 424]
  format_select_dw_8_6_ext: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 426]
  log_min_mkey_entity_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 427]
  reserved_at_1b0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 432]
  general_obj_types_127_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
  reserved_at_200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  reserved_at_220: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 544]
  sw_vhca_id_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 545]
  sw_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 546]
  reserved_at_230: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 560]
  reserved_at_240: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 576]
  ts_cqe_metadata_size2wqe_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 587]
  reserved_at_250: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 592]
  reserved_at_260: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  format_select_dw_gtpu_dw_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 640]
  format_select_dw_gtpu_dw_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 648]
  format_select_dw_gtpu_dw_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 656]
  format_select_dw_gtpu_first_ext_dw_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 664]
  generate_wqe_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  reserved_at_2c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 704]
  reserved_at_380: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 896]
  min_mkey_log_entity_size_fixed_buffer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 907]
  ec_vf_vport_base: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 912]
  reserved_at_3a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 928]
  max_mkey_log_entity_size_fixed_buffer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 930]
  reserved_at_3a8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 936]
  max_mkey_log_entity_size_mtt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 938]
  max_rqt_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 944]
  reserved_at_3c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  reserved_at_3e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 992]
  pcc_ifa2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1008]
  reserved_at_3f1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 1009]
  reserved_at_400: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1024]
  min_mkey_log_entity_size_fixed_buffer_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1025]
  reserved_at_402: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 1026]
  return_reg_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1040]
  reserved_at_420: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 1056]
  flow_table_hash_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1084]
  reserved_at_440: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1088]
  max_num_eqs_24b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1096]
  reserved_at_460: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[324]], 1120]
  load_balance_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1444]
  reserved_at_5a8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1448]
  query_adjacent_functions_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1472]
  ingress_egress_esw_vport_connect: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1473]
  function_id_type_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1474]
  reserved_at_5c3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1475]
  lag_per_mp_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1476]
  reserved_at_5c5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 1477]
  delegate_vhca_management_profiles: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1488]
  delegated_vhca_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1504]
  delegate_vhca_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1520]
  reserved_at_600: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[512]], 1536]
class enum_mlx5_ifc_flow_destination_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_IFC_FLOW_DESTINATION_TYPE_VPORT = enum_mlx5_ifc_flow_destination_type.define('MLX5_IFC_FLOW_DESTINATION_TYPE_VPORT', 0)
MLX5_IFC_FLOW_DESTINATION_TYPE_FLOW_TABLE = enum_mlx5_ifc_flow_destination_type.define('MLX5_IFC_FLOW_DESTINATION_TYPE_FLOW_TABLE', 1)
MLX5_IFC_FLOW_DESTINATION_TYPE_TIR = enum_mlx5_ifc_flow_destination_type.define('MLX5_IFC_FLOW_DESTINATION_TYPE_TIR', 2)
MLX5_IFC_FLOW_DESTINATION_TYPE_FLOW_SAMPLER = enum_mlx5_ifc_flow_destination_type.define('MLX5_IFC_FLOW_DESTINATION_TYPE_FLOW_SAMPLER', 6)
MLX5_IFC_FLOW_DESTINATION_TYPE_UPLINK = enum_mlx5_ifc_flow_destination_type.define('MLX5_IFC_FLOW_DESTINATION_TYPE_UPLINK', 8)
MLX5_IFC_FLOW_DESTINATION_TYPE_TABLE_TYPE = enum_mlx5_ifc_flow_destination_type.define('MLX5_IFC_FLOW_DESTINATION_TYPE_TABLE_TYPE', 10)

class enum_mlx5_flow_table_miss_action(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FLOW_TABLE_MISS_ACTION_DEF = enum_mlx5_flow_table_miss_action.define('MLX5_FLOW_TABLE_MISS_ACTION_DEF', 0)
MLX5_FLOW_TABLE_MISS_ACTION_FWD = enum_mlx5_flow_table_miss_action.define('MLX5_FLOW_TABLE_MISS_ACTION_FWD', 1)
MLX5_FLOW_TABLE_MISS_ACTION_SWITCH_DOMAIN = enum_mlx5_flow_table_miss_action.define('MLX5_FLOW_TABLE_MISS_ACTION_SWITCH_DOMAIN', 2)

@c.record
class struct_mlx5_ifc_dest_format_struct_bits(c.Struct):
  SIZE = 64
  destination_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  destination_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  destination_eswitch_owner_vhca_id_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  packet_reformat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 34]
  destination_table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  destination_eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
@c.record
class struct_mlx5_ifc_flow_counter_list_bits(c.Struct):
  SIZE = 64
  flow_counter_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_extended_dest_format_bits(c.Struct):
  SIZE = 128
  destination_entry: Annotated[struct_mlx5_ifc_dest_format_struct_bits, 0]
  packet_reformat_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class union_mlx5_ifc_dest_format_flow_counter_list_auto_bits(c.Struct):
  SIZE = 128
  extended_dest_format: Annotated[struct_mlx5_ifc_extended_dest_format_bits, 0]
  flow_counter_list: Annotated[struct_mlx5_ifc_flow_counter_list_bits, 0]
@c.record
class struct_mlx5_ifc_fte_match_param_bits(c.Struct):
  SIZE = 4096
  outer_headers: Annotated[struct_mlx5_ifc_fte_match_set_lyr_2_4_bits, 0]
  misc_parameters: Annotated[struct_mlx5_ifc_fte_match_set_misc_bits, 512]
  inner_headers: Annotated[struct_mlx5_ifc_fte_match_set_lyr_2_4_bits, 1024]
  misc_parameters_2: Annotated[struct_mlx5_ifc_fte_match_set_misc2_bits, 1536]
  misc_parameters_3: Annotated[struct_mlx5_ifc_fte_match_set_misc3_bits, 2048]
  misc_parameters_4: Annotated[struct_mlx5_ifc_fte_match_set_misc4_bits, 2560]
  misc_parameters_5: Annotated[struct_mlx5_ifc_fte_match_set_misc5_bits, 3072]
  reserved_at_e00: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[512]], 3584]
class _anonenum30(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_SRC_IP = _anonenum30.define('MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_SRC_IP', 0)
MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_DST_IP = _anonenum30.define('MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_DST_IP', 1)
MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_L4_SPORT = _anonenum30.define('MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_L4_SPORT', 2)
MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_L4_DPORT = _anonenum30.define('MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_L4_DPORT', 3)
MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_IPSEC_SPI = _anonenum30.define('MLX5_RX_HASH_FIELD_SELECT_SELECTED_FIELDS_IPSEC_SPI', 4)

@c.record
class struct_mlx5_ifc_rx_hash_field_select_bits(c.Struct):
  SIZE = 32
  l3_prot_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  l4_prot_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  selected_fields: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 2]
class _anonenum31(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_WQ_WQ_TYPE_WQ_LINKED_LIST = _anonenum31.define('MLX5_WQ_WQ_TYPE_WQ_LINKED_LIST', 0)
MLX5_WQ_WQ_TYPE_WQ_CYCLIC = _anonenum31.define('MLX5_WQ_WQ_TYPE_WQ_CYCLIC', 1)

class _anonenum32(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_WQ_END_PADDING_MODE_END_PAD_NONE = _anonenum32.define('MLX5_WQ_END_PADDING_MODE_END_PAD_NONE', 0)
MLX5_WQ_END_PADDING_MODE_END_PAD_ALIGN = _anonenum32.define('MLX5_WQ_END_PADDING_MODE_END_PAD_ALIGN', 1)

@c.record
class struct_mlx5_ifc_wq_bits(c.Struct):
  SIZE = 1536
  wq_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  wq_signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  end_padding_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 5]
  cd_slave: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  hds_skip_first_sge: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  log2_hds_buf_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 33]
  reserved_at_24: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 36]
  page_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 43]
  lwm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  uar_page: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  dbr_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  hw_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  sw_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 256]
  log_wq_stride: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 268]
  reserved_at_110: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 272]
  log_wq_pg_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 275]
  reserved_at_118: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 280]
  log_wq_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 283]
  dbr_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 288]
  wq_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 289]
  reserved_at_122: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 290]
  log_hairpin_num_packets: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 291]
  reserved_at_128: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 296]
  log_hairpin_data_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 299]
  reserved_at_130: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 304]
  log_wqe_num_of_strides: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 308]
  two_byte_shift_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 312]
  reserved_at_139: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 313]
  log_wqe_stride_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 317]
  dbr_umem_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  wq_umem_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  wq_umem_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 384]
  headers_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  shampo_enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 480]
  reserved_at_1e1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 481]
  shampo_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 482]
  reserved_at_1e4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 484]
  log_reservation_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 485]
  reserved_at_1e8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 488]
  log_max_num_of_packets_per_reservation: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 493]
  reserved_at_1f0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 496]
  log_headers_entry_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 502]
  reserved_at_1f8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 504]
  log_headers_buffer_entry_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 508]
  reserved_at_200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1024]], 512]
  pas: Annotated[c.Array[struct_mlx5_ifc_cmd_pas_bits, Literal[0]], 1536]
@c.record
class struct_mlx5_ifc_rq_num_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  rq_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
@c.record
class struct_mlx5_ifc_rq_vhca_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  rq_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  rq_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
@c.record
class struct_mlx5_ifc_mac_address_layout_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  mac_addr_47_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  mac_addr_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_vlan_layout_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 0]
  vlan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 20]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_cong_control_r_roce_ecn_np_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 0]
  min_time_between_cnps: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[18]], 192]
  cnp_dscp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 210]
  reserved_at_d8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 216]
  cnp_prio_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 220]
  cnp_802p_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 221]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1824]], 224]
@c.record
class struct_mlx5_ifc_cong_control_r_roce_ecn_rp_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 0]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 96]
  clamp_tgt_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 100]
  reserved_at_65: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 101]
  clamp_tgt_rate_after_time_inc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 104]
  reserved_at_69: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[23]], 105]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  rpg_time_reset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  rpg_byte_reset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  rpg_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  rpg_max_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  rpg_ai_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  rpg_hai_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  rpg_gd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  rpg_min_dec_fac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  rpg_min_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 448]
  rate_to_set_on_first_cnp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  dce_tcp_g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  dce_tcp_rtt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  rate_reduce_monitor_period: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  reserved_at_320: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 800]
  initial_alpha_value: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 832]
  reserved_at_360: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1184]], 864]
@c.record
class struct_mlx5_ifc_cong_control_r_roce_general_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  rtt_resp_dscp_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 144]
  reserved_at_91: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 145]
  rtt_resp_dscp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 154]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1888]], 160]
@c.record
class struct_mlx5_ifc_cong_control_802_1qau_rp_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
  rppp_max_rps: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  rpg_time_reset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  rpg_byte_reset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  rpg_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  rpg_max_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  rpg_ai_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  rpg_hai_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  rpg_gd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  rpg_min_dec_fac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  rpg_min_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1600]], 448]
class _anonenum33(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_LOG_CQ_SIZE = _anonenum33.define('MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_LOG_CQ_SIZE', 1)
MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_PAGE_OFFSET = _anonenum33.define('MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_PAGE_OFFSET', 2)
MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_LOG_PAGE_SIZE = _anonenum33.define('MLX5_RESIZE_FIELD_SELECT_RESIZE_FIELD_SELECT_LOG_PAGE_SIZE', 4)

@c.record
class struct_mlx5_ifc_resize_field_select_bits(c.Struct):
  SIZE = 32
  resize_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_resource_dump_bits(c.Struct):
  SIZE = 2048
  more_dump: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  inline_dump: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 2]
  seq_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
  segment_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  index1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  index2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  num_of_obj1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  num_of_obj2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  device_opaque: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
  inline_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[52]], 384]
@c.record
class struct_mlx5_ifc_resource_dump_menu_record_bits(c.Struct):
  SIZE = 416
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  num_of_obj2_supports_active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  num_of_obj2_supports_all: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  must_have_num_of_obj2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  support_num_of_obj2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  num_of_obj1_supports_active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  num_of_obj1_supports_all: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 9]
  must_have_num_of_obj1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 10]
  support_num_of_obj1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 11]
  must_have_index2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 12]
  support_index2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  must_have_index1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  support_index1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 15]
  segment_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  segment_name: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[4]], 32]
  index1_name: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[4]], 160]
  index2_name: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[4]], 288]
@c.record
class struct_mlx5_ifc_resource_dump_segment_header_bits(c.Struct):
  SIZE = 32
  length_dw: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  segment_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
@c.record
class struct_mlx5_ifc_resource_dump_command_segment_bits(c.Struct):
  SIZE = 160
  segment_header: Annotated[struct_mlx5_ifc_resource_dump_segment_header_bits, 0]
  segment_called: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  index1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  index2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  num_of_obj1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  num_of_obj2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
@c.record
class struct_mlx5_ifc_resource_dump_error_segment_bits(c.Struct):
  SIZE = 384
  segment_header: Annotated[struct_mlx5_ifc_resource_dump_segment_header_bits, 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  syndrome_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  error: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 128]
@c.record
class struct_mlx5_ifc_resource_dump_info_segment_bits(c.Struct):
  SIZE = 128
  segment_header: Annotated[struct_mlx5_ifc_resource_dump_segment_header_bits, 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 32]
  dump_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  hw_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  fw_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_resource_dump_menu_segment_bits(c.Struct):
  SIZE = 64
  segment_header: Annotated[struct_mlx5_ifc_resource_dump_segment_header_bits, 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  num_of_records: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  record: Annotated[c.Array[struct_mlx5_ifc_resource_dump_menu_record_bits, Literal[0]], 64]
@c.record
class struct_mlx5_ifc_resource_dump_resource_segment_bits(c.Struct):
  SIZE = 128
  segment_header: Annotated[struct_mlx5_ifc_resource_dump_segment_header_bits, 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  index1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  index2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  payload: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[0]], 128]
@c.record
class struct_mlx5_ifc_resource_dump_terminate_segment_bits(c.Struct):
  SIZE = 32
  segment_header: Annotated[struct_mlx5_ifc_resource_dump_segment_header_bits, 0]
@c.record
class struct_mlx5_ifc_menu_resource_dump_response_bits(c.Struct):
  SIZE = 384
  info: Annotated[struct_mlx5_ifc_resource_dump_info_segment_bits, 0]
  cmd: Annotated[struct_mlx5_ifc_resource_dump_command_segment_bits, 128]
  menu: Annotated[struct_mlx5_ifc_resource_dump_menu_segment_bits, 288]
  terminate: Annotated[struct_mlx5_ifc_resource_dump_terminate_segment_bits, 352]
class _anonenum34(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_CQ_PERIOD = _anonenum34.define('MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_CQ_PERIOD', 1)
MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_CQ_MAX_COUNT = _anonenum34.define('MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_CQ_MAX_COUNT', 2)
MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_OI = _anonenum34.define('MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_OI', 4)
MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_C_EQN = _anonenum34.define('MLX5_MODIFY_FIELD_SELECT_MODIFY_FIELD_SELECT_C_EQN', 8)

@c.record
class struct_mlx5_ifc_modify_field_select_bits(c.Struct):
  SIZE = 32
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_field_select_r_roce_np_bits(c.Struct):
  SIZE = 32
  field_select_r_roce_np: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_field_select_r_roce_rp_bits(c.Struct):
  SIZE = 32
  field_select_r_roce_rp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
class _anonenum35(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPPP_MAX_RPS = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPPP_MAX_RPS', 4)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_TIME_RESET = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_TIME_RESET', 8)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_BYTE_RESET = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_BYTE_RESET', 16)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_THRESHOLD = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_THRESHOLD', 32)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MAX_RATE = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MAX_RATE', 64)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_AI_RATE = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_AI_RATE', 128)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_HAI_RATE = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_HAI_RATE', 256)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_GD = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_GD', 512)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MIN_DEC_FAC = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MIN_DEC_FAC', 1024)
MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MIN_RATE = _anonenum35.define('MLX5_FIELD_SELECT_802_1QAU_RP_FIELD_SELECT_8021QAURP_RPG_MIN_RATE', 2048)

@c.record
class struct_mlx5_ifc_field_select_802_1qau_rp_bits(c.Struct):
  SIZE = 32
  field_select_8021qaurp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_phys_layer_recovery_cntrs_bits(c.Struct):
  SIZE = 1984
  total_successful_recovery_events: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1952]], 32]
@c.record
class struct_mlx5_ifc_phys_layer_cntrs_bits(c.Struct):
  SIZE = 1984
  time_since_last_clear_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  time_since_last_clear_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  symbol_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  symbol_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  sync_headers_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  sync_headers_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  edpl_bip_errors_lane0_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  edpl_bip_errors_lane0_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  edpl_bip_errors_lane1_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  edpl_bip_errors_lane1_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  edpl_bip_errors_lane2_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  edpl_bip_errors_lane2_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  edpl_bip_errors_lane3_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  edpl_bip_errors_lane3_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  fc_fec_corrected_blocks_lane0_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  fc_fec_corrected_blocks_lane0_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  fc_fec_corrected_blocks_lane1_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  fc_fec_corrected_blocks_lane1_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  fc_fec_corrected_blocks_lane2_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  fc_fec_corrected_blocks_lane2_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  fc_fec_corrected_blocks_lane3_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 640]
  fc_fec_corrected_blocks_lane3_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  fc_fec_uncorrectable_blocks_lane0_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  fc_fec_uncorrectable_blocks_lane0_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  fc_fec_uncorrectable_blocks_lane1_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  fc_fec_uncorrectable_blocks_lane1_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 800]
  fc_fec_uncorrectable_blocks_lane2_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 832]
  fc_fec_uncorrectable_blocks_lane2_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 864]
  fc_fec_uncorrectable_blocks_lane3_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 896]
  fc_fec_uncorrectable_blocks_lane3_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  rs_fec_corrected_blocks_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  rs_fec_corrected_blocks_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 992]
  rs_fec_uncorrectable_blocks_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1024]
  rs_fec_uncorrectable_blocks_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1056]
  rs_fec_no_errors_blocks_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1088]
  rs_fec_no_errors_blocks_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1120]
  rs_fec_single_error_blocks_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1152]
  rs_fec_single_error_blocks_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1184]
  rs_fec_corrected_symbols_total_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1216]
  rs_fec_corrected_symbols_total_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1248]
  rs_fec_corrected_symbols_lane0_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1280]
  rs_fec_corrected_symbols_lane0_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1312]
  rs_fec_corrected_symbols_lane1_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1344]
  rs_fec_corrected_symbols_lane1_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1376]
  rs_fec_corrected_symbols_lane2_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1408]
  rs_fec_corrected_symbols_lane2_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1440]
  rs_fec_corrected_symbols_lane3_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1472]
  rs_fec_corrected_symbols_lane3_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1504]
  link_down_events: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1536]
  successful_recovery_events: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1568]
  reserved_at_640: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 1600]
@c.record
class struct_mlx5_ifc_phys_layer_statistical_cntrs_bits(c.Struct):
  SIZE = 1984
  time_since_last_clear_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  time_since_last_clear_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  phy_received_bits_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  phy_received_bits_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  phy_symbol_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  phy_symbol_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  phy_corrected_bits_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  phy_corrected_bits_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  phy_corrected_bits_lane0_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  phy_corrected_bits_lane0_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  phy_corrected_bits_lane1_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  phy_corrected_bits_lane1_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  phy_corrected_bits_lane2_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  phy_corrected_bits_lane2_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  phy_corrected_bits_lane3_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  phy_corrected_bits_lane3_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  reserved_at_200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1472]], 512]
@c.record
class struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 320
  symbol_error_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  link_error_recovery_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  link_downed_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  port_rcv_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  port_rcv_remote_physical_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  port_rcv_switch_relay_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  port_xmit_discards: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  port_xmit_constraint_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  port_rcv_constraint_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 104]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 112]
  link_overrun_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  vl_15_dropped: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 160]
  port_xmit_wait: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
@c.record
class struct_mlx5_ifc_ib_ext_port_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[768]], 0]
  port_xmit_data_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  port_xmit_data_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 800]
  port_rcv_data_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 832]
  port_rcv_data_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 864]
  port_xmit_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 896]
  port_xmit_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  port_rcv_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  port_rcv_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 992]
  reserved_at_400: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 1024]
  port_unicast_xmit_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1152]
  port_unicast_xmit_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1184]
  port_multicast_xmit_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1216]
  port_multicast_xmit_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1248]
  port_unicast_rcv_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1280]
  port_unicast_rcv_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1312]
  port_multicast_rcv_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1344]
  port_multicast_rcv_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1376]
  reserved_at_580: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[576]], 1408]
@c.record
class struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  transmit_queue_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  transmit_queue_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  no_buffer_discard_uc_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  no_buffer_discard_uc_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1856]], 128]
@c.record
class struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  wred_discard_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  wred_discard_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  ecn_marked_tc_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  ecn_marked_tc_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1856]], 128]
@c.record
class struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  rx_octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  rx_octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  rx_frames_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  rx_frames_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  tx_octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  tx_octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 384]
  tx_frames_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  tx_frames_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  rx_pause_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 640]
  rx_pause_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  rx_pause_duration_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  rx_pause_duration_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  tx_pause_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  tx_pause_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 800]
  tx_pause_duration_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 832]
  tx_pause_duration_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 864]
  rx_pause_transition_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 896]
  rx_pause_transition_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  rx_discards_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  rx_discards_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 992]
  device_stall_minor_watermark_cnt_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1024]
  device_stall_minor_watermark_cnt_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1056]
  device_stall_critical_watermark_cnt_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1088]
  device_stall_critical_watermark_cnt_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1120]
  reserved_at_480: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[832]], 1152]
@c.record
class struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  port_transmit_wait_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  port_transmit_wait_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 64]
  rx_buffer_almost_full_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  rx_buffer_almost_full_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  rx_buffer_full_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  rx_buffer_full_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  rx_icrc_encapsulated_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  rx_icrc_encapsulated_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  reserved_at_200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1472]], 512]
@c.record
class struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  dot3stats_alignment_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  dot3stats_alignment_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  dot3stats_fcs_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  dot3stats_fcs_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  dot3stats_single_collision_frames_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  dot3stats_single_collision_frames_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  dot3stats_multiple_collision_frames_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  dot3stats_multiple_collision_frames_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  dot3stats_sqe_test_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  dot3stats_sqe_test_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  dot3stats_deferred_transmissions_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  dot3stats_deferred_transmissions_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  dot3stats_late_collisions_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  dot3stats_late_collisions_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  dot3stats_excessive_collisions_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  dot3stats_excessive_collisions_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  dot3stats_internal_mac_transmit_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  dot3stats_internal_mac_transmit_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  dot3stats_carrier_sense_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  dot3stats_carrier_sense_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  dot3stats_frame_too_longs_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 640]
  dot3stats_frame_too_longs_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  dot3stats_internal_mac_receive_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  dot3stats_internal_mac_receive_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  dot3stats_symbol_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  dot3stats_symbol_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 800]
  dot3control_in_unknown_opcodes_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 832]
  dot3control_in_unknown_opcodes_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 864]
  dot3in_pause_frames_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 896]
  dot3in_pause_frames_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  dot3out_pause_frames_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  dot3out_pause_frames_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 992]
  reserved_at_400: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[960]], 1024]
@c.record
class struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  ether_stats_drop_events_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  ether_stats_drop_events_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  ether_stats_octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  ether_stats_octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  ether_stats_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  ether_stats_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  ether_stats_broadcast_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  ether_stats_broadcast_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  ether_stats_multicast_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  ether_stats_multicast_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  ether_stats_crc_align_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  ether_stats_crc_align_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  ether_stats_undersize_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  ether_stats_undersize_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  ether_stats_oversize_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  ether_stats_oversize_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  ether_stats_fragments_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  ether_stats_fragments_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  ether_stats_jabbers_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  ether_stats_jabbers_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  ether_stats_collisions_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 640]
  ether_stats_collisions_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  ether_stats_pkts64octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  ether_stats_pkts64octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  ether_stats_pkts65to127octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  ether_stats_pkts65to127octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 800]
  ether_stats_pkts128to255octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 832]
  ether_stats_pkts128to255octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 864]
  ether_stats_pkts256to511octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 896]
  ether_stats_pkts256to511octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  ether_stats_pkts512to1023octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  ether_stats_pkts512to1023octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 992]
  ether_stats_pkts1024to1518octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1024]
  ether_stats_pkts1024to1518octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1056]
  ether_stats_pkts1519to2047octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1088]
  ether_stats_pkts1519to2047octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1120]
  ether_stats_pkts2048to4095octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1152]
  ether_stats_pkts2048to4095octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1184]
  ether_stats_pkts4096to8191octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1216]
  ether_stats_pkts4096to8191octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1248]
  ether_stats_pkts8192to10239octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1280]
  ether_stats_pkts8192to10239octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1312]
  reserved_at_540: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[640]], 1344]
@c.record
class struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  if_in_octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  if_in_octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  if_in_ucast_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  if_in_ucast_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  if_in_discards_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  if_in_discards_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  if_in_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  if_in_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  if_in_unknown_protos_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  if_in_unknown_protos_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  if_out_octets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  if_out_octets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  if_out_ucast_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  if_out_ucast_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  if_out_discards_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  if_out_discards_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  if_out_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  if_out_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  if_in_multicast_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  if_in_multicast_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  if_in_broadcast_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 640]
  if_in_broadcast_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  if_out_multicast_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  if_out_multicast_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  if_out_broadcast_pkts_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  if_out_broadcast_pkts_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 800]
  reserved_at_340: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1152]], 832]
@c.record
class struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  a_frames_transmitted_ok_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  a_frames_transmitted_ok_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  a_frames_received_ok_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  a_frames_received_ok_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  a_frame_check_sequence_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  a_frame_check_sequence_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  a_alignment_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  a_alignment_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  a_octets_transmitted_ok_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  a_octets_transmitted_ok_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  a_octets_received_ok_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  a_octets_received_ok_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  a_multicast_frames_xmitted_ok_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  a_multicast_frames_xmitted_ok_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  a_broadcast_frames_xmitted_ok_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  a_broadcast_frames_xmitted_ok_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  a_multicast_frames_received_ok_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  a_multicast_frames_received_ok_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  a_broadcast_frames_received_ok_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  a_broadcast_frames_received_ok_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  a_in_range_length_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 640]
  a_in_range_length_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  a_out_of_range_length_field_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  a_out_of_range_length_field_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  a_frame_too_long_errors_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  a_frame_too_long_errors_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 800]
  a_symbol_error_during_carrier_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 832]
  a_symbol_error_during_carrier_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 864]
  a_mac_control_frames_transmitted_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 896]
  a_mac_control_frames_transmitted_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  a_mac_control_frames_received_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  a_mac_control_frames_received_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 992]
  a_unsupported_opcodes_received_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1024]
  a_unsupported_opcodes_received_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1056]
  a_pause_mac_ctrl_frames_received_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1088]
  a_pause_mac_ctrl_frames_received_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1120]
  a_pause_mac_ctrl_frames_transmitted_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1152]
  a_pause_mac_ctrl_frames_transmitted_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1184]
  reserved_at_4c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[768]], 1216]
@c.record
class struct_mlx5_ifc_pcie_perf_cntrs_grp_data_layout_bits(c.Struct):
  SIZE = 1984
  life_time_counter_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  life_time_counter_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  rx_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  tx_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  l0_to_recovery_eieos: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  l0_to_recovery_ts: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  l0_to_recovery_framing: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  l0_to_recovery_retrain: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  crc_error_dllp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  crc_error_tlp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  tx_overflow_buffer_pkt_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  tx_overflow_buffer_pkt_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  outbound_stalled_reads: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  outbound_stalled_writes: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  outbound_stalled_reads_events: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  outbound_stalled_writes_events: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  reserved_at_200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1472]], 512]
@c.record
class struct_mlx5_ifc_cmd_inter_comp_event_bits(c.Struct):
  SIZE = 224
  command_completion_vector: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 32]
@c.record
class struct_mlx5_ifc_stall_vl_event_bits(c.Struct):
  SIZE = 192
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 0]
  port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 24]
  reserved_at_19: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 25]
  vl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 32]
@c.record
class struct_mlx5_ifc_db_bf_congestion_event_bits(c.Struct):
  SIZE = 192
  event_subtype: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  congestion_level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 32]
@c.record
class struct_mlx5_ifc_gpio_event_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 0]
  gpio_event_hi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  gpio_event_lo: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 160]
@c.record
class struct_mlx5_ifc_port_state_change_event_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 68]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 96]
@c.record
class struct_mlx5_ifc_dropped_packet_logged_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 0]
@c.record
class struct_mlx5_ifc_nic_cap_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 0]
  vhca_icm_ctrl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 26]
  reserved_at_1b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 27]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
@c.record
class struct_mlx5_ifc_default_timeout_bits(c.Struct):
  SIZE = 32
  to_multiplier: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 0]
  reserved_at_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 3]
  to_value: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 12]
@c.record
class struct_mlx5_ifc_dtor_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  pcie_toggle_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 64]
  health_poll_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 160]
  full_crdump_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 192]
  fw_reset_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 224]
  flush_on_err_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 256]
  pci_sync_update_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 288]
  tear_down_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 320]
  fsm_reactivate_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 352]
  reclaim_pages_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 384]
  reclaim_vfs_pages_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 416]
  reset_unload_to: Annotated[struct_mlx5_ifc_default_timeout_bits, 448]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
@c.record
class struct_mlx5_ifc_vhca_icm_ctrl_reg_bits(c.Struct):
  SIZE = 512
  vhca_id_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 1]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 32]
  cur_alloc_icm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[288]], 224]
class _anonenum36(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CQ_ERROR_SYNDROME_CQ_OVERRUN = _anonenum36.define('MLX5_CQ_ERROR_SYNDROME_CQ_OVERRUN', 1)
MLX5_CQ_ERROR_SYNDROME_CQ_ACCESS_VIOLATION_ERROR = _anonenum36.define('MLX5_CQ_ERROR_SYNDROME_CQ_ACCESS_VIOLATION_ERROR', 2)

@c.record
class struct_mlx5_ifc_cq_error_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 96]
@c.record
class struct_mlx5_ifc_rdma_page_fault_event_bits(c.Struct):
  SIZE = 224
  bytes_committed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  r_key: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  packet_len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  rdma_op_len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  rdma_va: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 192]
  rdma: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 197]
  write: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 198]
  requestor: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 199]
  qp_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
@c.record
class struct_mlx5_ifc_wqe_associated_page_fault_event_bits(c.Struct):
  SIZE = 224
  bytes_committed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  wqe_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 96]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 192]
  rdma: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 197]
  write_read: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 198]
  requestor: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 199]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
@c.record
class struct_mlx5_ifc_qp_events_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 0]
  type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  reserved_at_a8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  qpn_rqn_sqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
@c.record
class struct_mlx5_ifc_dct_events_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 0]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  dct_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
@c.record
class struct_mlx5_ifc_comp_event_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 0]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  cq_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
class _anonenum37(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_STATE_RST = _anonenum37.define('MLX5_QPC_STATE_RST', 0)
MLX5_QPC_STATE_INIT = _anonenum37.define('MLX5_QPC_STATE_INIT', 1)
MLX5_QPC_STATE_RTR = _anonenum37.define('MLX5_QPC_STATE_RTR', 2)
MLX5_QPC_STATE_RTS = _anonenum37.define('MLX5_QPC_STATE_RTS', 3)
MLX5_QPC_STATE_SQER = _anonenum37.define('MLX5_QPC_STATE_SQER', 4)
MLX5_QPC_STATE_ERR = _anonenum37.define('MLX5_QPC_STATE_ERR', 6)
MLX5_QPC_STATE_SQD = _anonenum37.define('MLX5_QPC_STATE_SQD', 7)
MLX5_QPC_STATE_SUSPENDED = _anonenum37.define('MLX5_QPC_STATE_SUSPENDED', 9)

class _anonenum38(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_ST_RC = _anonenum38.define('MLX5_QPC_ST_RC', 0)
MLX5_QPC_ST_UC = _anonenum38.define('MLX5_QPC_ST_UC', 1)
MLX5_QPC_ST_UD = _anonenum38.define('MLX5_QPC_ST_UD', 2)
MLX5_QPC_ST_XRC = _anonenum38.define('MLX5_QPC_ST_XRC', 3)
MLX5_QPC_ST_DCI = _anonenum38.define('MLX5_QPC_ST_DCI', 5)
MLX5_QPC_ST_QP0 = _anonenum38.define('MLX5_QPC_ST_QP0', 7)
MLX5_QPC_ST_QP1 = _anonenum38.define('MLX5_QPC_ST_QP1', 8)
MLX5_QPC_ST_RAW_DATAGRAM = _anonenum38.define('MLX5_QPC_ST_RAW_DATAGRAM', 9)
MLX5_QPC_ST_REG_UMR = _anonenum38.define('MLX5_QPC_ST_REG_UMR', 12)

class _anonenum39(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_PM_STATE_ARMED = _anonenum39.define('MLX5_QPC_PM_STATE_ARMED', 0)
MLX5_QPC_PM_STATE_REARM = _anonenum39.define('MLX5_QPC_PM_STATE_REARM', 1)
MLX5_QPC_PM_STATE_RESERVED = _anonenum39.define('MLX5_QPC_PM_STATE_RESERVED', 2)
MLX5_QPC_PM_STATE_MIGRATED = _anonenum39.define('MLX5_QPC_PM_STATE_MIGRATED', 3)

class _anonenum40(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_OFFLOAD_TYPE_RNDV = _anonenum40.define('MLX5_QPC_OFFLOAD_TYPE_RNDV', 1)

class _anonenum41(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_END_PADDING_MODE_SCATTER_AS_IS = _anonenum41.define('MLX5_QPC_END_PADDING_MODE_SCATTER_AS_IS', 0)
MLX5_QPC_END_PADDING_MODE_PAD_TO_CACHE_LINE_ALIGNMENT = _anonenum41.define('MLX5_QPC_END_PADDING_MODE_PAD_TO_CACHE_LINE_ALIGNMENT', 1)

class _anonenum42(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_MTU_256_BYTES = _anonenum42.define('MLX5_QPC_MTU_256_BYTES', 1)
MLX5_QPC_MTU_512_BYTES = _anonenum42.define('MLX5_QPC_MTU_512_BYTES', 2)
MLX5_QPC_MTU_1K_BYTES = _anonenum42.define('MLX5_QPC_MTU_1K_BYTES', 3)
MLX5_QPC_MTU_2K_BYTES = _anonenum42.define('MLX5_QPC_MTU_2K_BYTES', 4)
MLX5_QPC_MTU_4K_BYTES = _anonenum42.define('MLX5_QPC_MTU_4K_BYTES', 5)
MLX5_QPC_MTU_RAW_ETHERNET_QP = _anonenum42.define('MLX5_QPC_MTU_RAW_ETHERNET_QP', 7)

class _anonenum43(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_ATOMIC_MODE_IB_SPEC = _anonenum43.define('MLX5_QPC_ATOMIC_MODE_IB_SPEC', 1)
MLX5_QPC_ATOMIC_MODE_ONLY_8B = _anonenum43.define('MLX5_QPC_ATOMIC_MODE_ONLY_8B', 2)
MLX5_QPC_ATOMIC_MODE_UP_TO_8B = _anonenum43.define('MLX5_QPC_ATOMIC_MODE_UP_TO_8B', 3)
MLX5_QPC_ATOMIC_MODE_UP_TO_16B = _anonenum43.define('MLX5_QPC_ATOMIC_MODE_UP_TO_16B', 4)
MLX5_QPC_ATOMIC_MODE_UP_TO_32B = _anonenum43.define('MLX5_QPC_ATOMIC_MODE_UP_TO_32B', 5)
MLX5_QPC_ATOMIC_MODE_UP_TO_64B = _anonenum43.define('MLX5_QPC_ATOMIC_MODE_UP_TO_64B', 6)
MLX5_QPC_ATOMIC_MODE_UP_TO_128B = _anonenum43.define('MLX5_QPC_ATOMIC_MODE_UP_TO_128B', 7)
MLX5_QPC_ATOMIC_MODE_UP_TO_256B = _anonenum43.define('MLX5_QPC_ATOMIC_MODE_UP_TO_256B', 8)

class _anonenum44(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_CS_REQ_DISABLE = _anonenum44.define('MLX5_QPC_CS_REQ_DISABLE', 0)
MLX5_QPC_CS_REQ_UP_TO_32B = _anonenum44.define('MLX5_QPC_CS_REQ_UP_TO_32B', 17)
MLX5_QPC_CS_REQ_UP_TO_64B = _anonenum44.define('MLX5_QPC_CS_REQ_UP_TO_64B', 34)

class _anonenum45(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QPC_CS_RES_DISABLE = _anonenum45.define('MLX5_QPC_CS_RES_DISABLE', 0)
MLX5_QPC_CS_RES_UP_TO_32B = _anonenum45.define('MLX5_QPC_CS_RES_UP_TO_32B', 1)
MLX5_QPC_CS_RES_UP_TO_64B = _anonenum45.define('MLX5_QPC_CS_RES_UP_TO_64B', 2)

class _anonenum46(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_TIMESTAMP_FORMAT_FREE_RUNNING = _anonenum46.define('MLX5_TIMESTAMP_FORMAT_FREE_RUNNING', 0)
MLX5_TIMESTAMP_FORMAT_DEFAULT = _anonenum46.define('MLX5_TIMESTAMP_FORMAT_DEFAULT', 1)
MLX5_TIMESTAMP_FORMAT_REAL_TIME = _anonenum46.define('MLX5_TIMESTAMP_FORMAT_REAL_TIME', 2)

@c.record
class struct_mlx5_ifc_qpc_bits(c.Struct):
  SIZE = 1856
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  lag_tx_port_affinity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  st: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 16]
  isolate_vl_tc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 18]
  pm_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 19]
  reserved_at_15: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 21]
  req_e2e_credit_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 22]
  offload_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 24]
  end_padding_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 28]
  reserved_at_1e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 30]
  wq_signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  block_lb_mc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  atomic_like_write_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  latency_sensitive: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 35]
  reserved_at_24: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 36]
  drain_sigerr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 37]
  reserved_at_26: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 38]
  dp_ordering_force: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 39]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  mtu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 64]
  log_msg_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 67]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 72]
  log_rq_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 73]
  log_rq_stride: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 77]
  no_sq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 80]
  log_sq_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 81]
  reserved_at_55: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 85]
  retry_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 86]
  ts_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 88]
  reserved_at_5a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 90]
  rlky: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 91]
  ulp_stateless_offload_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  counter_set_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  uar_page: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  user_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 160]
  log_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 163]
  remote_qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  primary_address_path: Annotated[struct_mlx5_ifc_ads_bits, 192]
  secondary_address_path: Annotated[struct_mlx5_ifc_ads_bits, 544]
  log_ack_req_freq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 896]
  reserved_at_384: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 900]
  log_sra_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 904]
  reserved_at_38b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 907]
  retry_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 909]
  rnr_retry: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 912]
  reserved_at_393: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 915]
  fre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 916]
  cur_rnr_retry: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 917]
  cur_retry_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 920]
  reserved_at_39b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 923]
  reserved_at_3a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  reserved_at_3c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 960]
  next_send_psn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 968]
  reserved_at_3e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 992]
  log_num_dci_stream_channels: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 995]
  cqn_snd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1000]
  reserved_at_400: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1024]
  log_num_dci_errored_streams: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1027]
  deth_sqpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1032]
  reserved_at_420: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1056]
  reserved_at_440: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1088]
  last_acked_psn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1096]
  reserved_at_460: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1120]
  ssn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1128]
  reserved_at_480: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1152]
  log_rra_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1160]
  reserved_at_48b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1163]
  atomic_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1164]
  rre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1168]
  rwe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1169]
  rae: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1170]
  reserved_at_493: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1171]
  page_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 1172]
  reserved_at_49a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1178]
  dp_ordering_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1180]
  cd_slave_receive: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1181]
  cd_slave_send: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1182]
  cd_master: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1183]
  reserved_at_4a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1184]
  min_rnr_nak: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1187]
  next_rcv_psn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1192]
  reserved_at_4c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1216]
  xrcd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1224]
  reserved_at_4e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1248]
  cqn_rcv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1256]
  dbr_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1280]
  q_key: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1344]
  reserved_at_560: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 1376]
  rq_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1381]
  srqn_rmpn_xrqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1384]
  reserved_at_580: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1408]
  rmsn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 1416]
  hw_sq_wqebb_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1440]
  sw_sq_wqebb_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1456]
  hw_rq_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1472]
  sw_rq_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1504]
  reserved_at_600: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1536]
  reserved_at_620: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 1568]
  cgs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1583]
  cs_req: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1584]
  cs_res: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 1592]
  dc_access_key: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1600]
  reserved_at_680: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1664]
  dbr_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1667]
  reserved_at_684: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[188]], 1668]
@c.record
class struct_mlx5_ifc_roce_addr_layout_bits(c.Struct):
  SIZE = 256
  source_l3_address: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 0]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 128]
  vlan_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 131]
  vlan_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 132]
  source_mac_47_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  source_mac_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 192]
  roce_l3_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 212]
  roce_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 216]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_crypto_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 0]
  synchronize_dek: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  int_kek_manual: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  int_kek_auto: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  reserved_at_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 6]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 32]
  log_dek_max_alloc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 35]
  reserved_at_28: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 40]
  log_max_num_deks: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 43]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 96]
  log_dek_granularity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 99]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 104]
  log_max_num_int_kek: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 107]
  sw_wrapped_dek: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1920]], 128]
@c.record
class struct_mlx5_ifc_shampo_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 0]
  shampo_log_max_reservation_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 3]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 8]
  shampo_log_min_reservation_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 11]
  shampo_min_mss_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  shampo_header_split: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  shampo_header_split_data_merge: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  shampo_log_max_headers_entry_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 35]
  reserved_at_28: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1984]], 64]
@c.record
class union_mlx5_ifc_hca_cap_union_bits(c.Struct):
  SIZE = 32768
  cmd_hca_cap: Annotated[struct_mlx5_ifc_cmd_hca_cap_bits, 0]
  cmd_hca_cap_2: Annotated[struct_mlx5_ifc_cmd_hca_cap_2_bits, 0]
  odp_cap: Annotated[struct_mlx5_ifc_odp_cap_bits, 0]
  atomic_caps: Annotated[struct_mlx5_ifc_atomic_caps_bits, 0]
  roce_cap: Annotated[struct_mlx5_ifc_roce_cap_bits, 0]
  per_protocol_networking_offload_caps: Annotated[struct_mlx5_ifc_per_protocol_networking_offload_caps_bits, 0]
  flow_table_nic_cap: Annotated[struct_mlx5_ifc_flow_table_nic_cap_bits, 0]
  flow_table_eswitch_cap: Annotated[struct_mlx5_ifc_flow_table_eswitch_cap_bits, 0]
  wqe_based_flow_table_cap: Annotated[struct_mlx5_ifc_wqe_based_flow_table_cap_bits, 0]
  esw_cap: Annotated[struct_mlx5_ifc_esw_cap_bits, 0]
  e_switch_cap: Annotated[struct_mlx5_ifc_e_switch_cap_bits, 0]
  port_selection_cap: Annotated[struct_mlx5_ifc_port_selection_cap_bits, 0]
  qos_cap: Annotated[struct_mlx5_ifc_qos_cap_bits, 0]
  debug_cap: Annotated[struct_mlx5_ifc_debug_cap_bits, 0]
  fpga_cap: Annotated[struct_mlx5_ifc_fpga_cap_bits, 0]
  tls_cap: Annotated[struct_mlx5_ifc_tls_cap_bits, 0]
  device_mem_cap: Annotated[struct_mlx5_ifc_device_mem_cap_bits, 0]
  virtio_emulation_cap: Annotated[struct_mlx5_ifc_virtio_emulation_cap_bits, 0]
  macsec_cap: Annotated[struct_mlx5_ifc_macsec_cap_bits, 0]
  crypto_cap: Annotated[struct_mlx5_ifc_crypto_cap_bits, 0]
  ipsec_cap: Annotated[struct_mlx5_ifc_ipsec_cap_bits, 0]
  psp_cap: Annotated[struct_mlx5_ifc_psp_cap_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32768]], 0]
@c.record
class struct_mlx5_ifc_fpga_cap_bits(c.Struct):
  SIZE = 2048
  fpga_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  fpga_device: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  register_file_ver: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  fpga_ctrl_modify: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 65]
  access_reg_query_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 70]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 72]
  access_reg_modify_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 78]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  image_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  image_date: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  image_time: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  shell_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 256]
  shell_caps: Annotated[struct_mlx5_ifc_fpga_shell_caps_bits, 384]
  reserved_at_380: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 896]
  ieee_vendor_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 904]
  sandbox_product_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 928]
  sandbox_product_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 944]
  sandbox_basic_caps: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  reserved_at_3e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 992]
  sandbox_extended_caps_len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 1008]
  sandbox_extended_caps_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1024]
  fpga_ddr_start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1088]
  fpga_cr_space_start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1152]
  fpga_ddr_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1216]
  fpga_cr_space_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1248]
  reserved_at_500: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[768]], 1280]
@c.record
class struct_mlx5_ifc_fpga_shell_caps_bits(c.Struct):
  SIZE = 512
  max_num_qps: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  total_rcv_credits: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 32]
  qp_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 46]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 48]
  rae: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 53]
  rwe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 54]
  rre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 55]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 56]
  dc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 60]
  ud: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 61]
  uc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 62]
  rc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 64]
  log_ddr_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 90]
  max_fpga_qp_msg_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
class _anonenum47(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FLOW_CONTEXT_ACTION_ALLOW = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_ALLOW', 1)
MLX5_FLOW_CONTEXT_ACTION_DROP = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_DROP', 2)
MLX5_FLOW_CONTEXT_ACTION_FWD_DEST = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_FWD_DEST', 4)
MLX5_FLOW_CONTEXT_ACTION_COUNT = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_COUNT', 8)
MLX5_FLOW_CONTEXT_ACTION_PACKET_REFORMAT = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_PACKET_REFORMAT', 16)
MLX5_FLOW_CONTEXT_ACTION_DECAP = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_DECAP', 32)
MLX5_FLOW_CONTEXT_ACTION_MOD_HDR = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_MOD_HDR', 64)
MLX5_FLOW_CONTEXT_ACTION_VLAN_POP = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_VLAN_POP', 128)
MLX5_FLOW_CONTEXT_ACTION_VLAN_PUSH = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_VLAN_PUSH', 256)
MLX5_FLOW_CONTEXT_ACTION_VLAN_POP_2 = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_VLAN_POP_2', 1024)
MLX5_FLOW_CONTEXT_ACTION_VLAN_PUSH_2 = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_VLAN_PUSH_2', 2048)
MLX5_FLOW_CONTEXT_ACTION_CRYPTO_DECRYPT = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_CRYPTO_DECRYPT', 4096)
MLX5_FLOW_CONTEXT_ACTION_CRYPTO_ENCRYPT = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_CRYPTO_ENCRYPT', 8192)
MLX5_FLOW_CONTEXT_ACTION_EXECUTE_ASO = _anonenum47.define('MLX5_FLOW_CONTEXT_ACTION_EXECUTE_ASO', 16384)

class _anonenum48(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FLOW_CONTEXT_FLOW_SOURCE_ANY_VPORT = _anonenum48.define('MLX5_FLOW_CONTEXT_FLOW_SOURCE_ANY_VPORT', 0)
MLX5_FLOW_CONTEXT_FLOW_SOURCE_UPLINK = _anonenum48.define('MLX5_FLOW_CONTEXT_FLOW_SOURCE_UPLINK', 1)
MLX5_FLOW_CONTEXT_FLOW_SOURCE_LOCAL_VPORT = _anonenum48.define('MLX5_FLOW_CONTEXT_FLOW_SOURCE_LOCAL_VPORT', 2)

class _anonenum49(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_IPSEC = _anonenum49.define('MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_IPSEC', 0)
MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_MACSEC = _anonenum49.define('MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_MACSEC', 1)
MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_PSP = _anonenum49.define('MLX5_FLOW_CONTEXT_ENCRYPT_DECRYPT_TYPE_PSP', 2)

@c.record
class struct_mlx5_ifc_vlan_bits(c.Struct):
  SIZE = 32
  ethtype: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 16]
  cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 19]
  vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 20]
class _anonenum50(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FLOW_METER_COLOR_RED = _anonenum50.define('MLX5_FLOW_METER_COLOR_RED', 0)
MLX5_FLOW_METER_COLOR_YELLOW = _anonenum50.define('MLX5_FLOW_METER_COLOR_YELLOW', 1)
MLX5_FLOW_METER_COLOR_GREEN = _anonenum50.define('MLX5_FLOW_METER_COLOR_GREEN', 2)
MLX5_FLOW_METER_COLOR_UNDEFINED = _anonenum50.define('MLX5_FLOW_METER_COLOR_UNDEFINED', 3)

class _anonenum51(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_EXE_ASO_FLOW_METER = _anonenum51.define('MLX5_EXE_ASO_FLOW_METER', 2)

@c.record
class struct_mlx5_ifc_exe_aso_ctrl_flow_meter_bits(c.Struct):
  SIZE = 32
  return_reg_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  aso_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 8]
  action: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  init_color: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 29]
  meter_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
@c.record
class union_mlx5_ifc_exe_aso_ctrl(c.Struct):
  SIZE = 32
  exe_aso_ctrl_flow_meter: Annotated[struct_mlx5_ifc_exe_aso_ctrl_flow_meter_bits, 0]
@c.record
class struct_mlx5_ifc_execute_aso_bits(c.Struct):
  SIZE = 64
  valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 1]
  aso_object_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  exe_aso_ctrl: Annotated[union_mlx5_ifc_exe_aso_ctrl, 32]
@c.record
class struct_mlx5_ifc_flow_context_bits(c.Struct):
  SIZE = 6144
  push_vlan: Annotated[struct_mlx5_ifc_vlan_bits, 0]
  group_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  flow_tag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  action: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  extended_destination: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 128]
  uplink_hairpin_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 129]
  flow_source: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 130]
  encrypt_decrypt_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 132]
  destination_list_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  flow_counter_list_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  packet_reformat_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  modify_header_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  push_vlan_2: Annotated[struct_mlx5_ifc_vlan_bits, 256]
  encrypt_decrypt_obj_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 320]
  match_value: Annotated[struct_mlx5_ifc_fte_match_param_bits, 512]
  execute_aso: Annotated[c.Array[struct_mlx5_ifc_execute_aso_bits, Literal[4]], 4608]
  reserved_at_1300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1280]], 4864]
  destination: Annotated[c.Array[union_mlx5_ifc_dest_format_flow_counter_list_auto_bits, Literal[0]], 6144]
class _anonenum52(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_XRC_SRQC_STATE_GOOD = _anonenum52.define('MLX5_XRC_SRQC_STATE_GOOD', 0)
MLX5_XRC_SRQC_STATE_ERROR = _anonenum52.define('MLX5_XRC_SRQC_STATE_ERROR', 1)

@c.record
class struct_mlx5_ifc_xrc_srqc_bits(c.Struct):
  SIZE = 512
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  log_xrc_srq_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  wq_signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  cont_srq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  rlky: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 35]
  basic_cyclic_rcv_wqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 36]
  log_rq_stride: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 37]
  xrcd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  page_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 64]
  reserved_at_46: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 70]
  dbr_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 71]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  user_index_equal_xrc_srqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 128]
  reserved_at_81: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 129]
  log_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 130]
  user_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  lwm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  wqe_cnt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 256]
  db_record_addr_h: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  db_record_addr_l: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 352]
  reserved_at_17e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 382]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 384]
@c.record
class struct_mlx5_ifc_vnic_diagnostic_statistics_bits(c.Struct):
  SIZE = 4096
  counter_error_queues: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  total_error_queues: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  send_queue_priority_update_flow: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  nic_receive_steering_discard: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  receive_discard_vport_down: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  transmit_discard_vport_down: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 256]
  async_eq_overrun: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  comp_eq_overrun: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  invalid_command: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  quota_exceeded_command: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  internal_rq_out_of_buffer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  cq_overrun: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  eth_wqe_too_small: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  reserved_at_220: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 576]
  generated_pkt_steering_fail: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 768]
  handled_pkt_steering_fail: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 832]
  bar_uar_access: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 896]
  odp_local_triggered_page_fault: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  odp_remote_triggered_page_fault: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  reserved_at_3c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3104]], 992]
@c.record
class struct_mlx5_ifc_traffic_counter_bits(c.Struct):
  SIZE = 128
  packets: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  octets: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_tisc_bits(c.Struct):
  SIZE = 1280
  strict_lag_tx_port_affinity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  tls_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 2]
  lag_tx_port_affinity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 8]
  prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 32]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 288]
  transport_domain: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 296]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 320]
  underlay_qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 328]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 352]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 360]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[896]], 384]
class _anonenum53(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_TIRC_DISP_TYPE_DIRECT = _anonenum53.define('MLX5_TIRC_DISP_TYPE_DIRECT', 0)
MLX5_TIRC_DISP_TYPE_INDIRECT = _anonenum53.define('MLX5_TIRC_DISP_TYPE_INDIRECT', 1)

class _anonenum54(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_TIRC_PACKET_MERGE_MASK_IPV4_LRO = _anonenum54.define('MLX5_TIRC_PACKET_MERGE_MASK_IPV4_LRO', 0)
MLX5_TIRC_PACKET_MERGE_MASK_IPV6_LRO = _anonenum54.define('MLX5_TIRC_PACKET_MERGE_MASK_IPV6_LRO', 1)

class _anonenum55(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RX_HASH_FN_NONE = _anonenum55.define('MLX5_RX_HASH_FN_NONE', 0)
MLX5_RX_HASH_FN_INVERTED_XOR8 = _anonenum55.define('MLX5_RX_HASH_FN_INVERTED_XOR8', 1)
MLX5_RX_HASH_FN_TOEPLITZ = _anonenum55.define('MLX5_RX_HASH_FN_TOEPLITZ', 2)

class _anonenum56(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_TIRC_SELF_LB_BLOCK_BLOCK_UNICAST = _anonenum56.define('MLX5_TIRC_SELF_LB_BLOCK_BLOCK_UNICAST', 1)
MLX5_TIRC_SELF_LB_BLOCK_BLOCK_MULTICAST = _anonenum56.define('MLX5_TIRC_SELF_LB_BLOCK_BLOCK_MULTICAST', 2)

@c.record
class struct_mlx5_ifc_tirc_bits(c.Struct):
  SIZE = 1920
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  disp_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 32]
  tls_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 36]
  reserved_at_25: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[27]], 37]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 128]
  lro_timeout_period_usecs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 132]
  packet_merge_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 148]
  lro_max_ip_payload_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 152]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 160]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 224]
  inline_rqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 232]
  rx_hash_symmetric: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 256]
  reserved_at_101: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 257]
  tunneled_offload_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 258]
  reserved_at_103: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 259]
  indirect_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 264]
  rx_hash_fn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 288]
  reserved_at_124: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 292]
  self_lb_block: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 294]
  transport_domain: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 296]
  rx_hash_toeplitz_key: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[10]], 320]
  rx_hash_field_selector_outer: Annotated[struct_mlx5_ifc_rx_hash_field_select_bits, 640]
  rx_hash_field_selector_inner: Annotated[struct_mlx5_ifc_rx_hash_field_select_bits, 672]
  reserved_at_2c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1216]], 704]
class _anonenum57(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_SRQC_STATE_GOOD = _anonenum57.define('MLX5_SRQC_STATE_GOOD', 0)
MLX5_SRQC_STATE_ERROR = _anonenum57.define('MLX5_SRQC_STATE_ERROR', 1)

@c.record
class struct_mlx5_ifc_srqc_bits(c.Struct):
  SIZE = 512
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  log_srq_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  wq_signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  cont_srq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  rlky: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 35]
  reserved_at_24: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 36]
  log_rq_stride: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 37]
  xrcd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  page_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 64]
  reserved_at_46: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 70]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 128]
  log_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 130]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  lwm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  wqe_cnt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 256]
  dbr_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 384]
class _anonenum58(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_SQC_STATE_RST = _anonenum58.define('MLX5_SQC_STATE_RST', 0)
MLX5_SQC_STATE_RDY = _anonenum58.define('MLX5_SQC_STATE_RDY', 1)
MLX5_SQC_STATE_ERR = _anonenum58.define('MLX5_SQC_STATE_ERR', 3)

@c.record
class struct_mlx5_ifc_sqc_bits(c.Struct):
  SIZE = 1920
  rlky: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  cd_master: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  fre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  flush_in_error_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  allow_multi_pkt_send_wqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  min_wqe_inline_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 5]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 8]
  reg_umr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 12]
  allow_swp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  hairpin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  non_wire: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 15]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 16]
  ts_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 26]
  reserved_at_1c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  user_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  hairpin_peer_rq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  hairpin_peer_vhca: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  ts_cqe_to_dest_cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  packet_pacing_rate_limit_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  tis_lst_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 256]
  qos_queue_group_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 272]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 288]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 352]
  tis_num_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 360]
  wq: Annotated[struct_mlx5_ifc_wq_bits, 384]
class _anonenum59(Annotated[int, ctypes.c_uint32], c.Enum): pass
SCHEDULING_CONTEXT_ELEMENT_TYPE_TSAR = _anonenum59.define('SCHEDULING_CONTEXT_ELEMENT_TYPE_TSAR', 0)
SCHEDULING_CONTEXT_ELEMENT_TYPE_VPORT = _anonenum59.define('SCHEDULING_CONTEXT_ELEMENT_TYPE_VPORT', 1)
SCHEDULING_CONTEXT_ELEMENT_TYPE_VPORT_TC = _anonenum59.define('SCHEDULING_CONTEXT_ELEMENT_TYPE_VPORT_TC', 2)
SCHEDULING_CONTEXT_ELEMENT_TYPE_PARA_VPORT_TC = _anonenum59.define('SCHEDULING_CONTEXT_ELEMENT_TYPE_PARA_VPORT_TC', 3)
SCHEDULING_CONTEXT_ELEMENT_TYPE_QUEUE_GROUP = _anonenum59.define('SCHEDULING_CONTEXT_ELEMENT_TYPE_QUEUE_GROUP', 4)
SCHEDULING_CONTEXT_ELEMENT_TYPE_RATE_LIMIT = _anonenum59.define('SCHEDULING_CONTEXT_ELEMENT_TYPE_RATE_LIMIT', 5)

class _anonenum60(Annotated[int, ctypes.c_uint32], c.Enum): pass
ELEMENT_TYPE_CAP_MASK_TSAR = _anonenum60.define('ELEMENT_TYPE_CAP_MASK_TSAR', 1)
ELEMENT_TYPE_CAP_MASK_VPORT = _anonenum60.define('ELEMENT_TYPE_CAP_MASK_VPORT', 2)
ELEMENT_TYPE_CAP_MASK_VPORT_TC = _anonenum60.define('ELEMENT_TYPE_CAP_MASK_VPORT_TC', 4)
ELEMENT_TYPE_CAP_MASK_PARA_VPORT_TC = _anonenum60.define('ELEMENT_TYPE_CAP_MASK_PARA_VPORT_TC', 8)
ELEMENT_TYPE_CAP_MASK_QUEUE_GROUP = _anonenum60.define('ELEMENT_TYPE_CAP_MASK_QUEUE_GROUP', 16)
ELEMENT_TYPE_CAP_MASK_RATE_LIMIT = _anonenum60.define('ELEMENT_TYPE_CAP_MASK_RATE_LIMIT', 32)

class _anonenum61(Annotated[int, ctypes.c_uint32], c.Enum): pass
TSAR_ELEMENT_TSAR_TYPE_DWRR = _anonenum61.define('TSAR_ELEMENT_TSAR_TYPE_DWRR', 0)
TSAR_ELEMENT_TSAR_TYPE_ROUND_ROBIN = _anonenum61.define('TSAR_ELEMENT_TSAR_TYPE_ROUND_ROBIN', 1)
TSAR_ELEMENT_TSAR_TYPE_ETS = _anonenum61.define('TSAR_ELEMENT_TSAR_TYPE_ETS', 2)
TSAR_ELEMENT_TSAR_TYPE_TC_ARB = _anonenum61.define('TSAR_ELEMENT_TSAR_TYPE_TC_ARB', 3)

class _anonenum62(Annotated[int, ctypes.c_uint32], c.Enum): pass
TSAR_TYPE_CAP_MASK_DWRR = _anonenum62.define('TSAR_TYPE_CAP_MASK_DWRR', 1)
TSAR_TYPE_CAP_MASK_ROUND_ROBIN = _anonenum62.define('TSAR_TYPE_CAP_MASK_ROUND_ROBIN', 2)
TSAR_TYPE_CAP_MASK_ETS = _anonenum62.define('TSAR_TYPE_CAP_MASK_ETS', 4)
TSAR_TYPE_CAP_MASK_TC_ARB = _anonenum62.define('TSAR_TYPE_CAP_MASK_TC_ARB', 8)

@c.record
class struct_mlx5_ifc_tsar_element_bits(c.Struct):
  SIZE = 32
  traffic_class: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  tsar_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
@c.record
class struct_mlx5_ifc_vport_element_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  eswitch_owner_vhca_id_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 5]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
@c.record
class struct_mlx5_ifc_vport_tc_element_bits(c.Struct):
  SIZE = 32
  traffic_class: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  eswitch_owner_vhca_id_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 5]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
@c.record
class union_mlx5_ifc_element_attributes_bits(c.Struct):
  SIZE = 32
  tsar: Annotated[struct_mlx5_ifc_tsar_element_bits, 0]
  vport: Annotated[struct_mlx5_ifc_vport_element_bits, 0]
  vport_tc: Annotated[struct_mlx5_ifc_vport_tc_element_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_scheduling_context_bits(c.Struct):
  SIZE = 512
  element_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  element_attributes: Annotated[union_mlx5_ifc_element_attributes_bits, 32]
  parent_element_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 96]
  bw_share: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  max_average_bw: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  max_bw_obj_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 256]
@c.record
class struct_mlx5_ifc_rqtc_bits(c.Struct):
  SIZE = 1920
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 0]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 160]
  list_q_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 165]
  reserved_at_a8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 168]
  rqt_max_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  rq_vhca_id_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 192]
  reserved_at_c1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 193]
  rqt_actual_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 208]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1696]], 224]
class _anonenum63(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_INLINE = _anonenum63.define('MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_INLINE', 0)
MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_RMP = _anonenum63.define('MLX5_RQC_MEM_RQ_TYPE_MEMORY_RQ_RMP', 1)

class _anonenum64(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RQC_STATE_RST = _anonenum64.define('MLX5_RQC_STATE_RST', 0)
MLX5_RQC_STATE_RDY = _anonenum64.define('MLX5_RQC_STATE_RDY', 1)
MLX5_RQC_STATE_ERR = _anonenum64.define('MLX5_RQC_STATE_ERR', 3)

class _anonenum65(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_BYTE = _anonenum65.define('MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_BYTE', 0)
MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_STRIDE = _anonenum65.define('MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_STRIDE', 1)
MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_PAGE = _anonenum65.define('MLX5_RQC_SHAMPO_NO_MATCH_ALIGNMENT_GRANULARITY_PAGE', 2)

class _anonenum66(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_NO_MATCH = _anonenum66.define('MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_NO_MATCH', 0)
MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_EXTENDED = _anonenum66.define('MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_EXTENDED', 1)
MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_FIVE_TUPLE = _anonenum66.define('MLX5_RQC_SHAMPO_MATCH_CRITERIA_TYPE_FIVE_TUPLE', 2)

@c.record
class struct_mlx5_ifc_rqc_bits(c.Struct):
  SIZE = 1920
  rlky: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  delay_drop_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  scatter_fcs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  vsd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  mem_rq_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 8]
  reserved_at_c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 12]
  flush_in_error_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  hairpin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  reserved_at_f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 15]
  ts_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 26]
  reserved_at_1c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  user_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  counter_set_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  rmpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  hairpin_peer_sq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 192]
  hairpin_peer_vhca: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 208]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[70]], 224]
  shampo_no_match_alignment_granularity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 294]
  reserved_at_128: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 296]
  shampo_match_criteria_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 302]
  reservation_timeout: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 304]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
  wq: Annotated[struct_mlx5_ifc_wq_bits, 384]
class _anonenum67(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RMPC_STATE_RDY = _anonenum67.define('MLX5_RMPC_STATE_RDY', 1)
MLX5_RMPC_STATE_ERR = _anonenum67.define('MLX5_RMPC_STATE_ERR', 3)

@c.record
class struct_mlx5_ifc_rmpc_bits(c.Struct):
  SIZE = 1920
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 8]
  reserved_at_c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 12]
  basic_cyclic_rcv_wqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 33]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 64]
  wq: Annotated[struct_mlx5_ifc_wq_bits, 384]
class _anonenum68(Annotated[int, ctypes.c_uint32], c.Enum): pass
VHCA_ID_TYPE_HW = _anonenum68.define('VHCA_ID_TYPE_HW', 0)
VHCA_ID_TYPE_SW = _anonenum68.define('VHCA_ID_TYPE_SW', 1)

@c.record
class struct_mlx5_ifc_nic_vport_context_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 0]
  min_wqe_inline_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 5]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[21]], 8]
  disable_mc_local_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  disable_uc_local_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 30]
  roce_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
  arm_change_event: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 33]
  event_on_mtu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 59]
  event_on_promisc_change: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 60]
  event_on_vlan_change: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 61]
  event_on_mc_address_change: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 62]
  event_on_uc_address_change: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
  vhca_id_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 65]
  affiliation_criteria: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  affiliated_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 96]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 256]
  sd_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 257]
  reserved_at_104: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 260]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 288]
  mtu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 304]
  system_image_guid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
  port_guid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 384]
  node_guid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
  reserved_at_200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 512]
  qkey_violation_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 832]
  reserved_at_350: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1072]], 848]
  promisc_uc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1920]
  promisc_mc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1921]
  promisc_all: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1922]
  reserved_at_783: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1923]
  allowed_list_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1925]
  reserved_at_788: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 1928]
  allowed_list_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 1940]
  permanent_address: Annotated[struct_mlx5_ifc_mac_address_layout_bits, 1952]
  reserved_at_7e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 2016]
  current_uc_mac_address: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2048]
class _anonenum69(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MKC_ACCESS_MODE_PA = _anonenum69.define('MLX5_MKC_ACCESS_MODE_PA', 0)
MLX5_MKC_ACCESS_MODE_MTT = _anonenum69.define('MLX5_MKC_ACCESS_MODE_MTT', 1)
MLX5_MKC_ACCESS_MODE_KLMS = _anonenum69.define('MLX5_MKC_ACCESS_MODE_KLMS', 2)
MLX5_MKC_ACCESS_MODE_KSM = _anonenum69.define('MLX5_MKC_ACCESS_MODE_KSM', 3)
MLX5_MKC_ACCESS_MODE_SW_ICM = _anonenum69.define('MLX5_MKC_ACCESS_MODE_SW_ICM', 4)
MLX5_MKC_ACCESS_MODE_MEMIC = _anonenum69.define('MLX5_MKC_ACCESS_MODE_MEMIC', 5)
MLX5_MKC_ACCESS_MODE_CROSSING = _anonenum69.define('MLX5_MKC_ACCESS_MODE_CROSSING', 6)

class _anonenum70(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MKC_PCIE_TPH_NO_STEERING_TAG_INDEX = _anonenum70.define('MLX5_MKC_PCIE_TPH_NO_STEERING_TAG_INDEX', 0)

@c.record
class struct_mlx5_ifc_mkc_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  free: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  access_mode_4_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 3]
  reserved_at_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 6]
  relaxed_ordering_write: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  reserved_at_e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  small_fence_on_rdma_read_response: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 15]
  umr_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 17]
  rw: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 18]
  rr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 19]
  lw: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 20]
  lr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 21]
  access_mode_1_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 22]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 24]
  ma_translation_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 26]
  reserved_at_1c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 32]
  mkey_7_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  length64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  bsf_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 97]
  sync_umr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 98]
  reserved_at_63: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 99]
  expected_sigerr_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 101]
  reserved_at_66: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 102]
  en_rinval: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 103]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  len: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  bsf_octword_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 288]
  crossing_target_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 384]
  reserved_at_190: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 400]
  translations_octword_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[25]], 448]
  relaxed_ordering_read: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 473]
  log_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 474]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 480]
  pcie_tph_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 485]
  pcie_tph_ph: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 486]
  pcie_tph_steering_tag_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 488]
  reserved_at_1f0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 496]
@c.record
class struct_mlx5_ifc_pkey_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  pkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
@c.record
class struct_mlx5_ifc_array128_auto_bits(c.Struct):
  SIZE = 128
  array128_auto: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 0]
@c.record
class struct_mlx5_ifc_hca_vport_context_bits(c.Struct):
  SIZE = 4096
  field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 32]
  sm_virt_aware: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 256]
  has_smi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 257]
  has_raw: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 258]
  grh_required: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 259]
  reserved_at_104: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 260]
  num_port_plane: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 264]
  port_physical_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 272]
  vport_state_policy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 276]
  port_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 280]
  vport_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 284]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  system_image_guid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
  port_guid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 384]
  node_guid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
  cap_mask1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  cap_mask1_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  cap_mask2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  cap_mask2_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 640]
  lid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 768]
  reserved_at_310: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 784]
  init_type_reply: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 788]
  lmc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 792]
  subnet_timeout: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 795]
  sm_lid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 800]
  sm_sl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 816]
  reserved_at_334: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 820]
  qkey_violation_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 832]
  pkey_violation_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 848]
  reserved_at_360: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3232]], 864]
@c.record
class struct_mlx5_ifc_esw_vport_context_bits(c.Struct):
  SIZE = 2048
  fdb_to_vport_reg_c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1]
  vport_svlan_strip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  vport_cvlan_strip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  vport_svlan_insert: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  vport_cvlan_insert: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 6]
  fdb_to_vport_reg_c_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  svlan_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  svlan_pcp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 65]
  svlan_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 68]
  cvlan_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 80]
  cvlan_pcp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 81]
  cvlan_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 84]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1824]], 96]
  sw_steering_vport_icm_address_rx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1920]
  sw_steering_vport_icm_address_tx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 1984]
class _anonenum71(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_EQC_STATUS_OK = _anonenum71.define('MLX5_EQC_STATUS_OK', 0)
MLX5_EQC_STATUS_EQ_WRITE_FAILURE = _anonenum71.define('MLX5_EQC_STATUS_EQ_WRITE_FAILURE', 10)

class _anonenum72(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_EQC_ST_ARMED = _anonenum72.define('MLX5_EQC_ST_ARMED', 9)
MLX5_EQC_ST_FIRED = _anonenum72.define('MLX5_EQC_ST_FIRED', 10)

@c.record
class struct_mlx5_ifc_eqc_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 4]
  ec: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  oi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  reserved_at_f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 15]
  st: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 64]
  page_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 84]
  reserved_at_5a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 90]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 96]
  log_eq_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 99]
  uar_page: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 160]
  intr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 180]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 192]
  log_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 195]
  reserved_at_c8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 224]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 320]
  consumer_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 328]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 352]
  producer_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 360]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 384]
class _anonenum73(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_DCTC_STATE_ACTIVE = _anonenum73.define('MLX5_DCTC_STATE_ACTIVE', 0)
MLX5_DCTC_STATE_DRAINING = _anonenum73.define('MLX5_DCTC_STATE_DRAINING', 1)
MLX5_DCTC_STATE_DRAINED = _anonenum73.define('MLX5_DCTC_STATE_DRAINED', 2)

class _anonenum74(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_DCTC_CS_RES_DISABLE = _anonenum74.define('MLX5_DCTC_CS_RES_DISABLE', 0)
MLX5_DCTC_CS_RES_NA = _anonenum74.define('MLX5_DCTC_CS_RES_NA', 1)
MLX5_DCTC_CS_RES_UP_TO_64B = _anonenum74.define('MLX5_DCTC_CS_RES_UP_TO_64B', 2)

class _anonenum75(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_DCTC_MTU_256_BYTES = _anonenum75.define('MLX5_DCTC_MTU_256_BYTES', 1)
MLX5_DCTC_MTU_512_BYTES = _anonenum75.define('MLX5_DCTC_MTU_512_BYTES', 2)
MLX5_DCTC_MTU_1K_BYTES = _anonenum75.define('MLX5_DCTC_MTU_1K_BYTES', 3)
MLX5_DCTC_MTU_2K_BYTES = _anonenum75.define('MLX5_DCTC_MTU_2K_BYTES', 4)
MLX5_DCTC_MTU_4K_BYTES = _anonenum75.define('MLX5_DCTC_MTU_4K_BYTES', 5)

@c.record
class struct_mlx5_ifc_dctc_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 32]
  dp_ordering_force: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 39]
  user_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  counter_set_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  atomic_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 104]
  rre: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 108]
  rwe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 109]
  rae: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 110]
  atomic_like_write_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 111]
  latency_sensitive: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 112]
  rlky: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 113]
  free_ar: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 114]
  reserved_at_73: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 115]
  dp_ordering_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 116]
  reserved_at_75: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 117]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  cs_res: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  reserved_at_90: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 144]
  min_rnr_nak: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 147]
  reserved_at_98: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 152]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  srqn_xrqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  tclass: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 224]
  reserved_at_e8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 232]
  flow_label: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 236]
  dc_access_key: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 256]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 320]
  mtu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 325]
  port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 328]
  pkey_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 336]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 352]
  my_addr_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 360]
  reserved_at_170: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 368]
  hop_limit: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 376]
  dc_access_key_violation_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  reserved_at_1a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 416]
  dei_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 436]
  eth_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 437]
  ecn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 440]
  dscp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 442]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
class _anonenum76(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CQC_STATUS_OK = _anonenum76.define('MLX5_CQC_STATUS_OK', 0)
MLX5_CQC_STATUS_CQ_OVERFLOW = _anonenum76.define('MLX5_CQC_STATUS_CQ_OVERFLOW', 9)
MLX5_CQC_STATUS_CQ_WRITE_FAIL = _anonenum76.define('MLX5_CQC_STATUS_CQ_WRITE_FAIL', 10)

class _anonenum77(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CQC_CQE_SZ_64_BYTES = _anonenum77.define('MLX5_CQC_CQE_SZ_64_BYTES', 0)
MLX5_CQC_CQE_SZ_128_BYTES = _anonenum77.define('MLX5_CQC_CQE_SZ_128_BYTES', 1)

class _anonenum78(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CQC_ST_SOLICITED_NOTIFICATION_REQUEST_ARMED = _anonenum78.define('MLX5_CQC_ST_SOLICITED_NOTIFICATION_REQUEST_ARMED', 6)
MLX5_CQC_ST_NOTIFICATION_REQUEST_ARMED = _anonenum78.define('MLX5_CQC_ST_NOTIFICATION_REQUEST_ARMED', 9)
MLX5_CQC_ST_FIRED = _anonenum78.define('MLX5_CQC_ST_FIRED', 10)

class enum_mlx5_cq_period_mode(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CQ_PERIOD_MODE_START_FROM_EQE = enum_mlx5_cq_period_mode.define('MLX5_CQ_PERIOD_MODE_START_FROM_EQE', 0)
MLX5_CQ_PERIOD_MODE_START_FROM_CQE = enum_mlx5_cq_period_mode.define('MLX5_CQ_PERIOD_MODE_START_FROM_CQE', 1)
MLX5_CQ_PERIOD_NUM_MODES = enum_mlx5_cq_period_mode.define('MLX5_CQ_PERIOD_NUM_MODES', 2)

@c.record
class struct_mlx5_ifc_cqc_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 4]
  dbr_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  apu_cq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  cqe_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 8]
  cc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 11]
  reserved_at_c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 12]
  scqe_break_moderation_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  oi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  cq_period_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 15]
  cqe_comp_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 17]
  mini_cqe_res_format: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 18]
  st: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 24]
  cqe_compression_layout: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 30]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 64]
  page_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 84]
  reserved_at_5a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 90]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 96]
  log_cq_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 99]
  uar_page: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 128]
  cq_period: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 132]
  cq_max_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  c_eqn_or_apu_element: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 192]
  log_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 195]
  reserved_at_c8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  last_notified_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 264]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 288]
  last_solicit_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 296]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 320]
  consumer_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 328]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 352]
  producer_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 360]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 384]
  dbr_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
@c.record
class union_mlx5_ifc_cong_control_roce_ecn_auto_bits(c.Struct):
  SIZE = 2048
  cong_control_802_1qau_rp: Annotated[struct_mlx5_ifc_cong_control_802_1qau_rp_bits, 0]
  cong_control_r_roce_ecn_rp: Annotated[struct_mlx5_ifc_cong_control_r_roce_ecn_rp_bits, 0]
  cong_control_r_roce_ecn_np: Annotated[struct_mlx5_ifc_cong_control_r_roce_ecn_np_bits, 0]
  cong_control_r_roce_general: Annotated[struct_mlx5_ifc_cong_control_r_roce_general_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2048]], 0]
@c.record
class struct_mlx5_ifc_query_adapter_param_block_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 0]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  ieee_vendor_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  vsd_vendor_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  vsd: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[208]], 256]
  vsd_contd_psid: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 1920]
class _anonenum79(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_XRQC_STATE_GOOD = _anonenum79.define('MLX5_XRQC_STATE_GOOD', 0)
MLX5_XRQC_STATE_ERROR = _anonenum79.define('MLX5_XRQC_STATE_ERROR', 1)

class _anonenum80(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_XRQC_TOPOLOGY_NO_SPECIAL_TOPOLOGY = _anonenum80.define('MLX5_XRQC_TOPOLOGY_NO_SPECIAL_TOPOLOGY', 0)
MLX5_XRQC_TOPOLOGY_TAG_MATCHING = _anonenum80.define('MLX5_XRQC_TOPOLOGY_TAG_MATCHING', 1)

class _anonenum81(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_XRQC_OFFLOAD_RNDV = _anonenum81.define('MLX5_XRQC_OFFLOAD_RNDV', 1)

@c.record
class struct_mlx5_ifc_tag_matching_topology_context_bits(c.Struct):
  SIZE = 128
  log_matching_list_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 4]
  append_next_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  sw_phase_cnt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  hw_phase_cnt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_xrqc_bits(c.Struct):
  SIZE = 2560
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  rlkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  reserved_at_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 5]
  topology: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 24]
  offload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  user_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 96]
  tag_matching_topology_context: Annotated[struct_mlx5_ifc_tag_matching_topology_context_bits, 256]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[640]], 384]
  wq: Annotated[struct_mlx5_ifc_wq_bits, 1024]
@c.record
class union_mlx5_ifc_modify_field_select_resize_field_select_auto_bits(c.Struct):
  SIZE = 32
  modify_field_select: Annotated[struct_mlx5_ifc_modify_field_select_bits, 0]
  resize_field_select: Annotated[struct_mlx5_ifc_resize_field_select_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
@c.record
class union_mlx5_ifc_field_select_802_1_r_roce_auto_bits(c.Struct):
  SIZE = 32
  field_select_802_1qau_rp: Annotated[struct_mlx5_ifc_field_select_802_1qau_rp_bits, 0]
  field_select_r_roce_rp: Annotated[struct_mlx5_ifc_field_select_r_roce_rp_bits, 0]
  field_select_r_roce_np: Annotated[struct_mlx5_ifc_field_select_r_roce_np_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_rs_histogram_cntrs_bits(c.Struct):
  SIZE = 1728
  hist: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[16]], 0]
  reserved_at_400: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[704]], 1024]
@c.record
class union_mlx5_ifc_eth_cntrs_grp_data_layout_auto_bits(c.Struct):
  SIZE = 1984
  eth_802_3_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits, 0]
  eth_2863_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits, 0]
  eth_2819_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits, 0]
  eth_3635_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits, 0]
  eth_extended_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits, 0]
  eth_per_prio_grp_data_layout: Annotated[struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits, 0]
  eth_per_tc_prio_grp_data_layout: Annotated[struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits, 0]
  eth_per_tc_congest_prio_grp_data_layout: Annotated[struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits, 0]
  ib_port_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits, 0]
  ib_ext_port_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_ib_ext_port_cntrs_grp_data_layout_bits, 0]
  phys_layer_cntrs: Annotated[struct_mlx5_ifc_phys_layer_cntrs_bits, 0]
  phys_layer_statistical_cntrs: Annotated[struct_mlx5_ifc_phys_layer_statistical_cntrs_bits, 0]
  phys_layer_recovery_cntrs: Annotated[struct_mlx5_ifc_phys_layer_recovery_cntrs_bits, 0]
  rs_histogram_cntrs: Annotated[struct_mlx5_ifc_rs_histogram_cntrs_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1984]], 0]
@c.record
class union_mlx5_ifc_pcie_cntrs_grp_data_layout_auto_bits(c.Struct):
  SIZE = 1984
  pcie_perf_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_pcie_perf_cntrs_grp_data_layout_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1984]], 0]
@c.record
class union_mlx5_ifc_event_auto_bits(c.Struct):
  SIZE = 224
  comp_event: Annotated[struct_mlx5_ifc_comp_event_bits, 0]
  dct_events: Annotated[struct_mlx5_ifc_dct_events_bits, 0]
  qp_events: Annotated[struct_mlx5_ifc_qp_events_bits, 0]
  wqe_associated_page_fault_event: Annotated[struct_mlx5_ifc_wqe_associated_page_fault_event_bits, 0]
  rdma_page_fault_event: Annotated[struct_mlx5_ifc_rdma_page_fault_event_bits, 0]
  cq_error: Annotated[struct_mlx5_ifc_cq_error_bits, 0]
  dropped_packet_logged: Annotated[struct_mlx5_ifc_dropped_packet_logged_bits, 0]
  port_state_change_event: Annotated[struct_mlx5_ifc_port_state_change_event_bits, 0]
  gpio_event: Annotated[struct_mlx5_ifc_gpio_event_bits, 0]
  db_bf_congestion_event: Annotated[struct_mlx5_ifc_db_bf_congestion_event_bits, 0]
  stall_vl_event: Annotated[struct_mlx5_ifc_stall_vl_event_bits, 0]
  cmd_inter_comp_event: Annotated[struct_mlx5_ifc_cmd_inter_comp_event_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 0]
@c.record
class struct_mlx5_ifc_health_buffer_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 0]
  assert_existptr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  assert_callra: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  time: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  fw_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  hw_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  rfr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 448]
  reserved_at_1c1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 449]
  valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 452]
  severity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 453]
  reserved_at_1c8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 456]
  irisc_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 480]
  synd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 488]
  ext_synd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 496]
@c.record
class struct_mlx5_ifc_register_loopback_control_bits(c.Struct):
  SIZE = 128
  no_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 1]
  port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
class _anonenum82(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_TEARDOWN_HCA_OUT_FORCE_STATE_SUCCESS = _anonenum82.define('MLX5_TEARDOWN_HCA_OUT_FORCE_STATE_SUCCESS', 0)
MLX5_TEARDOWN_HCA_OUT_FORCE_STATE_FAIL = _anonenum82.define('MLX5_TEARDOWN_HCA_OUT_FORCE_STATE_FAIL', 1)

@c.record
class struct_mlx5_ifc_teardown_hca_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[63]], 64]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 127]
class _anonenum83(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_TEARDOWN_HCA_IN_PROFILE_GRACEFUL_CLOSE = _anonenum83.define('MLX5_TEARDOWN_HCA_IN_PROFILE_GRACEFUL_CLOSE', 0)
MLX5_TEARDOWN_HCA_IN_PROFILE_FORCE_CLOSE = _anonenum83.define('MLX5_TEARDOWN_HCA_IN_PROFILE_FORCE_CLOSE', 1)
MLX5_TEARDOWN_HCA_IN_PROFILE_PREPARE_FAST_TEARDOWN = _anonenum83.define('MLX5_TEARDOWN_HCA_IN_PROFILE_PREPARE_FAST_TEARDOWN', 2)

@c.record
class struct_mlx5_ifc_teardown_hca_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  profile: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_sqerr2rts_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_sqerr2rts_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 2048]
@c.record
class struct_mlx5_ifc_sqd2rts_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_sqd2rts_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 2048]
@c.record
class struct_mlx5_ifc_set_roce_address_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_roce_address_in_bits(c.Struct):
  SIZE = 384
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  roce_address_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 80]
  vhca_port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  roce_address: Annotated[struct_mlx5_ifc_roce_addr_layout_bits, 128]
@c.record
class struct_mlx5_ifc_set_mad_demux_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class _anonenum84(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_SET_MAD_DEMUX_IN_DEMUX_MODE_PASS_ALL = _anonenum84.define('MLX5_SET_MAD_DEMUX_IN_DEMUX_MODE_PASS_ALL', 0)
MLX5_SET_MAD_DEMUX_IN_DEMUX_MODE_SELECTIVE = _anonenum84.define('MLX5_SET_MAD_DEMUX_IN_DEMUX_MODE_SELECTIVE', 2)

@c.record
class struct_mlx5_ifc_set_mad_demux_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 96]
  demux_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 102]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
@c.record
class struct_mlx5_ifc_set_l2_table_entry_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_l2_table_entry_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 64]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  silent_mode_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 240]
  silent_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 241]
  reserved_at_f2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 242]
  vlan_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 243]
  vlan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 244]
  mac_address: Annotated[struct_mlx5_ifc_mac_address_layout_bits, 256]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 320]
@c.record
class struct_mlx5_ifc_set_issi_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_issi_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  current_issi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_set_hca_cap_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_hca_cap_in_bits(c.Struct):
  SIZE = 32896
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  ec_vf_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 66]
  function_id_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 67]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 68]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  capability: Annotated[union_mlx5_ifc_hca_cap_union_bits, 128]
class _anonenum85(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_SET_FTE_MODIFY_ENABLE_MASK_ACTION = _anonenum85.define('MLX5_SET_FTE_MODIFY_ENABLE_MASK_ACTION', 0)
MLX5_SET_FTE_MODIFY_ENABLE_MASK_FLOW_TAG = _anonenum85.define('MLX5_SET_FTE_MODIFY_ENABLE_MASK_FLOW_TAG', 1)
MLX5_SET_FTE_MODIFY_ENABLE_MASK_DESTINATION_LIST = _anonenum85.define('MLX5_SET_FTE_MODIFY_ENABLE_MASK_DESTINATION_LIST', 2)
MLX5_SET_FTE_MODIFY_ENABLE_MASK_FLOW_COUNTERS = _anonenum85.define('MLX5_SET_FTE_MODIFY_ENABLE_MASK_FLOW_COUNTERS', 3)
MLX5_SET_FTE_MODIFY_ENABLE_MASK_IPSEC_OBJ_ID = _anonenum85.define('MLX5_SET_FTE_MODIFY_ENABLE_MASK_IPSEC_OBJ_ID', 4)

@c.record
class struct_mlx5_ifc_set_fte_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_fte_in_bits(c.Struct):
  SIZE = 6656
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  ignore_flow_level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 192]
  reserved_at_c1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[23]], 193]
  modify_enable_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 216]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  flow_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 288]
  flow_context: Annotated[struct_mlx5_ifc_flow_context_bits, 512]
@c.record
class struct_mlx5_ifc_dest_format_bits(c.Struct):
  SIZE = 64
  destination_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  destination_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  destination_eswitch_owner_vhca_id_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  packet_reformat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 34]
  destination_eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
@c.record
class struct_mlx5_ifc_rts2rts_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_rts2rts_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 2048]
@c.record
class struct_mlx5_ifc_rtr2rts_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_rtr2rts_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 2048]
@c.record
class struct_mlx5_ifc_rst2init_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_rst2init_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 2048]
@c.record
class struct_mlx5_ifc_query_xrq_out_bits(c.Struct):
  SIZE = 2688
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  xrq_context: Annotated[struct_mlx5_ifc_xrqc_bits, 128]
@c.record
class struct_mlx5_ifc_query_xrq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_xrc_srq_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  xrc_srq_context_entry: Annotated[struct_mlx5_ifc_xrc_srqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1536]], 640]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_query_xrc_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrc_srqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
class _anonenum86(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QUERY_VPORT_STATE_OUT_STATE_DOWN = _anonenum86.define('MLX5_QUERY_VPORT_STATE_OUT_STATE_DOWN', 0)
MLX5_QUERY_VPORT_STATE_OUT_STATE_UP = _anonenum86.define('MLX5_QUERY_VPORT_STATE_OUT_STATE_UP', 1)

@c.record
class struct_mlx5_ifc_query_vport_state_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  max_tx_speed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 112]
  admin_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 120]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 124]
@c.record
class struct_mlx5_ifc_array1024_auto_bits(c.Struct):
  SIZE = 1024
  array1024_auto: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_query_vuid_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 32]
  query_vfs_vuid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  data_direct: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 97]
  reserved_at_62: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 98]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_query_vuid_out_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[416]], 64]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 480]
  num_of_entries: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 496]
  vuid: Annotated[c.Array[struct_mlx5_ifc_array1024_auto_bits, Literal[0]], 512]
class _anonenum87(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_VPORT_STATE_OP_MOD_VNIC_VPORT = _anonenum87.define('MLX5_VPORT_STATE_OP_MOD_VNIC_VPORT', 0)
MLX5_VPORT_STATE_OP_MOD_ESW_VPORT = _anonenum87.define('MLX5_VPORT_STATE_OP_MOD_ESW_VPORT', 1)
MLX5_VPORT_STATE_OP_MOD_UPLINK = _anonenum87.define('MLX5_VPORT_STATE_OP_MOD_UPLINK', 2)

@c.record
class struct_mlx5_ifc_arm_monitor_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_arm_monitor_counter_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class _anonenum88(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QUERY_MONITOR_CNT_TYPE_PPCNT = _anonenum88.define('MLX5_QUERY_MONITOR_CNT_TYPE_PPCNT', 0)
MLX5_QUERY_MONITOR_CNT_TYPE_Q_COUNTER = _anonenum88.define('MLX5_QUERY_MONITOR_CNT_TYPE_Q_COUNTER', 1)

class enum_mlx5_monitor_counter_ppcnt(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QUERY_MONITOR_PPCNT_IN_RANGE_LENGTH_ERRORS = enum_mlx5_monitor_counter_ppcnt.define('MLX5_QUERY_MONITOR_PPCNT_IN_RANGE_LENGTH_ERRORS', 0)
MLX5_QUERY_MONITOR_PPCNT_OUT_OF_RANGE_LENGTH_FIELD = enum_mlx5_monitor_counter_ppcnt.define('MLX5_QUERY_MONITOR_PPCNT_OUT_OF_RANGE_LENGTH_FIELD', 1)
MLX5_QUERY_MONITOR_PPCNT_FRAME_TOO_LONG_ERRORS = enum_mlx5_monitor_counter_ppcnt.define('MLX5_QUERY_MONITOR_PPCNT_FRAME_TOO_LONG_ERRORS', 2)
MLX5_QUERY_MONITOR_PPCNT_FRAME_CHECK_SEQUENCE_ERRORS = enum_mlx5_monitor_counter_ppcnt.define('MLX5_QUERY_MONITOR_PPCNT_FRAME_CHECK_SEQUENCE_ERRORS', 3)
MLX5_QUERY_MONITOR_PPCNT_ALIGNMENT_ERRORS = enum_mlx5_monitor_counter_ppcnt.define('MLX5_QUERY_MONITOR_PPCNT_ALIGNMENT_ERRORS', 4)
MLX5_QUERY_MONITOR_PPCNT_IF_OUT_DISCARDS = enum_mlx5_monitor_counter_ppcnt.define('MLX5_QUERY_MONITOR_PPCNT_IF_OUT_DISCARDS', 5)

class _anonenum89(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QUERY_MONITOR_Q_COUNTER_RX_OUT_OF_BUFFER = _anonenum89.define('MLX5_QUERY_MONITOR_Q_COUNTER_RX_OUT_OF_BUFFER', 4)

@c.record
class struct_mlx5_ifc_monitor_counter_output_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  counter_group_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_set_monitor_counter_in_bits(c.Struct):
  SIZE = 576
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  num_of_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  monitor_counter: Annotated[c.Array[struct_mlx5_ifc_monitor_counter_output_bits, Literal[7]], 128]
@c.record
class struct_mlx5_ifc_set_monitor_counter_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_query_vport_state_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_vnic_env_out_bits(c.Struct):
  SIZE = 4224
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  vport_env: Annotated[struct_mlx5_ifc_vnic_diagnostic_statistics_bits, 128]
class _anonenum90(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QUERY_VNIC_ENV_IN_OP_MOD_VPORT_DIAG_STATISTICS = _anonenum90.define('MLX5_QUERY_VNIC_ENV_IN_OP_MOD_VPORT_DIAG_STATISTICS', 0)

@c.record
class struct_mlx5_ifc_query_vnic_env_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_vport_counter_out_bits(c.Struct):
  SIZE = 4224
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  received_errors: Annotated[struct_mlx5_ifc_traffic_counter_bits, 128]
  transmit_errors: Annotated[struct_mlx5_ifc_traffic_counter_bits, 256]
  received_ib_unicast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 384]
  transmitted_ib_unicast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 512]
  received_ib_multicast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 640]
  transmitted_ib_multicast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 768]
  received_eth_broadcast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 896]
  transmitted_eth_broadcast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 1024]
  received_eth_unicast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 1152]
  transmitted_eth_unicast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 1280]
  received_eth_multicast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 1408]
  transmitted_eth_multicast: Annotated[struct_mlx5_ifc_traffic_counter_bits, 1536]
  local_loopback: Annotated[struct_mlx5_ifc_traffic_counter_bits, 1664]
  reserved_at_700: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2432]], 1792]
class _anonenum91(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QUERY_VPORT_COUNTER_IN_OP_MOD_VPORT_COUNTERS = _anonenum91.define('MLX5_QUERY_VPORT_COUNTER_IN_OP_MOD_VPORT_COUNTERS', 0)

@c.record
class struct_mlx5_ifc_query_vport_counter_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 65]
  port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 96]
  clear: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 192]
  reserved_at_c1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 193]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_query_tis_out_bits(c.Struct):
  SIZE = 1408
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  tis_context: Annotated[struct_mlx5_ifc_tisc_bits, 128]
@c.record
class struct_mlx5_ifc_query_tis_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  tisn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_tir_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  tir_context: Annotated[struct_mlx5_ifc_tirc_bits, 256]
@c.record
class struct_mlx5_ifc_query_tir_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  tirn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_srq_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  srq_context_entry: Annotated[struct_mlx5_ifc_srqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1536]], 640]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_query_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  srqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_sq_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  sq_context: Annotated[struct_mlx5_ifc_sqc_bits, 256]
@c.record
class struct_mlx5_ifc_query_sq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  sqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_special_contexts_out_bits(c.Struct):
  SIZE = 256
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  dump_fill_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  resd_lkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  null_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  terminate_scatter_list_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  repeated_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_query_special_contexts_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_query_scheduling_element_out_bits(c.Struct):
  SIZE = 1024
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  scheduling_context: Annotated[struct_mlx5_ifc_scheduling_context_bits, 256]
  reserved_at_300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 768]
class _anonenum92(Annotated[int, ctypes.c_uint32], c.Enum): pass
SCHEDULING_HIERARCHY_E_SWITCH = _anonenum92.define('SCHEDULING_HIERARCHY_E_SWITCH', 2)
SCHEDULING_HIERARCHY_NIC = _anonenum92.define('SCHEDULING_HIERARCHY_NIC', 3)

@c.record
class struct_mlx5_ifc_query_scheduling_element_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  scheduling_hierarchy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  scheduling_element_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
@c.record
class struct_mlx5_ifc_query_rqt_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  rqt_context: Annotated[struct_mlx5_ifc_rqtc_bits, 256]
@c.record
class struct_mlx5_ifc_query_rqt_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rqtn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_rq_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  rq_context: Annotated[struct_mlx5_ifc_rqc_bits, 256]
@c.record
class struct_mlx5_ifc_query_rq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_roce_address_out_bits(c.Struct):
  SIZE = 384
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  roce_address: Annotated[struct_mlx5_ifc_roce_addr_layout_bits, 128]
@c.record
class struct_mlx5_ifc_query_roce_address_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  roce_address_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 80]
  vhca_port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_rmp_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  rmp_context: Annotated[struct_mlx5_ifc_rmpc_bits, 256]
@c.record
class struct_mlx5_ifc_query_rmp_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rmpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_cqe_error_syndrome_bits(c.Struct):
  SIZE = 32
  hw_error_syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  hw_syndrome_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 8]
  reserved_at_c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
  vendor_error_syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
@c.record
class struct_mlx5_ifc_qp_context_extension_bits(c.Struct):
  SIZE = 1536
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 0]
  error_syndrome: Annotated[struct_mlx5_ifc_cqe_error_syndrome_bits, 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1408]], 128]
@c.record
class struct_mlx5_ifc_qpc_extension_and_pas_list_in_bits(c.Struct):
  SIZE = 1536
  qpc_data_extension: Annotated[struct_mlx5_ifc_qp_context_extension_bits, 0]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 1536]
@c.record
class struct_mlx5_ifc_qp_pas_list_in_bits(c.Struct):
  SIZE = 0
  pas: Annotated[c.Array[struct_mlx5_ifc_cmd_pas_bits, Literal[0]], 0]
@c.record
class union_mlx5_ifc_qp_pas_or_qpc_ext_and_pas_bits(c.Struct):
  SIZE = 1536
  qp_pas_list: Annotated[struct_mlx5_ifc_qp_pas_list_in_bits, 0]
  qpc_ext_and_pas_list: Annotated[struct_mlx5_ifc_qpc_extension_and_pas_list_in_bits, 0]
@c.record
class struct_mlx5_ifc_query_qp_out_bits(c.Struct):
  SIZE = 3712
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 2048]
  qp_pas_or_qpc_ext_and_pas: Annotated[union_mlx5_ifc_qp_pas_or_qpc_ext_and_pas_bits, 2176]
@c.record
class struct_mlx5_ifc_query_qp_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  qpc_ext: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 65]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_q_counter_out_bits(c.Struct):
  SIZE = 2048
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  rx_write_requests: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  rx_read_requests: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  rx_atomic_requests: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  rx_dct_connect: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  out_of_buffer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  reserved_at_1a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  out_of_sequence: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  duplicate_request: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  reserved_at_220: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 544]
  rnr_nak_retry_err: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  reserved_at_260: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  packet_seq_err: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 640]
  reserved_at_2a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  implied_nak_seq_err: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  reserved_at_2e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  local_ack_timeout_err: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  reserved_at_320: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 800]
  req_rnr_retries_exceeded: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 896]
  reserved_at_3a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 928]
  resp_local_length_error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 960]
  req_local_length_error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 992]
  resp_local_qp_error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1024]
  local_operation_error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1056]
  resp_local_protection: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1088]
  req_local_protection: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1120]
  resp_cqe_error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1152]
  req_cqe_error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1184]
  req_mw_binding: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1216]
  req_bad_response: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1248]
  req_remote_invalid_request: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1280]
  resp_remote_invalid_request: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1312]
  req_remote_access_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1344]
  resp_remote_access_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1376]
  req_remote_operation_errors: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1408]
  req_transport_retries_exceeded: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1440]
  cq_overflow: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1472]
  resp_cqe_flush_error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1504]
  req_cqe_flush_error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1536]
  reserved_at_620: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1568]
  roce_adp_retrans: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1600]
  roce_adp_retrans_to: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1632]
  roce_slow_restart: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1664]
  roce_slow_restart_cnps: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1696]
  roce_slow_restart_trans: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 1728]
  reserved_at_6e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[288]], 1760]
@c.record
class struct_mlx5_ifc_query_q_counter_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 96]
  clear: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 192]
  aggregate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 193]
  reserved_at_c2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 194]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 224]
  counter_set_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 248]
@c.record
class struct_mlx5_ifc_query_pages_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  embedded_cpu_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  num_pages: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
class _anonenum93(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QUERY_PAGES_IN_OP_MOD_BOOT_PAGES = _anonenum93.define('MLX5_QUERY_PAGES_IN_OP_MOD_BOOT_PAGES', 1)
MLX5_QUERY_PAGES_IN_OP_MOD_INIT_PAGES = _anonenum93.define('MLX5_QUERY_PAGES_IN_OP_MOD_INIT_PAGES', 2)
MLX5_QUERY_PAGES_IN_OP_MOD_REGULAR_PAGES = _anonenum93.define('MLX5_QUERY_PAGES_IN_OP_MOD_REGULAR_PAGES', 3)

@c.record
class struct_mlx5_ifc_query_pages_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  embedded_cpu_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_nic_vport_context_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  nic_vport_context: Annotated[struct_mlx5_ifc_nic_vport_context_bits, 128]
@c.record
class struct_mlx5_ifc_query_nic_vport_context_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 96]
  allowed_list_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 101]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
@c.record
class struct_mlx5_ifc_query_mkey_out_bits(c.Struct):
  SIZE = 2432
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  memory_key_mkey_entry: Annotated[struct_mlx5_ifc_mkc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1536]], 640]
  bsf0_klm0_pas_mtt0_1: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 2176]
  bsf1_klm1_pas_mtt2_3: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 2304]
@c.record
class struct_mlx5_ifc_query_mkey_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  mkey_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  pg_access: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  reserved_at_61: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 97]
@c.record
class struct_mlx5_ifc_query_mad_demux_out_bits(c.Struct):
  SIZE = 160
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  mad_dumux_parameters_block: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
@c.record
class struct_mlx5_ifc_query_mad_demux_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_query_l2_table_entry_out_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 64]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[19]], 224]
  vlan_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 243]
  vlan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 244]
  mac_address: Annotated[struct_mlx5_ifc_mac_address_layout_bits, 256]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 320]
@c.record
class struct_mlx5_ifc_query_l2_table_entry_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 64]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 192]
@c.record
class struct_mlx5_ifc_query_issi_out_bits(c.Struct):
  SIZE = 896
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  current_issi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 96]
  reserved_at_100: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[76]], 256]
  supported_issi_dw0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 864]
@c.record
class struct_mlx5_ifc_query_issi_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_driver_version_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_driver_version_in_bits(c.Struct):
  SIZE = 640
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  driver_version: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[64]], 128]
@c.record
class struct_mlx5_ifc_query_hca_vport_pkey_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  pkey: Annotated[c.Array[struct_mlx5_ifc_pkey_bits, Literal[0]], 128]
@c.record
class struct_mlx5_ifc_query_hca_vport_pkey_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 65]
  port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  pkey_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
class _anonenum94(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_HCA_VPORT_SEL_PORT_GUID = _anonenum94.define('MLX5_HCA_VPORT_SEL_PORT_GUID', 1)
MLX5_HCA_VPORT_SEL_NODE_GUID = _anonenum94.define('MLX5_HCA_VPORT_SEL_NODE_GUID', 2)
MLX5_HCA_VPORT_SEL_STATE_POLICY = _anonenum94.define('MLX5_HCA_VPORT_SEL_STATE_POLICY', 4)

@c.record
class struct_mlx5_ifc_query_hca_vport_gid_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  gids_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  gid: Annotated[c.Array[struct_mlx5_ifc_array128_auto_bits, Literal[0]], 128]
@c.record
class struct_mlx5_ifc_query_hca_vport_gid_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 65]
  port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  gid_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_query_hca_vport_context_out_bits(c.Struct):
  SIZE = 4224
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  hca_vport_context: Annotated[struct_mlx5_ifc_hca_vport_context_bits, 128]
@c.record
class struct_mlx5_ifc_query_hca_vport_context_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 65]
  port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_hca_cap_out_bits(c.Struct):
  SIZE = 32896
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  capability: Annotated[union_mlx5_ifc_hca_cap_union_bits, 128]
@c.record
class struct_mlx5_ifc_query_hca_cap_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  ec_vf_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 66]
  function_id_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 67]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 68]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_other_hca_cap_bits(c.Struct):
  SIZE = 640
  roce: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[639]], 1]
@c.record
class struct_mlx5_ifc_query_other_hca_cap_out_bits(c.Struct):
  SIZE = 768
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  other_capability: Annotated[struct_mlx5_ifc_other_hca_cap_bits, 128]
@c.record
class struct_mlx5_ifc_query_other_hca_cap_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_modify_other_hca_cap_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_other_hca_cap_in_bits(c.Struct):
  SIZE = 768
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  other_capability: Annotated[struct_mlx5_ifc_other_hca_cap_bits, 128]
@c.record
class struct_mlx5_ifc_sw_owner_icm_root_params_bits(c.Struct):
  SIZE = 128
  sw_owner_icm_root_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  sw_owner_icm_root_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_rtc_params_bits(c.Struct):
  SIZE = 128
  rtc_id_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  rtc_id_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_flow_table_context_bits(c.Struct):
  SIZE = 320
  reformat_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  decap_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  sw_owner: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  termination_table: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  table_miss_action: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  rtc_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  reserved_at_11: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 17]
  log_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  table_miss_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  lag_master_next_table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 96]
  sws: Annotated[struct_mlx5_ifc_sw_owner_icm_root_params_bits, 192]
  hws: Annotated[struct_mlx5_ifc_rtc_params_bits, 192]
@c.record
class struct_mlx5_ifc_query_flow_table_out_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 64]
  flow_table_context: Annotated[struct_mlx5_ifc_flow_table_context_bits, 192]
@c.record
class struct_mlx5_ifc_query_flow_table_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 192]
@c.record
class struct_mlx5_ifc_query_fte_out_bits(c.Struct):
  SIZE = 6656
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[448]], 64]
  flow_context: Annotated[struct_mlx5_ifc_flow_context_bits, 512]
@c.record
class struct_mlx5_ifc_query_fte_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  flow_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 288]
@c.record
class struct_mlx5_ifc_match_definer_format_0_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 0]
  metadata_reg_c_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  metadata_reg_c_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  outer_dmac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  outer_dmac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 352]
  outer_ethertype: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 368]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 384]
  sx_sniffer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 385]
  functional_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 386]
  outer_ip_frag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 387]
  outer_qp_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 388]
  outer_encap_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 390]
  port_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 392]
  outer_l3_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 394]
  outer_l4_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 396]
  outer_first_vlan_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 398]
  outer_first_vlan_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 400]
  outer_first_vlan_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 403]
  outer_first_vlan_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 404]
  outer_l4_type_ext: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 416]
  reserved_at_1a4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 420]
  outer_ipsec_layer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 422]
  outer_l2_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 424]
  force_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 426]
  outer_l2_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 427]
  outer_l3_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 428]
  outer_l4_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 429]
  outer_second_vlan_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 430]
  outer_second_vlan_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 432]
  outer_second_vlan_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 435]
  outer_second_vlan_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 436]
  outer_smac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  outer_smac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 480]
  inner_ipv4_checksum_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 496]
  inner_l4_checksum_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 497]
  outer_ipv4_checksum_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 498]
  outer_l4_checksum_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 499]
  inner_l3_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 500]
  inner_l4_ok: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 501]
  outer_l3_ok_duplicate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 502]
  outer_l4_ok_duplicate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 503]
  outer_tcp_cwr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 504]
  outer_tcp_ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 505]
  outer_tcp_urg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 506]
  outer_tcp_ack: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 507]
  outer_tcp_psh: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 508]
  outer_tcp_rst: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 509]
  outer_tcp_syn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 510]
  outer_tcp_fin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 511]
@c.record
class struct_mlx5_ifc_match_definer_format_22_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 0]
  outer_ip_src_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  outer_ip_dest_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  outer_l4_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 320]
  outer_l4_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 336]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 352]
  sx_sniffer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 353]
  functional_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 354]
  outer_ip_frag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 355]
  outer_qp_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 356]
  outer_encap_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 358]
  port_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 360]
  outer_l3_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 362]
  outer_l4_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 364]
  outer_first_vlan_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 366]
  outer_first_vlan_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 368]
  outer_first_vlan_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 371]
  outer_first_vlan_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 372]
  metadata_reg_c_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  outer_dmac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  outer_smac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  outer_smac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 480]
  outer_dmac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 496]
@c.record
class struct_mlx5_ifc_match_definer_format_23_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 0]
  inner_ip_src_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  inner_ip_dest_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  inner_l4_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 320]
  inner_l4_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 336]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 352]
  sx_sniffer: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 353]
  functional_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 354]
  inner_ip_frag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 355]
  inner_qp_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 356]
  inner_encap_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 358]
  port_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 360]
  inner_l3_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 362]
  inner_l4_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 364]
  inner_first_vlan_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 366]
  inner_first_vlan_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 368]
  inner_first_vlan_cfi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 371]
  inner_first_vlan_vid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 372]
  tunnel_header_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  inner_dmac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  inner_smac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  inner_smac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 480]
  inner_dmac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 496]
@c.record
class struct_mlx5_ifc_match_definer_format_29_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 0]
  outer_ip_dest_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 192]
  outer_ip_src_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 320]
  outer_l4_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 448]
  outer_l4_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 464]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
@c.record
class struct_mlx5_ifc_match_definer_format_30_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 0]
  outer_ip_dest_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 160]
  outer_ip_src_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 288]
  outer_dmac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  outer_smac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  outer_smac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 480]
  outer_dmac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 496]
@c.record
class struct_mlx5_ifc_match_definer_format_31_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 0]
  inner_ip_dest_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 192]
  inner_ip_src_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 320]
  inner_l4_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 448]
  inner_l4_dport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 464]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
@c.record
class struct_mlx5_ifc_match_definer_format_32_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 0]
  inner_ip_dest_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 160]
  inner_ip_src_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 288]
  inner_dmac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  inner_smac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  inner_smac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 480]
  inner_dmac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 496]
class _anonenum95(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_IFC_DEFINER_FORMAT_ID_SELECT = _anonenum95.define('MLX5_IFC_DEFINER_FORMAT_ID_SELECT', 61)

@c.record
class struct_mlx5_ifc_match_definer_match_mask_bits(c.Struct):
  SIZE = 512
  reserved_at_1c0: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[5]], 0]
  match_dw_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  match_dw_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  match_dw_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  match_dw_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  match_dw_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  match_dw_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  match_dw_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  match_dw_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  match_dw_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  match_byte_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 448]
  match_byte_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 456]
  match_byte_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 464]
  match_byte_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 472]
  match_byte_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 480]
  match_byte_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 488]
  match_byte_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 496]
  match_byte_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 504]
@c.record
class struct_mlx5_ifc_match_definer_bits(c.Struct):
  SIZE = 1024
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  format_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 160]
  format_select_dw3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  format_select_dw2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 264]
  format_select_dw1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 272]
  format_select_dw0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 280]
  format_select_dw7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 288]
  format_select_dw6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 296]
  format_select_dw5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 304]
  format_select_dw4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 312]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 320]
  format_select_dw8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 344]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  format_select_byte3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 384]
  format_select_byte2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 392]
  format_select_byte1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 400]
  format_select_byte0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 408]
  format_select_byte7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 416]
  format_select_byte6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 424]
  format_select_byte5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 432]
  format_select_byte4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 440]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
  match_mask: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[16]], 512]
  match_mask_format: Annotated[struct_mlx5_ifc_match_definer_match_mask_bits, 512]
@c.record
class struct_mlx5_ifc_general_obj_create_param_bits(c.Struct):
  SIZE = 32
  alias_object: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 1]
  log_obj_range: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 3]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
@c.record
class struct_mlx5_ifc_general_obj_query_param_bits(c.Struct):
  SIZE = 32
  alias_object: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  obj_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 1]
@c.record
class struct_mlx5_ifc_general_obj_in_cmd_hdr_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  vhca_tunnel_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  obj_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  obj_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  op_param: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits_op_param, 96]
@c.record
class struct_mlx5_ifc_general_obj_in_cmd_hdr_bits_op_param(c.Struct):
  SIZE = 32
  create: Annotated[struct_mlx5_ifc_general_obj_create_param_bits, 0]
  query: Annotated[struct_mlx5_ifc_general_obj_query_param_bits, 0]
@c.record
class struct_mlx5_ifc_general_obj_out_cmd_hdr_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  obj_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_allow_other_vhca_access_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[80]], 64]
  object_type_to_be_accessed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  object_id_to_be_accessed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  access_key_raw: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 256]
  access_key: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 256]
@c.record
class struct_mlx5_ifc_allow_other_vhca_access_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_header_arg_bits(c.Struct):
  SIZE = 160
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  access_pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
@c.record
class struct_mlx5_ifc_create_modify_header_arg_in_bits(c.Struct):
  SIZE = 288
  hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  arg: Annotated[struct_mlx5_ifc_modify_header_arg_bits, 128]
@c.record
class struct_mlx5_ifc_create_match_definer_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  obj_context: Annotated[struct_mlx5_ifc_match_definer_bits, 128]
@c.record
class struct_mlx5_ifc_create_match_definer_out_bits(c.Struct):
  SIZE = 128
  general_obj_out_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0]
@c.record
class struct_mlx5_ifc_alias_context_bits(c.Struct):
  SIZE = 512
  vhca_id_to_be_accessed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[13]], 16]
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 29]
  object_id_to_be_accessed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  access_key_raw: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 128]
  access_key: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 128]
  metadata: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 384]
@c.record
class struct_mlx5_ifc_create_alias_obj_in_bits(c.Struct):
  SIZE = 640
  hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  alias_ctx: Annotated[struct_mlx5_ifc_alias_context_bits, 128]
class _anonenum96(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_OUTER_HEADERS = _anonenum96.define('MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_OUTER_HEADERS', 0)
MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS = _anonenum96.define('MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS', 1)
MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_INNER_HEADERS = _anonenum96.define('MLX5_QUERY_FLOW_GROUP_OUT_MATCH_CRITERIA_ENABLE_INNER_HEADERS', 2)
MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_2 = _anonenum96.define('MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_2', 3)
MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_3 = _anonenum96.define('MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_3', 4)
MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_4 = _anonenum96.define('MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_4', 5)
MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_5 = _anonenum96.define('MLX5_QUERY_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_5', 6)

@c.record
class struct_mlx5_ifc_query_flow_group_out_bits(c.Struct):
  SIZE = 8192
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 64]
  start_flow_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  end_flow_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 320]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 480]
  match_criteria_enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 504]
  match_criteria: Annotated[struct_mlx5_ifc_fte_match_param_bits, 512]
  reserved_at_1200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3584]], 4608]
@c.record
class struct_mlx5_ifc_query_flow_group_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  group_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[288]], 224]
@c.record
class struct_mlx5_ifc_query_flow_counter_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  flow_statistics: Annotated[c.Array[struct_mlx5_ifc_traffic_counter_bits, Literal[0]], 128]
@c.record
class struct_mlx5_ifc_query_flow_counter_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 64]
  clear: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 192]
  reserved_at_c1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 193]
  num_of_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 208]
  flow_counter_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_query_esw_vport_context_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  esw_vport_context: Annotated[struct_mlx5_ifc_esw_vport_context_bits, 128]
@c.record
class struct_mlx5_ifc_query_esw_vport_context_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_esw_vport_out_bits(c.Struct):
  SIZE = 96
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
@c.record
class struct_mlx5_ifc_destroy_esw_vport_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  vport_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_modify_esw_vport_context_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_esw_vport_context_fields_select_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[27]], 0]
  fdb_to_vport_reg_c_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 27]
  vport_cvlan_insert: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  vport_svlan_insert: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  vport_cvlan_strip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 30]
  vport_svlan_strip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
@c.record
class struct_mlx5_ifc_modify_esw_vport_context_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  field_select: Annotated[struct_mlx5_ifc_esw_vport_context_fields_select_bits, 96]
  esw_vport_context: Annotated[struct_mlx5_ifc_esw_vport_context_bits, 128]
@c.record
class struct_mlx5_ifc_query_eq_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  eq_context_entry: Annotated[struct_mlx5_ifc_eqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 640]
  event_bitmask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 704]
  reserved_at_300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1408]], 768]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_query_eq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  eq_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_packet_reformat_context_in_bits(c.Struct):
  SIZE = 64
  reformat_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 8]
  reformat_param_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 16]
  reformat_data_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 22]
  reformat_param_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  reserved_at_28: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  reformat_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[2]], 48]
  more_reformat_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[0]], 64]
@c.record
class struct_mlx5_ifc_query_packet_reformat_context_out_bits(c.Struct):
  SIZE = 224
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 64]
  packet_reformat_context: Annotated[c.Array[struct_mlx5_ifc_packet_reformat_context_in_bits, Literal[0]], 224]
@c.record
class struct_mlx5_ifc_query_packet_reformat_context_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  packet_reformat_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 96]
@c.record
class struct_mlx5_ifc_alloc_packet_reformat_context_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  packet_reformat_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
class _anonenum97(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_REFORMAT_CONTEXT_ANCHOR_MAC_START = _anonenum97.define('MLX5_REFORMAT_CONTEXT_ANCHOR_MAC_START', 1)
MLX5_REFORMAT_CONTEXT_ANCHOR_VLAN_START = _anonenum97.define('MLX5_REFORMAT_CONTEXT_ANCHOR_VLAN_START', 2)
MLX5_REFORMAT_CONTEXT_ANCHOR_IP_START = _anonenum97.define('MLX5_REFORMAT_CONTEXT_ANCHOR_IP_START', 7)
MLX5_REFORMAT_CONTEXT_ANCHOR_TCP_UDP_START = _anonenum97.define('MLX5_REFORMAT_CONTEXT_ANCHOR_TCP_UDP_START', 9)

class enum_mlx5_reformat_ctx_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_REFORMAT_TYPE_L2_TO_VXLAN = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_L2_TO_VXLAN', 0)
MLX5_REFORMAT_TYPE_L2_TO_NVGRE = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_L2_TO_NVGRE', 1)
MLX5_REFORMAT_TYPE_L2_TO_L2_TUNNEL = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_L2_TO_L2_TUNNEL', 2)
MLX5_REFORMAT_TYPE_L3_TUNNEL_TO_L2 = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_L3_TUNNEL_TO_L2', 3)
MLX5_REFORMAT_TYPE_L2_TO_L3_TUNNEL = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_L2_TO_L3_TUNNEL', 4)
MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_IPV4 = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_IPV4', 5)
MLX5_REFORMAT_TYPE_L2_TO_L3_ESP_TUNNEL = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_L2_TO_L3_ESP_TUNNEL', 6)
MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_UDPV4 = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_UDPV4', 7)
MLX5_REFORMAT_TYPE_DEL_ESP_TRANSPORT = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_DEL_ESP_TRANSPORT', 8)
MLX5_REFORMAT_TYPE_L3_ESP_TUNNEL_TO_L2 = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_L3_ESP_TUNNEL_TO_L2', 9)
MLX5_REFORMAT_TYPE_DEL_ESP_TRANSPORT_OVER_UDP = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_DEL_ESP_TRANSPORT_OVER_UDP', 10)
MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_IPV6 = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_IPV6', 11)
MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_UDPV6 = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_ADD_ESP_TRANSPORT_OVER_UDPV6', 12)
MLX5_REFORMAT_TYPE_ADD_PSP_TUNNEL = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_ADD_PSP_TUNNEL', 13)
MLX5_REFORMAT_TYPE_DEL_PSP_TUNNEL = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_DEL_PSP_TUNNEL', 14)
MLX5_REFORMAT_TYPE_INSERT_HDR = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_INSERT_HDR', 15)
MLX5_REFORMAT_TYPE_REMOVE_HDR = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_REMOVE_HDR', 16)
MLX5_REFORMAT_TYPE_ADD_MACSEC = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_ADD_MACSEC', 17)
MLX5_REFORMAT_TYPE_DEL_MACSEC = enum_mlx5_reformat_ctx_type.define('MLX5_REFORMAT_TYPE_DEL_MACSEC', 18)

@c.record
class struct_mlx5_ifc_alloc_packet_reformat_context_in_bits(c.Struct):
  SIZE = 288
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 64]
  packet_reformat_context: Annotated[struct_mlx5_ifc_packet_reformat_context_in_bits, 224]
@c.record
class struct_mlx5_ifc_dealloc_packet_reformat_context_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_packet_reformat_context_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  packet_reformat_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_set_action_in_bits(c.Struct):
  SIZE = 64
  action_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  field: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 4]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 16]
  offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 19]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 24]
  length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 27]
  data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_add_action_in_bits(c.Struct):
  SIZE = 64
  action_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  field: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 4]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_copy_action_in_bits(c.Struct):
  SIZE = 64
  action_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  src_field: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 4]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 16]
  src_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 19]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 24]
  length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 27]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 32]
  dst_field: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 36]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 48]
  dst_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 51]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
@c.record
class union_mlx5_ifc_set_add_copy_action_in_auto_bits(c.Struct):
  SIZE = 64
  set_action_in: Annotated[struct_mlx5_ifc_set_action_in_bits, 0]
  add_action_in: Annotated[struct_mlx5_ifc_add_action_in_bits, 0]
  copy_action_in: Annotated[struct_mlx5_ifc_copy_action_in_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
class _anonenum98(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ACTION_TYPE_SET = _anonenum98.define('MLX5_ACTION_TYPE_SET', 1)
MLX5_ACTION_TYPE_ADD = _anonenum98.define('MLX5_ACTION_TYPE_ADD', 2)
MLX5_ACTION_TYPE_COPY = _anonenum98.define('MLX5_ACTION_TYPE_COPY', 3)

class _anonenum99(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ACTION_IN_FIELD_OUT_SMAC_47_16 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_SMAC_47_16', 1)
MLX5_ACTION_IN_FIELD_OUT_SMAC_15_0 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_SMAC_15_0', 2)
MLX5_ACTION_IN_FIELD_OUT_ETHERTYPE = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_ETHERTYPE', 3)
MLX5_ACTION_IN_FIELD_OUT_DMAC_47_16 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_DMAC_47_16', 4)
MLX5_ACTION_IN_FIELD_OUT_DMAC_15_0 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_DMAC_15_0', 5)
MLX5_ACTION_IN_FIELD_OUT_IP_DSCP = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_IP_DSCP', 6)
MLX5_ACTION_IN_FIELD_OUT_TCP_FLAGS = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_TCP_FLAGS', 7)
MLX5_ACTION_IN_FIELD_OUT_TCP_SPORT = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_TCP_SPORT', 8)
MLX5_ACTION_IN_FIELD_OUT_TCP_DPORT = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_TCP_DPORT', 9)
MLX5_ACTION_IN_FIELD_OUT_IP_TTL = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_IP_TTL', 10)
MLX5_ACTION_IN_FIELD_OUT_UDP_SPORT = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_UDP_SPORT', 11)
MLX5_ACTION_IN_FIELD_OUT_UDP_DPORT = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_UDP_DPORT', 12)
MLX5_ACTION_IN_FIELD_OUT_SIPV6_127_96 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_SIPV6_127_96', 13)
MLX5_ACTION_IN_FIELD_OUT_SIPV6_95_64 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_SIPV6_95_64', 14)
MLX5_ACTION_IN_FIELD_OUT_SIPV6_63_32 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_SIPV6_63_32', 15)
MLX5_ACTION_IN_FIELD_OUT_SIPV6_31_0 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_SIPV6_31_0', 16)
MLX5_ACTION_IN_FIELD_OUT_DIPV6_127_96 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_DIPV6_127_96', 17)
MLX5_ACTION_IN_FIELD_OUT_DIPV6_95_64 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_DIPV6_95_64', 18)
MLX5_ACTION_IN_FIELD_OUT_DIPV6_63_32 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_DIPV6_63_32', 19)
MLX5_ACTION_IN_FIELD_OUT_DIPV6_31_0 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_DIPV6_31_0', 20)
MLX5_ACTION_IN_FIELD_OUT_SIPV4 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_SIPV4', 21)
MLX5_ACTION_IN_FIELD_OUT_DIPV4 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_DIPV4', 22)
MLX5_ACTION_IN_FIELD_OUT_FIRST_VID = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_FIRST_VID', 23)
MLX5_ACTION_IN_FIELD_OUT_IPV6_HOPLIMIT = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_IPV6_HOPLIMIT', 71)
MLX5_ACTION_IN_FIELD_METADATA_REG_A = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_A', 73)
MLX5_ACTION_IN_FIELD_METADATA_REG_B = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_B', 80)
MLX5_ACTION_IN_FIELD_METADATA_REG_C_0 = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_C_0', 81)
MLX5_ACTION_IN_FIELD_METADATA_REG_C_1 = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_C_1', 82)
MLX5_ACTION_IN_FIELD_METADATA_REG_C_2 = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_C_2', 83)
MLX5_ACTION_IN_FIELD_METADATA_REG_C_3 = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_C_3', 84)
MLX5_ACTION_IN_FIELD_METADATA_REG_C_4 = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_C_4', 85)
MLX5_ACTION_IN_FIELD_METADATA_REG_C_5 = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_C_5', 86)
MLX5_ACTION_IN_FIELD_METADATA_REG_C_6 = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_C_6', 87)
MLX5_ACTION_IN_FIELD_METADATA_REG_C_7 = _anonenum99.define('MLX5_ACTION_IN_FIELD_METADATA_REG_C_7', 88)
MLX5_ACTION_IN_FIELD_OUT_TCP_SEQ_NUM = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_TCP_SEQ_NUM', 89)
MLX5_ACTION_IN_FIELD_OUT_TCP_ACK_NUM = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_TCP_ACK_NUM', 91)
MLX5_ACTION_IN_FIELD_IPSEC_SYNDROME = _anonenum99.define('MLX5_ACTION_IN_FIELD_IPSEC_SYNDROME', 93)
MLX5_ACTION_IN_FIELD_OUT_EMD_47_32 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_EMD_47_32', 111)
MLX5_ACTION_IN_FIELD_OUT_EMD_31_0 = _anonenum99.define('MLX5_ACTION_IN_FIELD_OUT_EMD_31_0', 112)
MLX5_ACTION_IN_FIELD_PSP_SYNDROME = _anonenum99.define('MLX5_ACTION_IN_FIELD_PSP_SYNDROME', 113)

@c.record
class struct_mlx5_ifc_alloc_modify_header_context_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  modify_header_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_modify_header_context_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 104]
  num_of_actions: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
  actions: Annotated[c.Array[union_mlx5_ifc_set_add_copy_action_in_auto_bits, Literal[0]], 128]
@c.record
class struct_mlx5_ifc_dealloc_modify_header_context_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_modify_header_context_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  modify_header_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_modify_header_context_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  modify_header_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 96]
@c.record
class struct_mlx5_ifc_query_dct_out_bits(c.Struct):
  SIZE = 1024
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  dct_context_entry: Annotated[struct_mlx5_ifc_dctc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 640]
@c.record
class struct_mlx5_ifc_query_dct_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  dctn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_cq_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  cq_context: Annotated[struct_mlx5_ifc_cqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1536]], 640]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_query_cq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_cong_status_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  tag_enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 97]
  reserved_at_62: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 98]
@c.record
class struct_mlx5_ifc_query_cong_status_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  priority: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 88]
  cong_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_cong_statistics_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  rp_cur_flows: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  sum_flows: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  rp_cnp_ignored_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  rp_cnp_ignored_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  rp_cnp_handled_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  rp_cnp_handled_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 320]
  time_stamp_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 576]
  time_stamp_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 608]
  accumulators_period: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 640]
  np_ecn_marked_roce_packets_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 672]
  np_ecn_marked_roce_packets_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 704]
  np_cnp_sent_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 736]
  np_cnp_sent_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  reserved_at_320: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1376]], 800]
@c.record
class struct_mlx5_ifc_query_cong_statistics_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  clear: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 65]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_cong_params_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  congestion_parameters: Annotated[union_mlx5_ifc_cong_control_roce_ecn_auto_bits, 128]
@c.record
class struct_mlx5_ifc_query_cong_params_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 64]
  cong_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_adapter_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  query_adapter_struct: Annotated[struct_mlx5_ifc_query_adapter_param_block_bits, 128]
@c.record
class struct_mlx5_ifc_query_adapter_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_function_vhca_rid_info_reg_bits(c.Struct):
  SIZE = 128
  host_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  host_pci_device_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  host_pci_bus: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 24]
  pci_bus_assigned: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 27]
  function_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  parent_pci_device_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  parent_pci_bus: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_delegated_function_vhca_rid_info_bits(c.Struct):
  SIZE = 256
  function_vhca_rid_info: Annotated[struct_mlx5_ifc_function_vhca_rid_info_reg_bits, 0]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 128]
  manage_profile: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 152]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 160]
@c.record
class struct_mlx5_ifc_query_delegated_vhca_out_bits(c.Struct):
  SIZE = 256
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  functions_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 128]
  delegated_function_vhca_rid_info: Annotated[c.Array[struct_mlx5_ifc_delegated_function_vhca_rid_info_bits, Literal[0]], 256]
@c.record
class struct_mlx5_ifc_query_delegated_vhca_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_create_esw_vport_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  vport_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_create_esw_vport_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  managed_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_qp_2rst_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_qp_2rst_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_qp_2err_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_qp_2err_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_trans_page_fault_info_bits(c.Struct):
  SIZE = 64
  error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 1]
  page_fault_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 5]
  wq_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  fault_token: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
@c.record
class struct_mlx5_ifc_mem_page_fault_info_bits(c.Struct):
  SIZE = 64
  error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 1]
  fault_token_47_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  fault_token_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class union_mlx5_ifc_page_fault_resume_in_page_fault_info_auto_bits(c.Struct):
  SIZE = 64
  trans_page_fault_info: Annotated[struct_mlx5_ifc_trans_page_fault_info_bits, 0]
  mem_page_fault_info: Annotated[struct_mlx5_ifc_mem_page_fault_info_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
@c.record
class struct_mlx5_ifc_page_fault_resume_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_page_fault_resume_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  page_fault_info: Annotated[union_mlx5_ifc_page_fault_resume_in_page_fault_info_auto_bits, 64]
@c.record
class struct_mlx5_ifc_nop_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_nop_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_vport_state_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_vport_state_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  max_tx_speed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  ingress_connect: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 112]
  egress_connect: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 113]
  ingress_connect_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 114]
  egress_connect_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 115]
  reserved_at_74: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 116]
  admin_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 120]
  reserved_at_7c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 124]
@c.record
class struct_mlx5_ifc_modify_tis_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_tis_bitmask_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[29]], 32]
  lag_tx_port_affinity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 61]
  strict_lag_tx_port_affinity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 62]
  prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
@c.record
class struct_mlx5_ifc_modify_tis_in_bits(c.Struct):
  SIZE = 1536
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  tisn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  bitmask: Annotated[struct_mlx5_ifc_modify_tis_bitmask_bits, 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  ctx: Annotated[struct_mlx5_ifc_tisc_bits, 256]
@c.record
class struct_mlx5_ifc_modify_tir_bitmask_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[27]], 32]
  self_lb_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 59]
  reserved_at_3c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 60]
  hash: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 61]
  reserved_at_3e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 62]
  packet_merge: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
@c.record
class struct_mlx5_ifc_modify_tir_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_tir_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  tirn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  bitmask: Annotated[struct_mlx5_ifc_modify_tir_bitmask_bits, 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  ctx: Annotated[struct_mlx5_ifc_tirc_bits, 256]
@c.record
class struct_mlx5_ifc_modify_sq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_sq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  sq_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 68]
  sqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  modify_bitmask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  ctx: Annotated[struct_mlx5_ifc_sqc_bits, 256]
@c.record
class struct_mlx5_ifc_modify_scheduling_element_out_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[448]], 64]
class _anonenum100(Annotated[int, ctypes.c_uint32], c.Enum): pass
MODIFY_SCHEDULING_ELEMENT_IN_MODIFY_BITMASK_BW_SHARE = _anonenum100.define('MODIFY_SCHEDULING_ELEMENT_IN_MODIFY_BITMASK_BW_SHARE', 1)
MODIFY_SCHEDULING_ELEMENT_IN_MODIFY_BITMASK_MAX_AVERAGE_BW = _anonenum100.define('MODIFY_SCHEDULING_ELEMENT_IN_MODIFY_BITMASK_MAX_AVERAGE_BW', 2)

@c.record
class struct_mlx5_ifc_modify_scheduling_element_in_bits(c.Struct):
  SIZE = 1024
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  scheduling_hierarchy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  scheduling_element_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  modify_bitmask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  scheduling_context: Annotated[struct_mlx5_ifc_scheduling_context_bits, 256]
  reserved_at_300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 768]
@c.record
class struct_mlx5_ifc_modify_rqt_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_rqt_bitmask_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 32]
  rqn_list: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
@c.record
class struct_mlx5_ifc_modify_rqt_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rqtn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  bitmask: Annotated[struct_mlx5_ifc_rqt_bitmask_bits, 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  ctx: Annotated[struct_mlx5_ifc_rqtc_bits, 256]
@c.record
class struct_mlx5_ifc_modify_rq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class _anonenum101(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_VSD = _anonenum101.define('MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_VSD', 2)
MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_SCATTER_FCS = _anonenum101.define('MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_SCATTER_FCS', 4)
MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_RQ_COUNTER_SET_ID = _anonenum101.define('MLX5_MODIFY_RQ_IN_MODIFY_BITMASK_RQ_COUNTER_SET_ID', 8)

@c.record
class struct_mlx5_ifc_modify_rq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  rq_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 68]
  rqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  modify_bitmask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  ctx: Annotated[struct_mlx5_ifc_rqc_bits, 256]
@c.record
class struct_mlx5_ifc_modify_rmp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_rmp_bitmask_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 32]
  lwm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
@c.record
class struct_mlx5_ifc_modify_rmp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  rmp_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 68]
  rmpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  bitmask: Annotated[struct_mlx5_ifc_rmp_bitmask_bits, 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  ctx: Annotated[struct_mlx5_ifc_rmpc_bits, 256]
@c.record
class struct_mlx5_ifc_modify_nic_vport_context_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_nic_vport_field_select_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[18]], 0]
  affiliation: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 18]
  reserved_at_13: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 19]
  disable_uc_local_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 20]
  disable_mc_local_lb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 21]
  node_guid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 22]
  port_guid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 23]
  min_inline: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 24]
  mtu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 25]
  change_event: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 26]
  promisc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 27]
  permanent_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  addresses_list: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  roce_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 30]
  reserved_at_1f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
@c.record
class struct_mlx5_ifc_modify_nic_vport_context_in_bits(c.Struct):
  SIZE = 4096
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  field_select: Annotated[struct_mlx5_ifc_modify_nic_vport_field_select_bits, 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1920]], 128]
  nic_vport_context: Annotated[struct_mlx5_ifc_nic_vport_context_bits, 2048]
@c.record
class struct_mlx5_ifc_modify_hca_vport_context_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_hca_vport_context_in_bits(c.Struct):
  SIZE = 4224
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 65]
  port_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  hca_vport_context: Annotated[struct_mlx5_ifc_hca_vport_context_bits, 128]
@c.record
class struct_mlx5_ifc_modify_cq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class _anonenum102(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MODIFY_CQ_IN_OP_MOD_MODIFY_CQ = _anonenum102.define('MLX5_MODIFY_CQ_IN_OP_MOD_MODIFY_CQ', 0)
MLX5_MODIFY_CQ_IN_OP_MOD_RESIZE_CQ = _anonenum102.define('MLX5_MODIFY_CQ_IN_OP_MOD_RESIZE_CQ', 1)

@c.record
class struct_mlx5_ifc_modify_cq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  modify_field_select_resize_field_select: Annotated[union_mlx5_ifc_modify_field_select_resize_field_select_auto_bits, 96]
  cq_context: Annotated[struct_mlx5_ifc_cqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 640]
  cq_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 736]
  reserved_at_2e1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 737]
  reserved_at_300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1408]], 768]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_modify_cong_status_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_cong_status_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  priority: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 88]
  cong_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  tag_enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 97]
  reserved_at_62: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 98]
@c.record
class struct_mlx5_ifc_modify_cong_params_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_cong_params_in_bits(c.Struct):
  SIZE = 2304
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 64]
  cong_protocol: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  field_select: Annotated[union_mlx5_ifc_field_select_802_1_r_roce_auto_bits, 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 128]
  congestion_parameters: Annotated[union_mlx5_ifc_cong_control_roce_ecn_auto_bits, 256]
@c.record
class struct_mlx5_ifc_manage_pages_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  output_num_entries: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 128]
class _anonenum103(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MANAGE_PAGES_IN_OP_MOD_ALLOCATION_FAIL = _anonenum103.define('MLX5_MANAGE_PAGES_IN_OP_MOD_ALLOCATION_FAIL', 0)
MLX5_MANAGE_PAGES_IN_OP_MOD_ALLOCATION_SUCCESS = _anonenum103.define('MLX5_MANAGE_PAGES_IN_OP_MOD_ALLOCATION_SUCCESS', 1)
MLX5_MANAGE_PAGES_IN_OP_MOD_HCA_RETURN_PAGES = _anonenum103.define('MLX5_MANAGE_PAGES_IN_OP_MOD_HCA_RETURN_PAGES', 2)

@c.record
class struct_mlx5_ifc_manage_pages_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  embedded_cpu_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  input_num_entries: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 128]
@c.record
class struct_mlx5_ifc_mad_ifc_out_bits(c.Struct):
  SIZE = 2176
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  response_mad_packet: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[256]], 128]
@c.record
class struct_mlx5_ifc_mad_ifc_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  remote_lid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  plane_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 80]
  port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  mad: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[256]], 128]
@c.record
class struct_mlx5_ifc_init_hca_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_init_hca_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 96]
  sw_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 98]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  sw_owner_id: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[4]], 128]
@c.record
class struct_mlx5_ifc_init2rtr_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_init2rtr_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 2048]
@c.record
class struct_mlx5_ifc_init2init_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_init2init_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 2048]
@c.record
class struct_mlx5_ifc_get_dropped_packet_log_out_bits(c.Struct):
  SIZE = 1664
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  packet_headers_log: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[128]], 128]
  packet_syndrome: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[64]], 1152]
@c.record
class struct_mlx5_ifc_get_dropped_packet_log_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_gen_eqe_in_bits(c.Struct):
  SIZE = 640
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  eq_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  eqe: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[64]], 128]
@c.record
class struct_mlx5_ifc_gen_eq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_enable_hca_out_bits(c.Struct):
  SIZE = 96
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
@c.record
class struct_mlx5_ifc_enable_hca_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  embedded_cpu_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_drain_dct_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_drain_dct_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  dctn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_disable_hca_out_bits(c.Struct):
  SIZE = 96
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
@c.record
class struct_mlx5_ifc_disable_hca_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  embedded_cpu_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 65]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_detach_from_mcg_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_detach_from_mcg_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  multicast_gid: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 128]
@c.record
class struct_mlx5_ifc_destroy_xrq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_xrq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_xrc_srq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_xrc_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrc_srqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_tis_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_tis_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  tisn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_tir_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_tir_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  tirn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_srq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  srqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_sq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_sq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  sqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_scheduling_element_out_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[448]], 64]
@c.record
class struct_mlx5_ifc_destroy_scheduling_element_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  scheduling_hierarchy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  scheduling_element_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
@c.record
class struct_mlx5_ifc_destroy_rqt_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_rqt_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rqtn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_rq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_rq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_set_delay_drop_params_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  delay_drop_timeout: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_set_delay_drop_params_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_rmp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_rmp_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rmpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_qp_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_psv_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_psv_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  psvn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_mkey_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_mkey_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  mkey_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_flow_table_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_flow_table_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 192]
@c.record
class struct_mlx5_ifc_destroy_flow_group_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_flow_group_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  group_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[288]], 224]
@c.record
class struct_mlx5_ifc_destroy_eq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_eq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  eq_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_dct_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_dct_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  dctn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_cq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_cq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_delete_vxlan_udp_dport_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_delete_vxlan_udp_dport_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  vxlan_udp_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_delete_l2_table_entry_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_delete_l2_table_entry_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 64]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 192]
@c.record
class struct_mlx5_ifc_delete_fte_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_delete_fte_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  flow_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 288]
@c.record
class struct_mlx5_ifc_dealloc_xrcd_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_xrcd_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrcd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_dealloc_uar_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_uar_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  uar: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_dealloc_transport_domain_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_transport_domain_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  transport_domain: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_dealloc_q_counter_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_q_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  counter_set_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_dealloc_pd_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_pd_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_dealloc_flow_counter_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_flow_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  flow_counter_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_xrq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_xrq_in_bits(c.Struct):
  SIZE = 2688
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  xrq_context: Annotated[struct_mlx5_ifc_xrqc_bits, 128]
@c.record
class struct_mlx5_ifc_create_xrc_srq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrc_srqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_xrc_srq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  xrc_srq_context_entry: Annotated[struct_mlx5_ifc_xrc_srqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 640]
  xrc_srq_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 736]
  reserved_at_2e1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 737]
  reserved_at_300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1408]], 768]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_create_tis_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  tisn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_tis_in_bits(c.Struct):
  SIZE = 1536
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  ctx: Annotated[struct_mlx5_ifc_tisc_bits, 256]
@c.record
class struct_mlx5_ifc_create_tir_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  icm_address_63_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  icm_address_39_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  tirn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  icm_address_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_tir_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  ctx: Annotated[struct_mlx5_ifc_tirc_bits, 256]
@c.record
class struct_mlx5_ifc_create_srq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  srqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_srq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  srq_context_entry: Annotated[struct_mlx5_ifc_srqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1536]], 640]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_create_sq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  sqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_sq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  ctx: Annotated[struct_mlx5_ifc_sqc_bits, 256]
@c.record
class struct_mlx5_ifc_create_scheduling_element_out_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  scheduling_element_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[352]], 160]
@c.record
class struct_mlx5_ifc_create_scheduling_element_in_bits(c.Struct):
  SIZE = 1024
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  scheduling_hierarchy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 96]
  scheduling_context: Annotated[struct_mlx5_ifc_scheduling_context_bits, 256]
  reserved_at_300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 768]
@c.record
class struct_mlx5_ifc_create_rqt_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rqtn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_rqt_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  rqt_context: Annotated[struct_mlx5_ifc_rqtc_bits, 256]
@c.record
class struct_mlx5_ifc_create_rq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_rq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  ctx: Annotated[struct_mlx5_ifc_rqc_bits, 256]
@c.record
class struct_mlx5_ifc_create_rmp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  rmpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_rmp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 64]
  ctx: Annotated[struct_mlx5_ifc_rmpc_bits, 256]
@c.record
class struct_mlx5_ifc_create_qp_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_qp_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  qpc_ext: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 65]
  input_qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  opt_param_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  qpc: Annotated[struct_mlx5_ifc_qpc_bits, 192]
  wq_umem_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 2048]
  wq_umem_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 2112]
  wq_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2144]
  reserved_at_861: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 2145]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_create_psv_out_bits(c.Struct):
  SIZE = 256
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  psv0_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  psv1_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  psv2_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 224]
  psv3_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 232]
@c.record
class struct_mlx5_ifc_create_psv_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  num_psv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 68]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_mkey_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  mkey_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_mkey_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  pg_access: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  mkey_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 97]
  data_direct: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 98]
  reserved_at_63: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[29]], 99]
  memory_key_mkey_entry: Annotated[struct_mlx5_ifc_mkc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 640]
  translations_octword_actual_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 768]
  reserved_at_320: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1376]], 800]
  klm_pas_mtt: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[0]], 2176]
class _anonenum104(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FLOW_TABLE_TYPE_NIC_RX = _anonenum104.define('MLX5_FLOW_TABLE_TYPE_NIC_RX', 0)
MLX5_FLOW_TABLE_TYPE_NIC_TX = _anonenum104.define('MLX5_FLOW_TABLE_TYPE_NIC_TX', 1)
MLX5_FLOW_TABLE_TYPE_ESW_EGRESS_ACL = _anonenum104.define('MLX5_FLOW_TABLE_TYPE_ESW_EGRESS_ACL', 2)
MLX5_FLOW_TABLE_TYPE_ESW_INGRESS_ACL = _anonenum104.define('MLX5_FLOW_TABLE_TYPE_ESW_INGRESS_ACL', 3)
MLX5_FLOW_TABLE_TYPE_FDB = _anonenum104.define('MLX5_FLOW_TABLE_TYPE_FDB', 4)
MLX5_FLOW_TABLE_TYPE_SNIFFER_RX = _anonenum104.define('MLX5_FLOW_TABLE_TYPE_SNIFFER_RX', 5)
MLX5_FLOW_TABLE_TYPE_SNIFFER_TX = _anonenum104.define('MLX5_FLOW_TABLE_TYPE_SNIFFER_TX', 6)

@c.record
class struct_mlx5_ifc_create_flow_table_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  icm_address_63_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  icm_address_39_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  icm_address_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_flow_table_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  flow_table_context: Annotated[struct_mlx5_ifc_flow_table_context_bits, 192]
@c.record
class struct_mlx5_ifc_create_flow_group_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  group_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
class _anonenum105(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_TCAM_SUBTABLE = _anonenum105.define('MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_TCAM_SUBTABLE', 0)
MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_HASH_SPLIT = _anonenum105.define('MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_HASH_SPLIT', 1)

class _anonenum106(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_OUTER_HEADERS = _anonenum106.define('MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_OUTER_HEADERS', 0)
MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS = _anonenum106.define('MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS', 1)
MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_INNER_HEADERS = _anonenum106.define('MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_INNER_HEADERS', 2)
MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_2 = _anonenum106.define('MLX5_CREATE_FLOW_GROUP_IN_MATCH_CRITERIA_ENABLE_MISC_PARAMETERS_2', 3)

@c.record
class struct_mlx5_ifc_create_flow_group_in_bits(c.Struct):
  SIZE = 8192
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 136]
  group_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 140]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  source_eswitch_owner_vhca_id_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 192]
  reserved_at_c1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 193]
  start_flow_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  end_flow_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 320]
  match_definer_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 336]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 352]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 480]
  match_criteria_enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 504]
  match_criteria: Annotated[struct_mlx5_ifc_fte_match_param_bits, 512]
  reserved_at_1200: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3584]], 4608]
@c.record
class struct_mlx5_ifc_create_eq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  eq_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_eq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  eq_context_entry: Annotated[struct_mlx5_ifc_eqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 640]
  event_bitmask: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[4]], 704]
  reserved_at_3c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1216]], 960]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_create_dct_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  dctn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  ece: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_dct_in_bits(c.Struct):
  SIZE = 1024
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  dct_context_entry: Annotated[struct_mlx5_ifc_dctc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 640]
@c.record
class struct_mlx5_ifc_create_cq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  cqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_create_cq_in_bits(c.Struct):
  SIZE = 2176
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  cq_context: Annotated[struct_mlx5_ifc_cqc_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 640]
  cq_umem_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 736]
  reserved_at_2e1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1439]], 737]
  pas: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 2176]
@c.record
class struct_mlx5_ifc_config_int_moderation_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  min_delay: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 68]
  int_vector: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
class _anonenum107(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CONFIG_INT_MODERATION_IN_OP_MOD_WRITE = _anonenum107.define('MLX5_CONFIG_INT_MODERATION_IN_OP_MOD_WRITE', 0)
MLX5_CONFIG_INT_MODERATION_IN_OP_MOD_READ = _anonenum107.define('MLX5_CONFIG_INT_MODERATION_IN_OP_MOD_READ', 1)

@c.record
class struct_mlx5_ifc_config_int_moderation_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  min_delay: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 68]
  int_vector: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_attach_to_mcg_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_attach_to_mcg_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  multicast_gid: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 128]
@c.record
class struct_mlx5_ifc_arm_xrq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_arm_xrq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  lwm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_arm_xrc_srq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class _anonenum108(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ARM_XRC_SRQ_IN_OP_MOD_XRC_SRQ = _anonenum108.define('MLX5_ARM_XRC_SRQ_IN_OP_MOD_XRC_SRQ', 1)

@c.record
class struct_mlx5_ifc_arm_xrc_srq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrc_srqn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  lwm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_arm_rq_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class _anonenum109(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ARM_RQ_IN_OP_MOD_SRQ = _anonenum109.define('MLX5_ARM_RQ_IN_OP_MOD_SRQ', 1)
MLX5_ARM_RQ_IN_OP_MOD_XRQ = _anonenum109.define('MLX5_ARM_RQ_IN_OP_MOD_XRQ', 2)

@c.record
class struct_mlx5_ifc_arm_rq_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  srq_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  lwm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_arm_dct_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_arm_dct_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  dct_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_xrcd_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  xrcd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_xrcd_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_alloc_uar_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  uar: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_uar_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_alloc_transport_domain_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  transport_domain: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_transport_domain_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_alloc_q_counter_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  counter_set_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_q_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_alloc_pd_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_pd_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_alloc_flow_counter_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  flow_counter_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_flow_counter_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[51]], 64]
  flow_counter_bulk_log_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 115]
  flow_counter_bulk: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
@c.record
class struct_mlx5_ifc_add_vxlan_udp_dport_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_add_vxlan_udp_dport_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  vxlan_udp_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_set_pp_rate_limit_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_pp_rate_limit_context_bits(c.Struct):
  SIZE = 384
  rate_limit: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  burst_upper_bound: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  typical_packet_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[288]], 96]
@c.record
class struct_mlx5_ifc_set_pp_rate_limit_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  rate_limit_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  ctx: Annotated[struct_mlx5_ifc_set_pp_rate_limit_context_bits, 128]
@c.record
class struct_mlx5_ifc_access_register_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  register_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[0]], 128]
class _anonenum110(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_ACCESS_REGISTER_IN_OP_MOD_WRITE = _anonenum110.define('MLX5_ACCESS_REGISTER_IN_OP_MOD_WRITE', 0)
MLX5_ACCESS_REGISTER_IN_OP_MOD_READ = _anonenum110.define('MLX5_ACCESS_REGISTER_IN_OP_MOD_READ', 1)

@c.record
class struct_mlx5_ifc_access_register_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  register_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  argument: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  register_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[0]], 128]
@c.record
class struct_mlx5_ifc_sltp_reg_bits(c.Struct):
  SIZE = 160
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  pnat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 16]
  reserved_at_12: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 18]
  lane: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 64]
  polarity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 71]
  ob_tap0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 72]
  ob_tap1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 80]
  ob_tap2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 96]
  ob_preemp_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 108]
  ob_reg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 112]
  ob_bias: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
@c.record
class struct_mlx5_ifc_slrg_reg_bits(c.Struct):
  SIZE = 320
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  pnat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 16]
  reserved_at_12: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 18]
  lane: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  time_to_link_up: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 48]
  grade_lane_speed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 60]
  grade_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  grade: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 96]
  height_grade_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 100]
  height_grade: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  height_dz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  height_dv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  height_sigma: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 224]
  phase_grade_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 228]
  phase_grade: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 232]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  phase_eo_pos: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 264]
  reserved_at_110: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 272]
  phase_eo_neg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 280]
  ffe_set_tested: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 288]
  test_errors_per_lane: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 304]
@c.record
class struct_mlx5_ifc_pvlc_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 32]
  vl_hw_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 60]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 64]
  vl_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 96]
  vl_operational: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 124]
@c.record
class struct_mlx5_ifc_pude_reg_bits(c.Struct):
  SIZE = 128
  swid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 16]
  admin_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 24]
  oper_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
class _anonenum111(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_PTYS_CONNECTOR_TYPE_PORT_DA = _anonenum111.define('MLX5_PTYS_CONNECTOR_TYPE_PORT_DA', 7)

@c.record
class struct_mlx5_ifc_ptys_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  an_disable_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  an_disable_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  reserved_at_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 3]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  plane_ind: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 24]
  reserved_at_1c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  proto_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 29]
  an_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 32]
  reserved_at_24: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 36]
  data_rate_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  ext_eth_proto_capability: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  eth_proto_capability: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  ib_link_width_capability: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  ib_proto_capability: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  ext_eth_proto_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  eth_proto_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  ib_link_width_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  ib_proto_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  ext_eth_proto_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  eth_proto_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  ib_link_width_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 320]
  ib_proto_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 336]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 352]
  lane_rate_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 360]
  connector_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 380]
  eth_proto_lp_advertise: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  reserved_at_1a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 416]
@c.record
class struct_mlx5_ifc_mlcr_reg_bits(c.Struct):
  SIZE = 96
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 16]
  beacon_duration: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  beacon_remain: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
@c.record
class struct_mlx5_ifc_ptas_reg_bits(c.Struct):
  SIZE = 352
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  algorithm_options: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 48]
  repetitions_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 52]
  num_of_repetitions: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  grade_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  height_grade_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 72]
  phase_grade_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  height_grade_weight: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 80]
  phase_grade_weight: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  gisim_measure_bits: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  adaptive_tap_measure_bits: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  ber_bath_high_error_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  ber_bath_mid_error_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  ber_bath_low_error_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  one_ratio_high_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  one_ratio_high_mid_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 192]
  one_ratio_low_mid_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 208]
  one_ratio_low_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  ndeo_error_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  mixer_offset_step_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 256]
  reserved_at_110: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 272]
  mix90_phase_for_voltage_bath: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 280]
  mixer_offset_start: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 288]
  mixer_offset_end: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 304]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[21]], 320]
  ber_test_time: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 341]
@c.record
class struct_mlx5_ifc_pspa_reg_bits(c.Struct):
  SIZE = 64
  swid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  sub_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_pqdr_reg_bits(c.Struct):
  SIZE = 256
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 16]
  prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 21]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 24]
  mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 30]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  min_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  max_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  mark_probability_denominator: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 160]
@c.record
class struct_mlx5_ifc_ppsc_reg_bits(c.Struct):
  SIZE = 384
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 128]
  wrps_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 156]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 160]
  wrps_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 188]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  up_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 200]
  reserved_at_d0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 208]
  down_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 216]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 256]
  srps_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 284]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 288]
  srps_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 316]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
@c.record
class struct_mlx5_ifc_pplr_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  lb_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  lb_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
@c.record
class struct_mlx5_ifc_pplm_reg_bits(c.Struct):
  SIZE = 960
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  port_profile_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  static_port_profile: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 72]
  active_port_profile: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 80]
  reserved_at_58: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  retransmission_active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  fec_mode_active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  rs_fec_correction_bypass_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 128]
  reserved_at_84: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 132]
  fec_override_cap_56g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 140]
  fec_override_cap_100g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 144]
  fec_override_cap_50g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 148]
  fec_override_cap_25g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 152]
  fec_override_cap_10g_40g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 156]
  rs_fec_correction_bypass_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 160]
  reserved_at_a4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 164]
  fec_override_admin_56g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 172]
  fec_override_admin_100g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 176]
  fec_override_admin_50g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 180]
  fec_override_admin_25g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 184]
  fec_override_admin_10g_40g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 188]
  fec_override_cap_400g_8x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 192]
  fec_override_cap_200g_4x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 208]
  fec_override_cap_100g_2x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 224]
  fec_override_cap_50g_1x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  fec_override_admin_400g_8x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 256]
  fec_override_admin_200g_4x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 272]
  fec_override_admin_100g_2x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 288]
  fec_override_admin_50g_1x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 304]
  fec_override_cap_800g_8x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 320]
  fec_override_cap_400g_4x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 336]
  fec_override_cap_200g_2x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 352]
  fec_override_cap_100g_1x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 368]
  reserved_at_180: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 384]
  fec_override_admin_800g_8x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 544]
  fec_override_admin_400g_4x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 560]
  fec_override_admin_200g_2x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 576]
  fec_override_admin_100g_1x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 592]
  reserved_at_260: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 608]
  fec_override_cap_1600g_8x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 704]
  fec_override_cap_800g_4x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 720]
  fec_override_cap_400g_2x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 736]
  fec_override_cap_200g_1x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 752]
  fec_override_admin_1600g_8x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 768]
  fec_override_admin_800g_4x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 784]
  fec_override_admin_400g_2x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 800]
  fec_override_admin_200g_1x: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 816]
  reserved_at_340: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 832]
@c.record
class struct_mlx5_ifc_ppcnt_reg_bits(c.Struct):
  SIZE = 2048
  swid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  pnat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 16]
  reserved_at_12: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 18]
  grp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 26]
  clr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[19]], 33]
  plane_ind: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 52]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 56]
  prio_tc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 59]
  counter_set: Annotated[union_mlx5_ifc_eth_cntrs_grp_data_layout_auto_bits, 64]
@c.record
class struct_mlx5_ifc_mpein_reg_bits(c.Struct):
  SIZE = 384
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 0]
  depth: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 2]
  pcie_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  node: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  capability_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  link_width_enabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 72]
  link_speed_enabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  lane0_physical_position: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  link_width_active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 104]
  link_speed_active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  num_of_pfs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  num_of_vfs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  bdf0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  reserved_at_b0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  max_read_request_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 192]
  max_payload_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 196]
  reserved_at_c8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 200]
  pwr_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 205]
  port_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 208]
  reserved_at_d4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 212]
  lane_reversal: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 223]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 224]
  pci_power: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 244]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  device_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 288]
  port_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 304]
  reserved_at_138: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 312]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 320]
  receiver_detect_result: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 336]
  reserved_at_160: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
@c.record
class struct_mlx5_ifc_mpcnt_reg_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  pcie_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 16]
  grp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 26]
  clr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 33]
  counter_set: Annotated[union_mlx5_ifc_pcie_cntrs_grp_data_layout_auto_bits, 64]
@c.record
class struct_mlx5_ifc_ppad_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 0]
  single_mac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  mac_47_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  mac_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_pmtu_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  max_mtu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  admin_mtu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  oper_mtu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_pmpr_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  module: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 32]
  attenuation_5g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  attenuation_7g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 96]
  attenuation_12g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
@c.record
class struct_mlx5_ifc_pmpe_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  module: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 16]
  module_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
@c.record
class struct_mlx5_ifc_pmpc_reg_bits(c.Struct):
  SIZE = 256
  module_state_updated: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[32]], 0]
@c.record
class struct_mlx5_ifc_pmlpn_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  mlpn_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 33]
@c.record
class struct_mlx5_ifc_pmlp_reg_bits(c.Struct):
  SIZE = 512
  rxtx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 1]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  width: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  lane0_module_mapping: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  lane1_module_mapping: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  lane2_module_mapping: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  lane3_module_mapping: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[352]], 160]
@c.record
class struct_mlx5_ifc_pmaos_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  module: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 16]
  admin_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 24]
  oper_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  ase: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  ee: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 34]
  e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 62]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_plpc_reg_bits(c.Struct):
  SIZE = 320
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  profile_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 4]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 16]
  proto_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  lane_speed: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[23]], 64]
  lpbf: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 87]
  fec_mode_policy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  retransmission_capability: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  fec_mode_capability: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  retransmission_support_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  fec_mode_support_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  retransmission_request_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  fec_mode_request_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 192]
@c.record
class struct_mlx5_ifc_plib_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  ib_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
@c.record
class struct_mlx5_ifc_plbf_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[13]], 16]
  lbf_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 29]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_pipg_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  dic: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[25]], 33]
  ipg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 58]
  reserved_at_3e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 62]
@c.record
class struct_mlx5_ifc_pifr_reg_bits(c.Struct):
  SIZE = 768
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 32]
  port_filter: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 256]
  port_filter_update_en: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 512]
class _anonenum112(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_BUF_OWNERSHIP_UNKNOWN = _anonenum112.define('MLX5_BUF_OWNERSHIP_UNKNOWN', 0)
MLX5_BUF_OWNERSHIP_FW_OWNED = _anonenum112.define('MLX5_BUF_OWNERSHIP_FW_OWNED', 1)
MLX5_BUF_OWNERSHIP_SW_OWNED = _anonenum112.define('MLX5_BUF_OWNERSHIP_SW_OWNED', 2)

@c.record
class struct_mlx5_ifc_pfcc_reg_bits(c.Struct):
  SIZE = 256
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  buf_ownership: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 4]
  reserved_at_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 6]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 16]
  cable_length_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 26]
  ppan_mask_n: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 27]
  minor_stall_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  critical_stall_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  reserved_at_1e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 30]
  ppan: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 32]
  reserved_at_24: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 36]
  prio_mask_tx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  prio_mask_rx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  pptx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  aptx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  pptx_mask_n: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 66]
  reserved_at_43: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 67]
  pfctx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 72]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  pprx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  aprx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 97]
  pprx_mask_n: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 98]
  reserved_at_63: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 99]
  pfcrx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 104]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  device_stall_minor_watermark: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  device_stall_critical_watermark: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 160]
  cable_length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 184]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
@c.record
class struct_mlx5_ifc_pelc_reg_bits(c.Struct):
  SIZE = 448
  op: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  op_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  op_capability: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  op_request: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  op_active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  capability: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  request: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  active: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 256]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 320]
@c.record
class struct_mlx5_ifc_peir_reg_bits(c.Struct):
  SIZE = 96
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 32]
  error_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 44]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 64]
  lane: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 80]
  error_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
@c.record
class struct_mlx5_ifc_mpegc_reg_bits(c.Struct):
  SIZE = 352
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[48]], 0]
  field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  tx_overflow_sense: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  mark_cqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  mark_cnp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 66]
  reserved_at_43: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[27]], 67]
  tx_lossy_overflow_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 94]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 96]
@c.record
class struct_mlx5_ifc_mpir_reg_bits(c.Struct):
  SIZE = 128
  sdm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[27]], 1]
  host_buses: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  reserved_at_28: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
class _anonenum113(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MTUTC_FREQ_ADJ_UNITS_PPB = _anonenum113.define('MLX5_MTUTC_FREQ_ADJ_UNITS_PPB', 0)
MLX5_MTUTC_FREQ_ADJ_UNITS_SCALED_PPM = _anonenum113.define('MLX5_MTUTC_FREQ_ADJ_UNITS_SCALED_PPM', 1)

class _anonenum114(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MTUTC_OPERATION_SET_TIME_IMMEDIATE = _anonenum114.define('MLX5_MTUTC_OPERATION_SET_TIME_IMMEDIATE', 1)
MLX5_MTUTC_OPERATION_ADJUST_TIME = _anonenum114.define('MLX5_MTUTC_OPERATION_ADJUST_TIME', 2)
MLX5_MTUTC_OPERATION_ADJUST_FREQ_UTC = _anonenum114.define('MLX5_MTUTC_OPERATION_ADJUST_FREQ_UTC', 3)

@c.record
class struct_mlx5_ifc_mtutc_reg_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 0]
  freq_adj_units: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 5]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 8]
  log_max_freq_adjustment: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 11]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 16]
  operation: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  freq_adjustment: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  utc_sec: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 160]
  utc_nsec: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 162]
  time_adjustment: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
@c.record
class struct_mlx5_ifc_pcam_enhanced_features_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  ppcnt_recovery_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  reserved_at_11: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 17]
  cable_length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 24]
  reserved_at_19: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 25]
  fec_200G_per_lane_in_pplm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  reserved_at_1e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[42]], 30]
  fec_100G_per_lane_in_pplm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 72]
  reserved_at_49: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 73]
  buffer_ownership: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 83]
  resereved_at_54: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 84]
  fec_50G_per_lane_in_pplm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 104]
  reserved_at_69: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 105]
  rx_icrc_encapsulated_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 109]
  reserved_at_6e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 110]
  ptys_extended_ethernet: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 114]
  reserved_at_73: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 115]
  pfcc_mask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 118]
  reserved_at_77: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 119]
  per_lane_error_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 122]
  rx_buffer_fullness_counters: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 123]
  ptys_connector_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 124]
  reserved_at_7d: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 125]
  ppcnt_discard_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 126]
  ppcnt_statistical_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 127]
@c.record
class struct_mlx5_ifc_pcam_regs_5000_to_507f_bits(c.Struct):
  SIZE = 128
  port_access_reg_cap_mask_127_to_96: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  port_access_reg_cap_mask_95_to_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  port_access_reg_cap_mask_63: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  pphcr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  port_access_reg_cap_mask_61_to_36: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 66]
  pplm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 92]
  port_access_reg_cap_mask_34_to_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 93]
  port_access_reg_cap_mask_31_to_13: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[19]], 96]
  pbmc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 115]
  pptb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 116]
  port_access_reg_cap_mask_10_to_09: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 117]
  ppcnt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 119]
  port_access_reg_cap_mask_07_to_00: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
@c.record
class struct_mlx5_ifc_pcam_reg_bits(c.Struct):
  SIZE = 640
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  feature_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  access_reg_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  port_access_reg_cap_mask: Annotated[struct_mlx5_ifc_pcam_reg_bits_port_access_reg_cap_mask, 64]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 192]
  feature_cap_mask: Annotated[struct_mlx5_ifc_pcam_reg_bits_feature_cap_mask, 320]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[192]], 448]
@c.record
class struct_mlx5_ifc_pcam_reg_bits_port_access_reg_cap_mask(c.Struct):
  SIZE = 128
  regs_5000_to_507f: Annotated[struct_mlx5_ifc_pcam_regs_5000_to_507f_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
@c.record
class struct_mlx5_ifc_pcam_reg_bits_feature_cap_mask(c.Struct):
  SIZE = 128
  enhanced_features: Annotated[struct_mlx5_ifc_pcam_enhanced_features_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
@c.record
class struct_mlx5_ifc_mcam_enhanced_features_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[80]], 0]
  mtutc_freq_adj_units: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 80]
  mtutc_time_adjustment_extended_range: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 81]
  reserved_at_52: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 82]
  mcia_32dwords: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 93]
  out_pulse_duration_ns: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 94]
  npps_period: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 95]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 96]
  reset_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 106]
  ptpcyc2realtime_modify: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 107]
  reserved_at_6c: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 108]
  pci_status_and_power: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 110]
  reserved_at_6f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 111]
  mark_tx_action_cnp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 116]
  mark_tx_action_cqe: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 117]
  dynamic_tx_overflow: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 118]
  reserved_at_77: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 119]
  pcie_outbound_stalled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 123]
  tx_overflow_buffer_pkt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 124]
  mtpps_enh_out_per_adj: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 125]
  mtpps_fs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 126]
  pcie_performance_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 127]
@c.record
class struct_mlx5_ifc_mcam_access_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 0]
  mcda: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 28]
  mcc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  mcqi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 30]
  mcqs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 31]
  regs_95_to_90: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 32]
  mpir: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 38]
  regs_88_to_87: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 39]
  mpegc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 41]
  mtutc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 42]
  regs_84_to_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[17]], 43]
  tracer_registers: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 60]
  regs_63_to_46: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[18]], 64]
  mrtc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 82]
  regs_44_to_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 83]
  mfrl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 87]
  regs_39_to_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  regs_31_to_11: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[21]], 96]
  mtmp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 117]
  regs_9_to_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 118]
@c.record
class struct_mlx5_ifc_mcam_access_reg_bits1(c.Struct):
  SIZE = 128
  regs_127_to_96: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  regs_95_to_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  regs_63_to_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  regs_31_to_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_mcam_access_reg_bits2(c.Struct):
  SIZE = 128
  regs_127_to_99: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[29]], 0]
  mirc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 29]
  regs_97_to_96: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 30]
  regs_95_to_87: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 32]
  synce_registers: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 41]
  regs_84_to_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[21]], 43]
  regs_63_to_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  regs_31_to_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_mcam_access_reg_bits3(c.Struct):
  SIZE = 128
  regs_127_to_96: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  regs_95_to_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  regs_63_to_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  regs_31_to_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[29]], 96]
  mrtcq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 125]
  mtctr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 126]
  mtptm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 127]
@c.record
class struct_mlx5_ifc_mcam_reg_bits(c.Struct):
  SIZE = 576
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  feature_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  access_reg_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  mng_access_reg_cap_mask: Annotated[struct_mlx5_ifc_mcam_reg_bits_mng_access_reg_cap_mask, 64]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 192]
  mng_feature_cap_mask: Annotated[struct_mlx5_ifc_mcam_reg_bits_mng_feature_cap_mask, 320]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 448]
@c.record
class struct_mlx5_ifc_mcam_reg_bits_mng_access_reg_cap_mask(c.Struct):
  SIZE = 128
  access_regs: Annotated[struct_mlx5_ifc_mcam_access_reg_bits, 0]
  access_regs1: Annotated[struct_mlx5_ifc_mcam_access_reg_bits1, 0]
  access_regs2: Annotated[struct_mlx5_ifc_mcam_access_reg_bits2, 0]
  access_regs3: Annotated[struct_mlx5_ifc_mcam_access_reg_bits3, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
@c.record
class struct_mlx5_ifc_mcam_reg_bits_mng_feature_cap_mask(c.Struct):
  SIZE = 128
  enhanced_features: Annotated[struct_mlx5_ifc_mcam_enhanced_features_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
@c.record
class struct_mlx5_ifc_qcam_access_reg_cap_mask(c.Struct):
  SIZE = 128
  qcam_access_reg_cap_mask_127_to_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[108]], 0]
  qpdpm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 108]
  qcam_access_reg_cap_mask_18_to_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 109]
  qdpm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 124]
  qpts: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 125]
  qcap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 126]
  qcam_access_reg_cap_mask_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 127]
@c.record
class struct_mlx5_ifc_qcam_qos_feature_cap_mask(c.Struct):
  SIZE = 128
  qcam_qos_feature_cap_mask_127_to_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[123]], 0]
  qetcr_qshr_max_bw_val_msb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 123]
  qcam_qos_feature_cap_mask_3_to_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 124]
  qpts_trust_both: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 127]
@c.record
class struct_mlx5_ifc_qcam_reg_bits(c.Struct):
  SIZE = 576
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  feature_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  access_reg_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  qos_access_reg_cap_mask: Annotated[struct_mlx5_ifc_qcam_reg_bits_qos_access_reg_cap_mask, 64]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 192]
  qos_feature_cap_mask: Annotated[struct_mlx5_ifc_qcam_reg_bits_qos_feature_cap_mask, 320]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 448]
@c.record
class struct_mlx5_ifc_qcam_reg_bits_qos_access_reg_cap_mask(c.Struct):
  SIZE = 128
  reg_cap: Annotated[struct_mlx5_ifc_qcam_access_reg_cap_mask, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
@c.record
class struct_mlx5_ifc_qcam_reg_bits_qos_feature_cap_mask(c.Struct):
  SIZE = 128
  feature_cap: Annotated[struct_mlx5_ifc_qcam_qos_feature_cap_mask, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
@c.record
class struct_mlx5_ifc_core_dump_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 0]
  core_dump_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[48]], 32]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
@c.record
class struct_mlx5_ifc_pcap_reg_bits(c.Struct):
  SIZE = 160
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  port_capability_mask: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[4]], 32]
@c.record
class struct_mlx5_ifc_paos_reg_bits(c.Struct):
  SIZE = 128
  swid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 16]
  admin_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 24]
  oper_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  ase: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  ee: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 34]
  e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 62]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_pamp_reg_bits(c.Struct):
  SIZE = 352
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  opamp_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 16]
  opamp_group_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  start_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 48]
  num_of_indices: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 52]
  index_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], Literal[18]], 64]
@c.record
class struct_mlx5_ifc_pcmr_reg_bits(c.Struct):
  SIZE = 96
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  entropy_force_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  entropy_calc_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  entropy_gre_calc_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  reserved_at_23: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 35]
  rx_ts_over_crc_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 50]
  reserved_at_33: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 51]
  fcs_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 62]
  reserved_at_3f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
  entropy_force: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  entropy_calc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  entropy_gre_calc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 66]
  reserved_at_43: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 67]
  rx_ts_over_crc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 82]
  reserved_at_53: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 83]
  fcs_chk: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 94]
  reserved_at_5f: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 95]
@c.record
class struct_mlx5_ifc_lane_2_module_mapping_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  rx_lane: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 8]
  tx_lane: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  module: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
@c.record
class struct_mlx5_ifc_bufferx_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 0]
  lossy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  epsb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  xoff_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  xon_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
@c.record
class struct_mlx5_ifc_set_node_in_bits(c.Struct):
  SIZE = 512
  node_description: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[64]], 0]
@c.record
class struct_mlx5_ifc_register_power_settings_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 0]
  power_settings_level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
@c.record
class struct_mlx5_ifc_register_host_endianness_bits(c.Struct):
  SIZE = 128
  he: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 1]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
@c.record
class struct_mlx5_ifc_umr_pointer_desc_argument_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  addressh_63_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  addressl_31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_ud_adrs_vector_bits(c.Struct):
  SIZE = 384
  dc_key: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  ext: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 65]
  destination_qp_dct: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  static_rate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 96]
  sl_eth_prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 100]
  fl: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 104]
  mlid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 105]
  rlid_udp_sport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  rmac_47_16: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  rmac_15_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 192]
  tclass: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 208]
  hop_limit: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 216]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 224]
  grh: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 225]
  reserved_at_e2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 226]
  src_addr_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 228]
  flow_label: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 236]
  rgid_rip: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 256]
@c.record
class struct_mlx5_ifc_pages_req_event_bits(c.Struct):
  SIZE = 224
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  num_pages: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 64]
@c.record
class struct_mlx5_ifc_eqe_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  event_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  event_sub_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 32]
  event_data: Annotated[union_mlx5_ifc_event_auto_bits, 256]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 480]
  signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 496]
  reserved_at_1f8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 504]
  owner: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 511]
class _anonenum115(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_CMD_QUEUE_ENTRY_TYPE_PCIE_CMD_IF_TRANSPORT = _anonenum115.define('MLX5_CMD_QUEUE_ENTRY_TYPE_PCIE_CMD_IF_TRANSPORT', 7)

@c.record
class struct_mlx5_ifc_cmd_queue_entry_bits(c.Struct):
  SIZE = 512
  type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  input_length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  input_mailbox_pointer_63_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  input_mailbox_pointer_31_9: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[23]], 96]
  reserved_at_77: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 119]
  command_input_inline_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 128]
  command_output_inline_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[16]], 256]
  output_mailbox_pointer_63_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  output_mailbox_pointer_31_9: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[23]], 416]
  reserved_at_1b7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 439]
  output_length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  token: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 480]
  signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 488]
  reserved_at_1f0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 496]
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 504]
  ownership: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 511]
@c.record
class struct_mlx5_ifc_cmd_out_bits(c.Struct):
  SIZE = 96
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  command_output: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
@c.record
class struct_mlx5_ifc_cmd_in_bits(c.Struct):
  SIZE = 64
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  command: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[0]], 64]
@c.record
class struct_mlx5_ifc_cmd_if_box_bits(c.Struct):
  SIZE = 4608
  mailbox_data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[512]], 0]
  reserved_at_1000: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 4096]
  next_pointer_63_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 4480]
  next_pointer_31_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[22]], 4512]
  reserved_at_11b6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 4534]
  block_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 4544]
  reserved_at_11e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 4576]
  token: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 4584]
  ctrl_signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 4592]
  signature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 4600]
@c.record
class struct_mlx5_ifc_mtt_bits(c.Struct):
  SIZE = 64
  ptag_63_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  ptag_31_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 32]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 56]
  wr_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 62]
  rd_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
@c.record
class struct_mlx5_ifc_query_wol_rol_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  rol_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 80]
  wol_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_wol_rol_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_wol_rol_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_wol_rol_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  rol_mode_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  wol_mode_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  rol_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 80]
  wol_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
class _anonenum116(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_INITIAL_SEG_NIC_INTERFACE_FULL_DRIVER = _anonenum116.define('MLX5_INITIAL_SEG_NIC_INTERFACE_FULL_DRIVER', 0)
MLX5_INITIAL_SEG_NIC_INTERFACE_DISABLED = _anonenum116.define('MLX5_INITIAL_SEG_NIC_INTERFACE_DISABLED', 1)
MLX5_INITIAL_SEG_NIC_INTERFACE_NO_DRAM_NIC = _anonenum116.define('MLX5_INITIAL_SEG_NIC_INTERFACE_NO_DRAM_NIC', 2)
MLX5_INITIAL_SEG_NIC_INTERFACE_SW_RESET = _anonenum116.define('MLX5_INITIAL_SEG_NIC_INTERFACE_SW_RESET', 7)

class _anonenum117(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_FULL_DRIVER = _anonenum117.define('MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_FULL_DRIVER', 0)
MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_DISABLED = _anonenum117.define('MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_DISABLED', 1)
MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_NO_DRAM_NIC = _anonenum117.define('MLX5_INITIAL_SEG_NIC_INTERFACE_SUPPORTED_NO_DRAM_NIC', 2)

class _anonenum118(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_INITIAL_SEG_HEALTH_SYNDROME_FW_INTERNAL_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_FW_INTERNAL_ERR', 1)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_DEAD_IRISC = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_DEAD_IRISC', 7)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_HW_FATAL_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_HW_FATAL_ERR', 8)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_FW_CRC_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_FW_CRC_ERR', 9)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_FETCH_PCI_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_FETCH_PCI_ERR', 10)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_PAGE_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_PAGE_ERR', 11)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_ASYNCHRONOUS_EQ_BUF_OVERRUN = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_ASYNCHRONOUS_EQ_BUF_OVERRUN', 12)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_EQ_IN_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_EQ_IN_ERR', 13)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_EQ_INV = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_EQ_INV', 14)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_FFSER_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_FFSER_ERR', 15)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_HIGH_TEMP_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_HIGH_TEMP_ERR', 16)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_PCI_POISONED_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_ICM_PCI_POISONED_ERR', 18)
MLX5_INITIAL_SEG_HEALTH_SYNDROME_TRUST_LOCKDOWN_ERR = _anonenum118.define('MLX5_INITIAL_SEG_HEALTH_SYNDROME_TRUST_LOCKDOWN_ERR', 19)

@c.record
class struct_mlx5_ifc_initial_seg_bits(c.Struct):
  SIZE = 131168
  fw_rev_minor: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  fw_rev_major: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  cmd_interface_rev: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  fw_rev_subminor: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  cmdq_phy_addr_63_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  cmdq_phy_addr_31_12: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 160]
  reserved_at_b4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 180]
  nic_interface: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 182]
  log_cmdq_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 184]
  log_cmdq_stride: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 188]
  command_doorbell_vector: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3840]], 224]
  initializing: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4064]
  reserved_at_fe1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4065]
  nic_interface_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 4069]
  embedded_cpu: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4072]
  reserved_at_fe9: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[23]], 4073]
  health_buffer: Annotated[struct_mlx5_ifc_health_buffer_bits, 4096]
  no_dram_nic_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 4608]
  reserved_at_1220: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28224]], 4640]
  reserved_at_8060: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 32864]
  clear_int: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32895]
  health_syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32896]
  health_counter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 32904]
  reserved_at_80a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[98240]], 32928]
@c.record
class struct_mlx5_ifc_mtpps_reg_bits(c.Struct):
  SIZE = 480
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 0]
  cap_number_of_pps_pins: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 16]
  cap_max_num_of_pps_in_pins: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 20]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 24]
  cap_max_num_of_pps_out_pins: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[19]], 32]
  cap_log_min_npps_period: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 51]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 56]
  cap_log_min_out_pulse_duration_ns: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 59]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  cap_pin_3_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 68]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 72]
  cap_pin_2_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 80]
  cap_pin_1_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 84]
  reserved_at_58: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 88]
  cap_pin_0_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 96]
  cap_pin_7_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 100]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 104]
  cap_pin_6_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 108]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 112]
  cap_pin_5_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 116]
  reserved_at_78: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 120]
  cap_pin_4_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 124]
  field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  npps_period: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 256]
  reserved_at_101: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 257]
  pattern: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 268]
  reserved_at_110: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 272]
  pin_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 276]
  pin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 280]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 288]
  out_pulse_duration_ns: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 290]
  time_stamp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
  out_pulse_duration: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 384]
  out_periodic_adjustment: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 400]
  enhanced_out_periodic_adjustment: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
@c.record
class struct_mlx5_ifc_mtppse_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 0]
  pin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  event_arm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[27]], 33]
  event_generation_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 60]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_mcqs_reg_bits(c.Struct):
  SIZE = 128
  last_index_flag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 1]
  fw_device: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  component_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  identifier: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[23]], 64]
  component_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 87]
  component_update_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  last_update_state_changer_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 96]
  last_update_state_changer_host_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 100]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
@c.record
class struct_mlx5_ifc_mcqi_cap_bits(c.Struct):
  SIZE = 160
  supported_info_bitmask: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  component_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  max_component_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  log_mcda_word_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 96]
  reserved_at_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 100]
  mcda_max_write_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  rd_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 128]
  reserved_at_81: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 129]
  match_chip_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 130]
  match_psid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 131]
  check_user_timestamp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 132]
  match_base_guid_mac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 133]
  reserved_at_86: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 134]
@c.record
class struct_mlx5_ifc_mcqi_version_bits(c.Struct):
  SIZE = 992
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 0]
  build_time_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  user_defined_time_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 4]
  version_string_length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  build_time: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  user_defined_time: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  build_tool_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  version_string: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[92]], 256]
@c.record
class struct_mlx5_ifc_mcqi_activation_method_bits(c.Struct):
  SIZE = 32
  pending_server_ac_power_cycle: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  pending_server_dc_power_cycle: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  pending_server_reboot: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  pending_fw_reset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 3]
  auto_activate: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  all_hosts_sync: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  device_hw_reset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  reserved_at_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[25]], 7]
@c.record
class union_mlx5_ifc_mcqi_reg_data_bits(c.Struct):
  SIZE = 992
  mcqi_caps: Annotated[struct_mlx5_ifc_mcqi_cap_bits, 0]
  mcqi_version: Annotated[struct_mlx5_ifc_mcqi_version_bits, 0]
  mcqi_activation_mathod: Annotated[struct_mlx5_ifc_mcqi_activation_method_bits, 0]
@c.record
class struct_mlx5_ifc_mcqi_reg_bits(c.Struct):
  SIZE = 192
  read_pending_component: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 1]
  component_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[27]], 64]
  info_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 91]
  info_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  data_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  data: Annotated[c.Array[union_mlx5_ifc_mcqi_reg_data_bits, Literal[0]], 192]
@c.record
class struct_mlx5_ifc_mcc_reg_bits(c.Struct):
  SIZE = 256
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  time_elapsed_since_last_cmd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 4]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  instruction: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  component_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  update_handle: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  handle_owner_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 96]
  handle_owner_host_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 100]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 104]
  control_progress: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 105]
  error_code: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 112]
  reserved_at_78: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 120]
  control_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 124]
  component_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 160]
@c.record
class struct_mlx5_ifc_mcda_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  update_handle: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[0]], 128]
class _anonenum119(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MFRL_REG_PCI_RESET_METHOD_LINK_TOGGLE = _anonenum119.define('MLX5_MFRL_REG_PCI_RESET_METHOD_LINK_TOGGLE', 0)
MLX5_MFRL_REG_PCI_RESET_METHOD_HOT_RESET = _anonenum119.define('MLX5_MFRL_REG_PCI_RESET_METHOD_HOT_RESET', 1)

class _anonenum120(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MFRL_REG_RESET_STATE_IDLE = _anonenum120.define('MLX5_MFRL_REG_RESET_STATE_IDLE', 0)
MLX5_MFRL_REG_RESET_STATE_IN_NEGOTIATION = _anonenum120.define('MLX5_MFRL_REG_RESET_STATE_IN_NEGOTIATION', 1)
MLX5_MFRL_REG_RESET_STATE_RESET_IN_PROGRESS = _anonenum120.define('MLX5_MFRL_REG_RESET_STATE_RESET_IN_PROGRESS', 2)
MLX5_MFRL_REG_RESET_STATE_NEG_TIMEOUT = _anonenum120.define('MLX5_MFRL_REG_RESET_STATE_NEG_TIMEOUT', 3)
MLX5_MFRL_REG_RESET_STATE_NACK = _anonenum120.define('MLX5_MFRL_REG_RESET_STATE_NACK', 4)
MLX5_MFRL_REG_RESET_STATE_UNLOAD_TIMEOUT = _anonenum120.define('MLX5_MFRL_REG_RESET_STATE_UNLOAD_TIMEOUT', 5)

class _anonenum121(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MFRL_REG_RESET_TYPE_FULL_CHIP = _anonenum121.define('MLX5_MFRL_REG_RESET_TYPE_FULL_CHIP', 0)
MLX5_MFRL_REG_RESET_TYPE_NET_PORT_ALIVE = _anonenum121.define('MLX5_MFRL_REG_RESET_TYPE_NET_PORT_ALIVE', 1)

class _anonenum122(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MFRL_REG_RESET_LEVEL0 = _anonenum122.define('MLX5_MFRL_REG_RESET_LEVEL0', 0)
MLX5_MFRL_REG_RESET_LEVEL3 = _anonenum122.define('MLX5_MFRL_REG_RESET_LEVEL3', 1)
MLX5_MFRL_REG_RESET_LEVEL6 = _anonenum122.define('MLX5_MFRL_REG_RESET_LEVEL6', 2)

@c.record
class struct_mlx5_ifc_mfrl_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 32]
  pci_sync_for_fw_update_start: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 34]
  pci_sync_for_fw_update_resp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 35]
  rst_type_sel: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 37]
  pci_reset_req_method: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 40]
  reserved_at_2b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 43]
  reset_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 44]
  reset_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  reset_level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
@c.record
class struct_mlx5_ifc_mirc_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 0]
  status_code: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
@c.record
class struct_mlx5_ifc_pddr_monitor_opcode_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  monitor_opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
@c.record
class union_mlx5_ifc_pddr_troubleshooting_page_status_opcode_auto_bits(c.Struct):
  SIZE = 32
  pddr_monitor_opcode: Annotated[struct_mlx5_ifc_pddr_monitor_opcode_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
class _anonenum123(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_PDDR_REG_TRBLSH_GROUP_OPCODE_MONITOR = _anonenum123.define('MLX5_PDDR_REG_TRBLSH_GROUP_OPCODE_MONITOR', 0)

@c.record
class struct_mlx5_ifc_pddr_troubleshooting_page_bits(c.Struct):
  SIZE = 1984
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  group_opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  status_opcode: Annotated[union_mlx5_ifc_pddr_troubleshooting_page_status_opcode_auto_bits, 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  status_message: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[59]], 96]
@c.record
class union_mlx5_ifc_pddr_reg_page_data_auto_bits(c.Struct):
  SIZE = 1984
  pddr_troubleshooting_page: Annotated[struct_mlx5_ifc_pddr_troubleshooting_page_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1984]], 0]
class _anonenum124(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_PDDR_REG_PAGE_SELECT_TROUBLESHOOTING_INFO_PAGE = _anonenum124.define('MLX5_PDDR_REG_PAGE_SELECT_TROUBLESHOOTING_INFO_PAGE', 1)

@c.record
class struct_mlx5_ifc_pddr_reg_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  pnat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 16]
  reserved_at_12: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 18]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 32]
  page_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  page_data: Annotated[union_mlx5_ifc_pddr_reg_page_data_auto_bits, 64]
@c.record
class struct_mlx5_ifc_mrtc_reg_bits(c.Struct):
  SIZE = 128
  time_synced: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 1]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  time_h: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  time_l: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_mtcap_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[25]], 0]
  sensor_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 25]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  sensor_map: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_mtmp_reg_bits(c.Struct):
  SIZE = 256
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 0]
  sensor_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 20]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  temperature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  mte: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  mtr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  max_temperature: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  tee: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 96]
  reserved_at_62: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 98]
  temp_threshold_hi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  temp_threshold_lo: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  sensor_name_hi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  sensor_name_lo: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_mtptm_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  psta: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 16]
  reserved_at_11: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 17]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
class _anonenum125(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MTCTR_REQUEST_NOP = _anonenum125.define('MLX5_MTCTR_REQUEST_NOP', 0)
MLX5_MTCTR_REQUEST_PTM_ROOT_CLOCK = _anonenum125.define('MLX5_MTCTR_REQUEST_PTM_ROOT_CLOCK', 1)
MLX5_MTCTR_REQUEST_FREE_RUNNING_COUNTER = _anonenum125.define('MLX5_MTCTR_REQUEST_FREE_RUNNING_COUNTER', 2)
MLX5_MTCTR_REQUEST_REAL_TIME_CLOCK = _anonenum125.define('MLX5_MTCTR_REQUEST_REAL_TIME_CLOCK', 3)

@c.record
class struct_mlx5_ifc_mtctr_reg_bits(c.Struct):
  SIZE = 192
  first_clock_timestamp_request: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  second_clock_timestamp_request: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  first_clock_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  second_clock_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 33]
  reserved_at_22: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 34]
  first_clock_timestamp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  second_clock_timestamp: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
@c.record
class struct_mlx5_ifc_bin_range_layout_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 0]
  high_val: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 10]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 16]
  low_val: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 26]
@c.record
class struct_mlx5_ifc_pphcr_reg_bits(c.Struct):
  SIZE = 640
  active_hist_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  num_of_bins: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  bin_range: Annotated[c.Array[struct_mlx5_ifc_bin_range_layout_bits, Literal[16]], 128]
@c.record
class union_mlx5_ifc_ports_control_registers_document_bits(c.Struct):
  SIZE = 24800
  bufferx_reg: Annotated[struct_mlx5_ifc_bufferx_reg_bits, 0]
  eth_2819_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_2819_cntrs_grp_data_layout_bits, 0]
  eth_2863_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_2863_cntrs_grp_data_layout_bits, 0]
  eth_3635_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_3635_cntrs_grp_data_layout_bits, 0]
  eth_802_3_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_802_3_cntrs_grp_data_layout_bits, 0]
  eth_extended_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_eth_extended_cntrs_grp_data_layout_bits, 0]
  eth_per_prio_grp_data_layout: Annotated[struct_mlx5_ifc_eth_per_prio_grp_data_layout_bits, 0]
  eth_per_tc_prio_grp_data_layout: Annotated[struct_mlx5_ifc_eth_per_tc_prio_grp_data_layout_bits, 0]
  eth_per_tc_congest_prio_grp_data_layout: Annotated[struct_mlx5_ifc_eth_per_tc_congest_prio_grp_data_layout_bits, 0]
  lane_2_module_mapping: Annotated[struct_mlx5_ifc_lane_2_module_mapping_bits, 0]
  pamp_reg: Annotated[struct_mlx5_ifc_pamp_reg_bits, 0]
  paos_reg: Annotated[struct_mlx5_ifc_paos_reg_bits, 0]
  pcap_reg: Annotated[struct_mlx5_ifc_pcap_reg_bits, 0]
  pddr_monitor_opcode: Annotated[struct_mlx5_ifc_pddr_monitor_opcode_bits, 0]
  pddr_reg: Annotated[struct_mlx5_ifc_pddr_reg_bits, 0]
  pddr_troubleshooting_page: Annotated[struct_mlx5_ifc_pddr_troubleshooting_page_bits, 0]
  peir_reg: Annotated[struct_mlx5_ifc_peir_reg_bits, 0]
  pelc_reg: Annotated[struct_mlx5_ifc_pelc_reg_bits, 0]
  pfcc_reg: Annotated[struct_mlx5_ifc_pfcc_reg_bits, 0]
  ib_port_cntrs_grp_data_layout: Annotated[struct_mlx5_ifc_ib_port_cntrs_grp_data_layout_bits, 0]
  phys_layer_cntrs: Annotated[struct_mlx5_ifc_phys_layer_cntrs_bits, 0]
  pifr_reg: Annotated[struct_mlx5_ifc_pifr_reg_bits, 0]
  pipg_reg: Annotated[struct_mlx5_ifc_pipg_reg_bits, 0]
  plbf_reg: Annotated[struct_mlx5_ifc_plbf_reg_bits, 0]
  plib_reg: Annotated[struct_mlx5_ifc_plib_reg_bits, 0]
  plpc_reg: Annotated[struct_mlx5_ifc_plpc_reg_bits, 0]
  pmaos_reg: Annotated[struct_mlx5_ifc_pmaos_reg_bits, 0]
  pmlp_reg: Annotated[struct_mlx5_ifc_pmlp_reg_bits, 0]
  pmlpn_reg: Annotated[struct_mlx5_ifc_pmlpn_reg_bits, 0]
  pmpc_reg: Annotated[struct_mlx5_ifc_pmpc_reg_bits, 0]
  pmpe_reg: Annotated[struct_mlx5_ifc_pmpe_reg_bits, 0]
  pmpr_reg: Annotated[struct_mlx5_ifc_pmpr_reg_bits, 0]
  pmtu_reg: Annotated[struct_mlx5_ifc_pmtu_reg_bits, 0]
  ppad_reg: Annotated[struct_mlx5_ifc_ppad_reg_bits, 0]
  ppcnt_reg: Annotated[struct_mlx5_ifc_ppcnt_reg_bits, 0]
  mpein_reg: Annotated[struct_mlx5_ifc_mpein_reg_bits, 0]
  mpcnt_reg: Annotated[struct_mlx5_ifc_mpcnt_reg_bits, 0]
  pplm_reg: Annotated[struct_mlx5_ifc_pplm_reg_bits, 0]
  pplr_reg: Annotated[struct_mlx5_ifc_pplr_reg_bits, 0]
  ppsc_reg: Annotated[struct_mlx5_ifc_ppsc_reg_bits, 0]
  pqdr_reg: Annotated[struct_mlx5_ifc_pqdr_reg_bits, 0]
  pspa_reg: Annotated[struct_mlx5_ifc_pspa_reg_bits, 0]
  ptas_reg: Annotated[struct_mlx5_ifc_ptas_reg_bits, 0]
  ptys_reg: Annotated[struct_mlx5_ifc_ptys_reg_bits, 0]
  mlcr_reg: Annotated[struct_mlx5_ifc_mlcr_reg_bits, 0]
  pude_reg: Annotated[struct_mlx5_ifc_pude_reg_bits, 0]
  pvlc_reg: Annotated[struct_mlx5_ifc_pvlc_reg_bits, 0]
  slrg_reg: Annotated[struct_mlx5_ifc_slrg_reg_bits, 0]
  sltp_reg: Annotated[struct_mlx5_ifc_sltp_reg_bits, 0]
  mtpps_reg: Annotated[struct_mlx5_ifc_mtpps_reg_bits, 0]
  mtppse_reg: Annotated[struct_mlx5_ifc_mtppse_reg_bits, 0]
  fpga_access_reg: Annotated[struct_mlx5_ifc_fpga_access_reg_bits, 0]
  fpga_ctrl_bits: Annotated[struct_mlx5_ifc_fpga_ctrl_bits, 0]
  fpga_cap_bits: Annotated[struct_mlx5_ifc_fpga_cap_bits, 0]
  mcqi_reg: Annotated[struct_mlx5_ifc_mcqi_reg_bits, 0]
  mcc_reg: Annotated[struct_mlx5_ifc_mcc_reg_bits, 0]
  mcda_reg: Annotated[struct_mlx5_ifc_mcda_reg_bits, 0]
  mirc_reg: Annotated[struct_mlx5_ifc_mirc_reg_bits, 0]
  mfrl_reg: Annotated[struct_mlx5_ifc_mfrl_reg_bits, 0]
  mtutc_reg: Annotated[struct_mlx5_ifc_mtutc_reg_bits, 0]
  mrtc_reg: Annotated[struct_mlx5_ifc_mrtc_reg_bits, 0]
  mtcap_reg: Annotated[struct_mlx5_ifc_mtcap_reg_bits, 0]
  mtmp_reg: Annotated[struct_mlx5_ifc_mtmp_reg_bits, 0]
  mtptm_reg: Annotated[struct_mlx5_ifc_mtptm_reg_bits, 0]
  mtctr_reg: Annotated[struct_mlx5_ifc_mtctr_reg_bits, 0]
  pphcr_reg: Annotated[struct_mlx5_ifc_pphcr_reg_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24800]], 0]
@c.record
class struct_mlx5_ifc_fpga_access_reg_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  data: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], Literal[0]], 128]
@c.record
class struct_mlx5_ifc_fpga_ctrl_bits(c.Struct):
  SIZE = 128
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  operation: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  flash_select_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  flash_select_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class union_mlx5_ifc_debug_enhancements_document_bits(c.Struct):
  SIZE = 512
  health_buffer: Annotated[struct_mlx5_ifc_health_buffer_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[512]], 0]
@c.record
class union_mlx5_ifc_uplink_pci_interface_document_bits(c.Struct):
  SIZE = 131168
  initial_seg: Annotated[struct_mlx5_ifc_initial_seg_bits, 0]
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[131168]], 0]
@c.record
class struct_mlx5_ifc_set_flow_table_root_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_set_flow_table_root_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 136]
  table_of_other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 143]
  table_vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  underlay_qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  table_eswitch_owner_vhca_id_valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 224]
  reserved_at_e1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 225]
  table_eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 240]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 256]
class _anonenum126(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MODIFY_FLOW_TABLE_MISS_TABLE_ID = _anonenum126.define('MLX5_MODIFY_FLOW_TABLE_MISS_TABLE_ID', 1)
MLX5_MODIFY_FLOW_TABLE_LAG_NEXT_TABLE_ID = _anonenum126.define('MLX5_MODIFY_FLOW_TABLE_LAG_NEXT_TABLE_ID', 32768)

@c.record
class struct_mlx5_ifc_modify_flow_table_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_flow_table_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  other_vport: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vport_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  reserved_at_88: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  eswitch_owner_vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  flow_table_context: Annotated[struct_mlx5_ifc_flow_table_context_bits, 192]
@c.record
class struct_mlx5_ifc_ets_tcn_config_reg_bits(c.Struct):
  SIZE = 64
  g: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  b: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  r: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  reserved_at_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 3]
  group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 16]
  bw_allocation: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 25]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 32]
  max_bw_units: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 44]
  max_bw_value: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
@c.record
class struct_mlx5_ifc_ets_global_config_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 0]
  r: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  reserved_at_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[29]], 3]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 32]
  max_bw_units: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 44]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 48]
  max_bw_value: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
@c.record
class struct_mlx5_ifc_qetc_reg_bits(c.Struct):
  SIZE = 640
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  port_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[48]], 16]
  tc_configuration: Annotated[c.Array[struct_mlx5_ifc_ets_tcn_config_reg_bits, Literal[8]], 64]
  global_configuration: Annotated[struct_mlx5_ifc_ets_global_config_reg_bits, 576]
@c.record
class struct_mlx5_ifc_qpdpm_dscp_reg_bits(c.Struct):
  SIZE = 16
  e: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_01: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 1]
  prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 12]
@c.record
class struct_mlx5_ifc_qpdpm_reg_bits(c.Struct):
  SIZE = 1056
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  dscp: Annotated[c.Array[struct_mlx5_ifc_qpdpm_dscp_reg_bits, Literal[64]], 32]
@c.record
class struct_mlx5_ifc_qpts_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[45]], 16]
  trust_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 61]
@c.record
class struct_mlx5_ifc_pptb_reg_bits(c.Struct):
  SIZE = 96
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 0]
  mm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 2]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 16]
  cm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 22]
  um: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 23]
  pm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  prio_x_buff: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  pm_msb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 72]
  ctrl_buff: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 88]
  untagged_buff: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
@c.record
class struct_mlx5_ifc_sbcam_reg_bits(c.Struct):
  SIZE = 608
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  feature_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  access_reg_group: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  sb_access_reg_cap_mask: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[4]], 64]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 192]
  sb_feature_cap_mask: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[4]], 320]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
  cap_total_buffer_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 512]
  cap_cell_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 544]
  cap_max_pg_buffers: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 560]
  cap_num_pool_supported: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 568]
  reserved_at_240: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 576]
  cap_sbsr_stat_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 584]
  cap_max_tclass_data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 592]
  cap_max_cpu_ingress_tclass_sb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 600]
@c.record
class struct_mlx5_ifc_pbmc_reg_bits(c.Struct):
  SIZE = 864
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  xoff_timer_value: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  xoff_refresh: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 64]
  fullness_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 73]
  port_buffer_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  buffer: Annotated[c.Array[struct_mlx5_ifc_bufferx_reg_bits, Literal[10]], 96]
  reserved_at_2e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 736]
@c.record
class struct_mlx5_ifc_sbpr_reg_bits(c.Struct):
  SIZE = 192
  desc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  snap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 2]
  dir: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 6]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 8]
  pool: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  infi_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 32]
  reserved_at_21: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 33]
  size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 64]
  mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  buff_occupancy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  clr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 128]
  reserved_at_81: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 129]
  max_buff_occupancy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  ext_buff_occupancy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
@c.record
class struct_mlx5_ifc_sbcm_reg_bits(c.Struct):
  SIZE = 320
  desc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  snap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 2]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  pnat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 16]
  pg_buff: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 18]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 24]
  dir: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 30]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 32]
  exc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 63]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  buff_occupancy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  clr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 160]
  reserved_at_a1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 161]
  max_buff_occupancy: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 192]
  min_buff: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 200]
  infi_max: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 224]
  reserved_at_e1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 225]
  max_buff: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 232]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 288]
  pool: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 316]
@c.record
class struct_mlx5_ifc_qtct_reg_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  port_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[13]], 16]
  prio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 29]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[29]], 32]
  tclass: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 61]
@c.record
class struct_mlx5_ifc_mcia_reg_bits(c.Struct):
  SIZE = 512
  l: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 1]
  module: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  i2c_device_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  page_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  device_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  dword_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  dword_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  dword_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  dword_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  dword_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 256]
  dword_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 288]
  dword_6: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 320]
  dword_7: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 352]
  dword_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 384]
  dword_9: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 416]
  dword_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 448]
  dword_11: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
@c.record
class struct_mlx5_ifc_dcbx_param_bits(c.Struct):
  SIZE = 512
  dcbx_cee_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  dcbx_ieee_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  dcbx_standby_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 2]
  reserved_at_3: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 3]
  port_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 16]
  max_application_table_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 26]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[21]], 32]
  version_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 53]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 56]
  version_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 61]
  willing_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 65]
  pfc_cap_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 68]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 72]
  pfc_cap_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 80]
  num_of_tc_oper: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 84]
  reserved_at_58: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 88]
  num_of_tc_admin: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  remote_willing: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  reserved_at_61: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 97]
  remote_pfc_cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 100]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 104]
  remote_num_of_tc: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 124]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 128]
  error: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 152]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[352]], 160]
class _anonenum127(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_LAG_PORT_SELECT_MODE_QUEUE_AFFINITY = _anonenum127.define('MLX5_LAG_PORT_SELECT_MODE_QUEUE_AFFINITY', 0)
MLX5_LAG_PORT_SELECT_MODE_PORT_SELECT_FT = _anonenum127.define('MLX5_LAG_PORT_SELECT_MODE_PORT_SELECT_FT', 1)
MLX5_LAG_PORT_SELECT_MODE_PORT_SELECT_MPESW = _anonenum127.define('MLX5_LAG_PORT_SELECT_MODE_PORT_SELECT_MPESW', 2)

@c.record
class struct_mlx5_ifc_lagc_bits(c.Struct):
  SIZE = 64
  fdb_selection_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 1]
  port_select_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 21]
  reserved_at_18: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 24]
  lag_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 29]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 32]
  active_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 44]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 48]
  tx_remap_affinity_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 52]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 56]
  tx_remap_affinity_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 60]
@c.record
class struct_mlx5_ifc_create_lag_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_create_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  ctx: Annotated[struct_mlx5_ifc_lagc_bits, 64]
@c.record
class struct_mlx5_ifc_modify_lag_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_modify_lag_in_bits(c.Struct):
  SIZE = 192
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  ctx: Annotated[struct_mlx5_ifc_lagc_bits, 128]
@c.record
class struct_mlx5_ifc_query_lag_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  ctx: Annotated[struct_mlx5_ifc_lagc_bits, 64]
@c.record
class struct_mlx5_ifc_query_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_lag_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_create_vport_lag_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_create_vport_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_vport_lag_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_destroy_vport_lag_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class _anonenum128(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MODIFY_MEMIC_OP_MOD_ALLOC = _anonenum128.define('MLX5_MODIFY_MEMIC_OP_MOD_ALLOC', 0)
MLX5_MODIFY_MEMIC_OP_MOD_DEALLOC = _anonenum128.define('MLX5_MODIFY_MEMIC_OP_MOD_DEALLOC', 1)

@c.record
class struct_mlx5_ifc_modify_memic_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 96]
  memic_operation_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
  memic_start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 192]
@c.record
class struct_mlx5_ifc_modify_memic_out_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  memic_operation_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 192]
@c.record
class struct_mlx5_ifc_alloc_memic_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 96]
  log_memic_addr_alignment: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 120]
  range_start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  range_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  memic_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_alloc_memic_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  memic_start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_memic_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  memic_start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  memic_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_dealloc_memic_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_umem_bits(c.Struct):
  SIZE = 256
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 0]
  ats: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 128]
  reserved_at_81: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[26]], 129]
  log_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 155]
  page_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  num_of_mtt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  mtt: Annotated[c.Array[struct_mlx5_ifc_mtt_bits, Literal[0]], 256]
@c.record
class struct_mlx5_ifc_uctx_bits(c.Struct):
  SIZE = 384
  cap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[352]], 32]
@c.record
class struct_mlx5_ifc_sw_icm_bits(c.Struct):
  SIZE = 512
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  log_sw_icm_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  sw_icm_start_addr: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[320]], 192]
@c.record
class struct_mlx5_ifc_geneve_tlv_option_bits(c.Struct):
  SIZE = 512
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  geneve_option_fte_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  option_class: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  option_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 112]
  reserved_at_78: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 120]
  option_data_length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 123]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
@c.record
class struct_mlx5_ifc_create_umem_in_bits(c.Struct):
  SIZE = 384
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  umem: Annotated[struct_mlx5_ifc_umem_bits, 128]
@c.record
class struct_mlx5_ifc_create_umem_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  umem_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_umem_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  umem_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_umem_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_create_uctx_in_bits(c.Struct):
  SIZE = 512
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  uctx: Annotated[struct_mlx5_ifc_uctx_bits, 128]
@c.record
class struct_mlx5_ifc_create_uctx_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_uctx_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_destroy_uctx_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_create_sw_icm_in_bits(c.Struct):
  SIZE = 640
  hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  sw_icm: Annotated[struct_mlx5_ifc_sw_icm_bits, 128]
@c.record
class struct_mlx5_ifc_create_geneve_tlv_option_in_bits(c.Struct):
  SIZE = 640
  hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  geneve_tlv_opt: Annotated[struct_mlx5_ifc_geneve_tlv_option_bits, 128]
@c.record
class struct_mlx5_ifc_mtrc_string_db_param_bits(c.Struct):
  SIZE = 64
  string_db_base_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  string_db_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 40]
@c.record
class struct_mlx5_ifc_mtrc_cap_bits(c.Struct):
  SIZE = 1024
  trace_owner: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  trace_to_memory: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 2]
  trc_ver: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 6]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 8]
  num_string_db: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  first_string_trace: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  num_string_trace: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 40]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[40]], 48]
  log_max_trace_buffer_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  string_db_param: Annotated[c.Array[struct_mlx5_ifc_mtrc_string_db_param_bits, Literal[8]], 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 640]
@c.record
class struct_mlx5_ifc_mtrc_conf_bits(c.Struct):
  SIZE = 1024
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[28]], 0]
  trace_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 32]
  log_trace_buffer_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  trace_mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[928]], 96]
@c.record
class struct_mlx5_ifc_mtrc_stdb_bits(c.Struct):
  SIZE = 64
  string_db_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 0]
  reserved_at_4: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 4]
  read_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  start_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  string_db_data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[0]], 64]
@c.record
class struct_mlx5_ifc_mtrc_ctrl_bits(c.Struct):
  SIZE = 512
  trace_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 0]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 2]
  arm_event: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  reserved_at_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 5]
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[43]], 32]
  current_timestamp52_32: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[21]], 75]
  current_timestamp31_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
@c.record
class struct_mlx5_ifc_host_params_context_bits(c.Struct):
  SIZE = 512
  host_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 8]
  host_pf_not_exist: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 13]
  reserved_at_14: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 14]
  host_pf_disabled: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 15]
  host_num_of_vfs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  host_total_vfs: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  host_pci_bus: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  host_pci_device: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  host_pci_function: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
@c.record
class struct_mlx5_ifc_query_esw_functions_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_query_esw_functions_out_bits(c.Struct):
  SIZE = 1024
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  host_params_context: Annotated[struct_mlx5_ifc_host_params_context_bits, 128]
  reserved_at_280: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 640]
  host_sf_enable: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], Literal[0]], 1024]
@c.record
class struct_mlx5_ifc_sf_partition_bits(c.Struct):
  SIZE = 32
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  log_num_sf: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  log_sf_bar_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
@c.record
class struct_mlx5_ifc_query_sf_partitions_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 64]
  num_sf_partitions: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  sf_partition: Annotated[c.Array[struct_mlx5_ifc_sf_partition_bits, Literal[0]], 128]
@c.record
class struct_mlx5_ifc_query_sf_partitions_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_sf_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_dealloc_sf_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_alloc_sf_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_alloc_sf_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  reserved_at_10: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  function_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_affiliated_event_header_bits(c.Struct):
  SIZE = 64
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  obj_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  obj_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
class _anonenum129(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_GENERAL_OBJECT_TYPES_ENCRYPTION_KEY = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_ENCRYPTION_KEY', 12)
MLX5_GENERAL_OBJECT_TYPES_IPSEC = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_IPSEC', 19)
MLX5_GENERAL_OBJECT_TYPES_SAMPLER = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_SAMPLER', 32)
MLX5_GENERAL_OBJECT_TYPES_FLOW_METER_ASO = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_FLOW_METER_ASO', 36)
MLX5_GENERAL_OBJECT_TYPES_MACSEC = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_MACSEC', 39)
MLX5_GENERAL_OBJECT_TYPES_INT_KEK = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_INT_KEK', 71)
MLX5_GENERAL_OBJECT_TYPES_RDMA_CTRL = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_RDMA_CTRL', 83)
MLX5_GENERAL_OBJECT_TYPES_PCIE_CONG_EVENT = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_PCIE_CONG_EVENT', 88)
MLX5_GENERAL_OBJECT_TYPES_FLOW_TABLE_ALIAS = _anonenum129.define('MLX5_GENERAL_OBJECT_TYPES_FLOW_TABLE_ALIAS', 65301)

class _anonenum130(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_ENCRYPTION_KEY = _anonenum130.define('MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_ENCRYPTION_KEY', 0)
MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_IPSEC = _anonenum130.define('MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_IPSEC', 1)
MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_SAMPLER = _anonenum130.define('MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_SAMPLER', 2)
MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_FLOW_METER_ASO = _anonenum130.define('MLX5_HCA_CAP_GENERAL_OBJECT_TYPES_FLOW_METER_ASO', 3)

class _anonenum131(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_HCA_CAP_2_GENERAL_OBJECT_TYPES_RDMA_CTRL = _anonenum131.define('MLX5_HCA_CAP_2_GENERAL_OBJECT_TYPES_RDMA_CTRL', 0)
MLX5_HCA_CAP_2_GENERAL_OBJECT_TYPES_PCIE_CONG_EVENT = _anonenum131.define('MLX5_HCA_CAP_2_GENERAL_OBJECT_TYPES_PCIE_CONG_EVENT', 1)

class _anonenum132(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_IPSEC_OBJECT_ICV_LEN_16B = _anonenum132.define('MLX5_IPSEC_OBJECT_ICV_LEN_16B', 0)

class _anonenum133(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_IPSEC_ASO_REG_C_0_1 = _anonenum133.define('MLX5_IPSEC_ASO_REG_C_0_1', 0)
MLX5_IPSEC_ASO_REG_C_2_3 = _anonenum133.define('MLX5_IPSEC_ASO_REG_C_2_3', 1)
MLX5_IPSEC_ASO_REG_C_4_5 = _anonenum133.define('MLX5_IPSEC_ASO_REG_C_4_5', 2)
MLX5_IPSEC_ASO_REG_C_6_7 = _anonenum133.define('MLX5_IPSEC_ASO_REG_C_6_7', 3)

class _anonenum134(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_IPSEC_ASO_MODE = _anonenum134.define('MLX5_IPSEC_ASO_MODE', 0)
MLX5_IPSEC_ASO_REPLAY_PROTECTION = _anonenum134.define('MLX5_IPSEC_ASO_REPLAY_PROTECTION', 1)
MLX5_IPSEC_ASO_INC_SN = _anonenum134.define('MLX5_IPSEC_ASO_INC_SN', 2)

class _anonenum135(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_IPSEC_ASO_REPLAY_WIN_32BIT = _anonenum135.define('MLX5_IPSEC_ASO_REPLAY_WIN_32BIT', 0)
MLX5_IPSEC_ASO_REPLAY_WIN_64BIT = _anonenum135.define('MLX5_IPSEC_ASO_REPLAY_WIN_64BIT', 1)
MLX5_IPSEC_ASO_REPLAY_WIN_128BIT = _anonenum135.define('MLX5_IPSEC_ASO_REPLAY_WIN_128BIT', 2)
MLX5_IPSEC_ASO_REPLAY_WIN_256BIT = _anonenum135.define('MLX5_IPSEC_ASO_REPLAY_WIN_256BIT', 3)

@c.record
class struct_mlx5_ifc_ipsec_aso_bits(c.Struct):
  SIZE = 512
  valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_201: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 2]
  window_sz: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 4]
  soft_lft_arm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  hard_lft_arm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  remove_flow_enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  esn_event_arm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 9]
  reserved_at_20a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[22]], 10]
  remove_flow_pkt_cnt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  remove_flow_soft_lft: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_260: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 96]
  mode_parameter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  replay_protection_window: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 256]
@c.record
class struct_mlx5_ifc_ipsec_obj_bits(c.Struct):
  SIZE = 1024
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  full_offload: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  esn_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 66]
  esn_overlap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 67]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 68]
  icv_length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 70]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 72]
  aso_return_reg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  esn_msb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  dekn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  salt: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  implicit_iv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  ipsec_aso_access_pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 264]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 288]
  ipsec_aso: Annotated[struct_mlx5_ifc_ipsec_aso_bits, 512]
@c.record
class struct_mlx5_ifc_create_ipsec_obj_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  ipsec_object: Annotated[struct_mlx5_ifc_ipsec_obj_bits, 128]
class _anonenum136(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MODIFY_IPSEC_BITMASK_ESN_OVERLAP = _anonenum136.define('MLX5_MODIFY_IPSEC_BITMASK_ESN_OVERLAP', 0)
MLX5_MODIFY_IPSEC_BITMASK_ESN_MSB = _anonenum136.define('MLX5_MODIFY_IPSEC_BITMASK_ESN_MSB', 1)

@c.record
class struct_mlx5_ifc_query_ipsec_obj_out_bits(c.Struct):
  SIZE = 1152
  general_obj_out_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0]
  ipsec_object: Annotated[struct_mlx5_ifc_ipsec_obj_bits, 128]
@c.record
class struct_mlx5_ifc_modify_ipsec_obj_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  ipsec_object: Annotated[struct_mlx5_ifc_ipsec_obj_bits, 128]
class _anonenum137(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MACSEC_ASO_REPLAY_PROTECTION = _anonenum137.define('MLX5_MACSEC_ASO_REPLAY_PROTECTION', 1)

class _anonenum138(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MACSEC_ASO_REPLAY_WIN_32BIT = _anonenum138.define('MLX5_MACSEC_ASO_REPLAY_WIN_32BIT', 0)
MLX5_MACSEC_ASO_REPLAY_WIN_64BIT = _anonenum138.define('MLX5_MACSEC_ASO_REPLAY_WIN_64BIT', 1)
MLX5_MACSEC_ASO_REPLAY_WIN_128BIT = _anonenum138.define('MLX5_MACSEC_ASO_REPLAY_WIN_128BIT', 2)
MLX5_MACSEC_ASO_REPLAY_WIN_256BIT = _anonenum138.define('MLX5_MACSEC_ASO_REPLAY_WIN_256BIT', 3)

@c.record
class struct_mlx5_ifc_macsec_aso_bits(c.Struct):
  SIZE = 512
  valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  reserved_at_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 2]
  window_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 4]
  soft_lifetime_arm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 6]
  hard_lifetime_arm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 7]
  remove_flow_enable: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 8]
  epn_event_arm: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 9]
  reserved_at_a: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[22]], 10]
  remove_flow_packet_count: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  remove_flow_soft_lifetime: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 96]
  mode_parameter: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  replay_protection_window: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 256]
@c.record
class struct_mlx5_ifc_macsec_offload_obj_bits(c.Struct):
  SIZE = 1024
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  confidentiality_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  reserved_at_41: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  epn_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 66]
  epn_overlap: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 67]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 68]
  confidentiality_offset: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 70]
  reserved_at_48: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 72]
  aso_return_reg: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 76]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  epn_msb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  dekn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  sci: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  macsec_aso_access_pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 264]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 288]
  salt: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[3]], 384]
  reserved_at_1e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 480]
  macsec_aso: Annotated[struct_mlx5_ifc_macsec_aso_bits, 512]
@c.record
class struct_mlx5_ifc_create_macsec_obj_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  macsec_object: Annotated[struct_mlx5_ifc_macsec_offload_obj_bits, 128]
@c.record
class struct_mlx5_ifc_modify_macsec_obj_in_bits(c.Struct):
  SIZE = 1152
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  macsec_object: Annotated[struct_mlx5_ifc_macsec_offload_obj_bits, 128]
class _anonenum139(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MODIFY_MACSEC_BITMASK_EPN_OVERLAP = _anonenum139.define('MLX5_MODIFY_MACSEC_BITMASK_EPN_OVERLAP', 0)
MLX5_MODIFY_MACSEC_BITMASK_EPN_MSB = _anonenum139.define('MLX5_MODIFY_MACSEC_BITMASK_EPN_MSB', 1)

@c.record
class struct_mlx5_ifc_query_macsec_obj_out_bits(c.Struct):
  SIZE = 1152
  general_obj_out_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0]
  macsec_object: Annotated[struct_mlx5_ifc_macsec_offload_obj_bits, 128]
@c.record
class struct_mlx5_ifc_wrapped_dek_bits(c.Struct):
  SIZE = 1024
  gcm_iv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 0]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  const0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 128]
  key_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 129]
  reserved_at_82: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 130]
  key2_invalid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 132]
  reserved_at_85: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 133]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  key_purpose: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 160]
  reserved_at_a5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[19]], 165]
  kek_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 184]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  key1: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 256]
  key2: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 512]
  reserved_at_300: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 768]
  const1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 832]
  reserved_at_341: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[31]], 833]
  reserved_at_360: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 864]
  auth_tag: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], 896]
@c.record
class struct_mlx5_ifc_encryption_key_obj_bits(c.Struct):
  SIZE = 4096
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  sw_wrapped: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 72]
  reserved_at_49: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 73]
  key_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 84]
  reserved_at_58: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 88]
  key_purpose: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 92]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 128]
  opaque: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 384]
  reserved_at_1c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 448]
  key: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], Literal[8]], 512]
  sw_wrapped_dek: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], Literal[8]], 1536]
  reserved_at_a00: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1536]], 2560]
@c.record
class struct_mlx5_ifc_create_encryption_key_in_bits(c.Struct):
  SIZE = 4224
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  encryption_key_object: Annotated[struct_mlx5_ifc_encryption_key_obj_bits, 128]
@c.record
class struct_mlx5_ifc_modify_encryption_key_in_bits(c.Struct):
  SIZE = 4224
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  encryption_key_object: Annotated[struct_mlx5_ifc_encryption_key_obj_bits, 128]
class _anonenum140(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_FLOW_METER_MODE_BYTES_IP_LENGTH = _anonenum140.define('MLX5_FLOW_METER_MODE_BYTES_IP_LENGTH', 0)
MLX5_FLOW_METER_MODE_BYTES_CALC_WITH_L2 = _anonenum140.define('MLX5_FLOW_METER_MODE_BYTES_CALC_WITH_L2', 1)
MLX5_FLOW_METER_MODE_BYTES_CALC_WITH_L2_IPG = _anonenum140.define('MLX5_FLOW_METER_MODE_BYTES_CALC_WITH_L2_IPG', 2)
MLX5_FLOW_METER_MODE_NUM_PACKETS = _anonenum140.define('MLX5_FLOW_METER_MODE_NUM_PACKETS', 3)

@c.record
class struct_mlx5_ifc_flow_meter_parameters_bits(c.Struct):
  SIZE = 256
  valid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  bucket_overflow: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  start_color: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 2]
  both_buckets_on_green: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 4]
  reserved_at_5: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 5]
  meter_mode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 6]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 64]
  cbs_exponent: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 67]
  cbs_mantissa: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 72]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 80]
  cir_exponent: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 83]
  cir_mantissa: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 128]
  ebs_exponent: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 131]
  ebs_mantissa: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  reserved_at_90: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 144]
  eir_exponent: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 147]
  eir_mantissa: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 152]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 160]
@c.record
class struct_mlx5_ifc_flow_meter_aso_obj_bits(c.Struct):
  SIZE = 1024
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  meter_aso_access_pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[352]], 160]
  flow_meter_parameters: Annotated[c.Array[struct_mlx5_ifc_flow_meter_parameters_bits, Literal[2]], 512]
@c.record
class struct_mlx5_ifc_create_flow_meter_aso_obj_in_bits(c.Struct):
  SIZE = 1152
  hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  flow_meter_aso_obj: Annotated[struct_mlx5_ifc_flow_meter_aso_obj_bits, 128]
@c.record
class struct_mlx5_ifc_int_kek_obj_bits(c.Struct):
  SIZE = 2048
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  auto_gen: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 72]
  reserved_at_49: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[11]], 73]
  key_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 84]
  reserved_at_58: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 96]
  pd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
  key: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[128]], Literal[8]], 512]
  reserved_at_600: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[512]], 1536]
@c.record
class struct_mlx5_ifc_create_int_kek_obj_in_bits(c.Struct):
  SIZE = 2176
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  int_kek_object: Annotated[struct_mlx5_ifc_int_kek_obj_bits, 128]
@c.record
class struct_mlx5_ifc_create_int_kek_obj_out_bits(c.Struct):
  SIZE = 2176
  general_obj_out_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0]
  int_kek_object: Annotated[struct_mlx5_ifc_int_kek_obj_bits, 128]
@c.record
class struct_mlx5_ifc_sampler_obj_bits(c.Struct):
  SIZE = 480
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  table_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 72]
  reserved_at_50: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[15]], 80]
  ignore_flow_level: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 95]
  sample_ratio: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 128]
  sample_table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 136]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  default_table_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  sw_steering_icm_address_rx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  sw_steering_icm_address_tx: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 256]
  reserved_at_140: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[160]], 320]
@c.record
class struct_mlx5_ifc_create_sampler_obj_in_bits(c.Struct):
  SIZE = 608
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  sampler_object: Annotated[struct_mlx5_ifc_sampler_obj_bits, 128]
@c.record
class struct_mlx5_ifc_query_sampler_obj_out_bits(c.Struct):
  SIZE = 608
  general_obj_out_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0]
  sampler_object: Annotated[struct_mlx5_ifc_sampler_obj_bits, 128]
class _anonenum141(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_KEY_SIZE_128 = _anonenum141.define('MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_KEY_SIZE_128', 0)
MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_KEY_SIZE_256 = _anonenum141.define('MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_KEY_SIZE_256', 1)

class _anonenum142(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_TLS = _anonenum142.define('MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_TLS', 1)
MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_IPSEC = _anonenum142.define('MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_IPSEC', 2)
MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_MACSEC = _anonenum142.define('MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_MACSEC', 4)
MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_PSP = _anonenum142.define('MLX5_GENERAL_OBJECT_TYPE_ENCRYPTION_KEY_PURPOSE_PSP', 6)

@c.record
class struct_mlx5_ifc_tls_static_params_bits(c.Struct):
  SIZE = 512
  const_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 0]
  tls_version: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 2]
  const_1: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 6]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[20]], 8]
  encryption_standard: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 28]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  initial_record_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  resync_tcp_sn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  gcm_iv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  implicit_iv: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 256]
  dek_index: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 264]
  reserved_at_120: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[224]], 288]
@c.record
class struct_mlx5_ifc_tls_progress_params_bits(c.Struct):
  SIZE = 96
  next_record_tcp_sn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  hw_resync_tcp_sn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  record_tracker_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 64]
  auth_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 66]
  reserved_at_44: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 68]
  hw_offset_record_number: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 72]
class _anonenum143(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MTT_PERM_READ = _anonenum143.define('MLX5_MTT_PERM_READ', 1)
MLX5_MTT_PERM_WRITE = _anonenum143.define('MLX5_MTT_PERM_WRITE', 2)
MLX5_MTT_PERM_RW = _anonenum143.define('MLX5_MTT_PERM_RW', 3)

class _anonenum144(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_SUSPEND_VHCA_IN_OP_MOD_SUSPEND_INITIATOR = _anonenum144.define('MLX5_SUSPEND_VHCA_IN_OP_MOD_SUSPEND_INITIATOR', 0)
MLX5_SUSPEND_VHCA_IN_OP_MOD_SUSPEND_RESPONDER = _anonenum144.define('MLX5_SUSPEND_VHCA_IN_OP_MOD_SUSPEND_RESPONDER', 1)

@c.record
class struct_mlx5_ifc_suspend_vhca_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_suspend_vhca_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class _anonenum145(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_RESUME_VHCA_IN_OP_MOD_RESUME_RESPONDER = _anonenum145.define('MLX5_RESUME_VHCA_IN_OP_MOD_RESUME_RESPONDER', 0)
MLX5_RESUME_VHCA_IN_OP_MOD_RESUME_INITIATOR = _anonenum145.define('MLX5_RESUME_VHCA_IN_OP_MOD_RESUME_INITIATOR', 1)

@c.record
class struct_mlx5_ifc_resume_vhca_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_resume_vhca_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_query_vhca_migration_state_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  incremental: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  chunk: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_query_vhca_migration_state_out_bits(c.Struct):
  SIZE = 512
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  required_umem_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 160]
  remaining_total_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 192]
  reserved_at_100: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[256]], 256]
@c.record
class struct_mlx5_ifc_save_vhca_state_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  incremental: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  set_track: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 66]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  va: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_save_vhca_state_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  actual_image_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  next_required_umem_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
@c.record
class struct_mlx5_ifc_load_vhca_state_in_bits(c.Struct):
  SIZE = 256
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  va: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 128]
  mkey: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 192]
  size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
@c.record
class struct_mlx5_ifc_load_vhca_state_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_adv_rdma_cap_bits(c.Struct):
  SIZE = 16384
  rdma_transport_manager: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 0]
  rdma_transport_manager_other_eswitch: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 1]
  reserved_at_2: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 2]
  rcx_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 32]
  reserved_at_28: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 40]
  ps_entry_log_max_value: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 42]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 48]
  qp_max_ps_num_entry: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[10]], 54]
  mp_max_num_queues: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 64]
  ps_user_context_max_log_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 72]
  message_based_qp_and_striding_wq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 80]
  reserved_at_58: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 88]
  max_receive_send_message_size_stride: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 96]
  reserved_at_70: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
  max_receive_send_message_size_byte: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 128]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[352]], 160]
  rdma_transport_rx_flow_table_properties: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 512]
  rdma_transport_tx_flow_table_properties: Annotated[struct_mlx5_ifc_flow_table_prop_layout_bits, 1024]
  rdma_transport_rx_ft_field_support_2: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1536]
  rdma_transport_tx_ft_field_support_2: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1664]
  rdma_transport_rx_ft_field_bitmask_support_2: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1792]
  rdma_transport_tx_ft_field_bitmask_support_2: Annotated[struct_mlx5_ifc_flow_table_fields_supported_2_bits, 1920]
  reserved_at_800: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14336]], 2048]
@c.record
class struct_mlx5_ifc_adv_virtualization_cap_bits(c.Struct):
  SIZE = 2048
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 0]
  pg_track_log_max_num: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 3]
  pg_track_max_num_range: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  pg_track_log_min_addr_space: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 16]
  pg_track_log_max_addr_space: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 24]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 32]
  pg_track_log_min_msg_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 35]
  reserved_at_28: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 40]
  pg_track_log_max_msg_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 43]
  reserved_at_30: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 48]
  pg_track_log_min_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 51]
  reserved_at_38: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 56]
  pg_track_log_max_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 59]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1984]], 64]
@c.record
class struct_mlx5_ifc_page_track_report_entry_bits(c.Struct):
  SIZE = 64
  dirty_address_high: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  dirty_address_low: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
class _anonenum146(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_PAGE_TRACK_STATE_TRACKING = _anonenum146.define('MLX5_PAGE_TRACK_STATE_TRACKING', 0)
MLX5_PAGE_TRACK_STATE_REPORTING = _anonenum146.define('MLX5_PAGE_TRACK_STATE_REPORTING', 1)
MLX5_PAGE_TRACK_STATE_ERROR = _anonenum146.define('MLX5_PAGE_TRACK_STATE_ERROR', 2)

@c.record
class struct_mlx5_ifc_page_track_range_bits(c.Struct):
  SIZE = 128
  start_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_page_track_bits(c.Struct):
  SIZE = 384
  modify_field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  vhca_id: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 128]
  track_type: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 132]
  log_addr_space_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 136]
  reserved_at_90: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 144]
  log_page_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 147]
  reserved_at_98: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 152]
  log_msg_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 155]
  reserved_at_a0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 160]
  reporting_qpn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 168]
  reserved_at_c0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 192]
  num_ranges: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 216]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 224]
  range_start_address: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 256]
  length: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 320]
  track_range: Annotated[c.Array[struct_mlx5_ifc_page_track_range_bits, Literal[0]], 384]
@c.record
class struct_mlx5_ifc_create_page_track_obj_in_bits(c.Struct):
  SIZE = 512
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  obj_context: Annotated[struct_mlx5_ifc_page_track_bits, 128]
@c.record
class struct_mlx5_ifc_modify_page_track_obj_in_bits(c.Struct):
  SIZE = 512
  general_obj_in_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  obj_context: Annotated[struct_mlx5_ifc_page_track_bits, 128]
@c.record
class struct_mlx5_ifc_query_page_track_obj_out_bits(c.Struct):
  SIZE = 512
  general_obj_out_cmd_hdr: Annotated[struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0]
  obj_context: Annotated[struct_mlx5_ifc_page_track_bits, 128]
@c.record
class struct_mlx5_ifc_msecq_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[18]], 32]
  network_option: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 50]
  local_ssm_code: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 52]
  local_enhanced_ssm_code: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 56]
  local_clock_identity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
class _anonenum147(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MSEES_FIELD_SELECT_ENABLE = _anonenum147.define('MLX5_MSEES_FIELD_SELECT_ENABLE', 0)
MLX5_MSEES_FIELD_SELECT_ADMIN_STATUS = _anonenum147.define('MLX5_MSEES_FIELD_SELECT_ADMIN_STATUS', 1)
MLX5_MSEES_FIELD_SELECT_ADMIN_FREQ_MEASURE = _anonenum147.define('MLX5_MSEES_FIELD_SELECT_ADMIN_FREQ_MEASURE', 2)

class enum_mlx5_msees_admin_status(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MSEES_ADMIN_STATUS_FREE_RUNNING = enum_mlx5_msees_admin_status.define('MLX5_MSEES_ADMIN_STATUS_FREE_RUNNING', 0)
MLX5_MSEES_ADMIN_STATUS_TRACK = enum_mlx5_msees_admin_status.define('MLX5_MSEES_ADMIN_STATUS_TRACK', 1)

class enum_mlx5_msees_oper_status(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MSEES_OPER_STATUS_FREE_RUNNING = enum_mlx5_msees_oper_status.define('MLX5_MSEES_OPER_STATUS_FREE_RUNNING', 0)
MLX5_MSEES_OPER_STATUS_SELF_TRACK = enum_mlx5_msees_oper_status.define('MLX5_MSEES_OPER_STATUS_SELF_TRACK', 1)
MLX5_MSEES_OPER_STATUS_OTHER_TRACK = enum_mlx5_msees_oper_status.define('MLX5_MSEES_OPER_STATUS_OTHER_TRACK', 2)
MLX5_MSEES_OPER_STATUS_HOLDOVER = enum_mlx5_msees_oper_status.define('MLX5_MSEES_OPER_STATUS_HOLDOVER', 3)
MLX5_MSEES_OPER_STATUS_FAIL_HOLDOVER = enum_mlx5_msees_oper_status.define('MLX5_MSEES_OPER_STATUS_FAIL_HOLDOVER', 4)
MLX5_MSEES_OPER_STATUS_FAIL_FREE_RUNNING = enum_mlx5_msees_oper_status.define('MLX5_MSEES_OPER_STATUS_FAIL_FREE_RUNNING', 5)

class enum_mlx5_msees_failure_reason(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_MSEES_FAILURE_REASON_UNDEFINED_ERROR = enum_mlx5_msees_failure_reason.define('MLX5_MSEES_FAILURE_REASON_UNDEFINED_ERROR', 0)
MLX5_MSEES_FAILURE_REASON_PORT_DOWN = enum_mlx5_msees_failure_reason.define('MLX5_MSEES_FAILURE_REASON_PORT_DOWN', 1)
MLX5_MSEES_FAILURE_REASON_TOO_HIGH_FREQUENCY_DIFF = enum_mlx5_msees_failure_reason.define('MLX5_MSEES_FAILURE_REASON_TOO_HIGH_FREQUENCY_DIFF', 2)
MLX5_MSEES_FAILURE_REASON_NET_SYNCHRONIZER_DEVICE_ERROR = enum_mlx5_msees_failure_reason.define('MLX5_MSEES_FAILURE_REASON_NET_SYNCHRONIZER_DEVICE_ERROR', 3)
MLX5_MSEES_FAILURE_REASON_LACK_OF_RESOURCES = enum_mlx5_msees_failure_reason.define('MLX5_MSEES_FAILURE_REASON_LACK_OF_RESOURCES', 4)

@c.record
class struct_mlx5_ifc_msees_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  local_port: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 8]
  pnat: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 16]
  lp_msb: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 18]
  reserved_at_14: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 20]
  field_select: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  admin_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 64]
  oper_status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 68]
  ho_acq: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 72]
  reserved_at_49: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[12]], 73]
  admin_freq_measure: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 85]
  oper_freq_measure: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 86]
  failure_reason: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[9]], 87]
  frequency_diff: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
@c.record
class struct_mlx5_ifc_mrtcq_reg_bits(c.Struct):
  SIZE = 512
  reserved_at_0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  rt_clock_identity: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  reserved_at_80: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[384]], 128]
@c.record
class struct_mlx5_ifc_pcie_cong_event_obj_bits(c.Struct):
  SIZE = 1024
  modify_select_field: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  inbound_event_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 64]
  outbound_event_en: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 65]
  reserved_at_42: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[30]], 66]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 96]
  inbound_cong_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 97]
  reserved_at_64: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[1]], 100]
  outbound_cong_state: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 101]
  reserved_at_68: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 104]
  inbound_cong_low_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 128]
  inbound_cong_high_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 144]
  outbound_cong_low_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 160]
  outbound_cong_high_threshold: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 176]
  reserved_at_e0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[832]], 192]
@c.record
class struct_mlx5_ifc_pcie_cong_event_cmd_in_bits(c.Struct):
  SIZE = 1152
  hdr: Annotated[struct_mlx5_ifc_general_obj_in_cmd_hdr_bits, 0]
  cong_obj: Annotated[struct_mlx5_ifc_pcie_cong_event_obj_bits, 128]
@c.record
class struct_mlx5_ifc_pcie_cong_event_cmd_out_bits(c.Struct):
  SIZE = 1152
  hdr: Annotated[struct_mlx5_ifc_general_obj_out_cmd_hdr_bits, 0]
  cong_obj: Annotated[struct_mlx5_ifc_pcie_cong_event_obj_bits, 128]
class enum_mlx5e_pcie_cong_event_mod_field(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_PCIE_CONG_EVENT_MOD_EVENT_EN = enum_mlx5e_pcie_cong_event_mod_field.define('MLX5_PCIE_CONG_EVENT_MOD_EVENT_EN', 0)
MLX5_PCIE_CONG_EVENT_MOD_THRESH = enum_mlx5e_pcie_cong_event_mod_field.define('MLX5_PCIE_CONG_EVENT_MOD_THRESH', 1)

@c.record
class struct_mlx5_ifc_psp_rotate_key_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
@c.record
class struct_mlx5_ifc_psp_rotate_key_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
class enum_mlx5_psp_gen_spi_in_key_size(Annotated[int, ctypes.c_uint32], c.Enum): pass
MLX5_PSP_GEN_SPI_IN_KEY_SIZE_128 = enum_mlx5_psp_gen_spi_in_key_size.define('MLX5_PSP_GEN_SPI_IN_KEY_SIZE_128', 0)
MLX5_PSP_GEN_SPI_IN_KEY_SIZE_256 = enum_mlx5_psp_gen_spi_in_key_size.define('MLX5_PSP_GEN_SPI_IN_KEY_SIZE_256', 1)

@c.record
class struct_mlx5_ifc_key_spi_bits(c.Struct):
  SIZE = 384
  spi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 0]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[96]], 32]
  key: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], Literal[8]], 128]
@c.record
class struct_mlx5_ifc_psp_gen_spi_in_bits(c.Struct):
  SIZE = 128
  opcode: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  uid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  reserved_at_20: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 32]
  op_mod: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 48]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 64]
  key_size: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 96]
  reserved_at_62: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[14]], 98]
  num_of_spi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 112]
@c.record
class struct_mlx5_ifc_psp_gen_spi_out_bits(c.Struct):
  SIZE = 128
  status: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[8]], 0]
  reserved_at_8: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[24]], 8]
  syndrome: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 32]
  reserved_at_40: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 64]
  num_of_spi: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 80]
  reserved_at_60: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 96]
  key_spi: Annotated[c.Array[struct_mlx5_ifc_key_spi_bits, Literal[0]], 128]
c.init_records()
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