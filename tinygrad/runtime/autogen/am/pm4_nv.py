# mypy: ignore-errors
import ctypes
from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR
class union_PM4_MES_TYPE_3_HEADER(ctypes.Union): pass
union_PM4_MES_TYPE_3_HEADER._fields_ = []
class _anonstruct0(ctypes.Structure): pass
_anonstruct0._fields_ = []
enum_mes_set_resources_queue_type_enum = CEnum(ctypes.c_uint)
queue_type__mes_set_resources__kernel_interface_queue_kiq = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__kernel_interface_queue_kiq', 0)
queue_type__mes_set_resources__hsa_interface_queue_hiq = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__hsa_interface_queue_hiq', 1)
queue_type__mes_set_resources__hsa_debug_interface_queue = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__hsa_debug_interface_queue', 4)

class struct_pm4_mes_set_resources(ctypes.Structure): pass
struct_pm4_mes_set_resources._fields_ = []
class _anonunion1(ctypes.Union): pass
_anonunion1._fields_ = []
class _anonunion2(ctypes.Union): pass
_anonunion2._fields_ = []
class _anonstruct3(ctypes.Structure): pass
_anonstruct3._fields_ = []
class _anonunion4(ctypes.Union): pass
_anonunion4._fields_ = []
class _anonstruct5(ctypes.Structure): pass
_anonstruct5._fields_ = []
class _anonunion6(ctypes.Union): pass
_anonunion6._fields_ = []
class _anonstruct7(ctypes.Structure): pass
_anonstruct7._fields_ = []
class struct_pm4_mes_runlist(ctypes.Structure): pass
struct_pm4_mes_runlist._fields_ = []
class _anonunion8(ctypes.Union): pass
_anonunion8._fields_ = []
class _anonunion9(ctypes.Union): pass
_anonunion9._fields_ = []
class _anonstruct10(ctypes.Structure): pass
_anonstruct10._fields_ = []
class _anonunion11(ctypes.Union): pass
_anonunion11._fields_ = []
class _anonstruct12(ctypes.Structure): pass
_anonstruct12._fields_ = []
class struct_pm4_mes_map_process(ctypes.Structure): pass
struct_pm4_mes_map_process._fields_ = []
class _anonunion13(ctypes.Union): pass
_anonunion13._fields_ = []
class _anonunion14(ctypes.Union): pass
_anonunion14._fields_ = []
class _anonstruct15(ctypes.Structure): pass
_anonstruct15._fields_ = []
class _anonunion16(ctypes.Union): pass
_anonunion16._fields_ = []
class _anonstruct17(ctypes.Structure): pass
_anonstruct17._fields_ = []
class struct_PM4_MES_MAP_PROCESS_VM(ctypes.Structure): pass
struct_PM4_MES_MAP_PROCESS_VM._fields_ = []
class _anonunion18(ctypes.Union): pass
_anonunion18._fields_ = []
enum_mes_map_queues_queue_sel_enum = CEnum(ctypes.c_uint)
queue_sel__mes_map_queues__map_to_specified_queue_slots_vi = enum_mes_map_queues_queue_sel_enum.define('queue_sel__mes_map_queues__map_to_specified_queue_slots_vi', 0)
queue_sel__mes_map_queues__map_to_hws_determined_queue_slots_vi = enum_mes_map_queues_queue_sel_enum.define('queue_sel__mes_map_queues__map_to_hws_determined_queue_slots_vi', 1)

enum_mes_map_queues_queue_type_enum = CEnum(ctypes.c_uint)
queue_type__mes_map_queues__normal_compute_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__normal_compute_vi', 0)
queue_type__mes_map_queues__debug_interface_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__debug_interface_queue_vi', 1)
queue_type__mes_map_queues__normal_latency_static_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__normal_latency_static_queue_vi', 2)
queue_type__mes_map_queues__low_latency_static_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__low_latency_static_queue_vi', 3)

enum_mes_map_queues_engine_sel_enum = CEnum(ctypes.c_uint)
engine_sel__mes_map_queues__compute_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__compute_vi', 0)
engine_sel__mes_map_queues__sdma0_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__sdma0_vi', 2)
engine_sel__mes_map_queues__sdma1_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__sdma1_vi', 3)

enum_mes_map_queues_extended_engine_sel_enum = CEnum(ctypes.c_uint)
extended_engine_sel__mes_map_queues__legacy_engine_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__legacy_engine_sel', 0)
extended_engine_sel__mes_map_queues__sdma0_to_7_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__sdma0_to_7_sel', 1)
extended_engine_sel__mes_map_queues__sdma8_to_15_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__sdma8_to_15_sel', 2)

class struct_pm4_mes_map_queues(ctypes.Structure): pass
struct_pm4_mes_map_queues._fields_ = []
class _anonunion19(ctypes.Union): pass
_anonunion19._fields_ = []
class _anonunion20(ctypes.Union): pass
_anonunion20._fields_ = []
class _anonstruct21(ctypes.Structure): pass
_anonstruct21._fields_ = []
class _anonunion22(ctypes.Union): pass
_anonunion22._fields_ = []
class _anonstruct23(ctypes.Structure): pass
_anonstruct23._fields_ = []
enum_mes_query_status_interrupt_sel_enum = CEnum(ctypes.c_uint)
interrupt_sel__mes_query_status__completion_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__completion_status', 0)
interrupt_sel__mes_query_status__process_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__process_status', 1)
interrupt_sel__mes_query_status__queue_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__queue_status', 2)

enum_mes_query_status_command_enum = CEnum(ctypes.c_uint)
command__mes_query_status__interrupt_only = enum_mes_query_status_command_enum.define('command__mes_query_status__interrupt_only', 0)
command__mes_query_status__fence_only_immediate = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_only_immediate', 1)
command__mes_query_status__fence_only_after_write_ack = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_only_after_write_ack', 2)
command__mes_query_status__fence_wait_for_write_ack_send_interrupt = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_wait_for_write_ack_send_interrupt', 3)

enum_mes_query_status_engine_sel_enum = CEnum(ctypes.c_uint)
engine_sel__mes_query_status__compute = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__compute', 0)
engine_sel__mes_query_status__sdma0_queue = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__sdma0_queue', 2)
engine_sel__mes_query_status__sdma1_queue = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__sdma1_queue', 3)

class struct_pm4_mes_query_status(ctypes.Structure): pass
struct_pm4_mes_query_status._fields_ = []
class _anonunion24(ctypes.Union): pass
_anonunion24._fields_ = []
class _anonunion25(ctypes.Union): pass
_anonunion25._fields_ = []
class _anonstruct26(ctypes.Structure): pass
_anonstruct26._fields_ = []
class _anonunion27(ctypes.Union): pass
_anonunion27._fields_ = []
class _anonstruct28(ctypes.Structure): pass
_anonstruct28._fields_ = []
class _anonstruct29(ctypes.Structure): pass
_anonstruct29._fields_ = []
enum_mes_unmap_queues_action_enum = CEnum(ctypes.c_uint)
action__mes_unmap_queues__preempt_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__preempt_queues', 0)
action__mes_unmap_queues__reset_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__reset_queues', 1)
action__mes_unmap_queues__disable_process_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__disable_process_queues', 2)
action__mes_unmap_queues__reserved = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__reserved', 3)

enum_mes_unmap_queues_queue_sel_enum = CEnum(ctypes.c_uint)
queue_sel__mes_unmap_queues__perform_request_on_specified_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__perform_request_on_specified_queues', 0)
queue_sel__mes_unmap_queues__perform_request_on_pasid_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__perform_request_on_pasid_queues', 1)
queue_sel__mes_unmap_queues__unmap_all_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__unmap_all_queues', 2)
queue_sel__mes_unmap_queues__unmap_all_non_static_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__unmap_all_non_static_queues', 3)

enum_mes_unmap_queues_engine_sel_enum = CEnum(ctypes.c_uint)
engine_sel__mes_unmap_queues__compute = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__compute', 0)
engine_sel__mes_unmap_queues__sdma0 = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__sdma0', 2)
engine_sel__mes_unmap_queues__sdmal = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__sdmal', 3)

enum_mes_unmap_queues_extended_engine_sel_enum = CEnum(ctypes.c_uint)
extended_engine_sel__mes_unmap_queues__legacy_engine_sel = enum_mes_unmap_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_unmap_queues__legacy_engine_sel', 0)
extended_engine_sel__mes_unmap_queues__sdma0_to_7_sel = enum_mes_unmap_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_unmap_queues__sdma0_to_7_sel', 1)

class struct_pm4_mes_unmap_queues(ctypes.Structure): pass
struct_pm4_mes_unmap_queues._fields_ = []
class _anonunion30(ctypes.Union): pass
_anonunion30._fields_ = []
class _anonunion31(ctypes.Union): pass
_anonunion31._fields_ = []
class _anonstruct32(ctypes.Structure): pass
_anonstruct32._fields_ = []
class _anonunion33(ctypes.Union): pass
_anonunion33._fields_ = []
class _anonstruct34(ctypes.Structure): pass
_anonstruct34._fields_ = []
class _anonstruct35(ctypes.Structure): pass
_anonstruct35._fields_ = []
class _anonunion36(ctypes.Union): pass
_anonunion36._fields_ = []
class _anonstruct37(ctypes.Structure): pass
_anonstruct37._fields_ = []
class _anonunion38(ctypes.Union): pass
_anonunion38._fields_ = []
class _anonstruct39(ctypes.Structure): pass
_anonstruct39._fields_ = []
class _anonunion40(ctypes.Union): pass
_anonunion40._fields_ = []
class _anonstruct41(ctypes.Structure): pass
_anonstruct41._fields_ = []
enum_mec_release_mem_event_index_enum = CEnum(ctypes.c_uint)
event_index__mec_release_mem__end_of_pipe = enum_mec_release_mem_event_index_enum.define('event_index__mec_release_mem__end_of_pipe', 5)
event_index__mec_release_mem__shader_done = enum_mec_release_mem_event_index_enum.define('event_index__mec_release_mem__shader_done', 6)

enum_mec_release_mem_cache_policy_enum = CEnum(ctypes.c_uint)
cache_policy__mec_release_mem__lru = enum_mec_release_mem_cache_policy_enum.define('cache_policy__mec_release_mem__lru', 0)
cache_policy__mec_release_mem__stream = enum_mec_release_mem_cache_policy_enum.define('cache_policy__mec_release_mem__stream', 1)

enum_mec_release_mem_pq_exe_status_enum = CEnum(ctypes.c_uint)
pq_exe_status__mec_release_mem__default = enum_mec_release_mem_pq_exe_status_enum.define('pq_exe_status__mec_release_mem__default', 0)
pq_exe_status__mec_release_mem__phase_update = enum_mec_release_mem_pq_exe_status_enum.define('pq_exe_status__mec_release_mem__phase_update', 1)

enum_mec_release_mem_dst_sel_enum = CEnum(ctypes.c_uint)
dst_sel__mec_release_mem__memory_controller = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__memory_controller', 0)
dst_sel__mec_release_mem__tc_l2 = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__tc_l2', 1)
dst_sel__mec_release_mem__queue_write_pointer_register = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__queue_write_pointer_register', 2)
dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit', 3)

enum_mec_release_mem_int_sel_enum = CEnum(ctypes.c_uint)
int_sel__mec_release_mem__none = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__none', 0)
int_sel__mec_release_mem__send_interrupt_only = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_interrupt_only', 1)
int_sel__mec_release_mem__send_interrupt_after_write_confirm = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_interrupt_after_write_confirm', 2)
int_sel__mec_release_mem__send_data_after_write_confirm = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_data_after_write_confirm', 3)
int_sel__mec_release_mem__unconditionally_send_int_ctxid = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__unconditionally_send_int_ctxid', 4)
int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare', 5)
int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare', 6)

enum_mec_release_mem_data_sel_enum = CEnum(ctypes.c_uint)
data_sel__mec_release_mem__none = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__none', 0)
data_sel__mec_release_mem__send_32_bit_low = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_32_bit_low', 1)
data_sel__mec_release_mem__send_64_bit_data = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_64_bit_data', 2)
data_sel__mec_release_mem__send_gpu_clock_counter = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_gpu_clock_counter', 3)
data_sel__mec_release_mem__send_cp_perfcounter_hi_lo = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_cp_perfcounter_hi_lo', 4)
data_sel__mec_release_mem__store_gds_data_to_memory = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__store_gds_data_to_memory', 5)

class struct_pm4_mec_release_mem(ctypes.Structure): pass
struct_pm4_mec_release_mem._fields_ = []
class _anonunion42(ctypes.Union): pass
_anonunion42._fields_ = []
class _anonunion43(ctypes.Union): pass
_anonunion43._fields_ = []
class _anonstruct44(ctypes.Structure): pass
_anonstruct44._fields_ = []
class _anonunion45(ctypes.Union): pass
_anonunion45._fields_ = []
class _anonstruct46(ctypes.Structure): pass
_anonstruct46._fields_ = []
class _anonunion47(ctypes.Union): pass
_anonunion47._fields_ = []
class _anonstruct48(ctypes.Structure): pass
_anonstruct48._fields_ = []
class _anonstruct49(ctypes.Structure): pass
_anonstruct49._fields_ = []
class _anonunion50(ctypes.Union): pass
_anonunion50._fields_ = []
class _anonunion51(ctypes.Union): pass
_anonunion51._fields_ = []
class _anonstruct52(ctypes.Structure): pass
_anonstruct52._fields_ = []
class _anonunion53(ctypes.Union): pass
_anonunion53._fields_ = []
enum_WRITE_DATA_dst_sel_enum = CEnum(ctypes.c_uint)
dst_sel___write_data__mem_mapped_register = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__mem_mapped_register', 0)
dst_sel___write_data__tc_l2 = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__tc_l2', 2)
dst_sel___write_data__gds = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__gds', 3)
dst_sel___write_data__memory = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__memory', 5)
dst_sel___write_data__memory_mapped_adc_persistent_state = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__memory_mapped_adc_persistent_state', 6)

enum_WRITE_DATA_addr_incr_enum = CEnum(ctypes.c_uint)
addr_incr___write_data__increment_address = enum_WRITE_DATA_addr_incr_enum.define('addr_incr___write_data__increment_address', 0)
addr_incr___write_data__do_not_increment_address = enum_WRITE_DATA_addr_incr_enum.define('addr_incr___write_data__do_not_increment_address', 1)

enum_WRITE_DATA_wr_confirm_enum = CEnum(ctypes.c_uint)
wr_confirm___write_data__do_not_wait_for_write_confirmation = enum_WRITE_DATA_wr_confirm_enum.define('wr_confirm___write_data__do_not_wait_for_write_confirmation', 0)
wr_confirm___write_data__wait_for_write_confirmation = enum_WRITE_DATA_wr_confirm_enum.define('wr_confirm___write_data__wait_for_write_confirmation', 1)

enum_WRITE_DATA_cache_policy_enum = CEnum(ctypes.c_uint)
cache_policy___write_data__lru = enum_WRITE_DATA_cache_policy_enum.define('cache_policy___write_data__lru', 0)
cache_policy___write_data__stream = enum_WRITE_DATA_cache_policy_enum.define('cache_policy___write_data__stream', 1)

class struct_pm4_mec_write_data_mmio(ctypes.Structure): pass
struct_pm4_mec_write_data_mmio._fields_ = []
class _anonunion54(ctypes.Union): pass
_anonunion54._fields_ = []
class _anonunion55(ctypes.Union): pass
class _anonunion55_bitfields2(ctypes.Structure): pass
_anonunion55_bitfields2._fields_ = [
  ('reserved1', ctypes.c_uint,8),
  ('dst_sel', ctypes.c_uint,4),
  ('reserved2', ctypes.c_uint,4),
  ('addr_incr', ctypes.c_uint,1),
  ('reserved3', ctypes.c_uint,2),
  ('resume_vf', ctypes.c_uint,1),
  ('wr_confirm', ctypes.c_uint,1),
  ('reserved4', ctypes.c_uint,4),
  ('cache_policy', ctypes.c_uint,2),
  ('reserved5', ctypes.c_uint,5),
]
_anonunion55._fields_ = [
  ('bitfields2', _anonunion55_bitfields2),
  ('ordinal2', ctypes.c_uint),
]
class _anonunion56(ctypes.Union): pass
class _anonunion56_bitfields3(ctypes.Structure): pass
_anonunion56_bitfields3._fields_ = [
  ('dst_mmreg_addr', ctypes.c_uint,18),
  ('reserved6', ctypes.c_uint,14),
]
_anonunion56._fields_ = [
  ('bitfields3', _anonunion56_bitfields3),
  ('ordinal3', ctypes.c_uint),
]
_anonenum57 = CEnum(ctypes.c_uint)
CACHE_FLUSH_AND_INV_TS_EVENT = _anonenum57.define('CACHE_FLUSH_AND_INV_TS_EVENT', 20)

PACKET_TYPE0 = 0
PACKET_TYPE1 = 1
PACKET_TYPE2 = 2
PACKET_TYPE3 = 3
CP_PACKET_GET_TYPE = lambda h: (((h) >> 30) & 3)
CP_PACKET_GET_COUNT = lambda h: (((h) >> 16) & 0x3FFF)
CP_PACKET0_GET_REG = lambda h: ((h) & 0xFFFF)
CP_PACKET3_GET_OPCODE = lambda h: (((h) >> 8) & 0xFF)
PACKET0 = lambda reg,n: ((PACKET_TYPE0 << 30) | ((reg) & 0xFFFF) | ((n) & 0x3FFF) << 16)
CP_PACKET2 = 0x80000000
PACKET2_PAD_SHIFT = 0
PACKET2_PAD_MASK = (0x3fffffff << 0)
PACKET2 = lambda v: (CP_PACKET2 | REG_SET(PACKET2_PAD, (v)))
PACKET3 = lambda op,n: ((PACKET_TYPE3 << 30) | (((op) & 0xFF) << 8) | ((n) & 0x3FFF) << 16)
PACKET3_COMPUTE = lambda op,n: (PACKET3(op, n) | 1 << 1)
PACKET3_NOP = 0x10
PACKET3_SET_BASE = 0x11
PACKET3_BASE_INDEX = lambda x: ((x) << 0)
CE_PARTITION_BASE = 3
PACKET3_CLEAR_STATE = 0x12
PACKET3_INDEX_BUFFER_SIZE = 0x13
PACKET3_DISPATCH_DIRECT = 0x15
PACKET3_DISPATCH_INDIRECT = 0x16
PACKET3_INDIRECT_BUFFER_END = 0x17
PACKET3_INDIRECT_BUFFER_CNST_END = 0x19
PACKET3_ATOMIC_GDS = 0x1D
PACKET3_ATOMIC_MEM = 0x1E
PACKET3_OCCLUSION_QUERY = 0x1F
PACKET3_SET_PREDICATION = 0x20
PACKET3_REG_RMW = 0x21
PACKET3_COND_EXEC = 0x22
PACKET3_PRED_EXEC = 0x23
PACKET3_DRAW_INDIRECT = 0x24
PACKET3_DRAW_INDEX_INDIRECT = 0x25
PACKET3_INDEX_BASE = 0x26
PACKET3_DRAW_INDEX_2 = 0x27
PACKET3_CONTEXT_CONTROL = 0x28
PACKET3_INDEX_TYPE = 0x2A
PACKET3_DRAW_INDIRECT_MULTI = 0x2C
PACKET3_DRAW_INDEX_AUTO = 0x2D
PACKET3_NUM_INSTANCES = 0x2F
PACKET3_DRAW_INDEX_MULTI_AUTO = 0x30
PACKET3_INDIRECT_BUFFER_PRIV = 0x32
PACKET3_INDIRECT_BUFFER_CNST = 0x33
PACKET3_COND_INDIRECT_BUFFER_CNST = 0x33
PACKET3_STRMOUT_BUFFER_UPDATE = 0x34
PACKET3_DRAW_INDEX_OFFSET_2 = 0x35
PACKET3_DRAW_PREAMBLE = 0x36
PACKET3_WRITE_DATA = 0x37
WRITE_DATA_DST_SEL = lambda x: ((x) << 8)
WR_ONE_ADDR = (1 << 16)
WR_CONFIRM = (1 << 20)
WRITE_DATA_CACHE_POLICY = lambda x: ((x) << 25)
WRITE_DATA_ENGINE_SEL = lambda x: ((x) << 30)
PACKET3_DRAW_INDEX_INDIRECT_MULTI = 0x38
PACKET3_MEM_SEMAPHORE = 0x39
PACKET3_SEM_USE_MAILBOX = (0x1 << 16)
PACKET3_SEM_SEL_SIGNAL_TYPE = (0x1 << 20)
PACKET3_SEM_SEL_SIGNAL = (0x6 << 29)
PACKET3_SEM_SEL_WAIT = (0x7 << 29)
PACKET3_DRAW_INDEX_MULTI_INST = 0x3A
PACKET3_COPY_DW = 0x3B
PACKET3_WAIT_REG_MEM = 0x3C
WAIT_REG_MEM_FUNCTION = lambda x: ((x) << 0)
WAIT_REG_MEM_MEM_SPACE = lambda x: ((x) << 4)
WAIT_REG_MEM_OPERATION = lambda x: ((x) << 6)
WAIT_REG_MEM_ENGINE = lambda x: ((x) << 8)
PACKET3_INDIRECT_BUFFER = 0x3F
INDIRECT_BUFFER_VALID = (1 << 23)
INDIRECT_BUFFER_CACHE_POLICY = lambda x: ((x) << 28)
INDIRECT_BUFFER_PRE_ENB = lambda x: ((x) << 21)
INDIRECT_BUFFER_PRE_RESUME = lambda x: ((x) << 30)
PACKET3_COND_INDIRECT_BUFFER = 0x3F
PACKET3_COPY_DATA = 0x40
PACKET3_CP_DMA = 0x41
PACKET3_PFP_SYNC_ME = 0x42
PACKET3_SURFACE_SYNC = 0x43
PACKET3_ME_INITIALIZE = 0x44
PACKET3_COND_WRITE = 0x45
PACKET3_EVENT_WRITE = 0x46
EVENT_TYPE = lambda x: ((x) << 0)
EVENT_INDEX = lambda x: ((x) << 8)
PACKET3_EVENT_WRITE_EOP = 0x47
PACKET3_EVENT_WRITE_EOS = 0x48
PACKET3_RELEASE_MEM = 0x49
PACKET3_RELEASE_MEM_EVENT_TYPE = lambda x: ((x) << 0)
PACKET3_RELEASE_MEM_EVENT_INDEX = lambda x: ((x) << 8)
PACKET3_RELEASE_MEM_GCR_GLM_WB = (1 << 12)
PACKET3_RELEASE_MEM_GCR_GLM_INV = (1 << 13)
PACKET3_RELEASE_MEM_GCR_GLV_INV = (1 << 14)
PACKET3_RELEASE_MEM_GCR_GL1_INV = (1 << 15)
PACKET3_RELEASE_MEM_GCR_GL2_US = (1 << 16)
PACKET3_RELEASE_MEM_GCR_GL2_RANGE = (1 << 17)
PACKET3_RELEASE_MEM_GCR_GL2_DISCARD = (1 << 19)
PACKET3_RELEASE_MEM_GCR_GL2_INV = (1 << 20)
PACKET3_RELEASE_MEM_GCR_GL2_WB = (1 << 21)
PACKET3_RELEASE_MEM_GCR_SEQ = (1 << 22)
PACKET3_RELEASE_MEM_CACHE_POLICY = lambda x: ((x) << 25)
PACKET3_RELEASE_MEM_EXECUTE = (1 << 28)
PACKET3_RELEASE_MEM_DATA_SEL = lambda x: ((x) << 29)
PACKET3_RELEASE_MEM_INT_SEL = lambda x: ((x) << 24)
PACKET3_RELEASE_MEM_DST_SEL = lambda x: ((x) << 16)
PACKET3_PREAMBLE_CNTL = 0x4A
PACKET3_PREAMBLE_BEGIN_CLEAR_STATE = (2 << 28)
PACKET3_PREAMBLE_END_CLEAR_STATE = (3 << 28)
PACKET3_DMA_DATA = 0x50
PACKET3_DMA_DATA_ENGINE = lambda x: ((x) << 0)
PACKET3_DMA_DATA_SRC_CACHE_POLICY = lambda x: ((x) << 13)
PACKET3_DMA_DATA_DST_SEL = lambda x: ((x) << 20)
PACKET3_DMA_DATA_DST_CACHE_POLICY = lambda x: ((x) << 25)
PACKET3_DMA_DATA_SRC_SEL = lambda x: ((x) << 29)
PACKET3_DMA_DATA_CP_SYNC = (1 << 31)
PACKET3_DMA_DATA_CMD_SAS = (1 << 26)
PACKET3_DMA_DATA_CMD_DAS = (1 << 27)
PACKET3_DMA_DATA_CMD_SAIC = (1 << 28)
PACKET3_DMA_DATA_CMD_DAIC = (1 << 29)
PACKET3_DMA_DATA_CMD_RAW_WAIT = (1 << 30)
PACKET3_CONTEXT_REG_RMW = 0x51
PACKET3_GFX_CNTX_UPDATE = 0x52
PACKET3_BLK_CNTX_UPDATE = 0x53
PACKET3_INCR_UPDT_STATE = 0x55
PACKET3_ACQUIRE_MEM = 0x58
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV = lambda x: ((x) << 0)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_RANGE = lambda x: ((x) << 2)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB = lambda x: ((x) << 4)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV = lambda x: ((x) << 5)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB = lambda x: ((x) << 6)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV = lambda x: ((x) << 7)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV = lambda x: ((x) << 8)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV = lambda x: ((x) << 9)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_US = lambda x: ((x) << 10)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_RANGE = lambda x: ((x) << 11)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_DISCARD = lambda x: ((x) << 13)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV = lambda x: ((x) << 14)
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB = lambda x: ((x) << 15)
PACKET3_ACQUIRE_MEM_GCR_CNTL_SEQ = lambda x: ((x) << 16)
PACKET3_ACQUIRE_MEM_GCR_RANGE_IS_PA = (1 << 18)
PACKET3_REWIND = 0x59
PACKET3_INTERRUPT = 0x5A
PACKET3_GEN_PDEPTE = 0x5B
PACKET3_INDIRECT_BUFFER_PASID = 0x5C
PACKET3_PRIME_UTCL2 = 0x5D
PACKET3_LOAD_UCONFIG_REG = 0x5E
PACKET3_LOAD_SH_REG = 0x5F
PACKET3_LOAD_CONFIG_REG = 0x60
PACKET3_LOAD_CONTEXT_REG = 0x61
PACKET3_LOAD_COMPUTE_STATE = 0x62
PACKET3_LOAD_SH_REG_INDEX = 0x63
PACKET3_SET_CONFIG_REG = 0x68
PACKET3_SET_CONFIG_REG_START = 0x00002000
PACKET3_SET_CONFIG_REG_END = 0x00002c00
PACKET3_SET_CONTEXT_REG = 0x69
PACKET3_SET_CONTEXT_REG_START = 0x0000a000
PACKET3_SET_CONTEXT_REG_END = 0x0000a400
PACKET3_SET_CONTEXT_REG_INDEX = 0x6A
PACKET3_SET_VGPR_REG_DI_MULTI = 0x71
PACKET3_SET_SH_REG_DI = 0x72
PACKET3_SET_CONTEXT_REG_INDIRECT = 0x73
PACKET3_SET_SH_REG_DI_MULTI = 0x74
PACKET3_GFX_PIPE_LOCK = 0x75
PACKET3_SET_SH_REG = 0x76
PACKET3_SET_SH_REG_START = 0x00002c00
PACKET3_SET_SH_REG_END = 0x00003000
PACKET3_SET_SH_REG_OFFSET = 0x77
PACKET3_SET_QUEUE_REG = 0x78
PACKET3_SET_UCONFIG_REG = 0x79
PACKET3_SET_UCONFIG_REG_START = 0x0000c000
PACKET3_SET_UCONFIG_REG_END = 0x0000c400
PACKET3_SET_UCONFIG_REG_INDEX = 0x7A
PACKET3_FORWARD_HEADER = 0x7C
PACKET3_SCRATCH_RAM_WRITE = 0x7D
PACKET3_SCRATCH_RAM_READ = 0x7E
PACKET3_LOAD_CONST_RAM = 0x80
PACKET3_WRITE_CONST_RAM = 0x81
PACKET3_DUMP_CONST_RAM = 0x83
PACKET3_INCREMENT_CE_COUNTER = 0x84
PACKET3_INCREMENT_DE_COUNTER = 0x85
PACKET3_WAIT_ON_CE_COUNTER = 0x86
PACKET3_WAIT_ON_DE_COUNTER_DIFF = 0x88
PACKET3_SWITCH_BUFFER = 0x8B
PACKET3_DISPATCH_DRAW_PREAMBLE = 0x8C
PACKET3_DISPATCH_DRAW_PREAMBLE_ACE = 0x8C
PACKET3_DISPATCH_DRAW = 0x8D
PACKET3_DISPATCH_DRAW_ACE = 0x8D
PACKET3_GET_LOD_STATS = 0x8E
PACKET3_DRAW_MULTI_PREAMBLE = 0x8F
PACKET3_FRAME_CONTROL = 0x90
FRAME_TMZ = (1 << 0)
FRAME_CMD = lambda x: ((x) << 28)
PACKET3_INDEX_ATTRIBUTES_INDIRECT = 0x91
PACKET3_WAIT_REG_MEM64 = 0x93
PACKET3_COND_PREEMPT = 0x94
PACKET3_HDP_FLUSH = 0x95
PACKET3_COPY_DATA_RB = 0x96
PACKET3_INVALIDATE_TLBS = 0x98
PACKET3_INVALIDATE_TLBS_DST_SEL = lambda x: ((x) << 0)
PACKET3_INVALIDATE_TLBS_ALL_HUB = lambda x: ((x) << 4)
PACKET3_INVALIDATE_TLBS_PASID = lambda x: ((x) << 5)
PACKET3_AQL_PACKET = 0x99
PACKET3_DMA_DATA_FILL_MULTI = 0x9A
PACKET3_SET_SH_REG_INDEX = 0x9B
PACKET3_DRAW_INDIRECT_COUNT_MULTI = 0x9C
PACKET3_DRAW_INDEX_INDIRECT_COUNT_MULTI = 0x9D
PACKET3_DUMP_CONST_RAM_OFFSET = 0x9E
PACKET3_LOAD_CONTEXT_REG_INDEX = 0x9F
PACKET3_SET_RESOURCES = 0xA0
PACKET3_SET_RESOURCES_VMID_MASK = lambda x: ((x) << 0)
PACKET3_SET_RESOURCES_UNMAP_LATENTY = lambda x: ((x) << 16)
PACKET3_SET_RESOURCES_QUEUE_TYPE = lambda x: ((x) << 29)
PACKET3_MAP_PROCESS = 0xA1
PACKET3_MAP_QUEUES = 0xA2
PACKET3_MAP_QUEUES_QUEUE_SEL = lambda x: ((x) << 4)
PACKET3_MAP_QUEUES_VMID = lambda x: ((x) << 8)
PACKET3_MAP_QUEUES_QUEUE = lambda x: ((x) << 13)
PACKET3_MAP_QUEUES_PIPE = lambda x: ((x) << 16)
PACKET3_MAP_QUEUES_ME = lambda x: ((x) << 18)
PACKET3_MAP_QUEUES_QUEUE_TYPE = lambda x: ((x) << 21)
PACKET3_MAP_QUEUES_ALLOC_FORMAT = lambda x: ((x) << 24)
PACKET3_MAP_QUEUES_ENGINE_SEL = lambda x: ((x) << 26)
PACKET3_MAP_QUEUES_NUM_QUEUES = lambda x: ((x) << 29)
PACKET3_MAP_QUEUES_CHECK_DISABLE = lambda x: ((x) << 1)
PACKET3_MAP_QUEUES_DOORBELL_OFFSET = lambda x: ((x) << 2)
PACKET3_UNMAP_QUEUES = 0xA3
PACKET3_UNMAP_QUEUES_ACTION = lambda x: ((x) << 0)
PACKET3_UNMAP_QUEUES_QUEUE_SEL = lambda x: ((x) << 4)
PACKET3_UNMAP_QUEUES_ENGINE_SEL = lambda x: ((x) << 26)
PACKET3_UNMAP_QUEUES_NUM_QUEUES = lambda x: ((x) << 29)
PACKET3_UNMAP_QUEUES_PASID = lambda x: ((x) << 0)
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET0 = lambda x: ((x) << 2)
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET1 = lambda x: ((x) << 2)
PACKET3_UNMAP_QUEUES_RB_WPTR = lambda x: ((x) << 0)
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET2 = lambda x: ((x) << 2)
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET3 = lambda x: ((x) << 2)
PACKET3_QUERY_STATUS = 0xA4
PACKET3_QUERY_STATUS_CONTEXT_ID = lambda x: ((x) << 0)
PACKET3_QUERY_STATUS_INTERRUPT_SEL = lambda x: ((x) << 28)
PACKET3_QUERY_STATUS_COMMAND = lambda x: ((x) << 30)
PACKET3_QUERY_STATUS_PASID = lambda x: ((x) << 0)
PACKET3_QUERY_STATUS_DOORBELL_OFFSET = lambda x: ((x) << 2)
PACKET3_QUERY_STATUS_ENG_SEL = lambda x: ((x) << 25)
PACKET3_RUN_LIST = 0xA5
PACKET3_MAP_PROCESS_VM = 0xA6
PACKET3_SET_Q_PREEMPTION_MODE = 0xF0
PACKET3_SET_Q_PREEMPTION_MODE_IB_VMID = lambda x: ((x) << 0)
PACKET3_SET_Q_PREEMPTION_MODE_INIT_SHADOW_MEM = (1 << 0)