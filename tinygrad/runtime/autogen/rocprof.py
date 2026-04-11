# mypy: disable-error-code="empty-body"
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('rocprof', ['rocprof-trace-decoder', p:='/usr/local/lib/rocprof-trace-decoder.so', p.replace('so','dylib')])
class rocprofiler_thread_trace_decoder_status_t(ctypes.c_uint32, c.Enum): pass
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS', 0)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR', 1)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES', 2)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT', 3)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA', 4)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST', 5)

class enum_rocprofiler_thread_trace_decoder_record_type_t(ctypes.c_uint32, c.Enum): pass
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP', 0)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY', 1)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT', 2)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE', 3)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO', 4)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG', 5)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA', 6)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME', 7)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY', 8)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST', 9)

rocprof_trace_decoder_trace_callback_t: TypeAlias = c.CFUNCTYPE[rocprofiler_thread_trace_decoder_status_t, [enum_rocprofiler_thread_trace_decoder_record_type_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]]
@c.record
class struct_rocprofiler_thread_trace_decoder_pc_t(c.Struct):
  SIZE = 16
  address: 'uint64_t'
  code_object_id: 'uint64_t'
uint64_t: TypeAlias = ctypes.c_uint64
struct_rocprofiler_thread_trace_decoder_pc_t.register_fields([('address', uint64_t, 0), ('code_object_id', uint64_t, 8)])
rocprof_trace_decoder_isa_callback_t: TypeAlias = c.CFUNCTYPE[rocprofiler_thread_trace_decoder_status_t, [c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_uint64], c.POINTER[ctypes.c_uint64], struct_rocprofiler_thread_trace_decoder_pc_t, ctypes.c_void_p]]
rocprof_trace_decoder_se_data_callback_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[c.POINTER[ctypes.c_ubyte]], c.POINTER[ctypes.c_uint64], ctypes.c_void_p]]
@dll.bind
def rocprof_trace_decoder_parse_data(se_data_callback:rocprof_trace_decoder_se_data_callback_t, trace_callback:rocprof_trace_decoder_trace_callback_t, isa_callback:rocprof_trace_decoder_isa_callback_t, userdata:ctypes.c_void_p) -> rocprofiler_thread_trace_decoder_status_t: ...
class enum_rocprofiler_thread_trace_decoder_info_t(ctypes.c_uint32, c.Enum): pass
ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE', 0)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST', 1)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE', 2)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE', 3)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST', 4)

rocprofiler_thread_trace_decoder_info_t: TypeAlias = enum_rocprofiler_thread_trace_decoder_info_t
@dll.bind
def rocprof_trace_decoder_get_info_string(info:rocprofiler_thread_trace_decoder_info_t) -> c.POINTER[ctypes.c_char]: ...
@dll.bind
def rocprof_trace_decoder_get_status_string(status:rocprofiler_thread_trace_decoder_status_t) -> c.POINTER[ctypes.c_char]: ...
rocprofiler_thread_trace_decoder_debug_callback_t: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_int64, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char], ctypes.c_void_p]]
@dll.bind
def rocprof_trace_decoder_dump_data(data:c.POINTER[ctypes.c_char], data_size:uint64_t, cb:rocprofiler_thread_trace_decoder_debug_callback_t, userdata:ctypes.c_void_p) -> rocprofiler_thread_trace_decoder_status_t: ...
@c.record
class union_rocprof_trace_decoder_gfx9_header_t(c.Struct):
  SIZE = 8
  legacy_version: 'uint64_t'
  gfx9_version2: 'uint64_t'
  DSIMDM: 'uint64_t'
  DCU: 'uint64_t'
  reserved1: 'uint64_t'
  SEID: 'uint64_t'
  reserved2: 'uint64_t'
  raw: 'uint64_t'
union_rocprof_trace_decoder_gfx9_header_t.register_fields([('legacy_version', uint64_t, 0, 13, 0), ('gfx9_version2', uint64_t, 1, 3, 5), ('DSIMDM', uint64_t, 2, 4, 0), ('DCU', uint64_t, 2, 5, 4), ('reserved1', uint64_t, 3, 1, 1), ('SEID', uint64_t, 3, 6, 2), ('reserved2', uint64_t, 4, 32, 0), ('raw', uint64_t, 0)])
rocprof_trace_decoder_gfx9_header_t: TypeAlias = union_rocprof_trace_decoder_gfx9_header_t
@c.record
class union_rocprof_trace_decoder_instrument_enable_t(c.Struct):
  SIZE = 4
  char1: 'ctypes.c_uint32'
  char2: 'ctypes.c_uint32'
  char3: 'ctypes.c_uint32'
  char4: 'ctypes.c_uint32'
  u32All: 'ctypes.c_uint32'
union_rocprof_trace_decoder_instrument_enable_t.register_fields([('char1', ctypes.c_uint32, 0, 8, 0), ('char2', ctypes.c_uint32, 1, 8, 0), ('char3', ctypes.c_uint32, 2, 8, 0), ('char4', ctypes.c_uint32, 3, 8, 0), ('u32All', ctypes.c_uint32, 0)])
rocprof_trace_decoder_instrument_enable_t: TypeAlias = union_rocprof_trace_decoder_instrument_enable_t
@c.record
class union_rocprof_trace_decoder_packet_header_t(c.Struct):
  SIZE = 4
  opcode: 'ctypes.c_uint32'
  type: 'ctypes.c_uint32'
  data20: 'ctypes.c_uint32'
  u32All: 'ctypes.c_uint32'
union_rocprof_trace_decoder_packet_header_t.register_fields([('opcode', ctypes.c_uint32, 0, 8, 0), ('type', ctypes.c_uint32, 1, 4, 0), ('data20', ctypes.c_uint32, 1, 20, 4), ('u32All', ctypes.c_uint32, 0)])
rocprof_trace_decoder_packet_header_t: TypeAlias = union_rocprof_trace_decoder_packet_header_t
class enum_rocprof_trace_decoder_packet_opcode_t(ctypes.c_uint32, c.Enum): pass
ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ', 4)
ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP', 5)
ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO', 6)

rocprof_trace_decoder_packet_opcode_t: TypeAlias = enum_rocprof_trace_decoder_packet_opcode_t
class enum_rocprof_trace_decoder_agent_info_type_t(ctypes.c_uint32, c.Enum): pass
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ', 0)
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL', 1)
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST', 2)

rocprof_trace_decoder_agent_info_type_t: TypeAlias = enum_rocprof_trace_decoder_agent_info_type_t
@c.record
class union_rocprof_trace_decoder_codeobj_marker_tail_t(c.Struct):
  SIZE = 4
  isUnload: 'uint32_t'
  bFromStart: 'uint32_t'
  legacy_id: 'uint32_t'
  raw: 'uint32_t'
uint32_t: TypeAlias = ctypes.c_uint32
union_rocprof_trace_decoder_codeobj_marker_tail_t.register_fields([('isUnload', uint32_t, 0, 1, 0), ('bFromStart', uint32_t, 0, 1, 1), ('legacy_id', uint32_t, 0, 30, 2), ('raw', uint32_t, 0)])
rocprof_trace_decoder_codeobj_marker_tail_t: TypeAlias = union_rocprof_trace_decoder_codeobj_marker_tail_t
class enum_rocprof_trace_decoder_codeobj_marker_type_t(ctypes.c_uint32, c.Enum): pass
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL', 0)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO', 1)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO', 2)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI', 3)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI', 4)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO', 5)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI', 6)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST', 7)

rocprof_trace_decoder_codeobj_marker_type_t: TypeAlias = enum_rocprof_trace_decoder_codeobj_marker_type_t
rocprofiler_thread_trace_decoder_pc_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_pc_t
@c.record
class struct_rocprofiler_thread_trace_decoder_perfevent_t(c.Struct):
  SIZE = 24
  time: 'int64_t'
  events0: 'uint16_t'
  events1: 'uint16_t'
  events2: 'uint16_t'
  events3: 'uint16_t'
  CU: 'uint8_t'
  bank: 'uint8_t'
int64_t: TypeAlias = ctypes.c_int64
uint16_t: TypeAlias = ctypes.c_uint16
uint8_t: TypeAlias = ctypes.c_ubyte
struct_rocprofiler_thread_trace_decoder_perfevent_t.register_fields([('time', int64_t, 0), ('events0', uint16_t, 8), ('events1', uint16_t, 10), ('events2', uint16_t, 12), ('events3', uint16_t, 14), ('CU', uint8_t, 16), ('bank', uint8_t, 17)])
rocprofiler_thread_trace_decoder_perfevent_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_perfevent_t
@c.record
class struct_rocprofiler_thread_trace_decoder_occupancy_t(c.Struct):
  SIZE = 32
  pc: 'rocprofiler_thread_trace_decoder_pc_t'
  time: 'uint64_t'
  reserved: 'uint8_t'
  cu: 'uint8_t'
  simd: 'uint8_t'
  wave_id: 'uint8_t'
  start: 'uint32_t'
  _rsvd: 'uint32_t'
struct_rocprofiler_thread_trace_decoder_occupancy_t.register_fields([('pc', rocprofiler_thread_trace_decoder_pc_t, 0), ('time', uint64_t, 16), ('reserved', uint8_t, 24), ('cu', uint8_t, 25), ('simd', uint8_t, 26), ('wave_id', uint8_t, 27), ('start', uint32_t, 28, 1, 0), ('_rsvd', uint32_t, 28, 31, 1)])
rocprofiler_thread_trace_decoder_occupancy_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_occupancy_t
class enum_rocprofiler_thread_trace_decoder_wstate_type_t(ctypes.c_uint32, c.Enum): pass
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY', 0)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE', 1)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC', 2)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT', 3)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL', 4)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST', 5)

rocprofiler_thread_trace_decoder_wstate_type_t: TypeAlias = enum_rocprofiler_thread_trace_decoder_wstate_type_t
@c.record
class struct_rocprofiler_thread_trace_decoder_wave_state_t(c.Struct):
  SIZE = 8
  type: 'int32_t'
  duration: 'int32_t'
int32_t: TypeAlias = ctypes.c_int32
struct_rocprofiler_thread_trace_decoder_wave_state_t.register_fields([('type', int32_t, 0), ('duration', int32_t, 4)])
rocprofiler_thread_trace_decoder_wave_state_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_wave_state_t
class enum_rocprofiler_thread_trace_decoder_inst_category_t(ctypes.c_uint32, c.Enum): pass
ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE', 0)
ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM', 1)
ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU', 2)
ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM', 3)
ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT', 4)
ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS', 5)
ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU', 6)
ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP', 7)
ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT', 8)
ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED', 9)
ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT', 10)
ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE', 11)
ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH', 12)
ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST', 13)

rocprofiler_thread_trace_decoder_inst_category_t: TypeAlias = enum_rocprofiler_thread_trace_decoder_inst_category_t
@c.record
class struct_rocprofiler_thread_trace_decoder_inst_t(c.Struct):
  SIZE = 32
  category: 'uint32_t'
  stall: 'uint32_t'
  duration: 'int32_t'
  time: 'int64_t'
  pc: 'rocprofiler_thread_trace_decoder_pc_t'
struct_rocprofiler_thread_trace_decoder_inst_t.register_fields([('category', uint32_t, 0, 8, 0), ('stall', uint32_t, 1, 24, 0), ('duration', int32_t, 4), ('time', int64_t, 8), ('pc', rocprofiler_thread_trace_decoder_pc_t, 16)])
rocprofiler_thread_trace_decoder_inst_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_inst_t
@c.record
class struct_rocprofiler_thread_trace_decoder_wave_t(c.Struct):
  SIZE = 64
  cu: 'uint8_t'
  simd: 'uint8_t'
  wave_id: 'uint8_t'
  contexts: 'uint8_t'
  _rsvd1: 'uint32_t'
  _rsvd2: 'uint32_t'
  _rsvd3: 'uint32_t'
  begin_time: 'int64_t'
  end_time: 'int64_t'
  timeline_size: 'uint64_t'
  instructions_size: 'uint64_t'
  timeline_array: 'c.POINTER[rocprofiler_thread_trace_decoder_wave_state_t]'
  instructions_array: 'c.POINTER[rocprofiler_thread_trace_decoder_inst_t]'
struct_rocprofiler_thread_trace_decoder_wave_t.register_fields([('cu', uint8_t, 0), ('simd', uint8_t, 1), ('wave_id', uint8_t, 2), ('contexts', uint8_t, 3), ('_rsvd1', uint32_t, 4), ('_rsvd2', uint32_t, 8), ('_rsvd3', uint32_t, 12), ('begin_time', int64_t, 16), ('end_time', int64_t, 24), ('timeline_size', uint64_t, 32), ('instructions_size', uint64_t, 40), ('timeline_array', c.POINTER[rocprofiler_thread_trace_decoder_wave_state_t], 48), ('instructions_array', c.POINTER[rocprofiler_thread_trace_decoder_inst_t], 56)])
rocprofiler_thread_trace_decoder_wave_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_wave_t
@c.record
class struct_rocprofiler_thread_trace_decoder_realtime_t(c.Struct):
  SIZE = 24
  shader_clock: 'int64_t'
  realtime_clock: 'uint64_t'
  reserved: 'uint64_t'
struct_rocprofiler_thread_trace_decoder_realtime_t.register_fields([('shader_clock', int64_t, 0), ('realtime_clock', uint64_t, 8), ('reserved', uint64_t, 16)])
rocprofiler_thread_trace_decoder_realtime_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_realtime_t
class enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t(ctypes.c_uint32, c.Enum): pass
ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t.define('ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM', 0)
ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t.define('ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV', 1)

rocprofiler_thread_trace_decoder_shaderdata_flags_t: TypeAlias = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t
@c.record
class struct_rocprofiler_thread_trace_decoder_shaderdata_t(c.Struct):
  SIZE = 24
  time: 'int64_t'
  value: 'uint64_t'
  cu: 'uint8_t'
  simd: 'uint8_t'
  wave_id: 'uint8_t'
  flags: 'uint8_t'
  reserved: 'uint32_t'
struct_rocprofiler_thread_trace_decoder_shaderdata_t.register_fields([('time', int64_t, 0), ('value', uint64_t, 8), ('cu', uint8_t, 16), ('simd', uint8_t, 17), ('wave_id', uint8_t, 18), ('flags', uint8_t, 19), ('reserved', uint32_t, 20)])
rocprofiler_thread_trace_decoder_shaderdata_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_shaderdata_t
rocprofiler_thread_trace_decoder_record_type_t: TypeAlias = enum_rocprofiler_thread_trace_decoder_record_type_t
