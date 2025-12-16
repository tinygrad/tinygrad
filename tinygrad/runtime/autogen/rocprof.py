# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('rocprof', ['rocprof-trace-decoder', p:='/usr/local/lib/rocprof-trace-decoder.so', p.replace('so','dylib')])
rocprofiler_thread_trace_decoder_status_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS', 0)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR', 1)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES', 2)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT', 3)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA', 4)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST', 5)

enum_rocprofiler_thread_trace_decoder_record_type_t = CEnum(ctypes.c_uint32)
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

rocprof_trace_decoder_trace_callback_t = ctypes.CFUNCTYPE(rocprofiler_thread_trace_decoder_status_t, enum_rocprofiler_thread_trace_decoder_record_type_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)
class struct_rocprofiler_thread_trace_decoder_pc_t(Struct): pass
uint64_t = ctypes.c_uint64
struct_rocprofiler_thread_trace_decoder_pc_t.SIZE = 16
struct_rocprofiler_thread_trace_decoder_pc_t._fields_ = ['address', 'code_object_id']
setattr(struct_rocprofiler_thread_trace_decoder_pc_t, 'address', field(0, uint64_t))
setattr(struct_rocprofiler_thread_trace_decoder_pc_t, 'code_object_id', field(8, uint64_t))
rocprof_trace_decoder_isa_callback_t = ctypes.CFUNCTYPE(rocprofiler_thread_trace_decoder_status_t, Pointer(ctypes.c_char), Pointer(ctypes.c_uint64), Pointer(ctypes.c_uint64), struct_rocprofiler_thread_trace_decoder_pc_t, ctypes.c_void_p)
rocprof_trace_decoder_se_data_callback_t = ctypes.CFUNCTYPE(ctypes.c_uint64, Pointer(Pointer(ctypes.c_ubyte)), Pointer(ctypes.c_uint64), ctypes.c_void_p)
try: (rocprof_trace_decoder_parse_data:=dll.rocprof_trace_decoder_parse_data).restype, rocprof_trace_decoder_parse_data.argtypes = rocprofiler_thread_trace_decoder_status_t, [rocprof_trace_decoder_se_data_callback_t, rocprof_trace_decoder_trace_callback_t, rocprof_trace_decoder_isa_callback_t, ctypes.c_void_p]
except AttributeError: pass

enum_rocprofiler_thread_trace_decoder_info_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE', 0)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST', 1)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE', 2)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE', 3)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST', 4)

rocprofiler_thread_trace_decoder_info_t = enum_rocprofiler_thread_trace_decoder_info_t
try: (rocprof_trace_decoder_get_info_string:=dll.rocprof_trace_decoder_get_info_string).restype, rocprof_trace_decoder_get_info_string.argtypes = Pointer(ctypes.c_char), [rocprofiler_thread_trace_decoder_info_t]
except AttributeError: pass

try: (rocprof_trace_decoder_get_status_string:=dll.rocprof_trace_decoder_get_status_string).restype, rocprof_trace_decoder_get_status_string.argtypes = Pointer(ctypes.c_char), [rocprofiler_thread_trace_decoder_status_t]
except AttributeError: pass

rocprofiler_thread_trace_decoder_debug_callback_t = ctypes.CFUNCTYPE(None, ctypes.c_int64, Pointer(ctypes.c_char), Pointer(ctypes.c_char), ctypes.c_void_p)
try: (rocprof_trace_decoder_dump_data:=dll.rocprof_trace_decoder_dump_data).restype, rocprof_trace_decoder_dump_data.argtypes = rocprofiler_thread_trace_decoder_status_t, [Pointer(ctypes.c_char), uint64_t, rocprofiler_thread_trace_decoder_debug_callback_t, ctypes.c_void_p]
except AttributeError: pass

class union_rocprof_trace_decoder_gfx9_header_t(Union): pass
union_rocprof_trace_decoder_gfx9_header_t.SIZE = 8
union_rocprof_trace_decoder_gfx9_header_t._fields_ = ['legacy_version', 'gfx9_version2', 'DSIMDM', 'DCU', 'reserved1', 'SEID', 'reserved2', 'raw']
setattr(union_rocprof_trace_decoder_gfx9_header_t, 'legacy_version', field(0, uint64_t, 13, 0))
setattr(union_rocprof_trace_decoder_gfx9_header_t, 'gfx9_version2', field(1, uint64_t, 3, 5))
setattr(union_rocprof_trace_decoder_gfx9_header_t, 'DSIMDM', field(2, uint64_t, 4, 0))
setattr(union_rocprof_trace_decoder_gfx9_header_t, 'DCU', field(2, uint64_t, 5, 4))
setattr(union_rocprof_trace_decoder_gfx9_header_t, 'reserved1', field(3, uint64_t, 1, 1))
setattr(union_rocprof_trace_decoder_gfx9_header_t, 'SEID', field(3, uint64_t, 6, 2))
setattr(union_rocprof_trace_decoder_gfx9_header_t, 'reserved2', field(4, uint64_t, 32, 0))
setattr(union_rocprof_trace_decoder_gfx9_header_t, 'raw', field(0, uint64_t))
rocprof_trace_decoder_gfx9_header_t = union_rocprof_trace_decoder_gfx9_header_t
class union_rocprof_trace_decoder_instrument_enable_t(Union): pass
union_rocprof_trace_decoder_instrument_enable_t.SIZE = 4
union_rocprof_trace_decoder_instrument_enable_t._fields_ = ['char1', 'char2', 'char3', 'char4', 'u32All']
setattr(union_rocprof_trace_decoder_instrument_enable_t, 'char1', field(0, ctypes.c_uint32, 8, 0))
setattr(union_rocprof_trace_decoder_instrument_enable_t, 'char2', field(1, ctypes.c_uint32, 8, 0))
setattr(union_rocprof_trace_decoder_instrument_enable_t, 'char3', field(2, ctypes.c_uint32, 8, 0))
setattr(union_rocprof_trace_decoder_instrument_enable_t, 'char4', field(3, ctypes.c_uint32, 8, 0))
setattr(union_rocprof_trace_decoder_instrument_enable_t, 'u32All', field(0, ctypes.c_uint32))
rocprof_trace_decoder_instrument_enable_t = union_rocprof_trace_decoder_instrument_enable_t
class union_rocprof_trace_decoder_packet_header_t(Union): pass
union_rocprof_trace_decoder_packet_header_t.SIZE = 4
union_rocprof_trace_decoder_packet_header_t._fields_ = ['opcode', 'type', 'data20', 'u32All']
setattr(union_rocprof_trace_decoder_packet_header_t, 'opcode', field(0, ctypes.c_uint32, 8, 0))
setattr(union_rocprof_trace_decoder_packet_header_t, 'type', field(1, ctypes.c_uint32, 4, 0))
setattr(union_rocprof_trace_decoder_packet_header_t, 'data20', field(1, ctypes.c_uint32, 20, 4))
setattr(union_rocprof_trace_decoder_packet_header_t, 'u32All', field(0, ctypes.c_uint32))
rocprof_trace_decoder_packet_header_t = union_rocprof_trace_decoder_packet_header_t
enum_rocprof_trace_decoder_packet_opcode_t = CEnum(ctypes.c_uint32)
ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ', 4)
ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP', 5)
ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO', 6)

rocprof_trace_decoder_packet_opcode_t = enum_rocprof_trace_decoder_packet_opcode_t
enum_rocprof_trace_decoder_agent_info_type_t = CEnum(ctypes.c_uint32)
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ', 0)
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL', 1)
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST', 2)

rocprof_trace_decoder_agent_info_type_t = enum_rocprof_trace_decoder_agent_info_type_t
class union_rocprof_trace_decoder_codeobj_marker_tail_t(Union): pass
uint32_t = ctypes.c_uint32
union_rocprof_trace_decoder_codeobj_marker_tail_t.SIZE = 4
union_rocprof_trace_decoder_codeobj_marker_tail_t._fields_ = ['isUnload', 'bFromStart', 'legacy_id', 'raw']
setattr(union_rocprof_trace_decoder_codeobj_marker_tail_t, 'isUnload', field(0, uint32_t, 1, 0))
setattr(union_rocprof_trace_decoder_codeobj_marker_tail_t, 'bFromStart', field(0, uint32_t, 1, 1))
setattr(union_rocprof_trace_decoder_codeobj_marker_tail_t, 'legacy_id', field(0, uint32_t, 30, 2))
setattr(union_rocprof_trace_decoder_codeobj_marker_tail_t, 'raw', field(0, uint32_t))
rocprof_trace_decoder_codeobj_marker_tail_t = union_rocprof_trace_decoder_codeobj_marker_tail_t
enum_rocprof_trace_decoder_codeobj_marker_type_t = CEnum(ctypes.c_uint32)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL', 0)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO', 1)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO', 2)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI', 3)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI', 4)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO', 5)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI', 6)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST', 7)

rocprof_trace_decoder_codeobj_marker_type_t = enum_rocprof_trace_decoder_codeobj_marker_type_t
rocprofiler_thread_trace_decoder_pc_t = struct_rocprofiler_thread_trace_decoder_pc_t
class struct_rocprofiler_thread_trace_decoder_perfevent_t(Struct): pass
int64_t = ctypes.c_int64
uint16_t = ctypes.c_uint16
uint8_t = ctypes.c_ubyte
struct_rocprofiler_thread_trace_decoder_perfevent_t.SIZE = 24
struct_rocprofiler_thread_trace_decoder_perfevent_t._fields_ = ['time', 'events0', 'events1', 'events2', 'events3', 'CU', 'bank']
setattr(struct_rocprofiler_thread_trace_decoder_perfevent_t, 'time', field(0, int64_t))
setattr(struct_rocprofiler_thread_trace_decoder_perfevent_t, 'events0', field(8, uint16_t))
setattr(struct_rocprofiler_thread_trace_decoder_perfevent_t, 'events1', field(10, uint16_t))
setattr(struct_rocprofiler_thread_trace_decoder_perfevent_t, 'events2', field(12, uint16_t))
setattr(struct_rocprofiler_thread_trace_decoder_perfevent_t, 'events3', field(14, uint16_t))
setattr(struct_rocprofiler_thread_trace_decoder_perfevent_t, 'CU', field(16, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_perfevent_t, 'bank', field(17, uint8_t))
rocprofiler_thread_trace_decoder_perfevent_t = struct_rocprofiler_thread_trace_decoder_perfevent_t
class struct_rocprofiler_thread_trace_decoder_occupancy_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_occupancy_t.SIZE = 32
struct_rocprofiler_thread_trace_decoder_occupancy_t._fields_ = ['pc', 'time', 'reserved', 'cu', 'simd', 'wave_id', 'start', '_rsvd']
setattr(struct_rocprofiler_thread_trace_decoder_occupancy_t, 'pc', field(0, rocprofiler_thread_trace_decoder_pc_t))
setattr(struct_rocprofiler_thread_trace_decoder_occupancy_t, 'time', field(16, uint64_t))
setattr(struct_rocprofiler_thread_trace_decoder_occupancy_t, 'reserved', field(24, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_occupancy_t, 'cu', field(25, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_occupancy_t, 'simd', field(26, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_occupancy_t, 'wave_id', field(27, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_occupancy_t, 'start', field(28, uint32_t, 1, 0))
setattr(struct_rocprofiler_thread_trace_decoder_occupancy_t, '_rsvd', field(28, uint32_t, 31, 1))
rocprofiler_thread_trace_decoder_occupancy_t = struct_rocprofiler_thread_trace_decoder_occupancy_t
enum_rocprofiler_thread_trace_decoder_wstate_type_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY', 0)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE', 1)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC', 2)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT', 3)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL', 4)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST', 5)

rocprofiler_thread_trace_decoder_wstate_type_t = enum_rocprofiler_thread_trace_decoder_wstate_type_t
class struct_rocprofiler_thread_trace_decoder_wave_state_t(Struct): pass
int32_t = ctypes.c_int32
struct_rocprofiler_thread_trace_decoder_wave_state_t.SIZE = 8
struct_rocprofiler_thread_trace_decoder_wave_state_t._fields_ = ['type', 'duration']
setattr(struct_rocprofiler_thread_trace_decoder_wave_state_t, 'type', field(0, int32_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_state_t, 'duration', field(4, int32_t))
rocprofiler_thread_trace_decoder_wave_state_t = struct_rocprofiler_thread_trace_decoder_wave_state_t
enum_rocprofiler_thread_trace_decoder_inst_category_t = CEnum(ctypes.c_uint32)
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

rocprofiler_thread_trace_decoder_inst_category_t = enum_rocprofiler_thread_trace_decoder_inst_category_t
class struct_rocprofiler_thread_trace_decoder_inst_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_inst_t.SIZE = 32
struct_rocprofiler_thread_trace_decoder_inst_t._fields_ = ['category', 'stall', 'duration', 'time', 'pc']
setattr(struct_rocprofiler_thread_trace_decoder_inst_t, 'category', field(0, uint32_t, 8, 0))
setattr(struct_rocprofiler_thread_trace_decoder_inst_t, 'stall', field(1, uint32_t, 24, 0))
setattr(struct_rocprofiler_thread_trace_decoder_inst_t, 'duration', field(4, int32_t))
setattr(struct_rocprofiler_thread_trace_decoder_inst_t, 'time', field(8, int64_t))
setattr(struct_rocprofiler_thread_trace_decoder_inst_t, 'pc', field(16, rocprofiler_thread_trace_decoder_pc_t))
rocprofiler_thread_trace_decoder_inst_t = struct_rocprofiler_thread_trace_decoder_inst_t
class struct_rocprofiler_thread_trace_decoder_wave_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_wave_t.SIZE = 64
struct_rocprofiler_thread_trace_decoder_wave_t._fields_ = ['cu', 'simd', 'wave_id', 'contexts', '_rsvd1', '_rsvd2', '_rsvd3', 'begin_time', 'end_time', 'timeline_size', 'instructions_size', 'timeline_array', 'instructions_array']
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'cu', field(0, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'simd', field(1, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'wave_id', field(2, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'contexts', field(3, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, '_rsvd1', field(4, uint32_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, '_rsvd2', field(8, uint32_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, '_rsvd3', field(12, uint32_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'begin_time', field(16, int64_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'end_time', field(24, int64_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'timeline_size', field(32, uint64_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'instructions_size', field(40, uint64_t))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'timeline_array', field(48, Pointer(rocprofiler_thread_trace_decoder_wave_state_t)))
setattr(struct_rocprofiler_thread_trace_decoder_wave_t, 'instructions_array', field(56, Pointer(rocprofiler_thread_trace_decoder_inst_t)))
rocprofiler_thread_trace_decoder_wave_t = struct_rocprofiler_thread_trace_decoder_wave_t
class struct_rocprofiler_thread_trace_decoder_realtime_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_realtime_t.SIZE = 24
struct_rocprofiler_thread_trace_decoder_realtime_t._fields_ = ['shader_clock', 'realtime_clock', 'reserved']
setattr(struct_rocprofiler_thread_trace_decoder_realtime_t, 'shader_clock', field(0, int64_t))
setattr(struct_rocprofiler_thread_trace_decoder_realtime_t, 'realtime_clock', field(8, uint64_t))
setattr(struct_rocprofiler_thread_trace_decoder_realtime_t, 'reserved', field(16, uint64_t))
rocprofiler_thread_trace_decoder_realtime_t = struct_rocprofiler_thread_trace_decoder_realtime_t
enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t.define('ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM', 0)
ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t.define('ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV', 1)

rocprofiler_thread_trace_decoder_shaderdata_flags_t = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t
class struct_rocprofiler_thread_trace_decoder_shaderdata_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_shaderdata_t.SIZE = 24
struct_rocprofiler_thread_trace_decoder_shaderdata_t._fields_ = ['time', 'value', 'cu', 'simd', 'wave_id', 'flags', 'reserved']
setattr(struct_rocprofiler_thread_trace_decoder_shaderdata_t, 'time', field(0, int64_t))
setattr(struct_rocprofiler_thread_trace_decoder_shaderdata_t, 'value', field(8, uint64_t))
setattr(struct_rocprofiler_thread_trace_decoder_shaderdata_t, 'cu', field(16, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_shaderdata_t, 'simd', field(17, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_shaderdata_t, 'wave_id', field(18, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_shaderdata_t, 'flags', field(19, uint8_t))
setattr(struct_rocprofiler_thread_trace_decoder_shaderdata_t, 'reserved', field(20, uint32_t))
rocprofiler_thread_trace_decoder_shaderdata_t = struct_rocprofiler_thread_trace_decoder_shaderdata_t
rocprofiler_thread_trace_decoder_record_type_t = enum_rocprofiler_thread_trace_decoder_record_type_t
