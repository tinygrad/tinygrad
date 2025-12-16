# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
class struct_sqtt_data_info(Struct): pass
uint32_t = ctypes.c_uint32
struct_sqtt_data_info.SIZE = 12
struct_sqtt_data_info._fields_ = ['cur_offset', 'trace_status', 'gfx9_write_counter', 'gfx10_dropped_cntr']
setattr(struct_sqtt_data_info, 'cur_offset', field(0, uint32_t))
setattr(struct_sqtt_data_info, 'trace_status', field(4, uint32_t))
setattr(struct_sqtt_data_info, 'gfx9_write_counter', field(8, uint32_t))
setattr(struct_sqtt_data_info, 'gfx10_dropped_cntr', field(8, uint32_t))
class struct_sqtt_data_se(Struct): pass
struct_sqtt_data_se.SIZE = 32
struct_sqtt_data_se._fields_ = ['info', 'data_ptr', 'shader_engine', 'compute_unit']
setattr(struct_sqtt_data_se, 'info', field(0, struct_sqtt_data_info))
setattr(struct_sqtt_data_se, 'data_ptr', field(16, ctypes.c_void_p))
setattr(struct_sqtt_data_se, 'shader_engine', field(24, uint32_t))
setattr(struct_sqtt_data_se, 'compute_unit', field(28, uint32_t))
enum_sqtt_version = CEnum(ctypes.c_uint32)
SQTT_VERSION_NONE = enum_sqtt_version.define('SQTT_VERSION_NONE', 0)
SQTT_VERSION_2_2 = enum_sqtt_version.define('SQTT_VERSION_2_2', 5)
SQTT_VERSION_2_3 = enum_sqtt_version.define('SQTT_VERSION_2_3', 6)
SQTT_VERSION_2_4 = enum_sqtt_version.define('SQTT_VERSION_2_4', 7)
SQTT_VERSION_3_2 = enum_sqtt_version.define('SQTT_VERSION_3_2', 11)
SQTT_VERSION_3_3 = enum_sqtt_version.define('SQTT_VERSION_3_3', 12)

enum_sqtt_file_chunk_type = CEnum(ctypes.c_uint32)
SQTT_FILE_CHUNK_TYPE_ASIC_INFO = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_ASIC_INFO', 0)
SQTT_FILE_CHUNK_TYPE_SQTT_DESC = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_SQTT_DESC', 1)
SQTT_FILE_CHUNK_TYPE_SQTT_DATA = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_SQTT_DATA', 2)
SQTT_FILE_CHUNK_TYPE_API_INFO = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_API_INFO', 3)
SQTT_FILE_CHUNK_TYPE_RESERVED = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_RESERVED', 4)
SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS', 5)
SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION', 6)
SQTT_FILE_CHUNK_TYPE_CPU_INFO = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_CPU_INFO', 7)
SQTT_FILE_CHUNK_TYPE_SPM_DB = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_SPM_DB', 8)
SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE', 9)
SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS', 10)
SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION', 11)
SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE', 12)
SQTT_FILE_CHUNK_TYPE_COUNT = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_COUNT', 13)

class struct_sqtt_file_chunk_id(Struct): pass
int32_t = ctypes.c_int32
struct_sqtt_file_chunk_id.SIZE = 4
struct_sqtt_file_chunk_id._fields_ = ['type', 'index', 'reserved']
setattr(struct_sqtt_file_chunk_id, 'type', field(0, int32_t, 8, 0))
setattr(struct_sqtt_file_chunk_id, 'index', field(1, int32_t, 8, 0))
setattr(struct_sqtt_file_chunk_id, 'reserved', field(2, int32_t, 16, 0))
class struct_sqtt_file_chunk_header(Struct): pass
uint16_t = ctypes.c_uint16
struct_sqtt_file_chunk_header.SIZE = 16
struct_sqtt_file_chunk_header._fields_ = ['chunk_id', 'minor_version', 'major_version', 'size_in_bytes', 'padding']
setattr(struct_sqtt_file_chunk_header, 'chunk_id', field(0, struct_sqtt_file_chunk_id))
setattr(struct_sqtt_file_chunk_header, 'minor_version', field(4, uint16_t))
setattr(struct_sqtt_file_chunk_header, 'major_version', field(6, uint16_t))
setattr(struct_sqtt_file_chunk_header, 'size_in_bytes', field(8, int32_t))
setattr(struct_sqtt_file_chunk_header, 'padding', field(12, int32_t))
class struct_sqtt_file_header_flags(Struct): pass
struct_sqtt_file_header_flags.SIZE = 4
struct_sqtt_file_header_flags._fields_ = ['is_semaphore_queue_timing_etw', 'no_queue_semaphore_timestamps', 'reserved', 'value']
setattr(struct_sqtt_file_header_flags, 'is_semaphore_queue_timing_etw', field(0, uint32_t, 1, 0))
setattr(struct_sqtt_file_header_flags, 'no_queue_semaphore_timestamps', field(0, uint32_t, 1, 1))
setattr(struct_sqtt_file_header_flags, 'reserved', field(0, uint32_t, 30, 2))
setattr(struct_sqtt_file_header_flags, 'value', field(0, uint32_t))
class struct_sqtt_file_header(Struct): pass
struct_sqtt_file_header.SIZE = 56
struct_sqtt_file_header._fields_ = ['magic_number', 'version_major', 'version_minor', 'flags', 'chunk_offset', 'second', 'minute', 'hour', 'day_in_month', 'month', 'year', 'day_in_week', 'day_in_year', 'is_daylight_savings']
setattr(struct_sqtt_file_header, 'magic_number', field(0, uint32_t))
setattr(struct_sqtt_file_header, 'version_major', field(4, uint32_t))
setattr(struct_sqtt_file_header, 'version_minor', field(8, uint32_t))
setattr(struct_sqtt_file_header, 'flags', field(12, struct_sqtt_file_header_flags))
setattr(struct_sqtt_file_header, 'chunk_offset', field(16, int32_t))
setattr(struct_sqtt_file_header, 'second', field(20, int32_t))
setattr(struct_sqtt_file_header, 'minute', field(24, int32_t))
setattr(struct_sqtt_file_header, 'hour', field(28, int32_t))
setattr(struct_sqtt_file_header, 'day_in_month', field(32, int32_t))
setattr(struct_sqtt_file_header, 'month', field(36, int32_t))
setattr(struct_sqtt_file_header, 'year', field(40, int32_t))
setattr(struct_sqtt_file_header, 'day_in_week', field(44, int32_t))
setattr(struct_sqtt_file_header, 'day_in_year', field(48, int32_t))
setattr(struct_sqtt_file_header, 'is_daylight_savings', field(52, int32_t))
class struct_sqtt_file_chunk_cpu_info(Struct): pass
uint64_t = ctypes.c_uint64
struct_sqtt_file_chunk_cpu_info.SIZE = 112
struct_sqtt_file_chunk_cpu_info._fields_ = ['header', 'vendor_id', 'processor_brand', 'reserved', 'cpu_timestamp_freq', 'clock_speed', 'num_logical_cores', 'num_physical_cores', 'system_ram_size']
setattr(struct_sqtt_file_chunk_cpu_info, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_cpu_info, 'vendor_id', field(16, Array(uint32_t, 4)))
setattr(struct_sqtt_file_chunk_cpu_info, 'processor_brand', field(32, Array(uint32_t, 12)))
setattr(struct_sqtt_file_chunk_cpu_info, 'reserved', field(80, Array(uint32_t, 2)))
setattr(struct_sqtt_file_chunk_cpu_info, 'cpu_timestamp_freq', field(88, uint64_t))
setattr(struct_sqtt_file_chunk_cpu_info, 'clock_speed', field(96, uint32_t))
setattr(struct_sqtt_file_chunk_cpu_info, 'num_logical_cores', field(100, uint32_t))
setattr(struct_sqtt_file_chunk_cpu_info, 'num_physical_cores', field(104, uint32_t))
setattr(struct_sqtt_file_chunk_cpu_info, 'system_ram_size', field(108, uint32_t))
enum_sqtt_file_chunk_asic_info_flags = CEnum(ctypes.c_uint32)
SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING = enum_sqtt_file_chunk_asic_info_flags.define('SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING', 1)
SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED = enum_sqtt_file_chunk_asic_info_flags.define('SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED', 2)

enum_sqtt_gpu_type = CEnum(ctypes.c_uint32)
SQTT_GPU_TYPE_UNKNOWN = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_UNKNOWN', 0)
SQTT_GPU_TYPE_INTEGRATED = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_INTEGRATED', 1)
SQTT_GPU_TYPE_DISCRETE = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_DISCRETE', 2)
SQTT_GPU_TYPE_VIRTUAL = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_VIRTUAL', 3)

enum_sqtt_gfxip_level = CEnum(ctypes.c_uint32)
SQTT_GFXIP_LEVEL_NONE = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_NONE', 0)
SQTT_GFXIP_LEVEL_GFXIP_6 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_6', 1)
SQTT_GFXIP_LEVEL_GFXIP_7 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_7', 2)
SQTT_GFXIP_LEVEL_GFXIP_8 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_8', 3)
SQTT_GFXIP_LEVEL_GFXIP_8_1 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_8_1', 4)
SQTT_GFXIP_LEVEL_GFXIP_9 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_9', 5)
SQTT_GFXIP_LEVEL_GFXIP_10_1 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_10_1', 7)
SQTT_GFXIP_LEVEL_GFXIP_10_3 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_10_3', 9)
SQTT_GFXIP_LEVEL_GFXIP_11_0 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_11_0', 12)
SQTT_GFXIP_LEVEL_GFXIP_11_5 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_11_5', 13)
SQTT_GFXIP_LEVEL_GFXIP_12 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_12', 16)

enum_sqtt_memory_type = CEnum(ctypes.c_uint32)
SQTT_MEMORY_TYPE_UNKNOWN = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_UNKNOWN', 0)
SQTT_MEMORY_TYPE_DDR = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR', 1)
SQTT_MEMORY_TYPE_DDR2 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR2', 2)
SQTT_MEMORY_TYPE_DDR3 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR3', 3)
SQTT_MEMORY_TYPE_DDR4 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR4', 4)
SQTT_MEMORY_TYPE_DDR5 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR5', 5)
SQTT_MEMORY_TYPE_GDDR3 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_GDDR3', 16)
SQTT_MEMORY_TYPE_GDDR4 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_GDDR4', 17)
SQTT_MEMORY_TYPE_GDDR5 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_GDDR5', 18)
SQTT_MEMORY_TYPE_GDDR6 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_GDDR6', 19)
SQTT_MEMORY_TYPE_HBM = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_HBM', 32)
SQTT_MEMORY_TYPE_HBM2 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_HBM2', 33)
SQTT_MEMORY_TYPE_HBM3 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_HBM3', 34)
SQTT_MEMORY_TYPE_LPDDR4 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_LPDDR4', 48)
SQTT_MEMORY_TYPE_LPDDR5 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_LPDDR5', 49)

class struct_sqtt_file_chunk_asic_info(Struct): pass
int64_t = ctypes.c_int64
struct_sqtt_file_chunk_asic_info.SIZE = 768
struct_sqtt_file_chunk_asic_info._fields_ = ['header', 'flags', 'trace_shader_core_clock', 'trace_memory_clock', 'device_id', 'device_revision_id', 'vgprs_per_simd', 'sgprs_per_simd', 'shader_engines', 'compute_unit_per_shader_engine', 'simd_per_compute_unit', 'wavefronts_per_simd', 'minimum_vgpr_alloc', 'vgpr_alloc_granularity', 'minimum_sgpr_alloc', 'sgpr_alloc_granularity', 'hardware_contexts', 'gpu_type', 'gfxip_level', 'gpu_index', 'gds_size', 'gds_per_shader_engine', 'ce_ram_size', 'ce_ram_size_graphics', 'ce_ram_size_compute', 'max_number_of_dedicated_cus', 'vram_size', 'vram_bus_width', 'l2_cache_size', 'l1_cache_size', 'lds_size', 'gpu_name', 'alu_per_clock', 'texture_per_clock', 'prims_per_clock', 'pixels_per_clock', 'gpu_timestamp_frequency', 'max_shader_core_clock', 'max_memory_clock', 'memory_ops_per_clock', 'memory_chip_type', 'lds_granularity', 'cu_mask', 'reserved1', 'active_pixel_packer_mask', 'reserved2', 'gl1_cache_size', 'instruction_cache_size', 'scalar_cache_size', 'mall_cache_size', 'padding']
setattr(struct_sqtt_file_chunk_asic_info, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_asic_info, 'flags', field(16, uint64_t))
setattr(struct_sqtt_file_chunk_asic_info, 'trace_shader_core_clock', field(24, uint64_t))
setattr(struct_sqtt_file_chunk_asic_info, 'trace_memory_clock', field(32, uint64_t))
setattr(struct_sqtt_file_chunk_asic_info, 'device_id', field(40, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'device_revision_id', field(44, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'vgprs_per_simd', field(48, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'sgprs_per_simd', field(52, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'shader_engines', field(56, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'compute_unit_per_shader_engine', field(60, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'simd_per_compute_unit', field(64, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'wavefronts_per_simd', field(68, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'minimum_vgpr_alloc', field(72, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'vgpr_alloc_granularity', field(76, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'minimum_sgpr_alloc', field(80, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'sgpr_alloc_granularity', field(84, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'hardware_contexts', field(88, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'gpu_type', field(92, enum_sqtt_gpu_type))
setattr(struct_sqtt_file_chunk_asic_info, 'gfxip_level', field(96, enum_sqtt_gfxip_level))
setattr(struct_sqtt_file_chunk_asic_info, 'gpu_index', field(100, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'gds_size', field(104, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'gds_per_shader_engine', field(108, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'ce_ram_size', field(112, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'ce_ram_size_graphics', field(116, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'ce_ram_size_compute', field(120, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'max_number_of_dedicated_cus', field(124, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'vram_size', field(128, int64_t))
setattr(struct_sqtt_file_chunk_asic_info, 'vram_bus_width', field(136, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'l2_cache_size', field(140, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'l1_cache_size', field(144, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'lds_size', field(148, int32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'gpu_name', field(152, Array(ctypes.c_char, 256)))
setattr(struct_sqtt_file_chunk_asic_info, 'alu_per_clock', field(408, ctypes.c_float))
setattr(struct_sqtt_file_chunk_asic_info, 'texture_per_clock', field(412, ctypes.c_float))
setattr(struct_sqtt_file_chunk_asic_info, 'prims_per_clock', field(416, ctypes.c_float))
setattr(struct_sqtt_file_chunk_asic_info, 'pixels_per_clock', field(420, ctypes.c_float))
setattr(struct_sqtt_file_chunk_asic_info, 'gpu_timestamp_frequency', field(424, uint64_t))
setattr(struct_sqtt_file_chunk_asic_info, 'max_shader_core_clock', field(432, uint64_t))
setattr(struct_sqtt_file_chunk_asic_info, 'max_memory_clock', field(440, uint64_t))
setattr(struct_sqtt_file_chunk_asic_info, 'memory_ops_per_clock', field(448, uint32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'memory_chip_type', field(452, enum_sqtt_memory_type))
setattr(struct_sqtt_file_chunk_asic_info, 'lds_granularity', field(456, uint32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'cu_mask', field(460, Array(Array(uint16_t, 2), 32)))
setattr(struct_sqtt_file_chunk_asic_info, 'reserved1', field(588, Array(ctypes.c_char, 128)))
setattr(struct_sqtt_file_chunk_asic_info, 'active_pixel_packer_mask', field(716, Array(uint32_t, 4)))
setattr(struct_sqtt_file_chunk_asic_info, 'reserved2', field(732, Array(ctypes.c_char, 16)))
setattr(struct_sqtt_file_chunk_asic_info, 'gl1_cache_size', field(748, uint32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'instruction_cache_size', field(752, uint32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'scalar_cache_size', field(756, uint32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'mall_cache_size', field(760, uint32_t))
setattr(struct_sqtt_file_chunk_asic_info, 'padding', field(764, Array(ctypes.c_char, 4)))
enum_sqtt_api_type = CEnum(ctypes.c_uint32)
SQTT_API_TYPE_DIRECTX_12 = enum_sqtt_api_type.define('SQTT_API_TYPE_DIRECTX_12', 0)
SQTT_API_TYPE_VULKAN = enum_sqtt_api_type.define('SQTT_API_TYPE_VULKAN', 1)
SQTT_API_TYPE_GENERIC = enum_sqtt_api_type.define('SQTT_API_TYPE_GENERIC', 2)
SQTT_API_TYPE_OPENCL = enum_sqtt_api_type.define('SQTT_API_TYPE_OPENCL', 3)

enum_sqtt_instruction_trace_mode = CEnum(ctypes.c_uint32)
SQTT_INSTRUCTION_TRACE_DISABLED = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_DISABLED', 0)
SQTT_INSTRUCTION_TRACE_FULL_FRAME = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_FULL_FRAME', 1)
SQTT_INSTRUCTION_TRACE_API_PSO = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_API_PSO', 2)

enum_sqtt_profiling_mode = CEnum(ctypes.c_uint32)
SQTT_PROFILING_MODE_PRESENT = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_PRESENT', 0)
SQTT_PROFILING_MODE_USER_MARKERS = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_USER_MARKERS', 1)
SQTT_PROFILING_MODE_INDEX = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_INDEX', 2)
SQTT_PROFILING_MODE_TAG = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_TAG', 3)

class union_sqtt_profiling_mode_data(Union): pass
class _anonstruct0(Struct): pass
_anonstruct0.SIZE = 512
_anonstruct0._fields_ = ['start', 'end']
setattr(_anonstruct0, 'start', field(0, Array(ctypes.c_char, 256)))
setattr(_anonstruct0, 'end', field(256, Array(ctypes.c_char, 256)))
class _anonstruct1(Struct): pass
_anonstruct1.SIZE = 8
_anonstruct1._fields_ = ['start', 'end']
setattr(_anonstruct1, 'start', field(0, uint32_t))
setattr(_anonstruct1, 'end', field(4, uint32_t))
class _anonstruct2(Struct): pass
_anonstruct2.SIZE = 16
_anonstruct2._fields_ = ['begin_hi', 'begin_lo', 'end_hi', 'end_lo']
setattr(_anonstruct2, 'begin_hi', field(0, uint32_t))
setattr(_anonstruct2, 'begin_lo', field(4, uint32_t))
setattr(_anonstruct2, 'end_hi', field(8, uint32_t))
setattr(_anonstruct2, 'end_lo', field(12, uint32_t))
union_sqtt_profiling_mode_data.SIZE = 512
union_sqtt_profiling_mode_data._fields_ = ['user_marker_profiling_data', 'index_profiling_data', 'tag_profiling_data']
setattr(union_sqtt_profiling_mode_data, 'user_marker_profiling_data', field(0, _anonstruct0))
setattr(union_sqtt_profiling_mode_data, 'index_profiling_data', field(0, _anonstruct1))
setattr(union_sqtt_profiling_mode_data, 'tag_profiling_data', field(0, _anonstruct2))
class union_sqtt_instruction_trace_data(Union): pass
class _anonstruct3(Struct): pass
_anonstruct3.SIZE = 8
_anonstruct3._fields_ = ['api_pso_filter']
setattr(_anonstruct3, 'api_pso_filter', field(0, uint64_t))
class _anonstruct4(Struct): pass
_anonstruct4.SIZE = 4
_anonstruct4._fields_ = ['mask']
setattr(_anonstruct4, 'mask', field(0, uint32_t))
union_sqtt_instruction_trace_data.SIZE = 8
union_sqtt_instruction_trace_data._fields_ = ['api_pso_data', 'shader_engine_filter']
setattr(union_sqtt_instruction_trace_data, 'api_pso_data', field(0, _anonstruct3))
setattr(union_sqtt_instruction_trace_data, 'shader_engine_filter', field(0, _anonstruct4))
class struct_sqtt_file_chunk_api_info(Struct): pass
struct_sqtt_file_chunk_api_info.SIZE = 560
struct_sqtt_file_chunk_api_info._fields_ = ['header', 'api_type', 'major_version', 'minor_version', 'profiling_mode', 'reserved', 'profiling_mode_data', 'instruction_trace_mode', 'reserved2', 'instruction_trace_data']
setattr(struct_sqtt_file_chunk_api_info, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_api_info, 'api_type', field(16, enum_sqtt_api_type))
setattr(struct_sqtt_file_chunk_api_info, 'major_version', field(20, uint16_t))
setattr(struct_sqtt_file_chunk_api_info, 'minor_version', field(22, uint16_t))
setattr(struct_sqtt_file_chunk_api_info, 'profiling_mode', field(24, enum_sqtt_profiling_mode))
setattr(struct_sqtt_file_chunk_api_info, 'reserved', field(28, uint32_t))
setattr(struct_sqtt_file_chunk_api_info, 'profiling_mode_data', field(32, union_sqtt_profiling_mode_data))
setattr(struct_sqtt_file_chunk_api_info, 'instruction_trace_mode', field(544, enum_sqtt_instruction_trace_mode))
setattr(struct_sqtt_file_chunk_api_info, 'reserved2', field(548, uint32_t))
setattr(struct_sqtt_file_chunk_api_info, 'instruction_trace_data', field(552, union_sqtt_instruction_trace_data))
class struct_sqtt_code_object_database_record(Struct): pass
struct_sqtt_code_object_database_record.SIZE = 4
struct_sqtt_code_object_database_record._fields_ = ['size']
setattr(struct_sqtt_code_object_database_record, 'size', field(0, uint32_t))
class struct_sqtt_file_chunk_code_object_database(Struct): pass
struct_sqtt_file_chunk_code_object_database.SIZE = 32
struct_sqtt_file_chunk_code_object_database._fields_ = ['header', 'offset', 'flags', 'size', 'record_count']
setattr(struct_sqtt_file_chunk_code_object_database, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_code_object_database, 'offset', field(16, uint32_t))
setattr(struct_sqtt_file_chunk_code_object_database, 'flags', field(20, uint32_t))
setattr(struct_sqtt_file_chunk_code_object_database, 'size', field(24, uint32_t))
setattr(struct_sqtt_file_chunk_code_object_database, 'record_count', field(28, uint32_t))
class struct_sqtt_code_object_loader_events_record(Struct): pass
struct_sqtt_code_object_loader_events_record.SIZE = 40
struct_sqtt_code_object_loader_events_record._fields_ = ['loader_event_type', 'reserved', 'base_address', 'code_object_hash', 'time_stamp']
setattr(struct_sqtt_code_object_loader_events_record, 'loader_event_type', field(0, uint32_t))
setattr(struct_sqtt_code_object_loader_events_record, 'reserved', field(4, uint32_t))
setattr(struct_sqtt_code_object_loader_events_record, 'base_address', field(8, uint64_t))
setattr(struct_sqtt_code_object_loader_events_record, 'code_object_hash', field(16, Array(uint64_t, 2)))
setattr(struct_sqtt_code_object_loader_events_record, 'time_stamp', field(32, uint64_t))
class struct_sqtt_file_chunk_code_object_loader_events(Struct): pass
struct_sqtt_file_chunk_code_object_loader_events.SIZE = 32
struct_sqtt_file_chunk_code_object_loader_events._fields_ = ['header', 'offset', 'flags', 'record_size', 'record_count']
setattr(struct_sqtt_file_chunk_code_object_loader_events, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_code_object_loader_events, 'offset', field(16, uint32_t))
setattr(struct_sqtt_file_chunk_code_object_loader_events, 'flags', field(20, uint32_t))
setattr(struct_sqtt_file_chunk_code_object_loader_events, 'record_size', field(24, uint32_t))
setattr(struct_sqtt_file_chunk_code_object_loader_events, 'record_count', field(28, uint32_t))
class struct_sqtt_pso_correlation_record(Struct): pass
struct_sqtt_pso_correlation_record.SIZE = 88
struct_sqtt_pso_correlation_record._fields_ = ['api_pso_hash', 'pipeline_hash', 'api_level_obj_name']
setattr(struct_sqtt_pso_correlation_record, 'api_pso_hash', field(0, uint64_t))
setattr(struct_sqtt_pso_correlation_record, 'pipeline_hash', field(8, Array(uint64_t, 2)))
setattr(struct_sqtt_pso_correlation_record, 'api_level_obj_name', field(24, Array(ctypes.c_char, 64)))
class struct_sqtt_file_chunk_pso_correlation(Struct): pass
struct_sqtt_file_chunk_pso_correlation.SIZE = 32
struct_sqtt_file_chunk_pso_correlation._fields_ = ['header', 'offset', 'flags', 'record_size', 'record_count']
setattr(struct_sqtt_file_chunk_pso_correlation, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_pso_correlation, 'offset', field(16, uint32_t))
setattr(struct_sqtt_file_chunk_pso_correlation, 'flags', field(20, uint32_t))
setattr(struct_sqtt_file_chunk_pso_correlation, 'record_size', field(24, uint32_t))
setattr(struct_sqtt_file_chunk_pso_correlation, 'record_count', field(28, uint32_t))
class struct_sqtt_file_chunk_sqtt_desc(Struct): pass
class _anonstruct5(Struct): pass
_anonstruct5.SIZE = 4
_anonstruct5._fields_ = ['instrumentation_version']
setattr(_anonstruct5, 'instrumentation_version', field(0, int32_t))
class _anonstruct6(Struct): pass
int16_t = ctypes.c_int16
_anonstruct6.SIZE = 8
_anonstruct6._fields_ = ['instrumentation_spec_version', 'instrumentation_api_version', 'compute_unit_index']
setattr(_anonstruct6, 'instrumentation_spec_version', field(0, int16_t))
setattr(_anonstruct6, 'instrumentation_api_version', field(2, int16_t))
setattr(_anonstruct6, 'compute_unit_index', field(4, int32_t))
struct_sqtt_file_chunk_sqtt_desc.SIZE = 32
struct_sqtt_file_chunk_sqtt_desc._fields_ = ['header', 'shader_engine_index', 'sqtt_version', 'v0', 'v1']
setattr(struct_sqtt_file_chunk_sqtt_desc, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_sqtt_desc, 'shader_engine_index', field(16, int32_t))
setattr(struct_sqtt_file_chunk_sqtt_desc, 'sqtt_version', field(20, enum_sqtt_version))
setattr(struct_sqtt_file_chunk_sqtt_desc, 'v0', field(24, _anonstruct5))
setattr(struct_sqtt_file_chunk_sqtt_desc, 'v1', field(24, _anonstruct6))
class struct_sqtt_file_chunk_sqtt_data(Struct): pass
struct_sqtt_file_chunk_sqtt_data.SIZE = 24
struct_sqtt_file_chunk_sqtt_data._fields_ = ['header', 'offset', 'size']
setattr(struct_sqtt_file_chunk_sqtt_data, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_sqtt_data, 'offset', field(16, int32_t))
setattr(struct_sqtt_file_chunk_sqtt_data, 'size', field(20, int32_t))
class struct_sqtt_file_chunk_queue_event_timings(Struct): pass
struct_sqtt_file_chunk_queue_event_timings.SIZE = 32
struct_sqtt_file_chunk_queue_event_timings._fields_ = ['header', 'queue_info_table_record_count', 'queue_info_table_size', 'queue_event_table_record_count', 'queue_event_table_size']
setattr(struct_sqtt_file_chunk_queue_event_timings, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_queue_event_timings, 'queue_info_table_record_count', field(16, uint32_t))
setattr(struct_sqtt_file_chunk_queue_event_timings, 'queue_info_table_size', field(20, uint32_t))
setattr(struct_sqtt_file_chunk_queue_event_timings, 'queue_event_table_record_count', field(24, uint32_t))
setattr(struct_sqtt_file_chunk_queue_event_timings, 'queue_event_table_size', field(28, uint32_t))
enum_sqtt_queue_type = CEnum(ctypes.c_uint32)
SQTT_QUEUE_TYPE_UNKNOWN = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_UNKNOWN', 0)
SQTT_QUEUE_TYPE_UNIVERSAL = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_UNIVERSAL', 1)
SQTT_QUEUE_TYPE_COMPUTE = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_COMPUTE', 2)
SQTT_QUEUE_TYPE_DMA = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_DMA', 3)

enum_sqtt_engine_type = CEnum(ctypes.c_uint32)
SQTT_ENGINE_TYPE_UNKNOWN = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_UNKNOWN', 0)
SQTT_ENGINE_TYPE_UNIVERSAL = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_UNIVERSAL', 1)
SQTT_ENGINE_TYPE_COMPUTE = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_COMPUTE', 2)
SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE', 3)
SQTT_ENGINE_TYPE_DMA = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_DMA', 4)
SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL', 7)
SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS', 8)

class struct_sqtt_queue_hardware_info(Struct): pass
struct_sqtt_queue_hardware_info.SIZE = 4
struct_sqtt_queue_hardware_info._fields_ = ['queue_type', 'engine_type', 'reserved', 'value']
setattr(struct_sqtt_queue_hardware_info, 'queue_type', field(0, int32_t, 8, 0))
setattr(struct_sqtt_queue_hardware_info, 'engine_type', field(1, int32_t, 8, 0))
setattr(struct_sqtt_queue_hardware_info, 'reserved', field(2, uint32_t, 16, 0))
setattr(struct_sqtt_queue_hardware_info, 'value', field(0, uint32_t))
class struct_sqtt_queue_info_record(Struct): pass
struct_sqtt_queue_info_record.SIZE = 24
struct_sqtt_queue_info_record._fields_ = ['queue_id', 'queue_context', 'hardware_info', 'reserved']
setattr(struct_sqtt_queue_info_record, 'queue_id', field(0, uint64_t))
setattr(struct_sqtt_queue_info_record, 'queue_context', field(8, uint64_t))
setattr(struct_sqtt_queue_info_record, 'hardware_info', field(16, struct_sqtt_queue_hardware_info))
setattr(struct_sqtt_queue_info_record, 'reserved', field(20, uint32_t))
enum_sqtt_queue_event_type = CEnum(ctypes.c_uint32)
SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT', 0)
SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE', 1)
SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE', 2)
SQTT_QUEUE_TIMING_EVENT_PRESENT = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_PRESENT', 3)

class struct_sqtt_queue_event_record(Struct): pass
struct_sqtt_queue_event_record.SIZE = 56
struct_sqtt_queue_event_record._fields_ = ['event_type', 'sqtt_cb_id', 'frame_index', 'queue_info_index', 'submit_sub_index', 'api_id', 'cpu_timestamp', 'gpu_timestamps']
setattr(struct_sqtt_queue_event_record, 'event_type', field(0, enum_sqtt_queue_event_type))
setattr(struct_sqtt_queue_event_record, 'sqtt_cb_id', field(4, uint32_t))
setattr(struct_sqtt_queue_event_record, 'frame_index', field(8, uint64_t))
setattr(struct_sqtt_queue_event_record, 'queue_info_index', field(16, uint32_t))
setattr(struct_sqtt_queue_event_record, 'submit_sub_index', field(20, uint32_t))
setattr(struct_sqtt_queue_event_record, 'api_id', field(24, uint64_t))
setattr(struct_sqtt_queue_event_record, 'cpu_timestamp', field(32, uint64_t))
setattr(struct_sqtt_queue_event_record, 'gpu_timestamps', field(40, Array(uint64_t, 2)))
class struct_sqtt_file_chunk_clock_calibration(Struct): pass
struct_sqtt_file_chunk_clock_calibration.SIZE = 40
struct_sqtt_file_chunk_clock_calibration._fields_ = ['header', 'cpu_timestamp', 'gpu_timestamp', 'reserved']
setattr(struct_sqtt_file_chunk_clock_calibration, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_clock_calibration, 'cpu_timestamp', field(16, uint64_t))
setattr(struct_sqtt_file_chunk_clock_calibration, 'gpu_timestamp', field(24, uint64_t))
setattr(struct_sqtt_file_chunk_clock_calibration, 'reserved', field(32, uint64_t))
enum_elf_gfxip_level = CEnum(ctypes.c_uint32)
EF_AMDGPU_MACH_AMDGCN_GFX801 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX801', 40)
EF_AMDGPU_MACH_AMDGCN_GFX900 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX900', 44)
EF_AMDGPU_MACH_AMDGCN_GFX1010 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1010', 51)
EF_AMDGPU_MACH_AMDGCN_GFX1030 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1030', 54)
EF_AMDGPU_MACH_AMDGCN_GFX1100 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1100', 65)
EF_AMDGPU_MACH_AMDGCN_GFX1150 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1150', 67)
EF_AMDGPU_MACH_AMDGCN_GFX1200 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1200', 78)

class struct_sqtt_file_chunk_spm_db(Struct): pass
struct_sqtt_file_chunk_spm_db.SIZE = 40
struct_sqtt_file_chunk_spm_db._fields_ = ['header', 'flags', 'preamble_size', 'num_timestamps', 'num_spm_counter_info', 'spm_counter_info_size', 'sample_interval']
setattr(struct_sqtt_file_chunk_spm_db, 'header', field(0, struct_sqtt_file_chunk_header))
setattr(struct_sqtt_file_chunk_spm_db, 'flags', field(16, uint32_t))
setattr(struct_sqtt_file_chunk_spm_db, 'preamble_size', field(20, uint32_t))
setattr(struct_sqtt_file_chunk_spm_db, 'num_timestamps', field(24, uint32_t))
setattr(struct_sqtt_file_chunk_spm_db, 'num_spm_counter_info', field(28, uint32_t))
setattr(struct_sqtt_file_chunk_spm_db, 'spm_counter_info_size', field(32, uint32_t))
setattr(struct_sqtt_file_chunk_spm_db, 'sample_interval', field(36, uint32_t))
enum_rgp_sqtt_marker_identifier = CEnum(ctypes.c_uint32)
RGP_SQTT_MARKER_IDENTIFIER_EVENT = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_EVENT', 0)
RGP_SQTT_MARKER_IDENTIFIER_CB_START = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_CB_START', 1)
RGP_SQTT_MARKER_IDENTIFIER_CB_END = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_CB_END', 2)
RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START', 3)
RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END', 4)
RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT', 5)
RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API', 6)
RGP_SQTT_MARKER_IDENTIFIER_SYNC = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_SYNC', 7)
RGP_SQTT_MARKER_IDENTIFIER_PRESENT = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_PRESENT', 8)
RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION', 9)
RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS', 10)
RGP_SQTT_MARKER_IDENTIFIER_RESERVED2 = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RESERVED2', 11)
RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE', 12)
RGP_SQTT_MARKER_IDENTIFIER_RESERVED4 = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RESERVED4', 13)
RGP_SQTT_MARKER_IDENTIFIER_RESERVED5 = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RESERVED5', 14)
RGP_SQTT_MARKER_IDENTIFIER_RESERVED6 = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RESERVED6', 15)

class union_rgp_sqtt_marker_cb_id(Union): pass
class _anonstruct7(Struct): pass
_anonstruct7.SIZE = 4
_anonstruct7._fields_ = ['per_frame', 'frame_index', 'cb_index', 'reserved']
setattr(_anonstruct7, 'per_frame', field(0, uint32_t, 1, 0))
setattr(_anonstruct7, 'frame_index', field(0, uint32_t, 7, 1))
setattr(_anonstruct7, 'cb_index', field(1, uint32_t, 12, 0))
setattr(_anonstruct7, 'reserved', field(2, uint32_t, 12, 4))
class _anonstruct8(Struct): pass
_anonstruct8.SIZE = 4
_anonstruct8._fields_ = ['per_frame', 'cb_index', 'reserved']
setattr(_anonstruct8, 'per_frame', field(0, uint32_t, 1, 0))
setattr(_anonstruct8, 'cb_index', field(0, uint32_t, 19, 1))
setattr(_anonstruct8, 'reserved', field(2, uint32_t, 12, 4))
union_rgp_sqtt_marker_cb_id.SIZE = 4
union_rgp_sqtt_marker_cb_id._fields_ = ['per_frame_cb_id', 'global_cb_id', 'all']
setattr(union_rgp_sqtt_marker_cb_id, 'per_frame_cb_id', field(0, _anonstruct7))
setattr(union_rgp_sqtt_marker_cb_id, 'global_cb_id', field(0, _anonstruct8))
setattr(union_rgp_sqtt_marker_cb_id, 'all', field(0, uint32_t))
class struct_rgp_sqtt_marker_cb_start(Struct): pass
struct_rgp_sqtt_marker_cb_start.SIZE = 16
struct_rgp_sqtt_marker_cb_start._fields_ = ['identifier', 'ext_dwords', 'cb_id', 'queue', 'dword01', 'device_id_low', 'dword02', 'device_id_high', 'dword03', 'queue_flags', 'dword04']
setattr(struct_rgp_sqtt_marker_cb_start, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_cb_start, 'ext_dwords', field(0, uint32_t, 3, 4))
setattr(struct_rgp_sqtt_marker_cb_start, 'cb_id', field(0, uint32_t, 20, 7))
setattr(struct_rgp_sqtt_marker_cb_start, 'queue', field(3, uint32_t, 5, 3))
setattr(struct_rgp_sqtt_marker_cb_start, 'dword01', field(0, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_start, 'device_id_low', field(4, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_start, 'dword02', field(4, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_start, 'device_id_high', field(8, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_start, 'dword03', field(8, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_start, 'queue_flags', field(12, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_start, 'dword04', field(12, uint32_t))
class struct_rgp_sqtt_marker_cb_end(Struct): pass
struct_rgp_sqtt_marker_cb_end.SIZE = 12
struct_rgp_sqtt_marker_cb_end._fields_ = ['identifier', 'ext_dwords', 'cb_id', 'reserved', 'dword01', 'device_id_low', 'dword02', 'device_id_high', 'dword03']
setattr(struct_rgp_sqtt_marker_cb_end, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_cb_end, 'ext_dwords', field(0, uint32_t, 3, 4))
setattr(struct_rgp_sqtt_marker_cb_end, 'cb_id', field(0, uint32_t, 20, 7))
setattr(struct_rgp_sqtt_marker_cb_end, 'reserved', field(3, uint32_t, 5, 3))
setattr(struct_rgp_sqtt_marker_cb_end, 'dword01', field(0, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_end, 'device_id_low', field(4, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_end, 'dword02', field(4, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_end, 'device_id_high', field(8, uint32_t))
setattr(struct_rgp_sqtt_marker_cb_end, 'dword03', field(8, uint32_t))
enum_rgp_sqtt_marker_general_api_type = CEnum(ctypes.c_uint32)
ApiCmdBindPipeline = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBindPipeline', 0)
ApiCmdBindDescriptorSets = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBindDescriptorSets', 1)
ApiCmdBindIndexBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBindIndexBuffer', 2)
ApiCmdBindVertexBuffers = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBindVertexBuffers', 3)
ApiCmdDraw = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDraw', 4)
ApiCmdDrawIndexed = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndexed', 5)
ApiCmdDrawIndirect = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndirect', 6)
ApiCmdDrawIndexedIndirect = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndexedIndirect', 7)
ApiCmdDrawIndirectCountAMD = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndirectCountAMD', 8)
ApiCmdDrawIndexedIndirectCountAMD = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndexedIndirectCountAMD', 9)
ApiCmdDispatch = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDispatch', 10)
ApiCmdDispatchIndirect = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDispatchIndirect', 11)
ApiCmdCopyBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyBuffer', 12)
ApiCmdCopyImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyImage', 13)
ApiCmdBlitImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBlitImage', 14)
ApiCmdCopyBufferToImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyBufferToImage', 15)
ApiCmdCopyImageToBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyImageToBuffer', 16)
ApiCmdUpdateBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdUpdateBuffer', 17)
ApiCmdFillBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdFillBuffer', 18)
ApiCmdClearColorImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdClearColorImage', 19)
ApiCmdClearDepthStencilImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdClearDepthStencilImage', 20)
ApiCmdClearAttachments = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdClearAttachments', 21)
ApiCmdResolveImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdResolveImage', 22)
ApiCmdWaitEvents = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdWaitEvents', 23)
ApiCmdPipelineBarrier = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdPipelineBarrier', 24)
ApiCmdBeginQuery = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBeginQuery', 25)
ApiCmdEndQuery = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdEndQuery', 26)
ApiCmdResetQueryPool = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdResetQueryPool', 27)
ApiCmdWriteTimestamp = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdWriteTimestamp', 28)
ApiCmdCopyQueryPoolResults = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyQueryPoolResults', 29)
ApiCmdPushConstants = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdPushConstants', 30)
ApiCmdBeginRenderPass = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBeginRenderPass', 31)
ApiCmdNextSubpass = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdNextSubpass', 32)
ApiCmdEndRenderPass = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdEndRenderPass', 33)
ApiCmdExecuteCommands = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdExecuteCommands', 34)
ApiCmdSetViewport = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetViewport', 35)
ApiCmdSetScissor = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetScissor', 36)
ApiCmdSetLineWidth = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetLineWidth', 37)
ApiCmdSetDepthBias = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetDepthBias', 38)
ApiCmdSetBlendConstants = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetBlendConstants', 39)
ApiCmdSetDepthBounds = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetDepthBounds', 40)
ApiCmdSetStencilCompareMask = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetStencilCompareMask', 41)
ApiCmdSetStencilWriteMask = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetStencilWriteMask', 42)
ApiCmdSetStencilReference = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetStencilReference', 43)
ApiCmdDrawIndirectCount = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndirectCount', 44)
ApiCmdDrawIndexedIndirectCount = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndexedIndirectCount', 45)
ApiCmdDrawMeshTasksEXT = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawMeshTasksEXT', 47)
ApiCmdDrawMeshTasksIndirectCountEXT = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawMeshTasksIndirectCountEXT', 48)
ApiCmdDrawMeshTasksIndirectEXT = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawMeshTasksIndirectEXT', 49)
ApiRayTracingSeparateCompiled = enum_rgp_sqtt_marker_general_api_type.define('ApiRayTracingSeparateCompiled', 8388608)
ApiInvalid = enum_rgp_sqtt_marker_general_api_type.define('ApiInvalid', 4294967295)

class struct_rgp_sqtt_marker_general_api(Struct): pass
struct_rgp_sqtt_marker_general_api.SIZE = 4
struct_rgp_sqtt_marker_general_api._fields_ = ['identifier', 'ext_dwords', 'api_type', 'is_end', 'reserved', 'dword01']
setattr(struct_rgp_sqtt_marker_general_api, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_general_api, 'ext_dwords', field(0, uint32_t, 3, 4))
setattr(struct_rgp_sqtt_marker_general_api, 'api_type', field(0, uint32_t, 20, 7))
setattr(struct_rgp_sqtt_marker_general_api, 'is_end', field(3, uint32_t, 1, 3))
setattr(struct_rgp_sqtt_marker_general_api, 'reserved', field(3, uint32_t, 4, 4))
setattr(struct_rgp_sqtt_marker_general_api, 'dword01', field(0, uint32_t))
enum_rgp_sqtt_marker_event_type = CEnum(ctypes.c_uint32)
EventCmdDraw = enum_rgp_sqtt_marker_event_type.define('EventCmdDraw', 0)
EventCmdDrawIndexed = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndexed', 1)
EventCmdDrawIndirect = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndirect', 2)
EventCmdDrawIndexedIndirect = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndexedIndirect', 3)
EventCmdDrawIndirectCountAMD = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndirectCountAMD', 4)
EventCmdDrawIndexedIndirectCountAMD = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndexedIndirectCountAMD', 5)
EventCmdDispatch = enum_rgp_sqtt_marker_event_type.define('EventCmdDispatch', 6)
EventCmdDispatchIndirect = enum_rgp_sqtt_marker_event_type.define('EventCmdDispatchIndirect', 7)
EventCmdCopyBuffer = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyBuffer', 8)
EventCmdCopyImage = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyImage', 9)
EventCmdBlitImage = enum_rgp_sqtt_marker_event_type.define('EventCmdBlitImage', 10)
EventCmdCopyBufferToImage = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyBufferToImage', 11)
EventCmdCopyImageToBuffer = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyImageToBuffer', 12)
EventCmdUpdateBuffer = enum_rgp_sqtt_marker_event_type.define('EventCmdUpdateBuffer', 13)
EventCmdFillBuffer = enum_rgp_sqtt_marker_event_type.define('EventCmdFillBuffer', 14)
EventCmdClearColorImage = enum_rgp_sqtt_marker_event_type.define('EventCmdClearColorImage', 15)
EventCmdClearDepthStencilImage = enum_rgp_sqtt_marker_event_type.define('EventCmdClearDepthStencilImage', 16)
EventCmdClearAttachments = enum_rgp_sqtt_marker_event_type.define('EventCmdClearAttachments', 17)
EventCmdResolveImage = enum_rgp_sqtt_marker_event_type.define('EventCmdResolveImage', 18)
EventCmdWaitEvents = enum_rgp_sqtt_marker_event_type.define('EventCmdWaitEvents', 19)
EventCmdPipelineBarrier = enum_rgp_sqtt_marker_event_type.define('EventCmdPipelineBarrier', 20)
EventCmdResetQueryPool = enum_rgp_sqtt_marker_event_type.define('EventCmdResetQueryPool', 21)
EventCmdCopyQueryPoolResults = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyQueryPoolResults', 22)
EventRenderPassColorClear = enum_rgp_sqtt_marker_event_type.define('EventRenderPassColorClear', 23)
EventRenderPassDepthStencilClear = enum_rgp_sqtt_marker_event_type.define('EventRenderPassDepthStencilClear', 24)
EventRenderPassResolve = enum_rgp_sqtt_marker_event_type.define('EventRenderPassResolve', 25)
EventInternalUnknown = enum_rgp_sqtt_marker_event_type.define('EventInternalUnknown', 26)
EventCmdDrawIndirectCount = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndirectCount', 27)
EventCmdDrawIndexedIndirectCount = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndexedIndirectCount', 28)
EventCmdTraceRaysKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdTraceRaysKHR', 30)
EventCmdTraceRaysIndirectKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdTraceRaysIndirectKHR', 31)
EventCmdBuildAccelerationStructuresKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdBuildAccelerationStructuresKHR', 32)
EventCmdBuildAccelerationStructuresIndirectKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdBuildAccelerationStructuresIndirectKHR', 33)
EventCmdCopyAccelerationStructureKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyAccelerationStructureKHR', 34)
EventCmdCopyAccelerationStructureToMemoryKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyAccelerationStructureToMemoryKHR', 35)
EventCmdCopyMemoryToAccelerationStructureKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyMemoryToAccelerationStructureKHR', 36)
EventCmdDrawMeshTasksEXT = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawMeshTasksEXT', 41)
EventCmdDrawMeshTasksIndirectCountEXT = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawMeshTasksIndirectCountEXT', 42)
EventCmdDrawMeshTasksIndirectEXT = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawMeshTasksIndirectEXT', 43)
EventUnknown = enum_rgp_sqtt_marker_event_type.define('EventUnknown', 32767)
EventInvalid = enum_rgp_sqtt_marker_event_type.define('EventInvalid', 4294967295)

class struct_rgp_sqtt_marker_event(Struct): pass
struct_rgp_sqtt_marker_event.SIZE = 12
struct_rgp_sqtt_marker_event._fields_ = ['identifier', 'ext_dwords', 'api_type', 'has_thread_dims', 'dword01', 'cb_id', 'vertex_offset_reg_idx', 'instance_offset_reg_idx', 'draw_index_reg_idx', 'dword02', 'cmd_id', 'dword03']
setattr(struct_rgp_sqtt_marker_event, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_event, 'ext_dwords', field(0, uint32_t, 3, 4))
setattr(struct_rgp_sqtt_marker_event, 'api_type', field(0, uint32_t, 24, 7))
setattr(struct_rgp_sqtt_marker_event, 'has_thread_dims', field(3, uint32_t, 1, 7))
setattr(struct_rgp_sqtt_marker_event, 'dword01', field(0, uint32_t))
setattr(struct_rgp_sqtt_marker_event, 'cb_id', field(0, uint32_t, 20, 0))
setattr(struct_rgp_sqtt_marker_event, 'vertex_offset_reg_idx', field(2, uint32_t, 4, 4))
setattr(struct_rgp_sqtt_marker_event, 'instance_offset_reg_idx', field(3, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_event, 'draw_index_reg_idx', field(3, uint32_t, 4, 4))
setattr(struct_rgp_sqtt_marker_event, 'dword02', field(4, uint32_t))
setattr(struct_rgp_sqtt_marker_event, 'cmd_id', field(8, uint32_t))
setattr(struct_rgp_sqtt_marker_event, 'dword03', field(8, uint32_t))
class struct_rgp_sqtt_marker_event_with_dims(Struct): pass
struct_rgp_sqtt_marker_event_with_dims.SIZE = 24
struct_rgp_sqtt_marker_event_with_dims._fields_ = ['event', 'thread_x', 'thread_y', 'thread_z']
setattr(struct_rgp_sqtt_marker_event_with_dims, 'event', field(0, struct_rgp_sqtt_marker_event))
setattr(struct_rgp_sqtt_marker_event_with_dims, 'thread_x', field(12, uint32_t))
setattr(struct_rgp_sqtt_marker_event_with_dims, 'thread_y', field(16, uint32_t))
setattr(struct_rgp_sqtt_marker_event_with_dims, 'thread_z', field(20, uint32_t))
class struct_rgp_sqtt_marker_barrier_start(Struct): pass
struct_rgp_sqtt_marker_barrier_start.SIZE = 8
struct_rgp_sqtt_marker_barrier_start._fields_ = ['identifier', 'ext_dwords', 'cb_id', 'reserved', 'dword01', 'driver_reason', 'internal', 'dword02']
setattr(struct_rgp_sqtt_marker_barrier_start, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_barrier_start, 'ext_dwords', field(0, uint32_t, 3, 4))
setattr(struct_rgp_sqtt_marker_barrier_start, 'cb_id', field(0, uint32_t, 20, 7))
setattr(struct_rgp_sqtt_marker_barrier_start, 'reserved', field(3, uint32_t, 5, 3))
setattr(struct_rgp_sqtt_marker_barrier_start, 'dword01', field(0, uint32_t))
setattr(struct_rgp_sqtt_marker_barrier_start, 'driver_reason', field(0, uint32_t, 31, 0))
setattr(struct_rgp_sqtt_marker_barrier_start, 'internal', field(3, uint32_t, 1, 7))
setattr(struct_rgp_sqtt_marker_barrier_start, 'dword02', field(4, uint32_t))
class struct_rgp_sqtt_marker_barrier_end(Struct): pass
struct_rgp_sqtt_marker_barrier_end.SIZE = 8
struct_rgp_sqtt_marker_barrier_end._fields_ = ['identifier', 'ext_dwords', 'cb_id', 'wait_on_eop_ts', 'vs_partial_flush', 'ps_partial_flush', 'cs_partial_flush', 'pfp_sync_me', 'dword01', 'sync_cp_dma', 'inval_tcp', 'inval_sqI', 'inval_sqK', 'flush_tcc', 'inval_tcc', 'flush_cb', 'inval_cb', 'flush_db', 'inval_db', 'num_layout_transitions', 'inval_gl1', 'wait_on_ts', 'eop_ts_bottom_of_pipe', 'eos_ts_ps_done', 'eos_ts_cs_done', 'reserved', 'dword02']
setattr(struct_rgp_sqtt_marker_barrier_end, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_barrier_end, 'ext_dwords', field(0, uint32_t, 3, 4))
setattr(struct_rgp_sqtt_marker_barrier_end, 'cb_id', field(0, uint32_t, 20, 7))
setattr(struct_rgp_sqtt_marker_barrier_end, 'wait_on_eop_ts', field(3, uint32_t, 1, 3))
setattr(struct_rgp_sqtt_marker_barrier_end, 'vs_partial_flush', field(3, uint32_t, 1, 4))
setattr(struct_rgp_sqtt_marker_barrier_end, 'ps_partial_flush', field(3, uint32_t, 1, 5))
setattr(struct_rgp_sqtt_marker_barrier_end, 'cs_partial_flush', field(3, uint32_t, 1, 6))
setattr(struct_rgp_sqtt_marker_barrier_end, 'pfp_sync_me', field(3, uint32_t, 1, 7))
setattr(struct_rgp_sqtt_marker_barrier_end, 'dword01', field(0, uint32_t))
setattr(struct_rgp_sqtt_marker_barrier_end, 'sync_cp_dma', field(0, uint32_t, 1, 0))
setattr(struct_rgp_sqtt_marker_barrier_end, 'inval_tcp', field(0, uint32_t, 1, 1))
setattr(struct_rgp_sqtt_marker_barrier_end, 'inval_sqI', field(0, uint32_t, 1, 2))
setattr(struct_rgp_sqtt_marker_barrier_end, 'inval_sqK', field(0, uint32_t, 1, 3))
setattr(struct_rgp_sqtt_marker_barrier_end, 'flush_tcc', field(0, uint32_t, 1, 4))
setattr(struct_rgp_sqtt_marker_barrier_end, 'inval_tcc', field(0, uint32_t, 1, 5))
setattr(struct_rgp_sqtt_marker_barrier_end, 'flush_cb', field(0, uint32_t, 1, 6))
setattr(struct_rgp_sqtt_marker_barrier_end, 'inval_cb', field(0, uint32_t, 1, 7))
setattr(struct_rgp_sqtt_marker_barrier_end, 'flush_db', field(1, uint32_t, 1, 0))
setattr(struct_rgp_sqtt_marker_barrier_end, 'inval_db', field(1, uint32_t, 1, 1))
setattr(struct_rgp_sqtt_marker_barrier_end, 'num_layout_transitions', field(1, uint32_t, 16, 2))
setattr(struct_rgp_sqtt_marker_barrier_end, 'inval_gl1', field(3, uint32_t, 1, 2))
setattr(struct_rgp_sqtt_marker_barrier_end, 'wait_on_ts', field(3, uint32_t, 1, 3))
setattr(struct_rgp_sqtt_marker_barrier_end, 'eop_ts_bottom_of_pipe', field(3, uint32_t, 1, 4))
setattr(struct_rgp_sqtt_marker_barrier_end, 'eos_ts_ps_done', field(3, uint32_t, 1, 5))
setattr(struct_rgp_sqtt_marker_barrier_end, 'eos_ts_cs_done', field(3, uint32_t, 1, 6))
setattr(struct_rgp_sqtt_marker_barrier_end, 'reserved', field(3, uint32_t, 1, 7))
setattr(struct_rgp_sqtt_marker_barrier_end, 'dword02', field(4, uint32_t))
class struct_rgp_sqtt_marker_layout_transition(Struct): pass
struct_rgp_sqtt_marker_layout_transition.SIZE = 8
struct_rgp_sqtt_marker_layout_transition._fields_ = ['identifier', 'ext_dwords', 'depth_stencil_expand', 'htile_hiz_range_expand', 'depth_stencil_resummarize', 'dcc_decompress', 'fmask_decompress', 'fast_clear_eliminate', 'fmask_color_expand', 'init_mask_ram', 'reserved1', 'dword01', 'reserved2', 'dword02']
setattr(struct_rgp_sqtt_marker_layout_transition, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_layout_transition, 'ext_dwords', field(0, uint32_t, 3, 4))
setattr(struct_rgp_sqtt_marker_layout_transition, 'depth_stencil_expand', field(0, uint32_t, 1, 7))
setattr(struct_rgp_sqtt_marker_layout_transition, 'htile_hiz_range_expand', field(1, uint32_t, 1, 0))
setattr(struct_rgp_sqtt_marker_layout_transition, 'depth_stencil_resummarize', field(1, uint32_t, 1, 1))
setattr(struct_rgp_sqtt_marker_layout_transition, 'dcc_decompress', field(1, uint32_t, 1, 2))
setattr(struct_rgp_sqtt_marker_layout_transition, 'fmask_decompress', field(1, uint32_t, 1, 3))
setattr(struct_rgp_sqtt_marker_layout_transition, 'fast_clear_eliminate', field(1, uint32_t, 1, 4))
setattr(struct_rgp_sqtt_marker_layout_transition, 'fmask_color_expand', field(1, uint32_t, 1, 5))
setattr(struct_rgp_sqtt_marker_layout_transition, 'init_mask_ram', field(1, uint32_t, 1, 6))
setattr(struct_rgp_sqtt_marker_layout_transition, 'reserved1', field(1, uint32_t, 17, 7))
setattr(struct_rgp_sqtt_marker_layout_transition, 'dword01', field(0, uint32_t))
setattr(struct_rgp_sqtt_marker_layout_transition, 'reserved2', field(0, uint32_t, 32, 0))
setattr(struct_rgp_sqtt_marker_layout_transition, 'dword02', field(4, uint32_t))
class struct_rgp_sqtt_marker_user_event(Struct): pass
struct_rgp_sqtt_marker_user_event.SIZE = 4
struct_rgp_sqtt_marker_user_event._fields_ = ['identifier', 'reserved0', 'data_type', 'reserved1', 'dword01']
setattr(struct_rgp_sqtt_marker_user_event, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_user_event, 'reserved0', field(0, uint32_t, 8, 4))
setattr(struct_rgp_sqtt_marker_user_event, 'data_type', field(1, uint32_t, 8, 4))
setattr(struct_rgp_sqtt_marker_user_event, 'reserved1', field(2, uint32_t, 12, 4))
setattr(struct_rgp_sqtt_marker_user_event, 'dword01', field(0, uint32_t))
class struct_rgp_sqtt_marker_user_event_with_length(Struct): pass
struct_rgp_sqtt_marker_user_event_with_length.SIZE = 8
struct_rgp_sqtt_marker_user_event_with_length._fields_ = ['user_event', 'length']
setattr(struct_rgp_sqtt_marker_user_event_with_length, 'user_event', field(0, struct_rgp_sqtt_marker_user_event))
setattr(struct_rgp_sqtt_marker_user_event_with_length, 'length', field(4, uint32_t))
enum_rgp_sqtt_marker_user_event_type = CEnum(ctypes.c_uint32)
UserEventTrigger = enum_rgp_sqtt_marker_user_event_type.define('UserEventTrigger', 0)
UserEventPop = enum_rgp_sqtt_marker_user_event_type.define('UserEventPop', 1)
UserEventPush = enum_rgp_sqtt_marker_user_event_type.define('UserEventPush', 2)
UserEventObjectName = enum_rgp_sqtt_marker_user_event_type.define('UserEventObjectName', 3)

class struct_rgp_sqtt_marker_pipeline_bind(Struct): pass
struct_rgp_sqtt_marker_pipeline_bind.SIZE = 12
struct_rgp_sqtt_marker_pipeline_bind._fields_ = ['identifier', 'ext_dwords', 'bind_point', 'cb_id', 'reserved', 'dword01', 'api_pso_hash', 'dword02', 'dword03']
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'identifier', field(0, uint32_t, 4, 0))
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'ext_dwords', field(0, uint32_t, 3, 4))
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'bind_point', field(0, uint32_t, 1, 7))
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'cb_id', field(1, uint32_t, 20, 0))
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'reserved', field(3, uint32_t, 4, 4))
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'dword01', field(0, uint32_t))
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'api_pso_hash', field(4, Array(uint32_t, 2)))
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'dword02', field(0, uint32_t))
setattr(struct_rgp_sqtt_marker_pipeline_bind, 'dword03', field(4, uint32_t))
SQTT_FILE_MAGIC_NUMBER = 0x50303042
SQTT_FILE_VERSION_MAJOR = 1
SQTT_FILE_VERSION_MINOR = 5
SQTT_GPU_NAME_MAX_SIZE = 256
SQTT_MAX_NUM_SE = 32
SQTT_SA_PER_SE = 2
SQTT_ACTIVE_PIXEL_PACKER_MASK_DWORDS = 4