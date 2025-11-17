# mypy: ignore-errors
import ctypes, os
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
import gzip, base64
from tinygrad.helpers import OSX
from ctypes.util import find_library
def dll():
  try: return ctypes.CDLL(unwrap(find_library('tinymesa_cpu')))
  except: pass
  try: return ctypes.CDLL(unwrap((BASE:=os.getenv('MESA_PATH', f"/usr{'/local/' if OSX else '/'}lib"))+'/libtinymesa_cpu'+(EXT:='.dylib' if OSX else '.so')))
  except: pass
  try: return ctypes.CDLL(unwrap(f'{BASE}/libtinymesa{EXT}'))
  except: pass
  try: return ctypes.CDLL(unwrap('/opt/homebrew/lib/libtinymesa_cpu.dylib'))
  except: pass
  try: return ctypes.CDLL(unwrap('/opt/homebrew/lib/libtinymesa.dylib'))
  except: pass
  return None
dll = dll()

class struct_u_printf_info(Struct): pass
u_printf_info = struct_u_printf_info
uint32_t = ctypes.c_uint32
try: nir_debug = uint32_t.in_dll(dll, 'nir_debug')
except (ValueError,AttributeError): pass
try: nir_debug_print_shader = (ctypes.c_bool * 15).in_dll(dll, 'nir_debug_print_shader')
except (ValueError,AttributeError): pass
nir_component_mask_t = ctypes.c_uint16
try: (nir_process_debug_variable:=dll.nir_process_debug_variable).restype, nir_process_debug_variable.argtypes = None, []
except AttributeError: pass

try: (nir_component_mask_can_reinterpret:=dll.nir_component_mask_can_reinterpret).restype, nir_component_mask_can_reinterpret.argtypes = ctypes.c_bool, [nir_component_mask_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_component_mask_reinterpret:=dll.nir_component_mask_reinterpret).restype, nir_component_mask_reinterpret.argtypes = nir_component_mask_t, [nir_component_mask_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

class struct_nir_state_slot(Struct): pass
gl_state_index16 = ctypes.c_int16
struct_nir_state_slot._fields_ = [
  ('tokens', (gl_state_index16 * 4)),
]
nir_state_slot = struct_nir_state_slot
nir_rounding_mode = CEnum(ctypes.c_uint32)
nir_rounding_mode_undef = nir_rounding_mode.define('nir_rounding_mode_undef', 0)
nir_rounding_mode_rtne = nir_rounding_mode.define('nir_rounding_mode_rtne', 1)
nir_rounding_mode_ru = nir_rounding_mode.define('nir_rounding_mode_ru', 2)
nir_rounding_mode_rd = nir_rounding_mode.define('nir_rounding_mode_rd', 3)
nir_rounding_mode_rtz = nir_rounding_mode.define('nir_rounding_mode_rtz', 4)

nir_ray_query_value = CEnum(ctypes.c_uint32)
nir_ray_query_value_intersection_type = nir_ray_query_value.define('nir_ray_query_value_intersection_type', 0)
nir_ray_query_value_intersection_t = nir_ray_query_value.define('nir_ray_query_value_intersection_t', 1)
nir_ray_query_value_intersection_instance_custom_index = nir_ray_query_value.define('nir_ray_query_value_intersection_instance_custom_index', 2)
nir_ray_query_value_intersection_instance_id = nir_ray_query_value.define('nir_ray_query_value_intersection_instance_id', 3)
nir_ray_query_value_intersection_instance_sbt_index = nir_ray_query_value.define('nir_ray_query_value_intersection_instance_sbt_index', 4)
nir_ray_query_value_intersection_geometry_index = nir_ray_query_value.define('nir_ray_query_value_intersection_geometry_index', 5)
nir_ray_query_value_intersection_primitive_index = nir_ray_query_value.define('nir_ray_query_value_intersection_primitive_index', 6)
nir_ray_query_value_intersection_barycentrics = nir_ray_query_value.define('nir_ray_query_value_intersection_barycentrics', 7)
nir_ray_query_value_intersection_front_face = nir_ray_query_value.define('nir_ray_query_value_intersection_front_face', 8)
nir_ray_query_value_intersection_object_ray_direction = nir_ray_query_value.define('nir_ray_query_value_intersection_object_ray_direction', 9)
nir_ray_query_value_intersection_object_ray_origin = nir_ray_query_value.define('nir_ray_query_value_intersection_object_ray_origin', 10)
nir_ray_query_value_intersection_object_to_world = nir_ray_query_value.define('nir_ray_query_value_intersection_object_to_world', 11)
nir_ray_query_value_intersection_world_to_object = nir_ray_query_value.define('nir_ray_query_value_intersection_world_to_object', 12)
nir_ray_query_value_intersection_candidate_aabb_opaque = nir_ray_query_value.define('nir_ray_query_value_intersection_candidate_aabb_opaque', 13)
nir_ray_query_value_tmin = nir_ray_query_value.define('nir_ray_query_value_tmin', 14)
nir_ray_query_value_flags = nir_ray_query_value.define('nir_ray_query_value_flags', 15)
nir_ray_query_value_world_ray_direction = nir_ray_query_value.define('nir_ray_query_value_world_ray_direction', 16)
nir_ray_query_value_world_ray_origin = nir_ray_query_value.define('nir_ray_query_value_world_ray_origin', 17)
nir_ray_query_value_intersection_triangle_vertex_positions = nir_ray_query_value.define('nir_ray_query_value_intersection_triangle_vertex_positions', 18)

nir_resource_data_intel = CEnum(ctypes.c_uint32)
nir_resource_intel_bindless = nir_resource_data_intel.define('nir_resource_intel_bindless', 1)
nir_resource_intel_pushable = nir_resource_data_intel.define('nir_resource_intel_pushable', 2)
nir_resource_intel_sampler = nir_resource_data_intel.define('nir_resource_intel_sampler', 4)
nir_resource_intel_non_uniform = nir_resource_data_intel.define('nir_resource_intel_non_uniform', 8)
nir_resource_intel_sampler_embedded = nir_resource_data_intel.define('nir_resource_intel_sampler_embedded', 16)

nir_preamble_class = CEnum(ctypes.c_uint32)
nir_preamble_class_general = nir_preamble_class.define('nir_preamble_class_general', 0)
nir_preamble_class_image = nir_preamble_class.define('nir_preamble_class_image', 1)
nir_preamble_num_classes = nir_preamble_class.define('nir_preamble_num_classes', 2)

nir_cmat_signed = CEnum(ctypes.c_uint32)
NIR_CMAT_A_SIGNED = nir_cmat_signed.define('NIR_CMAT_A_SIGNED', 1)
NIR_CMAT_B_SIGNED = nir_cmat_signed.define('NIR_CMAT_B_SIGNED', 2)
NIR_CMAT_C_SIGNED = nir_cmat_signed.define('NIR_CMAT_C_SIGNED', 4)
NIR_CMAT_RESULT_SIGNED = nir_cmat_signed.define('NIR_CMAT_RESULT_SIGNED', 8)

class nir_const_value(ctypes.Union): pass
int8_t = ctypes.c_char
uint8_t = ctypes.c_ubyte
int16_t = ctypes.c_int16
uint16_t = ctypes.c_uint16
int32_t = ctypes.c_int32
int64_t = ctypes.c_int64
uint64_t = ctypes.c_uint64
nir_const_value._fields_ = [
  ('b', ctypes.c_bool),
  ('f32', ctypes.c_float),
  ('f64', ctypes.c_double),
  ('i8', int8_t),
  ('u8', uint8_t),
  ('i16', int16_t),
  ('u16', uint16_t),
  ('i32', int32_t),
  ('u32', uint32_t),
  ('i64', int64_t),
  ('u64', uint64_t),
]
try: (nir_const_value_for_float:=dll.nir_const_value_for_float).restype, nir_const_value_for_float.argtypes = nir_const_value, [ctypes.c_double, ctypes.c_uint32]
except AttributeError: pass

try: (nir_const_value_as_float:=dll.nir_const_value_as_float).restype, nir_const_value_as_float.argtypes = ctypes.c_double, [nir_const_value, ctypes.c_uint32]
except AttributeError: pass

class struct_nir_constant(Struct): pass
nir_constant = struct_nir_constant
struct_nir_constant._fields_ = [
  ('values', (nir_const_value * 16)),
  ('is_null_constant', ctypes.c_bool),
  ('num_elements', ctypes.c_uint32),
  ('elements', ctypes.POINTER(ctypes.POINTER(nir_constant))),
]
nir_depth_layout = CEnum(ctypes.c_uint32)
nir_depth_layout_none = nir_depth_layout.define('nir_depth_layout_none', 0)
nir_depth_layout_any = nir_depth_layout.define('nir_depth_layout_any', 1)
nir_depth_layout_greater = nir_depth_layout.define('nir_depth_layout_greater', 2)
nir_depth_layout_less = nir_depth_layout.define('nir_depth_layout_less', 3)
nir_depth_layout_unchanged = nir_depth_layout.define('nir_depth_layout_unchanged', 4)

nir_var_declaration_type = CEnum(ctypes.c_uint32)
nir_var_declared_normally = nir_var_declaration_type.define('nir_var_declared_normally', 0)
nir_var_declared_implicitly = nir_var_declaration_type.define('nir_var_declared_implicitly', 1)
nir_var_hidden = nir_var_declaration_type.define('nir_var_hidden', 2)

class struct_nir_variable_data(Struct): pass
class struct_nir_variable_data_0(ctypes.Union): pass
class struct_nir_variable_data_0_image(Struct): pass
enum_pipe_format = CEnum(ctypes.c_uint32)
PIPE_FORMAT_NONE = enum_pipe_format.define('PIPE_FORMAT_NONE', 0)
PIPE_FORMAT_R64_UINT = enum_pipe_format.define('PIPE_FORMAT_R64_UINT', 1)
PIPE_FORMAT_R64G64_UINT = enum_pipe_format.define('PIPE_FORMAT_R64G64_UINT', 2)
PIPE_FORMAT_R64G64B64_UINT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64_UINT', 3)
PIPE_FORMAT_R64G64B64A64_UINT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64A64_UINT', 4)
PIPE_FORMAT_R64_SINT = enum_pipe_format.define('PIPE_FORMAT_R64_SINT', 5)
PIPE_FORMAT_R64G64_SINT = enum_pipe_format.define('PIPE_FORMAT_R64G64_SINT', 6)
PIPE_FORMAT_R64G64B64_SINT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64_SINT', 7)
PIPE_FORMAT_R64G64B64A64_SINT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64A64_SINT', 8)
PIPE_FORMAT_R64_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R64_FLOAT', 9)
PIPE_FORMAT_R64G64_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R64G64_FLOAT', 10)
PIPE_FORMAT_R64G64B64_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64_FLOAT', 11)
PIPE_FORMAT_R64G64B64A64_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R64G64B64A64_FLOAT', 12)
PIPE_FORMAT_R32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32_FLOAT', 13)
PIPE_FORMAT_R32G32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32G32_FLOAT', 14)
PIPE_FORMAT_R32G32B32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_FLOAT', 15)
PIPE_FORMAT_R32G32B32A32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_FLOAT', 16)
PIPE_FORMAT_R32_UNORM = enum_pipe_format.define('PIPE_FORMAT_R32_UNORM', 17)
PIPE_FORMAT_R32G32_UNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32_UNORM', 18)
PIPE_FORMAT_R32G32B32_UNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_UNORM', 19)
PIPE_FORMAT_R32G32B32A32_UNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_UNORM', 20)
PIPE_FORMAT_R32_USCALED = enum_pipe_format.define('PIPE_FORMAT_R32_USCALED', 21)
PIPE_FORMAT_R32G32_USCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32_USCALED', 22)
PIPE_FORMAT_R32G32B32_USCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_USCALED', 23)
PIPE_FORMAT_R32G32B32A32_USCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_USCALED', 24)
PIPE_FORMAT_R32_SNORM = enum_pipe_format.define('PIPE_FORMAT_R32_SNORM', 25)
PIPE_FORMAT_R32G32_SNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32_SNORM', 26)
PIPE_FORMAT_R32G32B32_SNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_SNORM', 27)
PIPE_FORMAT_R32G32B32A32_SNORM = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_SNORM', 28)
PIPE_FORMAT_R32_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R32_SSCALED', 29)
PIPE_FORMAT_R32G32_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32_SSCALED', 30)
PIPE_FORMAT_R32G32B32_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_SSCALED', 31)
PIPE_FORMAT_R32G32B32A32_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_SSCALED', 32)
PIPE_FORMAT_R16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16_UNORM', 33)
PIPE_FORMAT_R16G16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16_UNORM', 34)
PIPE_FORMAT_R16G16B16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_UNORM', 35)
PIPE_FORMAT_R16G16B16A16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_UNORM', 36)
PIPE_FORMAT_R16_USCALED = enum_pipe_format.define('PIPE_FORMAT_R16_USCALED', 37)
PIPE_FORMAT_R16G16_USCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16_USCALED', 38)
PIPE_FORMAT_R16G16B16_USCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_USCALED', 39)
PIPE_FORMAT_R16G16B16A16_USCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_USCALED', 40)
PIPE_FORMAT_R16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16_SNORM', 41)
PIPE_FORMAT_R16G16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16_SNORM', 42)
PIPE_FORMAT_R16G16B16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_SNORM', 43)
PIPE_FORMAT_R16G16B16A16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_SNORM', 44)
PIPE_FORMAT_R16_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R16_SSCALED', 45)
PIPE_FORMAT_R16G16_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16_SSCALED', 46)
PIPE_FORMAT_R16G16B16_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_SSCALED', 47)
PIPE_FORMAT_R16G16B16A16_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_SSCALED', 48)
PIPE_FORMAT_R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_UNORM', 49)
PIPE_FORMAT_R8G8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8_UNORM', 50)
PIPE_FORMAT_R8G8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_UNORM', 51)
PIPE_FORMAT_B8G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_UNORM', 52)
PIPE_FORMAT_R8G8B8A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_UNORM', 53)
PIPE_FORMAT_B8G8R8A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_UNORM', 54)
PIPE_FORMAT_R8_USCALED = enum_pipe_format.define('PIPE_FORMAT_R8_USCALED', 55)
PIPE_FORMAT_R8G8_USCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8_USCALED', 56)
PIPE_FORMAT_R8G8B8_USCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_USCALED', 57)
PIPE_FORMAT_B8G8R8_USCALED = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_USCALED', 58)
PIPE_FORMAT_R8G8B8A8_USCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_USCALED', 59)
PIPE_FORMAT_B8G8R8A8_USCALED = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_USCALED', 60)
PIPE_FORMAT_A8B8G8R8_USCALED = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_USCALED', 61)
PIPE_FORMAT_R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8_SNORM', 62)
PIPE_FORMAT_R8G8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8_SNORM', 63)
PIPE_FORMAT_R8G8B8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_SNORM', 64)
PIPE_FORMAT_B8G8R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_SNORM', 65)
PIPE_FORMAT_R8G8B8A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_SNORM', 66)
PIPE_FORMAT_B8G8R8A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_SNORM', 67)
PIPE_FORMAT_R8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R8_SSCALED', 68)
PIPE_FORMAT_R8G8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8_SSCALED', 69)
PIPE_FORMAT_R8G8B8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_SSCALED', 70)
PIPE_FORMAT_B8G8R8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_SSCALED', 71)
PIPE_FORMAT_R8G8B8A8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_SSCALED', 72)
PIPE_FORMAT_B8G8R8A8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_SSCALED', 73)
PIPE_FORMAT_A8B8G8R8_SSCALED = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_SSCALED', 74)
PIPE_FORMAT_A8R8G8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_UNORM', 75)
PIPE_FORMAT_R32_FIXED = enum_pipe_format.define('PIPE_FORMAT_R32_FIXED', 76)
PIPE_FORMAT_R32G32_FIXED = enum_pipe_format.define('PIPE_FORMAT_R32G32_FIXED', 77)
PIPE_FORMAT_R32G32B32_FIXED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_FIXED', 78)
PIPE_FORMAT_R32G32B32A32_FIXED = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_FIXED', 79)
PIPE_FORMAT_R16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16_FLOAT', 80)
PIPE_FORMAT_R16G16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16G16_FLOAT', 81)
PIPE_FORMAT_R16G16B16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_FLOAT', 82)
PIPE_FORMAT_R16G16B16A16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_FLOAT', 83)
PIPE_FORMAT_R8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8_UINT', 84)
PIPE_FORMAT_R8G8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8G8_UINT', 85)
PIPE_FORMAT_R8G8B8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_UINT', 86)
PIPE_FORMAT_B8G8R8_UINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_UINT', 87)
PIPE_FORMAT_R8G8B8A8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_UINT', 88)
PIPE_FORMAT_B8G8R8A8_UINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_UINT', 89)
PIPE_FORMAT_R8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8_SINT', 90)
PIPE_FORMAT_R8G8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8G8_SINT', 91)
PIPE_FORMAT_R8G8B8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_SINT', 92)
PIPE_FORMAT_B8G8R8_SINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_SINT', 93)
PIPE_FORMAT_R8G8B8A8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_SINT', 94)
PIPE_FORMAT_B8G8R8A8_SINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_SINT', 95)
PIPE_FORMAT_R16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16_UINT', 96)
PIPE_FORMAT_R16G16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16G16_UINT', 97)
PIPE_FORMAT_R16G16B16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_UINT', 98)
PIPE_FORMAT_R16G16B16A16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_UINT', 99)
PIPE_FORMAT_R16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16_SINT', 100)
PIPE_FORMAT_R16G16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16G16_SINT', 101)
PIPE_FORMAT_R16G16B16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16_SINT', 102)
PIPE_FORMAT_R16G16B16A16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16A16_SINT', 103)
PIPE_FORMAT_R32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32_UINT', 104)
PIPE_FORMAT_R32G32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32G32_UINT', 105)
PIPE_FORMAT_R32G32B32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_UINT', 106)
PIPE_FORMAT_R32G32B32A32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_UINT', 107)
PIPE_FORMAT_R32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32_SINT', 108)
PIPE_FORMAT_R32G32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32G32_SINT', 109)
PIPE_FORMAT_R32G32B32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32_SINT', 110)
PIPE_FORMAT_R32G32B32A32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32A32_SINT', 111)
PIPE_FORMAT_R10G10B10A2_UNORM = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_UNORM', 112)
PIPE_FORMAT_R10G10B10A2_SNORM = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_SNORM', 113)
PIPE_FORMAT_R10G10B10A2_USCALED = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_USCALED', 114)
PIPE_FORMAT_R10G10B10A2_SSCALED = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_SSCALED', 115)
PIPE_FORMAT_B10G10R10A2_UNORM = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_UNORM', 116)
PIPE_FORMAT_B10G10R10A2_SNORM = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_SNORM', 117)
PIPE_FORMAT_B10G10R10A2_USCALED = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_USCALED', 118)
PIPE_FORMAT_B10G10R10A2_SSCALED = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_SSCALED', 119)
PIPE_FORMAT_R11G11B10_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R11G11B10_FLOAT', 120)
PIPE_FORMAT_R10G10B10A2_UINT = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_UINT', 121)
PIPE_FORMAT_R10G10B10A2_SINT = enum_pipe_format.define('PIPE_FORMAT_R10G10B10A2_SINT', 122)
PIPE_FORMAT_B10G10R10A2_UINT = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_UINT', 123)
PIPE_FORMAT_B10G10R10A2_SINT = enum_pipe_format.define('PIPE_FORMAT_B10G10R10A2_SINT', 124)
PIPE_FORMAT_B8G8R8X8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_UNORM', 125)
PIPE_FORMAT_X8B8G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_X8B8G8R8_UNORM', 126)
PIPE_FORMAT_X8R8G8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_X8R8G8B8_UNORM', 127)
PIPE_FORMAT_B5G5R5A1_UNORM = enum_pipe_format.define('PIPE_FORMAT_B5G5R5A1_UNORM', 128)
PIPE_FORMAT_R4G4B4A4_UNORM = enum_pipe_format.define('PIPE_FORMAT_R4G4B4A4_UNORM', 129)
PIPE_FORMAT_B4G4R4A4_UNORM = enum_pipe_format.define('PIPE_FORMAT_B4G4R4A4_UNORM', 130)
PIPE_FORMAT_R5G6B5_UNORM = enum_pipe_format.define('PIPE_FORMAT_R5G6B5_UNORM', 131)
PIPE_FORMAT_B5G6R5_UNORM = enum_pipe_format.define('PIPE_FORMAT_B5G6R5_UNORM', 132)
PIPE_FORMAT_L8_UNORM = enum_pipe_format.define('PIPE_FORMAT_L8_UNORM', 133)
PIPE_FORMAT_A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_A8_UNORM', 134)
PIPE_FORMAT_I8_UNORM = enum_pipe_format.define('PIPE_FORMAT_I8_UNORM', 135)
PIPE_FORMAT_L8A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_L8A8_UNORM', 136)
PIPE_FORMAT_L16_UNORM = enum_pipe_format.define('PIPE_FORMAT_L16_UNORM', 137)
PIPE_FORMAT_UYVY = enum_pipe_format.define('PIPE_FORMAT_UYVY', 138)
PIPE_FORMAT_VYUY = enum_pipe_format.define('PIPE_FORMAT_VYUY', 139)
PIPE_FORMAT_YUYV = enum_pipe_format.define('PIPE_FORMAT_YUYV', 140)
PIPE_FORMAT_YVYU = enum_pipe_format.define('PIPE_FORMAT_YVYU', 141)
PIPE_FORMAT_Z16_UNORM = enum_pipe_format.define('PIPE_FORMAT_Z16_UNORM', 142)
PIPE_FORMAT_Z16_UNORM_S8_UINT = enum_pipe_format.define('PIPE_FORMAT_Z16_UNORM_S8_UINT', 143)
PIPE_FORMAT_Z32_UNORM = enum_pipe_format.define('PIPE_FORMAT_Z32_UNORM', 144)
PIPE_FORMAT_Z32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_Z32_FLOAT', 145)
PIPE_FORMAT_Z24_UNORM_S8_UINT = enum_pipe_format.define('PIPE_FORMAT_Z24_UNORM_S8_UINT', 146)
PIPE_FORMAT_S8_UINT_Z24_UNORM = enum_pipe_format.define('PIPE_FORMAT_S8_UINT_Z24_UNORM', 147)
PIPE_FORMAT_Z24X8_UNORM = enum_pipe_format.define('PIPE_FORMAT_Z24X8_UNORM', 148)
PIPE_FORMAT_X8Z24_UNORM = enum_pipe_format.define('PIPE_FORMAT_X8Z24_UNORM', 149)
PIPE_FORMAT_S8_UINT = enum_pipe_format.define('PIPE_FORMAT_S8_UINT', 150)
PIPE_FORMAT_L8_SRGB = enum_pipe_format.define('PIPE_FORMAT_L8_SRGB', 151)
PIPE_FORMAT_R8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8_SRGB', 152)
PIPE_FORMAT_L8A8_SRGB = enum_pipe_format.define('PIPE_FORMAT_L8A8_SRGB', 153)
PIPE_FORMAT_R8G8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8G8_SRGB', 154)
PIPE_FORMAT_R8G8B8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_SRGB', 155)
PIPE_FORMAT_B8G8R8_SRGB = enum_pipe_format.define('PIPE_FORMAT_B8G8R8_SRGB', 156)
PIPE_FORMAT_A8B8G8R8_SRGB = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_SRGB', 157)
PIPE_FORMAT_X8B8G8R8_SRGB = enum_pipe_format.define('PIPE_FORMAT_X8B8G8R8_SRGB', 158)
PIPE_FORMAT_B8G8R8A8_SRGB = enum_pipe_format.define('PIPE_FORMAT_B8G8R8A8_SRGB', 159)
PIPE_FORMAT_B8G8R8X8_SRGB = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_SRGB', 160)
PIPE_FORMAT_A8R8G8B8_SRGB = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_SRGB', 161)
PIPE_FORMAT_X8R8G8B8_SRGB = enum_pipe_format.define('PIPE_FORMAT_X8R8G8B8_SRGB', 162)
PIPE_FORMAT_R8G8B8A8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8G8B8A8_SRGB', 163)
PIPE_FORMAT_DXT1_RGB = enum_pipe_format.define('PIPE_FORMAT_DXT1_RGB', 164)
PIPE_FORMAT_DXT1_RGBA = enum_pipe_format.define('PIPE_FORMAT_DXT1_RGBA', 165)
PIPE_FORMAT_DXT3_RGBA = enum_pipe_format.define('PIPE_FORMAT_DXT3_RGBA', 166)
PIPE_FORMAT_DXT5_RGBA = enum_pipe_format.define('PIPE_FORMAT_DXT5_RGBA', 167)
PIPE_FORMAT_DXT1_SRGB = enum_pipe_format.define('PIPE_FORMAT_DXT1_SRGB', 168)
PIPE_FORMAT_DXT1_SRGBA = enum_pipe_format.define('PIPE_FORMAT_DXT1_SRGBA', 169)
PIPE_FORMAT_DXT3_SRGBA = enum_pipe_format.define('PIPE_FORMAT_DXT3_SRGBA', 170)
PIPE_FORMAT_DXT5_SRGBA = enum_pipe_format.define('PIPE_FORMAT_DXT5_SRGBA', 171)
PIPE_FORMAT_RGTC1_UNORM = enum_pipe_format.define('PIPE_FORMAT_RGTC1_UNORM', 172)
PIPE_FORMAT_RGTC1_SNORM = enum_pipe_format.define('PIPE_FORMAT_RGTC1_SNORM', 173)
PIPE_FORMAT_RGTC2_UNORM = enum_pipe_format.define('PIPE_FORMAT_RGTC2_UNORM', 174)
PIPE_FORMAT_RGTC2_SNORM = enum_pipe_format.define('PIPE_FORMAT_RGTC2_SNORM', 175)
PIPE_FORMAT_R8G8_B8G8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8_B8G8_UNORM', 176)
PIPE_FORMAT_G8R8_G8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8R8_G8B8_UNORM', 177)
PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM', 178)
PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM', 179)
PIPE_FORMAT_X6R10_UNORM = enum_pipe_format.define('PIPE_FORMAT_X6R10_UNORM', 180)
PIPE_FORMAT_X6R10X6G10_UNORM = enum_pipe_format.define('PIPE_FORMAT_X6R10X6G10_UNORM', 181)
PIPE_FORMAT_X4R12_UNORM = enum_pipe_format.define('PIPE_FORMAT_X4R12_UNORM', 182)
PIPE_FORMAT_X4R12X4G12_UNORM = enum_pipe_format.define('PIPE_FORMAT_X4R12X4G12_UNORM', 183)
PIPE_FORMAT_R8SG8SB8UX8U_NORM = enum_pipe_format.define('PIPE_FORMAT_R8SG8SB8UX8U_NORM', 184)
PIPE_FORMAT_R5SG5SB6U_NORM = enum_pipe_format.define('PIPE_FORMAT_R5SG5SB6U_NORM', 185)
PIPE_FORMAT_A8B8G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_UNORM', 186)
PIPE_FORMAT_B5G5R5X1_UNORM = enum_pipe_format.define('PIPE_FORMAT_B5G5R5X1_UNORM', 187)
PIPE_FORMAT_R9G9B9E5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R9G9B9E5_FLOAT', 188)
PIPE_FORMAT_Z32_FLOAT_S8X24_UINT = enum_pipe_format.define('PIPE_FORMAT_Z32_FLOAT_S8X24_UINT', 189)
PIPE_FORMAT_R1_UNORM = enum_pipe_format.define('PIPE_FORMAT_R1_UNORM', 190)
PIPE_FORMAT_R10G10B10X2_USCALED = enum_pipe_format.define('PIPE_FORMAT_R10G10B10X2_USCALED', 191)
PIPE_FORMAT_R10G10B10X2_SNORM = enum_pipe_format.define('PIPE_FORMAT_R10G10B10X2_SNORM', 192)
PIPE_FORMAT_L4A4_UNORM = enum_pipe_format.define('PIPE_FORMAT_L4A4_UNORM', 193)
PIPE_FORMAT_A2R10G10B10_UNORM = enum_pipe_format.define('PIPE_FORMAT_A2R10G10B10_UNORM', 194)
PIPE_FORMAT_A2B10G10R10_UNORM = enum_pipe_format.define('PIPE_FORMAT_A2B10G10R10_UNORM', 195)
PIPE_FORMAT_R10SG10SB10SA2U_NORM = enum_pipe_format.define('PIPE_FORMAT_R10SG10SB10SA2U_NORM', 196)
PIPE_FORMAT_R8G8Bx_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8Bx_SNORM', 197)
PIPE_FORMAT_R8G8B8X8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_UNORM', 198)
PIPE_FORMAT_B4G4R4X4_UNORM = enum_pipe_format.define('PIPE_FORMAT_B4G4R4X4_UNORM', 199)
PIPE_FORMAT_X24S8_UINT = enum_pipe_format.define('PIPE_FORMAT_X24S8_UINT', 200)
PIPE_FORMAT_S8X24_UINT = enum_pipe_format.define('PIPE_FORMAT_S8X24_UINT', 201)
PIPE_FORMAT_X32_S8X24_UINT = enum_pipe_format.define('PIPE_FORMAT_X32_S8X24_UINT', 202)
PIPE_FORMAT_R3G3B2_UNORM = enum_pipe_format.define('PIPE_FORMAT_R3G3B2_UNORM', 203)
PIPE_FORMAT_B2G3R3_UNORM = enum_pipe_format.define('PIPE_FORMAT_B2G3R3_UNORM', 204)
PIPE_FORMAT_L16A16_UNORM = enum_pipe_format.define('PIPE_FORMAT_L16A16_UNORM', 205)
PIPE_FORMAT_A16_UNORM = enum_pipe_format.define('PIPE_FORMAT_A16_UNORM', 206)
PIPE_FORMAT_I16_UNORM = enum_pipe_format.define('PIPE_FORMAT_I16_UNORM', 207)
PIPE_FORMAT_LATC1_UNORM = enum_pipe_format.define('PIPE_FORMAT_LATC1_UNORM', 208)
PIPE_FORMAT_LATC1_SNORM = enum_pipe_format.define('PIPE_FORMAT_LATC1_SNORM', 209)
PIPE_FORMAT_LATC2_UNORM = enum_pipe_format.define('PIPE_FORMAT_LATC2_UNORM', 210)
PIPE_FORMAT_LATC2_SNORM = enum_pipe_format.define('PIPE_FORMAT_LATC2_SNORM', 211)
PIPE_FORMAT_A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_A8_SNORM', 212)
PIPE_FORMAT_L8_SNORM = enum_pipe_format.define('PIPE_FORMAT_L8_SNORM', 213)
PIPE_FORMAT_L8A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_L8A8_SNORM', 214)
PIPE_FORMAT_I8_SNORM = enum_pipe_format.define('PIPE_FORMAT_I8_SNORM', 215)
PIPE_FORMAT_A16_SNORM = enum_pipe_format.define('PIPE_FORMAT_A16_SNORM', 216)
PIPE_FORMAT_L16_SNORM = enum_pipe_format.define('PIPE_FORMAT_L16_SNORM', 217)
PIPE_FORMAT_L16A16_SNORM = enum_pipe_format.define('PIPE_FORMAT_L16A16_SNORM', 218)
PIPE_FORMAT_I16_SNORM = enum_pipe_format.define('PIPE_FORMAT_I16_SNORM', 219)
PIPE_FORMAT_A16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_A16_FLOAT', 220)
PIPE_FORMAT_L16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_L16_FLOAT', 221)
PIPE_FORMAT_L16A16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_L16A16_FLOAT', 222)
PIPE_FORMAT_I16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_I16_FLOAT', 223)
PIPE_FORMAT_A32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_A32_FLOAT', 224)
PIPE_FORMAT_L32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_L32_FLOAT', 225)
PIPE_FORMAT_L32A32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_L32A32_FLOAT', 226)
PIPE_FORMAT_I32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_I32_FLOAT', 227)
PIPE_FORMAT_YV12 = enum_pipe_format.define('PIPE_FORMAT_YV12', 228)
PIPE_FORMAT_YV16 = enum_pipe_format.define('PIPE_FORMAT_YV16', 229)
PIPE_FORMAT_IYUV = enum_pipe_format.define('PIPE_FORMAT_IYUV', 230)
PIPE_FORMAT_NV12 = enum_pipe_format.define('PIPE_FORMAT_NV12', 231)
PIPE_FORMAT_NV21 = enum_pipe_format.define('PIPE_FORMAT_NV21', 232)
PIPE_FORMAT_NV16 = enum_pipe_format.define('PIPE_FORMAT_NV16', 233)
PIPE_FORMAT_NV15 = enum_pipe_format.define('PIPE_FORMAT_NV15', 234)
PIPE_FORMAT_NV20 = enum_pipe_format.define('PIPE_FORMAT_NV20', 235)
PIPE_FORMAT_Y8_400_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_400_UNORM', 236)
PIPE_FORMAT_Y8_U8_V8_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_U8_V8_422_UNORM', 237)
PIPE_FORMAT_Y8_U8_V8_444_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_U8_V8_444_UNORM', 238)
PIPE_FORMAT_Y8_U8_V8_440_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_U8_V8_440_UNORM', 239)
PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM', 240)
PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM', 241)
PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM', 242)
PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM', 243)
PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM', 244)
PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM', 245)
PIPE_FORMAT_Y16_U16_V16_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y16_U16_V16_420_UNORM', 246)
PIPE_FORMAT_Y16_U16_V16_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y16_U16_V16_422_UNORM', 247)
PIPE_FORMAT_Y16_U16V16_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y16_U16V16_422_UNORM', 248)
PIPE_FORMAT_Y16_U16_V16_444_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y16_U16_V16_444_UNORM', 249)
PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED = enum_pipe_format.define('PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED', 250)
PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED = enum_pipe_format.define('PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED', 251)
PIPE_FORMAT_A4R4_UNORM = enum_pipe_format.define('PIPE_FORMAT_A4R4_UNORM', 252)
PIPE_FORMAT_R4A4_UNORM = enum_pipe_format.define('PIPE_FORMAT_R4A4_UNORM', 253)
PIPE_FORMAT_R8A8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8A8_UNORM', 254)
PIPE_FORMAT_A8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_A8R8_UNORM', 255)
PIPE_FORMAT_A8_UINT = enum_pipe_format.define('PIPE_FORMAT_A8_UINT', 256)
PIPE_FORMAT_I8_UINT = enum_pipe_format.define('PIPE_FORMAT_I8_UINT', 257)
PIPE_FORMAT_L8_UINT = enum_pipe_format.define('PIPE_FORMAT_L8_UINT', 258)
PIPE_FORMAT_L8A8_UINT = enum_pipe_format.define('PIPE_FORMAT_L8A8_UINT', 259)
PIPE_FORMAT_A8_SINT = enum_pipe_format.define('PIPE_FORMAT_A8_SINT', 260)
PIPE_FORMAT_I8_SINT = enum_pipe_format.define('PIPE_FORMAT_I8_SINT', 261)
PIPE_FORMAT_L8_SINT = enum_pipe_format.define('PIPE_FORMAT_L8_SINT', 262)
PIPE_FORMAT_L8A8_SINT = enum_pipe_format.define('PIPE_FORMAT_L8A8_SINT', 263)
PIPE_FORMAT_A16_UINT = enum_pipe_format.define('PIPE_FORMAT_A16_UINT', 264)
PIPE_FORMAT_I16_UINT = enum_pipe_format.define('PIPE_FORMAT_I16_UINT', 265)
PIPE_FORMAT_L16_UINT = enum_pipe_format.define('PIPE_FORMAT_L16_UINT', 266)
PIPE_FORMAT_L16A16_UINT = enum_pipe_format.define('PIPE_FORMAT_L16A16_UINT', 267)
PIPE_FORMAT_A16_SINT = enum_pipe_format.define('PIPE_FORMAT_A16_SINT', 268)
PIPE_FORMAT_I16_SINT = enum_pipe_format.define('PIPE_FORMAT_I16_SINT', 269)
PIPE_FORMAT_L16_SINT = enum_pipe_format.define('PIPE_FORMAT_L16_SINT', 270)
PIPE_FORMAT_L16A16_SINT = enum_pipe_format.define('PIPE_FORMAT_L16A16_SINT', 271)
PIPE_FORMAT_A32_UINT = enum_pipe_format.define('PIPE_FORMAT_A32_UINT', 272)
PIPE_FORMAT_I32_UINT = enum_pipe_format.define('PIPE_FORMAT_I32_UINT', 273)
PIPE_FORMAT_L32_UINT = enum_pipe_format.define('PIPE_FORMAT_L32_UINT', 274)
PIPE_FORMAT_L32A32_UINT = enum_pipe_format.define('PIPE_FORMAT_L32A32_UINT', 275)
PIPE_FORMAT_A32_SINT = enum_pipe_format.define('PIPE_FORMAT_A32_SINT', 276)
PIPE_FORMAT_I32_SINT = enum_pipe_format.define('PIPE_FORMAT_I32_SINT', 277)
PIPE_FORMAT_L32_SINT = enum_pipe_format.define('PIPE_FORMAT_L32_SINT', 278)
PIPE_FORMAT_L32A32_SINT = enum_pipe_format.define('PIPE_FORMAT_L32A32_SINT', 279)
PIPE_FORMAT_A8R8G8B8_UINT = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_UINT', 280)
PIPE_FORMAT_A8B8G8R8_UINT = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_UINT', 281)
PIPE_FORMAT_A2R10G10B10_UINT = enum_pipe_format.define('PIPE_FORMAT_A2R10G10B10_UINT', 282)
PIPE_FORMAT_A2B10G10R10_UINT = enum_pipe_format.define('PIPE_FORMAT_A2B10G10R10_UINT', 283)
PIPE_FORMAT_R5G6B5_UINT = enum_pipe_format.define('PIPE_FORMAT_R5G6B5_UINT', 284)
PIPE_FORMAT_B5G6R5_UINT = enum_pipe_format.define('PIPE_FORMAT_B5G6R5_UINT', 285)
PIPE_FORMAT_R5G5B5A1_UINT = enum_pipe_format.define('PIPE_FORMAT_R5G5B5A1_UINT', 286)
PIPE_FORMAT_B5G5R5A1_UINT = enum_pipe_format.define('PIPE_FORMAT_B5G5R5A1_UINT', 287)
PIPE_FORMAT_A1R5G5B5_UINT = enum_pipe_format.define('PIPE_FORMAT_A1R5G5B5_UINT', 288)
PIPE_FORMAT_A1B5G5R5_UINT = enum_pipe_format.define('PIPE_FORMAT_A1B5G5R5_UINT', 289)
PIPE_FORMAT_R4G4B4A4_UINT = enum_pipe_format.define('PIPE_FORMAT_R4G4B4A4_UINT', 290)
PIPE_FORMAT_B4G4R4A4_UINT = enum_pipe_format.define('PIPE_FORMAT_B4G4R4A4_UINT', 291)
PIPE_FORMAT_A4R4G4B4_UINT = enum_pipe_format.define('PIPE_FORMAT_A4R4G4B4_UINT', 292)
PIPE_FORMAT_A4B4G4R4_UINT = enum_pipe_format.define('PIPE_FORMAT_A4B4G4R4_UINT', 293)
PIPE_FORMAT_R3G3B2_UINT = enum_pipe_format.define('PIPE_FORMAT_R3G3B2_UINT', 294)
PIPE_FORMAT_B2G3R3_UINT = enum_pipe_format.define('PIPE_FORMAT_B2G3R3_UINT', 295)
PIPE_FORMAT_ETC1_RGB8 = enum_pipe_format.define('PIPE_FORMAT_ETC1_RGB8', 296)
PIPE_FORMAT_R8G8_R8B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8_R8B8_UNORM', 297)
PIPE_FORMAT_R8B8_R8G8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8B8_R8G8_UNORM', 298)
PIPE_FORMAT_G8R8_B8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8R8_B8R8_UNORM', 299)
PIPE_FORMAT_B8R8_G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8R8_G8R8_UNORM', 300)
PIPE_FORMAT_G8B8_G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8B8_G8R8_UNORM', 301)
PIPE_FORMAT_B8G8_R8G8_UNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8_R8G8_UNORM', 302)
PIPE_FORMAT_R8G8B8X8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_SNORM', 303)
PIPE_FORMAT_R8G8B8X8_SRGB = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_SRGB', 304)
PIPE_FORMAT_R8G8B8X8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_UINT', 305)
PIPE_FORMAT_R8G8B8X8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8G8B8X8_SINT', 306)
PIPE_FORMAT_B10G10R10X2_UNORM = enum_pipe_format.define('PIPE_FORMAT_B10G10R10X2_UNORM', 307)
PIPE_FORMAT_R16G16B16X16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_UNORM', 308)
PIPE_FORMAT_R16G16B16X16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_SNORM', 309)
PIPE_FORMAT_R16G16B16X16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_FLOAT', 310)
PIPE_FORMAT_R16G16B16X16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_UINT', 311)
PIPE_FORMAT_R16G16B16X16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16G16B16X16_SINT', 312)
PIPE_FORMAT_R32G32B32X32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32X32_FLOAT', 313)
PIPE_FORMAT_R32G32B32X32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32X32_UINT', 314)
PIPE_FORMAT_R32G32B32X32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32G32B32X32_SINT', 315)
PIPE_FORMAT_R8A8_SNORM = enum_pipe_format.define('PIPE_FORMAT_R8A8_SNORM', 316)
PIPE_FORMAT_R16A16_UNORM = enum_pipe_format.define('PIPE_FORMAT_R16A16_UNORM', 317)
PIPE_FORMAT_R16A16_SNORM = enum_pipe_format.define('PIPE_FORMAT_R16A16_SNORM', 318)
PIPE_FORMAT_R16A16_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R16A16_FLOAT', 319)
PIPE_FORMAT_R32A32_FLOAT = enum_pipe_format.define('PIPE_FORMAT_R32A32_FLOAT', 320)
PIPE_FORMAT_R8A8_UINT = enum_pipe_format.define('PIPE_FORMAT_R8A8_UINT', 321)
PIPE_FORMAT_R8A8_SINT = enum_pipe_format.define('PIPE_FORMAT_R8A8_SINT', 322)
PIPE_FORMAT_R16A16_UINT = enum_pipe_format.define('PIPE_FORMAT_R16A16_UINT', 323)
PIPE_FORMAT_R16A16_SINT = enum_pipe_format.define('PIPE_FORMAT_R16A16_SINT', 324)
PIPE_FORMAT_R32A32_UINT = enum_pipe_format.define('PIPE_FORMAT_R32A32_UINT', 325)
PIPE_FORMAT_R32A32_SINT = enum_pipe_format.define('PIPE_FORMAT_R32A32_SINT', 326)
PIPE_FORMAT_B5G6R5_SRGB = enum_pipe_format.define('PIPE_FORMAT_B5G6R5_SRGB', 327)
PIPE_FORMAT_BPTC_RGBA_UNORM = enum_pipe_format.define('PIPE_FORMAT_BPTC_RGBA_UNORM', 328)
PIPE_FORMAT_BPTC_SRGBA = enum_pipe_format.define('PIPE_FORMAT_BPTC_SRGBA', 329)
PIPE_FORMAT_BPTC_RGB_FLOAT = enum_pipe_format.define('PIPE_FORMAT_BPTC_RGB_FLOAT', 330)
PIPE_FORMAT_BPTC_RGB_UFLOAT = enum_pipe_format.define('PIPE_FORMAT_BPTC_RGB_UFLOAT', 331)
PIPE_FORMAT_G8R8_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8R8_UNORM', 332)
PIPE_FORMAT_G8R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_G8R8_SNORM', 333)
PIPE_FORMAT_G16R16_UNORM = enum_pipe_format.define('PIPE_FORMAT_G16R16_UNORM', 334)
PIPE_FORMAT_G16R16_SNORM = enum_pipe_format.define('PIPE_FORMAT_G16R16_SNORM', 335)
PIPE_FORMAT_A8B8G8R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_SNORM', 336)
PIPE_FORMAT_X8B8G8R8_SNORM = enum_pipe_format.define('PIPE_FORMAT_X8B8G8R8_SNORM', 337)
PIPE_FORMAT_ETC2_RGB8 = enum_pipe_format.define('PIPE_FORMAT_ETC2_RGB8', 338)
PIPE_FORMAT_ETC2_SRGB8 = enum_pipe_format.define('PIPE_FORMAT_ETC2_SRGB8', 339)
PIPE_FORMAT_ETC2_RGB8A1 = enum_pipe_format.define('PIPE_FORMAT_ETC2_RGB8A1', 340)
PIPE_FORMAT_ETC2_SRGB8A1 = enum_pipe_format.define('PIPE_FORMAT_ETC2_SRGB8A1', 341)
PIPE_FORMAT_ETC2_RGBA8 = enum_pipe_format.define('PIPE_FORMAT_ETC2_RGBA8', 342)
PIPE_FORMAT_ETC2_SRGBA8 = enum_pipe_format.define('PIPE_FORMAT_ETC2_SRGBA8', 343)
PIPE_FORMAT_ETC2_R11_UNORM = enum_pipe_format.define('PIPE_FORMAT_ETC2_R11_UNORM', 344)
PIPE_FORMAT_ETC2_R11_SNORM = enum_pipe_format.define('PIPE_FORMAT_ETC2_R11_SNORM', 345)
PIPE_FORMAT_ETC2_RG11_UNORM = enum_pipe_format.define('PIPE_FORMAT_ETC2_RG11_UNORM', 346)
PIPE_FORMAT_ETC2_RG11_SNORM = enum_pipe_format.define('PIPE_FORMAT_ETC2_RG11_SNORM', 347)
PIPE_FORMAT_ASTC_4x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4', 348)
PIPE_FORMAT_ASTC_5x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4', 349)
PIPE_FORMAT_ASTC_5x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5', 350)
PIPE_FORMAT_ASTC_6x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5', 351)
PIPE_FORMAT_ASTC_6x6 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6', 352)
PIPE_FORMAT_ASTC_8x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x5', 353)
PIPE_FORMAT_ASTC_8x6 = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x6', 354)
PIPE_FORMAT_ASTC_8x8 = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x8', 355)
PIPE_FORMAT_ASTC_10x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x5', 356)
PIPE_FORMAT_ASTC_10x6 = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x6', 357)
PIPE_FORMAT_ASTC_10x8 = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x8', 358)
PIPE_FORMAT_ASTC_10x10 = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x10', 359)
PIPE_FORMAT_ASTC_12x10 = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x10', 360)
PIPE_FORMAT_ASTC_12x12 = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x12', 361)
PIPE_FORMAT_ASTC_4x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4_SRGB', 362)
PIPE_FORMAT_ASTC_5x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4_SRGB', 363)
PIPE_FORMAT_ASTC_5x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5_SRGB', 364)
PIPE_FORMAT_ASTC_6x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5_SRGB', 365)
PIPE_FORMAT_ASTC_6x6_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6_SRGB', 366)
PIPE_FORMAT_ASTC_8x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x5_SRGB', 367)
PIPE_FORMAT_ASTC_8x6_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x6_SRGB', 368)
PIPE_FORMAT_ASTC_8x8_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x8_SRGB', 369)
PIPE_FORMAT_ASTC_10x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x5_SRGB', 370)
PIPE_FORMAT_ASTC_10x6_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x6_SRGB', 371)
PIPE_FORMAT_ASTC_10x8_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x8_SRGB', 372)
PIPE_FORMAT_ASTC_10x10_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x10_SRGB', 373)
PIPE_FORMAT_ASTC_12x10_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x10_SRGB', 374)
PIPE_FORMAT_ASTC_12x12_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x12_SRGB', 375)
PIPE_FORMAT_ASTC_3x3x3 = enum_pipe_format.define('PIPE_FORMAT_ASTC_3x3x3', 376)
PIPE_FORMAT_ASTC_4x3x3 = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x3x3', 377)
PIPE_FORMAT_ASTC_4x4x3 = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4x3', 378)
PIPE_FORMAT_ASTC_4x4x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4x4', 379)
PIPE_FORMAT_ASTC_5x4x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4x4', 380)
PIPE_FORMAT_ASTC_5x5x4 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5x4', 381)
PIPE_FORMAT_ASTC_5x5x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5x5', 382)
PIPE_FORMAT_ASTC_6x5x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5x5', 383)
PIPE_FORMAT_ASTC_6x6x5 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6x5', 384)
PIPE_FORMAT_ASTC_6x6x6 = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6x6', 385)
PIPE_FORMAT_ASTC_3x3x3_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_3x3x3_SRGB', 386)
PIPE_FORMAT_ASTC_4x3x3_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x3x3_SRGB', 387)
PIPE_FORMAT_ASTC_4x4x3_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4x3_SRGB', 388)
PIPE_FORMAT_ASTC_4x4x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4x4_SRGB', 389)
PIPE_FORMAT_ASTC_5x4x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4x4_SRGB', 390)
PIPE_FORMAT_ASTC_5x5x4_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5x4_SRGB', 391)
PIPE_FORMAT_ASTC_5x5x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5x5_SRGB', 392)
PIPE_FORMAT_ASTC_6x5x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5x5_SRGB', 393)
PIPE_FORMAT_ASTC_6x6x5_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6x5_SRGB', 394)
PIPE_FORMAT_ASTC_6x6x6_SRGB = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6x6_SRGB', 395)
PIPE_FORMAT_ASTC_4x4_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_4x4_FLOAT', 396)
PIPE_FORMAT_ASTC_5x4_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x4_FLOAT', 397)
PIPE_FORMAT_ASTC_5x5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_5x5_FLOAT', 398)
PIPE_FORMAT_ASTC_6x5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x5_FLOAT', 399)
PIPE_FORMAT_ASTC_6x6_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_6x6_FLOAT', 400)
PIPE_FORMAT_ASTC_8x5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x5_FLOAT', 401)
PIPE_FORMAT_ASTC_8x6_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x6_FLOAT', 402)
PIPE_FORMAT_ASTC_8x8_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_8x8_FLOAT', 403)
PIPE_FORMAT_ASTC_10x5_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x5_FLOAT', 404)
PIPE_FORMAT_ASTC_10x6_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x6_FLOAT', 405)
PIPE_FORMAT_ASTC_10x8_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x8_FLOAT', 406)
PIPE_FORMAT_ASTC_10x10_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_10x10_FLOAT', 407)
PIPE_FORMAT_ASTC_12x10_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x10_FLOAT', 408)
PIPE_FORMAT_ASTC_12x12_FLOAT = enum_pipe_format.define('PIPE_FORMAT_ASTC_12x12_FLOAT', 409)
PIPE_FORMAT_FXT1_RGB = enum_pipe_format.define('PIPE_FORMAT_FXT1_RGB', 410)
PIPE_FORMAT_FXT1_RGBA = enum_pipe_format.define('PIPE_FORMAT_FXT1_RGBA', 411)
PIPE_FORMAT_P010 = enum_pipe_format.define('PIPE_FORMAT_P010', 412)
PIPE_FORMAT_P012 = enum_pipe_format.define('PIPE_FORMAT_P012', 413)
PIPE_FORMAT_P016 = enum_pipe_format.define('PIPE_FORMAT_P016', 414)
PIPE_FORMAT_P030 = enum_pipe_format.define('PIPE_FORMAT_P030', 415)
PIPE_FORMAT_Y210 = enum_pipe_format.define('PIPE_FORMAT_Y210', 416)
PIPE_FORMAT_Y212 = enum_pipe_format.define('PIPE_FORMAT_Y212', 417)
PIPE_FORMAT_Y216 = enum_pipe_format.define('PIPE_FORMAT_Y216', 418)
PIPE_FORMAT_Y410 = enum_pipe_format.define('PIPE_FORMAT_Y410', 419)
PIPE_FORMAT_Y412 = enum_pipe_format.define('PIPE_FORMAT_Y412', 420)
PIPE_FORMAT_Y416 = enum_pipe_format.define('PIPE_FORMAT_Y416', 421)
PIPE_FORMAT_R10G10B10X2_UNORM = enum_pipe_format.define('PIPE_FORMAT_R10G10B10X2_UNORM', 422)
PIPE_FORMAT_A1R5G5B5_UNORM = enum_pipe_format.define('PIPE_FORMAT_A1R5G5B5_UNORM', 423)
PIPE_FORMAT_A1B5G5R5_UNORM = enum_pipe_format.define('PIPE_FORMAT_A1B5G5R5_UNORM', 424)
PIPE_FORMAT_X1B5G5R5_UNORM = enum_pipe_format.define('PIPE_FORMAT_X1B5G5R5_UNORM', 425)
PIPE_FORMAT_R5G5B5A1_UNORM = enum_pipe_format.define('PIPE_FORMAT_R5G5B5A1_UNORM', 426)
PIPE_FORMAT_A4R4G4B4_UNORM = enum_pipe_format.define('PIPE_FORMAT_A4R4G4B4_UNORM', 427)
PIPE_FORMAT_A4B4G4R4_UNORM = enum_pipe_format.define('PIPE_FORMAT_A4B4G4R4_UNORM', 428)
PIPE_FORMAT_G8R8_SINT = enum_pipe_format.define('PIPE_FORMAT_G8R8_SINT', 429)
PIPE_FORMAT_A8B8G8R8_SINT = enum_pipe_format.define('PIPE_FORMAT_A8B8G8R8_SINT', 430)
PIPE_FORMAT_X8B8G8R8_SINT = enum_pipe_format.define('PIPE_FORMAT_X8B8G8R8_SINT', 431)
PIPE_FORMAT_ATC_RGB = enum_pipe_format.define('PIPE_FORMAT_ATC_RGB', 432)
PIPE_FORMAT_ATC_RGBA_EXPLICIT = enum_pipe_format.define('PIPE_FORMAT_ATC_RGBA_EXPLICIT', 433)
PIPE_FORMAT_ATC_RGBA_INTERPOLATED = enum_pipe_format.define('PIPE_FORMAT_ATC_RGBA_INTERPOLATED', 434)
PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8 = enum_pipe_format.define('PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8', 435)
PIPE_FORMAT_AYUV = enum_pipe_format.define('PIPE_FORMAT_AYUV', 436)
PIPE_FORMAT_XYUV = enum_pipe_format.define('PIPE_FORMAT_XYUV', 437)
PIPE_FORMAT_R8G8B8_420_UNORM_PACKED = enum_pipe_format.define('PIPE_FORMAT_R8G8B8_420_UNORM_PACKED', 438)
PIPE_FORMAT_R8_G8B8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_G8B8_420_UNORM', 439)
PIPE_FORMAT_R8_B8G8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_B8G8_420_UNORM', 440)
PIPE_FORMAT_G8_B8R8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8_B8R8_420_UNORM', 441)
PIPE_FORMAT_R10G10B10_420_UNORM_PACKED = enum_pipe_format.define('PIPE_FORMAT_R10G10B10_420_UNORM_PACKED', 442)
PIPE_FORMAT_R10_G10B10_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R10_G10B10_420_UNORM', 443)
PIPE_FORMAT_R10_G10B10_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_R10_G10B10_422_UNORM', 444)
PIPE_FORMAT_R8_G8_B8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_G8_B8_420_UNORM', 445)
PIPE_FORMAT_R8_B8_G8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_B8_G8_420_UNORM', 446)
PIPE_FORMAT_G8_B8_R8_420_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8_B8_R8_420_UNORM', 447)
PIPE_FORMAT_R8_G8B8_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_G8B8_422_UNORM', 448)
PIPE_FORMAT_R8_B8G8_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_B8G8_422_UNORM', 449)
PIPE_FORMAT_G8_B8R8_422_UNORM = enum_pipe_format.define('PIPE_FORMAT_G8_B8R8_422_UNORM', 450)
PIPE_FORMAT_R8_G8_B8_UNORM = enum_pipe_format.define('PIPE_FORMAT_R8_G8_B8_UNORM', 451)
PIPE_FORMAT_Y8_UNORM = enum_pipe_format.define('PIPE_FORMAT_Y8_UNORM', 452)
PIPE_FORMAT_B8G8R8X8_SNORM = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_SNORM', 453)
PIPE_FORMAT_B8G8R8X8_UINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_UINT', 454)
PIPE_FORMAT_B8G8R8X8_SINT = enum_pipe_format.define('PIPE_FORMAT_B8G8R8X8_SINT', 455)
PIPE_FORMAT_A8R8G8B8_SNORM = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_SNORM', 456)
PIPE_FORMAT_A8R8G8B8_SINT = enum_pipe_format.define('PIPE_FORMAT_A8R8G8B8_SINT', 457)
PIPE_FORMAT_X8R8G8B8_SNORM = enum_pipe_format.define('PIPE_FORMAT_X8R8G8B8_SNORM', 458)
PIPE_FORMAT_X8R8G8B8_SINT = enum_pipe_format.define('PIPE_FORMAT_X8R8G8B8_SINT', 459)
PIPE_FORMAT_R5G5B5X1_UNORM = enum_pipe_format.define('PIPE_FORMAT_R5G5B5X1_UNORM', 460)
PIPE_FORMAT_X1R5G5B5_UNORM = enum_pipe_format.define('PIPE_FORMAT_X1R5G5B5_UNORM', 461)
PIPE_FORMAT_R4G4B4X4_UNORM = enum_pipe_format.define('PIPE_FORMAT_R4G4B4X4_UNORM', 462)
PIPE_FORMAT_B10G10R10X2_SNORM = enum_pipe_format.define('PIPE_FORMAT_B10G10R10X2_SNORM', 463)
PIPE_FORMAT_R5G6B5_SRGB = enum_pipe_format.define('PIPE_FORMAT_R5G6B5_SRGB', 464)
PIPE_FORMAT_R10G10B10X2_SINT = enum_pipe_format.define('PIPE_FORMAT_R10G10B10X2_SINT', 465)
PIPE_FORMAT_B10G10R10X2_SINT = enum_pipe_format.define('PIPE_FORMAT_B10G10R10X2_SINT', 466)
PIPE_FORMAT_G16R16_SINT = enum_pipe_format.define('PIPE_FORMAT_G16R16_SINT', 467)
PIPE_FORMAT_COUNT = enum_pipe_format.define('PIPE_FORMAT_COUNT', 468)

struct_nir_variable_data_0_image._fields_ = [
  ('format', enum_pipe_format),
]
class struct_nir_variable_data_0_sampler(Struct): pass
struct_nir_variable_data_0_sampler._fields_ = [
  ('is_inline_sampler', ctypes.c_uint32,1),
  ('addressing_mode', ctypes.c_uint32,3),
  ('normalized_coordinates', ctypes.c_uint32,1),
  ('filter_mode', ctypes.c_uint32,1),
]
class struct_nir_variable_data_0_xfb(Struct): pass
struct_nir_variable_data_0_xfb._fields_ = [
  ('buffer', uint16_t,2),
  ('stride', uint16_t),
]
struct_nir_variable_data_0._fields_ = [
  ('image', struct_nir_variable_data_0_image),
  ('sampler', struct_nir_variable_data_0_sampler),
  ('xfb', struct_nir_variable_data_0_xfb),
]
struct_nir_variable_data._anonymous_ = ['_0']
struct_nir_variable_data._fields_ = [
  ('mode', ctypes.c_uint32,21),
  ('read_only', ctypes.c_uint32,1),
  ('centroid', ctypes.c_uint32,1),
  ('sample', ctypes.c_uint32,1),
  ('patch', ctypes.c_uint32,1),
  ('invariant', ctypes.c_uint32,1),
  ('explicit_invariant', ctypes.c_uint32,1),
  ('ray_query', ctypes.c_uint32,1),
  ('precision', ctypes.c_uint32,2),
  ('assigned', ctypes.c_uint32,1),
  ('cannot_coalesce', ctypes.c_uint32,1),
  ('always_active_io', ctypes.c_uint32,1),
  ('interpolation', ctypes.c_uint32,3),
  ('location_frac', ctypes.c_uint32,2),
  ('compact', ctypes.c_uint32,1),
  ('fb_fetch_output', ctypes.c_uint32,1),
  ('bindless', ctypes.c_uint32,1),
  ('explicit_binding', ctypes.c_uint32,1),
  ('explicit_location', ctypes.c_uint32,1),
  ('implicit_sized_array', ctypes.c_uint32,1),
  ('max_array_access', ctypes.c_int32),
  ('has_initializer', ctypes.c_uint32,1),
  ('is_implicit_initializer', ctypes.c_uint32,1),
  ('is_xfb', ctypes.c_uint32,1),
  ('is_xfb_only', ctypes.c_uint32,1),
  ('explicit_xfb_buffer', ctypes.c_uint32,1),
  ('explicit_xfb_stride', ctypes.c_uint32,1),
  ('explicit_offset', ctypes.c_uint32,1),
  ('matrix_layout', ctypes.c_uint32,2),
  ('from_named_ifc_block', ctypes.c_uint32,1),
  ('from_ssbo_unsized_array', ctypes.c_uint32,1),
  ('must_be_shader_input', ctypes.c_uint32,1),
  ('used', ctypes.c_uint32,1),
  ('how_declared', ctypes.c_uint32,2),
  ('per_view', ctypes.c_uint32,1),
  ('per_primitive', ctypes.c_uint32,1),
  ('per_vertex', ctypes.c_uint32,1),
  ('aliased_shared_memory', ctypes.c_uint32,1),
  ('depth_layout', ctypes.c_uint32,3),
  ('stream', ctypes.c_uint32,9),
  ('access', ctypes.c_uint32,9),
  ('descriptor_set', ctypes.c_uint32,5),
  ('index', ctypes.c_uint32),
  ('binding', ctypes.c_uint32),
  ('location', ctypes.c_int32),
  ('alignment', ctypes.c_uint32),
  ('driver_location', ctypes.c_uint32),
  ('offset', ctypes.c_uint32),
  ('_0', struct_nir_variable_data_0),
  ('node_name', ctypes.POINTER(ctypes.c_char)),
]
nir_variable_data = struct_nir_variable_data
class struct_nir_variable(Struct): pass
class struct_exec_node(Struct): pass
struct_exec_node._fields_ = [
  ('next', ctypes.POINTER(struct_exec_node)),
  ('prev', ctypes.POINTER(struct_exec_node)),
]
class struct_glsl_type(Struct): pass
enum_glsl_base_type = CEnum(ctypes.c_uint32)
GLSL_TYPE_UINT = enum_glsl_base_type.define('GLSL_TYPE_UINT', 0)
GLSL_TYPE_INT = enum_glsl_base_type.define('GLSL_TYPE_INT', 1)
GLSL_TYPE_FLOAT = enum_glsl_base_type.define('GLSL_TYPE_FLOAT', 2)
GLSL_TYPE_FLOAT16 = enum_glsl_base_type.define('GLSL_TYPE_FLOAT16', 3)
GLSL_TYPE_BFLOAT16 = enum_glsl_base_type.define('GLSL_TYPE_BFLOAT16', 4)
GLSL_TYPE_FLOAT_E4M3FN = enum_glsl_base_type.define('GLSL_TYPE_FLOAT_E4M3FN', 5)
GLSL_TYPE_FLOAT_E5M2 = enum_glsl_base_type.define('GLSL_TYPE_FLOAT_E5M2', 6)
GLSL_TYPE_DOUBLE = enum_glsl_base_type.define('GLSL_TYPE_DOUBLE', 7)
GLSL_TYPE_UINT8 = enum_glsl_base_type.define('GLSL_TYPE_UINT8', 8)
GLSL_TYPE_INT8 = enum_glsl_base_type.define('GLSL_TYPE_INT8', 9)
GLSL_TYPE_UINT16 = enum_glsl_base_type.define('GLSL_TYPE_UINT16', 10)
GLSL_TYPE_INT16 = enum_glsl_base_type.define('GLSL_TYPE_INT16', 11)
GLSL_TYPE_UINT64 = enum_glsl_base_type.define('GLSL_TYPE_UINT64', 12)
GLSL_TYPE_INT64 = enum_glsl_base_type.define('GLSL_TYPE_INT64', 13)
GLSL_TYPE_BOOL = enum_glsl_base_type.define('GLSL_TYPE_BOOL', 14)
GLSL_TYPE_COOPERATIVE_MATRIX = enum_glsl_base_type.define('GLSL_TYPE_COOPERATIVE_MATRIX', 15)
GLSL_TYPE_SAMPLER = enum_glsl_base_type.define('GLSL_TYPE_SAMPLER', 16)
GLSL_TYPE_TEXTURE = enum_glsl_base_type.define('GLSL_TYPE_TEXTURE', 17)
GLSL_TYPE_IMAGE = enum_glsl_base_type.define('GLSL_TYPE_IMAGE', 18)
GLSL_TYPE_ATOMIC_UINT = enum_glsl_base_type.define('GLSL_TYPE_ATOMIC_UINT', 19)
GLSL_TYPE_STRUCT = enum_glsl_base_type.define('GLSL_TYPE_STRUCT', 20)
GLSL_TYPE_INTERFACE = enum_glsl_base_type.define('GLSL_TYPE_INTERFACE', 21)
GLSL_TYPE_ARRAY = enum_glsl_base_type.define('GLSL_TYPE_ARRAY', 22)
GLSL_TYPE_VOID = enum_glsl_base_type.define('GLSL_TYPE_VOID', 23)
GLSL_TYPE_SUBROUTINE = enum_glsl_base_type.define('GLSL_TYPE_SUBROUTINE', 24)
GLSL_TYPE_ERROR = enum_glsl_base_type.define('GLSL_TYPE_ERROR', 25)

class struct_glsl_cmat_description(Struct): pass
struct_glsl_cmat_description._fields_ = [
  ('element_type', uint8_t,5),
  ('scope', uint8_t,3),
  ('rows', uint8_t),
  ('cols', uint8_t),
  ('use', uint8_t),
]
uintptr_t = ctypes.c_uint64
class struct_glsl_type_fields(ctypes.Union): pass
glsl_type = struct_glsl_type
class struct_glsl_struct_field(Struct): pass
glsl_struct_field = struct_glsl_struct_field
class struct_glsl_struct_field_0(ctypes.Union): pass
class struct_glsl_struct_field_0_0(Struct): pass
struct_glsl_struct_field_0_0._fields_ = [
  ('interpolation', ctypes.c_uint32,3),
  ('centroid', ctypes.c_uint32,1),
  ('sample', ctypes.c_uint32,1),
  ('matrix_layout', ctypes.c_uint32,2),
  ('patch', ctypes.c_uint32,1),
  ('precision', ctypes.c_uint32,2),
  ('memory_read_only', ctypes.c_uint32,1),
  ('memory_write_only', ctypes.c_uint32,1),
  ('memory_coherent', ctypes.c_uint32,1),
  ('memory_volatile', ctypes.c_uint32,1),
  ('memory_restrict', ctypes.c_uint32,1),
  ('explicit_xfb_buffer', ctypes.c_uint32,1),
  ('implicit_sized_array', ctypes.c_uint32,1),
]
struct_glsl_struct_field_0._anonymous_ = ['_0']
struct_glsl_struct_field_0._fields_ = [
  ('_0', struct_glsl_struct_field_0_0),
  ('flags', ctypes.c_uint32),
]
struct_glsl_struct_field._anonymous_ = ['_0']
struct_glsl_struct_field._fields_ = [
  ('type', ctypes.POINTER(glsl_type)),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('location', ctypes.c_int32),
  ('component', ctypes.c_int32),
  ('offset', ctypes.c_int32),
  ('xfb_buffer', ctypes.c_int32),
  ('xfb_stride', ctypes.c_int32),
  ('image_format', enum_pipe_format),
  ('_0', struct_glsl_struct_field_0),
]
struct_glsl_type_fields._fields_ = [
  ('array', ctypes.POINTER(glsl_type)),
  ('structure', ctypes.POINTER(glsl_struct_field)),
]
struct_glsl_type._fields_ = [
  ('gl_type', uint32_t),
  ('base_type', enum_glsl_base_type,8),
  ('sampled_type', enum_glsl_base_type,8),
  ('sampler_dimensionality', ctypes.c_uint32,4),
  ('sampler_shadow', ctypes.c_uint32,1),
  ('sampler_array', ctypes.c_uint32,1),
  ('interface_packing', ctypes.c_uint32,2),
  ('interface_row_major', ctypes.c_uint32,1),
  ('cmat_desc', struct_glsl_cmat_description),
  ('packed', ctypes.c_uint32,1),
  ('has_builtin_name', ctypes.c_uint32,1),
  ('vector_elements', uint8_t),
  ('matrix_columns', uint8_t),
  ('length', ctypes.c_uint32),
  ('name_id', uintptr_t),
  ('explicit_stride', ctypes.c_uint32),
  ('explicit_alignment', ctypes.c_uint32),
  ('fields', struct_glsl_type_fields),
]
nir_variable = struct_nir_variable
struct_nir_variable._fields_ = [
  ('node', struct_exec_node),
  ('type', ctypes.POINTER(struct_glsl_type)),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('data', struct_nir_variable_data),
  ('index', ctypes.c_uint32),
  ('num_members', uint16_t),
  ('max_ifc_array_access', ctypes.POINTER(ctypes.c_int32)),
  ('num_state_slots', uint16_t),
  ('state_slots', ctypes.POINTER(nir_state_slot)),
  ('constant_initializer', ctypes.POINTER(nir_constant)),
  ('pointer_initializer', ctypes.POINTER(nir_variable)),
  ('interface_type', ctypes.POINTER(struct_glsl_type)),
  ('members', ctypes.POINTER(nir_variable_data)),
]
nir_instr_type = CEnum(ctypes.c_ubyte)
nir_instr_type_alu = nir_instr_type.define('nir_instr_type_alu', 0)
nir_instr_type_deref = nir_instr_type.define('nir_instr_type_deref', 1)
nir_instr_type_call = nir_instr_type.define('nir_instr_type_call', 2)
nir_instr_type_tex = nir_instr_type.define('nir_instr_type_tex', 3)
nir_instr_type_intrinsic = nir_instr_type.define('nir_instr_type_intrinsic', 4)
nir_instr_type_load_const = nir_instr_type.define('nir_instr_type_load_const', 5)
nir_instr_type_jump = nir_instr_type.define('nir_instr_type_jump', 6)
nir_instr_type_undef = nir_instr_type.define('nir_instr_type_undef', 7)
nir_instr_type_phi = nir_instr_type.define('nir_instr_type_phi', 8)
nir_instr_type_parallel_copy = nir_instr_type.define('nir_instr_type_parallel_copy', 9)

class struct_nir_instr(Struct): pass
class struct_nir_block(Struct): pass
nir_block = struct_nir_block
class struct_nir_cf_node(Struct): pass
nir_cf_node = struct_nir_cf_node
nir_cf_node_type = CEnum(ctypes.c_uint32)
nir_cf_node_block = nir_cf_node_type.define('nir_cf_node_block', 0)
nir_cf_node_if = nir_cf_node_type.define('nir_cf_node_if', 1)
nir_cf_node_loop = nir_cf_node_type.define('nir_cf_node_loop', 2)
nir_cf_node_function = nir_cf_node_type.define('nir_cf_node_function', 3)

nir_cf_node = struct_nir_cf_node
struct_nir_cf_node._fields_ = [
  ('node', struct_exec_node),
  ('type', nir_cf_node_type),
  ('parent', ctypes.POINTER(nir_cf_node)),
]
class struct_exec_list(Struct): pass
struct_exec_list._fields_ = [
  ('head_sentinel', struct_exec_node),
  ('tail_sentinel', struct_exec_node),
]
nir_block = struct_nir_block
class struct_set(Struct): pass
class struct_set_entry(Struct): pass
struct_set_entry._fields_ = [
  ('hash', uint32_t),
  ('key', ctypes.c_void_p),
]
struct_set._fields_ = [
  ('mem_ctx', ctypes.c_void_p),
  ('table', ctypes.POINTER(struct_set_entry)),
  ('key_hash_function', ctypes.CFUNCTYPE(uint32_t, ctypes.c_void_p)),
  ('key_equals_function', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)),
  ('size', uint32_t),
  ('rehash', uint32_t),
  ('size_magic', uint64_t),
  ('rehash_magic', uint64_t),
  ('max_entries', uint32_t),
  ('size_index', uint32_t),
  ('entries', uint32_t),
  ('deleted_entries', uint32_t),
]
struct_nir_block._fields_ = [
  ('cf_node', nir_cf_node),
  ('instr_list', struct_exec_list),
  ('index', ctypes.c_uint32),
  ('divergent', ctypes.c_bool),
  ('successors', (ctypes.POINTER(nir_block) * 2)),
  ('predecessors', ctypes.POINTER(struct_set)),
  ('imm_dom', ctypes.POINTER(nir_block)),
  ('num_dom_children', ctypes.c_uint32),
  ('dom_children', ctypes.POINTER(ctypes.POINTER(nir_block))),
  ('dom_frontier', ctypes.POINTER(struct_set)),
  ('dom_pre_index', uint32_t),
  ('dom_post_index', uint32_t),
  ('start_ip', uint32_t),
  ('end_ip', uint32_t),
  ('live_in', ctypes.POINTER(ctypes.c_uint32)),
  ('live_out', ctypes.POINTER(ctypes.c_uint32)),
]
struct_nir_instr._fields_ = [
  ('node', struct_exec_node),
  ('block', ctypes.POINTER(nir_block)),
  ('type', nir_instr_type),
  ('pass_flags', uint8_t),
  ('has_debug_info', ctypes.c_bool),
  ('index', uint32_t),
]
nir_instr = struct_nir_instr
class struct_nir_def(Struct): pass
class struct_list_head(Struct): pass
struct_list_head._fields_ = [
  ('prev', ctypes.POINTER(struct_list_head)),
  ('next', ctypes.POINTER(struct_list_head)),
]
struct_nir_def._fields_ = [
  ('parent_instr', ctypes.POINTER(nir_instr)),
  ('uses', struct_list_head),
  ('index', ctypes.c_uint32),
  ('num_components', uint8_t),
  ('bit_size', uint8_t),
  ('divergent', ctypes.c_bool),
  ('loop_invariant', ctypes.c_bool),
]
nir_def = struct_nir_def
class struct_nir_src(Struct): pass
struct_nir_src._fields_ = [
  ('_parent', uintptr_t),
  ('use_link', struct_list_head),
  ('ssa', ctypes.POINTER(nir_def)),
]
nir_src = struct_nir_src
try: (nir_src_is_divergent:=dll.nir_src_is_divergent).restype, nir_src_is_divergent.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_src)]
except AttributeError: pass

class struct_nir_alu_src(Struct): pass
struct_nir_alu_src._fields_ = [
  ('src', nir_src),
  ('swizzle', (uint8_t * 16)),
]
nir_alu_src = struct_nir_alu_src
nir_alu_type = CEnum(ctypes.c_ubyte)
nir_type_invalid = nir_alu_type.define('nir_type_invalid', 0)
nir_type_int = nir_alu_type.define('nir_type_int', 2)
nir_type_uint = nir_alu_type.define('nir_type_uint', 4)
nir_type_bool = nir_alu_type.define('nir_type_bool', 6)
nir_type_float = nir_alu_type.define('nir_type_float', 128)
nir_type_bool1 = nir_alu_type.define('nir_type_bool1', 7)
nir_type_bool8 = nir_alu_type.define('nir_type_bool8', 14)
nir_type_bool16 = nir_alu_type.define('nir_type_bool16', 22)
nir_type_bool32 = nir_alu_type.define('nir_type_bool32', 38)
nir_type_int1 = nir_alu_type.define('nir_type_int1', 3)
nir_type_int8 = nir_alu_type.define('nir_type_int8', 10)
nir_type_int16 = nir_alu_type.define('nir_type_int16', 18)
nir_type_int32 = nir_alu_type.define('nir_type_int32', 34)
nir_type_int64 = nir_alu_type.define('nir_type_int64', 66)
nir_type_uint1 = nir_alu_type.define('nir_type_uint1', 5)
nir_type_uint8 = nir_alu_type.define('nir_type_uint8', 12)
nir_type_uint16 = nir_alu_type.define('nir_type_uint16', 20)
nir_type_uint32 = nir_alu_type.define('nir_type_uint32', 36)
nir_type_uint64 = nir_alu_type.define('nir_type_uint64', 68)
nir_type_float16 = nir_alu_type.define('nir_type_float16', 144)
nir_type_float32 = nir_alu_type.define('nir_type_float32', 160)
nir_type_float64 = nir_alu_type.define('nir_type_float64', 192)

try: (nir_get_nir_type_for_glsl_base_type:=dll.nir_get_nir_type_for_glsl_base_type).restype, nir_get_nir_type_for_glsl_base_type.argtypes = nir_alu_type, [enum_glsl_base_type]
except AttributeError: pass

try: (nir_get_glsl_base_type_for_nir_type:=dll.nir_get_glsl_base_type_for_nir_type).restype, nir_get_glsl_base_type_for_nir_type.argtypes = enum_glsl_base_type, [nir_alu_type]
except AttributeError: pass

nir_op = CEnum(ctypes.c_uint32)
nir_op_alignbyte_amd = nir_op.define('nir_op_alignbyte_amd', 0)
nir_op_amul = nir_op.define('nir_op_amul', 1)
nir_op_andg_ir3 = nir_op.define('nir_op_andg_ir3', 2)
nir_op_b16all_fequal16 = nir_op.define('nir_op_b16all_fequal16', 3)
nir_op_b16all_fequal2 = nir_op.define('nir_op_b16all_fequal2', 4)
nir_op_b16all_fequal3 = nir_op.define('nir_op_b16all_fequal3', 5)
nir_op_b16all_fequal4 = nir_op.define('nir_op_b16all_fequal4', 6)
nir_op_b16all_fequal5 = nir_op.define('nir_op_b16all_fequal5', 7)
nir_op_b16all_fequal8 = nir_op.define('nir_op_b16all_fequal8', 8)
nir_op_b16all_iequal16 = nir_op.define('nir_op_b16all_iequal16', 9)
nir_op_b16all_iequal2 = nir_op.define('nir_op_b16all_iequal2', 10)
nir_op_b16all_iequal3 = nir_op.define('nir_op_b16all_iequal3', 11)
nir_op_b16all_iequal4 = nir_op.define('nir_op_b16all_iequal4', 12)
nir_op_b16all_iequal5 = nir_op.define('nir_op_b16all_iequal5', 13)
nir_op_b16all_iequal8 = nir_op.define('nir_op_b16all_iequal8', 14)
nir_op_b16any_fnequal16 = nir_op.define('nir_op_b16any_fnequal16', 15)
nir_op_b16any_fnequal2 = nir_op.define('nir_op_b16any_fnequal2', 16)
nir_op_b16any_fnequal3 = nir_op.define('nir_op_b16any_fnequal3', 17)
nir_op_b16any_fnequal4 = nir_op.define('nir_op_b16any_fnequal4', 18)
nir_op_b16any_fnequal5 = nir_op.define('nir_op_b16any_fnequal5', 19)
nir_op_b16any_fnequal8 = nir_op.define('nir_op_b16any_fnequal8', 20)
nir_op_b16any_inequal16 = nir_op.define('nir_op_b16any_inequal16', 21)
nir_op_b16any_inequal2 = nir_op.define('nir_op_b16any_inequal2', 22)
nir_op_b16any_inequal3 = nir_op.define('nir_op_b16any_inequal3', 23)
nir_op_b16any_inequal4 = nir_op.define('nir_op_b16any_inequal4', 24)
nir_op_b16any_inequal5 = nir_op.define('nir_op_b16any_inequal5', 25)
nir_op_b16any_inequal8 = nir_op.define('nir_op_b16any_inequal8', 26)
nir_op_b16csel = nir_op.define('nir_op_b16csel', 27)
nir_op_b2b1 = nir_op.define('nir_op_b2b1', 28)
nir_op_b2b16 = nir_op.define('nir_op_b2b16', 29)
nir_op_b2b32 = nir_op.define('nir_op_b2b32', 30)
nir_op_b2b8 = nir_op.define('nir_op_b2b8', 31)
nir_op_b2f16 = nir_op.define('nir_op_b2f16', 32)
nir_op_b2f32 = nir_op.define('nir_op_b2f32', 33)
nir_op_b2f64 = nir_op.define('nir_op_b2f64', 34)
nir_op_b2i1 = nir_op.define('nir_op_b2i1', 35)
nir_op_b2i16 = nir_op.define('nir_op_b2i16', 36)
nir_op_b2i32 = nir_op.define('nir_op_b2i32', 37)
nir_op_b2i64 = nir_op.define('nir_op_b2i64', 38)
nir_op_b2i8 = nir_op.define('nir_op_b2i8', 39)
nir_op_b32all_fequal16 = nir_op.define('nir_op_b32all_fequal16', 40)
nir_op_b32all_fequal2 = nir_op.define('nir_op_b32all_fequal2', 41)
nir_op_b32all_fequal3 = nir_op.define('nir_op_b32all_fequal3', 42)
nir_op_b32all_fequal4 = nir_op.define('nir_op_b32all_fequal4', 43)
nir_op_b32all_fequal5 = nir_op.define('nir_op_b32all_fequal5', 44)
nir_op_b32all_fequal8 = nir_op.define('nir_op_b32all_fequal8', 45)
nir_op_b32all_iequal16 = nir_op.define('nir_op_b32all_iequal16', 46)
nir_op_b32all_iequal2 = nir_op.define('nir_op_b32all_iequal2', 47)
nir_op_b32all_iequal3 = nir_op.define('nir_op_b32all_iequal3', 48)
nir_op_b32all_iequal4 = nir_op.define('nir_op_b32all_iequal4', 49)
nir_op_b32all_iequal5 = nir_op.define('nir_op_b32all_iequal5', 50)
nir_op_b32all_iequal8 = nir_op.define('nir_op_b32all_iequal8', 51)
nir_op_b32any_fnequal16 = nir_op.define('nir_op_b32any_fnequal16', 52)
nir_op_b32any_fnequal2 = nir_op.define('nir_op_b32any_fnequal2', 53)
nir_op_b32any_fnequal3 = nir_op.define('nir_op_b32any_fnequal3', 54)
nir_op_b32any_fnequal4 = nir_op.define('nir_op_b32any_fnequal4', 55)
nir_op_b32any_fnequal5 = nir_op.define('nir_op_b32any_fnequal5', 56)
nir_op_b32any_fnequal8 = nir_op.define('nir_op_b32any_fnequal8', 57)
nir_op_b32any_inequal16 = nir_op.define('nir_op_b32any_inequal16', 58)
nir_op_b32any_inequal2 = nir_op.define('nir_op_b32any_inequal2', 59)
nir_op_b32any_inequal3 = nir_op.define('nir_op_b32any_inequal3', 60)
nir_op_b32any_inequal4 = nir_op.define('nir_op_b32any_inequal4', 61)
nir_op_b32any_inequal5 = nir_op.define('nir_op_b32any_inequal5', 62)
nir_op_b32any_inequal8 = nir_op.define('nir_op_b32any_inequal8', 63)
nir_op_b32csel = nir_op.define('nir_op_b32csel', 64)
nir_op_b32fcsel_mdg = nir_op.define('nir_op_b32fcsel_mdg', 65)
nir_op_b8all_fequal16 = nir_op.define('nir_op_b8all_fequal16', 66)
nir_op_b8all_fequal2 = nir_op.define('nir_op_b8all_fequal2', 67)
nir_op_b8all_fequal3 = nir_op.define('nir_op_b8all_fequal3', 68)
nir_op_b8all_fequal4 = nir_op.define('nir_op_b8all_fequal4', 69)
nir_op_b8all_fequal5 = nir_op.define('nir_op_b8all_fequal5', 70)
nir_op_b8all_fequal8 = nir_op.define('nir_op_b8all_fequal8', 71)
nir_op_b8all_iequal16 = nir_op.define('nir_op_b8all_iequal16', 72)
nir_op_b8all_iequal2 = nir_op.define('nir_op_b8all_iequal2', 73)
nir_op_b8all_iequal3 = nir_op.define('nir_op_b8all_iequal3', 74)
nir_op_b8all_iequal4 = nir_op.define('nir_op_b8all_iequal4', 75)
nir_op_b8all_iequal5 = nir_op.define('nir_op_b8all_iequal5', 76)
nir_op_b8all_iequal8 = nir_op.define('nir_op_b8all_iequal8', 77)
nir_op_b8any_fnequal16 = nir_op.define('nir_op_b8any_fnequal16', 78)
nir_op_b8any_fnequal2 = nir_op.define('nir_op_b8any_fnequal2', 79)
nir_op_b8any_fnequal3 = nir_op.define('nir_op_b8any_fnequal3', 80)
nir_op_b8any_fnequal4 = nir_op.define('nir_op_b8any_fnequal4', 81)
nir_op_b8any_fnequal5 = nir_op.define('nir_op_b8any_fnequal5', 82)
nir_op_b8any_fnequal8 = nir_op.define('nir_op_b8any_fnequal8', 83)
nir_op_b8any_inequal16 = nir_op.define('nir_op_b8any_inequal16', 84)
nir_op_b8any_inequal2 = nir_op.define('nir_op_b8any_inequal2', 85)
nir_op_b8any_inequal3 = nir_op.define('nir_op_b8any_inequal3', 86)
nir_op_b8any_inequal4 = nir_op.define('nir_op_b8any_inequal4', 87)
nir_op_b8any_inequal5 = nir_op.define('nir_op_b8any_inequal5', 88)
nir_op_b8any_inequal8 = nir_op.define('nir_op_b8any_inequal8', 89)
nir_op_b8csel = nir_op.define('nir_op_b8csel', 90)
nir_op_ball_fequal16 = nir_op.define('nir_op_ball_fequal16', 91)
nir_op_ball_fequal2 = nir_op.define('nir_op_ball_fequal2', 92)
nir_op_ball_fequal3 = nir_op.define('nir_op_ball_fequal3', 93)
nir_op_ball_fequal4 = nir_op.define('nir_op_ball_fequal4', 94)
nir_op_ball_fequal5 = nir_op.define('nir_op_ball_fequal5', 95)
nir_op_ball_fequal8 = nir_op.define('nir_op_ball_fequal8', 96)
nir_op_ball_iequal16 = nir_op.define('nir_op_ball_iequal16', 97)
nir_op_ball_iequal2 = nir_op.define('nir_op_ball_iequal2', 98)
nir_op_ball_iequal3 = nir_op.define('nir_op_ball_iequal3', 99)
nir_op_ball_iequal4 = nir_op.define('nir_op_ball_iequal4', 100)
nir_op_ball_iequal5 = nir_op.define('nir_op_ball_iequal5', 101)
nir_op_ball_iequal8 = nir_op.define('nir_op_ball_iequal8', 102)
nir_op_bany_fnequal16 = nir_op.define('nir_op_bany_fnequal16', 103)
nir_op_bany_fnequal2 = nir_op.define('nir_op_bany_fnequal2', 104)
nir_op_bany_fnequal3 = nir_op.define('nir_op_bany_fnequal3', 105)
nir_op_bany_fnequal4 = nir_op.define('nir_op_bany_fnequal4', 106)
nir_op_bany_fnequal5 = nir_op.define('nir_op_bany_fnequal5', 107)
nir_op_bany_fnequal8 = nir_op.define('nir_op_bany_fnequal8', 108)
nir_op_bany_inequal16 = nir_op.define('nir_op_bany_inequal16', 109)
nir_op_bany_inequal2 = nir_op.define('nir_op_bany_inequal2', 110)
nir_op_bany_inequal3 = nir_op.define('nir_op_bany_inequal3', 111)
nir_op_bany_inequal4 = nir_op.define('nir_op_bany_inequal4', 112)
nir_op_bany_inequal5 = nir_op.define('nir_op_bany_inequal5', 113)
nir_op_bany_inequal8 = nir_op.define('nir_op_bany_inequal8', 114)
nir_op_bcsel = nir_op.define('nir_op_bcsel', 115)
nir_op_bf2f = nir_op.define('nir_op_bf2f', 116)
nir_op_bfdot16 = nir_op.define('nir_op_bfdot16', 117)
nir_op_bfdot2 = nir_op.define('nir_op_bfdot2', 118)
nir_op_bfdot2_bfadd = nir_op.define('nir_op_bfdot2_bfadd', 119)
nir_op_bfdot3 = nir_op.define('nir_op_bfdot3', 120)
nir_op_bfdot4 = nir_op.define('nir_op_bfdot4', 121)
nir_op_bfdot5 = nir_op.define('nir_op_bfdot5', 122)
nir_op_bfdot8 = nir_op.define('nir_op_bfdot8', 123)
nir_op_bffma = nir_op.define('nir_op_bffma', 124)
nir_op_bfi = nir_op.define('nir_op_bfi', 125)
nir_op_bfm = nir_op.define('nir_op_bfm', 126)
nir_op_bfmul = nir_op.define('nir_op_bfmul', 127)
nir_op_bit_count = nir_op.define('nir_op_bit_count', 128)
nir_op_bitfield_insert = nir_op.define('nir_op_bitfield_insert', 129)
nir_op_bitfield_reverse = nir_op.define('nir_op_bitfield_reverse', 130)
nir_op_bitfield_select = nir_op.define('nir_op_bitfield_select', 131)
nir_op_bitnz = nir_op.define('nir_op_bitnz', 132)
nir_op_bitnz16 = nir_op.define('nir_op_bitnz16', 133)
nir_op_bitnz32 = nir_op.define('nir_op_bitnz32', 134)
nir_op_bitnz8 = nir_op.define('nir_op_bitnz8', 135)
nir_op_bitz = nir_op.define('nir_op_bitz', 136)
nir_op_bitz16 = nir_op.define('nir_op_bitz16', 137)
nir_op_bitz32 = nir_op.define('nir_op_bitz32', 138)
nir_op_bitz8 = nir_op.define('nir_op_bitz8', 139)
nir_op_bounds_agx = nir_op.define('nir_op_bounds_agx', 140)
nir_op_byte_perm_amd = nir_op.define('nir_op_byte_perm_amd', 141)
nir_op_cube_amd = nir_op.define('nir_op_cube_amd', 142)
nir_op_e4m3fn2f = nir_op.define('nir_op_e4m3fn2f', 143)
nir_op_e5m22f = nir_op.define('nir_op_e5m22f', 144)
nir_op_extr_agx = nir_op.define('nir_op_extr_agx', 145)
nir_op_extract_i16 = nir_op.define('nir_op_extract_i16', 146)
nir_op_extract_i8 = nir_op.define('nir_op_extract_i8', 147)
nir_op_extract_u16 = nir_op.define('nir_op_extract_u16', 148)
nir_op_extract_u8 = nir_op.define('nir_op_extract_u8', 149)
nir_op_f2bf = nir_op.define('nir_op_f2bf', 150)
nir_op_f2e4m3fn = nir_op.define('nir_op_f2e4m3fn', 151)
nir_op_f2e4m3fn_sat = nir_op.define('nir_op_f2e4m3fn_sat', 152)
nir_op_f2e4m3fn_satfn = nir_op.define('nir_op_f2e4m3fn_satfn', 153)
nir_op_f2e5m2 = nir_op.define('nir_op_f2e5m2', 154)
nir_op_f2e5m2_sat = nir_op.define('nir_op_f2e5m2_sat', 155)
nir_op_f2f16 = nir_op.define('nir_op_f2f16', 156)
nir_op_f2f16_rtne = nir_op.define('nir_op_f2f16_rtne', 157)
nir_op_f2f16_rtz = nir_op.define('nir_op_f2f16_rtz', 158)
nir_op_f2f32 = nir_op.define('nir_op_f2f32', 159)
nir_op_f2f64 = nir_op.define('nir_op_f2f64', 160)
nir_op_f2fmp = nir_op.define('nir_op_f2fmp', 161)
nir_op_f2i1 = nir_op.define('nir_op_f2i1', 162)
nir_op_f2i16 = nir_op.define('nir_op_f2i16', 163)
nir_op_f2i32 = nir_op.define('nir_op_f2i32', 164)
nir_op_f2i64 = nir_op.define('nir_op_f2i64', 165)
nir_op_f2i8 = nir_op.define('nir_op_f2i8', 166)
nir_op_f2imp = nir_op.define('nir_op_f2imp', 167)
nir_op_f2snorm_16_v3d = nir_op.define('nir_op_f2snorm_16_v3d', 168)
nir_op_f2u1 = nir_op.define('nir_op_f2u1', 169)
nir_op_f2u16 = nir_op.define('nir_op_f2u16', 170)
nir_op_f2u32 = nir_op.define('nir_op_f2u32', 171)
nir_op_f2u64 = nir_op.define('nir_op_f2u64', 172)
nir_op_f2u8 = nir_op.define('nir_op_f2u8', 173)
nir_op_f2ump = nir_op.define('nir_op_f2ump', 174)
nir_op_f2unorm_16_v3d = nir_op.define('nir_op_f2unorm_16_v3d', 175)
nir_op_fabs = nir_op.define('nir_op_fabs', 176)
nir_op_fadd = nir_op.define('nir_op_fadd', 177)
nir_op_fall_equal16 = nir_op.define('nir_op_fall_equal16', 178)
nir_op_fall_equal2 = nir_op.define('nir_op_fall_equal2', 179)
nir_op_fall_equal3 = nir_op.define('nir_op_fall_equal3', 180)
nir_op_fall_equal4 = nir_op.define('nir_op_fall_equal4', 181)
nir_op_fall_equal5 = nir_op.define('nir_op_fall_equal5', 182)
nir_op_fall_equal8 = nir_op.define('nir_op_fall_equal8', 183)
nir_op_fany_nequal16 = nir_op.define('nir_op_fany_nequal16', 184)
nir_op_fany_nequal2 = nir_op.define('nir_op_fany_nequal2', 185)
nir_op_fany_nequal3 = nir_op.define('nir_op_fany_nequal3', 186)
nir_op_fany_nequal4 = nir_op.define('nir_op_fany_nequal4', 187)
nir_op_fany_nequal5 = nir_op.define('nir_op_fany_nequal5', 188)
nir_op_fany_nequal8 = nir_op.define('nir_op_fany_nequal8', 189)
nir_op_fceil = nir_op.define('nir_op_fceil', 190)
nir_op_fclamp_pos = nir_op.define('nir_op_fclamp_pos', 191)
nir_op_fcos = nir_op.define('nir_op_fcos', 192)
nir_op_fcos_amd = nir_op.define('nir_op_fcos_amd', 193)
nir_op_fcos_mdg = nir_op.define('nir_op_fcos_mdg', 194)
nir_op_fcsel = nir_op.define('nir_op_fcsel', 195)
nir_op_fcsel_ge = nir_op.define('nir_op_fcsel_ge', 196)
nir_op_fcsel_gt = nir_op.define('nir_op_fcsel_gt', 197)
nir_op_fdiv = nir_op.define('nir_op_fdiv', 198)
nir_op_fdot16 = nir_op.define('nir_op_fdot16', 199)
nir_op_fdot16_replicated = nir_op.define('nir_op_fdot16_replicated', 200)
nir_op_fdot2 = nir_op.define('nir_op_fdot2', 201)
nir_op_fdot2_replicated = nir_op.define('nir_op_fdot2_replicated', 202)
nir_op_fdot3 = nir_op.define('nir_op_fdot3', 203)
nir_op_fdot3_replicated = nir_op.define('nir_op_fdot3_replicated', 204)
nir_op_fdot4 = nir_op.define('nir_op_fdot4', 205)
nir_op_fdot4_replicated = nir_op.define('nir_op_fdot4_replicated', 206)
nir_op_fdot5 = nir_op.define('nir_op_fdot5', 207)
nir_op_fdot5_replicated = nir_op.define('nir_op_fdot5_replicated', 208)
nir_op_fdot8 = nir_op.define('nir_op_fdot8', 209)
nir_op_fdot8_replicated = nir_op.define('nir_op_fdot8_replicated', 210)
nir_op_fdph = nir_op.define('nir_op_fdph', 211)
nir_op_fdph_replicated = nir_op.define('nir_op_fdph_replicated', 212)
nir_op_feq = nir_op.define('nir_op_feq', 213)
nir_op_feq16 = nir_op.define('nir_op_feq16', 214)
nir_op_feq32 = nir_op.define('nir_op_feq32', 215)
nir_op_feq8 = nir_op.define('nir_op_feq8', 216)
nir_op_fequ = nir_op.define('nir_op_fequ', 217)
nir_op_fequ16 = nir_op.define('nir_op_fequ16', 218)
nir_op_fequ32 = nir_op.define('nir_op_fequ32', 219)
nir_op_fequ8 = nir_op.define('nir_op_fequ8', 220)
nir_op_fexp2 = nir_op.define('nir_op_fexp2', 221)
nir_op_ffloor = nir_op.define('nir_op_ffloor', 222)
nir_op_ffma = nir_op.define('nir_op_ffma', 223)
nir_op_ffmaz = nir_op.define('nir_op_ffmaz', 224)
nir_op_ffract = nir_op.define('nir_op_ffract', 225)
nir_op_fge = nir_op.define('nir_op_fge', 226)
nir_op_fge16 = nir_op.define('nir_op_fge16', 227)
nir_op_fge32 = nir_op.define('nir_op_fge32', 228)
nir_op_fge8 = nir_op.define('nir_op_fge8', 229)
nir_op_fgeu = nir_op.define('nir_op_fgeu', 230)
nir_op_fgeu16 = nir_op.define('nir_op_fgeu16', 231)
nir_op_fgeu32 = nir_op.define('nir_op_fgeu32', 232)
nir_op_fgeu8 = nir_op.define('nir_op_fgeu8', 233)
nir_op_find_lsb = nir_op.define('nir_op_find_lsb', 234)
nir_op_fisfinite = nir_op.define('nir_op_fisfinite', 235)
nir_op_fisfinite32 = nir_op.define('nir_op_fisfinite32', 236)
nir_op_fisnormal = nir_op.define('nir_op_fisnormal', 237)
nir_op_flog2 = nir_op.define('nir_op_flog2', 238)
nir_op_flrp = nir_op.define('nir_op_flrp', 239)
nir_op_flt = nir_op.define('nir_op_flt', 240)
nir_op_flt16 = nir_op.define('nir_op_flt16', 241)
nir_op_flt32 = nir_op.define('nir_op_flt32', 242)
nir_op_flt8 = nir_op.define('nir_op_flt8', 243)
nir_op_fltu = nir_op.define('nir_op_fltu', 244)
nir_op_fltu16 = nir_op.define('nir_op_fltu16', 245)
nir_op_fltu32 = nir_op.define('nir_op_fltu32', 246)
nir_op_fltu8 = nir_op.define('nir_op_fltu8', 247)
nir_op_fmax = nir_op.define('nir_op_fmax', 248)
nir_op_fmax_agx = nir_op.define('nir_op_fmax_agx', 249)
nir_op_fmin = nir_op.define('nir_op_fmin', 250)
nir_op_fmin_agx = nir_op.define('nir_op_fmin_agx', 251)
nir_op_fmod = nir_op.define('nir_op_fmod', 252)
nir_op_fmul = nir_op.define('nir_op_fmul', 253)
nir_op_fmulz = nir_op.define('nir_op_fmulz', 254)
nir_op_fneg = nir_op.define('nir_op_fneg', 255)
nir_op_fneo = nir_op.define('nir_op_fneo', 256)
nir_op_fneo16 = nir_op.define('nir_op_fneo16', 257)
nir_op_fneo32 = nir_op.define('nir_op_fneo32', 258)
nir_op_fneo8 = nir_op.define('nir_op_fneo8', 259)
nir_op_fneu = nir_op.define('nir_op_fneu', 260)
nir_op_fneu16 = nir_op.define('nir_op_fneu16', 261)
nir_op_fneu32 = nir_op.define('nir_op_fneu32', 262)
nir_op_fneu8 = nir_op.define('nir_op_fneu8', 263)
nir_op_ford = nir_op.define('nir_op_ford', 264)
nir_op_ford16 = nir_op.define('nir_op_ford16', 265)
nir_op_ford32 = nir_op.define('nir_op_ford32', 266)
nir_op_ford8 = nir_op.define('nir_op_ford8', 267)
nir_op_fpow = nir_op.define('nir_op_fpow', 268)
nir_op_fquantize2f16 = nir_op.define('nir_op_fquantize2f16', 269)
nir_op_frcp = nir_op.define('nir_op_frcp', 270)
nir_op_frem = nir_op.define('nir_op_frem', 271)
nir_op_frexp_exp = nir_op.define('nir_op_frexp_exp', 272)
nir_op_frexp_sig = nir_op.define('nir_op_frexp_sig', 273)
nir_op_fround_even = nir_op.define('nir_op_fround_even', 274)
nir_op_frsq = nir_op.define('nir_op_frsq', 275)
nir_op_fsat = nir_op.define('nir_op_fsat', 276)
nir_op_fsat_signed = nir_op.define('nir_op_fsat_signed', 277)
nir_op_fsign = nir_op.define('nir_op_fsign', 278)
nir_op_fsin = nir_op.define('nir_op_fsin', 279)
nir_op_fsin_agx = nir_op.define('nir_op_fsin_agx', 280)
nir_op_fsin_amd = nir_op.define('nir_op_fsin_amd', 281)
nir_op_fsin_mdg = nir_op.define('nir_op_fsin_mdg', 282)
nir_op_fsqrt = nir_op.define('nir_op_fsqrt', 283)
nir_op_fsub = nir_op.define('nir_op_fsub', 284)
nir_op_fsum2 = nir_op.define('nir_op_fsum2', 285)
nir_op_fsum3 = nir_op.define('nir_op_fsum3', 286)
nir_op_fsum4 = nir_op.define('nir_op_fsum4', 287)
nir_op_ftrunc = nir_op.define('nir_op_ftrunc', 288)
nir_op_funord = nir_op.define('nir_op_funord', 289)
nir_op_funord16 = nir_op.define('nir_op_funord16', 290)
nir_op_funord32 = nir_op.define('nir_op_funord32', 291)
nir_op_funord8 = nir_op.define('nir_op_funord8', 292)
nir_op_i2f16 = nir_op.define('nir_op_i2f16', 293)
nir_op_i2f32 = nir_op.define('nir_op_i2f32', 294)
nir_op_i2f64 = nir_op.define('nir_op_i2f64', 295)
nir_op_i2fmp = nir_op.define('nir_op_i2fmp', 296)
nir_op_i2i1 = nir_op.define('nir_op_i2i1', 297)
nir_op_i2i16 = nir_op.define('nir_op_i2i16', 298)
nir_op_i2i32 = nir_op.define('nir_op_i2i32', 299)
nir_op_i2i64 = nir_op.define('nir_op_i2i64', 300)
nir_op_i2i8 = nir_op.define('nir_op_i2i8', 301)
nir_op_i2imp = nir_op.define('nir_op_i2imp', 302)
nir_op_i32csel_ge = nir_op.define('nir_op_i32csel_ge', 303)
nir_op_i32csel_gt = nir_op.define('nir_op_i32csel_gt', 304)
nir_op_iabs = nir_op.define('nir_op_iabs', 305)
nir_op_iadd = nir_op.define('nir_op_iadd', 306)
nir_op_iadd3 = nir_op.define('nir_op_iadd3', 307)
nir_op_iadd_sat = nir_op.define('nir_op_iadd_sat', 308)
nir_op_iand = nir_op.define('nir_op_iand', 309)
nir_op_ibfe = nir_op.define('nir_op_ibfe', 310)
nir_op_ibitfield_extract = nir_op.define('nir_op_ibitfield_extract', 311)
nir_op_icsel_eqz = nir_op.define('nir_op_icsel_eqz', 312)
nir_op_idiv = nir_op.define('nir_op_idiv', 313)
nir_op_ieq = nir_op.define('nir_op_ieq', 314)
nir_op_ieq16 = nir_op.define('nir_op_ieq16', 315)
nir_op_ieq32 = nir_op.define('nir_op_ieq32', 316)
nir_op_ieq8 = nir_op.define('nir_op_ieq8', 317)
nir_op_ifind_msb = nir_op.define('nir_op_ifind_msb', 318)
nir_op_ifind_msb_rev = nir_op.define('nir_op_ifind_msb_rev', 319)
nir_op_ige = nir_op.define('nir_op_ige', 320)
nir_op_ige16 = nir_op.define('nir_op_ige16', 321)
nir_op_ige32 = nir_op.define('nir_op_ige32', 322)
nir_op_ige8 = nir_op.define('nir_op_ige8', 323)
nir_op_ihadd = nir_op.define('nir_op_ihadd', 324)
nir_op_ilea_agx = nir_op.define('nir_op_ilea_agx', 325)
nir_op_ilt = nir_op.define('nir_op_ilt', 326)
nir_op_ilt16 = nir_op.define('nir_op_ilt16', 327)
nir_op_ilt32 = nir_op.define('nir_op_ilt32', 328)
nir_op_ilt8 = nir_op.define('nir_op_ilt8', 329)
nir_op_imad = nir_op.define('nir_op_imad', 330)
nir_op_imad24_ir3 = nir_op.define('nir_op_imad24_ir3', 331)
nir_op_imadsh_mix16 = nir_op.define('nir_op_imadsh_mix16', 332)
nir_op_imadshl_agx = nir_op.define('nir_op_imadshl_agx', 333)
nir_op_imax = nir_op.define('nir_op_imax', 334)
nir_op_imin = nir_op.define('nir_op_imin', 335)
nir_op_imod = nir_op.define('nir_op_imod', 336)
nir_op_imsubshl_agx = nir_op.define('nir_op_imsubshl_agx', 337)
nir_op_imul = nir_op.define('nir_op_imul', 338)
nir_op_imul24 = nir_op.define('nir_op_imul24', 339)
nir_op_imul24_relaxed = nir_op.define('nir_op_imul24_relaxed', 340)
nir_op_imul_2x32_64 = nir_op.define('nir_op_imul_2x32_64', 341)
nir_op_imul_32x16 = nir_op.define('nir_op_imul_32x16', 342)
nir_op_imul_high = nir_op.define('nir_op_imul_high', 343)
nir_op_ine = nir_op.define('nir_op_ine', 344)
nir_op_ine16 = nir_op.define('nir_op_ine16', 345)
nir_op_ine32 = nir_op.define('nir_op_ine32', 346)
nir_op_ine8 = nir_op.define('nir_op_ine8', 347)
nir_op_ineg = nir_op.define('nir_op_ineg', 348)
nir_op_inot = nir_op.define('nir_op_inot', 349)
nir_op_insert_u16 = nir_op.define('nir_op_insert_u16', 350)
nir_op_insert_u8 = nir_op.define('nir_op_insert_u8', 351)
nir_op_interleave_agx = nir_op.define('nir_op_interleave_agx', 352)
nir_op_ior = nir_op.define('nir_op_ior', 353)
nir_op_irem = nir_op.define('nir_op_irem', 354)
nir_op_irhadd = nir_op.define('nir_op_irhadd', 355)
nir_op_ishl = nir_op.define('nir_op_ishl', 356)
nir_op_ishr = nir_op.define('nir_op_ishr', 357)
nir_op_isign = nir_op.define('nir_op_isign', 358)
nir_op_isub = nir_op.define('nir_op_isub', 359)
nir_op_isub_sat = nir_op.define('nir_op_isub_sat', 360)
nir_op_ixor = nir_op.define('nir_op_ixor', 361)
nir_op_ldexp = nir_op.define('nir_op_ldexp', 362)
nir_op_ldexp16_pan = nir_op.define('nir_op_ldexp16_pan', 363)
nir_op_lea_nv = nir_op.define('nir_op_lea_nv', 364)
nir_op_mov = nir_op.define('nir_op_mov', 365)
nir_op_mqsad_4x8 = nir_op.define('nir_op_mqsad_4x8', 366)
nir_op_msad_4x8 = nir_op.define('nir_op_msad_4x8', 367)
nir_op_pack_2x16_to_snorm_2x8_v3d = nir_op.define('nir_op_pack_2x16_to_snorm_2x8_v3d', 368)
nir_op_pack_2x16_to_unorm_10_2_v3d = nir_op.define('nir_op_pack_2x16_to_unorm_10_2_v3d', 369)
nir_op_pack_2x16_to_unorm_2x10_v3d = nir_op.define('nir_op_pack_2x16_to_unorm_2x10_v3d', 370)
nir_op_pack_2x16_to_unorm_2x8_v3d = nir_op.define('nir_op_pack_2x16_to_unorm_2x8_v3d', 371)
nir_op_pack_2x32_to_2x16_v3d = nir_op.define('nir_op_pack_2x32_to_2x16_v3d', 372)
nir_op_pack_32_2x16 = nir_op.define('nir_op_pack_32_2x16', 373)
nir_op_pack_32_2x16_split = nir_op.define('nir_op_pack_32_2x16_split', 374)
nir_op_pack_32_4x8 = nir_op.define('nir_op_pack_32_4x8', 375)
nir_op_pack_32_4x8_split = nir_op.define('nir_op_pack_32_4x8_split', 376)
nir_op_pack_32_to_r11g11b10_v3d = nir_op.define('nir_op_pack_32_to_r11g11b10_v3d', 377)
nir_op_pack_4x16_to_4x8_v3d = nir_op.define('nir_op_pack_4x16_to_4x8_v3d', 378)
nir_op_pack_64_2x32 = nir_op.define('nir_op_pack_64_2x32', 379)
nir_op_pack_64_2x32_split = nir_op.define('nir_op_pack_64_2x32_split', 380)
nir_op_pack_64_4x16 = nir_op.define('nir_op_pack_64_4x16', 381)
nir_op_pack_double_2x32_dxil = nir_op.define('nir_op_pack_double_2x32_dxil', 382)
nir_op_pack_half_2x16 = nir_op.define('nir_op_pack_half_2x16', 383)
nir_op_pack_half_2x16_rtz_split = nir_op.define('nir_op_pack_half_2x16_rtz_split', 384)
nir_op_pack_half_2x16_split = nir_op.define('nir_op_pack_half_2x16_split', 385)
nir_op_pack_sint_2x16 = nir_op.define('nir_op_pack_sint_2x16', 386)
nir_op_pack_snorm_2x16 = nir_op.define('nir_op_pack_snorm_2x16', 387)
nir_op_pack_snorm_4x8 = nir_op.define('nir_op_pack_snorm_4x8', 388)
nir_op_pack_uint_2x16 = nir_op.define('nir_op_pack_uint_2x16', 389)
nir_op_pack_uint_32_to_r10g10b10a2_v3d = nir_op.define('nir_op_pack_uint_32_to_r10g10b10a2_v3d', 390)
nir_op_pack_unorm_2x16 = nir_op.define('nir_op_pack_unorm_2x16', 391)
nir_op_pack_unorm_4x8 = nir_op.define('nir_op_pack_unorm_4x8', 392)
nir_op_pack_uvec2_to_uint = nir_op.define('nir_op_pack_uvec2_to_uint', 393)
nir_op_pack_uvec4_to_uint = nir_op.define('nir_op_pack_uvec4_to_uint', 394)
nir_op_prmt_nv = nir_op.define('nir_op_prmt_nv', 395)
nir_op_sdot_2x16_iadd = nir_op.define('nir_op_sdot_2x16_iadd', 396)
nir_op_sdot_2x16_iadd_sat = nir_op.define('nir_op_sdot_2x16_iadd_sat', 397)
nir_op_sdot_4x8_iadd = nir_op.define('nir_op_sdot_4x8_iadd', 398)
nir_op_sdot_4x8_iadd_sat = nir_op.define('nir_op_sdot_4x8_iadd_sat', 399)
nir_op_seq = nir_op.define('nir_op_seq', 400)
nir_op_sge = nir_op.define('nir_op_sge', 401)
nir_op_shfr = nir_op.define('nir_op_shfr', 402)
nir_op_shlg_ir3 = nir_op.define('nir_op_shlg_ir3', 403)
nir_op_shlm_ir3 = nir_op.define('nir_op_shlm_ir3', 404)
nir_op_shrg_ir3 = nir_op.define('nir_op_shrg_ir3', 405)
nir_op_shrm_ir3 = nir_op.define('nir_op_shrm_ir3', 406)
nir_op_slt = nir_op.define('nir_op_slt', 407)
nir_op_sne = nir_op.define('nir_op_sne', 408)
nir_op_sudot_4x8_iadd = nir_op.define('nir_op_sudot_4x8_iadd', 409)
nir_op_sudot_4x8_iadd_sat = nir_op.define('nir_op_sudot_4x8_iadd_sat', 410)
nir_op_u2f16 = nir_op.define('nir_op_u2f16', 411)
nir_op_u2f32 = nir_op.define('nir_op_u2f32', 412)
nir_op_u2f64 = nir_op.define('nir_op_u2f64', 413)
nir_op_u2fmp = nir_op.define('nir_op_u2fmp', 414)
nir_op_u2u1 = nir_op.define('nir_op_u2u1', 415)
nir_op_u2u16 = nir_op.define('nir_op_u2u16', 416)
nir_op_u2u32 = nir_op.define('nir_op_u2u32', 417)
nir_op_u2u64 = nir_op.define('nir_op_u2u64', 418)
nir_op_u2u8 = nir_op.define('nir_op_u2u8', 419)
nir_op_uabs_isub = nir_op.define('nir_op_uabs_isub', 420)
nir_op_uabs_usub = nir_op.define('nir_op_uabs_usub', 421)
nir_op_uadd_carry = nir_op.define('nir_op_uadd_carry', 422)
nir_op_uadd_sat = nir_op.define('nir_op_uadd_sat', 423)
nir_op_ubfe = nir_op.define('nir_op_ubfe', 424)
nir_op_ubitfield_extract = nir_op.define('nir_op_ubitfield_extract', 425)
nir_op_uclz = nir_op.define('nir_op_uclz', 426)
nir_op_udiv = nir_op.define('nir_op_udiv', 427)
nir_op_udiv_aligned_4 = nir_op.define('nir_op_udiv_aligned_4', 428)
nir_op_udot_2x16_uadd = nir_op.define('nir_op_udot_2x16_uadd', 429)
nir_op_udot_2x16_uadd_sat = nir_op.define('nir_op_udot_2x16_uadd_sat', 430)
nir_op_udot_4x8_uadd = nir_op.define('nir_op_udot_4x8_uadd', 431)
nir_op_udot_4x8_uadd_sat = nir_op.define('nir_op_udot_4x8_uadd_sat', 432)
nir_op_ufind_msb = nir_op.define('nir_op_ufind_msb', 433)
nir_op_ufind_msb_rev = nir_op.define('nir_op_ufind_msb_rev', 434)
nir_op_uge = nir_op.define('nir_op_uge', 435)
nir_op_uge16 = nir_op.define('nir_op_uge16', 436)
nir_op_uge32 = nir_op.define('nir_op_uge32', 437)
nir_op_uge8 = nir_op.define('nir_op_uge8', 438)
nir_op_uhadd = nir_op.define('nir_op_uhadd', 439)
nir_op_ulea_agx = nir_op.define('nir_op_ulea_agx', 440)
nir_op_ult = nir_op.define('nir_op_ult', 441)
nir_op_ult16 = nir_op.define('nir_op_ult16', 442)
nir_op_ult32 = nir_op.define('nir_op_ult32', 443)
nir_op_ult8 = nir_op.define('nir_op_ult8', 444)
nir_op_umad24 = nir_op.define('nir_op_umad24', 445)
nir_op_umad24_relaxed = nir_op.define('nir_op_umad24_relaxed', 446)
nir_op_umax = nir_op.define('nir_op_umax', 447)
nir_op_umax_4x8_vc4 = nir_op.define('nir_op_umax_4x8_vc4', 448)
nir_op_umin = nir_op.define('nir_op_umin', 449)
nir_op_umin_4x8_vc4 = nir_op.define('nir_op_umin_4x8_vc4', 450)
nir_op_umod = nir_op.define('nir_op_umod', 451)
nir_op_umul24 = nir_op.define('nir_op_umul24', 452)
nir_op_umul24_relaxed = nir_op.define('nir_op_umul24_relaxed', 453)
nir_op_umul_2x32_64 = nir_op.define('nir_op_umul_2x32_64', 454)
nir_op_umul_32x16 = nir_op.define('nir_op_umul_32x16', 455)
nir_op_umul_high = nir_op.define('nir_op_umul_high', 456)
nir_op_umul_low = nir_op.define('nir_op_umul_low', 457)
nir_op_umul_unorm_4x8_vc4 = nir_op.define('nir_op_umul_unorm_4x8_vc4', 458)
nir_op_unpack_32_2x16 = nir_op.define('nir_op_unpack_32_2x16', 459)
nir_op_unpack_32_2x16_split_x = nir_op.define('nir_op_unpack_32_2x16_split_x', 460)
nir_op_unpack_32_2x16_split_y = nir_op.define('nir_op_unpack_32_2x16_split_y', 461)
nir_op_unpack_32_4x8 = nir_op.define('nir_op_unpack_32_4x8', 462)
nir_op_unpack_64_2x32 = nir_op.define('nir_op_unpack_64_2x32', 463)
nir_op_unpack_64_2x32_split_x = nir_op.define('nir_op_unpack_64_2x32_split_x', 464)
nir_op_unpack_64_2x32_split_y = nir_op.define('nir_op_unpack_64_2x32_split_y', 465)
nir_op_unpack_64_4x16 = nir_op.define('nir_op_unpack_64_4x16', 466)
nir_op_unpack_double_2x32_dxil = nir_op.define('nir_op_unpack_double_2x32_dxil', 467)
nir_op_unpack_half_2x16 = nir_op.define('nir_op_unpack_half_2x16', 468)
nir_op_unpack_half_2x16_split_x = nir_op.define('nir_op_unpack_half_2x16_split_x', 469)
nir_op_unpack_half_2x16_split_y = nir_op.define('nir_op_unpack_half_2x16_split_y', 470)
nir_op_unpack_snorm_2x16 = nir_op.define('nir_op_unpack_snorm_2x16', 471)
nir_op_unpack_snorm_4x8 = nir_op.define('nir_op_unpack_snorm_4x8', 472)
nir_op_unpack_unorm_2x16 = nir_op.define('nir_op_unpack_unorm_2x16', 473)
nir_op_unpack_unorm_4x8 = nir_op.define('nir_op_unpack_unorm_4x8', 474)
nir_op_urhadd = nir_op.define('nir_op_urhadd', 475)
nir_op_urol = nir_op.define('nir_op_urol', 476)
nir_op_uror = nir_op.define('nir_op_uror', 477)
nir_op_usadd_4x8_vc4 = nir_op.define('nir_op_usadd_4x8_vc4', 478)
nir_op_ushr = nir_op.define('nir_op_ushr', 479)
nir_op_ussub_4x8_vc4 = nir_op.define('nir_op_ussub_4x8_vc4', 480)
nir_op_usub_borrow = nir_op.define('nir_op_usub_borrow', 481)
nir_op_usub_sat = nir_op.define('nir_op_usub_sat', 482)
nir_op_vec16 = nir_op.define('nir_op_vec16', 483)
nir_op_vec2 = nir_op.define('nir_op_vec2', 484)
nir_op_vec3 = nir_op.define('nir_op_vec3', 485)
nir_op_vec4 = nir_op.define('nir_op_vec4', 486)
nir_op_vec5 = nir_op.define('nir_op_vec5', 487)
nir_op_vec8 = nir_op.define('nir_op_vec8', 488)
nir_last_opcode = nir_op.define('nir_last_opcode', 488)
nir_num_opcodes = nir_op.define('nir_num_opcodes', 489)

try: (nir_type_conversion_op:=dll.nir_type_conversion_op).restype, nir_type_conversion_op.argtypes = nir_op, [nir_alu_type, nir_alu_type, nir_rounding_mode]
except AttributeError: pass

nir_atomic_op = CEnum(ctypes.c_uint32)
nir_atomic_op_iadd = nir_atomic_op.define('nir_atomic_op_iadd', 0)
nir_atomic_op_imin = nir_atomic_op.define('nir_atomic_op_imin', 1)
nir_atomic_op_umin = nir_atomic_op.define('nir_atomic_op_umin', 2)
nir_atomic_op_imax = nir_atomic_op.define('nir_atomic_op_imax', 3)
nir_atomic_op_umax = nir_atomic_op.define('nir_atomic_op_umax', 4)
nir_atomic_op_iand = nir_atomic_op.define('nir_atomic_op_iand', 5)
nir_atomic_op_ior = nir_atomic_op.define('nir_atomic_op_ior', 6)
nir_atomic_op_ixor = nir_atomic_op.define('nir_atomic_op_ixor', 7)
nir_atomic_op_xchg = nir_atomic_op.define('nir_atomic_op_xchg', 8)
nir_atomic_op_fadd = nir_atomic_op.define('nir_atomic_op_fadd', 9)
nir_atomic_op_fmin = nir_atomic_op.define('nir_atomic_op_fmin', 10)
nir_atomic_op_fmax = nir_atomic_op.define('nir_atomic_op_fmax', 11)
nir_atomic_op_cmpxchg = nir_atomic_op.define('nir_atomic_op_cmpxchg', 12)
nir_atomic_op_fcmpxchg = nir_atomic_op.define('nir_atomic_op_fcmpxchg', 13)
nir_atomic_op_inc_wrap = nir_atomic_op.define('nir_atomic_op_inc_wrap', 14)
nir_atomic_op_dec_wrap = nir_atomic_op.define('nir_atomic_op_dec_wrap', 15)
nir_atomic_op_ordered_add_gfx12_amd = nir_atomic_op.define('nir_atomic_op_ordered_add_gfx12_amd', 16)

try: (nir_atomic_op_to_alu:=dll.nir_atomic_op_to_alu).restype, nir_atomic_op_to_alu.argtypes = nir_op, [nir_atomic_op]
except AttributeError: pass

try: (nir_op_vec:=dll.nir_op_vec).restype, nir_op_vec.argtypes = nir_op, [ctypes.c_uint32]
except AttributeError: pass

try: (nir_op_is_vec:=dll.nir_op_is_vec).restype, nir_op_is_vec.argtypes = ctypes.c_bool, [nir_op]
except AttributeError: pass

nir_op_algebraic_property = CEnum(ctypes.c_uint32)
NIR_OP_IS_2SRC_COMMUTATIVE = nir_op_algebraic_property.define('NIR_OP_IS_2SRC_COMMUTATIVE', 1)
NIR_OP_IS_ASSOCIATIVE = nir_op_algebraic_property.define('NIR_OP_IS_ASSOCIATIVE', 2)
NIR_OP_IS_SELECTION = nir_op_algebraic_property.define('NIR_OP_IS_SELECTION', 4)

class struct_nir_op_info(Struct): pass
struct_nir_op_info._fields_ = [
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('num_inputs', uint8_t),
  ('output_size', uint8_t),
  ('output_type', nir_alu_type),
  ('input_sizes', (uint8_t * 16)),
  ('input_types', (nir_alu_type * 16)),
  ('algebraic_properties', nir_op_algebraic_property),
  ('is_conversion', ctypes.c_bool),
]
nir_op_info = struct_nir_op_info
try: nir_op_infos = (nir_op_info * 489).in_dll(dll, 'nir_op_infos')
except (ValueError,AttributeError): pass
class struct_nir_alu_instr(Struct): pass
struct_nir_alu_instr._fields_ = [
  ('instr', nir_instr),
  ('op', nir_op),
  ('exact', ctypes.c_bool,1),
  ('no_signed_wrap', ctypes.c_bool,1),
  ('no_unsigned_wrap', ctypes.c_bool,1),
  ('fp_fast_math', uint32_t,9),
  ('def', nir_def),
  ('src', (nir_alu_src * 0)),
]
nir_alu_instr = struct_nir_alu_instr
try: (nir_alu_src_copy:=dll.nir_alu_src_copy).restype, nir_alu_src_copy.argtypes = None, [ctypes.POINTER(nir_alu_src), ctypes.POINTER(nir_alu_src)]
except AttributeError: pass

try: (nir_alu_instr_src_read_mask:=dll.nir_alu_instr_src_read_mask).restype, nir_alu_instr_src_read_mask.argtypes = nir_component_mask_t, [ctypes.POINTER(nir_alu_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_ssa_alu_instr_src_components:=dll.nir_ssa_alu_instr_src_components).restype, nir_ssa_alu_instr_src_components.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_alu_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_alu_instr_is_comparison:=dll.nir_alu_instr_is_comparison).restype, nir_alu_instr_is_comparison.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr)]
except AttributeError: pass

try: (nir_const_value_negative_equal:=dll.nir_const_value_negative_equal).restype, nir_const_value_negative_equal.argtypes = ctypes.c_bool, [nir_const_value, nir_const_value, nir_alu_type]
except AttributeError: pass

try: (nir_alu_srcs_equal:=dll.nir_alu_srcs_equal).restype, nir_alu_srcs_equal.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr), ctypes.POINTER(nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_alu_srcs_negative_equal_typed:=dll.nir_alu_srcs_negative_equal_typed).restype, nir_alu_srcs_negative_equal_typed.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr), ctypes.POINTER(nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32, nir_alu_type]
except AttributeError: pass

try: (nir_alu_srcs_negative_equal:=dll.nir_alu_srcs_negative_equal).restype, nir_alu_srcs_negative_equal.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr), ctypes.POINTER(nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_alu_src_is_trivial_ssa:=dll.nir_alu_src_is_trivial_ssa).restype, nir_alu_src_is_trivial_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_alu_instr), ctypes.c_uint32]
except AttributeError: pass

nir_deref_type = CEnum(ctypes.c_uint32)
nir_deref_type_var = nir_deref_type.define('nir_deref_type_var', 0)
nir_deref_type_array = nir_deref_type.define('nir_deref_type_array', 1)
nir_deref_type_array_wildcard = nir_deref_type.define('nir_deref_type_array_wildcard', 2)
nir_deref_type_ptr_as_array = nir_deref_type.define('nir_deref_type_ptr_as_array', 3)
nir_deref_type_struct = nir_deref_type.define('nir_deref_type_struct', 4)
nir_deref_type_cast = nir_deref_type.define('nir_deref_type_cast', 5)

class struct_nir_deref_instr(Struct): pass
nir_variable_mode = CEnum(ctypes.c_uint32)
nir_var_system_value = nir_variable_mode.define('nir_var_system_value', 1)
nir_var_uniform = nir_variable_mode.define('nir_var_uniform', 2)
nir_var_shader_in = nir_variable_mode.define('nir_var_shader_in', 4)
nir_var_shader_out = nir_variable_mode.define('nir_var_shader_out', 8)
nir_var_image = nir_variable_mode.define('nir_var_image', 16)
nir_var_shader_call_data = nir_variable_mode.define('nir_var_shader_call_data', 32)
nir_var_ray_hit_attrib = nir_variable_mode.define('nir_var_ray_hit_attrib', 64)
nir_var_mem_ubo = nir_variable_mode.define('nir_var_mem_ubo', 128)
nir_var_mem_push_const = nir_variable_mode.define('nir_var_mem_push_const', 256)
nir_var_mem_ssbo = nir_variable_mode.define('nir_var_mem_ssbo', 512)
nir_var_mem_constant = nir_variable_mode.define('nir_var_mem_constant', 1024)
nir_var_mem_task_payload = nir_variable_mode.define('nir_var_mem_task_payload', 2048)
nir_var_mem_node_payload = nir_variable_mode.define('nir_var_mem_node_payload', 4096)
nir_var_mem_node_payload_in = nir_variable_mode.define('nir_var_mem_node_payload_in', 8192)
nir_var_function_in = nir_variable_mode.define('nir_var_function_in', 16384)
nir_var_function_out = nir_variable_mode.define('nir_var_function_out', 32768)
nir_var_function_inout = nir_variable_mode.define('nir_var_function_inout', 65536)
nir_var_shader_temp = nir_variable_mode.define('nir_var_shader_temp', 131072)
nir_var_function_temp = nir_variable_mode.define('nir_var_function_temp', 262144)
nir_var_mem_shared = nir_variable_mode.define('nir_var_mem_shared', 524288)
nir_var_mem_global = nir_variable_mode.define('nir_var_mem_global', 1048576)
nir_var_mem_generic = nir_variable_mode.define('nir_var_mem_generic', 1966080)
nir_var_read_only_modes = nir_variable_mode.define('nir_var_read_only_modes', 1159)
nir_var_vec_indexable_modes = nir_variable_mode.define('nir_var_vec_indexable_modes', 1969033)
nir_num_variable_modes = nir_variable_mode.define('nir_num_variable_modes', 21)
nir_var_all = nir_variable_mode.define('nir_var_all', 2097151)

class struct_nir_deref_instr_0(ctypes.Union): pass
struct_nir_deref_instr_0._fields_ = [
  ('var', ctypes.POINTER(nir_variable)),
  ('parent', nir_src),
]
class struct_nir_deref_instr_1(ctypes.Union): pass
class struct_nir_deref_instr_1_arr(Struct): pass
struct_nir_deref_instr_1_arr._fields_ = [
  ('index', nir_src),
  ('in_bounds', ctypes.c_bool),
]
class struct_nir_deref_instr_1_strct(Struct): pass
struct_nir_deref_instr_1_strct._fields_ = [
  ('index', ctypes.c_uint32),
]
class struct_nir_deref_instr_1_cast(Struct): pass
struct_nir_deref_instr_1_cast._fields_ = [
  ('ptr_stride', ctypes.c_uint32),
  ('align_mul', ctypes.c_uint32),
  ('align_offset', ctypes.c_uint32),
]
struct_nir_deref_instr_1._fields_ = [
  ('arr', struct_nir_deref_instr_1_arr),
  ('strct', struct_nir_deref_instr_1_strct),
  ('cast', struct_nir_deref_instr_1_cast),
]
struct_nir_deref_instr._anonymous_ = ['_0', '_1']
struct_nir_deref_instr._fields_ = [
  ('instr', nir_instr),
  ('deref_type', nir_deref_type),
  ('modes', nir_variable_mode),
  ('type', ctypes.POINTER(struct_glsl_type)),
  ('_0', struct_nir_deref_instr_0),
  ('_1', struct_nir_deref_instr_1),
  ('def', nir_def),
]
nir_deref_instr = struct_nir_deref_instr
try: (nir_deref_cast_is_trivial:=dll.nir_deref_cast_is_trivial).restype, nir_deref_cast_is_trivial.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

try: (nir_deref_instr_has_indirect:=dll.nir_deref_instr_has_indirect).restype, nir_deref_instr_has_indirect.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

try: (nir_deref_instr_is_known_out_of_bounds:=dll.nir_deref_instr_is_known_out_of_bounds).restype, nir_deref_instr_is_known_out_of_bounds.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

nir_deref_instr_has_complex_use_options = CEnum(ctypes.c_uint32)
nir_deref_instr_has_complex_use_allow_memcpy_src = nir_deref_instr_has_complex_use_options.define('nir_deref_instr_has_complex_use_allow_memcpy_src', 1)
nir_deref_instr_has_complex_use_allow_memcpy_dst = nir_deref_instr_has_complex_use_options.define('nir_deref_instr_has_complex_use_allow_memcpy_dst', 2)
nir_deref_instr_has_complex_use_allow_atomics = nir_deref_instr_has_complex_use_options.define('nir_deref_instr_has_complex_use_allow_atomics', 4)

try: (nir_deref_instr_has_complex_use:=dll.nir_deref_instr_has_complex_use).restype, nir_deref_instr_has_complex_use.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr), nir_deref_instr_has_complex_use_options]
except AttributeError: pass

try: (nir_deref_instr_remove_if_unused:=dll.nir_deref_instr_remove_if_unused).restype, nir_deref_instr_remove_if_unused.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

try: (nir_deref_instr_array_stride:=dll.nir_deref_instr_array_stride).restype, nir_deref_instr_array_stride.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

class struct_nir_call_instr(Struct): pass
class struct_nir_function(Struct): pass
nir_function = struct_nir_function
class struct_nir_shader(Struct): pass
nir_shader = struct_nir_shader
class struct_gc_ctx(Struct): pass
gc_ctx = struct_gc_ctx
class struct_nir_shader_compiler_options(Struct): pass
nir_shader_compiler_options = struct_nir_shader_compiler_options
class const_struct_nir_instr(Struct): pass
const_struct_nir_instr._fields_ = [
  ('node', struct_exec_node),
  ('block', ctypes.POINTER(nir_block)),
  ('type', nir_instr_type),
  ('pass_flags', uint8_t),
  ('has_debug_info', ctypes.c_bool),
  ('index', uint32_t),
]
nir_instr_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(const_struct_nir_instr), ctypes.c_void_p)
nir_lower_int64_options = CEnum(ctypes.c_uint32)
nir_lower_imul64 = nir_lower_int64_options.define('nir_lower_imul64', 1)
nir_lower_isign64 = nir_lower_int64_options.define('nir_lower_isign64', 2)
nir_lower_divmod64 = nir_lower_int64_options.define('nir_lower_divmod64', 4)
nir_lower_imul_high64 = nir_lower_int64_options.define('nir_lower_imul_high64', 8)
nir_lower_bcsel64 = nir_lower_int64_options.define('nir_lower_bcsel64', 16)
nir_lower_icmp64 = nir_lower_int64_options.define('nir_lower_icmp64', 32)
nir_lower_iadd64 = nir_lower_int64_options.define('nir_lower_iadd64', 64)
nir_lower_iabs64 = nir_lower_int64_options.define('nir_lower_iabs64', 128)
nir_lower_ineg64 = nir_lower_int64_options.define('nir_lower_ineg64', 256)
nir_lower_logic64 = nir_lower_int64_options.define('nir_lower_logic64', 512)
nir_lower_minmax64 = nir_lower_int64_options.define('nir_lower_minmax64', 1024)
nir_lower_shift64 = nir_lower_int64_options.define('nir_lower_shift64', 2048)
nir_lower_imul_2x32_64 = nir_lower_int64_options.define('nir_lower_imul_2x32_64', 4096)
nir_lower_extract64 = nir_lower_int64_options.define('nir_lower_extract64', 8192)
nir_lower_ufind_msb64 = nir_lower_int64_options.define('nir_lower_ufind_msb64', 16384)
nir_lower_bit_count64 = nir_lower_int64_options.define('nir_lower_bit_count64', 32768)
nir_lower_subgroup_shuffle64 = nir_lower_int64_options.define('nir_lower_subgroup_shuffle64', 65536)
nir_lower_scan_reduce_bitwise64 = nir_lower_int64_options.define('nir_lower_scan_reduce_bitwise64', 131072)
nir_lower_scan_reduce_iadd64 = nir_lower_int64_options.define('nir_lower_scan_reduce_iadd64', 262144)
nir_lower_vote_ieq64 = nir_lower_int64_options.define('nir_lower_vote_ieq64', 524288)
nir_lower_usub_sat64 = nir_lower_int64_options.define('nir_lower_usub_sat64', 1048576)
nir_lower_iadd_sat64 = nir_lower_int64_options.define('nir_lower_iadd_sat64', 2097152)
nir_lower_find_lsb64 = nir_lower_int64_options.define('nir_lower_find_lsb64', 4194304)
nir_lower_conv64 = nir_lower_int64_options.define('nir_lower_conv64', 8388608)
nir_lower_uadd_sat64 = nir_lower_int64_options.define('nir_lower_uadd_sat64', 16777216)
nir_lower_iadd3_64 = nir_lower_int64_options.define('nir_lower_iadd3_64', 33554432)
nir_lower_bitfield_reverse64 = nir_lower_int64_options.define('nir_lower_bitfield_reverse64', 67108864)
nir_lower_bitfield_extract64 = nir_lower_int64_options.define('nir_lower_bitfield_extract64', 134217728)

nir_lower_doubles_options = CEnum(ctypes.c_uint32)
nir_lower_drcp = nir_lower_doubles_options.define('nir_lower_drcp', 1)
nir_lower_dsqrt = nir_lower_doubles_options.define('nir_lower_dsqrt', 2)
nir_lower_drsq = nir_lower_doubles_options.define('nir_lower_drsq', 4)
nir_lower_dtrunc = nir_lower_doubles_options.define('nir_lower_dtrunc', 8)
nir_lower_dfloor = nir_lower_doubles_options.define('nir_lower_dfloor', 16)
nir_lower_dceil = nir_lower_doubles_options.define('nir_lower_dceil', 32)
nir_lower_dfract = nir_lower_doubles_options.define('nir_lower_dfract', 64)
nir_lower_dround_even = nir_lower_doubles_options.define('nir_lower_dround_even', 128)
nir_lower_dmod = nir_lower_doubles_options.define('nir_lower_dmod', 256)
nir_lower_dsub = nir_lower_doubles_options.define('nir_lower_dsub', 512)
nir_lower_ddiv = nir_lower_doubles_options.define('nir_lower_ddiv', 1024)
nir_lower_dsign = nir_lower_doubles_options.define('nir_lower_dsign', 2048)
nir_lower_dminmax = nir_lower_doubles_options.define('nir_lower_dminmax', 4096)
nir_lower_dsat = nir_lower_doubles_options.define('nir_lower_dsat', 8192)
nir_lower_fp64_full_software = nir_lower_doubles_options.define('nir_lower_fp64_full_software', 16384)

nir_divergence_options = CEnum(ctypes.c_uint32)
nir_divergence_single_prim_per_subgroup = nir_divergence_options.define('nir_divergence_single_prim_per_subgroup', 1)
nir_divergence_single_patch_per_tcs_subgroup = nir_divergence_options.define('nir_divergence_single_patch_per_tcs_subgroup', 2)
nir_divergence_single_patch_per_tes_subgroup = nir_divergence_options.define('nir_divergence_single_patch_per_tes_subgroup', 4)
nir_divergence_view_index_uniform = nir_divergence_options.define('nir_divergence_view_index_uniform', 8)
nir_divergence_single_frag_shading_rate_per_subgroup = nir_divergence_options.define('nir_divergence_single_frag_shading_rate_per_subgroup', 16)
nir_divergence_multiple_workgroup_per_compute_subgroup = nir_divergence_options.define('nir_divergence_multiple_workgroup_per_compute_subgroup', 32)
nir_divergence_shader_record_ptr_uniform = nir_divergence_options.define('nir_divergence_shader_record_ptr_uniform', 64)
nir_divergence_uniform_load_tears = nir_divergence_options.define('nir_divergence_uniform_load_tears', 128)
nir_divergence_ignore_undef_if_phi_srcs = nir_divergence_options.define('nir_divergence_ignore_undef_if_phi_srcs', 256)

nir_io_options = CEnum(ctypes.c_uint32)
nir_io_has_flexible_input_interpolation_except_flat = nir_io_options.define('nir_io_has_flexible_input_interpolation_except_flat', 1)
nir_io_dont_use_pos_for_non_fs_varyings = nir_io_options.define('nir_io_dont_use_pos_for_non_fs_varyings', 2)
nir_io_16bit_input_output_support = nir_io_options.define('nir_io_16bit_input_output_support', 4)
nir_io_mediump_is_32bit = nir_io_options.define('nir_io_mediump_is_32bit', 8)
nir_io_prefer_scalar_fs_inputs = nir_io_options.define('nir_io_prefer_scalar_fs_inputs', 16)
nir_io_mix_convergent_flat_with_interpolated = nir_io_options.define('nir_io_mix_convergent_flat_with_interpolated', 32)
nir_io_vectorizer_ignores_types = nir_io_options.define('nir_io_vectorizer_ignores_types', 64)
nir_io_always_interpolate_convergent_fs_inputs = nir_io_options.define('nir_io_always_interpolate_convergent_fs_inputs', 128)
nir_io_compaction_rotates_color_channels = nir_io_options.define('nir_io_compaction_rotates_color_channels', 256)
nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups = nir_io_options.define('nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups', 512)
nir_io_radv_intrinsic_component_workaround = nir_io_options.define('nir_io_radv_intrinsic_component_workaround', 1024)
nir_io_has_intrinsics = nir_io_options.define('nir_io_has_intrinsics', 65536)
nir_io_separate_clip_cull_distance_arrays = nir_io_options.define('nir_io_separate_clip_cull_distance_arrays', 131072)

struct_nir_shader_compiler_options._fields_ = [
  ('lower_fdiv', ctypes.c_bool),
  ('lower_ffma16', ctypes.c_bool),
  ('lower_ffma32', ctypes.c_bool),
  ('lower_ffma64', ctypes.c_bool),
  ('fuse_ffma16', ctypes.c_bool),
  ('fuse_ffma32', ctypes.c_bool),
  ('fuse_ffma64', ctypes.c_bool),
  ('lower_flrp16', ctypes.c_bool),
  ('lower_flrp32', ctypes.c_bool),
  ('lower_flrp64', ctypes.c_bool),
  ('lower_fpow', ctypes.c_bool),
  ('lower_fsat', ctypes.c_bool),
  ('lower_fsqrt', ctypes.c_bool),
  ('lower_sincos', ctypes.c_bool),
  ('lower_fmod', ctypes.c_bool),
  ('lower_bitfield_extract8', ctypes.c_bool),
  ('lower_bitfield_extract16', ctypes.c_bool),
  ('lower_bitfield_extract', ctypes.c_bool),
  ('lower_bitfield_insert', ctypes.c_bool),
  ('lower_bitfield_reverse', ctypes.c_bool),
  ('lower_bit_count', ctypes.c_bool),
  ('lower_ifind_msb', ctypes.c_bool),
  ('lower_ufind_msb', ctypes.c_bool),
  ('lower_find_lsb', ctypes.c_bool),
  ('lower_uadd_carry', ctypes.c_bool),
  ('lower_usub_borrow', ctypes.c_bool),
  ('lower_mul_high', ctypes.c_bool),
  ('lower_mul_high16', ctypes.c_bool),
  ('lower_fneg', ctypes.c_bool),
  ('lower_ineg', ctypes.c_bool),
  ('lower_fisnormal', ctypes.c_bool),
  ('lower_scmp', ctypes.c_bool),
  ('lower_vector_cmp', ctypes.c_bool),
  ('lower_bitops', ctypes.c_bool),
  ('lower_isign', ctypes.c_bool),
  ('lower_fsign', ctypes.c_bool),
  ('lower_iabs', ctypes.c_bool),
  ('lower_umax', ctypes.c_bool),
  ('lower_umin', ctypes.c_bool),
  ('lower_fminmax_signed_zero', ctypes.c_bool),
  ('lower_fdph', ctypes.c_bool),
  ('fdot_replicates', ctypes.c_bool),
  ('lower_ffloor', ctypes.c_bool),
  ('lower_ffract', ctypes.c_bool),
  ('lower_fceil', ctypes.c_bool),
  ('lower_ftrunc', ctypes.c_bool),
  ('lower_fround_even', ctypes.c_bool),
  ('lower_ldexp', ctypes.c_bool),
  ('lower_pack_half_2x16', ctypes.c_bool),
  ('lower_pack_unorm_2x16', ctypes.c_bool),
  ('lower_pack_snorm_2x16', ctypes.c_bool),
  ('lower_pack_unorm_4x8', ctypes.c_bool),
  ('lower_pack_snorm_4x8', ctypes.c_bool),
  ('lower_pack_64_2x32', ctypes.c_bool),
  ('lower_pack_64_4x16', ctypes.c_bool),
  ('lower_pack_32_2x16', ctypes.c_bool),
  ('lower_pack_64_2x32_split', ctypes.c_bool),
  ('lower_pack_32_2x16_split', ctypes.c_bool),
  ('lower_unpack_half_2x16', ctypes.c_bool),
  ('lower_unpack_unorm_2x16', ctypes.c_bool),
  ('lower_unpack_snorm_2x16', ctypes.c_bool),
  ('lower_unpack_unorm_4x8', ctypes.c_bool),
  ('lower_unpack_snorm_4x8', ctypes.c_bool),
  ('lower_unpack_64_2x32_split', ctypes.c_bool),
  ('lower_unpack_32_2x16_split', ctypes.c_bool),
  ('lower_pack_split', ctypes.c_bool),
  ('lower_extract_byte', ctypes.c_bool),
  ('lower_extract_word', ctypes.c_bool),
  ('lower_insert_byte', ctypes.c_bool),
  ('lower_insert_word', ctypes.c_bool),
  ('vertex_id_zero_based', ctypes.c_bool),
  ('lower_base_vertex', ctypes.c_bool),
  ('instance_id_includes_base_index', ctypes.c_bool),
  ('lower_helper_invocation', ctypes.c_bool),
  ('optimize_sample_mask_in', ctypes.c_bool),
  ('optimize_load_front_face_fsign', ctypes.c_bool),
  ('optimize_quad_vote_to_reduce', ctypes.c_bool),
  ('lower_cs_local_index_to_id', ctypes.c_bool),
  ('lower_cs_local_id_to_index', ctypes.c_bool),
  ('has_cs_global_id', ctypes.c_bool),
  ('lower_device_index_to_zero', ctypes.c_bool),
  ('lower_wpos_pntc', ctypes.c_bool),
  ('lower_hadd', ctypes.c_bool),
  ('lower_hadd64', ctypes.c_bool),
  ('lower_uadd_sat', ctypes.c_bool),
  ('lower_usub_sat', ctypes.c_bool),
  ('lower_iadd_sat', ctypes.c_bool),
  ('lower_mul_32x16', ctypes.c_bool),
  ('lower_bfloat16_conversions', ctypes.c_bool),
  ('vectorize_tess_levels', ctypes.c_bool),
  ('lower_to_scalar', ctypes.c_bool),
  ('lower_to_scalar_filter', nir_instr_filter_cb),
  ('vectorize_vec2_16bit', ctypes.c_bool),
  ('unify_interfaces', ctypes.c_bool),
  ('lower_interpolate_at', ctypes.c_bool),
  ('lower_mul_2x32_64', ctypes.c_bool),
  ('has_rotate8', ctypes.c_bool),
  ('has_rotate16', ctypes.c_bool),
  ('has_rotate32', ctypes.c_bool),
  ('has_shfr32', ctypes.c_bool),
  ('has_iadd3', ctypes.c_bool),
  ('has_amul', ctypes.c_bool),
  ('has_imul24', ctypes.c_bool),
  ('has_umul24', ctypes.c_bool),
  ('has_mul24_relaxed', ctypes.c_bool),
  ('has_imad32', ctypes.c_bool),
  ('has_umad24', ctypes.c_bool),
  ('has_fused_comp_and_csel', ctypes.c_bool),
  ('has_icsel_eqz64', ctypes.c_bool),
  ('has_icsel_eqz32', ctypes.c_bool),
  ('has_icsel_eqz16', ctypes.c_bool),
  ('has_fneo_fcmpu', ctypes.c_bool),
  ('has_ford_funord', ctypes.c_bool),
  ('has_fsub', ctypes.c_bool),
  ('has_isub', ctypes.c_bool),
  ('has_pack_32_4x8', ctypes.c_bool),
  ('has_texture_scaling', ctypes.c_bool),
  ('has_sdot_4x8', ctypes.c_bool),
  ('has_udot_4x8', ctypes.c_bool),
  ('has_sudot_4x8', ctypes.c_bool),
  ('has_sdot_4x8_sat', ctypes.c_bool),
  ('has_udot_4x8_sat', ctypes.c_bool),
  ('has_sudot_4x8_sat', ctypes.c_bool),
  ('has_dot_2x16', ctypes.c_bool),
  ('has_bfdot2_bfadd', ctypes.c_bool),
  ('has_fmulz', ctypes.c_bool),
  ('has_fmulz_no_denorms', ctypes.c_bool),
  ('has_find_msb_rev', ctypes.c_bool),
  ('has_pack_half_2x16_rtz', ctypes.c_bool),
  ('has_bit_test', ctypes.c_bool),
  ('has_bfe', ctypes.c_bool),
  ('has_bfm', ctypes.c_bool),
  ('has_bfi', ctypes.c_bool),
  ('has_bitfield_select', ctypes.c_bool),
  ('has_uclz', ctypes.c_bool),
  ('has_msad', ctypes.c_bool),
  ('has_f2e4m3fn_satfn', ctypes.c_bool),
  ('has_load_global_bounded', ctypes.c_bool),
  ('intel_vec4', ctypes.c_bool),
  ('avoid_ternary_with_two_constants', ctypes.c_bool),
  ('support_8bit_alu', ctypes.c_bool),
  ('support_16bit_alu', ctypes.c_bool),
  ('max_unroll_iterations', ctypes.c_uint32),
  ('max_unroll_iterations_aggressive', ctypes.c_uint32),
  ('max_unroll_iterations_fp64', ctypes.c_uint32),
  ('lower_uniforms_to_ubo', ctypes.c_bool),
  ('force_indirect_unrolling_sampler', ctypes.c_bool),
  ('no_integers', ctypes.c_bool),
  ('force_indirect_unrolling', nir_variable_mode),
  ('driver_functions', ctypes.c_bool),
  ('late_lower_int64', ctypes.c_bool),
  ('lower_int64_options', nir_lower_int64_options),
  ('lower_doubles_options', nir_lower_doubles_options),
  ('divergence_analysis_options', nir_divergence_options),
  ('support_indirect_inputs', uint8_t),
  ('support_indirect_outputs', uint8_t),
  ('lower_image_offset_to_range_base', ctypes.c_bool),
  ('lower_atomic_offset_to_range_base', ctypes.c_bool),
  ('preserve_mediump', ctypes.c_bool),
  ('lower_fquantize2f16', ctypes.c_bool),
  ('force_f2f16_rtz', ctypes.c_bool),
  ('lower_layer_fs_input_to_sysval', ctypes.c_bool),
  ('compact_arrays', ctypes.c_bool),
  ('discard_is_demote', ctypes.c_bool),
  ('has_ddx_intrinsics', ctypes.c_bool),
  ('scalarize_ddx', ctypes.c_bool),
  ('per_view_unique_driver_locations', ctypes.c_bool),
  ('compact_view_index', ctypes.c_bool),
  ('io_options', nir_io_options),
  ('skip_lower_packing_ops', ctypes.c_uint32),
  ('lower_mediump_io', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_nir_shader))),
  ('varying_expression_max_cost', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader))),
  ('varying_estimate_instr_cost', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr))),
  ('max_varying_expression_cost', ctypes.c_uint32),
]
class struct_shader_info(Struct): pass
blake3_hash = (ctypes.c_ubyte * 32)
enum_pipe_shader_type = CEnum(ctypes.c_int32)
MESA_SHADER_NONE = enum_pipe_shader_type.define('MESA_SHADER_NONE', -1)
MESA_SHADER_VERTEX = enum_pipe_shader_type.define('MESA_SHADER_VERTEX', 0)
PIPE_SHADER_VERTEX = enum_pipe_shader_type.define('PIPE_SHADER_VERTEX', 0)
MESA_SHADER_TESS_CTRL = enum_pipe_shader_type.define('MESA_SHADER_TESS_CTRL', 1)
PIPE_SHADER_TESS_CTRL = enum_pipe_shader_type.define('PIPE_SHADER_TESS_CTRL', 1)
MESA_SHADER_TESS_EVAL = enum_pipe_shader_type.define('MESA_SHADER_TESS_EVAL', 2)
PIPE_SHADER_TESS_EVAL = enum_pipe_shader_type.define('PIPE_SHADER_TESS_EVAL', 2)
MESA_SHADER_GEOMETRY = enum_pipe_shader_type.define('MESA_SHADER_GEOMETRY', 3)
PIPE_SHADER_GEOMETRY = enum_pipe_shader_type.define('PIPE_SHADER_GEOMETRY', 3)
MESA_SHADER_FRAGMENT = enum_pipe_shader_type.define('MESA_SHADER_FRAGMENT', 4)
PIPE_SHADER_FRAGMENT = enum_pipe_shader_type.define('PIPE_SHADER_FRAGMENT', 4)
MESA_SHADER_COMPUTE = enum_pipe_shader_type.define('MESA_SHADER_COMPUTE', 5)
PIPE_SHADER_COMPUTE = enum_pipe_shader_type.define('PIPE_SHADER_COMPUTE', 5)
PIPE_SHADER_TYPES = enum_pipe_shader_type.define('PIPE_SHADER_TYPES', 6)
MESA_SHADER_TASK = enum_pipe_shader_type.define('MESA_SHADER_TASK', 6)
PIPE_SHADER_TASK = enum_pipe_shader_type.define('PIPE_SHADER_TASK', 6)
MESA_SHADER_MESH = enum_pipe_shader_type.define('MESA_SHADER_MESH', 7)
PIPE_SHADER_MESH = enum_pipe_shader_type.define('PIPE_SHADER_MESH', 7)
PIPE_SHADER_MESH_TYPES = enum_pipe_shader_type.define('PIPE_SHADER_MESH_TYPES', 8)
MESA_SHADER_RAYGEN = enum_pipe_shader_type.define('MESA_SHADER_RAYGEN', 8)
MESA_SHADER_ANY_HIT = enum_pipe_shader_type.define('MESA_SHADER_ANY_HIT', 9)
MESA_SHADER_CLOSEST_HIT = enum_pipe_shader_type.define('MESA_SHADER_CLOSEST_HIT', 10)
MESA_SHADER_MISS = enum_pipe_shader_type.define('MESA_SHADER_MISS', 11)
MESA_SHADER_INTERSECTION = enum_pipe_shader_type.define('MESA_SHADER_INTERSECTION', 12)
MESA_SHADER_CALLABLE = enum_pipe_shader_type.define('MESA_SHADER_CALLABLE', 13)
MESA_SHADER_KERNEL = enum_pipe_shader_type.define('MESA_SHADER_KERNEL', 14)

gl_shader_stage = enum_pipe_shader_type
enum_gl_subgroup_size = CEnum(ctypes.c_ubyte)
SUBGROUP_SIZE_VARYING = enum_gl_subgroup_size.define('SUBGROUP_SIZE_VARYING', 0)
SUBGROUP_SIZE_UNIFORM = enum_gl_subgroup_size.define('SUBGROUP_SIZE_UNIFORM', 1)
SUBGROUP_SIZE_API_CONSTANT = enum_gl_subgroup_size.define('SUBGROUP_SIZE_API_CONSTANT', 2)
SUBGROUP_SIZE_FULL_SUBGROUPS = enum_gl_subgroup_size.define('SUBGROUP_SIZE_FULL_SUBGROUPS', 3)
SUBGROUP_SIZE_REQUIRE_4 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_4', 4)
SUBGROUP_SIZE_REQUIRE_8 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_8', 8)
SUBGROUP_SIZE_REQUIRE_16 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_16', 16)
SUBGROUP_SIZE_REQUIRE_32 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_32', 32)
SUBGROUP_SIZE_REQUIRE_64 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_64', 64)
SUBGROUP_SIZE_REQUIRE_128 = enum_gl_subgroup_size.define('SUBGROUP_SIZE_REQUIRE_128', 128)

enum_gl_derivative_group = CEnum(ctypes.c_uint32)
DERIVATIVE_GROUP_NONE = enum_gl_derivative_group.define('DERIVATIVE_GROUP_NONE', 0)
DERIVATIVE_GROUP_QUADS = enum_gl_derivative_group.define('DERIVATIVE_GROUP_QUADS', 1)
DERIVATIVE_GROUP_LINEAR = enum_gl_derivative_group.define('DERIVATIVE_GROUP_LINEAR', 2)

class struct_shader_info_0(ctypes.Union): pass
class struct_shader_info_0_vs(Struct): pass
struct_shader_info_0_vs._fields_ = [
  ('double_inputs', uint64_t),
  ('blit_sgprs_amd', uint8_t,4),
  ('tes_agx', ctypes.c_bool,1),
  ('window_space_position', ctypes.c_bool,1),
  ('needs_edge_flag', ctypes.c_bool,1),
]
class struct_shader_info_0_gs(Struct): pass
enum_mesa_prim = CEnum(ctypes.c_ubyte)
MESA_PRIM_POINTS = enum_mesa_prim.define('MESA_PRIM_POINTS', 0)
MESA_PRIM_LINES = enum_mesa_prim.define('MESA_PRIM_LINES', 1)
MESA_PRIM_LINE_LOOP = enum_mesa_prim.define('MESA_PRIM_LINE_LOOP', 2)
MESA_PRIM_LINE_STRIP = enum_mesa_prim.define('MESA_PRIM_LINE_STRIP', 3)
MESA_PRIM_TRIANGLES = enum_mesa_prim.define('MESA_PRIM_TRIANGLES', 4)
MESA_PRIM_TRIANGLE_STRIP = enum_mesa_prim.define('MESA_PRIM_TRIANGLE_STRIP', 5)
MESA_PRIM_TRIANGLE_FAN = enum_mesa_prim.define('MESA_PRIM_TRIANGLE_FAN', 6)
MESA_PRIM_QUADS = enum_mesa_prim.define('MESA_PRIM_QUADS', 7)
MESA_PRIM_QUAD_STRIP = enum_mesa_prim.define('MESA_PRIM_QUAD_STRIP', 8)
MESA_PRIM_POLYGON = enum_mesa_prim.define('MESA_PRIM_POLYGON', 9)
MESA_PRIM_LINES_ADJACENCY = enum_mesa_prim.define('MESA_PRIM_LINES_ADJACENCY', 10)
MESA_PRIM_LINE_STRIP_ADJACENCY = enum_mesa_prim.define('MESA_PRIM_LINE_STRIP_ADJACENCY', 11)
MESA_PRIM_TRIANGLES_ADJACENCY = enum_mesa_prim.define('MESA_PRIM_TRIANGLES_ADJACENCY', 12)
MESA_PRIM_TRIANGLE_STRIP_ADJACENCY = enum_mesa_prim.define('MESA_PRIM_TRIANGLE_STRIP_ADJACENCY', 13)
MESA_PRIM_PATCHES = enum_mesa_prim.define('MESA_PRIM_PATCHES', 14)
MESA_PRIM_MAX = enum_mesa_prim.define('MESA_PRIM_MAX', 14)
MESA_PRIM_COUNT = enum_mesa_prim.define('MESA_PRIM_COUNT', 15)
MESA_PRIM_UNKNOWN = enum_mesa_prim.define('MESA_PRIM_UNKNOWN', 28)

struct_shader_info_0_gs._fields_ = [
  ('output_primitive', enum_mesa_prim),
  ('input_primitive', enum_mesa_prim),
  ('vertices_out', uint16_t),
  ('invocations', uint8_t),
  ('vertices_in', uint8_t,3),
  ('uses_end_primitive', ctypes.c_bool,1),
  ('active_stream_mask', uint8_t,4),
]
class struct_shader_info_0_fs(Struct): pass
enum_gl_frag_depth_layout = CEnum(ctypes.c_uint32)
FRAG_DEPTH_LAYOUT_NONE = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_NONE', 0)
FRAG_DEPTH_LAYOUT_ANY = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_ANY', 1)
FRAG_DEPTH_LAYOUT_GREATER = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_GREATER', 2)
FRAG_DEPTH_LAYOUT_LESS = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_LESS', 3)
FRAG_DEPTH_LAYOUT_UNCHANGED = enum_gl_frag_depth_layout.define('FRAG_DEPTH_LAYOUT_UNCHANGED', 4)

enum_gl_frag_stencil_layout = CEnum(ctypes.c_uint32)
FRAG_STENCIL_LAYOUT_NONE = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_NONE', 0)
FRAG_STENCIL_LAYOUT_ANY = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_ANY', 1)
FRAG_STENCIL_LAYOUT_GREATER = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_GREATER', 2)
FRAG_STENCIL_LAYOUT_LESS = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_LESS', 3)
FRAG_STENCIL_LAYOUT_UNCHANGED = enum_gl_frag_stencil_layout.define('FRAG_STENCIL_LAYOUT_UNCHANGED', 4)

struct_shader_info_0_fs._fields_ = [
  ('uses_discard', ctypes.c_bool,1),
  ('uses_fbfetch_output', ctypes.c_bool,1),
  ('fbfetch_coherent', ctypes.c_bool,1),
  ('color_is_dual_source', ctypes.c_bool,1),
  ('require_full_quads', ctypes.c_bool,1),
  ('quad_derivatives', ctypes.c_bool,1),
  ('needs_coarse_quad_helper_invocations', ctypes.c_bool,1),
  ('needs_full_quad_helper_invocations', ctypes.c_bool,1),
  ('uses_sample_qualifier', ctypes.c_bool,1),
  ('uses_sample_shading', ctypes.c_bool,1),
  ('early_fragment_tests', ctypes.c_bool,1),
  ('inner_coverage', ctypes.c_bool,1),
  ('post_depth_coverage', ctypes.c_bool,1),
  ('pixel_center_integer', ctypes.c_bool,1),
  ('origin_upper_left', ctypes.c_bool,1),
  ('pixel_interlock_ordered', ctypes.c_bool,1),
  ('pixel_interlock_unordered', ctypes.c_bool,1),
  ('sample_interlock_ordered', ctypes.c_bool,1),
  ('sample_interlock_unordered', ctypes.c_bool,1),
  ('untyped_color_outputs', ctypes.c_bool,1),
  ('depth_layout', enum_gl_frag_depth_layout,3),
  ('color0_interp', ctypes.c_uint32,3),
  ('color0_sample', ctypes.c_bool,1),
  ('color0_centroid', ctypes.c_bool,1),
  ('color1_interp', ctypes.c_uint32,3),
  ('color1_sample', ctypes.c_bool,1),
  ('color1_centroid', ctypes.c_bool,1),
  ('advanced_blend_modes', ctypes.c_uint32),
  ('early_and_late_fragment_tests', ctypes.c_bool,1),
  ('stencil_front_layout', enum_gl_frag_stencil_layout,3),
  ('stencil_back_layout', enum_gl_frag_stencil_layout,3),
]
class struct_shader_info_0_cs(Struct): pass
struct_shader_info_0_cs._fields_ = [
  ('workgroup_size_hint', (uint16_t * 3)),
  ('user_data_components_amd', uint8_t,4),
  ('has_variable_shared_mem', ctypes.c_bool,1),
  ('has_cooperative_matrix', ctypes.c_bool,1),
  ('image_block_size_per_thread_agx', uint8_t),
  ('ptr_size', ctypes.c_uint32),
  ('shader_index', uint32_t),
  ('node_payloads_size', uint32_t),
  ('workgroup_count', (uint32_t * 3)),
]
class struct_shader_info_0_tess(Struct): pass
enum_tess_primitive_mode = CEnum(ctypes.c_uint32)
TESS_PRIMITIVE_UNSPECIFIED = enum_tess_primitive_mode.define('TESS_PRIMITIVE_UNSPECIFIED', 0)
TESS_PRIMITIVE_TRIANGLES = enum_tess_primitive_mode.define('TESS_PRIMITIVE_TRIANGLES', 1)
TESS_PRIMITIVE_QUADS = enum_tess_primitive_mode.define('TESS_PRIMITIVE_QUADS', 2)
TESS_PRIMITIVE_ISOLINES = enum_tess_primitive_mode.define('TESS_PRIMITIVE_ISOLINES', 3)

struct_shader_info_0_tess._fields_ = [
  ('_primitive_mode', enum_tess_primitive_mode),
  ('tcs_vertices_out', uint8_t),
  ('spacing', ctypes.c_uint32,2),
  ('ccw', ctypes.c_bool,1),
  ('point_mode', ctypes.c_bool,1),
  ('tcs_same_invocation_inputs_read', uint64_t),
  ('tcs_cross_invocation_inputs_read', uint64_t),
  ('tcs_cross_invocation_outputs_read', uint64_t),
  ('tcs_cross_invocation_outputs_written', uint64_t),
  ('tcs_outputs_read_by_tes', uint64_t),
  ('tcs_patch_outputs_read_by_tes', uint32_t),
  ('tcs_outputs_read_by_tes_16bit', uint16_t),
]
class struct_shader_info_0_mesh(Struct): pass
struct_shader_info_0_mesh._fields_ = [
  ('ms_cross_invocation_output_access', uint64_t),
  ('ts_mesh_dispatch_dimensions', (uint32_t * 3)),
  ('max_vertices_out', uint16_t),
  ('max_primitives_out', uint16_t),
  ('primitive_type', enum_mesa_prim),
  ('nv', ctypes.c_bool),
]
struct_shader_info_0._fields_ = [
  ('vs', struct_shader_info_0_vs),
  ('gs', struct_shader_info_0_gs),
  ('fs', struct_shader_info_0_fs),
  ('cs', struct_shader_info_0_cs),
  ('tess', struct_shader_info_0_tess),
  ('mesh', struct_shader_info_0_mesh),
]
struct_shader_info._anonymous_ = ['_0']
struct_shader_info._fields_ = [
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('label', ctypes.POINTER(ctypes.c_char)),
  ('internal', ctypes.c_bool),
  ('source_blake3', blake3_hash),
  ('stage', gl_shader_stage,8),
  ('prev_stage', gl_shader_stage,8),
  ('next_stage', gl_shader_stage,8),
  ('prev_stage_has_xfb', ctypes.c_bool),
  ('num_textures', uint8_t),
  ('num_ubos', uint8_t),
  ('num_abos', uint8_t),
  ('num_ssbos', uint8_t),
  ('num_images', uint8_t),
  ('inputs_read', uint64_t),
  ('dual_slot_inputs', uint64_t),
  ('outputs_written', uint64_t),
  ('outputs_read', uint64_t),
  ('system_values_read', (ctypes.c_uint32 * 4)),
  ('per_primitive_inputs', uint64_t),
  ('per_primitive_outputs', uint64_t),
  ('per_view_outputs', uint64_t),
  ('view_mask', uint32_t),
  ('inputs_read_16bit', uint16_t),
  ('outputs_written_16bit', uint16_t),
  ('outputs_read_16bit', uint16_t),
  ('inputs_read_indirectly_16bit', uint16_t),
  ('outputs_read_indirectly_16bit', uint16_t),
  ('outputs_written_indirectly_16bit', uint16_t),
  ('patch_inputs_read', uint32_t),
  ('patch_outputs_written', uint32_t),
  ('patch_outputs_read', uint32_t),
  ('inputs_read_indirectly', uint64_t),
  ('outputs_read_indirectly', uint64_t),
  ('outputs_written_indirectly', uint64_t),
  ('patch_inputs_read_indirectly', uint32_t),
  ('patch_outputs_read_indirectly', uint32_t),
  ('patch_outputs_written_indirectly', uint32_t),
  ('textures_used', (ctypes.c_uint32 * 4)),
  ('textures_used_by_txf', (ctypes.c_uint32 * 4)),
  ('samplers_used', (ctypes.c_uint32 * 1)),
  ('images_used', (ctypes.c_uint32 * 2)),
  ('image_buffers', (ctypes.c_uint32 * 2)),
  ('msaa_images', (ctypes.c_uint32 * 2)),
  ('float_controls_execution_mode', uint32_t),
  ('shared_size', ctypes.c_uint32),
  ('task_payload_size', ctypes.c_uint32),
  ('ray_queries', ctypes.c_uint32),
  ('workgroup_size', (uint16_t * 3)),
  ('subgroup_size', enum_gl_subgroup_size),
  ('num_subgroups', uint8_t),
  ('uses_wide_subgroup_intrinsics', ctypes.c_bool),
  ('xfb_stride', (uint8_t * 4)),
  ('inlinable_uniform_dw_offsets', (uint16_t * 4)),
  ('num_inlinable_uniforms', uint8_t,4),
  ('clip_distance_array_size', uint8_t,4),
  ('cull_distance_array_size', uint8_t,4),
  ('uses_texture_gather', ctypes.c_bool,1),
  ('uses_resource_info_query', ctypes.c_bool,1),
  ('bit_sizes_float', uint8_t),
  ('bit_sizes_int', uint8_t),
  ('first_ubo_is_default_ubo', ctypes.c_bool,1),
  ('separate_shader', ctypes.c_bool,1),
  ('has_transform_feedback_varyings', ctypes.c_bool,1),
  ('flrp_lowered', ctypes.c_bool,1),
  ('io_lowered', ctypes.c_bool,1),
  ('var_copies_lowered', ctypes.c_bool,1),
  ('writes_memory', ctypes.c_bool,1),
  ('layer_viewport_relative', ctypes.c_bool,1),
  ('uses_control_barrier', ctypes.c_bool,1),
  ('uses_memory_barrier', ctypes.c_bool,1),
  ('uses_bindless', ctypes.c_bool,1),
  ('shared_memory_explicit_layout', ctypes.c_bool,1),
  ('zero_initialize_shared_memory', ctypes.c_bool,1),
  ('workgroup_size_variable', ctypes.c_bool,1),
  ('uses_printf', ctypes.c_bool,1),
  ('maximally_reconverges', ctypes.c_bool,1),
  ('use_aco_amd', ctypes.c_bool,1),
  ('use_lowered_image_to_global', ctypes.c_bool,1),
  ('use_legacy_math_rules', ctypes.c_bool),
  ('derivative_group', enum_gl_derivative_group,2),
  ('_0', struct_shader_info_0),
]
class struct_nir_xfb_info(Struct): pass
nir_xfb_info = struct_nir_xfb_info
struct_nir_shader._fields_ = [
  ('gctx', ctypes.POINTER(gc_ctx)),
  ('variables', struct_exec_list),
  ('options', ctypes.POINTER(nir_shader_compiler_options)),
  ('info', struct_shader_info),
  ('functions', struct_exec_list),
  ('num_inputs', ctypes.c_uint32),
  ('num_uniforms', ctypes.c_uint32),
  ('num_outputs', ctypes.c_uint32),
  ('global_mem_size', ctypes.c_uint32),
  ('scratch_size', ctypes.c_uint32),
  ('constant_data', ctypes.c_void_p),
  ('constant_data_size', ctypes.c_uint32),
  ('xfb_info', ctypes.POINTER(nir_xfb_info)),
  ('printf_info_count', ctypes.c_uint32),
  ('printf_info', ctypes.POINTER(u_printf_info)),
  ('has_debug_info', ctypes.c_bool),
]
class struct_nir_parameter(Struct): pass
nir_parameter = struct_nir_parameter
struct_nir_parameter._fields_ = [
  ('num_components', uint8_t),
  ('bit_size', uint8_t),
  ('is_return', ctypes.c_bool),
  ('implicit_conversion_prohibited', ctypes.c_bool),
  ('is_uniform', ctypes.c_bool),
  ('mode', nir_variable_mode),
  ('driver_attributes', uint32_t),
  ('type', ctypes.POINTER(struct_glsl_type)),
  ('name', ctypes.POINTER(ctypes.c_char)),
]
class struct_nir_function_impl(Struct): pass
nir_function_impl = struct_nir_function_impl
nir_function = struct_nir_function
nir_metadata = CEnum(ctypes.c_int32)
nir_metadata_none = nir_metadata.define('nir_metadata_none', 0)
nir_metadata_block_index = nir_metadata.define('nir_metadata_block_index', 1)
nir_metadata_dominance = nir_metadata.define('nir_metadata_dominance', 2)
nir_metadata_live_defs = nir_metadata.define('nir_metadata_live_defs', 4)
nir_metadata_not_properly_reset = nir_metadata.define('nir_metadata_not_properly_reset', 8)
nir_metadata_loop_analysis = nir_metadata.define('nir_metadata_loop_analysis', 16)
nir_metadata_instr_index = nir_metadata.define('nir_metadata_instr_index', 32)
nir_metadata_divergence = nir_metadata.define('nir_metadata_divergence', 64)
nir_metadata_control_flow = nir_metadata.define('nir_metadata_control_flow', 3)
nir_metadata_all = nir_metadata.define('nir_metadata_all', -9)

struct_nir_function_impl._fields_ = [
  ('cf_node', nir_cf_node),
  ('function', ctypes.POINTER(nir_function)),
  ('preamble', ctypes.POINTER(nir_function)),
  ('body', struct_exec_list),
  ('end_block', ctypes.POINTER(nir_block)),
  ('locals', struct_exec_list),
  ('ssa_alloc', ctypes.c_uint32),
  ('num_blocks', ctypes.c_uint32),
  ('structured', ctypes.c_bool),
  ('valid_metadata', nir_metadata),
  ('loop_analysis_indirect_mask', nir_variable_mode),
  ('loop_analysis_force_unroll_sampler_indirect', ctypes.c_bool),
]
struct_nir_function._fields_ = [
  ('node', struct_exec_node),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('shader', ctypes.POINTER(nir_shader)),
  ('num_params', ctypes.c_uint32),
  ('params', ctypes.POINTER(nir_parameter)),
  ('impl', ctypes.POINTER(nir_function_impl)),
  ('driver_attributes', uint32_t),
  ('is_entrypoint', ctypes.c_bool),
  ('is_exported', ctypes.c_bool),
  ('is_preamble', ctypes.c_bool),
  ('should_inline', ctypes.c_bool),
  ('dont_inline', ctypes.c_bool),
  ('workgroup_size', (ctypes.c_uint32 * 3)),
  ('is_subroutine', ctypes.c_bool),
  ('is_tmp_globals_wrapper', ctypes.c_bool),
  ('num_subroutine_types', ctypes.c_int32),
  ('subroutine_types', ctypes.POINTER(ctypes.POINTER(struct_glsl_type))),
  ('subroutine_index', ctypes.c_int32),
  ('pass_flags', uint32_t),
]
struct_nir_call_instr._fields_ = [
  ('instr', nir_instr),
  ('callee', ctypes.POINTER(nir_function)),
  ('indirect_callee', nir_src),
  ('num_params', ctypes.c_uint32),
  ('params', (nir_src * 0)),
]
nir_call_instr = struct_nir_call_instr
class struct_nir_intrinsic_instr(Struct): pass
nir_intrinsic_op = CEnum(ctypes.c_uint32)
nir_intrinsic_accept_ray_intersection = nir_intrinsic_op.define('nir_intrinsic_accept_ray_intersection', 0)
nir_intrinsic_addr_mode_is = nir_intrinsic_op.define('nir_intrinsic_addr_mode_is', 1)
nir_intrinsic_al2p_nv = nir_intrinsic_op.define('nir_intrinsic_al2p_nv', 2)
nir_intrinsic_ald_nv = nir_intrinsic_op.define('nir_intrinsic_ald_nv', 3)
nir_intrinsic_alpha_to_coverage = nir_intrinsic_op.define('nir_intrinsic_alpha_to_coverage', 4)
nir_intrinsic_as_uniform = nir_intrinsic_op.define('nir_intrinsic_as_uniform', 5)
nir_intrinsic_ast_nv = nir_intrinsic_op.define('nir_intrinsic_ast_nv', 6)
nir_intrinsic_atomic_add_gen_prim_count_amd = nir_intrinsic_op.define('nir_intrinsic_atomic_add_gen_prim_count_amd', 7)
nir_intrinsic_atomic_add_gs_emit_prim_count_amd = nir_intrinsic_op.define('nir_intrinsic_atomic_add_gs_emit_prim_count_amd', 8)
nir_intrinsic_atomic_add_shader_invocation_count_amd = nir_intrinsic_op.define('nir_intrinsic_atomic_add_shader_invocation_count_amd', 9)
nir_intrinsic_atomic_add_xfb_prim_count_amd = nir_intrinsic_op.define('nir_intrinsic_atomic_add_xfb_prim_count_amd', 10)
nir_intrinsic_atomic_counter_add = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_add', 11)
nir_intrinsic_atomic_counter_add_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_add_deref', 12)
nir_intrinsic_atomic_counter_and = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_and', 13)
nir_intrinsic_atomic_counter_and_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_and_deref', 14)
nir_intrinsic_atomic_counter_comp_swap = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_comp_swap', 15)
nir_intrinsic_atomic_counter_comp_swap_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_comp_swap_deref', 16)
nir_intrinsic_atomic_counter_exchange = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_exchange', 17)
nir_intrinsic_atomic_counter_exchange_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_exchange_deref', 18)
nir_intrinsic_atomic_counter_inc = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_inc', 19)
nir_intrinsic_atomic_counter_inc_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_inc_deref', 20)
nir_intrinsic_atomic_counter_max = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_max', 21)
nir_intrinsic_atomic_counter_max_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_max_deref', 22)
nir_intrinsic_atomic_counter_min = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_min', 23)
nir_intrinsic_atomic_counter_min_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_min_deref', 24)
nir_intrinsic_atomic_counter_or = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_or', 25)
nir_intrinsic_atomic_counter_or_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_or_deref', 26)
nir_intrinsic_atomic_counter_post_dec = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_post_dec', 27)
nir_intrinsic_atomic_counter_post_dec_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_post_dec_deref', 28)
nir_intrinsic_atomic_counter_pre_dec = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_pre_dec', 29)
nir_intrinsic_atomic_counter_pre_dec_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_pre_dec_deref', 30)
nir_intrinsic_atomic_counter_read = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_read', 31)
nir_intrinsic_atomic_counter_read_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_read_deref', 32)
nir_intrinsic_atomic_counter_xor = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_xor', 33)
nir_intrinsic_atomic_counter_xor_deref = nir_intrinsic_op.define('nir_intrinsic_atomic_counter_xor_deref', 34)
nir_intrinsic_ballot = nir_intrinsic_op.define('nir_intrinsic_ballot', 35)
nir_intrinsic_ballot_bit_count_exclusive = nir_intrinsic_op.define('nir_intrinsic_ballot_bit_count_exclusive', 36)
nir_intrinsic_ballot_bit_count_inclusive = nir_intrinsic_op.define('nir_intrinsic_ballot_bit_count_inclusive', 37)
nir_intrinsic_ballot_bit_count_reduce = nir_intrinsic_op.define('nir_intrinsic_ballot_bit_count_reduce', 38)
nir_intrinsic_ballot_bitfield_extract = nir_intrinsic_op.define('nir_intrinsic_ballot_bitfield_extract', 39)
nir_intrinsic_ballot_find_lsb = nir_intrinsic_op.define('nir_intrinsic_ballot_find_lsb', 40)
nir_intrinsic_ballot_find_msb = nir_intrinsic_op.define('nir_intrinsic_ballot_find_msb', 41)
nir_intrinsic_ballot_relaxed = nir_intrinsic_op.define('nir_intrinsic_ballot_relaxed', 42)
nir_intrinsic_bar_break_nv = nir_intrinsic_op.define('nir_intrinsic_bar_break_nv', 43)
nir_intrinsic_bar_set_nv = nir_intrinsic_op.define('nir_intrinsic_bar_set_nv', 44)
nir_intrinsic_bar_sync_nv = nir_intrinsic_op.define('nir_intrinsic_bar_sync_nv', 45)
nir_intrinsic_barrier = nir_intrinsic_op.define('nir_intrinsic_barrier', 46)
nir_intrinsic_begin_invocation_interlock = nir_intrinsic_op.define('nir_intrinsic_begin_invocation_interlock', 47)
nir_intrinsic_bindgen_return = nir_intrinsic_op.define('nir_intrinsic_bindgen_return', 48)
nir_intrinsic_bindless_image_agx = nir_intrinsic_op.define('nir_intrinsic_bindless_image_agx', 49)
nir_intrinsic_bindless_image_atomic = nir_intrinsic_op.define('nir_intrinsic_bindless_image_atomic', 50)
nir_intrinsic_bindless_image_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_bindless_image_atomic_swap', 51)
nir_intrinsic_bindless_image_descriptor_amd = nir_intrinsic_op.define('nir_intrinsic_bindless_image_descriptor_amd', 52)
nir_intrinsic_bindless_image_format = nir_intrinsic_op.define('nir_intrinsic_bindless_image_format', 53)
nir_intrinsic_bindless_image_fragment_mask_load_amd = nir_intrinsic_op.define('nir_intrinsic_bindless_image_fragment_mask_load_amd', 54)
nir_intrinsic_bindless_image_levels = nir_intrinsic_op.define('nir_intrinsic_bindless_image_levels', 55)
nir_intrinsic_bindless_image_load = nir_intrinsic_op.define('nir_intrinsic_bindless_image_load', 56)
nir_intrinsic_bindless_image_load_raw_intel = nir_intrinsic_op.define('nir_intrinsic_bindless_image_load_raw_intel', 57)
nir_intrinsic_bindless_image_order = nir_intrinsic_op.define('nir_intrinsic_bindless_image_order', 58)
nir_intrinsic_bindless_image_samples = nir_intrinsic_op.define('nir_intrinsic_bindless_image_samples', 59)
nir_intrinsic_bindless_image_samples_identical = nir_intrinsic_op.define('nir_intrinsic_bindless_image_samples_identical', 60)
nir_intrinsic_bindless_image_size = nir_intrinsic_op.define('nir_intrinsic_bindless_image_size', 61)
nir_intrinsic_bindless_image_sparse_load = nir_intrinsic_op.define('nir_intrinsic_bindless_image_sparse_load', 62)
nir_intrinsic_bindless_image_store = nir_intrinsic_op.define('nir_intrinsic_bindless_image_store', 63)
nir_intrinsic_bindless_image_store_block_agx = nir_intrinsic_op.define('nir_intrinsic_bindless_image_store_block_agx', 64)
nir_intrinsic_bindless_image_store_raw_intel = nir_intrinsic_op.define('nir_intrinsic_bindless_image_store_raw_intel', 65)
nir_intrinsic_bindless_image_texel_address = nir_intrinsic_op.define('nir_intrinsic_bindless_image_texel_address', 66)
nir_intrinsic_bindless_resource_ir3 = nir_intrinsic_op.define('nir_intrinsic_bindless_resource_ir3', 67)
nir_intrinsic_brcst_active_ir3 = nir_intrinsic_op.define('nir_intrinsic_brcst_active_ir3', 68)
nir_intrinsic_btd_retire_intel = nir_intrinsic_op.define('nir_intrinsic_btd_retire_intel', 69)
nir_intrinsic_btd_spawn_intel = nir_intrinsic_op.define('nir_intrinsic_btd_spawn_intel', 70)
nir_intrinsic_btd_stack_push_intel = nir_intrinsic_op.define('nir_intrinsic_btd_stack_push_intel', 71)
nir_intrinsic_bvh64_intersect_ray_amd = nir_intrinsic_op.define('nir_intrinsic_bvh64_intersect_ray_amd', 72)
nir_intrinsic_bvh8_intersect_ray_amd = nir_intrinsic_op.define('nir_intrinsic_bvh8_intersect_ray_amd', 73)
nir_intrinsic_bvh_stack_rtn_amd = nir_intrinsic_op.define('nir_intrinsic_bvh_stack_rtn_amd', 74)
nir_intrinsic_cmat_binary_op = nir_intrinsic_op.define('nir_intrinsic_cmat_binary_op', 75)
nir_intrinsic_cmat_bitcast = nir_intrinsic_op.define('nir_intrinsic_cmat_bitcast', 76)
nir_intrinsic_cmat_construct = nir_intrinsic_op.define('nir_intrinsic_cmat_construct', 77)
nir_intrinsic_cmat_convert = nir_intrinsic_op.define('nir_intrinsic_cmat_convert', 78)
nir_intrinsic_cmat_copy = nir_intrinsic_op.define('nir_intrinsic_cmat_copy', 79)
nir_intrinsic_cmat_extract = nir_intrinsic_op.define('nir_intrinsic_cmat_extract', 80)
nir_intrinsic_cmat_insert = nir_intrinsic_op.define('nir_intrinsic_cmat_insert', 81)
nir_intrinsic_cmat_length = nir_intrinsic_op.define('nir_intrinsic_cmat_length', 82)
nir_intrinsic_cmat_load = nir_intrinsic_op.define('nir_intrinsic_cmat_load', 83)
nir_intrinsic_cmat_muladd = nir_intrinsic_op.define('nir_intrinsic_cmat_muladd', 84)
nir_intrinsic_cmat_muladd_amd = nir_intrinsic_op.define('nir_intrinsic_cmat_muladd_amd', 85)
nir_intrinsic_cmat_muladd_nv = nir_intrinsic_op.define('nir_intrinsic_cmat_muladd_nv', 86)
nir_intrinsic_cmat_scalar_op = nir_intrinsic_op.define('nir_intrinsic_cmat_scalar_op', 87)
nir_intrinsic_cmat_store = nir_intrinsic_op.define('nir_intrinsic_cmat_store', 88)
nir_intrinsic_cmat_transpose = nir_intrinsic_op.define('nir_intrinsic_cmat_transpose', 89)
nir_intrinsic_cmat_unary_op = nir_intrinsic_op.define('nir_intrinsic_cmat_unary_op', 90)
nir_intrinsic_convert_alu_types = nir_intrinsic_op.define('nir_intrinsic_convert_alu_types', 91)
nir_intrinsic_convert_cmat_intel = nir_intrinsic_op.define('nir_intrinsic_convert_cmat_intel', 92)
nir_intrinsic_copy_deref = nir_intrinsic_op.define('nir_intrinsic_copy_deref', 93)
nir_intrinsic_copy_fs_outputs_nv = nir_intrinsic_op.define('nir_intrinsic_copy_fs_outputs_nv', 94)
nir_intrinsic_copy_global_to_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_copy_global_to_uniform_ir3', 95)
nir_intrinsic_copy_push_const_to_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_copy_push_const_to_uniform_ir3', 96)
nir_intrinsic_copy_ubo_to_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_copy_ubo_to_uniform_ir3', 97)
nir_intrinsic_ddx = nir_intrinsic_op.define('nir_intrinsic_ddx', 98)
nir_intrinsic_ddx_coarse = nir_intrinsic_op.define('nir_intrinsic_ddx_coarse', 99)
nir_intrinsic_ddx_fine = nir_intrinsic_op.define('nir_intrinsic_ddx_fine', 100)
nir_intrinsic_ddy = nir_intrinsic_op.define('nir_intrinsic_ddy', 101)
nir_intrinsic_ddy_coarse = nir_intrinsic_op.define('nir_intrinsic_ddy_coarse', 102)
nir_intrinsic_ddy_fine = nir_intrinsic_op.define('nir_intrinsic_ddy_fine', 103)
nir_intrinsic_debug_break = nir_intrinsic_op.define('nir_intrinsic_debug_break', 104)
nir_intrinsic_decl_reg = nir_intrinsic_op.define('nir_intrinsic_decl_reg', 105)
nir_intrinsic_demote = nir_intrinsic_op.define('nir_intrinsic_demote', 106)
nir_intrinsic_demote_if = nir_intrinsic_op.define('nir_intrinsic_demote_if', 107)
nir_intrinsic_demote_samples = nir_intrinsic_op.define('nir_intrinsic_demote_samples', 108)
nir_intrinsic_deref_atomic = nir_intrinsic_op.define('nir_intrinsic_deref_atomic', 109)
nir_intrinsic_deref_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_deref_atomic_swap', 110)
nir_intrinsic_deref_buffer_array_length = nir_intrinsic_op.define('nir_intrinsic_deref_buffer_array_length', 111)
nir_intrinsic_deref_implicit_array_length = nir_intrinsic_op.define('nir_intrinsic_deref_implicit_array_length', 112)
nir_intrinsic_deref_mode_is = nir_intrinsic_op.define('nir_intrinsic_deref_mode_is', 113)
nir_intrinsic_deref_texture_src = nir_intrinsic_op.define('nir_intrinsic_deref_texture_src', 114)
nir_intrinsic_doorbell_agx = nir_intrinsic_op.define('nir_intrinsic_doorbell_agx', 115)
nir_intrinsic_dpas_intel = nir_intrinsic_op.define('nir_intrinsic_dpas_intel', 116)
nir_intrinsic_dpp16_shift_amd = nir_intrinsic_op.define('nir_intrinsic_dpp16_shift_amd', 117)
nir_intrinsic_elect = nir_intrinsic_op.define('nir_intrinsic_elect', 118)
nir_intrinsic_elect_any_ir3 = nir_intrinsic_op.define('nir_intrinsic_elect_any_ir3', 119)
nir_intrinsic_emit_primitive_poly = nir_intrinsic_op.define('nir_intrinsic_emit_primitive_poly', 120)
nir_intrinsic_emit_vertex = nir_intrinsic_op.define('nir_intrinsic_emit_vertex', 121)
nir_intrinsic_emit_vertex_nv = nir_intrinsic_op.define('nir_intrinsic_emit_vertex_nv', 122)
nir_intrinsic_emit_vertex_with_counter = nir_intrinsic_op.define('nir_intrinsic_emit_vertex_with_counter', 123)
nir_intrinsic_end_invocation_interlock = nir_intrinsic_op.define('nir_intrinsic_end_invocation_interlock', 124)
nir_intrinsic_end_primitive = nir_intrinsic_op.define('nir_intrinsic_end_primitive', 125)
nir_intrinsic_end_primitive_nv = nir_intrinsic_op.define('nir_intrinsic_end_primitive_nv', 126)
nir_intrinsic_end_primitive_with_counter = nir_intrinsic_op.define('nir_intrinsic_end_primitive_with_counter', 127)
nir_intrinsic_enqueue_node_payloads = nir_intrinsic_op.define('nir_intrinsic_enqueue_node_payloads', 128)
nir_intrinsic_exclusive_scan = nir_intrinsic_op.define('nir_intrinsic_exclusive_scan', 129)
nir_intrinsic_exclusive_scan_clusters_ir3 = nir_intrinsic_op.define('nir_intrinsic_exclusive_scan_clusters_ir3', 130)
nir_intrinsic_execute_callable = nir_intrinsic_op.define('nir_intrinsic_execute_callable', 131)
nir_intrinsic_execute_closest_hit_amd = nir_intrinsic_op.define('nir_intrinsic_execute_closest_hit_amd', 132)
nir_intrinsic_execute_miss_amd = nir_intrinsic_op.define('nir_intrinsic_execute_miss_amd', 133)
nir_intrinsic_export_agx = nir_intrinsic_op.define('nir_intrinsic_export_agx', 134)
nir_intrinsic_export_amd = nir_intrinsic_op.define('nir_intrinsic_export_amd', 135)
nir_intrinsic_export_dual_src_blend_amd = nir_intrinsic_op.define('nir_intrinsic_export_dual_src_blend_amd', 136)
nir_intrinsic_export_row_amd = nir_intrinsic_op.define('nir_intrinsic_export_row_amd', 137)
nir_intrinsic_fence_helper_exit_agx = nir_intrinsic_op.define('nir_intrinsic_fence_helper_exit_agx', 138)
nir_intrinsic_fence_mem_to_tex_agx = nir_intrinsic_op.define('nir_intrinsic_fence_mem_to_tex_agx', 139)
nir_intrinsic_fence_pbe_to_tex_agx = nir_intrinsic_op.define('nir_intrinsic_fence_pbe_to_tex_agx', 140)
nir_intrinsic_fence_pbe_to_tex_pixel_agx = nir_intrinsic_op.define('nir_intrinsic_fence_pbe_to_tex_pixel_agx', 141)
nir_intrinsic_final_primitive_nv = nir_intrinsic_op.define('nir_intrinsic_final_primitive_nv', 142)
nir_intrinsic_finalize_incoming_node_payload = nir_intrinsic_op.define('nir_intrinsic_finalize_incoming_node_payload', 143)
nir_intrinsic_first_invocation = nir_intrinsic_op.define('nir_intrinsic_first_invocation', 144)
nir_intrinsic_fs_out_nv = nir_intrinsic_op.define('nir_intrinsic_fs_out_nv', 145)
nir_intrinsic_gds_atomic_add_amd = nir_intrinsic_op.define('nir_intrinsic_gds_atomic_add_amd', 146)
nir_intrinsic_get_ssbo_size = nir_intrinsic_op.define('nir_intrinsic_get_ssbo_size', 147)
nir_intrinsic_get_ubo_size = nir_intrinsic_op.define('nir_intrinsic_get_ubo_size', 148)
nir_intrinsic_global_atomic = nir_intrinsic_op.define('nir_intrinsic_global_atomic', 149)
nir_intrinsic_global_atomic_2x32 = nir_intrinsic_op.define('nir_intrinsic_global_atomic_2x32', 150)
nir_intrinsic_global_atomic_agx = nir_intrinsic_op.define('nir_intrinsic_global_atomic_agx', 151)
nir_intrinsic_global_atomic_amd = nir_intrinsic_op.define('nir_intrinsic_global_atomic_amd', 152)
nir_intrinsic_global_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_global_atomic_swap', 153)
nir_intrinsic_global_atomic_swap_2x32 = nir_intrinsic_op.define('nir_intrinsic_global_atomic_swap_2x32', 154)
nir_intrinsic_global_atomic_swap_agx = nir_intrinsic_op.define('nir_intrinsic_global_atomic_swap_agx', 155)
nir_intrinsic_global_atomic_swap_amd = nir_intrinsic_op.define('nir_intrinsic_global_atomic_swap_amd', 156)
nir_intrinsic_ignore_ray_intersection = nir_intrinsic_op.define('nir_intrinsic_ignore_ray_intersection', 157)
nir_intrinsic_imadsp_nv = nir_intrinsic_op.define('nir_intrinsic_imadsp_nv', 158)
nir_intrinsic_image_atomic = nir_intrinsic_op.define('nir_intrinsic_image_atomic', 159)
nir_intrinsic_image_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_image_atomic_swap', 160)
nir_intrinsic_image_deref_atomic = nir_intrinsic_op.define('nir_intrinsic_image_deref_atomic', 161)
nir_intrinsic_image_deref_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_image_deref_atomic_swap', 162)
nir_intrinsic_image_deref_descriptor_amd = nir_intrinsic_op.define('nir_intrinsic_image_deref_descriptor_amd', 163)
nir_intrinsic_image_deref_format = nir_intrinsic_op.define('nir_intrinsic_image_deref_format', 164)
nir_intrinsic_image_deref_fragment_mask_load_amd = nir_intrinsic_op.define('nir_intrinsic_image_deref_fragment_mask_load_amd', 165)
nir_intrinsic_image_deref_levels = nir_intrinsic_op.define('nir_intrinsic_image_deref_levels', 166)
nir_intrinsic_image_deref_load = nir_intrinsic_op.define('nir_intrinsic_image_deref_load', 167)
nir_intrinsic_image_deref_load_info_nv = nir_intrinsic_op.define('nir_intrinsic_image_deref_load_info_nv', 168)
nir_intrinsic_image_deref_load_param_intel = nir_intrinsic_op.define('nir_intrinsic_image_deref_load_param_intel', 169)
nir_intrinsic_image_deref_load_raw_intel = nir_intrinsic_op.define('nir_intrinsic_image_deref_load_raw_intel', 170)
nir_intrinsic_image_deref_order = nir_intrinsic_op.define('nir_intrinsic_image_deref_order', 171)
nir_intrinsic_image_deref_samples = nir_intrinsic_op.define('nir_intrinsic_image_deref_samples', 172)
nir_intrinsic_image_deref_samples_identical = nir_intrinsic_op.define('nir_intrinsic_image_deref_samples_identical', 173)
nir_intrinsic_image_deref_size = nir_intrinsic_op.define('nir_intrinsic_image_deref_size', 174)
nir_intrinsic_image_deref_sparse_load = nir_intrinsic_op.define('nir_intrinsic_image_deref_sparse_load', 175)
nir_intrinsic_image_deref_store = nir_intrinsic_op.define('nir_intrinsic_image_deref_store', 176)
nir_intrinsic_image_deref_store_block_agx = nir_intrinsic_op.define('nir_intrinsic_image_deref_store_block_agx', 177)
nir_intrinsic_image_deref_store_raw_intel = nir_intrinsic_op.define('nir_intrinsic_image_deref_store_raw_intel', 178)
nir_intrinsic_image_deref_texel_address = nir_intrinsic_op.define('nir_intrinsic_image_deref_texel_address', 179)
nir_intrinsic_image_descriptor_amd = nir_intrinsic_op.define('nir_intrinsic_image_descriptor_amd', 180)
nir_intrinsic_image_format = nir_intrinsic_op.define('nir_intrinsic_image_format', 181)
nir_intrinsic_image_fragment_mask_load_amd = nir_intrinsic_op.define('nir_intrinsic_image_fragment_mask_load_amd', 182)
nir_intrinsic_image_levels = nir_intrinsic_op.define('nir_intrinsic_image_levels', 183)
nir_intrinsic_image_load = nir_intrinsic_op.define('nir_intrinsic_image_load', 184)
nir_intrinsic_image_load_raw_intel = nir_intrinsic_op.define('nir_intrinsic_image_load_raw_intel', 185)
nir_intrinsic_image_order = nir_intrinsic_op.define('nir_intrinsic_image_order', 186)
nir_intrinsic_image_samples = nir_intrinsic_op.define('nir_intrinsic_image_samples', 187)
nir_intrinsic_image_samples_identical = nir_intrinsic_op.define('nir_intrinsic_image_samples_identical', 188)
nir_intrinsic_image_size = nir_intrinsic_op.define('nir_intrinsic_image_size', 189)
nir_intrinsic_image_sparse_load = nir_intrinsic_op.define('nir_intrinsic_image_sparse_load', 190)
nir_intrinsic_image_store = nir_intrinsic_op.define('nir_intrinsic_image_store', 191)
nir_intrinsic_image_store_block_agx = nir_intrinsic_op.define('nir_intrinsic_image_store_block_agx', 192)
nir_intrinsic_image_store_raw_intel = nir_intrinsic_op.define('nir_intrinsic_image_store_raw_intel', 193)
nir_intrinsic_image_texel_address = nir_intrinsic_op.define('nir_intrinsic_image_texel_address', 194)
nir_intrinsic_inclusive_scan = nir_intrinsic_op.define('nir_intrinsic_inclusive_scan', 195)
nir_intrinsic_inclusive_scan_clusters_ir3 = nir_intrinsic_op.define('nir_intrinsic_inclusive_scan_clusters_ir3', 196)
nir_intrinsic_initialize_node_payloads = nir_intrinsic_op.define('nir_intrinsic_initialize_node_payloads', 197)
nir_intrinsic_interp_deref_at_centroid = nir_intrinsic_op.define('nir_intrinsic_interp_deref_at_centroid', 198)
nir_intrinsic_interp_deref_at_offset = nir_intrinsic_op.define('nir_intrinsic_interp_deref_at_offset', 199)
nir_intrinsic_interp_deref_at_sample = nir_intrinsic_op.define('nir_intrinsic_interp_deref_at_sample', 200)
nir_intrinsic_interp_deref_at_vertex = nir_intrinsic_op.define('nir_intrinsic_interp_deref_at_vertex', 201)
nir_intrinsic_inverse_ballot = nir_intrinsic_op.define('nir_intrinsic_inverse_ballot', 202)
nir_intrinsic_ipa_nv = nir_intrinsic_op.define('nir_intrinsic_ipa_nv', 203)
nir_intrinsic_is_helper_invocation = nir_intrinsic_op.define('nir_intrinsic_is_helper_invocation', 204)
nir_intrinsic_is_sparse_resident_zink = nir_intrinsic_op.define('nir_intrinsic_is_sparse_resident_zink', 205)
nir_intrinsic_is_sparse_texels_resident = nir_intrinsic_op.define('nir_intrinsic_is_sparse_texels_resident', 206)
nir_intrinsic_is_subgroup_invocation_lt_amd = nir_intrinsic_op.define('nir_intrinsic_is_subgroup_invocation_lt_amd', 207)
nir_intrinsic_isberd_nv = nir_intrinsic_op.define('nir_intrinsic_isberd_nv', 208)
nir_intrinsic_lane_permute_16_amd = nir_intrinsic_op.define('nir_intrinsic_lane_permute_16_amd', 209)
nir_intrinsic_last_invocation = nir_intrinsic_op.define('nir_intrinsic_last_invocation', 210)
nir_intrinsic_launch_mesh_workgroups = nir_intrinsic_op.define('nir_intrinsic_launch_mesh_workgroups', 211)
nir_intrinsic_launch_mesh_workgroups_with_payload_deref = nir_intrinsic_op.define('nir_intrinsic_launch_mesh_workgroups_with_payload_deref', 212)
nir_intrinsic_ldc_nv = nir_intrinsic_op.define('nir_intrinsic_ldc_nv', 213)
nir_intrinsic_ldcx_nv = nir_intrinsic_op.define('nir_intrinsic_ldcx_nv', 214)
nir_intrinsic_ldtram_nv = nir_intrinsic_op.define('nir_intrinsic_ldtram_nv', 215)
nir_intrinsic_load_aa_line_width = nir_intrinsic_op.define('nir_intrinsic_load_aa_line_width', 216)
nir_intrinsic_load_accel_struct_amd = nir_intrinsic_op.define('nir_intrinsic_load_accel_struct_amd', 217)
nir_intrinsic_load_active_samples_agx = nir_intrinsic_op.define('nir_intrinsic_load_active_samples_agx', 218)
nir_intrinsic_load_active_subgroup_count_agx = nir_intrinsic_op.define('nir_intrinsic_load_active_subgroup_count_agx', 219)
nir_intrinsic_load_active_subgroup_invocation_agx = nir_intrinsic_op.define('nir_intrinsic_load_active_subgroup_invocation_agx', 220)
nir_intrinsic_load_agx = nir_intrinsic_op.define('nir_intrinsic_load_agx', 221)
nir_intrinsic_load_alpha_reference_amd = nir_intrinsic_op.define('nir_intrinsic_load_alpha_reference_amd', 222)
nir_intrinsic_load_api_sample_mask_agx = nir_intrinsic_op.define('nir_intrinsic_load_api_sample_mask_agx', 223)
nir_intrinsic_load_attrib_clamp_agx = nir_intrinsic_op.define('nir_intrinsic_load_attrib_clamp_agx', 224)
nir_intrinsic_load_attribute_pan = nir_intrinsic_op.define('nir_intrinsic_load_attribute_pan', 225)
nir_intrinsic_load_back_face_agx = nir_intrinsic_op.define('nir_intrinsic_load_back_face_agx', 226)
nir_intrinsic_load_barycentric_at_offset = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_at_offset', 227)
nir_intrinsic_load_barycentric_at_offset_nv = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_at_offset_nv', 228)
nir_intrinsic_load_barycentric_at_sample = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_at_sample', 229)
nir_intrinsic_load_barycentric_centroid = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_centroid', 230)
nir_intrinsic_load_barycentric_coord_at_offset = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_at_offset', 231)
nir_intrinsic_load_barycentric_coord_at_sample = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_at_sample', 232)
nir_intrinsic_load_barycentric_coord_centroid = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_centroid', 233)
nir_intrinsic_load_barycentric_coord_pixel = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_pixel', 234)
nir_intrinsic_load_barycentric_coord_sample = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_coord_sample', 235)
nir_intrinsic_load_barycentric_model = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_model', 236)
nir_intrinsic_load_barycentric_optimize_amd = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_optimize_amd', 237)
nir_intrinsic_load_barycentric_pixel = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_pixel', 238)
nir_intrinsic_load_barycentric_sample = nir_intrinsic_op.define('nir_intrinsic_load_barycentric_sample', 239)
nir_intrinsic_load_base_global_invocation_id = nir_intrinsic_op.define('nir_intrinsic_load_base_global_invocation_id', 240)
nir_intrinsic_load_base_instance = nir_intrinsic_op.define('nir_intrinsic_load_base_instance', 241)
nir_intrinsic_load_base_vertex = nir_intrinsic_op.define('nir_intrinsic_load_base_vertex', 242)
nir_intrinsic_load_base_workgroup_id = nir_intrinsic_op.define('nir_intrinsic_load_base_workgroup_id', 243)
nir_intrinsic_load_blend_const_color_a_float = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_a_float', 244)
nir_intrinsic_load_blend_const_color_aaaa8888_unorm = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_aaaa8888_unorm', 245)
nir_intrinsic_load_blend_const_color_b_float = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_b_float', 246)
nir_intrinsic_load_blend_const_color_g_float = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_g_float', 247)
nir_intrinsic_load_blend_const_color_r_float = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_r_float', 248)
nir_intrinsic_load_blend_const_color_rgba = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_rgba', 249)
nir_intrinsic_load_blend_const_color_rgba8888_unorm = nir_intrinsic_op.define('nir_intrinsic_load_blend_const_color_rgba8888_unorm', 250)
nir_intrinsic_load_btd_global_arg_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_global_arg_addr_intel', 251)
nir_intrinsic_load_btd_local_arg_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_local_arg_addr_intel', 252)
nir_intrinsic_load_btd_resume_sbt_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_resume_sbt_addr_intel', 253)
nir_intrinsic_load_btd_shader_type_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_shader_type_intel', 254)
nir_intrinsic_load_btd_stack_id_intel = nir_intrinsic_op.define('nir_intrinsic_load_btd_stack_id_intel', 255)
nir_intrinsic_load_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_load_buffer_amd', 256)
nir_intrinsic_load_callable_sbt_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_callable_sbt_addr_intel', 257)
nir_intrinsic_load_callable_sbt_stride_intel = nir_intrinsic_op.define('nir_intrinsic_load_callable_sbt_stride_intel', 258)
nir_intrinsic_load_clamp_vertex_color_amd = nir_intrinsic_op.define('nir_intrinsic_load_clamp_vertex_color_amd', 259)
nir_intrinsic_load_clip_half_line_width_amd = nir_intrinsic_op.define('nir_intrinsic_load_clip_half_line_width_amd', 260)
nir_intrinsic_load_clip_z_coeff_agx = nir_intrinsic_op.define('nir_intrinsic_load_clip_z_coeff_agx', 261)
nir_intrinsic_load_coalesced_input_count = nir_intrinsic_op.define('nir_intrinsic_load_coalesced_input_count', 262)
nir_intrinsic_load_coefficients_agx = nir_intrinsic_op.define('nir_intrinsic_load_coefficients_agx', 263)
nir_intrinsic_load_color0 = nir_intrinsic_op.define('nir_intrinsic_load_color0', 264)
nir_intrinsic_load_color1 = nir_intrinsic_op.define('nir_intrinsic_load_color1', 265)
nir_intrinsic_load_const_buf_base_addr_lvp = nir_intrinsic_op.define('nir_intrinsic_load_const_buf_base_addr_lvp', 266)
nir_intrinsic_load_const_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_const_ir3', 267)
nir_intrinsic_load_constant = nir_intrinsic_op.define('nir_intrinsic_load_constant', 268)
nir_intrinsic_load_constant_agx = nir_intrinsic_op.define('nir_intrinsic_load_constant_agx', 269)
nir_intrinsic_load_constant_base_ptr = nir_intrinsic_op.define('nir_intrinsic_load_constant_base_ptr', 270)
nir_intrinsic_load_converted_output_pan = nir_intrinsic_op.define('nir_intrinsic_load_converted_output_pan', 271)
nir_intrinsic_load_core_id_agx = nir_intrinsic_op.define('nir_intrinsic_load_core_id_agx', 272)
nir_intrinsic_load_cull_any_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_any_enabled_amd', 273)
nir_intrinsic_load_cull_back_face_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_back_face_enabled_amd', 274)
nir_intrinsic_load_cull_ccw_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_ccw_amd', 275)
nir_intrinsic_load_cull_front_face_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_front_face_enabled_amd', 276)
nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd', 277)
nir_intrinsic_load_cull_mask = nir_intrinsic_op.define('nir_intrinsic_load_cull_mask', 278)
nir_intrinsic_load_cull_mask_and_flags_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_mask_and_flags_amd', 279)
nir_intrinsic_load_cull_small_line_precision_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_small_line_precision_amd', 280)
nir_intrinsic_load_cull_small_lines_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_small_lines_enabled_amd', 281)
nir_intrinsic_load_cull_small_triangle_precision_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_small_triangle_precision_amd', 282)
nir_intrinsic_load_cull_small_triangles_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_small_triangles_enabled_amd', 283)
nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd', 284)
nir_intrinsic_load_debug_log_desc_amd = nir_intrinsic_op.define('nir_intrinsic_load_debug_log_desc_amd', 285)
nir_intrinsic_load_depth_never_agx = nir_intrinsic_op.define('nir_intrinsic_load_depth_never_agx', 286)
nir_intrinsic_load_deref = nir_intrinsic_op.define('nir_intrinsic_load_deref', 287)
nir_intrinsic_load_deref_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_deref_block_intel', 288)
nir_intrinsic_load_draw_id = nir_intrinsic_op.define('nir_intrinsic_load_draw_id', 289)
nir_intrinsic_load_esgs_vertex_stride_amd = nir_intrinsic_op.define('nir_intrinsic_load_esgs_vertex_stride_amd', 290)
nir_intrinsic_load_exported_agx = nir_intrinsic_op.define('nir_intrinsic_load_exported_agx', 291)
nir_intrinsic_load_fb_layers_v3d = nir_intrinsic_op.define('nir_intrinsic_load_fb_layers_v3d', 292)
nir_intrinsic_load_fbfetch_image_desc_amd = nir_intrinsic_op.define('nir_intrinsic_load_fbfetch_image_desc_amd', 293)
nir_intrinsic_load_fbfetch_image_fmask_desc_amd = nir_intrinsic_op.define('nir_intrinsic_load_fbfetch_image_fmask_desc_amd', 294)
nir_intrinsic_load_fep_w_v3d = nir_intrinsic_op.define('nir_intrinsic_load_fep_w_v3d', 295)
nir_intrinsic_load_first_vertex = nir_intrinsic_op.define('nir_intrinsic_load_first_vertex', 296)
nir_intrinsic_load_fixed_point_size_agx = nir_intrinsic_op.define('nir_intrinsic_load_fixed_point_size_agx', 297)
nir_intrinsic_load_flat_mask = nir_intrinsic_op.define('nir_intrinsic_load_flat_mask', 298)
nir_intrinsic_load_force_vrs_rates_amd = nir_intrinsic_op.define('nir_intrinsic_load_force_vrs_rates_amd', 299)
nir_intrinsic_load_frag_coord = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord', 300)
nir_intrinsic_load_frag_coord_unscaled_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord_unscaled_ir3', 301)
nir_intrinsic_load_frag_coord_w = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord_w', 302)
nir_intrinsic_load_frag_coord_z = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord_z', 303)
nir_intrinsic_load_frag_coord_zw_pan = nir_intrinsic_op.define('nir_intrinsic_load_frag_coord_zw_pan', 304)
nir_intrinsic_load_frag_invocation_count = nir_intrinsic_op.define('nir_intrinsic_load_frag_invocation_count', 305)
nir_intrinsic_load_frag_offset_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_frag_offset_ir3', 306)
nir_intrinsic_load_frag_shading_rate = nir_intrinsic_op.define('nir_intrinsic_load_frag_shading_rate', 307)
nir_intrinsic_load_frag_size = nir_intrinsic_op.define('nir_intrinsic_load_frag_size', 308)
nir_intrinsic_load_frag_size_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_frag_size_ir3', 309)
nir_intrinsic_load_from_texture_handle_agx = nir_intrinsic_op.define('nir_intrinsic_load_from_texture_handle_agx', 310)
nir_intrinsic_load_front_face = nir_intrinsic_op.define('nir_intrinsic_load_front_face', 311)
nir_intrinsic_load_front_face_fsign = nir_intrinsic_op.define('nir_intrinsic_load_front_face_fsign', 312)
nir_intrinsic_load_fs_input_interp_deltas = nir_intrinsic_op.define('nir_intrinsic_load_fs_input_interp_deltas', 313)
nir_intrinsic_load_fs_msaa_intel = nir_intrinsic_op.define('nir_intrinsic_load_fs_msaa_intel', 314)
nir_intrinsic_load_fully_covered = nir_intrinsic_op.define('nir_intrinsic_load_fully_covered', 315)
nir_intrinsic_load_geometry_param_buffer_poly = nir_intrinsic_op.define('nir_intrinsic_load_geometry_param_buffer_poly', 316)
nir_intrinsic_load_global = nir_intrinsic_op.define('nir_intrinsic_load_global', 317)
nir_intrinsic_load_global_2x32 = nir_intrinsic_op.define('nir_intrinsic_load_global_2x32', 318)
nir_intrinsic_load_global_amd = nir_intrinsic_op.define('nir_intrinsic_load_global_amd', 319)
nir_intrinsic_load_global_base_ptr = nir_intrinsic_op.define('nir_intrinsic_load_global_base_ptr', 320)
nir_intrinsic_load_global_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_global_block_intel', 321)
nir_intrinsic_load_global_bounded = nir_intrinsic_op.define('nir_intrinsic_load_global_bounded', 322)
nir_intrinsic_load_global_constant = nir_intrinsic_op.define('nir_intrinsic_load_global_constant', 323)
nir_intrinsic_load_global_constant_bounded = nir_intrinsic_op.define('nir_intrinsic_load_global_constant_bounded', 324)
nir_intrinsic_load_global_constant_offset = nir_intrinsic_op.define('nir_intrinsic_load_global_constant_offset', 325)
nir_intrinsic_load_global_constant_uniform_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_global_constant_uniform_block_intel', 326)
nir_intrinsic_load_global_etna = nir_intrinsic_op.define('nir_intrinsic_load_global_etna', 327)
nir_intrinsic_load_global_invocation_id = nir_intrinsic_op.define('nir_intrinsic_load_global_invocation_id', 328)
nir_intrinsic_load_global_invocation_index = nir_intrinsic_op.define('nir_intrinsic_load_global_invocation_index', 329)
nir_intrinsic_load_global_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_global_ir3', 330)
nir_intrinsic_load_global_size = nir_intrinsic_op.define('nir_intrinsic_load_global_size', 331)
nir_intrinsic_load_gs_header_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_gs_header_ir3', 332)
nir_intrinsic_load_gs_vertex_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_gs_vertex_offset_amd', 333)
nir_intrinsic_load_gs_wave_id_amd = nir_intrinsic_op.define('nir_intrinsic_load_gs_wave_id_amd', 334)
nir_intrinsic_load_helper_arg_hi_agx = nir_intrinsic_op.define('nir_intrinsic_load_helper_arg_hi_agx', 335)
nir_intrinsic_load_helper_arg_lo_agx = nir_intrinsic_op.define('nir_intrinsic_load_helper_arg_lo_agx', 336)
nir_intrinsic_load_helper_invocation = nir_intrinsic_op.define('nir_intrinsic_load_helper_invocation', 337)
nir_intrinsic_load_helper_op_id_agx = nir_intrinsic_op.define('nir_intrinsic_load_helper_op_id_agx', 338)
nir_intrinsic_load_hit_attrib_amd = nir_intrinsic_op.define('nir_intrinsic_load_hit_attrib_amd', 339)
nir_intrinsic_load_hs_out_patch_data_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_hs_out_patch_data_offset_amd', 340)
nir_intrinsic_load_hs_patch_stride_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_hs_patch_stride_ir3', 341)
nir_intrinsic_load_initial_edgeflags_amd = nir_intrinsic_op.define('nir_intrinsic_load_initial_edgeflags_amd', 342)
nir_intrinsic_load_inline_data_intel = nir_intrinsic_op.define('nir_intrinsic_load_inline_data_intel', 343)
nir_intrinsic_load_input = nir_intrinsic_op.define('nir_intrinsic_load_input', 344)
nir_intrinsic_load_input_assembly_buffer_poly = nir_intrinsic_op.define('nir_intrinsic_load_input_assembly_buffer_poly', 345)
nir_intrinsic_load_input_attachment_conv_pan = nir_intrinsic_op.define('nir_intrinsic_load_input_attachment_conv_pan', 346)
nir_intrinsic_load_input_attachment_coord = nir_intrinsic_op.define('nir_intrinsic_load_input_attachment_coord', 347)
nir_intrinsic_load_input_attachment_target_pan = nir_intrinsic_op.define('nir_intrinsic_load_input_attachment_target_pan', 348)
nir_intrinsic_load_input_topology_poly = nir_intrinsic_op.define('nir_intrinsic_load_input_topology_poly', 349)
nir_intrinsic_load_input_vertex = nir_intrinsic_op.define('nir_intrinsic_load_input_vertex', 350)
nir_intrinsic_load_instance_id = nir_intrinsic_op.define('nir_intrinsic_load_instance_id', 351)
nir_intrinsic_load_interpolated_input = nir_intrinsic_op.define('nir_intrinsic_load_interpolated_input', 352)
nir_intrinsic_load_intersection_opaque_amd = nir_intrinsic_op.define('nir_intrinsic_load_intersection_opaque_amd', 353)
nir_intrinsic_load_invocation_id = nir_intrinsic_op.define('nir_intrinsic_load_invocation_id', 354)
nir_intrinsic_load_is_first_fan_agx = nir_intrinsic_op.define('nir_intrinsic_load_is_first_fan_agx', 355)
nir_intrinsic_load_is_indexed_draw = nir_intrinsic_op.define('nir_intrinsic_load_is_indexed_draw', 356)
nir_intrinsic_load_kernel_input = nir_intrinsic_op.define('nir_intrinsic_load_kernel_input', 357)
nir_intrinsic_load_layer_id = nir_intrinsic_op.define('nir_intrinsic_load_layer_id', 358)
nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd = nir_intrinsic_op.define('nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd', 359)
nir_intrinsic_load_leaf_opaque_intel = nir_intrinsic_op.define('nir_intrinsic_load_leaf_opaque_intel', 360)
nir_intrinsic_load_leaf_procedural_intel = nir_intrinsic_op.define('nir_intrinsic_load_leaf_procedural_intel', 361)
nir_intrinsic_load_line_coord = nir_intrinsic_op.define('nir_intrinsic_load_line_coord', 362)
nir_intrinsic_load_line_width = nir_intrinsic_op.define('nir_intrinsic_load_line_width', 363)
nir_intrinsic_load_local_invocation_id = nir_intrinsic_op.define('nir_intrinsic_load_local_invocation_id', 364)
nir_intrinsic_load_local_invocation_index = nir_intrinsic_op.define('nir_intrinsic_load_local_invocation_index', 365)
nir_intrinsic_load_local_pixel_agx = nir_intrinsic_op.define('nir_intrinsic_load_local_pixel_agx', 366)
nir_intrinsic_load_local_shared_r600 = nir_intrinsic_op.define('nir_intrinsic_load_local_shared_r600', 367)
nir_intrinsic_load_lshs_vertex_stride_amd = nir_intrinsic_op.define('nir_intrinsic_load_lshs_vertex_stride_amd', 368)
nir_intrinsic_load_max_polygon_intel = nir_intrinsic_op.define('nir_intrinsic_load_max_polygon_intel', 369)
nir_intrinsic_load_merged_wave_info_amd = nir_intrinsic_op.define('nir_intrinsic_load_merged_wave_info_amd', 370)
nir_intrinsic_load_mesh_view_count = nir_intrinsic_op.define('nir_intrinsic_load_mesh_view_count', 371)
nir_intrinsic_load_mesh_view_indices = nir_intrinsic_op.define('nir_intrinsic_load_mesh_view_indices', 372)
nir_intrinsic_load_multisampled_pan = nir_intrinsic_op.define('nir_intrinsic_load_multisampled_pan', 373)
nir_intrinsic_load_noperspective_varyings_pan = nir_intrinsic_op.define('nir_intrinsic_load_noperspective_varyings_pan', 374)
nir_intrinsic_load_num_subgroups = nir_intrinsic_op.define('nir_intrinsic_load_num_subgroups', 375)
nir_intrinsic_load_num_vertices = nir_intrinsic_op.define('nir_intrinsic_load_num_vertices', 376)
nir_intrinsic_load_num_vertices_per_primitive_amd = nir_intrinsic_op.define('nir_intrinsic_load_num_vertices_per_primitive_amd', 377)
nir_intrinsic_load_num_workgroups = nir_intrinsic_op.define('nir_intrinsic_load_num_workgroups', 378)
nir_intrinsic_load_ordered_id_amd = nir_intrinsic_op.define('nir_intrinsic_load_ordered_id_amd', 379)
nir_intrinsic_load_output = nir_intrinsic_op.define('nir_intrinsic_load_output', 380)
nir_intrinsic_load_packed_passthrough_primitive_amd = nir_intrinsic_op.define('nir_intrinsic_load_packed_passthrough_primitive_amd', 381)
nir_intrinsic_load_param = nir_intrinsic_op.define('nir_intrinsic_load_param', 382)
nir_intrinsic_load_patch_vertices_in = nir_intrinsic_op.define('nir_intrinsic_load_patch_vertices_in', 383)
nir_intrinsic_load_per_primitive_input = nir_intrinsic_op.define('nir_intrinsic_load_per_primitive_input', 384)
nir_intrinsic_load_per_primitive_output = nir_intrinsic_op.define('nir_intrinsic_load_per_primitive_output', 385)
nir_intrinsic_load_per_primitive_remap_intel = nir_intrinsic_op.define('nir_intrinsic_load_per_primitive_remap_intel', 386)
nir_intrinsic_load_per_vertex_input = nir_intrinsic_op.define('nir_intrinsic_load_per_vertex_input', 387)
nir_intrinsic_load_per_vertex_output = nir_intrinsic_op.define('nir_intrinsic_load_per_vertex_output', 388)
nir_intrinsic_load_per_view_output = nir_intrinsic_op.define('nir_intrinsic_load_per_view_output', 389)
nir_intrinsic_load_persp_center_rhw_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_persp_center_rhw_ir3', 390)
nir_intrinsic_load_pipeline_stat_query_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_pipeline_stat_query_enabled_amd', 391)
nir_intrinsic_load_pixel_coord = nir_intrinsic_op.define('nir_intrinsic_load_pixel_coord', 392)
nir_intrinsic_load_point_coord = nir_intrinsic_op.define('nir_intrinsic_load_point_coord', 393)
nir_intrinsic_load_point_coord_maybe_flipped = nir_intrinsic_op.define('nir_intrinsic_load_point_coord_maybe_flipped', 394)
nir_intrinsic_load_poly_line_smooth_enabled = nir_intrinsic_op.define('nir_intrinsic_load_poly_line_smooth_enabled', 395)
nir_intrinsic_load_polygon_stipple_agx = nir_intrinsic_op.define('nir_intrinsic_load_polygon_stipple_agx', 396)
nir_intrinsic_load_polygon_stipple_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_load_polygon_stipple_buffer_amd', 397)
nir_intrinsic_load_preamble = nir_intrinsic_op.define('nir_intrinsic_load_preamble', 398)
nir_intrinsic_load_prim_gen_query_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_prim_gen_query_enabled_amd', 399)
nir_intrinsic_load_prim_xfb_query_enabled_amd = nir_intrinsic_op.define('nir_intrinsic_load_prim_xfb_query_enabled_amd', 400)
nir_intrinsic_load_primitive_id = nir_intrinsic_op.define('nir_intrinsic_load_primitive_id', 401)
nir_intrinsic_load_primitive_location_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_primitive_location_ir3', 402)
nir_intrinsic_load_printf_buffer_address = nir_intrinsic_op.define('nir_intrinsic_load_printf_buffer_address', 403)
nir_intrinsic_load_printf_buffer_size = nir_intrinsic_op.define('nir_intrinsic_load_printf_buffer_size', 404)
nir_intrinsic_load_provoking_last = nir_intrinsic_op.define('nir_intrinsic_load_provoking_last', 405)
nir_intrinsic_load_provoking_vtx_amd = nir_intrinsic_op.define('nir_intrinsic_load_provoking_vtx_amd', 406)
nir_intrinsic_load_provoking_vtx_in_prim_amd = nir_intrinsic_op.define('nir_intrinsic_load_provoking_vtx_in_prim_amd', 407)
nir_intrinsic_load_push_constant = nir_intrinsic_op.define('nir_intrinsic_load_push_constant', 408)
nir_intrinsic_load_push_constant_zink = nir_intrinsic_op.define('nir_intrinsic_load_push_constant_zink', 409)
nir_intrinsic_load_r600_indirect_per_vertex_input = nir_intrinsic_op.define('nir_intrinsic_load_r600_indirect_per_vertex_input', 410)
nir_intrinsic_load_rasterization_primitive_amd = nir_intrinsic_op.define('nir_intrinsic_load_rasterization_primitive_amd', 411)
nir_intrinsic_load_rasterization_samples_amd = nir_intrinsic_op.define('nir_intrinsic_load_rasterization_samples_amd', 412)
nir_intrinsic_load_rasterization_stream = nir_intrinsic_op.define('nir_intrinsic_load_rasterization_stream', 413)
nir_intrinsic_load_raw_output_pan = nir_intrinsic_op.define('nir_intrinsic_load_raw_output_pan', 414)
nir_intrinsic_load_raw_vertex_id_pan = nir_intrinsic_op.define('nir_intrinsic_load_raw_vertex_id_pan', 415)
nir_intrinsic_load_raw_vertex_offset_pan = nir_intrinsic_op.define('nir_intrinsic_load_raw_vertex_offset_pan', 416)
nir_intrinsic_load_ray_base_mem_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_base_mem_addr_intel', 417)
nir_intrinsic_load_ray_flags = nir_intrinsic_op.define('nir_intrinsic_load_ray_flags', 418)
nir_intrinsic_load_ray_geometry_index = nir_intrinsic_op.define('nir_intrinsic_load_ray_geometry_index', 419)
nir_intrinsic_load_ray_hit_kind = nir_intrinsic_op.define('nir_intrinsic_load_ray_hit_kind', 420)
nir_intrinsic_load_ray_hit_sbt_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_hit_sbt_addr_intel', 421)
nir_intrinsic_load_ray_hit_sbt_stride_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_hit_sbt_stride_intel', 422)
nir_intrinsic_load_ray_hw_stack_size_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_hw_stack_size_intel', 423)
nir_intrinsic_load_ray_instance_custom_index = nir_intrinsic_op.define('nir_intrinsic_load_ray_instance_custom_index', 424)
nir_intrinsic_load_ray_launch_id = nir_intrinsic_op.define('nir_intrinsic_load_ray_launch_id', 425)
nir_intrinsic_load_ray_launch_size = nir_intrinsic_op.define('nir_intrinsic_load_ray_launch_size', 426)
nir_intrinsic_load_ray_miss_sbt_addr_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_miss_sbt_addr_intel', 427)
nir_intrinsic_load_ray_miss_sbt_stride_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_miss_sbt_stride_intel', 428)
nir_intrinsic_load_ray_num_dss_rt_stacks_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_num_dss_rt_stacks_intel', 429)
nir_intrinsic_load_ray_object_direction = nir_intrinsic_op.define('nir_intrinsic_load_ray_object_direction', 430)
nir_intrinsic_load_ray_object_origin = nir_intrinsic_op.define('nir_intrinsic_load_ray_object_origin', 431)
nir_intrinsic_load_ray_object_to_world = nir_intrinsic_op.define('nir_intrinsic_load_ray_object_to_world', 432)
nir_intrinsic_load_ray_query_global_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_query_global_intel', 433)
nir_intrinsic_load_ray_sw_stack_size_intel = nir_intrinsic_op.define('nir_intrinsic_load_ray_sw_stack_size_intel', 434)
nir_intrinsic_load_ray_t_max = nir_intrinsic_op.define('nir_intrinsic_load_ray_t_max', 435)
nir_intrinsic_load_ray_t_min = nir_intrinsic_op.define('nir_intrinsic_load_ray_t_min', 436)
nir_intrinsic_load_ray_tracing_stack_base_lvp = nir_intrinsic_op.define('nir_intrinsic_load_ray_tracing_stack_base_lvp', 437)
nir_intrinsic_load_ray_triangle_vertex_positions = nir_intrinsic_op.define('nir_intrinsic_load_ray_triangle_vertex_positions', 438)
nir_intrinsic_load_ray_world_direction = nir_intrinsic_op.define('nir_intrinsic_load_ray_world_direction', 439)
nir_intrinsic_load_ray_world_origin = nir_intrinsic_op.define('nir_intrinsic_load_ray_world_origin', 440)
nir_intrinsic_load_ray_world_to_object = nir_intrinsic_op.define('nir_intrinsic_load_ray_world_to_object', 441)
nir_intrinsic_load_readonly_output_pan = nir_intrinsic_op.define('nir_intrinsic_load_readonly_output_pan', 442)
nir_intrinsic_load_reg = nir_intrinsic_op.define('nir_intrinsic_load_reg', 443)
nir_intrinsic_load_reg_indirect = nir_intrinsic_op.define('nir_intrinsic_load_reg_indirect', 444)
nir_intrinsic_load_rel_patch_id_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_rel_patch_id_ir3', 445)
nir_intrinsic_load_reloc_const_intel = nir_intrinsic_op.define('nir_intrinsic_load_reloc_const_intel', 446)
nir_intrinsic_load_resume_shader_address_amd = nir_intrinsic_op.define('nir_intrinsic_load_resume_shader_address_amd', 447)
nir_intrinsic_load_ring_attr_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_attr_amd', 448)
nir_intrinsic_load_ring_attr_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_attr_offset_amd', 449)
nir_intrinsic_load_ring_es2gs_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_es2gs_offset_amd', 450)
nir_intrinsic_load_ring_esgs_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_esgs_amd', 451)
nir_intrinsic_load_ring_gs2vs_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_gs2vs_offset_amd', 452)
nir_intrinsic_load_ring_gsvs_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_gsvs_amd', 453)
nir_intrinsic_load_ring_mesh_scratch_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_mesh_scratch_amd', 454)
nir_intrinsic_load_ring_mesh_scratch_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_mesh_scratch_offset_amd', 455)
nir_intrinsic_load_ring_task_draw_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_task_draw_amd', 456)
nir_intrinsic_load_ring_task_payload_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_task_payload_amd', 457)
nir_intrinsic_load_ring_tess_factors_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_tess_factors_amd', 458)
nir_intrinsic_load_ring_tess_factors_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_tess_factors_offset_amd', 459)
nir_intrinsic_load_ring_tess_offchip_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_tess_offchip_amd', 460)
nir_intrinsic_load_ring_tess_offchip_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_ring_tess_offchip_offset_amd', 461)
nir_intrinsic_load_root_agx = nir_intrinsic_op.define('nir_intrinsic_load_root_agx', 462)
nir_intrinsic_load_rt_arg_scratch_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_rt_arg_scratch_offset_amd', 463)
nir_intrinsic_load_rt_conversion_pan = nir_intrinsic_op.define('nir_intrinsic_load_rt_conversion_pan', 464)
nir_intrinsic_load_sample_id = nir_intrinsic_op.define('nir_intrinsic_load_sample_id', 465)
nir_intrinsic_load_sample_id_no_per_sample = nir_intrinsic_op.define('nir_intrinsic_load_sample_id_no_per_sample', 466)
nir_intrinsic_load_sample_mask = nir_intrinsic_op.define('nir_intrinsic_load_sample_mask', 467)
nir_intrinsic_load_sample_mask_in = nir_intrinsic_op.define('nir_intrinsic_load_sample_mask_in', 468)
nir_intrinsic_load_sample_pos = nir_intrinsic_op.define('nir_intrinsic_load_sample_pos', 469)
nir_intrinsic_load_sample_pos_from_id = nir_intrinsic_op.define('nir_intrinsic_load_sample_pos_from_id', 470)
nir_intrinsic_load_sample_pos_or_center = nir_intrinsic_op.define('nir_intrinsic_load_sample_pos_or_center', 471)
nir_intrinsic_load_sample_positions_agx = nir_intrinsic_op.define('nir_intrinsic_load_sample_positions_agx', 472)
nir_intrinsic_load_sample_positions_amd = nir_intrinsic_op.define('nir_intrinsic_load_sample_positions_amd', 473)
nir_intrinsic_load_sample_positions_pan = nir_intrinsic_op.define('nir_intrinsic_load_sample_positions_pan', 474)
nir_intrinsic_load_sampler_handle_agx = nir_intrinsic_op.define('nir_intrinsic_load_sampler_handle_agx', 475)
nir_intrinsic_load_sampler_lod_parameters = nir_intrinsic_op.define('nir_intrinsic_load_sampler_lod_parameters', 476)
nir_intrinsic_load_samples_log2_agx = nir_intrinsic_op.define('nir_intrinsic_load_samples_log2_agx', 477)
nir_intrinsic_load_sbt_base_amd = nir_intrinsic_op.define('nir_intrinsic_load_sbt_base_amd', 478)
nir_intrinsic_load_sbt_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_sbt_offset_amd', 479)
nir_intrinsic_load_sbt_stride_amd = nir_intrinsic_op.define('nir_intrinsic_load_sbt_stride_amd', 480)
nir_intrinsic_load_scalar_arg_amd = nir_intrinsic_op.define('nir_intrinsic_load_scalar_arg_amd', 481)
nir_intrinsic_load_scratch = nir_intrinsic_op.define('nir_intrinsic_load_scratch', 482)
nir_intrinsic_load_scratch_base_ptr = nir_intrinsic_op.define('nir_intrinsic_load_scratch_base_ptr', 483)
nir_intrinsic_load_shader_call_data_offset_lvp = nir_intrinsic_op.define('nir_intrinsic_load_shader_call_data_offset_lvp', 484)
nir_intrinsic_load_shader_index = nir_intrinsic_op.define('nir_intrinsic_load_shader_index', 485)
nir_intrinsic_load_shader_output_pan = nir_intrinsic_op.define('nir_intrinsic_load_shader_output_pan', 486)
nir_intrinsic_load_shader_part_tests_zs_agx = nir_intrinsic_op.define('nir_intrinsic_load_shader_part_tests_zs_agx', 487)
nir_intrinsic_load_shader_record_ptr = nir_intrinsic_op.define('nir_intrinsic_load_shader_record_ptr', 488)
nir_intrinsic_load_shared = nir_intrinsic_op.define('nir_intrinsic_load_shared', 489)
nir_intrinsic_load_shared2_amd = nir_intrinsic_op.define('nir_intrinsic_load_shared2_amd', 490)
nir_intrinsic_load_shared_base_ptr = nir_intrinsic_op.define('nir_intrinsic_load_shared_base_ptr', 491)
nir_intrinsic_load_shared_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_shared_block_intel', 492)
nir_intrinsic_load_shared_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_shared_ir3', 493)
nir_intrinsic_load_shared_lock_nv = nir_intrinsic_op.define('nir_intrinsic_load_shared_lock_nv', 494)
nir_intrinsic_load_shared_uniform_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_shared_uniform_block_intel', 495)
nir_intrinsic_load_simd_width_intel = nir_intrinsic_op.define('nir_intrinsic_load_simd_width_intel', 496)
nir_intrinsic_load_sm_count_nv = nir_intrinsic_op.define('nir_intrinsic_load_sm_count_nv', 497)
nir_intrinsic_load_sm_id_nv = nir_intrinsic_op.define('nir_intrinsic_load_sm_id_nv', 498)
nir_intrinsic_load_smem_amd = nir_intrinsic_op.define('nir_intrinsic_load_smem_amd', 499)
nir_intrinsic_load_ssbo = nir_intrinsic_op.define('nir_intrinsic_load_ssbo', 500)
nir_intrinsic_load_ssbo_address = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_address', 501)
nir_intrinsic_load_ssbo_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_block_intel', 502)
nir_intrinsic_load_ssbo_intel = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_intel', 503)
nir_intrinsic_load_ssbo_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_ir3', 504)
nir_intrinsic_load_ssbo_uniform_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_ssbo_uniform_block_intel', 505)
nir_intrinsic_load_stack = nir_intrinsic_op.define('nir_intrinsic_load_stack', 506)
nir_intrinsic_load_stat_query_address_agx = nir_intrinsic_op.define('nir_intrinsic_load_stat_query_address_agx', 507)
nir_intrinsic_load_streamout_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_load_streamout_buffer_amd', 508)
nir_intrinsic_load_streamout_config_amd = nir_intrinsic_op.define('nir_intrinsic_load_streamout_config_amd', 509)
nir_intrinsic_load_streamout_offset_amd = nir_intrinsic_op.define('nir_intrinsic_load_streamout_offset_amd', 510)
nir_intrinsic_load_streamout_write_index_amd = nir_intrinsic_op.define('nir_intrinsic_load_streamout_write_index_amd', 511)
nir_intrinsic_load_subgroup_eq_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_eq_mask', 512)
nir_intrinsic_load_subgroup_ge_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_ge_mask', 513)
nir_intrinsic_load_subgroup_gt_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_gt_mask', 514)
nir_intrinsic_load_subgroup_id = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_id', 515)
nir_intrinsic_load_subgroup_id_shift_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_id_shift_ir3', 516)
nir_intrinsic_load_subgroup_invocation = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_invocation', 517)
nir_intrinsic_load_subgroup_le_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_le_mask', 518)
nir_intrinsic_load_subgroup_lt_mask = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_lt_mask', 519)
nir_intrinsic_load_subgroup_size = nir_intrinsic_op.define('nir_intrinsic_load_subgroup_size', 520)
nir_intrinsic_load_sysval_agx = nir_intrinsic_op.define('nir_intrinsic_load_sysval_agx', 521)
nir_intrinsic_load_sysval_nv = nir_intrinsic_op.define('nir_intrinsic_load_sysval_nv', 522)
nir_intrinsic_load_task_payload = nir_intrinsic_op.define('nir_intrinsic_load_task_payload', 523)
nir_intrinsic_load_task_ring_entry_amd = nir_intrinsic_op.define('nir_intrinsic_load_task_ring_entry_amd', 524)
nir_intrinsic_load_tcs_header_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_header_ir3', 525)
nir_intrinsic_load_tcs_in_param_base_r600 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_in_param_base_r600', 526)
nir_intrinsic_load_tcs_mem_attrib_stride = nir_intrinsic_op.define('nir_intrinsic_load_tcs_mem_attrib_stride', 527)
nir_intrinsic_load_tcs_num_patches_amd = nir_intrinsic_op.define('nir_intrinsic_load_tcs_num_patches_amd', 528)
nir_intrinsic_load_tcs_out_param_base_r600 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_out_param_base_r600', 529)
nir_intrinsic_load_tcs_primitive_mode_amd = nir_intrinsic_op.define('nir_intrinsic_load_tcs_primitive_mode_amd', 530)
nir_intrinsic_load_tcs_rel_patch_id_r600 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_rel_patch_id_r600', 531)
nir_intrinsic_load_tcs_tess_factor_base_r600 = nir_intrinsic_op.define('nir_intrinsic_load_tcs_tess_factor_base_r600', 532)
nir_intrinsic_load_tcs_tess_levels_to_tes_amd = nir_intrinsic_op.define('nir_intrinsic_load_tcs_tess_levels_to_tes_amd', 533)
nir_intrinsic_load_tess_coord = nir_intrinsic_op.define('nir_intrinsic_load_tess_coord', 534)
nir_intrinsic_load_tess_coord_xy = nir_intrinsic_op.define('nir_intrinsic_load_tess_coord_xy', 535)
nir_intrinsic_load_tess_factor_base_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_tess_factor_base_ir3', 536)
nir_intrinsic_load_tess_level_inner = nir_intrinsic_op.define('nir_intrinsic_load_tess_level_inner', 537)
nir_intrinsic_load_tess_level_inner_default = nir_intrinsic_op.define('nir_intrinsic_load_tess_level_inner_default', 538)
nir_intrinsic_load_tess_level_outer = nir_intrinsic_op.define('nir_intrinsic_load_tess_level_outer', 539)
nir_intrinsic_load_tess_level_outer_default = nir_intrinsic_op.define('nir_intrinsic_load_tess_level_outer_default', 540)
nir_intrinsic_load_tess_param_base_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_tess_param_base_ir3', 541)
nir_intrinsic_load_tess_param_buffer_poly = nir_intrinsic_op.define('nir_intrinsic_load_tess_param_buffer_poly', 542)
nir_intrinsic_load_tess_rel_patch_id_amd = nir_intrinsic_op.define('nir_intrinsic_load_tess_rel_patch_id_amd', 543)
nir_intrinsic_load_tex_sprite_mask_agx = nir_intrinsic_op.define('nir_intrinsic_load_tex_sprite_mask_agx', 544)
nir_intrinsic_load_texture_handle_agx = nir_intrinsic_op.define('nir_intrinsic_load_texture_handle_agx', 545)
nir_intrinsic_load_texture_scale = nir_intrinsic_op.define('nir_intrinsic_load_texture_scale', 546)
nir_intrinsic_load_texture_size_etna = nir_intrinsic_op.define('nir_intrinsic_load_texture_size_etna', 547)
nir_intrinsic_load_tlb_color_brcm = nir_intrinsic_op.define('nir_intrinsic_load_tlb_color_brcm', 548)
nir_intrinsic_load_topology_id_intel = nir_intrinsic_op.define('nir_intrinsic_load_topology_id_intel', 549)
nir_intrinsic_load_typed_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_load_typed_buffer_amd', 550)
nir_intrinsic_load_uav_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_uav_ir3', 551)
nir_intrinsic_load_ubo = nir_intrinsic_op.define('nir_intrinsic_load_ubo', 552)
nir_intrinsic_load_ubo_uniform_block_intel = nir_intrinsic_op.define('nir_intrinsic_load_ubo_uniform_block_intel', 553)
nir_intrinsic_load_ubo_vec4 = nir_intrinsic_op.define('nir_intrinsic_load_ubo_vec4', 554)
nir_intrinsic_load_uniform = nir_intrinsic_op.define('nir_intrinsic_load_uniform', 555)
nir_intrinsic_load_user_clip_plane = nir_intrinsic_op.define('nir_intrinsic_load_user_clip_plane', 556)
nir_intrinsic_load_user_data_amd = nir_intrinsic_op.define('nir_intrinsic_load_user_data_amd', 557)
nir_intrinsic_load_uvs_index_agx = nir_intrinsic_op.define('nir_intrinsic_load_uvs_index_agx', 558)
nir_intrinsic_load_vbo_base_agx = nir_intrinsic_op.define('nir_intrinsic_load_vbo_base_agx', 559)
nir_intrinsic_load_vector_arg_amd = nir_intrinsic_op.define('nir_intrinsic_load_vector_arg_amd', 560)
nir_intrinsic_load_vertex_id = nir_intrinsic_op.define('nir_intrinsic_load_vertex_id', 561)
nir_intrinsic_load_vertex_id_zero_base = nir_intrinsic_op.define('nir_intrinsic_load_vertex_id_zero_base', 562)
nir_intrinsic_load_view_index = nir_intrinsic_op.define('nir_intrinsic_load_view_index', 563)
nir_intrinsic_load_viewport_offset = nir_intrinsic_op.define('nir_intrinsic_load_viewport_offset', 564)
nir_intrinsic_load_viewport_scale = nir_intrinsic_op.define('nir_intrinsic_load_viewport_scale', 565)
nir_intrinsic_load_viewport_x_offset = nir_intrinsic_op.define('nir_intrinsic_load_viewport_x_offset', 566)
nir_intrinsic_load_viewport_x_scale = nir_intrinsic_op.define('nir_intrinsic_load_viewport_x_scale', 567)
nir_intrinsic_load_viewport_y_offset = nir_intrinsic_op.define('nir_intrinsic_load_viewport_y_offset', 568)
nir_intrinsic_load_viewport_y_scale = nir_intrinsic_op.define('nir_intrinsic_load_viewport_y_scale', 569)
nir_intrinsic_load_viewport_z_offset = nir_intrinsic_op.define('nir_intrinsic_load_viewport_z_offset', 570)
nir_intrinsic_load_viewport_z_scale = nir_intrinsic_op.define('nir_intrinsic_load_viewport_z_scale', 571)
nir_intrinsic_load_vs_output_buffer_poly = nir_intrinsic_op.define('nir_intrinsic_load_vs_output_buffer_poly', 572)
nir_intrinsic_load_vs_outputs_poly = nir_intrinsic_op.define('nir_intrinsic_load_vs_outputs_poly', 573)
nir_intrinsic_load_vs_primitive_stride_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_vs_primitive_stride_ir3', 574)
nir_intrinsic_load_vs_vertex_stride_ir3 = nir_intrinsic_op.define('nir_intrinsic_load_vs_vertex_stride_ir3', 575)
nir_intrinsic_load_vulkan_descriptor = nir_intrinsic_op.define('nir_intrinsic_load_vulkan_descriptor', 576)
nir_intrinsic_load_warp_id_nv = nir_intrinsic_op.define('nir_intrinsic_load_warp_id_nv', 577)
nir_intrinsic_load_warps_per_sm_nv = nir_intrinsic_op.define('nir_intrinsic_load_warps_per_sm_nv', 578)
nir_intrinsic_load_work_dim = nir_intrinsic_op.define('nir_intrinsic_load_work_dim', 579)
nir_intrinsic_load_workgroup_id = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_id', 580)
nir_intrinsic_load_workgroup_index = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_index', 581)
nir_intrinsic_load_workgroup_num_input_primitives_amd = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_num_input_primitives_amd', 582)
nir_intrinsic_load_workgroup_num_input_vertices_amd = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_num_input_vertices_amd', 583)
nir_intrinsic_load_workgroup_size = nir_intrinsic_op.define('nir_intrinsic_load_workgroup_size', 584)
nir_intrinsic_load_xfb_address = nir_intrinsic_op.define('nir_intrinsic_load_xfb_address', 585)
nir_intrinsic_load_xfb_index_buffer = nir_intrinsic_op.define('nir_intrinsic_load_xfb_index_buffer', 586)
nir_intrinsic_load_xfb_size = nir_intrinsic_op.define('nir_intrinsic_load_xfb_size', 587)
nir_intrinsic_load_xfb_state_address_gfx12_amd = nir_intrinsic_op.define('nir_intrinsic_load_xfb_state_address_gfx12_amd', 588)
nir_intrinsic_masked_swizzle_amd = nir_intrinsic_op.define('nir_intrinsic_masked_swizzle_amd', 589)
nir_intrinsic_mbcnt_amd = nir_intrinsic_op.define('nir_intrinsic_mbcnt_amd', 590)
nir_intrinsic_memcpy_deref = nir_intrinsic_op.define('nir_intrinsic_memcpy_deref', 591)
nir_intrinsic_nop = nir_intrinsic_op.define('nir_intrinsic_nop', 592)
nir_intrinsic_nop_amd = nir_intrinsic_op.define('nir_intrinsic_nop_amd', 593)
nir_intrinsic_optimization_barrier_sgpr_amd = nir_intrinsic_op.define('nir_intrinsic_optimization_barrier_sgpr_amd', 594)
nir_intrinsic_optimization_barrier_vgpr_amd = nir_intrinsic_op.define('nir_intrinsic_optimization_barrier_vgpr_amd', 595)
nir_intrinsic_ordered_add_loop_gfx12_amd = nir_intrinsic_op.define('nir_intrinsic_ordered_add_loop_gfx12_amd', 596)
nir_intrinsic_ordered_xfb_counter_add_gfx11_amd = nir_intrinsic_op.define('nir_intrinsic_ordered_xfb_counter_add_gfx11_amd', 597)
nir_intrinsic_overwrite_tes_arguments_amd = nir_intrinsic_op.define('nir_intrinsic_overwrite_tes_arguments_amd', 598)
nir_intrinsic_overwrite_vs_arguments_amd = nir_intrinsic_op.define('nir_intrinsic_overwrite_vs_arguments_amd', 599)
nir_intrinsic_pin_cx_handle_nv = nir_intrinsic_op.define('nir_intrinsic_pin_cx_handle_nv', 600)
nir_intrinsic_preamble_end_ir3 = nir_intrinsic_op.define('nir_intrinsic_preamble_end_ir3', 601)
nir_intrinsic_preamble_start_ir3 = nir_intrinsic_op.define('nir_intrinsic_preamble_start_ir3', 602)
nir_intrinsic_prefetch_sam_ir3 = nir_intrinsic_op.define('nir_intrinsic_prefetch_sam_ir3', 603)
nir_intrinsic_prefetch_tex_ir3 = nir_intrinsic_op.define('nir_intrinsic_prefetch_tex_ir3', 604)
nir_intrinsic_prefetch_ubo_ir3 = nir_intrinsic_op.define('nir_intrinsic_prefetch_ubo_ir3', 605)
nir_intrinsic_printf = nir_intrinsic_op.define('nir_intrinsic_printf', 606)
nir_intrinsic_printf_abort = nir_intrinsic_op.define('nir_intrinsic_printf_abort', 607)
nir_intrinsic_quad_ballot_agx = nir_intrinsic_op.define('nir_intrinsic_quad_ballot_agx', 608)
nir_intrinsic_quad_broadcast = nir_intrinsic_op.define('nir_intrinsic_quad_broadcast', 609)
nir_intrinsic_quad_swap_diagonal = nir_intrinsic_op.define('nir_intrinsic_quad_swap_diagonal', 610)
nir_intrinsic_quad_swap_horizontal = nir_intrinsic_op.define('nir_intrinsic_quad_swap_horizontal', 611)
nir_intrinsic_quad_swap_vertical = nir_intrinsic_op.define('nir_intrinsic_quad_swap_vertical', 612)
nir_intrinsic_quad_swizzle_amd = nir_intrinsic_op.define('nir_intrinsic_quad_swizzle_amd', 613)
nir_intrinsic_quad_vote_all = nir_intrinsic_op.define('nir_intrinsic_quad_vote_all', 614)
nir_intrinsic_quad_vote_any = nir_intrinsic_op.define('nir_intrinsic_quad_vote_any', 615)
nir_intrinsic_r600_indirect_vertex_at_index = nir_intrinsic_op.define('nir_intrinsic_r600_indirect_vertex_at_index', 616)
nir_intrinsic_ray_intersection_ir3 = nir_intrinsic_op.define('nir_intrinsic_ray_intersection_ir3', 617)
nir_intrinsic_read_attribute_payload_intel = nir_intrinsic_op.define('nir_intrinsic_read_attribute_payload_intel', 618)
nir_intrinsic_read_first_invocation = nir_intrinsic_op.define('nir_intrinsic_read_first_invocation', 619)
nir_intrinsic_read_getlast_ir3 = nir_intrinsic_op.define('nir_intrinsic_read_getlast_ir3', 620)
nir_intrinsic_read_invocation = nir_intrinsic_op.define('nir_intrinsic_read_invocation', 621)
nir_intrinsic_read_invocation_cond_ir3 = nir_intrinsic_op.define('nir_intrinsic_read_invocation_cond_ir3', 622)
nir_intrinsic_reduce = nir_intrinsic_op.define('nir_intrinsic_reduce', 623)
nir_intrinsic_reduce_clusters_ir3 = nir_intrinsic_op.define('nir_intrinsic_reduce_clusters_ir3', 624)
nir_intrinsic_report_ray_intersection = nir_intrinsic_op.define('nir_intrinsic_report_ray_intersection', 625)
nir_intrinsic_resource_intel = nir_intrinsic_op.define('nir_intrinsic_resource_intel', 626)
nir_intrinsic_rotate = nir_intrinsic_op.define('nir_intrinsic_rotate', 627)
nir_intrinsic_rq_confirm_intersection = nir_intrinsic_op.define('nir_intrinsic_rq_confirm_intersection', 628)
nir_intrinsic_rq_generate_intersection = nir_intrinsic_op.define('nir_intrinsic_rq_generate_intersection', 629)
nir_intrinsic_rq_initialize = nir_intrinsic_op.define('nir_intrinsic_rq_initialize', 630)
nir_intrinsic_rq_load = nir_intrinsic_op.define('nir_intrinsic_rq_load', 631)
nir_intrinsic_rq_proceed = nir_intrinsic_op.define('nir_intrinsic_rq_proceed', 632)
nir_intrinsic_rq_terminate = nir_intrinsic_op.define('nir_intrinsic_rq_terminate', 633)
nir_intrinsic_rt_execute_callable = nir_intrinsic_op.define('nir_intrinsic_rt_execute_callable', 634)
nir_intrinsic_rt_resume = nir_intrinsic_op.define('nir_intrinsic_rt_resume', 635)
nir_intrinsic_rt_return_amd = nir_intrinsic_op.define('nir_intrinsic_rt_return_amd', 636)
nir_intrinsic_rt_trace_ray = nir_intrinsic_op.define('nir_intrinsic_rt_trace_ray', 637)
nir_intrinsic_sample_mask_agx = nir_intrinsic_op.define('nir_intrinsic_sample_mask_agx', 638)
nir_intrinsic_select_vertex_poly = nir_intrinsic_op.define('nir_intrinsic_select_vertex_poly', 639)
nir_intrinsic_sendmsg_amd = nir_intrinsic_op.define('nir_intrinsic_sendmsg_amd', 640)
nir_intrinsic_set_vertex_and_primitive_count = nir_intrinsic_op.define('nir_intrinsic_set_vertex_and_primitive_count', 641)
nir_intrinsic_shader_clock = nir_intrinsic_op.define('nir_intrinsic_shader_clock', 642)
nir_intrinsic_shared_append_amd = nir_intrinsic_op.define('nir_intrinsic_shared_append_amd', 643)
nir_intrinsic_shared_atomic = nir_intrinsic_op.define('nir_intrinsic_shared_atomic', 644)
nir_intrinsic_shared_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_shared_atomic_swap', 645)
nir_intrinsic_shared_consume_amd = nir_intrinsic_op.define('nir_intrinsic_shared_consume_amd', 646)
nir_intrinsic_shuffle = nir_intrinsic_op.define('nir_intrinsic_shuffle', 647)
nir_intrinsic_shuffle_down = nir_intrinsic_op.define('nir_intrinsic_shuffle_down', 648)
nir_intrinsic_shuffle_down_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_shuffle_down_uniform_ir3', 649)
nir_intrinsic_shuffle_up = nir_intrinsic_op.define('nir_intrinsic_shuffle_up', 650)
nir_intrinsic_shuffle_up_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_shuffle_up_uniform_ir3', 651)
nir_intrinsic_shuffle_xor = nir_intrinsic_op.define('nir_intrinsic_shuffle_xor', 652)
nir_intrinsic_shuffle_xor_uniform_ir3 = nir_intrinsic_op.define('nir_intrinsic_shuffle_xor_uniform_ir3', 653)
nir_intrinsic_sleep_amd = nir_intrinsic_op.define('nir_intrinsic_sleep_amd', 654)
nir_intrinsic_sparse_residency_code_and = nir_intrinsic_op.define('nir_intrinsic_sparse_residency_code_and', 655)
nir_intrinsic_ssa_bar_nv = nir_intrinsic_op.define('nir_intrinsic_ssa_bar_nv', 656)
nir_intrinsic_ssbo_atomic = nir_intrinsic_op.define('nir_intrinsic_ssbo_atomic', 657)
nir_intrinsic_ssbo_atomic_ir3 = nir_intrinsic_op.define('nir_intrinsic_ssbo_atomic_ir3', 658)
nir_intrinsic_ssbo_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_ssbo_atomic_swap', 659)
nir_intrinsic_ssbo_atomic_swap_ir3 = nir_intrinsic_op.define('nir_intrinsic_ssbo_atomic_swap_ir3', 660)
nir_intrinsic_stack_map_agx = nir_intrinsic_op.define('nir_intrinsic_stack_map_agx', 661)
nir_intrinsic_stack_unmap_agx = nir_intrinsic_op.define('nir_intrinsic_stack_unmap_agx', 662)
nir_intrinsic_store_agx = nir_intrinsic_op.define('nir_intrinsic_store_agx', 663)
nir_intrinsic_store_buffer_amd = nir_intrinsic_op.define('nir_intrinsic_store_buffer_amd', 664)
nir_intrinsic_store_combined_output_pan = nir_intrinsic_op.define('nir_intrinsic_store_combined_output_pan', 665)
nir_intrinsic_store_const_ir3 = nir_intrinsic_op.define('nir_intrinsic_store_const_ir3', 666)
nir_intrinsic_store_deref = nir_intrinsic_op.define('nir_intrinsic_store_deref', 667)
nir_intrinsic_store_deref_block_intel = nir_intrinsic_op.define('nir_intrinsic_store_deref_block_intel', 668)
nir_intrinsic_store_global = nir_intrinsic_op.define('nir_intrinsic_store_global', 669)
nir_intrinsic_store_global_2x32 = nir_intrinsic_op.define('nir_intrinsic_store_global_2x32', 670)
nir_intrinsic_store_global_amd = nir_intrinsic_op.define('nir_intrinsic_store_global_amd', 671)
nir_intrinsic_store_global_block_intel = nir_intrinsic_op.define('nir_intrinsic_store_global_block_intel', 672)
nir_intrinsic_store_global_etna = nir_intrinsic_op.define('nir_intrinsic_store_global_etna', 673)
nir_intrinsic_store_global_ir3 = nir_intrinsic_op.define('nir_intrinsic_store_global_ir3', 674)
nir_intrinsic_store_hit_attrib_amd = nir_intrinsic_op.define('nir_intrinsic_store_hit_attrib_amd', 675)
nir_intrinsic_store_local_pixel_agx = nir_intrinsic_op.define('nir_intrinsic_store_local_pixel_agx', 676)
nir_intrinsic_store_local_shared_r600 = nir_intrinsic_op.define('nir_intrinsic_store_local_shared_r600', 677)
nir_intrinsic_store_output = nir_intrinsic_op.define('nir_intrinsic_store_output', 678)
nir_intrinsic_store_per_primitive_output = nir_intrinsic_op.define('nir_intrinsic_store_per_primitive_output', 679)
nir_intrinsic_store_per_primitive_payload_intel = nir_intrinsic_op.define('nir_intrinsic_store_per_primitive_payload_intel', 680)
nir_intrinsic_store_per_vertex_output = nir_intrinsic_op.define('nir_intrinsic_store_per_vertex_output', 681)
nir_intrinsic_store_per_view_output = nir_intrinsic_op.define('nir_intrinsic_store_per_view_output', 682)
nir_intrinsic_store_preamble = nir_intrinsic_op.define('nir_intrinsic_store_preamble', 683)
nir_intrinsic_store_raw_output_pan = nir_intrinsic_op.define('nir_intrinsic_store_raw_output_pan', 684)
nir_intrinsic_store_reg = nir_intrinsic_op.define('nir_intrinsic_store_reg', 685)
nir_intrinsic_store_reg_indirect = nir_intrinsic_op.define('nir_intrinsic_store_reg_indirect', 686)
nir_intrinsic_store_scalar_arg_amd = nir_intrinsic_op.define('nir_intrinsic_store_scalar_arg_amd', 687)
nir_intrinsic_store_scratch = nir_intrinsic_op.define('nir_intrinsic_store_scratch', 688)
nir_intrinsic_store_shared = nir_intrinsic_op.define('nir_intrinsic_store_shared', 689)
nir_intrinsic_store_shared2_amd = nir_intrinsic_op.define('nir_intrinsic_store_shared2_amd', 690)
nir_intrinsic_store_shared_block_intel = nir_intrinsic_op.define('nir_intrinsic_store_shared_block_intel', 691)
nir_intrinsic_store_shared_ir3 = nir_intrinsic_op.define('nir_intrinsic_store_shared_ir3', 692)
nir_intrinsic_store_shared_unlock_nv = nir_intrinsic_op.define('nir_intrinsic_store_shared_unlock_nv', 693)
nir_intrinsic_store_ssbo = nir_intrinsic_op.define('nir_intrinsic_store_ssbo', 694)
nir_intrinsic_store_ssbo_block_intel = nir_intrinsic_op.define('nir_intrinsic_store_ssbo_block_intel', 695)
nir_intrinsic_store_ssbo_intel = nir_intrinsic_op.define('nir_intrinsic_store_ssbo_intel', 696)
nir_intrinsic_store_ssbo_ir3 = nir_intrinsic_op.define('nir_intrinsic_store_ssbo_ir3', 697)
nir_intrinsic_store_stack = nir_intrinsic_op.define('nir_intrinsic_store_stack', 698)
nir_intrinsic_store_task_payload = nir_intrinsic_op.define('nir_intrinsic_store_task_payload', 699)
nir_intrinsic_store_tf_r600 = nir_intrinsic_op.define('nir_intrinsic_store_tf_r600', 700)
nir_intrinsic_store_tlb_sample_color_v3d = nir_intrinsic_op.define('nir_intrinsic_store_tlb_sample_color_v3d', 701)
nir_intrinsic_store_uvs_agx = nir_intrinsic_op.define('nir_intrinsic_store_uvs_agx', 702)
nir_intrinsic_store_vector_arg_amd = nir_intrinsic_op.define('nir_intrinsic_store_vector_arg_amd', 703)
nir_intrinsic_store_zs_agx = nir_intrinsic_op.define('nir_intrinsic_store_zs_agx', 704)
nir_intrinsic_strict_wqm_coord_amd = nir_intrinsic_op.define('nir_intrinsic_strict_wqm_coord_amd', 705)
nir_intrinsic_subfm_nv = nir_intrinsic_op.define('nir_intrinsic_subfm_nv', 706)
nir_intrinsic_suclamp_nv = nir_intrinsic_op.define('nir_intrinsic_suclamp_nv', 707)
nir_intrinsic_sueau_nv = nir_intrinsic_op.define('nir_intrinsic_sueau_nv', 708)
nir_intrinsic_suldga_nv = nir_intrinsic_op.define('nir_intrinsic_suldga_nv', 709)
nir_intrinsic_sustga_nv = nir_intrinsic_op.define('nir_intrinsic_sustga_nv', 710)
nir_intrinsic_task_payload_atomic = nir_intrinsic_op.define('nir_intrinsic_task_payload_atomic', 711)
nir_intrinsic_task_payload_atomic_swap = nir_intrinsic_op.define('nir_intrinsic_task_payload_atomic_swap', 712)
nir_intrinsic_terminate = nir_intrinsic_op.define('nir_intrinsic_terminate', 713)
nir_intrinsic_terminate_if = nir_intrinsic_op.define('nir_intrinsic_terminate_if', 714)
nir_intrinsic_terminate_ray = nir_intrinsic_op.define('nir_intrinsic_terminate_ray', 715)
nir_intrinsic_trace_ray = nir_intrinsic_op.define('nir_intrinsic_trace_ray', 716)
nir_intrinsic_trace_ray_intel = nir_intrinsic_op.define('nir_intrinsic_trace_ray_intel', 717)
nir_intrinsic_unit_test_amd = nir_intrinsic_op.define('nir_intrinsic_unit_test_amd', 718)
nir_intrinsic_unit_test_divergent_amd = nir_intrinsic_op.define('nir_intrinsic_unit_test_divergent_amd', 719)
nir_intrinsic_unit_test_uniform_amd = nir_intrinsic_op.define('nir_intrinsic_unit_test_uniform_amd', 720)
nir_intrinsic_unpin_cx_handle_nv = nir_intrinsic_op.define('nir_intrinsic_unpin_cx_handle_nv', 721)
nir_intrinsic_use = nir_intrinsic_op.define('nir_intrinsic_use', 722)
nir_intrinsic_vild_nv = nir_intrinsic_op.define('nir_intrinsic_vild_nv', 723)
nir_intrinsic_vote_all = nir_intrinsic_op.define('nir_intrinsic_vote_all', 724)
nir_intrinsic_vote_any = nir_intrinsic_op.define('nir_intrinsic_vote_any', 725)
nir_intrinsic_vote_feq = nir_intrinsic_op.define('nir_intrinsic_vote_feq', 726)
nir_intrinsic_vote_ieq = nir_intrinsic_op.define('nir_intrinsic_vote_ieq', 727)
nir_intrinsic_vulkan_resource_index = nir_intrinsic_op.define('nir_intrinsic_vulkan_resource_index', 728)
nir_intrinsic_vulkan_resource_reindex = nir_intrinsic_op.define('nir_intrinsic_vulkan_resource_reindex', 729)
nir_intrinsic_write_invocation_amd = nir_intrinsic_op.define('nir_intrinsic_write_invocation_amd', 730)
nir_intrinsic_xfb_counter_sub_gfx11_amd = nir_intrinsic_op.define('nir_intrinsic_xfb_counter_sub_gfx11_amd', 731)
nir_last_intrinsic = nir_intrinsic_op.define('nir_last_intrinsic', 731)
nir_num_intrinsics = nir_intrinsic_op.define('nir_num_intrinsics', 732)

struct_nir_intrinsic_instr._fields_ = [
  ('instr', nir_instr),
  ('intrinsic', nir_intrinsic_op),
  ('def', nir_def),
  ('num_components', uint8_t),
  ('const_index', (ctypes.c_int32 * 8)),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('src', (nir_src * 0)),
]
nir_intrinsic_instr = struct_nir_intrinsic_instr
nir_memory_semantics = CEnum(ctypes.c_uint32)
NIR_MEMORY_ACQUIRE = nir_memory_semantics.define('NIR_MEMORY_ACQUIRE', 1)
NIR_MEMORY_RELEASE = nir_memory_semantics.define('NIR_MEMORY_RELEASE', 2)
NIR_MEMORY_ACQ_REL = nir_memory_semantics.define('NIR_MEMORY_ACQ_REL', 3)
NIR_MEMORY_MAKE_AVAILABLE = nir_memory_semantics.define('NIR_MEMORY_MAKE_AVAILABLE', 4)
NIR_MEMORY_MAKE_VISIBLE = nir_memory_semantics.define('NIR_MEMORY_MAKE_VISIBLE', 8)

nir_intrinsic_semantic_flag = CEnum(ctypes.c_uint32)
NIR_INTRINSIC_CAN_ELIMINATE = nir_intrinsic_semantic_flag.define('NIR_INTRINSIC_CAN_ELIMINATE', 1)
NIR_INTRINSIC_CAN_REORDER = nir_intrinsic_semantic_flag.define('NIR_INTRINSIC_CAN_REORDER', 2)
NIR_INTRINSIC_SUBGROUP = nir_intrinsic_semantic_flag.define('NIR_INTRINSIC_SUBGROUP', 4)
NIR_INTRINSIC_QUADGROUP = nir_intrinsic_semantic_flag.define('NIR_INTRINSIC_QUADGROUP', 8)

class struct_nir_io_semantics(Struct): pass
struct_nir_io_semantics._fields_ = [
  ('location', ctypes.c_uint32,7),
  ('num_slots', ctypes.c_uint32,6),
  ('dual_source_blend_index', ctypes.c_uint32,1),
  ('fb_fetch_output', ctypes.c_uint32,1),
  ('fb_fetch_output_coherent', ctypes.c_uint32,1),
  ('gs_streams', ctypes.c_uint32,8),
  ('medium_precision', ctypes.c_uint32,1),
  ('per_view', ctypes.c_uint32,1),
  ('high_16bits', ctypes.c_uint32,1),
  ('high_dvec2', ctypes.c_uint32,1),
  ('no_varying', ctypes.c_uint32,1),
  ('no_sysval_output', ctypes.c_uint32,1),
  ('interp_explicit_strict', ctypes.c_uint32,1),
  ('_pad', ctypes.c_uint32,1),
]
nir_io_semantics = struct_nir_io_semantics
class struct_nir_io_xfb(Struct): pass
class struct_nir_io_xfb_out(Struct): pass
struct_nir_io_xfb_out._fields_ = [
  ('num_components', uint8_t,4),
  ('buffer', uint8_t,4),
  ('offset', uint8_t),
]
struct_nir_io_xfb._fields_ = [
  ('out', (struct_nir_io_xfb_out * 2)),
]
nir_io_xfb = struct_nir_io_xfb
try: (nir_instr_xfb_write_mask:=dll.nir_instr_xfb_write_mask).restype, nir_instr_xfb_write_mask.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

class struct_nir_intrinsic_info(Struct): pass
struct_nir_intrinsic_info._fields_ = [
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('num_srcs', uint8_t),
  ('src_components', (int8_t * 11)),
  ('has_dest', ctypes.c_bool),
  ('dest_components', uint8_t),
  ('dest_bit_sizes', uint8_t),
  ('bit_size_src', int8_t),
  ('num_indices', uint8_t),
  ('indices', (uint8_t * 8)),
  ('index_map', (uint8_t * 75)),
  ('flags', nir_intrinsic_semantic_flag),
]
nir_intrinsic_info = struct_nir_intrinsic_info
try: nir_intrinsic_infos = (nir_intrinsic_info * 732).in_dll(dll, 'nir_intrinsic_infos')
except (ValueError,AttributeError): pass
try: (nir_intrinsic_src_components:=dll.nir_intrinsic_src_components).restype, nir_intrinsic_src_components.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_intrinsic_dest_components:=dll.nir_intrinsic_dest_components).restype, nir_intrinsic_dest_components.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_intrinsic_instr_src_type:=dll.nir_intrinsic_instr_src_type).restype, nir_intrinsic_instr_src_type.argtypes = nir_alu_type, [ctypes.POINTER(nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_intrinsic_instr_dest_type:=dll.nir_intrinsic_instr_dest_type).restype, nir_intrinsic_instr_dest_type.argtypes = nir_alu_type, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_intrinsic_copy_const_indices:=dll.nir_intrinsic_copy_const_indices).restype, nir_intrinsic_copy_const_indices.argtypes = None, [ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_image_intrinsic_coord_components:=dll.nir_image_intrinsic_coord_components).restype, nir_image_intrinsic_coord_components.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_rewrite_image_intrinsic:=dll.nir_rewrite_image_intrinsic).restype, nir_rewrite_image_intrinsic.argtypes = None, [ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(nir_def), ctypes.c_bool]
except AttributeError: pass

try: (nir_intrinsic_can_reorder:=dll.nir_intrinsic_can_reorder).restype, nir_intrinsic_can_reorder.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_intrinsic_writes_external_memory:=dll.nir_intrinsic_writes_external_memory).restype, nir_intrinsic_writes_external_memory.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

enum_nir_tex_src_type = CEnum(ctypes.c_uint32)
nir_tex_src_coord = enum_nir_tex_src_type.define('nir_tex_src_coord', 0)
nir_tex_src_projector = enum_nir_tex_src_type.define('nir_tex_src_projector', 1)
nir_tex_src_comparator = enum_nir_tex_src_type.define('nir_tex_src_comparator', 2)
nir_tex_src_offset = enum_nir_tex_src_type.define('nir_tex_src_offset', 3)
nir_tex_src_bias = enum_nir_tex_src_type.define('nir_tex_src_bias', 4)
nir_tex_src_lod = enum_nir_tex_src_type.define('nir_tex_src_lod', 5)
nir_tex_src_min_lod = enum_nir_tex_src_type.define('nir_tex_src_min_lod', 6)
nir_tex_src_lod_bias_min_agx = enum_nir_tex_src_type.define('nir_tex_src_lod_bias_min_agx', 7)
nir_tex_src_ms_index = enum_nir_tex_src_type.define('nir_tex_src_ms_index', 8)
nir_tex_src_ms_mcs_intel = enum_nir_tex_src_type.define('nir_tex_src_ms_mcs_intel', 9)
nir_tex_src_ddx = enum_nir_tex_src_type.define('nir_tex_src_ddx', 10)
nir_tex_src_ddy = enum_nir_tex_src_type.define('nir_tex_src_ddy', 11)
nir_tex_src_texture_deref = enum_nir_tex_src_type.define('nir_tex_src_texture_deref', 12)
nir_tex_src_sampler_deref = enum_nir_tex_src_type.define('nir_tex_src_sampler_deref', 13)
nir_tex_src_texture_offset = enum_nir_tex_src_type.define('nir_tex_src_texture_offset', 14)
nir_tex_src_sampler_offset = enum_nir_tex_src_type.define('nir_tex_src_sampler_offset', 15)
nir_tex_src_texture_handle = enum_nir_tex_src_type.define('nir_tex_src_texture_handle', 16)
nir_tex_src_sampler_handle = enum_nir_tex_src_type.define('nir_tex_src_sampler_handle', 17)
nir_tex_src_sampler_deref_intrinsic = enum_nir_tex_src_type.define('nir_tex_src_sampler_deref_intrinsic', 18)
nir_tex_src_texture_deref_intrinsic = enum_nir_tex_src_type.define('nir_tex_src_texture_deref_intrinsic', 19)
nir_tex_src_plane = enum_nir_tex_src_type.define('nir_tex_src_plane', 20)
nir_tex_src_backend1 = enum_nir_tex_src_type.define('nir_tex_src_backend1', 21)
nir_tex_src_backend2 = enum_nir_tex_src_type.define('nir_tex_src_backend2', 22)
nir_num_tex_src_types = enum_nir_tex_src_type.define('nir_num_tex_src_types', 23)

nir_tex_src_type = enum_nir_tex_src_type
class struct_nir_tex_src(Struct): pass
struct_nir_tex_src._fields_ = [
  ('src', nir_src),
  ('src_type', nir_tex_src_type),
]
nir_tex_src = struct_nir_tex_src
enum_nir_texop = CEnum(ctypes.c_uint32)
nir_texop_tex = enum_nir_texop.define('nir_texop_tex', 0)
nir_texop_txb = enum_nir_texop.define('nir_texop_txb', 1)
nir_texop_txl = enum_nir_texop.define('nir_texop_txl', 2)
nir_texop_txd = enum_nir_texop.define('nir_texop_txd', 3)
nir_texop_txf = enum_nir_texop.define('nir_texop_txf', 4)
nir_texop_txf_ms = enum_nir_texop.define('nir_texop_txf_ms', 5)
nir_texop_txf_ms_fb = enum_nir_texop.define('nir_texop_txf_ms_fb', 6)
nir_texop_txf_ms_mcs_intel = enum_nir_texop.define('nir_texop_txf_ms_mcs_intel', 7)
nir_texop_txs = enum_nir_texop.define('nir_texop_txs', 8)
nir_texop_lod = enum_nir_texop.define('nir_texop_lod', 9)
nir_texop_tg4 = enum_nir_texop.define('nir_texop_tg4', 10)
nir_texop_query_levels = enum_nir_texop.define('nir_texop_query_levels', 11)
nir_texop_texture_samples = enum_nir_texop.define('nir_texop_texture_samples', 12)
nir_texop_samples_identical = enum_nir_texop.define('nir_texop_samples_identical', 13)
nir_texop_tex_prefetch = enum_nir_texop.define('nir_texop_tex_prefetch', 14)
nir_texop_lod_bias = enum_nir_texop.define('nir_texop_lod_bias', 15)
nir_texop_fragment_fetch_amd = enum_nir_texop.define('nir_texop_fragment_fetch_amd', 16)
nir_texop_fragment_mask_fetch_amd = enum_nir_texop.define('nir_texop_fragment_mask_fetch_amd', 17)
nir_texop_descriptor_amd = enum_nir_texop.define('nir_texop_descriptor_amd', 18)
nir_texop_sampler_descriptor_amd = enum_nir_texop.define('nir_texop_sampler_descriptor_amd', 19)
nir_texop_image_min_lod_agx = enum_nir_texop.define('nir_texop_image_min_lod_agx', 20)
nir_texop_has_custom_border_color_agx = enum_nir_texop.define('nir_texop_has_custom_border_color_agx', 21)
nir_texop_custom_border_color_agx = enum_nir_texop.define('nir_texop_custom_border_color_agx', 22)
nir_texop_hdr_dim_nv = enum_nir_texop.define('nir_texop_hdr_dim_nv', 23)
nir_texop_tex_type_nv = enum_nir_texop.define('nir_texop_tex_type_nv', 24)

nir_texop = enum_nir_texop
class struct_nir_tex_instr(Struct): pass
enum_glsl_sampler_dim = CEnum(ctypes.c_uint32)
GLSL_SAMPLER_DIM_1D = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_1D', 0)
GLSL_SAMPLER_DIM_2D = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_2D', 1)
GLSL_SAMPLER_DIM_3D = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_3D', 2)
GLSL_SAMPLER_DIM_CUBE = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_CUBE', 3)
GLSL_SAMPLER_DIM_RECT = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_RECT', 4)
GLSL_SAMPLER_DIM_BUF = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_BUF', 5)
GLSL_SAMPLER_DIM_EXTERNAL = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_EXTERNAL', 6)
GLSL_SAMPLER_DIM_MS = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_MS', 7)
GLSL_SAMPLER_DIM_SUBPASS = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_SUBPASS', 8)
GLSL_SAMPLER_DIM_SUBPASS_MS = enum_glsl_sampler_dim.define('GLSL_SAMPLER_DIM_SUBPASS_MS', 9)

struct_nir_tex_instr._fields_ = [
  ('instr', nir_instr),
  ('sampler_dim', enum_glsl_sampler_dim),
  ('dest_type', nir_alu_type),
  ('op', nir_texop),
  ('def', nir_def),
  ('src', ctypes.POINTER(nir_tex_src)),
  ('num_srcs', ctypes.c_uint32),
  ('coord_components', ctypes.c_uint32),
  ('is_array', ctypes.c_bool),
  ('is_shadow', ctypes.c_bool),
  ('is_new_style_shadow', ctypes.c_bool),
  ('is_sparse', ctypes.c_bool),
  ('component', ctypes.c_uint32,2),
  ('array_is_lowered_cube', ctypes.c_uint32,1),
  ('is_gather_implicit_lod', ctypes.c_uint32,1),
  ('skip_helpers', ctypes.c_uint32,1),
  ('tg4_offsets', ((int8_t * 2) * 4)),
  ('texture_non_uniform', ctypes.c_bool),
  ('sampler_non_uniform', ctypes.c_bool),
  ('offset_non_uniform', ctypes.c_bool),
  ('texture_index', ctypes.c_uint32),
  ('sampler_index', ctypes.c_uint32),
  ('backend_flags', uint32_t),
]
nir_tex_instr = struct_nir_tex_instr
try: (nir_tex_instr_need_sampler:=dll.nir_tex_instr_need_sampler).restype, nir_tex_instr_need_sampler.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (nir_tex_instr_result_size:=dll.nir_tex_instr_result_size).restype, nir_tex_instr_result_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (nir_tex_instr_is_query:=dll.nir_tex_instr_is_query).restype, nir_tex_instr_is_query.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (nir_tex_instr_has_implicit_derivative:=dll.nir_tex_instr_has_implicit_derivative).restype, nir_tex_instr_has_implicit_derivative.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (nir_tex_instr_src_type:=dll.nir_tex_instr_src_type).restype, nir_tex_instr_src_type.argtypes = nir_alu_type, [ctypes.POINTER(nir_tex_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_tex_instr_src_size:=dll.nir_tex_instr_src_size).restype, nir_tex_instr_src_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_tex_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_tex_instr_add_src:=dll.nir_tex_instr_add_src).restype, nir_tex_instr_add_src.argtypes = None, [ctypes.POINTER(nir_tex_instr), nir_tex_src_type, ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_tex_instr_remove_src:=dll.nir_tex_instr_remove_src).restype, nir_tex_instr_remove_src.argtypes = None, [ctypes.POINTER(nir_tex_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_tex_instr_has_explicit_tg4_offsets:=dll.nir_tex_instr_has_explicit_tg4_offsets).restype, nir_tex_instr_has_explicit_tg4_offsets.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

class struct_nir_load_const_instr(Struct): pass
struct_nir_load_const_instr._fields_ = [
  ('instr', nir_instr),
  ('def', nir_def),
  ('value', (nir_const_value * 0)),
]
nir_load_const_instr = struct_nir_load_const_instr
nir_jump_type = CEnum(ctypes.c_uint32)
nir_jump_return = nir_jump_type.define('nir_jump_return', 0)
nir_jump_halt = nir_jump_type.define('nir_jump_halt', 1)
nir_jump_break = nir_jump_type.define('nir_jump_break', 2)
nir_jump_continue = nir_jump_type.define('nir_jump_continue', 3)
nir_jump_goto = nir_jump_type.define('nir_jump_goto', 4)
nir_jump_goto_if = nir_jump_type.define('nir_jump_goto_if', 5)

class struct_nir_jump_instr(Struct): pass
struct_nir_jump_instr._fields_ = [
  ('instr', nir_instr),
  ('type', nir_jump_type),
  ('condition', nir_src),
  ('target', ctypes.POINTER(nir_block)),
  ('else_target', ctypes.POINTER(nir_block)),
]
nir_jump_instr = struct_nir_jump_instr
class struct_nir_undef_instr(Struct): pass
struct_nir_undef_instr._fields_ = [
  ('instr', nir_instr),
  ('def', nir_def),
]
nir_undef_instr = struct_nir_undef_instr
class struct_nir_phi_src(Struct): pass
struct_nir_phi_src._fields_ = [
  ('node', struct_exec_node),
  ('pred', ctypes.POINTER(nir_block)),
  ('src', nir_src),
]
nir_phi_src = struct_nir_phi_src
class struct_nir_phi_instr(Struct): pass
struct_nir_phi_instr._fields_ = [
  ('instr', nir_instr),
  ('srcs', struct_exec_list),
  ('def', nir_def),
]
nir_phi_instr = struct_nir_phi_instr
class struct_nir_parallel_copy_entry(Struct): pass
class struct_nir_parallel_copy_entry_dest(ctypes.Union): pass
struct_nir_parallel_copy_entry_dest._fields_ = [
  ('def', nir_def),
  ('reg', nir_src),
]
struct_nir_parallel_copy_entry._fields_ = [
  ('node', struct_exec_node),
  ('src_is_reg', ctypes.c_bool),
  ('dest_is_reg', ctypes.c_bool),
  ('src', nir_src),
  ('dest', struct_nir_parallel_copy_entry_dest),
]
nir_parallel_copy_entry = struct_nir_parallel_copy_entry
class struct_nir_parallel_copy_instr(Struct): pass
struct_nir_parallel_copy_instr._fields_ = [
  ('instr', nir_instr),
  ('entries', struct_exec_list),
]
nir_parallel_copy_instr = struct_nir_parallel_copy_instr
class struct_nir_instr_debug_info(Struct): pass
struct_nir_instr_debug_info._fields_ = [
  ('filename', ctypes.POINTER(ctypes.c_char)),
  ('line', uint32_t),
  ('column', uint32_t),
  ('spirv_offset', uint32_t),
  ('nir_line', uint32_t),
  ('variable_name', ctypes.POINTER(ctypes.c_char)),
  ('instr', nir_instr),
]
nir_instr_debug_info = struct_nir_instr_debug_info
class struct_nir_scalar(Struct): pass
struct_nir_scalar._fields_ = [
  ('def', ctypes.POINTER(nir_def)),
  ('comp', ctypes.c_uint32),
]
nir_scalar = struct_nir_scalar
try: (nir_scalar_chase_movs:=dll.nir_scalar_chase_movs).restype, nir_scalar_chase_movs.argtypes = nir_scalar, [nir_scalar]
except AttributeError: pass

class struct_nir_binding(Struct): pass
struct_nir_binding._fields_ = [
  ('success', ctypes.c_bool),
  ('var', ctypes.POINTER(nir_variable)),
  ('desc_set', ctypes.c_uint32),
  ('binding', ctypes.c_uint32),
  ('num_indices', ctypes.c_uint32),
  ('indices', (nir_src * 4)),
  ('read_first_invocation', ctypes.c_bool),
]
nir_binding = struct_nir_binding
try: (nir_chase_binding:=dll.nir_chase_binding).restype, nir_chase_binding.argtypes = nir_binding, [nir_src]
except AttributeError: pass

try: (nir_get_binding_variable:=dll.nir_get_binding_variable).restype, nir_get_binding_variable.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_binding]
except AttributeError: pass

try: (nir_block_contains_work:=dll.nir_block_contains_work).restype, nir_block_contains_work.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block)]
except AttributeError: pass

nir_selection_control = CEnum(ctypes.c_uint32)
nir_selection_control_none = nir_selection_control.define('nir_selection_control_none', 0)
nir_selection_control_flatten = nir_selection_control.define('nir_selection_control_flatten', 1)
nir_selection_control_dont_flatten = nir_selection_control.define('nir_selection_control_dont_flatten', 2)
nir_selection_control_divergent_always_taken = nir_selection_control.define('nir_selection_control_divergent_always_taken', 3)

class struct_nir_if(Struct): pass
struct_nir_if._fields_ = [
  ('cf_node', nir_cf_node),
  ('condition', nir_src),
  ('control', nir_selection_control),
  ('then_list', struct_exec_list),
  ('else_list', struct_exec_list),
]
nir_if = struct_nir_if
class struct_nir_loop_terminator(Struct): pass
struct_nir_loop_terminator._fields_ = [
  ('nif', ctypes.POINTER(nir_if)),
  ('conditional_instr', ctypes.POINTER(nir_instr)),
  ('break_block', ctypes.POINTER(nir_block)),
  ('continue_from_block', ctypes.POINTER(nir_block)),
  ('continue_from_then', ctypes.c_bool),
  ('induction_rhs', ctypes.c_bool),
  ('exact_trip_count_unknown', ctypes.c_bool),
  ('loop_terminator_link', struct_list_head),
]
nir_loop_terminator = struct_nir_loop_terminator
class struct_nir_loop_induction_variable(Struct): pass
struct_nir_loop_induction_variable._fields_ = [
  ('basis', ctypes.POINTER(nir_def)),
  ('def', ctypes.POINTER(nir_def)),
  ('init_src', ctypes.POINTER(nir_src)),
  ('update_src', ctypes.POINTER(nir_alu_src)),
]
nir_loop_induction_variable = struct_nir_loop_induction_variable
class struct_nir_loop_info(Struct): pass
class struct_hash_table(Struct): pass
class struct_hash_entry(Struct): pass
struct_hash_entry._fields_ = [
  ('hash', uint32_t),
  ('key', ctypes.c_void_p),
  ('data', ctypes.c_void_p),
]
struct_hash_table._fields_ = [
  ('table', ctypes.POINTER(struct_hash_entry)),
  ('key_hash_function', ctypes.CFUNCTYPE(uint32_t, ctypes.c_void_p)),
  ('key_equals_function', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)),
  ('deleted_key', ctypes.c_void_p),
  ('size', uint32_t),
  ('rehash', uint32_t),
  ('size_magic', uint64_t),
  ('rehash_magic', uint64_t),
  ('max_entries', uint32_t),
  ('size_index', uint32_t),
  ('entries', uint32_t),
  ('deleted_entries', uint32_t),
]
struct_nir_loop_info._fields_ = [
  ('instr_cost', ctypes.c_uint32),
  ('has_soft_fp64', ctypes.c_bool),
  ('guessed_trip_count', ctypes.c_uint32),
  ('max_trip_count', ctypes.c_uint32),
  ('exact_trip_count_known', ctypes.c_bool),
  ('force_unroll', ctypes.c_bool),
  ('complex_loop', ctypes.c_bool),
  ('limiting_terminator', ctypes.POINTER(nir_loop_terminator)),
  ('loop_terminator_list', struct_list_head),
  ('induction_vars', ctypes.POINTER(struct_hash_table)),
]
nir_loop_info = struct_nir_loop_info
nir_loop_control = CEnum(ctypes.c_uint32)
nir_loop_control_none = nir_loop_control.define('nir_loop_control_none', 0)
nir_loop_control_unroll = nir_loop_control.define('nir_loop_control_unroll', 1)
nir_loop_control_dont_unroll = nir_loop_control.define('nir_loop_control_dont_unroll', 2)

class struct_nir_loop(Struct): pass
struct_nir_loop._fields_ = [
  ('cf_node', nir_cf_node),
  ('body', struct_exec_list),
  ('continue_list', struct_exec_list),
  ('info', ctypes.POINTER(nir_loop_info)),
  ('control', nir_loop_control),
  ('partially_unrolled', ctypes.c_bool),
  ('divergent_continue', ctypes.c_bool),
  ('divergent_break', ctypes.c_bool),
]
nir_loop = struct_nir_loop
class const_struct_nir_intrinsic_instr(Struct): pass
const_struct_nir_intrinsic_instr._fields_ = [
  ('instr', nir_instr),
  ('intrinsic', nir_intrinsic_op),
  ('def', nir_def),
  ('num_components', uint8_t),
  ('const_index', (ctypes.c_int32 * 8)),
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('src', (nir_src * 0)),
]
nir_intrin_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(const_struct_nir_intrinsic_instr), ctypes.c_void_p)
nir_vectorize_cb = ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.POINTER(const_struct_nir_instr), ctypes.c_void_p)
try: (nir_remove_non_entrypoints:=dll.nir_remove_non_entrypoints).restype, nir_remove_non_entrypoints.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_non_exported:=dll.nir_remove_non_exported).restype, nir_remove_non_exported.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_entrypoints:=dll.nir_remove_entrypoints).restype, nir_remove_entrypoints.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_fixup_is_exported:=dll.nir_fixup_is_exported).restype, nir_fixup_is_exported.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

shader_info = struct_shader_info
try: (nir_shader_create:=dll.nir_shader_create).restype, nir_shader_create.argtypes = ctypes.POINTER(nir_shader), [ctypes.c_void_p, gl_shader_stage, ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(shader_info)]
except AttributeError: pass

try: (nir_shader_add_variable:=dll.nir_shader_add_variable).restype, nir_shader_add_variable.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_variable)]
except AttributeError: pass

try: (nir_variable_create:=dll.nir_variable_create).restype, nir_variable_create.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_local_variable_create:=dll.nir_local_variable_create).restype, nir_local_variable_create.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_function_impl), ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_state_variable_create:=dll.nir_state_variable_create).restype, nir_state_variable_create.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char), (gl_state_index16 * 4)]
except AttributeError: pass

try: (nir_get_variable_with_location:=dll.nir_get_variable_with_location).restype, nir_get_variable_with_location.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_int32, ctypes.POINTER(struct_glsl_type)]
except AttributeError: pass

try: (nir_create_variable_with_location:=dll.nir_create_variable_with_location).restype, nir_create_variable_with_location.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_int32, ctypes.POINTER(struct_glsl_type)]
except AttributeError: pass

try: (nir_find_variable_with_location:=dll.nir_find_variable_with_location).restype, nir_find_variable_with_location.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_uint32]
except AttributeError: pass

try: (nir_find_variable_with_driver_location:=dll.nir_find_variable_with_driver_location).restype, nir_find_variable_with_driver_location.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_uint32]
except AttributeError: pass

try: (nir_find_state_variable:=dll.nir_find_state_variable).restype, nir_find_state_variable.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), (gl_state_index16 * 4)]
except AttributeError: pass

try: (nir_find_sampler_variable_with_tex_index:=dll.nir_find_sampler_variable_with_tex_index).restype, nir_find_sampler_variable_with_tex_index.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_sort_variables_with_modes:=dll.nir_sort_variables_with_modes).restype, nir_sort_variables_with_modes.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(nir_variable), ctypes.POINTER(nir_variable)), nir_variable_mode]
except AttributeError: pass

try: (nir_function_create:=dll.nir_function_create).restype, nir_function_create.argtypes = ctypes.POINTER(nir_function), [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_function_impl_create:=dll.nir_function_impl_create).restype, nir_function_impl_create.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_function)]
except AttributeError: pass

try: (nir_function_impl_create_bare:=dll.nir_function_impl_create_bare).restype, nir_function_impl_create_bare.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_block_create:=dll.nir_block_create).restype, nir_block_create.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_if_create:=dll.nir_if_create).restype, nir_if_create.argtypes = ctypes.POINTER(nir_if), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_loop_create:=dll.nir_loop_create).restype, nir_loop_create.argtypes = ctypes.POINTER(nir_loop), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_cf_node_get_function:=dll.nir_cf_node_get_function).restype, nir_cf_node_get_function.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_metadata_require:=dll.nir_metadata_require).restype, nir_metadata_require.argtypes = None, [ctypes.POINTER(nir_function_impl), nir_metadata]
except AttributeError: pass

try: (nir_shader_preserve_all_metadata:=dll.nir_shader_preserve_all_metadata).restype, nir_shader_preserve_all_metadata.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_metadata_invalidate:=dll.nir_metadata_invalidate).restype, nir_metadata_invalidate.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_progress:=dll.nir_progress).restype, nir_progress.argtypes = ctypes.c_bool, [ctypes.c_bool, ctypes.POINTER(nir_function_impl), nir_metadata]
except AttributeError: pass

try: (nir_alu_instr_create:=dll.nir_alu_instr_create).restype, nir_alu_instr_create.argtypes = ctypes.POINTER(nir_alu_instr), [ctypes.POINTER(nir_shader), nir_op]
except AttributeError: pass

try: (nir_deref_instr_create:=dll.nir_deref_instr_create).restype, nir_deref_instr_create.argtypes = ctypes.POINTER(nir_deref_instr), [ctypes.POINTER(nir_shader), nir_deref_type]
except AttributeError: pass

try: (nir_jump_instr_create:=dll.nir_jump_instr_create).restype, nir_jump_instr_create.argtypes = ctypes.POINTER(nir_jump_instr), [ctypes.POINTER(nir_shader), nir_jump_type]
except AttributeError: pass

try: (nir_load_const_instr_create:=dll.nir_load_const_instr_create).restype, nir_load_const_instr_create.argtypes = ctypes.POINTER(nir_load_const_instr), [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_intrinsic_instr_create:=dll.nir_intrinsic_instr_create).restype, nir_intrinsic_instr_create.argtypes = ctypes.POINTER(nir_intrinsic_instr), [ctypes.POINTER(nir_shader), nir_intrinsic_op]
except AttributeError: pass

try: (nir_call_instr_create:=dll.nir_call_instr_create).restype, nir_call_instr_create.argtypes = ctypes.POINTER(nir_call_instr), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function)]
except AttributeError: pass

try: (nir_tex_instr_create:=dll.nir_tex_instr_create).restype, nir_tex_instr_create.argtypes = ctypes.POINTER(nir_tex_instr), [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_phi_instr_create:=dll.nir_phi_instr_create).restype, nir_phi_instr_create.argtypes = ctypes.POINTER(nir_phi_instr), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_phi_instr_add_src:=dll.nir_phi_instr_add_src).restype, nir_phi_instr_add_src.argtypes = ctypes.POINTER(nir_phi_src), [ctypes.POINTER(nir_phi_instr), ctypes.POINTER(nir_block), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_parallel_copy_instr_create:=dll.nir_parallel_copy_instr_create).restype, nir_parallel_copy_instr_create.argtypes = ctypes.POINTER(nir_parallel_copy_instr), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_undef_instr_create:=dll.nir_undef_instr_create).restype, nir_undef_instr_create.argtypes = ctypes.POINTER(nir_undef_instr), [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_alu_binop_identity:=dll.nir_alu_binop_identity).restype, nir_alu_binop_identity.argtypes = nir_const_value, [nir_op, ctypes.c_uint32]
except AttributeError: pass

nir_cursor_option = CEnum(ctypes.c_uint32)
nir_cursor_before_block = nir_cursor_option.define('nir_cursor_before_block', 0)
nir_cursor_after_block = nir_cursor_option.define('nir_cursor_after_block', 1)
nir_cursor_before_instr = nir_cursor_option.define('nir_cursor_before_instr', 2)
nir_cursor_after_instr = nir_cursor_option.define('nir_cursor_after_instr', 3)

class struct_nir_cursor(Struct): pass
class struct_nir_cursor_0(ctypes.Union): pass
struct_nir_cursor_0._fields_ = [
  ('block', ctypes.POINTER(nir_block)),
  ('instr', ctypes.POINTER(nir_instr)),
]
struct_nir_cursor._anonymous_ = ['_0']
struct_nir_cursor._fields_ = [
  ('option', nir_cursor_option),
  ('_0', struct_nir_cursor_0),
]
nir_cursor = struct_nir_cursor
try: (nir_cursors_equal:=dll.nir_cursors_equal).restype, nir_cursors_equal.argtypes = ctypes.c_bool, [nir_cursor, nir_cursor]
except AttributeError: pass

try: (nir_instr_insert:=dll.nir_instr_insert).restype, nir_instr_insert.argtypes = None, [nir_cursor, ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_move:=dll.nir_instr_move).restype, nir_instr_move.argtypes = ctypes.c_bool, [nir_cursor, ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_remove_v:=dll.nir_instr_remove_v).restype, nir_instr_remove_v.argtypes = None, [ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_free:=dll.nir_instr_free).restype, nir_instr_free.argtypes = None, [ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_free_list:=dll.nir_instr_free_list).restype, nir_instr_free_list.argtypes = None, [ctypes.POINTER(struct_exec_list)]
except AttributeError: pass

try: (nir_instr_free_and_dce:=dll.nir_instr_free_and_dce).restype, nir_instr_free_and_dce.argtypes = nir_cursor, [ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_def:=dll.nir_instr_def).restype, nir_instr_def.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_instr)]
except AttributeError: pass

nir_foreach_def_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_def), ctypes.c_void_p)
nir_foreach_src_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_src), ctypes.c_void_p)
try: (nir_foreach_phi_src_leaving_block:=dll.nir_foreach_phi_src_leaving_block).restype, nir_foreach_phi_src_leaving_block.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block), nir_foreach_src_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_src_as_const_value:=dll.nir_src_as_const_value).restype, nir_src_as_const_value.argtypes = ctypes.POINTER(nir_const_value), [nir_src]
except AttributeError: pass

try: (nir_src_as_string:=dll.nir_src_as_string).restype, nir_src_as_string.argtypes = ctypes.POINTER(ctypes.c_char), [nir_src]
except AttributeError: pass

try: (nir_src_is_always_uniform:=dll.nir_src_is_always_uniform).restype, nir_src_is_always_uniform.argtypes = ctypes.c_bool, [nir_src]
except AttributeError: pass

try: (nir_srcs_equal:=dll.nir_srcs_equal).restype, nir_srcs_equal.argtypes = ctypes.c_bool, [nir_src, nir_src]
except AttributeError: pass

try: (nir_instrs_equal:=dll.nir_instrs_equal).restype, nir_instrs_equal.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_src_get_block:=dll.nir_src_get_block).restype, nir_src_get_block.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_src)]
except AttributeError: pass

try: (nir_instr_init_src:=dll.nir_instr_init_src).restype, nir_instr_init_src.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_src), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_instr_clear_src:=dll.nir_instr_clear_src).restype, nir_instr_clear_src.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_src)]
except AttributeError: pass

try: (nir_instr_move_src:=dll.nir_instr_move_src).restype, nir_instr_move_src.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_src), ctypes.POINTER(nir_src)]
except AttributeError: pass

try: (nir_instr_is_before:=dll.nir_instr_is_before).restype, nir_instr_is_before.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_def_init:=dll.nir_def_init).restype, nir_def_init.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(nir_def), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_def_rewrite_uses:=dll.nir_def_rewrite_uses).restype, nir_def_rewrite_uses.argtypes = None, [ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_def_rewrite_uses_src:=dll.nir_def_rewrite_uses_src).restype, nir_def_rewrite_uses_src.argtypes = None, [ctypes.POINTER(nir_def), nir_src]
except AttributeError: pass

try: (nir_def_rewrite_uses_after:=dll.nir_def_rewrite_uses_after).restype, nir_def_rewrite_uses_after.argtypes = None, [ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_src_components_read:=dll.nir_src_components_read).restype, nir_src_components_read.argtypes = nir_component_mask_t, [ctypes.POINTER(nir_src)]
except AttributeError: pass

try: (nir_def_components_read:=dll.nir_def_components_read).restype, nir_def_components_read.argtypes = nir_component_mask_t, [ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_def_all_uses_are_fsat:=dll.nir_def_all_uses_are_fsat).restype, nir_def_all_uses_are_fsat.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_def_all_uses_ignore_sign_bit:=dll.nir_def_all_uses_ignore_sign_bit).restype, nir_def_all_uses_ignore_sign_bit.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_sort_unstructured_blocks:=dll.nir_sort_unstructured_blocks).restype, nir_sort_unstructured_blocks.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_block_unstructured_next:=dll.nir_block_unstructured_next).restype, nir_block_unstructured_next.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_unstructured_start_block:=dll.nir_unstructured_start_block).restype, nir_unstructured_start_block.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_block_cf_tree_next:=dll.nir_block_cf_tree_next).restype, nir_block_cf_tree_next.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_cf_tree_prev:=dll.nir_block_cf_tree_prev).restype, nir_block_cf_tree_prev.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_cf_node_cf_tree_first:=dll.nir_cf_node_cf_tree_first).restype, nir_cf_node_cf_tree_first.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_cf_node_cf_tree_last:=dll.nir_cf_node_cf_tree_last).restype, nir_cf_node_cf_tree_last.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_cf_node_cf_tree_next:=dll.nir_cf_node_cf_tree_next).restype, nir_cf_node_cf_tree_next.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_cf_node_cf_tree_prev:=dll.nir_cf_node_cf_tree_prev).restype, nir_cf_node_cf_tree_prev.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_block_get_following_if:=dll.nir_block_get_following_if).restype, nir_block_get_following_if.argtypes = ctypes.POINTER(nir_if), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_get_following_loop:=dll.nir_block_get_following_loop).restype, nir_block_get_following_loop.argtypes = ctypes.POINTER(nir_loop), [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_get_predecessors_sorted:=dll.nir_block_get_predecessors_sorted).restype, nir_block_get_predecessors_sorted.argtypes = ctypes.POINTER(ctypes.POINTER(nir_block)), [ctypes.POINTER(nir_block), ctypes.c_void_p]
except AttributeError: pass

try: (nir_index_ssa_defs:=dll.nir_index_ssa_defs).restype, nir_index_ssa_defs.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_index_instrs:=dll.nir_index_instrs).restype, nir_index_instrs.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_index_blocks:=dll.nir_index_blocks).restype, nir_index_blocks.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_shader_clear_pass_flags:=dll.nir_shader_clear_pass_flags).restype, nir_shader_clear_pass_flags.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_index_vars:=dll.nir_shader_index_vars).restype, nir_shader_index_vars.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_function_impl_index_vars:=dll.nir_function_impl_index_vars).restype, nir_function_impl_index_vars.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

class struct__IO_FILE(Struct): pass
FILE = struct__IO_FILE
class struct__IO_marker(Struct): pass
__off_t = ctypes.c_int64
_IO_lock_t = None
__off64_t = ctypes.c_int64
class struct__IO_codecvt(Struct): pass
class struct__IO_wide_data(Struct): pass
size_t = ctypes.c_uint64
struct__IO_FILE._fields_ = [
  ('_flags', ctypes.c_int32),
  ('_IO_read_ptr', ctypes.POINTER(ctypes.c_char)),
  ('_IO_read_end', ctypes.POINTER(ctypes.c_char)),
  ('_IO_read_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_write_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_write_ptr', ctypes.POINTER(ctypes.c_char)),
  ('_IO_write_end', ctypes.POINTER(ctypes.c_char)),
  ('_IO_buf_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_buf_end', ctypes.POINTER(ctypes.c_char)),
  ('_IO_save_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_backup_base', ctypes.POINTER(ctypes.c_char)),
  ('_IO_save_end', ctypes.POINTER(ctypes.c_char)),
  ('_markers', ctypes.POINTER(struct__IO_marker)),
  ('_chain', ctypes.POINTER(struct__IO_FILE)),
  ('_fileno', ctypes.c_int32),
  ('_flags2', ctypes.c_int32),
  ('_old_offset', ctypes.c_int64),
  ('_cur_column', ctypes.c_uint16),
  ('_vtable_offset', ctypes.c_char),
  ('_shortbuf', (ctypes.c_char * 1)),
  ('_lock', ctypes.POINTER(_IO_lock_t)),
  ('_offset', ctypes.c_int64),
  ('_codecvt', ctypes.POINTER(struct__IO_codecvt)),
  ('_wide_data', ctypes.POINTER(struct__IO_wide_data)),
  ('_freeres_list', ctypes.POINTER(struct__IO_FILE)),
  ('_freeres_buf', ctypes.c_void_p),
  ('__pad5', size_t),
  ('_mode', ctypes.c_int32),
  ('_unused2', (ctypes.c_char * 20)),
]
try: (nir_print_shader:=dll.nir_print_shader).restype, nir_print_shader.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_print_function_body:=dll.nir_print_function_body).restype, nir_print_function_body.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_print_shader_annotated:=dll.nir_print_shader_annotated).restype, nir_print_shader_annotated.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_print_instr:=dll.nir_print_instr).restype, nir_print_instr.argtypes = None, [ctypes.POINTER(nir_instr), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_print_deref:=dll.nir_print_deref).restype, nir_print_deref.argtypes = None, [ctypes.POINTER(nir_deref_instr), ctypes.POINTER(FILE)]
except AttributeError: pass

enum_mesa_log_level = CEnum(ctypes.c_uint32)
MESA_LOG_ERROR = enum_mesa_log_level.define('MESA_LOG_ERROR', 0)
MESA_LOG_WARN = enum_mesa_log_level.define('MESA_LOG_WARN', 1)
MESA_LOG_INFO = enum_mesa_log_level.define('MESA_LOG_INFO', 2)
MESA_LOG_DEBUG = enum_mesa_log_level.define('MESA_LOG_DEBUG', 3)

try: (nir_log_shader_annotated_tagged:=dll.nir_log_shader_annotated_tagged).restype, nir_log_shader_annotated_tagged.argtypes = None, [enum_mesa_log_level, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(nir_shader), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_shader_as_str:=dll.nir_shader_as_str).restype, nir_shader_as_str.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(nir_shader), ctypes.c_void_p]
except AttributeError: pass

try: (nir_shader_as_str_annotated:=dll.nir_shader_as_str_annotated).restype, nir_shader_as_str_annotated.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_hash_table), ctypes.c_void_p]
except AttributeError: pass

try: (nir_instr_as_str:=dll.nir_instr_as_str).restype, nir_instr_as_str.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(nir_instr), ctypes.c_void_p]
except AttributeError: pass

try: (nir_shader_gather_debug_info:=dll.nir_shader_gather_debug_info).restype, nir_shader_gather_debug_info.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_char), uint32_t]
except AttributeError: pass

try: (nir_instr_clone:=dll.nir_instr_clone).restype, nir_instr_clone.argtypes = ctypes.POINTER(nir_instr), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_clone_deep:=dll.nir_instr_clone_deep).restype, nir_instr_clone_deep.argtypes = ctypes.POINTER(nir_instr), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_instr), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_alu_instr_clone:=dll.nir_alu_instr_clone).restype, nir_alu_instr_clone.argtypes = ctypes.POINTER(nir_alu_instr), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_alu_instr)]
except AttributeError: pass

try: (nir_shader_clone:=dll.nir_shader_clone).restype, nir_shader_clone.argtypes = ctypes.POINTER(nir_shader), [ctypes.c_void_p, ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_function_clone:=dll.nir_function_clone).restype, nir_function_clone.argtypes = ctypes.POINTER(nir_function), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function)]
except AttributeError: pass

try: (nir_function_impl_clone:=dll.nir_function_impl_clone).restype, nir_function_impl_clone.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_function_impl_clone_remap_globals:=dll.nir_function_impl_clone_remap_globals).restype, nir_function_impl_clone_remap_globals.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function_impl), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_constant_clone:=dll.nir_constant_clone).restype, nir_constant_clone.argtypes = ctypes.POINTER(nir_constant), [ctypes.POINTER(nir_constant), ctypes.POINTER(nir_variable)]
except AttributeError: pass

try: (nir_variable_clone:=dll.nir_variable_clone).restype, nir_variable_clone.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_variable), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_replace:=dll.nir_shader_replace).restype, nir_shader_replace.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_serialize_deserialize:=dll.nir_shader_serialize_deserialize).restype, nir_shader_serialize_deserialize.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_validate_shader:=dll.nir_validate_shader).restype, nir_validate_shader.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_validate_ssa_dominance:=dll.nir_validate_ssa_dominance).restype, nir_validate_ssa_dominance.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_metadata_set_validation_flag:=dll.nir_metadata_set_validation_flag).restype, nir_metadata_set_validation_flag.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_metadata_check_validation_flag:=dll.nir_metadata_check_validation_flag).restype, nir_metadata_check_validation_flag.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_metadata_require_all:=dll.nir_metadata_require_all).restype, nir_metadata_require_all.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_instr_writemask_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(const_struct_nir_instr), ctypes.c_uint32, ctypes.c_void_p)
class struct_nir_builder(Struct): pass
nir_lower_instr_cb = ctypes.CFUNCTYPE(ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
try: (nir_function_impl_lower_instructions:=dll.nir_function_impl_lower_instructions).restype, nir_function_impl_lower_instructions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl), nir_instr_filter_cb, nir_lower_instr_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_shader_lower_instructions:=dll.nir_shader_lower_instructions).restype, nir_shader_lower_instructions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_filter_cb, nir_lower_instr_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_calc_dominance_impl:=dll.nir_calc_dominance_impl).restype, nir_calc_dominance_impl.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_calc_dominance:=dll.nir_calc_dominance).restype, nir_calc_dominance.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_dominance_lca:=dll.nir_dominance_lca).restype, nir_dominance_lca.argtypes = ctypes.POINTER(nir_block), [ctypes.POINTER(nir_block), ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_dominates:=dll.nir_block_dominates).restype, nir_block_dominates.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block), ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_block_is_unreachable:=dll.nir_block_is_unreachable).restype, nir_block_is_unreachable.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_dump_dom_tree_impl:=dll.nir_dump_dom_tree_impl).restype, nir_dump_dom_tree_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_dom_tree:=dll.nir_dump_dom_tree).restype, nir_dump_dom_tree.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_dom_frontier_impl:=dll.nir_dump_dom_frontier_impl).restype, nir_dump_dom_frontier_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_dom_frontier:=dll.nir_dump_dom_frontier).restype, nir_dump_dom_frontier.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_cfg_impl:=dll.nir_dump_cfg_impl).restype, nir_dump_cfg_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_dump_cfg:=dll.nir_dump_cfg).restype, nir_dump_cfg.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

try: (nir_gs_count_vertices_and_primitives:=dll.nir_gs_count_vertices_and_primitives).restype, nir_gs_count_vertices_and_primitives.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError: pass

nir_load_grouping = CEnum(ctypes.c_uint32)
nir_group_all = nir_load_grouping.define('nir_group_all', 0)
nir_group_same_resource_only = nir_load_grouping.define('nir_group_same_resource_only', 1)

try: (nir_group_loads:=dll.nir_group_loads).restype, nir_group_loads.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_load_grouping, ctypes.c_uint32]
except AttributeError: pass

try: (nir_shrink_vec_array_vars:=dll.nir_shrink_vec_array_vars).restype, nir_shrink_vec_array_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_split_array_vars:=dll.nir_split_array_vars).restype, nir_split_array_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_split_var_copies:=dll.nir_split_var_copies).restype, nir_split_var_copies.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_split_per_member_structs:=dll.nir_split_per_member_structs).restype, nir_split_per_member_structs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_split_struct_vars:=dll.nir_split_struct_vars).restype, nir_split_struct_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_returns_impl:=dll.nir_lower_returns_impl).restype, nir_lower_returns_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_lower_returns:=dll.nir_lower_returns).restype, nir_lower_returns.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_builder = struct_nir_builder
try: (nir_inline_function_impl:=dll.nir_inline_function_impl).restype, nir_inline_function_impl.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_function_impl), ctypes.POINTER(ctypes.POINTER(nir_def)), ctypes.POINTER(struct_hash_table)]
except AttributeError: pass

try: (nir_inline_functions:=dll.nir_inline_functions).restype, nir_inline_functions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_cleanup_functions:=dll.nir_cleanup_functions).restype, nir_cleanup_functions.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_link_shader_functions:=dll.nir_link_shader_functions).restype, nir_link_shader_functions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_calls_to_builtins:=dll.nir_lower_calls_to_builtins).restype, nir_lower_calls_to_builtins.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_find_inlinable_uniforms:=dll.nir_find_inlinable_uniforms).restype, nir_find_inlinable_uniforms.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_inline_uniforms:=dll.nir_inline_uniforms).restype, nir_inline_uniforms.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.POINTER(uint32_t), ctypes.POINTER(uint16_t)]
except AttributeError: pass

try: (nir_collect_src_uniforms:=dll.nir_collect_src_uniforms).restype, nir_collect_src_uniforms.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_src), ctypes.c_int32, ctypes.POINTER(uint32_t), ctypes.POINTER(uint8_t), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_add_inlinable_uniforms:=dll.nir_add_inlinable_uniforms).restype, nir_add_inlinable_uniforms.argtypes = None, [ctypes.POINTER(nir_src), ctypes.POINTER(nir_loop_info), ctypes.POINTER(uint32_t), ctypes.POINTER(uint8_t), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_propagate_invariant:=dll.nir_propagate_invariant).restype, nir_propagate_invariant.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_var_copy_instr:=dll.nir_lower_var_copy_instr).restype, nir_lower_var_copy_instr.argtypes = None, [ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_deref_copy_instr:=dll.nir_lower_deref_copy_instr).restype, nir_lower_deref_copy_instr.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_lower_var_copies:=dll.nir_lower_var_copies).restype, nir_lower_var_copies.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_memcpy:=dll.nir_opt_memcpy).restype, nir_opt_memcpy.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_memcpy:=dll.nir_lower_memcpy).restype, nir_lower_memcpy.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_fixup_deref_modes:=dll.nir_fixup_deref_modes).restype, nir_fixup_deref_modes.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_fixup_deref_types:=dll.nir_fixup_deref_types).restype, nir_fixup_deref_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_global_vars_to_local:=dll.nir_lower_global_vars_to_local).restype, nir_lower_global_vars_to_local.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_constant_to_temp:=dll.nir_lower_constant_to_temp).restype, nir_lower_constant_to_temp.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_lower_array_deref_of_vec_options = CEnum(ctypes.c_uint32)
nir_lower_direct_array_deref_of_vec_load = nir_lower_array_deref_of_vec_options.define('nir_lower_direct_array_deref_of_vec_load', 1)
nir_lower_indirect_array_deref_of_vec_load = nir_lower_array_deref_of_vec_options.define('nir_lower_indirect_array_deref_of_vec_load', 2)
nir_lower_direct_array_deref_of_vec_store = nir_lower_array_deref_of_vec_options.define('nir_lower_direct_array_deref_of_vec_store', 4)
nir_lower_indirect_array_deref_of_vec_store = nir_lower_array_deref_of_vec_options.define('nir_lower_indirect_array_deref_of_vec_store', 8)

try: (nir_lower_array_deref_of_vec:=dll.nir_lower_array_deref_of_vec).restype, nir_lower_array_deref_of_vec.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(nir_variable)), nir_lower_array_deref_of_vec_options]
except AttributeError: pass

try: (nir_lower_indirect_derefs:=dll.nir_lower_indirect_derefs).restype, nir_lower_indirect_derefs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, uint32_t]
except AttributeError: pass

try: (nir_lower_indirect_var_derefs:=dll.nir_lower_indirect_var_derefs).restype, nir_lower_indirect_var_derefs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_set)]
except AttributeError: pass

try: (nir_lower_locals_to_regs:=dll.nir_lower_locals_to_regs).restype, nir_lower_locals_to_regs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), uint8_t]
except AttributeError: pass

try: (nir_lower_io_vars_to_temporaries:=dll.nir_lower_io_vars_to_temporaries).restype, nir_lower_io_vars_to_temporaries.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function_impl), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

class const_struct_glsl_type(Struct): pass
const_struct_glsl_type._fields_ = [
  ('gl_type', uint32_t),
  ('base_type', enum_glsl_base_type,8),
  ('sampled_type', enum_glsl_base_type,8),
  ('sampler_dimensionality', ctypes.c_uint32,4),
  ('sampler_shadow', ctypes.c_uint32,1),
  ('sampler_array', ctypes.c_uint32,1),
  ('interface_packing', ctypes.c_uint32,2),
  ('interface_row_major', ctypes.c_uint32,1),
  ('cmat_desc', struct_glsl_cmat_description),
  ('packed', ctypes.c_uint32,1),
  ('has_builtin_name', ctypes.c_uint32,1),
  ('vector_elements', uint8_t),
  ('matrix_columns', uint8_t),
  ('length', ctypes.c_uint32),
  ('name_id', uintptr_t),
  ('explicit_stride', ctypes.c_uint32),
  ('explicit_alignment', ctypes.c_uint32),
  ('fields', struct_glsl_type_fields),
]
glsl_type_size_align_func = ctypes.CFUNCTYPE(None, ctypes.POINTER(const_struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32))
try: (nir_lower_vars_to_scratch:=dll.nir_lower_vars_to_scratch).restype, nir_lower_vars_to_scratch.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_int32, glsl_type_size_align_func, glsl_type_size_align_func]
except AttributeError: pass

try: (nir_lower_scratch_to_var:=dll.nir_lower_scratch_to_var).restype, nir_lower_scratch_to_var.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_clip_halfz:=dll.nir_lower_clip_halfz).restype, nir_lower_clip_halfz.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_gather_info:=dll.nir_shader_gather_info).restype, nir_shader_gather_info.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_gather_types:=dll.nir_gather_types).restype, nir_gather_types.argtypes = None, [ctypes.POINTER(nir_function_impl), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (nir_remove_unused_varyings:=dll.nir_remove_unused_varyings).restype, nir_remove_unused_varyings.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_unused_io_vars:=dll.nir_remove_unused_io_vars).restype, nir_remove_unused_io_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.POINTER(uint64_t), ctypes.POINTER(uint64_t)]
except AttributeError: pass

try: (nir_compact_varyings:=dll.nir_compact_varyings).restype, nir_compact_varyings.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_link_xfb_varyings:=dll.nir_link_xfb_varyings).restype, nir_link_xfb_varyings.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_link_opt_varyings:=dll.nir_link_opt_varyings).restype, nir_link_opt_varyings.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_link_varying_precision:=dll.nir_link_varying_precision).restype, nir_link_varying_precision.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_clone_uniform_variable:=dll.nir_clone_uniform_variable).restype, nir_clone_uniform_variable.argtypes = ctypes.POINTER(nir_variable), [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_variable), ctypes.c_bool]
except AttributeError: pass

try: (nir_clone_deref_instr:=dll.nir_clone_deref_instr).restype, nir_clone_deref_instr.argtypes = ctypes.POINTER(nir_deref_instr), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_variable), ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

nir_opt_varyings_progress = CEnum(ctypes.c_uint32)
nir_progress_producer = nir_opt_varyings_progress.define('nir_progress_producer', 1)
nir_progress_consumer = nir_opt_varyings_progress.define('nir_progress_consumer', 2)

try: (nir_opt_varyings:=dll.nir_opt_varyings).restype, nir_opt_varyings.argtypes = nir_opt_varyings_progress, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
except AttributeError: pass

gl_varying_slot = CEnum(ctypes.c_uint32)
VARYING_SLOT_POS = gl_varying_slot.define('VARYING_SLOT_POS', 0)
VARYING_SLOT_COL0 = gl_varying_slot.define('VARYING_SLOT_COL0', 1)
VARYING_SLOT_COL1 = gl_varying_slot.define('VARYING_SLOT_COL1', 2)
VARYING_SLOT_FOGC = gl_varying_slot.define('VARYING_SLOT_FOGC', 3)
VARYING_SLOT_TEX0 = gl_varying_slot.define('VARYING_SLOT_TEX0', 4)
VARYING_SLOT_TEX1 = gl_varying_slot.define('VARYING_SLOT_TEX1', 5)
VARYING_SLOT_TEX2 = gl_varying_slot.define('VARYING_SLOT_TEX2', 6)
VARYING_SLOT_TEX3 = gl_varying_slot.define('VARYING_SLOT_TEX3', 7)
VARYING_SLOT_TEX4 = gl_varying_slot.define('VARYING_SLOT_TEX4', 8)
VARYING_SLOT_TEX5 = gl_varying_slot.define('VARYING_SLOT_TEX5', 9)
VARYING_SLOT_TEX6 = gl_varying_slot.define('VARYING_SLOT_TEX6', 10)
VARYING_SLOT_TEX7 = gl_varying_slot.define('VARYING_SLOT_TEX7', 11)
VARYING_SLOT_PSIZ = gl_varying_slot.define('VARYING_SLOT_PSIZ', 12)
VARYING_SLOT_BFC0 = gl_varying_slot.define('VARYING_SLOT_BFC0', 13)
VARYING_SLOT_BFC1 = gl_varying_slot.define('VARYING_SLOT_BFC1', 14)
VARYING_SLOT_EDGE = gl_varying_slot.define('VARYING_SLOT_EDGE', 15)
VARYING_SLOT_CLIP_VERTEX = gl_varying_slot.define('VARYING_SLOT_CLIP_VERTEX', 16)
VARYING_SLOT_CLIP_DIST0 = gl_varying_slot.define('VARYING_SLOT_CLIP_DIST0', 17)
VARYING_SLOT_CLIP_DIST1 = gl_varying_slot.define('VARYING_SLOT_CLIP_DIST1', 18)
VARYING_SLOT_CULL_DIST0 = gl_varying_slot.define('VARYING_SLOT_CULL_DIST0', 19)
VARYING_SLOT_CULL_DIST1 = gl_varying_slot.define('VARYING_SLOT_CULL_DIST1', 20)
VARYING_SLOT_PRIMITIVE_ID = gl_varying_slot.define('VARYING_SLOT_PRIMITIVE_ID', 21)
VARYING_SLOT_LAYER = gl_varying_slot.define('VARYING_SLOT_LAYER', 22)
VARYING_SLOT_VIEWPORT = gl_varying_slot.define('VARYING_SLOT_VIEWPORT', 23)
VARYING_SLOT_FACE = gl_varying_slot.define('VARYING_SLOT_FACE', 24)
VARYING_SLOT_PNTC = gl_varying_slot.define('VARYING_SLOT_PNTC', 25)
VARYING_SLOT_TESS_LEVEL_OUTER = gl_varying_slot.define('VARYING_SLOT_TESS_LEVEL_OUTER', 26)
VARYING_SLOT_TESS_LEVEL_INNER = gl_varying_slot.define('VARYING_SLOT_TESS_LEVEL_INNER', 27)
VARYING_SLOT_BOUNDING_BOX0 = gl_varying_slot.define('VARYING_SLOT_BOUNDING_BOX0', 28)
VARYING_SLOT_BOUNDING_BOX1 = gl_varying_slot.define('VARYING_SLOT_BOUNDING_BOX1', 29)
VARYING_SLOT_VIEW_INDEX = gl_varying_slot.define('VARYING_SLOT_VIEW_INDEX', 30)
VARYING_SLOT_VIEWPORT_MASK = gl_varying_slot.define('VARYING_SLOT_VIEWPORT_MASK', 31)
VARYING_SLOT_PRIMITIVE_SHADING_RATE = gl_varying_slot.define('VARYING_SLOT_PRIMITIVE_SHADING_RATE', 24)
VARYING_SLOT_PRIMITIVE_COUNT = gl_varying_slot.define('VARYING_SLOT_PRIMITIVE_COUNT', 26)
VARYING_SLOT_PRIMITIVE_INDICES = gl_varying_slot.define('VARYING_SLOT_PRIMITIVE_INDICES', 27)
VARYING_SLOT_TASK_COUNT = gl_varying_slot.define('VARYING_SLOT_TASK_COUNT', 28)
VARYING_SLOT_CULL_PRIMITIVE = gl_varying_slot.define('VARYING_SLOT_CULL_PRIMITIVE', 28)
VARYING_SLOT_VAR0 = gl_varying_slot.define('VARYING_SLOT_VAR0', 32)
VARYING_SLOT_VAR1 = gl_varying_slot.define('VARYING_SLOT_VAR1', 33)
VARYING_SLOT_VAR2 = gl_varying_slot.define('VARYING_SLOT_VAR2', 34)
VARYING_SLOT_VAR3 = gl_varying_slot.define('VARYING_SLOT_VAR3', 35)
VARYING_SLOT_VAR4 = gl_varying_slot.define('VARYING_SLOT_VAR4', 36)
VARYING_SLOT_VAR5 = gl_varying_slot.define('VARYING_SLOT_VAR5', 37)
VARYING_SLOT_VAR6 = gl_varying_slot.define('VARYING_SLOT_VAR6', 38)
VARYING_SLOT_VAR7 = gl_varying_slot.define('VARYING_SLOT_VAR7', 39)
VARYING_SLOT_VAR8 = gl_varying_slot.define('VARYING_SLOT_VAR8', 40)
VARYING_SLOT_VAR9 = gl_varying_slot.define('VARYING_SLOT_VAR9', 41)
VARYING_SLOT_VAR10 = gl_varying_slot.define('VARYING_SLOT_VAR10', 42)
VARYING_SLOT_VAR11 = gl_varying_slot.define('VARYING_SLOT_VAR11', 43)
VARYING_SLOT_VAR12 = gl_varying_slot.define('VARYING_SLOT_VAR12', 44)
VARYING_SLOT_VAR13 = gl_varying_slot.define('VARYING_SLOT_VAR13', 45)
VARYING_SLOT_VAR14 = gl_varying_slot.define('VARYING_SLOT_VAR14', 46)
VARYING_SLOT_VAR15 = gl_varying_slot.define('VARYING_SLOT_VAR15', 47)
VARYING_SLOT_VAR16 = gl_varying_slot.define('VARYING_SLOT_VAR16', 48)
VARYING_SLOT_VAR17 = gl_varying_slot.define('VARYING_SLOT_VAR17', 49)
VARYING_SLOT_VAR18 = gl_varying_slot.define('VARYING_SLOT_VAR18', 50)
VARYING_SLOT_VAR19 = gl_varying_slot.define('VARYING_SLOT_VAR19', 51)
VARYING_SLOT_VAR20 = gl_varying_slot.define('VARYING_SLOT_VAR20', 52)
VARYING_SLOT_VAR21 = gl_varying_slot.define('VARYING_SLOT_VAR21', 53)
VARYING_SLOT_VAR22 = gl_varying_slot.define('VARYING_SLOT_VAR22', 54)
VARYING_SLOT_VAR23 = gl_varying_slot.define('VARYING_SLOT_VAR23', 55)
VARYING_SLOT_VAR24 = gl_varying_slot.define('VARYING_SLOT_VAR24', 56)
VARYING_SLOT_VAR25 = gl_varying_slot.define('VARYING_SLOT_VAR25', 57)
VARYING_SLOT_VAR26 = gl_varying_slot.define('VARYING_SLOT_VAR26', 58)
VARYING_SLOT_VAR27 = gl_varying_slot.define('VARYING_SLOT_VAR27', 59)
VARYING_SLOT_VAR28 = gl_varying_slot.define('VARYING_SLOT_VAR28', 60)
VARYING_SLOT_VAR29 = gl_varying_slot.define('VARYING_SLOT_VAR29', 61)
VARYING_SLOT_VAR30 = gl_varying_slot.define('VARYING_SLOT_VAR30', 62)
VARYING_SLOT_VAR31 = gl_varying_slot.define('VARYING_SLOT_VAR31', 63)
VARYING_SLOT_PATCH0 = gl_varying_slot.define('VARYING_SLOT_PATCH0', 64)
VARYING_SLOT_PATCH1 = gl_varying_slot.define('VARYING_SLOT_PATCH1', 65)
VARYING_SLOT_PATCH2 = gl_varying_slot.define('VARYING_SLOT_PATCH2', 66)
VARYING_SLOT_PATCH3 = gl_varying_slot.define('VARYING_SLOT_PATCH3', 67)
VARYING_SLOT_PATCH4 = gl_varying_slot.define('VARYING_SLOT_PATCH4', 68)
VARYING_SLOT_PATCH5 = gl_varying_slot.define('VARYING_SLOT_PATCH5', 69)
VARYING_SLOT_PATCH6 = gl_varying_slot.define('VARYING_SLOT_PATCH6', 70)
VARYING_SLOT_PATCH7 = gl_varying_slot.define('VARYING_SLOT_PATCH7', 71)
VARYING_SLOT_PATCH8 = gl_varying_slot.define('VARYING_SLOT_PATCH8', 72)
VARYING_SLOT_PATCH9 = gl_varying_slot.define('VARYING_SLOT_PATCH9', 73)
VARYING_SLOT_PATCH10 = gl_varying_slot.define('VARYING_SLOT_PATCH10', 74)
VARYING_SLOT_PATCH11 = gl_varying_slot.define('VARYING_SLOT_PATCH11', 75)
VARYING_SLOT_PATCH12 = gl_varying_slot.define('VARYING_SLOT_PATCH12', 76)
VARYING_SLOT_PATCH13 = gl_varying_slot.define('VARYING_SLOT_PATCH13', 77)
VARYING_SLOT_PATCH14 = gl_varying_slot.define('VARYING_SLOT_PATCH14', 78)
VARYING_SLOT_PATCH15 = gl_varying_slot.define('VARYING_SLOT_PATCH15', 79)
VARYING_SLOT_PATCH16 = gl_varying_slot.define('VARYING_SLOT_PATCH16', 80)
VARYING_SLOT_PATCH17 = gl_varying_slot.define('VARYING_SLOT_PATCH17', 81)
VARYING_SLOT_PATCH18 = gl_varying_slot.define('VARYING_SLOT_PATCH18', 82)
VARYING_SLOT_PATCH19 = gl_varying_slot.define('VARYING_SLOT_PATCH19', 83)
VARYING_SLOT_PATCH20 = gl_varying_slot.define('VARYING_SLOT_PATCH20', 84)
VARYING_SLOT_PATCH21 = gl_varying_slot.define('VARYING_SLOT_PATCH21', 85)
VARYING_SLOT_PATCH22 = gl_varying_slot.define('VARYING_SLOT_PATCH22', 86)
VARYING_SLOT_PATCH23 = gl_varying_slot.define('VARYING_SLOT_PATCH23', 87)
VARYING_SLOT_PATCH24 = gl_varying_slot.define('VARYING_SLOT_PATCH24', 88)
VARYING_SLOT_PATCH25 = gl_varying_slot.define('VARYING_SLOT_PATCH25', 89)
VARYING_SLOT_PATCH26 = gl_varying_slot.define('VARYING_SLOT_PATCH26', 90)
VARYING_SLOT_PATCH27 = gl_varying_slot.define('VARYING_SLOT_PATCH27', 91)
VARYING_SLOT_PATCH28 = gl_varying_slot.define('VARYING_SLOT_PATCH28', 92)
VARYING_SLOT_PATCH29 = gl_varying_slot.define('VARYING_SLOT_PATCH29', 93)
VARYING_SLOT_PATCH30 = gl_varying_slot.define('VARYING_SLOT_PATCH30', 94)
VARYING_SLOT_PATCH31 = gl_varying_slot.define('VARYING_SLOT_PATCH31', 95)
VARYING_SLOT_VAR0_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR0_16BIT', 96)
VARYING_SLOT_VAR1_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR1_16BIT', 97)
VARYING_SLOT_VAR2_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR2_16BIT', 98)
VARYING_SLOT_VAR3_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR3_16BIT', 99)
VARYING_SLOT_VAR4_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR4_16BIT', 100)
VARYING_SLOT_VAR5_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR5_16BIT', 101)
VARYING_SLOT_VAR6_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR6_16BIT', 102)
VARYING_SLOT_VAR7_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR7_16BIT', 103)
VARYING_SLOT_VAR8_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR8_16BIT', 104)
VARYING_SLOT_VAR9_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR9_16BIT', 105)
VARYING_SLOT_VAR10_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR10_16BIT', 106)
VARYING_SLOT_VAR11_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR11_16BIT', 107)
VARYING_SLOT_VAR12_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR12_16BIT', 108)
VARYING_SLOT_VAR13_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR13_16BIT', 109)
VARYING_SLOT_VAR14_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR14_16BIT', 110)
VARYING_SLOT_VAR15_16BIT = gl_varying_slot.define('VARYING_SLOT_VAR15_16BIT', 111)
NUM_TOTAL_VARYING_SLOTS = gl_varying_slot.define('NUM_TOTAL_VARYING_SLOTS', 112)

try: (nir_slot_is_sysval_output:=dll.nir_slot_is_sysval_output).restype, nir_slot_is_sysval_output.argtypes = ctypes.c_bool, [gl_varying_slot, gl_shader_stage]
except AttributeError: pass

try: (nir_slot_is_varying:=dll.nir_slot_is_varying).restype, nir_slot_is_varying.argtypes = ctypes.c_bool, [gl_varying_slot, gl_shader_stage]
except AttributeError: pass

try: (nir_slot_is_sysval_output_and_varying:=dll.nir_slot_is_sysval_output_and_varying).restype, nir_slot_is_sysval_output_and_varying.argtypes = ctypes.c_bool, [gl_varying_slot, gl_shader_stage]
except AttributeError: pass

try: (nir_remove_varying:=dll.nir_remove_varying).restype, nir_remove_varying.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr), gl_shader_stage]
except AttributeError: pass

try: (nir_remove_sysval_output:=dll.nir_remove_sysval_output).restype, nir_remove_sysval_output.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr), gl_shader_stage]
except AttributeError: pass

try: (nir_lower_amul:=dll.nir_lower_amul).restype, nir_lower_amul.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(const_struct_glsl_type), ctypes.c_bool)]
except AttributeError: pass

try: (nir_lower_ubo_vec4:=dll.nir_lower_ubo_vec4).restype, nir_lower_ubo_vec4.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_sort_variables_by_location:=dll.nir_sort_variables_by_location).restype, nir_sort_variables_by_location.argtypes = None, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_assign_io_var_locations:=dll.nir_assign_io_var_locations).restype, nir_assign_io_var_locations.argtypes = None, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.POINTER(ctypes.c_uint32), gl_shader_stage]
except AttributeError: pass

try: (nir_opt_clip_cull_const:=dll.nir_opt_clip_cull_const).restype, nir_opt_clip_cull_const.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_lower_io_options = CEnum(ctypes.c_uint32)
nir_lower_io_lower_64bit_to_32 = nir_lower_io_options.define('nir_lower_io_lower_64bit_to_32', 1)
nir_lower_io_lower_64bit_float_to_32 = nir_lower_io_options.define('nir_lower_io_lower_64bit_float_to_32', 2)
nir_lower_io_lower_64bit_to_32_new = nir_lower_io_options.define('nir_lower_io_lower_64bit_to_32_new', 4)
nir_lower_io_use_interpolated_input_intrinsics = nir_lower_io_options.define('nir_lower_io_use_interpolated_input_intrinsics', 8)

try: (nir_lower_io:=dll.nir_lower_io).restype, nir_lower_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(const_struct_glsl_type), ctypes.c_bool), nir_lower_io_options]
except AttributeError: pass

try: (nir_io_add_const_offset_to_base:=dll.nir_io_add_const_offset_to_base).restype, nir_io_add_const_offset_to_base.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_io_passes:=dll.nir_lower_io_passes).restype, nir_lower_io_passes.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_io_add_intrinsic_xfb_info:=dll.nir_io_add_intrinsic_xfb_info).restype, nir_io_add_intrinsic_xfb_info.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_io_indirect_loads:=dll.nir_lower_io_indirect_loads).restype, nir_lower_io_indirect_loads.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_vars_to_explicit_types:=dll.nir_lower_vars_to_explicit_types).restype, nir_lower_vars_to_explicit_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, glsl_type_size_align_func]
except AttributeError: pass

try: (nir_gather_explicit_io_initializers:=dll.nir_gather_explicit_io_initializers).restype, nir_gather_explicit_io_initializers.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.c_void_p, size_t, nir_variable_mode]
except AttributeError: pass

try: (nir_lower_vec3_to_vec4:=dll.nir_lower_vec3_to_vec4).restype, nir_lower_vec3_to_vec4.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

nir_address_format = CEnum(ctypes.c_uint32)
nir_address_format_32bit_global = nir_address_format.define('nir_address_format_32bit_global', 0)
nir_address_format_64bit_global = nir_address_format.define('nir_address_format_64bit_global', 1)
nir_address_format_2x32bit_global = nir_address_format.define('nir_address_format_2x32bit_global', 2)
nir_address_format_64bit_global_32bit_offset = nir_address_format.define('nir_address_format_64bit_global_32bit_offset', 3)
nir_address_format_64bit_bounded_global = nir_address_format.define('nir_address_format_64bit_bounded_global', 4)
nir_address_format_32bit_index_offset = nir_address_format.define('nir_address_format_32bit_index_offset', 5)
nir_address_format_32bit_index_offset_pack64 = nir_address_format.define('nir_address_format_32bit_index_offset_pack64', 6)
nir_address_format_vec2_index_32bit_offset = nir_address_format.define('nir_address_format_vec2_index_32bit_offset', 7)
nir_address_format_62bit_generic = nir_address_format.define('nir_address_format_62bit_generic', 8)
nir_address_format_32bit_offset = nir_address_format.define('nir_address_format_32bit_offset', 9)
nir_address_format_32bit_offset_as_64bit = nir_address_format.define('nir_address_format_32bit_offset_as_64bit', 10)
nir_address_format_logical = nir_address_format.define('nir_address_format_logical', 11)

try: (nir_address_format_bit_size:=dll.nir_address_format_bit_size).restype, nir_address_format_bit_size.argtypes = ctypes.c_uint32, [nir_address_format]
except AttributeError: pass

try: (nir_address_format_num_components:=dll.nir_address_format_num_components).restype, nir_address_format_num_components.argtypes = ctypes.c_uint32, [nir_address_format]
except AttributeError: pass

try: (nir_address_format_null_value:=dll.nir_address_format_null_value).restype, nir_address_format_null_value.argtypes = ctypes.POINTER(nir_const_value), [nir_address_format]
except AttributeError: pass

try: (nir_build_addr_iadd:=dll.nir_build_addr_iadd).restype, nir_build_addr_iadd.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), nir_address_format, nir_variable_mode, ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_addr_iadd_imm:=dll.nir_build_addr_iadd_imm).restype, nir_build_addr_iadd_imm.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), nir_address_format, nir_variable_mode, int64_t]
except AttributeError: pass

try: (nir_build_addr_ieq:=dll.nir_build_addr_ieq).restype, nir_build_addr_ieq.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), nir_address_format]
except AttributeError: pass

try: (nir_build_addr_isub:=dll.nir_build_addr_isub).restype, nir_build_addr_isub.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), nir_address_format]
except AttributeError: pass

try: (nir_explicit_io_address_from_deref:=dll.nir_explicit_io_address_from_deref).restype, nir_explicit_io_address_from_deref.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_deref_instr), ctypes.POINTER(nir_def), nir_address_format]
except AttributeError: pass

try: (nir_get_explicit_deref_align:=dll.nir_get_explicit_deref_align).restype, nir_get_explicit_deref_align.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr), ctypes.c_bool, ctypes.POINTER(uint32_t), ctypes.POINTER(uint32_t)]
except AttributeError: pass

try: (nir_lower_explicit_io_instr:=dll.nir_lower_explicit_io_instr).restype, nir_lower_explicit_io_instr.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_intrinsic_instr), ctypes.POINTER(nir_def), nir_address_format]
except AttributeError: pass

try: (nir_lower_explicit_io:=dll.nir_lower_explicit_io).restype, nir_lower_explicit_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, nir_address_format]
except AttributeError: pass

nir_mem_access_shift_method = CEnum(ctypes.c_uint32)
nir_mem_access_shift_method_scalar = nir_mem_access_shift_method.define('nir_mem_access_shift_method_scalar', 0)
nir_mem_access_shift_method_shift64 = nir_mem_access_shift_method.define('nir_mem_access_shift_method_shift64', 1)
nir_mem_access_shift_method_bytealign_amd = nir_mem_access_shift_method.define('nir_mem_access_shift_method_bytealign_amd', 2)

class struct_nir_mem_access_size_align(Struct): pass
struct_nir_mem_access_size_align._fields_ = [
  ('num_components', uint8_t),
  ('bit_size', uint8_t),
  ('align', uint16_t),
  ('shift', nir_mem_access_shift_method),
]
nir_mem_access_size_align = struct_nir_mem_access_size_align
enum_gl_access_qualifier = CEnum(ctypes.c_uint32)
ACCESS_COHERENT = enum_gl_access_qualifier.define('ACCESS_COHERENT', 1)
ACCESS_RESTRICT = enum_gl_access_qualifier.define('ACCESS_RESTRICT', 2)
ACCESS_VOLATILE = enum_gl_access_qualifier.define('ACCESS_VOLATILE', 4)
ACCESS_NON_READABLE = enum_gl_access_qualifier.define('ACCESS_NON_READABLE', 8)
ACCESS_NON_WRITEABLE = enum_gl_access_qualifier.define('ACCESS_NON_WRITEABLE', 16)
ACCESS_NON_UNIFORM = enum_gl_access_qualifier.define('ACCESS_NON_UNIFORM', 32)
ACCESS_CAN_REORDER = enum_gl_access_qualifier.define('ACCESS_CAN_REORDER', 64)
ACCESS_NON_TEMPORAL = enum_gl_access_qualifier.define('ACCESS_NON_TEMPORAL', 128)
ACCESS_INCLUDE_HELPERS = enum_gl_access_qualifier.define('ACCESS_INCLUDE_HELPERS', 256)
ACCESS_IS_SWIZZLED_AMD = enum_gl_access_qualifier.define('ACCESS_IS_SWIZZLED_AMD', 512)
ACCESS_USES_FORMAT_AMD = enum_gl_access_qualifier.define('ACCESS_USES_FORMAT_AMD', 1024)
ACCESS_FMASK_LOWERED_AMD = enum_gl_access_qualifier.define('ACCESS_FMASK_LOWERED_AMD', 2048)
ACCESS_CAN_SPECULATE = enum_gl_access_qualifier.define('ACCESS_CAN_SPECULATE', 4096)
ACCESS_CP_GE_COHERENT_AMD = enum_gl_access_qualifier.define('ACCESS_CP_GE_COHERENT_AMD', 8192)
ACCESS_IN_BOUNDS = enum_gl_access_qualifier.define('ACCESS_IN_BOUNDS', 16384)
ACCESS_KEEP_SCALAR = enum_gl_access_qualifier.define('ACCESS_KEEP_SCALAR', 32768)
ACCESS_SMEM_AMD = enum_gl_access_qualifier.define('ACCESS_SMEM_AMD', 65536)

nir_lower_mem_access_bit_sizes_cb = ctypes.CFUNCTYPE(struct_nir_mem_access_size_align, nir_intrinsic_op, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, enum_gl_access_qualifier, ctypes.c_void_p)
class struct_nir_lower_mem_access_bit_sizes_options(Struct): pass
struct_nir_lower_mem_access_bit_sizes_options._fields_ = [
  ('callback', nir_lower_mem_access_bit_sizes_cb),
  ('modes', nir_variable_mode),
  ('may_lower_unaligned_stores_to_atomics', ctypes.c_bool),
  ('cb_data', ctypes.c_void_p),
]
nir_lower_mem_access_bit_sizes_options = struct_nir_lower_mem_access_bit_sizes_options
try: (nir_lower_mem_access_bit_sizes:=dll.nir_lower_mem_access_bit_sizes).restype, nir_lower_mem_access_bit_sizes.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_mem_access_bit_sizes_options)]
except AttributeError: pass

try: (nir_lower_robust_access:=dll.nir_lower_robust_access).restype, nir_lower_robust_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_intrin_filter_cb, ctypes.c_void_p]
except AttributeError: pass

nir_should_vectorize_mem_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_void_p)
class struct_nir_load_store_vectorize_options(Struct): pass
struct_nir_load_store_vectorize_options._fields_ = [
  ('callback', nir_should_vectorize_mem_func),
  ('modes', nir_variable_mode),
  ('robust_modes', nir_variable_mode),
  ('cb_data', ctypes.c_void_p),
  ('has_shared2_amd', ctypes.c_bool),
]
nir_load_store_vectorize_options = struct_nir_load_store_vectorize_options
try: (nir_opt_load_store_vectorize:=dll.nir_opt_load_store_vectorize).restype, nir_opt_load_store_vectorize.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_load_store_vectorize_options)]
except AttributeError: pass

try: (nir_opt_load_store_update_alignments:=dll.nir_opt_load_store_update_alignments).restype, nir_opt_load_store_update_alignments.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_lower_shader_calls_should_remat_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
class struct_nir_lower_shader_calls_options(Struct): pass
struct_nir_lower_shader_calls_options._fields_ = [
  ('address_format', nir_address_format),
  ('stack_alignment', ctypes.c_uint32),
  ('localized_loads', ctypes.c_bool),
  ('vectorizer_callback', nir_should_vectorize_mem_func),
  ('vectorizer_data', ctypes.c_void_p),
  ('should_remat_callback', nir_lower_shader_calls_should_remat_func),
  ('should_remat_data', ctypes.c_void_p),
]
nir_lower_shader_calls_options = struct_nir_lower_shader_calls_options
try: (nir_lower_shader_calls:=dll.nir_lower_shader_calls).restype, nir_lower_shader_calls.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_shader_calls_options), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(nir_shader))), ctypes.POINTER(uint32_t), ctypes.c_void_p]
except AttributeError: pass

try: (nir_get_io_offset_src_number:=dll.nir_get_io_offset_src_number).restype, nir_get_io_offset_src_number.argtypes = ctypes.c_int32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_index_src_number:=dll.nir_get_io_index_src_number).restype, nir_get_io_index_src_number.argtypes = ctypes.c_int32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_arrayed_index_src_number:=dll.nir_get_io_arrayed_index_src_number).restype, nir_get_io_arrayed_index_src_number.argtypes = ctypes.c_int32, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_offset_src:=dll.nir_get_io_offset_src).restype, nir_get_io_offset_src.argtypes = ctypes.POINTER(nir_src), [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_index_src:=dll.nir_get_io_index_src).restype, nir_get_io_index_src.argtypes = ctypes.POINTER(nir_src), [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_io_arrayed_index_src:=dll.nir_get_io_arrayed_index_src).restype, nir_get_io_arrayed_index_src.argtypes = ctypes.POINTER(nir_src), [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_get_shader_call_payload_src:=dll.nir_get_shader_call_payload_src).restype, nir_get_shader_call_payload_src.argtypes = ctypes.POINTER(nir_src), [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_is_output_load:=dll.nir_is_output_load).restype, nir_is_output_load.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (nir_is_arrayed_io:=dll.nir_is_arrayed_io).restype, nir_is_arrayed_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_variable), gl_shader_stage]
except AttributeError: pass

try: (nir_lower_reg_intrinsics_to_ssa_impl:=dll.nir_lower_reg_intrinsics_to_ssa_impl).restype, nir_lower_reg_intrinsics_to_ssa_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_lower_reg_intrinsics_to_ssa:=dll.nir_lower_reg_intrinsics_to_ssa).restype, nir_lower_reg_intrinsics_to_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_vars_to_ssa:=dll.nir_lower_vars_to_ssa).restype, nir_lower_vars_to_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_dead_derefs:=dll.nir_remove_dead_derefs).restype, nir_remove_dead_derefs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_dead_derefs_impl:=dll.nir_remove_dead_derefs_impl).restype, nir_remove_dead_derefs_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

class struct_nir_remove_dead_variables_options(Struct): pass
struct_nir_remove_dead_variables_options._fields_ = [
  ('can_remove_var', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(nir_variable), ctypes.c_void_p)),
  ('can_remove_var_data', ctypes.c_void_p),
]
nir_remove_dead_variables_options = struct_nir_remove_dead_variables_options
try: (nir_remove_dead_variables:=dll.nir_remove_dead_variables).restype, nir_remove_dead_variables.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.POINTER(nir_remove_dead_variables_options)]
except AttributeError: pass

try: (nir_lower_variable_initializers:=dll.nir_lower_variable_initializers).restype, nir_lower_variable_initializers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_zero_initialize_shared_memory:=dll.nir_zero_initialize_shared_memory).restype, nir_zero_initialize_shared_memory.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_clear_shared_memory:=dll.nir_clear_shared_memory).restype, nir_clear_shared_memory.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

nir_opt_move_to_top_options = CEnum(ctypes.c_uint32)
nir_move_to_entry_block_only = nir_opt_move_to_top_options.define('nir_move_to_entry_block_only', 1)
nir_move_to_top_input_loads = nir_opt_move_to_top_options.define('nir_move_to_top_input_loads', 2)
nir_move_to_top_load_smem_amd = nir_opt_move_to_top_options.define('nir_move_to_top_load_smem_amd', 4)

try: (nir_opt_move_to_top:=dll.nir_opt_move_to_top).restype, nir_opt_move_to_top.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_opt_move_to_top_options]
except AttributeError: pass

try: (nir_move_vec_src_uses_to_dest:=dll.nir_move_vec_src_uses_to_dest).restype, nir_move_vec_src_uses_to_dest.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_move_output_stores_to_end:=dll.nir_move_output_stores_to_end).restype, nir_move_output_stores_to_end.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_vec_to_regs:=dll.nir_lower_vec_to_regs).restype, nir_lower_vec_to_regs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_writemask_filter_cb, ctypes.c_void_p]
except AttributeError: pass

enum_compare_func = CEnum(ctypes.c_uint32)
COMPARE_FUNC_NEVER = enum_compare_func.define('COMPARE_FUNC_NEVER', 0)
COMPARE_FUNC_LESS = enum_compare_func.define('COMPARE_FUNC_LESS', 1)
COMPARE_FUNC_EQUAL = enum_compare_func.define('COMPARE_FUNC_EQUAL', 2)
COMPARE_FUNC_LEQUAL = enum_compare_func.define('COMPARE_FUNC_LEQUAL', 3)
COMPARE_FUNC_GREATER = enum_compare_func.define('COMPARE_FUNC_GREATER', 4)
COMPARE_FUNC_NOTEQUAL = enum_compare_func.define('COMPARE_FUNC_NOTEQUAL', 5)
COMPARE_FUNC_GEQUAL = enum_compare_func.define('COMPARE_FUNC_GEQUAL', 6)
COMPARE_FUNC_ALWAYS = enum_compare_func.define('COMPARE_FUNC_ALWAYS', 7)

try: (nir_lower_alpha_test:=dll.nir_lower_alpha_test).restype, nir_lower_alpha_test.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), enum_compare_func, ctypes.c_bool, ctypes.POINTER(gl_state_index16)]
except AttributeError: pass

try: (nir_lower_alpha_to_coverage:=dll.nir_lower_alpha_to_coverage).restype, nir_lower_alpha_to_coverage.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), uint8_t, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_alpha_to_one:=dll.nir_lower_alpha_to_one).restype, nir_lower_alpha_to_one.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_alu:=dll.nir_lower_alu).restype, nir_lower_alu.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_flrp:=dll.nir_lower_flrp).restype, nir_lower_flrp.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool]
except AttributeError: pass

try: (nir_scale_fdiv:=dll.nir_scale_fdiv).restype, nir_scale_fdiv.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_alu_to_scalar:=dll.nir_lower_alu_to_scalar).restype, nir_lower_alu_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_filter_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_alu_width:=dll.nir_lower_alu_width).restype, nir_lower_alu_width.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_vectorize_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_alu_vec8_16_srcs:=dll.nir_lower_alu_vec8_16_srcs).restype, nir_lower_alu_vec8_16_srcs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_bool_to_bitsize:=dll.nir_lower_bool_to_bitsize).restype, nir_lower_bool_to_bitsize.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_bool_to_float:=dll.nir_lower_bool_to_float).restype, nir_lower_bool_to_float.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_bool_to_int32:=dll.nir_lower_bool_to_int32).restype, nir_lower_bool_to_int32.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_simplify_convert_alu_types:=dll.nir_opt_simplify_convert_alu_types).restype, nir_opt_simplify_convert_alu_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_const_arrays_to_uniforms:=dll.nir_lower_const_arrays_to_uniforms).restype, nir_lower_const_arrays_to_uniforms.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_lower_convert_alu_types:=dll.nir_lower_convert_alu_types).restype, nir_lower_convert_alu_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(nir_intrinsic_instr))]
except AttributeError: pass

try: (nir_lower_constant_convert_alu_types:=dll.nir_lower_constant_convert_alu_types).restype, nir_lower_constant_convert_alu_types.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_alu_conversion_to_intrinsic:=dll.nir_lower_alu_conversion_to_intrinsic).restype, nir_lower_alu_conversion_to_intrinsic.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_int_to_float:=dll.nir_lower_int_to_float).restype, nir_lower_int_to_float.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_load_const_to_scalar:=dll.nir_lower_load_const_to_scalar).restype, nir_lower_load_const_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_read_invocation_to_scalar:=dll.nir_lower_read_invocation_to_scalar).restype, nir_lower_read_invocation_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_phis_to_scalar:=dll.nir_lower_phis_to_scalar).restype, nir_lower_phis_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_vectorize_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_all_phis_to_scalar:=dll.nir_lower_all_phis_to_scalar).restype, nir_lower_all_phis_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_io_array_vars_to_elements:=dll.nir_lower_io_array_vars_to_elements).restype, nir_lower_io_array_vars_to_elements.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_io_array_vars_to_elements_no_indirects:=dll.nir_lower_io_array_vars_to_elements_no_indirects).restype, nir_lower_io_array_vars_to_elements_no_indirects.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_io_to_scalar:=dll.nir_lower_io_to_scalar).restype, nir_lower_io_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, nir_instr_filter_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_io_vars_to_scalar:=dll.nir_lower_io_vars_to_scalar).restype, nir_lower_io_vars_to_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_opt_vectorize_io_vars:=dll.nir_opt_vectorize_io_vars).restype, nir_opt_vectorize_io_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_tess_level_array_vars_to_vec:=dll.nir_lower_tess_level_array_vars_to_vec).restype, nir_lower_tess_level_array_vars_to_vec.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_create_passthrough_tcs_impl:=dll.nir_create_passthrough_tcs_impl).restype, nir_create_passthrough_tcs_impl.argtypes = ctypes.POINTER(nir_shader), [ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, uint8_t]
except AttributeError: pass

try: (nir_create_passthrough_tcs:=dll.nir_create_passthrough_tcs).restype, nir_create_passthrough_tcs.argtypes = ctypes.POINTER(nir_shader), [ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(nir_shader), uint8_t]
except AttributeError: pass

try: (nir_create_passthrough_gs:=dll.nir_create_passthrough_gs).restype, nir_create_passthrough_gs.argtypes = ctypes.POINTER(nir_shader), [ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(nir_shader), enum_mesa_prim, enum_mesa_prim, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_fragcolor:=dll.nir_lower_fragcolor).restype, nir_lower_fragcolor.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_lower_fragcoord_wtrans:=dll.nir_lower_fragcoord_wtrans).restype, nir_lower_fragcoord_wtrans.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_frag_coord_to_pixel_coord:=dll.nir_opt_frag_coord_to_pixel_coord).restype, nir_opt_frag_coord_to_pixel_coord.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_frag_coord_to_pixel_coord:=dll.nir_lower_frag_coord_to_pixel_coord).restype, nir_lower_frag_coord_to_pixel_coord.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_viewport_transform:=dll.nir_lower_viewport_transform).restype, nir_lower_viewport_transform.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_uniforms_to_ubo:=dll.nir_lower_uniforms_to_ubo).restype, nir_lower_uniforms_to_ubo.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_is_helper_invocation:=dll.nir_lower_is_helper_invocation).restype, nir_lower_is_helper_invocation.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_single_sampled:=dll.nir_lower_single_sampled).restype, nir_lower_single_sampled.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_atomics:=dll.nir_lower_atomics).restype, nir_lower_atomics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_filter_cb]
except AttributeError: pass

class struct_nir_lower_subgroups_options(Struct): pass
struct_nir_lower_subgroups_options._fields_ = [
  ('filter', nir_instr_filter_cb),
  ('filter_data', ctypes.c_void_p),
  ('subgroup_size', uint8_t),
  ('ballot_bit_size', uint8_t),
  ('ballot_components', uint8_t),
  ('lower_to_scalar', ctypes.c_bool,1),
  ('lower_vote_trivial', ctypes.c_bool,1),
  ('lower_vote_feq', ctypes.c_bool,1),
  ('lower_vote_ieq', ctypes.c_bool,1),
  ('lower_vote_bool_eq', ctypes.c_bool,1),
  ('lower_first_invocation_to_ballot', ctypes.c_bool,1),
  ('lower_read_first_invocation', ctypes.c_bool,1),
  ('lower_subgroup_masks', ctypes.c_bool,1),
  ('lower_relative_shuffle', ctypes.c_bool,1),
  ('lower_shuffle_to_32bit', ctypes.c_bool,1),
  ('lower_shuffle_to_swizzle_amd', ctypes.c_bool,1),
  ('lower_shuffle', ctypes.c_bool,1),
  ('lower_quad', ctypes.c_bool,1),
  ('lower_quad_broadcast_dynamic', ctypes.c_bool,1),
  ('lower_quad_broadcast_dynamic_to_const', ctypes.c_bool,1),
  ('lower_quad_vote', ctypes.c_bool,1),
  ('lower_elect', ctypes.c_bool,1),
  ('lower_read_invocation_to_cond', ctypes.c_bool,1),
  ('lower_rotate_to_shuffle', ctypes.c_bool,1),
  ('lower_rotate_clustered_to_shuffle', ctypes.c_bool,1),
  ('lower_ballot_bit_count_to_mbcnt_amd', ctypes.c_bool,1),
  ('lower_inverse_ballot', ctypes.c_bool,1),
  ('lower_reduce', ctypes.c_bool,1),
  ('lower_boolean_reduce', ctypes.c_bool,1),
  ('lower_boolean_shuffle', ctypes.c_bool,1),
]
nir_lower_subgroups_options = struct_nir_lower_subgroups_options
try: (nir_lower_subgroups:=dll.nir_lower_subgroups).restype, nir_lower_subgroups.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_subgroups_options)]
except AttributeError: pass

try: (nir_lower_system_values:=dll.nir_lower_system_values).restype, nir_lower_system_values.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_build_lowered_load_helper_invocation:=dll.nir_build_lowered_load_helper_invocation).restype, nir_build_lowered_load_helper_invocation.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder)]
except AttributeError: pass

class struct_nir_lower_compute_system_values_options(Struct): pass
struct_nir_lower_compute_system_values_options._fields_ = [
  ('has_base_global_invocation_id', ctypes.c_bool,1),
  ('has_base_workgroup_id', ctypes.c_bool,1),
  ('has_global_size', ctypes.c_bool,1),
  ('shuffle_local_ids_for_quad_derivatives', ctypes.c_bool,1),
  ('lower_local_invocation_index', ctypes.c_bool,1),
  ('lower_cs_local_id_to_index', ctypes.c_bool,1),
  ('lower_workgroup_id_to_index', ctypes.c_bool,1),
  ('global_id_is_32bit', ctypes.c_bool,1),
  ('shortcut_1d_workgroup_id', ctypes.c_bool,1),
  ('num_workgroups', (uint32_t * 3)),
]
nir_lower_compute_system_values_options = struct_nir_lower_compute_system_values_options
try: (nir_lower_compute_system_values:=dll.nir_lower_compute_system_values).restype, nir_lower_compute_system_values.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_compute_system_values_options)]
except AttributeError: pass

class struct_nir_lower_sysvals_to_varyings_options(Struct): pass
struct_nir_lower_sysvals_to_varyings_options._fields_ = [
  ('frag_coord', ctypes.c_bool,1),
  ('front_face', ctypes.c_bool,1),
  ('point_coord', ctypes.c_bool,1),
]
nir_lower_sysvals_to_varyings_options = struct_nir_lower_sysvals_to_varyings_options
try: (nir_lower_sysvals_to_varyings:=dll.nir_lower_sysvals_to_varyings).restype, nir_lower_sysvals_to_varyings.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_sysvals_to_varyings_options)]
except AttributeError: pass

enum_nir_lower_tex_packing = CEnum(ctypes.c_ubyte)
nir_lower_tex_packing_none = enum_nir_lower_tex_packing.define('nir_lower_tex_packing_none', 0)
nir_lower_tex_packing_16 = enum_nir_lower_tex_packing.define('nir_lower_tex_packing_16', 1)
nir_lower_tex_packing_8 = enum_nir_lower_tex_packing.define('nir_lower_tex_packing_8', 2)

class struct_nir_lower_tex_options(Struct): pass
struct_nir_lower_tex_options._fields_ = [
  ('lower_txp', ctypes.c_uint32),
  ('lower_txp_array', ctypes.c_bool),
  ('lower_txf_offset', ctypes.c_bool),
  ('lower_rect_offset', ctypes.c_bool),
  ('lower_offset_filter', nir_instr_filter_cb),
  ('lower_rect', ctypes.c_bool),
  ('lower_1d', ctypes.c_bool),
  ('lower_1d_shadow', ctypes.c_bool),
  ('lower_y_uv_external', ctypes.c_uint32),
  ('lower_y_vu_external', ctypes.c_uint32),
  ('lower_y_u_v_external', ctypes.c_uint32),
  ('lower_yx_xuxv_external', ctypes.c_uint32),
  ('lower_yx_xvxu_external', ctypes.c_uint32),
  ('lower_xy_uxvx_external', ctypes.c_uint32),
  ('lower_xy_vxux_external', ctypes.c_uint32),
  ('lower_ayuv_external', ctypes.c_uint32),
  ('lower_xyuv_external', ctypes.c_uint32),
  ('lower_yuv_external', ctypes.c_uint32),
  ('lower_yu_yv_external', ctypes.c_uint32),
  ('lower_yv_yu_external', ctypes.c_uint32),
  ('lower_y41x_external', ctypes.c_uint32),
  ('lower_sx10_external', ctypes.c_uint32),
  ('lower_sx12_external', ctypes.c_uint32),
  ('bt709_external', ctypes.c_uint32),
  ('bt2020_external', ctypes.c_uint32),
  ('yuv_full_range_external', ctypes.c_uint32),
  ('saturate_s', ctypes.c_uint32),
  ('saturate_t', ctypes.c_uint32),
  ('saturate_r', ctypes.c_uint32),
  ('swizzle_result', ctypes.c_uint32),
  ('swizzles', ((uint8_t * 4) * 32)),
  ('scale_factors', (ctypes.c_float * 32)),
  ('lower_srgb', ctypes.c_uint32),
  ('lower_txd_cube_map', ctypes.c_bool),
  ('lower_txd_3d', ctypes.c_bool),
  ('lower_txd_array', ctypes.c_bool),
  ('lower_txd_shadow', ctypes.c_bool),
  ('lower_txd', ctypes.c_bool),
  ('lower_txd_clamp', ctypes.c_bool),
  ('lower_txb_shadow_clamp', ctypes.c_bool),
  ('lower_txd_shadow_clamp', ctypes.c_bool),
  ('lower_txd_offset_clamp', ctypes.c_bool),
  ('lower_txd_clamp_bindless_sampler', ctypes.c_bool),
  ('lower_txd_clamp_if_sampler_index_not_lt_16', ctypes.c_bool),
  ('lower_txs_lod', ctypes.c_bool),
  ('lower_txs_cube_array', ctypes.c_bool),
  ('lower_tg4_broadcom_swizzle', ctypes.c_bool),
  ('lower_tg4_offsets', ctypes.c_bool),
  ('lower_to_fragment_fetch_amd', ctypes.c_bool),
  ('lower_tex_packing_cb', ctypes.CFUNCTYPE(enum_nir_lower_tex_packing, ctypes.POINTER(nir_tex_instr), ctypes.c_void_p)),
  ('lower_tex_packing_data', ctypes.c_void_p),
  ('lower_lod_zero_width', ctypes.c_bool),
  ('lower_sampler_lod_bias', ctypes.c_bool),
  ('lower_invalid_implicit_lod', ctypes.c_bool),
  ('lower_index_to_offset', ctypes.c_bool),
  ('callback_data', ctypes.c_void_p),
]
nir_lower_tex_options = struct_nir_lower_tex_options
try: (nir_lower_tex:=dll.nir_lower_tex).restype, nir_lower_tex.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_tex_options)]
except AttributeError: pass

class struct_nir_lower_tex_shadow_swizzle(Struct): pass
struct_nir_lower_tex_shadow_swizzle._fields_ = [
  ('swizzle_r', ctypes.c_uint32,3),
  ('swizzle_g', ctypes.c_uint32,3),
  ('swizzle_b', ctypes.c_uint32,3),
  ('swizzle_a', ctypes.c_uint32,3),
]
nir_lower_tex_shadow_swizzle = struct_nir_lower_tex_shadow_swizzle
try: (nir_lower_tex_shadow:=dll.nir_lower_tex_shadow).restype, nir_lower_tex_shadow.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.POINTER(enum_compare_func), ctypes.POINTER(nir_lower_tex_shadow_swizzle), ctypes.c_bool]
except AttributeError: pass

class struct_nir_lower_image_options(Struct): pass
struct_nir_lower_image_options._fields_ = [
  ('lower_cube_size', ctypes.c_bool),
  ('lower_to_fragment_mask_load_amd', ctypes.c_bool),
  ('lower_image_samples_to_one', ctypes.c_bool),
]
nir_lower_image_options = struct_nir_lower_image_options
try: (nir_lower_image:=dll.nir_lower_image).restype, nir_lower_image.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_image_options)]
except AttributeError: pass

try: (nir_lower_image_atomics_to_global:=dll.nir_lower_image_atomics_to_global).restype, nir_lower_image_atomics_to_global.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_intrin_filter_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_readonly_images_to_tex:=dll.nir_lower_readonly_images_to_tex).restype, nir_lower_readonly_images_to_tex.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

enum_nir_lower_non_uniform_access_type = CEnum(ctypes.c_uint32)
nir_lower_non_uniform_ubo_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_ubo_access', 1)
nir_lower_non_uniform_ssbo_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_ssbo_access', 2)
nir_lower_non_uniform_texture_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_texture_access', 4)
nir_lower_non_uniform_image_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_image_access', 8)
nir_lower_non_uniform_get_ssbo_size = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_get_ssbo_size', 16)
nir_lower_non_uniform_texture_offset_access = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_texture_offset_access', 32)
nir_lower_non_uniform_access_type_count = enum_nir_lower_non_uniform_access_type.define('nir_lower_non_uniform_access_type_count', 6)

class const_struct_nir_tex_instr(Struct): pass
const_struct_nir_tex_instr._fields_ = [
  ('instr', nir_instr),
  ('sampler_dim', enum_glsl_sampler_dim),
  ('dest_type', nir_alu_type),
  ('op', nir_texop),
  ('def', nir_def),
  ('src', ctypes.POINTER(nir_tex_src)),
  ('num_srcs', ctypes.c_uint32),
  ('coord_components', ctypes.c_uint32),
  ('is_array', ctypes.c_bool),
  ('is_shadow', ctypes.c_bool),
  ('is_new_style_shadow', ctypes.c_bool),
  ('is_sparse', ctypes.c_bool),
  ('component', ctypes.c_uint32,2),
  ('array_is_lowered_cube', ctypes.c_uint32,1),
  ('is_gather_implicit_lod', ctypes.c_uint32,1),
  ('skip_helpers', ctypes.c_uint32,1),
  ('tg4_offsets', ((int8_t * 2) * 4)),
  ('texture_non_uniform', ctypes.c_bool),
  ('sampler_non_uniform', ctypes.c_bool),
  ('offset_non_uniform', ctypes.c_bool),
  ('texture_index', ctypes.c_uint32),
  ('sampler_index', ctypes.c_uint32),
  ('backend_flags', uint32_t),
]
nir_lower_non_uniform_src_access_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(const_struct_nir_tex_instr), ctypes.c_uint32, ctypes.c_void_p)
class const_struct_nir_src(Struct): pass
const_struct_nir_src._fields_ = [
  ('_parent', uintptr_t),
  ('use_link', struct_list_head),
  ('ssa', ctypes.POINTER(nir_def)),
]
nir_lower_non_uniform_access_callback = ctypes.CFUNCTYPE(ctypes.c_uint16, ctypes.POINTER(const_struct_nir_src), ctypes.c_void_p)
class struct_nir_lower_non_uniform_access_options(Struct): pass
struct_nir_lower_non_uniform_access_options._fields_ = [
  ('types', enum_nir_lower_non_uniform_access_type),
  ('tex_src_callback', nir_lower_non_uniform_src_access_callback),
  ('callback', nir_lower_non_uniform_access_callback),
  ('callback_data', ctypes.c_void_p),
]
nir_lower_non_uniform_access_options = struct_nir_lower_non_uniform_access_options
try: (nir_has_non_uniform_access:=dll.nir_has_non_uniform_access).restype, nir_has_non_uniform_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), enum_nir_lower_non_uniform_access_type]
except AttributeError: pass

try: (nir_opt_non_uniform_access:=dll.nir_opt_non_uniform_access).restype, nir_opt_non_uniform_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_non_uniform_access:=dll.nir_lower_non_uniform_access).restype, nir_lower_non_uniform_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_non_uniform_access_options)]
except AttributeError: pass

class struct_nir_lower_idiv_options(Struct): pass
struct_nir_lower_idiv_options._fields_ = [
  ('allow_fp16', ctypes.c_bool),
]
nir_lower_idiv_options = struct_nir_lower_idiv_options
try: (nir_lower_idiv:=dll.nir_lower_idiv).restype, nir_lower_idiv.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_idiv_options)]
except AttributeError: pass

class struct_nir_input_attachment_options(Struct): pass
struct_nir_input_attachment_options._fields_ = [
  ('use_ia_coord_intrin', ctypes.c_bool),
  ('use_fragcoord_sysval', ctypes.c_bool),
  ('use_layer_id_sysval', ctypes.c_bool),
  ('use_view_id_for_layer', ctypes.c_bool),
  ('unscaled_depth_stencil_ir3', ctypes.c_bool),
  ('unscaled_input_attachment_ir3', uint32_t),
]
nir_input_attachment_options = struct_nir_input_attachment_options
try: (nir_lower_input_attachments:=dll.nir_lower_input_attachments).restype, nir_lower_input_attachments.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_input_attachment_options)]
except AttributeError: pass

try: (nir_lower_clip_vs:=dll.nir_lower_clip_vs).restype, nir_lower_clip_vs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, ((gl_state_index16 * 4) * 0)]
except AttributeError: pass

try: (nir_lower_clip_gs:=dll.nir_lower_clip_gs).restype, nir_lower_clip_gs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool, ((gl_state_index16 * 4) * 0)]
except AttributeError: pass

try: (nir_lower_clip_fs:=dll.nir_lower_clip_fs).restype, nir_lower_clip_fs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_clip_cull_distance_to_vec4s:=dll.nir_lower_clip_cull_distance_to_vec4s).restype, nir_lower_clip_cull_distance_to_vec4s.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_clip_cull_distance_array_vars:=dll.nir_lower_clip_cull_distance_array_vars).restype, nir_lower_clip_cull_distance_array_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_clip_disable:=dll.nir_lower_clip_disable).restype, nir_lower_clip_disable.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_lower_point_size_mov:=dll.nir_lower_point_size_mov).restype, nir_lower_point_size_mov.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(gl_state_index16)]
except AttributeError: pass

try: (nir_lower_frexp:=dll.nir_lower_frexp).restype, nir_lower_frexp.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_two_sided_color:=dll.nir_lower_two_sided_color).restype, nir_lower_two_sided_color.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_clamp_color_outputs:=dll.nir_lower_clamp_color_outputs).restype, nir_lower_clamp_color_outputs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_flatshade:=dll.nir_lower_flatshade).restype, nir_lower_flatshade.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_passthrough_edgeflags:=dll.nir_lower_passthrough_edgeflags).restype, nir_lower_passthrough_edgeflags.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_patch_vertices:=dll.nir_lower_patch_vertices).restype, nir_lower_patch_vertices.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.POINTER(gl_state_index16)]
except AttributeError: pass

class struct_nir_lower_wpos_ytransform_options(Struct): pass
struct_nir_lower_wpos_ytransform_options._fields_ = [
  ('state_tokens', (gl_state_index16 * 4)),
  ('fs_coord_origin_upper_left', ctypes.c_bool,1),
  ('fs_coord_origin_lower_left', ctypes.c_bool,1),
  ('fs_coord_pixel_center_integer', ctypes.c_bool,1),
  ('fs_coord_pixel_center_half_integer', ctypes.c_bool,1),
]
nir_lower_wpos_ytransform_options = struct_nir_lower_wpos_ytransform_options
try: (nir_lower_wpos_ytransform:=dll.nir_lower_wpos_ytransform).restype, nir_lower_wpos_ytransform.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_wpos_ytransform_options)]
except AttributeError: pass

try: (nir_lower_wpos_center:=dll.nir_lower_wpos_center).restype, nir_lower_wpos_center.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_pntc_ytransform:=dll.nir_lower_pntc_ytransform).restype, nir_lower_pntc_ytransform.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ((gl_state_index16 * 4) * 0)]
except AttributeError: pass

try: (nir_lower_wrmasks:=dll.nir_lower_wrmasks).restype, nir_lower_wrmasks.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_instr_filter_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_fb_read:=dll.nir_lower_fb_read).restype, nir_lower_fb_read.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_lower_drawpixels_options(Struct): pass
struct_nir_lower_drawpixels_options._fields_ = [
  ('texcoord_state_tokens', (gl_state_index16 * 4)),
  ('scale_state_tokens', (gl_state_index16 * 4)),
  ('bias_state_tokens', (gl_state_index16 * 4)),
  ('drawpix_sampler', ctypes.c_uint32),
  ('pixelmap_sampler', ctypes.c_uint32),
  ('pixel_maps', ctypes.c_bool,1),
  ('scale_and_bias', ctypes.c_bool,1),
]
nir_lower_drawpixels_options = struct_nir_lower_drawpixels_options
try: (nir_lower_drawpixels:=dll.nir_lower_drawpixels).restype, nir_lower_drawpixels.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_drawpixels_options)]
except AttributeError: pass

class struct_nir_lower_bitmap_options(Struct): pass
struct_nir_lower_bitmap_options._fields_ = [
  ('sampler', ctypes.c_uint32),
  ('swizzle_xxxx', ctypes.c_bool),
]
nir_lower_bitmap_options = struct_nir_lower_bitmap_options
try: (nir_lower_bitmap:=dll.nir_lower_bitmap).restype, nir_lower_bitmap.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_bitmap_options)]
except AttributeError: pass

try: (nir_lower_atomics_to_ssbo:=dll.nir_lower_atomics_to_ssbo).restype, nir_lower_atomics_to_ssbo.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

nir_lower_gs_intrinsics_flags = CEnum(ctypes.c_uint32)
nir_lower_gs_intrinsics_per_stream = nir_lower_gs_intrinsics_flags.define('nir_lower_gs_intrinsics_per_stream', 1)
nir_lower_gs_intrinsics_count_primitives = nir_lower_gs_intrinsics_flags.define('nir_lower_gs_intrinsics_count_primitives', 2)
nir_lower_gs_intrinsics_count_vertices_per_primitive = nir_lower_gs_intrinsics_flags.define('nir_lower_gs_intrinsics_count_vertices_per_primitive', 4)
nir_lower_gs_intrinsics_overwrite_incomplete = nir_lower_gs_intrinsics_flags.define('nir_lower_gs_intrinsics_overwrite_incomplete', 8)

try: (nir_lower_gs_intrinsics:=dll.nir_lower_gs_intrinsics).restype, nir_lower_gs_intrinsics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_gs_intrinsics_flags]
except AttributeError: pass

try: (nir_lower_halt_to_return:=dll.nir_lower_halt_to_return).restype, nir_lower_halt_to_return.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_tess_coord_z:=dll.nir_lower_tess_coord_z).restype, nir_lower_tess_coord_z.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

class struct_nir_lower_task_shader_options(Struct): pass
struct_nir_lower_task_shader_options._fields_ = [
  ('payload_to_shared_for_atomics', ctypes.c_bool,1),
  ('payload_to_shared_for_small_types', ctypes.c_bool,1),
  ('payload_offset_in_bytes', uint32_t),
]
nir_lower_task_shader_options = struct_nir_lower_task_shader_options
try: (nir_lower_task_shader:=dll.nir_lower_task_shader).restype, nir_lower_task_shader.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_task_shader_options]
except AttributeError: pass

nir_lower_bit_size_callback = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(const_struct_nir_instr), ctypes.c_void_p)
try: (nir_lower_bit_size:=dll.nir_lower_bit_size).restype, nir_lower_bit_size.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_bit_size_callback, ctypes.c_void_p]
except AttributeError: pass

try: (nir_lower_64bit_phis:=dll.nir_lower_64bit_phis).restype, nir_lower_64bit_phis.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_split_conversions_options(Struct): pass
struct_nir_split_conversions_options._fields_ = [
  ('callback', nir_lower_bit_size_callback),
  ('callback_data', ctypes.c_void_p),
  ('has_convert_alu_types', ctypes.c_bool),
]
nir_split_conversions_options = struct_nir_split_conversions_options
try: (nir_split_conversions:=dll.nir_split_conversions).restype, nir_split_conversions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_split_conversions_options)]
except AttributeError: pass

try: (nir_split_64bit_vec3_and_vec4:=dll.nir_split_64bit_vec3_and_vec4).restype, nir_split_64bit_vec3_and_vec4.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_int64_op_to_options_mask:=dll.nir_lower_int64_op_to_options_mask).restype, nir_lower_int64_op_to_options_mask.argtypes = nir_lower_int64_options, [nir_op]
except AttributeError: pass

try: (nir_lower_int64:=dll.nir_lower_int64).restype, nir_lower_int64.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_int64_float_conversions:=dll.nir_lower_int64_float_conversions).restype, nir_lower_int64_float_conversions.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_doubles_op_to_options_mask:=dll.nir_lower_doubles_op_to_options_mask).restype, nir_lower_doubles_op_to_options_mask.argtypes = nir_lower_doubles_options, [nir_op]
except AttributeError: pass

try: (nir_lower_doubles:=dll.nir_lower_doubles).restype, nir_lower_doubles.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_shader), nir_lower_doubles_options]
except AttributeError: pass

try: (nir_lower_pack:=dll.nir_lower_pack).restype, nir_lower_pack.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_get_io_intrinsic:=dll.nir_get_io_intrinsic).restype, nir_get_io_intrinsic.argtypes = ctypes.POINTER(nir_intrinsic_instr), [ctypes.POINTER(nir_instr), nir_variable_mode, ctypes.POINTER(nir_variable_mode)]
except AttributeError: pass

try: (nir_recompute_io_bases:=dll.nir_recompute_io_bases).restype, nir_recompute_io_bases.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_mediump_vars:=dll.nir_lower_mediump_vars).restype, nir_lower_mediump_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_lower_mediump_io:=dll.nir_lower_mediump_io).restype, nir_lower_mediump_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, uint64_t, ctypes.c_bool]
except AttributeError: pass

try: (nir_clear_mediump_io_flag:=dll.nir_clear_mediump_io_flag).restype, nir_clear_mediump_io_flag.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_opt_tex_srcs_options(Struct): pass
struct_nir_opt_tex_srcs_options._fields_ = [
  ('sampler_dims', ctypes.c_uint32),
  ('src_types', ctypes.c_uint32),
]
nir_opt_tex_srcs_options = struct_nir_opt_tex_srcs_options
class struct_nir_opt_16bit_tex_image_options(Struct): pass
struct_nir_opt_16bit_tex_image_options._fields_ = [
  ('rounding_mode', nir_rounding_mode),
  ('opt_tex_dest_types', nir_alu_type),
  ('opt_image_dest_types', nir_alu_type),
  ('integer_dest_saturates', ctypes.c_bool),
  ('opt_image_store_data', ctypes.c_bool),
  ('opt_image_srcs', ctypes.c_bool),
  ('opt_srcs_options_count', ctypes.c_uint32),
  ('opt_srcs_options', ctypes.POINTER(nir_opt_tex_srcs_options)),
]
nir_opt_16bit_tex_image_options = struct_nir_opt_16bit_tex_image_options
try: (nir_opt_16bit_tex_image:=dll.nir_opt_16bit_tex_image).restype, nir_opt_16bit_tex_image.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_16bit_tex_image_options)]
except AttributeError: pass

class struct_nir_tex_src_type_constraint(Struct): pass
struct_nir_tex_src_type_constraint._fields_ = [
  ('legalize_type', ctypes.c_bool),
  ('bit_size', uint8_t),
  ('match_src', nir_tex_src_type),
]
nir_tex_src_type_constraint = struct_nir_tex_src_type_constraint
nir_tex_src_type_constraints = (struct_nir_tex_src_type_constraint * 23)
try: (nir_legalize_16bit_sampler_srcs:=dll.nir_legalize_16bit_sampler_srcs).restype, nir_legalize_16bit_sampler_srcs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_tex_src_type_constraints]
except AttributeError: pass

try: (nir_lower_point_size:=dll.nir_lower_point_size).restype, nir_lower_point_size.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_float, ctypes.c_float]
except AttributeError: pass

try: (nir_lower_default_point_size:=dll.nir_lower_default_point_size).restype, nir_lower_default_point_size.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_texcoord_replace:=dll.nir_lower_texcoord_replace).restype, nir_lower_texcoord_replace.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_texcoord_replace_late:=dll.nir_lower_texcoord_replace_late).restype, nir_lower_texcoord_replace_late.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32, ctypes.c_bool]
except AttributeError: pass

nir_lower_interpolation_options = CEnum(ctypes.c_uint32)
nir_lower_interpolation_at_sample = nir_lower_interpolation_options.define('nir_lower_interpolation_at_sample', 2)
nir_lower_interpolation_at_offset = nir_lower_interpolation_options.define('nir_lower_interpolation_at_offset', 4)
nir_lower_interpolation_centroid = nir_lower_interpolation_options.define('nir_lower_interpolation_centroid', 8)
nir_lower_interpolation_pixel = nir_lower_interpolation_options.define('nir_lower_interpolation_pixel', 16)
nir_lower_interpolation_sample = nir_lower_interpolation_options.define('nir_lower_interpolation_sample', 32)

try: (nir_lower_interpolation:=dll.nir_lower_interpolation).restype, nir_lower_interpolation.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_interpolation_options]
except AttributeError: pass

nir_lower_discard_if_options = CEnum(ctypes.c_uint32)
nir_lower_demote_if_to_cf = nir_lower_discard_if_options.define('nir_lower_demote_if_to_cf', 1)
nir_lower_terminate_if_to_cf = nir_lower_discard_if_options.define('nir_lower_terminate_if_to_cf', 2)
nir_move_terminate_out_of_loops = nir_lower_discard_if_options.define('nir_move_terminate_out_of_loops', 4)

try: (nir_lower_discard_if:=dll.nir_lower_discard_if).restype, nir_lower_discard_if.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_discard_if_options]
except AttributeError: pass

try: (nir_lower_terminate_to_demote:=dll.nir_lower_terminate_to_demote).restype, nir_lower_terminate_to_demote.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_memory_model:=dll.nir_lower_memory_model).restype, nir_lower_memory_model.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_goto_ifs:=dll.nir_lower_goto_ifs).restype, nir_lower_goto_ifs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_continue_constructs:=dll.nir_lower_continue_constructs).restype, nir_lower_continue_constructs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_lower_multiview_options(Struct): pass
struct_nir_lower_multiview_options._fields_ = [
  ('view_mask', uint32_t),
  ('allowed_per_view_outputs', uint64_t),
]
nir_lower_multiview_options = struct_nir_lower_multiview_options
try: (nir_shader_uses_view_index:=dll.nir_shader_uses_view_index).restype, nir_shader_uses_view_index.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_can_lower_multiview:=dll.nir_can_lower_multiview).restype, nir_can_lower_multiview.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_multiview_options]
except AttributeError: pass

try: (nir_lower_multiview:=dll.nir_lower_multiview).restype, nir_lower_multiview.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_multiview_options]
except AttributeError: pass

try: (nir_lower_view_index_to_device_index:=dll.nir_lower_view_index_to_device_index).restype, nir_lower_view_index_to_device_index.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_lower_fp16_cast_options = CEnum(ctypes.c_uint32)
nir_lower_fp16_rtz = nir_lower_fp16_cast_options.define('nir_lower_fp16_rtz', 1)
nir_lower_fp16_rtne = nir_lower_fp16_cast_options.define('nir_lower_fp16_rtne', 2)
nir_lower_fp16_ru = nir_lower_fp16_cast_options.define('nir_lower_fp16_ru', 4)
nir_lower_fp16_rd = nir_lower_fp16_cast_options.define('nir_lower_fp16_rd', 8)
nir_lower_fp16_all = nir_lower_fp16_cast_options.define('nir_lower_fp16_all', 15)
nir_lower_fp16_split_fp64 = nir_lower_fp16_cast_options.define('nir_lower_fp16_split_fp64', 16)

try: (nir_lower_fp16_casts:=dll.nir_lower_fp16_casts).restype, nir_lower_fp16_casts.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_lower_fp16_cast_options]
except AttributeError: pass

try: (nir_normalize_cubemap_coords:=dll.nir_normalize_cubemap_coords).restype, nir_normalize_cubemap_coords.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_shader_supports_implicit_lod:=dll.nir_shader_supports_implicit_lod).restype, nir_shader_supports_implicit_lod.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_live_defs_impl:=dll.nir_live_defs_impl).restype, nir_live_defs_impl.argtypes = None, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_get_live_defs:=dll.nir_get_live_defs).restype, nir_get_live_defs.argtypes = ctypes.POINTER(ctypes.c_uint32), [nir_cursor, ctypes.c_void_p]
except AttributeError: pass

try: (nir_loop_analyze_impl:=dll.nir_loop_analyze_impl).restype, nir_loop_analyze_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), nir_variable_mode, ctypes.c_bool]
except AttributeError: pass

try: (nir_defs_interfere:=dll.nir_defs_interfere).restype, nir_defs_interfere.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_repair_ssa_impl:=dll.nir_repair_ssa_impl).restype, nir_repair_ssa_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_repair_ssa:=dll.nir_repair_ssa).restype, nir_repair_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_convert_loop_to_lcssa:=dll.nir_convert_loop_to_lcssa).restype, nir_convert_loop_to_lcssa.argtypes = None, [ctypes.POINTER(nir_loop)]
except AttributeError: pass

try: (nir_convert_to_lcssa:=dll.nir_convert_to_lcssa).restype, nir_convert_to_lcssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_divergence_analysis_impl:=dll.nir_divergence_analysis_impl).restype, nir_divergence_analysis_impl.argtypes = None, [ctypes.POINTER(nir_function_impl), nir_divergence_options]
except AttributeError: pass

try: (nir_divergence_analysis:=dll.nir_divergence_analysis).restype, nir_divergence_analysis.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_vertex_divergence_analysis:=dll.nir_vertex_divergence_analysis).restype, nir_vertex_divergence_analysis.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_has_divergent_loop:=dll.nir_has_divergent_loop).restype, nir_has_divergent_loop.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_rewrite_uses_to_load_reg:=dll.nir_rewrite_uses_to_load_reg).restype, nir_rewrite_uses_to_load_reg.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_convert_from_ssa:=dll.nir_convert_from_ssa).restype, nir_convert_from_ssa.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_phis_to_regs_block:=dll.nir_lower_phis_to_regs_block).restype, nir_lower_phis_to_regs_block.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_ssa_defs_to_regs_block:=dll.nir_lower_ssa_defs_to_regs_block).restype, nir_lower_ssa_defs_to_regs_block.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_rematerialize_deref_in_use_blocks:=dll.nir_rematerialize_deref_in_use_blocks).restype, nir_rematerialize_deref_in_use_blocks.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_deref_instr)]
except AttributeError: pass

try: (nir_rematerialize_derefs_in_use_blocks_impl:=dll.nir_rematerialize_derefs_in_use_blocks_impl).restype, nir_rematerialize_derefs_in_use_blocks_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_lower_samplers:=dll.nir_lower_samplers).restype, nir_lower_samplers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_cl_images:=dll.nir_lower_cl_images).restype, nir_lower_cl_images.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (nir_dedup_inline_samplers:=dll.nir_dedup_inline_samplers).restype, nir_dedup_inline_samplers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_lower_ssbo_options(Struct): pass
struct_nir_lower_ssbo_options._fields_ = [
  ('native_loads', ctypes.c_bool),
  ('native_offset', ctypes.c_bool),
]
nir_lower_ssbo_options = struct_nir_lower_ssbo_options
try: (nir_lower_ssbo:=dll.nir_lower_ssbo).restype, nir_lower_ssbo.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_ssbo_options)]
except AttributeError: pass

try: (nir_lower_helper_writes:=dll.nir_lower_helper_writes).restype, nir_lower_helper_writes.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

class struct_nir_lower_printf_options(Struct): pass
struct_nir_lower_printf_options._fields_ = [
  ('max_buffer_size', ctypes.c_uint32),
  ('ptr_bit_size', ctypes.c_uint32),
  ('hash_format_strings', ctypes.c_bool),
]
nir_lower_printf_options = struct_nir_lower_printf_options
try: (nir_lower_printf:=dll.nir_lower_printf).restype, nir_lower_printf.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_printf_options)]
except AttributeError: pass

try: (nir_lower_printf_buffer:=dll.nir_lower_printf_buffer).restype, nir_lower_printf_buffer.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), uint64_t, uint32_t]
except AttributeError: pass

try: (nir_opt_comparison_pre_impl:=dll.nir_opt_comparison_pre_impl).restype, nir_opt_comparison_pre_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_opt_comparison_pre:=dll.nir_opt_comparison_pre).restype, nir_opt_comparison_pre.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_opt_access_options(Struct): pass
struct_nir_opt_access_options._fields_ = [
  ('is_vulkan', ctypes.c_bool),
]
nir_opt_access_options = struct_nir_opt_access_options
try: (nir_opt_access:=dll.nir_opt_access).restype, nir_opt_access.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_access_options)]
except AttributeError: pass

try: (nir_opt_algebraic:=dll.nir_opt_algebraic).restype, nir_opt_algebraic.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_before_ffma:=dll.nir_opt_algebraic_before_ffma).restype, nir_opt_algebraic_before_ffma.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_before_lower_int64:=dll.nir_opt_algebraic_before_lower_int64).restype, nir_opt_algebraic_before_lower_int64.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_late:=dll.nir_opt_algebraic_late).restype, nir_opt_algebraic_late.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_distribute_src_mods:=dll.nir_opt_algebraic_distribute_src_mods).restype, nir_opt_algebraic_distribute_src_mods.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_algebraic_integer_promotion:=dll.nir_opt_algebraic_integer_promotion).restype, nir_opt_algebraic_integer_promotion.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_reassociate_matrix_mul:=dll.nir_opt_reassociate_matrix_mul).restype, nir_opt_reassociate_matrix_mul.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_constant_folding:=dll.nir_opt_constant_folding).restype, nir_opt_constant_folding.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_combine_barrier_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_void_p)
try: (nir_opt_combine_barriers:=dll.nir_opt_combine_barriers).restype, nir_opt_combine_barriers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_combine_barrier_cb, ctypes.c_void_p]
except AttributeError: pass

mesa_scope = CEnum(ctypes.c_uint32)
SCOPE_NONE = mesa_scope.define('SCOPE_NONE', 0)
SCOPE_INVOCATION = mesa_scope.define('SCOPE_INVOCATION', 1)
SCOPE_SUBGROUP = mesa_scope.define('SCOPE_SUBGROUP', 2)
SCOPE_SHADER_CALL = mesa_scope.define('SCOPE_SHADER_CALL', 3)
SCOPE_WORKGROUP = mesa_scope.define('SCOPE_WORKGROUP', 4)
SCOPE_QUEUE_FAMILY = mesa_scope.define('SCOPE_QUEUE_FAMILY', 5)
SCOPE_DEVICE = mesa_scope.define('SCOPE_DEVICE', 6)

try: (nir_opt_acquire_release_barriers:=dll.nir_opt_acquire_release_barriers).restype, nir_opt_acquire_release_barriers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), mesa_scope]
except AttributeError: pass

try: (nir_opt_barrier_modes:=dll.nir_opt_barrier_modes).restype, nir_opt_barrier_modes.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_minimize_call_live_states:=dll.nir_minimize_call_live_states).restype, nir_minimize_call_live_states.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_combine_stores:=dll.nir_opt_combine_stores).restype, nir_opt_combine_stores.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode]
except AttributeError: pass

try: (nir_copy_prop_impl:=dll.nir_copy_prop_impl).restype, nir_copy_prop_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_copy_prop:=dll.nir_copy_prop).restype, nir_copy_prop.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_copy_prop_vars:=dll.nir_opt_copy_prop_vars).restype, nir_opt_copy_prop_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_cse:=dll.nir_opt_cse).restype, nir_opt_cse.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_dce:=dll.nir_opt_dce).restype, nir_opt_dce.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_dead_cf:=dll.nir_opt_dead_cf).restype, nir_opt_dead_cf.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_dead_write_vars:=dll.nir_opt_dead_write_vars).restype, nir_opt_dead_write_vars.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_deref_impl:=dll.nir_opt_deref_impl).restype, nir_opt_deref_impl.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_function_impl)]
except AttributeError: pass

try: (nir_opt_deref:=dll.nir_opt_deref).restype, nir_opt_deref.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_find_array_copies:=dll.nir_opt_find_array_copies).restype, nir_opt_find_array_copies.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_def_is_frag_coord_z:=dll.nir_def_is_frag_coord_z).restype, nir_def_is_frag_coord_z.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_opt_fragdepth:=dll.nir_opt_fragdepth).restype, nir_opt_fragdepth.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_gcm:=dll.nir_opt_gcm).restype, nir_opt_gcm.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_generate_bfi:=dll.nir_opt_generate_bfi).restype, nir_opt_generate_bfi.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_idiv_const:=dll.nir_opt_idiv_const).restype, nir_opt_idiv_const.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_opt_mqsad:=dll.nir_opt_mqsad).restype, nir_opt_mqsad.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_opt_if_options = CEnum(ctypes.c_uint32)
nir_opt_if_optimize_phi_true_false = nir_opt_if_options.define('nir_opt_if_optimize_phi_true_false', 1)
nir_opt_if_avoid_64bit_phis = nir_opt_if_options.define('nir_opt_if_avoid_64bit_phis', 2)

try: (nir_opt_if:=dll.nir_opt_if).restype, nir_opt_if.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_opt_if_options]
except AttributeError: pass

try: (nir_opt_intrinsics:=dll.nir_opt_intrinsics).restype, nir_opt_intrinsics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_large_constants:=dll.nir_opt_large_constants).restype, nir_opt_large_constants.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), glsl_type_size_align_func, ctypes.c_uint32]
except AttributeError: pass

try: (nir_opt_licm:=dll.nir_opt_licm).restype, nir_opt_licm.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_loop:=dll.nir_opt_loop).restype, nir_opt_loop.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_loop_unroll:=dll.nir_opt_loop_unroll).restype, nir_opt_loop_unroll.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

nir_move_options = CEnum(ctypes.c_uint32)
nir_move_const_undef = nir_move_options.define('nir_move_const_undef', 1)
nir_move_load_ubo = nir_move_options.define('nir_move_load_ubo', 2)
nir_move_load_input = nir_move_options.define('nir_move_load_input', 4)
nir_move_comparisons = nir_move_options.define('nir_move_comparisons', 8)
nir_move_copies = nir_move_options.define('nir_move_copies', 16)
nir_move_load_ssbo = nir_move_options.define('nir_move_load_ssbo', 32)
nir_move_load_uniform = nir_move_options.define('nir_move_load_uniform', 64)
nir_move_alu = nir_move_options.define('nir_move_alu', 128)
nir_dont_move_byte_word_vecs = nir_move_options.define('nir_dont_move_byte_word_vecs', 256)

try: (nir_can_move_instr:=dll.nir_can_move_instr).restype, nir_can_move_instr.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_instr), nir_move_options]
except AttributeError: pass

try: (nir_opt_sink:=dll.nir_opt_sink).restype, nir_opt_sink.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_move_options]
except AttributeError: pass

try: (nir_opt_move:=dll.nir_opt_move).restype, nir_opt_move.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_move_options]
except AttributeError: pass

class struct_nir_opt_offsets_options(Struct): pass
struct_nir_opt_offsets_options._fields_ = [
  ('uniform_max', uint32_t),
  ('ubo_vec4_max', uint32_t),
  ('shared_max', uint32_t),
  ('shared_atomic_max', uint32_t),
  ('buffer_max', uint32_t),
  ('max_offset_cb', ctypes.CFUNCTYPE(uint32_t, ctypes.POINTER(nir_intrinsic_instr), ctypes.c_void_p)),
  ('max_offset_data', ctypes.c_void_p),
  ('allow_offset_wrap', ctypes.c_bool),
]
nir_opt_offsets_options = struct_nir_opt_offsets_options
try: (nir_opt_offsets:=dll.nir_opt_offsets).restype, nir_opt_offsets.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_offsets_options)]
except AttributeError: pass

class struct_nir_opt_peephole_select_options(Struct): pass
struct_nir_opt_peephole_select_options._fields_ = [
  ('limit', ctypes.c_uint32),
  ('indirect_load_ok', ctypes.c_bool),
  ('expensive_alu_ok', ctypes.c_bool),
  ('discard_ok', ctypes.c_bool),
]
nir_opt_peephole_select_options = struct_nir_opt_peephole_select_options
try: (nir_opt_peephole_select:=dll.nir_opt_peephole_select).restype, nir_opt_peephole_select.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_peephole_select_options)]
except AttributeError: pass

try: (nir_opt_reassociate_bfi:=dll.nir_opt_reassociate_bfi).restype, nir_opt_reassociate_bfi.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_rematerialize_compares:=dll.nir_opt_rematerialize_compares).restype, nir_opt_rematerialize_compares.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_remove_phis:=dll.nir_opt_remove_phis).restype, nir_opt_remove_phis.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_remove_single_src_phis_block:=dll.nir_remove_single_src_phis_block).restype, nir_remove_single_src_phis_block.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_block)]
except AttributeError: pass

try: (nir_opt_phi_precision:=dll.nir_opt_phi_precision).restype, nir_opt_phi_precision.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_phi_to_bool:=dll.nir_opt_phi_to_bool).restype, nir_opt_phi_to_bool.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_shrink_stores:=dll.nir_opt_shrink_stores).restype, nir_opt_shrink_stores.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_shrink_vectors:=dll.nir_opt_shrink_vectors).restype, nir_opt_shrink_vectors.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_undef:=dll.nir_opt_undef).restype, nir_opt_undef.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_undef_to_zero:=dll.nir_lower_undef_to_zero).restype, nir_lower_undef_to_zero.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_uniform_atomics:=dll.nir_opt_uniform_atomics).restype, nir_opt_uniform_atomics.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_uniform_subgroup:=dll.nir_opt_uniform_subgroup).restype, nir_opt_uniform_subgroup.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_lower_subgroups_options)]
except AttributeError: pass

try: (nir_opt_vectorize:=dll.nir_opt_vectorize).restype, nir_opt_vectorize.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_vectorize_cb, ctypes.c_void_p]
except AttributeError: pass

try: (nir_opt_vectorize_io:=dll.nir_opt_vectorize_io).restype, nir_opt_vectorize_io.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), nir_variable_mode, ctypes.c_bool]
except AttributeError: pass

try: (nir_opt_move_discards_to_top:=dll.nir_opt_move_discards_to_top).restype, nir_opt_move_discards_to_top.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_ray_queries:=dll.nir_opt_ray_queries).restype, nir_opt_ray_queries.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_ray_query_ranges:=dll.nir_opt_ray_query_ranges).restype, nir_opt_ray_query_ranges.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_opt_tex_skip_helpers:=dll.nir_opt_tex_skip_helpers).restype, nir_opt_tex_skip_helpers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_sweep:=dll.nir_sweep).restype, nir_sweep.argtypes = None, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

gl_system_value = CEnum(ctypes.c_uint32)
SYSTEM_VALUE_SUBGROUP_SIZE = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_SIZE', 0)
SYSTEM_VALUE_SUBGROUP_INVOCATION = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_INVOCATION', 1)
SYSTEM_VALUE_SUBGROUP_EQ_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_EQ_MASK', 2)
SYSTEM_VALUE_SUBGROUP_GE_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_GE_MASK', 3)
SYSTEM_VALUE_SUBGROUP_GT_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_GT_MASK', 4)
SYSTEM_VALUE_SUBGROUP_LE_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_LE_MASK', 5)
SYSTEM_VALUE_SUBGROUP_LT_MASK = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_LT_MASK', 6)
SYSTEM_VALUE_NUM_SUBGROUPS = gl_system_value.define('SYSTEM_VALUE_NUM_SUBGROUPS', 7)
SYSTEM_VALUE_SUBGROUP_ID = gl_system_value.define('SYSTEM_VALUE_SUBGROUP_ID', 8)
SYSTEM_VALUE_VERTEX_ID = gl_system_value.define('SYSTEM_VALUE_VERTEX_ID', 9)
SYSTEM_VALUE_INSTANCE_ID = gl_system_value.define('SYSTEM_VALUE_INSTANCE_ID', 10)
SYSTEM_VALUE_INSTANCE_INDEX = gl_system_value.define('SYSTEM_VALUE_INSTANCE_INDEX', 11)
SYSTEM_VALUE_VERTEX_ID_ZERO_BASE = gl_system_value.define('SYSTEM_VALUE_VERTEX_ID_ZERO_BASE', 12)
SYSTEM_VALUE_BASE_VERTEX = gl_system_value.define('SYSTEM_VALUE_BASE_VERTEX', 13)
SYSTEM_VALUE_FIRST_VERTEX = gl_system_value.define('SYSTEM_VALUE_FIRST_VERTEX', 14)
SYSTEM_VALUE_IS_INDEXED_DRAW = gl_system_value.define('SYSTEM_VALUE_IS_INDEXED_DRAW', 15)
SYSTEM_VALUE_BASE_INSTANCE = gl_system_value.define('SYSTEM_VALUE_BASE_INSTANCE', 16)
SYSTEM_VALUE_DRAW_ID = gl_system_value.define('SYSTEM_VALUE_DRAW_ID', 17)
SYSTEM_VALUE_INVOCATION_ID = gl_system_value.define('SYSTEM_VALUE_INVOCATION_ID', 18)
SYSTEM_VALUE_FRAG_COORD = gl_system_value.define('SYSTEM_VALUE_FRAG_COORD', 19)
SYSTEM_VALUE_PIXEL_COORD = gl_system_value.define('SYSTEM_VALUE_PIXEL_COORD', 20)
SYSTEM_VALUE_FRAG_COORD_Z = gl_system_value.define('SYSTEM_VALUE_FRAG_COORD_Z', 21)
SYSTEM_VALUE_FRAG_COORD_W = gl_system_value.define('SYSTEM_VALUE_FRAG_COORD_W', 22)
SYSTEM_VALUE_POINT_COORD = gl_system_value.define('SYSTEM_VALUE_POINT_COORD', 23)
SYSTEM_VALUE_LINE_COORD = gl_system_value.define('SYSTEM_VALUE_LINE_COORD', 24)
SYSTEM_VALUE_FRONT_FACE = gl_system_value.define('SYSTEM_VALUE_FRONT_FACE', 25)
SYSTEM_VALUE_FRONT_FACE_FSIGN = gl_system_value.define('SYSTEM_VALUE_FRONT_FACE_FSIGN', 26)
SYSTEM_VALUE_SAMPLE_ID = gl_system_value.define('SYSTEM_VALUE_SAMPLE_ID', 27)
SYSTEM_VALUE_SAMPLE_POS = gl_system_value.define('SYSTEM_VALUE_SAMPLE_POS', 28)
SYSTEM_VALUE_SAMPLE_POS_OR_CENTER = gl_system_value.define('SYSTEM_VALUE_SAMPLE_POS_OR_CENTER', 29)
SYSTEM_VALUE_SAMPLE_MASK_IN = gl_system_value.define('SYSTEM_VALUE_SAMPLE_MASK_IN', 30)
SYSTEM_VALUE_LAYER_ID = gl_system_value.define('SYSTEM_VALUE_LAYER_ID', 31)
SYSTEM_VALUE_HELPER_INVOCATION = gl_system_value.define('SYSTEM_VALUE_HELPER_INVOCATION', 32)
SYSTEM_VALUE_COLOR0 = gl_system_value.define('SYSTEM_VALUE_COLOR0', 33)
SYSTEM_VALUE_COLOR1 = gl_system_value.define('SYSTEM_VALUE_COLOR1', 34)
SYSTEM_VALUE_TESS_COORD = gl_system_value.define('SYSTEM_VALUE_TESS_COORD', 35)
SYSTEM_VALUE_VERTICES_IN = gl_system_value.define('SYSTEM_VALUE_VERTICES_IN', 36)
SYSTEM_VALUE_PRIMITIVE_ID = gl_system_value.define('SYSTEM_VALUE_PRIMITIVE_ID', 37)
SYSTEM_VALUE_TESS_LEVEL_OUTER = gl_system_value.define('SYSTEM_VALUE_TESS_LEVEL_OUTER', 38)
SYSTEM_VALUE_TESS_LEVEL_INNER = gl_system_value.define('SYSTEM_VALUE_TESS_LEVEL_INNER', 39)
SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT = gl_system_value.define('SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT', 40)
SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT = gl_system_value.define('SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT', 41)
SYSTEM_VALUE_LOCAL_INVOCATION_ID = gl_system_value.define('SYSTEM_VALUE_LOCAL_INVOCATION_ID', 42)
SYSTEM_VALUE_LOCAL_INVOCATION_INDEX = gl_system_value.define('SYSTEM_VALUE_LOCAL_INVOCATION_INDEX', 43)
SYSTEM_VALUE_GLOBAL_INVOCATION_ID = gl_system_value.define('SYSTEM_VALUE_GLOBAL_INVOCATION_ID', 44)
SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID = gl_system_value.define('SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID', 45)
SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX = gl_system_value.define('SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX', 46)
SYSTEM_VALUE_WORKGROUP_ID = gl_system_value.define('SYSTEM_VALUE_WORKGROUP_ID', 47)
SYSTEM_VALUE_BASE_WORKGROUP_ID = gl_system_value.define('SYSTEM_VALUE_BASE_WORKGROUP_ID', 48)
SYSTEM_VALUE_WORKGROUP_INDEX = gl_system_value.define('SYSTEM_VALUE_WORKGROUP_INDEX', 49)
SYSTEM_VALUE_NUM_WORKGROUPS = gl_system_value.define('SYSTEM_VALUE_NUM_WORKGROUPS', 50)
SYSTEM_VALUE_WORKGROUP_SIZE = gl_system_value.define('SYSTEM_VALUE_WORKGROUP_SIZE', 51)
SYSTEM_VALUE_GLOBAL_GROUP_SIZE = gl_system_value.define('SYSTEM_VALUE_GLOBAL_GROUP_SIZE', 52)
SYSTEM_VALUE_WORK_DIM = gl_system_value.define('SYSTEM_VALUE_WORK_DIM', 53)
SYSTEM_VALUE_USER_DATA_AMD = gl_system_value.define('SYSTEM_VALUE_USER_DATA_AMD', 54)
SYSTEM_VALUE_DEVICE_INDEX = gl_system_value.define('SYSTEM_VALUE_DEVICE_INDEX', 55)
SYSTEM_VALUE_VIEW_INDEX = gl_system_value.define('SYSTEM_VALUE_VIEW_INDEX', 56)
SYSTEM_VALUE_VERTEX_CNT = gl_system_value.define('SYSTEM_VALUE_VERTEX_CNT', 57)
SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL', 58)
SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE', 59)
SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID', 60)
SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW', 61)
SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL', 62)
SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID', 63)
SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE', 64)
SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL', 65)
SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD', 66)
SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD = gl_system_value.define('SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD', 67)
SYSTEM_VALUE_RAY_LAUNCH_ID = gl_system_value.define('SYSTEM_VALUE_RAY_LAUNCH_ID', 68)
SYSTEM_VALUE_RAY_LAUNCH_SIZE = gl_system_value.define('SYSTEM_VALUE_RAY_LAUNCH_SIZE', 69)
SYSTEM_VALUE_RAY_WORLD_ORIGIN = gl_system_value.define('SYSTEM_VALUE_RAY_WORLD_ORIGIN', 70)
SYSTEM_VALUE_RAY_WORLD_DIRECTION = gl_system_value.define('SYSTEM_VALUE_RAY_WORLD_DIRECTION', 71)
SYSTEM_VALUE_RAY_OBJECT_ORIGIN = gl_system_value.define('SYSTEM_VALUE_RAY_OBJECT_ORIGIN', 72)
SYSTEM_VALUE_RAY_OBJECT_DIRECTION = gl_system_value.define('SYSTEM_VALUE_RAY_OBJECT_DIRECTION', 73)
SYSTEM_VALUE_RAY_T_MIN = gl_system_value.define('SYSTEM_VALUE_RAY_T_MIN', 74)
SYSTEM_VALUE_RAY_T_MAX = gl_system_value.define('SYSTEM_VALUE_RAY_T_MAX', 75)
SYSTEM_VALUE_RAY_OBJECT_TO_WORLD = gl_system_value.define('SYSTEM_VALUE_RAY_OBJECT_TO_WORLD', 76)
SYSTEM_VALUE_RAY_WORLD_TO_OBJECT = gl_system_value.define('SYSTEM_VALUE_RAY_WORLD_TO_OBJECT', 77)
SYSTEM_VALUE_RAY_HIT_KIND = gl_system_value.define('SYSTEM_VALUE_RAY_HIT_KIND', 78)
SYSTEM_VALUE_RAY_FLAGS = gl_system_value.define('SYSTEM_VALUE_RAY_FLAGS', 79)
SYSTEM_VALUE_RAY_GEOMETRY_INDEX = gl_system_value.define('SYSTEM_VALUE_RAY_GEOMETRY_INDEX', 80)
SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX = gl_system_value.define('SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX', 81)
SYSTEM_VALUE_CULL_MASK = gl_system_value.define('SYSTEM_VALUE_CULL_MASK', 82)
SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS = gl_system_value.define('SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS', 83)
SYSTEM_VALUE_MESH_VIEW_COUNT = gl_system_value.define('SYSTEM_VALUE_MESH_VIEW_COUNT', 84)
SYSTEM_VALUE_MESH_VIEW_INDICES = gl_system_value.define('SYSTEM_VALUE_MESH_VIEW_INDICES', 85)
SYSTEM_VALUE_GS_HEADER_IR3 = gl_system_value.define('SYSTEM_VALUE_GS_HEADER_IR3', 86)
SYSTEM_VALUE_TCS_HEADER_IR3 = gl_system_value.define('SYSTEM_VALUE_TCS_HEADER_IR3', 87)
SYSTEM_VALUE_REL_PATCH_ID_IR3 = gl_system_value.define('SYSTEM_VALUE_REL_PATCH_ID_IR3', 88)
SYSTEM_VALUE_FRAG_SHADING_RATE = gl_system_value.define('SYSTEM_VALUE_FRAG_SHADING_RATE', 89)
SYSTEM_VALUE_FULLY_COVERED = gl_system_value.define('SYSTEM_VALUE_FULLY_COVERED', 90)
SYSTEM_VALUE_FRAG_SIZE = gl_system_value.define('SYSTEM_VALUE_FRAG_SIZE', 91)
SYSTEM_VALUE_FRAG_INVOCATION_COUNT = gl_system_value.define('SYSTEM_VALUE_FRAG_INVOCATION_COUNT', 92)
SYSTEM_VALUE_SHADER_INDEX = gl_system_value.define('SYSTEM_VALUE_SHADER_INDEX', 93)
SYSTEM_VALUE_COALESCED_INPUT_COUNT = gl_system_value.define('SYSTEM_VALUE_COALESCED_INPUT_COUNT', 94)
SYSTEM_VALUE_WARPS_PER_SM_NV = gl_system_value.define('SYSTEM_VALUE_WARPS_PER_SM_NV', 95)
SYSTEM_VALUE_SM_COUNT_NV = gl_system_value.define('SYSTEM_VALUE_SM_COUNT_NV', 96)
SYSTEM_VALUE_WARP_ID_NV = gl_system_value.define('SYSTEM_VALUE_WARP_ID_NV', 97)
SYSTEM_VALUE_SM_ID_NV = gl_system_value.define('SYSTEM_VALUE_SM_ID_NV', 98)
SYSTEM_VALUE_MAX = gl_system_value.define('SYSTEM_VALUE_MAX', 99)

try: (nir_intrinsic_from_system_value:=dll.nir_intrinsic_from_system_value).restype, nir_intrinsic_from_system_value.argtypes = nir_intrinsic_op, [gl_system_value]
except AttributeError: pass

try: (nir_system_value_from_intrinsic:=dll.nir_system_value_from_intrinsic).restype, nir_system_value_from_intrinsic.argtypes = gl_system_value, [nir_intrinsic_op]
except AttributeError: pass

class struct_nir_unsigned_upper_bound_config(Struct): pass
struct_nir_unsigned_upper_bound_config._fields_ = [
  ('min_subgroup_size', ctypes.c_uint32),
  ('max_subgroup_size', ctypes.c_uint32),
  ('max_workgroup_invocations', ctypes.c_uint32),
  ('max_workgroup_count', (ctypes.c_uint32 * 3)),
  ('max_workgroup_size', (ctypes.c_uint32 * 3)),
  ('vertex_attrib_max', (uint32_t * 32)),
]
nir_unsigned_upper_bound_config = struct_nir_unsigned_upper_bound_config
try: (nir_unsigned_upper_bound:=dll.nir_unsigned_upper_bound).restype, nir_unsigned_upper_bound.argtypes = uint32_t, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_hash_table), nir_scalar, ctypes.POINTER(nir_unsigned_upper_bound_config)]
except AttributeError: pass

try: (nir_addition_might_overflow:=dll.nir_addition_might_overflow).restype, nir_addition_might_overflow.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_hash_table), nir_scalar, ctypes.c_uint32, ctypes.POINTER(nir_unsigned_upper_bound_config)]
except AttributeError: pass

class struct_nir_opt_preamble_options(Struct): pass
struct_nir_opt_preamble_options._fields_ = [
  ('drawid_uniform', ctypes.c_bool),
  ('subgroup_size_uniform', ctypes.c_bool),
  ('load_workgroup_size_allowed', ctypes.c_bool),
  ('def_size', ctypes.CFUNCTYPE(None, ctypes.POINTER(nir_def), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(nir_preamble_class))),
  ('preamble_storage_size', (ctypes.c_uint32 * 2)),
  ('instr_cost_cb', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(nir_instr), ctypes.c_void_p)),
  ('rewrite_cost_cb', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(nir_def), ctypes.c_void_p)),
  ('avoid_instr_cb', nir_instr_filter_cb),
  ('cb_data', ctypes.c_void_p),
]
nir_opt_preamble_options = struct_nir_opt_preamble_options
try: (nir_opt_preamble:=dll.nir_opt_preamble).restype, nir_opt_preamble.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_opt_preamble_options), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (nir_shader_get_preamble:=dll.nir_shader_get_preamble).restype, nir_shader_get_preamble.argtypes = ctypes.POINTER(nir_function_impl), [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_lower_point_smooth:=dll.nir_lower_point_smooth).restype, nir_lower_point_smooth.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

try: (nir_lower_poly_line_smooth:=dll.nir_lower_poly_line_smooth).restype, nir_lower_poly_line_smooth.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_mod_analysis:=dll.nir_mod_analysis).restype, nir_mod_analysis.argtypes = ctypes.c_bool, [nir_scalar, nir_alu_type, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (nir_remove_tex_shadow:=dll.nir_remove_tex_shadow).restype, nir_remove_tex_shadow.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.c_uint32]
except AttributeError: pass

try: (nir_trivialize_registers:=dll.nir_trivialize_registers).restype, nir_trivialize_registers.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

try: (nir_static_workgroup_size:=dll.nir_static_workgroup_size).restype, nir_static_workgroup_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(nir_shader)]
except AttributeError: pass

class struct_nir_use_dominance_state(Struct): pass
nir_use_dominance_state = struct_nir_use_dominance_state
try: (nir_calc_use_dominance_impl:=dll.nir_calc_use_dominance_impl).restype, nir_calc_use_dominance_impl.argtypes = ctypes.POINTER(nir_use_dominance_state), [ctypes.POINTER(nir_function_impl), ctypes.c_bool]
except AttributeError: pass

try: (nir_get_immediate_use_dominator:=dll.nir_get_immediate_use_dominator).restype, nir_get_immediate_use_dominator.argtypes = ctypes.POINTER(nir_instr), [ctypes.POINTER(nir_use_dominance_state), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_use_dominance_lca:=dll.nir_use_dominance_lca).restype, nir_use_dominance_lca.argtypes = ctypes.POINTER(nir_instr), [ctypes.POINTER(nir_use_dominance_state), ctypes.POINTER(nir_instr), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_instr_dominates_use:=dll.nir_instr_dominates_use).restype, nir_instr_dominates_use.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_use_dominance_state), ctypes.POINTER(nir_instr), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_print_use_dominators:=dll.nir_print_use_dominators).restype, nir_print_use_dominators.argtypes = None, [ctypes.POINTER(nir_use_dominance_state), ctypes.POINTER(ctypes.POINTER(nir_instr)), ctypes.c_uint32]
except AttributeError: pass

class nir_output_deps(Struct): pass
class nir_output_deps_output(Struct): pass
nir_output_deps_output._fields_ = [
  ('instr_list', ctypes.POINTER(ctypes.POINTER(nir_instr))),
  ('num_instr', ctypes.c_uint32),
]
nir_output_deps._fields_ = [
  ('output', (nir_output_deps_output * 112)),
]
try: (nir_gather_output_dependencies:=dll.nir_gather_output_dependencies).restype, nir_gather_output_dependencies.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_output_deps)]
except AttributeError: pass

try: (nir_free_output_dependencies:=dll.nir_free_output_dependencies).restype, nir_free_output_dependencies.argtypes = None, [ctypes.POINTER(nir_output_deps)]
except AttributeError: pass

class nir_input_to_output_deps(Struct): pass
class nir_input_to_output_deps_output(Struct): pass
nir_input_to_output_deps_output._fields_ = [
  ('inputs', (ctypes.c_uint32 * 28)),
  ('defined', ctypes.c_bool),
  ('uses_ssbo_reads', ctypes.c_bool),
  ('uses_image_reads', ctypes.c_bool),
]
nir_input_to_output_deps._fields_ = [
  ('output', (nir_input_to_output_deps_output * 112)),
]
try: (nir_gather_input_to_output_dependencies:=dll.nir_gather_input_to_output_dependencies).restype, nir_gather_input_to_output_dependencies.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_input_to_output_deps)]
except AttributeError: pass

try: (nir_print_input_to_output_deps:=dll.nir_print_input_to_output_deps).restype, nir_print_input_to_output_deps.argtypes = None, [ctypes.POINTER(nir_input_to_output_deps), ctypes.POINTER(nir_shader), ctypes.POINTER(FILE)]
except AttributeError: pass

class nir_output_clipper_var_groups(Struct): pass
nir_output_clipper_var_groups._fields_ = [
  ('pos_only', (ctypes.c_uint32 * 28)),
  ('var_only', (ctypes.c_uint32 * 28)),
  ('both', (ctypes.c_uint32 * 28)),
]
try: (nir_gather_output_clipper_var_groups:=dll.nir_gather_output_clipper_var_groups).restype, nir_gather_output_clipper_var_groups.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(nir_output_clipper_var_groups)]
except AttributeError: pass

struct_nir_builder._fields_ = [
  ('cursor', nir_cursor),
  ('exact', ctypes.c_bool),
  ('fp_fast_math', uint32_t),
  ('shader', ctypes.POINTER(nir_shader)),
  ('impl', ctypes.POINTER(nir_function_impl)),
]
try: (nir_builder_init_simple_shader:=dll.nir_builder_init_simple_shader).restype, nir_builder_init_simple_shader.argtypes = nir_builder, [gl_shader_stage, ctypes.POINTER(nir_shader_compiler_options), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

nir_instr_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr), ctypes.c_void_p)
nir_intrinsic_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_void_p)
nir_alu_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_void_p)
nir_tex_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_tex_instr), ctypes.c_void_p)
nir_phi_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_phi_instr), ctypes.c_void_p)
try: (nir_builder_instr_insert:=dll.nir_builder_instr_insert).restype, nir_builder_instr_insert.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_builder_instr_insert_at_top:=dll.nir_builder_instr_insert_at_top).restype, nir_builder_instr_insert_at_top.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_instr)]
except AttributeError: pass

try: (nir_build_alu:=dll.nir_build_alu).restype, nir_build_alu.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu1:=dll.nir_build_alu1).restype, nir_build_alu1.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu2:=dll.nir_build_alu2).restype, nir_build_alu2.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu3:=dll.nir_build_alu3).restype, nir_build_alu3.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu4:=dll.nir_build_alu4).restype, nir_build_alu4.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_build_alu_src_arr:=dll.nir_build_alu_src_arr).restype, nir_build_alu_src_arr.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(ctypes.POINTER(nir_def))]
except AttributeError: pass

try: (nir_build_tex_deref_instr:=dll.nir_build_tex_deref_instr).restype, nir_build_tex_deref_instr.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_texop, ctypes.POINTER(nir_deref_instr), ctypes.POINTER(nir_deref_instr), ctypes.c_uint32, ctypes.POINTER(nir_tex_src)]
except AttributeError: pass

try: (nir_builder_cf_insert:=dll.nir_builder_cf_insert).restype, nir_builder_cf_insert.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_builder_is_inside_cf:=dll.nir_builder_is_inside_cf).restype, nir_builder_is_inside_cf.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_cf_node)]
except AttributeError: pass

try: (nir_push_if:=dll.nir_push_if).restype, nir_push_if.argtypes = ctypes.POINTER(nir_if), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_push_else:=dll.nir_push_else).restype, nir_push_else.argtypes = ctypes.POINTER(nir_if), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_if)]
except AttributeError: pass

try: (nir_pop_if:=dll.nir_pop_if).restype, nir_pop_if.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_if)]
except AttributeError: pass

try: (nir_if_phi:=dll.nir_if_phi).restype, nir_if_phi.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_push_loop:=dll.nir_push_loop).restype, nir_push_loop.argtypes = ctypes.POINTER(nir_loop), [ctypes.POINTER(nir_builder)]
except AttributeError: pass

try: (nir_push_continue:=dll.nir_push_continue).restype, nir_push_continue.argtypes = ctypes.POINTER(nir_loop), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_loop)]
except AttributeError: pass

try: (nir_pop_loop:=dll.nir_pop_loop).restype, nir_pop_loop.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_loop)]
except AttributeError: pass

try: (nir_builder_alu_instr_finish_and_insert:=dll.nir_builder_alu_instr_finish_and_insert).restype, nir_builder_alu_instr_finish_and_insert.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_alu_instr)]
except AttributeError: pass

try: (nir_build_alu_src_arr:=dll.nir_build_alu_src_arr).restype, nir_build_alu_src_arr.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_op, ctypes.POINTER(ctypes.POINTER(nir_def))]
except AttributeError: pass

try: (nir_load_system_value:=dll.nir_load_system_value).restype, nir_load_system_value.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), nir_intrinsic_op, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (nir_type_convert:=dll.nir_type_convert).restype, nir_type_convert.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), nir_alu_type, nir_alu_type, nir_rounding_mode]
except AttributeError: pass

try: (nir_vec_scalars:=dll.nir_vec_scalars).restype, nir_vec_scalars.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_scalar), ctypes.c_uint32]
except AttributeError: pass

try: (nir_ssa_for_alu_src:=dll.nir_ssa_for_alu_src).restype, nir_ssa_for_alu_src.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_alu_instr), ctypes.c_uint32]
except AttributeError: pass

try: (nir_build_string:=dll.nir_build_string).restype, nir_build_string.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_compare_func:=dll.nir_compare_func).restype, nir_compare_func.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), enum_compare_func, ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_gen_rect_vertices:=dll.nir_gen_rect_vertices).restype, nir_gen_rect_vertices.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(nir_def), ctypes.POINTER(nir_def)]
except AttributeError: pass

try: (nir_printf_fmt:=dll.nir_printf_fmt).restype, nir_printf_fmt.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_printf_fmt_at_px:=dll.nir_printf_fmt_at_px).restype, nir_printf_fmt_at_px.argtypes = None, [ctypes.POINTER(nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (nir_call_serialized:=dll.nir_call_serialized).restype, nir_call_serialized.argtypes = ctypes.POINTER(nir_def), [ctypes.POINTER(nir_builder), ctypes.POINTER(uint32_t), size_t, ctypes.POINTER(ctypes.POINTER(nir_def))]
except AttributeError: pass

nir_lower_packing_op = CEnum(ctypes.c_uint32)
nir_lower_packing_op_pack_64_2x32 = nir_lower_packing_op.define('nir_lower_packing_op_pack_64_2x32', 0)
nir_lower_packing_op_unpack_64_2x32 = nir_lower_packing_op.define('nir_lower_packing_op_unpack_64_2x32', 1)
nir_lower_packing_op_pack_64_4x16 = nir_lower_packing_op.define('nir_lower_packing_op_pack_64_4x16', 2)
nir_lower_packing_op_unpack_64_4x16 = nir_lower_packing_op.define('nir_lower_packing_op_unpack_64_4x16', 3)
nir_lower_packing_op_pack_32_2x16 = nir_lower_packing_op.define('nir_lower_packing_op_pack_32_2x16', 4)
nir_lower_packing_op_unpack_32_2x16 = nir_lower_packing_op.define('nir_lower_packing_op_unpack_32_2x16', 5)
nir_lower_packing_op_pack_32_4x8 = nir_lower_packing_op.define('nir_lower_packing_op_pack_32_4x8', 6)
nir_lower_packing_op_unpack_32_4x8 = nir_lower_packing_op.define('nir_lower_packing_op_unpack_32_4x8', 7)
nir_lower_packing_num_ops = nir_lower_packing_op.define('nir_lower_packing_num_ops', 8)

class struct_blob(Struct): pass
struct_blob._fields_ = [
  ('data', ctypes.POINTER(uint8_t)),
  ('allocated', size_t),
  ('size', size_t),
  ('fixed_allocation', ctypes.c_bool),
  ('out_of_memory', ctypes.c_bool),
]
try: (nir_serialize:=dll.nir_serialize).restype, nir_serialize.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(nir_shader), ctypes.c_bool]
except AttributeError: pass

class struct_blob_reader(Struct): pass
struct_blob_reader._fields_ = [
  ('data', ctypes.POINTER(uint8_t)),
  ('end', ctypes.POINTER(uint8_t)),
  ('current', ctypes.POINTER(uint8_t)),
  ('overrun', ctypes.c_bool),
]
try: (nir_deserialize:=dll.nir_deserialize).restype, nir_deserialize.argtypes = ctypes.POINTER(nir_shader), [ctypes.c_void_p, ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (nir_serialize_function:=dll.nir_serialize_function).restype, nir_serialize_function.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(nir_function)]
except AttributeError: pass

try: (nir_deserialize_function:=dll.nir_deserialize_function).restype, nir_deserialize_function.argtypes = ctypes.POINTER(nir_function), [ctypes.c_void_p, ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

nir_intrinsic_index_flag = CEnum(ctypes.c_uint32)
NIR_INTRINSIC_BASE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_BASE', 0)
NIR_INTRINSIC_WRITE_MASK = nir_intrinsic_index_flag.define('NIR_INTRINSIC_WRITE_MASK', 1)
NIR_INTRINSIC_STREAM_ID = nir_intrinsic_index_flag.define('NIR_INTRINSIC_STREAM_ID', 2)
NIR_INTRINSIC_UCP_ID = nir_intrinsic_index_flag.define('NIR_INTRINSIC_UCP_ID', 3)
NIR_INTRINSIC_RANGE_BASE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RANGE_BASE', 4)
NIR_INTRINSIC_RANGE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RANGE', 5)
NIR_INTRINSIC_DESC_SET = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DESC_SET', 6)
NIR_INTRINSIC_BINDING = nir_intrinsic_index_flag.define('NIR_INTRINSIC_BINDING', 7)
NIR_INTRINSIC_COMPONENT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_COMPONENT', 8)
NIR_INTRINSIC_COLUMN = nir_intrinsic_index_flag.define('NIR_INTRINSIC_COLUMN', 9)
NIR_INTRINSIC_INTERP_MODE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_INTERP_MODE', 10)
NIR_INTRINSIC_REDUCTION_OP = nir_intrinsic_index_flag.define('NIR_INTRINSIC_REDUCTION_OP', 11)
NIR_INTRINSIC_CLUSTER_SIZE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_CLUSTER_SIZE', 12)
NIR_INTRINSIC_PARAM_IDX = nir_intrinsic_index_flag.define('NIR_INTRINSIC_PARAM_IDX', 13)
NIR_INTRINSIC_IMAGE_DIM = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IMAGE_DIM', 14)
NIR_INTRINSIC_IMAGE_ARRAY = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IMAGE_ARRAY', 15)
NIR_INTRINSIC_FORMAT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_FORMAT', 16)
NIR_INTRINSIC_ACCESS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ACCESS', 17)
NIR_INTRINSIC_CALL_IDX = nir_intrinsic_index_flag.define('NIR_INTRINSIC_CALL_IDX', 18)
NIR_INTRINSIC_STACK_SIZE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_STACK_SIZE', 19)
NIR_INTRINSIC_ALIGN_MUL = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ALIGN_MUL', 20)
NIR_INTRINSIC_ALIGN_OFFSET = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ALIGN_OFFSET', 21)
NIR_INTRINSIC_DESC_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DESC_TYPE', 22)
NIR_INTRINSIC_SRC_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_TYPE', 23)
NIR_INTRINSIC_DEST_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DEST_TYPE', 24)
NIR_INTRINSIC_SRC_BASE_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_BASE_TYPE', 25)
NIR_INTRINSIC_SRC_BASE_TYPE2 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_BASE_TYPE2', 26)
NIR_INTRINSIC_DEST_BASE_TYPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DEST_BASE_TYPE', 27)
NIR_INTRINSIC_SWIZZLE_MASK = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SWIZZLE_MASK', 28)
NIR_INTRINSIC_FETCH_INACTIVE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_FETCH_INACTIVE', 29)
NIR_INTRINSIC_OFFSET0 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_OFFSET0', 30)
NIR_INTRINSIC_OFFSET1 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_OFFSET1', 31)
NIR_INTRINSIC_ST64 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ST64', 32)
NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD', 33)
NIR_INTRINSIC_DST_ACCESS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DST_ACCESS', 34)
NIR_INTRINSIC_SRC_ACCESS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_ACCESS', 35)
NIR_INTRINSIC_DRIVER_LOCATION = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DRIVER_LOCATION', 36)
NIR_INTRINSIC_MEMORY_SEMANTICS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_MEMORY_SEMANTICS', 37)
NIR_INTRINSIC_MEMORY_MODES = nir_intrinsic_index_flag.define('NIR_INTRINSIC_MEMORY_MODES', 38)
NIR_INTRINSIC_MEMORY_SCOPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_MEMORY_SCOPE', 39)
NIR_INTRINSIC_EXECUTION_SCOPE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_EXECUTION_SCOPE', 40)
NIR_INTRINSIC_IO_SEMANTICS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IO_SEMANTICS', 41)
NIR_INTRINSIC_IO_XFB = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IO_XFB', 42)
NIR_INTRINSIC_IO_XFB2 = nir_intrinsic_index_flag.define('NIR_INTRINSIC_IO_XFB2', 43)
NIR_INTRINSIC_RAY_QUERY_VALUE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RAY_QUERY_VALUE', 44)
NIR_INTRINSIC_COMMITTED = nir_intrinsic_index_flag.define('NIR_INTRINSIC_COMMITTED', 45)
NIR_INTRINSIC_ROUNDING_MODE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ROUNDING_MODE', 46)
NIR_INTRINSIC_SATURATE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SATURATE', 47)
NIR_INTRINSIC_SYNCHRONOUS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SYNCHRONOUS', 48)
NIR_INTRINSIC_VALUE_ID = nir_intrinsic_index_flag.define('NIR_INTRINSIC_VALUE_ID', 49)
NIR_INTRINSIC_SIGN_EXTEND = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SIGN_EXTEND', 50)
NIR_INTRINSIC_FLAGS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_FLAGS', 51)
NIR_INTRINSIC_ATOMIC_OP = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ATOMIC_OP', 52)
NIR_INTRINSIC_RESOURCE_BLOCK_INTEL = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RESOURCE_BLOCK_INTEL', 53)
NIR_INTRINSIC_RESOURCE_ACCESS_INTEL = nir_intrinsic_index_flag.define('NIR_INTRINSIC_RESOURCE_ACCESS_INTEL', 54)
NIR_INTRINSIC_NUM_COMPONENTS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NUM_COMPONENTS', 55)
NIR_INTRINSIC_NUM_ARRAY_ELEMS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NUM_ARRAY_ELEMS', 56)
NIR_INTRINSIC_BIT_SIZE = nir_intrinsic_index_flag.define('NIR_INTRINSIC_BIT_SIZE', 57)
NIR_INTRINSIC_DIVERGENT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DIVERGENT', 58)
NIR_INTRINSIC_LEGACY_FABS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_LEGACY_FABS', 59)
NIR_INTRINSIC_LEGACY_FNEG = nir_intrinsic_index_flag.define('NIR_INTRINSIC_LEGACY_FNEG', 60)
NIR_INTRINSIC_LEGACY_FSAT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_LEGACY_FSAT', 61)
NIR_INTRINSIC_CMAT_DESC = nir_intrinsic_index_flag.define('NIR_INTRINSIC_CMAT_DESC', 62)
NIR_INTRINSIC_MATRIX_LAYOUT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_MATRIX_LAYOUT', 63)
NIR_INTRINSIC_CMAT_SIGNED_MASK = nir_intrinsic_index_flag.define('NIR_INTRINSIC_CMAT_SIGNED_MASK', 64)
NIR_INTRINSIC_ALU_OP = nir_intrinsic_index_flag.define('NIR_INTRINSIC_ALU_OP', 65)
NIR_INTRINSIC_NEG_LO_AMD = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NEG_LO_AMD', 66)
NIR_INTRINSIC_NEG_HI_AMD = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NEG_HI_AMD', 67)
NIR_INTRINSIC_SYSTOLIC_DEPTH = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SYSTOLIC_DEPTH', 68)
NIR_INTRINSIC_REPEAT_COUNT = nir_intrinsic_index_flag.define('NIR_INTRINSIC_REPEAT_COUNT', 69)
NIR_INTRINSIC_DST_CMAT_DESC = nir_intrinsic_index_flag.define('NIR_INTRINSIC_DST_CMAT_DESC', 70)
NIR_INTRINSIC_SRC_CMAT_DESC = nir_intrinsic_index_flag.define('NIR_INTRINSIC_SRC_CMAT_DESC', 71)
NIR_INTRINSIC_EXPLICIT_COORD = nir_intrinsic_index_flag.define('NIR_INTRINSIC_EXPLICIT_COORD', 72)
NIR_INTRINSIC_FMT_IDX = nir_intrinsic_index_flag.define('NIR_INTRINSIC_FMT_IDX', 73)
NIR_INTRINSIC_PREAMBLE_CLASS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_PREAMBLE_CLASS', 74)
NIR_INTRINSIC_NUM_INDEX_FLAGS = nir_intrinsic_index_flag.define('NIR_INTRINSIC_NUM_INDEX_FLAGS', 75)

try: nir_intrinsic_index_names = (ctypes.POINTER(ctypes.c_char) * 75).in_dll(dll, 'nir_intrinsic_index_names')
except (ValueError,AttributeError): pass
enum_nv_device_type = CEnum(ctypes.c_ubyte)
NV_DEVICE_TYPE_IGP = enum_nv_device_type.define('NV_DEVICE_TYPE_IGP', 0)
NV_DEVICE_TYPE_DIS = enum_nv_device_type.define('NV_DEVICE_TYPE_DIS', 1)
NV_DEVICE_TYPE_SOC = enum_nv_device_type.define('NV_DEVICE_TYPE_SOC', 2)

class struct_nv_device_info(Struct): pass
class struct_nv_device_info_pci(Struct): pass
struct_nv_device_info_pci._fields_ = [
  ('domain', uint16_t),
  ('bus', uint8_t),
  ('dev', uint8_t),
  ('func', uint8_t),
  ('revision_id', uint8_t),
]
struct_nv_device_info._fields_ = [
  ('type', enum_nv_device_type),
  ('device_id', uint16_t),
  ('chipset', uint16_t),
  ('device_name', (ctypes.c_char * 64)),
  ('chipset_name', (ctypes.c_char * 16)),
  ('pci', struct_nv_device_info_pci),
  ('sm', uint8_t),
  ('gpc_count', uint8_t),
  ('tpc_count', uint16_t),
  ('mp_per_tpc', uint8_t),
  ('max_warps_per_mp', uint8_t),
  ('cls_copy', uint16_t),
  ('cls_eng2d', uint16_t),
  ('cls_eng3d', uint16_t),
  ('cls_m2mf', uint16_t),
  ('cls_compute', uint16_t),
  ('vram_size_B', uint64_t),
  ('bar_size_B', uint64_t),
]
class struct_nak_compiler(Struct): pass
try: (nak_compiler_create:=dll.nak_compiler_create).restype, nak_compiler_create.argtypes = ctypes.POINTER(struct_nak_compiler), [ctypes.POINTER(struct_nv_device_info)]
except AttributeError: pass

try: (nak_compiler_destroy:=dll.nak_compiler_destroy).restype, nak_compiler_destroy.argtypes = None, [ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

try: (nak_debug_flags:=dll.nak_debug_flags).restype, nak_debug_flags.argtypes = uint64_t, [ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

try: (nak_nir_options:=dll.nak_nir_options).restype, nak_nir_options.argtypes = ctypes.POINTER(struct_nir_shader_compiler_options), [ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

try: (nak_preprocess_nir:=dll.nak_preprocess_nir).restype, nak_preprocess_nir.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

try: (nak_nir_lower_image_addrs:=dll.nak_nir_lower_image_addrs).restype, nak_nir_lower_image_addrs.argtypes = ctypes.c_bool, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_nak_compiler)]
except AttributeError: pass

class struct_nak_sample_location(Struct): pass
struct_nak_sample_location._fields_ = [
  ('x_u4', uint8_t,4),
  ('y_u4', uint8_t,4),
]
class struct_nak_sample_mask(Struct): pass
struct_nak_sample_mask._fields_ = [
  ('sample_mask', uint16_t),
]
class struct_nak_fs_key(Struct): pass
struct_nak_fs_key._fields_ = [
  ('zs_self_dep', ctypes.c_bool),
  ('force_sample_shading', ctypes.c_bool),
  ('uses_underestimate', ctypes.c_bool),
  ('sample_info_cb', uint8_t),
  ('sample_locations_offset', uint32_t),
  ('sample_masks_offset', uint32_t),
]
try: (nak_postprocess_nir:=dll.nak_postprocess_nir).restype, nak_postprocess_nir.argtypes = None, [ctypes.POINTER(nir_shader), ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except AttributeError: pass

enum_nak_ts_domain = CEnum(ctypes.c_ubyte)
NAK_TS_DOMAIN_ISOLINE = enum_nak_ts_domain.define('NAK_TS_DOMAIN_ISOLINE', 0)
NAK_TS_DOMAIN_TRIANGLE = enum_nak_ts_domain.define('NAK_TS_DOMAIN_TRIANGLE', 1)
NAK_TS_DOMAIN_QUAD = enum_nak_ts_domain.define('NAK_TS_DOMAIN_QUAD', 2)

enum_nak_ts_spacing = CEnum(ctypes.c_ubyte)
NAK_TS_SPACING_INTEGER = enum_nak_ts_spacing.define('NAK_TS_SPACING_INTEGER', 0)
NAK_TS_SPACING_FRACT_ODD = enum_nak_ts_spacing.define('NAK_TS_SPACING_FRACT_ODD', 1)
NAK_TS_SPACING_FRACT_EVEN = enum_nak_ts_spacing.define('NAK_TS_SPACING_FRACT_EVEN', 2)

enum_nak_ts_prims = CEnum(ctypes.c_ubyte)
NAK_TS_PRIMS_POINTS = enum_nak_ts_prims.define('NAK_TS_PRIMS_POINTS', 0)
NAK_TS_PRIMS_LINES = enum_nak_ts_prims.define('NAK_TS_PRIMS_LINES', 1)
NAK_TS_PRIMS_TRIANGLES_CW = enum_nak_ts_prims.define('NAK_TS_PRIMS_TRIANGLES_CW', 2)
NAK_TS_PRIMS_TRIANGLES_CCW = enum_nak_ts_prims.define('NAK_TS_PRIMS_TRIANGLES_CCW', 3)

class struct_nak_xfb_info(Struct): pass
struct_nak_xfb_info._fields_ = [
  ('stride', (uint32_t * 4)),
  ('stream', (uint8_t * 4)),
  ('attr_count', (uint8_t * 4)),
  ('attr_index', ((uint8_t * 128) * 4)),
]
class struct_nak_shader_info(Struct): pass
class struct_nak_shader_info_0(ctypes.Union): pass
class struct_nak_shader_info_0_cs(Struct): pass
struct_nak_shader_info_0_cs._fields_ = [
  ('local_size', (uint16_t * 3)),
  ('smem_size', uint16_t),
  ('_pad', (uint8_t * 4)),
]
class struct_nak_shader_info_0_fs(Struct): pass
struct_nak_shader_info_0_fs._fields_ = [
  ('writes_depth', ctypes.c_bool),
  ('reads_sample_mask', ctypes.c_bool),
  ('post_depth_coverage', ctypes.c_bool),
  ('uses_sample_shading', ctypes.c_bool),
  ('early_fragment_tests', ctypes.c_bool),
  ('_pad', (uint8_t * 7)),
]
class struct_nak_shader_info_0_ts(Struct): pass
struct_nak_shader_info_0_ts._fields_ = [
  ('domain', enum_nak_ts_domain),
  ('spacing', enum_nak_ts_spacing),
  ('prims', enum_nak_ts_prims),
  ('_pad', (uint8_t * 9)),
]
struct_nak_shader_info_0._fields_ = [
  ('cs', struct_nak_shader_info_0_cs),
  ('fs', struct_nak_shader_info_0_fs),
  ('ts', struct_nak_shader_info_0_ts),
  ('_pad', (uint8_t * 12)),
]
class struct_nak_shader_info_vtg(Struct): pass
struct_nak_shader_info_vtg._fields_ = [
  ('writes_layer', ctypes.c_bool),
  ('writes_point_size', ctypes.c_bool),
  ('writes_vprs_table_index', ctypes.c_bool),
  ('clip_enable', uint8_t),
  ('cull_enable', uint8_t),
  ('_pad', (uint8_t * 3)),
  ('xfb', struct_nak_xfb_info),
]
struct_nak_shader_info._anonymous_ = ['_0']
struct_nak_shader_info._fields_ = [
  ('stage', gl_shader_stage),
  ('sm', uint8_t),
  ('num_gprs', uint8_t),
  ('num_control_barriers', uint8_t),
  ('_pad0', uint8_t),
  ('max_warps_per_sm', uint32_t),
  ('num_instrs', uint32_t),
  ('num_static_cycles', uint32_t),
  ('num_spills_to_mem', uint32_t),
  ('num_fills_from_mem', uint32_t),
  ('num_spills_to_reg', uint32_t),
  ('num_fills_from_reg', uint32_t),
  ('slm_size', uint32_t),
  ('crs_size', uint32_t),
  ('_0', struct_nak_shader_info_0),
  ('vtg', struct_nak_shader_info_vtg),
  ('hdr', (uint32_t * 32)),
]
class struct_nak_shader_bin(Struct): pass
struct_nak_shader_bin._fields_ = [
  ('info', struct_nak_shader_info),
  ('code_size', uint32_t),
  ('code', ctypes.c_void_p),
  ('asm_str', ctypes.POINTER(ctypes.c_char)),
]
try: (nak_shader_bin_destroy:=dll.nak_shader_bin_destroy).restype, nak_shader_bin_destroy.argtypes = None, [ctypes.POINTER(struct_nak_shader_bin)]
except AttributeError: pass

try: (nak_compile_shader:=dll.nak_compile_shader).restype, nak_compile_shader.argtypes = ctypes.POINTER(struct_nak_shader_bin), [ctypes.POINTER(nir_shader), ctypes.c_bool, ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except AttributeError: pass

class struct_nak_qmd_cbuf(Struct): pass
struct_nak_qmd_cbuf._fields_ = [
  ('index', uint32_t),
  ('size', uint32_t),
  ('addr', uint64_t),
]
class struct_nak_qmd_info(Struct): pass
struct_nak_qmd_info._fields_ = [
  ('addr', uint64_t),
  ('smem_size', uint16_t),
  ('smem_max', uint16_t),
  ('global_size', (uint32_t * 3)),
  ('num_cbufs', uint32_t),
  ('cbufs', (struct_nak_qmd_cbuf * 8)),
]
try: (nak_qmd_size_B:=dll.nak_qmd_size_B).restype, nak_qmd_size_B.argtypes = uint32_t, [ctypes.POINTER(struct_nv_device_info)]
except AttributeError: pass

try: (nak_fill_qmd:=dll.nak_fill_qmd).restype, nak_fill_qmd.argtypes = None, [ctypes.POINTER(struct_nv_device_info), ctypes.POINTER(struct_nak_shader_info), ctypes.POINTER(struct_nak_qmd_info), ctypes.c_void_p, size_t]
except AttributeError: pass

class struct_nak_qmd_dispatch_size_layout(Struct): pass
struct_nak_qmd_dispatch_size_layout._fields_ = [
  ('x_start', uint16_t),
  ('x_end', uint16_t),
  ('y_start', uint16_t),
  ('y_end', uint16_t),
  ('z_start', uint16_t),
  ('z_end', uint16_t),
]
try: (nak_get_qmd_dispatch_size_layout:=dll.nak_get_qmd_dispatch_size_layout).restype, nak_get_qmd_dispatch_size_layout.argtypes = struct_nak_qmd_dispatch_size_layout, [ctypes.POINTER(struct_nv_device_info)]
except AttributeError: pass

class struct_nak_qmd_cbuf_desc_layout(Struct): pass
struct_nak_qmd_cbuf_desc_layout._fields_ = [
  ('addr_shift', uint16_t),
  ('addr_lo_start', uint16_t),
  ('addr_lo_end', uint16_t),
  ('addr_hi_start', uint16_t),
  ('addr_hi_end', uint16_t),
]
try: (nak_get_qmd_cbuf_desc_layout:=dll.nak_get_qmd_cbuf_desc_layout).restype, nak_get_qmd_cbuf_desc_layout.argtypes = struct_nak_qmd_cbuf_desc_layout, [ctypes.POINTER(struct_nv_device_info), uint8_t]
except AttributeError: pass

class struct_lp_context_ref(Struct): pass
class struct_LLVMOpaqueContext(Struct): pass
LLVMContextRef = ctypes.POINTER(struct_LLVMOpaqueContext)
struct_lp_context_ref._fields_ = [
  ('ref', LLVMContextRef),
  ('owned', ctypes.c_bool),
]
lp_context_ref = struct_lp_context_ref
class struct_lp_passmgr(Struct): pass
class struct_LLVMOpaqueModule(Struct): pass
LLVMModuleRef = ctypes.POINTER(struct_LLVMOpaqueModule)
try: (lp_passmgr_create:=dll.lp_passmgr_create).restype, lp_passmgr_create.argtypes = ctypes.c_bool, [LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(struct_lp_passmgr))]
except AttributeError: pass

class struct_LLVMOpaqueTargetMachine(Struct): pass
LLVMTargetMachineRef = ctypes.POINTER(struct_LLVMOpaqueTargetMachine)
try: (lp_passmgr_run:=dll.lp_passmgr_run).restype, lp_passmgr_run.argtypes = None, [ctypes.POINTER(struct_lp_passmgr), LLVMModuleRef, LLVMTargetMachineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_passmgr_dispose:=dll.lp_passmgr_dispose).restype, lp_passmgr_dispose.argtypes = None, [ctypes.POINTER(struct_lp_passmgr)]
except AttributeError: pass

class struct_lp_cached_code(Struct): pass
struct_lp_cached_code._fields_ = [
  ('data', ctypes.c_void_p),
  ('data_size', size_t),
  ('dont_cache', ctypes.c_bool),
  ('jit_obj_cache', ctypes.c_void_p),
]
class struct_lp_generated_code(Struct): pass
class struct_LLVMOpaqueTargetLibraryInfotData(Struct): pass
LLVMTargetLibraryInfoRef = ctypes.POINTER(struct_LLVMOpaqueTargetLibraryInfotData)
try: (gallivm_create_target_library_info:=dll.gallivm_create_target_library_info).restype, gallivm_create_target_library_info.argtypes = LLVMTargetLibraryInfoRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (gallivm_dispose_target_library_info:=dll.gallivm_dispose_target_library_info).restype, gallivm_dispose_target_library_info.argtypes = None, [LLVMTargetLibraryInfoRef]
except AttributeError: pass

try: (lp_set_target_options:=dll.lp_set_target_options).restype, lp_set_target_options.argtypes = None, []
except AttributeError: pass

try: (lp_bld_init_native_targets:=dll.lp_bld_init_native_targets).restype, lp_bld_init_native_targets.argtypes = None, []
except AttributeError: pass

class struct_LLVMOpaqueExecutionEngine(Struct): pass
LLVMExecutionEngineRef = ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)
class struct_LLVMOpaqueMCJITMemoryManager(Struct): pass
LLVMMCJITMemoryManagerRef = ctypes.POINTER(struct_LLVMOpaqueMCJITMemoryManager)
try: (lp_build_create_jit_compiler_for_module:=dll.lp_build_create_jit_compiler_for_module).restype, lp_build_create_jit_compiler_for_module.argtypes = ctypes.c_int32, [ctypes.POINTER(LLVMExecutionEngineRef), ctypes.POINTER(ctypes.POINTER(struct_lp_generated_code)), ctypes.POINTER(struct_lp_cached_code), LLVMModuleRef, LLVMMCJITMemoryManagerRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (lp_free_generated_code:=dll.lp_free_generated_code).restype, lp_free_generated_code.argtypes = None, [ctypes.POINTER(struct_lp_generated_code)]
except AttributeError: pass

try: (lp_get_default_memory_manager:=dll.lp_get_default_memory_manager).restype, lp_get_default_memory_manager.argtypes = LLVMMCJITMemoryManagerRef, []
except AttributeError: pass

try: (lp_free_memory_manager:=dll.lp_free_memory_manager).restype, lp_free_memory_manager.argtypes = None, [LLVMMCJITMemoryManagerRef]
except AttributeError: pass

class struct_LLVMOpaqueValue(Struct): pass
LLVMValueRef = ctypes.POINTER(struct_LLVMOpaqueValue)
try: (lp_get_called_value:=dll.lp_get_called_value).restype, lp_get_called_value.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (lp_is_function:=dll.lp_is_function).restype, lp_is_function.argtypes = ctypes.c_bool, [LLVMValueRef]
except AttributeError: pass

try: (lp_free_objcache:=dll.lp_free_objcache).restype, lp_free_objcache.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

try: (lp_set_module_stack_alignment_override:=dll.lp_set_module_stack_alignment_override).restype, lp_set_module_stack_alignment_override.argtypes = None, [LLVMModuleRef, ctypes.c_uint32]
except AttributeError: pass

try: lp_native_vector_width = ctypes.c_uint32.in_dll(dll, 'lp_native_vector_width')
except (ValueError,AttributeError): pass
class struct_lp_type(Struct): pass
struct_lp_type._fields_ = [
  ('floating', ctypes.c_uint32,1),
  ('fixed', ctypes.c_uint32,1),
  ('sign', ctypes.c_uint32,1),
  ('norm', ctypes.c_uint32,1),
  ('signed_zero_preserve', ctypes.c_uint32,1),
  ('nan_preserve', ctypes.c_uint32,1),
  ('width', ctypes.c_uint32,14),
  ('length', ctypes.c_uint32,14),
]
class struct_lp_build_context(Struct): pass
class struct_gallivm_state(Struct): pass
class struct_LLVMOpaqueType(Struct): pass
LLVMTypeRef = ctypes.POINTER(struct_LLVMOpaqueType)
struct_lp_build_context._fields_ = [
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
  ('type', struct_lp_type),
  ('elem_type', LLVMTypeRef),
  ('vec_type', LLVMTypeRef),
  ('int_elem_type', LLVMTypeRef),
  ('int_vec_type', LLVMTypeRef),
  ('undef', LLVMValueRef),
  ('zero', LLVMValueRef),
  ('one', LLVMValueRef),
]
try: (lp_build_elem_type:=dll.lp_build_elem_type).restype, lp_build_elem_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_vec_type:=dll.lp_build_vec_type).restype, lp_build_vec_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_check_elem_type:=dll.lp_check_elem_type).restype, lp_check_elem_type.argtypes = ctypes.c_bool, [struct_lp_type, LLVMTypeRef]
except AttributeError: pass

try: (lp_check_vec_type:=dll.lp_check_vec_type).restype, lp_check_vec_type.argtypes = ctypes.c_bool, [struct_lp_type, LLVMTypeRef]
except AttributeError: pass

try: (lp_check_value:=dll.lp_check_value).restype, lp_check_value.argtypes = ctypes.c_bool, [struct_lp_type, LLVMValueRef]
except AttributeError: pass

try: (lp_build_int_elem_type:=dll.lp_build_int_elem_type).restype, lp_build_int_elem_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_int_vec_type:=dll.lp_build_int_vec_type).restype, lp_build_int_vec_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_elem_type:=dll.lp_elem_type).restype, lp_elem_type.argtypes = struct_lp_type, [struct_lp_type]
except AttributeError: pass

try: (lp_uint_type:=dll.lp_uint_type).restype, lp_uint_type.argtypes = struct_lp_type, [struct_lp_type]
except AttributeError: pass

try: (lp_int_type:=dll.lp_int_type).restype, lp_int_type.argtypes = struct_lp_type, [struct_lp_type]
except AttributeError: pass

try: (lp_wider_type:=dll.lp_wider_type).restype, lp_wider_type.argtypes = struct_lp_type, [struct_lp_type]
except AttributeError: pass

try: (lp_sizeof_llvm_type:=dll.lp_sizeof_llvm_type).restype, lp_sizeof_llvm_type.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

LLVMTypeKind = CEnum(ctypes.c_uint32)
LLVMVoidTypeKind = LLVMTypeKind.define('LLVMVoidTypeKind', 0)
LLVMHalfTypeKind = LLVMTypeKind.define('LLVMHalfTypeKind', 1)
LLVMFloatTypeKind = LLVMTypeKind.define('LLVMFloatTypeKind', 2)
LLVMDoubleTypeKind = LLVMTypeKind.define('LLVMDoubleTypeKind', 3)
LLVMX86_FP80TypeKind = LLVMTypeKind.define('LLVMX86_FP80TypeKind', 4)
LLVMFP128TypeKind = LLVMTypeKind.define('LLVMFP128TypeKind', 5)
LLVMPPC_FP128TypeKind = LLVMTypeKind.define('LLVMPPC_FP128TypeKind', 6)
LLVMLabelTypeKind = LLVMTypeKind.define('LLVMLabelTypeKind', 7)
LLVMIntegerTypeKind = LLVMTypeKind.define('LLVMIntegerTypeKind', 8)
LLVMFunctionTypeKind = LLVMTypeKind.define('LLVMFunctionTypeKind', 9)
LLVMStructTypeKind = LLVMTypeKind.define('LLVMStructTypeKind', 10)
LLVMArrayTypeKind = LLVMTypeKind.define('LLVMArrayTypeKind', 11)
LLVMPointerTypeKind = LLVMTypeKind.define('LLVMPointerTypeKind', 12)
LLVMVectorTypeKind = LLVMTypeKind.define('LLVMVectorTypeKind', 13)
LLVMMetadataTypeKind = LLVMTypeKind.define('LLVMMetadataTypeKind', 14)
LLVMTokenTypeKind = LLVMTypeKind.define('LLVMTokenTypeKind', 16)
LLVMScalableVectorTypeKind = LLVMTypeKind.define('LLVMScalableVectorTypeKind', 17)
LLVMBFloatTypeKind = LLVMTypeKind.define('LLVMBFloatTypeKind', 18)
LLVMX86_AMXTypeKind = LLVMTypeKind.define('LLVMX86_AMXTypeKind', 19)
LLVMTargetExtTypeKind = LLVMTypeKind.define('LLVMTargetExtTypeKind', 20)

try: (lp_typekind_name:=dll.lp_typekind_name).restype, lp_typekind_name.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTypeKind]
except AttributeError: pass

try: (lp_dump_llvmtype:=dll.lp_dump_llvmtype).restype, lp_dump_llvmtype.argtypes = None, [LLVMTypeRef]
except AttributeError: pass

try: (lp_build_context_init:=dll.lp_build_context_init).restype, lp_build_context_init.argtypes = None, [ctypes.POINTER(struct_lp_build_context), ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_count_ir_module:=dll.lp_build_count_ir_module).restype, lp_build_count_ir_module.argtypes = ctypes.c_uint32, [LLVMModuleRef]
except AttributeError: pass

class struct_lp_jit_texture(Struct): pass
class struct_LLVMOpaqueTargetData(Struct): pass
LLVMTargetDataRef = ctypes.POINTER(struct_LLVMOpaqueTargetData)
class struct_LLVMOpaqueBuilder(Struct): pass
LLVMBuilderRef = ctypes.POINTER(struct_LLVMOpaqueBuilder)
class struct_LLVMOpaqueDIBuilder(Struct): pass
LLVMDIBuilderRef = ctypes.POINTER(struct_LLVMOpaqueDIBuilder)
class struct_LLVMOpaqueMetadata(Struct): pass
LLVMMetadataRef = ctypes.POINTER(struct_LLVMOpaqueMetadata)
struct_gallivm_state._fields_ = [
  ('module_name', ctypes.POINTER(ctypes.c_char)),
  ('file_name', ctypes.POINTER(ctypes.c_char)),
  ('module', LLVMModuleRef),
  ('target', LLVMTargetDataRef),
  ('engine', LLVMExecutionEngineRef),
  ('passmgr', ctypes.POINTER(struct_lp_passmgr)),
  ('memorymgr', LLVMMCJITMemoryManagerRef),
  ('code', ctypes.POINTER(struct_lp_generated_code)),
  ('context', LLVMContextRef),
  ('builder', LLVMBuilderRef),
  ('di_builder', LLVMDIBuilderRef),
  ('cache', ctypes.POINTER(struct_lp_cached_code)),
  ('compiled', ctypes.c_uint32),
  ('coro_malloc_hook', LLVMValueRef),
  ('coro_free_hook', LLVMValueRef),
  ('debug_printf_hook', LLVMValueRef),
  ('coro_malloc_hook_type', LLVMTypeRef),
  ('coro_free_hook_type', LLVMTypeRef),
  ('di_function', LLVMMetadataRef),
  ('file', LLVMMetadataRef),
  ('get_time_hook', LLVMValueRef),
  ('texture_descriptor', LLVMValueRef),
  ('texture_dynamic_state', ctypes.POINTER(struct_lp_jit_texture)),
  ('sampler_descriptor', LLVMValueRef),
]
try: (lp_build_init_native_width:=dll.lp_build_init_native_width).restype, lp_build_init_native_width.argtypes = ctypes.c_uint32, []
except AttributeError: pass

try: (lp_build_init:=dll.lp_build_init).restype, lp_build_init.argtypes = ctypes.c_bool, []
except AttributeError: pass

try: (gallivm_create:=dll.gallivm_create).restype, gallivm_create.argtypes = ctypes.POINTER(struct_gallivm_state), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(lp_context_ref), ctypes.POINTER(struct_lp_cached_code)]
except AttributeError: pass

try: (gallivm_destroy:=dll.gallivm_destroy).restype, gallivm_destroy.argtypes = None, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (gallivm_free_ir:=dll.gallivm_free_ir).restype, gallivm_free_ir.argtypes = None, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (gallivm_verify_function:=dll.gallivm_verify_function).restype, gallivm_verify_function.argtypes = None, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError: pass

try: (gallivm_add_global_mapping:=dll.gallivm_add_global_mapping).restype, gallivm_add_global_mapping.argtypes = None, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, ctypes.c_void_p]
except AttributeError: pass

try: (gallivm_compile_module:=dll.gallivm_compile_module).restype, gallivm_compile_module.argtypes = None, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

func_pointer = ctypes.CFUNCTYPE(None, )
try: (gallivm_jit_function:=dll.gallivm_jit_function).restype, gallivm_jit_function.argtypes = func_pointer, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (gallivm_stub_func:=dll.gallivm_stub_func).restype, gallivm_stub_func.argtypes = None, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError: pass

try: (gallivm_get_perf_flags:=dll.gallivm_get_perf_flags).restype, gallivm_get_perf_flags.argtypes = ctypes.c_uint32, []
except AttributeError: pass

try: (lp_init_clock_hook:=dll.lp_init_clock_hook).restype, lp_init_clock_hook.argtypes = None, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (lp_init_env_options:=dll.lp_init_env_options).restype, lp_init_env_options.argtypes = None, []
except AttributeError: pass

class struct_lp_build_tgsi_params(Struct): pass
class struct_lp_build_mask_context(Struct): pass
class struct_lp_build_skip_context(Struct): pass
class struct_LLVMOpaqueBasicBlock(Struct): pass
LLVMBasicBlockRef = ctypes.POINTER(struct_LLVMOpaqueBasicBlock)
struct_lp_build_skip_context._fields_ = [
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
  ('block', LLVMBasicBlockRef),
]
struct_lp_build_mask_context._fields_ = [
  ('skip', struct_lp_build_skip_context),
  ('reg_type', LLVMTypeRef),
  ('var_type', LLVMTypeRef),
  ('var', LLVMValueRef),
]
class struct_lp_bld_tgsi_system_values(Struct): pass
struct_lp_bld_tgsi_system_values._fields_ = [
  ('instance_id', LLVMValueRef),
  ('base_instance', LLVMValueRef),
  ('vertex_id', LLVMValueRef),
  ('vertex_id_nobase', LLVMValueRef),
  ('prim_id', LLVMValueRef),
  ('basevertex', LLVMValueRef),
  ('firstvertex', LLVMValueRef),
  ('invocation_id', LLVMValueRef),
  ('draw_id', LLVMValueRef),
  ('thread_id', (LLVMValueRef * 3)),
  ('block_id', (LLVMValueRef * 3)),
  ('grid_size', (LLVMValueRef * 3)),
  ('front_facing', LLVMValueRef),
  ('work_dim', LLVMValueRef),
  ('block_size', (LLVMValueRef * 3)),
  ('tess_coord', LLVMValueRef),
  ('tess_outer', LLVMValueRef),
  ('tess_inner', LLVMValueRef),
  ('vertices_in', LLVMValueRef),
  ('sample_id', LLVMValueRef),
  ('sample_pos_type', LLVMTypeRef),
  ('sample_pos', LLVMValueRef),
  ('sample_mask_in', LLVMValueRef),
  ('view_index', LLVMValueRef),
  ('subgroup_id', LLVMValueRef),
  ('num_subgroups', LLVMValueRef),
]
class struct_lp_build_sampler_soa(Struct): pass
class struct_lp_sampler_params(Struct): pass
class struct_lp_derivatives(Struct): pass
struct_lp_derivatives._fields_ = [
  ('ddx', (LLVMValueRef * 3)),
  ('ddy', (LLVMValueRef * 3)),
]
struct_lp_sampler_params._fields_ = [
  ('type', struct_lp_type),
  ('texture_index', ctypes.c_uint32),
  ('sampler_index', ctypes.c_uint32),
  ('texture_index_offset', LLVMValueRef),
  ('sample_key', ctypes.c_uint32),
  ('resources_type', LLVMTypeRef),
  ('resources_ptr', LLVMValueRef),
  ('thread_data_type', LLVMTypeRef),
  ('thread_data_ptr', LLVMValueRef),
  ('coords', ctypes.POINTER(LLVMValueRef)),
  ('offsets', ctypes.POINTER(LLVMValueRef)),
  ('ms_index', LLVMValueRef),
  ('lod', LLVMValueRef),
  ('min_lod', LLVMValueRef),
  ('derivs', ctypes.POINTER(struct_lp_derivatives)),
  ('texel', ctypes.POINTER(LLVMValueRef)),
  ('texture_resource', LLVMValueRef),
  ('sampler_resource', LLVMValueRef),
  ('exec_mask', LLVMValueRef),
  ('exec_mask_nz', ctypes.c_bool),
]
class struct_lp_sampler_size_query_params(Struct): pass
enum_lp_sampler_lod_property = CEnum(ctypes.c_uint32)
LP_SAMPLER_LOD_SCALAR = enum_lp_sampler_lod_property.define('LP_SAMPLER_LOD_SCALAR', 0)
LP_SAMPLER_LOD_PER_ELEMENT = enum_lp_sampler_lod_property.define('LP_SAMPLER_LOD_PER_ELEMENT', 1)
LP_SAMPLER_LOD_PER_QUAD = enum_lp_sampler_lod_property.define('LP_SAMPLER_LOD_PER_QUAD', 2)

struct_lp_sampler_size_query_params._fields_ = [
  ('int_type', struct_lp_type),
  ('texture_unit', ctypes.c_uint32),
  ('texture_unit_offset', LLVMValueRef),
  ('target', ctypes.c_uint32),
  ('resources_type', LLVMTypeRef),
  ('resources_ptr', LLVMValueRef),
  ('is_sviewinfo', ctypes.c_bool),
  ('samples_only', ctypes.c_bool),
  ('ms', ctypes.c_bool),
  ('lod_property', enum_lp_sampler_lod_property),
  ('explicit_lod', LLVMValueRef),
  ('sizes_out', ctypes.POINTER(LLVMValueRef)),
  ('resource', LLVMValueRef),
  ('exec_mask', LLVMValueRef),
  ('exec_mask_nz', ctypes.c_bool),
  ('format', enum_pipe_format),
]
struct_lp_build_sampler_soa._fields_ = [
  ('emit_tex_sample', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_sampler_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_params))),
  ('emit_size_query', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_sampler_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params))),
]
class struct_tgsi_shader_info(Struct): pass
struct_tgsi_shader_info._fields_ = [
  ('num_inputs', uint8_t),
  ('num_outputs', uint8_t),
  ('input_semantic_name', (uint8_t * 80)),
  ('input_semantic_index', (uint8_t * 80)),
  ('input_interpolate', (uint8_t * 80)),
  ('input_interpolate_loc', (uint8_t * 80)),
  ('input_usage_mask', (uint8_t * 80)),
  ('output_semantic_name', (uint8_t * 80)),
  ('output_semantic_index', (uint8_t * 80)),
  ('output_usagemask', (uint8_t * 80)),
  ('output_streams', (uint8_t * 80)),
  ('num_system_values', uint8_t),
  ('system_value_semantic_name', (uint8_t * 80)),
  ('processor', uint8_t),
  ('file_mask', (uint32_t * 15)),
  ('file_count', (ctypes.c_uint32 * 15)),
  ('file_max', (ctypes.c_int32 * 15)),
  ('const_file_max', (ctypes.c_int32 * 32)),
  ('const_buffers_declared', ctypes.c_uint32),
  ('samplers_declared', ctypes.c_uint32),
  ('sampler_targets', (uint8_t * 128)),
  ('sampler_type', (uint8_t * 128)),
  ('num_stream_output_components', (uint8_t * 4)),
  ('input_array_first', (uint8_t * 80)),
  ('output_array_first', (uint8_t * 80)),
  ('immediate_count', ctypes.c_uint32),
  ('num_instructions', ctypes.c_uint32),
  ('opcode_count', (ctypes.c_uint32 * 252)),
  ('reads_pervertex_outputs', ctypes.c_bool),
  ('reads_perpatch_outputs', ctypes.c_bool),
  ('reads_tessfactor_outputs', ctypes.c_bool),
  ('reads_z', ctypes.c_bool),
  ('writes_z', ctypes.c_bool),
  ('writes_stencil', ctypes.c_bool),
  ('writes_samplemask', ctypes.c_bool),
  ('writes_edgeflag', ctypes.c_bool),
  ('uses_kill', ctypes.c_bool),
  ('uses_instanceid', ctypes.c_bool),
  ('uses_vertexid', ctypes.c_bool),
  ('uses_vertexid_nobase', ctypes.c_bool),
  ('uses_basevertex', ctypes.c_bool),
  ('uses_primid', ctypes.c_bool),
  ('uses_frontface', ctypes.c_bool),
  ('uses_invocationid', ctypes.c_bool),
  ('uses_grid_size', ctypes.c_bool),
  ('writes_position', ctypes.c_bool),
  ('writes_psize', ctypes.c_bool),
  ('writes_clipvertex', ctypes.c_bool),
  ('writes_viewport_index', ctypes.c_bool),
  ('writes_layer', ctypes.c_bool),
  ('writes_memory', ctypes.c_bool),
  ('uses_fbfetch', ctypes.c_bool),
  ('num_written_culldistance', ctypes.c_uint32),
  ('num_written_clipdistance', ctypes.c_uint32),
  ('images_declared', ctypes.c_uint32),
  ('msaa_images_declared', ctypes.c_uint32),
  ('images_buffers', ctypes.c_uint32),
  ('shader_buffers_declared', ctypes.c_uint32),
  ('shader_buffers_load', ctypes.c_uint32),
  ('shader_buffers_store', ctypes.c_uint32),
  ('shader_buffers_atomic', ctypes.c_uint32),
  ('hw_atomic_declared', ctypes.c_uint32),
  ('indirect_files', ctypes.c_uint32),
  ('dim_indirect_files', ctypes.c_uint32),
  ('properties', (ctypes.c_uint32 * 29)),
]
class struct_lp_build_gs_iface(Struct): pass
struct_lp_build_gs_iface._fields_ = [
  ('fetch_input', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, LLVMValueRef)),
  ('emit_vertex', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.POINTER((LLVMValueRef * 4)), LLVMValueRef, LLVMValueRef, LLVMValueRef)),
  ('end_primitive', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), ctypes.POINTER(struct_lp_build_context), LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32)),
  ('gs_epilogue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_gs_iface), LLVMValueRef, LLVMValueRef, ctypes.c_uint32)),
]
class struct_lp_build_tcs_iface(Struct): pass
struct_lp_build_tcs_iface._fields_ = [
  ('emit_prologue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
  ('emit_epilogue', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
  ('emit_barrier', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_context))),
  ('emit_store_output', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, LLVMValueRef, LLVMValueRef)),
  ('emit_fetch_input', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef)),
  ('emit_fetch_output', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_tcs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, uint32_t)),
]
class struct_lp_build_tes_iface(Struct): pass
struct_lp_build_tes_iface._fields_ = [
  ('fetch_vertex_input', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_tes_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef)),
  ('fetch_patch_input', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_tes_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_bool, LLVMValueRef, LLVMValueRef)),
]
class struct_lp_build_mesh_iface(Struct): pass
struct_lp_build_mesh_iface._fields_ = [
  ('emit_store_output', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_mesh_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, ctypes.c_bool, LLVMValueRef, LLVMValueRef, LLVMValueRef)),
  ('emit_vertex_and_primitive_count', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_mesh_iface), ctypes.POINTER(struct_lp_build_context), LLVMValueRef, LLVMValueRef)),
]
class struct_lp_build_image_soa(Struct): pass
class struct_lp_img_params(Struct): pass
LLVMAtomicRMWBinOp = CEnum(ctypes.c_uint32)
LLVMAtomicRMWBinOpXchg = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpXchg', 0)
LLVMAtomicRMWBinOpAdd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpAdd', 1)
LLVMAtomicRMWBinOpSub = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpSub', 2)
LLVMAtomicRMWBinOpAnd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpAnd', 3)
LLVMAtomicRMWBinOpNand = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpNand', 4)
LLVMAtomicRMWBinOpOr = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpOr', 5)
LLVMAtomicRMWBinOpXor = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpXor', 6)
LLVMAtomicRMWBinOpMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpMax', 7)
LLVMAtomicRMWBinOpMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpMin', 8)
LLVMAtomicRMWBinOpUMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUMax', 9)
LLVMAtomicRMWBinOpUMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUMin', 10)
LLVMAtomicRMWBinOpFAdd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFAdd', 11)
LLVMAtomicRMWBinOpFSub = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFSub', 12)
LLVMAtomicRMWBinOpFMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFMax', 13)
LLVMAtomicRMWBinOpFMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFMin', 14)
LLVMAtomicRMWBinOpUIncWrap = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUIncWrap', 15)
LLVMAtomicRMWBinOpUDecWrap = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUDecWrap', 16)
LLVMAtomicRMWBinOpUSubCond = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUSubCond', 17)
LLVMAtomicRMWBinOpUSubSat = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUSubSat', 18)

struct_lp_img_params._fields_ = [
  ('type', struct_lp_type),
  ('image_index', ctypes.c_uint32),
  ('image_index_offset', LLVMValueRef),
  ('img_op', ctypes.c_uint32),
  ('target', ctypes.c_uint32),
  ('packed_op', ctypes.c_uint32),
  ('op', LLVMAtomicRMWBinOp),
  ('exec_mask', LLVMValueRef),
  ('exec_mask_nz', ctypes.c_bool),
  ('resources_type', LLVMTypeRef),
  ('resources_ptr', LLVMValueRef),
  ('thread_data_type', LLVMTypeRef),
  ('thread_data_ptr', LLVMValueRef),
  ('coords', ctypes.POINTER(LLVMValueRef)),
  ('ms_index', LLVMValueRef),
  ('indata', (LLVMValueRef * 4)),
  ('indata2', (LLVMValueRef * 4)),
  ('outdata', ctypes.POINTER(LLVMValueRef)),
  ('resource', LLVMValueRef),
  ('format', enum_pipe_format),
]
struct_lp_build_image_soa._fields_ = [
  ('emit_op', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_image_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_img_params))),
  ('emit_size_query', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_image_soa), ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params))),
]
class struct_lp_build_coro_suspend_info(Struct): pass
class struct_lp_build_fs_iface(Struct): pass
struct_lp_build_fs_iface._fields_ = [
  ('interp_fn', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_fs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, LLVMValueRef, (LLVMValueRef * 2))),
  ('fb_fetch', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_lp_build_fs_iface), ctypes.POINTER(struct_lp_build_context), ctypes.c_int32, (LLVMValueRef * 4))),
]
struct_lp_build_tgsi_params._fields_ = [
  ('type', struct_lp_type),
  ('mask', ctypes.POINTER(struct_lp_build_mask_context)),
  ('consts_ptr', LLVMValueRef),
  ('const_sizes_ptr', LLVMValueRef),
  ('system_values', ctypes.POINTER(struct_lp_bld_tgsi_system_values)),
  ('inputs', ctypes.POINTER((LLVMValueRef * 4))),
  ('num_inputs', ctypes.c_int32),
  ('context_type', LLVMTypeRef),
  ('context_ptr', LLVMValueRef),
  ('resources_type', LLVMTypeRef),
  ('resources_ptr', LLVMValueRef),
  ('thread_data_type', LLVMTypeRef),
  ('thread_data_ptr', LLVMValueRef),
  ('sampler', ctypes.POINTER(struct_lp_build_sampler_soa)),
  ('info', ctypes.POINTER(struct_tgsi_shader_info)),
  ('gs_iface', ctypes.POINTER(struct_lp_build_gs_iface)),
  ('tcs_iface', ctypes.POINTER(struct_lp_build_tcs_iface)),
  ('tes_iface', ctypes.POINTER(struct_lp_build_tes_iface)),
  ('mesh_iface', ctypes.POINTER(struct_lp_build_mesh_iface)),
  ('ssbo_ptr', LLVMValueRef),
  ('ssbo_sizes_ptr', LLVMValueRef),
  ('image', ctypes.POINTER(struct_lp_build_image_soa)),
  ('shared_ptr', LLVMValueRef),
  ('payload_ptr', LLVMValueRef),
  ('coro', ctypes.POINTER(struct_lp_build_coro_suspend_info)),
  ('fs_iface', ctypes.POINTER(struct_lp_build_fs_iface)),
  ('gs_vertex_streams', ctypes.c_uint32),
  ('current_func', LLVMValueRef),
  ('fns', ctypes.POINTER(struct_hash_table)),
  ('scratch_ptr', LLVMValueRef),
  ('call_context_ptr', LLVMValueRef),
]
try: (lp_build_nir_soa:=dll.lp_build_nir_soa).restype, lp_build_nir_soa.argtypes = None, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_lp_build_tgsi_params), ctypes.POINTER((LLVMValueRef * 4))]
except AttributeError: pass

try: (lp_build_nir_soa_func:=dll.lp_build_nir_soa_func).restype, lp_build_nir_soa_func.argtypes = None, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(nir_function_impl), ctypes.POINTER(struct_lp_build_tgsi_params), ctypes.POINTER((LLVMValueRef * 4))]
except AttributeError: pass

class struct_lp_build_sampler_aos(Struct): pass
enum_tgsi_texture_type = CEnum(ctypes.c_uint32)
TGSI_TEXTURE_BUFFER = enum_tgsi_texture_type.define('TGSI_TEXTURE_BUFFER', 0)
TGSI_TEXTURE_1D = enum_tgsi_texture_type.define('TGSI_TEXTURE_1D', 1)
TGSI_TEXTURE_2D = enum_tgsi_texture_type.define('TGSI_TEXTURE_2D', 2)
TGSI_TEXTURE_3D = enum_tgsi_texture_type.define('TGSI_TEXTURE_3D', 3)
TGSI_TEXTURE_CUBE = enum_tgsi_texture_type.define('TGSI_TEXTURE_CUBE', 4)
TGSI_TEXTURE_RECT = enum_tgsi_texture_type.define('TGSI_TEXTURE_RECT', 5)
TGSI_TEXTURE_SHADOW1D = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOW1D', 6)
TGSI_TEXTURE_SHADOW2D = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOW2D', 7)
TGSI_TEXTURE_SHADOWRECT = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOWRECT', 8)
TGSI_TEXTURE_1D_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_1D_ARRAY', 9)
TGSI_TEXTURE_2D_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_2D_ARRAY', 10)
TGSI_TEXTURE_SHADOW1D_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOW1D_ARRAY', 11)
TGSI_TEXTURE_SHADOW2D_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOW2D_ARRAY', 12)
TGSI_TEXTURE_SHADOWCUBE = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOWCUBE', 13)
TGSI_TEXTURE_2D_MSAA = enum_tgsi_texture_type.define('TGSI_TEXTURE_2D_MSAA', 14)
TGSI_TEXTURE_2D_ARRAY_MSAA = enum_tgsi_texture_type.define('TGSI_TEXTURE_2D_ARRAY_MSAA', 15)
TGSI_TEXTURE_CUBE_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_CUBE_ARRAY', 16)
TGSI_TEXTURE_SHADOWCUBE_ARRAY = enum_tgsi_texture_type.define('TGSI_TEXTURE_SHADOWCUBE_ARRAY', 17)
TGSI_TEXTURE_UNKNOWN = enum_tgsi_texture_type.define('TGSI_TEXTURE_UNKNOWN', 18)
TGSI_TEXTURE_COUNT = enum_tgsi_texture_type.define('TGSI_TEXTURE_COUNT', 19)

enum_lp_build_tex_modifier = CEnum(ctypes.c_uint32)
LP_BLD_TEX_MODIFIER_NONE = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_NONE', 0)
LP_BLD_TEX_MODIFIER_PROJECTED = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_PROJECTED', 1)
LP_BLD_TEX_MODIFIER_LOD_BIAS = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_LOD_BIAS', 2)
LP_BLD_TEX_MODIFIER_EXPLICIT_LOD = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_EXPLICIT_LOD', 3)
LP_BLD_TEX_MODIFIER_EXPLICIT_DERIV = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_EXPLICIT_DERIV', 4)
LP_BLD_TEX_MODIFIER_LOD_ZERO = enum_lp_build_tex_modifier.define('LP_BLD_TEX_MODIFIER_LOD_ZERO', 5)

struct_lp_build_sampler_aos._fields_ = [
  ('emit_fetch_texel', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_lp_build_sampler_aos), ctypes.POINTER(struct_lp_build_context), enum_tgsi_texture_type, ctypes.c_uint32, LLVMValueRef, struct_lp_derivatives, enum_lp_build_tex_modifier)),
]
try: (lp_build_nir_aos:=dll.lp_build_nir_aos).restype, lp_build_nir_aos.argtypes = None, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_nir_shader), struct_lp_type, (ctypes.c_ubyte * 4), LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.POINTER(LLVMValueRef), ctypes.POINTER(struct_lp_build_sampler_aos)]
except AttributeError: pass

class struct_lp_build_fn(Struct): pass
struct_lp_build_fn._fields_ = [
  ('fn_type', LLVMTypeRef),
  ('fn', LLVMValueRef),
]
try: (lp_build_nir_soa_prepasses:=dll.lp_build_nir_soa_prepasses).restype, lp_build_nir_soa_prepasses.argtypes = None, [ctypes.POINTER(struct_nir_shader)]
except AttributeError: pass

try: (lp_build_opt_nir:=dll.lp_build_opt_nir).restype, lp_build_opt_nir.argtypes = None, [ctypes.POINTER(struct_nir_shader)]
except AttributeError: pass

try: (lp_translate_atomic_op:=dll.lp_translate_atomic_op).restype, lp_translate_atomic_op.argtypes = LLVMAtomicRMWBinOp, [nir_atomic_op]
except AttributeError: pass

try: (lp_build_nir_sample_key:=dll.lp_build_nir_sample_key).restype, lp_build_nir_sample_key.argtypes = uint32_t, [gl_shader_stage, ctypes.POINTER(nir_tex_instr)]
except AttributeError: pass

try: (lp_img_op_from_intrinsic:=dll.lp_img_op_from_intrinsic).restype, lp_img_op_from_intrinsic.argtypes = None, [ctypes.POINTER(struct_lp_img_params), ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

try: (lp_packed_img_op_from_intrinsic:=dll.lp_packed_img_op_from_intrinsic).restype, lp_packed_img_op_from_intrinsic.argtypes = uint32_t, [ctypes.POINTER(nir_intrinsic_instr)]
except AttributeError: pass

enum_lp_nir_call_context_args = CEnum(ctypes.c_uint32)
LP_NIR_CALL_CONTEXT_CONTEXT = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_CONTEXT', 0)
LP_NIR_CALL_CONTEXT_RESOURCES = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_RESOURCES', 1)
LP_NIR_CALL_CONTEXT_SHARED = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_SHARED', 2)
LP_NIR_CALL_CONTEXT_SCRATCH = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_SCRATCH', 3)
LP_NIR_CALL_CONTEXT_WORK_DIM = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_WORK_DIM', 4)
LP_NIR_CALL_CONTEXT_THREAD_ID_0 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_THREAD_ID_0', 5)
LP_NIR_CALL_CONTEXT_THREAD_ID_1 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_THREAD_ID_1', 6)
LP_NIR_CALL_CONTEXT_THREAD_ID_2 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_THREAD_ID_2', 7)
LP_NIR_CALL_CONTEXT_BLOCK_ID_0 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_ID_0', 8)
LP_NIR_CALL_CONTEXT_BLOCK_ID_1 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_ID_1', 9)
LP_NIR_CALL_CONTEXT_BLOCK_ID_2 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_ID_2', 10)
LP_NIR_CALL_CONTEXT_GRID_SIZE_0 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_GRID_SIZE_0', 11)
LP_NIR_CALL_CONTEXT_GRID_SIZE_1 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_GRID_SIZE_1', 12)
LP_NIR_CALL_CONTEXT_GRID_SIZE_2 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_GRID_SIZE_2', 13)
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_0 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_SIZE_0', 14)
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_1 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_SIZE_1', 15)
LP_NIR_CALL_CONTEXT_BLOCK_SIZE_2 = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_BLOCK_SIZE_2', 16)
LP_NIR_CALL_CONTEXT_MAX_ARGS = enum_lp_nir_call_context_args.define('LP_NIR_CALL_CONTEXT_MAX_ARGS', 17)

try: (lp_build_cs_func_call_context:=dll.lp_build_cs_func_call_context).restype, lp_build_cs_func_call_context.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), ctypes.c_int32, LLVMTypeRef, LLVMTypeRef]
except AttributeError: pass

try: (lp_build_struct_get_ptr2:=dll.lp_build_struct_get_ptr2).restype, lp_build_struct_get_ptr2.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_struct_get2:=dll.lp_build_struct_get2).restype, lp_build_struct_get2.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_array_get_ptr2:=dll.lp_build_array_get_ptr2).restype, lp_build_array_get_ptr2.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_array_get2:=dll.lp_build_array_get2).restype, lp_build_array_get2.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_pointer_get2:=dll.lp_build_pointer_get2).restype, lp_build_pointer_get2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_pointer_get_unaligned2:=dll.lp_build_pointer_get_unaligned2).restype, lp_build_pointer_get_unaligned2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (lp_build_pointer_set:=dll.lp_build_pointer_set).restype, lp_build_pointer_set.argtypes = None, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_pointer_set_unaligned:=dll.lp_build_pointer_set_unaligned).restype, lp_build_pointer_set_unaligned.argtypes = None, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

class struct_lp_sampler_dynamic_state(Struct): pass
struct_lp_sampler_dynamic_state._fields_ = [
  ('width', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('height', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('depth', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('first_level', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('last_level', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('row_stride', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(LLVMTypeRef))),
  ('img_stride', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(LLVMTypeRef))),
  ('base_ptr', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('mip_offsets', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(LLVMTypeRef))),
  ('num_samples', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('sample_stride', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('min_lod', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('max_lod', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('lod_bias', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('border_color', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('cache_ptr', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32)),
  ('residency', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
  ('base_offset', ctypes.CFUNCTYPE(LLVMValueRef, ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef)),
]
class struct_lp_jit_buffer(Struct): pass
class struct_lp_jit_buffer_0(ctypes.Union): pass
struct_lp_jit_buffer_0._fields_ = [
  ('u', ctypes.POINTER(uint32_t)),
  ('f', ctypes.POINTER(ctypes.c_float)),
]
struct_lp_jit_buffer._anonymous_ = ['_0']
struct_lp_jit_buffer._fields_ = [
  ('_0', struct_lp_jit_buffer_0),
  ('num_elements', uint32_t),
]
_anonenum0 = CEnum(ctypes.c_uint32)
LP_JIT_BUFFER_BASE = _anonenum0.define('LP_JIT_BUFFER_BASE', 0)
LP_JIT_BUFFER_NUM_ELEMENTS = _anonenum0.define('LP_JIT_BUFFER_NUM_ELEMENTS', 1)
LP_JIT_BUFFER_NUM_FIELDS = _anonenum0.define('LP_JIT_BUFFER_NUM_FIELDS', 2)

try: (lp_llvm_descriptor_base:=dll.lp_llvm_descriptor_base).restype, lp_llvm_descriptor_base.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (lp_llvm_buffer_base:=dll.lp_llvm_buffer_base).restype, lp_llvm_buffer_base.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (lp_llvm_buffer_num_elements:=dll.lp_llvm_buffer_num_elements).restype, lp_llvm_buffer_num_elements.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

class struct_lp_jit_texture_0(ctypes.Union): pass
class struct_lp_jit_texture_0_0(Struct): pass
struct_lp_jit_texture_0_0._fields_ = [
  ('row_stride', (uint32_t * 16)),
  ('img_stride', (uint32_t * 16)),
]
struct_lp_jit_texture_0._anonymous_ = ['_0']
struct_lp_jit_texture_0._fields_ = [
  ('_0', struct_lp_jit_texture_0_0),
  ('residency', ctypes.c_void_p),
]
struct_lp_jit_texture._anonymous_ = ['_0']
struct_lp_jit_texture._fields_ = [
  ('base', ctypes.c_void_p),
  ('width', uint32_t),
  ('height', uint16_t),
  ('depth', uint16_t),
  ('_0', struct_lp_jit_texture_0),
  ('first_level', uint8_t),
  ('last_level', uint8_t),
  ('mip_offsets', (uint32_t * 16)),
  ('sampler_index', uint32_t),
]
_anonenum1 = CEnum(ctypes.c_uint32)
LP_JIT_TEXTURE_BASE = _anonenum1.define('LP_JIT_TEXTURE_BASE', 0)
LP_JIT_TEXTURE_WIDTH = _anonenum1.define('LP_JIT_TEXTURE_WIDTH', 1)
LP_JIT_TEXTURE_HEIGHT = _anonenum1.define('LP_JIT_TEXTURE_HEIGHT', 2)
LP_JIT_TEXTURE_DEPTH = _anonenum1.define('LP_JIT_TEXTURE_DEPTH', 3)
LP_JIT_TEXTURE_ROW_STRIDE = _anonenum1.define('LP_JIT_TEXTURE_ROW_STRIDE', 4)
LP_JIT_TEXTURE_IMG_STRIDE = _anonenum1.define('LP_JIT_TEXTURE_IMG_STRIDE', 5)
LP_JIT_TEXTURE_FIRST_LEVEL = _anonenum1.define('LP_JIT_TEXTURE_FIRST_LEVEL', 6)
LP_JIT_TEXTURE_LAST_LEVEL = _anonenum1.define('LP_JIT_TEXTURE_LAST_LEVEL', 7)
LP_JIT_TEXTURE_MIP_OFFSETS = _anonenum1.define('LP_JIT_TEXTURE_MIP_OFFSETS', 8)
LP_JIT_SAMPLER_INDEX_DUMMY = _anonenum1.define('LP_JIT_SAMPLER_INDEX_DUMMY', 9)
LP_JIT_TEXTURE_NUM_FIELDS = _anonenum1.define('LP_JIT_TEXTURE_NUM_FIELDS', 10)

class struct_lp_jit_sampler(Struct): pass
struct_lp_jit_sampler._fields_ = [
  ('min_lod', ctypes.c_float),
  ('max_lod', ctypes.c_float),
  ('lod_bias', ctypes.c_float),
  ('border_color', (ctypes.c_float * 4)),
]
_anonenum2 = CEnum(ctypes.c_uint32)
LP_JIT_SAMPLER_MIN_LOD = _anonenum2.define('LP_JIT_SAMPLER_MIN_LOD', 0)
LP_JIT_SAMPLER_MAX_LOD = _anonenum2.define('LP_JIT_SAMPLER_MAX_LOD', 1)
LP_JIT_SAMPLER_LOD_BIAS = _anonenum2.define('LP_JIT_SAMPLER_LOD_BIAS', 2)
LP_JIT_SAMPLER_BORDER_COLOR = _anonenum2.define('LP_JIT_SAMPLER_BORDER_COLOR', 3)
LP_JIT_SAMPLER_NUM_FIELDS = _anonenum2.define('LP_JIT_SAMPLER_NUM_FIELDS', 4)

class struct_lp_jit_image(Struct): pass
struct_lp_jit_image._fields_ = [
  ('base', ctypes.c_void_p),
  ('width', uint32_t),
  ('height', uint16_t),
  ('depth', uint16_t),
  ('num_samples', uint8_t),
  ('sample_stride', uint32_t),
  ('row_stride', uint32_t),
  ('img_stride', uint32_t),
  ('residency', ctypes.c_void_p),
  ('base_offset', uint32_t),
]
_anonenum3 = CEnum(ctypes.c_uint32)
LP_JIT_IMAGE_BASE = _anonenum3.define('LP_JIT_IMAGE_BASE', 0)
LP_JIT_IMAGE_WIDTH = _anonenum3.define('LP_JIT_IMAGE_WIDTH', 1)
LP_JIT_IMAGE_HEIGHT = _anonenum3.define('LP_JIT_IMAGE_HEIGHT', 2)
LP_JIT_IMAGE_DEPTH = _anonenum3.define('LP_JIT_IMAGE_DEPTH', 3)
LP_JIT_IMAGE_NUM_SAMPLES = _anonenum3.define('LP_JIT_IMAGE_NUM_SAMPLES', 4)
LP_JIT_IMAGE_SAMPLE_STRIDE = _anonenum3.define('LP_JIT_IMAGE_SAMPLE_STRIDE', 5)
LP_JIT_IMAGE_ROW_STRIDE = _anonenum3.define('LP_JIT_IMAGE_ROW_STRIDE', 6)
LP_JIT_IMAGE_IMG_STRIDE = _anonenum3.define('LP_JIT_IMAGE_IMG_STRIDE', 7)
LP_JIT_IMAGE_RESIDENCY = _anonenum3.define('LP_JIT_IMAGE_RESIDENCY', 8)
LP_JIT_IMAGE_BASE_OFFSET = _anonenum3.define('LP_JIT_IMAGE_BASE_OFFSET', 9)
LP_JIT_IMAGE_NUM_FIELDS = _anonenum3.define('LP_JIT_IMAGE_NUM_FIELDS', 10)

class struct_lp_jit_resources(Struct): pass
struct_lp_jit_resources._fields_ = [
  ('constants', (struct_lp_jit_buffer * 16)),
  ('ssbos', (struct_lp_jit_buffer * 32)),
  ('textures', (struct_lp_jit_texture * 128)),
  ('samplers', (struct_lp_jit_sampler * 32)),
  ('images', (struct_lp_jit_image * 64)),
]
_anonenum4 = CEnum(ctypes.c_uint32)
LP_JIT_RES_CONSTANTS = _anonenum4.define('LP_JIT_RES_CONSTANTS', 0)
LP_JIT_RES_SSBOS = _anonenum4.define('LP_JIT_RES_SSBOS', 1)
LP_JIT_RES_TEXTURES = _anonenum4.define('LP_JIT_RES_TEXTURES', 2)
LP_JIT_RES_SAMPLERS = _anonenum4.define('LP_JIT_RES_SAMPLERS', 3)
LP_JIT_RES_IMAGES = _anonenum4.define('LP_JIT_RES_IMAGES', 4)
LP_JIT_RES_COUNT = _anonenum4.define('LP_JIT_RES_COUNT', 5)

try: (lp_build_jit_resources_type:=dll.lp_build_jit_resources_type).restype, lp_build_jit_resources_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

_anonenum5 = CEnum(ctypes.c_uint32)
LP_JIT_VERTEX_HEADER_VERTEX_ID = _anonenum5.define('LP_JIT_VERTEX_HEADER_VERTEX_ID', 0)
LP_JIT_VERTEX_HEADER_CLIP_POS = _anonenum5.define('LP_JIT_VERTEX_HEADER_CLIP_POS', 1)
LP_JIT_VERTEX_HEADER_DATA = _anonenum5.define('LP_JIT_VERTEX_HEADER_DATA', 2)

try: (lp_build_create_jit_vertex_header_type:=dll.lp_build_create_jit_vertex_header_type).restype, lp_build_create_jit_vertex_header_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), ctypes.c_int32]
except AttributeError: pass

try: (lp_build_jit_fill_sampler_dynamic_state:=dll.lp_build_jit_fill_sampler_dynamic_state).restype, lp_build_jit_fill_sampler_dynamic_state.argtypes = None, [ctypes.POINTER(struct_lp_sampler_dynamic_state)]
except AttributeError: pass

try: (lp_build_jit_fill_image_dynamic_state:=dll.lp_build_jit_fill_image_dynamic_state).restype, lp_build_jit_fill_image_dynamic_state.argtypes = None, [ctypes.POINTER(struct_lp_sampler_dynamic_state)]
except AttributeError: pass

try: (lp_build_sample_function_type:=dll.lp_build_sample_function_type).restype, lp_build_sample_function_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), uint32_t]
except AttributeError: pass

try: (lp_build_size_function_type:=dll.lp_build_size_function_type).restype, lp_build_size_function_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_sampler_size_query_params)]
except AttributeError: pass

try: (lp_build_image_function_type:=dll.lp_build_image_function_type).restype, lp_build_image_function_type.argtypes = LLVMTypeRef, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(struct_lp_img_params), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

class struct_lp_texture_handle_state(Struct): pass
class struct_lp_static_texture_state(Struct): pass
enum_pipe_texture_target = CEnum(ctypes.c_uint32)
PIPE_BUFFER = enum_pipe_texture_target.define('PIPE_BUFFER', 0)
PIPE_TEXTURE_1D = enum_pipe_texture_target.define('PIPE_TEXTURE_1D', 1)
PIPE_TEXTURE_2D = enum_pipe_texture_target.define('PIPE_TEXTURE_2D', 2)
PIPE_TEXTURE_3D = enum_pipe_texture_target.define('PIPE_TEXTURE_3D', 3)
PIPE_TEXTURE_CUBE = enum_pipe_texture_target.define('PIPE_TEXTURE_CUBE', 4)
PIPE_TEXTURE_RECT = enum_pipe_texture_target.define('PIPE_TEXTURE_RECT', 5)
PIPE_TEXTURE_1D_ARRAY = enum_pipe_texture_target.define('PIPE_TEXTURE_1D_ARRAY', 6)
PIPE_TEXTURE_2D_ARRAY = enum_pipe_texture_target.define('PIPE_TEXTURE_2D_ARRAY', 7)
PIPE_TEXTURE_CUBE_ARRAY = enum_pipe_texture_target.define('PIPE_TEXTURE_CUBE_ARRAY', 8)
PIPE_MAX_TEXTURE_TYPES = enum_pipe_texture_target.define('PIPE_MAX_TEXTURE_TYPES', 9)

struct_lp_static_texture_state._fields_ = [
  ('format', enum_pipe_format),
  ('res_format', enum_pipe_format),
  ('swizzle_r', ctypes.c_uint32,3),
  ('swizzle_g', ctypes.c_uint32,3),
  ('swizzle_b', ctypes.c_uint32,3),
  ('swizzle_a', ctypes.c_uint32,3),
  ('target', enum_pipe_texture_target,5),
  ('res_target', enum_pipe_texture_target,5),
  ('pot_width', ctypes.c_uint32,1),
  ('pot_height', ctypes.c_uint32,1),
  ('pot_depth', ctypes.c_uint32,1),
  ('level_zero_only', ctypes.c_uint32,1),
  ('tiled', ctypes.c_uint32,1),
  ('tiled_samples', ctypes.c_uint32,5),
]
struct_lp_texture_handle_state._fields_ = [
  ('static_state', struct_lp_static_texture_state),
  ('dynamic_state', struct_lp_jit_texture),
]
class struct_lp_texture_functions(Struct): pass
struct_lp_texture_functions._fields_ = [
  ('sample_functions', ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
  ('sampler_count', uint32_t),
  ('fetch_functions', ctypes.POINTER(ctypes.c_void_p)),
  ('size_function', ctypes.c_void_p),
  ('samples_function', ctypes.c_void_p),
  ('image_functions', ctypes.POINTER(ctypes.c_void_p)),
  ('state', struct_lp_texture_handle_state),
  ('sampled', ctypes.c_bool),
  ('storage', ctypes.c_bool),
  ('matrix', ctypes.c_void_p),
]
class struct_lp_texture_handle(Struct): pass
struct_lp_texture_handle._fields_ = [
  ('functions', ctypes.c_void_p),
  ('sampler_index', uint32_t),
]
class struct_lp_jit_bindless_texture(Struct): pass
struct_lp_jit_bindless_texture._fields_ = [
  ('base', ctypes.c_void_p),
  ('residency', ctypes.c_void_p),
  ('sampler_index', uint32_t),
]
class struct_lp_descriptor(Struct): pass
class struct_lp_descriptor_0(ctypes.Union): pass
class struct_lp_descriptor_0_0(Struct): pass
struct_lp_descriptor_0_0._fields_ = [
  ('texture', struct_lp_jit_bindless_texture),
  ('sampler', struct_lp_jit_sampler),
]
class struct_lp_descriptor_0_1(Struct): pass
struct_lp_descriptor_0_1._fields_ = [
  ('image', struct_lp_jit_image),
]
struct_lp_descriptor_0._anonymous_ = ['_0', '_1']
struct_lp_descriptor_0._fields_ = [
  ('_0', struct_lp_descriptor_0_0),
  ('_1', struct_lp_descriptor_0_1),
  ('buffer', struct_lp_jit_buffer),
  ('accel_struct', uint64_t),
]
struct_lp_descriptor._anonymous_ = ['_0']
struct_lp_descriptor._fields_ = [
  ('_0', struct_lp_descriptor_0),
  ('functions', ctypes.c_void_p),
]
try: (lp_build_flow_skip_begin:=dll.lp_build_flow_skip_begin).restype, lp_build_flow_skip_begin.argtypes = None, [ctypes.POINTER(struct_lp_build_skip_context), ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (lp_build_flow_skip_cond_break:=dll.lp_build_flow_skip_cond_break).restype, lp_build_flow_skip_cond_break.argtypes = None, [ctypes.POINTER(struct_lp_build_skip_context), LLVMValueRef]
except AttributeError: pass

try: (lp_build_flow_skip_end:=dll.lp_build_flow_skip_end).restype, lp_build_flow_skip_end.argtypes = None, [ctypes.POINTER(struct_lp_build_skip_context)]
except AttributeError: pass

try: (lp_build_mask_begin:=dll.lp_build_mask_begin).restype, lp_build_mask_begin.argtypes = None, [ctypes.POINTER(struct_lp_build_mask_context), ctypes.POINTER(struct_gallivm_state), struct_lp_type, LLVMValueRef]
except AttributeError: pass

try: (lp_build_mask_value:=dll.lp_build_mask_value).restype, lp_build_mask_value.argtypes = LLVMValueRef, [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError: pass

try: (lp_build_mask_update:=dll.lp_build_mask_update).restype, lp_build_mask_update.argtypes = None, [ctypes.POINTER(struct_lp_build_mask_context), LLVMValueRef]
except AttributeError: pass

try: (lp_build_mask_force:=dll.lp_build_mask_force).restype, lp_build_mask_force.argtypes = None, [ctypes.POINTER(struct_lp_build_mask_context), LLVMValueRef]
except AttributeError: pass

try: (lp_build_mask_check:=dll.lp_build_mask_check).restype, lp_build_mask_check.argtypes = None, [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError: pass

try: (lp_build_mask_end:=dll.lp_build_mask_end).restype, lp_build_mask_end.argtypes = LLVMValueRef, [ctypes.POINTER(struct_lp_build_mask_context)]
except AttributeError: pass

class struct_lp_build_loop_state(Struct): pass
struct_lp_build_loop_state._fields_ = [
  ('block', LLVMBasicBlockRef),
  ('counter_var', LLVMValueRef),
  ('counter', LLVMValueRef),
  ('counter_type', LLVMTypeRef),
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
]
try: (lp_build_loop_begin:=dll.lp_build_loop_begin).restype, lp_build_loop_begin.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError: pass

try: (lp_build_loop_end:=dll.lp_build_loop_end).restype, lp_build_loop_end.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_loop_force_set_counter:=dll.lp_build_loop_force_set_counter).restype, lp_build_loop_force_set_counter.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef]
except AttributeError: pass

try: (lp_build_loop_force_reload_counter:=dll.lp_build_loop_force_reload_counter).restype, lp_build_loop_force_reload_counter.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state)]
except AttributeError: pass

LLVMIntPredicate = CEnum(ctypes.c_uint32)
LLVMIntEQ = LLVMIntPredicate.define('LLVMIntEQ', 32)
LLVMIntNE = LLVMIntPredicate.define('LLVMIntNE', 33)
LLVMIntUGT = LLVMIntPredicate.define('LLVMIntUGT', 34)
LLVMIntUGE = LLVMIntPredicate.define('LLVMIntUGE', 35)
LLVMIntULT = LLVMIntPredicate.define('LLVMIntULT', 36)
LLVMIntULE = LLVMIntPredicate.define('LLVMIntULE', 37)
LLVMIntSGT = LLVMIntPredicate.define('LLVMIntSGT', 38)
LLVMIntSGE = LLVMIntPredicate.define('LLVMIntSGE', 39)
LLVMIntSLT = LLVMIntPredicate.define('LLVMIntSLT', 40)
LLVMIntSLE = LLVMIntPredicate.define('LLVMIntSLE', 41)

try: (lp_build_loop_end_cond:=dll.lp_build_loop_end_cond).restype, lp_build_loop_end_cond.argtypes = None, [ctypes.POINTER(struct_lp_build_loop_state), LLVMValueRef, LLVMValueRef, LLVMIntPredicate]
except AttributeError: pass

class struct_lp_build_for_loop_state(Struct): pass
struct_lp_build_for_loop_state._fields_ = [
  ('begin', LLVMBasicBlockRef),
  ('body', LLVMBasicBlockRef),
  ('exit', LLVMBasicBlockRef),
  ('counter_var', LLVMValueRef),
  ('counter', LLVMValueRef),
  ('counter_type', LLVMTypeRef),
  ('step', LLVMValueRef),
  ('cond', LLVMIntPredicate),
  ('end', LLVMValueRef),
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
]
try: (lp_build_for_loop_begin:=dll.lp_build_for_loop_begin).restype, lp_build_for_loop_begin.argtypes = None, [ctypes.POINTER(struct_lp_build_for_loop_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef, LLVMIntPredicate, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (lp_build_for_loop_end:=dll.lp_build_for_loop_end).restype, lp_build_for_loop_end.argtypes = None, [ctypes.POINTER(struct_lp_build_for_loop_state)]
except AttributeError: pass

class struct_lp_build_if_state(Struct): pass
struct_lp_build_if_state._fields_ = [
  ('gallivm', ctypes.POINTER(struct_gallivm_state)),
  ('condition', LLVMValueRef),
  ('entry_block', LLVMBasicBlockRef),
  ('true_block', LLVMBasicBlockRef),
  ('false_block', LLVMBasicBlockRef),
  ('merge_block', LLVMBasicBlockRef),
]
try: (lp_build_if:=dll.lp_build_if).restype, lp_build_if.argtypes = None, [ctypes.POINTER(struct_lp_build_if_state), ctypes.POINTER(struct_gallivm_state), LLVMValueRef]
except AttributeError: pass

try: (lp_build_else:=dll.lp_build_else).restype, lp_build_else.argtypes = None, [ctypes.POINTER(struct_lp_build_if_state)]
except AttributeError: pass

try: (lp_build_endif:=dll.lp_build_endif).restype, lp_build_endif.argtypes = None, [ctypes.POINTER(struct_lp_build_if_state)]
except AttributeError: pass

try: (lp_build_insert_new_block:=dll.lp_build_insert_new_block).restype, lp_build_insert_new_block.argtypes = LLVMBasicBlockRef, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_create_builder_at_entry:=dll.lp_create_builder_at_entry).restype, lp_create_builder_at_entry.argtypes = LLVMBuilderRef, [ctypes.POINTER(struct_gallivm_state)]
except AttributeError: pass

try: (lp_build_alloca:=dll.lp_build_alloca).restype, lp_build_alloca.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_alloca_undef:=dll.lp_build_alloca_undef).restype, lp_build_alloca_undef.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_array_alloca:=dll.lp_build_array_alloca).restype, lp_build_array_alloca.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_mantissa:=dll.lp_mantissa).restype, lp_mantissa.argtypes = ctypes.c_uint32, [struct_lp_type]
except AttributeError: pass

try: (lp_const_shift:=dll.lp_const_shift).restype, lp_const_shift.argtypes = ctypes.c_uint32, [struct_lp_type]
except AttributeError: pass

try: (lp_const_offset:=dll.lp_const_offset).restype, lp_const_offset.argtypes = ctypes.c_uint32, [struct_lp_type]
except AttributeError: pass

try: (lp_const_scale:=dll.lp_const_scale).restype, lp_const_scale.argtypes = ctypes.c_double, [struct_lp_type]
except AttributeError: pass

try: (lp_const_min:=dll.lp_const_min).restype, lp_const_min.argtypes = ctypes.c_double, [struct_lp_type]
except AttributeError: pass

try: (lp_const_max:=dll.lp_const_max).restype, lp_const_max.argtypes = ctypes.c_double, [struct_lp_type]
except AttributeError: pass

try: (lp_const_eps:=dll.lp_const_eps).restype, lp_const_eps.argtypes = ctypes.c_double, [struct_lp_type]
except AttributeError: pass

try: (lp_build_undef:=dll.lp_build_undef).restype, lp_build_undef.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_zero:=dll.lp_build_zero).restype, lp_build_zero.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_one:=dll.lp_build_one).restype, lp_build_one.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_const_elem:=dll.lp_build_const_elem).restype, lp_build_const_elem.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double]
except AttributeError: pass

try: (lp_build_const_vec:=dll.lp_build_const_vec).restype, lp_build_const_vec.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double]
except AttributeError: pass

try: (lp_build_const_int_vec:=dll.lp_build_const_int_vec).restype, lp_build_const_int_vec.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_int64]
except AttributeError: pass

try: (lp_build_const_channel_vec:=dll.lp_build_const_channel_vec).restype, lp_build_const_channel_vec.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type]
except AttributeError: pass

try: (lp_build_const_aos:=dll.lp_build_const_aos).restype, lp_build_const_aos.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError: pass

try: (lp_build_const_mask_aos:=dll.lp_build_const_mask_aos).restype, lp_build_const_mask_aos.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (lp_build_const_mask_aos_swizzled:=dll.lp_build_const_mask_aos_swizzled).restype, lp_build_const_mask_aos_swizzled.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), struct_lp_type, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError: pass

try: (lp_build_const_string:=dll.lp_build_const_string).restype, lp_build_const_string.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_const_func_pointer:=dll.lp_build_const_func_pointer).restype, lp_build_const_func_pointer.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), ctypes.c_void_p, LLVMTypeRef, ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lp_build_const_func_pointer_from_type:=dll.lp_build_const_func_pointer_from_type).restype, lp_build_const_func_pointer_from_type.argtypes = LLVMValueRef, [ctypes.POINTER(struct_gallivm_state), ctypes.c_void_p, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_type_singleton_init_or_ref:=dll.glsl_type_singleton_init_or_ref).restype, glsl_type_singleton_init_or_ref.argtypes = None, []
except AttributeError: pass

try: (glsl_type_singleton_decref:=dll.glsl_type_singleton_decref).restype, glsl_type_singleton_decref.argtypes = None, []
except AttributeError: pass

try: (encode_type_to_blob:=dll.encode_type_to_blob).restype, encode_type_to_blob.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (decode_type_from_blob:=dll.decode_type_from_blob).restype, decode_type_from_blob.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (glsl_apply_signedness_to_base_type:=dll.glsl_apply_signedness_to_base_type).restype, glsl_apply_signedness_to_base_type.argtypes = enum_glsl_base_type, [enum_glsl_base_type, ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_sampler_dim_coordinate_components:=dll.glsl_get_sampler_dim_coordinate_components).restype, glsl_get_sampler_dim_coordinate_components.argtypes = ctypes.c_int32, [enum_glsl_sampler_dim]
except AttributeError: pass

enum_glsl_matrix_layout = CEnum(ctypes.c_uint32)
GLSL_MATRIX_LAYOUT_INHERITED = enum_glsl_matrix_layout.define('GLSL_MATRIX_LAYOUT_INHERITED', 0)
GLSL_MATRIX_LAYOUT_COLUMN_MAJOR = enum_glsl_matrix_layout.define('GLSL_MATRIX_LAYOUT_COLUMN_MAJOR', 1)
GLSL_MATRIX_LAYOUT_ROW_MAJOR = enum_glsl_matrix_layout.define('GLSL_MATRIX_LAYOUT_ROW_MAJOR', 2)

_anonenum6 = CEnum(ctypes.c_uint32)
GLSL_PRECISION_NONE = _anonenum6.define('GLSL_PRECISION_NONE', 0)
GLSL_PRECISION_HIGH = _anonenum6.define('GLSL_PRECISION_HIGH', 1)
GLSL_PRECISION_MEDIUM = _anonenum6.define('GLSL_PRECISION_MEDIUM', 2)
GLSL_PRECISION_LOW = _anonenum6.define('GLSL_PRECISION_LOW', 3)

enum_glsl_cmat_use = CEnum(ctypes.c_uint32)
GLSL_CMAT_USE_NONE = enum_glsl_cmat_use.define('GLSL_CMAT_USE_NONE', 0)
GLSL_CMAT_USE_A = enum_glsl_cmat_use.define('GLSL_CMAT_USE_A', 1)
GLSL_CMAT_USE_B = enum_glsl_cmat_use.define('GLSL_CMAT_USE_B', 2)
GLSL_CMAT_USE_ACCUMULATOR = enum_glsl_cmat_use.define('GLSL_CMAT_USE_ACCUMULATOR', 3)

try: (glsl_get_type_name:=dll.glsl_get_type_name).restype, glsl_get_type_name.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_vector:=dll.glsl_type_is_vector).restype, glsl_type_is_vector.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_scalar:=dll.glsl_type_is_scalar).restype, glsl_type_is_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_vector_or_scalar:=dll.glsl_type_is_vector_or_scalar).restype, glsl_type_is_vector_or_scalar.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_matrix:=dll.glsl_type_is_matrix).restype, glsl_type_is_matrix.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_array_or_matrix:=dll.glsl_type_is_array_or_matrix).restype, glsl_type_is_array_or_matrix.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_dual_slot:=dll.glsl_type_is_dual_slot).restype, glsl_type_is_dual_slot.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_is_leaf:=dll.glsl_type_is_leaf).restype, glsl_type_is_leaf.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_bare_type:=dll.glsl_get_bare_type).restype, glsl_get_bare_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_scalar_type:=dll.glsl_get_scalar_type).restype, glsl_get_scalar_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_base_glsl_type:=dll.glsl_get_base_glsl_type).restype, glsl_get_base_glsl_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_length:=dll.glsl_get_length).restype, glsl_get_length.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_wrap_in_arrays:=dll.glsl_type_wrap_in_arrays).restype, glsl_type_wrap_in_arrays.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_aoa_size:=dll.glsl_get_aoa_size).restype, glsl_get_aoa_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_array_element:=dll.glsl_get_array_element).restype, glsl_get_array_element.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_without_array:=dll.glsl_without_array).restype, glsl_without_array.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_without_array_or_matrix:=dll.glsl_without_array_or_matrix).restype, glsl_without_array_or_matrix.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_wrap_in_arrays:=dll.glsl_type_wrap_in_arrays).restype, glsl_type_wrap_in_arrays.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cmat_element:=dll.glsl_get_cmat_element).restype, glsl_get_cmat_element.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cmat_description:=dll.glsl_get_cmat_description).restype, glsl_get_cmat_description.argtypes = ctypes.POINTER(struct_glsl_cmat_description), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_atomic_size:=dll.glsl_atomic_size).restype, glsl_atomic_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_contains_32bit:=dll.glsl_type_contains_32bit).restype, glsl_type_contains_32bit.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_contains_64bit:=dll.glsl_type_contains_64bit).restype, glsl_type_contains_64bit.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_contains_image:=dll.glsl_type_contains_image).restype, glsl_type_contains_image.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_atomic:=dll.glsl_contains_atomic).restype, glsl_contains_atomic.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_double:=dll.glsl_contains_double).restype, glsl_contains_double.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_integer:=dll.glsl_contains_integer).restype, glsl_contains_integer.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_opaque:=dll.glsl_contains_opaque).restype, glsl_contains_opaque.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_sampler:=dll.glsl_contains_sampler).restype, glsl_contains_sampler.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_array:=dll.glsl_contains_array).restype, glsl_contains_array.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_contains_subroutine:=dll.glsl_contains_subroutine).restype, glsl_contains_subroutine.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_sampler_coordinate_components:=dll.glsl_get_sampler_coordinate_components).restype, glsl_get_sampler_coordinate_components.argtypes = ctypes.c_int32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_compare_no_precision:=dll.glsl_type_compare_no_precision).restype, glsl_type_compare_no_precision.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_record_compare:=dll.glsl_record_compare).restype, glsl_record_compare.argtypes = ctypes.c_bool, [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_struct_field:=dll.glsl_get_struct_field).restype, glsl_get_struct_field.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_get_struct_field_data:=dll.glsl_get_struct_field_data).restype, glsl_get_struct_field_data.argtypes = ctypes.POINTER(glsl_struct_field), [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_get_struct_location_offset:=dll.glsl_get_struct_location_offset).restype, glsl_get_struct_location_offset.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_get_field_index:=dll.glsl_get_field_index).restype, glsl_get_field_index.argtypes = ctypes.c_int32, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_get_field_type:=dll.glsl_get_field_type).restype, glsl_get_field_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_vec_type:=dll.glsl_vec_type).restype, glsl_vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_f16vec_type:=dll.glsl_f16vec_type).restype, glsl_f16vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_bf16vec_type:=dll.glsl_bf16vec_type).restype, glsl_bf16vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_e4m3fnvec_type:=dll.glsl_e4m3fnvec_type).restype, glsl_e4m3fnvec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_e5m2vec_type:=dll.glsl_e5m2vec_type).restype, glsl_e5m2vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_dvec_type:=dll.glsl_dvec_type).restype, glsl_dvec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_ivec_type:=dll.glsl_ivec_type).restype, glsl_ivec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_uvec_type:=dll.glsl_uvec_type).restype, glsl_uvec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_bvec_type:=dll.glsl_bvec_type).restype, glsl_bvec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_i64vec_type:=dll.glsl_i64vec_type).restype, glsl_i64vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_u64vec_type:=dll.glsl_u64vec_type).restype, glsl_u64vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_i16vec_type:=dll.glsl_i16vec_type).restype, glsl_i16vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_u16vec_type:=dll.glsl_u16vec_type).restype, glsl_u16vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_i8vec_type:=dll.glsl_i8vec_type).restype, glsl_i8vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_u8vec_type:=dll.glsl_u8vec_type).restype, glsl_u8vec_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32]
except AttributeError: pass

try: (glsl_simple_explicit_type:=dll.glsl_simple_explicit_type).restype, glsl_simple_explicit_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
except AttributeError: pass

try: (glsl_sampler_type:=dll.glsl_sampler_type).restype, glsl_sampler_type.argtypes = ctypes.POINTER(glsl_type), [enum_glsl_sampler_dim, ctypes.c_bool, ctypes.c_bool, enum_glsl_base_type]
except AttributeError: pass

try: (glsl_bare_sampler_type:=dll.glsl_bare_sampler_type).restype, glsl_bare_sampler_type.argtypes = ctypes.POINTER(glsl_type), []
except AttributeError: pass

try: (glsl_bare_shadow_sampler_type:=dll.glsl_bare_shadow_sampler_type).restype, glsl_bare_shadow_sampler_type.argtypes = ctypes.POINTER(glsl_type), []
except AttributeError: pass

try: (glsl_texture_type:=dll.glsl_texture_type).restype, glsl_texture_type.argtypes = ctypes.POINTER(glsl_type), [enum_glsl_sampler_dim, ctypes.c_bool, enum_glsl_base_type]
except AttributeError: pass

try: (glsl_image_type:=dll.glsl_image_type).restype, glsl_image_type.argtypes = ctypes.POINTER(glsl_type), [enum_glsl_sampler_dim, ctypes.c_bool, enum_glsl_base_type]
except AttributeError: pass

try: (glsl_array_type:=dll.glsl_array_type).restype, glsl_array_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (glsl_cmat_type:=dll.glsl_cmat_type).restype, glsl_cmat_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(struct_glsl_cmat_description)]
except AttributeError: pass

try: (glsl_struct_type_with_explicit_alignment:=dll.glsl_struct_type_with_explicit_alignment).restype, glsl_struct_type_with_explicit_alignment.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_struct_field), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_bool, ctypes.c_uint32]
except AttributeError: pass

enum_glsl_interface_packing = CEnum(ctypes.c_uint32)
GLSL_INTERFACE_PACKING_STD140 = enum_glsl_interface_packing.define('GLSL_INTERFACE_PACKING_STD140', 0)
GLSL_INTERFACE_PACKING_SHARED = enum_glsl_interface_packing.define('GLSL_INTERFACE_PACKING_SHARED', 1)
GLSL_INTERFACE_PACKING_PACKED = enum_glsl_interface_packing.define('GLSL_INTERFACE_PACKING_PACKED', 2)
GLSL_INTERFACE_PACKING_STD430 = enum_glsl_interface_packing.define('GLSL_INTERFACE_PACKING_STD430', 3)

try: (glsl_interface_type:=dll.glsl_interface_type).restype, glsl_interface_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_struct_field), ctypes.c_uint32, enum_glsl_interface_packing, ctypes.c_bool, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_subroutine_type:=dll.glsl_subroutine_type).restype, glsl_subroutine_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (glsl_get_row_type:=dll.glsl_get_row_type).restype, glsl_get_row_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_column_type:=dll.glsl_get_column_type).restype, glsl_get_column_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_explicit_type_for_size_align:=dll.glsl_get_explicit_type_for_size_align).restype, glsl_get_explicit_type_for_size_align.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), glsl_type_size_align_func, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_type_replace_vec3_with_vec4:=dll.glsl_type_replace_vec3_with_vec4).restype, glsl_type_replace_vec3_with_vec4.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_float16_type:=dll.glsl_float16_type).restype, glsl_float16_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_int16_type:=dll.glsl_int16_type).restype, glsl_int16_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_uint16_type:=dll.glsl_uint16_type).restype, glsl_uint16_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_to_16bit:=dll.glsl_type_to_16bit).restype, glsl_type_to_16bit.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_replace_vector_type:=dll.glsl_replace_vector_type).restype, glsl_replace_vector_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_channel_type:=dll.glsl_channel_type).restype, glsl_channel_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_mul_type:=dll.glsl_get_mul_type).restype, glsl_get_mul_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_get_sampler_count:=dll.glsl_type_get_sampler_count).restype, glsl_type_get_sampler_count.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_get_texture_count:=dll.glsl_type_get_texture_count).restype, glsl_type_get_texture_count.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_get_image_count:=dll.glsl_type_get_image_count).restype, glsl_type_get_image_count.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_count_vec4_slots:=dll.glsl_count_vec4_slots).restype, glsl_count_vec4_slots.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool, ctypes.c_bool]
except AttributeError: pass

try: (glsl_count_dword_slots:=dll.glsl_count_dword_slots).restype, glsl_count_dword_slots.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_component_slots:=dll.glsl_get_component_slots).restype, glsl_get_component_slots.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_component_slots_aligned:=dll.glsl_get_component_slots_aligned).restype, glsl_get_component_slots_aligned.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_uint32]
except AttributeError: pass

try: (glsl_varying_count:=dll.glsl_varying_count).restype, glsl_varying_count.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_type_uniform_locations:=dll.glsl_type_uniform_locations).restype, glsl_type_uniform_locations.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cl_size:=dll.glsl_get_cl_size).restype, glsl_get_cl_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cl_alignment:=dll.glsl_get_cl_alignment).restype, glsl_get_cl_alignment.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type)]
except AttributeError: pass

try: (glsl_get_cl_type_size_align:=dll.glsl_get_cl_type_size_align).restype, glsl_get_cl_type_size_align.argtypes = None, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_get_internal_ifc_packing:=dll.glsl_get_internal_ifc_packing).restype, glsl_get_internal_ifc_packing.argtypes = enum_glsl_interface_packing, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std140_base_alignment:=dll.glsl_get_std140_base_alignment).restype, glsl_get_std140_base_alignment.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std140_size:=dll.glsl_get_std140_size).restype, glsl_get_std140_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std430_array_stride:=dll.glsl_get_std430_array_stride).restype, glsl_get_std430_array_stride.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std430_base_alignment:=dll.glsl_get_std430_base_alignment).restype, glsl_get_std430_base_alignment.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_std430_size:=dll.glsl_get_std430_size).restype, glsl_get_std430_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_explicit_size:=dll.glsl_get_explicit_size).restype, glsl_get_explicit_size.argtypes = ctypes.c_uint32, [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_explicit_std140_type:=dll.glsl_get_explicit_std140_type).restype, glsl_get_explicit_std140_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_get_explicit_std430_type:=dll.glsl_get_explicit_std430_type).restype, glsl_get_explicit_std430_type.argtypes = ctypes.POINTER(glsl_type), [ctypes.POINTER(glsl_type), ctypes.c_bool]
except AttributeError: pass

try: (glsl_size_align_handle_array_and_structs:=dll.glsl_size_align_handle_array_and_structs).restype, glsl_size_align_handle_array_and_structs.argtypes = None, [ctypes.POINTER(glsl_type), glsl_type_size_align_func, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_get_natural_size_align_bytes:=dll.glsl_get_natural_size_align_bytes).restype, glsl_get_natural_size_align_bytes.argtypes = None, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_get_word_size_align_bytes:=dll.glsl_get_word_size_align_bytes).restype, glsl_get_word_size_align_bytes.argtypes = None, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (glsl_get_vec4_size_align_bytes:=dll.glsl_get_vec4_size_align_bytes).restype, glsl_get_vec4_size_align_bytes.argtypes = None, [ctypes.POINTER(glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (blob_init:=dll.blob_init).restype, blob_init.argtypes = None, [ctypes.POINTER(struct_blob)]
except AttributeError: pass

try: (blob_init_fixed:=dll.blob_init_fixed).restype, blob_init_fixed.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.c_void_p, size_t]
except AttributeError: pass

try: (blob_finish_get_buffer:=dll.blob_finish_get_buffer).restype, blob_finish_get_buffer.argtypes = None, [ctypes.POINTER(struct_blob), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t)]
except AttributeError: pass

try: (blob_align:=dll.blob_align).restype, blob_align.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t]
except AttributeError: pass

try: (blob_write_bytes:=dll.blob_write_bytes).restype, blob_write_bytes.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), ctypes.c_void_p, size_t]
except AttributeError: pass

intptr_t = ctypes.c_int64
try: (blob_reserve_bytes:=dll.blob_reserve_bytes).restype, blob_reserve_bytes.argtypes = intptr_t, [ctypes.POINTER(struct_blob), size_t]
except AttributeError: pass

try: (blob_reserve_uint32:=dll.blob_reserve_uint32).restype, blob_reserve_uint32.argtypes = intptr_t, [ctypes.POINTER(struct_blob)]
except AttributeError: pass

try: (blob_reserve_intptr:=dll.blob_reserve_intptr).restype, blob_reserve_intptr.argtypes = intptr_t, [ctypes.POINTER(struct_blob)]
except AttributeError: pass

try: (blob_overwrite_bytes:=dll.blob_overwrite_bytes).restype, blob_overwrite_bytes.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (blob_write_uint8:=dll.blob_write_uint8).restype, blob_write_uint8.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), uint8_t]
except AttributeError: pass

try: (blob_overwrite_uint8:=dll.blob_overwrite_uint8).restype, blob_overwrite_uint8.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t, uint8_t]
except AttributeError: pass

try: (blob_write_uint16:=dll.blob_write_uint16).restype, blob_write_uint16.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), uint16_t]
except AttributeError: pass

try: (blob_write_uint32:=dll.blob_write_uint32).restype, blob_write_uint32.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), uint32_t]
except AttributeError: pass

try: (blob_overwrite_uint32:=dll.blob_overwrite_uint32).restype, blob_overwrite_uint32.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t, uint32_t]
except AttributeError: pass

try: (blob_write_uint64:=dll.blob_write_uint64).restype, blob_write_uint64.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), uint64_t]
except AttributeError: pass

try: (blob_write_intptr:=dll.blob_write_intptr).restype, blob_write_intptr.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), intptr_t]
except AttributeError: pass

try: (blob_overwrite_intptr:=dll.blob_overwrite_intptr).restype, blob_overwrite_intptr.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), size_t, intptr_t]
except AttributeError: pass

try: (blob_write_string:=dll.blob_write_string).restype, blob_write_string.argtypes = ctypes.c_bool, [ctypes.POINTER(struct_blob), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (blob_reader_init:=dll.blob_reader_init).restype, blob_reader_init.argtypes = None, [ctypes.POINTER(struct_blob_reader), ctypes.c_void_p, size_t]
except AttributeError: pass

try: (blob_reader_align:=dll.blob_reader_align).restype, blob_reader_align.argtypes = None, [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError: pass

try: (blob_read_bytes:=dll.blob_read_bytes).restype, blob_read_bytes.argtypes = ctypes.c_void_p, [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError: pass

try: (blob_copy_bytes:=dll.blob_copy_bytes).restype, blob_copy_bytes.argtypes = None, [ctypes.POINTER(struct_blob_reader), ctypes.c_void_p, size_t]
except AttributeError: pass

try: (blob_skip_bytes:=dll.blob_skip_bytes).restype, blob_skip_bytes.argtypes = None, [ctypes.POINTER(struct_blob_reader), size_t]
except AttributeError: pass

try: (blob_read_uint8:=dll.blob_read_uint8).restype, blob_read_uint8.argtypes = uint8_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_uint16:=dll.blob_read_uint16).restype, blob_read_uint16.argtypes = uint16_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_uint32:=dll.blob_read_uint32).restype, blob_read_uint32.argtypes = uint32_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_uint64:=dll.blob_read_uint64).restype, blob_read_uint64.argtypes = uint64_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_intptr:=dll.blob_read_intptr).restype, blob_read_intptr.argtypes = intptr_t, [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (blob_read_string:=dll.blob_read_string).restype, blob_read_string.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(struct_blob_reader)]
except AttributeError: pass

try: (ralloc_context:=dll.ralloc_context).restype, ralloc_context.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_size:=dll.ralloc_size).restype, ralloc_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (rzalloc_size:=dll.rzalloc_size).restype, rzalloc_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (reralloc_size:=dll.reralloc_size).restype, reralloc_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (rerzalloc_size:=dll.rerzalloc_size).restype, rerzalloc_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t]
except AttributeError: pass

try: (ralloc_array_size:=dll.ralloc_array_size).restype, ralloc_array_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (rzalloc_array_size:=dll.rzalloc_array_size).restype, rzalloc_array_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (reralloc_array_size:=dll.reralloc_array_size).restype, reralloc_array_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32]
except AttributeError: pass

try: (rerzalloc_array_size:=dll.rerzalloc_array_size).restype, rerzalloc_array_size.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (ralloc_free:=dll.ralloc_free).restype, ralloc_free.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_steal:=dll.ralloc_steal).restype, ralloc_steal.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_adopt:=dll.ralloc_adopt).restype, ralloc_adopt.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_parent:=dll.ralloc_parent).restype, ralloc_parent.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
except AttributeError: pass

try: (ralloc_set_destructor:=dll.ralloc_set_destructor).restype, ralloc_set_destructor.argtypes = None, [ctypes.c_void_p, ctypes.CFUNCTYPE(None, ctypes.c_void_p)]
except AttributeError: pass

try: (ralloc_memdup:=dll.ralloc_memdup).restype, ralloc_memdup.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (ralloc_strdup:=dll.ralloc_strdup).restype, ralloc_strdup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ralloc_strndup:=dll.ralloc_strndup).restype, ralloc_strndup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (ralloc_strcat:=dll.ralloc_strcat).restype, ralloc_strcat.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ralloc_strncat:=dll.ralloc_strncat).restype, ralloc_strncat.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (ralloc_str_append:=dll.ralloc_str_append).restype, ralloc_str_append.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), size_t, size_t]
except AttributeError: pass

try: (ralloc_asprintf:=dll.ralloc_asprintf).restype, ralloc_asprintf.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct___va_list_tag(Struct): pass
struct___va_list_tag._fields_ = [
  ('gp_offset', ctypes.c_uint32),
  ('fp_offset', ctypes.c_uint32),
  ('overflow_arg_area', ctypes.c_void_p),
  ('reg_save_area', ctypes.c_void_p),
]
va_list = (struct___va_list_tag * 1)
try: (ralloc_vasprintf:=dll.ralloc_vasprintf).restype, ralloc_vasprintf.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (ralloc_asprintf_rewrite_tail:=dll.ralloc_asprintf_rewrite_tail).restype, ralloc_asprintf_rewrite_tail.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ralloc_vasprintf_rewrite_tail:=dll.ralloc_vasprintf_rewrite_tail).restype, ralloc_vasprintf_rewrite_tail.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (ralloc_asprintf_append:=dll.ralloc_asprintf_append).restype, ralloc_asprintf_append.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (ralloc_vasprintf_append:=dll.ralloc_vasprintf_append).restype, ralloc_vasprintf_append.argtypes = ctypes.c_bool, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (ralloc_total_size:=dll.ralloc_total_size).restype, ralloc_total_size.argtypes = size_t, [ctypes.c_void_p]
except AttributeError: pass

try: (gc_context:=dll.gc_context).restype, gc_context.argtypes = ctypes.POINTER(gc_ctx), [ctypes.c_void_p]
except AttributeError: pass

try: (gc_alloc_size:=dll.gc_alloc_size).restype, gc_alloc_size.argtypes = ctypes.c_void_p, [ctypes.POINTER(gc_ctx), size_t, size_t]
except AttributeError: pass

try: (gc_zalloc_size:=dll.gc_zalloc_size).restype, gc_zalloc_size.argtypes = ctypes.c_void_p, [ctypes.POINTER(gc_ctx), size_t, size_t]
except AttributeError: pass

try: (gc_free:=dll.gc_free).restype, gc_free.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

try: (gc_get_context:=dll.gc_get_context).restype, gc_get_context.argtypes = ctypes.POINTER(gc_ctx), [ctypes.c_void_p]
except AttributeError: pass

try: (gc_sweep_start:=dll.gc_sweep_start).restype, gc_sweep_start.argtypes = None, [ctypes.POINTER(gc_ctx)]
except AttributeError: pass

try: (gc_mark_live:=dll.gc_mark_live).restype, gc_mark_live.argtypes = None, [ctypes.POINTER(gc_ctx), ctypes.c_void_p]
except AttributeError: pass

try: (gc_sweep_end:=dll.gc_sweep_end).restype, gc_sweep_end.argtypes = None, [ctypes.POINTER(gc_ctx)]
except AttributeError: pass

class struct_linear_ctx(Struct): pass
linear_ctx = struct_linear_ctx
try: (linear_alloc_child:=dll.linear_alloc_child).restype, linear_alloc_child.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx), ctypes.c_uint32]
except AttributeError: pass

class linear_opts(Struct): pass
linear_opts._fields_ = [
  ('min_buffer_size', ctypes.c_uint32),
]
try: (linear_context:=dll.linear_context).restype, linear_context.argtypes = ctypes.POINTER(linear_ctx), [ctypes.c_void_p]
except AttributeError: pass

try: (linear_context_with_opts:=dll.linear_context_with_opts).restype, linear_context_with_opts.argtypes = ctypes.POINTER(linear_ctx), [ctypes.c_void_p, ctypes.POINTER(linear_opts)]
except AttributeError: pass

try: (linear_zalloc_child:=dll.linear_zalloc_child).restype, linear_zalloc_child.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx), ctypes.c_uint32]
except AttributeError: pass

try: (linear_free_context:=dll.linear_free_context).restype, linear_free_context.argtypes = None, [ctypes.POINTER(linear_ctx)]
except AttributeError: pass

try: (ralloc_steal_linear_context:=dll.ralloc_steal_linear_context).restype, ralloc_steal_linear_context.argtypes = None, [ctypes.c_void_p, ctypes.POINTER(linear_ctx)]
except AttributeError: pass

try: (ralloc_parent_of_linear_context:=dll.ralloc_parent_of_linear_context).restype, ralloc_parent_of_linear_context.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx)]
except AttributeError: pass

try: (linear_alloc_child_array:=dll.linear_alloc_child_array).restype, linear_alloc_child_array.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (linear_zalloc_child_array:=dll.linear_zalloc_child_array).restype, linear_zalloc_child_array.argtypes = ctypes.c_void_p, [ctypes.POINTER(linear_ctx), size_t, ctypes.c_uint32]
except AttributeError: pass

try: (linear_strdup:=dll.linear_strdup).restype, linear_strdup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (linear_asprintf:=dll.linear_asprintf).restype, linear_asprintf.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (linear_vasprintf:=dll.linear_vasprintf).restype, linear_vasprintf.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (linear_asprintf_append:=dll.linear_asprintf_append).restype, linear_asprintf_append.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (linear_vasprintf_append:=dll.linear_vasprintf_append).restype, linear_vasprintf_append.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (linear_asprintf_rewrite_tail:=dll.linear_asprintf_rewrite_tail).restype, linear_asprintf_rewrite_tail.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (linear_vasprintf_rewrite_tail:=dll.linear_vasprintf_rewrite_tail).restype, linear_vasprintf_rewrite_tail.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char), va_list]
except AttributeError: pass

try: (linear_strcat:=dll.linear_strcat).restype, linear_strcat.argtypes = ctypes.c_bool, [ctypes.POINTER(linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

_anonenum7 = CEnum(ctypes.c_uint32)
RALLOC_PRINT_INFO_SUMMARY_ONLY = _anonenum7.define('RALLOC_PRINT_INFO_SUMMARY_ONLY', 1)

try: (ralloc_print_info:=dll.ralloc_print_info).restype, ralloc_print_info.argtypes = None, [ctypes.POINTER(FILE), ctypes.c_void_p, ctypes.c_uint32]
except AttributeError: pass

NIR_DEBUG_CLONE = (1 << 0)
NIR_DEBUG_SERIALIZE = (1 << 1)
NIR_DEBUG_NOVALIDATE = (1 << 2)
NIR_DEBUG_EXTENDED_VALIDATION = (1 << 3)
NIR_DEBUG_TGSI = (1 << 4)
NIR_DEBUG_PRINT_VS = (1 << 5)
NIR_DEBUG_PRINT_TCS = (1 << 6)
NIR_DEBUG_PRINT_TES = (1 << 7)
NIR_DEBUG_PRINT_GS = (1 << 8)
NIR_DEBUG_PRINT_FS = (1 << 9)
NIR_DEBUG_PRINT_CS = (1 << 10)
NIR_DEBUG_PRINT_TS = (1 << 11)
NIR_DEBUG_PRINT_MS = (1 << 12)
NIR_DEBUG_PRINT_RGS = (1 << 13)
NIR_DEBUG_PRINT_AHS = (1 << 14)
NIR_DEBUG_PRINT_CHS = (1 << 15)
NIR_DEBUG_PRINT_MHS = (1 << 16)
NIR_DEBUG_PRINT_IS = (1 << 17)
NIR_DEBUG_PRINT_CBS = (1 << 18)
NIR_DEBUG_PRINT_KS = (1 << 19)
NIR_DEBUG_PRINT_NO_INLINE_CONSTS = (1 << 20)
NIR_DEBUG_PRINT_INTERNAL = (1 << 21)
NIR_DEBUG_PRINT_PASS_FLAGS = (1 << 22)
NIR_DEBUG_INVALIDATE_METADATA = (1 << 23)
NIR_DEBUG_PRINT_STRUCT_DECLS = (1 << 24)
NIR_DEBUG_PRINT = (NIR_DEBUG_PRINT_VS | NIR_DEBUG_PRINT_TCS | NIR_DEBUG_PRINT_TES | NIR_DEBUG_PRINT_GS | NIR_DEBUG_PRINT_FS | NIR_DEBUG_PRINT_CS | NIR_DEBUG_PRINT_TS | NIR_DEBUG_PRINT_MS | NIR_DEBUG_PRINT_RGS | NIR_DEBUG_PRINT_AHS | NIR_DEBUG_PRINT_CHS | NIR_DEBUG_PRINT_MHS | NIR_DEBUG_PRINT_IS | NIR_DEBUG_PRINT_CBS | NIR_DEBUG_PRINT_KS)
NIR_FALSE = 0
NIR_TRUE = (~0)
NIR_MAX_VEC_COMPONENTS = 16
NIR_MAX_MATRIX_COLUMNS = 4
NIR_STREAM_PACKED = (1 << 8)
NIR_VARIABLE_NO_INDEX = ~0
nir_foreach_variable_in_list = lambda var,var_list: foreach_list_typed(nir_variable, var, node, var_list)
nir_foreach_variable_in_list_safe = lambda var,var_list: foreach_list_typed_safe(nir_variable, var, node, var_list)
nir_foreach_shader_in_variable = lambda var,shader: nir_foreach_variable_with_modes(var, shader, nir_var_shader_in)
nir_foreach_shader_in_variable_safe = lambda var,shader: nir_foreach_variable_with_modes_safe(var, shader, nir_var_shader_in)
nir_foreach_shader_out_variable = lambda var,shader: nir_foreach_variable_with_modes(var, shader, nir_var_shader_out)
nir_foreach_shader_out_variable_safe = lambda var,shader: nir_foreach_variable_with_modes_safe(var, shader, nir_var_shader_out)
nir_foreach_uniform_variable = lambda var,shader: nir_foreach_variable_with_modes(var, shader, nir_var_uniform)
nir_foreach_uniform_variable_safe = lambda var,shader: nir_foreach_variable_with_modes_safe(var, shader, nir_var_uniform)
nir_foreach_image_variable = lambda var,shader: nir_foreach_variable_with_modes(var, shader, nir_var_image)
nir_foreach_image_variable_safe = lambda var,shader: nir_foreach_variable_with_modes_safe(var, shader, nir_var_image)
NIR_SRC_PARENT_IS_IF = (0x1)
NIR_ALU_MAX_INPUTS = NIR_MAX_VEC_COMPONENTS
NIR_INTRINSIC_MAX_CONST_INDEX = 8
NIR_ALIGN_MUL_MAX = 0x40000000
NIR_INTRINSIC_MAX_INPUTS = 11
nir_log_shadere = lambda s: nir_log_shader_annotated_tagged(MESA_LOG_ERROR, (MESA_LOG_TAG), (s), NULL)
nir_log_shaderw = lambda s: nir_log_shader_annotated_tagged(MESA_LOG_WARN, (MESA_LOG_TAG), (s), NULL)
nir_log_shaderi = lambda s: nir_log_shader_annotated_tagged(MESA_LOG_INFO, (MESA_LOG_TAG), (s), NULL)
nir_log_shader_annotated = lambda s,annotations: nir_log_shader_annotated_tagged(MESA_LOG_ERROR, (MESA_LOG_TAG), (s), annotations)
NIR_STRINGIZE = lambda x: NIR_STRINGIZE_INNER(x)
NVIDIA_VENDOR_ID = 0x10de
NAK_SUBGROUP_SIZE = 32
NAK_QMD_ALIGN_B = 256
NAK_MAX_QMD_SIZE_B = 384
NAK_MAX_QMD_DWORDS = (NAK_MAX_QMD_SIZE_B / 4)
LP_MAX_VECTOR_WIDTH = 512
LP_MIN_VECTOR_ALIGN = 64
LP_MAX_VECTOR_LENGTH = (LP_MAX_VECTOR_WIDTH/8)
LP_RESV_FUNC_ARGS = 2
LP_JIT_TEXTURE_SAMPLE_STRIDE = 15
lp_jit_resources_constants = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_CONSTANTS, "constants")
lp_jit_resources_ssbos = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_SSBOS, "ssbos")
lp_jit_resources_textures = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_TEXTURES, "textures")
lp_jit_resources_samplers = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_SAMPLERS, "samplers")
lp_jit_resources_images = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_RES_IMAGES, "images")
lp_jit_vertex_header_id = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_VERTEX_HEADER_VERTEX_ID, "id")
lp_jit_vertex_header_clip_pos = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_VERTEX_HEADER_CLIP_POS, "clip_pos")
lp_jit_vertex_header_data = lambda _gallivm,_type,_ptr: lp_build_struct__get_ptr2(_gallivm, _type, _ptr, LP_JIT_VERTEX_HEADER_DATA, "data")
LP_MAX_TEX_FUNC_ARGS = 32
gc_alloc = lambda ctx,type,count: gc_alloc_size(ctx, sizeof(type) * (count), alignof(type))
gc_zalloc = lambda ctx,type,count: gc_zalloc_size(ctx, sizeof(type) * (count), alignof(type))
gc_alloc_zla = lambda ctx,type,type2,count: gc_alloc_size(ctx, sizeof(type) + sizeof(type2) * (count), MAX2(alignof(type), alignof(type2)))
gc_zalloc_zla = lambda ctx,type,type2,count: gc_zalloc_size(ctx, sizeof(type) + sizeof(type2) * (count), MAX2(alignof(type), alignof(type2)))
DECLARE_RALLOC_CXX_OPERATORS = lambda type: DECLARE_RALLOC_CXX_OPERATORS_TEMPLATE(type, ralloc_size)
DECLARE_RZALLOC_CXX_OPERATORS = lambda type: DECLARE_RALLOC_CXX_OPERATORS_TEMPLATE(type, rzalloc_size)
DECLARE_LINEAR_ALLOC_CXX_OPERATORS = lambda type: DECLARE_LINEAR_ALLOC_CXX_OPERATORS_TEMPLATE(type, linear_alloc_child)
DECLARE_LINEAR_ZALLOC_CXX_OPERATORS = lambda type: DECLARE_LINEAR_ALLOC_CXX_OPERATORS_TEMPLATE(type, linear_zalloc_child)
lvp_nir_options = gzip.decompress(base64.b64decode("H4sIAAAAAAAAA2NgZGRkYGAAkYxgCsQFsxigwgwQBoxmhCqFq2WEKwIrAEGIkQxoAEMALwCqVsCiGUwLMHA0QPn29nBJkswHANb8YpH4AAAA"))