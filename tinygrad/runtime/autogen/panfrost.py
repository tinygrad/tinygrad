# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes, os, sysconfig
from typing import Literal
from tinygrad.runtime.support import c
from tinygrad.runtime.support.c import record
from tinygrad.runtime.autogen.mesa import struct_nir_shader_compiler_options, struct_nir_shader

dll = c.DLL('panfrost', os.getenv('MESA_PATH', 'tinymesa'), os.path.join(sysconfig.get_paths()['platlib'], 'tinymesa'), emsg='pip install tinymesa==25.2.7.2')

@record
class struct_pipe_reference(c.Struct): SIZE = 4
struct_pipe_reference.register_fields([('count', ctypes.c_int32, 0)])
@record
class struct_pipe_screen(c.Struct): SIZE = 2000
@record
class struct_pipe_context(c.Struct): SIZE = 1264
@record
class struct_pipe_resource(c.Struct): SIZE = 112
struct_pipe_resource.register_fields([
  ('reference', struct_pipe_reference, 0), ('width0', ctypes.c_uint32, 64), ('height0', ctypes.c_uint16, 68),
  ('depth0', ctypes.c_uint16, 70), ('array_size', ctypes.c_uint16, 72), ('format', ctypes.c_uint16, 74, 16, 0),
  ('target', ctypes.c_uint8, 76, 8, 0), ('last_level', ctypes.c_uint8, 77), ('nr_samples', ctypes.c_uint8, 78),
  ('nr_storage_samples', ctypes.c_uint8, 79), ('nr_sparse_levels', ctypes.c_uint8, 80), ('usage', ctypes.c_uint32, 81, 4, 4),
  ('bind', ctypes.c_uint32, 84), ('flags', ctypes.c_uint32, 88), ('next', c.POINTER[struct_pipe_resource], 96),
  ('screen', c.POINTER[struct_pipe_screen], 104)])

@record
class union_driOptionValue(c.Struct): SIZE = 8
union_driOptionValue.register_fields([('_bool', ctypes.c_ubyte, 0), ('_int', ctypes.c_int32, 0),
                                      ('_uint64', ctypes.c_uint64, 0), ('_float', ctypes.c_float, 0),
                                      ('_string', c.POINTER[ctypes.c_ubyte], 0)])
@record
class struct_driOptionRange(c.Struct): SIZE = 16
struct_driOptionRange.register_fields([('start', union_driOptionValue, 0), ('end', union_driOptionValue, 8)])
@record
class struct_driOptionInfo(c.Struct): SIZE = 32
struct_driOptionInfo.register_fields([('name', c.POINTER[ctypes.c_char], 0), ('type', ctypes.c_uint32, 8),
                                     ('range', struct_driOptionRange, 16)])
@record
class struct_driOptionCache(c.Struct): SIZE = 24
struct_driOptionCache.register_fields([('info', c.POINTER[struct_driOptionInfo], 0),
                                      ('values', c.POINTER[union_driOptionValue], 8), ('tableSize', ctypes.c_uint32, 16)])
@record
class struct_driEnumDescription(c.Struct): SIZE = 16
struct_driEnumDescription.register_fields([('value', ctypes.c_int32, 0), ('desc', c.POINTER[ctypes.c_ubyte], 8)])
@record
class struct_driOptionDescription(c.Struct): SIZE = 128
struct_driOptionDescription.register_fields([('desc', c.POINTER[ctypes.c_char], 0), ('info', struct_driOptionInfo, 8),
                                            ('value', union_driOptionValue, 40),
                                            ('enums', c.Array[struct_driEnumDescription, Literal[5]], 48)])
@record
class struct_pipe_screen_config(c.Struct): SIZE = 24
struct_pipe_screen_config.register_fields([('driver_name_is_inferred', ctypes.c_bool, 0),
  ('options', c.POINTER[struct_driOptionCache], 8), ('options_info', c.POINTER[struct_driOptionCache], 16)])

@record
class struct_pipe_constant_buffer(c.Struct): SIZE = 24
struct_pipe_constant_buffer.register_fields([
  ('buffer', c.POINTER[struct_pipe_resource], 0), ('buffer_offset', ctypes.c_uint32, 8),
  ('buffer_size', ctypes.c_uint32, 12), ('user_buffer', ctypes.c_void_p, 16)])
@record
class struct_pipe_grid_info(c.Struct): SIZE = 104
struct_pipe_grid_info.register_fields([
  ('variable_shared_mem', ctypes.c_uint32, 0), ('work_dim', ctypes.c_uint32, 4),
  ('block', c.Array[ctypes.c_uint32, Literal[3]], 8), ('last_block', c.Array[ctypes.c_uint32, Literal[3]], 20),
  ('grid', c.Array[ctypes.c_uint32, Literal[3]], 32), ('grid_base', c.Array[ctypes.c_uint32, Literal[3]], 44),
  ('indirect', c.POINTER[struct_pipe_resource], 56), ('indirect_offset', ctypes.c_uint32, 64),
  ('indirect_stride', ctypes.c_uint32, 68), ('draw_count', ctypes.c_uint32, 72),
  ('indirect_draw_count_offset', ctypes.c_uint32, 76), ('indirect_draw_count', c.POINTER[struct_pipe_resource], 80),
  ('num_globals', ctypes.c_uint32, 88), ('globals', c.POINTER[c.POINTER[struct_pipe_resource]], 96)])
@record
class struct_pipe_box(c.Struct): SIZE = 20
struct_pipe_box.register_fields([('x', ctypes.c_int32, 0), ('width', ctypes.c_int32, 4), ('y', ctypes.c_int32, 8),
                                 ('height', ctypes.c_int32, 12), ('z', ctypes.c_int16, 16), ('depth', ctypes.c_int16, 18)])
@record
class struct_pipe_tex2d_from_buf(c.Struct): SIZE = 12
struct_pipe_tex2d_from_buf.register_fields([('offset', ctypes.c_uint32, 0), ('row_stride', ctypes.c_uint16, 4),
                                            ('width', ctypes.c_uint16, 6), ('height', ctypes.c_uint16, 8)])
@record
class struct_pipe_sampler_state(c.Struct): SIZE = 36
struct_pipe_sampler_state.register_fields([
  ('wrap_s', ctypes.c_uint32, 0, 3, 0), ('wrap_t', ctypes.c_uint32, 0, 3, 3), ('wrap_r', ctypes.c_uint32, 0, 3, 6),
  ('min_img_filter', ctypes.c_uint32, 1, 1, 1), ('min_mip_filter', ctypes.c_uint32, 1, 2, 2),
  ('mag_img_filter', ctypes.c_uint32, 1, 1, 4), ('compare_mode', ctypes.c_uint32, 1, 1, 5),
  ('compare_func', ctypes.c_uint32, 1, 3, 6), ('unnormalized_coords', ctypes.c_uint32, 2, 1, 1)])
@record
class struct_pipe_sampler_view(c.Struct): SIZE = 40
struct_pipe_sampler_view.register_fields([
  ('format', ctypes.c_uint32, 0, 12, 0), ('astc_decode_format', ctypes.c_uint32, 0, 2, 12),
  ('is_tex2d_from_buf', ctypes.c_uint32, 0, 1, 14), ('target', ctypes.c_uint32, 0, 5, 15),
  ('swizzle_r', ctypes.c_uint32, 0, 3, 20), ('swizzle_g', ctypes.c_uint32, 0, 3, 23),
  ('swizzle_b', ctypes.c_uint32, 0, 3, 26), ('swizzle_a', ctypes.c_uint32, 0, 3, 29),
  ('texture', c.POINTER[struct_pipe_resource], 8), ('context', c.POINTER[struct_pipe_context], 16),
  ('u', struct_pipe_tex2d_from_buf, 24)])
@record
class struct_pipe_image_view(c.Struct): SIZE = 32
struct_pipe_image_view.register_fields([
  ('resource', c.POINTER[struct_pipe_resource], 0), ('format', ctypes.c_uint32, 8, 16, 0),
  ('access', ctypes.c_uint16, 12), ('shader_access', ctypes.c_uint16, 14), ('u', struct_pipe_tex2d_from_buf, 16)])
@record
class struct_pipe_compute_state(c.Struct): SIZE = 24
struct_pipe_compute_state.register_fields([('ir_type', ctypes.c_uint32, 0), ('prog', ctypes.c_void_p, 8),
                                           ('static_shared_mem', ctypes.c_uint32, 16)])

@record
class struct_pan_ubo_word(c.Struct): SIZE = 4
struct_pan_ubo_word.register_fields([('ubo', ctypes.c_uint16, 0), ('offset', ctypes.c_uint16, 2)])
@record
class struct_pan_shader_info(c.Struct): SIZE = 1496
struct_pan_shader_info.register_fields([
  ('stage', ctypes.c_uint32, 0), ('work_reg_count', ctypes.c_uint32, 4), ('tls_size', ctypes.c_uint32, 8),
  ('wls_size', ctypes.c_uint32, 12), ('preload', ctypes.c_uint64, 128), ('allow_merging_workgroups', ctypes.c_ubyte, 136),
  ('contains_barrier', ctypes.c_ubyte, 168), ('writes_global', ctypes.c_ubyte, 170),
  ('sampler_count', ctypes.c_uint32, 188), ('texture_count', ctypes.c_uint32, 192), ('ubo_count', ctypes.c_uint32, 196),
  ('attributes_read_count', ctypes.c_uint32, 200), ('attribute_count', ctypes.c_uint32, 204),
  ('attributes_read', ctypes.c_uint32, 208), ('push_count', ctypes.c_uint32, 820),
  ('push_words', c.Array[struct_pan_ubo_word, Literal[128]], 824), ('ubo_mask', ctypes.c_uint32, 1336)])
@record
class struct_panfrost_sysvals(c.Struct): SIZE = 132
struct_panfrost_sysvals.register_fields([('sysvals', c.Array[ctypes.c_uint32, Literal[32]], 0), ('sysval_count', ctypes.c_uint32, 128)])
@record
class struct_panfrost_native_shader(c.Struct): SIZE = 1656
struct_panfrost_native_shader.register_fields([('info', struct_pan_shader_info, 0), ('sysvals', struct_panfrost_sysvals, 1496),
  ('binary', ctypes.c_void_p, 1632), ('binary_size', ctypes.c_uint64, 1640)])
@record
class struct_spirv_shader(c.Struct): SIZE = 24
struct_spirv_shader.register_fields([('words', ctypes.c_void_p, 0), ('num_words', ctypes.c_uint64, 8),
                                     ('tcs_vertices_out_word', ctypes.c_uint32, 16)])

struct_pipe_screen.register_fields([
  ('destroy', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_screen]]], 1400),
  ('context_create', c.CFUNCTYPE[c.POINTER[struct_pipe_context], [c.POINTER[struct_pipe_screen], ctypes.c_void_p, ctypes.c_uint32]], 1472),
  ('resource_create', c.CFUNCTYPE[c.POINTER[struct_pipe_resource], [c.POINTER[struct_pipe_screen], c.POINTER[struct_pipe_resource]]], 1512),
  ('resource_destroy', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_screen], c.POINTER[struct_pipe_resource]]], 1600),
  ('fence_reference', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_screen], c.POINTER[ctypes.c_void_p], ctypes.c_void_p]], 1616),
  ('fence_finish', c.CFUNCTYPE[ctypes.c_bool, [c.POINTER[struct_pipe_screen], ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]], 1624)])
struct_pipe_context.register_fields([
  ('screen', c.POINTER[struct_pipe_screen], 0),
  ('destroy', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context]]], 72),
  ('create_sampler_state', c.CFUNCTYPE[ctypes.c_void_p, [c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_sampler_state]]], 288),
  ('bind_sampler_states', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]], 296),
  ('delete_sampler_state', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_void_p]], 304),
  ('set_constant_buffer', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, c.POINTER[struct_pipe_constant_buffer]]], 592),
  ('set_sampler_views', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[c.POINTER[struct_pipe_sampler_view]]]], 656),
  ('set_shader_images', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[struct_pipe_image_view]]], 704),
  ('flush', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], c.POINTER[ctypes.c_void_p], ctypes.c_uint32]], 832),
  ('create_sampler_view', c.CFUNCTYPE[c.POINTER[struct_pipe_sampler_view], [c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_resource], c.POINTER[struct_pipe_sampler_view]]], 864),
  ('sampler_view_release', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_sampler_view]]], 880),
  ('buffer_map', c.CFUNCTYPE[ctypes.c_void_p, [c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_resource], ctypes.c_uint32, ctypes.c_uint32, c.POINTER[struct_pipe_box], c.POINTER[ctypes.c_void_p]]], 904),
  ('buffer_unmap', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_void_p]], 920),
  ('create_compute_state', c.CFUNCTYPE[ctypes.c_void_p, [c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_compute_state]]], 1000),
  ('bind_compute_state', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_void_p]], 1008),
  ('delete_compute_state', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_void_p]], 1016),
  ('set_global_binding', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], ctypes.c_uint32, ctypes.c_uint32, c.POINTER[c.POINTER[struct_pipe_resource]], c.POINTER[c.POINTER[ctypes.c_uint32]]]], 1040),
  ('launch_grid', c.CFUNCTYPE[None, [c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_grid_info]]], 1048)])

@dll.bind(c.POINTER[struct_pipe_screen], ctypes.c_int32, c.POINTER[struct_pipe_screen_config])
def panfrost_drm_screen_create(drmFD:int, config:c.POINTER[struct_pipe_screen_config]) -> c.POINTER[struct_pipe_screen]: ...
@dll.bind(c.POINTER[struct_nir_shader_compiler_options], ctypes.c_uint32)
def pan_shader_get_compiler_options(arch:int) -> c.POINTER[struct_nir_shader_compiler_options]: ...
@dll.bind(ctypes.c_uint64, c.POINTER[struct_pipe_resource])
def panfrost_resource_address(resource:c.POINTER[struct_pipe_resource]) -> int: ...
@dll.bind(ctypes.c_uint64, ctypes.c_void_p)
def panfrost_compute_shader_desc(cso:ctypes.c_void_p) -> int: ...
@dll.bind(None, ctypes.c_void_p, c.POINTER[struct_pan_shader_info])
def panfrost_compute_shader_info(cso:ctypes.c_void_p, info:c.POINTER[struct_pan_shader_info]) -> None: ...
@dll.bind(None, c.POINTER[struct_pipe_context], ctypes.c_void_p, c.POINTER[struct_pipe_grid_info], c.POINTER[ctypes.c_uint32])
def panfrost_compute_tls_info(ctx:c.POINTER[struct_pipe_context], cso:ctypes.c_void_p, grid:c.POINTER[struct_pipe_grid_info], out:c.POINTER[ctypes.c_uint32]) -> None: ...
@dll.bind(ctypes.c_uint32, c.POINTER[struct_pipe_context])
def panfrost_context_group_handle(ctx:c.POINTER[struct_pipe_context]) -> int: ...
@dll.bind(ctypes.c_uint32, c.POINTER[struct_pipe_screen], ctypes.c_uint32)
def panfrost_max_threads(screen:c.POINTER[struct_pipe_screen], work_reg_count:int) -> int: ...
@dll.bind(ctypes.c_bool, c.POINTER[struct_nir_shader])
def panfrost_nir_has_float_compare(shader:c.POINTER[struct_nir_shader]) -> bool: ...
@dll.bind(None, c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_sampler_view], ctypes.c_void_p)
def panfrost_make_texture_desc(ctx:c.POINTER[struct_pipe_context], tmpl:c.POINTER[struct_pipe_sampler_view], out:ctypes.c_void_p) -> None: ...
@dll.bind(None, ctypes.c_void_p, ctypes.c_void_p)
def panfrost_sampler_desc(state:ctypes.c_void_p, out:ctypes.c_void_p) -> None: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], ctypes.c_void_p)
def panfrost_compute_bind(ctx:c.POINTER[struct_pipe_context], state:ctypes.c_void_p) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_constant_buffer], ctypes.c_void_p)
def panfrost_compute_set_constants(ctx:c.POINTER[struct_pipe_context], cb:c.POINTER[struct_pipe_constant_buffer], args:ctypes.c_void_p) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_grid_info])
def panfrost_compute_launch(ctx:c.POINTER[struct_pipe_context], grid:c.POINTER[struct_pipe_grid_info]) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], ctypes.c_uint64, c.POINTER[struct_pipe_sampler_view])
def panfrost_compute_create_sampler_view(ctx:c.POINTER[struct_pipe_context], address:int,
  tmpl:c.POINTER[struct_pipe_sampler_view]) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], ctypes.c_void_p, c.POINTER[c.POINTER[struct_pipe_sampler_view]], ctypes.c_uint32)
def panfrost_compute_set_samplers(ctx:c.POINTER[struct_pipe_context], sampler:ctypes.c_void_p,
  views:c.POINTER[c.POINTER[struct_pipe_sampler_view]], count:int) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], c.POINTER[struct_pipe_sampler_view])
def panfrost_compute_release_sampler_view(ctx:c.POINTER[struct_pipe_context], view:c.POINTER[struct_pipe_sampler_view]) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], ctypes.c_uint32, c.POINTER[struct_pipe_image_view], ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)
def panfrost_compute_set_images(ctx:c.POINTER[struct_pipe_context], count:int, views:c.POINTER[struct_pipe_image_view],
  address0:int, address1:int, address2:int, address3:int) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context])
def panfrost_compute_flush(ctx:c.POINTER[struct_pipe_context]) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32)
def panfrost_buffer_write(ctx:c.POINTER[struct_pipe_context], address:int, data:ctypes.c_void_p, size:int) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32)
def panfrost_buffer_read(ctx:c.POINTER[struct_pipe_context], address:int, data:ctypes.c_void_p, size:int) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32)
def panfrost_buffer_copy(ctx:c.POINTER[struct_pipe_context], dst_address:int, src_address:int, size:int) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], c.POINTER[ctypes.c_uint64], ctypes.c_uint64)
def panfrost_compute_wait(ctx:c.POINTER[struct_pipe_context], address:c.POINTER[ctypes.c_uint64], value:int) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], c.POINTER[ctypes.c_uint64], ctypes.c_uint64)
def panfrost_compute_signal(ctx:c.POINTER[struct_pipe_context], address:c.POINTER[ctypes.c_uint64], value:int) -> int: ...
@dll.bind(ctypes.c_void_p, c.POINTER[struct_pipe_context], c.POINTER[ctypes.c_uint64])
def panfrost_compute_timestamp(ctx:c.POINTER[struct_pipe_context], address:c.POINTER[ctypes.c_uint64]) -> int: ...
@dll.bind(c.POINTER[struct_spirv_shader], ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
def nir_to_spirv(shader:ctypes.c_void_p, info:ctypes.c_void_p, screen:ctypes.c_void_p) -> c.POINTER[struct_spirv_shader]: ...
@dll.bind(None, c.POINTER[struct_spirv_shader])
def spirv_shader_delete(shader:c.POINTER[struct_spirv_shader]) -> None: ...
@dll.bind(None, c.POINTER[struct_driOptionCache], c.POINTER[struct_driOptionDescription], ctypes.c_uint32)
def driParseOptionInfo(info:c.POINTER[struct_driOptionCache], configOptions:c.POINTER[struct_driOptionDescription], numOptions:int) -> None: ...
@dll.bind(None, c.POINTER[struct_driOptionCache])
def driDestroyOptionCache(cache:c.POINTER[struct_driOptionCache]) -> None: ...
@dll.bind(None, c.POINTER[struct_driOptionCache])
def driDestroyOptionInfo(info:c.POINTER[struct_driOptionCache]) -> None: ...
@dll.bind(None)
def glsl_type_singleton_init_or_ref() -> None: ...
@dll.bind(None)
def glsl_type_singleton_decref() -> None: ...

PIPE_BUFFER = 0
PIPE_TEXTURE_2D = 1
PIPE_USAGE_DEFAULT = 0
PIPE_BIND_GLOBAL = 1 << 13
PIPE_BIND_SHADER_BUFFER = 1 << 14
PIPE_BIND_SAMPLER_VIEW = 1 << 3
PIPE_BIND_SHADER_IMAGE = 1 << 15
PIPE_BIND_LINEAR = 1 << 21
PIPE_MAP_READ = 1
PIPE_MAP_WRITE = 2
PIPE_SHADER_IR_NIR = 1
PIPE_SHADER_IR_NATIVE = 2
PIPE_SHADER_COMPUTE = 5
PIPE_IMAGE_ACCESS_READ_WRITE = 3
PIPE_IMAGE_ACCESS_TEX2D_FROM_BUFFER = 1 << 4
PIPE_FORMAT_R8_UINT = 84
PIPE_FORMAT_R16G16B16A16_FLOAT = 83
PIPE_FORMAT_R32G32B32A32_FLOAT = 16
PIPE_TEX_WRAP_CLAMP_TO_BORDER = 4
PIPE_TEX_FILTER_NEAREST = 0
PIPE_CONTEXT_COMPUTE_ONLY = 1
PIPE_CONTEXT_NO_LOD_BIAS = 1 << 8
DRI_BOOL = 0
DRI_INT = 2
DRI_UINT64 = 3
DRI_SECTION = 6
