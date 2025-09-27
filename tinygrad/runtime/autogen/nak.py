# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-DHAVE_ENDIAN_H', '-DHAVE_STRUCT_TIMESPEC', '-DHAVE_PTHREAD', '-I/tmp/mesa-mesa-25.1.0/src', '-I/tmp/mesa-mesa-25.1.0/include', '-I/tmp/mesa-mesa-25.1.0/src/compiler/nir', '-I/tmp/mesa-mesa-25.1.0/gen']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes
from tinygrad.runtime.support.mesa import nak as dll


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except (AttributeError, RuntimeError):
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['FIXME_STUB'] = dll #  ctypes.CDLL('FIXME_STUB')



# values for enumeration 'nv_device_type'
nv_device_type__enumvalues = {
    0: 'NV_DEVICE_TYPE_IGP',
    1: 'NV_DEVICE_TYPE_DIS',
    2: 'NV_DEVICE_TYPE_SOC',
}
NV_DEVICE_TYPE_IGP = 0
NV_DEVICE_TYPE_DIS = 1
NV_DEVICE_TYPE_SOC = 2
nv_device_type = ctypes.c_uint32 # enum
class struct_nv_device_info(Structure):
    pass

class struct_nv_device_info_pci(Structure):
    pass

struct_nv_device_info_pci._pack_ = 1 # source:False
struct_nv_device_info_pci._fields_ = [
    ('domain', ctypes.c_uint16),
    ('bus', ctypes.c_ubyte),
    ('dev', ctypes.c_ubyte),
    ('func', ctypes.c_ubyte),
    ('revision_id', ctypes.c_ubyte),
]

struct_nv_device_info._pack_ = 1 # source:False
struct_nv_device_info._fields_ = [
    ('type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('device_id', ctypes.c_uint16),
    ('chipset', ctypes.c_uint16),
    ('device_name', ctypes.c_char * 64),
    ('chipset_name', ctypes.c_char * 16),
    ('pci', struct_nv_device_info_pci),
    ('sm', ctypes.c_ubyte),
    ('gpc_count', ctypes.c_ubyte),
    ('tpc_count', ctypes.c_uint16),
    ('mp_per_tpc', ctypes.c_ubyte),
    ('max_warps_per_mp', ctypes.c_ubyte),
    ('cls_copy', ctypes.c_uint16),
    ('cls_eng2d', ctypes.c_uint16),
    ('cls_eng3d', ctypes.c_uint16),
    ('cls_m2mf', ctypes.c_uint16),
    ('cls_compute', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('vram_size_B', ctypes.c_uint64),
    ('bar_size_B', ctypes.c_uint64),
]

size_t = ctypes.c_uint64
try:
    nv_device_uuid = _libraries['FIXME_STUB'].nv_device_uuid
    nv_device_uuid.restype = None
    nv_device_uuid.argtypes = [ctypes.POINTER(struct_nv_device_info), ctypes.POINTER(ctypes.c_ubyte), size_t, ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
class struct_nak_compiler(Structure):
    pass

try:
    nak_compiler_create = _libraries['FIXME_STUB'].nak_compiler_create
    nak_compiler_create.restype = ctypes.POINTER(struct_nak_compiler)
    nak_compiler_create.argtypes = [ctypes.POINTER(struct_nv_device_info)]
except (AttributeError, RuntimeError):
    pass
try:
    nak_compiler_destroy = _libraries['FIXME_STUB'].nak_compiler_destroy
    nak_compiler_destroy.restype = None
    nak_compiler_destroy.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except (AttributeError, RuntimeError):
    pass
uint64_t = ctypes.c_uint64
try:
    nak_debug_flags = _libraries['FIXME_STUB'].nak_debug_flags
    nak_debug_flags.restype = uint64_t
    nak_debug_flags.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except (AttributeError, RuntimeError):
    pass
class struct_nir_shader_compiler_options(Structure):
    pass

class struct_nir_instr(Structure):
    pass


# values for enumeration 'c__EA_nir_variable_mode'
c__EA_nir_variable_mode__enumvalues = {
    1: 'nir_var_system_value',
    2: 'nir_var_uniform',
    4: 'nir_var_shader_in',
    8: 'nir_var_shader_out',
    16: 'nir_var_image',
    32: 'nir_var_shader_call_data',
    64: 'nir_var_ray_hit_attrib',
    128: 'nir_var_mem_ubo',
    256: 'nir_var_mem_push_const',
    512: 'nir_var_mem_ssbo',
    1024: 'nir_var_mem_constant',
    2048: 'nir_var_mem_task_payload',
    4096: 'nir_var_mem_node_payload',
    8192: 'nir_var_mem_node_payload_in',
    16384: 'nir_var_function_in',
    32768: 'nir_var_function_out',
    65536: 'nir_var_function_inout',
    131072: 'nir_var_shader_temp',
    262144: 'nir_var_function_temp',
    524288: 'nir_var_mem_shared',
    1048576: 'nir_var_mem_global',
    1966080: 'nir_var_mem_generic',
    1159: 'nir_var_read_only_modes',
    1969033: 'nir_var_vec_indexable_modes',
    21: 'nir_num_variable_modes',
    2097151: 'nir_var_all',
}
nir_var_system_value = 1
nir_var_uniform = 2
nir_var_shader_in = 4
nir_var_shader_out = 8
nir_var_image = 16
nir_var_shader_call_data = 32
nir_var_ray_hit_attrib = 64
nir_var_mem_ubo = 128
nir_var_mem_push_const = 256
nir_var_mem_ssbo = 512
nir_var_mem_constant = 1024
nir_var_mem_task_payload = 2048
nir_var_mem_node_payload = 4096
nir_var_mem_node_payload_in = 8192
nir_var_function_in = 16384
nir_var_function_out = 32768
nir_var_function_inout = 65536
nir_var_shader_temp = 131072
nir_var_function_temp = 262144
nir_var_mem_shared = 524288
nir_var_mem_global = 1048576
nir_var_mem_generic = 1966080
nir_var_read_only_modes = 1159
nir_var_vec_indexable_modes = 1969033
nir_num_variable_modes = 21
nir_var_all = 2097151
c__EA_nir_variable_mode = ctypes.c_uint32 # enum
nir_variable_mode = c__EA_nir_variable_mode
nir_variable_mode__enumvalues = c__EA_nir_variable_mode__enumvalues

# values for enumeration 'c__EA_nir_lower_int64_options'
c__EA_nir_lower_int64_options__enumvalues = {
    1: 'nir_lower_imul64',
    2: 'nir_lower_isign64',
    4: 'nir_lower_divmod64',
    8: 'nir_lower_imul_high64',
    16: 'nir_lower_bcsel64',
    32: 'nir_lower_icmp64',
    64: 'nir_lower_iadd64',
    128: 'nir_lower_iabs64',
    256: 'nir_lower_ineg64',
    512: 'nir_lower_logic64',
    1024: 'nir_lower_minmax64',
    2048: 'nir_lower_shift64',
    4096: 'nir_lower_imul_2x32_64',
    8192: 'nir_lower_extract64',
    16384: 'nir_lower_ufind_msb64',
    32768: 'nir_lower_bit_count64',
    65536: 'nir_lower_subgroup_shuffle64',
    131072: 'nir_lower_scan_reduce_bitwise64',
    262144: 'nir_lower_scan_reduce_iadd64',
    524288: 'nir_lower_vote_ieq64',
    1048576: 'nir_lower_usub_sat64',
    2097152: 'nir_lower_iadd_sat64',
    4194304: 'nir_lower_find_lsb64',
    8388608: 'nir_lower_conv64',
    16777216: 'nir_lower_uadd_sat64',
    33554432: 'nir_lower_iadd3_64',
}
nir_lower_imul64 = 1
nir_lower_isign64 = 2
nir_lower_divmod64 = 4
nir_lower_imul_high64 = 8
nir_lower_bcsel64 = 16
nir_lower_icmp64 = 32
nir_lower_iadd64 = 64
nir_lower_iabs64 = 128
nir_lower_ineg64 = 256
nir_lower_logic64 = 512
nir_lower_minmax64 = 1024
nir_lower_shift64 = 2048
nir_lower_imul_2x32_64 = 4096
nir_lower_extract64 = 8192
nir_lower_ufind_msb64 = 16384
nir_lower_bit_count64 = 32768
nir_lower_subgroup_shuffle64 = 65536
nir_lower_scan_reduce_bitwise64 = 131072
nir_lower_scan_reduce_iadd64 = 262144
nir_lower_vote_ieq64 = 524288
nir_lower_usub_sat64 = 1048576
nir_lower_iadd_sat64 = 2097152
nir_lower_find_lsb64 = 4194304
nir_lower_conv64 = 8388608
nir_lower_uadd_sat64 = 16777216
nir_lower_iadd3_64 = 33554432
c__EA_nir_lower_int64_options = ctypes.c_uint32 # enum
nir_lower_int64_options = c__EA_nir_lower_int64_options
nir_lower_int64_options__enumvalues = c__EA_nir_lower_int64_options__enumvalues

# values for enumeration 'c__EA_nir_lower_doubles_options'
c__EA_nir_lower_doubles_options__enumvalues = {
    1: 'nir_lower_drcp',
    2: 'nir_lower_dsqrt',
    4: 'nir_lower_drsq',
    8: 'nir_lower_dtrunc',
    16: 'nir_lower_dfloor',
    32: 'nir_lower_dceil',
    64: 'nir_lower_dfract',
    128: 'nir_lower_dround_even',
    256: 'nir_lower_dmod',
    512: 'nir_lower_dsub',
    1024: 'nir_lower_ddiv',
    2048: 'nir_lower_dsign',
    4096: 'nir_lower_dminmax',
    8192: 'nir_lower_dsat',
    16384: 'nir_lower_fp64_full_software',
}
nir_lower_drcp = 1
nir_lower_dsqrt = 2
nir_lower_drsq = 4
nir_lower_dtrunc = 8
nir_lower_dfloor = 16
nir_lower_dceil = 32
nir_lower_dfract = 64
nir_lower_dround_even = 128
nir_lower_dmod = 256
nir_lower_dsub = 512
nir_lower_ddiv = 1024
nir_lower_dsign = 2048
nir_lower_dminmax = 4096
nir_lower_dsat = 8192
nir_lower_fp64_full_software = 16384
c__EA_nir_lower_doubles_options = ctypes.c_uint32 # enum
nir_lower_doubles_options = c__EA_nir_lower_doubles_options
nir_lower_doubles_options__enumvalues = c__EA_nir_lower_doubles_options__enumvalues

# values for enumeration 'c__EA_nir_divergence_options'
c__EA_nir_divergence_options__enumvalues = {
    1: 'nir_divergence_single_prim_per_subgroup',
    2: 'nir_divergence_single_patch_per_tcs_subgroup',
    4: 'nir_divergence_single_patch_per_tes_subgroup',
    8: 'nir_divergence_view_index_uniform',
    16: 'nir_divergence_single_frag_shading_rate_per_subgroup',
    32: 'nir_divergence_multiple_workgroup_per_compute_subgroup',
    64: 'nir_divergence_shader_record_ptr_uniform',
    128: 'nir_divergence_uniform_load_tears',
    256: 'nir_divergence_ignore_undef_if_phi_srcs',
}
nir_divergence_single_prim_per_subgroup = 1
nir_divergence_single_patch_per_tcs_subgroup = 2
nir_divergence_single_patch_per_tes_subgroup = 4
nir_divergence_view_index_uniform = 8
nir_divergence_single_frag_shading_rate_per_subgroup = 16
nir_divergence_multiple_workgroup_per_compute_subgroup = 32
nir_divergence_shader_record_ptr_uniform = 64
nir_divergence_uniform_load_tears = 128
nir_divergence_ignore_undef_if_phi_srcs = 256
c__EA_nir_divergence_options = ctypes.c_uint32 # enum
nir_divergence_options = c__EA_nir_divergence_options
nir_divergence_options__enumvalues = c__EA_nir_divergence_options__enumvalues

# values for enumeration 'c__EA_nir_io_options'
c__EA_nir_io_options__enumvalues = {
    1: 'nir_io_has_flexible_input_interpolation_except_flat',
    2: 'nir_io_dont_use_pos_for_non_fs_varyings',
    4: 'nir_io_16bit_input_output_support',
    8: 'nir_io_mediump_is_32bit',
    16: 'nir_io_prefer_scalar_fs_inputs',
    32: 'nir_io_mix_convergent_flat_with_interpolated',
    64: 'nir_io_vectorizer_ignores_types',
    128: 'nir_io_always_interpolate_convergent_fs_inputs',
    256: 'nir_io_compaction_rotates_color_channels',
    65536: 'nir_io_has_intrinsics',
    131072: 'nir_io_dont_optimize',
    262144: 'nir_io_separate_clip_cull_distance_arrays',
}
nir_io_has_flexible_input_interpolation_except_flat = 1
nir_io_dont_use_pos_for_non_fs_varyings = 2
nir_io_16bit_input_output_support = 4
nir_io_mediump_is_32bit = 8
nir_io_prefer_scalar_fs_inputs = 16
nir_io_mix_convergent_flat_with_interpolated = 32
nir_io_vectorizer_ignores_types = 64
nir_io_always_interpolate_convergent_fs_inputs = 128
nir_io_compaction_rotates_color_channels = 256
nir_io_has_intrinsics = 65536
nir_io_dont_optimize = 131072
nir_io_separate_clip_cull_distance_arrays = 262144
c__EA_nir_io_options = ctypes.c_uint32 # enum
nir_io_options = c__EA_nir_io_options
nir_io_options__enumvalues = c__EA_nir_io_options__enumvalues
class struct_nir_shader(Structure):
    pass

struct_nir_shader_compiler_options._pack_ = 1 # source:False
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
    ('lower_fdot', ctypes.c_bool),
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
    ('lower_all_io_to_temps', ctypes.c_bool),
    ('vertex_id_zero_based', ctypes.c_bool),
    ('lower_base_vertex', ctypes.c_bool),
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
    ('vectorize_tess_levels', ctypes.c_bool),
    ('lower_to_scalar', ctypes.c_bool),
    ('lower_to_scalar_filter', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
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
    ('support_indirect_inputs', ctypes.c_ubyte),
    ('support_indirect_outputs', ctypes.c_ubyte),
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

try:
    nak_nir_options = _libraries['FIXME_STUB'].nak_nir_options
    nak_nir_options.restype = ctypes.POINTER(struct_nir_shader_compiler_options)
    nak_nir_options.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except (AttributeError, RuntimeError):
    pass
try:
    nak_preprocess_nir = _libraries['FIXME_STUB'].nak_preprocess_nir
    nak_preprocess_nir.restype = None
    nak_preprocess_nir.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nak_compiler)]
except (AttributeError, RuntimeError):
    pass
class struct_nak_sample_location(Structure):
    pass

struct_nak_sample_location._pack_ = 1 # source:False
struct_nak_sample_location._fields_ = [
    ('x_u4', ctypes.c_ubyte, 4),
    ('y_u4', ctypes.c_ubyte, 4),
]

class struct_nak_sample_mask(Structure):
    pass

struct_nak_sample_mask._pack_ = 1 # source:False
struct_nak_sample_mask._fields_ = [
    ('sample_mask', ctypes.c_uint16),
]

class struct_nak_fs_key(Structure):
    pass

struct_nak_fs_key._pack_ = 1 # source:False
struct_nak_fs_key._fields_ = [
    ('zs_self_dep', ctypes.c_bool),
    ('force_sample_shading', ctypes.c_bool),
    ('uses_underestimate', ctypes.c_bool),
    ('sample_info_cb', ctypes.c_ubyte),
    ('sample_locations_offset', ctypes.c_uint32),
    ('sample_masks_offset', ctypes.c_uint32),
]

try:
    nak_postprocess_nir = _libraries['FIXME_STUB'].nak_postprocess_nir
    nak_postprocess_nir.restype = None
    nak_postprocess_nir.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except (AttributeError, RuntimeError):
    pass

# values for enumeration 'nak_ts_domain'
nak_ts_domain__enumvalues = {
    0: 'NAK_TS_DOMAIN_ISOLINE',
    1: 'NAK_TS_DOMAIN_TRIANGLE',
    2: 'NAK_TS_DOMAIN_QUAD',
}
NAK_TS_DOMAIN_ISOLINE = 0
NAK_TS_DOMAIN_TRIANGLE = 1
NAK_TS_DOMAIN_QUAD = 2
nak_ts_domain = ctypes.c_uint32 # enum

# values for enumeration 'nak_ts_spacing'
nak_ts_spacing__enumvalues = {
    0: 'NAK_TS_SPACING_INTEGER',
    1: 'NAK_TS_SPACING_FRACT_ODD',
    2: 'NAK_TS_SPACING_FRACT_EVEN',
}
NAK_TS_SPACING_INTEGER = 0
NAK_TS_SPACING_FRACT_ODD = 1
NAK_TS_SPACING_FRACT_EVEN = 2
nak_ts_spacing = ctypes.c_uint32 # enum

# values for enumeration 'nak_ts_prims'
nak_ts_prims__enumvalues = {
    0: 'NAK_TS_PRIMS_POINTS',
    1: 'NAK_TS_PRIMS_LINES',
    2: 'NAK_TS_PRIMS_TRIANGLES_CW',
    3: 'NAK_TS_PRIMS_TRIANGLES_CCW',
}
NAK_TS_PRIMS_POINTS = 0
NAK_TS_PRIMS_LINES = 1
NAK_TS_PRIMS_TRIANGLES_CW = 2
NAK_TS_PRIMS_TRIANGLES_CCW = 3
nak_ts_prims = ctypes.c_uint32 # enum
class struct_nak_xfb_info(Structure):
    pass

struct_nak_xfb_info._pack_ = 1 # source:False
struct_nak_xfb_info._fields_ = [
    ('stride', ctypes.c_uint32 * 4),
    ('stream', ctypes.c_ubyte * 4),
    ('attr_count', ctypes.c_ubyte * 4),
    ('attr_index', ctypes.c_ubyte * 128 * 4),
]

class struct_nak_shader_info(Structure):
    pass


# values for enumeration 'pipe_shader_type'
pipe_shader_type__enumvalues = {
    -1: 'MESA_SHADER_NONE',
    0: 'MESA_SHADER_VERTEX',
    0: 'PIPE_SHADER_VERTEX',
    1: 'MESA_SHADER_TESS_CTRL',
    1: 'PIPE_SHADER_TESS_CTRL',
    2: 'MESA_SHADER_TESS_EVAL',
    2: 'PIPE_SHADER_TESS_EVAL',
    3: 'MESA_SHADER_GEOMETRY',
    3: 'PIPE_SHADER_GEOMETRY',
    4: 'MESA_SHADER_FRAGMENT',
    4: 'PIPE_SHADER_FRAGMENT',
    5: 'MESA_SHADER_COMPUTE',
    5: 'PIPE_SHADER_COMPUTE',
    6: 'PIPE_SHADER_TYPES',
    6: 'MESA_SHADER_TASK',
    6: 'PIPE_SHADER_TASK',
    7: 'MESA_SHADER_MESH',
    7: 'PIPE_SHADER_MESH',
    8: 'PIPE_SHADER_MESH_TYPES',
    8: 'MESA_SHADER_RAYGEN',
    9: 'MESA_SHADER_ANY_HIT',
    10: 'MESA_SHADER_CLOSEST_HIT',
    11: 'MESA_SHADER_MISS',
    12: 'MESA_SHADER_INTERSECTION',
    13: 'MESA_SHADER_CALLABLE',
    14: 'MESA_SHADER_KERNEL',
}
MESA_SHADER_NONE = -1
MESA_SHADER_VERTEX = 0
PIPE_SHADER_VERTEX = 0
MESA_SHADER_TESS_CTRL = 1
PIPE_SHADER_TESS_CTRL = 1
MESA_SHADER_TESS_EVAL = 2
PIPE_SHADER_TESS_EVAL = 2
MESA_SHADER_GEOMETRY = 3
PIPE_SHADER_GEOMETRY = 3
MESA_SHADER_FRAGMENT = 4
PIPE_SHADER_FRAGMENT = 4
MESA_SHADER_COMPUTE = 5
PIPE_SHADER_COMPUTE = 5
PIPE_SHADER_TYPES = 6
MESA_SHADER_TASK = 6
PIPE_SHADER_TASK = 6
MESA_SHADER_MESH = 7
PIPE_SHADER_MESH = 7
PIPE_SHADER_MESH_TYPES = 8
MESA_SHADER_RAYGEN = 8
MESA_SHADER_ANY_HIT = 9
MESA_SHADER_CLOSEST_HIT = 10
MESA_SHADER_MISS = 11
MESA_SHADER_INTERSECTION = 12
MESA_SHADER_CALLABLE = 13
MESA_SHADER_KERNEL = 14
pipe_shader_type = ctypes.c_int32 # enum
class union_nak_shader_info_0(Union):
    pass

class struct_nak_shader_info_0_cs(Structure):
    pass

struct_nak_shader_info_0_cs._pack_ = 1 # source:False
struct_nak_shader_info_0_cs._fields_ = [
    ('local_size', ctypes.c_uint16 * 3),
    ('smem_size', ctypes.c_uint16),
    ('_pad', ctypes.c_ubyte * 4),
]

class struct_nak_shader_info_0_fs(Structure):
    pass

struct_nak_shader_info_0_fs._pack_ = 1 # source:False
struct_nak_shader_info_0_fs._fields_ = [
    ('writes_depth', ctypes.c_bool),
    ('reads_sample_mask', ctypes.c_bool),
    ('post_depth_coverage', ctypes.c_bool),
    ('uses_sample_shading', ctypes.c_bool),
    ('early_fragment_tests', ctypes.c_bool),
    ('_pad', ctypes.c_ubyte * 7),
]

class struct_nak_shader_info_0_ts(Structure):
    pass

struct_nak_shader_info_0_ts._pack_ = 1 # source:False
struct_nak_shader_info_0_ts._fields_ = [
    ('domain', ctypes.c_ubyte),
    ('spacing', ctypes.c_ubyte),
    ('prims', ctypes.c_ubyte),
    ('_pad', ctypes.c_ubyte * 9),
]

union_nak_shader_info_0._pack_ = 1 # source:False
union_nak_shader_info_0._fields_ = [
    ('cs', struct_nak_shader_info_0_cs),
    ('fs', struct_nak_shader_info_0_fs),
    ('ts', struct_nak_shader_info_0_ts),
    ('_pad', ctypes.c_ubyte * 12),
]

class struct_nak_shader_info_vtg(Structure):
    pass

struct_nak_shader_info_vtg._pack_ = 1 # source:False
struct_nak_shader_info_vtg._fields_ = [
    ('writes_layer', ctypes.c_bool),
    ('writes_point_size', ctypes.c_bool),
    ('writes_vprs_table_index', ctypes.c_bool),
    ('clip_enable', ctypes.c_ubyte),
    ('cull_enable', ctypes.c_ubyte),
    ('_pad', ctypes.c_ubyte * 3),
    ('xfb', struct_nak_xfb_info),
]

struct_nak_shader_info._pack_ = 1 # source:False
struct_nak_shader_info._anonymous_ = ('_0',)
struct_nak_shader_info._fields_ = [
    ('stage', pipe_shader_type),
    ('sm', ctypes.c_ubyte),
    ('num_gprs', ctypes.c_ubyte),
    ('num_control_barriers', ctypes.c_ubyte),
    ('_pad0', ctypes.c_ubyte),
    ('max_warps_per_sm', ctypes.c_uint32),
    ('num_instrs', ctypes.c_uint32),
    ('num_static_cycles', ctypes.c_uint32),
    ('num_spills_to_mem', ctypes.c_uint32),
    ('num_fills_from_mem', ctypes.c_uint32),
    ('num_spills_to_reg', ctypes.c_uint32),
    ('num_fills_from_reg', ctypes.c_uint32),
    ('slm_size', ctypes.c_uint32),
    ('crs_size', ctypes.c_uint32),
    ('_0', union_nak_shader_info_0),
    ('vtg', struct_nak_shader_info_vtg),
    ('hdr', ctypes.c_uint32 * 32),
]

class struct_nak_shader_bin(Structure):
    pass

struct_nak_shader_bin._pack_ = 1 # source:False
struct_nak_shader_bin._fields_ = [
    ('info', struct_nak_shader_info),
    ('code_size', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('code', ctypes.POINTER(None)),
    ('asm_str', ctypes.POINTER(ctypes.c_char)),
]

try:
    nak_shader_bin_destroy = _libraries['FIXME_STUB'].nak_shader_bin_destroy
    nak_shader_bin_destroy.restype = None
    nak_shader_bin_destroy.argtypes = [ctypes.POINTER(struct_nak_shader_bin)]
except (AttributeError, RuntimeError):
    pass
try:
    nak_compile_shader = _libraries['FIXME_STUB'].nak_compile_shader
    nak_compile_shader.restype = ctypes.POINTER(struct_nak_shader_bin)
    nak_compile_shader.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except (AttributeError, RuntimeError):
    pass
class struct_nak_qmd_cbuf(Structure):
    pass

struct_nak_qmd_cbuf._pack_ = 1 # source:False
struct_nak_qmd_cbuf._fields_ = [
    ('index', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('addr', ctypes.c_uint64),
]

class struct_nak_qmd_info(Structure):
    pass

struct_nak_qmd_info._pack_ = 1 # source:False
struct_nak_qmd_info._fields_ = [
    ('addr', ctypes.c_uint64),
    ('smem_size', ctypes.c_uint16),
    ('smem_max', ctypes.c_uint16),
    ('global_size', ctypes.c_uint32 * 3),
    ('num_cbufs', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('cbufs', struct_nak_qmd_cbuf * 8),
]

try:
    nak_fill_qmd = _libraries['FIXME_STUB'].nak_fill_qmd
    nak_fill_qmd.restype = None
    nak_fill_qmd.argtypes = [ctypes.POINTER(struct_nv_device_info), ctypes.POINTER(struct_nak_shader_info), ctypes.POINTER(struct_nak_qmd_info), ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
class struct_nak_qmd_dispatch_size_layout(Structure):
    pass

struct_nak_qmd_dispatch_size_layout._pack_ = 1 # source:False
struct_nak_qmd_dispatch_size_layout._fields_ = [
    ('x_start', ctypes.c_uint16),
    ('x_end', ctypes.c_uint16),
    ('y_start', ctypes.c_uint16),
    ('y_end', ctypes.c_uint16),
    ('z_start', ctypes.c_uint16),
    ('z_end', ctypes.c_uint16),
]

try:
    nak_get_qmd_dispatch_size_layout = _libraries['FIXME_STUB'].nak_get_qmd_dispatch_size_layout
    nak_get_qmd_dispatch_size_layout.restype = struct_nak_qmd_dispatch_size_layout
    nak_get_qmd_dispatch_size_layout.argtypes = [ctypes.POINTER(struct_nv_device_info)]
except (AttributeError, RuntimeError):
    pass
class struct_nak_qmd_cbuf_desc_layout(Structure):
    pass

struct_nak_qmd_cbuf_desc_layout._pack_ = 1 # source:False
struct_nak_qmd_cbuf_desc_layout._fields_ = [
    ('addr_shift', ctypes.c_uint16),
    ('addr_lo_start', ctypes.c_uint16),
    ('addr_lo_end', ctypes.c_uint16),
    ('addr_hi_start', ctypes.c_uint16),
    ('addr_hi_end', ctypes.c_uint16),
]

uint8_t = ctypes.c_uint8
try:
    nak_get_qmd_cbuf_desc_layout = _libraries['FIXME_STUB'].nak_get_qmd_cbuf_desc_layout
    nak_get_qmd_cbuf_desc_layout.restype = struct_nak_qmd_cbuf_desc_layout
    nak_get_qmd_cbuf_desc_layout.argtypes = [ctypes.POINTER(struct_nv_device_info), uint8_t]
except (AttributeError, RuntimeError):
    pass
nir_instr_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))

# values for enumeration 'c__EA_nir_lower_packing_op'
c__EA_nir_lower_packing_op__enumvalues = {
    0: 'nir_lower_packing_op_pack_64_2x32',
    1: 'nir_lower_packing_op_unpack_64_2x32',
    2: 'nir_lower_packing_op_pack_64_4x16',
    3: 'nir_lower_packing_op_unpack_64_4x16',
    4: 'nir_lower_packing_op_pack_32_2x16',
    5: 'nir_lower_packing_op_unpack_32_2x16',
    6: 'nir_lower_packing_op_pack_32_4x8',
    7: 'nir_lower_packing_op_unpack_32_4x8',
    8: 'nir_lower_packing_num_ops',
}
nir_lower_packing_op_pack_64_2x32 = 0
nir_lower_packing_op_unpack_64_2x32 = 1
nir_lower_packing_op_pack_64_4x16 = 2
nir_lower_packing_op_unpack_64_4x16 = 3
nir_lower_packing_op_pack_32_2x16 = 4
nir_lower_packing_op_unpack_32_2x16 = 5
nir_lower_packing_op_pack_32_4x8 = 6
nir_lower_packing_op_unpack_32_4x8 = 7
nir_lower_packing_num_ops = 8
c__EA_nir_lower_packing_op = ctypes.c_uint32 # enum
nir_lower_packing_op = c__EA_nir_lower_packing_op
nir_lower_packing_op__enumvalues = c__EA_nir_lower_packing_op__enumvalues
nir_shader_compiler_options = struct_nir_shader_compiler_options
class struct_blob(Structure):
    pass

struct_blob._pack_ = 1 # source:False
struct_blob._fields_ = [
    ('data', ctypes.POINTER(ctypes.c_ubyte)),
    ('allocated', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('fixed_allocation', ctypes.c_bool),
    ('out_of_memory', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 6),
]

class struct_blob_reader(Structure):
    pass

struct_blob_reader._pack_ = 1 # source:False
struct_blob_reader._fields_ = [
    ('data', ctypes.POINTER(ctypes.c_ubyte)),
    ('end', ctypes.POINTER(ctypes.c_ubyte)),
    ('current', ctypes.POINTER(ctypes.c_ubyte)),
    ('overrun', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

try:
    blob_init = _libraries['FIXME_STUB'].blob_init
    blob_init.restype = None
    blob_init.argtypes = [ctypes.POINTER(struct_blob)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_init_fixed = _libraries['FIXME_STUB'].blob_init_fixed
    blob_init_fixed.restype = None
    blob_init_fixed.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_finish = _libraries['FIXME_STUB'].blob_finish
    blob_finish.restype = None
    blob_finish.argtypes = [ctypes.POINTER(struct_blob)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_finish_get_buffer = _libraries['FIXME_STUB'].blob_finish_get_buffer
    blob_finish_get_buffer.restype = None
    blob_finish_get_buffer.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_align = _libraries['FIXME_STUB'].blob_align
    blob_align.restype = ctypes.c_bool
    blob_align.argtypes = [ctypes.POINTER(struct_blob), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_write_bytes = _libraries['FIXME_STUB'].blob_write_bytes
    blob_write_bytes.restype = ctypes.c_bool
    blob_write_bytes.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
intptr_t = ctypes.c_int64
try:
    blob_reserve_bytes = _libraries['FIXME_STUB'].blob_reserve_bytes
    blob_reserve_bytes.restype = intptr_t
    blob_reserve_bytes.argtypes = [ctypes.POINTER(struct_blob), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_reserve_uint32 = _libraries['FIXME_STUB'].blob_reserve_uint32
    blob_reserve_uint32.restype = intptr_t
    blob_reserve_uint32.argtypes = [ctypes.POINTER(struct_blob)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_reserve_intptr = _libraries['FIXME_STUB'].blob_reserve_intptr
    blob_reserve_intptr.restype = intptr_t
    blob_reserve_intptr.argtypes = [ctypes.POINTER(struct_blob)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_overwrite_bytes = _libraries['FIXME_STUB'].blob_overwrite_bytes
    blob_overwrite_bytes.restype = ctypes.c_bool
    blob_overwrite_bytes.argtypes = [ctypes.POINTER(struct_blob), size_t, ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_write_uint8 = _libraries['FIXME_STUB'].blob_write_uint8
    blob_write_uint8.restype = ctypes.c_bool
    blob_write_uint8.argtypes = [ctypes.POINTER(struct_blob), uint8_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_overwrite_uint8 = _libraries['FIXME_STUB'].blob_overwrite_uint8
    blob_overwrite_uint8.restype = ctypes.c_bool
    blob_overwrite_uint8.argtypes = [ctypes.POINTER(struct_blob), size_t, uint8_t]
except (AttributeError, RuntimeError):
    pass
uint16_t = ctypes.c_uint16
try:
    blob_write_uint16 = _libraries['FIXME_STUB'].blob_write_uint16
    blob_write_uint16.restype = ctypes.c_bool
    blob_write_uint16.argtypes = [ctypes.POINTER(struct_blob), uint16_t]
except (AttributeError, RuntimeError):
    pass
uint32_t = ctypes.c_uint32
try:
    blob_write_uint32 = _libraries['FIXME_STUB'].blob_write_uint32
    blob_write_uint32.restype = ctypes.c_bool
    blob_write_uint32.argtypes = [ctypes.POINTER(struct_blob), uint32_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_overwrite_uint32 = _libraries['FIXME_STUB'].blob_overwrite_uint32
    blob_overwrite_uint32.restype = ctypes.c_bool
    blob_overwrite_uint32.argtypes = [ctypes.POINTER(struct_blob), size_t, uint32_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_write_uint64 = _libraries['FIXME_STUB'].blob_write_uint64
    blob_write_uint64.restype = ctypes.c_bool
    blob_write_uint64.argtypes = [ctypes.POINTER(struct_blob), uint64_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_write_intptr = _libraries['FIXME_STUB'].blob_write_intptr
    blob_write_intptr.restype = ctypes.c_bool
    blob_write_intptr.argtypes = [ctypes.POINTER(struct_blob), intptr_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_overwrite_intptr = _libraries['FIXME_STUB'].blob_overwrite_intptr
    blob_overwrite_intptr.restype = ctypes.c_bool
    blob_overwrite_intptr.argtypes = [ctypes.POINTER(struct_blob), size_t, intptr_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_write_string = _libraries['FIXME_STUB'].blob_write_string
    blob_write_string.restype = ctypes.c_bool
    blob_write_string.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_reader_init = _libraries['FIXME_STUB'].blob_reader_init
    blob_reader_init.restype = None
    blob_reader_init.argtypes = [ctypes.POINTER(struct_blob_reader), ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_reader_align = _libraries['FIXME_STUB'].blob_reader_align
    blob_reader_align.restype = None
    blob_reader_align.argtypes = [ctypes.POINTER(struct_blob_reader), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_read_bytes = _libraries['FIXME_STUB'].blob_read_bytes
    blob_read_bytes.restype = ctypes.POINTER(None)
    blob_read_bytes.argtypes = [ctypes.POINTER(struct_blob_reader), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_copy_bytes = _libraries['FIXME_STUB'].blob_copy_bytes
    blob_copy_bytes.restype = None
    blob_copy_bytes.argtypes = [ctypes.POINTER(struct_blob_reader), ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_skip_bytes = _libraries['FIXME_STUB'].blob_skip_bytes
    blob_skip_bytes.restype = None
    blob_skip_bytes.argtypes = [ctypes.POINTER(struct_blob_reader), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    blob_read_uint8 = _libraries['FIXME_STUB'].blob_read_uint8
    blob_read_uint8.restype = uint8_t
    blob_read_uint8.argtypes = [ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_read_uint16 = _libraries['FIXME_STUB'].blob_read_uint16
    blob_read_uint16.restype = uint16_t
    blob_read_uint16.argtypes = [ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_read_uint32 = _libraries['FIXME_STUB'].blob_read_uint32
    blob_read_uint32.restype = uint32_t
    blob_read_uint32.argtypes = [ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_read_uint64 = _libraries['FIXME_STUB'].blob_read_uint64
    blob_read_uint64.restype = uint64_t
    blob_read_uint64.argtypes = [ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_read_intptr = _libraries['FIXME_STUB'].blob_read_intptr
    blob_read_intptr.restype = intptr_t
    blob_read_intptr.argtypes = [ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
try:
    blob_read_string = _libraries['FIXME_STUB'].blob_read_string
    blob_read_string.restype = ctypes.POINTER(ctypes.c_char)
    blob_read_string.argtypes = [ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
try:
    nir_serialize = _libraries['FIXME_STUB'].nir_serialize
    nir_serialize.restype = None
    nir_serialize.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    nir_deserialize = _libraries['FIXME_STUB'].nir_deserialize
    nir_deserialize.restype = ctypes.POINTER(struct_nir_shader)
    nir_deserialize.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
class struct_nir_function(Structure):
    pass

try:
    nir_serialize_function = _libraries['FIXME_STUB'].nir_serialize_function
    nir_serialize_function.restype = None
    nir_serialize_function.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(struct_nir_function)]
except (AttributeError, RuntimeError):
    pass
try:
    nir_deserialize_function = _libraries['FIXME_STUB'].nir_deserialize_function
    nir_deserialize_function.restype = ctypes.POINTER(struct_nir_function)
    nir_deserialize_function.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
class struct_glsl_type(Structure):
    pass


# values for enumeration 'glsl_base_type'
glsl_base_type__enumvalues = {
    0: 'GLSL_TYPE_UINT',
    1: 'GLSL_TYPE_INT',
    2: 'GLSL_TYPE_FLOAT',
    3: 'GLSL_TYPE_FLOAT16',
    4: 'GLSL_TYPE_DOUBLE',
    5: 'GLSL_TYPE_UINT8',
    6: 'GLSL_TYPE_INT8',
    7: 'GLSL_TYPE_UINT16',
    8: 'GLSL_TYPE_INT16',
    9: 'GLSL_TYPE_UINT64',
    10: 'GLSL_TYPE_INT64',
    11: 'GLSL_TYPE_BOOL',
    12: 'GLSL_TYPE_COOPERATIVE_MATRIX',
    13: 'GLSL_TYPE_SAMPLER',
    14: 'GLSL_TYPE_TEXTURE',
    15: 'GLSL_TYPE_IMAGE',
    16: 'GLSL_TYPE_ATOMIC_UINT',
    17: 'GLSL_TYPE_STRUCT',
    18: 'GLSL_TYPE_INTERFACE',
    19: 'GLSL_TYPE_ARRAY',
    20: 'GLSL_TYPE_VOID',
    21: 'GLSL_TYPE_SUBROUTINE',
    22: 'GLSL_TYPE_ERROR',
}
GLSL_TYPE_UINT = 0
GLSL_TYPE_INT = 1
GLSL_TYPE_FLOAT = 2
GLSL_TYPE_FLOAT16 = 3
GLSL_TYPE_DOUBLE = 4
GLSL_TYPE_UINT8 = 5
GLSL_TYPE_INT8 = 6
GLSL_TYPE_UINT16 = 7
GLSL_TYPE_INT16 = 8
GLSL_TYPE_UINT64 = 9
GLSL_TYPE_INT64 = 10
GLSL_TYPE_BOOL = 11
GLSL_TYPE_COOPERATIVE_MATRIX = 12
GLSL_TYPE_SAMPLER = 13
GLSL_TYPE_TEXTURE = 14
GLSL_TYPE_IMAGE = 15
GLSL_TYPE_ATOMIC_UINT = 16
GLSL_TYPE_STRUCT = 17
GLSL_TYPE_INTERFACE = 18
GLSL_TYPE_ARRAY = 19
GLSL_TYPE_VOID = 20
GLSL_TYPE_SUBROUTINE = 21
GLSL_TYPE_ERROR = 22
glsl_base_type = ctypes.c_uint32 # enum
class struct_glsl_cmat_description(Structure):
    pass

struct_glsl_cmat_description._pack_ = 1 # source:False
struct_glsl_cmat_description._fields_ = [
    ('element_type', ctypes.c_ubyte, 5),
    ('scope', ctypes.c_ubyte, 3),
    ('rows', ctypes.c_ubyte, 8),
    ('cols', ctypes.c_ubyte),
    ('use', ctypes.c_ubyte),
]

class union_glsl_type_fields(Union):
    pass

class struct_glsl_struct_field(Structure):
    pass

union_glsl_type_fields._pack_ = 1 # source:False
union_glsl_type_fields._fields_ = [
    ('array', ctypes.POINTER(struct_glsl_type)),
    ('structure', ctypes.POINTER(struct_glsl_struct_field)),
]

struct_glsl_type._pack_ = 1 # source:False
struct_glsl_type._fields_ = [
    ('gl_type', ctypes.c_uint32),
    ('base_type', glsl_base_type, 8),
    ('sampled_type', glsl_base_type, 8),
    ('sampler_dimensionality', glsl_base_type, 4),
    ('sampler_shadow', glsl_base_type, 1),
    ('sampler_array', glsl_base_type, 1),
    ('interface_packing', glsl_base_type, 2),
    ('interface_row_major', glsl_base_type, 1),
    ('PADDING_0', ctypes.c_uint8, 7),
    ('cmat_desc', struct_glsl_cmat_description),
    ('packed', ctypes.c_uint32, 1),
    ('has_builtin_name', ctypes.c_uint32, 1),
    ('PADDING_1', ctypes.c_uint8, 6),
    ('vector_elements', ctypes.c_uint32, 8),
    ('matrix_columns', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte),
    ('length', ctypes.c_uint32),
    ('PADDING_3', ctypes.c_ubyte * 4),
    ('name_id', ctypes.c_uint64),
    ('explicit_stride', ctypes.c_uint32),
    ('explicit_alignment', ctypes.c_uint32),
    ('fields', union_glsl_type_fields),
]

glsl_type = struct_glsl_type

# values for enumeration 'pipe_format'
pipe_format__enumvalues = {
    0: 'PIPE_FORMAT_NONE',
    1: 'PIPE_FORMAT_R64_UINT',
    2: 'PIPE_FORMAT_R64G64_UINT',
    3: 'PIPE_FORMAT_R64G64B64_UINT',
    4: 'PIPE_FORMAT_R64G64B64A64_UINT',
    5: 'PIPE_FORMAT_R64_SINT',
    6: 'PIPE_FORMAT_R64G64_SINT',
    7: 'PIPE_FORMAT_R64G64B64_SINT',
    8: 'PIPE_FORMAT_R64G64B64A64_SINT',
    9: 'PIPE_FORMAT_R64_FLOAT',
    10: 'PIPE_FORMAT_R64G64_FLOAT',
    11: 'PIPE_FORMAT_R64G64B64_FLOAT',
    12: 'PIPE_FORMAT_R64G64B64A64_FLOAT',
    13: 'PIPE_FORMAT_R32_FLOAT',
    14: 'PIPE_FORMAT_R32G32_FLOAT',
    15: 'PIPE_FORMAT_R32G32B32_FLOAT',
    16: 'PIPE_FORMAT_R32G32B32A32_FLOAT',
    17: 'PIPE_FORMAT_R32_UNORM',
    18: 'PIPE_FORMAT_R32G32_UNORM',
    19: 'PIPE_FORMAT_R32G32B32_UNORM',
    20: 'PIPE_FORMAT_R32G32B32A32_UNORM',
    21: 'PIPE_FORMAT_R32_USCALED',
    22: 'PIPE_FORMAT_R32G32_USCALED',
    23: 'PIPE_FORMAT_R32G32B32_USCALED',
    24: 'PIPE_FORMAT_R32G32B32A32_USCALED',
    25: 'PIPE_FORMAT_R32_SNORM',
    26: 'PIPE_FORMAT_R32G32_SNORM',
    27: 'PIPE_FORMAT_R32G32B32_SNORM',
    28: 'PIPE_FORMAT_R32G32B32A32_SNORM',
    29: 'PIPE_FORMAT_R32_SSCALED',
    30: 'PIPE_FORMAT_R32G32_SSCALED',
    31: 'PIPE_FORMAT_R32G32B32_SSCALED',
    32: 'PIPE_FORMAT_R32G32B32A32_SSCALED',
    33: 'PIPE_FORMAT_R16_UNORM',
    34: 'PIPE_FORMAT_R16G16_UNORM',
    35: 'PIPE_FORMAT_R16G16B16_UNORM',
    36: 'PIPE_FORMAT_R16G16B16A16_UNORM',
    37: 'PIPE_FORMAT_R16_USCALED',
    38: 'PIPE_FORMAT_R16G16_USCALED',
    39: 'PIPE_FORMAT_R16G16B16_USCALED',
    40: 'PIPE_FORMAT_R16G16B16A16_USCALED',
    41: 'PIPE_FORMAT_R16_SNORM',
    42: 'PIPE_FORMAT_R16G16_SNORM',
    43: 'PIPE_FORMAT_R16G16B16_SNORM',
    44: 'PIPE_FORMAT_R16G16B16A16_SNORM',
    45: 'PIPE_FORMAT_R16_SSCALED',
    46: 'PIPE_FORMAT_R16G16_SSCALED',
    47: 'PIPE_FORMAT_R16G16B16_SSCALED',
    48: 'PIPE_FORMAT_R16G16B16A16_SSCALED',
    49: 'PIPE_FORMAT_R8_UNORM',
    50: 'PIPE_FORMAT_R8G8_UNORM',
    51: 'PIPE_FORMAT_R8G8B8_UNORM',
    52: 'PIPE_FORMAT_B8G8R8_UNORM',
    53: 'PIPE_FORMAT_R8G8B8A8_UNORM',
    54: 'PIPE_FORMAT_B8G8R8A8_UNORM',
    55: 'PIPE_FORMAT_R8_USCALED',
    56: 'PIPE_FORMAT_R8G8_USCALED',
    57: 'PIPE_FORMAT_R8G8B8_USCALED',
    58: 'PIPE_FORMAT_B8G8R8_USCALED',
    59: 'PIPE_FORMAT_R8G8B8A8_USCALED',
    60: 'PIPE_FORMAT_B8G8R8A8_USCALED',
    61: 'PIPE_FORMAT_A8B8G8R8_USCALED',
    62: 'PIPE_FORMAT_R8_SNORM',
    63: 'PIPE_FORMAT_R8G8_SNORM',
    64: 'PIPE_FORMAT_R8G8B8_SNORM',
    65: 'PIPE_FORMAT_B8G8R8_SNORM',
    66: 'PIPE_FORMAT_R8G8B8A8_SNORM',
    67: 'PIPE_FORMAT_B8G8R8A8_SNORM',
    68: 'PIPE_FORMAT_R8_SSCALED',
    69: 'PIPE_FORMAT_R8G8_SSCALED',
    70: 'PIPE_FORMAT_R8G8B8_SSCALED',
    71: 'PIPE_FORMAT_B8G8R8_SSCALED',
    72: 'PIPE_FORMAT_R8G8B8A8_SSCALED',
    73: 'PIPE_FORMAT_B8G8R8A8_SSCALED',
    74: 'PIPE_FORMAT_A8B8G8R8_SSCALED',
    75: 'PIPE_FORMAT_A8R8G8B8_UNORM',
    76: 'PIPE_FORMAT_R32_FIXED',
    77: 'PIPE_FORMAT_R32G32_FIXED',
    78: 'PIPE_FORMAT_R32G32B32_FIXED',
    79: 'PIPE_FORMAT_R32G32B32A32_FIXED',
    80: 'PIPE_FORMAT_R16_FLOAT',
    81: 'PIPE_FORMAT_R16G16_FLOAT',
    82: 'PIPE_FORMAT_R16G16B16_FLOAT',
    83: 'PIPE_FORMAT_R16G16B16A16_FLOAT',
    84: 'PIPE_FORMAT_R8_UINT',
    85: 'PIPE_FORMAT_R8G8_UINT',
    86: 'PIPE_FORMAT_R8G8B8_UINT',
    87: 'PIPE_FORMAT_B8G8R8_UINT',
    88: 'PIPE_FORMAT_R8G8B8A8_UINT',
    89: 'PIPE_FORMAT_B8G8R8A8_UINT',
    90: 'PIPE_FORMAT_R8_SINT',
    91: 'PIPE_FORMAT_R8G8_SINT',
    92: 'PIPE_FORMAT_R8G8B8_SINT',
    93: 'PIPE_FORMAT_B8G8R8_SINT',
    94: 'PIPE_FORMAT_R8G8B8A8_SINT',
    95: 'PIPE_FORMAT_B8G8R8A8_SINT',
    96: 'PIPE_FORMAT_R16_UINT',
    97: 'PIPE_FORMAT_R16G16_UINT',
    98: 'PIPE_FORMAT_R16G16B16_UINT',
    99: 'PIPE_FORMAT_R16G16B16A16_UINT',
    100: 'PIPE_FORMAT_R16_SINT',
    101: 'PIPE_FORMAT_R16G16_SINT',
    102: 'PIPE_FORMAT_R16G16B16_SINT',
    103: 'PIPE_FORMAT_R16G16B16A16_SINT',
    104: 'PIPE_FORMAT_R32_UINT',
    105: 'PIPE_FORMAT_R32G32_UINT',
    106: 'PIPE_FORMAT_R32G32B32_UINT',
    107: 'PIPE_FORMAT_R32G32B32A32_UINT',
    108: 'PIPE_FORMAT_R32_SINT',
    109: 'PIPE_FORMAT_R32G32_SINT',
    110: 'PIPE_FORMAT_R32G32B32_SINT',
    111: 'PIPE_FORMAT_R32G32B32A32_SINT',
    112: 'PIPE_FORMAT_R10G10B10A2_UNORM',
    113: 'PIPE_FORMAT_R10G10B10A2_SNORM',
    114: 'PIPE_FORMAT_R10G10B10A2_USCALED',
    115: 'PIPE_FORMAT_R10G10B10A2_SSCALED',
    116: 'PIPE_FORMAT_B10G10R10A2_UNORM',
    117: 'PIPE_FORMAT_B10G10R10A2_SNORM',
    118: 'PIPE_FORMAT_B10G10R10A2_USCALED',
    119: 'PIPE_FORMAT_B10G10R10A2_SSCALED',
    120: 'PIPE_FORMAT_R11G11B10_FLOAT',
    121: 'PIPE_FORMAT_R10G10B10A2_UINT',
    122: 'PIPE_FORMAT_R10G10B10A2_SINT',
    123: 'PIPE_FORMAT_B10G10R10A2_UINT',
    124: 'PIPE_FORMAT_B10G10R10A2_SINT',
    125: 'PIPE_FORMAT_B8G8R8X8_UNORM',
    126: 'PIPE_FORMAT_X8B8G8R8_UNORM',
    127: 'PIPE_FORMAT_X8R8G8B8_UNORM',
    128: 'PIPE_FORMAT_B5G5R5A1_UNORM',
    129: 'PIPE_FORMAT_R4G4B4A4_UNORM',
    130: 'PIPE_FORMAT_B4G4R4A4_UNORM',
    131: 'PIPE_FORMAT_R5G6B5_UNORM',
    132: 'PIPE_FORMAT_B5G6R5_UNORM',
    133: 'PIPE_FORMAT_L8_UNORM',
    134: 'PIPE_FORMAT_A8_UNORM',
    135: 'PIPE_FORMAT_I8_UNORM',
    136: 'PIPE_FORMAT_L8A8_UNORM',
    137: 'PIPE_FORMAT_L16_UNORM',
    138: 'PIPE_FORMAT_UYVY',
    139: 'PIPE_FORMAT_VYUY',
    140: 'PIPE_FORMAT_YUYV',
    141: 'PIPE_FORMAT_YVYU',
    142: 'PIPE_FORMAT_Z16_UNORM',
    143: 'PIPE_FORMAT_Z16_UNORM_S8_UINT',
    144: 'PIPE_FORMAT_Z32_UNORM',
    145: 'PIPE_FORMAT_Z32_FLOAT',
    146: 'PIPE_FORMAT_Z24_UNORM_S8_UINT',
    147: 'PIPE_FORMAT_S8_UINT_Z24_UNORM',
    148: 'PIPE_FORMAT_Z24X8_UNORM',
    149: 'PIPE_FORMAT_X8Z24_UNORM',
    150: 'PIPE_FORMAT_S8_UINT',
    151: 'PIPE_FORMAT_L8_SRGB',
    152: 'PIPE_FORMAT_R8_SRGB',
    153: 'PIPE_FORMAT_L8A8_SRGB',
    154: 'PIPE_FORMAT_R8G8_SRGB',
    155: 'PIPE_FORMAT_R8G8B8_SRGB',
    156: 'PIPE_FORMAT_B8G8R8_SRGB',
    157: 'PIPE_FORMAT_A8B8G8R8_SRGB',
    158: 'PIPE_FORMAT_X8B8G8R8_SRGB',
    159: 'PIPE_FORMAT_B8G8R8A8_SRGB',
    160: 'PIPE_FORMAT_B8G8R8X8_SRGB',
    161: 'PIPE_FORMAT_A8R8G8B8_SRGB',
    162: 'PIPE_FORMAT_X8R8G8B8_SRGB',
    163: 'PIPE_FORMAT_R8G8B8A8_SRGB',
    164: 'PIPE_FORMAT_DXT1_RGB',
    165: 'PIPE_FORMAT_DXT1_RGBA',
    166: 'PIPE_FORMAT_DXT3_RGBA',
    167: 'PIPE_FORMAT_DXT5_RGBA',
    168: 'PIPE_FORMAT_DXT1_SRGB',
    169: 'PIPE_FORMAT_DXT1_SRGBA',
    170: 'PIPE_FORMAT_DXT3_SRGBA',
    171: 'PIPE_FORMAT_DXT5_SRGBA',
    172: 'PIPE_FORMAT_RGTC1_UNORM',
    173: 'PIPE_FORMAT_RGTC1_SNORM',
    174: 'PIPE_FORMAT_RGTC2_UNORM',
    175: 'PIPE_FORMAT_RGTC2_SNORM',
    176: 'PIPE_FORMAT_R8G8_B8G8_UNORM',
    177: 'PIPE_FORMAT_G8R8_G8B8_UNORM',
    178: 'PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM',
    179: 'PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM',
    180: 'PIPE_FORMAT_X6R10_UNORM',
    181: 'PIPE_FORMAT_X6R10X6G10_UNORM',
    182: 'PIPE_FORMAT_X4R12_UNORM',
    183: 'PIPE_FORMAT_X4R12X4G12_UNORM',
    184: 'PIPE_FORMAT_R8SG8SB8UX8U_NORM',
    185: 'PIPE_FORMAT_R5SG5SB6U_NORM',
    186: 'PIPE_FORMAT_A8B8G8R8_UNORM',
    187: 'PIPE_FORMAT_B5G5R5X1_UNORM',
    188: 'PIPE_FORMAT_R9G9B9E5_FLOAT',
    189: 'PIPE_FORMAT_Z32_FLOAT_S8X24_UINT',
    190: 'PIPE_FORMAT_R1_UNORM',
    191: 'PIPE_FORMAT_R10G10B10X2_USCALED',
    192: 'PIPE_FORMAT_R10G10B10X2_SNORM',
    193: 'PIPE_FORMAT_L4A4_UNORM',
    194: 'PIPE_FORMAT_A2R10G10B10_UNORM',
    195: 'PIPE_FORMAT_A2B10G10R10_UNORM',
    196: 'PIPE_FORMAT_R10SG10SB10SA2U_NORM',
    197: 'PIPE_FORMAT_R8G8Bx_SNORM',
    198: 'PIPE_FORMAT_R8G8B8X8_UNORM',
    199: 'PIPE_FORMAT_B4G4R4X4_UNORM',
    200: 'PIPE_FORMAT_X24S8_UINT',
    201: 'PIPE_FORMAT_S8X24_UINT',
    202: 'PIPE_FORMAT_X32_S8X24_UINT',
    203: 'PIPE_FORMAT_R3G3B2_UNORM',
    204: 'PIPE_FORMAT_B2G3R3_UNORM',
    205: 'PIPE_FORMAT_L16A16_UNORM',
    206: 'PIPE_FORMAT_A16_UNORM',
    207: 'PIPE_FORMAT_I16_UNORM',
    208: 'PIPE_FORMAT_LATC1_UNORM',
    209: 'PIPE_FORMAT_LATC1_SNORM',
    210: 'PIPE_FORMAT_LATC2_UNORM',
    211: 'PIPE_FORMAT_LATC2_SNORM',
    212: 'PIPE_FORMAT_A8_SNORM',
    213: 'PIPE_FORMAT_L8_SNORM',
    214: 'PIPE_FORMAT_L8A8_SNORM',
    215: 'PIPE_FORMAT_I8_SNORM',
    216: 'PIPE_FORMAT_A16_SNORM',
    217: 'PIPE_FORMAT_L16_SNORM',
    218: 'PIPE_FORMAT_L16A16_SNORM',
    219: 'PIPE_FORMAT_I16_SNORM',
    220: 'PIPE_FORMAT_A16_FLOAT',
    221: 'PIPE_FORMAT_L16_FLOAT',
    222: 'PIPE_FORMAT_L16A16_FLOAT',
    223: 'PIPE_FORMAT_I16_FLOAT',
    224: 'PIPE_FORMAT_A32_FLOAT',
    225: 'PIPE_FORMAT_L32_FLOAT',
    226: 'PIPE_FORMAT_L32A32_FLOAT',
    227: 'PIPE_FORMAT_I32_FLOAT',
    228: 'PIPE_FORMAT_YV12',
    229: 'PIPE_FORMAT_YV16',
    230: 'PIPE_FORMAT_IYUV',
    231: 'PIPE_FORMAT_NV12',
    232: 'PIPE_FORMAT_NV21',
    233: 'PIPE_FORMAT_NV16',
    234: 'PIPE_FORMAT_NV15',
    235: 'PIPE_FORMAT_NV20',
    236: 'PIPE_FORMAT_Y8_400_UNORM',
    237: 'PIPE_FORMAT_Y8_U8_V8_422_UNORM',
    238: 'PIPE_FORMAT_Y8_U8_V8_444_UNORM',
    239: 'PIPE_FORMAT_Y8_U8_V8_440_UNORM',
    240: 'PIPE_FORMAT_Y16_U16_V16_420_UNORM',
    241: 'PIPE_FORMAT_Y16_U16_V16_422_UNORM',
    242: 'PIPE_FORMAT_Y16_U16V16_422_UNORM',
    243: 'PIPE_FORMAT_Y16_U16_V16_444_UNORM',
    244: 'PIPE_FORMAT_A4R4_UNORM',
    245: 'PIPE_FORMAT_R4A4_UNORM',
    246: 'PIPE_FORMAT_R8A8_UNORM',
    247: 'PIPE_FORMAT_A8R8_UNORM',
    248: 'PIPE_FORMAT_A8_UINT',
    249: 'PIPE_FORMAT_I8_UINT',
    250: 'PIPE_FORMAT_L8_UINT',
    251: 'PIPE_FORMAT_L8A8_UINT',
    252: 'PIPE_FORMAT_A8_SINT',
    253: 'PIPE_FORMAT_I8_SINT',
    254: 'PIPE_FORMAT_L8_SINT',
    255: 'PIPE_FORMAT_L8A8_SINT',
    256: 'PIPE_FORMAT_A16_UINT',
    257: 'PIPE_FORMAT_I16_UINT',
    258: 'PIPE_FORMAT_L16_UINT',
    259: 'PIPE_FORMAT_L16A16_UINT',
    260: 'PIPE_FORMAT_A16_SINT',
    261: 'PIPE_FORMAT_I16_SINT',
    262: 'PIPE_FORMAT_L16_SINT',
    263: 'PIPE_FORMAT_L16A16_SINT',
    264: 'PIPE_FORMAT_A32_UINT',
    265: 'PIPE_FORMAT_I32_UINT',
    266: 'PIPE_FORMAT_L32_UINT',
    267: 'PIPE_FORMAT_L32A32_UINT',
    268: 'PIPE_FORMAT_A32_SINT',
    269: 'PIPE_FORMAT_I32_SINT',
    270: 'PIPE_FORMAT_L32_SINT',
    271: 'PIPE_FORMAT_L32A32_SINT',
    272: 'PIPE_FORMAT_A8R8G8B8_UINT',
    273: 'PIPE_FORMAT_A8B8G8R8_UINT',
    274: 'PIPE_FORMAT_A2R10G10B10_UINT',
    275: 'PIPE_FORMAT_A2B10G10R10_UINT',
    276: 'PIPE_FORMAT_R5G6B5_UINT',
    277: 'PIPE_FORMAT_B5G6R5_UINT',
    278: 'PIPE_FORMAT_R5G5B5A1_UINT',
    279: 'PIPE_FORMAT_B5G5R5A1_UINT',
    280: 'PIPE_FORMAT_A1R5G5B5_UINT',
    281: 'PIPE_FORMAT_A1B5G5R5_UINT',
    282: 'PIPE_FORMAT_R4G4B4A4_UINT',
    283: 'PIPE_FORMAT_B4G4R4A4_UINT',
    284: 'PIPE_FORMAT_A4R4G4B4_UINT',
    285: 'PIPE_FORMAT_A4B4G4R4_UINT',
    286: 'PIPE_FORMAT_R3G3B2_UINT',
    287: 'PIPE_FORMAT_B2G3R3_UINT',
    288: 'PIPE_FORMAT_ETC1_RGB8',
    289: 'PIPE_FORMAT_R8G8_R8B8_UNORM',
    290: 'PIPE_FORMAT_R8B8_R8G8_UNORM',
    291: 'PIPE_FORMAT_G8R8_B8R8_UNORM',
    292: 'PIPE_FORMAT_B8R8_G8R8_UNORM',
    293: 'PIPE_FORMAT_G8B8_G8R8_UNORM',
    294: 'PIPE_FORMAT_B8G8_R8G8_UNORM',
    295: 'PIPE_FORMAT_R8G8B8X8_SNORM',
    296: 'PIPE_FORMAT_R8G8B8X8_SRGB',
    297: 'PIPE_FORMAT_R8G8B8X8_UINT',
    298: 'PIPE_FORMAT_R8G8B8X8_SINT',
    299: 'PIPE_FORMAT_B10G10R10X2_UNORM',
    300: 'PIPE_FORMAT_R16G16B16X16_UNORM',
    301: 'PIPE_FORMAT_R16G16B16X16_SNORM',
    302: 'PIPE_FORMAT_R16G16B16X16_FLOAT',
    303: 'PIPE_FORMAT_R16G16B16X16_UINT',
    304: 'PIPE_FORMAT_R16G16B16X16_SINT',
    305: 'PIPE_FORMAT_R32G32B32X32_FLOAT',
    306: 'PIPE_FORMAT_R32G32B32X32_UINT',
    307: 'PIPE_FORMAT_R32G32B32X32_SINT',
    308: 'PIPE_FORMAT_R8A8_SNORM',
    309: 'PIPE_FORMAT_R16A16_UNORM',
    310: 'PIPE_FORMAT_R16A16_SNORM',
    311: 'PIPE_FORMAT_R16A16_FLOAT',
    312: 'PIPE_FORMAT_R32A32_FLOAT',
    313: 'PIPE_FORMAT_R8A8_UINT',
    314: 'PIPE_FORMAT_R8A8_SINT',
    315: 'PIPE_FORMAT_R16A16_UINT',
    316: 'PIPE_FORMAT_R16A16_SINT',
    317: 'PIPE_FORMAT_R32A32_UINT',
    318: 'PIPE_FORMAT_R32A32_SINT',
    319: 'PIPE_FORMAT_B5G6R5_SRGB',
    320: 'PIPE_FORMAT_BPTC_RGBA_UNORM',
    321: 'PIPE_FORMAT_BPTC_SRGBA',
    322: 'PIPE_FORMAT_BPTC_RGB_FLOAT',
    323: 'PIPE_FORMAT_BPTC_RGB_UFLOAT',
    324: 'PIPE_FORMAT_G8R8_UNORM',
    325: 'PIPE_FORMAT_G8R8_SNORM',
    326: 'PIPE_FORMAT_G16R16_UNORM',
    327: 'PIPE_FORMAT_G16R16_SNORM',
    328: 'PIPE_FORMAT_A8B8G8R8_SNORM',
    329: 'PIPE_FORMAT_X8B8G8R8_SNORM',
    330: 'PIPE_FORMAT_ETC2_RGB8',
    331: 'PIPE_FORMAT_ETC2_SRGB8',
    332: 'PIPE_FORMAT_ETC2_RGB8A1',
    333: 'PIPE_FORMAT_ETC2_SRGB8A1',
    334: 'PIPE_FORMAT_ETC2_RGBA8',
    335: 'PIPE_FORMAT_ETC2_SRGBA8',
    336: 'PIPE_FORMAT_ETC2_R11_UNORM',
    337: 'PIPE_FORMAT_ETC2_R11_SNORM',
    338: 'PIPE_FORMAT_ETC2_RG11_UNORM',
    339: 'PIPE_FORMAT_ETC2_RG11_SNORM',
    340: 'PIPE_FORMAT_ASTC_4x4',
    341: 'PIPE_FORMAT_ASTC_5x4',
    342: 'PIPE_FORMAT_ASTC_5x5',
    343: 'PIPE_FORMAT_ASTC_6x5',
    344: 'PIPE_FORMAT_ASTC_6x6',
    345: 'PIPE_FORMAT_ASTC_8x5',
    346: 'PIPE_FORMAT_ASTC_8x6',
    347: 'PIPE_FORMAT_ASTC_8x8',
    348: 'PIPE_FORMAT_ASTC_10x5',
    349: 'PIPE_FORMAT_ASTC_10x6',
    350: 'PIPE_FORMAT_ASTC_10x8',
    351: 'PIPE_FORMAT_ASTC_10x10',
    352: 'PIPE_FORMAT_ASTC_12x10',
    353: 'PIPE_FORMAT_ASTC_12x12',
    354: 'PIPE_FORMAT_ASTC_4x4_SRGB',
    355: 'PIPE_FORMAT_ASTC_5x4_SRGB',
    356: 'PIPE_FORMAT_ASTC_5x5_SRGB',
    357: 'PIPE_FORMAT_ASTC_6x5_SRGB',
    358: 'PIPE_FORMAT_ASTC_6x6_SRGB',
    359: 'PIPE_FORMAT_ASTC_8x5_SRGB',
    360: 'PIPE_FORMAT_ASTC_8x6_SRGB',
    361: 'PIPE_FORMAT_ASTC_8x8_SRGB',
    362: 'PIPE_FORMAT_ASTC_10x5_SRGB',
    363: 'PIPE_FORMAT_ASTC_10x6_SRGB',
    364: 'PIPE_FORMAT_ASTC_10x8_SRGB',
    365: 'PIPE_FORMAT_ASTC_10x10_SRGB',
    366: 'PIPE_FORMAT_ASTC_12x10_SRGB',
    367: 'PIPE_FORMAT_ASTC_12x12_SRGB',
    368: 'PIPE_FORMAT_ASTC_3x3x3',
    369: 'PIPE_FORMAT_ASTC_4x3x3',
    370: 'PIPE_FORMAT_ASTC_4x4x3',
    371: 'PIPE_FORMAT_ASTC_4x4x4',
    372: 'PIPE_FORMAT_ASTC_5x4x4',
    373: 'PIPE_FORMAT_ASTC_5x5x4',
    374: 'PIPE_FORMAT_ASTC_5x5x5',
    375: 'PIPE_FORMAT_ASTC_6x5x5',
    376: 'PIPE_FORMAT_ASTC_6x6x5',
    377: 'PIPE_FORMAT_ASTC_6x6x6',
    378: 'PIPE_FORMAT_ASTC_3x3x3_SRGB',
    379: 'PIPE_FORMAT_ASTC_4x3x3_SRGB',
    380: 'PIPE_FORMAT_ASTC_4x4x3_SRGB',
    381: 'PIPE_FORMAT_ASTC_4x4x4_SRGB',
    382: 'PIPE_FORMAT_ASTC_5x4x4_SRGB',
    383: 'PIPE_FORMAT_ASTC_5x5x4_SRGB',
    384: 'PIPE_FORMAT_ASTC_5x5x5_SRGB',
    385: 'PIPE_FORMAT_ASTC_6x5x5_SRGB',
    386: 'PIPE_FORMAT_ASTC_6x6x5_SRGB',
    387: 'PIPE_FORMAT_ASTC_6x6x6_SRGB',
    388: 'PIPE_FORMAT_FXT1_RGB',
    389: 'PIPE_FORMAT_FXT1_RGBA',
    390: 'PIPE_FORMAT_P010',
    391: 'PIPE_FORMAT_P012',
    392: 'PIPE_FORMAT_P016',
    393: 'PIPE_FORMAT_P030',
    394: 'PIPE_FORMAT_Y210',
    395: 'PIPE_FORMAT_Y212',
    396: 'PIPE_FORMAT_Y216',
    397: 'PIPE_FORMAT_Y410',
    398: 'PIPE_FORMAT_Y412',
    399: 'PIPE_FORMAT_Y416',
    400: 'PIPE_FORMAT_R10G10B10X2_UNORM',
    401: 'PIPE_FORMAT_A1R5G5B5_UNORM',
    402: 'PIPE_FORMAT_A1B5G5R5_UNORM',
    403: 'PIPE_FORMAT_X1B5G5R5_UNORM',
    404: 'PIPE_FORMAT_R5G5B5A1_UNORM',
    405: 'PIPE_FORMAT_A4R4G4B4_UNORM',
    406: 'PIPE_FORMAT_A4B4G4R4_UNORM',
    407: 'PIPE_FORMAT_G8R8_SINT',
    408: 'PIPE_FORMAT_A8B8G8R8_SINT',
    409: 'PIPE_FORMAT_X8B8G8R8_SINT',
    410: 'PIPE_FORMAT_ATC_RGB',
    411: 'PIPE_FORMAT_ATC_RGBA_EXPLICIT',
    412: 'PIPE_FORMAT_ATC_RGBA_INTERPOLATED',
    413: 'PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8',
    414: 'PIPE_FORMAT_AYUV',
    415: 'PIPE_FORMAT_XYUV',
    416: 'PIPE_FORMAT_R8_G8B8_420_UNORM',
    417: 'PIPE_FORMAT_R8_B8G8_420_UNORM',
    418: 'PIPE_FORMAT_G8_B8R8_420_UNORM',
    419: 'PIPE_FORMAT_R10_G10B10_420_UNORM',
    420: 'PIPE_FORMAT_R10_G10B10_422_UNORM',
    421: 'PIPE_FORMAT_R8_G8_B8_420_UNORM',
    422: 'PIPE_FORMAT_R8_B8_G8_420_UNORM',
    423: 'PIPE_FORMAT_G8_B8_R8_420_UNORM',
    424: 'PIPE_FORMAT_R8_G8B8_422_UNORM',
    425: 'PIPE_FORMAT_R8_B8G8_422_UNORM',
    426: 'PIPE_FORMAT_G8_B8R8_422_UNORM',
    427: 'PIPE_FORMAT_R8_G8_B8_UNORM',
    428: 'PIPE_FORMAT_Y8_UNORM',
    429: 'PIPE_FORMAT_B8G8R8X8_SNORM',
    430: 'PIPE_FORMAT_B8G8R8X8_UINT',
    431: 'PIPE_FORMAT_B8G8R8X8_SINT',
    432: 'PIPE_FORMAT_A8R8G8B8_SNORM',
    433: 'PIPE_FORMAT_A8R8G8B8_SINT',
    434: 'PIPE_FORMAT_X8R8G8B8_SNORM',
    435: 'PIPE_FORMAT_X8R8G8B8_SINT',
    436: 'PIPE_FORMAT_R5G5B5X1_UNORM',
    437: 'PIPE_FORMAT_X1R5G5B5_UNORM',
    438: 'PIPE_FORMAT_R4G4B4X4_UNORM',
    439: 'PIPE_FORMAT_B10G10R10X2_SNORM',
    440: 'PIPE_FORMAT_R5G6B5_SRGB',
    441: 'PIPE_FORMAT_R10G10B10X2_SINT',
    442: 'PIPE_FORMAT_B10G10R10X2_SINT',
    443: 'PIPE_FORMAT_G16R16_SINT',
    444: 'PIPE_FORMAT_COUNT',
}
PIPE_FORMAT_NONE = 0
PIPE_FORMAT_R64_UINT = 1
PIPE_FORMAT_R64G64_UINT = 2
PIPE_FORMAT_R64G64B64_UINT = 3
PIPE_FORMAT_R64G64B64A64_UINT = 4
PIPE_FORMAT_R64_SINT = 5
PIPE_FORMAT_R64G64_SINT = 6
PIPE_FORMAT_R64G64B64_SINT = 7
PIPE_FORMAT_R64G64B64A64_SINT = 8
PIPE_FORMAT_R64_FLOAT = 9
PIPE_FORMAT_R64G64_FLOAT = 10
PIPE_FORMAT_R64G64B64_FLOAT = 11
PIPE_FORMAT_R64G64B64A64_FLOAT = 12
PIPE_FORMAT_R32_FLOAT = 13
PIPE_FORMAT_R32G32_FLOAT = 14
PIPE_FORMAT_R32G32B32_FLOAT = 15
PIPE_FORMAT_R32G32B32A32_FLOAT = 16
PIPE_FORMAT_R32_UNORM = 17
PIPE_FORMAT_R32G32_UNORM = 18
PIPE_FORMAT_R32G32B32_UNORM = 19
PIPE_FORMAT_R32G32B32A32_UNORM = 20
PIPE_FORMAT_R32_USCALED = 21
PIPE_FORMAT_R32G32_USCALED = 22
PIPE_FORMAT_R32G32B32_USCALED = 23
PIPE_FORMAT_R32G32B32A32_USCALED = 24
PIPE_FORMAT_R32_SNORM = 25
PIPE_FORMAT_R32G32_SNORM = 26
PIPE_FORMAT_R32G32B32_SNORM = 27
PIPE_FORMAT_R32G32B32A32_SNORM = 28
PIPE_FORMAT_R32_SSCALED = 29
PIPE_FORMAT_R32G32_SSCALED = 30
PIPE_FORMAT_R32G32B32_SSCALED = 31
PIPE_FORMAT_R32G32B32A32_SSCALED = 32
PIPE_FORMAT_R16_UNORM = 33
PIPE_FORMAT_R16G16_UNORM = 34
PIPE_FORMAT_R16G16B16_UNORM = 35
PIPE_FORMAT_R16G16B16A16_UNORM = 36
PIPE_FORMAT_R16_USCALED = 37
PIPE_FORMAT_R16G16_USCALED = 38
PIPE_FORMAT_R16G16B16_USCALED = 39
PIPE_FORMAT_R16G16B16A16_USCALED = 40
PIPE_FORMAT_R16_SNORM = 41
PIPE_FORMAT_R16G16_SNORM = 42
PIPE_FORMAT_R16G16B16_SNORM = 43
PIPE_FORMAT_R16G16B16A16_SNORM = 44
PIPE_FORMAT_R16_SSCALED = 45
PIPE_FORMAT_R16G16_SSCALED = 46
PIPE_FORMAT_R16G16B16_SSCALED = 47
PIPE_FORMAT_R16G16B16A16_SSCALED = 48
PIPE_FORMAT_R8_UNORM = 49
PIPE_FORMAT_R8G8_UNORM = 50
PIPE_FORMAT_R8G8B8_UNORM = 51
PIPE_FORMAT_B8G8R8_UNORM = 52
PIPE_FORMAT_R8G8B8A8_UNORM = 53
PIPE_FORMAT_B8G8R8A8_UNORM = 54
PIPE_FORMAT_R8_USCALED = 55
PIPE_FORMAT_R8G8_USCALED = 56
PIPE_FORMAT_R8G8B8_USCALED = 57
PIPE_FORMAT_B8G8R8_USCALED = 58
PIPE_FORMAT_R8G8B8A8_USCALED = 59
PIPE_FORMAT_B8G8R8A8_USCALED = 60
PIPE_FORMAT_A8B8G8R8_USCALED = 61
PIPE_FORMAT_R8_SNORM = 62
PIPE_FORMAT_R8G8_SNORM = 63
PIPE_FORMAT_R8G8B8_SNORM = 64
PIPE_FORMAT_B8G8R8_SNORM = 65
PIPE_FORMAT_R8G8B8A8_SNORM = 66
PIPE_FORMAT_B8G8R8A8_SNORM = 67
PIPE_FORMAT_R8_SSCALED = 68
PIPE_FORMAT_R8G8_SSCALED = 69
PIPE_FORMAT_R8G8B8_SSCALED = 70
PIPE_FORMAT_B8G8R8_SSCALED = 71
PIPE_FORMAT_R8G8B8A8_SSCALED = 72
PIPE_FORMAT_B8G8R8A8_SSCALED = 73
PIPE_FORMAT_A8B8G8R8_SSCALED = 74
PIPE_FORMAT_A8R8G8B8_UNORM = 75
PIPE_FORMAT_R32_FIXED = 76
PIPE_FORMAT_R32G32_FIXED = 77
PIPE_FORMAT_R32G32B32_FIXED = 78
PIPE_FORMAT_R32G32B32A32_FIXED = 79
PIPE_FORMAT_R16_FLOAT = 80
PIPE_FORMAT_R16G16_FLOAT = 81
PIPE_FORMAT_R16G16B16_FLOAT = 82
PIPE_FORMAT_R16G16B16A16_FLOAT = 83
PIPE_FORMAT_R8_UINT = 84
PIPE_FORMAT_R8G8_UINT = 85
PIPE_FORMAT_R8G8B8_UINT = 86
PIPE_FORMAT_B8G8R8_UINT = 87
PIPE_FORMAT_R8G8B8A8_UINT = 88
PIPE_FORMAT_B8G8R8A8_UINT = 89
PIPE_FORMAT_R8_SINT = 90
PIPE_FORMAT_R8G8_SINT = 91
PIPE_FORMAT_R8G8B8_SINT = 92
PIPE_FORMAT_B8G8R8_SINT = 93
PIPE_FORMAT_R8G8B8A8_SINT = 94
PIPE_FORMAT_B8G8R8A8_SINT = 95
PIPE_FORMAT_R16_UINT = 96
PIPE_FORMAT_R16G16_UINT = 97
PIPE_FORMAT_R16G16B16_UINT = 98
PIPE_FORMAT_R16G16B16A16_UINT = 99
PIPE_FORMAT_R16_SINT = 100
PIPE_FORMAT_R16G16_SINT = 101
PIPE_FORMAT_R16G16B16_SINT = 102
PIPE_FORMAT_R16G16B16A16_SINT = 103
PIPE_FORMAT_R32_UINT = 104
PIPE_FORMAT_R32G32_UINT = 105
PIPE_FORMAT_R32G32B32_UINT = 106
PIPE_FORMAT_R32G32B32A32_UINT = 107
PIPE_FORMAT_R32_SINT = 108
PIPE_FORMAT_R32G32_SINT = 109
PIPE_FORMAT_R32G32B32_SINT = 110
PIPE_FORMAT_R32G32B32A32_SINT = 111
PIPE_FORMAT_R10G10B10A2_UNORM = 112
PIPE_FORMAT_R10G10B10A2_SNORM = 113
PIPE_FORMAT_R10G10B10A2_USCALED = 114
PIPE_FORMAT_R10G10B10A2_SSCALED = 115
PIPE_FORMAT_B10G10R10A2_UNORM = 116
PIPE_FORMAT_B10G10R10A2_SNORM = 117
PIPE_FORMAT_B10G10R10A2_USCALED = 118
PIPE_FORMAT_B10G10R10A2_SSCALED = 119
PIPE_FORMAT_R11G11B10_FLOAT = 120
PIPE_FORMAT_R10G10B10A2_UINT = 121
PIPE_FORMAT_R10G10B10A2_SINT = 122
PIPE_FORMAT_B10G10R10A2_UINT = 123
PIPE_FORMAT_B10G10R10A2_SINT = 124
PIPE_FORMAT_B8G8R8X8_UNORM = 125
PIPE_FORMAT_X8B8G8R8_UNORM = 126
PIPE_FORMAT_X8R8G8B8_UNORM = 127
PIPE_FORMAT_B5G5R5A1_UNORM = 128
PIPE_FORMAT_R4G4B4A4_UNORM = 129
PIPE_FORMAT_B4G4R4A4_UNORM = 130
PIPE_FORMAT_R5G6B5_UNORM = 131
PIPE_FORMAT_B5G6R5_UNORM = 132
PIPE_FORMAT_L8_UNORM = 133
PIPE_FORMAT_A8_UNORM = 134
PIPE_FORMAT_I8_UNORM = 135
PIPE_FORMAT_L8A8_UNORM = 136
PIPE_FORMAT_L16_UNORM = 137
PIPE_FORMAT_UYVY = 138
PIPE_FORMAT_VYUY = 139
PIPE_FORMAT_YUYV = 140
PIPE_FORMAT_YVYU = 141
PIPE_FORMAT_Z16_UNORM = 142
PIPE_FORMAT_Z16_UNORM_S8_UINT = 143
PIPE_FORMAT_Z32_UNORM = 144
PIPE_FORMAT_Z32_FLOAT = 145
PIPE_FORMAT_Z24_UNORM_S8_UINT = 146
PIPE_FORMAT_S8_UINT_Z24_UNORM = 147
PIPE_FORMAT_Z24X8_UNORM = 148
PIPE_FORMAT_X8Z24_UNORM = 149
PIPE_FORMAT_S8_UINT = 150
PIPE_FORMAT_L8_SRGB = 151
PIPE_FORMAT_R8_SRGB = 152
PIPE_FORMAT_L8A8_SRGB = 153
PIPE_FORMAT_R8G8_SRGB = 154
PIPE_FORMAT_R8G8B8_SRGB = 155
PIPE_FORMAT_B8G8R8_SRGB = 156
PIPE_FORMAT_A8B8G8R8_SRGB = 157
PIPE_FORMAT_X8B8G8R8_SRGB = 158
PIPE_FORMAT_B8G8R8A8_SRGB = 159
PIPE_FORMAT_B8G8R8X8_SRGB = 160
PIPE_FORMAT_A8R8G8B8_SRGB = 161
PIPE_FORMAT_X8R8G8B8_SRGB = 162
PIPE_FORMAT_R8G8B8A8_SRGB = 163
PIPE_FORMAT_DXT1_RGB = 164
PIPE_FORMAT_DXT1_RGBA = 165
PIPE_FORMAT_DXT3_RGBA = 166
PIPE_FORMAT_DXT5_RGBA = 167
PIPE_FORMAT_DXT1_SRGB = 168
PIPE_FORMAT_DXT1_SRGBA = 169
PIPE_FORMAT_DXT3_SRGBA = 170
PIPE_FORMAT_DXT5_SRGBA = 171
PIPE_FORMAT_RGTC1_UNORM = 172
PIPE_FORMAT_RGTC1_SNORM = 173
PIPE_FORMAT_RGTC2_UNORM = 174
PIPE_FORMAT_RGTC2_SNORM = 175
PIPE_FORMAT_R8G8_B8G8_UNORM = 176
PIPE_FORMAT_G8R8_G8B8_UNORM = 177
PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM = 178
PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM = 179
PIPE_FORMAT_X6R10_UNORM = 180
PIPE_FORMAT_X6R10X6G10_UNORM = 181
PIPE_FORMAT_X4R12_UNORM = 182
PIPE_FORMAT_X4R12X4G12_UNORM = 183
PIPE_FORMAT_R8SG8SB8UX8U_NORM = 184
PIPE_FORMAT_R5SG5SB6U_NORM = 185
PIPE_FORMAT_A8B8G8R8_UNORM = 186
PIPE_FORMAT_B5G5R5X1_UNORM = 187
PIPE_FORMAT_R9G9B9E5_FLOAT = 188
PIPE_FORMAT_Z32_FLOAT_S8X24_UINT = 189
PIPE_FORMAT_R1_UNORM = 190
PIPE_FORMAT_R10G10B10X2_USCALED = 191
PIPE_FORMAT_R10G10B10X2_SNORM = 192
PIPE_FORMAT_L4A4_UNORM = 193
PIPE_FORMAT_A2R10G10B10_UNORM = 194
PIPE_FORMAT_A2B10G10R10_UNORM = 195
PIPE_FORMAT_R10SG10SB10SA2U_NORM = 196
PIPE_FORMAT_R8G8Bx_SNORM = 197
PIPE_FORMAT_R8G8B8X8_UNORM = 198
PIPE_FORMAT_B4G4R4X4_UNORM = 199
PIPE_FORMAT_X24S8_UINT = 200
PIPE_FORMAT_S8X24_UINT = 201
PIPE_FORMAT_X32_S8X24_UINT = 202
PIPE_FORMAT_R3G3B2_UNORM = 203
PIPE_FORMAT_B2G3R3_UNORM = 204
PIPE_FORMAT_L16A16_UNORM = 205
PIPE_FORMAT_A16_UNORM = 206
PIPE_FORMAT_I16_UNORM = 207
PIPE_FORMAT_LATC1_UNORM = 208
PIPE_FORMAT_LATC1_SNORM = 209
PIPE_FORMAT_LATC2_UNORM = 210
PIPE_FORMAT_LATC2_SNORM = 211
PIPE_FORMAT_A8_SNORM = 212
PIPE_FORMAT_L8_SNORM = 213
PIPE_FORMAT_L8A8_SNORM = 214
PIPE_FORMAT_I8_SNORM = 215
PIPE_FORMAT_A16_SNORM = 216
PIPE_FORMAT_L16_SNORM = 217
PIPE_FORMAT_L16A16_SNORM = 218
PIPE_FORMAT_I16_SNORM = 219
PIPE_FORMAT_A16_FLOAT = 220
PIPE_FORMAT_L16_FLOAT = 221
PIPE_FORMAT_L16A16_FLOAT = 222
PIPE_FORMAT_I16_FLOAT = 223
PIPE_FORMAT_A32_FLOAT = 224
PIPE_FORMAT_L32_FLOAT = 225
PIPE_FORMAT_L32A32_FLOAT = 226
PIPE_FORMAT_I32_FLOAT = 227
PIPE_FORMAT_YV12 = 228
PIPE_FORMAT_YV16 = 229
PIPE_FORMAT_IYUV = 230
PIPE_FORMAT_NV12 = 231
PIPE_FORMAT_NV21 = 232
PIPE_FORMAT_NV16 = 233
PIPE_FORMAT_NV15 = 234
PIPE_FORMAT_NV20 = 235
PIPE_FORMAT_Y8_400_UNORM = 236
PIPE_FORMAT_Y8_U8_V8_422_UNORM = 237
PIPE_FORMAT_Y8_U8_V8_444_UNORM = 238
PIPE_FORMAT_Y8_U8_V8_440_UNORM = 239
PIPE_FORMAT_Y16_U16_V16_420_UNORM = 240
PIPE_FORMAT_Y16_U16_V16_422_UNORM = 241
PIPE_FORMAT_Y16_U16V16_422_UNORM = 242
PIPE_FORMAT_Y16_U16_V16_444_UNORM = 243
PIPE_FORMAT_A4R4_UNORM = 244
PIPE_FORMAT_R4A4_UNORM = 245
PIPE_FORMAT_R8A8_UNORM = 246
PIPE_FORMAT_A8R8_UNORM = 247
PIPE_FORMAT_A8_UINT = 248
PIPE_FORMAT_I8_UINT = 249
PIPE_FORMAT_L8_UINT = 250
PIPE_FORMAT_L8A8_UINT = 251
PIPE_FORMAT_A8_SINT = 252
PIPE_FORMAT_I8_SINT = 253
PIPE_FORMAT_L8_SINT = 254
PIPE_FORMAT_L8A8_SINT = 255
PIPE_FORMAT_A16_UINT = 256
PIPE_FORMAT_I16_UINT = 257
PIPE_FORMAT_L16_UINT = 258
PIPE_FORMAT_L16A16_UINT = 259
PIPE_FORMAT_A16_SINT = 260
PIPE_FORMAT_I16_SINT = 261
PIPE_FORMAT_L16_SINT = 262
PIPE_FORMAT_L16A16_SINT = 263
PIPE_FORMAT_A32_UINT = 264
PIPE_FORMAT_I32_UINT = 265
PIPE_FORMAT_L32_UINT = 266
PIPE_FORMAT_L32A32_UINT = 267
PIPE_FORMAT_A32_SINT = 268
PIPE_FORMAT_I32_SINT = 269
PIPE_FORMAT_L32_SINT = 270
PIPE_FORMAT_L32A32_SINT = 271
PIPE_FORMAT_A8R8G8B8_UINT = 272
PIPE_FORMAT_A8B8G8R8_UINT = 273
PIPE_FORMAT_A2R10G10B10_UINT = 274
PIPE_FORMAT_A2B10G10R10_UINT = 275
PIPE_FORMAT_R5G6B5_UINT = 276
PIPE_FORMAT_B5G6R5_UINT = 277
PIPE_FORMAT_R5G5B5A1_UINT = 278
PIPE_FORMAT_B5G5R5A1_UINT = 279
PIPE_FORMAT_A1R5G5B5_UINT = 280
PIPE_FORMAT_A1B5G5R5_UINT = 281
PIPE_FORMAT_R4G4B4A4_UINT = 282
PIPE_FORMAT_B4G4R4A4_UINT = 283
PIPE_FORMAT_A4R4G4B4_UINT = 284
PIPE_FORMAT_A4B4G4R4_UINT = 285
PIPE_FORMAT_R3G3B2_UINT = 286
PIPE_FORMAT_B2G3R3_UINT = 287
PIPE_FORMAT_ETC1_RGB8 = 288
PIPE_FORMAT_R8G8_R8B8_UNORM = 289
PIPE_FORMAT_R8B8_R8G8_UNORM = 290
PIPE_FORMAT_G8R8_B8R8_UNORM = 291
PIPE_FORMAT_B8R8_G8R8_UNORM = 292
PIPE_FORMAT_G8B8_G8R8_UNORM = 293
PIPE_FORMAT_B8G8_R8G8_UNORM = 294
PIPE_FORMAT_R8G8B8X8_SNORM = 295
PIPE_FORMAT_R8G8B8X8_SRGB = 296
PIPE_FORMAT_R8G8B8X8_UINT = 297
PIPE_FORMAT_R8G8B8X8_SINT = 298
PIPE_FORMAT_B10G10R10X2_UNORM = 299
PIPE_FORMAT_R16G16B16X16_UNORM = 300
PIPE_FORMAT_R16G16B16X16_SNORM = 301
PIPE_FORMAT_R16G16B16X16_FLOAT = 302
PIPE_FORMAT_R16G16B16X16_UINT = 303
PIPE_FORMAT_R16G16B16X16_SINT = 304
PIPE_FORMAT_R32G32B32X32_FLOAT = 305
PIPE_FORMAT_R32G32B32X32_UINT = 306
PIPE_FORMAT_R32G32B32X32_SINT = 307
PIPE_FORMAT_R8A8_SNORM = 308
PIPE_FORMAT_R16A16_UNORM = 309
PIPE_FORMAT_R16A16_SNORM = 310
PIPE_FORMAT_R16A16_FLOAT = 311
PIPE_FORMAT_R32A32_FLOAT = 312
PIPE_FORMAT_R8A8_UINT = 313
PIPE_FORMAT_R8A8_SINT = 314
PIPE_FORMAT_R16A16_UINT = 315
PIPE_FORMAT_R16A16_SINT = 316
PIPE_FORMAT_R32A32_UINT = 317
PIPE_FORMAT_R32A32_SINT = 318
PIPE_FORMAT_B5G6R5_SRGB = 319
PIPE_FORMAT_BPTC_RGBA_UNORM = 320
PIPE_FORMAT_BPTC_SRGBA = 321
PIPE_FORMAT_BPTC_RGB_FLOAT = 322
PIPE_FORMAT_BPTC_RGB_UFLOAT = 323
PIPE_FORMAT_G8R8_UNORM = 324
PIPE_FORMAT_G8R8_SNORM = 325
PIPE_FORMAT_G16R16_UNORM = 326
PIPE_FORMAT_G16R16_SNORM = 327
PIPE_FORMAT_A8B8G8R8_SNORM = 328
PIPE_FORMAT_X8B8G8R8_SNORM = 329
PIPE_FORMAT_ETC2_RGB8 = 330
PIPE_FORMAT_ETC2_SRGB8 = 331
PIPE_FORMAT_ETC2_RGB8A1 = 332
PIPE_FORMAT_ETC2_SRGB8A1 = 333
PIPE_FORMAT_ETC2_RGBA8 = 334
PIPE_FORMAT_ETC2_SRGBA8 = 335
PIPE_FORMAT_ETC2_R11_UNORM = 336
PIPE_FORMAT_ETC2_R11_SNORM = 337
PIPE_FORMAT_ETC2_RG11_UNORM = 338
PIPE_FORMAT_ETC2_RG11_SNORM = 339
PIPE_FORMAT_ASTC_4x4 = 340
PIPE_FORMAT_ASTC_5x4 = 341
PIPE_FORMAT_ASTC_5x5 = 342
PIPE_FORMAT_ASTC_6x5 = 343
PIPE_FORMAT_ASTC_6x6 = 344
PIPE_FORMAT_ASTC_8x5 = 345
PIPE_FORMAT_ASTC_8x6 = 346
PIPE_FORMAT_ASTC_8x8 = 347
PIPE_FORMAT_ASTC_10x5 = 348
PIPE_FORMAT_ASTC_10x6 = 349
PIPE_FORMAT_ASTC_10x8 = 350
PIPE_FORMAT_ASTC_10x10 = 351
PIPE_FORMAT_ASTC_12x10 = 352
PIPE_FORMAT_ASTC_12x12 = 353
PIPE_FORMAT_ASTC_4x4_SRGB = 354
PIPE_FORMAT_ASTC_5x4_SRGB = 355
PIPE_FORMAT_ASTC_5x5_SRGB = 356
PIPE_FORMAT_ASTC_6x5_SRGB = 357
PIPE_FORMAT_ASTC_6x6_SRGB = 358
PIPE_FORMAT_ASTC_8x5_SRGB = 359
PIPE_FORMAT_ASTC_8x6_SRGB = 360
PIPE_FORMAT_ASTC_8x8_SRGB = 361
PIPE_FORMAT_ASTC_10x5_SRGB = 362
PIPE_FORMAT_ASTC_10x6_SRGB = 363
PIPE_FORMAT_ASTC_10x8_SRGB = 364
PIPE_FORMAT_ASTC_10x10_SRGB = 365
PIPE_FORMAT_ASTC_12x10_SRGB = 366
PIPE_FORMAT_ASTC_12x12_SRGB = 367
PIPE_FORMAT_ASTC_3x3x3 = 368
PIPE_FORMAT_ASTC_4x3x3 = 369
PIPE_FORMAT_ASTC_4x4x3 = 370
PIPE_FORMAT_ASTC_4x4x4 = 371
PIPE_FORMAT_ASTC_5x4x4 = 372
PIPE_FORMAT_ASTC_5x5x4 = 373
PIPE_FORMAT_ASTC_5x5x5 = 374
PIPE_FORMAT_ASTC_6x5x5 = 375
PIPE_FORMAT_ASTC_6x6x5 = 376
PIPE_FORMAT_ASTC_6x6x6 = 377
PIPE_FORMAT_ASTC_3x3x3_SRGB = 378
PIPE_FORMAT_ASTC_4x3x3_SRGB = 379
PIPE_FORMAT_ASTC_4x4x3_SRGB = 380
PIPE_FORMAT_ASTC_4x4x4_SRGB = 381
PIPE_FORMAT_ASTC_5x4x4_SRGB = 382
PIPE_FORMAT_ASTC_5x5x4_SRGB = 383
PIPE_FORMAT_ASTC_5x5x5_SRGB = 384
PIPE_FORMAT_ASTC_6x5x5_SRGB = 385
PIPE_FORMAT_ASTC_6x6x5_SRGB = 386
PIPE_FORMAT_ASTC_6x6x6_SRGB = 387
PIPE_FORMAT_FXT1_RGB = 388
PIPE_FORMAT_FXT1_RGBA = 389
PIPE_FORMAT_P010 = 390
PIPE_FORMAT_P012 = 391
PIPE_FORMAT_P016 = 392
PIPE_FORMAT_P030 = 393
PIPE_FORMAT_Y210 = 394
PIPE_FORMAT_Y212 = 395
PIPE_FORMAT_Y216 = 396
PIPE_FORMAT_Y410 = 397
PIPE_FORMAT_Y412 = 398
PIPE_FORMAT_Y416 = 399
PIPE_FORMAT_R10G10B10X2_UNORM = 400
PIPE_FORMAT_A1R5G5B5_UNORM = 401
PIPE_FORMAT_A1B5G5R5_UNORM = 402
PIPE_FORMAT_X1B5G5R5_UNORM = 403
PIPE_FORMAT_R5G5B5A1_UNORM = 404
PIPE_FORMAT_A4R4G4B4_UNORM = 405
PIPE_FORMAT_A4B4G4R4_UNORM = 406
PIPE_FORMAT_G8R8_SINT = 407
PIPE_FORMAT_A8B8G8R8_SINT = 408
PIPE_FORMAT_X8B8G8R8_SINT = 409
PIPE_FORMAT_ATC_RGB = 410
PIPE_FORMAT_ATC_RGBA_EXPLICIT = 411
PIPE_FORMAT_ATC_RGBA_INTERPOLATED = 412
PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8 = 413
PIPE_FORMAT_AYUV = 414
PIPE_FORMAT_XYUV = 415
PIPE_FORMAT_R8_G8B8_420_UNORM = 416
PIPE_FORMAT_R8_B8G8_420_UNORM = 417
PIPE_FORMAT_G8_B8R8_420_UNORM = 418
PIPE_FORMAT_R10_G10B10_420_UNORM = 419
PIPE_FORMAT_R10_G10B10_422_UNORM = 420
PIPE_FORMAT_R8_G8_B8_420_UNORM = 421
PIPE_FORMAT_R8_B8_G8_420_UNORM = 422
PIPE_FORMAT_G8_B8_R8_420_UNORM = 423
PIPE_FORMAT_R8_G8B8_422_UNORM = 424
PIPE_FORMAT_R8_B8G8_422_UNORM = 425
PIPE_FORMAT_G8_B8R8_422_UNORM = 426
PIPE_FORMAT_R8_G8_B8_UNORM = 427
PIPE_FORMAT_Y8_UNORM = 428
PIPE_FORMAT_B8G8R8X8_SNORM = 429
PIPE_FORMAT_B8G8R8X8_UINT = 430
PIPE_FORMAT_B8G8R8X8_SINT = 431
PIPE_FORMAT_A8R8G8B8_SNORM = 432
PIPE_FORMAT_A8R8G8B8_SINT = 433
PIPE_FORMAT_X8R8G8B8_SNORM = 434
PIPE_FORMAT_X8R8G8B8_SINT = 435
PIPE_FORMAT_R5G5B5X1_UNORM = 436
PIPE_FORMAT_X1R5G5B5_UNORM = 437
PIPE_FORMAT_R4G4B4X4_UNORM = 438
PIPE_FORMAT_B10G10R10X2_SNORM = 439
PIPE_FORMAT_R5G6B5_SRGB = 440
PIPE_FORMAT_R10G10B10X2_SINT = 441
PIPE_FORMAT_B10G10R10X2_SINT = 442
PIPE_FORMAT_G16R16_SINT = 443
PIPE_FORMAT_COUNT = 444
pipe_format = ctypes.c_uint32 # enum
class union_glsl_struct_field_0(Union):
    pass

class struct_glsl_struct_field_0_0(Structure):
    pass

struct_glsl_struct_field_0_0._pack_ = 1 # source:False
struct_glsl_struct_field_0_0._fields_ = [
    ('interpolation', ctypes.c_uint32, 3),
    ('centroid', ctypes.c_uint32, 1),
    ('sample', ctypes.c_uint32, 1),
    ('matrix_layout', ctypes.c_uint32, 2),
    ('patch', ctypes.c_uint32, 1),
    ('precision', ctypes.c_uint32, 2),
    ('memory_read_only', ctypes.c_uint32, 1),
    ('memory_write_only', ctypes.c_uint32, 1),
    ('memory_coherent', ctypes.c_uint32, 1),
    ('memory_volatile', ctypes.c_uint32, 1),
    ('memory_restrict', ctypes.c_uint32, 1),
    ('explicit_xfb_buffer', ctypes.c_uint32, 1),
    ('implicit_sized_array', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint16, 15),
]

union_glsl_struct_field_0._pack_ = 1 # source:False
union_glsl_struct_field_0._anonymous_ = ('_0',)
union_glsl_struct_field_0._fields_ = [
    ('_0', struct_glsl_struct_field_0_0),
    ('flags', ctypes.c_uint32),
]

struct_glsl_struct_field._pack_ = 1 # source:False
struct_glsl_struct_field._anonymous_ = ('_0',)
struct_glsl_struct_field._fields_ = [
    ('type', ctypes.POINTER(struct_glsl_type)),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('location', ctypes.c_int32),
    ('component', ctypes.c_int32),
    ('offset', ctypes.c_int32),
    ('xfb_buffer', ctypes.c_int32),
    ('xfb_stride', ctypes.c_int32),
    ('image_format', pipe_format),
    ('_0', union_glsl_struct_field_0),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

glsl_struct_field = struct_glsl_struct_field
try:
    glsl_type_singleton_init_or_ref = _libraries['FIXME_STUB'].glsl_type_singleton_init_or_ref
    glsl_type_singleton_init_or_ref.restype = None
    glsl_type_singleton_init_or_ref.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_singleton_decref = _libraries['FIXME_STUB'].glsl_type_singleton_decref
    glsl_type_singleton_decref.restype = None
    glsl_type_singleton_decref.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    encode_type_to_blob = _libraries['FIXME_STUB'].encode_type_to_blob
    encode_type_to_blob.restype = None
    encode_type_to_blob.argtypes = [ctypes.POINTER(struct_blob), ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    decode_type_from_blob = _libraries['FIXME_STUB'].decode_type_from_blob
    decode_type_from_blob.restype = ctypes.POINTER(struct_glsl_type)
    decode_type_from_blob.argtypes = [ctypes.POINTER(struct_blob_reader)]
except (AttributeError, RuntimeError):
    pass
glsl_type_size_align_func = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32))
try:
    glsl_base_type_bit_size = _libraries['FIXME_STUB'].glsl_base_type_bit_size
    glsl_base_type_bit_size.restype = ctypes.c_uint32
    glsl_base_type_bit_size.argtypes = [glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_base_type_is_16bit = _libraries['FIXME_STUB'].glsl_base_type_is_16bit
    glsl_base_type_is_16bit.restype = ctypes.c_bool
    glsl_base_type_is_16bit.argtypes = [glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_base_type_is_64bit = _libraries['FIXME_STUB'].glsl_base_type_is_64bit
    glsl_base_type_is_64bit.restype = ctypes.c_bool
    glsl_base_type_is_64bit.argtypes = [glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_base_type_is_integer = _libraries['FIXME_STUB'].glsl_base_type_is_integer
    glsl_base_type_is_integer.restype = ctypes.c_bool
    glsl_base_type_is_integer.argtypes = [glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_base_type_get_bit_size = _libraries['FIXME_STUB'].glsl_base_type_get_bit_size
    glsl_base_type_get_bit_size.restype = ctypes.c_uint32
    glsl_base_type_get_bit_size.argtypes = [glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_unsigned_base_type_of = _libraries['FIXME_STUB'].glsl_unsigned_base_type_of
    glsl_unsigned_base_type_of.restype = glsl_base_type
    glsl_unsigned_base_type_of.argtypes = [glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_signed_base_type_of = _libraries['FIXME_STUB'].glsl_signed_base_type_of
    glsl_signed_base_type_of.restype = glsl_base_type
    glsl_signed_base_type_of.argtypes = [glsl_base_type]
except (AttributeError, RuntimeError):
    pass

# values for enumeration 'glsl_sampler_dim'
glsl_sampler_dim__enumvalues = {
    0: 'GLSL_SAMPLER_DIM_1D',
    1: 'GLSL_SAMPLER_DIM_2D',
    2: 'GLSL_SAMPLER_DIM_3D',
    3: 'GLSL_SAMPLER_DIM_CUBE',
    4: 'GLSL_SAMPLER_DIM_RECT',
    5: 'GLSL_SAMPLER_DIM_BUF',
    6: 'GLSL_SAMPLER_DIM_EXTERNAL',
    7: 'GLSL_SAMPLER_DIM_MS',
    8: 'GLSL_SAMPLER_DIM_SUBPASS',
    9: 'GLSL_SAMPLER_DIM_SUBPASS_MS',
}
GLSL_SAMPLER_DIM_1D = 0
GLSL_SAMPLER_DIM_2D = 1
GLSL_SAMPLER_DIM_3D = 2
GLSL_SAMPLER_DIM_CUBE = 3
GLSL_SAMPLER_DIM_RECT = 4
GLSL_SAMPLER_DIM_BUF = 5
GLSL_SAMPLER_DIM_EXTERNAL = 6
GLSL_SAMPLER_DIM_MS = 7
GLSL_SAMPLER_DIM_SUBPASS = 8
GLSL_SAMPLER_DIM_SUBPASS_MS = 9
glsl_sampler_dim = ctypes.c_uint32 # enum
try:
    glsl_get_sampler_dim_coordinate_components = _libraries['FIXME_STUB'].glsl_get_sampler_dim_coordinate_components
    glsl_get_sampler_dim_coordinate_components.restype = ctypes.c_int32
    glsl_get_sampler_dim_coordinate_components.argtypes = [glsl_sampler_dim]
except (AttributeError, RuntimeError):
    pass

# values for enumeration 'glsl_matrix_layout'
glsl_matrix_layout__enumvalues = {
    0: 'GLSL_MATRIX_LAYOUT_INHERITED',
    1: 'GLSL_MATRIX_LAYOUT_COLUMN_MAJOR',
    2: 'GLSL_MATRIX_LAYOUT_ROW_MAJOR',
}
GLSL_MATRIX_LAYOUT_INHERITED = 0
GLSL_MATRIX_LAYOUT_COLUMN_MAJOR = 1
GLSL_MATRIX_LAYOUT_ROW_MAJOR = 2
glsl_matrix_layout = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_GLSL_PRECISION_NONE'
c__Ea_GLSL_PRECISION_NONE__enumvalues = {
    0: 'GLSL_PRECISION_NONE',
    1: 'GLSL_PRECISION_HIGH',
    2: 'GLSL_PRECISION_MEDIUM',
    3: 'GLSL_PRECISION_LOW',
}
GLSL_PRECISION_NONE = 0
GLSL_PRECISION_HIGH = 1
GLSL_PRECISION_MEDIUM = 2
GLSL_PRECISION_LOW = 3
c__Ea_GLSL_PRECISION_NONE = ctypes.c_uint32 # enum

# values for enumeration 'glsl_cmat_use'
glsl_cmat_use__enumvalues = {
    0: 'GLSL_CMAT_USE_NONE',
    1: 'GLSL_CMAT_USE_A',
    2: 'GLSL_CMAT_USE_B',
    3: 'GLSL_CMAT_USE_ACCUMULATOR',
}
GLSL_CMAT_USE_NONE = 0
GLSL_CMAT_USE_A = 1
GLSL_CMAT_USE_B = 2
GLSL_CMAT_USE_ACCUMULATOR = 3
glsl_cmat_use = ctypes.c_uint32 # enum
try:
    glsl_get_type_name = _libraries['FIXME_STUB'].glsl_get_type_name
    glsl_get_type_name.restype = ctypes.POINTER(ctypes.c_char)
    glsl_get_type_name.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_base_type = _libraries['FIXME_STUB'].glsl_get_base_type
    glsl_get_base_type.restype = glsl_base_type
    glsl_get_base_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_bit_size = _libraries['FIXME_STUB'].glsl_get_bit_size
    glsl_get_bit_size.restype = ctypes.c_uint32
    glsl_get_bit_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_boolean = _libraries['FIXME_STUB'].glsl_type_is_boolean
    glsl_type_is_boolean.restype = ctypes.c_bool
    glsl_type_is_boolean.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_sampler = _libraries['FIXME_STUB'].glsl_type_is_sampler
    glsl_type_is_sampler.restype = ctypes.c_bool
    glsl_type_is_sampler.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_texture = _libraries['FIXME_STUB'].glsl_type_is_texture
    glsl_type_is_texture.restype = ctypes.c_bool
    glsl_type_is_texture.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_image = _libraries['FIXME_STUB'].glsl_type_is_image
    glsl_type_is_image.restype = ctypes.c_bool
    glsl_type_is_image.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_atomic_uint = _libraries['FIXME_STUB'].glsl_type_is_atomic_uint
    glsl_type_is_atomic_uint.restype = ctypes.c_bool
    glsl_type_is_atomic_uint.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_struct = _libraries['FIXME_STUB'].glsl_type_is_struct
    glsl_type_is_struct.restype = ctypes.c_bool
    glsl_type_is_struct.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_interface = _libraries['FIXME_STUB'].glsl_type_is_interface
    glsl_type_is_interface.restype = ctypes.c_bool
    glsl_type_is_interface.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_array = _libraries['FIXME_STUB'].glsl_type_is_array
    glsl_type_is_array.restype = ctypes.c_bool
    glsl_type_is_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_cmat = _libraries['FIXME_STUB'].glsl_type_is_cmat
    glsl_type_is_cmat.restype = ctypes.c_bool
    glsl_type_is_cmat.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_void = _libraries['FIXME_STUB'].glsl_type_is_void
    glsl_type_is_void.restype = ctypes.c_bool
    glsl_type_is_void.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_subroutine = _libraries['FIXME_STUB'].glsl_type_is_subroutine
    glsl_type_is_subroutine.restype = ctypes.c_bool
    glsl_type_is_subroutine.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_error = _libraries['FIXME_STUB'].glsl_type_is_error
    glsl_type_is_error.restype = ctypes.c_bool
    glsl_type_is_error.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_double = _libraries['FIXME_STUB'].glsl_type_is_double
    glsl_type_is_double.restype = ctypes.c_bool
    glsl_type_is_double.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_float = _libraries['FIXME_STUB'].glsl_type_is_float
    glsl_type_is_float.restype = ctypes.c_bool
    glsl_type_is_float.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_numeric = _libraries['FIXME_STUB'].glsl_type_is_numeric
    glsl_type_is_numeric.restype = ctypes.c_bool
    glsl_type_is_numeric.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_integer = _libraries['FIXME_STUB'].glsl_type_is_integer
    glsl_type_is_integer.restype = ctypes.c_bool
    glsl_type_is_integer.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_struct_or_ifc = _libraries['FIXME_STUB'].glsl_type_is_struct_or_ifc
    glsl_type_is_struct_or_ifc.restype = ctypes.c_bool
    glsl_type_is_struct_or_ifc.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_packed = _libraries['FIXME_STUB'].glsl_type_is_packed
    glsl_type_is_packed.restype = ctypes.c_bool
    glsl_type_is_packed.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_16bit = _libraries['FIXME_STUB'].glsl_type_is_16bit
    glsl_type_is_16bit.restype = ctypes.c_bool
    glsl_type_is_16bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_32bit = _libraries['FIXME_STUB'].glsl_type_is_32bit
    glsl_type_is_32bit.restype = ctypes.c_bool
    glsl_type_is_32bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_64bit = _libraries['FIXME_STUB'].glsl_type_is_64bit
    glsl_type_is_64bit.restype = ctypes.c_bool
    glsl_type_is_64bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_integer_16 = _libraries['FIXME_STUB'].glsl_type_is_integer_16
    glsl_type_is_integer_16.restype = ctypes.c_bool
    glsl_type_is_integer_16.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_integer_32 = _libraries['FIXME_STUB'].glsl_type_is_integer_32
    glsl_type_is_integer_32.restype = ctypes.c_bool
    glsl_type_is_integer_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_integer_64 = _libraries['FIXME_STUB'].glsl_type_is_integer_64
    glsl_type_is_integer_64.restype = ctypes.c_bool
    glsl_type_is_integer_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_integer_32_64 = _libraries['FIXME_STUB'].glsl_type_is_integer_32_64
    glsl_type_is_integer_32_64.restype = ctypes.c_bool
    glsl_type_is_integer_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_integer_16_32 = _libraries['FIXME_STUB'].glsl_type_is_integer_16_32
    glsl_type_is_integer_16_32.restype = ctypes.c_bool
    glsl_type_is_integer_16_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_integer_16_32_64 = _libraries['FIXME_STUB'].glsl_type_is_integer_16_32_64
    glsl_type_is_integer_16_32_64.restype = ctypes.c_bool
    glsl_type_is_integer_16_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_float_16 = _libraries['FIXME_STUB'].glsl_type_is_float_16
    glsl_type_is_float_16.restype = ctypes.c_bool
    glsl_type_is_float_16.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_float_16_32 = _libraries['FIXME_STUB'].glsl_type_is_float_16_32
    glsl_type_is_float_16_32.restype = ctypes.c_bool
    glsl_type_is_float_16_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_float_16_32_64 = _libraries['FIXME_STUB'].glsl_type_is_float_16_32_64
    glsl_type_is_float_16_32_64.restype = ctypes.c_bool
    glsl_type_is_float_16_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_int_16_32_64 = _libraries['FIXME_STUB'].glsl_type_is_int_16_32_64
    glsl_type_is_int_16_32_64.restype = ctypes.c_bool
    glsl_type_is_int_16_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_uint_16_32_64 = _libraries['FIXME_STUB'].glsl_type_is_uint_16_32_64
    glsl_type_is_uint_16_32_64.restype = ctypes.c_bool
    glsl_type_is_uint_16_32_64.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_int_16_32 = _libraries['FIXME_STUB'].glsl_type_is_int_16_32
    glsl_type_is_int_16_32.restype = ctypes.c_bool
    glsl_type_is_int_16_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_uint_16_32 = _libraries['FIXME_STUB'].glsl_type_is_uint_16_32
    glsl_type_is_uint_16_32.restype = ctypes.c_bool
    glsl_type_is_uint_16_32.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_unsized_array = _libraries['FIXME_STUB'].glsl_type_is_unsized_array
    glsl_type_is_unsized_array.restype = ctypes.c_bool
    glsl_type_is_unsized_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_array_of_arrays = _libraries['FIXME_STUB'].glsl_type_is_array_of_arrays
    glsl_type_is_array_of_arrays.restype = ctypes.c_bool
    glsl_type_is_array_of_arrays.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_bare_sampler = _libraries['FIXME_STUB'].glsl_type_is_bare_sampler
    glsl_type_is_bare_sampler.restype = ctypes.c_bool
    glsl_type_is_bare_sampler.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_vector = _libraries['FIXME_STUB'].glsl_type_is_vector
    glsl_type_is_vector.restype = ctypes.c_bool
    glsl_type_is_vector.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_scalar = _libraries['FIXME_STUB'].glsl_type_is_scalar
    glsl_type_is_scalar.restype = ctypes.c_bool
    glsl_type_is_scalar.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_vector_or_scalar = _libraries['FIXME_STUB'].glsl_type_is_vector_or_scalar
    glsl_type_is_vector_or_scalar.restype = ctypes.c_bool
    glsl_type_is_vector_or_scalar.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_matrix = _libraries['FIXME_STUB'].glsl_type_is_matrix
    glsl_type_is_matrix.restype = ctypes.c_bool
    glsl_type_is_matrix.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_array_or_matrix = _libraries['FIXME_STUB'].glsl_type_is_array_or_matrix
    glsl_type_is_array_or_matrix.restype = ctypes.c_bool
    glsl_type_is_array_or_matrix.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_dual_slot = _libraries['FIXME_STUB'].glsl_type_is_dual_slot
    glsl_type_is_dual_slot.restype = ctypes.c_bool
    glsl_type_is_dual_slot.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_is_leaf = _libraries['FIXME_STUB'].glsl_type_is_leaf
    glsl_type_is_leaf.restype = ctypes.c_bool
    glsl_type_is_leaf.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_matrix_type_is_row_major = _libraries['FIXME_STUB'].glsl_matrix_type_is_row_major
    glsl_matrix_type_is_row_major.restype = ctypes.c_bool
    glsl_matrix_type_is_row_major.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_sampler_type_is_shadow = _libraries['FIXME_STUB'].glsl_sampler_type_is_shadow
    glsl_sampler_type_is_shadow.restype = ctypes.c_bool
    glsl_sampler_type_is_shadow.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_sampler_type_is_array = _libraries['FIXME_STUB'].glsl_sampler_type_is_array
    glsl_sampler_type_is_array.restype = ctypes.c_bool
    glsl_sampler_type_is_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_struct_type_is_packed = _libraries['FIXME_STUB'].glsl_struct_type_is_packed
    glsl_struct_type_is_packed.restype = ctypes.c_bool
    glsl_struct_type_is_packed.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_bare_type = _libraries['FIXME_STUB'].glsl_get_bare_type
    glsl_get_bare_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_bare_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_scalar_type = _libraries['FIXME_STUB'].glsl_get_scalar_type
    glsl_get_scalar_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_scalar_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_base_glsl_type = _libraries['FIXME_STUB'].glsl_get_base_glsl_type
    glsl_get_base_glsl_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_base_glsl_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_length = _libraries['FIXME_STUB'].glsl_get_length
    glsl_get_length.restype = ctypes.c_uint32
    glsl_get_length.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_vector_elements = _libraries['FIXME_STUB'].glsl_get_vector_elements
    glsl_get_vector_elements.restype = ctypes.c_uint32
    glsl_get_vector_elements.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_components = _libraries['FIXME_STUB'].glsl_get_components
    glsl_get_components.restype = ctypes.c_uint32
    glsl_get_components.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_matrix_columns = _libraries['FIXME_STUB'].glsl_get_matrix_columns
    glsl_get_matrix_columns.restype = ctypes.c_uint32
    glsl_get_matrix_columns.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_wrap_in_arrays = _libraries['FIXME_STUB'].glsl_type_wrap_in_arrays
    glsl_type_wrap_in_arrays.restype = ctypes.POINTER(struct_glsl_type)
    glsl_type_wrap_in_arrays.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_array_size = _libraries['FIXME_STUB'].glsl_array_size
    glsl_array_size.restype = ctypes.c_int32
    glsl_array_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_aoa_size = _libraries['FIXME_STUB'].glsl_get_aoa_size
    glsl_get_aoa_size.restype = ctypes.c_uint32
    glsl_get_aoa_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_array_element = _libraries['FIXME_STUB'].glsl_get_array_element
    glsl_get_array_element.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_array_element.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_without_array = _libraries['FIXME_STUB'].glsl_without_array
    glsl_without_array.restype = ctypes.POINTER(struct_glsl_type)
    glsl_without_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_without_array_or_matrix = _libraries['FIXME_STUB'].glsl_without_array_or_matrix
    glsl_without_array_or_matrix.restype = ctypes.POINTER(struct_glsl_type)
    glsl_without_array_or_matrix.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_cmat_element = _libraries['FIXME_STUB'].glsl_get_cmat_element
    glsl_get_cmat_element.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_cmat_element.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_cmat_description = _libraries['FIXME_STUB'].glsl_get_cmat_description
    glsl_get_cmat_description.restype = ctypes.POINTER(struct_glsl_cmat_description)
    glsl_get_cmat_description.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_atomic_size = _libraries['FIXME_STUB'].glsl_atomic_size
    glsl_atomic_size.restype = ctypes.c_uint32
    glsl_atomic_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_contains_32bit = _libraries['FIXME_STUB'].glsl_type_contains_32bit
    glsl_type_contains_32bit.restype = ctypes.c_bool
    glsl_type_contains_32bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_contains_64bit = _libraries['FIXME_STUB'].glsl_type_contains_64bit
    glsl_type_contains_64bit.restype = ctypes.c_bool
    glsl_type_contains_64bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_contains_image = _libraries['FIXME_STUB'].glsl_type_contains_image
    glsl_type_contains_image.restype = ctypes.c_bool
    glsl_type_contains_image.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_contains_atomic = _libraries['FIXME_STUB'].glsl_contains_atomic
    glsl_contains_atomic.restype = ctypes.c_bool
    glsl_contains_atomic.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_contains_double = _libraries['FIXME_STUB'].glsl_contains_double
    glsl_contains_double.restype = ctypes.c_bool
    glsl_contains_double.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_contains_integer = _libraries['FIXME_STUB'].glsl_contains_integer
    glsl_contains_integer.restype = ctypes.c_bool
    glsl_contains_integer.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_contains_opaque = _libraries['FIXME_STUB'].glsl_contains_opaque
    glsl_contains_opaque.restype = ctypes.c_bool
    glsl_contains_opaque.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_contains_sampler = _libraries['FIXME_STUB'].glsl_contains_sampler
    glsl_contains_sampler.restype = ctypes.c_bool
    glsl_contains_sampler.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_contains_array = _libraries['FIXME_STUB'].glsl_contains_array
    glsl_contains_array.restype = ctypes.c_bool
    glsl_contains_array.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_contains_subroutine = _libraries['FIXME_STUB'].glsl_contains_subroutine
    glsl_contains_subroutine.restype = ctypes.c_bool
    glsl_contains_subroutine.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_sampler_dim = _libraries['FIXME_STUB'].glsl_get_sampler_dim
    glsl_get_sampler_dim.restype = glsl_sampler_dim
    glsl_get_sampler_dim.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_sampler_result_type = _libraries['FIXME_STUB'].glsl_get_sampler_result_type
    glsl_get_sampler_result_type.restype = glsl_base_type
    glsl_get_sampler_result_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_sampler_coordinate_components = _libraries['FIXME_STUB'].glsl_get_sampler_coordinate_components
    glsl_get_sampler_coordinate_components.restype = ctypes.c_int32
    glsl_get_sampler_coordinate_components.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_compare_no_precision = _libraries['FIXME_STUB'].glsl_type_compare_no_precision
    glsl_type_compare_no_precision.restype = ctypes.c_bool
    glsl_type_compare_no_precision.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_record_compare = _libraries['FIXME_STUB'].glsl_record_compare
    glsl_record_compare.restype = ctypes.c_bool
    glsl_record_compare.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(struct_glsl_type), ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_struct_field = _libraries['FIXME_STUB'].glsl_get_struct_field
    glsl_get_struct_field.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_struct_field.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_struct_field_data = _libraries['FIXME_STUB'].glsl_get_struct_field_data
    glsl_get_struct_field_data.restype = ctypes.POINTER(struct_glsl_struct_field)
    glsl_get_struct_field_data.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_struct_location_offset = _libraries['FIXME_STUB'].glsl_get_struct_location_offset
    glsl_get_struct_location_offset.restype = ctypes.c_uint32
    glsl_get_struct_location_offset.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_field_index = _libraries['FIXME_STUB'].glsl_get_field_index
    glsl_get_field_index.restype = ctypes.c_int32
    glsl_get_field_index.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_field_type = _libraries['FIXME_STUB'].glsl_get_field_type
    glsl_get_field_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_field_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_struct_field_offset = _libraries['FIXME_STUB'].glsl_get_struct_field_offset
    glsl_get_struct_field_offset.restype = ctypes.c_int32
    glsl_get_struct_field_offset.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_struct_elem_name = _libraries['FIXME_STUB'].glsl_get_struct_elem_name
    glsl_get_struct_elem_name.restype = ctypes.POINTER(ctypes.c_char)
    glsl_get_struct_elem_name.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_void_type = _libraries['FIXME_STUB'].glsl_void_type
    glsl_void_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_void_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_float_type = _libraries['FIXME_STUB'].glsl_float_type
    glsl_float_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_float_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_float16_t_type = _libraries['FIXME_STUB'].glsl_float16_t_type
    glsl_float16_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_float16_t_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_double_type = _libraries['FIXME_STUB'].glsl_double_type
    glsl_double_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_double_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_vec2_type = _libraries['FIXME_STUB'].glsl_vec2_type
    glsl_vec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_vec2_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_dvec2_type = _libraries['FIXME_STUB'].glsl_dvec2_type
    glsl_dvec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_dvec2_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uvec2_type = _libraries['FIXME_STUB'].glsl_uvec2_type
    glsl_uvec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uvec2_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_ivec2_type = _libraries['FIXME_STUB'].glsl_ivec2_type
    glsl_ivec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_ivec2_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_bvec2_type = _libraries['FIXME_STUB'].glsl_bvec2_type
    glsl_bvec2_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bvec2_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_vec4_type = _libraries['FIXME_STUB'].glsl_vec4_type
    glsl_vec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_vec4_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_dvec4_type = _libraries['FIXME_STUB'].glsl_dvec4_type
    glsl_dvec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_dvec4_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uvec4_type = _libraries['FIXME_STUB'].glsl_uvec4_type
    glsl_uvec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uvec4_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_ivec4_type = _libraries['FIXME_STUB'].glsl_ivec4_type
    glsl_ivec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_ivec4_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_bvec4_type = _libraries['FIXME_STUB'].glsl_bvec4_type
    glsl_bvec4_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bvec4_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_int_type = _libraries['FIXME_STUB'].glsl_int_type
    glsl_int_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uint_type = _libraries['FIXME_STUB'].glsl_uint_type
    glsl_uint_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_int64_t_type = _libraries['FIXME_STUB'].glsl_int64_t_type
    glsl_int64_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int64_t_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uint64_t_type = _libraries['FIXME_STUB'].glsl_uint64_t_type
    glsl_uint64_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint64_t_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_int16_t_type = _libraries['FIXME_STUB'].glsl_int16_t_type
    glsl_int16_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int16_t_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uint16_t_type = _libraries['FIXME_STUB'].glsl_uint16_t_type
    glsl_uint16_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint16_t_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_int8_t_type = _libraries['FIXME_STUB'].glsl_int8_t_type
    glsl_int8_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int8_t_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uint8_t_type = _libraries['FIXME_STUB'].glsl_uint8_t_type
    glsl_uint8_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint8_t_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_bool_type = _libraries['FIXME_STUB'].glsl_bool_type
    glsl_bool_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bool_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_atomic_uint_type = _libraries['FIXME_STUB'].glsl_atomic_uint_type
    glsl_atomic_uint_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_atomic_uint_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_floatN_t_type = _libraries['FIXME_STUB'].glsl_floatN_t_type
    glsl_floatN_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_floatN_t_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_intN_t_type = _libraries['FIXME_STUB'].glsl_intN_t_type
    glsl_intN_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_intN_t_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uintN_t_type = _libraries['FIXME_STUB'].glsl_uintN_t_type
    glsl_uintN_t_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uintN_t_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_vec_type = _libraries['FIXME_STUB'].glsl_vec_type
    glsl_vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_vec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_f16vec_type = _libraries['FIXME_STUB'].glsl_f16vec_type
    glsl_f16vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_f16vec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_dvec_type = _libraries['FIXME_STUB'].glsl_dvec_type
    glsl_dvec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_dvec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_ivec_type = _libraries['FIXME_STUB'].glsl_ivec_type
    glsl_ivec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_ivec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uvec_type = _libraries['FIXME_STUB'].glsl_uvec_type
    glsl_uvec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uvec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_bvec_type = _libraries['FIXME_STUB'].glsl_bvec_type
    glsl_bvec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bvec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_i64vec_type = _libraries['FIXME_STUB'].glsl_i64vec_type
    glsl_i64vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_i64vec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_u64vec_type = _libraries['FIXME_STUB'].glsl_u64vec_type
    glsl_u64vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_u64vec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_i16vec_type = _libraries['FIXME_STUB'].glsl_i16vec_type
    glsl_i16vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_i16vec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_u16vec_type = _libraries['FIXME_STUB'].glsl_u16vec_type
    glsl_u16vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_u16vec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_i8vec_type = _libraries['FIXME_STUB'].glsl_i8vec_type
    glsl_i8vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_i8vec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_u8vec_type = _libraries['FIXME_STUB'].glsl_u8vec_type
    glsl_u8vec_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_u8vec_type.argtypes = [ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_simple_explicit_type = _libraries['FIXME_STUB'].glsl_simple_explicit_type
    glsl_simple_explicit_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_simple_explicit_type.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_simple_type = _libraries['FIXME_STUB'].glsl_simple_type
    glsl_simple_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_simple_type.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_sampler_type = _libraries['FIXME_STUB'].glsl_sampler_type
    glsl_sampler_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_sampler_type.argtypes = [glsl_sampler_dim, ctypes.c_bool, ctypes.c_bool, glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_bare_sampler_type = _libraries['FIXME_STUB'].glsl_bare_sampler_type
    glsl_bare_sampler_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bare_sampler_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_bare_shadow_sampler_type = _libraries['FIXME_STUB'].glsl_bare_shadow_sampler_type
    glsl_bare_shadow_sampler_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_bare_shadow_sampler_type.argtypes = []
except (AttributeError, RuntimeError):
    pass
try:
    glsl_texture_type = _libraries['FIXME_STUB'].glsl_texture_type
    glsl_texture_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_texture_type.argtypes = [glsl_sampler_dim, ctypes.c_bool, glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_image_type = _libraries['FIXME_STUB'].glsl_image_type
    glsl_image_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_image_type.argtypes = [glsl_sampler_dim, ctypes.c_bool, glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_array_type = _libraries['FIXME_STUB'].glsl_array_type
    glsl_array_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_array_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_cmat_type = _libraries['FIXME_STUB'].glsl_cmat_type
    glsl_cmat_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_cmat_type.argtypes = [ctypes.POINTER(struct_glsl_cmat_description)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_struct_type_with_explicit_alignment = _libraries['FIXME_STUB'].glsl_struct_type_with_explicit_alignment
    glsl_struct_type_with_explicit_alignment.restype = ctypes.POINTER(struct_glsl_type)
    glsl_struct_type_with_explicit_alignment.argtypes = [ctypes.POINTER(struct_glsl_struct_field), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_bool, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_struct_type = _libraries['FIXME_STUB'].glsl_struct_type
    glsl_struct_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_struct_type.argtypes = [ctypes.POINTER(struct_glsl_struct_field), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass

# values for enumeration 'glsl_interface_packing'
glsl_interface_packing__enumvalues = {
    0: 'GLSL_INTERFACE_PACKING_STD140',
    1: 'GLSL_INTERFACE_PACKING_SHARED',
    2: 'GLSL_INTERFACE_PACKING_PACKED',
    3: 'GLSL_INTERFACE_PACKING_STD430',
}
GLSL_INTERFACE_PACKING_STD140 = 0
GLSL_INTERFACE_PACKING_SHARED = 1
GLSL_INTERFACE_PACKING_PACKED = 2
GLSL_INTERFACE_PACKING_STD430 = 3
glsl_interface_packing = ctypes.c_uint32 # enum
try:
    glsl_interface_type = _libraries['FIXME_STUB'].glsl_interface_type
    glsl_interface_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_interface_type.argtypes = [ctypes.POINTER(struct_glsl_struct_field), ctypes.c_uint32, glsl_interface_packing, ctypes.c_bool, ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_subroutine_type = _libraries['FIXME_STUB'].glsl_subroutine_type
    glsl_subroutine_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_subroutine_type.argtypes = [ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_row_type = _libraries['FIXME_STUB'].glsl_get_row_type
    glsl_get_row_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_row_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_column_type = _libraries['FIXME_STUB'].glsl_get_column_type
    glsl_get_column_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_column_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_explicit_type_for_size_align = _libraries['FIXME_STUB'].glsl_get_explicit_type_for_size_align
    glsl_get_explicit_type_for_size_align.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_explicit_type_for_size_align.argtypes = [ctypes.POINTER(struct_glsl_type), glsl_type_size_align_func, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_replace_vec3_with_vec4 = _libraries['FIXME_STUB'].glsl_type_replace_vec3_with_vec4
    glsl_type_replace_vec3_with_vec4.restype = ctypes.POINTER(struct_glsl_type)
    glsl_type_replace_vec3_with_vec4.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_float16_type = _libraries['FIXME_STUB'].glsl_float16_type
    glsl_float16_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_float16_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_int16_type = _libraries['FIXME_STUB'].glsl_int16_type
    glsl_int16_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_int16_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_uint16_type = _libraries['FIXME_STUB'].glsl_uint16_type
    glsl_uint16_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_uint16_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_to_16bit = _libraries['FIXME_STUB'].glsl_type_to_16bit
    glsl_type_to_16bit.restype = ctypes.POINTER(struct_glsl_type)
    glsl_type_to_16bit.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_scalar_type = _libraries['FIXME_STUB'].glsl_scalar_type
    glsl_scalar_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_scalar_type.argtypes = [glsl_base_type]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_vector_type = _libraries['FIXME_STUB'].glsl_vector_type
    glsl_vector_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_vector_type.argtypes = [glsl_base_type, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_matrix_type = _libraries['FIXME_STUB'].glsl_matrix_type
    glsl_matrix_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_matrix_type.argtypes = [glsl_base_type, ctypes.c_uint32, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_explicit_matrix_type = _libraries['FIXME_STUB'].glsl_explicit_matrix_type
    glsl_explicit_matrix_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_explicit_matrix_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32, ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_transposed_type = _libraries['FIXME_STUB'].glsl_transposed_type
    glsl_transposed_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_transposed_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_texture_type_to_sampler = _libraries['FIXME_STUB'].glsl_texture_type_to_sampler
    glsl_texture_type_to_sampler.restype = ctypes.POINTER(struct_glsl_type)
    glsl_texture_type_to_sampler.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_sampler_type_to_texture = _libraries['FIXME_STUB'].glsl_sampler_type_to_texture
    glsl_sampler_type_to_texture.restype = ctypes.POINTER(struct_glsl_type)
    glsl_sampler_type_to_texture.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_replace_vector_type = _libraries['FIXME_STUB'].glsl_replace_vector_type
    glsl_replace_vector_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_replace_vector_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_channel_type = _libraries['FIXME_STUB'].glsl_channel_type
    glsl_channel_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_channel_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_mul_type = _libraries['FIXME_STUB'].glsl_get_mul_type
    glsl_get_mul_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_mul_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_get_sampler_count = _libraries['FIXME_STUB'].glsl_type_get_sampler_count
    glsl_type_get_sampler_count.restype = ctypes.c_uint32
    glsl_type_get_sampler_count.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_get_texture_count = _libraries['FIXME_STUB'].glsl_type_get_texture_count
    glsl_type_get_texture_count.restype = ctypes.c_uint32
    glsl_type_get_texture_count.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_get_image_count = _libraries['FIXME_STUB'].glsl_type_get_image_count
    glsl_type_get_image_count.restype = ctypes.c_uint32
    glsl_type_get_image_count.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_count_vec4_slots = _libraries['FIXME_STUB'].glsl_count_vec4_slots
    glsl_count_vec4_slots.restype = ctypes.c_uint32
    glsl_count_vec4_slots.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool, ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_count_dword_slots = _libraries['FIXME_STUB'].glsl_count_dword_slots
    glsl_count_dword_slots.restype = ctypes.c_uint32
    glsl_count_dword_slots.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_component_slots = _libraries['FIXME_STUB'].glsl_get_component_slots
    glsl_get_component_slots.restype = ctypes.c_uint32
    glsl_get_component_slots.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_component_slots_aligned = _libraries['FIXME_STUB'].glsl_get_component_slots_aligned
    glsl_get_component_slots_aligned.restype = ctypes.c_uint32
    glsl_get_component_slots_aligned.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_varying_count = _libraries['FIXME_STUB'].glsl_varying_count
    glsl_varying_count.restype = ctypes.c_uint32
    glsl_varying_count.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_type_uniform_locations = _libraries['FIXME_STUB'].glsl_type_uniform_locations
    glsl_type_uniform_locations.restype = ctypes.c_uint32
    glsl_type_uniform_locations.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_count_attribute_slots = _libraries['FIXME_STUB'].glsl_count_attribute_slots
    glsl_count_attribute_slots.restype = ctypes.c_uint32
    glsl_count_attribute_slots.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_cl_size = _libraries['FIXME_STUB'].glsl_get_cl_size
    glsl_get_cl_size.restype = ctypes.c_uint32
    glsl_get_cl_size.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_cl_alignment = _libraries['FIXME_STUB'].glsl_get_cl_alignment
    glsl_get_cl_alignment.restype = ctypes.c_uint32
    glsl_get_cl_alignment.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_cl_type_size_align = _libraries['FIXME_STUB'].glsl_get_cl_type_size_align
    glsl_get_cl_type_size_align.restype = None
    glsl_get_cl_type_size_align.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_internal_ifc_packing = _libraries['FIXME_STUB'].glsl_get_internal_ifc_packing
    glsl_get_internal_ifc_packing.restype = glsl_interface_packing
    glsl_get_internal_ifc_packing.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_ifc_packing = _libraries['FIXME_STUB'].glsl_get_ifc_packing
    glsl_get_ifc_packing.restype = glsl_interface_packing
    glsl_get_ifc_packing.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_std140_base_alignment = _libraries['FIXME_STUB'].glsl_get_std140_base_alignment
    glsl_get_std140_base_alignment.restype = ctypes.c_uint32
    glsl_get_std140_base_alignment.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_std140_size = _libraries['FIXME_STUB'].glsl_get_std140_size
    glsl_get_std140_size.restype = ctypes.c_uint32
    glsl_get_std140_size.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_std430_array_stride = _libraries['FIXME_STUB'].glsl_get_std430_array_stride
    glsl_get_std430_array_stride.restype = ctypes.c_uint32
    glsl_get_std430_array_stride.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_std430_base_alignment = _libraries['FIXME_STUB'].glsl_get_std430_base_alignment
    glsl_get_std430_base_alignment.restype = ctypes.c_uint32
    glsl_get_std430_base_alignment.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_std430_size = _libraries['FIXME_STUB'].glsl_get_std430_size
    glsl_get_std430_size.restype = ctypes.c_uint32
    glsl_get_std430_size.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_explicit_size = _libraries['FIXME_STUB'].glsl_get_explicit_size
    glsl_get_explicit_size.restype = ctypes.c_uint32
    glsl_get_explicit_size.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_explicit_stride = _libraries['FIXME_STUB'].glsl_get_explicit_stride
    glsl_get_explicit_stride.restype = ctypes.c_uint32
    glsl_get_explicit_stride.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_explicit_alignment = _libraries['FIXME_STUB'].glsl_get_explicit_alignment
    glsl_get_explicit_alignment.restype = ctypes.c_uint32
    glsl_get_explicit_alignment.argtypes = [ctypes.POINTER(struct_glsl_type)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_explicit_std140_type = _libraries['FIXME_STUB'].glsl_get_explicit_std140_type
    glsl_get_explicit_std140_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_explicit_std140_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_explicit_std430_type = _libraries['FIXME_STUB'].glsl_get_explicit_std430_type
    glsl_get_explicit_std430_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_explicit_std430_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_explicit_interface_type = _libraries['FIXME_STUB'].glsl_get_explicit_interface_type
    glsl_get_explicit_interface_type.restype = ctypes.POINTER(struct_glsl_type)
    glsl_get_explicit_interface_type.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.c_bool]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_size_align_handle_array_and_structs = _libraries['FIXME_STUB'].glsl_size_align_handle_array_and_structs
    glsl_size_align_handle_array_and_structs.restype = None
    glsl_size_align_handle_array_and_structs.argtypes = [ctypes.POINTER(struct_glsl_type), glsl_type_size_align_func, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_natural_size_align_bytes = _libraries['FIXME_STUB'].glsl_get_natural_size_align_bytes
    glsl_get_natural_size_align_bytes.restype = None
    glsl_get_natural_size_align_bytes.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_word_size_align_bytes = _libraries['FIXME_STUB'].glsl_get_word_size_align_bytes
    glsl_get_word_size_align_bytes.restype = None
    glsl_get_word_size_align_bytes.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except (AttributeError, RuntimeError):
    pass
try:
    glsl_get_vec4_size_align_bytes = _libraries['FIXME_STUB'].glsl_get_vec4_size_align_bytes
    glsl_get_vec4_size_align_bytes.restype = None
    glsl_get_vec4_size_align_bytes.argtypes = [ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_context = _libraries['FIXME_STUB'].ralloc_context
    ralloc_context.restype = ctypes.POINTER(None)
    ralloc_context.argtypes = [ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_size = _libraries['FIXME_STUB'].ralloc_size
    ralloc_size.restype = ctypes.POINTER(None)
    ralloc_size.argtypes = [ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    rzalloc_size = _libraries['FIXME_STUB'].rzalloc_size
    rzalloc_size.restype = ctypes.POINTER(None)
    rzalloc_size.argtypes = [ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    reralloc_size = _libraries['FIXME_STUB'].reralloc_size
    reralloc_size.restype = ctypes.POINTER(None)
    reralloc_size.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    rerzalloc_size = _libraries['FIXME_STUB'].rerzalloc_size
    rerzalloc_size.restype = ctypes.POINTER(None)
    rerzalloc_size.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, size_t]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_array_size = _libraries['FIXME_STUB'].ralloc_array_size
    ralloc_array_size.restype = ctypes.POINTER(None)
    ralloc_array_size.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    rzalloc_array_size = _libraries['FIXME_STUB'].rzalloc_array_size
    rzalloc_array_size.restype = ctypes.POINTER(None)
    rzalloc_array_size.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    reralloc_array_size = _libraries['FIXME_STUB'].reralloc_array_size
    reralloc_array_size.restype = ctypes.POINTER(None)
    reralloc_array_size.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    rerzalloc_array_size = _libraries['FIXME_STUB'].rerzalloc_array_size
    rerzalloc_array_size.restype = ctypes.POINTER(None)
    rerzalloc_array_size.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t, ctypes.c_uint32, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_free = _libraries['FIXME_STUB'].ralloc_free
    ralloc_free.restype = None
    ralloc_free.argtypes = [ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_steal = _libraries['FIXME_STUB'].ralloc_steal
    ralloc_steal.restype = None
    ralloc_steal.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_adopt = _libraries['FIXME_STUB'].ralloc_adopt
    ralloc_adopt.restype = None
    ralloc_adopt.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_parent = _libraries['FIXME_STUB'].ralloc_parent
    ralloc_parent.restype = ctypes.POINTER(None)
    ralloc_parent.argtypes = [ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_set_destructor = _libraries['FIXME_STUB'].ralloc_set_destructor
    ralloc_set_destructor.restype = None
    ralloc_set_destructor.argtypes = [ctypes.POINTER(None), ctypes.CFUNCTYPE(None, ctypes.POINTER(None))]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_memdup = _libraries['FIXME_STUB'].ralloc_memdup
    ralloc_memdup.restype = ctypes.POINTER(None)
    ralloc_memdup.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_strdup = _libraries['FIXME_STUB'].ralloc_strdup
    ralloc_strdup.restype = ctypes.POINTER(ctypes.c_char)
    ralloc_strdup.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_strndup = _libraries['FIXME_STUB'].ralloc_strndup
    ralloc_strndup.restype = ctypes.POINTER(ctypes.c_char)
    ralloc_strndup.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_strcat = _libraries['FIXME_STUB'].ralloc_strcat
    ralloc_strcat.restype = ctypes.c_bool
    ralloc_strcat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_strncat = _libraries['FIXME_STUB'].ralloc_strncat
    ralloc_strncat.restype = ctypes.c_bool
    ralloc_strncat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), size_t]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_str_append = _libraries['FIXME_STUB'].ralloc_str_append
    ralloc_str_append.restype = ctypes.c_bool
    ralloc_str_append.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), size_t, size_t]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_asprintf = _libraries['FIXME_STUB'].ralloc_asprintf
    ralloc_asprintf.restype = ctypes.POINTER(ctypes.c_char)
    ralloc_asprintf.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
class struct___va_list_tag(Structure):
    pass

struct___va_list_tag._pack_ = 1 # source:False
struct___va_list_tag._fields_ = [
    ('gp_offset', ctypes.c_uint32),
    ('fp_offset', ctypes.c_uint32),
    ('overflow_arg_area', ctypes.POINTER(None)),
    ('reg_save_area', ctypes.POINTER(None)),
]

va_list = struct___va_list_tag * 1
try:
    ralloc_vasprintf = _libraries['FIXME_STUB'].ralloc_vasprintf
    ralloc_vasprintf.restype = ctypes.POINTER(ctypes.c_char)
    ralloc_vasprintf.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char), va_list]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_asprintf_rewrite_tail = _libraries['FIXME_STUB'].ralloc_asprintf_rewrite_tail
    ralloc_asprintf_rewrite_tail.restype = ctypes.c_bool
    ralloc_asprintf_rewrite_tail.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_vasprintf_rewrite_tail = _libraries['FIXME_STUB'].ralloc_vasprintf_rewrite_tail
    ralloc_vasprintf_rewrite_tail.restype = ctypes.c_bool
    ralloc_vasprintf_rewrite_tail.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), va_list]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_asprintf_append = _libraries['FIXME_STUB'].ralloc_asprintf_append
    ralloc_asprintf_append.restype = ctypes.c_bool
    ralloc_asprintf_append.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_vasprintf_append = _libraries['FIXME_STUB'].ralloc_vasprintf_append
    ralloc_vasprintf_append.restype = ctypes.c_bool
    ralloc_vasprintf_append.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), va_list]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_total_size = _libraries['FIXME_STUB'].ralloc_total_size
    ralloc_total_size.restype = size_t
    ralloc_total_size.argtypes = [ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
class struct_gc_ctx(Structure):
    pass

gc_ctx = struct_gc_ctx
try:
    gc_context = _libraries['FIXME_STUB'].gc_context
    gc_context.restype = ctypes.POINTER(struct_gc_ctx)
    gc_context.argtypes = [ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    gc_alloc_size = _libraries['FIXME_STUB'].gc_alloc_size
    gc_alloc_size.restype = ctypes.POINTER(None)
    gc_alloc_size.argtypes = [ctypes.POINTER(struct_gc_ctx), size_t, size_t]
except (AttributeError, RuntimeError):
    pass
try:
    gc_zalloc_size = _libraries['FIXME_STUB'].gc_zalloc_size
    gc_zalloc_size.restype = ctypes.POINTER(None)
    gc_zalloc_size.argtypes = [ctypes.POINTER(struct_gc_ctx), size_t, size_t]
except (AttributeError, RuntimeError):
    pass
try:
    gc_free = _libraries['FIXME_STUB'].gc_free
    gc_free.restype = None
    gc_free.argtypes = [ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    gc_get_context = _libraries['FIXME_STUB'].gc_get_context
    gc_get_context.restype = ctypes.POINTER(struct_gc_ctx)
    gc_get_context.argtypes = [ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    gc_sweep_start = _libraries['FIXME_STUB'].gc_sweep_start
    gc_sweep_start.restype = None
    gc_sweep_start.argtypes = [ctypes.POINTER(struct_gc_ctx)]
except (AttributeError, RuntimeError):
    pass
try:
    gc_mark_live = _libraries['FIXME_STUB'].gc_mark_live
    gc_mark_live.restype = None
    gc_mark_live.argtypes = [ctypes.POINTER(struct_gc_ctx), ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    gc_sweep_end = _libraries['FIXME_STUB'].gc_sweep_end
    gc_sweep_end.restype = None
    gc_sweep_end.argtypes = [ctypes.POINTER(struct_gc_ctx)]
except (AttributeError, RuntimeError):
    pass
class struct_linear_ctx(Structure):
    pass

linear_ctx = struct_linear_ctx
try:
    linear_alloc_child = _libraries['FIXME_STUB'].linear_alloc_child
    linear_alloc_child.restype = ctypes.POINTER(None)
    linear_alloc_child.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
class struct_c__SA_linear_opts(Structure):
    pass

struct_c__SA_linear_opts._pack_ = 1 # source:False
struct_c__SA_linear_opts._fields_ = [
    ('min_buffer_size', ctypes.c_uint32),
]

linear_opts = struct_c__SA_linear_opts
try:
    linear_context = _libraries['FIXME_STUB'].linear_context
    linear_context.restype = ctypes.POINTER(struct_linear_ctx)
    linear_context.argtypes = [ctypes.POINTER(None)]
except (AttributeError, RuntimeError):
    pass
try:
    linear_context_with_opts = _libraries['FIXME_STUB'].linear_context_with_opts
    linear_context_with_opts.restype = ctypes.POINTER(struct_linear_ctx)
    linear_context_with_opts.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_c__SA_linear_opts)]
except (AttributeError, RuntimeError):
    pass
try:
    linear_zalloc_child = _libraries['FIXME_STUB'].linear_zalloc_child
    linear_zalloc_child.restype = ctypes.POINTER(None)
    linear_zalloc_child.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    linear_free_context = _libraries['FIXME_STUB'].linear_free_context
    linear_free_context.restype = None
    linear_free_context.argtypes = [ctypes.POINTER(struct_linear_ctx)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_steal_linear_context = _libraries['FIXME_STUB'].ralloc_steal_linear_context
    ralloc_steal_linear_context.restype = None
    ralloc_steal_linear_context.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_linear_ctx)]
except (AttributeError, RuntimeError):
    pass
try:
    ralloc_parent_of_linear_context = _libraries['FIXME_STUB'].ralloc_parent_of_linear_context
    ralloc_parent_of_linear_context.restype = ctypes.POINTER(None)
    ralloc_parent_of_linear_context.argtypes = [ctypes.POINTER(struct_linear_ctx)]
except (AttributeError, RuntimeError):
    pass
try:
    linear_alloc_child_array = _libraries['FIXME_STUB'].linear_alloc_child_array
    linear_alloc_child_array.restype = ctypes.POINTER(None)
    linear_alloc_child_array.argtypes = [ctypes.POINTER(struct_linear_ctx), size_t, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    linear_zalloc_child_array = _libraries['FIXME_STUB'].linear_zalloc_child_array
    linear_zalloc_child_array.restype = ctypes.POINTER(None)
    linear_zalloc_child_array.argtypes = [ctypes.POINTER(struct_linear_ctx), size_t, ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
try:
    linear_strdup = _libraries['FIXME_STUB'].linear_strdup
    linear_strdup.restype = ctypes.POINTER(ctypes.c_char)
    linear_strdup.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    linear_asprintf = _libraries['FIXME_STUB'].linear_asprintf
    linear_asprintf.restype = ctypes.POINTER(ctypes.c_char)
    linear_asprintf.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    linear_vasprintf = _libraries['FIXME_STUB'].linear_vasprintf
    linear_vasprintf.restype = ctypes.POINTER(ctypes.c_char)
    linear_vasprintf.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.c_char), va_list]
except (AttributeError, RuntimeError):
    pass
try:
    linear_asprintf_append = _libraries['FIXME_STUB'].linear_asprintf_append
    linear_asprintf_append.restype = ctypes.c_bool
    linear_asprintf_append.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    linear_vasprintf_append = _libraries['FIXME_STUB'].linear_vasprintf_append
    linear_vasprintf_append.restype = ctypes.c_bool
    linear_vasprintf_append.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char), va_list]
except (AttributeError, RuntimeError):
    pass
try:
    linear_asprintf_rewrite_tail = _libraries['FIXME_STUB'].linear_asprintf_rewrite_tail
    linear_asprintf_rewrite_tail.restype = ctypes.c_bool
    linear_asprintf_rewrite_tail.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass
try:
    linear_vasprintf_rewrite_tail = _libraries['FIXME_STUB'].linear_vasprintf_rewrite_tail
    linear_vasprintf_rewrite_tail.restype = ctypes.c_bool
    linear_vasprintf_rewrite_tail.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_char), va_list]
except (AttributeError, RuntimeError):
    pass
try:
    linear_strcat = _libraries['FIXME_STUB'].linear_strcat
    linear_strcat.restype = ctypes.c_bool
    linear_strcat.argtypes = [ctypes.POINTER(struct_linear_ctx), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except (AttributeError, RuntimeError):
    pass

# values for enumeration 'c__Ea_RALLOC_PRINT_INFO_SUMMARY_ONLY'
c__Ea_RALLOC_PRINT_INFO_SUMMARY_ONLY__enumvalues = {
    1: 'RALLOC_PRINT_INFO_SUMMARY_ONLY',
}
RALLOC_PRINT_INFO_SUMMARY_ONLY = 1
c__Ea_RALLOC_PRINT_INFO_SUMMARY_ONLY = ctypes.c_uint32 # enum
class struct__IO_FILE(Structure):
    pass

class struct__IO_marker(Structure):
    pass

class struct__IO_codecvt(Structure):
    pass

class struct__IO_wide_data(Structure):
    pass

struct__IO_FILE._pack_ = 1 # source:False
struct__IO_FILE._fields_ = [
    ('_flags', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
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
    ('_vtable_offset', ctypes.c_byte),
    ('_shortbuf', ctypes.c_char * 1),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('_lock', ctypes.POINTER(None)),
    ('_offset', ctypes.c_int64),
    ('_codecvt', ctypes.POINTER(struct__IO_codecvt)),
    ('_wide_data', ctypes.POINTER(struct__IO_wide_data)),
    ('_freeres_list', ctypes.POINTER(struct__IO_FILE)),
    ('_freeres_buf', ctypes.POINTER(None)),
    ('__pad5', ctypes.c_uint64),
    ('_mode', ctypes.c_int32),
    ('_unused2', ctypes.c_char * 20),
]

try:
    ralloc_print_info = _libraries['FIXME_STUB'].ralloc_print_info
    ralloc_print_info.restype = None
    ralloc_print_info.argtypes = [ctypes.POINTER(struct__IO_FILE), ctypes.POINTER(None), ctypes.c_uint32]
except (AttributeError, RuntimeError):
    pass
__all__ = \
    ['GLSL_CMAT_USE_A', 'GLSL_CMAT_USE_ACCUMULATOR',
    'GLSL_CMAT_USE_B', 'GLSL_CMAT_USE_NONE',
    'GLSL_INTERFACE_PACKING_PACKED', 'GLSL_INTERFACE_PACKING_SHARED',
    'GLSL_INTERFACE_PACKING_STD140', 'GLSL_INTERFACE_PACKING_STD430',
    'GLSL_MATRIX_LAYOUT_COLUMN_MAJOR', 'GLSL_MATRIX_LAYOUT_INHERITED',
    'GLSL_MATRIX_LAYOUT_ROW_MAJOR', 'GLSL_PRECISION_HIGH',
    'GLSL_PRECISION_LOW', 'GLSL_PRECISION_MEDIUM',
    'GLSL_PRECISION_NONE', 'GLSL_SAMPLER_DIM_1D',
    'GLSL_SAMPLER_DIM_2D', 'GLSL_SAMPLER_DIM_3D',
    'GLSL_SAMPLER_DIM_BUF', 'GLSL_SAMPLER_DIM_CUBE',
    'GLSL_SAMPLER_DIM_EXTERNAL', 'GLSL_SAMPLER_DIM_MS',
    'GLSL_SAMPLER_DIM_RECT', 'GLSL_SAMPLER_DIM_SUBPASS',
    'GLSL_SAMPLER_DIM_SUBPASS_MS', 'GLSL_TYPE_ARRAY',
    'GLSL_TYPE_ATOMIC_UINT', 'GLSL_TYPE_BOOL',
    'GLSL_TYPE_COOPERATIVE_MATRIX', 'GLSL_TYPE_DOUBLE',
    'GLSL_TYPE_ERROR', 'GLSL_TYPE_FLOAT', 'GLSL_TYPE_FLOAT16',
    'GLSL_TYPE_IMAGE', 'GLSL_TYPE_INT', 'GLSL_TYPE_INT16',
    'GLSL_TYPE_INT64', 'GLSL_TYPE_INT8', 'GLSL_TYPE_INTERFACE',
    'GLSL_TYPE_SAMPLER', 'GLSL_TYPE_STRUCT', 'GLSL_TYPE_SUBROUTINE',
    'GLSL_TYPE_TEXTURE', 'GLSL_TYPE_UINT', 'GLSL_TYPE_UINT16',
    'GLSL_TYPE_UINT64', 'GLSL_TYPE_UINT8', 'GLSL_TYPE_VOID',
    'MESA_SHADER_ANY_HIT', 'MESA_SHADER_CALLABLE',
    'MESA_SHADER_CLOSEST_HIT', 'MESA_SHADER_COMPUTE',
    'MESA_SHADER_FRAGMENT', 'MESA_SHADER_GEOMETRY',
    'MESA_SHADER_INTERSECTION', 'MESA_SHADER_KERNEL',
    'MESA_SHADER_MESH', 'MESA_SHADER_MISS', 'MESA_SHADER_NONE',
    'MESA_SHADER_RAYGEN', 'MESA_SHADER_TASK', 'MESA_SHADER_TESS_CTRL',
    'MESA_SHADER_TESS_EVAL', 'MESA_SHADER_VERTEX',
    'NAK_TS_DOMAIN_ISOLINE', 'NAK_TS_DOMAIN_QUAD',
    'NAK_TS_DOMAIN_TRIANGLE', 'NAK_TS_PRIMS_LINES',
    'NAK_TS_PRIMS_POINTS', 'NAK_TS_PRIMS_TRIANGLES_CCW',
    'NAK_TS_PRIMS_TRIANGLES_CW', 'NAK_TS_SPACING_FRACT_EVEN',
    'NAK_TS_SPACING_FRACT_ODD', 'NAK_TS_SPACING_INTEGER',
    'NV_DEVICE_TYPE_DIS', 'NV_DEVICE_TYPE_IGP', 'NV_DEVICE_TYPE_SOC',
    'PIPE_FORMAT_A16_FLOAT', 'PIPE_FORMAT_A16_SINT',
    'PIPE_FORMAT_A16_SNORM', 'PIPE_FORMAT_A16_UINT',
    'PIPE_FORMAT_A16_UNORM', 'PIPE_FORMAT_A1B5G5R5_UINT',
    'PIPE_FORMAT_A1B5G5R5_UNORM', 'PIPE_FORMAT_A1R5G5B5_UINT',
    'PIPE_FORMAT_A1R5G5B5_UNORM', 'PIPE_FORMAT_A2B10G10R10_UINT',
    'PIPE_FORMAT_A2B10G10R10_UNORM', 'PIPE_FORMAT_A2R10G10B10_UINT',
    'PIPE_FORMAT_A2R10G10B10_UNORM', 'PIPE_FORMAT_A32_FLOAT',
    'PIPE_FORMAT_A32_SINT', 'PIPE_FORMAT_A32_UINT',
    'PIPE_FORMAT_A4B4G4R4_UINT', 'PIPE_FORMAT_A4B4G4R4_UNORM',
    'PIPE_FORMAT_A4R4G4B4_UINT', 'PIPE_FORMAT_A4R4G4B4_UNORM',
    'PIPE_FORMAT_A4R4_UNORM', 'PIPE_FORMAT_A8B8G8R8_SINT',
    'PIPE_FORMAT_A8B8G8R8_SNORM', 'PIPE_FORMAT_A8B8G8R8_SRGB',
    'PIPE_FORMAT_A8B8G8R8_SSCALED', 'PIPE_FORMAT_A8B8G8R8_UINT',
    'PIPE_FORMAT_A8B8G8R8_UNORM', 'PIPE_FORMAT_A8B8G8R8_USCALED',
    'PIPE_FORMAT_A8R8G8B8_SINT', 'PIPE_FORMAT_A8R8G8B8_SNORM',
    'PIPE_FORMAT_A8R8G8B8_SRGB', 'PIPE_FORMAT_A8R8G8B8_UINT',
    'PIPE_FORMAT_A8R8G8B8_UNORM', 'PIPE_FORMAT_A8R8_UNORM',
    'PIPE_FORMAT_A8_SINT', 'PIPE_FORMAT_A8_SNORM',
    'PIPE_FORMAT_A8_UINT', 'PIPE_FORMAT_A8_UNORM',
    'PIPE_FORMAT_ASTC_10x10', 'PIPE_FORMAT_ASTC_10x10_SRGB',
    'PIPE_FORMAT_ASTC_10x5', 'PIPE_FORMAT_ASTC_10x5_SRGB',
    'PIPE_FORMAT_ASTC_10x6', 'PIPE_FORMAT_ASTC_10x6_SRGB',
    'PIPE_FORMAT_ASTC_10x8', 'PIPE_FORMAT_ASTC_10x8_SRGB',
    'PIPE_FORMAT_ASTC_12x10', 'PIPE_FORMAT_ASTC_12x10_SRGB',
    'PIPE_FORMAT_ASTC_12x12', 'PIPE_FORMAT_ASTC_12x12_SRGB',
    'PIPE_FORMAT_ASTC_3x3x3', 'PIPE_FORMAT_ASTC_3x3x3_SRGB',
    'PIPE_FORMAT_ASTC_4x3x3', 'PIPE_FORMAT_ASTC_4x3x3_SRGB',
    'PIPE_FORMAT_ASTC_4x4', 'PIPE_FORMAT_ASTC_4x4_SRGB',
    'PIPE_FORMAT_ASTC_4x4x3', 'PIPE_FORMAT_ASTC_4x4x3_SRGB',
    'PIPE_FORMAT_ASTC_4x4x4', 'PIPE_FORMAT_ASTC_4x4x4_SRGB',
    'PIPE_FORMAT_ASTC_5x4', 'PIPE_FORMAT_ASTC_5x4_SRGB',
    'PIPE_FORMAT_ASTC_5x4x4', 'PIPE_FORMAT_ASTC_5x4x4_SRGB',
    'PIPE_FORMAT_ASTC_5x5', 'PIPE_FORMAT_ASTC_5x5_SRGB',
    'PIPE_FORMAT_ASTC_5x5x4', 'PIPE_FORMAT_ASTC_5x5x4_SRGB',
    'PIPE_FORMAT_ASTC_5x5x5', 'PIPE_FORMAT_ASTC_5x5x5_SRGB',
    'PIPE_FORMAT_ASTC_6x5', 'PIPE_FORMAT_ASTC_6x5_SRGB',
    'PIPE_FORMAT_ASTC_6x5x5', 'PIPE_FORMAT_ASTC_6x5x5_SRGB',
    'PIPE_FORMAT_ASTC_6x6', 'PIPE_FORMAT_ASTC_6x6_SRGB',
    'PIPE_FORMAT_ASTC_6x6x5', 'PIPE_FORMAT_ASTC_6x6x5_SRGB',
    'PIPE_FORMAT_ASTC_6x6x6', 'PIPE_FORMAT_ASTC_6x6x6_SRGB',
    'PIPE_FORMAT_ASTC_8x5', 'PIPE_FORMAT_ASTC_8x5_SRGB',
    'PIPE_FORMAT_ASTC_8x6', 'PIPE_FORMAT_ASTC_8x6_SRGB',
    'PIPE_FORMAT_ASTC_8x8', 'PIPE_FORMAT_ASTC_8x8_SRGB',
    'PIPE_FORMAT_ATC_RGB', 'PIPE_FORMAT_ATC_RGBA_EXPLICIT',
    'PIPE_FORMAT_ATC_RGBA_INTERPOLATED', 'PIPE_FORMAT_AYUV',
    'PIPE_FORMAT_B10G10R10A2_SINT', 'PIPE_FORMAT_B10G10R10A2_SNORM',
    'PIPE_FORMAT_B10G10R10A2_SSCALED', 'PIPE_FORMAT_B10G10R10A2_UINT',
    'PIPE_FORMAT_B10G10R10A2_UNORM',
    'PIPE_FORMAT_B10G10R10A2_USCALED', 'PIPE_FORMAT_B10G10R10X2_SINT',
    'PIPE_FORMAT_B10G10R10X2_SNORM', 'PIPE_FORMAT_B10G10R10X2_UNORM',
    'PIPE_FORMAT_B2G3R3_UINT', 'PIPE_FORMAT_B2G3R3_UNORM',
    'PIPE_FORMAT_B4G4R4A4_UINT', 'PIPE_FORMAT_B4G4R4A4_UNORM',
    'PIPE_FORMAT_B4G4R4X4_UNORM', 'PIPE_FORMAT_B5G5R5A1_UINT',
    'PIPE_FORMAT_B5G5R5A1_UNORM', 'PIPE_FORMAT_B5G5R5X1_UNORM',
    'PIPE_FORMAT_B5G6R5_SRGB', 'PIPE_FORMAT_B5G6R5_UINT',
    'PIPE_FORMAT_B5G6R5_UNORM', 'PIPE_FORMAT_B8G8R8A8_SINT',
    'PIPE_FORMAT_B8G8R8A8_SNORM', 'PIPE_FORMAT_B8G8R8A8_SRGB',
    'PIPE_FORMAT_B8G8R8A8_SSCALED', 'PIPE_FORMAT_B8G8R8A8_UINT',
    'PIPE_FORMAT_B8G8R8A8_UNORM', 'PIPE_FORMAT_B8G8R8A8_USCALED',
    'PIPE_FORMAT_B8G8R8X8_SINT', 'PIPE_FORMAT_B8G8R8X8_SNORM',
    'PIPE_FORMAT_B8G8R8X8_SRGB', 'PIPE_FORMAT_B8G8R8X8_UINT',
    'PIPE_FORMAT_B8G8R8X8_UNORM', 'PIPE_FORMAT_B8G8R8_SINT',
    'PIPE_FORMAT_B8G8R8_SNORM', 'PIPE_FORMAT_B8G8R8_SRGB',
    'PIPE_FORMAT_B8G8R8_SSCALED', 'PIPE_FORMAT_B8G8R8_UINT',
    'PIPE_FORMAT_B8G8R8_UNORM', 'PIPE_FORMAT_B8G8R8_USCALED',
    'PIPE_FORMAT_B8G8_R8G8_UNORM', 'PIPE_FORMAT_B8R8_G8R8_UNORM',
    'PIPE_FORMAT_BPTC_RGBA_UNORM', 'PIPE_FORMAT_BPTC_RGB_FLOAT',
    'PIPE_FORMAT_BPTC_RGB_UFLOAT', 'PIPE_FORMAT_BPTC_SRGBA',
    'PIPE_FORMAT_COUNT', 'PIPE_FORMAT_DXT1_RGB',
    'PIPE_FORMAT_DXT1_RGBA', 'PIPE_FORMAT_DXT1_SRGB',
    'PIPE_FORMAT_DXT1_SRGBA', 'PIPE_FORMAT_DXT3_RGBA',
    'PIPE_FORMAT_DXT3_SRGBA', 'PIPE_FORMAT_DXT5_RGBA',
    'PIPE_FORMAT_DXT5_SRGBA', 'PIPE_FORMAT_ETC1_RGB8',
    'PIPE_FORMAT_ETC2_R11_SNORM', 'PIPE_FORMAT_ETC2_R11_UNORM',
    'PIPE_FORMAT_ETC2_RG11_SNORM', 'PIPE_FORMAT_ETC2_RG11_UNORM',
    'PIPE_FORMAT_ETC2_RGB8', 'PIPE_FORMAT_ETC2_RGB8A1',
    'PIPE_FORMAT_ETC2_RGBA8', 'PIPE_FORMAT_ETC2_SRGB8',
    'PIPE_FORMAT_ETC2_SRGB8A1', 'PIPE_FORMAT_ETC2_SRGBA8',
    'PIPE_FORMAT_FXT1_RGB', 'PIPE_FORMAT_FXT1_RGBA',
    'PIPE_FORMAT_G16R16_SINT', 'PIPE_FORMAT_G16R16_SNORM',
    'PIPE_FORMAT_G16R16_UNORM', 'PIPE_FORMAT_G8B8_G8R8_UNORM',
    'PIPE_FORMAT_G8R8_B8R8_UNORM', 'PIPE_FORMAT_G8R8_G8B8_UNORM',
    'PIPE_FORMAT_G8R8_SINT', 'PIPE_FORMAT_G8R8_SNORM',
    'PIPE_FORMAT_G8R8_UNORM', 'PIPE_FORMAT_G8_B8R8_420_UNORM',
    'PIPE_FORMAT_G8_B8R8_422_UNORM', 'PIPE_FORMAT_G8_B8_R8_420_UNORM',
    'PIPE_FORMAT_I16_FLOAT', 'PIPE_FORMAT_I16_SINT',
    'PIPE_FORMAT_I16_SNORM', 'PIPE_FORMAT_I16_UINT',
    'PIPE_FORMAT_I16_UNORM', 'PIPE_FORMAT_I32_FLOAT',
    'PIPE_FORMAT_I32_SINT', 'PIPE_FORMAT_I32_UINT',
    'PIPE_FORMAT_I8_SINT', 'PIPE_FORMAT_I8_SNORM',
    'PIPE_FORMAT_I8_UINT', 'PIPE_FORMAT_I8_UNORM', 'PIPE_FORMAT_IYUV',
    'PIPE_FORMAT_L16A16_FLOAT', 'PIPE_FORMAT_L16A16_SINT',
    'PIPE_FORMAT_L16A16_SNORM', 'PIPE_FORMAT_L16A16_UINT',
    'PIPE_FORMAT_L16A16_UNORM', 'PIPE_FORMAT_L16_FLOAT',
    'PIPE_FORMAT_L16_SINT', 'PIPE_FORMAT_L16_SNORM',
    'PIPE_FORMAT_L16_UINT', 'PIPE_FORMAT_L16_UNORM',
    'PIPE_FORMAT_L32A32_FLOAT', 'PIPE_FORMAT_L32A32_SINT',
    'PIPE_FORMAT_L32A32_UINT', 'PIPE_FORMAT_L32_FLOAT',
    'PIPE_FORMAT_L32_SINT', 'PIPE_FORMAT_L32_UINT',
    'PIPE_FORMAT_L4A4_UNORM', 'PIPE_FORMAT_L8A8_SINT',
    'PIPE_FORMAT_L8A8_SNORM', 'PIPE_FORMAT_L8A8_SRGB',
    'PIPE_FORMAT_L8A8_UINT', 'PIPE_FORMAT_L8A8_UNORM',
    'PIPE_FORMAT_L8_SINT', 'PIPE_FORMAT_L8_SNORM',
    'PIPE_FORMAT_L8_SRGB', 'PIPE_FORMAT_L8_UINT',
    'PIPE_FORMAT_L8_UNORM', 'PIPE_FORMAT_LATC1_SNORM',
    'PIPE_FORMAT_LATC1_UNORM', 'PIPE_FORMAT_LATC2_SNORM',
    'PIPE_FORMAT_LATC2_UNORM', 'PIPE_FORMAT_NONE', 'PIPE_FORMAT_NV12',
    'PIPE_FORMAT_NV15', 'PIPE_FORMAT_NV16', 'PIPE_FORMAT_NV20',
    'PIPE_FORMAT_NV21', 'PIPE_FORMAT_P010', 'PIPE_FORMAT_P012',
    'PIPE_FORMAT_P016', 'PIPE_FORMAT_P030',
    'PIPE_FORMAT_R10G10B10A2_SINT', 'PIPE_FORMAT_R10G10B10A2_SNORM',
    'PIPE_FORMAT_R10G10B10A2_SSCALED', 'PIPE_FORMAT_R10G10B10A2_UINT',
    'PIPE_FORMAT_R10G10B10A2_UNORM',
    'PIPE_FORMAT_R10G10B10A2_USCALED', 'PIPE_FORMAT_R10G10B10X2_SINT',
    'PIPE_FORMAT_R10G10B10X2_SNORM', 'PIPE_FORMAT_R10G10B10X2_UNORM',
    'PIPE_FORMAT_R10G10B10X2_USCALED',
    'PIPE_FORMAT_R10SG10SB10SA2U_NORM',
    'PIPE_FORMAT_R10_G10B10_420_UNORM',
    'PIPE_FORMAT_R10_G10B10_422_UNORM', 'PIPE_FORMAT_R11G11B10_FLOAT',
    'PIPE_FORMAT_R16A16_FLOAT', 'PIPE_FORMAT_R16A16_SINT',
    'PIPE_FORMAT_R16A16_SNORM', 'PIPE_FORMAT_R16A16_UINT',
    'PIPE_FORMAT_R16A16_UNORM', 'PIPE_FORMAT_R16G16B16A16_FLOAT',
    'PIPE_FORMAT_R16G16B16A16_SINT', 'PIPE_FORMAT_R16G16B16A16_SNORM',
    'PIPE_FORMAT_R16G16B16A16_SSCALED',
    'PIPE_FORMAT_R16G16B16A16_UINT', 'PIPE_FORMAT_R16G16B16A16_UNORM',
    'PIPE_FORMAT_R16G16B16A16_USCALED',
    'PIPE_FORMAT_R16G16B16X16_FLOAT', 'PIPE_FORMAT_R16G16B16X16_SINT',
    'PIPE_FORMAT_R16G16B16X16_SNORM', 'PIPE_FORMAT_R16G16B16X16_UINT',
    'PIPE_FORMAT_R16G16B16X16_UNORM', 'PIPE_FORMAT_R16G16B16_FLOAT',
    'PIPE_FORMAT_R16G16B16_SINT', 'PIPE_FORMAT_R16G16B16_SNORM',
    'PIPE_FORMAT_R16G16B16_SSCALED', 'PIPE_FORMAT_R16G16B16_UINT',
    'PIPE_FORMAT_R16G16B16_UNORM', 'PIPE_FORMAT_R16G16B16_USCALED',
    'PIPE_FORMAT_R16G16_FLOAT', 'PIPE_FORMAT_R16G16_SINT',
    'PIPE_FORMAT_R16G16_SNORM', 'PIPE_FORMAT_R16G16_SSCALED',
    'PIPE_FORMAT_R16G16_UINT', 'PIPE_FORMAT_R16G16_UNORM',
    'PIPE_FORMAT_R16G16_USCALED', 'PIPE_FORMAT_R16_FLOAT',
    'PIPE_FORMAT_R16_SINT', 'PIPE_FORMAT_R16_SNORM',
    'PIPE_FORMAT_R16_SSCALED', 'PIPE_FORMAT_R16_UINT',
    'PIPE_FORMAT_R16_UNORM', 'PIPE_FORMAT_R16_USCALED',
    'PIPE_FORMAT_R1_UNORM', 'PIPE_FORMAT_R32A32_FLOAT',
    'PIPE_FORMAT_R32A32_SINT', 'PIPE_FORMAT_R32A32_UINT',
    'PIPE_FORMAT_R32G32B32A32_FIXED',
    'PIPE_FORMAT_R32G32B32A32_FLOAT', 'PIPE_FORMAT_R32G32B32A32_SINT',
    'PIPE_FORMAT_R32G32B32A32_SNORM',
    'PIPE_FORMAT_R32G32B32A32_SSCALED',
    'PIPE_FORMAT_R32G32B32A32_UINT', 'PIPE_FORMAT_R32G32B32A32_UNORM',
    'PIPE_FORMAT_R32G32B32A32_USCALED',
    'PIPE_FORMAT_R32G32B32X32_FLOAT', 'PIPE_FORMAT_R32G32B32X32_SINT',
    'PIPE_FORMAT_R32G32B32X32_UINT', 'PIPE_FORMAT_R32G32B32_FIXED',
    'PIPE_FORMAT_R32G32B32_FLOAT', 'PIPE_FORMAT_R32G32B32_SINT',
    'PIPE_FORMAT_R32G32B32_SNORM', 'PIPE_FORMAT_R32G32B32_SSCALED',
    'PIPE_FORMAT_R32G32B32_UINT', 'PIPE_FORMAT_R32G32B32_UNORM',
    'PIPE_FORMAT_R32G32B32_USCALED', 'PIPE_FORMAT_R32G32_FIXED',
    'PIPE_FORMAT_R32G32_FLOAT', 'PIPE_FORMAT_R32G32_SINT',
    'PIPE_FORMAT_R32G32_SNORM', 'PIPE_FORMAT_R32G32_SSCALED',
    'PIPE_FORMAT_R32G32_UINT', 'PIPE_FORMAT_R32G32_UNORM',
    'PIPE_FORMAT_R32G32_USCALED', 'PIPE_FORMAT_R32_FIXED',
    'PIPE_FORMAT_R32_FLOAT', 'PIPE_FORMAT_R32_SINT',
    'PIPE_FORMAT_R32_SNORM', 'PIPE_FORMAT_R32_SSCALED',
    'PIPE_FORMAT_R32_UINT', 'PIPE_FORMAT_R32_UNORM',
    'PIPE_FORMAT_R32_USCALED', 'PIPE_FORMAT_R3G3B2_UINT',
    'PIPE_FORMAT_R3G3B2_UNORM', 'PIPE_FORMAT_R4A4_UNORM',
    'PIPE_FORMAT_R4G4B4A4_UINT', 'PIPE_FORMAT_R4G4B4A4_UNORM',
    'PIPE_FORMAT_R4G4B4X4_UNORM', 'PIPE_FORMAT_R5G5B5A1_UINT',
    'PIPE_FORMAT_R5G5B5A1_UNORM', 'PIPE_FORMAT_R5G5B5X1_UNORM',
    'PIPE_FORMAT_R5G6B5_SRGB', 'PIPE_FORMAT_R5G6B5_UINT',
    'PIPE_FORMAT_R5G6B5_UNORM', 'PIPE_FORMAT_R5SG5SB6U_NORM',
    'PIPE_FORMAT_R64G64B64A64_FLOAT', 'PIPE_FORMAT_R64G64B64A64_SINT',
    'PIPE_FORMAT_R64G64B64A64_UINT', 'PIPE_FORMAT_R64G64B64_FLOAT',
    'PIPE_FORMAT_R64G64B64_SINT', 'PIPE_FORMAT_R64G64B64_UINT',
    'PIPE_FORMAT_R64G64_FLOAT', 'PIPE_FORMAT_R64G64_SINT',
    'PIPE_FORMAT_R64G64_UINT', 'PIPE_FORMAT_R64_FLOAT',
    'PIPE_FORMAT_R64_SINT', 'PIPE_FORMAT_R64_UINT',
    'PIPE_FORMAT_R8A8_SINT', 'PIPE_FORMAT_R8A8_SNORM',
    'PIPE_FORMAT_R8A8_UINT', 'PIPE_FORMAT_R8A8_UNORM',
    'PIPE_FORMAT_R8B8_R8G8_UNORM', 'PIPE_FORMAT_R8G8B8A8_SINT',
    'PIPE_FORMAT_R8G8B8A8_SNORM', 'PIPE_FORMAT_R8G8B8A8_SRGB',
    'PIPE_FORMAT_R8G8B8A8_SSCALED', 'PIPE_FORMAT_R8G8B8A8_UINT',
    'PIPE_FORMAT_R8G8B8A8_UNORM', 'PIPE_FORMAT_R8G8B8A8_USCALED',
    'PIPE_FORMAT_R8G8B8X8_SINT', 'PIPE_FORMAT_R8G8B8X8_SNORM',
    'PIPE_FORMAT_R8G8B8X8_SRGB', 'PIPE_FORMAT_R8G8B8X8_UINT',
    'PIPE_FORMAT_R8G8B8X8_UNORM', 'PIPE_FORMAT_R8G8B8_SINT',
    'PIPE_FORMAT_R8G8B8_SNORM', 'PIPE_FORMAT_R8G8B8_SRGB',
    'PIPE_FORMAT_R8G8B8_SSCALED', 'PIPE_FORMAT_R8G8B8_UINT',
    'PIPE_FORMAT_R8G8B8_UNORM', 'PIPE_FORMAT_R8G8B8_USCALED',
    'PIPE_FORMAT_R8G8Bx_SNORM', 'PIPE_FORMAT_R8G8_B8G8_UNORM',
    'PIPE_FORMAT_R8G8_R8B8_UNORM', 'PIPE_FORMAT_R8G8_SINT',
    'PIPE_FORMAT_R8G8_SNORM', 'PIPE_FORMAT_R8G8_SRGB',
    'PIPE_FORMAT_R8G8_SSCALED', 'PIPE_FORMAT_R8G8_UINT',
    'PIPE_FORMAT_R8G8_UNORM', 'PIPE_FORMAT_R8G8_USCALED',
    'PIPE_FORMAT_R8SG8SB8UX8U_NORM', 'PIPE_FORMAT_R8_B8G8_420_UNORM',
    'PIPE_FORMAT_R8_B8G8_422_UNORM', 'PIPE_FORMAT_R8_B8_G8_420_UNORM',
    'PIPE_FORMAT_R8_G8B8_420_UNORM', 'PIPE_FORMAT_R8_G8B8_422_UNORM',
    'PIPE_FORMAT_R8_G8_B8_420_UNORM', 'PIPE_FORMAT_R8_G8_B8_UNORM',
    'PIPE_FORMAT_R8_SINT', 'PIPE_FORMAT_R8_SNORM',
    'PIPE_FORMAT_R8_SRGB', 'PIPE_FORMAT_R8_SSCALED',
    'PIPE_FORMAT_R8_UINT', 'PIPE_FORMAT_R8_UNORM',
    'PIPE_FORMAT_R8_USCALED', 'PIPE_FORMAT_R9G9B9E5_FLOAT',
    'PIPE_FORMAT_RGTC1_SNORM', 'PIPE_FORMAT_RGTC1_UNORM',
    'PIPE_FORMAT_RGTC2_SNORM', 'PIPE_FORMAT_RGTC2_UNORM',
    'PIPE_FORMAT_S8X24_UINT', 'PIPE_FORMAT_S8_UINT',
    'PIPE_FORMAT_S8_UINT_Z24_UNORM', 'PIPE_FORMAT_UYVY',
    'PIPE_FORMAT_VYUY', 'PIPE_FORMAT_X1B5G5R5_UNORM',
    'PIPE_FORMAT_X1R5G5B5_UNORM', 'PIPE_FORMAT_X24S8_UINT',
    'PIPE_FORMAT_X32_S8X24_UINT',
    'PIPE_FORMAT_X4G12_X4B12X4R12_420_UNORM',
    'PIPE_FORMAT_X4R12X4G12_UNORM', 'PIPE_FORMAT_X4R12_UNORM',
    'PIPE_FORMAT_X6G10_X6B10X6R10_420_UNORM',
    'PIPE_FORMAT_X6R10X6G10_UNORM', 'PIPE_FORMAT_X6R10_UNORM',
    'PIPE_FORMAT_X8B8G8R8_SINT', 'PIPE_FORMAT_X8B8G8R8_SNORM',
    'PIPE_FORMAT_X8B8G8R8_SRGB', 'PIPE_FORMAT_X8B8G8R8_UNORM',
    'PIPE_FORMAT_X8R8G8B8_SINT', 'PIPE_FORMAT_X8R8G8B8_SNORM',
    'PIPE_FORMAT_X8R8G8B8_SRGB', 'PIPE_FORMAT_X8R8G8B8_UNORM',
    'PIPE_FORMAT_X8Z24_UNORM', 'PIPE_FORMAT_XYUV',
    'PIPE_FORMAT_Y16_U16V16_422_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_420_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_422_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_444_UNORM', 'PIPE_FORMAT_Y210',
    'PIPE_FORMAT_Y212', 'PIPE_FORMAT_Y216', 'PIPE_FORMAT_Y410',
    'PIPE_FORMAT_Y412', 'PIPE_FORMAT_Y416',
    'PIPE_FORMAT_Y8_400_UNORM', 'PIPE_FORMAT_Y8_U8_V8_422_UNORM',
    'PIPE_FORMAT_Y8_U8_V8_440_UNORM',
    'PIPE_FORMAT_Y8_U8_V8_444_UNORM', 'PIPE_FORMAT_Y8_UNORM',
    'PIPE_FORMAT_YUYV', 'PIPE_FORMAT_YV12', 'PIPE_FORMAT_YV16',
    'PIPE_FORMAT_YVYU', 'PIPE_FORMAT_Z16_UNORM',
    'PIPE_FORMAT_Z16_UNORM_S8_UINT', 'PIPE_FORMAT_Z24X8_UNORM',
    'PIPE_FORMAT_Z24_UNORM_S8_UINT',
    'PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8',
    'PIPE_FORMAT_Z32_FLOAT', 'PIPE_FORMAT_Z32_FLOAT_S8X24_UINT',
    'PIPE_FORMAT_Z32_UNORM', 'PIPE_SHADER_COMPUTE',
    'PIPE_SHADER_FRAGMENT', 'PIPE_SHADER_GEOMETRY',
    'PIPE_SHADER_MESH', 'PIPE_SHADER_MESH_TYPES', 'PIPE_SHADER_TASK',
    'PIPE_SHADER_TESS_CTRL', 'PIPE_SHADER_TESS_EVAL',
    'PIPE_SHADER_TYPES', 'PIPE_SHADER_VERTEX',
    'RALLOC_PRINT_INFO_SUMMARY_ONLY', 'blob_align', 'blob_copy_bytes',
    'blob_finish', 'blob_finish_get_buffer', 'blob_init',
    'blob_init_fixed', 'blob_overwrite_bytes',
    'blob_overwrite_intptr', 'blob_overwrite_uint32',
    'blob_overwrite_uint8', 'blob_read_bytes', 'blob_read_intptr',
    'blob_read_string', 'blob_read_uint16', 'blob_read_uint32',
    'blob_read_uint64', 'blob_read_uint8', 'blob_reader_align',
    'blob_reader_init', 'blob_reserve_bytes', 'blob_reserve_intptr',
    'blob_reserve_uint32', 'blob_skip_bytes', 'blob_write_bytes',
    'blob_write_intptr', 'blob_write_string', 'blob_write_uint16',
    'blob_write_uint32', 'blob_write_uint64', 'blob_write_uint8',
    'c__EA_nir_divergence_options', 'c__EA_nir_io_options',
    'c__EA_nir_lower_doubles_options',
    'c__EA_nir_lower_int64_options', 'c__EA_nir_lower_packing_op',
    'c__EA_nir_variable_mode', 'c__Ea_GLSL_PRECISION_NONE',
    'c__Ea_RALLOC_PRINT_INFO_SUMMARY_ONLY', 'decode_type_from_blob',
    'encode_type_to_blob', 'gc_alloc_size', 'gc_context', 'gc_ctx',
    'gc_free', 'gc_get_context', 'gc_mark_live', 'gc_sweep_end',
    'gc_sweep_start', 'gc_zalloc_size', 'glsl_array_size',
    'glsl_array_type', 'glsl_atomic_size', 'glsl_atomic_uint_type',
    'glsl_bare_sampler_type', 'glsl_bare_shadow_sampler_type',
    'glsl_base_type', 'glsl_base_type_bit_size',
    'glsl_base_type_get_bit_size', 'glsl_base_type_is_16bit',
    'glsl_base_type_is_64bit', 'glsl_base_type_is_integer',
    'glsl_bool_type', 'glsl_bvec2_type', 'glsl_bvec4_type',
    'glsl_bvec_type', 'glsl_channel_type', 'glsl_cmat_type',
    'glsl_cmat_use', 'glsl_contains_array', 'glsl_contains_atomic',
    'glsl_contains_double', 'glsl_contains_integer',
    'glsl_contains_opaque', 'glsl_contains_sampler',
    'glsl_contains_subroutine', 'glsl_count_attribute_slots',
    'glsl_count_dword_slots', 'glsl_count_vec4_slots',
    'glsl_double_type', 'glsl_dvec2_type', 'glsl_dvec4_type',
    'glsl_dvec_type', 'glsl_explicit_matrix_type', 'glsl_f16vec_type',
    'glsl_float16_t_type', 'glsl_float16_type', 'glsl_floatN_t_type',
    'glsl_float_type', 'glsl_get_aoa_size', 'glsl_get_array_element',
    'glsl_get_bare_type', 'glsl_get_base_glsl_type',
    'glsl_get_base_type', 'glsl_get_bit_size',
    'glsl_get_cl_alignment', 'glsl_get_cl_size',
    'glsl_get_cl_type_size_align', 'glsl_get_cmat_description',
    'glsl_get_cmat_element', 'glsl_get_column_type',
    'glsl_get_component_slots', 'glsl_get_component_slots_aligned',
    'glsl_get_components', 'glsl_get_explicit_alignment',
    'glsl_get_explicit_interface_type', 'glsl_get_explicit_size',
    'glsl_get_explicit_std140_type', 'glsl_get_explicit_std430_type',
    'glsl_get_explicit_stride',
    'glsl_get_explicit_type_for_size_align', 'glsl_get_field_index',
    'glsl_get_field_type', 'glsl_get_ifc_packing',
    'glsl_get_internal_ifc_packing', 'glsl_get_length',
    'glsl_get_matrix_columns', 'glsl_get_mul_type',
    'glsl_get_natural_size_align_bytes', 'glsl_get_row_type',
    'glsl_get_sampler_coordinate_components', 'glsl_get_sampler_dim',
    'glsl_get_sampler_dim_coordinate_components',
    'glsl_get_sampler_result_type', 'glsl_get_scalar_type',
    'glsl_get_std140_base_alignment', 'glsl_get_std140_size',
    'glsl_get_std430_array_stride', 'glsl_get_std430_base_alignment',
    'glsl_get_std430_size', 'glsl_get_struct_elem_name',
    'glsl_get_struct_field', 'glsl_get_struct_field_data',
    'glsl_get_struct_field_offset', 'glsl_get_struct_location_offset',
    'glsl_get_type_name', 'glsl_get_vec4_size_align_bytes',
    'glsl_get_vector_elements', 'glsl_get_word_size_align_bytes',
    'glsl_i16vec_type', 'glsl_i64vec_type', 'glsl_i8vec_type',
    'glsl_image_type', 'glsl_int16_t_type', 'glsl_int16_type',
    'glsl_int64_t_type', 'glsl_int8_t_type', 'glsl_intN_t_type',
    'glsl_int_type', 'glsl_interface_packing', 'glsl_interface_type',
    'glsl_ivec2_type', 'glsl_ivec4_type', 'glsl_ivec_type',
    'glsl_matrix_layout', 'glsl_matrix_type',
    'glsl_matrix_type_is_row_major', 'glsl_record_compare',
    'glsl_replace_vector_type', 'glsl_sampler_dim',
    'glsl_sampler_type', 'glsl_sampler_type_is_array',
    'glsl_sampler_type_is_shadow', 'glsl_sampler_type_to_texture',
    'glsl_scalar_type', 'glsl_signed_base_type_of',
    'glsl_simple_explicit_type', 'glsl_simple_type',
    'glsl_size_align_handle_array_and_structs', 'glsl_struct_field',
    'glsl_struct_type', 'glsl_struct_type_is_packed',
    'glsl_struct_type_with_explicit_alignment',
    'glsl_subroutine_type', 'glsl_texture_type',
    'glsl_texture_type_to_sampler', 'glsl_transposed_type',
    'glsl_type', 'glsl_type_compare_no_precision',
    'glsl_type_contains_32bit', 'glsl_type_contains_64bit',
    'glsl_type_contains_image', 'glsl_type_get_image_count',
    'glsl_type_get_sampler_count', 'glsl_type_get_texture_count',
    'glsl_type_is_16bit', 'glsl_type_is_32bit', 'glsl_type_is_64bit',
    'glsl_type_is_array', 'glsl_type_is_array_of_arrays',
    'glsl_type_is_array_or_matrix', 'glsl_type_is_atomic_uint',
    'glsl_type_is_bare_sampler', 'glsl_type_is_boolean',
    'glsl_type_is_cmat', 'glsl_type_is_double',
    'glsl_type_is_dual_slot', 'glsl_type_is_error',
    'glsl_type_is_float', 'glsl_type_is_float_16',
    'glsl_type_is_float_16_32', 'glsl_type_is_float_16_32_64',
    'glsl_type_is_image', 'glsl_type_is_int_16_32',
    'glsl_type_is_int_16_32_64', 'glsl_type_is_integer',
    'glsl_type_is_integer_16', 'glsl_type_is_integer_16_32',
    'glsl_type_is_integer_16_32_64', 'glsl_type_is_integer_32',
    'glsl_type_is_integer_32_64', 'glsl_type_is_integer_64',
    'glsl_type_is_interface', 'glsl_type_is_leaf',
    'glsl_type_is_matrix', 'glsl_type_is_numeric',
    'glsl_type_is_packed', 'glsl_type_is_sampler',
    'glsl_type_is_scalar', 'glsl_type_is_struct',
    'glsl_type_is_struct_or_ifc', 'glsl_type_is_subroutine',
    'glsl_type_is_texture', 'glsl_type_is_uint_16_32',
    'glsl_type_is_uint_16_32_64', 'glsl_type_is_unsized_array',
    'glsl_type_is_vector', 'glsl_type_is_vector_or_scalar',
    'glsl_type_is_void', 'glsl_type_replace_vec3_with_vec4',
    'glsl_type_singleton_decref', 'glsl_type_singleton_init_or_ref',
    'glsl_type_size_align_func', 'glsl_type_to_16bit',
    'glsl_type_uniform_locations', 'glsl_type_wrap_in_arrays',
    'glsl_u16vec_type', 'glsl_u64vec_type', 'glsl_u8vec_type',
    'glsl_uint16_t_type', 'glsl_uint16_type', 'glsl_uint64_t_type',
    'glsl_uint8_t_type', 'glsl_uintN_t_type', 'glsl_uint_type',
    'glsl_unsigned_base_type_of', 'glsl_uvec2_type',
    'glsl_uvec4_type', 'glsl_uvec_type', 'glsl_varying_count',
    'glsl_vec2_type', 'glsl_vec4_type', 'glsl_vec_type',
    'glsl_vector_type', 'glsl_void_type', 'glsl_without_array',
    'glsl_without_array_or_matrix', 'intptr_t', 'linear_alloc_child',
    'linear_alloc_child_array', 'linear_asprintf',
    'linear_asprintf_append', 'linear_asprintf_rewrite_tail',
    'linear_context', 'linear_context_with_opts', 'linear_ctx',
    'linear_free_context', 'linear_opts', 'linear_strcat',
    'linear_strdup', 'linear_vasprintf', 'linear_vasprintf_append',
    'linear_vasprintf_rewrite_tail', 'linear_zalloc_child',
    'linear_zalloc_child_array', 'nak_compile_shader',
    'nak_compiler_create', 'nak_compiler_destroy', 'nak_debug_flags',
    'nak_fill_qmd', 'nak_get_qmd_cbuf_desc_layout',
    'nak_get_qmd_dispatch_size_layout', 'nak_nir_options',
    'nak_postprocess_nir', 'nak_preprocess_nir',
    'nak_shader_bin_destroy', 'nak_ts_domain', 'nak_ts_prims',
    'nak_ts_spacing', 'nir_deserialize', 'nir_deserialize_function',
    'nir_divergence_ignore_undef_if_phi_srcs',
    'nir_divergence_multiple_workgroup_per_compute_subgroup',
    'nir_divergence_options', 'nir_divergence_options__enumvalues',
    'nir_divergence_shader_record_ptr_uniform',
    'nir_divergence_single_frag_shading_rate_per_subgroup',
    'nir_divergence_single_patch_per_tcs_subgroup',
    'nir_divergence_single_patch_per_tes_subgroup',
    'nir_divergence_single_prim_per_subgroup',
    'nir_divergence_uniform_load_tears',
    'nir_divergence_view_index_uniform', 'nir_instr_filter_cb',
    'nir_io_16bit_input_output_support',
    'nir_io_always_interpolate_convergent_fs_inputs',
    'nir_io_compaction_rotates_color_channels',
    'nir_io_dont_optimize', 'nir_io_dont_use_pos_for_non_fs_varyings',
    'nir_io_has_flexible_input_interpolation_except_flat',
    'nir_io_has_intrinsics', 'nir_io_mediump_is_32bit',
    'nir_io_mix_convergent_flat_with_interpolated', 'nir_io_options',
    'nir_io_options__enumvalues', 'nir_io_prefer_scalar_fs_inputs',
    'nir_io_separate_clip_cull_distance_arrays',
    'nir_io_vectorizer_ignores_types', 'nir_lower_bcsel64',
    'nir_lower_bit_count64', 'nir_lower_conv64', 'nir_lower_dceil',
    'nir_lower_ddiv', 'nir_lower_dfloor', 'nir_lower_dfract',
    'nir_lower_divmod64', 'nir_lower_dminmax', 'nir_lower_dmod',
    'nir_lower_doubles_options',
    'nir_lower_doubles_options__enumvalues', 'nir_lower_drcp',
    'nir_lower_dround_even', 'nir_lower_drsq', 'nir_lower_dsat',
    'nir_lower_dsign', 'nir_lower_dsqrt', 'nir_lower_dsub',
    'nir_lower_dtrunc', 'nir_lower_extract64', 'nir_lower_find_lsb64',
    'nir_lower_fp64_full_software', 'nir_lower_iabs64',
    'nir_lower_iadd3_64', 'nir_lower_iadd64', 'nir_lower_iadd_sat64',
    'nir_lower_icmp64', 'nir_lower_imul64', 'nir_lower_imul_2x32_64',
    'nir_lower_imul_high64', 'nir_lower_ineg64',
    'nir_lower_int64_options', 'nir_lower_int64_options__enumvalues',
    'nir_lower_isign64', 'nir_lower_logic64', 'nir_lower_minmax64',
    'nir_lower_packing_num_ops', 'nir_lower_packing_op',
    'nir_lower_packing_op__enumvalues',
    'nir_lower_packing_op_pack_32_2x16',
    'nir_lower_packing_op_pack_32_4x8',
    'nir_lower_packing_op_pack_64_2x32',
    'nir_lower_packing_op_pack_64_4x16',
    'nir_lower_packing_op_unpack_32_2x16',
    'nir_lower_packing_op_unpack_32_4x8',
    'nir_lower_packing_op_unpack_64_2x32',
    'nir_lower_packing_op_unpack_64_4x16',
    'nir_lower_scan_reduce_bitwise64', 'nir_lower_scan_reduce_iadd64',
    'nir_lower_shift64', 'nir_lower_subgroup_shuffle64',
    'nir_lower_uadd_sat64', 'nir_lower_ufind_msb64',
    'nir_lower_usub_sat64', 'nir_lower_vote_ieq64',
    'nir_num_variable_modes', 'nir_serialize',
    'nir_serialize_function', 'nir_shader_compiler_options',
    'nir_var_all', 'nir_var_function_in', 'nir_var_function_inout',
    'nir_var_function_out', 'nir_var_function_temp', 'nir_var_image',
    'nir_var_mem_constant', 'nir_var_mem_generic',
    'nir_var_mem_global', 'nir_var_mem_node_payload',
    'nir_var_mem_node_payload_in', 'nir_var_mem_push_const',
    'nir_var_mem_shared', 'nir_var_mem_ssbo',
    'nir_var_mem_task_payload', 'nir_var_mem_ubo',
    'nir_var_ray_hit_attrib', 'nir_var_read_only_modes',
    'nir_var_shader_call_data', 'nir_var_shader_in',
    'nir_var_shader_out', 'nir_var_shader_temp',
    'nir_var_system_value', 'nir_var_uniform',
    'nir_var_vec_indexable_modes', 'nir_variable_mode',
    'nir_variable_mode__enumvalues', 'nv_device_type',
    'nv_device_uuid', 'pipe_format', 'pipe_shader_type',
    'ralloc_adopt', 'ralloc_array_size', 'ralloc_asprintf',
    'ralloc_asprintf_append', 'ralloc_asprintf_rewrite_tail',
    'ralloc_context', 'ralloc_free', 'ralloc_memdup', 'ralloc_parent',
    'ralloc_parent_of_linear_context', 'ralloc_print_info',
    'ralloc_set_destructor', 'ralloc_size', 'ralloc_steal',
    'ralloc_steal_linear_context', 'ralloc_str_append',
    'ralloc_strcat', 'ralloc_strdup', 'ralloc_strncat',
    'ralloc_strndup', 'ralloc_total_size', 'ralloc_vasprintf',
    'ralloc_vasprintf_append', 'ralloc_vasprintf_rewrite_tail',
    'reralloc_array_size', 'reralloc_size', 'rerzalloc_array_size',
    'rerzalloc_size', 'rzalloc_array_size', 'rzalloc_size', 'size_t',
    'struct__IO_FILE', 'struct__IO_codecvt', 'struct__IO_marker',
    'struct__IO_wide_data', 'struct___va_list_tag', 'struct_blob',
    'struct_blob_reader', 'struct_c__SA_linear_opts', 'struct_gc_ctx',
    'struct_glsl_cmat_description', 'struct_glsl_struct_field',
    'struct_glsl_struct_field_0_0', 'struct_glsl_type',
    'struct_linear_ctx', 'struct_nak_compiler', 'struct_nak_fs_key',
    'struct_nak_qmd_cbuf', 'struct_nak_qmd_cbuf_desc_layout',
    'struct_nak_qmd_dispatch_size_layout', 'struct_nak_qmd_info',
    'struct_nak_sample_location', 'struct_nak_sample_mask',
    'struct_nak_shader_bin', 'struct_nak_shader_info',
    'struct_nak_shader_info_0_cs', 'struct_nak_shader_info_0_fs',
    'struct_nak_shader_info_0_ts', 'struct_nak_shader_info_vtg',
    'struct_nak_xfb_info', 'struct_nir_function', 'struct_nir_instr',
    'struct_nir_shader', 'struct_nir_shader_compiler_options',
    'struct_nv_device_info', 'struct_nv_device_info_pci', 'uint16_t',
    'uint32_t', 'uint64_t', 'uint8_t', 'union_glsl_struct_field_0',
    'union_glsl_type_fields', 'union_nak_shader_info_0', 'va_list']
def __getattr__(nm): raise AttributeError() if nm.startswith('__') else dll.error
