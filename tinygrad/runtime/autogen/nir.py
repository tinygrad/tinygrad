# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-DHAVE_ENDIAN_H', '-DHAVE_STRUCT_TIMESPEC', '-DHAVE_PTHREAD', '-I/tmp/mesa-9e0991eff5aea2e064fc16d5c7fa0ee6cd52d894/src', '-I/tmp/mesa-9e0991eff5aea2e064fc16d5c7fa0ee6cd52d894/include', '-I/tmp/mesa-9e0991eff5aea2e064fc16d5c7fa0ee6cd52d894/src/compiler/nir', '-I/tmp/mesa-9e0991eff5aea2e064fc16d5c7fa0ee6cd52d894/gen']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, os


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

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['FIXME_STUB'] = ctypes.CDLL(os.getenv('MESA_PATH', '/usr/')+'/lib/x86_64-linux-gnu/libvulkan_nouveau.so') #  ctypes.CDLL('FIXME_STUB')
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





class struct_exec_node(Structure):
    pass

struct_exec_node._pack_ = 1 # source:False
struct_exec_node._fields_ = [
    ('next', ctypes.POINTER(struct_exec_node)),
    ('prev', ctypes.POINTER(struct_exec_node)),
]

try:
    exec_node_init = _libraries['FIXME_STUB'].exec_node_init
    exec_node_init.restype = None
    exec_node_init.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_get_next_const = _libraries['FIXME_STUB'].exec_node_get_next_const
    exec_node_get_next_const.restype = ctypes.POINTER(struct_exec_node)
    exec_node_get_next_const.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_get_next = _libraries['FIXME_STUB'].exec_node_get_next
    exec_node_get_next.restype = ctypes.POINTER(struct_exec_node)
    exec_node_get_next.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_get_prev_const = _libraries['FIXME_STUB'].exec_node_get_prev_const
    exec_node_get_prev_const.restype = ctypes.POINTER(struct_exec_node)
    exec_node_get_prev_const.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_get_prev = _libraries['FIXME_STUB'].exec_node_get_prev
    exec_node_get_prev.restype = ctypes.POINTER(struct_exec_node)
    exec_node_get_prev.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_remove = _libraries['FIXME_STUB'].exec_node_remove
    exec_node_remove.restype = None
    exec_node_remove.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_self_link = _libraries['FIXME_STUB'].exec_node_self_link
    exec_node_self_link.restype = None
    exec_node_self_link.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_insert_after = _libraries['FIXME_STUB'].exec_node_insert_after
    exec_node_insert_after.restype = None
    exec_node_insert_after.argtypes = [ctypes.POINTER(struct_exec_node), ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_insert_node_before = _libraries['FIXME_STUB'].exec_node_insert_node_before
    exec_node_insert_node_before.restype = None
    exec_node_insert_node_before.argtypes = [ctypes.POINTER(struct_exec_node), ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_is_tail_sentinel = _libraries['FIXME_STUB'].exec_node_is_tail_sentinel
    exec_node_is_tail_sentinel.restype = ctypes.c_bool
    exec_node_is_tail_sentinel.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_node_is_head_sentinel = _libraries['FIXME_STUB'].exec_node_is_head_sentinel
    exec_node_is_head_sentinel.restype = ctypes.c_bool
    exec_node_is_head_sentinel.argtypes = [ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
class struct_exec_list(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('head_sentinel', struct_exec_node),
    ('tail_sentinel', struct_exec_node),
     ]

try:
    exec_list_make_empty = _libraries['FIXME_STUB'].exec_list_make_empty
    exec_list_make_empty.restype = None
    exec_list_make_empty.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_is_empty = _libraries['FIXME_STUB'].exec_list_is_empty
    exec_list_is_empty.restype = ctypes.c_bool
    exec_list_is_empty.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_is_singular = _libraries['FIXME_STUB'].exec_list_is_singular
    exec_list_is_singular.restype = ctypes.c_bool
    exec_list_is_singular.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_get_head_const = _libraries['FIXME_STUB'].exec_list_get_head_const
    exec_list_get_head_const.restype = ctypes.POINTER(struct_exec_node)
    exec_list_get_head_const.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_get_head = _libraries['FIXME_STUB'].exec_list_get_head
    exec_list_get_head.restype = ctypes.POINTER(struct_exec_node)
    exec_list_get_head.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_get_head_raw = _libraries['FIXME_STUB'].exec_list_get_head_raw
    exec_list_get_head_raw.restype = ctypes.POINTER(struct_exec_node)
    exec_list_get_head_raw.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_get_tail = _libraries['FIXME_STUB'].exec_list_get_tail
    exec_list_get_tail.restype = ctypes.POINTER(struct_exec_node)
    exec_list_get_tail.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_length = _libraries['FIXME_STUB'].exec_list_length
    exec_list_length.restype = ctypes.c_uint32
    exec_list_length.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_push_head = _libraries['FIXME_STUB'].exec_list_push_head
    exec_list_push_head.restype = None
    exec_list_push_head.argtypes = [ctypes.POINTER(struct_exec_list), ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_list_push_tail = _libraries['FIXME_STUB'].exec_list_push_tail
    exec_list_push_tail.restype = None
    exec_list_push_tail.argtypes = [ctypes.POINTER(struct_exec_list), ctypes.POINTER(struct_exec_node)]
except AttributeError:
    pass
try:
    exec_list_pop_head = _libraries['FIXME_STUB'].exec_list_pop_head
    exec_list_pop_head.restype = ctypes.POINTER(struct_exec_node)
    exec_list_pop_head.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_move_nodes_to = _libraries['FIXME_STUB'].exec_list_move_nodes_to
    exec_list_move_nodes_to.restype = None
    exec_list_move_nodes_to.argtypes = [ctypes.POINTER(struct_exec_list), ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_append = _libraries['FIXME_STUB'].exec_list_append
    exec_list_append.restype = None
    exec_list_append.argtypes = [ctypes.POINTER(struct_exec_list), ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_node_insert_list_after = _libraries['FIXME_STUB'].exec_node_insert_list_after
    exec_node_insert_list_after.restype = None
    exec_node_insert_list_after.argtypes = [ctypes.POINTER(struct_exec_node), ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    exec_list_validate = _libraries['FIXME_STUB'].exec_list_validate
    exec_list_validate.restype = None
    exec_list_validate.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass

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
    67108864: 'nir_lower_bitfield_reverse64',
    134217728: 'nir_lower_bitfield_extract64',
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
nir_lower_bitfield_reverse64 = 67108864
nir_lower_bitfield_extract64 = 134217728
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
    2048: 'nir_divergence_vertex',
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
nir_divergence_vertex = 2048
c__EA_nir_divergence_options = ctypes.c_uint32 # enum
nir_divergence_options = c__EA_nir_divergence_options
nir_divergence_options__enumvalues = c__EA_nir_divergence_options__enumvalues
class struct_nir_instr(Structure):
    pass

class struct_nir_block(Structure):
    pass

struct_nir_instr._pack_ = 1 # source:False
struct_nir_instr._fields_ = [
    ('node', struct_exec_node),
    ('block', ctypes.POINTER(struct_nir_block)),
    ('type', ctypes.c_ubyte),
    ('pass_flags', ctypes.c_ubyte),
    ('has_debug_info', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('index', ctypes.c_uint32),
]

class struct_set(Structure):
    pass

class struct_nir_cf_node(Structure):
    pass


# values for enumeration 'c__EA_nir_cf_node_type'
c__EA_nir_cf_node_type__enumvalues = {
    0: 'nir_cf_node_block',
    1: 'nir_cf_node_if',
    2: 'nir_cf_node_loop',
    3: 'nir_cf_node_function',
}
nir_cf_node_block = 0
nir_cf_node_if = 1
nir_cf_node_loop = 2
nir_cf_node_function = 3
c__EA_nir_cf_node_type = ctypes.c_uint32 # enum
struct_nir_cf_node._pack_ = 1 # source:False
struct_nir_cf_node._fields_ = [
    ('node', struct_exec_node),
    ('type', c__EA_nir_cf_node_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('parent', ctypes.POINTER(struct_nir_cf_node)),
]

struct_nir_block._pack_ = 1 # source:False
struct_nir_block._fields_ = [
    ('cf_node', struct_nir_cf_node),
    ('instr_list', struct_exec_list),
    ('index', ctypes.c_uint32),
    ('divergent', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('successors', ctypes.POINTER(struct_nir_block) * 2),
    ('predecessors', ctypes.POINTER(struct_set)),
    ('imm_dom', ctypes.POINTER(struct_nir_block)),
    ('num_dom_children', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dom_children', ctypes.POINTER(ctypes.POINTER(struct_nir_block))),
    ('dom_frontier', ctypes.POINTER(struct_set)),
    ('dom_pre_index', ctypes.c_uint32),
    ('dom_post_index', ctypes.c_uint32),
    ('start_ip', ctypes.c_uint32),
    ('end_ip', ctypes.c_uint32),
    ('live_in', ctypes.POINTER(ctypes.c_uint32)),
    ('live_out', ctypes.POINTER(ctypes.c_uint32)),
]

class struct_set_entry(Structure):
    pass

struct_set._pack_ = 1 # source:False
struct_set._fields_ = [
    ('mem_ctx', ctypes.POINTER(None)),
    ('table', ctypes.POINTER(struct_set_entry)),
    ('key_hash_function', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(None))),
    ('key_equals_function', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(None), ctypes.POINTER(None))),
    ('size', ctypes.c_uint32),
    ('rehash', ctypes.c_uint32),
    ('size_magic', ctypes.c_uint64),
    ('rehash_magic', ctypes.c_uint64),
    ('max_entries', ctypes.c_uint32),
    ('size_index', ctypes.c_uint32),
    ('entries', ctypes.c_uint32),
    ('deleted_entries', ctypes.c_uint32),
]

struct_set_entry._pack_ = 1 # source:False
struct_set_entry._fields_ = [
    ('hash', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('key', ctypes.POINTER(None)),
]

nir_instr_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))

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
    512: 'nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups',
    1024: 'nir_io_radv_intrinsic_component_workaround',
    65536: 'nir_io_has_intrinsics',
    131072: 'nir_io_separate_clip_cull_distance_arrays',
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
nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups = 512
nir_io_radv_intrinsic_component_workaround = 1024
nir_io_has_intrinsics = 65536
nir_io_separate_clip_cull_distance_arrays = 131072
c__EA_nir_io_options = ctypes.c_uint32 # enum
nir_io_options = c__EA_nir_io_options
nir_io_options__enumvalues = c__EA_nir_io_options__enumvalues

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
class struct_nir_shader_compiler_options(Structure):
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
    ('PADDING_0', ctypes.c_ubyte * 5),
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
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('max_unroll_iterations', ctypes.c_uint32),
    ('max_unroll_iterations_aggressive', ctypes.c_uint32),
    ('max_unroll_iterations_fp64', ctypes.c_uint32),
    ('lower_uniforms_to_ubo', ctypes.c_bool),
    ('force_indirect_unrolling_sampler', ctypes.c_bool),
    ('no_integers', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte),
    ('force_indirect_unrolling', c__EA_nir_variable_mode),
    ('driver_functions', ctypes.c_bool),
    ('late_lower_int64', ctypes.c_bool),
    ('PADDING_3', ctypes.c_ubyte * 2),
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
    ('PADDING_4', ctypes.c_ubyte * 2),
    ('io_options', nir_io_options),
    ('skip_lower_packing_ops', ctypes.c_uint32),
    ('subgroup_size', ctypes.c_ubyte),
    ('ballot_bit_size', ctypes.c_ubyte),
    ('ballot_components', ctypes.c_ubyte),
    ('PADDING_5', ctypes.c_ubyte * 5),
    ('lower_mediump_io', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_nir_shader))),
    ('varying_expression_max_cost', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader))),
    ('varying_estimate_instr_cost', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr))),
    ('max_varying_expression_cost', ctypes.c_uint32),
    ('PADDING_6', ctypes.c_ubyte * 4),
]

class struct_gc_ctx(Structure):
    pass

class struct_nir_xfb_info(Structure):
    pass

class struct_u_printf_info(Structure):
    pass

class struct_shader_info(Structure):
    pass


# values for enumeration 'mesa_shader_stage'
mesa_shader_stage__enumvalues = {
    -1: 'MESA_SHADER_NONE',
    0: 'MESA_SHADER_VERTEX',
    1: 'MESA_SHADER_TESS_CTRL',
    2: 'MESA_SHADER_TESS_EVAL',
    3: 'MESA_SHADER_GEOMETRY',
    4: 'MESA_SHADER_FRAGMENT',
    5: 'MESA_SHADER_COMPUTE',
    6: 'MESA_SHADER_TASK',
    7: 'MESA_SHADER_MESH',
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
MESA_SHADER_TESS_CTRL = 1
MESA_SHADER_TESS_EVAL = 2
MESA_SHADER_GEOMETRY = 3
MESA_SHADER_FRAGMENT = 4
MESA_SHADER_COMPUTE = 5
MESA_SHADER_TASK = 6
MESA_SHADER_MESH = 7
MESA_SHADER_RAYGEN = 8
MESA_SHADER_ANY_HIT = 9
MESA_SHADER_CLOSEST_HIT = 10
MESA_SHADER_MISS = 11
MESA_SHADER_INTERSECTION = 12
MESA_SHADER_CALLABLE = 13
MESA_SHADER_KERNEL = 14
mesa_shader_stage = ctypes.c_int32 # enum

# values for enumeration 'gl_subgroup_size'
gl_subgroup_size__enumvalues = {
    0: 'SUBGROUP_SIZE_VARYING',
    1: 'SUBGROUP_SIZE_UNIFORM',
    2: 'SUBGROUP_SIZE_API_CONSTANT',
    3: 'SUBGROUP_SIZE_FULL_SUBGROUPS',
    4: 'SUBGROUP_SIZE_REQUIRE_4',
    8: 'SUBGROUP_SIZE_REQUIRE_8',
    16: 'SUBGROUP_SIZE_REQUIRE_16',
    32: 'SUBGROUP_SIZE_REQUIRE_32',
    64: 'SUBGROUP_SIZE_REQUIRE_64',
    128: 'SUBGROUP_SIZE_REQUIRE_128',
}
SUBGROUP_SIZE_VARYING = 0
SUBGROUP_SIZE_UNIFORM = 1
SUBGROUP_SIZE_API_CONSTANT = 2
SUBGROUP_SIZE_FULL_SUBGROUPS = 3
SUBGROUP_SIZE_REQUIRE_4 = 4
SUBGROUP_SIZE_REQUIRE_8 = 8
SUBGROUP_SIZE_REQUIRE_16 = 16
SUBGROUP_SIZE_REQUIRE_32 = 32
SUBGROUP_SIZE_REQUIRE_64 = 64
SUBGROUP_SIZE_REQUIRE_128 = 128
gl_subgroup_size = ctypes.c_uint32 # enum

# values for enumeration 'gl_derivative_group'
gl_derivative_group__enumvalues = {
    0: 'DERIVATIVE_GROUP_NONE',
    1: 'DERIVATIVE_GROUP_QUADS',
    2: 'DERIVATIVE_GROUP_LINEAR',
}
DERIVATIVE_GROUP_NONE = 0
DERIVATIVE_GROUP_QUADS = 1
DERIVATIVE_GROUP_LINEAR = 2
gl_derivative_group = ctypes.c_uint32 # enum
class union_shader_info_0(Union):
    pass

class struct_shader_info_0_vs(Structure):
    pass

struct_shader_info_0_vs._pack_ = 1 # source:False
struct_shader_info_0_vs._fields_ = [
    ('double_inputs', ctypes.c_uint64),
    ('blit_sgprs_amd', ctypes.c_ubyte, 4),
    ('tes_agx', ctypes.c_ubyte, 1),
    ('window_space_position', ctypes.c_ubyte, 1),
    ('needs_edge_flag', ctypes.c_ubyte, 1),
    ('PADDING_0', ctypes.c_uint64, 57),
]

class struct_shader_info_0_gs(Structure):
    pass


# values for enumeration 'mesa_prim'
mesa_prim__enumvalues = {
    0: 'MESA_PRIM_POINTS',
    1: 'MESA_PRIM_LINES',
    2: 'MESA_PRIM_LINE_LOOP',
    3: 'MESA_PRIM_LINE_STRIP',
    4: 'MESA_PRIM_TRIANGLES',
    5: 'MESA_PRIM_TRIANGLE_STRIP',
    6: 'MESA_PRIM_TRIANGLE_FAN',
    7: 'MESA_PRIM_QUADS',
    8: 'MESA_PRIM_QUAD_STRIP',
    9: 'MESA_PRIM_POLYGON',
    10: 'MESA_PRIM_LINES_ADJACENCY',
    11: 'MESA_PRIM_LINE_STRIP_ADJACENCY',
    12: 'MESA_PRIM_TRIANGLES_ADJACENCY',
    13: 'MESA_PRIM_TRIANGLE_STRIP_ADJACENCY',
    14: 'MESA_PRIM_PATCHES',
    14: 'MESA_PRIM_MAX',
    15: 'MESA_PRIM_COUNT',
    28: 'MESA_PRIM_UNKNOWN',
}
MESA_PRIM_POINTS = 0
MESA_PRIM_LINES = 1
MESA_PRIM_LINE_LOOP = 2
MESA_PRIM_LINE_STRIP = 3
MESA_PRIM_TRIANGLES = 4
MESA_PRIM_TRIANGLE_STRIP = 5
MESA_PRIM_TRIANGLE_FAN = 6
MESA_PRIM_QUADS = 7
MESA_PRIM_QUAD_STRIP = 8
MESA_PRIM_POLYGON = 9
MESA_PRIM_LINES_ADJACENCY = 10
MESA_PRIM_LINE_STRIP_ADJACENCY = 11
MESA_PRIM_TRIANGLES_ADJACENCY = 12
MESA_PRIM_TRIANGLE_STRIP_ADJACENCY = 13
MESA_PRIM_PATCHES = 14
MESA_PRIM_MAX = 14
MESA_PRIM_COUNT = 15
MESA_PRIM_UNKNOWN = 28
mesa_prim = ctypes.c_uint32 # enum
struct_shader_info_0_gs._pack_ = 1 # source:False
struct_shader_info_0_gs._fields_ = [
    ('output_primitive', mesa_prim),
    ('input_primitive', mesa_prim),
    ('vertices_out', ctypes.c_uint16),
    ('invocations', ctypes.c_ubyte),
    ('vertices_in', ctypes.c_ubyte, 3),
    ('uses_end_primitive', ctypes.c_ubyte, 1),
    ('active_stream_mask', ctypes.c_ubyte, 4),
]

class struct_shader_info_0_fs(Structure):
    pass


# values for enumeration 'c_uint64'
c_uint64__enumvalues = {
    0: 'FRAG_DEPTH_LAYOUT_NONE',
    1: 'FRAG_DEPTH_LAYOUT_ANY',
    2: 'FRAG_DEPTH_LAYOUT_GREATER',
    3: 'FRAG_DEPTH_LAYOUT_LESS',
    4: 'FRAG_DEPTH_LAYOUT_UNCHANGED',
}
FRAG_DEPTH_LAYOUT_NONE = 0
FRAG_DEPTH_LAYOUT_ANY = 1
FRAG_DEPTH_LAYOUT_GREATER = 2
FRAG_DEPTH_LAYOUT_LESS = 3
FRAG_DEPTH_LAYOUT_UNCHANGED = 4
c_uint64 = ctypes.c_uint32 # enum

# values for enumeration 'c_bool'
c_bool__enumvalues = {
    0: 'FRAG_STENCIL_LAYOUT_NONE',
    1: 'FRAG_STENCIL_LAYOUT_ANY',
    2: 'FRAG_STENCIL_LAYOUT_GREATER',
    3: 'FRAG_STENCIL_LAYOUT_LESS',
    4: 'FRAG_STENCIL_LAYOUT_UNCHANGED',
}
FRAG_STENCIL_LAYOUT_NONE = 0
FRAG_STENCIL_LAYOUT_ANY = 1
FRAG_STENCIL_LAYOUT_GREATER = 2
FRAG_STENCIL_LAYOUT_LESS = 3
FRAG_STENCIL_LAYOUT_UNCHANGED = 4
c_bool = ctypes.c_uint32 # enum
struct_shader_info_0_fs._pack_ = 1 # source:False
struct_shader_info_0_fs._fields_ = [
    ('uses_discard', ctypes.c_uint64, 1),
    ('uses_fbfetch_output', ctypes.c_uint64, 1),
    ('fbfetch_coherent', ctypes.c_uint64, 1),
    ('color_is_dual_source', ctypes.c_uint64, 1),
    ('require_full_quads', ctypes.c_uint64, 1),
    ('quad_derivatives', ctypes.c_uint64, 1),
    ('needs_coarse_quad_helper_invocations', ctypes.c_uint64, 1),
    ('needs_full_quad_helper_invocations', ctypes.c_uint64, 1),
    ('uses_sample_qualifier', ctypes.c_uint64, 1),
    ('uses_sample_shading', ctypes.c_uint64, 1),
    ('early_fragment_tests', ctypes.c_uint64, 1),
    ('inner_coverage', ctypes.c_uint64, 1),
    ('post_depth_coverage', ctypes.c_uint64, 1),
    ('pixel_center_integer', ctypes.c_uint64, 1),
    ('origin_upper_left', ctypes.c_uint64, 1),
    ('pixel_interlock_ordered', ctypes.c_uint64, 1),
    ('pixel_interlock_unordered', ctypes.c_uint64, 1),
    ('sample_interlock_ordered', ctypes.c_uint64, 1),
    ('sample_interlock_unordered', ctypes.c_uint64, 1),
    ('untyped_color_outputs', ctypes.c_uint64, 1),
    ('depth_layout', c_uint64, 3),
    ('color0_interp', ctypes.c_uint64, 3),
    ('color0_sample', ctypes.c_uint64, 1),
    ('color0_centroid', ctypes.c_uint64, 1),
    ('color1_interp', ctypes.c_uint64, 3),
    ('color1_sample', ctypes.c_uint64, 1),
    ('color1_centroid', ctypes.c_uint64, 1),
    ('PADDING_0', ctypes.c_uint32, 31),
    ('advanced_blend_modes', ctypes.c_uint32),
    ('early_and_late_fragment_tests', ctypes.c_bool, 1),
    ('stencil_front_layout', c_bool, 3),
    ('stencil_back_layout', c_bool, 3),
    ('PADDING_1', ctypes.c_uint32, 25),
]

class struct_shader_info_0_cs(Structure):
    pass

struct_shader_info_0_cs._pack_ = 1 # source:False
struct_shader_info_0_cs._fields_ = [
    ('workgroup_size_hint', ctypes.c_uint16 * 3),
    ('user_data_components_amd', ctypes.c_ubyte, 4),
    ('has_variable_shared_mem', ctypes.c_ubyte, 1),
    ('has_cooperative_matrix', ctypes.c_ubyte, 1),
    ('PADDING_0', ctypes.c_uint8, 2),
    ('image_block_size_per_thread_agx', ctypes.c_ubyte, 8),
    ('ptr_size', ctypes.c_uint32),
    ('shader_index', ctypes.c_uint32),
    ('node_payloads_size', ctypes.c_uint32),
    ('workgroup_count', ctypes.c_uint32 * 3),
]

class struct_shader_info_0_tess(Structure):
    pass


# values for enumeration 'tess_primitive_mode'
tess_primitive_mode__enumvalues = {
    0: 'TESS_PRIMITIVE_UNSPECIFIED',
    1: 'TESS_PRIMITIVE_TRIANGLES',
    2: 'TESS_PRIMITIVE_QUADS',
    3: 'TESS_PRIMITIVE_ISOLINES',
}
TESS_PRIMITIVE_UNSPECIFIED = 0
TESS_PRIMITIVE_TRIANGLES = 1
TESS_PRIMITIVE_QUADS = 2
TESS_PRIMITIVE_ISOLINES = 3
tess_primitive_mode = ctypes.c_uint32 # enum
struct_shader_info_0_tess._pack_ = 1 # source:False
struct_shader_info_0_tess._fields_ = [
    ('_primitive_mode', tess_primitive_mode),
    ('tcs_vertices_out', ctypes.c_ubyte),
    ('spacing', ctypes.c_uint32, 2),
    ('ccw', ctypes.c_uint32, 1),
    ('point_mode', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint32, 20),
    ('tcs_same_invocation_inputs_read', ctypes.c_uint64),
    ('tcs_cross_invocation_inputs_read', ctypes.c_uint64),
    ('tcs_cross_invocation_outputs_read', ctypes.c_uint64),
    ('tcs_cross_invocation_outputs_written', ctypes.c_uint64),
    ('tcs_outputs_read_by_tes', ctypes.c_uint64),
    ('tcs_patch_outputs_read_by_tes', ctypes.c_uint32),
    ('tcs_outputs_read_by_tes_16bit', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 2),
]

class struct_shader_info_0_mesh(Structure):
    pass

struct_shader_info_0_mesh._pack_ = 1 # source:False
struct_shader_info_0_mesh._fields_ = [
    ('ms_cross_invocation_output_access', ctypes.c_uint64),
    ('ts_mesh_dispatch_dimensions', ctypes.c_uint32 * 3),
    ('max_vertices_out', ctypes.c_uint16),
    ('max_primitives_out', ctypes.c_uint16),
    ('primitive_type', mesa_prim),
    ('nv', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

union_shader_info_0._pack_ = 1 # source:False
union_shader_info_0._fields_ = [
    ('vs', struct_shader_info_0_vs),
    ('gs', struct_shader_info_0_gs),
    ('fs', struct_shader_info_0_fs),
    ('cs', struct_shader_info_0_cs),
    ('tess', struct_shader_info_0_tess),
    ('mesh', struct_shader_info_0_mesh),
    ('PADDING_0', ctypes.c_ubyte * 24),
]

struct_shader_info._pack_ = 1 # source:False
struct_shader_info._anonymous_ = ('_0',)
struct_shader_info._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('label', ctypes.POINTER(ctypes.c_char)),
    ('internal', ctypes.c_bool),
    ('source_blake3', ctypes.c_ubyte * 32),
    ('stage', mesa_shader_stage, 8),
    ('prev_stage', mesa_shader_stage, 8),
    ('next_stage', mesa_shader_stage, 8),
    ('prev_stage_has_xfb', mesa_shader_stage, 8),
    ('num_textures', ctypes.c_ubyte),
    ('num_ubos', ctypes.c_ubyte),
    ('num_abos', ctypes.c_ubyte),
    ('num_ssbos', ctypes.c_ubyte),
    ('num_images', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('inputs_read', ctypes.c_uint64),
    ('dual_slot_inputs', ctypes.c_uint64),
    ('outputs_written', ctypes.c_uint64),
    ('outputs_read', ctypes.c_uint64),
    ('system_values_read', ctypes.c_uint32 * 4),
    ('perspective_varyings', ctypes.c_uint64),
    ('linear_varyings', ctypes.c_uint64),
    ('per_primitive_inputs', ctypes.c_uint64),
    ('per_primitive_outputs', ctypes.c_uint64),
    ('per_view_outputs', ctypes.c_uint64),
    ('view_mask', ctypes.c_uint32),
    ('inputs_read_16bit', ctypes.c_uint16),
    ('outputs_written_16bit', ctypes.c_uint16),
    ('outputs_read_16bit', ctypes.c_uint16),
    ('inputs_read_indirectly_16bit', ctypes.c_uint16),
    ('outputs_read_indirectly_16bit', ctypes.c_uint16),
    ('outputs_written_indirectly_16bit', ctypes.c_uint16),
    ('patch_inputs_read', ctypes.c_uint32),
    ('patch_outputs_written', ctypes.c_uint32),
    ('patch_outputs_read', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('inputs_read_indirectly', ctypes.c_uint64),
    ('outputs_read_indirectly', ctypes.c_uint64),
    ('outputs_written_indirectly', ctypes.c_uint64),
    ('patch_inputs_read_indirectly', ctypes.c_uint32),
    ('patch_outputs_read_indirectly', ctypes.c_uint32),
    ('patch_outputs_written_indirectly', ctypes.c_uint32),
    ('textures_used', ctypes.c_uint32 * 4),
    ('textures_used_by_txf', ctypes.c_uint32 * 4),
    ('samplers_used', ctypes.c_uint32 * 1),
    ('images_used', ctypes.c_uint32 * 2),
    ('image_buffers', ctypes.c_uint32 * 2),
    ('msaa_images', ctypes.c_uint32 * 2),
    ('float_controls_execution_mode', ctypes.c_uint32),
    ('shared_size', ctypes.c_uint32),
    ('task_payload_size', ctypes.c_uint32),
    ('ray_queries', ctypes.c_uint32),
    ('workgroup_size', ctypes.c_uint16 * 3),
    ('PADDING_2', ctypes.c_ubyte * 2),
    ('subgroup_size', gl_subgroup_size),
    ('num_subgroups', ctypes.c_ubyte),
    ('uses_wide_subgroup_intrinsics', ctypes.c_bool),
    ('xfb_stride', ctypes.c_ubyte * 4),
    ('inlinable_uniform_dw_offsets', ctypes.c_uint16 * 4),
    ('num_inlinable_uniforms', ctypes.c_ubyte, 4),
    ('clip_distance_array_size', ctypes.c_ubyte, 4),
    ('cull_distance_array_size', ctypes.c_ubyte, 4),
    ('uses_texture_gather', ctypes.c_ubyte, 1),
    ('uses_resource_info_query', ctypes.c_ubyte, 1),
    ('PADDING_3', ctypes.c_uint8, 2),
    ('bit_sizes_float', ctypes.c_ubyte, 8),
    ('bit_sizes_int', ctypes.c_ubyte),
    ('first_ubo_is_default_ubo', ctypes.c_bool, 1),
    ('separate_shader', ctypes.c_bool, 1),
    ('known_interpolation_qualifiers', ctypes.c_bool, 1),
    ('has_transform_feedback_varyings', ctypes.c_bool, 1),
    ('flrp_lowered', ctypes.c_bool, 1),
    ('io_lowered', ctypes.c_bool, 1),
    ('var_copies_lowered', ctypes.c_bool, 1),
    ('writes_memory', ctypes.c_bool, 1),
    ('layer_viewport_relative', ctypes.c_bool, 1),
    ('uses_control_barrier', ctypes.c_bool, 1),
    ('uses_memory_barrier', ctypes.c_bool, 1),
    ('uses_bindless', ctypes.c_bool, 1),
    ('shared_memory_explicit_layout', ctypes.c_bool, 1),
    ('zero_initialize_shared_memory', ctypes.c_bool, 1),
    ('workgroup_size_variable', ctypes.c_bool, 1),
    ('uses_printf', ctypes.c_bool, 1),
    ('maximally_reconverges', ctypes.c_bool, 1),
    ('use_aco_amd', ctypes.c_bool, 1),
    ('use_lowered_image_to_global', ctypes.c_bool, 1),
    ('PADDING_4', ctypes.c_uint8, 5),
    ('use_legacy_math_rules', ctypes.c_bool, 8),
    ('derivative_group', gl_derivative_group, 2),
    ('PADDING_5', ctypes.c_uint64, 46),
    ('_0', union_shader_info_0),
]

struct_nir_shader._pack_ = 1 # source:False
struct_nir_shader._fields_ = [
    ('gctx', ctypes.POINTER(struct_gc_ctx)),
    ('variables', struct_exec_list),
    ('options', ctypes.POINTER(struct_nir_shader_compiler_options)),
    ('info', struct_shader_info),
    ('functions', struct_exec_list),
    ('num_inputs', ctypes.c_uint32),
    ('num_uniforms', ctypes.c_uint32),
    ('num_outputs', ctypes.c_uint32),
    ('global_mem_size', ctypes.c_uint32),
    ('scratch_size', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('constant_data', ctypes.POINTER(None)),
    ('constant_data_size', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('xfb_info', ctypes.POINTER(struct_nir_xfb_info)),
    ('printf_info_count', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('printf_info', ctypes.POINTER(struct_u_printf_info)),
    ('has_debug_info', ctypes.c_bool),
    ('PADDING_3', ctypes.c_ubyte * 7),
]

nir_shader_compiler_options = struct_nir_shader_compiler_options
u_printf_info = struct_u_printf_info
nir_debug = 0 # Variable ctypes.c_uint32
nir_debug_print_shader = [] # Variable ctypes.c_bool * 15
nir_component_mask_t = ctypes.c_uint16
try:
    nir_round_up_components = _libraries['FIXME_STUB'].nir_round_up_components
    nir_round_up_components.restype = ctypes.c_uint32
    nir_round_up_components.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_round_down_components = _libraries['FIXME_STUB'].nir_round_down_components
    nir_round_down_components.restype = ctypes.c_uint32
    nir_round_down_components.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_component_mask = _libraries['FIXME_STUB'].nir_component_mask
    nir_component_mask.restype = nir_component_mask_t
    nir_component_mask.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_process_debug_variable = _libraries['FIXME_STUB'].nir_process_debug_variable
    nir_process_debug_variable.restype = None
    nir_process_debug_variable.argtypes = []
except AttributeError:
    pass
try:
    nir_component_mask_can_reinterpret = _libraries['FIXME_STUB'].nir_component_mask_can_reinterpret
    nir_component_mask_can_reinterpret.restype = ctypes.c_bool
    nir_component_mask_can_reinterpret.argtypes = [nir_component_mask_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_component_mask_reinterpret = _libraries['FIXME_STUB'].nir_component_mask_reinterpret
    nir_component_mask_reinterpret.restype = nir_component_mask_t
    nir_component_mask_reinterpret.argtypes = [nir_component_mask_t, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
class struct_nir_state_slot(Structure):
    pass

struct_nir_state_slot._pack_ = 1 # source:False
struct_nir_state_slot._fields_ = [
    ('tokens', ctypes.c_int16 * 4),
]

nir_state_slot = struct_nir_state_slot

# values for enumeration 'c__EA_nir_rounding_mode'
c__EA_nir_rounding_mode__enumvalues = {
    0: 'nir_rounding_mode_undef',
    1: 'nir_rounding_mode_rtne',
    2: 'nir_rounding_mode_ru',
    3: 'nir_rounding_mode_rd',
    4: 'nir_rounding_mode_rtz',
}
nir_rounding_mode_undef = 0
nir_rounding_mode_rtne = 1
nir_rounding_mode_ru = 2
nir_rounding_mode_rd = 3
nir_rounding_mode_rtz = 4
c__EA_nir_rounding_mode = ctypes.c_uint32 # enum
nir_rounding_mode = c__EA_nir_rounding_mode
nir_rounding_mode__enumvalues = c__EA_nir_rounding_mode__enumvalues

# values for enumeration 'c__EA_nir_ray_query_value'
c__EA_nir_ray_query_value__enumvalues = {
    0: 'nir_ray_query_value_intersection_type',
    1: 'nir_ray_query_value_intersection_t',
    2: 'nir_ray_query_value_intersection_instance_custom_index',
    3: 'nir_ray_query_value_intersection_instance_id',
    4: 'nir_ray_query_value_intersection_instance_sbt_index',
    5: 'nir_ray_query_value_intersection_geometry_index',
    6: 'nir_ray_query_value_intersection_primitive_index',
    7: 'nir_ray_query_value_intersection_barycentrics',
    8: 'nir_ray_query_value_intersection_front_face',
    9: 'nir_ray_query_value_intersection_object_ray_direction',
    10: 'nir_ray_query_value_intersection_object_ray_origin',
    11: 'nir_ray_query_value_intersection_object_to_world',
    12: 'nir_ray_query_value_intersection_world_to_object',
    13: 'nir_ray_query_value_intersection_candidate_aabb_opaque',
    14: 'nir_ray_query_value_tmin',
    15: 'nir_ray_query_value_flags',
    16: 'nir_ray_query_value_world_ray_direction',
    17: 'nir_ray_query_value_world_ray_origin',
    18: 'nir_ray_query_value_intersection_triangle_vertex_positions',
}
nir_ray_query_value_intersection_type = 0
nir_ray_query_value_intersection_t = 1
nir_ray_query_value_intersection_instance_custom_index = 2
nir_ray_query_value_intersection_instance_id = 3
nir_ray_query_value_intersection_instance_sbt_index = 4
nir_ray_query_value_intersection_geometry_index = 5
nir_ray_query_value_intersection_primitive_index = 6
nir_ray_query_value_intersection_barycentrics = 7
nir_ray_query_value_intersection_front_face = 8
nir_ray_query_value_intersection_object_ray_direction = 9
nir_ray_query_value_intersection_object_ray_origin = 10
nir_ray_query_value_intersection_object_to_world = 11
nir_ray_query_value_intersection_world_to_object = 12
nir_ray_query_value_intersection_candidate_aabb_opaque = 13
nir_ray_query_value_tmin = 14
nir_ray_query_value_flags = 15
nir_ray_query_value_world_ray_direction = 16
nir_ray_query_value_world_ray_origin = 17
nir_ray_query_value_intersection_triangle_vertex_positions = 18
c__EA_nir_ray_query_value = ctypes.c_uint32 # enum
nir_ray_query_value = c__EA_nir_ray_query_value
nir_ray_query_value__enumvalues = c__EA_nir_ray_query_value__enumvalues

# values for enumeration 'c__EA_nir_resource_data_intel'
c__EA_nir_resource_data_intel__enumvalues = {
    1: 'nir_resource_intel_bindless',
    2: 'nir_resource_intel_pushable',
    4: 'nir_resource_intel_sampler',
    8: 'nir_resource_intel_non_uniform',
    16: 'nir_resource_intel_sampler_embedded',
}
nir_resource_intel_bindless = 1
nir_resource_intel_pushable = 2
nir_resource_intel_sampler = 4
nir_resource_intel_non_uniform = 8
nir_resource_intel_sampler_embedded = 16
c__EA_nir_resource_data_intel = ctypes.c_uint32 # enum
nir_resource_data_intel = c__EA_nir_resource_data_intel
nir_resource_data_intel__enumvalues = c__EA_nir_resource_data_intel__enumvalues

# values for enumeration 'c__EA_nir_preamble_class'
c__EA_nir_preamble_class__enumvalues = {
    0: 'nir_preamble_class_general',
    1: 'nir_preamble_class_image',
    2: 'nir_preamble_class_sampler',
    3: 'nir_preamble_num_classes',
}
nir_preamble_class_general = 0
nir_preamble_class_image = 1
nir_preamble_class_sampler = 2
nir_preamble_num_classes = 3
c__EA_nir_preamble_class = ctypes.c_uint32 # enum
nir_preamble_class = c__EA_nir_preamble_class
nir_preamble_class__enumvalues = c__EA_nir_preamble_class__enumvalues

# values for enumeration 'c__EA_nir_cmat_signed'
c__EA_nir_cmat_signed__enumvalues = {
    1: 'NIR_CMAT_A_SIGNED',
    2: 'NIR_CMAT_B_SIGNED',
    4: 'NIR_CMAT_C_SIGNED',
    8: 'NIR_CMAT_RESULT_SIGNED',
}
NIR_CMAT_A_SIGNED = 1
NIR_CMAT_B_SIGNED = 2
NIR_CMAT_C_SIGNED = 4
NIR_CMAT_RESULT_SIGNED = 8
c__EA_nir_cmat_signed = ctypes.c_uint32 # enum
nir_cmat_signed = c__EA_nir_cmat_signed
nir_cmat_signed__enumvalues = c__EA_nir_cmat_signed__enumvalues
class union_c__UA_nir_const_value(Union):
    pass

union_c__UA_nir_const_value._pack_ = 1 # source:False
union_c__UA_nir_const_value._fields_ = [
    ('b', ctypes.c_bool),
    ('f32', ctypes.c_float),
    ('f64', ctypes.c_double),
    ('i8', ctypes.c_byte),
    ('u8', ctypes.c_ubyte),
    ('i16', ctypes.c_int16),
    ('u16', ctypes.c_uint16),
    ('i32', ctypes.c_int32),
    ('u32', ctypes.c_uint32),
    ('i64', ctypes.c_int64),
    ('u64', ctypes.c_uint64),
]

nir_const_value = union_c__UA_nir_const_value
uint64_t = ctypes.c_uint64
try:
    nir_const_value_for_raw_uint = _libraries['FIXME_STUB'].nir_const_value_for_raw_uint
    nir_const_value_for_raw_uint.restype = nir_const_value
    nir_const_value_for_raw_uint.argtypes = [uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
int64_t = ctypes.c_int64
try:
    nir_const_value_for_int = _libraries['FIXME_STUB'].nir_const_value_for_int
    nir_const_value_for_int.restype = nir_const_value
    nir_const_value_for_int.argtypes = [int64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_for_uint = _libraries['FIXME_STUB'].nir_const_value_for_uint
    nir_const_value_for_uint.restype = nir_const_value
    nir_const_value_for_uint.argtypes = [uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_for_bool = _libraries['FIXME_STUB'].nir_const_value_for_bool
    nir_const_value_for_bool.restype = nir_const_value
    nir_const_value_for_bool.argtypes = [ctypes.c_bool, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_for_float = _libraries['FIXME_STUB'].nir_const_value_for_float
    nir_const_value_for_float.restype = nir_const_value
    nir_const_value_for_float.argtypes = [ctypes.c_double, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_as_int = _libraries['FIXME_STUB'].nir_const_value_as_int
    nir_const_value_as_int.restype = int64_t
    nir_const_value_as_int.argtypes = [nir_const_value, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_as_uint = _libraries['FIXME_STUB'].nir_const_value_as_uint
    nir_const_value_as_uint.restype = uint64_t
    nir_const_value_as_uint.argtypes = [nir_const_value, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_as_bool = _libraries['FIXME_STUB'].nir_const_value_as_bool
    nir_const_value_as_bool.restype = ctypes.c_bool
    nir_const_value_as_bool.argtypes = [nir_const_value, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_const_value_as_float = _libraries['FIXME_STUB'].nir_const_value_as_float
    nir_const_value_as_float.restype = ctypes.c_double
    nir_const_value_as_float.argtypes = [nir_const_value, ctypes.c_uint32]
except AttributeError:
    pass
class struct_nir_constant(Structure):
    pass

struct_nir_constant._pack_ = 1 # source:False
struct_nir_constant._fields_ = [
    ('values', union_c__UA_nir_const_value * 16),
    ('is_null_constant', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('num_elements', ctypes.c_uint32),
    ('elements', ctypes.POINTER(ctypes.POINTER(struct_nir_constant))),
]

nir_constant = struct_nir_constant

# values for enumeration 'c__EA_nir_depth_layout'
c__EA_nir_depth_layout__enumvalues = {
    0: 'nir_depth_layout_none',
    1: 'nir_depth_layout_any',
    2: 'nir_depth_layout_greater',
    3: 'nir_depth_layout_less',
    4: 'nir_depth_layout_unchanged',
}
nir_depth_layout_none = 0
nir_depth_layout_any = 1
nir_depth_layout_greater = 2
nir_depth_layout_less = 3
nir_depth_layout_unchanged = 4
c__EA_nir_depth_layout = ctypes.c_uint32 # enum
nir_depth_layout = c__EA_nir_depth_layout
nir_depth_layout__enumvalues = c__EA_nir_depth_layout__enumvalues

# values for enumeration 'c__EA_nir_var_declaration_type'
c__EA_nir_var_declaration_type__enumvalues = {
    0: 'nir_var_declared_normally',
    1: 'nir_var_declared_implicitly',
    2: 'nir_var_hidden',
}
nir_var_declared_normally = 0
nir_var_declared_implicitly = 1
nir_var_hidden = 2
c__EA_nir_var_declaration_type = ctypes.c_uint32 # enum
nir_var_declaration_type = c__EA_nir_var_declaration_type
nir_var_declaration_type__enumvalues = c__EA_nir_var_declaration_type__enumvalues
class struct_nir_variable_data(Structure):
    pass

class union_nir_variable_data_0(Union):
    pass

class struct_nir_variable_data_0_image(Structure):
    pass


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
    240: 'PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM',
    241: 'PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM',
    242: 'PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM',
    243: 'PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM',
    244: 'PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM',
    245: 'PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM',
    246: 'PIPE_FORMAT_Y16_U16_V16_420_UNORM',
    247: 'PIPE_FORMAT_Y16_U16_V16_422_UNORM',
    248: 'PIPE_FORMAT_Y16_U16V16_422_UNORM',
    249: 'PIPE_FORMAT_Y16_U16_V16_444_UNORM',
    250: 'PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED',
    251: 'PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED',
    252: 'PIPE_FORMAT_A4R4_UNORM',
    253: 'PIPE_FORMAT_R4A4_UNORM',
    254: 'PIPE_FORMAT_R8A8_UNORM',
    255: 'PIPE_FORMAT_A8R8_UNORM',
    256: 'PIPE_FORMAT_A8_UINT',
    257: 'PIPE_FORMAT_I8_UINT',
    258: 'PIPE_FORMAT_L8_UINT',
    259: 'PIPE_FORMAT_L8A8_UINT',
    260: 'PIPE_FORMAT_A8_SINT',
    261: 'PIPE_FORMAT_I8_SINT',
    262: 'PIPE_FORMAT_L8_SINT',
    263: 'PIPE_FORMAT_L8A8_SINT',
    264: 'PIPE_FORMAT_A16_UINT',
    265: 'PIPE_FORMAT_I16_UINT',
    266: 'PIPE_FORMAT_L16_UINT',
    267: 'PIPE_FORMAT_L16A16_UINT',
    268: 'PIPE_FORMAT_A16_SINT',
    269: 'PIPE_FORMAT_I16_SINT',
    270: 'PIPE_FORMAT_L16_SINT',
    271: 'PIPE_FORMAT_L16A16_SINT',
    272: 'PIPE_FORMAT_A32_UINT',
    273: 'PIPE_FORMAT_I32_UINT',
    274: 'PIPE_FORMAT_L32_UINT',
    275: 'PIPE_FORMAT_L32A32_UINT',
    276: 'PIPE_FORMAT_A32_SINT',
    277: 'PIPE_FORMAT_I32_SINT',
    278: 'PIPE_FORMAT_L32_SINT',
    279: 'PIPE_FORMAT_L32A32_SINT',
    280: 'PIPE_FORMAT_A8R8G8B8_UINT',
    281: 'PIPE_FORMAT_A8B8G8R8_UINT',
    282: 'PIPE_FORMAT_A2R10G10B10_UINT',
    283: 'PIPE_FORMAT_A2B10G10R10_UINT',
    284: 'PIPE_FORMAT_R5G6B5_UINT',
    285: 'PIPE_FORMAT_B5G6R5_UINT',
    286: 'PIPE_FORMAT_R5G5B5A1_UINT',
    287: 'PIPE_FORMAT_B5G5R5A1_UINT',
    288: 'PIPE_FORMAT_A1R5G5B5_UINT',
    289: 'PIPE_FORMAT_A1B5G5R5_UINT',
    290: 'PIPE_FORMAT_R4G4B4A4_UINT',
    291: 'PIPE_FORMAT_B4G4R4A4_UINT',
    292: 'PIPE_FORMAT_A4R4G4B4_UINT',
    293: 'PIPE_FORMAT_A4B4G4R4_UINT',
    294: 'PIPE_FORMAT_R3G3B2_UINT',
    295: 'PIPE_FORMAT_B2G3R3_UINT',
    296: 'PIPE_FORMAT_ETC1_RGB8',
    297: 'PIPE_FORMAT_R8G8_R8B8_UNORM',
    298: 'PIPE_FORMAT_R8B8_R8G8_UNORM',
    299: 'PIPE_FORMAT_G8R8_B8R8_UNORM',
    300: 'PIPE_FORMAT_B8R8_G8R8_UNORM',
    301: 'PIPE_FORMAT_G8B8_G8R8_UNORM',
    302: 'PIPE_FORMAT_B8G8_R8G8_UNORM',
    303: 'PIPE_FORMAT_R8G8B8X8_SNORM',
    304: 'PIPE_FORMAT_R8G8B8X8_SRGB',
    305: 'PIPE_FORMAT_R8G8B8X8_UINT',
    306: 'PIPE_FORMAT_R8G8B8X8_SINT',
    307: 'PIPE_FORMAT_B10G10R10X2_UNORM',
    308: 'PIPE_FORMAT_R16G16B16X16_UNORM',
    309: 'PIPE_FORMAT_R16G16B16X16_SNORM',
    310: 'PIPE_FORMAT_R16G16B16X16_FLOAT',
    311: 'PIPE_FORMAT_R16G16B16X16_UINT',
    312: 'PIPE_FORMAT_R16G16B16X16_SINT',
    313: 'PIPE_FORMAT_R32G32B32X32_FLOAT',
    314: 'PIPE_FORMAT_R32G32B32X32_UINT',
    315: 'PIPE_FORMAT_R32G32B32X32_SINT',
    316: 'PIPE_FORMAT_R8A8_SNORM',
    317: 'PIPE_FORMAT_R16A16_UNORM',
    318: 'PIPE_FORMAT_R16A16_SNORM',
    319: 'PIPE_FORMAT_R16A16_FLOAT',
    320: 'PIPE_FORMAT_R32A32_FLOAT',
    321: 'PIPE_FORMAT_R8A8_UINT',
    322: 'PIPE_FORMAT_R8A8_SINT',
    323: 'PIPE_FORMAT_R16A16_UINT',
    324: 'PIPE_FORMAT_R16A16_SINT',
    325: 'PIPE_FORMAT_R32A32_UINT',
    326: 'PIPE_FORMAT_R32A32_SINT',
    327: 'PIPE_FORMAT_B5G6R5_SRGB',
    328: 'PIPE_FORMAT_BPTC_RGBA_UNORM',
    329: 'PIPE_FORMAT_BPTC_SRGBA',
    330: 'PIPE_FORMAT_BPTC_RGB_FLOAT',
    331: 'PIPE_FORMAT_BPTC_RGB_UFLOAT',
    332: 'PIPE_FORMAT_G8R8_UNORM',
    333: 'PIPE_FORMAT_G8R8_SNORM',
    334: 'PIPE_FORMAT_G16R16_UNORM',
    335: 'PIPE_FORMAT_G16R16_SNORM',
    336: 'PIPE_FORMAT_A8B8G8R8_SNORM',
    337: 'PIPE_FORMAT_X8B8G8R8_SNORM',
    338: 'PIPE_FORMAT_ETC2_RGB8',
    339: 'PIPE_FORMAT_ETC2_SRGB8',
    340: 'PIPE_FORMAT_ETC2_RGB8A1',
    341: 'PIPE_FORMAT_ETC2_SRGB8A1',
    342: 'PIPE_FORMAT_ETC2_RGBA8',
    343: 'PIPE_FORMAT_ETC2_SRGBA8',
    344: 'PIPE_FORMAT_ETC2_R11_UNORM',
    345: 'PIPE_FORMAT_ETC2_R11_SNORM',
    346: 'PIPE_FORMAT_ETC2_RG11_UNORM',
    347: 'PIPE_FORMAT_ETC2_RG11_SNORM',
    348: 'PIPE_FORMAT_ASTC_4x4',
    349: 'PIPE_FORMAT_ASTC_5x4',
    350: 'PIPE_FORMAT_ASTC_5x5',
    351: 'PIPE_FORMAT_ASTC_6x5',
    352: 'PIPE_FORMAT_ASTC_6x6',
    353: 'PIPE_FORMAT_ASTC_8x5',
    354: 'PIPE_FORMAT_ASTC_8x6',
    355: 'PIPE_FORMAT_ASTC_8x8',
    356: 'PIPE_FORMAT_ASTC_10x5',
    357: 'PIPE_FORMAT_ASTC_10x6',
    358: 'PIPE_FORMAT_ASTC_10x8',
    359: 'PIPE_FORMAT_ASTC_10x10',
    360: 'PIPE_FORMAT_ASTC_12x10',
    361: 'PIPE_FORMAT_ASTC_12x12',
    362: 'PIPE_FORMAT_ASTC_4x4_SRGB',
    363: 'PIPE_FORMAT_ASTC_5x4_SRGB',
    364: 'PIPE_FORMAT_ASTC_5x5_SRGB',
    365: 'PIPE_FORMAT_ASTC_6x5_SRGB',
    366: 'PIPE_FORMAT_ASTC_6x6_SRGB',
    367: 'PIPE_FORMAT_ASTC_8x5_SRGB',
    368: 'PIPE_FORMAT_ASTC_8x6_SRGB',
    369: 'PIPE_FORMAT_ASTC_8x8_SRGB',
    370: 'PIPE_FORMAT_ASTC_10x5_SRGB',
    371: 'PIPE_FORMAT_ASTC_10x6_SRGB',
    372: 'PIPE_FORMAT_ASTC_10x8_SRGB',
    373: 'PIPE_FORMAT_ASTC_10x10_SRGB',
    374: 'PIPE_FORMAT_ASTC_12x10_SRGB',
    375: 'PIPE_FORMAT_ASTC_12x12_SRGB',
    376: 'PIPE_FORMAT_ASTC_3x3x3',
    377: 'PIPE_FORMAT_ASTC_4x3x3',
    378: 'PIPE_FORMAT_ASTC_4x4x3',
    379: 'PIPE_FORMAT_ASTC_4x4x4',
    380: 'PIPE_FORMAT_ASTC_5x4x4',
    381: 'PIPE_FORMAT_ASTC_5x5x4',
    382: 'PIPE_FORMAT_ASTC_5x5x5',
    383: 'PIPE_FORMAT_ASTC_6x5x5',
    384: 'PIPE_FORMAT_ASTC_6x6x5',
    385: 'PIPE_FORMAT_ASTC_6x6x6',
    386: 'PIPE_FORMAT_ASTC_3x3x3_SRGB',
    387: 'PIPE_FORMAT_ASTC_4x3x3_SRGB',
    388: 'PIPE_FORMAT_ASTC_4x4x3_SRGB',
    389: 'PIPE_FORMAT_ASTC_4x4x4_SRGB',
    390: 'PIPE_FORMAT_ASTC_5x4x4_SRGB',
    391: 'PIPE_FORMAT_ASTC_5x5x4_SRGB',
    392: 'PIPE_FORMAT_ASTC_5x5x5_SRGB',
    393: 'PIPE_FORMAT_ASTC_6x5x5_SRGB',
    394: 'PIPE_FORMAT_ASTC_6x6x5_SRGB',
    395: 'PIPE_FORMAT_ASTC_6x6x6_SRGB',
    396: 'PIPE_FORMAT_ASTC_4x4_FLOAT',
    397: 'PIPE_FORMAT_ASTC_5x4_FLOAT',
    398: 'PIPE_FORMAT_ASTC_5x5_FLOAT',
    399: 'PIPE_FORMAT_ASTC_6x5_FLOAT',
    400: 'PIPE_FORMAT_ASTC_6x6_FLOAT',
    401: 'PIPE_FORMAT_ASTC_8x5_FLOAT',
    402: 'PIPE_FORMAT_ASTC_8x6_FLOAT',
    403: 'PIPE_FORMAT_ASTC_8x8_FLOAT',
    404: 'PIPE_FORMAT_ASTC_10x5_FLOAT',
    405: 'PIPE_FORMAT_ASTC_10x6_FLOAT',
    406: 'PIPE_FORMAT_ASTC_10x8_FLOAT',
    407: 'PIPE_FORMAT_ASTC_10x10_FLOAT',
    408: 'PIPE_FORMAT_ASTC_12x10_FLOAT',
    409: 'PIPE_FORMAT_ASTC_12x12_FLOAT',
    410: 'PIPE_FORMAT_FXT1_RGB',
    411: 'PIPE_FORMAT_FXT1_RGBA',
    412: 'PIPE_FORMAT_P010',
    413: 'PIPE_FORMAT_P012',
    414: 'PIPE_FORMAT_P016',
    415: 'PIPE_FORMAT_P030',
    416: 'PIPE_FORMAT_Y210',
    417: 'PIPE_FORMAT_Y212',
    418: 'PIPE_FORMAT_Y216',
    419: 'PIPE_FORMAT_Y410',
    420: 'PIPE_FORMAT_Y412',
    421: 'PIPE_FORMAT_Y416',
    422: 'PIPE_FORMAT_R10G10B10X2_UNORM',
    423: 'PIPE_FORMAT_A1R5G5B5_UNORM',
    424: 'PIPE_FORMAT_A1B5G5R5_UNORM',
    425: 'PIPE_FORMAT_X1B5G5R5_UNORM',
    426: 'PIPE_FORMAT_R5G5B5A1_UNORM',
    427: 'PIPE_FORMAT_A4R4G4B4_UNORM',
    428: 'PIPE_FORMAT_A4B4G4R4_UNORM',
    429: 'PIPE_FORMAT_G8R8_SINT',
    430: 'PIPE_FORMAT_A8B8G8R8_SINT',
    431: 'PIPE_FORMAT_X8B8G8R8_SINT',
    432: 'PIPE_FORMAT_ATC_RGB',
    433: 'PIPE_FORMAT_ATC_RGBA_EXPLICIT',
    434: 'PIPE_FORMAT_ATC_RGBA_INTERPOLATED',
    435: 'PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8',
    436: 'PIPE_FORMAT_AYUV',
    437: 'PIPE_FORMAT_XYUV',
    438: 'PIPE_FORMAT_R8G8B8_420_UNORM_PACKED',
    439: 'PIPE_FORMAT_R8_G8B8_420_UNORM',
    440: 'PIPE_FORMAT_R8_B8G8_420_UNORM',
    441: 'PIPE_FORMAT_G8_B8R8_420_UNORM',
    442: 'PIPE_FORMAT_R10G10B10_420_UNORM_PACKED',
    443: 'PIPE_FORMAT_R10_G10B10_420_UNORM',
    444: 'PIPE_FORMAT_R10_G10B10_422_UNORM',
    445: 'PIPE_FORMAT_R8_G8_B8_420_UNORM',
    446: 'PIPE_FORMAT_R8_B8_G8_420_UNORM',
    447: 'PIPE_FORMAT_G8_B8_R8_420_UNORM',
    448: 'PIPE_FORMAT_R8_G8B8_422_UNORM',
    449: 'PIPE_FORMAT_R8_B8G8_422_UNORM',
    450: 'PIPE_FORMAT_G8_B8R8_422_UNORM',
    451: 'PIPE_FORMAT_R8_G8_B8_UNORM',
    452: 'PIPE_FORMAT_Y8_UNORM',
    453: 'PIPE_FORMAT_B8G8R8X8_SNORM',
    454: 'PIPE_FORMAT_B8G8R8X8_UINT',
    455: 'PIPE_FORMAT_B8G8R8X8_SINT',
    456: 'PIPE_FORMAT_A8R8G8B8_SNORM',
    457: 'PIPE_FORMAT_A8R8G8B8_SINT',
    458: 'PIPE_FORMAT_X8R8G8B8_SNORM',
    459: 'PIPE_FORMAT_X8R8G8B8_SINT',
    460: 'PIPE_FORMAT_R5G5B5X1_UNORM',
    461: 'PIPE_FORMAT_X1R5G5B5_UNORM',
    462: 'PIPE_FORMAT_R4G4B4X4_UNORM',
    463: 'PIPE_FORMAT_B10G10R10X2_SNORM',
    464: 'PIPE_FORMAT_R5G6B5_SRGB',
    465: 'PIPE_FORMAT_R10G10B10X2_SINT',
    466: 'PIPE_FORMAT_B10G10R10X2_SINT',
    467: 'PIPE_FORMAT_G16R16_SINT',
    468: 'PIPE_FORMAT_COUNT',
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
PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM = 240
PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM = 241
PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM = 242
PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM = 243
PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM = 244
PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM = 245
PIPE_FORMAT_Y16_U16_V16_420_UNORM = 246
PIPE_FORMAT_Y16_U16_V16_422_UNORM = 247
PIPE_FORMAT_Y16_U16V16_422_UNORM = 248
PIPE_FORMAT_Y16_U16_V16_444_UNORM = 249
PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED = 250
PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED = 251
PIPE_FORMAT_A4R4_UNORM = 252
PIPE_FORMAT_R4A4_UNORM = 253
PIPE_FORMAT_R8A8_UNORM = 254
PIPE_FORMAT_A8R8_UNORM = 255
PIPE_FORMAT_A8_UINT = 256
PIPE_FORMAT_I8_UINT = 257
PIPE_FORMAT_L8_UINT = 258
PIPE_FORMAT_L8A8_UINT = 259
PIPE_FORMAT_A8_SINT = 260
PIPE_FORMAT_I8_SINT = 261
PIPE_FORMAT_L8_SINT = 262
PIPE_FORMAT_L8A8_SINT = 263
PIPE_FORMAT_A16_UINT = 264
PIPE_FORMAT_I16_UINT = 265
PIPE_FORMAT_L16_UINT = 266
PIPE_FORMAT_L16A16_UINT = 267
PIPE_FORMAT_A16_SINT = 268
PIPE_FORMAT_I16_SINT = 269
PIPE_FORMAT_L16_SINT = 270
PIPE_FORMAT_L16A16_SINT = 271
PIPE_FORMAT_A32_UINT = 272
PIPE_FORMAT_I32_UINT = 273
PIPE_FORMAT_L32_UINT = 274
PIPE_FORMAT_L32A32_UINT = 275
PIPE_FORMAT_A32_SINT = 276
PIPE_FORMAT_I32_SINT = 277
PIPE_FORMAT_L32_SINT = 278
PIPE_FORMAT_L32A32_SINT = 279
PIPE_FORMAT_A8R8G8B8_UINT = 280
PIPE_FORMAT_A8B8G8R8_UINT = 281
PIPE_FORMAT_A2R10G10B10_UINT = 282
PIPE_FORMAT_A2B10G10R10_UINT = 283
PIPE_FORMAT_R5G6B5_UINT = 284
PIPE_FORMAT_B5G6R5_UINT = 285
PIPE_FORMAT_R5G5B5A1_UINT = 286
PIPE_FORMAT_B5G5R5A1_UINT = 287
PIPE_FORMAT_A1R5G5B5_UINT = 288
PIPE_FORMAT_A1B5G5R5_UINT = 289
PIPE_FORMAT_R4G4B4A4_UINT = 290
PIPE_FORMAT_B4G4R4A4_UINT = 291
PIPE_FORMAT_A4R4G4B4_UINT = 292
PIPE_FORMAT_A4B4G4R4_UINT = 293
PIPE_FORMAT_R3G3B2_UINT = 294
PIPE_FORMAT_B2G3R3_UINT = 295
PIPE_FORMAT_ETC1_RGB8 = 296
PIPE_FORMAT_R8G8_R8B8_UNORM = 297
PIPE_FORMAT_R8B8_R8G8_UNORM = 298
PIPE_FORMAT_G8R8_B8R8_UNORM = 299
PIPE_FORMAT_B8R8_G8R8_UNORM = 300
PIPE_FORMAT_G8B8_G8R8_UNORM = 301
PIPE_FORMAT_B8G8_R8G8_UNORM = 302
PIPE_FORMAT_R8G8B8X8_SNORM = 303
PIPE_FORMAT_R8G8B8X8_SRGB = 304
PIPE_FORMAT_R8G8B8X8_UINT = 305
PIPE_FORMAT_R8G8B8X8_SINT = 306
PIPE_FORMAT_B10G10R10X2_UNORM = 307
PIPE_FORMAT_R16G16B16X16_UNORM = 308
PIPE_FORMAT_R16G16B16X16_SNORM = 309
PIPE_FORMAT_R16G16B16X16_FLOAT = 310
PIPE_FORMAT_R16G16B16X16_UINT = 311
PIPE_FORMAT_R16G16B16X16_SINT = 312
PIPE_FORMAT_R32G32B32X32_FLOAT = 313
PIPE_FORMAT_R32G32B32X32_UINT = 314
PIPE_FORMAT_R32G32B32X32_SINT = 315
PIPE_FORMAT_R8A8_SNORM = 316
PIPE_FORMAT_R16A16_UNORM = 317
PIPE_FORMAT_R16A16_SNORM = 318
PIPE_FORMAT_R16A16_FLOAT = 319
PIPE_FORMAT_R32A32_FLOAT = 320
PIPE_FORMAT_R8A8_UINT = 321
PIPE_FORMAT_R8A8_SINT = 322
PIPE_FORMAT_R16A16_UINT = 323
PIPE_FORMAT_R16A16_SINT = 324
PIPE_FORMAT_R32A32_UINT = 325
PIPE_FORMAT_R32A32_SINT = 326
PIPE_FORMAT_B5G6R5_SRGB = 327
PIPE_FORMAT_BPTC_RGBA_UNORM = 328
PIPE_FORMAT_BPTC_SRGBA = 329
PIPE_FORMAT_BPTC_RGB_FLOAT = 330
PIPE_FORMAT_BPTC_RGB_UFLOAT = 331
PIPE_FORMAT_G8R8_UNORM = 332
PIPE_FORMAT_G8R8_SNORM = 333
PIPE_FORMAT_G16R16_UNORM = 334
PIPE_FORMAT_G16R16_SNORM = 335
PIPE_FORMAT_A8B8G8R8_SNORM = 336
PIPE_FORMAT_X8B8G8R8_SNORM = 337
PIPE_FORMAT_ETC2_RGB8 = 338
PIPE_FORMAT_ETC2_SRGB8 = 339
PIPE_FORMAT_ETC2_RGB8A1 = 340
PIPE_FORMAT_ETC2_SRGB8A1 = 341
PIPE_FORMAT_ETC2_RGBA8 = 342
PIPE_FORMAT_ETC2_SRGBA8 = 343
PIPE_FORMAT_ETC2_R11_UNORM = 344
PIPE_FORMAT_ETC2_R11_SNORM = 345
PIPE_FORMAT_ETC2_RG11_UNORM = 346
PIPE_FORMAT_ETC2_RG11_SNORM = 347
PIPE_FORMAT_ASTC_4x4 = 348
PIPE_FORMAT_ASTC_5x4 = 349
PIPE_FORMAT_ASTC_5x5 = 350
PIPE_FORMAT_ASTC_6x5 = 351
PIPE_FORMAT_ASTC_6x6 = 352
PIPE_FORMAT_ASTC_8x5 = 353
PIPE_FORMAT_ASTC_8x6 = 354
PIPE_FORMAT_ASTC_8x8 = 355
PIPE_FORMAT_ASTC_10x5 = 356
PIPE_FORMAT_ASTC_10x6 = 357
PIPE_FORMAT_ASTC_10x8 = 358
PIPE_FORMAT_ASTC_10x10 = 359
PIPE_FORMAT_ASTC_12x10 = 360
PIPE_FORMAT_ASTC_12x12 = 361
PIPE_FORMAT_ASTC_4x4_SRGB = 362
PIPE_FORMAT_ASTC_5x4_SRGB = 363
PIPE_FORMAT_ASTC_5x5_SRGB = 364
PIPE_FORMAT_ASTC_6x5_SRGB = 365
PIPE_FORMAT_ASTC_6x6_SRGB = 366
PIPE_FORMAT_ASTC_8x5_SRGB = 367
PIPE_FORMAT_ASTC_8x6_SRGB = 368
PIPE_FORMAT_ASTC_8x8_SRGB = 369
PIPE_FORMAT_ASTC_10x5_SRGB = 370
PIPE_FORMAT_ASTC_10x6_SRGB = 371
PIPE_FORMAT_ASTC_10x8_SRGB = 372
PIPE_FORMAT_ASTC_10x10_SRGB = 373
PIPE_FORMAT_ASTC_12x10_SRGB = 374
PIPE_FORMAT_ASTC_12x12_SRGB = 375
PIPE_FORMAT_ASTC_3x3x3 = 376
PIPE_FORMAT_ASTC_4x3x3 = 377
PIPE_FORMAT_ASTC_4x4x3 = 378
PIPE_FORMAT_ASTC_4x4x4 = 379
PIPE_FORMAT_ASTC_5x4x4 = 380
PIPE_FORMAT_ASTC_5x5x4 = 381
PIPE_FORMAT_ASTC_5x5x5 = 382
PIPE_FORMAT_ASTC_6x5x5 = 383
PIPE_FORMAT_ASTC_6x6x5 = 384
PIPE_FORMAT_ASTC_6x6x6 = 385
PIPE_FORMAT_ASTC_3x3x3_SRGB = 386
PIPE_FORMAT_ASTC_4x3x3_SRGB = 387
PIPE_FORMAT_ASTC_4x4x3_SRGB = 388
PIPE_FORMAT_ASTC_4x4x4_SRGB = 389
PIPE_FORMAT_ASTC_5x4x4_SRGB = 390
PIPE_FORMAT_ASTC_5x5x4_SRGB = 391
PIPE_FORMAT_ASTC_5x5x5_SRGB = 392
PIPE_FORMAT_ASTC_6x5x5_SRGB = 393
PIPE_FORMAT_ASTC_6x6x5_SRGB = 394
PIPE_FORMAT_ASTC_6x6x6_SRGB = 395
PIPE_FORMAT_ASTC_4x4_FLOAT = 396
PIPE_FORMAT_ASTC_5x4_FLOAT = 397
PIPE_FORMAT_ASTC_5x5_FLOAT = 398
PIPE_FORMAT_ASTC_6x5_FLOAT = 399
PIPE_FORMAT_ASTC_6x6_FLOAT = 400
PIPE_FORMAT_ASTC_8x5_FLOAT = 401
PIPE_FORMAT_ASTC_8x6_FLOAT = 402
PIPE_FORMAT_ASTC_8x8_FLOAT = 403
PIPE_FORMAT_ASTC_10x5_FLOAT = 404
PIPE_FORMAT_ASTC_10x6_FLOAT = 405
PIPE_FORMAT_ASTC_10x8_FLOAT = 406
PIPE_FORMAT_ASTC_10x10_FLOAT = 407
PIPE_FORMAT_ASTC_12x10_FLOAT = 408
PIPE_FORMAT_ASTC_12x12_FLOAT = 409
PIPE_FORMAT_FXT1_RGB = 410
PIPE_FORMAT_FXT1_RGBA = 411
PIPE_FORMAT_P010 = 412
PIPE_FORMAT_P012 = 413
PIPE_FORMAT_P016 = 414
PIPE_FORMAT_P030 = 415
PIPE_FORMAT_Y210 = 416
PIPE_FORMAT_Y212 = 417
PIPE_FORMAT_Y216 = 418
PIPE_FORMAT_Y410 = 419
PIPE_FORMAT_Y412 = 420
PIPE_FORMAT_Y416 = 421
PIPE_FORMAT_R10G10B10X2_UNORM = 422
PIPE_FORMAT_A1R5G5B5_UNORM = 423
PIPE_FORMAT_A1B5G5R5_UNORM = 424
PIPE_FORMAT_X1B5G5R5_UNORM = 425
PIPE_FORMAT_R5G5B5A1_UNORM = 426
PIPE_FORMAT_A4R4G4B4_UNORM = 427
PIPE_FORMAT_A4B4G4R4_UNORM = 428
PIPE_FORMAT_G8R8_SINT = 429
PIPE_FORMAT_A8B8G8R8_SINT = 430
PIPE_FORMAT_X8B8G8R8_SINT = 431
PIPE_FORMAT_ATC_RGB = 432
PIPE_FORMAT_ATC_RGBA_EXPLICIT = 433
PIPE_FORMAT_ATC_RGBA_INTERPOLATED = 434
PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8 = 435
PIPE_FORMAT_AYUV = 436
PIPE_FORMAT_XYUV = 437
PIPE_FORMAT_R8G8B8_420_UNORM_PACKED = 438
PIPE_FORMAT_R8_G8B8_420_UNORM = 439
PIPE_FORMAT_R8_B8G8_420_UNORM = 440
PIPE_FORMAT_G8_B8R8_420_UNORM = 441
PIPE_FORMAT_R10G10B10_420_UNORM_PACKED = 442
PIPE_FORMAT_R10_G10B10_420_UNORM = 443
PIPE_FORMAT_R10_G10B10_422_UNORM = 444
PIPE_FORMAT_R8_G8_B8_420_UNORM = 445
PIPE_FORMAT_R8_B8_G8_420_UNORM = 446
PIPE_FORMAT_G8_B8_R8_420_UNORM = 447
PIPE_FORMAT_R8_G8B8_422_UNORM = 448
PIPE_FORMAT_R8_B8G8_422_UNORM = 449
PIPE_FORMAT_G8_B8R8_422_UNORM = 450
PIPE_FORMAT_R8_G8_B8_UNORM = 451
PIPE_FORMAT_Y8_UNORM = 452
PIPE_FORMAT_B8G8R8X8_SNORM = 453
PIPE_FORMAT_B8G8R8X8_UINT = 454
PIPE_FORMAT_B8G8R8X8_SINT = 455
PIPE_FORMAT_A8R8G8B8_SNORM = 456
PIPE_FORMAT_A8R8G8B8_SINT = 457
PIPE_FORMAT_X8R8G8B8_SNORM = 458
PIPE_FORMAT_X8R8G8B8_SINT = 459
PIPE_FORMAT_R5G5B5X1_UNORM = 460
PIPE_FORMAT_X1R5G5B5_UNORM = 461
PIPE_FORMAT_R4G4B4X4_UNORM = 462
PIPE_FORMAT_B10G10R10X2_SNORM = 463
PIPE_FORMAT_R5G6B5_SRGB = 464
PIPE_FORMAT_R10G10B10X2_SINT = 465
PIPE_FORMAT_B10G10R10X2_SINT = 466
PIPE_FORMAT_G16R16_SINT = 467
PIPE_FORMAT_COUNT = 468
pipe_format = ctypes.c_uint32 # enum
struct_nir_variable_data_0_image._pack_ = 1 # source:False
struct_nir_variable_data_0_image._fields_ = [
    ('format', pipe_format),
]

class struct_nir_variable_data_0_sampler(Structure):
    pass

struct_nir_variable_data_0_sampler._pack_ = 1 # source:False
struct_nir_variable_data_0_sampler._fields_ = [
    ('is_inline_sampler', ctypes.c_uint32, 1),
    ('addressing_mode', ctypes.c_uint32, 3),
    ('normalized_coordinates', ctypes.c_uint32, 1),
    ('filter_mode', ctypes.c_uint32, 1),
    ('PADDING_0', ctypes.c_uint32, 26),
]

class struct_nir_variable_data_0_xfb(Structure):
    pass

struct_nir_variable_data_0_xfb._pack_ = 1 # source:False
struct_nir_variable_data_0_xfb._fields_ = [
    ('buffer', ctypes.c_uint16, 2),
    ('PADDING_0', ctypes.c_uint16, 14),
    ('stride', ctypes.c_uint16),
]

union_nir_variable_data_0._pack_ = 1 # source:False
union_nir_variable_data_0._fields_ = [
    ('image', struct_nir_variable_data_0_image),
    ('sampler', struct_nir_variable_data_0_sampler),
    ('xfb', struct_nir_variable_data_0_xfb),
]

struct_nir_variable_data._pack_ = 1 # source:False
struct_nir_variable_data._anonymous_ = ('_0',)
struct_nir_variable_data._fields_ = [
    ('mode', ctypes.c_uint64, 21),
    ('read_only', ctypes.c_uint64, 1),
    ('centroid', ctypes.c_uint64, 1),
    ('sample', ctypes.c_uint64, 1),
    ('patch', ctypes.c_uint64, 1),
    ('invariant', ctypes.c_uint64, 1),
    ('explicit_invariant', ctypes.c_uint64, 1),
    ('ray_query', ctypes.c_uint64, 1),
    ('precision', ctypes.c_uint64, 2),
    ('assigned', ctypes.c_uint64, 1),
    ('cannot_coalesce', ctypes.c_uint64, 1),
    ('always_active_io', ctypes.c_uint64, 1),
    ('interpolation', ctypes.c_uint64, 3),
    ('location_frac', ctypes.c_uint64, 2),
    ('compact', ctypes.c_uint64, 1),
    ('fb_fetch_output', ctypes.c_uint64, 1),
    ('bindless', ctypes.c_uint64, 1),
    ('explicit_binding', ctypes.c_uint64, 1),
    ('explicit_location', ctypes.c_uint64, 1),
    ('implicit_sized_array', ctypes.c_uint64, 1),
    ('PADDING_0', ctypes.c_uint32, 20),
    ('max_array_access', ctypes.c_int32),
    ('has_initializer', ctypes.c_uint64, 1),
    ('is_implicit_initializer', ctypes.c_uint64, 1),
    ('is_xfb', ctypes.c_uint64, 1),
    ('is_xfb_only', ctypes.c_uint64, 1),
    ('explicit_xfb_buffer', ctypes.c_uint64, 1),
    ('explicit_xfb_stride', ctypes.c_uint64, 1),
    ('explicit_offset', ctypes.c_uint64, 1),
    ('matrix_layout', ctypes.c_uint64, 2),
    ('from_named_ifc_block', ctypes.c_uint64, 1),
    ('from_ssbo_unsized_array', ctypes.c_uint64, 1),
    ('must_be_shader_input', ctypes.c_uint64, 1),
    ('used', ctypes.c_uint64, 1),
    ('how_declared', ctypes.c_uint64, 2),
    ('per_view', ctypes.c_uint64, 1),
    ('per_primitive', ctypes.c_uint64, 1),
    ('per_vertex', ctypes.c_uint64, 1),
    ('aliased_shared_memory', ctypes.c_uint64, 1),
    ('depth_layout', ctypes.c_uint64, 3),
    ('stream', ctypes.c_uint64, 9),
    ('PADDING_1', ctypes.c_uint8, 1),
    ('access', ctypes.c_uint64, 9),
    ('descriptor_set', ctypes.c_uint64, 5),
    ('PADDING_2', ctypes.c_uint32, 18),
    ('index', ctypes.c_uint32),
    ('binding', ctypes.c_uint32),
    ('location', ctypes.c_int32),
    ('alignment', ctypes.c_uint32),
    ('driver_location', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('_0', union_nir_variable_data_0),
    ('node_name', ctypes.POINTER(ctypes.c_char)),
]

nir_variable_data = struct_nir_variable_data
class struct_nir_variable(Structure):
    pass

class struct_glsl_type(Structure):
    pass

struct_nir_variable._pack_ = 1 # source:False
struct_nir_variable._fields_ = [
    ('node', struct_exec_node),
    ('type', ctypes.POINTER(struct_glsl_type)),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('data', struct_nir_variable_data),
    ('index', ctypes.c_uint32),
    ('num_members', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('max_ifc_array_access', ctypes.POINTER(ctypes.c_int32)),
    ('num_state_slots', ctypes.c_uint16),
    ('PADDING_1', ctypes.c_ubyte * 6),
    ('state_slots', ctypes.POINTER(struct_nir_state_slot)),
    ('constant_initializer', ctypes.POINTER(struct_nir_constant)),
    ('pointer_initializer', ctypes.POINTER(struct_nir_variable)),
    ('interface_type', ctypes.POINTER(struct_glsl_type)),
    ('members', ctypes.POINTER(struct_nir_variable_data)),
    ('_name_storage', ctypes.c_char * 16),
]


# values for enumeration 'glsl_base_type'
glsl_base_type__enumvalues = {
    0: 'GLSL_TYPE_UINT',
    1: 'GLSL_TYPE_INT',
    2: 'GLSL_TYPE_FLOAT',
    3: 'GLSL_TYPE_FLOAT16',
    4: 'GLSL_TYPE_BFLOAT16',
    5: 'GLSL_TYPE_FLOAT_E4M3FN',
    6: 'GLSL_TYPE_FLOAT_E5M2',
    7: 'GLSL_TYPE_DOUBLE',
    8: 'GLSL_TYPE_UINT8',
    9: 'GLSL_TYPE_INT8',
    10: 'GLSL_TYPE_UINT16',
    11: 'GLSL_TYPE_INT16',
    12: 'GLSL_TYPE_UINT64',
    13: 'GLSL_TYPE_INT64',
    14: 'GLSL_TYPE_BOOL',
    15: 'GLSL_TYPE_COOPERATIVE_MATRIX',
    16: 'GLSL_TYPE_SAMPLER',
    17: 'GLSL_TYPE_TEXTURE',
    18: 'GLSL_TYPE_IMAGE',
    19: 'GLSL_TYPE_ATOMIC_UINT',
    20: 'GLSL_TYPE_STRUCT',
    21: 'GLSL_TYPE_INTERFACE',
    22: 'GLSL_TYPE_ARRAY',
    23: 'GLSL_TYPE_VOID',
    24: 'GLSL_TYPE_SUBROUTINE',
    25: 'GLSL_TYPE_ERROR',
}
GLSL_TYPE_UINT = 0
GLSL_TYPE_INT = 1
GLSL_TYPE_FLOAT = 2
GLSL_TYPE_FLOAT16 = 3
GLSL_TYPE_BFLOAT16 = 4
GLSL_TYPE_FLOAT_E4M3FN = 5
GLSL_TYPE_FLOAT_E5M2 = 6
GLSL_TYPE_DOUBLE = 7
GLSL_TYPE_UINT8 = 8
GLSL_TYPE_INT8 = 9
GLSL_TYPE_UINT16 = 10
GLSL_TYPE_INT16 = 11
GLSL_TYPE_UINT64 = 12
GLSL_TYPE_INT64 = 13
GLSL_TYPE_BOOL = 14
GLSL_TYPE_COOPERATIVE_MATRIX = 15
GLSL_TYPE_SAMPLER = 16
GLSL_TYPE_TEXTURE = 17
GLSL_TYPE_IMAGE = 18
GLSL_TYPE_ATOMIC_UINT = 19
GLSL_TYPE_STRUCT = 20
GLSL_TYPE_INTERFACE = 21
GLSL_TYPE_ARRAY = 22
GLSL_TYPE_VOID = 23
GLSL_TYPE_SUBROUTINE = 24
GLSL_TYPE_ERROR = 25
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

nir_variable = struct_nir_variable
try:
    _nir_shader_variable_has_mode = _libraries['FIXME_STUB']._nir_shader_variable_has_mode
    _nir_shader_variable_has_mode.restype = ctypes.c_bool
    _nir_shader_variable_has_mode.argtypes = [ctypes.POINTER(struct_nir_variable), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_variable_is_global = _libraries['FIXME_STUB'].nir_variable_is_global
    nir_variable_is_global.restype = ctypes.c_bool
    nir_variable_is_global.argtypes = [ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_instr_type'
c__EA_nir_instr_type__enumvalues = {
    0: 'nir_instr_type_alu',
    1: 'nir_instr_type_deref',
    2: 'nir_instr_type_call',
    3: 'nir_instr_type_tex',
    4: 'nir_instr_type_intrinsic',
    5: 'nir_instr_type_load_const',
    6: 'nir_instr_type_jump',
    7: 'nir_instr_type_undef',
    8: 'nir_instr_type_phi',
    9: 'nir_instr_type_parallel_copy',
}
nir_instr_type_alu = 0
nir_instr_type_deref = 1
nir_instr_type_call = 2
nir_instr_type_tex = 3
nir_instr_type_intrinsic = 4
nir_instr_type_load_const = 5
nir_instr_type_jump = 6
nir_instr_type_undef = 7
nir_instr_type_phi = 8
nir_instr_type_parallel_copy = 9
c__EA_nir_instr_type = ctypes.c_uint32 # enum
nir_instr_type = c__EA_nir_instr_type
nir_instr_type__enumvalues = c__EA_nir_instr_type__enumvalues
nir_instr = struct_nir_instr
try:
    nir_instr_next = _libraries['FIXME_STUB'].nir_instr_next
    nir_instr_next.restype = ctypes.POINTER(struct_nir_instr)
    nir_instr_next.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_prev = _libraries['FIXME_STUB'].nir_instr_prev
    nir_instr_prev.restype = ctypes.POINTER(struct_nir_instr)
    nir_instr_prev.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_is_first = _libraries['FIXME_STUB'].nir_instr_is_first
    nir_instr_is_first.restype = ctypes.c_bool
    nir_instr_is_first.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_is_last = _libraries['FIXME_STUB'].nir_instr_is_last
    nir_instr_is_last.restype = ctypes.c_bool
    nir_instr_is_last.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
class struct_nir_def(Structure):
    pass

class struct_list_head(Structure):
    pass

struct_list_head._pack_ = 1 # source:False
struct_list_head._fields_ = [
    ('prev', ctypes.POINTER(struct_list_head)),
    ('next', ctypes.POINTER(struct_list_head)),
]

struct_nir_def._pack_ = 1 # source:False
struct_nir_def._fields_ = [
    ('parent_instr', ctypes.POINTER(struct_nir_instr)),
    ('uses', struct_list_head),
    ('index', ctypes.c_uint32),
    ('num_components', ctypes.c_ubyte),
    ('bit_size', ctypes.c_ubyte),
    ('divergent', ctypes.c_bool),
    ('loop_invariant', ctypes.c_bool),
]

nir_def = struct_nir_def
try:
    nir_def_block = _libraries['FIXME_STUB'].nir_def_block
    nir_def_block.restype = ctypes.POINTER(struct_nir_block)
    nir_def_block.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
class struct_nir_src(Structure):
    pass

struct_nir_src._pack_ = 1 # source:False
struct_nir_src._fields_ = [
    ('_parent', ctypes.c_uint64),
    ('use_link', struct_list_head),
    ('ssa', ctypes.POINTER(struct_nir_def)),
]

nir_src = struct_nir_src
try:
    nir_src_is_if = _libraries['FIXME_STUB'].nir_src_is_if
    nir_src_is_if.restype = ctypes.c_bool
    nir_src_is_if.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_src_parent_instr = _libraries['FIXME_STUB'].nir_src_parent_instr
    nir_src_parent_instr.restype = ctypes.POINTER(struct_nir_instr)
    nir_src_parent_instr.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
class struct_nir_if(Structure):
    pass


# values for enumeration 'c__EA_nir_selection_control'
c__EA_nir_selection_control__enumvalues = {
    0: 'nir_selection_control_none',
    1: 'nir_selection_control_flatten',
    2: 'nir_selection_control_dont_flatten',
    3: 'nir_selection_control_divergent_always_taken',
}
nir_selection_control_none = 0
nir_selection_control_flatten = 1
nir_selection_control_dont_flatten = 2
nir_selection_control_divergent_always_taken = 3
c__EA_nir_selection_control = ctypes.c_uint32 # enum
struct_nir_if._pack_ = 1 # source:False
struct_nir_if._fields_ = [
    ('cf_node', struct_nir_cf_node),
    ('condition', nir_src),
    ('control', c__EA_nir_selection_control),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('then_list', struct_exec_list),
    ('else_list', struct_exec_list),
]

try:
    nir_src_parent_if = _libraries['FIXME_STUB'].nir_src_parent_if
    nir_src_parent_if.restype = ctypes.POINTER(struct_nir_if)
    nir_src_parent_if.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    _nir_src_set_parent = _libraries['FIXME_STUB']._nir_src_set_parent
    _nir_src_set_parent.restype = None
    _nir_src_set_parent.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(None), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_src_set_parent_instr = _libraries['FIXME_STUB'].nir_src_set_parent_instr
    nir_src_set_parent_instr.restype = None
    nir_src_set_parent_instr.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_src_set_parent_if = _libraries['FIXME_STUB'].nir_src_set_parent_if
    nir_src_set_parent_if.restype = None
    nir_src_set_parent_if.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_src_init = _libraries['FIXME_STUB'].nir_src_init
    nir_src_init.restype = nir_src
    nir_src_init.argtypes = []
except AttributeError:
    pass
try:
    nir_def_used_by_if = _libraries['FIXME_STUB'].nir_def_used_by_if
    nir_def_used_by_if.restype = ctypes.c_bool
    nir_def_used_by_if.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_only_used_by_if = _libraries['FIXME_STUB'].nir_def_only_used_by_if
    nir_def_only_used_by_if.restype = ctypes.c_bool
    nir_def_only_used_by_if.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_src_for_ssa = _libraries['FIXME_STUB'].nir_src_for_ssa
    nir_src_for_ssa.restype = nir_src
    nir_src_for_ssa.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_src_bit_size = _libraries['FIXME_STUB'].nir_src_bit_size
    nir_src_bit_size.restype = ctypes.c_uint32
    nir_src_bit_size.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_num_components = _libraries['FIXME_STUB'].nir_src_num_components
    nir_src_num_components.restype = ctypes.c_uint32
    nir_src_num_components.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_is_const = _libraries['FIXME_STUB'].nir_src_is_const
    nir_src_is_const.restype = ctypes.c_bool
    nir_src_is_const.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_is_undef = _libraries['FIXME_STUB'].nir_src_is_undef
    nir_src_is_undef.restype = ctypes.c_bool
    nir_src_is_undef.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_is_divergent = _libraries['FIXME_STUB'].nir_src_is_divergent
    nir_src_is_divergent.restype = ctypes.c_bool
    nir_src_is_divergent.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_is_same_comp_swizzle = _libraries['FIXME_STUB'].nir_is_same_comp_swizzle
    nir_is_same_comp_swizzle.restype = ctypes.c_bool
    nir_is_same_comp_swizzle.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_sequential_comp_swizzle = _libraries['FIXME_STUB'].nir_is_sequential_comp_swizzle
    nir_is_sequential_comp_swizzle.restype = ctypes.c_bool
    nir_is_sequential_comp_swizzle.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32]
except AttributeError:
    pass
class struct_nir_alu_src(Structure):
    pass

struct_nir_alu_src._pack_ = 1 # source:False
struct_nir_alu_src._fields_ = [
    ('src', nir_src),
    ('swizzle', ctypes.c_ubyte * 16),
]

nir_alu_src = struct_nir_alu_src

# values for enumeration 'c__EA_nir_alu_type'
c__EA_nir_alu_type__enumvalues = {
    0: 'nir_type_invalid',
    2: 'nir_type_int',
    4: 'nir_type_uint',
    6: 'nir_type_bool',
    128: 'nir_type_float',
    7: 'nir_type_bool1',
    14: 'nir_type_bool8',
    22: 'nir_type_bool16',
    38: 'nir_type_bool32',
    3: 'nir_type_int1',
    10: 'nir_type_int8',
    18: 'nir_type_int16',
    34: 'nir_type_int32',
    66: 'nir_type_int64',
    5: 'nir_type_uint1',
    12: 'nir_type_uint8',
    20: 'nir_type_uint16',
    36: 'nir_type_uint32',
    68: 'nir_type_uint64',
    144: 'nir_type_float16',
    160: 'nir_type_float32',
    192: 'nir_type_float64',
}
nir_type_invalid = 0
nir_type_int = 2
nir_type_uint = 4
nir_type_bool = 6
nir_type_float = 128
nir_type_bool1 = 7
nir_type_bool8 = 14
nir_type_bool16 = 22
nir_type_bool32 = 38
nir_type_int1 = 3
nir_type_int8 = 10
nir_type_int16 = 18
nir_type_int32 = 34
nir_type_int64 = 66
nir_type_uint1 = 5
nir_type_uint8 = 12
nir_type_uint16 = 20
nir_type_uint32 = 36
nir_type_uint64 = 68
nir_type_float16 = 144
nir_type_float32 = 160
nir_type_float64 = 192
c__EA_nir_alu_type = ctypes.c_uint32 # enum
nir_alu_type = c__EA_nir_alu_type
nir_alu_type__enumvalues = c__EA_nir_alu_type__enumvalues
try:
    nir_get_nir_type_for_glsl_base_type = _libraries['FIXME_STUB'].nir_get_nir_type_for_glsl_base_type
    nir_get_nir_type_for_glsl_base_type.restype = nir_alu_type
    nir_get_nir_type_for_glsl_base_type.argtypes = [glsl_base_type]
except AttributeError:
    pass
try:
    nir_get_nir_type_for_glsl_type = _libraries['FIXME_STUB'].nir_get_nir_type_for_glsl_type
    nir_get_nir_type_for_glsl_type.restype = nir_alu_type
    nir_get_nir_type_for_glsl_type.argtypes = [ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_get_glsl_base_type_for_nir_type = _libraries['FIXME_STUB'].nir_get_glsl_base_type_for_nir_type
    nir_get_glsl_base_type_for_nir_type.restype = glsl_base_type
    nir_get_glsl_base_type_for_nir_type.argtypes = [nir_alu_type]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_op'
c__EA_nir_op__enumvalues = {
    0: 'nir_op_alignbyte_amd',
    1: 'nir_op_amul',
    2: 'nir_op_andg_ir3',
    3: 'nir_op_b16all_fequal16',
    4: 'nir_op_b16all_fequal2',
    5: 'nir_op_b16all_fequal3',
    6: 'nir_op_b16all_fequal4',
    7: 'nir_op_b16all_fequal5',
    8: 'nir_op_b16all_fequal8',
    9: 'nir_op_b16all_iequal16',
    10: 'nir_op_b16all_iequal2',
    11: 'nir_op_b16all_iequal3',
    12: 'nir_op_b16all_iequal4',
    13: 'nir_op_b16all_iequal5',
    14: 'nir_op_b16all_iequal8',
    15: 'nir_op_b16any_fnequal16',
    16: 'nir_op_b16any_fnequal2',
    17: 'nir_op_b16any_fnequal3',
    18: 'nir_op_b16any_fnequal4',
    19: 'nir_op_b16any_fnequal5',
    20: 'nir_op_b16any_fnequal8',
    21: 'nir_op_b16any_inequal16',
    22: 'nir_op_b16any_inequal2',
    23: 'nir_op_b16any_inequal3',
    24: 'nir_op_b16any_inequal4',
    25: 'nir_op_b16any_inequal5',
    26: 'nir_op_b16any_inequal8',
    27: 'nir_op_b16csel',
    28: 'nir_op_b2b1',
    29: 'nir_op_b2b16',
    30: 'nir_op_b2b32',
    31: 'nir_op_b2b8',
    32: 'nir_op_b2f16',
    33: 'nir_op_b2f32',
    34: 'nir_op_b2f64',
    35: 'nir_op_b2i1',
    36: 'nir_op_b2i16',
    37: 'nir_op_b2i32',
    38: 'nir_op_b2i64',
    39: 'nir_op_b2i8',
    40: 'nir_op_b32all_fequal16',
    41: 'nir_op_b32all_fequal2',
    42: 'nir_op_b32all_fequal3',
    43: 'nir_op_b32all_fequal4',
    44: 'nir_op_b32all_fequal5',
    45: 'nir_op_b32all_fequal8',
    46: 'nir_op_b32all_iequal16',
    47: 'nir_op_b32all_iequal2',
    48: 'nir_op_b32all_iequal3',
    49: 'nir_op_b32all_iequal4',
    50: 'nir_op_b32all_iequal5',
    51: 'nir_op_b32all_iequal8',
    52: 'nir_op_b32any_fnequal16',
    53: 'nir_op_b32any_fnequal2',
    54: 'nir_op_b32any_fnequal3',
    55: 'nir_op_b32any_fnequal4',
    56: 'nir_op_b32any_fnequal5',
    57: 'nir_op_b32any_fnequal8',
    58: 'nir_op_b32any_inequal16',
    59: 'nir_op_b32any_inequal2',
    60: 'nir_op_b32any_inequal3',
    61: 'nir_op_b32any_inequal4',
    62: 'nir_op_b32any_inequal5',
    63: 'nir_op_b32any_inequal8',
    64: 'nir_op_b32csel',
    65: 'nir_op_b32fcsel_mdg',
    66: 'nir_op_b8all_fequal16',
    67: 'nir_op_b8all_fequal2',
    68: 'nir_op_b8all_fequal3',
    69: 'nir_op_b8all_fequal4',
    70: 'nir_op_b8all_fequal5',
    71: 'nir_op_b8all_fequal8',
    72: 'nir_op_b8all_iequal16',
    73: 'nir_op_b8all_iequal2',
    74: 'nir_op_b8all_iequal3',
    75: 'nir_op_b8all_iequal4',
    76: 'nir_op_b8all_iequal5',
    77: 'nir_op_b8all_iequal8',
    78: 'nir_op_b8any_fnequal16',
    79: 'nir_op_b8any_fnequal2',
    80: 'nir_op_b8any_fnequal3',
    81: 'nir_op_b8any_fnequal4',
    82: 'nir_op_b8any_fnequal5',
    83: 'nir_op_b8any_fnequal8',
    84: 'nir_op_b8any_inequal16',
    85: 'nir_op_b8any_inequal2',
    86: 'nir_op_b8any_inequal3',
    87: 'nir_op_b8any_inequal4',
    88: 'nir_op_b8any_inequal5',
    89: 'nir_op_b8any_inequal8',
    90: 'nir_op_b8csel',
    91: 'nir_op_ball_fequal16',
    92: 'nir_op_ball_fequal2',
    93: 'nir_op_ball_fequal3',
    94: 'nir_op_ball_fequal4',
    95: 'nir_op_ball_fequal5',
    96: 'nir_op_ball_fequal8',
    97: 'nir_op_ball_iequal16',
    98: 'nir_op_ball_iequal2',
    99: 'nir_op_ball_iequal3',
    100: 'nir_op_ball_iequal4',
    101: 'nir_op_ball_iequal5',
    102: 'nir_op_ball_iequal8',
    103: 'nir_op_bany_fnequal16',
    104: 'nir_op_bany_fnequal2',
    105: 'nir_op_bany_fnequal3',
    106: 'nir_op_bany_fnequal4',
    107: 'nir_op_bany_fnequal5',
    108: 'nir_op_bany_fnequal8',
    109: 'nir_op_bany_inequal16',
    110: 'nir_op_bany_inequal2',
    111: 'nir_op_bany_inequal3',
    112: 'nir_op_bany_inequal4',
    113: 'nir_op_bany_inequal5',
    114: 'nir_op_bany_inequal8',
    115: 'nir_op_bcsel',
    116: 'nir_op_bf2f',
    117: 'nir_op_bfdot16',
    118: 'nir_op_bfdot2',
    119: 'nir_op_bfdot2_bfadd',
    120: 'nir_op_bfdot3',
    121: 'nir_op_bfdot4',
    122: 'nir_op_bfdot5',
    123: 'nir_op_bfdot8',
    124: 'nir_op_bffma',
    125: 'nir_op_bfi',
    126: 'nir_op_bfm',
    127: 'nir_op_bfmul',
    128: 'nir_op_bit_count',
    129: 'nir_op_bitfield_insert',
    130: 'nir_op_bitfield_reverse',
    131: 'nir_op_bitfield_select',
    132: 'nir_op_bitnz',
    133: 'nir_op_bitnz16',
    134: 'nir_op_bitnz32',
    135: 'nir_op_bitnz8',
    136: 'nir_op_bitz',
    137: 'nir_op_bitz16',
    138: 'nir_op_bitz32',
    139: 'nir_op_bitz8',
    140: 'nir_op_bounds_agx',
    141: 'nir_op_byte_perm_amd',
    142: 'nir_op_cube_amd',
    143: 'nir_op_e4m3fn2f',
    144: 'nir_op_e5m22f',
    145: 'nir_op_extr_agx',
    146: 'nir_op_extract_i16',
    147: 'nir_op_extract_i8',
    148: 'nir_op_extract_u16',
    149: 'nir_op_extract_u8',
    150: 'nir_op_f2bf',
    151: 'nir_op_f2e4m3fn',
    152: 'nir_op_f2e4m3fn_sat',
    153: 'nir_op_f2e4m3fn_satfn',
    154: 'nir_op_f2e5m2',
    155: 'nir_op_f2e5m2_sat',
    156: 'nir_op_f2f16',
    157: 'nir_op_f2f16_rtne',
    158: 'nir_op_f2f16_rtz',
    159: 'nir_op_f2f32',
    160: 'nir_op_f2f64',
    161: 'nir_op_f2fmp',
    162: 'nir_op_f2i1',
    163: 'nir_op_f2i16',
    164: 'nir_op_f2i32',
    165: 'nir_op_f2i64',
    166: 'nir_op_f2i8',
    167: 'nir_op_f2imp',
    168: 'nir_op_f2snorm_16_v3d',
    169: 'nir_op_f2u1',
    170: 'nir_op_f2u16',
    171: 'nir_op_f2u32',
    172: 'nir_op_f2u64',
    173: 'nir_op_f2u8',
    174: 'nir_op_f2ump',
    175: 'nir_op_f2unorm_16_v3d',
    176: 'nir_op_fabs',
    177: 'nir_op_fadd',
    178: 'nir_op_fall_equal16',
    179: 'nir_op_fall_equal2',
    180: 'nir_op_fall_equal3',
    181: 'nir_op_fall_equal4',
    182: 'nir_op_fall_equal5',
    183: 'nir_op_fall_equal8',
    184: 'nir_op_fany_nequal16',
    185: 'nir_op_fany_nequal2',
    186: 'nir_op_fany_nequal3',
    187: 'nir_op_fany_nequal4',
    188: 'nir_op_fany_nequal5',
    189: 'nir_op_fany_nequal8',
    190: 'nir_op_fceil',
    191: 'nir_op_fclamp_pos',
    192: 'nir_op_fcos',
    193: 'nir_op_fcos_amd',
    194: 'nir_op_fcos_mdg',
    195: 'nir_op_fcsel',
    196: 'nir_op_fcsel_ge',
    197: 'nir_op_fcsel_gt',
    198: 'nir_op_fdiv',
    199: 'nir_op_fdot16',
    200: 'nir_op_fdot16_replicated',
    201: 'nir_op_fdot2',
    202: 'nir_op_fdot2_replicated',
    203: 'nir_op_fdot3',
    204: 'nir_op_fdot3_replicated',
    205: 'nir_op_fdot4',
    206: 'nir_op_fdot4_replicated',
    207: 'nir_op_fdot5',
    208: 'nir_op_fdot5_replicated',
    209: 'nir_op_fdot8',
    210: 'nir_op_fdot8_replicated',
    211: 'nir_op_fdph',
    212: 'nir_op_fdph_replicated',
    213: 'nir_op_feq',
    214: 'nir_op_feq16',
    215: 'nir_op_feq32',
    216: 'nir_op_feq8',
    217: 'nir_op_fequ',
    218: 'nir_op_fequ16',
    219: 'nir_op_fequ32',
    220: 'nir_op_fequ8',
    221: 'nir_op_fexp2',
    222: 'nir_op_ffloor',
    223: 'nir_op_ffma',
    224: 'nir_op_ffmaz',
    225: 'nir_op_ffract',
    226: 'nir_op_fge',
    227: 'nir_op_fge16',
    228: 'nir_op_fge32',
    229: 'nir_op_fge8',
    230: 'nir_op_fgeu',
    231: 'nir_op_fgeu16',
    232: 'nir_op_fgeu32',
    233: 'nir_op_fgeu8',
    234: 'nir_op_find_lsb',
    235: 'nir_op_fisfinite',
    236: 'nir_op_fisfinite32',
    237: 'nir_op_fisnormal',
    238: 'nir_op_flog2',
    239: 'nir_op_flrp',
    240: 'nir_op_flt',
    241: 'nir_op_flt16',
    242: 'nir_op_flt32',
    243: 'nir_op_flt8',
    244: 'nir_op_fltu',
    245: 'nir_op_fltu16',
    246: 'nir_op_fltu32',
    247: 'nir_op_fltu8',
    248: 'nir_op_fmax',
    249: 'nir_op_fmax_agx',
    250: 'nir_op_fmin',
    251: 'nir_op_fmin_agx',
    252: 'nir_op_fmod',
    253: 'nir_op_fmul',
    254: 'nir_op_fmulz',
    255: 'nir_op_fneg',
    256: 'nir_op_fneo',
    257: 'nir_op_fneo16',
    258: 'nir_op_fneo32',
    259: 'nir_op_fneo8',
    260: 'nir_op_fneu',
    261: 'nir_op_fneu16',
    262: 'nir_op_fneu32',
    263: 'nir_op_fneu8',
    264: 'nir_op_ford',
    265: 'nir_op_ford16',
    266: 'nir_op_ford32',
    267: 'nir_op_ford8',
    268: 'nir_op_fpow',
    269: 'nir_op_fquantize2f16',
    270: 'nir_op_frcp',
    271: 'nir_op_frem',
    272: 'nir_op_frexp_exp',
    273: 'nir_op_frexp_sig',
    274: 'nir_op_fround_even',
    275: 'nir_op_frsq',
    276: 'nir_op_fsat',
    277: 'nir_op_fsat_signed',
    278: 'nir_op_fsign',
    279: 'nir_op_fsin',
    280: 'nir_op_fsin_agx',
    281: 'nir_op_fsin_amd',
    282: 'nir_op_fsin_mdg',
    283: 'nir_op_fsqrt',
    284: 'nir_op_fsub',
    285: 'nir_op_fsum2',
    286: 'nir_op_fsum3',
    287: 'nir_op_fsum4',
    288: 'nir_op_ftrunc',
    289: 'nir_op_funord',
    290: 'nir_op_funord16',
    291: 'nir_op_funord32',
    292: 'nir_op_funord8',
    293: 'nir_op_i2f16',
    294: 'nir_op_i2f32',
    295: 'nir_op_i2f64',
    296: 'nir_op_i2fmp',
    297: 'nir_op_i2i1',
    298: 'nir_op_i2i16',
    299: 'nir_op_i2i32',
    300: 'nir_op_i2i64',
    301: 'nir_op_i2i8',
    302: 'nir_op_i2imp',
    303: 'nir_op_i32csel_ge',
    304: 'nir_op_i32csel_gt',
    305: 'nir_op_iabs',
    306: 'nir_op_iadd',
    307: 'nir_op_iadd3',
    308: 'nir_op_iadd_sat',
    309: 'nir_op_iand',
    310: 'nir_op_ibfe',
    311: 'nir_op_ibitfield_extract',
    312: 'nir_op_icsel_eqz',
    313: 'nir_op_idiv',
    314: 'nir_op_ieq',
    315: 'nir_op_ieq16',
    316: 'nir_op_ieq32',
    317: 'nir_op_ieq8',
    318: 'nir_op_ifind_msb',
    319: 'nir_op_ifind_msb_rev',
    320: 'nir_op_ige',
    321: 'nir_op_ige16',
    322: 'nir_op_ige32',
    323: 'nir_op_ige8',
    324: 'nir_op_ihadd',
    325: 'nir_op_ilea_agx',
    326: 'nir_op_ilt',
    327: 'nir_op_ilt16',
    328: 'nir_op_ilt32',
    329: 'nir_op_ilt8',
    330: 'nir_op_imad',
    331: 'nir_op_imad24_ir3',
    332: 'nir_op_imadsh_mix16',
    333: 'nir_op_imadshl_agx',
    334: 'nir_op_imax',
    335: 'nir_op_imin',
    336: 'nir_op_imod',
    337: 'nir_op_imsubshl_agx',
    338: 'nir_op_imul',
    339: 'nir_op_imul24',
    340: 'nir_op_imul24_relaxed',
    341: 'nir_op_imul_2x32_64',
    342: 'nir_op_imul_32x16',
    343: 'nir_op_imul_high',
    344: 'nir_op_ine',
    345: 'nir_op_ine16',
    346: 'nir_op_ine32',
    347: 'nir_op_ine8',
    348: 'nir_op_ineg',
    349: 'nir_op_inot',
    350: 'nir_op_insert_u16',
    351: 'nir_op_insert_u8',
    352: 'nir_op_interleave_agx',
    353: 'nir_op_ior',
    354: 'nir_op_irem',
    355: 'nir_op_irhadd',
    356: 'nir_op_ishl',
    357: 'nir_op_ishr',
    358: 'nir_op_isign',
    359: 'nir_op_isub',
    360: 'nir_op_isub_sat',
    361: 'nir_op_ixor',
    362: 'nir_op_ldexp',
    363: 'nir_op_ldexp16_pan',
    364: 'nir_op_lea_nv',
    365: 'nir_op_mov',
    366: 'nir_op_mqsad_4x8',
    367: 'nir_op_msad_4x8',
    368: 'nir_op_pack_2x16_to_snorm_2x8_v3d',
    369: 'nir_op_pack_2x16_to_unorm_10_2_v3d',
    370: 'nir_op_pack_2x16_to_unorm_2x10_v3d',
    371: 'nir_op_pack_2x16_to_unorm_2x8_v3d',
    372: 'nir_op_pack_2x32_to_2x16_v3d',
    373: 'nir_op_pack_32_2x16',
    374: 'nir_op_pack_32_2x16_split',
    375: 'nir_op_pack_32_4x8',
    376: 'nir_op_pack_32_4x8_split',
    377: 'nir_op_pack_32_to_r11g11b10_v3d',
    378: 'nir_op_pack_4x16_to_4x8_v3d',
    379: 'nir_op_pack_64_2x32',
    380: 'nir_op_pack_64_2x32_split',
    381: 'nir_op_pack_64_4x16',
    382: 'nir_op_pack_double_2x32_dxil',
    383: 'nir_op_pack_half_2x16',
    384: 'nir_op_pack_half_2x16_rtz_split',
    385: 'nir_op_pack_half_2x16_split',
    386: 'nir_op_pack_sint_2x16',
    387: 'nir_op_pack_snorm_2x16',
    388: 'nir_op_pack_snorm_4x8',
    389: 'nir_op_pack_uint_2x16',
    390: 'nir_op_pack_uint_32_to_r10g10b10a2_v3d',
    391: 'nir_op_pack_unorm_2x16',
    392: 'nir_op_pack_unorm_4x8',
    393: 'nir_op_pack_uvec2_to_uint',
    394: 'nir_op_pack_uvec4_to_uint',
    395: 'nir_op_prmt_nv',
    396: 'nir_op_sdot_2x16_iadd',
    397: 'nir_op_sdot_2x16_iadd_sat',
    398: 'nir_op_sdot_4x8_iadd',
    399: 'nir_op_sdot_4x8_iadd_sat',
    400: 'nir_op_seq',
    401: 'nir_op_sge',
    402: 'nir_op_shfr',
    403: 'nir_op_shlg_ir3',
    404: 'nir_op_shlm_ir3',
    405: 'nir_op_shrg_ir3',
    406: 'nir_op_shrm_ir3',
    407: 'nir_op_slt',
    408: 'nir_op_sne',
    409: 'nir_op_sudot_4x8_iadd',
    410: 'nir_op_sudot_4x8_iadd_sat',
    411: 'nir_op_u2f16',
    412: 'nir_op_u2f32',
    413: 'nir_op_u2f64',
    414: 'nir_op_u2fmp',
    415: 'nir_op_u2u1',
    416: 'nir_op_u2u16',
    417: 'nir_op_u2u32',
    418: 'nir_op_u2u64',
    419: 'nir_op_u2u8',
    420: 'nir_op_uabs_isub',
    421: 'nir_op_uabs_usub',
    422: 'nir_op_uadd_carry',
    423: 'nir_op_uadd_sat',
    424: 'nir_op_ubfe',
    425: 'nir_op_ubitfield_extract',
    426: 'nir_op_uclz',
    427: 'nir_op_udiv',
    428: 'nir_op_udiv_aligned_4',
    429: 'nir_op_udot_2x16_uadd',
    430: 'nir_op_udot_2x16_uadd_sat',
    431: 'nir_op_udot_4x8_uadd',
    432: 'nir_op_udot_4x8_uadd_sat',
    433: 'nir_op_ufind_msb',
    434: 'nir_op_ufind_msb_rev',
    435: 'nir_op_uge',
    436: 'nir_op_uge16',
    437: 'nir_op_uge32',
    438: 'nir_op_uge8',
    439: 'nir_op_uhadd',
    440: 'nir_op_ulea_agx',
    441: 'nir_op_ult',
    442: 'nir_op_ult16',
    443: 'nir_op_ult32',
    444: 'nir_op_ult8',
    445: 'nir_op_umad24',
    446: 'nir_op_umad24_relaxed',
    447: 'nir_op_umax',
    448: 'nir_op_umax_4x8_vc4',
    449: 'nir_op_umin',
    450: 'nir_op_umin_4x8_vc4',
    451: 'nir_op_umod',
    452: 'nir_op_umul24',
    453: 'nir_op_umul24_relaxed',
    454: 'nir_op_umul_2x32_64',
    455: 'nir_op_umul_32x16',
    456: 'nir_op_umul_high',
    457: 'nir_op_umul_low',
    458: 'nir_op_umul_unorm_4x8_vc4',
    459: 'nir_op_unpack_32_2x16',
    460: 'nir_op_unpack_32_2x16_split_x',
    461: 'nir_op_unpack_32_2x16_split_y',
    462: 'nir_op_unpack_32_4x8',
    463: 'nir_op_unpack_64_2x32',
    464: 'nir_op_unpack_64_2x32_split_x',
    465: 'nir_op_unpack_64_2x32_split_y',
    466: 'nir_op_unpack_64_4x16',
    467: 'nir_op_unpack_double_2x32_dxil',
    468: 'nir_op_unpack_half_2x16',
    469: 'nir_op_unpack_half_2x16_split_x',
    470: 'nir_op_unpack_half_2x16_split_y',
    471: 'nir_op_unpack_snorm_2x16',
    472: 'nir_op_unpack_snorm_4x8',
    473: 'nir_op_unpack_unorm_2x16',
    474: 'nir_op_unpack_unorm_4x8',
    475: 'nir_op_urhadd',
    476: 'nir_op_urol',
    477: 'nir_op_uror',
    478: 'nir_op_usadd_4x8_vc4',
    479: 'nir_op_ushr',
    480: 'nir_op_ussub_4x8_vc4',
    481: 'nir_op_usub_borrow',
    482: 'nir_op_usub_sat',
    483: 'nir_op_vec16',
    484: 'nir_op_vec2',
    485: 'nir_op_vec3',
    486: 'nir_op_vec4',
    487: 'nir_op_vec5',
    488: 'nir_op_vec8',
    488: 'nir_last_opcode',
    489: 'nir_num_opcodes',
}
nir_op_alignbyte_amd = 0
nir_op_amul = 1
nir_op_andg_ir3 = 2
nir_op_b16all_fequal16 = 3
nir_op_b16all_fequal2 = 4
nir_op_b16all_fequal3 = 5
nir_op_b16all_fequal4 = 6
nir_op_b16all_fequal5 = 7
nir_op_b16all_fequal8 = 8
nir_op_b16all_iequal16 = 9
nir_op_b16all_iequal2 = 10
nir_op_b16all_iequal3 = 11
nir_op_b16all_iequal4 = 12
nir_op_b16all_iequal5 = 13
nir_op_b16all_iequal8 = 14
nir_op_b16any_fnequal16 = 15
nir_op_b16any_fnequal2 = 16
nir_op_b16any_fnequal3 = 17
nir_op_b16any_fnequal4 = 18
nir_op_b16any_fnequal5 = 19
nir_op_b16any_fnequal8 = 20
nir_op_b16any_inequal16 = 21
nir_op_b16any_inequal2 = 22
nir_op_b16any_inequal3 = 23
nir_op_b16any_inequal4 = 24
nir_op_b16any_inequal5 = 25
nir_op_b16any_inequal8 = 26
nir_op_b16csel = 27
nir_op_b2b1 = 28
nir_op_b2b16 = 29
nir_op_b2b32 = 30
nir_op_b2b8 = 31
nir_op_b2f16 = 32
nir_op_b2f32 = 33
nir_op_b2f64 = 34
nir_op_b2i1 = 35
nir_op_b2i16 = 36
nir_op_b2i32 = 37
nir_op_b2i64 = 38
nir_op_b2i8 = 39
nir_op_b32all_fequal16 = 40
nir_op_b32all_fequal2 = 41
nir_op_b32all_fequal3 = 42
nir_op_b32all_fequal4 = 43
nir_op_b32all_fequal5 = 44
nir_op_b32all_fequal8 = 45
nir_op_b32all_iequal16 = 46
nir_op_b32all_iequal2 = 47
nir_op_b32all_iequal3 = 48
nir_op_b32all_iequal4 = 49
nir_op_b32all_iequal5 = 50
nir_op_b32all_iequal8 = 51
nir_op_b32any_fnequal16 = 52
nir_op_b32any_fnequal2 = 53
nir_op_b32any_fnequal3 = 54
nir_op_b32any_fnequal4 = 55
nir_op_b32any_fnequal5 = 56
nir_op_b32any_fnequal8 = 57
nir_op_b32any_inequal16 = 58
nir_op_b32any_inequal2 = 59
nir_op_b32any_inequal3 = 60
nir_op_b32any_inequal4 = 61
nir_op_b32any_inequal5 = 62
nir_op_b32any_inequal8 = 63
nir_op_b32csel = 64
nir_op_b32fcsel_mdg = 65
nir_op_b8all_fequal16 = 66
nir_op_b8all_fequal2 = 67
nir_op_b8all_fequal3 = 68
nir_op_b8all_fequal4 = 69
nir_op_b8all_fequal5 = 70
nir_op_b8all_fequal8 = 71
nir_op_b8all_iequal16 = 72
nir_op_b8all_iequal2 = 73
nir_op_b8all_iequal3 = 74
nir_op_b8all_iequal4 = 75
nir_op_b8all_iequal5 = 76
nir_op_b8all_iequal8 = 77
nir_op_b8any_fnequal16 = 78
nir_op_b8any_fnequal2 = 79
nir_op_b8any_fnequal3 = 80
nir_op_b8any_fnequal4 = 81
nir_op_b8any_fnequal5 = 82
nir_op_b8any_fnequal8 = 83
nir_op_b8any_inequal16 = 84
nir_op_b8any_inequal2 = 85
nir_op_b8any_inequal3 = 86
nir_op_b8any_inequal4 = 87
nir_op_b8any_inequal5 = 88
nir_op_b8any_inequal8 = 89
nir_op_b8csel = 90
nir_op_ball_fequal16 = 91
nir_op_ball_fequal2 = 92
nir_op_ball_fequal3 = 93
nir_op_ball_fequal4 = 94
nir_op_ball_fequal5 = 95
nir_op_ball_fequal8 = 96
nir_op_ball_iequal16 = 97
nir_op_ball_iequal2 = 98
nir_op_ball_iequal3 = 99
nir_op_ball_iequal4 = 100
nir_op_ball_iequal5 = 101
nir_op_ball_iequal8 = 102
nir_op_bany_fnequal16 = 103
nir_op_bany_fnequal2 = 104
nir_op_bany_fnequal3 = 105
nir_op_bany_fnequal4 = 106
nir_op_bany_fnequal5 = 107
nir_op_bany_fnequal8 = 108
nir_op_bany_inequal16 = 109
nir_op_bany_inequal2 = 110
nir_op_bany_inequal3 = 111
nir_op_bany_inequal4 = 112
nir_op_bany_inequal5 = 113
nir_op_bany_inequal8 = 114
nir_op_bcsel = 115
nir_op_bf2f = 116
nir_op_bfdot16 = 117
nir_op_bfdot2 = 118
nir_op_bfdot2_bfadd = 119
nir_op_bfdot3 = 120
nir_op_bfdot4 = 121
nir_op_bfdot5 = 122
nir_op_bfdot8 = 123
nir_op_bffma = 124
nir_op_bfi = 125
nir_op_bfm = 126
nir_op_bfmul = 127
nir_op_bit_count = 128
nir_op_bitfield_insert = 129
nir_op_bitfield_reverse = 130
nir_op_bitfield_select = 131
nir_op_bitnz = 132
nir_op_bitnz16 = 133
nir_op_bitnz32 = 134
nir_op_bitnz8 = 135
nir_op_bitz = 136
nir_op_bitz16 = 137
nir_op_bitz32 = 138
nir_op_bitz8 = 139
nir_op_bounds_agx = 140
nir_op_byte_perm_amd = 141
nir_op_cube_amd = 142
nir_op_e4m3fn2f = 143
nir_op_e5m22f = 144
nir_op_extr_agx = 145
nir_op_extract_i16 = 146
nir_op_extract_i8 = 147
nir_op_extract_u16 = 148
nir_op_extract_u8 = 149
nir_op_f2bf = 150
nir_op_f2e4m3fn = 151
nir_op_f2e4m3fn_sat = 152
nir_op_f2e4m3fn_satfn = 153
nir_op_f2e5m2 = 154
nir_op_f2e5m2_sat = 155
nir_op_f2f16 = 156
nir_op_f2f16_rtne = 157
nir_op_f2f16_rtz = 158
nir_op_f2f32 = 159
nir_op_f2f64 = 160
nir_op_f2fmp = 161
nir_op_f2i1 = 162
nir_op_f2i16 = 163
nir_op_f2i32 = 164
nir_op_f2i64 = 165
nir_op_f2i8 = 166
nir_op_f2imp = 167
nir_op_f2snorm_16_v3d = 168
nir_op_f2u1 = 169
nir_op_f2u16 = 170
nir_op_f2u32 = 171
nir_op_f2u64 = 172
nir_op_f2u8 = 173
nir_op_f2ump = 174
nir_op_f2unorm_16_v3d = 175
nir_op_fabs = 176
nir_op_fadd = 177
nir_op_fall_equal16 = 178
nir_op_fall_equal2 = 179
nir_op_fall_equal3 = 180
nir_op_fall_equal4 = 181
nir_op_fall_equal5 = 182
nir_op_fall_equal8 = 183
nir_op_fany_nequal16 = 184
nir_op_fany_nequal2 = 185
nir_op_fany_nequal3 = 186
nir_op_fany_nequal4 = 187
nir_op_fany_nequal5 = 188
nir_op_fany_nequal8 = 189
nir_op_fceil = 190
nir_op_fclamp_pos = 191
nir_op_fcos = 192
nir_op_fcos_amd = 193
nir_op_fcos_mdg = 194
nir_op_fcsel = 195
nir_op_fcsel_ge = 196
nir_op_fcsel_gt = 197
nir_op_fdiv = 198
nir_op_fdot16 = 199
nir_op_fdot16_replicated = 200
nir_op_fdot2 = 201
nir_op_fdot2_replicated = 202
nir_op_fdot3 = 203
nir_op_fdot3_replicated = 204
nir_op_fdot4 = 205
nir_op_fdot4_replicated = 206
nir_op_fdot5 = 207
nir_op_fdot5_replicated = 208
nir_op_fdot8 = 209
nir_op_fdot8_replicated = 210
nir_op_fdph = 211
nir_op_fdph_replicated = 212
nir_op_feq = 213
nir_op_feq16 = 214
nir_op_feq32 = 215
nir_op_feq8 = 216
nir_op_fequ = 217
nir_op_fequ16 = 218
nir_op_fequ32 = 219
nir_op_fequ8 = 220
nir_op_fexp2 = 221
nir_op_ffloor = 222
nir_op_ffma = 223
nir_op_ffmaz = 224
nir_op_ffract = 225
nir_op_fge = 226
nir_op_fge16 = 227
nir_op_fge32 = 228
nir_op_fge8 = 229
nir_op_fgeu = 230
nir_op_fgeu16 = 231
nir_op_fgeu32 = 232
nir_op_fgeu8 = 233
nir_op_find_lsb = 234
nir_op_fisfinite = 235
nir_op_fisfinite32 = 236
nir_op_fisnormal = 237
nir_op_flog2 = 238
nir_op_flrp = 239
nir_op_flt = 240
nir_op_flt16 = 241
nir_op_flt32 = 242
nir_op_flt8 = 243
nir_op_fltu = 244
nir_op_fltu16 = 245
nir_op_fltu32 = 246
nir_op_fltu8 = 247
nir_op_fmax = 248
nir_op_fmax_agx = 249
nir_op_fmin = 250
nir_op_fmin_agx = 251
nir_op_fmod = 252
nir_op_fmul = 253
nir_op_fmulz = 254
nir_op_fneg = 255
nir_op_fneo = 256
nir_op_fneo16 = 257
nir_op_fneo32 = 258
nir_op_fneo8 = 259
nir_op_fneu = 260
nir_op_fneu16 = 261
nir_op_fneu32 = 262
nir_op_fneu8 = 263
nir_op_ford = 264
nir_op_ford16 = 265
nir_op_ford32 = 266
nir_op_ford8 = 267
nir_op_fpow = 268
nir_op_fquantize2f16 = 269
nir_op_frcp = 270
nir_op_frem = 271
nir_op_frexp_exp = 272
nir_op_frexp_sig = 273
nir_op_fround_even = 274
nir_op_frsq = 275
nir_op_fsat = 276
nir_op_fsat_signed = 277
nir_op_fsign = 278
nir_op_fsin = 279
nir_op_fsin_agx = 280
nir_op_fsin_amd = 281
nir_op_fsin_mdg = 282
nir_op_fsqrt = 283
nir_op_fsub = 284
nir_op_fsum2 = 285
nir_op_fsum3 = 286
nir_op_fsum4 = 287
nir_op_ftrunc = 288
nir_op_funord = 289
nir_op_funord16 = 290
nir_op_funord32 = 291
nir_op_funord8 = 292
nir_op_i2f16 = 293
nir_op_i2f32 = 294
nir_op_i2f64 = 295
nir_op_i2fmp = 296
nir_op_i2i1 = 297
nir_op_i2i16 = 298
nir_op_i2i32 = 299
nir_op_i2i64 = 300
nir_op_i2i8 = 301
nir_op_i2imp = 302
nir_op_i32csel_ge = 303
nir_op_i32csel_gt = 304
nir_op_iabs = 305
nir_op_iadd = 306
nir_op_iadd3 = 307
nir_op_iadd_sat = 308
nir_op_iand = 309
nir_op_ibfe = 310
nir_op_ibitfield_extract = 311
nir_op_icsel_eqz = 312
nir_op_idiv = 313
nir_op_ieq = 314
nir_op_ieq16 = 315
nir_op_ieq32 = 316
nir_op_ieq8 = 317
nir_op_ifind_msb = 318
nir_op_ifind_msb_rev = 319
nir_op_ige = 320
nir_op_ige16 = 321
nir_op_ige32 = 322
nir_op_ige8 = 323
nir_op_ihadd = 324
nir_op_ilea_agx = 325
nir_op_ilt = 326
nir_op_ilt16 = 327
nir_op_ilt32 = 328
nir_op_ilt8 = 329
nir_op_imad = 330
nir_op_imad24_ir3 = 331
nir_op_imadsh_mix16 = 332
nir_op_imadshl_agx = 333
nir_op_imax = 334
nir_op_imin = 335
nir_op_imod = 336
nir_op_imsubshl_agx = 337
nir_op_imul = 338
nir_op_imul24 = 339
nir_op_imul24_relaxed = 340
nir_op_imul_2x32_64 = 341
nir_op_imul_32x16 = 342
nir_op_imul_high = 343
nir_op_ine = 344
nir_op_ine16 = 345
nir_op_ine32 = 346
nir_op_ine8 = 347
nir_op_ineg = 348
nir_op_inot = 349
nir_op_insert_u16 = 350
nir_op_insert_u8 = 351
nir_op_interleave_agx = 352
nir_op_ior = 353
nir_op_irem = 354
nir_op_irhadd = 355
nir_op_ishl = 356
nir_op_ishr = 357
nir_op_isign = 358
nir_op_isub = 359
nir_op_isub_sat = 360
nir_op_ixor = 361
nir_op_ldexp = 362
nir_op_ldexp16_pan = 363
nir_op_lea_nv = 364
nir_op_mov = 365
nir_op_mqsad_4x8 = 366
nir_op_msad_4x8 = 367
nir_op_pack_2x16_to_snorm_2x8_v3d = 368
nir_op_pack_2x16_to_unorm_10_2_v3d = 369
nir_op_pack_2x16_to_unorm_2x10_v3d = 370
nir_op_pack_2x16_to_unorm_2x8_v3d = 371
nir_op_pack_2x32_to_2x16_v3d = 372
nir_op_pack_32_2x16 = 373
nir_op_pack_32_2x16_split = 374
nir_op_pack_32_4x8 = 375
nir_op_pack_32_4x8_split = 376
nir_op_pack_32_to_r11g11b10_v3d = 377
nir_op_pack_4x16_to_4x8_v3d = 378
nir_op_pack_64_2x32 = 379
nir_op_pack_64_2x32_split = 380
nir_op_pack_64_4x16 = 381
nir_op_pack_double_2x32_dxil = 382
nir_op_pack_half_2x16 = 383
nir_op_pack_half_2x16_rtz_split = 384
nir_op_pack_half_2x16_split = 385
nir_op_pack_sint_2x16 = 386
nir_op_pack_snorm_2x16 = 387
nir_op_pack_snorm_4x8 = 388
nir_op_pack_uint_2x16 = 389
nir_op_pack_uint_32_to_r10g10b10a2_v3d = 390
nir_op_pack_unorm_2x16 = 391
nir_op_pack_unorm_4x8 = 392
nir_op_pack_uvec2_to_uint = 393
nir_op_pack_uvec4_to_uint = 394
nir_op_prmt_nv = 395
nir_op_sdot_2x16_iadd = 396
nir_op_sdot_2x16_iadd_sat = 397
nir_op_sdot_4x8_iadd = 398
nir_op_sdot_4x8_iadd_sat = 399
nir_op_seq = 400
nir_op_sge = 401
nir_op_shfr = 402
nir_op_shlg_ir3 = 403
nir_op_shlm_ir3 = 404
nir_op_shrg_ir3 = 405
nir_op_shrm_ir3 = 406
nir_op_slt = 407
nir_op_sne = 408
nir_op_sudot_4x8_iadd = 409
nir_op_sudot_4x8_iadd_sat = 410
nir_op_u2f16 = 411
nir_op_u2f32 = 412
nir_op_u2f64 = 413
nir_op_u2fmp = 414
nir_op_u2u1 = 415
nir_op_u2u16 = 416
nir_op_u2u32 = 417
nir_op_u2u64 = 418
nir_op_u2u8 = 419
nir_op_uabs_isub = 420
nir_op_uabs_usub = 421
nir_op_uadd_carry = 422
nir_op_uadd_sat = 423
nir_op_ubfe = 424
nir_op_ubitfield_extract = 425
nir_op_uclz = 426
nir_op_udiv = 427
nir_op_udiv_aligned_4 = 428
nir_op_udot_2x16_uadd = 429
nir_op_udot_2x16_uadd_sat = 430
nir_op_udot_4x8_uadd = 431
nir_op_udot_4x8_uadd_sat = 432
nir_op_ufind_msb = 433
nir_op_ufind_msb_rev = 434
nir_op_uge = 435
nir_op_uge16 = 436
nir_op_uge32 = 437
nir_op_uge8 = 438
nir_op_uhadd = 439
nir_op_ulea_agx = 440
nir_op_ult = 441
nir_op_ult16 = 442
nir_op_ult32 = 443
nir_op_ult8 = 444
nir_op_umad24 = 445
nir_op_umad24_relaxed = 446
nir_op_umax = 447
nir_op_umax_4x8_vc4 = 448
nir_op_umin = 449
nir_op_umin_4x8_vc4 = 450
nir_op_umod = 451
nir_op_umul24 = 452
nir_op_umul24_relaxed = 453
nir_op_umul_2x32_64 = 454
nir_op_umul_32x16 = 455
nir_op_umul_high = 456
nir_op_umul_low = 457
nir_op_umul_unorm_4x8_vc4 = 458
nir_op_unpack_32_2x16 = 459
nir_op_unpack_32_2x16_split_x = 460
nir_op_unpack_32_2x16_split_y = 461
nir_op_unpack_32_4x8 = 462
nir_op_unpack_64_2x32 = 463
nir_op_unpack_64_2x32_split_x = 464
nir_op_unpack_64_2x32_split_y = 465
nir_op_unpack_64_4x16 = 466
nir_op_unpack_double_2x32_dxil = 467
nir_op_unpack_half_2x16 = 468
nir_op_unpack_half_2x16_split_x = 469
nir_op_unpack_half_2x16_split_y = 470
nir_op_unpack_snorm_2x16 = 471
nir_op_unpack_snorm_4x8 = 472
nir_op_unpack_unorm_2x16 = 473
nir_op_unpack_unorm_4x8 = 474
nir_op_urhadd = 475
nir_op_urol = 476
nir_op_uror = 477
nir_op_usadd_4x8_vc4 = 478
nir_op_ushr = 479
nir_op_ussub_4x8_vc4 = 480
nir_op_usub_borrow = 481
nir_op_usub_sat = 482
nir_op_vec16 = 483
nir_op_vec2 = 484
nir_op_vec3 = 485
nir_op_vec4 = 486
nir_op_vec5 = 487
nir_op_vec8 = 488
nir_last_opcode = 488
nir_num_opcodes = 489
c__EA_nir_op = ctypes.c_uint32 # enum
nir_op = c__EA_nir_op
nir_op__enumvalues = c__EA_nir_op__enumvalues
try:
    nir_type_conversion_op = _libraries['FIXME_STUB'].nir_type_conversion_op
    nir_type_conversion_op.restype = nir_op
    nir_type_conversion_op.argtypes = [nir_alu_type, nir_alu_type, nir_rounding_mode]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_atomic_op'
c__EA_nir_atomic_op__enumvalues = {
    0: 'nir_atomic_op_iadd',
    1: 'nir_atomic_op_imin',
    2: 'nir_atomic_op_umin',
    3: 'nir_atomic_op_imax',
    4: 'nir_atomic_op_umax',
    5: 'nir_atomic_op_iand',
    6: 'nir_atomic_op_ior',
    7: 'nir_atomic_op_ixor',
    8: 'nir_atomic_op_xchg',
    9: 'nir_atomic_op_fadd',
    10: 'nir_atomic_op_fmin',
    11: 'nir_atomic_op_fmax',
    12: 'nir_atomic_op_cmpxchg',
    13: 'nir_atomic_op_fcmpxchg',
    14: 'nir_atomic_op_inc_wrap',
    15: 'nir_atomic_op_dec_wrap',
    16: 'nir_atomic_op_ordered_add_gfx12_amd',
}
nir_atomic_op_iadd = 0
nir_atomic_op_imin = 1
nir_atomic_op_umin = 2
nir_atomic_op_imax = 3
nir_atomic_op_umax = 4
nir_atomic_op_iand = 5
nir_atomic_op_ior = 6
nir_atomic_op_ixor = 7
nir_atomic_op_xchg = 8
nir_atomic_op_fadd = 9
nir_atomic_op_fmin = 10
nir_atomic_op_fmax = 11
nir_atomic_op_cmpxchg = 12
nir_atomic_op_fcmpxchg = 13
nir_atomic_op_inc_wrap = 14
nir_atomic_op_dec_wrap = 15
nir_atomic_op_ordered_add_gfx12_amd = 16
c__EA_nir_atomic_op = ctypes.c_uint32 # enum
nir_atomic_op = c__EA_nir_atomic_op
nir_atomic_op__enumvalues = c__EA_nir_atomic_op__enumvalues
try:
    nir_atomic_op_type = _libraries['FIXME_STUB'].nir_atomic_op_type
    nir_atomic_op_type.restype = nir_alu_type
    nir_atomic_op_type.argtypes = [nir_atomic_op]
except AttributeError:
    pass
try:
    nir_atomic_op_to_alu = _libraries['FIXME_STUB'].nir_atomic_op_to_alu
    nir_atomic_op_to_alu.restype = nir_op
    nir_atomic_op_to_alu.argtypes = [nir_atomic_op]
except AttributeError:
    pass
try:
    nir_op_vec = _libraries['FIXME_STUB'].nir_op_vec
    nir_op_vec.restype = nir_op
    nir_op_vec.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_op_is_vec = _libraries['FIXME_STUB'].nir_op_is_vec
    nir_op_is_vec.restype = ctypes.c_bool
    nir_op_is_vec.argtypes = [nir_op]
except AttributeError:
    pass
try:
    nir_op_is_vec_or_mov = _libraries['FIXME_STUB'].nir_op_is_vec_or_mov
    nir_op_is_vec_or_mov.restype = ctypes.c_bool
    nir_op_is_vec_or_mov.argtypes = [nir_op]
except AttributeError:
    pass
try:
    nir_is_float_control_signed_zero_preserve = _libraries['FIXME_STUB'].nir_is_float_control_signed_zero_preserve
    nir_is_float_control_signed_zero_preserve.restype = ctypes.c_bool
    nir_is_float_control_signed_zero_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_float_control_inf_preserve = _libraries['FIXME_STUB'].nir_is_float_control_inf_preserve
    nir_is_float_control_inf_preserve.restype = ctypes.c_bool
    nir_is_float_control_inf_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_float_control_nan_preserve = _libraries['FIXME_STUB'].nir_is_float_control_nan_preserve
    nir_is_float_control_nan_preserve.restype = ctypes.c_bool
    nir_is_float_control_nan_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_float_control_signed_zero_inf_nan_preserve = _libraries['FIXME_STUB'].nir_is_float_control_signed_zero_inf_nan_preserve
    nir_is_float_control_signed_zero_inf_nan_preserve.restype = ctypes.c_bool
    nir_is_float_control_signed_zero_inf_nan_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_denorm_flush_to_zero = _libraries['FIXME_STUB'].nir_is_denorm_flush_to_zero
    nir_is_denorm_flush_to_zero.restype = ctypes.c_bool
    nir_is_denorm_flush_to_zero.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_denorm_preserve = _libraries['FIXME_STUB'].nir_is_denorm_preserve
    nir_is_denorm_preserve.restype = ctypes.c_bool
    nir_is_denorm_preserve.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_rounding_mode_rtne = _libraries['FIXME_STUB'].nir_is_rounding_mode_rtne
    nir_is_rounding_mode_rtne.restype = ctypes.c_bool
    nir_is_rounding_mode_rtne.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_is_rounding_mode_rtz = _libraries['FIXME_STUB'].nir_is_rounding_mode_rtz
    nir_is_rounding_mode_rtz.restype = ctypes.c_bool
    nir_is_rounding_mode_rtz.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_has_any_rounding_mode_rtz = _libraries['FIXME_STUB'].nir_has_any_rounding_mode_rtz
    nir_has_any_rounding_mode_rtz.restype = ctypes.c_bool
    nir_has_any_rounding_mode_rtz.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_has_any_rounding_mode_rtne = _libraries['FIXME_STUB'].nir_has_any_rounding_mode_rtne
    nir_has_any_rounding_mode_rtne.restype = ctypes.c_bool
    nir_has_any_rounding_mode_rtne.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_get_rounding_mode_from_float_controls = _libraries['FIXME_STUB'].nir_get_rounding_mode_from_float_controls
    nir_get_rounding_mode_from_float_controls.restype = nir_rounding_mode
    nir_get_rounding_mode_from_float_controls.argtypes = [ctypes.c_uint32, nir_alu_type]
except AttributeError:
    pass
try:
    nir_has_any_rounding_mode_enabled = _libraries['FIXME_STUB'].nir_has_any_rounding_mode_enabled
    nir_has_any_rounding_mode_enabled.restype = ctypes.c_bool
    nir_has_any_rounding_mode_enabled.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_op_algebraic_property'
c__EA_nir_op_algebraic_property__enumvalues = {
    1: 'NIR_OP_IS_2SRC_COMMUTATIVE',
    2: 'NIR_OP_IS_ASSOCIATIVE',
    4: 'NIR_OP_IS_SELECTION',
    8: 'NIR_OP_IS_INEXACT_ASSOCIATIVE',
}
NIR_OP_IS_2SRC_COMMUTATIVE = 1
NIR_OP_IS_ASSOCIATIVE = 2
NIR_OP_IS_SELECTION = 4
NIR_OP_IS_INEXACT_ASSOCIATIVE = 8
c__EA_nir_op_algebraic_property = ctypes.c_uint32 # enum
nir_op_algebraic_property = c__EA_nir_op_algebraic_property
nir_op_algebraic_property__enumvalues = c__EA_nir_op_algebraic_property__enumvalues
class struct_nir_op_info(Structure):
    pass

struct_nir_op_info._pack_ = 1 # source:False
struct_nir_op_info._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('num_inputs', ctypes.c_ubyte),
    ('output_size', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('output_type', nir_alu_type),
    ('input_sizes', ctypes.c_ubyte * 16),
    ('input_types', c__EA_nir_alu_type * 16),
    ('algebraic_properties', nir_op_algebraic_property),
    ('is_conversion', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
]

nir_op_info = struct_nir_op_info
nir_op_infos = struct_nir_op_info * 489 # Variable struct_nir_op_info * 489
try:
    nir_op_is_selection = _libraries['FIXME_STUB'].nir_op_is_selection
    nir_op_is_selection.restype = ctypes.c_bool
    nir_op_is_selection.argtypes = [nir_op]
except AttributeError:
    pass
class struct_nir_alu_instr(Structure):
    pass

struct_nir_alu_instr._pack_ = 1 # source:False
struct_nir_alu_instr._fields_ = [
    ('instr', nir_instr),
    ('op', nir_op),
    ('exact', ctypes.c_bool, 1),
    ('no_signed_wrap', ctypes.c_bool, 1),
    ('no_unsigned_wrap', ctypes.c_bool, 1),
    ('fp_fast_math', ctypes.c_uint32, 9),
    ('PADDING_0', ctypes.c_uint32, 20),
    ('def', nir_def),
    ('src', struct_nir_alu_src * 0),
]

nir_alu_instr = struct_nir_alu_instr
try:
    nir_alu_instr_is_signed_zero_preserve = _libraries['FIXME_STUB'].nir_alu_instr_is_signed_zero_preserve
    nir_alu_instr_is_signed_zero_preserve.restype = ctypes.c_bool
    nir_alu_instr_is_signed_zero_preserve.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_alu_instr_is_inf_preserve = _libraries['FIXME_STUB'].nir_alu_instr_is_inf_preserve
    nir_alu_instr_is_inf_preserve.restype = ctypes.c_bool
    nir_alu_instr_is_inf_preserve.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_alu_instr_is_nan_preserve = _libraries['FIXME_STUB'].nir_alu_instr_is_nan_preserve
    nir_alu_instr_is_nan_preserve.restype = ctypes.c_bool
    nir_alu_instr_is_nan_preserve.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_alu_instr_is_signed_zero_inf_nan_preserve = _libraries['FIXME_STUB'].nir_alu_instr_is_signed_zero_inf_nan_preserve
    nir_alu_instr_is_signed_zero_inf_nan_preserve.restype = ctypes.c_bool
    nir_alu_instr_is_signed_zero_inf_nan_preserve.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_alu_src_copy = _libraries['FIXME_STUB'].nir_alu_src_copy
    nir_alu_src_copy.restype = None
    nir_alu_src_copy.argtypes = [ctypes.POINTER(struct_nir_alu_src), ctypes.POINTER(struct_nir_alu_src)]
except AttributeError:
    pass
try:
    nir_alu_instr_src_read_mask = _libraries['FIXME_STUB'].nir_alu_instr_src_read_mask
    nir_alu_instr_src_read_mask.restype = nir_component_mask_t
    nir_alu_instr_src_read_mask.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_ssa_alu_instr_src_components = _libraries['FIXME_STUB'].nir_ssa_alu_instr_src_components
    nir_ssa_alu_instr_src_components.restype = ctypes.c_uint32
    nir_ssa_alu_instr_src_components.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_instr_channel_used = _libraries['FIXME_STUB'].nir_alu_instr_channel_used
    nir_alu_instr_channel_used.restype = ctypes.c_bool
    nir_alu_instr_channel_used.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_instr_is_comparison = _libraries['FIXME_STUB'].nir_alu_instr_is_comparison
    nir_alu_instr_is_comparison.restype = ctypes.c_bool
    nir_alu_instr_is_comparison.argtypes = [ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_const_value_negative_equal = _libraries['FIXME_STUB'].nir_const_value_negative_equal
    nir_const_value_negative_equal.restype = ctypes.c_bool
    nir_const_value_negative_equal.argtypes = [nir_const_value, nir_const_value, nir_alu_type]
except AttributeError:
    pass
try:
    nir_alu_srcs_equal = _libraries['FIXME_STUB'].nir_alu_srcs_equal
    nir_alu_srcs_equal.restype = ctypes.c_bool
    nir_alu_srcs_equal.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_srcs_negative_equal_typed = _libraries['FIXME_STUB'].nir_alu_srcs_negative_equal_typed
    nir_alu_srcs_negative_equal_typed.restype = ctypes.c_bool
    nir_alu_srcs_negative_equal_typed.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32, nir_alu_type]
except AttributeError:
    pass
try:
    nir_alu_srcs_negative_equal = _libraries['FIXME_STUB'].nir_alu_srcs_negative_equal
    nir_alu_srcs_negative_equal.restype = ctypes.c_bool
    nir_alu_srcs_negative_equal.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_src_is_trivial_ssa = _libraries['FIXME_STUB'].nir_alu_src_is_trivial_ssa
    nir_alu_src_is_trivial_ssa.restype = ctypes.c_bool
    nir_alu_src_is_trivial_ssa.argtypes = [ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_deref_type'
c__EA_nir_deref_type__enumvalues = {
    0: 'nir_deref_type_var',
    1: 'nir_deref_type_array',
    2: 'nir_deref_type_array_wildcard',
    3: 'nir_deref_type_ptr_as_array',
    4: 'nir_deref_type_struct',
    5: 'nir_deref_type_cast',
}
nir_deref_type_var = 0
nir_deref_type_array = 1
nir_deref_type_array_wildcard = 2
nir_deref_type_ptr_as_array = 3
nir_deref_type_struct = 4
nir_deref_type_cast = 5
c__EA_nir_deref_type = ctypes.c_uint32 # enum
nir_deref_type = c__EA_nir_deref_type
nir_deref_type__enumvalues = c__EA_nir_deref_type__enumvalues
class struct_nir_deref_instr(Structure):
    pass

class union_nir_deref_instr_0(Union):
    pass

union_nir_deref_instr_0._pack_ = 1 # source:False
union_nir_deref_instr_0._fields_ = [
    ('var', ctypes.POINTER(struct_nir_variable)),
    ('parent', nir_src),
]

class union_nir_deref_instr_1(Union):
    pass

class struct_nir_deref_instr_1_arr(Structure):
    pass

struct_nir_deref_instr_1_arr._pack_ = 1 # source:False
struct_nir_deref_instr_1_arr._fields_ = [
    ('index', nir_src),
    ('in_bounds', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

class struct_nir_deref_instr_1_strct(Structure):
    pass

struct_nir_deref_instr_1_strct._pack_ = 1 # source:False
struct_nir_deref_instr_1_strct._fields_ = [
    ('index', ctypes.c_uint32),
]

class struct_nir_deref_instr_1_cast(Structure):
    pass

struct_nir_deref_instr_1_cast._pack_ = 1 # source:False
struct_nir_deref_instr_1_cast._fields_ = [
    ('ptr_stride', ctypes.c_uint32),
    ('align_mul', ctypes.c_uint32),
    ('align_offset', ctypes.c_uint32),
]

union_nir_deref_instr_1._pack_ = 1 # source:False
union_nir_deref_instr_1._fields_ = [
    ('arr', struct_nir_deref_instr_1_arr),
    ('strct', struct_nir_deref_instr_1_strct),
    ('cast', struct_nir_deref_instr_1_cast),
    ('PADDING_0', ctypes.c_ubyte * 28),
]

struct_nir_deref_instr._pack_ = 1 # source:False
struct_nir_deref_instr._anonymous_ = ('_0', '_1',)
struct_nir_deref_instr._fields_ = [
    ('instr', nir_instr),
    ('deref_type', nir_deref_type),
    ('modes', c__EA_nir_variable_mode),
    ('type', ctypes.POINTER(struct_glsl_type)),
    ('_0', union_nir_deref_instr_0),
    ('_1', union_nir_deref_instr_1),
    ('def', nir_def),
]

nir_deref_instr = struct_nir_deref_instr
try:
    nir_deref_cast_is_trivial = _libraries['FIXME_STUB'].nir_deref_cast_is_trivial
    nir_deref_cast_is_trivial.restype = ctypes.c_bool
    nir_deref_cast_is_trivial.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
nir_variable_mode = c__EA_nir_variable_mode
nir_variable_mode__enumvalues = c__EA_nir_variable_mode__enumvalues
try:
    nir_deref_mode_may_be = _libraries['FIXME_STUB'].nir_deref_mode_may_be
    nir_deref_mode_may_be.restype = ctypes.c_bool
    nir_deref_mode_may_be.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_deref_mode_must_be = _libraries['FIXME_STUB'].nir_deref_mode_must_be
    nir_deref_mode_must_be.restype = ctypes.c_bool
    nir_deref_mode_must_be.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_deref_mode_is = _libraries['FIXME_STUB'].nir_deref_mode_is
    nir_deref_mode_is.restype = ctypes.c_bool
    nir_deref_mode_is.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_deref_mode_is_one_of = _libraries['FIXME_STUB'].nir_deref_mode_is_one_of
    nir_deref_mode_is_one_of.restype = ctypes.c_bool
    nir_deref_mode_is_one_of.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_deref_mode_is_in_set = _libraries['FIXME_STUB'].nir_deref_mode_is_in_set
    nir_deref_mode_is_in_set.restype = ctypes.c_bool
    nir_deref_mode_is_in_set.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_src_as_deref = _libraries['FIXME_STUB'].nir_src_as_deref
    nir_src_as_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_src_as_deref.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_deref_instr_parent = _libraries['FIXME_STUB'].nir_deref_instr_parent
    nir_deref_instr_parent.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_deref_instr_parent.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_deref_instr_get_variable = _libraries['FIXME_STUB'].nir_deref_instr_get_variable
    nir_deref_instr_get_variable.restype = ctypes.POINTER(struct_nir_variable)
    nir_deref_instr_get_variable.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_deref_instr_has_indirect = _libraries['FIXME_STUB'].nir_deref_instr_has_indirect
    nir_deref_instr_has_indirect.restype = ctypes.c_bool
    nir_deref_instr_has_indirect.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_deref_instr_is_known_out_of_bounds = _libraries['FIXME_STUB'].nir_deref_instr_is_known_out_of_bounds
    nir_deref_instr_is_known_out_of_bounds.restype = ctypes.c_bool
    nir_deref_instr_is_known_out_of_bounds.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_deref_instr_has_complex_use_options'
c__EA_nir_deref_instr_has_complex_use_options__enumvalues = {
    1: 'nir_deref_instr_has_complex_use_allow_memcpy_src',
    2: 'nir_deref_instr_has_complex_use_allow_memcpy_dst',
    4: 'nir_deref_instr_has_complex_use_allow_atomics',
}
nir_deref_instr_has_complex_use_allow_memcpy_src = 1
nir_deref_instr_has_complex_use_allow_memcpy_dst = 2
nir_deref_instr_has_complex_use_allow_atomics = 4
c__EA_nir_deref_instr_has_complex_use_options = ctypes.c_uint32 # enum
nir_deref_instr_has_complex_use_options = c__EA_nir_deref_instr_has_complex_use_options
nir_deref_instr_has_complex_use_options__enumvalues = c__EA_nir_deref_instr_has_complex_use_options__enumvalues
try:
    nir_deref_instr_has_complex_use = _libraries['FIXME_STUB'].nir_deref_instr_has_complex_use
    nir_deref_instr_has_complex_use.restype = ctypes.c_bool
    nir_deref_instr_has_complex_use.argtypes = [ctypes.POINTER(struct_nir_deref_instr), nir_deref_instr_has_complex_use_options]
except AttributeError:
    pass
try:
    nir_deref_instr_remove_if_unused = _libraries['FIXME_STUB'].nir_deref_instr_remove_if_unused
    nir_deref_instr_remove_if_unused.restype = ctypes.c_bool
    nir_deref_instr_remove_if_unused.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_deref_instr_array_stride = _libraries['FIXME_STUB'].nir_deref_instr_array_stride
    nir_deref_instr_array_stride.restype = ctypes.c_uint32
    nir_deref_instr_array_stride.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
class struct_nir_call_instr(Structure):
    pass

class struct_nir_function(Structure):
    pass

struct_nir_call_instr._pack_ = 1 # source:False
struct_nir_call_instr._fields_ = [
    ('instr', nir_instr),
    ('callee', ctypes.POINTER(struct_nir_function)),
    ('indirect_callee', nir_src),
    ('num_params', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', struct_nir_src * 0),
]

class struct_nir_parameter(Structure):
    pass

class struct_nir_function_impl(Structure):
    pass

struct_nir_function._pack_ = 1 # source:False
struct_nir_function._fields_ = [
    ('node', struct_exec_node),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('shader', ctypes.POINTER(struct_nir_shader)),
    ('num_params', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', ctypes.POINTER(struct_nir_parameter)),
    ('impl', ctypes.POINTER(struct_nir_function_impl)),
    ('driver_attributes', ctypes.c_uint32),
    ('is_entrypoint', ctypes.c_bool),
    ('is_exported', ctypes.c_bool),
    ('is_preamble', ctypes.c_bool),
    ('should_inline', ctypes.c_bool),
    ('dont_inline', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('workgroup_size', ctypes.c_uint32 * 3),
    ('is_subroutine', ctypes.c_bool),
    ('is_tmp_globals_wrapper', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte * 2),
    ('num_subroutine_types', ctypes.c_int32),
    ('subroutine_types', ctypes.POINTER(ctypes.POINTER(struct_glsl_type))),
    ('subroutine_index', ctypes.c_int32),
    ('pass_flags', ctypes.c_uint32),
]

struct_nir_parameter._pack_ = 1 # source:False
struct_nir_parameter._fields_ = [
    ('num_components', ctypes.c_ubyte),
    ('bit_size', ctypes.c_ubyte),
    ('is_return', ctypes.c_bool),
    ('implicit_conversion_prohibited', ctypes.c_bool),
    ('is_uniform', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('mode', nir_variable_mode),
    ('driver_attributes', ctypes.c_uint32),
    ('type', ctypes.POINTER(struct_glsl_type)),
    ('name', ctypes.POINTER(ctypes.c_char)),
]


# values for enumeration 'c__EA_nir_metadata'
c__EA_nir_metadata__enumvalues = {
    0: 'nir_metadata_none',
    1: 'nir_metadata_block_index',
    2: 'nir_metadata_dominance',
    4: 'nir_metadata_live_defs',
    8: 'nir_metadata_not_properly_reset',
    16: 'nir_metadata_loop_analysis',
    32: 'nir_metadata_instr_index',
    64: 'nir_metadata_divergence',
    3: 'nir_metadata_control_flow',
    -9: 'nir_metadata_all',
}
nir_metadata_none = 0
nir_metadata_block_index = 1
nir_metadata_dominance = 2
nir_metadata_live_defs = 4
nir_metadata_not_properly_reset = 8
nir_metadata_loop_analysis = 16
nir_metadata_instr_index = 32
nir_metadata_divergence = 64
nir_metadata_control_flow = 3
nir_metadata_all = -9
c__EA_nir_metadata = ctypes.c_int32 # enum
struct_nir_function_impl._pack_ = 1 # source:False
struct_nir_function_impl._fields_ = [
    ('cf_node', struct_nir_cf_node),
    ('function', ctypes.POINTER(struct_nir_function)),
    ('preamble', ctypes.POINTER(struct_nir_function)),
    ('body', struct_exec_list),
    ('end_block', ctypes.POINTER(struct_nir_block)),
    ('locals', struct_exec_list),
    ('ssa_alloc', ctypes.c_uint32),
    ('num_blocks', ctypes.c_uint32),
    ('structured', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('valid_metadata', c__EA_nir_metadata),
    ('loop_analysis_indirect_mask', nir_variable_mode),
    ('loop_analysis_force_unroll_sampler_indirect', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 3),
]

nir_call_instr = struct_nir_call_instr

# values for enumeration 'c__EA_nir_intrinsic_op'
c__EA_nir_intrinsic_op__enumvalues = {
    0: 'nir_intrinsic_accept_ray_intersection',
    1: 'nir_intrinsic_addr_mode_is',
    2: 'nir_intrinsic_al2p_nv',
    3: 'nir_intrinsic_ald_nv',
    4: 'nir_intrinsic_alpha_to_coverage',
    5: 'nir_intrinsic_as_uniform',
    6: 'nir_intrinsic_ast_nv',
    7: 'nir_intrinsic_atomic_add_gen_prim_count_amd',
    8: 'nir_intrinsic_atomic_add_gs_emit_prim_count_amd',
    9: 'nir_intrinsic_atomic_add_shader_invocation_count_amd',
    10: 'nir_intrinsic_atomic_add_xfb_prim_count_amd',
    11: 'nir_intrinsic_atomic_counter_add',
    12: 'nir_intrinsic_atomic_counter_add_deref',
    13: 'nir_intrinsic_atomic_counter_and',
    14: 'nir_intrinsic_atomic_counter_and_deref',
    15: 'nir_intrinsic_atomic_counter_comp_swap',
    16: 'nir_intrinsic_atomic_counter_comp_swap_deref',
    17: 'nir_intrinsic_atomic_counter_exchange',
    18: 'nir_intrinsic_atomic_counter_exchange_deref',
    19: 'nir_intrinsic_atomic_counter_inc',
    20: 'nir_intrinsic_atomic_counter_inc_deref',
    21: 'nir_intrinsic_atomic_counter_max',
    22: 'nir_intrinsic_atomic_counter_max_deref',
    23: 'nir_intrinsic_atomic_counter_min',
    24: 'nir_intrinsic_atomic_counter_min_deref',
    25: 'nir_intrinsic_atomic_counter_or',
    26: 'nir_intrinsic_atomic_counter_or_deref',
    27: 'nir_intrinsic_atomic_counter_post_dec',
    28: 'nir_intrinsic_atomic_counter_post_dec_deref',
    29: 'nir_intrinsic_atomic_counter_pre_dec',
    30: 'nir_intrinsic_atomic_counter_pre_dec_deref',
    31: 'nir_intrinsic_atomic_counter_read',
    32: 'nir_intrinsic_atomic_counter_read_deref',
    33: 'nir_intrinsic_atomic_counter_xor',
    34: 'nir_intrinsic_atomic_counter_xor_deref',
    35: 'nir_intrinsic_ballot',
    36: 'nir_intrinsic_ballot_bit_count_exclusive',
    37: 'nir_intrinsic_ballot_bit_count_inclusive',
    38: 'nir_intrinsic_ballot_bit_count_reduce',
    39: 'nir_intrinsic_ballot_bitfield_extract',
    40: 'nir_intrinsic_ballot_find_lsb',
    41: 'nir_intrinsic_ballot_find_msb',
    42: 'nir_intrinsic_ballot_relaxed',
    43: 'nir_intrinsic_bar_break_nv',
    44: 'nir_intrinsic_bar_set_nv',
    45: 'nir_intrinsic_bar_sync_nv',
    46: 'nir_intrinsic_barrier',
    47: 'nir_intrinsic_begin_invocation_interlock',
    48: 'nir_intrinsic_bindgen_return',
    49: 'nir_intrinsic_bindless_image_agx',
    50: 'nir_intrinsic_bindless_image_atomic',
    51: 'nir_intrinsic_bindless_image_atomic_swap',
    52: 'nir_intrinsic_bindless_image_descriptor_amd',
    53: 'nir_intrinsic_bindless_image_format',
    54: 'nir_intrinsic_bindless_image_fragment_mask_load_amd',
    55: 'nir_intrinsic_bindless_image_levels',
    56: 'nir_intrinsic_bindless_image_load',
    57: 'nir_intrinsic_bindless_image_load_raw_intel',
    58: 'nir_intrinsic_bindless_image_order',
    59: 'nir_intrinsic_bindless_image_samples',
    60: 'nir_intrinsic_bindless_image_samples_identical',
    61: 'nir_intrinsic_bindless_image_size',
    62: 'nir_intrinsic_bindless_image_sparse_load',
    63: 'nir_intrinsic_bindless_image_store',
    64: 'nir_intrinsic_bindless_image_store_block_agx',
    65: 'nir_intrinsic_bindless_image_store_raw_intel',
    66: 'nir_intrinsic_bindless_image_texel_address',
    67: 'nir_intrinsic_bindless_resource_ir3',
    68: 'nir_intrinsic_bindless_sampler_agx',
    69: 'nir_intrinsic_brcst_active_ir3',
    70: 'nir_intrinsic_btd_retire_intel',
    71: 'nir_intrinsic_btd_spawn_intel',
    72: 'nir_intrinsic_btd_stack_push_intel',
    73: 'nir_intrinsic_bvh64_intersect_ray_amd',
    74: 'nir_intrinsic_bvh8_intersect_ray_amd',
    75: 'nir_intrinsic_bvh_stack_rtn_amd',
    76: 'nir_intrinsic_cmat_binary_op',
    77: 'nir_intrinsic_cmat_bitcast',
    78: 'nir_intrinsic_cmat_construct',
    79: 'nir_intrinsic_cmat_convert',
    80: 'nir_intrinsic_cmat_copy',
    81: 'nir_intrinsic_cmat_extract',
    82: 'nir_intrinsic_cmat_insert',
    83: 'nir_intrinsic_cmat_length',
    84: 'nir_intrinsic_cmat_load',
    85: 'nir_intrinsic_cmat_muladd',
    86: 'nir_intrinsic_cmat_muladd_amd',
    87: 'nir_intrinsic_cmat_muladd_nv',
    88: 'nir_intrinsic_cmat_scalar_op',
    89: 'nir_intrinsic_cmat_store',
    90: 'nir_intrinsic_cmat_transpose',
    91: 'nir_intrinsic_cmat_unary_op',
    92: 'nir_intrinsic_convert_alu_types',
    93: 'nir_intrinsic_convert_cmat_intel',
    94: 'nir_intrinsic_copy_deref',
    95: 'nir_intrinsic_copy_fs_outputs_nv',
    96: 'nir_intrinsic_copy_global_to_uniform_ir3',
    97: 'nir_intrinsic_copy_push_const_to_uniform_ir3',
    98: 'nir_intrinsic_copy_ubo_to_uniform_ir3',
    99: 'nir_intrinsic_ddx',
    100: 'nir_intrinsic_ddx_coarse',
    101: 'nir_intrinsic_ddx_fine',
    102: 'nir_intrinsic_ddy',
    103: 'nir_intrinsic_ddy_coarse',
    104: 'nir_intrinsic_ddy_fine',
    105: 'nir_intrinsic_debug_break',
    106: 'nir_intrinsic_decl_reg',
    107: 'nir_intrinsic_demote',
    108: 'nir_intrinsic_demote_if',
    109: 'nir_intrinsic_demote_samples',
    110: 'nir_intrinsic_deref_atomic',
    111: 'nir_intrinsic_deref_atomic_swap',
    112: 'nir_intrinsic_deref_buffer_array_length',
    113: 'nir_intrinsic_deref_implicit_array_length',
    114: 'nir_intrinsic_deref_mode_is',
    115: 'nir_intrinsic_deref_texture_src',
    116: 'nir_intrinsic_doorbell_agx',
    117: 'nir_intrinsic_dpas_intel',
    118: 'nir_intrinsic_dpp16_shift_amd',
    119: 'nir_intrinsic_elect',
    120: 'nir_intrinsic_elect_any_ir3',
    121: 'nir_intrinsic_emit_primitive_poly',
    122: 'nir_intrinsic_emit_vertex',
    123: 'nir_intrinsic_emit_vertex_nv',
    124: 'nir_intrinsic_emit_vertex_with_counter',
    125: 'nir_intrinsic_end_invocation_interlock',
    126: 'nir_intrinsic_end_primitive',
    127: 'nir_intrinsic_end_primitive_nv',
    128: 'nir_intrinsic_end_primitive_with_counter',
    129: 'nir_intrinsic_enqueue_node_payloads',
    130: 'nir_intrinsic_exclusive_scan',
    131: 'nir_intrinsic_exclusive_scan_clusters_ir3',
    132: 'nir_intrinsic_execute_callable',
    133: 'nir_intrinsic_execute_closest_hit_amd',
    134: 'nir_intrinsic_execute_miss_amd',
    135: 'nir_intrinsic_export_agx',
    136: 'nir_intrinsic_export_amd',
    137: 'nir_intrinsic_export_dual_src_blend_amd',
    138: 'nir_intrinsic_export_row_amd',
    139: 'nir_intrinsic_fence_helper_exit_agx',
    140: 'nir_intrinsic_fence_mem_to_tex_agx',
    141: 'nir_intrinsic_fence_pbe_to_tex_agx',
    142: 'nir_intrinsic_fence_pbe_to_tex_pixel_agx',
    143: 'nir_intrinsic_final_primitive_nv',
    144: 'nir_intrinsic_finalize_incoming_node_payload',
    145: 'nir_intrinsic_first_invocation',
    146: 'nir_intrinsic_fs_out_nv',
    147: 'nir_intrinsic_gds_atomic_add_amd',
    148: 'nir_intrinsic_get_ssbo_size',
    149: 'nir_intrinsic_get_ubo_size',
    150: 'nir_intrinsic_global_atomic',
    151: 'nir_intrinsic_global_atomic_2x32',
    152: 'nir_intrinsic_global_atomic_agx',
    153: 'nir_intrinsic_global_atomic_amd',
    154: 'nir_intrinsic_global_atomic_swap',
    155: 'nir_intrinsic_global_atomic_swap_2x32',
    156: 'nir_intrinsic_global_atomic_swap_agx',
    157: 'nir_intrinsic_global_atomic_swap_amd',
    158: 'nir_intrinsic_ignore_ray_intersection',
    159: 'nir_intrinsic_imadsp_nv',
    160: 'nir_intrinsic_image_atomic',
    161: 'nir_intrinsic_image_atomic_swap',
    162: 'nir_intrinsic_image_deref_atomic',
    163: 'nir_intrinsic_image_deref_atomic_swap',
    164: 'nir_intrinsic_image_deref_descriptor_amd',
    165: 'nir_intrinsic_image_deref_format',
    166: 'nir_intrinsic_image_deref_fragment_mask_load_amd',
    167: 'nir_intrinsic_image_deref_levels',
    168: 'nir_intrinsic_image_deref_load',
    169: 'nir_intrinsic_image_deref_load_info_nv',
    170: 'nir_intrinsic_image_deref_load_param_intel',
    171: 'nir_intrinsic_image_deref_load_raw_intel',
    172: 'nir_intrinsic_image_deref_order',
    173: 'nir_intrinsic_image_deref_samples',
    174: 'nir_intrinsic_image_deref_samples_identical',
    175: 'nir_intrinsic_image_deref_size',
    176: 'nir_intrinsic_image_deref_sparse_load',
    177: 'nir_intrinsic_image_deref_store',
    178: 'nir_intrinsic_image_deref_store_block_agx',
    179: 'nir_intrinsic_image_deref_store_raw_intel',
    180: 'nir_intrinsic_image_deref_texel_address',
    181: 'nir_intrinsic_image_descriptor_amd',
    182: 'nir_intrinsic_image_format',
    183: 'nir_intrinsic_image_fragment_mask_load_amd',
    184: 'nir_intrinsic_image_levels',
    185: 'nir_intrinsic_image_load',
    186: 'nir_intrinsic_image_load_raw_intel',
    187: 'nir_intrinsic_image_order',
    188: 'nir_intrinsic_image_samples',
    189: 'nir_intrinsic_image_samples_identical',
    190: 'nir_intrinsic_image_size',
    191: 'nir_intrinsic_image_sparse_load',
    192: 'nir_intrinsic_image_store',
    193: 'nir_intrinsic_image_store_block_agx',
    194: 'nir_intrinsic_image_store_raw_intel',
    195: 'nir_intrinsic_image_texel_address',
    196: 'nir_intrinsic_inclusive_scan',
    197: 'nir_intrinsic_inclusive_scan_clusters_ir3',
    198: 'nir_intrinsic_initialize_node_payloads',
    199: 'nir_intrinsic_interp_deref_at_centroid',
    200: 'nir_intrinsic_interp_deref_at_offset',
    201: 'nir_intrinsic_interp_deref_at_sample',
    202: 'nir_intrinsic_interp_deref_at_vertex',
    203: 'nir_intrinsic_inverse_ballot',
    204: 'nir_intrinsic_ipa_nv',
    205: 'nir_intrinsic_is_helper_invocation',
    206: 'nir_intrinsic_is_sparse_resident_zink',
    207: 'nir_intrinsic_is_sparse_texels_resident',
    208: 'nir_intrinsic_is_subgroup_invocation_lt_amd',
    209: 'nir_intrinsic_isberd_nv',
    210: 'nir_intrinsic_lane_permute_16_amd',
    211: 'nir_intrinsic_last_invocation',
    212: 'nir_intrinsic_launch_mesh_workgroups',
    213: 'nir_intrinsic_launch_mesh_workgroups_with_payload_deref',
    214: 'nir_intrinsic_ldc_nv',
    215: 'nir_intrinsic_ldcx_nv',
    216: 'nir_intrinsic_ldtram_nv',
    217: 'nir_intrinsic_load_aa_line_width',
    218: 'nir_intrinsic_load_accel_struct_amd',
    219: 'nir_intrinsic_load_active_samples_agx',
    220: 'nir_intrinsic_load_active_subgroup_count_agx',
    221: 'nir_intrinsic_load_active_subgroup_invocation_agx',
    222: 'nir_intrinsic_load_agx',
    223: 'nir_intrinsic_load_alpha_reference_amd',
    224: 'nir_intrinsic_load_api_sample_mask_agx',
    225: 'nir_intrinsic_load_attrib_clamp_agx',
    226: 'nir_intrinsic_load_attribute_pan',
    227: 'nir_intrinsic_load_back_face_agx',
    228: 'nir_intrinsic_load_barycentric_at_offset',
    229: 'nir_intrinsic_load_barycentric_at_offset_nv',
    230: 'nir_intrinsic_load_barycentric_at_sample',
    231: 'nir_intrinsic_load_barycentric_centroid',
    232: 'nir_intrinsic_load_barycentric_coord_at_offset',
    233: 'nir_intrinsic_load_barycentric_coord_at_sample',
    234: 'nir_intrinsic_load_barycentric_coord_centroid',
    235: 'nir_intrinsic_load_barycentric_coord_pixel',
    236: 'nir_intrinsic_load_barycentric_coord_sample',
    237: 'nir_intrinsic_load_barycentric_model',
    238: 'nir_intrinsic_load_barycentric_optimize_amd',
    239: 'nir_intrinsic_load_barycentric_pixel',
    240: 'nir_intrinsic_load_barycentric_sample',
    241: 'nir_intrinsic_load_base_global_invocation_id',
    242: 'nir_intrinsic_load_base_instance',
    243: 'nir_intrinsic_load_base_vertex',
    244: 'nir_intrinsic_load_base_workgroup_id',
    245: 'nir_intrinsic_load_blend_const_color_a_float',
    246: 'nir_intrinsic_load_blend_const_color_aaaa8888_unorm',
    247: 'nir_intrinsic_load_blend_const_color_b_float',
    248: 'nir_intrinsic_load_blend_const_color_g_float',
    249: 'nir_intrinsic_load_blend_const_color_r_float',
    250: 'nir_intrinsic_load_blend_const_color_rgba',
    251: 'nir_intrinsic_load_blend_const_color_rgba8888_unorm',
    252: 'nir_intrinsic_load_btd_global_arg_addr_intel',
    253: 'nir_intrinsic_load_btd_local_arg_addr_intel',
    254: 'nir_intrinsic_load_btd_resume_sbt_addr_intel',
    255: 'nir_intrinsic_load_btd_shader_type_intel',
    256: 'nir_intrinsic_load_btd_stack_id_intel',
    257: 'nir_intrinsic_load_buffer_amd',
    258: 'nir_intrinsic_load_callable_sbt_addr_intel',
    259: 'nir_intrinsic_load_callable_sbt_stride_intel',
    260: 'nir_intrinsic_load_clamp_vertex_color_amd',
    261: 'nir_intrinsic_load_clip_half_line_width_amd',
    262: 'nir_intrinsic_load_clip_z_coeff_agx',
    263: 'nir_intrinsic_load_coalesced_input_count',
    264: 'nir_intrinsic_load_coefficients_agx',
    265: 'nir_intrinsic_load_color0',
    266: 'nir_intrinsic_load_color1',
    267: 'nir_intrinsic_load_const_buf_base_addr_lvp',
    268: 'nir_intrinsic_load_const_ir3',
    269: 'nir_intrinsic_load_constant',
    270: 'nir_intrinsic_load_constant_agx',
    271: 'nir_intrinsic_load_constant_base_ptr',
    272: 'nir_intrinsic_load_converted_output_pan',
    273: 'nir_intrinsic_load_core_count_arm',
    274: 'nir_intrinsic_load_core_id',
    275: 'nir_intrinsic_load_core_max_id_arm',
    276: 'nir_intrinsic_load_cull_any_enabled_amd',
    277: 'nir_intrinsic_load_cull_back_face_enabled_amd',
    278: 'nir_intrinsic_load_cull_ccw_amd',
    279: 'nir_intrinsic_load_cull_front_face_enabled_amd',
    280: 'nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd',
    281: 'nir_intrinsic_load_cull_mask',
    282: 'nir_intrinsic_load_cull_mask_and_flags_amd',
    283: 'nir_intrinsic_load_cull_small_line_precision_amd',
    284: 'nir_intrinsic_load_cull_small_lines_enabled_amd',
    285: 'nir_intrinsic_load_cull_small_triangle_precision_amd',
    286: 'nir_intrinsic_load_cull_small_triangles_enabled_amd',
    287: 'nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd',
    288: 'nir_intrinsic_load_debug_log_desc_amd',
    289: 'nir_intrinsic_load_depth_never_agx',
    290: 'nir_intrinsic_load_deref',
    291: 'nir_intrinsic_load_deref_block_intel',
    292: 'nir_intrinsic_load_descriptor_set_agx',
    293: 'nir_intrinsic_load_draw_id',
    294: 'nir_intrinsic_load_esgs_vertex_stride_amd',
    295: 'nir_intrinsic_load_exported_agx',
    296: 'nir_intrinsic_load_fb_layers_v3d',
    297: 'nir_intrinsic_load_fbfetch_image_desc_amd',
    298: 'nir_intrinsic_load_fbfetch_image_fmask_desc_amd',
    299: 'nir_intrinsic_load_fep_w_v3d',
    300: 'nir_intrinsic_load_first_vertex',
    301: 'nir_intrinsic_load_fixed_point_size_agx',
    302: 'nir_intrinsic_load_flat_mask',
    303: 'nir_intrinsic_load_force_vrs_rates_amd',
    304: 'nir_intrinsic_load_frag_coord',
    305: 'nir_intrinsic_load_frag_coord_unscaled_ir3',
    306: 'nir_intrinsic_load_frag_coord_w',
    307: 'nir_intrinsic_load_frag_coord_z',
    308: 'nir_intrinsic_load_frag_coord_zw_pan',
    309: 'nir_intrinsic_load_frag_invocation_count',
    310: 'nir_intrinsic_load_frag_offset_ir3',
    311: 'nir_intrinsic_load_frag_shading_rate',
    312: 'nir_intrinsic_load_frag_size',
    313: 'nir_intrinsic_load_frag_size_ir3',
    314: 'nir_intrinsic_load_from_texture_handle_agx',
    315: 'nir_intrinsic_load_front_face',
    316: 'nir_intrinsic_load_front_face_fsign',
    317: 'nir_intrinsic_load_fs_input_interp_deltas',
    318: 'nir_intrinsic_load_fs_msaa_intel',
    319: 'nir_intrinsic_load_fully_covered',
    320: 'nir_intrinsic_load_geometry_param_buffer_poly',
    321: 'nir_intrinsic_load_global',
    322: 'nir_intrinsic_load_global_2x32',
    323: 'nir_intrinsic_load_global_amd',
    324: 'nir_intrinsic_load_global_base_ptr',
    325: 'nir_intrinsic_load_global_block_intel',
    326: 'nir_intrinsic_load_global_bounded',
    327: 'nir_intrinsic_load_global_constant',
    328: 'nir_intrinsic_load_global_constant_bounded',
    329: 'nir_intrinsic_load_global_constant_offset',
    330: 'nir_intrinsic_load_global_constant_uniform_block_intel',
    331: 'nir_intrinsic_load_global_etna',
    332: 'nir_intrinsic_load_global_invocation_id',
    333: 'nir_intrinsic_load_global_invocation_index',
    334: 'nir_intrinsic_load_global_ir3',
    335: 'nir_intrinsic_load_global_size',
    336: 'nir_intrinsic_load_gs_header_ir3',
    337: 'nir_intrinsic_load_gs_vertex_offset_amd',
    338: 'nir_intrinsic_load_gs_wave_id_amd',
    339: 'nir_intrinsic_load_helper_arg_hi_agx',
    340: 'nir_intrinsic_load_helper_arg_lo_agx',
    341: 'nir_intrinsic_load_helper_invocation',
    342: 'nir_intrinsic_load_helper_op_id_agx',
    343: 'nir_intrinsic_load_hit_attrib_amd',
    344: 'nir_intrinsic_load_hs_out_patch_data_offset_amd',
    345: 'nir_intrinsic_load_hs_patch_stride_ir3',
    346: 'nir_intrinsic_load_initial_edgeflags_amd',
    347: 'nir_intrinsic_load_inline_data_intel',
    348: 'nir_intrinsic_load_input',
    349: 'nir_intrinsic_load_input_assembly_buffer_poly',
    350: 'nir_intrinsic_load_input_attachment_conv_pan',
    351: 'nir_intrinsic_load_input_attachment_coord',
    352: 'nir_intrinsic_load_input_attachment_target_pan',
    353: 'nir_intrinsic_load_input_topology_poly',
    354: 'nir_intrinsic_load_input_vertex',
    355: 'nir_intrinsic_load_instance_id',
    356: 'nir_intrinsic_load_interpolated_input',
    357: 'nir_intrinsic_load_intersection_opaque_amd',
    358: 'nir_intrinsic_load_invocation_id',
    359: 'nir_intrinsic_load_is_first_fan_agx',
    360: 'nir_intrinsic_load_is_indexed_draw',
    361: 'nir_intrinsic_load_kernel_input',
    362: 'nir_intrinsic_load_layer_id',
    363: 'nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd',
    364: 'nir_intrinsic_load_leaf_opaque_intel',
    365: 'nir_intrinsic_load_leaf_procedural_intel',
    366: 'nir_intrinsic_load_line_coord',
    367: 'nir_intrinsic_load_line_width',
    368: 'nir_intrinsic_load_local_invocation_id',
    369: 'nir_intrinsic_load_local_invocation_index',
    370: 'nir_intrinsic_load_local_pixel_agx',
    371: 'nir_intrinsic_load_local_shared_r600',
    372: 'nir_intrinsic_load_lshs_vertex_stride_amd',
    373: 'nir_intrinsic_load_max_polygon_intel',
    374: 'nir_intrinsic_load_merged_wave_info_amd',
    375: 'nir_intrinsic_load_mesh_view_count',
    376: 'nir_intrinsic_load_mesh_view_indices',
    377: 'nir_intrinsic_load_multisampled_pan',
    378: 'nir_intrinsic_load_noperspective_varyings_pan',
    379: 'nir_intrinsic_load_num_subgroups',
    380: 'nir_intrinsic_load_num_vertices',
    381: 'nir_intrinsic_load_num_vertices_per_primitive_amd',
    382: 'nir_intrinsic_load_num_workgroups',
    383: 'nir_intrinsic_load_ordered_id_amd',
    384: 'nir_intrinsic_load_output',
    385: 'nir_intrinsic_load_packed_passthrough_primitive_amd',
    386: 'nir_intrinsic_load_param',
    387: 'nir_intrinsic_load_patch_vertices_in',
    388: 'nir_intrinsic_load_per_primitive_input',
    389: 'nir_intrinsic_load_per_primitive_output',
    390: 'nir_intrinsic_load_per_primitive_remap_intel',
    391: 'nir_intrinsic_load_per_vertex_input',
    392: 'nir_intrinsic_load_per_vertex_output',
    393: 'nir_intrinsic_load_per_view_output',
    394: 'nir_intrinsic_load_persp_center_rhw_ir3',
    395: 'nir_intrinsic_load_pipeline_stat_query_enabled_amd',
    396: 'nir_intrinsic_load_pixel_coord',
    397: 'nir_intrinsic_load_point_coord',
    398: 'nir_intrinsic_load_point_coord_maybe_flipped',
    399: 'nir_intrinsic_load_poly_line_smooth_enabled',
    400: 'nir_intrinsic_load_polygon_stipple_agx',
    401: 'nir_intrinsic_load_polygon_stipple_buffer_amd',
    402: 'nir_intrinsic_load_preamble',
    403: 'nir_intrinsic_load_prim_gen_query_enabled_amd',
    404: 'nir_intrinsic_load_prim_xfb_query_enabled_amd',
    405: 'nir_intrinsic_load_primitive_id',
    406: 'nir_intrinsic_load_primitive_location_ir3',
    407: 'nir_intrinsic_load_printf_buffer_address',
    408: 'nir_intrinsic_load_printf_buffer_size',
    409: 'nir_intrinsic_load_provoking_last',
    410: 'nir_intrinsic_load_provoking_vtx_amd',
    411: 'nir_intrinsic_load_provoking_vtx_in_prim_amd',
    412: 'nir_intrinsic_load_push_constant',
    413: 'nir_intrinsic_load_push_constant_zink',
    414: 'nir_intrinsic_load_r600_per_vertex_input',
    415: 'nir_intrinsic_load_rasterization_primitive_amd',
    416: 'nir_intrinsic_load_rasterization_samples_amd',
    417: 'nir_intrinsic_load_rasterization_stream',
    418: 'nir_intrinsic_load_raw_output_pan',
    419: 'nir_intrinsic_load_raw_vertex_id_pan',
    420: 'nir_intrinsic_load_raw_vertex_offset_pan',
    421: 'nir_intrinsic_load_ray_base_mem_addr_intel',
    422: 'nir_intrinsic_load_ray_flags',
    423: 'nir_intrinsic_load_ray_geometry_index',
    424: 'nir_intrinsic_load_ray_hit_kind',
    425: 'nir_intrinsic_load_ray_hit_sbt_addr_intel',
    426: 'nir_intrinsic_load_ray_hit_sbt_stride_intel',
    427: 'nir_intrinsic_load_ray_hw_stack_size_intel',
    428: 'nir_intrinsic_load_ray_instance_custom_index',
    429: 'nir_intrinsic_load_ray_launch_id',
    430: 'nir_intrinsic_load_ray_launch_size',
    431: 'nir_intrinsic_load_ray_miss_sbt_addr_intel',
    432: 'nir_intrinsic_load_ray_miss_sbt_stride_intel',
    433: 'nir_intrinsic_load_ray_num_dss_rt_stacks_intel',
    434: 'nir_intrinsic_load_ray_object_direction',
    435: 'nir_intrinsic_load_ray_object_origin',
    436: 'nir_intrinsic_load_ray_object_to_world',
    437: 'nir_intrinsic_load_ray_query_global_intel',
    438: 'nir_intrinsic_load_ray_sw_stack_size_intel',
    439: 'nir_intrinsic_load_ray_t_max',
    440: 'nir_intrinsic_load_ray_t_min',
    441: 'nir_intrinsic_load_ray_tracing_stack_base_lvp',
    442: 'nir_intrinsic_load_ray_triangle_vertex_positions',
    443: 'nir_intrinsic_load_ray_world_direction',
    444: 'nir_intrinsic_load_ray_world_origin',
    445: 'nir_intrinsic_load_ray_world_to_object',
    446: 'nir_intrinsic_load_readonly_output_pan',
    447: 'nir_intrinsic_load_reg',
    448: 'nir_intrinsic_load_reg_indirect',
    449: 'nir_intrinsic_load_rel_patch_id_ir3',
    450: 'nir_intrinsic_load_reloc_const_intel',
    451: 'nir_intrinsic_load_resume_shader_address_amd',
    452: 'nir_intrinsic_load_ring_attr_amd',
    453: 'nir_intrinsic_load_ring_attr_offset_amd',
    454: 'nir_intrinsic_load_ring_es2gs_offset_amd',
    455: 'nir_intrinsic_load_ring_esgs_amd',
    456: 'nir_intrinsic_load_ring_gs2vs_offset_amd',
    457: 'nir_intrinsic_load_ring_gsvs_amd',
    458: 'nir_intrinsic_load_ring_mesh_scratch_amd',
    459: 'nir_intrinsic_load_ring_mesh_scratch_offset_amd',
    460: 'nir_intrinsic_load_ring_task_draw_amd',
    461: 'nir_intrinsic_load_ring_task_payload_amd',
    462: 'nir_intrinsic_load_ring_tess_factors_amd',
    463: 'nir_intrinsic_load_ring_tess_factors_offset_amd',
    464: 'nir_intrinsic_load_ring_tess_offchip_amd',
    465: 'nir_intrinsic_load_ring_tess_offchip_offset_amd',
    466: 'nir_intrinsic_load_root_agx',
    467: 'nir_intrinsic_load_rt_arg_scratch_offset_amd',
    468: 'nir_intrinsic_load_rt_conversion_pan',
    469: 'nir_intrinsic_load_sample_id',
    470: 'nir_intrinsic_load_sample_mask',
    471: 'nir_intrinsic_load_sample_mask_in',
    472: 'nir_intrinsic_load_sample_pos',
    473: 'nir_intrinsic_load_sample_pos_from_id',
    474: 'nir_intrinsic_load_sample_pos_or_center',
    475: 'nir_intrinsic_load_sample_positions_agx',
    476: 'nir_intrinsic_load_sample_positions_amd',
    477: 'nir_intrinsic_load_sample_positions_pan',
    478: 'nir_intrinsic_load_sampler_handle_agx',
    479: 'nir_intrinsic_load_sampler_lod_parameters',
    480: 'nir_intrinsic_load_samples_log2_agx',
    481: 'nir_intrinsic_load_sbt_base_amd',
    482: 'nir_intrinsic_load_sbt_offset_amd',
    483: 'nir_intrinsic_load_sbt_stride_amd',
    484: 'nir_intrinsic_load_scalar_arg_amd',
    485: 'nir_intrinsic_load_scratch',
    486: 'nir_intrinsic_load_scratch_base_ptr',
    487: 'nir_intrinsic_load_shader_call_data_offset_lvp',
    488: 'nir_intrinsic_load_shader_index',
    489: 'nir_intrinsic_load_shader_output_pan',
    490: 'nir_intrinsic_load_shader_part_tests_zs_agx',
    491: 'nir_intrinsic_load_shader_record_ptr',
    492: 'nir_intrinsic_load_shared',
    493: 'nir_intrinsic_load_shared2_amd',
    494: 'nir_intrinsic_load_shared_base_ptr',
    495: 'nir_intrinsic_load_shared_block_intel',
    496: 'nir_intrinsic_load_shared_ir3',
    497: 'nir_intrinsic_load_shared_lock_nv',
    498: 'nir_intrinsic_load_shared_uniform_block_intel',
    499: 'nir_intrinsic_load_simd_width_intel',
    500: 'nir_intrinsic_load_sm_count_nv',
    501: 'nir_intrinsic_load_sm_id_nv',
    502: 'nir_intrinsic_load_smem_amd',
    503: 'nir_intrinsic_load_ssbo',
    504: 'nir_intrinsic_load_ssbo_address',
    505: 'nir_intrinsic_load_ssbo_block_intel',
    506: 'nir_intrinsic_load_ssbo_intel',
    507: 'nir_intrinsic_load_ssbo_ir3',
    508: 'nir_intrinsic_load_ssbo_uniform_block_intel',
    509: 'nir_intrinsic_load_stack',
    510: 'nir_intrinsic_load_stat_query_address_agx',
    511: 'nir_intrinsic_load_streamout_buffer_amd',
    512: 'nir_intrinsic_load_streamout_config_amd',
    513: 'nir_intrinsic_load_streamout_offset_amd',
    514: 'nir_intrinsic_load_streamout_write_index_amd',
    515: 'nir_intrinsic_load_subgroup_eq_mask',
    516: 'nir_intrinsic_load_subgroup_ge_mask',
    517: 'nir_intrinsic_load_subgroup_gt_mask',
    518: 'nir_intrinsic_load_subgroup_id',
    519: 'nir_intrinsic_load_subgroup_id_shift_ir3',
    520: 'nir_intrinsic_load_subgroup_invocation',
    521: 'nir_intrinsic_load_subgroup_le_mask',
    522: 'nir_intrinsic_load_subgroup_lt_mask',
    523: 'nir_intrinsic_load_subgroup_size',
    524: 'nir_intrinsic_load_sysval_agx',
    525: 'nir_intrinsic_load_sysval_nv',
    526: 'nir_intrinsic_load_task_payload',
    527: 'nir_intrinsic_load_task_ring_entry_amd',
    528: 'nir_intrinsic_load_tcs_header_ir3',
    529: 'nir_intrinsic_load_tcs_in_param_base_r600',
    530: 'nir_intrinsic_load_tcs_mem_attrib_stride',
    531: 'nir_intrinsic_load_tcs_num_patches_amd',
    532: 'nir_intrinsic_load_tcs_out_param_base_r600',
    533: 'nir_intrinsic_load_tcs_primitive_mode_amd',
    534: 'nir_intrinsic_load_tcs_rel_patch_id_r600',
    535: 'nir_intrinsic_load_tcs_tess_factor_base_r600',
    536: 'nir_intrinsic_load_tcs_tess_levels_to_tes_amd',
    537: 'nir_intrinsic_load_tess_coord',
    538: 'nir_intrinsic_load_tess_coord_xy',
    539: 'nir_intrinsic_load_tess_factor_base_ir3',
    540: 'nir_intrinsic_load_tess_level_inner',
    541: 'nir_intrinsic_load_tess_level_inner_default',
    542: 'nir_intrinsic_load_tess_level_outer',
    543: 'nir_intrinsic_load_tess_level_outer_default',
    544: 'nir_intrinsic_load_tess_param_base_ir3',
    545: 'nir_intrinsic_load_tess_param_buffer_poly',
    546: 'nir_intrinsic_load_tess_rel_patch_id_amd',
    547: 'nir_intrinsic_load_tex_sprite_mask_agx',
    548: 'nir_intrinsic_load_texture_handle_agx',
    549: 'nir_intrinsic_load_texture_scale',
    550: 'nir_intrinsic_load_texture_size_etna',
    551: 'nir_intrinsic_load_tlb_color_brcm',
    552: 'nir_intrinsic_load_topology_id_intel',
    553: 'nir_intrinsic_load_typed_buffer_amd',
    554: 'nir_intrinsic_load_uav_ir3',
    555: 'nir_intrinsic_load_ubo',
    556: 'nir_intrinsic_load_ubo_uniform_block_intel',
    557: 'nir_intrinsic_load_ubo_vec4',
    558: 'nir_intrinsic_load_uniform',
    559: 'nir_intrinsic_load_user_clip_plane',
    560: 'nir_intrinsic_load_user_data_amd',
    561: 'nir_intrinsic_load_uvs_index_agx',
    562: 'nir_intrinsic_load_vbo_base_agx',
    563: 'nir_intrinsic_load_vbo_stride_agx',
    564: 'nir_intrinsic_load_vector_arg_amd',
    565: 'nir_intrinsic_load_vertex_id',
    566: 'nir_intrinsic_load_vertex_id_zero_base',
    567: 'nir_intrinsic_load_view_index',
    568: 'nir_intrinsic_load_viewport_offset',
    569: 'nir_intrinsic_load_viewport_scale',
    570: 'nir_intrinsic_load_viewport_x_offset',
    571: 'nir_intrinsic_load_viewport_x_scale',
    572: 'nir_intrinsic_load_viewport_y_offset',
    573: 'nir_intrinsic_load_viewport_y_scale',
    574: 'nir_intrinsic_load_viewport_z_offset',
    575: 'nir_intrinsic_load_viewport_z_scale',
    576: 'nir_intrinsic_load_vs_output_buffer_poly',
    577: 'nir_intrinsic_load_vs_outputs_poly',
    578: 'nir_intrinsic_load_vs_primitive_stride_ir3',
    579: 'nir_intrinsic_load_vs_vertex_stride_ir3',
    580: 'nir_intrinsic_load_vulkan_descriptor',
    581: 'nir_intrinsic_load_warp_id_arm',
    582: 'nir_intrinsic_load_warp_id_nv',
    583: 'nir_intrinsic_load_warp_max_id_arm',
    584: 'nir_intrinsic_load_warps_per_sm_nv',
    585: 'nir_intrinsic_load_work_dim',
    586: 'nir_intrinsic_load_workgroup_id',
    587: 'nir_intrinsic_load_workgroup_index',
    588: 'nir_intrinsic_load_workgroup_num_input_primitives_amd',
    589: 'nir_intrinsic_load_workgroup_num_input_vertices_amd',
    590: 'nir_intrinsic_load_workgroup_size',
    591: 'nir_intrinsic_load_xfb_address',
    592: 'nir_intrinsic_load_xfb_index_buffer',
    593: 'nir_intrinsic_load_xfb_size',
    594: 'nir_intrinsic_load_xfb_state_address_gfx12_amd',
    595: 'nir_intrinsic_masked_swizzle_amd',
    596: 'nir_intrinsic_mbcnt_amd',
    597: 'nir_intrinsic_memcpy_deref',
    598: 'nir_intrinsic_nop',
    599: 'nir_intrinsic_nop_amd',
    600: 'nir_intrinsic_optimization_barrier_sgpr_amd',
    601: 'nir_intrinsic_optimization_barrier_vgpr_amd',
    602: 'nir_intrinsic_ordered_add_loop_gfx12_amd',
    603: 'nir_intrinsic_ordered_xfb_counter_add_gfx11_amd',
    604: 'nir_intrinsic_overwrite_tes_arguments_amd',
    605: 'nir_intrinsic_overwrite_vs_arguments_amd',
    606: 'nir_intrinsic_pin_cx_handle_nv',
    607: 'nir_intrinsic_preamble_end_ir3',
    608: 'nir_intrinsic_preamble_start_ir3',
    609: 'nir_intrinsic_prefetch_sam_ir3',
    610: 'nir_intrinsic_prefetch_tex_ir3',
    611: 'nir_intrinsic_prefetch_ubo_ir3',
    612: 'nir_intrinsic_printf',
    613: 'nir_intrinsic_printf_abort',
    614: 'nir_intrinsic_quad_ballot_agx',
    615: 'nir_intrinsic_quad_broadcast',
    616: 'nir_intrinsic_quad_swap_diagonal',
    617: 'nir_intrinsic_quad_swap_horizontal',
    618: 'nir_intrinsic_quad_swap_vertical',
    619: 'nir_intrinsic_quad_swizzle_amd',
    620: 'nir_intrinsic_quad_vote_all',
    621: 'nir_intrinsic_quad_vote_any',
    622: 'nir_intrinsic_r600_indirect_vertex_at_index',
    623: 'nir_intrinsic_ray_intersection_ir3',
    624: 'nir_intrinsic_read_attribute_payload_intel',
    625: 'nir_intrinsic_read_first_invocation',
    626: 'nir_intrinsic_read_getlast_ir3',
    627: 'nir_intrinsic_read_invocation',
    628: 'nir_intrinsic_read_invocation_cond_ir3',
    629: 'nir_intrinsic_reduce',
    630: 'nir_intrinsic_reduce_clusters_ir3',
    631: 'nir_intrinsic_report_ray_intersection',
    632: 'nir_intrinsic_resource_intel',
    633: 'nir_intrinsic_rotate',
    634: 'nir_intrinsic_rq_confirm_intersection',
    635: 'nir_intrinsic_rq_generate_intersection',
    636: 'nir_intrinsic_rq_initialize',
    637: 'nir_intrinsic_rq_load',
    638: 'nir_intrinsic_rq_proceed',
    639: 'nir_intrinsic_rq_terminate',
    640: 'nir_intrinsic_rt_execute_callable',
    641: 'nir_intrinsic_rt_resume',
    642: 'nir_intrinsic_rt_return_amd',
    643: 'nir_intrinsic_rt_trace_ray',
    644: 'nir_intrinsic_sample_mask_agx',
    645: 'nir_intrinsic_select_vertex_poly',
    646: 'nir_intrinsic_sendmsg_amd',
    647: 'nir_intrinsic_set_vertex_and_primitive_count',
    648: 'nir_intrinsic_shader_clock',
    649: 'nir_intrinsic_shared_append_amd',
    650: 'nir_intrinsic_shared_atomic',
    651: 'nir_intrinsic_shared_atomic_swap',
    652: 'nir_intrinsic_shared_consume_amd',
    653: 'nir_intrinsic_shuffle',
    654: 'nir_intrinsic_shuffle_down',
    655: 'nir_intrinsic_shuffle_down_uniform_ir3',
    656: 'nir_intrinsic_shuffle_up',
    657: 'nir_intrinsic_shuffle_up_uniform_ir3',
    658: 'nir_intrinsic_shuffle_xor',
    659: 'nir_intrinsic_shuffle_xor_uniform_ir3',
    660: 'nir_intrinsic_sleep_amd',
    661: 'nir_intrinsic_sparse_residency_code_and',
    662: 'nir_intrinsic_ssa_bar_nv',
    663: 'nir_intrinsic_ssbo_atomic',
    664: 'nir_intrinsic_ssbo_atomic_ir3',
    665: 'nir_intrinsic_ssbo_atomic_swap',
    666: 'nir_intrinsic_ssbo_atomic_swap_ir3',
    667: 'nir_intrinsic_stack_map_agx',
    668: 'nir_intrinsic_stack_unmap_agx',
    669: 'nir_intrinsic_store_agx',
    670: 'nir_intrinsic_store_buffer_amd',
    671: 'nir_intrinsic_store_combined_output_pan',
    672: 'nir_intrinsic_store_const_ir3',
    673: 'nir_intrinsic_store_deref',
    674: 'nir_intrinsic_store_deref_block_intel',
    675: 'nir_intrinsic_store_global',
    676: 'nir_intrinsic_store_global_2x32',
    677: 'nir_intrinsic_store_global_amd',
    678: 'nir_intrinsic_store_global_block_intel',
    679: 'nir_intrinsic_store_global_etna',
    680: 'nir_intrinsic_store_global_ir3',
    681: 'nir_intrinsic_store_hit_attrib_amd',
    682: 'nir_intrinsic_store_local_pixel_agx',
    683: 'nir_intrinsic_store_local_shared_r600',
    684: 'nir_intrinsic_store_output',
    685: 'nir_intrinsic_store_per_primitive_output',
    686: 'nir_intrinsic_store_per_primitive_payload_intel',
    687: 'nir_intrinsic_store_per_vertex_output',
    688: 'nir_intrinsic_store_per_view_output',
    689: 'nir_intrinsic_store_preamble',
    690: 'nir_intrinsic_store_raw_output_pan',
    691: 'nir_intrinsic_store_reg',
    692: 'nir_intrinsic_store_reg_indirect',
    693: 'nir_intrinsic_store_scalar_arg_amd',
    694: 'nir_intrinsic_store_scratch',
    695: 'nir_intrinsic_store_shared',
    696: 'nir_intrinsic_store_shared2_amd',
    697: 'nir_intrinsic_store_shared_block_intel',
    698: 'nir_intrinsic_store_shared_ir3',
    699: 'nir_intrinsic_store_shared_unlock_nv',
    700: 'nir_intrinsic_store_ssbo',
    701: 'nir_intrinsic_store_ssbo_block_intel',
    702: 'nir_intrinsic_store_ssbo_intel',
    703: 'nir_intrinsic_store_ssbo_ir3',
    704: 'nir_intrinsic_store_stack',
    705: 'nir_intrinsic_store_task_payload',
    706: 'nir_intrinsic_store_tf_r600',
    707: 'nir_intrinsic_store_tlb_sample_color_v3d',
    708: 'nir_intrinsic_store_uvs_agx',
    709: 'nir_intrinsic_store_vector_arg_amd',
    710: 'nir_intrinsic_store_zs_agx',
    711: 'nir_intrinsic_strict_wqm_coord_amd',
    712: 'nir_intrinsic_subfm_nv',
    713: 'nir_intrinsic_suclamp_nv',
    714: 'nir_intrinsic_sueau_nv',
    715: 'nir_intrinsic_suldga_nv',
    716: 'nir_intrinsic_sustga_nv',
    717: 'nir_intrinsic_task_payload_atomic',
    718: 'nir_intrinsic_task_payload_atomic_swap',
    719: 'nir_intrinsic_terminate',
    720: 'nir_intrinsic_terminate_if',
    721: 'nir_intrinsic_terminate_ray',
    722: 'nir_intrinsic_trace_ray',
    723: 'nir_intrinsic_trace_ray_intel',
    724: 'nir_intrinsic_unit_test_amd',
    725: 'nir_intrinsic_unit_test_divergent_amd',
    726: 'nir_intrinsic_unit_test_uniform_amd',
    727: 'nir_intrinsic_unpin_cx_handle_nv',
    728: 'nir_intrinsic_use',
    729: 'nir_intrinsic_vild_nv',
    730: 'nir_intrinsic_vote_all',
    731: 'nir_intrinsic_vote_any',
    732: 'nir_intrinsic_vote_feq',
    733: 'nir_intrinsic_vote_ieq',
    734: 'nir_intrinsic_vulkan_resource_index',
    735: 'nir_intrinsic_vulkan_resource_reindex',
    736: 'nir_intrinsic_write_invocation_amd',
    737: 'nir_intrinsic_xfb_counter_sub_gfx11_amd',
    737: 'nir_last_intrinsic',
    738: 'nir_num_intrinsics',
}
nir_intrinsic_accept_ray_intersection = 0
nir_intrinsic_addr_mode_is = 1
nir_intrinsic_al2p_nv = 2
nir_intrinsic_ald_nv = 3
nir_intrinsic_alpha_to_coverage = 4
nir_intrinsic_as_uniform = 5
nir_intrinsic_ast_nv = 6
nir_intrinsic_atomic_add_gen_prim_count_amd = 7
nir_intrinsic_atomic_add_gs_emit_prim_count_amd = 8
nir_intrinsic_atomic_add_shader_invocation_count_amd = 9
nir_intrinsic_atomic_add_xfb_prim_count_amd = 10
nir_intrinsic_atomic_counter_add = 11
nir_intrinsic_atomic_counter_add_deref = 12
nir_intrinsic_atomic_counter_and = 13
nir_intrinsic_atomic_counter_and_deref = 14
nir_intrinsic_atomic_counter_comp_swap = 15
nir_intrinsic_atomic_counter_comp_swap_deref = 16
nir_intrinsic_atomic_counter_exchange = 17
nir_intrinsic_atomic_counter_exchange_deref = 18
nir_intrinsic_atomic_counter_inc = 19
nir_intrinsic_atomic_counter_inc_deref = 20
nir_intrinsic_atomic_counter_max = 21
nir_intrinsic_atomic_counter_max_deref = 22
nir_intrinsic_atomic_counter_min = 23
nir_intrinsic_atomic_counter_min_deref = 24
nir_intrinsic_atomic_counter_or = 25
nir_intrinsic_atomic_counter_or_deref = 26
nir_intrinsic_atomic_counter_post_dec = 27
nir_intrinsic_atomic_counter_post_dec_deref = 28
nir_intrinsic_atomic_counter_pre_dec = 29
nir_intrinsic_atomic_counter_pre_dec_deref = 30
nir_intrinsic_atomic_counter_read = 31
nir_intrinsic_atomic_counter_read_deref = 32
nir_intrinsic_atomic_counter_xor = 33
nir_intrinsic_atomic_counter_xor_deref = 34
nir_intrinsic_ballot = 35
nir_intrinsic_ballot_bit_count_exclusive = 36
nir_intrinsic_ballot_bit_count_inclusive = 37
nir_intrinsic_ballot_bit_count_reduce = 38
nir_intrinsic_ballot_bitfield_extract = 39
nir_intrinsic_ballot_find_lsb = 40
nir_intrinsic_ballot_find_msb = 41
nir_intrinsic_ballot_relaxed = 42
nir_intrinsic_bar_break_nv = 43
nir_intrinsic_bar_set_nv = 44
nir_intrinsic_bar_sync_nv = 45
nir_intrinsic_barrier = 46
nir_intrinsic_begin_invocation_interlock = 47
nir_intrinsic_bindgen_return = 48
nir_intrinsic_bindless_image_agx = 49
nir_intrinsic_bindless_image_atomic = 50
nir_intrinsic_bindless_image_atomic_swap = 51
nir_intrinsic_bindless_image_descriptor_amd = 52
nir_intrinsic_bindless_image_format = 53
nir_intrinsic_bindless_image_fragment_mask_load_amd = 54
nir_intrinsic_bindless_image_levels = 55
nir_intrinsic_bindless_image_load = 56
nir_intrinsic_bindless_image_load_raw_intel = 57
nir_intrinsic_bindless_image_order = 58
nir_intrinsic_bindless_image_samples = 59
nir_intrinsic_bindless_image_samples_identical = 60
nir_intrinsic_bindless_image_size = 61
nir_intrinsic_bindless_image_sparse_load = 62
nir_intrinsic_bindless_image_store = 63
nir_intrinsic_bindless_image_store_block_agx = 64
nir_intrinsic_bindless_image_store_raw_intel = 65
nir_intrinsic_bindless_image_texel_address = 66
nir_intrinsic_bindless_resource_ir3 = 67
nir_intrinsic_bindless_sampler_agx = 68
nir_intrinsic_brcst_active_ir3 = 69
nir_intrinsic_btd_retire_intel = 70
nir_intrinsic_btd_spawn_intel = 71
nir_intrinsic_btd_stack_push_intel = 72
nir_intrinsic_bvh64_intersect_ray_amd = 73
nir_intrinsic_bvh8_intersect_ray_amd = 74
nir_intrinsic_bvh_stack_rtn_amd = 75
nir_intrinsic_cmat_binary_op = 76
nir_intrinsic_cmat_bitcast = 77
nir_intrinsic_cmat_construct = 78
nir_intrinsic_cmat_convert = 79
nir_intrinsic_cmat_copy = 80
nir_intrinsic_cmat_extract = 81
nir_intrinsic_cmat_insert = 82
nir_intrinsic_cmat_length = 83
nir_intrinsic_cmat_load = 84
nir_intrinsic_cmat_muladd = 85
nir_intrinsic_cmat_muladd_amd = 86
nir_intrinsic_cmat_muladd_nv = 87
nir_intrinsic_cmat_scalar_op = 88
nir_intrinsic_cmat_store = 89
nir_intrinsic_cmat_transpose = 90
nir_intrinsic_cmat_unary_op = 91
nir_intrinsic_convert_alu_types = 92
nir_intrinsic_convert_cmat_intel = 93
nir_intrinsic_copy_deref = 94
nir_intrinsic_copy_fs_outputs_nv = 95
nir_intrinsic_copy_global_to_uniform_ir3 = 96
nir_intrinsic_copy_push_const_to_uniform_ir3 = 97
nir_intrinsic_copy_ubo_to_uniform_ir3 = 98
nir_intrinsic_ddx = 99
nir_intrinsic_ddx_coarse = 100
nir_intrinsic_ddx_fine = 101
nir_intrinsic_ddy = 102
nir_intrinsic_ddy_coarse = 103
nir_intrinsic_ddy_fine = 104
nir_intrinsic_debug_break = 105
nir_intrinsic_decl_reg = 106
nir_intrinsic_demote = 107
nir_intrinsic_demote_if = 108
nir_intrinsic_demote_samples = 109
nir_intrinsic_deref_atomic = 110
nir_intrinsic_deref_atomic_swap = 111
nir_intrinsic_deref_buffer_array_length = 112
nir_intrinsic_deref_implicit_array_length = 113
nir_intrinsic_deref_mode_is = 114
nir_intrinsic_deref_texture_src = 115
nir_intrinsic_doorbell_agx = 116
nir_intrinsic_dpas_intel = 117
nir_intrinsic_dpp16_shift_amd = 118
nir_intrinsic_elect = 119
nir_intrinsic_elect_any_ir3 = 120
nir_intrinsic_emit_primitive_poly = 121
nir_intrinsic_emit_vertex = 122
nir_intrinsic_emit_vertex_nv = 123
nir_intrinsic_emit_vertex_with_counter = 124
nir_intrinsic_end_invocation_interlock = 125
nir_intrinsic_end_primitive = 126
nir_intrinsic_end_primitive_nv = 127
nir_intrinsic_end_primitive_with_counter = 128
nir_intrinsic_enqueue_node_payloads = 129
nir_intrinsic_exclusive_scan = 130
nir_intrinsic_exclusive_scan_clusters_ir3 = 131
nir_intrinsic_execute_callable = 132
nir_intrinsic_execute_closest_hit_amd = 133
nir_intrinsic_execute_miss_amd = 134
nir_intrinsic_export_agx = 135
nir_intrinsic_export_amd = 136
nir_intrinsic_export_dual_src_blend_amd = 137
nir_intrinsic_export_row_amd = 138
nir_intrinsic_fence_helper_exit_agx = 139
nir_intrinsic_fence_mem_to_tex_agx = 140
nir_intrinsic_fence_pbe_to_tex_agx = 141
nir_intrinsic_fence_pbe_to_tex_pixel_agx = 142
nir_intrinsic_final_primitive_nv = 143
nir_intrinsic_finalize_incoming_node_payload = 144
nir_intrinsic_first_invocation = 145
nir_intrinsic_fs_out_nv = 146
nir_intrinsic_gds_atomic_add_amd = 147
nir_intrinsic_get_ssbo_size = 148
nir_intrinsic_get_ubo_size = 149
nir_intrinsic_global_atomic = 150
nir_intrinsic_global_atomic_2x32 = 151
nir_intrinsic_global_atomic_agx = 152
nir_intrinsic_global_atomic_amd = 153
nir_intrinsic_global_atomic_swap = 154
nir_intrinsic_global_atomic_swap_2x32 = 155
nir_intrinsic_global_atomic_swap_agx = 156
nir_intrinsic_global_atomic_swap_amd = 157
nir_intrinsic_ignore_ray_intersection = 158
nir_intrinsic_imadsp_nv = 159
nir_intrinsic_image_atomic = 160
nir_intrinsic_image_atomic_swap = 161
nir_intrinsic_image_deref_atomic = 162
nir_intrinsic_image_deref_atomic_swap = 163
nir_intrinsic_image_deref_descriptor_amd = 164
nir_intrinsic_image_deref_format = 165
nir_intrinsic_image_deref_fragment_mask_load_amd = 166
nir_intrinsic_image_deref_levels = 167
nir_intrinsic_image_deref_load = 168
nir_intrinsic_image_deref_load_info_nv = 169
nir_intrinsic_image_deref_load_param_intel = 170
nir_intrinsic_image_deref_load_raw_intel = 171
nir_intrinsic_image_deref_order = 172
nir_intrinsic_image_deref_samples = 173
nir_intrinsic_image_deref_samples_identical = 174
nir_intrinsic_image_deref_size = 175
nir_intrinsic_image_deref_sparse_load = 176
nir_intrinsic_image_deref_store = 177
nir_intrinsic_image_deref_store_block_agx = 178
nir_intrinsic_image_deref_store_raw_intel = 179
nir_intrinsic_image_deref_texel_address = 180
nir_intrinsic_image_descriptor_amd = 181
nir_intrinsic_image_format = 182
nir_intrinsic_image_fragment_mask_load_amd = 183
nir_intrinsic_image_levels = 184
nir_intrinsic_image_load = 185
nir_intrinsic_image_load_raw_intel = 186
nir_intrinsic_image_order = 187
nir_intrinsic_image_samples = 188
nir_intrinsic_image_samples_identical = 189
nir_intrinsic_image_size = 190
nir_intrinsic_image_sparse_load = 191
nir_intrinsic_image_store = 192
nir_intrinsic_image_store_block_agx = 193
nir_intrinsic_image_store_raw_intel = 194
nir_intrinsic_image_texel_address = 195
nir_intrinsic_inclusive_scan = 196
nir_intrinsic_inclusive_scan_clusters_ir3 = 197
nir_intrinsic_initialize_node_payloads = 198
nir_intrinsic_interp_deref_at_centroid = 199
nir_intrinsic_interp_deref_at_offset = 200
nir_intrinsic_interp_deref_at_sample = 201
nir_intrinsic_interp_deref_at_vertex = 202
nir_intrinsic_inverse_ballot = 203
nir_intrinsic_ipa_nv = 204
nir_intrinsic_is_helper_invocation = 205
nir_intrinsic_is_sparse_resident_zink = 206
nir_intrinsic_is_sparse_texels_resident = 207
nir_intrinsic_is_subgroup_invocation_lt_amd = 208
nir_intrinsic_isberd_nv = 209
nir_intrinsic_lane_permute_16_amd = 210
nir_intrinsic_last_invocation = 211
nir_intrinsic_launch_mesh_workgroups = 212
nir_intrinsic_launch_mesh_workgroups_with_payload_deref = 213
nir_intrinsic_ldc_nv = 214
nir_intrinsic_ldcx_nv = 215
nir_intrinsic_ldtram_nv = 216
nir_intrinsic_load_aa_line_width = 217
nir_intrinsic_load_accel_struct_amd = 218
nir_intrinsic_load_active_samples_agx = 219
nir_intrinsic_load_active_subgroup_count_agx = 220
nir_intrinsic_load_active_subgroup_invocation_agx = 221
nir_intrinsic_load_agx = 222
nir_intrinsic_load_alpha_reference_amd = 223
nir_intrinsic_load_api_sample_mask_agx = 224
nir_intrinsic_load_attrib_clamp_agx = 225
nir_intrinsic_load_attribute_pan = 226
nir_intrinsic_load_back_face_agx = 227
nir_intrinsic_load_barycentric_at_offset = 228
nir_intrinsic_load_barycentric_at_offset_nv = 229
nir_intrinsic_load_barycentric_at_sample = 230
nir_intrinsic_load_barycentric_centroid = 231
nir_intrinsic_load_barycentric_coord_at_offset = 232
nir_intrinsic_load_barycentric_coord_at_sample = 233
nir_intrinsic_load_barycentric_coord_centroid = 234
nir_intrinsic_load_barycentric_coord_pixel = 235
nir_intrinsic_load_barycentric_coord_sample = 236
nir_intrinsic_load_barycentric_model = 237
nir_intrinsic_load_barycentric_optimize_amd = 238
nir_intrinsic_load_barycentric_pixel = 239
nir_intrinsic_load_barycentric_sample = 240
nir_intrinsic_load_base_global_invocation_id = 241
nir_intrinsic_load_base_instance = 242
nir_intrinsic_load_base_vertex = 243
nir_intrinsic_load_base_workgroup_id = 244
nir_intrinsic_load_blend_const_color_a_float = 245
nir_intrinsic_load_blend_const_color_aaaa8888_unorm = 246
nir_intrinsic_load_blend_const_color_b_float = 247
nir_intrinsic_load_blend_const_color_g_float = 248
nir_intrinsic_load_blend_const_color_r_float = 249
nir_intrinsic_load_blend_const_color_rgba = 250
nir_intrinsic_load_blend_const_color_rgba8888_unorm = 251
nir_intrinsic_load_btd_global_arg_addr_intel = 252
nir_intrinsic_load_btd_local_arg_addr_intel = 253
nir_intrinsic_load_btd_resume_sbt_addr_intel = 254
nir_intrinsic_load_btd_shader_type_intel = 255
nir_intrinsic_load_btd_stack_id_intel = 256
nir_intrinsic_load_buffer_amd = 257
nir_intrinsic_load_callable_sbt_addr_intel = 258
nir_intrinsic_load_callable_sbt_stride_intel = 259
nir_intrinsic_load_clamp_vertex_color_amd = 260
nir_intrinsic_load_clip_half_line_width_amd = 261
nir_intrinsic_load_clip_z_coeff_agx = 262
nir_intrinsic_load_coalesced_input_count = 263
nir_intrinsic_load_coefficients_agx = 264
nir_intrinsic_load_color0 = 265
nir_intrinsic_load_color1 = 266
nir_intrinsic_load_const_buf_base_addr_lvp = 267
nir_intrinsic_load_const_ir3 = 268
nir_intrinsic_load_constant = 269
nir_intrinsic_load_constant_agx = 270
nir_intrinsic_load_constant_base_ptr = 271
nir_intrinsic_load_converted_output_pan = 272
nir_intrinsic_load_core_count_arm = 273
nir_intrinsic_load_core_id = 274
nir_intrinsic_load_core_max_id_arm = 275
nir_intrinsic_load_cull_any_enabled_amd = 276
nir_intrinsic_load_cull_back_face_enabled_amd = 277
nir_intrinsic_load_cull_ccw_amd = 278
nir_intrinsic_load_cull_front_face_enabled_amd = 279
nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd = 280
nir_intrinsic_load_cull_mask = 281
nir_intrinsic_load_cull_mask_and_flags_amd = 282
nir_intrinsic_load_cull_small_line_precision_amd = 283
nir_intrinsic_load_cull_small_lines_enabled_amd = 284
nir_intrinsic_load_cull_small_triangle_precision_amd = 285
nir_intrinsic_load_cull_small_triangles_enabled_amd = 286
nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd = 287
nir_intrinsic_load_debug_log_desc_amd = 288
nir_intrinsic_load_depth_never_agx = 289
nir_intrinsic_load_deref = 290
nir_intrinsic_load_deref_block_intel = 291
nir_intrinsic_load_descriptor_set_agx = 292
nir_intrinsic_load_draw_id = 293
nir_intrinsic_load_esgs_vertex_stride_amd = 294
nir_intrinsic_load_exported_agx = 295
nir_intrinsic_load_fb_layers_v3d = 296
nir_intrinsic_load_fbfetch_image_desc_amd = 297
nir_intrinsic_load_fbfetch_image_fmask_desc_amd = 298
nir_intrinsic_load_fep_w_v3d = 299
nir_intrinsic_load_first_vertex = 300
nir_intrinsic_load_fixed_point_size_agx = 301
nir_intrinsic_load_flat_mask = 302
nir_intrinsic_load_force_vrs_rates_amd = 303
nir_intrinsic_load_frag_coord = 304
nir_intrinsic_load_frag_coord_unscaled_ir3 = 305
nir_intrinsic_load_frag_coord_w = 306
nir_intrinsic_load_frag_coord_z = 307
nir_intrinsic_load_frag_coord_zw_pan = 308
nir_intrinsic_load_frag_invocation_count = 309
nir_intrinsic_load_frag_offset_ir3 = 310
nir_intrinsic_load_frag_shading_rate = 311
nir_intrinsic_load_frag_size = 312
nir_intrinsic_load_frag_size_ir3 = 313
nir_intrinsic_load_from_texture_handle_agx = 314
nir_intrinsic_load_front_face = 315
nir_intrinsic_load_front_face_fsign = 316
nir_intrinsic_load_fs_input_interp_deltas = 317
nir_intrinsic_load_fs_msaa_intel = 318
nir_intrinsic_load_fully_covered = 319
nir_intrinsic_load_geometry_param_buffer_poly = 320
nir_intrinsic_load_global = 321
nir_intrinsic_load_global_2x32 = 322
nir_intrinsic_load_global_amd = 323
nir_intrinsic_load_global_base_ptr = 324
nir_intrinsic_load_global_block_intel = 325
nir_intrinsic_load_global_bounded = 326
nir_intrinsic_load_global_constant = 327
nir_intrinsic_load_global_constant_bounded = 328
nir_intrinsic_load_global_constant_offset = 329
nir_intrinsic_load_global_constant_uniform_block_intel = 330
nir_intrinsic_load_global_etna = 331
nir_intrinsic_load_global_invocation_id = 332
nir_intrinsic_load_global_invocation_index = 333
nir_intrinsic_load_global_ir3 = 334
nir_intrinsic_load_global_size = 335
nir_intrinsic_load_gs_header_ir3 = 336
nir_intrinsic_load_gs_vertex_offset_amd = 337
nir_intrinsic_load_gs_wave_id_amd = 338
nir_intrinsic_load_helper_arg_hi_agx = 339
nir_intrinsic_load_helper_arg_lo_agx = 340
nir_intrinsic_load_helper_invocation = 341
nir_intrinsic_load_helper_op_id_agx = 342
nir_intrinsic_load_hit_attrib_amd = 343
nir_intrinsic_load_hs_out_patch_data_offset_amd = 344
nir_intrinsic_load_hs_patch_stride_ir3 = 345
nir_intrinsic_load_initial_edgeflags_amd = 346
nir_intrinsic_load_inline_data_intel = 347
nir_intrinsic_load_input = 348
nir_intrinsic_load_input_assembly_buffer_poly = 349
nir_intrinsic_load_input_attachment_conv_pan = 350
nir_intrinsic_load_input_attachment_coord = 351
nir_intrinsic_load_input_attachment_target_pan = 352
nir_intrinsic_load_input_topology_poly = 353
nir_intrinsic_load_input_vertex = 354
nir_intrinsic_load_instance_id = 355
nir_intrinsic_load_interpolated_input = 356
nir_intrinsic_load_intersection_opaque_amd = 357
nir_intrinsic_load_invocation_id = 358
nir_intrinsic_load_is_first_fan_agx = 359
nir_intrinsic_load_is_indexed_draw = 360
nir_intrinsic_load_kernel_input = 361
nir_intrinsic_load_layer_id = 362
nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd = 363
nir_intrinsic_load_leaf_opaque_intel = 364
nir_intrinsic_load_leaf_procedural_intel = 365
nir_intrinsic_load_line_coord = 366
nir_intrinsic_load_line_width = 367
nir_intrinsic_load_local_invocation_id = 368
nir_intrinsic_load_local_invocation_index = 369
nir_intrinsic_load_local_pixel_agx = 370
nir_intrinsic_load_local_shared_r600 = 371
nir_intrinsic_load_lshs_vertex_stride_amd = 372
nir_intrinsic_load_max_polygon_intel = 373
nir_intrinsic_load_merged_wave_info_amd = 374
nir_intrinsic_load_mesh_view_count = 375
nir_intrinsic_load_mesh_view_indices = 376
nir_intrinsic_load_multisampled_pan = 377
nir_intrinsic_load_noperspective_varyings_pan = 378
nir_intrinsic_load_num_subgroups = 379
nir_intrinsic_load_num_vertices = 380
nir_intrinsic_load_num_vertices_per_primitive_amd = 381
nir_intrinsic_load_num_workgroups = 382
nir_intrinsic_load_ordered_id_amd = 383
nir_intrinsic_load_output = 384
nir_intrinsic_load_packed_passthrough_primitive_amd = 385
nir_intrinsic_load_param = 386
nir_intrinsic_load_patch_vertices_in = 387
nir_intrinsic_load_per_primitive_input = 388
nir_intrinsic_load_per_primitive_output = 389
nir_intrinsic_load_per_primitive_remap_intel = 390
nir_intrinsic_load_per_vertex_input = 391
nir_intrinsic_load_per_vertex_output = 392
nir_intrinsic_load_per_view_output = 393
nir_intrinsic_load_persp_center_rhw_ir3 = 394
nir_intrinsic_load_pipeline_stat_query_enabled_amd = 395
nir_intrinsic_load_pixel_coord = 396
nir_intrinsic_load_point_coord = 397
nir_intrinsic_load_point_coord_maybe_flipped = 398
nir_intrinsic_load_poly_line_smooth_enabled = 399
nir_intrinsic_load_polygon_stipple_agx = 400
nir_intrinsic_load_polygon_stipple_buffer_amd = 401
nir_intrinsic_load_preamble = 402
nir_intrinsic_load_prim_gen_query_enabled_amd = 403
nir_intrinsic_load_prim_xfb_query_enabled_amd = 404
nir_intrinsic_load_primitive_id = 405
nir_intrinsic_load_primitive_location_ir3 = 406
nir_intrinsic_load_printf_buffer_address = 407
nir_intrinsic_load_printf_buffer_size = 408
nir_intrinsic_load_provoking_last = 409
nir_intrinsic_load_provoking_vtx_amd = 410
nir_intrinsic_load_provoking_vtx_in_prim_amd = 411
nir_intrinsic_load_push_constant = 412
nir_intrinsic_load_push_constant_zink = 413
nir_intrinsic_load_r600_per_vertex_input = 414
nir_intrinsic_load_rasterization_primitive_amd = 415
nir_intrinsic_load_rasterization_samples_amd = 416
nir_intrinsic_load_rasterization_stream = 417
nir_intrinsic_load_raw_output_pan = 418
nir_intrinsic_load_raw_vertex_id_pan = 419
nir_intrinsic_load_raw_vertex_offset_pan = 420
nir_intrinsic_load_ray_base_mem_addr_intel = 421
nir_intrinsic_load_ray_flags = 422
nir_intrinsic_load_ray_geometry_index = 423
nir_intrinsic_load_ray_hit_kind = 424
nir_intrinsic_load_ray_hit_sbt_addr_intel = 425
nir_intrinsic_load_ray_hit_sbt_stride_intel = 426
nir_intrinsic_load_ray_hw_stack_size_intel = 427
nir_intrinsic_load_ray_instance_custom_index = 428
nir_intrinsic_load_ray_launch_id = 429
nir_intrinsic_load_ray_launch_size = 430
nir_intrinsic_load_ray_miss_sbt_addr_intel = 431
nir_intrinsic_load_ray_miss_sbt_stride_intel = 432
nir_intrinsic_load_ray_num_dss_rt_stacks_intel = 433
nir_intrinsic_load_ray_object_direction = 434
nir_intrinsic_load_ray_object_origin = 435
nir_intrinsic_load_ray_object_to_world = 436
nir_intrinsic_load_ray_query_global_intel = 437
nir_intrinsic_load_ray_sw_stack_size_intel = 438
nir_intrinsic_load_ray_t_max = 439
nir_intrinsic_load_ray_t_min = 440
nir_intrinsic_load_ray_tracing_stack_base_lvp = 441
nir_intrinsic_load_ray_triangle_vertex_positions = 442
nir_intrinsic_load_ray_world_direction = 443
nir_intrinsic_load_ray_world_origin = 444
nir_intrinsic_load_ray_world_to_object = 445
nir_intrinsic_load_readonly_output_pan = 446
nir_intrinsic_load_reg = 447
nir_intrinsic_load_reg_indirect = 448
nir_intrinsic_load_rel_patch_id_ir3 = 449
nir_intrinsic_load_reloc_const_intel = 450
nir_intrinsic_load_resume_shader_address_amd = 451
nir_intrinsic_load_ring_attr_amd = 452
nir_intrinsic_load_ring_attr_offset_amd = 453
nir_intrinsic_load_ring_es2gs_offset_amd = 454
nir_intrinsic_load_ring_esgs_amd = 455
nir_intrinsic_load_ring_gs2vs_offset_amd = 456
nir_intrinsic_load_ring_gsvs_amd = 457
nir_intrinsic_load_ring_mesh_scratch_amd = 458
nir_intrinsic_load_ring_mesh_scratch_offset_amd = 459
nir_intrinsic_load_ring_task_draw_amd = 460
nir_intrinsic_load_ring_task_payload_amd = 461
nir_intrinsic_load_ring_tess_factors_amd = 462
nir_intrinsic_load_ring_tess_factors_offset_amd = 463
nir_intrinsic_load_ring_tess_offchip_amd = 464
nir_intrinsic_load_ring_tess_offchip_offset_amd = 465
nir_intrinsic_load_root_agx = 466
nir_intrinsic_load_rt_arg_scratch_offset_amd = 467
nir_intrinsic_load_rt_conversion_pan = 468
nir_intrinsic_load_sample_id = 469
nir_intrinsic_load_sample_mask = 470
nir_intrinsic_load_sample_mask_in = 471
nir_intrinsic_load_sample_pos = 472
nir_intrinsic_load_sample_pos_from_id = 473
nir_intrinsic_load_sample_pos_or_center = 474
nir_intrinsic_load_sample_positions_agx = 475
nir_intrinsic_load_sample_positions_amd = 476
nir_intrinsic_load_sample_positions_pan = 477
nir_intrinsic_load_sampler_handle_agx = 478
nir_intrinsic_load_sampler_lod_parameters = 479
nir_intrinsic_load_samples_log2_agx = 480
nir_intrinsic_load_sbt_base_amd = 481
nir_intrinsic_load_sbt_offset_amd = 482
nir_intrinsic_load_sbt_stride_amd = 483
nir_intrinsic_load_scalar_arg_amd = 484
nir_intrinsic_load_scratch = 485
nir_intrinsic_load_scratch_base_ptr = 486
nir_intrinsic_load_shader_call_data_offset_lvp = 487
nir_intrinsic_load_shader_index = 488
nir_intrinsic_load_shader_output_pan = 489
nir_intrinsic_load_shader_part_tests_zs_agx = 490
nir_intrinsic_load_shader_record_ptr = 491
nir_intrinsic_load_shared = 492
nir_intrinsic_load_shared2_amd = 493
nir_intrinsic_load_shared_base_ptr = 494
nir_intrinsic_load_shared_block_intel = 495
nir_intrinsic_load_shared_ir3 = 496
nir_intrinsic_load_shared_lock_nv = 497
nir_intrinsic_load_shared_uniform_block_intel = 498
nir_intrinsic_load_simd_width_intel = 499
nir_intrinsic_load_sm_count_nv = 500
nir_intrinsic_load_sm_id_nv = 501
nir_intrinsic_load_smem_amd = 502
nir_intrinsic_load_ssbo = 503
nir_intrinsic_load_ssbo_address = 504
nir_intrinsic_load_ssbo_block_intel = 505
nir_intrinsic_load_ssbo_intel = 506
nir_intrinsic_load_ssbo_ir3 = 507
nir_intrinsic_load_ssbo_uniform_block_intel = 508
nir_intrinsic_load_stack = 509
nir_intrinsic_load_stat_query_address_agx = 510
nir_intrinsic_load_streamout_buffer_amd = 511
nir_intrinsic_load_streamout_config_amd = 512
nir_intrinsic_load_streamout_offset_amd = 513
nir_intrinsic_load_streamout_write_index_amd = 514
nir_intrinsic_load_subgroup_eq_mask = 515
nir_intrinsic_load_subgroup_ge_mask = 516
nir_intrinsic_load_subgroup_gt_mask = 517
nir_intrinsic_load_subgroup_id = 518
nir_intrinsic_load_subgroup_id_shift_ir3 = 519
nir_intrinsic_load_subgroup_invocation = 520
nir_intrinsic_load_subgroup_le_mask = 521
nir_intrinsic_load_subgroup_lt_mask = 522
nir_intrinsic_load_subgroup_size = 523
nir_intrinsic_load_sysval_agx = 524
nir_intrinsic_load_sysval_nv = 525
nir_intrinsic_load_task_payload = 526
nir_intrinsic_load_task_ring_entry_amd = 527
nir_intrinsic_load_tcs_header_ir3 = 528
nir_intrinsic_load_tcs_in_param_base_r600 = 529
nir_intrinsic_load_tcs_mem_attrib_stride = 530
nir_intrinsic_load_tcs_num_patches_amd = 531
nir_intrinsic_load_tcs_out_param_base_r600 = 532
nir_intrinsic_load_tcs_primitive_mode_amd = 533
nir_intrinsic_load_tcs_rel_patch_id_r600 = 534
nir_intrinsic_load_tcs_tess_factor_base_r600 = 535
nir_intrinsic_load_tcs_tess_levels_to_tes_amd = 536
nir_intrinsic_load_tess_coord = 537
nir_intrinsic_load_tess_coord_xy = 538
nir_intrinsic_load_tess_factor_base_ir3 = 539
nir_intrinsic_load_tess_level_inner = 540
nir_intrinsic_load_tess_level_inner_default = 541
nir_intrinsic_load_tess_level_outer = 542
nir_intrinsic_load_tess_level_outer_default = 543
nir_intrinsic_load_tess_param_base_ir3 = 544
nir_intrinsic_load_tess_param_buffer_poly = 545
nir_intrinsic_load_tess_rel_patch_id_amd = 546
nir_intrinsic_load_tex_sprite_mask_agx = 547
nir_intrinsic_load_texture_handle_agx = 548
nir_intrinsic_load_texture_scale = 549
nir_intrinsic_load_texture_size_etna = 550
nir_intrinsic_load_tlb_color_brcm = 551
nir_intrinsic_load_topology_id_intel = 552
nir_intrinsic_load_typed_buffer_amd = 553
nir_intrinsic_load_uav_ir3 = 554
nir_intrinsic_load_ubo = 555
nir_intrinsic_load_ubo_uniform_block_intel = 556
nir_intrinsic_load_ubo_vec4 = 557
nir_intrinsic_load_uniform = 558
nir_intrinsic_load_user_clip_plane = 559
nir_intrinsic_load_user_data_amd = 560
nir_intrinsic_load_uvs_index_agx = 561
nir_intrinsic_load_vbo_base_agx = 562
nir_intrinsic_load_vbo_stride_agx = 563
nir_intrinsic_load_vector_arg_amd = 564
nir_intrinsic_load_vertex_id = 565
nir_intrinsic_load_vertex_id_zero_base = 566
nir_intrinsic_load_view_index = 567
nir_intrinsic_load_viewport_offset = 568
nir_intrinsic_load_viewport_scale = 569
nir_intrinsic_load_viewport_x_offset = 570
nir_intrinsic_load_viewport_x_scale = 571
nir_intrinsic_load_viewport_y_offset = 572
nir_intrinsic_load_viewport_y_scale = 573
nir_intrinsic_load_viewport_z_offset = 574
nir_intrinsic_load_viewport_z_scale = 575
nir_intrinsic_load_vs_output_buffer_poly = 576
nir_intrinsic_load_vs_outputs_poly = 577
nir_intrinsic_load_vs_primitive_stride_ir3 = 578
nir_intrinsic_load_vs_vertex_stride_ir3 = 579
nir_intrinsic_load_vulkan_descriptor = 580
nir_intrinsic_load_warp_id_arm = 581
nir_intrinsic_load_warp_id_nv = 582
nir_intrinsic_load_warp_max_id_arm = 583
nir_intrinsic_load_warps_per_sm_nv = 584
nir_intrinsic_load_work_dim = 585
nir_intrinsic_load_workgroup_id = 586
nir_intrinsic_load_workgroup_index = 587
nir_intrinsic_load_workgroup_num_input_primitives_amd = 588
nir_intrinsic_load_workgroup_num_input_vertices_amd = 589
nir_intrinsic_load_workgroup_size = 590
nir_intrinsic_load_xfb_address = 591
nir_intrinsic_load_xfb_index_buffer = 592
nir_intrinsic_load_xfb_size = 593
nir_intrinsic_load_xfb_state_address_gfx12_amd = 594
nir_intrinsic_masked_swizzle_amd = 595
nir_intrinsic_mbcnt_amd = 596
nir_intrinsic_memcpy_deref = 597
nir_intrinsic_nop = 598
nir_intrinsic_nop_amd = 599
nir_intrinsic_optimization_barrier_sgpr_amd = 600
nir_intrinsic_optimization_barrier_vgpr_amd = 601
nir_intrinsic_ordered_add_loop_gfx12_amd = 602
nir_intrinsic_ordered_xfb_counter_add_gfx11_amd = 603
nir_intrinsic_overwrite_tes_arguments_amd = 604
nir_intrinsic_overwrite_vs_arguments_amd = 605
nir_intrinsic_pin_cx_handle_nv = 606
nir_intrinsic_preamble_end_ir3 = 607
nir_intrinsic_preamble_start_ir3 = 608
nir_intrinsic_prefetch_sam_ir3 = 609
nir_intrinsic_prefetch_tex_ir3 = 610
nir_intrinsic_prefetch_ubo_ir3 = 611
nir_intrinsic_printf = 612
nir_intrinsic_printf_abort = 613
nir_intrinsic_quad_ballot_agx = 614
nir_intrinsic_quad_broadcast = 615
nir_intrinsic_quad_swap_diagonal = 616
nir_intrinsic_quad_swap_horizontal = 617
nir_intrinsic_quad_swap_vertical = 618
nir_intrinsic_quad_swizzle_amd = 619
nir_intrinsic_quad_vote_all = 620
nir_intrinsic_quad_vote_any = 621
nir_intrinsic_r600_indirect_vertex_at_index = 622
nir_intrinsic_ray_intersection_ir3 = 623
nir_intrinsic_read_attribute_payload_intel = 624
nir_intrinsic_read_first_invocation = 625
nir_intrinsic_read_getlast_ir3 = 626
nir_intrinsic_read_invocation = 627
nir_intrinsic_read_invocation_cond_ir3 = 628
nir_intrinsic_reduce = 629
nir_intrinsic_reduce_clusters_ir3 = 630
nir_intrinsic_report_ray_intersection = 631
nir_intrinsic_resource_intel = 632
nir_intrinsic_rotate = 633
nir_intrinsic_rq_confirm_intersection = 634
nir_intrinsic_rq_generate_intersection = 635
nir_intrinsic_rq_initialize = 636
nir_intrinsic_rq_load = 637
nir_intrinsic_rq_proceed = 638
nir_intrinsic_rq_terminate = 639
nir_intrinsic_rt_execute_callable = 640
nir_intrinsic_rt_resume = 641
nir_intrinsic_rt_return_amd = 642
nir_intrinsic_rt_trace_ray = 643
nir_intrinsic_sample_mask_agx = 644
nir_intrinsic_select_vertex_poly = 645
nir_intrinsic_sendmsg_amd = 646
nir_intrinsic_set_vertex_and_primitive_count = 647
nir_intrinsic_shader_clock = 648
nir_intrinsic_shared_append_amd = 649
nir_intrinsic_shared_atomic = 650
nir_intrinsic_shared_atomic_swap = 651
nir_intrinsic_shared_consume_amd = 652
nir_intrinsic_shuffle = 653
nir_intrinsic_shuffle_down = 654
nir_intrinsic_shuffle_down_uniform_ir3 = 655
nir_intrinsic_shuffle_up = 656
nir_intrinsic_shuffle_up_uniform_ir3 = 657
nir_intrinsic_shuffle_xor = 658
nir_intrinsic_shuffle_xor_uniform_ir3 = 659
nir_intrinsic_sleep_amd = 660
nir_intrinsic_sparse_residency_code_and = 661
nir_intrinsic_ssa_bar_nv = 662
nir_intrinsic_ssbo_atomic = 663
nir_intrinsic_ssbo_atomic_ir3 = 664
nir_intrinsic_ssbo_atomic_swap = 665
nir_intrinsic_ssbo_atomic_swap_ir3 = 666
nir_intrinsic_stack_map_agx = 667
nir_intrinsic_stack_unmap_agx = 668
nir_intrinsic_store_agx = 669
nir_intrinsic_store_buffer_amd = 670
nir_intrinsic_store_combined_output_pan = 671
nir_intrinsic_store_const_ir3 = 672
nir_intrinsic_store_deref = 673
nir_intrinsic_store_deref_block_intel = 674
nir_intrinsic_store_global = 675
nir_intrinsic_store_global_2x32 = 676
nir_intrinsic_store_global_amd = 677
nir_intrinsic_store_global_block_intel = 678
nir_intrinsic_store_global_etna = 679
nir_intrinsic_store_global_ir3 = 680
nir_intrinsic_store_hit_attrib_amd = 681
nir_intrinsic_store_local_pixel_agx = 682
nir_intrinsic_store_local_shared_r600 = 683
nir_intrinsic_store_output = 684
nir_intrinsic_store_per_primitive_output = 685
nir_intrinsic_store_per_primitive_payload_intel = 686
nir_intrinsic_store_per_vertex_output = 687
nir_intrinsic_store_per_view_output = 688
nir_intrinsic_store_preamble = 689
nir_intrinsic_store_raw_output_pan = 690
nir_intrinsic_store_reg = 691
nir_intrinsic_store_reg_indirect = 692
nir_intrinsic_store_scalar_arg_amd = 693
nir_intrinsic_store_scratch = 694
nir_intrinsic_store_shared = 695
nir_intrinsic_store_shared2_amd = 696
nir_intrinsic_store_shared_block_intel = 697
nir_intrinsic_store_shared_ir3 = 698
nir_intrinsic_store_shared_unlock_nv = 699
nir_intrinsic_store_ssbo = 700
nir_intrinsic_store_ssbo_block_intel = 701
nir_intrinsic_store_ssbo_intel = 702
nir_intrinsic_store_ssbo_ir3 = 703
nir_intrinsic_store_stack = 704
nir_intrinsic_store_task_payload = 705
nir_intrinsic_store_tf_r600 = 706
nir_intrinsic_store_tlb_sample_color_v3d = 707
nir_intrinsic_store_uvs_agx = 708
nir_intrinsic_store_vector_arg_amd = 709
nir_intrinsic_store_zs_agx = 710
nir_intrinsic_strict_wqm_coord_amd = 711
nir_intrinsic_subfm_nv = 712
nir_intrinsic_suclamp_nv = 713
nir_intrinsic_sueau_nv = 714
nir_intrinsic_suldga_nv = 715
nir_intrinsic_sustga_nv = 716
nir_intrinsic_task_payload_atomic = 717
nir_intrinsic_task_payload_atomic_swap = 718
nir_intrinsic_terminate = 719
nir_intrinsic_terminate_if = 720
nir_intrinsic_terminate_ray = 721
nir_intrinsic_trace_ray = 722
nir_intrinsic_trace_ray_intel = 723
nir_intrinsic_unit_test_amd = 724
nir_intrinsic_unit_test_divergent_amd = 725
nir_intrinsic_unit_test_uniform_amd = 726
nir_intrinsic_unpin_cx_handle_nv = 727
nir_intrinsic_use = 728
nir_intrinsic_vild_nv = 729
nir_intrinsic_vote_all = 730
nir_intrinsic_vote_any = 731
nir_intrinsic_vote_feq = 732
nir_intrinsic_vote_ieq = 733
nir_intrinsic_vulkan_resource_index = 734
nir_intrinsic_vulkan_resource_reindex = 735
nir_intrinsic_write_invocation_amd = 736
nir_intrinsic_xfb_counter_sub_gfx11_amd = 737
nir_last_intrinsic = 737
nir_num_intrinsics = 738
c__EA_nir_intrinsic_op = ctypes.c_uint32 # enum
nir_intrinsic_op = c__EA_nir_intrinsic_op
nir_intrinsic_op__enumvalues = c__EA_nir_intrinsic_op__enumvalues

# values for enumeration 'c__EA_nir_intrinsic_index_flag'
c__EA_nir_intrinsic_index_flag__enumvalues = {
    0: 'NIR_INTRINSIC_BASE',
    1: 'NIR_INTRINSIC_WRITE_MASK',
    2: 'NIR_INTRINSIC_STREAM_ID',
    3: 'NIR_INTRINSIC_UCP_ID',
    4: 'NIR_INTRINSIC_RANGE_BASE',
    5: 'NIR_INTRINSIC_RANGE',
    6: 'NIR_INTRINSIC_DESC_SET',
    7: 'NIR_INTRINSIC_BINDING',
    8: 'NIR_INTRINSIC_COMPONENT',
    9: 'NIR_INTRINSIC_COLUMN',
    10: 'NIR_INTRINSIC_INTERP_MODE',
    11: 'NIR_INTRINSIC_REDUCTION_OP',
    12: 'NIR_INTRINSIC_CLUSTER_SIZE',
    13: 'NIR_INTRINSIC_PARAM_IDX',
    14: 'NIR_INTRINSIC_IMAGE_DIM',
    15: 'NIR_INTRINSIC_IMAGE_ARRAY',
    16: 'NIR_INTRINSIC_FORMAT',
    17: 'NIR_INTRINSIC_ACCESS',
    18: 'NIR_INTRINSIC_CALL_IDX',
    19: 'NIR_INTRINSIC_STACK_SIZE',
    20: 'NIR_INTRINSIC_ALIGN_MUL',
    21: 'NIR_INTRINSIC_ALIGN_OFFSET',
    22: 'NIR_INTRINSIC_DESC_TYPE',
    23: 'NIR_INTRINSIC_SRC_TYPE',
    24: 'NIR_INTRINSIC_DEST_TYPE',
    25: 'NIR_INTRINSIC_SRC_BASE_TYPE',
    26: 'NIR_INTRINSIC_SRC_BASE_TYPE2',
    27: 'NIR_INTRINSIC_DEST_BASE_TYPE',
    28: 'NIR_INTRINSIC_SWIZZLE_MASK',
    29: 'NIR_INTRINSIC_FETCH_INACTIVE',
    30: 'NIR_INTRINSIC_OFFSET0',
    31: 'NIR_INTRINSIC_OFFSET1',
    32: 'NIR_INTRINSIC_ST64',
    33: 'NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD',
    34: 'NIR_INTRINSIC_DST_ACCESS',
    35: 'NIR_INTRINSIC_SRC_ACCESS',
    36: 'NIR_INTRINSIC_DRIVER_LOCATION',
    37: 'NIR_INTRINSIC_MEMORY_SEMANTICS',
    38: 'NIR_INTRINSIC_MEMORY_MODES',
    39: 'NIR_INTRINSIC_MEMORY_SCOPE',
    40: 'NIR_INTRINSIC_EXECUTION_SCOPE',
    41: 'NIR_INTRINSIC_IO_SEMANTICS',
    42: 'NIR_INTRINSIC_IO_XFB',
    43: 'NIR_INTRINSIC_IO_XFB2',
    44: 'NIR_INTRINSIC_RAY_QUERY_VALUE',
    45: 'NIR_INTRINSIC_COMMITTED',
    46: 'NIR_INTRINSIC_ROUNDING_MODE',
    47: 'NIR_INTRINSIC_SATURATE',
    48: 'NIR_INTRINSIC_SYNCHRONOUS',
    49: 'NIR_INTRINSIC_VALUE_ID',
    50: 'NIR_INTRINSIC_SIGN_EXTEND',
    51: 'NIR_INTRINSIC_FLAGS',
    52: 'NIR_INTRINSIC_ATOMIC_OP',
    53: 'NIR_INTRINSIC_RESOURCE_BLOCK_INTEL',
    54: 'NIR_INTRINSIC_RESOURCE_ACCESS_INTEL',
    55: 'NIR_INTRINSIC_NUM_COMPONENTS',
    56: 'NIR_INTRINSIC_NUM_ARRAY_ELEMS',
    57: 'NIR_INTRINSIC_BIT_SIZE',
    58: 'NIR_INTRINSIC_DIVERGENT',
    59: 'NIR_INTRINSIC_LEGACY_FABS',
    60: 'NIR_INTRINSIC_LEGACY_FNEG',
    61: 'NIR_INTRINSIC_LEGACY_FSAT',
    62: 'NIR_INTRINSIC_CMAT_DESC',
    63: 'NIR_INTRINSIC_MATRIX_LAYOUT',
    64: 'NIR_INTRINSIC_CMAT_SIGNED_MASK',
    65: 'NIR_INTRINSIC_ALU_OP',
    66: 'NIR_INTRINSIC_NEG_LO_AMD',
    67: 'NIR_INTRINSIC_NEG_HI_AMD',
    68: 'NIR_INTRINSIC_SYSTOLIC_DEPTH',
    69: 'NIR_INTRINSIC_REPEAT_COUNT',
    70: 'NIR_INTRINSIC_DST_CMAT_DESC',
    71: 'NIR_INTRINSIC_SRC_CMAT_DESC',
    72: 'NIR_INTRINSIC_EXPLICIT_COORD',
    73: 'NIR_INTRINSIC_FMT_IDX',
    74: 'NIR_INTRINSIC_PREAMBLE_CLASS',
    75: 'NIR_INTRINSIC_NUM_INDEX_FLAGS',
}
NIR_INTRINSIC_BASE = 0
NIR_INTRINSIC_WRITE_MASK = 1
NIR_INTRINSIC_STREAM_ID = 2
NIR_INTRINSIC_UCP_ID = 3
NIR_INTRINSIC_RANGE_BASE = 4
NIR_INTRINSIC_RANGE = 5
NIR_INTRINSIC_DESC_SET = 6
NIR_INTRINSIC_BINDING = 7
NIR_INTRINSIC_COMPONENT = 8
NIR_INTRINSIC_COLUMN = 9
NIR_INTRINSIC_INTERP_MODE = 10
NIR_INTRINSIC_REDUCTION_OP = 11
NIR_INTRINSIC_CLUSTER_SIZE = 12
NIR_INTRINSIC_PARAM_IDX = 13
NIR_INTRINSIC_IMAGE_DIM = 14
NIR_INTRINSIC_IMAGE_ARRAY = 15
NIR_INTRINSIC_FORMAT = 16
NIR_INTRINSIC_ACCESS = 17
NIR_INTRINSIC_CALL_IDX = 18
NIR_INTRINSIC_STACK_SIZE = 19
NIR_INTRINSIC_ALIGN_MUL = 20
NIR_INTRINSIC_ALIGN_OFFSET = 21
NIR_INTRINSIC_DESC_TYPE = 22
NIR_INTRINSIC_SRC_TYPE = 23
NIR_INTRINSIC_DEST_TYPE = 24
NIR_INTRINSIC_SRC_BASE_TYPE = 25
NIR_INTRINSIC_SRC_BASE_TYPE2 = 26
NIR_INTRINSIC_DEST_BASE_TYPE = 27
NIR_INTRINSIC_SWIZZLE_MASK = 28
NIR_INTRINSIC_FETCH_INACTIVE = 29
NIR_INTRINSIC_OFFSET0 = 30
NIR_INTRINSIC_OFFSET1 = 31
NIR_INTRINSIC_ST64 = 32
NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD = 33
NIR_INTRINSIC_DST_ACCESS = 34
NIR_INTRINSIC_SRC_ACCESS = 35
NIR_INTRINSIC_DRIVER_LOCATION = 36
NIR_INTRINSIC_MEMORY_SEMANTICS = 37
NIR_INTRINSIC_MEMORY_MODES = 38
NIR_INTRINSIC_MEMORY_SCOPE = 39
NIR_INTRINSIC_EXECUTION_SCOPE = 40
NIR_INTRINSIC_IO_SEMANTICS = 41
NIR_INTRINSIC_IO_XFB = 42
NIR_INTRINSIC_IO_XFB2 = 43
NIR_INTRINSIC_RAY_QUERY_VALUE = 44
NIR_INTRINSIC_COMMITTED = 45
NIR_INTRINSIC_ROUNDING_MODE = 46
NIR_INTRINSIC_SATURATE = 47
NIR_INTRINSIC_SYNCHRONOUS = 48
NIR_INTRINSIC_VALUE_ID = 49
NIR_INTRINSIC_SIGN_EXTEND = 50
NIR_INTRINSIC_FLAGS = 51
NIR_INTRINSIC_ATOMIC_OP = 52
NIR_INTRINSIC_RESOURCE_BLOCK_INTEL = 53
NIR_INTRINSIC_RESOURCE_ACCESS_INTEL = 54
NIR_INTRINSIC_NUM_COMPONENTS = 55
NIR_INTRINSIC_NUM_ARRAY_ELEMS = 56
NIR_INTRINSIC_BIT_SIZE = 57
NIR_INTRINSIC_DIVERGENT = 58
NIR_INTRINSIC_LEGACY_FABS = 59
NIR_INTRINSIC_LEGACY_FNEG = 60
NIR_INTRINSIC_LEGACY_FSAT = 61
NIR_INTRINSIC_CMAT_DESC = 62
NIR_INTRINSIC_MATRIX_LAYOUT = 63
NIR_INTRINSIC_CMAT_SIGNED_MASK = 64
NIR_INTRINSIC_ALU_OP = 65
NIR_INTRINSIC_NEG_LO_AMD = 66
NIR_INTRINSIC_NEG_HI_AMD = 67
NIR_INTRINSIC_SYSTOLIC_DEPTH = 68
NIR_INTRINSIC_REPEAT_COUNT = 69
NIR_INTRINSIC_DST_CMAT_DESC = 70
NIR_INTRINSIC_SRC_CMAT_DESC = 71
NIR_INTRINSIC_EXPLICIT_COORD = 72
NIR_INTRINSIC_FMT_IDX = 73
NIR_INTRINSIC_PREAMBLE_CLASS = 74
NIR_INTRINSIC_NUM_INDEX_FLAGS = 75
c__EA_nir_intrinsic_index_flag = ctypes.c_uint32 # enum
nir_intrinsic_index_flag = c__EA_nir_intrinsic_index_flag
nir_intrinsic_index_flag__enumvalues = c__EA_nir_intrinsic_index_flag__enumvalues
nir_intrinsic_index_names = [] # Variable ctypes.POINTER(ctypes.c_char) * 75
class struct_nir_intrinsic_instr(Structure):
    pass

struct_nir_intrinsic_instr._pack_ = 1 # source:False
struct_nir_intrinsic_instr._fields_ = [
    ('instr', nir_instr),
    ('intrinsic', nir_intrinsic_op),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('def', nir_def),
    ('num_components', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('const_index', ctypes.c_int32 * 8),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('src', struct_nir_src * 0),
]

nir_intrinsic_instr = struct_nir_intrinsic_instr
try:
    nir_intrinsic_get_var = _libraries['FIXME_STUB'].nir_intrinsic_get_var
    nir_intrinsic_get_var.restype = ctypes.POINTER(struct_nir_variable)
    nir_intrinsic_get_var.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_memory_semantics'
c__EA_nir_memory_semantics__enumvalues = {
    1: 'NIR_MEMORY_ACQUIRE',
    2: 'NIR_MEMORY_RELEASE',
    3: 'NIR_MEMORY_ACQ_REL',
    4: 'NIR_MEMORY_MAKE_AVAILABLE',
    8: 'NIR_MEMORY_MAKE_VISIBLE',
}
NIR_MEMORY_ACQUIRE = 1
NIR_MEMORY_RELEASE = 2
NIR_MEMORY_ACQ_REL = 3
NIR_MEMORY_MAKE_AVAILABLE = 4
NIR_MEMORY_MAKE_VISIBLE = 8
c__EA_nir_memory_semantics = ctypes.c_uint32 # enum
nir_memory_semantics = c__EA_nir_memory_semantics
nir_memory_semantics__enumvalues = c__EA_nir_memory_semantics__enumvalues

# values for enumeration 'c__EA_nir_intrinsic_semantic_flag'
c__EA_nir_intrinsic_semantic_flag__enumvalues = {
    1: 'NIR_INTRINSIC_CAN_ELIMINATE',
    2: 'NIR_INTRINSIC_CAN_REORDER',
    4: 'NIR_INTRINSIC_SUBGROUP',
    8: 'NIR_INTRINSIC_QUADGROUP',
}
NIR_INTRINSIC_CAN_ELIMINATE = 1
NIR_INTRINSIC_CAN_REORDER = 2
NIR_INTRINSIC_SUBGROUP = 4
NIR_INTRINSIC_QUADGROUP = 8
c__EA_nir_intrinsic_semantic_flag = ctypes.c_uint32 # enum
nir_intrinsic_semantic_flag = c__EA_nir_intrinsic_semantic_flag
nir_intrinsic_semantic_flag__enumvalues = c__EA_nir_intrinsic_semantic_flag__enumvalues
class struct_nir_io_semantics(Structure):
    pass

struct_nir_io_semantics._pack_ = 1 # source:False
struct_nir_io_semantics._fields_ = [
    ('location', ctypes.c_uint32, 7),
    ('num_slots', ctypes.c_uint32, 6),
    ('dual_source_blend_index', ctypes.c_uint32, 1),
    ('fb_fetch_output', ctypes.c_uint32, 1),
    ('fb_fetch_output_coherent', ctypes.c_uint32, 1),
    ('gs_streams', ctypes.c_uint32, 8),
    ('medium_precision', ctypes.c_uint32, 1),
    ('per_view', ctypes.c_uint32, 1),
    ('high_16bits', ctypes.c_uint32, 1),
    ('high_dvec2', ctypes.c_uint32, 1),
    ('no_varying', ctypes.c_uint32, 1),
    ('no_sysval_output', ctypes.c_uint32, 1),
    ('interp_explicit_strict', ctypes.c_uint32, 1),
    ('_pad', ctypes.c_uint32, 1),
]

nir_io_semantics = struct_nir_io_semantics
class struct_nir_io_xfb(Structure):
    pass

class struct_nir_io_xfb_0(Structure):
    pass

struct_nir_io_xfb_0._pack_ = 1 # source:False
struct_nir_io_xfb_0._fields_ = [
    ('num_components', ctypes.c_ubyte, 4),
    ('buffer', ctypes.c_ubyte, 4),
    ('offset', ctypes.c_ubyte, 8),
]

struct_nir_io_xfb._pack_ = 1 # source:False
struct_nir_io_xfb._fields_ = [
    ('out', struct_nir_io_xfb_0 * 2),
]

nir_io_xfb = struct_nir_io_xfb
try:
    nir_instr_xfb_write_mask = _libraries['FIXME_STUB'].nir_instr_xfb_write_mask
    nir_instr_xfb_write_mask.restype = ctypes.c_uint32
    nir_instr_xfb_write_mask.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
class struct_nir_intrinsic_info(Structure):
    pass

struct_nir_intrinsic_info._pack_ = 1 # source:False
struct_nir_intrinsic_info._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('num_srcs', ctypes.c_ubyte),
    ('src_components', ctypes.c_byte * 11),
    ('has_dest', ctypes.c_bool),
    ('dest_components', ctypes.c_ubyte),
    ('dest_bit_sizes', ctypes.c_ubyte),
    ('bit_size_src', ctypes.c_byte),
    ('num_indices', ctypes.c_ubyte),
    ('indices', ctypes.c_ubyte * 8),
    ('index_map', ctypes.c_ubyte * 75),
    ('flags', nir_intrinsic_semantic_flag),
]

nir_intrinsic_info = struct_nir_intrinsic_info
nir_intrinsic_infos = struct_nir_intrinsic_info * 738 # Variable struct_nir_intrinsic_info * 738
try:
    nir_intrinsic_src_components = _libraries['FIXME_STUB'].nir_intrinsic_src_components
    nir_intrinsic_src_components.restype = ctypes.c_uint32
    nir_intrinsic_src_components.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_intrinsic_dest_components = _libraries['FIXME_STUB'].nir_intrinsic_dest_components
    nir_intrinsic_dest_components.restype = ctypes.c_uint32
    nir_intrinsic_dest_components.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_instr_src_type = _libraries['FIXME_STUB'].nir_intrinsic_instr_src_type
    nir_intrinsic_instr_src_type.restype = nir_alu_type
    nir_intrinsic_instr_src_type.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_intrinsic_instr_dest_type = _libraries['FIXME_STUB'].nir_intrinsic_instr_dest_type
    nir_intrinsic_instr_dest_type.restype = nir_alu_type
    nir_intrinsic_instr_dest_type.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_copy_const_indices = _libraries['FIXME_STUB'].nir_intrinsic_copy_const_indices
    nir_intrinsic_copy_const_indices.restype = None
    nir_intrinsic_copy_const_indices.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_set_align = _libraries['FIXME_STUB'].nir_intrinsic_set_align
    nir_intrinsic_set_align.restype = None
    nir_intrinsic_set_align.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
uint32_t = ctypes.c_uint32
try:
    nir_combined_align = _libraries['FIXME_STUB'].nir_combined_align
    nir_combined_align.restype = uint32_t
    nir_combined_align.argtypes = [uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_intrinsic_align = _libraries['FIXME_STUB'].nir_intrinsic_align
    nir_intrinsic_align.restype = ctypes.c_uint32
    nir_intrinsic_align.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_has_align = _libraries['FIXME_STUB'].nir_intrinsic_has_align
    nir_intrinsic_has_align.restype = ctypes.c_bool
    nir_intrinsic_has_align.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_image_intrinsic_coord_components = _libraries['FIXME_STUB'].nir_image_intrinsic_coord_components
    nir_image_intrinsic_coord_components.restype = ctypes.c_uint32
    nir_image_intrinsic_coord_components.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_rewrite_image_intrinsic = _libraries['FIXME_STUB'].nir_rewrite_image_intrinsic
    nir_rewrite_image_intrinsic.restype = None
    nir_rewrite_image_intrinsic.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_def), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_intrinsic_can_reorder = _libraries['FIXME_STUB'].nir_intrinsic_can_reorder
    nir_intrinsic_can_reorder.restype = ctypes.c_bool
    nir_intrinsic_can_reorder.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_instr_can_speculate = _libraries['FIXME_STUB'].nir_instr_can_speculate
    nir_instr_can_speculate.restype = ctypes.c_bool
    nir_instr_can_speculate.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_writes_external_memory = _libraries['FIXME_STUB'].nir_intrinsic_writes_external_memory
    nir_intrinsic_writes_external_memory.restype = ctypes.c_bool
    nir_intrinsic_writes_external_memory.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_intrinsic_has_semantic = _libraries['FIXME_STUB'].nir_intrinsic_has_semantic
    nir_intrinsic_has_semantic.restype = ctypes.c_bool
    nir_intrinsic_has_semantic.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), nir_intrinsic_semantic_flag]
except AttributeError:
    pass
try:
    nir_intrinsic_is_ray_query = _libraries['FIXME_STUB'].nir_intrinsic_is_ray_query
    nir_intrinsic_is_ray_query.restype = ctypes.c_bool
    nir_intrinsic_is_ray_query.argtypes = [nir_intrinsic_op]
except AttributeError:
    pass

# values for enumeration 'nir_tex_src_type'
nir_tex_src_type__enumvalues = {
    0: 'nir_tex_src_coord',
    1: 'nir_tex_src_projector',
    2: 'nir_tex_src_comparator',
    3: 'nir_tex_src_offset',
    4: 'nir_tex_src_bias',
    5: 'nir_tex_src_lod',
    6: 'nir_tex_src_min_lod',
    7: 'nir_tex_src_lod_bias_min_agx',
    8: 'nir_tex_src_ms_index',
    9: 'nir_tex_src_ms_mcs_intel',
    10: 'nir_tex_src_ddx',
    11: 'nir_tex_src_ddy',
    12: 'nir_tex_src_texture_deref',
    13: 'nir_tex_src_sampler_deref',
    14: 'nir_tex_src_texture_offset',
    15: 'nir_tex_src_sampler_offset',
    16: 'nir_tex_src_texture_handle',
    17: 'nir_tex_src_sampler_handle',
    18: 'nir_tex_src_sampler_deref_intrinsic',
    19: 'nir_tex_src_texture_deref_intrinsic',
    20: 'nir_tex_src_plane',
    21: 'nir_tex_src_backend1',
    22: 'nir_tex_src_backend2',
    23: 'nir_num_tex_src_types',
}
nir_tex_src_coord = 0
nir_tex_src_projector = 1
nir_tex_src_comparator = 2
nir_tex_src_offset = 3
nir_tex_src_bias = 4
nir_tex_src_lod = 5
nir_tex_src_min_lod = 6
nir_tex_src_lod_bias_min_agx = 7
nir_tex_src_ms_index = 8
nir_tex_src_ms_mcs_intel = 9
nir_tex_src_ddx = 10
nir_tex_src_ddy = 11
nir_tex_src_texture_deref = 12
nir_tex_src_sampler_deref = 13
nir_tex_src_texture_offset = 14
nir_tex_src_sampler_offset = 15
nir_tex_src_texture_handle = 16
nir_tex_src_sampler_handle = 17
nir_tex_src_sampler_deref_intrinsic = 18
nir_tex_src_texture_deref_intrinsic = 19
nir_tex_src_plane = 20
nir_tex_src_backend1 = 21
nir_tex_src_backend2 = 22
nir_num_tex_src_types = 23
nir_tex_src_type = ctypes.c_uint32 # enum
class struct_nir_tex_src(Structure):
    pass

struct_nir_tex_src._pack_ = 1 # source:False
struct_nir_tex_src._fields_ = [
    ('src', nir_src),
    ('src_type', nir_tex_src_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

nir_tex_src = struct_nir_tex_src

# values for enumeration 'nir_texop'
nir_texop__enumvalues = {
    0: 'nir_texop_tex',
    1: 'nir_texop_txb',
    2: 'nir_texop_txl',
    3: 'nir_texop_txd',
    4: 'nir_texop_txf',
    5: 'nir_texop_txf_ms',
    6: 'nir_texop_txf_ms_fb',
    7: 'nir_texop_txf_ms_mcs_intel',
    8: 'nir_texop_txs',
    9: 'nir_texop_lod',
    10: 'nir_texop_tg4',
    11: 'nir_texop_query_levels',
    12: 'nir_texop_texture_samples',
    13: 'nir_texop_samples_identical',
    14: 'nir_texop_tex_prefetch',
    15: 'nir_texop_lod_bias',
    16: 'nir_texop_fragment_fetch_amd',
    17: 'nir_texop_fragment_mask_fetch_amd',
    18: 'nir_texop_descriptor_amd',
    19: 'nir_texop_sampler_descriptor_amd',
    20: 'nir_texop_image_min_lod_agx',
    21: 'nir_texop_has_custom_border_color_agx',
    22: 'nir_texop_custom_border_color_agx',
    23: 'nir_texop_hdr_dim_nv',
    24: 'nir_texop_tex_type_nv',
    25: 'nir_texop_sample_pos_nv',
}
nir_texop_tex = 0
nir_texop_txb = 1
nir_texop_txl = 2
nir_texop_txd = 3
nir_texop_txf = 4
nir_texop_txf_ms = 5
nir_texop_txf_ms_fb = 6
nir_texop_txf_ms_mcs_intel = 7
nir_texop_txs = 8
nir_texop_lod = 9
nir_texop_tg4 = 10
nir_texop_query_levels = 11
nir_texop_texture_samples = 12
nir_texop_samples_identical = 13
nir_texop_tex_prefetch = 14
nir_texop_lod_bias = 15
nir_texop_fragment_fetch_amd = 16
nir_texop_fragment_mask_fetch_amd = 17
nir_texop_descriptor_amd = 18
nir_texop_sampler_descriptor_amd = 19
nir_texop_image_min_lod_agx = 20
nir_texop_has_custom_border_color_agx = 21
nir_texop_custom_border_color_agx = 22
nir_texop_hdr_dim_nv = 23
nir_texop_tex_type_nv = 24
nir_texop_sample_pos_nv = 25
nir_texop = ctypes.c_uint32 # enum
class struct_nir_tex_instr(Structure):
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
struct_nir_tex_instr._pack_ = 1 # source:False
struct_nir_tex_instr._fields_ = [
    ('instr', nir_instr),
    ('sampler_dim', glsl_sampler_dim),
    ('dest_type', nir_alu_type),
    ('op', nir_texop),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('def', nir_def),
    ('src', ctypes.POINTER(struct_nir_tex_src)),
    ('num_srcs', ctypes.c_uint32),
    ('coord_components', ctypes.c_uint32),
    ('is_array', ctypes.c_bool),
    ('is_shadow', ctypes.c_bool),
    ('is_new_style_shadow', ctypes.c_bool),
    ('is_sparse', ctypes.c_bool),
    ('component', ctypes.c_uint32, 2),
    ('array_is_lowered_cube', ctypes.c_uint32, 1),
    ('is_gather_implicit_lod', ctypes.c_uint32, 1),
    ('skip_helpers', ctypes.c_uint32, 1),
    ('PADDING_1', ctypes.c_uint8, 3),
    ('tg4_offsets', ctypes.c_byte * 2 * 4),
    ('texture_non_uniform', ctypes.c_bool),
    ('sampler_non_uniform', ctypes.c_bool),
    ('offset_non_uniform', ctypes.c_bool),
    ('can_speculate', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte * 3),
    ('texture_index', ctypes.c_uint32),
    ('sampler_index', ctypes.c_uint32),
    ('backend_flags', ctypes.c_uint32),
]

nir_tex_instr = struct_nir_tex_instr
try:
    nir_tex_instr_need_sampler = _libraries['FIXME_STUB'].nir_tex_instr_need_sampler
    nir_tex_instr_need_sampler.restype = ctypes.c_bool
    nir_tex_instr_need_sampler.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_result_size = _libraries['FIXME_STUB'].nir_tex_instr_result_size
    nir_tex_instr_result_size.restype = ctypes.c_uint32
    nir_tex_instr_result_size.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_dest_size = _libraries['FIXME_STUB'].nir_tex_instr_dest_size
    nir_tex_instr_dest_size.restype = ctypes.c_uint32
    nir_tex_instr_dest_size.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_is_query = _libraries['FIXME_STUB'].nir_tex_instr_is_query
    nir_tex_instr_is_query.restype = ctypes.c_bool
    nir_tex_instr_is_query.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_has_implicit_derivative = _libraries['FIXME_STUB'].nir_tex_instr_has_implicit_derivative
    nir_tex_instr_has_implicit_derivative.restype = ctypes.c_bool
    nir_tex_instr_has_implicit_derivative.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
try:
    nir_tex_instr_src_type = _libraries['FIXME_STUB'].nir_tex_instr_src_type
    nir_tex_instr_src_type.restype = nir_alu_type
    nir_tex_instr_src_type.argtypes = [ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_tex_instr_src_size = _libraries['FIXME_STUB'].nir_tex_instr_src_size
    nir_tex_instr_src_size.restype = ctypes.c_uint32
    nir_tex_instr_src_size.argtypes = [ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_tex_instr_src_index = _libraries['FIXME_STUB'].nir_tex_instr_src_index
    nir_tex_instr_src_index.restype = ctypes.c_int32
    nir_tex_instr_src_index.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_tex_instr_add_src = _libraries['FIXME_STUB'].nir_tex_instr_add_src
    nir_tex_instr_add_src.restype = None
    nir_tex_instr_add_src.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_tex_instr_remove_src = _libraries['FIXME_STUB'].nir_tex_instr_remove_src
    nir_tex_instr_remove_src.restype = None
    nir_tex_instr_remove_src.argtypes = [ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_get_tex_src = _libraries['FIXME_STUB'].nir_get_tex_src
    nir_get_tex_src.restype = ctypes.POINTER(struct_nir_def)
    nir_get_tex_src.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_get_tex_deref = _libraries['FIXME_STUB'].nir_get_tex_deref
    nir_get_tex_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_get_tex_deref.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_steal_tex_src = _libraries['FIXME_STUB'].nir_steal_tex_src
    nir_steal_tex_src.restype = ctypes.POINTER(struct_nir_def)
    nir_steal_tex_src.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_steal_tex_deref = _libraries['FIXME_STUB'].nir_steal_tex_deref
    nir_steal_tex_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_steal_tex_deref.argtypes = [ctypes.POINTER(struct_nir_tex_instr), nir_tex_src_type]
except AttributeError:
    pass
try:
    nir_tex_instr_has_explicit_tg4_offsets = _libraries['FIXME_STUB'].nir_tex_instr_has_explicit_tg4_offsets
    nir_tex_instr_has_explicit_tg4_offsets.restype = ctypes.c_bool
    nir_tex_instr_has_explicit_tg4_offsets.argtypes = [ctypes.POINTER(struct_nir_tex_instr)]
except AttributeError:
    pass
class struct_nir_load_const_instr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instr', nir_instr),
    ('def', nir_def),
    ('value', union_c__UA_nir_const_value * 0),
     ]

nir_load_const_instr = struct_nir_load_const_instr

# values for enumeration 'c__EA_nir_jump_type'
c__EA_nir_jump_type__enumvalues = {
    0: 'nir_jump_return',
    1: 'nir_jump_halt',
    2: 'nir_jump_break',
    3: 'nir_jump_continue',
    4: 'nir_jump_goto',
    5: 'nir_jump_goto_if',
}
nir_jump_return = 0
nir_jump_halt = 1
nir_jump_break = 2
nir_jump_continue = 3
nir_jump_goto = 4
nir_jump_goto_if = 5
c__EA_nir_jump_type = ctypes.c_uint32 # enum
nir_jump_type = c__EA_nir_jump_type
nir_jump_type__enumvalues = c__EA_nir_jump_type__enumvalues
class struct_nir_jump_instr(Structure):
    pass

struct_nir_jump_instr._pack_ = 1 # source:False
struct_nir_jump_instr._fields_ = [
    ('instr', nir_instr),
    ('type', nir_jump_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('condition', nir_src),
    ('target', ctypes.POINTER(struct_nir_block)),
    ('else_target', ctypes.POINTER(struct_nir_block)),
]

nir_jump_instr = struct_nir_jump_instr
class struct_nir_undef_instr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instr', nir_instr),
    ('def', nir_def),
     ]

nir_undef_instr = struct_nir_undef_instr
class struct_nir_phi_src(Structure):
    pass

struct_nir_phi_src._pack_ = 1 # source:False
struct_nir_phi_src._fields_ = [
    ('node', struct_exec_node),
    ('pred', ctypes.POINTER(struct_nir_block)),
    ('src', nir_src),
]

nir_phi_src = struct_nir_phi_src
class struct_nir_phi_instr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instr', nir_instr),
    ('srcs', struct_exec_list),
    ('def', nir_def),
     ]

nir_phi_instr = struct_nir_phi_instr
try:
    nir_phi_get_src_from_block = _libraries['FIXME_STUB'].nir_phi_get_src_from_block
    nir_phi_get_src_from_block.restype = ctypes.POINTER(struct_nir_phi_src)
    nir_phi_get_src_from_block.argtypes = [ctypes.POINTER(struct_nir_phi_instr), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
class struct_nir_parallel_copy_entry(Structure):
    pass

class union_nir_parallel_copy_entry_dest(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('def', nir_def),
    ('reg', nir_src),
     ]

struct_nir_parallel_copy_entry._pack_ = 1 # source:False
struct_nir_parallel_copy_entry._fields_ = [
    ('node', struct_exec_node),
    ('src_is_reg', ctypes.c_bool),
    ('dest_is_reg', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('src', nir_src),
    ('dest', union_nir_parallel_copy_entry_dest),
]

nir_parallel_copy_entry = struct_nir_parallel_copy_entry
class struct_nir_parallel_copy_instr(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instr', nir_instr),
    ('entries', struct_exec_list),
     ]

nir_parallel_copy_instr = struct_nir_parallel_copy_instr
class struct_nir_instr_debug_info(Structure):
    pass

struct_nir_instr_debug_info._pack_ = 1 # source:False
struct_nir_instr_debug_info._fields_ = [
    ('filename', ctypes.POINTER(ctypes.c_char)),
    ('line', ctypes.c_uint32),
    ('column', ctypes.c_uint32),
    ('spirv_offset', ctypes.c_uint32),
    ('nir_line', ctypes.c_uint32),
    ('variable_name', ctypes.POINTER(ctypes.c_char)),
    ('instr', nir_instr),
]

nir_instr_debug_info = struct_nir_instr_debug_info
try:
    nir_instr_as_alu = _libraries['FIXME_STUB'].nir_instr_as_alu
    nir_instr_as_alu.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_instr_as_alu.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_deref = _libraries['FIXME_STUB'].nir_instr_as_deref
    nir_instr_as_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_instr_as_deref.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_call = _libraries['FIXME_STUB'].nir_instr_as_call
    nir_instr_as_call.restype = ctypes.POINTER(struct_nir_call_instr)
    nir_instr_as_call.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_jump = _libraries['FIXME_STUB'].nir_instr_as_jump
    nir_instr_as_jump.restype = ctypes.POINTER(struct_nir_jump_instr)
    nir_instr_as_jump.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_tex = _libraries['FIXME_STUB'].nir_instr_as_tex
    nir_instr_as_tex.restype = ctypes.POINTER(struct_nir_tex_instr)
    nir_instr_as_tex.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_intrinsic = _libraries['FIXME_STUB'].nir_instr_as_intrinsic
    nir_instr_as_intrinsic.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_instr_as_intrinsic.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_load_const = _libraries['FIXME_STUB'].nir_instr_as_load_const
    nir_instr_as_load_const.restype = ctypes.POINTER(struct_nir_load_const_instr)
    nir_instr_as_load_const.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_undef = _libraries['FIXME_STUB'].nir_instr_as_undef
    nir_instr_as_undef.restype = ctypes.POINTER(struct_nir_undef_instr)
    nir_instr_as_undef.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_phi = _libraries['FIXME_STUB'].nir_instr_as_phi
    nir_instr_as_phi.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_instr_as_phi.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_as_parallel_copy = _libraries['FIXME_STUB'].nir_instr_as_parallel_copy
    nir_instr_as_parallel_copy.restype = ctypes.POINTER(struct_nir_parallel_copy_instr)
    nir_instr_as_parallel_copy.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_src_comp_as_int = _libraries['FIXME_STUB'].nir_src_comp_as_int
    nir_src_comp_as_int.restype = int64_t
    nir_src_comp_as_int.argtypes = [nir_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_src_as_int = _libraries['FIXME_STUB'].nir_src_as_int
    nir_src_as_int.restype = int64_t
    nir_src_as_int.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_comp_as_uint = _libraries['FIXME_STUB'].nir_src_comp_as_uint
    nir_src_comp_as_uint.restype = uint64_t
    nir_src_comp_as_uint.argtypes = [nir_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_src_as_uint = _libraries['FIXME_STUB'].nir_src_as_uint
    nir_src_as_uint.restype = uint64_t
    nir_src_as_uint.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_comp_as_bool = _libraries['FIXME_STUB'].nir_src_comp_as_bool
    nir_src_comp_as_bool.restype = ctypes.c_bool
    nir_src_comp_as_bool.argtypes = [nir_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_src_as_bool = _libraries['FIXME_STUB'].nir_src_as_bool
    nir_src_as_bool.restype = ctypes.c_bool
    nir_src_as_bool.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_comp_as_float = _libraries['FIXME_STUB'].nir_src_comp_as_float
    nir_src_comp_as_float.restype = ctypes.c_double
    nir_src_comp_as_float.argtypes = [nir_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_src_as_float = _libraries['FIXME_STUB'].nir_src_as_float
    nir_src_as_float.restype = ctypes.c_double
    nir_src_as_float.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_def_as_alu = _libraries['FIXME_STUB'].nir_def_as_alu
    nir_def_as_alu.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_def_as_alu.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_as_intrinsic = _libraries['FIXME_STUB'].nir_def_as_intrinsic
    nir_def_as_intrinsic.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_def_as_intrinsic.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_as_tex = _libraries['FIXME_STUB'].nir_def_as_tex
    nir_def_as_tex.restype = ctypes.POINTER(struct_nir_tex_instr)
    nir_def_as_tex.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_as_phi = _libraries['FIXME_STUB'].nir_def_as_phi
    nir_def_as_phi.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_def_as_phi.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_as_deref = _libraries['FIXME_STUB'].nir_def_as_deref
    nir_def_as_deref.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_def_as_deref.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_as_load_const = _libraries['FIXME_STUB'].nir_def_as_load_const
    nir_def_as_load_const.restype = ctypes.POINTER(struct_nir_load_const_instr)
    nir_def_as_load_const.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
class struct_nir_scalar(Structure):
    pass

struct_nir_scalar._pack_ = 1 # source:False
struct_nir_scalar._fields_ = [
    ('def', ctypes.POINTER(struct_nir_def)),
    ('comp', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

nir_scalar = struct_nir_scalar
try:
    nir_scalar_is_const = _libraries['FIXME_STUB'].nir_scalar_is_const
    nir_scalar_is_const.restype = ctypes.c_bool
    nir_scalar_is_const.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_is_undef = _libraries['FIXME_STUB'].nir_scalar_is_undef
    nir_scalar_is_undef.restype = ctypes.c_bool
    nir_scalar_is_undef.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_const_value = _libraries['FIXME_STUB'].nir_scalar_as_const_value
    nir_scalar_as_const_value.restype = nir_const_value
    nir_scalar_as_const_value.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_int = _libraries['FIXME_STUB'].nir_scalar_as_int
    nir_scalar_as_int.restype = int64_t
    nir_scalar_as_int.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_uint = _libraries['FIXME_STUB'].nir_scalar_as_uint
    nir_scalar_as_uint.restype = uint64_t
    nir_scalar_as_uint.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_bool = _libraries['FIXME_STUB'].nir_scalar_as_bool
    nir_scalar_as_bool.restype = ctypes.c_bool
    nir_scalar_as_bool.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_as_float = _libraries['FIXME_STUB'].nir_scalar_as_float
    nir_scalar_as_float.restype = ctypes.c_double
    nir_scalar_as_float.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_is_alu = _libraries['FIXME_STUB'].nir_scalar_is_alu
    nir_scalar_is_alu.restype = ctypes.c_bool
    nir_scalar_is_alu.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_alu_op = _libraries['FIXME_STUB'].nir_scalar_alu_op
    nir_scalar_alu_op.restype = nir_op
    nir_scalar_alu_op.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_is_intrinsic = _libraries['FIXME_STUB'].nir_scalar_is_intrinsic
    nir_scalar_is_intrinsic.restype = ctypes.c_bool
    nir_scalar_is_intrinsic.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_intrinsic_op = _libraries['FIXME_STUB'].nir_scalar_intrinsic_op
    nir_scalar_intrinsic_op.restype = nir_intrinsic_op
    nir_scalar_intrinsic_op.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_scalar_chase_alu_src = _libraries['FIXME_STUB'].nir_scalar_chase_alu_src
    nir_scalar_chase_alu_src.restype = nir_scalar
    nir_scalar_chase_alu_src.argtypes = [nir_scalar, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_scalar_chase_movs = _libraries['FIXME_STUB'].nir_scalar_chase_movs
    nir_scalar_chase_movs.restype = nir_scalar
    nir_scalar_chase_movs.argtypes = [nir_scalar]
except AttributeError:
    pass
try:
    nir_get_scalar = _libraries['FIXME_STUB'].nir_get_scalar
    nir_get_scalar.restype = nir_scalar
    nir_get_scalar.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_scalar_resolved = _libraries['FIXME_STUB'].nir_scalar_resolved
    nir_scalar_resolved.restype = nir_scalar
    nir_scalar_resolved.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_scalar_equal = _libraries['FIXME_STUB'].nir_scalar_equal
    nir_scalar_equal.restype = ctypes.c_bool
    nir_scalar_equal.argtypes = [nir_scalar, nir_scalar]
except AttributeError:
    pass
try:
    nir_alu_src_as_uint = _libraries['FIXME_STUB'].nir_alu_src_as_uint
    nir_alu_src_as_uint.restype = uint64_t
    nir_alu_src_as_uint.argtypes = [nir_alu_src]
except AttributeError:
    pass
class struct_nir_binding(Structure):
    pass

struct_nir_binding._pack_ = 1 # source:False
struct_nir_binding._fields_ = [
    ('success', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('var', ctypes.POINTER(struct_nir_variable)),
    ('desc_set', ctypes.c_uint32),
    ('binding', ctypes.c_uint32),
    ('num_indices', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('indices', struct_nir_src * 4),
    ('read_first_invocation', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte * 7),
]

nir_binding = struct_nir_binding
try:
    nir_chase_binding = _libraries['FIXME_STUB'].nir_chase_binding
    nir_chase_binding.restype = nir_binding
    nir_chase_binding.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_get_binding_variable = _libraries['FIXME_STUB'].nir_get_binding_variable
    nir_get_binding_variable.restype = ctypes.POINTER(struct_nir_variable)
    nir_get_binding_variable.argtypes = [ctypes.POINTER(struct_nir_shader), nir_binding]
except AttributeError:
    pass
nir_cf_node_type = c__EA_nir_cf_node_type
nir_cf_node_type__enumvalues = c__EA_nir_cf_node_type__enumvalues
nir_cf_node = struct_nir_cf_node
nir_block = struct_nir_block
try:
    nir_block_is_reachable = _libraries['FIXME_STUB'].nir_block_is_reachable
    nir_block_is_reachable.restype = ctypes.c_bool
    nir_block_is_reachable.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_first_instr = _libraries['FIXME_STUB'].nir_block_first_instr
    nir_block_first_instr.restype = ctypes.POINTER(struct_nir_instr)
    nir_block_first_instr.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_last_instr = _libraries['FIXME_STUB'].nir_block_last_instr
    nir_block_last_instr.restype = ctypes.POINTER(struct_nir_instr)
    nir_block_last_instr.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_ends_in_jump = _libraries['FIXME_STUB'].nir_block_ends_in_jump
    nir_block_ends_in_jump.restype = ctypes.c_bool
    nir_block_ends_in_jump.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_ends_in_return_or_halt = _libraries['FIXME_STUB'].nir_block_ends_in_return_or_halt
    nir_block_ends_in_return_or_halt.restype = ctypes.c_bool
    nir_block_ends_in_return_or_halt.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_ends_in_break = _libraries['FIXME_STUB'].nir_block_ends_in_break
    nir_block_ends_in_break.restype = ctypes.c_bool
    nir_block_ends_in_break.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_contains_work = _libraries['FIXME_STUB'].nir_block_contains_work
    nir_block_contains_work.restype = ctypes.c_bool
    nir_block_contains_work.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_first_phi_in_block = _libraries['FIXME_STUB'].nir_first_phi_in_block
    nir_first_phi_in_block.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_first_phi_in_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_next_phi = _libraries['FIXME_STUB'].nir_next_phi
    nir_next_phi.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_next_phi.argtypes = [ctypes.POINTER(struct_nir_phi_instr)]
except AttributeError:
    pass
try:
    nir_block_last_phi_instr = _libraries['FIXME_STUB'].nir_block_last_phi_instr
    nir_block_last_phi_instr.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_block_last_phi_instr.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
nir_selection_control = c__EA_nir_selection_control
nir_selection_control__enumvalues = c__EA_nir_selection_control__enumvalues
nir_if = struct_nir_if
class struct_nir_loop_terminator(Structure):
    pass

struct_nir_loop_terminator._pack_ = 1 # source:False
struct_nir_loop_terminator._fields_ = [
    ('nif', ctypes.POINTER(struct_nir_if)),
    ('conditional_instr', ctypes.POINTER(struct_nir_instr)),
    ('break_block', ctypes.POINTER(struct_nir_block)),
    ('continue_from_block', ctypes.POINTER(struct_nir_block)),
    ('continue_from_then', ctypes.c_bool),
    ('induction_rhs', ctypes.c_bool),
    ('exact_trip_count_unknown', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 5),
    ('loop_terminator_link', struct_list_head),
]

nir_loop_terminator = struct_nir_loop_terminator
class struct_nir_loop_induction_variable(Structure):
    pass

struct_nir_loop_induction_variable._pack_ = 1 # source:False
struct_nir_loop_induction_variable._fields_ = [
    ('basis', ctypes.POINTER(struct_nir_def)),
    ('def', ctypes.POINTER(struct_nir_def)),
    ('init_src', ctypes.POINTER(struct_nir_src)),
    ('update_src', ctypes.POINTER(struct_nir_alu_src)),
]

nir_loop_induction_variable = struct_nir_loop_induction_variable
class struct_nir_loop_info(Structure):
    pass

class struct_hash_table(Structure):
    pass

struct_nir_loop_info._pack_ = 1 # source:False
struct_nir_loop_info._fields_ = [
    ('instr_cost', ctypes.c_uint32),
    ('has_soft_fp64', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('guessed_trip_count', ctypes.c_uint32),
    ('max_trip_count', ctypes.c_uint32),
    ('exact_trip_count_known', ctypes.c_bool),
    ('force_unroll', ctypes.c_bool),
    ('complex_loop', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 5),
    ('limiting_terminator', ctypes.POINTER(struct_nir_loop_terminator)),
    ('loop_terminator_list', struct_list_head),
    ('induction_vars', ctypes.POINTER(struct_hash_table)),
]

class struct_hash_entry(Structure):
    pass

struct_hash_table._pack_ = 1 # source:False
struct_hash_table._fields_ = [
    ('table', ctypes.POINTER(struct_hash_entry)),
    ('key_hash_function', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(None))),
    ('key_equals_function', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(None), ctypes.POINTER(None))),
    ('deleted_key', ctypes.POINTER(None)),
    ('size', ctypes.c_uint32),
    ('rehash', ctypes.c_uint32),
    ('size_magic', ctypes.c_uint64),
    ('rehash_magic', ctypes.c_uint64),
    ('max_entries', ctypes.c_uint32),
    ('size_index', ctypes.c_uint32),
    ('entries', ctypes.c_uint32),
    ('deleted_entries', ctypes.c_uint32),
]

struct_hash_entry._pack_ = 1 # source:False
struct_hash_entry._fields_ = [
    ('hash', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('key', ctypes.POINTER(None)),
    ('data', ctypes.POINTER(None)),
]

nir_loop_info = struct_nir_loop_info

# values for enumeration 'c__EA_nir_loop_control'
c__EA_nir_loop_control__enumvalues = {
    0: 'nir_loop_control_none',
    1: 'nir_loop_control_unroll',
    2: 'nir_loop_control_dont_unroll',
}
nir_loop_control_none = 0
nir_loop_control_unroll = 1
nir_loop_control_dont_unroll = 2
c__EA_nir_loop_control = ctypes.c_uint32 # enum
nir_loop_control = c__EA_nir_loop_control
nir_loop_control__enumvalues = c__EA_nir_loop_control__enumvalues
class struct_nir_loop(Structure):
    pass

struct_nir_loop._pack_ = 1 # source:False
struct_nir_loop._fields_ = [
    ('cf_node', nir_cf_node),
    ('body', struct_exec_list),
    ('continue_list', struct_exec_list),
    ('info', ctypes.POINTER(struct_nir_loop_info)),
    ('control', nir_loop_control),
    ('partially_unrolled', ctypes.c_bool),
    ('divergent_continue', ctypes.c_bool),
    ('divergent_break', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
]

nir_loop = struct_nir_loop
try:
    nir_loop_is_divergent = _libraries['FIXME_STUB'].nir_loop_is_divergent
    nir_loop_is_divergent.restype = ctypes.c_bool
    nir_loop_is_divergent.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
nir_metadata = c__EA_nir_metadata
nir_metadata__enumvalues = c__EA_nir_metadata__enumvalues
nir_function_impl = struct_nir_function_impl
try:
    nir_start_block = _libraries['FIXME_STUB'].nir_start_block
    nir_start_block.restype = ctypes.POINTER(struct_nir_block)
    nir_start_block.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_impl_last_block = _libraries['FIXME_STUB'].nir_impl_last_block
    nir_impl_last_block.restype = ctypes.POINTER(struct_nir_block)
    nir_impl_last_block.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_cf_node_next = _libraries['FIXME_STUB'].nir_cf_node_next
    nir_cf_node_next.restype = ctypes.POINTER(struct_nir_cf_node)
    nir_cf_node_next.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_prev = _libraries['FIXME_STUB'].nir_cf_node_prev
    nir_cf_node_prev.restype = ctypes.POINTER(struct_nir_cf_node)
    nir_cf_node_prev.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_is_first = _libraries['FIXME_STUB'].nir_cf_node_is_first
    nir_cf_node_is_first.restype = ctypes.c_bool
    nir_cf_node_is_first.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_is_last = _libraries['FIXME_STUB'].nir_cf_node_is_last
    nir_cf_node_is_last.restype = ctypes.c_bool
    nir_cf_node_is_last.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_as_block = _libraries['FIXME_STUB'].nir_cf_node_as_block
    nir_cf_node_as_block.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_as_block.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_as_if = _libraries['FIXME_STUB'].nir_cf_node_as_if
    nir_cf_node_as_if.restype = ctypes.POINTER(struct_nir_if)
    nir_cf_node_as_if.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_as_loop = _libraries['FIXME_STUB'].nir_cf_node_as_loop
    nir_cf_node_as_loop.restype = ctypes.POINTER(struct_nir_loop)
    nir_cf_node_as_loop.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_as_function = _libraries['FIXME_STUB'].nir_cf_node_as_function
    nir_cf_node_as_function.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_cf_node_as_function.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_if_first_then_block = _libraries['FIXME_STUB'].nir_if_first_then_block
    nir_if_first_then_block.restype = ctypes.POINTER(struct_nir_block)
    nir_if_first_then_block.argtypes = [ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_if_last_then_block = _libraries['FIXME_STUB'].nir_if_last_then_block
    nir_if_last_then_block.restype = ctypes.POINTER(struct_nir_block)
    nir_if_last_then_block.argtypes = [ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_if_first_else_block = _libraries['FIXME_STUB'].nir_if_first_else_block
    nir_if_first_else_block.restype = ctypes.POINTER(struct_nir_block)
    nir_if_first_else_block.argtypes = [ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_if_last_else_block = _libraries['FIXME_STUB'].nir_if_last_else_block
    nir_if_last_else_block.restype = ctypes.POINTER(struct_nir_block)
    nir_if_last_else_block.argtypes = [ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_loop_first_block = _libraries['FIXME_STUB'].nir_loop_first_block
    nir_loop_first_block.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_first_block.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_last_block = _libraries['FIXME_STUB'].nir_loop_last_block
    nir_loop_last_block.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_last_block.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_has_continue_construct = _libraries['FIXME_STUB'].nir_loop_has_continue_construct
    nir_loop_has_continue_construct.restype = ctypes.c_bool
    nir_loop_has_continue_construct.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_first_continue_block = _libraries['FIXME_STUB'].nir_loop_first_continue_block
    nir_loop_first_continue_block.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_first_continue_block.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_last_continue_block = _libraries['FIXME_STUB'].nir_loop_last_continue_block
    nir_loop_last_continue_block.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_last_continue_block.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_loop_continue_target = _libraries['FIXME_STUB'].nir_loop_continue_target
    nir_loop_continue_target.restype = ctypes.POINTER(struct_nir_block)
    nir_loop_continue_target.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_cf_list_is_empty_block = _libraries['FIXME_STUB'].nir_cf_list_is_empty_block
    nir_cf_list_is_empty_block.restype = ctypes.c_bool
    nir_cf_list_is_empty_block.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
nir_parameter = struct_nir_parameter
nir_function = struct_nir_function
nir_intrin_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
nir_vectorize_cb = ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
nir_shader = struct_nir_shader
try:
    nir_foreach_function_with_impl_first = _libraries['FIXME_STUB'].nir_foreach_function_with_impl_first
    nir_foreach_function_with_impl_first.restype = ctypes.POINTER(struct_nir_function)
    nir_foreach_function_with_impl_first.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_foreach_function_with_impl_next = _libraries['FIXME_STUB'].nir_foreach_function_with_impl_next
    nir_foreach_function_with_impl_next.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_foreach_function_with_impl_next.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_nir_function))]
except AttributeError:
    pass
try:
    nir_shader_get_entrypoint = _libraries['FIXME_STUB'].nir_shader_get_entrypoint
    nir_shader_get_entrypoint.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_shader_get_entrypoint.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_get_function_for_name = _libraries['FIXME_STUB'].nir_shader_get_function_for_name
    nir_shader_get_function_for_name.restype = ctypes.POINTER(struct_nir_function)
    nir_shader_get_function_for_name.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_remove_non_entrypoints = _libraries['FIXME_STUB'].nir_remove_non_entrypoints
    nir_remove_non_entrypoints.restype = None
    nir_remove_non_entrypoints.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_non_exported = _libraries['FIXME_STUB'].nir_remove_non_exported
    nir_remove_non_exported.restype = None
    nir_remove_non_exported.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_entrypoints = _libraries['FIXME_STUB'].nir_remove_entrypoints
    nir_remove_entrypoints.restype = None
    nir_remove_entrypoints.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_fixup_is_exported = _libraries['FIXME_STUB'].nir_fixup_is_exported
    nir_fixup_is_exported.restype = None
    nir_fixup_is_exported.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_create = _libraries['FIXME_STUB'].nir_shader_create
    nir_shader_create.restype = ctypes.POINTER(struct_nir_shader)
    nir_shader_create.argtypes = [ctypes.POINTER(None), mesa_shader_stage, ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_shader_info)]
except AttributeError:
    pass
try:
    nir_shader_add_variable = _libraries['FIXME_STUB'].nir_shader_add_variable
    nir_shader_add_variable.restype = None
    nir_shader_add_variable.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_function_impl_add_variable = _libraries['FIXME_STUB'].nir_function_impl_add_variable
    nir_function_impl_add_variable.restype = None
    nir_function_impl_add_variable.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_create_zeroed = _libraries['FIXME_STUB'].nir_variable_create_zeroed
    nir_variable_create_zeroed.restype = ctypes.POINTER(struct_nir_variable)
    nir_variable_create_zeroed.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_variable_set_name = _libraries['FIXME_STUB'].nir_variable_set_name
    nir_variable_set_name.restype = None
    nir_variable_set_name.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_variable_set_namef = _libraries['FIXME_STUB'].nir_variable_set_namef
    nir_variable_set_namef.restype = None
    nir_variable_set_namef.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_variable_append_namef = _libraries['FIXME_STUB'].nir_variable_append_namef
    nir_variable_append_namef.restype = None
    nir_variable_append_namef.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_variable_steal_name = _libraries['FIXME_STUB'].nir_variable_steal_name
    nir_variable_steal_name.restype = None
    nir_variable_steal_name.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_create = _libraries['FIXME_STUB'].nir_variable_create
    nir_variable_create.restype = ctypes.POINTER(struct_nir_variable)
    nir_variable_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_local_variable_create = _libraries['FIXME_STUB'].nir_local_variable_create
    nir_local_variable_create.restype = ctypes.POINTER(struct_nir_variable)
    nir_local_variable_create.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_state_variable_create = _libraries['FIXME_STUB'].nir_state_variable_create
    nir_state_variable_create.restype = ctypes.POINTER(struct_nir_variable)
    nir_state_variable_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_char), ctypes.c_int16 * 4]
except AttributeError:
    pass
try:
    nir_get_variable_with_location = _libraries['FIXME_STUB'].nir_get_variable_with_location
    nir_get_variable_with_location.restype = ctypes.POINTER(struct_nir_variable)
    nir_get_variable_with_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_int32, ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_create_variable_with_location = _libraries['FIXME_STUB'].nir_create_variable_with_location
    nir_create_variable_with_location.restype = ctypes.POINTER(struct_nir_variable)
    nir_create_variable_with_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_int32, ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_find_variable_with_location = _libraries['FIXME_STUB'].nir_find_variable_with_location
    nir_find_variable_with_location.restype = ctypes.POINTER(struct_nir_variable)
    nir_find_variable_with_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_find_variable_with_driver_location = _libraries['FIXME_STUB'].nir_find_variable_with_driver_location
    nir_find_variable_with_driver_location.restype = ctypes.POINTER(struct_nir_variable)
    nir_find_variable_with_driver_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_find_state_variable = _libraries['FIXME_STUB'].nir_find_state_variable
    nir_find_state_variable.restype = ctypes.POINTER(struct_nir_variable)
    nir_find_state_variable.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_int16 * 4]
except AttributeError:
    pass
try:
    nir_find_sampler_variable_with_tex_index = _libraries['FIXME_STUB'].nir_find_sampler_variable_with_tex_index
    nir_find_sampler_variable_with_tex_index.restype = ctypes.POINTER(struct_nir_variable)
    nir_find_sampler_variable_with_tex_index.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_sort_variables_with_modes = _libraries['FIXME_STUB'].nir_sort_variables_with_modes
    nir_sort_variables_with_modes.restype = None
    nir_sort_variables_with_modes.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_variable)), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_function_create = _libraries['FIXME_STUB'].nir_function_create
    nir_function_create.restype = ctypes.POINTER(struct_nir_function)
    nir_function_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_function_set_impl = _libraries['FIXME_STUB'].nir_function_set_impl
    nir_function_set_impl.restype = None
    nir_function_set_impl.argtypes = [ctypes.POINTER(struct_nir_function), ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_function_impl_create = _libraries['FIXME_STUB'].nir_function_impl_create
    nir_function_impl_create.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_function_impl_create.argtypes = [ctypes.POINTER(struct_nir_function)]
except AttributeError:
    pass
try:
    nir_function_impl_create_bare = _libraries['FIXME_STUB'].nir_function_impl_create_bare
    nir_function_impl_create_bare.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_function_impl_create_bare.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_block_create = _libraries['FIXME_STUB'].nir_block_create
    nir_block_create.restype = ctypes.POINTER(struct_nir_block)
    nir_block_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_if_create = _libraries['FIXME_STUB'].nir_if_create
    nir_if_create.restype = ctypes.POINTER(struct_nir_if)
    nir_if_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_loop_create = _libraries['FIXME_STUB'].nir_loop_create
    nir_loop_create.restype = ctypes.POINTER(struct_nir_loop)
    nir_loop_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_cf_node_get_function = _libraries['FIXME_STUB'].nir_cf_node_get_function
    nir_cf_node_get_function.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_cf_node_get_function.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_metadata_require = _libraries['FIXME_STUB'].nir_metadata_require
    nir_metadata_require.restype = None
    nir_metadata_require.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_metadata]
except AttributeError:
    pass
try:
    nir_shader_preserve_all_metadata = _libraries['FIXME_STUB'].nir_shader_preserve_all_metadata
    nir_shader_preserve_all_metadata.restype = None
    nir_shader_preserve_all_metadata.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_metadata_invalidate = _libraries['FIXME_STUB'].nir_metadata_invalidate
    nir_metadata_invalidate.restype = None
    nir_metadata_invalidate.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_progress = _libraries['FIXME_STUB'].nir_progress
    nir_progress.restype = ctypes.c_bool
    nir_progress.argtypes = [ctypes.c_bool, ctypes.POINTER(struct_nir_function_impl), nir_metadata]
except AttributeError:
    pass
try:
    nir_no_progress = _libraries['FIXME_STUB'].nir_no_progress
    nir_no_progress.restype = ctypes.c_bool
    nir_no_progress.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_alu_instr_create = _libraries['FIXME_STUB'].nir_alu_instr_create
    nir_alu_instr_create.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_alu_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_op]
except AttributeError:
    pass
try:
    nir_deref_instr_create = _libraries['FIXME_STUB'].nir_deref_instr_create
    nir_deref_instr_create.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_deref_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_deref_type]
except AttributeError:
    pass
try:
    nir_jump_instr_create = _libraries['FIXME_STUB'].nir_jump_instr_create
    nir_jump_instr_create.restype = ctypes.POINTER(struct_nir_jump_instr)
    nir_jump_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_jump_type]
except AttributeError:
    pass
try:
    nir_load_const_instr_create = _libraries['FIXME_STUB'].nir_load_const_instr_create
    nir_load_const_instr_create.restype = ctypes.POINTER(struct_nir_load_const_instr)
    nir_load_const_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_intrinsic_instr_create = _libraries['FIXME_STUB'].nir_intrinsic_instr_create
    nir_intrinsic_instr_create.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_intrinsic_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrinsic_op]
except AttributeError:
    pass
try:
    nir_call_instr_create = _libraries['FIXME_STUB'].nir_call_instr_create
    nir_call_instr_create.restype = ctypes.POINTER(struct_nir_call_instr)
    nir_call_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function)]
except AttributeError:
    pass
try:
    nir_tex_instr_create = _libraries['FIXME_STUB'].nir_tex_instr_create
    nir_tex_instr_create.restype = ctypes.POINTER(struct_nir_tex_instr)
    nir_tex_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_phi_instr_create = _libraries['FIXME_STUB'].nir_phi_instr_create
    nir_phi_instr_create.restype = ctypes.POINTER(struct_nir_phi_instr)
    nir_phi_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_phi_instr_add_src = _libraries['FIXME_STUB'].nir_phi_instr_add_src
    nir_phi_instr_add_src.restype = ctypes.POINTER(struct_nir_phi_src)
    nir_phi_instr_add_src.argtypes = [ctypes.POINTER(struct_nir_phi_instr), ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_parallel_copy_instr_create = _libraries['FIXME_STUB'].nir_parallel_copy_instr_create
    nir_parallel_copy_instr_create.restype = ctypes.POINTER(struct_nir_parallel_copy_instr)
    nir_parallel_copy_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_undef_instr_create = _libraries['FIXME_STUB'].nir_undef_instr_create
    nir_undef_instr_create.restype = ctypes.POINTER(struct_nir_undef_instr)
    nir_undef_instr_create.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alu_binop_identity = _libraries['FIXME_STUB'].nir_alu_binop_identity
    nir_alu_binop_identity.restype = nir_const_value
    nir_alu_binop_identity.argtypes = [nir_op, ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_cursor_option'
c__EA_nir_cursor_option__enumvalues = {
    0: 'nir_cursor_before_block',
    1: 'nir_cursor_after_block',
    2: 'nir_cursor_before_instr',
    3: 'nir_cursor_after_instr',
}
nir_cursor_before_block = 0
nir_cursor_after_block = 1
nir_cursor_before_instr = 2
nir_cursor_after_instr = 3
c__EA_nir_cursor_option = ctypes.c_uint32 # enum
nir_cursor_option = c__EA_nir_cursor_option
nir_cursor_option__enumvalues = c__EA_nir_cursor_option__enumvalues
class struct_nir_cursor(Structure):
    pass

class union_nir_cursor_0(Union):
    pass

union_nir_cursor_0._pack_ = 1 # source:False
union_nir_cursor_0._fields_ = [
    ('block', ctypes.POINTER(struct_nir_block)),
    ('instr', ctypes.POINTER(struct_nir_instr)),
]

struct_nir_cursor._pack_ = 1 # source:False
struct_nir_cursor._anonymous_ = ('_0',)
struct_nir_cursor._fields_ = [
    ('option', nir_cursor_option),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_0', union_nir_cursor_0),
]

nir_cursor = struct_nir_cursor
try:
    nir_cursor_current_block = _libraries['FIXME_STUB'].nir_cursor_current_block
    nir_cursor_current_block.restype = ctypes.POINTER(struct_nir_block)
    nir_cursor_current_block.argtypes = [nir_cursor]
except AttributeError:
    pass
try:
    nir_cursors_equal = _libraries['FIXME_STUB'].nir_cursors_equal
    nir_cursors_equal.restype = ctypes.c_bool
    nir_cursors_equal.argtypes = [nir_cursor, nir_cursor]
except AttributeError:
    pass
try:
    nir_before_block = _libraries['FIXME_STUB'].nir_before_block
    nir_before_block.restype = nir_cursor
    nir_before_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_after_block = _libraries['FIXME_STUB'].nir_after_block
    nir_after_block.restype = nir_cursor
    nir_after_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_before_instr = _libraries['FIXME_STUB'].nir_before_instr
    nir_before_instr.restype = nir_cursor
    nir_before_instr.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_after_instr = _libraries['FIXME_STUB'].nir_after_instr
    nir_after_instr.restype = nir_cursor
    nir_after_instr.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_before_block_after_phis = _libraries['FIXME_STUB'].nir_before_block_after_phis
    nir_before_block_after_phis.restype = nir_cursor
    nir_before_block_after_phis.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_after_block_before_jump = _libraries['FIXME_STUB'].nir_after_block_before_jump
    nir_after_block_before_jump.restype = nir_cursor
    nir_after_block_before_jump.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_before_src = _libraries['FIXME_STUB'].nir_before_src
    nir_before_src.restype = nir_cursor
    nir_before_src.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_before_cf_node = _libraries['FIXME_STUB'].nir_before_cf_node
    nir_before_cf_node.restype = nir_cursor
    nir_before_cf_node.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_after_cf_node = _libraries['FIXME_STUB'].nir_after_cf_node
    nir_after_cf_node.restype = nir_cursor
    nir_after_cf_node.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_after_phis = _libraries['FIXME_STUB'].nir_after_phis
    nir_after_phis.restype = nir_cursor
    nir_after_phis.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_after_instr_and_phis = _libraries['FIXME_STUB'].nir_after_instr_and_phis
    nir_after_instr_and_phis.restype = nir_cursor
    nir_after_instr_and_phis.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_after_cf_node_and_phis = _libraries['FIXME_STUB'].nir_after_cf_node_and_phis
    nir_after_cf_node_and_phis.restype = nir_cursor
    nir_after_cf_node_and_phis.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_before_cf_list = _libraries['FIXME_STUB'].nir_before_cf_list
    nir_before_cf_list.restype = nir_cursor
    nir_before_cf_list.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    nir_after_cf_list = _libraries['FIXME_STUB'].nir_after_cf_list
    nir_after_cf_list.restype = nir_cursor
    nir_after_cf_list.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    nir_before_impl = _libraries['FIXME_STUB'].nir_before_impl
    nir_before_impl.restype = nir_cursor
    nir_before_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_after_impl = _libraries['FIXME_STUB'].nir_after_impl
    nir_after_impl.restype = nir_cursor
    nir_after_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_instr_insert = _libraries['FIXME_STUB'].nir_instr_insert
    nir_instr_insert.restype = None
    nir_instr_insert.argtypes = [nir_cursor, ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_move = _libraries['FIXME_STUB'].nir_instr_move
    nir_instr_move.restype = ctypes.c_bool
    nir_instr_move.argtypes = [nir_cursor, ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_before = _libraries['FIXME_STUB'].nir_instr_insert_before
    nir_instr_insert_before.restype = None
    nir_instr_insert_before.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_after = _libraries['FIXME_STUB'].nir_instr_insert_after
    nir_instr_insert_after.restype = None
    nir_instr_insert_after.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_before_block = _libraries['FIXME_STUB'].nir_instr_insert_before_block
    nir_instr_insert_before_block.restype = None
    nir_instr_insert_before_block.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_after_block = _libraries['FIXME_STUB'].nir_instr_insert_after_block
    nir_instr_insert_after_block.restype = None
    nir_instr_insert_after_block.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_before_cf = _libraries['FIXME_STUB'].nir_instr_insert_before_cf
    nir_instr_insert_before_cf.restype = None
    nir_instr_insert_before_cf.argtypes = [ctypes.POINTER(struct_nir_cf_node), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_after_cf = _libraries['FIXME_STUB'].nir_instr_insert_after_cf
    nir_instr_insert_after_cf.restype = None
    nir_instr_insert_after_cf.argtypes = [ctypes.POINTER(struct_nir_cf_node), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_before_cf_list = _libraries['FIXME_STUB'].nir_instr_insert_before_cf_list
    nir_instr_insert_before_cf_list.restype = None
    nir_instr_insert_before_cf_list.argtypes = [ctypes.POINTER(struct_exec_list), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_insert_after_cf_list = _libraries['FIXME_STUB'].nir_instr_insert_after_cf_list
    nir_instr_insert_after_cf_list.restype = None
    nir_instr_insert_after_cf_list.argtypes = [ctypes.POINTER(struct_exec_list), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_remove_v = _libraries['FIXME_STUB'].nir_instr_remove_v
    nir_instr_remove_v.restype = None
    nir_instr_remove_v.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_free = _libraries['FIXME_STUB'].nir_instr_free
    nir_instr_free.restype = None
    nir_instr_free.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_free_list = _libraries['FIXME_STUB'].nir_instr_free_list
    nir_instr_free_list.restype = None
    nir_instr_free_list.argtypes = [ctypes.POINTER(struct_exec_list)]
except AttributeError:
    pass
try:
    nir_instr_remove = _libraries['FIXME_STUB'].nir_instr_remove
    nir_instr_remove.restype = nir_cursor
    nir_instr_remove.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_free_and_dce = _libraries['FIXME_STUB'].nir_instr_free_and_dce
    nir_instr_free_and_dce.restype = nir_cursor
    nir_instr_free_and_dce.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_def = _libraries['FIXME_STUB'].nir_instr_def
    nir_instr_def.restype = ctypes.POINTER(struct_nir_def)
    nir_instr_def.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_get_debug_info = _libraries['FIXME_STUB'].nir_instr_get_debug_info
    nir_instr_get_debug_info.restype = ctypes.POINTER(struct_nir_instr_debug_info)
    nir_instr_get_debug_info.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_get_gc_pointer = _libraries['FIXME_STUB'].nir_instr_get_gc_pointer
    nir_instr_get_gc_pointer.restype = ctypes.POINTER(None)
    nir_instr_get_gc_pointer.argtypes = [ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
nir_foreach_def_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_def), ctypes.POINTER(None))
nir_foreach_src_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_src), ctypes.POINTER(None))
try:
    nir_foreach_src = _libraries['FIXME_STUB'].nir_foreach_src
    nir_foreach_src.restype = ctypes.c_bool
    nir_foreach_src.argtypes = [ctypes.POINTER(struct_nir_instr), nir_foreach_src_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_foreach_phi_src_leaving_block = _libraries['FIXME_STUB'].nir_foreach_phi_src_leaving_block
    nir_foreach_phi_src_leaving_block.restype = ctypes.c_bool
    nir_foreach_phi_src_leaving_block.argtypes = [ctypes.POINTER(struct_nir_block), nir_foreach_src_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_src_as_const_value = _libraries['FIXME_STUB'].nir_src_as_const_value
    nir_src_as_const_value.restype = ctypes.POINTER(union_c__UA_nir_const_value)
    nir_src_as_const_value.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_as_alu_instr = _libraries['FIXME_STUB'].nir_src_as_alu_instr
    nir_src_as_alu_instr.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_src_as_alu_instr.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_as_intrinsic = _libraries['FIXME_STUB'].nir_src_as_intrinsic
    nir_src_as_intrinsic.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_src_as_intrinsic.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_as_string = _libraries['FIXME_STUB'].nir_src_as_string
    nir_src_as_string.restype = ctypes.POINTER(ctypes.c_char)
    nir_src_as_string.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_src_is_always_uniform = _libraries['FIXME_STUB'].nir_src_is_always_uniform
    nir_src_is_always_uniform.restype = ctypes.c_bool
    nir_src_is_always_uniform.argtypes = [nir_src]
except AttributeError:
    pass
try:
    nir_srcs_equal = _libraries['FIXME_STUB'].nir_srcs_equal
    nir_srcs_equal.restype = ctypes.c_bool
    nir_srcs_equal.argtypes = [nir_src, nir_src]
except AttributeError:
    pass
try:
    nir_instrs_equal = _libraries['FIXME_STUB'].nir_instrs_equal
    nir_instrs_equal.restype = ctypes.c_bool
    nir_instrs_equal.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_src_get_block = _libraries['FIXME_STUB'].nir_src_get_block
    nir_src_get_block.restype = ctypes.POINTER(struct_nir_block)
    nir_src_get_block.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_src_rewrite = _libraries['FIXME_STUB'].nir_src_rewrite
    nir_src_rewrite.restype = None
    nir_src_rewrite.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_alu_src_rewrite_scalar = _libraries['FIXME_STUB'].nir_alu_src_rewrite_scalar
    nir_alu_src_rewrite_scalar.restype = None
    nir_alu_src_rewrite_scalar.argtypes = [ctypes.POINTER(struct_nir_alu_src), nir_scalar]
except AttributeError:
    pass
try:
    nir_instr_init_src = _libraries['FIXME_STUB'].nir_instr_init_src
    nir_instr_init_src.restype = None
    nir_instr_init_src.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_instr_clear_src = _libraries['FIXME_STUB'].nir_instr_clear_src
    nir_instr_clear_src.restype = None
    nir_instr_clear_src.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_instr_move_src = _libraries['FIXME_STUB'].nir_instr_move_src
    nir_instr_move_src.restype = None
    nir_instr_move_src.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_instr_is_before = _libraries['FIXME_STUB'].nir_instr_is_before
    nir_instr_is_before.restype = ctypes.c_bool
    nir_instr_is_before.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_def_init = _libraries['FIXME_STUB'].nir_def_init
    nir_def_init.restype = None
    nir_def_init.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_def_init_for_type = _libraries['FIXME_STUB'].nir_def_init_for_type
    nir_def_init_for_type.restype = None
    nir_def_init_for_type.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_def_rewrite_uses = _libraries['FIXME_STUB'].nir_def_rewrite_uses
    nir_def_rewrite_uses.restype = None
    nir_def_rewrite_uses.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_rewrite_uses_src = _libraries['FIXME_STUB'].nir_def_rewrite_uses_src
    nir_def_rewrite_uses_src.restype = None
    nir_def_rewrite_uses_src.argtypes = [ctypes.POINTER(struct_nir_def), nir_src]
except AttributeError:
    pass
try:
    nir_def_rewrite_uses_after_instr = _libraries['FIXME_STUB'].nir_def_rewrite_uses_after_instr
    nir_def_rewrite_uses_after_instr.restype = None
    nir_def_rewrite_uses_after_instr.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_def_rewrite_uses_after = _libraries['FIXME_STUB'].nir_def_rewrite_uses_after
    nir_def_rewrite_uses_after.restype = None
    nir_def_rewrite_uses_after.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_replace = _libraries['FIXME_STUB'].nir_def_replace
    nir_def_replace.restype = None
    nir_def_replace.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_src_components_read = _libraries['FIXME_STUB'].nir_src_components_read
    nir_src_components_read.restype = nir_component_mask_t
    nir_src_components_read.argtypes = [ctypes.POINTER(struct_nir_src)]
except AttributeError:
    pass
try:
    nir_def_components_read = _libraries['FIXME_STUB'].nir_def_components_read
    nir_def_components_read.restype = nir_component_mask_t
    nir_def_components_read.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_all_uses_are_fsat = _libraries['FIXME_STUB'].nir_def_all_uses_are_fsat
    nir_def_all_uses_are_fsat.restype = ctypes.c_bool
    nir_def_all_uses_are_fsat.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_all_uses_ignore_sign_bit = _libraries['FIXME_STUB'].nir_def_all_uses_ignore_sign_bit
    nir_def_all_uses_ignore_sign_bit.restype = ctypes.c_bool
    nir_def_all_uses_ignore_sign_bit.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_first_component_read = _libraries['FIXME_STUB'].nir_def_first_component_read
    nir_def_first_component_read.restype = ctypes.c_int32
    nir_def_first_component_read.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_last_component_read = _libraries['FIXME_STUB'].nir_def_last_component_read
    nir_def_last_component_read.restype = ctypes.c_int32
    nir_def_last_component_read.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_def_is_unused = _libraries['FIXME_STUB'].nir_def_is_unused
    nir_def_is_unused.restype = ctypes.c_bool
    nir_def_is_unused.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_sort_unstructured_blocks = _libraries['FIXME_STUB'].nir_sort_unstructured_blocks
    nir_sort_unstructured_blocks.restype = None
    nir_sort_unstructured_blocks.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_block_unstructured_next = _libraries['FIXME_STUB'].nir_block_unstructured_next
    nir_block_unstructured_next.restype = ctypes.POINTER(struct_nir_block)
    nir_block_unstructured_next.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_unstructured_start_block = _libraries['FIXME_STUB'].nir_unstructured_start_block
    nir_unstructured_start_block.restype = ctypes.POINTER(struct_nir_block)
    nir_unstructured_start_block.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_block_cf_tree_next = _libraries['FIXME_STUB'].nir_block_cf_tree_next
    nir_block_cf_tree_next.restype = ctypes.POINTER(struct_nir_block)
    nir_block_cf_tree_next.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_cf_tree_prev = _libraries['FIXME_STUB'].nir_block_cf_tree_prev
    nir_block_cf_tree_prev.restype = ctypes.POINTER(struct_nir_block)
    nir_block_cf_tree_prev.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_cf_node_cf_tree_first = _libraries['FIXME_STUB'].nir_cf_node_cf_tree_first
    nir_cf_node_cf_tree_first.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_cf_tree_first.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_cf_tree_last = _libraries['FIXME_STUB'].nir_cf_node_cf_tree_last
    nir_cf_node_cf_tree_last.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_cf_tree_last.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_cf_tree_next = _libraries['FIXME_STUB'].nir_cf_node_cf_tree_next
    nir_cf_node_cf_tree_next.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_cf_tree_next.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_cf_node_cf_tree_prev = _libraries['FIXME_STUB'].nir_cf_node_cf_tree_prev
    nir_cf_node_cf_tree_prev.restype = ctypes.POINTER(struct_nir_block)
    nir_cf_node_cf_tree_prev.argtypes = [ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_block_get_following_if = _libraries['FIXME_STUB'].nir_block_get_following_if
    nir_block_get_following_if.restype = ctypes.POINTER(struct_nir_if)
    nir_block_get_following_if.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_get_following_loop = _libraries['FIXME_STUB'].nir_block_get_following_loop
    nir_block_get_following_loop.restype = ctypes.POINTER(struct_nir_loop)
    nir_block_get_following_loop.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_get_predecessors_sorted = _libraries['FIXME_STUB'].nir_block_get_predecessors_sorted
    nir_block_get_predecessors_sorted.restype = ctypes.POINTER(ctypes.POINTER(struct_nir_block))
    nir_block_get_predecessors_sorted.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_index_ssa_defs = _libraries['FIXME_STUB'].nir_index_ssa_defs
    nir_index_ssa_defs.restype = None
    nir_index_ssa_defs.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_index_instrs = _libraries['FIXME_STUB'].nir_index_instrs
    nir_index_instrs.restype = ctypes.c_uint32
    nir_index_instrs.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_index_blocks = _libraries['FIXME_STUB'].nir_index_blocks
    nir_index_blocks.restype = None
    nir_index_blocks.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_shader_clear_pass_flags = _libraries['FIXME_STUB'].nir_shader_clear_pass_flags
    nir_shader_clear_pass_flags.restype = None
    nir_shader_clear_pass_flags.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_index_vars = _libraries['FIXME_STUB'].nir_shader_index_vars
    nir_shader_index_vars.restype = ctypes.c_uint32
    nir_shader_index_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_function_impl_index_vars = _libraries['FIXME_STUB'].nir_function_impl_index_vars
    nir_function_impl_index_vars.restype = ctypes.c_uint32
    nir_function_impl_index_vars.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
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
    nir_print_shader = _libraries['FIXME_STUB'].nir_print_shader
    nir_print_shader.restype = None
    nir_print_shader.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_print_function_body = _libraries['FIXME_STUB'].nir_print_function_body
    nir_print_function_body.restype = None
    nir_print_function_body.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_print_shader_annotated = _libraries['FIXME_STUB'].nir_print_shader_annotated
    nir_print_shader_annotated.restype = None
    nir_print_shader_annotated.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_print_instr = _libraries['FIXME_STUB'].nir_print_instr
    nir_print_instr.restype = None
    nir_print_instr.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_print_deref = _libraries['FIXME_STUB'].nir_print_deref
    nir_print_deref.restype = None
    nir_print_deref.argtypes = [ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass

# values for enumeration 'mesa_log_level'
mesa_log_level__enumvalues = {
    0: 'MESA_LOG_ERROR',
    1: 'MESA_LOG_WARN',
    2: 'MESA_LOG_INFO',
    3: 'MESA_LOG_DEBUG',
}
MESA_LOG_ERROR = 0
MESA_LOG_WARN = 1
MESA_LOG_INFO = 2
MESA_LOG_DEBUG = 3
mesa_log_level = ctypes.c_uint32 # enum
try:
    nir_log_shader_annotated_tagged = _libraries['FIXME_STUB'].nir_log_shader_annotated_tagged
    nir_log_shader_annotated_tagged.restype = None
    nir_log_shader_annotated_tagged.argtypes = [mesa_log_level, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_shader_as_str = _libraries['FIXME_STUB'].nir_shader_as_str
    nir_shader_as_str.restype = ctypes.POINTER(ctypes.c_char)
    nir_shader_as_str.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_as_str_annotated = _libraries['FIXME_STUB'].nir_shader_as_str_annotated
    nir_shader_as_str_annotated.restype = ctypes.POINTER(ctypes.c_char)
    nir_shader_as_str_annotated.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_hash_table), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_instr_as_str = _libraries['FIXME_STUB'].nir_instr_as_str
    nir_instr_as_str.restype = ctypes.POINTER(ctypes.c_char)
    nir_instr_as_str.argtypes = [ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_gather_debug_info = _libraries['FIXME_STUB'].nir_shader_gather_debug_info
    nir_shader_gather_debug_info.restype = ctypes.POINTER(ctypes.c_char)
    nir_shader_gather_debug_info.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char), uint32_t]
except AttributeError:
    pass
try:
    nir_instr_clone = _libraries['FIXME_STUB'].nir_instr_clone
    nir_instr_clone.restype = ctypes.POINTER(struct_nir_instr)
    nir_instr_clone.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_clone_deep = _libraries['FIXME_STUB'].nir_instr_clone_deep
    nir_instr_clone_deep.restype = ctypes.POINTER(struct_nir_instr)
    nir_instr_clone_deep.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_alu_instr_clone = _libraries['FIXME_STUB'].nir_alu_instr_clone
    nir_alu_instr_clone.restype = ctypes.POINTER(struct_nir_alu_instr)
    nir_alu_instr_clone.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_shader_clone = _libraries['FIXME_STUB'].nir_shader_clone
    nir_shader_clone.restype = ctypes.POINTER(struct_nir_shader)
    nir_shader_clone.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_function_clone = _libraries['FIXME_STUB'].nir_function_clone
    nir_function_clone.restype = ctypes.POINTER(struct_nir_function)
    nir_function_clone.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function)]
except AttributeError:
    pass
try:
    nir_function_impl_clone = _libraries['FIXME_STUB'].nir_function_impl_clone
    nir_function_impl_clone.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_function_impl_clone.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_function_impl_clone_remap_globals = _libraries['FIXME_STUB'].nir_function_impl_clone_remap_globals
    nir_function_impl_clone_remap_globals.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_function_impl_clone_remap_globals.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_constant_clone = _libraries['FIXME_STUB'].nir_constant_clone
    nir_constant_clone.restype = ctypes.POINTER(struct_nir_constant)
    nir_constant_clone.argtypes = [ctypes.POINTER(struct_nir_constant), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_variable_clone = _libraries['FIXME_STUB'].nir_variable_clone
    nir_variable_clone.restype = ctypes.POINTER(struct_nir_variable)
    nir_variable_clone.argtypes = [ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_replace = _libraries['FIXME_STUB'].nir_shader_replace
    nir_shader_replace.restype = None
    nir_shader_replace.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_serialize_deserialize = _libraries['FIXME_STUB'].nir_shader_serialize_deserialize
    nir_shader_serialize_deserialize.restype = None
    nir_shader_serialize_deserialize.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_validate_shader = _libraries['FIXME_STUB'].nir_validate_shader
    nir_validate_shader.restype = None
    nir_validate_shader.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_validate_ssa_dominance = _libraries['FIXME_STUB'].nir_validate_ssa_dominance
    nir_validate_ssa_dominance.restype = None
    nir_validate_ssa_dominance.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_metadata_set_validation_flag = _libraries['FIXME_STUB'].nir_metadata_set_validation_flag
    nir_metadata_set_validation_flag.restype = None
    nir_metadata_set_validation_flag.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_metadata_check_validation_flag = _libraries['FIXME_STUB'].nir_metadata_check_validation_flag
    nir_metadata_check_validation_flag.restype = None
    nir_metadata_check_validation_flag.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_metadata_require_all = _libraries['FIXME_STUB'].nir_metadata_require_all
    nir_metadata_require_all.restype = None
    nir_metadata_require_all.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    should_skip_nir = _libraries['FIXME_STUB'].should_skip_nir
    should_skip_nir.restype = ctypes.c_bool
    should_skip_nir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    should_print_nir = _libraries['FIXME_STUB'].should_print_nir
    should_print_nir.restype = ctypes.c_bool
    should_print_nir.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
nir_instr_writemask_filter_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.c_uint32, ctypes.POINTER(None))
class struct_nir_builder(Structure):
    pass

struct_nir_builder._pack_ = 0 # source:False
struct_nir_builder._fields_ = [
    ('cursor', nir_cursor),
    ('exact', ctypes.c_bool),
    ('fp_fast_math', ctypes.c_uint32),
    ('shader', ctypes.POINTER(struct_nir_shader)),
    ('impl', ctypes.POINTER(struct_nir_function_impl)),
]

nir_lower_instr_cb = ctypes.CFUNCTYPE(ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
try:
    nir_function_impl_lower_instructions = _libraries['FIXME_STUB'].nir_function_impl_lower_instructions
    nir_function_impl_lower_instructions.restype = ctypes.c_bool
    nir_function_impl_lower_instructions.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_instr_filter_cb, nir_lower_instr_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_lower_instructions = _libraries['FIXME_STUB'].nir_shader_lower_instructions
    nir_shader_lower_instructions.restype = ctypes.c_bool
    nir_shader_lower_instructions.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_filter_cb, nir_lower_instr_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_calc_dominance_impl = _libraries['FIXME_STUB'].nir_calc_dominance_impl
    nir_calc_dominance_impl.restype = None
    nir_calc_dominance_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_calc_dominance = _libraries['FIXME_STUB'].nir_calc_dominance
    nir_calc_dominance.restype = None
    nir_calc_dominance.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_dominance_lca = _libraries['FIXME_STUB'].nir_dominance_lca
    nir_dominance_lca.restype = ctypes.POINTER(struct_nir_block)
    nir_dominance_lca.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_dominates = _libraries['FIXME_STUB'].nir_block_dominates
    nir_block_dominates.restype = ctypes.c_bool
    nir_block_dominates.argtypes = [ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_block_is_unreachable = _libraries['FIXME_STUB'].nir_block_is_unreachable
    nir_block_is_unreachable.restype = ctypes.c_bool
    nir_block_is_unreachable.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_dump_dom_tree_impl = _libraries['FIXME_STUB'].nir_dump_dom_tree_impl
    nir_dump_dom_tree_impl.restype = None
    nir_dump_dom_tree_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_dom_tree = _libraries['FIXME_STUB'].nir_dump_dom_tree
    nir_dump_dom_tree.restype = None
    nir_dump_dom_tree.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_dom_frontier_impl = _libraries['FIXME_STUB'].nir_dump_dom_frontier_impl
    nir_dump_dom_frontier_impl.restype = None
    nir_dump_dom_frontier_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_dom_frontier = _libraries['FIXME_STUB'].nir_dump_dom_frontier
    nir_dump_dom_frontier.restype = None
    nir_dump_dom_frontier.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_cfg_impl = _libraries['FIXME_STUB'].nir_dump_cfg_impl
    nir_dump_cfg_impl.restype = None
    nir_dump_cfg_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_dump_cfg = _libraries['FIXME_STUB'].nir_dump_cfg
    nir_dump_cfg.restype = None
    nir_dump_cfg.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
try:
    nir_gs_count_vertices_and_primitives = _libraries['FIXME_STUB'].nir_gs_count_vertices_and_primitives
    nir_gs_count_vertices_and_primitives.restype = None
    nir_gs_count_vertices_and_primitives.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_load_grouping'
c__EA_nir_load_grouping__enumvalues = {
    0: 'nir_group_all',
    1: 'nir_group_same_resource_only',
}
nir_group_all = 0
nir_group_same_resource_only = 1
c__EA_nir_load_grouping = ctypes.c_uint32 # enum
nir_load_grouping = c__EA_nir_load_grouping
nir_load_grouping__enumvalues = c__EA_nir_load_grouping__enumvalues
try:
    nir_opt_group_loads = _libraries['FIXME_STUB'].nir_opt_group_loads
    nir_opt_group_loads.restype = ctypes.c_bool
    nir_opt_group_loads.argtypes = [ctypes.POINTER(struct_nir_shader), nir_load_grouping, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_shrink_vec_array_vars = _libraries['FIXME_STUB'].nir_shrink_vec_array_vars
    nir_shrink_vec_array_vars.restype = ctypes.c_bool
    nir_shrink_vec_array_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_split_array_vars = _libraries['FIXME_STUB'].nir_split_array_vars
    nir_split_array_vars.restype = ctypes.c_bool
    nir_split_array_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_split_var_copies = _libraries['FIXME_STUB'].nir_split_var_copies
    nir_split_var_copies.restype = ctypes.c_bool
    nir_split_var_copies.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_split_per_member_structs = _libraries['FIXME_STUB'].nir_split_per_member_structs
    nir_split_per_member_structs.restype = ctypes.c_bool
    nir_split_per_member_structs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_split_struct_vars = _libraries['FIXME_STUB'].nir_split_struct_vars
    nir_split_struct_vars.restype = ctypes.c_bool
    nir_split_struct_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_returns_impl = _libraries['FIXME_STUB'].nir_lower_returns_impl
    nir_lower_returns_impl.restype = ctypes.c_bool
    nir_lower_returns_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_lower_returns = _libraries['FIXME_STUB'].nir_lower_returns
    nir_lower_returns.restype = ctypes.c_bool
    nir_lower_returns.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_inline_function_impl = _libraries['FIXME_STUB'].nir_inline_function_impl
    nir_inline_function_impl.restype = ctypes.POINTER(struct_nir_def)
    nir_inline_function_impl.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.POINTER(struct_hash_table)]
except AttributeError:
    pass
try:
    nir_inline_functions = _libraries['FIXME_STUB'].nir_inline_functions
    nir_inline_functions.restype = ctypes.c_bool
    nir_inline_functions.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_cleanup_functions = _libraries['FIXME_STUB'].nir_cleanup_functions
    nir_cleanup_functions.restype = None
    nir_cleanup_functions.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_link_shader_functions = _libraries['FIXME_STUB'].nir_link_shader_functions
    nir_link_shader_functions.restype = ctypes.c_bool
    nir_link_shader_functions.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_calls_to_builtins = _libraries['FIXME_STUB'].nir_lower_calls_to_builtins
    nir_lower_calls_to_builtins.restype = ctypes.c_bool
    nir_lower_calls_to_builtins.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_find_inlinable_uniforms = _libraries['FIXME_STUB'].nir_find_inlinable_uniforms
    nir_find_inlinable_uniforms.restype = None
    nir_find_inlinable_uniforms.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_inline_uniforms = _libraries['FIXME_STUB'].nir_inline_uniforms
    nir_inline_uniforms.restype = ctypes.c_bool
    nir_inline_uniforms.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass
try:
    nir_collect_src_uniforms = _libraries['FIXME_STUB'].nir_collect_src_uniforms
    nir_collect_src_uniforms.restype = ctypes.c_bool
    nir_collect_src_uniforms.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_add_inlinable_uniforms = _libraries['FIXME_STUB'].nir_add_inlinable_uniforms
    nir_add_inlinable_uniforms.restype = None
    nir_add_inlinable_uniforms.argtypes = [ctypes.POINTER(struct_nir_src), ctypes.POINTER(struct_nir_loop_info), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_inline_sysval = _libraries['FIXME_STUB'].nir_inline_sysval
    nir_inline_sysval.restype = ctypes.c_bool
    nir_inline_sysval.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrinsic_op, uint64_t]
except AttributeError:
    pass
try:
    nir_propagate_invariant = _libraries['FIXME_STUB'].nir_propagate_invariant
    nir_propagate_invariant.restype = ctypes.c_bool
    nir_propagate_invariant.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_var_copy_instr = _libraries['FIXME_STUB'].nir_lower_var_copy_instr
    nir_lower_var_copy_instr.restype = None
    nir_lower_var_copy_instr.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_deref_copy_instr = _libraries['FIXME_STUB'].nir_lower_deref_copy_instr
    nir_lower_deref_copy_instr.restype = None
    nir_lower_deref_copy_instr.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_lower_var_copies = _libraries['FIXME_STUB'].nir_lower_var_copies
    nir_lower_var_copies.restype = ctypes.c_bool
    nir_lower_var_copies.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_memcpy = _libraries['FIXME_STUB'].nir_opt_memcpy
    nir_opt_memcpy.restype = ctypes.c_bool
    nir_opt_memcpy.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_memcpy = _libraries['FIXME_STUB'].nir_lower_memcpy
    nir_lower_memcpy.restype = ctypes.c_bool
    nir_lower_memcpy.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_fixup_deref_modes = _libraries['FIXME_STUB'].nir_fixup_deref_modes
    nir_fixup_deref_modes.restype = ctypes.c_bool
    nir_fixup_deref_modes.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_fixup_deref_types = _libraries['FIXME_STUB'].nir_fixup_deref_types
    nir_fixup_deref_types.restype = ctypes.c_bool
    nir_fixup_deref_types.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_global_vars_to_local = _libraries['FIXME_STUB'].nir_lower_global_vars_to_local
    nir_lower_global_vars_to_local.restype = ctypes.c_bool
    nir_lower_global_vars_to_local.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_constant_to_temp = _libraries['FIXME_STUB'].nir_lower_constant_to_temp
    nir_lower_constant_to_temp.restype = None
    nir_lower_constant_to_temp.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_array_deref_of_vec_options'
c__EA_nir_lower_array_deref_of_vec_options__enumvalues = {
    1: 'nir_lower_direct_array_deref_of_vec_load',
    2: 'nir_lower_indirect_array_deref_of_vec_load',
    4: 'nir_lower_direct_array_deref_of_vec_store',
    8: 'nir_lower_indirect_array_deref_of_vec_store',
}
nir_lower_direct_array_deref_of_vec_load = 1
nir_lower_indirect_array_deref_of_vec_load = 2
nir_lower_direct_array_deref_of_vec_store = 4
nir_lower_indirect_array_deref_of_vec_store = 8
c__EA_nir_lower_array_deref_of_vec_options = ctypes.c_uint32 # enum
nir_lower_array_deref_of_vec_options = c__EA_nir_lower_array_deref_of_vec_options
nir_lower_array_deref_of_vec_options__enumvalues = c__EA_nir_lower_array_deref_of_vec_options__enumvalues
try:
    nir_lower_array_deref_of_vec = _libraries['FIXME_STUB'].nir_lower_array_deref_of_vec
    nir_lower_array_deref_of_vec.restype = ctypes.c_bool
    nir_lower_array_deref_of_vec.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_variable)), nir_lower_array_deref_of_vec_options]
except AttributeError:
    pass
try:
    nir_lower_indirect_derefs = _libraries['FIXME_STUB'].nir_lower_indirect_derefs
    nir_lower_indirect_derefs.restype = ctypes.c_bool
    nir_lower_indirect_derefs.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, uint32_t]
except AttributeError:
    pass
try:
    nir_lower_indirect_var_derefs = _libraries['FIXME_STUB'].nir_lower_indirect_var_derefs
    nir_lower_indirect_var_derefs.restype = ctypes.c_bool
    nir_lower_indirect_var_derefs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_set)]
except AttributeError:
    pass
uint8_t = ctypes.c_uint8
try:
    nir_lower_locals_to_regs = _libraries['FIXME_STUB'].nir_lower_locals_to_regs
    nir_lower_locals_to_regs.restype = ctypes.c_bool
    nir_lower_locals_to_regs.argtypes = [ctypes.POINTER(struct_nir_shader), uint8_t]
except AttributeError:
    pass
try:
    nir_lower_io_vars_to_temporaries = _libraries['FIXME_STUB'].nir_lower_io_vars_to_temporaries
    nir_lower_io_vars_to_temporaries.restype = ctypes.c_bool
    nir_lower_io_vars_to_temporaries.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
glsl_type_size_align_func = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_glsl_type), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32))
try:
    nir_lower_vars_to_scratch = _libraries['FIXME_STUB'].nir_lower_vars_to_scratch
    nir_lower_vars_to_scratch.restype = ctypes.c_bool
    nir_lower_vars_to_scratch.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_int32, glsl_type_size_align_func, glsl_type_size_align_func]
except AttributeError:
    pass
try:
    nir_lower_scratch_to_var = _libraries['FIXME_STUB'].nir_lower_scratch_to_var
    nir_lower_scratch_to_var.restype = ctypes.c_bool
    nir_lower_scratch_to_var.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_clip_halfz = _libraries['FIXME_STUB'].nir_lower_clip_halfz
    nir_lower_clip_halfz.restype = ctypes.c_bool
    nir_lower_clip_halfz.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_gather_info = _libraries['FIXME_STUB'].nir_shader_gather_info
    nir_shader_gather_info.restype = None
    nir_shader_gather_info.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_gather_types = _libraries['FIXME_STUB'].nir_gather_types
    nir_gather_types.restype = None
    nir_gather_types.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    nir_remove_unused_varyings = _libraries['FIXME_STUB'].nir_remove_unused_varyings
    nir_remove_unused_varyings.restype = ctypes.c_bool
    nir_remove_unused_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_unused_io_vars = _libraries['FIXME_STUB'].nir_remove_unused_io_vars
    nir_remove_unused_io_vars.restype = ctypes.c_bool
    nir_remove_unused_io_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    nir_compact_varyings = _libraries['FIXME_STUB'].nir_compact_varyings
    nir_compact_varyings.restype = None
    nir_compact_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_link_xfb_varyings = _libraries['FIXME_STUB'].nir_link_xfb_varyings
    nir_link_xfb_varyings.restype = None
    nir_link_xfb_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_link_opt_varyings = _libraries['FIXME_STUB'].nir_link_opt_varyings
    nir_link_opt_varyings.restype = ctypes.c_bool
    nir_link_opt_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_link_varying_precision = _libraries['FIXME_STUB'].nir_link_varying_precision
    nir_link_varying_precision.restype = None
    nir_link_varying_precision.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_clone_uniform_variable = _libraries['FIXME_STUB'].nir_clone_uniform_variable
    nir_clone_uniform_variable.restype = ctypes.POINTER(struct_nir_variable)
    nir_clone_uniform_variable.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_variable), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_clone_deref_instr = _libraries['FIXME_STUB'].nir_clone_deref_instr
    nir_clone_deref_instr.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_clone_deref_instr.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_opt_varyings_progress'
c__EA_nir_opt_varyings_progress__enumvalues = {
    1: 'nir_progress_producer',
    2: 'nir_progress_consumer',
}
nir_progress_producer = 1
nir_progress_consumer = 2
c__EA_nir_opt_varyings_progress = ctypes.c_uint32 # enum
nir_opt_varyings_progress = c__EA_nir_opt_varyings_progress
nir_opt_varyings_progress__enumvalues = c__EA_nir_opt_varyings_progress__enumvalues
try:
    nir_opt_varyings = _libraries['FIXME_STUB'].nir_opt_varyings
    nir_opt_varyings.restype = nir_opt_varyings_progress
    nir_opt_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_varying_var_mask = _libraries['FIXME_STUB'].nir_varying_var_mask
    nir_varying_var_mask.restype = ctypes.c_uint32
    nir_varying_var_mask.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_varyings_bulk = _libraries['FIXME_STUB'].nir_opt_varyings_bulk
    nir_opt_varyings_bulk.restype = None
    nir_opt_varyings_bulk.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_nir_shader)), uint32_t, ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_nir_shader))]
except AttributeError:
    pass

# values for enumeration 'c__EA_gl_varying_slot'
c__EA_gl_varying_slot__enumvalues = {
    0: 'VARYING_SLOT_POS',
    1: 'VARYING_SLOT_COL0',
    2: 'VARYING_SLOT_COL1',
    3: 'VARYING_SLOT_FOGC',
    4: 'VARYING_SLOT_TEX0',
    5: 'VARYING_SLOT_TEX1',
    6: 'VARYING_SLOT_TEX2',
    7: 'VARYING_SLOT_TEX3',
    8: 'VARYING_SLOT_TEX4',
    9: 'VARYING_SLOT_TEX5',
    10: 'VARYING_SLOT_TEX6',
    11: 'VARYING_SLOT_TEX7',
    12: 'VARYING_SLOT_PSIZ',
    13: 'VARYING_SLOT_BFC0',
    14: 'VARYING_SLOT_BFC1',
    15: 'VARYING_SLOT_EDGE',
    16: 'VARYING_SLOT_CLIP_VERTEX',
    17: 'VARYING_SLOT_CLIP_DIST0',
    18: 'VARYING_SLOT_CLIP_DIST1',
    19: 'VARYING_SLOT_CULL_DIST0',
    20: 'VARYING_SLOT_CULL_DIST1',
    21: 'VARYING_SLOT_PRIMITIVE_ID',
    22: 'VARYING_SLOT_LAYER',
    23: 'VARYING_SLOT_VIEWPORT',
    24: 'VARYING_SLOT_FACE',
    25: 'VARYING_SLOT_PNTC',
    26: 'VARYING_SLOT_TESS_LEVEL_OUTER',
    27: 'VARYING_SLOT_TESS_LEVEL_INNER',
    28: 'VARYING_SLOT_BOUNDING_BOX0',
    29: 'VARYING_SLOT_BOUNDING_BOX1',
    30: 'VARYING_SLOT_VIEW_INDEX',
    31: 'VARYING_SLOT_VIEWPORT_MASK',
    24: 'VARYING_SLOT_PRIMITIVE_SHADING_RATE',
    26: 'VARYING_SLOT_PRIMITIVE_COUNT',
    27: 'VARYING_SLOT_PRIMITIVE_INDICES',
    28: 'VARYING_SLOT_TASK_COUNT',
    28: 'VARYING_SLOT_CULL_PRIMITIVE',
    32: 'VARYING_SLOT_VAR0',
    33: 'VARYING_SLOT_VAR1',
    34: 'VARYING_SLOT_VAR2',
    35: 'VARYING_SLOT_VAR3',
    36: 'VARYING_SLOT_VAR4',
    37: 'VARYING_SLOT_VAR5',
    38: 'VARYING_SLOT_VAR6',
    39: 'VARYING_SLOT_VAR7',
    40: 'VARYING_SLOT_VAR8',
    41: 'VARYING_SLOT_VAR9',
    42: 'VARYING_SLOT_VAR10',
    43: 'VARYING_SLOT_VAR11',
    44: 'VARYING_SLOT_VAR12',
    45: 'VARYING_SLOT_VAR13',
    46: 'VARYING_SLOT_VAR14',
    47: 'VARYING_SLOT_VAR15',
    48: 'VARYING_SLOT_VAR16',
    49: 'VARYING_SLOT_VAR17',
    50: 'VARYING_SLOT_VAR18',
    51: 'VARYING_SLOT_VAR19',
    52: 'VARYING_SLOT_VAR20',
    53: 'VARYING_SLOT_VAR21',
    54: 'VARYING_SLOT_VAR22',
    55: 'VARYING_SLOT_VAR23',
    56: 'VARYING_SLOT_VAR24',
    57: 'VARYING_SLOT_VAR25',
    58: 'VARYING_SLOT_VAR26',
    59: 'VARYING_SLOT_VAR27',
    60: 'VARYING_SLOT_VAR28',
    61: 'VARYING_SLOT_VAR29',
    62: 'VARYING_SLOT_VAR30',
    63: 'VARYING_SLOT_VAR31',
    64: 'VARYING_SLOT_PATCH0',
    65: 'VARYING_SLOT_PATCH1',
    66: 'VARYING_SLOT_PATCH2',
    67: 'VARYING_SLOT_PATCH3',
    68: 'VARYING_SLOT_PATCH4',
    69: 'VARYING_SLOT_PATCH5',
    70: 'VARYING_SLOT_PATCH6',
    71: 'VARYING_SLOT_PATCH7',
    72: 'VARYING_SLOT_PATCH8',
    73: 'VARYING_SLOT_PATCH9',
    74: 'VARYING_SLOT_PATCH10',
    75: 'VARYING_SLOT_PATCH11',
    76: 'VARYING_SLOT_PATCH12',
    77: 'VARYING_SLOT_PATCH13',
    78: 'VARYING_SLOT_PATCH14',
    79: 'VARYING_SLOT_PATCH15',
    80: 'VARYING_SLOT_PATCH16',
    81: 'VARYING_SLOT_PATCH17',
    82: 'VARYING_SLOT_PATCH18',
    83: 'VARYING_SLOT_PATCH19',
    84: 'VARYING_SLOT_PATCH20',
    85: 'VARYING_SLOT_PATCH21',
    86: 'VARYING_SLOT_PATCH22',
    87: 'VARYING_SLOT_PATCH23',
    88: 'VARYING_SLOT_PATCH24',
    89: 'VARYING_SLOT_PATCH25',
    90: 'VARYING_SLOT_PATCH26',
    91: 'VARYING_SLOT_PATCH27',
    92: 'VARYING_SLOT_PATCH28',
    93: 'VARYING_SLOT_PATCH29',
    94: 'VARYING_SLOT_PATCH30',
    95: 'VARYING_SLOT_PATCH31',
    96: 'VARYING_SLOT_VAR0_16BIT',
    97: 'VARYING_SLOT_VAR1_16BIT',
    98: 'VARYING_SLOT_VAR2_16BIT',
    99: 'VARYING_SLOT_VAR3_16BIT',
    100: 'VARYING_SLOT_VAR4_16BIT',
    101: 'VARYING_SLOT_VAR5_16BIT',
    102: 'VARYING_SLOT_VAR6_16BIT',
    103: 'VARYING_SLOT_VAR7_16BIT',
    104: 'VARYING_SLOT_VAR8_16BIT',
    105: 'VARYING_SLOT_VAR9_16BIT',
    106: 'VARYING_SLOT_VAR10_16BIT',
    107: 'VARYING_SLOT_VAR11_16BIT',
    108: 'VARYING_SLOT_VAR12_16BIT',
    109: 'VARYING_SLOT_VAR13_16BIT',
    110: 'VARYING_SLOT_VAR14_16BIT',
    111: 'VARYING_SLOT_VAR15_16BIT',
    112: 'NUM_TOTAL_VARYING_SLOTS',
}
VARYING_SLOT_POS = 0
VARYING_SLOT_COL0 = 1
VARYING_SLOT_COL1 = 2
VARYING_SLOT_FOGC = 3
VARYING_SLOT_TEX0 = 4
VARYING_SLOT_TEX1 = 5
VARYING_SLOT_TEX2 = 6
VARYING_SLOT_TEX3 = 7
VARYING_SLOT_TEX4 = 8
VARYING_SLOT_TEX5 = 9
VARYING_SLOT_TEX6 = 10
VARYING_SLOT_TEX7 = 11
VARYING_SLOT_PSIZ = 12
VARYING_SLOT_BFC0 = 13
VARYING_SLOT_BFC1 = 14
VARYING_SLOT_EDGE = 15
VARYING_SLOT_CLIP_VERTEX = 16
VARYING_SLOT_CLIP_DIST0 = 17
VARYING_SLOT_CLIP_DIST1 = 18
VARYING_SLOT_CULL_DIST0 = 19
VARYING_SLOT_CULL_DIST1 = 20
VARYING_SLOT_PRIMITIVE_ID = 21
VARYING_SLOT_LAYER = 22
VARYING_SLOT_VIEWPORT = 23
VARYING_SLOT_FACE = 24
VARYING_SLOT_PNTC = 25
VARYING_SLOT_TESS_LEVEL_OUTER = 26
VARYING_SLOT_TESS_LEVEL_INNER = 27
VARYING_SLOT_BOUNDING_BOX0 = 28
VARYING_SLOT_BOUNDING_BOX1 = 29
VARYING_SLOT_VIEW_INDEX = 30
VARYING_SLOT_VIEWPORT_MASK = 31
VARYING_SLOT_PRIMITIVE_SHADING_RATE = 24
VARYING_SLOT_PRIMITIVE_COUNT = 26
VARYING_SLOT_PRIMITIVE_INDICES = 27
VARYING_SLOT_TASK_COUNT = 28
VARYING_SLOT_CULL_PRIMITIVE = 28
VARYING_SLOT_VAR0 = 32
VARYING_SLOT_VAR1 = 33
VARYING_SLOT_VAR2 = 34
VARYING_SLOT_VAR3 = 35
VARYING_SLOT_VAR4 = 36
VARYING_SLOT_VAR5 = 37
VARYING_SLOT_VAR6 = 38
VARYING_SLOT_VAR7 = 39
VARYING_SLOT_VAR8 = 40
VARYING_SLOT_VAR9 = 41
VARYING_SLOT_VAR10 = 42
VARYING_SLOT_VAR11 = 43
VARYING_SLOT_VAR12 = 44
VARYING_SLOT_VAR13 = 45
VARYING_SLOT_VAR14 = 46
VARYING_SLOT_VAR15 = 47
VARYING_SLOT_VAR16 = 48
VARYING_SLOT_VAR17 = 49
VARYING_SLOT_VAR18 = 50
VARYING_SLOT_VAR19 = 51
VARYING_SLOT_VAR20 = 52
VARYING_SLOT_VAR21 = 53
VARYING_SLOT_VAR22 = 54
VARYING_SLOT_VAR23 = 55
VARYING_SLOT_VAR24 = 56
VARYING_SLOT_VAR25 = 57
VARYING_SLOT_VAR26 = 58
VARYING_SLOT_VAR27 = 59
VARYING_SLOT_VAR28 = 60
VARYING_SLOT_VAR29 = 61
VARYING_SLOT_VAR30 = 62
VARYING_SLOT_VAR31 = 63
VARYING_SLOT_PATCH0 = 64
VARYING_SLOT_PATCH1 = 65
VARYING_SLOT_PATCH2 = 66
VARYING_SLOT_PATCH3 = 67
VARYING_SLOT_PATCH4 = 68
VARYING_SLOT_PATCH5 = 69
VARYING_SLOT_PATCH6 = 70
VARYING_SLOT_PATCH7 = 71
VARYING_SLOT_PATCH8 = 72
VARYING_SLOT_PATCH9 = 73
VARYING_SLOT_PATCH10 = 74
VARYING_SLOT_PATCH11 = 75
VARYING_SLOT_PATCH12 = 76
VARYING_SLOT_PATCH13 = 77
VARYING_SLOT_PATCH14 = 78
VARYING_SLOT_PATCH15 = 79
VARYING_SLOT_PATCH16 = 80
VARYING_SLOT_PATCH17 = 81
VARYING_SLOT_PATCH18 = 82
VARYING_SLOT_PATCH19 = 83
VARYING_SLOT_PATCH20 = 84
VARYING_SLOT_PATCH21 = 85
VARYING_SLOT_PATCH22 = 86
VARYING_SLOT_PATCH23 = 87
VARYING_SLOT_PATCH24 = 88
VARYING_SLOT_PATCH25 = 89
VARYING_SLOT_PATCH26 = 90
VARYING_SLOT_PATCH27 = 91
VARYING_SLOT_PATCH28 = 92
VARYING_SLOT_PATCH29 = 93
VARYING_SLOT_PATCH30 = 94
VARYING_SLOT_PATCH31 = 95
VARYING_SLOT_VAR0_16BIT = 96
VARYING_SLOT_VAR1_16BIT = 97
VARYING_SLOT_VAR2_16BIT = 98
VARYING_SLOT_VAR3_16BIT = 99
VARYING_SLOT_VAR4_16BIT = 100
VARYING_SLOT_VAR5_16BIT = 101
VARYING_SLOT_VAR6_16BIT = 102
VARYING_SLOT_VAR7_16BIT = 103
VARYING_SLOT_VAR8_16BIT = 104
VARYING_SLOT_VAR9_16BIT = 105
VARYING_SLOT_VAR10_16BIT = 106
VARYING_SLOT_VAR11_16BIT = 107
VARYING_SLOT_VAR12_16BIT = 108
VARYING_SLOT_VAR13_16BIT = 109
VARYING_SLOT_VAR14_16BIT = 110
VARYING_SLOT_VAR15_16BIT = 111
NUM_TOTAL_VARYING_SLOTS = 112
c__EA_gl_varying_slot = ctypes.c_uint32 # enum
gl_varying_slot = c__EA_gl_varying_slot
gl_varying_slot__enumvalues = c__EA_gl_varying_slot__enumvalues
try:
    nir_slot_is_sysval_output = _libraries['FIXME_STUB'].nir_slot_is_sysval_output
    nir_slot_is_sysval_output.restype = ctypes.c_bool
    nir_slot_is_sysval_output.argtypes = [gl_varying_slot, mesa_shader_stage]
except AttributeError:
    pass
try:
    nir_slot_is_varying = _libraries['FIXME_STUB'].nir_slot_is_varying
    nir_slot_is_varying.restype = ctypes.c_bool
    nir_slot_is_varying.argtypes = [gl_varying_slot, mesa_shader_stage]
except AttributeError:
    pass
try:
    nir_slot_is_sysval_output_and_varying = _libraries['FIXME_STUB'].nir_slot_is_sysval_output_and_varying
    nir_slot_is_sysval_output_and_varying.restype = ctypes.c_bool
    nir_slot_is_sysval_output_and_varying.argtypes = [gl_varying_slot, mesa_shader_stage]
except AttributeError:
    pass
try:
    nir_remove_varying = _libraries['FIXME_STUB'].nir_remove_varying
    nir_remove_varying.restype = ctypes.c_bool
    nir_remove_varying.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), mesa_shader_stage]
except AttributeError:
    pass
try:
    nir_remove_sysval_output = _libraries['FIXME_STUB'].nir_remove_sysval_output
    nir_remove_sysval_output.restype = ctypes.c_bool
    nir_remove_sysval_output.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), mesa_shader_stage]
except AttributeError:
    pass
try:
    nir_lower_amul = _libraries['FIXME_STUB'].nir_lower_amul
    nir_lower_amul.restype = ctypes.c_bool
    nir_lower_amul.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_glsl_type), ctypes.c_bool)]
except AttributeError:
    pass
try:
    nir_lower_ubo_vec4 = _libraries['FIXME_STUB'].nir_lower_ubo_vec4
    nir_lower_ubo_vec4.restype = ctypes.c_bool
    nir_lower_ubo_vec4.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_sort_variables_by_location = _libraries['FIXME_STUB'].nir_sort_variables_by_location
    nir_sort_variables_by_location.restype = None
    nir_sort_variables_by_location.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_assign_io_var_locations = _libraries['FIXME_STUB'].nir_assign_io_var_locations
    nir_assign_io_var_locations.restype = None
    nir_assign_io_var_locations.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.POINTER(ctypes.c_uint32), mesa_shader_stage]
except AttributeError:
    pass
try:
    nir_opt_clip_cull_const = _libraries['FIXME_STUB'].nir_opt_clip_cull_const
    nir_opt_clip_cull_const.restype = ctypes.c_bool
    nir_opt_clip_cull_const.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_io_options'
c__EA_nir_lower_io_options__enumvalues = {
    1: 'nir_lower_io_lower_64bit_to_32',
    2: 'nir_lower_io_lower_64bit_float_to_32',
    4: 'nir_lower_io_lower_64bit_to_32_new',
    8: 'nir_lower_io_use_interpolated_input_intrinsics',
}
nir_lower_io_lower_64bit_to_32 = 1
nir_lower_io_lower_64bit_float_to_32 = 2
nir_lower_io_lower_64bit_to_32_new = 4
nir_lower_io_use_interpolated_input_intrinsics = 8
c__EA_nir_lower_io_options = ctypes.c_uint32 # enum
nir_lower_io_options = c__EA_nir_lower_io_options
nir_lower_io_options__enumvalues = c__EA_nir_lower_io_options__enumvalues
try:
    nir_lower_io = _libraries['FIXME_STUB'].nir_lower_io
    nir_lower_io.restype = ctypes.c_bool
    nir_lower_io.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_glsl_type), ctypes.c_bool), nir_lower_io_options]
except AttributeError:
    pass
try:
    nir_io_add_const_offset_to_base = _libraries['FIXME_STUB'].nir_io_add_const_offset_to_base
    nir_io_add_const_offset_to_base.restype = ctypes.c_bool
    nir_io_add_const_offset_to_base.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_io_passes = _libraries['FIXME_STUB'].nir_lower_io_passes
    nir_lower_io_passes.restype = None
    nir_lower_io_passes.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_io_add_intrinsic_xfb_info = _libraries['FIXME_STUB'].nir_io_add_intrinsic_xfb_info
    nir_io_add_intrinsic_xfb_info.restype = ctypes.c_bool
    nir_io_add_intrinsic_xfb_info.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_io_indirect_loads = _libraries['FIXME_STUB'].nir_lower_io_indirect_loads
    nir_lower_io_indirect_loads.restype = ctypes.c_bool
    nir_lower_io_indirect_loads.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_vars_to_explicit_types = _libraries['FIXME_STUB'].nir_lower_vars_to_explicit_types
    nir_lower_vars_to_explicit_types.restype = ctypes.c_bool
    nir_lower_vars_to_explicit_types.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, glsl_type_size_align_func]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    nir_gather_explicit_io_initializers = _libraries['FIXME_STUB'].nir_gather_explicit_io_initializers
    nir_gather_explicit_io_initializers.restype = None
    nir_gather_explicit_io_initializers.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(None), size_t, nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_vec3_to_vec4 = _libraries['FIXME_STUB'].nir_lower_vec3_to_vec4
    nir_lower_vec3_to_vec4.restype = ctypes.c_bool
    nir_lower_vec3_to_vec4.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_address_format'
c__EA_nir_address_format__enumvalues = {
    0: 'nir_address_format_32bit_global',
    1: 'nir_address_format_64bit_global',
    2: 'nir_address_format_2x32bit_global',
    3: 'nir_address_format_64bit_global_32bit_offset',
    4: 'nir_address_format_64bit_bounded_global',
    5: 'nir_address_format_32bit_index_offset',
    6: 'nir_address_format_32bit_index_offset_pack64',
    7: 'nir_address_format_vec2_index_32bit_offset',
    8: 'nir_address_format_62bit_generic',
    9: 'nir_address_format_32bit_offset',
    10: 'nir_address_format_32bit_offset_as_64bit',
    11: 'nir_address_format_logical',
}
nir_address_format_32bit_global = 0
nir_address_format_64bit_global = 1
nir_address_format_2x32bit_global = 2
nir_address_format_64bit_global_32bit_offset = 3
nir_address_format_64bit_bounded_global = 4
nir_address_format_32bit_index_offset = 5
nir_address_format_32bit_index_offset_pack64 = 6
nir_address_format_vec2_index_32bit_offset = 7
nir_address_format_62bit_generic = 8
nir_address_format_32bit_offset = 9
nir_address_format_32bit_offset_as_64bit = 10
nir_address_format_logical = 11
c__EA_nir_address_format = ctypes.c_uint32 # enum
nir_address_format = c__EA_nir_address_format
nir_address_format__enumvalues = c__EA_nir_address_format__enumvalues
try:
    nir_address_format_bit_size = _libraries['FIXME_STUB'].nir_address_format_bit_size
    nir_address_format_bit_size.restype = ctypes.c_uint32
    nir_address_format_bit_size.argtypes = [nir_address_format]
except AttributeError:
    pass
try:
    nir_address_format_num_components = _libraries['FIXME_STUB'].nir_address_format_num_components
    nir_address_format_num_components.restype = ctypes.c_uint32
    nir_address_format_num_components.argtypes = [nir_address_format]
except AttributeError:
    pass
try:
    nir_address_format_to_glsl_type = _libraries['FIXME_STUB'].nir_address_format_to_glsl_type
    nir_address_format_to_glsl_type.restype = ctypes.POINTER(struct_glsl_type)
    nir_address_format_to_glsl_type.argtypes = [nir_address_format]
except AttributeError:
    pass
try:
    nir_address_format_null_value = _libraries['FIXME_STUB'].nir_address_format_null_value
    nir_address_format_null_value.restype = ctypes.POINTER(union_c__UA_nir_const_value)
    nir_address_format_null_value.argtypes = [nir_address_format]
except AttributeError:
    pass
try:
    nir_build_addr_iadd = _libraries['FIXME_STUB'].nir_build_addr_iadd
    nir_build_addr_iadd.restype = ctypes.POINTER(struct_nir_def)
    nir_build_addr_iadd.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_address_format, nir_variable_mode, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_addr_iadd_imm = _libraries['FIXME_STUB'].nir_build_addr_iadd_imm
    nir_build_addr_iadd_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_build_addr_iadd_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_address_format, nir_variable_mode, int64_t]
except AttributeError:
    pass
try:
    nir_build_addr_ieq = _libraries['FIXME_STUB'].nir_build_addr_ieq
    nir_build_addr_ieq.restype = ctypes.POINTER(struct_nir_def)
    nir_build_addr_ieq.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), nir_address_format]
except AttributeError:
    pass
try:
    nir_build_addr_isub = _libraries['FIXME_STUB'].nir_build_addr_isub
    nir_build_addr_isub.restype = ctypes.POINTER(struct_nir_def)
    nir_build_addr_isub.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), nir_address_format]
except AttributeError:
    pass
try:
    nir_explicit_io_address_from_deref = _libraries['FIXME_STUB'].nir_explicit_io_address_from_deref
    nir_explicit_io_address_from_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_explicit_io_address_from_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), nir_address_format]
except AttributeError:
    pass
try:
    nir_get_explicit_deref_align = _libraries['FIXME_STUB'].nir_get_explicit_deref_align
    nir_get_explicit_deref_align.restype = ctypes.c_bool
    nir_get_explicit_deref_align.argtypes = [ctypes.POINTER(struct_nir_deref_instr), ctypes.c_bool, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    nir_lower_explicit_io_instr = _libraries['FIXME_STUB'].nir_lower_explicit_io_instr
    nir_lower_explicit_io_instr.restype = None
    nir_lower_explicit_io_instr.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_def), nir_address_format]
except AttributeError:
    pass
try:
    nir_lower_explicit_io = _libraries['FIXME_STUB'].nir_lower_explicit_io
    nir_lower_explicit_io.restype = ctypes.c_bool
    nir_lower_explicit_io.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, nir_address_format]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_mem_access_shift_method'
c__EA_nir_mem_access_shift_method__enumvalues = {
    0: 'nir_mem_access_shift_method_scalar',
    1: 'nir_mem_access_shift_method_shift64',
    2: 'nir_mem_access_shift_method_bytealign_amd',
}
nir_mem_access_shift_method_scalar = 0
nir_mem_access_shift_method_shift64 = 1
nir_mem_access_shift_method_bytealign_amd = 2
c__EA_nir_mem_access_shift_method = ctypes.c_uint32 # enum
nir_mem_access_shift_method = c__EA_nir_mem_access_shift_method
nir_mem_access_shift_method__enumvalues = c__EA_nir_mem_access_shift_method__enumvalues
class struct_nir_mem_access_size_align(Structure):
    pass

struct_nir_mem_access_size_align._pack_ = 1 # source:False
struct_nir_mem_access_size_align._fields_ = [
    ('num_components', ctypes.c_ubyte),
    ('bit_size', ctypes.c_ubyte),
    ('align', ctypes.c_uint16),
    ('shift', nir_mem_access_shift_method),
]

nir_mem_access_size_align = struct_nir_mem_access_size_align

# values for enumeration 'gl_access_qualifier'
gl_access_qualifier__enumvalues = {
    1: 'ACCESS_COHERENT',
    2: 'ACCESS_RESTRICT',
    4: 'ACCESS_VOLATILE',
    8: 'ACCESS_NON_READABLE',
    16: 'ACCESS_NON_WRITEABLE',
    32: 'ACCESS_NON_UNIFORM',
    64: 'ACCESS_CAN_REORDER',
    128: 'ACCESS_NON_TEMPORAL',
    256: 'ACCESS_INCLUDE_HELPERS',
    512: 'ACCESS_IS_SWIZZLED_AMD',
    1024: 'ACCESS_USES_FORMAT_AMD',
    2048: 'ACCESS_FMASK_LOWERED_AMD',
    4096: 'ACCESS_CAN_SPECULATE',
    8192: 'ACCESS_CP_GE_COHERENT_AMD',
    16384: 'ACCESS_IN_BOUNDS',
    32768: 'ACCESS_KEEP_SCALAR',
    65536: 'ACCESS_SMEM_AMD',
    131072: 'ACCESS_SKIP_HELPERS',
}
ACCESS_COHERENT = 1
ACCESS_RESTRICT = 2
ACCESS_VOLATILE = 4
ACCESS_NON_READABLE = 8
ACCESS_NON_WRITEABLE = 16
ACCESS_NON_UNIFORM = 32
ACCESS_CAN_REORDER = 64
ACCESS_NON_TEMPORAL = 128
ACCESS_INCLUDE_HELPERS = 256
ACCESS_IS_SWIZZLED_AMD = 512
ACCESS_USES_FORMAT_AMD = 1024
ACCESS_FMASK_LOWERED_AMD = 2048
ACCESS_CAN_SPECULATE = 4096
ACCESS_CP_GE_COHERENT_AMD = 8192
ACCESS_IN_BOUNDS = 16384
ACCESS_KEEP_SCALAR = 32768
ACCESS_SMEM_AMD = 65536
ACCESS_SKIP_HELPERS = 131072
gl_access_qualifier = ctypes.c_uint32 # enum
nir_lower_mem_access_bit_sizes_cb = ctypes.CFUNCTYPE(struct_nir_mem_access_size_align, c__EA_nir_intrinsic_op, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, gl_access_qualifier, ctypes.POINTER(None))
class struct_nir_lower_mem_access_bit_sizes_options(Structure):
    pass

struct_nir_lower_mem_access_bit_sizes_options._pack_ = 1 # source:False
struct_nir_lower_mem_access_bit_sizes_options._fields_ = [
    ('callback', ctypes.CFUNCTYPE(struct_nir_mem_access_size_align, c__EA_nir_intrinsic_op, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool, gl_access_qualifier, ctypes.POINTER(None))),
    ('modes', nir_variable_mode),
    ('may_lower_unaligned_stores_to_atomics', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('cb_data', ctypes.POINTER(None)),
]

nir_lower_mem_access_bit_sizes_options = struct_nir_lower_mem_access_bit_sizes_options
try:
    nir_lower_mem_access_bit_sizes = _libraries['FIXME_STUB'].nir_lower_mem_access_bit_sizes
    nir_lower_mem_access_bit_sizes.restype = ctypes.c_bool
    nir_lower_mem_access_bit_sizes.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_mem_access_bit_sizes_options)]
except AttributeError:
    pass
try:
    nir_lower_robust_access = _libraries['FIXME_STUB'].nir_lower_robust_access
    nir_lower_robust_access.restype = ctypes.c_bool
    nir_lower_robust_access.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrin_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
nir_should_vectorize_mem_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
class struct_nir_load_store_vectorize_options(Structure):
    pass

struct_nir_load_store_vectorize_options._pack_ = 1 # source:False
struct_nir_load_store_vectorize_options._fields_ = [
    ('callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))),
    ('modes', nir_variable_mode),
    ('robust_modes', nir_variable_mode),
    ('cb_data', ctypes.POINTER(None)),
    ('has_shared2_amd', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

nir_load_store_vectorize_options = struct_nir_load_store_vectorize_options
try:
    nir_opt_load_store_vectorize = _libraries['FIXME_STUB'].nir_opt_load_store_vectorize
    nir_opt_load_store_vectorize.restype = ctypes.c_bool
    nir_opt_load_store_vectorize.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_load_store_vectorize_options)]
except AttributeError:
    pass
try:
    nir_opt_load_store_update_alignments = _libraries['FIXME_STUB'].nir_opt_load_store_update_alignments
    nir_opt_load_store_update_alignments.restype = ctypes.c_bool
    nir_opt_load_store_update_alignments.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
nir_lower_shader_calls_should_remat_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
class struct_nir_lower_shader_calls_options(Structure):
    pass

struct_nir_lower_shader_calls_options._pack_ = 1 # source:False
struct_nir_lower_shader_calls_options._fields_ = [
    ('address_format', nir_address_format),
    ('stack_alignment', ctypes.c_uint32),
    ('localized_loads', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('vectorizer_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int64, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))),
    ('vectorizer_data', ctypes.POINTER(None)),
    ('should_remat_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('should_remat_data', ctypes.POINTER(None)),
]

nir_lower_shader_calls_options = struct_nir_lower_shader_calls_options
try:
    nir_lower_shader_calls = _libraries['FIXME_STUB'].nir_lower_shader_calls
    nir_lower_shader_calls.restype = ctypes.c_bool
    nir_lower_shader_calls.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_shader_calls_options), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct_nir_shader))), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_get_io_offset_src_number = _libraries['FIXME_STUB'].nir_get_io_offset_src_number
    nir_get_io_offset_src_number.restype = ctypes.c_int32
    nir_get_io_offset_src_number.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_index_src_number = _libraries['FIXME_STUB'].nir_get_io_index_src_number
    nir_get_io_index_src_number.restype = ctypes.c_int32
    nir_get_io_index_src_number.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_arrayed_index_src_number = _libraries['FIXME_STUB'].nir_get_io_arrayed_index_src_number
    nir_get_io_arrayed_index_src_number.restype = ctypes.c_int32
    nir_get_io_arrayed_index_src_number.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_offset_src = _libraries['FIXME_STUB'].nir_get_io_offset_src
    nir_get_io_offset_src.restype = ctypes.POINTER(struct_nir_src)
    nir_get_io_offset_src.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_index_src = _libraries['FIXME_STUB'].nir_get_io_index_src
    nir_get_io_index_src.restype = ctypes.POINTER(struct_nir_src)
    nir_get_io_index_src.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_io_arrayed_index_src = _libraries['FIXME_STUB'].nir_get_io_arrayed_index_src
    nir_get_io_arrayed_index_src.restype = ctypes.POINTER(struct_nir_src)
    nir_get_io_arrayed_index_src.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_get_shader_call_payload_src = _libraries['FIXME_STUB'].nir_get_shader_call_payload_src
    nir_get_shader_call_payload_src.restype = ctypes.POINTER(struct_nir_src)
    nir_get_shader_call_payload_src.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_is_output_load = _libraries['FIXME_STUB'].nir_is_output_load
    nir_is_output_load.restype = ctypes.c_bool
    nir_is_output_load.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_is_arrayed_io = _libraries['FIXME_STUB'].nir_is_arrayed_io
    nir_is_arrayed_io.restype = ctypes.c_bool
    nir_is_arrayed_io.argtypes = [ctypes.POINTER(struct_nir_variable), mesa_shader_stage]
except AttributeError:
    pass
try:
    nir_lower_reg_intrinsics_to_ssa_impl = _libraries['FIXME_STUB'].nir_lower_reg_intrinsics_to_ssa_impl
    nir_lower_reg_intrinsics_to_ssa_impl.restype = ctypes.c_bool
    nir_lower_reg_intrinsics_to_ssa_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_lower_reg_intrinsics_to_ssa = _libraries['FIXME_STUB'].nir_lower_reg_intrinsics_to_ssa
    nir_lower_reg_intrinsics_to_ssa.restype = ctypes.c_bool
    nir_lower_reg_intrinsics_to_ssa.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_vars_to_ssa = _libraries['FIXME_STUB'].nir_lower_vars_to_ssa
    nir_lower_vars_to_ssa.restype = ctypes.c_bool
    nir_lower_vars_to_ssa.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_dead_derefs = _libraries['FIXME_STUB'].nir_remove_dead_derefs
    nir_remove_dead_derefs.restype = ctypes.c_bool
    nir_remove_dead_derefs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_dead_derefs_impl = _libraries['FIXME_STUB'].nir_remove_dead_derefs_impl
    nir_remove_dead_derefs_impl.restype = ctypes.c_bool
    nir_remove_dead_derefs_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
class struct_nir_remove_dead_variables_options(Structure):
    pass

struct_nir_remove_dead_variables_options._pack_ = 1 # source:False
struct_nir_remove_dead_variables_options._fields_ = [
    ('can_remove_var', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_variable), ctypes.POINTER(None))),
    ('can_remove_var_data', ctypes.POINTER(None)),
]

nir_remove_dead_variables_options = struct_nir_remove_dead_variables_options
try:
    nir_remove_dead_variables = _libraries['FIXME_STUB'].nir_remove_dead_variables
    nir_remove_dead_variables.restype = ctypes.c_bool
    nir_remove_dead_variables.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.POINTER(struct_nir_remove_dead_variables_options)]
except AttributeError:
    pass
try:
    nir_lower_variable_initializers = _libraries['FIXME_STUB'].nir_lower_variable_initializers
    nir_lower_variable_initializers.restype = ctypes.c_bool
    nir_lower_variable_initializers.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_zero_initialize_shared_memory = _libraries['FIXME_STUB'].nir_zero_initialize_shared_memory
    nir_zero_initialize_shared_memory.restype = ctypes.c_bool
    nir_zero_initialize_shared_memory.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_clear_shared_memory = _libraries['FIXME_STUB'].nir_clear_shared_memory
    nir_clear_shared_memory.restype = ctypes.c_bool
    nir_clear_shared_memory.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_opt_move_to_top_options'
c__EA_nir_opt_move_to_top_options__enumvalues = {
    1: 'nir_move_to_entry_block_only',
    2: 'nir_move_to_top_input_loads',
    4: 'nir_move_to_top_load_smem_amd',
}
nir_move_to_entry_block_only = 1
nir_move_to_top_input_loads = 2
nir_move_to_top_load_smem_amd = 4
c__EA_nir_opt_move_to_top_options = ctypes.c_uint32 # enum
nir_opt_move_to_top_options = c__EA_nir_opt_move_to_top_options
nir_opt_move_to_top_options__enumvalues = c__EA_nir_opt_move_to_top_options__enumvalues
try:
    nir_opt_move_to_top = _libraries['FIXME_STUB'].nir_opt_move_to_top
    nir_opt_move_to_top.restype = ctypes.c_bool
    nir_opt_move_to_top.argtypes = [ctypes.POINTER(struct_nir_shader), nir_opt_move_to_top_options]
except AttributeError:
    pass
try:
    nir_move_vec_src_uses_to_dest = _libraries['FIXME_STUB'].nir_move_vec_src_uses_to_dest
    nir_move_vec_src_uses_to_dest.restype = ctypes.c_bool
    nir_move_vec_src_uses_to_dest.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_move_output_stores_to_end = _libraries['FIXME_STUB'].nir_move_output_stores_to_end
    nir_move_output_stores_to_end.restype = ctypes.c_bool
    nir_move_output_stores_to_end.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_vec_to_regs = _libraries['FIXME_STUB'].nir_lower_vec_to_regs
    nir_lower_vec_to_regs.restype = ctypes.c_bool
    nir_lower_vec_to_regs.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_writemask_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'compare_func'
compare_func__enumvalues = {
    0: 'COMPARE_FUNC_NEVER',
    1: 'COMPARE_FUNC_LESS',
    2: 'COMPARE_FUNC_EQUAL',
    3: 'COMPARE_FUNC_LEQUAL',
    4: 'COMPARE_FUNC_GREATER',
    5: 'COMPARE_FUNC_NOTEQUAL',
    6: 'COMPARE_FUNC_GEQUAL',
    7: 'COMPARE_FUNC_ALWAYS',
}
COMPARE_FUNC_NEVER = 0
COMPARE_FUNC_LESS = 1
COMPARE_FUNC_EQUAL = 2
COMPARE_FUNC_LEQUAL = 3
COMPARE_FUNC_GREATER = 4
COMPARE_FUNC_NOTEQUAL = 5
COMPARE_FUNC_GEQUAL = 6
COMPARE_FUNC_ALWAYS = 7
compare_func = ctypes.c_uint32 # enum
try:
    nir_lower_alpha_test = _libraries['FIXME_STUB'].nir_lower_alpha_test
    nir_lower_alpha_test.restype = ctypes.c_bool
    nir_lower_alpha_test.argtypes = [ctypes.POINTER(struct_nir_shader), compare_func, ctypes.c_bool, ctypes.POINTER(ctypes.c_int16)]
except AttributeError:
    pass
try:
    nir_lower_alpha_to_coverage = _libraries['FIXME_STUB'].nir_lower_alpha_to_coverage
    nir_lower_alpha_to_coverage.restype = ctypes.c_bool
    nir_lower_alpha_to_coverage.argtypes = [ctypes.POINTER(struct_nir_shader), uint8_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_alpha_to_one = _libraries['FIXME_STUB'].nir_lower_alpha_to_one
    nir_lower_alpha_to_one.restype = ctypes.c_bool
    nir_lower_alpha_to_one.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_alu = _libraries['FIXME_STUB'].nir_lower_alu
    nir_lower_alu.restype = ctypes.c_bool
    nir_lower_alu.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_flrp = _libraries['FIXME_STUB'].nir_lower_flrp
    nir_lower_flrp.restype = ctypes.c_bool
    nir_lower_flrp.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_scale_fdiv = _libraries['FIXME_STUB'].nir_scale_fdiv
    nir_scale_fdiv.restype = ctypes.c_bool
    nir_scale_fdiv.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_alu_to_scalar = _libraries['FIXME_STUB'].nir_lower_alu_to_scalar
    nir_lower_alu_to_scalar.restype = ctypes.c_bool
    nir_lower_alu_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_alu_width = _libraries['FIXME_STUB'].nir_lower_alu_width
    nir_lower_alu_width.restype = ctypes.c_bool
    nir_lower_alu_width.argtypes = [ctypes.POINTER(struct_nir_shader), nir_vectorize_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_alu_vec8_16_srcs = _libraries['FIXME_STUB'].nir_lower_alu_vec8_16_srcs
    nir_lower_alu_vec8_16_srcs.restype = ctypes.c_bool
    nir_lower_alu_vec8_16_srcs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_bool_to_bitsize = _libraries['FIXME_STUB'].nir_lower_bool_to_bitsize
    nir_lower_bool_to_bitsize.restype = ctypes.c_bool
    nir_lower_bool_to_bitsize.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_bool_to_float = _libraries['FIXME_STUB'].nir_lower_bool_to_float
    nir_lower_bool_to_float.restype = ctypes.c_bool
    nir_lower_bool_to_float.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_bool_to_int32 = _libraries['FIXME_STUB'].nir_lower_bool_to_int32
    nir_lower_bool_to_int32.restype = ctypes.c_bool
    nir_lower_bool_to_int32.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_simplify_convert_alu_types = _libraries['FIXME_STUB'].nir_opt_simplify_convert_alu_types
    nir_opt_simplify_convert_alu_types.restype = ctypes.c_bool
    nir_opt_simplify_convert_alu_types.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_const_arrays_to_uniforms = _libraries['FIXME_STUB'].nir_lower_const_arrays_to_uniforms
    nir_lower_const_arrays_to_uniforms.restype = ctypes.c_bool
    nir_lower_const_arrays_to_uniforms.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_lower_convert_alu_types = _libraries['FIXME_STUB'].nir_lower_convert_alu_types
    nir_lower_convert_alu_types.restype = ctypes.c_bool
    nir_lower_convert_alu_types.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr))]
except AttributeError:
    pass
try:
    nir_lower_constant_convert_alu_types = _libraries['FIXME_STUB'].nir_lower_constant_convert_alu_types
    nir_lower_constant_convert_alu_types.restype = ctypes.c_bool
    nir_lower_constant_convert_alu_types.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_alu_conversion_to_intrinsic = _libraries['FIXME_STUB'].nir_lower_alu_conversion_to_intrinsic
    nir_lower_alu_conversion_to_intrinsic.restype = ctypes.c_bool
    nir_lower_alu_conversion_to_intrinsic.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_int_to_float = _libraries['FIXME_STUB'].nir_lower_int_to_float
    nir_lower_int_to_float.restype = ctypes.c_bool
    nir_lower_int_to_float.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_load_const_to_scalar = _libraries['FIXME_STUB'].nir_lower_load_const_to_scalar
    nir_lower_load_const_to_scalar.restype = ctypes.c_bool
    nir_lower_load_const_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_read_invocation_to_scalar = _libraries['FIXME_STUB'].nir_lower_read_invocation_to_scalar
    nir_lower_read_invocation_to_scalar.restype = ctypes.c_bool
    nir_lower_read_invocation_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_phis_to_scalar = _libraries['FIXME_STUB'].nir_lower_phis_to_scalar
    nir_lower_phis_to_scalar.restype = ctypes.c_bool
    nir_lower_phis_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader), nir_vectorize_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_all_phis_to_scalar = _libraries['FIXME_STUB'].nir_lower_all_phis_to_scalar
    nir_lower_all_phis_to_scalar.restype = ctypes.c_bool
    nir_lower_all_phis_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_io_array_vars_to_elements = _libraries['FIXME_STUB'].nir_lower_io_array_vars_to_elements
    nir_lower_io_array_vars_to_elements.restype = None
    nir_lower_io_array_vars_to_elements.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_io_array_vars_to_elements_no_indirects = _libraries['FIXME_STUB'].nir_lower_io_array_vars_to_elements_no_indirects
    nir_lower_io_array_vars_to_elements_no_indirects.restype = ctypes.c_bool
    nir_lower_io_array_vars_to_elements_no_indirects.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_io_to_scalar = _libraries['FIXME_STUB'].nir_lower_io_to_scalar
    nir_lower_io_to_scalar.restype = ctypes.c_bool
    nir_lower_io_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, nir_instr_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_io_vars_to_scalar = _libraries['FIXME_STUB'].nir_lower_io_vars_to_scalar
    nir_lower_io_vars_to_scalar.restype = ctypes.c_bool
    nir_lower_io_vars_to_scalar.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_opt_vectorize_io_vars = _libraries['FIXME_STUB'].nir_opt_vectorize_io_vars
    nir_opt_vectorize_io_vars.restype = ctypes.c_bool
    nir_opt_vectorize_io_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_tess_level_array_vars_to_vec = _libraries['FIXME_STUB'].nir_lower_tess_level_array_vars_to_vec
    nir_lower_tess_level_array_vars_to_vec.restype = ctypes.c_bool
    nir_lower_tess_level_array_vars_to_vec.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_create_passthrough_tcs_impl = _libraries['FIXME_STUB'].nir_create_passthrough_tcs_impl
    nir_create_passthrough_tcs_impl.restype = ctypes.POINTER(struct_nir_shader)
    nir_create_passthrough_tcs_impl.argtypes = [ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, uint8_t]
except AttributeError:
    pass
try:
    nir_create_passthrough_tcs = _libraries['FIXME_STUB'].nir_create_passthrough_tcs
    nir_create_passthrough_tcs.restype = ctypes.POINTER(struct_nir_shader)
    nir_create_passthrough_tcs.argtypes = [ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_nir_shader), uint8_t]
except AttributeError:
    pass
try:
    nir_create_passthrough_gs = _libraries['FIXME_STUB'].nir_create_passthrough_gs
    nir_create_passthrough_gs.restype = ctypes.POINTER(struct_nir_shader)
    nir_create_passthrough_gs.argtypes = [ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(struct_nir_shader), mesa_prim, mesa_prim, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_fragcolor = _libraries['FIXME_STUB'].nir_lower_fragcolor
    nir_lower_fragcolor.restype = ctypes.c_bool
    nir_lower_fragcolor.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_lower_fragcoord_wtrans = _libraries['FIXME_STUB'].nir_lower_fragcoord_wtrans
    nir_lower_fragcoord_wtrans.restype = ctypes.c_bool
    nir_lower_fragcoord_wtrans.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_frag_coord_to_pixel_coord = _libraries['FIXME_STUB'].nir_opt_frag_coord_to_pixel_coord
    nir_opt_frag_coord_to_pixel_coord.restype = ctypes.c_bool
    nir_opt_frag_coord_to_pixel_coord.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_frag_coord_to_pixel_coord = _libraries['FIXME_STUB'].nir_lower_frag_coord_to_pixel_coord
    nir_lower_frag_coord_to_pixel_coord.restype = ctypes.c_bool
    nir_lower_frag_coord_to_pixel_coord.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_viewport_transform = _libraries['FIXME_STUB'].nir_lower_viewport_transform
    nir_lower_viewport_transform.restype = ctypes.c_bool
    nir_lower_viewport_transform.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_uniforms_to_ubo = _libraries['FIXME_STUB'].nir_lower_uniforms_to_ubo
    nir_lower_uniforms_to_ubo.restype = ctypes.c_bool
    nir_lower_uniforms_to_ubo.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_is_helper_invocation = _libraries['FIXME_STUB'].nir_lower_is_helper_invocation
    nir_lower_is_helper_invocation.restype = ctypes.c_bool
    nir_lower_is_helper_invocation.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_single_sampled = _libraries['FIXME_STUB'].nir_lower_single_sampled
    nir_lower_single_sampled.restype = ctypes.c_bool
    nir_lower_single_sampled.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_sample_shading = _libraries['FIXME_STUB'].nir_lower_sample_shading
    nir_lower_sample_shading.restype = ctypes.c_bool
    nir_lower_sample_shading.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_atomics = _libraries['FIXME_STUB'].nir_lower_atomics
    nir_lower_atomics.restype = ctypes.c_bool
    nir_lower_atomics.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_filter_cb]
except AttributeError:
    pass
class struct_nir_lower_subgroups_options(Structure):
    pass

struct_nir_lower_subgroups_options._pack_ = 1 # source:False
struct_nir_lower_subgroups_options._fields_ = [
    ('filter', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('filter_data', ctypes.POINTER(None)),
    ('subgroup_size', ctypes.c_ubyte),
    ('ballot_bit_size', ctypes.c_ubyte),
    ('ballot_components', ctypes.c_ubyte),
    ('lower_to_scalar', ctypes.c_bool, 1),
    ('lower_fp64', ctypes.c_bool, 1),
    ('lower_vote_trivial', ctypes.c_bool, 1),
    ('lower_vote_feq', ctypes.c_bool, 1),
    ('lower_vote_ieq', ctypes.c_bool, 1),
    ('lower_vote_bool_eq', ctypes.c_bool, 1),
    ('lower_first_invocation_to_ballot', ctypes.c_bool, 1),
    ('lower_read_first_invocation', ctypes.c_bool, 1),
    ('lower_subgroup_masks', ctypes.c_bool, 1),
    ('lower_relative_shuffle', ctypes.c_bool, 1),
    ('lower_shuffle_to_32bit', ctypes.c_bool, 1),
    ('lower_shuffle_to_swizzle_amd', ctypes.c_bool, 1),
    ('lower_shuffle', ctypes.c_bool, 1),
    ('lower_quad', ctypes.c_bool, 1),
    ('lower_quad_broadcast_dynamic', ctypes.c_bool, 1),
    ('lower_quad_broadcast_dynamic_to_const', ctypes.c_bool, 1),
    ('lower_quad_vote', ctypes.c_bool, 1),
    ('lower_elect', ctypes.c_bool, 1),
    ('lower_read_invocation_to_cond', ctypes.c_bool, 1),
    ('lower_rotate_to_shuffle', ctypes.c_bool, 1),
    ('lower_rotate_clustered_to_shuffle', ctypes.c_bool, 1),
    ('lower_ballot_bit_count_to_mbcnt_amd', ctypes.c_bool, 1),
    ('lower_inverse_ballot', ctypes.c_bool, 1),
    ('lower_reduce', ctypes.c_bool, 1),
    ('lower_boolean_reduce', ctypes.c_bool, 1),
    ('lower_boolean_shuffle', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint16, 14),
]

nir_lower_subgroups_options = struct_nir_lower_subgroups_options
try:
    nir_lower_subgroups = _libraries['FIXME_STUB'].nir_lower_subgroups
    nir_lower_subgroups.restype = ctypes.c_bool
    nir_lower_subgroups.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_subgroups_options)]
except AttributeError:
    pass
try:
    nir_lower_system_values = _libraries['FIXME_STUB'].nir_lower_system_values
    nir_lower_system_values.restype = ctypes.c_bool
    nir_lower_system_values.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_build_lowered_load_helper_invocation = _libraries['FIXME_STUB'].nir_build_lowered_load_helper_invocation
    nir_build_lowered_load_helper_invocation.restype = ctypes.POINTER(struct_nir_def)
    nir_build_lowered_load_helper_invocation.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
class struct_nir_lower_compute_system_values_options(Structure):
    pass

struct_nir_lower_compute_system_values_options._pack_ = 1 # source:False
struct_nir_lower_compute_system_values_options._fields_ = [
    ('has_base_global_invocation_id', ctypes.c_bool, 1),
    ('has_base_workgroup_id', ctypes.c_bool, 1),
    ('has_global_size', ctypes.c_bool, 1),
    ('shuffle_local_ids_for_quad_derivatives', ctypes.c_bool, 1),
    ('lower_local_invocation_index', ctypes.c_bool, 1),
    ('lower_cs_local_id_to_index', ctypes.c_bool, 1),
    ('lower_workgroup_id_to_index', ctypes.c_bool, 1),
    ('global_id_is_32bit', ctypes.c_bool, 1),
    ('shortcut_1d_workgroup_id', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint32, 23),
    ('num_workgroups', ctypes.c_uint32 * 3),
]

nir_lower_compute_system_values_options = struct_nir_lower_compute_system_values_options
try:
    nir_lower_compute_system_values = _libraries['FIXME_STUB'].nir_lower_compute_system_values
    nir_lower_compute_system_values.restype = ctypes.c_bool
    nir_lower_compute_system_values.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_compute_system_values_options)]
except AttributeError:
    pass
class struct_nir_lower_sysvals_to_varyings_options(Structure):
    pass

struct_nir_lower_sysvals_to_varyings_options._pack_ = 1 # source:False
struct_nir_lower_sysvals_to_varyings_options._fields_ = [
    ('frag_coord', ctypes.c_bool, 1),
    ('front_face', ctypes.c_bool, 1),
    ('point_coord', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint8, 5),
]

nir_lower_sysvals_to_varyings_options = struct_nir_lower_sysvals_to_varyings_options
try:
    nir_lower_sysvals_to_varyings = _libraries['FIXME_STUB'].nir_lower_sysvals_to_varyings
    nir_lower_sysvals_to_varyings.restype = ctypes.c_bool
    nir_lower_sysvals_to_varyings.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_sysvals_to_varyings_options)]
except AttributeError:
    pass

# values for enumeration 'nir_lower_tex_packing'
nir_lower_tex_packing__enumvalues = {
    0: 'nir_lower_tex_packing_none',
    1: 'nir_lower_tex_packing_16',
    2: 'nir_lower_tex_packing_8',
}
nir_lower_tex_packing_none = 0
nir_lower_tex_packing_16 = 1
nir_lower_tex_packing_8 = 2
nir_lower_tex_packing = ctypes.c_uint32 # enum
class struct_nir_lower_tex_options(Structure):
    pass

struct_nir_lower_tex_options._pack_ = 1 # source:False
struct_nir_lower_tex_options._fields_ = [
    ('lower_txp', ctypes.c_uint32),
    ('lower_txp_array', ctypes.c_bool),
    ('lower_txf_offset', ctypes.c_bool),
    ('lower_rect_offset', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('lower_offset_filter', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('lower_rect', ctypes.c_bool),
    ('lower_1d', ctypes.c_bool),
    ('lower_1d_shadow', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte),
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
    ('swizzles', ctypes.c_ubyte * 4 * 32),
    ('scale_factors', ctypes.c_float * 32),
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
    ('lower_tex_packing_cb', ctypes.CFUNCTYPE(nir_lower_tex_packing, ctypes.POINTER(struct_nir_tex_instr), ctypes.POINTER(None))),
    ('lower_tex_packing_data', ctypes.POINTER(None)),
    ('lower_lod_zero_width', ctypes.c_bool),
    ('lower_sampler_lod_bias', ctypes.c_bool),
    ('lower_invalid_implicit_lod', ctypes.c_bool),
    ('lower_index_to_offset', ctypes.c_bool),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('callback_data', ctypes.POINTER(None)),
]

nir_lower_tex_options = struct_nir_lower_tex_options
try:
    nir_lower_tex = _libraries['FIXME_STUB'].nir_lower_tex
    nir_lower_tex.restype = ctypes.c_bool
    nir_lower_tex.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_tex_options)]
except AttributeError:
    pass
class struct_nir_lower_tex_shadow_swizzle(Structure):
    pass

struct_nir_lower_tex_shadow_swizzle._pack_ = 1 # source:False
struct_nir_lower_tex_shadow_swizzle._fields_ = [
    ('swizzle_r', ctypes.c_uint32, 3),
    ('swizzle_g', ctypes.c_uint32, 3),
    ('swizzle_b', ctypes.c_uint32, 3),
    ('swizzle_a', ctypes.c_uint32, 3),
    ('PADDING_0', ctypes.c_uint32, 20),
]

nir_lower_tex_shadow_swizzle = struct_nir_lower_tex_shadow_swizzle
try:
    nir_lower_tex_shadow = _libraries['FIXME_STUB'].nir_lower_tex_shadow
    nir_lower_tex_shadow.restype = ctypes.c_bool
    nir_lower_tex_shadow.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.POINTER(compare_func), ctypes.POINTER(struct_nir_lower_tex_shadow_swizzle), ctypes.c_bool]
except AttributeError:
    pass
class struct_nir_lower_image_options(Structure):
    pass

struct_nir_lower_image_options._pack_ = 1 # source:False
struct_nir_lower_image_options._fields_ = [
    ('lower_cube_size', ctypes.c_bool),
    ('lower_to_fragment_mask_load_amd', ctypes.c_bool),
    ('lower_image_samples_to_one', ctypes.c_bool),
]

nir_lower_image_options = struct_nir_lower_image_options
try:
    nir_lower_image = _libraries['FIXME_STUB'].nir_lower_image
    nir_lower_image.restype = ctypes.c_bool
    nir_lower_image.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_image_options)]
except AttributeError:
    pass
try:
    nir_lower_image_atomics_to_global = _libraries['FIXME_STUB'].nir_lower_image_atomics_to_global
    nir_lower_image_atomics_to_global.restype = ctypes.c_bool
    nir_lower_image_atomics_to_global.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrin_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_readonly_images_to_tex = _libraries['FIXME_STUB'].nir_lower_readonly_images_to_tex
    nir_lower_readonly_images_to_tex.restype = ctypes.c_bool
    nir_lower_readonly_images_to_tex.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass

# values for enumeration 'nir_lower_non_uniform_access_type'
nir_lower_non_uniform_access_type__enumvalues = {
    1: 'nir_lower_non_uniform_ubo_access',
    2: 'nir_lower_non_uniform_ssbo_access',
    4: 'nir_lower_non_uniform_texture_access',
    8: 'nir_lower_non_uniform_image_access',
    16: 'nir_lower_non_uniform_get_ssbo_size',
    32: 'nir_lower_non_uniform_texture_offset_access',
    6: 'nir_lower_non_uniform_access_type_count',
}
nir_lower_non_uniform_ubo_access = 1
nir_lower_non_uniform_ssbo_access = 2
nir_lower_non_uniform_texture_access = 4
nir_lower_non_uniform_image_access = 8
nir_lower_non_uniform_get_ssbo_size = 16
nir_lower_non_uniform_texture_offset_access = 32
nir_lower_non_uniform_access_type_count = 6
nir_lower_non_uniform_access_type = ctypes.c_uint32 # enum
nir_lower_non_uniform_src_access_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32, ctypes.POINTER(None))
nir_lower_non_uniform_access_callback = ctypes.CFUNCTYPE(ctypes.c_uint16, ctypes.POINTER(struct_nir_src), ctypes.POINTER(None))
class struct_nir_lower_non_uniform_access_options(Structure):
    pass

struct_nir_lower_non_uniform_access_options._pack_ = 1 # source:False
struct_nir_lower_non_uniform_access_options._fields_ = [
    ('types', nir_lower_non_uniform_access_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('tex_src_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_tex_instr), ctypes.c_uint32, ctypes.POINTER(None))),
    ('callback', ctypes.CFUNCTYPE(ctypes.c_uint16, ctypes.POINTER(struct_nir_src), ctypes.POINTER(None))),
    ('callback_data', ctypes.POINTER(None)),
]

nir_lower_non_uniform_access_options = struct_nir_lower_non_uniform_access_options
try:
    nir_has_non_uniform_access = _libraries['FIXME_STUB'].nir_has_non_uniform_access
    nir_has_non_uniform_access.restype = ctypes.c_bool
    nir_has_non_uniform_access.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_non_uniform_access_type]
except AttributeError:
    pass
try:
    nir_opt_non_uniform_access = _libraries['FIXME_STUB'].nir_opt_non_uniform_access
    nir_opt_non_uniform_access.restype = ctypes.c_bool
    nir_opt_non_uniform_access.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_non_uniform_access = _libraries['FIXME_STUB'].nir_lower_non_uniform_access
    nir_lower_non_uniform_access.restype = ctypes.c_bool
    nir_lower_non_uniform_access.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_non_uniform_access_options)]
except AttributeError:
    pass
class struct_nir_lower_idiv_options(Structure):
    pass

struct_nir_lower_idiv_options._pack_ = 1 # source:False
struct_nir_lower_idiv_options._fields_ = [
    ('allow_fp16', ctypes.c_bool),
]

nir_lower_idiv_options = struct_nir_lower_idiv_options
try:
    nir_lower_idiv = _libraries['FIXME_STUB'].nir_lower_idiv
    nir_lower_idiv.restype = ctypes.c_bool
    nir_lower_idiv.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_idiv_options)]
except AttributeError:
    pass
class struct_nir_input_attachment_options(Structure):
    pass

struct_nir_input_attachment_options._pack_ = 1 # source:False
struct_nir_input_attachment_options._fields_ = [
    ('use_ia_coord_intrin', ctypes.c_bool),
    ('use_fragcoord_sysval', ctypes.c_bool),
    ('use_layer_id_sysval', ctypes.c_bool),
    ('use_view_id_for_layer', ctypes.c_bool),
    ('unscaled_depth_stencil_ir3', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('unscaled_input_attachment_ir3', ctypes.c_uint32),
]

nir_input_attachment_options = struct_nir_input_attachment_options
try:
    nir_lower_input_attachments = _libraries['FIXME_STUB'].nir_lower_input_attachments
    nir_lower_input_attachments.restype = ctypes.c_bool
    nir_lower_input_attachments.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_input_attachment_options)]
except AttributeError:
    pass
try:
    nir_lower_clip_vs = _libraries['FIXME_STUB'].nir_lower_clip_vs
    nir_lower_clip_vs.restype = ctypes.c_bool
    nir_lower_clip_vs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, ctypes.c_int16 * 4 * 0]
except AttributeError:
    pass
try:
    nir_lower_clip_gs = _libraries['FIXME_STUB'].nir_lower_clip_gs
    nir_lower_clip_gs.restype = ctypes.c_bool
    nir_lower_clip_gs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_int16 * 4 * 0]
except AttributeError:
    pass
try:
    nir_lower_clip_fs = _libraries['FIXME_STUB'].nir_lower_clip_fs
    nir_lower_clip_fs.restype = ctypes.c_bool
    nir_lower_clip_fs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_clip_cull_distance_to_vec4s = _libraries['FIXME_STUB'].nir_lower_clip_cull_distance_to_vec4s
    nir_lower_clip_cull_distance_to_vec4s.restype = ctypes.c_bool
    nir_lower_clip_cull_distance_to_vec4s.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_clip_cull_distance_array_vars = _libraries['FIXME_STUB'].nir_lower_clip_cull_distance_array_vars
    nir_lower_clip_cull_distance_array_vars.restype = ctypes.c_bool
    nir_lower_clip_cull_distance_array_vars.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_clip_disable = _libraries['FIXME_STUB'].nir_lower_clip_disable
    nir_lower_clip_disable.restype = ctypes.c_bool
    nir_lower_clip_disable.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_lower_point_size_mov = _libraries['FIXME_STUB'].nir_lower_point_size_mov
    nir_lower_point_size_mov.restype = ctypes.c_bool
    nir_lower_point_size_mov.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(ctypes.c_int16)]
except AttributeError:
    pass
try:
    nir_lower_frexp = _libraries['FIXME_STUB'].nir_lower_frexp
    nir_lower_frexp.restype = ctypes.c_bool
    nir_lower_frexp.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_two_sided_color = _libraries['FIXME_STUB'].nir_lower_two_sided_color
    nir_lower_two_sided_color.restype = ctypes.c_bool
    nir_lower_two_sided_color.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_clamp_color_outputs = _libraries['FIXME_STUB'].nir_lower_clamp_color_outputs
    nir_lower_clamp_color_outputs.restype = ctypes.c_bool
    nir_lower_clamp_color_outputs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_flatshade = _libraries['FIXME_STUB'].nir_lower_flatshade
    nir_lower_flatshade.restype = ctypes.c_bool
    nir_lower_flatshade.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_passthrough_edgeflags = _libraries['FIXME_STUB'].nir_lower_passthrough_edgeflags
    nir_lower_passthrough_edgeflags.restype = ctypes.c_bool
    nir_lower_passthrough_edgeflags.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_patch_vertices = _libraries['FIXME_STUB'].nir_lower_patch_vertices
    nir_lower_patch_vertices.restype = ctypes.c_bool
    nir_lower_patch_vertices.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.POINTER(ctypes.c_int16)]
except AttributeError:
    pass
class struct_nir_lower_wpos_ytransform_options(Structure):
    pass

struct_nir_lower_wpos_ytransform_options._pack_ = 1 # source:False
struct_nir_lower_wpos_ytransform_options._fields_ = [
    ('state_tokens', ctypes.c_int16 * 4),
    ('fs_coord_origin_upper_left', ctypes.c_bool, 1),
    ('fs_coord_origin_lower_left', ctypes.c_bool, 1),
    ('fs_coord_pixel_center_integer', ctypes.c_bool, 1),
    ('fs_coord_pixel_center_half_integer', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint16, 12),
]

nir_lower_wpos_ytransform_options = struct_nir_lower_wpos_ytransform_options
try:
    nir_lower_wpos_ytransform = _libraries['FIXME_STUB'].nir_lower_wpos_ytransform
    nir_lower_wpos_ytransform.restype = ctypes.c_bool
    nir_lower_wpos_ytransform.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_wpos_ytransform_options)]
except AttributeError:
    pass
try:
    nir_lower_wpos_center = _libraries['FIXME_STUB'].nir_lower_wpos_center
    nir_lower_wpos_center.restype = ctypes.c_bool
    nir_lower_wpos_center.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_pntc_ytransform = _libraries['FIXME_STUB'].nir_lower_pntc_ytransform
    nir_lower_pntc_ytransform.restype = ctypes.c_bool
    nir_lower_pntc_ytransform.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_int16 * 4 * 0]
except AttributeError:
    pass
try:
    nir_lower_wrmasks = _libraries['FIXME_STUB'].nir_lower_wrmasks
    nir_lower_wrmasks.restype = ctypes.c_bool
    nir_lower_wrmasks.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_filter_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_fb_read = _libraries['FIXME_STUB'].nir_lower_fb_read
    nir_lower_fb_read.restype = ctypes.c_bool
    nir_lower_fb_read.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_lower_drawpixels_options(Structure):
    pass

struct_nir_lower_drawpixels_options._pack_ = 1 # source:False
struct_nir_lower_drawpixels_options._fields_ = [
    ('texcoord_state_tokens', ctypes.c_int16 * 4),
    ('scale_state_tokens', ctypes.c_int16 * 4),
    ('bias_state_tokens', ctypes.c_int16 * 4),
    ('drawpix_sampler', ctypes.c_uint32),
    ('pixelmap_sampler', ctypes.c_uint32),
    ('pixel_maps', ctypes.c_bool, 1),
    ('scale_and_bias', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint32, 30),
]

nir_lower_drawpixels_options = struct_nir_lower_drawpixels_options
try:
    nir_lower_drawpixels = _libraries['FIXME_STUB'].nir_lower_drawpixels
    nir_lower_drawpixels.restype = ctypes.c_bool
    nir_lower_drawpixels.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_drawpixels_options)]
except AttributeError:
    pass
class struct_nir_lower_bitmap_options(Structure):
    pass

struct_nir_lower_bitmap_options._pack_ = 1 # source:False
struct_nir_lower_bitmap_options._fields_ = [
    ('sampler', ctypes.c_uint32),
    ('swizzle_xxxx', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

nir_lower_bitmap_options = struct_nir_lower_bitmap_options
try:
    nir_lower_bitmap = _libraries['FIXME_STUB'].nir_lower_bitmap
    nir_lower_bitmap.restype = ctypes.c_bool
    nir_lower_bitmap.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_bitmap_options)]
except AttributeError:
    pass
try:
    nir_lower_atomics_to_ssbo = _libraries['FIXME_STUB'].nir_lower_atomics_to_ssbo
    nir_lower_atomics_to_ssbo.restype = ctypes.c_bool
    nir_lower_atomics_to_ssbo.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_gs_intrinsics_flags'
c__EA_nir_lower_gs_intrinsics_flags__enumvalues = {
    1: 'nir_lower_gs_intrinsics_per_stream',
    2: 'nir_lower_gs_intrinsics_count_primitives',
    4: 'nir_lower_gs_intrinsics_count_vertices_per_primitive',
    8: 'nir_lower_gs_intrinsics_overwrite_incomplete',
}
nir_lower_gs_intrinsics_per_stream = 1
nir_lower_gs_intrinsics_count_primitives = 2
nir_lower_gs_intrinsics_count_vertices_per_primitive = 4
nir_lower_gs_intrinsics_overwrite_incomplete = 8
c__EA_nir_lower_gs_intrinsics_flags = ctypes.c_uint32 # enum
nir_lower_gs_intrinsics_flags = c__EA_nir_lower_gs_intrinsics_flags
nir_lower_gs_intrinsics_flags__enumvalues = c__EA_nir_lower_gs_intrinsics_flags__enumvalues
try:
    nir_lower_gs_intrinsics = _libraries['FIXME_STUB'].nir_lower_gs_intrinsics
    nir_lower_gs_intrinsics.restype = ctypes.c_bool
    nir_lower_gs_intrinsics.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_gs_intrinsics_flags]
except AttributeError:
    pass
try:
    nir_lower_halt_to_return = _libraries['FIXME_STUB'].nir_lower_halt_to_return
    nir_lower_halt_to_return.restype = ctypes.c_bool
    nir_lower_halt_to_return.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_tess_coord_z = _libraries['FIXME_STUB'].nir_lower_tess_coord_z
    nir_lower_tess_coord_z.restype = ctypes.c_bool
    nir_lower_tess_coord_z.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
class struct_nir_lower_task_shader_options(Structure):
    pass

struct_nir_lower_task_shader_options._pack_ = 1 # source:False
struct_nir_lower_task_shader_options._fields_ = [
    ('payload_to_shared_for_atomics', ctypes.c_bool, 1),
    ('payload_to_shared_for_small_types', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint32, 30),
    ('payload_offset_in_bytes', ctypes.c_uint32),
]

nir_lower_task_shader_options = struct_nir_lower_task_shader_options
try:
    nir_lower_task_shader = _libraries['FIXME_STUB'].nir_lower_task_shader
    nir_lower_task_shader.restype = ctypes.c_bool
    nir_lower_task_shader.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_task_shader_options]
except AttributeError:
    pass
nir_lower_bit_size_callback = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
try:
    nir_lower_bit_size = _libraries['FIXME_STUB'].nir_lower_bit_size
    nir_lower_bit_size.restype = ctypes.c_bool
    nir_lower_bit_size.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_bit_size_callback, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_lower_64bit_phis = _libraries['FIXME_STUB'].nir_lower_64bit_phis
    nir_lower_64bit_phis.restype = ctypes.c_bool
    nir_lower_64bit_phis.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_split_conversions_options(Structure):
    pass

struct_nir_split_conversions_options._pack_ = 1 # source:False
struct_nir_split_conversions_options._fields_ = [
    ('callback', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('callback_data', ctypes.POINTER(None)),
    ('has_convert_alu_types', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

nir_split_conversions_options = struct_nir_split_conversions_options
try:
    nir_split_conversions = _libraries['FIXME_STUB'].nir_split_conversions
    nir_split_conversions.restype = ctypes.c_bool
    nir_split_conversions.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_split_conversions_options)]
except AttributeError:
    pass
try:
    nir_split_64bit_vec3_and_vec4 = _libraries['FIXME_STUB'].nir_split_64bit_vec3_and_vec4
    nir_split_64bit_vec3_and_vec4.restype = ctypes.c_bool
    nir_split_64bit_vec3_and_vec4.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_int64_op_to_options_mask = _libraries['FIXME_STUB'].nir_lower_int64_op_to_options_mask
    nir_lower_int64_op_to_options_mask.restype = nir_lower_int64_options
    nir_lower_int64_op_to_options_mask.argtypes = [nir_op]
except AttributeError:
    pass
try:
    nir_lower_int64 = _libraries['FIXME_STUB'].nir_lower_int64
    nir_lower_int64.restype = ctypes.c_bool
    nir_lower_int64.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_int64_float_conversions = _libraries['FIXME_STUB'].nir_lower_int64_float_conversions
    nir_lower_int64_float_conversions.restype = ctypes.c_bool
    nir_lower_int64_float_conversions.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_doubles_op_to_options_mask = _libraries['FIXME_STUB'].nir_lower_doubles_op_to_options_mask
    nir_lower_doubles_op_to_options_mask.restype = nir_lower_doubles_options
    nir_lower_doubles_op_to_options_mask.argtypes = [nir_op]
except AttributeError:
    pass
try:
    nir_lower_doubles = _libraries['FIXME_STUB'].nir_lower_doubles
    nir_lower_doubles.restype = ctypes.c_bool
    nir_lower_doubles.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_shader), nir_lower_doubles_options]
except AttributeError:
    pass
try:
    nir_lower_pack = _libraries['FIXME_STUB'].nir_lower_pack
    nir_lower_pack.restype = ctypes.c_bool
    nir_lower_pack.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_get_io_intrinsic = _libraries['FIXME_STUB'].nir_get_io_intrinsic
    nir_get_io_intrinsic.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_get_io_intrinsic.argtypes = [ctypes.POINTER(struct_nir_instr), nir_variable_mode, ctypes.POINTER(c__EA_nir_variable_mode)]
except AttributeError:
    pass
try:
    nir_recompute_io_bases = _libraries['FIXME_STUB'].nir_recompute_io_bases
    nir_recompute_io_bases.restype = ctypes.c_bool
    nir_recompute_io_bases.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_mediump_vars = _libraries['FIXME_STUB'].nir_lower_mediump_vars
    nir_lower_mediump_vars.restype = ctypes.c_bool
    nir_lower_mediump_vars.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_lower_mediump_io = _libraries['FIXME_STUB'].nir_lower_mediump_io
    nir_lower_mediump_io.restype = ctypes.c_bool
    nir_lower_mediump_io.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, uint64_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_clear_mediump_io_flag = _libraries['FIXME_STUB'].nir_clear_mediump_io_flag
    nir_clear_mediump_io_flag.restype = ctypes.c_bool
    nir_clear_mediump_io_flag.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_opt_tex_srcs_options(Structure):
    pass

struct_nir_opt_tex_srcs_options._pack_ = 1 # source:False
struct_nir_opt_tex_srcs_options._fields_ = [
    ('sampler_dims', ctypes.c_uint32),
    ('src_types', ctypes.c_uint32),
]

nir_opt_tex_srcs_options = struct_nir_opt_tex_srcs_options
class struct_nir_opt_16bit_tex_image_options(Structure):
    pass

struct_nir_opt_16bit_tex_image_options._pack_ = 1 # source:False
struct_nir_opt_16bit_tex_image_options._fields_ = [
    ('rounding_mode', nir_rounding_mode),
    ('opt_tex_dest_types', nir_alu_type),
    ('opt_image_dest_types', nir_alu_type),
    ('integer_dest_saturates', ctypes.c_bool),
    ('opt_image_store_data', ctypes.c_bool),
    ('opt_image_srcs', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('opt_srcs_options_count', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('opt_srcs_options', ctypes.POINTER(struct_nir_opt_tex_srcs_options)),
]

nir_opt_16bit_tex_image_options = struct_nir_opt_16bit_tex_image_options
try:
    nir_opt_16bit_tex_image = _libraries['FIXME_STUB'].nir_opt_16bit_tex_image
    nir_opt_16bit_tex_image.restype = ctypes.c_bool
    nir_opt_16bit_tex_image.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_16bit_tex_image_options)]
except AttributeError:
    pass
class struct_nir_tex_src_type_constraint(Structure):
    pass

struct_nir_tex_src_type_constraint._pack_ = 1 # source:False
struct_nir_tex_src_type_constraint._fields_ = [
    ('legalize_type', ctypes.c_bool),
    ('bit_size', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('match_src', nir_tex_src_type),
]

nir_tex_src_type_constraint = struct_nir_tex_src_type_constraint
nir_tex_src_type_constraints = struct_nir_tex_src_type_constraint * 23
try:
    nir_legalize_16bit_sampler_srcs = _libraries['FIXME_STUB'].nir_legalize_16bit_sampler_srcs
    nir_legalize_16bit_sampler_srcs.restype = ctypes.c_bool
    nir_legalize_16bit_sampler_srcs.argtypes = [ctypes.POINTER(struct_nir_shader), nir_tex_src_type_constraints]
except AttributeError:
    pass
try:
    nir_lower_point_size = _libraries['FIXME_STUB'].nir_lower_point_size
    nir_lower_point_size.restype = ctypes.c_bool
    nir_lower_point_size.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_lower_default_point_size = _libraries['FIXME_STUB'].nir_lower_default_point_size
    nir_lower_default_point_size.restype = ctypes.c_bool
    nir_lower_default_point_size.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_texcoord_replace = _libraries['FIXME_STUB'].nir_lower_texcoord_replace
    nir_lower_texcoord_replace.restype = ctypes.c_bool
    nir_lower_texcoord_replace.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_texcoord_replace_late = _libraries['FIXME_STUB'].nir_lower_texcoord_replace_late
    nir_lower_texcoord_replace_late.restype = ctypes.c_bool
    nir_lower_texcoord_replace_late.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32, ctypes.c_bool]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_interpolation_options'
c__EA_nir_lower_interpolation_options__enumvalues = {
    2: 'nir_lower_interpolation_at_sample',
    4: 'nir_lower_interpolation_at_offset',
    8: 'nir_lower_interpolation_centroid',
    16: 'nir_lower_interpolation_pixel',
    32: 'nir_lower_interpolation_sample',
}
nir_lower_interpolation_at_sample = 2
nir_lower_interpolation_at_offset = 4
nir_lower_interpolation_centroid = 8
nir_lower_interpolation_pixel = 16
nir_lower_interpolation_sample = 32
c__EA_nir_lower_interpolation_options = ctypes.c_uint32 # enum
nir_lower_interpolation_options = c__EA_nir_lower_interpolation_options
nir_lower_interpolation_options__enumvalues = c__EA_nir_lower_interpolation_options__enumvalues
try:
    nir_lower_interpolation = _libraries['FIXME_STUB'].nir_lower_interpolation
    nir_lower_interpolation.restype = ctypes.c_bool
    nir_lower_interpolation.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_interpolation_options]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_discard_if_options'
c__EA_nir_lower_discard_if_options__enumvalues = {
    1: 'nir_lower_demote_if_to_cf',
    2: 'nir_lower_terminate_if_to_cf',
    4: 'nir_move_terminate_out_of_loops',
}
nir_lower_demote_if_to_cf = 1
nir_lower_terminate_if_to_cf = 2
nir_move_terminate_out_of_loops = 4
c__EA_nir_lower_discard_if_options = ctypes.c_uint32 # enum
nir_lower_discard_if_options = c__EA_nir_lower_discard_if_options
nir_lower_discard_if_options__enumvalues = c__EA_nir_lower_discard_if_options__enumvalues
try:
    nir_lower_discard_if = _libraries['FIXME_STUB'].nir_lower_discard_if
    nir_lower_discard_if.restype = ctypes.c_bool
    nir_lower_discard_if.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_discard_if_options]
except AttributeError:
    pass
try:
    nir_lower_terminate_to_demote = _libraries['FIXME_STUB'].nir_lower_terminate_to_demote
    nir_lower_terminate_to_demote.restype = ctypes.c_bool
    nir_lower_terminate_to_demote.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_memory_model = _libraries['FIXME_STUB'].nir_lower_memory_model
    nir_lower_memory_model.restype = ctypes.c_bool
    nir_lower_memory_model.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_goto_ifs = _libraries['FIXME_STUB'].nir_lower_goto_ifs
    nir_lower_goto_ifs.restype = ctypes.c_bool
    nir_lower_goto_ifs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_continue_constructs = _libraries['FIXME_STUB'].nir_lower_continue_constructs
    nir_lower_continue_constructs.restype = ctypes.c_bool
    nir_lower_continue_constructs.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_lower_multiview_options(Structure):
    pass

struct_nir_lower_multiview_options._pack_ = 1 # source:False
struct_nir_lower_multiview_options._fields_ = [
    ('view_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('allowed_per_view_outputs', ctypes.c_uint64),
]

nir_lower_multiview_options = struct_nir_lower_multiview_options
try:
    nir_shader_uses_view_index = _libraries['FIXME_STUB'].nir_shader_uses_view_index
    nir_shader_uses_view_index.restype = ctypes.c_bool
    nir_shader_uses_view_index.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_can_lower_multiview = _libraries['FIXME_STUB'].nir_can_lower_multiview
    nir_can_lower_multiview.restype = ctypes.c_bool
    nir_can_lower_multiview.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_multiview_options]
except AttributeError:
    pass
try:
    nir_lower_multiview = _libraries['FIXME_STUB'].nir_lower_multiview
    nir_lower_multiview.restype = ctypes.c_bool
    nir_lower_multiview.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_multiview_options]
except AttributeError:
    pass
try:
    nir_lower_view_index_to_device_index = _libraries['FIXME_STUB'].nir_lower_view_index_to_device_index
    nir_lower_view_index_to_device_index.restype = ctypes.c_bool
    nir_lower_view_index_to_device_index.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_lower_fp16_cast_options'
c__EA_nir_lower_fp16_cast_options__enumvalues = {
    1: 'nir_lower_fp16_rtz',
    2: 'nir_lower_fp16_rtne',
    4: 'nir_lower_fp16_ru',
    8: 'nir_lower_fp16_rd',
    15: 'nir_lower_fp16_all',
    16: 'nir_lower_fp16_split_fp64',
}
nir_lower_fp16_rtz = 1
nir_lower_fp16_rtne = 2
nir_lower_fp16_ru = 4
nir_lower_fp16_rd = 8
nir_lower_fp16_all = 15
nir_lower_fp16_split_fp64 = 16
c__EA_nir_lower_fp16_cast_options = ctypes.c_uint32 # enum
nir_lower_fp16_cast_options = c__EA_nir_lower_fp16_cast_options
nir_lower_fp16_cast_options__enumvalues = c__EA_nir_lower_fp16_cast_options__enumvalues
try:
    nir_lower_fp16_casts = _libraries['FIXME_STUB'].nir_lower_fp16_casts
    nir_lower_fp16_casts.restype = ctypes.c_bool
    nir_lower_fp16_casts.argtypes = [ctypes.POINTER(struct_nir_shader), nir_lower_fp16_cast_options]
except AttributeError:
    pass
try:
    nir_normalize_cubemap_coords = _libraries['FIXME_STUB'].nir_normalize_cubemap_coords
    nir_normalize_cubemap_coords.restype = ctypes.c_bool
    nir_normalize_cubemap_coords.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_shader_supports_implicit_lod = _libraries['FIXME_STUB'].nir_shader_supports_implicit_lod
    nir_shader_supports_implicit_lod.restype = ctypes.c_bool
    nir_shader_supports_implicit_lod.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_live_defs_impl = _libraries['FIXME_STUB'].nir_live_defs_impl
    nir_live_defs_impl.restype = None
    nir_live_defs_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_get_live_defs = _libraries['FIXME_STUB'].nir_get_live_defs
    nir_get_live_defs.restype = ctypes.POINTER(ctypes.c_uint32)
    nir_get_live_defs.argtypes = [nir_cursor, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_loop_analyze_impl = _libraries['FIXME_STUB'].nir_loop_analyze_impl
    nir_loop_analyze_impl.restype = None
    nir_loop_analyze_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_variable_mode, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_defs_interfere = _libraries['FIXME_STUB'].nir_defs_interfere
    nir_defs_interfere.restype = ctypes.c_bool
    nir_defs_interfere.argtypes = [ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_repair_ssa_impl = _libraries['FIXME_STUB'].nir_repair_ssa_impl
    nir_repair_ssa_impl.restype = ctypes.c_bool
    nir_repair_ssa_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_repair_ssa = _libraries['FIXME_STUB'].nir_repair_ssa
    nir_repair_ssa.restype = ctypes.c_bool
    nir_repair_ssa.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_convert_loop_to_lcssa = _libraries['FIXME_STUB'].nir_convert_loop_to_lcssa
    nir_convert_loop_to_lcssa.restype = None
    nir_convert_loop_to_lcssa.argtypes = [ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_convert_to_lcssa = _libraries['FIXME_STUB'].nir_convert_to_lcssa
    nir_convert_to_lcssa.restype = ctypes.c_bool
    nir_convert_to_lcssa.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_divergence_analysis_impl = _libraries['FIXME_STUB'].nir_divergence_analysis_impl
    nir_divergence_analysis_impl.restype = None
    nir_divergence_analysis_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_divergence_options]
except AttributeError:
    pass
try:
    nir_divergence_analysis = _libraries['FIXME_STUB'].nir_divergence_analysis
    nir_divergence_analysis.restype = None
    nir_divergence_analysis.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_vertex_divergence_analysis = _libraries['FIXME_STUB'].nir_vertex_divergence_analysis
    nir_vertex_divergence_analysis.restype = None
    nir_vertex_divergence_analysis.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_has_divergent_loop = _libraries['FIXME_STUB'].nir_has_divergent_loop
    nir_has_divergent_loop.restype = ctypes.c_bool
    nir_has_divergent_loop.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_rewrite_uses_to_load_reg = _libraries['FIXME_STUB'].nir_rewrite_uses_to_load_reg
    nir_rewrite_uses_to_load_reg.restype = None
    nir_rewrite_uses_to_load_reg.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_convert_from_ssa = _libraries['FIXME_STUB'].nir_convert_from_ssa
    nir_convert_from_ssa.restype = ctypes.c_bool
    nir_convert_from_ssa.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_phis_to_regs_block = _libraries['FIXME_STUB'].nir_lower_phis_to_regs_block
    nir_lower_phis_to_regs_block.restype = ctypes.c_bool
    nir_lower_phis_to_regs_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_lower_ssa_defs_to_regs_block = _libraries['FIXME_STUB'].nir_lower_ssa_defs_to_regs_block
    nir_lower_ssa_defs_to_regs_block.restype = ctypes.c_bool
    nir_lower_ssa_defs_to_regs_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_rematerialize_deref_in_use_blocks = _libraries['FIXME_STUB'].nir_rematerialize_deref_in_use_blocks
    nir_rematerialize_deref_in_use_blocks.restype = ctypes.c_bool
    nir_rematerialize_deref_in_use_blocks.argtypes = [ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_rematerialize_derefs_in_use_blocks_impl = _libraries['FIXME_STUB'].nir_rematerialize_derefs_in_use_blocks_impl
    nir_rematerialize_derefs_in_use_blocks_impl.restype = ctypes.c_bool
    nir_rematerialize_derefs_in_use_blocks_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_lower_samplers = _libraries['FIXME_STUB'].nir_lower_samplers
    nir_lower_samplers.restype = ctypes.c_bool
    nir_lower_samplers.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_cl_images = _libraries['FIXME_STUB'].nir_lower_cl_images
    nir_lower_cl_images.restype = ctypes.c_bool
    nir_lower_cl_images.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_dedup_inline_samplers = _libraries['FIXME_STUB'].nir_dedup_inline_samplers
    nir_dedup_inline_samplers.restype = ctypes.c_bool
    nir_dedup_inline_samplers.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_lower_ssbo_options(Structure):
    pass

struct_nir_lower_ssbo_options._pack_ = 1 # source:False
struct_nir_lower_ssbo_options._fields_ = [
    ('native_loads', ctypes.c_bool),
    ('native_offset', ctypes.c_bool),
]

nir_lower_ssbo_options = struct_nir_lower_ssbo_options
try:
    nir_lower_ssbo = _libraries['FIXME_STUB'].nir_lower_ssbo
    nir_lower_ssbo.restype = ctypes.c_bool
    nir_lower_ssbo.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_ssbo_options)]
except AttributeError:
    pass
try:
    nir_lower_helper_writes = _libraries['FIXME_STUB'].nir_lower_helper_writes
    nir_lower_helper_writes.restype = ctypes.c_bool
    nir_lower_helper_writes.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
class struct_nir_lower_printf_options(Structure):
    pass

struct_nir_lower_printf_options._pack_ = 1 # source:False
struct_nir_lower_printf_options._fields_ = [
    ('max_buffer_size', ctypes.c_uint32),
    ('ptr_bit_size', ctypes.c_uint32),
    ('hash_format_strings', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

nir_lower_printf_options = struct_nir_lower_printf_options
try:
    nir_lower_printf = _libraries['FIXME_STUB'].nir_lower_printf
    nir_lower_printf.restype = ctypes.c_bool
    nir_lower_printf.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_printf_options)]
except AttributeError:
    pass
try:
    nir_opt_comparison_pre_impl = _libraries['FIXME_STUB'].nir_opt_comparison_pre_impl
    nir_opt_comparison_pre_impl.restype = ctypes.c_bool
    nir_opt_comparison_pre_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_opt_comparison_pre = _libraries['FIXME_STUB'].nir_opt_comparison_pre
    nir_opt_comparison_pre.restype = ctypes.c_bool
    nir_opt_comparison_pre.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_nir_opt_access_options(Structure):
    pass

struct_nir_opt_access_options._pack_ = 1 # source:False
struct_nir_opt_access_options._fields_ = [
    ('is_vulkan', ctypes.c_bool),
]

nir_opt_access_options = struct_nir_opt_access_options
try:
    nir_opt_access = _libraries['FIXME_STUB'].nir_opt_access
    nir_opt_access.restype = ctypes.c_bool
    nir_opt_access.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_access_options)]
except AttributeError:
    pass
try:
    nir_opt_algebraic = _libraries['FIXME_STUB'].nir_opt_algebraic
    nir_opt_algebraic.restype = ctypes.c_bool
    nir_opt_algebraic.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_before_ffma = _libraries['FIXME_STUB'].nir_opt_algebraic_before_ffma
    nir_opt_algebraic_before_ffma.restype = ctypes.c_bool
    nir_opt_algebraic_before_ffma.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_before_lower_int64 = _libraries['FIXME_STUB'].nir_opt_algebraic_before_lower_int64
    nir_opt_algebraic_before_lower_int64.restype = ctypes.c_bool
    nir_opt_algebraic_before_lower_int64.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_late = _libraries['FIXME_STUB'].nir_opt_algebraic_late
    nir_opt_algebraic_late.restype = ctypes.c_bool
    nir_opt_algebraic_late.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_distribute_src_mods = _libraries['FIXME_STUB'].nir_opt_algebraic_distribute_src_mods
    nir_opt_algebraic_distribute_src_mods.restype = ctypes.c_bool
    nir_opt_algebraic_distribute_src_mods.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_algebraic_integer_promotion = _libraries['FIXME_STUB'].nir_opt_algebraic_integer_promotion
    nir_opt_algebraic_integer_promotion.restype = ctypes.c_bool
    nir_opt_algebraic_integer_promotion.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_reassociate_matrix_mul = _libraries['FIXME_STUB'].nir_opt_reassociate_matrix_mul
    nir_opt_reassociate_matrix_mul.restype = ctypes.c_bool
    nir_opt_reassociate_matrix_mul.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_constant_folding = _libraries['FIXME_STUB'].nir_opt_constant_folding
    nir_opt_constant_folding.restype = ctypes.c_bool
    nir_opt_constant_folding.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
nir_combine_barrier_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
try:
    nir_opt_combine_barriers = _libraries['FIXME_STUB'].nir_opt_combine_barriers
    nir_opt_combine_barriers.restype = ctypes.c_bool
    nir_opt_combine_barriers.argtypes = [ctypes.POINTER(struct_nir_shader), nir_combine_barrier_cb, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'c__EA_mesa_scope'
c__EA_mesa_scope__enumvalues = {
    0: 'SCOPE_NONE',
    1: 'SCOPE_INVOCATION',
    2: 'SCOPE_SUBGROUP',
    3: 'SCOPE_SHADER_CALL',
    4: 'SCOPE_WORKGROUP',
    5: 'SCOPE_QUEUE_FAMILY',
    6: 'SCOPE_DEVICE',
}
SCOPE_NONE = 0
SCOPE_INVOCATION = 1
SCOPE_SUBGROUP = 2
SCOPE_SHADER_CALL = 3
SCOPE_WORKGROUP = 4
SCOPE_QUEUE_FAMILY = 5
SCOPE_DEVICE = 6
c__EA_mesa_scope = ctypes.c_uint32 # enum
mesa_scope = c__EA_mesa_scope
mesa_scope__enumvalues = c__EA_mesa_scope__enumvalues
try:
    nir_opt_acquire_release_barriers = _libraries['FIXME_STUB'].nir_opt_acquire_release_barriers
    nir_opt_acquire_release_barriers.restype = ctypes.c_bool
    nir_opt_acquire_release_barriers.argtypes = [ctypes.POINTER(struct_nir_shader), mesa_scope]
except AttributeError:
    pass
try:
    nir_opt_barrier_modes = _libraries['FIXME_STUB'].nir_opt_barrier_modes
    nir_opt_barrier_modes.restype = ctypes.c_bool
    nir_opt_barrier_modes.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_minimize_call_live_states = _libraries['FIXME_STUB'].nir_minimize_call_live_states
    nir_minimize_call_live_states.restype = ctypes.c_bool
    nir_minimize_call_live_states.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_combine_stores = _libraries['FIXME_STUB'].nir_opt_combine_stores
    nir_opt_combine_stores.restype = ctypes.c_bool
    nir_opt_combine_stores.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode]
except AttributeError:
    pass
try:
    nir_copy_prop_impl = _libraries['FIXME_STUB'].nir_copy_prop_impl
    nir_copy_prop_impl.restype = ctypes.c_bool
    nir_copy_prop_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_copy_prop = _libraries['FIXME_STUB'].nir_copy_prop
    nir_copy_prop.restype = ctypes.c_bool
    nir_copy_prop.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_copy_prop_vars = _libraries['FIXME_STUB'].nir_opt_copy_prop_vars
    nir_opt_copy_prop_vars.restype = ctypes.c_bool
    nir_opt_copy_prop_vars.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_cse = _libraries['FIXME_STUB'].nir_opt_cse
    nir_opt_cse.restype = ctypes.c_bool
    nir_opt_cse.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_dce = _libraries['FIXME_STUB'].nir_opt_dce
    nir_opt_dce.restype = ctypes.c_bool
    nir_opt_dce.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_dead_cf = _libraries['FIXME_STUB'].nir_opt_dead_cf
    nir_opt_dead_cf.restype = ctypes.c_bool
    nir_opt_dead_cf.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_dead_write_vars = _libraries['FIXME_STUB'].nir_opt_dead_write_vars
    nir_opt_dead_write_vars.restype = ctypes.c_bool
    nir_opt_dead_write_vars.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_deref_impl = _libraries['FIXME_STUB'].nir_opt_deref_impl
    nir_opt_deref_impl.restype = ctypes.c_bool
    nir_opt_deref_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_opt_deref = _libraries['FIXME_STUB'].nir_opt_deref
    nir_opt_deref.restype = ctypes.c_bool
    nir_opt_deref.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_find_array_copies = _libraries['FIXME_STUB'].nir_opt_find_array_copies
    nir_opt_find_array_copies.restype = ctypes.c_bool
    nir_opt_find_array_copies.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_def_is_frag_coord_z = _libraries['FIXME_STUB'].nir_def_is_frag_coord_z
    nir_def_is_frag_coord_z.restype = ctypes.c_bool
    nir_def_is_frag_coord_z.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_opt_fragdepth = _libraries['FIXME_STUB'].nir_opt_fragdepth
    nir_opt_fragdepth.restype = ctypes.c_bool
    nir_opt_fragdepth.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_gcm = _libraries['FIXME_STUB'].nir_opt_gcm
    nir_opt_gcm.restype = ctypes.c_bool
    nir_opt_gcm.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_generate_bfi = _libraries['FIXME_STUB'].nir_opt_generate_bfi
    nir_opt_generate_bfi.restype = ctypes.c_bool
    nir_opt_generate_bfi.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_idiv_const = _libraries['FIXME_STUB'].nir_opt_idiv_const
    nir_opt_idiv_const.restype = ctypes.c_bool
    nir_opt_idiv_const.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_opt_mqsad = _libraries['FIXME_STUB'].nir_opt_mqsad
    nir_opt_mqsad.restype = ctypes.c_bool
    nir_opt_mqsad.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_opt_if_options'
c__EA_nir_opt_if_options__enumvalues = {
    1: 'nir_opt_if_optimize_phi_true_false',
    2: 'nir_opt_if_avoid_64bit_phis',
}
nir_opt_if_optimize_phi_true_false = 1
nir_opt_if_avoid_64bit_phis = 2
c__EA_nir_opt_if_options = ctypes.c_uint32 # enum
nir_opt_if_options = c__EA_nir_opt_if_options
nir_opt_if_options__enumvalues = c__EA_nir_opt_if_options__enumvalues
try:
    nir_opt_if = _libraries['FIXME_STUB'].nir_opt_if
    nir_opt_if.restype = ctypes.c_bool
    nir_opt_if.argtypes = [ctypes.POINTER(struct_nir_shader), nir_opt_if_options]
except AttributeError:
    pass
try:
    nir_opt_intrinsics = _libraries['FIXME_STUB'].nir_opt_intrinsics
    nir_opt_intrinsics.restype = ctypes.c_bool
    nir_opt_intrinsics.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_large_constants = _libraries['FIXME_STUB'].nir_opt_large_constants
    nir_opt_large_constants.restype = ctypes.c_bool
    nir_opt_large_constants.argtypes = [ctypes.POINTER(struct_nir_shader), glsl_type_size_align_func, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_opt_licm = _libraries['FIXME_STUB'].nir_opt_licm
    nir_opt_licm.restype = ctypes.c_bool
    nir_opt_licm.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_loop = _libraries['FIXME_STUB'].nir_opt_loop
    nir_opt_loop.restype = ctypes.c_bool
    nir_opt_loop.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_loop_unroll = _libraries['FIXME_STUB'].nir_opt_loop_unroll
    nir_opt_loop_unroll.restype = ctypes.c_bool
    nir_opt_loop_unroll.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_move_options'
c__EA_nir_move_options__enumvalues = {
    1: 'nir_move_const_undef',
    2: 'nir_move_alu',
    4: 'nir_move_copies',
    8: 'nir_move_comparisons',
    16: 'nir_dont_move_byte_word_vecs',
    256: 'nir_move_tex_sample',
    512: 'nir_move_tex_load',
    1024: 'nir_move_tex_load_fragment_mask',
    2048: 'nir_move_tex_lod',
    4096: 'nir_move_tex_query',
    8192: 'nir_move_load_image',
    16384: 'nir_move_load_image_fragment_mask',
    32768: 'nir_move_query_image',
    65536: 'nir_move_load_input',
    131072: 'nir_move_load_global',
    262144: 'nir_move_load_ubo',
    524288: 'nir_move_load_ssbo',
    1048576: 'nir_move_load_uniform',
    2097152: 'nir_move_load_buffer_amd',
    4194304: 'nir_move_load_frag_coord',
    1073741824: 'nir_move_only_convergent',
    2147483648: 'nir_move_only_divergent',
}
nir_move_const_undef = 1
nir_move_alu = 2
nir_move_copies = 4
nir_move_comparisons = 8
nir_dont_move_byte_word_vecs = 16
nir_move_tex_sample = 256
nir_move_tex_load = 512
nir_move_tex_load_fragment_mask = 1024
nir_move_tex_lod = 2048
nir_move_tex_query = 4096
nir_move_load_image = 8192
nir_move_load_image_fragment_mask = 16384
nir_move_query_image = 32768
nir_move_load_input = 65536
nir_move_load_global = 131072
nir_move_load_ubo = 262144
nir_move_load_ssbo = 524288
nir_move_load_uniform = 1048576
nir_move_load_buffer_amd = 2097152
nir_move_load_frag_coord = 4194304
nir_move_only_convergent = 1073741824
nir_move_only_divergent = 2147483648
c__EA_nir_move_options = ctypes.c_uint32 # enum
nir_move_options = c__EA_nir_move_options
nir_move_options__enumvalues = c__EA_nir_move_options__enumvalues
try:
    nir_can_move_instr = _libraries['FIXME_STUB'].nir_can_move_instr
    nir_can_move_instr.restype = ctypes.c_bool
    nir_can_move_instr.argtypes = [ctypes.POINTER(struct_nir_instr), nir_move_options]
except AttributeError:
    pass
try:
    nir_opt_sink = _libraries['FIXME_STUB'].nir_opt_sink
    nir_opt_sink.restype = ctypes.c_bool
    nir_opt_sink.argtypes = [ctypes.POINTER(struct_nir_shader), nir_move_options]
except AttributeError:
    pass
try:
    nir_opt_move = _libraries['FIXME_STUB'].nir_opt_move
    nir_opt_move.restype = ctypes.c_bool
    nir_opt_move.argtypes = [ctypes.POINTER(struct_nir_shader), nir_move_options]
except AttributeError:
    pass
class struct_nir_opt_offsets_options(Structure):
    pass

struct_nir_opt_offsets_options._pack_ = 1 # source:False
struct_nir_opt_offsets_options._fields_ = [
    ('uniform_max', ctypes.c_uint32),
    ('ubo_vec4_max', ctypes.c_uint32),
    ('shared_max', ctypes.c_uint32),
    ('shared_atomic_max', ctypes.c_uint32),
    ('buffer_max', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('max_offset_cb', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))),
    ('max_offset_data', ctypes.POINTER(None)),
    ('allow_offset_wrap', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

nir_opt_offsets_options = struct_nir_opt_offsets_options
try:
    nir_opt_offsets = _libraries['FIXME_STUB'].nir_opt_offsets
    nir_opt_offsets.restype = ctypes.c_bool
    nir_opt_offsets.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_offsets_options)]
except AttributeError:
    pass
class struct_nir_opt_peephole_select_options(Structure):
    pass

struct_nir_opt_peephole_select_options._pack_ = 1 # source:False
struct_nir_opt_peephole_select_options._fields_ = [
    ('limit', ctypes.c_uint32),
    ('indirect_load_ok', ctypes.c_bool),
    ('expensive_alu_ok', ctypes.c_bool),
    ('discard_ok', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
]

nir_opt_peephole_select_options = struct_nir_opt_peephole_select_options
try:
    nir_opt_peephole_select = _libraries['FIXME_STUB'].nir_opt_peephole_select
    nir_opt_peephole_select.restype = ctypes.c_bool
    nir_opt_peephole_select.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_peephole_select_options)]
except AttributeError:
    pass

# values for enumeration 'c__EA_nir_reassociate_options'
c__EA_nir_reassociate_options__enumvalues = {
    1: 'nir_reassociate_cse_heuristic',
    2: 'nir_reassociate_scalar_math',
}
nir_reassociate_cse_heuristic = 1
nir_reassociate_scalar_math = 2
c__EA_nir_reassociate_options = ctypes.c_uint32 # enum
nir_reassociate_options = c__EA_nir_reassociate_options
nir_reassociate_options__enumvalues = c__EA_nir_reassociate_options__enumvalues
try:
    nir_opt_reassociate = _libraries['FIXME_STUB'].nir_opt_reassociate
    nir_opt_reassociate.restype = ctypes.c_bool
    nir_opt_reassociate.argtypes = [ctypes.POINTER(struct_nir_shader), nir_reassociate_options]
except AttributeError:
    pass
try:
    nir_opt_reassociate_loop = _libraries['FIXME_STUB'].nir_opt_reassociate_loop
    nir_opt_reassociate_loop.restype = ctypes.c_bool
    nir_opt_reassociate_loop.argtypes = [ctypes.POINTER(struct_nir_shader), nir_reassociate_options]
except AttributeError:
    pass
try:
    nir_opt_reassociate_bfi = _libraries['FIXME_STUB'].nir_opt_reassociate_bfi
    nir_opt_reassociate_bfi.restype = ctypes.c_bool
    nir_opt_reassociate_bfi.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_rematerialize_compares = _libraries['FIXME_STUB'].nir_opt_rematerialize_compares
    nir_opt_rematerialize_compares.restype = ctypes.c_bool
    nir_opt_rematerialize_compares.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_remove_phis = _libraries['FIXME_STUB'].nir_opt_remove_phis
    nir_opt_remove_phis.restype = ctypes.c_bool
    nir_opt_remove_phis.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_remove_single_src_phis_block = _libraries['FIXME_STUB'].nir_remove_single_src_phis_block
    nir_remove_single_src_phis_block.restype = ctypes.c_bool
    nir_remove_single_src_phis_block.argtypes = [ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_opt_phi_precision = _libraries['FIXME_STUB'].nir_opt_phi_precision
    nir_opt_phi_precision.restype = ctypes.c_bool
    nir_opt_phi_precision.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_phi_to_bool = _libraries['FIXME_STUB'].nir_opt_phi_to_bool
    nir_opt_phi_to_bool.restype = ctypes.c_bool
    nir_opt_phi_to_bool.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_shrink_stores = _libraries['FIXME_STUB'].nir_opt_shrink_stores
    nir_opt_shrink_stores.restype = ctypes.c_bool
    nir_opt_shrink_stores.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_shrink_vectors = _libraries['FIXME_STUB'].nir_opt_shrink_vectors
    nir_opt_shrink_vectors.restype = ctypes.c_bool
    nir_opt_shrink_vectors.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_undef = _libraries['FIXME_STUB'].nir_opt_undef
    nir_opt_undef.restype = ctypes.c_bool
    nir_opt_undef.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_undef_to_zero = _libraries['FIXME_STUB'].nir_lower_undef_to_zero
    nir_lower_undef_to_zero.restype = ctypes.c_bool
    nir_lower_undef_to_zero.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_uniform_atomics = _libraries['FIXME_STUB'].nir_opt_uniform_atomics
    nir_opt_uniform_atomics.restype = ctypes.c_bool
    nir_opt_uniform_atomics.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_uniform_subgroup = _libraries['FIXME_STUB'].nir_opt_uniform_subgroup
    nir_opt_uniform_subgroup.restype = ctypes.c_bool
    nir_opt_uniform_subgroup.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_lower_subgroups_options)]
except AttributeError:
    pass
try:
    nir_opt_vectorize = _libraries['FIXME_STUB'].nir_opt_vectorize
    nir_opt_vectorize.restype = ctypes.c_bool
    nir_opt_vectorize.argtypes = [ctypes.POINTER(struct_nir_shader), nir_vectorize_cb, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_opt_vectorize_io = _libraries['FIXME_STUB'].nir_opt_vectorize_io
    nir_opt_vectorize_io.restype = ctypes.c_bool
    nir_opt_vectorize_io.argtypes = [ctypes.POINTER(struct_nir_shader), nir_variable_mode, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_opt_move_discards_to_top = _libraries['FIXME_STUB'].nir_opt_move_discards_to_top
    nir_opt_move_discards_to_top.restype = ctypes.c_bool
    nir_opt_move_discards_to_top.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_ray_queries = _libraries['FIXME_STUB'].nir_opt_ray_queries
    nir_opt_ray_queries.restype = ctypes.c_bool
    nir_opt_ray_queries.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_opt_ray_query_ranges = _libraries['FIXME_STUB'].nir_opt_ray_query_ranges
    nir_opt_ray_query_ranges.restype = ctypes.c_bool
    nir_opt_ray_query_ranges.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
nir_skip_helpers_instrinsic_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
class struct_nir_opt_load_skip_helpers_options(Structure):
    pass

struct_nir_opt_load_skip_helpers_options._pack_ = 1 # source:False
struct_nir_opt_load_skip_helpers_options._fields_ = [
    ('no_add_divergence', ctypes.c_bool, 1),
    ('PADDING_0', ctypes.c_uint64, 63),
    ('intrinsic_cb', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))),
    ('intrinsic_cb_data', ctypes.POINTER(None)),
]

nir_opt_load_skip_helpers_options = struct_nir_opt_load_skip_helpers_options
try:
    nir_opt_load_skip_helpers = _libraries['FIXME_STUB'].nir_opt_load_skip_helpers
    nir_opt_load_skip_helpers.restype = ctypes.c_bool
    nir_opt_load_skip_helpers.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_load_skip_helpers_options)]
except AttributeError:
    pass
try:
    nir_sweep = _libraries['FIXME_STUB'].nir_sweep
    nir_sweep.restype = None
    nir_sweep.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass

# values for enumeration 'c__EA_gl_system_value'
c__EA_gl_system_value__enumvalues = {
    0: 'SYSTEM_VALUE_SUBGROUP_SIZE',
    1: 'SYSTEM_VALUE_SUBGROUP_INVOCATION',
    2: 'SYSTEM_VALUE_SUBGROUP_EQ_MASK',
    3: 'SYSTEM_VALUE_SUBGROUP_GE_MASK',
    4: 'SYSTEM_VALUE_SUBGROUP_GT_MASK',
    5: 'SYSTEM_VALUE_SUBGROUP_LE_MASK',
    6: 'SYSTEM_VALUE_SUBGROUP_LT_MASK',
    7: 'SYSTEM_VALUE_NUM_SUBGROUPS',
    8: 'SYSTEM_VALUE_SUBGROUP_ID',
    9: 'SYSTEM_VALUE_VERTEX_ID',
    10: 'SYSTEM_VALUE_INSTANCE_ID',
    11: 'SYSTEM_VALUE_INSTANCE_INDEX',
    12: 'SYSTEM_VALUE_VERTEX_ID_ZERO_BASE',
    13: 'SYSTEM_VALUE_BASE_VERTEX',
    14: 'SYSTEM_VALUE_FIRST_VERTEX',
    15: 'SYSTEM_VALUE_IS_INDEXED_DRAW',
    16: 'SYSTEM_VALUE_BASE_INSTANCE',
    17: 'SYSTEM_VALUE_DRAW_ID',
    18: 'SYSTEM_VALUE_INVOCATION_ID',
    19: 'SYSTEM_VALUE_FRAG_COORD',
    20: 'SYSTEM_VALUE_PIXEL_COORD',
    21: 'SYSTEM_VALUE_FRAG_COORD_Z',
    22: 'SYSTEM_VALUE_FRAG_COORD_W',
    23: 'SYSTEM_VALUE_POINT_COORD',
    24: 'SYSTEM_VALUE_LINE_COORD',
    25: 'SYSTEM_VALUE_FRONT_FACE',
    26: 'SYSTEM_VALUE_FRONT_FACE_FSIGN',
    27: 'SYSTEM_VALUE_SAMPLE_ID',
    28: 'SYSTEM_VALUE_SAMPLE_POS',
    29: 'SYSTEM_VALUE_SAMPLE_POS_OR_CENTER',
    30: 'SYSTEM_VALUE_SAMPLE_MASK_IN',
    31: 'SYSTEM_VALUE_LAYER_ID',
    32: 'SYSTEM_VALUE_HELPER_INVOCATION',
    33: 'SYSTEM_VALUE_COLOR0',
    34: 'SYSTEM_VALUE_COLOR1',
    35: 'SYSTEM_VALUE_TESS_COORD',
    36: 'SYSTEM_VALUE_VERTICES_IN',
    37: 'SYSTEM_VALUE_PRIMITIVE_ID',
    38: 'SYSTEM_VALUE_TESS_LEVEL_OUTER',
    39: 'SYSTEM_VALUE_TESS_LEVEL_INNER',
    40: 'SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT',
    41: 'SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT',
    42: 'SYSTEM_VALUE_LOCAL_INVOCATION_ID',
    43: 'SYSTEM_VALUE_LOCAL_INVOCATION_INDEX',
    44: 'SYSTEM_VALUE_GLOBAL_INVOCATION_ID',
    45: 'SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID',
    46: 'SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX',
    47: 'SYSTEM_VALUE_WORKGROUP_ID',
    48: 'SYSTEM_VALUE_BASE_WORKGROUP_ID',
    49: 'SYSTEM_VALUE_WORKGROUP_INDEX',
    50: 'SYSTEM_VALUE_NUM_WORKGROUPS',
    51: 'SYSTEM_VALUE_WORKGROUP_SIZE',
    52: 'SYSTEM_VALUE_GLOBAL_GROUP_SIZE',
    53: 'SYSTEM_VALUE_WORK_DIM',
    54: 'SYSTEM_VALUE_USER_DATA_AMD',
    55: 'SYSTEM_VALUE_DEVICE_INDEX',
    56: 'SYSTEM_VALUE_VIEW_INDEX',
    57: 'SYSTEM_VALUE_VERTEX_CNT',
    58: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL',
    59: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE',
    60: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID',
    61: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW',
    62: 'SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL',
    63: 'SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID',
    64: 'SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE',
    65: 'SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL',
    66: 'SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD',
    67: 'SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD',
    68: 'SYSTEM_VALUE_RAY_LAUNCH_ID',
    69: 'SYSTEM_VALUE_RAY_LAUNCH_SIZE',
    70: 'SYSTEM_VALUE_RAY_WORLD_ORIGIN',
    71: 'SYSTEM_VALUE_RAY_WORLD_DIRECTION',
    72: 'SYSTEM_VALUE_RAY_OBJECT_ORIGIN',
    73: 'SYSTEM_VALUE_RAY_OBJECT_DIRECTION',
    74: 'SYSTEM_VALUE_RAY_T_MIN',
    75: 'SYSTEM_VALUE_RAY_T_MAX',
    76: 'SYSTEM_VALUE_RAY_OBJECT_TO_WORLD',
    77: 'SYSTEM_VALUE_RAY_WORLD_TO_OBJECT',
    78: 'SYSTEM_VALUE_RAY_HIT_KIND',
    79: 'SYSTEM_VALUE_RAY_FLAGS',
    80: 'SYSTEM_VALUE_RAY_GEOMETRY_INDEX',
    81: 'SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX',
    82: 'SYSTEM_VALUE_CULL_MASK',
    83: 'SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS',
    84: 'SYSTEM_VALUE_MESH_VIEW_COUNT',
    85: 'SYSTEM_VALUE_MESH_VIEW_INDICES',
    86: 'SYSTEM_VALUE_GS_HEADER_IR3',
    87: 'SYSTEM_VALUE_TCS_HEADER_IR3',
    88: 'SYSTEM_VALUE_REL_PATCH_ID_IR3',
    89: 'SYSTEM_VALUE_FRAG_SHADING_RATE',
    90: 'SYSTEM_VALUE_FULLY_COVERED',
    91: 'SYSTEM_VALUE_FRAG_SIZE',
    92: 'SYSTEM_VALUE_FRAG_INVOCATION_COUNT',
    93: 'SYSTEM_VALUE_SHADER_INDEX',
    94: 'SYSTEM_VALUE_COALESCED_INPUT_COUNT',
    95: 'SYSTEM_VALUE_WARPS_PER_SM_NV',
    96: 'SYSTEM_VALUE_SM_COUNT_NV',
    97: 'SYSTEM_VALUE_WARP_ID_NV',
    98: 'SYSTEM_VALUE_SM_ID_NV',
    99: 'SYSTEM_VALUE_CORE_ID',
    100: 'SYSTEM_VALUE_CORE_COUNT_ARM',
    101: 'SYSTEM_VALUE_CORE_MAX_ID_ARM',
    102: 'SYSTEM_VALUE_WARP_ID_ARM',
    103: 'SYSTEM_VALUE_WARP_MAX_ID_ARM',
    104: 'SYSTEM_VALUE_MAX',
}
SYSTEM_VALUE_SUBGROUP_SIZE = 0
SYSTEM_VALUE_SUBGROUP_INVOCATION = 1
SYSTEM_VALUE_SUBGROUP_EQ_MASK = 2
SYSTEM_VALUE_SUBGROUP_GE_MASK = 3
SYSTEM_VALUE_SUBGROUP_GT_MASK = 4
SYSTEM_VALUE_SUBGROUP_LE_MASK = 5
SYSTEM_VALUE_SUBGROUP_LT_MASK = 6
SYSTEM_VALUE_NUM_SUBGROUPS = 7
SYSTEM_VALUE_SUBGROUP_ID = 8
SYSTEM_VALUE_VERTEX_ID = 9
SYSTEM_VALUE_INSTANCE_ID = 10
SYSTEM_VALUE_INSTANCE_INDEX = 11
SYSTEM_VALUE_VERTEX_ID_ZERO_BASE = 12
SYSTEM_VALUE_BASE_VERTEX = 13
SYSTEM_VALUE_FIRST_VERTEX = 14
SYSTEM_VALUE_IS_INDEXED_DRAW = 15
SYSTEM_VALUE_BASE_INSTANCE = 16
SYSTEM_VALUE_DRAW_ID = 17
SYSTEM_VALUE_INVOCATION_ID = 18
SYSTEM_VALUE_FRAG_COORD = 19
SYSTEM_VALUE_PIXEL_COORD = 20
SYSTEM_VALUE_FRAG_COORD_Z = 21
SYSTEM_VALUE_FRAG_COORD_W = 22
SYSTEM_VALUE_POINT_COORD = 23
SYSTEM_VALUE_LINE_COORD = 24
SYSTEM_VALUE_FRONT_FACE = 25
SYSTEM_VALUE_FRONT_FACE_FSIGN = 26
SYSTEM_VALUE_SAMPLE_ID = 27
SYSTEM_VALUE_SAMPLE_POS = 28
SYSTEM_VALUE_SAMPLE_POS_OR_CENTER = 29
SYSTEM_VALUE_SAMPLE_MASK_IN = 30
SYSTEM_VALUE_LAYER_ID = 31
SYSTEM_VALUE_HELPER_INVOCATION = 32
SYSTEM_VALUE_COLOR0 = 33
SYSTEM_VALUE_COLOR1 = 34
SYSTEM_VALUE_TESS_COORD = 35
SYSTEM_VALUE_VERTICES_IN = 36
SYSTEM_VALUE_PRIMITIVE_ID = 37
SYSTEM_VALUE_TESS_LEVEL_OUTER = 38
SYSTEM_VALUE_TESS_LEVEL_INNER = 39
SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT = 40
SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT = 41
SYSTEM_VALUE_LOCAL_INVOCATION_ID = 42
SYSTEM_VALUE_LOCAL_INVOCATION_INDEX = 43
SYSTEM_VALUE_GLOBAL_INVOCATION_ID = 44
SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID = 45
SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX = 46
SYSTEM_VALUE_WORKGROUP_ID = 47
SYSTEM_VALUE_BASE_WORKGROUP_ID = 48
SYSTEM_VALUE_WORKGROUP_INDEX = 49
SYSTEM_VALUE_NUM_WORKGROUPS = 50
SYSTEM_VALUE_WORKGROUP_SIZE = 51
SYSTEM_VALUE_GLOBAL_GROUP_SIZE = 52
SYSTEM_VALUE_WORK_DIM = 53
SYSTEM_VALUE_USER_DATA_AMD = 54
SYSTEM_VALUE_DEVICE_INDEX = 55
SYSTEM_VALUE_VIEW_INDEX = 56
SYSTEM_VALUE_VERTEX_CNT = 57
SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL = 58
SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE = 59
SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID = 60
SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW = 61
SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL = 62
SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID = 63
SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE = 64
SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL = 65
SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD = 66
SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD = 67
SYSTEM_VALUE_RAY_LAUNCH_ID = 68
SYSTEM_VALUE_RAY_LAUNCH_SIZE = 69
SYSTEM_VALUE_RAY_WORLD_ORIGIN = 70
SYSTEM_VALUE_RAY_WORLD_DIRECTION = 71
SYSTEM_VALUE_RAY_OBJECT_ORIGIN = 72
SYSTEM_VALUE_RAY_OBJECT_DIRECTION = 73
SYSTEM_VALUE_RAY_T_MIN = 74
SYSTEM_VALUE_RAY_T_MAX = 75
SYSTEM_VALUE_RAY_OBJECT_TO_WORLD = 76
SYSTEM_VALUE_RAY_WORLD_TO_OBJECT = 77
SYSTEM_VALUE_RAY_HIT_KIND = 78
SYSTEM_VALUE_RAY_FLAGS = 79
SYSTEM_VALUE_RAY_GEOMETRY_INDEX = 80
SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX = 81
SYSTEM_VALUE_CULL_MASK = 82
SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS = 83
SYSTEM_VALUE_MESH_VIEW_COUNT = 84
SYSTEM_VALUE_MESH_VIEW_INDICES = 85
SYSTEM_VALUE_GS_HEADER_IR3 = 86
SYSTEM_VALUE_TCS_HEADER_IR3 = 87
SYSTEM_VALUE_REL_PATCH_ID_IR3 = 88
SYSTEM_VALUE_FRAG_SHADING_RATE = 89
SYSTEM_VALUE_FULLY_COVERED = 90
SYSTEM_VALUE_FRAG_SIZE = 91
SYSTEM_VALUE_FRAG_INVOCATION_COUNT = 92
SYSTEM_VALUE_SHADER_INDEX = 93
SYSTEM_VALUE_COALESCED_INPUT_COUNT = 94
SYSTEM_VALUE_WARPS_PER_SM_NV = 95
SYSTEM_VALUE_SM_COUNT_NV = 96
SYSTEM_VALUE_WARP_ID_NV = 97
SYSTEM_VALUE_SM_ID_NV = 98
SYSTEM_VALUE_CORE_ID = 99
SYSTEM_VALUE_CORE_COUNT_ARM = 100
SYSTEM_VALUE_CORE_MAX_ID_ARM = 101
SYSTEM_VALUE_WARP_ID_ARM = 102
SYSTEM_VALUE_WARP_MAX_ID_ARM = 103
SYSTEM_VALUE_MAX = 104
c__EA_gl_system_value = ctypes.c_uint32 # enum
gl_system_value = c__EA_gl_system_value
gl_system_value__enumvalues = c__EA_gl_system_value__enumvalues
try:
    nir_intrinsic_from_system_value = _libraries['FIXME_STUB'].nir_intrinsic_from_system_value
    nir_intrinsic_from_system_value.restype = nir_intrinsic_op
    nir_intrinsic_from_system_value.argtypes = [gl_system_value]
except AttributeError:
    pass
try:
    nir_system_value_from_intrinsic = _libraries['FIXME_STUB'].nir_system_value_from_intrinsic
    nir_system_value_from_intrinsic.restype = gl_system_value
    nir_system_value_from_intrinsic.argtypes = [nir_intrinsic_op]
except AttributeError:
    pass
try:
    nir_variable_is_in_ubo = _libraries['FIXME_STUB'].nir_variable_is_in_ubo
    nir_variable_is_in_ubo.restype = ctypes.c_bool
    nir_variable_is_in_ubo.argtypes = [ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_is_in_ssbo = _libraries['FIXME_STUB'].nir_variable_is_in_ssbo
    nir_variable_is_in_ssbo.restype = ctypes.c_bool
    nir_variable_is_in_ssbo.argtypes = [ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_is_in_block = _libraries['FIXME_STUB'].nir_variable_is_in_block
    nir_variable_is_in_block.restype = ctypes.c_bool
    nir_variable_is_in_block.argtypes = [ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_variable_count_slots = _libraries['FIXME_STUB'].nir_variable_count_slots
    nir_variable_count_slots.restype = ctypes.c_uint32
    nir_variable_count_slots.argtypes = [ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_glsl_type)]
except AttributeError:
    pass
try:
    nir_deref_count_slots = _libraries['FIXME_STUB'].nir_deref_count_slots
    nir_deref_count_slots.restype = ctypes.c_uint32
    nir_deref_count_slots.argtypes = [ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
class struct_nir_unsigned_upper_bound_config(Structure):
    pass

struct_nir_unsigned_upper_bound_config._pack_ = 1 # source:False
struct_nir_unsigned_upper_bound_config._fields_ = [
    ('min_subgroup_size', ctypes.c_uint32),
    ('max_subgroup_size', ctypes.c_uint32),
    ('max_workgroup_invocations', ctypes.c_uint32),
    ('max_workgroup_count', ctypes.c_uint32 * 3),
    ('max_workgroup_size', ctypes.c_uint32 * 3),
    ('vertex_attrib_max', ctypes.c_uint32 * 32),
]

nir_unsigned_upper_bound_config = struct_nir_unsigned_upper_bound_config
try:
    nir_unsigned_upper_bound = _libraries['FIXME_STUB'].nir_unsigned_upper_bound
    nir_unsigned_upper_bound.restype = uint32_t
    nir_unsigned_upper_bound.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_hash_table), nir_scalar, ctypes.POINTER(struct_nir_unsigned_upper_bound_config)]
except AttributeError:
    pass
try:
    nir_addition_might_overflow = _libraries['FIXME_STUB'].nir_addition_might_overflow
    nir_addition_might_overflow.restype = ctypes.c_bool
    nir_addition_might_overflow.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_hash_table), nir_scalar, ctypes.c_uint32, ctypes.POINTER(struct_nir_unsigned_upper_bound_config)]
except AttributeError:
    pass
class struct_nir_opt_preamble_options(Structure):
    pass

struct_nir_opt_preamble_options._pack_ = 1 # source:False
struct_nir_opt_preamble_options._fields_ = [
    ('drawid_uniform', ctypes.c_bool),
    ('subgroup_size_uniform', ctypes.c_bool),
    ('load_workgroup_size_allowed', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 5),
    ('def_size', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_nir_def), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(c__EA_nir_preamble_class))),
    ('preamble_storage_size', ctypes.c_uint32 * 3),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('instr_cost_cb', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('rewrite_cost_cb', ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(struct_nir_def), ctypes.POINTER(None))),
    ('avoid_instr_cb', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))),
    ('cb_data', ctypes.POINTER(None)),
]

nir_opt_preamble_options = struct_nir_opt_preamble_options
try:
    nir_opt_preamble = _libraries['FIXME_STUB'].nir_opt_preamble
    nir_opt_preamble.restype = ctypes.c_bool
    nir_opt_preamble.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nir_opt_preamble_options), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    nir_shader_get_preamble = _libraries['FIXME_STUB'].nir_shader_get_preamble
    nir_shader_get_preamble.restype = ctypes.POINTER(struct_nir_function_impl)
    nir_shader_get_preamble.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_lower_point_smooth = _libraries['FIXME_STUB'].nir_lower_point_smooth
    nir_lower_point_smooth.restype = ctypes.c_bool
    nir_lower_point_smooth.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_lower_poly_line_smooth = _libraries['FIXME_STUB'].nir_lower_poly_line_smooth
    nir_lower_poly_line_smooth.restype = ctypes.c_bool
    nir_lower_poly_line_smooth.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_mod_analysis = _libraries['FIXME_STUB'].nir_mod_analysis
    nir_mod_analysis.restype = ctypes.c_bool
    nir_mod_analysis.argtypes = [nir_scalar, nir_alu_type, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    nir_remove_tex_shadow = _libraries['FIXME_STUB'].nir_remove_tex_shadow
    nir_remove_tex_shadow.restype = ctypes.c_bool
    nir_remove_tex_shadow.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_trivialize_registers = _libraries['FIXME_STUB'].nir_trivialize_registers
    nir_trivialize_registers.restype = ctypes.c_bool
    nir_trivialize_registers.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_static_workgroup_size = _libraries['FIXME_STUB'].nir_static_workgroup_size
    nir_static_workgroup_size.restype = ctypes.c_uint32
    nir_static_workgroup_size.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_reg_get_decl = _libraries['FIXME_STUB'].nir_reg_get_decl
    nir_reg_get_decl.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_reg_get_decl.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_next_decl_reg = _libraries['FIXME_STUB'].nir_next_decl_reg
    nir_next_decl_reg.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_next_decl_reg.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_after_reg_decls = _libraries['FIXME_STUB'].nir_after_reg_decls
    nir_after_reg_decls.restype = nir_cursor
    nir_after_reg_decls.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_is_load_reg = _libraries['FIXME_STUB'].nir_is_load_reg
    nir_is_load_reg.restype = ctypes.c_bool
    nir_is_load_reg.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_is_store_reg = _libraries['FIXME_STUB'].nir_is_store_reg
    nir_is_store_reg.restype = ctypes.c_bool
    nir_is_store_reg.argtypes = [ctypes.POINTER(struct_nir_intrinsic_instr)]
except AttributeError:
    pass
try:
    nir_load_reg_for_def = _libraries['FIXME_STUB'].nir_load_reg_for_def
    nir_load_reg_for_def.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_load_reg_for_def.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_store_reg_for_def = _libraries['FIXME_STUB'].nir_store_reg_for_def
    nir_store_reg_for_def.restype = ctypes.POINTER(struct_nir_intrinsic_instr)
    nir_store_reg_for_def.argtypes = [ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
class struct_nir_use_dominance_state(Structure):
    pass

nir_use_dominance_state = struct_nir_use_dominance_state
try:
    nir_calc_use_dominance_impl = _libraries['FIXME_STUB'].nir_calc_use_dominance_impl
    nir_calc_use_dominance_impl.restype = ctypes.POINTER(struct_nir_use_dominance_state)
    nir_calc_use_dominance_impl.argtypes = [ctypes.POINTER(struct_nir_function_impl), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_get_immediate_use_dominator = _libraries['FIXME_STUB'].nir_get_immediate_use_dominator
    nir_get_immediate_use_dominator.restype = ctypes.POINTER(struct_nir_instr)
    nir_get_immediate_use_dominator.argtypes = [ctypes.POINTER(struct_nir_use_dominance_state), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_use_dominance_lca = _libraries['FIXME_STUB'].nir_use_dominance_lca
    nir_use_dominance_lca.restype = ctypes.POINTER(struct_nir_instr)
    nir_use_dominance_lca.argtypes = [ctypes.POINTER(struct_nir_use_dominance_state), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_instr_dominates_use = _libraries['FIXME_STUB'].nir_instr_dominates_use
    nir_instr_dominates_use.restype = ctypes.c_bool
    nir_instr_dominates_use.argtypes = [ctypes.POINTER(struct_nir_use_dominance_state), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_print_use_dominators = _libraries['FIXME_STUB'].nir_print_use_dominators
    nir_print_use_dominators.restype = None
    nir_print_use_dominators.argtypes = [ctypes.POINTER(struct_nir_use_dominance_state), ctypes.POINTER(ctypes.POINTER(struct_nir_instr)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_verts_in_output_prim = _libraries['FIXME_STUB'].nir_verts_in_output_prim
    nir_verts_in_output_prim.restype = ctypes.c_uint32
    nir_verts_in_output_prim.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
class struct_c__SA_nir_output_deps(Structure):
    pass

class struct_c__SA_nir_output_deps_0(Structure):
    pass

struct_c__SA_nir_output_deps_0._pack_ = 1 # source:False
struct_c__SA_nir_output_deps_0._fields_ = [
    ('instr_list', ctypes.POINTER(ctypes.POINTER(struct_nir_instr))),
    ('num_instr', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_c__SA_nir_output_deps._pack_ = 1 # source:False
struct_c__SA_nir_output_deps._fields_ = [
    ('output', struct_c__SA_nir_output_deps_0 * 112),
]

nir_output_deps = struct_c__SA_nir_output_deps
try:
    nir_gather_output_dependencies = _libraries['FIXME_STUB'].nir_gather_output_dependencies
    nir_gather_output_dependencies.restype = None
    nir_gather_output_dependencies.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_c__SA_nir_output_deps)]
except AttributeError:
    pass
try:
    nir_free_output_dependencies = _libraries['FIXME_STUB'].nir_free_output_dependencies
    nir_free_output_dependencies.restype = None
    nir_free_output_dependencies.argtypes = [ctypes.POINTER(struct_c__SA_nir_output_deps)]
except AttributeError:
    pass
class struct_c__SA_nir_input_to_output_deps(Structure):
    pass

class struct_c__SA_nir_input_to_output_deps_0(Structure):
    pass

struct_c__SA_nir_input_to_output_deps_0._pack_ = 1 # source:False
struct_c__SA_nir_input_to_output_deps_0._fields_ = [
    ('inputs', ctypes.c_uint32 * 28),
    ('defined', ctypes.c_bool),
    ('uses_ssbo_reads', ctypes.c_bool),
    ('uses_image_reads', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
]

struct_c__SA_nir_input_to_output_deps._pack_ = 1 # source:False
struct_c__SA_nir_input_to_output_deps._fields_ = [
    ('output', struct_c__SA_nir_input_to_output_deps_0 * 112),
]

nir_input_to_output_deps = struct_c__SA_nir_input_to_output_deps
try:
    nir_gather_input_to_output_dependencies = _libraries['FIXME_STUB'].nir_gather_input_to_output_dependencies
    nir_gather_input_to_output_dependencies.restype = None
    nir_gather_input_to_output_dependencies.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_c__SA_nir_input_to_output_deps)]
except AttributeError:
    pass
try:
    nir_print_input_to_output_deps = _libraries['FIXME_STUB'].nir_print_input_to_output_deps
    nir_print_input_to_output_deps.restype = None
    nir_print_input_to_output_deps.argtypes = [ctypes.POINTER(struct_c__SA_nir_input_to_output_deps), ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct__IO_FILE)]
except AttributeError:
    pass
class struct_c__SA_nir_output_clipper_var_groups(Structure):
    pass

struct_c__SA_nir_output_clipper_var_groups._pack_ = 1 # source:False
struct_c__SA_nir_output_clipper_var_groups._fields_ = [
    ('pos_only', ctypes.c_uint32 * 28),
    ('var_only', ctypes.c_uint32 * 28),
    ('both', ctypes.c_uint32 * 28),
]

nir_output_clipper_var_groups = struct_c__SA_nir_output_clipper_var_groups
try:
    nir_gather_output_clipper_var_groups = _libraries['FIXME_STUB'].nir_gather_output_clipper_var_groups
    nir_gather_output_clipper_var_groups.restype = None
    nir_gather_output_clipper_var_groups.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_c__SA_nir_output_clipper_var_groups)]
except AttributeError:
    pass
nir_builder = struct_nir_builder
try:
    nir_builder_create = _libraries['FIXME_STUB'].nir_builder_create
    nir_builder_create.restype = nir_builder
    nir_builder_create.argtypes = [ctypes.POINTER(struct_nir_function_impl)]
except AttributeError:
    pass
try:
    nir_builder_at = _libraries['FIXME_STUB'].nir_builder_at
    nir_builder_at.restype = nir_builder
    nir_builder_at.argtypes = [nir_cursor]
except AttributeError:
    pass
try:
    nir_builder_init_simple_shader = _libraries['FIXME_STUB'].nir_builder_init_simple_shader
    nir_builder_init_simple_shader.restype = nir_builder
    nir_builder_init_simple_shader.argtypes = [mesa_shader_stage, ctypes.POINTER(struct_nir_shader_compiler_options), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
nir_instr_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr), ctypes.POINTER(None))
nir_intrinsic_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_intrinsic_instr), ctypes.POINTER(None))
nir_alu_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_alu_instr), ctypes.POINTER(None))
nir_tex_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_tex_instr), ctypes.POINTER(None))
nir_phi_pass_cb = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_phi_instr), ctypes.POINTER(None))
try:
    nir_function_instructions_pass = _libraries['FIXME_STUB'].nir_function_instructions_pass
    nir_function_instructions_pass.restype = ctypes.c_bool
    nir_function_instructions_pass.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_instr_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_instructions_pass = _libraries['FIXME_STUB'].nir_shader_instructions_pass
    nir_shader_instructions_pass.restype = ctypes.c_bool
    nir_shader_instructions_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_instr_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_function_intrinsics_pass = _libraries['FIXME_STUB'].nir_function_intrinsics_pass
    nir_function_intrinsics_pass.restype = ctypes.c_bool
    nir_function_intrinsics_pass.argtypes = [ctypes.POINTER(struct_nir_function_impl), nir_intrinsic_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_intrinsics_pass = _libraries['FIXME_STUB'].nir_shader_intrinsics_pass
    nir_shader_intrinsics_pass.restype = ctypes.c_bool
    nir_shader_intrinsics_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_intrinsic_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_alu_pass = _libraries['FIXME_STUB'].nir_shader_alu_pass
    nir_shader_alu_pass.restype = ctypes.c_bool
    nir_shader_alu_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_alu_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_tex_pass = _libraries['FIXME_STUB'].nir_shader_tex_pass
    nir_shader_tex_pass.restype = ctypes.c_bool
    nir_shader_tex_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_tex_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_shader_phi_pass = _libraries['FIXME_STUB'].nir_shader_phi_pass
    nir_shader_phi_pass.restype = ctypes.c_bool
    nir_shader_phi_pass.argtypes = [ctypes.POINTER(struct_nir_shader), nir_phi_pass_cb, nir_metadata, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    nir_builder_instr_insert = _libraries['FIXME_STUB'].nir_builder_instr_insert
    nir_builder_instr_insert.restype = None
    nir_builder_instr_insert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_builder_instr_insert_at_top = _libraries['FIXME_STUB'].nir_builder_instr_insert_at_top
    nir_builder_instr_insert_at_top.restype = None
    nir_builder_instr_insert_at_top.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_instr)]
except AttributeError:
    pass
try:
    nir_builder_last_instr = _libraries['FIXME_STUB'].nir_builder_last_instr
    nir_builder_last_instr.restype = ctypes.POINTER(struct_nir_instr)
    nir_builder_last_instr.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_build_alu = _libraries['FIXME_STUB'].nir_build_alu
    nir_build_alu.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu1 = _libraries['FIXME_STUB'].nir_build_alu1
    nir_build_alu1.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu1.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu2 = _libraries['FIXME_STUB'].nir_build_alu2
    nir_build_alu2.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu2.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu3 = _libraries['FIXME_STUB'].nir_build_alu3
    nir_build_alu3.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu3.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu4 = _libraries['FIXME_STUB'].nir_build_alu4
    nir_build_alu4.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu4.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_alu_src_arr = _libraries['FIXME_STUB'].nir_build_alu_src_arr
    nir_build_alu_src_arr.restype = ctypes.POINTER(struct_nir_def)
    nir_build_alu_src_arr.argtypes = [ctypes.POINTER(struct_nir_builder), nir_op, ctypes.POINTER(ctypes.POINTER(struct_nir_def))]
except AttributeError:
    pass
try:
    nir_builder_cf_insert = _libraries['FIXME_STUB'].nir_builder_cf_insert
    nir_builder_cf_insert.restype = None
    nir_builder_cf_insert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_builder_is_inside_cf = _libraries['FIXME_STUB'].nir_builder_is_inside_cf
    nir_builder_is_inside_cf.restype = ctypes.c_bool
    nir_builder_is_inside_cf.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_cf_node)]
except AttributeError:
    pass
try:
    nir_push_if = _libraries['FIXME_STUB'].nir_push_if
    nir_push_if.restype = ctypes.POINTER(struct_nir_if)
    nir_push_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_push_else = _libraries['FIXME_STUB'].nir_push_else
    nir_push_else.restype = ctypes.POINTER(struct_nir_if)
    nir_push_else.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_pop_if = _libraries['FIXME_STUB'].nir_pop_if
    nir_pop_if.restype = None
    nir_pop_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_if)]
except AttributeError:
    pass
try:
    nir_if_phi = _libraries['FIXME_STUB'].nir_if_phi
    nir_if_phi.restype = ctypes.POINTER(struct_nir_def)
    nir_if_phi.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_push_loop = _libraries['FIXME_STUB'].nir_push_loop
    nir_push_loop.restype = ctypes.POINTER(struct_nir_loop)
    nir_push_loop.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_push_continue = _libraries['FIXME_STUB'].nir_push_continue
    nir_push_continue.restype = ctypes.POINTER(struct_nir_loop)
    nir_push_continue.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_pop_loop = _libraries['FIXME_STUB'].nir_pop_loop
    nir_pop_loop.restype = None
    nir_pop_loop.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_loop)]
except AttributeError:
    pass
try:
    nir_undef = _libraries['FIXME_STUB'].nir_undef
    nir_undef.restype = ctypes.POINTER(struct_nir_def)
    nir_undef.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_imm = _libraries['FIXME_STUB'].nir_build_imm
    nir_build_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_build_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(union_c__UA_nir_const_value)]
except AttributeError:
    pass
try:
    nir_imm_zero = _libraries['FIXME_STUB'].nir_imm_zero
    nir_imm_zero.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_zero.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_boolN_t = _libraries['FIXME_STUB'].nir_imm_boolN_t
    nir_imm_boolN_t.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_boolN_t.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_bool, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_bool = _libraries['FIXME_STUB'].nir_imm_bool
    nir_imm_bool.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_bool.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_imm_true = _libraries['FIXME_STUB'].nir_imm_true
    nir_imm_true.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_true.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_imm_false = _libraries['FIXME_STUB'].nir_imm_false
    nir_imm_false.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_false.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_imm_floatN_t = _libraries['FIXME_STUB'].nir_imm_floatN_t
    nir_imm_floatN_t.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_floatN_t.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_double, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_float16 = _libraries['FIXME_STUB'].nir_imm_float16
    nir_imm_float16.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_float16.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_float = _libraries['FIXME_STUB'].nir_imm_float
    nir_imm_float.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_float.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_double = _libraries['FIXME_STUB'].nir_imm_double
    nir_imm_double.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_double.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_imm_vec2 = _libraries['FIXME_STUB'].nir_imm_vec2
    nir_imm_vec2.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_vec2.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_vec3 = _libraries['FIXME_STUB'].nir_imm_vec3
    nir_imm_vec3.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_vec3.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float, ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_vec4 = _libraries['FIXME_STUB'].nir_imm_vec4
    nir_imm_vec4.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_vec4.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_vec4_16 = _libraries['FIXME_STUB'].nir_imm_vec4_16
    nir_imm_vec4_16.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_vec4_16.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    nir_imm_intN_t = _libraries['FIXME_STUB'].nir_imm_intN_t
    nir_imm_intN_t.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_intN_t.argtypes = [ctypes.POINTER(struct_nir_builder), uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_int = _libraries['FIXME_STUB'].nir_imm_int
    nir_imm_int.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_int.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_imm_int64 = _libraries['FIXME_STUB'].nir_imm_int64
    nir_imm_int64.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_int64.argtypes = [ctypes.POINTER(struct_nir_builder), int64_t]
except AttributeError:
    pass
try:
    nir_imm_ivec2 = _libraries['FIXME_STUB'].nir_imm_ivec2
    nir_imm_ivec2.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec2.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_imm_ivec3_intN = _libraries['FIXME_STUB'].nir_imm_ivec3_intN
    nir_imm_ivec3_intN.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec3_intN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_uvec2_intN = _libraries['FIXME_STUB'].nir_imm_uvec2_intN
    nir_imm_uvec2_intN.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_uvec2_intN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_uvec3_intN = _libraries['FIXME_STUB'].nir_imm_uvec3_intN
    nir_imm_uvec3_intN.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_uvec3_intN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_ivec3 = _libraries['FIXME_STUB'].nir_imm_ivec3
    nir_imm_ivec3.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec3.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_imm_ivec4_intN = _libraries['FIXME_STUB'].nir_imm_ivec4_intN
    nir_imm_ivec4_intN.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec4_intN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_imm_ivec4 = _libraries['FIXME_STUB'].nir_imm_ivec4
    nir_imm_ivec4.restype = ctypes.POINTER(struct_nir_def)
    nir_imm_ivec4.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_builder_alu_instr_finish_and_insert = _libraries['FIXME_STUB'].nir_builder_alu_instr_finish_and_insert
    nir_builder_alu_instr_finish_and_insert.restype = ctypes.POINTER(struct_nir_def)
    nir_builder_alu_instr_finish_and_insert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_alu_instr)]
except AttributeError:
    pass
try:
    nir_load_system_value = _libraries['FIXME_STUB'].nir_load_system_value
    nir_load_system_value.restype = ctypes.POINTER(struct_nir_def)
    nir_load_system_value.argtypes = [ctypes.POINTER(struct_nir_builder), nir_intrinsic_op, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_type_convert = _libraries['FIXME_STUB'].nir_type_convert
    nir_type_convert.restype = ctypes.POINTER(struct_nir_def)
    nir_type_convert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_alu_type, nir_alu_type, nir_rounding_mode]
except AttributeError:
    pass
try:
    nir_convert_to_bit_size = _libraries['FIXME_STUB'].nir_convert_to_bit_size
    nir_convert_to_bit_size.restype = ctypes.POINTER(struct_nir_def)
    nir_convert_to_bit_size.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_alu_type, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_i2iN = _libraries['FIXME_STUB'].nir_i2iN
    nir_i2iN.restype = ctypes.POINTER(struct_nir_def)
    nir_i2iN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_u2uN = _libraries['FIXME_STUB'].nir_u2uN
    nir_u2uN.restype = ctypes.POINTER(struct_nir_def)
    nir_u2uN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_b2bN = _libraries['FIXME_STUB'].nir_b2bN
    nir_b2bN.restype = ctypes.POINTER(struct_nir_def)
    nir_b2bN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_f2fN = _libraries['FIXME_STUB'].nir_f2fN
    nir_f2fN.restype = ctypes.POINTER(struct_nir_def)
    nir_f2fN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_i2b = _libraries['FIXME_STUB'].nir_i2b
    nir_i2b.restype = ctypes.POINTER(struct_nir_def)
    nir_i2b.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_b2iN = _libraries['FIXME_STUB'].nir_b2iN
    nir_b2iN.restype = ctypes.POINTER(struct_nir_def)
    nir_b2iN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_b2fN = _libraries['FIXME_STUB'].nir_b2fN
    nir_b2fN.restype = ctypes.POINTER(struct_nir_def)
    nir_b2fN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_i2fN = _libraries['FIXME_STUB'].nir_i2fN
    nir_i2fN.restype = ctypes.POINTER(struct_nir_def)
    nir_i2fN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_u2fN = _libraries['FIXME_STUB'].nir_u2fN
    nir_u2fN.restype = ctypes.POINTER(struct_nir_def)
    nir_u2fN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_f2uN = _libraries['FIXME_STUB'].nir_f2uN
    nir_f2uN.restype = ctypes.POINTER(struct_nir_def)
    nir_f2uN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_f2iN = _libraries['FIXME_STUB'].nir_f2iN
    nir_f2iN.restype = ctypes.POINTER(struct_nir_def)
    nir_f2iN.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_vec = _libraries['FIXME_STUB'].nir_vec
    nir_vec.restype = ctypes.POINTER(struct_nir_def)
    nir_vec.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_vec_scalars = _libraries['FIXME_STUB'].nir_vec_scalars
    nir_vec_scalars.restype = ctypes.POINTER(struct_nir_def)
    nir_vec_scalars.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_scalar), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_mov_alu = _libraries['FIXME_STUB'].nir_mov_alu
    nir_mov_alu.restype = ctypes.POINTER(struct_nir_def)
    nir_mov_alu.argtypes = [ctypes.POINTER(struct_nir_builder), nir_alu_src, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_swizzle = _libraries['FIXME_STUB'].nir_swizzle
    nir_swizzle.restype = ctypes.POINTER(struct_nir_def)
    nir_swizzle.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_fdot = _libraries['FIXME_STUB'].nir_fdot
    nir_fdot.restype = ctypes.POINTER(struct_nir_def)
    nir_fdot.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_bfdot = _libraries['FIXME_STUB'].nir_bfdot
    nir_bfdot.restype = ctypes.POINTER(struct_nir_def)
    nir_bfdot.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ball_iequal = _libraries['FIXME_STUB'].nir_ball_iequal
    nir_ball_iequal.restype = ctypes.POINTER(struct_nir_def)
    nir_ball_iequal.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ball = _libraries['FIXME_STUB'].nir_ball
    nir_ball.restype = ctypes.POINTER(struct_nir_def)
    nir_ball.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_bany_inequal = _libraries['FIXME_STUB'].nir_bany_inequal
    nir_bany_inequal.restype = ctypes.POINTER(struct_nir_def)
    nir_bany_inequal.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_bany = _libraries['FIXME_STUB'].nir_bany
    nir_bany.restype = ctypes.POINTER(struct_nir_def)
    nir_bany.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_channel = _libraries['FIXME_STUB'].nir_channel
    nir_channel.restype = ctypes.POINTER(struct_nir_def)
    nir_channel.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_mov_scalar = _libraries['FIXME_STUB'].nir_mov_scalar
    nir_mov_scalar.restype = ctypes.POINTER(struct_nir_def)
    nir_mov_scalar.argtypes = [ctypes.POINTER(struct_nir_builder), nir_scalar]
except AttributeError:
    pass
try:
    nir_channel_or_undef = _libraries['FIXME_STUB'].nir_channel_or_undef
    nir_channel_or_undef.restype = ctypes.POINTER(struct_nir_def)
    nir_channel_or_undef.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_int32]
except AttributeError:
    pass
try:
    nir_channels = _libraries['FIXME_STUB'].nir_channels
    nir_channels.restype = ctypes.POINTER(struct_nir_def)
    nir_channels.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_component_mask_t]
except AttributeError:
    pass
try:
    _nir_select_from_array_helper = _libraries['FIXME_STUB']._nir_select_from_array_helper
    _nir_select_from_array_helper.restype = ctypes.POINTER(struct_nir_def)
    _nir_select_from_array_helper.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_select_from_ssa_def_array = _libraries['FIXME_STUB'].nir_select_from_ssa_def_array
    nir_select_from_ssa_def_array.restype = ctypes.POINTER(struct_nir_def)
    nir_select_from_ssa_def_array.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.c_uint32, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_vector_extract = _libraries['FIXME_STUB'].nir_vector_extract
    nir_vector_extract.restype = ctypes.POINTER(struct_nir_def)
    nir_vector_extract.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_vector_insert_imm = _libraries['FIXME_STUB'].nir_vector_insert_imm
    nir_vector_insert_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_vector_insert_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_vector_insert = _libraries['FIXME_STUB'].nir_vector_insert
    nir_vector_insert.restype = ctypes.POINTER(struct_nir_def)
    nir_vector_insert.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_replicate = _libraries['FIXME_STUB'].nir_replicate
    nir_replicate.restype = ctypes.POINTER(struct_nir_def)
    nir_replicate.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_iadd_imm = _libraries['FIXME_STUB'].nir_iadd_imm
    nir_iadd_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_iadd_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_iadd_imm_nuw = _libraries['FIXME_STUB'].nir_iadd_imm_nuw
    nir_iadd_imm_nuw.restype = ctypes.POINTER(struct_nir_def)
    nir_iadd_imm_nuw.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_iadd_nuw = _libraries['FIXME_STUB'].nir_iadd_nuw
    nir_iadd_nuw.restype = ctypes.POINTER(struct_nir_def)
    nir_iadd_nuw.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_fgt_imm = _libraries['FIXME_STUB'].nir_fgt_imm
    nir_fgt_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fgt_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_fle_imm = _libraries['FIXME_STUB'].nir_fle_imm
    nir_fle_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fle_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_isub_imm = _libraries['FIXME_STUB'].nir_isub_imm
    nir_isub_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_isub_imm.argtypes = [ctypes.POINTER(struct_nir_builder), uint64_t, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_imax_imm = _libraries['FIXME_STUB'].nir_imax_imm
    nir_imax_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_imax_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), int64_t]
except AttributeError:
    pass
try:
    nir_imin_imm = _libraries['FIXME_STUB'].nir_imin_imm
    nir_imin_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_imin_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), int64_t]
except AttributeError:
    pass
try:
    nir_umax_imm = _libraries['FIXME_STUB'].nir_umax_imm
    nir_umax_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_umax_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_umin_imm = _libraries['FIXME_STUB'].nir_umin_imm
    nir_umin_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_umin_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    _nir_mul_imm = _libraries['FIXME_STUB']._nir_mul_imm
    _nir_mul_imm.restype = ctypes.POINTER(struct_nir_def)
    _nir_mul_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    nir_imul_imm = _libraries['FIXME_STUB'].nir_imul_imm
    nir_imul_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_imul_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_amul_imm = _libraries['FIXME_STUB'].nir_amul_imm
    nir_amul_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_amul_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_fadd_imm = _libraries['FIXME_STUB'].nir_fadd_imm
    nir_fadd_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fadd_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_fsub_imm = _libraries['FIXME_STUB'].nir_fsub_imm
    nir_fsub_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fsub_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_double, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_fmul_imm = _libraries['FIXME_STUB'].nir_fmul_imm
    nir_fmul_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fmul_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_fdiv_imm = _libraries['FIXME_STUB'].nir_fdiv_imm
    nir_fdiv_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fdiv_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_fpow_imm = _libraries['FIXME_STUB'].nir_fpow_imm
    nir_fpow_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_fpow_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_iand_imm = _libraries['FIXME_STUB'].nir_iand_imm
    nir_iand_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_iand_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_test_mask = _libraries['FIXME_STUB'].nir_test_mask
    nir_test_mask.restype = ctypes.POINTER(struct_nir_def)
    nir_test_mask.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_ior_imm = _libraries['FIXME_STUB'].nir_ior_imm
    nir_ior_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ior_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_ishl_imm = _libraries['FIXME_STUB'].nir_ishl_imm
    nir_ishl_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ishl_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_ishr_imm = _libraries['FIXME_STUB'].nir_ishr_imm
    nir_ishr_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ishr_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_ushr_imm = _libraries['FIXME_STUB'].nir_ushr_imm
    nir_ushr_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ushr_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t]
except AttributeError:
    pass
try:
    nir_imod_imm = _libraries['FIXME_STUB'].nir_imod_imm
    nir_imod_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_imod_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_udiv_imm = _libraries['FIXME_STUB'].nir_udiv_imm
    nir_udiv_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_udiv_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_umod_imm = _libraries['FIXME_STUB'].nir_umod_imm
    nir_umod_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_umod_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_align_imm = _libraries['FIXME_STUB'].nir_align_imm
    nir_align_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_align_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t]
except AttributeError:
    pass
try:
    nir_ibfe_imm = _libraries['FIXME_STUB'].nir_ibfe_imm
    nir_ibfe_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ibfe_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_ubfe_imm = _libraries['FIXME_STUB'].nir_ubfe_imm
    nir_ubfe_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ubfe_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_ubitfield_extract_imm = _libraries['FIXME_STUB'].nir_ubitfield_extract_imm
    nir_ubitfield_extract_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ubitfield_extract_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_ibitfield_extract_imm = _libraries['FIXME_STUB'].nir_ibitfield_extract_imm
    nir_ibitfield_extract_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_ibitfield_extract_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_bitfield_insert_imm = _libraries['FIXME_STUB'].nir_bitfield_insert_imm
    nir_bitfield_insert_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_bitfield_insert_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_extract_u8_imm = _libraries['FIXME_STUB'].nir_extract_u8_imm
    nir_extract_u8_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_extract_u8_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_extract_i8_imm = _libraries['FIXME_STUB'].nir_extract_i8_imm
    nir_extract_i8_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_extract_i8_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_fclamp = _libraries['FIXME_STUB'].nir_fclamp
    nir_fclamp.restype = ctypes.POINTER(struct_nir_def)
    nir_fclamp.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_iclamp = _libraries['FIXME_STUB'].nir_iclamp
    nir_iclamp.restype = ctypes.POINTER(struct_nir_def)
    nir_iclamp.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_uclamp = _libraries['FIXME_STUB'].nir_uclamp
    nir_uclamp.restype = ctypes.POINTER(struct_nir_def)
    nir_uclamp.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ffma_imm12 = _libraries['FIXME_STUB'].nir_ffma_imm12
    nir_ffma_imm12.restype = ctypes.POINTER(struct_nir_def)
    nir_ffma_imm12.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double, ctypes.c_double]
except AttributeError:
    pass
try:
    nir_ffma_imm1 = _libraries['FIXME_STUB'].nir_ffma_imm1
    nir_ffma_imm1.restype = ctypes.POINTER(struct_nir_def)
    nir_ffma_imm1.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_double, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ffma_imm2 = _libraries['FIXME_STUB'].nir_ffma_imm2
    nir_ffma_imm2.restype = ctypes.POINTER(struct_nir_def)
    nir_ffma_imm2.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.c_double]
except AttributeError:
    pass
try:
    nir_a_minus_bc = _libraries['FIXME_STUB'].nir_a_minus_bc
    nir_a_minus_bc.restype = ctypes.POINTER(struct_nir_def)
    nir_a_minus_bc.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_pack_bits = _libraries['FIXME_STUB'].nir_pack_bits
    nir_pack_bits.restype = ctypes.POINTER(struct_nir_def)
    nir_pack_bits.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_unpack_bits = _libraries['FIXME_STUB'].nir_unpack_bits
    nir_unpack_bits.restype = ctypes.POINTER(struct_nir_def)
    nir_unpack_bits.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_extract_bits = _libraries['FIXME_STUB'].nir_extract_bits
    nir_extract_bits.restype = ctypes.POINTER(struct_nir_def)
    nir_extract_bits.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.POINTER(struct_nir_def)), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_bitcast_vector = _libraries['FIXME_STUB'].nir_bitcast_vector
    nir_bitcast_vector.restype = ctypes.POINTER(struct_nir_def)
    nir_bitcast_vector.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_trim_vector = _libraries['FIXME_STUB'].nir_trim_vector
    nir_trim_vector.restype = ctypes.POINTER(struct_nir_def)
    nir_trim_vector.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_pad_vector = _libraries['FIXME_STUB'].nir_pad_vector
    nir_pad_vector.restype = ctypes.POINTER(struct_nir_def)
    nir_pad_vector.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_pad_vector_imm_int = _libraries['FIXME_STUB'].nir_pad_vector_imm_int
    nir_pad_vector_imm_int.restype = ctypes.POINTER(struct_nir_def)
    nir_pad_vector_imm_int.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), uint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_pad_vec4 = _libraries['FIXME_STUB'].nir_pad_vec4
    nir_pad_vec4.restype = ctypes.POINTER(struct_nir_def)
    nir_pad_vec4.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_resize_vector = _libraries['FIXME_STUB'].nir_resize_vector
    nir_resize_vector.restype = ctypes.POINTER(struct_nir_def)
    nir_resize_vector.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_shift_channels = _libraries['FIXME_STUB'].nir_shift_channels
    nir_shift_channels.restype = ctypes.POINTER(struct_nir_def)
    nir_shift_channels.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_ssa_for_alu_src = _libraries['FIXME_STUB'].nir_ssa_for_alu_src
    nir_ssa_for_alu_src.restype = ctypes.POINTER(struct_nir_def)
    nir_ssa_for_alu_src.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_alu_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_get_ptr_bitsize = _libraries['FIXME_STUB'].nir_get_ptr_bitsize
    nir_get_ptr_bitsize.restype = ctypes.c_uint32
    nir_get_ptr_bitsize.argtypes = [ctypes.POINTER(struct_nir_shader)]
except AttributeError:
    pass
try:
    nir_build_deref_var = _libraries['FIXME_STUB'].nir_build_deref_var
    nir_build_deref_var.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_build_deref_array = _libraries['FIXME_STUB'].nir_build_deref_array
    nir_build_deref_array.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_array.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_deref_array_imm = _libraries['FIXME_STUB'].nir_build_deref_array_imm
    nir_build_deref_array_imm.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_array_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), int64_t]
except AttributeError:
    pass
try:
    nir_build_deref_ptr_as_array = _libraries['FIXME_STUB'].nir_build_deref_ptr_as_array
    nir_build_deref_ptr_as_array.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_ptr_as_array.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_deref_array_wildcard = _libraries['FIXME_STUB'].nir_build_deref_array_wildcard
    nir_build_deref_array_wildcard.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_array_wildcard.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_build_deref_struct = _libraries['FIXME_STUB'].nir_build_deref_struct
    nir_build_deref_struct.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_struct.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_deref_cast_with_alignment = _libraries['FIXME_STUB'].nir_build_deref_cast_with_alignment
    nir_build_deref_cast_with_alignment.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_cast_with_alignment.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_variable_mode, ctypes.POINTER(struct_glsl_type), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_deref_cast = _libraries['FIXME_STUB'].nir_build_deref_cast
    nir_build_deref_cast.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_cast.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_variable_mode, ctypes.POINTER(struct_glsl_type), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_alignment_deref_cast = _libraries['FIXME_STUB'].nir_alignment_deref_cast
    nir_alignment_deref_cast.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_alignment_deref_cast.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), uint32_t, uint32_t]
except AttributeError:
    pass
try:
    nir_build_deref_follower = _libraries['FIXME_STUB'].nir_build_deref_follower
    nir_build_deref_follower.restype = ctypes.POINTER(struct_nir_deref_instr)
    nir_build_deref_follower.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_load_deref_with_access = _libraries['FIXME_STUB'].nir_load_deref_with_access
    nir_load_deref_with_access.restype = ctypes.POINTER(struct_nir_def)
    nir_load_deref_with_access.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), gl_access_qualifier]
except AttributeError:
    pass
try:
    nir_load_deref = _libraries['FIXME_STUB'].nir_load_deref
    nir_load_deref.restype = ctypes.POINTER(struct_nir_def)
    nir_load_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_store_deref_with_access = _libraries['FIXME_STUB'].nir_store_deref_with_access
    nir_store_deref_with_access.restype = None
    nir_store_deref_with_access.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, gl_access_qualifier]
except AttributeError:
    pass
try:
    nir_store_deref = _libraries['FIXME_STUB'].nir_store_deref
    nir_store_deref.restype = None
    nir_store_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_write_masked_store = _libraries['FIXME_STUB'].nir_build_write_masked_store
    nir_build_write_masked_store.restype = None
    nir_build_write_masked_store.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_build_write_masked_stores = _libraries['FIXME_STUB'].nir_build_write_masked_stores
    nir_build_write_masked_stores.restype = None
    nir_build_write_masked_stores.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_copy_deref_with_access = _libraries['FIXME_STUB'].nir_copy_deref_with_access
    nir_copy_deref_with_access.restype = None
    nir_copy_deref_with_access.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), gl_access_qualifier, gl_access_qualifier]
except AttributeError:
    pass
try:
    nir_copy_deref = _libraries['FIXME_STUB'].nir_copy_deref
    nir_copy_deref.restype = None
    nir_copy_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr)]
except AttributeError:
    pass
try:
    nir_memcpy_deref_with_access = _libraries['FIXME_STUB'].nir_memcpy_deref_with_access
    nir_memcpy_deref_with_access.restype = None
    nir_memcpy_deref_with_access.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def), gl_access_qualifier, gl_access_qualifier]
except AttributeError:
    pass
try:
    nir_memcpy_deref = _libraries['FIXME_STUB'].nir_memcpy_deref
    nir_memcpy_deref.restype = None
    nir_memcpy_deref.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_deref_instr), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_load_var = _libraries['FIXME_STUB'].nir_load_var
    nir_load_var.restype = ctypes.POINTER(struct_nir_def)
    nir_load_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_store_var = _libraries['FIXME_STUB'].nir_store_var
    nir_store_var.restype = None
    nir_store_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_copy_var = _libraries['FIXME_STUB'].nir_copy_var
    nir_copy_var.restype = None
    nir_copy_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_variable)]
except AttributeError:
    pass
try:
    nir_load_array_var = _libraries['FIXME_STUB'].nir_load_array_var
    nir_load_array_var.restype = ctypes.POINTER(struct_nir_def)
    nir_load_array_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_load_array_var_imm = _libraries['FIXME_STUB'].nir_load_array_var_imm
    nir_load_array_var_imm.restype = ctypes.POINTER(struct_nir_def)
    nir_load_array_var_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), int64_t]
except AttributeError:
    pass
try:
    nir_store_array_var = _libraries['FIXME_STUB'].nir_store_array_var
    nir_store_array_var.restype = None
    nir_store_array_var.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_store_array_var_imm = _libraries['FIXME_STUB'].nir_store_array_var_imm
    nir_store_array_var_imm.restype = None
    nir_store_array_var_imm.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_variable), int64_t, ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_load_global = _libraries['FIXME_STUB'].nir_load_global
    nir_load_global.restype = ctypes.POINTER(struct_nir_def)
    nir_load_global.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_store_global = _libraries['FIXME_STUB'].nir_store_global
    nir_store_global.restype = None
    nir_store_global.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.POINTER(struct_nir_def), nir_component_mask_t]
except AttributeError:
    pass
try:
    nir_load_global_constant = _libraries['FIXME_STUB'].nir_load_global_constant
    nir_load_global_constant.restype = ctypes.POINTER(struct_nir_def)
    nir_load_global_constant.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_load_param = _libraries['FIXME_STUB'].nir_load_param
    nir_load_param.restype = ctypes.POINTER(struct_nir_def)
    nir_load_param.argtypes = [ctypes.POINTER(struct_nir_builder), uint32_t]
except AttributeError:
    pass
try:
    nir_decl_reg = _libraries['FIXME_STUB'].nir_decl_reg
    nir_decl_reg.restype = ctypes.POINTER(struct_nir_def)
    nir_decl_reg.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_load_reg = _libraries['FIXME_STUB'].nir_load_reg
    nir_load_reg.restype = ctypes.POINTER(struct_nir_def)
    nir_load_reg.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_store_reg = _libraries['FIXME_STUB'].nir_store_reg
    nir_store_reg.restype = None
    nir_store_reg.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_tex_src_for_ssa = _libraries['FIXME_STUB'].nir_tex_src_for_ssa
    nir_tex_src_for_ssa.restype = nir_tex_src
    nir_tex_src_for_ssa.argtypes = [nir_tex_src_type, ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_deriv = _libraries['FIXME_STUB'].nir_build_deriv
    nir_build_deriv.restype = ctypes.POINTER(struct_nir_def)
    nir_build_deriv.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), nir_intrinsic_op]
except AttributeError:
    pass
try:
    nir_ddx = _libraries['FIXME_STUB'].nir_ddx
    nir_ddx.restype = ctypes.POINTER(struct_nir_def)
    nir_ddx.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddx_fine = _libraries['FIXME_STUB'].nir_ddx_fine
    nir_ddx_fine.restype = ctypes.POINTER(struct_nir_def)
    nir_ddx_fine.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddx_coarse = _libraries['FIXME_STUB'].nir_ddx_coarse
    nir_ddx_coarse.restype = ctypes.POINTER(struct_nir_def)
    nir_ddx_coarse.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddy = _libraries['FIXME_STUB'].nir_ddy
    nir_ddy.restype = ctypes.POINTER(struct_nir_def)
    nir_ddy.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddy_fine = _libraries['FIXME_STUB'].nir_ddy_fine
    nir_ddy_fine.restype = ctypes.POINTER(struct_nir_def)
    nir_ddy_fine.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_ddy_coarse = _libraries['FIXME_STUB'].nir_ddy_coarse
    nir_ddy_coarse.restype = ctypes.POINTER(struct_nir_def)
    nir_ddy_coarse.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
class struct_nir_tex_builder(Structure):
    pass

struct_nir_tex_builder._pack_ = 1 # source:False
struct_nir_tex_builder._fields_ = [
    ('coord', ctypes.POINTER(struct_nir_def)),
    ('ms_index', ctypes.POINTER(struct_nir_def)),
    ('lod', ctypes.POINTER(struct_nir_def)),
    ('bias', ctypes.POINTER(struct_nir_def)),
    ('comparator', ctypes.POINTER(struct_nir_def)),
    ('texture_index', ctypes.c_uint32),
    ('sampler_index', ctypes.c_uint32),
    ('texture_offset', ctypes.POINTER(struct_nir_def)),
    ('sampler_offset', ctypes.POINTER(struct_nir_def)),
    ('texture_handle', ctypes.POINTER(struct_nir_def)),
    ('sampler_handle', ctypes.POINTER(struct_nir_def)),
    ('texture_deref', ctypes.POINTER(struct_nir_deref_instr)),
    ('sampler_deref', ctypes.POINTER(struct_nir_deref_instr)),
    ('dim', glsl_sampler_dim),
    ('dest_type', nir_alu_type),
    ('is_array', ctypes.c_bool),
    ('can_speculate', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('backend_flags', ctypes.c_uint32),
]

try:
    nir_build_tex_struct = _libraries['FIXME_STUB'].nir_build_tex_struct
    nir_build_tex_struct.restype = ctypes.POINTER(struct_nir_def)
    nir_build_tex_struct.argtypes = [ctypes.POINTER(struct_nir_builder), nir_texop, struct_nir_tex_builder]
except AttributeError:
    pass
try:
    nir_mask = _libraries['FIXME_STUB'].nir_mask
    nir_mask.restype = ctypes.POINTER(struct_nir_def)
    nir_mask.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_load_barycentric = _libraries['FIXME_STUB'].nir_load_barycentric
    nir_load_barycentric.restype = ctypes.POINTER(struct_nir_def)
    nir_load_barycentric.argtypes = [ctypes.POINTER(struct_nir_builder), nir_intrinsic_op, ctypes.c_uint32]
except AttributeError:
    pass
try:
    nir_jump = _libraries['FIXME_STUB'].nir_jump
    nir_jump.restype = None
    nir_jump.argtypes = [ctypes.POINTER(struct_nir_builder), nir_jump_type]
except AttributeError:
    pass
try:
    nir_goto = _libraries['FIXME_STUB'].nir_goto
    nir_goto.restype = None
    nir_goto.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_goto_if = _libraries['FIXME_STUB'].nir_goto_if
    nir_goto_if.restype = None
    nir_goto_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_block), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_block)]
except AttributeError:
    pass
try:
    nir_break_if = _libraries['FIXME_STUB'].nir_break_if
    nir_break_if.restype = None
    nir_break_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_call = _libraries['FIXME_STUB'].nir_build_call
    nir_build_call.restype = None
    nir_build_call.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_function), size_t, ctypes.POINTER(ctypes.POINTER(struct_nir_def))]
except AttributeError:
    pass
try:
    nir_build_indirect_call = _libraries['FIXME_STUB'].nir_build_indirect_call
    nir_build_indirect_call.restype = None
    nir_build_indirect_call.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_function), ctypes.POINTER(struct_nir_def), size_t, ctypes.POINTER(ctypes.POINTER(struct_nir_def))]
except AttributeError:
    pass
try:
    nir_discard = _libraries['FIXME_STUB'].nir_discard
    nir_discard.restype = None
    nir_discard.argtypes = [ctypes.POINTER(struct_nir_builder)]
except AttributeError:
    pass
try:
    nir_discard_if = _libraries['FIXME_STUB'].nir_discard_if
    nir_discard_if.restype = None
    nir_discard_if.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_build_string = _libraries['FIXME_STUB'].nir_build_string
    nir_build_string.restype = ctypes.POINTER(struct_nir_def)
    nir_build_string.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_compare_func = _libraries['FIXME_STUB'].nir_compare_func
    nir_compare_func.restype = ctypes.POINTER(struct_nir_def)
    nir_compare_func.argtypes = [ctypes.POINTER(struct_nir_builder), compare_func, ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_scoped_memory_barrier = _libraries['FIXME_STUB'].nir_scoped_memory_barrier
    nir_scoped_memory_barrier.restype = None
    nir_scoped_memory_barrier.argtypes = [ctypes.POINTER(struct_nir_builder), mesa_scope, nir_memory_semantics, nir_variable_mode]
except AttributeError:
    pass
try:
    nir_gen_rect_vertices = _libraries['FIXME_STUB'].nir_gen_rect_vertices
    nir_gen_rect_vertices.restype = ctypes.POINTER(struct_nir_def)
    nir_gen_rect_vertices.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(struct_nir_def), ctypes.POINTER(struct_nir_def)]
except AttributeError:
    pass
try:
    nir_printf_fmt = _libraries['FIXME_STUB'].nir_printf_fmt
    nir_printf_fmt.restype = None
    nir_printf_fmt.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_printf_fmt_at_px = _libraries['FIXME_STUB'].nir_printf_fmt_at_px
    nir_printf_fmt_at_px.restype = None
    nir_printf_fmt_at_px.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nir_call_serialized = _libraries['FIXME_STUB'].nir_call_serialized
    nir_call_serialized.restype = ctypes.POINTER(struct_nir_def)
    nir_call_serialized.argtypes = [ctypes.POINTER(struct_nir_builder), ctypes.POINTER(ctypes.c_uint32), size_t, ctypes.POINTER(ctypes.POINTER(struct_nir_def))]
except AttributeError:
    pass
__all__ = \
    ['ACCESS_CAN_REORDER', 'ACCESS_CAN_SPECULATE', 'ACCESS_COHERENT',
    'ACCESS_CP_GE_COHERENT_AMD', 'ACCESS_FMASK_LOWERED_AMD',
    'ACCESS_INCLUDE_HELPERS', 'ACCESS_IN_BOUNDS',
    'ACCESS_IS_SWIZZLED_AMD', 'ACCESS_KEEP_SCALAR',
    'ACCESS_NON_READABLE', 'ACCESS_NON_TEMPORAL',
    'ACCESS_NON_UNIFORM', 'ACCESS_NON_WRITEABLE', 'ACCESS_RESTRICT',
    'ACCESS_SKIP_HELPERS', 'ACCESS_SMEM_AMD',
    'ACCESS_USES_FORMAT_AMD', 'ACCESS_VOLATILE',
    'COMPARE_FUNC_ALWAYS', 'COMPARE_FUNC_EQUAL',
    'COMPARE_FUNC_GEQUAL', 'COMPARE_FUNC_GREATER',
    'COMPARE_FUNC_LEQUAL', 'COMPARE_FUNC_LESS', 'COMPARE_FUNC_NEVER',
    'COMPARE_FUNC_NOTEQUAL', 'DERIVATIVE_GROUP_LINEAR',
    'DERIVATIVE_GROUP_NONE', 'DERIVATIVE_GROUP_QUADS',
    'FRAG_DEPTH_LAYOUT_ANY', 'FRAG_DEPTH_LAYOUT_GREATER',
    'FRAG_DEPTH_LAYOUT_LESS', 'FRAG_DEPTH_LAYOUT_NONE',
    'FRAG_DEPTH_LAYOUT_UNCHANGED', 'FRAG_STENCIL_LAYOUT_ANY',
    'FRAG_STENCIL_LAYOUT_GREATER', 'FRAG_STENCIL_LAYOUT_LESS',
    'FRAG_STENCIL_LAYOUT_NONE', 'FRAG_STENCIL_LAYOUT_UNCHANGED',
    'GLSL_SAMPLER_DIM_1D', 'GLSL_SAMPLER_DIM_2D',
    'GLSL_SAMPLER_DIM_3D', 'GLSL_SAMPLER_DIM_BUF',
    'GLSL_SAMPLER_DIM_CUBE', 'GLSL_SAMPLER_DIM_EXTERNAL',
    'GLSL_SAMPLER_DIM_MS', 'GLSL_SAMPLER_DIM_RECT',
    'GLSL_SAMPLER_DIM_SUBPASS', 'GLSL_SAMPLER_DIM_SUBPASS_MS',
    'GLSL_TYPE_ARRAY', 'GLSL_TYPE_ATOMIC_UINT', 'GLSL_TYPE_BFLOAT16',
    'GLSL_TYPE_BOOL', 'GLSL_TYPE_COOPERATIVE_MATRIX',
    'GLSL_TYPE_DOUBLE', 'GLSL_TYPE_ERROR', 'GLSL_TYPE_FLOAT',
    'GLSL_TYPE_FLOAT16', 'GLSL_TYPE_FLOAT_E4M3FN',
    'GLSL_TYPE_FLOAT_E5M2', 'GLSL_TYPE_IMAGE', 'GLSL_TYPE_INT',
    'GLSL_TYPE_INT16', 'GLSL_TYPE_INT64', 'GLSL_TYPE_INT8',
    'GLSL_TYPE_INTERFACE', 'GLSL_TYPE_SAMPLER', 'GLSL_TYPE_STRUCT',
    'GLSL_TYPE_SUBROUTINE', 'GLSL_TYPE_TEXTURE', 'GLSL_TYPE_UINT',
    'GLSL_TYPE_UINT16', 'GLSL_TYPE_UINT64', 'GLSL_TYPE_UINT8',
    'GLSL_TYPE_VOID', 'MESA_LOG_DEBUG', 'MESA_LOG_ERROR',
    'MESA_LOG_INFO', 'MESA_LOG_WARN', 'MESA_PRIM_COUNT',
    'MESA_PRIM_LINES', 'MESA_PRIM_LINES_ADJACENCY',
    'MESA_PRIM_LINE_LOOP', 'MESA_PRIM_LINE_STRIP',
    'MESA_PRIM_LINE_STRIP_ADJACENCY', 'MESA_PRIM_MAX',
    'MESA_PRIM_PATCHES', 'MESA_PRIM_POINTS', 'MESA_PRIM_POLYGON',
    'MESA_PRIM_QUADS', 'MESA_PRIM_QUAD_STRIP', 'MESA_PRIM_TRIANGLES',
    'MESA_PRIM_TRIANGLES_ADJACENCY', 'MESA_PRIM_TRIANGLE_FAN',
    'MESA_PRIM_TRIANGLE_STRIP', 'MESA_PRIM_TRIANGLE_STRIP_ADJACENCY',
    'MESA_PRIM_UNKNOWN', 'MESA_SHADER_ANY_HIT',
    'MESA_SHADER_CALLABLE', 'MESA_SHADER_CLOSEST_HIT',
    'MESA_SHADER_COMPUTE', 'MESA_SHADER_FRAGMENT',
    'MESA_SHADER_GEOMETRY', 'MESA_SHADER_INTERSECTION',
    'MESA_SHADER_KERNEL', 'MESA_SHADER_MESH', 'MESA_SHADER_MISS',
    'MESA_SHADER_NONE', 'MESA_SHADER_RAYGEN', 'MESA_SHADER_TASK',
    'MESA_SHADER_TESS_CTRL', 'MESA_SHADER_TESS_EVAL',
    'MESA_SHADER_VERTEX', 'NIR_CMAT_A_SIGNED', 'NIR_CMAT_B_SIGNED',
    'NIR_CMAT_C_SIGNED', 'NIR_CMAT_RESULT_SIGNED',
    'NIR_INTRINSIC_ACCESS', 'NIR_INTRINSIC_ALIGN_MUL',
    'NIR_INTRINSIC_ALIGN_OFFSET', 'NIR_INTRINSIC_ALU_OP',
    'NIR_INTRINSIC_ARG_UPPER_BOUND_U32_AMD',
    'NIR_INTRINSIC_ATOMIC_OP', 'NIR_INTRINSIC_BASE',
    'NIR_INTRINSIC_BINDING', 'NIR_INTRINSIC_BIT_SIZE',
    'NIR_INTRINSIC_CALL_IDX', 'NIR_INTRINSIC_CAN_ELIMINATE',
    'NIR_INTRINSIC_CAN_REORDER', 'NIR_INTRINSIC_CLUSTER_SIZE',
    'NIR_INTRINSIC_CMAT_DESC', 'NIR_INTRINSIC_CMAT_SIGNED_MASK',
    'NIR_INTRINSIC_COLUMN', 'NIR_INTRINSIC_COMMITTED',
    'NIR_INTRINSIC_COMPONENT', 'NIR_INTRINSIC_DESC_SET',
    'NIR_INTRINSIC_DESC_TYPE', 'NIR_INTRINSIC_DEST_BASE_TYPE',
    'NIR_INTRINSIC_DEST_TYPE', 'NIR_INTRINSIC_DIVERGENT',
    'NIR_INTRINSIC_DRIVER_LOCATION', 'NIR_INTRINSIC_DST_ACCESS',
    'NIR_INTRINSIC_DST_CMAT_DESC', 'NIR_INTRINSIC_EXECUTION_SCOPE',
    'NIR_INTRINSIC_EXPLICIT_COORD', 'NIR_INTRINSIC_FETCH_INACTIVE',
    'NIR_INTRINSIC_FLAGS', 'NIR_INTRINSIC_FMT_IDX',
    'NIR_INTRINSIC_FORMAT', 'NIR_INTRINSIC_IMAGE_ARRAY',
    'NIR_INTRINSIC_IMAGE_DIM', 'NIR_INTRINSIC_INTERP_MODE',
    'NIR_INTRINSIC_IO_SEMANTICS', 'NIR_INTRINSIC_IO_XFB',
    'NIR_INTRINSIC_IO_XFB2', 'NIR_INTRINSIC_LEGACY_FABS',
    'NIR_INTRINSIC_LEGACY_FNEG', 'NIR_INTRINSIC_LEGACY_FSAT',
    'NIR_INTRINSIC_MATRIX_LAYOUT', 'NIR_INTRINSIC_MEMORY_MODES',
    'NIR_INTRINSIC_MEMORY_SCOPE', 'NIR_INTRINSIC_MEMORY_SEMANTICS',
    'NIR_INTRINSIC_NEG_HI_AMD', 'NIR_INTRINSIC_NEG_LO_AMD',
    'NIR_INTRINSIC_NUM_ARRAY_ELEMS', 'NIR_INTRINSIC_NUM_COMPONENTS',
    'NIR_INTRINSIC_NUM_INDEX_FLAGS', 'NIR_INTRINSIC_OFFSET0',
    'NIR_INTRINSIC_OFFSET1', 'NIR_INTRINSIC_PARAM_IDX',
    'NIR_INTRINSIC_PREAMBLE_CLASS', 'NIR_INTRINSIC_QUADGROUP',
    'NIR_INTRINSIC_RANGE', 'NIR_INTRINSIC_RANGE_BASE',
    'NIR_INTRINSIC_RAY_QUERY_VALUE', 'NIR_INTRINSIC_REDUCTION_OP',
    'NIR_INTRINSIC_REPEAT_COUNT',
    'NIR_INTRINSIC_RESOURCE_ACCESS_INTEL',
    'NIR_INTRINSIC_RESOURCE_BLOCK_INTEL',
    'NIR_INTRINSIC_ROUNDING_MODE', 'NIR_INTRINSIC_SATURATE',
    'NIR_INTRINSIC_SIGN_EXTEND', 'NIR_INTRINSIC_SRC_ACCESS',
    'NIR_INTRINSIC_SRC_BASE_TYPE', 'NIR_INTRINSIC_SRC_BASE_TYPE2',
    'NIR_INTRINSIC_SRC_CMAT_DESC', 'NIR_INTRINSIC_SRC_TYPE',
    'NIR_INTRINSIC_ST64', 'NIR_INTRINSIC_STACK_SIZE',
    'NIR_INTRINSIC_STREAM_ID', 'NIR_INTRINSIC_SUBGROUP',
    'NIR_INTRINSIC_SWIZZLE_MASK', 'NIR_INTRINSIC_SYNCHRONOUS',
    'NIR_INTRINSIC_SYSTOLIC_DEPTH', 'NIR_INTRINSIC_UCP_ID',
    'NIR_INTRINSIC_VALUE_ID', 'NIR_INTRINSIC_WRITE_MASK',
    'NIR_MEMORY_ACQUIRE', 'NIR_MEMORY_ACQ_REL',
    'NIR_MEMORY_MAKE_AVAILABLE', 'NIR_MEMORY_MAKE_VISIBLE',
    'NIR_MEMORY_RELEASE', 'NIR_OP_IS_2SRC_COMMUTATIVE',
    'NIR_OP_IS_ASSOCIATIVE', 'NIR_OP_IS_INEXACT_ASSOCIATIVE',
    'NIR_OP_IS_SELECTION', 'NUM_TOTAL_VARYING_SLOTS',
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
    'PIPE_FORMAT_ASTC_10x10', 'PIPE_FORMAT_ASTC_10x10_FLOAT',
    'PIPE_FORMAT_ASTC_10x10_SRGB', 'PIPE_FORMAT_ASTC_10x5',
    'PIPE_FORMAT_ASTC_10x5_FLOAT', 'PIPE_FORMAT_ASTC_10x5_SRGB',
    'PIPE_FORMAT_ASTC_10x6', 'PIPE_FORMAT_ASTC_10x6_FLOAT',
    'PIPE_FORMAT_ASTC_10x6_SRGB', 'PIPE_FORMAT_ASTC_10x8',
    'PIPE_FORMAT_ASTC_10x8_FLOAT', 'PIPE_FORMAT_ASTC_10x8_SRGB',
    'PIPE_FORMAT_ASTC_12x10', 'PIPE_FORMAT_ASTC_12x10_FLOAT',
    'PIPE_FORMAT_ASTC_12x10_SRGB', 'PIPE_FORMAT_ASTC_12x12',
    'PIPE_FORMAT_ASTC_12x12_FLOAT', 'PIPE_FORMAT_ASTC_12x12_SRGB',
    'PIPE_FORMAT_ASTC_3x3x3', 'PIPE_FORMAT_ASTC_3x3x3_SRGB',
    'PIPE_FORMAT_ASTC_4x3x3', 'PIPE_FORMAT_ASTC_4x3x3_SRGB',
    'PIPE_FORMAT_ASTC_4x4', 'PIPE_FORMAT_ASTC_4x4_FLOAT',
    'PIPE_FORMAT_ASTC_4x4_SRGB', 'PIPE_FORMAT_ASTC_4x4x3',
    'PIPE_FORMAT_ASTC_4x4x3_SRGB', 'PIPE_FORMAT_ASTC_4x4x4',
    'PIPE_FORMAT_ASTC_4x4x4_SRGB', 'PIPE_FORMAT_ASTC_5x4',
    'PIPE_FORMAT_ASTC_5x4_FLOAT', 'PIPE_FORMAT_ASTC_5x4_SRGB',
    'PIPE_FORMAT_ASTC_5x4x4', 'PIPE_FORMAT_ASTC_5x4x4_SRGB',
    'PIPE_FORMAT_ASTC_5x5', 'PIPE_FORMAT_ASTC_5x5_FLOAT',
    'PIPE_FORMAT_ASTC_5x5_SRGB', 'PIPE_FORMAT_ASTC_5x5x4',
    'PIPE_FORMAT_ASTC_5x5x4_SRGB', 'PIPE_FORMAT_ASTC_5x5x5',
    'PIPE_FORMAT_ASTC_5x5x5_SRGB', 'PIPE_FORMAT_ASTC_6x5',
    'PIPE_FORMAT_ASTC_6x5_FLOAT', 'PIPE_FORMAT_ASTC_6x5_SRGB',
    'PIPE_FORMAT_ASTC_6x5x5', 'PIPE_FORMAT_ASTC_6x5x5_SRGB',
    'PIPE_FORMAT_ASTC_6x6', 'PIPE_FORMAT_ASTC_6x6_FLOAT',
    'PIPE_FORMAT_ASTC_6x6_SRGB', 'PIPE_FORMAT_ASTC_6x6x5',
    'PIPE_FORMAT_ASTC_6x6x5_SRGB', 'PIPE_FORMAT_ASTC_6x6x6',
    'PIPE_FORMAT_ASTC_6x6x6_SRGB', 'PIPE_FORMAT_ASTC_8x5',
    'PIPE_FORMAT_ASTC_8x5_FLOAT', 'PIPE_FORMAT_ASTC_8x5_SRGB',
    'PIPE_FORMAT_ASTC_8x6', 'PIPE_FORMAT_ASTC_8x6_FLOAT',
    'PIPE_FORMAT_ASTC_8x6_SRGB', 'PIPE_FORMAT_ASTC_8x8',
    'PIPE_FORMAT_ASTC_8x8_FLOAT', 'PIPE_FORMAT_ASTC_8x8_SRGB',
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
    'PIPE_FORMAT_R10G10B10_420_UNORM_PACKED',
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
    'PIPE_FORMAT_R8G8B8X8_UNORM',
    'PIPE_FORMAT_R8G8B8_420_UNORM_PACKED', 'PIPE_FORMAT_R8G8B8_SINT',
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
    'PIPE_FORMAT_Y10U10V10_420_UNORM_PACKED',
    'PIPE_FORMAT_Y10X6_U10X6_V10X6_420_UNORM',
    'PIPE_FORMAT_Y10X6_U10X6_V10X6_422_UNORM',
    'PIPE_FORMAT_Y10X6_U10X6_V10X6_444_UNORM',
    'PIPE_FORMAT_Y12X4_U12X4_V12X4_420_UNORM',
    'PIPE_FORMAT_Y12X4_U12X4_V12X4_422_UNORM',
    'PIPE_FORMAT_Y12X4_U12X4_V12X4_444_UNORM',
    'PIPE_FORMAT_Y16_U16V16_422_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_420_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_422_UNORM',
    'PIPE_FORMAT_Y16_U16_V16_444_UNORM', 'PIPE_FORMAT_Y210',
    'PIPE_FORMAT_Y212', 'PIPE_FORMAT_Y216', 'PIPE_FORMAT_Y410',
    'PIPE_FORMAT_Y412', 'PIPE_FORMAT_Y416',
    'PIPE_FORMAT_Y8U8V8_420_UNORM_PACKED', 'PIPE_FORMAT_Y8_400_UNORM',
    'PIPE_FORMAT_Y8_U8_V8_422_UNORM',
    'PIPE_FORMAT_Y8_U8_V8_440_UNORM',
    'PIPE_FORMAT_Y8_U8_V8_444_UNORM', 'PIPE_FORMAT_Y8_UNORM',
    'PIPE_FORMAT_YUYV', 'PIPE_FORMAT_YV12', 'PIPE_FORMAT_YV16',
    'PIPE_FORMAT_YVYU', 'PIPE_FORMAT_Z16_UNORM',
    'PIPE_FORMAT_Z16_UNORM_S8_UINT', 'PIPE_FORMAT_Z24X8_UNORM',
    'PIPE_FORMAT_Z24_UNORM_S8_UINT',
    'PIPE_FORMAT_Z24_UNORM_S8_UINT_AS_R8G8B8A8',
    'PIPE_FORMAT_Z32_FLOAT', 'PIPE_FORMAT_Z32_FLOAT_S8X24_UINT',
    'PIPE_FORMAT_Z32_UNORM', 'SCOPE_DEVICE', 'SCOPE_INVOCATION',
    'SCOPE_NONE', 'SCOPE_QUEUE_FAMILY', 'SCOPE_SHADER_CALL',
    'SCOPE_SUBGROUP', 'SCOPE_WORKGROUP', 'SUBGROUP_SIZE_API_CONSTANT',
    'SUBGROUP_SIZE_FULL_SUBGROUPS', 'SUBGROUP_SIZE_REQUIRE_128',
    'SUBGROUP_SIZE_REQUIRE_16', 'SUBGROUP_SIZE_REQUIRE_32',
    'SUBGROUP_SIZE_REQUIRE_4', 'SUBGROUP_SIZE_REQUIRE_64',
    'SUBGROUP_SIZE_REQUIRE_8', 'SUBGROUP_SIZE_UNIFORM',
    'SUBGROUP_SIZE_VARYING',
    'SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID',
    'SYSTEM_VALUE_BARYCENTRIC_LINEAR_COORD',
    'SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL',
    'SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTER_RHW',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_COORD',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL',
    'SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE',
    'SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL',
    'SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID',
    'SYSTEM_VALUE_BASE_INSTANCE', 'SYSTEM_VALUE_BASE_VERTEX',
    'SYSTEM_VALUE_BASE_WORKGROUP_ID',
    'SYSTEM_VALUE_COALESCED_INPUT_COUNT', 'SYSTEM_VALUE_COLOR0',
    'SYSTEM_VALUE_COLOR1', 'SYSTEM_VALUE_CORE_COUNT_ARM',
    'SYSTEM_VALUE_CORE_ID', 'SYSTEM_VALUE_CORE_MAX_ID_ARM',
    'SYSTEM_VALUE_CULL_MASK', 'SYSTEM_VALUE_DEVICE_INDEX',
    'SYSTEM_VALUE_DRAW_ID', 'SYSTEM_VALUE_FIRST_VERTEX',
    'SYSTEM_VALUE_FRAG_COORD', 'SYSTEM_VALUE_FRAG_COORD_W',
    'SYSTEM_VALUE_FRAG_COORD_Z', 'SYSTEM_VALUE_FRAG_INVOCATION_COUNT',
    'SYSTEM_VALUE_FRAG_SHADING_RATE', 'SYSTEM_VALUE_FRAG_SIZE',
    'SYSTEM_VALUE_FRONT_FACE', 'SYSTEM_VALUE_FRONT_FACE_FSIGN',
    'SYSTEM_VALUE_FULLY_COVERED', 'SYSTEM_VALUE_GLOBAL_GROUP_SIZE',
    'SYSTEM_VALUE_GLOBAL_INVOCATION_ID',
    'SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX',
    'SYSTEM_VALUE_GS_HEADER_IR3', 'SYSTEM_VALUE_HELPER_INVOCATION',
    'SYSTEM_VALUE_INSTANCE_ID', 'SYSTEM_VALUE_INSTANCE_INDEX',
    'SYSTEM_VALUE_INVOCATION_ID', 'SYSTEM_VALUE_IS_INDEXED_DRAW',
    'SYSTEM_VALUE_LAYER_ID', 'SYSTEM_VALUE_LINE_COORD',
    'SYSTEM_VALUE_LOCAL_INVOCATION_ID',
    'SYSTEM_VALUE_LOCAL_INVOCATION_INDEX', 'SYSTEM_VALUE_MAX',
    'SYSTEM_VALUE_MESH_VIEW_COUNT', 'SYSTEM_VALUE_MESH_VIEW_INDICES',
    'SYSTEM_VALUE_NUM_SUBGROUPS', 'SYSTEM_VALUE_NUM_WORKGROUPS',
    'SYSTEM_VALUE_PIXEL_COORD', 'SYSTEM_VALUE_POINT_COORD',
    'SYSTEM_VALUE_PRIMITIVE_ID', 'SYSTEM_VALUE_RAY_FLAGS',
    'SYSTEM_VALUE_RAY_GEOMETRY_INDEX', 'SYSTEM_VALUE_RAY_HIT_KIND',
    'SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX',
    'SYSTEM_VALUE_RAY_LAUNCH_ID', 'SYSTEM_VALUE_RAY_LAUNCH_SIZE',
    'SYSTEM_VALUE_RAY_OBJECT_DIRECTION',
    'SYSTEM_VALUE_RAY_OBJECT_ORIGIN',
    'SYSTEM_VALUE_RAY_OBJECT_TO_WORLD',
    'SYSTEM_VALUE_RAY_TRIANGLE_VERTEX_POSITIONS',
    'SYSTEM_VALUE_RAY_T_MAX', 'SYSTEM_VALUE_RAY_T_MIN',
    'SYSTEM_VALUE_RAY_WORLD_DIRECTION',
    'SYSTEM_VALUE_RAY_WORLD_ORIGIN',
    'SYSTEM_VALUE_RAY_WORLD_TO_OBJECT',
    'SYSTEM_VALUE_REL_PATCH_ID_IR3', 'SYSTEM_VALUE_SAMPLE_ID',
    'SYSTEM_VALUE_SAMPLE_MASK_IN', 'SYSTEM_VALUE_SAMPLE_POS',
    'SYSTEM_VALUE_SAMPLE_POS_OR_CENTER', 'SYSTEM_VALUE_SHADER_INDEX',
    'SYSTEM_VALUE_SM_COUNT_NV', 'SYSTEM_VALUE_SM_ID_NV',
    'SYSTEM_VALUE_SUBGROUP_EQ_MASK', 'SYSTEM_VALUE_SUBGROUP_GE_MASK',
    'SYSTEM_VALUE_SUBGROUP_GT_MASK', 'SYSTEM_VALUE_SUBGROUP_ID',
    'SYSTEM_VALUE_SUBGROUP_INVOCATION',
    'SYSTEM_VALUE_SUBGROUP_LE_MASK', 'SYSTEM_VALUE_SUBGROUP_LT_MASK',
    'SYSTEM_VALUE_SUBGROUP_SIZE', 'SYSTEM_VALUE_TCS_HEADER_IR3',
    'SYSTEM_VALUE_TESS_COORD', 'SYSTEM_VALUE_TESS_LEVEL_INNER',
    'SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT',
    'SYSTEM_VALUE_TESS_LEVEL_OUTER',
    'SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT',
    'SYSTEM_VALUE_USER_DATA_AMD', 'SYSTEM_VALUE_VERTEX_CNT',
    'SYSTEM_VALUE_VERTEX_ID', 'SYSTEM_VALUE_VERTEX_ID_ZERO_BASE',
    'SYSTEM_VALUE_VERTICES_IN', 'SYSTEM_VALUE_VIEW_INDEX',
    'SYSTEM_VALUE_WARPS_PER_SM_NV', 'SYSTEM_VALUE_WARP_ID_ARM',
    'SYSTEM_VALUE_WARP_ID_NV', 'SYSTEM_VALUE_WARP_MAX_ID_ARM',
    'SYSTEM_VALUE_WORKGROUP_ID', 'SYSTEM_VALUE_WORKGROUP_INDEX',
    'SYSTEM_VALUE_WORKGROUP_SIZE', 'SYSTEM_VALUE_WORK_DIM',
    'TESS_PRIMITIVE_ISOLINES', 'TESS_PRIMITIVE_QUADS',
    'TESS_PRIMITIVE_TRIANGLES', 'TESS_PRIMITIVE_UNSPECIFIED',
    'VARYING_SLOT_BFC0', 'VARYING_SLOT_BFC1',
    'VARYING_SLOT_BOUNDING_BOX0', 'VARYING_SLOT_BOUNDING_BOX1',
    'VARYING_SLOT_CLIP_DIST0', 'VARYING_SLOT_CLIP_DIST1',
    'VARYING_SLOT_CLIP_VERTEX', 'VARYING_SLOT_COL0',
    'VARYING_SLOT_COL1', 'VARYING_SLOT_CULL_DIST0',
    'VARYING_SLOT_CULL_DIST1', 'VARYING_SLOT_CULL_PRIMITIVE',
    'VARYING_SLOT_EDGE', 'VARYING_SLOT_FACE', 'VARYING_SLOT_FOGC',
    'VARYING_SLOT_LAYER', 'VARYING_SLOT_PATCH0',
    'VARYING_SLOT_PATCH1', 'VARYING_SLOT_PATCH10',
    'VARYING_SLOT_PATCH11', 'VARYING_SLOT_PATCH12',
    'VARYING_SLOT_PATCH13', 'VARYING_SLOT_PATCH14',
    'VARYING_SLOT_PATCH15', 'VARYING_SLOT_PATCH16',
    'VARYING_SLOT_PATCH17', 'VARYING_SLOT_PATCH18',
    'VARYING_SLOT_PATCH19', 'VARYING_SLOT_PATCH2',
    'VARYING_SLOT_PATCH20', 'VARYING_SLOT_PATCH21',
    'VARYING_SLOT_PATCH22', 'VARYING_SLOT_PATCH23',
    'VARYING_SLOT_PATCH24', 'VARYING_SLOT_PATCH25',
    'VARYING_SLOT_PATCH26', 'VARYING_SLOT_PATCH27',
    'VARYING_SLOT_PATCH28', 'VARYING_SLOT_PATCH29',
    'VARYING_SLOT_PATCH3', 'VARYING_SLOT_PATCH30',
    'VARYING_SLOT_PATCH31', 'VARYING_SLOT_PATCH4',
    'VARYING_SLOT_PATCH5', 'VARYING_SLOT_PATCH6',
    'VARYING_SLOT_PATCH7', 'VARYING_SLOT_PATCH8',
    'VARYING_SLOT_PATCH9', 'VARYING_SLOT_PNTC', 'VARYING_SLOT_POS',
    'VARYING_SLOT_PRIMITIVE_COUNT', 'VARYING_SLOT_PRIMITIVE_ID',
    'VARYING_SLOT_PRIMITIVE_INDICES',
    'VARYING_SLOT_PRIMITIVE_SHADING_RATE', 'VARYING_SLOT_PSIZ',
    'VARYING_SLOT_TASK_COUNT', 'VARYING_SLOT_TESS_LEVEL_INNER',
    'VARYING_SLOT_TESS_LEVEL_OUTER', 'VARYING_SLOT_TEX0',
    'VARYING_SLOT_TEX1', 'VARYING_SLOT_TEX2', 'VARYING_SLOT_TEX3',
    'VARYING_SLOT_TEX4', 'VARYING_SLOT_TEX5', 'VARYING_SLOT_TEX6',
    'VARYING_SLOT_TEX7', 'VARYING_SLOT_VAR0',
    'VARYING_SLOT_VAR0_16BIT', 'VARYING_SLOT_VAR1',
    'VARYING_SLOT_VAR10', 'VARYING_SLOT_VAR10_16BIT',
    'VARYING_SLOT_VAR11', 'VARYING_SLOT_VAR11_16BIT',
    'VARYING_SLOT_VAR12', 'VARYING_SLOT_VAR12_16BIT',
    'VARYING_SLOT_VAR13', 'VARYING_SLOT_VAR13_16BIT',
    'VARYING_SLOT_VAR14', 'VARYING_SLOT_VAR14_16BIT',
    'VARYING_SLOT_VAR15', 'VARYING_SLOT_VAR15_16BIT',
    'VARYING_SLOT_VAR16', 'VARYING_SLOT_VAR17', 'VARYING_SLOT_VAR18',
    'VARYING_SLOT_VAR19', 'VARYING_SLOT_VAR1_16BIT',
    'VARYING_SLOT_VAR2', 'VARYING_SLOT_VAR20', 'VARYING_SLOT_VAR21',
    'VARYING_SLOT_VAR22', 'VARYING_SLOT_VAR23', 'VARYING_SLOT_VAR24',
    'VARYING_SLOT_VAR25', 'VARYING_SLOT_VAR26', 'VARYING_SLOT_VAR27',
    'VARYING_SLOT_VAR28', 'VARYING_SLOT_VAR29',
    'VARYING_SLOT_VAR2_16BIT', 'VARYING_SLOT_VAR3',
    'VARYING_SLOT_VAR30', 'VARYING_SLOT_VAR31',
    'VARYING_SLOT_VAR3_16BIT', 'VARYING_SLOT_VAR4',
    'VARYING_SLOT_VAR4_16BIT', 'VARYING_SLOT_VAR5',
    'VARYING_SLOT_VAR5_16BIT', 'VARYING_SLOT_VAR6',
    'VARYING_SLOT_VAR6_16BIT', 'VARYING_SLOT_VAR7',
    'VARYING_SLOT_VAR7_16BIT', 'VARYING_SLOT_VAR8',
    'VARYING_SLOT_VAR8_16BIT', 'VARYING_SLOT_VAR9',
    'VARYING_SLOT_VAR9_16BIT', 'VARYING_SLOT_VIEWPORT',
    'VARYING_SLOT_VIEWPORT_MASK', 'VARYING_SLOT_VIEW_INDEX',
    '_nir_mul_imm', '_nir_select_from_array_helper',
    '_nir_shader_variable_has_mode', '_nir_src_set_parent',
    'c__EA_gl_system_value', 'c__EA_gl_varying_slot',
    'c__EA_mesa_scope', 'c__EA_nir_address_format',
    'c__EA_nir_alu_type', 'c__EA_nir_atomic_op',
    'c__EA_nir_cf_node_type', 'c__EA_nir_cmat_signed',
    'c__EA_nir_cursor_option', 'c__EA_nir_depth_layout',
    'c__EA_nir_deref_instr_has_complex_use_options',
    'c__EA_nir_deref_type', 'c__EA_nir_divergence_options',
    'c__EA_nir_instr_type', 'c__EA_nir_intrinsic_index_flag',
    'c__EA_nir_intrinsic_op', 'c__EA_nir_intrinsic_semantic_flag',
    'c__EA_nir_io_options', 'c__EA_nir_jump_type',
    'c__EA_nir_load_grouping', 'c__EA_nir_loop_control',
    'c__EA_nir_lower_array_deref_of_vec_options',
    'c__EA_nir_lower_discard_if_options',
    'c__EA_nir_lower_doubles_options',
    'c__EA_nir_lower_fp16_cast_options',
    'c__EA_nir_lower_gs_intrinsics_flags',
    'c__EA_nir_lower_int64_options',
    'c__EA_nir_lower_interpolation_options',
    'c__EA_nir_lower_io_options', 'c__EA_nir_lower_packing_op',
    'c__EA_nir_mem_access_shift_method', 'c__EA_nir_memory_semantics',
    'c__EA_nir_metadata', 'c__EA_nir_move_options', 'c__EA_nir_op',
    'c__EA_nir_op_algebraic_property', 'c__EA_nir_opt_if_options',
    'c__EA_nir_opt_move_to_top_options',
    'c__EA_nir_opt_varyings_progress', 'c__EA_nir_preamble_class',
    'c__EA_nir_ray_query_value', 'c__EA_nir_reassociate_options',
    'c__EA_nir_resource_data_intel', 'c__EA_nir_rounding_mode',
    'c__EA_nir_selection_control', 'c__EA_nir_var_declaration_type',
    'c__EA_nir_variable_mode', 'c_bool', 'c_uint64', 'compare_func',
    'exec_list_append', 'exec_list_get_head',
    'exec_list_get_head_const', 'exec_list_get_head_raw',
    'exec_list_get_tail', 'exec_list_is_empty',
    'exec_list_is_singular', 'exec_list_length',
    'exec_list_make_empty', 'exec_list_move_nodes_to',
    'exec_list_pop_head', 'exec_list_push_head',
    'exec_list_push_tail', 'exec_list_validate', 'exec_node_get_next',
    'exec_node_get_next_const', 'exec_node_get_prev',
    'exec_node_get_prev_const', 'exec_node_init',
    'exec_node_insert_after', 'exec_node_insert_list_after',
    'exec_node_insert_node_before', 'exec_node_is_head_sentinel',
    'exec_node_is_tail_sentinel', 'exec_node_remove',
    'exec_node_self_link', 'gl_access_qualifier',
    'gl_derivative_group', 'gl_subgroup_size', 'gl_system_value',
    'gl_system_value__enumvalues', 'gl_varying_slot',
    'gl_varying_slot__enumvalues', 'glsl_base_type',
    'glsl_sampler_dim', 'glsl_type_size_align_func', 'int64_t',
    'mesa_log_level', 'mesa_prim', 'mesa_scope',
    'mesa_scope__enumvalues', 'mesa_shader_stage', 'nir_a_minus_bc',
    'nir_add_inlinable_uniforms', 'nir_addition_might_overflow',
    'nir_address_format', 'nir_address_format_2x32bit_global',
    'nir_address_format_32bit_global',
    'nir_address_format_32bit_index_offset',
    'nir_address_format_32bit_index_offset_pack64',
    'nir_address_format_32bit_offset',
    'nir_address_format_32bit_offset_as_64bit',
    'nir_address_format_62bit_generic',
    'nir_address_format_64bit_bounded_global',
    'nir_address_format_64bit_global',
    'nir_address_format_64bit_global_32bit_offset',
    'nir_address_format__enumvalues', 'nir_address_format_bit_size',
    'nir_address_format_logical', 'nir_address_format_null_value',
    'nir_address_format_num_components',
    'nir_address_format_to_glsl_type',
    'nir_address_format_vec2_index_32bit_offset', 'nir_after_block',
    'nir_after_block_before_jump', 'nir_after_cf_list',
    'nir_after_cf_node', 'nir_after_cf_node_and_phis',
    'nir_after_impl', 'nir_after_instr', 'nir_after_instr_and_phis',
    'nir_after_phis', 'nir_after_reg_decls', 'nir_align_imm',
    'nir_alignment_deref_cast', 'nir_alu_binop_identity',
    'nir_alu_instr', 'nir_alu_instr_channel_used',
    'nir_alu_instr_clone', 'nir_alu_instr_create',
    'nir_alu_instr_is_comparison', 'nir_alu_instr_is_inf_preserve',
    'nir_alu_instr_is_nan_preserve',
    'nir_alu_instr_is_signed_zero_inf_nan_preserve',
    'nir_alu_instr_is_signed_zero_preserve',
    'nir_alu_instr_src_read_mask', 'nir_alu_pass_cb', 'nir_alu_src',
    'nir_alu_src_as_uint', 'nir_alu_src_copy',
    'nir_alu_src_is_trivial_ssa', 'nir_alu_src_rewrite_scalar',
    'nir_alu_srcs_equal', 'nir_alu_srcs_negative_equal',
    'nir_alu_srcs_negative_equal_typed', 'nir_alu_type',
    'nir_alu_type__enumvalues', 'nir_amul_imm',
    'nir_assign_io_var_locations', 'nir_atomic_op',
    'nir_atomic_op__enumvalues', 'nir_atomic_op_cmpxchg',
    'nir_atomic_op_dec_wrap', 'nir_atomic_op_fadd',
    'nir_atomic_op_fcmpxchg', 'nir_atomic_op_fmax',
    'nir_atomic_op_fmin', 'nir_atomic_op_iadd', 'nir_atomic_op_iand',
    'nir_atomic_op_imax', 'nir_atomic_op_imin',
    'nir_atomic_op_inc_wrap', 'nir_atomic_op_ior',
    'nir_atomic_op_ixor', 'nir_atomic_op_ordered_add_gfx12_amd',
    'nir_atomic_op_to_alu', 'nir_atomic_op_type',
    'nir_atomic_op_umax', 'nir_atomic_op_umin', 'nir_atomic_op_xchg',
    'nir_b2bN', 'nir_b2fN', 'nir_b2iN', 'nir_ball', 'nir_ball_iequal',
    'nir_bany', 'nir_bany_inequal', 'nir_before_block',
    'nir_before_block_after_phis', 'nir_before_cf_list',
    'nir_before_cf_node', 'nir_before_impl', 'nir_before_instr',
    'nir_before_src', 'nir_bfdot', 'nir_binding',
    'nir_bitcast_vector', 'nir_bitfield_insert_imm', 'nir_block',
    'nir_block_cf_tree_next', 'nir_block_cf_tree_prev',
    'nir_block_contains_work', 'nir_block_create',
    'nir_block_dominates', 'nir_block_ends_in_break',
    'nir_block_ends_in_jump', 'nir_block_ends_in_return_or_halt',
    'nir_block_first_instr', 'nir_block_get_following_if',
    'nir_block_get_following_loop',
    'nir_block_get_predecessors_sorted', 'nir_block_is_reachable',
    'nir_block_is_unreachable', 'nir_block_last_instr',
    'nir_block_last_phi_instr', 'nir_block_unstructured_next',
    'nir_break_if', 'nir_build_addr_iadd', 'nir_build_addr_iadd_imm',
    'nir_build_addr_ieq', 'nir_build_addr_isub', 'nir_build_alu',
    'nir_build_alu1', 'nir_build_alu2', 'nir_build_alu3',
    'nir_build_alu4', 'nir_build_alu_src_arr', 'nir_build_call',
    'nir_build_deref_array', 'nir_build_deref_array_imm',
    'nir_build_deref_array_wildcard', 'nir_build_deref_cast',
    'nir_build_deref_cast_with_alignment', 'nir_build_deref_follower',
    'nir_build_deref_ptr_as_array', 'nir_build_deref_struct',
    'nir_build_deref_var', 'nir_build_deriv', 'nir_build_imm',
    'nir_build_indirect_call',
    'nir_build_lowered_load_helper_invocation', 'nir_build_string',
    'nir_build_tex_struct', 'nir_build_write_masked_store',
    'nir_build_write_masked_stores', 'nir_builder',
    'nir_builder_alu_instr_finish_and_insert', 'nir_builder_at',
    'nir_builder_cf_insert', 'nir_builder_create',
    'nir_builder_init_simple_shader', 'nir_builder_instr_insert',
    'nir_builder_instr_insert_at_top', 'nir_builder_is_inside_cf',
    'nir_builder_last_instr', 'nir_calc_dominance',
    'nir_calc_dominance_impl', 'nir_calc_use_dominance_impl',
    'nir_call_instr', 'nir_call_instr_create', 'nir_call_serialized',
    'nir_can_lower_multiview', 'nir_can_move_instr',
    'nir_cf_list_is_empty_block', 'nir_cf_node',
    'nir_cf_node_as_block', 'nir_cf_node_as_function',
    'nir_cf_node_as_if', 'nir_cf_node_as_loop', 'nir_cf_node_block',
    'nir_cf_node_cf_tree_first', 'nir_cf_node_cf_tree_last',
    'nir_cf_node_cf_tree_next', 'nir_cf_node_cf_tree_prev',
    'nir_cf_node_function', 'nir_cf_node_get_function',
    'nir_cf_node_if', 'nir_cf_node_is_first', 'nir_cf_node_is_last',
    'nir_cf_node_loop', 'nir_cf_node_next', 'nir_cf_node_prev',
    'nir_cf_node_type', 'nir_cf_node_type__enumvalues', 'nir_channel',
    'nir_channel_or_undef', 'nir_channels', 'nir_chase_binding',
    'nir_cleanup_functions', 'nir_clear_mediump_io_flag',
    'nir_clear_shared_memory', 'nir_clone_deref_instr',
    'nir_clone_uniform_variable', 'nir_cmat_signed',
    'nir_cmat_signed__enumvalues', 'nir_collect_src_uniforms',
    'nir_combine_barrier_cb', 'nir_combined_align',
    'nir_compact_varyings', 'nir_compare_func', 'nir_component_mask',
    'nir_component_mask_can_reinterpret',
    'nir_component_mask_reinterpret', 'nir_component_mask_t',
    'nir_const_value', 'nir_const_value_as_bool',
    'nir_const_value_as_float', 'nir_const_value_as_int',
    'nir_const_value_as_uint', 'nir_const_value_for_bool',
    'nir_const_value_for_float', 'nir_const_value_for_int',
    'nir_const_value_for_raw_uint', 'nir_const_value_for_uint',
    'nir_const_value_negative_equal', 'nir_constant',
    'nir_constant_clone', 'nir_convert_from_ssa',
    'nir_convert_loop_to_lcssa', 'nir_convert_to_bit_size',
    'nir_convert_to_lcssa', 'nir_copy_deref',
    'nir_copy_deref_with_access', 'nir_copy_prop',
    'nir_copy_prop_impl', 'nir_copy_var', 'nir_create_passthrough_gs',
    'nir_create_passthrough_tcs', 'nir_create_passthrough_tcs_impl',
    'nir_create_variable_with_location', 'nir_cursor',
    'nir_cursor_after_block', 'nir_cursor_after_instr',
    'nir_cursor_before_block', 'nir_cursor_before_instr',
    'nir_cursor_current_block', 'nir_cursor_option',
    'nir_cursor_option__enumvalues', 'nir_cursors_equal', 'nir_ddx',
    'nir_ddx_coarse', 'nir_ddx_fine', 'nir_ddy', 'nir_ddy_coarse',
    'nir_ddy_fine', 'nir_debug', 'nir_debug_print_shader',
    'nir_decl_reg', 'nir_dedup_inline_samplers', 'nir_def',
    'nir_def_all_uses_are_fsat', 'nir_def_all_uses_ignore_sign_bit',
    'nir_def_as_alu', 'nir_def_as_deref', 'nir_def_as_intrinsic',
    'nir_def_as_load_const', 'nir_def_as_phi', 'nir_def_as_tex',
    'nir_def_block', 'nir_def_components_read',
    'nir_def_first_component_read', 'nir_def_init',
    'nir_def_init_for_type', 'nir_def_is_frag_coord_z',
    'nir_def_is_unused', 'nir_def_last_component_read',
    'nir_def_only_used_by_if', 'nir_def_replace',
    'nir_def_rewrite_uses', 'nir_def_rewrite_uses_after',
    'nir_def_rewrite_uses_after_instr', 'nir_def_rewrite_uses_src',
    'nir_def_used_by_if', 'nir_defs_interfere', 'nir_depth_layout',
    'nir_depth_layout__enumvalues', 'nir_depth_layout_any',
    'nir_depth_layout_greater', 'nir_depth_layout_less',
    'nir_depth_layout_none', 'nir_depth_layout_unchanged',
    'nir_deref_cast_is_trivial', 'nir_deref_count_slots',
    'nir_deref_instr', 'nir_deref_instr_array_stride',
    'nir_deref_instr_create', 'nir_deref_instr_get_variable',
    'nir_deref_instr_has_complex_use',
    'nir_deref_instr_has_complex_use_allow_atomics',
    'nir_deref_instr_has_complex_use_allow_memcpy_dst',
    'nir_deref_instr_has_complex_use_allow_memcpy_src',
    'nir_deref_instr_has_complex_use_options',
    'nir_deref_instr_has_complex_use_options__enumvalues',
    'nir_deref_instr_has_indirect',
    'nir_deref_instr_is_known_out_of_bounds',
    'nir_deref_instr_parent', 'nir_deref_instr_remove_if_unused',
    'nir_deref_mode_is', 'nir_deref_mode_is_in_set',
    'nir_deref_mode_is_one_of', 'nir_deref_mode_may_be',
    'nir_deref_mode_must_be', 'nir_deref_type',
    'nir_deref_type__enumvalues', 'nir_deref_type_array',
    'nir_deref_type_array_wildcard', 'nir_deref_type_cast',
    'nir_deref_type_ptr_as_array', 'nir_deref_type_struct',
    'nir_deref_type_var', 'nir_discard', 'nir_discard_if',
    'nir_divergence_analysis', 'nir_divergence_analysis_impl',
    'nir_divergence_ignore_undef_if_phi_srcs',
    'nir_divergence_multiple_workgroup_per_compute_subgroup',
    'nir_divergence_options', 'nir_divergence_options__enumvalues',
    'nir_divergence_shader_record_ptr_uniform',
    'nir_divergence_single_frag_shading_rate_per_subgroup',
    'nir_divergence_single_patch_per_tcs_subgroup',
    'nir_divergence_single_patch_per_tes_subgroup',
    'nir_divergence_single_prim_per_subgroup',
    'nir_divergence_uniform_load_tears', 'nir_divergence_vertex',
    'nir_divergence_view_index_uniform', 'nir_dominance_lca',
    'nir_dont_move_byte_word_vecs', 'nir_dump_cfg',
    'nir_dump_cfg_impl', 'nir_dump_dom_frontier',
    'nir_dump_dom_frontier_impl', 'nir_dump_dom_tree',
    'nir_dump_dom_tree_impl', 'nir_explicit_io_address_from_deref',
    'nir_extract_bits', 'nir_extract_i8_imm', 'nir_extract_u8_imm',
    'nir_f2fN', 'nir_f2iN', 'nir_f2uN', 'nir_fadd_imm', 'nir_fclamp',
    'nir_fdiv_imm', 'nir_fdot', 'nir_ffma_imm1', 'nir_ffma_imm12',
    'nir_ffma_imm2', 'nir_fgt_imm', 'nir_find_inlinable_uniforms',
    'nir_find_sampler_variable_with_tex_index',
    'nir_find_state_variable',
    'nir_find_variable_with_driver_location',
    'nir_find_variable_with_location', 'nir_first_phi_in_block',
    'nir_fixup_deref_modes', 'nir_fixup_deref_types',
    'nir_fixup_is_exported', 'nir_fle_imm', 'nir_fmul_imm',
    'nir_foreach_def_cb', 'nir_foreach_function_with_impl_first',
    'nir_foreach_function_with_impl_next',
    'nir_foreach_phi_src_leaving_block', 'nir_foreach_src',
    'nir_foreach_src_cb', 'nir_fpow_imm',
    'nir_free_output_dependencies', 'nir_fsub_imm', 'nir_function',
    'nir_function_clone', 'nir_function_create', 'nir_function_impl',
    'nir_function_impl_add_variable', 'nir_function_impl_clone',
    'nir_function_impl_clone_remap_globals',
    'nir_function_impl_create', 'nir_function_impl_create_bare',
    'nir_function_impl_index_vars',
    'nir_function_impl_lower_instructions',
    'nir_function_instructions_pass', 'nir_function_intrinsics_pass',
    'nir_function_set_impl', 'nir_gather_explicit_io_initializers',
    'nir_gather_input_to_output_dependencies',
    'nir_gather_output_clipper_var_groups',
    'nir_gather_output_dependencies', 'nir_gather_types',
    'nir_gen_rect_vertices', 'nir_get_binding_variable',
    'nir_get_explicit_deref_align',
    'nir_get_glsl_base_type_for_nir_type',
    'nir_get_immediate_use_dominator', 'nir_get_io_arrayed_index_src',
    'nir_get_io_arrayed_index_src_number', 'nir_get_io_index_src',
    'nir_get_io_index_src_number', 'nir_get_io_intrinsic',
    'nir_get_io_offset_src', 'nir_get_io_offset_src_number',
    'nir_get_live_defs', 'nir_get_nir_type_for_glsl_base_type',
    'nir_get_nir_type_for_glsl_type', 'nir_get_ptr_bitsize',
    'nir_get_rounding_mode_from_float_controls', 'nir_get_scalar',
    'nir_get_shader_call_payload_src', 'nir_get_tex_deref',
    'nir_get_tex_src', 'nir_get_variable_with_location', 'nir_goto',
    'nir_goto_if', 'nir_group_all', 'nir_group_same_resource_only',
    'nir_gs_count_vertices_and_primitives',
    'nir_has_any_rounding_mode_enabled',
    'nir_has_any_rounding_mode_rtne', 'nir_has_any_rounding_mode_rtz',
    'nir_has_divergent_loop', 'nir_has_non_uniform_access', 'nir_i2b',
    'nir_i2fN', 'nir_i2iN', 'nir_iadd_imm', 'nir_iadd_imm_nuw',
    'nir_iadd_nuw', 'nir_iand_imm', 'nir_ibfe_imm',
    'nir_ibitfield_extract_imm', 'nir_iclamp', 'nir_if',
    'nir_if_create', 'nir_if_first_else_block',
    'nir_if_first_then_block', 'nir_if_last_else_block',
    'nir_if_last_then_block', 'nir_if_phi',
    'nir_image_intrinsic_coord_components', 'nir_imax_imm',
    'nir_imin_imm', 'nir_imm_bool', 'nir_imm_boolN_t',
    'nir_imm_double', 'nir_imm_false', 'nir_imm_float',
    'nir_imm_float16', 'nir_imm_floatN_t', 'nir_imm_int',
    'nir_imm_int64', 'nir_imm_intN_t', 'nir_imm_ivec2',
    'nir_imm_ivec3', 'nir_imm_ivec3_intN', 'nir_imm_ivec4',
    'nir_imm_ivec4_intN', 'nir_imm_true', 'nir_imm_uvec2_intN',
    'nir_imm_uvec3_intN', 'nir_imm_vec2', 'nir_imm_vec3',
    'nir_imm_vec4', 'nir_imm_vec4_16', 'nir_imm_zero', 'nir_imod_imm',
    'nir_impl_last_block', 'nir_imul_imm', 'nir_index_blocks',
    'nir_index_instrs', 'nir_index_ssa_defs',
    'nir_inline_function_impl', 'nir_inline_functions',
    'nir_inline_sysval', 'nir_inline_uniforms',
    'nir_input_attachment_options', 'nir_input_to_output_deps',
    'nir_instr', 'nir_instr_as_alu', 'nir_instr_as_call',
    'nir_instr_as_deref', 'nir_instr_as_intrinsic',
    'nir_instr_as_jump', 'nir_instr_as_load_const',
    'nir_instr_as_parallel_copy', 'nir_instr_as_phi',
    'nir_instr_as_str', 'nir_instr_as_tex', 'nir_instr_as_undef',
    'nir_instr_can_speculate', 'nir_instr_clear_src',
    'nir_instr_clone', 'nir_instr_clone_deep', 'nir_instr_debug_info',
    'nir_instr_def', 'nir_instr_dominates_use', 'nir_instr_filter_cb',
    'nir_instr_free', 'nir_instr_free_and_dce', 'nir_instr_free_list',
    'nir_instr_get_debug_info', 'nir_instr_get_gc_pointer',
    'nir_instr_init_src', 'nir_instr_insert',
    'nir_instr_insert_after', 'nir_instr_insert_after_block',
    'nir_instr_insert_after_cf', 'nir_instr_insert_after_cf_list',
    'nir_instr_insert_before', 'nir_instr_insert_before_block',
    'nir_instr_insert_before_cf', 'nir_instr_insert_before_cf_list',
    'nir_instr_is_before', 'nir_instr_is_first', 'nir_instr_is_last',
    'nir_instr_move', 'nir_instr_move_src', 'nir_instr_next',
    'nir_instr_pass_cb', 'nir_instr_prev', 'nir_instr_remove',
    'nir_instr_remove_v', 'nir_instr_type',
    'nir_instr_type__enumvalues', 'nir_instr_type_alu',
    'nir_instr_type_call', 'nir_instr_type_deref',
    'nir_instr_type_intrinsic', 'nir_instr_type_jump',
    'nir_instr_type_load_const', 'nir_instr_type_parallel_copy',
    'nir_instr_type_phi', 'nir_instr_type_tex',
    'nir_instr_type_undef', 'nir_instr_writemask_filter_cb',
    'nir_instr_xfb_write_mask', 'nir_instrs_equal',
    'nir_intrin_filter_cb', 'nir_intrinsic_accept_ray_intersection',
    'nir_intrinsic_addr_mode_is', 'nir_intrinsic_al2p_nv',
    'nir_intrinsic_ald_nv', 'nir_intrinsic_align',
    'nir_intrinsic_alpha_to_coverage', 'nir_intrinsic_as_uniform',
    'nir_intrinsic_ast_nv',
    'nir_intrinsic_atomic_add_gen_prim_count_amd',
    'nir_intrinsic_atomic_add_gs_emit_prim_count_amd',
    'nir_intrinsic_atomic_add_shader_invocation_count_amd',
    'nir_intrinsic_atomic_add_xfb_prim_count_amd',
    'nir_intrinsic_atomic_counter_add',
    'nir_intrinsic_atomic_counter_add_deref',
    'nir_intrinsic_atomic_counter_and',
    'nir_intrinsic_atomic_counter_and_deref',
    'nir_intrinsic_atomic_counter_comp_swap',
    'nir_intrinsic_atomic_counter_comp_swap_deref',
    'nir_intrinsic_atomic_counter_exchange',
    'nir_intrinsic_atomic_counter_exchange_deref',
    'nir_intrinsic_atomic_counter_inc',
    'nir_intrinsic_atomic_counter_inc_deref',
    'nir_intrinsic_atomic_counter_max',
    'nir_intrinsic_atomic_counter_max_deref',
    'nir_intrinsic_atomic_counter_min',
    'nir_intrinsic_atomic_counter_min_deref',
    'nir_intrinsic_atomic_counter_or',
    'nir_intrinsic_atomic_counter_or_deref',
    'nir_intrinsic_atomic_counter_post_dec',
    'nir_intrinsic_atomic_counter_post_dec_deref',
    'nir_intrinsic_atomic_counter_pre_dec',
    'nir_intrinsic_atomic_counter_pre_dec_deref',
    'nir_intrinsic_atomic_counter_read',
    'nir_intrinsic_atomic_counter_read_deref',
    'nir_intrinsic_atomic_counter_xor',
    'nir_intrinsic_atomic_counter_xor_deref', 'nir_intrinsic_ballot',
    'nir_intrinsic_ballot_bit_count_exclusive',
    'nir_intrinsic_ballot_bit_count_inclusive',
    'nir_intrinsic_ballot_bit_count_reduce',
    'nir_intrinsic_ballot_bitfield_extract',
    'nir_intrinsic_ballot_find_lsb', 'nir_intrinsic_ballot_find_msb',
    'nir_intrinsic_ballot_relaxed', 'nir_intrinsic_bar_break_nv',
    'nir_intrinsic_bar_set_nv', 'nir_intrinsic_bar_sync_nv',
    'nir_intrinsic_barrier',
    'nir_intrinsic_begin_invocation_interlock',
    'nir_intrinsic_bindgen_return',
    'nir_intrinsic_bindless_image_agx',
    'nir_intrinsic_bindless_image_atomic',
    'nir_intrinsic_bindless_image_atomic_swap',
    'nir_intrinsic_bindless_image_descriptor_amd',
    'nir_intrinsic_bindless_image_format',
    'nir_intrinsic_bindless_image_fragment_mask_load_amd',
    'nir_intrinsic_bindless_image_levels',
    'nir_intrinsic_bindless_image_load',
    'nir_intrinsic_bindless_image_load_raw_intel',
    'nir_intrinsic_bindless_image_order',
    'nir_intrinsic_bindless_image_samples',
    'nir_intrinsic_bindless_image_samples_identical',
    'nir_intrinsic_bindless_image_size',
    'nir_intrinsic_bindless_image_sparse_load',
    'nir_intrinsic_bindless_image_store',
    'nir_intrinsic_bindless_image_store_block_agx',
    'nir_intrinsic_bindless_image_store_raw_intel',
    'nir_intrinsic_bindless_image_texel_address',
    'nir_intrinsic_bindless_resource_ir3',
    'nir_intrinsic_bindless_sampler_agx',
    'nir_intrinsic_brcst_active_ir3',
    'nir_intrinsic_btd_retire_intel', 'nir_intrinsic_btd_spawn_intel',
    'nir_intrinsic_btd_stack_push_intel',
    'nir_intrinsic_bvh64_intersect_ray_amd',
    'nir_intrinsic_bvh8_intersect_ray_amd',
    'nir_intrinsic_bvh_stack_rtn_amd', 'nir_intrinsic_can_reorder',
    'nir_intrinsic_cmat_binary_op', 'nir_intrinsic_cmat_bitcast',
    'nir_intrinsic_cmat_construct', 'nir_intrinsic_cmat_convert',
    'nir_intrinsic_cmat_copy', 'nir_intrinsic_cmat_extract',
    'nir_intrinsic_cmat_insert', 'nir_intrinsic_cmat_length',
    'nir_intrinsic_cmat_load', 'nir_intrinsic_cmat_muladd',
    'nir_intrinsic_cmat_muladd_amd', 'nir_intrinsic_cmat_muladd_nv',
    'nir_intrinsic_cmat_scalar_op', 'nir_intrinsic_cmat_store',
    'nir_intrinsic_cmat_transpose', 'nir_intrinsic_cmat_unary_op',
    'nir_intrinsic_convert_alu_types',
    'nir_intrinsic_convert_cmat_intel',
    'nir_intrinsic_copy_const_indices', 'nir_intrinsic_copy_deref',
    'nir_intrinsic_copy_fs_outputs_nv',
    'nir_intrinsic_copy_global_to_uniform_ir3',
    'nir_intrinsic_copy_push_const_to_uniform_ir3',
    'nir_intrinsic_copy_ubo_to_uniform_ir3', 'nir_intrinsic_ddx',
    'nir_intrinsic_ddx_coarse', 'nir_intrinsic_ddx_fine',
    'nir_intrinsic_ddy', 'nir_intrinsic_ddy_coarse',
    'nir_intrinsic_ddy_fine', 'nir_intrinsic_debug_break',
    'nir_intrinsic_decl_reg', 'nir_intrinsic_demote',
    'nir_intrinsic_demote_if', 'nir_intrinsic_demote_samples',
    'nir_intrinsic_deref_atomic', 'nir_intrinsic_deref_atomic_swap',
    'nir_intrinsic_deref_buffer_array_length',
    'nir_intrinsic_deref_implicit_array_length',
    'nir_intrinsic_deref_mode_is', 'nir_intrinsic_deref_texture_src',
    'nir_intrinsic_dest_components', 'nir_intrinsic_doorbell_agx',
    'nir_intrinsic_dpas_intel', 'nir_intrinsic_dpp16_shift_amd',
    'nir_intrinsic_elect', 'nir_intrinsic_elect_any_ir3',
    'nir_intrinsic_emit_primitive_poly', 'nir_intrinsic_emit_vertex',
    'nir_intrinsic_emit_vertex_nv',
    'nir_intrinsic_emit_vertex_with_counter',
    'nir_intrinsic_end_invocation_interlock',
    'nir_intrinsic_end_primitive', 'nir_intrinsic_end_primitive_nv',
    'nir_intrinsic_end_primitive_with_counter',
    'nir_intrinsic_enqueue_node_payloads',
    'nir_intrinsic_exclusive_scan',
    'nir_intrinsic_exclusive_scan_clusters_ir3',
    'nir_intrinsic_execute_callable',
    'nir_intrinsic_execute_closest_hit_amd',
    'nir_intrinsic_execute_miss_amd', 'nir_intrinsic_export_agx',
    'nir_intrinsic_export_amd',
    'nir_intrinsic_export_dual_src_blend_amd',
    'nir_intrinsic_export_row_amd',
    'nir_intrinsic_fence_helper_exit_agx',
    'nir_intrinsic_fence_mem_to_tex_agx',
    'nir_intrinsic_fence_pbe_to_tex_agx',
    'nir_intrinsic_fence_pbe_to_tex_pixel_agx',
    'nir_intrinsic_final_primitive_nv',
    'nir_intrinsic_finalize_incoming_node_payload',
    'nir_intrinsic_first_invocation',
    'nir_intrinsic_from_system_value', 'nir_intrinsic_fs_out_nv',
    'nir_intrinsic_gds_atomic_add_amd', 'nir_intrinsic_get_ssbo_size',
    'nir_intrinsic_get_ubo_size', 'nir_intrinsic_get_var',
    'nir_intrinsic_global_atomic', 'nir_intrinsic_global_atomic_2x32',
    'nir_intrinsic_global_atomic_agx',
    'nir_intrinsic_global_atomic_amd',
    'nir_intrinsic_global_atomic_swap',
    'nir_intrinsic_global_atomic_swap_2x32',
    'nir_intrinsic_global_atomic_swap_agx',
    'nir_intrinsic_global_atomic_swap_amd', 'nir_intrinsic_has_align',
    'nir_intrinsic_has_semantic',
    'nir_intrinsic_ignore_ray_intersection',
    'nir_intrinsic_imadsp_nv', 'nir_intrinsic_image_atomic',
    'nir_intrinsic_image_atomic_swap',
    'nir_intrinsic_image_deref_atomic',
    'nir_intrinsic_image_deref_atomic_swap',
    'nir_intrinsic_image_deref_descriptor_amd',
    'nir_intrinsic_image_deref_format',
    'nir_intrinsic_image_deref_fragment_mask_load_amd',
    'nir_intrinsic_image_deref_levels',
    'nir_intrinsic_image_deref_load',
    'nir_intrinsic_image_deref_load_info_nv',
    'nir_intrinsic_image_deref_load_param_intel',
    'nir_intrinsic_image_deref_load_raw_intel',
    'nir_intrinsic_image_deref_order',
    'nir_intrinsic_image_deref_samples',
    'nir_intrinsic_image_deref_samples_identical',
    'nir_intrinsic_image_deref_size',
    'nir_intrinsic_image_deref_sparse_load',
    'nir_intrinsic_image_deref_store',
    'nir_intrinsic_image_deref_store_block_agx',
    'nir_intrinsic_image_deref_store_raw_intel',
    'nir_intrinsic_image_deref_texel_address',
    'nir_intrinsic_image_descriptor_amd',
    'nir_intrinsic_image_format',
    'nir_intrinsic_image_fragment_mask_load_amd',
    'nir_intrinsic_image_levels', 'nir_intrinsic_image_load',
    'nir_intrinsic_image_load_raw_intel', 'nir_intrinsic_image_order',
    'nir_intrinsic_image_samples',
    'nir_intrinsic_image_samples_identical',
    'nir_intrinsic_image_size', 'nir_intrinsic_image_sparse_load',
    'nir_intrinsic_image_store',
    'nir_intrinsic_image_store_block_agx',
    'nir_intrinsic_image_store_raw_intel',
    'nir_intrinsic_image_texel_address',
    'nir_intrinsic_inclusive_scan',
    'nir_intrinsic_inclusive_scan_clusters_ir3',
    'nir_intrinsic_index_flag',
    'nir_intrinsic_index_flag__enumvalues',
    'nir_intrinsic_index_names', 'nir_intrinsic_info',
    'nir_intrinsic_infos', 'nir_intrinsic_initialize_node_payloads',
    'nir_intrinsic_instr', 'nir_intrinsic_instr_create',
    'nir_intrinsic_instr_dest_type', 'nir_intrinsic_instr_src_type',
    'nir_intrinsic_interp_deref_at_centroid',
    'nir_intrinsic_interp_deref_at_offset',
    'nir_intrinsic_interp_deref_at_sample',
    'nir_intrinsic_interp_deref_at_vertex',
    'nir_intrinsic_inverse_ballot', 'nir_intrinsic_ipa_nv',
    'nir_intrinsic_is_helper_invocation',
    'nir_intrinsic_is_ray_query',
    'nir_intrinsic_is_sparse_resident_zink',
    'nir_intrinsic_is_sparse_texels_resident',
    'nir_intrinsic_is_subgroup_invocation_lt_amd',
    'nir_intrinsic_isberd_nv', 'nir_intrinsic_lane_permute_16_amd',
    'nir_intrinsic_last_invocation',
    'nir_intrinsic_launch_mesh_workgroups',
    'nir_intrinsic_launch_mesh_workgroups_with_payload_deref',
    'nir_intrinsic_ldc_nv', 'nir_intrinsic_ldcx_nv',
    'nir_intrinsic_ldtram_nv', 'nir_intrinsic_load_aa_line_width',
    'nir_intrinsic_load_accel_struct_amd',
    'nir_intrinsic_load_active_samples_agx',
    'nir_intrinsic_load_active_subgroup_count_agx',
    'nir_intrinsic_load_active_subgroup_invocation_agx',
    'nir_intrinsic_load_agx',
    'nir_intrinsic_load_alpha_reference_amd',
    'nir_intrinsic_load_api_sample_mask_agx',
    'nir_intrinsic_load_attrib_clamp_agx',
    'nir_intrinsic_load_attribute_pan',
    'nir_intrinsic_load_back_face_agx',
    'nir_intrinsic_load_barycentric_at_offset',
    'nir_intrinsic_load_barycentric_at_offset_nv',
    'nir_intrinsic_load_barycentric_at_sample',
    'nir_intrinsic_load_barycentric_centroid',
    'nir_intrinsic_load_barycentric_coord_at_offset',
    'nir_intrinsic_load_barycentric_coord_at_sample',
    'nir_intrinsic_load_barycentric_coord_centroid',
    'nir_intrinsic_load_barycentric_coord_pixel',
    'nir_intrinsic_load_barycentric_coord_sample',
    'nir_intrinsic_load_barycentric_model',
    'nir_intrinsic_load_barycentric_optimize_amd',
    'nir_intrinsic_load_barycentric_pixel',
    'nir_intrinsic_load_barycentric_sample',
    'nir_intrinsic_load_base_global_invocation_id',
    'nir_intrinsic_load_base_instance',
    'nir_intrinsic_load_base_vertex',
    'nir_intrinsic_load_base_workgroup_id',
    'nir_intrinsic_load_blend_const_color_a_float',
    'nir_intrinsic_load_blend_const_color_aaaa8888_unorm',
    'nir_intrinsic_load_blend_const_color_b_float',
    'nir_intrinsic_load_blend_const_color_g_float',
    'nir_intrinsic_load_blend_const_color_r_float',
    'nir_intrinsic_load_blend_const_color_rgba',
    'nir_intrinsic_load_blend_const_color_rgba8888_unorm',
    'nir_intrinsic_load_btd_global_arg_addr_intel',
    'nir_intrinsic_load_btd_local_arg_addr_intel',
    'nir_intrinsic_load_btd_resume_sbt_addr_intel',
    'nir_intrinsic_load_btd_shader_type_intel',
    'nir_intrinsic_load_btd_stack_id_intel',
    'nir_intrinsic_load_buffer_amd',
    'nir_intrinsic_load_callable_sbt_addr_intel',
    'nir_intrinsic_load_callable_sbt_stride_intel',
    'nir_intrinsic_load_clamp_vertex_color_amd',
    'nir_intrinsic_load_clip_half_line_width_amd',
    'nir_intrinsic_load_clip_z_coeff_agx',
    'nir_intrinsic_load_coalesced_input_count',
    'nir_intrinsic_load_coefficients_agx',
    'nir_intrinsic_load_color0', 'nir_intrinsic_load_color1',
    'nir_intrinsic_load_const_buf_base_addr_lvp',
    'nir_intrinsic_load_const_ir3', 'nir_intrinsic_load_constant',
    'nir_intrinsic_load_constant_agx',
    'nir_intrinsic_load_constant_base_ptr',
    'nir_intrinsic_load_converted_output_pan',
    'nir_intrinsic_load_core_count_arm', 'nir_intrinsic_load_core_id',
    'nir_intrinsic_load_core_max_id_arm',
    'nir_intrinsic_load_cull_any_enabled_amd',
    'nir_intrinsic_load_cull_back_face_enabled_amd',
    'nir_intrinsic_load_cull_ccw_amd',
    'nir_intrinsic_load_cull_front_face_enabled_amd',
    'nir_intrinsic_load_cull_line_viewport_xy_scale_and_offset_amd',
    'nir_intrinsic_load_cull_mask',
    'nir_intrinsic_load_cull_mask_and_flags_amd',
    'nir_intrinsic_load_cull_small_line_precision_amd',
    'nir_intrinsic_load_cull_small_lines_enabled_amd',
    'nir_intrinsic_load_cull_small_triangle_precision_amd',
    'nir_intrinsic_load_cull_small_triangles_enabled_amd',
    'nir_intrinsic_load_cull_triangle_viewport_xy_scale_and_offset_amd',
    'nir_intrinsic_load_debug_log_desc_amd',
    'nir_intrinsic_load_depth_never_agx', 'nir_intrinsic_load_deref',
    'nir_intrinsic_load_deref_block_intel',
    'nir_intrinsic_load_descriptor_set_agx',
    'nir_intrinsic_load_draw_id',
    'nir_intrinsic_load_esgs_vertex_stride_amd',
    'nir_intrinsic_load_exported_agx',
    'nir_intrinsic_load_fb_layers_v3d',
    'nir_intrinsic_load_fbfetch_image_desc_amd',
    'nir_intrinsic_load_fbfetch_image_fmask_desc_amd',
    'nir_intrinsic_load_fep_w_v3d', 'nir_intrinsic_load_first_vertex',
    'nir_intrinsic_load_fixed_point_size_agx',
    'nir_intrinsic_load_flat_mask',
    'nir_intrinsic_load_force_vrs_rates_amd',
    'nir_intrinsic_load_frag_coord',
    'nir_intrinsic_load_frag_coord_unscaled_ir3',
    'nir_intrinsic_load_frag_coord_w',
    'nir_intrinsic_load_frag_coord_z',
    'nir_intrinsic_load_frag_coord_zw_pan',
    'nir_intrinsic_load_frag_invocation_count',
    'nir_intrinsic_load_frag_offset_ir3',
    'nir_intrinsic_load_frag_shading_rate',
    'nir_intrinsic_load_frag_size',
    'nir_intrinsic_load_frag_size_ir3',
    'nir_intrinsic_load_from_texture_handle_agx',
    'nir_intrinsic_load_front_face',
    'nir_intrinsic_load_front_face_fsign',
    'nir_intrinsic_load_fs_input_interp_deltas',
    'nir_intrinsic_load_fs_msaa_intel',
    'nir_intrinsic_load_fully_covered',
    'nir_intrinsic_load_geometry_param_buffer_poly',
    'nir_intrinsic_load_global', 'nir_intrinsic_load_global_2x32',
    'nir_intrinsic_load_global_amd',
    'nir_intrinsic_load_global_base_ptr',
    'nir_intrinsic_load_global_block_intel',
    'nir_intrinsic_load_global_bounded',
    'nir_intrinsic_load_global_constant',
    'nir_intrinsic_load_global_constant_bounded',
    'nir_intrinsic_load_global_constant_offset',
    'nir_intrinsic_load_global_constant_uniform_block_intel',
    'nir_intrinsic_load_global_etna',
    'nir_intrinsic_load_global_invocation_id',
    'nir_intrinsic_load_global_invocation_index',
    'nir_intrinsic_load_global_ir3', 'nir_intrinsic_load_global_size',
    'nir_intrinsic_load_gs_header_ir3',
    'nir_intrinsic_load_gs_vertex_offset_amd',
    'nir_intrinsic_load_gs_wave_id_amd',
    'nir_intrinsic_load_helper_arg_hi_agx',
    'nir_intrinsic_load_helper_arg_lo_agx',
    'nir_intrinsic_load_helper_invocation',
    'nir_intrinsic_load_helper_op_id_agx',
    'nir_intrinsic_load_hit_attrib_amd',
    'nir_intrinsic_load_hs_out_patch_data_offset_amd',
    'nir_intrinsic_load_hs_patch_stride_ir3',
    'nir_intrinsic_load_initial_edgeflags_amd',
    'nir_intrinsic_load_inline_data_intel',
    'nir_intrinsic_load_input',
    'nir_intrinsic_load_input_assembly_buffer_poly',
    'nir_intrinsic_load_input_attachment_conv_pan',
    'nir_intrinsic_load_input_attachment_coord',
    'nir_intrinsic_load_input_attachment_target_pan',
    'nir_intrinsic_load_input_topology_poly',
    'nir_intrinsic_load_input_vertex',
    'nir_intrinsic_load_instance_id',
    'nir_intrinsic_load_interpolated_input',
    'nir_intrinsic_load_intersection_opaque_amd',
    'nir_intrinsic_load_invocation_id',
    'nir_intrinsic_load_is_first_fan_agx',
    'nir_intrinsic_load_is_indexed_draw',
    'nir_intrinsic_load_kernel_input', 'nir_intrinsic_load_layer_id',
    'nir_intrinsic_load_lds_ngg_gs_out_vertex_base_amd',
    'nir_intrinsic_load_leaf_opaque_intel',
    'nir_intrinsic_load_leaf_procedural_intel',
    'nir_intrinsic_load_line_coord', 'nir_intrinsic_load_line_width',
    'nir_intrinsic_load_local_invocation_id',
    'nir_intrinsic_load_local_invocation_index',
    'nir_intrinsic_load_local_pixel_agx',
    'nir_intrinsic_load_local_shared_r600',
    'nir_intrinsic_load_lshs_vertex_stride_amd',
    'nir_intrinsic_load_max_polygon_intel',
    'nir_intrinsic_load_merged_wave_info_amd',
    'nir_intrinsic_load_mesh_view_count',
    'nir_intrinsic_load_mesh_view_indices',
    'nir_intrinsic_load_multisampled_pan',
    'nir_intrinsic_load_noperspective_varyings_pan',
    'nir_intrinsic_load_num_subgroups',
    'nir_intrinsic_load_num_vertices',
    'nir_intrinsic_load_num_vertices_per_primitive_amd',
    'nir_intrinsic_load_num_workgroups',
    'nir_intrinsic_load_ordered_id_amd', 'nir_intrinsic_load_output',
    'nir_intrinsic_load_packed_passthrough_primitive_amd',
    'nir_intrinsic_load_param',
    'nir_intrinsic_load_patch_vertices_in',
    'nir_intrinsic_load_per_primitive_input',
    'nir_intrinsic_load_per_primitive_output',
    'nir_intrinsic_load_per_primitive_remap_intel',
    'nir_intrinsic_load_per_vertex_input',
    'nir_intrinsic_load_per_vertex_output',
    'nir_intrinsic_load_per_view_output',
    'nir_intrinsic_load_persp_center_rhw_ir3',
    'nir_intrinsic_load_pipeline_stat_query_enabled_amd',
    'nir_intrinsic_load_pixel_coord',
    'nir_intrinsic_load_point_coord',
    'nir_intrinsic_load_point_coord_maybe_flipped',
    'nir_intrinsic_load_poly_line_smooth_enabled',
    'nir_intrinsic_load_polygon_stipple_agx',
    'nir_intrinsic_load_polygon_stipple_buffer_amd',
    'nir_intrinsic_load_preamble',
    'nir_intrinsic_load_prim_gen_query_enabled_amd',
    'nir_intrinsic_load_prim_xfb_query_enabled_amd',
    'nir_intrinsic_load_primitive_id',
    'nir_intrinsic_load_primitive_location_ir3',
    'nir_intrinsic_load_printf_buffer_address',
    'nir_intrinsic_load_printf_buffer_size',
    'nir_intrinsic_load_provoking_last',
    'nir_intrinsic_load_provoking_vtx_amd',
    'nir_intrinsic_load_provoking_vtx_in_prim_amd',
    'nir_intrinsic_load_push_constant',
    'nir_intrinsic_load_push_constant_zink',
    'nir_intrinsic_load_r600_per_vertex_input',
    'nir_intrinsic_load_rasterization_primitive_amd',
    'nir_intrinsic_load_rasterization_samples_amd',
    'nir_intrinsic_load_rasterization_stream',
    'nir_intrinsic_load_raw_output_pan',
    'nir_intrinsic_load_raw_vertex_id_pan',
    'nir_intrinsic_load_raw_vertex_offset_pan',
    'nir_intrinsic_load_ray_base_mem_addr_intel',
    'nir_intrinsic_load_ray_flags',
    'nir_intrinsic_load_ray_geometry_index',
    'nir_intrinsic_load_ray_hit_kind',
    'nir_intrinsic_load_ray_hit_sbt_addr_intel',
    'nir_intrinsic_load_ray_hit_sbt_stride_intel',
    'nir_intrinsic_load_ray_hw_stack_size_intel',
    'nir_intrinsic_load_ray_instance_custom_index',
    'nir_intrinsic_load_ray_launch_id',
    'nir_intrinsic_load_ray_launch_size',
    'nir_intrinsic_load_ray_miss_sbt_addr_intel',
    'nir_intrinsic_load_ray_miss_sbt_stride_intel',
    'nir_intrinsic_load_ray_num_dss_rt_stacks_intel',
    'nir_intrinsic_load_ray_object_direction',
    'nir_intrinsic_load_ray_object_origin',
    'nir_intrinsic_load_ray_object_to_world',
    'nir_intrinsic_load_ray_query_global_intel',
    'nir_intrinsic_load_ray_sw_stack_size_intel',
    'nir_intrinsic_load_ray_t_max', 'nir_intrinsic_load_ray_t_min',
    'nir_intrinsic_load_ray_tracing_stack_base_lvp',
    'nir_intrinsic_load_ray_triangle_vertex_positions',
    'nir_intrinsic_load_ray_world_direction',
    'nir_intrinsic_load_ray_world_origin',
    'nir_intrinsic_load_ray_world_to_object',
    'nir_intrinsic_load_readonly_output_pan',
    'nir_intrinsic_load_reg', 'nir_intrinsic_load_reg_indirect',
    'nir_intrinsic_load_rel_patch_id_ir3',
    'nir_intrinsic_load_reloc_const_intel',
    'nir_intrinsic_load_resume_shader_address_amd',
    'nir_intrinsic_load_ring_attr_amd',
    'nir_intrinsic_load_ring_attr_offset_amd',
    'nir_intrinsic_load_ring_es2gs_offset_amd',
    'nir_intrinsic_load_ring_esgs_amd',
    'nir_intrinsic_load_ring_gs2vs_offset_amd',
    'nir_intrinsic_load_ring_gsvs_amd',
    'nir_intrinsic_load_ring_mesh_scratch_amd',
    'nir_intrinsic_load_ring_mesh_scratch_offset_amd',
    'nir_intrinsic_load_ring_task_draw_amd',
    'nir_intrinsic_load_ring_task_payload_amd',
    'nir_intrinsic_load_ring_tess_factors_amd',
    'nir_intrinsic_load_ring_tess_factors_offset_amd',
    'nir_intrinsic_load_ring_tess_offchip_amd',
    'nir_intrinsic_load_ring_tess_offchip_offset_amd',
    'nir_intrinsic_load_root_agx',
    'nir_intrinsic_load_rt_arg_scratch_offset_amd',
    'nir_intrinsic_load_rt_conversion_pan',
    'nir_intrinsic_load_sample_id', 'nir_intrinsic_load_sample_mask',
    'nir_intrinsic_load_sample_mask_in',
    'nir_intrinsic_load_sample_pos',
    'nir_intrinsic_load_sample_pos_from_id',
    'nir_intrinsic_load_sample_pos_or_center',
    'nir_intrinsic_load_sample_positions_agx',
    'nir_intrinsic_load_sample_positions_amd',
    'nir_intrinsic_load_sample_positions_pan',
    'nir_intrinsic_load_sampler_handle_agx',
    'nir_intrinsic_load_sampler_lod_parameters',
    'nir_intrinsic_load_samples_log2_agx',
    'nir_intrinsic_load_sbt_base_amd',
    'nir_intrinsic_load_sbt_offset_amd',
    'nir_intrinsic_load_sbt_stride_amd',
    'nir_intrinsic_load_scalar_arg_amd', 'nir_intrinsic_load_scratch',
    'nir_intrinsic_load_scratch_base_ptr',
    'nir_intrinsic_load_shader_call_data_offset_lvp',
    'nir_intrinsic_load_shader_index',
    'nir_intrinsic_load_shader_output_pan',
    'nir_intrinsic_load_shader_part_tests_zs_agx',
    'nir_intrinsic_load_shader_record_ptr',
    'nir_intrinsic_load_shared', 'nir_intrinsic_load_shared2_amd',
    'nir_intrinsic_load_shared_base_ptr',
    'nir_intrinsic_load_shared_block_intel',
    'nir_intrinsic_load_shared_ir3',
    'nir_intrinsic_load_shared_lock_nv',
    'nir_intrinsic_load_shared_uniform_block_intel',
    'nir_intrinsic_load_simd_width_intel',
    'nir_intrinsic_load_sm_count_nv', 'nir_intrinsic_load_sm_id_nv',
    'nir_intrinsic_load_smem_amd', 'nir_intrinsic_load_ssbo',
    'nir_intrinsic_load_ssbo_address',
    'nir_intrinsic_load_ssbo_block_intel',
    'nir_intrinsic_load_ssbo_intel', 'nir_intrinsic_load_ssbo_ir3',
    'nir_intrinsic_load_ssbo_uniform_block_intel',
    'nir_intrinsic_load_stack',
    'nir_intrinsic_load_stat_query_address_agx',
    'nir_intrinsic_load_streamout_buffer_amd',
    'nir_intrinsic_load_streamout_config_amd',
    'nir_intrinsic_load_streamout_offset_amd',
    'nir_intrinsic_load_streamout_write_index_amd',
    'nir_intrinsic_load_subgroup_eq_mask',
    'nir_intrinsic_load_subgroup_ge_mask',
    'nir_intrinsic_load_subgroup_gt_mask',
    'nir_intrinsic_load_subgroup_id',
    'nir_intrinsic_load_subgroup_id_shift_ir3',
    'nir_intrinsic_load_subgroup_invocation',
    'nir_intrinsic_load_subgroup_le_mask',
    'nir_intrinsic_load_subgroup_lt_mask',
    'nir_intrinsic_load_subgroup_size',
    'nir_intrinsic_load_sysval_agx', 'nir_intrinsic_load_sysval_nv',
    'nir_intrinsic_load_task_payload',
    'nir_intrinsic_load_task_ring_entry_amd',
    'nir_intrinsic_load_tcs_header_ir3',
    'nir_intrinsic_load_tcs_in_param_base_r600',
    'nir_intrinsic_load_tcs_mem_attrib_stride',
    'nir_intrinsic_load_tcs_num_patches_amd',
    'nir_intrinsic_load_tcs_out_param_base_r600',
    'nir_intrinsic_load_tcs_primitive_mode_amd',
    'nir_intrinsic_load_tcs_rel_patch_id_r600',
    'nir_intrinsic_load_tcs_tess_factor_base_r600',
    'nir_intrinsic_load_tcs_tess_levels_to_tes_amd',
    'nir_intrinsic_load_tess_coord',
    'nir_intrinsic_load_tess_coord_xy',
    'nir_intrinsic_load_tess_factor_base_ir3',
    'nir_intrinsic_load_tess_level_inner',
    'nir_intrinsic_load_tess_level_inner_default',
    'nir_intrinsic_load_tess_level_outer',
    'nir_intrinsic_load_tess_level_outer_default',
    'nir_intrinsic_load_tess_param_base_ir3',
    'nir_intrinsic_load_tess_param_buffer_poly',
    'nir_intrinsic_load_tess_rel_patch_id_amd',
    'nir_intrinsic_load_tex_sprite_mask_agx',
    'nir_intrinsic_load_texture_handle_agx',
    'nir_intrinsic_load_texture_scale',
    'nir_intrinsic_load_texture_size_etna',
    'nir_intrinsic_load_tlb_color_brcm',
    'nir_intrinsic_load_topology_id_intel',
    'nir_intrinsic_load_typed_buffer_amd',
    'nir_intrinsic_load_uav_ir3', 'nir_intrinsic_load_ubo',
    'nir_intrinsic_load_ubo_uniform_block_intel',
    'nir_intrinsic_load_ubo_vec4', 'nir_intrinsic_load_uniform',
    'nir_intrinsic_load_user_clip_plane',
    'nir_intrinsic_load_user_data_amd',
    'nir_intrinsic_load_uvs_index_agx',
    'nir_intrinsic_load_vbo_base_agx',
    'nir_intrinsic_load_vbo_stride_agx',
    'nir_intrinsic_load_vector_arg_amd',
    'nir_intrinsic_load_vertex_id',
    'nir_intrinsic_load_vertex_id_zero_base',
    'nir_intrinsic_load_view_index',
    'nir_intrinsic_load_viewport_offset',
    'nir_intrinsic_load_viewport_scale',
    'nir_intrinsic_load_viewport_x_offset',
    'nir_intrinsic_load_viewport_x_scale',
    'nir_intrinsic_load_viewport_y_offset',
    'nir_intrinsic_load_viewport_y_scale',
    'nir_intrinsic_load_viewport_z_offset',
    'nir_intrinsic_load_viewport_z_scale',
    'nir_intrinsic_load_vs_output_buffer_poly',
    'nir_intrinsic_load_vs_outputs_poly',
    'nir_intrinsic_load_vs_primitive_stride_ir3',
    'nir_intrinsic_load_vs_vertex_stride_ir3',
    'nir_intrinsic_load_vulkan_descriptor',
    'nir_intrinsic_load_warp_id_arm', 'nir_intrinsic_load_warp_id_nv',
    'nir_intrinsic_load_warp_max_id_arm',
    'nir_intrinsic_load_warps_per_sm_nv',
    'nir_intrinsic_load_work_dim', 'nir_intrinsic_load_workgroup_id',
    'nir_intrinsic_load_workgroup_index',
    'nir_intrinsic_load_workgroup_num_input_primitives_amd',
    'nir_intrinsic_load_workgroup_num_input_vertices_amd',
    'nir_intrinsic_load_workgroup_size',
    'nir_intrinsic_load_xfb_address',
    'nir_intrinsic_load_xfb_index_buffer',
    'nir_intrinsic_load_xfb_size',
    'nir_intrinsic_load_xfb_state_address_gfx12_amd',
    'nir_intrinsic_masked_swizzle_amd', 'nir_intrinsic_mbcnt_amd',
    'nir_intrinsic_memcpy_deref', 'nir_intrinsic_nop',
    'nir_intrinsic_nop_amd', 'nir_intrinsic_op',
    'nir_intrinsic_op__enumvalues',
    'nir_intrinsic_optimization_barrier_sgpr_amd',
    'nir_intrinsic_optimization_barrier_vgpr_amd',
    'nir_intrinsic_ordered_add_loop_gfx12_amd',
    'nir_intrinsic_ordered_xfb_counter_add_gfx11_amd',
    'nir_intrinsic_overwrite_tes_arguments_amd',
    'nir_intrinsic_overwrite_vs_arguments_amd',
    'nir_intrinsic_pass_cb', 'nir_intrinsic_pin_cx_handle_nv',
    'nir_intrinsic_preamble_end_ir3',
    'nir_intrinsic_preamble_start_ir3',
    'nir_intrinsic_prefetch_sam_ir3',
    'nir_intrinsic_prefetch_tex_ir3',
    'nir_intrinsic_prefetch_ubo_ir3', 'nir_intrinsic_printf',
    'nir_intrinsic_printf_abort', 'nir_intrinsic_quad_ballot_agx',
    'nir_intrinsic_quad_broadcast',
    'nir_intrinsic_quad_swap_diagonal',
    'nir_intrinsic_quad_swap_horizontal',
    'nir_intrinsic_quad_swap_vertical',
    'nir_intrinsic_quad_swizzle_amd', 'nir_intrinsic_quad_vote_all',
    'nir_intrinsic_quad_vote_any',
    'nir_intrinsic_r600_indirect_vertex_at_index',
    'nir_intrinsic_ray_intersection_ir3',
    'nir_intrinsic_read_attribute_payload_intel',
    'nir_intrinsic_read_first_invocation',
    'nir_intrinsic_read_getlast_ir3', 'nir_intrinsic_read_invocation',
    'nir_intrinsic_read_invocation_cond_ir3', 'nir_intrinsic_reduce',
    'nir_intrinsic_reduce_clusters_ir3',
    'nir_intrinsic_report_ray_intersection',
    'nir_intrinsic_resource_intel', 'nir_intrinsic_rotate',
    'nir_intrinsic_rq_confirm_intersection',
    'nir_intrinsic_rq_generate_intersection',
    'nir_intrinsic_rq_initialize', 'nir_intrinsic_rq_load',
    'nir_intrinsic_rq_proceed', 'nir_intrinsic_rq_terminate',
    'nir_intrinsic_rt_execute_callable', 'nir_intrinsic_rt_resume',
    'nir_intrinsic_rt_return_amd', 'nir_intrinsic_rt_trace_ray',
    'nir_intrinsic_sample_mask_agx',
    'nir_intrinsic_select_vertex_poly', 'nir_intrinsic_semantic_flag',
    'nir_intrinsic_semantic_flag__enumvalues',
    'nir_intrinsic_sendmsg_amd', 'nir_intrinsic_set_align',
    'nir_intrinsic_set_vertex_and_primitive_count',
    'nir_intrinsic_shader_clock', 'nir_intrinsic_shared_append_amd',
    'nir_intrinsic_shared_atomic', 'nir_intrinsic_shared_atomic_swap',
    'nir_intrinsic_shared_consume_amd', 'nir_intrinsic_shuffle',
    'nir_intrinsic_shuffle_down',
    'nir_intrinsic_shuffle_down_uniform_ir3',
    'nir_intrinsic_shuffle_up',
    'nir_intrinsic_shuffle_up_uniform_ir3',
    'nir_intrinsic_shuffle_xor',
    'nir_intrinsic_shuffle_xor_uniform_ir3',
    'nir_intrinsic_sleep_amd',
    'nir_intrinsic_sparse_residency_code_and',
    'nir_intrinsic_src_components', 'nir_intrinsic_ssa_bar_nv',
    'nir_intrinsic_ssbo_atomic', 'nir_intrinsic_ssbo_atomic_ir3',
    'nir_intrinsic_ssbo_atomic_swap',
    'nir_intrinsic_ssbo_atomic_swap_ir3',
    'nir_intrinsic_stack_map_agx', 'nir_intrinsic_stack_unmap_agx',
    'nir_intrinsic_store_agx', 'nir_intrinsic_store_buffer_amd',
    'nir_intrinsic_store_combined_output_pan',
    'nir_intrinsic_store_const_ir3', 'nir_intrinsic_store_deref',
    'nir_intrinsic_store_deref_block_intel',
    'nir_intrinsic_store_global', 'nir_intrinsic_store_global_2x32',
    'nir_intrinsic_store_global_amd',
    'nir_intrinsic_store_global_block_intel',
    'nir_intrinsic_store_global_etna',
    'nir_intrinsic_store_global_ir3',
    'nir_intrinsic_store_hit_attrib_amd',
    'nir_intrinsic_store_local_pixel_agx',
    'nir_intrinsic_store_local_shared_r600',
    'nir_intrinsic_store_output',
    'nir_intrinsic_store_per_primitive_output',
    'nir_intrinsic_store_per_primitive_payload_intel',
    'nir_intrinsic_store_per_vertex_output',
    'nir_intrinsic_store_per_view_output',
    'nir_intrinsic_store_preamble',
    'nir_intrinsic_store_raw_output_pan', 'nir_intrinsic_store_reg',
    'nir_intrinsic_store_reg_indirect',
    'nir_intrinsic_store_scalar_arg_amd',
    'nir_intrinsic_store_scratch', 'nir_intrinsic_store_shared',
    'nir_intrinsic_store_shared2_amd',
    'nir_intrinsic_store_shared_block_intel',
    'nir_intrinsic_store_shared_ir3',
    'nir_intrinsic_store_shared_unlock_nv',
    'nir_intrinsic_store_ssbo',
    'nir_intrinsic_store_ssbo_block_intel',
    'nir_intrinsic_store_ssbo_intel', 'nir_intrinsic_store_ssbo_ir3',
    'nir_intrinsic_store_stack', 'nir_intrinsic_store_task_payload',
    'nir_intrinsic_store_tf_r600',
    'nir_intrinsic_store_tlb_sample_color_v3d',
    'nir_intrinsic_store_uvs_agx',
    'nir_intrinsic_store_vector_arg_amd',
    'nir_intrinsic_store_zs_agx',
    'nir_intrinsic_strict_wqm_coord_amd', 'nir_intrinsic_subfm_nv',
    'nir_intrinsic_suclamp_nv', 'nir_intrinsic_sueau_nv',
    'nir_intrinsic_suldga_nv', 'nir_intrinsic_sustga_nv',
    'nir_intrinsic_task_payload_atomic',
    'nir_intrinsic_task_payload_atomic_swap',
    'nir_intrinsic_terminate', 'nir_intrinsic_terminate_if',
    'nir_intrinsic_terminate_ray', 'nir_intrinsic_trace_ray',
    'nir_intrinsic_trace_ray_intel', 'nir_intrinsic_unit_test_amd',
    'nir_intrinsic_unit_test_divergent_amd',
    'nir_intrinsic_unit_test_uniform_amd',
    'nir_intrinsic_unpin_cx_handle_nv', 'nir_intrinsic_use',
    'nir_intrinsic_vild_nv', 'nir_intrinsic_vote_all',
    'nir_intrinsic_vote_any', 'nir_intrinsic_vote_feq',
    'nir_intrinsic_vote_ieq', 'nir_intrinsic_vulkan_resource_index',
    'nir_intrinsic_vulkan_resource_reindex',
    'nir_intrinsic_write_invocation_amd',
    'nir_intrinsic_writes_external_memory',
    'nir_intrinsic_xfb_counter_sub_gfx11_amd',
    'nir_io_16bit_input_output_support',
    'nir_io_add_const_offset_to_base',
    'nir_io_add_intrinsic_xfb_info',
    'nir_io_always_interpolate_convergent_fs_inputs',
    'nir_io_compaction_groups_tes_inputs_into_pos_and_var_groups',
    'nir_io_compaction_rotates_color_channels',
    'nir_io_dont_use_pos_for_non_fs_varyings',
    'nir_io_has_flexible_input_interpolation_except_flat',
    'nir_io_has_intrinsics', 'nir_io_mediump_is_32bit',
    'nir_io_mix_convergent_flat_with_interpolated', 'nir_io_options',
    'nir_io_options__enumvalues', 'nir_io_prefer_scalar_fs_inputs',
    'nir_io_radv_intrinsic_component_workaround', 'nir_io_semantics',
    'nir_io_separate_clip_cull_distance_arrays',
    'nir_io_vectorizer_ignores_types', 'nir_io_xfb', 'nir_ior_imm',
    'nir_is_arrayed_io', 'nir_is_denorm_flush_to_zero',
    'nir_is_denorm_preserve', 'nir_is_float_control_inf_preserve',
    'nir_is_float_control_nan_preserve',
    'nir_is_float_control_signed_zero_inf_nan_preserve',
    'nir_is_float_control_signed_zero_preserve', 'nir_is_load_reg',
    'nir_is_output_load', 'nir_is_rounding_mode_rtne',
    'nir_is_rounding_mode_rtz', 'nir_is_same_comp_swizzle',
    'nir_is_sequential_comp_swizzle', 'nir_is_store_reg',
    'nir_ishl_imm', 'nir_ishr_imm', 'nir_isub_imm', 'nir_jump',
    'nir_jump_break', 'nir_jump_continue', 'nir_jump_goto',
    'nir_jump_goto_if', 'nir_jump_halt', 'nir_jump_instr',
    'nir_jump_instr_create', 'nir_jump_return', 'nir_jump_type',
    'nir_jump_type__enumvalues', 'nir_last_intrinsic',
    'nir_last_opcode', 'nir_legalize_16bit_sampler_srcs',
    'nir_link_opt_varyings', 'nir_link_shader_functions',
    'nir_link_varying_precision', 'nir_link_xfb_varyings',
    'nir_live_defs_impl', 'nir_load_array_var',
    'nir_load_array_var_imm', 'nir_load_barycentric',
    'nir_load_const_instr', 'nir_load_const_instr_create',
    'nir_load_deref', 'nir_load_deref_with_access', 'nir_load_global',
    'nir_load_global_constant', 'nir_load_grouping',
    'nir_load_grouping__enumvalues', 'nir_load_param', 'nir_load_reg',
    'nir_load_reg_for_def', 'nir_load_store_vectorize_options',
    'nir_load_system_value', 'nir_load_var',
    'nir_local_variable_create', 'nir_log_shader_annotated_tagged',
    'nir_loop', 'nir_loop_analyze_impl', 'nir_loop_continue_target',
    'nir_loop_control', 'nir_loop_control__enumvalues',
    'nir_loop_control_dont_unroll', 'nir_loop_control_none',
    'nir_loop_control_unroll', 'nir_loop_create',
    'nir_loop_first_block', 'nir_loop_first_continue_block',
    'nir_loop_has_continue_construct', 'nir_loop_induction_variable',
    'nir_loop_info', 'nir_loop_is_divergent', 'nir_loop_last_block',
    'nir_loop_last_continue_block', 'nir_loop_terminator',
    'nir_lower_64bit_phis', 'nir_lower_all_phis_to_scalar',
    'nir_lower_alpha_test', 'nir_lower_alpha_to_coverage',
    'nir_lower_alpha_to_one', 'nir_lower_alu',
    'nir_lower_alu_conversion_to_intrinsic',
    'nir_lower_alu_to_scalar', 'nir_lower_alu_vec8_16_srcs',
    'nir_lower_alu_width', 'nir_lower_amul',
    'nir_lower_array_deref_of_vec',
    'nir_lower_array_deref_of_vec_options',
    'nir_lower_array_deref_of_vec_options__enumvalues',
    'nir_lower_atomics', 'nir_lower_atomics_to_ssbo',
    'nir_lower_bcsel64', 'nir_lower_bit_count64',
    'nir_lower_bit_size', 'nir_lower_bit_size_callback',
    'nir_lower_bitfield_extract64', 'nir_lower_bitfield_reverse64',
    'nir_lower_bitmap', 'nir_lower_bitmap_options',
    'nir_lower_bool_to_bitsize', 'nir_lower_bool_to_float',
    'nir_lower_bool_to_int32', 'nir_lower_calls_to_builtins',
    'nir_lower_cl_images', 'nir_lower_clamp_color_outputs',
    'nir_lower_clip_cull_distance_array_vars',
    'nir_lower_clip_cull_distance_to_vec4s', 'nir_lower_clip_disable',
    'nir_lower_clip_fs', 'nir_lower_clip_gs', 'nir_lower_clip_halfz',
    'nir_lower_clip_vs', 'nir_lower_compute_system_values',
    'nir_lower_compute_system_values_options',
    'nir_lower_const_arrays_to_uniforms',
    'nir_lower_constant_convert_alu_types',
    'nir_lower_constant_to_temp', 'nir_lower_continue_constructs',
    'nir_lower_conv64', 'nir_lower_convert_alu_types',
    'nir_lower_dceil', 'nir_lower_ddiv',
    'nir_lower_default_point_size', 'nir_lower_demote_if_to_cf',
    'nir_lower_deref_copy_instr', 'nir_lower_dfloor',
    'nir_lower_dfract', 'nir_lower_direct_array_deref_of_vec_load',
    'nir_lower_direct_array_deref_of_vec_store',
    'nir_lower_discard_if', 'nir_lower_discard_if_options',
    'nir_lower_discard_if_options__enumvalues', 'nir_lower_divmod64',
    'nir_lower_dminmax', 'nir_lower_dmod', 'nir_lower_doubles',
    'nir_lower_doubles_op_to_options_mask',
    'nir_lower_doubles_options',
    'nir_lower_doubles_options__enumvalues', 'nir_lower_drawpixels',
    'nir_lower_drawpixels_options', 'nir_lower_drcp',
    'nir_lower_dround_even', 'nir_lower_drsq', 'nir_lower_dsat',
    'nir_lower_dsign', 'nir_lower_dsqrt', 'nir_lower_dsub',
    'nir_lower_dtrunc', 'nir_lower_explicit_io',
    'nir_lower_explicit_io_instr', 'nir_lower_extract64',
    'nir_lower_fb_read', 'nir_lower_find_lsb64',
    'nir_lower_flatshade', 'nir_lower_flrp', 'nir_lower_fp16_all',
    'nir_lower_fp16_cast_options',
    'nir_lower_fp16_cast_options__enumvalues', 'nir_lower_fp16_casts',
    'nir_lower_fp16_rd', 'nir_lower_fp16_rtne', 'nir_lower_fp16_rtz',
    'nir_lower_fp16_ru', 'nir_lower_fp16_split_fp64',
    'nir_lower_fp64_full_software',
    'nir_lower_frag_coord_to_pixel_coord', 'nir_lower_fragcolor',
    'nir_lower_fragcoord_wtrans', 'nir_lower_frexp',
    'nir_lower_global_vars_to_local', 'nir_lower_goto_ifs',
    'nir_lower_gs_intrinsics',
    'nir_lower_gs_intrinsics_count_primitives',
    'nir_lower_gs_intrinsics_count_vertices_per_primitive',
    'nir_lower_gs_intrinsics_flags',
    'nir_lower_gs_intrinsics_flags__enumvalues',
    'nir_lower_gs_intrinsics_overwrite_incomplete',
    'nir_lower_gs_intrinsics_per_stream', 'nir_lower_halt_to_return',
    'nir_lower_helper_writes', 'nir_lower_iabs64',
    'nir_lower_iadd3_64', 'nir_lower_iadd64', 'nir_lower_iadd_sat64',
    'nir_lower_icmp64', 'nir_lower_idiv', 'nir_lower_idiv_options',
    'nir_lower_image', 'nir_lower_image_atomics_to_global',
    'nir_lower_image_options', 'nir_lower_imul64',
    'nir_lower_imul_2x32_64', 'nir_lower_imul_high64',
    'nir_lower_indirect_array_deref_of_vec_load',
    'nir_lower_indirect_array_deref_of_vec_store',
    'nir_lower_indirect_derefs', 'nir_lower_indirect_var_derefs',
    'nir_lower_ineg64', 'nir_lower_input_attachments',
    'nir_lower_instr_cb', 'nir_lower_int64',
    'nir_lower_int64_float_conversions',
    'nir_lower_int64_op_to_options_mask', 'nir_lower_int64_options',
    'nir_lower_int64_options__enumvalues', 'nir_lower_int_to_float',
    'nir_lower_interpolation', 'nir_lower_interpolation_at_offset',
    'nir_lower_interpolation_at_sample',
    'nir_lower_interpolation_centroid',
    'nir_lower_interpolation_options',
    'nir_lower_interpolation_options__enumvalues',
    'nir_lower_interpolation_pixel', 'nir_lower_interpolation_sample',
    'nir_lower_io', 'nir_lower_io_array_vars_to_elements',
    'nir_lower_io_array_vars_to_elements_no_indirects',
    'nir_lower_io_indirect_loads',
    'nir_lower_io_lower_64bit_float_to_32',
    'nir_lower_io_lower_64bit_to_32',
    'nir_lower_io_lower_64bit_to_32_new', 'nir_lower_io_options',
    'nir_lower_io_options__enumvalues', 'nir_lower_io_passes',
    'nir_lower_io_to_scalar',
    'nir_lower_io_use_interpolated_input_intrinsics',
    'nir_lower_io_vars_to_scalar', 'nir_lower_io_vars_to_temporaries',
    'nir_lower_is_helper_invocation', 'nir_lower_isign64',
    'nir_lower_load_const_to_scalar', 'nir_lower_locals_to_regs',
    'nir_lower_logic64', 'nir_lower_mediump_io',
    'nir_lower_mediump_vars', 'nir_lower_mem_access_bit_sizes',
    'nir_lower_mem_access_bit_sizes_cb',
    'nir_lower_mem_access_bit_sizes_options', 'nir_lower_memcpy',
    'nir_lower_memory_model', 'nir_lower_minmax64',
    'nir_lower_multiview', 'nir_lower_multiview_options',
    'nir_lower_non_uniform_access',
    'nir_lower_non_uniform_access_callback',
    'nir_lower_non_uniform_access_options',
    'nir_lower_non_uniform_access_type',
    'nir_lower_non_uniform_access_type_count',
    'nir_lower_non_uniform_get_ssbo_size',
    'nir_lower_non_uniform_image_access',
    'nir_lower_non_uniform_src_access_callback',
    'nir_lower_non_uniform_ssbo_access',
    'nir_lower_non_uniform_texture_access',
    'nir_lower_non_uniform_texture_offset_access',
    'nir_lower_non_uniform_ubo_access', 'nir_lower_pack',
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
    'nir_lower_passthrough_edgeflags', 'nir_lower_patch_vertices',
    'nir_lower_phis_to_regs_block', 'nir_lower_phis_to_scalar',
    'nir_lower_pntc_ytransform', 'nir_lower_point_size',
    'nir_lower_point_size_mov', 'nir_lower_point_smooth',
    'nir_lower_poly_line_smooth', 'nir_lower_printf',
    'nir_lower_printf_options', 'nir_lower_read_invocation_to_scalar',
    'nir_lower_readonly_images_to_tex',
    'nir_lower_reg_intrinsics_to_ssa',
    'nir_lower_reg_intrinsics_to_ssa_impl', 'nir_lower_returns',
    'nir_lower_returns_impl', 'nir_lower_robust_access',
    'nir_lower_sample_shading', 'nir_lower_samplers',
    'nir_lower_scan_reduce_bitwise64', 'nir_lower_scan_reduce_iadd64',
    'nir_lower_scratch_to_var', 'nir_lower_shader_calls',
    'nir_lower_shader_calls_options',
    'nir_lower_shader_calls_should_remat_func', 'nir_lower_shift64',
    'nir_lower_single_sampled', 'nir_lower_ssa_defs_to_regs_block',
    'nir_lower_ssbo', 'nir_lower_ssbo_options',
    'nir_lower_subgroup_shuffle64', 'nir_lower_subgroups',
    'nir_lower_subgroups_options', 'nir_lower_system_values',
    'nir_lower_sysvals_to_varyings',
    'nir_lower_sysvals_to_varyings_options', 'nir_lower_task_shader',
    'nir_lower_task_shader_options', 'nir_lower_terminate_if_to_cf',
    'nir_lower_terminate_to_demote', 'nir_lower_tess_coord_z',
    'nir_lower_tess_level_array_vars_to_vec', 'nir_lower_tex',
    'nir_lower_tex_options', 'nir_lower_tex_packing',
    'nir_lower_tex_packing_16', 'nir_lower_tex_packing_8',
    'nir_lower_tex_packing_none', 'nir_lower_tex_shadow',
    'nir_lower_tex_shadow_swizzle', 'nir_lower_texcoord_replace',
    'nir_lower_texcoord_replace_late', 'nir_lower_two_sided_color',
    'nir_lower_uadd_sat64', 'nir_lower_ubo_vec4',
    'nir_lower_ufind_msb64', 'nir_lower_undef_to_zero',
    'nir_lower_uniforms_to_ubo', 'nir_lower_usub_sat64',
    'nir_lower_var_copies', 'nir_lower_var_copy_instr',
    'nir_lower_variable_initializers',
    'nir_lower_vars_to_explicit_types', 'nir_lower_vars_to_scratch',
    'nir_lower_vars_to_ssa', 'nir_lower_vec3_to_vec4',
    'nir_lower_vec_to_regs', 'nir_lower_view_index_to_device_index',
    'nir_lower_viewport_transform', 'nir_lower_vote_ieq64',
    'nir_lower_wpos_center', 'nir_lower_wpos_ytransform',
    'nir_lower_wpos_ytransform_options', 'nir_lower_wrmasks',
    'nir_mask', 'nir_mem_access_shift_method',
    'nir_mem_access_shift_method__enumvalues',
    'nir_mem_access_shift_method_bytealign_amd',
    'nir_mem_access_shift_method_scalar',
    'nir_mem_access_shift_method_shift64',
    'nir_mem_access_size_align', 'nir_memcpy_deref',
    'nir_memcpy_deref_with_access', 'nir_memory_semantics',
    'nir_memory_semantics__enumvalues', 'nir_metadata',
    'nir_metadata__enumvalues', 'nir_metadata_all',
    'nir_metadata_block_index', 'nir_metadata_check_validation_flag',
    'nir_metadata_control_flow', 'nir_metadata_divergence',
    'nir_metadata_dominance', 'nir_metadata_instr_index',
    'nir_metadata_invalidate', 'nir_metadata_live_defs',
    'nir_metadata_loop_analysis', 'nir_metadata_none',
    'nir_metadata_not_properly_reset', 'nir_metadata_require',
    'nir_metadata_require_all', 'nir_metadata_set_validation_flag',
    'nir_minimize_call_live_states', 'nir_mod_analysis',
    'nir_mov_alu', 'nir_mov_scalar', 'nir_move_alu',
    'nir_move_comparisons', 'nir_move_const_undef', 'nir_move_copies',
    'nir_move_load_buffer_amd', 'nir_move_load_frag_coord',
    'nir_move_load_global', 'nir_move_load_image',
    'nir_move_load_image_fragment_mask', 'nir_move_load_input',
    'nir_move_load_ssbo', 'nir_move_load_ubo',
    'nir_move_load_uniform', 'nir_move_only_convergent',
    'nir_move_only_divergent', 'nir_move_options',
    'nir_move_options__enumvalues', 'nir_move_output_stores_to_end',
    'nir_move_query_image', 'nir_move_terminate_out_of_loops',
    'nir_move_tex_load', 'nir_move_tex_load_fragment_mask',
    'nir_move_tex_lod', 'nir_move_tex_query', 'nir_move_tex_sample',
    'nir_move_to_entry_block_only', 'nir_move_to_top_input_loads',
    'nir_move_to_top_load_smem_amd', 'nir_move_vec_src_uses_to_dest',
    'nir_next_decl_reg', 'nir_next_phi', 'nir_no_progress',
    'nir_normalize_cubemap_coords', 'nir_num_intrinsics',
    'nir_num_opcodes', 'nir_num_tex_src_types',
    'nir_num_variable_modes', 'nir_op', 'nir_op__enumvalues',
    'nir_op_algebraic_property',
    'nir_op_algebraic_property__enumvalues', 'nir_op_alignbyte_amd',
    'nir_op_amul', 'nir_op_andg_ir3', 'nir_op_b16all_fequal16',
    'nir_op_b16all_fequal2', 'nir_op_b16all_fequal3',
    'nir_op_b16all_fequal4', 'nir_op_b16all_fequal5',
    'nir_op_b16all_fequal8', 'nir_op_b16all_iequal16',
    'nir_op_b16all_iequal2', 'nir_op_b16all_iequal3',
    'nir_op_b16all_iequal4', 'nir_op_b16all_iequal5',
    'nir_op_b16all_iequal8', 'nir_op_b16any_fnequal16',
    'nir_op_b16any_fnequal2', 'nir_op_b16any_fnequal3',
    'nir_op_b16any_fnequal4', 'nir_op_b16any_fnequal5',
    'nir_op_b16any_fnequal8', 'nir_op_b16any_inequal16',
    'nir_op_b16any_inequal2', 'nir_op_b16any_inequal3',
    'nir_op_b16any_inequal4', 'nir_op_b16any_inequal5',
    'nir_op_b16any_inequal8', 'nir_op_b16csel', 'nir_op_b2b1',
    'nir_op_b2b16', 'nir_op_b2b32', 'nir_op_b2b8', 'nir_op_b2f16',
    'nir_op_b2f32', 'nir_op_b2f64', 'nir_op_b2i1', 'nir_op_b2i16',
    'nir_op_b2i32', 'nir_op_b2i64', 'nir_op_b2i8',
    'nir_op_b32all_fequal16', 'nir_op_b32all_fequal2',
    'nir_op_b32all_fequal3', 'nir_op_b32all_fequal4',
    'nir_op_b32all_fequal5', 'nir_op_b32all_fequal8',
    'nir_op_b32all_iequal16', 'nir_op_b32all_iequal2',
    'nir_op_b32all_iequal3', 'nir_op_b32all_iequal4',
    'nir_op_b32all_iequal5', 'nir_op_b32all_iequal8',
    'nir_op_b32any_fnequal16', 'nir_op_b32any_fnequal2',
    'nir_op_b32any_fnequal3', 'nir_op_b32any_fnequal4',
    'nir_op_b32any_fnequal5', 'nir_op_b32any_fnequal8',
    'nir_op_b32any_inequal16', 'nir_op_b32any_inequal2',
    'nir_op_b32any_inequal3', 'nir_op_b32any_inequal4',
    'nir_op_b32any_inequal5', 'nir_op_b32any_inequal8',
    'nir_op_b32csel', 'nir_op_b32fcsel_mdg', 'nir_op_b8all_fequal16',
    'nir_op_b8all_fequal2', 'nir_op_b8all_fequal3',
    'nir_op_b8all_fequal4', 'nir_op_b8all_fequal5',
    'nir_op_b8all_fequal8', 'nir_op_b8all_iequal16',
    'nir_op_b8all_iequal2', 'nir_op_b8all_iequal3',
    'nir_op_b8all_iequal4', 'nir_op_b8all_iequal5',
    'nir_op_b8all_iequal8', 'nir_op_b8any_fnequal16',
    'nir_op_b8any_fnequal2', 'nir_op_b8any_fnequal3',
    'nir_op_b8any_fnequal4', 'nir_op_b8any_fnequal5',
    'nir_op_b8any_fnequal8', 'nir_op_b8any_inequal16',
    'nir_op_b8any_inequal2', 'nir_op_b8any_inequal3',
    'nir_op_b8any_inequal4', 'nir_op_b8any_inequal5',
    'nir_op_b8any_inequal8', 'nir_op_b8csel', 'nir_op_ball_fequal16',
    'nir_op_ball_fequal2', 'nir_op_ball_fequal3',
    'nir_op_ball_fequal4', 'nir_op_ball_fequal5',
    'nir_op_ball_fequal8', 'nir_op_ball_iequal16',
    'nir_op_ball_iequal2', 'nir_op_ball_iequal3',
    'nir_op_ball_iequal4', 'nir_op_ball_iequal5',
    'nir_op_ball_iequal8', 'nir_op_bany_fnequal16',
    'nir_op_bany_fnequal2', 'nir_op_bany_fnequal3',
    'nir_op_bany_fnequal4', 'nir_op_bany_fnequal5',
    'nir_op_bany_fnequal8', 'nir_op_bany_inequal16',
    'nir_op_bany_inequal2', 'nir_op_bany_inequal3',
    'nir_op_bany_inequal4', 'nir_op_bany_inequal5',
    'nir_op_bany_inequal8', 'nir_op_bcsel', 'nir_op_bf2f',
    'nir_op_bfdot16', 'nir_op_bfdot2', 'nir_op_bfdot2_bfadd',
    'nir_op_bfdot3', 'nir_op_bfdot4', 'nir_op_bfdot5',
    'nir_op_bfdot8', 'nir_op_bffma', 'nir_op_bfi', 'nir_op_bfm',
    'nir_op_bfmul', 'nir_op_bit_count', 'nir_op_bitfield_insert',
    'nir_op_bitfield_reverse', 'nir_op_bitfield_select',
    'nir_op_bitnz', 'nir_op_bitnz16', 'nir_op_bitnz32',
    'nir_op_bitnz8', 'nir_op_bitz', 'nir_op_bitz16', 'nir_op_bitz32',
    'nir_op_bitz8', 'nir_op_bounds_agx', 'nir_op_byte_perm_amd',
    'nir_op_cube_amd', 'nir_op_e4m3fn2f', 'nir_op_e5m22f',
    'nir_op_extr_agx', 'nir_op_extract_i16', 'nir_op_extract_i8',
    'nir_op_extract_u16', 'nir_op_extract_u8', 'nir_op_f2bf',
    'nir_op_f2e4m3fn', 'nir_op_f2e4m3fn_sat', 'nir_op_f2e4m3fn_satfn',
    'nir_op_f2e5m2', 'nir_op_f2e5m2_sat', 'nir_op_f2f16',
    'nir_op_f2f16_rtne', 'nir_op_f2f16_rtz', 'nir_op_f2f32',
    'nir_op_f2f64', 'nir_op_f2fmp', 'nir_op_f2i1', 'nir_op_f2i16',
    'nir_op_f2i32', 'nir_op_f2i64', 'nir_op_f2i8', 'nir_op_f2imp',
    'nir_op_f2snorm_16_v3d', 'nir_op_f2u1', 'nir_op_f2u16',
    'nir_op_f2u32', 'nir_op_f2u64', 'nir_op_f2u8', 'nir_op_f2ump',
    'nir_op_f2unorm_16_v3d', 'nir_op_fabs', 'nir_op_fadd',
    'nir_op_fall_equal16', 'nir_op_fall_equal2', 'nir_op_fall_equal3',
    'nir_op_fall_equal4', 'nir_op_fall_equal5', 'nir_op_fall_equal8',
    'nir_op_fany_nequal16', 'nir_op_fany_nequal2',
    'nir_op_fany_nequal3', 'nir_op_fany_nequal4',
    'nir_op_fany_nequal5', 'nir_op_fany_nequal8', 'nir_op_fceil',
    'nir_op_fclamp_pos', 'nir_op_fcos', 'nir_op_fcos_amd',
    'nir_op_fcos_mdg', 'nir_op_fcsel', 'nir_op_fcsel_ge',
    'nir_op_fcsel_gt', 'nir_op_fdiv', 'nir_op_fdot16',
    'nir_op_fdot16_replicated', 'nir_op_fdot2',
    'nir_op_fdot2_replicated', 'nir_op_fdot3',
    'nir_op_fdot3_replicated', 'nir_op_fdot4',
    'nir_op_fdot4_replicated', 'nir_op_fdot5',
    'nir_op_fdot5_replicated', 'nir_op_fdot8',
    'nir_op_fdot8_replicated', 'nir_op_fdph',
    'nir_op_fdph_replicated', 'nir_op_feq', 'nir_op_feq16',
    'nir_op_feq32', 'nir_op_feq8', 'nir_op_fequ', 'nir_op_fequ16',
    'nir_op_fequ32', 'nir_op_fequ8', 'nir_op_fexp2', 'nir_op_ffloor',
    'nir_op_ffma', 'nir_op_ffmaz', 'nir_op_ffract', 'nir_op_fge',
    'nir_op_fge16', 'nir_op_fge32', 'nir_op_fge8', 'nir_op_fgeu',
    'nir_op_fgeu16', 'nir_op_fgeu32', 'nir_op_fgeu8',
    'nir_op_find_lsb', 'nir_op_fisfinite', 'nir_op_fisfinite32',
    'nir_op_fisnormal', 'nir_op_flog2', 'nir_op_flrp', 'nir_op_flt',
    'nir_op_flt16', 'nir_op_flt32', 'nir_op_flt8', 'nir_op_fltu',
    'nir_op_fltu16', 'nir_op_fltu32', 'nir_op_fltu8', 'nir_op_fmax',
    'nir_op_fmax_agx', 'nir_op_fmin', 'nir_op_fmin_agx',
    'nir_op_fmod', 'nir_op_fmul', 'nir_op_fmulz', 'nir_op_fneg',
    'nir_op_fneo', 'nir_op_fneo16', 'nir_op_fneo32', 'nir_op_fneo8',
    'nir_op_fneu', 'nir_op_fneu16', 'nir_op_fneu32', 'nir_op_fneu8',
    'nir_op_ford', 'nir_op_ford16', 'nir_op_ford32', 'nir_op_ford8',
    'nir_op_fpow', 'nir_op_fquantize2f16', 'nir_op_frcp',
    'nir_op_frem', 'nir_op_frexp_exp', 'nir_op_frexp_sig',
    'nir_op_fround_even', 'nir_op_frsq', 'nir_op_fsat',
    'nir_op_fsat_signed', 'nir_op_fsign', 'nir_op_fsin',
    'nir_op_fsin_agx', 'nir_op_fsin_amd', 'nir_op_fsin_mdg',
    'nir_op_fsqrt', 'nir_op_fsub', 'nir_op_fsum2', 'nir_op_fsum3',
    'nir_op_fsum4', 'nir_op_ftrunc', 'nir_op_funord',
    'nir_op_funord16', 'nir_op_funord32', 'nir_op_funord8',
    'nir_op_i2f16', 'nir_op_i2f32', 'nir_op_i2f64', 'nir_op_i2fmp',
    'nir_op_i2i1', 'nir_op_i2i16', 'nir_op_i2i32', 'nir_op_i2i64',
    'nir_op_i2i8', 'nir_op_i2imp', 'nir_op_i32csel_ge',
    'nir_op_i32csel_gt', 'nir_op_iabs', 'nir_op_iadd', 'nir_op_iadd3',
    'nir_op_iadd_sat', 'nir_op_iand', 'nir_op_ibfe',
    'nir_op_ibitfield_extract', 'nir_op_icsel_eqz', 'nir_op_idiv',
    'nir_op_ieq', 'nir_op_ieq16', 'nir_op_ieq32', 'nir_op_ieq8',
    'nir_op_ifind_msb', 'nir_op_ifind_msb_rev', 'nir_op_ige',
    'nir_op_ige16', 'nir_op_ige32', 'nir_op_ige8', 'nir_op_ihadd',
    'nir_op_ilea_agx', 'nir_op_ilt', 'nir_op_ilt16', 'nir_op_ilt32',
    'nir_op_ilt8', 'nir_op_imad', 'nir_op_imad24_ir3',
    'nir_op_imadsh_mix16', 'nir_op_imadshl_agx', 'nir_op_imax',
    'nir_op_imin', 'nir_op_imod', 'nir_op_imsubshl_agx',
    'nir_op_imul', 'nir_op_imul24', 'nir_op_imul24_relaxed',
    'nir_op_imul_2x32_64', 'nir_op_imul_32x16', 'nir_op_imul_high',
    'nir_op_ine', 'nir_op_ine16', 'nir_op_ine32', 'nir_op_ine8',
    'nir_op_ineg', 'nir_op_info', 'nir_op_infos', 'nir_op_inot',
    'nir_op_insert_u16', 'nir_op_insert_u8', 'nir_op_interleave_agx',
    'nir_op_ior', 'nir_op_irem', 'nir_op_irhadd',
    'nir_op_is_selection', 'nir_op_is_vec', 'nir_op_is_vec_or_mov',
    'nir_op_ishl', 'nir_op_ishr', 'nir_op_isign', 'nir_op_isub',
    'nir_op_isub_sat', 'nir_op_ixor', 'nir_op_ldexp',
    'nir_op_ldexp16_pan', 'nir_op_lea_nv', 'nir_op_mov',
    'nir_op_mqsad_4x8', 'nir_op_msad_4x8',
    'nir_op_pack_2x16_to_snorm_2x8_v3d',
    'nir_op_pack_2x16_to_unorm_10_2_v3d',
    'nir_op_pack_2x16_to_unorm_2x10_v3d',
    'nir_op_pack_2x16_to_unorm_2x8_v3d',
    'nir_op_pack_2x32_to_2x16_v3d', 'nir_op_pack_32_2x16',
    'nir_op_pack_32_2x16_split', 'nir_op_pack_32_4x8',
    'nir_op_pack_32_4x8_split', 'nir_op_pack_32_to_r11g11b10_v3d',
    'nir_op_pack_4x16_to_4x8_v3d', 'nir_op_pack_64_2x32',
    'nir_op_pack_64_2x32_split', 'nir_op_pack_64_4x16',
    'nir_op_pack_double_2x32_dxil', 'nir_op_pack_half_2x16',
    'nir_op_pack_half_2x16_rtz_split', 'nir_op_pack_half_2x16_split',
    'nir_op_pack_sint_2x16', 'nir_op_pack_snorm_2x16',
    'nir_op_pack_snorm_4x8', 'nir_op_pack_uint_2x16',
    'nir_op_pack_uint_32_to_r10g10b10a2_v3d',
    'nir_op_pack_unorm_2x16', 'nir_op_pack_unorm_4x8',
    'nir_op_pack_uvec2_to_uint', 'nir_op_pack_uvec4_to_uint',
    'nir_op_prmt_nv', 'nir_op_sdot_2x16_iadd',
    'nir_op_sdot_2x16_iadd_sat', 'nir_op_sdot_4x8_iadd',
    'nir_op_sdot_4x8_iadd_sat', 'nir_op_seq', 'nir_op_sge',
    'nir_op_shfr', 'nir_op_shlg_ir3', 'nir_op_shlm_ir3',
    'nir_op_shrg_ir3', 'nir_op_shrm_ir3', 'nir_op_slt', 'nir_op_sne',
    'nir_op_sudot_4x8_iadd', 'nir_op_sudot_4x8_iadd_sat',
    'nir_op_u2f16', 'nir_op_u2f32', 'nir_op_u2f64', 'nir_op_u2fmp',
    'nir_op_u2u1', 'nir_op_u2u16', 'nir_op_u2u32', 'nir_op_u2u64',
    'nir_op_u2u8', 'nir_op_uabs_isub', 'nir_op_uabs_usub',
    'nir_op_uadd_carry', 'nir_op_uadd_sat', 'nir_op_ubfe',
    'nir_op_ubitfield_extract', 'nir_op_uclz', 'nir_op_udiv',
    'nir_op_udiv_aligned_4', 'nir_op_udot_2x16_uadd',
    'nir_op_udot_2x16_uadd_sat', 'nir_op_udot_4x8_uadd',
    'nir_op_udot_4x8_uadd_sat', 'nir_op_ufind_msb',
    'nir_op_ufind_msb_rev', 'nir_op_uge', 'nir_op_uge16',
    'nir_op_uge32', 'nir_op_uge8', 'nir_op_uhadd', 'nir_op_ulea_agx',
    'nir_op_ult', 'nir_op_ult16', 'nir_op_ult32', 'nir_op_ult8',
    'nir_op_umad24', 'nir_op_umad24_relaxed', 'nir_op_umax',
    'nir_op_umax_4x8_vc4', 'nir_op_umin', 'nir_op_umin_4x8_vc4',
    'nir_op_umod', 'nir_op_umul24', 'nir_op_umul24_relaxed',
    'nir_op_umul_2x32_64', 'nir_op_umul_32x16', 'nir_op_umul_high',
    'nir_op_umul_low', 'nir_op_umul_unorm_4x8_vc4',
    'nir_op_unpack_32_2x16', 'nir_op_unpack_32_2x16_split_x',
    'nir_op_unpack_32_2x16_split_y', 'nir_op_unpack_32_4x8',
    'nir_op_unpack_64_2x32', 'nir_op_unpack_64_2x32_split_x',
    'nir_op_unpack_64_2x32_split_y', 'nir_op_unpack_64_4x16',
    'nir_op_unpack_double_2x32_dxil', 'nir_op_unpack_half_2x16',
    'nir_op_unpack_half_2x16_split_x',
    'nir_op_unpack_half_2x16_split_y', 'nir_op_unpack_snorm_2x16',
    'nir_op_unpack_snorm_4x8', 'nir_op_unpack_unorm_2x16',
    'nir_op_unpack_unorm_4x8', 'nir_op_urhadd', 'nir_op_urol',
    'nir_op_uror', 'nir_op_usadd_4x8_vc4', 'nir_op_ushr',
    'nir_op_ussub_4x8_vc4', 'nir_op_usub_borrow', 'nir_op_usub_sat',
    'nir_op_vec', 'nir_op_vec16', 'nir_op_vec2', 'nir_op_vec3',
    'nir_op_vec4', 'nir_op_vec5', 'nir_op_vec8',
    'nir_opt_16bit_tex_image', 'nir_opt_16bit_tex_image_options',
    'nir_opt_access', 'nir_opt_access_options',
    'nir_opt_acquire_release_barriers', 'nir_opt_algebraic',
    'nir_opt_algebraic_before_ffma',
    'nir_opt_algebraic_before_lower_int64',
    'nir_opt_algebraic_distribute_src_mods',
    'nir_opt_algebraic_integer_promotion', 'nir_opt_algebraic_late',
    'nir_opt_barrier_modes', 'nir_opt_clip_cull_const',
    'nir_opt_combine_barriers', 'nir_opt_combine_stores',
    'nir_opt_comparison_pre', 'nir_opt_comparison_pre_impl',
    'nir_opt_constant_folding', 'nir_opt_copy_prop_vars',
    'nir_opt_cse', 'nir_opt_dce', 'nir_opt_dead_cf',
    'nir_opt_dead_write_vars', 'nir_opt_deref', 'nir_opt_deref_impl',
    'nir_opt_find_array_copies', 'nir_opt_frag_coord_to_pixel_coord',
    'nir_opt_fragdepth', 'nir_opt_gcm', 'nir_opt_generate_bfi',
    'nir_opt_group_loads', 'nir_opt_idiv_const', 'nir_opt_if',
    'nir_opt_if_avoid_64bit_phis',
    'nir_opt_if_optimize_phi_true_false', 'nir_opt_if_options',
    'nir_opt_if_options__enumvalues', 'nir_opt_intrinsics',
    'nir_opt_large_constants', 'nir_opt_licm',
    'nir_opt_load_skip_helpers', 'nir_opt_load_skip_helpers_options',
    'nir_opt_load_store_update_alignments',
    'nir_opt_load_store_vectorize', 'nir_opt_loop',
    'nir_opt_loop_unroll', 'nir_opt_memcpy', 'nir_opt_move',
    'nir_opt_move_discards_to_top', 'nir_opt_move_to_top',
    'nir_opt_move_to_top_options',
    'nir_opt_move_to_top_options__enumvalues', 'nir_opt_mqsad',
    'nir_opt_non_uniform_access', 'nir_opt_offsets',
    'nir_opt_offsets_options', 'nir_opt_peephole_select',
    'nir_opt_peephole_select_options', 'nir_opt_phi_precision',
    'nir_opt_phi_to_bool', 'nir_opt_preamble',
    'nir_opt_preamble_options', 'nir_opt_ray_queries',
    'nir_opt_ray_query_ranges', 'nir_opt_reassociate',
    'nir_opt_reassociate_bfi', 'nir_opt_reassociate_loop',
    'nir_opt_reassociate_matrix_mul',
    'nir_opt_rematerialize_compares', 'nir_opt_remove_phis',
    'nir_opt_shrink_stores', 'nir_opt_shrink_vectors',
    'nir_opt_simplify_convert_alu_types', 'nir_opt_sink',
    'nir_opt_tex_srcs_options', 'nir_opt_undef',
    'nir_opt_uniform_atomics', 'nir_opt_uniform_subgroup',
    'nir_opt_varyings', 'nir_opt_varyings_bulk',
    'nir_opt_varyings_progress',
    'nir_opt_varyings_progress__enumvalues', 'nir_opt_vectorize',
    'nir_opt_vectorize_io', 'nir_opt_vectorize_io_vars',
    'nir_output_clipper_var_groups', 'nir_output_deps',
    'nir_pack_bits', 'nir_pad_vec4', 'nir_pad_vector',
    'nir_pad_vector_imm_int', 'nir_parallel_copy_entry',
    'nir_parallel_copy_instr', 'nir_parallel_copy_instr_create',
    'nir_parameter', 'nir_phi_get_src_from_block', 'nir_phi_instr',
    'nir_phi_instr_add_src', 'nir_phi_instr_create',
    'nir_phi_pass_cb', 'nir_phi_src', 'nir_pop_if', 'nir_pop_loop',
    'nir_preamble_class', 'nir_preamble_class__enumvalues',
    'nir_preamble_class_general', 'nir_preamble_class_image',
    'nir_preamble_class_sampler', 'nir_preamble_num_classes',
    'nir_print_deref', 'nir_print_function_body',
    'nir_print_input_to_output_deps', 'nir_print_instr',
    'nir_print_shader', 'nir_print_shader_annotated',
    'nir_print_use_dominators', 'nir_printf_fmt',
    'nir_printf_fmt_at_px', 'nir_process_debug_variable',
    'nir_progress', 'nir_progress_consumer', 'nir_progress_producer',
    'nir_propagate_invariant', 'nir_push_continue', 'nir_push_else',
    'nir_push_if', 'nir_push_loop', 'nir_ray_query_value',
    'nir_ray_query_value__enumvalues', 'nir_ray_query_value_flags',
    'nir_ray_query_value_intersection_barycentrics',
    'nir_ray_query_value_intersection_candidate_aabb_opaque',
    'nir_ray_query_value_intersection_front_face',
    'nir_ray_query_value_intersection_geometry_index',
    'nir_ray_query_value_intersection_instance_custom_index',
    'nir_ray_query_value_intersection_instance_id',
    'nir_ray_query_value_intersection_instance_sbt_index',
    'nir_ray_query_value_intersection_object_ray_direction',
    'nir_ray_query_value_intersection_object_ray_origin',
    'nir_ray_query_value_intersection_object_to_world',
    'nir_ray_query_value_intersection_primitive_index',
    'nir_ray_query_value_intersection_t',
    'nir_ray_query_value_intersection_triangle_vertex_positions',
    'nir_ray_query_value_intersection_type',
    'nir_ray_query_value_intersection_world_to_object',
    'nir_ray_query_value_tmin',
    'nir_ray_query_value_world_ray_direction',
    'nir_ray_query_value_world_ray_origin',
    'nir_reassociate_cse_heuristic', 'nir_reassociate_options',
    'nir_reassociate_options__enumvalues',
    'nir_reassociate_scalar_math', 'nir_recompute_io_bases',
    'nir_reg_get_decl', 'nir_rematerialize_deref_in_use_blocks',
    'nir_rematerialize_derefs_in_use_blocks_impl',
    'nir_remove_dead_derefs', 'nir_remove_dead_derefs_impl',
    'nir_remove_dead_variables', 'nir_remove_dead_variables_options',
    'nir_remove_entrypoints', 'nir_remove_non_entrypoints',
    'nir_remove_non_exported', 'nir_remove_single_src_phis_block',
    'nir_remove_sysval_output', 'nir_remove_tex_shadow',
    'nir_remove_unused_io_vars', 'nir_remove_unused_varyings',
    'nir_remove_varying', 'nir_repair_ssa', 'nir_repair_ssa_impl',
    'nir_replicate', 'nir_resize_vector', 'nir_resource_data_intel',
    'nir_resource_data_intel__enumvalues',
    'nir_resource_intel_bindless', 'nir_resource_intel_non_uniform',
    'nir_resource_intel_pushable', 'nir_resource_intel_sampler',
    'nir_resource_intel_sampler_embedded',
    'nir_rewrite_image_intrinsic', 'nir_rewrite_uses_to_load_reg',
    'nir_round_down_components', 'nir_round_up_components',
    'nir_rounding_mode', 'nir_rounding_mode__enumvalues',
    'nir_rounding_mode_rd', 'nir_rounding_mode_rtne',
    'nir_rounding_mode_rtz', 'nir_rounding_mode_ru',
    'nir_rounding_mode_undef', 'nir_scalar', 'nir_scalar_alu_op',
    'nir_scalar_as_bool', 'nir_scalar_as_const_value',
    'nir_scalar_as_float', 'nir_scalar_as_int', 'nir_scalar_as_uint',
    'nir_scalar_chase_alu_src', 'nir_scalar_chase_movs',
    'nir_scalar_equal', 'nir_scalar_intrinsic_op',
    'nir_scalar_is_alu', 'nir_scalar_is_const',
    'nir_scalar_is_intrinsic', 'nir_scalar_is_undef',
    'nir_scalar_resolved', 'nir_scale_fdiv',
    'nir_scoped_memory_barrier', 'nir_select_from_ssa_def_array',
    'nir_selection_control', 'nir_selection_control__enumvalues',
    'nir_selection_control_divergent_always_taken',
    'nir_selection_control_dont_flatten',
    'nir_selection_control_flatten', 'nir_selection_control_none',
    'nir_shader', 'nir_shader_add_variable', 'nir_shader_alu_pass',
    'nir_shader_as_str', 'nir_shader_as_str_annotated',
    'nir_shader_clear_pass_flags', 'nir_shader_clone',
    'nir_shader_compiler_options', 'nir_shader_create',
    'nir_shader_gather_debug_info', 'nir_shader_gather_info',
    'nir_shader_get_entrypoint', 'nir_shader_get_function_for_name',
    'nir_shader_get_preamble', 'nir_shader_index_vars',
    'nir_shader_instructions_pass', 'nir_shader_intrinsics_pass',
    'nir_shader_lower_instructions', 'nir_shader_phi_pass',
    'nir_shader_preserve_all_metadata', 'nir_shader_replace',
    'nir_shader_serialize_deserialize',
    'nir_shader_supports_implicit_lod', 'nir_shader_tex_pass',
    'nir_shader_uses_view_index', 'nir_shift_channels',
    'nir_should_vectorize_mem_func', 'nir_shrink_vec_array_vars',
    'nir_skip_helpers_instrinsic_cb', 'nir_slot_is_sysval_output',
    'nir_slot_is_sysval_output_and_varying', 'nir_slot_is_varying',
    'nir_sort_unstructured_blocks', 'nir_sort_variables_by_location',
    'nir_sort_variables_with_modes', 'nir_split_64bit_vec3_and_vec4',
    'nir_split_array_vars', 'nir_split_conversions',
    'nir_split_conversions_options', 'nir_split_per_member_structs',
    'nir_split_struct_vars', 'nir_split_var_copies', 'nir_src',
    'nir_src_as_alu_instr', 'nir_src_as_bool',
    'nir_src_as_const_value', 'nir_src_as_deref', 'nir_src_as_float',
    'nir_src_as_int', 'nir_src_as_intrinsic', 'nir_src_as_string',
    'nir_src_as_uint', 'nir_src_bit_size', 'nir_src_comp_as_bool',
    'nir_src_comp_as_float', 'nir_src_comp_as_int',
    'nir_src_comp_as_uint', 'nir_src_components_read',
    'nir_src_for_ssa', 'nir_src_get_block', 'nir_src_init',
    'nir_src_is_always_uniform', 'nir_src_is_const',
    'nir_src_is_divergent', 'nir_src_is_if', 'nir_src_is_undef',
    'nir_src_num_components', 'nir_src_parent_if',
    'nir_src_parent_instr', 'nir_src_rewrite',
    'nir_src_set_parent_if', 'nir_src_set_parent_instr',
    'nir_srcs_equal', 'nir_ssa_alu_instr_src_components',
    'nir_ssa_for_alu_src', 'nir_start_block', 'nir_state_slot',
    'nir_state_variable_create', 'nir_static_workgroup_size',
    'nir_steal_tex_deref', 'nir_steal_tex_src', 'nir_store_array_var',
    'nir_store_array_var_imm', 'nir_store_deref',
    'nir_store_deref_with_access', 'nir_store_global',
    'nir_store_reg', 'nir_store_reg_for_def', 'nir_store_var',
    'nir_sweep', 'nir_swizzle', 'nir_system_value_from_intrinsic',
    'nir_test_mask', 'nir_tex_instr', 'nir_tex_instr_add_src',
    'nir_tex_instr_create', 'nir_tex_instr_dest_size',
    'nir_tex_instr_has_explicit_tg4_offsets',
    'nir_tex_instr_has_implicit_derivative', 'nir_tex_instr_is_query',
    'nir_tex_instr_need_sampler', 'nir_tex_instr_remove_src',
    'nir_tex_instr_result_size', 'nir_tex_instr_src_index',
    'nir_tex_instr_src_size', 'nir_tex_instr_src_type',
    'nir_tex_pass_cb', 'nir_tex_src', 'nir_tex_src_backend1',
    'nir_tex_src_backend2', 'nir_tex_src_bias',
    'nir_tex_src_comparator', 'nir_tex_src_coord', 'nir_tex_src_ddx',
    'nir_tex_src_ddy', 'nir_tex_src_for_ssa', 'nir_tex_src_lod',
    'nir_tex_src_lod_bias_min_agx', 'nir_tex_src_min_lod',
    'nir_tex_src_ms_index', 'nir_tex_src_ms_mcs_intel',
    'nir_tex_src_offset', 'nir_tex_src_plane',
    'nir_tex_src_projector', 'nir_tex_src_sampler_deref',
    'nir_tex_src_sampler_deref_intrinsic',
    'nir_tex_src_sampler_handle', 'nir_tex_src_sampler_offset',
    'nir_tex_src_texture_deref',
    'nir_tex_src_texture_deref_intrinsic',
    'nir_tex_src_texture_handle', 'nir_tex_src_texture_offset',
    'nir_tex_src_type', 'nir_tex_src_type_constraint',
    'nir_tex_src_type_constraints', 'nir_texop',
    'nir_texop_custom_border_color_agx', 'nir_texop_descriptor_amd',
    'nir_texop_fragment_fetch_amd',
    'nir_texop_fragment_mask_fetch_amd',
    'nir_texop_has_custom_border_color_agx', 'nir_texop_hdr_dim_nv',
    'nir_texop_image_min_lod_agx', 'nir_texop_lod',
    'nir_texop_lod_bias', 'nir_texop_query_levels',
    'nir_texop_sample_pos_nv', 'nir_texop_sampler_descriptor_amd',
    'nir_texop_samples_identical', 'nir_texop_tex',
    'nir_texop_tex_prefetch', 'nir_texop_tex_type_nv',
    'nir_texop_texture_samples', 'nir_texop_tg4', 'nir_texop_txb',
    'nir_texop_txd', 'nir_texop_txf', 'nir_texop_txf_ms',
    'nir_texop_txf_ms_fb', 'nir_texop_txf_ms_mcs_intel',
    'nir_texop_txl', 'nir_texop_txs', 'nir_trim_vector',
    'nir_trivialize_registers', 'nir_type_bool', 'nir_type_bool1',
    'nir_type_bool16', 'nir_type_bool32', 'nir_type_bool8',
    'nir_type_conversion_op', 'nir_type_convert', 'nir_type_float',
    'nir_type_float16', 'nir_type_float32', 'nir_type_float64',
    'nir_type_int', 'nir_type_int1', 'nir_type_int16',
    'nir_type_int32', 'nir_type_int64', 'nir_type_int8',
    'nir_type_invalid', 'nir_type_uint', 'nir_type_uint1',
    'nir_type_uint16', 'nir_type_uint32', 'nir_type_uint64',
    'nir_type_uint8', 'nir_u2fN', 'nir_u2uN', 'nir_ubfe_imm',
    'nir_ubitfield_extract_imm', 'nir_uclamp', 'nir_udiv_imm',
    'nir_umax_imm', 'nir_umin_imm', 'nir_umod_imm', 'nir_undef',
    'nir_undef_instr', 'nir_undef_instr_create', 'nir_unpack_bits',
    'nir_unsigned_upper_bound', 'nir_unsigned_upper_bound_config',
    'nir_unstructured_start_block', 'nir_use_dominance_lca',
    'nir_use_dominance_state', 'nir_ushr_imm', 'nir_validate_shader',
    'nir_validate_ssa_dominance', 'nir_var_all',
    'nir_var_declaration_type',
    'nir_var_declaration_type__enumvalues',
    'nir_var_declared_implicitly', 'nir_var_declared_normally',
    'nir_var_function_in', 'nir_var_function_inout',
    'nir_var_function_out', 'nir_var_function_temp', 'nir_var_hidden',
    'nir_var_image', 'nir_var_mem_constant', 'nir_var_mem_generic',
    'nir_var_mem_global', 'nir_var_mem_node_payload',
    'nir_var_mem_node_payload_in', 'nir_var_mem_push_const',
    'nir_var_mem_shared', 'nir_var_mem_ssbo',
    'nir_var_mem_task_payload', 'nir_var_mem_ubo',
    'nir_var_ray_hit_attrib', 'nir_var_read_only_modes',
    'nir_var_shader_call_data', 'nir_var_shader_in',
    'nir_var_shader_out', 'nir_var_shader_temp',
    'nir_var_system_value', 'nir_var_uniform',
    'nir_var_vec_indexable_modes', 'nir_variable',
    'nir_variable_append_namef', 'nir_variable_clone',
    'nir_variable_count_slots', 'nir_variable_create',
    'nir_variable_create_zeroed', 'nir_variable_data',
    'nir_variable_is_global', 'nir_variable_is_in_block',
    'nir_variable_is_in_ssbo', 'nir_variable_is_in_ubo',
    'nir_variable_mode', 'nir_variable_mode__enumvalues',
    'nir_variable_set_name', 'nir_variable_set_namef',
    'nir_variable_steal_name', 'nir_varying_var_mask', 'nir_vec',
    'nir_vec_scalars', 'nir_vector_extract', 'nir_vector_insert',
    'nir_vector_insert_imm', 'nir_vectorize_cb',
    'nir_vertex_divergence_analysis', 'nir_verts_in_output_prim',
    'nir_zero_initialize_shared_memory', 'pipe_format',
    'should_print_nir', 'should_skip_nir', 'size_t',
    'struct__IO_FILE', 'struct__IO_codecvt', 'struct__IO_marker',
    'struct__IO_wide_data', 'struct_c__SA_nir_input_to_output_deps',
    'struct_c__SA_nir_input_to_output_deps_0',
    'struct_c__SA_nir_output_clipper_var_groups',
    'struct_c__SA_nir_output_deps', 'struct_c__SA_nir_output_deps_0',
    'struct_exec_list', 'struct_exec_node', 'struct_gc_ctx',
    'struct_glsl_cmat_description', 'struct_glsl_struct_field',
    'struct_glsl_struct_field_0_0', 'struct_glsl_type',
    'struct_hash_entry', 'struct_hash_table', 'struct_list_head',
    'struct_nir_alu_instr', 'struct_nir_alu_src',
    'struct_nir_binding', 'struct_nir_block', 'struct_nir_builder',
    'struct_nir_call_instr', 'struct_nir_cf_node',
    'struct_nir_constant', 'struct_nir_cursor', 'struct_nir_def',
    'struct_nir_deref_instr', 'struct_nir_deref_instr_1_arr',
    'struct_nir_deref_instr_1_cast', 'struct_nir_deref_instr_1_strct',
    'struct_nir_function', 'struct_nir_function_impl',
    'struct_nir_if', 'struct_nir_input_attachment_options',
    'struct_nir_instr', 'struct_nir_instr_debug_info',
    'struct_nir_intrinsic_info', 'struct_nir_intrinsic_instr',
    'struct_nir_io_semantics', 'struct_nir_io_xfb',
    'struct_nir_io_xfb_0', 'struct_nir_jump_instr',
    'struct_nir_load_const_instr',
    'struct_nir_load_store_vectorize_options', 'struct_nir_loop',
    'struct_nir_loop_induction_variable', 'struct_nir_loop_info',
    'struct_nir_loop_terminator', 'struct_nir_lower_bitmap_options',
    'struct_nir_lower_compute_system_values_options',
    'struct_nir_lower_drawpixels_options',
    'struct_nir_lower_idiv_options', 'struct_nir_lower_image_options',
    'struct_nir_lower_mem_access_bit_sizes_options',
    'struct_nir_lower_multiview_options',
    'struct_nir_lower_non_uniform_access_options',
    'struct_nir_lower_printf_options',
    'struct_nir_lower_shader_calls_options',
    'struct_nir_lower_ssbo_options',
    'struct_nir_lower_subgroups_options',
    'struct_nir_lower_sysvals_to_varyings_options',
    'struct_nir_lower_task_shader_options',
    'struct_nir_lower_tex_options',
    'struct_nir_lower_tex_shadow_swizzle',
    'struct_nir_lower_wpos_ytransform_options',
    'struct_nir_mem_access_size_align', 'struct_nir_op_info',
    'struct_nir_opt_16bit_tex_image_options',
    'struct_nir_opt_access_options',
    'struct_nir_opt_load_skip_helpers_options',
    'struct_nir_opt_offsets_options',
    'struct_nir_opt_peephole_select_options',
    'struct_nir_opt_preamble_options',
    'struct_nir_opt_tex_srcs_options',
    'struct_nir_parallel_copy_entry',
    'struct_nir_parallel_copy_instr', 'struct_nir_parameter',
    'struct_nir_phi_instr', 'struct_nir_phi_src',
    'struct_nir_remove_dead_variables_options', 'struct_nir_scalar',
    'struct_nir_shader', 'struct_nir_shader_compiler_options',
    'struct_nir_split_conversions_options', 'struct_nir_src',
    'struct_nir_state_slot', 'struct_nir_tex_builder',
    'struct_nir_tex_instr', 'struct_nir_tex_src',
    'struct_nir_tex_src_type_constraint', 'struct_nir_undef_instr',
    'struct_nir_unsigned_upper_bound_config',
    'struct_nir_use_dominance_state', 'struct_nir_variable',
    'struct_nir_variable_data', 'struct_nir_variable_data_0_image',
    'struct_nir_variable_data_0_sampler',
    'struct_nir_variable_data_0_xfb', 'struct_nir_xfb_info',
    'struct_set', 'struct_set_entry', 'struct_shader_info',
    'struct_shader_info_0_cs', 'struct_shader_info_0_fs',
    'struct_shader_info_0_gs', 'struct_shader_info_0_mesh',
    'struct_shader_info_0_tess', 'struct_shader_info_0_vs',
    'struct_u_printf_info', 'tess_primitive_mode', 'u_printf_info',
    'uint32_t', 'uint64_t', 'uint8_t', 'union_c__UA_nir_const_value',
    'union_glsl_struct_field_0', 'union_glsl_type_fields',
    'union_nir_cursor_0', 'union_nir_deref_instr_0',
    'union_nir_deref_instr_1', 'union_nir_parallel_copy_entry_dest',
    'union_nir_variable_data_0', 'union_shader_info_0']
