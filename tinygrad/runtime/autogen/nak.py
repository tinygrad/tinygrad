# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-DHAVE_ENDIAN_H', '-I/tmp/mesa-9e0991eff5aea2e064fc16d5c7fa0ee6cd52d894/src', '-I/tmp/mesa-9e0991eff5aea2e064fc16d5c7fa0ee6cd52d894/include', '-I/tmp/mesa-9e0991eff5aea2e064fc16d5c7fa0ee6cd52d894/src/compiler/nir']
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
except AttributeError:
    pass
class struct_nak_compiler(Structure):
    pass

try:
    nak_compiler_create = _libraries['FIXME_STUB'].nak_compiler_create
    nak_compiler_create.restype = ctypes.POINTER(struct_nak_compiler)
    nak_compiler_create.argtypes = [ctypes.POINTER(struct_nv_device_info)]
except AttributeError:
    pass
try:
    nak_compiler_destroy = _libraries['FIXME_STUB'].nak_compiler_destroy
    nak_compiler_destroy.restype = None
    nak_compiler_destroy.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
uint64_t = ctypes.c_uint64
try:
    nak_debug_flags = _libraries['FIXME_STUB'].nak_debug_flags
    nak_debug_flags.restype = uint64_t
    nak_debug_flags.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
class struct_nir_shader_compiler_options(Structure):
    pass

try:
    nak_nir_options = _libraries['FIXME_STUB'].nak_nir_options
    nak_nir_options.restype = ctypes.POINTER(struct_nir_shader_compiler_options)
    nak_nir_options.argtypes = [ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
class struct_nir_shader(Structure):
    pass

try:
    nak_preprocess_nir = _libraries['FIXME_STUB'].nak_preprocess_nir
    nak_preprocess_nir.restype = None
    nak_preprocess_nir.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
    pass
try:
    nak_nir_lower_image_addrs = _libraries['FIXME_STUB'].nak_nir_lower_image_addrs
    nak_nir_lower_image_addrs.restype = ctypes.c_bool
    nak_nir_lower_image_addrs.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nak_compiler)]
except AttributeError:
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
try:
    nak_postprocess_nir = _libraries['FIXME_STUB'].nak_postprocess_nir
    nak_postprocess_nir.restype = None
    nak_postprocess_nir.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except AttributeError:
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
    ('stage', mesa_shader_stage),
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
except AttributeError:
    pass
try:
    nak_compile_shader = _libraries['FIXME_STUB'].nak_compile_shader
    nak_compile_shader.restype = ctypes.POINTER(struct_nak_shader_bin)
    nak_compile_shader.argtypes = [ctypes.POINTER(struct_nir_shader), ctypes.c_bool, ctypes.POINTER(struct_nak_compiler), nir_variable_mode, ctypes.POINTER(struct_nak_fs_key)]
except AttributeError:
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
except AttributeError:
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
except AttributeError:
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
except AttributeError:
    pass
__all__ = \
    ['MESA_SHADER_ANY_HIT', 'MESA_SHADER_CALLABLE',
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
    'c__EA_nir_variable_mode', 'mesa_shader_stage',
    'nak_compile_shader', 'nak_compiler_create',
    'nak_compiler_destroy', 'nak_debug_flags', 'nak_fill_qmd',
    'nak_get_qmd_cbuf_desc_layout',
    'nak_get_qmd_dispatch_size_layout', 'nak_nir_lower_image_addrs',
    'nak_nir_options', 'nak_postprocess_nir', 'nak_preprocess_nir',
    'nak_shader_bin_destroy', 'nak_ts_domain', 'nak_ts_prims',
    'nak_ts_spacing', 'nir_num_variable_modes', 'nir_var_all',
    'nir_var_function_in', 'nir_var_function_inout',
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
    'nv_device_uuid', 'size_t', 'struct_nak_compiler',
    'struct_nak_fs_key', 'struct_nak_qmd_cbuf',
    'struct_nak_qmd_cbuf_desc_layout',
    'struct_nak_qmd_dispatch_size_layout', 'struct_nak_qmd_info',
    'struct_nak_sample_location', 'struct_nak_sample_mask',
    'struct_nak_shader_bin', 'struct_nak_shader_info',
    'struct_nak_shader_info_0_cs', 'struct_nak_shader_info_0_fs',
    'struct_nak_shader_info_0_ts', 'struct_nak_shader_info_vtg',
    'struct_nak_xfb_info', 'struct_nir_shader',
    'struct_nir_shader_compiler_options', 'struct_nv_device_info',
    'struct_nv_device_info_pci', 'uint64_t', 'uint8_t',
    'union_nak_shader_info_0']
