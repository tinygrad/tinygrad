# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/opt/rocm/include', '-x', 'c++']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


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





SDMA_OP_COPY = 1 # Variable ctypes.c_uint32
SDMA_OP_FENCE = 5 # Variable ctypes.c_uint32
SDMA_OP_TRAP = 6 # Variable ctypes.c_uint32
SDMA_OP_POLL_REGMEM = 8 # Variable ctypes.c_uint32
SDMA_OP_ATOMIC = 10 # Variable ctypes.c_uint32
SDMA_OP_CONST_FILL = 11 # Variable ctypes.c_uint32
SDMA_OP_TIMESTAMP = 13 # Variable ctypes.c_uint32
SDMA_OP_GCR = 17 # Variable ctypes.c_uint32
SDMA_SUBOP_COPY_LINEAR = 0 # Variable ctypes.c_uint32
SDMA_SUBOP_COPY_LINEAR_RECT = 4 # Variable ctypes.c_uint32
SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2 # Variable ctypes.c_uint32
SDMA_SUBOP_USER_GCR = 1 # Variable ctypes.c_uint32
SDMA_ATOMIC_ADD64 = 47 # Variable ctypes.c_uint32
class struct_SDMA_PKT_COPY_LINEAR_TAG(Structure):
    pass

class union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('extra_info', ctypes.c_uint32, 16),
]

union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_1_0._fields_ = [
    ('count', ctypes.c_uint32, 22),
    ('reserved_0', ctypes.c_uint32, 10),
]

union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_2_0._fields_ = [
    ('reserved_0', ctypes.c_uint32, 16),
    ('dst_swap', ctypes.c_uint32, 2),
    ('reserved_1', ctypes.c_uint32, 6),
    ('src_swap', ctypes.c_uint32, 2),
    ('reserved_2', ctypes.c_uint32, 6),
]

union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_3_0._fields_ = [
    ('src_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_4_0._fields_ = [
    ('src_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_5_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_6_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_COPY_LINEAR_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION),
    ('COUNT_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION),
    ('PARAMETER_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION),
    ('SRC_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION),
    ('SRC_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION),
]

SDMA_PKT_COPY_LINEAR = struct_SDMA_PKT_COPY_LINEAR_TAG
class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG(Structure):
    pass

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved', ctypes.c_uint32, 13),
    ('element', ctypes.c_uint32, 3),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0._fields_ = [
    ('src_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0._fields_ = [
    ('src_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0._fields_ = [
    ('src_offset_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('src_offset_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0._fields_ = [
    ('src_offset_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 2),
    ('src_pitch', ctypes.c_uint32, 19),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0._fields_ = [
    ('src_slice_pitch', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0),
    ('DW_7_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0._fields_ = [
    ('dst_offset_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('dst_offset_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0),
    ('DW_8_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0._fields_ = [
    ('dst_offset_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 2),
    ('dst_pitch', ctypes.c_uint32, 19),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0),
    ('DW_9_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0._fields_ = [
    ('dst_slice_pitch', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0),
    ('DW_10_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0._fields_ = [
    ('rect_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('rect_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0),
    ('DW_11_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0._fields_ = [
    ('rect_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 5),
    ('dst_swap', ctypes.c_uint32, 2),
    ('reserved_2', ctypes.c_uint32, 6),
    ('src_swap', ctypes.c_uint32, 2),
    ('reserved_3', ctypes.c_uint32, 6),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0),
    ('DW_12_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION),
    ('SRC_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION),
    ('SRC_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION),
    ('SRC_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION),
    ('SRC_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION),
    ('SRC_PARAMETER_3_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION),
    ('DST_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION),
    ('DST_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION),
    ('DST_PARAMETER_3_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION),
    ('RECT_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION),
    ('RECT_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION),
]

SDMA_PKT_COPY_LINEAR_RECT = struct_SDMA_PKT_COPY_LINEAR_RECT_TAG
class struct_SDMA_PKT_CONSTANT_FILL_TAG(Structure):
    pass

class union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('sw', ctypes.c_uint32, 2),
    ('reserved_0', ctypes.c_uint32, 12),
    ('fillsize', ctypes.c_uint32, 2),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0._fields_ = [
    ('src_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0._fields_ = [
    ('count', ctypes.c_uint32, 22),
    ('reserved_0', ctypes.c_uint32, 10),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_CONSTANT_FILL_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION),
    ('DATA_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION),
    ('COUNT_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION),
]

SDMA_PKT_CONSTANT_FILL = struct_SDMA_PKT_CONSTANT_FILL_TAG
class struct_SDMA_PKT_FENCE_TAG(Structure):
    pass

class union_SDMA_PKT_FENCE_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('mtype', ctypes.c_uint32, 3),
    ('gcc', ctypes.c_uint32, 1),
    ('sys', ctypes.c_uint32, 1),
    ('pad1', ctypes.c_uint32, 1),
    ('snp', ctypes.c_uint32, 1),
    ('gpa', ctypes.c_uint32, 1),
    ('l2_policy', ctypes.c_uint32, 2),
    ('reserved_0', ctypes.c_uint32, 6),
]

union_SDMA_PKT_FENCE_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_DATA_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_3_0._fields_ = [
    ('data', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_DATA_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_DATA_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_DATA_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_FENCE_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_FENCE_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION),
    ('DATA_UNION', union_SDMA_PKT_FENCE_TAG_DATA_UNION),
]

SDMA_PKT_FENCE = struct_SDMA_PKT_FENCE_TAG
class struct_SDMA_PKT_POLL_REGMEM_TAG(Structure):
    pass

class union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 10),
    ('hdp_flush', ctypes.c_uint32, 1),
    ('reserved_1', ctypes.c_uint32, 1),
    ('func', ctypes.c_uint32, 3),
    ('mem_poll', ctypes.c_uint32, 1),
]

union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_3_0._fields_ = [
    ('value', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_4_0._fields_ = [
    ('mask', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_5_0._fields_ = [
    ('interval', ctypes.c_uint32, 16),
    ('retry_count', ctypes.c_uint32, 12),
    ('reserved_0', ctypes.c_uint32, 4),
]

union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_POLL_REGMEM_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION),
    ('VALUE_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION),
    ('MASK_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION),
    ('DW5_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION),
]

SDMA_PKT_POLL_REGMEM = struct_SDMA_PKT_POLL_REGMEM_TAG
class struct_SDMA_PKT_ATOMIC_TAG(Structure):
    pass

class union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('l', ctypes.c_uint32, 1),
    ('reserved_0', ctypes.c_uint32, 8),
    ('operation', ctypes.c_uint32, 7),
]

union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_3_0._fields_ = [
    ('src_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_4_0._fields_ = [
    ('src_data_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_5_0._fields_ = [
    ('cmp_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_6_0._fields_ = [
    ('cmp_data_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_7_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_7_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_7_0._fields_ = [
    ('loop_interval', ctypes.c_uint32, 13),
    ('reserved_0', ctypes.c_uint32, 19),
]

union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_7_0),
    ('DW_7_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_ATOMIC_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION),
    ('SRC_DATA_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION),
    ('SRC_DATA_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION),
    ('CMP_DATA_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION),
    ('CMP_DATA_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION),
    ('LOOP_UNION', union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION),
]

SDMA_PKT_ATOMIC = struct_SDMA_PKT_ATOMIC_TAG
class struct_SDMA_PKT_TIMESTAMP_TAG(Structure):
    pass

class union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 16),
]

union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_TIMESTAMP_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION),
]

SDMA_PKT_TIMESTAMP = struct_SDMA_PKT_TIMESTAMP_TAG
class struct_SDMA_PKT_TRAP_TAG(Structure):
    pass

class union_SDMA_PKT_TRAP_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_TRAP_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_TRAP_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 16),
]

union_SDMA_PKT_TRAP_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TRAP_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TRAP_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TRAP_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION(Union):
    pass

class struct_SDMA_PKT_TRAP_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_TRAP_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG_1_0._fields_ = [
    ('int_ctx', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TRAP_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_TRAP_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_TRAP_TAG_HEADER_UNION),
    ('INT_CONTEXT_UNION', union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION),
]

SDMA_PKT_TRAP = struct_SDMA_PKT_TRAP_TAG
class struct_SDMA_PKT_HDP_FLUSH_TAG(Structure):
    pass

struct_SDMA_PKT_HDP_FLUSH_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_HDP_FLUSH_TAG._fields_ = [
    ('DW_0_DATA', ctypes.c_uint32),
    ('DW_1_DATA', ctypes.c_uint32),
    ('DW_2_DATA', ctypes.c_uint32),
    ('DW_3_DATA', ctypes.c_uint32),
    ('DW_4_DATA', ctypes.c_uint32),
    ('DW_5_DATA', ctypes.c_uint32),
]

SDMA_PKT_HDP_FLUSH = struct_SDMA_PKT_HDP_FLUSH_TAG
hdp_flush_cmd = struct_SDMA_PKT_HDP_FLUSH_TAG # Variable struct_SDMA_PKT_HDP_FLUSH_TAG
class struct_SDMA_PKT_GCR_TAG(Structure):
    pass

class union_SDMA_PKT_GCR_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('_2', ctypes.c_uint32, 16),
]

union_SDMA_PKT_GCR_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD1_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_1_0._fields_ = [
    ('_0', ctypes.c_uint32, 7),
    ('BaseVA_LO', ctypes.c_uint32, 25),
]

union_SDMA_PKT_GCR_TAG_WORD1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD2_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_2_0._fields_ = [
    ('BaseVA_HI', ctypes.c_uint32, 16),
    ('GCR_CONTROL_GLI_INV', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GL1_RANGE', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GLM_WB', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLM_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLK_WB', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLK_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLV_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL1_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_US', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_RANGE', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GL2_DISCARD', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_WB', ctypes.c_uint32, 1),
]

union_SDMA_PKT_GCR_TAG_WORD2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD3_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_3_0._fields_ = [
    ('GCR_CONTROL_RANGE_IS_PA', ctypes.c_uint32, 1),
    ('GCR_CONTROL_SEQ', ctypes.c_uint32, 2),
    ('_2', ctypes.c_uint32, 4),
    ('LimitVA_LO', ctypes.c_uint32, 25),
]

union_SDMA_PKT_GCR_TAG_WORD3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD4_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_4_0._fields_ = [
    ('LimitVA_HI', ctypes.c_uint32, 16),
    ('_1', ctypes.c_uint32, 8),
    ('VMID', ctypes.c_uint32, 4),
    ('_3', ctypes.c_uint32, 4),
]

union_SDMA_PKT_GCR_TAG_WORD4_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD4_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD4_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_GCR_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_GCR_TAG_HEADER_UNION),
    ('WORD1_UNION', union_SDMA_PKT_GCR_TAG_WORD1_UNION),
    ('WORD2_UNION', union_SDMA_PKT_GCR_TAG_WORD2_UNION),
    ('WORD3_UNION', union_SDMA_PKT_GCR_TAG_WORD3_UNION),
    ('WORD4_UNION', union_SDMA_PKT_GCR_TAG_WORD4_UNION),
]

SDMA_PKT_GCR = struct_SDMA_PKT_GCR_TAG
__all__ = \
    ['SDMA_ATOMIC_ADD64', 'SDMA_OP_ATOMIC', 'SDMA_OP_CONST_FILL',
    'SDMA_OP_COPY', 'SDMA_OP_FENCE', 'SDMA_OP_GCR',
    'SDMA_OP_POLL_REGMEM', 'SDMA_OP_TIMESTAMP', 'SDMA_OP_TRAP',
    'SDMA_PKT_ATOMIC', 'SDMA_PKT_CONSTANT_FILL',
    'SDMA_PKT_COPY_LINEAR', 'SDMA_PKT_COPY_LINEAR_RECT',
    'SDMA_PKT_FENCE', 'SDMA_PKT_GCR', 'SDMA_PKT_HDP_FLUSH',
    'SDMA_PKT_POLL_REGMEM', 'SDMA_PKT_TIMESTAMP', 'SDMA_PKT_TRAP',
    'SDMA_SUBOP_COPY_LINEAR', 'SDMA_SUBOP_COPY_LINEAR_RECT',
    'SDMA_SUBOP_TIMESTAMP_GET_GLOBAL', 'SDMA_SUBOP_USER_GCR',
    'hdp_flush_cmd', 'struct_SDMA_PKT_ATOMIC_TAG',
    'struct_SDMA_PKT_ATOMIC_TAG_0_0',
    'struct_SDMA_PKT_ATOMIC_TAG_1_0',
    'struct_SDMA_PKT_ATOMIC_TAG_2_0',
    'struct_SDMA_PKT_ATOMIC_TAG_3_0',
    'struct_SDMA_PKT_ATOMIC_TAG_4_0',
    'struct_SDMA_PKT_ATOMIC_TAG_5_0',
    'struct_SDMA_PKT_ATOMIC_TAG_6_0',
    'struct_SDMA_PKT_ATOMIC_TAG_7_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_0_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_1_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_2_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_3_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_5_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_6_0',
    'struct_SDMA_PKT_FENCE_TAG', 'struct_SDMA_PKT_FENCE_TAG_0_0',
    'struct_SDMA_PKT_FENCE_TAG_1_0', 'struct_SDMA_PKT_FENCE_TAG_2_0',
    'struct_SDMA_PKT_FENCE_TAG_3_0', 'struct_SDMA_PKT_GCR_TAG',
    'struct_SDMA_PKT_GCR_TAG_0_0', 'struct_SDMA_PKT_GCR_TAG_1_0',
    'struct_SDMA_PKT_GCR_TAG_2_0', 'struct_SDMA_PKT_GCR_TAG_3_0',
    'struct_SDMA_PKT_GCR_TAG_4_0', 'struct_SDMA_PKT_HDP_FLUSH_TAG',
    'struct_SDMA_PKT_POLL_REGMEM_TAG',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_0_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_1_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_2_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_3_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_4_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_5_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG',
    'struct_SDMA_PKT_TIMESTAMP_TAG_0_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG_1_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG_2_0', 'struct_SDMA_PKT_TRAP_TAG',
    'struct_SDMA_PKT_TRAP_TAG_0_0', 'struct_SDMA_PKT_TRAP_TAG_1_0',
    'union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION',
    'union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_FENCE_TAG_DATA_UNION',
    'union_SDMA_PKT_FENCE_TAG_HEADER_UNION',
    'union_SDMA_PKT_GCR_TAG_HEADER_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD1_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD2_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD3_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD4_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION',
    'union_SDMA_PKT_TRAP_TAG_HEADER_UNION',
    'union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION']
 #/*
# * Copyright 2019 Advanced Micro Devices, Inc.
# *
# * Permission is hereby granted, free of charge, to any person obtaining a
# * copy of this software and associated documentation files (the "Software"),
# * to deal in the Software without restriction, including without limitation
# * the rights to use, copy, modify, merge, publish, distribute, sublicense,
# * and/or sell copies of the Software, and to permit persons to whom the
# * Software is furnished to do so, subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be included in
# * all copies or substantial portions of the Software.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
# * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# * OTHER DEALINGS IN THE SOFTWARE.
# *
# */

#ifndef NVD_H
#define NVD_H

 #/**
# * Navi's PM4 definitions
# */
PACKET_TYPE0 = 0
PACKET_TYPE1 = 1
PACKET_TYPE2 = 2
PACKET_TYPE3 = 3

def CP_PACKET_GET_TYPE(h): return (((h) >> 30) & 3)
def CP_PACKET_GET_COUNT(h): return (((h) >> 16) & 0x3FFF)
def CP_PACKET0_GET_REG(h): return ((h) & 0xFFFF)
def CP_PACKET3_GET_OPCODE(h): return (((h) >> 8) & 0xFF)
def PACKET0(reg, n): return ((PACKET_TYPE0 << 30) |				\
			 ((reg) & 0xFFFF) |			\
			 ((n) & 0x3FFF) << 16)
CP_PACKET2 = 0x80000000
PACKET2_PAD_SHIFT = 0
PACKET2_PAD_MASK = (0x3fffffff << 0)

def PACKET2(v): return (CP_PACKET2 | REG_SET(PACKET2_PAD, (v)))

def PACKET3(op, n): return ((PACKET_TYPE3 << 30) |				\
			 (((op) & 0xFF) << 8) |				\
			 ((n) & 0x3FFF) << 16)

def PACKET3_COMPUTE(op, n): return (PACKET3(op, n) | 1 << 1)

 #/* Packet 3 types */
PACKET3_NOP = 0x10
PACKET3_SET_BASE = 0x11
def PACKET3_BASE_INDEX(x): return ((x) << 0)
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
def WRITE_DATA_DST_SEL(x): return ((x) << 8)
		 #/* 0 - register
#		 * 1 - memory (sync - via GRBM)
#		 * 2 - gl2
#		 * 3 - gds
#		 * 4 - reserved
#		 * 5 - memory (async - direct)
#		 */
WR_ONE_ADDR = (1 << 16)
WR_CONFIRM = (1 << 20)
def WRITE_DATA_CACHE_POLICY(x): return ((x) << 25)
		 #/* 0 - LRU
#		 * 1 - Stream
#		 */
def WRITE_DATA_ENGINE_SEL(x): return ((x) << 30)
		 #/* 0 - me
#		 * 1 - pfp
#		 * 2 - ce
#		 */
PACKET3_DRAW_INDEX_INDIRECT_MULTI = 0x38
PACKET3_MEM_SEMAPHORE = 0x39
PACKET3_SEM_USE_MAILBOX = (0x1 << 16)
PACKET3_SEM_SEL_SIGNAL_TYPE = (0x1 << 20)  #/* 0 = increment, 1 = write 1 */
PACKET3_SEM_SEL_SIGNAL = (0x6 << 29)
PACKET3_SEM_SEL_WAIT = (0x7 << 29)
PACKET3_DRAW_INDEX_MULTI_INST = 0x3A
PACKET3_COPY_DW = 0x3B
PACKET3_WAIT_REG_MEM = 0x3C
def WAIT_REG_MEM_FUNCTION(x): return ((x) << 0)
		 #/* 0 - always
#		 * 1 - <
#		 * 2 - <=
#		 * 3 - ==
#		 * 4 - !=
#		 * 5 - >=
#		 * 6 - >
#		 */
def WAIT_REG_MEM_MEM_SPACE(x): return ((x) << 4)
		 #/* 0 - reg
#		 * 1 - mem
#		 */
def WAIT_REG_MEM_OPERATION(x): return ((x) << 6)
		 #/* 0 - wait_reg_mem
#		 * 1 - wr_wait_wr_reg
#		 */
def WAIT_REG_MEM_ENGINE(x): return ((x) << 8)
		 #/* 0 - me
#		 * 1 - pfp
#		 */
PACKET3_INDIRECT_BUFFER = 0x3F
INDIRECT_BUFFER_VALID = (1 << 23)
def INDIRECT_BUFFER_CACHE_POLICY(x): return ((x) << 28)
		 #/* 0 - LRU
#		 * 1 - Stream
#		 * 2 - Bypass
#		 */
def INDIRECT_BUFFER_PRE_ENB(x): return ((x) << 21)
def INDIRECT_BUFFER_PRE_RESUME(x): return ((x) << 30)
PACKET3_COND_INDIRECT_BUFFER = 0x3F
PACKET3_COPY_DATA = 0x40
PACKET3_CP_DMA = 0x41
PACKET3_PFP_SYNC_ME = 0x42
PACKET3_SURFACE_SYNC = 0x43
PACKET3_ME_INITIALIZE = 0x44
PACKET3_COND_WRITE = 0x45
PACKET3_EVENT_WRITE = 0x46
def EVENT_TYPE(x): return ((x) << 0)
def EVENT_INDEX(x): return ((x) << 8)
		 #/* 0 - any non-TS event
#		 * 1 - ZPASS_DONE, PIXEL_PIPE_STAT_*
#		 * 2 - SAMPLE_PIPELINESTAT
#		 * 3 - SAMPLE_STREAMOUTSTAT*
#		 * 4 - *S_PARTIAL_FLUSH
#		 */
PACKET3_EVENT_WRITE_EOP = 0x47
PACKET3_EVENT_WRITE_EOS = 0x48
PACKET3_RELEASE_MEM = 0x49
def PACKET3_RELEASE_MEM_EVENT_TYPE(x): return ((x) << 0)
def PACKET3_RELEASE_MEM_EVENT_INDEX(x): return ((x) << 8)
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
def PACKET3_RELEASE_MEM_CACHE_POLICY(x): return ((x) << 25)
		 #/* 0 - cache_policy__me_release_mem__lru
#		 * 1 - cache_policy__me_release_mem__stream
#		 * 2 - cache_policy__me_release_mem__noa
#		 * 3 - cache_policy__me_release_mem__bypass
#		 */
PACKET3_RELEASE_MEM_EXECUTE = (1 << 28)

def PACKET3_RELEASE_MEM_DATA_SEL(x): return ((x) << 29)
		 #/* 0 - discard
#		 * 1 - send low 32bit data
#		 * 2 - send 64bit data
#		 * 3 - send 64bit GPU counter value
#		 * 4 - send 64bit sys counter value
#		 */
def PACKET3_RELEASE_MEM_INT_SEL(x): return ((x) << 24)
		 #/* 0 - none
#		 * 1 - interrupt only (DATA_SEL = 0)
#		 * 2 - interrupt when data write is confirmed
#		 */
def PACKET3_RELEASE_MEM_DST_SEL(x): return ((x) << 16)
		 #/* 0 - MC
#		 * 1 - TC/L2
#		 */



PACKET3_PREAMBLE_CNTL = 0x4A
PACKET3_PREAMBLE_BEGIN_CLEAR_STATE = (2 << 28)
PACKET3_PREAMBLE_END_CLEAR_STATE = (3 << 28)
PACKET3_DMA_DATA = 0x50
 #/* 1. header
# * 2. CONTROL
# * 3. SRC_ADDR_LO or DATA [31:0]
# * 4. SRC_ADDR_HI [31:0]
# * 5. DST_ADDR_LO [31:0]
# * 6. DST_ADDR_HI [7:0]
# * 7. COMMAND [31:26] | BYTE_COUNT [25:0]
# */
 #/* CONTROL */
def PACKET3_DMA_DATA_ENGINE(x): return ((x) << 0)
		 #/* 0 - ME
#		 * 1 - PFP
#		 */
def PACKET3_DMA_DATA_SRC_CACHE_POLICY(x): return ((x) << 13)
		 #/* 0 - LRU
#		 * 1 - Stream
#		 */
def PACKET3_DMA_DATA_DST_SEL(x): return ((x) << 20)
		 #/* 0 - DST_ADDR using DAS
#		 * 1 - GDS
#		 * 3 - DST_ADDR using L2
#		 */
def PACKET3_DMA_DATA_DST_CACHE_POLICY(x): return ((x) << 25)
		 #/* 0 - LRU
#		 * 1 - Stream
#		 */
def PACKET3_DMA_DATA_SRC_SEL(x): return ((x) << 29)
		 #/* 0 - SRC_ADDR using SAS
#		 * 1 - GDS
#		 * 2 - DATA
#		 * 3 - SRC_ADDR using L2
#		 */
PACKET3_DMA_DATA_CP_SYNC = (1 << 31)
 #/* COMMAND */
PACKET3_DMA_DATA_CMD_SAS = (1 << 26)
		 #/* 0 - memory
#		 * 1 - register
#		 */
PACKET3_DMA_DATA_CMD_DAS = (1 << 27)
		 #/* 0 - memory
#		 * 1 - register
#		 */
PACKET3_DMA_DATA_CMD_SAIC = (1 << 28)
PACKET3_DMA_DATA_CMD_DAIC = (1 << 29)
PACKET3_DMA_DATA_CMD_RAW_WAIT = (1 << 30)
PACKET3_CONTEXT_REG_RMW = 0x51
PACKET3_GFX_CNTX_UPDATE = 0x52
PACKET3_BLK_CNTX_UPDATE = 0x53
PACKET3_INCR_UPDT_STATE = 0x55
PACKET3_ACQUIRE_MEM = 0x58
 #/* 1.  HEADER
# * 2.  COHER_CNTL [30:0]
# * 2.1 ENGINE_SEL [31:31]
# * 2.  COHER_SIZE [31:0]
# * 3.  COHER_SIZE_HI [7:0]
# * 4.  COHER_BASE_LO [31:0]
# * 5.  COHER_BASE_HI [23:0]
# * 7.  POLL_INTERVAL [15:0]
# * 8.  GCR_CNTL [18:0]
# */
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(x): return ((x) << 0)
		 #/*
#		 * 0:NOP
#		 * 1:ALL
#		 * 2:RANGE
#		 * 3:FIRST_LAST
#		 */
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_RANGE(x): return ((x) << 2)
		 #/*
#		 * 0:ALL
#		 * 1:reserved
#		 * 2:RANGE
#		 * 3:FIRST_LAST
#		 */
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(x): return ((x) << 4)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(x): return ((x) << 5)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(x): return ((x) << 6)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(x): return ((x) << 7)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(x): return ((x) << 8)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(x): return ((x) << 9)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_US(x): return ((x) << 10)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_RANGE(x): return ((x) << 11)
		 #/*
#		 * 0:ALL
#		 * 1:VOL
#		 * 2:RANGE
#		 * 3:FIRST_LAST
#		 */
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_DISCARD(x): return ((x) << 13)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(x): return ((x) << 14)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(x): return ((x) << 15)
def PACKET3_ACQUIRE_MEM_GCR_CNTL_SEQ(x): return ((x) << 16)
		 #/*
#		 * 0: PARALLEL
#		 * 1: FORWARD
#		 * 2: REVERSE
#		 */
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
def FRAME_CMD(x): return ((x) << 28)
			 #/*
#			 * x=0: tmz_begin
#			 * x=1: tmz_end
#			 */
PACKET3_INDEX_ATTRIBUTES_INDIRECT = 0x91
PACKET3_WAIT_REG_MEM64 = 0x93
PACKET3_COND_PREEMPT = 0x94
PACKET3_HDP_FLUSH = 0x95
PACKET3_COPY_DATA_RB = 0x96
PACKET3_INVALIDATE_TLBS = 0x98
def PACKET3_INVALIDATE_TLBS_DST_SEL(x): return ((x) << 0)
def PACKET3_INVALIDATE_TLBS_ALL_HUB(x): return ((x) << 4)
def PACKET3_INVALIDATE_TLBS_PASID(x): return ((x) << 5)
PACKET3_AQL_PACKET = 0x99
PACKET3_DMA_DATA_FILL_MULTI = 0x9A
PACKET3_SET_SH_REG_INDEX = 0x9B
PACKET3_DRAW_INDIRECT_COUNT_MULTI = 0x9C
PACKET3_DRAW_INDEX_INDIRECT_COUNT_MULTI = 0x9D
PACKET3_DUMP_CONST_RAM_OFFSET = 0x9E
PACKET3_LOAD_CONTEXT_REG_INDEX = 0x9F
PACKET3_SET_RESOURCES = 0xA0
 #/* 1. header
# * 2. CONTROL
# * 3. QUEUE_MASK_LO [31:0]
# * 4. QUEUE_MASK_HI [31:0]
# * 5. GWS_MASK_LO [31:0]
# * 6. GWS_MASK_HI [31:0]
# * 7. OAC_MASK [15:0]
# * 8. GDS_HEAP_SIZE [16:11] | GDS_HEAP_BASE [5:0]
# */
def PACKET3_SET_RESOURCES_VMID_MASK(x): return ((x) << 0)
def PACKET3_SET_RESOURCES_UNMAP_LATENTY(x): return ((x) << 16)
def PACKET3_SET_RESOURCES_QUEUE_TYPE(x): return ((x) << 29)
PACKET3_MAP_PROCESS = 0xA1
PACKET3_MAP_QUEUES = 0xA2
 #/* 1. header
# * 2. CONTROL
# * 3. CONTROL2
# * 4. MQD_ADDR_LO [31:0]
# * 5. MQD_ADDR_HI [31:0]
# * 6. WPTR_ADDR_LO [31:0]
# * 7. WPTR_ADDR_HI [31:0]
# */
 #/* CONTROL */
def PACKET3_MAP_QUEUES_QUEUE_SEL(x): return ((x) << 4)
def PACKET3_MAP_QUEUES_VMID(x): return ((x) << 8)
def PACKET3_MAP_QUEUES_QUEUE(x): return ((x) << 13)
def PACKET3_MAP_QUEUES_PIPE(x): return ((x) << 16)
def PACKET3_MAP_QUEUES_ME(x): return ((x) << 18)
def PACKET3_MAP_QUEUES_QUEUE_TYPE(x): return ((x) << 21)
def PACKET3_MAP_QUEUES_ALLOC_FORMAT(x): return ((x) << 24)
def PACKET3_MAP_QUEUES_ENGINE_SEL(x): return ((x) << 26)
def PACKET3_MAP_QUEUES_NUM_QUEUES(x): return ((x) << 29)
 #/* CONTROL2 */
def PACKET3_MAP_QUEUES_CHECK_DISABLE(x): return ((x) << 1)
def PACKET3_MAP_QUEUES_DOORBELL_OFFSET(x): return ((x) << 2)
PACKET3_UNMAP_QUEUES = 0xA3
 #/* 1. header
# * 2. CONTROL
# * 3. CONTROL2
# * 4. CONTROL3
# * 5. CONTROL4
# * 6. CONTROL5
# */
 #/* CONTROL */
def PACKET3_UNMAP_QUEUES_ACTION(x): return ((x) << 0)
		 #/* 0 - PREEMPT_QUEUES
#		 * 1 - RESET_QUEUES
#		 * 2 - DISABLE_PROCESS_QUEUES
#		 * 3 - PREEMPT_QUEUES_NO_UNMAP
#		 */
def PACKET3_UNMAP_QUEUES_QUEUE_SEL(x): return ((x) << 4)
def PACKET3_UNMAP_QUEUES_ENGINE_SEL(x): return ((x) << 26)
def PACKET3_UNMAP_QUEUES_NUM_QUEUES(x): return ((x) << 29)
 #/* CONTROL2a */
def PACKET3_UNMAP_QUEUES_PASID(x): return ((x) << 0)
 #/* CONTROL2b */
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET0(x): return ((x) << 2)
 #/* CONTROL3a */
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET1(x): return ((x) << 2)
 #/* CONTROL3b */
def PACKET3_UNMAP_QUEUES_RB_WPTR(x): return ((x) << 0)
 #/* CONTROL4 */
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET2(x): return ((x) << 2)
 #/* CONTROL5 */
def PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET3(x): return ((x) << 2)
PACKET3_QUERY_STATUS = 0xA4
 #/* 1. header
# * 2. CONTROL
# * 3. CONTROL2
# * 4. ADDR_LO [31:0]
# * 5. ADDR_HI [31:0]
# * 6. DATA_LO [31:0]
# * 7. DATA_HI [31:0]
# */
 #/* CONTROL */
def PACKET3_QUERY_STATUS_CONTEXT_ID(x): return ((x) << 0)
def PACKET3_QUERY_STATUS_INTERRUPT_SEL(x): return ((x) << 28)
def PACKET3_QUERY_STATUS_COMMAND(x): return ((x) << 30)
 #/* CONTROL2a */
def PACKET3_QUERY_STATUS_PASID(x): return ((x) << 0)
 #/* CONTROL2b */
def PACKET3_QUERY_STATUS_DOORBELL_OFFSET(x): return ((x) << 2)
def PACKET3_QUERY_STATUS_ENG_SEL(x): return ((x) << 25)
PACKET3_RUN_LIST = 0xA5
PACKET3_MAP_PROCESS_VM = 0xA6
 #/* GFX11 */
PACKET3_SET_Q_PREEMPTION_MODE = 0xF0
def PACKET3_SET_Q_PREEMPTION_MODE_IB_VMID(x): return ((x) << 0)
PACKET3_SET_Q_PREEMPTION_MODE_INIT_SHADOW_MEM = (1 << 0)

#endif
